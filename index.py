import concurrent.futures
import math
import os
import traceback
import glob
import csv
import time
import random
import logging

import yfinance as yf
import pandas as pd
import requests
from dotenv import load_dotenv

from typing import List
from concurrent.futures import ThreadPoolExecutor
from typing import List, Any
from requests.exceptions import ChunkedEncodingError, RequestException
from urllib3.exceptions import ProtocolError

# Load environment variables
load_dotenv()

def setup_logging():
    """Configure logging with timestamp, level, and message."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)

logger = setup_logging()

def get_proxy_session():
    """
    Create a session with Webshare proxy configuration.
    """
    try:
        # Get credentials from environment variables
        username = os.getenv('WEBSHARE_USERNAME')
        password = os.getenv('WEBSHARE_PASSWORD')
        
        if not username or not password:
            raise ValueError("Webshare proxy credentials not found in environment variables")
        
        # Create session with rotating proxy
        session = requests.Session()
        proxy_url = f"http://{username}:{password}@p.webshare.io:80/"
        session.proxies = {
            "http": proxy_url,
            "https": proxy_url
        }
        
        # Test the proxy connection
        test_response = session.get("https://ipv4.webshare.io/")
        test_response.raise_for_status()
        logger.info(f"Connected via IP: {test_response.text.strip()}")
        
        return session
    except Exception as e:
        logger.error(f"Error setting up Webshare proxy: {e}")
        raise

def fetch_stock_data(symbol: str, use_proxy: bool, max_retries: int, initial_delay: int) -> None:
    """
    Fetch stock data for a given symbol and save it to a CSV file.
    
    Args:
        symbol: Stock symbol to fetch
        use_proxy: Whether to use proxy for requests
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay between retries
    """
    logger.info(f"Starting fetch for symbol: {symbol}")
    delay = initial_delay
    session = None
    
    try:
        if use_proxy:
            logger.info(f"Setting up Webshare proxy for {symbol}")
            session = get_proxy_session()
            logger.info(f"Webshare proxy ready for {symbol}")

        for attempt in range(max_retries):
            try:
                logger.info(f"Fetching {symbol} with{' ' if use_proxy else ' no '}proxy (Attempt {attempt + 1}/{max_retries})")
                
                # Create a custom handler to capture error logs
                class ErrorCaptureHandler(logging.Handler):
                    def __init__(self):
                        super().__init__()
                        self.error_messages = []
                    
                    def emit(self, record):
                        if record.levelno >= logging.ERROR:
                            self.error_messages.append(record.getMessage())

                # Add the custom handler
                error_handler = ErrorCaptureHandler()
                logging.getLogger('yfinance').addHandler(error_handler)
                
                PERIOD = os.getenv('PERIOD')
                INTERVAL = os.getenv('INTERVAL')
                GROUP_BY = os.getenv('GROUP_BY')

                if use_proxy:
                    df = yf.download(symbol, period=PERIOD, interval=INTERVAL, group_by=GROUP_BY, session=session)
                else:
                    df = yf.download(symbol, period=PERIOD, interval=INTERVAL, group_by=GROUP_BY)
                    
                # Check if YFPricesMissingError was logged
                if any("YFPricesMissingError" in msg or "no price data found" in msg 
                      for msg in error_handler.error_messages):
                    logger.warning(f"No price data found for {symbol}, likely delisted. Skipping retries.")
                    # write_to_csv(symbol, f"{os.getenv('DIR_PATH')}/failed.csv")
                    return

                # Remove the custom handler
                logging.getLogger('yfinance').removeHandler(error_handler)
                
                logger.info(f"DataFrame columns: {df.columns.tolist()}")
                logger.info(f"DataFrame shape: {df.shape}")
                
                if df.empty:
                    raise ValueError("Empty dataframe returned")
                
                # Flatten the multi-index columns if they exist
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(1)
                
                # Check if required columns exist
                required_columns = ["Open", "High", "Low", "Close"]
                missing_columns = [col for col in required_columns if col not in df.columns]
                if missing_columns:
                    raise ValueError(f"Missing required columns: {missing_columns}")
                
                df['Ticker'] = symbol.replace(".JK", "")
                
                # Round down the Open, High, Low, and Close columns using math.floor
                df[["Open", "High", "Low", "Close"]] = df[["Open", "High", "Low", "Close"]].apply(lambda x: x.apply(math.floor))
                
                df[["Ticker", "Open", "High", "Low", "Close", "Volume"]].to_csv(f"{os.getenv('DIR_PATH')}/csv/{symbol}.csv")
                logger.info(f"Successfully saved data for {symbol} to {os.getenv('DIR_PATH')}/csv/{symbol}.csv")
                return
            except (ChunkedEncodingError, ProtocolError, RequestException) as net_error:
                logger.warning(f"Network error while fetching {symbol}: {net_error}")
            except ValueError as ve:
                logger.warning(f"Value error while fetching {symbol}: {ve}")
            except Exception as error:
                logger.error(f"Unexpected error fetching {symbol} with proxy {use_proxy}: {error}")
            
            if attempt < max_retries - 1:
                wait_time = delay * (2 ** attempt) + random.uniform(0, 1)
                logger.info(f"Retrying {symbol} in {wait_time:.2f} seconds...")
                time.sleep(wait_time)
    
    finally:
        # Cleanup session if it was created
        if session:
            try:
                session.close()
            except Exception as e:
                logger.error(f"Error closing session for {symbol}: {e}")
    
    logger.error(f"Failed to fetch {symbol} after {max_retries} attempts")
    write_to_csv(symbol, f"{os.getenv('DIR_PATH')}/failed.csv")

def write_to_csv(data: Any, file_name: str) -> None:
    """Write data to a CSV file."""
    if isinstance(data, str):
        row = [data]
    else:
        item = data.split(",")
        symbol = "IHSG" if item[0] == "JKSE" else item[0]
        row = [symbol] + item[1:6] + [item[7]]
    
    with open(file_name, 'a', newline='', encoding='utf-8') as f:
        csv.writer(f).writerow(row)

def fetch_async(stock_list: List[str], use_proxy: bool, max_retries: int, initial_delay: int) -> List[str]:
    """
    Fetch stock data asynchronously for a list of symbols and return list of failed stocks.
    
    Args:
        stock_list: List of stock symbols to fetch
        use_proxy: Whether to use proxy for requests
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay between retries
    """
    failed_stocks = []
    with ThreadPoolExecutor(max_workers=int(os.getenv('MAX_WORKERS', 1))) as executor:
        future_to_stock = {
            executor.submit(fetch_stock_data, symbol, use_proxy, max_retries, initial_delay): symbol 
            for symbol in stock_list
        }
        for future in concurrent.futures.as_completed(future_to_stock):
            symbol = future_to_stock[future]
            try:
                result = future.result()
                if result is not None:
                    logger.error(f"Async result error for {symbol}")
                    logger.error(result)
                    failed_stocks.append(symbol)
            except Exception as error:
                logger.error(f"Exception error occurred for {symbol}:")
                logger.error(error)
                logger.error(traceback.format_exc())
                failed_stocks.append(symbol)
    return failed_stocks

def retry_failed_fetches(max_retries: int, initial_delay: int) -> None:
    """Retry fetching data for failed stocks with exponential backoff."""
    failed_csv_path = f"{os.getenv('DIR_PATH')}/failed.csv"
    if not is_empty_csv(failed_csv_path):
        with open(failed_csv_path, "r") as file:
            stock_list = [row[0] for row in csv.reader(file)]
        
        logger.info(f"Retrying {len(stock_list)} failed stocks")
        delay = initial_delay
        for attempt in range(max_retries):
            print(f"Retry attempt {attempt + 1}/{max_retries}")
            remaining_stocks = fetch_async(stock_list)
            
            if not remaining_stocks:
                print("All failed stocks successfully fetched.")
                # Clear the failed.csv file
                open(failed_csv_path, 'w').close()
                return
            
            if attempt < max_retries - 1:  # No need to wait after the last attempt
                wait_time = delay * (2 ** attempt) + random.uniform(0, 1)
                print(f"Retrying remaining stocks in {wait_time:.2f} seconds...")
                time.sleep(wait_time)
            
            stock_list = remaining_stocks
        
        # If we've exhausted all retries, update the failed.csv with remaining stocks
        with open(failed_csv_path, 'w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerows([[stock] for stock in remaining_stocks])
        print(f"Failed to fetch {len(remaining_stocks)} stocks after {max_retries} retry attempts.")
    else:
        print("Nothing to retry")

def merge_csv_files() -> None:
    """Merge all individual stock CSV files into a single result file."""
    files = glob.glob(f"{os.getenv('DIR_PATH')}/csv/*.csv")
    df = pd.concat((pd.read_csv(f, header=0) for f in files))
    df.to_csv(f"{os.getenv('DIR_PATH')}/results.csv", index=False)
    df.to_csv(f"{os.getenv('DIR_PATH')}/results.gama", index=False)

def is_empty_csv(path: str) -> bool:
    """Check if a CSV file is empty (contains only header)."""
    with open(path) as csvfile:
        return sum(1 for _ in csv.reader(csvfile)) <= 1
    
def get_stock_list() -> List[str]:
    """
    Extract stock codes from the CSV file and format them.
    """
    csv_path = os.getenv('STOCK_LIST_PATH')
    df = pd.read_csv(csv_path)
    stock_codes = df['Kode'].tolist()
    
    # Add '.JK' to each stock code
    formatted_codes = [f"{code}.JK" for code in stock_codes]
    
    return formatted_codes

if __name__ == '__main__':
    # Configuration
    USE_PROXY = False
    MAX_RETRIES = int(os.getenv('MAX_RETRIES'))
    INITIAL_DELAY = int(float(os.getenv('INITIAL_DELAY')))  # Convert from float string to int
    
    logger.info("Starting IDX updater...")
    start_time = time.time()

    stock_list = get_stock_list()
    logger.info(f"Loaded {len(stock_list)} stocks to process")
    logger.info(f"Proxy usage is {'enabled' if USE_PROXY else 'disabled'}")

    logger.info("Creating required directories and files...")
    os.makedirs("csv", exist_ok=True)
    open(f"{os.getenv('DIR_PATH')}/failed.csv", "w").close()
    open(f"{os.getenv('DIR_PATH')}/results.csv", "w").close()

    logger.info("Starting async fetch for all stocks...")
    fetch_async(stock_list, use_proxy=USE_PROXY, max_retries=MAX_RETRIES, initial_delay=INITIAL_DELAY)

    logger.info("Starting retry process for failed fetches...")
    retry_failed_fetches(max_retries=MAX_RETRIES, initial_delay=INITIAL_DELAY)

    logger.info("Merging CSV files...")
    merge_csv_files()

    elapsed_time = time.time() - start_time
    logger.info(f"Process completed in {elapsed_time:.2f} seconds")
