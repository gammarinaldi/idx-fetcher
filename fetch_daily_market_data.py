import concurrent.futures
import math
import os
import traceback
import glob
import csv
import time
import random
import logging
from typing import List, Any
from concurrent.futures import ThreadPoolExecutor
from requests.exceptions import ChunkedEncodingError, RequestException
from urllib3.exceptions import ProtocolError
import yfinance as yf
import pandas as pd
import requests
from dotenv import load_dotenv
from pymongo import MongoClient
from curl_cffi import requests
from datetime import datetime

# Load environment variables
load_dotenv(override=True)

def setup_logging():
    """Configure logging with timestamp, level, and message."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)

logger = setup_logging()

def is_market_closed() -> bool:
    """
    Check if today is a market holiday or weekend.
    
    Returns:
        bool: True if market is closed (holiday or weekend), False otherwise
    """
    today = datetime.now()
    
    # Check if it's weekend (Saturday = 5, Sunday = 6)
    if today.weekday() >= 5:  # Saturday or Sunday
        logger.info(f"Today is weekend ({today.strftime('%A')}) - Market is closed")
        return True
    
    # Check if it's a holiday
    try:
        month = today.month
        year = today.year
        
        # Call holiday API
        response = requests.get(f"https://dayoffapi.vercel.app/api?month={month}&year={year}")
        if response.status_code == 200:
            holidays = response.json()
            today_str = today.strftime("%Y-%m-%d")
            
            # Check if today is in the holiday list
            for holiday in holidays:
                # Convert holiday date to match format (add leading zero if needed)
                holiday_date = holiday["tanggal"]
                if len(holiday_date.split("-")[2]) == 1:  # If day is single digit
                    parts = holiday_date.split("-")
                    holiday_date = f"{parts[0]}-{parts[1]}-0{parts[2]}"
                
                if holiday_date == today_str:
                    logger.info(f"Today is a holiday: {holiday['keterangan']}")
                    return True
        return False
    except Exception as e:
        logger.warning(f"Error checking holiday: {str(e)}")
        return False

def setup_mongodb() -> MongoClient:
    """Initialize and return a MongoDB client."""
    mongodb_uri = os.getenv('MONGODB_URI')
    if not mongodb_uri:
        raise ValueError("MONGODB_URI not found in environment variables")
    return MongoClient(mongodb_uri)

def create_mongodb_indexes(collection_name: str) -> None:
    """
    Create indexes for the MongoDB collection to ensure uniqueness and query performance.
    
    Args:
        collection_name: Name of the MongoDB collection
    """
    logger.info(f"Creating indexes for collection: {collection_name}")
    client = setup_mongodb()
    db = client['algosaham_db']
    collection = db[collection_name]
    
    try:
        # Create compound unique index on date and ticker to prevent duplicates
        collection.create_index([("date", 1), ("ticker", 1)], unique=True, background=True)
        logger.info("Successfully created unique compound index on (date, ticker)")
        
        # Create individual indexes for common queries
        collection.create_index([("date", 1)], background=True)
        collection.create_index([("ticker", 1)], background=True)
        logger.info("Successfully created individual indexes on date and ticker")
        
    except Exception as e:
        logger.warning(f"Index creation warning (may already exist): {str(e)}")
    finally:
        client.close()

def upload_to_mongodb(csv_path: str, collection_name: str, batch_size: int = 1000) -> None:
    """
    Upload CSV data to MongoDB in batches with duplicate handling using upsert operations.
    
    Args:
        csv_path: Path to the CSV file
        collection_name: Name of the MongoDB collection to upload to
        batch_size: Number of rows to upload in each batch
    """
    logger.info(f"Starting upload to MongoDB collection: {collection_name}")
    client = setup_mongodb()
    db = client['algosaham_db']  # Use specific database name
    collection = db[collection_name]
    
    try:
        # Create indexes if they don't exist
        create_mongodb_indexes(collection_name)
        
        total_updated = 0
        total_inserted = 0
        
        # Read the CSV file in chunks
        for chunk_num, chunk in enumerate(pd.read_csv(csv_path, chunksize=batch_size)):
            logger.info(f"Processing chunk {chunk_num + 1}")
            
            # Convert column names to lowercase
            chunk.columns = chunk.columns.str.lower()
            
            # Convert date column to datetime
            chunk['date'] = pd.to_datetime(chunk['date'])
            
            # Convert the chunk to a list of dictionaries
            records = chunk.to_dict('records')
            
            # Use upsert operations to handle duplicates
            if records:
                batch_updated = 0
                batch_inserted = 0
                
                for record in records:
                    try:
                        # Use upsert to either insert new or update existing record
                        result = collection.replace_one(
                            {"date": record["date"], "ticker": record["ticker"]},
                            record,
                            upsert=True
                        )
                        
                        if result.upserted_id:
                            batch_inserted += 1
                        elif result.modified_count > 0:
                            batch_updated += 1
                            
                    except Exception as record_error:
                        logger.warning(f"Error processing record {record.get('ticker', 'unknown')} on {record.get('date', 'unknown')}: {str(record_error)}")
                        continue
                
                total_updated += batch_updated
                total_inserted += batch_inserted
                
                logger.info(f"Batch complete - Inserted: {batch_inserted}, Updated: {batch_updated}")
        
        logger.info(f"Upload complete - Total Inserted: {total_inserted}, Total Updated: {total_updated}")
            
    except Exception as e:
        logger.error(f"Error during MongoDB upload: {str(e)}")
        raise
    finally:
        client.close()

def fetch_stock_data(symbol: str, max_retries: int, initial_delay: int) -> None:
    """
    Fetch stock data for a given symbol and save it to a CSV file.
    
    Args:
        symbol: Stock symbol to fetch
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay between retries
    """
    logger.info(f"Starting fetch for symbol: {symbol}")
    delay = initial_delay
    
    try:
        for attempt in range(max_retries):
            try:
                logger.info(f"Fetching {symbol} (Attempt {attempt + 1}/{max_retries})")
                
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

                session = requests.Session(impersonate="chrome")
                df = yf.download(symbol, period=PERIOD, interval=INTERVAL, group_by=GROUP_BY, session=session)
                    
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
    with open(file_name, 'a', newline='', encoding='utf-8') as f:
        csv.writer(f).writerow([data])

def fetch_async(stock_list: List[str], max_retries: int, initial_delay: int) -> List[str]:
    """
    Fetch stock data asynchronously for a list of symbols and return list of failed stocks.
    
    Args:
        stock_list: List of stock symbols to fetch
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay between retries
    """
    failed_stocks = []
    with ThreadPoolExecutor(max_workers=int(os.getenv('MAX_WORKERS', 1))) as executor:
        future_to_stock = {
            executor.submit(fetch_stock_data, symbol, max_retries, initial_delay): symbol 
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
    Get list of stock codes from CSV file and format them.
    """
    csv_path = os.getenv('DIR_PATH') + "/" + os.getenv('STOCK_LIST_PATH')
    df = pd.read_csv(csv_path)
    stock_codes = df['Kode'].tolist()
    
    # Format stock codes with special handling for JKSE
    formatted_codes = []
    for code in stock_codes:
        if code == "JKSE":
            formatted_codes.append("^JKSE")
        else:
            formatted_codes.append(f"{code}.JK")
    
    return formatted_codes

if __name__ == '__main__':
    # Configuration
    MAX_RETRIES = int(os.getenv('MAX_RETRIES'))
    INITIAL_DELAY = int(float(os.getenv('INITIAL_DELAY')))
    
    logger.info("Starting IDX updater...")
    
    # Check if market is closed (weekend or holiday)
    if is_market_closed():
        logger.info("Market is closed (weekend or holiday) - Skipping market data fetch process.")
        exit()
    
    start_time = time.time()

    stock_list = get_stock_list()
    logger.info(f"Loaded {len(stock_list)} stocks to process")

    logger.info("Creating required directories and files...")
    os.makedirs("csv", exist_ok=True)
    open(f"{os.getenv('DIR_PATH')}/failed.csv", "w").close()
    open(f"{os.getenv('DIR_PATH')}/results.csv", "w").close()

    logger.info("Starting async fetch for all stocks...")
    fetch_async(stock_list, max_retries=MAX_RETRIES, initial_delay=INITIAL_DELAY)

    logger.info("Starting retry process for failed fetches...")
    retry_failed_fetches(max_retries=MAX_RETRIES, initial_delay=INITIAL_DELAY)

    logger.info("Merging CSV files...")
    merge_csv_files()

    # Check if MongoDB upload is enabled
    if os.getenv('UPLOAD_TO_MONGODB', 'FALSE').upper() == 'TRUE':
        # Upload to MongoDB
        logger.info("Starting MongoDB upload...")
        upload_to_mongodb(
            csv_path=f"{os.getenv('DIR_PATH')}/results.csv",
            collection_name="daily_market_data",
            batch_size=1000
        )
    else:
        logger.info("MongoDB upload skipped as UPLOAD_TO_MONGODB is set to FALSE")

    elapsed_time = time.time() - start_time
    logger.info(f"Process completed in {elapsed_time:.2f} seconds")
