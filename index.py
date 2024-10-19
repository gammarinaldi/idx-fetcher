import concurrent.futures
import math
import os
import traceback
import glob
import csv
import time
import random

import yfinance as yf
import pandas as pd
import proxlist

from typing import List
from concurrent.futures import ThreadPoolExecutor
from typing import List, Any
from requests.exceptions import ChunkedEncodingError, RequestException
from urllib3.exceptions import ProtocolError

def fetch_stock_data(symbol: str, max_retries: int = 5, initial_delay: float = 1.0) -> None:
    """Fetch stock data for a given symbol and save it to a CSV file."""
    delay = initial_delay
    for attempt in range(max_retries):
        try:
            try:
                proxy = proxlist.random_proxy()
            except Exception as e:
                print(f"Error getting random proxy: {e}")
                proxy = None
            
            print(f"Fetching {symbol} with proxy {proxy} (Attempt {attempt + 1}/{max_retries})")

            df = yf.download(symbol, period="max", interval="1d", group_by="ticker", proxy=proxy)
            if df.empty:
                raise ValueError("Empty dataframe returned")
            
            df['Ticker'] = symbol.replace(".JK", "")
            
            # Round down the Open, High, Low, and Close columns using math.floor
            df[["Open", "High", "Low", "Close"]] = df[["Open", "High", "Low", "Close"]].apply(lambda x: x.apply(math.floor))
            
            df[["Ticker", "Open", "High", "Low", "Close", "Volume"]].to_csv(f"csv/{symbol}.csv")
            print(f"Fetching {symbol} with proxy {proxy}: success!")
            return  # Success, exit the function
        except (ChunkedEncodingError, ProtocolError, RequestException) as net_error:
            print(f"Network error while fetching {symbol}: {net_error}")
        except ValueError as ve:
            print(f"Value error while fetching {symbol}: {ve}")
        except Exception as error:
            error_message = f"Unexpected error fetching {symbol}"
            if proxy:
                error_message += f" with proxy {proxy}"
            error_message += f": {error}"
            print(error_message)
        
        # If we get here, an error occurred. Wait before retrying.
        if attempt < max_retries - 1:  # No need to wait after the last attempt
            wait_time = delay * (2 ** attempt) + random.uniform(0, 1)
            print(f"Retrying in {wait_time:.2f} seconds...")
            time.sleep(wait_time)
    
    # If we've exhausted all retries, write to the failed CSV
    write_to_csv(symbol, "failed.csv")
    print(f"Failed to fetch {symbol} after {max_retries} attempts")

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

def fetch_async(stock_list: List[str], max_retries: int = 5, initial_delay: float = 1.0) -> List[str]:
    """Fetch stock data asynchronously for a list of symbols and return list of failed stocks."""
    failed_stocks = []
    with ThreadPoolExecutor(max_workers=1) as executor:
        future_to_stock = {executor.submit(fetch_stock_data, symbol, max_retries, initial_delay): symbol for symbol in stock_list}
        for future in concurrent.futures.as_completed(future_to_stock):
            symbol = future_to_stock[future]
            try:
                result = future.result()
                if result is not None:
                    print(f"Async result error for {symbol}")
                    print(result)
                    failed_stocks.append(symbol)
            except Exception as error:
                print(f"Exception error occurred for {symbol}:")
                print(error)
                print(traceback.format_exc())
                failed_stocks.append(symbol)
    return failed_stocks

def retry_failed_fetches(max_retries: int = 3, initial_delay: float = 5.0) -> None:
    """Retry fetching data for failed stocks with exponential backoff."""
    failed_csv_path = "failed.csv"
    if not is_empty_csv(failed_csv_path):
        with open(failed_csv_path, "r") as file:
            stock_list = [row[0] for row in csv.reader(file)]
        
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
    files = glob.glob("csv/*.csv")
    df = pd.concat((pd.read_csv(f, header=0) for f in files))
    df.to_csv("results.csv", index=False)

def is_empty_csv(path: str) -> bool:
    """Check if a CSV file is empty (contains only header)."""
    with open(path) as csvfile:
        return sum(1 for _ in csv.reader(csvfile)) <= 1
    
def get_stock_list() -> List[str]:
    """
    Extract stock codes from the Excel file and format them.
    """
    excel_path = "Daftar-Saham-20241019.xlsx"
    df = pd.read_excel(excel_path)
    stock_codes = df['Kode'].tolist()
    
    # Add '.JK' to each stock code
    formatted_codes = [f"{code}.JK" for code in stock_codes]
    
    return formatted_codes

if __name__ == '__main__':
    print("Start IDX updater...")
    start_time = time.time()

    stock_list = get_stock_list()

    # Create csv folder
    os.makedirs("csv", exist_ok=True)

    # Create failed.csv
    open("failed.csv", "w").close()

    # Create results.csv
    open("results.csv", "w").close()

    # Fetch data
    fetch_async(stock_list)

    # Retry failed fetches
    retry_failed_fetches(max_retries=3, initial_delay=5.0)

    # Merge CSV files
    merge_csv_files()

    elapsed_time = time.time() - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")
