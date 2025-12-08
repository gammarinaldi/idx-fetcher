import concurrent.futures
import math
import os
import sys
import traceback
import csv
import time
import random
import logging
import threading
from typing import List, Any, Optional
from concurrent.futures import ThreadPoolExecutor
from requests.exceptions import ChunkedEncodingError, RequestException
from urllib3.exceptions import ProtocolError
import yfinance as yf
import pandas as pd
import requests
from dotenv import load_dotenv
from pymongo import MongoClient, ReplaceOne
from curl_cffi import requests
from datetime import datetime
import queue
import platform
import pytz

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

def get_system_timezone_info():
    """
    Detect system timezone and return timezone information.
    
    Returns:
        tuple: (is_utc, timezone_name, utc_offset)
    """
    try:
        # Get system timezone
        if platform.system() == "Windows":
            # On Windows, get timezone info from registry or environment
            import winreg
            try:
                with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\TimeZoneInformation") as key:
                    timezone_name = winreg.QueryValueEx(key, "TimeZoneKeyName")[0]
            except:
                timezone_name = os.environ.get('TZ', 'UTC')
        else:
            # On Unix-like systems
            timezone_name = os.environ.get('TZ', 'UTC')
        
        # Get current UTC offset
        utc_now = datetime.now(pytz.UTC)
        local_now = datetime.now()
        utc_offset = (local_now - utc_now.replace(tzinfo=None)).total_seconds() / 3600
        
        # Check if system is in UTC+0
        is_utc = abs(utc_offset) < 0.1  # Allow small floating point errors
        
        logger.info(f"System timezone: {timezone_name}, UTC offset: {utc_offset:.1f}, Is UTC: {is_utc}")
        
        return is_utc, timezone_name, utc_offset
        
    except Exception as e:
        logger.warning(f"Could not detect system timezone: {e}, defaulting to UTC")
        return True, 'UTC', 0.0

def get_target_timezone():
    """
    Get the target timezone for Jakarta market operations.
    Returns Jakarta timezone object.
    """
    return pytz.timezone('Asia/Jakarta')

def is_market_closed() -> bool:
    """
    Check if today is a market holiday or weekend.
    
    Returns:
        bool: True if market is closed (holiday or weekend), False otherwise
    """
    # Get current time in Asia/Jakarta timezone (UTC+7)
    jakarta_tz = get_target_timezone()
    today = datetime.now(jakarta_tz)
    
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
    # Get database name from environment variable or extract from URI
    db_name = os.getenv('MONGODB_DATABASE')
    if not db_name:
        mongodb_uri = os.getenv('MONGODB_URI')
        db_name = mongodb_uri.split('/')[-1].split('?')[0] if mongodb_uri else 'sahamify_db'
    db = client[db_name]
    collection = db[collection_name]
    
    try:
        # Drop existing unique index if it exists to allow multiple entries per day
        try:
            collection.drop_index([("date", 1), ("ticker", 1)])
            logger.info("Dropped existing unique index on (date, ticker)")
        except Exception as drop_error:
            logger.info(f"No existing unique index to drop: {str(drop_error)}")
        
        # Create compound index on date and ticker (non-unique to allow multiple entries per day)
        collection.create_index([("date", 1), ("ticker", 1)], background=True)
        logger.info("Successfully created compound index on (date, ticker)")
        
        # Create individual indexes for common queries
        collection.create_index([("date", 1)], background=True)
        collection.create_index([("ticker", 1)], background=True)
        logger.info("Successfully created individual indexes on date and ticker")
        
    except Exception as e:
        logger.warning(f"Index creation warning (may already exist): {str(e)}")
    finally:
        client.close()

class ThreadSafeResultsWriter:
    """Thread-safe writer for results files with batching."""
    
    def __init__(self, results_csv_path: str, results_gama_path: str, batch_size: int = 100):
        self.results_csv_path = results_csv_path
        self.results_gama_path = results_gama_path
        self.batch_size = batch_size
        self.data_queue = queue.Queue()
        self.lock = threading.Lock()
        self.total_records = 0
        
        # Initialize files with headers
        self._initialize_files()
    
    def _initialize_files(self):
        """Initialize result files with CSV headers."""
        headers = "Ticker,Date,Open,High,Low,Close,Volume\n"
        
        with self.lock:
            with open(self.results_csv_path, 'w', encoding='utf-8') as f:
                f.write(headers)
            with open(self.results_gama_path, 'w', encoding='utf-8') as f:
                f.write(headers)
    
    def add_data(self, ticker: str, df: pd.DataFrame):
        """Add stock data to the queue for batch writing."""
        try:
            # Process the dataframe
            processed_df = self._process_dataframe(ticker, df)
            if not processed_df.empty:
                self.data_queue.put(processed_df)
                self._write_batch_if_needed()
        except Exception as e:
            logger.error(f"Error processing data for {ticker}: {str(e)}")
    
    def _process_dataframe(self, ticker: str, df: pd.DataFrame) -> pd.DataFrame:
        """Process and format the dataframe for output."""
        # Flatten multi-index columns if they exist
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(1)
        
        # Check required columns
        required_columns = ["Open", "High", "Low", "Close"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Clean ticker name
        clean_ticker = ticker.replace(".JK", "")
        df['Ticker'] = clean_ticker
        
        # Round down prices using math.floor
        df[["Open", "High", "Low", "Close"]] = df[["Open", "High", "Low", "Close"]].apply(
            lambda x: x.apply(math.floor)
        )
        
        # Reset index to make Date a column
        df = df.reset_index()
        if 'Date' not in df.columns and 'Datetime' in df.columns:
            df = df.rename(columns={'Datetime': 'Date'})
        
        # Select and order columns
        output_columns = ["Ticker", "Date", "Open", "High", "Low", "Close", "Volume"]
        available_columns = [col for col in output_columns if col in df.columns]
        
        return df[available_columns]
    
    def _write_batch_if_needed(self):
        """Write batch to files if queue size reaches batch_size."""
        if self.data_queue.qsize() >= self.batch_size:
            self._write_batch()
    
    def _write_batch(self):
        """Write all queued data to files."""
        if self.data_queue.empty():
            return
        
        batch_data = []
        while not self.data_queue.empty():
            try:
                df = self.data_queue.get_nowait()
                batch_data.append(df)
            except queue.Empty:
                break
        
        if batch_data:
            combined_df = pd.concat(batch_data, ignore_index=True)
            self._write_to_files(combined_df)
            self.total_records += len(combined_df)
            logger.info(f"Wrote batch of {len(combined_df)} records. Total: {self.total_records}")
    
    def _write_to_files(self, df: pd.DataFrame):
        """Write dataframe to both result files."""
        with self.lock:
            # Append to CSV files
            df.to_csv(self.results_csv_path, mode='a', header=False, index=False, encoding='utf-8')
            df.to_csv(self.results_gama_path, mode='a', header=False, index=False, encoding='utf-8')
    
    def flush(self):
        """Write any remaining data in the queue."""
        self._write_batch()
        logger.info(f"Final flush completed. Total records written: {self.total_records}")

class OptimizedMongoDBUploader:
    """Optimized MongoDB uploader that processes data in batches."""
    
    def __init__(self, collection_name: str, batch_size: int = 1000):
        self.collection_name = collection_name
        self.batch_size = batch_size
        self.client = None
        self.collection = None
        self.total_inserted = 0
        self.total_updated = 0
        self.total_operations = 0
        
    def __enter__(self):
        """Context manager entry."""
        try:
            self.client = setup_mongodb()
            # Test connection
            self.client.admin.command('ping')
            logger.info("MongoDB connection test successful")
            
            # Get database name from environment variable or extract from URI
            db_name = os.getenv('MONGODB_DATABASE')
            if not db_name:
                mongodb_uri = os.getenv('MONGODB_URI')
                db_name = mongodb_uri.split('/')[-1].split('?')[0] if mongodb_uri else 'sahamify_db'
            db = self.client[db_name]
            self.collection = db[self.collection_name]
            
            logger.info(f"Connected to database: {db_name}, collection: {self.collection_name}")
            
            # Create indexes
            create_mongodb_indexes(self.collection_name)
            return self
        except Exception as e:
            logger.error(f"Failed to initialize MongoDB connection: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self.client:
            self.client.close()
    
    def upload_batch(self, df: pd.DataFrame):
        """Upload a batch of data to MongoDB using bulk operations for much faster performance."""
        if df.empty:
            return 0, 0
        
        # Prepare data for MongoDB
        df_copy = df.copy()
        df_copy.columns = df_copy.columns.str.lower()
        # Convert date to datetime and add current time with minutes in Jakarta timezone
        jakarta_tz = get_target_timezone()
        current_time = datetime.now(jakarta_tz)
        df_copy['date'] = pd.to_datetime(df_copy['date']).dt.date.apply(
            lambda x: datetime.combine(x, current_time.time().replace(second=0, microsecond=0))
        )
        
        records = df_copy.to_dict('records')
        
        # Clean up old data for same ticker and date (without time)
        self._cleanup_old_data_for_tickers(records)
        
        # Use bulk_write for much better performance
        # Instead of individual operations, batch them all together
        operations = []
        for record in records:
            operations.append(
                ReplaceOne(
                    {"date": record["date"], "ticker": record["ticker"]},
                    record,
                    upsert=True
                )
            )
        
        total_inserted = 0
        total_updated = 0
        
        try:
            # Execute all operations in a single bulk write (MUCH faster!)
            if operations:
                result = self.collection.bulk_write(operations, ordered=False)
                total_inserted = result.upserted_count
                total_updated = result.modified_count
                self.total_inserted += total_inserted
                self.total_updated += total_updated
                self.total_operations += len(operations)
                logger.info(f"Bulk upload: {total_inserted} inserted, {total_updated} updated from {len(operations)} operations")
        except Exception as e:
            logger.error(f"Error during bulk write: {str(e)}")
            logger.error(traceback.format_exc())
            # Re-raise to ensure the error is caught and logged at a higher level
            raise
        
        return total_inserted, total_updated
    
    def _cleanup_old_data_for_tickers(self, records: List[dict]):
        """Clean up old data for the same ticker and date (without time) to keep only fresh data."""
        if not records:
            return
        
        try:
            # Group records by ticker and date (without time)
            ticker_date_groups = {}
            for record in records:
                ticker = record["ticker"]
                date_without_time = record["date"].date()  # Get date without time
                key = (ticker, date_without_time)
                
                if key not in ticker_date_groups:
                    ticker_date_groups[key] = []
                ticker_date_groups[key].append(record)
            
            # Delete old records for each ticker-date combination
            total_deleted = 0
            for (ticker, date_without_time), group_records in ticker_date_groups.items():
                # Delete all existing records for this ticker and date
                delete_result = self.collection.delete_many({
                    "ticker": ticker,
                    "date": {
                        "$gte": datetime.combine(date_without_time, datetime.min.time()),
                        "$lt": datetime.combine(date_without_time + pd.Timedelta(days=1), datetime.min.time())
                    }
                })
                
                if delete_result.deleted_count > 0:
                    total_deleted += delete_result.deleted_count
                    logger.info(f"Cleaned up {delete_result.deleted_count} old records for {ticker} on {date_without_time}")
            
            if total_deleted > 0:
                logger.info(f"Total old records cleaned up: {total_deleted}")
                
        except Exception as e:
            logger.error(f"Error during cleanup of old data: {str(e)}")
            # Continue with upload even if cleanup fails

def fetch_stock_data_optimized(symbol: str, max_retries: int, initial_delay: int, 
                              results_writer: ThreadSafeResultsWriter, 
                              mongo_uploader: Optional[OptimizedMongoDBUploader] = None) -> bool:
    
    """
    Fetch stock data for a given symbol and write directly to results files.
    
    Args:
        symbol: Stock symbol to fetch
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay between retries
        results_writer: Thread-safe writer for results files
        mongo_uploader: Optional MongoDB uploader for direct database insertion
    
    Returns:
        bool: True if successful, False if failed
    """
    logger.info(f"Starting fetch for symbol: {symbol}")
    delay = initial_delay
    session = None
    
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
                    return False

                # Remove the custom handler
                logging.getLogger('yfinance').removeHandler(error_handler)
                
                if df.empty:
                    raise ValueError("Empty dataframe returned")
                
                logger.info(f"Successfully fetched data for {symbol} - Shape: {df.shape}")
                
                # Write directly to results files
                results_writer.add_data(symbol, df)
                
                # Optionally upload to MongoDB directly
                if mongo_uploader:
                    try:
                        processed_df = results_writer._process_dataframe(symbol, df)
                        if not processed_df.empty:
                            inserted, updated = mongo_uploader.upload_batch(processed_df)
                            logger.info(f"MongoDB upload for {symbol}: {inserted} inserted, {updated} updated")
                        else:
                            logger.warning(f"Processed dataframe for {symbol} is empty - skipping MongoDB upload")
                    except Exception as mongo_error:
                        logger.error(f"Failed to upload {symbol} to MongoDB: {str(mongo_error)}")
                        logger.error(traceback.format_exc())
                        # Don't re-raise - continue processing other symbols, but track the failure
                        # The final check will verify if any data was uploaded
                
                return True
                
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
    return False

def write_to_csv(data: Any, file_name: str) -> None:
    """Write data to a CSV file."""
    with open(file_name, 'a', newline='', encoding='utf-8') as f:
        csv.writer(f).writerow([data])

def fetch_async_optimized(stock_list: List[str], max_retries: int, initial_delay: int,
                         results_writer: ThreadSafeResultsWriter,
                         mongo_uploader: Optional[OptimizedMongoDBUploader] = None) -> List[str]:
    """
    Fetch stock data asynchronously for a list of symbols using optimized approach.
    
    Args:
        stock_list: List of stock symbols to fetch
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay between retries
        results_writer: Thread-safe writer for results files
        mongo_uploader: Optional MongoDB uploader for direct database insertion
    
    Returns:
        List[str]: List of failed stock symbols
    """
    failed_stocks = []
    
    with ThreadPoolExecutor(max_workers=int(os.getenv('MAX_WORKERS', 1))) as executor:
        future_to_stock = {
            executor.submit(fetch_stock_data_optimized, symbol, max_retries, initial_delay, 
                          results_writer, mongo_uploader): symbol 
            for symbol in stock_list
        }
        
        for future in concurrent.futures.as_completed(future_to_stock):
            symbol = future_to_stock[future]
            try:
                success = future.result()
                
                if not success:
                    logger.error(f"Failed to fetch data for {symbol}")
                    failed_stocks.append(symbol)
                    
            except Exception as error:
                logger.error(f"Exception error occurred for {symbol}:")
                logger.error(error)
                logger.error(traceback.format_exc())
                failed_stocks.append(symbol)
    
    # Flush any remaining data
    results_writer.flush()
                
    return failed_stocks

def retry_failed_fetches_optimized(max_retries: int, initial_delay: int,
                                 results_writer: ThreadSafeResultsWriter,
                                 mongo_uploader: Optional[OptimizedMongoDBUploader] = None) -> None:
    """Retry fetching data for failed stocks with optimized approach."""
    dir_path = os.getenv('DIR_PATH', '/app')
    failed_csv_path = os.path.join(dir_path, "failed.csv")
    
    if os.path.exists(failed_csv_path) and os.path.getsize(failed_csv_path) > 0:
        with open(failed_csv_path, "r") as file:
            stock_list = [row[0] for row in csv.reader(file)]
        
        logger.info(f"Retrying {len(stock_list)} failed stocks")
        
        for attempt in range(max_retries):
            logger.info(f"Retry attempt {attempt + 1}/{max_retries}")
            remaining_stocks = fetch_async_optimized(stock_list, max_retries=1, 
                                                   initial_delay=initial_delay,
                                                   results_writer=results_writer,
                                                   mongo_uploader=mongo_uploader)
            
            if not remaining_stocks:
                logger.info("All failed stocks successfully fetched.")
                # Clear the failed.csv file
                open(failed_csv_path, 'w').close()
                return
            
            if attempt < max_retries - 1:  # No need to wait after the last attempt
                wait_time = initial_delay * (2 ** attempt) + random.uniform(0, 1)
                logger.info(f"Retrying remaining stocks in {wait_time:.2f} seconds...")
                time.sleep(wait_time)
            
            stock_list = remaining_stocks
        
        # If we've exhausted all retries, update the failed.csv with remaining stocks
        with open(failed_csv_path, 'w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerows([[stock] for stock in remaining_stocks])
        logger.error(f"Failed to fetch {len(remaining_stocks)} stocks after {max_retries} retry attempts.")
    else:
        logger.info("No failed stocks to retry")

def is_empty_csv(path: str) -> bool:
    """Check if a CSV file is empty (contains only header)."""
    if not os.path.exists(path):
        return True
    with open(path) as csvfile:
        return sum(1 for _ in csv.reader(csvfile)) <= 1
    
def get_stock_list() -> List[str]:
    """
    Get list of stock codes from CSV file and format them.
    """
    # Get directory path and stock list filename from environment variables
    dir_path = os.getenv('DIR_PATH', '/app')  # Default to /app for Docker
    stock_list_filename = os.getenv('STOCK_LIST_PATH', 'stock_list.csv')
    
    # Construct the CSV path
    csv_path = os.path.join(dir_path, stock_list_filename)
    
    # If the file doesn't exist at the expected path, try alternative locations
    if not os.path.exists(csv_path):
        logger.warning(f"Stock list not found at {csv_path}, trying alternative locations...")
        
        # Try current directory
        current_dir_path = os.path.join(os.getcwd(), stock_list_filename)
        if os.path.exists(current_dir_path):
            csv_path = current_dir_path
            logger.info(f"Found stock list at current directory: {csv_path}")
        else:
            # Try script directory
            script_dir = os.path.dirname(os.path.abspath(__file__))
            script_dir_path = os.path.join(script_dir, stock_list_filename)
            if os.path.exists(script_dir_path):
                csv_path = script_dir_path
                logger.info(f"Found stock list at script directory: {csv_path}")
            else:
                # Last resort: try just the filename in current directory
                if os.path.exists(stock_list_filename):
                    csv_path = stock_list_filename
                    logger.info(f"Found stock list in current directory: {csv_path}")
                else:
                    raise FileNotFoundError(f"Stock list file not found. Tried: {csv_path}, {current_dir_path}, {script_dir_path}, {stock_list_filename}")
    
    logger.info(f"Loading stock list from: {csv_path}")
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
    
    logger.info("Starting Optimized IDX updater...")
    
    # Check if market is closed (weekend or holiday)
    if os.getenv('ENABLE_HOLIDAY_CHECK', 'TRUE').upper() == 'TRUE':
        if is_market_closed():
            logger.info("Market is closed (weekend or holiday) - Skipping market data fetch process.")
            exit()
    
    start_time = time.time()

    stock_list = get_stock_list()
    logger.info(f"Loaded {len(stock_list)} stocks to process")

    logger.info("Initializing optimized workflow...")
    
    # Initialize result files
    dir_path = os.getenv('DIR_PATH', '/app')
    results_csv = os.path.join(dir_path, "results.csv")
    results_gama = os.path.join(dir_path, "results.gama")
    failed_csv = os.path.join(dir_path, "failed.csv")
    
    # Initialize failed.csv
    open(failed_csv, "w").close()
    
    # Initialize thread-safe results writer
    results_writer = ThreadSafeResultsWriter(results_csv, results_gama, batch_size=50)
    
    # Initialize MongoDB uploader if enabled
    mongo_uploader = None
    upload_to_mongodb = os.getenv('UPLOAD_TO_MONGODB', 'FALSE').upper() == 'TRUE'
    logger.info(f"MongoDB upload status: {'ENABLED' if upload_to_mongodb else 'DISABLED'}")
    
    if upload_to_mongodb:
        logger.info("MongoDB upload enabled - initializing uploader")
        mongo_collection = os.getenv('MONGODB_COLLECTION', 'daily_market_data')
        if not mongo_collection:
            logger.error("MONGODB_COLLECTION environment variable is not set")
            raise ValueError("MONGODB_COLLECTION environment variable is required when UPLOAD_TO_MONGODB is enabled")
        try:
            mongo_uploader = OptimizedMongoDBUploader(mongo_collection, batch_size=1000)
            mongo_uploader.__enter__()
            logger.info(f"MongoDB uploader initialized successfully for collection: {mongo_collection}")
        except Exception as e:
            logger.error(f"Failed to initialize MongoDB uploader: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    else:
        logger.warning("MongoDB upload is DISABLED - data will only be written to CSV files")

    mongo_upload_success = True
    try:
        logger.info("Starting optimized async fetch for all stocks...")
        failed_stocks = fetch_async_optimized(stock_list, max_retries=MAX_RETRIES, 
                                            initial_delay=INITIAL_DELAY,
                                            results_writer=results_writer,
                                            mongo_uploader=mongo_uploader)
        
        # Write failed stocks to CSV
        if failed_stocks:
            with open(failed_csv, 'w', newline='', encoding='utf-8') as f:
                csv.writer(f).writerows([[stock] for stock in failed_stocks])
            logger.warning(f"Failed to fetch {len(failed_stocks)} stocks on first attempt")

        logger.info("Starting retry process for failed fetches...")
        retry_failed_fetches_optimized(max_retries=MAX_RETRIES, 
                                     initial_delay=INITIAL_DELAY,
                                     results_writer=results_writer,
                                     mongo_uploader=mongo_uploader)

    except Exception as e:
        logger.error(f"Critical error during data fetch/upload: {str(e)}")
        logger.error(traceback.format_exc())
        mongo_upload_success = False
        raise
    finally:
        # Cleanup MongoDB connection
        if mongo_uploader:
            try:
                mongo_uploader.__exit__(None, None, None)
                # Log MongoDB upload summary
                logger.info("=" * 60)
                logger.info("MongoDB Upload Summary:")
                logger.info(f"  Total operations: {mongo_uploader.total_operations}")
                logger.info(f"  Total inserted: {mongo_uploader.total_inserted}")
                logger.info(f"  Total updated: {mongo_uploader.total_updated}")
                logger.info(f"  Collection: {mongo_uploader.collection_name}")
                if mongo_uploader.total_operations == 0:
                    logger.warning("  WARNING: No data was uploaded to MongoDB!")
                    mongo_upload_success = False
                logger.info("=" * 60)
            except Exception as e:
                logger.error(f"Error during MongoDB cleanup: {str(e)}")
                mongo_upload_success = False

    elapsed_time = time.time() - start_time
    logger.info(f"Optimized process completed in {elapsed_time:.2f} seconds")
    logger.info(f"Results saved to {results_csv} and {results_gama}")
    
    # Exit with error code if MongoDB upload was enabled but failed
    if upload_to_mongodb and not mongo_upload_success:
        logger.error("MongoDB upload was enabled but failed - exiting with error code")
        sys.exit(1)
