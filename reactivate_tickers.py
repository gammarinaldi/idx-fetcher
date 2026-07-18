"""
Bulk reactivate tickers: set is_active=false -> true, reset failure counters.

Usage:
  python reactivate_tickers.py          # aktifkan semua yg is_active=false
  python reactivate_tickers.py --all    # sama, eksplisit
  python reactivate_tickers.py BBCA ACES  # aktifkan ticker tertentu
"""
import os
import sys
import logging
import time
import random
from datetime import datetime
from typing import List, Optional
from dotenv import load_dotenv
from pymongo import MongoClient
from mongodb_tunnel import start_ssh_tunnel

load_dotenv(override=True)


def setup_logging():
    log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
    log_level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    level = log_level_map.get(log_level, logging.INFO)

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)


logger = setup_logging()


def setup_mongodb(max_retries: int = 3, initial_delay: int = 2) -> MongoClient:
    mongodb_uri = os.getenv('MONGODB_URI')
    if not mongodb_uri:
        raise ValueError("MONGODB_URI not found in environment variables")

    delay = initial_delay
    last_error = None

    for attempt in range(max_retries + 1):
        try:
            if attempt > 0:
                logger.info(f"Retrying MongoDB connection (Attempt {attempt}/{max_retries})...")
                start_ssh_tunnel(force=True)
                time.sleep(2)
            else:
                start_ssh_tunnel(force=False)

            client = MongoClient(mongodb_uri, serverSelectionTimeoutMS=5000)
            client.admin.command('ping')

            if attempt > 0:
                logger.info("MongoDB connection established successfully on retry")

            return client

        except Exception as e:
            last_error = e
            if attempt < max_retries:
                wait_time = delay * (2 ** attempt) + random.uniform(0, 1)
                logger.warning(f"MongoDB connection failed: {str(e)}")
                logger.info(f"Retrying in {wait_time:.2f} seconds...")
                time.sleep(wait_time)
            else:
                logger.error(f"Failed to connect to MongoDB after {max_retries} retries: {str(last_error)}")
                raise last_error


def get_db_collection(client: MongoClient):
    db_name = os.getenv('MONGODB_DATABASE')
    if not db_name:
        mongodb_uri = os.getenv('MONGODB_URI', '')
        db_name = mongodb_uri.split('/')[-1].split('?')[0] if mongodb_uri else 'sahamify_db'
    db = client[db_name]
    return db['tickers']


def reactivate_tickers(ticker_filter: Optional[List[str]] = None) -> int:
    client = setup_mongodb()
    collection = get_db_collection(client)

    current_time = datetime.now()
    update_fields = {
        "is_active": True,
        "delisted": False,
        "delisted_at": None,
        "consecutive_failures": 0,
        "last_failed_at": None,
        "status_note": f"reactivated manually at {current_time.strftime('%Y-%m-%d %H:%M:%S')}"
    }

    if ticker_filter:
        query = {"ticker": {"$in": [t.upper().strip() for t in ticker_filter]}}
        logger.info(f"Reactivating specific tickers: {ticker_filter}")
    else:
        query = {"is_active": False}
        logger.info("Reactivating ALL tickers with is_active=false")

    result = collection.update_many(
        query,
        {"$set": update_fields}
    )

    client.close()

    if result.modified_count > 0:
        logger.info(f"Reactivated {result.modified_count} ticker(s)")
    else:
        logger.info("No tickers were modified (either not found or already active)")

    return result.modified_count


def main():
    args = sys.argv[1:]

    if not args or args[0] == '--all':
        count = reactivate_tickers()
    else:
        count = reactivate_tickers(ticker_filter=[a.upper().strip() for a in args])

    if count > 0:
        print(f"\n{'='*50}")
        print(f"  {count} ticker(s) reactivated successfully.")
        print(f"{'='*50}")
    else:
        print("\nNo tickers were modified.")


if __name__ == '__main__':
    main()
