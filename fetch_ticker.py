import os
import logging
import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient
from mongodb_tunnel import start_ssh_tunnel
import time

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

def setup_mongodb() -> MongoClient:
    """Initialize and return a MongoDB client."""
    # Start SSH tunnel if configured
    start_ssh_tunnel()
    
    mongodb_uri = os.getenv('MONGODB_URI')
    if not mongodb_uri:
        raise ValueError("MONGODB_URI not found in environment variables")
    return MongoClient(mongodb_uri)

def export_stock_list_to_mongodb(csv_path: str) -> None:
    """
    Export stock list from CSV to MongoDB.
    Only exports stocks that don't exist or have empty kode values.
    
    Args:
        csv_path: Path to the stock list CSV file
    """
    logger.info(f"Starting stock list export to MongoDB")
    client = setup_mongodb()
    # Get database name from environment variable or extract from URI
    db_name = os.getenv('MONGODB_DATABASE')
    if not db_name:
        mongodb_uri = os.getenv('MONGODB_URI')
        db_name = mongodb_uri.split('/')[-1].split('?')[0] if mongodb_uri else 'sahamify_db'
    db = client[db_name]
    collection = db['tickers']
    
    try:
        # Read the CSV file
        df = pd.read_csv(csv_path)
        
        # Convert column names to lowercase
        df.columns = df.columns.str.lower()
        
        # Get existing records
        existing_records = list(collection.find({}, {'kode': 1, '_id': 0}))
        existing_kodes = {record['kode'] for record in existing_records if record.get('kode')}
        
        # Filter records to insert
        records_to_insert = []
        now = pd.Timestamp.now()
        
        for record in df.to_dict('records'):
            if record['kode'] not in existing_kodes:
                # Add createdAt, updatedAt and is_active fields
                record['createdAt'] = now
                record['updatedAt'] = now
                record['is_active'] = True
                records_to_insert.append(record)
        
        # Insert new records
        if records_to_insert:
            result = collection.insert_many(records_to_insert)
            logger.info(f"Successfully uploaded {len(result.inserted_ids)} new stock records")
        else:
            logger.info("No new records to insert")
            
    except Exception as e:
        logger.error(f"Error during stock list export: {str(e)}")
        raise
    finally:
        client.close()

if __name__ == '__main__':
    logger.info("Starting stock list export process...")
    start_time = time.time()
    
    export_stock_list_to_mongodb(
        csv_path=os.getenv('DIR_PATH') + "/" + os.getenv('STOCK_LIST_PATH')
    )
    
    elapsed_time = time.time() - start_time
    logger.info(f"Process completed in {elapsed_time:.2f} seconds") 