import os
import logging
import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient
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
    mongodb_uri = os.getenv('MONGODB_URI')
    if not mongodb_uri:
        raise ValueError("MONGODB_URI not found in environment variables")
    return MongoClient(mongodb_uri)

def export_stock_list_to_mongodb(csv_path: str) -> None:
    """
    Export stock list from CSV to MongoDB.
    
    Args:
        csv_path: Path to the stock list CSV file
    """
    logger.info(f"Starting stock list export to MongoDB")
    client = setup_mongodb()
    db = client['algosaham_db']
    collection = db['ticker']
    
    try:
        # Read the CSV file
        df = pd.read_csv(csv_path)
        
        # Convert column names to lowercase
        df.columns = df.columns.str.lower()
        
        # Convert DataFrame to list of dictionaries
        records = df.to_dict('records')
        
        # Clear existing data in the collection
        collection.delete_many({})
        
        # Insert new data
        if records:
            result = collection.insert_many(records)
            logger.info(f"Successfully uploaded {len(result.inserted_ids)} stock records")
            
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