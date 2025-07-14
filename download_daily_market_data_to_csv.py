import os
import logging
import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient

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

def download_daily_market_data_to_csv(output_file: str = None) -> None:
    """
    Download all data from daily_market_data collection to CSV file.
    
    Args:
        output_file: Path to the output CSV file. If None, uses default name with timestamp.
    """
    logger.info("Starting download of daily_market_data collection to CSV")
    
    
    client = setup_mongodb()
    db = client['algosaham_db']
    collection = db['daily_market_data']
    
    try:
        # Get total count of documents
        total_documents = collection.count_documents({})
        logger.info(f"Found {total_documents} documents in daily_market_data collection")
        
        # Fetch all documents from the collection with default sorting by date ascending
        logger.info("Fetching all documents from daily_market_data collection...")
        cursor = collection.find({}).sort("date", 1)
        
        # Convert cursor to list of dictionaries
        documents = list(cursor)
        logger.info(f"Successfully fetched {len(documents)} documents")
        
        if len(documents) == 0:
            logger.warning("No documents found in daily_market_data collection")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(documents)
        
        # Remove MongoDB _id column if it exists
        if '_id' in df.columns:
            df = df.drop('_id', axis=1)
        
        # Convert date column to string format for better CSV compatibility
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
        
        # Save to CSV
        df.to_csv(output_file, index=False)
        logger.info(f"Successfully saved {len(df)} records to {output_file}")
        
        # Display some statistics
        logger.info(f"CSV file created: {output_file}")
        logger.info(f"File size: {os.path.getsize(output_file) / 1024:.2f} KB")
        logger.info(f"Columns: {list(df.columns)}")
        logger.info(f"Data types: {df.dtypes.to_dict()}")
        
        # Show sample data
        if not df.empty:
            logger.info("Sample data (first 5 rows):")
            logger.info(df.head().to_string())
        
    except Exception as e:
        logger.error(f"Error downloading data to CSV: {str(e)}")
        raise
    finally:
        client.close()

def download_with_filters(output_file: str = None, filters: dict = None, sort_by: str = None) -> None:
    """
    Download filtered data from daily_market_data collection to CSV file.
    
    Args:
        output_file: Path to the output CSV file. If None, uses default name with timestamp.
        filters: MongoDB query filters (e.g., {"ticker": "BBCA"})
        sort_by: Field to sort by (e.g., "date" or "-date" for descending)
    """
    logger.info("Starting filtered download of daily_market_data collection to CSV")
    
    client = setup_mongodb()
    db = client['algosaham_db']
    collection = db['daily_market_data']
    
    try:
        # Apply filters if provided
        query = filters or {}
        total_documents = collection.count_documents(query)
        logger.info(f"Found {total_documents} documents matching filters: {query}")
        
        # Fetch documents with default sorting by date ascending
        logger.info("Fetching documents from daily_market_data collection...")
        cursor = collection.find(query).sort("date", 1)
        
        # Apply custom sorting if specified (overrides default)
        if sort_by:
            cursor = collection.find(query).sort(sort_by, 1 if not sort_by.startswith('-') else -1)
            logger.info(f"Sorting by: {sort_by}")
        else:
            logger.info("Using default sorting by date ascending")
        
        # Convert cursor to list of dictionaries
        documents = list(cursor)
        logger.info(f"Successfully fetched {len(documents)} documents")
        
        if len(documents) == 0:
            logger.warning("No documents found matching the specified filters")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(documents)
        
        # Remove MongoDB _id column if it exists
        if '_id' in df.columns:
            df = df.drop('_id', axis=1)
        
        # Convert date column to string format for better CSV compatibility
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
        
        # Save to CSV
        df.to_csv(output_file, index=False)
        logger.info(f"Successfully saved {len(df)} records to {output_file}")
        
        # Display some statistics
        logger.info(f"CSV file created: {output_file}")
        logger.info(f"File size: {os.path.getsize(output_file) / 1024:.2f} KB")
        
    except Exception as e:
        logger.error(f"Error downloading filtered data to CSV: {str(e)}")
        raise
    finally:
        client.close()

def get_collection_stats() -> None:
    """Display statistics about the daily_market_data collection."""
    logger.info("Getting collection statistics...")
    
    client = setup_mongodb()
    db = client['algosaham_db']
    collection = db['daily_market_data']
    
    try:
        # Get total count
        total_documents = collection.count_documents({})
        logger.info(f"Total documents: {total_documents}")
        
        if total_documents == 0:
            logger.warning("Collection is empty")
        else:
            # Get unique tickers
            unique_tickers = collection.distinct("ticker")
            logger.info(f"Unique tickers: {len(unique_tickers)}")
            logger.info(f"Sample tickers: {unique_tickers[:10]}")
            
            # Get date range
            date_stats = collection.aggregate([
                {"$group": {
                    "_id": None,
                    "min_date": {"$min": "$date"},
                    "max_date": {"$max": "$date"}
                }}
            ])
            
            date_info = list(date_stats)
            if date_info:
                logger.info(f"Date range: {date_info[0]['min_date']} to {date_info[0]['max_date']}")
            
            # Get sample document structure
            sample_doc = collection.find_one()
            if sample_doc:
                logger.info(f"Document structure: {list(sample_doc.keys())}")
        
    except Exception as e:
        logger.error(f"Error getting collection stats: {str(e)}")
        raise
    finally:
        client.close()

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Download daily_market_data collection to CSV')
    parser.add_argument('--output', '-o', help='Output CSV file path')
    parser.add_argument('--filter', '-f', help='Filter by ticker (e.g., BBCA)')
    parser.add_argument('--sort', '-s', help='Sort by field (e.g., date or -date for descending)')
    parser.add_argument('--stats', action='store_true', help='Show collection statistics only')
    
    args = parser.parse_args()
    
    try:
        if args.stats:
            get_collection_stats()
        elif args.filter:
            filters = {"ticker": args.filter}
            download_with_filters(args.output, filters, args.sort)
        else:
            download_daily_market_data_to_csv(args.output)
            
    except Exception as e:
        logger.error(f"Script failed: {str(e)}")
        exit(1) 