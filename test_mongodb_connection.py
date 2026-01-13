import os
from pymongo import MongoClient
from dotenv import load_dotenv
from mongodb_tunnel import start_ssh_tunnel
import logging

# Load environment variables
load_dotenv()

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)

logger = setup_logging()

def test_mongodb_connection():
    # Start SSH tunnel if configured
    start_ssh_tunnel()
    
    mongodb_uri = os.getenv('MONGODB_URI')
    logger.info(f"Attempting to connect to MongoDB at: {mongodb_uri}")
    
    try:
        # Try to connect with a shorter timeout
        client = MongoClient(mongodb_uri, serverSelectionTimeoutMS=5000)
        
        # Force a connection to verify it works
        client.admin.command('ping')
        logger.info("Successfully connected to MongoDB!")
        
        # List all databases
        databases = client.list_database_names()
        logger.info(f"Available databases: {databases}")
        
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {str(e)}")
        logger.error("Please check if:")
        logger.error("1. MongoDB service is running")
        logger.error("2. The IP address and port are correct")
        logger.error("3. Firewall is not blocking the connection")
        logger.error("4. MongoDB is configured to accept remote connections")
    finally:
        if 'client' in locals():
            client.close()

if __name__ == '__main__':
    test_mongodb_connection() 