import schedule
import time
import subprocess
import logging
import os
import sys
from datetime import datetime
import pytz
import requests
import platform
from pymongo import MongoClient
from mongodb_tunnel import start_ssh_tunnel

# Configure logging
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'scheduler.log')

# Try to create the log file with proper permissions
try:
    # Touch the log file to ensure it exists and has proper permissions
    with open(log_file, 'a') as f:
        pass
except PermissionError:
    # If we can't write to the logs directory, fall back to current directory
    log_file = 'scheduler.log'
    print(f"Warning: Cannot write to logs directory, using {log_file} instead")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

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

def run_fetch_script():
    """Run the fetch_market_data.py script."""
    try:
        logger.info("Starting scheduled execution of fetch_market_data.py")
        
        # Check if market is closed (weekend or holiday)
        if is_market_closed():
            logger.info("Market is closed (weekend or holiday) - Skipping market data fetch process.")
            return
        
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        fetch_script_path = os.path.join(script_dir, "fetch_market_data.py")
        
        # Ensure we're in the correct working directory
        os.chdir(script_dir)
        
        # Check if the script exists
        if not os.path.exists(fetch_script_path):
            logger.error(f"Script not found: {fetch_script_path}")
            return
        
        # Run the script
        result = subprocess.run(
            [sys.executable, fetch_script_path],
            capture_output=True,
            text=True,
            cwd=script_dir
        )
        
        # Always log stdout and stderr for debugging
        if result.stdout:
            # Log each line of stdout for better visibility
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    logger.info(f"Script output: {line}")
        
        if result.stderr:
            # Log each line of stderr
            for line in result.stderr.strip().split('\n'):
                if line.strip():
                    logger.warning(f"Script stderr: {line}")
        
        if result.returncode == 0:
            logger.info("fetch_market_data.py completed successfully (exit code 0)")
        else:
            logger.error(f"fetch_market_data.py failed with return code {result.returncode}")
                
    except Exception as e:
        logger.error(f"Error running fetch_market_data.py: {str(e)}")

def validate_mongodb_connection():
    """
    Validate MongoDB connection and retrieve sample data to ensure DB access is working.
    This function is called at startup to verify MongoDB connectivity.
    
    Returns:
        bool: True if connection is successful, False otherwise
    """
    # Start SSH tunnel if configured
    start_ssh_tunnel()
    
    mongodb_uri = os.getenv('MONGODB_URI')
    mongodb_database = os.getenv('MONGODB_DATABASE')
    mongodb_collection = os.getenv('MONGODB_COLLECTION', 'daily_market_data')
    
    # Only validate if MongoDB upload is enabled
    upload_to_mongodb = os.getenv('UPLOAD_TO_MONGODB', 'FALSE').upper() == 'TRUE'
    if not upload_to_mongodb:
        logger.info("MongoDB upload is disabled - skipping MongoDB validation")
        return True
    
    if not mongodb_uri:
        logger.warning("MONGODB_URI not found in environment variables - skipping MongoDB validation")
        return True
    
    logger.info("Validating MongoDB connection...")
    logger.info(f"MongoDB URI: {mongodb_uri.split('@')[-1] if '@' in mongodb_uri else '***'}")  # Hide credentials
    logger.info(f"Database: {mongodb_database}")
    logger.info(f"Collection: {mongodb_collection}")
    
    client = None
    try:
        # Attempt to connect with timeout
        client = MongoClient(mongodb_uri, serverSelectionTimeoutMS=10000)
        
        # Test connection with ping
        client.admin.command('ping')
        logger.info("✓ MongoDB connection successful!")
        
        # Get database name
        if not mongodb_database:
            # Extract database name from URI if not provided
            db_name = mongodb_uri.split('/')[-1].split('?')[0] if mongodb_uri else 'sahamify_db'
            logger.info(f"Using database from URI: {db_name}")
        else:
            db_name = mongodb_database
        
        # Access database and collection
        db = client[db_name]
        collection = db[mongodb_collection]
        
        # Get database stats
        db_stats = db.command('dbStats')
        logger.info(f"Database '{db_name}' stats:")
        logger.info(f"  - Collections: {db_stats.get('collections', 0)}")
        logger.info(f"  - Data size: {db_stats.get('dataSize', 0) / 1024 / 1024:.2f} MB")
        logger.info(f"  - Storage size: {db_stats.get('storageSize', 0) / 1024 / 1024:.2f} MB")
        
        # Check if collection exists and get stats
        collection_names = db.list_collection_names()
        if mongodb_collection in collection_names:
            try:
                collection_stats = db.command('collStats', mongodb_collection)
                document_count = collection_stats.get('count', 0)
                logger.info(f"Collection '{mongodb_collection}' stats:")
                logger.info(f"  - Document count: {document_count}")
                logger.info(f"  - Size: {collection_stats.get('size', 0) / 1024 / 1024:.2f} MB")
                
                # Retrieve sample data (up to 3 documents)
                if document_count > 0:
                    logger.info("Retrieving sample data from collection...")
                    sample_docs = list(collection.find().limit(3))
                    
                    if sample_docs:
                        logger.info(f"✓ Successfully retrieved {len(sample_docs)} sample document(s):")
                        for i, doc in enumerate(sample_docs, 1):
                            # Remove _id for cleaner output, show key fields
                            sample = {k: v for k, v in doc.items() if k != '_id'}
                            # Truncate long values for readability
                            sample_str = {}
                            for key, value in sample.items():
                                if isinstance(value, str) and len(value) > 50:
                                    sample_str[key] = value[:50] + "..."
                                else:
                                    sample_str[key] = value
                            logger.info(f"  Sample {i}: {sample_str}")
                    else:
                        logger.warning("Collection exists but no documents found")
                else:
                    logger.info("Collection is empty (no documents yet)")
                
                # Test read access by checking if we can list indexes
                indexes = collection.list_indexes()
                index_list = list(indexes)
                logger.info(f"✓ Database access verified - {len(index_list)} index(es) found")
            except Exception as coll_error:
                logger.warning(f"Could not get collection stats: {str(coll_error)}")
                logger.info("Collection exists but stats unavailable - this is okay")
        else:
            logger.info(f"Collection '{mongodb_collection}' does not exist yet (will be created on first write)")
            logger.info("✓ Database access verified - ready to create collection when needed")
        
        logger.info("✓ MongoDB validation completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"✗ MongoDB validation failed: {str(e)}")
        logger.error("Please check:")
        logger.error("  1. MongoDB service is running and accessible")
        logger.error("  2. MONGODB_URI is correct")
        logger.error("  3. Network connectivity to MongoDB server")
        logger.error("  4. MongoDB authentication credentials (if required)")
        return False
        
    finally:
        if client:
            client.close()

def get_next_run_time():
    """Get the next scheduled run time in UTC+7."""
    jakarta_tz = get_target_timezone()
    now = datetime.now(jakarta_tz)
    
    # Scheduled times: 12:01 and 17:01 UTC+7
    scheduled_times = [12, 17]
    
    # Find the next scheduled time today
    for hour in scheduled_times:
        next_run = now.replace(hour=hour, minute=1, second=0, microsecond=0)
        if now.time() < next_run.time():
            return next_run
    
    # If all scheduled times have passed today, schedule for 12:01 tomorrow
    next_run = now.replace(hour=12, minute=1, second=0, microsecond=0)
    next_run = next_run.replace(day=next_run.day + 1)
    
    return next_run

def main():
    """Main function to set up and run the scheduler."""
    logger.info("Starting IDX Fetcher Scheduler")
    
    # Validate MongoDB connection at startup
    if not validate_mongodb_connection():
        logger.error("MongoDB validation failed. Scheduler will continue but MongoDB operations may fail.")
        logger.error("Please fix MongoDB connection issues before the next scheduled run.")
    
    # Detect system timezone
    is_utc, timezone_name, utc_offset = get_system_timezone_info()
    
    # Set up the schedule to run at 12:01 and 17:01 UTC+7 every day
    jakarta_tz = get_target_timezone()
    
    if is_utc:
        # System is in UTC+0, use UTC times directly
        schedule_time_1201 = "05:01"  # 12:01 Jakarta = 05:01 UTC
        schedule_time_1701 = "10:01"  # 17:01 Jakarta = 10:01 UTC
        logger.info("System detected as UTC+0 - Using UTC-based scheduling")
    else:
        # System is not UTC, convert Jakarta time to local time for scheduling
        # This is a fallback for systems that aren't UTC+0
        schedule_time_1201 = "12:01"  # Will be interpreted as local time
        schedule_time_1701 = "17:01"  # Will be interpreted as local time
        logger.info(f"System detected as {timezone_name} (UTC{utc_offset:+.1f}) - Using local time scheduling")
    
    schedule.every().day.at(schedule_time_1201).do(run_fetch_script)
    schedule.every().day.at(schedule_time_1701).do(run_fetch_script)
    
    # Log the next scheduled run
    next_run = get_next_run_time()
    logger.info(f"Next scheduled run: {next_run.strftime('%Y-%m-%d %H:%M:%S')} UTC+7")
    
    if is_utc:
        logger.info(f"Scheduled to run daily at 12:01 and 17:01 WIB (UTC+7) - Server times: {schedule_time_1201} and {schedule_time_1701} UTC")
    else:
        logger.info(f"Scheduled to run daily at {schedule_time_1201} and {schedule_time_1701} local time ({timezone_name})")
    
    logger.info("Scheduler is running. Press Ctrl+C to stop.")
    
    try:
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
            
    except KeyboardInterrupt:
        logger.info("Scheduler stopped by user")
    except Exception as e:
        logger.error(f"Scheduler error: {str(e)}")

if __name__ == "__main__":
    main() 