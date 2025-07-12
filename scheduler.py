import schedule
import time
import subprocess
import logging
import os
import sys
from datetime import datetime
import pytz

# Configure logging
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'scheduler.log')

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

def run_fetch_script():
    """Run the fetch_daily_market_data.py script."""
    try:
        logger.info("Starting scheduled execution of fetch_daily_market_data.py")
        
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        fetch_script_path = os.path.join(script_dir, "fetch_daily_market_data.py")
        
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
        
        if result.returncode == 0:
            logger.info("fetch_daily_market_data.py completed successfully")
            if result.stdout:
                logger.info(f"Script output: {result.stdout}")
        else:
            logger.error(f"fetch_daily_market_data.py failed with return code {result.returncode}")
            if result.stderr:
                logger.error(f"Script error: {result.stderr}")
            if result.stdout:
                logger.info(f"Script output: {result.stdout}")
                
    except Exception as e:
        logger.error(f"Error running fetch_daily_market_data.py: {str(e)}")

def get_next_run_time():
    """Get the next scheduled run time in UTC+7."""
    utc7_tz = pytz.timezone('Asia/Jakarta')  # UTC+7 timezone
    now = datetime.now(utc7_tz)
    
    # Schedule for 21:00 UTC+7
    next_run = now.replace(hour=21, minute=0, second=0, microsecond=0)
    
    # If it's already past 21:00 today, schedule for tomorrow
    if now.time() >= next_run.time():
        next_run = next_run.replace(day=next_run.day + 1)
    
    return next_run

def main():
    """Main function to set up and run the scheduler."""
    logger.info("Starting IDX Fetcher Scheduler")
    
    # Set up the schedule to run at 21:00 UTC+7 every day
    schedule.every().day.at("21:00").do(run_fetch_script)
    
    # Log the next scheduled run
    next_run = get_next_run_time()
    logger.info(f"Next scheduled run: {next_run.strftime('%Y-%m-%d %H:%M:%S')} UTC+7")
    
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