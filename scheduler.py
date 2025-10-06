import schedule
import time
import subprocess
import logging
import os
import sys
from datetime import datetime
import pytz
import requests

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

def run_fetch_script():
    """Run the fetch_market_data_optimized.py script."""
    try:
        logger.info("Starting scheduled execution of fetch_market_data_optimized.py")
        
        # Check if market is closed (weekend or holiday)
        if is_market_closed():
            logger.info("Market is closed (weekend or holiday) - Skipping market data fetch process.")
            return
        
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        fetch_script_path = os.path.join(script_dir, "fetch_market_data_optimized.py")
        
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
            logger.info("fetch_market_data_optimized.py completed successfully")
            if result.stdout:
                logger.info(f"Script output: {result.stdout}")
        else:
            logger.error(f"fetch_market_data_optimized.py failed with return code {result.returncode}")
            if result.stderr:
                logger.error(f"Script error: {result.stderr}")
            if result.stdout:
                logger.info(f"Script output: {result.stdout}")
                
    except Exception as e:
        logger.error(f"Error running fetch_market_data_optimized.py: {str(e)}")

def get_next_run_time():
    """Get the next scheduled run time in UTC+7."""
    utc7_tz = pytz.timezone('Asia/Jakarta')  # UTC+7 timezone
    now = datetime.now(utc7_tz)
    
    # Scheduled times: 13:00 and 17:00 UTC+7
    scheduled_times = [13, 17]
    
    # Find the next scheduled time today
    for hour in scheduled_times:
        next_run = now.replace(hour=hour, minute=0, second=0, microsecond=0)
        if now.time() < next_run.time():
            return next_run
    
    # If all scheduled times have passed today, schedule for 13:00 tomorrow
    next_run = now.replace(hour=13, minute=0, second=0, microsecond=0)
    next_run = next_run.replace(day=next_run.day + 1)
    
    return next_run

def main():
    """Main function to set up and run the scheduler."""
    logger.info("Starting IDX Fetcher Scheduler")
    
    # Set up the schedule to run at 13:00 and 17:00 UTC+7 every day
    schedule.every().day.at("13:00").do(run_fetch_script)
    schedule.every().day.at("17:00").do(run_fetch_script)
    
    # Log the next scheduled run
    next_run = get_next_run_time()
    logger.info(f"Next scheduled run: {next_run.strftime('%Y-%m-%d %H:%M:%S')} UTC+7")
    logger.info("Scheduled to run daily at 13:00 and 17:00 UTC+7 (Asia/Jakarta timezone)")
    
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