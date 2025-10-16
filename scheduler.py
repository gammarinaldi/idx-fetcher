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
        
        if result.returncode == 0:
            logger.info("fetch_market_data.py completed successfully")
            if result.stdout:
                logger.info(f"Script output: {result.stdout}")
        else:
            logger.error(f"fetch_market_data.py failed with return code {result.returncode}")
            if result.stderr:
                logger.error(f"Script error: {result.stderr}")
            if result.stdout:
                logger.info(f"Script output: {result.stdout}")
                
    except Exception as e:
        logger.error(f"Error running fetch_market_data.py: {str(e)}")

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