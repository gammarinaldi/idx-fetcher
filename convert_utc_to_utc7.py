import pandas as pd
import os
from datetime import datetime
import logging

def setup_logging():
    """Configure logging with timestamp, level, and message."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)

logger = setup_logging()

def convert_utc_to_utc7(csv_path: str, output_path: str = None) -> None:
    """
    Convert UTC timestamps to UTC+7 in the CSV file.
    
    Args:
        csv_path: Path to the input CSV file
        output_path: Path for the output CSV file (optional, defaults to input_path with _utc7 suffix)
    """
    if output_path is None:
        # Create output filename with _utc7 suffix
        base_name = os.path.splitext(csv_path)[0]
        extension = os.path.splitext(csv_path)[1]
        output_path = f"{base_name}_utc7{extension}"
    
    logger.info(f"Reading CSV file: {csv_path}")
    
    try:
        # Read the CSV file
        df = pd.read_csv(csv_path)
        
        logger.info(f"Original data shape: {df.shape}")
        logger.info(f"Columns: {df.columns.tolist()}")
        
        # Check if Datetime column exists
        if 'Datetime' not in df.columns:
            logger.error("Datetime column not found in CSV file")
            return
        
        # Convert Datetime column to datetime with UTC timezone
        logger.info("Converting Datetime column to UTC+7...")
        df['Datetime'] = pd.to_datetime(df['Datetime'], utc=True)
        
        # Convert UTC to UTC+7 (Indonesia timezone)
        df['Datetime'] = df['Datetime'].dt.tz_convert('Asia/Jakarta')
        
        # Show sample of converted data
        logger.info("Sample of converted timestamps:")
        sample_times = df['Datetime'].head(10)
        for i, time in enumerate(sample_times):
            logger.info(f"  {i+1}: {time}")
        
        # Save the converted data
        logger.info(f"Saving converted data to: {output_path}")
        df.to_csv(output_path, index=False)
        
        logger.info(f"Conversion completed successfully!")
        logger.info(f"Original file: {csv_path}")
        logger.info(f"Converted file: {output_path}")
        
        # Show file size comparison
        original_size = os.path.getsize(csv_path) / (1024 * 1024)  # MB
        converted_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        logger.info(f"Original file size: {original_size:.2f} MB")
        logger.info(f"Converted file size: {converted_size:.2f} MB")
        
    except FileNotFoundError:
        logger.error(f"File not found: {csv_path}")
    except Exception as e:
        logger.error(f"Error during conversion: {str(e)}")
        raise

def main():
    """Main function to run the UTC to UTC+7 conversion."""
    logger.info("Starting UTC to UTC+7 conversion...")
    
    # Check if results.csv exists
    csv_path = "results.csv"
    if not os.path.exists(csv_path):
        logger.error(f"File {csv_path} not found in current directory")
        return
    
    # Convert the file
    convert_utc_to_utc7(csv_path)
    
    logger.info("Conversion process completed!")

if __name__ == "__main__":
    main() 