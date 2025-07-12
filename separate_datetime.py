import pandas as pd
import os
import argparse
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

def separate_datetime(csv_path: str, output_path: str = None, 
                     datetime_column: str = 'Datetime',
                     keep_original: bool = False) -> None:
    """
    Separate Datetime column into Date and Time columns.
    
    Args:
        csv_path: Path to the input CSV file
        output_path: Path for the output CSV file (optional)
        datetime_column: Name of the datetime column (default: 'Datetime')
        keep_original: Whether to keep the original datetime column (default: False)
    """
    if output_path is None:
        # Create output filename with _separated suffix
        base_name = os.path.splitext(csv_path)[0]
        extension = os.path.splitext(csv_path)[1]
        output_path = f"{base_name}_separated{extension}"
    
    logger.info(f"Reading CSV file: {csv_path}")
    
    try:
        # Read the CSV file
        df = pd.read_csv(csv_path)
        
        logger.info(f"Original data shape: {df.shape}")
        logger.info(f"Columns: {df.columns.tolist()}")
        
        # Check if datetime column exists
        if datetime_column not in df.columns:
            logger.error(f"Column '{datetime_column}' not found in CSV file")
            logger.info(f"Available columns: {df.columns.tolist()}")
            return
        
        # Convert datetime column to pandas datetime
        logger.info(f"Converting {datetime_column} column to datetime...")
        df[datetime_column] = pd.to_datetime(df[datetime_column])
        
        # Extract Date and Time components
        logger.info("Extracting Date and Time components...")
        
        # Extract date (YYYY-MM-DD format)
        df['Date'] = df[datetime_column].dt.date
        
        # Extract time (HH:MM:SS format)
        df['Time'] = df[datetime_column].dt.time
        
        # Extract timezone info if present
        if df[datetime_column].dt.tz is not None:
            df['Timezone'] = df[datetime_column].dt.tz.zone
            logger.info(f"Timezone detected: {df['Timezone'].iloc[0]}")
        
        # Reorder columns to put Date and Time after Datetime
        columns = df.columns.tolist()
        datetime_idx = columns.index(datetime_column)
        
        # Remove Date and Time from their current positions
        columns.remove('Date')
        columns.remove('Time')
        
        # Insert Date and Time after Datetime
        if 'Timezone' in columns:
            columns.remove('Timezone')
            new_columns = columns[:datetime_idx+1] + ['Date', 'Time', 'Timezone'] + columns[datetime_idx+1:]
        else:
            new_columns = columns[:datetime_idx+1] + ['Date', 'Time'] + columns[datetime_idx+1:]
        
        df = df[new_columns]
        
        # Remove original datetime column if not keeping it
        if not keep_original:
            df = df.drop(columns=[datetime_column])
            logger.info(f"Removed original {datetime_column} column")
        else:
            logger.info(f"Kept original {datetime_column} column")
        
        # Show sample of separated data
        logger.info("Sample of separated data:")
        sample_data = df.head(10)
        for i, row in sample_data.iterrows():
            if 'Timezone' in df.columns:
                logger.info(f"  {i+1}: Date={row['Date']}, Time={row['Time']}, TZ={row['Timezone']}")
            else:
                logger.info(f"  {i+1}: Date={row['Date']}, Time={row['Time']}")
        
        # Save the separated data
        logger.info(f"Saving separated data to: {output_path}")
        df.to_csv(output_path, index=False)
        
        logger.info(f"Separation completed successfully!")
        logger.info(f"Original file: {csv_path}")
        logger.info(f"Separated file: {output_path}")
        logger.info(f"Final columns: {df.columns.tolist()}")
        
        # Show file size comparison
        original_size = os.path.getsize(csv_path) / (1024 * 1024)  # MB
        separated_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        logger.info(f"Original file size: {original_size:.2f} MB")
        logger.info(f"Separated file size: {separated_size:.2f} MB")
        
    except FileNotFoundError:
        logger.error(f"File not found: {csv_path}")
    except Exception as e:
        logger.error(f"Error during separation: {str(e)}")
        raise

def main():
    """Main function to run the datetime separation."""
    parser = argparse.ArgumentParser(description='Separate Datetime column into Date and Time columns')
    parser.add_argument('input_file', nargs='?', default='results_utc7.csv', 
                       help='Input CSV file (default: results_utc7.csv)')
    parser.add_argument('-o', '--output', help='Output CSV file path')
    parser.add_argument('--datetime-column', default='Datetime',
                       help='Name of datetime column (default: Datetime)')
    parser.add_argument('--keep-original', action='store_true',
                       help='Keep the original datetime column')
    
    args = parser.parse_args()
    
    logger.info("Starting datetime separation...")
    
    # Check if input file exists
    if not os.path.exists(args.input_file):
        logger.error(f"File {args.input_file} not found in current directory")
        return
    
    # Separate the datetime
    separate_datetime(
        csv_path=args.input_file,
        output_path=args.output,
        datetime_column=args.datetime_column,
        keep_original=args.keep_original
    )
    
    logger.info("Separation process completed!")

if __name__ == "__main__":
    main() 