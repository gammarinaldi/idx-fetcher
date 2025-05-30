import os
import logging
import time
import yfinance as yf
import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient
from typing import List, Dict, Any

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

def get_stock_list() -> List[str]:
    """Get list of stock codes from MongoDB ticker collection."""
    client = setup_mongodb()
    db = client['algosaham_db']
    collection = db['ticker']
    
    try:
        # Get all stock codes and add .JK suffix
        stocks = collection.find({}, {'kode': 1, '_id': 0})
        stock_list = [f"{stock['kode']}.JK" for stock in stocks]
        return stock_list
    finally:
        client.close()

def get_fundamental_data(symbol: str) -> Dict[str, Any]:
    """
    Fetch fundamental data for a given stock symbol.
    
    Args:
        symbol: Stock symbol to fetch data for
        
    Returns:
        Dictionary containing fundamental data
    """
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        
        # Extract required fields
        fundamental_data = {
            'symbol': symbol.replace('.JK', ''),
            'long_business_summary': info.get('longBusinessSummary'),
            'sector': info.get('sector'),
            'industry': info.get('industry'),
            'trailing_pe': info.get('trailingPE'),
            'forward_pe': info.get('forwardPE'),
            'price_to_book': info.get('priceToBook'),
            'market_cap': info.get('marketCap'),
            'enterprise_value': info.get('enterpriseValue'),
            'gross_margins': info.get('grossMargins'),
            'operating_margins': info.get('operatingMargins'),
            'profit_margins': info.get('profitMargins'),
            'return_on_equity': info.get('returnOnEquity'),
            'return_on_assets': info.get('returnOnAssets'),
            'debt_to_equity': info.get('debtToEquity'),
            'current_ratio': info.get('currentRatio'),
            'quick_ratio': info.get('quickRatio'),
            'total_debt': info.get('totalDebt'),
            'total_cash': info.get('totalCash'),
            'revenue_growth': info.get('revenueGrowth'),
            'earnings_growth': info.get('earningsGrowth'),
            'free_cashflow': info.get('freeCashflow'),
            'ebitda': info.get('ebitda'),
            'dividend_yield': info.get('dividendYield'),
            'dividend_rate': info.get('dividendRate'),
            'payout_ratio': info.get('payoutRatio'),
            'last_updated': pd.Timestamp.now()
        }
        
        return fundamental_data
    except Exception as e:
        logger.error(f"Error fetching fundamental data for {symbol}: {str(e)}")
        return None

def store_fundamental_data(data: Dict[str, Any]) -> None:
    """
    Store fundamental data in MongoDB.
    
    Args:
        data: Dictionary containing fundamental data
    """
    if not data:
        return
        
    client = setup_mongodb()
    db = client['algosaham_db']
    collection = db['fundamental_data']
    
    try:
        # Update or insert the document
        collection.update_one(
            {'symbol': data['symbol']},
            {'$set': data},
            upsert=True
        )
        logger.info(f"Successfully stored fundamental data for {data['symbol']}")
    except Exception as e:
        logger.error(f"Error storing fundamental data for {data['symbol']}: {str(e)}")
    finally:
        client.close()

def process_stocks() -> None:
    """Process all stocks and store their fundamental data."""
    stock_list = get_stock_list()
    logger.info(f"Found {len(stock_list)} stocks to process")
    
    for symbol in stock_list:
        logger.info(f"Processing {symbol}")
        data = get_fundamental_data(symbol)
        if data:
            store_fundamental_data(data)
        time.sleep(1)  # Add delay to avoid rate limiting

if __name__ == '__main__':
    logger.info("Starting fundamental data fetch process...")
    start_time = time.time()
    
    process_stocks()
    
    elapsed_time = time.time() - start_time
    logger.info(f"Process completed in {elapsed_time:.2f} seconds") 