import os
import sys
import time
import logging
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

import pandas as pd
import yfinance as yf
from curl_cffi import requests
from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne
from mongodb_tunnel import start_ssh_tunnel
import pytz

# Load environment variables
load_dotenv(override=True)

def setup_logging():
    """Configure logging with timestamp, level, and message."""
    log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
    log_level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    level = log_level_map.get(log_level, logging.INFO)
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)

logger = setup_logging()

def get_proxy_config():
    """Get proxy configuration from environment variables."""
    use_proxy = os.getenv('USE_PROXY', 'TRUE').upper() == 'TRUE'
    if not use_proxy:
        return None

    proxy_host = os.getenv('PROXY_HOST')
    proxy_port = os.getenv('PROXY_PORT')
    proxy_username = os.getenv('PROXY_USERNAME')
    proxy_password = os.getenv('PROXY_PASSWORD')
    
    if not proxy_host or not proxy_port:
        return None
    
    if proxy_username and proxy_password:
        proxy_url = f"http://{proxy_username}:{proxy_password}@{proxy_host}:{proxy_port}"
    else:
        proxy_url = f"http://{proxy_host}:{proxy_port}"
    
    return {
        'http': proxy_url,
        'https': proxy_url
    }

def setup_mongodb() -> MongoClient:
    """Initialize and return a MongoDB client."""
    start_ssh_tunnel()
    mongodb_uri = os.getenv('MONGODB_URI')
    if not mongodb_uri:
        raise ValueError("MONGODB_URI not found in environment variables")
    return MongoClient(mongodb_uri)

def get_target_timezone():
    return pytz.timezone('Asia/Jakarta')

def get_stock_list() -> List[str]:
    """Get list of stock codes from MongoDB 'tickers' collection."""
    logger.info("Loading stock list from MongoDB 'tickers' collection...")
    try:
        client = setup_mongodb()
        db_name = os.getenv('MONGODB_DATABASE', 'sahamify_db')
        db = client[db_name]
        collection = db['tickers']
        query = {"is_active": True}
        tickers = collection.find(query, {"ticker": 1, "_id": 0})
        stock_codes = [doc["ticker"] for doc in tickers]
        client.close()
        
        formatted_codes = []
        for code in stock_codes:
            if code == "JKSE":
                formatted_codes.append("^JKSE")
            else:
                formatted_codes.append(f"{code}.JK")
        return formatted_codes
    except Exception as e:
        logger.error(f"Failed to load stock list: {str(e)}")
        raise

def safe_get(data: Dict, *keys: str, default: Any = None) -> Any:
    """Safely get nested keys from a dictionary."""
    for key in keys:
        if isinstance(data, dict):
            data = data.get(key)
        else:
            return default
    return data if data is not None else default

def safe_float(value: Any) -> Optional[float]:
    try:
        if value is None or (isinstance(value, float) and (pd.isna(value) or value != value)):
            return None
        return float(value)
    except:
        return None

def calculate_growth(series: pd.Series) -> Optional[float]:
    """Calculate YoY growth from a series (usually financials row)."""
    if series is None or len(series) < 2:
        return None
    try:
        current = series.iloc[0]
        previous = series.iloc[1]
        if previous and previous != 0:
            return ((current - previous) / abs(previous)) * 100
    except:
        pass
    return None

def fetch_ticker_data(symbol: str) -> Optional[Dict[str, Any]]:
    """Fetch and process fundamental data for a single ticker."""
    logger.info(f"Fetching fundamental data for {symbol}")
    proxy_config = get_proxy_config()
    session = requests.Session(impersonate="chrome", proxies=proxy_config) if proxy_config else requests.Session(impersonate="chrome")
    
    try:
        ticker = yf.Ticker(symbol, session=session)
        info = ticker.info
        
        # Financials
        financials = ticker.financials
        balance_sheet = ticker.balance_sheet
        cashflow = ticker.cashflow
        
        # Helper to get first value from index
        def get_fin_val(df, label, idx=0):
            try:
                if df is not None and label in df.index:
                    val = df.loc[label].iloc[idx]
                    return safe_float(val)
            except:
                pass
            return None

        def get_fin_series(df, label):
            try:
                if df is not None and label in df.index:
                    return df.loc[label]
            except:
                pass
            return None

        # Efficiency & Health Calculations
        revenue = get_fin_val(financials, 'Total Revenue')
        ebit = get_fin_val(financials, 'EBIT')
        inventory = get_fin_val(balance_sheet, 'Inventory')
        receivables = get_fin_val(balance_sheet, 'Accounts Receivable')
        payables = get_fin_val(balance_sheet, 'Accounts Payable')
        cost_of_revenue = get_fin_val(financials, 'Cost Of Revenue')
        total_assets = get_fin_val(balance_sheet, 'Total Assets')
        total_equity = get_fin_val(balance_sheet, 'Stockholders Equity') or get_fin_val(balance_sheet, 'Total Equity Gross Minority Interest')
        
        days_sales_outstanding = (receivables / revenue * 365) if receivables and revenue else None
        days_inventory = (inventory / cost_of_revenue * 365) if inventory and cost_of_revenue else None
        days_payables_outstanding = (payables / cost_of_revenue * 365) if payables and cost_of_revenue else None
        cash_conversion_cycle = (days_sales_outstanding + days_inventory - days_payables_outstanding) if days_sales_outstanding and days_inventory and days_payables_outstanding else None
        receivables_turnover = (revenue / receivables) if revenue and receivables else None
        inventory_turnover = (cost_of_revenue / inventory) if cost_of_revenue and inventory else None
        
        interest_expense = get_fin_val(financials, 'Interest Expense')
        interest_coverage = (ebit / abs(interest_expense)) if ebit and interest_expense and interest_expense != 0 else None
        
        current_liabilities = get_fin_val(balance_sheet, 'Total Current Liabilities') or get_fin_val(balance_sheet, 'Current Liabilities')
        total_liabilities = get_fin_val(balance_sheet, 'Total Liabilities Net Minority Interest') or get_fin_val(balance_sheet, 'Total Liabilities')
        
        total_debt_to_assets = (safe_float(info.get('totalDebt')) / total_assets) if info.get('totalDebt') and total_assets else None

        # Altman Z-Score (Manufacturing/General)
        working_capital = get_fin_val(balance_sheet, 'Working Capital')
        retained_earnings = get_fin_val(balance_sheet, 'Retained Earnings')
        mkt_cap = safe_float(info.get("marketCap"))
        
        altman_z = None
        if all(v is not None for v in [working_capital, total_assets, retained_earnings, ebit, mkt_cap, total_liabilities, revenue]):
            A = working_capital / total_assets
            B = retained_earnings / total_assets
            C = ebit / total_assets
            D = mkt_cap / total_liabilities
            E = revenue / total_assets
            altman_z = (1.2 * A) + (1.4 * B) + (3.3 * C) + (0.6 * D) + (1.0 * E)

        # ROCE & ROIC
        roce = (ebit / (total_assets - current_liabilities)) * 100 if ebit and total_assets and current_liabilities and (total_assets - current_liabilities) != 0 else None
        
        # Simple ROIC: (EBIT * (1 - TaxRate)) / Invested Capital
        invested_capital = get_fin_val(balance_sheet, 'Invested Capital')
        roic = None
        if ebit and invested_capital and invested_capital != 0:
            roic = (ebit * 0.78 / invested_capital) * 100 # Assuming 22% tax (PPh Badan Indonesia)
        elif safe_float(info.get("returnOnCapital")):
            roic = safe_float(info.get("returnOnCapital")) * 100

        # Operating Cash Flow Label Helper
        ocf = get_fin_val(cashflow, 'Operating Cash Flow') or get_fin_val(cashflow, 'Cash Flowsfromusedin Operating Activities Direct') or get_fin_val(cashflow, 'Net Cash From Operating Activities')
        
        # Period Returns
        history = ticker.history(period="2y")
        returns = {}
        if not history.empty:
            latest_close = history['Close'].iloc[-1]
            now_dt = history.index[-1]
            
            def calculate_period_return(days):
                target_dt = now_dt - timedelta(days=days)
                try:
                    # Use nearest available trading day
                    idx = history.index.get_indexer([target_dt], method='pad')[0]
                    if idx >= 0:
                        old_close = history['Close'].iloc[idx]
                        return ((latest_close - old_close) / old_close) * 100
                except:
                    pass
                return None

            returns = {
                "oneWeekReturn": calculate_period_return(7),
                "oneMonthReturn": calculate_period_return(30),
                "threeMonthReturn": calculate_period_return(90),
                "sixMonthReturn": calculate_period_return(180),
                "oneYearReturn": calculate_period_return(365),
            }
            
            # YTD
            start_of_year = pd.Timestamp(datetime(now_dt.year, 1, 1), tz=now_dt.tz)
            try:
                ytd_idx = history.index.get_indexer([start_of_year], method='pad')[0]
                if ytd_idx >= 0:
                    ytd_close = history['Close'].iloc[ytd_idx]
                    returns["yearToDateReturn"] = ((latest_close - ytd_close) / ytd_close) * 100
            except:
                returns["yearToDateReturn"] = None

            returns["fiftyTwoWeekHigh"] = safe_float(history['High'].max())
            returns["fiftyTwoWeekLow"] = safe_float(history['Low'].min())

        # Final Data Structure
        clean_ticker = symbol.replace(".JK", "")
        data = {
            "ticker": clean_ticker,
            "stats": {
                "marketCap": safe_float(info.get("marketCap")),
                "currentShareOutstanding": safe_float(info.get("sharesOutstanding")),
                "enterpriseValue": safe_float(info.get("enterpriseValue")),
                "freeFloatPercent": safe_float(info.get("floatShares")) / safe_float(info.get("sharesOutstanding")) * 100 if info.get("floatShares") and info.get("sharesOutstanding") else None
            },
            "ratios": {
                "valuation": {
                    "peRatioTTM": safe_float(info.get("peRatioTTM") or info.get("trailingPE")),
                    "peRatioAnnualized": safe_float(info.get("forwardPE")),
                    "priceToBook": safe_float(info.get("priceToBook")),
                    "priceToSales": safe_float(info.get("priceToSalesTrailing12Months")),
                    "earningsYield": (1 / info.get("trailingPE") * 100) if info.get("trailingPE") else None,
                    "pegRatio": safe_float(info.get("trailingPegRatio") or info.get("pegRatio")) or ((safe_float(info.get("trailingPE")) / calculate_growth(get_fin_series(financials, 'Net Income'))) if info.get("trailingPE") and calculate_growth(get_fin_series(financials, 'Net Income')) else None),
                    "evToEbit": (safe_float(info.get("enterpriseValue")) / ebit) if info.get("enterpriseValue") and ebit else None,
                    "evToEbitda": safe_float(info.get("enterpriseToEbitda")),
                    "priceToCashflow": (mkt_cap / ocf) if mkt_cap and ocf else safe_float(info.get("priceToCashFlow")),
                    "priceToFreeCashflow": (mkt_cap / get_fin_val(cashflow, 'Free Cash Flow')) if mkt_cap and get_fin_val(cashflow, 'Free Cash Flow') else None
                },
                "growth": {
                    "revenueGrowthYoY": calculate_growth(get_fin_series(financials, 'Total Revenue')),
                    "grossProfitGrowthYoY": calculate_growth(get_fin_series(financials, 'Gross Profit')),
                    "netIncomeGrowthYoY": calculate_growth(get_fin_series(financials, 'Net Income'))
                },
                "profitability": {
                    "netProfitMargin": safe_float(info.get("profitMargins")) * 100 if info.get("profitMargins") else None,
                    "operatingProfitMargin": safe_float(info.get("operatingMargins")) * 100 if info.get("operatingMargins") else None,
                    "grossProfitMargin": safe_float(info.get("grossMargins")) * 100 if info.get("grossMargins") else None,
                    "returnOnEquity": safe_float(info.get("returnOnEquity")) * 100 if info.get("returnOnEquity") else None,
                    "returnOnAssets": safe_float(info.get("returnOnAssets")) * 100 if info.get("returnOnAssets") else None,
                    "returnOnCapitalEmployed": roce,
                    "returnOnInvestedCapital": roic,
                    "assetTurnover": (revenue / total_assets) if revenue and total_assets else None
                },
                "perShare": {
                    "epsTTM": safe_float(info.get("trailingEps")),
                    "epsAnnualized": safe_float(info.get("forwardEps")),
                    "revenuePerShare": safe_float(info.get("revenuePerShare")),
                    "bookValuePerShare": safe_float(info.get("bookValue")),
                    "cashPerShare": safe_float(info.get("totalCashPerShare")),
                    "freeCashflowPerShare": (get_fin_val(cashflow, 'Free Cash Flow') / info.get("sharesOutstanding")) if get_fin_val(cashflow, 'Free Cash Flow') and info.get("sharesOutstanding") else None
                },
                "financialHealth": {
                    "currentRatio": safe_float(info.get("currentRatio")),
                    "quickRatio": safe_float(info.get("quickRatio")),
                    "debtToEquity": safe_float(info.get("debtToEquity")),
                    "ltDebtToEquity": (get_fin_val(balance_sheet, 'Long Term Debt') / total_equity * 100) if get_fin_val(balance_sheet, 'Long Term Debt') and total_equity else None,
                    "totalLiabilitiesToEquity": (total_liabilities / total_equity) if total_liabilities and total_equity else None,
                    "totalDebtToAssets": total_debt_to_assets,
                    "financialLeverage": (total_assets / total_equity) if total_assets and total_equity else None,
                    "altmanZScore": altman_z,
                    "interestCoverage": interest_coverage,
                    "freeCashflow": get_fin_val(cashflow, 'Free Cash Flow')
                },
                "efficiency": {
                    "daysSalesOutstanding": days_sales_outstanding,
                    "daysInventory": days_inventory,
                    "daysPayablesOutstanding": days_payables_outstanding,
                    "cashConversionCycle": cash_conversion_cycle,
                    "receivablesTurnover": receivables_turnover,
                    "inventoryTurnover": inventory_turnover
                },
                "marketPerformance": returns,
                "marketRank": {
                    "piotroskiFScore": calculate_piotroski_score(financials, balance_sheet, cashflow),
                    "relativeStrengthRating": None,
                    "marketCapRank": None,
                    "peRatioRank": None,
                    "priceToSalesRank": None,
                    "priceToBookRank": None,
                    "near52WeekHighRank": None
                },
                "dividend": {
                    "dividend": safe_float(info.get("dividendRate")),
                    "dividendTTM": safe_float(info.get("trailingAnnualDividendRate")),
                    "payoutRatio": safe_float(info.get("payoutRatio")) * 100 if info.get("payoutRatio") else None,
                    "dividendYield": safe_float(info.get("dividendYield")),
                    "latestDividendExDate": datetime.fromtimestamp(info.get("exDividendDate")).strftime("%d %b %y") if info.get("exDividendDate") else None
                }
            },
            "updatedAt": datetime.now(get_target_timezone())
        }
        
        return data

    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {str(e)}")
        # logger.error(traceback.format_exc())
        return None
    finally:
        session.close()

def calculate_piotroski_score(financials, balance_sheet, cashflow) -> Optional[int]:
    """Calculate Piotroski F-Score (0-9)."""
    if financials is None or balance_sheet is None or cashflow is None or financials.empty or balance_sheet.empty:
        return None
        
    def get_val(df, label, idx=0):
        try:
            val = df.loc[label].iloc[idx]
            return safe_float(val)
        except:
            return None

    # Need at least 2 years of data for most metrics
    if financials.shape[1] < 2 or balance_sheet.shape[1] < 2:
        return None

    score = 0
    
    # helper for current and previous values
    def get_pair(df, label):
        return get_val(df, label, 0), get_val(df, label, 1)

    # 1. ROA (Net Income / Total Assets)
    ni, ni_prev = get_pair(financials, 'Net Income')
    assets, assets_prev = get_pair(balance_sheet, 'Total Assets')
    
    roa = ni / assets if ni is not None and assets else -1
    if roa > 0: score += 1
    
    # 2. Operating Cash Flow
    cfo = get_val(cashflow, 'Operating Cash Flow') or get_val(cashflow, 'Cash Flowsfromusedin Operating Activities Direct')
    if cfo and cfo > 0: score += 1
    
    # 3. Change in ROA
    roa_prev = ni_prev / assets_prev if ni_prev is not None and assets_prev else -1
    if roa > roa_prev: score += 1
    
    # 4. Accruals (CFO > NI)
    if cfo is not None and ni is not None and cfo > ni: score += 1
    
    # 5. Change in Leverage (Long Term Debt / Total Assets) - Lower is better
    ltd = get_val(balance_sheet, 'Long Term Debt') or 0
    ltd_prev = get_val(balance_sheet, 'Long Term Debt', 1) or 0
    lev = ltd / assets if assets else 0
    lev_prev = ltd_prev / assets_prev if assets_prev else 0
    if lev < lev_prev: score += 1
    
    # 6. Change in Current Ratio - Higher is better
    ca = get_val(balance_sheet, 'Total Current Assets') or get_val(balance_sheet, 'Current Assets')
    cl = get_val(balance_sheet, 'Total Current Liabilities') or get_val(balance_sheet, 'Current Liabilities')
    curr_ratio = ca / cl if ca and cl else 0
    
    ca_prev = get_val(balance_sheet, 'Total Current Assets', 1) or get_val(balance_sheet, 'Current Assets', 1)
    cl_prev = get_val(balance_sheet, 'Total Current Liabilities', 1) or get_val(balance_sheet, 'Current Liabilities', 1)
    curr_ratio_prev = ca_prev / cl_prev if ca_prev and cl_prev else 0
    if curr_ratio > curr_ratio_prev: score += 1
    
    # 7. No New Shares (Shares outstanding)
    shares = get_val(balance_sheet, 'Ordinary Shares Number')
    shares_prev = get_val(balance_sheet, 'Ordinary Shares Number', 1)
    if shares and shares_prev and shares <= shares_prev: score += 1
    
    # 8. Change in Gross Margin
    gp, gp_prev = get_pair(financials, 'Gross Profit')
    rev, rev_prev = get_pair(financials, 'Total Revenue')
    gm = gp / rev if gp and rev else 0
    gm_prev = gp_prev / rev_prev if gp_prev and rev_prev else 0
    if gm > gm_prev: score += 1
    
    # 9. Change in Asset Turnover
    at = rev / assets if rev and assets else 0
    at_prev = rev_prev / assets_prev if rev_prev and assets_prev else 0
    if at > at_prev: score += 1
    
    return int(score)

def calculate_market_ranks(collection):
    """Post-processing to calculate rankings across all tickers."""
    logger.info("Calculating market ranks and relative strength...")
    all_data = list(collection.find({}, {
        "ticker": 1,
        "stats.marketCap": 1,
        "ratios.valuation.peRatioTTM": 1,
        "ratios.valuation.priceToSales": 1,
        "ratios.valuation.priceToBook": 1,
        "ratios.marketPerformance.oneYearReturn": 1,
        "ratios.marketPerformance.fiftyTwoWeekHigh": 1,
        "ratios.marketPerformance.fiftyTwoWeekLow": 1
    }))
    
    if not all_data: return

    df = pd.DataFrame([
        {
            "ticker": d["ticker"],
            "marketCap": d.get("stats", {}).get("marketCap"),
            "pe": d.get("ratios", {}).get("valuation", {}).get("peRatioTTM"),
            "ps": d.get("ratios", {}).get("valuation", {}).get("priceToSales"),
            "pb": d.get("ratios", {}).get("valuation", {}).get("priceToBook"),
            "return1y": d.get("ratios", {}).get("marketPerformance", {}).get("oneYearReturn"),
            "high52": d.get("ratios", {}).get("marketPerformance", {}).get("fiftyTwoWeekHigh")
        } for d in all_data
    ])

    # 1. Market Cap Rank (1 is largest)
    df['marketCapRank'] = df['marketCap'].rank(ascending=False, method='min')
    
    # 2. Ratios Ranks (1 is lowest/best depending on metric, usually lower is better for valuation)
    df['peRatioRank'] = df['pe'].rank(ascending=True, method='min')
    df['priceToSalesRank'] = df['ps'].rank(ascending=True, method='min')
    df['priceToBookRank'] = df['pb'].rank(ascending=True, method='min')
    
    # 3. Relative Strength (1-99 percentile)
    # Using 1y return as a proxy for RS
    df['rsRating'] = df['return1y'].rank(pct=True) * 99
    
    # 4. Near 52 Week High Rank
    # Logic: price / high52. Higher is better.
    # Note: We don't have current price here, but 1y return + high52 can be used if we had it. 
    # For now let's just use the rank of the return as a proxy or skip.

    updates = []
    for _, row in df.iterrows():
        updates.append(
            UpdateOne(
                {"ticker": row["ticker"]},
                {"$set": {
                    "ratios.marketRank.marketCapRank": safe_float(row["marketCapRank"]),
                    "ratios.marketRank.peRatioRank": safe_float(row["peRatioRank"]),
                    "ratios.marketRank.priceToSalesRank": safe_float(row["priceToSalesRank"]),
                    "ratios.marketRank.priceToBookRank": safe_float(row["priceToBookRank"]),
                    "ratios.marketRank.relativeStrengthRating": safe_float(row["rsRating"])
                }}
            )
        )
    
    if updates:
        collection.bulk_write(updates)
        logger.info(f"Updated ranks for {len(updates)} tickers")

def main():
    logger.info("Starting fundamental data fetcher")
    
    # Get tickers to process
    cmd_tickers = sys.argv[1:]
    if cmd_tickers:
        stock_list = []
        for code in cmd_tickers:
            code = code.upper().strip()
            if not code.endswith(".JK") and not code.startswith("^"):
                stock_list.append(f"{code}.JK")
            else:
                stock_list.append(code)
    else:
        stock_list = get_stock_list()
    
    logger.info(f"Processing {len(stock_list)} tickers")
    
    client = setup_mongodb()
    db_name = os.getenv('MONGODB_DATABASE', 'sahamify_db')
    db = client[db_name]
    collection = db['yfinance_fundamental_data']
    
    # Create index (Sparse unique index to allow existing documents without 'ticker' field)
    try:
        collection.create_index("ticker", unique=True, sparse=True)
    except Exception as e:
        logger.warning(f"Could not create unique index on 'ticker': {e}")
    
    success_count = 0
    fail_count = 0
    
    for symbol in stock_list:
        try:
            data = fetch_ticker_data(symbol)
            if data:
                # Use update_one to preserve existing fields not in our schema if any, 
                # but replace the main blocks we care about
                collection.update_one(
                    {"ticker": data["ticker"]},
                    {"$set": data},
                    upsert=True
                )
                success_count += 1
                logger.info(f"Successfully processed {data['ticker']}")
            else:
                fail_count += 1
                logger.warning(f"Failed to fetch data for {symbol}")
            
            # Simple rate limiting
            time.sleep(random.uniform(0.5, 1.5))
            
        except Exception as e:
            logger.error(f"Error processing {symbol}: {str(e)}")
            fail_count += 1
            
    # Post-processing for ranks
    if not cmd_tickers: # Only rank if we process the whole market (or a large batch)
        calculate_market_ranks(collection)
        
    client.close()
    logger.info(f"Finished. Success: {success_count}, Failed: {fail_count}")

if __name__ == "__main__":
    main()
