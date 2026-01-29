import yfinance as yf
from curl_cffi import requests

session = requests.Session(impersonate="chrome")
ticker = yf.Ticker("ASII.JK", session=session)
print(f"Dividend Yield: {ticker.info.get('dividendYield')}")
print(f"Trailing PE: {ticker.info.get('trailingPE')}")
session.close()
