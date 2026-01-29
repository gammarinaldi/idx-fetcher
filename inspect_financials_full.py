import yfinance as yf
from curl_cffi import requests

session = requests.Session(impersonate="chrome")
ticker = yf.Ticker("ASII.JK", session=session)
print("--- FINANCIALS ---")
for x in ticker.financials.index.tolist(): print(x)
print("\n--- BALANCE SHEET ---")
for x in ticker.balance_sheet.index.tolist(): print(x)
print("\n--- CASHFLOW ---")
for x in ticker.cashflow.index.tolist(): print(x)
session.close()
