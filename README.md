# IDX Updater

Get historical data based for Indonesia Stock Exchange.
Get list of stocks from [idx.co.id](https://www.idx.co.id/id/data-pasar/data-saham/daftar-saham).

## Features

- Fetch data from Yahoo Finance
- Save data to CSV
- Retry failed fetches
- Use proxy to fetch data

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python3 index.py
```

## Example

You can change period, interval, and group by based on your needs.

- period: data period to download (either use period parameter or use start and end) Valid periods are:
  - “1d”, “5d”, “1mo”, “3mo”, “6mo”, “1y”, “2y”, “5y”, “10y”, “ytd”, “max”
- interval: data interval (1m data is only for available for last 7 days, and data interval <1d for the last 60 days) Valid intervals are:
  - “1m”, “2m”, “5m”, “15m”, “30m”, “60m”, “90m”, “1h”, “1d”, “5d”, “1wk”, “1mo”, “3mo”

Refer to [yfinance](https://pypi.org/project/yfinance/) for more information.

```python
df = yf.download(symbol, period="10y", interval="1d", group_by="ticker", proxy=proxy)
```