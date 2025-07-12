# IDX Fetcher

Get historical OHLCV data from Indonesia Stock Exchange.
Stocks list source: [idx.co.id](https://www.idx.co.id/id/data-pasar/data-saham/daftar-saham).

## Features

- Fetch OHLCV data from Yahoo Finance
- Save data to CSV
- Retry failed fetches
- Use proxy to fetch data
- Enable/disable proxy usage
- Automated daily scheduling at 21:00 UTC+7
- Indonesian holiday and weekend detection (skips execution when market is closed)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Manual Execution

```bash
python fetch_daily_market_data.py
```

### Automated Scheduling

To run the script automatically every day at 21:00 UTC+7:

#### Option 1: Using Python directly
```bash
python scheduler.py
```

#### Option 2: Using the batch file (Windows)
```bash
start_scheduler.bat
```

#### Option 3: Using Docker (Recommended for VPS)
```bash
# Quick deployment
./deploy.sh

# Or manually:
docker-compose up -d
```

The scheduler will:
- Run `fetch_daily_market_data.py` every day at 21:00 UTC+7
- Log all activities to `scheduler.log`
- Display real-time status in the console
- Handle errors gracefully and continue scheduling

To stop the scheduler, press `Ctrl+C` (for local) or use `docker-compose down` (for Docker).

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

## Docker Deployment (VPS)

### Prerequisites
- Docker and Docker Compose installed on your VPS
- Stock list CSV file (`stock_list.csv`) with a 'Kode' column

### Quick Deployment
1. Upload your project files to the VPS
2. Ensure you have `stock_list.csv` in the project directory
3. Run the deployment script:
   ```bash
   ./deploy.sh
   ```

### Manual Deployment
1. Create environment file:
   ```bash
   cp env.example .env
   # Edit .env with your configuration
   ```

2. Create directories:
   ```bash
   mkdir -p data logs
   ```

3. Build and start:
   ```bash
   docker-compose up -d
   ```

### Docker Commands
```bash
# View logs
docker-compose logs -f

# Stop service
docker-compose down

# Restart service
docker-compose restart

# View status
docker-compose ps

# Update and restart
docker-compose down
docker-compose build
docker-compose up -d
```

### Troubleshooting

#### Permission Issues
If you encounter permission errors, run the fix script:
```bash
chmod +x fix_permissions.sh
./fix_permissions.sh
```

Or manually fix permissions:
```bash
# Stop the container
docker-compose down

# Fix permissions
sudo chown -R 1000:1000 data logs
chmod 755 data logs

# Rebuild and restart
docker-compose build --no-cache
docker-compose up -d
```

### Data Persistence
- CSV files and results: `./data/`
- Scheduler logs: `./logs/`
- Stock list: `./stock_list.csv` (mounted as read-only)

### Environment Variables
Copy `env.example` to `.env` and configure:
- `MONGODB_URI`: Your MongoDB connection string
- `UPLOAD_TO_MONGODB`: Set to `TRUE` to enable MongoDB upload
- `ENABLE_MARKET_CLOSED_CHECK`: Set to `TRUE` to skip execution on weekends and Indonesian holidays (default: TRUE)
- Other variables as needed
