@echo off
echo Starting IDX Fetcher Scheduler...
echo This will run fetch_daily_market_data.py every day at 21:00 UTC+7
echo.
echo Press Ctrl+C to stop the scheduler
echo.
python scheduler.py
pause 