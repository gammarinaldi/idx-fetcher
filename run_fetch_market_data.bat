@echo off
cd /d "%~dp0"
call ".venv\Scripts\activate.bat"
python fetch_market_data.py %*
pause
