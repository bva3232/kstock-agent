@echo off
cd /d "%~dp0"
if not exist logs mkdir logs
"C:\Users\BAE\AppData\Local\Programs\Python\Python312\python.exe" run_premarket.py >> logs\premarket.log 2>&1
