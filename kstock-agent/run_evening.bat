@echo off
cd /d "%~dp0"
if not exist logs mkdir logs
"C:\Users\BAE\AppData\Local\Programs\Python\Python312\python.exe" run_evening.py >> logs\evening.log 2>&1
