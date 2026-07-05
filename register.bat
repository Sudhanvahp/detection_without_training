@echo off
cd /d "%~dp0"
"%~dp0venv\Scripts\python.exe" tools\register_faces.py %*
pause
