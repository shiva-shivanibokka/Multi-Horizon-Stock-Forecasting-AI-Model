@echo off
echo 🎯 Stock Prediction Dashboard Startup
echo ================================================

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python from https://python.org
    pause
    exit /b 1
)

REM Check if Node.js is installed
node --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Node.js is not installed or not in PATH
    echo Please install Node.js from https://nodejs.org
    pause
    exit /b 1
)

REM Check if npm is installed
npm --version >nul 2>&1
if errorlevel 1 (
    echo ❌ npm is not installed or not in PATH
    echo Please install npm with Node.js
    pause
    exit /b 1
)

echo ✅ Python and Node.js are installed

REM Check if backend dependencies are installed
if not exist "transformer_final\requirements.txt" (
    echo ❌ Backend requirements.txt not found
    pause
    exit /b 1
)

REM Check if frontend dependencies are installed
if not exist "frontend\node_modules" (
    echo ⚠️  Frontend dependencies not installed
    echo Installing frontend dependencies...
    cd frontend
    npm install
    cd ..
)

echo ✅ Dependencies check complete

REM Start the Python startup script
echo 🚀 Starting application...
python start_app.py

pause 