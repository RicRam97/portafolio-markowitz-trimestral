@echo off
title Markowitz Portfolio Server Boot
echo ===================================================
echo   Starting Markowitz Portfolio Analyzer Servers
echo ===================================================

:: Start API Backend
echo [1/3] Starting Python FastAPI Backend...
start cmd /k "cd /d %~dp0 && python api.py"

:: Delay for backend to initialize
timeout /t 2 /nobreak > nul

:: Start Vite Frontend
echo [2/3] Starting Vite Frontend Server...
start cmd /k "cd /d %~dp0frontend && npm run dev"

:: Delay for frontend to initialize
echo [3/3] Waiting for servers to come online before opening browser...
timeout /t 3 /nobreak > nul

:: Open Browser
start http://localhost:5173

echo Done!
exit
