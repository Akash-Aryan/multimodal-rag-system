@echo off
echo ============================================
echo     Multimodal RAG API Server
echo ============================================
echo.
echo [INFO] Starting your RAG API server...
echo [INFO] This window will show server logs
echo [INFO] Do NOT close this window while using the API
echo.
echo ============================================
echo     Access your API at these URLs:
echo ============================================
echo   Main API: http://localhost:8000
echo   Documentation: http://localhost:8000/docs  
echo   Health Check: http://localhost:8000/health
echo ============================================
echo.
echo [INFO] Press Ctrl+C to stop the server
echo.

REM Change to the script's directory
cd /d "%~dp0"

REM Run the Python server
python api_server.py

echo.
echo [INFO] Server has stopped.
echo [INFO] Press any key to close this window...
pause > nul
