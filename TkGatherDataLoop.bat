
call .venv\Scripts\activate.bat 

:LOOP
python TkGatherData.py -ipc
if %ERRORLEVEL%==0 goto LOOP
echo "Timeout: "
echo %ERRORLEVEL%
timeout /t %ERRORLEVEL%
goto LOOP