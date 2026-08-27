
call .venv\Scripts\activate.bat 

:LOOP
rem python TkGatherData.py -ipc -mpc
python TkGatherData.py -ipc
if %ERRORLEVEL%==0 goto LOOP
echo "Timeout: "
echo %ERRORLEVEL%
timeout /t %ERRORLEVEL%
goto LOOP