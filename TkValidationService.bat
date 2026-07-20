
call .venv\Scripts\activate.bat 

:LOOP
python TkValidationService.py
if %ERRORLEVEL%==0 goto LOOP
echo "Timeout: "
echo %ERRORLEVEL%
timeout /t %ERRORLEVEL%
goto LOOP