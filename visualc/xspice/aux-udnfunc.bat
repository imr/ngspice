rem invoke as
rem    .\aux-udnfunc.bat xtraevt
REM
REM set sub=%1
REM
REM for /F %%n in (..\..\src\xspice\icm\%sub%\udnpath.lst) do (
REM echo Processing paths in ..\..\src\xspice\icm\%sub%\udnpath.lst
REM for /F "tokens=*" %%n in ('bin\cmpp.exe ^
REM        -p "..\..\src\xspice\icm\%sub%\udnpath.lst"') do (
REM    if not exist "icm\%sub%\%%n" mkdir "icm\%sub%\%%n"
REM    copy /Y "..\..\src\xspice\icm\%sub%\%%n\udnfunc.c" ^
REM            "icm\%sub%\%%n\%%n-udnfunc.c"
REM )
