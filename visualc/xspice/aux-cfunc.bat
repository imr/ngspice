rem invoke as
rem    .\aux-cfunc.bat analog

REM sub is the parent directory of all of the code models and user types that
REM are being built
set sub=%1



REM Step 1: Process the .lst files for code models and nodes to create
REM source files and related outputs
echo Processing lst files for code model and user-defined nodes ^
at src/xspice/icm/%sub%

echo CMPP_IDIR=../../src/xspice/icm/%sub%
set CMPP_IDIR=../../src/xspice/icm/%sub%

echo CMPP_ODIR=icm/%sub%
set CMPP_ODIR=icm/%sub%
if not exist "icm\%sub%" mkdir "icm\%sub%"
echo running cmpp.exe -lst
bin\cmpp.exe -lst



REM Step 2: For each path in in modpath.lst prepare the directory for a build
REM Test for any code models and process if found
REM ERRORLEVEL = -1 if cannot read modpath.lst file; else = # paths found
echo Processing paths in ..\..\src\xspice\icm\%sub%\modpath.lst
set cmpp_cmd=bin\cmpp.exe -p "..\..\src\xspice\icm\%sub%\modpath.lst"
%cmpp_cmd% > NUL
if %ERRORLEVEL% LSS 0 goto ERROR_EXIT_CM
if %ERRORLEVEL% EQU 0 goto UDN
for /F "tokens=*" %%n in ('%cmpp_cmd%') do (
    echo CMPP_IDIR=../../src/xspice/icm/%sub%/%%n
    set CMPP_IDIR=../../src/xspice/icm/%sub%/%%n

    echo CMPP_ODIR=icm/%sub%/%%n
    set CMPP_ODIR=icm/%sub%/%%n
    if not exist "icm\%sub%\%%n" mkdir "icm\%sub%\%%n"
    echo running cmpp.exe -ifs
    bin\cmpp.exe -ifs
    echo running cmpp.exe -mod
    bin\cmpp.exe -mod
    pushd "icm\%sub%\%%n"
    if exist "%%n-cfunc.c" del "%%n-cfunc.c"
    if exist "%%n-ifspec.c" del "%%n-ifspec.c"
    rename cfunc.c "%%n-cfunc.c"
    rename ifspec.c "%%n-ifspec.c"
    popd
)


:UDN
REM Step 3: For each path in in udnpath.lst prepare the directory for a build
REM Test for any user-defined types and process if found
REM ERRORLEVEL = -1 if cannot read udnpath.lst file; else = # paths found
echo Processing paths in ..\..\src\xspice\icm\%sub%\udnpath.lst
set cmpp_cmd=bin\cmpp.exe -p "..\..\src\xspice\icm\%sub%\udnpath.lst"
%cmpp_cmd% > NUL
if %ERRORLEVEL% LSS 0 goto ERROR_EXIT_UDN
if %ERRORLEVEL% EQU 0 goto EXIT
for /F "tokens=*" %%n in ('%cmpp_cmd%') do (
    if not exist "icm\%sub%\%%n" mkdir "icm\%sub%\%%n"
    copy /Y "..\..\src\xspice\icm\%sub%\%%n\udnfunc.c" ^
            "icm\%sub%\%%n\%%n-udnfunc.c"
)

:EXIT
exit 0

:ERROR_EXIT_MOD
echo "Unable to obtain paths for code models"
exit -1

:ERROR_EXIT_UDN
echo "Unable to obtain paths for user-defined types"
exit -1

