@echo off

REM copy ngspice.exe, codemodels *.cm to C:\Spiced or Spice64d
REM arguments to make-install-vngspiced:
REM %1: path to ngspice.exe, %2, %3: fftw or 64 (64 bit)

if "%2" == "64" goto b64
if "%3" == "64" goto b64

set dst=c:\Spiced
set cmsrc=.\codemodels\Win32\Debug

if not exist "%dst%\bin\" mkdir "%dst%\bin"
if not exist "%dst%\lib\ngspice\" mkdir "%dst%\lib\ngspice"
if not exist "%dst%\share\ngspice\scripts\" mkdir "%dst%\share\ngspice\scripts"

copy "C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\redist\x86\Microsoft.VC140.OPENMP\vcomp140.dll" %dst%\bin\
copy %cmsrc%\analog.cm %dst%\lib\ngspice\analog.cm
copy %cmsrc%\digital.cm %dst%\lib\ngspice\digital.cm
copy %cmsrc%\table.cm %dst%\lib\ngspice\table.cm
copy %cmsrc%\xtraevt.cm %dst%\lib\ngspice\xtraevt.cm
copy %cmsrc%\xtradev.cm %dst%\lib\ngspice\xtradev.cm
copy %cmsrc%\spice2poly.cm %dst%\lib\ngspice\spice2poly.cm
copy .\spinit_all %dst%\share\ngspice\scripts\spinit
copy .\spinitd .\spinit



if "%2" == "fftw" goto copy2
if "%3" == "fftw" goto copy2

copy %1\ngspice.exe %dst%\bin\
goto end

:copy2
copy %1\ngspice.exe %dst%\bin\
copy ..\..\fftw-3.3-dll32\libfftw3-3.dll %dst%\bin\
goto end

:b64

set dst=c:\Spice64d
set cmsrc=.\codemodels\x64\Debug

if not exist "%dst%\bin\" mkdir "%dst%\bin"
if not exist "%dst%\lib\ngspice\" mkdir "%dst%\lib\ngspice"
if not exist "%dst%\share\ngspice\scripts\" mkdir "%dst%\share\ngspice\scripts"
copy "C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\redist\x64\Microsoft.VC140.OPENMP\vcomp140.dll" %dst%\bin\
copy %cmsrc%\analog64.cm %dst%\lib\ngspice\analog.cm
copy %cmsrc%\digital64.cm %dst%\lib\ngspice\digital.cm
copy %cmsrc%\table64.cm %dst%\lib\ngspice\table.cm
copy %cmsrc%\xtraevt64.cm %dst%\lib\ngspice\xtraevt.cm
copy %cmsrc%\xtradev64.cm %dst%\lib\ngspice\xtradev.cm
copy %cmsrc%\spice2poly64.cm %dst%\lib\ngspice\spice2poly.cm
copy .\spinit_all %dst%\share\ngspice\scripts\spinit
copy .\spinitd64 .\spinit

REM ADDED TO ALLOW USE OF CODE MODELS OF THE CURRENT SOLUTION
set ngspice_home=%1
echo Adding code models to the directory of the ngspice program
copy %cmsrc%\analog64.cm %ngspice_home%\analog64.cm
copy %cmsrc%\digital64.cm %ngspice_home%\digital64.cm
copy %cmsrc%\table64.cm %ngspice_home%\table64.cm
copy %cmsrc%\xtraevt64.cm %ngspice_home%\xtraevt64.cm
copy %cmsrc%\xtradev64.cm %ngspice_home%\xtradev64.cm
copy %cmsrc%\spice2poly64.cm %ngspice_home%\spice2poly64.cm
copy spinit_dbg64_src %ngspice_home%\spinit
REM END OF ADDED TO ALLOW USE OF CODE MODELS OF THE CURRENT SOLUTION

if "%2" == "fftw" goto copy2-64
if "%3" == "fftw" goto copy2-64

copy %1\ngspice.exe %dst%\bin\
goto end

:copy2-64
copy %1\ngspice.exe %dst%\bin\
copy ..\..\fftw-3.3-dll64\libfftw3-3.dll %dst%\bin\

:end
