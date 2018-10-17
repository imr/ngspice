@echo off

REM copy ngspice.exe, codemodels *.cm to C:\Spice or Spice64
REM arguments to make-install-vngspiced:
REM %1: path to ngspice.exe, %2, %3: fftw or 64 (64 bit)

if "%2" == "64" goto b64
if "%3" == "64" goto b64

set dst=c:\Spice
set cmsrc=.\codemodels\Win32\Release

mkdir %dst%\bin
mkdir %dst%\lib\ngspice
mkdir %dst%\share\ngspice\scripts

copy "C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\redist\x86\Microsoft.VC140.OPENMP\vcomp140.dll" %dst%\bin\
copy %cmsrc%\analog.cm %dst%\lib\ngspice\analog.cm
copy %cmsrc%\digital.cm %dst%\lib\ngspice\digital.cm
copy %cmsrc%\table.cm %dst%\lib\ngspice\table.cm
copy %cmsrc%\xtraevt.cm %dst%\lib\ngspice\xtraevt.cm
copy %cmsrc%\xtradev.cm %dst%\lib\ngspice\xtradev.cm
copy %cmsrc%\spice2poly.cm %dst%\lib\ngspice\spice2poly.cm
copy .\spinit_all %dst%\share\ngspice\scripts\spinit
copy .\spinitr .\spinit

if "%2" == "fftw" goto copy2
if "%3" == "fftw" goto copy2

copy %1\ngspice.exe %dst%\bin\
goto end

:copy2
copy %1\ngspice.exe %dst%\bin\
copy ..\..\fftw-3.3-dll32\libfftw3-3.dll %dst%\bin\
goto end

:b64

set dst=c:\Spice64
set cmsrc=.\codemodels\x64\Release

mkdir %dst%\bin
mkdir %dst%\lib\ngspice
mkdir %dst%\share\ngspice\scripts

copy "C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\redist\x64\Microsoft.VC140.OPENMP\vcomp140.dll" %dst%\bin\
copy %cmsrc%\analog64.cm %dst%\lib\ngspice\analog.cm
copy %cmsrc%\digital64.cm %dst%\lib\ngspice\digital.cm
copy %cmsrc%\table64.cm %dst%\lib\ngspice\table.cm
copy %cmsrc%\xtraevt64.cm %dst%\lib\ngspice\xtraevt.cm
copy %cmsrc%\xtradev64.cm %dst%\lib\ngspice\xtradev.cm
copy %cmsrc%\spice2poly64.cm %dst%\lib\ngspice\spice2poly.cm
copy .\spinit_all %dst%\share\ngspice\scripts\spinit
copy .\spinitr64 .\spinit

if "%2" == "fftw" goto copy2-64
if "%3" == "fftw" goto copy2-64

copy %1\ngspice.exe %dst%\bin\
goto end

:copy2-64
copy %1\ngspice.exe %dst%\bin\
copy ..\..\fftw-3.3-dll64\libfftw3-3.dll %dst%\bin\

:end
