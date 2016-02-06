@echo off
REM start vngspice.sln
REM compile cmpp.exe, codemodels, ngspice.exe as 32 bit debug version
REM copy files to C:\Spiced

REM start /WAIT "C:\Program Files (x86)\Microsoft Visual Studio 14.0\Common7\IDE\devenv.exe" "D:\Spice_general\ngspice\visualc\vngspice.sln" /Rebuild "ReleaseOMP|x86"

REM start /w "C:\Program Files (x86)\Microsoft Visual Studio 14.0\Common7\IDE\devenv.com" "D:\Spice_general\ngspice\visualc\vngspice.sln" /Rebuild 

if "%1" == "64" goto b64
if "%2" == "64" goto b64

set dst=c:\Spiced
set cmsrc=.\codemodels\Win32\Debug

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
copy .\spinitd %dst%\share\ngspice\scripts\spinit

if "%1" == "fftw" goto copy2
if "%2" == "fftw" goto copy2

copy .\vngspice\Debug.Win32\ngspice.exe %dst%\bin\
goto end

:copy2
copy .\vngspice-fftw\Debug.Win32\ngspice.exe %dst%\bin\
copy ..\..\fftw-3.3.4-dll32\libfftw3-3.dll %dst%\bin\
goto end

:b64

set dst=c:\Spice64d
set cmsrc=.\codemodels\x64\Debug

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
copy .\spinitd64 %dst%\share\ngspice\scripts\spinit

if "%1" == "fftw" goto copy2-64
if "%2" == "fftw" goto copy2-64

copy .\vngspice\Debug.x64\ngspice.exe %dst%\bin\
goto end

:copy2-64
copy .\vngspice-fftw\Debug.x64\ngspice.exe %dst%\bin\
copy ..\..\fftw-3.3.4-dll64\libfftw3-3.dll %dst%\bin\

:end
