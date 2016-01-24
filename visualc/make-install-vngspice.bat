@echo off
REM start vngspice.sln
REM compile cmpp.exe, codemodels, ngspice.exe
REM copy files to C:\Spice

REM start /WAIT "C:\Program Files (x86)\Microsoft Visual Studio 14.0\Common7\IDE\devenv.exe" "D:\Spice_general\ngspice\visualc\vngspice.sln" /Rebuild "ReleaseOMP|x86"

REM start /w "C:\Program Files (x86)\Microsoft Visual Studio 14.0\Common7\IDE\devenv.com" "D:\Spice_general\ngspice\visualc\vngspice.sln" /Rebuild 

if "%1" == "64" goto b64
if "%2" == "64" goto b64

set dst=c:\Spice
set cmsrc=Release.Win32

mkdir %dst%\bin
mkdir %dst%\lib\ngspice
mkdir %dst%\share\ngspice\scripts

copy "C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\redist\x86\Microsoft.VC140.OPENMP\vcomp140.dll" %dst%\bin\
copy analog\%cmsrc%\analog.cm %dst%\lib\ngspice\
copy digital\%cmsrc%\digital.cm %dst%\lib\ngspice\
copy table\%cmsrc%\table.cm %dst%\lib\ngspice\
copy xtraevt\%cmsrc%\xtraevt.cm %dst%\lib\ngspice\
copy xtradev\%cmsrc%\xtradev.cm %dst%\lib\ngspice\
copy spice2poly\%cmsrc%\spice2poly.cm %dst%\lib\ngspice\
copy .\spinit %dst%\share\ngspice\scripts\spinit

if "%1" == "fftw" goto copy2
if "%2" == "fftw" goto copy2

copy .\vngspice\ReleaseOMP.Win32\ngspice.exe %dst%\bin\
goto end

:copy2
copy .\vngspice-fftw\ReleaseOMP.Win32\ngspice.exe %dst%\bin\
copy ..\..\fftw-3.3.4-dll32\libfftw3-3.dll %dst%\bin\
goto end

:b64

set dst=c:\Spice64
set cmsrc=Release.x64

mkdir %dst%\bin
mkdir %dst%\lib\ngspice
mkdir %dst%\share\ngspice\scripts

copy "C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\redist\x64\Microsoft.VC140.OPENMP\vcomp140.dll" %dst%\bin\
copy analog\%cmsrc%\analog.cm %dst%\lib\ngspice\
copy digital\%cmsrc%\digital.cm %dst%\lib\ngspice\
copy table\%cmsrc%\table.cm %dst%\lib\ngspice\
copy xtraevt\%cmsrc%\xtraevt.cm %dst%\lib\ngspice\
copy xtradev\%cmsrc%\xtradev.cm %dst%\lib\ngspice\
copy spice2poly\%cmsrc%\spice2poly.cm %dst%\lib\ngspice\
copy .\spinit64 %dst%\share\ngspice\scripts\spinit

if "%1" == "fftw" goto copy2-64
if "%2" == "fftw" goto copy2-64

copy .\vngspice\ReleaseOMP.x64\ngspice.exe %dst%\bin\
goto end

:copy2-64
copy .\vngspice-fftw\ReleaseOMP.x64\ngspice.exe %dst%\bin\
copy ..\..\fftw-3.3.4-dll64\libfftw3-3.dll %dst%\bin\

:end
