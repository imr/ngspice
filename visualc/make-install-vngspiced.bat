@echo off
REM start vngspice.sln
REM compile cmpp.exe, codemodels, ngspice.exe as 32 bit debug version
REM copy files to C:\Spiced

REM start /WAIT "C:\Program Files (x86)\Microsoft Visual Studio 14.0\Common7\IDE\devenv.exe" "D:\Spice_general\ngspice\visualc\vngspice.sln" /Rebuild "ReleaseOMP|x86"

REM start /w "C:\Program Files (x86)\Microsoft Visual Studio 14.0\Common7\IDE\devenv.com" "D:\Spice_general\ngspice\visualc\vngspice.sln" /Rebuild 

if "%1" == "64" goto b64

if "%2" == "64" goto b64

mkdir c:\Spiced\bin
mkdir c:\Spiced\lib\ngspice
mkdir C:\Spiced\share\ngspice\scripts

copy "C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\redist\x86\Microsoft.VC140.OPENMP\vcomp140.dll" c:\Spiced\bin\vcomp140.dll
copy .\codemodels\Win32\Debug\analog.cm c:\Spiced\lib\ngspice\analog.cm
copy .\codemodels\Win32\Debug\digital.cm c:\Spiced\lib\ngspice\digital.cm
copy .\codemodels\Win32\Debug\table.cm c:\Spiced\lib\ngspice\table.cm
copy .\codemodels\Win32\Debug\xtraevt.cm c:\Spiced\lib\ngspice\xtraevt.cm
copy .\codemodels\Win32\Debug\xtradev.cm c:\Spiced\lib\ngspice\xtradev.cm
copy .\codemodels\Win32\Debug\spice2poly.cm c:\Spiced\lib\ngspice\spice2poly.cm
copy .\spinitd C:\Spiced\share\ngspice\scripts\spinit

if "%1" == "fftw" goto copy2

copy .\vngspice\Debug.Win32\ngspice.exe c:\Spiced\bin\ngspice.exe
goto end

:copy2
copy .\vngspice-fftw\Debug.Win32\ngspice.exe c:\Spiced\bin\ngspice.exe
copy "..\..\fftw-3.3.4-dll32\libfftw3-3.dll" "C:\Spiced\bin\libfftw3-3.dll"

:b64

mkdir c:\Spice64d\bin
mkdir c:\Spice64d\lib\ngspice
mkdir C:\Spice64d\share\ngspice\scripts

copy "C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\redist\x64\Microsoft.VC140.OPENMP\vcomp140.dll" c:\Spice64d\bin\vcomp140.dll
copy .\codemodels\x64\Debug\analog64.cm c:\Spice64d\lib\ngspice\analog.cm
copy .\codemodels\x64\Debug\digital64.cm c:\Spice64d\lib\ngspice\digital.cm
copy .\codemodels\x64\Debug\table64.cm c:\Spice64d\lib\ngspice\table.cm
copy .\codemodels\x64\Debug\xtraevt64.cm c:\Spice64d\lib\ngspice\xtraevt.cm
copy .\codemodels\x64\Debug\xtradev64.cm c:\Spice64d\lib\ngspice\xtradev.cm
copy .\codemodels\x64\Debug\spice2poly64.cm c:\Spice64d\lib\ngspice\spice2poly.cm
copy .\spinitd64 C:\Spice64d\share\ngspice\scripts\spinit

if "%1" == "fftw" goto copy2-64

if "%2" == "fftw" goto copy2-64

copy .\vngspice\Debug.x64\ngspice.exe c:\Spice64d\bin\ngspice.exe
goto end

:copy2-64
copy .\vngspice-fftw\Debug.x64\ngspice.exe c:\Spice64d\bin\ngspice.exe
copy "..\..\fftw-3.3.4-dll64\libfftw3-3.dll" "C:\Spice64d\bin\libfftw3-3.dll"

:end
