@echo off
REM start vngspice.sln
REM compile cmpp.exe, codemodels, ngspice.exe
REM copy files to C:\Spice

REM start /WAIT "C:\Program Files (x86)\Microsoft Visual Studio 14.0\Common7\IDE\devenv.exe" "D:\Spice_general\ngspice\visualc\vngspice.sln" /Rebuild "ReleaseOMP|x86"

REM start /w "C:\Program Files (x86)\Microsoft Visual Studio 14.0\Common7\IDE\devenv.com" "D:\Spice_general\ngspice\visualc\vngspice.sln" /Rebuild 


mkdir c:\Spice\bin
mkdir c:\Spice\lib\ngspice
mkdir C:\Spice\share\ngspice\scripts

copy "C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\redist\x86\Microsoft.VC140.OPENMP\vcomp140.dll" c:\Spice\bin\vcomp140.dll
copy .\codemodels\Win32\Release\analog.cm c:\Spice\lib\ngspice\analog.cm
copy .\codemodels\Win32\Release\digital.cm c:\Spice\lib\ngspice\digital.cm
copy .\codemodels\Win32\Release\table.cm c:\Spice\lib\ngspice\table.cm
copy .\codemodels\Win32\Release\xtraevt.cm c:\Spice\lib\ngspice\xtraevt.cm
copy .\codemodels\Win32\Release\xtradev.cm c:\Spice\lib\ngspice\xtradev.cm
copy .\codemodels\Win32\Release\spice2poly.cm c:\Spice\lib\ngspice\spice2poly.cm
copy .\spinit C:\Spice\share\ngspice\scripts\spinit

if "%1" == "fftw" goto copy2

copy .\vngspice\ReleaseOMP.Win32\ngspice.exe c:\Spice\bin\ngspice.exe
goto end

:copy2
copy .\vngspice-fftw\ReleaseOMP.Win32\ngspice.exe c:\Spice\bin\ngspice.exe
copy "..\..\fftw-3.3.4-dll32\libfftw3-3.dll" "C:\Spice\bin\libfftw3-3.dll"

:end
