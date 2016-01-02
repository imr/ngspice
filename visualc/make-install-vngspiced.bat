@echo off
REM start vngspice.sln
REM compile cmpp.exe, codemodels, ngspice.exe as 32 bit debug version
REM copy files to C:\Spiced

REM start /WAIT "C:\Program Files (x86)\Microsoft Visual Studio 14.0\Common7\IDE\devenv.exe" "D:\Spice_general\ngspice\visualc\vngspice.sln" /Rebuild "ReleaseOMP|x86"

REM start /w "C:\Program Files (x86)\Microsoft Visual Studio 14.0\Common7\IDE\devenv.com" "D:\Spice_general\ngspice\visualc\vngspice.sln" /Rebuild 

mkdir c:\Spiced\bin
mkdir c:\Spiced\lib\ngspice
mkdir C:\Spiced\share\ngspice\scripts

copy .\vngspice\Debug.Win32\ngspice.exe c:\Spiced\bin\ngspice.exe
copy "C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\redist\x86\Microsoft.VC140.OPENMP\vcomp140.dll" c:\Spiced\bin\vcomp140.dll
copy .\codemodels\Win32\Debug\analog.cm c:\Spiced\lib\ngspice\analog.cm
copy .\codemodels\Win32\Debug\digital.cm c:\Spiced\lib\ngspice\digital.cm
copy .\codemodels\Win32\Debug\table.cm c:\Spiced\lib\ngspice\table.cm
copy .\codemodels\Win32\Debug\xtraevt.cm c:\Spiced\lib\ngspice\xtraevt.cm
copy .\codemodels\Win32\Debug\xtradev.cm c:\Spiced\lib\ngspice\xtradev.cm
copy .\codemodels\Win32\Debug\spice2poly.cm c:\Spiced\lib\ngspice\spice2poly.cm
copy .\spinitd C:\Spiced\share\ngspice\scripts\spinit
