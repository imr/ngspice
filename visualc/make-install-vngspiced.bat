@echo off

REM copy ngspice.exe, codemodels *.cm to C:\Spiced or Spice64d
REM arguments to make-install-vngspiced:
REM %1: path to ngspice.exe, %2: fftw

set dst=c:\Spice64d
set cmsrc=.\codemodels\x64\Debug

mkdir %dst%\bin
mkdir %dst%\lib\ngspice

if "%2" == "omp" (
    copy "c:\Program Files\Microsoft Visual Studio\2022\Community\VC\Redist\MSVC\14.42.34433\debug_nonredist\x64\Microsoft.VC143.OpenMP.LLVM\libomp140.x86_64.dll" %dst%\bin\
)
if "%3" == "omp" (
    copy "c:\Program Files\Microsoft Visual Studio\2022\Community\VC\Redist\MSVC\14.42.34433\debug_nonredist\x64\Microsoft.VC143.OpenMP.LLVM\libomp140.x86_64.dll" %dst%\bin\
)
copy %cmsrc%\analog64.cm %dst%\lib\ngspice\analog.cm
copy %cmsrc%\digital64.cm %dst%\lib\ngspice\digital.cm
copy %cmsrc%\table64.cm %dst%\lib\ngspice\table.cm
copy %cmsrc%\xtraevt64.cm %dst%\lib\ngspice\xtraevt.cm
copy %cmsrc%\xtradev64.cm %dst%\lib\ngspice\xtradev.cm
copy %cmsrc%\spice2poly64.cm %dst%\lib\ngspice\spice2poly.cm
copy xspice\verilog\ivlng.dll %dst%\lib\ngspice\ivlng.dll
copy xspice\verilog\shim.vpi %dst%\lib\ngspice\ivlng.vpi

if "%2" == "fftw" goto copy2-64

copy %1\ngspice.exe %dst%\bin\
copy .\spinitd64 .\spinit
goto end

:copy2-64
copy %1\ngspice.exe %dst%\bin\
copy ..\..\fftw-3.3-dll64\libfftw3-3.dll %dst%\bin\

:end
mkdir %dst%\share\ngspice\scripts\src\ngspice
copy .\spinit_all %dst%\share\ngspice\scripts\spinit

cd ..\src
copy ciderinit %dst%\share\ngspice\scripts
copy devaxis %dst%\share\ngspice\scripts
copy devload %dst%\share\ngspice\scripts
copy setplot %dst%\share\ngspice\scripts
copy spectrum %dst%\share\ngspice\scripts
copy xspice\verilog\vlnggen %dst%\share\ngspice\scripts
copy xspice\verilog\MSVC.CMD %dst%\share\ngspice\scripts
copy xspice\verilog\*.cpp %dst%\share\ngspice\scripts\src
copy include\ngspice\cosim.h %dst%\share\ngspice\scripts\src\ngspice
copy include\ngspice\miftypes.h %dst%\share\ngspice\scripts\src\ngspice
copy include\ngspice\cmtypes.h %dst%\share\ngspice\scripts\src\ngspice
copy xspice\verilog\coroutine*.h %dst%\share\ngspice\scripts\src\ngspice
copy xspice\vhdl\ghnggen %dst%\share\ngspice\scripts
copy xspice\vhdl\ghdl_shim.* %dst%\share\ngspice\scripts\src
copy xspice\vhdl\ghdl_vpi.c %dst%\share\ngspice\scripts\src
