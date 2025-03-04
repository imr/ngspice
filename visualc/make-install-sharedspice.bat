@echo off

REM copy ngspice.dll, codemodels *.cm to Spice64
REM arguments to make-install-vngspice:
REM %1: path to ngspice.dll, %2: release/debug %3: fftw, %4: omp

if "%2" == "release" (
    set dst=c:\Spice64_dll
    set cmsrc=.\codemodels\x64\Release
)
if "%2" == "debug" (
    set dst=c:\Spice64d_dll
    set cmsrc=.\codemodels\x64\Debug
    copy .\spinitd64 .\spinit
)

mkdir %dst%\dll-vs
mkdir %dst%\lib\lib-vs
mkdir %dst%\lib\ngspice
mkdir %dst%\include\ngspice

if "%3" == "omp" (
    copy "c:\Program Files\Microsoft Visual Studio\2022\Community\VC\Redist\MSVC\14.42.34433\debug_nonredist\x64\Microsoft.VC143.OpenMP.LLVM\libomp140.x86_64.dll" %dst%\dll-vs\
)
if "%4" == "omp" (
    copy "c:\Program Files\Microsoft Visual Studio\2022\Community\VC\Redist\MSVC\14.42.34433\debug_nonredist\x64\Microsoft.VC143.OpenMP.LLVM\libomp140.x86_64.dll" %dst%\dll-vs\
)
copy %cmsrc%\analog64.cm %dst%\lib\ngspice\analog.cm
copy %cmsrc%\digital64.cm %dst%\lib\ngspice\digital.cm
copy %cmsrc%\table64.cm %dst%\lib\ngspice\table.cm
copy %cmsrc%\xtraevt64.cm %dst%\lib\ngspice\xtraevt.cm
copy %cmsrc%\xtradev64.cm %dst%\lib\ngspice\xtradev.cm
copy %cmsrc%\spice2poly64.cm %dst%\lib\ngspice\spice2poly.cm
copy xspice\verilog\ivlng.dll %dst%\lib\ngspice\ivlng.dll
copy xspice\verilog\shim.vpi %dst%\lib\ngspice\ivlng.vpi

if "%3" == "fftw" goto copy-fftw

copy %1\ngspice.dll %dst%\dll-vs\
goto end

:copy-fftw
copy %1\ngspice.dll %dst%\dll-vs\
copy ..\..\fftw-3.3-dll64\libfftw3-3.dll %dst%\dll-vs\

:end
copy %1\ngspice.lib %dst%\lib\lib-vs\
copy %1\ngspice.exp %dst%\lib\lib-vs\
copy %1\..\..\..\src\include\ngspice\sharedspice.h %dst%\include\ngspice\

mkdir %dst%\share\ngspice\scripts
copy .\spinit_all %dst%\share\ngspice\scripts\spinit
