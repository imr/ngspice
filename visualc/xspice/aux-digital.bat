rem Pre-build commands for digital.cm.
rem Make support components for Icarus Verilog co-simulation:
rem ivlng.dll and ivlng.vpi.  Then run aux-cfunc.bat.

md verilog
pushd verilog
set src=..\..\..\src\xspice\verilog
set inc=..\..\..\src\include
CL.EXE /O2 /LD /EHsc /Feivlng.DLL /I%src% /I%inc% %src%\icarus_shim.c ^
/link /EXPORT:Cosim_setup /EXPORT:Get_ng_vvp

rem Make a dummy libvvp.obj, needed for shim.vpi (to be renamed ivlng.vpi).

lib.exe /def:%src%\libvvp.def /machine:X64
CL.EXE /O2 /LD /EHsc /Feshim.vpi /I. /I%inc% %src%\vpi.c libvvp.lib ivlng.lib /link /DLL /EXPORT:vlog_startup_routines
dir
popd
.\aux-cfunc.bat digital
