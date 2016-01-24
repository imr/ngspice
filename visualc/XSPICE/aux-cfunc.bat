rem invoke as
rem    .\aux-cfunc.bat analog

set sub=%1
set cmpp=.\cmpp\Release.Win32\cmpp.exe

set CMPP_IDIR=../../src/xspice/icm/%sub%
set CMPP_ODIR=icm/%sub%
if not exist icm\%sub% mkdir icm\%sub%
%cmpp% -lst

for /F %%n in (..\..\src\xspice\icm\%sub%\modpath.lst) do (
  set CMPP_IDIR=../../src/xspice/icm/%sub%/%%n
  set CMPP_ODIR=icm/%sub%/%%n
  if not exist icm\%sub%\%%n mkdir icm\%sub%\%%n
  %cmpp% -ifs
  %cmpp% -mod
  pushd icm\%sub%\%%n
  if exist %%n-cfunc.c del %%n-cfunc.c
  if exist %%n-ifspec.c del %%n-ifspec.c
  rename cfunc.c %%n-cfunc.c
  rename ifspec.c %%n-ifspec.c
  popd
)

