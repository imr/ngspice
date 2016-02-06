rem invoke as
rem    .\aux-udnfunc.bat xtraevt

set sub=%1

for /F %%n in (..\..\src\xspice\icm\%sub%\udnpath.lst) do (
  if not exist icm\%sub%\%%n mkdir icm\%sub%\%%n
  copy /Y ..\..\src\xspice\icm\%sub%\%%n\udnfunc.c icm\%sub%\%%n\%%n-udnfunc.c
)
