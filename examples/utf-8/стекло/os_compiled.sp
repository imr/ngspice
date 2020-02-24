*ng_script test for oscompiled

.control
if $oscompiled = 0
  echo [0], Other
endif
if $oscompiled = 1
  echo [1], MINGW for MS Windows
endif
if $oscompiled = 2
  echo [2], Cygwin for MS Windows
endif
if $oscompiled = 3
  echo [3], FreeBSD
endif
if $oscompiled = 4
  echo [4], OpenBSD
endif
if $oscompiled = 5
  echo [5], Solaris
endif
if $oscompiled = 6
  echo [6], Linux
endif
if $oscompiled = 7
  echo [7], macOS
endif
if $oscompiled = 8
  echo [8], Visual Studio for MS Windows 
endif

.endc

.end
