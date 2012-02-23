## Process this file with automake

SUBDIRS = cmaths ni sparse poly deriv misc fft
DIST_SUBDIRS = cmaths ni sparse poly deriv misc fft

if KLU_WANTED
SUBDIRS += KLU
DIST_SUBDIRS += KLU
endif

if SuperLU_WANTED
SUBDIRS += SuperLU
DIST_SUBDIRS += SuperLU
endif

MAINTAINERCLEANFILES = Makefile.in

if !KLU_WANTED
if SuperLU_WANTED
MAINTAINERCLEANFILES += KLU/Makefile.in
else
MAINTAINERCLEANFILES += KLU/Makefile.in SuperLU/Makefile.in
endif
else
if !SuperLU_WANTED
MAINTAINERCLEANFILES += SuperLU/Makefile.in
endif
endif
