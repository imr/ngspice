## Process this file with automake

SUBDIRS = cmaths ni sparse poly deriv misc fft
DIST_SUBDIRS = cmaths ni sparse poly deriv misc fft

if KLU_WANTED
SUBDIRS += KLU
DIST_SUBDIRS += KLU
endif

MAINTAINERCLEANFILES = Makefile.in KLU/Makefile.in

if KLU_WANTED
MAINTAINERCLEANFILES -= KLU/Makefile.in
endif
