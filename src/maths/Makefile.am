## Process this file with automake

SUBDIRS = cmaths ni sparse poly deriv misc fft
DIST_SUBDIRS = cmaths ni sparse poly deriv misc fft KLU SuperLU UMFPACK

if KLU_WANTED
SUBDIRS += KLU
endif

if SuperLU_WANTED
SUBDIRS += SuperLU
endif

if UMFPACK_WANTED
SUBDIRS += UMFPACK
endif

MAINTAINERCLEANFILES = Makefile.in
