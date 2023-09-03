## Process this file with automake

SUBDIRS = cmaths ni sparse poly deriv misc fft
DIST_SUBDIRS = cmaths ni sparse poly deriv misc fft dense KLU

if SP_WANTED
SUBDIRS += dense
endif

if KLU_WANTED
SUBDIRS += KLU
endif

MAINTAINERCLEANFILES = Makefile.in
