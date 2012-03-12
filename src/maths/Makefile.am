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

if UMFPACK_WANTED
SUBDIRS += UMFPACK
DIST_SUBDIRS += UMFPACK
endif

MAINTAINERCLEANFILES = Makefile.in

if !UMFPACK_WANTED
if !KLU_WANTED
if !SuperLU_WANTED #SPARSE_WANTED
MAINTAINERCLEANFILES += KLU/Makefile.in SuperLU/Makefile.in UMFPACK/Makefile.in
else #SuperLU_WANTED
MAINTAINERCLEANFILES += KLU/Makefile.in UMFPACK/Makefile.in
endif
else #KLU_WANTED
MAINTAINERCLEANFILES += SuperLU/Makefile.in UMFPACK/Makefile.in
endif
else #UMFPACK_WANTED
MAINTAINERCLEANFILES += KLU/Makefile.in SuperLU/Makefile.in
endif
