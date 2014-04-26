## Process this file with automake to produce Makefile.in

noinst_LTLIBRARIES = libind.la

libind_la_SOURCES =	\
	ind.c		\
	indacld.c	\
	indask.c	\
	inddefs.h	\
	indext.h	\
	indinit.c	\
	indinit.h	\
	inditf.h	\
	indload.c	\
	indmask.c	\
	indmpar.c	\
	indparam.c	\
	indpzld.c	\
	indsacl.c	\
	indsetup.c	\
	indsload.c	\
	indsprt.c	\
	indsset.c	\
	indsupd.c	\
	indtemp.c	\
	indtrunc.c	\
	mutacld.c	\
	mutask.c	\
	mutparam.c	\
	mutpzld.c	\
	mutsetup.c	\
	mutsprt.c	\
	mutsset.c	\
	muttemp.c


if KLU_WANTED
libind_la_SOURCES += indbindCSC.c
libind_la_SOURCES += mutbindCSC.c
endif

AM_CPPFLAGS = @AM_CPPFLAGS@ -I$(top_srcdir)/src/include
AM_CFLAGS = $(STATIC)

if USE_CUSPICE_WANTED
.cu.lo:
	$(AM_V_GEN)$(top_srcdir)/src/libtool_wrapper_for_cuda.tcl $@ $(AM_CFLAGS) $(NVCC) $(CUDA_CFLAGS) $(AM_CPPFLAGS) -c $<

libind_la_SOURCES +=		\
	CUSPICE/indtopology.c	\
	CUSPICE/cuindfree.c	\
	CUSPICE/cuindload.cu	\
	CUSPICE/cuindsetup.c	\
	CUSPICE/cuindtemp.c

libind_la_SOURCES +=		\
	CUSPICE/muttopology.c	\
	CUSPICE/cumutfree.c	\
	CUSPICE/cumutload.cu	\
	CUSPICE/cumutsetup.c	\
	CUSPICE/cumuttemp.c

AM_CPPFLAGS += $(CUDA_CPPFLAGS)
endif

MAINTAINERCLEANFILES = Makefile.in
