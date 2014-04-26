## Process this file with automake to produce Makefile.in

noinst_LTLIBRARIES = libres.la

libres_la_SOURCES =	\
	res.c		\
	resask.c	\
	resdefs.h	\
	resext.h	\
	resinit.c	\
	resinit.h	\
	resitf.h	\
	resload.c	\
	resmask.c	\
	resmpar.c	\
	resnoise.c	\
	resparam.c	\
	respzld.c	\
	ressacl.c	\
	ressetup.c	\
	ressoachk.c	\
	ressload.c	\
	ressprt.c	\
	ressset.c	\
	restemp.c


if KLU_WANTED
libres_la_SOURCES += resbindCSC.c
endif

AM_CPPFLAGS = @AM_CPPFLAGS@ -I$(top_srcdir)/src/include
AM_CFLAGS = $(STATIC)

if USE_CUSPICE_WANTED
.cu.lo:
	$(AM_V_GEN)$(top_srcdir)/src/libtool_wrapper_for_cuda.tcl $@ $(AM_CFLAGS) $(NVCC) $(CUDA_CFLAGS) $(AM_CPPFLAGS) -c $<

libres_la_SOURCES +=		\
	CUSPICE/restopology.c	\
	CUSPICE/curesfree.c	\
	CUSPICE/curesload.cu	\
	CUSPICE/curessetup.c	\
	CUSPICE/curestemp.c

AM_CPPFLAGS += $(CUDA_CPPFLAGS)
endif

MAINTAINERCLEANFILES = Makefile.in
