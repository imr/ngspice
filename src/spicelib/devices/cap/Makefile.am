## Process this file with automake to produce Makefile.in

noinst_LTLIBRARIES = libcap.la

libcap_la_SOURCES =	\
	cap.c		\
	capacld.c	\
	capask.c	\
	capdefs.h	\
	capext.h	\
	capgetic.c	\
	capinit.c	\
	capinit.h	\
	capitf.h	\
	capload.c	\
	capmask.c	\
	capmpar.c	\
	capparam.c	\
	cappzld.c	\
	capsacl.c	\
	capsetup.c	\
	capsload.c	\
	capsoachk.c	\
	capsprt.c	\
	capsset.c	\
	capsupd.c	\
	captemp.c	\
	captrunc.c


if KLU_WANTED
libcap_la_SOURCES += capbindCSC.c
endif

AM_CPPFLAGS = @AM_CPPFLAGS@ -I$(top_srcdir)/src/include
AM_CFLAGS = $(STATIC)

if USE_CUSPICE_WANTED
.cu.lo:
	$(AM_V_GEN)$(top_srcdir)/src/libtool_wrapper_for_cuda.tcl $@ $(AM_CFLAGS) $(NVCC) $(CUDA_CFLAGS) $(AM_CPPFLAGS) -c $<

libcap_la_SOURCES +=		\
	CUSPICE/captopology.c	\
	CUSPICE/cucapfree.c	\
	CUSPICE/cucapgetic.c	\
	CUSPICE/cucapload.cu	\
	CUSPICE/cucapsetup.c	\
	CUSPICE/cucaptemp.c

AM_CPPFLAGS += $(CUDA_CPPFLAGS)
endif

MAINTAINERCLEANFILES = Makefile.in
