## Process this file with automake to produce Makefile.in

noinst_LTLIBRARIES = libvsrc.la

libvsrc_la_SOURCES =	\
	vsrc.c		\
	vsrcacct.c	\
	vsrcacld.c	\
	vsrcask.c	\
	vsrcdefs.h	\
	vsrcdel.c	\
	vsrcext.h	\
	vsrcfbr.c	\
	vsrcinit.c	\
	vsrcinit.h	\
	vsrcitf.h	\
	vsrcload.c	\
	vsrcpar.c	\
	vsrcpzld.c	\
	vsrcpzs.c	\
	vsrcset.c	\
	vsrctemp.c


if KLU_WANTED
libvsrc_la_SOURCES += vsrcbindCSC.c
endif

AM_CPPFLAGS = @AM_CPPFLAGS@ -I$(top_srcdir)/src/include
AM_CFLAGS = $(STATIC)

if USE_CUSPICE_WANTED
.cu.lo:
	$(AM_V_GEN)$(top_srcdir)/src/libtool_wrapper_for_cuda.tcl $@ $(AM_CFLAGS) $(NVCC) $(CUDA_CFLAGS) $(AM_CPPFLAGS) -c $<

libvsrc_la_SOURCES +=		\
	CUSPICE/vsrctopology.c	\
	CUSPICE/cuvsrcfree.c	\
	CUSPICE/cuvsrcload.cu	\
	CUSPICE/cuvsrcsetup.c	\
	CUSPICE/cuvsrctemp.c

AM_CPPFLAGS += $(CUDA_CPPFLAGS)
endif

MAINTAINERCLEANFILES = Makefile.in
