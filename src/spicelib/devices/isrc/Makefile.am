## Process this file with automake to produce Makefile.in

noinst_LTLIBRARIES = libisrc.la

libisrc_la_SOURCES =	\
	isrc.c		\
	isrcacct.c	\
	isrcacld.c	\
	isrcask.c	\
	isrcdefs.h	\
	isrcdel.c	\
	isrcdest.c	\
	isrcext.h	\
	isrcinit.c	\
	isrcinit.h	\
	isrcitf.h	\
	isrcload.c	\
	isrcmdel.c	\
	isrcpar.c	\
	isrctemp.c



AM_CPPFLAGS = @AM_CPPFLAGS@ -I$(top_srcdir)/src/include
AM_CFLAGS = $(STATIC)

if USE_CUSPICE_WANTED
.cu.lo:
	$(AM_V_GEN)$(top_srcdir)/src/libtool_wrapper_for_cuda.tcl $@ $(AM_CFLAGS) $(NVCC) $(CUDA_CFLAGS) $(AM_CPPFLAGS) -c $<

libisrc_la_SOURCES +=		\
	isrcsetup.c		\
	CUSPICE/isrctopology.c	\
	CUSPICE/cuisrcfree.c	\
	CUSPICE/cuisrcload.cu	\
	CUSPICE/cuisrcsetup.c	\
	CUSPICE/cuisrctemp.c

AM_CPPFLAGS += $(CUDA_CPPFLAGS)
endif

MAINTAINERCLEANFILES = Makefile.in
