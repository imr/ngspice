## Process this file with automake to produce Makefile.in

noinst_LTLIBRARIES = libbsim4.la

libbsim4_la_SOURCES =	\
	b4.c		\
	b4acld.c	\
	b4ask.c		\
	b4check.c	\
	b4cvtest.c	\
	b4del.c		\
	b4dest.c	\
	b4geo.c		\
	b4getic.c	\
	b4ld.c		\
	b4mask.c	\
	b4mdel.c	\
	b4mpar.c	\
	b4noi.c		\
	b4par.c		\
	b4pzld.c	\
	b4set.c		\
	b4soachk.c	\
	b4temp.c	\
	b4trunc.c	\
	bsim4def.h	\
	bsim4ext.h	\
	bsim4init.c	\
	bsim4init.h	\
	bsim4itf.h


if KLU_WANTED
libbsim4_la_SOURCES += b4bindCSC.c
endif

AM_CPPFLAGS = @AM_CPPFLAGS@ -I$(top_srcdir)/src/include
AM_CFLAGS = $(STATIC)

if USE_CUSPICE_WANTED
.cu.lo:
	$(AM_V_GEN)$(top_srcdir)/src/libtool_wrapper_for_cuda.tcl $@ $(AM_CFLAGS) $(NVCC) $(CUDA_CFLAGS) $(AM_CPPFLAGS) -c $<

libbsim4_la_SOURCES +=		\
	CUSPICE/b4topology.c	\
	CUSPICE/cubsim4free.c	\
	CUSPICE/cubsim4getic.c	\
	CUSPICE/cubsim4load.cu	\
	CUSPICE/cubsim4setup.c	\
	CUSPICE/cubsim4temp.c

AM_CPPFLAGS += $(CUDA_CPPFLAGS)
endif

MAINTAINERCLEANFILES = Makefile.in

EXTRA_DIST = B4TERMS_OF_USE
