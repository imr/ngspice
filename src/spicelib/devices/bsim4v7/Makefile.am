## Process this file with automake to produce Makefile.in

noinst_LTLIBRARIES = libbsim4v7.la

libbsim4v7_la_SOURCES =	\
	b4v7.c		\
	b4v7acld.c	\
	b4v7ask.c		\
	b4v7check.c	\
	b4v7cvtest.c	\
	b4v7del.c		\
	b4v7dest.c	\
	b4v7geo.c		\
	b4v7getic.c	\
	b4v7ld.c		\
	b4v7mask.c	\
	b4v7mdel.c	\
	b4v7mpar.c	\
	b4v7noi.c		\
	b4v7par.c		\
	b4v7pzld.c	\
	b4v7set.c		\
	b4v7soachk.c	\
	b4v7temp.c	\
	b4v7trunc.c	\
	bsim4v7def.h	\
	bsim4v7ext.h	\
	bsim4v7init.c	\
	bsim4v7init.h	\
	bsim4v7itf.h


if KLU_WANTED
libbsim4v7_la_SOURCES += b4v7bindCSC.c
endif

AM_CPPFLAGS = @AM_CPPFLAGS@ -I$(top_srcdir)/src/include
AM_CFLAGS = $(STATIC)

if USE_CUSPICE_WANTED
.cu.lo:
	$(AM_V_GEN)$(top_srcdir)/src/libtool_wrapper_for_cuda.tcl $@ $(AM_CFLAGS) $(NVCC) $(CUDA_CFLAGS) $(AM_CPPFLAGS) -c $<

libbsim4v7_la_SOURCES +=		\
	CUSPICE/b4v7topology.c		\
	CUSPICE/cubsim4v7free.c		\
	CUSPICE/cubsim4v7getic.c	\
	CUSPICE/cubsim4v7load.cu	\
	CUSPICE/cubsim4v7setup.c	\
	CUSPICE/cubsim4v7temp.c		\
	CUSPICE/cubsim4v7trunc.cu

AM_CPPFLAGS += $(CUDA_CPPFLAGS) -I$(top_srcdir)/src/spicelib/analysis
endif

MAINTAINERCLEANFILES = Makefile.in

EXTRA_DIST = B4TERMS_OF_USE
