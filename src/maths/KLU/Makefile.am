## Process this file with automake to produce Makefile.in
##
## KLU sparse solver: ngspice interface (klusmp.c) and ngspice additions.
## KLU itself comes either from an installed SuiteSparse (SYSTEM_KLU) or from
## the bundled copy in SuiteSparse/.

if !SYSTEM_KLU
SUBDIRS = SuiteSparse
endif

noinst_LTLIBRARIES = libKLU_real.la libKLU_complex.la libKLU.la

noinst_HEADERS = \
	klu_ngspice.h

# ngspice additions, compiled for real and (with -DCOMPLEX) complex entries
libKLU_real_la_SOURCES = \
	klu_multiply.c \
	klu_utils.c

libKLU_complex_la_SOURCES = \
	klu_multiply.c \
	klu_utils.c

libKLU_la_SOURCES = \
	klu_compat.c \
	klusmp.c

libKLU_real_la_CPPFLAGS = @AM_CPPFLAGS@ -I$(top_srcdir)/src/include
libKLU_complex_la_CPPFLAGS = @AM_CPPFLAGS@ -I$(top_srcdir)/src/include -DCOMPLEX
libKLU_la_CPPFLAGS = @AM_CPPFLAGS@ -I$(top_srcdir)/src/include

libKLU_la_LIBADD = \
	libKLU_real.la \
	libKLU_complex.la

if !SYSTEM_KLU
libKLU_la_LIBADD += SuiteSparse/libSuiteSparse.la
endif

AM_CFLAGS = $(STATIC)

MAINTAINERCLEANFILES = Makefile.in
