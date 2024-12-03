## Process this file with automake to produce Makefile.in

EXTRA_DIST = SuiteSparse.org

noinst_LTLIBRARIES = libKLU_real.la libKLU_complex.la libKLU.la

noinst_HEADERS = \
    klu_internal.h \
    klu_version.h


libKLU_real_la_SOURCES = 	\
	klu_multiply.c		\
	klu_utils.c


libKLU_real_la_CPPFLAGS = @AM_CPPFLAGS@ -I$(top_srcdir)/src/include


libKLU_complex_la_SOURCES = 	\
	klu_multiply.c		\
	klu_utils.c


libKLU_complex_la_CPPFLAGS = @AM_CPPFLAGS@ -I$(top_srcdir)/src/include -DCOMPLEX


libKLU_la_SOURCES =		\
	klusmp.c

libKLU_la_LIBADD =		\
	libKLU_real.la		\
	libKLU_complex.la


libKLU_la_CPPFLAGS = @AM_CPPFLAGS@ -I$(top_srcdir)/src/include
AM_CFLAGS = $(STATIC)

MAINTAINERCLEANFILES = Makefile.in
