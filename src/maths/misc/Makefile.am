## Process this file with automake to produce Makefile.in

noinst_LTLIBRARIES = libmathmisc.la

libmathmisc_la_SOURCES = \
	accuracy.c	\
	accuracy.h	\
	bernoull.h	\
	bernoull.c	\
	erfc.c		\
	equality.c	\
	logb.c		\
	scalb.c		\
	norm.h		\
	norm.c		\
	randnumb.c


if !HAVE_DECL_ISINF
libmathmisc_la_SOURCES += isinf.c
endif

if !HAVE_DECL_ISNAN
libmathmisc_la_SOURCES += isnan.c
endif


EXTRA_DIST = test_accuracy.c test_erfc.c

AM_CPPFLAGS = @AM_CPPFLAGS@ -I$(top_srcdir)/src/include
AM_CFLAGS = $(STATIC)

MAINTAINERCLEANFILES = Makefile.in
