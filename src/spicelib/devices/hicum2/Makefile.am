## Process this file with automake to produce Makefile.in

noinst_LTLIBRARIES = libhicum.la

libhicum_la_SOURCES =	\
	hicum.c		\
	hicumacld.c	\
	hicumask.c	\
	hicumconv.c	\
	hicumdefs.h	\
	hicumext.h	\
	hicumgetic.c	\
	hicuminit.c	\
	hicuminit.h	\
	hicumitf.h	\
	hicumload.c	\
	hicummask.c	\
	hicummpar.c	\
	hicumnoise.c	\
	hicumparam.c	\
	hicumpzld.c	\
	hicumsetup.c	\
	hicumsoachk.c	\
	hicumtemp.c	\
	hicumtrunc.c



AM_CPPFLAGS = @AM_CPPFLAGS@ -I$(top_srcdir)/src/include
AM_CFLAGS = $(STATIC)

MAINTAINERCLEANFILES = Makefile.in
