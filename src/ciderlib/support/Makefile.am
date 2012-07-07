## Process this file with automake to produce Makefile.in

noinst_LTLIBRARIES = libcidersuprt.la

libcidersuprt_la_SOURCES =	\
		database.c	\
		devprint.c	\
		geominfo.c	\
		globals.c	\
		integset.c	\
		integuse.c	\
		logfile.c	\
		mater.c		\
		misc.c		\
		mobil.c		\
		recomb.c	\
		suprem.c	\
		suprmitf.c

AM_CPPFLAGS = @AM_CPPFLAGS@ -I$(top_srcdir)/src/include
AM_CFLAGS = $(STATIC)
MAINTAINERCLEANFILES = Makefile.in

EXTRA_DIST = readme
