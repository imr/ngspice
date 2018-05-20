## Process this file with automake to produce Makefile.in

EXTRA_DIST = readme.txt

noinst_LTLIBRARIES = libnumparam.la

libnumparam_la_SOURCES = \
	spicenum.c	\
	xpressn.c	\
	mystring.c	\
	general.h	\
	numpaif.h	\
	numparam.h

AM_CPPFLAGS = @AM_CPPFLAGS@ -I$(top_srcdir)/src/include
AM_CFLAGS = $(STATIC)
MAINTAINERCLEANFILES = Makefile.in
