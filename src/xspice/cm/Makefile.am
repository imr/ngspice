## Process this file with automake to produce Makefile.in
#
# JW 3/9/01 - had a go and makeing an   autoconf script.

noinst_LTLIBRARIES = libcmxsp.la

libcmxsp_la_SOURCES = \
	cm.c       \
	cmevt.c    \
	cmmeters.c \
	cmutil.c   \
	cmexport.c



AM_CPPFLAGS = @AM_CPPFLAGS@ -I$(top_srcdir)/src/include -I$(top_srcdir)/src/spicelib/devices
AM_CFLAGS = $(STATIC)

MAINTAINERCLEANFILES = Makefile.in
