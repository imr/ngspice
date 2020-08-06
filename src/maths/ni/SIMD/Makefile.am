## Process this file with automake to produce Makefile.in

noinst_LTLIBRARIES = libnisimd.la

libnisimd_la_SOURCES = \
		simdniinteg.c	

AM_CPPFLAGS = @AM_CPPFLAGS@  -I$(top_srcdir)/src/include -I$(top_srcdir)/src/spicelib/devices
AM_CFLAGS = $(STATIC) $(SIMD_CFLAGS)
MAINTAINERCLEANFILES = Makefile.in
