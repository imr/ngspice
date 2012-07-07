## Process this file with automake to produce Makefile.in

noinst_LTLIBRARIES = libpoly.la

libpoly_la_SOURCES = \
	interpolate.c \
	interpolate.h \
	polyfit.c \
	polyfit.h \
	polyderiv.c \
	polyderiv.h \
	polyeval.c \
	polyeval.h

EXTRA_DIST = poly.h

AM_CPPFLAGS = @AM_CPPFLAGS@ -I$(top_srcdir)/src/include
AM_CFLAGS = $(STATIC)

MAINTAINERCLEANFILES = Makefile.in
