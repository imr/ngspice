## Process this file with automake to produce Makefile.in

noinst_LTLIBRARIES = libmathfft.la

libmathfft_la_SOURCES =		\
		fftext.c	\
		fftlib.c	\
		fftlib.h	\
		matlib.c	\
		matlib.h

EXTRA_DIST = NOTE Read-Me


AM_CPPFLAGS = @AM_CPPFLAGS@ -I$(top_srcdir)/src/include
AM_CFLAGS = $(STATIC)
MAINTAINERCLEANFILES = Makefile.in
