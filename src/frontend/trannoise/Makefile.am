noinst_LTLIBRARIES = libtrannoise.la

libtrannoise_la_SOURCES = \
	FastNorm3.c	\
	1-f-code.c	\
	wallace.c

AM_CPPFLAGS = @AM_CPPFLAGS@ -I$(top_srcdir)/src/include -I$(top_srcdir)/src/frontend
AM_CFLAGS = $(STATIC)

MAINTAINERCLEANFILES = Makefile.in
