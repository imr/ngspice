## Process this file with automake to produce Makefile.in

noinst_LTLIBRARIES = libparser.la

libparser_la_SOURCES = \
		backq.c		\
		backq.h		\
		complete.c	\
		complete.h	\
		cshpar.c	\
		cshpar.h	\
		glob.c		\
		glob.h		\
		input.c		\
		input.h		\
		lexical.c	\
		lexical.h	\
		numparse.c	\
		numparse.h	\
		unixcom.c	\
		unixcom.h


if !HAVE_QSORT
libparser_la_SOURCES += std.c
endif

AM_CPPFLAGS = @AM_CPPFLAGS@ -I$(top_srcdir)/src/include
AM_CFLAGS = $(STATIC)
MAINTAINERCLEANFILES = Makefile.in
