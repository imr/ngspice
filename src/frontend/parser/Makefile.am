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
		std.c		\
		unixcom.c	\
		unixcom.h


AM_CPPFLAGS = @AM_CPPFLAGS@ -I$(top_srcdir)/src/include
AM_CFLAGS = $(STATIC)
MAINTAINERCLEANFILES = Makefile.in
