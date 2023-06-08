## Process this file with automake to produce Makefile.in

noinst_LTLIBRARIES = libsparse.la

libsparse_la_SOURCES = \
	spAllocate.c	\
	spBuild.c	\
	spConfig.h	\
	spDefs.h	\
	spExtra.c	\
	spFactor.c	\
	spOutput.c	\
	spSMP.c		\
	spSolve.c	\
	spUtils.c




AM_CPPFLAGS = @AM_CPPFLAGS@ -I$(top_srcdir)/src/include
AM_CFLAGS = $(STATIC)

MAINTAINERCLEANFILES = Makefile.in
