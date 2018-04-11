## Process this file with automake to produce Makefile.in

noinst_LTLIBRARIES = libccvs.la

libccvs_la_SOURCES =	\
	ccvs.c		\
	ccvsask.c	\
	ccvsdefs.h	\
	ccvsext.h	\
	ccvsfbr.c	\
	ccvsinit.c	\
	ccvsinit.h	\
	ccvsitf.h	\
	ccvsload.c	\
	ccvspar.c	\
	ccvspzld.c	\
	ccvssacl.c	\
	ccvsset.c	\
	ccvssld.c	\
	ccvssprt.c	\
	ccvssset.c



AM_CPPFLAGS = @AM_CPPFLAGS@ -I$(top_srcdir)/src/include
AM_CFLAGS = $(STATIC)

MAINTAINERCLEANFILES = Makefile.in
