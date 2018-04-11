## Process this file with automake to produce Makefile.in

noinst_LTLIBRARIES = libvcvs.la

libvcvs_la_SOURCES =	\
	vcvs.c		\
	vcvsask.c	\
	vcvsdefs.h	\
	vcvsext.h	\
	vcvsfbr.c	\
	vcvsinit.c	\
	vcvsinit.h	\
	vcvsitf.h	\
	vcvsload.c	\
	vcvspar.c	\
	vcvspzld.c	\
	vcvssacl.c	\
	vcvsset.c	\
	vcvssld.c	\
	vcvssprt.c	\
	vcvssset.c



AM_CPPFLAGS = @AM_CPPFLAGS@ -I$(top_srcdir)/src/include
AM_CFLAGS = $(STATIC)
MAINTAINERCLEANFILES = Makefile.in
