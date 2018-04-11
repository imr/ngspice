## Process this file with automake to produce Makefile.in

noinst_LTLIBRARIES = libtra.la

libtra_la_SOURCES =	\
	tra.c		\
	traacct.c	\
	traacld.c	\
	traask.c	\
	tradefs.h	\
	traext.h	\
	trainit.c	\
	trainit.h	\
	traitf.h	\
	traload.c	\
	traparam.c	\
	trasetup.c	\
	tratemp.c	\
	tratrunc.c


AM_CPPFLAGS = @AM_CPPFLAGS@ -I$(top_srcdir)/src/include
AM_CFLAGS = $(STATIC)
MAINTAINERCLEANFILES = Makefile.in
