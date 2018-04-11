## Process this file with automake to produce Makefile.in

noinst_LTLIBRARIES = libmes.la

libmes_la_SOURCES =	\
	mes.c		\
	mesacl.c	\
	mesask.c	\
	mesdefs.h	\
	mesdisto.c	\
	mesdset.c	\
	mesext.h	\
	mesgetic.c	\
	mesinit.c	\
	mesinit.h	\
	mesitf.h	\
	mesload.c	\
	mesmask.c	\
	mesmpar.c	\
	mesnoise.c	\
	mesparam.c	\
	mespzld.c	\
	messetup.c	\
	mestemp.c	\
	mestrunc.c



AM_CPPFLAGS = @AM_CPPFLAGS@ -I$(top_srcdir)/src/include
AM_CFLAGS = $(STATIC)
MAINTAINERCLEANFILES = Makefile.in
