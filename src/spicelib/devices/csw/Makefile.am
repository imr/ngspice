## Process this file with automake to produce Makefile.in

noinst_LTLIBRARIES = libcsw.la

libcsw_la_SOURCES =	\
	csw.c		\
	cswacld.c	\
	cswask.c	\
	cswdefs.h	\
	cswdel.c	\
	cswdest.c	\
	cswext.h	\
	cswinit.c	\
	cswinit.h	\
	cswitf.h	\
	cswload.c	\
	cswmask.c	\
	cswmdel.c	\
	cswmpar.c	\
	cswnoise.c	\
	cswparam.c	\
	cswpzld.c	\
	cswsetup.c	\
	cswtrunc.c



AM_CPPFLAGS = @AM_CPPFLAGS@ -I$(top_srcdir)/src/include
AM_CFLAGS = $(STATIC)
MAINTAINERCLEANFILES = Makefile.in
