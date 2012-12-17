## Process this file with automake to produce Makefile.in

if DEVLIB_WANTED
lib_LTLIBRARIES = libccvs.la
else
noinst_LTLIBRARIES = libccvs.la
endif

libccvs_la_SOURCES =	\
	ccvs.c		\
	ccvsask.c	\
	ccvsdefs.h	\
	ccvsdel.c	\
	ccvsdest.c	\
	ccvsext.h	\
	ccvsfbr.c	\
	ccvsinit.c	\
	ccvsinit.h	\
	ccvsitf.h	\
	ccvsload.c	\
	ccvsmdel.c	\
	ccvspar.c	\
	ccvspzld.c	\
	ccvssacl.c	\
	ccvsset.c	\
	ccvssld.c	\
	ccvssprt.c	\
	ccvssset.c



AM_CPPFLAGS = @AM_CPPFLAGS@ -I$(top_srcdir)/src/include
if DEVLIB_WANTED
AM_CFLAGS = -fPIC
AM_LDFLAGS = -shared
else
AM_CFLAGS = $(STATIC)
endif
MAINTAINERCLEANFILES = Makefile.in
