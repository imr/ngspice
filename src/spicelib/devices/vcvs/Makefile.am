## Process this file with automake to produce Makefile.in

if DEVLIB_WANTED
lib_LTLIBRARIES = libvcvs.la
else
noinst_LTLIBRARIES = libvcvs.la
endif

libvcvs_la_SOURCES =	\
	vcvs.c		\
	vcvsask.c	\
	vcvsdefs.h	\
	vcvsdel.c	\
	vcvsdest.c	\
	vcvsext.h	\
	vcvsfbr.c	\
	vcvsinit.c	\
	vcvsinit.h	\
	vcvsitf.h	\
	vcvsload.c	\
	vcvsmdel.c	\
	vcvspar.c	\
	vcvspzld.c	\
	vcvssacl.c	\
	vcvsset.c	\
	vcvssld.c	\
	vcvssprt.c	\
	vcvssset.c



AM_CPPFLAGS = @AM_CPPFLAGS@ -I$(top_srcdir)/src/include
if DEVLIB_WANTED
AM_CFLAGS = -fPIC
AM_LDFLAGS = -shared
else
AM_CFLAGS = $(STATIC)
endif
MAINTAINERCLEANFILES = Makefile.in
