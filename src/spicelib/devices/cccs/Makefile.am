## Process this file with automake to produce Makefile.in

if DEVLIB_WANTED
lib_LTLIBRARIES = libcccs.la
else
noinst_LTLIBRARIES = libcccs.la
endif

libcccs_la_SOURCES =	\
	cccs.c		\
	cccsask.c	\
	cccsdefs.h	\
	cccsdel.c	\
	cccsdest.c	\
	cccsext.h	\
	cccsinit.c	\
	cccsinit.h	\
	cccsitf.h	\
	cccsload.c	\
	cccsmdel.c	\
	cccspar.c	\
	cccspzld.c	\
	cccssacl.c	\
	cccsset.c	\
	cccssld.c	\
	cccssprt.c	\
	cccssset.c



AM_CPPFLAGS = @AM_CPPFLAGS@ -I$(top_srcdir)/src/include
if DEVLIB_WANTED
AM_CFLAGS = -fPIC
AM_LDFLAGS = -shared
else
AM_CFLAGS = $(STATIC)
endif
MAINTAINERCLEANFILES = Makefile.in
