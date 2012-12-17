## Process this file with automake to produce Makefile.in

if DEVLIB_WANTED
lib_LTLIBRARIES = libvccs.la
else
noinst_LTLIBRARIES = libvccs.la
endif

libvccs_la_SOURCES =	\
	vccs.c		\
	vccsask.c	\
	vccsdefs.h	\
	vccsdel.c	\
	vccsdest.c	\
	vccsext.h	\
	vccsinit.c	\
	vccsinit.h	\
	vccsitf.h	\
	vccsload.c	\
	vccsmdel.c	\
	vccspar.c	\
	vccspzld.c	\
	vccssacl.c	\
	vccsset.c	\
	vccssld.c	\
	vccssprt.c	\
	vccssset.c



AM_CPPFLAGS = @AM_CPPFLAGS@ -I$(top_srcdir)/src/include
if DEVLIB_WANTED
AM_CFLAGS = -fPIC
AM_LDFLAGS = -shared
else
AM_CFLAGS = $(STATIC)
endif
MAINTAINERCLEANFILES = Makefile.in
