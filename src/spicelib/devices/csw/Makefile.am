## Process this file with automake to produce Makefile.in

if DEVLIB_WANTED
lib_LTLIBRARIES = libcsw.la
else
noinst_LTLIBRARIES = libcsw.la
endif

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
if DEVLIB_WANTED
AM_CFLAGS = -fPIC
AM_LDFLAGS = -shared
else
AM_CFLAGS = $(STATIC)
endif
MAINTAINERCLEANFILES = Makefile.in
