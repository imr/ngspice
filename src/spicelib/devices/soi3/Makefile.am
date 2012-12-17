## Process this file with automake to produce Makefile.in

if DEVLIB_WANTED
lib_LTLIBRARIES = libsoi3.la
else
noinst_LTLIBRARIES = libsoi3.la
endif

libsoi3_la_SOURCES =	\
	soi3.c		\
	soi3acld.c	\
	soi3ask.c	\
	soi3cap.c	\
	soi3conv.c	\
	soi3defs.h	\
	soi3del.c	\
	soi3dest.c	\
	soi3ext.h	\
	soi3ic.c	\
	soi3init.c	\
	soi3init.h	\
	soi3itf.h	\
	soi3load.c	\
	soi3mask.c	\
	soi3mdel.c	\
	soi3mpar.c	\
	soi3nois.c	\
	soi3par.c	\
	soi3set.c	\
	soi3temp.c	\
	soi3trun.c



AM_CPPFLAGS = @AM_CPPFLAGS@ -I$(top_srcdir)/src/include
if DEVLIB_WANTED
AM_CFLAGS = -fPIC
AM_LDFLAGS = -shared
else
AM_CFLAGS = $(STATIC)
endif
MAINTAINERCLEANFILES = Makefile.in
