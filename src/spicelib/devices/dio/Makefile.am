## Process this file with automake to produce Makefile.in

if DEVLIB_WANTED
lib_LTLIBRARIES = libdio.la
else
noinst_LTLIBRARIES = libdio.la
endif

libdio_la_SOURCES =	\
	dio.c		\
	dioacld.c	\
	dioask.c	\
	dioconv.c	\
	diodefs.h	\
	diodel.c	\
	diodest.c	\
	diodisto.c	\
	diodset.c	\
	dioext.h	\
	diogetic.c	\
	dioinit.c	\
	dioinit.h	\
	dioitf.h	\
	dioload.c	\
	diomask.c	\
	diomdel.c	\
	diompar.c	\
	dionoise.c	\
	dioparam.c	\
	diopzld.c	\
	diosacl.c	\
	diosetup.c	\
	diosload.c	\
	diosprt.c	\
	diosset.c	\
	diosupd.c	\
	diotemp.c	\
	diotrunc.c


AM_CPPFLAGS = @AM_CPPFLAGS@ -I$(top_srcdir)/src/include
if DEVLIB_WANTED
AM_CFLAGS = -fPIC
AM_LDFLAGS = -shared
else
AM_CFLAGS = $(STATIC)
endif
MAINTAINERCLEANFILES = Makefile.in
