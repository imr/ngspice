## Process this file with automake to produce Makefile.in

noinst_LTLIBRARIES = libcideroned.la

libcideroned_la_SOURCES =	\
		oneadmit.c	\
		oneaval.c	\
		onecond.c	\
		onecont.c	\
		oneddefs.h	\
		onedest.c	\
		onedext.h	\
		onedopng.c	\
		onefreez.c	\
		onemesh.c	\
		onepoiss.c	\
		oneprint.c	\
		oneproj.c	\
		oneread.c	\
		onesetup.c	\
		onesolve.c

EXTRA_DIST = notes readme
AM_CPPFLAGS = @AM_CPPFLAGS@ -I$(top_srcdir)/src/include
AM_CFLAGS = $(STATIC)
MAINTAINERCLEANFILES = Makefile.in
