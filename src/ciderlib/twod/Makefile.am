## Process this file with automake to produce Makefile.in

noinst_LTLIBRARIES = libcidertwod.la

libcidertwod_la_SOURCES =	\
		twoadmit.c	\
		twoaval.c	\
		twocond.c	\
		twocont.c	\
		twocurr.c	\
		twoddefs.h	\
		twodest.c	\
		twodext.h	\
		twodopng.c	\
		twoelect.c	\
		twofield.c	\
		twomesh.c	\
		twomobdv.c	\
		twomobfn.c	\
		twomobil.c	\
		twoncont.c	\
		twopcont.c	\
		twopoiss.c	\
		twoprint.c	\
		twoproj.c	\
		tworead.c	\
		twosetbc.c	\
		twosetup.c	\
		twosolve.c

EXTRA_DIST = readme
AM_CPPFLAGS = @AM_CPPFLAGS@ -I$(top_srcdir)/src/include
AM_CFLAGS = $(STATIC)
MAINTAINERCLEANFILES = Makefile.in
