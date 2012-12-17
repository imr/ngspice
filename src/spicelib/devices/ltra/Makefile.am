## Process this file with automake to produce Makefile.in

if DEVLIB_WANTED
lib_LTLIBRARIES = libltra.la
else
noinst_LTLIBRARIES = libltra.la
endif

libltra_la_SOURCES =	\
	ltra.c		\
	ltraacct.c	\
	ltraacld.c	\
	ltraask.c	\
	ltradefs.h	\
	ltradel.c	\
	ltradest.c	\
	ltraext.h	\
	ltrainit.c	\
	ltrainit.h	\
	ltraitf.h	\
	ltraload.c	\
	ltramask.c	\
	ltramdel.c	\
	ltramisc.c	\
	ltrampar.c	\
	ltrapar.c	\
	ltraset.c	\
	ltratemp.c	\
	ltratrun.c



AM_CPPFLAGS = @AM_CPPFLAGS@ -I$(top_srcdir)/src/include
if DEVLIB_WANTED
AM_CFLAGS = -fPIC
AM_LDFLAGS = -shared
else
AM_CFLAGS = $(STATIC)
endif
MAINTAINERCLEANFILES = Makefile.in
