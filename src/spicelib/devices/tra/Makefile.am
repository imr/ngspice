## Process this file with automake to produce Makefile.in

if DEVLIB_WANTED
lib_LTLIBRARIES = libtra.la
else
noinst_LTLIBRARIES = libtra.la
endif

libtra_la_SOURCES =	\
	tra.c		\
	traacct.c	\
	traacld.c	\
	traask.c	\
	tradefs.h	\
	tradel.c	\
	tradest.c	\
	traext.h	\
	trainit.c	\
	trainit.h	\
	traitf.h	\
	traload.c	\
	tramdel.c	\
	traparam.c	\
	trasetup.c	\
	tratemp.c	\
	tratrunc.c


AM_CPPFLAGS = @AM_CPPFLAGS@ -I$(top_srcdir)/src/include
if DEVLIB_WANTED
AM_CFLAGS = -fPIC
AM_LDFLAGS = -shared
else
AM_CFLAGS = $(STATIC)
endif
MAINTAINERCLEANFILES = Makefile.in
