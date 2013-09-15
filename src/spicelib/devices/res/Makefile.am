## Process this file with automake to produce Makefile.in

noinst_LTLIBRARIES = libres.la

libres_la_SOURCES =	\
	res.c		\
	resask.c	\
	resdefs.h	\
	resdel.c	\
	resdest.c	\
	resext.h	\
	resinit.c	\
	resinit.h	\
	resitf.h	\
	resload.c	\
	resmask.c	\
	resmdel.c	\
	resmpar.c	\
	resnoise.c	\
	resparam.c	\
	respzld.c	\
	ressacl.c	\
	ressetup.c	\
	ressload.c	\
	ressprt.c	\
	ressset.c	\
	restemp.c


if KLU_WANTED
libres_la_SOURCES += resbindCSC.c
endif

if SuperLU_WANTED
libres_la_SOURCES += resbindCSC.c
endif

if UMFPACK_WANTED
libres_la_SOURCES += resbindCSC.c
endif

if KIRCHHOFF_WANTED
libres_la_SOURCES += resnode.c
endif

AM_CPPFLAGS = @AM_CPPFLAGS@ -I$(top_srcdir)/src/include
AM_CFLAGS = $(STATIC)

MAINTAINERCLEANFILES = Makefile.in
