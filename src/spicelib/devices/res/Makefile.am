## Process this file with automake to produce Makefile.in

if DEVLIB_WANTED
lib_LTLIBRARIES = libres.la
else
noinst_LTLIBRARIES = libres.la
endif

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



AM_CPPFLAGS = @AM_CPPFLAGS@ -I$(top_srcdir)/src/include
if DEVLIB_WANTED
AM_CFLAGS = -fPIC
AM_LDFLAGS = -shared
else
AM_CFLAGS = $(STATIC)
endif
MAINTAINERCLEANFILES = Makefile.in
