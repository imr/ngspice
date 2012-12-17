## Process this file with automake to produce Makefile.in

if DEVLIB_WANTED
lib_LTLIBRARIES = libmes.la
else
noinst_LTLIBRARIES = libmes.la
endif

libmes_la_SOURCES =	\
	mes.c		\
	mesacl.c	\
	mesask.c	\
	mesdefs.h	\
	mesdel.c	\
	mesdest.c	\
	mesdisto.c	\
	mesdset.c	\
	mesext.h	\
	mesgetic.c	\
	mesinit.c	\
	mesinit.h	\
	mesitf.h	\
	mesload.c	\
	mesmask.c	\
	mesmdel.c	\
	mesmpar.c	\
	mesnoise.c	\
	mesparam.c	\
	mespzld.c	\
	messetup.c	\
	mestemp.c	\
	mestrunc.c



AM_CPPFLAGS = @AM_CPPFLAGS@ -I$(top_srcdir)/src/include
if DEVLIB_WANTED
AM_CFLAGS = -fPIC
AM_LDFLAGS = -shared
else
AM_CFLAGS = $(STATIC)
endif
MAINTAINERCLEANFILES = Makefile.in
