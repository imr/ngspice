## Process this file with automake to produce Makefile.in

if DEVLIB_WANTED
lib_LTLIBRARIES = libbsim1.la
else
noinst_LTLIBRARIES = libbsim1.la
endif

libbsim1_la_SOURCES =	\
	b1.c		\
	b1acld.c	\
	b1ask.c		\
	b1cvtest.c	\
	b1del.c		\
	b1dest.c	\
	b1disto.c	\
	b1dset.c	\
	b1eval.c	\
	b1getic.c	\
	b1ld.c		\
	b1mask.c	\
	b1mdel.c	\
	b1moscap.c	\
	b1mpar.c	\
	b1par.c		\
	b1noi.c		\
	b1pzld.c	\
	b1set.c		\
	b1temp.c	\
	b1trunc.c	\
	bsim1def.h	\
	bsim1ext.h	\
	bsim1init.c	\
	bsim1init.h	\
	bsim1itf.h



AM_CPPFLAGS = @AM_CPPFLAGS@ -I$(top_srcdir)/src/include
if DEVLIB_WANTED
AM_CFLAGS = -fPIC
AM_LDFLAGS = -shared
else
AM_CFLAGS = $(STATIC)
endif
MAINTAINERCLEANFILES = Makefile.in
