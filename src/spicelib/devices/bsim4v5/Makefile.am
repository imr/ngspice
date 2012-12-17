## Process this file with automake to produce Makefile.in

if DEVLIB_WANTED
lib_LTLIBRARIES = libbsim4v5.la
else
noinst_LTLIBRARIES = libbsim4v5.la
endif

libbsim4v5_la_SOURCES =	\
	b4v5.c		\
	b4v5acld.c	\
	b4v5ask.c	\
	b4v5check.c	\
	b4v5cvtest.c	\
	b4v5del.c	\
	b4v5dest.c	\
	b4v5geo.c	\
	b4v5getic.c	\
	b4v5ld.c	\
	b4v5mask.c	\
	b4v5mdel.c	\
	b4v5mpar.c	\
	b4v5noi.c	\
	b4v5par.c	\
	b4v5pzld.c	\
	b4v5set.c	\
	b4v5temp.c	\
	b4v5trunc.c	\
	bsim4v5def.h	\
	bsim4v5ext.h	\
	bsim4v5init.c	\
	bsim4v5init.h	\
	bsim4v5itf.h



AM_CPPFLAGS = @AM_CPPFLAGS@ -I$(top_srcdir)/src/include
if DEVLIB_WANTED
AM_CFLAGS = -fPIC
AM_LDFLAGS = -shared
else
AM_CFLAGS = $(STATIC)
endif
MAINTAINERCLEANFILES = Makefile.in

EXTRA_DIST = B4TERMS_OF_USE
