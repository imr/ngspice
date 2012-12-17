## Process this file with automake to produce Makefile.in

if DEVLIB_WANTED
lib_LTLIBRARIES = libbsim3.la
else
noinst_LTLIBRARIES = libbsim3.la
endif

libbsim3_la_SOURCES =	\
	b3.c		\
	b3acld.c	\
	b3ask.c		\
	b3check.c	\
	b3cvtest.c	\
	b3del.c		\
	b3dest.c	\
	b3getic.c	\
	b3ld.c		\
	b3mask.c	\
	b3mdel.c	\
	b3mpar.c	\
	b3noi.c		\
	b3par.c		\
	b3pzld.c	\
	b3set.c		\
	b3temp.c	\
	b3trunc.c	\
	bsim3def.h	\
	bsim3ext.h	\
	bsim3init.c	\
	bsim3init.h	\
	bsim3itf.h



AM_CPPFLAGS = @AM_CPPFLAGS@ -I$(top_srcdir)/src/include
if DEVLIB_WANTED
AM_CFLAGS = -fPIC
AM_LDFLAGS = -shared
else
AM_CFLAGS = $(STATIC)
endif
MAINTAINERCLEANFILES = Makefile.in

EXTRA_DIST = B3TERMS_OF_USE
