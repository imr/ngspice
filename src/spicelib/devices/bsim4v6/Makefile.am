## Process this file with automake to produce Makefile.in

noinst_LTLIBRARIES = libbsim4v6.la

libbsim4v6_la_SOURCES =	\
	b4v6.c		\
	b4v6acld.c	\
	b4v6ask.c		\
	b4v6check.c	\
	b4v6cvtest.c	\
	b4v6geo.c		\
	b4v6getic.c	\
	b4v6ld.c		\
	b4v6mask.c	\
	b4v6mdel.c	\
	b4v6mpar.c	\
	b4v6noi.c		\
	b4v6par.c		\
	b4v6pzld.c	\
	b4v6set.c		\
	b4v6soachk.c	\
	b4v6temp.c	\
	b4v6trunc.c	\
	bsim4v6def.h	\
	bsim4v6ext.h	\
	bsim4v6init.c	\
	bsim4v6init.h	\
	bsim4v6itf.h



AM_CPPFLAGS = @AM_CPPFLAGS@ -I$(top_srcdir)/src/include
AM_CFLAGS = $(STATIC)

MAINTAINERCLEANFILES = Makefile.in

EXTRA_DIST = B4TERMS_OF_USE
