## Process this file with automake to produce Makefile.in

noinst_LTLIBRARIES = libbsim3v32.la

libbsim3v32_la_SOURCES = \
	b3v32.c		\
	b3v32acld.c	\
	b3v32ask.c	\
	b3v32check.c	\
	b3v32cvtest.c	\
	b3v32getic.c	\
	b3v32ld.c	\
	b3v32mask.c	\
	b3v32mdel.c	\
	b3v32mpar.c	\
	b3v32noi.c	\
	b3v32par.c	\
	b3v32pzld.c	\
	b3v32set.c	\
	b3v32soachk.c	\
	b3v32temp.c	\
	b3v32trunc.c	\
	bsim3v32def.h	\
	bsim3v32ext.h	\
	bsim3v32init.c	\
	bsim3v32init.h	\
	bsim3v32itf.h



AM_CPPFLAGS = @AM_CPPFLAGS@ -I$(top_srcdir)/src/include
AM_CFLAGS = $(STATIC)

MAINTAINERCLEANFILES = Makefile.in

EXTRA_DIST = B3TERMS_OF_USE
