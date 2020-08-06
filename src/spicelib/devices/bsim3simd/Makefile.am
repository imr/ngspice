## Process this file with automake to produce Makefile.in

noinst_LTLIBRARIES = libbsim3simd.la

libbsim3simd_la_SOURCES =	\
	b3.c		\
	b3acld.c	\
	b3ask.c		\
	b3check.c	\
	b3cvtest.c	\
	b3getic.c	\
	b3ld.c		\
	b3mask.c	\
	b3mdel.c	\
	b3mpar.c	\
	b3noi.c		\
	b3par.c		\
	b3pzld.c	\
	b3set.c		\
	b3soachk.c	\
	b3temp.c	\
	b3trunc.c	\
	bsim3def.h	\
	bsim3ext.h	\
	bsim3init.c	\
	bsim3init.h	\
	bsim3itf.h	\
	b3ldseq.c	\
	b3ldsel.c	\
	b3ldsimd.c
	



AM_CPPFLAGS = @AM_CPPFLAGS@ -I$(top_srcdir)/src/include
AM_CFLAGS = $(STATIC) $(SIMD_CFLAGS)

MAINTAINERCLEANFILES = Makefile.in

EXTRA_DIST = B3TERMS_OF_USE
