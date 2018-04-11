## Process this file with automake to produce Makefile.in

noinst_LTLIBRARIES = libbsim3soipd.la

libbsim3soipd_la_SOURCES = \
	b3soipd.c	\
	b3soipdacld.c	\
	b3soipdask.c	\
	b3soipdcheck.c	\
	b3soipdcvtest.c	\
	b3soipdgetic.c	\
	b3soipdld.c	\
	b3soipdmask.c	\
	b3soipdmpar.c	\
	b3soipdnoi.c	\
	b3soipdpar.c	\
	b3soipdpzld.c	\
	b3soipdset.c	\
	b3soipdtemp.c	\
	b3soipdtrunc.c	\
	b3soipddef.h	\
	b3soipdext.h	\
	b3soipdinit.c	\
	b3soipdinit.h	\
	b3soipditf.h



AM_CPPFLAGS = @AM_CPPFLAGS@ -I$(top_srcdir)/src/include
AM_CFLAGS = $(STATIC)

MAINTAINERCLEANFILES = Makefile.in

EXTRA_DIST = BsimTerms_use
