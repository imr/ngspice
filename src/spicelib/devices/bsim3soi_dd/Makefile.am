## Process this file with automake to produce Makefile.in

noinst_LTLIBRARIES = libbsim3soidd.la

libbsim3soidd_la_SOURCES = \
	b3soidd.c	\
	b3soiddacld.c	\
	b3soiddask.c	\
	b3soiddcheck.c	\
	b3soiddcvtest.c	\
	b3soiddgetic.c	\
	b3soiddld.c	\
	b3soiddmask.c	\
	b3soiddmpar.c	\
	b3soiddnoi.c	\
	b3soiddpar.c	\
	b3soiddpzld.c	\
	b3soiddset.c	\
	b3soiddtemp.c	\
	b3soiddtrunc.c	\
	b3soidddef.h	\
	b3soiddext.h	\
	b3soiddinit.c	\
	b3soiddinit.h	\
	b3soidditf.h



AM_CPPFLAGS = @AM_CPPFLAGS@ -I$(top_srcdir)/src/include
AM_CFLAGS = $(STATIC)

MAINTAINERCLEANFILES = Makefile.in

EXTRA_DIST = BsimTerms_use
