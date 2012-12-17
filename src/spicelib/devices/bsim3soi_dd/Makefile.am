## Process this file with automake to produce Makefile.in

if DEVLIB_WANTED
lib_LTLIBRARIES = libbsim3soidd.la
else
noinst_LTLIBRARIES = libbsim3soidd.la
endif

libbsim3soidd_la_SOURCES = \
	b3soidd.c	\
	b3soiddacld.c	\
	b3soiddask.c	\
	b3soiddcheck.c	\
	b3soiddcvtest.c	\
	b3soidddel.c	\
	b3soidddest.c	\
	b3soiddgetic.c	\
	b3soiddld.c	\
	b3soiddmask.c	\
	b3soiddmdel.c	\
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
if DEVLIB_WANTED
AM_CFLAGS = -fPIC
AM_LDFLAGS = -shared
else
AM_CFLAGS = $(STATIC)
endif
MAINTAINERCLEANFILES = Makefile.in

EXTRA_DIST = BsimTerms_use
