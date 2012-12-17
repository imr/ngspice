## Process this file with automake to produce Makefile.in

if DEVLIB_WANTED
lib_LTLIBRARIES = libnumd.la
else
noinst_LTLIBRARIES = libnumd.la
endif

libnumd_la_SOURCES =	\
	numd.c		\
	numdacld.c	\
	numdask.c	\
	numddefs.h	\
	numddel.c	\
	numddest.c	\
	numddump.c	\
	numdext.h	\
	numdinit.c	\
	numdinit.h	\
	numditf.h	\
	numdload.c	\
	numdmdel.c	\
	numdmpar.c	\
	numdparm.c	\
	numdpzld.c	\
	numdset.c	\
	numdtemp.c	\
	numdtrun.c


AM_CPPFLAGS = @AM_CPPFLAGS@ -I$(top_srcdir)/src/include
if DEVLIB_WANTED
AM_CFLAGS = -fPIC
AM_LDFLAGS = -shared
else
AM_CFLAGS = $(STATIC)
endif
MAINTAINERCLEANFILES = Makefile.in
