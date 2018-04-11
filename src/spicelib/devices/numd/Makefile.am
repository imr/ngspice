## Process this file with automake to produce Makefile.in

noinst_LTLIBRARIES = libnumd.la

libnumd_la_SOURCES =	\
	numd.c		\
	numdacld.c	\
	numdask.c	\
	numddefs.h	\
	numddel.c	\
	numddump.c	\
	numdext.h	\
	numdinit.c	\
	numdinit.h	\
	numditf.h	\
	numdload.c	\
	numdmpar.c	\
	numdparm.c	\
	numdpzld.c	\
	numdset.c	\
	numdtemp.c	\
	numdtrun.c


AM_CPPFLAGS = @AM_CPPFLAGS@ -I$(top_srcdir)/src/include
AM_CFLAGS = $(STATIC)

MAINTAINERCLEANFILES = Makefile.in
