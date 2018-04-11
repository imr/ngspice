## Process this file with automake to produce Makefile.in

noinst_LTLIBRARIES = libnumos.la

libnumos_la_SOURCES =	\
	numm.c		\
	nummacld.c	\
	nummask.c	\
	numosdef.h	\
	nummdel.c	\
	nummdump.c	\
	numosext.h	\
	numosinit.c	\
	numosinit.h	\
	numositf.h	\
	nummload.c	\
	nummmpar.c	\
	nummparm.c	\
	nummpzld.c	\
	nummset.c	\
	nummtemp.c	\
	nummtrun.c



AM_CPPFLAGS = @AM_CPPFLAGS@ -I$(top_srcdir)/src/include
AM_CFLAGS = $(STATIC)

MAINTAINERCLEANFILES = Makefile.in
