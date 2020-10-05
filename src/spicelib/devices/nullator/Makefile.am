## Process this file with automake to produce Makefile.in

noinst_LTLIBRARIES = libnula.la

libnula_la_SOURCES =	\
	nula.c		\
	nulaask.c	\
	nuladefs.h	\
	nulaext.h	\
	nulainit.c	\
	nulainit.h	\
	nulaitf.h	\
	nulaload.c	\
	nulapar.c	\
	nulaset.c	\
	nulapzld.c

#       nulafbr.c	
#	nulapzld.c	\
#	nulasacl.c	\
#	nulaset.c	\
#	nulasld.c	\
#	nulasprt.c	\
#	nulasset.c



AM_CPPFLAGS = @AM_CPPFLAGS@ -I$(top_srcdir)/src/include
AM_CFLAGS = $(STATIC)
MAINTAINERCLEANFILES = Makefile.in
