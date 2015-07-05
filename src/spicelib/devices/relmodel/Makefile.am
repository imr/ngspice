## Process this file with automake to produce Makefile.in

noinst_LTLIBRARIES = librelmodel.la

if RELAN_WANTED
librelmodel_la_SOURCES =	\
	relmodel.c		\
	relmodelcalcaging.c	\
	relmodelmask.c		\
	relmodelmpar.c		\
	relmodeldefs.h		\
	relmodelext.h		\
	relmodelinit.c		\
	relmodelinit.h		\
	relmodelitf.h		\
	relmodelsetup.c
endif


AM_CPPFLAGS = @AM_CPPFLAGS@ -I$(top_srcdir)/src/include
AM_CFLAGS = $(STATIC)

MAINTAINERCLEANFILES = Makefile.in
