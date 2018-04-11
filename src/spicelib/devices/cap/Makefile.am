## Process this file with automake to produce Makefile.in

noinst_LTLIBRARIES = libcap.la

libcap_la_SOURCES =	\
	cap.c		\
	capacld.c	\
	capask.c	\
	capdefs.h	\
	capext.h	\
	capgetic.c	\
	capinit.c	\
	capinit.h	\
	capitf.h	\
	capload.c	\
	capmask.c	\
	capmpar.c	\
	capparam.c	\
	cappzld.c	\
	capsacl.c	\
	capsetup.c	\
	capsload.c	\
	capsoachk.c	\
	capsprt.c	\
	capsset.c	\
	capsupd.c	\
	captemp.c	\
	captrunc.c



AM_CPPFLAGS = @AM_CPPFLAGS@ -I$(top_srcdir)/src/include
AM_CFLAGS = $(STATIC)

MAINTAINERCLEANFILES = Makefile.in
