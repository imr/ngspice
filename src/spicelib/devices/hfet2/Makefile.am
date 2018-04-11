## Process this file with automake to produce Makefile.in

noinst_LTLIBRARIES = libhfet2.la

libhfet2_la_SOURCES =	\
	hfet2.c		\
	hfet2acl.c	\
	hfet2ask.c	\
	hfet2defs.h	\
	hfet2ext.h	\
	hfet2getic.c	\
	hfet2init.c	\
	hfet2init.h	\
	hfet2itf.h	\
	hfet2load.c	\
	hfet2mask.c	\
	hfet2mpar.c	\
	hfet2param.c	\
	hfet2pzl.c	\
	hfet2setup.c	\
	hfet2temp.c	\
	hfet2trunc.c



AM_CPPFLAGS = @AM_CPPFLAGS@ -I$(top_srcdir)/src/include
AM_CFLAGS = $(STATIC)
MAINTAINERCLEANFILES = Makefile.in
