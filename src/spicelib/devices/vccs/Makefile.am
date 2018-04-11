## Process this file with automake to produce Makefile.in

noinst_LTLIBRARIES = libvccs.la

libvccs_la_SOURCES =	\
	vccs.c		\
	vccsask.c	\
	vccsdefs.h	\
	vccsext.h	\
	vccsinit.c	\
	vccsinit.h	\
	vccsitf.h	\
	vccsload.c	\
	vccspar.c	\
	vccspzld.c	\
	vccssacl.c	\
	vccsset.c	\
	vccssld.c	\
	vccssprt.c	\
	vccssset.c



AM_CPPFLAGS = @AM_CPPFLAGS@ -I$(top_srcdir)/src/include
AM_CFLAGS = $(STATIC)
MAINTAINERCLEANFILES = Makefile.in
