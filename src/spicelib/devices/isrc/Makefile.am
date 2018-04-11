## Process this file with automake to produce Makefile.in

noinst_LTLIBRARIES = libisrc.la

libisrc_la_SOURCES =	\
	isrc.c		\
	isrcacct.c	\
	isrcacld.c	\
	isrcask.c	\
	isrcdefs.h	\
	isrcdel.c	\
	isrcext.h	\
	isrcinit.c	\
	isrcinit.h	\
	isrcitf.h	\
	isrcload.c	\
	isrcpar.c	\
	isrctemp.c



AM_CPPFLAGS = @AM_CPPFLAGS@ -I$(top_srcdir)/src/include
AM_CFLAGS = $(STATIC)
MAINTAINERCLEANFILES = Makefile.in
