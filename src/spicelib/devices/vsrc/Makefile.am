## Process this file with automake to produce Makefile.in

noinst_LTLIBRARIES = libvsrc.la

libvsrc_la_SOURCES =	\
	vsrc.c		\
	vsrcacct.c	\
	vsrcacld.c	\
	vsrcask.c	\
	vsrcdefs.h	\
	vsrcdel.c	\
	vsrcext.h	\
	vsrcfbr.c	\
	vsrcinit.c	\
	vsrcinit.h	\
	vsrcitf.h	\
	vsrcload.c	\
	vsrcpar.c	\
	vsrcpzld.c	\
	vsrcpzs.c	\
	vsrcset.c	\
	vsrctemp.c



AM_CPPFLAGS = @AM_CPPFLAGS@ -I$(top_srcdir)/src/include
AM_CFLAGS = $(STATIC)

MAINTAINERCLEANFILES = Makefile.in
