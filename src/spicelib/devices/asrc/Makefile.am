## Process this file with automake to produce Makefile.in

noinst_LTLIBRARIES = libasrc.la

libasrc_la_SOURCES =	\
	asrc.c		\
	asrcacld.c	\
	asrcask.c	\
	asrcconv.c	\
	asrcdefs.h	\
	asrcdel.c	\
	asrcdest.c	\
	asrcext.h	\
	asrcfbr.c	\
	asrcitf.h	\
	asrcinit.c	\
	asrcinit.h	\
	asrcload.c	\
	asrcpar.c	\
	asrcpzld.c	\
	asrcset.c \
	asrctemp.c


AM_CPPFLAGS = @AM_CPPFLAGS@ -I$(top_srcdir)/src/include
AM_CFLAGS = $(STATIC)
MAINTAINERCLEANFILES = Makefile.in
