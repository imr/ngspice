## Process this file with automake to produce Makefile.in

if DEVLIB_WANTED
lib_LTLIBRARIES = libisrc.la
else
noinst_LTLIBRARIES = libisrc.la
endif

libisrc_la_SOURCES =	\
	isrc.c		\
	isrcacct.c	\
	isrcacld.c	\
	isrcask.c	\
	isrcdefs.h	\
	isrcdel.c	\
	isrcdest.c	\
	isrcext.h	\
	isrcinit.c	\
	isrcinit.h	\
	isrcitf.h	\
	isrcload.c	\
	isrcmdel.c	\
	isrcpar.c	\
	isrctemp.c



AM_CPPFLAGS = @AM_CPPFLAGS@ -I$(top_srcdir)/src/include
if DEVLIB_WANTED
AM_CFLAGS = -fPIC
AM_LDFLAGS = -shared
else
AM_CFLAGS = $(STATIC)
endif
MAINTAINERCLEANFILES = Makefile.in
