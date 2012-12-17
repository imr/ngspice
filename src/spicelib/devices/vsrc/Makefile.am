## Process this file with automake to produce Makefile.in

if DEVLIB_WANTED
lib_LTLIBRARIES = libvsrc.la
else
noinst_LTLIBRARIES = libvsrc.la
endif

libvsrc_la_SOURCES =	\
	vsrc.c		\
	vsrcacct.c	\
	vsrcacld.c	\
	vsrcask.c	\
	vsrcdefs.h	\
	vsrcdel.c	\
	vsrcdest.c	\
	vsrcext.h	\
	vsrcfbr.c	\
	vsrcinit.c	\
	vsrcinit.h	\
	vsrcitf.h	\
	vsrcload.c	\
	vsrcmdel.c	\
	vsrcpar.c	\
	vsrcpzld.c	\
	vsrcpzs.c	\
	vsrcset.c	\
	vsrctemp.c



AM_CPPFLAGS = @AM_CPPFLAGS@ -I$(top_srcdir)/src/include
if DEVLIB_WANTED
AM_CFLAGS = -fPIC
AM_LDFLAGS = -shared
else
AM_CFLAGS = $(STATIC)
endif
MAINTAINERCLEANFILES = Makefile.in
