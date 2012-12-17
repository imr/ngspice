## Process this file with automake to produce Makefile.in

if DEVLIB_WANTED
lib_LTLIBRARIES = libasrc.la
else
noinst_LTLIBRARIES = libasrc.la
endif

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
	asrcmdel.c	\
	asrcpar.c	\
	asrcpzld.c	\
	asrcset.c \
	asrctemp.c


AM_CPPFLAGS = @AM_CPPFLAGS@ -I$(top_srcdir)/src/include
if DEVLIB_WANTED
AM_CFLAGS = -fPIC
AM_LDFLAGS = -shared
else
AM_CFLAGS = $(STATIC)
endif
MAINTAINERCLEANFILES = Makefile.in
