## Process this file with automake to produce Makefile.in

noinst_LTLIBRARIES = liburc.la

liburc_la_SOURCES =	\
	urc.c		\
	urcask.c	\
	urcdefs.h	\
	urcdel.c	\
	urcdest.c	\
	urcext.h	\
	urcinit.c	\
	urcinit.h	\
	urcitf.h	\
	urcmask.c	\
	urcmdel.c	\
	urcmpar.c	\
	urcparam.c	\
	urcsetup.c



AM_CPPFLAGS = @AM_CPPFLAGS@ -I$(top_srcdir)/src/include
AM_CFLAGS = $(STATIC)
MAINTAINERCLEANFILES = Makefile.in
