## Process this file with automake to produce Makefile.in

noinst_LTLIBRARIES = libbalun.la

libbalun_la_SOURCES =	\
	balun.c		\
	balunask.c	\
	balundefs.h	\
	balunext.h	\
	baluninit.c	\
	baluninit.h	\
	balunitf.h	\
	balunload.c	\
	balunpar.c	\
	balunpzld.c	\
	balunset.c	



AM_CPPFLAGS = @AM_CPPFLAGS@ -I$(top_srcdir)/src/include
AM_CFLAGS = $(STATIC)
MAINTAINERCLEANFILES = Makefile.in
