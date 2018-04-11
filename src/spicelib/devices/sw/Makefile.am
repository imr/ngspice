## Process this file with automake to produce Makefile.in

noinst_LTLIBRARIES = libsw.la

libsw_la_SOURCES =	\
	sw.c		\
	swacload.c	\
	swask.c		\
	swdefs.h	\
	swext.h		\
	swinit.c	\
	swinit.h	\
	switf.h		\
	swload.c	\
	swmask.c	\
	swmparam.c	\
	swnoise.c	\
	swparam.c	\
	swpzload.c	\
	swsetup.c	\
	swtrunc.c



AM_CPPFLAGS = @AM_CPPFLAGS@ -I$(top_srcdir)/src/include
AM_CFLAGS = $(STATIC)
MAINTAINERCLEANFILES = Makefile.in
