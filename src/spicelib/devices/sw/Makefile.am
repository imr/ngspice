## Process this file with automake to produce Makefile.in

if DEVLIB_WANTED
lib_LTLIBRARIES = libsw.la
else
noinst_LTLIBRARIES = libsw.la
endif

libsw_la_SOURCES =	\
	sw.c		\
	swacload.c	\
	swask.c		\
	swdefs.h	\
	swdelete.c	\
	swdest.c	\
	swext.h		\
	swinit.c	\
	swinit.h	\
	switf.h		\
	swload.c	\
	swmask.c	\
	swmdel.c	\
	swmparam.c	\
	swnoise.c	\
	swparam.c	\
	swpzload.c	\
	swsetup.c	\
	swtrunc.c



AM_CPPFLAGS = @AM_CPPFLAGS@ -I$(top_srcdir)/src/include
if DEVLIB_WANTED
AM_CFLAGS = -fPIC
AM_LDFLAGS = -shared
else
AM_CFLAGS = $(STATIC)
endif
MAINTAINERCLEANFILES = Makefile.in
