## Process this file with automake to produce Makefile.in

if DEVLIB_WANTED
lib_LTLIBRARIES = libndev.la
else
noinst_LTLIBRARIES = libndev.la
endif

libndev_la_SOURCES =	\
	ndev.c		\
	ndevacld.c	\
	ndevaccept.c	\
	ndevask.c	\
	ndevdefs.h	\
	ndevdel.c	\
	ndevdest.c	\
	ndevdump.c	\
	ndevext.h	\
	ndevexch.h	\
	ndevinit.c	\
	ndevinit.h	\
	ndevitf.h	\
	ndevload.c	\
	ndevmdel.c	\
	ndevmpar.c	\
	ndevparm.c	\
	ndevpzld.c	\
	ndevset.c	\
	ndevtemp.c	\
	ndevtrun.c


AM_CPPFLAGS = @AM_CPPFLAGS@ -I$(top_srcdir)/src/include
if DEVLIB_WANTED
AM_CFLAGS = -fPIC
AM_LDFLAGS = -shared
else
AM_CFLAGS = $(STATIC)
endif
MAINTAINERCLEANFILES = Makefile.in
