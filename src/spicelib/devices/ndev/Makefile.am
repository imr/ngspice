## Process this file with automake to produce Makefile.in

noinst_LTLIBRARIES = libndev.la

libndev_la_SOURCES =	\
	ndev.c		\
	ndevacld.c	\
	ndevaccept.c	\
	ndevask.c	\
	ndevdefs.h	\
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
AM_CFLAGS = $(STATIC)

MAINTAINERCLEANFILES = Makefile.in
