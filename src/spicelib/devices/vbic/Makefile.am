## Process this file with automake to produce Makefile.in

noinst_LTLIBRARIES = libvbic.la

libvbic_la_SOURCES =	\
	vbic.c		\
	vbicacld.c	\
	vbicask.c	\
	vbicconv.c	\
	vbicdefs.h	\
	vbicext.h	\
	vbicgetic.c	\
	vbicinit.c	\
	vbicinit.h	\
	vbicitf.h	\
	vbicload.c	\
	vbicmask.c	\
	vbicmpar.c	\
	vbicnoise.c	\
	vbicparam.c	\
	vbicpzld.c	\
	vbicsetup.c	\
	vbicsoachk.c	\
	vbictemp.c	\
	vbictrunc.c



AM_CPPFLAGS = @AM_CPPFLAGS@ -I$(top_srcdir)/src/include
AM_CFLAGS = $(STATIC)

MAINTAINERCLEANFILES = Makefile.in
