## Process this file with automake to produce Makefile.in

if DEVLIB_WANTED
lib_LTLIBRARIES = libvbic.la
else
noinst_LTLIBRARIES = libvbic.la
endif

libvbic_la_SOURCES =	\
	vbic.c		\
	vbicacld.c	\
	vbicask.c	\
	vbicconv.c	\
	vbicdefs.h	\
	vbicdel.c	\
	vbicdest.c	\
	vbicext.h	\
	vbicgetic.c	\
	vbicinit.c	\
	vbicinit.h	\
	vbicitf.h	\
	vbicload.c	\
	vbicmask.c	\
	vbicmdel.c	\
	vbicmpar.c	\
	vbicnoise.c	\
	vbicparam.c	\
	vbicpzld.c	\
	vbicsetup.c	\
	vbictemp.c	\
	vbictrunc.c



AM_CPPFLAGS = @AM_CPPFLAGS@ -I$(top_srcdir)/src/include
if DEVLIB_WANTED
AM_CFLAGS = -fPIC
AM_LDFLAGS = -shared
else
AM_CFLAGS = $(STATIC)
endif
MAINTAINERCLEANFILES = Makefile.in
