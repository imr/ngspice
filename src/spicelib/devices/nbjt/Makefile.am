## Process this file with automake to produce Makefile.in

if DEVLIB_WANTED
lib_LTLIBRARIES = libnbjt.la
else
noinst_LTLIBRARIES = libnbjt.la
endif

libnbjt_la_SOURCES =	\
	nbjt.c		\
	nbjtacld.c	\
	nbjtask.c	\
	nbjtdefs.h	\
	nbjtdel.c	\
	nbjtdest.c	\
	nbjtdump.c	\
	nbjtext.h	\
	nbjtinit.c	\
	nbjtinit.h	\
	nbjtitf.h	\
	nbjtload.c	\
	nbjtmdel.c	\
	nbjtmpar.c	\
	nbjtparm.c	\
	nbjtpzld.c	\
	nbjtset.c	\
	nbjttemp.c	\
	nbjttrun.c



AM_CPPFLAGS = @AM_CPPFLAGS@ -I$(top_srcdir)/src/include
if DEVLIB_WANTED
AM_CFLAGS = -fPIC
AM_LDFLAGS = -shared
else
AM_CFLAGS = $(STATIC)
endif
MAINTAINERCLEANFILES = Makefile.in
