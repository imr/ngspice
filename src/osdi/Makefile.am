## Process this file with automake to produce Makefile.in

noinst_LTLIBRARIES = libosdi.la

libosdi_la_SOURCES =	\
	osdi.h		\
	osdidefs.h	\
	osdiext.h	\
	osdiinit.c	\
	osdiload.c	\
	osdiacld.c \
	osdiparam.c	\
	osdiregistry.c	\
	osdisetup.c \
	osditrunc.c \
	osdipzld.c \
	osdicallbacks.c


libosdi_la_LIBADD = $(XSPICEDLLIBS)

AM_CPPFLAGS = @AM_CPPFLAGS@ -I$(top_srcdir)/src/include
AM_CFLAGS = $(STATIC)
MAINTAINERCLEANFILES = Makefile.in
