## Process this file with automake to produce Makefile.in

if DEVLIB_WANTED
lib_LTLIBRARIES = libcpl.la
else
noinst_LTLIBRARIES = libcpl.la
endif

libcpl_la_SOURCES = \
	cpl.c      \
	cplask.c   \
	cpldefs.h  \
	cpldest.c  \
	cplext.h   \
	cplinit.h  \
	cplitf.h   \
	cplmask.c  \
	cplmdel.c  \
	cplparam.c \
	cpldel.c   \
	cplload.c  \
	cplmpar.c  \
	cplsetup.c \
	cplinit.c

AM_CPPFLAGS = @AM_CPPFLAGS@ -I$(top_srcdir)/src/include
if DEVLIB_WANTED
AM_CFLAGS = -fPIC
AM_LDFLAGS = -shared
else
AM_CFLAGS = $(STATIC)
endif
MAINTAINERCLEANFILES = Makefile.in
