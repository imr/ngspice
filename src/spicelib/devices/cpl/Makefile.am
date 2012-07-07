## Process this file with automake to produce Makefile.in

noinst_LTLIBRARIES = libcpl.la

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
AM_CFLAGS = $(STATIC)

MAINTAINERCLEANFILES = Makefile.in
