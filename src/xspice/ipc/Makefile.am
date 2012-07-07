## Process this file with automake to produce Makefile.in
#
# JW 3/9/01 - had a go and makeing an   autoconf script.

noinst_LTLIBRARIES = libipcxsp.la

libipcxsp_la_SOURCES = \
	ipcaegis.c   \
	ipc.c        \
	ipcsockets.c \
	ipcstdio.c   \
	ipctiein.c

AM_CPPFLAGS = @AM_CPPFLAGS@ -I$(top_srcdir)/src/include -I$(top_srcdir)/src/spicelib/devices
AM_CFLAGS = $(STATIC)

MAINTAINERCLEANFILES = Makefile.in
