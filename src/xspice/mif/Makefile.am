## Process this file with automake to produce Makefile.in
#
# JW 3/9/01 - had a go and makeing an   autoconf script.

noinst_LTLIBRARIES = libmifxsp.la

libmifxsp_la_SOURCES = \
	mif_inp2.c \
	mifgetmod.c \
	mifgetvalue.c \
	mifload.c \
	mifmpara.c \
	mifsetup.c \
	mifutil.c \
	mifask.c \
	mifmask.c \
	miftrunc.c \
	mifconvt.c \
	mifdelete.c \
	mifmdelete.c \
	mifdestr.c \
	mif.c

AM_CPPFLAGS = @AM_CPPFLAGS@ -I$(top_srcdir)/src/include
AM_CFLAGS = $(STATIC)

MAINTAINERCLEANFILES = Makefile.in
