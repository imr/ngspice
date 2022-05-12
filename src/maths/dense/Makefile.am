## Process this file with automake to produce Makefile.in

noinst_LTLIBRARIES = libdense.la

libdense_la_SOURCES = \
		dense.c \
		dense.h \
		denseinlines.h


AM_CPPFLAGS = @AM_CPPFLAGS@ -I$(top_srcdir)/src/include -I$(top_srcdir)/src/maths/dense
AM_CFLAGS = $(STATIC)
MAINTAINERCLEANFILES = Makefile.in
