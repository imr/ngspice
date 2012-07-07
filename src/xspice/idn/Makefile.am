## Process this file with automake to produce Makefile.in
#
# JW 3/9/01 - had a go and makeing an   autoconf script.

noinst_LTLIBRARIES = libidnxsp.la

libidnxsp_la_SOURCES = \
	idndig.c

AM_CPPFLAGS = @AM_CPPFLAGS@ -I$(top_srcdir)/src/include
AM_CFLAGS = $(STATIC)

MAINTAINERCLEANFILES = Makefile.in
