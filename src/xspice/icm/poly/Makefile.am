## Process this file with automake to produce Makefile.in
#
# JW 3/9/01 - had a go and makeing an   autoconf script.

noinst_LIBRARIES = libidnxsp.a

libidnxsp_a_SOURCES = \
	ifspec.c      \
	cfunc.c

INCLUDES = -I$(top_srcdir)/src/include

MAINTAINERCLEANFILES = Makefile.in

ifspec.c:
	cmpp -ifs
cfunc.c:
	cmpp -mod cfunc.mod
