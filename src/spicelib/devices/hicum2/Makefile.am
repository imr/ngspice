## Process this file with automake to produce Makefile.in

noinst_LTLIBRARIES = libhicum2.la

libhicum2_la_SOURCES =	\
	hicum2.c		\
	hicum2acld.c	\
	hicum2ask.c	    \
	hicum2conv.c	\
	hicum2defs.h	\
	hicum2ext.h   	\
	hicum2getic.c	\
	hicum2init.c	\
	hicum2init.h	\
	hicum2itf.h	    \
	hicumL2.hpp     \
	hicumL2.cpp     \
	hicumL2temp.hpp	\
	hicumL2temp.cpp	\
	hicum2mask.c	\
	hicum2mpar.c	\
	hicum2noise.c	\
	hicum2param.c	\
	hicum2pzld.c	\
	hicum2setup.c	\
	hicum2soachk.c	\
	hicum2trunc.c

AM_CFLAGS   = -O -g -Wall -Wextra -I$(top_srcdir)/src/include
AM_CPPFLAGS = @AM_CPPFLAGS@ -O -g -Wall -Wextra
AM_CXXFLAGS = -I$(top_srcdir)/src/include/cppduals -I$(top_srcdir)/src/include -std=c++11 -O -g -Wall -Wextra

MAINTAINERCLEANFILES = Makefile.in
