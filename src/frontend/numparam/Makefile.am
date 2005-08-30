## Process this file with automake to produce Makefile.in

EXTRA_DIST = downgrad.txt ngconfig.sh readme.txt nupatest.c washprog.c

noinst_LIBRARIES = libnumparam.a

libnumparam_a_SOURCES = 		\
	spicenum.c	\
	xpressn.c	\
	mystring.c	\
	general.h	\
	numpaif.h	\
	numparam.h

INCLUDES = -I$(top_srcdir)/src/include
MAINTAINERCLEANFILES = Makefile.in
