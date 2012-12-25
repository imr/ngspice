## Process this file with automake to produce Makefile.in

noinst_LTLIBRARIES = libni.la

libni_la_SOURCES = \
		niaciter.c	\
		nicomcof.c	\
		niconv.c	\
		nidest.c	\
		niditer.c	\
		niinit.c	\
		niinteg.c	\
		niiter.c	\
		niniter.c	\
		nipred.c	\
		nipzmeth.c	\
		nireinit.c	\
		nisenre.c


AM_CPPFLAGS = @AM_CPPFLAGS@ -I$(top_srcdir)/src/include
AM_CFLAGS = $(STATIC)

MAINTAINERCLEANFILES = Makefile.in
