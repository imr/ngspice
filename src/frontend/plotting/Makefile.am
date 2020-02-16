noinst_LTLIBRARIES = libplotting.la

libplotting_la_SOURCES = \
	plotting.c	\
	plotting.h	\
	agraf.c		\
	agraf.h		\
	clip.c		\
	clip.h		\
	gnuplot.c	\
	gnuplot.h	\
	graf.c		\
	graf.h		\
	graphdb.c	\
	graphdb.h	\
	grid.c		\
	pvec.c		\
	pvec.h		\
	plot5.c		\
	plot5.h		\
	plotcurv.c	\
	plotcurv.h	\
	plotit.c	\
	plotit.h

if !NO_X
libplotting_la_SOURCES += \
	x11.c		\
	x11.h

endif

AM_CPPFLAGS = @AM_CPPFLAGS@ -I$(top_srcdir)/src/include -I$(top_srcdir)/src/frontend @X_CFLAGS@
AM_CFLAGS = $(STATIC)

MAINTAINERCLEANFILES = Makefile.in
