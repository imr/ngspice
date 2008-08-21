noinst_LIBRARIES = libplotting.a

libplotting_a_SOURCES = \
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
	grid.h		\
	pvec.c		\
	pvec.h		\
	plot5.c		\
	plot5.h		\
	plotcurv.c	\
	plotcurv.h	\
	plotit.c	\
	plotit.h	

if !NO_X
libplotting_a_SOURCES += \
	x11.c		\
	x11.h		\
	xgraph.c	\
	xgraph.h
endif

INCLUDES = -I$(top_srcdir)/src/include -I$(top_srcdir)/src/frontend @X_CFLAGS@ 

MAINTAINERCLEANFILES = Makefile.in
