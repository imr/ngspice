## Process this file with automake to produce Makefile.in

noinst_LTLIBRARIES = libnumd2.la

libnumd2_la_SOURCES =	\
	nud2.c		\
	nud2acld.c	\
	nud2ask.c	\
	numd2def.h	\
	nud2del.c	\
	nud2dump.c	\
	numd2ext.h	\
	numd2init.c	\
	numd2init.h	\
	numd2itf.h	\
	nud2load.c	\
	nud2mpar.c	\
	nud2parm.c	\
	nud2pzld.c	\
	nud2set.c	\
	nud2temp.c	\
	nud2trun.c



AM_CPPFLAGS = @AM_CPPFLAGS@ -I$(top_srcdir)/src/include
AM_CFLAGS = $(STATIC)

MAINTAINERCLEANFILES = Makefile.in
