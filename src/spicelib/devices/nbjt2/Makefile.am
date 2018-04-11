## Process this file with automake to produce Makefile.in

noinst_LTLIBRARIES = libnbjt2.la

libnbjt2_la_SOURCES =	\
	nbt2.c		\
	nbt2acld.c	\
	nbt2ask.c	\
	nbjt2def.h	\
	nbt2del.c	\
	nbt2dump.c	\
	nbjt2ext.h	\
	nbt2init.c	\
	nbt2init.h	\
	nbjt2itf.h	\
	nbt2load.c	\
	nbt2mpar.c	\
	nbt2parm.c	\
	nbt2pzld.c	\
	nbt2set.c	\
	nbt2temp.c	\
	nbt2trun.c



AM_CPPFLAGS = @AM_CPPFLAGS@ -I$(top_srcdir)/src/include
AM_CFLAGS = $(STATIC)

MAINTAINERCLEANFILES = Makefile.in
