## Process this file with automake to produce Makefile.in

if DEVLIB_WANTED
lib_LTLIBRARIES = libnbjt2.la
else
noinst_LTLIBRARIES = libnbjt2.la
endif

libnbjt2_la_SOURCES =	\
	nbt2.c		\
	nbt2acld.c	\
	nbt2ask.c	\
	nbjt2def.h	\
	nbt2del.c	\
	nbt2dest.c	\
	nbt2dump.c	\
	nbjt2ext.h	\
	nbt2init.c	\
	nbt2init.h	\
	nbjt2itf.h	\
	nbt2load.c	\
	nbt2mdel.c	\
	nbt2mpar.c	\
	nbt2parm.c	\
	nbt2pzld.c	\
	nbt2set.c	\
	nbt2temp.c	\
	nbt2trun.c



AM_CPPFLAGS = @AM_CPPFLAGS@ -I$(top_srcdir)/src/include
if DEVLIB_WANTED
AM_CFLAGS = -fPIC
AM_LDFLAGS = -shared
else
AM_CFLAGS = $(STATIC)
endif
MAINTAINERCLEANFILES = Makefile.in
