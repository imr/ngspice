## Process this file with automake to produce Makefile.in

if DEVLIB_WANTED
lib_LTLIBRARIES = libnumos.la
else
noinst_LTLIBRARIES = libnumos.la
endif

libnumos_la_SOURCES =	\
	numm.c		\
	nummacld.c	\
	nummask.c	\
	numosdef.h	\
	nummdel.c	\
	nummdest.c	\
	nummdump.c	\
	numosext.h	\
	numosinit.c	\
	numosinit.h	\
	numositf.h	\
	nummload.c	\
	nummmdel.c	\
	nummmpar.c	\
	nummparm.c	\
	nummpzld.c	\
	nummset.c	\
	nummtemp.c	\
	nummtrun.c



AM_CPPFLAGS = @AM_CPPFLAGS@ -I$(top_srcdir)/src/include
if DEVLIB_WANTED
AM_CFLAGS = -fPIC
AM_LDFLAGS = -shared
else
AM_CFLAGS = $(STATIC)
endif
MAINTAINERCLEANFILES = Makefile.in
