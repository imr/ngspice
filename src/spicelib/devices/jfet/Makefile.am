## Process this file with automake to produce Makefile.in

if DEVLIB_WANTED
lib_LTLIBRARIES = libjfet.la
else
noinst_LTLIBRARIES = libjfet.la
endif

libjfet_la_SOURCES =	\
	jfet.c		\
	jfetacld.c	\
	jfetask.c	\
	jfetdefs.h	\
	jfetdel.c	\
	jfetdest.c	\
	jfetdist.c	\
	jfetdset.c	\
	jfetext.h	\
	jfetic.c	\
	jfetinit.c	\
	jfetinit.h	\
	jfetitf.h	\
	jfetload.c	\
	jfetmask.c	\
	jfetmdel.c	\
	jfetmpar.c	\
	jfetnoi.c	\
	jfetpar.c	\
	jfetpzld.c	\
	jfetset.c	\
	jfettemp.c	\
	jfettrun.c



AM_CPPFLAGS = @AM_CPPFLAGS@ -I$(top_srcdir)/src/include
if DEVLIB_WANTED
AM_CFLAGS = -fPIC
AM_LDFLAGS = -shared
else
AM_CFLAGS = $(STATIC)
endif
MAINTAINERCLEANFILES = Makefile.in
