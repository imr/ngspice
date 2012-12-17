## Process this file with automake to produce Makefile.in

if DEVLIB_WANTED
lib_LTLIBRARIES = libhfet.la
else
noinst_LTLIBRARIES = libhfet.la
endif

libhfet_la_SOURCES =	\
	hfet.c		\
	hfetacl.c	\
	hfetask.c	\
	hfetdefs.h	\
	hfetdel.c	\
	hfetdest.c	\
	hfetext.h	\
	hfetgetic.c	\
	hfetinit.c	\
	hfetinit.h	\
	hfetitf.h	\
	hfetload.c	\
	hfetmask.c	\
	hfetmdel.c	\
	hfetmpar.c	\
	hfetparam.c	\
	hfetpzl.c	\
	hfetsetup.c	\
	hfettemp.c	\
	hfettrunc.c



AM_CPPFLAGS = @AM_CPPFLAGS@ -I$(top_srcdir)/src/include
if DEVLIB_WANTED
AM_CFLAGS = -fPIC
AM_LDFLAGS = -shared
else
AM_CFLAGS = $(STATIC)
endif
MAINTAINERCLEANFILES = Makefile.in
