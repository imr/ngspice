## Process this file with automake to produce Makefile.in

noinst_LTLIBRARIES = libhfet.la

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
	hfetnode.c	\
	hfetparam.c	\
	hfetpzl.c	\
	hfetsetup.c	\
	hfettemp.c	\
	hfettrunc.c



AM_CPPFLAGS = @AM_CPPFLAGS@ -I$(top_srcdir)/src/include
AM_CFLAGS = $(STATIC)
MAINTAINERCLEANFILES = Makefile.in
