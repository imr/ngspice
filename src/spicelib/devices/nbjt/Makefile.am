## Process this file with automake to produce Makefile.in

noinst_LTLIBRARIES = libnbjt.la

libnbjt_la_SOURCES =	\
	nbjt.c		\
	nbjtacld.c	\
	nbjtask.c	\
	nbjtdefs.h	\
	nbjtdel.c	\
	nbjtdump.c	\
	nbjtext.h	\
	nbjtinit.c	\
	nbjtinit.h	\
	nbjtitf.h	\
	nbjtload.c	\
	nbjtmpar.c	\
	nbjtparm.c	\
	nbjtpzld.c	\
	nbjtset.c	\
	nbjttemp.c	\
	nbjttrun.c



AM_CPPFLAGS = @AM_CPPFLAGS@ -I$(top_srcdir)/src/include
AM_CFLAGS = $(STATIC)

MAINTAINERCLEANFILES = Makefile.in
