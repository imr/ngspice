## Process this file with automake to produce Makefile.in

noinst_LTLIBRARIES = libmesa.la

libmesa_la_SOURCES =	\
	mesa.c		\
	mesaacl.c	\
	mesaask.c	\
	mesadefs.h	\
	mesaext.h	\
	mesagetic.c	\
	mesainit.c	\
	mesainit.h	\
	mesaitf.h	\
	mesaload.c	\
	mesamask.c	\
	mesamparam.c	\
	mesaparam.c	\
	mesapzl.c	\
	mesasetup.c	\
	mesatemp.c	\
	mesatrunc.c



AM_CPPFLAGS = @AM_CPPFLAGS@ -I$(top_srcdir)/src/include
AM_CFLAGS = $(STATIC)
MAINTAINERCLEANFILES = Makefile.in
