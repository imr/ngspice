## Process this file with automake to produce Makefile.in

if DEVLIB_WANTED
lib_LTLIBRARIES = libmesa.la
else
noinst_LTLIBRARIES = libmesa.la
endif

libmesa_la_SOURCES =	\
	mesa.c		\
	mesaacl.c	\
	mesaask.c	\
	mesadefs.h	\
	mesadel.c	\
	mesadest.c	\
	mesaext.h	\
	mesagetic.c	\
	mesainit.c	\
	mesainit.h	\
	mesaitf.h	\
	mesaload.c	\
	mesamask.c	\
	mesamdel.c	\
	mesamparam.c	\
	mesaparam.c	\
	mesapzl.c	\
	mesasetup.c	\
	mesatemp.c	\
	mesatrunc.c



AM_CPPFLAGS = @AM_CPPFLAGS@ -I$(top_srcdir)/src/include
if DEVLIB_WANTED
AM_CFLAGS = -fPIC
AM_LDFLAGS = -shared
else
AM_CFLAGS = $(STATIC)
endif
MAINTAINERCLEANFILES = Makefile.in
