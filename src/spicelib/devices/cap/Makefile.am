## Process this file with automake to produce Makefile.in

if DEVLIB_WANTED
lib_LTLIBRARIES = libcap.la
else
noinst_LTLIBRARIES = libcap.la
endif

libcap_la_SOURCES =	\
	cap.c		\
	capacld.c	\
	capask.c	\
	capdefs.h	\
	capdel.c	\
	capdest.c	\
	capext.h	\
	capgetic.c	\
	capinit.c	\
	capinit.h	\
	capitf.h	\
	capload.c	\
	capmask.c	\
	capmdel.c	\
	capmpar.c	\
	capparam.c	\
	cappzld.c	\
	capsacl.c	\
	capsetup.c	\
	capsload.c	\
	capsprt.c	\
	capsset.c	\
	capsupd.c	\
	captemp.c	\
	captrunc.c



AM_CPPFLAGS = @AM_CPPFLAGS@ -I$(top_srcdir)/src/include
if DEVLIB_WANTED
AM_CFLAGS = -fPIC
AM_LDFLAGS = -shared
else
AM_CFLAGS = $(STATIC)
endif
MAINTAINERCLEANFILES = Makefile.in
