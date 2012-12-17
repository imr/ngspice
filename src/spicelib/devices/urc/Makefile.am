## Process this file with automake to produce Makefile.in

if DEVLIB_WANTED
lib_LTLIBRARIES = liburc.la
else
noinst_LTLIBRARIES = liburc.la
endif

liburc_la_SOURCES =	\
	urc.c		\
	urcask.c	\
	urcdefs.h	\
	urcdel.c	\
	urcdest.c	\
	urcext.h	\
	urcinit.c	\
	urcinit.h	\
	urcitf.h	\
	urcmask.c	\
	urcmdel.c	\
	urcmpar.c	\
	urcparam.c	\
	urcsetup.c



AM_CPPFLAGS = @AM_CPPFLAGS@ -I$(top_srcdir)/src/include
if DEVLIB_WANTED
AM_CFLAGS = -fPIC
AM_LDFLAGS = -shared
else
AM_CFLAGS = $(STATIC)
endif
MAINTAINERCLEANFILES = Makefile.in
