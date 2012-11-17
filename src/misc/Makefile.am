## Process this file with automake to produce Makefile.in

noinst_LTLIBRARIES = libmisc.la


libmisc_la_SOURCES = \
		dup2.c		\
		dstring.c	\
		dup2.h		\
		hash.c		\
		ivars.c		\
		ivars.h		\
		mktemp.c	\
		mktemp.h	\
		printnum.c	\
		printnum.h	\
		string.c	\
		tilde.c		\
		tilde.h		\
		misc_time.c	\
		misc_time.h	\
		wlist.c		\
		util.c		\
		util.h


### getopt_long_bsd.c ###
if !HAVE_GETOPT_H
if !HAVE_GETOPT_LONG
libmisc_la_SOURCES += getopt_long_bsd.c getopt_bsd.h
else
libmisc_la_SOURCES += getopt_long_bsd.c getopt_bsd.h
endif
else
if !HAVE_GETOPT_LONG
libmisc_la_SOURCES += getopt_long_bsd.c getopt_bsd.h
endif
endif

### alloc.c ###
if !HAVE_LIBGC_WANTED
libmisc_la_SOURCES += alloc.c alloc.h
endif


AM_CPPFLAGS = @AM_CPPFLAGS@ -I$(top_srcdir)/src/include
AM_CFLAGS = $(STATIC)
MAINTAINERCLEANFILES = Makefile.in
