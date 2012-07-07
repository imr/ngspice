## Process this file with automake to produce Makefile.in

noinst_LTLIBRARIES = libmisc.la


libmisc_la_SOURCES = \
		getopt_long_bsd.c	\
		getopt_bsd.h	\
		alloc.c		\
		alloc.h		\
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

## Note that the getopt files get compiled unconditionnaly but some
## magic #define away the body of their own code if the compilation environment
## provides an implementation of its own (like GNU libc)




AM_CPPFLAGS = @AM_CPPFLAGS@ -I$(top_srcdir)/src/include
AM_CFLAGS = $(STATIC)
MAINTAINERCLEANFILES = Makefile.in
