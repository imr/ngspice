## Process this file with automake to produce Makefile.in
## $Id$

noinst_LIBRARIES = libmisc.a


libmisc_a_SOURCES = \
		getopt1.c	\
		getopt.c	\
		getopt.h	\
		alloc.c		\
		alloc.h		\
		dup2.c		\
		dup2.h		\
		ivars.c		\
		ivars.h		\
		missing_math.c	\
		missing_math.h	\
		mktemp.c	\
		mktemp.h	\
		printnum.c	\
		printnum.h	\
		string.c	\
		stringutil.h	\
		tilde.c		\
		tilde.h		\
		misc_time.c	\
		misc_time.h	\
		wlist.c	    \
		util.c		\
		util.h

## Note that the getopt files get compiled unconditionnaly but some
## magic #define away the body of their own code if the compilation environment
## provides an implementation of its own (like GNU libc)




INCLUDES = -I$(top_srcdir)/src/include
MAINTAINERCLEANFILES = Makefile.in
