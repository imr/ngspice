## Process this file with automake to produce Makefile.in

noinst_LIBRARIES = libmisc.a

EXTRA_libmisc_a_SOURCES = \
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
		terminal.c	\
		terminal.h	\
		tilde.c		\
		tilde.h		\
		misc_time.c	\
		misc_time.h	\
		wlist.c	

if HAVE_GETOPT_LONG
libmisc_a_SOURCES = \
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
		terminal.c	\
		terminal.h	\
		tilde.c		\
		tilde.h		\
		misc_time.c	\
		misc_time.h	\
		wlist.c	


else
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
		terminal.c	\
		terminal.h	\
		tilde.c		\
		tilde.h		\
		misc_time.c	\
		misc_time.h	\
		wlist.c	

endif




INCLUDES = -I$(top_srcdir)/src/include
MAINTAINERCLEANFILES = Makefile.in
