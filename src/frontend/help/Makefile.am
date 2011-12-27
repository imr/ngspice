## Process this file with automake to produce Makefile.in

if !NO_X

noinst_LTLIBRARIES = libhlp.la

libhlp_la_SOURCES = \
		help.c		\
		provide.c	\
		readhelp.c	\
		textdisp.c	\
		xdisplay.c	\
		x11disp.c



AM_CPPFLAGS = @AM_CPPFLAGS@ -I$(top_srcdir)/src/include @X_CFLAGS@

endif

MAINTAINERCLEANFILES = Makefile.in
