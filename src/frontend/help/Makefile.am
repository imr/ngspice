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



libhlp_la_CPPFLAGS = @AM_CPPFLAGS@ -I$(top_srcdir)/src/include @X_CFLAGS@
libhlp_la_CFLAGS = $(STATIC)

endif

MAINTAINERCLEANFILES = Makefile.in
