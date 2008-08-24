## Process this file with automake to produce Makefile.in

if !NO_X

noinst_LIBRARIES = libhlp.a

libhlp_a_SOURCES = \
		help.c	\
		provide.c	\
		readhelp.c	\
		textdisp.c	\
		xdisplay.c      \
		x11disp.c	



AM_CPPFLAGS = -I$(top_srcdir)/src/include @X_CFLAGS@

endif 

MAINTAINERCLEANFILES = Makefile.in
