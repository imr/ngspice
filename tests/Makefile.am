## Process this file with automake to produce Makefile.in

SUBDIRS = 	bsim3   \
	bsim3soi    \
    bsim4       \
	hisim       \
	hisimhv


DIST_SUBDIRS =	bsim3 \
	bsim3soi	\
    bsim4		\
	hisim		\
	hisimhv

if XSPICE_WANTED
SUBDIRS += xspice
endif

DIST_SUBDIRS += xspice

TESTS_ENVIRONMENT = $(SHELL) $(top_srcdir)/tests/bin/check.sh $(top_builddir)/src/ngspice

EXTRA_DIST = \
	README bin


MAINTAINERCLEANFILES = Makefile.in
