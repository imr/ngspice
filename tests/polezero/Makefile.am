## Process this file with automake to produce Makefile.in

TESTS = \
	filt_multistage.cir	\
	filt_rc.cir		\
	filt_bridge_t.cir

TESTS_ENVIRONMENT = $(SHELL) $(srcdir)/../check.sh $(top_builddir)/src/ngspice

EXTRA_DIST = \
	$(TESTS)		\
	$(TESTS:.cir=.out)

MAINTAINERCLEANFILES = Makefile.in
