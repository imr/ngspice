## Process this file with automake to produce Makefile.in

TESTS = \
	vsrc-pwl-1.cir \
	vsrc-pwl-2.cir \
	isrc-pwl-1.cir

TESTS_ENVIRONMENT = SPICE_SCRIPTS=$(srcdir) $(SHELL) $(top_srcdir)/tests/bin/check.sh "$(top_builddir)/src/ngspice -r foobaz"

EXTRA_DIST = \
	$(TESTS) \
	$(TESTS:.cir=.out)

MAINTAINERCLEANFILES = Makefile.in
