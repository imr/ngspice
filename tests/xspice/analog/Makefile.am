## Analog code-model AC Jacobian regression.
TESTS = switch-ac.cir
TESTS_ENVIRONMENT = ngspice_vpath=$(srcdir) SPICE_SCRIPTS=. $(SHELL) $(top_srcdir)/tests/bin/check.sh $(top_builddir)/src/ngspice
EXTRA_DIST = $(TESTS) $(TESTS:.cir=.out)
MAINTAINERCLEANFILES = Makefile.in
