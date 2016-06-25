## Process this file with automake to produce Makefile.in


TESTS = sens-ac-1.cir sens-ac-2.cir sens-dc-1.cir sens-dc-2.cir

TESTS_ENVIRONMENT = ngspice_vpath=$(srcdir) $(SHELL) $(top_srcdir)/tests/bin/check.sh $(top_builddir)/src/ngspice

EXTRA_DIST = \
	$(TESTS) \
	$(TESTS:.cir=.out)

MAINTAINERCLEANFILES = Makefile.in
