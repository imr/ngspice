## Process this file with automake to produce Makefile.in


TESTS = minus-minus.cir xpressn-1.cir xpressn-2.cir xpressn-3.cir bxpressn-1.cir

TESTS_ENVIRONMENT = ngspice_vpath=$(srcdir) $(SHELL) $(top_srcdir)/tests/bin/check.sh $(top_builddir)/src/ngspice

EXTRA_DIST = \
	$(TESTS) \
	$(TESTS:.cir=.out)

MAINTAINERCLEANFILES = Makefile.in
