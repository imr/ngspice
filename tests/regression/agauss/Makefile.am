## Process this file with automake to produce Makefile.in


TESTS = \
	test-agauss-1.cir \
	test-agauss-2.cir \
	test-agauss-3.cir \
	test-agauss-4.cir \
	test-agauss-5.cir \
	test-agauss-6.cir

TESTS_ENVIRONMENT = ngspice_vpath=$(srcdir) $(SHELL) $(top_srcdir)/tests/bin/check.sh $(top_builddir)/src/ngspice

EXTRA_DIST = \
	$(TESTS) \
	$(TESTS:.cir=.out)

MAINTAINERCLEANFILES = Makefile.in
