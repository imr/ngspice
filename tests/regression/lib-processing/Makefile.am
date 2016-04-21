## Process this file with automake to produce Makefile.in


TESTS = ex1a.cir ex1b.cir ex2a.cir ex3a.cir scope-1.cir scope-2.cir scope-3.cir

TESTS_ENVIRONMENT = ngspice_vpath=$(srcdir) $(SHELL) $(top_srcdir)/tests/bin/check.sh $(top_builddir)/src/ngspice

EXTRA_DIST = \
	ex1.lib ex2.lib ex3.lib extra3.lib \
	$(TESTS) \
	$(TESTS:.cir=.out)

MAINTAINERCLEANFILES = Makefile.in
