## Process this file with automake to produce Makefile.in


TESTS = \
	mos6inv.cir		\
	simpleinv.cir

TESTS_ENVIRONMENT = $(SHELL) $(top_srcdir)/tests/bin/check.sh $(top_builddir)/src/ngspice

EXTRA_DIST = \
	$(TESTS) \
	$(TESTS:.cir=.out)

MAINTAINERCLEANFILES = Makefile.in
