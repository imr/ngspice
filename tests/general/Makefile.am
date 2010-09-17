## Process this file with automake to produce Makefile.in


TESTS = \
	mosamp.cir		\
	mosmem.cir		\
	rc.cir			\
	rca3040.cir		\
	rtlinv.cir		\
	schmitt.cir		\
	diffpair.cir		\
	fourbitadder.cir

TESTS_ENVIRONMENT = $(SHELL) $(top_srcdir)/tests/bin/check.sh $(top_builddir)/src/ngspice

EXTRA_DIST = \
	$(TESTS) \
	$(TESTS:.cir=.out)

MAINTAINERCLEANFILES = Makefile.in
