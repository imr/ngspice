## Process this file with automake to produce Makefile.in


TESTS = \
	diffpair.cir		\
	fourbitadder.cir	\
	mosamp.cir		\
	mosmem.cir		\
	rc.cir			\
	rca3040.cir		\
	rtlinv.cir		\
	schmitt.cir

TESTS_ENVIRONMENT = $(SHELL) $(srcdir)/check.sh $(top_builddir)/src/ngspice

EXTRA_DIST = \
	$(TESTS)		\
	$(TESTS:.cir=.out)

MAINTAINERCLEANFILES = Makefile.in
