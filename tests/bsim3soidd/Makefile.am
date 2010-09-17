## Process this file with automake to produce Makefile.in

TESTS = \
	t3.cir		\
	t4.cir		\
	t5.cir		\
	inv2.cir	\
	RampVg2.cir

TESTS_ENVIRONMENT = $(SHELL) $(top_srcdir)/tests/bin/check.sh $(top_builddir)/src/ngspice

EXTRA_DIST = \
	nmosdd.mod		\
	pmosdd.mod		\
	$(TESTS)		\
	$(TESTS:.cir=.out)

MAINTAINERCLEANFILES = Makefile.in
