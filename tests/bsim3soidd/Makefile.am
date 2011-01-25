## Process this file with automake to produce Makefile.in

TESTS = \
	t3.cir		\
	t4.cir		\
	t5.cir		\
	inv2.cir	\
	RampVg2.cir

TESTS_ENVIRONMENT = ngspice_vpath=$(srcdir) $(SHELL) $(top_srcdir)/tests/bin/check.sh $(top_builddir)/src/ngspice

EXTRA_DIST = \
	nmosdd.mod		\
	pmosdd.mod		\
	$(TESTS)		\
	$(TESTS:.cir=.out)

EXTRA_DIST += ring51.cir lib.h

MAINTAINERCLEANFILES = Makefile.in
