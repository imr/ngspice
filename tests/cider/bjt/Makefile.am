## Process this file with automake to produce Makefile.in


TESTS = astable.cir	\
	colposc.cir	\
	ecp.cir		\
	invchain.cir	\
	meclgate.cir	\
	pz.cir		\
	rtlinv.cir	\
	vco.cir

TESTS_ENVIRONMENT = $(SHELL) $(srcdir)/check.sh $(top_builddir)/src/ngspice

EXTRA_DIST = \
	pebjt.lib		\
	$(TESTS)		\
	$(TESTS:.cir=.out)

MAINTAINERCLEANFILES = Makefile.in
