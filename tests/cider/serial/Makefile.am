## Process this file with automake to produce Makefile.in


TESTS = astable.cir	\
	charge.cir	\
	colposc.cir	\
	dbridge.cir	\
	invchain.cir	\
	meclgate.cir	\
	nmosinv.cir	\
	pass.cir	\
	pullup.cir	\
	recovery.cir	\
	rtlinv.cir	\
	vco.cir

TESTS_ENVIRONMENT = $(SHELL) $(srcdir)/check.sh $(top_builddir)/src/ngspice

EXTRA_DIST = \
	readme			\
	$(TESTS)		\
	$(TESTS:.cir=.out)

MAINTAINERCLEANFILES = Makefile.in
