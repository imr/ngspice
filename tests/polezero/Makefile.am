## Process this file with automake to produce Makefile.in

TESTS = \
	filt_multistage.cir	\
	filt_rc.cir		\
	filt_bridge_t.cir	\
	pz2.cir			\
	pzt.cir			\
	simplepz.cir

TESTS_ENVIRONMENT = $(SHELL) $(top_srcdir)/tests/bin/check.sh $(top_builddir)/src/ngspice

EXTRA_DIST = \
	$(TESTS) \
	$(TESTS:.cir=.out)

MAINTAINERCLEANFILES = Makefile.in
