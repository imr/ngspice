## Process this file with automake to produce Makefile.in



TESTS = CEamp.cir		\
	FG.cir			\
	FO.cir			\
	diffamp.cir		\
	noise_scale_test.cir	\
	temp.cir

TESTS_ENVIRONMENT = $(SHELL) $(top_srcdir)/tests/bin/check.sh $(top_builddir)/src/ngspice

EXTRA_DIST = \
        $(TESTS) \
        $(TESTS:.cir=.out)


MAINTAINERCLEANFILES = Makefile.in
