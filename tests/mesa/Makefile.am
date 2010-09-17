## Process this file with automake to produce Makefile.in

TESTS = \
	mesa11.cir		\
	mesa13.cir		\
	mesa14.cir		\
	mesa15.cir		\
	mesa21.cir		\
	mesa.cir		\
	mesgout.cir		\
	mesinv.cir		\
	mesosc.cir

TESTS_ENVIRONMENT = $(SHELL) $(top_srcdir)/tests/bin/check.sh $(top_builddir)/src/ngspice

EXTRA_DIST = \
	$(TESTS) \
	$(TESTS:.cir=.out)

EXTRA_DIST += mesa12.cir mesa-12.cir

MAINTAINERCLEANFILES = Makefile.in
