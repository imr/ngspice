## Process this file with automake to produce Makefile.in

TESTS = \
	cpl_ibm2.cir		\
	cpl3_4_line.cir		\
	ltra1_1_line.cir	\
	ltra2_2_line.cir	\
	txl1_1_line.cir		\
	txl2_3_line.cir


TESTS_ENVIRONMENT = $(SHELL) $(top_srcdir)/tests/bin/check.sh $(top_builddir)/src/ngspice

EXTRA_DIST = \
	$(TESTS) \
	$(TESTS:.cir=.out)

MAINTAINERCLEANFILES = Makefile.in
