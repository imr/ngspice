## Process this file with automake to produce Makefile.in

TESTS = \
	res_array.cir		\
	res_simple.cir		\
	res_partition.cir

TESTS_ENVIRONMENT = $(SHELL) $(srcdir)/../check.sh $(top_builddir)/src/ngspice

EXTRA_DIST = \
	$(TESTS)		\
	$(TESTS:.sh=.out)

MAINTAINERCLEANFILES = Makefile.in
