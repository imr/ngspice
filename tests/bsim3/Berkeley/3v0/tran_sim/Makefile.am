## Process this file with automake to produce Makefile.in

TESTS = comparator.sp


TESTS_ENVIRONMENT = $(SHELL) $(srcdir)/check.sh $(top_builddir)/src/ngspice

EXTRA_DIST = \
	$(TESTS)		\
	$(TESTS:.sp=.out)

MAINTAINERCLEANFILES = Makefile.in
