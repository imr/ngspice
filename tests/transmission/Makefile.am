## Process this file with automake to produce Makefile.in

TESTS = \
	ltra_1.cir	\
	ltra_2.cir	\
	ltra_3.cir
	

TESTS_ENVIRONMENT = $(SHELL) $(srcdir)/../check.sh $(top_builddir)/src/ngspice

EXTRA_DIST = \
	$(TESTS)		\
	$(TESTS:.cir=.out)

MAINTAINERCLEANFILES = Makefile.in
