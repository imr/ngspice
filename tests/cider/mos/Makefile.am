## Process this file with automake to produce Makefile.in


TESTS = bootinv.cir	\
	charge.cir	\
	cmosinv.cir	\
	nmosinv.cir	\
	pass.cir	\
	ringosc.cir

TESTS_ENVIRONMENT = $(SHELL) $(srcdir)/check.sh $(top_builddir)/src/ngspice

EXTRA_DIST = \
	$(TESTS)		\
	$(TESTS:.cir=.out)

MAINTAINERCLEANFILES = Makefile.in
