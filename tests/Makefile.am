## Process this file with automake to produce Makefile.in

SUBDIRS = resistance filters polezero bsim3soipd

TESTS = \
	diffpair.cir		\
	fourbitadder.cir

TESTS_ENVIRONMENT = $(SHELL) $(srcdir)/check.sh $(top_builddir)/src/ngspice

EXTRA_DIST = \
	README			\
	check.sh		\
	$(TESTS)		\
	$(TESTS:.cir=.out)

MAINTAINERCLEANFILES = Makefile.in
