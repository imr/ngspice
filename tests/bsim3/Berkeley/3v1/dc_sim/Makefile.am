## Process this file with automake to produce Makefile.in

TESTS = \
	test1.v31	\
	test2.v31	\
	test3.v31	\
	test4.v31	\
	test5.v31	\
	test6.v31	\
	test7.v31	\
	test8.v31	\
	test9.v31	\
	test10.v31	\
	test11.v31	\
	test12.v31	\
	test13.v31	\
	test14.v31
	

TESTS_ENVIRONMENT = $(SHELL) $(srcdir)/check.sh $(top_builddir)/src/ngspice

EXTRA_DIST = \
	model.card		\
	modelcard.nmos		\
	modelcard.pmos		\
	$(TESTS)		\
	$(TESTS:.v31=.out)

MAINTAINERCLEANFILES = Makefile.in
