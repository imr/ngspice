## Process this file with automake to produce Makefile.in

TESTS = \
	test1.v3	\
	test2.v3	\
	test3.v3	\
	test4.v3	\
	test5.v3	\
	test6.v3	\
	test7.v3	\
	test8.v3	\
	test9.v3	\
	test10.v3	\
	test11.v3	\
	test12.v3	\
	test13.v3	\
	test14.v3
	

TESTS_ENVIRONMENT = $(SHELL) $(srcdir)/check.sh $(top_builddir)/src/ngspice

EXTRA_DIST = \
	model.card		\
	modelcard.nmos		\
	modelcard.pmos		\
	$(TESTS)		\
	$(TESTS:.v3=.out)

MAINTAINERCLEANFILES = Makefile.in
