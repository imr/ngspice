## Process this file with automake to produce Makefile.in

TESTS = \
	test1.cir	\
	test1_gedl.cir	\
	test2.cir	\
	test2_gedl.cir	\
	test3.cir	\
	test3_gedl.cir	\
	test4.cir	\
	test4_gedl.cir	\
	test5.cir	\
	test6.cir	\
	test7.cir	\
	test8.cir	\
	test9.cir	\
	test10.cir	\
	test11.cir	\
	test12.cir	\
	test13.cir	\
	test14.cir	\
	comprt.cir	\
	gstage.cir	\
	oneshot.cir	\
	opamp.cir	\
	ro_17.cir

TESTS_ENVIRONMENT = $(SHELL) $(top_srcdir)/tests/bin/check.sh $(top_builddir)/src/ngspice

EXTRA_DIST = \
	modelcard.nmos		\
	modelcard.pmos		\
	$(TESTS)		\
	$(TESTS:.cir=.out)

MAINTAINERCLEANFILES = Makefile.in
