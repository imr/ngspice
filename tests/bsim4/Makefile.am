## Process this file with automake to produce Makefile.in

TESTS = \
	test1.cir	\
	test2.cir	\
	test3.cir	\
	test4.cir	\
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
	opamp.cir

TESTS_ENVIRONMENT = $(SHELL) $(srcdir)/../check.sh $(top_builddir)/src/ngspice

EXTRA_DIST = \
	$(TESTS)		\
	$(TESTS:.cir=.out)

MAINTAINERCLEANFILES = Makefile.in
