## Process this file with automake to produce Makefile.in

TESTS = \
        test1.cir    \
        test2.cir    \
        test3.cir    \
        test4.cir    \
        test5.cir    \
        test6.cir    \
        test7.cir    \
        test8.cir    \
        inv_dc.cir   \
        inv_tr.cir

TESTS_ENVIRONMENT = $(SHELL) $(top_srcdir)/tests/bin/check.sh $(top_builddir)/src/ngspice

EXTRA_DIST = \
	nmos4p0.mod		\
	pmos4p0.mod		\
	$(TESTS)		\
	$(TESTS:.cir=.out)

MAINTAINERCLEANFILES = Makefile.in
