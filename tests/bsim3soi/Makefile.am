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

TESTS_ENVIRONMENT = ngspice_vpath=$(srcdir) $(SHELL) $(top_srcdir)/tests/bin/check.sh $(top_builddir)/src/ngspice

EXTRA_DIST = \
	nmos4p3.mod		\
	pmos4p3.mod		\
	$(TESTS)		\
	$(TESTS:.cir=.out)

EXTRA_DIST += ring51_41.cir ring51_41.out

MAINTAINERCLEANFILES = Makefile.in
