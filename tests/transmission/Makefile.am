## Process this file with automake to produce Makefile.in

TESTS = \
	cpl_ksp.cir	\
	cpl_sp.cir	\
	cpl2_ksp.cir	\
	cpl2_sp.cir	\
	ibm1.cir	\
	ibm2.cir	\
	ibm3.cir	\
	ltra_1.cir	\
	ltra_2.cir	\
	ltra_3.cir	\
	txl_ksp.cir	\
	txl_sp.cir	\
	txl2_ksp.cir	\
	txl2_sp.cir	\
	txl3_ksp.cir	\
	txl3_sp.cir	\
	txl4_ksp.cir	\
	txl4_sp.cir	\
	txl5_ksp.cir	\
	txl5_sp.cir	\
	txl6_ksp.cir	\
	txl6_sp.cir
	
	

TESTS_ENVIRONMENT = $(SHELL) $(top_srcdir)/tests/bin/check.sh $(top_builddir)/src/ngspice

EXTRA_DIST = \
	$(TESTS)		\
	$(TESTS:.cir=.out)

MAINTAINERCLEANFILES = Makefile.in
