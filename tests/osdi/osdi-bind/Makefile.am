## Process this file with automake to produce Makefile.in

TESTS = osdi-bind-1.cir osdi-bind-2.cir osdi-bind-3.cir diode-numeric-node-1.cir diode-numeric-node-2.cir builtin-module-name-1.cir model-type-mismatch-1.cir model-binning-1.cir

TESTS_ENVIRONMENT = NGSPICE_OSDI_DIR=$(abs_top_builddir)/tests/osdi/.libs ngspice_vpath=$(srcdir) $(SHELL) $(top_srcdir)/tests/bin/check.sh $(top_builddir)/src/ngspice

EXTRA_DIST = \
	$(TESTS) \
	$(TESTS:.cir=.out)

MAINTAINERCLEANFILES = Makefile.in
