## Process this file with automake to produce Makefile.in

TESTS = osdi-bind-1.cir osdi-bind-2.cir osdi-bind-3.cir osdi-bind-4.cir osdi-bind-5.cir diode-numeric-node-1.cir diode-numeric-node-2.cir builtin-module-name-1.cir model-type-mismatch-1.cir model-binning-1.cir polarity-values-1.cir quoted-list-1.cir none-version-message-1.cir name-chars-1.cir version-expression-1.cir version-decimal-1.cir polarity-values-2.cir

TESTS_ENVIRONMENT = NGSPICE_OSDI_DIR=$(abs_top_builddir)/tests/osdi/.libs ngspice_vpath=$(srcdir) $(SHELL) $(top_srcdir)/tests/bin/check.sh $(top_builddir)/src/ngspice

EXTRA_DIST = \
	$(TESTS) \
	$(TESTS:.cir=.out)

MAINTAINERCLEANFILES = Makefile.in
