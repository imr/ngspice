## Process this file with automake to produce Makefile.in

TESTS = alter-1.cmd

TESTS_ENVIRONMENT = \
	SPICE_SCRIPTS=$(top_srcdir)/tests/bin \
	ngspice_vpath=$(srcdir) \
	$(SHELL) -c '$(top_builddir)/src/ngspice -p < $$1' --

EXTRA_DIST = \
	$(TESTS) \
    alter-1.cir

MAINTAINERCLEANFILES = Makefile.in
