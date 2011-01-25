## Process this file with automake to produce Makefile.in

TESTS = \
	d_ram.cir          \
	d_source.cir       \
	d_state.cir

TESTS_ENVIRONMENT = ngspice_vpath=$(srcdir) SPICE_SCRIPTS=. $(SHELL) $(top_srcdir)/tests/bin/check.sh "$(top_builddir)/src/ngspice -r foobaz"

EXTRA_DIST = \
	d_ram-stimulus.txt \
	d_source-stimulus.txt \
	d_state-stimulus.txt \
	d_state-updn.txt \
	$(TESTS) \
	$(TESTS:.cir=.out)

MAINTAINERCLEANFILES = Makefile.in
