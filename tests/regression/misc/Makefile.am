## Process this file with automake to produce Makefile.in


TESTS = bugs-1.cir bugs-2.cir dollar-1.cir empty-1.cir resume-1.cir log-functions-1.cir alter-vec.cir test-noise-2.cir test-noise-3.cir ac-zero.cir asrc-tc-1.cir asrc-tc-2.cir if-elseif.cir

TESTS_ENVIRONMENT = ngspice_vpath=$(srcdir) $(SHELL) $(top_srcdir)/tests/bin/check.sh $(top_builddir)/src/ngspice

EXTRA_DIST = \
	$(TESTS) \
	$(TESTS:.cir=.out)

MAINTAINERCLEANFILES = Makefile.in
