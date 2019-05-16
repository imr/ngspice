## Process this file with automake to produce Makefile.in


TESTS = hicum2/qaSpec

TESTS_ENVIRONMENT = \
	$(SHELL) $(top_srcdir)/tests/bin/check_cmc.sh \
	$(top_builddir)/src/ngspice

EXTRA_DIST = \
	$(TESTS) \
	hicum2

MAINTAINERCLEANFILES = Makefile.in

clean-local:
	rm -rf hicum2/results ngspiceCkt
