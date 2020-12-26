## Process this file with automake to produce Makefile.in


TESTS = npn/qaSpec

TESTS_ENVIRONMENT = \
	$(SHELL) $(top_srcdir)/tests/bin/check_cmc.sh \
	$(top_builddir)/src/ngspice

EXTRA_DIST = \
	$(TESTS) \
	npn

MAINTAINERCLEANFILES = Makefile.in

clean-local:
	rm -rf npn/results ngspiceCkt
