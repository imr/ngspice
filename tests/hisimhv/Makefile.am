## Process this file with automake to produce Makefile.in


TESTS = nmos/qaSpec.basic pmos/qaSpec.basic

TESTS_ENVIRONMENT = \
	$(SHELL) $(top_srcdir)/tests/bin/check_cmc.sh \
	$(top_builddir)/src/ngspice

EXTRA_DIST = \
	$(TESTS) \
	$(TESTS:.cir=.out)

MAINTAINERCLEANFILES = Makefile.in

clean-local:
	rm -rf results ngspiceCkt
