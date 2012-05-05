## Process this file with automake to produce Makefile.in


TESTS = nmos/qaSpec pmos/qaSpec

TESTS_ENVIRONMENT = \
	$(SHELL) $(top_srcdir)/tests/bin/check_cmc.sh \
	$(top_builddir)/src/ngspice

EXTRA_DIST = \
	$(TESTS) \
	nmos pmos

MAINTAINERCLEANFILES = Makefile.in

clean-local:
	rm -rf nmos/results pmos/results ngspiceCkt bsim4.out
