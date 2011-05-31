## Process this file with automake to produce Makefile.in


TESTS = nmos/qaSpec pmos/qaSpec

TESTS_ENVIRONMENT = \
	$(SHELL) $(top_srcdir)/tests/bin/check_cmc.sh \
	$(top_builddir)/src/ngspice

EXTRA_DIST = \
	$(TESTS) \
	nmos pmos \
	test1.cir \
	test1.out

MAINTAINERCLEANFILES = Makefile.in

clean-local:
	rm -rf results ngspiceCkt
