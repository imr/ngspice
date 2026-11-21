## Process this file with automake to produce Makefile.in

# The test decks, after the modules they load
SUBDIRS = . osdi-bind

# The same modules twice, so that modules of the same name can be loaded
check_LTLIBRARIES = dummy_modules.la dummy_modules_tenfold.la
dummy_modules_la_SOURCES = dummy_modules.c
dummy_modules_la_CPPFLAGS = -I$(top_srcdir)/src/osdi
dummy_modules_la_LDFLAGS = -module -avoid-version -shrext .osdi -rpath $(abs_builddir)
dummy_modules_tenfold_la_SOURCES = dummy_modules.c
dummy_modules_tenfold_la_CPPFLAGS = -I$(top_srcdir)/src/osdi -DCONDUCTANCE_FACTOR=10
dummy_modules_tenfold_la_LDFLAGS = $(dummy_modules_la_LDFLAGS)

MAINTAINERCLEANFILES = Makefile.in
