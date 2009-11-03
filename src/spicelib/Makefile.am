# Process this file with automake

EXTRA_DIST = README

SUBDIRS = analysis parser devices

dist-hook:
	rm -rf `find $(distdir)/devices -name .deps`
	rm -rf `find $(distdir)/devices -name CVS`

MAINTAINERCLEANFILES = Makefile.in
