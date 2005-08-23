## Process this file with automake to produce Makefile.in

SUBDIRS = @XGRAPHDIR@ doc src man tests
DIST_SUBDIRS = @XGRAPHDIR@ doc src man tests

EXTRA_DIST = FAQ autogen.sh Stuarts_Poly_Notes  \
		ANALYSES BUGS AUTHORS ChangeLog \
		DEVICES NEWS README contrib patches examples @NOTXGRAPH@

MAINTAINERCLEANFILES = Makefile.in aclocal.m4 config.guess \
			config.h.in config.sub configure install-sh \
			missing mkinstalldirs stamp-h.in ltconfig \
			ltmain.sh autom4te.cache

dist-hook:
	rm -rf `find $(distdir)/contrib -name CVS`
	rm -rf `find $(distdir)/patches -name CVS`
	rm -rf `find $(distdir)/examples -name CVS`

mrproper: maintainer-clean
	rm -f `find . -type f -name "*~" -print`
	rm -f `find . -type f -name "*.orig" -print`
	rm -f `find . -type f -name "*.rej" -print`


