## Process this file with automake to produce Makefile.in

SUBDIRS = @XGRAPHDIR@ src man tests manual
DIST_SUBDIRS = @XGRAPHDIR@ src man tests manual

EXTRA_DIST = FAQ autogen.sh Stuarts_Poly_Notes  \
		ANALYSES BUGS AUTHORS ChangeLog \
		DEVICES NEWS README README.tcl \
		README.adms contrib \
		examples visualc @NOTXGRAPH@ \
		libtool

MAINTAINERCLEANFILES = Makefile.in aclocal.m4 config.guess \
			config.h.in config.sub configure install-sh \
			missing mkinstalldirs stamp-h.in ltconfig \
			ltmain.sh autom4te.cache

dist-hook:
	rm -rf `find $(distdir)/contrib -name CVS`
	rm -rf `find $(distdir)/patches -name CVS`
	rm -rf `find $(distdir)/examples -name CVS`
	rm -rf `find $(distdir)/visualc -name CVS`
	rm -rf `find $(distdir)/visualc/include -name CVS`
	rm -rf `find $(distdir)/xgraph -name CVS`
	rm -rf `find $(distdir)/xgraph/bitmaps -name CVS`
	rm -rf `find $(distdir)/xgraph/examples -name CVS`
	rm -rf `find $(distdir)/unsupported -name CVS`
	rm -rf `find $(distdir)/xgraph -name .cvsignore`
	rm -rf `find $(distdir)/xgraph -name .deps`

mrproper: maintainer-clean
	rm -f `find . -type f -name "*~" -print`
	rm -f `find . -type f -name "*.orig" -print`
	rm -f `find . -type f -name "*.rej" -print`

