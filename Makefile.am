## Process this file with automake to produce Makefile.in

SUBDIRS = @XGRAPHDIR@ src man tests
DIST_SUBDIRS = @XGRAPHDIR@ src man tests

EXTRA_DIST = FAQ autogen.sh Stuarts_Poly_Notes  \
		ANALYSES BUGS AUTHORS ChangeLog \
		DEVICES NEWS README README.tcl \
		README.adms contrib \
		examples visualc @NOTXGRAPH@

MAINTAINERCLEANFILES = Makefile.in aclocal.m4 config/config.guess \
			config.h.in config/config.sub configure config/install-sh \
			config/missing mkinstalldirs stamp-h.in ltconfig \
			config/ltmain.sh autom4te.cache \
			config/depcomp config/ylwrap

ACLOCAL_AMFLAGS = -I m4

dist-hook:
	rm -rf `find $(distdir)/contrib -name CVS`
	rm -rf `find $(distdir)/patches -name CVS`
	rm -rf `find $(distdir)/examples -name CVS`
	rm -rf `find $(distdir)/visualc -name CVS`
	rm -rf `find $(distdir)/visualc/include -name CVS`
	rm -rf `find $(distdir)/xgraph -name .cvsignore`
	rm -rf `find $(distdir)/xgraph -name .deps`
	rm -rf `find $(distdir)/xgraph -name CVS`
	rm -rf `find $(distdir)/xgraph -name config.status`
	rm -rf `find $(distdir)/xgraph -name config.log`
	rm -rf `find $(distdir)/xgraph -name aclocal.m4`
	rm -rf `find $(distdir)/xgraph -name Makefile`
	rm -rf `find $(distdir)/xgraph/bitmaps -name CVS`
	rm -rf `find $(distdir)/xgraph/examples -name CVS`
	rm -rf `find $(distdir)/unsupported -name CVS`

mrproper: maintainer-clean
	rm -f `find . -type f -name "*~" -print`
	rm -f `find . -type f -name "*.orig" -print`
	rm -f `find . -type f -name "*.rej" -print`

