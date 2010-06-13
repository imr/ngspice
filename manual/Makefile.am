## Process this file with automake to produce Makefile.in

dist_doc_DATA = manual.pdf

DISTCLEANFILES = manual.pdf

MAINTAINERCLEANFILES = $(DISTCLEANFILES) Makefile.in

manual.pdf :
	make -C  $(top_srcdir)/../ng-spice-manuals            manual.pdf
	cp       $(top_srcdir)/../ng-spice-manuals/manual.pdf $@
