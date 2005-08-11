## Process this file with automake to produce Makefile.in

EXTRA_DIST = ngspice.ps

info_TEXINFOS = ngspice.texi

CLEANFILES = *.fns *.pg *.tp *.aux  *.cp *.ky *.vr *.fn *.log *.toc *.vrs 

DISTCLEANFILES = $CLEANFILES *.ps *.dvi *.info*

MAINTAINERCLEANFILES = $DISTCLEANFILES Makefile.in

AM_MAKEINFOFLAGS = -D "top_srcdir .."
