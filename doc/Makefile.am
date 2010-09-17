## Process this file with automake to produce Makefile.in

EXTRA_DIST = ngspice.ps ngspice.pdf

info_TEXINFOS = ngspice.texi

CLEANFILES = *.fns *.pg *.tp *.aux  *.cp *.ky *.vr *.fn *.log *.toc *.vrs

DISTCLEANFILES = $(CLEANFILES) *.ps *.dvi *.info*

MAINTAINERCLEANFILES = $(DISTCLEANFILES) Makefile.in

# This adds the root directory of the ngspice source distribution
# to be included in the search path for includes in the ngspice.texi file
# (which is necessary for the AUTHORS file to be found)
AM_MAKEINFOFLAGS = -I $(top_srcdir)
TEXI2DVI = texi2dvi -I $(top_srcdir)
