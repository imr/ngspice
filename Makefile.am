## Process this file with automake to produce Makefile.in

SUBDIRS = doc src man tests

EXTRA_DIST = FAQ acconfig.h autogen.sh NOTES  BUGS AUTHORS ChangeLog \
		DEVICES NEWS README

MAINTAINERCLEANFILES = Makefile.in aclocal.m4 config.guess \
			config.h.in config.sub configure install-sh \
			missing mkinstalldirs stamp-h.in ltconfig \
			ltmain.sh

mrproper: maintainer-clean
	rm -f `find . -type f -name "*~" -print`
	rm -f `find . -type f -name "*.orig" -print`
	rm -f `find . -type f -name "*.rej" -print`


