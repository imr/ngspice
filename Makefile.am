## Process this file with automake to produce Makefile.in

SUBDIRS = src man tests
DIST_SUBDIRS = src man tests

EXTRA_DIST = FAQ autogen.sh Stuarts_Poly_Notes  \
		ANALYSES BUGS AUTHORS ChangeLog \
		DEVICES NEWS README README.tcl \
		README.shared-xspice README.vdmos \
		README.adms README-old.adms \
		examples m4 visualc \
		build-for-mac-os.sh \
		cross-compile.sh cross-compile-shared.sh \
		compile_min.sh compile_linux.sh compile_min_shared.sh \
		compile_macos.sh ngspice.pc.in .gitignore

MAINTAINERCLEANFILES = Makefile.in aclocal.m4 ar-lib compile config.guess \
			src/include/ngspice/config.h.in config.sub configure install-sh \
			missing mkinstalldirs stamp-h.in ltconfig \
			ltmain.sh depcomp ylwrap ngspice.pc

ACLOCAL_AMFLAGS = -I m4

if SHARED_MODULE
pkgconfigdir = $(libdir)/pkgconfig
pkgconfig_DATA = ngspice.pc
endif SHARED_MODULE

mrproper: maintainer-clean
	rm -f `find $(top_srcdir) -type f -name "*~" -print`
	rm -f `find $(top_srcdir) -type f -name "*.orig" -print`
	rm -f `find $(top_srcdir) -type f -name "*.rej" -print`
