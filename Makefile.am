## Process this file with automake to produce Makefile.in

SUBDIRS = src man tests
DIST_SUBDIRS = src man tests

EXTRA_DIST = FAQ autogen.sh Stuarts_Poly_Notes  \
		ANALYSES BUGS AUTHORS INSTALL \
		DEVICES INTERNALS NEWS README README.tcl README.optran\
		README.shared-xspice README.vdmos README_OSDI.md\
		README_OSDI_howto README.cpl-gc README.utf8 \
		examples m4 visualc \
		cross-compile.sh cross-compile-shared.sh \
		compile_min.sh compile_linux.sh compile_min_shared.sh \
		compile_linux_shared.sh compile_cyg_make_short_check_64.sh \
		compile_macos_clang.sh compile_macos_gcc.sh \
		compile_linux_dist.sh ngspice.pc.in .gitignore

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
