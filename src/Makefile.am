## Process this file with automake to produce Makefile.in

SUBDIRS = misc maths frontend spicelib include @XSPICEDIR@

bin_PROGRAMS =  ngspice ngnutmeg nghelp ngsconvert ngproc2mod ngmultidec makeidx

EXTRA_DIST = ngspice.txt spinit setplot spectrum

helpdatadir = $(pkgdatadir)/helpdir

helpdata_DATA = ngspice.txt

initdatadir = $(pkgdatadir)/scripts

initdata_DATA = spinit setplot spectrum


DYNAMIC_DEVICELIBS = 			\
	spicelib/devices/asrc/libasrc.a		\
	spicelib/devices/bjt/libbjt.a		\
	spicelib/devices/bsim1/libbsim1.a	\
	spicelib/devices/bsim2/libbsim2.a	\
	spicelib/devices/bsim3/libbsim3.a	\
	spicelib/devices/bsim3v1/libbsim3v1.a	\
	spicelib/devices/bsim3v2/libbsim3v2.a	\
	spicelib/devices/bsim4/libbsim4.a	\
	spicelib/devices/cap/libcap.a		\
	spicelib/devices/bsim3soi_pd/libbsim3soipd.a	\
	spicelib/devices/bsim3soi_fd/libbsim3soifd.a   \
	spicelib/devices/bsim3soi_dd/libbsim3soidd.a	\
	spicelib/devices/cccs/libcccs.a		\
	spicelib/devices/ccvs/libccvs.a		\
	spicelib/devices/ccvs/libccvs.a		\
	spicelib/devices/cpl/libcpl.a                  \
	spicelib/devices/csw/libcsw.a		\
	spicelib/devices/dio/libdio.a		\
	@EKVLIB@				\
	spicelib/devices/ind/libind.a		\
	spicelib/devices/isrc/libisrc.a		\
	spicelib/devices/hfet1/libhfet.a		\
	spicelib/devices/hfet2/libhfet2.a		\
	spicelib/devices/jfet/libjfet.a		\
	spicelib/devices/jfet2/libjfet2.a	\
	spicelib/devices/ltra/libltra.a		\
	spicelib/devices/mes/libmes.a		\
	spicelib/devices/mesa/libmesa.a         \
	spicelib/devices/mos1/libmos1.a		\
	spicelib/devices/mos2/libmos2.a		\
	spicelib/devices/mos3/libmos3.a		\
	spicelib/devices/mos6/libmos6.a		\
	spicelib/devices/mos9/libmos9.a		\
	spicelib/devices/res/libres.a		\
	spicelib/devices/soi3/libsoi3.a	\
	spicelib/devices/sw/libsw.a		\
	spicelib/devices/txl/libtxl.a          \
	spicelib/devices/tra/libtra.a		\
	spicelib/devices/urc/liburc.a		\
	spicelib/devices/vccs/libvccs.a		\
	spicelib/devices/vcvs/libvcvs.a		\
	spicelib/devices/vsrc/libvsrc.a



## ----- Note that I moved this stuff to here because it was causing automake
##       to choke when it was in the DYNAMIC_DEVICELIBS list above -----
## This lib deleted from DYNAMIC_DEVICELIBS by sdb 'cause there's no source for it.
##	spicelib/devices/bjt2/libbjt2.a		\
## poly added to dynamic libs by SDB on 6.1.2003
##	xspice/icm/poly/libpoly.a

## Build ngspice first:

ngspice_SOURCES = \
	conf.c \
	conf.h \
	ngspice.c

ngspice_LDADD = \
	spice.o \
	frontend/libfte.a \
	frontend/plotting/libplotting.a \
	@XSPICELIB1@ \
	spicelib/devices/dev.o \
	$(DYNAMIC_DEVICELIBS)  \
	spicelib/analysis/libckt.a \
	spicelib/devices/libdev.a \
	@XSPICELIB2@		  \
	frontend/parser/libparser.a \
	frontend/help/libhlp.a \
	spicelib/parser/libinp.a \
	maths/deriv/libderiv.a \
	maths/cmaths/libcmaths.a \
	maths/poly/libpoly.a \
	maths/ni/libni.a \
	maths/sparse/libsparse.a \
	misc/libmisc.a

spice.o: main.c 
		$(COMPILE) -DSIMULATOR -o spice.o  -c $(srcdir)/main.c

## nutmeg:

ngnutmeg_SOURCES = \
	main.c \
	conf.c \
	conf.h \
	ngnutmeg.c

ngnutmeg_LDADD = \
	frontend/libfte.a \
	frontend/plotting/libplotting.a \
	frontend/parser/libparser.a \
	frontend/help/libhlp.a \
	maths/cmaths/libcmaths.a \
	maths/poly/libpoly.a \
	misc/libmisc.a 

## help:

nghelp_SOURCES = nghelp.c

nghelp_LDADD =  \
	frontend/help/libhlp.a \
	frontend/parser/libparser.a \
	frontend/libfte.a \
	misc/libmisc.a 

## sconvert:

ngsconvert_SOURCES =  ngsconvert.c



ngsconvert_LDADD =  \
	frontend/libfte.a \
	frontend/parser/libparser.a \
	misc/libmisc.a

## proc2mod:

ngproc2mod_SOURCES = ngproc2mod.c

ngproc2mod_LDADD =  \
	frontend/parser/libparser.a \
	spicelib/parser/libinp.a \
	misc/libmisc.a 



## multidec:

ngmultidec_SOURCES = ngmultidec.c

ngmultidec_LDADD =  \
	maths/sparse/libsparse.a \
	misc/libmisc.a 

## makeidx:

makeidx_SOURCES = makeidx.c

makeidx_LDADD =  \
	misc/libmisc.a 

## create index for online help:

all:
	./makeidx $(srcdir)/ngspice.txt


## General Includes and libraries:


INCLUDES = -I$(top_srcdir)/src/include -I$(top_srcdir)/src/spicelib/devices @X_CFLAGS@

LIBS = @LIBS@ @X_LIBS@ @X_PRE_LIBS@ @X_EXTRA_LIBS@

MAINTAINERCLEANFILES = Makefile.in

CLEANFILES = pkgIndex.tcl libspice.so

TCL_PKG_PATH = @TCL_PACKAGE_PATH@

TCLSPICE_VERSION = 0.2.12

TCL_FILES = libspice.so pkgIndex.tcl

LIBSPICE_OBJS = tclspice.o

pkgIndex.tcl: pkgIndex.tcl.in
	rm -f pkgIndex.tcl
	sed -e 's;%LIB_DIR%;$(TCL_PKG_PATH);g' pkgIndex.tcl.in | \
	sed -e 's;%VERSION%;$(TCLSPICE_VERSION);g' > pkgIndex.tcl

install-tcl: install-tcl-recursive install-tclspice

install-tcl-recursive:
	list='$(SUBDIRS)'; for subdir in $$list; do \
	  echo "Making $$target in $$subdir"; \
	  (cd $$subdir && $(MAKE) $(AM_MAKEFLAGS) install) \
	   || case "$$amf" in *=*) exit 1;; *k*) fail=yes;; *) exit 1;; esac; \
	done; 

install-tclspice: ${TCL_FILES}
	$(mkinstalldirs) $(DESTDIR)$(TCL_PKG_PATH)
	$(INSTALL_DATA) libspice.so $(DESTDIR)$(TCL_PKG_PATH)
	$(mkinstalldirs) $(DESTDIR)$(TCL_PKG_PATH)/spice
	$(INSTALL_DATA) pkgIndex.tcl $(DESTDIR)$(TCL_PKG_PATH)/spice
	$(INSTALL_DATA) tclspice_source.tcl $(DESTDIR)$(TCL_PKG_PATH)/spice
	$(mkinstalldirs) $(DESTDIR)$(bindir)
	$(INSTALL_PROGRAM) spicewish $(DESTDIR)$(bindir)

tclspice.o: tclspice.c
	$(COMPILE) -c -fpic tclspice.c -DTCLSPICE_version="\"$(TCLSPICE_VERSION)\""

libspice.so:  $(ngspice_OBJECTS) $(LIBSPICE_OBJS) $(ngspice_DEPENDENCIES)
	$(LINK) $(ngspice_LDFLAGS) $(ngspice_OBJECTS) $(ngspice_LDADD) $(LIBS) $(LIBSPICE_OBJS) -shared -lpthread -Wl,--version-script=tclspice.map
	if test -f .libs/$@ ; then \
	mv .libs/$@ ./ ;\
	rmdir .libs ;\
	fi

tcl: tcl-recursive

tcl-recursive:
	list='$(SUBDIRS)'; for subdir in $$list; do \
	echo "Making $$target in $$subdir"; \
	(cd $$subdir && $(MAKE) $(AM_MAKEFLAGS) all) \
	|| case "$$amf" in *=*) exit 1;; *k*) fail=yes;; *) exit 1;; esac; \
	done;	
	make ${TCL_FILES}
