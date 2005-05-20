## Process this file with automake to produce Makefile.in

SUBDIRS = misc maths frontend spicelib include @XSPICEDIR@ @CIDERDIR@
DIST_SUBDIRS = misc maths frontend spicelib include xspice ciderlib

bin_PROGRAMS =  ngspice ngnutmeg nghelp makeidx
	        	        
if !WINDOWS
bin_PROGRAMS += ngsconvert ngproc2mod ngmultidec
endif

EXTRA_DIST = ngspice.txt ngspice.idx setplot spectrum \
	     devload devaxis ciderinit winmain.c

helpdatadir = $(pkgdatadir)/helpdir

helpdata_DATA = ngspice.txt ngspice.idx

initdatadir = $(pkgdatadir)/scripts

initdata_DATA = spinit setplot spectrum @CIDERSCRIPTS@


DYNAMIC_DEVICELIBS = 					\
	spicelib/devices/asrc/libasrc.a			\
	spicelib/devices/bjt/libbjt.a			\
	spicelib/devices/bjt2/libbjt2.a			\
	spicelib/devices/bsim1/libbsim1.a		\
	spicelib/devices/bsim2/libbsim2.a		\
	spicelib/devices/bsim3/libbsim3.a		\
	spicelib/devices/bsim3v0/libbsim3v0.a		\
	spicelib/devices/bsim3v1/libbsim3v1.a		\
	spicelib/devices/bsim3v1s/libbsim3v1s.a		\
	spicelib/devices/bsim3v1a/libbsim3v1a.a		\
	spicelib/devices/bsim4/libbsim4.a		\
	spicelib/devices/cap/libcap.a			\
	spicelib/devices/bsim3soi/libbsim3soi.a		\
	spicelib/devices/bsim3soi_pd/libbsim3soipd.a	\
	spicelib/devices/bsim3soi_fd/libbsim3soifd.a   	\
	spicelib/devices/bsim3soi_dd/libbsim3soidd.a	\
	spicelib/devices/cccs/libcccs.a			\
	spicelib/devices/ccvs/libccvs.a			\
	spicelib/devices/ccvs/libccvs.a			\
	spicelib/devices/cpl/libcpl.a			\
	spicelib/devices/csw/libcsw.a			\
	spicelib/devices/dio/libdio.a			\
	@EKVLIB@					\
	spicelib/devices/ind/libind.a			\
	spicelib/devices/isrc/libisrc.a			\
	spicelib/devices/hfet1/libhfet.a		\
	spicelib/devices/hfet2/libhfet2.a		\
	spicelib/devices/hisim/libhisim.a		\
	spicelib/devices/jfet/libjfet.a			\
	spicelib/devices/jfet2/libjfet2.a		\
	spicelib/devices/ltra/libltra.a			\
	spicelib/devices/mes/libmes.a			\
	spicelib/devices/mesa/libmesa.a         	\
	spicelib/devices/mos1/libmos1.a			\
	spicelib/devices/mos2/libmos2.a			\
	spicelib/devices/mos3/libmos3.a			\
	spicelib/devices/mos6/libmos6.a			\
	spicelib/devices/mos9/libmos9.a			\
	spicelib/devices/res/libres.a			\
	spicelib/devices/soi3/libsoi3.a			\
	spicelib/devices/sw/libsw.a			\
	spicelib/devices/txl/libtxl.a			\
	spicelib/devices/tra/libtra.a			\
	spicelib/devices/urc/liburc.a			\
	spicelib/devices/vbic/libvbic.a			\
	spicelib/devices/vccs/libvccs.a			\
	spicelib/devices/vcvs/libvcvs.a			\
	spicelib/devices/vsrc/libvsrc.a			\
	@NUMDEV@
	
	
## Build ngspice first:

ngspice_SOURCES = \
	conf.c \
	conf.h \
	ngspice.c
	
if WINDOWS
ngspice_SOURCES += winmain.c
endif

ngspice_LDADD = \
	spice.o \
	frontend/libfte.a \
	@WINDISPLIB@ \
	frontend/plotting/libplotting.a \
	@XSPICELIB1@ \
	spicelib/devices/dev.o \
	$(DYNAMIC_DEVICELIBS)  \
	spicelib/analysis/libckt.a \
	spicelib/devices/libdev.a \
	@XSPICELIB2@		  \
	frontend/parser/libparser.a \
	@NUMPARAMLIB@           \
	frontend/help/libhlp.a \
	spicelib/parser/libinp.a \
	@CIDERSIM@		\
	maths/deriv/libderiv.a \
	maths/cmaths/libcmaths.a \
	@CIDERMATHDIR@          \
	maths/poly/libpoly.a \
	maths/ni/libni.a \
	maths/sparse/libsparse.a \
	misc/libmisc.a


winmain.o: winmain.c
		$(COMPILE) -DSIMULATOR -o winmain.o  -c $(srcdir)/winmain.c

spice.o: main.c
		$(COMPILE) -DSIMULATOR -o spice.o  -c $(srcdir)/main.c

## nutmeg:

ngnutmeg_SOURCES = \
	main.c \
	conf.c \
	conf.h \
	ngnutmeg.c

if WINDOWS
ngnutmeg_SOURCES += winmain.c
endif

ngnutmeg_LDADD = \
	frontend/libfte.a \
	@WINDISPLIB@ \
	frontend/plotting/libplotting.a \
	frontend/parser/libparser.a \
	@NUMPARAMLIB@		\
	frontend/help/libhlp.a \
	maths/cmaths/libcmaths.a \
	maths/poly/libpoly.a \
	misc/libmisc.a

## help:

nghelp_SOURCES = conf.c nghelp.c

if WINDOWS
nghelp_SOURCES += winmain.c
endif

nghelp_LDADD =  \
	frontend/help/libhlp.a \
	frontend/parser/libparser.a \
	frontend/libfte.a \
	misc/libmisc.a 

## These programs currently fail to build on Windows
if !WINDOWS 
	
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
	
endif !WINDOWS

## makeidx:

makeidx_SOURCES = makeidx.c

## create index for online help:

ngspice.idx: makeidx
	./makeidx $(srcdir)/ngspice.txt


## General Includes and libraries:


INCLUDES = -I$(top_srcdir)/src/include -I$(top_srcdir)/src/spicelib/devices @X_CFLAGS@

LIBS = @LIBS@ @X_LIBS@ @X_PRE_LIBS@ @X_EXTRA_LIBS@

MAINTAINERCLEANFILES = Makefile.in
