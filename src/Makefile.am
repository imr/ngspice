## Process this file with automake to produce Makefile.in

SUBDIRS = misc maths frontend spicelib include

bin_PROGRAMS =  ngspice ngnutmeg nghelp ngsconvert ngproc2mod ngmultidec makeidx

EXTRA_DIST = ngspice.txt ngspice.idx spinit setplot spectrum

helpdatadir = $(pkgdatadir)/helpdir

helpdata_DATA = ngspice.idx ngspice.txt

initdatadir = $(pkgdatadir)/scripts

initdata_DATA = spinit setplot spectrum


DYNAMIC_DEVICELIBS = 			\
	spicelib/devices/asrc/libasrc.la		\
	spicelib/devices/bjt/libbjt.la		\
##	spicelib/devices/bjt2/libbjt2.la		\
	spicelib/devices/bsim1/libbsim1.la	\
	spicelib/devices/bsim2/libbsim2.la	\
	spicelib/devices/bsim3/libbsim3.la	\
	spicelib/devices/bsim3v1/libbsim3v1.la	\
	spicelib/devices/bsim3v2/libbsim3v2.la	\
	spicelib/devices/bsim4/libbsim4.la	\
	spicelib/devices/cap/libcap.la		\
	spicelib/devices/bsim3soi_pd/libbsim3soipd.la	\
	spicelib/devices/bsim3soi_fd/libbsim3soifd.la   \
	spicelib/devices/bsim3soi_dd/libbsim3soidd.la	\
	spicelib/devices/cccs/libcccs.la		\
	spicelib/devices/ccvs/libccvs.la		\
	spicelib/devices/ccvs/libccvs.la		\
	spicelib/devices/csw/libcsw.la		\
	spicelib/devices/dio/libdio.la		\
	spicelib/devices/ind/libind.la		\
	spicelib/devices/isrc/libisrc.la		\
	spicelib/devices/jfet/libjfet.la		\
	spicelib/devices/jfet2/libjfet2.la	\
	spicelib/devices/ltra/libltra.la		\
	spicelib/devices/mes/libmes.la		\
	spicelib/devices/mos1/libmos1.la		\
	spicelib/devices/mos2/libmos2.la		\
	spicelib/devices/mos3/libmos3.la		\
	spicelib/devices/mos6/libmos6.la		\
	spicelib/devices/mos9/libmos9.la		\
	spicelib/devices/res/libres.la		\
	spicelib/devices/sw/libsw.la		\
	spicelib/devices/tra/libtra.la		\
	spicelib/devices/urc/liburc.la		\
	spicelib/devices/vccs/libvccs.la		\
	spicelib/devices/vcvs/libvcvs.la		\
	spicelib/devices/vsrc/libvsrc.la

## Build ngspice first:

ngspice_SOURCES = \
	conf.c \
	conf.h \
	ngspice.c

ngspice_LDADD = \
	spice.o \
	frontend/libfte.a \
	frontend/plotting/libplotting.a \
	$(DYNAMIC_DEVICELIBS) \
	spicelib/analysis/libckt.a \
	spicelib/devices/libdev.a \
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
