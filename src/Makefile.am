## Process this file with automake to produce Makefile.in

SUBDIRS =  analysis circuit devices frontend hlp maths misc parser include

bin_PROGRAMS =  ngspice nutmeg help sconvert proc2mod multidec makeidx

EXTRA_DIST = ngspice.txt ngspice.idx

helpdatadir = $(pkgdatadir)/helpdir

helpdata_DATA = ngspice.idx ngspice.txt



DYNAMIC_DEVICELIBS = 			\
	devices/asrc/libasrc.la		\
	devices/bjt/libbjt.la		\
	devices/bsim1/libbsim1.la	\
	devices/bsim2/libbsim2.la	\
	devices/bsim3/libbsim3.la	\
	devices/bsim4/libbsim4.la	\
	devices/bsim3v1/libbsim3v1.la	\
	devices/bsim3v2/libbsim3v2.la	\
	devices/cap/libcap.la		\
	devices/cccs/libcccs.la		\
	devices/ccvs/libccvs.la		\
	devices/csw/libcsw.la		\
	devices/devsup/libdevsup.la	\
	devices/dio/libdio.la		\
	devices/disto/libdisto.la	\
	devices/ind/libind.la		\
	devices/isrc/libisrc.la		\
	devices/jfet/libjfet.la		\
	devices/jfet2/libjfet2.la	\
	devices/ltra/libltra.la		\
	devices/cccs/libcccs.la		\
	devices/ccvs/libccvs.la		\
	devices/csw/libcsw.la		\
	devices/devsup/libdevsup.la	\
	devices/dio/libdio.la		\
	devices/disto/libdisto.la	\
	devices/ind/libind.la		\
	devices/isrc/libisrc.la		\
	devices/jfet/libjfet.la		\
	devices/jfet2/libjfet2.la	\
	devices/ltra/libltra.la		\
	devices/mes/libmes.la		\
	devices/mos1/libmos1.la		\
	devices/mos2/libmos2.la		\
	devices/mos3/libmos3.la		\
	devices/mos6/libmos6.la		\
	devices/res/libres.la		\
	devices/sw/libsw.la		\
	devices/tra/libtra.la		\
	devices/urc/liburc.la		\
	devices/vccs/libvccs.la		\
	devices/vcvs/libvcvs.la		\
	devices/vsrc/libvsrc.la

## Build ngspice first:

ngspice_SOURCES = \
	conf.c \
	conf.h \
	ngspice.c

ngspice_LDADD = \
	spice.o \
	frontend/libfte.a \
	$(DYNAMIC_DEVICELIBS) \
	analysis/libckt.a \
	parser/libparser.a \
	hlp/libhlp.a \
	circuit/libinp.a \
	maths/cmaths/libcmaths.a \
	maths/ni/libni.a \
	maths/sparse/libsparse.a \
	misc/libmisc.a	


spice.o: main.c 
		$(COMPILE) -DSIMULATOR -o spice.o  -c $(srcdir)/main.c

## nutmeg:

nutmeg_SOURCES = \
	main.c \
	conf.c \
	conf.h \
	nutmeg.c

nutmeg_LDADD = \
	frontend/libfte.a \
	parser/libparser.a \
	hlp/libhlp.a \
	maths/cmaths/libcmaths.a \
	misc/libmisc.a 



## help:

help_SOURCES = help.c

help_LDADD =  \
	hlp/libhlp.a \
	parser/libparser.a \
	misc/libmisc.a 


## sconvert:

sconvert_SOURCES =  sconvert.c



sconvert_LDADD =  \
	frontend/libfte.a \
	parser/libparser.a \
	misc/libmisc.a 


## proc2mod:

proc2mod_SOURCES = proc2mod.c

proc2mod_LDADD =  \
	parser/libparser.a \
	circuit/libinp.a \
	misc/libmisc.a 



## multidec:

multidec_SOURCES = multidec.c

multidec_LDADD =  \
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


INCLUDES = -I$(top_srcdir)/src/include -I$(top_srcdir)/src/devices @X_CFLAGS@

LIBS = @LIBS@ @X_LIBS@ @X_PRE_LIBS@ @X_EXTRA_LIBS@

MAINTAINERCLEANFILES = Makefile.in
