##
## Interface: ngspice 1.0.0.0
## created by: admsXml-2.3.6

## Process this file with automake to produce Makefile.in

ADMSXMLINTERFACE=$(srcdir)/../admst

noinst_LTLIBRARIES = libbsimcmg.la

BUILT_SOURCES = \
	bsimcmg.c	\
	bsimcmg.hxx	\
	bsimcmgacld.c	\
	bsimcmgask.c	\
	bsimcmgdefs.h	\
	bsimcmgdel.c	\
	bsimcmgdest.c	\
	bsimcmgext.h	\
	bsimcmgguesstopology.c	\
	bsimcmginit.c	\
	bsimcmginit.h	\
	bsimcmgitf.h	\
	bsimcmgload.c	\
	bsimcmgmask.c	\
	bsimcmgmdel.c	\
	bsimcmgmpar.c	\
	bsimcmgpar.c	\
	bsimcmgpzld.c	\
	bsimcmgsetup.c	\
	bsimcmgtemp.c	\
	bsimcmgtrunc.c

libbsimcmg_la_SOURCES = 	\
	$(BUILT_SOURCES)

CLEANFILES = \
	$(BUILT_SOURCES)	\
	bsimcmgnoise.c	\
	.bsimcmg.va.adms \
	.adms.implicit.xml \
	.interface.xml \
	*.h \
	*.xml

#TODO (not implemented) \
	bsimcmgconv.c	\
	bsimcmggetic.c

AM_CPPFLAGS = @AM_CPPFLAGS@ -I$(top_srcdir)/src/include
AM_CFLAGS = $(STATIC)

MAINTAINERCLEANFILES = Makefile.in \
	Makefile.am

DISTCLEANFILES = Makefile.am \
	Makefile.in

%.c %.hxx \
	%acld.c %ask.c %defs.h %del.c %dest.c %ext.h %guesstopology.c \
	%init.c %init.h %itf.h %load.c %mask.c %mdel.c %mpar.c %par.c \
	%pzld.c %setup.c %temp.c %trunc.c \
 : $(srcdir)/admsva/%.va \
	$(ADMSXMLINTERFACE)/ngspiceVersion.xml \
	$(ADMSXMLINTERFACE)/analogfunction.xml \
	$(ADMSXMLINTERFACE)/ngspiceMODULEitf.h.xml \
	$(ADMSXMLINTERFACE)/ngspiceMODULEinit.c.xml \
	$(ADMSXMLINTERFACE)/ngspiceMODULEinit.h.xml \
	$(ADMSXMLINTERFACE)/ngspiceMODULEext.h.xml \
	$(ADMSXMLINTERFACE)/ngspiceMODULEdefs.h.xml \
	$(ADMSXMLINTERFACE)/ngspiceMODULEask.c.xml \
	$(ADMSXMLINTERFACE)/ngspiceMODULEmask.c.xml \
	$(ADMSXMLINTERFACE)/ngspiceMODULEpar.c.xml \
	$(ADMSXMLINTERFACE)/ngspiceMODULEmpar.c.xml \
	$(ADMSXMLINTERFACE)/ngspiceMODULEload.c.xml \
	$(ADMSXMLINTERFACE)/ngspiceMODULEacld.c.xml \
	$(ADMSXMLINTERFACE)/ngspiceMODULEpzld.c.xml \
	$(ADMSXMLINTERFACE)/ngspiceMODULEtemp.c.xml \
	$(ADMSXMLINTERFACE)/ngspiceMODULEtrunc.c.xml \
	$(ADMSXMLINTERFACE)/ngspiceMODULEsetup.c.xml \
	$(ADMSXMLINTERFACE)/ngspiceMODULEdel.c.xml \
	$(ADMSXMLINTERFACE)/ngspiceMODULEmdel.c.xml \
	$(ADMSXMLINTERFACE)/ngspiceMODULEdest.c.xml \
	$(ADMSXMLINTERFACE)/ngspiceMODULEnoise.c.xml \
	$(ADMSXMLINTERFACE)/ngspiceMODULEguesstopology.c.xml \
	$(ADMSXMLINTERFACE)/ngspiceMODULE.hxx.xml \
	$(ADMSXMLINTERFACE)/ngspiceMODULE.c.xml
	$(ADMSXML) -I$(srcdir) -I$(srcdir)/admsva $< \
	-e $(ADMSXMLINTERFACE)/ngspiceVersion.xml \
	-e $(ADMSXMLINTERFACE)/analogfunction.xml \
	-e $(ADMSXMLINTERFACE)/ngspiceMODULEitf.h.xml \
	-e $(ADMSXMLINTERFACE)/ngspiceMODULEinit.c.xml \
	-e $(ADMSXMLINTERFACE)/ngspiceMODULEinit.h.xml \
	-e $(ADMSXMLINTERFACE)/ngspiceMODULEext.h.xml \
	-e $(ADMSXMLINTERFACE)/ngspiceMODULEdefs.h.xml \
	-e $(ADMSXMLINTERFACE)/ngspiceMODULEask.c.xml \
	-e $(ADMSXMLINTERFACE)/ngspiceMODULEmask.c.xml \
	-e $(ADMSXMLINTERFACE)/ngspiceMODULEpar.c.xml \
	-e $(ADMSXMLINTERFACE)/ngspiceMODULEmpar.c.xml \
	-e $(ADMSXMLINTERFACE)/ngspiceMODULEload.c.xml \
	-e $(ADMSXMLINTERFACE)/ngspiceMODULEacld.c.xml \
	-e $(ADMSXMLINTERFACE)/ngspiceMODULEpzld.c.xml \
	-e $(ADMSXMLINTERFACE)/ngspiceMODULEtemp.c.xml \
	-e $(ADMSXMLINTERFACE)/ngspiceMODULEtrunc.c.xml \
	-e $(ADMSXMLINTERFACE)/ngspiceMODULEsetup.c.xml \
	-e $(ADMSXMLINTERFACE)/ngspiceMODULEdel.c.xml \
	-e $(ADMSXMLINTERFACE)/ngspiceMODULEmdel.c.xml \
	-e $(ADMSXMLINTERFACE)/ngspiceMODULEdest.c.xml \
	-e $(ADMSXMLINTERFACE)/ngspiceMODULEnoise.c.xml \
	-e $(ADMSXMLINTERFACE)/ngspiceMODULEguesstopology.c.xml \
	-e $(ADMSXMLINTERFACE)/ngspiceMODULE.hxx.xml \
	-e $(ADMSXMLINTERFACE)/ngspiceMODULE.c.xml
