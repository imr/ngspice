# -*- makefile -*-
# (compile "make -i -f my.mak to")
# (compile "make -i -f my.mak do 2>&1 > log")
# (compile "make -i -f my.mak to 2>&1 > log")

CFLAGS=-I ../../../../../include -I ../../../../../../../w32/src/include

SRCS = \
bsimcmgacld.c bsimcmg.analogfunction.c bsimcmgask.c bsimcmg.c bsimcmgdel.c bsimcmgdest.c bsimcmgguesstopology.c bsimcmginit.c bsimcmgload.c bsimcmgmask.c bsimcmgmdel.c bsimcmgmpar.c bsimcmgnoise.c bsimcmgpar.c bsimcmgpzld.c bsimcmgsetup.c bsimcmgtemp.c bsimcmgtrunc.c

to : $(SRCS:%.c=%.o)

scripts = \
	-e ../../admst/ngspiceVersion.xml \
	-e ../../admst/analogfunction.xml \
	-e ../../admst/ngspiceMODULEitf.h.xml \
	-e ../../admst/ngspiceMODULEinit.c.xml \
	-e ../../admst/ngspiceMODULEinit.h.xml \
	-e ../../admst/ngspiceMODULEext.h.xml \
	-e ../../admst/ngspiceMODULEdefs.h.xml \
	-e ../../admst/ngspiceMODULEask.c.xml \
	-e ../../admst/ngspiceMODULEmask.c.xml \
	-e ../../admst/ngspiceMODULEpar.c.xml \
	-e ../../admst/ngspiceMODULEmpar.c.xml \
	-e ../../admst/ngspiceMODULEload.c.xml \
	-e ../../admst/ngspiceMODULEacld.c.xml \
	-e ../../admst/ngspiceMODULEpzld.c.xml \
	-e ../../admst/ngspiceMODULEtemp.c.xml \
	-e ../../admst/ngspiceMODULEtrunc.c.xml \
	-e ../../admst/ngspiceMODULEsetup.c.xml \
	-e ../../admst/ngspiceMODULEdel.c.xml \
	-e ../../admst/ngspiceMODULEmdel.c.xml \
	-e ../../admst/ngspiceMODULEdest.c.xml \
	-e ../../admst/ngspiceMODULEnoise.c.xml \
	-e ../../admst/ngspiceMODULEguesstopology.c.xml \
	-e ../../admst/ngspiceMODULE.hxx.xml \
	-e ../../admst/ngspiceMODULE.c.xml

$(SRCS) : do

do : bsimcmg.va
	../../../../../../adms/ADMS/admsXml/admsXml  $(scripts) $<
