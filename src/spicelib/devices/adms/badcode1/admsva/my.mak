# -*- makefile -*-
# (compile "make -i -f my.mak to")

CFLAGS=-I ../../../../../include -I ../../../../../../../w32/src/include

SRCS = \
r2_cmcacld.c r2_cmc.analogfunction.c r2_cmcask.c r2_cmc.c r2_cmcdel.c r2_cmcdest.c r2_cmcguesstopology.c r2_cmcinit.c r2_cmcload.c r2_cmcmask.c r2_cmcmdel.c r2_cmcmpar.c r2_cmcnoise.c r2_cmcpar.c r2_cmcpzld.c r2_cmcsetup.c r2_cmctemp.c r2_cmctrunc.c

to : $(SRCS:%.c=%.o)

scripts = \
	-x \
	-e ../../admst/adms.implicit.xml \
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

do : r2_cmc.va
	../../../../../../adms/ADMS/admsXml/admsXml  $(scripts) $<
