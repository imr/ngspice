# -*- makefile -*-
# (compile "make -i -f my.mak to")

CFLAGS=-I ../../../../../include -I ../../../../../../../w32/src/include

SRCS = \
bjt504tacld.c bjt504task.c bjt504t.c bjt504tdel.c bjt504tdest.c bjt504tguesstopology.c bjt504tinit.c bjt504tload.c bjt504tmask.c bjt504tmdel.c bjt504tmpar.c bjt504tnoise.c bjt504tpar.c bjt504tpzld.c bjt504tsetup.c bjt504ttemp.c bjt504ttrunc.c bjt504tva.analogfunction.c

to : $(SRCS:%.c=%.o)

scripts = \
	-e ../../admst/ngspiceVersion.xml \
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

do : bjt504t.va
	../../../../../../adms/ADMS/admsXml/admsXml  $(scripts) $<
