# -*- makefile -*-
# (compile "make -i -f my.mak to")

CFLAGS=-I ../../../../../include -I ../../../../../../../w32/src/include

SRCS = \
psp102acld.c psp102ask.c psp102.c psp102del.c psp102dest.c psp102guesstopology.c psp102init.c psp102load.c psp102mask.c psp102mdel.c psp102mpar.c psp102noise.c psp102par.c psp102pzld.c psp102setup.c psp102temp.c psp102trunc.c

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

to : $(SRCS:%.c=%.o)

$(SRCS) : do

do : psp102.va
	../../../../../../adms/ADMS/admsXml/admsXml  $(scripts) $<

