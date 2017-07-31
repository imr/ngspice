# -*- makefile -*-
# (compile "make -i -f my.mak to")

CFLAGS=-I ../../../../../include -I ../../../../../../../w32/src/include

SRCS = \
bsim6acld.c bsim6.analogfunction.c bsim6ask.c bsim6.c bsim6del.c bsim6dest.c bsim6guesstopology.c bsim6init.c bsim6load.c bsim6mask.c bsim6mdel.c bsim6mpar.c bsim6noise.c bsim6par.c bsim6pzld.c bsim6setup.c bsim6temp.c bsim6trunc.c

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

do : bsim6.va
	../../../../../../adms/ADMS/admsXml/admsXml  $(scripts) $<
