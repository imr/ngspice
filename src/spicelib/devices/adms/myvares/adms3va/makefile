# -*- makefile -*-

device=myvares

top_srcdir=/where/is/ngspice
srcdir=$(top_srcdir)/src/spicelib/devices/adms/$(device)/adms3va

top_builddir=/where/to/build

ADMSXMLINTERFACE=$(top_srcdir)/src/spicelib/devices/adms/admst
include $(ADMSXMLINTERFACE)/makefile.adms3


all: lib$(device).so

clean:
	rm -f $(CLEANFILES)
