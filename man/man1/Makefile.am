## Process this file with automake to produce Makefile.in


man_MANS =

if XSPICE_WANTED
man_MANS += cmpp.1
endif XSPICE_WANTED

if !TCL_MODULE
if !SHARED_MODULE
man_MANS += ngspice.1
if OLDAPPS
man_MANS += ngnutmeg.1 ngsconvert.1 ngmultidec.1 ngmakeidx.1 ngproc2mod.1
endif OLDAPPS
endif !SHARED_MODULE
endif !TCL_MODULE

EXTRA_DIST = ngspice.1 ngnutmeg.1 ngsconvert.1 ngmultidec.1 cmpp.1 ngmakeidx.1 ngproc2mod.1

MAINTAINERCLEANFILES = Makefile.in
