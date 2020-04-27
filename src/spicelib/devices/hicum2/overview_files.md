# Overview of Spice Files for HiCUM

This file gives an overview of the files needed for the ngspice HiCUM version, e.g. their:
- intent
- status
- assignee (Mario or Markus)

# hicum2.c
# hicum2acld..c
# hicum2ask.c
# hicum2conv.c
# hicum2defs.h
# hicum2ext.h
# hicum2getic.h
# hicum2init.h
# hicum2itf.h
# hicum2load.c
# hicum2mask.c
# hicum2mpar.c
# hicum2noise.c
# hicum2param.c
# hicum2pzld.c
# hicum2setup.c
# hicum2soachk.c
# hicum2temp.c
    * Temperature scaling of all parameters
    * Models are implemented, missing are the derivatives
    * As most models are easy and just temperature dependent -> no dual numbers
    * Assignee: Mario
# hicum2trunc.c
