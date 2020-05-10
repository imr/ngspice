# Overview of Spice Files for HiCUM

This file gives an overview of the files needed for the ngspice HiCUM version, e.g. their:
- intent
- status
- assignee (Mario or Markus)

# hicum2.c
    * Definition of the external instance and model structure.
    * Including the variables which can be accessed from the outside.
# hicum2acld.c
# hicum2ask.c
    * Define how the instance output data is saved.
# hicum2conv.c
# hicum2defs.h
    * Define the internal data structure
# hicum2ext.h
# hicum2getic.h
# hicum2init.h
# hicum2itf.h
# hicum2load.c
# hicum2mask.c
    * Define how the model output data is saved.
# hicum2mpar.c
    * Check which parameters for the model were given in the netlist. If a parameter is given, save it and set the XXXGiven flag.
# hicum2noise.c
# hicum2param.c
# hicum2pzld.c
# hicum2setup.c
# hicum2soachk.c
# hicum2temp.c
    * Temperature scaling of all parameters
    * Models are implemented, missing are the derivatives
    * As most models are easy and just temperature dependent -> no dual numbers -> Markus: I disagree
    * Assignee: Mario
# hicum2trunc.c

# hicumL2.cpp
Implemented equivalent circuit elements:
- Ijbei
- Ijbci
- Cjei
- Cjci
- It
- Crbi (Mario check this)
- Iavl 
- Ibhrec 
- rbi 
- Ijbep 
- Ijbep 
- Ijbcx 
- Cjcx 
- Cjs 
- Cjep 
- Ibet 
- Ijsc
Missing:
- Ibpsi

# Working in the DC case without self heating:
See test case in DMT where this is compared against ADS.
- re
- Ibiei
- Ibici

# useful stuff
- non-ancient explanation how equation system of spice looks:
https://spicesharp.github.io/SpiceSharp/articles/custom_components/modified_nodal_analysis.html#nonlinear-components
