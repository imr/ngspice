/**********
STAG version 2.7
Copyright 2000 owned by the United Kingdom Secretary of State for Defence
acting through the Defence Evaluation and Research Agency.
Developed by :     Jim Benson,
                   Department of Electronics and Computer Science,
                   University of Southampton,
                   United Kingdom.
With help from :   Nele D'Halleweyn, Ketan Mistry, Bill Redman-White, and Craig Easson.

Based on STAG version 2.1
Developed by :     Mike Lee,
With help from :   Bernard Tenbroek, Bill Redman-White, Mike Uren, Chris Edwards
                   and John Bunyan.
Acknowledgements : Rupert Howes and Pete Mole.
**********/

/********** 
Modified by Paolo Nenzi 2002
ngspice integration
**********/

#include "ngspice.h"
#include "soi3defs.h"
#include "sperror.h"
#include "suffix.h"


int
SOI3mDelete(GENmodel **inModel, IFuid modname, GENmodel *kill)
{
    SOI3model **model = (SOI3model **)inModel;
    SOI3model *modfast = (SOI3model *)kill;
    SOI3instance *here;
    SOI3instance *prev = NULL;
    SOI3model **oldmod;
    oldmod = model;
    for( ; *model ; model = &((*model)->SOI3nextModel)) {
        if( (*model)->SOI3modName == modname ||
                (modfast && *model == modfast) ) goto delgot;
        oldmod = model;
    }
    return(E_NOMOD);

delgot:
    *oldmod = (*model)->SOI3nextModel; /* cut deleted device out of list */
    for(here = (*model)->SOI3instances ; here ; here = here->SOI3nextInstance) {
        if(prev) FREE(prev);
        prev = here;
    }
    if(prev) FREE(prev);
    FREE(*model);
    return(OK);

}
