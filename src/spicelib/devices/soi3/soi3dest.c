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
#include "suffix.h"


void
SOI3destroy(GENmodel **inModel)
{
    SOI3model **model = (SOI3model**)inModel;
    SOI3instance *here;
    SOI3instance *prev = NULL;
    SOI3model *mod = *model;
    SOI3model *oldmod = NULL;

    for( ; mod ; mod = mod->SOI3nextModel) {
        if(oldmod) FREE(oldmod);
        oldmod = mod;
        prev = (SOI3instance *)NULL;
        for(here = mod->SOI3instances ; here ; here = here->SOI3nextInstance) {
            if(prev){
              /*  if(prev->SOI3sens) FREE(prev->SOI3sens);  */
                FREE(prev);
            }
            prev = here;
        }
        if(prev) FREE(prev);
    }
    if(oldmod) FREE(oldmod);
    *model = NULL;
}
