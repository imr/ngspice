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
SOI3delete(GENmodel *inModel, IFuid name, GENinstance **inst)
{
    SOI3model *model = (SOI3model *)inModel;
    SOI3instance **fast = (SOI3instance **)inst;
    SOI3instance **prev = NULL;
    SOI3instance *here;

    for( ; model ; model = model->SOI3nextModel) {
        prev = &(model->SOI3instances);
        for(here = *prev; here ; here = *prev) {
            if(here->SOI3name == name || (fast && here==*fast) ) {
                *prev= here->SOI3nextInstance;
                FREE(here);
                return(OK);
            }
            prev = &(here->SOI3nextInstance);
        }
    }
    return(E_NODEV);
}
