/**********
STAG version 2.6
Copyright 2000 owned by the United Kingdom Secretary of State for Defence
acting through the Defence Evaluation and Research Agency.
Developed by :     Jim Benson,
                   Department of Electronics and Computer Science,
                   University of Southampton,
                   United Kingdom.
With help from :   Nele D'Halleweyn, Bill Redman-White, and Craig Easson.

Based on STAG version 2.1
Developed by :     Mike Lee,
With help from :   Bernard Tenbroek, Bill Redman-White, Mike Uren, Chris Edwards
                   and John Bunyan.
Acknowledgements : Rupert Howes and Pete Mole.
**********/

/* Modified: 2001 Paolo Nenzi */

#include "ngspice.h"
#include <stdio.h>
#include "cktdefs.h"
#include "soi3defs.h"
#include "sperror.h"
#include "suffix.h"


int
SOI3trunc(inModel,ckt,timeStep)
    GENmodel *inModel;
    register CKTcircuit *ckt;
    double *timeStep;
{
    register SOI3model *model = (SOI3model *)inModel;
    register SOI3instance *here;

    for( ; model != NULL; model = model->SOI3nextModel)
    {
        for(here=model->SOI3instances;here!=NULL;here = here->SOI3nextInstance)
        {
            CKTterr(here->SOI3qgf,ckt,timeStep);
            CKTterr(here->SOI3qd,ckt,timeStep);
            CKTterr(here->SOI3qs,ckt,timeStep);
        }
    }
    return(OK);
}
