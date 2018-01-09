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

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "soi3defs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
SOI3trunc(GENmodel *inModel, CKTcircuit *ckt, double *timeStep)
{
     SOI3model *model = (SOI3model *)inModel;
     SOI3instance *here;

    for( ; model != NULL; model = SOI3nextModel(model))
    {
        for(here=SOI3instances(model);here!=NULL;here = SOI3nextInstance(here))
        {
            CKTterr(here->SOI3qgf,ckt,timeStep);
            CKTterr(here->SOI3qd,ckt,timeStep);
            CKTterr(here->SOI3qs,ckt,timeStep);
        }
    }
    return(OK);
}
