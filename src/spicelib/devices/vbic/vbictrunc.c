/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Model Author: 1995 Colin McAndrew Motorola
Spice3 Implementation: 2003 Dietmar Warning DAnalyse GmbH
**********/

/*
 * This routine performs truncation error calculations for
 * VBICs in the circuit.
 */

#include "ngspice.h"
#include "cktdefs.h"
#include "vbicdefs.h"
#include "sperror.h"
#include "suffix.h"


int
VBICtrunc(GENmodel *inModel, CKTcircuit *ckt, double *timeStep)
{
    VBICmodel *model = (VBICmodel*)inModel;
    VBICinstance *here;

    for( ; model != NULL; model = model->VBICnextModel) {
        for(here=model->VBICinstances;here!=NULL;
            here = here->VBICnextInstance){
            if (here->VBICowner != ARCHme) continue;

            CKTterr(here->VBICqbe,ckt,timeStep);
            CKTterr(here->VBICqbex,ckt,timeStep);
            CKTterr(here->VBICqbc,ckt,timeStep);
            CKTterr(here->VBICqbcx,ckt,timeStep);
            CKTterr(here->VBICqbep,ckt,timeStep);
            CKTterr(here->VBICqbeo,ckt,timeStep);
            CKTterr(here->VBICqbco,ckt,timeStep);
            CKTterr(here->VBICqbcp,ckt,timeStep);
        }
    }
    return(OK);
}
