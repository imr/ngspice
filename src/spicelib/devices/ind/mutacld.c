/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "inddefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
MUTacLoad(GENmodel *inModel, CKTcircuit *ckt)
{
    MUTmodel *model = (MUTmodel*)inModel;
    double val;
    MUTinstance *here;

    for( ; model != NULL; model = MUTnextModel(model)) {
        for( here = MUTinstances(model);here != NULL; 
                here = MUTnextInstance(here)) {
    
            val = ckt->CKTomega * here->MUTfactor;
            *(here->MUTbr1br2Ptr +1) -= val;
            *(here->MUTbr2br1Ptr +1) -= val;
        }
    }
    return(OK);

}
