/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#include "ngspice.h"
#include "cktdefs.h"
#include "isrcdefs.h"
#include "sperror.h"
#include "suffix.h"

int
ISRCacLoad(GENmodel *inModel, CKTcircuit *ckt)
{
    ISRCmodel *model = (ISRCmodel*)inModel;
    ISRCinstance *here;

    for( ; model != NULL; model = model->ISRCnextModel ) {

        /* loop through all the instances of the model */
        for (here = model->ISRCinstances; here != NULL ;
                here=here->ISRCnextInstance) {
	    if (here->ISRCowner != ARCHme) continue;

            *(ckt->CKTrhs + (here->ISRCposNode)) += 
                here->ISRCacReal;
            *(ckt->CKTrhs + (here->ISRCnegNode)) -= 
                here->ISRCacReal;
            *(ckt->CKTirhs + (here->ISRCposNode)) += 
                here->ISRCacImag;
            *(ckt->CKTirhs + (here->ISRCnegNode)) -= 
                here->ISRCacImag;
        }
    }
    return(OK);

}
