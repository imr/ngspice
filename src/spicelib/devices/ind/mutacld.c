/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

#include "ngspice.h"
#include "cktdefs.h"
#include "inddefs.h"
#include "sperror.h"
#include "suffix.h"


#ifdef MUTUAL
int
MUTacLoad(GENmodel *inModel, CKTcircuit *ckt)
{
    MUTmodel *model = (MUTmodel*)inModel;
    double val;
    MUTinstance *here;

    for( ; model != NULL; model = model->MUTnextModel) {
        for( here = model->MUTinstances;here != NULL; 
                here = here->MUTnextInstance) {
	    if (here->MUTowner != ARCHme) continue;
    
            val = ckt->CKTomega * here->MUTfactor;
            *(here->MUTbr1br2 +1) -= val;
            *(here->MUTbr2br1 +1) -= val;
        }
    }
    return(OK);

}
#endif /* MUTUAL */
