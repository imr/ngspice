/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified by Dietmar Warning 2003
**********/
/*
 */

#include "ngspice.h"
#include "cktdefs.h"
#include "diodefs.h"
#include "sperror.h"
#include "suffix.h"


int
DIOacLoad(GENmodel *inModel, CKTcircuit *ckt)
{
    DIOmodel *model = (DIOmodel*)inModel;
    double gspr;
    double geq;
    double xceq;
    DIOinstance *here;

    /*  loop through all the diode models */
    for( ; model != NULL; model = model->DIOnextModel ) {

        /* loop through all the instances of the model */
        for (here = model->DIOinstances; here != NULL ;
                here=here->DIOnextInstance) {
	    if (here->DIOowner != ARCHme) continue;
            gspr=here->DIOtConductance*here->DIOarea*here->DIOm;
            geq= *(ckt->CKTstate0 + here->DIOconduct);
            xceq= *(ckt->CKTstate0 + here->DIOcapCurrent) * ckt->CKTomega;
            *(here->DIOposPosPtr ) += gspr;
            *(here->DIOnegNegPtr ) += geq;
            *(here->DIOnegNegPtr +1 ) += xceq;
            *(here->DIOposPrimePosPrimePtr ) += geq+gspr;
            *(here->DIOposPrimePosPrimePtr +1 ) += xceq;
            *(here->DIOposPosPrimePtr ) -= gspr;
            *(here->DIOnegPosPrimePtr ) -= geq;
            *(here->DIOnegPosPrimePtr +1 ) -= xceq;
            *(here->DIOposPrimePosPtr ) -= gspr;
            *(here->DIOposPrimeNegPtr ) -= geq;
            *(here->DIOposPrimeNegPtr +1 ) -= xceq;
        }
    }
    return(OK);

}
