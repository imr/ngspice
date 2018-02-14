/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "tradefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

int
TRAacLoad(GENmodel *inModel, CKTcircuit *ckt)
        /* actually load the current values into the 
         * sparse matrix previously provided 
         */
{
    TRAmodel *model = (TRAmodel *)inModel;
    TRAinstance *here;
    double real;
    double imag;

    /*  loop through all the transmission line models */
    for( ; model != NULL; model = TRAnextModel(model)) {

        /* loop through all the instances of the model */
        for (here = TRAinstances(model); here != NULL ;
                here=TRAnextInstance(here)) {

            real = cos(-ckt->CKTomega*here->TRAtd);
            imag = sin(-ckt->CKTomega*here->TRAtd);
            
            *(here->TRApos1Pos1Ptr) += here->TRAconduct;
            *(here->TRApos1Int1Ptr) -= here->TRAconduct;
            *(here->TRAneg1Ibr1Ptr) -= 1;
            *(here->TRApos2Pos2Ptr) += here->TRAconduct;
            *(here->TRAneg2Ibr2Ptr) -= 1;
            *(here->TRAint1Pos1Ptr) -= here->TRAconduct;
            *(here->TRAint1Int1Ptr) += here->TRAconduct;
            *(here->TRAint1Ibr1Ptr) += 1;
            *(here->TRAint2Int2Ptr) += here->TRAconduct;
            *(here->TRAint2Ibr2Ptr) += 1;
            *(here->TRAibr1Neg1Ptr) -= 1;
            *(here->TRAibr1Pos2Ptr+0) -= real;
            *(here->TRAibr1Pos2Ptr+1) -= imag;
            *(here->TRAibr1Neg2Ptr+0) += real;
            *(here->TRAibr1Neg2Ptr+1) += imag;
            *(here->TRAibr1Int1Ptr) += 1;
            *(here->TRAibr1Ibr2Ptr+0) -= real * here->TRAimped;
            *(here->TRAibr1Ibr2Ptr+1) -= imag * here->TRAimped;
            *(here->TRAibr2Pos1Ptr+0) -= real;
            *(here->TRAibr2Pos1Ptr+1) -= imag;
            *(here->TRAibr2Neg1Ptr+0) += real;
            *(here->TRAibr2Neg1Ptr+1) += imag;
            *(here->TRAibr2Neg2Ptr) -= 1;
            *(here->TRAibr2Int2Ptr) += 1;
            *(here->TRAibr2Ibr1Ptr+0) -= real * here->TRAimped;
            *(here->TRAibr2Ibr1Ptr+1) -= imag * here->TRAimped;
            *(here->TRApos2Int2Ptr) -= here->TRAconduct;
            *(here->TRAint2Pos2Ptr) -= here->TRAconduct;

        }
    }
    return(OK);
}
