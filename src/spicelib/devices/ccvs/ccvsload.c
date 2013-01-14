/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ccvsdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


/*ARGSUSED*/
int
CCVSload(GENmodel *inModel, CKTcircuit *ckt)

        /* actually load the current voltage value into the 
         * sparse matrix previously provided 
         */
{
    CCVSmodel *model = (CCVSmodel*)inModel;
    CCVSinstance *here;

    NG_IGNORE(ckt);

    /*  loop through all the voltage source models */
    for( ; model != NULL; model = model->CCVSnextModel ) {

        /* loop through all the instances of the model */
        for (here = model->CCVSinstances; here != NULL ;
                here=here->CCVSnextInstance) {
            
            *(here->CCVSposIbrptr) += 1.0 ;
            *(here->CCVSnegIbrptr) -= 1.0 ;
            *(here->CCVSibrPosptr) += 1.0 ;
            *(here->CCVSibrNegptr) -= 1.0 ;
            *(here->CCVSibrContBrptr) -= here->CCVScoeff ;

            *(ckt->CKTfvk+here->CCVSposNode) += *(ckt->CKTrhsOld+here->CCVSbranch) ;
            *(ckt->CKTfvk+here->CCVSnegNode) -= *(ckt->CKTrhsOld+here->CCVSbranch) ;
        }
    }
    return(OK);
}
