/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: Apr 2000 - Paolo Nenzi
**********/


#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ngspice/complex.h"
#include "ngspice/sperror.h"
#include "resdefs.h"



int
RESpzLoad(GENmodel *inModel, CKTcircuit *ckt, SPcomplex *s)
        /* actually load the current resistance value into the 
         * sparse matrix previously provided 
         */
{
    RESmodel *model = (RESmodel *)inModel;
    RESinstance *here;
    double g;

    NG_IGNORE(s);
    NG_IGNORE(ckt);

    /*  loop through all the resistor models */
    for( ; model != NULL; model = model->RESnextModel ) {

        /* loop through all the instances of the model */
        for (here = model->RESinstances; here != NULL ;
                here=here->RESnextInstance) {

            if (here->RESacresGiven)
                g = here->RESm * here->RESacConduct;
            else
                g = here->RESm * here->RESconduct;

            *(here->RES_posPrime_posPrime) += g;
            *(here->RES_neg_neg)           += g;
            *(here->RES_posPrime_neg)      -= g;
            *(here->RES_neg_PosPrime)      -= g;

            if (here->RESbranch) {
                *(here->RES_pos_ibr)      += 1.0;
                *(here->RES_posPrime_ibr) -= 1.0;
                *(here->RES_ibr_pos)      += 1.0;
                *(here->RES_ibr_posPrime) -= 1.0;
            }
        }
    }
    return(OK);
}
