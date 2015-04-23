/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: Apr 2000 - Paolo Nenzi
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "resdefs.h"
#include "ngspice/sperror.h"


/* actually load the current resistance value into the sparse matrix
 * previously provided */
int
RESload(GENmodel *inModel, CKTcircuit *ckt)
{
    RESmodel *model = (RESmodel *)inModel;
    double m;

    /*  loop through all the resistor models */
    for( ; model != NULL; model = model->RESnextModel ) {
        RESinstance *here;

        /* loop through all the instances of the model */
        for (here = model->RESinstances; here != NULL ;
                here = here->RESnextInstance) {

            here->REScurrent = (*(ckt->CKTrhsOld+here->RESposNode) -
                                *(ckt->CKTrhsOld+here->RESnegNode)) * here->RESconduct;

            m = (here->RESm);

            *(here->RES_posPrime_pos) += m * here->RESconduct;
            *(here->RES_neg_neg)      += m * here->RESconduct;
            *(here->RES_posPrime_neg) -= m * here->RESconduct;
            *(here->RES_neg_pos)      -= m * here->RESconduct;

            if (here->RESbranch) {
                *(here->RES_pos_ibr)      += 1.0;
                *(here->RES_posPrime_ibr) -= 1.0;
            }
        }
    }
    return(OK);
}


/* actually load the current resistance value into the sparse matrix
 * previously provided */
int
RESacload(GENmodel *inModel, CKTcircuit *ckt)
{
    RESmodel *model = (RESmodel *)inModel;
    double g;

    NG_IGNORE(ckt);

    /*  loop through all the resistor models */
    for( ; model != NULL; model = model->RESnextModel ) {
        RESinstance *here;

        /* loop through all the instances of the model */
        for (here = model->RESinstances; here != NULL ;
             here = here->RESnextInstance) {

            if (here->RESacresGiven)
                g = here->RESm * here->RESacConduct;
            else
                g = here->RESm * here->RESconduct;

            *(here->RES_posPrime_pos) += g;
            *(here->RES_neg_neg)      += g;
            *(here->RES_posPrime_neg) -= g;
            *(here->RES_neg_pos)      -= g;

            if (here->RESbranch) {
                *(here->RES_pos_ibr)      += 1.0;
                *(here->RES_posPrime_ibr) -= 1.0;
            }
        }
    }
    return(OK);
}
