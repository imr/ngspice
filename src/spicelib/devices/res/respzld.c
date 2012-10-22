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
    double m;

    NG_IGNORE(s);
    NG_IGNORE(ckt);

    /*  loop through all the resistor models */
    for( ; model != NULL; model = model->RESnextModel ) {

        /* loop through all the instances of the model */
        for (here = model->RESinstances; here != NULL ;
                here=here->RESnextInstance) {

	    m = here->RESm;
	    
            *(here->RESposPosptr) += m * here->RESconduct;
            *(here->RESnegNegptr) += m * here->RESconduct;
            *(here->RESposNegptr) -= m * here->RESconduct;
            *(here->RESnegPosptr) -= m * here->RESconduct;
        }
    }
    return(OK);
}
