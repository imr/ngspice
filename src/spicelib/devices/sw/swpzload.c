/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Gordon Jacobs
**********/
/*
 */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "swdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/complex.h"
#include "ngspice/suffix.h"


/* ARGSUSED */
int
SWpzLoad(GENmodel *inModel, CKTcircuit *ckt, SPcomplex *s)
        /* load the current values into the 
         * sparse matrix previously provided 
         * during AC analysis.
         */
{
    SWmodel *model = (SWmodel *)inModel;
    SWinstance *here;
    double g_now;

    NG_IGNORE(s);

    /*  loop through all the switch models */
    for( ; model != NULL; model = SWnextModel(model)) {

        /* loop through all the instances of the model */
        for (here = SWinstances(model); here != NULL ;
                here=SWnextInstance(here)) {

            /* In AC analysis, just propogate the state... */

            if (ckt->CKTstate0[here->SWswitchstate] > 0)
                g_now = model->SWonConduct;
            else
                g_now = model->SWoffConduct;

            *(here->SWposPosPtr) += g_now;
            *(here->SWposNegPtr) -= g_now;
            *(here->SWnegPosPtr) -= g_now;
            *(here->SWnegNegPtr) += g_now;
        }
    }
    return(OK);
}
