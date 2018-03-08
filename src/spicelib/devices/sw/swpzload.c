/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Gordon Jacobs
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "swdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/complex.h"
#include "ngspice/suffix.h"


int
SWpzLoad(GENmodel *inModel, CKTcircuit *ckt, SPcomplex *s)
{
    SWmodel *model = (SWmodel *) inModel;
    SWinstance *here;
    double g_now;
    int current_state;

    NG_IGNORE(s);

    for (; model; model = SWnextModel(model))
        for (here = SWinstances(model); here; here = SWnextInstance(here)) {

            /* In AC analysis, just propogate the state... */

            current_state = (int) ckt->CKTstate0[here->SWswitchstate];

            g_now = current_state ? model->SWonConduct : model->SWoffConduct;

            *(here->SWposPosPtr) += g_now;
            *(here->SWposNegPtr) -= g_now;
            *(here->SWnegPosPtr) -= g_now;
            *(here->SWnegNegPtr) += g_now;
        }

    return OK;
}
