/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Gordon Jacobs
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "swdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
SWacLoad(GENmodel *inModel, CKTcircuit *ckt)
{
    SWmodel *model = (SWmodel *) inModel;
    SWinstance *here;
    double g_now;
    int current_state;

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
