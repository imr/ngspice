/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Gordon Jacobs
Modified: 2001 Jon Engelbert
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ngspice/fteext.h"
#include "cswdefs.h"
#include "ngspice/trandefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

static void
verify(int state, char *msg)
{
    switch (state) {
    case REALLY_ON:
    case REALLY_OFF:
    case HYST_ON:
    case HYST_OFF:
        break;
    default:
        internalerror(msg);
    }
}


int
CSWload(GENmodel *inModel, CKTcircuit *ckt)
{
    CSWmodel *model = (CSWmodel *) inModel;
    CSWinstance *here;

    for (; model; model = CSWnextModel(model))
        for (here = CSWinstances(model); here; here = CSWnextInstance(here)) {

            int old_current_state = (int) ckt->CKTstate0[here->CSWswitchstate];
            int previous_state = (int) ckt->CKTstate1[here->CSWswitchstate];
            int current_state;

            double g_now;
            double i_ctrl = ckt->CKTrhsOld[here->CSWcontBranch];

            /* decide the state of the switch */

            if (ckt->CKTmode & (MODEINITFIX | MODEINITJCT)) {

                if (here->CSWzero_state) {
                    /* switch specified "on" */
                    if (i_ctrl > model->CSWiThreshold + fabs(model->CSWiHysteresis))
                        current_state = REALLY_ON;
                    else
                        current_state = HYST_ON;
                } else {
                    if (i_ctrl < model->CSWiThreshold - fabs(model->CSWiHysteresis))
                        current_state = REALLY_OFF;
                    else
                        current_state = HYST_OFF;
                }

            } else if (ckt->CKTmode & (MODEINITSMSIG)) {

                current_state = previous_state;

            } else if (ckt->CKTmode & (MODEINITFLOAT)) {

                /* fixme, missleading comment: */
                /* use state0 since INITTRAN or INITPRED already called */
                if (i_ctrl > (model->CSWiThreshold + fabs(model->CSWiHysteresis)))
                    current_state = REALLY_ON;
                else if (i_ctrl < (model->CSWiThreshold - fabs(model->CSWiHysteresis)))
                    current_state = REALLY_OFF;
                else
                    if (model->CSWiHysteresis > 0) {
                        current_state = previous_state;
                    } else {
                        /* in hysteresis... change value if going from low to hysteresis,
                         * or from hi to hysteresis. */

                        verify(previous_state, "bad value for previous_state in swload");

                        /* if previous state was in hysteresis, then don't change the state.. */
                        if (previous_state == REALLY_ON)
                            current_state = HYST_ON;
                        else if (previous_state == REALLY_OFF)
                            current_state = HYST_OFF;
                        else
                            current_state = previous_state;
                    }

                if ((current_state > 0) != (old_current_state > 0)) {
                    ckt->CKTnoncon++;    /* ensure one more iteration */
                    ckt->CKTtroubleElt = (GENinstance *) here;
                }

            } else if (ckt->CKTmode & (MODEINITTRAN | MODEINITPRED)) {

                if (i_ctrl > (model->CSWiThreshold + fabs(model->CSWiHysteresis)))
                    current_state = REALLY_ON;
                else if (i_ctrl < (model->CSWiThreshold - fabs(model->CSWiHysteresis)))
                    current_state = REALLY_OFF;
                else
                    if (model->CSWiHysteresis > 0) {
                        current_state = previous_state;
                    } else {

                        verify(previous_state, "bad value for previous_state in cswload");

                        if (previous_state == REALLY_ON)
                            current_state = HYST_ON;
                        else if (previous_state == REALLY_OFF)
                            current_state = HYST_OFF;
                        else
                            current_state = previous_state;
                    }

            } else {
                internalerror("bad things in swload");
                controlled_exit(1);
            }

            ckt->CKTstate0[here->CSWswitchstate] = current_state;
            ckt->CKTstate0[here->CSWctrlvalue] = i_ctrl;

            if (current_state > 0)
                g_now = model->CSWonConduct;
            else
                g_now = model->CSWoffConduct;

            here->CSWcond = g_now;

            *(here->CSWposPosPtr) += g_now;
            *(here->CSWposNegPtr) -= g_now;
            *(here->CSWnegPosPtr) -= g_now;
            *(here->CSWnegNegPtr) += g_now;
        }

    return OK;
}
