/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Gordon Jacobs
Modified: 2001 Jon Engelbert
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "cswdefs.h"
#include "ngspice/fteext.h"
#include "ngspice/trandefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
CSWload(GENmodel *inModel, CKTcircuit *ckt)
{
    CSWmodel *model = (CSWmodel *) inModel;
    CSWinstance *here;
    double g_now;
    double i_ctrl;
    double previous_state = -1;
    double current_state = -1, old_current_state = -1;
    double REALLY_OFF = 0, REALLY_ON = 1;
    /* switch is on or off, not in hysteresis region. */
    double HYST_OFF = 2, HYST_ON = 3;
    /* switch is on or off while control value is in hysteresis region. */

    for (; model; model = CSWnextModel(model))
        for (here = CSWinstances(model); here; here = CSWnextInstance(here)) {

            old_current_state = ckt->CKTstate0[here->CSWswitchstate];
            previous_state = ckt->CKTstate1[here->CSWswitchstate];
            i_ctrl = ckt->CKTrhsOld[here->CSWcontBranch];

            /* decide the state of the switch */

            if (ckt->CKTmode & (MODEINITFIX | MODEINITJCT)) {

                if (here->CSWzero_stateGiven) {
                    /* switch specified "on" */
                    if (model->CSWiHysteresis >= 0 && i_ctrl > model->CSWiThreshold + model->CSWiHysteresis)
                        current_state = REALLY_ON;
                    else if (model->CSWiHysteresis < 0 && i_ctrl > model->CSWiThreshold - model->CSWiHysteresis)
                        current_state = REALLY_ON;
                    else
                        current_state = HYST_ON;
                } else {
                    if (model->CSWiHysteresis >= 0 && i_ctrl < model->CSWiThreshold - model->CSWiHysteresis)
                        current_state = REALLY_OFF;
                    else if (model->CSWiHysteresis < 0 && i_ctrl < model->CSWiThreshold + model->CSWiHysteresis)
                        current_state = REALLY_OFF;
                    else
                        current_state = HYST_OFF;
                }

            } else if (ckt->CKTmode & (MODEINITSMSIG)) {

                current_state = previous_state;

            } else if (ckt->CKTmode & (MODEINITFLOAT)) {
                /* use state0 since INITTRAN or INITPRED already called */

                if (model->CSWiHysteresis > 0) {
                    if (i_ctrl > (model->CSWiThreshold + model->CSWiHysteresis))
                        current_state = REALLY_ON;
                    else if (i_ctrl < (model->CSWiThreshold - model->CSWiHysteresis))
                        current_state = REALLY_OFF;
                    else
                        current_state = previous_state;
                } else {
                    if (i_ctrl > (model->CSWiThreshold - model->CSWiHysteresis))
                        current_state = REALLY_ON;
                    else if (i_ctrl < (model->CSWiThreshold + model->CSWiHysteresis))
                        current_state = REALLY_OFF;
                    else {
                        /* in hysteresis... change value if going from low to hysteresis,
                         * or from hi to hysteresis. */

                        /* if previous state was in hysteresis, then don't change the state.. */
                        if (previous_state == HYST_OFF || previous_state == HYST_ON)
                            current_state = previous_state;
                        else if (previous_state == REALLY_ON)
                            current_state = HYST_OFF;
                        else if (previous_state == REALLY_OFF)
                            current_state = HYST_ON;
                        else
                            internalerror("bad value for previous region in swload");
                    }
                }

                if (current_state != old_current_state) {
                    ckt->CKTnoncon++;    /* ensure one more iteration */
                    ckt->CKTtroubleElt = (GENinstance *) here;
                }

            } else if (ckt->CKTmode & (MODEINITTRAN | MODEINITPRED)) {

                if (model->CSWiHysteresis > 0) {
                    if (i_ctrl > (model->CSWiThreshold + model->CSWiHysteresis))
                        current_state = REALLY_ON;
                    else if (i_ctrl < (model->CSWiThreshold - model->CSWiHysteresis))
                        current_state = REALLY_OFF;
                    else
                        current_state = previous_state;
                } else {
                    if (i_ctrl > (model->CSWiThreshold - model->CSWiHysteresis))
                        current_state = REALLY_ON;
                    else if (i_ctrl < (model->CSWiThreshold + model->CSWiHysteresis))
                        current_state = REALLY_OFF;
                    else {
                        /* in hysteresis... change value if going from low to hysteresis,
                         * or from hi to hysteresis. */

                        /* if previous state was in hysteresis, then don't change the state.. */
                        if (previous_state == HYST_OFF || previous_state == HYST_ON)
                            current_state = previous_state;
                        else if (previous_state == REALLY_ON)
                            current_state = HYST_OFF;
                        else if (previous_state == REALLY_OFF)
                            current_state = HYST_ON;
                        else
                            internalerror("bad value for previous region in cswload");
                    }
                }
            }

            ckt->CKTstate0[here->CSWswitchstate] = current_state;
            ckt->CKTstate1[here->CSWswitchstate] = previous_state;

//            ckt->CKTstate0[here->CSWctrlvalue] = i_ctrl;
            /* FIXME: without this statement truncation is not used, as with SW.
               But test are needed to check if improvments are possible. */

            if (current_state == REALLY_ON || current_state == HYST_ON)
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
