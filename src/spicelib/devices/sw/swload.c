/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Gordon Jacobs
Modified: 2001 Jon Engelbert
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ngspice/fteext.h"
#include "swdefs.h"
#include "ngspice/trandefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
SWload(GENmodel *inModel, CKTcircuit *ckt)
{
    SWmodel *model = (SWmodel *) inModel;
    SWinstance *here;
    double g_now;
    double v_ctrl;
    int previous_state;
    int current_state;
    int old_current_state;

    for (; model; model = SWnextModel(model)) {
        for (here = SWinstances(model); here; here=SWnextInstance(here)) {

            old_current_state = (int) ckt->CKTstate0[here->SWswitchstate];
            previous_state = (int) ckt->CKTstate1[here->SWswitchstate];

            v_ctrl =
                ckt->CKTrhsOld [here->SWposCntrlNode] -
                ckt->CKTrhsOld [here->SWnegCntrlNode];

            /* decide the state of the switch */

            if (ckt->CKTmode & (MODEINITFIX | MODEINITJCT)) {

                if (here->SWzero_stateGiven) {
                    /* switch specified "on" */
                    if ((model->SWvHysteresis >= 0) && (v_ctrl > (model->SWvThreshold + model->SWvHysteresis)))
                        current_state = REALLY_ON;
                    else if ((model->SWvHysteresis < 0) && (v_ctrl > (model->SWvThreshold - model->SWvHysteresis)))
                        current_state = REALLY_ON;
                    else
                        current_state = HYST_ON;
                } else {
                    if ((model->SWvHysteresis >= 0) && (v_ctrl < (model->SWvThreshold - model->SWvHysteresis)))
                        current_state = REALLY_OFF;
                    else if ((model->SWvHysteresis < 0) && (v_ctrl < (model->SWvThreshold + model->SWvHysteresis)))
                        current_state = REALLY_OFF;
                    else
                        current_state = HYST_OFF;
                }

            } else if (ckt->CKTmode & (MODEINITSMSIG)) {

                current_state = previous_state;

            } else if (ckt->CKTmode & (MODEINITFLOAT)) {

                /* use state0 since INITTRAN or INITPRED already called */
                if (model->SWvHysteresis > 0) {
                    if (v_ctrl > (model->SWvThreshold + model->SWvHysteresis)) {
                        current_state = REALLY_ON;
                    } else if (v_ctrl < (model->SWvThreshold -  model->SWvHysteresis)) {
                        current_state = REALLY_OFF;
                    } else {
                        current_state = old_current_state;
                    }
                } else {        // negative hysteresis case.
                    if (v_ctrl > (model->SWvThreshold - model->SWvHysteresis)) {
                        current_state = REALLY_ON;
                    } else if (v_ctrl < (model->SWvThreshold +  model->SWvHysteresis)) {
                        current_state = REALLY_OFF;
                    } else {        // in hysteresis... change value if going from low to hysteresis, or from hi to hysteresis.
                        // if previous state was in hysteresis, then don't change the state..
                        if ((previous_state == HYST_OFF) || (previous_state == HYST_ON)) {
                            current_state = previous_state;
                        } else if (previous_state == REALLY_ON) {
                            current_state = HYST_OFF;
                        } else if (previous_state == REALLY_OFF) {
                            current_state = HYST_ON;
                        } else {
                            internalerror("bad value for previous state in swload");
                            controlled_exit(1);
                        }
                    }
                }

                if ((current_state > 0) != (old_current_state > 0)) {
                    ckt->CKTnoncon++;       /* ensure one more iteration */
                    ckt->CKTtroubleElt = (GENinstance *) here;
                }

            } else if (ckt->CKTmode & (MODEINITTRAN | MODEINITPRED)) {

                if (model->SWvHysteresis > 0) {
                    if (v_ctrl > (model->SWvThreshold + model->SWvHysteresis))
                        current_state = REALLY_ON;
                    else if (v_ctrl < (model->SWvThreshold - model->SWvHysteresis))
                        current_state = REALLY_OFF;
                    else
                        current_state = previous_state;
                } else {        // negative hysteresis case.
                    if (v_ctrl > (model->SWvThreshold - model->SWvHysteresis))
                        current_state = REALLY_ON;
                    else if (v_ctrl < (model->SWvThreshold + model->SWvHysteresis))
                        current_state = REALLY_OFF;
                    else {
                        if ((previous_state == HYST_ON) || (previous_state == HYST_OFF))
                            current_state = previous_state;
                        else if (previous_state == REALLY_ON)
                            current_state = REALLY_OFF;
                        else if (previous_state == REALLY_OFF)
                            current_state = REALLY_ON;
                        else {
                            internalerror("bad value for state in swload");
                            controlled_exit(1);
                        }
                    }
                }

            } else {
                internalerror("bad things in swload");
                controlled_exit(1);
            }

// code added to force the state to be updated.
// there is a possible problem.  What if, during the transient analysis, the time is stepped
// forward enough to change the switch's state, but that time point is rejected as being too
// distant and then the time is pushed back to a time before the switch changed states.
// After analyzing the transient code, it seems that this is not a problem because state updating
// occurs before the convergence loop in transient processing.

            ckt->CKTstate0[here->SWswitchstate] = current_state;
            ckt->CKTstate0[here->SWctrlvalue] = v_ctrl;

            if (current_state > 0)
                g_now = model->SWonConduct;
            else
                g_now = model->SWoffConduct;

            here->SWcond = g_now;

            *(here->SWposPosPtr) += g_now;
            *(here->SWposNegPtr) -= g_now;
            *(here->SWnegPosPtr) -= g_now;
            *(here->SWnegNegPtr) += g_now;
        }
    }

    return OK;
}
