/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Gordon Jacobs
Modified: 2001 Jon Engelbert
**********/

#include "ngspice.h"
#include "cktdefs.h"
#include "fteext.h"
#include "swdefs.h"
#include "trandefs.h"
#include "sperror.h"
#include "suffix.h"

int
SWload(GENmodel *inModel, CKTcircuit *ckt)
        /* actually load the current values into the 
         * sparse matrix previously provided 
         */
{
    SWmodel *model = (SWmodel *) inModel;
    SWinstance *here;
    double g_now;
    double v_ctrl;
    double previous_state = -1; 
    double current_state = -1;
	double old_current_state = -1;
	double REALLY_OFF = 0, REALLY_ON = 1;	// switch is on or off, not in hysteresis region.
	double HYST_OFF = 2, HYST_ON = 3;	// switch is on or off while control value is in hysteresis region.
//    double previous_region = -1; 
//    double current_region = -1;

    /*  loop through all the switch models */
    for( ; model != NULL; model = model->SWnextModel ) {

        /* loop through all the instances of the model */
        for (here = model->SWinstances; here != NULL ;
                here=here->SWnextInstance) {
	    if (here->SWowner != ARCHme) continue;
	     
			old_current_state = *(ckt->CKTstates[0] + here->SWstate);
			previous_state = *(ckt->CKTstates[1] + here->SWstate);

			v_ctrl = *(ckt->CKTrhsOld + here->SWposCntrlNode)
                         - *(ckt->CKTrhsOld + here->SWnegCntrlNode);
		  
		  /* decide the state of the switch */
		  
            if(ckt->CKTmode & (MODEINITFIX|MODEINITJCT)) {

                if(here->SWzero_stateGiven) {
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
				} else {	// negative hysteresis case.
					if (v_ctrl > (model->SWvThreshold - model->SWvHysteresis))
					{
						current_state = REALLY_ON;
					} else if (v_ctrl < (model->SWvThreshold +  model->SWvHysteresis))
					{
						current_state = REALLY_OFF;
					} else {	// in hysteresis... change value if going from low to hysteresis, or from hi to hysteresis.
						// if previous state was in hysteresis, then don't change the state..
						if ((previous_state == HYST_OFF) || (previous_state == HYST_ON)) {
							current_state = previous_state;
						} else if (previous_state == REALLY_ON) {
							current_state = HYST_OFF;
						} else if (previous_state == REALLY_OFF) {
							current_state = HYST_ON;
						} else
							internalerror("bad value for previous state in swload");
					}
				}

                if(current_state != old_current_state) {
                    ckt->CKTnoncon++;       /* ensure one more iteration */
					ckt->CKTtroubleElt = (GENinstance *) here;
                }


            } else if(ckt->CKTmode & (MODEINITTRAN|MODEINITPRED) ) {

				if (model->SWvHysteresis > 0) {
					if (v_ctrl > (model->SWvThreshold + model->SWvHysteresis))
						current_state = REALLY_ON;
					else if (v_ctrl < (model->SWvThreshold -  model->SWvHysteresis))
						current_state = REALLY_OFF;
					else 
						current_state = previous_state;
				} else {	// negative hysteresis case.
					if (v_ctrl > (model->SWvThreshold - model->SWvHysteresis))
						current_state = REALLY_ON;
					else if (v_ctrl < (model->SWvThreshold +  model->SWvHysteresis))
						current_state = REALLY_OFF;
					else {
						current_state = 0.0;
						if ((previous_state == HYST_ON) || (previous_state == HYST_OFF)) {
							current_state = previous_state;
						} else if (previous_state == REALLY_ON) {
							current_state = REALLY_OFF;
						} else if (previous_state == REALLY_OFF) {
							current_state = REALLY_ON;
						}
					}
				}
           }
// code added to force the state to be updated.
// there is a possible problem.  What if, during the transient analysis, the time is stepped
// forward enough to change the switch's state, but that time point is rejected as being too
// distant and then the time is pushed back to a time before the switch changed states.   
// After analyzing the transient code, it seems that this is not a problem because state updating
// occurs before the convergence loop in transient processing.
			*(ckt->CKTstates[0] + here->SWstate) = current_state;

            if ((current_state == REALLY_ON) || (current_state == HYST_ON)) 
				g_now = model->SWonConduct;
			else 
				g_now = model->SWoffConduct;
            here->SWcond = g_now;

            *(here->SWposPosptr) += g_now;
            *(here->SWposNegptr) -= g_now;
            *(here->SWnegPosptr) -= g_now;
            *(here->SWnegNegptr) += g_now;
        }
    }
    return(OK);
}
