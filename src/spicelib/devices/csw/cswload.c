/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Gordon Jacobs
Modified: 2001 Jon Engelbert
**********/

#include "ngspice.h"
#include "cktdefs.h"
#include "cswdefs.h"
#include "fteext.h"
#include "trandefs.h"
#include "sperror.h"
#include "suffix.h"

int
CSWload(GENmodel *inModel, CKTcircuit *ckt)

        /* actually load the current values into the 
         * sparse matrix previously provided 
         */
{
    CSWmodel *model = (CSWmodel*)inModel;
    CSWinstance *here;
    double g_now;
    double i_ctrl;
    double previous_state = -1; 
    double current_state = -1, old_current_state = -1;
	double REALLY_OFF = 0, REALLY_ON = 1;	// switch is on or off, not in hysteresis region.
	double HYST_OFF = 2, HYST_ON = 3;	// switch is on or off while control value is in hysteresis region.

    /*  loop through all the switch models */
    for( ; model != NULL; model = model->CSWnextModel ) {

        /* loop through all the instances of the model */
        for (here = model->CSWinstances; here != NULL ;
                here=here->CSWnextInstance) {
                	
        if (here->CSWowner != ARCHme) continue;
			
			old_current_state = *(ckt->CKTstates[0] + here->CSWstate);
			previous_state = *(ckt->CKTstates[1] + here->CSWstate);
            i_ctrl = *(ckt->CKTrhsOld + 
                    here->CSWcontBranch);

            /* decide the state of the switch */

            if(ckt->CKTmode & (MODEINITFIX|MODEINITJCT)) {

                if(here->CSWzero_stateGiven) {
                        /* switch specified "on" */
					if ((model->CSWiHysteresis >= 0) && (i_ctrl > (model->CSWiThreshold + model->CSWiHysteresis))) 
						current_state = REALLY_ON;
					else if ((model->CSWiHysteresis < 0) && (i_ctrl > (model->CSWiThreshold - model->CSWiHysteresis))) 
						current_state = REALLY_ON;
					else 
						current_state = HYST_ON;
                } else {
					if ((model->CSWiHysteresis >= 0) && (i_ctrl < (model->CSWiThreshold - model->CSWiHysteresis))) 
						current_state = REALLY_OFF;
					else if ((model->CSWiHysteresis < 0) && (i_ctrl < (model->CSWiThreshold + model->CSWiHysteresis))) 
						current_state = REALLY_OFF;
					else 
						current_state = HYST_OFF;
                }

            } else if (ckt->CKTmode & (MODEINITSMSIG)) {

                current_state = previous_state;

            } else if (ckt->CKTmode & (MODEINITFLOAT)) {
                /* use state0 since INITTRAN or INITPRED already called */

				if (model->CSWiHysteresis > 0) {
					if (i_ctrl > (model->CSWiThreshold + model->CSWiHysteresis)) {
						current_state = REALLY_ON;
					} else if (i_ctrl < (model->CSWiThreshold -  model->CSWiHysteresis)) {
						current_state = REALLY_OFF;
					} else {
						current_state = previous_state;
					}
				} else {
					if (i_ctrl > (model->CSWiThreshold - model->CSWiHysteresis)) 					{
						current_state = REALLY_ON;
					} else if (i_ctrl < (model->CSWiThreshold +  model->CSWiHysteresis)) {
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
							internalerror("bad value for previous region in swload");
					}
				}

                if(current_state != old_current_state) {
						ckt->CKTnoncon++;    /* ensure one more iteration */
						ckt->CKTtroubleElt = (GENinstance *) here;
                }

            } else if (ckt->CKTmode & (MODEINITTRAN|MODEINITPRED)) {

				if (model->CSWiHysteresis > 0) {
					if (i_ctrl > (model->CSWiThreshold + model->CSWiHysteresis)) {
						current_state = REALLY_ON;
					} else if (i_ctrl < (model->CSWiThreshold -  model->CSWiHysteresis)) {
						current_state = REALLY_OFF;
					} else {
						current_state = previous_state;
					}
				} else {
					if (i_ctrl > (model->CSWiThreshold - model->CSWiHysteresis)) 					{
						current_state = REALLY_ON;
					} else if (i_ctrl < (model->CSWiThreshold +  model->CSWiHysteresis)) {
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
							internalerror("bad value for previous region in cswload");
					}
				}
             }

			*(ckt->CKTstates[0] + here->CSWstate) = current_state;
			*(ckt->CKTstates[1] + here->CSWstate) = previous_state;
            if ((current_state == REALLY_ON) || (current_state == HYST_ON)) 
				g_now = model->CSWonConduct;
			else 
				g_now = model->CSWoffConduct;
            here->CSWcond = g_now;

            *(here->CSWposPosptr) += g_now;
            *(here->CSWposNegptr) -= g_now;
            *(here->CSWnegPosptr) -= g_now;
            *(here->CSWnegNegptr) += g_now;
        }
    }
    return(OK);
}
