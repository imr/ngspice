/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Gordon Jacobs
Modified: 2000 AlansFixes
**********/

#include "ngspice.h"
#include <stdio.h>
#include "cktdefs.h"
#include "cswdefs.h"
#include "trandefs.h"
#include "sperror.h"
#include "suffix.h"

int
CSWload(inModel,ckt)
    GENmodel *inModel;
    CKTcircuit *ckt;

        /* actually load the current values into the 
         * sparse matrix previously provided 
         */
{
    CSWmodel *model = (CSWmodel*)inModel;
    CSWinstance *here;
    double g_now;
    double i_ctrl;
    double previous_state; 
    double current_state;

    /*  loop through all the switch models */
    for( ; model != NULL; model = model->CSWnextModel ) {

        /* loop through all the instances of the model */
        for (here = model->CSWinstances; here != NULL ;
                here=here->CSWnextInstance) {
	    if (here->CSWowner != ARCHme) continue;

            /* decide the state of the switch */

            if(ckt->CKTmode & (MODEINITFIX|MODEINITJCT)) {

                if(here->CSWzero_stateGiven) {
                        /* switch specified "on" */
                    *(ckt->CKTstate0 + here->CSWstate) = 1.0;
                    current_state = 1.0;
                } else {
                    *(ckt->CKTstate0 + here->CSWstate) = 0.0;
                    current_state = 0.0;
                }
                
                *(ckt->CKTstate0 + (here->CSWstate+1)) = 0;
                
            } else if (ckt->CKTmode & (MODEINITSMSIG)) {

                previous_state = *(ckt->CKTstate0 + here->CSWstate);
                current_state = previous_state;

            } else if (ckt->CKTmode & (MODEINITFLOAT)) {
                /* use state0 since INITTRAN or INITPRED already called */
                previous_state = *(ckt->CKTstate0 + here->CSWstate);
                i_ctrl = *(ckt->CKTrhsOld + 
                        here->CSWcontBranch);
                if(i_ctrl > (model->CSWiThreshold+model->CSWiHysteresis)) {
                    *(ckt->CKTstate0 + here->CSWstate) = 1.0;
                    current_state = 1.0;
                }
                else if(i_ctrl < (model->CSWiThreshold - 
                        model->CSWiHysteresis)) {
                    *(ckt->CKTstate0 + here->CSWstate) = 0.0;
                    current_state = 0.0;
                }
                else {
                    current_state = previous_state;
                }

                *(ckt->CKTstate0 + (here->CSWstate+1)) = i_ctrl;

                if(current_state != previous_state) {
                    ckt->CKTnoncon++;    /* ensure one more iteration */
		    ckt->CKTtroubleElt = (GENinstance *) here;
                }

            } else if (ckt->CKTmode & (MODEINITTRAN|MODEINITPRED)) {

                previous_state = *(ckt->CKTstate1 + here->CSWstate);
                i_ctrl = *(ckt->CKTrhsOld + 
                        here->CSWcontBranch);

                if(i_ctrl > (model->CSWiThreshold+model->CSWiHysteresis)) {
                    current_state = 1;
                } else if(i_ctrl < (model->CSWiThreshold - 
                        model->CSWiHysteresis))  {
                    current_state = 0;
                } else {
                    current_state = previous_state;
                }

                if(current_state == 0) {
                    *(ckt->CKTstate0 + here->CSWstate) = 0.0;
                } else {
                    *(ckt->CKTstate0 + here->CSWstate) = 1.0;
                }

                *(ckt->CKTstate0 + (here->CSWstate+1)) = i_ctrl;

            }

            g_now = current_state?(model->CSWonConduct):(model->CSWoffConduct);
            here->CSWcond = g_now;

            *(here->CSWposPosptr) += g_now;
            *(here->CSWposNegptr) -= g_now;
            *(here->CSWnegPosptr) -= g_now;
            *(here->CSWnegNegptr) += g_now;
        }
    }
    return(OK);
}
