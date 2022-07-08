/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: September 2003 Paolo Nenzi
**********/
/*
 */

/* load the capacitor structure with those pointers needed later
 * for fast matrix loading
 */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "capdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


/*ARGSUSED*/
int
CAPtemp(GENmodel *inModel, CKTcircuit *ckt)

{
    CAPmodel *model = (CAPmodel*)inModel;
    CAPinstance *here;
    double difference;
    double factor;
    double tc1, tc2;

    /*  loop through all the capacitor models */
    for( ; model != NULL; model = CAPnextModel(model)) {

        /* loop through all the instances of the model */
        for (here = CAPinstances(model); here != NULL ;
                here=CAPnextInstance(here)) {

            /* Default Value Processing for Capacitor Instance */
            if(!here->CAPtempGiven) {
                here->CAPtemp   = ckt->CKTtemp;
                if(!here->CAPdtempGiven)   here->CAPdtemp  = 0.0;
            } else { /* CAPtempGiven */
                here->CAPdtemp = 0.0;
                if (here->CAPdtempGiven)
                    printf("%s: Instance temperature specified, dtemp ignored\n",
                           here->CAPname);
            }

            if (!here->CAPwidthGiven) {
                here->CAPwidth = model->CAPdefWidth;
            }
            if (!here->CAPscaleGiven) here->CAPscale = 1.0;
            if (!here->CAPmGiven)     here->CAPm     = 1.0;

            if (!here->CAPcapGiven) { /* No instance capacitance given */
                if (!model->CAPmCapGiven) { /* No model capacitange given */
                    here->CAPcapac =
                        model->CAPcj *
                        (here->CAPwidth - model->CAPnarrow) *
                        (here->CAPlength - model->CAPshort) +
                        model->CAPcjsw * 2 * (
                            (here->CAPlength - model->CAPshort) +
                            (here->CAPwidth - model->CAPnarrow));
                }
                else {
                    here->CAPcapac = model->CAPmCap;
                }
            }
            else
                here->CAPcapac = here->CAPcapacinst; /* reset capacitance to instance value */

            difference = (here->CAPtemp + here->CAPdtemp) - model->CAPtnom;

            /* instance parameters tc1 and tc2 will override
               model parameters tc1 and tc2 */
            if (here->CAPtc1Given)
                tc1 = here->CAPtc1; /* instance */
            else
                tc1 = model->CAPtempCoeff1; /* model */

            if (here->CAPtc2Given)
                tc2 = here->CAPtc2;
            else
                tc2 = model->CAPtempCoeff2;

            factor = 1.0 + tc1*difference +
                     tc2*difference*difference;

            here->CAPcapac = here->CAPcapac * factor * here->CAPscale;

        }
    }
    return(OK);
}

