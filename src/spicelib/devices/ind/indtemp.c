/**********
Copyright 2003 Paolo Nenzi
Author: 2003 Paolo Nenzi
**********/
/*
 */


#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "inddefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


/*ARGSUSED*/
int
INDtemp(GENmodel *inModel, CKTcircuit *ckt)
{
    INDmodel *model = (INDmodel*)inModel;
    INDinstance *here;
    double difference;
    double factor;
    double tc1, tc2;

    /*  loop through all the inductor models */
    for( ; model != NULL; model = INDnextModel(model)) {

        /* loop through all the instances of the model */
        for (here = INDinstances(model); here != NULL ;
                here=INDnextInstance(here)) {

            /* Default Value Processing for Inductor Instance */

            if(!here->INDtempGiven) {
                here->INDtemp = ckt->CKTtemp;
                if(!here->INDdtempGiven)   here->INDdtemp  = 0.0;
            } else { /* INDtempGiven */
                here->INDdtemp = 0.0;
                if (here->INDdtempGiven)
                    printf("%s: Instance temperature specified, dtemp ignored\n",
                           here->INDname);
            }

            if (!here->INDscaleGiven) here->INDscale = 1.0;
            if (!here->INDmGiven)     here->INDm     = 1.0;
            if (!here->INDntGiven)    here->INDnt    = 0.0;

            if (!here->INDindGiven) { /* No instance inductance given */
                if (here->INDntGiven)
                    here->INDinduct = model->INDspecInd * here->INDnt * here->INDnt;
                else
                    here->INDinduct = model->INDmInd;
            }
            else
                here->INDinduct = here->INDinductinst; /* reset inductance to instance value */

            difference = (here->INDtemp + here->INDdtemp) - model->INDtnom;

            /* instance parameters tc1 and tc2 will override
               model parameters tc1 and tc2 */
            if (here->INDtc1Given)
                tc1 = here->INDtc1; /* instance */
            else
                tc1 = model->INDtempCoeff1; /* model */

            if (here->INDtc2Given)
                tc2 = here->INDtc2;
            else
                tc2 = model->INDtempCoeff2;

            factor = 1.0 + tc1*difference + tc2*difference*difference;

            here->INDinduct = here->INDinduct * factor * here->INDscale;

        }
    }
    return(OK);
}

