/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified Apr 2000 - Paolo Nenzi
Modified: 2000 AlanSfixes
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "resdefs.h"
#include "ngspice/sperror.h"

int
REStemp(GENmodel *inModel, CKTcircuit *ckt)
/* perform the temperature update to the resistors
 * calculate the conductance as a function of the
 * given nominal and current temperatures - the
 * resistance given in the struct is the nominal
 * temperature resistance
 */
{
    RESmodel *model =  (RESmodel *)inModel;
    RESinstance *here;


    /*  loop through all the resistor models */
    for( ; model != NULL; model = RESnextModel(model)) {

        /* loop through all the instances of the model */
        for (here = RESinstances(model); here != NULL ;
                here=RESnextInstance(here)) {

            /* Default Value Processing for Resistor Instance */

            if (!here->REStempGiven) {
                here->REStemp = ckt->CKTtemp;
                if (!here->RESdtempGiven)
                    here->RESdtemp = 0.0;
            } else {
                here->RESdtemp = 0.0;
                if (here->RESdtempGiven)
                    printf("%s: Instance temperature specified, dtemp ignored\n", here->RESname);
            }

            RESupdate_conduct(here, TRUE);
        }
    }

    return OK;
}


void
RESupdate_conduct(RESinstance *here, bool spill_warnings)
{
    RESmodel *model = RESmodPtr(here);
    double factor;
    double difference;
    double tc1, tc2, tce;

    if (!here->RESresGiven) {
        if (here->RESlength * here->RESwidth * model->RESsheetRes > 0.0) {
            here->RESresist =
                (here->RESlength - 2 * model->RESshort) /
                (here->RESwidth - 2 * model->RESnarrow) *
                model->RESsheetRes;
        } else if (model->RESresGiven) {
            here->RESresist = model->RESres;
        } else {
            if (spill_warnings)
                SPfrontEnd->IFerrorf (ERR_WARNING,
                                      "%s: resistance to low, set to 1 mOhm", here->RESname);
            here->RESresist = 1e-03;
        }
    }

    difference = (here->REStemp + here->RESdtemp) - model->REStnom;

    /* instance parameters tc1,tc2 and tce will override
       model parameters tc1,tc2 and tce */
    if (here->REStc1Given)
        tc1 = here->REStc1; /* instance */
    else
        tc1 = model->REStempCoeff1; /* model */

    if (here->REStc2Given)
        tc2 = here->REStc2;
    else
        tc2 = model->REStempCoeff2;

    if (here->REStceGiven)
        tce = here->REStce;
    else
        tce = model->REStempCoeffe;

    if (here->REStceGiven || model->REStceGiven)
        factor = pow(1.01, tce * difference);
    else
        factor = (((tc2 * difference) + tc1) * difference) + 1.0;

    if (!here->RESscaleGiven)
        here->RESscale = 1;

    here->RESconduct = here->RESm / (here->RESresist * factor * here->RESscale);

    /* Paolo Nenzi:  AC value */
    if (here->RESacresGiven) {
        here->RESacConduct = here->RESm / (here->RESacResist * factor * here->RESscale);
    } else {
        here->RESacConduct = here->RESconduct;
        here->RESacResist = here->RESresist;
    }
}
