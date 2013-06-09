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
    double factor;
    double difference;
    double tc1, tc2;


    /*  loop through all the resistor models */
    for( ; model != NULL; model = model->RESnextModel ) {

        /* Default Value Processing for Resistor Models */
        if(!model->REStnomGiven) model->REStnom         = ckt->CKTnomTemp;
        if(!model->RESsheetResGiven) model->RESsheetRes = 0.0;
        if(!model->RESdefWidthGiven) model->RESdefWidth = 10e-6; /*M*/
        if(!model->RESdefLengthGiven) model->RESdefLength = 10e-6;
        if(!model->REStc1Given) model->REStempCoeff1    = 0.0;
        if(!model->REStc2Given) model->REStempCoeff2    = 0.0;
        if(!model->RESnarrowGiven) model->RESnarrow     = 0.0;
        if(!model->RESshortGiven) model->RESshort       = 0.0;
        if(!model->RESfNcoefGiven) model->RESfNcoef     = 0.0;
        if(!model->RESfNexpGiven) model->RESfNexp       = 1.0;

        /* loop through all the instances of the model */
        for (here = model->RESinstances; here != NULL ;
                here=here->RESnextInstance) {

            /* Default Value Processing for Resistor Instance */

            if(!here->REStempGiven) {
                here->REStemp   = ckt->CKTtemp;
                if(!here->RESdtempGiven)   here->RESdtemp  = 0.0;
            } else { /* REStempGiven */
                here->RESdtemp = 0.0;
                if (here->RESdtempGiven)
                    printf("%s: Instance temperature specified, dtemp ignored\n", here->RESname);
            }

            if(!here->RESwidthGiven)   here->RESwidth  = model->RESdefWidth;
            if(!here->RESlengthGiven)  here->RESlength = model->RESdefLength;
            if(!here->RESscaleGiven)   here->RESscale  = 1.0;
            if(!here->RESmGiven)       here->RESm      = 1.0;
            if(!here->RESnoisyGiven)   here->RESnoisy  = 1;
            if(!here->RESresGiven)  {
                if(here->RESlength * here->RESwidth * model->RESsheetRes > 0.0) {
                    here->RESresist = model->RESsheetRes * (here->RESlength -
                                                            model->RESshort) / (here->RESwidth - model->RESnarrow);
                } else {
                    if(model->RESresGiven) {
                        here->RESresist = model->RESres;
                    } else {
                        SPfrontEnd->IFerror (ERR_WARNING,
                                             "%s: resistance to low, set to 1 mOhm", &(here->RESname));
                        here->RESresist = 1e-03;
                    }
                }
            }

            difference = (here->REStemp + here->RESdtemp) - model->REStnom;

            /* instance parameters tc1 and tc2 will override
               model parameters tc1 and tc2 */
            if (here->REStc1Given)
                tc1 = here->REStc1; /* instance */
            else
                tc1 = model->REStempCoeff1; /* model */

            if (here->REStc2Given)
                tc2 = here->REStc2;
            else
                tc2 = model->REStempCoeff2;

            factor = 1.0 + tc1*difference +
                     tc2*difference*difference;

            here -> RESconduct = (1.0/(here->RESresist * factor * here->RESscale));

            /* Paolo Nenzi:  AC value */
            if(here->RESacresGiven)
                here->RESacConduct = (1.0/(here->RESacResist * factor * here->RESscale));
            else {
                here -> RESacConduct = here -> RESconduct;
                here -> RESacResist = here -> RESresist;
            }
        }
    }
    return(OK);
}
