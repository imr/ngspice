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

#ifdef USE_CUSPICE
#include "ngspice/CUSPICE/CUSPICE.h"
#endif

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
    double tc1, tc2, tce;

#ifdef USE_CUSPICE
    int i, status ;
#endif

    /*  loop through all the resistor models */
    for( ; model != NULL; model = model->RESnextModel ) {

#ifdef USE_CUSPICE
        i = 0 ;
#endif

        /* loop through all the instances of the model */
        for (here = model->RESinstances; here != NULL ;
                here=here->RESnextInstance) {

            if (!here->RESresGiven) {
                if (here->RESlength * here->RESwidth * model->RESsheetRes > 0.0) {
                    here->RESresist =
                        (here->RESlength - model->RESshort) /
                        (here->RESwidth - model->RESnarrow) *
                        model->RESsheetRes;
                } else if (model->RESresGiven) {
                    here->RESresist = model->RESres;
                } else {
                    SPfrontEnd->IFerrorf (ERR_WARNING,
                                          "%s: resistance to low, set to 1 mOhm", here->RESname);
                    here->RESresist = 1e-03;
                }
            }

            /* Default Value Processing for Resistor Instance */

            if(!here->REStempGiven) {
                here->REStemp   = ckt->CKTtemp;
                if(!here->RESdtempGiven)   here->RESdtemp  = 0.0;
            } else { /* REStempGiven */
                here->RESdtemp = 0.0;
                if (here->RESdtempGiven)
                    printf("%s: Instance temperature specified, dtemp ignored\n", here->RESname);
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

            if ((here->REStceGiven)||(model->REStceGiven))
                factor = pow(1.01, tce * difference);
            else
                factor = (((tc2 * difference) + tc1) * difference) + 1.0;

            here -> RESconduct = (1.0/(here->RESresist * factor * here->RESscale));

            /* Paolo Nenzi:  AC value */
            if(here->RESacresGiven)
                here->RESacConduct = (1.0/(here->RESacResist * factor * here->RESscale));
            else {
                here -> RESacConduct = here -> RESconduct;
                here -> RESacResist = here -> RESresist;
            }

#ifdef USE_CUSPICE
            model->RESparamCPU.REStc1GivenArray[i] = here->REStc1Given ;
            model->RESparamCPU.REStc2GivenArray[i] = here->REStc2Given ;
            model->RESparamCPU.RESmGivenArray[i] = here->RESmGiven ;
            model->RESparamCPU.REStc1Array[i] = here->REStc1 ;
            model->RESparamCPU.REStc2Array[i] = here->REStc2 ;
            model->RESparamCPU.RESmArray[i] = here->RESm ;
            model->RESparamCPU.RESposNodeArray[i] = here->RESposNode ;
            model->RESparamCPU.RESnegNodeArray[i] = here->RESnegNode ;
            model->RESparamCPU.RESconductArray[i] = here->RESconduct ;
            model->RESparamCPU.REStempArray[i] = here->REStemp ;
            model->RESparamCPU.RESdtempArray[i] = here->RESdtemp ;

            i++ ;
#endif

        }

#ifdef USE_CUSPICE
        status = cuREStemp ((GENmodel *)model) ;
        if (status != 0)
            return (E_NOMEM) ;
#endif

    }
    return (OK) ;
}
