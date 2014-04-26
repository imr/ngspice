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

#ifdef USE_CUSPICE
#include "ngspice/CUSPICE/CUSPICE.h"
#endif

/*ARGSUSED*/
int
CAPtemp(GENmodel *inModel, CKTcircuit *ckt)

{
    CAPmodel *model = (CAPmodel*)inModel;
    CAPinstance *here;
    double difference;
    double factor;
    double tc1, tc2;

#ifdef USE_CUSPICE
    int i, status ;
#endif

    /*  loop through all the capacitor models */
    for( ; model != NULL; model = CAPnextModel(model)) {

#ifdef USE_CUSPICE
        i = 0 ;
#endif

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

            if (!here->CAPcapGiven)  { /* No instance capacitance given */
                if (!model->CAPmCapGiven) { /* No model capacitange given */
                    here->CAPcapac =
                        model->CAPcj *
                        (here->CAPwidth - model->CAPnarrow) *
                        (here->CAPlength - model->CAPshort) +
                        model->CAPcjsw * 2 * (
                            (here->CAPlength - model->CAPshort) +
                            (here->CAPwidth - model->CAPnarrow) );
                } else {
                    here->CAPcapac = model->CAPmCap;
                }
            }

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

#ifdef USE_CUSPICE
            model->CAPparamCPU.CAPcapacArray[i] = here->CAPcapac ;
            model->CAPparamCPU.CAPmArray[i] = here->CAPm ;
            model->CAPparamCPU.CAPposNodeArray[i] = here->CAPposNode ;
            model->CAPparamCPU.CAPnegNodeArray[i] = here->CAPnegNode ;
            model->CAPparamCPU.CAPstateArray[i] = here->CAPstate ;

            i++ ;
#endif

        }

#ifdef USE_CUSPICE
        status = cuCAPtemp ((GENmodel *)model) ;
        if (status != 0)
            return (E_NOMEM) ;
#endif

    }
    return (OK) ;
}

