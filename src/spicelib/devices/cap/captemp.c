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

#include "ngspice.h"
#include "cktdefs.h"
#include "capdefs.h"
#include "sperror.h"
#include "suffix.h"


/*ARGSUSED*/
int
CAPtemp(GENmodel *inModel, CKTcircuit *ckt)

{
    CAPmodel *model = (CAPmodel*)inModel;
    CAPinstance *here;
    double difference;
    double factor;

    /*  loop through all the capacitor models */
    for( ; model != NULL; model = model->CAPnextModel ) {

        /* loop through all the instances of the model */
        for (here = model->CAPinstances; here != NULL ;
                here=here->CAPnextInstance) {
	    if (here->CAPowner != ARCHme) continue;

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
	        if (!model->CAPmCapGiven){ /* No model capacitange given */
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
	    
	    factor = 1.0 + (model->CAPtempCoeff1)*difference +
	             (model->CAPtempCoeff2)*difference*difference;
            
	    here->CAPcapac = here->CAPcapac * factor * here->CAPscale;     
		     
	}
    }
    return(OK);
}

