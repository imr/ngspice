/**********
Copyright 2003 Paolo Nenzi
Author: 2003 Paolo Nenzi
**********/
/*
 */


#include "ngspice.h"
#include "cktdefs.h"
#include "inddefs.h"
#include "sperror.h"
#include "suffix.h"


/*ARGSUSED*/
int
INDtemp(GENmodel *inModel, CKTcircuit *ckt)
{
    INDmodel *model = (INDmodel*)inModel;
    INDinstance *here;
    double difference;
    double factor;

    /*  loop through all the inductor models */
    for( ; model != NULL; model = model->INDnextModel ) {

        /* loop through all the instances of the model */
        for (here = model->INDinstances; here != NULL ;
                here=here->INDnextInstance) {
	    if (here->INDowner != ARCHme) continue;

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
	    difference = (here->INDtemp + here->INDdtemp) - model->INDtnom;
	    
	    factor = 1.0 + (model->INDtempCoeff1)*difference +
	             (model->INDtempCoeff2)*difference*difference;
            
	    here->INDinduct = here->INDinduct * factor * here->INDscale;
	    here->INDinduct = here->INDinduct / here->INDm;     
	     
	}
    }
    return(OK);
}

