/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: Apr 2000 - Paolo Nenzi
**********/

#include "ngspice.h"
#include "cktdefs.h"
#include "resdefs.h"
#include "sperror.h"


/* actually load the current resistance value into the sparse matrix
 * previously provided */
int
RESload(GENmodel *inModel, CKTcircuit *ckt)
{
    RESmodel *model = (RESmodel *)inModel;
    double m;
    double difference;
    double factor;

		
    /*  loop through all the resistor models */
    for( ; model != NULL; model = model->RESnextModel ) {
	RESinstance *here;

        /* loop through all the instances of the model */
        for (here = model->RESinstances; here != NULL ;
	    here = here->RESnextInstance) {
	    
	    if(!here->REStc1Given) here->REStc1    = 0.0;
	    if(!here->REStc2Given) here->REStc2    = 0.0;
	    if(!here->RESmGiven)   here->RESm      = 1.0;

	    if (here->RESowner != ARCHme) continue;

	    here->REScurrent = (*(ckt->CKTrhsOld+here->RESposNode) - 
		*(ckt->CKTrhsOld+here->RESnegNode)) * here->RESconduct;
		
	    difference = (here->REStemp + here->RESdtemp) - 300.15;
	    factor = 1.0 + (here->REStc1)*difference + 
		(here->REStc2)*difference*difference;
	    
	    m = (here->RESm)/factor; 

	    *(here->RESposPosptr) += m * here->RESconduct;
	    *(here->RESnegNegptr) += m * here->RESconduct;
	    *(here->RESposNegptr) -= m * here->RESconduct;
	    *(here->RESnegPosptr) -= m * here->RESconduct;
        }
    }
    return(OK);
}


/* actually load the current resistance value into the sparse matrix
 * previously provided */
int
RESacload(GENmodel *inModel, CKTcircuit *ckt)
{
    RESmodel *model = (RESmodel *)inModel;
    double m;
    double difference;
    double factor;

    /*  loop through all the resistor models */
    for( ; model != NULL; model = model->RESnextModel ) {
	RESinstance *here;

        /* loop through all the instances of the model */
        for (here = model->RESinstances; here != NULL ;
	     here = here->RESnextInstance) {
	    
	    if(!here->REStc1Given) here->REStc1    = 0.0;
	    if(!here->REStc2Given) here->REStc2    = 0.0;
	    if(!here->RESmGiven)   here->RESm      = 1.0;
            
	    if (here->RESowner != ARCHme) continue;
	    
	    difference = (here->REStemp + here->RESdtemp) - 300.15;
	    factor = 1.0 + (here->REStc1)*difference + 
		(here->REStc2)*difference*difference;
	    
	    m = (here->RESm)/factor; 
	    
            if(here->RESacresGiven) {
                *(here->RESposPosptr) += m * here->RESacConduct;
                *(here->RESnegNegptr) += m * here->RESacConduct;
                *(here->RESposNegptr) -= m * here->RESacConduct;
                *(here->RESnegPosptr) -= m * here->RESacConduct;
            } else {
                *(here->RESposPosptr) += m * here->RESconduct;
                *(here->RESnegNegptr) += m * here->RESconduct;
                *(here->RESposNegptr) -= m * here->RESconduct;
                *(here->RESnegPosptr) -= m * here->RESconduct;
            }
        }
    }
    return(OK);
}
