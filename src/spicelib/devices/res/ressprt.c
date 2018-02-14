/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: Apr 2002 - Paolo Nenzi

This function is obsolete (was used by an old sensitivity analysis)
**********/

/* Pretty print the sensitivity info for all 
 * the resistors in the circuit.
 */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "resdefs.h"
#include "ngspice/sperror.h"


void
RESsPrint(GENmodel *inModel, CKTcircuit *ckt)
{
    RESmodel *model = (RESmodel *)inModel;
    RESinstance *here;
    printf("RESISTORS-----------------\n");

    /*  loop through all the resistor models */
    for( ; model != NULL; model = RESnextModel(model)) {

        printf("Model name:%s\n",model->RESmodName);

        /* loop through all the instances of the model */
        for (here = RESinstances(model); here != NULL ;
                here=RESnextInstance(here)) {

            printf("    Instance name:%s\n",here->RESname);
            printf("      Positive, negative nodes: %s, %s\n",
            CKTnodName(ckt,here->RESposNode),CKTnodName(ckt,here->RESnegNode));
	            
	    printf("  Multiplier: %g ",here->RESm);
            printf(here->RESmGiven ? "(specified)\n" : "(default)\n");
	        
            printf("      Resistance: %f ",here->RESresist);
            printf(here->RESresGiven ? "(specified)\n" : "(default)\n");
            printf("    RESsenParmNo:%d\n",here->RESsenParmNo);

        }
    }
}
