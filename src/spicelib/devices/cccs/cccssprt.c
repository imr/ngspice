/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

/* Pretty print the sensitivity info for all 
 * the CCCS in the circuit.
 */

#include "ngspice.h"
#include "smpdefs.h"
#include "cktdefs.h"
#include "cccsdefs.h"
#include "sperror.h"
#include "suffix.h"


void
CCCSsPrint(GENmodel *inModel, CKTcircuit *ckt)
{
    CCCSmodel *model = (CCCSmodel*)inModel;
    CCCSinstance *here;

    printf("CURRENT CONTROLLED CURRENT SOURCES-----------------\n");
    /*  loop through all the voltage source models */
    for( ; model != NULL; model = model->CCCSnextModel ) {

        printf("Model name:%s\n",model->CCCSmodName);

        /* loop through all the instances of the model */
        for (here = model->CCCSinstances; here != NULL ;
                here=here->CCCSnextInstance) {
	    if (here->CCCSowner != ARCHme) continue;

            printf("    Instance name:%s\n",here->CCCSname);
            printf("      Positive, negative nodes: %s, %s\n",
                    CKTnodName(ckt,here->CCCSposNode),
                    CKTnodName(ckt,here->CCCSnegNode));
            printf("      Controlling source name: %s\n",
                    here->CCCScontName);
            printf("      Controlling Branch equation number: %s\n",
                    CKTnodName(ckt,here->CCCScontBranch));
            printf("      Coefficient: %f\n",here->CCCScoeff);
            printf("    CCCSsenParmNo:%d\n",here->CCCSsenParmNo);

        }
    } 
}
