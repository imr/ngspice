/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

/* Pretty print the sensitivity info for all 
 * the capacitors in the circuit.
 */

#include "ngspice.h"
#include <stdio.h>
#include "cktdefs.h"
#include "capdefs.h"
#include "trandefs.h"
#include "sperror.h"
#include "suffix.h"


void
CAPsPrint(inModel,ckt)
    GENmodel *inModel;
    register CKTcircuit *ckt;
{
    register CAPmodel *model = (CAPmodel*)inModel;
    register CAPinstance *here;

    printf("CAPACITORS-----------------\n");
    /*  loop through all the capacitor models */
    for( ; model != NULL; model = model->CAPnextModel ) {

        printf("Model name:%s\n",model->CAPmodName);

        /* loop through all the instances of the model */
        for (here = model->CAPinstances; here != NULL ;
                here=here->CAPnextInstance) {
	    if (here->CAPowner != ARCHme) continue;
                

            printf("    Instance name:%s\n",here->CAPname);
            printf("      Positive, negative nodes: %s, %s\n",
            CKTnodName(ckt,here->CAPposNode),CKTnodName(ckt,here->CAPnegNode));
            printf("      Capacitance: %e",here->CAPcapac);
            printf(here->CAPcapGiven ? "(specified)\n" : "(default)\n");
            printf("    CAPsenParmNo:%d\n",here->CAPsenParmNo);

        }
    }
}

