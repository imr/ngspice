/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: September 2003 Paolo Nenzi

This function is obsolete (was used by an old sensitivity analysis)
**********/
/*
 */

/* Pretty print the sensitivity info for all 
 * the capacitors in the circuit.
 */

#include "ngspice.h"
#include "cktdefs.h"
#include "capdefs.h"
#include "trandefs.h"
#include "sperror.h"
#include "suffix.h"


void
CAPsPrint(GENmodel *inModel, CKTcircuit *ckt)
{
    CAPmodel *model = (CAPmodel*)inModel;
    CAPinstance *here;

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

