/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles

This function is obsolete (was used by an old sensitivity analysis)
**********/
/*
 */

/* Pretty print the sensitivity info for all 
 * the diodes in the circuit.
 */

#include "ngspice/ngspice.h"
#include "ngspice/smpdefs.h"
#include "ngspice/cktdefs.h"
#include "diodefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


void
DIOsPrint(GENmodel *inModel, CKTcircuit *ckt)
{
    DIOmodel *model = (DIOmodel*)inModel;
    DIOinstance *here;

    printf("DIOS-----------------\n");
    /*  loop through all the diode models */
    for( ; model != NULL; model = DIOnextModel(model)) {

        printf("Model name:%s\n",model->DIOmodName);

        /* loop through all the instances of the model */
        for (here = DIOinstances(model); here != NULL ;
                here=DIOnextInstance(here)) {

            printf("    Instance name:%s\n",here->DIOname);
            printf("      Positive, negative nodes: %s, %s\n",
            CKTnodName(ckt,here->DIOposNode),CKTnodName(ckt,here->DIOnegNode));
            printf("      Area: %g ",here->DIOarea);
            printf(here->DIOareaGiven ? "(specified)\n" : "(default)\n");
            printf("    DIOsenParmNo:%d\n",here->DIOsenParmNo);

        }
    }
}
