/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles

This function is obsolete (was used by an old sensitivity analysis)
**********/

/* Pretty print the sensitivity info for all 
 * the inductors in the circuit.
 */

#include "ngspice.h"
#include "smpdefs.h"
#include "cktdefs.h"
#include "inddefs.h"
#include "sperror.h"
#include "suffix.h"

void
INDsPrint(GENmodel *inModel, CKTcircuit *ckt)
{
    INDmodel *model = (INDmodel*)inModel;
    INDinstance *here;

    printf("INDUCTORS----------\n");
    /*  loop through all the inductor models */
    for( ; model != NULL; model = model->INDnextModel ) {

        printf("Model name:%s\n",model->INDmodName);

        /* loop through all the instances of the model */
        for (here = model->INDinstances; here != NULL ;
                here=here->INDnextInstance) {
	    if (here->INDowner != ARCHme) continue;

            printf("    Instance name:%s\n",here->INDname);
            printf("      Positive, negative nodes: %s, %s\n",
            CKTnodName(ckt,here->INDposNode),CKTnodName(ckt,here->INDnegNode));
            printf("      Branch Equation: %s\n",CKTnodName(ckt,here->INDbrEq));
            printf("      Inductance: %g ",here->INDinduct);
            printf(here->INDindGiven ? "(specified)\n" : "(default)\n");
            printf("    INDsenParmNo:%d\n",here->INDsenParmNo);
        }
    }
}
