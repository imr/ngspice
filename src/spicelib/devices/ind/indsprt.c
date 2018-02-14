/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles

This function is obsolete (was used by an old sensitivity analysis)
**********/

/* Pretty print the sensitivity info for all 
 * the inductors in the circuit.
 */

#include "ngspice/ngspice.h"
#include "ngspice/smpdefs.h"
#include "ngspice/cktdefs.h"
#include "inddefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

void
INDsPrint(GENmodel *inModel, CKTcircuit *ckt)
{
    INDmodel *model = (INDmodel*)inModel;
    INDinstance *here;

    printf("INDUCTORS----------\n");
    /*  loop through all the inductor models */
    for( ; model != NULL; model = INDnextModel(model)) {

        printf("Model name:%s\n",model->INDmodName);

        /* loop through all the instances of the model */
        for (here = INDinstances(model); here != NULL ;
                here=INDnextInstance(here)) {

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
