/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles

This function is obsolete (was used by an old sensitivity analysis)
**********/

/* Pretty print the sensitivity info for all 
 * the mutual inductors in the circuit.
 */

#include "ngspice/ngspice.h"
#include "ngspice/smpdefs.h"
#include "ngspice/cktdefs.h"
#include "inddefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


/* ARGSUSED */
void
MUTsPrint(GENmodel *inModel, CKTcircuit* ckt)
{
    MUTmodel *model = (MUTmodel*)inModel;
    MUTinstance *here;

    NG_IGNORE(ckt);

    printf("MUTUAL INDUCTORS-----------------\n");
    /*  loop through all the inductor models */
    for( ; model != NULL; model = MUTnextModel(model)) {

        printf("Model name:%s\n",model->MUTmodName);

        /* loop through all the instances of the model */
        for (here = MUTinstances(model); here != NULL ;
                here=MUTnextInstance(here)) {

            printf("    Instance name:%s\n",here->MUTname);
            printf("      Mutual Inductance: %g ",here->MUTcoupling);
            printf(here->MUTindGiven ? "(specified)\n" : "(default)\n");
            printf("      coupling factor: %g \n",here->MUTfactor);
            printf("      inductor 1 name: %s \n",here->MUTindName1);
            printf("      inductor 2 name: %s \n",here->MUTindName2);
            printf("    MUTsenParmNo:%d\n",here->MUTsenParmNo);

        }
    }
}
