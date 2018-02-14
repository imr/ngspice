/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles

This function is obsolete (was used by an old sensitivity analysis)
**********/

/* Pretty print the sensitivity info for all 
 * the bjts in the circuit.
 */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ngspice/smpdefs.h"
#include "bjtdefs.h"
#include "ngspice/const.h"
#include "ngspice/sperror.h"
#include "ngspice/ifsim.h"
#include "ngspice/suffix.h"


void
BJTsPrint(GENmodel *inModel, CKTcircuit *ckt)
{
    BJTmodel *model = (BJTmodel*)inModel;
    BJTinstance *here;

    printf("BJTS-----------------\n");
    /*  loop through all the BJT models */
    for( ; model != NULL; model = BJTnextModel(model)) {

        printf("Model name:%s\n",model->BJTmodName);

        /* loop through all the instances of the model */
        for (here = BJTinstances(model); here != NULL ;
                here=BJTnextInstance(here)) {

            ckt->CKTsenInfo->SEN_parmVal[here->BJTsenParmNo] = here->BJTarea;

            printf("    Instance name:%s\n",here->BJTname);
            printf("      Collector, Base , Emitter nodes: %s, %s ,%s\n",
            CKTnodName(ckt,here->BJTcolNode),CKTnodName(ckt,here->BJTbaseNode),
            CKTnodName(ckt,here->BJTemitNode));

            printf("      Area: %g ",here->BJTarea);
            printf(here->BJTareaGiven ? "(specified)\n" : "(default)\n");
            printf("    BJTsenParmNo:%d\n",here->BJTsenParmNo);

        }
    }
}
