/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: Alan Gillespie

This function is obsolete (was used by an old sensitivity analysis)
**********/

    /* Pretty print the sensitivity info for all the MOS9 
     * devices  in the circuit.
     */

#include "ngspice.h"
#include "smpdefs.h"
#include "cktdefs.h"
#include "mos9defs.h"
#include "sperror.h"
#include "suffix.h"

void
MOS9sPrint(GENmodel *inModel, CKTcircuit *ckt)
{
    MOS9model *model = (MOS9model *)inModel;
    MOS9instance *here;

    printf("LEVEL 9 MOSFETS (AG) -----------------\n");
    /*  loop through all the MOS9 models */
    for( ; model != NULL; model = model->MOS9nextModel ) {

        printf("Model name:%s\n",model->MOS9modName);

        /* loop through all the instances of the model */
        for (here = model->MOS9instances; here != NULL ;
                here=here->MOS9nextInstance) {
            if (here->MOS9owner != ARCHme) continue;

            printf("    Instance name:%s\n",here->MOS9name);
            printf("      Drain, Gate , Source nodes: %s, %s ,%s\n",
            CKTnodName(ckt,here->MOS9dNode),CKTnodName(ckt,here->MOS9gNode),
            CKTnodName(ckt,here->MOS9sNode));

            printf("  Multiplier: %g ",here->MOS9m);
            printf(here->MOS9mGiven ? "(specified)\n" : "(default)\n");
            printf("      Length: %g ",here->MOS9l);
            printf(here->MOS9lGiven ? "(specified)\n" : "(default)\n");
            printf("      Width: %g ",here->MOS9w);
            printf(here->MOS9wGiven ? "(specified)\n" : "(default)\n");
            if(here->MOS9sens_l == 1){
                printf("    MOS9senParmNo:l = %d ",here->MOS9senParmNo);
            }
            else{ 
                printf("    MOS9senParmNo:l = 0 ");
            }
            if(here->MOS9sens_w == 1){
                printf("    w = %d \n",here->MOS9senParmNo + here->MOS9sens_l);
            }
            else{ 
                printf("    w = 0 \n");
            }


        }
    }
}

