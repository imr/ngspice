/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: 2000 AlansFixes

This function is obsolete (was used by an old sensitivity analysis)
**********/

/* Pretty print the sensitivity info for all 
 * the MOS1 devices in the circuit.
 */

#include "ngspice/ngspice.h"
#include "ngspice/smpdefs.h"
#include "ngspice/cktdefs.h"
#include "mos1defs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

void
MOS1sPrint(GENmodel *inModel, CKTcircuit *ckt)
/* Pretty print the sensitivity info for all the MOS1 
         * devices  in the circuit.
         */
{
    MOS1model *model = (MOS1model *)inModel;
    MOS1instance *here;

    printf("LEVEL 1 MOSFETS-----------------\n");
    /*  loop through all the MOS1 models */
    for( ; model != NULL; model = MOS1nextModel(model)) {

        printf("Model name:%s\n",model->MOS1modName);

        /* loop through all the instances of the model */
        for (here = MOS1instances(model); here != NULL ;
                here=MOS1nextInstance(here)) {

            printf("    Instance name:%s\n",here->MOS1name);
            printf("      Drain, Gate , Source nodes: %s, %s ,%s\n",
            CKTnodName(ckt,here->MOS1dNode),CKTnodName(ckt,here->MOS1gNode),
            CKTnodName(ckt,here->MOS1sNode));
            
            printf("  Multiplier: %g ",here->MOS1m);
            printf(here->MOS1mGiven ? "(specified)\n" : "(default)\n");
            
            printf("      Length: %g ",here->MOS1l);
            printf(here->MOS1lGiven ? "(specified)\n" : "(default)\n");
            printf("      Width: %g ",here->MOS1w);
            printf(here->MOS1wGiven ? "(specified)\n" : "(default)\n");
            if(here->MOS1sens_l == 1){
                printf("    MOS1senParmNo:l = %d ",here->MOS1senParmNo);
            }
            else{ 
                printf("    MOS1senParmNo:l = 0 ");
            }
            if(here->MOS1sens_w == 1){
                printf("    w = %d \n",here->MOS1senParmNo + here->MOS1sens_l);
            }
            else{ 
                printf("    w = 0 \n");
            }


        }
    }
}

