/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: 2000 AlansFixes

This function is obsolete (was used by an old sensitivity analysis)
**********/

#include "ngspice/ngspice.h"
#include "ngspice/smpdefs.h"
#include "ngspice/cktdefs.h"
#include "mos2defs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

void
MOS2sPrint(GENmodel *inModel, CKTcircuit *ckt)
        /* Pretty print the sensitivity info for all the MOS2 
         * devices  in the circuit.
         */
{
    MOS2model *model = (MOS2model *)inModel;
    MOS2instance *here;

    printf("LEVEL 2 MOSFETS-----------------\n");
    /*  loop through all the MOS2 models */
    for( ; model != NULL; model = MOS2nextModel(model)) {

        printf("Model name:%s\n",model->MOS2modName);

        /* loop through all the instances of the model */
        for (here = MOS2instances(model); here != NULL ;
                here=MOS2nextInstance(here)) {

            printf("    Instance name:%s\n",here->MOS2name);
            printf("      Drain, Gate , Source nodes: %s, %s ,%s\n",
            CKTnodName(ckt,here->MOS2dNode),CKTnodName(ckt,here->MOS2gNode),
            CKTnodName(ckt,here->MOS2sNode));

            printf("  Multiplier: %g ",here->MOS2m);
            printf(here->MOS2mGiven ? "(specified)\n" : "(default)\n"); 
            printf("      Length: %g ",here->MOS2l);
            printf(here->MOS2lGiven ? "(specified)\n" : "(default)\n");
            printf("      Width: %g ",here->MOS2w);
            printf(here->MOS2wGiven ? "(specified)\n" : "(default)\n");
            if(here->MOS2sens_l == 1){
                printf("    MOS2senParmNo:l = %d ",here->MOS2senParmNo);
            }
            else{ 
                printf("    MOS2senParmNo:l = 0 ");
            }
            if(here->MOS2sens_w == 1){
                printf("    w = %d \n",here->MOS2senParmNo + here->MOS2sens_l);
            }
            else{ 
                printf("    w = 0 \n");
            }


        }
    }
}

