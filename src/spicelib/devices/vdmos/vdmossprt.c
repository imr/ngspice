/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: 2000 AlansFixes

This function is obsolete (was used by an old sensitivity analysis)
**********/

/* Pretty print the sensitivity info for all 
 * the VDMOS devices in the circuit.
 */

#include "ngspice/ngspice.h"
#include "ngspice/smpdefs.h"
#include "ngspice/cktdefs.h"
#include "vdmosdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

void
VDMOSsPrint(GENmodel *inModel, CKTcircuit *ckt)
/* Pretty print the sensitivity info for all the VDMOS 
         * devices  in the circuit.
         */
{
    VDMOSmodel *model = (VDMOSmodel *)inModel;
    VDMOSinstance *here;

    printf("LEVEL 1 MOSFETS-----------------\n");
    /*  loop through all the VDMOS models */
    for( ; model != NULL; model = VDMOSnextModel(model)) {

        printf("Model name:%s\n",model->VDMOSmodName);

        /* loop through all the instances of the model */
        for (here = VDMOSinstances(model); here != NULL ;
                here=VDMOSnextInstance(here)) {

            printf("    Instance name:%s\n",here->VDMOSname);
            printf("      Drain, Gate , Source nodes: %s, %s ,%s\n",
            CKTnodName(ckt,here->VDMOSdNode),CKTnodName(ckt,here->VDMOSgNode),
            CKTnodName(ckt,here->VDMOSsNode));
            
            printf("  Multiplier: %g ",here->VDMOSm);
            printf(here->VDMOSmGiven ? "(specified)\n" : "(default)\n");
            
            printf("      Length: %g ",here->VDMOSl);
            printf(here->VDMOSlGiven ? "(specified)\n" : "(default)\n");
            printf("      Width: %g ",here->VDMOSw);
            printf(here->VDMOSwGiven ? "(specified)\n" : "(default)\n");
            if(here->VDMOSsens_l == 1){
                printf("    VDMOSsenParmNo:l = %d ",here->VDMOSsenParmNo);
            }
            else{ 
                printf("    VDMOSsenParmNo:l = 0 ");
            }
            if(here->VDMOSsens_w == 1){
                printf("    w = %d \n",here->VDMOSsenParmNo + here->VDMOSsens_l);
            }
            else{ 
                printf("    w = 0 \n");
            }


        }
    }
}

