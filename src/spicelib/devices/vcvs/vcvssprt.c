/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

/* Pretty print the sensitivity info for 
 * all the VCVS in the circuit.
 */

#include "ngspice.h"
#include "cktdefs.h"
#include "vcvsdefs.h"
#include "sperror.h"
#include "suffix.h"


void
VCVSsPrint(GENmodel *inModel, CKTcircuit *ckt)
{
    VCVSmodel *model = (VCVSmodel *)inModel;
    VCVSinstance *here;

    printf("VOLTAGE CONTROLLED VOLTAGE SOURCES-----------------\n");
    /*  loop through all the voltage source models */
    for( ; model != NULL; model = model->VCVSnextModel ) {

        printf("Model name:%s\n",model->VCVSmodName);

        /* loop through all the instances of the model */
        for (here = model->VCVSinstances; here != NULL ;
                here=here->VCVSnextInstance) {
	    if (here->VCVSowner != ARCHme) continue;

            printf("    Instance name:%s\n",here->VCVSname);
            printf("      Positive, negative nodes: %s, %s\n",
                    CKTnodName(ckt,here->VCVSposNode),
                    CKTnodName(ckt,here->VCVSnegNode));
            printf("      Controlling Positive, negative nodes: %s, %s\n",
                    CKTnodName(ckt,here->VCVScontPosNode),
                    CKTnodName(ckt,here->VCVScontNegNode));
            printf("      Branch equation number: %s\n",
                    CKTnodName(ckt,here->VCVSbranch));
            printf("      Coefficient: %f\n",here->VCVScoeff);
            printf("    VCVSsenParmNo:%d\n",here->VCVSsenParmNo);
        }
    }
}
