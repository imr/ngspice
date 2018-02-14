/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

/* Pretty print the sensitivity info for 
 * all the VCCS in the circuit.
 */

#include "ngspice/ngspice.h"
#include "ngspice/smpdefs.h"
#include "ngspice/cktdefs.h"
#include "vccsdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


void
VCCSsPrint(GENmodel *inModel, CKTcircuit *ckt)
{
    VCCSmodel *model = (VCCSmodel *)inModel;
    VCCSinstance *here;

    printf("VOLTAGE CONTROLLED CURRENT SOURCES-----------------\n");
    /*  loop through all the source models */
    for( ; model != NULL; model = VCCSnextModel(model)) {

        printf("Model name:%s\n",model->VCCSmodName);

        /* loop through all the instances of the model */
        for (here = VCCSinstances(model); here != NULL ;
                here=VCCSnextInstance(here)) {

            printf("    Instance name:%s\n",here->VCCSname);
            printf("      Positive, negative nodes: %s, %s\n",
            CKTnodName(ckt,here->VCCSposNode),
                    CKTnodName(ckt,here->VCCSnegNode));
            printf("      Controlling Positive, negative nodes: %s, %s\n",
            CKTnodName(ckt,here->VCCScontPosNode),
                    CKTnodName(ckt,here->VCCScontNegNode));
            printf("      Coefficient: %f\n",here->VCCScoeff);
            printf("    VCCSsenParmNo:%d\n",here->VCCSsenParmNo);
        }
    }
}
