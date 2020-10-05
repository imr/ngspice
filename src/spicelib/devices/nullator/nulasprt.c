/**********
Author: Florian Ballenegger 2020
Adapted from VCVS device code.
**********/
/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

/* Pretty print the sensitivity info for 
 * all the NULA in the circuit.
 */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "nuladefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


void
NULAsPrint(GENmodel *inModel, CKTcircuit *ckt)
{
    NULAmodel *model = (NULAmodel *)inModel;
    NULAinstance *here;

    printf("VOLTAGE CONTROLLED VOLTAGE SOURCES-----------------\n");
    /*  loop through all the voltage source models */
    for( ; model != NULL; model = NULAnextModel(model)) {

        printf("Model name:%s\n",model->NULAmodName);

        /* loop through all the instances of the model */
        for (here = NULAinstances(model); here != NULL ;
                here=NULAnextInstance(here)) {

            printf("    Instance name:%s\n",here->NULAname);
            printf("      Positive, negative nodes: %s, %s\n",
                    CKTnodName(ckt,here->NULAposNode),
                    CKTnodName(ckt,here->NULAnegNode));
            printf("      Controlling Positive, negative nodes: %s, %s\n",
                    CKTnodName(ckt,here->NULAcontPosNode),
                    CKTnodName(ckt,here->NULAcontNegNode));
            printf("      Branch equation number: %s\n",
                    CKTnodName(ckt,here->NULAbranch));
            printf("      Coefficient: %f\n",here->NULAcoeff);
            printf("    NULAsenParmNo:%d\n",here->NULAsenParmNo);
        }
    }
}
