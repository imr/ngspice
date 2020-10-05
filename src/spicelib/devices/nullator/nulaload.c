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

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "nuladefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


/*ARGSUSED*/
int
NULAload(GENmodel *inModel, CKTcircuit *ckt)
        /* actually load the current voltage value into the 
         * sparse matrix previously provided 
         */
{
    NULAmodel *model = (NULAmodel *)inModel;
    NULAinstance *here;

    NG_IGNORE(ckt);

    /*  loop through all the voltage source models */
    for( ; model != NULL; model = NULAnextModel(model)) {

        /* loop through all the instances of the model */
        for (here = NULAinstances(model); here != NULL ;
                here=NULAnextInstance(here)) {
            
            *(here->NULAibrContPosPtr) += 1.0 ;
            *(here->NULAibrContNegPtr) -= 1.0 ;
	    *(ckt->CKTrhs + (here->NULAbranch)) += here->NULAoffset;
        }
    }
    return(OK);
}
