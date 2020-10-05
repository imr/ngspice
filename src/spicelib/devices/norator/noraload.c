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
#include "noradefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


/*ARGSUSED*/
int
NORAload(GENmodel *inModel, CKTcircuit *ckt)
        /* actually load the current voltage value into the 
         * sparse matrix previously provided 
         */
{
    NORAmodel *model = (NORAmodel *)inModel;
    NORAinstance *here;

    NG_IGNORE(ckt);

    /*  loop through all the voltage source models */
    for( ; model != NULL; model = NORAnextModel(model)) {

        /* loop through all the instances of the model */
        for (here = NORAinstances(model); here != NULL ;
                here=NORAnextInstance(here)) {
            
            *(here->NORAposIbrPtr) += 1.0 ;
            *(here->NORAnegIbrPtr) -= 1.0 ;
	    /* equations to be loaded by a nullator :
            *(here->NORAibrContPosPtr) += 1.0 ;
            *(here->NORAibrContNegPtr) -= 1.0 ;
	    *(ckt->CKTrhs + (here->NORAbranch)) += here->NORAoffset;
	    */
        }
    }
    return(OK);
}
