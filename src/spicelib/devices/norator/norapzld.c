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
#include "ngspice/complex.h"
#include "ngspice/sperror.h"
#include "noradefs.h"
#include "ngspice/suffix.h"

/* file not used - use regular load function */

/*ARGSUSED*/
int
NORApzLoad(GENmodel *inModel, CKTcircuit *ckt, SPcomplex *s)
        /* actually load the current voltage value into the 
         * sparse matrix previously provided 
         */
{
    NORAmodel *model = (NORAmodel *)inModel;
    NORAinstance *here;

    NG_IGNORE(s);
    NG_IGNORE(ckt);

    /*  loop through all the voltage source models */
    for( ; model != NULL; model = NORAnextModel(model)) {

        /* loop through all the instances of the model */
        for (here = NORAinstances(model); here != NULL ;
                here=NORAnextInstance(here)) {
            
            *(here->NORAposIbrPtr) += 1.0 ;
            *(here->NORAnegIbrPtr) -= 1.0 ;
        }
    }
    return(OK);
}
