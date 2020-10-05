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
#include "nuladefs.h"
#include "ngspice/suffix.h"


/*ARGSUSED*/
int
NULApzLoad(GENmodel *inModel, CKTcircuit *ckt, SPcomplex *s)
        /* actually load the current voltage value into the 
         * sparse matrix previously provided 
         */
{
    NULAmodel *model = (NULAmodel *)inModel;
    NULAinstance *here;

    NG_IGNORE(s);
    NG_IGNORE(ckt);

    /*  loop through all the voltage source models */
    for( ; model != NULL; model = NULAnextModel(model)) {

        /* loop through all the instances of the model */
        for (here = NULAinstances(model); here != NULL ;
                here=NULAnextInstance(here)) {
            
            *(here->NULAibrContPosPtr) += 1.0 ;
            *(here->NULAibrContNegPtr) -= 1.0 ;
        }
    }
    return(OK);
}
