/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

    /* load the voltage source structure with those pointers needed later 
     * for fast matrix loading 
     */

#include "ngspice/ngspice.h"
#include "ngspice/smpdefs.h"
#include "ngspice/cktdefs.h"
#include "cccsdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


/*ARGSUSED*/
int
CCCSsetup(SMPmatrix *matrix, GENmodel *inModel, CKTcircuit *ckt, int *states)
{
    CCCSmodel *model = (CCCSmodel*)inModel;
    CCCSinstance *here;

    NG_IGNORE(states);

    /*  loop through all the voltage source models */
    for( ; model != NULL; model = CCCSnextModel(model)) {

        /* loop through all the instances of the model */
        for (here = CCCSinstances(model); here != NULL ;
                here=CCCSnextInstance(here)) {
            
            here->CCCScontBranch = CKTfndBranch(ckt,here->CCCScontName);
            if(here->CCCScontBranch == 0) {
                SPfrontEnd->IFerrorf (ERR_FATAL,
                        "%s: unknown controlling source %s", here->CCCSname, here->CCCScontName);
                return(E_BADPARM);
            }

/* macro to make elements with built in test for out of memory */
#define TSTALLOC(ptr,first,second) \
do { if((here->ptr = SMPmakeElt(matrix, here->first, here->second)) == NULL){\
    return(E_NOMEM);\
} } while(0)

            TSTALLOC(CCCSposContBrPtr,CCCSposNode,CCCScontBranch);
            TSTALLOC(CCCSnegContBrPtr,CCCSnegNode,CCCScontBranch);
        }
    }
    return(OK);
}
