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
    for( ; model != NULL; model = model->CCCSnextModel ) {

        /* loop through all the instances of the model */
        for (here = model->CCCSinstances; here != NULL ;
                here=here->CCCSnextInstance) {
            
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

            if (here->CCCSbranch == 0) {
                here->CCCSposPrimeNode = here->CCCSposNode;
            } else {
                here->CCCSposPrimeNode = here->CCCSbranch;
                TSTALLOC(CCCS_pos_ibr, CCCSposNode, CCCSbranch);
                TSTALLOC(CCCS_posPrime_ibr, CCCSposPrimeNode, CCCSbranch);
            }

            TSTALLOC(CCCSposContBrptr, CCCSposPrimeNode, CCCScontBranch);
            TSTALLOC(CCCSnegContBrptr, CCCSnegNode, CCCScontBranch);
        }
    }
    return(OK);
}


int
CCCSunsetup(GENmodel *inModel, CKTcircuit *ckt)
{
    CCCSmodel *model = (CCCSmodel *) inModel;
    CCCSinstance *here;

    for (; model; model = model->CCCSnextModel)
        for (here = model->CCCSinstances; here; here = here->CCCSnextInstance)
            if (here->CCCSbranch) {
                CKTdltNNum(ckt, here->CCCSbranch);
                here->CCCSbranch = 0;
                here->CCCSposPrimeNode = 0;
            }

    return OK;
}
