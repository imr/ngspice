/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

    /* load the current source structure with those pointers needed later
     * for fast matrix loading
     */

#include "ngspice/ngspice.h"
#include "ngspice/smpdefs.h"
#include "ngspice/cktdefs.h"
#include "isrcdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


/*ARGSUSED*/
int
ISRCsetup(SMPmatrix *matrix, GENmodel *inModel, CKTcircuit *ckt, int *states)
{
    ISRCmodel *model = (ISRCmodel *)inModel;
    ISRCinstance *here;

    NG_IGNORE(states);
    NG_IGNORE(ckt);

    /*  loop through all the current source models */
    for( ; model != NULL; model = model->ISRCnextModel ) {

        /* loop through all the instances of the model */
        for (here = model->ISRCinstances; here != NULL ;
                here=here->ISRCnextInstance) {

/* macro to make elements with built in test for out of memory */
#define TSTALLOC(ptr,first,second) \
do { if((here->ptr = SMPmakeElt(matrix, here->first, here->second)) == NULL){\
    return(E_NOMEM);\
} } while(0)

            if (here->ISRCbranch == 0) {
                here->ISRCposPrimeNode = here->ISRCposNode;
            } else {
                here->ISRCposPrimeNode = here->ISRCbranch;
                TSTALLOC(ISRC_pos_ibr, ISRCposNode, ISRCbranch);
                TSTALLOC(ISRC_posPrime_ibr, ISRCposPrimeNode, ISRCbranch);
            }
        }
    }
    return(OK);
}


int
ISRCunsetup(GENmodel *inModel, CKTcircuit *ckt)
{
    ISRCmodel *model = (ISRCmodel *) inModel;
    ISRCinstance *here;

    for (; model; model = model->ISRCnextModel)
        for (here = model->ISRCinstances; here; here = here->ISRCnextInstance)
            if (here->ISRCbranch) {
                CKTdltNNum(ckt, here->ISRCbranch);
                here->ISRCbranch = 0;
                here->ISRCposPrimeNode = 0;
            }

    return OK;
}
