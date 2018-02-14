/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#include "ngspice/ngspice.h"
#include "ngspice/smpdefs.h"
#include "ngspice/cktdefs.h"
#include "vsrcdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

/* load the voltage source structure with those pointers needed later
 * for fast matrix loading */
int
VSRCpzSetup(SMPmatrix *matrix, GENmodel *inModel, CKTcircuit *ckt,
	    int *state)
{
    VSRCmodel *model = (VSRCmodel *)inModel;
    VSRCinstance *here;
    CKTnode *tmp;
    int error;

    NG_IGNORE(state);

    /*  loop through all the voltage source models */
    for( ; model != NULL; model = VSRCnextModel(model)) {

        /* loop through all the instances of the model */
        for (here = VSRCinstances(model); here != NULL ;
	     here = VSRCnextInstance(here)) {
            
            if (here->VSRCbranch == 0) {
                error = CKTmkCur(ckt,&tmp,here->VSRCname,"branch");
                if(error) return(error);
                here->VSRCbranch = tmp->number;
            }

/* macro to make elements with built in test for out of memory */
#define TSTALLOC(ptr,first,second) \
do { if((here->ptr = SMPmakeElt(matrix, here->first, here->second)) == NULL){\
    return(E_NOMEM);\
} } while(0)

            TSTALLOC(VSRCposIbrPtr, VSRCposNode, VSRCbranch);
            TSTALLOC(VSRCnegIbrPtr, VSRCnegNode, VSRCbranch);
            TSTALLOC(VSRCibrNegPtr, VSRCbranch, VSRCnegNode);
            TSTALLOC(VSRCibrPosPtr, VSRCbranch, VSRCposNode);
            TSTALLOC(VSRCibrIbrPtr, VSRCbranch, VSRCbranch);
        }
    }
    return(OK);
}
