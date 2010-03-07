/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1988 Kanwar Jit Singh
**********/
/*
 * singh@ic.Berkeley.edu
 */

#include "ngspice.h"
#include "cktdefs.h"
#include "asrcdefs.h"
#include "sperror.h"
#include "suffix.h"


/*ARGSUSED*/
int
ASRCacLoad(GENmodel *inModel, CKTcircuit *ckt)
{

    /*
     * Actually load the current voltage value into the
     * sparse matrix previously provided. The values have
     * been precomputed and stored with the instance model.
     */

    ASRCmodel *model = (ASRCmodel*) inModel;
    ASRCinstance *here;
    int i, j;
    double *derivs;
    double rhs;

    /*  loop through all the Arbitrary source models */
    for( ; model != NULL; model = model->ASRCnextModel ) {

        /* loop through all the instances of the model */
        for (here = model->ASRCinstances; here != NULL ;
	     here = here->ASRCnextInstance) {

	    if (here->ASRCowner != ARCHme)
                continue;

	    /*
	     * Get the function and its derivatives from the
	     * field in the instance structure. The field is
	     * an array of doubles holding the rhs, and the
	     * entries of the jacobian.
	     */

	    j=0;
	    derivs = here->ASRCacValues;
	    rhs = (here->ASRCacValues)[here->ASRCtree->numVars];

            if( here->ASRCtype == ASRC_VOLTAGE) {
                *(here->ASRCposptr[j++]) += 1.0;
                *(here->ASRCposptr[j++]) -= 1.0;
                *(here->ASRCposptr[j++]) -= 1.0;
                *(here->ASRCposptr[j++]) += 1.0;
            }

	    for(i=0; i < here->ASRCtree->numVars; i++) {
		switch(here->ASRCtree->varTypes[i]) {
		case IF_INSTANCE:
		    if( here->ASRCtype == ASRC_VOLTAGE) {
			/* CCVS */
			*(here->ASRCposptr[j++]) -= derivs[i];
		    } else{
			/* CCCS */
			*(here->ASRCposptr[j++]) += derivs[i];
			*(here->ASRCposptr[j++]) -= derivs[i];
		    }
		    break;
		case IF_NODE:
		    if(here->ASRCtype == ASRC_VOLTAGE) {
			/* VCVS */
			*(here->ASRCposptr[j++]) -= derivs[i];
		    } else {
			/*VCCS*/
			*(here->ASRCposptr[j++]) += derivs[i];
			*(here->ASRCposptr[j++]) -= derivs[i];
		    }
		    break;
		default:
		    return(E_BADPARM);
		}
	    }
	}
    }

    return(OK);
}
