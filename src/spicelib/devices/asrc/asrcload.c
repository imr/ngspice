/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1987 Kanwar Jit Singh
**********/
/*
 * singh@ic.Berkeley.edu
 */

#include "ngspice.h"
#include "cktdefs.h"
#include "asrcdefs.h"
#include "sperror.h"
#include "suffix.h"

double *asrc_vals, *asrc_derivs;
int	asrc_nvals;

/*ARGSUSED*/
int
ASRCload(GENmodel *inModel, CKTcircuit *ckt)
{

    /* actually load the current voltage value into the 
     * sparse matrix previously provided 
     */

    ASRCmodel *model = (ASRCmodel*)inModel;
    ASRCinstance *here;
    int i, v_first, j, branch;
    int node_num;
    int size;
    double rhs;

    /*  loop through all the Arbitrary source models */
    for( ; model != NULL; model = model->ASRCnextModel ) {

        /* loop through all the instances of the model */
        for (here = model->ASRCinstances; here != NULL ;
	     here=here->ASRCnextInstance)
	{
	    if (here->ASRCowner != ARCHme) continue;
            
	    /*
	     * Get the function and its derivatives evaluated 
	     */
	    v_first = 1;
	    i = here->ASRCtree->numVars;
	    if (asrc_nvals < i) {
		if (asrc_nvals) {
		    FREE(asrc_vals);
		    FREE(asrc_derivs);
		}
		asrc_nvals = i;
		asrc_vals = NEWN(double, i);
		asrc_derivs = NEWN(double, i);
	    }

	    j=0;

	    /*
	     * Fill the vector of values from the previous solution
	     */
	    for( i=0; i < here->ASRCtree->numVars; i++){
		if( here->ASRCtree->varTypes[i] == IF_INSTANCE){
		    branch = CKTfndBranch(ckt,
					  here->ASRCtree->vars[i].uValue);
		    asrc_vals[i] = *(ckt->CKTrhsOld+branch);
		} else {
		    node_num = ((CKTnode *)(here->ASRCtree->vars[i].
					    nValue))->number;
		    asrc_vals[i] = *(ckt->CKTrhsOld+node_num);
		}
	    }

	    if ((*(here->ASRCtree->IFeval))(here->ASRCtree,ckt->CKTgmin, &rhs,
					    asrc_vals,asrc_derivs) == OK)
	    {

		/* The convergence test */
		here->ASRCprev_value = rhs;

		/* The ac load precomputation and storage */

		if (ckt->CKTmode & MODEINITSMSIG) {
		    size = (here->ASRCtree->numVars)+1 ;
		    here->ASRCacValues = NEWN(double, size);
		    for ( i = 0; i < here->ASRCtree->numVars; i++){
			here->ASRCacValues[i] = asrc_derivs[i]; 
		    }
		}

		for(i=0; i < here->ASRCtree->numVars; i++) {
		    rhs -= (asrc_vals[i] * asrc_derivs[i]);
		    switch(here->ASRCtree->varTypes[i]){
		    case IF_INSTANCE:
			if( here->ASRCtype == ASRC_VOLTAGE){
			    /* CCVS */
			    if(v_first){
				*(here->ASRCposptr[j++]) += 1.0;
				*(here->ASRCposptr[j++]) -= 1.0;
				*(here->ASRCposptr[j++]) -= 1.0;
				*(here->ASRCposptr[j++]) += 1.0;
				v_first = 0;
			    }
			    *(here->ASRCposptr[j++]) -= asrc_derivs[i];
			} else{
			    /* CCCS */
			    *(here->ASRCposptr[j++]) += asrc_derivs[i];
			    *(here->ASRCposptr[j++]) -= asrc_derivs[i];
			}
			break;

		    case IF_NODE:
			if(here->ASRCtype == ASRC_VOLTAGE) {
			    /* VCVS */
			    if( v_first){
				*(here->ASRCposptr[j++]) += 1.0;
				*(here->ASRCposptr[j++]) -= 1.0;
				*(here->ASRCposptr[j++]) -= 1.0;
				*(here->ASRCposptr[j++]) += 1.0;
				v_first = 0;
			    }
			    *(here->ASRCposptr[j++]) -= asrc_derivs[i];
			} else {
			    /*VCCS*/
			    *(here->ASRCposptr[j++]) += asrc_derivs[i];
			    *(here->ASRCposptr[j++]) -= asrc_derivs[i];
			}
			break;

		    default:
			return(E_BADPARM);
		    }
		}

		/* Insert the RHS */
		if( here->ASRCtype == ASRC_VOLTAGE){
		    *(ckt->CKTrhs+(here->ASRCbranch)) += rhs;
		} else {
		    *(ckt->CKTrhs+(here->ASRCposNode)) -= rhs;
		    *(ckt->CKTrhs+(here->ASRCnegNode)) += rhs;
		}

		/* Store the rhs for small signal analysis */
		if (ckt->CKTmode & MODEINITSMSIG){
		    here->ASRCacValues[here->ASRCtree->numVars] = rhs; 
		}
	    } else{
		return(E_BADPARM);
	    }
	}
    }
    return(OK);
}
