/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1987 Kanwar Jit Singh
**********/

#include "ngspice.h"
#include "cktdefs.h"
#include "asrcdefs.h"
#include "sperror.h"
#include "suffix.h"
#include "complex.h"

/*ARGSUSED*/
int
ASRCpzLoad(GENmodel *inModel, CKTcircuit *ckt, SPcomplex *s)

        /* actually load the current voltage value into the 
         * sparse matrix previously provided 
         */
{
    ASRCmodel *model = (ASRCmodel*)inModel;
    ASRCinstance *here;
    double value;
    int i, v_first, j, branch;
    int node_num;

    /*  loop through all the Arbitrary source models */
    for( ; model != NULL; model = model->ASRCnextModel ) {

        /* loop through all the instances of the model */
        for (here = model->ASRCinstances; here != NULL ;
                here=here->ASRCnextInstance)
	{
	    if (here->ASRCowner != ARCHme) continue;
	    j = 0;
            /* Get the function evaluated and the derivatives too */
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

            /* Fill the vector of values from the previous solution */
            for( i=0; i < here->ASRCtree->numVars; i++){
                if( here->ASRCtree->varTypes[i] == IF_INSTANCE){
                     branch = CKTfndBranch(ckt,here->ASRCtree->vars[i].uValue);
                     asrc_vals[i] = *(ckt->CKTrhsOld+branch);
                } else {
                    node_num = ((CKTnode *)(here->ASRCtree->vars[i].nValue))->
                            number;
                    asrc_vals[i] = *(ckt->CKTrhsOld+node_num);
                }
            }

            if( (*(here->ASRCtree->IFeval))(here->ASRCtree, ckt->CKTgmin,
		    &value, asrc_vals, asrc_derivs) == OK){
                for(i=0; i < here->ASRCtree->numVars; i++){
                    switch(here->ASRCtree->varTypes[i]){
                        case IF_INSTANCE:
                            if( here->ASRCtype == ASRC_VOLTAGE){
                                /* CCVS */
                                if(v_first){
                                    *(here->ASRCposptr[j++]) += 1.0;
                                    *(here->ASRCposptr[j++]) -= 1.0;
                                    *(here->ASRCposptr[j++]) -= 1.0;
                                    *(here->ASRCposptr[j++]) += 1.0;
                                    *(here->ASRCposptr[j++]) -= asrc_derivs[i];
                                    v_first = 0;
                                } else {
                                    *(here->ASRCposptr[j++]) -= asrc_derivs[i];
                                }
                            } else {
                                /* CCCS */
				*(here->ASRCposptr[j++]) += asrc_derivs[i];
				*(here->ASRCposptr[j++]) -= asrc_derivs[i];
                            }
                            break;
                        case IF_NODE:
                            if(here->ASRCtype == ASRC_VOLTAGE){
                                /* VCVS */
                                if( v_first){
                                    *(here->ASRCposptr[j++]) += 1.0;
                                    *(here->ASRCposptr[j++]) -= 1.0;
                                    *(here->ASRCposptr[j++]) -= 1.0;
                                    *(here->ASRCposptr[j++]) += 1.0;
                                    *(here->ASRCposptr[j++]) -= asrc_derivs[i];
                                    v_first = 0;
                                } else {
                                    *(here->ASRCposptr[j++]) -= asrc_derivs[i];
                                }
                            } else {
                /* VCCS */
                                *(here->ASRCposptr[j++]) += asrc_derivs[i];
                                *(here->ASRCposptr[j++]) -= asrc_derivs[i];
                            }
                            break;
                        default:
                            return(E_BADPARM);
                    }
                }
            } else {
                return(E_BADPARM);
            }
        }
    }
    return(OK);
}
