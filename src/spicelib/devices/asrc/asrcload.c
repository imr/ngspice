/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1987 Kanwar Jit Singh
**********/
/*
 * singh@ic.Berkeley.edu
 */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "asrcdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

double *asrc_vals, *asrc_derivs;
int    asrc_nvals;

/*ARGSUSED*/
int
ASRCload(GENmodel *inModel, CKTcircuit *ckt)
{

    /* actually load the current voltage value into the
     * sparse matrix previously provided
     */

    ASRCmodel *model = (ASRCmodel*) inModel;
    ASRCinstance *here;
    int i, j;
    double rhs;
    double difference;
    double factor;

    /*  loop through all the Arbitrary source models */
    for( ; model != NULL; model = model->ASRCnextModel ) {

        /* loop through all the instances of the model */
        for (here = model->ASRCinstances; here != NULL ;
             here=here->ASRCnextInstance)
        {
            difference = (here->ASRCtemp + here->ASRCdtemp) - 300.15;
            factor = 1.0 + (here->ASRCtc1)*difference + (here->ASRCtc2)*difference*difference;
            if(here->ASRCreciproctc == 1) {
                factor = 1/factor;
            }

            /*
             * Get the function and its derivatives evaluated
             */
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
            for( i=0; i < here->ASRCtree->numVars; i++)
                if( here->ASRCtree->varTypes[i] == IF_INSTANCE) {
                    int branch = CKTfndBranch(ckt, here->ASRCtree->vars[i].uValue);
                    asrc_vals[i] = *(ckt->CKTrhsOld + branch);
                } else {
                    int node_num = (here->ASRCtree->vars[i].nValue) -> number;
                    asrc_vals[i] = *(ckt->CKTrhsOld + node_num);
                }

            if (here->ASRCtree->IFeval (here->ASRCtree, ckt->CKTgmin, &rhs, asrc_vals, asrc_derivs) != OK)
                return(E_BADPARM);

            /* The convergence test */
            here->ASRCprev_value = rhs;

            /* The ac load precomputation and storage */

            if (ckt->CKTmode & MODEINITSMSIG) {
                int size = (here->ASRCtree->numVars) + 1 ;
                here->ASRCacValues = NEWN(double, size);
                for ( i = 0; i < here->ASRCtree->numVars; i++)
                    here->ASRCacValues[i] = asrc_derivs[i];
            }

            if( here->ASRCtype == ASRC_VOLTAGE) {
                *(here->ASRCposptr[j++]) += 1.0;
                *(here->ASRCposptr[j++]) -= 1.0;
                *(here->ASRCposptr[j++]) -= 1.0;
                *(here->ASRCposptr[j++]) += 1.0;
            }

            for(i=0; i < here->ASRCtree->numVars; i++) {
                rhs -= (asrc_vals[i] * asrc_derivs[i]);

                switch(here->ASRCtree->varTypes[i]) {
                    case IF_INSTANCE:
                        if( here->ASRCtype == ASRC_VOLTAGE) {
                            /* CCVS */
                            *(here->ASRCposptr[j++]) -= asrc_derivs[i] * factor;
                        } else{
                            /* CCCS */
                            *(here->ASRCposptr[j++]) += asrc_derivs[i] * factor;
                            *(here->ASRCposptr[j++]) -= asrc_derivs[i] * factor;
                        }
                        break;

                    case IF_NODE:
                        if(here->ASRCtype == ASRC_VOLTAGE) {
                            /* VCVS */
                            *(here->ASRCposptr[j++]) -= asrc_derivs[i] * factor;
                        } else {
                            /* VCCS */
                            *(here->ASRCposptr[j++]) += asrc_derivs[i] * factor;
                            *(here->ASRCposptr[j++]) -= asrc_derivs[i] * factor;
                        }
                        break;

                    default:
                        return(E_BADPARM);
                }
            }

            /* Insert the RHS */
            if( here->ASRCtype == ASRC_VOLTAGE) {
                *(ckt->CKTrhs+(here->ASRCbranch)) += factor * rhs;
            } else {
                *(ckt->CKTrhs+(here->ASRCposNode)) -= factor * rhs;
                *(ckt->CKTrhs+(here->ASRCnegNode)) += factor * rhs;
            }

            /* Store the rhs for small signal analysis */
            if (ckt->CKTmode & MODEINITSMSIG) {
                here->ASRCacValues[here->ASRCtree->numVars] = factor * rhs;
            }
        }
    }

    return(OK);
}
