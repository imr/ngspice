/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1987 Kanwar Jit Singh
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "asrcdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"
#include "ngspice/complex.h"

/*ARGSUSED*/
int
ASRCpzLoad(GENmodel *inModel, CKTcircuit *ckt, SPcomplex *s)

        /* actually load the current voltage value into the
         * sparse matrix previously provided
         */
{
    ASRCmodel *model = (ASRCmodel*) inModel;
    ASRCinstance *here;
    double value;
    int i, j;
    double difference;
    double factor;

    NG_IGNORE(s);

    /*  loop through all the Arbitrary source models */
    for( ; model != NULL; model = model->ASRCnextModel ) {

        /* loop through all the instances of the model */
        for (here = model->ASRCinstances; here != NULL ;
                here=here->ASRCnextInstance)
        {
	    difference = (here->ASRCtemp + here->ASRCdtemp) - 300.15;
	    factor = 1.0 + (here->ASRCtc1)*difference + 
		    (here->ASRCtc2)*difference*difference;
	    if(here->ASRCreciproctc == 1) {
		    factor = 1/factor;
		  }

            j = 0;

            /* Get the function evaluated and the derivatives too */
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
            for( i=0; i < here->ASRCtree->numVars; i++) {
                if( here->ASRCtree->varTypes[i] == IF_INSTANCE) {
                     int branch = CKTfndBranch(ckt,here->ASRCtree->vars[i].uValue);
                     asrc_vals[i] = *(ckt->CKTrhsOld + branch);
                } else {
                    int node_num = (here->ASRCtree->vars[i].nValue) -> number;
                    asrc_vals[i] = *(ckt->CKTrhsOld + node_num);
                }
            }

            if(here->ASRCtree->IFeval (here->ASRCtree, ckt->CKTgmin, &value, asrc_vals, asrc_derivs) != OK)
                return(E_BADPARM);

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
                        *(here->ASRCposptr[j++]) -= asrc_derivs[i] / factor;
                    } else {
                        /* CCCS */
                        *(here->ASRCposptr[j++]) += asrc_derivs[i] / factor;
                        *(here->ASRCposptr[j++]) -= asrc_derivs[i] / factor;
                    }
                    break;
                case IF_NODE:
                    if(here->ASRCtype == ASRC_VOLTAGE) {
                        /* VCVS */
                        *(here->ASRCposptr[j++]) -= asrc_derivs[i] / factor;
                    } else {
                        /* VCCS */
                        *(here->ASRCposptr[j++]) += asrc_derivs[i] / factor;
                        *(here->ASRCposptr[j++]) -= asrc_derivs[i] / factor;
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
