/**********
Author: 2012 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "hsm2def.h"
#include "ngspice/sperror.h"

int
HSM2bindCSC(GENmodel *inModel, CKTcircuit *ckt)
{
    HSM2model *model = (HSM2model *)inModel;
    int i ;

    /*  loop through all the hsm2 models */
    for( ; model != NULL; model = model->HSM2nextModel ) {
	HSM2instance *here;

        /* loop through all the instances of the model */
        for (here = model->HSM2instances; here != NULL ;
	    here = here->HSM2nextInstance) {

		i = 0 ;
		if ((here-> HSM2dNodePrime != 0) && (here-> HSM2bNodePrime != 0)) {
			while (here->HSM2DPbpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HSM2DPbpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> HSM2sNodePrime != 0) && (here-> HSM2bNodePrime != 0)) {
			while (here->HSM2SPbpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HSM2SPbpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> HSM2gNodePrime != 0) && (here-> HSM2bNodePrime != 0)) {
			while (here->HSM2GPbpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HSM2GPbpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> HSM2bNodePrime != 0) && (here-> HSM2dNodePrime != 0)) {
			while (here->HSM2BPdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HSM2BPdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> HSM2bNodePrime != 0) && (here-> HSM2sNodePrime != 0)) {
			while (here->HSM2BPspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HSM2BPspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> HSM2bNodePrime != 0) && (here-> HSM2gNodePrime != 0)) {
			while (here->HSM2BPgpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HSM2BPgpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> HSM2bNodePrime != 0) && (here-> HSM2bNodePrime != 0)) {
			while (here->HSM2BPbpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HSM2BPbpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> HSM2dNode != 0) && (here-> HSM2dNode != 0)) {
			while (here->HSM2DdPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HSM2DdPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> HSM2gNodePrime != 0) && (here-> HSM2gNodePrime != 0)) {
			while (here->HSM2GPgpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HSM2GPgpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> HSM2sNode != 0) && (here-> HSM2sNode != 0)) {
			while (here->HSM2SsPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HSM2SsPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> HSM2dNodePrime != 0) && (here-> HSM2dNodePrime != 0)) {
			while (here->HSM2DPdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HSM2DPdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> HSM2sNodePrime != 0) && (here-> HSM2sNodePrime != 0)) {
			while (here->HSM2SPspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HSM2SPspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> HSM2dNode != 0) && (here-> HSM2dNodePrime != 0)) {
			while (here->HSM2DdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HSM2DdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> HSM2gNodePrime != 0) && (here-> HSM2dNodePrime != 0)) {
			while (here->HSM2GPdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HSM2GPdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> HSM2gNodePrime != 0) && (here-> HSM2sNodePrime != 0)) {
			while (here->HSM2GPspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HSM2GPspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> HSM2sNode != 0) && (here-> HSM2sNodePrime != 0)) {
			while (here->HSM2SspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HSM2SspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> HSM2dNodePrime != 0) && (here-> HSM2sNodePrime != 0)) {
			while (here->HSM2DPspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HSM2DPspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> HSM2dNodePrime != 0) && (here-> HSM2dNode != 0)) {
			while (here->HSM2DPdPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HSM2DPdPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> HSM2dNodePrime != 0) && (here-> HSM2gNodePrime != 0)) {
			while (here->HSM2DPgpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HSM2DPgpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> HSM2sNodePrime != 0) && (here-> HSM2gNodePrime != 0)) {
			while (here->HSM2SPgpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HSM2SPgpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> HSM2sNodePrime != 0) && (here-> HSM2sNode != 0)) {
			while (here->HSM2SPsPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HSM2SPsPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> HSM2sNodePrime != 0) && (here-> HSM2dNodePrime != 0)) {
			while (here->HSM2SPdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HSM2SPdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

                if ( here->HSM2_corg == 1 ) {

		i = 0 ;
		if ((here-> HSM2gNode != 0) && (here-> HSM2gNode != 0)) {
			while (here->HSM2GgPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HSM2GgPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> HSM2gNode != 0) && (here-> HSM2gNodePrime != 0)) {
			while (here->HSM2GgpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HSM2GgpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> HSM2gNodePrime != 0) && (here-> HSM2gNode != 0)) {
			while (here->HSM2GPgPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HSM2GPgPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> HSM2gNode != 0) && (here-> HSM2dNodePrime != 0)) {
			while (here->HSM2GdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HSM2GdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> HSM2gNode != 0) && (here-> HSM2sNodePrime != 0)) {
			while (here->HSM2GspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HSM2GspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> HSM2gNode != 0) && (here-> HSM2bNodePrime != 0)) {
			while (here->HSM2GbpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HSM2GbpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
                } /* IF */

                if ( here->HSM2_corbnet == 1 ) { /* consider body resistance net */

		i = 0 ;
		if ((here-> HSM2dNodePrime != 0) && (here-> HSM2dbNode != 0)) {
			while (here->HSM2DPdbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HSM2DPdbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> HSM2sNodePrime != 0) && (here-> HSM2sbNode != 0)) {
			while (here->HSM2SPsbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HSM2SPsbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> HSM2dbNode != 0) && (here-> HSM2dNodePrime != 0)) {
			while (here->HSM2DBdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HSM2DBdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> HSM2dbNode != 0) && (here-> HSM2dbNode != 0)) {
			while (here->HSM2DBdbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HSM2DBdbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> HSM2dbNode != 0) && (here-> HSM2bNodePrime != 0)) {
			while (here->HSM2DBbpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HSM2DBbpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> HSM2dbNode != 0) && (here-> HSM2bNode != 0)) {
			while (here->HSM2DBbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HSM2DBbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> HSM2bNodePrime != 0) && (here-> HSM2dbNode != 0)) {
			while (here->HSM2BPdbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HSM2BPdbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> HSM2bNodePrime != 0) && (here-> HSM2bNode != 0)) {
			while (here->HSM2BPbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HSM2BPbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> HSM2bNodePrime != 0) && (here-> HSM2sbNode != 0)) {
			while (here->HSM2BPsbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HSM2BPsbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> HSM2sbNode != 0) && (here-> HSM2sNodePrime != 0)) {
			while (here->HSM2SBspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HSM2SBspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> HSM2sbNode != 0) && (here-> HSM2bNodePrime != 0)) {
			while (here->HSM2SBbpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HSM2SBbpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> HSM2sbNode != 0) && (here-> HSM2bNode != 0)) {
			while (here->HSM2SBbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HSM2SBbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> HSM2sbNode != 0) && (here-> HSM2sbNode != 0)) {
			while (here->HSM2SBsbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HSM2SBsbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> HSM2bNode != 0) && (here-> HSM2dbNode != 0)) {
			while (here->HSM2BdbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HSM2BdbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> HSM2bNode != 0) && (here-> HSM2bNodePrime != 0)) {
			while (here->HSM2BbpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HSM2BbpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> HSM2bNode != 0) && (here-> HSM2sbNode != 0)) {
			while (here->HSM2BsbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HSM2BsbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> HSM2bNode != 0) && (here-> HSM2bNode != 0)) {
			while (here->HSM2BbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HSM2BbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
                } /* IF */
	}
    }
    return(OK);
}

int
HSM2bindCSCComplex(GENmodel *inModel, CKTcircuit *ckt)
{
    HSM2model *model = (HSM2model *)inModel;
    int i ;

    /*  loop through all the hsm2 models */
    for( ; model != NULL; model = model->HSM2nextModel ) {
	HSM2instance *here;

        /* loop through all the instances of the model */
        for (here = model->HSM2instances; here != NULL ;
	    here = here->HSM2nextInstance) {

		i = 0 ;
		if ((here-> HSM2dNodePrime != 0) && (here-> HSM2bNodePrime != 0)) {
			while (here->HSM2DPbpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HSM2DPbpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> HSM2sNodePrime != 0) && (here-> HSM2bNodePrime != 0)) {
			while (here->HSM2SPbpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HSM2SPbpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> HSM2gNodePrime != 0) && (here-> HSM2bNodePrime != 0)) {
			while (here->HSM2GPbpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HSM2GPbpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> HSM2bNodePrime != 0) && (here-> HSM2dNodePrime != 0)) {
			while (here->HSM2BPdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HSM2BPdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> HSM2bNodePrime != 0) && (here-> HSM2sNodePrime != 0)) {
			while (here->HSM2BPspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HSM2BPspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> HSM2bNodePrime != 0) && (here-> HSM2gNodePrime != 0)) {
			while (here->HSM2BPgpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HSM2BPgpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> HSM2bNodePrime != 0) && (here-> HSM2bNodePrime != 0)) {
			while (here->HSM2BPbpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HSM2BPbpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> HSM2dNode != 0) && (here-> HSM2dNode != 0)) {
			while (here->HSM2DdPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HSM2DdPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> HSM2gNodePrime != 0) && (here-> HSM2gNodePrime != 0)) {
			while (here->HSM2GPgpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HSM2GPgpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> HSM2sNode != 0) && (here-> HSM2sNode != 0)) {
			while (here->HSM2SsPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HSM2SsPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> HSM2dNodePrime != 0) && (here-> HSM2dNodePrime != 0)) {
			while (here->HSM2DPdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HSM2DPdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> HSM2sNodePrime != 0) && (here-> HSM2sNodePrime != 0)) {
			while (here->HSM2SPspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HSM2SPspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> HSM2dNode != 0) && (here-> HSM2dNodePrime != 0)) {
			while (here->HSM2DdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HSM2DdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> HSM2gNodePrime != 0) && (here-> HSM2dNodePrime != 0)) {
			while (here->HSM2GPdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HSM2GPdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> HSM2gNodePrime != 0) && (here-> HSM2sNodePrime != 0)) {
			while (here->HSM2GPspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HSM2GPspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> HSM2sNode != 0) && (here-> HSM2sNodePrime != 0)) {
			while (here->HSM2SspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HSM2SspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> HSM2dNodePrime != 0) && (here-> HSM2sNodePrime != 0)) {
			while (here->HSM2DPspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HSM2DPspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> HSM2dNodePrime != 0) && (here-> HSM2dNode != 0)) {
			while (here->HSM2DPdPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HSM2DPdPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> HSM2dNodePrime != 0) && (here-> HSM2gNodePrime != 0)) {
			while (here->HSM2DPgpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HSM2DPgpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> HSM2sNodePrime != 0) && (here-> HSM2gNodePrime != 0)) {
			while (here->HSM2SPgpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HSM2SPgpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> HSM2sNodePrime != 0) && (here-> HSM2sNode != 0)) {
			while (here->HSM2SPsPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HSM2SPsPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> HSM2sNodePrime != 0) && (here-> HSM2dNodePrime != 0)) {
			while (here->HSM2SPdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HSM2SPdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

                if ( here->HSM2_corg == 1 ) {

		i = 0 ;
		if ((here-> HSM2gNode != 0) && (here-> HSM2gNode != 0)) {
			while (here->HSM2GgPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HSM2GgPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> HSM2gNode != 0) && (here-> HSM2gNodePrime != 0)) {
			while (here->HSM2GgpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HSM2GgpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> HSM2gNodePrime != 0) && (here-> HSM2gNode != 0)) {
			while (here->HSM2GPgPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HSM2GPgPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> HSM2gNode != 0) && (here-> HSM2dNodePrime != 0)) {
			while (here->HSM2GdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HSM2GdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> HSM2gNode != 0) && (here-> HSM2sNodePrime != 0)) {
			while (here->HSM2GspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HSM2GspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> HSM2gNode != 0) && (here-> HSM2bNodePrime != 0)) {
			while (here->HSM2GbpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HSM2GbpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
                } /* IF */

                if ( here->HSM2_corbnet == 1 ) { /* consider body resistance net */

		i = 0 ;
		if ((here-> HSM2dNodePrime != 0) && (here-> HSM2dbNode != 0)) {
			while (here->HSM2DPdbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HSM2DPdbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> HSM2sNodePrime != 0) && (here-> HSM2sbNode != 0)) {
			while (here->HSM2SPsbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HSM2SPsbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> HSM2dbNode != 0) && (here-> HSM2dNodePrime != 0)) {
			while (here->HSM2DBdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HSM2DBdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> HSM2dbNode != 0) && (here-> HSM2dbNode != 0)) {
			while (here->HSM2DBdbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HSM2DBdbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> HSM2dbNode != 0) && (here-> HSM2bNodePrime != 0)) {
			while (here->HSM2DBbpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HSM2DBbpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> HSM2dbNode != 0) && (here-> HSM2bNode != 0)) {
			while (here->HSM2DBbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HSM2DBbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> HSM2bNodePrime != 0) && (here-> HSM2dbNode != 0)) {
			while (here->HSM2BPdbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HSM2BPdbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> HSM2bNodePrime != 0) && (here-> HSM2bNode != 0)) {
			while (here->HSM2BPbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HSM2BPbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> HSM2bNodePrime != 0) && (here-> HSM2sbNode != 0)) {
			while (here->HSM2BPsbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HSM2BPsbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> HSM2sbNode != 0) && (here-> HSM2sNodePrime != 0)) {
			while (here->HSM2SBspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HSM2SBspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> HSM2sbNode != 0) && (here-> HSM2bNodePrime != 0)) {
			while (here->HSM2SBbpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HSM2SBbpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> HSM2sbNode != 0) && (here-> HSM2bNode != 0)) {
			while (here->HSM2SBbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HSM2SBbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> HSM2sbNode != 0) && (here-> HSM2sbNode != 0)) {
			while (here->HSM2SBsbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HSM2SBsbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> HSM2bNode != 0) && (here-> HSM2dbNode != 0)) {
			while (here->HSM2BdbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HSM2BdbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> HSM2bNode != 0) && (here-> HSM2bNodePrime != 0)) {
			while (here->HSM2BbpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HSM2BbpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> HSM2bNode != 0) && (here-> HSM2sbNode != 0)) {
			while (here->HSM2BsbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HSM2BsbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> HSM2bNode != 0) && (here-> HSM2bNode != 0)) {
			while (here->HSM2BbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HSM2BbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
                } /* IF */
	}
    }
    return(OK);
}
