/**********
Author: 2012 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "mos1defs.h"
#include "ngspice/sperror.h"

int
MOS1bindCSC(GENmodel *inModel, CKTcircuit *ckt)
{
    MOS1model *model = (MOS1model *)inModel;
    int i ;

    /*  loop through all the mos1 models */
    for( ; model != NULL; model = model->MOS1nextModel ) {
	MOS1instance *here;

        /* loop through all the instances of the model */
        for (here = model->MOS1instances; here != NULL ;
	    here = here->MOS1nextInstance) {

		i = 0 ;
		if ((here->MOS1dNode != 0) && (here->MOS1dNode != 0)) {
			while (here->MOS1DdPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MOS1DdPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->MOS1gNode != 0) && (here->MOS1gNode != 0)) {
			while (here->MOS1GgPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MOS1GgPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->MOS1sNode != 0) && (here->MOS1sNode != 0)) {
			while (here->MOS1SsPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MOS1SsPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->MOS1bNode != 0) && (here->MOS1bNode != 0)) {
			while (here->MOS1BbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MOS1BbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->MOS1dNodePrime != 0) && (here->MOS1dNodePrime != 0)) {
			while (here->MOS1DPdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MOS1DPdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->MOS1sNodePrime != 0) && (here->MOS1sNodePrime != 0)) {
			while (here->MOS1SPspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MOS1SPspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->MOS1dNode != 0) && (here->MOS1dNodePrime != 0)) {
			while (here->MOS1DdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MOS1DdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->MOS1gNode != 0) && (here->MOS1bNode != 0)) {
			while (here->MOS1GbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MOS1GbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->MOS1gNode != 0) && (here->MOS1dNodePrime != 0)) {
			while (here->MOS1GdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MOS1GdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->MOS1gNode != 0) && (here->MOS1sNodePrime != 0)) {
			while (here->MOS1GspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MOS1GspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->MOS1sNode != 0) && (here->MOS1sNodePrime != 0)) {
			while (here->MOS1SspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MOS1SspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->MOS1bNode != 0) && (here->MOS1dNodePrime != 0)) {
			while (here->MOS1BdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MOS1BdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->MOS1bNode != 0) && (here->MOS1sNodePrime != 0)) {
			while (here->MOS1BspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MOS1BspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->MOS1dNodePrime != 0) && (here->MOS1sNodePrime != 0)) {
			while (here->MOS1DPspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MOS1DPspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->MOS1dNodePrime != 0) && (here->MOS1dNode != 0)) {
			while (here->MOS1DPdPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MOS1DPdPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->MOS1bNode != 0) && (here->MOS1gNode != 0)) {
			while (here->MOS1BgPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MOS1BgPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->MOS1dNodePrime != 0) && (here->MOS1gNode != 0)) {
			while (here->MOS1DPgPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MOS1DPgPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->MOS1sNodePrime != 0) && (here->MOS1gNode != 0)) {
			while (here->MOS1SPgPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MOS1SPgPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->MOS1sNodePrime != 0) && (here->MOS1sNode != 0)) {
			while (here->MOS1SPsPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MOS1SPsPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->MOS1dNodePrime != 0) && (here->MOS1bNode != 0)) {
			while (here->MOS1DPbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MOS1DPbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->MOS1sNodePrime != 0) && (here->MOS1bNode != 0)) {
			while (here->MOS1SPbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MOS1SPbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->MOS1sNodePrime != 0) && (here->MOS1dNodePrime != 0)) {
			while (here->MOS1SPdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MOS1SPdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
	}
    }
    return(OK);
}

int
MOS1bindCSCComplex(GENmodel *inModel, CKTcircuit *ckt)
{
    MOS1model *model = (MOS1model *)inModel;
    int i ;

    /*  loop through all the mos1 models */
    for( ; model != NULL; model = model->MOS1nextModel ) {
	MOS1instance *here;

        /* loop through all the instances of the model */
        for (here = model->MOS1instances; here != NULL ;
	    here = here->MOS1nextInstance) {

		i = 0 ;
		if ((here->MOS1dNode != 0) && (here->MOS1dNode != 0)) {
			while (here->MOS1DdPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MOS1DdPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->MOS1gNode != 0) && (here->MOS1gNode != 0)) {
			while (here->MOS1GgPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MOS1GgPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->MOS1sNode != 0) && (here->MOS1sNode != 0)) {
			while (here->MOS1SsPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MOS1SsPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->MOS1bNode != 0) && (here->MOS1bNode != 0)) {
			while (here->MOS1BbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MOS1BbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->MOS1dNodePrime != 0) && (here->MOS1dNodePrime != 0)) {
			while (here->MOS1DPdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MOS1DPdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->MOS1sNodePrime != 0) && (here->MOS1sNodePrime != 0)) {
			while (here->MOS1SPspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MOS1SPspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->MOS1dNode != 0) && (here->MOS1dNodePrime != 0)) {
			while (here->MOS1DdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MOS1DdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->MOS1gNode != 0) && (here->MOS1bNode != 0)) {
			while (here->MOS1GbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MOS1GbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->MOS1gNode != 0) && (here->MOS1dNodePrime != 0)) {
			while (here->MOS1GdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MOS1GdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->MOS1gNode != 0) && (here->MOS1sNodePrime != 0)) {
			while (here->MOS1GspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MOS1GspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->MOS1sNode != 0) && (here->MOS1sNodePrime != 0)) {
			while (here->MOS1SspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MOS1SspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->MOS1bNode != 0) && (here->MOS1dNodePrime != 0)) {
			while (here->MOS1BdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MOS1BdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->MOS1bNode != 0) && (here->MOS1sNodePrime != 0)) {
			while (here->MOS1BspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MOS1BspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->MOS1dNodePrime != 0) && (here->MOS1sNodePrime != 0)) {
			while (here->MOS1DPspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MOS1DPspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->MOS1dNodePrime != 0) && (here->MOS1dNode != 0)) {
			while (here->MOS1DPdPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MOS1DPdPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->MOS1bNode != 0) && (here->MOS1gNode != 0)) {
			while (here->MOS1BgPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MOS1BgPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->MOS1dNodePrime != 0) && (here->MOS1gNode != 0)) {
			while (here->MOS1DPgPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MOS1DPgPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->MOS1sNodePrime != 0) && (here->MOS1gNode != 0)) {
			while (here->MOS1SPgPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MOS1SPgPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->MOS1sNodePrime != 0) && (here->MOS1sNode != 0)) {
			while (here->MOS1SPsPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MOS1SPsPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->MOS1dNodePrime != 0) && (here->MOS1bNode != 0)) {
			while (here->MOS1DPbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MOS1DPbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->MOS1sNodePrime != 0) && (here->MOS1bNode != 0)) {
			while (here->MOS1SPbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MOS1SPbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->MOS1sNodePrime != 0) && (here->MOS1dNodePrime != 0)) {
			while (here->MOS1SPdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MOS1SPdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
	}
    }
    return(OK);
}
