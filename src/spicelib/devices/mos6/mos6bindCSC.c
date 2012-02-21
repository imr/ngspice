/**********
Author: 2012 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "mos6defs.h"
#include "ngspice/sperror.h"

int
MOS6bindCSC(GENmodel *inModel, CKTcircuit *ckt)
{
    MOS6model *model = (MOS6model *)inModel;
    int i ;

    /*  loop through all the mos6 models */
    for( ; model != NULL; model = model->MOS6nextModel ) {
	MOS6instance *here;

        /* loop through all the instances of the model */
        for (here = model->MOS6instances; here != NULL ;
	    here = here->MOS6nextInstance) {

		i = 0 ;
		if ((here->MOS6dNode != 0) && (here->MOS6dNode != 0)) {
			while (here->MOS6DdPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MOS6DdPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->MOS6gNode != 0) && (here->MOS6gNode != 0)) {
			while (here->MOS6GgPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MOS6GgPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->MOS6sNode != 0) && (here->MOS6sNode != 0)) {
			while (here->MOS6SsPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MOS6SsPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->MOS6bNode != 0) && (here->MOS6bNode != 0)) {
			while (here->MOS6BbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MOS6BbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->MOS6dNodePrime != 0) && (here->MOS6dNodePrime != 0)) {
			while (here->MOS6DPdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MOS6DPdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->MOS6sNodePrime != 0) && (here->MOS6sNodePrime != 0)) {
			while (here->MOS6SPspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MOS6SPspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->MOS6dNode != 0) && (here->MOS6dNodePrime != 0)) {
			while (here->MOS6DdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MOS6DdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->MOS6gNode != 0) && (here->MOS6bNode != 0)) {
			while (here->MOS6GbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MOS6GbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->MOS6gNode != 0) && (here->MOS6dNodePrime != 0)) {
			while (here->MOS6GdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MOS6GdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->MOS6gNode != 0) && (here->MOS6sNodePrime != 0)) {
			while (here->MOS6GspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MOS6GspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->MOS6sNode != 0) && (here->MOS6sNodePrime != 0)) {
			while (here->MOS6SspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MOS6SspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->MOS6bNode != 0) && (here->MOS6dNodePrime != 0)) {
			while (here->MOS6BdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MOS6BdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->MOS6bNode != 0) && (here->MOS6sNodePrime != 0)) {
			while (here->MOS6BspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MOS6BspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->MOS6dNodePrime != 0) && (here->MOS6sNodePrime != 0)) {
			while (here->MOS6DPspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MOS6DPspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->MOS6dNodePrime != 0) && (here->MOS6dNode != 0)) {
			while (here->MOS6DPdPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MOS6DPdPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->MOS6bNode != 0) && (here->MOS6gNode != 0)) {
			while (here->MOS6BgPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MOS6BgPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->MOS6dNodePrime != 0) && (here->MOS6gNode != 0)) {
			while (here->MOS6DPgPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MOS6DPgPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->MOS6sNodePrime != 0) && (here->MOS6gNode != 0)) {
			while (here->MOS6SPgPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MOS6SPgPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->MOS6sNodePrime != 0) && (here->MOS6sNode != 0)) {
			while (here->MOS6SPsPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MOS6SPsPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->MOS6dNodePrime != 0) && (here->MOS6bNode != 0)) {
			while (here->MOS6DPbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MOS6DPbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->MOS6sNodePrime != 0) && (here->MOS6bNode != 0)) {
			while (here->MOS6SPbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MOS6SPbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->MOS6sNodePrime != 0) && (here->MOS6dNodePrime != 0)) {
			while (here->MOS6SPdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MOS6SPdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
	}
    }
    return(OK);
}

int
MOS6bindCSCComplex(GENmodel *inModel, CKTcircuit *ckt)
{
    MOS6model *model = (MOS6model *)inModel;
    int i ;

    /*  loop through all the mos6 models */
    for( ; model != NULL; model = model->MOS6nextModel ) {
	MOS6instance *here;

        /* loop through all the instances of the model */
        for (here = model->MOS6instances; here != NULL ;
	    here = here->MOS6nextInstance) {

		i = 0 ;
		if ((here->MOS6dNode != 0) && (here->MOS6dNode != 0)) {
			while (here->MOS6DdPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MOS6DdPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->MOS6gNode != 0) && (here->MOS6gNode != 0)) {
			while (here->MOS6GgPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MOS6GgPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->MOS6sNode != 0) && (here->MOS6sNode != 0)) {
			while (here->MOS6SsPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MOS6SsPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->MOS6bNode != 0) && (here->MOS6bNode != 0)) {
			while (here->MOS6BbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MOS6BbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->MOS6dNodePrime != 0) && (here->MOS6dNodePrime != 0)) {
			while (here->MOS6DPdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MOS6DPdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->MOS6sNodePrime != 0) && (here->MOS6sNodePrime != 0)) {
			while (here->MOS6SPspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MOS6SPspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->MOS6dNode != 0) && (here->MOS6dNodePrime != 0)) {
			while (here->MOS6DdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MOS6DdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->MOS6gNode != 0) && (here->MOS6bNode != 0)) {
			while (here->MOS6GbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MOS6GbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->MOS6gNode != 0) && (here->MOS6dNodePrime != 0)) {
			while (here->MOS6GdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MOS6GdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->MOS6gNode != 0) && (here->MOS6sNodePrime != 0)) {
			while (here->MOS6GspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MOS6GspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->MOS6sNode != 0) && (here->MOS6sNodePrime != 0)) {
			while (here->MOS6SspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MOS6SspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->MOS6bNode != 0) && (here->MOS6dNodePrime != 0)) {
			while (here->MOS6BdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MOS6BdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->MOS6bNode != 0) && (here->MOS6sNodePrime != 0)) {
			while (here->MOS6BspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MOS6BspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->MOS6dNodePrime != 0) && (here->MOS6sNodePrime != 0)) {
			while (here->MOS6DPspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MOS6DPspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->MOS6dNodePrime != 0) && (here->MOS6dNode != 0)) {
			while (here->MOS6DPdPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MOS6DPdPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->MOS6bNode != 0) && (here->MOS6gNode != 0)) {
			while (here->MOS6BgPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MOS6BgPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->MOS6dNodePrime != 0) && (here->MOS6gNode != 0)) {
			while (here->MOS6DPgPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MOS6DPgPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->MOS6sNodePrime != 0) && (here->MOS6gNode != 0)) {
			while (here->MOS6SPgPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MOS6SPgPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->MOS6sNodePrime != 0) && (here->MOS6sNode != 0)) {
			while (here->MOS6SPsPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MOS6SPsPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->MOS6dNodePrime != 0) && (here->MOS6bNode != 0)) {
			while (here->MOS6DPbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MOS6DPbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->MOS6sNodePrime != 0) && (here->MOS6bNode != 0)) {
			while (here->MOS6SPbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MOS6SPbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->MOS6sNodePrime != 0) && (here->MOS6dNodePrime != 0)) {
			while (here->MOS6SPdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MOS6SPdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
	}
    }
    return(OK);
}
