/**********
Author: 2012 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "bsim2def.h"
#include "ngspice/sperror.h"

int
B2bindCSC(GENmodel *inModel, CKTcircuit *ckt)
{
    B2model *model = (B2model *)inModel;
    int i ;

    /*  loop through all the b2 models */
    for( ; model != NULL; model = model->B2nextModel ) {
	B2instance *here;

        /* loop through all the instances of the model */
        for (here = model->B2instances; here != NULL ;
	    here = here->B2nextInstance) {

		i = 0 ;
		if ((here-> B2dNode != 0) && (here-> B2dNode != 0)) {
			while (here->B2DdPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B2DdPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B2gNode != 0) && (here-> B2gNode != 0)) {
			while (here->B2GgPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B2GgPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B2sNode != 0) && (here-> B2sNode != 0)) {
			while (here->B2SsPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B2SsPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B2bNode != 0) && (here-> B2bNode != 0)) {
			while (here->B2BbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B2BbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B2dNodePrime != 0) && (here-> B2dNodePrime != 0)) {
			while (here->B2DPdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B2DPdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B2sNodePrime != 0) && (here-> B2sNodePrime != 0)) {
			while (here->B2SPspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B2SPspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B2dNode != 0) && (here-> B2dNodePrime != 0)) {
			while (here->B2DdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B2DdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B2gNode != 0) && (here-> B2bNode != 0)) {
			while (here->B2GbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B2GbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B2gNode != 0) && (here-> B2dNodePrime != 0)) {
			while (here->B2GdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B2GdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B2gNode != 0) && (here-> B2sNodePrime != 0)) {
			while (here->B2GspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B2GspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B2sNode != 0) && (here-> B2sNodePrime != 0)) {
			while (here->B2SspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B2SspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B2bNode != 0) && (here-> B2dNodePrime != 0)) {
			while (here->B2BdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B2BdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B2bNode != 0) && (here-> B2sNodePrime != 0)) {
			while (here->B2BspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B2BspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B2dNodePrime != 0) && (here-> B2sNodePrime != 0)) {
			while (here->B2DPspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B2DPspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B2dNodePrime != 0) && (here-> B2dNode != 0)) {
			while (here->B2DPdPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B2DPdPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B2bNode != 0) && (here-> B2gNode != 0)) {
			while (here->B2BgPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B2BgPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B2dNodePrime != 0) && (here-> B2gNode != 0)) {
			while (here->B2DPgPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B2DPgPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B2sNodePrime != 0) && (here-> B2gNode != 0)) {
			while (here->B2SPgPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B2SPgPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B2sNodePrime != 0) && (here-> B2sNode != 0)) {
			while (here->B2SPsPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B2SPsPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B2dNodePrime != 0) && (here-> B2bNode != 0)) {
			while (here->B2DPbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B2DPbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B2sNodePrime != 0) && (here-> B2bNode != 0)) {
			while (here->B2SPbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B2SPbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B2sNodePrime != 0) && (here-> B2dNodePrime != 0)) {
			while (here->B2SPdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B2SPdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
	}
    }
    return(OK);
}

int
B2bindCSCComplex(GENmodel *inModel, CKTcircuit *ckt)
{
    B2model *model = (B2model *)inModel;
    int i ;

    /*  loop through all the b2 models */
    for( ; model != NULL; model = model->B2nextModel ) {
	B2instance *here;

        /* loop through all the instances of the model */
        for (here = model->B2instances; here != NULL ;
	    here = here->B2nextInstance) {

		i = 0 ;
		if ((here-> B2dNode != 0) && (here-> B2dNode != 0)) {
			while (here->B2DdPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B2DdPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B2gNode != 0) && (here-> B2gNode != 0)) {
			while (here->B2GgPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B2GgPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B2sNode != 0) && (here-> B2sNode != 0)) {
			while (here->B2SsPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B2SsPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B2bNode != 0) && (here-> B2bNode != 0)) {
			while (here->B2BbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B2BbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B2dNodePrime != 0) && (here-> B2dNodePrime != 0)) {
			while (here->B2DPdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B2DPdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B2sNodePrime != 0) && (here-> B2sNodePrime != 0)) {
			while (here->B2SPspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B2SPspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B2dNode != 0) && (here-> B2dNodePrime != 0)) {
			while (here->B2DdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B2DdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B2gNode != 0) && (here-> B2bNode != 0)) {
			while (here->B2GbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B2GbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B2gNode != 0) && (here-> B2dNodePrime != 0)) {
			while (here->B2GdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B2GdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B2gNode != 0) && (here-> B2sNodePrime != 0)) {
			while (here->B2GspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B2GspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B2sNode != 0) && (here-> B2sNodePrime != 0)) {
			while (here->B2SspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B2SspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B2bNode != 0) && (here-> B2dNodePrime != 0)) {
			while (here->B2BdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B2BdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B2bNode != 0) && (here-> B2sNodePrime != 0)) {
			while (here->B2BspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B2BspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B2dNodePrime != 0) && (here-> B2sNodePrime != 0)) {
			while (here->B2DPspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B2DPspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B2dNodePrime != 0) && (here-> B2dNode != 0)) {
			while (here->B2DPdPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B2DPdPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B2bNode != 0) && (here-> B2gNode != 0)) {
			while (here->B2BgPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B2BgPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B2dNodePrime != 0) && (here-> B2gNode != 0)) {
			while (here->B2DPgPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B2DPgPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B2sNodePrime != 0) && (here-> B2gNode != 0)) {
			while (here->B2SPgPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B2SPgPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B2sNodePrime != 0) && (here-> B2sNode != 0)) {
			while (here->B2SPsPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B2SPsPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B2dNodePrime != 0) && (here-> B2bNode != 0)) {
			while (here->B2DPbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B2DPbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B2sNodePrime != 0) && (here-> B2bNode != 0)) {
			while (here->B2SPbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B2SPbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B2sNodePrime != 0) && (here-> B2dNodePrime != 0)) {
			while (here->B2SPdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B2SPdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
	}
    }
    return(OK);
}
