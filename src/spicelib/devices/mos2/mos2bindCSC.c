/**********
Author: 2012 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "mos2defs.h"
#include "ngspice/sperror.h"

int
MOS2bindCSC(GENmodel *inModel, CKTcircuit *ckt)
{
    MOS2model *model = (MOS2model *)inModel;
    int i ;

    /*  loop through all the mos2 models */
    for( ; model != NULL; model = model->MOS2nextModel ) {
	MOS2instance *here;

        /* loop through all the instances of the model */
        for (here = model->MOS2instances; here != NULL ;
	    here = here->MOS2nextInstance) {

		i = 0 ;
		if ((here-> MOS2dNode != 0) && (here-> MOS2dNode != 0)) {
			while (here->MOS2DdPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MOS2DdPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> MOS2gNode != 0) && (here-> MOS2gNode != 0)) {
			while (here->MOS2GgPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MOS2GgPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> MOS2sNode != 0) && (here-> MOS2sNode != 0)) {
			while (here->MOS2SsPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MOS2SsPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> MOS2bNode != 0) && (here-> MOS2bNode != 0)) {
			while (here->MOS2BbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MOS2BbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> MOS2dNodePrime != 0) && (here-> MOS2dNodePrime != 0)) {
			while (here->MOS2DPdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MOS2DPdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> MOS2sNodePrime != 0) && (here-> MOS2sNodePrime != 0)) {
			while (here->MOS2SPspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MOS2SPspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> MOS2dNode != 0) && (here-> MOS2dNodePrime != 0)) {
			while (here->MOS2DdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MOS2DdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> MOS2gNode != 0) && (here-> MOS2bNode != 0)) {
			while (here->MOS2GbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MOS2GbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> MOS2gNode != 0) && (here-> MOS2dNodePrime != 0)) {
			while (here->MOS2GdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MOS2GdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> MOS2gNode != 0) && (here-> MOS2sNodePrime != 0)) {
			while (here->MOS2GspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MOS2GspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> MOS2sNode != 0) && (here-> MOS2sNodePrime != 0)) {
			while (here->MOS2SspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MOS2SspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> MOS2bNode != 0) && (here-> MOS2dNodePrime != 0)) {
			while (here->MOS2BdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MOS2BdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> MOS2bNode != 0) && (here-> MOS2sNodePrime != 0)) {
			while (here->MOS2BspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MOS2BspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> MOS2dNodePrime != 0) && (here-> MOS2sNodePrime != 0)) {
			while (here->MOS2DPspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MOS2DPspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> MOS2dNodePrime != 0) && (here-> MOS2dNode != 0)) {
			while (here->MOS2DPdPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MOS2DPdPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> MOS2bNode != 0) && (here-> MOS2gNode != 0)) {
			while (here->MOS2BgPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MOS2BgPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> MOS2dNodePrime != 0) && (here-> MOS2gNode != 0)) {
			while (here->MOS2DPgPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MOS2DPgPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> MOS2sNodePrime != 0) && (here-> MOS2gNode != 0)) {
			while (here->MOS2SPgPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MOS2SPgPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> MOS2sNodePrime != 0) && (here-> MOS2sNode != 0)) {
			while (here->MOS2SPsPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MOS2SPsPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> MOS2dNodePrime != 0) && (here-> MOS2bNode != 0)) {
			while (here->MOS2DPbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MOS2DPbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> MOS2sNodePrime != 0) && (here-> MOS2bNode != 0)) {
			while (here->MOS2SPbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MOS2SPbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> MOS2sNodePrime != 0) && (here-> MOS2dNodePrime != 0)) {
			while (here->MOS2SPdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MOS2SPdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
	}
    }
    return(OK);
}

int
MOS2bindCSCComplex(GENmodel *inModel, CKTcircuit *ckt)
{
    MOS2model *model = (MOS2model *)inModel;
    int i ;

    /*  loop through all the mos2 models */
    for( ; model != NULL; model = model->MOS2nextModel ) {
	MOS2instance *here;

        /* loop through all the instances of the model */
        for (here = model->MOS2instances; here != NULL ;
	    here = here->MOS2nextInstance) {

		i = 0 ;
		if ((here-> MOS2dNode != 0) && (here-> MOS2dNode != 0)) {
			while (here->MOS2DdPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MOS2DdPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> MOS2gNode != 0) && (here-> MOS2gNode != 0)) {
			while (here->MOS2GgPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MOS2GgPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> MOS2sNode != 0) && (here-> MOS2sNode != 0)) {
			while (here->MOS2SsPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MOS2SsPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> MOS2bNode != 0) && (here-> MOS2bNode != 0)) {
			while (here->MOS2BbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MOS2BbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> MOS2dNodePrime != 0) && (here-> MOS2dNodePrime != 0)) {
			while (here->MOS2DPdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MOS2DPdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> MOS2sNodePrime != 0) && (here-> MOS2sNodePrime != 0)) {
			while (here->MOS2SPspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MOS2SPspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> MOS2dNode != 0) && (here-> MOS2dNodePrime != 0)) {
			while (here->MOS2DdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MOS2DdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> MOS2gNode != 0) && (here-> MOS2bNode != 0)) {
			while (here->MOS2GbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MOS2GbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> MOS2gNode != 0) && (here-> MOS2dNodePrime != 0)) {
			while (here->MOS2GdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MOS2GdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> MOS2gNode != 0) && (here-> MOS2sNodePrime != 0)) {
			while (here->MOS2GspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MOS2GspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> MOS2sNode != 0) && (here-> MOS2sNodePrime != 0)) {
			while (here->MOS2SspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MOS2SspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> MOS2bNode != 0) && (here-> MOS2dNodePrime != 0)) {
			while (here->MOS2BdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MOS2BdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> MOS2bNode != 0) && (here-> MOS2sNodePrime != 0)) {
			while (here->MOS2BspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MOS2BspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> MOS2dNodePrime != 0) && (here-> MOS2sNodePrime != 0)) {
			while (here->MOS2DPspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MOS2DPspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> MOS2dNodePrime != 0) && (here-> MOS2dNode != 0)) {
			while (here->MOS2DPdPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MOS2DPdPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> MOS2bNode != 0) && (here-> MOS2gNode != 0)) {
			while (here->MOS2BgPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MOS2BgPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> MOS2dNodePrime != 0) && (here-> MOS2gNode != 0)) {
			while (here->MOS2DPgPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MOS2DPgPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> MOS2sNodePrime != 0) && (here-> MOS2gNode != 0)) {
			while (here->MOS2SPgPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MOS2SPgPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> MOS2sNodePrime != 0) && (here-> MOS2sNode != 0)) {
			while (here->MOS2SPsPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MOS2SPsPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> MOS2dNodePrime != 0) && (here-> MOS2bNode != 0)) {
			while (here->MOS2DPbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MOS2DPbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> MOS2sNodePrime != 0) && (here-> MOS2bNode != 0)) {
			while (here->MOS2SPbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MOS2SPbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> MOS2sNodePrime != 0) && (here-> MOS2dNodePrime != 0)) {
			while (here->MOS2SPdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MOS2SPdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
	}
    }
    return(OK);
}
