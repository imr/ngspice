/**********
Author: 2012 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "mos3defs.h"
#include "ngspice/sperror.h"

int
MOS3bindCSC(GENmodel *inModel, CKTcircuit *ckt)
{
    MOS3model *model = (MOS3model *)inModel;
    int i ;

    /*  loop through all the mos3 models */
    for( ; model != NULL; model = model->MOS3nextModel ) {
	MOS3instance *here;

        /* loop through all the instances of the model */
        for (here = model->MOS3instances; here != NULL ;
	    here = here->MOS3nextInstance) {

		i = 0 ;
		if ((here-> MOS3dNode != 0) && (here-> MOS3dNode != 0)) {
			while (here->MOS3DdPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MOS3DdPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> MOS3gNode != 0) && (here-> MOS3gNode != 0)) {
			while (here->MOS3GgPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MOS3GgPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> MOS3sNode != 0) && (here-> MOS3sNode != 0)) {
			while (here->MOS3SsPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MOS3SsPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> MOS3bNode != 0) && (here-> MOS3bNode != 0)) {
			while (here->MOS3BbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MOS3BbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> MOS3dNodePrime != 0) && (here-> MOS3dNodePrime != 0)) {
			while (here->MOS3DPdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MOS3DPdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> MOS3sNodePrime != 0) && (here-> MOS3sNodePrime != 0)) {
			while (here->MOS3SPspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MOS3SPspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> MOS3dNode != 0) && (here-> MOS3dNodePrime != 0)) {
			while (here->MOS3DdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MOS3DdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> MOS3gNode != 0) && (here-> MOS3bNode != 0)) {
			while (here->MOS3GbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MOS3GbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> MOS3gNode != 0) && (here-> MOS3dNodePrime != 0)) {
			while (here->MOS3GdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MOS3GdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> MOS3gNode != 0) && (here-> MOS3sNodePrime != 0)) {
			while (here->MOS3GspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MOS3GspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> MOS3sNode != 0) && (here-> MOS3sNodePrime != 0)) {
			while (here->MOS3SspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MOS3SspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> MOS3bNode != 0) && (here-> MOS3dNodePrime != 0)) {
			while (here->MOS3BdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MOS3BdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> MOS3bNode != 0) && (here-> MOS3sNodePrime != 0)) {
			while (here->MOS3BspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MOS3BspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> MOS3dNodePrime != 0) && (here-> MOS3sNodePrime != 0)) {
			while (here->MOS3DPspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MOS3DPspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> MOS3dNodePrime != 0) && (here-> MOS3dNode != 0)) {
			while (here->MOS3DPdPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MOS3DPdPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> MOS3bNode != 0) && (here-> MOS3gNode != 0)) {
			while (here->MOS3BgPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MOS3BgPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> MOS3dNodePrime != 0) && (here-> MOS3gNode != 0)) {
			while (here->MOS3DPgPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MOS3DPgPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> MOS3sNodePrime != 0) && (here-> MOS3gNode != 0)) {
			while (here->MOS3SPgPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MOS3SPgPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> MOS3sNodePrime != 0) && (here-> MOS3sNode != 0)) {
			while (here->MOS3SPsPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MOS3SPsPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> MOS3dNodePrime != 0) && (here-> MOS3bNode != 0)) {
			while (here->MOS3DPbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MOS3DPbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> MOS3sNodePrime != 0) && (here-> MOS3bNode != 0)) {
			while (here->MOS3SPbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MOS3SPbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> MOS3sNodePrime != 0) && (here-> MOS3dNodePrime != 0)) {
			while (here->MOS3SPdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MOS3SPdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
	}
    }
    return(OK);
}

int
MOS3bindCSCComplex(GENmodel *inModel, CKTcircuit *ckt)
{
    MOS3model *model = (MOS3model *)inModel;
    int i ;

    /*  loop through all the mos3 models */
    for( ; model != NULL; model = model->MOS3nextModel ) {
	MOS3instance *here;

        /* loop through all the instances of the model */
        for (here = model->MOS3instances; here != NULL ;
	    here = here->MOS3nextInstance) {

		i = 0 ;
		if ((here-> MOS3dNode != 0) && (here-> MOS3dNode != 0)) {
			while (here->MOS3DdPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MOS3DdPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> MOS3gNode != 0) && (here-> MOS3gNode != 0)) {
			while (here->MOS3GgPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MOS3GgPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> MOS3sNode != 0) && (here-> MOS3sNode != 0)) {
			while (here->MOS3SsPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MOS3SsPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> MOS3bNode != 0) && (here-> MOS3bNode != 0)) {
			while (here->MOS3BbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MOS3BbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> MOS3dNodePrime != 0) && (here-> MOS3dNodePrime != 0)) {
			while (here->MOS3DPdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MOS3DPdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> MOS3sNodePrime != 0) && (here-> MOS3sNodePrime != 0)) {
			while (here->MOS3SPspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MOS3SPspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> MOS3dNode != 0) && (here-> MOS3dNodePrime != 0)) {
			while (here->MOS3DdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MOS3DdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> MOS3gNode != 0) && (here-> MOS3bNode != 0)) {
			while (here->MOS3GbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MOS3GbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> MOS3gNode != 0) && (here-> MOS3dNodePrime != 0)) {
			while (here->MOS3GdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MOS3GdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> MOS3gNode != 0) && (here-> MOS3sNodePrime != 0)) {
			while (here->MOS3GspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MOS3GspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> MOS3sNode != 0) && (here-> MOS3sNodePrime != 0)) {
			while (here->MOS3SspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MOS3SspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> MOS3bNode != 0) && (here-> MOS3dNodePrime != 0)) {
			while (here->MOS3BdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MOS3BdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> MOS3bNode != 0) && (here-> MOS3sNodePrime != 0)) {
			while (here->MOS3BspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MOS3BspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> MOS3dNodePrime != 0) && (here-> MOS3sNodePrime != 0)) {
			while (here->MOS3DPspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MOS3DPspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> MOS3dNodePrime != 0) && (here-> MOS3dNode != 0)) {
			while (here->MOS3DPdPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MOS3DPdPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> MOS3bNode != 0) && (here-> MOS3gNode != 0)) {
			while (here->MOS3BgPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MOS3BgPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> MOS3dNodePrime != 0) && (here-> MOS3gNode != 0)) {
			while (here->MOS3DPgPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MOS3DPgPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> MOS3sNodePrime != 0) && (here-> MOS3gNode != 0)) {
			while (here->MOS3SPgPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MOS3SPgPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> MOS3sNodePrime != 0) && (here-> MOS3sNode != 0)) {
			while (here->MOS3SPsPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MOS3SPsPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> MOS3dNodePrime != 0) && (here-> MOS3bNode != 0)) {
			while (here->MOS3DPbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MOS3DPbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> MOS3sNodePrime != 0) && (here-> MOS3bNode != 0)) {
			while (here->MOS3SPbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MOS3SPbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> MOS3sNodePrime != 0) && (here-> MOS3dNodePrime != 0)) {
			while (here->MOS3SPdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MOS3SPdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
	}
    }
    return(OK);
}
