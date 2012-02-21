/**********
Author: 2012 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "mos9defs.h"
#include "ngspice/sperror.h"

int
MOS9bindCSC(GENmodel *inModel, CKTcircuit *ckt)
{
    MOS9model *model = (MOS9model *)inModel;
    int i ;

    /*  loop through all the mos9 models */
    for( ; model != NULL; model = model->MOS9nextModel ) {
	MOS9instance *here;

        /* loop through all the instances of the model */
        for (here = model->MOS9instances; here != NULL ;
	    here = here->MOS9nextInstance) {

		i = 0 ;
		if ((here-> MOS9dNode != 0) && (here-> MOS9dNode != 0)) {
			while (here->MOS9DdPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MOS9DdPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> MOS9gNode != 0) && (here-> MOS9gNode != 0)) {
			while (here->MOS9GgPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MOS9GgPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> MOS9sNode != 0) && (here-> MOS9sNode != 0)) {
			while (here->MOS9SsPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MOS9SsPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> MOS9bNode != 0) && (here-> MOS9bNode != 0)) {
			while (here->MOS9BbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MOS9BbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> MOS9dNodePrime != 0) && (here-> MOS9dNodePrime != 0)) {
			while (here->MOS9DPdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MOS9DPdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> MOS9sNodePrime != 0) && (here-> MOS9sNodePrime != 0)) {
			while (here->MOS9SPspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MOS9SPspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> MOS9dNode != 0) && (here-> MOS9dNodePrime != 0)) {
			while (here->MOS9DdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MOS9DdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> MOS9gNode != 0) && (here-> MOS9bNode != 0)) {
			while (here->MOS9GbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MOS9GbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> MOS9gNode != 0) && (here-> MOS9dNodePrime != 0)) {
			while (here->MOS9GdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MOS9GdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> MOS9gNode != 0) && (here-> MOS9sNodePrime != 0)) {
			while (here->MOS9GspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MOS9GspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> MOS9sNode != 0) && (here-> MOS9sNodePrime != 0)) {
			while (here->MOS9SspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MOS9SspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> MOS9bNode != 0) && (here-> MOS9dNodePrime != 0)) {
			while (here->MOS9BdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MOS9BdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> MOS9bNode != 0) && (here-> MOS9sNodePrime != 0)) {
			while (here->MOS9BspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MOS9BspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> MOS9dNodePrime != 0) && (here-> MOS9sNodePrime != 0)) {
			while (here->MOS9DPspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MOS9DPspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> MOS9dNodePrime != 0) && (here-> MOS9dNode != 0)) {
			while (here->MOS9DPdPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MOS9DPdPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> MOS9bNode != 0) && (here-> MOS9gNode != 0)) {
			while (here->MOS9BgPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MOS9BgPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> MOS9dNodePrime != 0) && (here-> MOS9gNode != 0)) {
			while (here->MOS9DPgPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MOS9DPgPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> MOS9sNodePrime != 0) && (here-> MOS9gNode != 0)) {
			while (here->MOS9SPgPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MOS9SPgPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> MOS9sNodePrime != 0) && (here-> MOS9sNode != 0)) {
			while (here->MOS9SPsPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MOS9SPsPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> MOS9dNodePrime != 0) && (here-> MOS9bNode != 0)) {
			while (here->MOS9DPbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MOS9DPbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> MOS9sNodePrime != 0) && (here-> MOS9bNode != 0)) {
			while (here->MOS9SPbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MOS9SPbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> MOS9sNodePrime != 0) && (here-> MOS9dNodePrime != 0)) {
			while (here->MOS9SPdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MOS9SPdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
	}
    }
    return(OK);
}

int
MOS9bindCSCComplex(GENmodel *inModel, CKTcircuit *ckt)
{
    MOS9model *model = (MOS9model *)inModel;
    int i ;

    /*  loop through all the mos9 models */
    for( ; model != NULL; model = model->MOS9nextModel ) {
	MOS9instance *here;

        /* loop through all the instances of the model */
        for (here = model->MOS9instances; here != NULL ;
	    here = here->MOS9nextInstance) {

		i = 0 ;
		if ((here-> MOS9dNode != 0) && (here-> MOS9dNode != 0)) {
			while (here->MOS9DdPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MOS9DdPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> MOS9gNode != 0) && (here-> MOS9gNode != 0)) {
			while (here->MOS9GgPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MOS9GgPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> MOS9sNode != 0) && (here-> MOS9sNode != 0)) {
			while (here->MOS9SsPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MOS9SsPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> MOS9bNode != 0) && (here-> MOS9bNode != 0)) {
			while (here->MOS9BbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MOS9BbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> MOS9dNodePrime != 0) && (here-> MOS9dNodePrime != 0)) {
			while (here->MOS9DPdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MOS9DPdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> MOS9sNodePrime != 0) && (here-> MOS9sNodePrime != 0)) {
			while (here->MOS9SPspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MOS9SPspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> MOS9dNode != 0) && (here-> MOS9dNodePrime != 0)) {
			while (here->MOS9DdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MOS9DdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> MOS9gNode != 0) && (here-> MOS9bNode != 0)) {
			while (here->MOS9GbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MOS9GbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> MOS9gNode != 0) && (here-> MOS9dNodePrime != 0)) {
			while (here->MOS9GdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MOS9GdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> MOS9gNode != 0) && (here-> MOS9sNodePrime != 0)) {
			while (here->MOS9GspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MOS9GspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> MOS9sNode != 0) && (here-> MOS9sNodePrime != 0)) {
			while (here->MOS9SspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MOS9SspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> MOS9bNode != 0) && (here-> MOS9dNodePrime != 0)) {
			while (here->MOS9BdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MOS9BdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> MOS9bNode != 0) && (here-> MOS9sNodePrime != 0)) {
			while (here->MOS9BspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MOS9BspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> MOS9dNodePrime != 0) && (here-> MOS9sNodePrime != 0)) {
			while (here->MOS9DPspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MOS9DPspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> MOS9dNodePrime != 0) && (here-> MOS9dNode != 0)) {
			while (here->MOS9DPdPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MOS9DPdPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> MOS9bNode != 0) && (here-> MOS9gNode != 0)) {
			while (here->MOS9BgPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MOS9BgPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> MOS9dNodePrime != 0) && (here-> MOS9gNode != 0)) {
			while (here->MOS9DPgPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MOS9DPgPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> MOS9sNodePrime != 0) && (here-> MOS9gNode != 0)) {
			while (here->MOS9SPgPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MOS9SPgPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> MOS9sNodePrime != 0) && (here-> MOS9sNode != 0)) {
			while (here->MOS9SPsPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MOS9SPsPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> MOS9dNodePrime != 0) && (here-> MOS9bNode != 0)) {
			while (here->MOS9DPbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MOS9DPbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> MOS9sNodePrime != 0) && (here-> MOS9bNode != 0)) {
			while (here->MOS9SPbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MOS9SPbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> MOS9sNodePrime != 0) && (here-> MOS9dNodePrime != 0)) {
			while (here->MOS9SPdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MOS9SPdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
	}
    }
    return(OK);
}
