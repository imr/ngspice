/**********
Author: 2012 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "bsim3def.h"
#include "ngspice/sperror.h"

int
BSIM3bindCSC(GENmodel *inModel, CKTcircuit *ckt)
{
    BSIM3model *model = (BSIM3model *)inModel;
    int i ;

    /*  loop through all the b3 models */
    for( ; model != NULL; model = model->BSIM3nextModel ) {
	BSIM3instance *here;

        /* loop through all the instances of the model */
        for (here = model->BSIM3instances; here != NULL ;
	    here = here->BSIM3nextInstance) {

		i = 0 ;
		if ((here-> BSIM3dNode != 0) && (here-> BSIM3dNode != 0)) {
			while (here->BSIM3DdPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM3DdPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3gNode != 0) && (here-> BSIM3gNode != 0)) {
			while (here->BSIM3GgPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM3GgPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3sNode != 0) && (here-> BSIM3sNode != 0)) {
			while (here->BSIM3SsPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM3SsPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3bNode != 0) && (here-> BSIM3bNode != 0)) {
			while (here->BSIM3BbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM3BbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3dNodePrime != 0) && (here-> BSIM3dNodePrime != 0)) {
			while (here->BSIM3DPdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM3DPdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3sNodePrime != 0) && (here-> BSIM3sNodePrime != 0)) {
			while (here->BSIM3SPspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM3SPspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3dNode != 0) && (here-> BSIM3dNodePrime != 0)) {
			while (here->BSIM3DdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM3DdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3gNode != 0) && (here-> BSIM3bNode != 0)) {
			while (here->BSIM3GbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM3GbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3gNode != 0) && (here-> BSIM3dNodePrime != 0)) {
			while (here->BSIM3GdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM3GdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3gNode != 0) && (here-> BSIM3sNodePrime != 0)) {
			while (here->BSIM3GspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM3GspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3sNode != 0) && (here-> BSIM3sNodePrime != 0)) {
			while (here->BSIM3SspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM3SspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3bNode != 0) && (here-> BSIM3dNodePrime != 0)) {
			while (here->BSIM3BdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM3BdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3bNode != 0) && (here-> BSIM3sNodePrime != 0)) {
			while (here->BSIM3BspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM3BspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3dNodePrime != 0) && (here-> BSIM3sNodePrime != 0)) {
			while (here->BSIM3DPspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM3DPspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3dNodePrime != 0) && (here-> BSIM3dNode != 0)) {
			while (here->BSIM3DPdPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM3DPdPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3bNode != 0) && (here-> BSIM3gNode != 0)) {
			while (here->BSIM3BgPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM3BgPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3dNodePrime != 0) && (here-> BSIM3gNode != 0)) {
			while (here->BSIM3DPgPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM3DPgPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3sNodePrime != 0) && (here-> BSIM3gNode != 0)) {
			while (here->BSIM3SPgPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM3SPgPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3sNodePrime != 0) && (here-> BSIM3sNode != 0)) {
			while (here->BSIM3SPsPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM3SPsPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3dNodePrime != 0) && (here-> BSIM3bNode != 0)) {
			while (here->BSIM3DPbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM3DPbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3sNodePrime != 0) && (here-> BSIM3bNode != 0)) {
			while (here->BSIM3SPbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM3SPbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3sNodePrime != 0) && (here-> BSIM3dNodePrime != 0)) {
			while (here->BSIM3SPdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM3SPdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3qNode != 0) && (here-> BSIM3qNode != 0)) {
			while (here->BSIM3QqPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM3QqPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3qNode != 0) && (here-> BSIM3dNodePrime != 0)) {
			while (here->BSIM3QdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM3QdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3qNode != 0) && (here-> BSIM3sNodePrime != 0)) {
			while (here->BSIM3QspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM3QspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3qNode != 0) && (here-> BSIM3gNode != 0)) {
			while (here->BSIM3QgPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM3QgPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3qNode != 0) && (here-> BSIM3bNode != 0)) {
			while (here->BSIM3QbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM3QbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3dNodePrime != 0) && (here-> BSIM3qNode != 0)) {
			while (here->BSIM3DPqPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM3DPqPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3sNodePrime != 0) && (here-> BSIM3qNode != 0)) {
			while (here->BSIM3SPqPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM3SPqPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3gNode != 0) && (here-> BSIM3qNode != 0)) {
			while (here->BSIM3GqPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM3GqPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3bNode != 0) && (here-> BSIM3qNode != 0)) {
			while (here->BSIM3BqPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM3BqPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
	}
    }
    return(OK);
}

int
BSIM3bindCSCComplex(GENmodel *inModel, CKTcircuit *ckt)
{
    BSIM3model *model = (BSIM3model *)inModel;
    int i ;

    /*  loop through all the b3 models */
    for( ; model != NULL; model = model->BSIM3nextModel ) {
	BSIM3instance *here;

        /* loop through all the instances of the model */
        for (here = model->BSIM3instances; here != NULL ;
	    here = here->BSIM3nextInstance) {

		i = 0 ;
		if ((here-> BSIM3dNode != 0) && (here-> BSIM3dNode != 0)) {
			while (here->BSIM3DdPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM3DdPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3gNode != 0) && (here-> BSIM3gNode != 0)) {
			while (here->BSIM3GgPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM3GgPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3sNode != 0) && (here-> BSIM3sNode != 0)) {
			while (here->BSIM3SsPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM3SsPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3bNode != 0) && (here-> BSIM3bNode != 0)) {
			while (here->BSIM3BbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM3BbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3dNodePrime != 0) && (here-> BSIM3dNodePrime != 0)) {
			while (here->BSIM3DPdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM3DPdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3sNodePrime != 0) && (here-> BSIM3sNodePrime != 0)) {
			while (here->BSIM3SPspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM3SPspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3dNode != 0) && (here-> BSIM3dNodePrime != 0)) {
			while (here->BSIM3DdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM3DdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3gNode != 0) && (here-> BSIM3bNode != 0)) {
			while (here->BSIM3GbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM3GbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3gNode != 0) && (here-> BSIM3dNodePrime != 0)) {
			while (here->BSIM3GdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM3GdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3gNode != 0) && (here-> BSIM3sNodePrime != 0)) {
			while (here->BSIM3GspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM3GspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3sNode != 0) && (here-> BSIM3sNodePrime != 0)) {
			while (here->BSIM3SspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM3SspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3bNode != 0) && (here-> BSIM3dNodePrime != 0)) {
			while (here->BSIM3BdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM3BdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3bNode != 0) && (here-> BSIM3sNodePrime != 0)) {
			while (here->BSIM3BspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM3BspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3dNodePrime != 0) && (here-> BSIM3sNodePrime != 0)) {
			while (here->BSIM3DPspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM3DPspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3dNodePrime != 0) && (here-> BSIM3dNode != 0)) {
			while (here->BSIM3DPdPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM3DPdPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3bNode != 0) && (here-> BSIM3gNode != 0)) {
			while (here->BSIM3BgPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM3BgPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3dNodePrime != 0) && (here-> BSIM3gNode != 0)) {
			while (here->BSIM3DPgPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM3DPgPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3sNodePrime != 0) && (here-> BSIM3gNode != 0)) {
			while (here->BSIM3SPgPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM3SPgPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3sNodePrime != 0) && (here-> BSIM3sNode != 0)) {
			while (here->BSIM3SPsPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM3SPsPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3dNodePrime != 0) && (here-> BSIM3bNode != 0)) {
			while (here->BSIM3DPbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM3DPbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3sNodePrime != 0) && (here-> BSIM3bNode != 0)) {
			while (here->BSIM3SPbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM3SPbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3sNodePrime != 0) && (here-> BSIM3dNodePrime != 0)) {
			while (here->BSIM3SPdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM3SPdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3qNode != 0) && (here-> BSIM3qNode != 0)) {
			while (here->BSIM3QqPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM3QqPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3qNode != 0) && (here-> BSIM3dNodePrime != 0)) {
			while (here->BSIM3QdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM3QdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3qNode != 0) && (here-> BSIM3sNodePrime != 0)) {
			while (here->BSIM3QspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM3QspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3qNode != 0) && (here-> BSIM3gNode != 0)) {
			while (here->BSIM3QgPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM3QgPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3qNode != 0) && (here-> BSIM3bNode != 0)) {
			while (here->BSIM3QbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM3QbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3dNodePrime != 0) && (here-> BSIM3qNode != 0)) {
			while (here->BSIM3DPqPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM3DPqPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3sNodePrime != 0) && (here-> BSIM3qNode != 0)) {
			while (here->BSIM3SPqPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM3SPqPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3gNode != 0) && (here-> BSIM3qNode != 0)) {
			while (here->BSIM3GqPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM3GqPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3bNode != 0) && (here-> BSIM3qNode != 0)) {
			while (here->BSIM3BqPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM3BqPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
	}
    }
    return(OK);
}

int
BSIM3bindCSCComplexToReal (GENmodel *inModel, CKTcircuit *ckt)
{
    BSIM3model *model = (BSIM3model *)inModel ;
    BSIM3instance *here ;
    int i ;

    /*  loop through all the bsim3 models */
    for ( ; model != NULL ; model = model->BSIM3nextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->BSIM3instances ; here != NULL ; here = here->BSIM3nextInstance)
        {
            i = 0 ;
            if ((here->BSIM3dNode != 0) && (here->BSIM3dNode != 0))
            {
                while (here->BSIM3DdPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM3DdPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM3gNode != 0) && (here->BSIM3gNode != 0))
            {
                while (here->BSIM3GgPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM3GgPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM3sNode != 0) && (here->BSIM3sNode != 0))
            {
                while (here->BSIM3SsPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM3SsPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM3bNode != 0) && (here->BSIM3bNode != 0))
            {
                while (here->BSIM3BbPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM3BbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM3dNodePrime != 0) && (here->BSIM3dNodePrime != 0))
            {
                while (here->BSIM3DPdpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM3DPdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM3sNodePrime != 0) && (here->BSIM3sNodePrime != 0))
            {
                while (here->BSIM3SPspPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM3SPspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM3dNode != 0) && (here->BSIM3dNodePrime != 0))
            {
                while (here->BSIM3DdpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM3DdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM3gNode != 0) && (here->BSIM3bNode != 0))
            {
                while (here->BSIM3GbPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM3GbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM3gNode != 0) && (here->BSIM3dNodePrime != 0))
            {
                while (here->BSIM3GdpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM3GdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM3gNode != 0) && (here->BSIM3sNodePrime != 0))
            {
                while (here->BSIM3GspPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM3GspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM3sNode != 0) && (here->BSIM3sNodePrime != 0))
            {
                while (here->BSIM3SspPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM3SspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM3bNode != 0) && (here->BSIM3dNodePrime != 0))
            {
                while (here->BSIM3BdpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM3BdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM3bNode != 0) && (here->BSIM3sNodePrime != 0))
            {
                while (here->BSIM3BspPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM3BspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM3dNodePrime != 0) && (here->BSIM3sNodePrime != 0))
            {
                while (here->BSIM3DPspPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM3DPspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM3dNodePrime != 0) && (here->BSIM3dNode != 0))
            {
                while (here->BSIM3DPdPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM3DPdPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM3bNode != 0) && (here->BSIM3gNode != 0))
            {
                while (here->BSIM3BgPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM3BgPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM3dNodePrime != 0) && (here->BSIM3gNode != 0))
            {
                while (here->BSIM3DPgPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM3DPgPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM3sNodePrime != 0) && (here->BSIM3gNode != 0))
            {
                while (here->BSIM3SPgPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM3SPgPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM3sNodePrime != 0) && (here->BSIM3sNode != 0))
            {
                while (here->BSIM3SPsPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM3SPsPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM3dNodePrime != 0) && (here->BSIM3bNode != 0))
            {
                while (here->BSIM3DPbPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM3DPbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM3sNodePrime != 0) && (here->BSIM3bNode != 0))
            {
                while (here->BSIM3SPbPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM3SPbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM3sNodePrime != 0) && (here->BSIM3dNodePrime != 0))
            {
                while (here->BSIM3SPdpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM3SPdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM3qNode != 0) && (here->BSIM3qNode != 0))
            {
                while (here->BSIM3QqPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM3QqPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM3qNode != 0) && (here->BSIM3dNodePrime != 0))
            {
                while (here->BSIM3QdpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM3QdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM3qNode != 0) && (here->BSIM3sNodePrime != 0))
            {
                while (here->BSIM3QspPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM3QspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM3qNode != 0) && (here->BSIM3gNode != 0))
            {
                while (here->BSIM3QgPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM3QgPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM3qNode != 0) && (here->BSIM3bNode != 0))
            {
                while (here->BSIM3QbPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM3QbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM3dNodePrime != 0) && (here->BSIM3qNode != 0))
            {
                while (here->BSIM3DPqPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM3DPqPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM3sNodePrime != 0) && (here->BSIM3qNode != 0))
            {
                while (here->BSIM3SPqPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM3SPqPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM3gNode != 0) && (here->BSIM3qNode != 0))
            {
                while (here->BSIM3GqPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM3GqPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM3bNode != 0) && (here->BSIM3qNode != 0))
            {
                while (here->BSIM3BqPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM3BqPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }
        }
    }

    return (OK) ;
}