/**********
Author: 2012 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "bsim3v0def.h"
#include "ngspice/sperror.h"

int
BSIM3v0bindCSC(GENmodel *inModel, CKTcircuit *ckt)
{
    BSIM3v0model *model = (BSIM3v0model *)inModel;
    int i ;

    /*  loop through all the b3v0 models */
    for( ; model != NULL; model = model->BSIM3v0nextModel ) {
	BSIM3v0instance *here;

        /* loop through all the instances of the model */
        for (here = model->BSIM3v0instances; here != NULL ;
	    here = here->BSIM3v0nextInstance) {

		i = 0 ;
		if ((here-> BSIM3v0dNode != 0) && (here-> BSIM3v0dNode != 0)) {
			while (here->BSIM3v0DdPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM3v0DdPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v0gNode != 0) && (here-> BSIM3v0gNode != 0)) {
			while (here->BSIM3v0GgPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM3v0GgPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v0sNode != 0) && (here-> BSIM3v0sNode != 0)) {
			while (here->BSIM3v0SsPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM3v0SsPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v0bNode != 0) && (here-> BSIM3v0bNode != 0)) {
			while (here->BSIM3v0BbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM3v0BbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v0dNodePrime != 0) && (here-> BSIM3v0dNodePrime != 0)) {
			while (here->BSIM3v0DPdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM3v0DPdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v0sNodePrime != 0) && (here-> BSIM3v0sNodePrime != 0)) {
			while (here->BSIM3v0SPspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM3v0SPspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v0dNode != 0) && (here-> BSIM3v0dNodePrime != 0)) {
			while (here->BSIM3v0DdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM3v0DdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v0gNode != 0) && (here-> BSIM3v0bNode != 0)) {
			while (here->BSIM3v0GbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM3v0GbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v0gNode != 0) && (here-> BSIM3v0dNodePrime != 0)) {
			while (here->BSIM3v0GdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM3v0GdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v0gNode != 0) && (here-> BSIM3v0sNodePrime != 0)) {
			while (here->BSIM3v0GspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM3v0GspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v0sNode != 0) && (here-> BSIM3v0sNodePrime != 0)) {
			while (here->BSIM3v0SspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM3v0SspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v0bNode != 0) && (here-> BSIM3v0dNodePrime != 0)) {
			while (here->BSIM3v0BdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM3v0BdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v0bNode != 0) && (here-> BSIM3v0sNodePrime != 0)) {
			while (here->BSIM3v0BspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM3v0BspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v0dNodePrime != 0) && (here-> BSIM3v0sNodePrime != 0)) {
			while (here->BSIM3v0DPspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM3v0DPspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v0dNodePrime != 0) && (here-> BSIM3v0dNode != 0)) {
			while (here->BSIM3v0DPdPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM3v0DPdPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v0bNode != 0) && (here-> BSIM3v0gNode != 0)) {
			while (here->BSIM3v0BgPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM3v0BgPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v0dNodePrime != 0) && (here-> BSIM3v0gNode != 0)) {
			while (here->BSIM3v0DPgPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM3v0DPgPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v0sNodePrime != 0) && (here-> BSIM3v0gNode != 0)) {
			while (here->BSIM3v0SPgPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM3v0SPgPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v0sNodePrime != 0) && (here-> BSIM3v0sNode != 0)) {
			while (here->BSIM3v0SPsPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM3v0SPsPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v0dNodePrime != 0) && (here-> BSIM3v0bNode != 0)) {
			while (here->BSIM3v0DPbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM3v0DPbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v0sNodePrime != 0) && (here-> BSIM3v0bNode != 0)) {
			while (here->BSIM3v0SPbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM3v0SPbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v0sNodePrime != 0) && (here-> BSIM3v0dNodePrime != 0)) {
			while (here->BSIM3v0SPdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM3v0SPdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v0qNode != 0) && (here-> BSIM3v0qNode != 0)) {
			while (here->BSIM3v0QqPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM3v0QqPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v0qNode != 0) && (here-> BSIM3v0dNodePrime != 0)) {
			while (here->BSIM3v0QdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM3v0QdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v0qNode != 0) && (here-> BSIM3v0sNodePrime != 0)) {
			while (here->BSIM3v0QspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM3v0QspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v0qNode != 0) && (here-> BSIM3v0gNode != 0)) {
			while (here->BSIM3v0QgPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM3v0QgPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v0qNode != 0) && (here-> BSIM3v0bNode != 0)) {
			while (here->BSIM3v0QbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM3v0QbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v0dNodePrime != 0) && (here-> BSIM3v0qNode != 0)) {
			while (here->BSIM3v0DPqPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM3v0DPqPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v0sNodePrime != 0) && (here-> BSIM3v0qNode != 0)) {
			while (here->BSIM3v0SPqPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM3v0SPqPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v0gNode != 0) && (here-> BSIM3v0qNode != 0)) {
			while (here->BSIM3v0GqPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM3v0GqPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v0bNode != 0) && (here-> BSIM3v0qNode != 0)) {
			while (here->BSIM3v0BqPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM3v0BqPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
	}
    }
    return(OK);
}

int
BSIM3v0bindCSCComplex(GENmodel *inModel, CKTcircuit *ckt)
{
    BSIM3v0model *model = (BSIM3v0model *)inModel;
    int i ;

    /*  loop through all the b3v0 models */
    for( ; model != NULL; model = model->BSIM3v0nextModel ) {
	BSIM3v0instance *here;

        /* loop through all the instances of the model */
        for (here = model->BSIM3v0instances; here != NULL ;
	    here = here->BSIM3v0nextInstance) {

		i = 0 ;
		if ((here-> BSIM3v0dNode != 0) && (here-> BSIM3v0dNode != 0)) {
			while (here->BSIM3v0DdPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM3v0DdPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v0gNode != 0) && (here-> BSIM3v0gNode != 0)) {
			while (here->BSIM3v0GgPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM3v0GgPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v0sNode != 0) && (here-> BSIM3v0sNode != 0)) {
			while (here->BSIM3v0SsPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM3v0SsPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v0bNode != 0) && (here-> BSIM3v0bNode != 0)) {
			while (here->BSIM3v0BbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM3v0BbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v0dNodePrime != 0) && (here-> BSIM3v0dNodePrime != 0)) {
			while (here->BSIM3v0DPdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM3v0DPdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v0sNodePrime != 0) && (here-> BSIM3v0sNodePrime != 0)) {
			while (here->BSIM3v0SPspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM3v0SPspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v0dNode != 0) && (here-> BSIM3v0dNodePrime != 0)) {
			while (here->BSIM3v0DdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM3v0DdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v0gNode != 0) && (here-> BSIM3v0bNode != 0)) {
			while (here->BSIM3v0GbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM3v0GbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v0gNode != 0) && (here-> BSIM3v0dNodePrime != 0)) {
			while (here->BSIM3v0GdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM3v0GdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v0gNode != 0) && (here-> BSIM3v0sNodePrime != 0)) {
			while (here->BSIM3v0GspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM3v0GspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v0sNode != 0) && (here-> BSIM3v0sNodePrime != 0)) {
			while (here->BSIM3v0SspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM3v0SspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v0bNode != 0) && (here-> BSIM3v0dNodePrime != 0)) {
			while (here->BSIM3v0BdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM3v0BdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v0bNode != 0) && (here-> BSIM3v0sNodePrime != 0)) {
			while (here->BSIM3v0BspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM3v0BspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v0dNodePrime != 0) && (here-> BSIM3v0sNodePrime != 0)) {
			while (here->BSIM3v0DPspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM3v0DPspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v0dNodePrime != 0) && (here-> BSIM3v0dNode != 0)) {
			while (here->BSIM3v0DPdPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM3v0DPdPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v0bNode != 0) && (here-> BSIM3v0gNode != 0)) {
			while (here->BSIM3v0BgPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM3v0BgPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v0dNodePrime != 0) && (here-> BSIM3v0gNode != 0)) {
			while (here->BSIM3v0DPgPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM3v0DPgPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v0sNodePrime != 0) && (here-> BSIM3v0gNode != 0)) {
			while (here->BSIM3v0SPgPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM3v0SPgPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v0sNodePrime != 0) && (here-> BSIM3v0sNode != 0)) {
			while (here->BSIM3v0SPsPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM3v0SPsPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v0dNodePrime != 0) && (here-> BSIM3v0bNode != 0)) {
			while (here->BSIM3v0DPbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM3v0DPbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v0sNodePrime != 0) && (here-> BSIM3v0bNode != 0)) {
			while (here->BSIM3v0SPbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM3v0SPbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v0sNodePrime != 0) && (here-> BSIM3v0dNodePrime != 0)) {
			while (here->BSIM3v0SPdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM3v0SPdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v0qNode != 0) && (here-> BSIM3v0qNode != 0)) {
			while (here->BSIM3v0QqPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM3v0QqPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v0qNode != 0) && (here-> BSIM3v0dNodePrime != 0)) {
			while (here->BSIM3v0QdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM3v0QdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v0qNode != 0) && (here-> BSIM3v0sNodePrime != 0)) {
			while (here->BSIM3v0QspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM3v0QspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v0qNode != 0) && (here-> BSIM3v0gNode != 0)) {
			while (here->BSIM3v0QgPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM3v0QgPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v0qNode != 0) && (here-> BSIM3v0bNode != 0)) {
			while (here->BSIM3v0QbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM3v0QbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v0dNodePrime != 0) && (here-> BSIM3v0qNode != 0)) {
			while (here->BSIM3v0DPqPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM3v0DPqPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v0sNodePrime != 0) && (here-> BSIM3v0qNode != 0)) {
			while (here->BSIM3v0SPqPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM3v0SPqPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v0gNode != 0) && (here-> BSIM3v0qNode != 0)) {
			while (here->BSIM3v0GqPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM3v0GqPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v0bNode != 0) && (here-> BSIM3v0qNode != 0)) {
			while (here->BSIM3v0BqPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM3v0BqPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
	}
    }
    return(OK);
}

int
BSIM3v0bindCSCComplexToReal (GENmodel *inModel, CKTcircuit *ckt)
{
    BSIM3v0model *model = (BSIM3v0model *)inModel ;
    BSIM3v0instance *here ;
    int i ;

    /*  loop through all the bsim3v0 models */
    for ( ; model != NULL ; model = model->BSIM3v0nextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->BSIM3v0instances ; here != NULL ; here = here->BSIM3v0nextInstance)
        {
            i = 0 ;
            if ((here->BSIM3v0dNode != 0) && (here->BSIM3v0dNode != 0))
            {
                while (here->BSIM3v0DdPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM3v0DdPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM3v0gNode != 0) && (here->BSIM3v0gNode != 0))
            {
                while (here->BSIM3v0GgPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM3v0GgPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM3v0sNode != 0) && (here->BSIM3v0sNode != 0))
            {
                while (here->BSIM3v0SsPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM3v0SsPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM3v0bNode != 0) && (here->BSIM3v0bNode != 0))
            {
                while (here->BSIM3v0BbPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM3v0BbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM3v0dNodePrime != 0) && (here->BSIM3v0dNodePrime != 0))
            {
                while (here->BSIM3v0DPdpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM3v0DPdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM3v0sNodePrime != 0) && (here->BSIM3v0sNodePrime != 0))
            {
                while (here->BSIM3v0SPspPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM3v0SPspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM3v0dNode != 0) && (here->BSIM3v0dNodePrime != 0))
            {
                while (here->BSIM3v0DdpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM3v0DdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM3v0gNode != 0) && (here->BSIM3v0bNode != 0))
            {
                while (here->BSIM3v0GbPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM3v0GbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM3v0gNode != 0) && (here->BSIM3v0dNodePrime != 0))
            {
                while (here->BSIM3v0GdpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM3v0GdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM3v0gNode != 0) && (here->BSIM3v0sNodePrime != 0))
            {
                while (here->BSIM3v0GspPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM3v0GspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM3v0sNode != 0) && (here->BSIM3v0sNodePrime != 0))
            {
                while (here->BSIM3v0SspPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM3v0SspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM3v0bNode != 0) && (here->BSIM3v0dNodePrime != 0))
            {
                while (here->BSIM3v0BdpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM3v0BdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM3v0bNode != 0) && (here->BSIM3v0sNodePrime != 0))
            {
                while (here->BSIM3v0BspPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM3v0BspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM3v0dNodePrime != 0) && (here->BSIM3v0sNodePrime != 0))
            {
                while (here->BSIM3v0DPspPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM3v0DPspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM3v0dNodePrime != 0) && (here->BSIM3v0dNode != 0))
            {
                while (here->BSIM3v0DPdPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM3v0DPdPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM3v0bNode != 0) && (here->BSIM3v0gNode != 0))
            {
                while (here->BSIM3v0BgPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM3v0BgPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM3v0dNodePrime != 0) && (here->BSIM3v0gNode != 0))
            {
                while (here->BSIM3v0DPgPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM3v0DPgPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM3v0sNodePrime != 0) && (here->BSIM3v0gNode != 0))
            {
                while (here->BSIM3v0SPgPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM3v0SPgPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM3v0sNodePrime != 0) && (here->BSIM3v0sNode != 0))
            {
                while (here->BSIM3v0SPsPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM3v0SPsPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM3v0dNodePrime != 0) && (here->BSIM3v0bNode != 0))
            {
                while (here->BSIM3v0DPbPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM3v0DPbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM3v0sNodePrime != 0) && (here->BSIM3v0bNode != 0))
            {
                while (here->BSIM3v0SPbPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM3v0SPbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM3v0sNodePrime != 0) && (here->BSIM3v0dNodePrime != 0))
            {
                while (here->BSIM3v0SPdpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM3v0SPdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM3v0qNode != 0) && (here->BSIM3v0qNode != 0))
            {
                while (here->BSIM3v0QqPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM3v0QqPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM3v0qNode != 0) && (here->BSIM3v0dNodePrime != 0))
            {
                while (here->BSIM3v0QdpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM3v0QdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM3v0qNode != 0) && (here->BSIM3v0sNodePrime != 0))
            {
                while (here->BSIM3v0QspPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM3v0QspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM3v0qNode != 0) && (here->BSIM3v0gNode != 0))
            {
                while (here->BSIM3v0QgPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM3v0QgPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM3v0qNode != 0) && (here->BSIM3v0bNode != 0))
            {
                while (here->BSIM3v0QbPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM3v0QbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM3v0dNodePrime != 0) && (here->BSIM3v0qNode != 0))
            {
                while (here->BSIM3v0DPqPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM3v0DPqPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM3v0sNodePrime != 0) && (here->BSIM3v0qNode != 0))
            {
                while (here->BSIM3v0SPqPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM3v0SPqPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM3v0gNode != 0) && (here->BSIM3v0qNode != 0))
            {
                while (here->BSIM3v0GqPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM3v0GqPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM3v0bNode != 0) && (here->BSIM3v0qNode != 0))
            {
                while (here->BSIM3v0BqPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM3v0BqPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }
        }
    }

    return (OK) ;
}