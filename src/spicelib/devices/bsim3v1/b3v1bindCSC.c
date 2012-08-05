/**********
Author: 2012 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "bsim3v1def.h"
#include "ngspice/sperror.h"

int
BSIM3v1bindCSC(GENmodel *inModel, CKTcircuit *ckt)
{
    BSIM3v1model *model = (BSIM3v1model *)inModel;
    int i ;

    /*  loop through all the b3v1 models */
    for( ; model != NULL; model = model->BSIM3v1nextModel ) {
	BSIM3v1instance *here;

        /* loop through all the instances of the model */
        for (here = model->BSIM3v1instances; here != NULL ;
	    here = here->BSIM3v1nextInstance) {

		i = 0 ;
		if ((here-> BSIM3v1dNode != 0) && (here-> BSIM3v1dNode != 0)) {
			while (here->BSIM3v1DdPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM3v1DdPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v1gNode != 0) && (here-> BSIM3v1gNode != 0)) {
			while (here->BSIM3v1GgPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM3v1GgPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v1sNode != 0) && (here-> BSIM3v1sNode != 0)) {
			while (here->BSIM3v1SsPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM3v1SsPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v1bNode != 0) && (here-> BSIM3v1bNode != 0)) {
			while (here->BSIM3v1BbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM3v1BbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v1dNodePrime != 0) && (here-> BSIM3v1dNodePrime != 0)) {
			while (here->BSIM3v1DPdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM3v1DPdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v1sNodePrime != 0) && (here-> BSIM3v1sNodePrime != 0)) {
			while (here->BSIM3v1SPspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM3v1SPspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v1dNode != 0) && (here-> BSIM3v1dNodePrime != 0)) {
			while (here->BSIM3v1DdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM3v1DdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v1gNode != 0) && (here-> BSIM3v1bNode != 0)) {
			while (here->BSIM3v1GbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM3v1GbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v1gNode != 0) && (here-> BSIM3v1dNodePrime != 0)) {
			while (here->BSIM3v1GdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM3v1GdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v1gNode != 0) && (here-> BSIM3v1sNodePrime != 0)) {
			while (here->BSIM3v1GspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM3v1GspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v1sNode != 0) && (here-> BSIM3v1sNodePrime != 0)) {
			while (here->BSIM3v1SspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM3v1SspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v1bNode != 0) && (here-> BSIM3v1dNodePrime != 0)) {
			while (here->BSIM3v1BdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM3v1BdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v1bNode != 0) && (here-> BSIM3v1sNodePrime != 0)) {
			while (here->BSIM3v1BspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM3v1BspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v1dNodePrime != 0) && (here-> BSIM3v1sNodePrime != 0)) {
			while (here->BSIM3v1DPspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM3v1DPspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v1dNodePrime != 0) && (here-> BSIM3v1dNode != 0)) {
			while (here->BSIM3v1DPdPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM3v1DPdPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v1bNode != 0) && (here-> BSIM3v1gNode != 0)) {
			while (here->BSIM3v1BgPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM3v1BgPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v1dNodePrime != 0) && (here-> BSIM3v1gNode != 0)) {
			while (here->BSIM3v1DPgPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM3v1DPgPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v1sNodePrime != 0) && (here-> BSIM3v1gNode != 0)) {
			while (here->BSIM3v1SPgPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM3v1SPgPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v1sNodePrime != 0) && (here-> BSIM3v1sNode != 0)) {
			while (here->BSIM3v1SPsPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM3v1SPsPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v1dNodePrime != 0) && (here-> BSIM3v1bNode != 0)) {
			while (here->BSIM3v1DPbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM3v1DPbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v1sNodePrime != 0) && (here-> BSIM3v1bNode != 0)) {
			while (here->BSIM3v1SPbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM3v1SPbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v1sNodePrime != 0) && (here-> BSIM3v1dNodePrime != 0)) {
			while (here->BSIM3v1SPdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM3v1SPdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v1qNode != 0) && (here-> BSIM3v1qNode != 0)) {
			while (here->BSIM3v1QqPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM3v1QqPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v1qNode != 0) && (here-> BSIM3v1dNodePrime != 0)) {
			while (here->BSIM3v1QdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM3v1QdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v1qNode != 0) && (here-> BSIM3v1sNodePrime != 0)) {
			while (here->BSIM3v1QspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM3v1QspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v1qNode != 0) && (here-> BSIM3v1gNode != 0)) {
			while (here->BSIM3v1QgPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM3v1QgPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v1qNode != 0) && (here-> BSIM3v1bNode != 0)) {
			while (here->BSIM3v1QbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM3v1QbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v1dNodePrime != 0) && (here-> BSIM3v1qNode != 0)) {
			while (here->BSIM3v1DPqPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM3v1DPqPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v1sNodePrime != 0) && (here-> BSIM3v1qNode != 0)) {
			while (here->BSIM3v1SPqPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM3v1SPqPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v1gNode != 0) && (here-> BSIM3v1qNode != 0)) {
			while (here->BSIM3v1GqPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM3v1GqPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v1bNode != 0) && (here-> BSIM3v1qNode != 0)) {
			while (here->BSIM3v1BqPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM3v1BqPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
	}
    }
    return(OK);
}

int
BSIM3v1bindCSCComplex(GENmodel *inModel, CKTcircuit *ckt)
{
    BSIM3v1model *model = (BSIM3v1model *)inModel;
    int i ;

    /*  loop through all the b3v1 models */
    for( ; model != NULL; model = model->BSIM3v1nextModel ) {
	BSIM3v1instance *here;

        /* loop through all the instances of the model */
        for (here = model->BSIM3v1instances; here != NULL ;
	    here = here->BSIM3v1nextInstance) {

		i = 0 ;
		if ((here-> BSIM3v1dNode != 0) && (here-> BSIM3v1dNode != 0)) {
			while (here->BSIM3v1DdPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM3v1DdPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v1gNode != 0) && (here-> BSIM3v1gNode != 0)) {
			while (here->BSIM3v1GgPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM3v1GgPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v1sNode != 0) && (here-> BSIM3v1sNode != 0)) {
			while (here->BSIM3v1SsPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM3v1SsPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v1bNode != 0) && (here-> BSIM3v1bNode != 0)) {
			while (here->BSIM3v1BbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM3v1BbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v1dNodePrime != 0) && (here-> BSIM3v1dNodePrime != 0)) {
			while (here->BSIM3v1DPdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM3v1DPdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v1sNodePrime != 0) && (here-> BSIM3v1sNodePrime != 0)) {
			while (here->BSIM3v1SPspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM3v1SPspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v1dNode != 0) && (here-> BSIM3v1dNodePrime != 0)) {
			while (here->BSIM3v1DdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM3v1DdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v1gNode != 0) && (here-> BSIM3v1bNode != 0)) {
			while (here->BSIM3v1GbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM3v1GbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v1gNode != 0) && (here-> BSIM3v1dNodePrime != 0)) {
			while (here->BSIM3v1GdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM3v1GdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v1gNode != 0) && (here-> BSIM3v1sNodePrime != 0)) {
			while (here->BSIM3v1GspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM3v1GspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v1sNode != 0) && (here-> BSIM3v1sNodePrime != 0)) {
			while (here->BSIM3v1SspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM3v1SspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v1bNode != 0) && (here-> BSIM3v1dNodePrime != 0)) {
			while (here->BSIM3v1BdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM3v1BdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v1bNode != 0) && (here-> BSIM3v1sNodePrime != 0)) {
			while (here->BSIM3v1BspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM3v1BspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v1dNodePrime != 0) && (here-> BSIM3v1sNodePrime != 0)) {
			while (here->BSIM3v1DPspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM3v1DPspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v1dNodePrime != 0) && (here-> BSIM3v1dNode != 0)) {
			while (here->BSIM3v1DPdPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM3v1DPdPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v1bNode != 0) && (here-> BSIM3v1gNode != 0)) {
			while (here->BSIM3v1BgPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM3v1BgPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v1dNodePrime != 0) && (here-> BSIM3v1gNode != 0)) {
			while (here->BSIM3v1DPgPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM3v1DPgPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v1sNodePrime != 0) && (here-> BSIM3v1gNode != 0)) {
			while (here->BSIM3v1SPgPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM3v1SPgPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v1sNodePrime != 0) && (here-> BSIM3v1sNode != 0)) {
			while (here->BSIM3v1SPsPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM3v1SPsPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v1dNodePrime != 0) && (here-> BSIM3v1bNode != 0)) {
			while (here->BSIM3v1DPbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM3v1DPbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v1sNodePrime != 0) && (here-> BSIM3v1bNode != 0)) {
			while (here->BSIM3v1SPbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM3v1SPbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v1sNodePrime != 0) && (here-> BSIM3v1dNodePrime != 0)) {
			while (here->BSIM3v1SPdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM3v1SPdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v1qNode != 0) && (here-> BSIM3v1qNode != 0)) {
			while (here->BSIM3v1QqPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM3v1QqPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v1qNode != 0) && (here-> BSIM3v1dNodePrime != 0)) {
			while (here->BSIM3v1QdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM3v1QdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v1qNode != 0) && (here-> BSIM3v1sNodePrime != 0)) {
			while (here->BSIM3v1QspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM3v1QspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v1qNode != 0) && (here-> BSIM3v1gNode != 0)) {
			while (here->BSIM3v1QgPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM3v1QgPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v1qNode != 0) && (here-> BSIM3v1bNode != 0)) {
			while (here->BSIM3v1QbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM3v1QbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v1dNodePrime != 0) && (here-> BSIM3v1qNode != 0)) {
			while (here->BSIM3v1DPqPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM3v1DPqPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v1sNodePrime != 0) && (here-> BSIM3v1qNode != 0)) {
			while (here->BSIM3v1SPqPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM3v1SPqPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v1gNode != 0) && (here-> BSIM3v1qNode != 0)) {
			while (here->BSIM3v1GqPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM3v1GqPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v1bNode != 0) && (here-> BSIM3v1qNode != 0)) {
			while (here->BSIM3v1BqPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM3v1BqPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
	}
    }
    return(OK);
}

int
BSIM3v1bindCSCComplexToReal (GENmodel *inModel, CKTcircuit *ckt)
{
    BSIM3v1model *model = (BSIM3v1model *)inModel ;
    BSIM3v1instance *here ;
    int i ;

    /*  loop through all the bsim3v1 models */
    for ( ; model != NULL ; model = model->BSIM3v1nextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->BSIM3v1instances ; here != NULL ; here = here->BSIM3v1nextInstance)
        {
            i = 0 ;
            if ((here->BSIM3v1dNode != 0) && (here->BSIM3v1dNode != 0))
            {
                while (here->BSIM3v1DdPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM3v1DdPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM3v1gNode != 0) && (here->BSIM3v1gNode != 0))
            {
                while (here->BSIM3v1GgPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM3v1GgPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM3v1sNode != 0) && (here->BSIM3v1sNode != 0))
            {
                while (here->BSIM3v1SsPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM3v1SsPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM3v1bNode != 0) && (here->BSIM3v1bNode != 0))
            {
                while (here->BSIM3v1BbPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM3v1BbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM3v1dNodePrime != 0) && (here->BSIM3v1dNodePrime != 0))
            {
                while (here->BSIM3v1DPdpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM3v1DPdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM3v1sNodePrime != 0) && (here->BSIM3v1sNodePrime != 0))
            {
                while (here->BSIM3v1SPspPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM3v1SPspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM3v1dNode != 0) && (here->BSIM3v1dNodePrime != 0))
            {
                while (here->BSIM3v1DdpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM3v1DdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM3v1gNode != 0) && (here->BSIM3v1bNode != 0))
            {
                while (here->BSIM3v1GbPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM3v1GbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM3v1gNode != 0) && (here->BSIM3v1dNodePrime != 0))
            {
                while (here->BSIM3v1GdpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM3v1GdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM3v1gNode != 0) && (here->BSIM3v1sNodePrime != 0))
            {
                while (here->BSIM3v1GspPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM3v1GspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM3v1sNode != 0) && (here->BSIM3v1sNodePrime != 0))
            {
                while (here->BSIM3v1SspPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM3v1SspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM3v1bNode != 0) && (here->BSIM3v1dNodePrime != 0))
            {
                while (here->BSIM3v1BdpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM3v1BdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM3v1bNode != 0) && (here->BSIM3v1sNodePrime != 0))
            {
                while (here->BSIM3v1BspPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM3v1BspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM3v1dNodePrime != 0) && (here->BSIM3v1sNodePrime != 0))
            {
                while (here->BSIM3v1DPspPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM3v1DPspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM3v1dNodePrime != 0) && (here->BSIM3v1dNode != 0))
            {
                while (here->BSIM3v1DPdPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM3v1DPdPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM3v1bNode != 0) && (here->BSIM3v1gNode != 0))
            {
                while (here->BSIM3v1BgPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM3v1BgPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM3v1dNodePrime != 0) && (here->BSIM3v1gNode != 0))
            {
                while (here->BSIM3v1DPgPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM3v1DPgPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM3v1sNodePrime != 0) && (here->BSIM3v1gNode != 0))
            {
                while (here->BSIM3v1SPgPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM3v1SPgPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM3v1sNodePrime != 0) && (here->BSIM3v1sNode != 0))
            {
                while (here->BSIM3v1SPsPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM3v1SPsPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM3v1dNodePrime != 0) && (here->BSIM3v1bNode != 0))
            {
                while (here->BSIM3v1DPbPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM3v1DPbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM3v1sNodePrime != 0) && (here->BSIM3v1bNode != 0))
            {
                while (here->BSIM3v1SPbPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM3v1SPbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM3v1sNodePrime != 0) && (here->BSIM3v1dNodePrime != 0))
            {
                while (here->BSIM3v1SPdpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM3v1SPdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM3v1qNode != 0) && (here->BSIM3v1qNode != 0))
            {
                while (here->BSIM3v1QqPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM3v1QqPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM3v1qNode != 0) && (here->BSIM3v1dNodePrime != 0))
            {
                while (here->BSIM3v1QdpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM3v1QdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM3v1qNode != 0) && (here->BSIM3v1sNodePrime != 0))
            {
                while (here->BSIM3v1QspPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM3v1QspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM3v1qNode != 0) && (here->BSIM3v1gNode != 0))
            {
                while (here->BSIM3v1QgPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM3v1QgPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM3v1qNode != 0) && (here->BSIM3v1bNode != 0))
            {
                while (here->BSIM3v1QbPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM3v1QbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM3v1dNodePrime != 0) && (here->BSIM3v1qNode != 0))
            {
                while (here->BSIM3v1DPqPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM3v1DPqPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM3v1sNodePrime != 0) && (here->BSIM3v1qNode != 0))
            {
                while (here->BSIM3v1SPqPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM3v1SPqPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM3v1gNode != 0) && (here->BSIM3v1qNode != 0))
            {
                while (here->BSIM3v1GqPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM3v1GqPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM3v1bNode != 0) && (here->BSIM3v1qNode != 0))
            {
                while (here->BSIM3v1BqPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM3v1BqPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }
        }
    }

    return (OK) ;
}