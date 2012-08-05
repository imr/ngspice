/**********
Author: 2012 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "bsim3v32def.h"
#include "ngspice/sperror.h"

int
BSIM3v32bindCSC(GENmodel *inModel, CKTcircuit *ckt)
{
    BSIM3v32model *model = (BSIM3v32model *)inModel;
    int i ;

    /*  loop through all the b3v32 models */
    for( ; model != NULL; model = model->BSIM3v32nextModel ) {
	BSIM3v32instance *here;

        /* loop through all the instances of the model */
        for (here = model->BSIM3v32instances; here != NULL ;
	    here = here->BSIM3v32nextInstance) {

		i = 0 ;
		if ((here-> BSIM3v32dNode != 0) && (here-> BSIM3v32dNode != 0)) {
			while (here->BSIM3v32DdPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM3v32DdPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v32gNode != 0) && (here-> BSIM3v32gNode != 0)) {
			while (here->BSIM3v32GgPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM3v32GgPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v32sNode != 0) && (here-> BSIM3v32sNode != 0)) {
			while (here->BSIM3v32SsPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM3v32SsPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v32bNode != 0) && (here-> BSIM3v32bNode != 0)) {
			while (here->BSIM3v32BbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM3v32BbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v32dNodePrime != 0) && (here-> BSIM3v32dNodePrime != 0)) {
			while (here->BSIM3v32DPdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM3v32DPdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v32sNodePrime != 0) && (here-> BSIM3v32sNodePrime != 0)) {
			while (here->BSIM3v32SPspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM3v32SPspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v32dNode != 0) && (here-> BSIM3v32dNodePrime != 0)) {
			while (here->BSIM3v32DdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM3v32DdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v32gNode != 0) && (here-> BSIM3v32bNode != 0)) {
			while (here->BSIM3v32GbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM3v32GbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v32gNode != 0) && (here-> BSIM3v32dNodePrime != 0)) {
			while (here->BSIM3v32GdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM3v32GdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v32gNode != 0) && (here-> BSIM3v32sNodePrime != 0)) {
			while (here->BSIM3v32GspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM3v32GspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v32sNode != 0) && (here-> BSIM3v32sNodePrime != 0)) {
			while (here->BSIM3v32SspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM3v32SspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v32bNode != 0) && (here-> BSIM3v32dNodePrime != 0)) {
			while (here->BSIM3v32BdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM3v32BdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v32bNode != 0) && (here-> BSIM3v32sNodePrime != 0)) {
			while (here->BSIM3v32BspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM3v32BspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v32dNodePrime != 0) && (here-> BSIM3v32sNodePrime != 0)) {
			while (here->BSIM3v32DPspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM3v32DPspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v32dNodePrime != 0) && (here-> BSIM3v32dNode != 0)) {
			while (here->BSIM3v32DPdPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM3v32DPdPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v32bNode != 0) && (here-> BSIM3v32gNode != 0)) {
			while (here->BSIM3v32BgPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM3v32BgPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v32dNodePrime != 0) && (here-> BSIM3v32gNode != 0)) {
			while (here->BSIM3v32DPgPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM3v32DPgPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v32sNodePrime != 0) && (here-> BSIM3v32gNode != 0)) {
			while (here->BSIM3v32SPgPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM3v32SPgPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v32sNodePrime != 0) && (here-> BSIM3v32sNode != 0)) {
			while (here->BSIM3v32SPsPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM3v32SPsPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v32dNodePrime != 0) && (here-> BSIM3v32bNode != 0)) {
			while (here->BSIM3v32DPbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM3v32DPbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v32sNodePrime != 0) && (here-> BSIM3v32bNode != 0)) {
			while (here->BSIM3v32SPbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM3v32SPbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v32sNodePrime != 0) && (here-> BSIM3v32dNodePrime != 0)) {
			while (here->BSIM3v32SPdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM3v32SPdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v32qNode != 0) && (here-> BSIM3v32qNode != 0)) {
			while (here->BSIM3v32QqPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM3v32QqPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v32qNode != 0) && (here-> BSIM3v32dNodePrime != 0)) {
			while (here->BSIM3v32QdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM3v32QdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v32qNode != 0) && (here-> BSIM3v32sNodePrime != 0)) {
			while (here->BSIM3v32QspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM3v32QspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v32qNode != 0) && (here-> BSIM3v32gNode != 0)) {
			while (here->BSIM3v32QgPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM3v32QgPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v32qNode != 0) && (here-> BSIM3v32bNode != 0)) {
			while (here->BSIM3v32QbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM3v32QbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v32dNodePrime != 0) && (here-> BSIM3v32qNode != 0)) {
			while (here->BSIM3v32DPqPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM3v32DPqPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v32sNodePrime != 0) && (here-> BSIM3v32qNode != 0)) {
			while (here->BSIM3v32SPqPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM3v32SPqPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v32gNode != 0) && (here-> BSIM3v32qNode != 0)) {
			while (here->BSIM3v32GqPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM3v32GqPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v32bNode != 0) && (here-> BSIM3v32qNode != 0)) {
			while (here->BSIM3v32BqPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM3v32BqPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
	}
    }
    return(OK);
}

int
BSIM3v32bindCSCComplex(GENmodel *inModel, CKTcircuit *ckt)
{
    BSIM3v32model *model = (BSIM3v32model *)inModel;
    int i ;

    /*  loop through all the b3v32 models */
    for( ; model != NULL; model = model->BSIM3v32nextModel ) {
	BSIM3v32instance *here;

        /* loop through all the instances of the model */
        for (here = model->BSIM3v32instances; here != NULL ;
	    here = here->BSIM3v32nextInstance) {

		i = 0 ;
		if ((here-> BSIM3v32dNode != 0) && (here-> BSIM3v32dNode != 0)) {
			while (here->BSIM3v32DdPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM3v32DdPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v32gNode != 0) && (here-> BSIM3v32gNode != 0)) {
			while (here->BSIM3v32GgPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM3v32GgPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v32sNode != 0) && (here-> BSIM3v32sNode != 0)) {
			while (here->BSIM3v32SsPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM3v32SsPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v32bNode != 0) && (here-> BSIM3v32bNode != 0)) {
			while (here->BSIM3v32BbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM3v32BbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v32dNodePrime != 0) && (here-> BSIM3v32dNodePrime != 0)) {
			while (here->BSIM3v32DPdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM3v32DPdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v32sNodePrime != 0) && (here-> BSIM3v32sNodePrime != 0)) {
			while (here->BSIM3v32SPspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM3v32SPspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v32dNode != 0) && (here-> BSIM3v32dNodePrime != 0)) {
			while (here->BSIM3v32DdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM3v32DdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v32gNode != 0) && (here-> BSIM3v32bNode != 0)) {
			while (here->BSIM3v32GbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM3v32GbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v32gNode != 0) && (here-> BSIM3v32dNodePrime != 0)) {
			while (here->BSIM3v32GdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM3v32GdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v32gNode != 0) && (here-> BSIM3v32sNodePrime != 0)) {
			while (here->BSIM3v32GspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM3v32GspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v32sNode != 0) && (here-> BSIM3v32sNodePrime != 0)) {
			while (here->BSIM3v32SspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM3v32SspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v32bNode != 0) && (here-> BSIM3v32dNodePrime != 0)) {
			while (here->BSIM3v32BdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM3v32BdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v32bNode != 0) && (here-> BSIM3v32sNodePrime != 0)) {
			while (here->BSIM3v32BspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM3v32BspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v32dNodePrime != 0) && (here-> BSIM3v32sNodePrime != 0)) {
			while (here->BSIM3v32DPspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM3v32DPspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v32dNodePrime != 0) && (here-> BSIM3v32dNode != 0)) {
			while (here->BSIM3v32DPdPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM3v32DPdPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v32bNode != 0) && (here-> BSIM3v32gNode != 0)) {
			while (here->BSIM3v32BgPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM3v32BgPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v32dNodePrime != 0) && (here-> BSIM3v32gNode != 0)) {
			while (here->BSIM3v32DPgPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM3v32DPgPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v32sNodePrime != 0) && (here-> BSIM3v32gNode != 0)) {
			while (here->BSIM3v32SPgPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM3v32SPgPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v32sNodePrime != 0) && (here-> BSIM3v32sNode != 0)) {
			while (here->BSIM3v32SPsPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM3v32SPsPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v32dNodePrime != 0) && (here-> BSIM3v32bNode != 0)) {
			while (here->BSIM3v32DPbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM3v32DPbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v32sNodePrime != 0) && (here-> BSIM3v32bNode != 0)) {
			while (here->BSIM3v32SPbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM3v32SPbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v32sNodePrime != 0) && (here-> BSIM3v32dNodePrime != 0)) {
			while (here->BSIM3v32SPdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM3v32SPdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v32qNode != 0) && (here-> BSIM3v32qNode != 0)) {
			while (here->BSIM3v32QqPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM3v32QqPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v32qNode != 0) && (here-> BSIM3v32dNodePrime != 0)) {
			while (here->BSIM3v32QdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM3v32QdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v32qNode != 0) && (here-> BSIM3v32sNodePrime != 0)) {
			while (here->BSIM3v32QspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM3v32QspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v32qNode != 0) && (here-> BSIM3v32gNode != 0)) {
			while (here->BSIM3v32QgPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM3v32QgPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v32qNode != 0) && (here-> BSIM3v32bNode != 0)) {
			while (here->BSIM3v32QbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM3v32QbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v32dNodePrime != 0) && (here-> BSIM3v32qNode != 0)) {
			while (here->BSIM3v32DPqPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM3v32DPqPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v32sNodePrime != 0) && (here-> BSIM3v32qNode != 0)) {
			while (here->BSIM3v32SPqPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM3v32SPqPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v32gNode != 0) && (here-> BSIM3v32qNode != 0)) {
			while (here->BSIM3v32GqPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM3v32GqPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM3v32bNode != 0) && (here-> BSIM3v32qNode != 0)) {
			while (here->BSIM3v32BqPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM3v32BqPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
	}
    }
    return(OK);
}

int
BSIM3v32bindCSCComplexToReal (GENmodel *inModel, CKTcircuit *ckt)
{
    BSIM3v32model *model = (BSIM3v32model *)inModel ;
    BSIM3v32instance *here ;
    int i ;

    /*  loop through all the bsim3v32 models */
    for ( ; model != NULL ; model = model->BSIM3v32nextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->BSIM3v32instances ; here != NULL ; here = here->BSIM3v32nextInstance)
        {
            i = 0 ;
            if ((here->BSIM3v32dNode != 0) && (here->BSIM3v32dNode != 0))
            {
                while (here->BSIM3v32DdPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM3v32DdPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM3v32gNode != 0) && (here->BSIM3v32gNode != 0))
            {
                while (here->BSIM3v32GgPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM3v32GgPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM3v32sNode != 0) && (here->BSIM3v32sNode != 0))
            {
                while (here->BSIM3v32SsPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM3v32SsPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM3v32bNode != 0) && (here->BSIM3v32bNode != 0))
            {
                while (here->BSIM3v32BbPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM3v32BbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM3v32dNodePrime != 0) && (here->BSIM3v32dNodePrime != 0))
            {
                while (here->BSIM3v32DPdpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM3v32DPdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM3v32sNodePrime != 0) && (here->BSIM3v32sNodePrime != 0))
            {
                while (here->BSIM3v32SPspPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM3v32SPspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM3v32dNode != 0) && (here->BSIM3v32dNodePrime != 0))
            {
                while (here->BSIM3v32DdpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM3v32DdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM3v32gNode != 0) && (here->BSIM3v32bNode != 0))
            {
                while (here->BSIM3v32GbPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM3v32GbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM3v32gNode != 0) && (here->BSIM3v32dNodePrime != 0))
            {
                while (here->BSIM3v32GdpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM3v32GdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM3v32gNode != 0) && (here->BSIM3v32sNodePrime != 0))
            {
                while (here->BSIM3v32GspPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM3v32GspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM3v32sNode != 0) && (here->BSIM3v32sNodePrime != 0))
            {
                while (here->BSIM3v32SspPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM3v32SspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM3v32bNode != 0) && (here->BSIM3v32dNodePrime != 0))
            {
                while (here->BSIM3v32BdpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM3v32BdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM3v32bNode != 0) && (here->BSIM3v32sNodePrime != 0))
            {
                while (here->BSIM3v32BspPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM3v32BspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM3v32dNodePrime != 0) && (here->BSIM3v32sNodePrime != 0))
            {
                while (here->BSIM3v32DPspPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM3v32DPspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM3v32dNodePrime != 0) && (here->BSIM3v32dNode != 0))
            {
                while (here->BSIM3v32DPdPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM3v32DPdPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM3v32bNode != 0) && (here->BSIM3v32gNode != 0))
            {
                while (here->BSIM3v32BgPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM3v32BgPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM3v32dNodePrime != 0) && (here->BSIM3v32gNode != 0))
            {
                while (here->BSIM3v32DPgPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM3v32DPgPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM3v32sNodePrime != 0) && (here->BSIM3v32gNode != 0))
            {
                while (here->BSIM3v32SPgPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM3v32SPgPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM3v32sNodePrime != 0) && (here->BSIM3v32sNode != 0))
            {
                while (here->BSIM3v32SPsPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM3v32SPsPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM3v32dNodePrime != 0) && (here->BSIM3v32bNode != 0))
            {
                while (here->BSIM3v32DPbPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM3v32DPbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM3v32sNodePrime != 0) && (here->BSIM3v32bNode != 0))
            {
                while (here->BSIM3v32SPbPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM3v32SPbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM3v32sNodePrime != 0) && (here->BSIM3v32dNodePrime != 0))
            {
                while (here->BSIM3v32SPdpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM3v32SPdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM3v32qNode != 0) && (here->BSIM3v32qNode != 0))
            {
                while (here->BSIM3v32QqPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM3v32QqPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM3v32qNode != 0) && (here->BSIM3v32dNodePrime != 0))
            {
                while (here->BSIM3v32QdpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM3v32QdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM3v32qNode != 0) && (here->BSIM3v32sNodePrime != 0))
            {
                while (here->BSIM3v32QspPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM3v32QspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM3v32qNode != 0) && (here->BSIM3v32gNode != 0))
            {
                while (here->BSIM3v32QgPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM3v32QgPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM3v32qNode != 0) && (here->BSIM3v32bNode != 0))
            {
                while (here->BSIM3v32QbPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM3v32QbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM3v32dNodePrime != 0) && (here->BSIM3v32qNode != 0))
            {
                while (here->BSIM3v32DPqPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM3v32DPqPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM3v32sNodePrime != 0) && (here->BSIM3v32qNode != 0))
            {
                while (here->BSIM3v32SPqPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM3v32SPqPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM3v32gNode != 0) && (here->BSIM3v32qNode != 0))
            {
                while (here->BSIM3v32GqPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM3v32GqPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM3v32bNode != 0) && (here->BSIM3v32qNode != 0))
            {
                while (here->BSIM3v32BqPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM3v32BqPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }
        }
    }

    return (OK) ;
}