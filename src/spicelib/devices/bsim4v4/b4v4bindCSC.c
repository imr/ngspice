/**********
Author: 2012 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "bsim4v4def.h"
#include "ngspice/sperror.h"

int
BSIM4v4bindCSC(GENmodel *inModel, CKTcircuit *ckt)
{
    BSIM4v4model *model = (BSIM4v4model *)inModel;
    int i ;

    /*  loop through all the b4v4 models */
    for( ; model != NULL; model = model->BSIM4v4nextModel ) {
	BSIM4v4instance *here;

        /* loop through all the instances of the model */
        for (here = model->BSIM4v4instances; here != NULL ;
	    here = here->BSIM4v4nextInstance) {

		i = 0 ;
		if ((here-> BSIM4v4dNodePrime != 0) && (here-> BSIM4v4bNodePrime != 0)) {
			while (here->BSIM4v4DPbpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v4DPbpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v4gNodePrime != 0) && (here-> BSIM4v4bNodePrime != 0)) {
			while (here->BSIM4v4GPbpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v4GPbpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v4sNodePrime != 0) && (here-> BSIM4v4bNodePrime != 0)) {
			while (here->BSIM4v4SPbpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v4SPbpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v4bNodePrime != 0) && (here-> BSIM4v4dNodePrime != 0)) {
			while (here->BSIM4v4BPdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v4BPdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v4bNodePrime != 0) && (here-> BSIM4v4gNodePrime != 0)) {
			while (here->BSIM4v4BPgpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v4BPgpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v4bNodePrime != 0) && (here-> BSIM4v4sNodePrime != 0)) {
			while (here->BSIM4v4BPspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v4BPspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v4bNodePrime != 0) && (here-> BSIM4v4bNodePrime != 0)) {
			while (here->BSIM4v4BPbpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v4BPbpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v4dNode != 0) && (here-> BSIM4v4dNode != 0)) {
			while (here->BSIM4v4DdPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v4DdPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v4gNodePrime != 0) && (here-> BSIM4v4gNodePrime != 0)) {
			while (here->BSIM4v4GPgpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v4GPgpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v4sNode != 0) && (here-> BSIM4v4sNode != 0)) {
			while (here->BSIM4v4SsPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v4SsPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v4dNodePrime != 0) && (here-> BSIM4v4dNodePrime != 0)) {
			while (here->BSIM4v4DPdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v4DPdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v4sNodePrime != 0) && (here-> BSIM4v4sNodePrime != 0)) {
			while (here->BSIM4v4SPspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v4SPspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v4dNode != 0) && (here-> BSIM4v4dNodePrime != 0)) {
			while (here->BSIM4v4DdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v4DdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v4gNodePrime != 0) && (here-> BSIM4v4dNodePrime != 0)) {
			while (here->BSIM4v4GPdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v4GPdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v4gNodePrime != 0) && (here-> BSIM4v4sNodePrime != 0)) {
			while (here->BSIM4v4GPspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v4GPspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v4sNode != 0) && (here-> BSIM4v4sNodePrime != 0)) {
			while (here->BSIM4v4SspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v4SspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v4dNodePrime != 0) && (here-> BSIM4v4sNodePrime != 0)) {
			while (here->BSIM4v4DPspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v4DPspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v4dNodePrime != 0) && (here-> BSIM4v4dNode != 0)) {
			while (here->BSIM4v4DPdPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v4DPdPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v4dNodePrime != 0) && (here-> BSIM4v4gNodePrime != 0)) {
			while (here->BSIM4v4DPgpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v4DPgpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v4sNodePrime != 0) && (here-> BSIM4v4gNodePrime != 0)) {
			while (here->BSIM4v4SPgpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v4SPgpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v4sNodePrime != 0) && (here-> BSIM4v4sNode != 0)) {
			while (here->BSIM4v4SPsPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v4SPsPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v4sNodePrime != 0) && (here-> BSIM4v4dNodePrime != 0)) {
			while (here->BSIM4v4SPdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v4SPdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v4qNode != 0) && (here-> BSIM4v4qNode != 0)) {
			while (here->BSIM4v4QqPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v4QqPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v4qNode != 0) && (here-> BSIM4v4bNodePrime != 0)) {
			while (here->BSIM4v4QbpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v4QbpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v4qNode != 0) && (here-> BSIM4v4dNodePrime != 0)) {
			while (here->BSIM4v4QdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v4QdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v4qNode != 0) && (here-> BSIM4v4sNodePrime != 0)) {
			while (here->BSIM4v4QspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v4QspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v4qNode != 0) && (here-> BSIM4v4gNodePrime != 0)) {
			while (here->BSIM4v4QgpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v4QgpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v4dNodePrime != 0) && (here-> BSIM4v4qNode != 0)) {
			while (here->BSIM4v4DPqPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v4DPqPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v4sNodePrime != 0) && (here-> BSIM4v4qNode != 0)) {
			while (here->BSIM4v4SPqPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v4SPqPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v4gNodePrime != 0) && (here-> BSIM4v4qNode != 0)) {
			while (here->BSIM4v4GPqPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v4GPqPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

                if (here->BSIM4v4rgateMod != 0) {

		i = 0 ;
		if ((here-> BSIM4v4gNodeExt != 0) && (here-> BSIM4v4gNodeExt != 0)) {
			while (here->BSIM4v4GEgePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v4GEgePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v4gNodeExt != 0) && (here-> BSIM4v4gNodePrime != 0)) {
			while (here->BSIM4v4GEgpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v4GEgpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v4gNodePrime != 0) && (here-> BSIM4v4gNodeExt != 0)) {
			while (here->BSIM4v4GPgePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v4GPgePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v4gNodeExt != 0) && (here-> BSIM4v4dNodePrime != 0)) {
			while (here->BSIM4v4GEdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v4GEdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v4gNodeExt != 0) && (here-> BSIM4v4sNodePrime != 0)) {
			while (here->BSIM4v4GEspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v4GEspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v4gNodeExt != 0) && (here-> BSIM4v4bNodePrime != 0)) {
			while (here->BSIM4v4GEbpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v4GEbpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v4gNodeMid != 0) && (here-> BSIM4v4dNodePrime != 0)) {
			while (here->BSIM4v4GMdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v4GMdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v4gNodeMid != 0) && (here-> BSIM4v4gNodePrime != 0)) {
			while (here->BSIM4v4GMgpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v4GMgpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v4gNodeMid != 0) && (here-> BSIM4v4gNodeMid != 0)) {
			while (here->BSIM4v4GMgmPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v4GMgmPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v4gNodeMid != 0) && (here-> BSIM4v4gNodeExt != 0)) {
			while (here->BSIM4v4GMgePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v4GMgePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v4gNodeMid != 0) && (here-> BSIM4v4sNodePrime != 0)) {
			while (here->BSIM4v4GMspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v4GMspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v4gNodeMid != 0) && (here-> BSIM4v4bNodePrime != 0)) {
			while (here->BSIM4v4GMbpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v4GMbpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v4dNodePrime != 0) && (here-> BSIM4v4gNodeMid != 0)) {
			while (here->BSIM4v4DPgmPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v4DPgmPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v4gNodePrime != 0) && (here-> BSIM4v4gNodeMid != 0)) {
			while (here->BSIM4v4GPgmPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v4GPgmPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v4gNodeExt != 0) && (here-> BSIM4v4gNodeMid != 0)) {
			while (here->BSIM4v4GEgmPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v4GEgmPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v4sNodePrime != 0) && (here-> BSIM4v4gNodeMid != 0)) {
			while (here->BSIM4v4SPgmPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v4SPgmPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v4bNodePrime != 0) && (here-> BSIM4v4gNodeMid != 0)) {
			while (here->BSIM4v4BPgmPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v4BPgmPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
                } /* IF */

                if (here->BSIM4v4rbodyMod) {

		i = 0 ;
		if ((here-> BSIM4v4dNodePrime != 0) && (here-> BSIM4v4dbNode != 0)) {
			while (here->BSIM4v4DPdbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v4DPdbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v4sNodePrime != 0) && (here-> BSIM4v4sbNode != 0)) {
			while (here->BSIM4v4SPsbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v4SPsbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v4dbNode != 0) && (here-> BSIM4v4dNodePrime != 0)) {
			while (here->BSIM4v4DBdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v4DBdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v4dbNode != 0) && (here-> BSIM4v4dbNode != 0)) {
			while (here->BSIM4v4DBdbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v4DBdbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v4dbNode != 0) && (here-> BSIM4v4bNodePrime != 0)) {
			while (here->BSIM4v4DBbpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v4DBbpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v4dbNode != 0) && (here-> BSIM4v4bNode != 0)) {
			while (here->BSIM4v4DBbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v4DBbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v4bNodePrime != 0) && (here-> BSIM4v4dbNode != 0)) {
			while (here->BSIM4v4BPdbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v4BPdbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v4bNodePrime != 0) && (here-> BSIM4v4bNode != 0)) {
			while (here->BSIM4v4BPbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v4BPbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v4bNodePrime != 0) && (here-> BSIM4v4sbNode != 0)) {
			while (here->BSIM4v4BPsbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v4BPsbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v4sbNode != 0) && (here-> BSIM4v4sNodePrime != 0)) {
			while (here->BSIM4v4SBspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v4SBspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v4sbNode != 0) && (here-> BSIM4v4bNodePrime != 0)) {
			while (here->BSIM4v4SBbpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v4SBbpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v4sbNode != 0) && (here-> BSIM4v4bNode != 0)) {
			while (here->BSIM4v4SBbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v4SBbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v4sbNode != 0) && (here-> BSIM4v4sbNode != 0)) {
			while (here->BSIM4v4SBsbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v4SBsbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v4bNode != 0) && (here-> BSIM4v4dbNode != 0)) {
			while (here->BSIM4v4BdbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v4BdbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v4bNode != 0) && (here-> BSIM4v4bNodePrime != 0)) {
			while (here->BSIM4v4BbpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v4BbpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v4bNode != 0) && (here-> BSIM4v4sbNode != 0)) {
			while (here->BSIM4v4BsbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v4BsbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v4bNode != 0) && (here-> BSIM4v4bNode != 0)) {
			while (here->BSIM4v4BbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v4BbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
                } /* IF */

                if (model->BSIM4v4rdsMod) {

		i = 0 ;
		if ((here-> BSIM4v4dNode != 0) && (here-> BSIM4v4gNodePrime != 0)) {
			while (here->BSIM4v4DgpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v4DgpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v4dNode != 0) && (here-> BSIM4v4sNodePrime != 0)) {
			while (here->BSIM4v4DspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v4DspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v4dNode != 0) && (here-> BSIM4v4bNodePrime != 0)) {
			while (here->BSIM4v4DbpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v4DbpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v4sNode != 0) && (here-> BSIM4v4dNodePrime != 0)) {
			while (here->BSIM4v4SdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v4SdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v4sNode != 0) && (here-> BSIM4v4gNodePrime != 0)) {
			while (here->BSIM4v4SgpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v4SgpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v4sNode != 0) && (here-> BSIM4v4bNodePrime != 0)) {
			while (here->BSIM4v4SbpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v4SbpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
                } /* IF */
	}
    }
    return(OK);
}

int
BSIM4v4bindCSCComplex(GENmodel *inModel, CKTcircuit *ckt)
{
    BSIM4v4model *model = (BSIM4v4model *)inModel;
    int i ;

    /*  loop through all the b4v4 models */
    for( ; model != NULL; model = model->BSIM4v4nextModel ) {
	BSIM4v4instance *here;

        /* loop through all the instances of the model */
        for (here = model->BSIM4v4instances; here != NULL ;
	    here = here->BSIM4v4nextInstance) {

		i = 0 ;
		if ((here-> BSIM4v4dNodePrime != 0) && (here-> BSIM4v4bNodePrime != 0)) {
			while (here->BSIM4v4DPbpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v4DPbpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v4gNodePrime != 0) && (here-> BSIM4v4bNodePrime != 0)) {
			while (here->BSIM4v4GPbpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v4GPbpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v4sNodePrime != 0) && (here-> BSIM4v4bNodePrime != 0)) {
			while (here->BSIM4v4SPbpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v4SPbpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v4bNodePrime != 0) && (here-> BSIM4v4dNodePrime != 0)) {
			while (here->BSIM4v4BPdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v4BPdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v4bNodePrime != 0) && (here-> BSIM4v4gNodePrime != 0)) {
			while (here->BSIM4v4BPgpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v4BPgpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v4bNodePrime != 0) && (here-> BSIM4v4sNodePrime != 0)) {
			while (here->BSIM4v4BPspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v4BPspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v4bNodePrime != 0) && (here-> BSIM4v4bNodePrime != 0)) {
			while (here->BSIM4v4BPbpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v4BPbpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v4dNode != 0) && (here-> BSIM4v4dNode != 0)) {
			while (here->BSIM4v4DdPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v4DdPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v4gNodePrime != 0) && (here-> BSIM4v4gNodePrime != 0)) {
			while (here->BSIM4v4GPgpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v4GPgpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v4sNode != 0) && (here-> BSIM4v4sNode != 0)) {
			while (here->BSIM4v4SsPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v4SsPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v4dNodePrime != 0) && (here-> BSIM4v4dNodePrime != 0)) {
			while (here->BSIM4v4DPdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v4DPdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v4sNodePrime != 0) && (here-> BSIM4v4sNodePrime != 0)) {
			while (here->BSIM4v4SPspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v4SPspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v4dNode != 0) && (here-> BSIM4v4dNodePrime != 0)) {
			while (here->BSIM4v4DdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v4DdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v4gNodePrime != 0) && (here-> BSIM4v4dNodePrime != 0)) {
			while (here->BSIM4v4GPdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v4GPdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v4gNodePrime != 0) && (here-> BSIM4v4sNodePrime != 0)) {
			while (here->BSIM4v4GPspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v4GPspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v4sNode != 0) && (here-> BSIM4v4sNodePrime != 0)) {
			while (here->BSIM4v4SspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v4SspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v4dNodePrime != 0) && (here-> BSIM4v4sNodePrime != 0)) {
			while (here->BSIM4v4DPspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v4DPspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v4dNodePrime != 0) && (here-> BSIM4v4dNode != 0)) {
			while (here->BSIM4v4DPdPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v4DPdPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v4dNodePrime != 0) && (here-> BSIM4v4gNodePrime != 0)) {
			while (here->BSIM4v4DPgpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v4DPgpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v4sNodePrime != 0) && (here-> BSIM4v4gNodePrime != 0)) {
			while (here->BSIM4v4SPgpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v4SPgpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v4sNodePrime != 0) && (here-> BSIM4v4sNode != 0)) {
			while (here->BSIM4v4SPsPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v4SPsPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v4sNodePrime != 0) && (here-> BSIM4v4dNodePrime != 0)) {
			while (here->BSIM4v4SPdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v4SPdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v4qNode != 0) && (here-> BSIM4v4qNode != 0)) {
			while (here->BSIM4v4QqPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v4QqPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v4qNode != 0) && (here-> BSIM4v4bNodePrime != 0)) {
			while (here->BSIM4v4QbpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v4QbpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v4qNode != 0) && (here-> BSIM4v4dNodePrime != 0)) {
			while (here->BSIM4v4QdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v4QdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v4qNode != 0) && (here-> BSIM4v4sNodePrime != 0)) {
			while (here->BSIM4v4QspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v4QspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v4qNode != 0) && (here-> BSIM4v4gNodePrime != 0)) {
			while (here->BSIM4v4QgpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v4QgpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v4dNodePrime != 0) && (here-> BSIM4v4qNode != 0)) {
			while (here->BSIM4v4DPqPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v4DPqPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v4sNodePrime != 0) && (here-> BSIM4v4qNode != 0)) {
			while (here->BSIM4v4SPqPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v4SPqPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v4gNodePrime != 0) && (here-> BSIM4v4qNode != 0)) {
			while (here->BSIM4v4GPqPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v4GPqPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

                if (here->BSIM4v4rgateMod != 0) {

		i = 0 ;
		if ((here-> BSIM4v4gNodeExt != 0) && (here-> BSIM4v4gNodeExt != 0)) {
			while (here->BSIM4v4GEgePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v4GEgePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v4gNodeExt != 0) && (here-> BSIM4v4gNodePrime != 0)) {
			while (here->BSIM4v4GEgpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v4GEgpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v4gNodePrime != 0) && (here-> BSIM4v4gNodeExt != 0)) {
			while (here->BSIM4v4GPgePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v4GPgePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v4gNodeExt != 0) && (here-> BSIM4v4dNodePrime != 0)) {
			while (here->BSIM4v4GEdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v4GEdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v4gNodeExt != 0) && (here-> BSIM4v4sNodePrime != 0)) {
			while (here->BSIM4v4GEspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v4GEspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v4gNodeExt != 0) && (here-> BSIM4v4bNodePrime != 0)) {
			while (here->BSIM4v4GEbpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v4GEbpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v4gNodeMid != 0) && (here-> BSIM4v4dNodePrime != 0)) {
			while (here->BSIM4v4GMdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v4GMdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v4gNodeMid != 0) && (here-> BSIM4v4gNodePrime != 0)) {
			while (here->BSIM4v4GMgpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v4GMgpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v4gNodeMid != 0) && (here-> BSIM4v4gNodeMid != 0)) {
			while (here->BSIM4v4GMgmPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v4GMgmPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v4gNodeMid != 0) && (here-> BSIM4v4gNodeExt != 0)) {
			while (here->BSIM4v4GMgePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v4GMgePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v4gNodeMid != 0) && (here-> BSIM4v4sNodePrime != 0)) {
			while (here->BSIM4v4GMspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v4GMspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v4gNodeMid != 0) && (here-> BSIM4v4bNodePrime != 0)) {
			while (here->BSIM4v4GMbpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v4GMbpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v4dNodePrime != 0) && (here-> BSIM4v4gNodeMid != 0)) {
			while (here->BSIM4v4DPgmPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v4DPgmPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v4gNodePrime != 0) && (here-> BSIM4v4gNodeMid != 0)) {
			while (here->BSIM4v4GPgmPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v4GPgmPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v4gNodeExt != 0) && (here-> BSIM4v4gNodeMid != 0)) {
			while (here->BSIM4v4GEgmPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v4GEgmPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v4sNodePrime != 0) && (here-> BSIM4v4gNodeMid != 0)) {
			while (here->BSIM4v4SPgmPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v4SPgmPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v4bNodePrime != 0) && (here-> BSIM4v4gNodeMid != 0)) {
			while (here->BSIM4v4BPgmPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v4BPgmPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
                }

                if (here->BSIM4v4rbodyMod) {

		i = 0 ;
		if ((here-> BSIM4v4dNodePrime != 0) && (here-> BSIM4v4dbNode != 0)) {
			while (here->BSIM4v4DPdbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v4DPdbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v4sNodePrime != 0) && (here-> BSIM4v4sbNode != 0)) {
			while (here->BSIM4v4SPsbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v4SPsbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v4dbNode != 0) && (here-> BSIM4v4dNodePrime != 0)) {
			while (here->BSIM4v4DBdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v4DBdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v4dbNode != 0) && (here-> BSIM4v4dbNode != 0)) {
			while (here->BSIM4v4DBdbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v4DBdbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v4dbNode != 0) && (here-> BSIM4v4bNodePrime != 0)) {
			while (here->BSIM4v4DBbpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v4DBbpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v4dbNode != 0) && (here-> BSIM4v4bNode != 0)) {
			while (here->BSIM4v4DBbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v4DBbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v4bNodePrime != 0) && (here-> BSIM4v4dbNode != 0)) {
			while (here->BSIM4v4BPdbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v4BPdbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v4bNodePrime != 0) && (here-> BSIM4v4bNode != 0)) {
			while (here->BSIM4v4BPbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v4BPbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v4bNodePrime != 0) && (here-> BSIM4v4sbNode != 0)) {
			while (here->BSIM4v4BPsbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v4BPsbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v4sbNode != 0) && (here-> BSIM4v4sNodePrime != 0)) {
			while (here->BSIM4v4SBspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v4SBspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v4sbNode != 0) && (here-> BSIM4v4bNodePrime != 0)) {
			while (here->BSIM4v4SBbpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v4SBbpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v4sbNode != 0) && (here-> BSIM4v4bNode != 0)) {
			while (here->BSIM4v4SBbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v4SBbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v4sbNode != 0) && (here-> BSIM4v4sbNode != 0)) {
			while (here->BSIM4v4SBsbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v4SBsbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v4bNode != 0) && (here-> BSIM4v4dbNode != 0)) {
			while (here->BSIM4v4BdbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v4BdbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v4bNode != 0) && (here-> BSIM4v4bNodePrime != 0)) {
			while (here->BSIM4v4BbpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v4BbpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v4bNode != 0) && (here-> BSIM4v4sbNode != 0)) {
			while (here->BSIM4v4BsbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v4BsbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v4bNode != 0) && (here-> BSIM4v4bNode != 0)) {
			while (here->BSIM4v4BbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v4BbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
                } /* IF */

                if (model->BSIM4v4rdsMod) {

		i = 0 ;
		if ((here-> BSIM4v4dNode != 0) && (here-> BSIM4v4gNodePrime != 0)) {
			while (here->BSIM4v4DgpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v4DgpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v4dNode != 0) && (here-> BSIM4v4sNodePrime != 0)) {
			while (here->BSIM4v4DspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v4DspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v4dNode != 0) && (here-> BSIM4v4bNodePrime != 0)) {
			while (here->BSIM4v4DbpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v4DbpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v4sNode != 0) && (here-> BSIM4v4dNodePrime != 0)) {
			while (here->BSIM4v4SdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v4SdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v4sNode != 0) && (here-> BSIM4v4gNodePrime != 0)) {
			while (here->BSIM4v4SgpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v4SgpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v4sNode != 0) && (here-> BSIM4v4bNodePrime != 0)) {
			while (here->BSIM4v4SbpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v4SbpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
                } /* IF */
	}
    }
    return(OK);
}

int
BSIM4v4bindCSCComplexToReal (GENmodel *inModel, CKTcircuit *ckt)
{
    BSIM4v4model *model = (BSIM4v4model *)inModel ;
    BSIM4v4instance *here ;
    int i ;

    /*  loop through all the bsim4v4 models */
    for ( ; model != NULL ; model = model->BSIM4v4nextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->BSIM4v4instances ; here != NULL ; here = here->BSIM4v4nextInstance)
        {
            i = 0 ;
            if ((here->BSIM4v4dNodePrime != 0) && (here->BSIM4v4bNodePrime != 0))
            {
                while (here->BSIM4v4DPbpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v4DPbpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v4gNodePrime != 0) && (here->BSIM4v4bNodePrime != 0))
            {
                while (here->BSIM4v4GPbpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v4GPbpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v4sNodePrime != 0) && (here->BSIM4v4bNodePrime != 0))
            {
                while (here->BSIM4v4SPbpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v4SPbpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v4bNodePrime != 0) && (here->BSIM4v4dNodePrime != 0))
            {
                while (here->BSIM4v4BPdpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v4BPdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v4bNodePrime != 0) && (here->BSIM4v4gNodePrime != 0))
            {
                while (here->BSIM4v4BPgpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v4BPgpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v4bNodePrime != 0) && (here->BSIM4v4sNodePrime != 0))
            {
                while (here->BSIM4v4BPspPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v4BPspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v4bNodePrime != 0) && (here->BSIM4v4bNodePrime != 0))
            {
                while (here->BSIM4v4BPbpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v4BPbpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v4dNode != 0) && (here->BSIM4v4dNode != 0))
            {
                while (here->BSIM4v4DdPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v4DdPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v4gNodePrime != 0) && (here->BSIM4v4gNodePrime != 0))
            {
                while (here->BSIM4v4GPgpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v4GPgpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v4sNode != 0) && (here->BSIM4v4sNode != 0))
            {
                while (here->BSIM4v4SsPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v4SsPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v4dNodePrime != 0) && (here->BSIM4v4dNodePrime != 0))
            {
                while (here->BSIM4v4DPdpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v4DPdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v4sNodePrime != 0) && (here->BSIM4v4sNodePrime != 0))
            {
                while (here->BSIM4v4SPspPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v4SPspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v4dNode != 0) && (here->BSIM4v4dNodePrime != 0))
            {
                while (here->BSIM4v4DdpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v4DdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v4gNodePrime != 0) && (here->BSIM4v4dNodePrime != 0))
            {
                while (here->BSIM4v4GPdpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v4GPdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v4gNodePrime != 0) && (here->BSIM4v4sNodePrime != 0))
            {
                while (here->BSIM4v4GPspPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v4GPspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v4sNode != 0) && (here->BSIM4v4sNodePrime != 0))
            {
                while (here->BSIM4v4SspPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v4SspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v4dNodePrime != 0) && (here->BSIM4v4sNodePrime != 0))
            {
                while (here->BSIM4v4DPspPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v4DPspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v4dNodePrime != 0) && (here->BSIM4v4dNode != 0))
            {
                while (here->BSIM4v4DPdPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v4DPdPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v4dNodePrime != 0) && (here->BSIM4v4gNodePrime != 0))
            {
                while (here->BSIM4v4DPgpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v4DPgpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v4sNodePrime != 0) && (here->BSIM4v4gNodePrime != 0))
            {
                while (here->BSIM4v4SPgpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v4SPgpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v4sNodePrime != 0) && (here->BSIM4v4sNode != 0))
            {
                while (here->BSIM4v4SPsPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v4SPsPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v4sNodePrime != 0) && (here->BSIM4v4dNodePrime != 0))
            {
                while (here->BSIM4v4SPdpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v4SPdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v4qNode != 0) && (here->BSIM4v4qNode != 0))
            {
                while (here->BSIM4v4QqPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v4QqPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v4qNode != 0) && (here->BSIM4v4bNodePrime != 0))
            {
                while (here->BSIM4v4QbpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v4QbpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v4qNode != 0) && (here->BSIM4v4dNodePrime != 0))
            {
                while (here->BSIM4v4QdpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v4QdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v4qNode != 0) && (here->BSIM4v4sNodePrime != 0))
            {
                while (here->BSIM4v4QspPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v4QspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v4qNode != 0) && (here->BSIM4v4gNodePrime != 0))
            {
                while (here->BSIM4v4QgpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v4QgpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v4dNodePrime != 0) && (here->BSIM4v4qNode != 0))
            {
                while (here->BSIM4v4DPqPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v4DPqPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v4sNodePrime != 0) && (here->BSIM4v4qNode != 0))
            {
                while (here->BSIM4v4SPqPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v4SPqPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v4gNodePrime != 0) && (here->BSIM4v4qNode != 0))
            {
                while (here->BSIM4v4GPqPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v4GPqPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v4gNodeExt != 0) && (here->BSIM4v4gNodeExt != 0))
            {
                while (here->BSIM4v4GEgePtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v4GEgePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v4gNodeExt != 0) && (here->BSIM4v4gNodePrime != 0))
            {
                while (here->BSIM4v4GEgpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v4GEgpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v4gNodePrime != 0) && (here->BSIM4v4gNodeExt != 0))
            {
                while (here->BSIM4v4GPgePtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v4GPgePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v4gNodeExt != 0) && (here->BSIM4v4dNodePrime != 0))
            {
                while (here->BSIM4v4GEdpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v4GEdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v4gNodeExt != 0) && (here->BSIM4v4sNodePrime != 0))
            {
                while (here->BSIM4v4GEspPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v4GEspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v4gNodeExt != 0) && (here->BSIM4v4bNodePrime != 0))
            {
                while (here->BSIM4v4GEbpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v4GEbpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v4gNodeMid != 0) && (here->BSIM4v4dNodePrime != 0))
            {
                while (here->BSIM4v4GMdpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v4GMdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v4gNodeMid != 0) && (here->BSIM4v4gNodePrime != 0))
            {
                while (here->BSIM4v4GMgpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v4GMgpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v4gNodeMid != 0) && (here->BSIM4v4gNodeMid != 0))
            {
                while (here->BSIM4v4GMgmPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v4GMgmPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v4gNodeMid != 0) && (here->BSIM4v4gNodeExt != 0))
            {
                while (here->BSIM4v4GMgePtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v4GMgePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v4gNodeMid != 0) && (here->BSIM4v4sNodePrime != 0))
            {
                while (here->BSIM4v4GMspPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v4GMspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v4gNodeMid != 0) && (here->BSIM4v4bNodePrime != 0))
            {
                while (here->BSIM4v4GMbpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v4GMbpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v4dNodePrime != 0) && (here->BSIM4v4gNodeMid != 0))
            {
                while (here->BSIM4v4DPgmPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v4DPgmPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v4gNodePrime != 0) && (here->BSIM4v4gNodeMid != 0))
            {
                while (here->BSIM4v4GPgmPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v4GPgmPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v4gNodeExt != 0) && (here->BSIM4v4gNodeMid != 0))
            {
                while (here->BSIM4v4GEgmPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v4GEgmPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v4sNodePrime != 0) && (here->BSIM4v4gNodeMid != 0))
            {
                while (here->BSIM4v4SPgmPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v4SPgmPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v4bNodePrime != 0) && (here->BSIM4v4gNodeMid != 0))
            {
                while (here->BSIM4v4BPgmPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v4BPgmPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v4dNodePrime != 0) && (here->BSIM4v4dbNode != 0))
            {
                while (here->BSIM4v4DPdbPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v4DPdbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v4sNodePrime != 0) && (here->BSIM4v4sbNode != 0))
            {
                while (here->BSIM4v4SPsbPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v4SPsbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v4dbNode != 0) && (here->BSIM4v4dNodePrime != 0))
            {
                while (here->BSIM4v4DBdpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v4DBdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v4dbNode != 0) && (here->BSIM4v4dbNode != 0))
            {
                while (here->BSIM4v4DBdbPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v4DBdbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v4dbNode != 0) && (here->BSIM4v4bNodePrime != 0))
            {
                while (here->BSIM4v4DBbpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v4DBbpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v4dbNode != 0) && (here->BSIM4v4bNode != 0))
            {
                while (here->BSIM4v4DBbPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v4DBbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v4bNodePrime != 0) && (here->BSIM4v4dbNode != 0))
            {
                while (here->BSIM4v4BPdbPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v4BPdbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v4bNodePrime != 0) && (here->BSIM4v4bNode != 0))
            {
                while (here->BSIM4v4BPbPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v4BPbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v4bNodePrime != 0) && (here->BSIM4v4sbNode != 0))
            {
                while (here->BSIM4v4BPsbPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v4BPsbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v4sbNode != 0) && (here->BSIM4v4sNodePrime != 0))
            {
                while (here->BSIM4v4SBspPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v4SBspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v4sbNode != 0) && (here->BSIM4v4bNodePrime != 0))
            {
                while (here->BSIM4v4SBbpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v4SBbpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v4sbNode != 0) && (here->BSIM4v4bNode != 0))
            {
                while (here->BSIM4v4SBbPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v4SBbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v4sbNode != 0) && (here->BSIM4v4sbNode != 0))
            {
                while (here->BSIM4v4SBsbPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v4SBsbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v4bNode != 0) && (here->BSIM4v4dbNode != 0))
            {
                while (here->BSIM4v4BdbPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v4BdbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v4bNode != 0) && (here->BSIM4v4bNodePrime != 0))
            {
                while (here->BSIM4v4BbpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v4BbpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v4bNode != 0) && (here->BSIM4v4sbNode != 0))
            {
                while (here->BSIM4v4BsbPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v4BsbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v4bNode != 0) && (here->BSIM4v4bNode != 0))
            {
                while (here->BSIM4v4BbPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v4BbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v4dNode != 0) && (here->BSIM4v4gNodePrime != 0))
            {
                while (here->BSIM4v4DgpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v4DgpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v4dNode != 0) && (here->BSIM4v4sNodePrime != 0))
            {
                while (here->BSIM4v4DspPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v4DspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v4dNode != 0) && (here->BSIM4v4bNodePrime != 0))
            {
                while (here->BSIM4v4DbpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v4DbpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v4sNode != 0) && (here->BSIM4v4dNodePrime != 0))
            {
                while (here->BSIM4v4SdpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v4SdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v4sNode != 0) && (here->BSIM4v4gNodePrime != 0))
            {
                while (here->BSIM4v4SgpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v4SgpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v4sNode != 0) && (here->BSIM4v4bNodePrime != 0))
            {
                while (here->BSIM4v4SbpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v4SbpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }
        }
    }

    return (OK) ;
}