/**********
Author: 2012 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "bsim4v5def.h"
#include "ngspice/sperror.h"

int
BSIM4v5bindCSC(GENmodel *inModel, CKTcircuit *ckt)
{
    BSIM4v5model *model = (BSIM4v5model *)inModel;
    int i ;

    /*  loop through all the b4v5 models */
    for( ; model != NULL; model = model->BSIM4v5nextModel ) {
	BSIM4v5instance *here;

        /* loop through all the instances of the model */
        for (here = model->BSIM4v5instances; here != NULL ;
	    here = here->BSIM4v5nextInstance) {

		i = 0 ;
		if ((here-> BSIM4v5dNodePrime != 0) && (here-> BSIM4v5bNodePrime != 0)) {
			while (here->BSIM4v5DPbpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v5DPbpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v5gNodePrime != 0) && (here-> BSIM4v5bNodePrime != 0)) {
			while (here->BSIM4v5GPbpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v5GPbpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v5sNodePrime != 0) && (here-> BSIM4v5bNodePrime != 0)) {
			while (here->BSIM4v5SPbpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v5SPbpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v5bNodePrime != 0) && (here-> BSIM4v5dNodePrime != 0)) {
			while (here->BSIM4v5BPdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v5BPdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v5bNodePrime != 0) && (here-> BSIM4v5gNodePrime != 0)) {
			while (here->BSIM4v5BPgpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v5BPgpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v5bNodePrime != 0) && (here-> BSIM4v5sNodePrime != 0)) {
			while (here->BSIM4v5BPspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v5BPspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v5bNodePrime != 0) && (here-> BSIM4v5bNodePrime != 0)) {
			while (here->BSIM4v5BPbpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v5BPbpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v5dNode != 0) && (here-> BSIM4v5dNode != 0)) {
			while (here->BSIM4v5DdPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v5DdPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v5gNodePrime != 0) && (here-> BSIM4v5gNodePrime != 0)) {
			while (here->BSIM4v5GPgpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v5GPgpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v5sNode != 0) && (here-> BSIM4v5sNode != 0)) {
			while (here->BSIM4v5SsPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v5SsPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v5dNodePrime != 0) && (here-> BSIM4v5dNodePrime != 0)) {
			while (here->BSIM4v5DPdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v5DPdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v5sNodePrime != 0) && (here-> BSIM4v5sNodePrime != 0)) {
			while (here->BSIM4v5SPspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v5SPspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v5dNode != 0) && (here-> BSIM4v5dNodePrime != 0)) {
			while (here->BSIM4v5DdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v5DdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v5gNodePrime != 0) && (here-> BSIM4v5dNodePrime != 0)) {
			while (here->BSIM4v5GPdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v5GPdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v5gNodePrime != 0) && (here-> BSIM4v5sNodePrime != 0)) {
			while (here->BSIM4v5GPspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v5GPspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v5sNode != 0) && (here-> BSIM4v5sNodePrime != 0)) {
			while (here->BSIM4v5SspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v5SspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v5dNodePrime != 0) && (here-> BSIM4v5sNodePrime != 0)) {
			while (here->BSIM4v5DPspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v5DPspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v5dNodePrime != 0) && (here-> BSIM4v5dNode != 0)) {
			while (here->BSIM4v5DPdPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v5DPdPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v5dNodePrime != 0) && (here-> BSIM4v5gNodePrime != 0)) {
			while (here->BSIM4v5DPgpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v5DPgpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v5sNodePrime != 0) && (here-> BSIM4v5gNodePrime != 0)) {
			while (here->BSIM4v5SPgpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v5SPgpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v5sNodePrime != 0) && (here-> BSIM4v5sNode != 0)) {
			while (here->BSIM4v5SPsPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v5SPsPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v5sNodePrime != 0) && (here-> BSIM4v5dNodePrime != 0)) {
			while (here->BSIM4v5SPdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v5SPdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v5qNode != 0) && (here-> BSIM4v5qNode != 0)) {
			while (here->BSIM4v5QqPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v5QqPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v5qNode != 0) && (here-> BSIM4v5bNodePrime != 0)) {
			while (here->BSIM4v5QbpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v5QbpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v5qNode != 0) && (here-> BSIM4v5dNodePrime != 0)) {
			while (here->BSIM4v5QdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v5QdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v5qNode != 0) && (here-> BSIM4v5sNodePrime != 0)) {
			while (here->BSIM4v5QspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v5QspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v5qNode != 0) && (here-> BSIM4v5gNodePrime != 0)) {
			while (here->BSIM4v5QgpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v5QgpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v5dNodePrime != 0) && (here-> BSIM4v5qNode != 0)) {
			while (here->BSIM4v5DPqPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v5DPqPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v5sNodePrime != 0) && (here-> BSIM4v5qNode != 0)) {
			while (here->BSIM4v5SPqPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v5SPqPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v5gNodePrime != 0) && (here-> BSIM4v5qNode != 0)) {
			while (here->BSIM4v5GPqPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v5GPqPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

                if (here->BSIM4v5rgateMod != 0) {

		i = 0 ;
		if ((here-> BSIM4v5gNodeExt != 0) && (here-> BSIM4v5gNodeExt != 0)) {
			while (here->BSIM4v5GEgePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v5GEgePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v5gNodeExt != 0) && (here-> BSIM4v5gNodePrime != 0)) {
			while (here->BSIM4v5GEgpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v5GEgpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v5gNodePrime != 0) && (here-> BSIM4v5gNodeExt != 0)) {
			while (here->BSIM4v5GPgePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v5GPgePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v5gNodeExt != 0) && (here-> BSIM4v5dNodePrime != 0)) {
			while (here->BSIM4v5GEdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v5GEdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v5gNodeExt != 0) && (here-> BSIM4v5sNodePrime != 0)) {
			while (here->BSIM4v5GEspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v5GEspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v5gNodeExt != 0) && (here-> BSIM4v5bNodePrime != 0)) {
			while (here->BSIM4v5GEbpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v5GEbpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v5gNodeMid != 0) && (here-> BSIM4v5dNodePrime != 0)) {
			while (here->BSIM4v5GMdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v5GMdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v5gNodeMid != 0) && (here-> BSIM4v5gNodePrime != 0)) {
			while (here->BSIM4v5GMgpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v5GMgpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v5gNodeMid != 0) && (here-> BSIM4v5gNodeMid != 0)) {
			while (here->BSIM4v5GMgmPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v5GMgmPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v5gNodeMid != 0) && (here-> BSIM4v5gNodeExt != 0)) {
			while (here->BSIM4v5GMgePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v5GMgePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v5gNodeMid != 0) && (here-> BSIM4v5sNodePrime != 0)) {
			while (here->BSIM4v5GMspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v5GMspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v5gNodeMid != 0) && (here-> BSIM4v5bNodePrime != 0)) {
			while (here->BSIM4v5GMbpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v5GMbpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v5dNodePrime != 0) && (here-> BSIM4v5gNodeMid != 0)) {
			while (here->BSIM4v5DPgmPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v5DPgmPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v5gNodePrime != 0) && (here-> BSIM4v5gNodeMid != 0)) {
			while (here->BSIM4v5GPgmPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v5GPgmPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v5gNodeExt != 0) && (here-> BSIM4v5gNodeMid != 0)) {
			while (here->BSIM4v5GEgmPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v5GEgmPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v5sNodePrime != 0) && (here-> BSIM4v5gNodeMid != 0)) {
			while (here->BSIM4v5SPgmPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v5SPgmPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v5bNodePrime != 0) && (here-> BSIM4v5gNodeMid != 0)) {
			while (here->BSIM4v5BPgmPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v5BPgmPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
                } /* IF */

                if ((here->BSIM4v5rbodyMod ==1) || (here->BSIM4v5rbodyMod ==2)) {

		i = 0 ;
		if ((here-> BSIM4v5dNodePrime != 0) && (here-> BSIM4v5dbNode != 0)) {
			while (here->BSIM4v5DPdbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v5DPdbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v5sNodePrime != 0) && (here-> BSIM4v5sbNode != 0)) {
			while (here->BSIM4v5SPsbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v5SPsbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v5dbNode != 0) && (here-> BSIM4v5dNodePrime != 0)) {
			while (here->BSIM4v5DBdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v5DBdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v5dbNode != 0) && (here-> BSIM4v5dbNode != 0)) {
			while (here->BSIM4v5DBdbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v5DBdbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v5dbNode != 0) && (here-> BSIM4v5bNodePrime != 0)) {
			while (here->BSIM4v5DBbpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v5DBbpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v5dbNode != 0) && (here-> BSIM4v5bNode != 0)) {
			while (here->BSIM4v5DBbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v5DBbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v5bNodePrime != 0) && (here-> BSIM4v5dbNode != 0)) {
			while (here->BSIM4v5BPdbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v5BPdbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v5bNodePrime != 0) && (here-> BSIM4v5bNode != 0)) {
			while (here->BSIM4v5BPbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v5BPbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v5bNodePrime != 0) && (here-> BSIM4v5sbNode != 0)) {
			while (here->BSIM4v5BPsbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v5BPsbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v5sbNode != 0) && (here-> BSIM4v5sNodePrime != 0)) {
			while (here->BSIM4v5SBspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v5SBspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v5sbNode != 0) && (here-> BSIM4v5bNodePrime != 0)) {
			while (here->BSIM4v5SBbpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v5SBbpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v5sbNode != 0) && (here-> BSIM4v5bNode != 0)) {
			while (here->BSIM4v5SBbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v5SBbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v5sbNode != 0) && (here-> BSIM4v5sbNode != 0)) {
			while (here->BSIM4v5SBsbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v5SBsbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v5bNode != 0) && (here-> BSIM4v5dbNode != 0)) {
			while (here->BSIM4v5BdbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v5BdbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v5bNode != 0) && (here-> BSIM4v5bNodePrime != 0)) {
			while (here->BSIM4v5BbpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v5BbpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v5bNode != 0) && (here-> BSIM4v5sbNode != 0)) {
			while (here->BSIM4v5BsbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v5BsbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v5bNode != 0) && (here-> BSIM4v5bNode != 0)) {
			while (here->BSIM4v5BbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v5BbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
                } /* IF */

                if (model->BSIM4v5rdsMod) {

		i = 0 ;
		if ((here-> BSIM4v5dNode != 0) && (here-> BSIM4v5gNodePrime != 0)) {
			while (here->BSIM4v5DgpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v5DgpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v5dNode != 0) && (here-> BSIM4v5sNodePrime != 0)) {
			while (here->BSIM4v5DspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v5DspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v5dNode != 0) && (here-> BSIM4v5bNodePrime != 0)) {
			while (here->BSIM4v5DbpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v5DbpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v5sNode != 0) && (here-> BSIM4v5dNodePrime != 0)) {
			while (here->BSIM4v5SdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v5SdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v5sNode != 0) && (here-> BSIM4v5gNodePrime != 0)) {
			while (here->BSIM4v5SgpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v5SgpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v5sNode != 0) && (here-> BSIM4v5bNodePrime != 0)) {
			while (here->BSIM4v5SbpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v5SbpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
                } /* IF */
	}
    }
    return(OK);
}

int
BSIM4v5bindCSCComplex(GENmodel *inModel, CKTcircuit *ckt)
{
    BSIM4v5model *model = (BSIM4v5model *)inModel;
    int i ;

    /*  loop through all the b4v5 models */
    for( ; model != NULL; model = model->BSIM4v5nextModel ) {
	BSIM4v5instance *here;

        /* loop through all the instances of the model */
        for (here = model->BSIM4v5instances; here != NULL ;
	    here = here->BSIM4v5nextInstance) {

		i = 0 ;
		if ((here-> BSIM4v5dNodePrime != 0) && (here-> BSIM4v5bNodePrime != 0)) {
			while (here->BSIM4v5DPbpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v5DPbpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v5gNodePrime != 0) && (here-> BSIM4v5bNodePrime != 0)) {
			while (here->BSIM4v5GPbpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v5GPbpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v5sNodePrime != 0) && (here-> BSIM4v5bNodePrime != 0)) {
			while (here->BSIM4v5SPbpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v5SPbpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v5bNodePrime != 0) && (here-> BSIM4v5dNodePrime != 0)) {
			while (here->BSIM4v5BPdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v5BPdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v5bNodePrime != 0) && (here-> BSIM4v5gNodePrime != 0)) {
			while (here->BSIM4v5BPgpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v5BPgpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v5bNodePrime != 0) && (here-> BSIM4v5sNodePrime != 0)) {
			while (here->BSIM4v5BPspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v5BPspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v5bNodePrime != 0) && (here-> BSIM4v5bNodePrime != 0)) {
			while (here->BSIM4v5BPbpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v5BPbpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v5dNode != 0) && (here-> BSIM4v5dNode != 0)) {
			while (here->BSIM4v5DdPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v5DdPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v5gNodePrime != 0) && (here-> BSIM4v5gNodePrime != 0)) {
			while (here->BSIM4v5GPgpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v5GPgpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v5sNode != 0) && (here-> BSIM4v5sNode != 0)) {
			while (here->BSIM4v5SsPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v5SsPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v5dNodePrime != 0) && (here-> BSIM4v5dNodePrime != 0)) {
			while (here->BSIM4v5DPdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v5DPdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v5sNodePrime != 0) && (here-> BSIM4v5sNodePrime != 0)) {
			while (here->BSIM4v5SPspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v5SPspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v5dNode != 0) && (here-> BSIM4v5dNodePrime != 0)) {
			while (here->BSIM4v5DdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v5DdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v5gNodePrime != 0) && (here-> BSIM4v5dNodePrime != 0)) {
			while (here->BSIM4v5GPdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v5GPdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v5gNodePrime != 0) && (here-> BSIM4v5sNodePrime != 0)) {
			while (here->BSIM4v5GPspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v5GPspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v5sNode != 0) && (here-> BSIM4v5sNodePrime != 0)) {
			while (here->BSIM4v5SspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v5SspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v5dNodePrime != 0) && (here-> BSIM4v5sNodePrime != 0)) {
			while (here->BSIM4v5DPspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v5DPspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v5dNodePrime != 0) && (here-> BSIM4v5dNode != 0)) {
			while (here->BSIM4v5DPdPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v5DPdPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v5dNodePrime != 0) && (here-> BSIM4v5gNodePrime != 0)) {
			while (here->BSIM4v5DPgpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v5DPgpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v5sNodePrime != 0) && (here-> BSIM4v5gNodePrime != 0)) {
			while (here->BSIM4v5SPgpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v5SPgpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v5sNodePrime != 0) && (here-> BSIM4v5sNode != 0)) {
			while (here->BSIM4v5SPsPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v5SPsPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v5sNodePrime != 0) && (here-> BSIM4v5dNodePrime != 0)) {
			while (here->BSIM4v5SPdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v5SPdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v5qNode != 0) && (here-> BSIM4v5qNode != 0)) {
			while (here->BSIM4v5QqPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v5QqPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v5qNode != 0) && (here-> BSIM4v5bNodePrime != 0)) {
			while (here->BSIM4v5QbpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v5QbpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v5qNode != 0) && (here-> BSIM4v5dNodePrime != 0)) {
			while (here->BSIM4v5QdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v5QdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v5qNode != 0) && (here-> BSIM4v5sNodePrime != 0)) {
			while (here->BSIM4v5QspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v5QspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v5qNode != 0) && (here-> BSIM4v5gNodePrime != 0)) {
			while (here->BSIM4v5QgpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v5QgpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v5dNodePrime != 0) && (here-> BSIM4v5qNode != 0)) {
			while (here->BSIM4v5DPqPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v5DPqPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v5sNodePrime != 0) && (here-> BSIM4v5qNode != 0)) {
			while (here->BSIM4v5SPqPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v5SPqPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v5gNodePrime != 0) && (here-> BSIM4v5qNode != 0)) {
			while (here->BSIM4v5GPqPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v5GPqPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

                if (here->BSIM4v5rgateMod != 0) {

		i = 0 ;
		if ((here-> BSIM4v5gNodeExt != 0) && (here-> BSIM4v5gNodeExt != 0)) {
			while (here->BSIM4v5GEgePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v5GEgePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v5gNodeExt != 0) && (here-> BSIM4v5gNodePrime != 0)) {
			while (here->BSIM4v5GEgpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v5GEgpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v5gNodePrime != 0) && (here-> BSIM4v5gNodeExt != 0)) {
			while (here->BSIM4v5GPgePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v5GPgePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v5gNodeExt != 0) && (here-> BSIM4v5dNodePrime != 0)) {
			while (here->BSIM4v5GEdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v5GEdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v5gNodeExt != 0) && (here-> BSIM4v5sNodePrime != 0)) {
			while (here->BSIM4v5GEspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v5GEspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v5gNodeExt != 0) && (here-> BSIM4v5bNodePrime != 0)) {
			while (here->BSIM4v5GEbpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v5GEbpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v5gNodeMid != 0) && (here-> BSIM4v5dNodePrime != 0)) {
			while (here->BSIM4v5GMdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v5GMdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v5gNodeMid != 0) && (here-> BSIM4v5gNodePrime != 0)) {
			while (here->BSIM4v5GMgpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v5GMgpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v5gNodeMid != 0) && (here-> BSIM4v5gNodeMid != 0)) {
			while (here->BSIM4v5GMgmPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v5GMgmPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v5gNodeMid != 0) && (here-> BSIM4v5gNodeExt != 0)) {
			while (here->BSIM4v5GMgePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v5GMgePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v5gNodeMid != 0) && (here-> BSIM4v5sNodePrime != 0)) {
			while (here->BSIM4v5GMspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v5GMspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v5gNodeMid != 0) && (here-> BSIM4v5bNodePrime != 0)) {
			while (here->BSIM4v5GMbpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v5GMbpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v5dNodePrime != 0) && (here-> BSIM4v5gNodeMid != 0)) {
			while (here->BSIM4v5DPgmPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v5DPgmPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v5gNodePrime != 0) && (here-> BSIM4v5gNodeMid != 0)) {
			while (here->BSIM4v5GPgmPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v5GPgmPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v5gNodeExt != 0) && (here-> BSIM4v5gNodeMid != 0)) {
			while (here->BSIM4v5GEgmPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v5GEgmPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v5sNodePrime != 0) && (here-> BSIM4v5gNodeMid != 0)) {
			while (here->BSIM4v5SPgmPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v5SPgmPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v5bNodePrime != 0) && (here-> BSIM4v5gNodeMid != 0)) {
			while (here->BSIM4v5BPgmPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v5BPgmPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
                } /* IF */

                if ((here->BSIM4v5rbodyMod ==1) || (here->BSIM4v5rbodyMod ==2)) {

		i = 0 ;
		if ((here-> BSIM4v5dNodePrime != 0) && (here-> BSIM4v5dbNode != 0)) {
			while (here->BSIM4v5DPdbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v5DPdbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v5sNodePrime != 0) && (here-> BSIM4v5sbNode != 0)) {
			while (here->BSIM4v5SPsbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v5SPsbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v5dbNode != 0) && (here-> BSIM4v5dNodePrime != 0)) {
			while (here->BSIM4v5DBdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v5DBdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v5dbNode != 0) && (here-> BSIM4v5dbNode != 0)) {
			while (here->BSIM4v5DBdbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v5DBdbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v5dbNode != 0) && (here-> BSIM4v5bNodePrime != 0)) {
			while (here->BSIM4v5DBbpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v5DBbpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v5dbNode != 0) && (here-> BSIM4v5bNode != 0)) {
			while (here->BSIM4v5DBbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v5DBbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v5bNodePrime != 0) && (here-> BSIM4v5dbNode != 0)) {
			while (here->BSIM4v5BPdbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v5BPdbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v5bNodePrime != 0) && (here-> BSIM4v5bNode != 0)) {
			while (here->BSIM4v5BPbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v5BPbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v5bNodePrime != 0) && (here-> BSIM4v5sbNode != 0)) {
			while (here->BSIM4v5BPsbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v5BPsbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v5sbNode != 0) && (here-> BSIM4v5sNodePrime != 0)) {
			while (here->BSIM4v5SBspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v5SBspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v5sbNode != 0) && (here-> BSIM4v5bNodePrime != 0)) {
			while (here->BSIM4v5SBbpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v5SBbpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v5sbNode != 0) && (here-> BSIM4v5bNode != 0)) {
			while (here->BSIM4v5SBbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v5SBbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v5sbNode != 0) && (here-> BSIM4v5sbNode != 0)) {
			while (here->BSIM4v5SBsbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v5SBsbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v5bNode != 0) && (here-> BSIM4v5dbNode != 0)) {
			while (here->BSIM4v5BdbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v5BdbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v5bNode != 0) && (here-> BSIM4v5bNodePrime != 0)) {
			while (here->BSIM4v5BbpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v5BbpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v5bNode != 0) && (here-> BSIM4v5sbNode != 0)) {
			while (here->BSIM4v5BsbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v5BsbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v5bNode != 0) && (here-> BSIM4v5bNode != 0)) {
			while (here->BSIM4v5BbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v5BbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
                } /* IF */

                if (model->BSIM4v5rdsMod) {

		i = 0 ;
		if ((here-> BSIM4v5dNode != 0) && (here-> BSIM4v5gNodePrime != 0)) {
			while (here->BSIM4v5DgpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v5DgpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v5dNode != 0) && (here-> BSIM4v5sNodePrime != 0)) {
			while (here->BSIM4v5DspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v5DspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v5dNode != 0) && (here-> BSIM4v5bNodePrime != 0)) {
			while (here->BSIM4v5DbpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v5DbpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v5sNode != 0) && (here-> BSIM4v5dNodePrime != 0)) {
			while (here->BSIM4v5SdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v5SdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v5sNode != 0) && (here-> BSIM4v5gNodePrime != 0)) {
			while (here->BSIM4v5SgpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v5SgpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4v5sNode != 0) && (here-> BSIM4v5bNodePrime != 0)) {
			while (here->BSIM4v5SbpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v5SbpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
                } /* IF */
	}
    }
    return(OK);
}

int
BSIM4v5bindCSCComplexToReal (GENmodel *inModel, CKTcircuit *ckt)
{
    BSIM4v5model *model = (BSIM4v5model *)inModel ;
    BSIM4v5instance *here ;
    int i ;

    /*  loop through all the bsim4v5 models */
    for ( ; model != NULL ; model = model->BSIM4v5nextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->BSIM4v5instances ; here != NULL ; here = here->BSIM4v5nextInstance)
        {
            i = 0 ;
            if ((here->BSIM4v5dNodePrime != 0) && (here->BSIM4v5bNodePrime != 0))
            {
                while (here->BSIM4v5DPbpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v5DPbpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v5gNodePrime != 0) && (here->BSIM4v5bNodePrime != 0))
            {
                while (here->BSIM4v5GPbpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v5GPbpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v5sNodePrime != 0) && (here->BSIM4v5bNodePrime != 0))
            {
                while (here->BSIM4v5SPbpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v5SPbpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v5bNodePrime != 0) && (here->BSIM4v5dNodePrime != 0))
            {
                while (here->BSIM4v5BPdpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v5BPdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v5bNodePrime != 0) && (here->BSIM4v5gNodePrime != 0))
            {
                while (here->BSIM4v5BPgpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v5BPgpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v5bNodePrime != 0) && (here->BSIM4v5sNodePrime != 0))
            {
                while (here->BSIM4v5BPspPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v5BPspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v5bNodePrime != 0) && (here->BSIM4v5bNodePrime != 0))
            {
                while (here->BSIM4v5BPbpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v5BPbpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v5dNode != 0) && (here->BSIM4v5dNode != 0))
            {
                while (here->BSIM4v5DdPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v5DdPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v5gNodePrime != 0) && (here->BSIM4v5gNodePrime != 0))
            {
                while (here->BSIM4v5GPgpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v5GPgpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v5sNode != 0) && (here->BSIM4v5sNode != 0))
            {
                while (here->BSIM4v5SsPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v5SsPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v5dNodePrime != 0) && (here->BSIM4v5dNodePrime != 0))
            {
                while (here->BSIM4v5DPdpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v5DPdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v5sNodePrime != 0) && (here->BSIM4v5sNodePrime != 0))
            {
                while (here->BSIM4v5SPspPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v5SPspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v5dNode != 0) && (here->BSIM4v5dNodePrime != 0))
            {
                while (here->BSIM4v5DdpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v5DdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v5gNodePrime != 0) && (here->BSIM4v5dNodePrime != 0))
            {
                while (here->BSIM4v5GPdpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v5GPdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v5gNodePrime != 0) && (here->BSIM4v5sNodePrime != 0))
            {
                while (here->BSIM4v5GPspPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v5GPspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v5sNode != 0) && (here->BSIM4v5sNodePrime != 0))
            {
                while (here->BSIM4v5SspPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v5SspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v5dNodePrime != 0) && (here->BSIM4v5sNodePrime != 0))
            {
                while (here->BSIM4v5DPspPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v5DPspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v5dNodePrime != 0) && (here->BSIM4v5dNode != 0))
            {
                while (here->BSIM4v5DPdPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v5DPdPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v5dNodePrime != 0) && (here->BSIM4v5gNodePrime != 0))
            {
                while (here->BSIM4v5DPgpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v5DPgpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v5sNodePrime != 0) && (here->BSIM4v5gNodePrime != 0))
            {
                while (here->BSIM4v5SPgpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v5SPgpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v5sNodePrime != 0) && (here->BSIM4v5sNode != 0))
            {
                while (here->BSIM4v5SPsPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v5SPsPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v5sNodePrime != 0) && (here->BSIM4v5dNodePrime != 0))
            {
                while (here->BSIM4v5SPdpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v5SPdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v5qNode != 0) && (here->BSIM4v5qNode != 0))
            {
                while (here->BSIM4v5QqPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v5QqPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v5qNode != 0) && (here->BSIM4v5bNodePrime != 0))
            {
                while (here->BSIM4v5QbpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v5QbpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v5qNode != 0) && (here->BSIM4v5dNodePrime != 0))
            {
                while (here->BSIM4v5QdpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v5QdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v5qNode != 0) && (here->BSIM4v5sNodePrime != 0))
            {
                while (here->BSIM4v5QspPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v5QspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v5qNode != 0) && (here->BSIM4v5gNodePrime != 0))
            {
                while (here->BSIM4v5QgpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v5QgpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v5dNodePrime != 0) && (here->BSIM4v5qNode != 0))
            {
                while (here->BSIM4v5DPqPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v5DPqPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v5sNodePrime != 0) && (here->BSIM4v5qNode != 0))
            {
                while (here->BSIM4v5SPqPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v5SPqPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v5gNodePrime != 0) && (here->BSIM4v5qNode != 0))
            {
                while (here->BSIM4v5GPqPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v5GPqPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v5gNodeExt != 0) && (here->BSIM4v5gNodeExt != 0))
            {
                while (here->BSIM4v5GEgePtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v5GEgePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v5gNodeExt != 0) && (here->BSIM4v5gNodePrime != 0))
            {
                while (here->BSIM4v5GEgpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v5GEgpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v5gNodePrime != 0) && (here->BSIM4v5gNodeExt != 0))
            {
                while (here->BSIM4v5GPgePtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v5GPgePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v5gNodeExt != 0) && (here->BSIM4v5dNodePrime != 0))
            {
                while (here->BSIM4v5GEdpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v5GEdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v5gNodeExt != 0) && (here->BSIM4v5sNodePrime != 0))
            {
                while (here->BSIM4v5GEspPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v5GEspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v5gNodeExt != 0) && (here->BSIM4v5bNodePrime != 0))
            {
                while (here->BSIM4v5GEbpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v5GEbpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v5gNodeMid != 0) && (here->BSIM4v5dNodePrime != 0))
            {
                while (here->BSIM4v5GMdpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v5GMdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v5gNodeMid != 0) && (here->BSIM4v5gNodePrime != 0))
            {
                while (here->BSIM4v5GMgpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v5GMgpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v5gNodeMid != 0) && (here->BSIM4v5gNodeMid != 0))
            {
                while (here->BSIM4v5GMgmPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v5GMgmPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v5gNodeMid != 0) && (here->BSIM4v5gNodeExt != 0))
            {
                while (here->BSIM4v5GMgePtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v5GMgePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v5gNodeMid != 0) && (here->BSIM4v5sNodePrime != 0))
            {
                while (here->BSIM4v5GMspPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v5GMspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v5gNodeMid != 0) && (here->BSIM4v5bNodePrime != 0))
            {
                while (here->BSIM4v5GMbpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v5GMbpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v5dNodePrime != 0) && (here->BSIM4v5gNodeMid != 0))
            {
                while (here->BSIM4v5DPgmPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v5DPgmPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v5gNodePrime != 0) && (here->BSIM4v5gNodeMid != 0))
            {
                while (here->BSIM4v5GPgmPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v5GPgmPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v5gNodeExt != 0) && (here->BSIM4v5gNodeMid != 0))
            {
                while (here->BSIM4v5GEgmPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v5GEgmPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v5sNodePrime != 0) && (here->BSIM4v5gNodeMid != 0))
            {
                while (here->BSIM4v5SPgmPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v5SPgmPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v5bNodePrime != 0) && (here->BSIM4v5gNodeMid != 0))
            {
                while (here->BSIM4v5BPgmPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v5BPgmPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v5dNodePrime != 0) && (here->BSIM4v5dbNode != 0))
            {
                while (here->BSIM4v5DPdbPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v5DPdbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v5sNodePrime != 0) && (here->BSIM4v5sbNode != 0))
            {
                while (here->BSIM4v5SPsbPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v5SPsbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v5dbNode != 0) && (here->BSIM4v5dNodePrime != 0))
            {
                while (here->BSIM4v5DBdpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v5DBdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v5dbNode != 0) && (here->BSIM4v5dbNode != 0))
            {
                while (here->BSIM4v5DBdbPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v5DBdbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v5dbNode != 0) && (here->BSIM4v5bNodePrime != 0))
            {
                while (here->BSIM4v5DBbpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v5DBbpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v5dbNode != 0) && (here->BSIM4v5bNode != 0))
            {
                while (here->BSIM4v5DBbPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v5DBbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v5bNodePrime != 0) && (here->BSIM4v5dbNode != 0))
            {
                while (here->BSIM4v5BPdbPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v5BPdbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v5bNodePrime != 0) && (here->BSIM4v5bNode != 0))
            {
                while (here->BSIM4v5BPbPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v5BPbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v5bNodePrime != 0) && (here->BSIM4v5sbNode != 0))
            {
                while (here->BSIM4v5BPsbPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v5BPsbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v5sbNode != 0) && (here->BSIM4v5sNodePrime != 0))
            {
                while (here->BSIM4v5SBspPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v5SBspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v5sbNode != 0) && (here->BSIM4v5bNodePrime != 0))
            {
                while (here->BSIM4v5SBbpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v5SBbpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v5sbNode != 0) && (here->BSIM4v5bNode != 0))
            {
                while (here->BSIM4v5SBbPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v5SBbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v5sbNode != 0) && (here->BSIM4v5sbNode != 0))
            {
                while (here->BSIM4v5SBsbPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v5SBsbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v5bNode != 0) && (here->BSIM4v5dbNode != 0))
            {
                while (here->BSIM4v5BdbPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v5BdbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v5bNode != 0) && (here->BSIM4v5bNodePrime != 0))
            {
                while (here->BSIM4v5BbpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v5BbpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v5bNode != 0) && (here->BSIM4v5sbNode != 0))
            {
                while (here->BSIM4v5BsbPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v5BsbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v5bNode != 0) && (here->BSIM4v5bNode != 0))
            {
                while (here->BSIM4v5BbPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v5BbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v5dNode != 0) && (here->BSIM4v5gNodePrime != 0))
            {
                while (here->BSIM4v5DgpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v5DgpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v5dNode != 0) && (here->BSIM4v5sNodePrime != 0))
            {
                while (here->BSIM4v5DspPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v5DspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v5dNode != 0) && (here->BSIM4v5bNodePrime != 0))
            {
                while (here->BSIM4v5DbpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v5DbpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v5sNode != 0) && (here->BSIM4v5dNodePrime != 0))
            {
                while (here->BSIM4v5SdpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v5SdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v5sNode != 0) && (here->BSIM4v5gNodePrime != 0))
            {
                while (here->BSIM4v5SgpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v5SgpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v5sNode != 0) && (here->BSIM4v5bNodePrime != 0))
            {
                while (here->BSIM4v5SbpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v5SbpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }
        }
    }

    return (OK) ;
}