/**********
Author: 2012 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "bsim4def.h"
#include "ngspice/sperror.h"

int
BSIM4bindCSC(GENmodel *inModel, CKTcircuit *ckt)
{
    BSIM4model *model = (BSIM4model *)inModel;
    int i ;

    /*  loop through all the b4 models */
    for( ; model != NULL; model = model->BSIM4nextModel ) {
	BSIM4instance *here;

        /* loop through all the instances of the model */
        for (here = model->BSIM4instances; here != NULL ;
	    here = here->BSIM4nextInstance) {

		i = 0 ;
		if ((here-> BSIM4dNodePrime != 0) && (here-> BSIM4bNodePrime != 0)) {
			while (here->BSIM4DPbpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4DPbpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4gNodePrime != 0) && (here-> BSIM4bNodePrime != 0)) {
			while (here->BSIM4GPbpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4GPbpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4sNodePrime != 0) && (here-> BSIM4bNodePrime != 0)) {
			while (here->BSIM4SPbpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4SPbpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4bNodePrime != 0) && (here-> BSIM4dNodePrime != 0)) {
			while (here->BSIM4BPdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4BPdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4bNodePrime != 0) && (here-> BSIM4gNodePrime != 0)) {
			while (here->BSIM4BPgpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4BPgpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4bNodePrime != 0) && (here-> BSIM4sNodePrime != 0)) {
			while (here->BSIM4BPspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4BPspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4bNodePrime != 0) && (here-> BSIM4bNodePrime != 0)) {
			while (here->BSIM4BPbpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4BPbpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4dNode != 0) && (here-> BSIM4dNode != 0)) {
			while (here->BSIM4DdPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4DdPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4gNodePrime != 0) && (here-> BSIM4gNodePrime != 0)) {
			while (here->BSIM4GPgpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4GPgpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4sNode != 0) && (here-> BSIM4sNode != 0)) {
			while (here->BSIM4SsPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4SsPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4dNodePrime != 0) && (here-> BSIM4dNodePrime != 0)) {
			while (here->BSIM4DPdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4DPdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4sNodePrime != 0) && (here-> BSIM4sNodePrime != 0)) {
			while (here->BSIM4SPspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4SPspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4dNode != 0) && (here-> BSIM4dNodePrime != 0)) {
			while (here->BSIM4DdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4DdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4gNodePrime != 0) && (here-> BSIM4dNodePrime != 0)) {
			while (here->BSIM4GPdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4GPdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4gNodePrime != 0) && (here-> BSIM4sNodePrime != 0)) {
			while (here->BSIM4GPspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4GPspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4sNode != 0) && (here-> BSIM4sNodePrime != 0)) {
			while (here->BSIM4SspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4SspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4dNodePrime != 0) && (here-> BSIM4sNodePrime != 0)) {
			while (here->BSIM4DPspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4DPspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4dNodePrime != 0) && (here-> BSIM4dNode != 0)) {
			while (here->BSIM4DPdPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4DPdPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4dNodePrime != 0) && (here-> BSIM4gNodePrime != 0)) {
			while (here->BSIM4DPgpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4DPgpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4sNodePrime != 0) && (here-> BSIM4gNodePrime != 0)) {
			while (here->BSIM4SPgpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4SPgpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4sNodePrime != 0) && (here-> BSIM4sNode != 0)) {
			while (here->BSIM4SPsPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4SPsPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4sNodePrime != 0) && (here-> BSIM4dNodePrime != 0)) {
			while (here->BSIM4SPdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4SPdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4qNode != 0) && (here-> BSIM4qNode != 0)) {
			while (here->BSIM4QqPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4QqPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4qNode != 0) && (here-> BSIM4bNodePrime != 0)) {
			while (here->BSIM4QbpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4QbpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4qNode != 0) && (here-> BSIM4dNodePrime != 0)) {
			while (here->BSIM4QdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4QdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4qNode != 0) && (here-> BSIM4sNodePrime != 0)) {
			while (here->BSIM4QspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4QspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4qNode != 0) && (here-> BSIM4gNodePrime != 0)) {
			while (here->BSIM4QgpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4QgpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4dNodePrime != 0) && (here-> BSIM4qNode != 0)) {
			while (here->BSIM4DPqPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4DPqPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4sNodePrime != 0) && (here-> BSIM4qNode != 0)) {
			while (here->BSIM4SPqPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4SPqPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4gNodePrime != 0) && (here-> BSIM4qNode != 0)) {
			while (here->BSIM4GPqPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4GPqPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

                if (here->BSIM4rgateMod != 0) {

		i = 0 ;
		if ((here-> BSIM4gNodeExt != 0) && (here-> BSIM4gNodeExt != 0)) {
			while (here->BSIM4GEgePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4GEgePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4gNodeExt != 0) && (here-> BSIM4gNodePrime != 0)) {
			while (here->BSIM4GEgpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4GEgpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4gNodePrime != 0) && (here-> BSIM4gNodeExt != 0)) {
			while (here->BSIM4GPgePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4GPgePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4gNodeExt != 0) && (here-> BSIM4dNodePrime != 0)) {
			while (here->BSIM4GEdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4GEdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4gNodeExt != 0) && (here-> BSIM4sNodePrime != 0)) {
			while (here->BSIM4GEspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4GEspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4gNodeExt != 0) && (here-> BSIM4bNodePrime != 0)) {
			while (here->BSIM4GEbpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4GEbpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4gNodeMid != 0) && (here-> BSIM4dNodePrime != 0)) {
			while (here->BSIM4GMdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4GMdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4gNodeMid != 0) && (here-> BSIM4gNodePrime != 0)) {
			while (here->BSIM4GMgpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4GMgpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4gNodeMid != 0) && (here-> BSIM4gNodeMid != 0)) {
			while (here->BSIM4GMgmPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4GMgmPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4gNodeMid != 0) && (here-> BSIM4gNodeExt != 0)) {
			while (here->BSIM4GMgePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4GMgePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4gNodeMid != 0) && (here-> BSIM4sNodePrime != 0)) {
			while (here->BSIM4GMspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4GMspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4gNodeMid != 0) && (here-> BSIM4bNodePrime != 0)) {
			while (here->BSIM4GMbpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4GMbpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4dNodePrime != 0) && (here-> BSIM4gNodeMid != 0)) {
			while (here->BSIM4DPgmPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4DPgmPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4gNodePrime != 0) && (here-> BSIM4gNodeMid != 0)) {
			while (here->BSIM4GPgmPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4GPgmPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4gNodeExt != 0) && (here-> BSIM4gNodeMid != 0)) {
			while (here->BSIM4GEgmPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4GEgmPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4sNodePrime != 0) && (here-> BSIM4gNodeMid != 0)) {
			while (here->BSIM4SPgmPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4SPgmPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4bNodePrime != 0) && (here-> BSIM4gNodeMid != 0)) {
			while (here->BSIM4BPgmPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4BPgmPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
                } /* IF */

                if ((here->BSIM4rbodyMod ==1) || (here->BSIM4rbodyMod ==2)) {

		i = 0 ;
		if ((here-> BSIM4dNodePrime != 0) && (here-> BSIM4dbNode != 0)) {
			while (here->BSIM4DPdbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4DPdbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4sNodePrime != 0) && (here-> BSIM4sbNode != 0)) {
			while (here->BSIM4SPsbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4SPsbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4dbNode != 0) && (here-> BSIM4dNodePrime != 0)) {
			while (here->BSIM4DBdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4DBdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4dbNode != 0) && (here-> BSIM4dbNode != 0)) {
			while (here->BSIM4DBdbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4DBdbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4dbNode != 0) && (here-> BSIM4bNodePrime != 0)) {
			while (here->BSIM4DBbpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4DBbpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4dbNode != 0) && (here-> BSIM4bNode != 0)) {
			while (here->BSIM4DBbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4DBbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4bNodePrime != 0) && (here-> BSIM4dbNode != 0)) {
			while (here->BSIM4BPdbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4BPdbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4bNodePrime != 0) && (here-> BSIM4bNode != 0)) {
			while (here->BSIM4BPbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4BPbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4bNodePrime != 0) && (here-> BSIM4sbNode != 0)) {
			while (here->BSIM4BPsbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4BPsbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4sbNode != 0) && (here-> BSIM4sNodePrime != 0)) {
			while (here->BSIM4SBspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4SBspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4sbNode != 0) && (here-> BSIM4bNodePrime != 0)) {
			while (here->BSIM4SBbpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4SBbpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4sbNode != 0) && (here-> BSIM4bNode != 0)) {
			while (here->BSIM4SBbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4SBbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4sbNode != 0) && (here-> BSIM4sbNode != 0)) {
			while (here->BSIM4SBsbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4SBsbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4bNode != 0) && (here-> BSIM4dbNode != 0)) {
			while (here->BSIM4BdbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4BdbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4bNode != 0) && (here-> BSIM4bNodePrime != 0)) {
			while (here->BSIM4BbpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4BbpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4bNode != 0) && (here-> BSIM4sbNode != 0)) {
			while (here->BSIM4BsbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4BsbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4bNode != 0) && (here-> BSIM4bNode != 0)) {
			while (here->BSIM4BbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4BbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
                } /* IF */

                if (model->BSIM4rdsMod) {

		i = 0 ;
		if ((here-> BSIM4dNode != 0) && (here-> BSIM4gNodePrime != 0)) {
			while (here->BSIM4DgpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4DgpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4dNode != 0) && (here-> BSIM4sNodePrime != 0)) {
			while (here->BSIM4DspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4DspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4dNode != 0) && (here-> BSIM4bNodePrime != 0)) {
			while (here->BSIM4DbpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4DbpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4sNode != 0) && (here-> BSIM4dNodePrime != 0)) {
			while (here->BSIM4SdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4SdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4sNode != 0) && (here-> BSIM4gNodePrime != 0)) {
			while (here->BSIM4SgpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4SgpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4sNode != 0) && (here-> BSIM4bNodePrime != 0)) {
			while (here->BSIM4SbpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4SbpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
                } /* IF */
	}
    }
    return(OK);
}

int
BSIM4bindCSCComplex(GENmodel *inModel, CKTcircuit *ckt)
{
    BSIM4model *model = (BSIM4model *)inModel;
    int i ;

    /*  loop through all the b4 models */
    for( ; model != NULL; model = model->BSIM4nextModel ) {
	BSIM4instance *here;

        /* loop through all the instances of the model */
        for (here = model->BSIM4instances; here != NULL ;
	    here = here->BSIM4nextInstance) {

		i = 0 ;
		if ((here-> BSIM4dNodePrime != 0) && (here-> BSIM4bNodePrime != 0)) {
			while (here->BSIM4DPbpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4DPbpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4gNodePrime != 0) && (here-> BSIM4bNodePrime != 0)) {
			while (here->BSIM4GPbpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4GPbpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4sNodePrime != 0) && (here-> BSIM4bNodePrime != 0)) {
			while (here->BSIM4SPbpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4SPbpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4bNodePrime != 0) && (here-> BSIM4dNodePrime != 0)) {
			while (here->BSIM4BPdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4BPdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4bNodePrime != 0) && (here-> BSIM4gNodePrime != 0)) {
			while (here->BSIM4BPgpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4BPgpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4bNodePrime != 0) && (here-> BSIM4sNodePrime != 0)) {
			while (here->BSIM4BPspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4BPspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4bNodePrime != 0) && (here-> BSIM4bNodePrime != 0)) {
			while (here->BSIM4BPbpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4BPbpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4dNode != 0) && (here-> BSIM4dNode != 0)) {
			while (here->BSIM4DdPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4DdPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4gNodePrime != 0) && (here-> BSIM4gNodePrime != 0)) {
			while (here->BSIM4GPgpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4GPgpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4sNode != 0) && (here-> BSIM4sNode != 0)) {
			while (here->BSIM4SsPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4SsPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4dNodePrime != 0) && (here-> BSIM4dNodePrime != 0)) {
			while (here->BSIM4DPdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4DPdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4sNodePrime != 0) && (here-> BSIM4sNodePrime != 0)) {
			while (here->BSIM4SPspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4SPspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4dNode != 0) && (here-> BSIM4dNodePrime != 0)) {
			while (here->BSIM4DdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4DdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4gNodePrime != 0) && (here-> BSIM4dNodePrime != 0)) {
			while (here->BSIM4GPdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4GPdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4gNodePrime != 0) && (here-> BSIM4sNodePrime != 0)) {
			while (here->BSIM4GPspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4GPspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4sNode != 0) && (here-> BSIM4sNodePrime != 0)) {
			while (here->BSIM4SspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4SspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4dNodePrime != 0) && (here-> BSIM4sNodePrime != 0)) {
			while (here->BSIM4DPspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4DPspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4dNodePrime != 0) && (here-> BSIM4dNode != 0)) {
			while (here->BSIM4DPdPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4DPdPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4dNodePrime != 0) && (here-> BSIM4gNodePrime != 0)) {
			while (here->BSIM4DPgpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4DPgpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4sNodePrime != 0) && (here-> BSIM4gNodePrime != 0)) {
			while (here->BSIM4SPgpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4SPgpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4sNodePrime != 0) && (here-> BSIM4sNode != 0)) {
			while (here->BSIM4SPsPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4SPsPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4sNodePrime != 0) && (here-> BSIM4dNodePrime != 0)) {
			while (here->BSIM4SPdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4SPdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4qNode != 0) && (here-> BSIM4qNode != 0)) {
			while (here->BSIM4QqPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4QqPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4qNode != 0) && (here-> BSIM4bNodePrime != 0)) {
			while (here->BSIM4QbpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4QbpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4qNode != 0) && (here-> BSIM4dNodePrime != 0)) {
			while (here->BSIM4QdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4QdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4qNode != 0) && (here-> BSIM4sNodePrime != 0)) {
			while (here->BSIM4QspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4QspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4qNode != 0) && (here-> BSIM4gNodePrime != 0)) {
			while (here->BSIM4QgpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4QgpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4dNodePrime != 0) && (here-> BSIM4qNode != 0)) {
			while (here->BSIM4DPqPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4DPqPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4sNodePrime != 0) && (here-> BSIM4qNode != 0)) {
			while (here->BSIM4SPqPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4SPqPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4gNodePrime != 0) && (here-> BSIM4qNode != 0)) {
			while (here->BSIM4GPqPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4GPqPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

                if (here->BSIM4rgateMod != 0) {

		i = 0 ;
		if ((here-> BSIM4gNodeExt != 0) && (here-> BSIM4gNodeExt != 0)) {
			while (here->BSIM4GEgePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4GEgePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4gNodeExt != 0) && (here-> BSIM4gNodePrime != 0)) {
			while (here->BSIM4GEgpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4GEgpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4gNodePrime != 0) && (here-> BSIM4gNodeExt != 0)) {
			while (here->BSIM4GPgePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4GPgePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4gNodeExt != 0) && (here-> BSIM4dNodePrime != 0)) {
			while (here->BSIM4GEdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4GEdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4gNodeExt != 0) && (here-> BSIM4sNodePrime != 0)) {
			while (here->BSIM4GEspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4GEspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4gNodeExt != 0) && (here-> BSIM4bNodePrime != 0)) {
			while (here->BSIM4GEbpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4GEbpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4gNodeMid != 0) && (here-> BSIM4dNodePrime != 0)) {
			while (here->BSIM4GMdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4GMdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4gNodeMid != 0) && (here-> BSIM4gNodePrime != 0)) {
			while (here->BSIM4GMgpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4GMgpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4gNodeMid != 0) && (here-> BSIM4gNodeMid != 0)) {
			while (here->BSIM4GMgmPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4GMgmPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4gNodeMid != 0) && (here-> BSIM4gNodeExt != 0)) {
			while (here->BSIM4GMgePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4GMgePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4gNodeMid != 0) && (here-> BSIM4sNodePrime != 0)) {
			while (here->BSIM4GMspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4GMspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4gNodeMid != 0) && (here-> BSIM4bNodePrime != 0)) {
			while (here->BSIM4GMbpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4GMbpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4dNodePrime != 0) && (here-> BSIM4gNodeMid != 0)) {
			while (here->BSIM4DPgmPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4DPgmPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4gNodePrime != 0) && (here-> BSIM4gNodeMid != 0)) {
			while (here->BSIM4GPgmPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4GPgmPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4gNodeExt != 0) && (here-> BSIM4gNodeMid != 0)) {
			while (here->BSIM4GEgmPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4GEgmPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4sNodePrime != 0) && (here-> BSIM4gNodeMid != 0)) {
			while (here->BSIM4SPgmPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4SPgmPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4bNodePrime != 0) && (here-> BSIM4gNodeMid != 0)) {
			while (here->BSIM4BPgmPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4BPgmPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
                } /* IF */

                if ((here->BSIM4rbodyMod ==1) || (here->BSIM4rbodyMod ==2)) {

		i = 0 ;
		if ((here-> BSIM4dNodePrime != 0) && (here-> BSIM4dbNode != 0)) {
			while (here->BSIM4DPdbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4DPdbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4sNodePrime != 0) && (here-> BSIM4sbNode != 0)) {
			while (here->BSIM4SPsbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4SPsbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4dbNode != 0) && (here-> BSIM4dNodePrime != 0)) {
			while (here->BSIM4DBdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4DBdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4dbNode != 0) && (here-> BSIM4dbNode != 0)) {
			while (here->BSIM4DBdbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4DBdbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4dbNode != 0) && (here-> BSIM4bNodePrime != 0)) {
			while (here->BSIM4DBbpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4DBbpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4dbNode != 0) && (here-> BSIM4bNode != 0)) {
			while (here->BSIM4DBbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4DBbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4bNodePrime != 0) && (here-> BSIM4dbNode != 0)) {
			while (here->BSIM4BPdbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4BPdbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4bNodePrime != 0) && (here-> BSIM4bNode != 0)) {
			while (here->BSIM4BPbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4BPbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4bNodePrime != 0) && (here-> BSIM4sbNode != 0)) {
			while (here->BSIM4BPsbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4BPsbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4sbNode != 0) && (here-> BSIM4sNodePrime != 0)) {
			while (here->BSIM4SBspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4SBspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4sbNode != 0) && (here-> BSIM4bNodePrime != 0)) {
			while (here->BSIM4SBbpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4SBbpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4sbNode != 0) && (here-> BSIM4bNode != 0)) {
			while (here->BSIM4SBbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4SBbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4sbNode != 0) && (here-> BSIM4sbNode != 0)) {
			while (here->BSIM4SBsbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4SBsbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4bNode != 0) && (here-> BSIM4dbNode != 0)) {
			while (here->BSIM4BdbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4BdbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4bNode != 0) && (here-> BSIM4bNodePrime != 0)) {
			while (here->BSIM4BbpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4BbpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4bNode != 0) && (here-> BSIM4sbNode != 0)) {
			while (here->BSIM4BsbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4BsbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4bNode != 0) && (here-> BSIM4bNode != 0)) {
			while (here->BSIM4BbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4BbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
                } /* IF */

                if (model->BSIM4rdsMod) {

		i = 0 ;
		if ((here-> BSIM4dNode != 0) && (here-> BSIM4gNodePrime != 0)) {
			while (here->BSIM4DgpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4DgpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4dNode != 0) && (here-> BSIM4sNodePrime != 0)) {
			while (here->BSIM4DspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4DspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4dNode != 0) && (here-> BSIM4bNodePrime != 0)) {
			while (here->BSIM4DbpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4DbpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4sNode != 0) && (here-> BSIM4dNodePrime != 0)) {
			while (here->BSIM4SdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4SdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4sNode != 0) && (here-> BSIM4gNodePrime != 0)) {
			while (here->BSIM4SgpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4SgpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> BSIM4sNode != 0) && (here-> BSIM4bNodePrime != 0)) {
			while (here->BSIM4SbpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4SbpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
                } /* IF */
	}
    }
    return(OK);
}
