/**********
Author: 2012 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "bsim4v6def.h"
#include "ngspice/sperror.h"

int
BSIM4v6bindCSC(GENmodel *inModel, CKTcircuit *ckt)
{
    BSIM4v6model *model = (BSIM4v6model *)inModel;
    BSIM4v6instance *here;
    int i ;

    for (; model != NULL; model = model->BSIM4v6nextModel)
    {    for (here = model->BSIM4v6instances; here != NULL; 
             here = here->BSIM4v6nextInstance)
         {

	/* SEARCH FOR THE NEW POSITIONS */

		i = 0 ;
		if ((here->BSIM4v6dNodePrime != 0) && (here->BSIM4v6bNodePrime != 0)) {
			while (here->BSIM4v6DPbpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v6DPbpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6gNodePrime != 0) && (here->BSIM4v6bNodePrime != 0)) {
			while (here->BSIM4v6GPbpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v6GPbpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6sNodePrime != 0) && (here->BSIM4v6bNodePrime != 0)) {
			while (here->BSIM4v6SPbpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v6SPbpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6bNodePrime != 0) && (here->BSIM4v6dNodePrime != 0)) {
			while (here->BSIM4v6BPdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v6BPdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6bNodePrime != 0) && (here->BSIM4v6gNodePrime != 0)) {
			while (here->BSIM4v6BPgpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v6BPgpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6bNodePrime != 0) && (here->BSIM4v6sNodePrime != 0)) {
			while (here->BSIM4v6BPspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v6BPspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6bNodePrime != 0) && (here->BSIM4v6bNodePrime != 0)) {
			while (here->BSIM4v6BPbpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v6BPbpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6dNode != 0) && (here->BSIM4v6dNode != 0)) {
			while (here->BSIM4v6DdPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v6DdPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6gNodePrime != 0) && (here->BSIM4v6gNodePrime != 0)) {
			while (here->BSIM4v6GPgpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v6GPgpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6sNode != 0) && (here->BSIM4v6sNode != 0)) {
			while (here->BSIM4v6SsPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v6SsPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6dNodePrime != 0) && (here->BSIM4v6dNodePrime != 0)) {
			while (here->BSIM4v6DPdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v6DPdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6sNodePrime != 0) && (here->BSIM4v6sNodePrime != 0)) {
			while (here->BSIM4v6SPspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v6SPspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6dNode != 0) && (here->BSIM4v6dNodePrime != 0)) {
			while (here->BSIM4v6DdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v6DdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6gNodePrime != 0) && (here->BSIM4v6dNodePrime != 0)) {
			while (here->BSIM4v6GPdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v6GPdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6gNodePrime != 0) && (here->BSIM4v6sNodePrime != 0)) {
			while (here->BSIM4v6GPspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v6GPspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6sNode != 0) && (here->BSIM4v6sNodePrime != 0)) {
			while (here->BSIM4v6SspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v6SspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6dNodePrime != 0) && (here->BSIM4v6sNodePrime != 0)) {
			while (here->BSIM4v6DPspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v6DPspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6dNodePrime != 0) && (here->BSIM4v6dNode != 0)) {
			while (here->BSIM4v6DPdPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v6DPdPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6dNodePrime != 0) && (here->BSIM4v6gNodePrime != 0)) {
			while (here->BSIM4v6DPgpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v6DPgpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6sNodePrime != 0) && (here->BSIM4v6gNodePrime != 0)) {
			while (here->BSIM4v6SPgpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v6SPgpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6sNodePrime != 0) && (here->BSIM4v6sNode != 0)) {
			while (here->BSIM4v6SPsPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v6SPsPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6sNodePrime != 0) && (here->BSIM4v6dNodePrime != 0)) {
			while (here->BSIM4v6SPdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v6SPdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6qNode != 0) && (here->BSIM4v6qNode != 0)) {
			while (here->BSIM4v6QqPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v6QqPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6qNode != 0) && (here->BSIM4v6bNodePrime != 0)) {
			while (here->BSIM4v6QbpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v6QbpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6qNode != 0) && (here->BSIM4v6dNodePrime != 0)) {
			while (here->BSIM4v6QdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v6QdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6qNode != 0) && (here->BSIM4v6sNodePrime != 0)) {
			while (here->BSIM4v6QspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v6QspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6qNode != 0) && (here->BSIM4v6gNodePrime != 0)) {
			while (here->BSIM4v6QgpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v6QgpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6dNodePrime != 0) && (here->BSIM4v6qNode != 0)) {
			while (here->BSIM4v6DPqPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v6DPqPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6sNodePrime != 0) && (here->BSIM4v6qNode != 0)) {
			while (here->BSIM4v6SPqPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v6SPqPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6gNodePrime != 0) && (here->BSIM4v6qNode != 0)) {
			while (here->BSIM4v6GPqPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v6GPqPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

	    if (here->BSIM4v6rgateMod != 0) {

		i = 0 ;
		if ((here->BSIM4v6gNodeExt != 0) && (here->BSIM4v6gNodeExt != 0)) {
			while (here->BSIM4v6GEgePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v6GEgePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6gNodeExt != 0) && (here->BSIM4v6gNodePrime != 0)) {
			while (here->BSIM4v6GEgpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v6GEgpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6gNodePrime != 0) && (here->BSIM4v6gNodeExt != 0)) {
			while (here->BSIM4v6GPgePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v6GPgePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6gNodeExt != 0) && (here->BSIM4v6dNodePrime != 0)) {
			while (here->BSIM4v6GEdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v6GEdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6gNodeExt != 0) && (here->BSIM4v6sNodePrime != 0)) {
			while (here->BSIM4v6GEspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v6GEspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6gNodeExt != 0) && (here->BSIM4v6bNodePrime != 0)) {
			while (here->BSIM4v6GEbpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v6GEbpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6gNodeMid != 0) && (here->BSIM4v6dNodePrime != 0)) {
			while (here->BSIM4v6GMdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v6GMdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6gNodeMid != 0) && (here->BSIM4v6gNodePrime != 0)) {
			while (here->BSIM4v6GMgpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v6GMgpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6gNodeMid != 0) && (here->BSIM4v6gNodeMid != 0)) {
			while (here->BSIM4v6GMgmPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v6GMgmPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6gNodeMid != 0) && (here->BSIM4v6gNodeExt != 0)) {
			while (here->BSIM4v6GMgePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v6GMgePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6gNodeMid != 0) && (here->BSIM4v6sNodePrime != 0)) {
			while (here->BSIM4v6GMspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v6GMspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6gNodeMid != 0) && (here->BSIM4v6bNodePrime != 0)) {
			while (here->BSIM4v6GMbpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v6GMbpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6dNodePrime != 0) && (here->BSIM4v6gNodeMid != 0)) {
			while (here->BSIM4v6DPgmPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v6DPgmPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6gNodePrime != 0) && (here->BSIM4v6gNodeMid != 0)) {
			while (here->BSIM4v6GPgmPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v6GPgmPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6gNodeExt != 0) && (here->BSIM4v6gNodeMid != 0)) {
			while (here->BSIM4v6GEgmPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v6GEgmPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6sNodePrime != 0) && (here->BSIM4v6gNodeMid != 0)) {
			while (here->BSIM4v6SPgmPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v6SPgmPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6bNodePrime != 0) && (here->BSIM4v6gNodeMid != 0)) {
			while (here->BSIM4v6BPgmPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v6BPgmPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

	    }

	    if ((here->BSIM4v6rbodyMod ==1) || (here->BSIM4v6rbodyMod ==2)) {

		i = 0 ;
		if ((here->BSIM4v6dNodePrime != 0) && (here->BSIM4v6dbNode != 0)) {
			while (here->BSIM4v6DPdbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v6DPdbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6sNodePrime != 0) && (here->BSIM4v6sbNode != 0)) {
			while (here->BSIM4v6SPsbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v6SPsbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6dbNode != 0) && (here->BSIM4v6dNodePrime != 0)) {
			while (here->BSIM4v6DBdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v6DBdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6dbNode != 0) && (here->BSIM4v6dbNode != 0)) {
			while (here->BSIM4v6DBdbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v6DBdbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6dbNode != 0) && (here->BSIM4v6bNodePrime != 0)) {
			while (here->BSIM4v6DBbpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v6DBbpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6dbNode != 0) && (here->BSIM4v6bNode != 0)) {
			while (here->BSIM4v6DBbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v6DBbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6bNodePrime != 0) && (here->BSIM4v6dbNode != 0)) {
			while (here->BSIM4v6BPdbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v6BPdbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6bNodePrime != 0) && (here->BSIM4v6bNode != 0)) {
			while (here->BSIM4v6BPbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v6BPbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6bNodePrime != 0) && (here->BSIM4v6sbNode != 0)) {
			while (here->BSIM4v6BPsbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v6BPsbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6sbNode != 0) && (here->BSIM4v6sNodePrime != 0)) {
			while (here->BSIM4v6SBspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v6SBspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6sbNode != 0) && (here->BSIM4v6bNodePrime != 0)) {
			while (here->BSIM4v6SBbpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v6SBbpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6sbNode != 0) && (here->BSIM4v6bNode != 0)) {
			while (here->BSIM4v6SBbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v6SBbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6sbNode != 0) && (here->BSIM4v6sbNode != 0)) {
			while (here->BSIM4v6SBsbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v6SBsbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6bNode != 0) && (here->BSIM4v6dbNode != 0)) {
			while (here->BSIM4v6BdbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v6BdbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6bNode != 0) && (here->BSIM4v6bNodePrime != 0)) {
			while (here->BSIM4v6BbpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v6BbpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6bNode != 0) && (here->BSIM4v6sbNode != 0)) {
			while (here->BSIM4v6BsbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v6BsbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6bNode != 0) && (here->BSIM4v6bNode != 0)) {
			while (here->BSIM4v6BbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v6BbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

	    }

            if (model->BSIM4v6rdsMod) {

		i = 0 ;
		if ((here->BSIM4v6dNode != 0) && (here->BSIM4v6gNodePrime != 0)) {
			while (here->BSIM4v6DgpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v6DgpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6dNode != 0) && (here->BSIM4v6sNodePrime != 0)) {
			while (here->BSIM4v6DspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v6DspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6dNode != 0) && (here->BSIM4v6bNodePrime != 0)) {
			while (here->BSIM4v6DbpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v6DbpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6sNode != 0) && (here->BSIM4v6dNodePrime != 0)) {
			while (here->BSIM4v6SdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v6SdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6sNode != 0) && (here->BSIM4v6gNodePrime != 0)) {
			while (here->BSIM4v6SgpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v6SgpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6sNode != 0) && (here->BSIM4v6bNodePrime != 0)) {
			while (here->BSIM4v6SbpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BSIM4v6SbpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

	    }
	}
    }
    return(OK);
}

int
BSIM4v6bindCSCComplex(GENmodel *inModel, CKTcircuit *ckt)
{
    BSIM4v6model *model = (BSIM4v6model *)inModel;
    BSIM4v6instance *here;
    int i ;

    for (; model != NULL; model = model->BSIM4v6nextModel)
    {    for (here = model->BSIM4v6instances; here != NULL; 
             here = here->BSIM4v6nextInstance)
         {

	/* SEARCH FOR THE NEW POSITIONS */

		i = 0 ;
		if ((here->BSIM4v6dNodePrime != 0) && (here->BSIM4v6bNodePrime != 0)) {
			while (here->BSIM4v6DPbpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v6DPbpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6gNodePrime != 0) && (here->BSIM4v6bNodePrime != 0)) {
			while (here->BSIM4v6GPbpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v6GPbpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6sNodePrime != 0) && (here->BSIM4v6bNodePrime != 0)) {
			while (here->BSIM4v6SPbpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v6SPbpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6bNodePrime != 0) && (here->BSIM4v6dNodePrime != 0)) {
			while (here->BSIM4v6BPdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v6BPdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6bNodePrime != 0) && (here->BSIM4v6gNodePrime != 0)) {
			while (here->BSIM4v6BPgpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v6BPgpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6bNodePrime != 0) && (here->BSIM4v6sNodePrime != 0)) {
			while (here->BSIM4v6BPspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v6BPspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6bNodePrime != 0) && (here->BSIM4v6bNodePrime != 0)) {
			while (here->BSIM4v6BPbpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v6BPbpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6dNode != 0) && (here->BSIM4v6dNode != 0)) {
			while (here->BSIM4v6DdPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v6DdPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6gNodePrime != 0) && (here->BSIM4v6gNodePrime != 0)) {
			while (here->BSIM4v6GPgpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v6GPgpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6sNode != 0) && (here->BSIM4v6sNode != 0)) {
			while (here->BSIM4v6SsPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v6SsPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6dNodePrime != 0) && (here->BSIM4v6dNodePrime != 0)) {
			while (here->BSIM4v6DPdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v6DPdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6sNodePrime != 0) && (here->BSIM4v6sNodePrime != 0)) {
			while (here->BSIM4v6SPspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v6SPspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6dNode != 0) && (here->BSIM4v6dNodePrime != 0)) {
			while (here->BSIM4v6DdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v6DdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6gNodePrime != 0) && (here->BSIM4v6dNodePrime != 0)) {
			while (here->BSIM4v6GPdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v6GPdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6gNodePrime != 0) && (here->BSIM4v6sNodePrime != 0)) {
			while (here->BSIM4v6GPspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v6GPspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6sNode != 0) && (here->BSIM4v6sNodePrime != 0)) {
			while (here->BSIM4v6SspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v6SspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6dNodePrime != 0) && (here->BSIM4v6sNodePrime != 0)) {
			while (here->BSIM4v6DPspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v6DPspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6dNodePrime != 0) && (here->BSIM4v6dNode != 0)) {
			while (here->BSIM4v6DPdPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v6DPdPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6dNodePrime != 0) && (here->BSIM4v6gNodePrime != 0)) {
			while (here->BSIM4v6DPgpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v6DPgpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6sNodePrime != 0) && (here->BSIM4v6gNodePrime != 0)) {
			while (here->BSIM4v6SPgpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v6SPgpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6sNodePrime != 0) && (here->BSIM4v6sNode != 0)) {
			while (here->BSIM4v6SPsPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v6SPsPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6sNodePrime != 0) && (here->BSIM4v6dNodePrime != 0)) {
			while (here->BSIM4v6SPdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v6SPdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6qNode != 0) && (here->BSIM4v6qNode != 0)) {
			while (here->BSIM4v6QqPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v6QqPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6qNode != 0) && (here->BSIM4v6bNodePrime != 0)) {
			while (here->BSIM4v6QbpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v6QbpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6qNode != 0) && (here->BSIM4v6dNodePrime != 0)) {
			while (here->BSIM4v6QdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v6QdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6qNode != 0) && (here->BSIM4v6sNodePrime != 0)) {
			while (here->BSIM4v6QspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v6QspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6qNode != 0) && (here->BSIM4v6gNodePrime != 0)) {
			while (here->BSIM4v6QgpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v6QgpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6dNodePrime != 0) && (here->BSIM4v6qNode != 0)) {
			while (here->BSIM4v6DPqPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v6DPqPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6sNodePrime != 0) && (here->BSIM4v6qNode != 0)) {
			while (here->BSIM4v6SPqPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v6SPqPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6gNodePrime != 0) && (here->BSIM4v6qNode != 0)) {
			while (here->BSIM4v6GPqPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v6GPqPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

	    if (here->BSIM4v6rgateMod != 0) {

		i = 0 ;
		if ((here->BSIM4v6gNodeExt != 0) && (here->BSIM4v6gNodeExt != 0)) {
			while (here->BSIM4v6GEgePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v6GEgePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6gNodeExt != 0) && (here->BSIM4v6gNodePrime != 0)) {
			while (here->BSIM4v6GEgpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v6GEgpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6gNodePrime != 0) && (here->BSIM4v6gNodeExt != 0)) {
			while (here->BSIM4v6GPgePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v6GPgePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6gNodeExt != 0) && (here->BSIM4v6dNodePrime != 0)) {
			while (here->BSIM4v6GEdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v6GEdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6gNodeExt != 0) && (here->BSIM4v6sNodePrime != 0)) {
			while (here->BSIM4v6GEspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v6GEspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6gNodeExt != 0) && (here->BSIM4v6bNodePrime != 0)) {
			while (here->BSIM4v6GEbpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v6GEbpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6gNodeMid != 0) && (here->BSIM4v6dNodePrime != 0)) {
			while (here->BSIM4v6GMdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v6GMdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6gNodeMid != 0) && (here->BSIM4v6gNodePrime != 0)) {
			while (here->BSIM4v6GMgpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v6GMgpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6gNodeMid != 0) && (here->BSIM4v6gNodeMid != 0)) {
			while (here->BSIM4v6GMgmPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v6GMgmPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6gNodeMid != 0) && (here->BSIM4v6gNodeExt != 0)) {
			while (here->BSIM4v6GMgePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v6GMgePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6gNodeMid != 0) && (here->BSIM4v6sNodePrime != 0)) {
			while (here->BSIM4v6GMspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v6GMspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6gNodeMid != 0) && (here->BSIM4v6bNodePrime != 0)) {
			while (here->BSIM4v6GMbpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v6GMbpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6dNodePrime != 0) && (here->BSIM4v6gNodeMid != 0)) {
			while (here->BSIM4v6DPgmPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v6DPgmPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6gNodePrime != 0) && (here->BSIM4v6gNodeMid != 0)) {
			while (here->BSIM4v6GPgmPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v6GPgmPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6gNodeExt != 0) && (here->BSIM4v6gNodeMid != 0)) {
			while (here->BSIM4v6GEgmPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v6GEgmPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6sNodePrime != 0) && (here->BSIM4v6gNodeMid != 0)) {
			while (here->BSIM4v6SPgmPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v6SPgmPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6bNodePrime != 0) && (here->BSIM4v6gNodeMid != 0)) {
			while (here->BSIM4v6BPgmPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v6BPgmPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

	    }

	    if ((here->BSIM4v6rbodyMod ==1) || (here->BSIM4v6rbodyMod ==2)) {

		i = 0 ;
		if ((here->BSIM4v6dNodePrime != 0) && (here->BSIM4v6dbNode != 0)) {
			while (here->BSIM4v6DPdbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v6DPdbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6sNodePrime != 0) && (here->BSIM4v6sbNode != 0)) {
			while (here->BSIM4v6SPsbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v6SPsbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6dbNode != 0) && (here->BSIM4v6dNodePrime != 0)) {
			while (here->BSIM4v6DBdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v6DBdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6dbNode != 0) && (here->BSIM4v6dbNode != 0)) {
			while (here->BSIM4v6DBdbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v6DBdbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6dbNode != 0) && (here->BSIM4v6bNodePrime != 0)) {
			while (here->BSIM4v6DBbpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v6DBbpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6dbNode != 0) && (here->BSIM4v6bNode != 0)) {
			while (here->BSIM4v6DBbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v6DBbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6bNodePrime != 0) && (here->BSIM4v6dbNode != 0)) {
			while (here->BSIM4v6BPdbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v6BPdbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6bNodePrime != 0) && (here->BSIM4v6bNode != 0)) {
			while (here->BSIM4v6BPbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v6BPbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6bNodePrime != 0) && (here->BSIM4v6sbNode != 0)) {
			while (here->BSIM4v6BPsbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v6BPsbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6sbNode != 0) && (here->BSIM4v6sNodePrime != 0)) {
			while (here->BSIM4v6SBspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v6SBspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6sbNode != 0) && (here->BSIM4v6bNodePrime != 0)) {
			while (here->BSIM4v6SBbpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v6SBbpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6sbNode != 0) && (here->BSIM4v6bNode != 0)) {
			while (here->BSIM4v6SBbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v6SBbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6sbNode != 0) && (here->BSIM4v6sbNode != 0)) {
			while (here->BSIM4v6SBsbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v6SBsbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6bNode != 0) && (here->BSIM4v6dbNode != 0)) {
			while (here->BSIM4v6BdbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v6BdbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6bNode != 0) && (here->BSIM4v6bNodePrime != 0)) {
			while (here->BSIM4v6BbpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v6BbpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6bNode != 0) && (here->BSIM4v6sbNode != 0)) {
			while (here->BSIM4v6BsbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v6BsbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6bNode != 0) && (here->BSIM4v6bNode != 0)) {
			while (here->BSIM4v6BbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v6BbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

	    }

            if (model->BSIM4v6rdsMod) {

		i = 0 ;
		if ((here->BSIM4v6dNode != 0) && (here->BSIM4v6gNodePrime != 0)) {
			while (here->BSIM4v6DgpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v6DgpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6dNode != 0) && (here->BSIM4v6sNodePrime != 0)) {
			while (here->BSIM4v6DspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v6DspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6dNode != 0) && (here->BSIM4v6bNodePrime != 0)) {
			while (here->BSIM4v6DbpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v6DbpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6sNode != 0) && (here->BSIM4v6dNodePrime != 0)) {
			while (here->BSIM4v6SdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v6SdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6sNode != 0) && (here->BSIM4v6gNodePrime != 0)) {
			while (here->BSIM4v6SgpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v6SgpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6sNode != 0) && (here->BSIM4v6bNodePrime != 0)) {
			while (here->BSIM4v6SbpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BSIM4v6SbpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

	    }
	}
    }
    return(OK);
}

int
BSIM4v6bindCSCComplexToReal (GENmodel *inModel, CKTcircuit *ckt)
{
    BSIM4v6model *model = (BSIM4v6model *)inModel ;
    BSIM4v6instance *here ;
    int i ;

    /*  loop through all the bsim4v6 models */
    for ( ; model != NULL ; model = model->BSIM4v6nextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->BSIM4v6instances ; here != NULL ; here = here->BSIM4v6nextInstance)
        {
            i = 0 ;
            if ((here->BSIM4v6dNodePrime != 0) && (here->BSIM4v6bNodePrime != 0))
            {
                while (here->BSIM4v6DPbpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v6DPbpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v6gNodePrime != 0) && (here->BSIM4v6bNodePrime != 0))
            {
                while (here->BSIM4v6GPbpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v6GPbpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v6sNodePrime != 0) && (here->BSIM4v6bNodePrime != 0))
            {
                while (here->BSIM4v6SPbpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v6SPbpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v6bNodePrime != 0) && (here->BSIM4v6dNodePrime != 0))
            {
                while (here->BSIM4v6BPdpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v6BPdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v6bNodePrime != 0) && (here->BSIM4v6gNodePrime != 0))
            {
                while (here->BSIM4v6BPgpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v6BPgpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v6bNodePrime != 0) && (here->BSIM4v6sNodePrime != 0))
            {
                while (here->BSIM4v6BPspPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v6BPspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v6bNodePrime != 0) && (here->BSIM4v6bNodePrime != 0))
            {
                while (here->BSIM4v6BPbpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v6BPbpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v6dNode != 0) && (here->BSIM4v6dNode != 0))
            {
                while (here->BSIM4v6DdPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v6DdPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v6gNodePrime != 0) && (here->BSIM4v6gNodePrime != 0))
            {
                while (here->BSIM4v6GPgpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v6GPgpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v6sNode != 0) && (here->BSIM4v6sNode != 0))
            {
                while (here->BSIM4v6SsPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v6SsPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v6dNodePrime != 0) && (here->BSIM4v6dNodePrime != 0))
            {
                while (here->BSIM4v6DPdpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v6DPdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v6sNodePrime != 0) && (here->BSIM4v6sNodePrime != 0))
            {
                while (here->BSIM4v6SPspPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v6SPspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v6dNode != 0) && (here->BSIM4v6dNodePrime != 0))
            {
                while (here->BSIM4v6DdpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v6DdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v6gNodePrime != 0) && (here->BSIM4v6dNodePrime != 0))
            {
                while (here->BSIM4v6GPdpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v6GPdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v6gNodePrime != 0) && (here->BSIM4v6sNodePrime != 0))
            {
                while (here->BSIM4v6GPspPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v6GPspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v6sNode != 0) && (here->BSIM4v6sNodePrime != 0))
            {
                while (here->BSIM4v6SspPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v6SspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v6dNodePrime != 0) && (here->BSIM4v6sNodePrime != 0))
            {
                while (here->BSIM4v6DPspPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v6DPspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v6dNodePrime != 0) && (here->BSIM4v6dNode != 0))
            {
                while (here->BSIM4v6DPdPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v6DPdPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v6dNodePrime != 0) && (here->BSIM4v6gNodePrime != 0))
            {
                while (here->BSIM4v6DPgpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v6DPgpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v6sNodePrime != 0) && (here->BSIM4v6gNodePrime != 0))
            {
                while (here->BSIM4v6SPgpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v6SPgpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v6sNodePrime != 0) && (here->BSIM4v6sNode != 0))
            {
                while (here->BSIM4v6SPsPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v6SPsPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v6sNodePrime != 0) && (here->BSIM4v6dNodePrime != 0))
            {
                while (here->BSIM4v6SPdpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v6SPdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v6qNode != 0) && (here->BSIM4v6qNode != 0))
            {
                while (here->BSIM4v6QqPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v6QqPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v6qNode != 0) && (here->BSIM4v6bNodePrime != 0))
            {
                while (here->BSIM4v6QbpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v6QbpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v6qNode != 0) && (here->BSIM4v6dNodePrime != 0))
            {
                while (here->BSIM4v6QdpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v6QdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v6qNode != 0) && (here->BSIM4v6sNodePrime != 0))
            {
                while (here->BSIM4v6QspPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v6QspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v6qNode != 0) && (here->BSIM4v6gNodePrime != 0))
            {
                while (here->BSIM4v6QgpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v6QgpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v6dNodePrime != 0) && (here->BSIM4v6qNode != 0))
            {
                while (here->BSIM4v6DPqPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v6DPqPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v6sNodePrime != 0) && (here->BSIM4v6qNode != 0))
            {
                while (here->BSIM4v6SPqPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v6SPqPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v6gNodePrime != 0) && (here->BSIM4v6qNode != 0))
            {
                while (here->BSIM4v6GPqPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v6GPqPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v6gNodeExt != 0) && (here->BSIM4v6gNodeExt != 0))
            {
                while (here->BSIM4v6GEgePtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v6GEgePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v6gNodeExt != 0) && (here->BSIM4v6gNodePrime != 0))
            {
                while (here->BSIM4v6GEgpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v6GEgpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v6gNodePrime != 0) && (here->BSIM4v6gNodeExt != 0))
            {
                while (here->BSIM4v6GPgePtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v6GPgePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v6gNodeExt != 0) && (here->BSIM4v6dNodePrime != 0))
            {
                while (here->BSIM4v6GEdpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v6GEdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v6gNodeExt != 0) && (here->BSIM4v6sNodePrime != 0))
            {
                while (here->BSIM4v6GEspPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v6GEspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v6gNodeExt != 0) && (here->BSIM4v6bNodePrime != 0))
            {
                while (here->BSIM4v6GEbpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v6GEbpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v6gNodeMid != 0) && (here->BSIM4v6dNodePrime != 0))
            {
                while (here->BSIM4v6GMdpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v6GMdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v6gNodeMid != 0) && (here->BSIM4v6gNodePrime != 0))
            {
                while (here->BSIM4v6GMgpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v6GMgpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v6gNodeMid != 0) && (here->BSIM4v6gNodeMid != 0))
            {
                while (here->BSIM4v6GMgmPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v6GMgmPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v6gNodeMid != 0) && (here->BSIM4v6gNodeExt != 0))
            {
                while (here->BSIM4v6GMgePtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v6GMgePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v6gNodeMid != 0) && (here->BSIM4v6sNodePrime != 0))
            {
                while (here->BSIM4v6GMspPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v6GMspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v6gNodeMid != 0) && (here->BSIM4v6bNodePrime != 0))
            {
                while (here->BSIM4v6GMbpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v6GMbpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v6dNodePrime != 0) && (here->BSIM4v6gNodeMid != 0))
            {
                while (here->BSIM4v6DPgmPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v6DPgmPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v6gNodePrime != 0) && (here->BSIM4v6gNodeMid != 0))
            {
                while (here->BSIM4v6GPgmPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v6GPgmPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v6gNodeExt != 0) && (here->BSIM4v6gNodeMid != 0))
            {
                while (here->BSIM4v6GEgmPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v6GEgmPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v6sNodePrime != 0) && (here->BSIM4v6gNodeMid != 0))
            {
                while (here->BSIM4v6SPgmPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v6SPgmPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v6bNodePrime != 0) && (here->BSIM4v6gNodeMid != 0))
            {
                while (here->BSIM4v6BPgmPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v6BPgmPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v6dNodePrime != 0) && (here->BSIM4v6dbNode != 0))
            {
                while (here->BSIM4v6DPdbPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v6DPdbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v6sNodePrime != 0) && (here->BSIM4v6sbNode != 0))
            {
                while (here->BSIM4v6SPsbPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v6SPsbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v6dbNode != 0) && (here->BSIM4v6dNodePrime != 0))
            {
                while (here->BSIM4v6DBdpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v6DBdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v6dbNode != 0) && (here->BSIM4v6dbNode != 0))
            {
                while (here->BSIM4v6DBdbPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v6DBdbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v6dbNode != 0) && (here->BSIM4v6bNodePrime != 0))
            {
                while (here->BSIM4v6DBbpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v6DBbpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v6dbNode != 0) && (here->BSIM4v6bNode != 0))
            {
                while (here->BSIM4v6DBbPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v6DBbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v6bNodePrime != 0) && (here->BSIM4v6dbNode != 0))
            {
                while (here->BSIM4v6BPdbPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v6BPdbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v6bNodePrime != 0) && (here->BSIM4v6bNode != 0))
            {
                while (here->BSIM4v6BPbPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v6BPbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v6bNodePrime != 0) && (here->BSIM4v6sbNode != 0))
            {
                while (here->BSIM4v6BPsbPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v6BPsbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v6sbNode != 0) && (here->BSIM4v6sNodePrime != 0))
            {
                while (here->BSIM4v6SBspPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v6SBspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v6sbNode != 0) && (here->BSIM4v6bNodePrime != 0))
            {
                while (here->BSIM4v6SBbpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v6SBbpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v6sbNode != 0) && (here->BSIM4v6bNode != 0))
            {
                while (here->BSIM4v6SBbPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v6SBbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v6sbNode != 0) && (here->BSIM4v6sbNode != 0))
            {
                while (here->BSIM4v6SBsbPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v6SBsbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v6bNode != 0) && (here->BSIM4v6dbNode != 0))
            {
                while (here->BSIM4v6BdbPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v6BdbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v6bNode != 0) && (here->BSIM4v6bNodePrime != 0))
            {
                while (here->BSIM4v6BbpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v6BbpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v6bNode != 0) && (here->BSIM4v6sbNode != 0))
            {
                while (here->BSIM4v6BsbPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v6BsbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v6bNode != 0) && (here->BSIM4v6bNode != 0))
            {
                while (here->BSIM4v6BbPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v6BbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v6dNode != 0) && (here->BSIM4v6gNodePrime != 0))
            {
                while (here->BSIM4v6DgpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v6DgpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v6dNode != 0) && (here->BSIM4v6sNodePrime != 0))
            {
                while (here->BSIM4v6DspPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v6DspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v6dNode != 0) && (here->BSIM4v6bNodePrime != 0))
            {
                while (here->BSIM4v6DbpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v6DbpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v6sNode != 0) && (here->BSIM4v6dNodePrime != 0))
            {
                while (here->BSIM4v6SdpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v6SdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v6sNode != 0) && (here->BSIM4v6gNodePrime != 0))
            {
                while (here->BSIM4v6SgpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v6SgpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BSIM4v6sNode != 0) && (here->BSIM4v6bNodePrime != 0))
            {
                while (here->BSIM4v6SbpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BSIM4v6SbpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }
        }
    }

    return (OK) ;
}