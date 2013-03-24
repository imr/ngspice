/**********
Author: 2012 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "bsim4v6def.h"
#include "ngspice/sperror.h"

int
BSIM4v6bindklu(GENmodel *inModel, CKTcircuit *ckt)
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
			while (here->BSIM4v6DPbpPtr != ckt->CKTmatrix->CKTkluBind_Sparse [i]) i ++ ;
			here->BSIM4v6DPbpPtr = ckt->CKTmatrix->CKTkluBind_KLU [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6gNodePrime != 0) && (here->BSIM4v6bNodePrime != 0)) {
			while (here->BSIM4v6GPbpPtr != ckt->CKTmatrix->CKTkluBind_Sparse [i]) i ++ ;
			here->BSIM4v6GPbpPtr = ckt->CKTmatrix->CKTkluBind_KLU [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6sNodePrime != 0) && (here->BSIM4v6bNodePrime != 0)) {
			while (here->BSIM4v6SPbpPtr != ckt->CKTmatrix->CKTkluBind_Sparse [i]) i ++ ;
			here->BSIM4v6SPbpPtr = ckt->CKTmatrix->CKTkluBind_KLU [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6bNodePrime != 0) && (here->BSIM4v6dNodePrime != 0)) {
			while (here->BSIM4v6BPdpPtr != ckt->CKTmatrix->CKTkluBind_Sparse [i]) i ++ ;
			here->BSIM4v6BPdpPtr = ckt->CKTmatrix->CKTkluBind_KLU [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6bNodePrime != 0) && (here->BSIM4v6gNodePrime != 0)) {
			while (here->BSIM4v6BPgpPtr != ckt->CKTmatrix->CKTkluBind_Sparse [i]) i ++ ;
			here->BSIM4v6BPgpPtr = ckt->CKTmatrix->CKTkluBind_KLU [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6bNodePrime != 0) && (here->BSIM4v6sNodePrime != 0)) {
			while (here->BSIM4v6BPspPtr != ckt->CKTmatrix->CKTkluBind_Sparse [i]) i ++ ;
			here->BSIM4v6BPspPtr = ckt->CKTmatrix->CKTkluBind_KLU [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6bNodePrime != 0) && (here->BSIM4v6bNodePrime != 0)) {
			while (here->BSIM4v6BPbpPtr != ckt->CKTmatrix->CKTkluBind_Sparse [i]) i ++ ;
			here->BSIM4v6BPbpPtr = ckt->CKTmatrix->CKTkluBind_KLU [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6dNode != 0) && (here->BSIM4v6dNode != 0)) {
			while (here->BSIM4v6DdPtr != ckt->CKTmatrix->CKTkluBind_Sparse [i]) i ++ ;
			here->BSIM4v6DdPtr = ckt->CKTmatrix->CKTkluBind_KLU [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6gNodePrime != 0) && (here->BSIM4v6gNodePrime != 0)) {
			while (here->BSIM4v6GPgpPtr != ckt->CKTmatrix->CKTkluBind_Sparse [i]) i ++ ;
			here->BSIM4v6GPgpPtr = ckt->CKTmatrix->CKTkluBind_KLU [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6sNode != 0) && (here->BSIM4v6sNode != 0)) {
			while (here->BSIM4v6SsPtr != ckt->CKTmatrix->CKTkluBind_Sparse [i]) i ++ ;
			here->BSIM4v6SsPtr = ckt->CKTmatrix->CKTkluBind_KLU [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6dNodePrime != 0) && (here->BSIM4v6dNodePrime != 0)) {
			while (here->BSIM4v6DPdpPtr != ckt->CKTmatrix->CKTkluBind_Sparse [i]) i ++ ;
			here->BSIM4v6DPdpPtr = ckt->CKTmatrix->CKTkluBind_KLU [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6sNodePrime != 0) && (here->BSIM4v6sNodePrime != 0)) {
			while (here->BSIM4v6SPspPtr != ckt->CKTmatrix->CKTkluBind_Sparse [i]) i ++ ;
			here->BSIM4v6SPspPtr = ckt->CKTmatrix->CKTkluBind_KLU [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6dNode != 0) && (here->BSIM4v6dNodePrime != 0)) {
			while (here->BSIM4v6DdpPtr != ckt->CKTmatrix->CKTkluBind_Sparse [i]) i ++ ;
			here->BSIM4v6DdpPtr = ckt->CKTmatrix->CKTkluBind_KLU [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6gNodePrime != 0) && (here->BSIM4v6dNodePrime != 0)) {
			while (here->BSIM4v6GPdpPtr != ckt->CKTmatrix->CKTkluBind_Sparse [i]) i ++ ;
			here->BSIM4v6GPdpPtr = ckt->CKTmatrix->CKTkluBind_KLU [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6gNodePrime != 0) && (here->BSIM4v6sNodePrime != 0)) {
			while (here->BSIM4v6GPspPtr != ckt->CKTmatrix->CKTkluBind_Sparse [i]) i ++ ;
			here->BSIM4v6GPspPtr = ckt->CKTmatrix->CKTkluBind_KLU [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6sNode != 0) && (here->BSIM4v6sNodePrime != 0)) {
			while (here->BSIM4v6SspPtr != ckt->CKTmatrix->CKTkluBind_Sparse [i]) i ++ ;
			here->BSIM4v6SspPtr = ckt->CKTmatrix->CKTkluBind_KLU [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6dNodePrime != 0) && (here->BSIM4v6sNodePrime != 0)) {
			while (here->BSIM4v6DPspPtr != ckt->CKTmatrix->CKTkluBind_Sparse [i]) i ++ ;
			here->BSIM4v6DPspPtr = ckt->CKTmatrix->CKTkluBind_KLU [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6dNodePrime != 0) && (here->BSIM4v6dNode != 0)) {
			while (here->BSIM4v6DPdPtr != ckt->CKTmatrix->CKTkluBind_Sparse [i]) i ++ ;
			here->BSIM4v6DPdPtr = ckt->CKTmatrix->CKTkluBind_KLU [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6dNodePrime != 0) && (here->BSIM4v6gNodePrime != 0)) {
			while (here->BSIM4v6DPgpPtr != ckt->CKTmatrix->CKTkluBind_Sparse [i]) i ++ ;
			here->BSIM4v6DPgpPtr = ckt->CKTmatrix->CKTkluBind_KLU [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6sNodePrime != 0) && (here->BSIM4v6gNodePrime != 0)) {
			while (here->BSIM4v6SPgpPtr != ckt->CKTmatrix->CKTkluBind_Sparse [i]) i ++ ;
			here->BSIM4v6SPgpPtr = ckt->CKTmatrix->CKTkluBind_KLU [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6sNodePrime != 0) && (here->BSIM4v6sNode != 0)) {
			while (here->BSIM4v6SPsPtr != ckt->CKTmatrix->CKTkluBind_Sparse [i]) i ++ ;
			here->BSIM4v6SPsPtr = ckt->CKTmatrix->CKTkluBind_KLU [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6sNodePrime != 0) && (here->BSIM4v6dNodePrime != 0)) {
			while (here->BSIM4v6SPdpPtr != ckt->CKTmatrix->CKTkluBind_Sparse [i]) i ++ ;
			here->BSIM4v6SPdpPtr = ckt->CKTmatrix->CKTkluBind_KLU [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6qNode != 0) && (here->BSIM4v6qNode != 0)) {
			while (here->BSIM4v6QqPtr != ckt->CKTmatrix->CKTkluBind_Sparse [i]) i ++ ;
			here->BSIM4v6QqPtr = ckt->CKTmatrix->CKTkluBind_KLU [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6qNode != 0) && (here->BSIM4v6bNodePrime != 0)) {
			while (here->BSIM4v6QbpPtr != ckt->CKTmatrix->CKTkluBind_Sparse [i]) i ++ ;
			here->BSIM4v6QbpPtr = ckt->CKTmatrix->CKTkluBind_KLU [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6qNode != 0) && (here->BSIM4v6dNodePrime != 0)) {
			while (here->BSIM4v6QdpPtr != ckt->CKTmatrix->CKTkluBind_Sparse [i]) i ++ ;
			here->BSIM4v6QdpPtr = ckt->CKTmatrix->CKTkluBind_KLU [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6qNode != 0) && (here->BSIM4v6sNodePrime != 0)) {
			while (here->BSIM4v6QspPtr != ckt->CKTmatrix->CKTkluBind_Sparse [i]) i ++ ;
			here->BSIM4v6QspPtr = ckt->CKTmatrix->CKTkluBind_KLU [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6qNode != 0) && (here->BSIM4v6gNodePrime != 0)) {
			while (here->BSIM4v6QgpPtr != ckt->CKTmatrix->CKTkluBind_Sparse [i]) i ++ ;
			here->BSIM4v6QgpPtr = ckt->CKTmatrix->CKTkluBind_KLU [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6dNodePrime != 0) && (here->BSIM4v6qNode != 0)) {
			while (here->BSIM4v6DPqPtr != ckt->CKTmatrix->CKTkluBind_Sparse [i]) i ++ ;
			here->BSIM4v6DPqPtr = ckt->CKTmatrix->CKTkluBind_KLU [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6sNodePrime != 0) && (here->BSIM4v6qNode != 0)) {
			while (here->BSIM4v6SPqPtr != ckt->CKTmatrix->CKTkluBind_Sparse [i]) i ++ ;
			here->BSIM4v6SPqPtr = ckt->CKTmatrix->CKTkluBind_KLU [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6gNodePrime != 0) && (here->BSIM4v6qNode != 0)) {
			while (here->BSIM4v6GPqPtr != ckt->CKTmatrix->CKTkluBind_Sparse [i]) i ++ ;
			here->BSIM4v6GPqPtr = ckt->CKTmatrix->CKTkluBind_KLU [i] ;
		}

	    if (here->BSIM4v6rgateMod != 0) {

		i = 0 ;
		if ((here->BSIM4v6gNodeExt != 0) && (here->BSIM4v6gNodeExt != 0)) {
			while (here->BSIM4v6GEgePtr != ckt->CKTmatrix->CKTkluBind_Sparse [i]) i ++ ;
			here->BSIM4v6GEgePtr = ckt->CKTmatrix->CKTkluBind_KLU [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6gNodeExt != 0) && (here->BSIM4v6gNodePrime != 0)) {
			while (here->BSIM4v6GEgpPtr != ckt->CKTmatrix->CKTkluBind_Sparse [i]) i ++ ;
			here->BSIM4v6GEgpPtr = ckt->CKTmatrix->CKTkluBind_KLU [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6gNodePrime != 0) && (here->BSIM4v6gNodeExt != 0)) {
			while (here->BSIM4v6GPgePtr != ckt->CKTmatrix->CKTkluBind_Sparse [i]) i ++ ;
			here->BSIM4v6GPgePtr = ckt->CKTmatrix->CKTkluBind_KLU [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6gNodeExt != 0) && (here->BSIM4v6dNodePrime != 0)) {
			while (here->BSIM4v6GEdpPtr != ckt->CKTmatrix->CKTkluBind_Sparse [i]) i ++ ;
			here->BSIM4v6GEdpPtr = ckt->CKTmatrix->CKTkluBind_KLU [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6gNodeExt != 0) && (here->BSIM4v6sNodePrime != 0)) {
			while (here->BSIM4v6GEspPtr != ckt->CKTmatrix->CKTkluBind_Sparse [i]) i ++ ;
			here->BSIM4v6GEspPtr = ckt->CKTmatrix->CKTkluBind_KLU [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6gNodeExt != 0) && (here->BSIM4v6bNodePrime != 0)) {
			while (here->BSIM4v6GEbpPtr != ckt->CKTmatrix->CKTkluBind_Sparse [i]) i ++ ;
			here->BSIM4v6GEbpPtr = ckt->CKTmatrix->CKTkluBind_KLU [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6gNodeMid != 0) && (here->BSIM4v6dNodePrime != 0)) {
			while (here->BSIM4v6GMdpPtr != ckt->CKTmatrix->CKTkluBind_Sparse [i]) i ++ ;
			here->BSIM4v6GMdpPtr = ckt->CKTmatrix->CKTkluBind_KLU [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6gNodeMid != 0) && (here->BSIM4v6gNodePrime != 0)) {
			while (here->BSIM4v6GMgpPtr != ckt->CKTmatrix->CKTkluBind_Sparse [i]) i ++ ;
			here->BSIM4v6GMgpPtr = ckt->CKTmatrix->CKTkluBind_KLU [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6gNodeMid != 0) && (here->BSIM4v6gNodeMid != 0)) {
			while (here->BSIM4v6GMgmPtr != ckt->CKTmatrix->CKTkluBind_Sparse [i]) i ++ ;
			here->BSIM4v6GMgmPtr = ckt->CKTmatrix->CKTkluBind_KLU [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6gNodeMid != 0) && (here->BSIM4v6gNodeExt != 0)) {
			while (here->BSIM4v6GMgePtr != ckt->CKTmatrix->CKTkluBind_Sparse [i]) i ++ ;
			here->BSIM4v6GMgePtr = ckt->CKTmatrix->CKTkluBind_KLU [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6gNodeMid != 0) && (here->BSIM4v6sNodePrime != 0)) {
			while (here->BSIM4v6GMspPtr != ckt->CKTmatrix->CKTkluBind_Sparse [i]) i ++ ;
			here->BSIM4v6GMspPtr = ckt->CKTmatrix->CKTkluBind_KLU [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6gNodeMid != 0) && (here->BSIM4v6bNodePrime != 0)) {
			while (here->BSIM4v6GMbpPtr != ckt->CKTmatrix->CKTkluBind_Sparse [i]) i ++ ;
			here->BSIM4v6GMbpPtr = ckt->CKTmatrix->CKTkluBind_KLU [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6dNodePrime != 0) && (here->BSIM4v6gNodeMid != 0)) {
			while (here->BSIM4v6DPgmPtr != ckt->CKTmatrix->CKTkluBind_Sparse [i]) i ++ ;
			here->BSIM4v6DPgmPtr = ckt->CKTmatrix->CKTkluBind_KLU [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6gNodePrime != 0) && (here->BSIM4v6gNodeMid != 0)) {
			while (here->BSIM4v6GPgmPtr != ckt->CKTmatrix->CKTkluBind_Sparse [i]) i ++ ;
			here->BSIM4v6GPgmPtr = ckt->CKTmatrix->CKTkluBind_KLU [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6gNodeExt != 0) && (here->BSIM4v6gNodeMid != 0)) {
			while (here->BSIM4v6GEgmPtr != ckt->CKTmatrix->CKTkluBind_Sparse [i]) i ++ ;
			here->BSIM4v6GEgmPtr = ckt->CKTmatrix->CKTkluBind_KLU [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6sNodePrime != 0) && (here->BSIM4v6gNodeMid != 0)) {
			while (here->BSIM4v6SPgmPtr != ckt->CKTmatrix->CKTkluBind_Sparse [i]) i ++ ;
			here->BSIM4v6SPgmPtr = ckt->CKTmatrix->CKTkluBind_KLU [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6bNodePrime != 0) && (here->BSIM4v6gNodeMid != 0)) {
			while (here->BSIM4v6BPgmPtr != ckt->CKTmatrix->CKTkluBind_Sparse [i]) i ++ ;
			here->BSIM4v6BPgmPtr = ckt->CKTmatrix->CKTkluBind_KLU [i] ;
		}

	    }

	    if ((here->BSIM4v6rbodyMod ==1) || (here->BSIM4v6rbodyMod ==2)) {

		i = 0 ;
		if ((here->BSIM4v6dNodePrime != 0) && (here->BSIM4v6dbNode != 0)) {
			while (here->BSIM4v6DPdbPtr != ckt->CKTmatrix->CKTkluBind_Sparse [i]) i ++ ;
			here->BSIM4v6DPdbPtr = ckt->CKTmatrix->CKTkluBind_KLU [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6sNodePrime != 0) && (here->BSIM4v6sbNode != 0)) {
			while (here->BSIM4v6SPsbPtr != ckt->CKTmatrix->CKTkluBind_Sparse [i]) i ++ ;
			here->BSIM4v6SPsbPtr = ckt->CKTmatrix->CKTkluBind_KLU [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6dbNode != 0) && (here->BSIM4v6dNodePrime != 0)) {
			while (here->BSIM4v6DBdpPtr != ckt->CKTmatrix->CKTkluBind_Sparse [i]) i ++ ;
			here->BSIM4v6DBdpPtr = ckt->CKTmatrix->CKTkluBind_KLU [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6dbNode != 0) && (here->BSIM4v6dbNode != 0)) {
			while (here->BSIM4v6DBdbPtr != ckt->CKTmatrix->CKTkluBind_Sparse [i]) i ++ ;
			here->BSIM4v6DBdbPtr = ckt->CKTmatrix->CKTkluBind_KLU [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6dbNode != 0) && (here->BSIM4v6bNodePrime != 0)) {
			while (here->BSIM4v6DBbpPtr != ckt->CKTmatrix->CKTkluBind_Sparse [i]) i ++ ;
			here->BSIM4v6DBbpPtr = ckt->CKTmatrix->CKTkluBind_KLU [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6dbNode != 0) && (here->BSIM4v6bNode != 0)) {
			while (here->BSIM4v6DBbPtr != ckt->CKTmatrix->CKTkluBind_Sparse [i]) i ++ ;
			here->BSIM4v6DBbPtr = ckt->CKTmatrix->CKTkluBind_KLU [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6bNodePrime != 0) && (here->BSIM4v6dbNode != 0)) {
			while (here->BSIM4v6BPdbPtr != ckt->CKTmatrix->CKTkluBind_Sparse [i]) i ++ ;
			here->BSIM4v6BPdbPtr = ckt->CKTmatrix->CKTkluBind_KLU [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6bNodePrime != 0) && (here->BSIM4v6bNode != 0)) {
			while (here->BSIM4v6BPbPtr != ckt->CKTmatrix->CKTkluBind_Sparse [i]) i ++ ;
			here->BSIM4v6BPbPtr = ckt->CKTmatrix->CKTkluBind_KLU [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6bNodePrime != 0) && (here->BSIM4v6sbNode != 0)) {
			while (here->BSIM4v6BPsbPtr != ckt->CKTmatrix->CKTkluBind_Sparse [i]) i ++ ;
			here->BSIM4v6BPsbPtr = ckt->CKTmatrix->CKTkluBind_KLU [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6sbNode != 0) && (here->BSIM4v6sNodePrime != 0)) {
			while (here->BSIM4v6SBspPtr != ckt->CKTmatrix->CKTkluBind_Sparse [i]) i ++ ;
			here->BSIM4v6SBspPtr = ckt->CKTmatrix->CKTkluBind_KLU [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6sbNode != 0) && (here->BSIM4v6bNodePrime != 0)) {
			while (here->BSIM4v6SBbpPtr != ckt->CKTmatrix->CKTkluBind_Sparse [i]) i ++ ;
			here->BSIM4v6SBbpPtr = ckt->CKTmatrix->CKTkluBind_KLU [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6sbNode != 0) && (here->BSIM4v6bNode != 0)) {
			while (here->BSIM4v6SBbPtr != ckt->CKTmatrix->CKTkluBind_Sparse [i]) i ++ ;
			here->BSIM4v6SBbPtr = ckt->CKTmatrix->CKTkluBind_KLU [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6sbNode != 0) && (here->BSIM4v6sbNode != 0)) {
			while (here->BSIM4v6SBsbPtr != ckt->CKTmatrix->CKTkluBind_Sparse [i]) i ++ ;
			here->BSIM4v6SBsbPtr = ckt->CKTmatrix->CKTkluBind_KLU [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6bNode != 0) && (here->BSIM4v6dbNode != 0)) {
			while (here->BSIM4v6BdbPtr != ckt->CKTmatrix->CKTkluBind_Sparse [i]) i ++ ;
			here->BSIM4v6BdbPtr = ckt->CKTmatrix->CKTkluBind_KLU [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6bNode != 0) && (here->BSIM4v6bNodePrime != 0)) {
			while (here->BSIM4v6BbpPtr != ckt->CKTmatrix->CKTkluBind_Sparse [i]) i ++ ;
			here->BSIM4v6BbpPtr = ckt->CKTmatrix->CKTkluBind_KLU [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6bNode != 0) && (here->BSIM4v6sbNode != 0)) {
			while (here->BSIM4v6BsbPtr != ckt->CKTmatrix->CKTkluBind_Sparse [i]) i ++ ;
			here->BSIM4v6BsbPtr = ckt->CKTmatrix->CKTkluBind_KLU [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6bNode != 0) && (here->BSIM4v6bNode != 0)) {
			while (here->BSIM4v6BbPtr != ckt->CKTmatrix->CKTkluBind_Sparse [i]) i ++ ;
			here->BSIM4v6BbPtr = ckt->CKTmatrix->CKTkluBind_KLU [i] ;
		}

	    }

            if (model->BSIM4v6rdsMod) {

		i = 0 ;
		if ((here->BSIM4v6dNode != 0) && (here->BSIM4v6gNodePrime != 0)) {
			while (here->BSIM4v6DgpPtr != ckt->CKTmatrix->CKTkluBind_Sparse [i]) i ++ ;
			here->BSIM4v6DgpPtr = ckt->CKTmatrix->CKTkluBind_KLU [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6dNode != 0) && (here->BSIM4v6sNodePrime != 0)) {
			while (here->BSIM4v6DspPtr != ckt->CKTmatrix->CKTkluBind_Sparse [i]) i ++ ;
			here->BSIM4v6DspPtr = ckt->CKTmatrix->CKTkluBind_KLU [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6dNode != 0) && (here->BSIM4v6bNodePrime != 0)) {
			while (here->BSIM4v6DbpPtr != ckt->CKTmatrix->CKTkluBind_Sparse [i]) i ++ ;
			here->BSIM4v6DbpPtr = ckt->CKTmatrix->CKTkluBind_KLU [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6sNode != 0) && (here->BSIM4v6dNodePrime != 0)) {
			while (here->BSIM4v6SdpPtr != ckt->CKTmatrix->CKTkluBind_Sparse [i]) i ++ ;
			here->BSIM4v6SdpPtr = ckt->CKTmatrix->CKTkluBind_KLU [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6sNode != 0) && (here->BSIM4v6gNodePrime != 0)) {
			while (here->BSIM4v6SgpPtr != ckt->CKTmatrix->CKTkluBind_Sparse [i]) i ++ ;
			here->BSIM4v6SgpPtr = ckt->CKTmatrix->CKTkluBind_KLU [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6sNode != 0) && (here->BSIM4v6bNodePrime != 0)) {
			while (here->BSIM4v6SbpPtr != ckt->CKTmatrix->CKTkluBind_Sparse [i]) i ++ ;
			here->BSIM4v6SbpPtr = ckt->CKTmatrix->CKTkluBind_KLU [i] ;
		}

	    }
	}
    }
    return(OK);
}

int
BSIM4v6bindkluComplex(GENmodel *inModel, CKTcircuit *ckt)
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
			while (here->BSIM4v6DPbpPtr != ckt->CKTmatrix->CKTkluBind_KLU [i]) i ++ ;
			here->BSIM4v6DPbpPtr = ckt->CKTmatrix->CKTkluBind_KLU_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6gNodePrime != 0) && (here->BSIM4v6bNodePrime != 0)) {
			while (here->BSIM4v6GPbpPtr != ckt->CKTmatrix->CKTkluBind_KLU [i]) i ++ ;
			here->BSIM4v6GPbpPtr = ckt->CKTmatrix->CKTkluBind_KLU_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6sNodePrime != 0) && (here->BSIM4v6bNodePrime != 0)) {
			while (here->BSIM4v6SPbpPtr != ckt->CKTmatrix->CKTkluBind_KLU [i]) i ++ ;
			here->BSIM4v6SPbpPtr = ckt->CKTmatrix->CKTkluBind_KLU_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6bNodePrime != 0) && (here->BSIM4v6dNodePrime != 0)) {
			while (here->BSIM4v6BPdpPtr != ckt->CKTmatrix->CKTkluBind_KLU [i]) i ++ ;
			here->BSIM4v6BPdpPtr = ckt->CKTmatrix->CKTkluBind_KLU_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6bNodePrime != 0) && (here->BSIM4v6gNodePrime != 0)) {
			while (here->BSIM4v6BPgpPtr != ckt->CKTmatrix->CKTkluBind_KLU [i]) i ++ ;
			here->BSIM4v6BPgpPtr = ckt->CKTmatrix->CKTkluBind_KLU_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6bNodePrime != 0) && (here->BSIM4v6sNodePrime != 0)) {
			while (here->BSIM4v6BPspPtr != ckt->CKTmatrix->CKTkluBind_KLU [i]) i ++ ;
			here->BSIM4v6BPspPtr = ckt->CKTmatrix->CKTkluBind_KLU_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6bNodePrime != 0) && (here->BSIM4v6bNodePrime != 0)) {
			while (here->BSIM4v6BPbpPtr != ckt->CKTmatrix->CKTkluBind_KLU [i]) i ++ ;
			here->BSIM4v6BPbpPtr = ckt->CKTmatrix->CKTkluBind_KLU_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6dNode != 0) && (here->BSIM4v6dNode != 0)) {
			while (here->BSIM4v6DdPtr != ckt->CKTmatrix->CKTkluBind_KLU [i]) i ++ ;
			here->BSIM4v6DdPtr = ckt->CKTmatrix->CKTkluBind_KLU_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6gNodePrime != 0) && (here->BSIM4v6gNodePrime != 0)) {
			while (here->BSIM4v6GPgpPtr != ckt->CKTmatrix->CKTkluBind_KLU [i]) i ++ ;
			here->BSIM4v6GPgpPtr = ckt->CKTmatrix->CKTkluBind_KLU_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6sNode != 0) && (here->BSIM4v6sNode != 0)) {
			while (here->BSIM4v6SsPtr != ckt->CKTmatrix->CKTkluBind_KLU [i]) i ++ ;
			here->BSIM4v6SsPtr = ckt->CKTmatrix->CKTkluBind_KLU_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6dNodePrime != 0) && (here->BSIM4v6dNodePrime != 0)) {
			while (here->BSIM4v6DPdpPtr != ckt->CKTmatrix->CKTkluBind_KLU [i]) i ++ ;
			here->BSIM4v6DPdpPtr = ckt->CKTmatrix->CKTkluBind_KLU_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6sNodePrime != 0) && (here->BSIM4v6sNodePrime != 0)) {
			while (here->BSIM4v6SPspPtr != ckt->CKTmatrix->CKTkluBind_KLU [i]) i ++ ;
			here->BSIM4v6SPspPtr = ckt->CKTmatrix->CKTkluBind_KLU_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6dNode != 0) && (here->BSIM4v6dNodePrime != 0)) {
			while (here->BSIM4v6DdpPtr != ckt->CKTmatrix->CKTkluBind_KLU [i]) i ++ ;
			here->BSIM4v6DdpPtr = ckt->CKTmatrix->CKTkluBind_KLU_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6gNodePrime != 0) && (here->BSIM4v6dNodePrime != 0)) {
			while (here->BSIM4v6GPdpPtr != ckt->CKTmatrix->CKTkluBind_KLU [i]) i ++ ;
			here->BSIM4v6GPdpPtr = ckt->CKTmatrix->CKTkluBind_KLU_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6gNodePrime != 0) && (here->BSIM4v6sNodePrime != 0)) {
			while (here->BSIM4v6GPspPtr != ckt->CKTmatrix->CKTkluBind_KLU [i]) i ++ ;
			here->BSIM4v6GPspPtr = ckt->CKTmatrix->CKTkluBind_KLU_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6sNode != 0) && (here->BSIM4v6sNodePrime != 0)) {
			while (here->BSIM4v6SspPtr != ckt->CKTmatrix->CKTkluBind_KLU [i]) i ++ ;
			here->BSIM4v6SspPtr = ckt->CKTmatrix->CKTkluBind_KLU_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6dNodePrime != 0) && (here->BSIM4v6sNodePrime != 0)) {
			while (here->BSIM4v6DPspPtr != ckt->CKTmatrix->CKTkluBind_KLU [i]) i ++ ;
			here->BSIM4v6DPspPtr = ckt->CKTmatrix->CKTkluBind_KLU_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6dNodePrime != 0) && (here->BSIM4v6dNode != 0)) {
			while (here->BSIM4v6DPdPtr != ckt->CKTmatrix->CKTkluBind_KLU [i]) i ++ ;
			here->BSIM4v6DPdPtr = ckt->CKTmatrix->CKTkluBind_KLU_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6dNodePrime != 0) && (here->BSIM4v6gNodePrime != 0)) {
			while (here->BSIM4v6DPgpPtr != ckt->CKTmatrix->CKTkluBind_KLU [i]) i ++ ;
			here->BSIM4v6DPgpPtr = ckt->CKTmatrix->CKTkluBind_KLU_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6sNodePrime != 0) && (here->BSIM4v6gNodePrime != 0)) {
			while (here->BSIM4v6SPgpPtr != ckt->CKTmatrix->CKTkluBind_KLU [i]) i ++ ;
			here->BSIM4v6SPgpPtr = ckt->CKTmatrix->CKTkluBind_KLU_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6sNodePrime != 0) && (here->BSIM4v6sNode != 0)) {
			while (here->BSIM4v6SPsPtr != ckt->CKTmatrix->CKTkluBind_KLU [i]) i ++ ;
			here->BSIM4v6SPsPtr = ckt->CKTmatrix->CKTkluBind_KLU_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6sNodePrime != 0) && (here->BSIM4v6dNodePrime != 0)) {
			while (here->BSIM4v6SPdpPtr != ckt->CKTmatrix->CKTkluBind_KLU [i]) i ++ ;
			here->BSIM4v6SPdpPtr = ckt->CKTmatrix->CKTkluBind_KLU_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6qNode != 0) && (here->BSIM4v6qNode != 0)) {
			while (here->BSIM4v6QqPtr != ckt->CKTmatrix->CKTkluBind_KLU [i]) i ++ ;
			here->BSIM4v6QqPtr = ckt->CKTmatrix->CKTkluBind_KLU_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6qNode != 0) && (here->BSIM4v6bNodePrime != 0)) {
			while (here->BSIM4v6QbpPtr != ckt->CKTmatrix->CKTkluBind_KLU [i]) i ++ ;
			here->BSIM4v6QbpPtr = ckt->CKTmatrix->CKTkluBind_KLU_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6qNode != 0) && (here->BSIM4v6dNodePrime != 0)) {
			while (here->BSIM4v6QdpPtr != ckt->CKTmatrix->CKTkluBind_KLU [i]) i ++ ;
			here->BSIM4v6QdpPtr = ckt->CKTmatrix->CKTkluBind_KLU_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6qNode != 0) && (here->BSIM4v6sNodePrime != 0)) {
			while (here->BSIM4v6QspPtr != ckt->CKTmatrix->CKTkluBind_KLU [i]) i ++ ;
			here->BSIM4v6QspPtr = ckt->CKTmatrix->CKTkluBind_KLU_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6qNode != 0) && (here->BSIM4v6gNodePrime != 0)) {
			while (here->BSIM4v6QgpPtr != ckt->CKTmatrix->CKTkluBind_KLU [i]) i ++ ;
			here->BSIM4v6QgpPtr = ckt->CKTmatrix->CKTkluBind_KLU_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6dNodePrime != 0) && (here->BSIM4v6qNode != 0)) {
			while (here->BSIM4v6DPqPtr != ckt->CKTmatrix->CKTkluBind_KLU [i]) i ++ ;
			here->BSIM4v6DPqPtr = ckt->CKTmatrix->CKTkluBind_KLU_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6sNodePrime != 0) && (here->BSIM4v6qNode != 0)) {
			while (here->BSIM4v6SPqPtr != ckt->CKTmatrix->CKTkluBind_KLU [i]) i ++ ;
			here->BSIM4v6SPqPtr = ckt->CKTmatrix->CKTkluBind_KLU_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6gNodePrime != 0) && (here->BSIM4v6qNode != 0)) {
			while (here->BSIM4v6GPqPtr != ckt->CKTmatrix->CKTkluBind_KLU [i]) i ++ ;
			here->BSIM4v6GPqPtr = ckt->CKTmatrix->CKTkluBind_KLU_Complex [i] ;
		}

	    if (here->BSIM4v6rgateMod != 0) {

		i = 0 ;
		if ((here->BSIM4v6gNodeExt != 0) && (here->BSIM4v6gNodeExt != 0)) {
			while (here->BSIM4v6GEgePtr != ckt->CKTmatrix->CKTkluBind_KLU [i]) i ++ ;
			here->BSIM4v6GEgePtr = ckt->CKTmatrix->CKTkluBind_KLU_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6gNodeExt != 0) && (here->BSIM4v6gNodePrime != 0)) {
			while (here->BSIM4v6GEgpPtr != ckt->CKTmatrix->CKTkluBind_KLU [i]) i ++ ;
			here->BSIM4v6GEgpPtr = ckt->CKTmatrix->CKTkluBind_KLU_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6gNodePrime != 0) && (here->BSIM4v6gNodeExt != 0)) {
			while (here->BSIM4v6GPgePtr != ckt->CKTmatrix->CKTkluBind_KLU [i]) i ++ ;
			here->BSIM4v6GPgePtr = ckt->CKTmatrix->CKTkluBind_KLU_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6gNodeExt != 0) && (here->BSIM4v6dNodePrime != 0)) {
			while (here->BSIM4v6GEdpPtr != ckt->CKTmatrix->CKTkluBind_KLU [i]) i ++ ;
			here->BSIM4v6GEdpPtr = ckt->CKTmatrix->CKTkluBind_KLU_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6gNodeExt != 0) && (here->BSIM4v6sNodePrime != 0)) {
			while (here->BSIM4v6GEspPtr != ckt->CKTmatrix->CKTkluBind_KLU [i]) i ++ ;
			here->BSIM4v6GEspPtr = ckt->CKTmatrix->CKTkluBind_KLU_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6gNodeExt != 0) && (here->BSIM4v6bNodePrime != 0)) {
			while (here->BSIM4v6GEbpPtr != ckt->CKTmatrix->CKTkluBind_KLU [i]) i ++ ;
			here->BSIM4v6GEbpPtr = ckt->CKTmatrix->CKTkluBind_KLU_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6gNodeMid != 0) && (here->BSIM4v6dNodePrime != 0)) {
			while (here->BSIM4v6GMdpPtr != ckt->CKTmatrix->CKTkluBind_KLU [i]) i ++ ;
			here->BSIM4v6GMdpPtr = ckt->CKTmatrix->CKTkluBind_KLU_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6gNodeMid != 0) && (here->BSIM4v6gNodePrime != 0)) {
			while (here->BSIM4v6GMgpPtr != ckt->CKTmatrix->CKTkluBind_KLU [i]) i ++ ;
			here->BSIM4v6GMgpPtr = ckt->CKTmatrix->CKTkluBind_KLU_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6gNodeMid != 0) && (here->BSIM4v6gNodeMid != 0)) {
			while (here->BSIM4v6GMgmPtr != ckt->CKTmatrix->CKTkluBind_KLU [i]) i ++ ;
			here->BSIM4v6GMgmPtr = ckt->CKTmatrix->CKTkluBind_KLU_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6gNodeMid != 0) && (here->BSIM4v6gNodeExt != 0)) {
			while (here->BSIM4v6GMgePtr != ckt->CKTmatrix->CKTkluBind_KLU [i]) i ++ ;
			here->BSIM4v6GMgePtr = ckt->CKTmatrix->CKTkluBind_KLU_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6gNodeMid != 0) && (here->BSIM4v6sNodePrime != 0)) {
			while (here->BSIM4v6GMspPtr != ckt->CKTmatrix->CKTkluBind_KLU [i]) i ++ ;
			here->BSIM4v6GMspPtr = ckt->CKTmatrix->CKTkluBind_KLU_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6gNodeMid != 0) && (here->BSIM4v6bNodePrime != 0)) {
			while (here->BSIM4v6GMbpPtr != ckt->CKTmatrix->CKTkluBind_KLU [i]) i ++ ;
			here->BSIM4v6GMbpPtr = ckt->CKTmatrix->CKTkluBind_KLU_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6dNodePrime != 0) && (here->BSIM4v6gNodeMid != 0)) {
			while (here->BSIM4v6DPgmPtr != ckt->CKTmatrix->CKTkluBind_KLU [i]) i ++ ;
			here->BSIM4v6DPgmPtr = ckt->CKTmatrix->CKTkluBind_KLU_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6gNodePrime != 0) && (here->BSIM4v6gNodeMid != 0)) {
			while (here->BSIM4v6GPgmPtr != ckt->CKTmatrix->CKTkluBind_KLU [i]) i ++ ;
			here->BSIM4v6GPgmPtr = ckt->CKTmatrix->CKTkluBind_KLU_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6gNodeExt != 0) && (here->BSIM4v6gNodeMid != 0)) {
			while (here->BSIM4v6GEgmPtr != ckt->CKTmatrix->CKTkluBind_KLU [i]) i ++ ;
			here->BSIM4v6GEgmPtr = ckt->CKTmatrix->CKTkluBind_KLU_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6sNodePrime != 0) && (here->BSIM4v6gNodeMid != 0)) {
			while (here->BSIM4v6SPgmPtr != ckt->CKTmatrix->CKTkluBind_KLU [i]) i ++ ;
			here->BSIM4v6SPgmPtr = ckt->CKTmatrix->CKTkluBind_KLU_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6bNodePrime != 0) && (here->BSIM4v6gNodeMid != 0)) {
			while (here->BSIM4v6BPgmPtr != ckt->CKTmatrix->CKTkluBind_KLU [i]) i ++ ;
			here->BSIM4v6BPgmPtr = ckt->CKTmatrix->CKTkluBind_KLU_Complex [i] ;
		}

	    }

	    if ((here->BSIM4v6rbodyMod ==1) || (here->BSIM4v6rbodyMod ==2)) {

		i = 0 ;
		if ((here->BSIM4v6dNodePrime != 0) && (here->BSIM4v6dbNode != 0)) {
			while (here->BSIM4v6DPdbPtr != ckt->CKTmatrix->CKTkluBind_KLU [i]) i ++ ;
			here->BSIM4v6DPdbPtr = ckt->CKTmatrix->CKTkluBind_KLU_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6sNodePrime != 0) && (here->BSIM4v6sbNode != 0)) {
			while (here->BSIM4v6SPsbPtr != ckt->CKTmatrix->CKTkluBind_KLU [i]) i ++ ;
			here->BSIM4v6SPsbPtr = ckt->CKTmatrix->CKTkluBind_KLU_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6dbNode != 0) && (here->BSIM4v6dNodePrime != 0)) {
			while (here->BSIM4v6DBdpPtr != ckt->CKTmatrix->CKTkluBind_KLU [i]) i ++ ;
			here->BSIM4v6DBdpPtr = ckt->CKTmatrix->CKTkluBind_KLU_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6dbNode != 0) && (here->BSIM4v6dbNode != 0)) {
			while (here->BSIM4v6DBdbPtr != ckt->CKTmatrix->CKTkluBind_KLU [i]) i ++ ;
			here->BSIM4v6DBdbPtr = ckt->CKTmatrix->CKTkluBind_KLU_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6dbNode != 0) && (here->BSIM4v6bNodePrime != 0)) {
			while (here->BSIM4v6DBbpPtr != ckt->CKTmatrix->CKTkluBind_KLU [i]) i ++ ;
			here->BSIM4v6DBbpPtr = ckt->CKTmatrix->CKTkluBind_KLU_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6dbNode != 0) && (here->BSIM4v6bNode != 0)) {
			while (here->BSIM4v6DBbPtr != ckt->CKTmatrix->CKTkluBind_KLU [i]) i ++ ;
			here->BSIM4v6DBbPtr = ckt->CKTmatrix->CKTkluBind_KLU_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6bNodePrime != 0) && (here->BSIM4v6dbNode != 0)) {
			while (here->BSIM4v6BPdbPtr != ckt->CKTmatrix->CKTkluBind_KLU [i]) i ++ ;
			here->BSIM4v6BPdbPtr = ckt->CKTmatrix->CKTkluBind_KLU_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6bNodePrime != 0) && (here->BSIM4v6bNode != 0)) {
			while (here->BSIM4v6BPbPtr != ckt->CKTmatrix->CKTkluBind_KLU [i]) i ++ ;
			here->BSIM4v6BPbPtr = ckt->CKTmatrix->CKTkluBind_KLU_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6bNodePrime != 0) && (here->BSIM4v6sbNode != 0)) {
			while (here->BSIM4v6BPsbPtr != ckt->CKTmatrix->CKTkluBind_KLU [i]) i ++ ;
			here->BSIM4v6BPsbPtr = ckt->CKTmatrix->CKTkluBind_KLU_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6sbNode != 0) && (here->BSIM4v6sNodePrime != 0)) {
			while (here->BSIM4v6SBspPtr != ckt->CKTmatrix->CKTkluBind_KLU [i]) i ++ ;
			here->BSIM4v6SBspPtr = ckt->CKTmatrix->CKTkluBind_KLU_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6sbNode != 0) && (here->BSIM4v6bNodePrime != 0)) {
			while (here->BSIM4v6SBbpPtr != ckt->CKTmatrix->CKTkluBind_KLU [i]) i ++ ;
			here->BSIM4v6SBbpPtr = ckt->CKTmatrix->CKTkluBind_KLU_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6sbNode != 0) && (here->BSIM4v6bNode != 0)) {
			while (here->BSIM4v6SBbPtr != ckt->CKTmatrix->CKTkluBind_KLU [i]) i ++ ;
			here->BSIM4v6SBbPtr = ckt->CKTmatrix->CKTkluBind_KLU_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6sbNode != 0) && (here->BSIM4v6sbNode != 0)) {
			while (here->BSIM4v6SBsbPtr != ckt->CKTmatrix->CKTkluBind_KLU [i]) i ++ ;
			here->BSIM4v6SBsbPtr = ckt->CKTmatrix->CKTkluBind_KLU_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6bNode != 0) && (here->BSIM4v6dbNode != 0)) {
			while (here->BSIM4v6BdbPtr != ckt->CKTmatrix->CKTkluBind_KLU [i]) i ++ ;
			here->BSIM4v6BdbPtr = ckt->CKTmatrix->CKTkluBind_KLU_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6bNode != 0) && (here->BSIM4v6bNodePrime != 0)) {
			while (here->BSIM4v6BbpPtr != ckt->CKTmatrix->CKTkluBind_KLU [i]) i ++ ;
			here->BSIM4v6BbpPtr = ckt->CKTmatrix->CKTkluBind_KLU_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6bNode != 0) && (here->BSIM4v6sbNode != 0)) {
			while (here->BSIM4v6BsbPtr != ckt->CKTmatrix->CKTkluBind_KLU [i]) i ++ ;
			here->BSIM4v6BsbPtr = ckt->CKTmatrix->CKTkluBind_KLU_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6bNode != 0) && (here->BSIM4v6bNode != 0)) {
			while (here->BSIM4v6BbPtr != ckt->CKTmatrix->CKTkluBind_KLU [i]) i ++ ;
			here->BSIM4v6BbPtr = ckt->CKTmatrix->CKTkluBind_KLU_Complex [i] ;
		}

	    }

            if (model->BSIM4v6rdsMod) {

		i = 0 ;
		if ((here->BSIM4v6dNode != 0) && (here->BSIM4v6gNodePrime != 0)) {
			while (here->BSIM4v6DgpPtr != ckt->CKTmatrix->CKTkluBind_KLU [i]) i ++ ;
			here->BSIM4v6DgpPtr = ckt->CKTmatrix->CKTkluBind_KLU_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6dNode != 0) && (here->BSIM4v6sNodePrime != 0)) {
			while (here->BSIM4v6DspPtr != ckt->CKTmatrix->CKTkluBind_KLU [i]) i ++ ;
			here->BSIM4v6DspPtr = ckt->CKTmatrix->CKTkluBind_KLU_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6dNode != 0) && (here->BSIM4v6bNodePrime != 0)) {
			while (here->BSIM4v6DbpPtr != ckt->CKTmatrix->CKTkluBind_KLU [i]) i ++ ;
			here->BSIM4v6DbpPtr = ckt->CKTmatrix->CKTkluBind_KLU_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6sNode != 0) && (here->BSIM4v6dNodePrime != 0)) {
			while (here->BSIM4v6SdpPtr != ckt->CKTmatrix->CKTkluBind_KLU [i]) i ++ ;
			here->BSIM4v6SdpPtr = ckt->CKTmatrix->CKTkluBind_KLU_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6sNode != 0) && (here->BSIM4v6gNodePrime != 0)) {
			while (here->BSIM4v6SgpPtr != ckt->CKTmatrix->CKTkluBind_KLU [i]) i ++ ;
			here->BSIM4v6SgpPtr = ckt->CKTmatrix->CKTkluBind_KLU_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6sNode != 0) && (here->BSIM4v6bNodePrime != 0)) {
			while (here->BSIM4v6SbpPtr != ckt->CKTmatrix->CKTkluBind_KLU [i]) i ++ ;
			here->BSIM4v6SbpPtr = ckt->CKTmatrix->CKTkluBind_KLU_Complex [i] ;
		}

	    }
	}
    }
    return(OK);
}
