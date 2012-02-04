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
			while (here->BSIM4v6DPbpPtr != ckt->CKTkluBind_Sparse [i]) i ++ ;
			here->BSIM4v6DPbpPtr = ckt->CKTkluBind_KLU [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6gNodePrime != 0) && (here->BSIM4v6bNodePrime != 0)) {
			while (here->BSIM4v6GPbpPtr != ckt->CKTkluBind_Sparse [i]) i ++ ;
			here->BSIM4v6GPbpPtr = ckt->CKTkluBind_KLU [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6sNodePrime != 0) && (here->BSIM4v6bNodePrime != 0)) {
			while (here->BSIM4v6SPbpPtr != ckt->CKTkluBind_Sparse [i]) i ++ ;
			here->BSIM4v6SPbpPtr = ckt->CKTkluBind_KLU [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6bNodePrime != 0) && (here->BSIM4v6dNodePrime != 0)) {
			while (here->BSIM4v6BPdpPtr != ckt->CKTkluBind_Sparse [i]) i ++ ;
			here->BSIM4v6BPdpPtr = ckt->CKTkluBind_KLU [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6bNodePrime != 0) && (here->BSIM4v6gNodePrime != 0)) {
			while (here->BSIM4v6BPgpPtr != ckt->CKTkluBind_Sparse [i]) i ++ ;
			here->BSIM4v6BPgpPtr = ckt->CKTkluBind_KLU [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6bNodePrime != 0) && (here->BSIM4v6sNodePrime != 0)) {
			while (here->BSIM4v6BPspPtr != ckt->CKTkluBind_Sparse [i]) i ++ ;
			here->BSIM4v6BPspPtr = ckt->CKTkluBind_KLU [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6bNodePrime != 0) && (here->BSIM4v6bNodePrime != 0)) {
			while (here->BSIM4v6BPbpPtr != ckt->CKTkluBind_Sparse [i]) i ++ ;
			here->BSIM4v6BPbpPtr = ckt->CKTkluBind_KLU [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6dNode != 0) && (here->BSIM4v6dNode != 0)) {
			while (here->BSIM4v6DdPtr != ckt->CKTkluBind_Sparse [i]) i ++ ;
			here->BSIM4v6DdPtr = ckt->CKTkluBind_KLU [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6gNodePrime != 0) && (here->BSIM4v6gNodePrime != 0)) {
			while (here->BSIM4v6GPgpPtr != ckt->CKTkluBind_Sparse [i]) i ++ ;
			here->BSIM4v6GPgpPtr = ckt->CKTkluBind_KLU [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6sNode != 0) && (here->BSIM4v6sNode != 0)) {
			while (here->BSIM4v6SsPtr != ckt->CKTkluBind_Sparse [i]) i ++ ;
			here->BSIM4v6SsPtr = ckt->CKTkluBind_KLU [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6dNodePrime != 0) && (here->BSIM4v6dNodePrime != 0)) {
			while (here->BSIM4v6DPdpPtr != ckt->CKTkluBind_Sparse [i]) i ++ ;
			here->BSIM4v6DPdpPtr = ckt->CKTkluBind_KLU [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6sNodePrime != 0) && (here->BSIM4v6sNodePrime != 0)) {
			while (here->BSIM4v6SPspPtr != ckt->CKTkluBind_Sparse [i]) i ++ ;
			here->BSIM4v6SPspPtr = ckt->CKTkluBind_KLU [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6dNode != 0) && (here->BSIM4v6dNodePrime != 0)) {
			while (here->BSIM4v6DdpPtr != ckt->CKTkluBind_Sparse [i]) i ++ ;
			here->BSIM4v6DdpPtr = ckt->CKTkluBind_KLU [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6gNodePrime != 0) && (here->BSIM4v6dNodePrime != 0)) {
			while (here->BSIM4v6GPdpPtr != ckt->CKTkluBind_Sparse [i]) i ++ ;
			here->BSIM4v6GPdpPtr = ckt->CKTkluBind_KLU [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6gNodePrime != 0) && (here->BSIM4v6sNodePrime != 0)) {
			while (here->BSIM4v6GPspPtr != ckt->CKTkluBind_Sparse [i]) i ++ ;
			here->BSIM4v6GPspPtr = ckt->CKTkluBind_KLU [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6sNode != 0) && (here->BSIM4v6sNodePrime != 0)) {
			while (here->BSIM4v6SspPtr != ckt->CKTkluBind_Sparse [i]) i ++ ;
			here->BSIM4v6SspPtr = ckt->CKTkluBind_KLU [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6dNodePrime != 0) && (here->BSIM4v6sNodePrime != 0)) {
			while (here->BSIM4v6DPspPtr != ckt->CKTkluBind_Sparse [i]) i ++ ;
			here->BSIM4v6DPspPtr = ckt->CKTkluBind_KLU [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6dNodePrime != 0) && (here->BSIM4v6dNode != 0)) {
			while (here->BSIM4v6DPdPtr != ckt->CKTkluBind_Sparse [i]) i ++ ;
			here->BSIM4v6DPdPtr = ckt->CKTkluBind_KLU [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6dNodePrime != 0) && (here->BSIM4v6gNodePrime != 0)) {
			while (here->BSIM4v6DPgpPtr != ckt->CKTkluBind_Sparse [i]) i ++ ;
			here->BSIM4v6DPgpPtr = ckt->CKTkluBind_KLU [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6sNodePrime != 0) && (here->BSIM4v6gNodePrime != 0)) {
			while (here->BSIM4v6SPgpPtr != ckt->CKTkluBind_Sparse [i]) i ++ ;
			here->BSIM4v6SPgpPtr = ckt->CKTkluBind_KLU [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6sNodePrime != 0) && (here->BSIM4v6sNode != 0)) {
			while (here->BSIM4v6SPsPtr != ckt->CKTkluBind_Sparse [i]) i ++ ;
			here->BSIM4v6SPsPtr = ckt->CKTkluBind_KLU [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6sNodePrime != 0) && (here->BSIM4v6dNodePrime != 0)) {
			while (here->BSIM4v6SPdpPtr != ckt->CKTkluBind_Sparse [i]) i ++ ;
			here->BSIM4v6SPdpPtr = ckt->CKTkluBind_KLU [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6qNode != 0) && (here->BSIM4v6qNode != 0)) {
			while (here->BSIM4v6QqPtr != ckt->CKTkluBind_Sparse [i]) i ++ ;
			here->BSIM4v6QqPtr = ckt->CKTkluBind_KLU [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6qNode != 0) && (here->BSIM4v6bNodePrime != 0)) {
			while (here->BSIM4v6QbpPtr != ckt->CKTkluBind_Sparse [i]) i ++ ;
			here->BSIM4v6QbpPtr = ckt->CKTkluBind_KLU [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6qNode != 0) && (here->BSIM4v6dNodePrime != 0)) {
			while (here->BSIM4v6QdpPtr != ckt->CKTkluBind_Sparse [i]) i ++ ;
			here->BSIM4v6QdpPtr = ckt->CKTkluBind_KLU [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6qNode != 0) && (here->BSIM4v6sNodePrime != 0)) {
			while (here->BSIM4v6QspPtr != ckt->CKTkluBind_Sparse [i]) i ++ ;
			here->BSIM4v6QspPtr = ckt->CKTkluBind_KLU [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6qNode != 0) && (here->BSIM4v6gNodePrime != 0)) {
			while (here->BSIM4v6QgpPtr != ckt->CKTkluBind_Sparse [i]) i ++ ;
			here->BSIM4v6QgpPtr = ckt->CKTkluBind_KLU [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6dNodePrime != 0) && (here->BSIM4v6qNode != 0)) {
			while (here->BSIM4v6DPqPtr != ckt->CKTkluBind_Sparse [i]) i ++ ;
			here->BSIM4v6DPqPtr = ckt->CKTkluBind_KLU [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6sNodePrime != 0) && (here->BSIM4v6qNode != 0)) {
			while (here->BSIM4v6SPqPtr != ckt->CKTkluBind_Sparse [i]) i ++ ;
			here->BSIM4v6SPqPtr = ckt->CKTkluBind_KLU [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6gNodePrime != 0) && (here->BSIM4v6qNode != 0)) {
			while (here->BSIM4v6GPqPtr != ckt->CKTkluBind_Sparse [i]) i ++ ;
			here->BSIM4v6GPqPtr = ckt->CKTkluBind_KLU [i] ;
		}

	    if (here->BSIM4v6rgateMod != 0) {

		i = 0 ;
		if ((here->BSIM4v6gNodeExt != 0) && (here->BSIM4v6gNodeExt != 0)) {
			while (here->BSIM4v6GEgePtr != ckt->CKTkluBind_Sparse [i]) i ++ ;
			here->BSIM4v6GEgePtr = ckt->CKTkluBind_KLU [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6gNodeExt != 0) && (here->BSIM4v6gNodePrime != 0)) {
			while (here->BSIM4v6GEgpPtr != ckt->CKTkluBind_Sparse [i]) i ++ ;
			here->BSIM4v6GEgpPtr = ckt->CKTkluBind_KLU [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6gNodePrime != 0) && (here->BSIM4v6gNodeExt != 0)) {
			while (here->BSIM4v6GPgePtr != ckt->CKTkluBind_Sparse [i]) i ++ ;
			here->BSIM4v6GPgePtr = ckt->CKTkluBind_KLU [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6gNodeExt != 0) && (here->BSIM4v6dNodePrime != 0)) {
			while (here->BSIM4v6GEdpPtr != ckt->CKTkluBind_Sparse [i]) i ++ ;
			here->BSIM4v6GEdpPtr = ckt->CKTkluBind_KLU [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6gNodeExt != 0) && (here->BSIM4v6sNodePrime != 0)) {
			while (here->BSIM4v6GEspPtr != ckt->CKTkluBind_Sparse [i]) i ++ ;
			here->BSIM4v6GEspPtr = ckt->CKTkluBind_KLU [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6gNodeExt != 0) && (here->BSIM4v6bNodePrime != 0)) {
			while (here->BSIM4v6GEbpPtr != ckt->CKTkluBind_Sparse [i]) i ++ ;
			here->BSIM4v6GEbpPtr = ckt->CKTkluBind_KLU [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6gNodeMid != 0) && (here->BSIM4v6dNodePrime != 0)) {
			while (here->BSIM4v6GMdpPtr != ckt->CKTkluBind_Sparse [i]) i ++ ;
			here->BSIM4v6GMdpPtr = ckt->CKTkluBind_KLU [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6gNodeMid != 0) && (here->BSIM4v6gNodePrime != 0)) {
			while (here->BSIM4v6GMgpPtr != ckt->CKTkluBind_Sparse [i]) i ++ ;
			here->BSIM4v6GMgpPtr = ckt->CKTkluBind_KLU [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6gNodeMid != 0) && (here->BSIM4v6gNodeMid != 0)) {
			while (here->BSIM4v6GMgmPtr != ckt->CKTkluBind_Sparse [i]) i ++ ;
			here->BSIM4v6GMgmPtr = ckt->CKTkluBind_KLU [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6gNodeMid != 0) && (here->BSIM4v6gNodeExt != 0)) {
			while (here->BSIM4v6GMgePtr != ckt->CKTkluBind_Sparse [i]) i ++ ;
			here->BSIM4v6GMgePtr = ckt->CKTkluBind_KLU [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6gNodeMid != 0) && (here->BSIM4v6sNodePrime != 0)) {
			while (here->BSIM4v6GMspPtr != ckt->CKTkluBind_Sparse [i]) i ++ ;
			here->BSIM4v6GMspPtr = ckt->CKTkluBind_KLU [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6gNodeMid != 0) && (here->BSIM4v6bNodePrime != 0)) {
			while (here->BSIM4v6GMbpPtr != ckt->CKTkluBind_Sparse [i]) i ++ ;
			here->BSIM4v6GMbpPtr = ckt->CKTkluBind_KLU [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6dNodePrime != 0) && (here->BSIM4v6gNodeMid != 0)) {
			while (here->BSIM4v6DPgmPtr != ckt->CKTkluBind_Sparse [i]) i ++ ;
			here->BSIM4v6DPgmPtr = ckt->CKTkluBind_KLU [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6gNodePrime != 0) && (here->BSIM4v6gNodeMid != 0)) {
			while (here->BSIM4v6GPgmPtr != ckt->CKTkluBind_Sparse [i]) i ++ ;
			here->BSIM4v6GPgmPtr = ckt->CKTkluBind_KLU [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6gNodeExt != 0) && (here->BSIM4v6gNodeMid != 0)) {
			while (here->BSIM4v6GEgmPtr != ckt->CKTkluBind_Sparse [i]) i ++ ;
			here->BSIM4v6GEgmPtr = ckt->CKTkluBind_KLU [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6sNodePrime != 0) && (here->BSIM4v6gNodeMid != 0)) {
			while (here->BSIM4v6SPgmPtr != ckt->CKTkluBind_Sparse [i]) i ++ ;
			here->BSIM4v6SPgmPtr = ckt->CKTkluBind_KLU [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6bNodePrime != 0) && (here->BSIM4v6gNodeMid != 0)) {
			while (here->BSIM4v6BPgmPtr != ckt->CKTkluBind_Sparse [i]) i ++ ;
			here->BSIM4v6BPgmPtr = ckt->CKTkluBind_KLU [i] ;
		}

	    }

	    if ((here->BSIM4v6rbodyMod ==1) || (here->BSIM4v6rbodyMod ==2)) {

		i = 0 ;
		if ((here->BSIM4v6dNodePrime != 0) && (here->BSIM4v6dbNode != 0)) {
			while (here->BSIM4v6DPdbPtr != ckt->CKTkluBind_Sparse [i]) i ++ ;
			here->BSIM4v6DPdbPtr = ckt->CKTkluBind_KLU [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6sNodePrime != 0) && (here->BSIM4v6sbNode != 0)) {
			while (here->BSIM4v6SPsbPtr != ckt->CKTkluBind_Sparse [i]) i ++ ;
			here->BSIM4v6SPsbPtr = ckt->CKTkluBind_KLU [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6dbNode != 0) && (here->BSIM4v6dNodePrime != 0)) {
			while (here->BSIM4v6DBdpPtr != ckt->CKTkluBind_Sparse [i]) i ++ ;
			here->BSIM4v6DBdpPtr = ckt->CKTkluBind_KLU [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6dbNode != 0) && (here->BSIM4v6dbNode != 0)) {
			while (here->BSIM4v6DBdbPtr != ckt->CKTkluBind_Sparse [i]) i ++ ;
			here->BSIM4v6DBdbPtr = ckt->CKTkluBind_KLU [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6dbNode != 0) && (here->BSIM4v6bNodePrime != 0)) {
			while (here->BSIM4v6DBbpPtr != ckt->CKTkluBind_Sparse [i]) i ++ ;
			here->BSIM4v6DBbpPtr = ckt->CKTkluBind_KLU [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6dbNode != 0) && (here->BSIM4v6bNode != 0)) {
			while (here->BSIM4v6DBbPtr != ckt->CKTkluBind_Sparse [i]) i ++ ;
			here->BSIM4v6DBbPtr = ckt->CKTkluBind_KLU [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6bNodePrime != 0) && (here->BSIM4v6dbNode != 0)) {
			while (here->BSIM4v6BPdbPtr != ckt->CKTkluBind_Sparse [i]) i ++ ;
			here->BSIM4v6BPdbPtr = ckt->CKTkluBind_KLU [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6bNodePrime != 0) && (here->BSIM4v6bNode != 0)) {
			while (here->BSIM4v6BPbPtr != ckt->CKTkluBind_Sparse [i]) i ++ ;
			here->BSIM4v6BPbPtr = ckt->CKTkluBind_KLU [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6bNodePrime != 0) && (here->BSIM4v6sbNode != 0)) {
			while (here->BSIM4v6BPsbPtr != ckt->CKTkluBind_Sparse [i]) i ++ ;
			here->BSIM4v6BPsbPtr = ckt->CKTkluBind_KLU [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6sbNode != 0) && (here->BSIM4v6sNodePrime != 0)) {
			while (here->BSIM4v6SBspPtr != ckt->CKTkluBind_Sparse [i]) i ++ ;
			here->BSIM4v6SBspPtr = ckt->CKTkluBind_KLU [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6sbNode != 0) && (here->BSIM4v6bNodePrime != 0)) {
			while (here->BSIM4v6SBbpPtr != ckt->CKTkluBind_Sparse [i]) i ++ ;
			here->BSIM4v6SBbpPtr = ckt->CKTkluBind_KLU [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6sbNode != 0) && (here->BSIM4v6bNode != 0)) {
			while (here->BSIM4v6SBbPtr != ckt->CKTkluBind_Sparse [i]) i ++ ;
			here->BSIM4v6SBbPtr = ckt->CKTkluBind_KLU [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6sbNode != 0) && (here->BSIM4v6sbNode != 0)) {
			while (here->BSIM4v6SBsbPtr != ckt->CKTkluBind_Sparse [i]) i ++ ;
			here->BSIM4v6SBsbPtr = ckt->CKTkluBind_KLU [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6bNode != 0) && (here->BSIM4v6dbNode != 0)) {
			while (here->BSIM4v6BdbPtr != ckt->CKTkluBind_Sparse [i]) i ++ ;
			here->BSIM4v6BdbPtr = ckt->CKTkluBind_KLU [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6bNode != 0) && (here->BSIM4v6bNodePrime != 0)) {
			while (here->BSIM4v6BbpPtr != ckt->CKTkluBind_Sparse [i]) i ++ ;
			here->BSIM4v6BbpPtr = ckt->CKTkluBind_KLU [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6bNode != 0) && (here->BSIM4v6sbNode != 0)) {
			while (here->BSIM4v6BsbPtr != ckt->CKTkluBind_Sparse [i]) i ++ ;
			here->BSIM4v6BsbPtr = ckt->CKTkluBind_KLU [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6bNode != 0) && (here->BSIM4v6bNode != 0)) {
			while (here->BSIM4v6BbPtr != ckt->CKTkluBind_Sparse [i]) i ++ ;
			here->BSIM4v6BbPtr = ckt->CKTkluBind_KLU [i] ;
		}

	    }

            if (model->BSIM4v6rdsMod) {

		i = 0 ;
		if ((here->BSIM4v6dNode != 0) && (here->BSIM4v6gNodePrime != 0)) {
			while (here->BSIM4v6DgpPtr != ckt->CKTkluBind_Sparse [i]) i ++ ;
			here->BSIM4v6DgpPtr = ckt->CKTkluBind_KLU [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6dNode != 0) && (here->BSIM4v6sNodePrime != 0)) {
			while (here->BSIM4v6DspPtr != ckt->CKTkluBind_Sparse [i]) i ++ ;
			here->BSIM4v6DspPtr = ckt->CKTkluBind_KLU [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6dNode != 0) && (here->BSIM4v6bNodePrime != 0)) {
			while (here->BSIM4v6DbpPtr != ckt->CKTkluBind_Sparse [i]) i ++ ;
			here->BSIM4v6DbpPtr = ckt->CKTkluBind_KLU [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6sNode != 0) && (here->BSIM4v6dNodePrime != 0)) {
			while (here->BSIM4v6SdpPtr != ckt->CKTkluBind_Sparse [i]) i ++ ;
			here->BSIM4v6SdpPtr = ckt->CKTkluBind_KLU [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6sNode != 0) && (here->BSIM4v6gNodePrime != 0)) {
			while (here->BSIM4v6SgpPtr != ckt->CKTkluBind_Sparse [i]) i ++ ;
			here->BSIM4v6SgpPtr = ckt->CKTkluBind_KLU [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6sNode != 0) && (here->BSIM4v6bNodePrime != 0)) {
			while (here->BSIM4v6SbpPtr != ckt->CKTkluBind_Sparse [i]) i ++ ;
			here->BSIM4v6SbpPtr = ckt->CKTkluBind_KLU [i] ;
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
			while (here->BSIM4v6DPbpPtr != ckt->CKTkluBind_KLU [i]) i ++ ;
			here->BSIM4v6DPbpPtr = ckt->CKTkluBind_KLU_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6gNodePrime != 0) && (here->BSIM4v6bNodePrime != 0)) {
			while (here->BSIM4v6GPbpPtr != ckt->CKTkluBind_KLU [i]) i ++ ;
			here->BSIM4v6GPbpPtr = ckt->CKTkluBind_KLU_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6sNodePrime != 0) && (here->BSIM4v6bNodePrime != 0)) {
			while (here->BSIM4v6SPbpPtr != ckt->CKTkluBind_KLU [i]) i ++ ;
			here->BSIM4v6SPbpPtr = ckt->CKTkluBind_KLU_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6bNodePrime != 0) && (here->BSIM4v6dNodePrime != 0)) {
			while (here->BSIM4v6BPdpPtr != ckt->CKTkluBind_KLU [i]) i ++ ;
			here->BSIM4v6BPdpPtr = ckt->CKTkluBind_KLU_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6bNodePrime != 0) && (here->BSIM4v6gNodePrime != 0)) {
			while (here->BSIM4v6BPgpPtr != ckt->CKTkluBind_KLU [i]) i ++ ;
			here->BSIM4v6BPgpPtr = ckt->CKTkluBind_KLU_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6bNodePrime != 0) && (here->BSIM4v6sNodePrime != 0)) {
			while (here->BSIM4v6BPspPtr != ckt->CKTkluBind_KLU [i]) i ++ ;
			here->BSIM4v6BPspPtr = ckt->CKTkluBind_KLU_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6bNodePrime != 0) && (here->BSIM4v6bNodePrime != 0)) {
			while (here->BSIM4v6BPbpPtr != ckt->CKTkluBind_KLU [i]) i ++ ;
			here->BSIM4v6BPbpPtr = ckt->CKTkluBind_KLU_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6dNode != 0) && (here->BSIM4v6dNode != 0)) {
			while (here->BSIM4v6DdPtr != ckt->CKTkluBind_KLU [i]) i ++ ;
			here->BSIM4v6DdPtr = ckt->CKTkluBind_KLU_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6gNodePrime != 0) && (here->BSIM4v6gNodePrime != 0)) {
			while (here->BSIM4v6GPgpPtr != ckt->CKTkluBind_KLU [i]) i ++ ;
			here->BSIM4v6GPgpPtr = ckt->CKTkluBind_KLU_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6sNode != 0) && (here->BSIM4v6sNode != 0)) {
			while (here->BSIM4v6SsPtr != ckt->CKTkluBind_KLU [i]) i ++ ;
			here->BSIM4v6SsPtr = ckt->CKTkluBind_KLU_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6dNodePrime != 0) && (here->BSIM4v6dNodePrime != 0)) {
			while (here->BSIM4v6DPdpPtr != ckt->CKTkluBind_KLU [i]) i ++ ;
			here->BSIM4v6DPdpPtr = ckt->CKTkluBind_KLU_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6sNodePrime != 0) && (here->BSIM4v6sNodePrime != 0)) {
			while (here->BSIM4v6SPspPtr != ckt->CKTkluBind_KLU [i]) i ++ ;
			here->BSIM4v6SPspPtr = ckt->CKTkluBind_KLU_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6dNode != 0) && (here->BSIM4v6dNodePrime != 0)) {
			while (here->BSIM4v6DdpPtr != ckt->CKTkluBind_KLU [i]) i ++ ;
			here->BSIM4v6DdpPtr = ckt->CKTkluBind_KLU_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6gNodePrime != 0) && (here->BSIM4v6dNodePrime != 0)) {
			while (here->BSIM4v6GPdpPtr != ckt->CKTkluBind_KLU [i]) i ++ ;
			here->BSIM4v6GPdpPtr = ckt->CKTkluBind_KLU_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6gNodePrime != 0) && (here->BSIM4v6sNodePrime != 0)) {
			while (here->BSIM4v6GPspPtr != ckt->CKTkluBind_KLU [i]) i ++ ;
			here->BSIM4v6GPspPtr = ckt->CKTkluBind_KLU_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6sNode != 0) && (here->BSIM4v6sNodePrime != 0)) {
			while (here->BSIM4v6SspPtr != ckt->CKTkluBind_KLU [i]) i ++ ;
			here->BSIM4v6SspPtr = ckt->CKTkluBind_KLU_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6dNodePrime != 0) && (here->BSIM4v6sNodePrime != 0)) {
			while (here->BSIM4v6DPspPtr != ckt->CKTkluBind_KLU [i]) i ++ ;
			here->BSIM4v6DPspPtr = ckt->CKTkluBind_KLU_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6dNodePrime != 0) && (here->BSIM4v6dNode != 0)) {
			while (here->BSIM4v6DPdPtr != ckt->CKTkluBind_KLU [i]) i ++ ;
			here->BSIM4v6DPdPtr = ckt->CKTkluBind_KLU_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6dNodePrime != 0) && (here->BSIM4v6gNodePrime != 0)) {
			while (here->BSIM4v6DPgpPtr != ckt->CKTkluBind_KLU [i]) i ++ ;
			here->BSIM4v6DPgpPtr = ckt->CKTkluBind_KLU_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6sNodePrime != 0) && (here->BSIM4v6gNodePrime != 0)) {
			while (here->BSIM4v6SPgpPtr != ckt->CKTkluBind_KLU [i]) i ++ ;
			here->BSIM4v6SPgpPtr = ckt->CKTkluBind_KLU_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6sNodePrime != 0) && (here->BSIM4v6sNode != 0)) {
			while (here->BSIM4v6SPsPtr != ckt->CKTkluBind_KLU [i]) i ++ ;
			here->BSIM4v6SPsPtr = ckt->CKTkluBind_KLU_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6sNodePrime != 0) && (here->BSIM4v6dNodePrime != 0)) {
			while (here->BSIM4v6SPdpPtr != ckt->CKTkluBind_KLU [i]) i ++ ;
			here->BSIM4v6SPdpPtr = ckt->CKTkluBind_KLU_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6qNode != 0) && (here->BSIM4v6qNode != 0)) {
			while (here->BSIM4v6QqPtr != ckt->CKTkluBind_KLU [i]) i ++ ;
			here->BSIM4v6QqPtr = ckt->CKTkluBind_KLU_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6qNode != 0) && (here->BSIM4v6bNodePrime != 0)) {
			while (here->BSIM4v6QbpPtr != ckt->CKTkluBind_KLU [i]) i ++ ;
			here->BSIM4v6QbpPtr = ckt->CKTkluBind_KLU_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6qNode != 0) && (here->BSIM4v6dNodePrime != 0)) {
			while (here->BSIM4v6QdpPtr != ckt->CKTkluBind_KLU [i]) i ++ ;
			here->BSIM4v6QdpPtr = ckt->CKTkluBind_KLU_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6qNode != 0) && (here->BSIM4v6sNodePrime != 0)) {
			while (here->BSIM4v6QspPtr != ckt->CKTkluBind_KLU [i]) i ++ ;
			here->BSIM4v6QspPtr = ckt->CKTkluBind_KLU_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6qNode != 0) && (here->BSIM4v6gNodePrime != 0)) {
			while (here->BSIM4v6QgpPtr != ckt->CKTkluBind_KLU [i]) i ++ ;
			here->BSIM4v6QgpPtr = ckt->CKTkluBind_KLU_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6dNodePrime != 0) && (here->BSIM4v6qNode != 0)) {
			while (here->BSIM4v6DPqPtr != ckt->CKTkluBind_KLU [i]) i ++ ;
			here->BSIM4v6DPqPtr = ckt->CKTkluBind_KLU_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6sNodePrime != 0) && (here->BSIM4v6qNode != 0)) {
			while (here->BSIM4v6SPqPtr != ckt->CKTkluBind_KLU [i]) i ++ ;
			here->BSIM4v6SPqPtr = ckt->CKTkluBind_KLU_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6gNodePrime != 0) && (here->BSIM4v6qNode != 0)) {
			while (here->BSIM4v6GPqPtr != ckt->CKTkluBind_KLU [i]) i ++ ;
			here->BSIM4v6GPqPtr = ckt->CKTkluBind_KLU_Complex [i] ;
		}

	    if (here->BSIM4v6rgateMod != 0) {

		i = 0 ;
		if ((here->BSIM4v6gNodeExt != 0) && (here->BSIM4v6gNodeExt != 0)) {
			while (here->BSIM4v6GEgePtr != ckt->CKTkluBind_KLU [i]) i ++ ;
			here->BSIM4v6GEgePtr = ckt->CKTkluBind_KLU_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6gNodeExt != 0) && (here->BSIM4v6gNodePrime != 0)) {
			while (here->BSIM4v6GEgpPtr != ckt->CKTkluBind_KLU [i]) i ++ ;
			here->BSIM4v6GEgpPtr = ckt->CKTkluBind_KLU_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6gNodePrime != 0) && (here->BSIM4v6gNodeExt != 0)) {
			while (here->BSIM4v6GPgePtr != ckt->CKTkluBind_KLU [i]) i ++ ;
			here->BSIM4v6GPgePtr = ckt->CKTkluBind_KLU_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6gNodeExt != 0) && (here->BSIM4v6dNodePrime != 0)) {
			while (here->BSIM4v6GEdpPtr != ckt->CKTkluBind_KLU [i]) i ++ ;
			here->BSIM4v6GEdpPtr = ckt->CKTkluBind_KLU_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6gNodeExt != 0) && (here->BSIM4v6sNodePrime != 0)) {
			while (here->BSIM4v6GEspPtr != ckt->CKTkluBind_KLU [i]) i ++ ;
			here->BSIM4v6GEspPtr = ckt->CKTkluBind_KLU_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6gNodeExt != 0) && (here->BSIM4v6bNodePrime != 0)) {
			while (here->BSIM4v6GEbpPtr != ckt->CKTkluBind_KLU [i]) i ++ ;
			here->BSIM4v6GEbpPtr = ckt->CKTkluBind_KLU_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6gNodeMid != 0) && (here->BSIM4v6dNodePrime != 0)) {
			while (here->BSIM4v6GMdpPtr != ckt->CKTkluBind_KLU [i]) i ++ ;
			here->BSIM4v6GMdpPtr = ckt->CKTkluBind_KLU_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6gNodeMid != 0) && (here->BSIM4v6gNodePrime != 0)) {
			while (here->BSIM4v6GMgpPtr != ckt->CKTkluBind_KLU [i]) i ++ ;
			here->BSIM4v6GMgpPtr = ckt->CKTkluBind_KLU_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6gNodeMid != 0) && (here->BSIM4v6gNodeMid != 0)) {
			while (here->BSIM4v6GMgmPtr != ckt->CKTkluBind_KLU [i]) i ++ ;
			here->BSIM4v6GMgmPtr = ckt->CKTkluBind_KLU_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6gNodeMid != 0) && (here->BSIM4v6gNodeExt != 0)) {
			while (here->BSIM4v6GMgePtr != ckt->CKTkluBind_KLU [i]) i ++ ;
			here->BSIM4v6GMgePtr = ckt->CKTkluBind_KLU_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6gNodeMid != 0) && (here->BSIM4v6sNodePrime != 0)) {
			while (here->BSIM4v6GMspPtr != ckt->CKTkluBind_KLU [i]) i ++ ;
			here->BSIM4v6GMspPtr = ckt->CKTkluBind_KLU_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6gNodeMid != 0) && (here->BSIM4v6bNodePrime != 0)) {
			while (here->BSIM4v6GMbpPtr != ckt->CKTkluBind_KLU [i]) i ++ ;
			here->BSIM4v6GMbpPtr = ckt->CKTkluBind_KLU_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6dNodePrime != 0) && (here->BSIM4v6gNodeMid != 0)) {
			while (here->BSIM4v6DPgmPtr != ckt->CKTkluBind_KLU [i]) i ++ ;
			here->BSIM4v6DPgmPtr = ckt->CKTkluBind_KLU_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6gNodePrime != 0) && (here->BSIM4v6gNodeMid != 0)) {
			while (here->BSIM4v6GPgmPtr != ckt->CKTkluBind_KLU [i]) i ++ ;
			here->BSIM4v6GPgmPtr = ckt->CKTkluBind_KLU_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6gNodeExt != 0) && (here->BSIM4v6gNodeMid != 0)) {
			while (here->BSIM4v6GEgmPtr != ckt->CKTkluBind_KLU [i]) i ++ ;
			here->BSIM4v6GEgmPtr = ckt->CKTkluBind_KLU_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6sNodePrime != 0) && (here->BSIM4v6gNodeMid != 0)) {
			while (here->BSIM4v6SPgmPtr != ckt->CKTkluBind_KLU [i]) i ++ ;
			here->BSIM4v6SPgmPtr = ckt->CKTkluBind_KLU_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6bNodePrime != 0) && (here->BSIM4v6gNodeMid != 0)) {
			while (here->BSIM4v6BPgmPtr != ckt->CKTkluBind_KLU [i]) i ++ ;
			here->BSIM4v6BPgmPtr = ckt->CKTkluBind_KLU_Complex [i] ;
		}

	    }

	    if ((here->BSIM4v6rbodyMod ==1) || (here->BSIM4v6rbodyMod ==2)) {

		i = 0 ;
		if ((here->BSIM4v6dNodePrime != 0) && (here->BSIM4v6dbNode != 0)) {
			while (here->BSIM4v6DPdbPtr != ckt->CKTkluBind_KLU [i]) i ++ ;
			here->BSIM4v6DPdbPtr = ckt->CKTkluBind_KLU_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6sNodePrime != 0) && (here->BSIM4v6sbNode != 0)) {
			while (here->BSIM4v6SPsbPtr != ckt->CKTkluBind_KLU [i]) i ++ ;
			here->BSIM4v6SPsbPtr = ckt->CKTkluBind_KLU_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6dbNode != 0) && (here->BSIM4v6dNodePrime != 0)) {
			while (here->BSIM4v6DBdpPtr != ckt->CKTkluBind_KLU [i]) i ++ ;
			here->BSIM4v6DBdpPtr = ckt->CKTkluBind_KLU_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6dbNode != 0) && (here->BSIM4v6dbNode != 0)) {
			while (here->BSIM4v6DBdbPtr != ckt->CKTkluBind_KLU [i]) i ++ ;
			here->BSIM4v6DBdbPtr = ckt->CKTkluBind_KLU_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6dbNode != 0) && (here->BSIM4v6bNodePrime != 0)) {
			while (here->BSIM4v6DBbpPtr != ckt->CKTkluBind_KLU [i]) i ++ ;
			here->BSIM4v6DBbpPtr = ckt->CKTkluBind_KLU_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6dbNode != 0) && (here->BSIM4v6bNode != 0)) {
			while (here->BSIM4v6DBbPtr != ckt->CKTkluBind_KLU [i]) i ++ ;
			here->BSIM4v6DBbPtr = ckt->CKTkluBind_KLU_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6bNodePrime != 0) && (here->BSIM4v6dbNode != 0)) {
			while (here->BSIM4v6BPdbPtr != ckt->CKTkluBind_KLU [i]) i ++ ;
			here->BSIM4v6BPdbPtr = ckt->CKTkluBind_KLU_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6bNodePrime != 0) && (here->BSIM4v6bNode != 0)) {
			while (here->BSIM4v6BPbPtr != ckt->CKTkluBind_KLU [i]) i ++ ;
			here->BSIM4v6BPbPtr = ckt->CKTkluBind_KLU_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6bNodePrime != 0) && (here->BSIM4v6sbNode != 0)) {
			while (here->BSIM4v6BPsbPtr != ckt->CKTkluBind_KLU [i]) i ++ ;
			here->BSIM4v6BPsbPtr = ckt->CKTkluBind_KLU_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6sbNode != 0) && (here->BSIM4v6sNodePrime != 0)) {
			while (here->BSIM4v6SBspPtr != ckt->CKTkluBind_KLU [i]) i ++ ;
			here->BSIM4v6SBspPtr = ckt->CKTkluBind_KLU_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6sbNode != 0) && (here->BSIM4v6bNodePrime != 0)) {
			while (here->BSIM4v6SBbpPtr != ckt->CKTkluBind_KLU [i]) i ++ ;
			here->BSIM4v6SBbpPtr = ckt->CKTkluBind_KLU_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6sbNode != 0) && (here->BSIM4v6bNode != 0)) {
			while (here->BSIM4v6SBbPtr != ckt->CKTkluBind_KLU [i]) i ++ ;
			here->BSIM4v6SBbPtr = ckt->CKTkluBind_KLU_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6sbNode != 0) && (here->BSIM4v6sbNode != 0)) {
			while (here->BSIM4v6SBsbPtr != ckt->CKTkluBind_KLU [i]) i ++ ;
			here->BSIM4v6SBsbPtr = ckt->CKTkluBind_KLU_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6bNode != 0) && (here->BSIM4v6dbNode != 0)) {
			while (here->BSIM4v6BdbPtr != ckt->CKTkluBind_KLU [i]) i ++ ;
			here->BSIM4v6BdbPtr = ckt->CKTkluBind_KLU_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6bNode != 0) && (here->BSIM4v6bNodePrime != 0)) {
			while (here->BSIM4v6BbpPtr != ckt->CKTkluBind_KLU [i]) i ++ ;
			here->BSIM4v6BbpPtr = ckt->CKTkluBind_KLU_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6bNode != 0) && (here->BSIM4v6sbNode != 0)) {
			while (here->BSIM4v6BsbPtr != ckt->CKTkluBind_KLU [i]) i ++ ;
			here->BSIM4v6BsbPtr = ckt->CKTkluBind_KLU_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6bNode != 0) && (here->BSIM4v6bNode != 0)) {
			while (here->BSIM4v6BbPtr != ckt->CKTkluBind_KLU [i]) i ++ ;
			here->BSIM4v6BbPtr = ckt->CKTkluBind_KLU_Complex [i] ;
		}

	    }

            if (model->BSIM4v6rdsMod) {

		i = 0 ;
		if ((here->BSIM4v6dNode != 0) && (here->BSIM4v6gNodePrime != 0)) {
			while (here->BSIM4v6DgpPtr != ckt->CKTkluBind_KLU [i]) i ++ ;
			here->BSIM4v6DgpPtr = ckt->CKTkluBind_KLU_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6dNode != 0) && (here->BSIM4v6sNodePrime != 0)) {
			while (here->BSIM4v6DspPtr != ckt->CKTkluBind_KLU [i]) i ++ ;
			here->BSIM4v6DspPtr = ckt->CKTkluBind_KLU_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6dNode != 0) && (here->BSIM4v6bNodePrime != 0)) {
			while (here->BSIM4v6DbpPtr != ckt->CKTkluBind_KLU [i]) i ++ ;
			here->BSIM4v6DbpPtr = ckt->CKTkluBind_KLU_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6sNode != 0) && (here->BSIM4v6dNodePrime != 0)) {
			while (here->BSIM4v6SdpPtr != ckt->CKTkluBind_KLU [i]) i ++ ;
			here->BSIM4v6SdpPtr = ckt->CKTkluBind_KLU_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6sNode != 0) && (here->BSIM4v6gNodePrime != 0)) {
			while (here->BSIM4v6SgpPtr != ckt->CKTkluBind_KLU [i]) i ++ ;
			here->BSIM4v6SgpPtr = ckt->CKTkluBind_KLU_Complex [i] ;
		}

		i = 0 ;
		if ((here->BSIM4v6sNode != 0) && (here->BSIM4v6bNodePrime != 0)) {
			while (here->BSIM4v6SbpPtr != ckt->CKTkluBind_KLU [i]) i ++ ;
			here->BSIM4v6SbpPtr = ckt->CKTkluBind_KLU_Complex [i] ;
		}

	    }
	}
    }
    return(OK);
}
