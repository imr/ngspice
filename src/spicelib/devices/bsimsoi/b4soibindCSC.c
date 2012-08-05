/**********
Author: 2012 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "b4soidef.h"
#include "ngspice/sperror.h"

int
B4SOIbindCSC(GENmodel *inModel, CKTcircuit *ckt)
{
    B4SOImodel *model = (B4SOImodel *)inModel;
    int i ;

    /*  loop through all the b4soi models */
    for( ; model != NULL; model = model->B4SOInextModel ) {
	B4SOIinstance *here;

        /* loop through all the instances of the model */
        for (here = model->B4SOIinstances; here != NULL ;
	    here = here->B4SOInextInstance) {

                if ((model->B4SOIshMod == 1) && (here->B4SOIrth0!=0.0)) {

		i = 0 ;
		if ((here-> B4SOItempNode != 0) && (here-> B4SOItempNode != 0)) {
			while (here->B4SOITemptempPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B4SOITemptempPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B4SOItempNode != 0) && (here-> B4SOIdNodePrime != 0)) {
			while (here->B4SOITempdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B4SOITempdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B4SOItempNode != 0) && (here-> B4SOIsNodePrime != 0)) {
			while (here->B4SOITempspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B4SOITempspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B4SOItempNode != 0) && (here-> B4SOIgNode != 0)) {
			while (here->B4SOITempgPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B4SOITempgPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B4SOItempNode != 0) && (here-> B4SOIbNode != 0)) {
			while (here->B4SOITempbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B4SOITempbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIgNode != 0) && (here-> B4SOItempNode != 0)) {
			while (here->B4SOIGtempPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B4SOIGtempPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIdNodePrime != 0) && (here-> B4SOItempNode != 0)) {
			while (here->B4SOIDPtempPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B4SOIDPtempPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIsNodePrime != 0) && (here-> B4SOItempNode != 0)) {
			while (here->B4SOISPtempPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B4SOISPtempPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIeNode != 0) && (here-> B4SOItempNode != 0)) {
			while (here->B4SOIEtempPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B4SOIEtempPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIbNode != 0) && (here-> B4SOItempNode != 0)) {
			while (here->B4SOIBtempPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B4SOIBtempPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

                if (here->B4SOIbodyMod == 1) {

		i = 0 ;
		if ((here-> B4SOIpNode != 0) && (here-> B4SOItempNode != 0)) {
			while (here->B4SOIPtempPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B4SOIPtempPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
                } /* IF */

                if (here->B4SOIsoiMod != 0) { /* v3.2 */

		i = 0 ;
		if ((here-> B4SOItempNode != 0) && (here-> B4SOIeNode != 0)) {
			while (here->B4SOITempePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B4SOITempePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
                } /* IF */
                } /* IF */

                if (here->B4SOIbodyMod == 2) {
                    /* Don't create any Jacobian entry for pNode */
                }
                else if (here->B4SOIbodyMod == 1) { 

		i = 0 ;
		if ((here-> B4SOIbNode != 0) && (here-> B4SOIpNode != 0)) {
			while (here->B4SOIBpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B4SOIBpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIpNode != 0) && (here-> B4SOIbNode != 0)) {
			while (here->B4SOIPbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B4SOIPbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIpNode != 0) && (here-> B4SOIpNode != 0)) {
			while (here->B4SOIPpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B4SOIPpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIpNode != 0) && (here-> B4SOIgNode != 0)) {
			while (here->B4SOIPgPtr  != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B4SOIPgPtr  = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIgNode != 0) && (here-> B4SOIpNode != 0)) {
			while (here->B4SOIGpPtr  != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B4SOIGpPtr  = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
                } /* ELSE */

                if (here->B4SOIrgateMod != 0) {

		i = 0 ;
		if ((here-> B4SOIgNodeExt != 0) && (here-> B4SOIgNodeExt != 0)) {
			while (here->B4SOIGEgePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B4SOIGEgePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIgNodeExt != 0) && (here-> B4SOIgNode != 0)) {
			while (here->B4SOIGEgPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B4SOIGEgPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIgNode != 0) && (here-> B4SOIgNodeExt != 0)) {
			while (here->B4SOIGgePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B4SOIGgePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIgNodeExt != 0) && (here-> B4SOIdNodePrime != 0)) {
			while (here->B4SOIGEdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B4SOIGEdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIgNodeExt != 0) && (here-> B4SOIsNodePrime != 0)) {
			while (here->B4SOIGEspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B4SOIGEspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		if (here->B4SOIsoiMod !=2) /* v3.2 */

		i = 0 ;
		if ((here-> B4SOIgNodeExt != 0) && (here-> B4SOIbNode != 0)) {
			while (here->B4SOIGEbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B4SOIGEbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIgNodeMid != 0) && (here-> B4SOIdNodePrime != 0)) {
			while (here->B4SOIGMdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B4SOIGMdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIgNodeMid != 0) && (here-> B4SOIgNode != 0)) {
			while (here->B4SOIGMgPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B4SOIGMgPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIgNodeMid != 0) && (here-> B4SOIgNodeMid != 0)) {
			while (here->B4SOIGMgmPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B4SOIGMgmPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIgNodeMid != 0) && (here-> B4SOIgNodeExt != 0)) {
			while (here->B4SOIGMgePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B4SOIGMgePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIgNodeMid != 0) && (here-> B4SOIsNodePrime != 0)) {
			while (here->B4SOIGMspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B4SOIGMspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		if (here->B4SOIsoiMod !=2) /* v3.2 */

		i = 0 ;
		if ((here-> B4SOIgNodeMid != 0) && (here-> B4SOIbNode != 0)) {
			while (here->B4SOIGMbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B4SOIGMbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIgNodeMid != 0) && (here-> B4SOIeNode != 0)) {
			while (here->B4SOIGMePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B4SOIGMePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIdNodePrime != 0) && (here-> B4SOIgNodeMid != 0)) {
			while (here->B4SOIDPgmPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B4SOIDPgmPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIgNode != 0) && (here-> B4SOIgNodeMid != 0)) {
			while (here->B4SOIGgmPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B4SOIGgmPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIgNodeExt != 0) && (here-> B4SOIgNodeMid != 0)) {
			while (here->B4SOIGEgmPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B4SOIGEgmPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIsNodePrime != 0) && (here-> B4SOIgNodeMid != 0)) {
			while (here->B4SOISPgmPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B4SOISPgmPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIeNode != 0) && (here-> B4SOIgNodeMid != 0)) {
			while (here->B4SOIEgmPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B4SOIEgmPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
                } /* IF */

                if (here->B4SOIsoiMod != 2) {

		i = 0 ;
		if ((here-> B4SOIeNode != 0) && (here-> B4SOIbNode != 0)) {
			while (here->B4SOIEbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B4SOIEbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIgNode != 0) && (here-> B4SOIbNode != 0)) {
			while (here->B4SOIGbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B4SOIGbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIdNodePrime != 0) && (here-> B4SOIbNode != 0)) {
			while (here->B4SOIDPbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B4SOIDPbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIsNodePrime != 0) && (here-> B4SOIbNode != 0)) {
			while (here->B4SOISPbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B4SOISPbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIbNode != 0) && (here-> B4SOIeNode != 0)) {
			while (here->B4SOIBePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B4SOIBePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIbNode != 0) && (here-> B4SOIgNode != 0)) {
			while (here->B4SOIBgPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B4SOIBgPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIbNode != 0) && (here-> B4SOIdNodePrime != 0)) {
			while (here->B4SOIBdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B4SOIBdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIbNode != 0) && (here-> B4SOIsNodePrime != 0)) {
			while (here->B4SOIBspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B4SOIBspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIbNode != 0) && (here-> B4SOIbNode != 0)) {
			while (here->B4SOIBbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B4SOIBbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
                } /* IF */

		i = 0 ;
		if ((here-> B4SOIeNode != 0) && (here-> B4SOIgNode != 0)) {
			while (here->B4SOIEgPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B4SOIEgPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIeNode != 0) && (here-> B4SOIdNodePrime != 0)) {
			while (here->B4SOIEdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B4SOIEdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIeNode != 0) && (here-> B4SOIsNodePrime != 0)) {
			while (here->B4SOIEspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B4SOIEspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIgNode != 0) && (here-> B4SOIeNode != 0)) {
			while (here->B4SOIGePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B4SOIGePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIdNodePrime != 0) && (here-> B4SOIeNode != 0)) {
			while (here->B4SOIDPePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B4SOIDPePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIsNodePrime != 0) && (here-> B4SOIeNode != 0)) {
			while (here->B4SOISPePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B4SOISPePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIeNode != 0) && (here-> B4SOIbNode != 0)) {
			while (here->B4SOIEbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B4SOIEbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIeNode != 0) && (here-> B4SOIeNode != 0)) {
			while (here->B4SOIEePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B4SOIEePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIgNode != 0) && (here-> B4SOIgNode != 0)) {
			while (here->B4SOIGgPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B4SOIGgPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIgNode != 0) && (here-> B4SOIdNodePrime != 0)) {
			while (here->B4SOIGdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B4SOIGdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIgNode != 0) && (here-> B4SOIsNodePrime != 0)) {
			while (here->B4SOIGspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B4SOIGspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIdNodePrime != 0) && (here-> B4SOIgNode != 0)) {
			while (here->B4SOIDPgPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B4SOIDPgPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIdNodePrime != 0) && (here-> B4SOIdNodePrime != 0)) {
			while (here->B4SOIDPdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B4SOIDPdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIdNodePrime != 0) && (here-> B4SOIsNodePrime != 0)) {
			while (here->B4SOIDPspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B4SOIDPspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIdNodePrime != 0) && (here-> B4SOIdNode != 0)) {
			while (here->B4SOIDPdPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B4SOIDPdPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIsNodePrime != 0) && (here-> B4SOIgNode != 0)) {
			while (here->B4SOISPgPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B4SOISPgPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIsNodePrime != 0) && (here-> B4SOIdNodePrime != 0)) {
			while (here->B4SOISPdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B4SOISPdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIsNodePrime != 0) && (here-> B4SOIsNodePrime != 0)) {
			while (here->B4SOISPspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B4SOISPspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIsNodePrime != 0) && (here-> B4SOIsNode != 0)) {
			while (here->B4SOISPsPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B4SOISPsPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIdNode != 0) && (here-> B4SOIdNode != 0)) {
			while (here->B4SOIDdPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B4SOIDdPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIdNode != 0) && (here-> B4SOIdNodePrime != 0)) {
			while (here->B4SOIDdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B4SOIDdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIsNode != 0) && (here-> B4SOIsNode != 0)) {
			while (here->B4SOISsPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B4SOISsPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIsNode != 0) && (here-> B4SOIsNodePrime != 0)) {
			while (here->B4SOISspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B4SOISspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

                if (here->B4SOIrbodyMod == 1) {

		i = 0 ;
		if ((here-> B4SOIdNodePrime != 0) && (here-> B4SOIdbNode != 0)) {
			while (here->B4SOIDPdbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B4SOIDPdbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIsNodePrime != 0) && (here-> B4SOIsbNode != 0)) {
			while (here->B4SOISPsbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B4SOISPsbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIdbNode != 0) && (here-> B4SOIdNodePrime != 0)) {
			while (here->B4SOIDBdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B4SOIDBdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIdbNode != 0) && (here-> B4SOIdbNode != 0)) {
			while (here->B4SOIDBdbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B4SOIDBdbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIdbNode != 0) && (here-> B4SOIbNode != 0)) {
			while (here->B4SOIDBbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B4SOIDBbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIsbNode != 0) && (here-> B4SOIsNodePrime != 0)) {
			while (here->B4SOISBspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B4SOISBspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIsbNode != 0) && (here-> B4SOIsbNode != 0)) {
			while (here->B4SOISBsbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B4SOISBsbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIsbNode != 0) && (here-> B4SOIbNode != 0)) {
			while (here->B4SOISBbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B4SOISBbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIbNode != 0) && (here-> B4SOIdbNode != 0)) {
			while (here->B4SOIBdbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B4SOIBdbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIbNode != 0) && (here-> B4SOIsbNode != 0)) {
			while (here->B4SOIBsbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B4SOIBsbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
                } /* IF */

                if (model->B4SOIrdsMod) {

		i = 0 ;
		if ((here-> B4SOIdNode != 0) && (here-> B4SOIgNode != 0)) {
			while (here->B4SOIDgPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B4SOIDgPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIdNode != 0) && (here-> B4SOIsNodePrime != 0)) {
			while (here->B4SOIDspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B4SOIDspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIsNode != 0) && (here-> B4SOIdNodePrime != 0)) {
			while (here->B4SOISdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B4SOISdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIsNode != 0) && (here-> B4SOIgNode != 0)) {
			while (here->B4SOISgPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B4SOISgPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		if (model->B4SOIsoiMod != 2)  {

		i = 0 ;
		if ((here-> B4SOIdNode != 0) && (here-> B4SOIbNode != 0)) {
			while (here->B4SOIDbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B4SOIDbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIsNode != 0) && (here-> B4SOIbNode != 0)) {
			while (here->B4SOISbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B4SOISbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
                } /* ELSE */
                } /* ELSE */

                if (here->B4SOIdebugMod != 0) {

		i = 0 ;
		if ((here-> B4SOIvbsNode != 0) && (here-> B4SOIvbsNode != 0)) {
			while (here->B4SOIVbsPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B4SOIVbsPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIidsNode != 0) && (here-> B4SOIidsNode != 0)) {
			while (here->B4SOIIdsPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B4SOIIdsPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIicNode != 0) && (here-> B4SOIicNode != 0)) {
			while (here->B4SOIIcPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B4SOIIcPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIibsNode != 0) && (here-> B4SOIibsNode != 0)) {
			while (here->B4SOIIbsPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B4SOIIbsPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIibdNode != 0) && (here-> B4SOIibdNode != 0)) {
			while (here->B4SOIIbdPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B4SOIIbdPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIiiiNode != 0) && (here-> B4SOIiiiNode != 0)) {
			while (here->B4SOIIiiPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B4SOIIiiPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIigNode != 0) && (here-> B4SOIigNode != 0)) {
			while (here->B4SOIIgPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B4SOIIgPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIgiggNode != 0) && (here-> B4SOIgiggNode != 0)) {
			while (here->B4SOIGiggPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B4SOIGiggPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIgigdNode != 0) && (here-> B4SOIgigdNode != 0)) {
			while (here->B4SOIGigdPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B4SOIGigdPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIgigbNode != 0) && (here-> B4SOIgigbNode != 0)) {
			while (here->B4SOIGigbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B4SOIGigbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIigidlNode != 0) && (here-> B4SOIigidlNode != 0)) {
			while (here->B4SOIIgidlPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B4SOIIgidlPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIitunNode != 0) && (here-> B4SOIitunNode != 0)) {
			while (here->B4SOIItunPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B4SOIItunPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIibpNode != 0) && (here-> B4SOIibpNode != 0)) {
			while (here->B4SOIIbpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B4SOIIbpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIcbbNode != 0) && (here-> B4SOIcbbNode != 0)) {
			while (here->B4SOICbbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B4SOICbbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIcbdNode != 0) && (here-> B4SOIcbdNode != 0)) {
			while (here->B4SOICbdPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B4SOICbdPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIcbgNode != 0) && (here-> B4SOIcbgNode != 0)) {
			while (here->B4SOICbgPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B4SOICbgPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIqbfNode != 0) && (here-> B4SOIqbfNode != 0)) {
			while (here->B4SOIQbfPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B4SOIQbfPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIqjsNode != 0) && (here-> B4SOIqjsNode != 0)) {
			while (here->B4SOIQjsPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B4SOIQjsPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIqjdNode != 0) && (here-> B4SOIqjdNode != 0)) {
			while (here->B4SOIQjdPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B4SOIQjdPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
                } /* IF */
	}
    }
    return(OK);
}

int
B4SOIbindCSCComplex(GENmodel *inModel, CKTcircuit *ckt)
{
    B4SOImodel *model = (B4SOImodel *)inModel;
    int i ;

    /*  loop through all the b4soi models */
    for( ; model != NULL; model = model->B4SOInextModel ) {
	B4SOIinstance *here;

        /* loop through all the instances of the model */
        for (here = model->B4SOIinstances; here != NULL ;
	    here = here->B4SOInextInstance) {

                if ((model->B4SOIshMod == 1) && (here->B4SOIrth0!=0.0)) {

		i = 0 ;
		if ((here-> B4SOItempNode != 0) && (here-> B4SOItempNode != 0)) {
			while (here->B4SOITemptempPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B4SOITemptempPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B4SOItempNode != 0) && (here-> B4SOIdNodePrime != 0)) {
			while (here->B4SOITempdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B4SOITempdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B4SOItempNode != 0) && (here-> B4SOIsNodePrime != 0)) {
			while (here->B4SOITempspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B4SOITempspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B4SOItempNode != 0) && (here-> B4SOIgNode != 0)) {
			while (here->B4SOITempgPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B4SOITempgPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B4SOItempNode != 0) && (here-> B4SOIbNode != 0)) {
			while (here->B4SOITempbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B4SOITempbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIgNode != 0) && (here-> B4SOItempNode != 0)) {
			while (here->B4SOIGtempPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B4SOIGtempPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIdNodePrime != 0) && (here-> B4SOItempNode != 0)) {
			while (here->B4SOIDPtempPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B4SOIDPtempPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIsNodePrime != 0) && (here-> B4SOItempNode != 0)) {
			while (here->B4SOISPtempPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B4SOISPtempPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIeNode != 0) && (here-> B4SOItempNode != 0)) {
			while (here->B4SOIEtempPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B4SOIEtempPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIbNode != 0) && (here-> B4SOItempNode != 0)) {
			while (here->B4SOIBtempPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B4SOIBtempPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

                if (here->B4SOIbodyMod == 1) {

		i = 0 ;
		if ((here-> B4SOIpNode != 0) && (here-> B4SOItempNode != 0)) {
			while (here->B4SOIPtempPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B4SOIPtempPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
                } /* IF */

                if (here->B4SOIsoiMod != 0) { /* v3.2 */

		i = 0 ;
		if ((here-> B4SOItempNode != 0) && (here-> B4SOIeNode != 0)) {
			while (here->B4SOITempePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B4SOITempePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
                } /* IF */
                } /* IF */

                if (here->B4SOIbodyMod == 2) {
                    /* Don't create any Jacobian entry for pNode */
                }
                else if (here->B4SOIbodyMod == 1) { 

		i = 0 ;
		if ((here-> B4SOIbNode != 0) && (here-> B4SOIpNode != 0)) {
			while (here->B4SOIBpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B4SOIBpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIpNode != 0) && (here-> B4SOIbNode != 0)) {
			while (here->B4SOIPbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B4SOIPbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIpNode != 0) && (here-> B4SOIpNode != 0)) {
			while (here->B4SOIPpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B4SOIPpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIpNode != 0) && (here-> B4SOIgNode != 0)) {
			while (here->B4SOIPgPtr  != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B4SOIPgPtr  = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIgNode != 0) && (here-> B4SOIpNode != 0)) {
			while (here->B4SOIGpPtr  != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B4SOIGpPtr  = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
                } /* ELSE */

                if (here->B4SOIrgateMod != 0) {

		i = 0 ;
		if ((here-> B4SOIgNodeExt != 0) && (here-> B4SOIgNodeExt != 0)) {
			while (here->B4SOIGEgePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B4SOIGEgePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIgNodeExt != 0) && (here-> B4SOIgNode != 0)) {
			while (here->B4SOIGEgPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B4SOIGEgPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIgNode != 0) && (here-> B4SOIgNodeExt != 0)) {
			while (here->B4SOIGgePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B4SOIGgePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIgNodeExt != 0) && (here-> B4SOIdNodePrime != 0)) {
			while (here->B4SOIGEdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B4SOIGEdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIgNodeExt != 0) && (here-> B4SOIsNodePrime != 0)) {
			while (here->B4SOIGEspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B4SOIGEspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		if (here->B4SOIsoiMod !=2) /* v3.2 */

		i = 0 ;
		if ((here-> B4SOIgNodeExt != 0) && (here-> B4SOIbNode != 0)) {
			while (here->B4SOIGEbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B4SOIGEbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIgNodeMid != 0) && (here-> B4SOIdNodePrime != 0)) {
			while (here->B4SOIGMdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B4SOIGMdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIgNodeMid != 0) && (here-> B4SOIgNode != 0)) {
			while (here->B4SOIGMgPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B4SOIGMgPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIgNodeMid != 0) && (here-> B4SOIgNodeMid != 0)) {
			while (here->B4SOIGMgmPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B4SOIGMgmPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIgNodeMid != 0) && (here-> B4SOIgNodeExt != 0)) {
			while (here->B4SOIGMgePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B4SOIGMgePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIgNodeMid != 0) && (here-> B4SOIsNodePrime != 0)) {
			while (here->B4SOIGMspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B4SOIGMspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		if (here->B4SOIsoiMod !=2) /* v3.2 */

		i = 0 ;
		if ((here-> B4SOIgNodeMid != 0) && (here-> B4SOIbNode != 0)) {
			while (here->B4SOIGMbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B4SOIGMbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIgNodeMid != 0) && (here-> B4SOIeNode != 0)) {
			while (here->B4SOIGMePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B4SOIGMePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIdNodePrime != 0) && (here-> B4SOIgNodeMid != 0)) {
			while (here->B4SOIDPgmPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B4SOIDPgmPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIgNode != 0) && (here-> B4SOIgNodeMid != 0)) {
			while (here->B4SOIGgmPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B4SOIGgmPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIgNodeExt != 0) && (here-> B4SOIgNodeMid != 0)) {
			while (here->B4SOIGEgmPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B4SOIGEgmPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIsNodePrime != 0) && (here-> B4SOIgNodeMid != 0)) {
			while (here->B4SOISPgmPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B4SOISPgmPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIeNode != 0) && (here-> B4SOIgNodeMid != 0)) {
			while (here->B4SOIEgmPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B4SOIEgmPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
                } /* IF */

                if (here->B4SOIsoiMod != 2) {

		i = 0 ;
		if ((here-> B4SOIeNode != 0) && (here-> B4SOIbNode != 0)) {
			while (here->B4SOIEbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B4SOIEbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIgNode != 0) && (here-> B4SOIbNode != 0)) {
			while (here->B4SOIGbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B4SOIGbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIdNodePrime != 0) && (here-> B4SOIbNode != 0)) {
			while (here->B4SOIDPbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B4SOIDPbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIsNodePrime != 0) && (here-> B4SOIbNode != 0)) {
			while (here->B4SOISPbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B4SOISPbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIbNode != 0) && (here-> B4SOIeNode != 0)) {
			while (here->B4SOIBePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B4SOIBePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIbNode != 0) && (here-> B4SOIgNode != 0)) {
			while (here->B4SOIBgPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B4SOIBgPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIbNode != 0) && (here-> B4SOIdNodePrime != 0)) {
			while (here->B4SOIBdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B4SOIBdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIbNode != 0) && (here-> B4SOIsNodePrime != 0)) {
			while (here->B4SOIBspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B4SOIBspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIbNode != 0) && (here-> B4SOIbNode != 0)) {
			while (here->B4SOIBbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B4SOIBbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
                } /* IF */

		i = 0 ;
		if ((here-> B4SOIeNode != 0) && (here-> B4SOIgNode != 0)) {
			while (here->B4SOIEgPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B4SOIEgPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIeNode != 0) && (here-> B4SOIdNodePrime != 0)) {
			while (here->B4SOIEdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B4SOIEdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIeNode != 0) && (here-> B4SOIsNodePrime != 0)) {
			while (here->B4SOIEspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B4SOIEspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIgNode != 0) && (here-> B4SOIeNode != 0)) {
			while (here->B4SOIGePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B4SOIGePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIdNodePrime != 0) && (here-> B4SOIeNode != 0)) {
			while (here->B4SOIDPePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B4SOIDPePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIsNodePrime != 0) && (here-> B4SOIeNode != 0)) {
			while (here->B4SOISPePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B4SOISPePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIeNode != 0) && (here-> B4SOIbNode != 0)) {
			while (here->B4SOIEbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B4SOIEbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIeNode != 0) && (here-> B4SOIeNode != 0)) {
			while (here->B4SOIEePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B4SOIEePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIgNode != 0) && (here-> B4SOIgNode != 0)) {
			while (here->B4SOIGgPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B4SOIGgPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIgNode != 0) && (here-> B4SOIdNodePrime != 0)) {
			while (here->B4SOIGdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B4SOIGdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIgNode != 0) && (here-> B4SOIsNodePrime != 0)) {
			while (here->B4SOIGspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B4SOIGspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIdNodePrime != 0) && (here-> B4SOIgNode != 0)) {
			while (here->B4SOIDPgPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B4SOIDPgPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIdNodePrime != 0) && (here-> B4SOIdNodePrime != 0)) {
			while (here->B4SOIDPdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B4SOIDPdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIdNodePrime != 0) && (here-> B4SOIsNodePrime != 0)) {
			while (here->B4SOIDPspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B4SOIDPspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIdNodePrime != 0) && (here-> B4SOIdNode != 0)) {
			while (here->B4SOIDPdPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B4SOIDPdPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIsNodePrime != 0) && (here-> B4SOIgNode != 0)) {
			while (here->B4SOISPgPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B4SOISPgPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIsNodePrime != 0) && (here-> B4SOIdNodePrime != 0)) {
			while (here->B4SOISPdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B4SOISPdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIsNodePrime != 0) && (here-> B4SOIsNodePrime != 0)) {
			while (here->B4SOISPspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B4SOISPspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIsNodePrime != 0) && (here-> B4SOIsNode != 0)) {
			while (here->B4SOISPsPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B4SOISPsPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIdNode != 0) && (here-> B4SOIdNode != 0)) {
			while (here->B4SOIDdPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B4SOIDdPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIdNode != 0) && (here-> B4SOIdNodePrime != 0)) {
			while (here->B4SOIDdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B4SOIDdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIsNode != 0) && (here-> B4SOIsNode != 0)) {
			while (here->B4SOISsPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B4SOISsPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIsNode != 0) && (here-> B4SOIsNodePrime != 0)) {
			while (here->B4SOISspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B4SOISspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

                if (here->B4SOIrbodyMod == 1) {

		i = 0 ;
		if ((here-> B4SOIdNodePrime != 0) && (here-> B4SOIdbNode != 0)) {
			while (here->B4SOIDPdbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B4SOIDPdbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIsNodePrime != 0) && (here-> B4SOIsbNode != 0)) {
			while (here->B4SOISPsbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B4SOISPsbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIdbNode != 0) && (here-> B4SOIdNodePrime != 0)) {
			while (here->B4SOIDBdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B4SOIDBdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIdbNode != 0) && (here-> B4SOIdbNode != 0)) {
			while (here->B4SOIDBdbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B4SOIDBdbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIdbNode != 0) && (here-> B4SOIbNode != 0)) {
			while (here->B4SOIDBbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B4SOIDBbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIsbNode != 0) && (here-> B4SOIsNodePrime != 0)) {
			while (here->B4SOISBspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B4SOISBspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIsbNode != 0) && (here-> B4SOIsbNode != 0)) {
			while (here->B4SOISBsbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B4SOISBsbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIsbNode != 0) && (here-> B4SOIbNode != 0)) {
			while (here->B4SOISBbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B4SOISBbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIbNode != 0) && (here-> B4SOIdbNode != 0)) {
			while (here->B4SOIBdbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B4SOIBdbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIbNode != 0) && (here-> B4SOIsbNode != 0)) {
			while (here->B4SOIBsbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B4SOIBsbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
                } /* IF */

                if (model->B4SOIrdsMod) {

		i = 0 ;
		if ((here-> B4SOIdNode != 0) && (here-> B4SOIgNode != 0)) {
			while (here->B4SOIDgPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B4SOIDgPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIdNode != 0) && (here-> B4SOIsNodePrime != 0)) {
			while (here->B4SOIDspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B4SOIDspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIsNode != 0) && (here-> B4SOIdNodePrime != 0)) {
			while (here->B4SOISdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B4SOISdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIsNode != 0) && (here-> B4SOIgNode != 0)) {
			while (here->B4SOISgPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B4SOISgPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		if (model->B4SOIsoiMod != 2)  {

		i = 0 ;
		if ((here-> B4SOIdNode != 0) && (here-> B4SOIbNode != 0)) {
			while (here->B4SOIDbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B4SOIDbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIsNode != 0) && (here-> B4SOIbNode != 0)) {
			while (here->B4SOISbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B4SOISbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
                } /* IF */
                } /* IF */

                if (here->B4SOIdebugMod != 0) {

		i = 0 ;
		if ((here-> B4SOIvbsNode != 0) && (here-> B4SOIvbsNode != 0)) {
			while (here->B4SOIVbsPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B4SOIVbsPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIidsNode != 0) && (here-> B4SOIidsNode != 0)) {
			while (here->B4SOIIdsPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B4SOIIdsPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIicNode != 0) && (here-> B4SOIicNode != 0)) {
			while (here->B4SOIIcPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B4SOIIcPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIibsNode != 0) && (here-> B4SOIibsNode != 0)) {
			while (here->B4SOIIbsPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B4SOIIbsPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIibdNode != 0) && (here-> B4SOIibdNode != 0)) {
			while (here->B4SOIIbdPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B4SOIIbdPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIiiiNode != 0) && (here-> B4SOIiiiNode != 0)) {
			while (here->B4SOIIiiPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B4SOIIiiPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIigNode != 0) && (here-> B4SOIigNode != 0)) {
			while (here->B4SOIIgPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B4SOIIgPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIgiggNode != 0) && (here-> B4SOIgiggNode != 0)) {
			while (here->B4SOIGiggPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B4SOIGiggPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIgigdNode != 0) && (here-> B4SOIgigdNode != 0)) {
			while (here->B4SOIGigdPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B4SOIGigdPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIgigbNode != 0) && (here-> B4SOIgigbNode != 0)) {
			while (here->B4SOIGigbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B4SOIGigbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIigidlNode != 0) && (here-> B4SOIigidlNode != 0)) {
			while (here->B4SOIIgidlPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B4SOIIgidlPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIitunNode != 0) && (here-> B4SOIitunNode != 0)) {
			while (here->B4SOIItunPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B4SOIItunPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIibpNode != 0) && (here-> B4SOIibpNode != 0)) {
			while (here->B4SOIIbpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B4SOIIbpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIcbbNode != 0) && (here-> B4SOIcbbNode != 0)) {
			while (here->B4SOICbbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B4SOICbbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIcbdNode != 0) && (here-> B4SOIcbdNode != 0)) {
			while (here->B4SOICbdPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B4SOICbdPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIcbgNode != 0) && (here-> B4SOIcbgNode != 0)) {
			while (here->B4SOICbgPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B4SOICbgPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIqbfNode != 0) && (here-> B4SOIqbfNode != 0)) {
			while (here->B4SOIQbfPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B4SOIQbfPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIqjsNode != 0) && (here-> B4SOIqjsNode != 0)) {
			while (here->B4SOIQjsPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B4SOIQjsPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B4SOIqjdNode != 0) && (here-> B4SOIqjdNode != 0)) {
			while (here->B4SOIQjdPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B4SOIQjdPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
                } /* IF */
	}
    }
    return(OK);
}

int
B4SOIbindCSCComplexToReal (GENmodel *inModel, CKTcircuit *ckt)
{
    B4SOImodel *model = (B4SOImodel *)inModel ;
    B4SOIinstance *here ;
    int i ;

    /*  loop through all the bsim4SiliconOnInsulator models */
    for ( ; model != NULL ; model = model->B4SOInextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->B4SOIinstances ; here != NULL ; here = here->B4SOInextInstance)
        {
            i = 0 ;
            if ((here->B4SOItempNode != 0) && (here->B4SOItempNode != 0))
            {
                while (here->B4SOITemptempPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B4SOITemptempPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B4SOItempNode != 0) && (here->B4SOIdNodePrime != 0))
            {
                while (here->B4SOITempdpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B4SOITempdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B4SOItempNode != 0) && (here->B4SOIsNodePrime != 0))
            {
                while (here->B4SOITempspPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B4SOITempspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B4SOItempNode != 0) && (here->B4SOIgNode != 0))
            {
                while (here->B4SOITempgPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B4SOITempgPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B4SOItempNode != 0) && (here->B4SOIbNode != 0))
            {
                while (here->B4SOITempbPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B4SOITempbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B4SOIgNode != 0) && (here->B4SOItempNode != 0))
            {
                while (here->B4SOIGtempPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B4SOIGtempPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B4SOIdNodePrime != 0) && (here->B4SOItempNode != 0))
            {
                while (here->B4SOIDPtempPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B4SOIDPtempPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B4SOIsNodePrime != 0) && (here->B4SOItempNode != 0))
            {
                while (here->B4SOISPtempPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B4SOISPtempPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B4SOIeNode != 0) && (here->B4SOItempNode != 0))
            {
                while (here->B4SOIEtempPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B4SOIEtempPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B4SOIbNode != 0) && (here->B4SOItempNode != 0))
            {
                while (here->B4SOIBtempPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B4SOIBtempPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B4SOIpNode != 0) && (here->B4SOItempNode != 0))
            {
                while (here->B4SOIPtempPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B4SOIPtempPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B4SOItempNode != 0) && (here->B4SOIeNode != 0))
            {
                while (here->B4SOITempePtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B4SOITempePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B4SOIbNode != 0) && (here->B4SOIpNode != 0))
            {
                while (here->B4SOIBpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B4SOIBpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B4SOIpNode != 0) && (here->B4SOIbNode != 0))
            {
                while (here->B4SOIPbPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B4SOIPbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B4SOIpNode != 0) && (here->B4SOIpNode != 0))
            {
                while (here->B4SOIPpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B4SOIPpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B4SOIpNode != 0) && (here->B4SOIgNode != 0))
            {
                while (here->B4SOIPgPtr  != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B4SOIPgPtr  = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B4SOIgNode != 0) && (here->B4SOIpNode != 0))
            {
                while (here->B4SOIGpPtr  != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B4SOIGpPtr  = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B4SOIgNodeExt != 0) && (here->B4SOIgNodeExt != 0))
            {
                while (here->B4SOIGEgePtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B4SOIGEgePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B4SOIgNodeExt != 0) && (here->B4SOIgNode != 0))
            {
                while (here->B4SOIGEgPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B4SOIGEgPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B4SOIgNode != 0) && (here->B4SOIgNodeExt != 0))
            {
                while (here->B4SOIGgePtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B4SOIGgePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B4SOIgNodeExt != 0) && (here->B4SOIdNodePrime != 0))
            {
                while (here->B4SOIGEdpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B4SOIGEdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B4SOIgNodeExt != 0) && (here->B4SOIsNodePrime != 0))
            {
                while (here->B4SOIGEspPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B4SOIGEspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B4SOIgNodeExt != 0) && (here->B4SOIbNode != 0))
            {
                while (here->B4SOIGEbPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B4SOIGEbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B4SOIgNodeMid != 0) && (here->B4SOIdNodePrime != 0))
            {
                while (here->B4SOIGMdpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B4SOIGMdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B4SOIgNodeMid != 0) && (here->B4SOIgNode != 0))
            {
                while (here->B4SOIGMgPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B4SOIGMgPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B4SOIgNodeMid != 0) && (here->B4SOIgNodeMid != 0))
            {
                while (here->B4SOIGMgmPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B4SOIGMgmPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B4SOIgNodeMid != 0) && (here->B4SOIgNodeExt != 0))
            {
                while (here->B4SOIGMgePtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B4SOIGMgePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B4SOIgNodeMid != 0) && (here->B4SOIsNodePrime != 0))
            {
                while (here->B4SOIGMspPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B4SOIGMspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B4SOIgNodeMid != 0) && (here->B4SOIbNode != 0))
            {
                while (here->B4SOIGMbPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B4SOIGMbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B4SOIgNodeMid != 0) && (here->B4SOIeNode != 0))
            {
                while (here->B4SOIGMePtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B4SOIGMePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B4SOIdNodePrime != 0) && (here->B4SOIgNodeMid != 0))
            {
                while (here->B4SOIDPgmPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B4SOIDPgmPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B4SOIgNode != 0) && (here->B4SOIgNodeMid != 0))
            {
                while (here->B4SOIGgmPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B4SOIGgmPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B4SOIgNodeExt != 0) && (here->B4SOIgNodeMid != 0))
            {
                while (here->B4SOIGEgmPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B4SOIGEgmPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B4SOIsNodePrime != 0) && (here->B4SOIgNodeMid != 0))
            {
                while (here->B4SOISPgmPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B4SOISPgmPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B4SOIeNode != 0) && (here->B4SOIgNodeMid != 0))
            {
                while (here->B4SOIEgmPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B4SOIEgmPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B4SOIeNode != 0) && (here->B4SOIbNode != 0))
            {
                while (here->B4SOIEbPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B4SOIEbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B4SOIgNode != 0) && (here->B4SOIbNode != 0))
            {
                while (here->B4SOIGbPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B4SOIGbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B4SOIdNodePrime != 0) && (here->B4SOIbNode != 0))
            {
                while (here->B4SOIDPbPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B4SOIDPbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B4SOIsNodePrime != 0) && (here->B4SOIbNode != 0))
            {
                while (here->B4SOISPbPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B4SOISPbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B4SOIbNode != 0) && (here->B4SOIeNode != 0))
            {
                while (here->B4SOIBePtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B4SOIBePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B4SOIbNode != 0) && (here->B4SOIgNode != 0))
            {
                while (here->B4SOIBgPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B4SOIBgPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B4SOIbNode != 0) && (here->B4SOIdNodePrime != 0))
            {
                while (here->B4SOIBdpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B4SOIBdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B4SOIbNode != 0) && (here->B4SOIsNodePrime != 0))
            {
                while (here->B4SOIBspPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B4SOIBspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B4SOIbNode != 0) && (here->B4SOIbNode != 0))
            {
                while (here->B4SOIBbPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B4SOIBbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B4SOIeNode != 0) && (here->B4SOIgNode != 0))
            {
                while (here->B4SOIEgPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B4SOIEgPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B4SOIeNode != 0) && (here->B4SOIdNodePrime != 0))
            {
                while (here->B4SOIEdpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B4SOIEdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B4SOIeNode != 0) && (here->B4SOIsNodePrime != 0))
            {
                while (here->B4SOIEspPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B4SOIEspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B4SOIgNode != 0) && (here->B4SOIeNode != 0))
            {
                while (here->B4SOIGePtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B4SOIGePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B4SOIdNodePrime != 0) && (here->B4SOIeNode != 0))
            {
                while (here->B4SOIDPePtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B4SOIDPePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B4SOIsNodePrime != 0) && (here->B4SOIeNode != 0))
            {
                while (here->B4SOISPePtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B4SOISPePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B4SOIeNode != 0) && (here->B4SOIbNode != 0))
            {
                while (here->B4SOIEbPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B4SOIEbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B4SOIeNode != 0) && (here->B4SOIeNode != 0))
            {
                while (here->B4SOIEePtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B4SOIEePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B4SOIgNode != 0) && (here->B4SOIgNode != 0))
            {
                while (here->B4SOIGgPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B4SOIGgPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B4SOIgNode != 0) && (here->B4SOIdNodePrime != 0))
            {
                while (here->B4SOIGdpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B4SOIGdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B4SOIgNode != 0) && (here->B4SOIsNodePrime != 0))
            {
                while (here->B4SOIGspPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B4SOIGspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B4SOIdNodePrime != 0) && (here->B4SOIgNode != 0))
            {
                while (here->B4SOIDPgPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B4SOIDPgPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B4SOIdNodePrime != 0) && (here->B4SOIdNodePrime != 0))
            {
                while (here->B4SOIDPdpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B4SOIDPdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B4SOIdNodePrime != 0) && (here->B4SOIsNodePrime != 0))
            {
                while (here->B4SOIDPspPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B4SOIDPspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B4SOIdNodePrime != 0) && (here->B4SOIdNode != 0))
            {
                while (here->B4SOIDPdPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B4SOIDPdPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B4SOIsNodePrime != 0) && (here->B4SOIgNode != 0))
            {
                while (here->B4SOISPgPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B4SOISPgPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B4SOIsNodePrime != 0) && (here->B4SOIdNodePrime != 0))
            {
                while (here->B4SOISPdpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B4SOISPdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B4SOIsNodePrime != 0) && (here->B4SOIsNodePrime != 0))
            {
                while (here->B4SOISPspPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B4SOISPspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B4SOIsNodePrime != 0) && (here->B4SOIsNode != 0))
            {
                while (here->B4SOISPsPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B4SOISPsPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B4SOIdNode != 0) && (here->B4SOIdNode != 0))
            {
                while (here->B4SOIDdPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B4SOIDdPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B4SOIdNode != 0) && (here->B4SOIdNodePrime != 0))
            {
                while (here->B4SOIDdpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B4SOIDdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B4SOIsNode != 0) && (here->B4SOIsNode != 0))
            {
                while (here->B4SOISsPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B4SOISsPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B4SOIsNode != 0) && (here->B4SOIsNodePrime != 0))
            {
                while (here->B4SOISspPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B4SOISspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B4SOIdNodePrime != 0) && (here->B4SOIdbNode != 0))
            {
                while (here->B4SOIDPdbPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B4SOIDPdbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B4SOIsNodePrime != 0) && (here->B4SOIsbNode != 0))
            {
                while (here->B4SOISPsbPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B4SOISPsbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B4SOIdbNode != 0) && (here->B4SOIdNodePrime != 0))
            {
                while (here->B4SOIDBdpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B4SOIDBdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B4SOIdbNode != 0) && (here->B4SOIdbNode != 0))
            {
                while (here->B4SOIDBdbPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B4SOIDBdbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B4SOIdbNode != 0) && (here->B4SOIbNode != 0))
            {
                while (here->B4SOIDBbPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B4SOIDBbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B4SOIsbNode != 0) && (here->B4SOIsNodePrime != 0))
            {
                while (here->B4SOISBspPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B4SOISBspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B4SOIsbNode != 0) && (here->B4SOIsbNode != 0))
            {
                while (here->B4SOISBsbPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B4SOISBsbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B4SOIsbNode != 0) && (here->B4SOIbNode != 0))
            {
                while (here->B4SOISBbPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B4SOISBbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B4SOIbNode != 0) && (here->B4SOIdbNode != 0))
            {
                while (here->B4SOIBdbPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B4SOIBdbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B4SOIbNode != 0) && (here->B4SOIsbNode != 0))
            {
                while (here->B4SOIBsbPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B4SOIBsbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B4SOIdNode != 0) && (here->B4SOIgNode != 0))
            {
                while (here->B4SOIDgPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B4SOIDgPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B4SOIdNode != 0) && (here->B4SOIsNodePrime != 0))
            {
                while (here->B4SOIDspPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B4SOIDspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B4SOIsNode != 0) && (here->B4SOIdNodePrime != 0))
            {
                while (here->B4SOISdpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B4SOISdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B4SOIsNode != 0) && (here->B4SOIgNode != 0))
            {
                while (here->B4SOISgPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B4SOISgPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B4SOIdNode != 0) && (here->B4SOIbNode != 0))
            {
                while (here->B4SOIDbPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B4SOIDbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B4SOIsNode != 0) && (here->B4SOIbNode != 0))
            {
                while (here->B4SOISbPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B4SOISbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B4SOIvbsNode != 0) && (here->B4SOIvbsNode != 0))
            {
                while (here->B4SOIVbsPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B4SOIVbsPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B4SOIidsNode != 0) && (here->B4SOIidsNode != 0))
            {
                while (here->B4SOIIdsPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B4SOIIdsPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B4SOIicNode != 0) && (here->B4SOIicNode != 0))
            {
                while (here->B4SOIIcPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B4SOIIcPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B4SOIibsNode != 0) && (here->B4SOIibsNode != 0))
            {
                while (here->B4SOIIbsPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B4SOIIbsPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B4SOIibdNode != 0) && (here->B4SOIibdNode != 0))
            {
                while (here->B4SOIIbdPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B4SOIIbdPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B4SOIiiiNode != 0) && (here->B4SOIiiiNode != 0))
            {
                while (here->B4SOIIiiPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B4SOIIiiPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B4SOIigNode != 0) && (here->B4SOIigNode != 0))
            {
                while (here->B4SOIIgPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B4SOIIgPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B4SOIgiggNode != 0) && (here->B4SOIgiggNode != 0))
            {
                while (here->B4SOIGiggPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B4SOIGiggPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B4SOIgigdNode != 0) && (here->B4SOIgigdNode != 0))
            {
                while (here->B4SOIGigdPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B4SOIGigdPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B4SOIgigbNode != 0) && (here->B4SOIgigbNode != 0))
            {
                while (here->B4SOIGigbPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B4SOIGigbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B4SOIigidlNode != 0) && (here->B4SOIigidlNode != 0))
            {
                while (here->B4SOIIgidlPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B4SOIIgidlPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B4SOIitunNode != 0) && (here->B4SOIitunNode != 0))
            {
                while (here->B4SOIItunPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B4SOIItunPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B4SOIibpNode != 0) && (here->B4SOIibpNode != 0))
            {
                while (here->B4SOIIbpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B4SOIIbpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B4SOIcbbNode != 0) && (here->B4SOIcbbNode != 0))
            {
                while (here->B4SOICbbPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B4SOICbbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B4SOIcbdNode != 0) && (here->B4SOIcbdNode != 0))
            {
                while (here->B4SOICbdPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B4SOICbdPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B4SOIcbgNode != 0) && (here->B4SOIcbgNode != 0))
            {
                while (here->B4SOICbgPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B4SOICbgPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B4SOIqbfNode != 0) && (here->B4SOIqbfNode != 0))
            {
                while (here->B4SOIQbfPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B4SOIQbfPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B4SOIqjsNode != 0) && (here->B4SOIqjsNode != 0))
            {
                while (here->B4SOIQjsPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B4SOIQjsPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B4SOIqjdNode != 0) && (here->B4SOIqjdNode != 0))
            {
                while (here->B4SOIQjdPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B4SOIQjdPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }
        }
    }

    return (OK) ;
}