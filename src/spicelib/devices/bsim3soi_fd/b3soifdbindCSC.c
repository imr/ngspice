/**********
Author: 2012 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "b3soifddef.h"
#include "ngspice/sperror.h"

int
B3SOIFDbindCSC(GENmodel *inModel, CKTcircuit *ckt)
{
    B3SOIFDmodel *model = (B3SOIFDmodel *)inModel;
    int i ;

    /*  loop through all the b3soifd models */
    for( ; model != NULL; model = model->B3SOIFDnextModel ) {
	B3SOIFDinstance *here;

        /* loop through all the instances of the model */
        for (here = model->B3SOIFDinstances; here != NULL ;
	    here = here->B3SOIFDnextInstance) {

                if ((model->B3SOIFDshMod == 1) && (here->B3SOIFDrth0!=0.0)) {

		i = 0 ;
		if ((here-> B3SOIFDtempNode != 0) && (here-> B3SOIFDtempNode != 0)) {
			while (here->B3SOIFDTemptempPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIFDTemptempPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDtempNode != 0) && (here-> B3SOIFDdNodePrime != 0)) {
			while (here->B3SOIFDTempdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIFDTempdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDtempNode != 0) && (here-> B3SOIFDsNodePrime != 0)) {
			while (here->B3SOIFDTempspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIFDTempspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDtempNode != 0) && (here-> B3SOIFDgNode != 0)) {
			while (here->B3SOIFDTempgPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIFDTempgPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDtempNode != 0) && (here-> B3SOIFDbNode != 0)) {
			while (here->B3SOIFDTempbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIFDTempbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDtempNode != 0) && (here-> B3SOIFDeNode != 0)) {
			while (here->B3SOIFDTempePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIFDTempePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDgNode != 0) && (here-> B3SOIFDtempNode != 0)) {
			while (here->B3SOIFDGtempPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIFDGtempPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDdNodePrime != 0) && (here-> B3SOIFDtempNode != 0)) {
			while (here->B3SOIFDDPtempPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIFDDPtempPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDsNodePrime != 0) && (here-> B3SOIFDtempNode != 0)) {
			while (here->B3SOIFDSPtempPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIFDSPtempPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDeNode != 0) && (here-> B3SOIFDtempNode != 0)) {
			while (here->B3SOIFDEtempPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIFDEtempPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDbNode != 0) && (here-> B3SOIFDtempNode != 0)) {
			while (here->B3SOIFDBtempPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIFDBtempPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

                if (here->B3SOIFDbodyMod == 1) {

		i = 0 ;
		if ((here-> B3SOIFDpNode != 0) && (here-> B3SOIFDtempNode != 0)) {
			while (here->B3SOIFDPtempPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIFDPtempPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
                } /* IF */
                } /* IF */

                if (here->B3SOIFDbodyMod == 2) {
                }
                else if (here->B3SOIFDbodyMod == 1) {

		i = 0 ;
		if ((here-> B3SOIFDbNode != 0) && (here-> B3SOIFDpNode != 0)) {
			while (here->B3SOIFDBpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIFDBpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDpNode != 0) && (here-> B3SOIFDbNode != 0)) {
			while (here->B3SOIFDPbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIFDPbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDpNode != 0) && (here-> B3SOIFDpNode != 0)) {
			while (here->B3SOIFDPpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIFDPpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDpNode != 0) && (here-> B3SOIFDgNode != 0)) {
			while (here->B3SOIFDPgPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIFDPgPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDpNode != 0) && (here-> B3SOIFDdNodePrime != 0)) {
			while (here->B3SOIFDPdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIFDPdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDpNode != 0) && (here-> B3SOIFDsNodePrime != 0)) {
			while (here->B3SOIFDPspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIFDPspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDpNode != 0) && (here-> B3SOIFDeNode != 0)) {
			while (here->B3SOIFDPePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIFDPePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
                } /* ELSE */

		i = 0 ;
		if ((here-> B3SOIFDeNode != 0) && (here-> B3SOIFDgNode != 0)) {
			while (here->B3SOIFDEgPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIFDEgPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDeNode != 0) && (here-> B3SOIFDdNodePrime != 0)) {
			while (here->B3SOIFDEdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIFDEdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDeNode != 0) && (here-> B3SOIFDsNodePrime != 0)) {
			while (here->B3SOIFDEspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIFDEspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDgNode != 0) && (here-> B3SOIFDeNode != 0)) {
			while (here->B3SOIFDGePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIFDGePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDdNodePrime != 0) && (here-> B3SOIFDeNode != 0)) {
			while (here->B3SOIFDDPePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIFDDPePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDsNodePrime != 0) && (here-> B3SOIFDeNode != 0)) {
			while (here->B3SOIFDSPePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIFDSPePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDeNode != 0) && (here-> B3SOIFDbNode != 0)) {
			while (here->B3SOIFDEbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIFDEbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDeNode != 0) && (here-> B3SOIFDeNode != 0)) {
			while (here->B3SOIFDEePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIFDEePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDgNode != 0) && (here-> B3SOIFDgNode != 0)) {
			while (here->B3SOIFDGgPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIFDGgPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDgNode != 0) && (here-> B3SOIFDdNodePrime != 0)) {
			while (here->B3SOIFDGdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIFDGdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDgNode != 0) && (here-> B3SOIFDsNodePrime != 0)) {
			while (here->B3SOIFDGspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIFDGspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDdNodePrime != 0) && (here-> B3SOIFDgNode != 0)) {
			while (here->B3SOIFDDPgPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIFDDPgPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDdNodePrime != 0) && (here-> B3SOIFDdNodePrime != 0)) {
			while (here->B3SOIFDDPdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIFDDPdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDdNodePrime != 0) && (here-> B3SOIFDsNodePrime != 0)) {
			while (here->B3SOIFDDPspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIFDDPspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDdNodePrime != 0) && (here-> B3SOIFDdNode != 0)) {
			while (here->B3SOIFDDPdPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIFDDPdPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDsNodePrime != 0) && (here-> B3SOIFDgNode != 0)) {
			while (here->B3SOIFDSPgPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIFDSPgPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDsNodePrime != 0) && (here-> B3SOIFDdNodePrime != 0)) {
			while (here->B3SOIFDSPdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIFDSPdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDsNodePrime != 0) && (here-> B3SOIFDsNodePrime != 0)) {
			while (here->B3SOIFDSPspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIFDSPspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDsNodePrime != 0) && (here-> B3SOIFDsNode != 0)) {
			while (here->B3SOIFDSPsPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIFDSPsPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDdNode != 0) && (here-> B3SOIFDdNode != 0)) {
			while (here->B3SOIFDDdPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIFDDdPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDdNode != 0) && (here-> B3SOIFDdNodePrime != 0)) {
			while (here->B3SOIFDDdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIFDDdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDsNode != 0) && (here-> B3SOIFDsNode != 0)) {
			while (here->B3SOIFDSsPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIFDSsPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDsNode != 0) && (here-> B3SOIFDsNodePrime != 0)) {
			while (here->B3SOIFDSspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIFDSspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

                if ((here->B3SOIFDdebugMod > 1) || (here->B3SOIFDdebugMod == -1)) {

		i = 0 ;
		if ((here-> B3SOIFDvbsNode != 0) && (here-> B3SOIFDvbsNode != 0)) {
			while (here->B3SOIFDVbsPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIFDVbsPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDidsNode != 0) && (here-> B3SOIFDidsNode != 0)) {
			while (here->B3SOIFDIdsPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIFDIdsPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDicNode != 0) && (here-> B3SOIFDicNode != 0)) {
			while (here->B3SOIFDIcPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIFDIcPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDibsNode != 0) && (here-> B3SOIFDibsNode != 0)) {
			while (here->B3SOIFDIbsPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIFDIbsPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDibdNode != 0) && (here-> B3SOIFDibdNode != 0)) {
			while (here->B3SOIFDIbdPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIFDIbdPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDiiiNode != 0) && (here-> B3SOIFDiiiNode != 0)) {
			while (here->B3SOIFDIiiPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIFDIiiPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDigidlNode != 0) && (here-> B3SOIFDigidlNode != 0)) {
			while (here->B3SOIFDIgidlPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIFDIgidlPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDitunNode != 0) && (here-> B3SOIFDitunNode != 0)) {
			while (here->B3SOIFDItunPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIFDItunPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDibpNode != 0) && (here-> B3SOIFDibpNode != 0)) {
			while (here->B3SOIFDIbpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIFDIbpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDabeffNode != 0) && (here-> B3SOIFDabeffNode != 0)) {
			while (here->B3SOIFDAbeffPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIFDAbeffPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDvbs0effNode != 0) && (here-> B3SOIFDvbs0effNode != 0)) {
			while (here->B3SOIFDVbs0effPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIFDVbs0effPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDvbseffNode != 0) && (here-> B3SOIFDvbseffNode != 0)) {
			while (here->B3SOIFDVbseffPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIFDVbseffPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDxcNode != 0) && (here-> B3SOIFDxcNode != 0)) {
			while (here->B3SOIFDXcPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIFDXcPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDcbbNode != 0) && (here-> B3SOIFDcbbNode != 0)) {
			while (here->B3SOIFDCbbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIFDCbbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDcbdNode != 0) && (here-> B3SOIFDcbdNode != 0)) {
			while (here->B3SOIFDCbdPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIFDCbdPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDcbgNode != 0) && (here-> B3SOIFDcbgNode != 0)) {
			while (here->B3SOIFDCbgPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIFDCbgPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDqbNode != 0) && (here-> B3SOIFDqbNode != 0)) {
			while (here->B3SOIFDqbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIFDqbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDqbfNode != 0) && (here-> B3SOIFDqbfNode != 0)) {
			while (here->B3SOIFDQbfPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIFDQbfPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDqjsNode != 0) && (here-> B3SOIFDqjsNode != 0)) {
			while (here->B3SOIFDQjsPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIFDQjsPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDqjdNode != 0) && (here-> B3SOIFDqjdNode != 0)) {
			while (here->B3SOIFDQjdPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIFDQjdPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDgmNode != 0) && (here-> B3SOIFDgmNode != 0)) {
			while (here->B3SOIFDGmPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIFDGmPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDgmbsNode != 0) && (here-> B3SOIFDgmbsNode != 0)) {
			while (here->B3SOIFDGmbsPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIFDGmbsPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDgdsNode != 0) && (here-> B3SOIFDgdsNode != 0)) {
			while (here->B3SOIFDGdsPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIFDGdsPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDgmeNode != 0) && (here-> B3SOIFDgmeNode != 0)) {
			while (here->B3SOIFDGmePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIFDGmePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDvbs0teffNode != 0) && (here-> B3SOIFDvbs0teffNode != 0)) {
			while (here->B3SOIFDVbs0teffPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIFDVbs0teffPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDvthNode != 0) && (here-> B3SOIFDvthNode != 0)) {
			while (here->B3SOIFDVthPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIFDVthPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDvgsteffNode != 0) && (here-> B3SOIFDvgsteffNode != 0)) {
			while (here->B3SOIFDVgsteffPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIFDVgsteffPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDxcsatNode != 0) && (here-> B3SOIFDxcsatNode != 0)) {
			while (here->B3SOIFDXcsatPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIFDXcsatPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDvcscvNode != 0) && (here-> B3SOIFDvcscvNode != 0)) {
			while (here->B3SOIFDVcscvPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIFDVcscvPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDvdscvNode != 0) && (here-> B3SOIFDvdscvNode != 0)) {
			while (here->B3SOIFDVdscvPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIFDVdscvPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDcbeNode != 0) && (here-> B3SOIFDcbeNode != 0)) {
			while (here->B3SOIFDCbePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIFDCbePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDdum1Node != 0) && (here-> B3SOIFDdum1Node != 0)) {
			while (here->B3SOIFDDum1Ptr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIFDDum1Ptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDdum2Node != 0) && (here-> B3SOIFDdum2Node != 0)) {
			while (here->B3SOIFDDum2Ptr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIFDDum2Ptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDdum3Node != 0) && (here-> B3SOIFDdum3Node != 0)) {
			while (here->B3SOIFDDum3Ptr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIFDDum3Ptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDdum4Node != 0) && (here-> B3SOIFDdum4Node != 0)) {
			while (here->B3SOIFDDum4Ptr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIFDDum4Ptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDdum5Node != 0) && (here-> B3SOIFDdum5Node != 0)) {
			while (here->B3SOIFDDum5Ptr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIFDDum5Ptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDqaccNode != 0) && (here-> B3SOIFDqaccNode != 0)) {
			while (here->B3SOIFDQaccPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIFDQaccPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDqsub0Node != 0) && (here-> B3SOIFDqsub0Node != 0)) {
			while (here->B3SOIFDQsub0Ptr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIFDQsub0Ptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDqsubs1Node != 0) && (here-> B3SOIFDqsubs1Node != 0)) {
			while (here->B3SOIFDQsubs1Ptr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIFDQsubs1Ptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDqsubs2Node != 0) && (here-> B3SOIFDqsubs2Node != 0)) {
			while (here->B3SOIFDQsubs2Ptr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIFDQsubs2Ptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDqeNode != 0) && (here-> B3SOIFDqeNode != 0)) {
			while (here->B3SOIFDqePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIFDqePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDqdNode != 0) && (here-> B3SOIFDqdNode != 0)) {
			while (here->B3SOIFDqdPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIFDqdPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDqgNode != 0) && (here-> B3SOIFDqgNode != 0)) {
			while (here->B3SOIFDqgPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIFDqgPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
                } /* IF */
	}
    }
    return(OK);
}

int
B3SOIFDbindCSCComplex(GENmodel *inModel, CKTcircuit *ckt)
{
    B3SOIFDmodel *model = (B3SOIFDmodel *)inModel;
    int i ;

    /*  loop through all the b3soifd models */
    for( ; model != NULL; model = model->B3SOIFDnextModel ) {
	B3SOIFDinstance *here;

        /* loop through all the instances of the model */
        for (here = model->B3SOIFDinstances; here != NULL ;
	    here = here->B3SOIFDnextInstance) {

                if ((model->B3SOIFDshMod == 1) && (here->B3SOIFDrth0!=0.0)) {

		i = 0 ;
		if ((here-> B3SOIFDtempNode != 0) && (here-> B3SOIFDtempNode != 0)) {
			while (here->B3SOIFDTemptempPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIFDTemptempPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDtempNode != 0) && (here-> B3SOIFDdNodePrime != 0)) {
			while (here->B3SOIFDTempdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIFDTempdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDtempNode != 0) && (here-> B3SOIFDsNodePrime != 0)) {
			while (here->B3SOIFDTempspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIFDTempspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDtempNode != 0) && (here-> B3SOIFDgNode != 0)) {
			while (here->B3SOIFDTempgPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIFDTempgPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDtempNode != 0) && (here-> B3SOIFDbNode != 0)) {
			while (here->B3SOIFDTempbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIFDTempbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDtempNode != 0) && (here-> B3SOIFDeNode != 0)) {
			while (here->B3SOIFDTempePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIFDTempePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDgNode != 0) && (here-> B3SOIFDtempNode != 0)) {
			while (here->B3SOIFDGtempPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIFDGtempPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDdNodePrime != 0) && (here-> B3SOIFDtempNode != 0)) {
			while (here->B3SOIFDDPtempPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIFDDPtempPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDsNodePrime != 0) && (here-> B3SOIFDtempNode != 0)) {
			while (here->B3SOIFDSPtempPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIFDSPtempPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDeNode != 0) && (here-> B3SOIFDtempNode != 0)) {
			while (here->B3SOIFDEtempPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIFDEtempPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDbNode != 0) && (here-> B3SOIFDtempNode != 0)) {
			while (here->B3SOIFDBtempPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIFDBtempPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

                if (here->B3SOIFDbodyMod == 1) {

		i = 0 ;
		if ((here-> B3SOIFDpNode != 0) && (here-> B3SOIFDtempNode != 0)) {
			while (here->B3SOIFDPtempPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIFDPtempPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
                } /* IF */
                } /* IF */

                if (here->B3SOIFDbodyMod == 2) {
                }
                else if (here->B3SOIFDbodyMod == 1) {

		i = 0 ;
		if ((here-> B3SOIFDbNode != 0) && (here-> B3SOIFDpNode != 0)) {
			while (here->B3SOIFDBpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIFDBpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDpNode != 0) && (here-> B3SOIFDbNode != 0)) {
			while (here->B3SOIFDPbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIFDPbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDpNode != 0) && (here-> B3SOIFDpNode != 0)) {
			while (here->B3SOIFDPpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIFDPpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDpNode != 0) && (here-> B3SOIFDgNode != 0)) {
			while (here->B3SOIFDPgPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIFDPgPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDpNode != 0) && (here-> B3SOIFDdNodePrime != 0)) {
			while (here->B3SOIFDPdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIFDPdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDpNode != 0) && (here-> B3SOIFDsNodePrime != 0)) {
			while (here->B3SOIFDPspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIFDPspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDpNode != 0) && (here-> B3SOIFDeNode != 0)) {
			while (here->B3SOIFDPePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIFDPePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
                } /* ELSE */

		i = 0 ;
		if ((here-> B3SOIFDeNode != 0) && (here-> B3SOIFDgNode != 0)) {
			while (here->B3SOIFDEgPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIFDEgPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDeNode != 0) && (here-> B3SOIFDdNodePrime != 0)) {
			while (here->B3SOIFDEdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIFDEdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDeNode != 0) && (here-> B3SOIFDsNodePrime != 0)) {
			while (here->B3SOIFDEspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIFDEspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDgNode != 0) && (here-> B3SOIFDeNode != 0)) {
			while (here->B3SOIFDGePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIFDGePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDdNodePrime != 0) && (here-> B3SOIFDeNode != 0)) {
			while (here->B3SOIFDDPePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIFDDPePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDsNodePrime != 0) && (here-> B3SOIFDeNode != 0)) {
			while (here->B3SOIFDSPePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIFDSPePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDeNode != 0) && (here-> B3SOIFDbNode != 0)) {
			while (here->B3SOIFDEbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIFDEbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDeNode != 0) && (here-> B3SOIFDeNode != 0)) {
			while (here->B3SOIFDEePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIFDEePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDgNode != 0) && (here-> B3SOIFDgNode != 0)) {
			while (here->B3SOIFDGgPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIFDGgPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDgNode != 0) && (here-> B3SOIFDdNodePrime != 0)) {
			while (here->B3SOIFDGdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIFDGdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDgNode != 0) && (here-> B3SOIFDsNodePrime != 0)) {
			while (here->B3SOIFDGspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIFDGspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDdNodePrime != 0) && (here-> B3SOIFDgNode != 0)) {
			while (here->B3SOIFDDPgPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIFDDPgPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDdNodePrime != 0) && (here-> B3SOIFDdNodePrime != 0)) {
			while (here->B3SOIFDDPdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIFDDPdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDdNodePrime != 0) && (here-> B3SOIFDsNodePrime != 0)) {
			while (here->B3SOIFDDPspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIFDDPspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDdNodePrime != 0) && (here-> B3SOIFDdNode != 0)) {
			while (here->B3SOIFDDPdPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIFDDPdPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDsNodePrime != 0) && (here-> B3SOIFDgNode != 0)) {
			while (here->B3SOIFDSPgPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIFDSPgPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDsNodePrime != 0) && (here-> B3SOIFDdNodePrime != 0)) {
			while (here->B3SOIFDSPdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIFDSPdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDsNodePrime != 0) && (here-> B3SOIFDsNodePrime != 0)) {
			while (here->B3SOIFDSPspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIFDSPspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDsNodePrime != 0) && (here-> B3SOIFDsNode != 0)) {
			while (here->B3SOIFDSPsPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIFDSPsPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDdNode != 0) && (here-> B3SOIFDdNode != 0)) {
			while (here->B3SOIFDDdPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIFDDdPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDdNode != 0) && (here-> B3SOIFDdNodePrime != 0)) {
			while (here->B3SOIFDDdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIFDDdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDsNode != 0) && (here-> B3SOIFDsNode != 0)) {
			while (here->B3SOIFDSsPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIFDSsPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDsNode != 0) && (here-> B3SOIFDsNodePrime != 0)) {
			while (here->B3SOIFDSspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIFDSspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

                if ((here->B3SOIFDdebugMod > 1) || (here->B3SOIFDdebugMod == -1)) {

		i = 0 ;
		if ((here-> B3SOIFDvbsNode != 0) && (here-> B3SOIFDvbsNode != 0)) {
			while (here->B3SOIFDVbsPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIFDVbsPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDidsNode != 0) && (here-> B3SOIFDidsNode != 0)) {
			while (here->B3SOIFDIdsPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIFDIdsPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDicNode != 0) && (here-> B3SOIFDicNode != 0)) {
			while (here->B3SOIFDIcPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIFDIcPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDibsNode != 0) && (here-> B3SOIFDibsNode != 0)) {
			while (here->B3SOIFDIbsPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIFDIbsPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDibdNode != 0) && (here-> B3SOIFDibdNode != 0)) {
			while (here->B3SOIFDIbdPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIFDIbdPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDiiiNode != 0) && (here-> B3SOIFDiiiNode != 0)) {
			while (here->B3SOIFDIiiPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIFDIiiPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDigidlNode != 0) && (here-> B3SOIFDigidlNode != 0)) {
			while (here->B3SOIFDIgidlPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIFDIgidlPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDitunNode != 0) && (here-> B3SOIFDitunNode != 0)) {
			while (here->B3SOIFDItunPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIFDItunPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDibpNode != 0) && (here-> B3SOIFDibpNode != 0)) {
			while (here->B3SOIFDIbpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIFDIbpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDabeffNode != 0) && (here-> B3SOIFDabeffNode != 0)) {
			while (here->B3SOIFDAbeffPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIFDAbeffPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDvbs0effNode != 0) && (here-> B3SOIFDvbs0effNode != 0)) {
			while (here->B3SOIFDVbs0effPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIFDVbs0effPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDvbseffNode != 0) && (here-> B3SOIFDvbseffNode != 0)) {
			while (here->B3SOIFDVbseffPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIFDVbseffPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDxcNode != 0) && (here-> B3SOIFDxcNode != 0)) {
			while (here->B3SOIFDXcPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIFDXcPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDcbbNode != 0) && (here-> B3SOIFDcbbNode != 0)) {
			while (here->B3SOIFDCbbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIFDCbbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDcbdNode != 0) && (here-> B3SOIFDcbdNode != 0)) {
			while (here->B3SOIFDCbdPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIFDCbdPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDcbgNode != 0) && (here-> B3SOIFDcbgNode != 0)) {
			while (here->B3SOIFDCbgPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIFDCbgPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDqbNode != 0) && (here-> B3SOIFDqbNode != 0)) {
			while (here->B3SOIFDqbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIFDqbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDqbfNode != 0) && (here-> B3SOIFDqbfNode != 0)) {
			while (here->B3SOIFDQbfPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIFDQbfPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDqjsNode != 0) && (here-> B3SOIFDqjsNode != 0)) {
			while (here->B3SOIFDQjsPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIFDQjsPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDqjdNode != 0) && (here-> B3SOIFDqjdNode != 0)) {
			while (here->B3SOIFDQjdPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIFDQjdPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDgmNode != 0) && (here-> B3SOIFDgmNode != 0)) {
			while (here->B3SOIFDGmPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIFDGmPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDgmbsNode != 0) && (here-> B3SOIFDgmbsNode != 0)) {
			while (here->B3SOIFDGmbsPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIFDGmbsPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDgdsNode != 0) && (here-> B3SOIFDgdsNode != 0)) {
			while (here->B3SOIFDGdsPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIFDGdsPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDgmeNode != 0) && (here-> B3SOIFDgmeNode != 0)) {
			while (here->B3SOIFDGmePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIFDGmePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDvbs0teffNode != 0) && (here-> B3SOIFDvbs0teffNode != 0)) {
			while (here->B3SOIFDVbs0teffPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIFDVbs0teffPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDvthNode != 0) && (here-> B3SOIFDvthNode != 0)) {
			while (here->B3SOIFDVthPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIFDVthPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDvgsteffNode != 0) && (here-> B3SOIFDvgsteffNode != 0)) {
			while (here->B3SOIFDVgsteffPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIFDVgsteffPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDxcsatNode != 0) && (here-> B3SOIFDxcsatNode != 0)) {
			while (here->B3SOIFDXcsatPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIFDXcsatPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDvcscvNode != 0) && (here-> B3SOIFDvcscvNode != 0)) {
			while (here->B3SOIFDVcscvPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIFDVcscvPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDvdscvNode != 0) && (here-> B3SOIFDvdscvNode != 0)) {
			while (here->B3SOIFDVdscvPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIFDVdscvPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDcbeNode != 0) && (here-> B3SOIFDcbeNode != 0)) {
			while (here->B3SOIFDCbePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIFDCbePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDdum1Node != 0) && (here-> B3SOIFDdum1Node != 0)) {
			while (here->B3SOIFDDum1Ptr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIFDDum1Ptr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDdum2Node != 0) && (here-> B3SOIFDdum2Node != 0)) {
			while (here->B3SOIFDDum2Ptr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIFDDum2Ptr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDdum3Node != 0) && (here-> B3SOIFDdum3Node != 0)) {
			while (here->B3SOIFDDum3Ptr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIFDDum3Ptr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDdum4Node != 0) && (here-> B3SOIFDdum4Node != 0)) {
			while (here->B3SOIFDDum4Ptr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIFDDum4Ptr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDdum5Node != 0) && (here-> B3SOIFDdum5Node != 0)) {
			while (here->B3SOIFDDum5Ptr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIFDDum5Ptr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDqaccNode != 0) && (here-> B3SOIFDqaccNode != 0)) {
			while (here->B3SOIFDQaccPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIFDQaccPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDqsub0Node != 0) && (here-> B3SOIFDqsub0Node != 0)) {
			while (here->B3SOIFDQsub0Ptr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIFDQsub0Ptr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDqsubs1Node != 0) && (here-> B3SOIFDqsubs1Node != 0)) {
			while (here->B3SOIFDQsubs1Ptr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIFDQsubs1Ptr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDqsubs2Node != 0) && (here-> B3SOIFDqsubs2Node != 0)) {
			while (here->B3SOIFDQsubs2Ptr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIFDQsubs2Ptr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDqeNode != 0) && (here-> B3SOIFDqeNode != 0)) {
			while (here->B3SOIFDqePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIFDqePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDqdNode != 0) && (here-> B3SOIFDqdNode != 0)) {
			while (here->B3SOIFDqdPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIFDqdPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIFDqgNode != 0) && (here-> B3SOIFDqgNode != 0)) {
			while (here->B3SOIFDqgPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIFDqgPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
                } /* IF */
	}
    }
    return(OK);
}

int
B3SOIFDbindCSCComplexToReal (GENmodel *inModel, CKTcircuit *ckt)
{
    B3SOIFDmodel *model = (B3SOIFDmodel *)inModel ;
    B3SOIFDinstance *here ;
    int i ;

    /*  loop through all the bsim3SiliconOnInsulatorFullyDepleted models */
    for ( ; model != NULL ; model = model->B3SOIFDnextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->B3SOIFDinstances ; here != NULL ; here = here->B3SOIFDnextInstance)
        {
            i = 0 ;
            if ((here->B3SOIFDtempNode != 0) && (here->B3SOIFDtempNode != 0))
            {
                while (here->B3SOIFDTemptempPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B3SOIFDTemptempPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B3SOIFDtempNode != 0) && (here->B3SOIFDdNodePrime != 0))
            {
                while (here->B3SOIFDTempdpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B3SOIFDTempdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B3SOIFDtempNode != 0) && (here->B3SOIFDsNodePrime != 0))
            {
                while (here->B3SOIFDTempspPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B3SOIFDTempspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B3SOIFDtempNode != 0) && (here->B3SOIFDgNode != 0))
            {
                while (here->B3SOIFDTempgPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B3SOIFDTempgPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B3SOIFDtempNode != 0) && (here->B3SOIFDbNode != 0))
            {
                while (here->B3SOIFDTempbPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B3SOIFDTempbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B3SOIFDtempNode != 0) && (here->B3SOIFDeNode != 0))
            {
                while (here->B3SOIFDTempePtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B3SOIFDTempePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B3SOIFDgNode != 0) && (here->B3SOIFDtempNode != 0))
            {
                while (here->B3SOIFDGtempPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B3SOIFDGtempPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B3SOIFDdNodePrime != 0) && (here->B3SOIFDtempNode != 0))
            {
                while (here->B3SOIFDDPtempPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B3SOIFDDPtempPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B3SOIFDsNodePrime != 0) && (here->B3SOIFDtempNode != 0))
            {
                while (here->B3SOIFDSPtempPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B3SOIFDSPtempPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B3SOIFDeNode != 0) && (here->B3SOIFDtempNode != 0))
            {
                while (here->B3SOIFDEtempPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B3SOIFDEtempPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B3SOIFDbNode != 0) && (here->B3SOIFDtempNode != 0))
            {
                while (here->B3SOIFDBtempPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B3SOIFDBtempPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B3SOIFDpNode != 0) && (here->B3SOIFDtempNode != 0))
            {
                while (here->B3SOIFDPtempPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B3SOIFDPtempPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B3SOIFDbNode != 0) && (here->B3SOIFDpNode != 0))
            {
                while (here->B3SOIFDBpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B3SOIFDBpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B3SOIFDpNode != 0) && (here->B3SOIFDbNode != 0))
            {
                while (here->B3SOIFDPbPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B3SOIFDPbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B3SOIFDpNode != 0) && (here->B3SOIFDpNode != 0))
            {
                while (here->B3SOIFDPpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B3SOIFDPpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B3SOIFDpNode != 0) && (here->B3SOIFDgNode != 0))
            {
                while (here->B3SOIFDPgPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B3SOIFDPgPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B3SOIFDpNode != 0) && (here->B3SOIFDdNodePrime != 0))
            {
                while (here->B3SOIFDPdpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B3SOIFDPdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B3SOIFDpNode != 0) && (here->B3SOIFDsNodePrime != 0))
            {
                while (here->B3SOIFDPspPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B3SOIFDPspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B3SOIFDpNode != 0) && (here->B3SOIFDeNode != 0))
            {
                while (here->B3SOIFDPePtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B3SOIFDPePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B3SOIFDeNode != 0) && (here->B3SOIFDgNode != 0))
            {
                while (here->B3SOIFDEgPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B3SOIFDEgPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B3SOIFDeNode != 0) && (here->B3SOIFDdNodePrime != 0))
            {
                while (here->B3SOIFDEdpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B3SOIFDEdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B3SOIFDeNode != 0) && (here->B3SOIFDsNodePrime != 0))
            {
                while (here->B3SOIFDEspPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B3SOIFDEspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B3SOIFDgNode != 0) && (here->B3SOIFDeNode != 0))
            {
                while (here->B3SOIFDGePtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B3SOIFDGePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B3SOIFDdNodePrime != 0) && (here->B3SOIFDeNode != 0))
            {
                while (here->B3SOIFDDPePtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B3SOIFDDPePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B3SOIFDsNodePrime != 0) && (here->B3SOIFDeNode != 0))
            {
                while (here->B3SOIFDSPePtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B3SOIFDSPePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B3SOIFDeNode != 0) && (here->B3SOIFDbNode != 0))
            {
                while (here->B3SOIFDEbPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B3SOIFDEbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B3SOIFDeNode != 0) && (here->B3SOIFDeNode != 0))
            {
                while (here->B3SOIFDEePtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B3SOIFDEePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B3SOIFDgNode != 0) && (here->B3SOIFDgNode != 0))
            {
                while (here->B3SOIFDGgPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B3SOIFDGgPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B3SOIFDgNode != 0) && (here->B3SOIFDdNodePrime != 0))
            {
                while (here->B3SOIFDGdpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B3SOIFDGdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B3SOIFDgNode != 0) && (here->B3SOIFDsNodePrime != 0))
            {
                while (here->B3SOIFDGspPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B3SOIFDGspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B3SOIFDdNodePrime != 0) && (here->B3SOIFDgNode != 0))
            {
                while (here->B3SOIFDDPgPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B3SOIFDDPgPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B3SOIFDdNodePrime != 0) && (here->B3SOIFDdNodePrime != 0))
            {
                while (here->B3SOIFDDPdpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B3SOIFDDPdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B3SOIFDdNodePrime != 0) && (here->B3SOIFDsNodePrime != 0))
            {
                while (here->B3SOIFDDPspPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B3SOIFDDPspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B3SOIFDdNodePrime != 0) && (here->B3SOIFDdNode != 0))
            {
                while (here->B3SOIFDDPdPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B3SOIFDDPdPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B3SOIFDsNodePrime != 0) && (here->B3SOIFDgNode != 0))
            {
                while (here->B3SOIFDSPgPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B3SOIFDSPgPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B3SOIFDsNodePrime != 0) && (here->B3SOIFDdNodePrime != 0))
            {
                while (here->B3SOIFDSPdpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B3SOIFDSPdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B3SOIFDsNodePrime != 0) && (here->B3SOIFDsNodePrime != 0))
            {
                while (here->B3SOIFDSPspPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B3SOIFDSPspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B3SOIFDsNodePrime != 0) && (here->B3SOIFDsNode != 0))
            {
                while (here->B3SOIFDSPsPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B3SOIFDSPsPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B3SOIFDdNode != 0) && (here->B3SOIFDdNode != 0))
            {
                while (here->B3SOIFDDdPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B3SOIFDDdPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B3SOIFDdNode != 0) && (here->B3SOIFDdNodePrime != 0))
            {
                while (here->B3SOIFDDdpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B3SOIFDDdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B3SOIFDsNode != 0) && (here->B3SOIFDsNode != 0))
            {
                while (here->B3SOIFDSsPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B3SOIFDSsPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B3SOIFDsNode != 0) && (here->B3SOIFDsNodePrime != 0))
            {
                while (here->B3SOIFDSspPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B3SOIFDSspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B3SOIFDvbsNode != 0) && (here->B3SOIFDvbsNode != 0))
            {
                while (here->B3SOIFDVbsPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B3SOIFDVbsPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B3SOIFDidsNode != 0) && (here->B3SOIFDidsNode != 0))
            {
                while (here->B3SOIFDIdsPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B3SOIFDIdsPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B3SOIFDicNode != 0) && (here->B3SOIFDicNode != 0))
            {
                while (here->B3SOIFDIcPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B3SOIFDIcPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B3SOIFDibsNode != 0) && (here->B3SOIFDibsNode != 0))
            {
                while (here->B3SOIFDIbsPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B3SOIFDIbsPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B3SOIFDibdNode != 0) && (here->B3SOIFDibdNode != 0))
            {
                while (here->B3SOIFDIbdPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B3SOIFDIbdPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B3SOIFDiiiNode != 0) && (here->B3SOIFDiiiNode != 0))
            {
                while (here->B3SOIFDIiiPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B3SOIFDIiiPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B3SOIFDigidlNode != 0) && (here->B3SOIFDigidlNode != 0))
            {
                while (here->B3SOIFDIgidlPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B3SOIFDIgidlPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B3SOIFDitunNode != 0) && (here->B3SOIFDitunNode != 0))
            {
                while (here->B3SOIFDItunPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B3SOIFDItunPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B3SOIFDibpNode != 0) && (here->B3SOIFDibpNode != 0))
            {
                while (here->B3SOIFDIbpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B3SOIFDIbpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B3SOIFDabeffNode != 0) && (here->B3SOIFDabeffNode != 0))
            {
                while (here->B3SOIFDAbeffPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B3SOIFDAbeffPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B3SOIFDvbs0effNode != 0) && (here->B3SOIFDvbs0effNode != 0))
            {
                while (here->B3SOIFDVbs0effPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B3SOIFDVbs0effPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B3SOIFDvbseffNode != 0) && (here->B3SOIFDvbseffNode != 0))
            {
                while (here->B3SOIFDVbseffPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B3SOIFDVbseffPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B3SOIFDxcNode != 0) && (here->B3SOIFDxcNode != 0))
            {
                while (here->B3SOIFDXcPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B3SOIFDXcPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B3SOIFDcbbNode != 0) && (here->B3SOIFDcbbNode != 0))
            {
                while (here->B3SOIFDCbbPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B3SOIFDCbbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B3SOIFDcbdNode != 0) && (here->B3SOIFDcbdNode != 0))
            {
                while (here->B3SOIFDCbdPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B3SOIFDCbdPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B3SOIFDcbgNode != 0) && (here->B3SOIFDcbgNode != 0))
            {
                while (here->B3SOIFDCbgPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B3SOIFDCbgPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B3SOIFDqbNode != 0) && (here->B3SOIFDqbNode != 0))
            {
                while (here->B3SOIFDqbPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B3SOIFDqbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B3SOIFDqbfNode != 0) && (here->B3SOIFDqbfNode != 0))
            {
                while (here->B3SOIFDQbfPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B3SOIFDQbfPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B3SOIFDqjsNode != 0) && (here->B3SOIFDqjsNode != 0))
            {
                while (here->B3SOIFDQjsPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B3SOIFDQjsPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B3SOIFDqjdNode != 0) && (here->B3SOIFDqjdNode != 0))
            {
                while (here->B3SOIFDQjdPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B3SOIFDQjdPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B3SOIFDgmNode != 0) && (here->B3SOIFDgmNode != 0))
            {
                while (here->B3SOIFDGmPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B3SOIFDGmPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B3SOIFDgmbsNode != 0) && (here->B3SOIFDgmbsNode != 0))
            {
                while (here->B3SOIFDGmbsPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B3SOIFDGmbsPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B3SOIFDgdsNode != 0) && (here->B3SOIFDgdsNode != 0))
            {
                while (here->B3SOIFDGdsPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B3SOIFDGdsPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B3SOIFDgmeNode != 0) && (here->B3SOIFDgmeNode != 0))
            {
                while (here->B3SOIFDGmePtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B3SOIFDGmePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B3SOIFDvbs0teffNode != 0) && (here->B3SOIFDvbs0teffNode != 0))
            {
                while (here->B3SOIFDVbs0teffPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B3SOIFDVbs0teffPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B3SOIFDvthNode != 0) && (here->B3SOIFDvthNode != 0))
            {
                while (here->B3SOIFDVthPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B3SOIFDVthPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B3SOIFDvgsteffNode != 0) && (here->B3SOIFDvgsteffNode != 0))
            {
                while (here->B3SOIFDVgsteffPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B3SOIFDVgsteffPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B3SOIFDxcsatNode != 0) && (here->B3SOIFDxcsatNode != 0))
            {
                while (here->B3SOIFDXcsatPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B3SOIFDXcsatPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B3SOIFDvcscvNode != 0) && (here->B3SOIFDvcscvNode != 0))
            {
                while (here->B3SOIFDVcscvPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B3SOIFDVcscvPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B3SOIFDvdscvNode != 0) && (here->B3SOIFDvdscvNode != 0))
            {
                while (here->B3SOIFDVdscvPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B3SOIFDVdscvPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B3SOIFDcbeNode != 0) && (here->B3SOIFDcbeNode != 0))
            {
                while (here->B3SOIFDCbePtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B3SOIFDCbePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B3SOIFDdum1Node != 0) && (here->B3SOIFDdum1Node != 0))
            {
                while (here->B3SOIFDDum1Ptr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B3SOIFDDum1Ptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B3SOIFDdum2Node != 0) && (here->B3SOIFDdum2Node != 0))
            {
                while (here->B3SOIFDDum2Ptr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B3SOIFDDum2Ptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B3SOIFDdum3Node != 0) && (here->B3SOIFDdum3Node != 0))
            {
                while (here->B3SOIFDDum3Ptr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B3SOIFDDum3Ptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B3SOIFDdum4Node != 0) && (here->B3SOIFDdum4Node != 0))
            {
                while (here->B3SOIFDDum4Ptr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B3SOIFDDum4Ptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B3SOIFDdum5Node != 0) && (here->B3SOIFDdum5Node != 0))
            {
                while (here->B3SOIFDDum5Ptr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B3SOIFDDum5Ptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B3SOIFDqaccNode != 0) && (here->B3SOIFDqaccNode != 0))
            {
                while (here->B3SOIFDQaccPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B3SOIFDQaccPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B3SOIFDqsub0Node != 0) && (here->B3SOIFDqsub0Node != 0))
            {
                while (here->B3SOIFDQsub0Ptr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B3SOIFDQsub0Ptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B3SOIFDqsubs1Node != 0) && (here->B3SOIFDqsubs1Node != 0))
            {
                while (here->B3SOIFDQsubs1Ptr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B3SOIFDQsubs1Ptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B3SOIFDqsubs2Node != 0) && (here->B3SOIFDqsubs2Node != 0))
            {
                while (here->B3SOIFDQsubs2Ptr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B3SOIFDQsubs2Ptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B3SOIFDqeNode != 0) && (here->B3SOIFDqeNode != 0))
            {
                while (here->B3SOIFDqePtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B3SOIFDqePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B3SOIFDqdNode != 0) && (here->B3SOIFDqdNode != 0))
            {
                while (here->B3SOIFDqdPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B3SOIFDqdPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->B3SOIFDqgNode != 0) && (here->B3SOIFDqgNode != 0))
            {
                while (here->B3SOIFDqgPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->B3SOIFDqgPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }
        }
    }

    return (OK) ;
}