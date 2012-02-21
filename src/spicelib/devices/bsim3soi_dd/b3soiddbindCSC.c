/**********
Author: 2012 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "b3soidddef.h"
#include "ngspice/sperror.h"

int
B3SOIDDbindCSC(GENmodel *inModel, CKTcircuit *ckt)
{
    B3SOIDDmodel *model = (B3SOIDDmodel *)inModel;
    int i ;

    /*  loop through all the b3soidd models */
    for( ; model != NULL; model = model->B3SOIDDnextModel ) {
	B3SOIDDinstance *here;

        /* loop through all the instances of the model */
        for (here = model->B3SOIDDinstances; here != NULL ;
	    here = here->B3SOIDDnextInstance) {

                if ((model->B3SOIDDshMod == 1) && (here->B3SOIDDrth0!=0.0)) {

		i = 0 ;
		if ((here-> B3SOIDDtempNode != 0) && (here-> B3SOIDDtempNode != 0)) {
			while (here->B3SOIDDTemptempPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIDDTemptempPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDtempNode != 0) && (here-> B3SOIDDdNodePrime != 0)) {
			while (here->B3SOIDDTempdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIDDTempdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDtempNode != 0) && (here-> B3SOIDDsNodePrime != 0)) {
			while (here->B3SOIDDTempspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIDDTempspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDtempNode != 0) && (here-> B3SOIDDgNode != 0)) {
			while (here->B3SOIDDTempgPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIDDTempgPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDtempNode != 0) && (here-> B3SOIDDbNode != 0)) {
			while (here->B3SOIDDTempbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIDDTempbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDtempNode != 0) && (here-> B3SOIDDeNode != 0)) {
			while (here->B3SOIDDTempePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIDDTempePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDgNode != 0) && (here-> B3SOIDDtempNode != 0)) {
			while (here->B3SOIDDGtempPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIDDGtempPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDdNodePrime != 0) && (here-> B3SOIDDtempNode != 0)) {
			while (here->B3SOIDDDPtempPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIDDDPtempPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDsNodePrime != 0) && (here-> B3SOIDDtempNode != 0)) {
			while (here->B3SOIDDSPtempPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIDDSPtempPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDeNode != 0) && (here-> B3SOIDDtempNode != 0)) {
			while (here->B3SOIDDEtempPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIDDEtempPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDbNode != 0) && (here-> B3SOIDDtempNode != 0)) {
			while (here->B3SOIDDBtempPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIDDBtempPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

                if (here->B3SOIDDbodyMod == 1) {

		i = 0 ;
		if ((here-> B3SOIDDpNode != 0) && (here-> B3SOIDDtempNode != 0)) {
			while (here->B3SOIDDPtempPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIDDPtempPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
                } /* IF */
                } /* IF */

                if (here->B3SOIDDbodyMod == 2) {
                }
                else if (here->B3SOIDDbodyMod == 1) {

		i = 0 ;
		if ((here-> B3SOIDDbNode != 0) && (here-> B3SOIDDpNode != 0)) {
			while (here->B3SOIDDBpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIDDBpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDpNode != 0) && (here-> B3SOIDDbNode != 0)) {
			while (here->B3SOIDDPbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIDDPbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDpNode != 0) && (here-> B3SOIDDpNode != 0)) {
			while (here->B3SOIDDPpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIDDPpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDpNode != 0) && (here-> B3SOIDDgNode != 0)) {
			while (here->B3SOIDDPgPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIDDPgPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDpNode != 0) && (here-> B3SOIDDdNodePrime != 0)) {
			while (here->B3SOIDDPdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIDDPdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDpNode != 0) && (here-> B3SOIDDsNodePrime != 0)) {
			while (here->B3SOIDDPspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIDDPspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDpNode != 0) && (here-> B3SOIDDeNode != 0)) {
			while (here->B3SOIDDPePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIDDPePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
                } /* ELSE */

		i = 0 ;
		if ((here-> B3SOIDDeNode != 0) && (here-> B3SOIDDgNode != 0)) {
			while (here->B3SOIDDEgPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIDDEgPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDeNode != 0) && (here-> B3SOIDDdNodePrime != 0)) {
			while (here->B3SOIDDEdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIDDEdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDeNode != 0) && (here-> B3SOIDDsNodePrime != 0)) {
			while (here->B3SOIDDEspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIDDEspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDgNode != 0) && (here-> B3SOIDDeNode != 0)) {
			while (here->B3SOIDDGePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIDDGePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDdNodePrime != 0) && (here-> B3SOIDDeNode != 0)) {
			while (here->B3SOIDDDPePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIDDDPePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDsNodePrime != 0) && (here-> B3SOIDDeNode != 0)) {
			while (here->B3SOIDDSPePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIDDSPePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDeNode != 0) && (here-> B3SOIDDbNode != 0)) {
			while (here->B3SOIDDEbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIDDEbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDgNode != 0) && (here-> B3SOIDDbNode != 0)) {
			while (here->B3SOIDDGbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIDDGbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDdNodePrime != 0) && (here-> B3SOIDDbNode != 0)) {
			while (here->B3SOIDDDPbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIDDDPbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDsNodePrime != 0) && (here-> B3SOIDDbNode != 0)) {
			while (here->B3SOIDDSPbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIDDSPbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDbNode != 0) && (here-> B3SOIDDeNode != 0)) {
			while (here->B3SOIDDBePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIDDBePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDbNode != 0) && (here-> B3SOIDDgNode != 0)) {
			while (here->B3SOIDDBgPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIDDBgPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDbNode != 0) && (here-> B3SOIDDdNodePrime != 0)) {
			while (here->B3SOIDDBdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIDDBdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDbNode != 0) && (here-> B3SOIDDsNodePrime != 0)) {
			while (here->B3SOIDDBspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIDDBspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDbNode != 0) && (here-> B3SOIDDbNode != 0)) {
			while (here->B3SOIDDBbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIDDBbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDeNode != 0) && (here-> B3SOIDDbNode != 0)) {
			while (here->B3SOIDDEbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIDDEbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDeNode != 0) && (here-> B3SOIDDeNode != 0)) {
			while (here->B3SOIDDEePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIDDEePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDgNode != 0) && (here-> B3SOIDDgNode != 0)) {
			while (here->B3SOIDDGgPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIDDGgPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDgNode != 0) && (here-> B3SOIDDdNodePrime != 0)) {
			while (here->B3SOIDDGdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIDDGdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDgNode != 0) && (here-> B3SOIDDsNodePrime != 0)) {
			while (here->B3SOIDDGspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIDDGspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDdNodePrime != 0) && (here-> B3SOIDDgNode != 0)) {
			while (here->B3SOIDDDPgPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIDDDPgPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDdNodePrime != 0) && (here-> B3SOIDDdNodePrime != 0)) {
			while (here->B3SOIDDDPdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIDDDPdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDdNodePrime != 0) && (here-> B3SOIDDsNodePrime != 0)) {
			while (here->B3SOIDDDPspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIDDDPspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDdNodePrime != 0) && (here-> B3SOIDDdNode != 0)) {
			while (here->B3SOIDDDPdPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIDDDPdPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDsNodePrime != 0) && (here-> B3SOIDDgNode != 0)) {
			while (here->B3SOIDDSPgPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIDDSPgPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDsNodePrime != 0) && (here-> B3SOIDDdNodePrime != 0)) {
			while (here->B3SOIDDSPdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIDDSPdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDsNodePrime != 0) && (here-> B3SOIDDsNodePrime != 0)) {
			while (here->B3SOIDDSPspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIDDSPspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDsNodePrime != 0) && (here-> B3SOIDDsNode != 0)) {
			while (here->B3SOIDDSPsPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIDDSPsPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDdNode != 0) && (here-> B3SOIDDdNode != 0)) {
			while (here->B3SOIDDDdPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIDDDdPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDdNode != 0) && (here-> B3SOIDDdNodePrime != 0)) {
			while (here->B3SOIDDDdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIDDDdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDsNode != 0) && (here-> B3SOIDDsNode != 0)) {
			while (here->B3SOIDDSsPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIDDSsPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDsNode != 0) && (here-> B3SOIDDsNodePrime != 0)) {
			while (here->B3SOIDDSspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIDDSspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

                if ((here->B3SOIDDdebugMod > 1) || (here->B3SOIDDdebugMod == -1)) {

		i = 0 ;
		if ((here-> B3SOIDDvbsNode != 0) && (here-> B3SOIDDvbsNode != 0)) {
			while (here->B3SOIDDVbsPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIDDVbsPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDidsNode != 0) && (here-> B3SOIDDidsNode != 0)) {
			while (here->B3SOIDDIdsPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIDDIdsPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDicNode != 0) && (here-> B3SOIDDicNode != 0)) {
			while (here->B3SOIDDIcPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIDDIcPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDibsNode != 0) && (here-> B3SOIDDibsNode != 0)) {
			while (here->B3SOIDDIbsPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIDDIbsPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDibdNode != 0) && (here-> B3SOIDDibdNode != 0)) {
			while (here->B3SOIDDIbdPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIDDIbdPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDiiiNode != 0) && (here-> B3SOIDDiiiNode != 0)) {
			while (here->B3SOIDDIiiPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIDDIiiPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDigidlNode != 0) && (here-> B3SOIDDigidlNode != 0)) {
			while (here->B3SOIDDIgidlPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIDDIgidlPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDitunNode != 0) && (here-> B3SOIDDitunNode != 0)) {
			while (here->B3SOIDDItunPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIDDItunPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDibpNode != 0) && (here-> B3SOIDDibpNode != 0)) {
			while (here->B3SOIDDIbpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIDDIbpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDabeffNode != 0) && (here-> B3SOIDDabeffNode != 0)) {
			while (here->B3SOIDDAbeffPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIDDAbeffPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDvbs0effNode != 0) && (here-> B3SOIDDvbs0effNode != 0)) {
			while (here->B3SOIDDVbs0effPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIDDVbs0effPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDvbseffNode != 0) && (here-> B3SOIDDvbseffNode != 0)) {
			while (here->B3SOIDDVbseffPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIDDVbseffPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDxcNode != 0) && (here-> B3SOIDDxcNode != 0)) {
			while (here->B3SOIDDXcPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIDDXcPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDcbbNode != 0) && (here-> B3SOIDDcbbNode != 0)) {
			while (here->B3SOIDDCbbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIDDCbbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDcbdNode != 0) && (here-> B3SOIDDcbdNode != 0)) {
			while (here->B3SOIDDCbdPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIDDCbdPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDcbgNode != 0) && (here-> B3SOIDDcbgNode != 0)) {
			while (here->B3SOIDDCbgPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIDDCbgPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDqbNode != 0) && (here-> B3SOIDDqbNode != 0)) {
			while (here->B3SOIDDqbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIDDqbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDqbfNode != 0) && (here-> B3SOIDDqbfNode != 0)) {
			while (here->B3SOIDDQbfPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIDDQbfPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDqjsNode != 0) && (here-> B3SOIDDqjsNode != 0)) {
			while (here->B3SOIDDQjsPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIDDQjsPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDqjdNode != 0) && (here-> B3SOIDDqjdNode != 0)) {
			while (here->B3SOIDDQjdPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIDDQjdPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDgmNode != 0) && (here-> B3SOIDDgmNode != 0)) {
			while (here->B3SOIDDGmPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIDDGmPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDgmbsNode != 0) && (here-> B3SOIDDgmbsNode != 0)) {
			while (here->B3SOIDDGmbsPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIDDGmbsPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDgdsNode != 0) && (here-> B3SOIDDgdsNode != 0)) {
			while (here->B3SOIDDGdsPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIDDGdsPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDgmeNode != 0) && (here-> B3SOIDDgmeNode != 0)) {
			while (here->B3SOIDDGmePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIDDGmePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDvbs0teffNode != 0) && (here-> B3SOIDDvbs0teffNode != 0)) {
			while (here->B3SOIDDVbs0teffPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIDDVbs0teffPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDvthNode != 0) && (here-> B3SOIDDvthNode != 0)) {
			while (here->B3SOIDDVthPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIDDVthPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDvgsteffNode != 0) && (here-> B3SOIDDvgsteffNode != 0)) {
			while (here->B3SOIDDVgsteffPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIDDVgsteffPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDxcsatNode != 0) && (here-> B3SOIDDxcsatNode != 0)) {
			while (here->B3SOIDDXcsatPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIDDXcsatPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDvcscvNode != 0) && (here-> B3SOIDDvcscvNode != 0)) {
			while (here->B3SOIDDVcscvPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIDDVcscvPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDvdscvNode != 0) && (here-> B3SOIDDvdscvNode != 0)) {
			while (here->B3SOIDDVdscvPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIDDVdscvPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDcbeNode != 0) && (here-> B3SOIDDcbeNode != 0)) {
			while (here->B3SOIDDCbePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIDDCbePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDdum1Node != 0) && (here-> B3SOIDDdum1Node != 0)) {
			while (here->B3SOIDDDum1Ptr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIDDDum1Ptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDdum2Node != 0) && (here-> B3SOIDDdum2Node != 0)) {
			while (here->B3SOIDDDum2Ptr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIDDDum2Ptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDdum3Node != 0) && (here-> B3SOIDDdum3Node != 0)) {
			while (here->B3SOIDDDum3Ptr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIDDDum3Ptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDdum4Node != 0) && (here-> B3SOIDDdum4Node != 0)) {
			while (here->B3SOIDDDum4Ptr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIDDDum4Ptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDdum5Node != 0) && (here-> B3SOIDDdum5Node != 0)) {
			while (here->B3SOIDDDum5Ptr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIDDDum5Ptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDqaccNode != 0) && (here-> B3SOIDDqaccNode != 0)) {
			while (here->B3SOIDDQaccPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIDDQaccPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDqsub0Node != 0) && (here-> B3SOIDDqsub0Node != 0)) {
			while (here->B3SOIDDQsub0Ptr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIDDQsub0Ptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDqsubs1Node != 0) && (here-> B3SOIDDqsubs1Node != 0)) {
			while (here->B3SOIDDQsubs1Ptr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIDDQsubs1Ptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDqsubs2Node != 0) && (here-> B3SOIDDqsubs2Node != 0)) {
			while (here->B3SOIDDQsubs2Ptr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIDDQsubs2Ptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDqeNode != 0) && (here-> B3SOIDDqeNode != 0)) {
			while (here->B3SOIDDqePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIDDqePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDqdNode != 0) && (here-> B3SOIDDqdNode != 0)) {
			while (here->B3SOIDDqdPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIDDqdPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDqgNode != 0) && (here-> B3SOIDDqgNode != 0)) {
			while (here->B3SOIDDqgPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIDDqgPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
                } /* IF */
	}
    }
    return(OK);
}

int
B3SOIDDbindCSCComplex(GENmodel *inModel, CKTcircuit *ckt)
{
    B3SOIDDmodel *model = (B3SOIDDmodel *)inModel;
    int i ;

    /*  loop through all the b3soidd models */
    for( ; model != NULL; model = model->B3SOIDDnextModel ) {
	B3SOIDDinstance *here;

        /* loop through all the instances of the model */
        for (here = model->B3SOIDDinstances; here != NULL ;
	    here = here->B3SOIDDnextInstance) {

                if ((model->B3SOIDDshMod == 1) && (here->B3SOIDDrth0!=0.0)) {

		i = 0 ;
		if ((here-> B3SOIDDtempNode != 0) && (here-> B3SOIDDtempNode != 0)) {
			while (here->B3SOIDDTemptempPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIDDTemptempPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDtempNode != 0) && (here-> B3SOIDDdNodePrime != 0)) {
			while (here->B3SOIDDTempdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIDDTempdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDtempNode != 0) && (here-> B3SOIDDsNodePrime != 0)) {
			while (here->B3SOIDDTempspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIDDTempspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDtempNode != 0) && (here-> B3SOIDDgNode != 0)) {
			while (here->B3SOIDDTempgPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIDDTempgPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDtempNode != 0) && (here-> B3SOIDDbNode != 0)) {
			while (here->B3SOIDDTempbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIDDTempbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDtempNode != 0) && (here-> B3SOIDDeNode != 0)) {
			while (here->B3SOIDDTempePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIDDTempePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDgNode != 0) && (here-> B3SOIDDtempNode != 0)) {
			while (here->B3SOIDDGtempPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIDDGtempPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDdNodePrime != 0) && (here-> B3SOIDDtempNode != 0)) {
			while (here->B3SOIDDDPtempPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIDDDPtempPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDsNodePrime != 0) && (here-> B3SOIDDtempNode != 0)) {
			while (here->B3SOIDDSPtempPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIDDSPtempPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDeNode != 0) && (here-> B3SOIDDtempNode != 0)) {
			while (here->B3SOIDDEtempPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIDDEtempPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDbNode != 0) && (here-> B3SOIDDtempNode != 0)) {
			while (here->B3SOIDDBtempPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIDDBtempPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

                if (here->B3SOIDDbodyMod == 1) {

		i = 0 ;
		if ((here-> B3SOIDDpNode != 0) && (here-> B3SOIDDtempNode != 0)) {
			while (here->B3SOIDDPtempPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIDDPtempPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
                } /* IF */
                } /* IF */

                if (here->B3SOIDDbodyMod == 2) {
                }
                else if (here->B3SOIDDbodyMod == 1) {

		i = 0 ;
		if ((here-> B3SOIDDbNode != 0) && (here-> B3SOIDDpNode != 0)) {
			while (here->B3SOIDDBpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIDDBpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDpNode != 0) && (here-> B3SOIDDbNode != 0)) {
			while (here->B3SOIDDPbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIDDPbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDpNode != 0) && (here-> B3SOIDDpNode != 0)) {
			while (here->B3SOIDDPpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIDDPpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDpNode != 0) && (here-> B3SOIDDgNode != 0)) {
			while (here->B3SOIDDPgPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIDDPgPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDpNode != 0) && (here-> B3SOIDDdNodePrime != 0)) {
			while (here->B3SOIDDPdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIDDPdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDpNode != 0) && (here-> B3SOIDDsNodePrime != 0)) {
			while (here->B3SOIDDPspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIDDPspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDpNode != 0) && (here-> B3SOIDDeNode != 0)) {
			while (here->B3SOIDDPePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIDDPePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
                } /* ELSE */

		i = 0 ;
		if ((here-> B3SOIDDeNode != 0) && (here-> B3SOIDDgNode != 0)) {
			while (here->B3SOIDDEgPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIDDEgPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDeNode != 0) && (here-> B3SOIDDdNodePrime != 0)) {
			while (here->B3SOIDDEdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIDDEdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDeNode != 0) && (here-> B3SOIDDsNodePrime != 0)) {
			while (here->B3SOIDDEspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIDDEspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDgNode != 0) && (here-> B3SOIDDeNode != 0)) {
			while (here->B3SOIDDGePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIDDGePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDdNodePrime != 0) && (here-> B3SOIDDeNode != 0)) {
			while (here->B3SOIDDDPePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIDDDPePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDsNodePrime != 0) && (here-> B3SOIDDeNode != 0)) {
			while (here->B3SOIDDSPePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIDDSPePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDeNode != 0) && (here-> B3SOIDDbNode != 0)) {
			while (here->B3SOIDDEbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIDDEbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDgNode != 0) && (here-> B3SOIDDbNode != 0)) {
			while (here->B3SOIDDGbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIDDGbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDdNodePrime != 0) && (here-> B3SOIDDbNode != 0)) {
			while (here->B3SOIDDDPbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIDDDPbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDsNodePrime != 0) && (here-> B3SOIDDbNode != 0)) {
			while (here->B3SOIDDSPbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIDDSPbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDbNode != 0) && (here-> B3SOIDDeNode != 0)) {
			while (here->B3SOIDDBePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIDDBePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDbNode != 0) && (here-> B3SOIDDgNode != 0)) {
			while (here->B3SOIDDBgPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIDDBgPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDbNode != 0) && (here-> B3SOIDDdNodePrime != 0)) {
			while (here->B3SOIDDBdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIDDBdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDbNode != 0) && (here-> B3SOIDDsNodePrime != 0)) {
			while (here->B3SOIDDBspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIDDBspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDbNode != 0) && (here-> B3SOIDDbNode != 0)) {
			while (here->B3SOIDDBbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIDDBbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDeNode != 0) && (here-> B3SOIDDbNode != 0)) {
			while (here->B3SOIDDEbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIDDEbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDeNode != 0) && (here-> B3SOIDDeNode != 0)) {
			while (here->B3SOIDDEePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIDDEePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDgNode != 0) && (here-> B3SOIDDgNode != 0)) {
			while (here->B3SOIDDGgPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIDDGgPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDgNode != 0) && (here-> B3SOIDDdNodePrime != 0)) {
			while (here->B3SOIDDGdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIDDGdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDgNode != 0) && (here-> B3SOIDDsNodePrime != 0)) {
			while (here->B3SOIDDGspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIDDGspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDdNodePrime != 0) && (here-> B3SOIDDgNode != 0)) {
			while (here->B3SOIDDDPgPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIDDDPgPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDdNodePrime != 0) && (here-> B3SOIDDdNodePrime != 0)) {
			while (here->B3SOIDDDPdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIDDDPdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDdNodePrime != 0) && (here-> B3SOIDDsNodePrime != 0)) {
			while (here->B3SOIDDDPspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIDDDPspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDdNodePrime != 0) && (here-> B3SOIDDdNode != 0)) {
			while (here->B3SOIDDDPdPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIDDDPdPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDsNodePrime != 0) && (here-> B3SOIDDgNode != 0)) {
			while (here->B3SOIDDSPgPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIDDSPgPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDsNodePrime != 0) && (here-> B3SOIDDdNodePrime != 0)) {
			while (here->B3SOIDDSPdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIDDSPdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDsNodePrime != 0) && (here-> B3SOIDDsNodePrime != 0)) {
			while (here->B3SOIDDSPspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIDDSPspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDsNodePrime != 0) && (here-> B3SOIDDsNode != 0)) {
			while (here->B3SOIDDSPsPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIDDSPsPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDdNode != 0) && (here-> B3SOIDDdNode != 0)) {
			while (here->B3SOIDDDdPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIDDDdPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDdNode != 0) && (here-> B3SOIDDdNodePrime != 0)) {
			while (here->B3SOIDDDdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIDDDdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDsNode != 0) && (here-> B3SOIDDsNode != 0)) {
			while (here->B3SOIDDSsPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIDDSsPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDsNode != 0) && (here-> B3SOIDDsNodePrime != 0)) {
			while (here->B3SOIDDSspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIDDSspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

                if ((here->B3SOIDDdebugMod > 1) || (here->B3SOIDDdebugMod == -1)) {

		i = 0 ;
		if ((here-> B3SOIDDvbsNode != 0) && (here-> B3SOIDDvbsNode != 0)) {
			while (here->B3SOIDDVbsPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIDDVbsPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDidsNode != 0) && (here-> B3SOIDDidsNode != 0)) {
			while (here->B3SOIDDIdsPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIDDIdsPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDicNode != 0) && (here-> B3SOIDDicNode != 0)) {
			while (here->B3SOIDDIcPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIDDIcPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDibsNode != 0) && (here-> B3SOIDDibsNode != 0)) {
			while (here->B3SOIDDIbsPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIDDIbsPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDibdNode != 0) && (here-> B3SOIDDibdNode != 0)) {
			while (here->B3SOIDDIbdPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIDDIbdPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDiiiNode != 0) && (here-> B3SOIDDiiiNode != 0)) {
			while (here->B3SOIDDIiiPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIDDIiiPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDigidlNode != 0) && (here-> B3SOIDDigidlNode != 0)) {
			while (here->B3SOIDDIgidlPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIDDIgidlPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDitunNode != 0) && (here-> B3SOIDDitunNode != 0)) {
			while (here->B3SOIDDItunPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIDDItunPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDibpNode != 0) && (here-> B3SOIDDibpNode != 0)) {
			while (here->B3SOIDDIbpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIDDIbpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDabeffNode != 0) && (here-> B3SOIDDabeffNode != 0)) {
			while (here->B3SOIDDAbeffPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIDDAbeffPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDvbs0effNode != 0) && (here-> B3SOIDDvbs0effNode != 0)) {
			while (here->B3SOIDDVbs0effPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIDDVbs0effPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDvbseffNode != 0) && (here-> B3SOIDDvbseffNode != 0)) {
			while (here->B3SOIDDVbseffPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIDDVbseffPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDxcNode != 0) && (here-> B3SOIDDxcNode != 0)) {
			while (here->B3SOIDDXcPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIDDXcPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDcbbNode != 0) && (here-> B3SOIDDcbbNode != 0)) {
			while (here->B3SOIDDCbbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIDDCbbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDcbdNode != 0) && (here-> B3SOIDDcbdNode != 0)) {
			while (here->B3SOIDDCbdPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIDDCbdPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDcbgNode != 0) && (here-> B3SOIDDcbgNode != 0)) {
			while (here->B3SOIDDCbgPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIDDCbgPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDqbNode != 0) && (here-> B3SOIDDqbNode != 0)) {
			while (here->B3SOIDDqbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIDDqbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDqbfNode != 0) && (here-> B3SOIDDqbfNode != 0)) {
			while (here->B3SOIDDQbfPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIDDQbfPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDqjsNode != 0) && (here-> B3SOIDDqjsNode != 0)) {
			while (here->B3SOIDDQjsPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIDDQjsPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDqjdNode != 0) && (here-> B3SOIDDqjdNode != 0)) {
			while (here->B3SOIDDQjdPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIDDQjdPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDgmNode != 0) && (here-> B3SOIDDgmNode != 0)) {
			while (here->B3SOIDDGmPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIDDGmPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDgmbsNode != 0) && (here-> B3SOIDDgmbsNode != 0)) {
			while (here->B3SOIDDGmbsPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIDDGmbsPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDgdsNode != 0) && (here-> B3SOIDDgdsNode != 0)) {
			while (here->B3SOIDDGdsPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIDDGdsPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDgmeNode != 0) && (here-> B3SOIDDgmeNode != 0)) {
			while (here->B3SOIDDGmePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIDDGmePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDvbs0teffNode != 0) && (here-> B3SOIDDvbs0teffNode != 0)) {
			while (here->B3SOIDDVbs0teffPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIDDVbs0teffPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDvthNode != 0) && (here-> B3SOIDDvthNode != 0)) {
			while (here->B3SOIDDVthPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIDDVthPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDvgsteffNode != 0) && (here-> B3SOIDDvgsteffNode != 0)) {
			while (here->B3SOIDDVgsteffPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIDDVgsteffPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDxcsatNode != 0) && (here-> B3SOIDDxcsatNode != 0)) {
			while (here->B3SOIDDXcsatPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIDDXcsatPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDvcscvNode != 0) && (here-> B3SOIDDvcscvNode != 0)) {
			while (here->B3SOIDDVcscvPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIDDVcscvPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDvdscvNode != 0) && (here-> B3SOIDDvdscvNode != 0)) {
			while (here->B3SOIDDVdscvPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIDDVdscvPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDcbeNode != 0) && (here-> B3SOIDDcbeNode != 0)) {
			while (here->B3SOIDDCbePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIDDCbePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDdum1Node != 0) && (here-> B3SOIDDdum1Node != 0)) {
			while (here->B3SOIDDDum1Ptr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIDDDum1Ptr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDdum2Node != 0) && (here-> B3SOIDDdum2Node != 0)) {
			while (here->B3SOIDDDum2Ptr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIDDDum2Ptr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDdum3Node != 0) && (here-> B3SOIDDdum3Node != 0)) {
			while (here->B3SOIDDDum3Ptr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIDDDum3Ptr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDdum4Node != 0) && (here-> B3SOIDDdum4Node != 0)) {
			while (here->B3SOIDDDum4Ptr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIDDDum4Ptr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDdum5Node != 0) && (here-> B3SOIDDdum5Node != 0)) {
			while (here->B3SOIDDDum5Ptr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIDDDum5Ptr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDqaccNode != 0) && (here-> B3SOIDDqaccNode != 0)) {
			while (here->B3SOIDDQaccPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIDDQaccPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDqsub0Node != 0) && (here-> B3SOIDDqsub0Node != 0)) {
			while (here->B3SOIDDQsub0Ptr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIDDQsub0Ptr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDqsubs1Node != 0) && (here-> B3SOIDDqsubs1Node != 0)) {
			while (here->B3SOIDDQsubs1Ptr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIDDQsubs1Ptr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDqsubs2Node != 0) && (here-> B3SOIDDqsubs2Node != 0)) {
			while (here->B3SOIDDQsubs2Ptr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIDDQsubs2Ptr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDqeNode != 0) && (here-> B3SOIDDqeNode != 0)) {
			while (here->B3SOIDDqePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIDDqePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDqdNode != 0) && (here-> B3SOIDDqdNode != 0)) {
			while (here->B3SOIDDqdPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIDDqdPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIDDqgNode != 0) && (here-> B3SOIDDqgNode != 0)) {
			while (here->B3SOIDDqgPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIDDqgPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
                } /* IF */
	}
    }
    return(OK);
}
