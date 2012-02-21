/**********
Author: 2012 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "b3soipddef.h"
#include "ngspice/sperror.h"

int
B3SOIPDbindCSC(GENmodel *inModel, CKTcircuit *ckt)
{
    B3SOIPDmodel *model = (B3SOIPDmodel *)inModel;
    int i ;

    /*  loop through all the b3soipd models */
    for( ; model != NULL; model = model->B3SOIPDnextModel ) {
	B3SOIPDinstance *here;

        /* loop through all the instances of the model */
        for (here = model->B3SOIPDinstances; here != NULL ;
	    here = here->B3SOIPDnextInstance) {

                if ((model->B3SOIPDshMod == 1) && (here->B3SOIPDrth0!=0.0)) {

		i = 0 ;
		if ((here-> B3SOIPDtempNode != 0) && (here-> B3SOIPDtempNode != 0)) {
			while (here->B3SOIPDTemptempPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIPDTemptempPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIPDtempNode != 0) && (here-> B3SOIPDdNodePrime != 0)) {
			while (here->B3SOIPDTempdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIPDTempdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIPDtempNode != 0) && (here-> B3SOIPDsNodePrime != 0)) {
			while (here->B3SOIPDTempspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIPDTempspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIPDtempNode != 0) && (here-> B3SOIPDgNode != 0)) {
			while (here->B3SOIPDTempgPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIPDTempgPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIPDtempNode != 0) && (here-> B3SOIPDbNode != 0)) {
			while (here->B3SOIPDTempbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIPDTempbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIPDgNode != 0) && (here-> B3SOIPDtempNode != 0)) {
			while (here->B3SOIPDGtempPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIPDGtempPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIPDdNodePrime != 0) && (here-> B3SOIPDtempNode != 0)) {
			while (here->B3SOIPDDPtempPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIPDDPtempPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIPDsNodePrime != 0) && (here-> B3SOIPDtempNode != 0)) {
			while (here->B3SOIPDSPtempPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIPDSPtempPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIPDeNode != 0) && (here-> B3SOIPDtempNode != 0)) {
			while (here->B3SOIPDEtempPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIPDEtempPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIPDbNode != 0) && (here-> B3SOIPDtempNode != 0)) {
			while (here->B3SOIPDBtempPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIPDBtempPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

                if (here->B3SOIPDbodyMod == 1) {

		i = 0 ;
		if ((here-> B3SOIPDpNode != 0) && (here-> B3SOIPDtempNode != 0)) {
			while (here->B3SOIPDPtempPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIPDPtempPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
                } /* IF */
                } /* IF */

                if (here->B3SOIPDbodyMod == 2) {
                }
                else if (here->B3SOIPDbodyMod == 1) {

		i = 0 ;
		if ((here-> B3SOIPDbNode != 0) && (here-> B3SOIPDpNode != 0)) {
			while (here->B3SOIPDBpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIPDBpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIPDpNode != 0) && (here-> B3SOIPDbNode != 0)) {
			while (here->B3SOIPDPbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIPDPbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIPDpNode != 0) && (here-> B3SOIPDpNode != 0)) {
			while (here->B3SOIPDPpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIPDPpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
                } /* ELSE */

		i = 0 ;
		if ((here-> B3SOIPDeNode != 0) && (here-> B3SOIPDbNode != 0)) {
			while (here->B3SOIPDEbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIPDEbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIPDgNode != 0) && (here-> B3SOIPDbNode != 0)) {
			while (here->B3SOIPDGbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIPDGbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIPDdNodePrime != 0) && (here-> B3SOIPDbNode != 0)) {
			while (here->B3SOIPDDPbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIPDDPbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIPDsNodePrime != 0) && (here-> B3SOIPDbNode != 0)) {
			while (here->B3SOIPDSPbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIPDSPbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIPDbNode != 0) && (here-> B3SOIPDeNode != 0)) {
			while (here->B3SOIPDBePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIPDBePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIPDbNode != 0) && (here-> B3SOIPDgNode != 0)) {
			while (here->B3SOIPDBgPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIPDBgPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIPDbNode != 0) && (here-> B3SOIPDdNodePrime != 0)) {
			while (here->B3SOIPDBdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIPDBdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIPDbNode != 0) && (here-> B3SOIPDsNodePrime != 0)) {
			while (here->B3SOIPDBspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIPDBspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIPDbNode != 0) && (here-> B3SOIPDbNode != 0)) {
			while (here->B3SOIPDBbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIPDBbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIPDeNode != 0) && (here-> B3SOIPDgNode != 0)) {
			while (here->B3SOIPDEgPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIPDEgPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIPDeNode != 0) && (here-> B3SOIPDdNodePrime != 0)) {
			while (here->B3SOIPDEdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIPDEdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIPDeNode != 0) && (here-> B3SOIPDsNodePrime != 0)) {
			while (here->B3SOIPDEspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIPDEspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIPDgNode != 0) && (here-> B3SOIPDeNode != 0)) {
			while (here->B3SOIPDGePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIPDGePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIPDdNodePrime != 0) && (here-> B3SOIPDeNode != 0)) {
			while (here->B3SOIPDDPePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIPDDPePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIPDsNodePrime != 0) && (here-> B3SOIPDeNode != 0)) {
			while (here->B3SOIPDSPePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIPDSPePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIPDeNode != 0) && (here-> B3SOIPDbNode != 0)) {
			while (here->B3SOIPDEbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIPDEbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIPDeNode != 0) && (here-> B3SOIPDeNode != 0)) {
			while (here->B3SOIPDEePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIPDEePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIPDgNode != 0) && (here-> B3SOIPDgNode != 0)) {
			while (here->B3SOIPDGgPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIPDGgPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIPDgNode != 0) && (here-> B3SOIPDdNodePrime != 0)) {
			while (here->B3SOIPDGdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIPDGdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIPDgNode != 0) && (here-> B3SOIPDsNodePrime != 0)) {
			while (here->B3SOIPDGspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIPDGspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIPDdNodePrime != 0) && (here-> B3SOIPDgNode != 0)) {
			while (here->B3SOIPDDPgPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIPDDPgPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIPDdNodePrime != 0) && (here-> B3SOIPDdNodePrime != 0)) {
			while (here->B3SOIPDDPdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIPDDPdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIPDdNodePrime != 0) && (here-> B3SOIPDsNodePrime != 0)) {
			while (here->B3SOIPDDPspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIPDDPspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIPDdNodePrime != 0) && (here-> B3SOIPDdNode != 0)) {
			while (here->B3SOIPDDPdPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIPDDPdPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIPDsNodePrime != 0) && (here-> B3SOIPDgNode != 0)) {
			while (here->B3SOIPDSPgPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIPDSPgPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIPDsNodePrime != 0) && (here-> B3SOIPDdNodePrime != 0)) {
			while (here->B3SOIPDSPdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIPDSPdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIPDsNodePrime != 0) && (here-> B3SOIPDsNodePrime != 0)) {
			while (here->B3SOIPDSPspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIPDSPspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIPDsNodePrime != 0) && (here-> B3SOIPDsNode != 0)) {
			while (here->B3SOIPDSPsPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIPDSPsPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIPDdNode != 0) && (here-> B3SOIPDdNode != 0)) {
			while (here->B3SOIPDDdPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIPDDdPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIPDdNode != 0) && (here-> B3SOIPDdNodePrime != 0)) {
			while (here->B3SOIPDDdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIPDDdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIPDsNode != 0) && (here-> B3SOIPDsNode != 0)) {
			while (here->B3SOIPDSsPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIPDSsPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIPDsNode != 0) && (here-> B3SOIPDsNodePrime != 0)) {
			while (here->B3SOIPDSspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIPDSspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

                if (here->B3SOIPDdebugMod != 0) {

		i = 0 ;
		if ((here-> B3SOIPDvbsNode != 0) && (here-> B3SOIPDvbsNode != 0)) {
			while (here->B3SOIPDVbsPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIPDVbsPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIPDidsNode != 0) && (here-> B3SOIPDidsNode != 0)) {
			while (here->B3SOIPDIdsPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIPDIdsPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIPDicNode != 0) && (here-> B3SOIPDicNode != 0)) {
			while (here->B3SOIPDIcPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIPDIcPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIPDibsNode != 0) && (here-> B3SOIPDibsNode != 0)) {
			while (here->B3SOIPDIbsPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIPDIbsPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIPDibdNode != 0) && (here-> B3SOIPDibdNode != 0)) {
			while (here->B3SOIPDIbdPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIPDIbdPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIPDiiiNode != 0) && (here-> B3SOIPDiiiNode != 0)) {
			while (here->B3SOIPDIiiPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIPDIiiPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIPDigNode != 0) && (here-> B3SOIPDigNode != 0)) {
			while (here->B3SOIPDIgPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIPDIgPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIPDgiggNode != 0) && (here-> B3SOIPDgiggNode != 0)) {
			while (here->B3SOIPDGiggPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIPDGiggPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIPDgigdNode != 0) && (here-> B3SOIPDgigdNode != 0)) {
			while (here->B3SOIPDGigdPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIPDGigdPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIPDgigbNode != 0) && (here-> B3SOIPDgigbNode != 0)) {
			while (here->B3SOIPDGigbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIPDGigbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIPDigidlNode != 0) && (here-> B3SOIPDigidlNode != 0)) {
			while (here->B3SOIPDIgidlPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIPDIgidlPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIPDitunNode != 0) && (here-> B3SOIPDitunNode != 0)) {
			while (here->B3SOIPDItunPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIPDItunPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIPDibpNode != 0) && (here-> B3SOIPDibpNode != 0)) {
			while (here->B3SOIPDIbpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIPDIbpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIPDcbbNode != 0) && (here-> B3SOIPDcbbNode != 0)) {
			while (here->B3SOIPDCbbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIPDCbbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIPDcbdNode != 0) && (here-> B3SOIPDcbdNode != 0)) {
			while (here->B3SOIPDCbdPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIPDCbdPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIPDcbgNode != 0) && (here-> B3SOIPDcbgNode != 0)) {
			while (here->B3SOIPDCbgPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIPDCbgPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIPDqbfNode != 0) && (here-> B3SOIPDqbfNode != 0)) {
			while (here->B3SOIPDQbfPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIPDQbfPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIPDqjsNode != 0) && (here-> B3SOIPDqjsNode != 0)) {
			while (here->B3SOIPDQjsPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIPDQjsPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIPDqjdNode != 0) && (here-> B3SOIPDqjdNode != 0)) {
			while (here->B3SOIPDQjdPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->B3SOIPDQjdPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
                } /* IF */
	}
    }
    return(OK);
}

int
B3SOIPDbindCSCComplex(GENmodel *inModel, CKTcircuit *ckt)
{
    B3SOIPDmodel *model = (B3SOIPDmodel *)inModel;
    int i ;

    /*  loop through all the b3soipd models */
    for( ; model != NULL; model = model->B3SOIPDnextModel ) {
	B3SOIPDinstance *here;

        /* loop through all the instances of the model */
        for (here = model->B3SOIPDinstances; here != NULL ;
	    here = here->B3SOIPDnextInstance) {

                if ((model->B3SOIPDshMod == 1) && (here->B3SOIPDrth0!=0.0)) {

		i = 0 ;
		if ((here-> B3SOIPDtempNode != 0) && (here-> B3SOIPDtempNode != 0)) {
			while (here->B3SOIPDTemptempPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIPDTemptempPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIPDtempNode != 0) && (here-> B3SOIPDdNodePrime != 0)) {
			while (here->B3SOIPDTempdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIPDTempdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIPDtempNode != 0) && (here-> B3SOIPDsNodePrime != 0)) {
			while (here->B3SOIPDTempspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIPDTempspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIPDtempNode != 0) && (here-> B3SOIPDgNode != 0)) {
			while (here->B3SOIPDTempgPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIPDTempgPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIPDtempNode != 0) && (here-> B3SOIPDbNode != 0)) {
			while (here->B3SOIPDTempbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIPDTempbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIPDgNode != 0) && (here-> B3SOIPDtempNode != 0)) {
			while (here->B3SOIPDGtempPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIPDGtempPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIPDdNodePrime != 0) && (here-> B3SOIPDtempNode != 0)) {
			while (here->B3SOIPDDPtempPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIPDDPtempPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIPDsNodePrime != 0) && (here-> B3SOIPDtempNode != 0)) {
			while (here->B3SOIPDSPtempPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIPDSPtempPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIPDeNode != 0) && (here-> B3SOIPDtempNode != 0)) {
			while (here->B3SOIPDEtempPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIPDEtempPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIPDbNode != 0) && (here-> B3SOIPDtempNode != 0)) {
			while (here->B3SOIPDBtempPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIPDBtempPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

                if (here->B3SOIPDbodyMod == 1) {

		i = 0 ;
		if ((here-> B3SOIPDpNode != 0) && (here-> B3SOIPDtempNode != 0)) {
			while (here->B3SOIPDPtempPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIPDPtempPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
                } /* IF */
                } /* IF */

                if (here->B3SOIPDbodyMod == 2) {
                }
                else if (here->B3SOIPDbodyMod == 1) {

		i = 0 ;
		if ((here-> B3SOIPDbNode != 0) && (here-> B3SOIPDpNode != 0)) {
			while (here->B3SOIPDBpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIPDBpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIPDpNode != 0) && (here-> B3SOIPDbNode != 0)) {
			while (here->B3SOIPDPbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIPDPbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIPDpNode != 0) && (here-> B3SOIPDpNode != 0)) {
			while (here->B3SOIPDPpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIPDPpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
                } /* ELSE */

		i = 0 ;
		if ((here-> B3SOIPDeNode != 0) && (here-> B3SOIPDbNode != 0)) {
			while (here->B3SOIPDEbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIPDEbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIPDgNode != 0) && (here-> B3SOIPDbNode != 0)) {
			while (here->B3SOIPDGbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIPDGbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIPDdNodePrime != 0) && (here-> B3SOIPDbNode != 0)) {
			while (here->B3SOIPDDPbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIPDDPbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIPDsNodePrime != 0) && (here-> B3SOIPDbNode != 0)) {
			while (here->B3SOIPDSPbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIPDSPbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIPDbNode != 0) && (here-> B3SOIPDeNode != 0)) {
			while (here->B3SOIPDBePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIPDBePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIPDbNode != 0) && (here-> B3SOIPDgNode != 0)) {
			while (here->B3SOIPDBgPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIPDBgPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIPDbNode != 0) && (here-> B3SOIPDdNodePrime != 0)) {
			while (here->B3SOIPDBdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIPDBdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIPDbNode != 0) && (here-> B3SOIPDsNodePrime != 0)) {
			while (here->B3SOIPDBspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIPDBspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIPDbNode != 0) && (here-> B3SOIPDbNode != 0)) {
			while (here->B3SOIPDBbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIPDBbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIPDeNode != 0) && (here-> B3SOIPDgNode != 0)) {
			while (here->B3SOIPDEgPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIPDEgPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIPDeNode != 0) && (here-> B3SOIPDdNodePrime != 0)) {
			while (here->B3SOIPDEdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIPDEdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIPDeNode != 0) && (here-> B3SOIPDsNodePrime != 0)) {
			while (here->B3SOIPDEspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIPDEspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIPDgNode != 0) && (here-> B3SOIPDeNode != 0)) {
			while (here->B3SOIPDGePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIPDGePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIPDdNodePrime != 0) && (here-> B3SOIPDeNode != 0)) {
			while (here->B3SOIPDDPePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIPDDPePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIPDsNodePrime != 0) && (here-> B3SOIPDeNode != 0)) {
			while (here->B3SOIPDSPePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIPDSPePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIPDeNode != 0) && (here-> B3SOIPDbNode != 0)) {
			while (here->B3SOIPDEbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIPDEbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIPDeNode != 0) && (here-> B3SOIPDeNode != 0)) {
			while (here->B3SOIPDEePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIPDEePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIPDgNode != 0) && (here-> B3SOIPDgNode != 0)) {
			while (here->B3SOIPDGgPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIPDGgPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIPDgNode != 0) && (here-> B3SOIPDdNodePrime != 0)) {
			while (here->B3SOIPDGdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIPDGdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIPDgNode != 0) && (here-> B3SOIPDsNodePrime != 0)) {
			while (here->B3SOIPDGspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIPDGspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIPDdNodePrime != 0) && (here-> B3SOIPDgNode != 0)) {
			while (here->B3SOIPDDPgPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIPDDPgPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIPDdNodePrime != 0) && (here-> B3SOIPDdNodePrime != 0)) {
			while (here->B3SOIPDDPdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIPDDPdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIPDdNodePrime != 0) && (here-> B3SOIPDsNodePrime != 0)) {
			while (here->B3SOIPDDPspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIPDDPspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIPDdNodePrime != 0) && (here-> B3SOIPDdNode != 0)) {
			while (here->B3SOIPDDPdPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIPDDPdPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIPDsNodePrime != 0) && (here-> B3SOIPDgNode != 0)) {
			while (here->B3SOIPDSPgPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIPDSPgPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIPDsNodePrime != 0) && (here-> B3SOIPDdNodePrime != 0)) {
			while (here->B3SOIPDSPdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIPDSPdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIPDsNodePrime != 0) && (here-> B3SOIPDsNodePrime != 0)) {
			while (here->B3SOIPDSPspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIPDSPspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIPDsNodePrime != 0) && (here-> B3SOIPDsNode != 0)) {
			while (here->B3SOIPDSPsPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIPDSPsPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIPDdNode != 0) && (here-> B3SOIPDdNode != 0)) {
			while (here->B3SOIPDDdPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIPDDdPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIPDdNode != 0) && (here-> B3SOIPDdNodePrime != 0)) {
			while (here->B3SOIPDDdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIPDDdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIPDsNode != 0) && (here-> B3SOIPDsNode != 0)) {
			while (here->B3SOIPDSsPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIPDSsPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIPDsNode != 0) && (here-> B3SOIPDsNodePrime != 0)) {
			while (here->B3SOIPDSspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIPDSspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

                if (here->B3SOIPDdebugMod != 0) {

		i = 0 ;
		if ((here-> B3SOIPDvbsNode != 0) && (here-> B3SOIPDvbsNode != 0)) {
			while (here->B3SOIPDVbsPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIPDVbsPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIPDidsNode != 0) && (here-> B3SOIPDidsNode != 0)) {
			while (here->B3SOIPDIdsPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIPDIdsPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIPDicNode != 0) && (here-> B3SOIPDicNode != 0)) {
			while (here->B3SOIPDIcPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIPDIcPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIPDibsNode != 0) && (here-> B3SOIPDibsNode != 0)) {
			while (here->B3SOIPDIbsPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIPDIbsPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIPDibdNode != 0) && (here-> B3SOIPDibdNode != 0)) {
			while (here->B3SOIPDIbdPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIPDIbdPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIPDiiiNode != 0) && (here-> B3SOIPDiiiNode != 0)) {
			while (here->B3SOIPDIiiPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIPDIiiPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIPDigNode != 0) && (here-> B3SOIPDigNode != 0)) {
			while (here->B3SOIPDIgPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIPDIgPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIPDgiggNode != 0) && (here-> B3SOIPDgiggNode != 0)) {
			while (here->B3SOIPDGiggPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIPDGiggPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIPDgigdNode != 0) && (here-> B3SOIPDgigdNode != 0)) {
			while (here->B3SOIPDGigdPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIPDGigdPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIPDgigbNode != 0) && (here-> B3SOIPDgigbNode != 0)) {
			while (here->B3SOIPDGigbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIPDGigbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIPDigidlNode != 0) && (here-> B3SOIPDigidlNode != 0)) {
			while (here->B3SOIPDIgidlPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIPDIgidlPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIPDitunNode != 0) && (here-> B3SOIPDitunNode != 0)) {
			while (here->B3SOIPDItunPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIPDItunPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIPDibpNode != 0) && (here-> B3SOIPDibpNode != 0)) {
			while (here->B3SOIPDIbpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIPDIbpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIPDcbbNode != 0) && (here-> B3SOIPDcbbNode != 0)) {
			while (here->B3SOIPDCbbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIPDCbbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIPDcbdNode != 0) && (here-> B3SOIPDcbdNode != 0)) {
			while (here->B3SOIPDCbdPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIPDCbdPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIPDcbgNode != 0) && (here-> B3SOIPDcbgNode != 0)) {
			while (here->B3SOIPDCbgPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIPDCbgPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIPDqbfNode != 0) && (here-> B3SOIPDqbfNode != 0)) {
			while (here->B3SOIPDQbfPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIPDQbfPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIPDqjsNode != 0) && (here-> B3SOIPDqjsNode != 0)) {
			while (here->B3SOIPDQjsPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIPDQjsPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> B3SOIPDqjdNode != 0) && (here-> B3SOIPDqjdNode != 0)) {
			while (here->B3SOIPDQjdPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->B3SOIPDQjdPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
                } /* IF */
	}
    }
    return(OK);
}
