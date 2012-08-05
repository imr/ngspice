/**********
Author: 2012 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "hsmhvdef.h"
#include "ngspice/sperror.h"

int
HSMHVbindCSC(GENmodel *inModel, CKTcircuit *ckt)
{
    HSMHVmodel *model = (HSMHVmodel *)inModel;
    int i ;

    /*  loop through all the hsmhv models */
    for( ; model != NULL; model = model->HSMHVnextModel ) {
	HSMHVinstance *here;

        /* loop through all the instances of the model */
        for (here = model->HSMHVinstances; here != NULL ;
	    here = here->HSMHVnextInstance) {

		i = 0 ;
		if ((here-> HSMHVdNodePrime != 0) && (here-> HSMHVbNodePrime != 0)) {
			while (here->HSMHVDPbpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HSMHVDPbpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVsNodePrime != 0) && (here-> HSMHVbNodePrime != 0)) {
			while (here->HSMHVSPbpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HSMHVSPbpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVgNodePrime != 0) && (here-> HSMHVbNodePrime != 0)) {
			while (here->HSMHVGPbpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HSMHVGPbpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVbNodePrime != 0) && (here-> HSMHVdNode != 0)) {
			while (here->HSMHVBPdPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HSMHVBPdPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVbNodePrime != 0) && (here-> HSMHVdNodePrime != 0)) {
			while (here->HSMHVBPdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HSMHVBPdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVbNodePrime != 0) && (here-> HSMHVsNode != 0)) {
			while (here->HSMHVBPsPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HSMHVBPsPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVbNodePrime != 0) && (here-> HSMHVsNodePrime != 0)) {
			while (here->HSMHVBPspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HSMHVBPspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVbNodePrime != 0) && (here-> HSMHVgNodePrime != 0)) {
			while (here->HSMHVBPgpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HSMHVBPgpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVbNodePrime != 0) && (here-> HSMHVbNodePrime != 0)) {
			while (here->HSMHVBPbpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HSMHVBPbpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVdNode != 0) && (here-> HSMHVdNode != 0)) {
			while (here->HSMHVDdPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HSMHVDdPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVgNodePrime != 0) && (here-> HSMHVgNodePrime != 0)) {
			while (here->HSMHVGPgpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HSMHVGPgpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVsNode != 0) && (here-> HSMHVsNode != 0)) {
			while (here->HSMHVSsPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HSMHVSsPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVdNodePrime != 0) && (here-> HSMHVdNodePrime != 0)) {
			while (here->HSMHVDPdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HSMHVDPdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVsNodePrime != 0) && (here-> HSMHVsNodePrime != 0)) {
			while (here->HSMHVSPspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HSMHVSPspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVdNode != 0) && (here-> HSMHVdNodePrime != 0)) {
			while (here->HSMHVDdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HSMHVDdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVgNodePrime != 0) && (here-> HSMHVdNodePrime != 0)) {
			while (here->HSMHVGPdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HSMHVGPdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVgNodePrime != 0) && (here-> HSMHVsNodePrime != 0)) {
			while (here->HSMHVGPspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HSMHVGPspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVsNode != 0) && (here-> HSMHVsNodePrime != 0)) {
			while (here->HSMHVSspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HSMHVSspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVdNodePrime != 0) && (here-> HSMHVsNodePrime != 0)) {
			while (here->HSMHVDPspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HSMHVDPspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVdNodePrime != 0) && (here-> HSMHVdNode != 0)) {
			while (here->HSMHVDPdPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HSMHVDPdPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVdNodePrime != 0) && (here-> HSMHVgNodePrime != 0)) {
			while (here->HSMHVDPgpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HSMHVDPgpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVsNodePrime != 0) && (here-> HSMHVgNodePrime != 0)) {
			while (here->HSMHVSPgpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HSMHVSPgpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVsNodePrime != 0) && (here-> HSMHVsNode != 0)) {
			while (here->HSMHVSPsPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HSMHVSPsPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVsNodePrime != 0) && (here-> HSMHVdNodePrime != 0)) {
			while (here->HSMHVSPdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HSMHVSPdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVgNode != 0) && (here-> HSMHVgNode != 0)) {
			while (here->HSMHVGgPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HSMHVGgPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVgNode != 0) && (here-> HSMHVgNodePrime != 0)) {
			while (here->HSMHVGgpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HSMHVGgpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVgNodePrime != 0) && (here-> HSMHVgNode != 0)) {
			while (here->HSMHVGPgPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HSMHVGPgPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVdNode != 0) && (here-> HSMHVdbNode != 0)) {
			while (here->HSMHVDdbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HSMHVDdbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVsNode != 0) && (here-> HSMHVsbNode != 0)) {
			while (here->HSMHVSsbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HSMHVSsbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVdbNode != 0) && (here-> HSMHVdNode != 0)) {
			while (here->HSMHVDBdPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HSMHVDBdPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVdbNode != 0) && (here-> HSMHVdbNode != 0)) {
			while (here->HSMHVDBdbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HSMHVDBdbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVdbNode != 0) && (here-> HSMHVbNodePrime != 0)) {
			while (here->HSMHVDBbpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HSMHVDBbpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVbNodePrime != 0) && (here-> HSMHVdbNode != 0)) {
			while (here->HSMHVBPdbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HSMHVBPdbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVbNodePrime != 0) && (here-> HSMHVbNode != 0)) {
			while (here->HSMHVBPbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HSMHVBPbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVbNodePrime != 0) && (here-> HSMHVsbNode != 0)) {
			while (here->HSMHVBPsbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HSMHVBPsbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVsbNode != 0) && (here-> HSMHVsNode != 0)) {
			while (here->HSMHVSBsPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HSMHVSBsPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVsbNode != 0) && (here-> HSMHVbNodePrime != 0)) {
			while (here->HSMHVSBbpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HSMHVSBbpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVsbNode != 0) && (here-> HSMHVsbNode != 0)) {
			while (here->HSMHVSBsbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HSMHVSBsbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVbNode != 0) && (here-> HSMHVbNodePrime != 0)) {
			while (here->HSMHVBbpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HSMHVBbpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVbNode != 0) && (here-> HSMHVbNode != 0)) {
			while (here->HSMHVBbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HSMHVBbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVdNode != 0) && (here-> HSMHVgNodePrime != 0)) {
			while (here->HSMHVDgpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HSMHVDgpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVdNode != 0) && (here-> HSMHVsNode != 0)) {
			while (here->HSMHVDsPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HSMHVDsPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVdNode != 0) && (here-> HSMHVbNodePrime != 0)) {
			while (here->HSMHVDbpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HSMHVDbpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVdNodePrime != 0) && (here-> HSMHVsNode != 0)) {
			while (here->HSMHVDPsPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HSMHVDPsPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVsNode != 0) && (here-> HSMHVgNodePrime != 0)) {
			while (here->HSMHVSgpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HSMHVSgpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVsNode != 0) && (here-> HSMHVdNode != 0)) {
			while (here->HSMHVSdPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HSMHVSdPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVsNode != 0) && (here-> HSMHVbNodePrime != 0)) {
			while (here->HSMHVSbpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HSMHVSbpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVsNodePrime != 0) && (here-> HSMHVdNode != 0)) {
			while (here->HSMHVSPdPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HSMHVSPdPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVgNodePrime != 0) && (here-> HSMHVdNode != 0)) {
			while (here->HSMHVGPdPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HSMHVGPdPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVgNodePrime != 0) && (here-> HSMHVsNode != 0)) {
			while (here->HSMHVGPsPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HSMHVGPsPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

                if ( here->HSMHVsubNode > 0 ) { /* 5th substrate node */

		i = 0 ;
		if ((here-> HSMHVdNode != 0) && (here-> HSMHVsubNode != 0)) {
			while (here->HSMHVDsubPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HSMHVDsubPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVdNodePrime != 0) && (here-> HSMHVsubNode != 0)) {
			while (here->HSMHVDPsubPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HSMHVDPsubPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVsNode != 0) && (here-> HSMHVsubNode != 0)) {
			while (here->HSMHVSsubPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HSMHVSsubPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVsNodePrime != 0) && (here-> HSMHVsubNode != 0)) {
			while (here->HSMHVSPsubPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HSMHVSPsubPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
                } /* IF */

                if ( here->HSMHV_coselfheat >  0 ) { /* self heating */

		i = 0 ;
		if ((here-> HSMHVtempNode != 0) && (here-> HSMHVtempNode != 0)) {
			while (here->HSMHVTemptempPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HSMHVTemptempPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVtempNode != 0) && (here-> HSMHVdNode != 0)) {
			while (here->HSMHVTempdPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HSMHVTempdPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVtempNode != 0) && (here-> HSMHVdNodePrime != 0)) {
			while (here->HSMHVTempdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HSMHVTempdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVtempNode != 0) && (here-> HSMHVsNode != 0)) {
			while (here->HSMHVTempsPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HSMHVTempsPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVtempNode != 0) && (here-> HSMHVsNodePrime != 0)) {
			while (here->HSMHVTempspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HSMHVTempspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVdNodePrime != 0) && (here-> HSMHVtempNode != 0)) {
			while (here->HSMHVDPtempPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HSMHVDPtempPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVsNodePrime != 0) && (here-> HSMHVtempNode != 0)) {
			while (here->HSMHVSPtempPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HSMHVSPtempPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVtempNode != 0) && (here-> HSMHVgNodePrime != 0)) {
			while (here->HSMHVTempgpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HSMHVTempgpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVtempNode != 0) && (here-> HSMHVbNodePrime != 0)) {
			while (here->HSMHVTempbpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HSMHVTempbpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVgNodePrime != 0) && (here-> HSMHVtempNode != 0)) {
			while (here->HSMHVGPtempPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HSMHVGPtempPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVbNodePrime != 0) && (here-> HSMHVtempNode != 0)) {
			while (here->HSMHVBPtempPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HSMHVBPtempPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVdbNode != 0) && (here-> HSMHVtempNode != 0)) {
			while (here->HSMHVDBtempPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HSMHVDBtempPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVsbNode != 0) && (here-> HSMHVtempNode != 0)) {
			while (here->HSMHVSBtempPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HSMHVSBtempPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVdNode != 0) && (here-> HSMHVtempNode != 0)) {
			while (here->HSMHVDtempPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HSMHVDtempPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVsNode != 0) && (here-> HSMHVtempNode != 0)) {
			while (here->HSMHVStempPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HSMHVStempPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
                } /* IF */

                if ( model->HSMHV_conqs ) { /* flat handling of NQS */

		i = 0 ;
		if ((here-> HSMHVdNodePrime != 0) && (here-> HSMHVqiNode != 0)) {
			while (here->HSMHVDPqiPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HSMHVDPqiPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVgNodePrime != 0) && (here-> HSMHVqiNode != 0)) {
			while (here->HSMHVGPqiPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HSMHVGPqiPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVgNodePrime != 0) && (here-> HSMHVqbNode != 0)) {
			while (here->HSMHVGPqbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HSMHVGPqbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVsNodePrime != 0) && (here-> HSMHVqiNode != 0)) {
			while (here->HSMHVSPqiPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HSMHVSPqiPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVbNodePrime != 0) && (here-> HSMHVqbNode != 0)) {
			while (here->HSMHVBPqbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HSMHVBPqbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVqiNode != 0) && (here-> HSMHVdNodePrime != 0)) {
			while (here->HSMHVQIdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HSMHVQIdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVqiNode != 0) && (here-> HSMHVgNodePrime != 0)) {
			while (here->HSMHVQIgpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HSMHVQIgpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVqiNode != 0) && (here-> HSMHVsNodePrime != 0)) {
			while (here->HSMHVQIspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HSMHVQIspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVqiNode != 0) && (here-> HSMHVbNodePrime != 0)) {
			while (here->HSMHVQIbpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HSMHVQIbpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVqiNode != 0) && (here-> HSMHVqiNode != 0)) {
			while (here->HSMHVQIqiPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HSMHVQIqiPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVqbNode != 0) && (here-> HSMHVdNodePrime != 0)) {
			while (here->HSMHVQBdpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HSMHVQBdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVqbNode != 0) && (here-> HSMHVgNodePrime != 0)) {
			while (here->HSMHVQBgpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HSMHVQBgpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVqbNode != 0) && (here-> HSMHVsNodePrime != 0)) {
			while (here->HSMHVQBspPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HSMHVQBspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVqbNode != 0) && (here-> HSMHVbNodePrime != 0)) {
			while (here->HSMHVQBbpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HSMHVQBbpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVqbNode != 0) && (here-> HSMHVqbNode != 0)) {
			while (here->HSMHVQBqbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HSMHVQBqbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

                if ( here->HSMHV_coselfheat >  0 ) { /* self heating */

		i = 0 ;
		if ((here-> HSMHVqiNode != 0) && (here-> HSMHVtempNode != 0)) {
			while (here->HSMHVQItempPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HSMHVQItempPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVqbNode != 0) && (here-> HSMHVtempNode != 0)) {
			while (here->HSMHVQBtempPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->HSMHVQBtempPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
                }
                }
	}
    }
    return(OK);
}

int
HSMHVbindCSCComplex(GENmodel *inModel, CKTcircuit *ckt)
{
    HSMHVmodel *model = (HSMHVmodel *)inModel;
    int i ;

    /*  loop through all the hsmhv models */
    for( ; model != NULL; model = model->HSMHVnextModel ) {
	HSMHVinstance *here;

        /* loop through all the instances of the model */
        for (here = model->HSMHVinstances; here != NULL ;
	    here = here->HSMHVnextInstance) {

		i = 0 ;
		if ((here-> HSMHVdNodePrime != 0) && (here-> HSMHVbNodePrime != 0)) {
			while (here->HSMHVDPbpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HSMHVDPbpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVsNodePrime != 0) && (here-> HSMHVbNodePrime != 0)) {
			while (here->HSMHVSPbpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HSMHVSPbpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVgNodePrime != 0) && (here-> HSMHVbNodePrime != 0)) {
			while (here->HSMHVGPbpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HSMHVGPbpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVbNodePrime != 0) && (here-> HSMHVdNode != 0)) {
			while (here->HSMHVBPdPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HSMHVBPdPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVbNodePrime != 0) && (here-> HSMHVdNodePrime != 0)) {
			while (here->HSMHVBPdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HSMHVBPdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVbNodePrime != 0) && (here-> HSMHVsNode != 0)) {
			while (here->HSMHVBPsPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HSMHVBPsPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVbNodePrime != 0) && (here-> HSMHVsNodePrime != 0)) {
			while (here->HSMHVBPspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HSMHVBPspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVbNodePrime != 0) && (here-> HSMHVgNodePrime != 0)) {
			while (here->HSMHVBPgpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HSMHVBPgpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVbNodePrime != 0) && (here-> HSMHVbNodePrime != 0)) {
			while (here->HSMHVBPbpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HSMHVBPbpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVdNode != 0) && (here-> HSMHVdNode != 0)) {
			while (here->HSMHVDdPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HSMHVDdPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVgNodePrime != 0) && (here-> HSMHVgNodePrime != 0)) {
			while (here->HSMHVGPgpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HSMHVGPgpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVsNode != 0) && (here-> HSMHVsNode != 0)) {
			while (here->HSMHVSsPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HSMHVSsPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVdNodePrime != 0) && (here-> HSMHVdNodePrime != 0)) {
			while (here->HSMHVDPdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HSMHVDPdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVsNodePrime != 0) && (here-> HSMHVsNodePrime != 0)) {
			while (here->HSMHVSPspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HSMHVSPspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVdNode != 0) && (here-> HSMHVdNodePrime != 0)) {
			while (here->HSMHVDdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HSMHVDdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVgNodePrime != 0) && (here-> HSMHVdNodePrime != 0)) {
			while (here->HSMHVGPdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HSMHVGPdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVgNodePrime != 0) && (here-> HSMHVsNodePrime != 0)) {
			while (here->HSMHVGPspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HSMHVGPspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVsNode != 0) && (here-> HSMHVsNodePrime != 0)) {
			while (here->HSMHVSspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HSMHVSspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVdNodePrime != 0) && (here-> HSMHVsNodePrime != 0)) {
			while (here->HSMHVDPspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HSMHVDPspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVdNodePrime != 0) && (here-> HSMHVdNode != 0)) {
			while (here->HSMHVDPdPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HSMHVDPdPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVdNodePrime != 0) && (here-> HSMHVgNodePrime != 0)) {
			while (here->HSMHVDPgpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HSMHVDPgpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVsNodePrime != 0) && (here-> HSMHVgNodePrime != 0)) {
			while (here->HSMHVSPgpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HSMHVSPgpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVsNodePrime != 0) && (here-> HSMHVsNode != 0)) {
			while (here->HSMHVSPsPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HSMHVSPsPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVsNodePrime != 0) && (here-> HSMHVdNodePrime != 0)) {
			while (here->HSMHVSPdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HSMHVSPdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVgNode != 0) && (here-> HSMHVgNode != 0)) {
			while (here->HSMHVGgPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HSMHVGgPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVgNode != 0) && (here-> HSMHVgNodePrime != 0)) {
			while (here->HSMHVGgpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HSMHVGgpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVgNodePrime != 0) && (here-> HSMHVgNode != 0)) {
			while (here->HSMHVGPgPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HSMHVGPgPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVdNode != 0) && (here-> HSMHVdbNode != 0)) {
			while (here->HSMHVDdbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HSMHVDdbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVsNode != 0) && (here-> HSMHVsbNode != 0)) {
			while (here->HSMHVSsbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HSMHVSsbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVdbNode != 0) && (here-> HSMHVdNode != 0)) {
			while (here->HSMHVDBdPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HSMHVDBdPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVdbNode != 0) && (here-> HSMHVdbNode != 0)) {
			while (here->HSMHVDBdbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HSMHVDBdbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVdbNode != 0) && (here-> HSMHVbNodePrime != 0)) {
			while (here->HSMHVDBbpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HSMHVDBbpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVbNodePrime != 0) && (here-> HSMHVdbNode != 0)) {
			while (here->HSMHVBPdbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HSMHVBPdbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVbNodePrime != 0) && (here-> HSMHVbNode != 0)) {
			while (here->HSMHVBPbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HSMHVBPbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVbNodePrime != 0) && (here-> HSMHVsbNode != 0)) {
			while (here->HSMHVBPsbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HSMHVBPsbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVsbNode != 0) && (here-> HSMHVsNode != 0)) {
			while (here->HSMHVSBsPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HSMHVSBsPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVsbNode != 0) && (here-> HSMHVbNodePrime != 0)) {
			while (here->HSMHVSBbpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HSMHVSBbpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVsbNode != 0) && (here-> HSMHVsbNode != 0)) {
			while (here->HSMHVSBsbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HSMHVSBsbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVbNode != 0) && (here-> HSMHVbNodePrime != 0)) {
			while (here->HSMHVBbpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HSMHVBbpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVbNode != 0) && (here-> HSMHVbNode != 0)) {
			while (here->HSMHVBbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HSMHVBbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVdNode != 0) && (here-> HSMHVgNodePrime != 0)) {
			while (here->HSMHVDgpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HSMHVDgpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVdNode != 0) && (here-> HSMHVsNode != 0)) {
			while (here->HSMHVDsPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HSMHVDsPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVdNode != 0) && (here-> HSMHVbNodePrime != 0)) {
			while (here->HSMHVDbpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HSMHVDbpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVdNodePrime != 0) && (here-> HSMHVsNode != 0)) {
			while (here->HSMHVDPsPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HSMHVDPsPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVsNode != 0) && (here-> HSMHVgNodePrime != 0)) {
			while (here->HSMHVSgpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HSMHVSgpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVsNode != 0) && (here-> HSMHVdNode != 0)) {
			while (here->HSMHVSdPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HSMHVSdPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVsNode != 0) && (here-> HSMHVbNodePrime != 0)) {
			while (here->HSMHVSbpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HSMHVSbpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVsNodePrime != 0) && (here-> HSMHVdNode != 0)) {
			while (here->HSMHVSPdPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HSMHVSPdPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVgNodePrime != 0) && (here-> HSMHVdNode != 0)) {
			while (here->HSMHVGPdPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HSMHVGPdPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVgNodePrime != 0) && (here-> HSMHVsNode != 0)) {
			while (here->HSMHVGPsPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HSMHVGPsPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

                if ( here->HSMHVsubNode > 0 ) { /* 5th substrate node */

		i = 0 ;
		if ((here-> HSMHVdNode != 0) && (here-> HSMHVsubNode != 0)) {
			while (here->HSMHVDsubPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HSMHVDsubPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVdNodePrime != 0) && (here-> HSMHVsubNode != 0)) {
			while (here->HSMHVDPsubPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HSMHVDPsubPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVsNode != 0) && (here-> HSMHVsubNode != 0)) {
			while (here->HSMHVSsubPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HSMHVSsubPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVsNodePrime != 0) && (here-> HSMHVsubNode != 0)) {
			while (here->HSMHVSPsubPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HSMHVSPsubPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
                } /* IF */

                if ( here->HSMHV_coselfheat >  0 ) { /* self heating */

		i = 0 ;
		if ((here-> HSMHVtempNode != 0) && (here-> HSMHVtempNode != 0)) {
			while (here->HSMHVTemptempPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HSMHVTemptempPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVtempNode != 0) && (here-> HSMHVdNode != 0)) {
			while (here->HSMHVTempdPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HSMHVTempdPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVtempNode != 0) && (here-> HSMHVdNodePrime != 0)) {
			while (here->HSMHVTempdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HSMHVTempdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVtempNode != 0) && (here-> HSMHVsNode != 0)) {
			while (here->HSMHVTempsPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HSMHVTempsPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVtempNode != 0) && (here-> HSMHVsNodePrime != 0)) {
			while (here->HSMHVTempspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HSMHVTempspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVdNodePrime != 0) && (here-> HSMHVtempNode != 0)) {
			while (here->HSMHVDPtempPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HSMHVDPtempPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVsNodePrime != 0) && (here-> HSMHVtempNode != 0)) {
			while (here->HSMHVSPtempPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HSMHVSPtempPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVtempNode != 0) && (here-> HSMHVgNodePrime != 0)) {
			while (here->HSMHVTempgpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HSMHVTempgpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVtempNode != 0) && (here-> HSMHVbNodePrime != 0)) {
			while (here->HSMHVTempbpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HSMHVTempbpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVgNodePrime != 0) && (here-> HSMHVtempNode != 0)) {
			while (here->HSMHVGPtempPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HSMHVGPtempPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVbNodePrime != 0) && (here-> HSMHVtempNode != 0)) {
			while (here->HSMHVBPtempPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HSMHVBPtempPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVdbNode != 0) && (here-> HSMHVtempNode != 0)) {
			while (here->HSMHVDBtempPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HSMHVDBtempPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVsbNode != 0) && (here-> HSMHVtempNode != 0)) {
			while (here->HSMHVSBtempPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HSMHVSBtempPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVdNode != 0) && (here-> HSMHVtempNode != 0)) {
			while (here->HSMHVDtempPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HSMHVDtempPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVsNode != 0) && (here-> HSMHVtempNode != 0)) {
			while (here->HSMHVStempPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HSMHVStempPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
                } /* IF */

                if ( model->HSMHV_conqs ) { /* flat handling of NQS */

		i = 0 ;
		if ((here-> HSMHVdNodePrime != 0) && (here-> HSMHVqiNode != 0)) {
			while (here->HSMHVDPqiPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HSMHVDPqiPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVgNodePrime != 0) && (here-> HSMHVqiNode != 0)) {
			while (here->HSMHVGPqiPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HSMHVGPqiPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVgNodePrime != 0) && (here-> HSMHVqbNode != 0)) {
			while (here->HSMHVGPqbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HSMHVGPqbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVsNodePrime != 0) && (here-> HSMHVqiNode != 0)) {
			while (here->HSMHVSPqiPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HSMHVSPqiPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVbNodePrime != 0) && (here-> HSMHVqbNode != 0)) {
			while (here->HSMHVBPqbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HSMHVBPqbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVqiNode != 0) && (here-> HSMHVdNodePrime != 0)) {
			while (here->HSMHVQIdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HSMHVQIdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVqiNode != 0) && (here-> HSMHVgNodePrime != 0)) {
			while (here->HSMHVQIgpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HSMHVQIgpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVqiNode != 0) && (here-> HSMHVsNodePrime != 0)) {
			while (here->HSMHVQIspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HSMHVQIspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVqiNode != 0) && (here-> HSMHVbNodePrime != 0)) {
			while (here->HSMHVQIbpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HSMHVQIbpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVqiNode != 0) && (here-> HSMHVqiNode != 0)) {
			while (here->HSMHVQIqiPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HSMHVQIqiPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVqbNode != 0) && (here-> HSMHVdNodePrime != 0)) {
			while (here->HSMHVQBdpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HSMHVQBdpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVqbNode != 0) && (here-> HSMHVgNodePrime != 0)) {
			while (here->HSMHVQBgpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HSMHVQBgpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVqbNode != 0) && (here-> HSMHVsNodePrime != 0)) {
			while (here->HSMHVQBspPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HSMHVQBspPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVqbNode != 0) && (here-> HSMHVbNodePrime != 0)) {
			while (here->HSMHVQBbpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HSMHVQBbpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVqbNode != 0) && (here-> HSMHVqbNode != 0)) {
			while (here->HSMHVQBqbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HSMHVQBqbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

                if ( here->HSMHV_coselfheat >  0 ) { /* self heating */

		i = 0 ;
		if ((here-> HSMHVqiNode != 0) && (here-> HSMHVtempNode != 0)) {
			while (here->HSMHVQItempPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HSMHVQItempPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> HSMHVqbNode != 0) && (here-> HSMHVtempNode != 0)) {
			while (here->HSMHVQBtempPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->HSMHVQBtempPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
                } /* IF */
                } /* IF */
	}
    }
    return(OK);
}

int
HSMHVbindCSCComplexToReal (GENmodel *inModel, CKTcircuit *ckt)
{
    HSMHVmodel *model = (HSMHVmodel *)inModel ;
    HSMHVinstance *here ;
    int i ;

    /*  loop through all the hsmhv models */
    for ( ; model != NULL ; model = model->HSMHVnextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->HSMHVinstances ; here != NULL ; here = here->HSMHVnextInstance)
        {
            i = 0 ;
            if ((here->HSMHVdNodePrime != 0) && (here->HSMHVbNodePrime != 0))
            {
                while (here->HSMHVDPbpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->HSMHVDPbpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->HSMHVsNodePrime != 0) && (here->HSMHVbNodePrime != 0))
            {
                while (here->HSMHVSPbpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->HSMHVSPbpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->HSMHVgNodePrime != 0) && (here->HSMHVbNodePrime != 0))
            {
                while (here->HSMHVGPbpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->HSMHVGPbpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->HSMHVbNodePrime != 0) && (here->HSMHVdNode != 0))
            {
                while (here->HSMHVBPdPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->HSMHVBPdPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->HSMHVbNodePrime != 0) && (here->HSMHVdNodePrime != 0))
            {
                while (here->HSMHVBPdpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->HSMHVBPdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->HSMHVbNodePrime != 0) && (here->HSMHVsNode != 0))
            {
                while (here->HSMHVBPsPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->HSMHVBPsPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->HSMHVbNodePrime != 0) && (here->HSMHVsNodePrime != 0))
            {
                while (here->HSMHVBPspPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->HSMHVBPspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->HSMHVbNodePrime != 0) && (here->HSMHVgNodePrime != 0))
            {
                while (here->HSMHVBPgpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->HSMHVBPgpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->HSMHVbNodePrime != 0) && (here->HSMHVbNodePrime != 0))
            {
                while (here->HSMHVBPbpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->HSMHVBPbpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->HSMHVdNode != 0) && (here->HSMHVdNode != 0))
            {
                while (here->HSMHVDdPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->HSMHVDdPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->HSMHVgNodePrime != 0) && (here->HSMHVgNodePrime != 0))
            {
                while (here->HSMHVGPgpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->HSMHVGPgpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->HSMHVsNode != 0) && (here->HSMHVsNode != 0))
            {
                while (here->HSMHVSsPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->HSMHVSsPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->HSMHVdNodePrime != 0) && (here->HSMHVdNodePrime != 0))
            {
                while (here->HSMHVDPdpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->HSMHVDPdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->HSMHVsNodePrime != 0) && (here->HSMHVsNodePrime != 0))
            {
                while (here->HSMHVSPspPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->HSMHVSPspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->HSMHVdNode != 0) && (here->HSMHVdNodePrime != 0))
            {
                while (here->HSMHVDdpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->HSMHVDdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->HSMHVgNodePrime != 0) && (here->HSMHVdNodePrime != 0))
            {
                while (here->HSMHVGPdpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->HSMHVGPdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->HSMHVgNodePrime != 0) && (here->HSMHVsNodePrime != 0))
            {
                while (here->HSMHVGPspPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->HSMHVGPspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->HSMHVsNode != 0) && (here->HSMHVsNodePrime != 0))
            {
                while (here->HSMHVSspPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->HSMHVSspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->HSMHVdNodePrime != 0) && (here->HSMHVsNodePrime != 0))
            {
                while (here->HSMHVDPspPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->HSMHVDPspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->HSMHVdNodePrime != 0) && (here->HSMHVdNode != 0))
            {
                while (here->HSMHVDPdPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->HSMHVDPdPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->HSMHVdNodePrime != 0) && (here->HSMHVgNodePrime != 0))
            {
                while (here->HSMHVDPgpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->HSMHVDPgpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->HSMHVsNodePrime != 0) && (here->HSMHVgNodePrime != 0))
            {
                while (here->HSMHVSPgpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->HSMHVSPgpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->HSMHVsNodePrime != 0) && (here->HSMHVsNode != 0))
            {
                while (here->HSMHVSPsPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->HSMHVSPsPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->HSMHVsNodePrime != 0) && (here->HSMHVdNodePrime != 0))
            {
                while (here->HSMHVSPdpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->HSMHVSPdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->HSMHVgNode != 0) && (here->HSMHVgNode != 0))
            {
                while (here->HSMHVGgPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->HSMHVGgPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->HSMHVgNode != 0) && (here->HSMHVgNodePrime != 0))
            {
                while (here->HSMHVGgpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->HSMHVGgpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->HSMHVgNodePrime != 0) && (here->HSMHVgNode != 0))
            {
                while (here->HSMHVGPgPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->HSMHVGPgPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

/*            i = 0 ;
            if ((here->HSMHVgNode != 0) && (here->HSMHVdNodePrime != 0))
            {
                while (here->HSMHVGdpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->HSMHVGdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->HSMHVgNode != 0) && (here->HSMHVsNodePrime != 0))
            {
                while (here->HSMHVGspPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->HSMHVGspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->HSMHVgNode != 0) && (here->HSMHVbNodePrime != 0))
            {
                while (here->HSMHVGbpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->HSMHVGbpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }
*/
            i = 0 ;
            if ((here->HSMHVdNode != 0) && (here->HSMHVdbNode != 0))
            {
                while (here->HSMHVDdbPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->HSMHVDdbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->HSMHVsNode != 0) && (here->HSMHVsbNode != 0))
            {
                while (here->HSMHVSsbPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->HSMHVSsbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->HSMHVdbNode != 0) && (here->HSMHVdNode != 0))
            {
                while (here->HSMHVDBdPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->HSMHVDBdPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->HSMHVdbNode != 0) && (here->HSMHVdbNode != 0))
            {
                while (here->HSMHVDBdbPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->HSMHVDBdbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->HSMHVdbNode != 0) && (here->HSMHVbNodePrime != 0))
            {
                while (here->HSMHVDBbpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->HSMHVDBbpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

/*            i = 0 ;
            if ((here->HSMHVdbNode != 0) && (here->HSMHVbNode != 0))
            {
                while (here->HSMHVDBbPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->HSMHVDBbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }
*/
            i = 0 ;
            if ((here->HSMHVbNodePrime != 0) && (here->HSMHVdbNode != 0))
            {
                while (here->HSMHVBPdbPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->HSMHVBPdbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->HSMHVbNodePrime != 0) && (here->HSMHVbNode != 0))
            {
                while (here->HSMHVBPbPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->HSMHVBPbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->HSMHVbNodePrime != 0) && (here->HSMHVsbNode != 0))
            {
                while (here->HSMHVBPsbPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->HSMHVBPsbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->HSMHVsbNode != 0) && (here->HSMHVsNode != 0))
            {
                while (here->HSMHVSBsPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->HSMHVSBsPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->HSMHVsbNode != 0) && (here->HSMHVbNodePrime != 0))
            {
                while (here->HSMHVSBbpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->HSMHVSBbpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

/*            i = 0 ;
            if ((here->HSMHVsbNode != 0) && (here->HSMHVbNode != 0))
            {
                while (here->HSMHVSBbPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->HSMHVSBbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }
*/
            i = 0 ;
            if ((here->HSMHVsbNode != 0) && (here->HSMHVsbNode != 0))
            {
                while (here->HSMHVSBsbPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->HSMHVSBsbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

/*            i = 0 ;
            if ((here->HSMHVbNode != 0) && (here->HSMHVdbNode != 0))
            {
                while (here->HSMHVBdbPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->HSMHVBdbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }
*/
            i = 0 ;
            if ((here->HSMHVbNode != 0) && (here->HSMHVbNodePrime != 0))
            {
                while (here->HSMHVBbpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->HSMHVBbpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

/*            i = 0 ;
            if ((here->HSMHVbNode != 0) && (here->HSMHVsbNode != 0))
            {
                while (here->HSMHVBsbPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->HSMHVBsbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }
*/
            i = 0 ;
            if ((here->HSMHVbNode != 0) && (here->HSMHVbNode != 0))
            {
                while (here->HSMHVBbPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->HSMHVBbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->HSMHVdNode != 0) && (here->HSMHVgNodePrime != 0))
            {
                while (here->HSMHVDgpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->HSMHVDgpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->HSMHVdNode != 0) && (here->HSMHVsNode != 0))
            {
                while (here->HSMHVDsPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->HSMHVDsPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->HSMHVdNode != 0) && (here->HSMHVbNodePrime != 0))
            {
                while (here->HSMHVDbpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->HSMHVDbpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->HSMHVdNodePrime != 0) && (here->HSMHVsNode != 0))
            {
                while (here->HSMHVDPsPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->HSMHVDPsPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->HSMHVsNode != 0) && (here->HSMHVgNodePrime != 0))
            {
                while (here->HSMHVSgpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->HSMHVSgpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->HSMHVsNode != 0) && (here->HSMHVdNode != 0))
            {
                while (here->HSMHVSdPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->HSMHVSdPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->HSMHVsNode != 0) && (here->HSMHVbNodePrime != 0))
            {
                while (here->HSMHVSbpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->HSMHVSbpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->HSMHVsNodePrime != 0) && (here->HSMHVdNode != 0))
            {
                while (here->HSMHVSPdPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->HSMHVSPdPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->HSMHVgNodePrime != 0) && (here->HSMHVdNode != 0))
            {
                while (here->HSMHVGPdPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->HSMHVGPdPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->HSMHVgNodePrime != 0) && (here->HSMHVsNode != 0))
            {
                while (here->HSMHVGPsPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->HSMHVGPsPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->HSMHVdNode != 0) && (here->HSMHVsubNode != 0))
            {
                while (here->HSMHVDsubPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->HSMHVDsubPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->HSMHVdNodePrime != 0) && (here->HSMHVsubNode != 0))
            {
                while (here->HSMHVDPsubPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->HSMHVDPsubPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->HSMHVsNode != 0) && (here->HSMHVsubNode != 0))
            {
                while (here->HSMHVSsubPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->HSMHVSsubPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->HSMHVsNodePrime != 0) && (here->HSMHVsubNode != 0))
            {
                while (here->HSMHVSPsubPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->HSMHVSPsubPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->HSMHVtempNode != 0) && (here->HSMHVtempNode != 0))
            {
                while (here->HSMHVTemptempPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->HSMHVTemptempPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->HSMHVtempNode != 0) && (here->HSMHVdNode != 0))
            {
                while (here->HSMHVTempdPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->HSMHVTempdPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->HSMHVtempNode != 0) && (here->HSMHVdNodePrime != 0))
            {
                while (here->HSMHVTempdpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->HSMHVTempdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->HSMHVtempNode != 0) && (here->HSMHVsNode != 0))
            {
                while (here->HSMHVTempsPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->HSMHVTempsPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->HSMHVtempNode != 0) && (here->HSMHVsNodePrime != 0))
            {
                while (here->HSMHVTempspPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->HSMHVTempspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->HSMHVdNodePrime != 0) && (here->HSMHVtempNode != 0))
            {
                while (here->HSMHVDPtempPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->HSMHVDPtempPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->HSMHVsNodePrime != 0) && (here->HSMHVtempNode != 0))
            {
                while (here->HSMHVSPtempPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->HSMHVSPtempPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->HSMHVtempNode != 0) && (here->HSMHVgNodePrime != 0))
            {
                while (here->HSMHVTempgpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->HSMHVTempgpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->HSMHVtempNode != 0) && (here->HSMHVbNodePrime != 0))
            {
                while (here->HSMHVTempbpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->HSMHVTempbpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->HSMHVgNodePrime != 0) && (here->HSMHVtempNode != 0))
            {
                while (here->HSMHVGPtempPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->HSMHVGPtempPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->HSMHVbNodePrime != 0) && (here->HSMHVtempNode != 0))
            {
                while (here->HSMHVBPtempPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->HSMHVBPtempPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->HSMHVdbNode != 0) && (here->HSMHVtempNode != 0))
            {
                while (here->HSMHVDBtempPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->HSMHVDBtempPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->HSMHVsbNode != 0) && (here->HSMHVtempNode != 0))
            {
                while (here->HSMHVSBtempPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->HSMHVSBtempPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->HSMHVdNode != 0) && (here->HSMHVtempNode != 0))
            {
                while (here->HSMHVDtempPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->HSMHVDtempPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->HSMHVsNode != 0) && (here->HSMHVtempNode != 0))
            {
                while (here->HSMHVStempPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->HSMHVStempPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->HSMHVdNodePrime != 0) && (here->HSMHVqiNode != 0))
            {
                while (here->HSMHVDPqiPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->HSMHVDPqiPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->HSMHVgNodePrime != 0) && (here->HSMHVqiNode != 0))
            {
                while (here->HSMHVGPqiPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->HSMHVGPqiPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->HSMHVgNodePrime != 0) && (here->HSMHVqbNode != 0))
            {
                while (here->HSMHVGPqbPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->HSMHVGPqbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->HSMHVsNodePrime != 0) && (here->HSMHVqiNode != 0))
            {
                while (here->HSMHVSPqiPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->HSMHVSPqiPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->HSMHVbNodePrime != 0) && (here->HSMHVqbNode != 0))
            {
                while (here->HSMHVBPqbPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->HSMHVBPqbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->HSMHVqiNode != 0) && (here->HSMHVdNodePrime != 0))
            {
                while (here->HSMHVQIdpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->HSMHVQIdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->HSMHVqiNode != 0) && (here->HSMHVgNodePrime != 0))
            {
                while (here->HSMHVQIgpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->HSMHVQIgpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->HSMHVqiNode != 0) && (here->HSMHVsNodePrime != 0))
            {
                while (here->HSMHVQIspPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->HSMHVQIspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->HSMHVqiNode != 0) && (here->HSMHVbNodePrime != 0))
            {
                while (here->HSMHVQIbpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->HSMHVQIbpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->HSMHVqiNode != 0) && (here->HSMHVqiNode != 0))
            {
                while (here->HSMHVQIqiPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->HSMHVQIqiPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->HSMHVqbNode != 0) && (here->HSMHVdNodePrime != 0))
            {
                while (here->HSMHVQBdpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->HSMHVQBdpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->HSMHVqbNode != 0) && (here->HSMHVgNodePrime != 0))
            {
                while (here->HSMHVQBgpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->HSMHVQBgpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->HSMHVqbNode != 0) && (here->HSMHVsNodePrime != 0))
            {
                while (here->HSMHVQBspPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->HSMHVQBspPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->HSMHVqbNode != 0) && (here->HSMHVbNodePrime != 0))
            {
                while (here->HSMHVQBbpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->HSMHVQBbpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->HSMHVqbNode != 0) && (here->HSMHVqbNode != 0))
            {
                while (here->HSMHVQBqbPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->HSMHVQBqbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->HSMHVqiNode != 0) && (here->HSMHVtempNode != 0))
            {
                while (here->HSMHVQItempPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->HSMHVQItempPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->HSMHVqbNode != 0) && (here->HSMHVtempNode != 0))
            {
                while (here->HSMHVQBtempPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->HSMHVQBtempPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }
        }
    }

    return (OK) ;
}
