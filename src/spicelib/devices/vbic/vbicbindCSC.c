/**********
Author: 2012 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "vbicdefs.h"
#include "ngspice/sperror.h"

int
VBICbindCSC(GENmodel *inModel, CKTcircuit *ckt)
{
    VBICmodel *model = (VBICmodel *)inModel;
    int i ;

    /*  loop through all the vbic models */
    for( ; model != NULL; model = model->VBICnextModel ) {
	VBICinstance *here;

        /* loop through all the instances of the model */
        for (here = model->VBICinstances; here != NULL ;
	    here = here->VBICnextInstance) {

		i = 0 ;
		if ((here->VBICcollNode != 0) && (here->VBICcollNode != 0)) {
			while (here->VBICcollCollPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->VBICcollCollPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here->VBICbaseNode != 0) && (here->VBICbaseNode != 0)) {
			while (here->VBICbaseBasePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->VBICbaseBasePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here->VBICemitNode != 0) && (here->VBICemitNode != 0)) {
			while (here->VBICemitEmitPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->VBICemitEmitPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here->VBICsubsNode != 0) && (here->VBICsubsNode != 0)) {
			while (here->VBICsubsSubsPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->VBICsubsSubsPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here->VBICcollCXNode != 0) && (here->VBICcollCXNode != 0)) {
			while (here->VBICcollCXCollCXPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->VBICcollCXCollCXPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here->VBICcollCINode != 0) && (here->VBICcollCINode != 0)) {
			while (here->VBICcollCICollCIPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->VBICcollCICollCIPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here->VBICbaseBXNode != 0) && (here->VBICbaseBXNode != 0)) {
			while (here->VBICbaseBXBaseBXPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->VBICbaseBXBaseBXPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here->VBICbaseBINode != 0) && (here->VBICbaseBINode != 0)) {
			while (here->VBICbaseBIBaseBIPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->VBICbaseBIBaseBIPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here->VBICemitEINode != 0) && (here->VBICemitEINode != 0)) {
			while (here->VBICemitEIEmitEIPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->VBICemitEIEmitEIPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here->VBICbaseBPNode != 0) && (here->VBICbaseBPNode != 0)) {
			while (here->VBICbaseBPBaseBPPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->VBICbaseBPBaseBPPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here->VBICsubsSINode != 0) && (here->VBICsubsSINode != 0)) {
			while (here->VBICsubsSISubsSIPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->VBICsubsSISubsSIPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here->VBICbaseNode != 0) && (here->VBICemitNode != 0)) {
			while (here->VBICbaseEmitPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->VBICbaseEmitPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here->VBICemitNode != 0) && (here->VBICbaseNode != 0)) {
			while (here->VBICemitBasePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->VBICemitBasePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here->VBICbaseNode != 0) && (here->VBICcollNode != 0)) {
			while (here->VBICbaseCollPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->VBICbaseCollPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here->VBICcollNode != 0) && (here->VBICbaseNode != 0)) {
			while (here->VBICcollBasePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->VBICcollBasePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here->VBICcollNode != 0) && (here->VBICcollCXNode != 0)) {
			while (here->VBICcollCollCXPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->VBICcollCollCXPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here->VBICbaseNode != 0) && (here->VBICbaseBXNode != 0)) {
			while (here->VBICbaseBaseBXPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->VBICbaseBaseBXPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here->VBICemitNode != 0) && (here->VBICemitEINode != 0)) {
			while (here->VBICemitEmitEIPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->VBICemitEmitEIPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here->VBICsubsNode != 0) && (here->VBICsubsSINode != 0)) {
			while (here->VBICsubsSubsSIPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->VBICsubsSubsSIPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here->VBICcollCXNode != 0) && (here->VBICcollCINode != 0)) {
			while (here->VBICcollCXCollCIPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->VBICcollCXCollCIPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here->VBICcollCXNode != 0) && (here->VBICbaseBXNode != 0)) {
			while (here->VBICcollCXBaseBXPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->VBICcollCXBaseBXPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here->VBICcollCXNode != 0) && (here->VBICbaseBINode != 0)) {
			while (here->VBICcollCXBaseBIPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->VBICcollCXBaseBIPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here->VBICcollCXNode != 0) && (here->VBICbaseBPNode != 0)) {
			while (here->VBICcollCXBaseBPPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->VBICcollCXBaseBPPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here->VBICcollCINode != 0) && (here->VBICbaseBINode != 0)) {
			while (here->VBICcollCIBaseBIPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->VBICcollCIBaseBIPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here->VBICcollCINode != 0) && (here->VBICemitEINode != 0)) {
			while (here->VBICcollCIEmitEIPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->VBICcollCIEmitEIPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here->VBICbaseBXNode != 0) && (here->VBICbaseBINode != 0)) {
			while (here->VBICbaseBXBaseBIPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->VBICbaseBXBaseBIPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here->VBICbaseBXNode != 0) && (here->VBICemitEINode != 0)) {
			while (here->VBICbaseBXEmitEIPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->VBICbaseBXEmitEIPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here->VBICbaseBXNode != 0) && (here->VBICbaseBPNode != 0)) {
			while (here->VBICbaseBXBaseBPPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->VBICbaseBXBaseBPPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here->VBICbaseBXNode != 0) && (here->VBICsubsSINode != 0)) {
			while (here->VBICbaseBXSubsSIPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->VBICbaseBXSubsSIPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here->VBICbaseBINode != 0) && (here->VBICemitEINode != 0)) {
			while (here->VBICbaseBIEmitEIPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->VBICbaseBIEmitEIPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here->VBICbaseBPNode != 0) && (here->VBICsubsSINode != 0)) {
			while (here->VBICbaseBPSubsSIPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->VBICbaseBPSubsSIPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here->VBICcollCXNode != 0) && (here->VBICcollNode != 0)) {
			while (here->VBICcollCXCollPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->VBICcollCXCollPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here->VBICbaseBXNode != 0) && (here->VBICbaseNode != 0)) {
			while (here->VBICbaseBXBasePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->VBICbaseBXBasePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here->VBICemitEINode != 0) && (here->VBICemitNode != 0)) {
			while (here->VBICemitEIEmitPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->VBICemitEIEmitPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here->VBICsubsSINode != 0) && (here->VBICsubsNode != 0)) {
			while (here->VBICsubsSISubsPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->VBICsubsSISubsPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here->VBICcollCINode != 0) && (here->VBICcollCXNode != 0)) {
			while (here->VBICcollCICollCXPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->VBICcollCICollCXPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here->VBICbaseBINode != 0) && (here->VBICcollCXNode != 0)) {
			while (here->VBICbaseBICollCXPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->VBICbaseBICollCXPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here->VBICbaseBPNode != 0) && (here->VBICcollCXNode != 0)) {
			while (here->VBICbaseBPCollCXPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->VBICbaseBPCollCXPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here->VBICbaseBXNode != 0) && (here->VBICcollCINode != 0)) {
			while (here->VBICbaseBXCollCIPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->VBICbaseBXCollCIPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here->VBICbaseBINode != 0) && (here->VBICcollCINode != 0)) {
			while (here->VBICbaseBICollCIPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->VBICbaseBICollCIPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here->VBICemitEINode != 0) && (here->VBICcollCINode != 0)) {
			while (here->VBICemitEICollCIPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->VBICemitEICollCIPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here->VBICbaseBPNode != 0) && (here->VBICcollCINode != 0)) {
			while (here->VBICbaseBPCollCIPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->VBICbaseBPCollCIPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here->VBICbaseBINode != 0) && (here->VBICbaseBXNode != 0)) {
			while (here->VBICbaseBIBaseBXPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->VBICbaseBIBaseBXPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here->VBICemitEINode != 0) && (here->VBICbaseBXNode != 0)) {
			while (here->VBICemitEIBaseBXPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->VBICemitEIBaseBXPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here->VBICbaseBPNode != 0) && (here->VBICbaseBXNode != 0)) {
			while (here->VBICbaseBPBaseBXPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->VBICbaseBPBaseBXPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here->VBICsubsSINode != 0) && (here->VBICbaseBXNode != 0)) {
			while (here->VBICsubsSIBaseBXPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->VBICsubsSIBaseBXPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here->VBICemitEINode != 0) && (here->VBICbaseBINode != 0)) {
			while (here->VBICemitEIBaseBIPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->VBICemitEIBaseBIPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here->VBICbaseBPNode != 0) && (here->VBICbaseBINode != 0)) {
			while (here->VBICbaseBPBaseBIPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->VBICbaseBPBaseBIPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here->VBICsubsSINode != 0) && (here->VBICcollCINode != 0)) {
			while (here->VBICsubsSICollCIPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->VBICsubsSICollCIPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here->VBICsubsSINode != 0) && (here->VBICbaseBINode != 0)) {
			while (here->VBICsubsSIBaseBIPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->VBICsubsSIBaseBIPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here->VBICsubsSINode != 0) && (here->VBICbaseBPNode != 0)) {
			while (here->VBICsubsSIBaseBPPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->VBICsubsSIBaseBPPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
	}
    }
    return(OK);
}

int
VBICbindCSCComplex(GENmodel *inModel, CKTcircuit *ckt)
{
    VBICmodel *model = (VBICmodel *)inModel;
    int i ;

    /*  loop through all the vbic models */
    for( ; model != NULL; model = model->VBICnextModel ) {
	VBICinstance *here;

        /* loop through all the instances of the model */
        for (here = model->VBICinstances; here != NULL ;
	    here = here->VBICnextInstance) {

		i = 0 ;
		if ((here->VBICcollNode != 0) && (here->VBICcollNode != 0)) {
			while (here->VBICcollCollPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->VBICcollCollPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here->VBICbaseNode != 0) && (here->VBICbaseNode != 0)) {
			while (here->VBICbaseBasePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->VBICbaseBasePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here->VBICemitNode != 0) && (here->VBICemitNode != 0)) {
			while (here->VBICemitEmitPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->VBICemitEmitPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here->VBICsubsNode != 0) && (here->VBICsubsNode != 0)) {
			while (here->VBICsubsSubsPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->VBICsubsSubsPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here->VBICcollCXNode != 0) && (here->VBICcollCXNode != 0)) {
			while (here->VBICcollCXCollCXPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->VBICcollCXCollCXPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here->VBICcollCINode != 0) && (here->VBICcollCINode != 0)) {
			while (here->VBICcollCICollCIPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->VBICcollCICollCIPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here->VBICbaseBXNode != 0) && (here->VBICbaseBXNode != 0)) {
			while (here->VBICbaseBXBaseBXPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->VBICbaseBXBaseBXPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here->VBICbaseBINode != 0) && (here->VBICbaseBINode != 0)) {
			while (here->VBICbaseBIBaseBIPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->VBICbaseBIBaseBIPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here->VBICemitEINode != 0) && (here->VBICemitEINode != 0)) {
			while (here->VBICemitEIEmitEIPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->VBICemitEIEmitEIPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here->VBICbaseBPNode != 0) && (here->VBICbaseBPNode != 0)) {
			while (here->VBICbaseBPBaseBPPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->VBICbaseBPBaseBPPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here->VBICsubsSINode != 0) && (here->VBICsubsSINode != 0)) {
			while (here->VBICsubsSISubsSIPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->VBICsubsSISubsSIPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here->VBICbaseNode != 0) && (here->VBICemitNode != 0)) {
			while (here->VBICbaseEmitPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->VBICbaseEmitPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here->VBICemitNode != 0) && (here->VBICbaseNode != 0)) {
			while (here->VBICemitBasePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->VBICemitBasePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here->VBICbaseNode != 0) && (here->VBICcollNode != 0)) {
			while (here->VBICbaseCollPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->VBICbaseCollPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here->VBICcollNode != 0) && (here->VBICbaseNode != 0)) {
			while (here->VBICcollBasePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->VBICcollBasePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here->VBICcollNode != 0) && (here->VBICcollCXNode != 0)) {
			while (here->VBICcollCollCXPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->VBICcollCollCXPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here->VBICbaseNode != 0) && (here->VBICbaseBXNode != 0)) {
			while (here->VBICbaseBaseBXPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->VBICbaseBaseBXPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here->VBICemitNode != 0) && (here->VBICemitEINode != 0)) {
			while (here->VBICemitEmitEIPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->VBICemitEmitEIPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here->VBICsubsNode != 0) && (here->VBICsubsSINode != 0)) {
			while (here->VBICsubsSubsSIPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->VBICsubsSubsSIPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here->VBICcollCXNode != 0) && (here->VBICcollCINode != 0)) {
			while (here->VBICcollCXCollCIPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->VBICcollCXCollCIPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here->VBICcollCXNode != 0) && (here->VBICbaseBXNode != 0)) {
			while (here->VBICcollCXBaseBXPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->VBICcollCXBaseBXPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here->VBICcollCXNode != 0) && (here->VBICbaseBINode != 0)) {
			while (here->VBICcollCXBaseBIPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->VBICcollCXBaseBIPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here->VBICcollCXNode != 0) && (here->VBICbaseBPNode != 0)) {
			while (here->VBICcollCXBaseBPPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->VBICcollCXBaseBPPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here->VBICcollCINode != 0) && (here->VBICbaseBINode != 0)) {
			while (here->VBICcollCIBaseBIPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->VBICcollCIBaseBIPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here->VBICcollCINode != 0) && (here->VBICemitEINode != 0)) {
			while (here->VBICcollCIEmitEIPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->VBICcollCIEmitEIPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here->VBICbaseBXNode != 0) && (here->VBICbaseBINode != 0)) {
			while (here->VBICbaseBXBaseBIPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->VBICbaseBXBaseBIPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here->VBICbaseBXNode != 0) && (here->VBICemitEINode != 0)) {
			while (here->VBICbaseBXEmitEIPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->VBICbaseBXEmitEIPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here->VBICbaseBXNode != 0) && (here->VBICbaseBPNode != 0)) {
			while (here->VBICbaseBXBaseBPPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->VBICbaseBXBaseBPPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here->VBICbaseBXNode != 0) && (here->VBICsubsSINode != 0)) {
			while (here->VBICbaseBXSubsSIPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->VBICbaseBXSubsSIPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here->VBICbaseBINode != 0) && (here->VBICemitEINode != 0)) {
			while (here->VBICbaseBIEmitEIPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->VBICbaseBIEmitEIPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here->VBICbaseBPNode != 0) && (here->VBICsubsSINode != 0)) {
			while (here->VBICbaseBPSubsSIPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->VBICbaseBPSubsSIPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here->VBICcollCXNode != 0) && (here->VBICcollNode != 0)) {
			while (here->VBICcollCXCollPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->VBICcollCXCollPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here->VBICbaseBXNode != 0) && (here->VBICbaseNode != 0)) {
			while (here->VBICbaseBXBasePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->VBICbaseBXBasePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here->VBICemitEINode != 0) && (here->VBICemitNode != 0)) {
			while (here->VBICemitEIEmitPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->VBICemitEIEmitPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here->VBICsubsSINode != 0) && (here->VBICsubsNode != 0)) {
			while (here->VBICsubsSISubsPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->VBICsubsSISubsPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here->VBICcollCINode != 0) && (here->VBICcollCXNode != 0)) {
			while (here->VBICcollCICollCXPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->VBICcollCICollCXPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here->VBICbaseBINode != 0) && (here->VBICcollCXNode != 0)) {
			while (here->VBICbaseBICollCXPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->VBICbaseBICollCXPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here->VBICbaseBPNode != 0) && (here->VBICcollCXNode != 0)) {
			while (here->VBICbaseBPCollCXPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->VBICbaseBPCollCXPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here->VBICbaseBXNode != 0) && (here->VBICcollCINode != 0)) {
			while (here->VBICbaseBXCollCIPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->VBICbaseBXCollCIPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here->VBICbaseBINode != 0) && (here->VBICcollCINode != 0)) {
			while (here->VBICbaseBICollCIPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->VBICbaseBICollCIPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here->VBICemitEINode != 0) && (here->VBICcollCINode != 0)) {
			while (here->VBICemitEICollCIPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->VBICemitEICollCIPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here->VBICbaseBPNode != 0) && (here->VBICcollCINode != 0)) {
			while (here->VBICbaseBPCollCIPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->VBICbaseBPCollCIPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here->VBICbaseBINode != 0) && (here->VBICbaseBXNode != 0)) {
			while (here->VBICbaseBIBaseBXPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->VBICbaseBIBaseBXPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here->VBICemitEINode != 0) && (here->VBICbaseBXNode != 0)) {
			while (here->VBICemitEIBaseBXPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->VBICemitEIBaseBXPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here->VBICbaseBPNode != 0) && (here->VBICbaseBXNode != 0)) {
			while (here->VBICbaseBPBaseBXPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->VBICbaseBPBaseBXPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here->VBICsubsSINode != 0) && (here->VBICbaseBXNode != 0)) {
			while (here->VBICsubsSIBaseBXPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->VBICsubsSIBaseBXPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here->VBICemitEINode != 0) && (here->VBICbaseBINode != 0)) {
			while (here->VBICemitEIBaseBIPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->VBICemitEIBaseBIPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here->VBICbaseBPNode != 0) && (here->VBICbaseBINode != 0)) {
			while (here->VBICbaseBPBaseBIPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->VBICbaseBPBaseBIPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here->VBICsubsSINode != 0) && (here->VBICcollCINode != 0)) {
			while (here->VBICsubsSICollCIPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->VBICsubsSICollCIPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here->VBICsubsSINode != 0) && (here->VBICbaseBINode != 0)) {
			while (here->VBICsubsSIBaseBIPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->VBICsubsSIBaseBIPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here->VBICsubsSINode != 0) && (here->VBICbaseBPNode != 0)) {
			while (here->VBICsubsSIBaseBPPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->VBICsubsSIBaseBPPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
	}
    }
    return(OK);
}
