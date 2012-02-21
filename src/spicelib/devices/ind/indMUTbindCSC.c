/**********
Author: 2012 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "inddefs.h"
#include "ngspice/sperror.h"

int
INDbindCSC(GENmodel *inModel, CKTcircuit *ckt)
{
    INDmodel *model = (INDmodel *)inModel;
    INDinstance *here;
    int i ;

        /*  loop through all the INDacitor models */
        for( ; model != NULL; model = model->INDnextModel ) {

            /* loop through all the instances of the model */
            for (here = model->INDinstances; here != NULL ;
                    here=here->INDnextInstance) {

		i = 0 ;
		if ((here->INDposNode != 0) && (here->INDbrEq != 0)) {
			while (here->INDposIbrptr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->INDposIbrptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->INDnegNode != 0) && (here->INDbrEq != 0)) {
			while (here->INDnegIbrptr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->INDnegIbrptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->INDbrEq != 0) && (here->INDnegNode != 0)) {
			while (here->INDibrNegptr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->INDibrNegptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->INDbrEq != 0) && (here->INDposNode != 0)) {
			while (here->INDibrPosptr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->INDibrPosptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->INDbrEq != 0) && (here->INDbrEq != 0)) {
			while (here->INDibrIbrptr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->INDibrIbrptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
	    }
	}
    return(OK);
}

int
INDbindCSCComplex(GENmodel *inModel, CKTcircuit *ckt)
{
    INDmodel *model = (INDmodel *)inModel;
    INDinstance *here;
    int i ;

        /*  loop through all the INDacitor models */
        for( ; model != NULL; model = model->INDnextModel ) {

            /* loop through all the instances of the model */
            for (here = model->INDinstances; here != NULL ;
                    here=here->INDnextInstance) {

		i = 0 ;
		if ((here->INDposNode != 0) && (here->INDbrEq != 0)) {
			while (here->INDposIbrptr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->INDposIbrptr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->INDnegNode != 0) && (here->INDbrEq != 0)) {
			while (here->INDnegIbrptr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->INDnegIbrptr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->INDbrEq != 0) && (here->INDnegNode != 0)) {
			while (here->INDibrNegptr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->INDibrNegptr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->INDbrEq != 0) && (here->INDposNode != 0)) {
			while (here->INDibrPosptr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->INDibrPosptr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->INDbrEq != 0) && (here->INDbrEq != 0)) {
			while (here->INDibrIbrptr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->INDibrIbrptr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
	    }
	}
    return(OK);
}

#ifdef MUTUAL
int
MUTbindCSC(GENmodel *inModel, CKTcircuit *ckt)
{
    MUTmodel *model = (MUTmodel *)inModel;
    MUTinstance *here;
    int i ;

        /*  loop through all the INDacitor models */
        for( ; model != NULL; model = model->MUTnextModel ) {

            /* loop through all the instances of the model */
            for (here = model->MUTinstances; here != NULL ;
                    here=here->MUTnextInstance) {

		i = 0 ;
		if ((here->MUTind1->INDbrEq != 0) && (here->MUTind2->INDbrEq != 0)) {
			while (here->MUTbr1br2 != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MUTbr1br2 = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->MUTind2->INDbrEq != 0) && (here->MUTind1->INDbrEq != 0)) {
			while (here->MUTbr2br1 != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->MUTbr2br1 = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
	    }
	}
    return(OK);
}

int
MUTbindCSCComplex(GENmodel *inModel, CKTcircuit *ckt)
{
    MUTmodel *model = (MUTmodel *)inModel;
    MUTinstance *here;
    int i ;

        /*  loop through all the INDacitor models */
        for( ; model != NULL; model = model->MUTnextModel ) {

            /* loop through all the instances of the model */
            for (here = model->MUTinstances; here != NULL ;
                    here=here->MUTnextInstance) {

		i = 0 ;
		if ((here->MUTind1->INDbrEq != 0) && (here->MUTind2->INDbrEq != 0)) {
			while (here->MUTbr1br2 != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MUTbr1br2 = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->MUTind2->INDbrEq != 0) && (here->MUTind1->INDbrEq != 0)) {
			while (here->MUTbr2br1 != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->MUTbr2br1 = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
	    }
	}
    return(OK);
}
#endif
