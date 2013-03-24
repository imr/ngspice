/**********
Author: 2012 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "capdefs.h"
#include "ngspice/sperror.h"

int
CAPbindklu(GENmodel *inModel, CKTcircuit *ckt)
{
    CAPmodel *model = (CAPmodel *)inModel;
    CAPinstance *here;
    int i ;

        /*  loop through all the capacitor models */
        for( ; model != NULL; model = model->CAPnextModel ) {

            /* loop through all the instances of the model */
            for (here = model->CAPinstances; here != NULL ;
                    here=here->CAPnextInstance) {

		i = 0 ;
		if ((here->CAPposNode != 0) && (here->CAPposNode != 0)) {
			while (here->CAPposPosptr != ckt->CKTmatrix->CKTkluBind_Sparse [i]) i ++ ;
			here->CAPposPosptr = ckt->CKTmatrix->CKTkluBind_KLU [i] ;
		}

		i = 0 ;
		if ((here->CAPnegNode != 0) && (here->CAPnegNode != 0)) {
			while (here->CAPnegNegptr != ckt->CKTmatrix->CKTkluBind_Sparse [i]) i ++ ;
			here->CAPnegNegptr = ckt->CKTmatrix->CKTkluBind_KLU [i] ;
		}

		i = 0 ;
		if ((here->CAPposNode != 0) && (here->CAPnegNode != 0)) {
			while (here->CAPposNegptr != ckt->CKTmatrix->CKTkluBind_Sparse [i]) i ++ ;
			here->CAPposNegptr = ckt->CKTmatrix->CKTkluBind_KLU [i] ;
		}

		i = 0 ;
		if ((here->CAPnegNode != 0) && (here->CAPposNode != 0)) {
			while (here->CAPnegPosptr != ckt->CKTmatrix->CKTkluBind_Sparse [i]) i ++ ;
			here->CAPnegPosptr = ckt->CKTmatrix->CKTkluBind_KLU [i] ;
		}
	    }
	}
    return(OK);
}

int
CAPbindkluComplex(GENmodel *inModel, CKTcircuit *ckt)
{
    CAPmodel *model = (CAPmodel *)inModel;
    CAPinstance *here;
    int i ;

        /*  loop through all the capacitor models */
        for( ; model != NULL; model = model->CAPnextModel ) {

            /* loop through all the instances of the model */
            for (here = model->CAPinstances; here != NULL ;
                    here=here->CAPnextInstance) {

		i = 0 ;
		if ((here->CAPposNode != 0) && (here->CAPposNode != 0)) {
			while (here->CAPposPosptr != ckt->CKTmatrix->CKTkluBind_KLU [i]) i ++ ;
			here->CAPposPosptr = ckt->CKTmatrix->CKTkluBind_KLU_Complex [i] ;
		}

		i = 0 ;
		if ((here->CAPnegNode != 0) && (here->CAPnegNode != 0)) {
			while (here->CAPnegNegptr != ckt->CKTmatrix->CKTkluBind_KLU [i]) i ++ ;
			here->CAPnegNegptr = ckt->CKTmatrix->CKTkluBind_KLU_Complex [i] ;
		}

		i = 0 ;
		if ((here->CAPposNode != 0) && (here->CAPnegNode != 0)) {
			while (here->CAPposNegptr != ckt->CKTmatrix->CKTkluBind_KLU [i]) i ++ ;
			here->CAPposNegptr = ckt->CKTmatrix->CKTkluBind_KLU_Complex [i] ;
		}

		i = 0 ;
		if ((here->CAPnegNode != 0) && (here->CAPposNode != 0)) {
			while (here->CAPnegPosptr != ckt->CKTmatrix->CKTkluBind_KLU [i]) i ++ ;
			here->CAPnegPosptr = ckt->CKTmatrix->CKTkluBind_KLU_Complex [i] ;
		}
	    }
	}
    return(OK);
}
