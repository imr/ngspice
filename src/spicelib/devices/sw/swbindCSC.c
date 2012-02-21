/**********
Author: 2012 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "swdefs.h"
#include "ngspice/sperror.h"

int
SWbindCSC(GENmodel *inModel, CKTcircuit *ckt)
{
    SWmodel *model = (SWmodel *)inModel;
    int i ;

    /*  loop through all the sw models */
    for( ; model != NULL; model = model->SWnextModel ) {
	SWinstance *here;

        /* loop through all the instances of the model */
        for (here = model->SWinstances; here != NULL ;
	    here = here->SWnextInstance) {

		i = 0 ;
		if ((here-> SWposNode != 0) && (here-> SWposNode != 0)) {
			while (here->SWposPosptr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->SWposPosptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here-> SWposNode != 0) && (here-> SWnegNode != 0)) {
			while (here->SWposNegptr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->SWposNegptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here-> SWnegNode != 0) && (here-> SWposNode != 0)) {
			while (here->SWnegPosptr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->SWnegPosptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here-> SWnegNode != 0) && (here-> SWnegNode != 0)) {
			while (here->SWnegNegptr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->SWnegNegptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
	}
    }
    return(OK);
}

int
SWbindCSCComplex(GENmodel *inModel, CKTcircuit *ckt)
{
    SWmodel *model = (SWmodel *)inModel;
    int i ;

    /*  loop through all the sw models */
    for( ; model != NULL; model = model->SWnextModel ) {
	SWinstance *here;

        /* loop through all the instances of the model */
        for (here = model->SWinstances; here != NULL ;
	    here = here->SWnextInstance) {

		i = 0 ;
		if ((here-> SWposNode != 0) && (here-> SWposNode != 0)) {
			while (here->SWposPosptr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->SWposPosptr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here-> SWposNode != 0) && (here-> SWnegNode != 0)) {
			while (here->SWposNegptr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->SWposNegptr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here-> SWnegNode != 0) && (here-> SWposNode != 0)) {
			while (here->SWnegPosptr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->SWnegPosptr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here-> SWnegNode != 0) && (here-> SWnegNode != 0)) {
			while (here->SWnegNegptr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->SWnegNegptr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
	}
    }
    return(OK);
}
