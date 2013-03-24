/**********
Author: 2012 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "resdefs.h"
#include "ngspice/sperror.h"

int
RESbindklu(GENmodel *inModel, CKTcircuit *ckt)
{
    RESmodel *model = (RESmodel *)inModel;
    int i ;

    /*  loop through all the resistor models */
    for( ; model != NULL; model = model->RESnextModel ) {
	RESinstance *here;

        /* loop through all the instances of the model */
        for (here = model->RESinstances; here != NULL ;
	    here = here->RESnextInstance) {

		i = 0 ;
		if ((here->RESposNode != 0) && (here->RESposNode != 0)) {
			while (here->RESposPosptr != ckt->CKTmatrix->CKTkluBind_Sparse [i]) i ++ ;
			here->RESposPosptr = ckt->CKTmatrix->CKTkluBind_KLU [i] ;
		}

		i = 0 ;
		if ((here->RESnegNode != 0) && (here->RESnegNode != 0)) {
			while (here->RESnegNegptr != ckt->CKTmatrix->CKTkluBind_Sparse [i]) i ++ ;
			here->RESnegNegptr = ckt->CKTmatrix->CKTkluBind_KLU [i] ;
		}

		i = 0 ;
		if ((here->RESposNode != 0) && (here->RESnegNode != 0)) {
			while (here->RESposNegptr != ckt->CKTmatrix->CKTkluBind_Sparse [i]) i ++ ;
			here->RESposNegptr = ckt->CKTmatrix->CKTkluBind_KLU [i] ;
		}

		i = 0 ;
		if ((here->RESnegNode != 0) && (here->RESposNode != 0)) {
			while (here->RESnegPosptr != ckt->CKTmatrix->CKTkluBind_Sparse [i]) i ++ ;
			here->RESnegPosptr = ckt->CKTmatrix->CKTkluBind_KLU [i] ;
		}
	}
    }
    return(OK);
}

int
RESbindkluComplex(GENmodel *inModel, CKTcircuit *ckt)
{
    RESmodel *model = (RESmodel *)inModel;
    int i ;

    /*  loop through all the resistor models */
    for( ; model != NULL; model = model->RESnextModel ) {
	RESinstance *here;

        /* loop through all the instances of the model */
        for (here = model->RESinstances; here != NULL ;
	    here = here->RESnextInstance) {

		i = 0 ;
		if ((here->RESposNode != 0) && (here->RESposNode != 0)) {
			while (here->RESposPosptr != ckt->CKTmatrix->CKTkluBind_KLU [i]) i ++ ;
			here->RESposPosptr = ckt->CKTmatrix->CKTkluBind_KLU_Complex [i] ;
		}

		i = 0 ;
		if ((here->RESnegNode != 0) && (here->RESnegNode != 0)) {
			while (here->RESnegNegptr != ckt->CKTmatrix->CKTkluBind_KLU [i]) i ++ ;
			here->RESnegNegptr = ckt->CKTmatrix->CKTkluBind_KLU_Complex [i] ;
		}

		i = 0 ;
		if ((here->RESposNode != 0) && (here->RESnegNode != 0)) {
			while (here->RESposNegptr != ckt->CKTmatrix->CKTkluBind_KLU [i]) i ++ ;
			here->RESposNegptr = ckt->CKTmatrix->CKTkluBind_KLU_Complex [i] ;
		}

		i = 0 ;
		if ((here->RESnegNode != 0) && (here->RESposNode != 0)) {
			while (here->RESnegPosptr != ckt->CKTmatrix->CKTkluBind_KLU [i]) i ++ ;
			here->RESnegPosptr = ckt->CKTmatrix->CKTkluBind_KLU_Complex [i] ;
		}
	}
    }
    return(OK);
}
