/**********
Author: 2012 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "diodefs.h"
#include "ngspice/sperror.h"

int
DIObindCSC(GENmodel *inModel, CKTcircuit *ckt)
{
    DIOmodel *model = (DIOmodel *)inModel;
    int i ;

    /*  loop through all the diode models */
    for( ; model != NULL; model = model->DIOnextModel ) {
	DIOinstance *here;

        /* loop through all the instances of the model */
        for (here = model->DIOinstances; here != NULL ;
	    here = here->DIOnextInstance) {

		i = 0 ;
		if ((here->DIOposNode != 0) && (here->DIOposPrimeNode != 0)) {
			while (here->DIOposPosPrimePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->DIOposPosPrimePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here->DIOnegNode != 0) && (here->DIOposPrimeNode != 0)) {
			while (here->DIOnegPosPrimePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->DIOnegPosPrimePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here->DIOposPrimeNode != 0) && (here->DIOposNode != 0)) {
			while (here->DIOposPrimePosPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->DIOposPrimePosPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here->DIOposPrimeNode != 0) && (here->DIOnegNode != 0)) {
			while (here->DIOposPrimeNegPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->DIOposPrimeNegPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here->DIOposNode != 0) && (here->DIOposNode != 0)) {
			while (here->DIOposPosPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->DIOposPosPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here->DIOnegNode != 0) && (here->DIOnegNode != 0)) {
			while (here->DIOnegNegPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->DIOnegNegPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here->DIOposPrimeNode != 0) && (here->DIOposPrimeNode != 0)) {
			while (here->DIOposPrimePosPrimePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->DIOposPrimePosPrimePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
	}
    }
    return(OK);
}

int
DIObindCSCComplex(GENmodel *inModel, CKTcircuit *ckt)
{
    DIOmodel *model = (DIOmodel *)inModel;
    int i ;

    /*  loop through all the diode models */
    for( ; model != NULL; model = model->DIOnextModel ) {
	DIOinstance *here;

        /* loop through all the instances of the model */
        for (here = model->DIOinstances; here != NULL ;
	    here = here->DIOnextInstance) {

		i = 0 ;
		if ((here->DIOposNode != 0) && (here->DIOposPrimeNode != 0)) {
			while (here->DIOposPosPrimePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->DIOposPosPrimePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here->DIOnegNode != 0) && (here->DIOposPrimeNode != 0)) {
			while (here->DIOnegPosPrimePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->DIOnegPosPrimePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here->DIOposPrimeNode != 0) && (here->DIOposNode != 0)) {
			while (here->DIOposPrimePosPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->DIOposPrimePosPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here->DIOposPrimeNode != 0) && (here->DIOnegNode != 0)) {
			while (here->DIOposPrimeNegPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->DIOposPrimeNegPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here->DIOposNode != 0) && (here->DIOposNode != 0)) {
			while (here->DIOposPosPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->DIOposPosPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here->DIOnegNode != 0) && (here->DIOnegNode != 0)) {
			while (here->DIOnegNegPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->DIOnegNegPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here->DIOposPrimeNode != 0) && (here->DIOposPrimeNode != 0)) {
			while (here->DIOposPrimePosPrimePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->DIOposPrimePosPrimePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
	}
    }
    return(OK);
}
