/**********
Author: 2012 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "resdefs.h"
#include "ngspice/sperror.h"

int
RESbindCSC(GENmodel *inModel, CKTcircuit *ckt)
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
			while (here->RESposPosptr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->RESposPosptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->RESnegNode != 0) && (here->RESnegNode != 0)) {
			while (here->RESnegNegptr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->RESnegNegptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->RESposNode != 0) && (here->RESnegNode != 0)) {
			while (here->RESposNegptr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->RESposNegptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->RESnegNode != 0) && (here->RESposNode != 0)) {
			while (here->RESnegPosptr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->RESnegPosptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
	}
    }
    return(OK);
}

int
RESbindCSCComplex(GENmodel *inModel, CKTcircuit *ckt)
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
			while (here->RESposPosptr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->RESposPosptr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->RESnegNode != 0) && (here->RESnegNode != 0)) {
			while (here->RESnegNegptr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->RESnegNegptr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->RESposNode != 0) && (here->RESnegNode != 0)) {
			while (here->RESposNegptr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->RESposNegptr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->RESnegNode != 0) && (here->RESposNode != 0)) {
			while (here->RESnegPosptr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->RESnegPosptr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
	}
    }
    return(OK);
}

int
RESbindCSCComplexToReal (GENmodel *inModel, CKTcircuit *ckt)
{
    RESmodel *model = (RESmodel *)inModel ;
    RESinstance *here ;
    int i ;

    /*  loop through all the resistor models */
    for ( ; model != NULL ; model = model->RESnextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->RESinstances ; here != NULL ; here = here->RESnextInstance)
        {
            i = 0 ;
            if ((here->RESposNode != 0) && (here->RESposNode != 0))
            {
                while (here->RESposPosptr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->RESposPosptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->RESnegNode != 0) && (here->RESnegNode != 0))
            {
                while (here->RESnegNegptr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->RESnegNegptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->RESposNode != 0) && (here->RESnegNode != 0))
            {
                while (here->RESposNegptr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->RESposNegptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->RESnegNode != 0) && (here->RESposNode != 0))
            {
                while (here->RESnegPosptr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->RESnegPosptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }
        }
    }

    return (OK) ;
}
