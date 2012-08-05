/**********
Author: 2012 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "txldefs.h"
#include "ngspice/sperror.h"

int
TXLbindCSC(GENmodel *inModel, CKTcircuit *ckt)
{
    TXLmodel *model = (TXLmodel *)inModel;
    int i ;

    /*  loop through all the txl models */
    for( ; model != NULL; model = model->TXLnextModel ) {
	TXLinstance *here;

        /* loop through all the instances of the model */
        for (here = model->TXLinstances; here != NULL ;
	    here = here->TXLnextInstance) {

		i = 0 ;
		if ((here-> TXLposNode != 0) && (here-> TXLposNode != 0)) {
			while (here->TXLposPosptr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->TXLposPosptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here-> TXLposNode != 0) && (here-> TXLnegNode != 0)) {
			while (here->TXLposNegptr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->TXLposNegptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here-> TXLnegNode != 0) && (here-> TXLposNode != 0)) {
			while (here->TXLnegPosptr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->TXLnegPosptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here-> TXLnegNode != 0) && (here-> TXLnegNode != 0)) {
			while (here->TXLnegNegptr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->TXLnegNegptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here-> TXLibr1 != 0) && (here-> TXLposNode != 0)) {
			while (here->TXLibr1Posptr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->TXLibr1Posptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here-> TXLibr2 != 0) && (here-> TXLnegNode != 0)) {
			while (here->TXLibr2Negptr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->TXLibr2Negptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here-> TXLnegNode != 0) && (here-> TXLibr2 != 0)) {
			while (here->TXLnegIbr2ptr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->TXLnegIbr2ptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here-> TXLposNode != 0) && (here-> TXLibr1 != 0)) {
			while (here->TXLposIbr1ptr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->TXLposIbr1ptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here-> TXLibr1 != 0) && (here-> TXLibr1 != 0)) {
			while (here->TXLibr1Ibr1ptr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->TXLibr1Ibr1ptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here-> TXLibr2 != 0) && (here-> TXLibr2 != 0)) {
			while (here->TXLibr2Ibr2ptr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->TXLibr2Ibr2ptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here-> TXLibr1 != 0) && (here-> TXLnegNode != 0)) {
			while (here->TXLibr1Negptr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->TXLibr1Negptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here-> TXLibr2 != 0) && (here-> TXLposNode != 0)) {
			while (here->TXLibr2Posptr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->TXLibr2Posptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here-> TXLibr1 != 0) && (here-> TXLibr2 != 0)) {
			while (here->TXLibr1Ibr2ptr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->TXLibr1Ibr2ptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here-> TXLibr2 != 0) && (here-> TXLibr1 != 0)) {
			while (here->TXLibr2Ibr1ptr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->TXLibr2Ibr1ptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
	}
    }
    return(OK);
}

int
TXLbindCSCComplex(GENmodel *inModel, CKTcircuit *ckt)
{
    TXLmodel *model = (TXLmodel *)inModel;
    int i ;

    /*  loop through all the txl models */
    for( ; model != NULL; model = model->TXLnextModel ) {
	TXLinstance *here;

        /* loop through all the instances of the model */
        for (here = model->TXLinstances; here != NULL ;
	    here = here->TXLnextInstance) {

		i = 0 ;
		if ((here-> TXLposNode != 0) && (here-> TXLposNode != 0)) {
			while (here->TXLposPosptr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->TXLposPosptr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here-> TXLposNode != 0) && (here-> TXLnegNode != 0)) {
			while (here->TXLposNegptr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->TXLposNegptr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here-> TXLnegNode != 0) && (here-> TXLposNode != 0)) {
			while (here->TXLnegPosptr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->TXLnegPosptr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here-> TXLnegNode != 0) && (here-> TXLnegNode != 0)) {
			while (here->TXLnegNegptr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->TXLnegNegptr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here-> TXLibr1 != 0) && (here-> TXLposNode != 0)) {
			while (here->TXLibr1Posptr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->TXLibr1Posptr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here-> TXLibr2 != 0) && (here-> TXLnegNode != 0)) {
			while (here->TXLibr2Negptr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->TXLibr2Negptr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here-> TXLnegNode != 0) && (here-> TXLibr2 != 0)) {
			while (here->TXLnegIbr2ptr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->TXLnegIbr2ptr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here-> TXLposNode != 0) && (here-> TXLibr1 != 0)) {
			while (here->TXLposIbr1ptr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->TXLposIbr1ptr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here-> TXLibr1 != 0) && (here-> TXLibr1 != 0)) {
			while (here->TXLibr1Ibr1ptr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->TXLibr1Ibr1ptr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here-> TXLibr2 != 0) && (here-> TXLibr2 != 0)) {
			while (here->TXLibr2Ibr2ptr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->TXLibr2Ibr2ptr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here-> TXLibr1 != 0) && (here-> TXLnegNode != 0)) {
			while (here->TXLibr1Negptr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->TXLibr1Negptr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here-> TXLibr2 != 0) && (here-> TXLposNode != 0)) {
			while (here->TXLibr2Posptr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->TXLibr2Posptr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here-> TXLibr1 != 0) && (here-> TXLibr2 != 0)) {
			while (here->TXLibr1Ibr2ptr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->TXLibr1Ibr2ptr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here-> TXLibr2 != 0) && (here-> TXLibr1 != 0)) {
			while (here->TXLibr2Ibr1ptr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->TXLibr2Ibr1ptr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
	}
    }
    return(OK);
}

int
TXLbindCSCComplexToReal (GENmodel *inModel, CKTcircuit *ckt)
{
    TXLmodel *model = (TXLmodel *)inModel ;
    TXLinstance *here ;
    int i ;

    /*  loop through all the txl models */
    for ( ; model != NULL ; model = model->TXLnextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->TXLinstances ; here != NULL ; here = here->TXLnextInstance)
        {
            i = 0 ;
            if ((here->TXLposNode != 0) && (here->TXLposNode != 0))
            {
                while (here->TXLposPosptr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->TXLposPosptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->TXLposNode != 0) && (here->TXLnegNode != 0))
            {
                while (here->TXLposNegptr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->TXLposNegptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->TXLnegNode != 0) && (here->TXLposNode != 0))
            {
                while (here->TXLnegPosptr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->TXLnegPosptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->TXLnegNode != 0) && (here->TXLnegNode != 0))
            {
                while (here->TXLnegNegptr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->TXLnegNegptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->TXLibr1 != 0) && (here->TXLposNode != 0))
            {
                while (here->TXLibr1Posptr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->TXLibr1Posptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->TXLibr2 != 0) && (here->TXLnegNode != 0))
            {
                while (here->TXLibr2Negptr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->TXLibr2Negptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->TXLnegNode != 0) && (here->TXLibr2 != 0))
            {
                while (here->TXLnegIbr2ptr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->TXLnegIbr2ptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->TXLposNode != 0) && (here->TXLibr1 != 0))
            {
                while (here->TXLposIbr1ptr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->TXLposIbr1ptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->TXLibr1 != 0) && (here->TXLibr1 != 0))
            {
                while (here->TXLibr1Ibr1ptr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->TXLibr1Ibr1ptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->TXLibr2 != 0) && (here->TXLibr2 != 0))
            {
                while (here->TXLibr2Ibr2ptr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->TXLibr2Ibr2ptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->TXLibr1 != 0) && (here->TXLnegNode != 0))
            {
                while (here->TXLibr1Negptr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->TXLibr1Negptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->TXLibr2 != 0) && (here->TXLposNode != 0))
            {
                while (here->TXLibr2Posptr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->TXLibr2Posptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->TXLibr1 != 0) && (here->TXLibr2 != 0))
            {
                while (here->TXLibr1Ibr2ptr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->TXLibr1Ibr2ptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->TXLibr2 != 0) && (here->TXLibr1 != 0))
            {
                while (here->TXLibr2Ibr1ptr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->TXLibr2Ibr1ptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }
        }
    }

    return (OK) ;
}