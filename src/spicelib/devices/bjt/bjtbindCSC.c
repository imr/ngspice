/**********
Author: 2012 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "bjtdefs.h"
#include "ngspice/sperror.h"

int
BJTbindCSC(GENmodel *inModel, CKTcircuit *ckt)
{
    BJTmodel *model = (BJTmodel *)inModel;
    int i ;

    /*  loop through all the bjt models */
    for( ; model != NULL; model = model->BJTnextModel ) {
	BJTinstance *here;

        /* loop through all the instances of the model */
        for (here = model->BJTinstances; here != NULL ;
	    here = here->BJTnextInstance) {

		i = 0 ;
		if ((here->BJTcolNode != 0) && (here->BJTcolPrimeNode != 0)) {
			while (here->BJTcolColPrimePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BJTcolColPrimePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->BJTbaseNode != 0) && (here->BJTbasePrimeNode != 0)) {
			while (here->BJTbaseBasePrimePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BJTbaseBasePrimePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->BJTemitNode != 0) && (here->BJTemitPrimeNode != 0)) {
			while (here->BJTemitEmitPrimePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BJTemitEmitPrimePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->BJTcolPrimeNode != 0) && (here->BJTcolNode != 0)) {
			while (here->BJTcolPrimeColPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BJTcolPrimeColPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->BJTcolPrimeNode != 0) && (here->BJTbasePrimeNode != 0)) {
			while (here->BJTcolPrimeBasePrimePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BJTcolPrimeBasePrimePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->BJTcolPrimeNode != 0) && (here->BJTemitPrimeNode != 0)) {
			while (here->BJTcolPrimeEmitPrimePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BJTcolPrimeEmitPrimePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->BJTbasePrimeNode != 0) && (here->BJTbaseNode != 0)) {
			while (here->BJTbasePrimeBasePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BJTbasePrimeBasePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->BJTbasePrimeNode != 0) && (here->BJTcolPrimeNode != 0)) {
			while (here->BJTbasePrimeColPrimePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BJTbasePrimeColPrimePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->BJTbasePrimeNode != 0) && (here->BJTemitPrimeNode != 0)) {
			while (here->BJTbasePrimeEmitPrimePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BJTbasePrimeEmitPrimePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->BJTemitPrimeNode != 0) && (here->BJTemitNode != 0)) {
			while (here->BJTemitPrimeEmitPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BJTemitPrimeEmitPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->BJTemitPrimeNode != 0) && (here->BJTcolPrimeNode != 0)) {
			while (here->BJTemitPrimeColPrimePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BJTemitPrimeColPrimePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->BJTemitPrimeNode != 0) && (here->BJTbasePrimeNode != 0)) {
			while (here->BJTemitPrimeBasePrimePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BJTemitPrimeBasePrimePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->BJTcolNode != 0) && (here->BJTcolNode != 0)) {
			while (here->BJTcolColPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BJTcolColPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->BJTbaseNode != 0) && (here->BJTbaseNode != 0)) {
			while (here->BJTbaseBasePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BJTbaseBasePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->BJTemitNode != 0) && (here->BJTemitNode != 0)) {
			while (here->BJTemitEmitPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BJTemitEmitPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->BJTcolPrimeNode != 0) && (here->BJTcolPrimeNode != 0)) {
			while (here->BJTcolPrimeColPrimePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BJTcolPrimeColPrimePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->BJTbasePrimeNode != 0) && (here->BJTbasePrimeNode != 0)) {
			while (here->BJTbasePrimeBasePrimePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BJTbasePrimeBasePrimePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->BJTemitPrimeNode != 0) && (here->BJTemitPrimeNode != 0)) {
			while (here->BJTemitPrimeEmitPrimePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BJTemitPrimeEmitPrimePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->BJTsubstNode != 0) && (here->BJTsubstNode != 0)) {
			while (here->BJTsubstSubstPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BJTsubstSubstPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

            if (model -> BJTsubs == LATERAL) {
              here -> BJTsubstConNode = here -> BJTbasePrimeNode;
              here -> BJTsubstConSubstConPtr = here -> BJTbasePrimeBasePrimePtr;
            } else {
              here -> BJTsubstConNode = here -> BJTcolPrimeNode;
              here -> BJTsubstConSubstConPtr = here -> BJTcolPrimeColPrimePtr;
            }

		i = 0 ;
		if ((here->BJTsubstConNode != 0) && (here->BJTsubstNode != 0)) {
			while (here->BJTsubstConSubstPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BJTsubstConSubstPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->BJTsubstNode != 0) && (here->BJTsubstConNode != 0)) {
			while (here->BJTsubstSubstConPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BJTsubstSubstConPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->BJTbaseNode != 0) && (here->BJTcolPrimeNode != 0)) {
			while (here->BJTbaseColPrimePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BJTbaseColPrimePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here->BJTcolPrimeNode != 0) && (here->BJTbaseNode != 0)) {
			while (here->BJTcolPrimeBasePtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->BJTcolPrimeBasePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
	}
    }
    return(OK);
}

int
BJTbindCSCComplex(GENmodel *inModel, CKTcircuit *ckt)
{
    BJTmodel *model = (BJTmodel *)inModel;
    int i ;

    /*  loop through all the bjt models */
    for( ; model != NULL; model = model->BJTnextModel ) {
	BJTinstance *here;

        /* loop through all the instances of the model */
        for (here = model->BJTinstances; here != NULL ;
	    here = here->BJTnextInstance) {

		i = 0 ;
		if ((here->BJTcolNode != 0) && (here->BJTcolPrimeNode != 0)) {
			while (here->BJTcolColPrimePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BJTcolColPrimePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->BJTbaseNode != 0) && (here->BJTbasePrimeNode != 0)) {
			while (here->BJTbaseBasePrimePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BJTbaseBasePrimePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->BJTemitNode != 0) && (here->BJTemitPrimeNode != 0)) {
			while (here->BJTemitEmitPrimePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BJTemitEmitPrimePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->BJTcolPrimeNode != 0) && (here->BJTcolNode != 0)) {
			while (here->BJTcolPrimeColPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BJTcolPrimeColPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->BJTcolPrimeNode != 0) && (here->BJTbasePrimeNode != 0)) {
			while (here->BJTcolPrimeBasePrimePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BJTcolPrimeBasePrimePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->BJTcolPrimeNode != 0) && (here->BJTemitPrimeNode != 0)) {
			while (here->BJTcolPrimeEmitPrimePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BJTcolPrimeEmitPrimePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->BJTbasePrimeNode != 0) && (here->BJTbaseNode != 0)) {
			while (here->BJTbasePrimeBasePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BJTbasePrimeBasePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->BJTbasePrimeNode != 0) && (here->BJTcolPrimeNode != 0)) {
			while (here->BJTbasePrimeColPrimePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BJTbasePrimeColPrimePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->BJTbasePrimeNode != 0) && (here->BJTemitPrimeNode != 0)) {
			while (here->BJTbasePrimeEmitPrimePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BJTbasePrimeEmitPrimePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->BJTemitPrimeNode != 0) && (here->BJTemitNode != 0)) {
			while (here->BJTemitPrimeEmitPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BJTemitPrimeEmitPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->BJTemitPrimeNode != 0) && (here->BJTcolPrimeNode != 0)) {
			while (here->BJTemitPrimeColPrimePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BJTemitPrimeColPrimePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->BJTemitPrimeNode != 0) && (here->BJTbasePrimeNode != 0)) {
			while (here->BJTemitPrimeBasePrimePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BJTemitPrimeBasePrimePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->BJTcolNode != 0) && (here->BJTcolNode != 0)) {
			while (here->BJTcolColPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BJTcolColPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->BJTbaseNode != 0) && (here->BJTbaseNode != 0)) {
			while (here->BJTbaseBasePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BJTbaseBasePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->BJTemitNode != 0) && (here->BJTemitNode != 0)) {
			while (here->BJTemitEmitPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BJTemitEmitPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->BJTcolPrimeNode != 0) && (here->BJTcolPrimeNode != 0)) {
			while (here->BJTcolPrimeColPrimePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BJTcolPrimeColPrimePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->BJTbasePrimeNode != 0) && (here->BJTbasePrimeNode != 0)) {
			while (here->BJTbasePrimeBasePrimePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BJTbasePrimeBasePrimePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->BJTemitPrimeNode != 0) && (here->BJTemitPrimeNode != 0)) {
			while (here->BJTemitPrimeEmitPrimePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BJTemitPrimeEmitPrimePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->BJTsubstNode != 0) && (here->BJTsubstNode != 0)) {
			while (here->BJTsubstSubstPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BJTsubstSubstPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

            if (model -> BJTsubs == LATERAL) {
              here -> BJTsubstConNode = here -> BJTbasePrimeNode;
              here -> BJTsubstConSubstConPtr = here -> BJTbasePrimeBasePrimePtr;
            } else {
              here -> BJTsubstConNode = here -> BJTcolPrimeNode;
              here -> BJTsubstConSubstConPtr = here -> BJTcolPrimeColPrimePtr;
            }

		i = 0 ;
		if ((here->BJTsubstConNode != 0) && (here->BJTsubstNode != 0)) {
			while (here->BJTsubstConSubstPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BJTsubstConSubstPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->BJTsubstNode != 0) && (here->BJTsubstConNode != 0)) {
			while (here->BJTsubstSubstConPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BJTsubstSubstConPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->BJTbaseNode != 0) && (here->BJTcolPrimeNode != 0)) {
			while (here->BJTbaseColPrimePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BJTbaseColPrimePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here->BJTcolPrimeNode != 0) && (here->BJTbaseNode != 0)) {
			while (here->BJTcolPrimeBasePtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->BJTcolPrimeBasePtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
	}
    }
    return(OK);
}

int
BJTbindCSCComplexToReal (GENmodel *inModel, CKTcircuit *ckt)
{
    BJTmodel *model = (BJTmodel *)inModel ;
    BJTinstance *here ;
    int i ;

    /*  loop through all the bjt models */
    for ( ; model != NULL ; model = model->BJTnextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->BJTinstances ; here != NULL ; here = here->BJTnextInstance)
        {
            i = 0 ;
            if ((here->BJTcolNode != 0) && (here->BJTcolPrimeNode != 0))
            {
                while (here->BJTcolColPrimePtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BJTcolColPrimePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BJTbaseNode != 0) && (here->BJTbasePrimeNode != 0))
            {
                while (here->BJTbaseBasePrimePtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BJTbaseBasePrimePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BJTemitNode != 0) && (here->BJTemitPrimeNode != 0))
            {
                while (here->BJTemitEmitPrimePtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BJTemitEmitPrimePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BJTcolPrimeNode != 0) && (here->BJTcolNode != 0))
            {
                while (here->BJTcolPrimeColPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BJTcolPrimeColPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BJTcolPrimeNode != 0) && (here->BJTbasePrimeNode != 0))
            {
                while (here->BJTcolPrimeBasePrimePtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BJTcolPrimeBasePrimePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BJTcolPrimeNode != 0) && (here->BJTemitPrimeNode != 0))
            {
                while (here->BJTcolPrimeEmitPrimePtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BJTcolPrimeEmitPrimePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BJTbasePrimeNode != 0) && (here->BJTbaseNode != 0))
            {
                while (here->BJTbasePrimeBasePtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BJTbasePrimeBasePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BJTbasePrimeNode != 0) && (here->BJTcolPrimeNode != 0))
            {
                while (here->BJTbasePrimeColPrimePtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BJTbasePrimeColPrimePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BJTbasePrimeNode != 0) && (here->BJTemitPrimeNode != 0))
            {
                while (here->BJTbasePrimeEmitPrimePtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BJTbasePrimeEmitPrimePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BJTemitPrimeNode != 0) && (here->BJTemitNode != 0))
            {
                while (here->BJTemitPrimeEmitPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BJTemitPrimeEmitPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BJTemitPrimeNode != 0) && (here->BJTcolPrimeNode != 0))
            {
                while (here->BJTemitPrimeColPrimePtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BJTemitPrimeColPrimePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BJTemitPrimeNode != 0) && (here->BJTbasePrimeNode != 0))
            {
                while (here->BJTemitPrimeBasePrimePtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BJTemitPrimeBasePrimePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BJTcolNode != 0) && (here->BJTcolNode != 0))
            {
                while (here->BJTcolColPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BJTcolColPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BJTbaseNode != 0) && (here->BJTbaseNode != 0))
            {
                while (here->BJTbaseBasePtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BJTbaseBasePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BJTemitNode != 0) && (here->BJTemitNode != 0))
            {
                while (here->BJTemitEmitPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BJTemitEmitPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BJTcolPrimeNode != 0) && (here->BJTcolPrimeNode != 0))
            {
                while (here->BJTcolPrimeColPrimePtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BJTcolPrimeColPrimePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BJTbasePrimeNode != 0) && (here->BJTbasePrimeNode != 0))
            {
                while (here->BJTbasePrimeBasePrimePtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BJTbasePrimeBasePrimePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BJTemitPrimeNode != 0) && (here->BJTemitPrimeNode != 0))
            {
                while (here->BJTemitPrimeEmitPrimePtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BJTemitPrimeEmitPrimePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BJTsubstNode != 0) && (here->BJTsubstNode != 0))
            {
                while (here->BJTsubstSubstPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BJTsubstSubstPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BJTsubstConNode != 0) && (here->BJTsubstNode != 0))
            {
                while (here->BJTsubstConSubstPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BJTsubstConSubstPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BJTsubstNode != 0) && (here->BJTsubstConNode != 0))
            {
                while (here->BJTsubstSubstConPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BJTsubstSubstConPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BJTbaseNode != 0) && (here->BJTcolPrimeNode != 0))
            {
                while (here->BJTbaseColPrimePtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BJTbaseColPrimePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->BJTcolPrimeNode != 0) && (here->BJTbaseNode != 0))
            {
                while (here->BJTcolPrimeBasePtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->BJTcolPrimeBasePtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }
        }
    }

    return (OK) ;
}