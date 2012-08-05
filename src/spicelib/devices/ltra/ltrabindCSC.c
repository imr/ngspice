/**********
Author: 2012 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ltradefs.h"
#include "ngspice/sperror.h"

int
LTRAbindCSC(GENmodel *inModel, CKTcircuit *ckt)
{
    LTRAmodel *model = (LTRAmodel *)inModel;
    int i ;

    /*  loop through all the ltra models */
    for( ; model != NULL; model = model->LTRAnextModel ) {
	LTRAinstance *here;

        /* loop through all the instances of the model */
        for (here = model->LTRAinstances; here != NULL ;
	    here = here->LTRAnextInstance) {

		i = 0 ;
		if ((here-> LTRAbrEq1 != 0) && (here-> LTRAposNode1 != 0)) {
			while (here->LTRAibr1Pos1Ptr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->LTRAibr1Pos1Ptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> LTRAbrEq1 != 0) && (here-> LTRAnegNode1 != 0)) {
			while (here->LTRAibr1Neg1Ptr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->LTRAibr1Neg1Ptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> LTRAbrEq1 != 0) && (here-> LTRAposNode2 != 0)) {
			while (here->LTRAibr1Pos2Ptr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->LTRAibr1Pos2Ptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> LTRAbrEq1 != 0) && (here-> LTRAnegNode2 != 0)) {
			while (here->LTRAibr1Neg2Ptr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->LTRAibr1Neg2Ptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> LTRAbrEq1 != 0) && (here-> LTRAbrEq1 != 0)) {
			while (here->LTRAibr1Ibr1Ptr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->LTRAibr1Ibr1Ptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> LTRAbrEq1 != 0) && (here-> LTRAbrEq2 != 0)) {
			while (here->LTRAibr1Ibr2Ptr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->LTRAibr1Ibr2Ptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> LTRAbrEq2 != 0) && (here-> LTRAposNode1 != 0)) {
			while (here->LTRAibr2Pos1Ptr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->LTRAibr2Pos1Ptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> LTRAbrEq2 != 0) && (here-> LTRAnegNode1 != 0)) {
			while (here->LTRAibr2Neg1Ptr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->LTRAibr2Neg1Ptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> LTRAbrEq2 != 0) && (here-> LTRAposNode2 != 0)) {
			while (here->LTRAibr2Pos2Ptr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->LTRAibr2Pos2Ptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> LTRAbrEq2 != 0) && (here-> LTRAnegNode2 != 0)) {
			while (here->LTRAibr2Neg2Ptr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->LTRAibr2Neg2Ptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> LTRAbrEq2 != 0) && (here-> LTRAbrEq1 != 0)) {
			while (here->LTRAibr2Ibr1Ptr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->LTRAibr2Ibr1Ptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> LTRAbrEq2 != 0) && (here-> LTRAbrEq2 != 0)) {
			while (here->LTRAibr2Ibr2Ptr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->LTRAibr2Ibr2Ptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> LTRAposNode1 != 0) && (here-> LTRAbrEq1 != 0)) {
			while (here->LTRApos1Ibr1Ptr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->LTRApos1Ibr1Ptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> LTRAnegNode1 != 0) && (here-> LTRAbrEq1 != 0)) {
			while (here->LTRAneg1Ibr1Ptr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->LTRAneg1Ibr1Ptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> LTRAposNode2 != 0) && (here-> LTRAbrEq2 != 0)) {
			while (here->LTRApos2Ibr2Ptr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->LTRApos2Ibr2Ptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> LTRAnegNode2 != 0) && (here-> LTRAbrEq2 != 0)) {
			while (here->LTRAneg2Ibr2Ptr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->LTRAneg2Ibr2Ptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> LTRAposNode1 != 0) && (here-> LTRAposNode1 != 0)) {
			while (here->LTRApos1Pos1Ptr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->LTRApos1Pos1Ptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> LTRAnegNode1 != 0) && (here-> LTRAnegNode1 != 0)) {
			while (here->LTRAneg1Neg1Ptr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->LTRAneg1Neg1Ptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> LTRAposNode2 != 0) && (here-> LTRAposNode2 != 0)) {
			while (here->LTRApos2Pos2Ptr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->LTRApos2Pos2Ptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

		i = 0 ;
		if ((here-> LTRAnegNode2 != 0) && (here-> LTRAnegNode2 != 0)) {
			while (here->LTRAneg2Neg2Ptr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->LTRAneg2Neg2Ptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
	}
    }
    return(OK);
}

int
LTRAbindCSCComplex(GENmodel *inModel, CKTcircuit *ckt)
{
    LTRAmodel *model = (LTRAmodel *)inModel;
    int i ;

    /*  loop through all the ltra models */
    for( ; model != NULL; model = model->LTRAnextModel ) {
	LTRAinstance *here;

        /* loop through all the instances of the model */
        for (here = model->LTRAinstances; here != NULL ;
	    here = here->LTRAnextInstance) {

		i = 0 ;
		if ((here-> LTRAbrEq1 != 0) && (here-> LTRAposNode1 != 0)) {
			while (here->LTRAibr1Pos1Ptr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->LTRAibr1Pos1Ptr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> LTRAbrEq1 != 0) && (here-> LTRAnegNode1 != 0)) {
			while (here->LTRAibr1Neg1Ptr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->LTRAibr1Neg1Ptr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> LTRAbrEq1 != 0) && (here-> LTRAposNode2 != 0)) {
			while (here->LTRAibr1Pos2Ptr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->LTRAibr1Pos2Ptr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> LTRAbrEq1 != 0) && (here-> LTRAnegNode2 != 0)) {
			while (here->LTRAibr1Neg2Ptr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->LTRAibr1Neg2Ptr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> LTRAbrEq1 != 0) && (here-> LTRAbrEq1 != 0)) {
			while (here->LTRAibr1Ibr1Ptr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->LTRAibr1Ibr1Ptr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> LTRAbrEq1 != 0) && (here-> LTRAbrEq2 != 0)) {
			while (here->LTRAibr1Ibr2Ptr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->LTRAibr1Ibr2Ptr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> LTRAbrEq2 != 0) && (here-> LTRAposNode1 != 0)) {
			while (here->LTRAibr2Pos1Ptr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->LTRAibr2Pos1Ptr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> LTRAbrEq2 != 0) && (here-> LTRAnegNode1 != 0)) {
			while (here->LTRAibr2Neg1Ptr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->LTRAibr2Neg1Ptr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> LTRAbrEq2 != 0) && (here-> LTRAposNode2 != 0)) {
			while (here->LTRAibr2Pos2Ptr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->LTRAibr2Pos2Ptr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> LTRAbrEq2 != 0) && (here-> LTRAnegNode2 != 0)) {
			while (here->LTRAibr2Neg2Ptr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->LTRAibr2Neg2Ptr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> LTRAbrEq2 != 0) && (here-> LTRAbrEq1 != 0)) {
			while (here->LTRAibr2Ibr1Ptr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->LTRAibr2Ibr1Ptr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> LTRAbrEq2 != 0) && (here-> LTRAbrEq2 != 0)) {
			while (here->LTRAibr2Ibr2Ptr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->LTRAibr2Ibr2Ptr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> LTRAposNode1 != 0) && (here-> LTRAbrEq1 != 0)) {
			while (here->LTRApos1Ibr1Ptr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->LTRApos1Ibr1Ptr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> LTRAnegNode1 != 0) && (here-> LTRAbrEq1 != 0)) {
			while (here->LTRAneg1Ibr1Ptr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->LTRAneg1Ibr1Ptr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> LTRAposNode2 != 0) && (here-> LTRAbrEq2 != 0)) {
			while (here->LTRApos2Ibr2Ptr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->LTRApos2Ibr2Ptr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> LTRAnegNode2 != 0) && (here-> LTRAbrEq2 != 0)) {
			while (here->LTRAneg2Ibr2Ptr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->LTRAneg2Ibr2Ptr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> LTRAposNode1 != 0) && (here-> LTRAposNode1 != 0)) {
			while (here->LTRApos1Pos1Ptr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->LTRApos1Pos1Ptr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> LTRAnegNode1 != 0) && (here-> LTRAnegNode1 != 0)) {
			while (here->LTRAneg1Neg1Ptr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->LTRAneg1Neg1Ptr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> LTRAposNode2 != 0) && (here-> LTRAposNode2 != 0)) {
			while (here->LTRApos2Pos2Ptr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->LTRApos2Pos2Ptr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

		i = 0 ;
		if ((here-> LTRAnegNode2 != 0) && (here-> LTRAnegNode2 != 0)) {
			while (here->LTRAneg2Neg2Ptr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->LTRAneg2Neg2Ptr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
	}
    }
    return(OK);
}

int
LTRAbindCSCComplexToReal (GENmodel *inModel, CKTcircuit *ckt)
{
    LTRAmodel *model = (LTRAmodel *)inModel ;
    LTRAinstance *here ;
    int i ;

    /*  loop through all the TransmissionLine models */
    for ( ; model != NULL ; model = model->LTRAnextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->LTRAinstances ; here != NULL ; here = here->LTRAnextInstance)
        {
            i = 0 ;
            if ((here->LTRAbrEq1 != 0) && (here->LTRAposNode1 != 0))
            {
                while (here->LTRAibr1Pos1Ptr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->LTRAibr1Pos1Ptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->LTRAbrEq1 != 0) && (here->LTRAnegNode1 != 0))
            {
                while (here->LTRAibr1Neg1Ptr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->LTRAibr1Neg1Ptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->LTRAbrEq1 != 0) && (here->LTRAposNode2 != 0))
            {
                while (here->LTRAibr1Pos2Ptr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->LTRAibr1Pos2Ptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->LTRAbrEq1 != 0) && (here->LTRAnegNode2 != 0))
            {
                while (here->LTRAibr1Neg2Ptr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->LTRAibr1Neg2Ptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->LTRAbrEq1 != 0) && (here->LTRAbrEq1 != 0))
            {
                while (here->LTRAibr1Ibr1Ptr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->LTRAibr1Ibr1Ptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->LTRAbrEq1 != 0) && (here->LTRAbrEq2 != 0))
            {
                while (here->LTRAibr1Ibr2Ptr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->LTRAibr1Ibr2Ptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->LTRAbrEq2 != 0) && (here->LTRAposNode1 != 0))
            {
                while (here->LTRAibr2Pos1Ptr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->LTRAibr2Pos1Ptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->LTRAbrEq2 != 0) && (here->LTRAnegNode1 != 0))
            {
                while (here->LTRAibr2Neg1Ptr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->LTRAibr2Neg1Ptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->LTRAbrEq2 != 0) && (here->LTRAposNode2 != 0))
            {
                while (here->LTRAibr2Pos2Ptr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->LTRAibr2Pos2Ptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->LTRAbrEq2 != 0) && (here->LTRAnegNode2 != 0))
            {
                while (here->LTRAibr2Neg2Ptr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->LTRAibr2Neg2Ptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->LTRAbrEq2 != 0) && (here->LTRAbrEq1 != 0))
            {
                while (here->LTRAibr2Ibr1Ptr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->LTRAibr2Ibr1Ptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->LTRAbrEq2 != 0) && (here->LTRAbrEq2 != 0))
            {
                while (here->LTRAibr2Ibr2Ptr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->LTRAibr2Ibr2Ptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->LTRAposNode1 != 0) && (here->LTRAbrEq1 != 0))
            {
                while (here->LTRApos1Ibr1Ptr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->LTRApos1Ibr1Ptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->LTRAnegNode1 != 0) && (here->LTRAbrEq1 != 0))
            {
                while (here->LTRAneg1Ibr1Ptr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->LTRAneg1Ibr1Ptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->LTRAposNode2 != 0) && (here->LTRAbrEq2 != 0))
            {
                while (here->LTRApos2Ibr2Ptr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->LTRApos2Ibr2Ptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->LTRAnegNode2 != 0) && (here->LTRAbrEq2 != 0))
            {
                while (here->LTRAneg2Ibr2Ptr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->LTRAneg2Ibr2Ptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->LTRAposNode1 != 0) && (here->LTRAposNode1 != 0))
            {
                while (here->LTRApos1Pos1Ptr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->LTRApos1Pos1Ptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->LTRAnegNode1 != 0) && (here->LTRAnegNode1 != 0))
            {
                while (here->LTRAneg1Neg1Ptr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->LTRAneg1Neg1Ptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->LTRAposNode2 != 0) && (here->LTRAposNode2 != 0))
            {
                while (here->LTRApos2Pos2Ptr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->LTRApos2Pos2Ptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->LTRAnegNode2 != 0) && (here->LTRAnegNode2 != 0))
            {
                while (here->LTRAneg2Neg2Ptr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->LTRAneg2Neg2Ptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }
        }
    }

    return (OK) ;
}