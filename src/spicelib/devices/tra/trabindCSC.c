/**********
Author: 2012 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "tradefs.h"
#include "ngspice/sperror.h"

int
TRAbindCSC(GENmodel *inModel, CKTcircuit *ckt)
{
    TRAmodel *model = (TRAmodel *)inModel;
    int i ;

    /*  loop through all the tra models */
    for( ; model != NULL; model = model->TRAnextModel ) {
	TRAinstance *here;

        /* loop through all the instances of the model */
        for (here = model->TRAinstances; here != NULL ;
	    here = here->TRAnextInstance) {

		i = 0 ;
		if ((here-> TRAbrEq1 != 0) && (here-> TRAbrEq2 != 0)) {
			while (here->TRAibr1Ibr2Ptr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->TRAibr1Ibr2Ptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here-> TRAbrEq1 != 0) && (here-> TRAintNode1 != 0)) {
			while (here->TRAibr1Int1Ptr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->TRAibr1Int1Ptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here-> TRAbrEq1 != 0) && (here-> TRAnegNode1 != 0)) {
			while (here->TRAibr1Neg1Ptr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->TRAibr1Neg1Ptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here-> TRAbrEq1 != 0) && (here-> TRAnegNode2 != 0)) {
			while (here->TRAibr1Neg2Ptr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->TRAibr1Neg2Ptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here-> TRAbrEq1 != 0) && (here-> TRAposNode2 != 0)) {
			while (here->TRAibr1Pos2Ptr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->TRAibr1Pos2Ptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here-> TRAbrEq2 != 0) && (here-> TRAbrEq1 != 0)) {
			while (here->TRAibr2Ibr1Ptr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->TRAibr2Ibr1Ptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here-> TRAbrEq2 != 0) && (here-> TRAintNode2 != 0)) {
			while (here->TRAibr2Int2Ptr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->TRAibr2Int2Ptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here-> TRAbrEq2 != 0) && (here-> TRAnegNode1 != 0)) {
			while (here->TRAibr2Neg1Ptr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->TRAibr2Neg1Ptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here-> TRAbrEq2 != 0) && (here-> TRAnegNode2 != 0)) {
			while (here->TRAibr2Neg2Ptr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->TRAibr2Neg2Ptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here-> TRAbrEq2 != 0) && (here-> TRAposNode1 != 0)) {
			while (here->TRAibr2Pos1Ptr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->TRAibr2Pos1Ptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here-> TRAintNode1 != 0) && (here-> TRAbrEq1 != 0)) {
			while (here->TRAint1Ibr1Ptr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->TRAint1Ibr1Ptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here-> TRAintNode1 != 0) && (here-> TRAintNode1 != 0)) {
			while (here->TRAint1Int1Ptr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->TRAint1Int1Ptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here-> TRAintNode1 != 0) && (here-> TRAposNode1 != 0)) {
			while (here->TRAint1Pos1Ptr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->TRAint1Pos1Ptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here-> TRAintNode2 != 0) && (here-> TRAbrEq2 != 0)) {
			while (here->TRAint2Ibr2Ptr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->TRAint2Ibr2Ptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here-> TRAintNode2 != 0) && (here-> TRAintNode2 != 0)) {
			while (here->TRAint2Int2Ptr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->TRAint2Int2Ptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here-> TRAintNode2 != 0) && (here-> TRAposNode2 != 0)) {
			while (here->TRAint2Pos2Ptr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->TRAint2Pos2Ptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here-> TRAnegNode1 != 0) && (here-> TRAbrEq1 != 0)) {
			while (here->TRAneg1Ibr1Ptr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->TRAneg1Ibr1Ptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here-> TRAnegNode2 != 0) && (here-> TRAbrEq2 != 0)) {
			while (here->TRAneg2Ibr2Ptr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->TRAneg2Ibr2Ptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here-> TRAposNode1 != 0) && (here-> TRAintNode1 != 0)) {
			while (here->TRApos1Int1Ptr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->TRApos1Int1Ptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here-> TRAposNode1 != 0) && (here-> TRAposNode1 != 0)) {
			while (here->TRApos1Pos1Ptr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->TRApos1Pos1Ptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here-> TRAposNode2 != 0) && (here-> TRAintNode2 != 0)) {
			while (here->TRApos2Int2Ptr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->TRApos2Int2Ptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here-> TRAposNode2 != 0) && (here-> TRAposNode2 != 0)) {
			while (here->TRApos2Pos2Ptr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->TRApos2Pos2Ptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
	}
    }
    return(OK);
}

int
TRAbindCSCComplex(GENmodel *inModel, CKTcircuit *ckt)
{
    TRAmodel *model = (TRAmodel *)inModel;
    int i ;

    /*  loop through all the tra models */
    for( ; model != NULL; model = model->TRAnextModel ) {
	TRAinstance *here;

        /* loop through all the instances of the model */
        for (here = model->TRAinstances; here != NULL ;
	    here = here->TRAnextInstance) {

		i = 0 ;
		if ((here-> TRAbrEq1 != 0) && (here-> TRAbrEq2 != 0)) {
			while (here->TRAibr1Ibr2Ptr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->TRAibr1Ibr2Ptr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here-> TRAbrEq1 != 0) && (here-> TRAintNode1 != 0)) {
			while (here->TRAibr1Int1Ptr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->TRAibr1Int1Ptr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here-> TRAbrEq1 != 0) && (here-> TRAnegNode1 != 0)) {
			while (here->TRAibr1Neg1Ptr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->TRAibr1Neg1Ptr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here-> TRAbrEq1 != 0) && (here-> TRAnegNode2 != 0)) {
			while (here->TRAibr1Neg2Ptr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->TRAibr1Neg2Ptr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here-> TRAbrEq1 != 0) && (here-> TRAposNode2 != 0)) {
			while (here->TRAibr1Pos2Ptr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->TRAibr1Pos2Ptr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here-> TRAbrEq2 != 0) && (here-> TRAbrEq1 != 0)) {
			while (here->TRAibr2Ibr1Ptr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->TRAibr2Ibr1Ptr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here-> TRAbrEq2 != 0) && (here-> TRAintNode2 != 0)) {
			while (here->TRAibr2Int2Ptr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->TRAibr2Int2Ptr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here-> TRAbrEq2 != 0) && (here-> TRAnegNode1 != 0)) {
			while (here->TRAibr2Neg1Ptr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->TRAibr2Neg1Ptr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here-> TRAbrEq2 != 0) && (here-> TRAnegNode2 != 0)) {
			while (here->TRAibr2Neg2Ptr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->TRAibr2Neg2Ptr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here-> TRAbrEq2 != 0) && (here-> TRAposNode1 != 0)) {
			while (here->TRAibr2Pos1Ptr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->TRAibr2Pos1Ptr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here-> TRAintNode1 != 0) && (here-> TRAbrEq1 != 0)) {
			while (here->TRAint1Ibr1Ptr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->TRAint1Ibr1Ptr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here-> TRAintNode1 != 0) && (here-> TRAintNode1 != 0)) {
			while (here->TRAint1Int1Ptr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->TRAint1Int1Ptr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here-> TRAintNode1 != 0) && (here-> TRAposNode1 != 0)) {
			while (here->TRAint1Pos1Ptr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->TRAint1Pos1Ptr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here-> TRAintNode2 != 0) && (here-> TRAbrEq2 != 0)) {
			while (here->TRAint2Ibr2Ptr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->TRAint2Ibr2Ptr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here-> TRAintNode2 != 0) && (here-> TRAintNode2 != 0)) {
			while (here->TRAint2Int2Ptr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->TRAint2Int2Ptr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here-> TRAintNode2 != 0) && (here-> TRAposNode2 != 0)) {
			while (here->TRAint2Pos2Ptr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->TRAint2Pos2Ptr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here-> TRAnegNode1 != 0) && (here-> TRAbrEq1 != 0)) {
			while (here->TRAneg1Ibr1Ptr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->TRAneg1Ibr1Ptr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here-> TRAnegNode2 != 0) && (here-> TRAbrEq2 != 0)) {
			while (here->TRAneg2Ibr2Ptr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->TRAneg2Ibr2Ptr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here-> TRAposNode1 != 0) && (here-> TRAintNode1 != 0)) {
			while (here->TRApos1Int1Ptr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->TRApos1Int1Ptr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here-> TRAposNode1 != 0) && (here-> TRAposNode1 != 0)) {
			while (here->TRApos1Pos1Ptr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->TRApos1Pos1Ptr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here-> TRAposNode2 != 0) && (here-> TRAintNode2 != 0)) {
			while (here->TRApos2Int2Ptr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->TRApos2Int2Ptr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here-> TRAposNode2 != 0) && (here-> TRAposNode2 != 0)) {
			while (here->TRApos2Pos2Ptr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->TRApos2Pos2Ptr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
	}
    }
    return(OK);
}

int
TRAbindCSCComplexToReal (GENmodel *inModel, CKTcircuit *ckt)
{
    TRAmodel *model = (TRAmodel *)inModel ;
    TRAinstance *here ;
    int i ;

    /*  loop through all the tra models */
    for ( ; model != NULL ; model = model->TRAnextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->TRAinstances ; here != NULL ; here = here->TRAnextInstance)
        {
            i = 0 ;
            if ((here->TRAbrEq1 != 0) && (here->TRAbrEq2 != 0))
            {
                while (here->TRAibr1Ibr2Ptr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->TRAibr1Ibr2Ptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->TRAbrEq1 != 0) && (here->TRAintNode1 != 0))
            {
                while (here->TRAibr1Int1Ptr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->TRAibr1Int1Ptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->TRAbrEq1 != 0) && (here->TRAnegNode1 != 0))
            {
                while (here->TRAibr1Neg1Ptr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->TRAibr1Neg1Ptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->TRAbrEq1 != 0) && (here->TRAnegNode2 != 0))
            {
                while (here->TRAibr1Neg2Ptr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->TRAibr1Neg2Ptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->TRAbrEq1 != 0) && (here->TRAposNode2 != 0))
            {
                while (here->TRAibr1Pos2Ptr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->TRAibr1Pos2Ptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->TRAbrEq2 != 0) && (here->TRAbrEq1 != 0))
            {
                while (here->TRAibr2Ibr1Ptr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->TRAibr2Ibr1Ptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->TRAbrEq2 != 0) && (here->TRAintNode2 != 0))
            {
                while (here->TRAibr2Int2Ptr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->TRAibr2Int2Ptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->TRAbrEq2 != 0) && (here->TRAnegNode1 != 0))
            {
                while (here->TRAibr2Neg1Ptr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->TRAibr2Neg1Ptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->TRAbrEq2 != 0) && (here->TRAnegNode2 != 0))
            {
                while (here->TRAibr2Neg2Ptr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->TRAibr2Neg2Ptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->TRAbrEq2 != 0) && (here->TRAposNode1 != 0))
            {
                while (here->TRAibr2Pos1Ptr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->TRAibr2Pos1Ptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->TRAintNode1 != 0) && (here->TRAbrEq1 != 0))
            {
                while (here->TRAint1Ibr1Ptr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->TRAint1Ibr1Ptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->TRAintNode1 != 0) && (here->TRAintNode1 != 0))
            {
                while (here->TRAint1Int1Ptr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->TRAint1Int1Ptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->TRAintNode1 != 0) && (here->TRAposNode1 != 0))
            {
                while (here->TRAint1Pos1Ptr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->TRAint1Pos1Ptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->TRAintNode2 != 0) && (here->TRAbrEq2 != 0))
            {
                while (here->TRAint2Ibr2Ptr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->TRAint2Ibr2Ptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->TRAintNode2 != 0) && (here->TRAintNode2 != 0))
            {
                while (here->TRAint2Int2Ptr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->TRAint2Int2Ptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->TRAintNode2 != 0) && (here->TRAposNode2 != 0))
            {
                while (here->TRAint2Pos2Ptr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->TRAint2Pos2Ptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->TRAnegNode1 != 0) && (here->TRAbrEq1 != 0))
            {
                while (here->TRAneg1Ibr1Ptr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->TRAneg1Ibr1Ptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->TRAnegNode2 != 0) && (here->TRAbrEq2 != 0))
            {
                while (here->TRAneg2Ibr2Ptr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->TRAneg2Ibr2Ptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->TRAposNode1 != 0) && (here->TRAintNode1 != 0))
            {
                while (here->TRApos1Int1Ptr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->TRApos1Int1Ptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->TRAposNode1 != 0) && (here->TRAposNode1 != 0))
            {
                while (here->TRApos1Pos1Ptr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->TRApos1Pos1Ptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->TRAposNode2 != 0) && (here->TRAintNode2 != 0))
            {
                while (here->TRApos2Int2Ptr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->TRApos2Int2Ptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->TRAposNode2 != 0) && (here->TRAposNode2 != 0))
            {
                while (here->TRApos2Pos2Ptr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->TRApos2Pos2Ptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }
        }
    }

    return (OK) ;
}