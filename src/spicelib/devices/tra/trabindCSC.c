/**********
Author: 2013 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "tradefs.h"
#include "ngspice/sperror.h"
#include "ngspice/klu-binding.h"

#include <stdlib.h>

static
int
BindCompare (const void *a, const void *b)
{
    BindElement *A, *B ;
    A = (BindElement *)a ;
    B = (BindElement *)b ;

    return ((int)(A->Sparse - B->Sparse)) ;
}

int
TRAbindCSC (GENmodel *inModel, CKTcircuit *ckt)
{
    TRAmodel *model = (TRAmodel *)inModel ;
    TRAinstance *here ;
    double *i ;
    BindElement *matched, *BindStruct ;
    size_t nz ;

    BindStruct = ckt->CKTmatrix->CKTbindStruct ;
    nz = (size_t)ckt->CKTmatrix->CKTklunz ;

    /* loop through all the TRA models */
    for ( ; model != NULL ; model = TRAnextModel(model))
    {
        /* loop through all the instances of the model */
        for (here = TRAinstances(model); here != NULL ; here = TRAnextInstance(here))
        {
            CREATE_KLU_BINDING_TABLE(TRAibr1Ibr2Ptr, TRAibr1Ibr2Binding, TRAbrEq1, TRAbrEq2);
            CREATE_KLU_BINDING_TABLE(TRAibr1Int1Ptr, TRAibr1Int1Binding, TRAbrEq1, TRAintNode1);
            CREATE_KLU_BINDING_TABLE(TRAibr1Neg1Ptr, TRAibr1Neg1Binding, TRAbrEq1, TRAnegNode1);
            CREATE_KLU_BINDING_TABLE(TRAibr1Neg2Ptr, TRAibr1Neg2Binding, TRAbrEq1, TRAnegNode2);
            CREATE_KLU_BINDING_TABLE(TRAibr1Pos2Ptr, TRAibr1Pos2Binding, TRAbrEq1, TRAposNode2);
            CREATE_KLU_BINDING_TABLE(TRAibr2Ibr1Ptr, TRAibr2Ibr1Binding, TRAbrEq2, TRAbrEq1);
            CREATE_KLU_BINDING_TABLE(TRAibr2Int2Ptr, TRAibr2Int2Binding, TRAbrEq2, TRAintNode2);
            CREATE_KLU_BINDING_TABLE(TRAibr2Neg1Ptr, TRAibr2Neg1Binding, TRAbrEq2, TRAnegNode1);
            CREATE_KLU_BINDING_TABLE(TRAibr2Neg2Ptr, TRAibr2Neg2Binding, TRAbrEq2, TRAnegNode2);
            CREATE_KLU_BINDING_TABLE(TRAibr2Pos1Ptr, TRAibr2Pos1Binding, TRAbrEq2, TRAposNode1);
            CREATE_KLU_BINDING_TABLE(TRAint1Ibr1Ptr, TRAint1Ibr1Binding, TRAintNode1, TRAbrEq1);
            CREATE_KLU_BINDING_TABLE(TRAint1Int1Ptr, TRAint1Int1Binding, TRAintNode1, TRAintNode1);
            CREATE_KLU_BINDING_TABLE(TRAint1Pos1Ptr, TRAint1Pos1Binding, TRAintNode1, TRAposNode1);
            CREATE_KLU_BINDING_TABLE(TRAint2Ibr2Ptr, TRAint2Ibr2Binding, TRAintNode2, TRAbrEq2);
            CREATE_KLU_BINDING_TABLE(TRAint2Int2Ptr, TRAint2Int2Binding, TRAintNode2, TRAintNode2);
            CREATE_KLU_BINDING_TABLE(TRAint2Pos2Ptr, TRAint2Pos2Binding, TRAintNode2, TRAposNode2);
            CREATE_KLU_BINDING_TABLE(TRAneg1Ibr1Ptr, TRAneg1Ibr1Binding, TRAnegNode1, TRAbrEq1);
            CREATE_KLU_BINDING_TABLE(TRAneg2Ibr2Ptr, TRAneg2Ibr2Binding, TRAnegNode2, TRAbrEq2);
            CREATE_KLU_BINDING_TABLE(TRApos1Int1Ptr, TRApos1Int1Binding, TRAposNode1, TRAintNode1);
            CREATE_KLU_BINDING_TABLE(TRApos1Pos1Ptr, TRApos1Pos1Binding, TRAposNode1, TRAposNode1);
            CREATE_KLU_BINDING_TABLE(TRApos2Int2Ptr, TRApos2Int2Binding, TRAposNode2, TRAintNode2);
            CREATE_KLU_BINDING_TABLE(TRApos2Pos2Ptr, TRApos2Pos2Binding, TRAposNode2, TRAposNode2);
        }
    }

    return (OK) ;
}

int
TRAbindCSCComplex (GENmodel *inModel, CKTcircuit *ckt)
{
    TRAmodel *model = (TRAmodel *)inModel ;
    TRAinstance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the TRA models */
    for ( ; model != NULL ; model = TRAnextModel(model))
    {
        /* loop through all the instances of the model */
        for (here = TRAinstances(model); here != NULL ; here = TRAnextInstance(here))
        {
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(TRAibr1Ibr2Ptr, TRAibr1Ibr2Binding, TRAbrEq1, TRAbrEq2);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(TRAibr1Int1Ptr, TRAibr1Int1Binding, TRAbrEq1, TRAintNode1);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(TRAibr1Neg1Ptr, TRAibr1Neg1Binding, TRAbrEq1, TRAnegNode1);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(TRAibr1Neg2Ptr, TRAibr1Neg2Binding, TRAbrEq1, TRAnegNode2);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(TRAibr1Pos2Ptr, TRAibr1Pos2Binding, TRAbrEq1, TRAposNode2);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(TRAibr2Ibr1Ptr, TRAibr2Ibr1Binding, TRAbrEq2, TRAbrEq1);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(TRAibr2Int2Ptr, TRAibr2Int2Binding, TRAbrEq2, TRAintNode2);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(TRAibr2Neg1Ptr, TRAibr2Neg1Binding, TRAbrEq2, TRAnegNode1);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(TRAibr2Neg2Ptr, TRAibr2Neg2Binding, TRAbrEq2, TRAnegNode2);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(TRAibr2Pos1Ptr, TRAibr2Pos1Binding, TRAbrEq2, TRAposNode1);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(TRAint1Ibr1Ptr, TRAint1Ibr1Binding, TRAintNode1, TRAbrEq1);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(TRAint1Int1Ptr, TRAint1Int1Binding, TRAintNode1, TRAintNode1);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(TRAint1Pos1Ptr, TRAint1Pos1Binding, TRAintNode1, TRAposNode1);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(TRAint2Ibr2Ptr, TRAint2Ibr2Binding, TRAintNode2, TRAbrEq2);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(TRAint2Int2Ptr, TRAint2Int2Binding, TRAintNode2, TRAintNode2);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(TRAint2Pos2Ptr, TRAint2Pos2Binding, TRAintNode2, TRAposNode2);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(TRAneg1Ibr1Ptr, TRAneg1Ibr1Binding, TRAnegNode1, TRAbrEq1);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(TRAneg2Ibr2Ptr, TRAneg2Ibr2Binding, TRAnegNode2, TRAbrEq2);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(TRApos1Int1Ptr, TRApos1Int1Binding, TRAposNode1, TRAintNode1);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(TRApos1Pos1Ptr, TRApos1Pos1Binding, TRAposNode1, TRAposNode1);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(TRApos2Int2Ptr, TRApos2Int2Binding, TRAposNode2, TRAintNode2);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(TRApos2Pos2Ptr, TRApos2Pos2Binding, TRAposNode2, TRAposNode2);
        }
    }

    return (OK) ;
}

int
TRAbindCSCComplexToReal (GENmodel *inModel, CKTcircuit *ckt)
{
    TRAmodel *model = (TRAmodel *)inModel ;
    TRAinstance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the TRA models */
    for ( ; model != NULL ; model = TRAnextModel(model))
    {
        /* loop through all the instances of the model */
        for (here = TRAinstances(model); here != NULL ; here = TRAnextInstance(here))
        {
            CONVERT_KLU_BINDING_TABLE_TO_REAL(TRAibr1Ibr2Ptr, TRAibr1Ibr2Binding, TRAbrEq1, TRAbrEq2);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(TRAibr1Int1Ptr, TRAibr1Int1Binding, TRAbrEq1, TRAintNode1);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(TRAibr1Neg1Ptr, TRAibr1Neg1Binding, TRAbrEq1, TRAnegNode1);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(TRAibr1Neg2Ptr, TRAibr1Neg2Binding, TRAbrEq1, TRAnegNode2);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(TRAibr1Pos2Ptr, TRAibr1Pos2Binding, TRAbrEq1, TRAposNode2);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(TRAibr2Ibr1Ptr, TRAibr2Ibr1Binding, TRAbrEq2, TRAbrEq1);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(TRAibr2Int2Ptr, TRAibr2Int2Binding, TRAbrEq2, TRAintNode2);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(TRAibr2Neg1Ptr, TRAibr2Neg1Binding, TRAbrEq2, TRAnegNode1);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(TRAibr2Neg2Ptr, TRAibr2Neg2Binding, TRAbrEq2, TRAnegNode2);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(TRAibr2Pos1Ptr, TRAibr2Pos1Binding, TRAbrEq2, TRAposNode1);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(TRAint1Ibr1Ptr, TRAint1Ibr1Binding, TRAintNode1, TRAbrEq1);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(TRAint1Int1Ptr, TRAint1Int1Binding, TRAintNode1, TRAintNode1);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(TRAint1Pos1Ptr, TRAint1Pos1Binding, TRAintNode1, TRAposNode1);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(TRAint2Ibr2Ptr, TRAint2Ibr2Binding, TRAintNode2, TRAbrEq2);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(TRAint2Int2Ptr, TRAint2Int2Binding, TRAintNode2, TRAintNode2);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(TRAint2Pos2Ptr, TRAint2Pos2Binding, TRAintNode2, TRAposNode2);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(TRAneg1Ibr1Ptr, TRAneg1Ibr1Binding, TRAnegNode1, TRAbrEq1);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(TRAneg2Ibr2Ptr, TRAneg2Ibr2Binding, TRAnegNode2, TRAbrEq2);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(TRApos1Int1Ptr, TRApos1Int1Binding, TRAposNode1, TRAintNode1);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(TRApos1Pos1Ptr, TRApos1Pos1Binding, TRAposNode1, TRAposNode1);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(TRApos2Int2Ptr, TRApos2Int2Binding, TRAposNode2, TRAintNode2);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(TRApos2Pos2Ptr, TRApos2Pos2Binding, TRAposNode2, TRAposNode2);
        }
    }

    return (OK) ;
}
