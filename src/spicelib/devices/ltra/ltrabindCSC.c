/**********
Author: 2013 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ltradefs.h"
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
LTRAbindCSC (GENmodel *inModel, CKTcircuit *ckt)
{
    LTRAmodel *model = (LTRAmodel *)inModel ;
    LTRAinstance *here ;
    double *i ;
    BindElement *matched, *BindStruct ;
    size_t nz ;

    BindStruct = ckt->CKTmatrix->CKTbindStruct ;
    nz = (size_t)ckt->CKTmatrix->CKTklunz ;

    /* loop through all the LTRA models */
    for ( ; model != NULL ; model = LTRAnextModel(model))
    {
        /* loop through all the instances of the model */
        for (here = LTRAinstances(model); here != NULL ; here = LTRAnextInstance(here))
        {
            CREATE_KLU_BINDING_TABLE(LTRAibr1Pos1Ptr, LTRAibr1Pos1Binding, LTRAbrEq1, LTRAposNode1);
            CREATE_KLU_BINDING_TABLE(LTRAibr1Neg1Ptr, LTRAibr1Neg1Binding, LTRAbrEq1, LTRAnegNode1);
            CREATE_KLU_BINDING_TABLE(LTRAibr1Pos2Ptr, LTRAibr1Pos2Binding, LTRAbrEq1, LTRAposNode2);
            CREATE_KLU_BINDING_TABLE(LTRAibr1Neg2Ptr, LTRAibr1Neg2Binding, LTRAbrEq1, LTRAnegNode2);
            CREATE_KLU_BINDING_TABLE(LTRAibr1Ibr1Ptr, LTRAibr1Ibr1Binding, LTRAbrEq1, LTRAbrEq1);
            CREATE_KLU_BINDING_TABLE(LTRAibr1Ibr2Ptr, LTRAibr1Ibr2Binding, LTRAbrEq1, LTRAbrEq2);
            CREATE_KLU_BINDING_TABLE(LTRAibr2Pos1Ptr, LTRAibr2Pos1Binding, LTRAbrEq2, LTRAposNode1);
            CREATE_KLU_BINDING_TABLE(LTRAibr2Neg1Ptr, LTRAibr2Neg1Binding, LTRAbrEq2, LTRAnegNode1);
            CREATE_KLU_BINDING_TABLE(LTRAibr2Pos2Ptr, LTRAibr2Pos2Binding, LTRAbrEq2, LTRAposNode2);
            CREATE_KLU_BINDING_TABLE(LTRAibr2Neg2Ptr, LTRAibr2Neg2Binding, LTRAbrEq2, LTRAnegNode2);
            CREATE_KLU_BINDING_TABLE(LTRAibr2Ibr1Ptr, LTRAibr2Ibr1Binding, LTRAbrEq2, LTRAbrEq1);
            CREATE_KLU_BINDING_TABLE(LTRAibr2Ibr2Ptr, LTRAibr2Ibr2Binding, LTRAbrEq2, LTRAbrEq2);
            CREATE_KLU_BINDING_TABLE(LTRApos1Ibr1Ptr, LTRApos1Ibr1Binding, LTRAposNode1, LTRAbrEq1);
            CREATE_KLU_BINDING_TABLE(LTRAneg1Ibr1Ptr, LTRAneg1Ibr1Binding, LTRAnegNode1, LTRAbrEq1);
            CREATE_KLU_BINDING_TABLE(LTRApos2Ibr2Ptr, LTRApos2Ibr2Binding, LTRAposNode2, LTRAbrEq2);
            CREATE_KLU_BINDING_TABLE(LTRAneg2Ibr2Ptr, LTRAneg2Ibr2Binding, LTRAnegNode2, LTRAbrEq2);
            CREATE_KLU_BINDING_TABLE(LTRApos1Pos1Ptr, LTRApos1Pos1Binding, LTRAposNode1, LTRAposNode1);
            CREATE_KLU_BINDING_TABLE(LTRAneg1Neg1Ptr, LTRAneg1Neg1Binding, LTRAnegNode1, LTRAnegNode1);
            CREATE_KLU_BINDING_TABLE(LTRApos2Pos2Ptr, LTRApos2Pos2Binding, LTRAposNode2, LTRAposNode2);
            CREATE_KLU_BINDING_TABLE(LTRAneg2Neg2Ptr, LTRAneg2Neg2Binding, LTRAnegNode2, LTRAnegNode2);
        }
    }

    return (OK) ;
}

int
LTRAbindCSCComplex (GENmodel *inModel, CKTcircuit *ckt)
{
    LTRAmodel *model = (LTRAmodel *)inModel ;
    LTRAinstance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the LTRA models */
    for ( ; model != NULL ; model = LTRAnextModel(model))
    {
        /* loop through all the instances of the model */
        for (here = LTRAinstances(model); here != NULL ; here = LTRAnextInstance(here))
        {
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(LTRAibr1Pos1Ptr, LTRAibr1Pos1Binding, LTRAbrEq1, LTRAposNode1);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(LTRAibr1Neg1Ptr, LTRAibr1Neg1Binding, LTRAbrEq1, LTRAnegNode1);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(LTRAibr1Pos2Ptr, LTRAibr1Pos2Binding, LTRAbrEq1, LTRAposNode2);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(LTRAibr1Neg2Ptr, LTRAibr1Neg2Binding, LTRAbrEq1, LTRAnegNode2);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(LTRAibr1Ibr1Ptr, LTRAibr1Ibr1Binding, LTRAbrEq1, LTRAbrEq1);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(LTRAibr1Ibr2Ptr, LTRAibr1Ibr2Binding, LTRAbrEq1, LTRAbrEq2);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(LTRAibr2Pos1Ptr, LTRAibr2Pos1Binding, LTRAbrEq2, LTRAposNode1);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(LTRAibr2Neg1Ptr, LTRAibr2Neg1Binding, LTRAbrEq2, LTRAnegNode1);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(LTRAibr2Pos2Ptr, LTRAibr2Pos2Binding, LTRAbrEq2, LTRAposNode2);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(LTRAibr2Neg2Ptr, LTRAibr2Neg2Binding, LTRAbrEq2, LTRAnegNode2);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(LTRAibr2Ibr1Ptr, LTRAibr2Ibr1Binding, LTRAbrEq2, LTRAbrEq1);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(LTRAibr2Ibr2Ptr, LTRAibr2Ibr2Binding, LTRAbrEq2, LTRAbrEq2);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(LTRApos1Ibr1Ptr, LTRApos1Ibr1Binding, LTRAposNode1, LTRAbrEq1);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(LTRAneg1Ibr1Ptr, LTRAneg1Ibr1Binding, LTRAnegNode1, LTRAbrEq1);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(LTRApos2Ibr2Ptr, LTRApos2Ibr2Binding, LTRAposNode2, LTRAbrEq2);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(LTRAneg2Ibr2Ptr, LTRAneg2Ibr2Binding, LTRAnegNode2, LTRAbrEq2);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(LTRApos1Pos1Ptr, LTRApos1Pos1Binding, LTRAposNode1, LTRAposNode1);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(LTRAneg1Neg1Ptr, LTRAneg1Neg1Binding, LTRAnegNode1, LTRAnegNode1);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(LTRApos2Pos2Ptr, LTRApos2Pos2Binding, LTRAposNode2, LTRAposNode2);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(LTRAneg2Neg2Ptr, LTRAneg2Neg2Binding, LTRAnegNode2, LTRAnegNode2);
        }
    }

    return (OK) ;
}

int
LTRAbindCSCComplexToReal (GENmodel *inModel, CKTcircuit *ckt)
{
    LTRAmodel *model = (LTRAmodel *)inModel ;
    LTRAinstance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the LTRA models */
    for ( ; model != NULL ; model = LTRAnextModel(model))
    {
        /* loop through all the instances of the model */
        for (here = LTRAinstances(model); here != NULL ; here = LTRAnextInstance(here))
        {
            CONVERT_KLU_BINDING_TABLE_TO_REAL(LTRAibr1Pos1Ptr, LTRAibr1Pos1Binding, LTRAbrEq1, LTRAposNode1);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(LTRAibr1Neg1Ptr, LTRAibr1Neg1Binding, LTRAbrEq1, LTRAnegNode1);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(LTRAibr1Pos2Ptr, LTRAibr1Pos2Binding, LTRAbrEq1, LTRAposNode2);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(LTRAibr1Neg2Ptr, LTRAibr1Neg2Binding, LTRAbrEq1, LTRAnegNode2);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(LTRAibr1Ibr1Ptr, LTRAibr1Ibr1Binding, LTRAbrEq1, LTRAbrEq1);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(LTRAibr1Ibr2Ptr, LTRAibr1Ibr2Binding, LTRAbrEq1, LTRAbrEq2);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(LTRAibr2Pos1Ptr, LTRAibr2Pos1Binding, LTRAbrEq2, LTRAposNode1);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(LTRAibr2Neg1Ptr, LTRAibr2Neg1Binding, LTRAbrEq2, LTRAnegNode1);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(LTRAibr2Pos2Ptr, LTRAibr2Pos2Binding, LTRAbrEq2, LTRAposNode2);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(LTRAibr2Neg2Ptr, LTRAibr2Neg2Binding, LTRAbrEq2, LTRAnegNode2);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(LTRAibr2Ibr1Ptr, LTRAibr2Ibr1Binding, LTRAbrEq2, LTRAbrEq1);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(LTRAibr2Ibr2Ptr, LTRAibr2Ibr2Binding, LTRAbrEq2, LTRAbrEq2);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(LTRApos1Ibr1Ptr, LTRApos1Ibr1Binding, LTRAposNode1, LTRAbrEq1);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(LTRAneg1Ibr1Ptr, LTRAneg1Ibr1Binding, LTRAnegNode1, LTRAbrEq1);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(LTRApos2Ibr2Ptr, LTRApos2Ibr2Binding, LTRAposNode2, LTRAbrEq2);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(LTRAneg2Ibr2Ptr, LTRAneg2Ibr2Binding, LTRAnegNode2, LTRAbrEq2);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(LTRApos1Pos1Ptr, LTRApos1Pos1Binding, LTRAposNode1, LTRAposNode1);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(LTRAneg1Neg1Ptr, LTRAneg1Neg1Binding, LTRAnegNode1, LTRAnegNode1);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(LTRApos2Pos2Ptr, LTRApos2Pos2Binding, LTRAposNode2, LTRAposNode2);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(LTRAneg2Neg2Ptr, LTRAneg2Neg2Binding, LTRAnegNode2, LTRAnegNode2);
        }
    }

    return (OK) ;
}
