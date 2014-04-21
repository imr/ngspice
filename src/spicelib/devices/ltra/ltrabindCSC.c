/**********
Author: 2013 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ltradefs.h"
#include "ngspice/sperror.h"

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
    for ( ; model != NULL ; model = model->LTRAnextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->LTRAinstances ; here != NULL ; here = here->LTRAnextInstance)
        {
            if ((here-> LTRAbrEq1 != 0) && (here-> LTRAposNode1 != 0))
            {
                i = here->LTRAibr1Pos1Ptr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->LTRAibr1Pos1StructPtr = matched ;
                here->LTRAibr1Pos1Ptr = matched->CSC ;
            }

            if ((here-> LTRAbrEq1 != 0) && (here-> LTRAnegNode1 != 0))
            {
                i = here->LTRAibr1Neg1Ptr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->LTRAibr1Neg1StructPtr = matched ;
                here->LTRAibr1Neg1Ptr = matched->CSC ;
            }

            if ((here-> LTRAbrEq1 != 0) && (here-> LTRAposNode2 != 0))
            {
                i = here->LTRAibr1Pos2Ptr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->LTRAibr1Pos2StructPtr = matched ;
                here->LTRAibr1Pos2Ptr = matched->CSC ;
            }

            if ((here-> LTRAbrEq1 != 0) && (here-> LTRAnegNode2 != 0))
            {
                i = here->LTRAibr1Neg2Ptr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->LTRAibr1Neg2StructPtr = matched ;
                here->LTRAibr1Neg2Ptr = matched->CSC ;
            }

            if ((here-> LTRAbrEq1 != 0) && (here-> LTRAbrEq1 != 0))
            {
                i = here->LTRAibr1Ibr1Ptr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->LTRAibr1Ibr1StructPtr = matched ;
                here->LTRAibr1Ibr1Ptr = matched->CSC ;
            }

            if ((here-> LTRAbrEq1 != 0) && (here-> LTRAbrEq2 != 0))
            {
                i = here->LTRAibr1Ibr2Ptr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->LTRAibr1Ibr2StructPtr = matched ;
                here->LTRAibr1Ibr2Ptr = matched->CSC ;
            }

            if ((here-> LTRAbrEq2 != 0) && (here-> LTRAposNode1 != 0))
            {
                i = here->LTRAibr2Pos1Ptr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->LTRAibr2Pos1StructPtr = matched ;
                here->LTRAibr2Pos1Ptr = matched->CSC ;
            }

            if ((here-> LTRAbrEq2 != 0) && (here-> LTRAnegNode1 != 0))
            {
                i = here->LTRAibr2Neg1Ptr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->LTRAibr2Neg1StructPtr = matched ;
                here->LTRAibr2Neg1Ptr = matched->CSC ;
            }

            if ((here-> LTRAbrEq2 != 0) && (here-> LTRAposNode2 != 0))
            {
                i = here->LTRAibr2Pos2Ptr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->LTRAibr2Pos2StructPtr = matched ;
                here->LTRAibr2Pos2Ptr = matched->CSC ;
            }

            if ((here-> LTRAbrEq2 != 0) && (here-> LTRAnegNode2 != 0))
            {
                i = here->LTRAibr2Neg2Ptr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->LTRAibr2Neg2StructPtr = matched ;
                here->LTRAibr2Neg2Ptr = matched->CSC ;
            }

            if ((here-> LTRAbrEq2 != 0) && (here-> LTRAbrEq1 != 0))
            {
                i = here->LTRAibr2Ibr1Ptr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->LTRAibr2Ibr1StructPtr = matched ;
                here->LTRAibr2Ibr1Ptr = matched->CSC ;
            }

            if ((here-> LTRAbrEq2 != 0) && (here-> LTRAbrEq2 != 0))
            {
                i = here->LTRAibr2Ibr2Ptr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->LTRAibr2Ibr2StructPtr = matched ;
                here->LTRAibr2Ibr2Ptr = matched->CSC ;
            }

            if ((here-> LTRAposNode1 != 0) && (here-> LTRAbrEq1 != 0))
            {
                i = here->LTRApos1Ibr1Ptr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->LTRApos1Ibr1StructPtr = matched ;
                here->LTRApos1Ibr1Ptr = matched->CSC ;
            }

            if ((here-> LTRAnegNode1 != 0) && (here-> LTRAbrEq1 != 0))
            {
                i = here->LTRAneg1Ibr1Ptr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->LTRAneg1Ibr1StructPtr = matched ;
                here->LTRAneg1Ibr1Ptr = matched->CSC ;
            }

            if ((here-> LTRAposNode2 != 0) && (here-> LTRAbrEq2 != 0))
            {
                i = here->LTRApos2Ibr2Ptr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->LTRApos2Ibr2StructPtr = matched ;
                here->LTRApos2Ibr2Ptr = matched->CSC ;
            }

            if ((here-> LTRAnegNode2 != 0) && (here-> LTRAbrEq2 != 0))
            {
                i = here->LTRAneg2Ibr2Ptr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->LTRAneg2Ibr2StructPtr = matched ;
                here->LTRAneg2Ibr2Ptr = matched->CSC ;
            }

            if ((here-> LTRAposNode1 != 0) && (here-> LTRAposNode1 != 0))
            {
                i = here->LTRApos1Pos1Ptr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->LTRApos1Pos1StructPtr = matched ;
                here->LTRApos1Pos1Ptr = matched->CSC ;
            }

            if ((here-> LTRAnegNode1 != 0) && (here-> LTRAnegNode1 != 0))
            {
                i = here->LTRAneg1Neg1Ptr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->LTRAneg1Neg1StructPtr = matched ;
                here->LTRAneg1Neg1Ptr = matched->CSC ;
            }

            if ((here-> LTRAposNode2 != 0) && (here-> LTRAposNode2 != 0))
            {
                i = here->LTRApos2Pos2Ptr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->LTRApos2Pos2StructPtr = matched ;
                here->LTRApos2Pos2Ptr = matched->CSC ;
            }

            if ((here-> LTRAnegNode2 != 0) && (here-> LTRAnegNode2 != 0))
            {
                i = here->LTRAneg2Neg2Ptr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->LTRAneg2Neg2StructPtr = matched ;
                here->LTRAneg2Neg2Ptr = matched->CSC ;
            }

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
    for ( ; model != NULL ; model = model->LTRAnextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->LTRAinstances ; here != NULL ; here = here->LTRAnextInstance)
        {
            if ((here-> LTRAbrEq1 != 0) && (here-> LTRAposNode1 != 0))
                here->LTRAibr1Pos1Ptr = here->LTRAibr1Pos1StructPtr->CSC_Complex ;

            if ((here-> LTRAbrEq1 != 0) && (here-> LTRAnegNode1 != 0))
                here->LTRAibr1Neg1Ptr = here->LTRAibr1Neg1StructPtr->CSC_Complex ;

            if ((here-> LTRAbrEq1 != 0) && (here-> LTRAposNode2 != 0))
                here->LTRAibr1Pos2Ptr = here->LTRAibr1Pos2StructPtr->CSC_Complex ;

            if ((here-> LTRAbrEq1 != 0) && (here-> LTRAnegNode2 != 0))
                here->LTRAibr1Neg2Ptr = here->LTRAibr1Neg2StructPtr->CSC_Complex ;

            if ((here-> LTRAbrEq1 != 0) && (here-> LTRAbrEq1 != 0))
                here->LTRAibr1Ibr1Ptr = here->LTRAibr1Ibr1StructPtr->CSC_Complex ;

            if ((here-> LTRAbrEq1 != 0) && (here-> LTRAbrEq2 != 0))
                here->LTRAibr1Ibr2Ptr = here->LTRAibr1Ibr2StructPtr->CSC_Complex ;

            if ((here-> LTRAbrEq2 != 0) && (here-> LTRAposNode1 != 0))
                here->LTRAibr2Pos1Ptr = here->LTRAibr2Pos1StructPtr->CSC_Complex ;

            if ((here-> LTRAbrEq2 != 0) && (here-> LTRAnegNode1 != 0))
                here->LTRAibr2Neg1Ptr = here->LTRAibr2Neg1StructPtr->CSC_Complex ;

            if ((here-> LTRAbrEq2 != 0) && (here-> LTRAposNode2 != 0))
                here->LTRAibr2Pos2Ptr = here->LTRAibr2Pos2StructPtr->CSC_Complex ;

            if ((here-> LTRAbrEq2 != 0) && (here-> LTRAnegNode2 != 0))
                here->LTRAibr2Neg2Ptr = here->LTRAibr2Neg2StructPtr->CSC_Complex ;

            if ((here-> LTRAbrEq2 != 0) && (here-> LTRAbrEq1 != 0))
                here->LTRAibr2Ibr1Ptr = here->LTRAibr2Ibr1StructPtr->CSC_Complex ;

            if ((here-> LTRAbrEq2 != 0) && (here-> LTRAbrEq2 != 0))
                here->LTRAibr2Ibr2Ptr = here->LTRAibr2Ibr2StructPtr->CSC_Complex ;

            if ((here-> LTRAposNode1 != 0) && (here-> LTRAbrEq1 != 0))
                here->LTRApos1Ibr1Ptr = here->LTRApos1Ibr1StructPtr->CSC_Complex ;

            if ((here-> LTRAnegNode1 != 0) && (here-> LTRAbrEq1 != 0))
                here->LTRAneg1Ibr1Ptr = here->LTRAneg1Ibr1StructPtr->CSC_Complex ;

            if ((here-> LTRAposNode2 != 0) && (here-> LTRAbrEq2 != 0))
                here->LTRApos2Ibr2Ptr = here->LTRApos2Ibr2StructPtr->CSC_Complex ;

            if ((here-> LTRAnegNode2 != 0) && (here-> LTRAbrEq2 != 0))
                here->LTRAneg2Ibr2Ptr = here->LTRAneg2Ibr2StructPtr->CSC_Complex ;

            if ((here-> LTRAposNode1 != 0) && (here-> LTRAposNode1 != 0))
                here->LTRApos1Pos1Ptr = here->LTRApos1Pos1StructPtr->CSC_Complex ;

            if ((here-> LTRAnegNode1 != 0) && (here-> LTRAnegNode1 != 0))
                here->LTRAneg1Neg1Ptr = here->LTRAneg1Neg1StructPtr->CSC_Complex ;

            if ((here-> LTRAposNode2 != 0) && (here-> LTRAposNode2 != 0))
                here->LTRApos2Pos2Ptr = here->LTRApos2Pos2StructPtr->CSC_Complex ;

            if ((here-> LTRAnegNode2 != 0) && (here-> LTRAnegNode2 != 0))
                here->LTRAneg2Neg2Ptr = here->LTRAneg2Neg2StructPtr->CSC_Complex ;

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
    for ( ; model != NULL ; model = model->LTRAnextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->LTRAinstances ; here != NULL ; here = here->LTRAnextInstance)
        {
            if ((here-> LTRAbrEq1 != 0) && (here-> LTRAposNode1 != 0))
                here->LTRAibr1Pos1Ptr = here->LTRAibr1Pos1StructPtr->CSC ;

            if ((here-> LTRAbrEq1 != 0) && (here-> LTRAnegNode1 != 0))
                here->LTRAibr1Neg1Ptr = here->LTRAibr1Neg1StructPtr->CSC ;

            if ((here-> LTRAbrEq1 != 0) && (here-> LTRAposNode2 != 0))
                here->LTRAibr1Pos2Ptr = here->LTRAibr1Pos2StructPtr->CSC ;

            if ((here-> LTRAbrEq1 != 0) && (here-> LTRAnegNode2 != 0))
                here->LTRAibr1Neg2Ptr = here->LTRAibr1Neg2StructPtr->CSC ;

            if ((here-> LTRAbrEq1 != 0) && (here-> LTRAbrEq1 != 0))
                here->LTRAibr1Ibr1Ptr = here->LTRAibr1Ibr1StructPtr->CSC ;

            if ((here-> LTRAbrEq1 != 0) && (here-> LTRAbrEq2 != 0))
                here->LTRAibr1Ibr2Ptr = here->LTRAibr1Ibr2StructPtr->CSC ;

            if ((here-> LTRAbrEq2 != 0) && (here-> LTRAposNode1 != 0))
                here->LTRAibr2Pos1Ptr = here->LTRAibr2Pos1StructPtr->CSC ;

            if ((here-> LTRAbrEq2 != 0) && (here-> LTRAnegNode1 != 0))
                here->LTRAibr2Neg1Ptr = here->LTRAibr2Neg1StructPtr->CSC ;

            if ((here-> LTRAbrEq2 != 0) && (here-> LTRAposNode2 != 0))
                here->LTRAibr2Pos2Ptr = here->LTRAibr2Pos2StructPtr->CSC ;

            if ((here-> LTRAbrEq2 != 0) && (here-> LTRAnegNode2 != 0))
                here->LTRAibr2Neg2Ptr = here->LTRAibr2Neg2StructPtr->CSC ;

            if ((here-> LTRAbrEq2 != 0) && (here-> LTRAbrEq1 != 0))
                here->LTRAibr2Ibr1Ptr = here->LTRAibr2Ibr1StructPtr->CSC ;

            if ((here-> LTRAbrEq2 != 0) && (here-> LTRAbrEq2 != 0))
                here->LTRAibr2Ibr2Ptr = here->LTRAibr2Ibr2StructPtr->CSC ;

            if ((here-> LTRAposNode1 != 0) && (here-> LTRAbrEq1 != 0))
                here->LTRApos1Ibr1Ptr = here->LTRApos1Ibr1StructPtr->CSC ;

            if ((here-> LTRAnegNode1 != 0) && (here-> LTRAbrEq1 != 0))
                here->LTRAneg1Ibr1Ptr = here->LTRAneg1Ibr1StructPtr->CSC ;

            if ((here-> LTRAposNode2 != 0) && (here-> LTRAbrEq2 != 0))
                here->LTRApos2Ibr2Ptr = here->LTRApos2Ibr2StructPtr->CSC ;

            if ((here-> LTRAnegNode2 != 0) && (here-> LTRAbrEq2 != 0))
                here->LTRAneg2Ibr2Ptr = here->LTRAneg2Ibr2StructPtr->CSC ;

            if ((here-> LTRAposNode1 != 0) && (here-> LTRAposNode1 != 0))
                here->LTRApos1Pos1Ptr = here->LTRApos1Pos1StructPtr->CSC ;

            if ((here-> LTRAnegNode1 != 0) && (here-> LTRAnegNode1 != 0))
                here->LTRAneg1Neg1Ptr = here->LTRAneg1Neg1StructPtr->CSC ;

            if ((here-> LTRAposNode2 != 0) && (here-> LTRAposNode2 != 0))
                here->LTRApos2Pos2Ptr = here->LTRApos2Pos2StructPtr->CSC ;

            if ((here-> LTRAnegNode2 != 0) && (here-> LTRAnegNode2 != 0))
                here->LTRAneg2Neg2Ptr = here->LTRAneg2Neg2StructPtr->CSC ;

        }
    }

    return (OK) ;
}
