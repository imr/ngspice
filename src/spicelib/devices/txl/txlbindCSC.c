/**********
Author: 2013 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "txldefs.h"
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
TXLbindCSC (GENmodel *inModel, CKTcircuit *ckt)
{
    TXLmodel *model = (TXLmodel *)inModel ;
    TXLinstance *here ;
    double *i ;
    BindElement *matched, *BindStruct ;
    size_t nz ;

    BindStruct = ckt->CKTmatrix->CKTbindStruct ;
    nz = (size_t)ckt->CKTmatrix->CKTklunz ;

    /* loop through all the TXL models */
    for ( ; model != NULL ; model = model->TXLnextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->TXLinstances ; here != NULL ; here = here->TXLnextInstance)
        {
            if ((here-> TXLposNode != 0) && (here-> TXLposNode != 0))
            {
                i = here->TXLposPosptr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->TXLposPosptrStructPtr = matched ;
                here->TXLposPosptr = matched->CSC ;
            }

            if ((here-> TXLposNode != 0) && (here-> TXLnegNode != 0))
            {
                i = here->TXLposNegptr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->TXLposNegptrStructPtr = matched ;
                here->TXLposNegptr = matched->CSC ;
            }

            if ((here-> TXLnegNode != 0) && (here-> TXLposNode != 0))
            {
                i = here->TXLnegPosptr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->TXLnegPosptrStructPtr = matched ;
                here->TXLnegPosptr = matched->CSC ;
            }

            if ((here-> TXLnegNode != 0) && (here-> TXLnegNode != 0))
            {
                i = here->TXLnegNegptr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->TXLnegNegptrStructPtr = matched ;
                here->TXLnegNegptr = matched->CSC ;
            }

            if ((here-> TXLibr1 != 0) && (here-> TXLposNode != 0))
            {
                i = here->TXLibr1Posptr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->TXLibr1PosptrStructPtr = matched ;
                here->TXLibr1Posptr = matched->CSC ;
            }

            if ((here-> TXLibr2 != 0) && (here-> TXLnegNode != 0))
            {
                i = here->TXLibr2Negptr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->TXLibr2NegptrStructPtr = matched ;
                here->TXLibr2Negptr = matched->CSC ;
            }

            if ((here-> TXLnegNode != 0) && (here-> TXLibr2 != 0))
            {
                i = here->TXLnegIbr2ptr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->TXLnegIbr2ptrStructPtr = matched ;
                here->TXLnegIbr2ptr = matched->CSC ;
            }

            if ((here-> TXLposNode != 0) && (here-> TXLibr1 != 0))
            {
                i = here->TXLposIbr1ptr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->TXLposIbr1ptrStructPtr = matched ;
                here->TXLposIbr1ptr = matched->CSC ;
            }

            if ((here-> TXLibr1 != 0) && (here-> TXLibr1 != 0))
            {
                i = here->TXLibr1Ibr1ptr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->TXLibr1Ibr1ptrStructPtr = matched ;
                here->TXLibr1Ibr1ptr = matched->CSC ;
            }

            if ((here-> TXLibr2 != 0) && (here-> TXLibr2 != 0))
            {
                i = here->TXLibr2Ibr2ptr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->TXLibr2Ibr2ptrStructPtr = matched ;
                here->TXLibr2Ibr2ptr = matched->CSC ;
            }

            if ((here-> TXLibr1 != 0) && (here-> TXLnegNode != 0))
            {
                i = here->TXLibr1Negptr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->TXLibr1NegptrStructPtr = matched ;
                here->TXLibr1Negptr = matched->CSC ;
            }

            if ((here-> TXLibr2 != 0) && (here-> TXLposNode != 0))
            {
                i = here->TXLibr2Posptr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->TXLibr2PosptrStructPtr = matched ;
                here->TXLibr2Posptr = matched->CSC ;
            }

            if ((here-> TXLibr1 != 0) && (here-> TXLibr2 != 0))
            {
                i = here->TXLibr1Ibr2ptr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->TXLibr1Ibr2ptrStructPtr = matched ;
                here->TXLibr1Ibr2ptr = matched->CSC ;
            }

            if ((here-> TXLibr2 != 0) && (here-> TXLibr1 != 0))
            {
                i = here->TXLibr2Ibr1ptr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->TXLibr2Ibr1ptrStructPtr = matched ;
                here->TXLibr2Ibr1ptr = matched->CSC ;
            }

        }
    }

    return (OK) ;
}

int
TXLbindCSCComplex (GENmodel *inModel, CKTcircuit *ckt)
{
    TXLmodel *model = (TXLmodel *)inModel ;
    TXLinstance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the TXL models */
    for ( ; model != NULL ; model = model->TXLnextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->TXLinstances ; here != NULL ; here = here->TXLnextInstance)
        {
            if ((here-> TXLposNode != 0) && (here-> TXLposNode != 0))
                here->TXLposPosptr = here->TXLposPosptrStructPtr->CSC_Complex ;

            if ((here-> TXLposNode != 0) && (here-> TXLnegNode != 0))
                here->TXLposNegptr = here->TXLposNegptrStructPtr->CSC_Complex ;

            if ((here-> TXLnegNode != 0) && (here-> TXLposNode != 0))
                here->TXLnegPosptr = here->TXLnegPosptrStructPtr->CSC_Complex ;

            if ((here-> TXLnegNode != 0) && (here-> TXLnegNode != 0))
                here->TXLnegNegptr = here->TXLnegNegptrStructPtr->CSC_Complex ;

            if ((here-> TXLibr1 != 0) && (here-> TXLposNode != 0))
                here->TXLibr1Posptr = here->TXLibr1PosptrStructPtr->CSC_Complex ;

            if ((here-> TXLibr2 != 0) && (here-> TXLnegNode != 0))
                here->TXLibr2Negptr = here->TXLibr2NegptrStructPtr->CSC_Complex ;

            if ((here-> TXLnegNode != 0) && (here-> TXLibr2 != 0))
                here->TXLnegIbr2ptr = here->TXLnegIbr2ptrStructPtr->CSC_Complex ;

            if ((here-> TXLposNode != 0) && (here-> TXLibr1 != 0))
                here->TXLposIbr1ptr = here->TXLposIbr1ptrStructPtr->CSC_Complex ;

            if ((here-> TXLibr1 != 0) && (here-> TXLibr1 != 0))
                here->TXLibr1Ibr1ptr = here->TXLibr1Ibr1ptrStructPtr->CSC_Complex ;

            if ((here-> TXLibr2 != 0) && (here-> TXLibr2 != 0))
                here->TXLibr2Ibr2ptr = here->TXLibr2Ibr2ptrStructPtr->CSC_Complex ;

            if ((here-> TXLibr1 != 0) && (here-> TXLnegNode != 0))
                here->TXLibr1Negptr = here->TXLibr1NegptrStructPtr->CSC_Complex ;

            if ((here-> TXLibr2 != 0) && (here-> TXLposNode != 0))
                here->TXLibr2Posptr = here->TXLibr2PosptrStructPtr->CSC_Complex ;

            if ((here-> TXLibr1 != 0) && (here-> TXLibr2 != 0))
                here->TXLibr1Ibr2ptr = here->TXLibr1Ibr2ptrStructPtr->CSC_Complex ;

            if ((here-> TXLibr2 != 0) && (here-> TXLibr1 != 0))
                here->TXLibr2Ibr1ptr = here->TXLibr2Ibr1ptrStructPtr->CSC_Complex ;

        }
    }

    return (OK) ;
}

int
TXLbindCSCComplexToReal (GENmodel *inModel, CKTcircuit *ckt)
{
    TXLmodel *model = (TXLmodel *)inModel ;
    TXLinstance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the TXL models */
    for ( ; model != NULL ; model = model->TXLnextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->TXLinstances ; here != NULL ; here = here->TXLnextInstance)
        {
            if ((here-> TXLposNode != 0) && (here-> TXLposNode != 0))
                here->TXLposPosptr = here->TXLposPosptrStructPtr->CSC ;

            if ((here-> TXLposNode != 0) && (here-> TXLnegNode != 0))
                here->TXLposNegptr = here->TXLposNegptrStructPtr->CSC ;

            if ((here-> TXLnegNode != 0) && (here-> TXLposNode != 0))
                here->TXLnegPosptr = here->TXLnegPosptrStructPtr->CSC ;

            if ((here-> TXLnegNode != 0) && (here-> TXLnegNode != 0))
                here->TXLnegNegptr = here->TXLnegNegptrStructPtr->CSC ;

            if ((here-> TXLibr1 != 0) && (here-> TXLposNode != 0))
                here->TXLibr1Posptr = here->TXLibr1PosptrStructPtr->CSC ;

            if ((here-> TXLibr2 != 0) && (here-> TXLnegNode != 0))
                here->TXLibr2Negptr = here->TXLibr2NegptrStructPtr->CSC ;

            if ((here-> TXLnegNode != 0) && (here-> TXLibr2 != 0))
                here->TXLnegIbr2ptr = here->TXLnegIbr2ptrStructPtr->CSC ;

            if ((here-> TXLposNode != 0) && (here-> TXLibr1 != 0))
                here->TXLposIbr1ptr = here->TXLposIbr1ptrStructPtr->CSC ;

            if ((here-> TXLibr1 != 0) && (here-> TXLibr1 != 0))
                here->TXLibr1Ibr1ptr = here->TXLibr1Ibr1ptrStructPtr->CSC ;

            if ((here-> TXLibr2 != 0) && (here-> TXLibr2 != 0))
                here->TXLibr2Ibr2ptr = here->TXLibr2Ibr2ptrStructPtr->CSC ;

            if ((here-> TXLibr1 != 0) && (here-> TXLnegNode != 0))
                here->TXLibr1Negptr = here->TXLibr1NegptrStructPtr->CSC ;

            if ((here-> TXLibr2 != 0) && (here-> TXLposNode != 0))
                here->TXLibr2Posptr = here->TXLibr2PosptrStructPtr->CSC ;

            if ((here-> TXLibr1 != 0) && (here-> TXLibr2 != 0))
                here->TXLibr1Ibr2ptr = here->TXLibr1Ibr2ptrStructPtr->CSC ;

            if ((here-> TXLibr2 != 0) && (here-> TXLibr1 != 0))
                here->TXLibr2Ibr1ptr = here->TXLibr2Ibr1ptrStructPtr->CSC ;

        }
    }

    return (OK) ;
}
