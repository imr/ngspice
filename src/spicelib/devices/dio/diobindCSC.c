/**********
Author: 2013 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "diodefs.h"
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
DIObindCSC (GENmodel *inModel, CKTcircuit *ckt)
{
    DIOmodel *model = (DIOmodel *)inModel ;
    DIOinstance *here ;
    double *i ;
    BindElement *matched, *BindStruct ;
    size_t nz ;

    BindStruct = ckt->CKTmatrix->CKTbindStruct ;
    nz = (size_t)ckt->CKTmatrix->CKTklunz ;

    /* loop through all the DIO models */
    for ( ; model != NULL ; model = model->DIOnextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->DIOinstances ; here != NULL ; here = here->DIOnextInstance)
        {
            if ((here->DIOposNode != 0) && (here->DIOposPrimeNode != 0))
            {
                i = here->DIOposPosPrimePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->DIOposPosPrimeStructPtr = matched ;
                here->DIOposPosPrimePtr = matched->CSC ;
            }

            if ((here->DIOnegNode != 0) && (here->DIOposPrimeNode != 0))
            {
                i = here->DIOnegPosPrimePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->DIOnegPosPrimeStructPtr = matched ;
                here->DIOnegPosPrimePtr = matched->CSC ;
            }

            if ((here->DIOposPrimeNode != 0) && (here->DIOposNode != 0))
            {
                i = here->DIOposPrimePosPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->DIOposPrimePosStructPtr = matched ;
                here->DIOposPrimePosPtr = matched->CSC ;
            }

            if ((here->DIOposPrimeNode != 0) && (here->DIOnegNode != 0))
            {
                i = here->DIOposPrimeNegPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->DIOposPrimeNegStructPtr = matched ;
                here->DIOposPrimeNegPtr = matched->CSC ;
            }

            if ((here->DIOposNode != 0) && (here->DIOposNode != 0))
            {
                i = here->DIOposPosPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->DIOposPosStructPtr = matched ;
                here->DIOposPosPtr = matched->CSC ;
            }

            if ((here->DIOnegNode != 0) && (here->DIOnegNode != 0))
            {
                i = here->DIOnegNegPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->DIOnegNegStructPtr = matched ;
                here->DIOnegNegPtr = matched->CSC ;
            }

            if ((here->DIOposPrimeNode != 0) && (here->DIOposPrimeNode != 0))
            {
                i = here->DIOposPrimePosPrimePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->DIOposPrimePosPrimeStructPtr = matched ;
                here->DIOposPrimePosPrimePtr = matched->CSC ;
            }

        }
    }

    return (OK) ;
}

int
DIObindCSCComplex (GENmodel *inModel, CKTcircuit *ckt)
{
    DIOmodel *model = (DIOmodel *)inModel ;
    DIOinstance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the DIO models */
    for ( ; model != NULL ; model = model->DIOnextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->DIOinstances ; here != NULL ; here = here->DIOnextInstance)
        {
            if ((here->DIOposNode != 0) && (here->DIOposPrimeNode != 0))
                here->DIOposPosPrimePtr = here->DIOposPosPrimeStructPtr->CSC_Complex ;

            if ((here->DIOnegNode != 0) && (here->DIOposPrimeNode != 0))
                here->DIOnegPosPrimePtr = here->DIOnegPosPrimeStructPtr->CSC_Complex ;

            if ((here->DIOposPrimeNode != 0) && (here->DIOposNode != 0))
                here->DIOposPrimePosPtr = here->DIOposPrimePosStructPtr->CSC_Complex ;

            if ((here->DIOposPrimeNode != 0) && (here->DIOnegNode != 0))
                here->DIOposPrimeNegPtr = here->DIOposPrimeNegStructPtr->CSC_Complex ;

            if ((here->DIOposNode != 0) && (here->DIOposNode != 0))
                here->DIOposPosPtr = here->DIOposPosStructPtr->CSC_Complex ;

            if ((here->DIOnegNode != 0) && (here->DIOnegNode != 0))
                here->DIOnegNegPtr = here->DIOnegNegStructPtr->CSC_Complex ;

            if ((here->DIOposPrimeNode != 0) && (here->DIOposPrimeNode != 0))
                here->DIOposPrimePosPrimePtr = here->DIOposPrimePosPrimeStructPtr->CSC_Complex ;

        }
    }

    return (OK) ;
}

int
DIObindCSCComplexToReal (GENmodel *inModel, CKTcircuit *ckt)
{
    DIOmodel *model = (DIOmodel *)inModel ;
    DIOinstance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the DIO models */
    for ( ; model != NULL ; model = model->DIOnextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->DIOinstances ; here != NULL ; here = here->DIOnextInstance)
        {
            if ((here->DIOposNode != 0) && (here->DIOposPrimeNode != 0))
                here->DIOposPosPrimePtr = here->DIOposPosPrimeStructPtr->CSC ;

            if ((here->DIOnegNode != 0) && (here->DIOposPrimeNode != 0))
                here->DIOnegPosPrimePtr = here->DIOnegPosPrimeStructPtr->CSC ;

            if ((here->DIOposPrimeNode != 0) && (here->DIOposNode != 0))
                here->DIOposPrimePosPtr = here->DIOposPrimePosStructPtr->CSC ;

            if ((here->DIOposPrimeNode != 0) && (here->DIOnegNode != 0))
                here->DIOposPrimeNegPtr = here->DIOposPrimeNegStructPtr->CSC ;

            if ((here->DIOposNode != 0) && (here->DIOposNode != 0))
                here->DIOposPosPtr = here->DIOposPosStructPtr->CSC ;

            if ((here->DIOnegNode != 0) && (here->DIOnegNode != 0))
                here->DIOnegNegPtr = here->DIOnegNegStructPtr->CSC ;

            if ((here->DIOposPrimeNode != 0) && (here->DIOposPrimeNode != 0))
                here->DIOposPrimePosPrimePtr = here->DIOposPrimePosPrimeStructPtr->CSC ;

        }
    }

    return (OK) ;
}
