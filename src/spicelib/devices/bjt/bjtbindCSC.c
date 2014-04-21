/**********
Author: 2013 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "bjtdefs.h"
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
BJTbindCSC (GENmodel *inModel, CKTcircuit *ckt)
{
    BJTmodel *model = (BJTmodel *)inModel ;
    BJTinstance *here ;
    double *i ;
    BindElement *matched, *BindStruct ;
    size_t nz ;

    BindStruct = ckt->CKTmatrix->CKTbindStruct ;
    nz = (size_t)ckt->CKTmatrix->CKTklunz ;

    /* loop through all the BJT models */
    for ( ; model != NULL ; model = model->BJTnextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->BJTinstances ; here != NULL ; here = here->BJTnextInstance)
        {
            if ((here->BJTcolNode != 0) && (here->BJTcolPrimeNode != 0))
            {
                i = here->BJTcolColPrimePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BJTcolColPrimeStructPtr = matched ;
                here->BJTcolColPrimePtr = matched->CSC ;
            }

            if ((here->BJTbaseNode != 0) && (here->BJTbasePrimeNode != 0))
            {
                i = here->BJTbaseBasePrimePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BJTbaseBasePrimeStructPtr = matched ;
                here->BJTbaseBasePrimePtr = matched->CSC ;
            }

            if ((here->BJTemitNode != 0) && (here->BJTemitPrimeNode != 0))
            {
                i = here->BJTemitEmitPrimePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BJTemitEmitPrimeStructPtr = matched ;
                here->BJTemitEmitPrimePtr = matched->CSC ;
            }

            if ((here->BJTcolPrimeNode != 0) && (here->BJTcolNode != 0))
            {
                i = here->BJTcolPrimeColPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BJTcolPrimeColStructPtr = matched ;
                here->BJTcolPrimeColPtr = matched->CSC ;
            }

            if ((here->BJTcolPrimeNode != 0) && (here->BJTbasePrimeNode != 0))
            {
                i = here->BJTcolPrimeBasePrimePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BJTcolPrimeBasePrimeStructPtr = matched ;
                here->BJTcolPrimeBasePrimePtr = matched->CSC ;
            }

            if ((here->BJTcolPrimeNode != 0) && (here->BJTemitPrimeNode != 0))
            {
                i = here->BJTcolPrimeEmitPrimePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BJTcolPrimeEmitPrimeStructPtr = matched ;
                here->BJTcolPrimeEmitPrimePtr = matched->CSC ;
            }

            if ((here->BJTbasePrimeNode != 0) && (here->BJTbaseNode != 0))
            {
                i = here->BJTbasePrimeBasePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BJTbasePrimeBaseStructPtr = matched ;
                here->BJTbasePrimeBasePtr = matched->CSC ;
            }

            if ((here->BJTbasePrimeNode != 0) && (here->BJTcolPrimeNode != 0))
            {
                i = here->BJTbasePrimeColPrimePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BJTbasePrimeColPrimeStructPtr = matched ;
                here->BJTbasePrimeColPrimePtr = matched->CSC ;
            }

            if ((here->BJTbasePrimeNode != 0) && (here->BJTemitPrimeNode != 0))
            {
                i = here->BJTbasePrimeEmitPrimePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BJTbasePrimeEmitPrimeStructPtr = matched ;
                here->BJTbasePrimeEmitPrimePtr = matched->CSC ;
            }

            if ((here->BJTemitPrimeNode != 0) && (here->BJTemitNode != 0))
            {
                i = here->BJTemitPrimeEmitPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BJTemitPrimeEmitStructPtr = matched ;
                here->BJTemitPrimeEmitPtr = matched->CSC ;
            }

            if ((here->BJTemitPrimeNode != 0) && (here->BJTcolPrimeNode != 0))
            {
                i = here->BJTemitPrimeColPrimePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BJTemitPrimeColPrimeStructPtr = matched ;
                here->BJTemitPrimeColPrimePtr = matched->CSC ;
            }

            if ((here->BJTemitPrimeNode != 0) && (here->BJTbasePrimeNode != 0))
            {
                i = here->BJTemitPrimeBasePrimePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BJTemitPrimeBasePrimeStructPtr = matched ;
                here->BJTemitPrimeBasePrimePtr = matched->CSC ;
            }

            if ((here->BJTcolNode != 0) && (here->BJTcolNode != 0))
            {
                i = here->BJTcolColPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BJTcolColStructPtr = matched ;
                here->BJTcolColPtr = matched->CSC ;
            }

            if ((here->BJTbaseNode != 0) && (here->BJTbaseNode != 0))
            {
                i = here->BJTbaseBasePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BJTbaseBaseStructPtr = matched ;
                here->BJTbaseBasePtr = matched->CSC ;
            }

            if ((here->BJTemitNode != 0) && (here->BJTemitNode != 0))
            {
                i = here->BJTemitEmitPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BJTemitEmitStructPtr = matched ;
                here->BJTemitEmitPtr = matched->CSC ;
            }

            if ((here->BJTcolPrimeNode != 0) && (here->BJTcolPrimeNode != 0))
            {
                i = here->BJTcolPrimeColPrimePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BJTcolPrimeColPrimeStructPtr = matched ;
                here->BJTcolPrimeColPrimePtr = matched->CSC ;
            }

            if ((here->BJTbasePrimeNode != 0) && (here->BJTbasePrimeNode != 0))
            {
                i = here->BJTbasePrimeBasePrimePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BJTbasePrimeBasePrimeStructPtr = matched ;
                here->BJTbasePrimeBasePrimePtr = matched->CSC ;
            }

            if ((here->BJTemitPrimeNode != 0) && (here->BJTemitPrimeNode != 0))
            {
                i = here->BJTemitPrimeEmitPrimePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BJTemitPrimeEmitPrimeStructPtr = matched ;
                here->BJTemitPrimeEmitPrimePtr = matched->CSC ;
            }

            if ((here->BJTsubstNode != 0) && (here->BJTsubstNode != 0))
            {
                i = here->BJTsubstSubstPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BJTsubstSubstStructPtr = matched ;
                here->BJTsubstSubstPtr = matched->CSC ;
            }

            if ((here->BJTsubstConNode != 0) && (here->BJTsubstNode != 0))
            {
                i = here->BJTsubstConSubstPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BJTsubstConSubstStructPtr = matched ;
                here->BJTsubstConSubstPtr = matched->CSC ;
            }

            if ((here->BJTsubstNode != 0) && (here->BJTsubstConNode != 0))
            {
                i = here->BJTsubstSubstConPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BJTsubstSubstConStructPtr = matched ;
                here->BJTsubstSubstConPtr = matched->CSC ;
            }

            if ((here->BJTbaseNode != 0) && (here->BJTcolPrimeNode != 0))
            {
                i = here->BJTbaseColPrimePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BJTbaseColPrimeStructPtr = matched ;
                here->BJTbaseColPrimePtr = matched->CSC ;
            }

            if ((here->BJTcolPrimeNode != 0) && (here->BJTbaseNode != 0))
            {
                i = here->BJTcolPrimeBasePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BJTcolPrimeBaseStructPtr = matched ;
                here->BJTcolPrimeBasePtr = matched->CSC ;
            }

        }
    }

    return (OK) ;
}

int
BJTbindCSCComplex (GENmodel *inModel, CKTcircuit *ckt)
{
    BJTmodel *model = (BJTmodel *)inModel ;
    BJTinstance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the BJT models */
    for ( ; model != NULL ; model = model->BJTnextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->BJTinstances ; here != NULL ; here = here->BJTnextInstance)
        {
            if ((here->BJTcolNode != 0) && (here->BJTcolPrimeNode != 0))
                here->BJTcolColPrimePtr = here->BJTcolColPrimeStructPtr->CSC_Complex ;

            if ((here->BJTbaseNode != 0) && (here->BJTbasePrimeNode != 0))
                here->BJTbaseBasePrimePtr = here->BJTbaseBasePrimeStructPtr->CSC_Complex ;

            if ((here->BJTemitNode != 0) && (here->BJTemitPrimeNode != 0))
                here->BJTemitEmitPrimePtr = here->BJTemitEmitPrimeStructPtr->CSC_Complex ;

            if ((here->BJTcolPrimeNode != 0) && (here->BJTcolNode != 0))
                here->BJTcolPrimeColPtr = here->BJTcolPrimeColStructPtr->CSC_Complex ;

            if ((here->BJTcolPrimeNode != 0) && (here->BJTbasePrimeNode != 0))
                here->BJTcolPrimeBasePrimePtr = here->BJTcolPrimeBasePrimeStructPtr->CSC_Complex ;

            if ((here->BJTcolPrimeNode != 0) && (here->BJTemitPrimeNode != 0))
                here->BJTcolPrimeEmitPrimePtr = here->BJTcolPrimeEmitPrimeStructPtr->CSC_Complex ;

            if ((here->BJTbasePrimeNode != 0) && (here->BJTbaseNode != 0))
                here->BJTbasePrimeBasePtr = here->BJTbasePrimeBaseStructPtr->CSC_Complex ;

            if ((here->BJTbasePrimeNode != 0) && (here->BJTcolPrimeNode != 0))
                here->BJTbasePrimeColPrimePtr = here->BJTbasePrimeColPrimeStructPtr->CSC_Complex ;

            if ((here->BJTbasePrimeNode != 0) && (here->BJTemitPrimeNode != 0))
                here->BJTbasePrimeEmitPrimePtr = here->BJTbasePrimeEmitPrimeStructPtr->CSC_Complex ;

            if ((here->BJTemitPrimeNode != 0) && (here->BJTemitNode != 0))
                here->BJTemitPrimeEmitPtr = here->BJTemitPrimeEmitStructPtr->CSC_Complex ;

            if ((here->BJTemitPrimeNode != 0) && (here->BJTcolPrimeNode != 0))
                here->BJTemitPrimeColPrimePtr = here->BJTemitPrimeColPrimeStructPtr->CSC_Complex ;

            if ((here->BJTemitPrimeNode != 0) && (here->BJTbasePrimeNode != 0))
                here->BJTemitPrimeBasePrimePtr = here->BJTemitPrimeBasePrimeStructPtr->CSC_Complex ;

            if ((here->BJTcolNode != 0) && (here->BJTcolNode != 0))
                here->BJTcolColPtr = here->BJTcolColStructPtr->CSC_Complex ;

            if ((here->BJTbaseNode != 0) && (here->BJTbaseNode != 0))
                here->BJTbaseBasePtr = here->BJTbaseBaseStructPtr->CSC_Complex ;

            if ((here->BJTemitNode != 0) && (here->BJTemitNode != 0))
                here->BJTemitEmitPtr = here->BJTemitEmitStructPtr->CSC_Complex ;

            if ((here->BJTcolPrimeNode != 0) && (here->BJTcolPrimeNode != 0))
                here->BJTcolPrimeColPrimePtr = here->BJTcolPrimeColPrimeStructPtr->CSC_Complex ;

            if ((here->BJTbasePrimeNode != 0) && (here->BJTbasePrimeNode != 0))
                here->BJTbasePrimeBasePrimePtr = here->BJTbasePrimeBasePrimeStructPtr->CSC_Complex ;

            if ((here->BJTemitPrimeNode != 0) && (here->BJTemitPrimeNode != 0))
                here->BJTemitPrimeEmitPrimePtr = here->BJTemitPrimeEmitPrimeStructPtr->CSC_Complex ;

            if ((here->BJTsubstNode != 0) && (here->BJTsubstNode != 0))
                here->BJTsubstSubstPtr = here->BJTsubstSubstStructPtr->CSC_Complex ;

            if ((here->BJTsubstConNode != 0) && (here->BJTsubstNode != 0))
                here->BJTsubstConSubstPtr = here->BJTsubstConSubstStructPtr->CSC_Complex ;

            if ((here->BJTsubstNode != 0) && (here->BJTsubstConNode != 0))
                here->BJTsubstSubstConPtr = here->BJTsubstSubstConStructPtr->CSC_Complex ;

            if ((here->BJTbaseNode != 0) && (here->BJTcolPrimeNode != 0))
                here->BJTbaseColPrimePtr = here->BJTbaseColPrimeStructPtr->CSC_Complex ;

            if ((here->BJTcolPrimeNode != 0) && (here->BJTbaseNode != 0))
                here->BJTcolPrimeBasePtr = here->BJTcolPrimeBaseStructPtr->CSC_Complex ;

        }
    }

    return (OK) ;
}

int
BJTbindCSCComplexToReal (GENmodel *inModel, CKTcircuit *ckt)
{
    BJTmodel *model = (BJTmodel *)inModel ;
    BJTinstance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the BJT models */
    for ( ; model != NULL ; model = model->BJTnextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->BJTinstances ; here != NULL ; here = here->BJTnextInstance)
        {
            if ((here->BJTcolNode != 0) && (here->BJTcolPrimeNode != 0))
                here->BJTcolColPrimePtr = here->BJTcolColPrimeStructPtr->CSC ;

            if ((here->BJTbaseNode != 0) && (here->BJTbasePrimeNode != 0))
                here->BJTbaseBasePrimePtr = here->BJTbaseBasePrimeStructPtr->CSC ;

            if ((here->BJTemitNode != 0) && (here->BJTemitPrimeNode != 0))
                here->BJTemitEmitPrimePtr = here->BJTemitEmitPrimeStructPtr->CSC ;

            if ((here->BJTcolPrimeNode != 0) && (here->BJTcolNode != 0))
                here->BJTcolPrimeColPtr = here->BJTcolPrimeColStructPtr->CSC ;

            if ((here->BJTcolPrimeNode != 0) && (here->BJTbasePrimeNode != 0))
                here->BJTcolPrimeBasePrimePtr = here->BJTcolPrimeBasePrimeStructPtr->CSC ;

            if ((here->BJTcolPrimeNode != 0) && (here->BJTemitPrimeNode != 0))
                here->BJTcolPrimeEmitPrimePtr = here->BJTcolPrimeEmitPrimeStructPtr->CSC ;

            if ((here->BJTbasePrimeNode != 0) && (here->BJTbaseNode != 0))
                here->BJTbasePrimeBasePtr = here->BJTbasePrimeBaseStructPtr->CSC ;

            if ((here->BJTbasePrimeNode != 0) && (here->BJTcolPrimeNode != 0))
                here->BJTbasePrimeColPrimePtr = here->BJTbasePrimeColPrimeStructPtr->CSC ;

            if ((here->BJTbasePrimeNode != 0) && (here->BJTemitPrimeNode != 0))
                here->BJTbasePrimeEmitPrimePtr = here->BJTbasePrimeEmitPrimeStructPtr->CSC ;

            if ((here->BJTemitPrimeNode != 0) && (here->BJTemitNode != 0))
                here->BJTemitPrimeEmitPtr = here->BJTemitPrimeEmitStructPtr->CSC ;

            if ((here->BJTemitPrimeNode != 0) && (here->BJTcolPrimeNode != 0))
                here->BJTemitPrimeColPrimePtr = here->BJTemitPrimeColPrimeStructPtr->CSC ;

            if ((here->BJTemitPrimeNode != 0) && (here->BJTbasePrimeNode != 0))
                here->BJTemitPrimeBasePrimePtr = here->BJTemitPrimeBasePrimeStructPtr->CSC ;

            if ((here->BJTcolNode != 0) && (here->BJTcolNode != 0))
                here->BJTcolColPtr = here->BJTcolColStructPtr->CSC ;

            if ((here->BJTbaseNode != 0) && (here->BJTbaseNode != 0))
                here->BJTbaseBasePtr = here->BJTbaseBaseStructPtr->CSC ;

            if ((here->BJTemitNode != 0) && (here->BJTemitNode != 0))
                here->BJTemitEmitPtr = here->BJTemitEmitStructPtr->CSC ;

            if ((here->BJTcolPrimeNode != 0) && (here->BJTcolPrimeNode != 0))
                here->BJTcolPrimeColPrimePtr = here->BJTcolPrimeColPrimeStructPtr->CSC ;

            if ((here->BJTbasePrimeNode != 0) && (here->BJTbasePrimeNode != 0))
                here->BJTbasePrimeBasePrimePtr = here->BJTbasePrimeBasePrimeStructPtr->CSC ;

            if ((here->BJTemitPrimeNode != 0) && (here->BJTemitPrimeNode != 0))
                here->BJTemitPrimeEmitPrimePtr = here->BJTemitPrimeEmitPrimeStructPtr->CSC ;

            if ((here->BJTsubstNode != 0) && (here->BJTsubstNode != 0))
                here->BJTsubstSubstPtr = here->BJTsubstSubstStructPtr->CSC ;

            if ((here->BJTsubstConNode != 0) && (here->BJTsubstNode != 0))
                here->BJTsubstConSubstPtr = here->BJTsubstConSubstStructPtr->CSC ;

            if ((here->BJTsubstNode != 0) && (here->BJTsubstConNode != 0))
                here->BJTsubstSubstConPtr = here->BJTsubstSubstConStructPtr->CSC ;

            if ((here->BJTbaseNode != 0) && (here->BJTcolPrimeNode != 0))
                here->BJTbaseColPrimePtr = here->BJTbaseColPrimeStructPtr->CSC ;

            if ((here->BJTcolPrimeNode != 0) && (here->BJTbaseNode != 0))
                here->BJTcolPrimeBasePtr = here->BJTcolPrimeBaseStructPtr->CSC ;

        }
    }

    return (OK) ;
}
