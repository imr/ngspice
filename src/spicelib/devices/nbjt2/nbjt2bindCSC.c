/**********
Author: 2013 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "nbjt2def.h"
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
NBJT2bindCSC (GENmodel *inModel, CKTcircuit *ckt)
{
    NBJT2model *model = (NBJT2model *)inModel ;
    NBJT2instance *here ;
    double *i ;
    BindElement *matched, *BindStruct ;
    size_t nz ;

    BindStruct = ckt->CKTmatrix->CKTbindStruct ;
    nz = (size_t)ckt->CKTmatrix->CKTklunz ;

    /* loop through all the NBJT2 models */
    for ( ; model != NULL ; model = model->NBJT2nextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->NBJT2instances ; here != NULL ; here = here->NBJT2nextInstance)
        {
            if ((here-> NBJT2colNode != 0) && (here-> NBJT2colNode != 0))
            {
                i = here->NBJT2colColPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->NBJT2colColStructPtr = matched ;
                here->NBJT2colColPtr = matched->CSC ;
            }

            if ((here-> NBJT2colNode != 0) && (here-> NBJT2baseNode != 0))
            {
                i = here->NBJT2colBasePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->NBJT2colBaseStructPtr = matched ;
                here->NBJT2colBasePtr = matched->CSC ;
            }

            if ((here-> NBJT2colNode != 0) && (here-> NBJT2emitNode != 0))
            {
                i = here->NBJT2colEmitPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->NBJT2colEmitStructPtr = matched ;
                here->NBJT2colEmitPtr = matched->CSC ;
            }

            if ((here-> NBJT2baseNode != 0) && (here-> NBJT2colNode != 0))
            {
                i = here->NBJT2baseColPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->NBJT2baseColStructPtr = matched ;
                here->NBJT2baseColPtr = matched->CSC ;
            }

            if ((here-> NBJT2baseNode != 0) && (here-> NBJT2baseNode != 0))
            {
                i = here->NBJT2baseBasePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->NBJT2baseBaseStructPtr = matched ;
                here->NBJT2baseBasePtr = matched->CSC ;
            }

            if ((here-> NBJT2baseNode != 0) && (here-> NBJT2emitNode != 0))
            {
                i = here->NBJT2baseEmitPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->NBJT2baseEmitStructPtr = matched ;
                here->NBJT2baseEmitPtr = matched->CSC ;
            }

            if ((here-> NBJT2emitNode != 0) && (here-> NBJT2colNode != 0))
            {
                i = here->NBJT2emitColPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->NBJT2emitColStructPtr = matched ;
                here->NBJT2emitColPtr = matched->CSC ;
            }

            if ((here-> NBJT2emitNode != 0) && (here-> NBJT2baseNode != 0))
            {
                i = here->NBJT2emitBasePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->NBJT2emitBaseStructPtr = matched ;
                here->NBJT2emitBasePtr = matched->CSC ;
            }

            if ((here-> NBJT2emitNode != 0) && (here-> NBJT2emitNode != 0))
            {
                i = here->NBJT2emitEmitPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->NBJT2emitEmitStructPtr = matched ;
                here->NBJT2emitEmitPtr = matched->CSC ;
            }

        }
    }

    return (OK) ;
}

int
NBJT2bindCSCComplex (GENmodel *inModel, CKTcircuit *ckt)
{
    NBJT2model *model = (NBJT2model *)inModel ;
    NBJT2instance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the NBJT2 models */
    for ( ; model != NULL ; model = model->NBJT2nextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->NBJT2instances ; here != NULL ; here = here->NBJT2nextInstance)
        {
            if ((here-> NBJT2colNode != 0) && (here-> NBJT2colNode != 0))
                here->NBJT2colColPtr = here->NBJT2colColStructPtr->CSC_Complex ;

            if ((here-> NBJT2colNode != 0) && (here-> NBJT2baseNode != 0))
                here->NBJT2colBasePtr = here->NBJT2colBaseStructPtr->CSC_Complex ;

            if ((here-> NBJT2colNode != 0) && (here-> NBJT2emitNode != 0))
                here->NBJT2colEmitPtr = here->NBJT2colEmitStructPtr->CSC_Complex ;

            if ((here-> NBJT2baseNode != 0) && (here-> NBJT2colNode != 0))
                here->NBJT2baseColPtr = here->NBJT2baseColStructPtr->CSC_Complex ;

            if ((here-> NBJT2baseNode != 0) && (here-> NBJT2baseNode != 0))
                here->NBJT2baseBasePtr = here->NBJT2baseBaseStructPtr->CSC_Complex ;

            if ((here-> NBJT2baseNode != 0) && (here-> NBJT2emitNode != 0))
                here->NBJT2baseEmitPtr = here->NBJT2baseEmitStructPtr->CSC_Complex ;

            if ((here-> NBJT2emitNode != 0) && (here-> NBJT2colNode != 0))
                here->NBJT2emitColPtr = here->NBJT2emitColStructPtr->CSC_Complex ;

            if ((here-> NBJT2emitNode != 0) && (here-> NBJT2baseNode != 0))
                here->NBJT2emitBasePtr = here->NBJT2emitBaseStructPtr->CSC_Complex ;

            if ((here-> NBJT2emitNode != 0) && (here-> NBJT2emitNode != 0))
                here->NBJT2emitEmitPtr = here->NBJT2emitEmitStructPtr->CSC_Complex ;

        }
    }

    return (OK) ;
}

int
NBJT2bindCSCComplexToReal (GENmodel *inModel, CKTcircuit *ckt)
{
    NBJT2model *model = (NBJT2model *)inModel ;
    NBJT2instance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the NBJT2 models */
    for ( ; model != NULL ; model = model->NBJT2nextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->NBJT2instances ; here != NULL ; here = here->NBJT2nextInstance)
        {
            if ((here-> NBJT2colNode != 0) && (here-> NBJT2colNode != 0))
                here->NBJT2colColPtr = here->NBJT2colColStructPtr->CSC ;

            if ((here-> NBJT2colNode != 0) && (here-> NBJT2baseNode != 0))
                here->NBJT2colBasePtr = here->NBJT2colBaseStructPtr->CSC ;

            if ((here-> NBJT2colNode != 0) && (here-> NBJT2emitNode != 0))
                here->NBJT2colEmitPtr = here->NBJT2colEmitStructPtr->CSC ;

            if ((here-> NBJT2baseNode != 0) && (here-> NBJT2colNode != 0))
                here->NBJT2baseColPtr = here->NBJT2baseColStructPtr->CSC ;

            if ((here-> NBJT2baseNode != 0) && (here-> NBJT2baseNode != 0))
                here->NBJT2baseBasePtr = here->NBJT2baseBaseStructPtr->CSC ;

            if ((here-> NBJT2baseNode != 0) && (here-> NBJT2emitNode != 0))
                here->NBJT2baseEmitPtr = here->NBJT2baseEmitStructPtr->CSC ;

            if ((here-> NBJT2emitNode != 0) && (here-> NBJT2colNode != 0))
                here->NBJT2emitColPtr = here->NBJT2emitColStructPtr->CSC ;

            if ((here-> NBJT2emitNode != 0) && (here-> NBJT2baseNode != 0))
                here->NBJT2emitBasePtr = here->NBJT2emitBaseStructPtr->CSC ;

            if ((here-> NBJT2emitNode != 0) && (here-> NBJT2emitNode != 0))
                here->NBJT2emitEmitPtr = here->NBJT2emitEmitStructPtr->CSC ;

        }
    }

    return (OK) ;
}
