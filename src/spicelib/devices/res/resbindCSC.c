/**********
Author: 2012 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "resdefs.h"
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
RESbindCSC (GENmodel *inModel, CKTcircuit *ckt)
{
    RESmodel *model = (RESmodel *)inModel ;
    RESinstance *here ;
    double *i ;
    BindElement *matched, *BindStruct ;
    size_t nz ;

    BindStruct = ckt->CKTmatrix->CKTbindStruct ;
    nz = (size_t)ckt->CKTmatrix->CKTklunz ;

    /* loop through all the resistor models */
    for ( ; model != NULL ; model = model->RESnextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->RESinstances ; here != NULL ; here = here->RESnextInstance)
        {
            if ((here->RESposNode != 0) && (here->RESposNode != 0))
            {
                i = here->RESposPosptr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->RESposPosStructPtr = matched ;
                here->RESposPosptr = matched->CSC ;
            }

            if ((here->RESnegNode != 0) && (here->RESnegNode != 0))
            {
                i = here->RESnegNegptr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->RESnegNegStructPtr = matched ;
                here->RESnegNegptr = matched->CSC ;
            }

            if ((here->RESposNode != 0) && (here->RESnegNode != 0))
            {
                i = here->RESposNegptr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->RESposNegStructPtr = matched ;
                here->RESposNegptr = matched->CSC ;
            }

            if ((here->RESnegNode != 0) && (here->RESposNode != 0))
            {
                i = here->RESnegPosptr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->RESnegPosStructPtr = matched ;
                here->RESnegPosptr = matched->CSC ;
            }
        }
    }

    return (OK) ;
}

int
RESbindCSCComplex (GENmodel *inModel, CKTcircuit *ckt)
{
    RESmodel *model = (RESmodel *)inModel ;
    RESinstance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the resistor models */
    for ( ; model != NULL ; model = model->RESnextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->RESinstances ; here != NULL ; here = here->RESnextInstance)
        {
            if ((here->RESposNode != 0) && (here->RESposNode != 0))
                here->RESposPosptr = here->RESposPosStructPtr->CSC_Complex ;

            if ((here->RESnegNode != 0) && (here->RESnegNode != 0))
                here->RESnegNegptr = here->RESnegNegStructPtr->CSC_Complex ;

            if ((here->RESposNode != 0) && (here->RESnegNode != 0))
                here->RESposNegptr = here->RESposNegStructPtr->CSC_Complex ;

            if ((here->RESnegNode != 0) && (here->RESposNode != 0))
                here->RESnegPosptr = here->RESnegPosStructPtr->CSC_Complex ;
        }
    }

    return (OK) ;
}

int
RESbindCSCComplexToReal (GENmodel *inModel, CKTcircuit *ckt)
{
    RESmodel *model = (RESmodel *)inModel ;
    RESinstance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the resistor models */
    for ( ; model != NULL ; model = model->RESnextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->RESinstances ; here != NULL ; here = here->RESnextInstance)
        {
            if ((here->RESposNode != 0) && (here->RESposNode != 0))
                here->RESposPosptr = here->RESposPosStructPtr->CSC ;

            if ((here->RESnegNode != 0) && (here->RESnegNode != 0))
                here->RESnegNegptr = here->RESnegNegStructPtr->CSC ;

            if ((here->RESposNode != 0) && (here->RESnegNode != 0))
                here->RESposNegptr = here->RESposNegStructPtr->CSC ;

            if ((here->RESnegNode != 0) && (here->RESposNode != 0))
                here->RESnegPosptr = here->RESnegPosStructPtr->CSC ;
        }
    }

    return (OK) ;
}
