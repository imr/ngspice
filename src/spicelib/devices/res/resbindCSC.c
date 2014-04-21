/**********
Author: 2013 Francesco Lannutti
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

    /* loop through all the RES models */
    for ( ; model != NULL ; model = model->RESnextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->RESinstances ; here != NULL ; here = here->RESnextInstance)
        {
            if ((here-> RESposNode != 0) && (here-> RESposNode != 0))
            {
                i = here->RESposPosptr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->RESposPosptrStructPtr = matched ;
                here->RESposPosptr = matched->CSC ;
            }

            if ((here-> RESnegNode != 0) && (here-> RESnegNode != 0))
            {
                i = here->RESnegNegptr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->RESnegNegptrStructPtr = matched ;
                here->RESnegNegptr = matched->CSC ;
            }

            if ((here-> RESposNode != 0) && (here-> RESnegNode != 0))
            {
                i = here->RESposNegptr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->RESposNegptrStructPtr = matched ;
                here->RESposNegptr = matched->CSC ;
            }

            if ((here-> RESnegNode != 0) && (here-> RESposNode != 0))
            {
                i = here->RESnegPosptr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->RESnegPosptrStructPtr = matched ;
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

    /* loop through all the RES models */
    for ( ; model != NULL ; model = model->RESnextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->RESinstances ; here != NULL ; here = here->RESnextInstance)
        {
            if ((here-> RESposNode != 0) && (here-> RESposNode != 0))
                here->RESposPosptr = here->RESposPosptrStructPtr->CSC_Complex ;

            if ((here-> RESnegNode != 0) && (here-> RESnegNode != 0))
                here->RESnegNegptr = here->RESnegNegptrStructPtr->CSC_Complex ;

            if ((here-> RESposNode != 0) && (here-> RESnegNode != 0))
                here->RESposNegptr = here->RESposNegptrStructPtr->CSC_Complex ;

            if ((here-> RESnegNode != 0) && (here-> RESposNode != 0))
                here->RESnegPosptr = here->RESnegPosptrStructPtr->CSC_Complex ;

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

    /* loop through all the RES models */
    for ( ; model != NULL ; model = model->RESnextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->RESinstances ; here != NULL ; here = here->RESnextInstance)
        {
            if ((here-> RESposNode != 0) && (here-> RESposNode != 0))
                here->RESposPosptr = here->RESposPosptrStructPtr->CSC ;

            if ((here-> RESnegNode != 0) && (here-> RESnegNode != 0))
                here->RESnegNegptr = here->RESnegNegptrStructPtr->CSC ;

            if ((here-> RESposNode != 0) && (here-> RESnegNode != 0))
                here->RESposNegptr = here->RESposNegptrStructPtr->CSC ;

            if ((here-> RESnegNode != 0) && (here-> RESposNode != 0))
                here->RESnegPosptr = here->RESnegPosptrStructPtr->CSC ;

        }
    }

    return (OK) ;
}
