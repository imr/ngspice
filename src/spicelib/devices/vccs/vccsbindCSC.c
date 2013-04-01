/**********
Author: 2013 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "vccsdefs.h"
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
VCCSbindCSC (GENmodel *inModel, CKTcircuit *ckt)
{
    VCCSmodel *model = (VCCSmodel *)inModel ;
    VCCSinstance *here ;
    double *i ;
    BindElement *matched, *BindStruct ;
    size_t nz ;

    BindStruct = ckt->CKTmatrix->CKTbindStruct ;
    nz = (size_t)ckt->CKTmatrix->CKTklunz ;

    /* loop through all the VCCS models */
    for ( ; model != NULL ; model = model->VCCSnextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->VCCSinstances ; here != NULL ; here = here->VCCSnextInstance)
        {
            if ((here-> VCCSposNode != 0) && (here-> VCCScontPosNode != 0))
            {
                i = here->VCCSposContPosptr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->VCCSposContPosptrStructPtr = matched ;
                here->VCCSposContPosptr = matched->CSC ;
            }

            if ((here-> VCCSposNode != 0) && (here-> VCCScontNegNode != 0))
            {
                i = here->VCCSposContNegptr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->VCCSposContNegptrStructPtr = matched ;
                here->VCCSposContNegptr = matched->CSC ;
            }

            if ((here-> VCCSnegNode != 0) && (here-> VCCScontPosNode != 0))
            {
                i = here->VCCSnegContPosptr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->VCCSnegContPosptrStructPtr = matched ;
                here->VCCSnegContPosptr = matched->CSC ;
            }

            if ((here-> VCCSnegNode != 0) && (here-> VCCScontNegNode != 0))
            {
                i = here->VCCSnegContNegptr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->VCCSnegContNegptrStructPtr = matched ;
                here->VCCSnegContNegptr = matched->CSC ;
            }

        }
    }

    return (OK) ;
}

int
VCCSbindCSCComplex (GENmodel *inModel, CKTcircuit *ckt)
{
    VCCSmodel *model = (VCCSmodel *)inModel ;
    VCCSinstance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the VCCS models */
    for ( ; model != NULL ; model = model->VCCSnextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->VCCSinstances ; here != NULL ; here = here->VCCSnextInstance)
        {
            if ((here-> VCCSposNode != 0) && (here-> VCCScontPosNode != 0))
                here->VCCSposContPosptr = here->VCCSposContPosptrStructPtr->CSC_Complex ;

            if ((here-> VCCSposNode != 0) && (here-> VCCScontNegNode != 0))
                here->VCCSposContNegptr = here->VCCSposContNegptrStructPtr->CSC_Complex ;

            if ((here-> VCCSnegNode != 0) && (here-> VCCScontPosNode != 0))
                here->VCCSnegContPosptr = here->VCCSnegContPosptrStructPtr->CSC_Complex ;

            if ((here-> VCCSnegNode != 0) && (here-> VCCScontNegNode != 0))
                here->VCCSnegContNegptr = here->VCCSnegContNegptrStructPtr->CSC_Complex ;

        }
    }

    return (OK) ;
}

int
VCCSbindCSCComplexToReal (GENmodel *inModel, CKTcircuit *ckt)
{
    VCCSmodel *model = (VCCSmodel *)inModel ;
    VCCSinstance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the VCCS models */
    for ( ; model != NULL ; model = model->VCCSnextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->VCCSinstances ; here != NULL ; here = here->VCCSnextInstance)
        {
            if ((here-> VCCSposNode != 0) && (here-> VCCScontPosNode != 0))
                here->VCCSposContPosptr = here->VCCSposContPosptrStructPtr->CSC ;

            if ((here-> VCCSposNode != 0) && (here-> VCCScontNegNode != 0))
                here->VCCSposContNegptr = here->VCCSposContNegptrStructPtr->CSC ;

            if ((here-> VCCSnegNode != 0) && (here-> VCCScontPosNode != 0))
                here->VCCSnegContPosptr = here->VCCSnegContPosptrStructPtr->CSC ;

            if ((here-> VCCSnegNode != 0) && (here-> VCCScontNegNode != 0))
                here->VCCSnegContNegptr = here->VCCSnegContNegptrStructPtr->CSC ;

        }
    }

    return (OK) ;
}
