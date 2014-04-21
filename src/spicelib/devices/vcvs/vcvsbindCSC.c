/**********
Author: 2013 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "vcvsdefs.h"
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
VCVSbindCSC (GENmodel *inModel, CKTcircuit *ckt)
{
    VCVSmodel *model = (VCVSmodel *)inModel ;
    VCVSinstance *here ;
    double *i ;
    BindElement *matched, *BindStruct ;
    size_t nz ;

    BindStruct = ckt->CKTmatrix->CKTbindStruct ;
    nz = (size_t)ckt->CKTmatrix->CKTklunz ;

    /* loop through all the VCVS models */
    for ( ; model != NULL ; model = model->VCVSnextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->VCVSinstances ; here != NULL ; here = here->VCVSnextInstance)
        {
            if ((here-> VCVSposNode != 0) && (here-> VCVSbranch != 0))
            {
                i = here->VCVSposIbrptr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->VCVSposIbrptrStructPtr = matched ;
                here->VCVSposIbrptr = matched->CSC ;
            }

            if ((here-> VCVSnegNode != 0) && (here-> VCVSbranch != 0))
            {
                i = here->VCVSnegIbrptr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->VCVSnegIbrptrStructPtr = matched ;
                here->VCVSnegIbrptr = matched->CSC ;
            }

            if ((here-> VCVSbranch != 0) && (here-> VCVSnegNode != 0))
            {
                i = here->VCVSibrNegptr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->VCVSibrNegptrStructPtr = matched ;
                here->VCVSibrNegptr = matched->CSC ;
            }

            if ((here-> VCVSbranch != 0) && (here-> VCVSposNode != 0))
            {
                i = here->VCVSibrPosptr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->VCVSibrPosptrStructPtr = matched ;
                here->VCVSibrPosptr = matched->CSC ;
            }

            if ((here-> VCVSbranch != 0) && (here-> VCVScontPosNode != 0))
            {
                i = here->VCVSibrContPosptr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->VCVSibrContPosptrStructPtr = matched ;
                here->VCVSibrContPosptr = matched->CSC ;
            }

            if ((here-> VCVSbranch != 0) && (here-> VCVScontNegNode != 0))
            {
                i = here->VCVSibrContNegptr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->VCVSibrContNegptrStructPtr = matched ;
                here->VCVSibrContNegptr = matched->CSC ;
            }

        }
    }

    return (OK) ;
}

int
VCVSbindCSCComplex (GENmodel *inModel, CKTcircuit *ckt)
{
    VCVSmodel *model = (VCVSmodel *)inModel ;
    VCVSinstance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the VCVS models */
    for ( ; model != NULL ; model = model->VCVSnextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->VCVSinstances ; here != NULL ; here = here->VCVSnextInstance)
        {
            if ((here-> VCVSposNode != 0) && (here-> VCVSbranch != 0))
                here->VCVSposIbrptr = here->VCVSposIbrptrStructPtr->CSC_Complex ;

            if ((here-> VCVSnegNode != 0) && (here-> VCVSbranch != 0))
                here->VCVSnegIbrptr = here->VCVSnegIbrptrStructPtr->CSC_Complex ;

            if ((here-> VCVSbranch != 0) && (here-> VCVSnegNode != 0))
                here->VCVSibrNegptr = here->VCVSibrNegptrStructPtr->CSC_Complex ;

            if ((here-> VCVSbranch != 0) && (here-> VCVSposNode != 0))
                here->VCVSibrPosptr = here->VCVSibrPosptrStructPtr->CSC_Complex ;

            if ((here-> VCVSbranch != 0) && (here-> VCVScontPosNode != 0))
                here->VCVSibrContPosptr = here->VCVSibrContPosptrStructPtr->CSC_Complex ;

            if ((here-> VCVSbranch != 0) && (here-> VCVScontNegNode != 0))
                here->VCVSibrContNegptr = here->VCVSibrContNegptrStructPtr->CSC_Complex ;

        }
    }

    return (OK) ;
}

int
VCVSbindCSCComplexToReal (GENmodel *inModel, CKTcircuit *ckt)
{
    VCVSmodel *model = (VCVSmodel *)inModel ;
    VCVSinstance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the VCVS models */
    for ( ; model != NULL ; model = model->VCVSnextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->VCVSinstances ; here != NULL ; here = here->VCVSnextInstance)
        {
            if ((here-> VCVSposNode != 0) && (here-> VCVSbranch != 0))
                here->VCVSposIbrptr = here->VCVSposIbrptrStructPtr->CSC ;

            if ((here-> VCVSnegNode != 0) && (here-> VCVSbranch != 0))
                here->VCVSnegIbrptr = here->VCVSnegIbrptrStructPtr->CSC ;

            if ((here-> VCVSbranch != 0) && (here-> VCVSnegNode != 0))
                here->VCVSibrNegptr = here->VCVSibrNegptrStructPtr->CSC ;

            if ((here-> VCVSbranch != 0) && (here-> VCVSposNode != 0))
                here->VCVSibrPosptr = here->VCVSibrPosptrStructPtr->CSC ;

            if ((here-> VCVSbranch != 0) && (here-> VCVScontPosNode != 0))
                here->VCVSibrContPosptr = here->VCVSibrContPosptrStructPtr->CSC ;

            if ((here-> VCVSbranch != 0) && (here-> VCVScontNegNode != 0))
                here->VCVSibrContNegptr = here->VCVSibrContNegptrStructPtr->CSC ;

        }
    }

    return (OK) ;
}
