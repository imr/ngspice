/**********
Author: 2013 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "bsim3v32def.h"
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
BSIM3v32bindCSC (GENmodel *inModel, CKTcircuit *ckt)
{
    BSIM3v32model *model = (BSIM3v32model *)inModel ;
    BSIM3v32instance *here ;
    double *i ;
    BindElement *matched, *BindStruct ;
    size_t nz ;

    BindStruct = ckt->CKTmatrix->CKTbindStruct ;
    nz = (size_t)ckt->CKTmatrix->CKTklunz ;

    /* loop through all the BSIM3v32 models */
    for ( ; model != NULL ; model = model->BSIM3v32nextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->BSIM3v32instances ; here != NULL ; here = here->BSIM3v32nextInstance)
        {
            if ((here-> BSIM3v32dNode != 0) && (here-> BSIM3v32dNode != 0))
            {
                i = here->BSIM3v32DdPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM3v32DdStructPtr = matched ;
                here->BSIM3v32DdPtr = matched->CSC ;
            }

            if ((here-> BSIM3v32gNode != 0) && (here-> BSIM3v32gNode != 0))
            {
                i = here->BSIM3v32GgPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM3v32GgStructPtr = matched ;
                here->BSIM3v32GgPtr = matched->CSC ;
            }

            if ((here-> BSIM3v32sNode != 0) && (here-> BSIM3v32sNode != 0))
            {
                i = here->BSIM3v32SsPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM3v32SsStructPtr = matched ;
                here->BSIM3v32SsPtr = matched->CSC ;
            }

            if ((here-> BSIM3v32bNode != 0) && (here-> BSIM3v32bNode != 0))
            {
                i = here->BSIM3v32BbPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM3v32BbStructPtr = matched ;
                here->BSIM3v32BbPtr = matched->CSC ;
            }

            if ((here-> BSIM3v32dNodePrime != 0) && (here-> BSIM3v32dNodePrime != 0))
            {
                i = here->BSIM3v32DPdpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM3v32DPdpStructPtr = matched ;
                here->BSIM3v32DPdpPtr = matched->CSC ;
            }

            if ((here-> BSIM3v32sNodePrime != 0) && (here-> BSIM3v32sNodePrime != 0))
            {
                i = here->BSIM3v32SPspPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM3v32SPspStructPtr = matched ;
                here->BSIM3v32SPspPtr = matched->CSC ;
            }

            if ((here-> BSIM3v32dNode != 0) && (here-> BSIM3v32dNodePrime != 0))
            {
                i = here->BSIM3v32DdpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM3v32DdpStructPtr = matched ;
                here->BSIM3v32DdpPtr = matched->CSC ;
            }

            if ((here-> BSIM3v32gNode != 0) && (here-> BSIM3v32bNode != 0))
            {
                i = here->BSIM3v32GbPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM3v32GbStructPtr = matched ;
                here->BSIM3v32GbPtr = matched->CSC ;
            }

            if ((here-> BSIM3v32gNode != 0) && (here-> BSIM3v32dNodePrime != 0))
            {
                i = here->BSIM3v32GdpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM3v32GdpStructPtr = matched ;
                here->BSIM3v32GdpPtr = matched->CSC ;
            }

            if ((here-> BSIM3v32gNode != 0) && (here-> BSIM3v32sNodePrime != 0))
            {
                i = here->BSIM3v32GspPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM3v32GspStructPtr = matched ;
                here->BSIM3v32GspPtr = matched->CSC ;
            }

            if ((here-> BSIM3v32sNode != 0) && (here-> BSIM3v32sNodePrime != 0))
            {
                i = here->BSIM3v32SspPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM3v32SspStructPtr = matched ;
                here->BSIM3v32SspPtr = matched->CSC ;
            }

            if ((here-> BSIM3v32bNode != 0) && (here-> BSIM3v32dNodePrime != 0))
            {
                i = here->BSIM3v32BdpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM3v32BdpStructPtr = matched ;
                here->BSIM3v32BdpPtr = matched->CSC ;
            }

            if ((here-> BSIM3v32bNode != 0) && (here-> BSIM3v32sNodePrime != 0))
            {
                i = here->BSIM3v32BspPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM3v32BspStructPtr = matched ;
                here->BSIM3v32BspPtr = matched->CSC ;
            }

            if ((here-> BSIM3v32dNodePrime != 0) && (here-> BSIM3v32sNodePrime != 0))
            {
                i = here->BSIM3v32DPspPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM3v32DPspStructPtr = matched ;
                here->BSIM3v32DPspPtr = matched->CSC ;
            }

            if ((here-> BSIM3v32dNodePrime != 0) && (here-> BSIM3v32dNode != 0))
            {
                i = here->BSIM3v32DPdPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM3v32DPdStructPtr = matched ;
                here->BSIM3v32DPdPtr = matched->CSC ;
            }

            if ((here-> BSIM3v32bNode != 0) && (here-> BSIM3v32gNode != 0))
            {
                i = here->BSIM3v32BgPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM3v32BgStructPtr = matched ;
                here->BSIM3v32BgPtr = matched->CSC ;
            }

            if ((here-> BSIM3v32dNodePrime != 0) && (here-> BSIM3v32gNode != 0))
            {
                i = here->BSIM3v32DPgPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM3v32DPgStructPtr = matched ;
                here->BSIM3v32DPgPtr = matched->CSC ;
            }

            if ((here-> BSIM3v32sNodePrime != 0) && (here-> BSIM3v32gNode != 0))
            {
                i = here->BSIM3v32SPgPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM3v32SPgStructPtr = matched ;
                here->BSIM3v32SPgPtr = matched->CSC ;
            }

            if ((here-> BSIM3v32sNodePrime != 0) && (here-> BSIM3v32sNode != 0))
            {
                i = here->BSIM3v32SPsPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM3v32SPsStructPtr = matched ;
                here->BSIM3v32SPsPtr = matched->CSC ;
            }

            if ((here-> BSIM3v32dNodePrime != 0) && (here-> BSIM3v32bNode != 0))
            {
                i = here->BSIM3v32DPbPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM3v32DPbStructPtr = matched ;
                here->BSIM3v32DPbPtr = matched->CSC ;
            }

            if ((here-> BSIM3v32sNodePrime != 0) && (here-> BSIM3v32bNode != 0))
            {
                i = here->BSIM3v32SPbPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM3v32SPbStructPtr = matched ;
                here->BSIM3v32SPbPtr = matched->CSC ;
            }

            if ((here-> BSIM3v32sNodePrime != 0) && (here-> BSIM3v32dNodePrime != 0))
            {
                i = here->BSIM3v32SPdpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM3v32SPdpStructPtr = matched ;
                here->BSIM3v32SPdpPtr = matched->CSC ;
            }

            if ((here-> BSIM3v32qNode != 0) && (here-> BSIM3v32qNode != 0))
            {
                i = here->BSIM3v32QqPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM3v32QqStructPtr = matched ;
                here->BSIM3v32QqPtr = matched->CSC ;
            }

            if ((here-> BSIM3v32qNode != 0) && (here-> BSIM3v32dNodePrime != 0))
            {
                i = here->BSIM3v32QdpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM3v32QdpStructPtr = matched ;
                here->BSIM3v32QdpPtr = matched->CSC ;
            }

            if ((here-> BSIM3v32qNode != 0) && (here-> BSIM3v32sNodePrime != 0))
            {
                i = here->BSIM3v32QspPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM3v32QspStructPtr = matched ;
                here->BSIM3v32QspPtr = matched->CSC ;
            }

            if ((here-> BSIM3v32qNode != 0) && (here-> BSIM3v32gNode != 0))
            {
                i = here->BSIM3v32QgPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM3v32QgStructPtr = matched ;
                here->BSIM3v32QgPtr = matched->CSC ;
            }

            if ((here-> BSIM3v32qNode != 0) && (here-> BSIM3v32bNode != 0))
            {
                i = here->BSIM3v32QbPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM3v32QbStructPtr = matched ;
                here->BSIM3v32QbPtr = matched->CSC ;
            }

            if ((here-> BSIM3v32dNodePrime != 0) && (here-> BSIM3v32qNode != 0))
            {
                i = here->BSIM3v32DPqPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM3v32DPqStructPtr = matched ;
                here->BSIM3v32DPqPtr = matched->CSC ;
            }

            if ((here-> BSIM3v32sNodePrime != 0) && (here-> BSIM3v32qNode != 0))
            {
                i = here->BSIM3v32SPqPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM3v32SPqStructPtr = matched ;
                here->BSIM3v32SPqPtr = matched->CSC ;
            }

            if ((here-> BSIM3v32gNode != 0) && (here-> BSIM3v32qNode != 0))
            {
                i = here->BSIM3v32GqPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM3v32GqStructPtr = matched ;
                here->BSIM3v32GqPtr = matched->CSC ;
            }

            if ((here-> BSIM3v32bNode != 0) && (here-> BSIM3v32qNode != 0))
            {
                i = here->BSIM3v32BqPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM3v32BqStructPtr = matched ;
                here->BSIM3v32BqPtr = matched->CSC ;
            }

        }
    }

    return (OK) ;
}

int
BSIM3v32bindCSCComplex (GENmodel *inModel, CKTcircuit *ckt)
{
    BSIM3v32model *model = (BSIM3v32model *)inModel ;
    BSIM3v32instance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the BSIM3v32 models */
    for ( ; model != NULL ; model = model->BSIM3v32nextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->BSIM3v32instances ; here != NULL ; here = here->BSIM3v32nextInstance)
        {
            if ((here-> BSIM3v32dNode != 0) && (here-> BSIM3v32dNode != 0))
                here->BSIM3v32DdPtr = here->BSIM3v32DdStructPtr->CSC_Complex ;

            if ((here-> BSIM3v32gNode != 0) && (here-> BSIM3v32gNode != 0))
                here->BSIM3v32GgPtr = here->BSIM3v32GgStructPtr->CSC_Complex ;

            if ((here-> BSIM3v32sNode != 0) && (here-> BSIM3v32sNode != 0))
                here->BSIM3v32SsPtr = here->BSIM3v32SsStructPtr->CSC_Complex ;

            if ((here-> BSIM3v32bNode != 0) && (here-> BSIM3v32bNode != 0))
                here->BSIM3v32BbPtr = here->BSIM3v32BbStructPtr->CSC_Complex ;

            if ((here-> BSIM3v32dNodePrime != 0) && (here-> BSIM3v32dNodePrime != 0))
                here->BSIM3v32DPdpPtr = here->BSIM3v32DPdpStructPtr->CSC_Complex ;

            if ((here-> BSIM3v32sNodePrime != 0) && (here-> BSIM3v32sNodePrime != 0))
                here->BSIM3v32SPspPtr = here->BSIM3v32SPspStructPtr->CSC_Complex ;

            if ((here-> BSIM3v32dNode != 0) && (here-> BSIM3v32dNodePrime != 0))
                here->BSIM3v32DdpPtr = here->BSIM3v32DdpStructPtr->CSC_Complex ;

            if ((here-> BSIM3v32gNode != 0) && (here-> BSIM3v32bNode != 0))
                here->BSIM3v32GbPtr = here->BSIM3v32GbStructPtr->CSC_Complex ;

            if ((here-> BSIM3v32gNode != 0) && (here-> BSIM3v32dNodePrime != 0))
                here->BSIM3v32GdpPtr = here->BSIM3v32GdpStructPtr->CSC_Complex ;

            if ((here-> BSIM3v32gNode != 0) && (here-> BSIM3v32sNodePrime != 0))
                here->BSIM3v32GspPtr = here->BSIM3v32GspStructPtr->CSC_Complex ;

            if ((here-> BSIM3v32sNode != 0) && (here-> BSIM3v32sNodePrime != 0))
                here->BSIM3v32SspPtr = here->BSIM3v32SspStructPtr->CSC_Complex ;

            if ((here-> BSIM3v32bNode != 0) && (here-> BSIM3v32dNodePrime != 0))
                here->BSIM3v32BdpPtr = here->BSIM3v32BdpStructPtr->CSC_Complex ;

            if ((here-> BSIM3v32bNode != 0) && (here-> BSIM3v32sNodePrime != 0))
                here->BSIM3v32BspPtr = here->BSIM3v32BspStructPtr->CSC_Complex ;

            if ((here-> BSIM3v32dNodePrime != 0) && (here-> BSIM3v32sNodePrime != 0))
                here->BSIM3v32DPspPtr = here->BSIM3v32DPspStructPtr->CSC_Complex ;

            if ((here-> BSIM3v32dNodePrime != 0) && (here-> BSIM3v32dNode != 0))
                here->BSIM3v32DPdPtr = here->BSIM3v32DPdStructPtr->CSC_Complex ;

            if ((here-> BSIM3v32bNode != 0) && (here-> BSIM3v32gNode != 0))
                here->BSIM3v32BgPtr = here->BSIM3v32BgStructPtr->CSC_Complex ;

            if ((here-> BSIM3v32dNodePrime != 0) && (here-> BSIM3v32gNode != 0))
                here->BSIM3v32DPgPtr = here->BSIM3v32DPgStructPtr->CSC_Complex ;

            if ((here-> BSIM3v32sNodePrime != 0) && (here-> BSIM3v32gNode != 0))
                here->BSIM3v32SPgPtr = here->BSIM3v32SPgStructPtr->CSC_Complex ;

            if ((here-> BSIM3v32sNodePrime != 0) && (here-> BSIM3v32sNode != 0))
                here->BSIM3v32SPsPtr = here->BSIM3v32SPsStructPtr->CSC_Complex ;

            if ((here-> BSIM3v32dNodePrime != 0) && (here-> BSIM3v32bNode != 0))
                here->BSIM3v32DPbPtr = here->BSIM3v32DPbStructPtr->CSC_Complex ;

            if ((here-> BSIM3v32sNodePrime != 0) && (here-> BSIM3v32bNode != 0))
                here->BSIM3v32SPbPtr = here->BSIM3v32SPbStructPtr->CSC_Complex ;

            if ((here-> BSIM3v32sNodePrime != 0) && (here-> BSIM3v32dNodePrime != 0))
                here->BSIM3v32SPdpPtr = here->BSIM3v32SPdpStructPtr->CSC_Complex ;

            if ((here-> BSIM3v32qNode != 0) && (here-> BSIM3v32qNode != 0))
                here->BSIM3v32QqPtr = here->BSIM3v32QqStructPtr->CSC_Complex ;

            if ((here-> BSIM3v32qNode != 0) && (here-> BSIM3v32dNodePrime != 0))
                here->BSIM3v32QdpPtr = here->BSIM3v32QdpStructPtr->CSC_Complex ;

            if ((here-> BSIM3v32qNode != 0) && (here-> BSIM3v32sNodePrime != 0))
                here->BSIM3v32QspPtr = here->BSIM3v32QspStructPtr->CSC_Complex ;

            if ((here-> BSIM3v32qNode != 0) && (here-> BSIM3v32gNode != 0))
                here->BSIM3v32QgPtr = here->BSIM3v32QgStructPtr->CSC_Complex ;

            if ((here-> BSIM3v32qNode != 0) && (here-> BSIM3v32bNode != 0))
                here->BSIM3v32QbPtr = here->BSIM3v32QbStructPtr->CSC_Complex ;

            if ((here-> BSIM3v32dNodePrime != 0) && (here-> BSIM3v32qNode != 0))
                here->BSIM3v32DPqPtr = here->BSIM3v32DPqStructPtr->CSC_Complex ;

            if ((here-> BSIM3v32sNodePrime != 0) && (here-> BSIM3v32qNode != 0))
                here->BSIM3v32SPqPtr = here->BSIM3v32SPqStructPtr->CSC_Complex ;

            if ((here-> BSIM3v32gNode != 0) && (here-> BSIM3v32qNode != 0))
                here->BSIM3v32GqPtr = here->BSIM3v32GqStructPtr->CSC_Complex ;

            if ((here-> BSIM3v32bNode != 0) && (here-> BSIM3v32qNode != 0))
                here->BSIM3v32BqPtr = here->BSIM3v32BqStructPtr->CSC_Complex ;

        }
    }

    return (OK) ;
}

int
BSIM3v32bindCSCComplexToReal (GENmodel *inModel, CKTcircuit *ckt)
{
    BSIM3v32model *model = (BSIM3v32model *)inModel ;
    BSIM3v32instance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the BSIM3v32 models */
    for ( ; model != NULL ; model = model->BSIM3v32nextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->BSIM3v32instances ; here != NULL ; here = here->BSIM3v32nextInstance)
        {
            if ((here-> BSIM3v32dNode != 0) && (here-> BSIM3v32dNode != 0))
                here->BSIM3v32DdPtr = here->BSIM3v32DdStructPtr->CSC ;

            if ((here-> BSIM3v32gNode != 0) && (here-> BSIM3v32gNode != 0))
                here->BSIM3v32GgPtr = here->BSIM3v32GgStructPtr->CSC ;

            if ((here-> BSIM3v32sNode != 0) && (here-> BSIM3v32sNode != 0))
                here->BSIM3v32SsPtr = here->BSIM3v32SsStructPtr->CSC ;

            if ((here-> BSIM3v32bNode != 0) && (here-> BSIM3v32bNode != 0))
                here->BSIM3v32BbPtr = here->BSIM3v32BbStructPtr->CSC ;

            if ((here-> BSIM3v32dNodePrime != 0) && (here-> BSIM3v32dNodePrime != 0))
                here->BSIM3v32DPdpPtr = here->BSIM3v32DPdpStructPtr->CSC ;

            if ((here-> BSIM3v32sNodePrime != 0) && (here-> BSIM3v32sNodePrime != 0))
                here->BSIM3v32SPspPtr = here->BSIM3v32SPspStructPtr->CSC ;

            if ((here-> BSIM3v32dNode != 0) && (here-> BSIM3v32dNodePrime != 0))
                here->BSIM3v32DdpPtr = here->BSIM3v32DdpStructPtr->CSC ;

            if ((here-> BSIM3v32gNode != 0) && (here-> BSIM3v32bNode != 0))
                here->BSIM3v32GbPtr = here->BSIM3v32GbStructPtr->CSC ;

            if ((here-> BSIM3v32gNode != 0) && (here-> BSIM3v32dNodePrime != 0))
                here->BSIM3v32GdpPtr = here->BSIM3v32GdpStructPtr->CSC ;

            if ((here-> BSIM3v32gNode != 0) && (here-> BSIM3v32sNodePrime != 0))
                here->BSIM3v32GspPtr = here->BSIM3v32GspStructPtr->CSC ;

            if ((here-> BSIM3v32sNode != 0) && (here-> BSIM3v32sNodePrime != 0))
                here->BSIM3v32SspPtr = here->BSIM3v32SspStructPtr->CSC ;

            if ((here-> BSIM3v32bNode != 0) && (here-> BSIM3v32dNodePrime != 0))
                here->BSIM3v32BdpPtr = here->BSIM3v32BdpStructPtr->CSC ;

            if ((here-> BSIM3v32bNode != 0) && (here-> BSIM3v32sNodePrime != 0))
                here->BSIM3v32BspPtr = here->BSIM3v32BspStructPtr->CSC ;

            if ((here-> BSIM3v32dNodePrime != 0) && (here-> BSIM3v32sNodePrime != 0))
                here->BSIM3v32DPspPtr = here->BSIM3v32DPspStructPtr->CSC ;

            if ((here-> BSIM3v32dNodePrime != 0) && (here-> BSIM3v32dNode != 0))
                here->BSIM3v32DPdPtr = here->BSIM3v32DPdStructPtr->CSC ;

            if ((here-> BSIM3v32bNode != 0) && (here-> BSIM3v32gNode != 0))
                here->BSIM3v32BgPtr = here->BSIM3v32BgStructPtr->CSC ;

            if ((here-> BSIM3v32dNodePrime != 0) && (here-> BSIM3v32gNode != 0))
                here->BSIM3v32DPgPtr = here->BSIM3v32DPgStructPtr->CSC ;

            if ((here-> BSIM3v32sNodePrime != 0) && (here-> BSIM3v32gNode != 0))
                here->BSIM3v32SPgPtr = here->BSIM3v32SPgStructPtr->CSC ;

            if ((here-> BSIM3v32sNodePrime != 0) && (here-> BSIM3v32sNode != 0))
                here->BSIM3v32SPsPtr = here->BSIM3v32SPsStructPtr->CSC ;

            if ((here-> BSIM3v32dNodePrime != 0) && (here-> BSIM3v32bNode != 0))
                here->BSIM3v32DPbPtr = here->BSIM3v32DPbStructPtr->CSC ;

            if ((here-> BSIM3v32sNodePrime != 0) && (here-> BSIM3v32bNode != 0))
                here->BSIM3v32SPbPtr = here->BSIM3v32SPbStructPtr->CSC ;

            if ((here-> BSIM3v32sNodePrime != 0) && (here-> BSIM3v32dNodePrime != 0))
                here->BSIM3v32SPdpPtr = here->BSIM3v32SPdpStructPtr->CSC ;

            if ((here-> BSIM3v32qNode != 0) && (here-> BSIM3v32qNode != 0))
                here->BSIM3v32QqPtr = here->BSIM3v32QqStructPtr->CSC ;

            if ((here-> BSIM3v32qNode != 0) && (here-> BSIM3v32dNodePrime != 0))
                here->BSIM3v32QdpPtr = here->BSIM3v32QdpStructPtr->CSC ;

            if ((here-> BSIM3v32qNode != 0) && (here-> BSIM3v32sNodePrime != 0))
                here->BSIM3v32QspPtr = here->BSIM3v32QspStructPtr->CSC ;

            if ((here-> BSIM3v32qNode != 0) && (here-> BSIM3v32gNode != 0))
                here->BSIM3v32QgPtr = here->BSIM3v32QgStructPtr->CSC ;

            if ((here-> BSIM3v32qNode != 0) && (here-> BSIM3v32bNode != 0))
                here->BSIM3v32QbPtr = here->BSIM3v32QbStructPtr->CSC ;

            if ((here-> BSIM3v32dNodePrime != 0) && (here-> BSIM3v32qNode != 0))
                here->BSIM3v32DPqPtr = here->BSIM3v32DPqStructPtr->CSC ;

            if ((here-> BSIM3v32sNodePrime != 0) && (here-> BSIM3v32qNode != 0))
                here->BSIM3v32SPqPtr = here->BSIM3v32SPqStructPtr->CSC ;

            if ((here-> BSIM3v32gNode != 0) && (here-> BSIM3v32qNode != 0))
                here->BSIM3v32GqPtr = here->BSIM3v32GqStructPtr->CSC ;

            if ((here-> BSIM3v32bNode != 0) && (here-> BSIM3v32qNode != 0))
                here->BSIM3v32BqPtr = here->BSIM3v32BqStructPtr->CSC ;

        }
    }

    return (OK) ;
}
