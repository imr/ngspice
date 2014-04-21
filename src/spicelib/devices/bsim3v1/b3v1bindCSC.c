/**********
Author: 2013 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "bsim3v1def.h"
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
BSIM3v1bindCSC (GENmodel *inModel, CKTcircuit *ckt)
{
    BSIM3v1model *model = (BSIM3v1model *)inModel ;
    BSIM3v1instance *here ;
    double *i ;
    BindElement *matched, *BindStruct ;
    size_t nz ;

    BindStruct = ckt->CKTmatrix->CKTbindStruct ;
    nz = (size_t)ckt->CKTmatrix->CKTklunz ;

    /* loop through all the BSIM3v1 models */
    for ( ; model != NULL ; model = model->BSIM3v1nextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->BSIM3v1instances ; here != NULL ; here = here->BSIM3v1nextInstance)
        {
            if ((here-> BSIM3v1dNode != 0) && (here-> BSIM3v1dNode != 0))
            {
                i = here->BSIM3v1DdPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM3v1DdStructPtr = matched ;
                here->BSIM3v1DdPtr = matched->CSC ;
            }

            if ((here-> BSIM3v1gNode != 0) && (here-> BSIM3v1gNode != 0))
            {
                i = here->BSIM3v1GgPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM3v1GgStructPtr = matched ;
                here->BSIM3v1GgPtr = matched->CSC ;
            }

            if ((here-> BSIM3v1sNode != 0) && (here-> BSIM3v1sNode != 0))
            {
                i = here->BSIM3v1SsPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM3v1SsStructPtr = matched ;
                here->BSIM3v1SsPtr = matched->CSC ;
            }

            if ((here-> BSIM3v1bNode != 0) && (here-> BSIM3v1bNode != 0))
            {
                i = here->BSIM3v1BbPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM3v1BbStructPtr = matched ;
                here->BSIM3v1BbPtr = matched->CSC ;
            }

            if ((here-> BSIM3v1dNodePrime != 0) && (here-> BSIM3v1dNodePrime != 0))
            {
                i = here->BSIM3v1DPdpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM3v1DPdpStructPtr = matched ;
                here->BSIM3v1DPdpPtr = matched->CSC ;
            }

            if ((here-> BSIM3v1sNodePrime != 0) && (here-> BSIM3v1sNodePrime != 0))
            {
                i = here->BSIM3v1SPspPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM3v1SPspStructPtr = matched ;
                here->BSIM3v1SPspPtr = matched->CSC ;
            }

            if ((here-> BSIM3v1dNode != 0) && (here-> BSIM3v1dNodePrime != 0))
            {
                i = here->BSIM3v1DdpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM3v1DdpStructPtr = matched ;
                here->BSIM3v1DdpPtr = matched->CSC ;
            }

            if ((here-> BSIM3v1gNode != 0) && (here-> BSIM3v1bNode != 0))
            {
                i = here->BSIM3v1GbPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM3v1GbStructPtr = matched ;
                here->BSIM3v1GbPtr = matched->CSC ;
            }

            if ((here-> BSIM3v1gNode != 0) && (here-> BSIM3v1dNodePrime != 0))
            {
                i = here->BSIM3v1GdpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM3v1GdpStructPtr = matched ;
                here->BSIM3v1GdpPtr = matched->CSC ;
            }

            if ((here-> BSIM3v1gNode != 0) && (here-> BSIM3v1sNodePrime != 0))
            {
                i = here->BSIM3v1GspPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM3v1GspStructPtr = matched ;
                here->BSIM3v1GspPtr = matched->CSC ;
            }

            if ((here-> BSIM3v1sNode != 0) && (here-> BSIM3v1sNodePrime != 0))
            {
                i = here->BSIM3v1SspPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM3v1SspStructPtr = matched ;
                here->BSIM3v1SspPtr = matched->CSC ;
            }

            if ((here-> BSIM3v1bNode != 0) && (here-> BSIM3v1dNodePrime != 0))
            {
                i = here->BSIM3v1BdpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM3v1BdpStructPtr = matched ;
                here->BSIM3v1BdpPtr = matched->CSC ;
            }

            if ((here-> BSIM3v1bNode != 0) && (here-> BSIM3v1sNodePrime != 0))
            {
                i = here->BSIM3v1BspPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM3v1BspStructPtr = matched ;
                here->BSIM3v1BspPtr = matched->CSC ;
            }

            if ((here-> BSIM3v1dNodePrime != 0) && (here-> BSIM3v1sNodePrime != 0))
            {
                i = here->BSIM3v1DPspPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM3v1DPspStructPtr = matched ;
                here->BSIM3v1DPspPtr = matched->CSC ;
            }

            if ((here-> BSIM3v1dNodePrime != 0) && (here-> BSIM3v1dNode != 0))
            {
                i = here->BSIM3v1DPdPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM3v1DPdStructPtr = matched ;
                here->BSIM3v1DPdPtr = matched->CSC ;
            }

            if ((here-> BSIM3v1bNode != 0) && (here-> BSIM3v1gNode != 0))
            {
                i = here->BSIM3v1BgPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM3v1BgStructPtr = matched ;
                here->BSIM3v1BgPtr = matched->CSC ;
            }

            if ((here-> BSIM3v1dNodePrime != 0) && (here-> BSIM3v1gNode != 0))
            {
                i = here->BSIM3v1DPgPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM3v1DPgStructPtr = matched ;
                here->BSIM3v1DPgPtr = matched->CSC ;
            }

            if ((here-> BSIM3v1sNodePrime != 0) && (here-> BSIM3v1gNode != 0))
            {
                i = here->BSIM3v1SPgPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM3v1SPgStructPtr = matched ;
                here->BSIM3v1SPgPtr = matched->CSC ;
            }

            if ((here-> BSIM3v1sNodePrime != 0) && (here-> BSIM3v1sNode != 0))
            {
                i = here->BSIM3v1SPsPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM3v1SPsStructPtr = matched ;
                here->BSIM3v1SPsPtr = matched->CSC ;
            }

            if ((here-> BSIM3v1dNodePrime != 0) && (here-> BSIM3v1bNode != 0))
            {
                i = here->BSIM3v1DPbPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM3v1DPbStructPtr = matched ;
                here->BSIM3v1DPbPtr = matched->CSC ;
            }

            if ((here-> BSIM3v1sNodePrime != 0) && (here-> BSIM3v1bNode != 0))
            {
                i = here->BSIM3v1SPbPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM3v1SPbStructPtr = matched ;
                here->BSIM3v1SPbPtr = matched->CSC ;
            }

            if ((here-> BSIM3v1sNodePrime != 0) && (here-> BSIM3v1dNodePrime != 0))
            {
                i = here->BSIM3v1SPdpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM3v1SPdpStructPtr = matched ;
                here->BSIM3v1SPdpPtr = matched->CSC ;
            }

            if ((here-> BSIM3v1qNode != 0) && (here-> BSIM3v1qNode != 0))
            {
                i = here->BSIM3v1QqPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM3v1QqStructPtr = matched ;
                here->BSIM3v1QqPtr = matched->CSC ;
            }

            if ((here-> BSIM3v1qNode != 0) && (here-> BSIM3v1dNodePrime != 0))
            {
                i = here->BSIM3v1QdpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM3v1QdpStructPtr = matched ;
                here->BSIM3v1QdpPtr = matched->CSC ;
            }

            if ((here-> BSIM3v1qNode != 0) && (here-> BSIM3v1sNodePrime != 0))
            {
                i = here->BSIM3v1QspPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM3v1QspStructPtr = matched ;
                here->BSIM3v1QspPtr = matched->CSC ;
            }

            if ((here-> BSIM3v1qNode != 0) && (here-> BSIM3v1gNode != 0))
            {
                i = here->BSIM3v1QgPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM3v1QgStructPtr = matched ;
                here->BSIM3v1QgPtr = matched->CSC ;
            }

            if ((here-> BSIM3v1qNode != 0) && (here-> BSIM3v1bNode != 0))
            {
                i = here->BSIM3v1QbPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM3v1QbStructPtr = matched ;
                here->BSIM3v1QbPtr = matched->CSC ;
            }

            if ((here-> BSIM3v1dNodePrime != 0) && (here-> BSIM3v1qNode != 0))
            {
                i = here->BSIM3v1DPqPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM3v1DPqStructPtr = matched ;
                here->BSIM3v1DPqPtr = matched->CSC ;
            }

            if ((here-> BSIM3v1sNodePrime != 0) && (here-> BSIM3v1qNode != 0))
            {
                i = here->BSIM3v1SPqPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM3v1SPqStructPtr = matched ;
                here->BSIM3v1SPqPtr = matched->CSC ;
            }

            if ((here-> BSIM3v1gNode != 0) && (here-> BSIM3v1qNode != 0))
            {
                i = here->BSIM3v1GqPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM3v1GqStructPtr = matched ;
                here->BSIM3v1GqPtr = matched->CSC ;
            }

            if ((here-> BSIM3v1bNode != 0) && (here-> BSIM3v1qNode != 0))
            {
                i = here->BSIM3v1BqPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM3v1BqStructPtr = matched ;
                here->BSIM3v1BqPtr = matched->CSC ;
            }

        }
    }

    return (OK) ;
}

int
BSIM3v1bindCSCComplex (GENmodel *inModel, CKTcircuit *ckt)
{
    BSIM3v1model *model = (BSIM3v1model *)inModel ;
    BSIM3v1instance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the BSIM3v1 models */
    for ( ; model != NULL ; model = model->BSIM3v1nextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->BSIM3v1instances ; here != NULL ; here = here->BSIM3v1nextInstance)
        {
            if ((here-> BSIM3v1dNode != 0) && (here-> BSIM3v1dNode != 0))
                here->BSIM3v1DdPtr = here->BSIM3v1DdStructPtr->CSC_Complex ;

            if ((here-> BSIM3v1gNode != 0) && (here-> BSIM3v1gNode != 0))
                here->BSIM3v1GgPtr = here->BSIM3v1GgStructPtr->CSC_Complex ;

            if ((here-> BSIM3v1sNode != 0) && (here-> BSIM3v1sNode != 0))
                here->BSIM3v1SsPtr = here->BSIM3v1SsStructPtr->CSC_Complex ;

            if ((here-> BSIM3v1bNode != 0) && (here-> BSIM3v1bNode != 0))
                here->BSIM3v1BbPtr = here->BSIM3v1BbStructPtr->CSC_Complex ;

            if ((here-> BSIM3v1dNodePrime != 0) && (here-> BSIM3v1dNodePrime != 0))
                here->BSIM3v1DPdpPtr = here->BSIM3v1DPdpStructPtr->CSC_Complex ;

            if ((here-> BSIM3v1sNodePrime != 0) && (here-> BSIM3v1sNodePrime != 0))
                here->BSIM3v1SPspPtr = here->BSIM3v1SPspStructPtr->CSC_Complex ;

            if ((here-> BSIM3v1dNode != 0) && (here-> BSIM3v1dNodePrime != 0))
                here->BSIM3v1DdpPtr = here->BSIM3v1DdpStructPtr->CSC_Complex ;

            if ((here-> BSIM3v1gNode != 0) && (here-> BSIM3v1bNode != 0))
                here->BSIM3v1GbPtr = here->BSIM3v1GbStructPtr->CSC_Complex ;

            if ((here-> BSIM3v1gNode != 0) && (here-> BSIM3v1dNodePrime != 0))
                here->BSIM3v1GdpPtr = here->BSIM3v1GdpStructPtr->CSC_Complex ;

            if ((here-> BSIM3v1gNode != 0) && (here-> BSIM3v1sNodePrime != 0))
                here->BSIM3v1GspPtr = here->BSIM3v1GspStructPtr->CSC_Complex ;

            if ((here-> BSIM3v1sNode != 0) && (here-> BSIM3v1sNodePrime != 0))
                here->BSIM3v1SspPtr = here->BSIM3v1SspStructPtr->CSC_Complex ;

            if ((here-> BSIM3v1bNode != 0) && (here-> BSIM3v1dNodePrime != 0))
                here->BSIM3v1BdpPtr = here->BSIM3v1BdpStructPtr->CSC_Complex ;

            if ((here-> BSIM3v1bNode != 0) && (here-> BSIM3v1sNodePrime != 0))
                here->BSIM3v1BspPtr = here->BSIM3v1BspStructPtr->CSC_Complex ;

            if ((here-> BSIM3v1dNodePrime != 0) && (here-> BSIM3v1sNodePrime != 0))
                here->BSIM3v1DPspPtr = here->BSIM3v1DPspStructPtr->CSC_Complex ;

            if ((here-> BSIM3v1dNodePrime != 0) && (here-> BSIM3v1dNode != 0))
                here->BSIM3v1DPdPtr = here->BSIM3v1DPdStructPtr->CSC_Complex ;

            if ((here-> BSIM3v1bNode != 0) && (here-> BSIM3v1gNode != 0))
                here->BSIM3v1BgPtr = here->BSIM3v1BgStructPtr->CSC_Complex ;

            if ((here-> BSIM3v1dNodePrime != 0) && (here-> BSIM3v1gNode != 0))
                here->BSIM3v1DPgPtr = here->BSIM3v1DPgStructPtr->CSC_Complex ;

            if ((here-> BSIM3v1sNodePrime != 0) && (here-> BSIM3v1gNode != 0))
                here->BSIM3v1SPgPtr = here->BSIM3v1SPgStructPtr->CSC_Complex ;

            if ((here-> BSIM3v1sNodePrime != 0) && (here-> BSIM3v1sNode != 0))
                here->BSIM3v1SPsPtr = here->BSIM3v1SPsStructPtr->CSC_Complex ;

            if ((here-> BSIM3v1dNodePrime != 0) && (here-> BSIM3v1bNode != 0))
                here->BSIM3v1DPbPtr = here->BSIM3v1DPbStructPtr->CSC_Complex ;

            if ((here-> BSIM3v1sNodePrime != 0) && (here-> BSIM3v1bNode != 0))
                here->BSIM3v1SPbPtr = here->BSIM3v1SPbStructPtr->CSC_Complex ;

            if ((here-> BSIM3v1sNodePrime != 0) && (here-> BSIM3v1dNodePrime != 0))
                here->BSIM3v1SPdpPtr = here->BSIM3v1SPdpStructPtr->CSC_Complex ;

            if ((here-> BSIM3v1qNode != 0) && (here-> BSIM3v1qNode != 0))
                here->BSIM3v1QqPtr = here->BSIM3v1QqStructPtr->CSC_Complex ;

            if ((here-> BSIM3v1qNode != 0) && (here-> BSIM3v1dNodePrime != 0))
                here->BSIM3v1QdpPtr = here->BSIM3v1QdpStructPtr->CSC_Complex ;

            if ((here-> BSIM3v1qNode != 0) && (here-> BSIM3v1sNodePrime != 0))
                here->BSIM3v1QspPtr = here->BSIM3v1QspStructPtr->CSC_Complex ;

            if ((here-> BSIM3v1qNode != 0) && (here-> BSIM3v1gNode != 0))
                here->BSIM3v1QgPtr = here->BSIM3v1QgStructPtr->CSC_Complex ;

            if ((here-> BSIM3v1qNode != 0) && (here-> BSIM3v1bNode != 0))
                here->BSIM3v1QbPtr = here->BSIM3v1QbStructPtr->CSC_Complex ;

            if ((here-> BSIM3v1dNodePrime != 0) && (here-> BSIM3v1qNode != 0))
                here->BSIM3v1DPqPtr = here->BSIM3v1DPqStructPtr->CSC_Complex ;

            if ((here-> BSIM3v1sNodePrime != 0) && (here-> BSIM3v1qNode != 0))
                here->BSIM3v1SPqPtr = here->BSIM3v1SPqStructPtr->CSC_Complex ;

            if ((here-> BSIM3v1gNode != 0) && (here-> BSIM3v1qNode != 0))
                here->BSIM3v1GqPtr = here->BSIM3v1GqStructPtr->CSC_Complex ;

            if ((here-> BSIM3v1bNode != 0) && (here-> BSIM3v1qNode != 0))
                here->BSIM3v1BqPtr = here->BSIM3v1BqStructPtr->CSC_Complex ;

        }
    }

    return (OK) ;
}

int
BSIM3v1bindCSCComplexToReal (GENmodel *inModel, CKTcircuit *ckt)
{
    BSIM3v1model *model = (BSIM3v1model *)inModel ;
    BSIM3v1instance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the BSIM3v1 models */
    for ( ; model != NULL ; model = model->BSIM3v1nextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->BSIM3v1instances ; here != NULL ; here = here->BSIM3v1nextInstance)
        {
            if ((here-> BSIM3v1dNode != 0) && (here-> BSIM3v1dNode != 0))
                here->BSIM3v1DdPtr = here->BSIM3v1DdStructPtr->CSC ;

            if ((here-> BSIM3v1gNode != 0) && (here-> BSIM3v1gNode != 0))
                here->BSIM3v1GgPtr = here->BSIM3v1GgStructPtr->CSC ;

            if ((here-> BSIM3v1sNode != 0) && (here-> BSIM3v1sNode != 0))
                here->BSIM3v1SsPtr = here->BSIM3v1SsStructPtr->CSC ;

            if ((here-> BSIM3v1bNode != 0) && (here-> BSIM3v1bNode != 0))
                here->BSIM3v1BbPtr = here->BSIM3v1BbStructPtr->CSC ;

            if ((here-> BSIM3v1dNodePrime != 0) && (here-> BSIM3v1dNodePrime != 0))
                here->BSIM3v1DPdpPtr = here->BSIM3v1DPdpStructPtr->CSC ;

            if ((here-> BSIM3v1sNodePrime != 0) && (here-> BSIM3v1sNodePrime != 0))
                here->BSIM3v1SPspPtr = here->BSIM3v1SPspStructPtr->CSC ;

            if ((here-> BSIM3v1dNode != 0) && (here-> BSIM3v1dNodePrime != 0))
                here->BSIM3v1DdpPtr = here->BSIM3v1DdpStructPtr->CSC ;

            if ((here-> BSIM3v1gNode != 0) && (here-> BSIM3v1bNode != 0))
                here->BSIM3v1GbPtr = here->BSIM3v1GbStructPtr->CSC ;

            if ((here-> BSIM3v1gNode != 0) && (here-> BSIM3v1dNodePrime != 0))
                here->BSIM3v1GdpPtr = here->BSIM3v1GdpStructPtr->CSC ;

            if ((here-> BSIM3v1gNode != 0) && (here-> BSIM3v1sNodePrime != 0))
                here->BSIM3v1GspPtr = here->BSIM3v1GspStructPtr->CSC ;

            if ((here-> BSIM3v1sNode != 0) && (here-> BSIM3v1sNodePrime != 0))
                here->BSIM3v1SspPtr = here->BSIM3v1SspStructPtr->CSC ;

            if ((here-> BSIM3v1bNode != 0) && (here-> BSIM3v1dNodePrime != 0))
                here->BSIM3v1BdpPtr = here->BSIM3v1BdpStructPtr->CSC ;

            if ((here-> BSIM3v1bNode != 0) && (here-> BSIM3v1sNodePrime != 0))
                here->BSIM3v1BspPtr = here->BSIM3v1BspStructPtr->CSC ;

            if ((here-> BSIM3v1dNodePrime != 0) && (here-> BSIM3v1sNodePrime != 0))
                here->BSIM3v1DPspPtr = here->BSIM3v1DPspStructPtr->CSC ;

            if ((here-> BSIM3v1dNodePrime != 0) && (here-> BSIM3v1dNode != 0))
                here->BSIM3v1DPdPtr = here->BSIM3v1DPdStructPtr->CSC ;

            if ((here-> BSIM3v1bNode != 0) && (here-> BSIM3v1gNode != 0))
                here->BSIM3v1BgPtr = here->BSIM3v1BgStructPtr->CSC ;

            if ((here-> BSIM3v1dNodePrime != 0) && (here-> BSIM3v1gNode != 0))
                here->BSIM3v1DPgPtr = here->BSIM3v1DPgStructPtr->CSC ;

            if ((here-> BSIM3v1sNodePrime != 0) && (here-> BSIM3v1gNode != 0))
                here->BSIM3v1SPgPtr = here->BSIM3v1SPgStructPtr->CSC ;

            if ((here-> BSIM3v1sNodePrime != 0) && (here-> BSIM3v1sNode != 0))
                here->BSIM3v1SPsPtr = here->BSIM3v1SPsStructPtr->CSC ;

            if ((here-> BSIM3v1dNodePrime != 0) && (here-> BSIM3v1bNode != 0))
                here->BSIM3v1DPbPtr = here->BSIM3v1DPbStructPtr->CSC ;

            if ((here-> BSIM3v1sNodePrime != 0) && (here-> BSIM3v1bNode != 0))
                here->BSIM3v1SPbPtr = here->BSIM3v1SPbStructPtr->CSC ;

            if ((here-> BSIM3v1sNodePrime != 0) && (here-> BSIM3v1dNodePrime != 0))
                here->BSIM3v1SPdpPtr = here->BSIM3v1SPdpStructPtr->CSC ;

            if ((here-> BSIM3v1qNode != 0) && (here-> BSIM3v1qNode != 0))
                here->BSIM3v1QqPtr = here->BSIM3v1QqStructPtr->CSC ;

            if ((here-> BSIM3v1qNode != 0) && (here-> BSIM3v1dNodePrime != 0))
                here->BSIM3v1QdpPtr = here->BSIM3v1QdpStructPtr->CSC ;

            if ((here-> BSIM3v1qNode != 0) && (here-> BSIM3v1sNodePrime != 0))
                here->BSIM3v1QspPtr = here->BSIM3v1QspStructPtr->CSC ;

            if ((here-> BSIM3v1qNode != 0) && (here-> BSIM3v1gNode != 0))
                here->BSIM3v1QgPtr = here->BSIM3v1QgStructPtr->CSC ;

            if ((here-> BSIM3v1qNode != 0) && (here-> BSIM3v1bNode != 0))
                here->BSIM3v1QbPtr = here->BSIM3v1QbStructPtr->CSC ;

            if ((here-> BSIM3v1dNodePrime != 0) && (here-> BSIM3v1qNode != 0))
                here->BSIM3v1DPqPtr = here->BSIM3v1DPqStructPtr->CSC ;

            if ((here-> BSIM3v1sNodePrime != 0) && (here-> BSIM3v1qNode != 0))
                here->BSIM3v1SPqPtr = here->BSIM3v1SPqStructPtr->CSC ;

            if ((here-> BSIM3v1gNode != 0) && (here-> BSIM3v1qNode != 0))
                here->BSIM3v1GqPtr = here->BSIM3v1GqStructPtr->CSC ;

            if ((here-> BSIM3v1bNode != 0) && (here-> BSIM3v1qNode != 0))
                here->BSIM3v1BqPtr = here->BSIM3v1BqStructPtr->CSC ;

        }
    }

    return (OK) ;
}
