/**********
Author: 2013 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "bsim3v0def.h"
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
BSIM3v0bindCSC (GENmodel *inModel, CKTcircuit *ckt)
{
    BSIM3v0model *model = (BSIM3v0model *)inModel ;
    BSIM3v0instance *here ;
    double *i ;
    BindElement *matched, *BindStruct ;
    size_t nz ;

    BindStruct = ckt->CKTmatrix->CKTbindStruct ;
    nz = (size_t)ckt->CKTmatrix->CKTklunz ;

    /* loop through all the BSIM3v0 models */
    for ( ; model != NULL ; model = model->BSIM3v0nextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->BSIM3v0instances ; here != NULL ; here = here->BSIM3v0nextInstance)
        {
            if ((here-> BSIM3v0dNode != 0) && (here-> BSIM3v0dNode != 0))
            {
                i = here->BSIM3v0DdPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM3v0DdStructPtr = matched ;
                here->BSIM3v0DdPtr = matched->CSC ;
            }

            if ((here-> BSIM3v0gNode != 0) && (here-> BSIM3v0gNode != 0))
            {
                i = here->BSIM3v0GgPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM3v0GgStructPtr = matched ;
                here->BSIM3v0GgPtr = matched->CSC ;
            }

            if ((here-> BSIM3v0sNode != 0) && (here-> BSIM3v0sNode != 0))
            {
                i = here->BSIM3v0SsPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM3v0SsStructPtr = matched ;
                here->BSIM3v0SsPtr = matched->CSC ;
            }

            if ((here-> BSIM3v0bNode != 0) && (here-> BSIM3v0bNode != 0))
            {
                i = here->BSIM3v0BbPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM3v0BbStructPtr = matched ;
                here->BSIM3v0BbPtr = matched->CSC ;
            }

            if ((here-> BSIM3v0dNodePrime != 0) && (here-> BSIM3v0dNodePrime != 0))
            {
                i = here->BSIM3v0DPdpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM3v0DPdpStructPtr = matched ;
                here->BSIM3v0DPdpPtr = matched->CSC ;
            }

            if ((here-> BSIM3v0sNodePrime != 0) && (here-> BSIM3v0sNodePrime != 0))
            {
                i = here->BSIM3v0SPspPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM3v0SPspStructPtr = matched ;
                here->BSIM3v0SPspPtr = matched->CSC ;
            }

            if ((here-> BSIM3v0dNode != 0) && (here-> BSIM3v0dNodePrime != 0))
            {
                i = here->BSIM3v0DdpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM3v0DdpStructPtr = matched ;
                here->BSIM3v0DdpPtr = matched->CSC ;
            }

            if ((here-> BSIM3v0gNode != 0) && (here-> BSIM3v0bNode != 0))
            {
                i = here->BSIM3v0GbPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM3v0GbStructPtr = matched ;
                here->BSIM3v0GbPtr = matched->CSC ;
            }

            if ((here-> BSIM3v0gNode != 0) && (here-> BSIM3v0dNodePrime != 0))
            {
                i = here->BSIM3v0GdpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM3v0GdpStructPtr = matched ;
                here->BSIM3v0GdpPtr = matched->CSC ;
            }

            if ((here-> BSIM3v0gNode != 0) && (here-> BSIM3v0sNodePrime != 0))
            {
                i = here->BSIM3v0GspPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM3v0GspStructPtr = matched ;
                here->BSIM3v0GspPtr = matched->CSC ;
            }

            if ((here-> BSIM3v0sNode != 0) && (here-> BSIM3v0sNodePrime != 0))
            {
                i = here->BSIM3v0SspPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM3v0SspStructPtr = matched ;
                here->BSIM3v0SspPtr = matched->CSC ;
            }

            if ((here-> BSIM3v0bNode != 0) && (here-> BSIM3v0dNodePrime != 0))
            {
                i = here->BSIM3v0BdpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM3v0BdpStructPtr = matched ;
                here->BSIM3v0BdpPtr = matched->CSC ;
            }

            if ((here-> BSIM3v0bNode != 0) && (here-> BSIM3v0sNodePrime != 0))
            {
                i = here->BSIM3v0BspPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM3v0BspStructPtr = matched ;
                here->BSIM3v0BspPtr = matched->CSC ;
            }

            if ((here-> BSIM3v0dNodePrime != 0) && (here-> BSIM3v0sNodePrime != 0))
            {
                i = here->BSIM3v0DPspPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM3v0DPspStructPtr = matched ;
                here->BSIM3v0DPspPtr = matched->CSC ;
            }

            if ((here-> BSIM3v0dNodePrime != 0) && (here-> BSIM3v0dNode != 0))
            {
                i = here->BSIM3v0DPdPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM3v0DPdStructPtr = matched ;
                here->BSIM3v0DPdPtr = matched->CSC ;
            }

            if ((here-> BSIM3v0bNode != 0) && (here-> BSIM3v0gNode != 0))
            {
                i = here->BSIM3v0BgPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM3v0BgStructPtr = matched ;
                here->BSIM3v0BgPtr = matched->CSC ;
            }

            if ((here-> BSIM3v0dNodePrime != 0) && (here-> BSIM3v0gNode != 0))
            {
                i = here->BSIM3v0DPgPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM3v0DPgStructPtr = matched ;
                here->BSIM3v0DPgPtr = matched->CSC ;
            }

            if ((here-> BSIM3v0sNodePrime != 0) && (here-> BSIM3v0gNode != 0))
            {
                i = here->BSIM3v0SPgPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM3v0SPgStructPtr = matched ;
                here->BSIM3v0SPgPtr = matched->CSC ;
            }

            if ((here-> BSIM3v0sNodePrime != 0) && (here-> BSIM3v0sNode != 0))
            {
                i = here->BSIM3v0SPsPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM3v0SPsStructPtr = matched ;
                here->BSIM3v0SPsPtr = matched->CSC ;
            }

            if ((here-> BSIM3v0dNodePrime != 0) && (here-> BSIM3v0bNode != 0))
            {
                i = here->BSIM3v0DPbPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM3v0DPbStructPtr = matched ;
                here->BSIM3v0DPbPtr = matched->CSC ;
            }

            if ((here-> BSIM3v0sNodePrime != 0) && (here-> BSIM3v0bNode != 0))
            {
                i = here->BSIM3v0SPbPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM3v0SPbStructPtr = matched ;
                here->BSIM3v0SPbPtr = matched->CSC ;
            }

            if ((here-> BSIM3v0sNodePrime != 0) && (here-> BSIM3v0dNodePrime != 0))
            {
                i = here->BSIM3v0SPdpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM3v0SPdpStructPtr = matched ;
                here->BSIM3v0SPdpPtr = matched->CSC ;
            }

            if ((here-> BSIM3v0qNode != 0) && (here-> BSIM3v0qNode != 0))
            {
                i = here->BSIM3v0QqPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM3v0QqStructPtr = matched ;
                here->BSIM3v0QqPtr = matched->CSC ;
            }

            if ((here-> BSIM3v0qNode != 0) && (here-> BSIM3v0dNodePrime != 0))
            {
                i = here->BSIM3v0QdpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM3v0QdpStructPtr = matched ;
                here->BSIM3v0QdpPtr = matched->CSC ;
            }

            if ((here-> BSIM3v0qNode != 0) && (here-> BSIM3v0sNodePrime != 0))
            {
                i = here->BSIM3v0QspPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM3v0QspStructPtr = matched ;
                here->BSIM3v0QspPtr = matched->CSC ;
            }

            if ((here-> BSIM3v0qNode != 0) && (here-> BSIM3v0gNode != 0))
            {
                i = here->BSIM3v0QgPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM3v0QgStructPtr = matched ;
                here->BSIM3v0QgPtr = matched->CSC ;
            }

            if ((here-> BSIM3v0qNode != 0) && (here-> BSIM3v0bNode != 0))
            {
                i = here->BSIM3v0QbPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM3v0QbStructPtr = matched ;
                here->BSIM3v0QbPtr = matched->CSC ;
            }

            if ((here-> BSIM3v0dNodePrime != 0) && (here-> BSIM3v0qNode != 0))
            {
                i = here->BSIM3v0DPqPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM3v0DPqStructPtr = matched ;
                here->BSIM3v0DPqPtr = matched->CSC ;
            }

            if ((here-> BSIM3v0sNodePrime != 0) && (here-> BSIM3v0qNode != 0))
            {
                i = here->BSIM3v0SPqPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM3v0SPqStructPtr = matched ;
                here->BSIM3v0SPqPtr = matched->CSC ;
            }

            if ((here-> BSIM3v0gNode != 0) && (here-> BSIM3v0qNode != 0))
            {
                i = here->BSIM3v0GqPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM3v0GqStructPtr = matched ;
                here->BSIM3v0GqPtr = matched->CSC ;
            }

            if ((here-> BSIM3v0bNode != 0) && (here-> BSIM3v0qNode != 0))
            {
                i = here->BSIM3v0BqPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM3v0BqStructPtr = matched ;
                here->BSIM3v0BqPtr = matched->CSC ;
            }

        }
    }

    return (OK) ;
}

int
BSIM3v0bindCSCComplex (GENmodel *inModel, CKTcircuit *ckt)
{
    BSIM3v0model *model = (BSIM3v0model *)inModel ;
    BSIM3v0instance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the BSIM3v0 models */
    for ( ; model != NULL ; model = model->BSIM3v0nextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->BSIM3v0instances ; here != NULL ; here = here->BSIM3v0nextInstance)
        {
            if ((here-> BSIM3v0dNode != 0) && (here-> BSIM3v0dNode != 0))
                here->BSIM3v0DdPtr = here->BSIM3v0DdStructPtr->CSC_Complex ;

            if ((here-> BSIM3v0gNode != 0) && (here-> BSIM3v0gNode != 0))
                here->BSIM3v0GgPtr = here->BSIM3v0GgStructPtr->CSC_Complex ;

            if ((here-> BSIM3v0sNode != 0) && (here-> BSIM3v0sNode != 0))
                here->BSIM3v0SsPtr = here->BSIM3v0SsStructPtr->CSC_Complex ;

            if ((here-> BSIM3v0bNode != 0) && (here-> BSIM3v0bNode != 0))
                here->BSIM3v0BbPtr = here->BSIM3v0BbStructPtr->CSC_Complex ;

            if ((here-> BSIM3v0dNodePrime != 0) && (here-> BSIM3v0dNodePrime != 0))
                here->BSIM3v0DPdpPtr = here->BSIM3v0DPdpStructPtr->CSC_Complex ;

            if ((here-> BSIM3v0sNodePrime != 0) && (here-> BSIM3v0sNodePrime != 0))
                here->BSIM3v0SPspPtr = here->BSIM3v0SPspStructPtr->CSC_Complex ;

            if ((here-> BSIM3v0dNode != 0) && (here-> BSIM3v0dNodePrime != 0))
                here->BSIM3v0DdpPtr = here->BSIM3v0DdpStructPtr->CSC_Complex ;

            if ((here-> BSIM3v0gNode != 0) && (here-> BSIM3v0bNode != 0))
                here->BSIM3v0GbPtr = here->BSIM3v0GbStructPtr->CSC_Complex ;

            if ((here-> BSIM3v0gNode != 0) && (here-> BSIM3v0dNodePrime != 0))
                here->BSIM3v0GdpPtr = here->BSIM3v0GdpStructPtr->CSC_Complex ;

            if ((here-> BSIM3v0gNode != 0) && (here-> BSIM3v0sNodePrime != 0))
                here->BSIM3v0GspPtr = here->BSIM3v0GspStructPtr->CSC_Complex ;

            if ((here-> BSIM3v0sNode != 0) && (here-> BSIM3v0sNodePrime != 0))
                here->BSIM3v0SspPtr = here->BSIM3v0SspStructPtr->CSC_Complex ;

            if ((here-> BSIM3v0bNode != 0) && (here-> BSIM3v0dNodePrime != 0))
                here->BSIM3v0BdpPtr = here->BSIM3v0BdpStructPtr->CSC_Complex ;

            if ((here-> BSIM3v0bNode != 0) && (here-> BSIM3v0sNodePrime != 0))
                here->BSIM3v0BspPtr = here->BSIM3v0BspStructPtr->CSC_Complex ;

            if ((here-> BSIM3v0dNodePrime != 0) && (here-> BSIM3v0sNodePrime != 0))
                here->BSIM3v0DPspPtr = here->BSIM3v0DPspStructPtr->CSC_Complex ;

            if ((here-> BSIM3v0dNodePrime != 0) && (here-> BSIM3v0dNode != 0))
                here->BSIM3v0DPdPtr = here->BSIM3v0DPdStructPtr->CSC_Complex ;

            if ((here-> BSIM3v0bNode != 0) && (here-> BSIM3v0gNode != 0))
                here->BSIM3v0BgPtr = here->BSIM3v0BgStructPtr->CSC_Complex ;

            if ((here-> BSIM3v0dNodePrime != 0) && (here-> BSIM3v0gNode != 0))
                here->BSIM3v0DPgPtr = here->BSIM3v0DPgStructPtr->CSC_Complex ;

            if ((here-> BSIM3v0sNodePrime != 0) && (here-> BSIM3v0gNode != 0))
                here->BSIM3v0SPgPtr = here->BSIM3v0SPgStructPtr->CSC_Complex ;

            if ((here-> BSIM3v0sNodePrime != 0) && (here-> BSIM3v0sNode != 0))
                here->BSIM3v0SPsPtr = here->BSIM3v0SPsStructPtr->CSC_Complex ;

            if ((here-> BSIM3v0dNodePrime != 0) && (here-> BSIM3v0bNode != 0))
                here->BSIM3v0DPbPtr = here->BSIM3v0DPbStructPtr->CSC_Complex ;

            if ((here-> BSIM3v0sNodePrime != 0) && (here-> BSIM3v0bNode != 0))
                here->BSIM3v0SPbPtr = here->BSIM3v0SPbStructPtr->CSC_Complex ;

            if ((here-> BSIM3v0sNodePrime != 0) && (here-> BSIM3v0dNodePrime != 0))
                here->BSIM3v0SPdpPtr = here->BSIM3v0SPdpStructPtr->CSC_Complex ;

            if ((here-> BSIM3v0qNode != 0) && (here-> BSIM3v0qNode != 0))
                here->BSIM3v0QqPtr = here->BSIM3v0QqStructPtr->CSC_Complex ;

            if ((here-> BSIM3v0qNode != 0) && (here-> BSIM3v0dNodePrime != 0))
                here->BSIM3v0QdpPtr = here->BSIM3v0QdpStructPtr->CSC_Complex ;

            if ((here-> BSIM3v0qNode != 0) && (here-> BSIM3v0sNodePrime != 0))
                here->BSIM3v0QspPtr = here->BSIM3v0QspStructPtr->CSC_Complex ;

            if ((here-> BSIM3v0qNode != 0) && (here-> BSIM3v0gNode != 0))
                here->BSIM3v0QgPtr = here->BSIM3v0QgStructPtr->CSC_Complex ;

            if ((here-> BSIM3v0qNode != 0) && (here-> BSIM3v0bNode != 0))
                here->BSIM3v0QbPtr = here->BSIM3v0QbStructPtr->CSC_Complex ;

            if ((here-> BSIM3v0dNodePrime != 0) && (here-> BSIM3v0qNode != 0))
                here->BSIM3v0DPqPtr = here->BSIM3v0DPqStructPtr->CSC_Complex ;

            if ((here-> BSIM3v0sNodePrime != 0) && (here-> BSIM3v0qNode != 0))
                here->BSIM3v0SPqPtr = here->BSIM3v0SPqStructPtr->CSC_Complex ;

            if ((here-> BSIM3v0gNode != 0) && (here-> BSIM3v0qNode != 0))
                here->BSIM3v0GqPtr = here->BSIM3v0GqStructPtr->CSC_Complex ;

            if ((here-> BSIM3v0bNode != 0) && (here-> BSIM3v0qNode != 0))
                here->BSIM3v0BqPtr = here->BSIM3v0BqStructPtr->CSC_Complex ;

        }
    }

    return (OK) ;
}

int
BSIM3v0bindCSCComplexToReal (GENmodel *inModel, CKTcircuit *ckt)
{
    BSIM3v0model *model = (BSIM3v0model *)inModel ;
    BSIM3v0instance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the BSIM3v0 models */
    for ( ; model != NULL ; model = model->BSIM3v0nextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->BSIM3v0instances ; here != NULL ; here = here->BSIM3v0nextInstance)
        {
            if ((here-> BSIM3v0dNode != 0) && (here-> BSIM3v0dNode != 0))
                here->BSIM3v0DdPtr = here->BSIM3v0DdStructPtr->CSC ;

            if ((here-> BSIM3v0gNode != 0) && (here-> BSIM3v0gNode != 0))
                here->BSIM3v0GgPtr = here->BSIM3v0GgStructPtr->CSC ;

            if ((here-> BSIM3v0sNode != 0) && (here-> BSIM3v0sNode != 0))
                here->BSIM3v0SsPtr = here->BSIM3v0SsStructPtr->CSC ;

            if ((here-> BSIM3v0bNode != 0) && (here-> BSIM3v0bNode != 0))
                here->BSIM3v0BbPtr = here->BSIM3v0BbStructPtr->CSC ;

            if ((here-> BSIM3v0dNodePrime != 0) && (here-> BSIM3v0dNodePrime != 0))
                here->BSIM3v0DPdpPtr = here->BSIM3v0DPdpStructPtr->CSC ;

            if ((here-> BSIM3v0sNodePrime != 0) && (here-> BSIM3v0sNodePrime != 0))
                here->BSIM3v0SPspPtr = here->BSIM3v0SPspStructPtr->CSC ;

            if ((here-> BSIM3v0dNode != 0) && (here-> BSIM3v0dNodePrime != 0))
                here->BSIM3v0DdpPtr = here->BSIM3v0DdpStructPtr->CSC ;

            if ((here-> BSIM3v0gNode != 0) && (here-> BSIM3v0bNode != 0))
                here->BSIM3v0GbPtr = here->BSIM3v0GbStructPtr->CSC ;

            if ((here-> BSIM3v0gNode != 0) && (here-> BSIM3v0dNodePrime != 0))
                here->BSIM3v0GdpPtr = here->BSIM3v0GdpStructPtr->CSC ;

            if ((here-> BSIM3v0gNode != 0) && (here-> BSIM3v0sNodePrime != 0))
                here->BSIM3v0GspPtr = here->BSIM3v0GspStructPtr->CSC ;

            if ((here-> BSIM3v0sNode != 0) && (here-> BSIM3v0sNodePrime != 0))
                here->BSIM3v0SspPtr = here->BSIM3v0SspStructPtr->CSC ;

            if ((here-> BSIM3v0bNode != 0) && (here-> BSIM3v0dNodePrime != 0))
                here->BSIM3v0BdpPtr = here->BSIM3v0BdpStructPtr->CSC ;

            if ((here-> BSIM3v0bNode != 0) && (here-> BSIM3v0sNodePrime != 0))
                here->BSIM3v0BspPtr = here->BSIM3v0BspStructPtr->CSC ;

            if ((here-> BSIM3v0dNodePrime != 0) && (here-> BSIM3v0sNodePrime != 0))
                here->BSIM3v0DPspPtr = here->BSIM3v0DPspStructPtr->CSC ;

            if ((here-> BSIM3v0dNodePrime != 0) && (here-> BSIM3v0dNode != 0))
                here->BSIM3v0DPdPtr = here->BSIM3v0DPdStructPtr->CSC ;

            if ((here-> BSIM3v0bNode != 0) && (here-> BSIM3v0gNode != 0))
                here->BSIM3v0BgPtr = here->BSIM3v0BgStructPtr->CSC ;

            if ((here-> BSIM3v0dNodePrime != 0) && (here-> BSIM3v0gNode != 0))
                here->BSIM3v0DPgPtr = here->BSIM3v0DPgStructPtr->CSC ;

            if ((here-> BSIM3v0sNodePrime != 0) && (here-> BSIM3v0gNode != 0))
                here->BSIM3v0SPgPtr = here->BSIM3v0SPgStructPtr->CSC ;

            if ((here-> BSIM3v0sNodePrime != 0) && (here-> BSIM3v0sNode != 0))
                here->BSIM3v0SPsPtr = here->BSIM3v0SPsStructPtr->CSC ;

            if ((here-> BSIM3v0dNodePrime != 0) && (here-> BSIM3v0bNode != 0))
                here->BSIM3v0DPbPtr = here->BSIM3v0DPbStructPtr->CSC ;

            if ((here-> BSIM3v0sNodePrime != 0) && (here-> BSIM3v0bNode != 0))
                here->BSIM3v0SPbPtr = here->BSIM3v0SPbStructPtr->CSC ;

            if ((here-> BSIM3v0sNodePrime != 0) && (here-> BSIM3v0dNodePrime != 0))
                here->BSIM3v0SPdpPtr = here->BSIM3v0SPdpStructPtr->CSC ;

            if ((here-> BSIM3v0qNode != 0) && (here-> BSIM3v0qNode != 0))
                here->BSIM3v0QqPtr = here->BSIM3v0QqStructPtr->CSC ;

            if ((here-> BSIM3v0qNode != 0) && (here-> BSIM3v0dNodePrime != 0))
                here->BSIM3v0QdpPtr = here->BSIM3v0QdpStructPtr->CSC ;

            if ((here-> BSIM3v0qNode != 0) && (here-> BSIM3v0sNodePrime != 0))
                here->BSIM3v0QspPtr = here->BSIM3v0QspStructPtr->CSC ;

            if ((here-> BSIM3v0qNode != 0) && (here-> BSIM3v0gNode != 0))
                here->BSIM3v0QgPtr = here->BSIM3v0QgStructPtr->CSC ;

            if ((here-> BSIM3v0qNode != 0) && (here-> BSIM3v0bNode != 0))
                here->BSIM3v0QbPtr = here->BSIM3v0QbStructPtr->CSC ;

            if ((here-> BSIM3v0dNodePrime != 0) && (here-> BSIM3v0qNode != 0))
                here->BSIM3v0DPqPtr = here->BSIM3v0DPqStructPtr->CSC ;

            if ((here-> BSIM3v0sNodePrime != 0) && (here-> BSIM3v0qNode != 0))
                here->BSIM3v0SPqPtr = here->BSIM3v0SPqStructPtr->CSC ;

            if ((here-> BSIM3v0gNode != 0) && (here-> BSIM3v0qNode != 0))
                here->BSIM3v0GqPtr = here->BSIM3v0GqStructPtr->CSC ;

            if ((here-> BSIM3v0bNode != 0) && (here-> BSIM3v0qNode != 0))
                here->BSIM3v0BqPtr = here->BSIM3v0BqStructPtr->CSC ;

        }
    }

    return (OK) ;
}
