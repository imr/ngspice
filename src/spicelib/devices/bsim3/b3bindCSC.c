/**********
Author: 2013 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "bsim3def.h"
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
BSIM3bindCSC (GENmodel *inModel, CKTcircuit *ckt)
{
    BSIM3model *model = (BSIM3model *)inModel ;
    BSIM3instance *here ;
    double *i ;
    BindElement *matched, *BindStruct ;
    size_t nz ;

    BindStruct = ckt->CKTmatrix->CKTbindStruct ;
    nz = (size_t)ckt->CKTmatrix->CKTklunz ;

    /* loop through all the BSIM3 models */
    for ( ; model != NULL ; model = model->BSIM3nextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->BSIM3instances ; here != NULL ; here = here->BSIM3nextInstance)
        {
            if ((here-> BSIM3dNode != 0) && (here-> BSIM3dNode != 0))
            {
                i = here->BSIM3DdPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM3DdStructPtr = matched ;
                here->BSIM3DdPtr = matched->CSC ;
            }

            if ((here-> BSIM3gNode != 0) && (here-> BSIM3gNode != 0))
            {
                i = here->BSIM3GgPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM3GgStructPtr = matched ;
                here->BSIM3GgPtr = matched->CSC ;
            }

            if ((here-> BSIM3sNode != 0) && (here-> BSIM3sNode != 0))
            {
                i = here->BSIM3SsPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM3SsStructPtr = matched ;
                here->BSIM3SsPtr = matched->CSC ;
            }

            if ((here-> BSIM3bNode != 0) && (here-> BSIM3bNode != 0))
            {
                i = here->BSIM3BbPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM3BbStructPtr = matched ;
                here->BSIM3BbPtr = matched->CSC ;
            }

            if ((here-> BSIM3dNodePrime != 0) && (here-> BSIM3dNodePrime != 0))
            {
                i = here->BSIM3DPdpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM3DPdpStructPtr = matched ;
                here->BSIM3DPdpPtr = matched->CSC ;
            }

            if ((here-> BSIM3sNodePrime != 0) && (here-> BSIM3sNodePrime != 0))
            {
                i = here->BSIM3SPspPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM3SPspStructPtr = matched ;
                here->BSIM3SPspPtr = matched->CSC ;
            }

            if ((here-> BSIM3dNode != 0) && (here-> BSIM3dNodePrime != 0))
            {
                i = here->BSIM3DdpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM3DdpStructPtr = matched ;
                here->BSIM3DdpPtr = matched->CSC ;
            }

            if ((here-> BSIM3gNode != 0) && (here-> BSIM3bNode != 0))
            {
                i = here->BSIM3GbPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM3GbStructPtr = matched ;
                here->BSIM3GbPtr = matched->CSC ;
            }

            if ((here-> BSIM3gNode != 0) && (here-> BSIM3dNodePrime != 0))
            {
                i = here->BSIM3GdpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM3GdpStructPtr = matched ;
                here->BSIM3GdpPtr = matched->CSC ;
            }

            if ((here-> BSIM3gNode != 0) && (here-> BSIM3sNodePrime != 0))
            {
                i = here->BSIM3GspPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM3GspStructPtr = matched ;
                here->BSIM3GspPtr = matched->CSC ;
            }

            if ((here-> BSIM3sNode != 0) && (here-> BSIM3sNodePrime != 0))
            {
                i = here->BSIM3SspPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM3SspStructPtr = matched ;
                here->BSIM3SspPtr = matched->CSC ;
            }

            if ((here-> BSIM3bNode != 0) && (here-> BSIM3dNodePrime != 0))
            {
                i = here->BSIM3BdpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM3BdpStructPtr = matched ;
                here->BSIM3BdpPtr = matched->CSC ;
            }

            if ((here-> BSIM3bNode != 0) && (here-> BSIM3sNodePrime != 0))
            {
                i = here->BSIM3BspPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM3BspStructPtr = matched ;
                here->BSIM3BspPtr = matched->CSC ;
            }

            if ((here-> BSIM3dNodePrime != 0) && (here-> BSIM3sNodePrime != 0))
            {
                i = here->BSIM3DPspPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM3DPspStructPtr = matched ;
                here->BSIM3DPspPtr = matched->CSC ;
            }

            if ((here-> BSIM3dNodePrime != 0) && (here-> BSIM3dNode != 0))
            {
                i = here->BSIM3DPdPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM3DPdStructPtr = matched ;
                here->BSIM3DPdPtr = matched->CSC ;
            }

            if ((here-> BSIM3bNode != 0) && (here-> BSIM3gNode != 0))
            {
                i = here->BSIM3BgPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM3BgStructPtr = matched ;
                here->BSIM3BgPtr = matched->CSC ;
            }

            if ((here-> BSIM3dNodePrime != 0) && (here-> BSIM3gNode != 0))
            {
                i = here->BSIM3DPgPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM3DPgStructPtr = matched ;
                here->BSIM3DPgPtr = matched->CSC ;
            }

            if ((here-> BSIM3sNodePrime != 0) && (here-> BSIM3gNode != 0))
            {
                i = here->BSIM3SPgPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM3SPgStructPtr = matched ;
                here->BSIM3SPgPtr = matched->CSC ;
            }

            if ((here-> BSIM3sNodePrime != 0) && (here-> BSIM3sNode != 0))
            {
                i = here->BSIM3SPsPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM3SPsStructPtr = matched ;
                here->BSIM3SPsPtr = matched->CSC ;
            }

            if ((here-> BSIM3dNodePrime != 0) && (here-> BSIM3bNode != 0))
            {
                i = here->BSIM3DPbPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM3DPbStructPtr = matched ;
                here->BSIM3DPbPtr = matched->CSC ;
            }

            if ((here-> BSIM3sNodePrime != 0) && (here-> BSIM3bNode != 0))
            {
                i = here->BSIM3SPbPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM3SPbStructPtr = matched ;
                here->BSIM3SPbPtr = matched->CSC ;
            }

            if ((here-> BSIM3sNodePrime != 0) && (here-> BSIM3dNodePrime != 0))
            {
                i = here->BSIM3SPdpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM3SPdpStructPtr = matched ;
                here->BSIM3SPdpPtr = matched->CSC ;
            }

            if ((here-> BSIM3qNode != 0) && (here-> BSIM3qNode != 0))
            {
                i = here->BSIM3QqPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM3QqStructPtr = matched ;
                here->BSIM3QqPtr = matched->CSC ;
            }

            if ((here-> BSIM3qNode != 0) && (here-> BSIM3dNodePrime != 0))
            {
                i = here->BSIM3QdpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM3QdpStructPtr = matched ;
                here->BSIM3QdpPtr = matched->CSC ;
            }

            if ((here-> BSIM3qNode != 0) && (here-> BSIM3gNode != 0))
            {
                i = here->BSIM3QgPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM3QgStructPtr = matched ;
                here->BSIM3QgPtr = matched->CSC ;
            }

            if ((here-> BSIM3qNode != 0) && (here-> BSIM3sNodePrime != 0))
            {
                i = here->BSIM3QspPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM3QspStructPtr = matched ;
                here->BSIM3QspPtr = matched->CSC ;
            }

            if ((here-> BSIM3qNode != 0) && (here-> BSIM3bNode != 0))
            {
                i = here->BSIM3QbPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM3QbStructPtr = matched ;
                here->BSIM3QbPtr = matched->CSC ;
            }

            if ((here-> BSIM3dNodePrime != 0) && (here-> BSIM3qNode != 0))
            {
                i = here->BSIM3DPqPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM3DPqStructPtr = matched ;
                here->BSIM3DPqPtr = matched->CSC ;
            }

            if ((here-> BSIM3gNode != 0) && (here-> BSIM3qNode != 0))
            {
                i = here->BSIM3GqPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM3GqStructPtr = matched ;
                here->BSIM3GqPtr = matched->CSC ;
            }

            if ((here-> BSIM3sNodePrime != 0) && (here-> BSIM3qNode != 0))
            {
                i = here->BSIM3SPqPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM3SPqStructPtr = matched ;
                here->BSIM3SPqPtr = matched->CSC ;
            }

            if ((here-> BSIM3bNode != 0) && (here-> BSIM3qNode != 0))
            {
                i = here->BSIM3BqPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM3BqStructPtr = matched ;
                here->BSIM3BqPtr = matched->CSC ;
            }

        }
    }

    return (OK) ;
}

int
BSIM3bindCSCComplex (GENmodel *inModel, CKTcircuit *ckt)
{
    BSIM3model *model = (BSIM3model *)inModel ;
    BSIM3instance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the BSIM3 models */
    for ( ; model != NULL ; model = model->BSIM3nextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->BSIM3instances ; here != NULL ; here = here->BSIM3nextInstance)
        {
            if ((here-> BSIM3dNode != 0) && (here-> BSIM3dNode != 0))
                here->BSIM3DdPtr = here->BSIM3DdStructPtr->CSC_Complex ;

            if ((here-> BSIM3gNode != 0) && (here-> BSIM3gNode != 0))
                here->BSIM3GgPtr = here->BSIM3GgStructPtr->CSC_Complex ;

            if ((here-> BSIM3sNode != 0) && (here-> BSIM3sNode != 0))
                here->BSIM3SsPtr = here->BSIM3SsStructPtr->CSC_Complex ;

            if ((here-> BSIM3bNode != 0) && (here-> BSIM3bNode != 0))
                here->BSIM3BbPtr = here->BSIM3BbStructPtr->CSC_Complex ;

            if ((here-> BSIM3dNodePrime != 0) && (here-> BSIM3dNodePrime != 0))
                here->BSIM3DPdpPtr = here->BSIM3DPdpStructPtr->CSC_Complex ;

            if ((here-> BSIM3sNodePrime != 0) && (here-> BSIM3sNodePrime != 0))
                here->BSIM3SPspPtr = here->BSIM3SPspStructPtr->CSC_Complex ;

            if ((here-> BSIM3dNode != 0) && (here-> BSIM3dNodePrime != 0))
                here->BSIM3DdpPtr = here->BSIM3DdpStructPtr->CSC_Complex ;

            if ((here-> BSIM3gNode != 0) && (here-> BSIM3bNode != 0))
                here->BSIM3GbPtr = here->BSIM3GbStructPtr->CSC_Complex ;

            if ((here-> BSIM3gNode != 0) && (here-> BSIM3dNodePrime != 0))
                here->BSIM3GdpPtr = here->BSIM3GdpStructPtr->CSC_Complex ;

            if ((here-> BSIM3gNode != 0) && (here-> BSIM3sNodePrime != 0))
                here->BSIM3GspPtr = here->BSIM3GspStructPtr->CSC_Complex ;

            if ((here-> BSIM3sNode != 0) && (here-> BSIM3sNodePrime != 0))
                here->BSIM3SspPtr = here->BSIM3SspStructPtr->CSC_Complex ;

            if ((here-> BSIM3bNode != 0) && (here-> BSIM3dNodePrime != 0))
                here->BSIM3BdpPtr = here->BSIM3BdpStructPtr->CSC_Complex ;

            if ((here-> BSIM3bNode != 0) && (here-> BSIM3sNodePrime != 0))
                here->BSIM3BspPtr = here->BSIM3BspStructPtr->CSC_Complex ;

            if ((here-> BSIM3dNodePrime != 0) && (here-> BSIM3sNodePrime != 0))
                here->BSIM3DPspPtr = here->BSIM3DPspStructPtr->CSC_Complex ;

            if ((here-> BSIM3dNodePrime != 0) && (here-> BSIM3dNode != 0))
                here->BSIM3DPdPtr = here->BSIM3DPdStructPtr->CSC_Complex ;

            if ((here-> BSIM3bNode != 0) && (here-> BSIM3gNode != 0))
                here->BSIM3BgPtr = here->BSIM3BgStructPtr->CSC_Complex ;

            if ((here-> BSIM3dNodePrime != 0) && (here-> BSIM3gNode != 0))
                here->BSIM3DPgPtr = here->BSIM3DPgStructPtr->CSC_Complex ;

            if ((here-> BSIM3sNodePrime != 0) && (here-> BSIM3gNode != 0))
                here->BSIM3SPgPtr = here->BSIM3SPgStructPtr->CSC_Complex ;

            if ((here-> BSIM3sNodePrime != 0) && (here-> BSIM3sNode != 0))
                here->BSIM3SPsPtr = here->BSIM3SPsStructPtr->CSC_Complex ;

            if ((here-> BSIM3dNodePrime != 0) && (here-> BSIM3bNode != 0))
                here->BSIM3DPbPtr = here->BSIM3DPbStructPtr->CSC_Complex ;

            if ((here-> BSIM3sNodePrime != 0) && (here-> BSIM3bNode != 0))
                here->BSIM3SPbPtr = here->BSIM3SPbStructPtr->CSC_Complex ;

            if ((here-> BSIM3sNodePrime != 0) && (here-> BSIM3dNodePrime != 0))
                here->BSIM3SPdpPtr = here->BSIM3SPdpStructPtr->CSC_Complex ;

            if ((here-> BSIM3qNode != 0) && (here-> BSIM3qNode != 0))
                here->BSIM3QqPtr = here->BSIM3QqStructPtr->CSC_Complex ;

            if ((here-> BSIM3qNode != 0) && (here-> BSIM3dNodePrime != 0))
                here->BSIM3QdpPtr = here->BSIM3QdpStructPtr->CSC_Complex ;

            if ((here-> BSIM3qNode != 0) && (here-> BSIM3gNode != 0))
                here->BSIM3QgPtr = here->BSIM3QgStructPtr->CSC_Complex ;

            if ((here-> BSIM3qNode != 0) && (here-> BSIM3sNodePrime != 0))
                here->BSIM3QspPtr = here->BSIM3QspStructPtr->CSC_Complex ;

            if ((here-> BSIM3qNode != 0) && (here-> BSIM3bNode != 0))
                here->BSIM3QbPtr = here->BSIM3QbStructPtr->CSC_Complex ;

            if ((here-> BSIM3dNodePrime != 0) && (here-> BSIM3qNode != 0))
                here->BSIM3DPqPtr = here->BSIM3DPqStructPtr->CSC_Complex ;

            if ((here-> BSIM3gNode != 0) && (here-> BSIM3qNode != 0))
                here->BSIM3GqPtr = here->BSIM3GqStructPtr->CSC_Complex ;

            if ((here-> BSIM3sNodePrime != 0) && (here-> BSIM3qNode != 0))
                here->BSIM3SPqPtr = here->BSIM3SPqStructPtr->CSC_Complex ;

            if ((here-> BSIM3bNode != 0) && (here-> BSIM3qNode != 0))
                here->BSIM3BqPtr = here->BSIM3BqStructPtr->CSC_Complex ;

        }
    }

    return (OK) ;
}

int
BSIM3bindCSCComplexToReal (GENmodel *inModel, CKTcircuit *ckt)
{
    BSIM3model *model = (BSIM3model *)inModel ;
    BSIM3instance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the BSIM3 models */
    for ( ; model != NULL ; model = model->BSIM3nextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->BSIM3instances ; here != NULL ; here = here->BSIM3nextInstance)
        {
            if ((here-> BSIM3dNode != 0) && (here-> BSIM3dNode != 0))
                here->BSIM3DdPtr = here->BSIM3DdStructPtr->CSC ;

            if ((here-> BSIM3gNode != 0) && (here-> BSIM3gNode != 0))
                here->BSIM3GgPtr = here->BSIM3GgStructPtr->CSC ;

            if ((here-> BSIM3sNode != 0) && (here-> BSIM3sNode != 0))
                here->BSIM3SsPtr = here->BSIM3SsStructPtr->CSC ;

            if ((here-> BSIM3bNode != 0) && (here-> BSIM3bNode != 0))
                here->BSIM3BbPtr = here->BSIM3BbStructPtr->CSC ;

            if ((here-> BSIM3dNodePrime != 0) && (here-> BSIM3dNodePrime != 0))
                here->BSIM3DPdpPtr = here->BSIM3DPdpStructPtr->CSC ;

            if ((here-> BSIM3sNodePrime != 0) && (here-> BSIM3sNodePrime != 0))
                here->BSIM3SPspPtr = here->BSIM3SPspStructPtr->CSC ;

            if ((here-> BSIM3dNode != 0) && (here-> BSIM3dNodePrime != 0))
                here->BSIM3DdpPtr = here->BSIM3DdpStructPtr->CSC ;

            if ((here-> BSIM3gNode != 0) && (here-> BSIM3bNode != 0))
                here->BSIM3GbPtr = here->BSIM3GbStructPtr->CSC ;

            if ((here-> BSIM3gNode != 0) && (here-> BSIM3dNodePrime != 0))
                here->BSIM3GdpPtr = here->BSIM3GdpStructPtr->CSC ;

            if ((here-> BSIM3gNode != 0) && (here-> BSIM3sNodePrime != 0))
                here->BSIM3GspPtr = here->BSIM3GspStructPtr->CSC ;

            if ((here-> BSIM3sNode != 0) && (here-> BSIM3sNodePrime != 0))
                here->BSIM3SspPtr = here->BSIM3SspStructPtr->CSC ;

            if ((here-> BSIM3bNode != 0) && (here-> BSIM3dNodePrime != 0))
                here->BSIM3BdpPtr = here->BSIM3BdpStructPtr->CSC ;

            if ((here-> BSIM3bNode != 0) && (here-> BSIM3sNodePrime != 0))
                here->BSIM3BspPtr = here->BSIM3BspStructPtr->CSC ;

            if ((here-> BSIM3dNodePrime != 0) && (here-> BSIM3sNodePrime != 0))
                here->BSIM3DPspPtr = here->BSIM3DPspStructPtr->CSC ;

            if ((here-> BSIM3dNodePrime != 0) && (here-> BSIM3dNode != 0))
                here->BSIM3DPdPtr = here->BSIM3DPdStructPtr->CSC ;

            if ((here-> BSIM3bNode != 0) && (here-> BSIM3gNode != 0))
                here->BSIM3BgPtr = here->BSIM3BgStructPtr->CSC ;

            if ((here-> BSIM3dNodePrime != 0) && (here-> BSIM3gNode != 0))
                here->BSIM3DPgPtr = here->BSIM3DPgStructPtr->CSC ;

            if ((here-> BSIM3sNodePrime != 0) && (here-> BSIM3gNode != 0))
                here->BSIM3SPgPtr = here->BSIM3SPgStructPtr->CSC ;

            if ((here-> BSIM3sNodePrime != 0) && (here-> BSIM3sNode != 0))
                here->BSIM3SPsPtr = here->BSIM3SPsStructPtr->CSC ;

            if ((here-> BSIM3dNodePrime != 0) && (here-> BSIM3bNode != 0))
                here->BSIM3DPbPtr = here->BSIM3DPbStructPtr->CSC ;

            if ((here-> BSIM3sNodePrime != 0) && (here-> BSIM3bNode != 0))
                here->BSIM3SPbPtr = here->BSIM3SPbStructPtr->CSC ;

            if ((here-> BSIM3sNodePrime != 0) && (here-> BSIM3dNodePrime != 0))
                here->BSIM3SPdpPtr = here->BSIM3SPdpStructPtr->CSC ;

            if ((here-> BSIM3qNode != 0) && (here-> BSIM3qNode != 0))
                here->BSIM3QqPtr = here->BSIM3QqStructPtr->CSC ;

            if ((here-> BSIM3qNode != 0) && (here-> BSIM3dNodePrime != 0))
                here->BSIM3QdpPtr = here->BSIM3QdpStructPtr->CSC ;

            if ((here-> BSIM3qNode != 0) && (here-> BSIM3gNode != 0))
                here->BSIM3QgPtr = here->BSIM3QgStructPtr->CSC ;

            if ((here-> BSIM3qNode != 0) && (here-> BSIM3sNodePrime != 0))
                here->BSIM3QspPtr = here->BSIM3QspStructPtr->CSC ;

            if ((here-> BSIM3qNode != 0) && (here-> BSIM3bNode != 0))
                here->BSIM3QbPtr = here->BSIM3QbStructPtr->CSC ;

            if ((here-> BSIM3dNodePrime != 0) && (here-> BSIM3qNode != 0))
                here->BSIM3DPqPtr = here->BSIM3DPqStructPtr->CSC ;

            if ((here-> BSIM3gNode != 0) && (here-> BSIM3qNode != 0))
                here->BSIM3GqPtr = here->BSIM3GqStructPtr->CSC ;

            if ((here-> BSIM3sNodePrime != 0) && (here-> BSIM3qNode != 0))
                here->BSIM3SPqPtr = here->BSIM3SPqStructPtr->CSC ;

            if ((here-> BSIM3bNode != 0) && (here-> BSIM3qNode != 0))
                here->BSIM3BqPtr = here->BSIM3BqStructPtr->CSC ;

        }
    }

    return (OK) ;
}
