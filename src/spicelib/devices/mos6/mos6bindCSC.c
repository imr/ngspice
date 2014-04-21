/**********
Author: 2012 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "mos6defs.h"
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
MOS6bindCSC (GENmodel *inModel, CKTcircuit *ckt)
{
    MOS6model *model = (MOS6model *)inModel ;
    MOS6instance *here ;
    double *i ;
    BindElement *matched, *BindStruct ;
    size_t nz ;

    BindStruct = ckt->CKTmatrix->CKTbindStruct ;
    nz = (size_t)ckt->CKTmatrix->CKTklunz ;

    /* loop through all the MOS6 models */
    for ( ; model != NULL ; model = model->MOS6nextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->MOS6instances ; here != NULL ; here = here->MOS6nextInstance)
        {
            if ((here->MOS6dNode != 0) && (here->MOS6dNode != 0))
            {
                i = here->MOS6DdPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MOS6DdStructPtr = matched ;
                here->MOS6DdPtr = matched->CSC ;
            }

            if ((here->MOS6gNode != 0) && (here->MOS6gNode != 0))
            {
                i = here->MOS6GgPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MOS6GgStructPtr = matched ;
                here->MOS6GgPtr = matched->CSC ;
            }

            if ((here->MOS6sNode != 0) && (here->MOS6sNode != 0))
            {
                i = here->MOS6SsPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MOS6SsStructPtr = matched ;
                here->MOS6SsPtr = matched->CSC ;
            }

            if ((here->MOS6bNode != 0) && (here->MOS6bNode != 0))
            {
                i = here->MOS6BbPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MOS6BbStructPtr = matched ;
                here->MOS6BbPtr = matched->CSC ;
            }

            if ((here->MOS6dNodePrime != 0) && (here->MOS6dNodePrime != 0))
            {
                i = here->MOS6DPdpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MOS6DPdpStructPtr = matched ;
                here->MOS6DPdpPtr = matched->CSC ;
            }

            if ((here->MOS6sNodePrime != 0) && (here->MOS6sNodePrime != 0))
            {
                i = here->MOS6SPspPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MOS6SPspStructPtr = matched ;
                here->MOS6SPspPtr = matched->CSC ;
            }

            if ((here->MOS6dNode != 0) && (here->MOS6dNodePrime != 0))
            {
                i = here->MOS6DdpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MOS6DdpStructPtr = matched ;
                here->MOS6DdpPtr = matched->CSC ;
            }

            if ((here->MOS6gNode != 0) && (here->MOS6bNode != 0))
            {
                i = here->MOS6GbPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MOS6GbStructPtr = matched ;
                here->MOS6GbPtr = matched->CSC ;
            }

            if ((here->MOS6gNode != 0) && (here->MOS6dNodePrime != 0))
            {
                i = here->MOS6GdpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MOS6GdpStructPtr = matched ;
                here->MOS6GdpPtr = matched->CSC ;
            }

            if ((here->MOS6gNode != 0) && (here->MOS6sNodePrime != 0))
            {
                i = here->MOS6GspPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MOS6GspStructPtr = matched ;
                here->MOS6GspPtr = matched->CSC ;
            }

            if ((here->MOS6sNode != 0) && (here->MOS6sNodePrime != 0))
            {
                i = here->MOS6SspPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MOS6SspStructPtr = matched ;
                here->MOS6SspPtr = matched->CSC ;
            }

            if ((here->MOS6bNode != 0) && (here->MOS6dNodePrime != 0))
            {
                i = here->MOS6BdpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MOS6BdpStructPtr = matched ;
                here->MOS6BdpPtr = matched->CSC ;
            }

            if ((here->MOS6bNode != 0) && (here->MOS6sNodePrime != 0))
            {
                i = here->MOS6BspPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MOS6BspStructPtr = matched ;
                here->MOS6BspPtr = matched->CSC ;
            }

            if ((here->MOS6dNodePrime != 0) && (here->MOS6sNodePrime != 0))
            {
                i = here->MOS6DPspPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MOS6DPspStructPtr = matched ;
                here->MOS6DPspPtr = matched->CSC ;
            }

            if ((here->MOS6dNodePrime != 0) && (here->MOS6dNode != 0))
            {
                i = here->MOS6DPdPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MOS6DPdStructPtr = matched ;
                here->MOS6DPdPtr = matched->CSC ;
            }

            if ((here->MOS6bNode != 0) && (here->MOS6gNode != 0))
            {
                i = here->MOS6BgPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MOS6BgStructPtr = matched ;
                here->MOS6BgPtr = matched->CSC ;
            }

            if ((here->MOS6dNodePrime != 0) && (here->MOS6gNode != 0))
            {
                i = here->MOS6DPgPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MOS6DPgStructPtr = matched ;
                here->MOS6DPgPtr = matched->CSC ;
            }

            if ((here->MOS6sNodePrime != 0) && (here->MOS6gNode != 0))
            {
                i = here->MOS6SPgPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MOS6SPgStructPtr = matched ;
                here->MOS6SPgPtr = matched->CSC ;
            }

            if ((here->MOS6sNodePrime != 0) && (here->MOS6sNode != 0))
            {
                i = here->MOS6SPsPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MOS6SPsStructPtr = matched ;
                here->MOS6SPsPtr = matched->CSC ;
            }

            if ((here->MOS6dNodePrime != 0) && (here->MOS6bNode != 0))
            {
                i = here->MOS6DPbPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MOS6DPbStructPtr = matched ;
                here->MOS6DPbPtr = matched->CSC ;
            }

            if ((here->MOS6sNodePrime != 0) && (here->MOS6bNode != 0))
            {
                i = here->MOS6SPbPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MOS6SPbStructPtr = matched ;
                here->MOS6SPbPtr = matched->CSC ;
            }

            if ((here->MOS6sNodePrime != 0) && (here->MOS6dNodePrime != 0))
            {
                i = here->MOS6SPdpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MOS6SPdpStructPtr = matched ;
                here->MOS6SPdpPtr = matched->CSC ;
            }
        }
    }

    return (OK) ;
}

int
MOS6bindCSCComplex (GENmodel *inModel, CKTcircuit *ckt)
{
    MOS6model *model = (MOS6model *)inModel ;
    MOS6instance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the MOS6 models */
    for ( ; model != NULL ; model = model->MOS6nextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->MOS6instances ; here != NULL ; here = here->MOS6nextInstance)
        {
            if ((here->MOS6dNode != 0) && (here->MOS6dNode != 0))
                here->MOS6DdPtr = here->MOS6DdStructPtr->CSC_Complex ;

            if ((here->MOS6gNode != 0) && (here->MOS6gNode != 0))
                here->MOS6GgPtr = here->MOS6GgStructPtr->CSC_Complex ;

            if ((here->MOS6sNode != 0) && (here->MOS6sNode != 0))
                here->MOS6SsPtr = here->MOS6SsStructPtr->CSC_Complex ;

            if ((here->MOS6bNode != 0) && (here->MOS6bNode != 0))
                here->MOS6BbPtr = here->MOS6BbStructPtr->CSC_Complex ;

            if ((here->MOS6dNodePrime != 0) && (here->MOS6dNodePrime != 0))
                here->MOS6DPdpPtr = here->MOS6DPdpStructPtr->CSC_Complex ;

            if ((here->MOS6sNodePrime != 0) && (here->MOS6sNodePrime != 0))
                here->MOS6SPspPtr = here->MOS6SPspStructPtr->CSC_Complex ;

            if ((here->MOS6dNode != 0) && (here->MOS6dNodePrime != 0))
                here->MOS6DdpPtr = here->MOS6DdpStructPtr->CSC_Complex ;

            if ((here->MOS6gNode != 0) && (here->MOS6bNode != 0))
                here->MOS6GbPtr = here->MOS6GbStructPtr->CSC_Complex ;

            if ((here->MOS6gNode != 0) && (here->MOS6dNodePrime != 0))
                here->MOS6GdpPtr = here->MOS6GdpStructPtr->CSC_Complex ;

            if ((here->MOS6gNode != 0) && (here->MOS6sNodePrime != 0))
                here->MOS6GspPtr = here->MOS6GspStructPtr->CSC_Complex ;

            if ((here->MOS6sNode != 0) && (here->MOS6sNodePrime != 0))
                here->MOS6SspPtr = here->MOS6SspStructPtr->CSC_Complex ;

            if ((here->MOS6bNode != 0) && (here->MOS6dNodePrime != 0))
                here->MOS6BdpPtr = here->MOS6BdpStructPtr->CSC_Complex ;

            if ((here->MOS6bNode != 0) && (here->MOS6sNodePrime != 0))
                here->MOS6BspPtr = here->MOS6BspStructPtr->CSC_Complex ;

            if ((here->MOS6dNodePrime != 0) && (here->MOS6sNodePrime != 0))
                here->MOS6DPspPtr = here->MOS6DPspStructPtr->CSC_Complex ;

            if ((here->MOS6dNodePrime != 0) && (here->MOS6dNode != 0))
                here->MOS6DPdPtr = here->MOS6DPdStructPtr->CSC_Complex ;

            if ((here->MOS6bNode != 0) && (here->MOS6gNode != 0))
                here->MOS6BgPtr = here->MOS6BgStructPtr->CSC_Complex ;

            if ((here->MOS6dNodePrime != 0) && (here->MOS6gNode != 0))
                here->MOS6DPgPtr = here->MOS6DPgStructPtr->CSC_Complex ;

            if ((here->MOS6sNodePrime != 0) && (here->MOS6gNode != 0))
                here->MOS6SPgPtr = here->MOS6SPgStructPtr->CSC_Complex ;

            if ((here->MOS6sNodePrime != 0) && (here->MOS6sNode != 0))
                here->MOS6SPsPtr = here->MOS6SPsStructPtr->CSC_Complex ;

            if ((here->MOS6dNodePrime != 0) && (here->MOS6bNode != 0))
                here->MOS6DPbPtr = here->MOS6DPbStructPtr->CSC_Complex ;

            if ((here->MOS6sNodePrime != 0) && (here->MOS6bNode != 0))
                here->MOS6SPbPtr = here->MOS6SPbStructPtr->CSC_Complex ;

            if ((here->MOS6sNodePrime != 0) && (here->MOS6dNodePrime != 0))
                here->MOS6SPdpPtr = here->MOS6SPdpStructPtr->CSC_Complex ;
        }
    }

    return (OK) ;
}

int
MOS6bindCSCComplexToReal (GENmodel *inModel, CKTcircuit *ckt)
{
    MOS6model *model = (MOS6model *)inModel ;
    MOS6instance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the MOS6 models */
    for ( ; model != NULL ; model = model->MOS6nextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->MOS6instances ; here != NULL ; here = here->MOS6nextInstance)
        {
            if ((here->MOS6dNode != 0) && (here->MOS6dNode != 0))
                here->MOS6DdPtr = here->MOS6DdStructPtr->CSC ;

            if ((here->MOS6gNode != 0) && (here->MOS6gNode != 0))
                here->MOS6GgPtr = here->MOS6GgStructPtr->CSC ;

            if ((here->MOS6sNode != 0) && (here->MOS6sNode != 0))
                here->MOS6SsPtr = here->MOS6SsStructPtr->CSC ;

            if ((here->MOS6bNode != 0) && (here->MOS6bNode != 0))
                here->MOS6BbPtr = here->MOS6BbStructPtr->CSC ;

            if ((here->MOS6dNodePrime != 0) && (here->MOS6dNodePrime != 0))
                here->MOS6DPdpPtr = here->MOS6DPdpStructPtr->CSC ;

            if ((here->MOS6sNodePrime != 0) && (here->MOS6sNodePrime != 0))
                here->MOS6SPspPtr = here->MOS6SPspStructPtr->CSC ;

            if ((here->MOS6dNode != 0) && (here->MOS6dNodePrime != 0))
                here->MOS6DdpPtr = here->MOS6DdpStructPtr->CSC ;

            if ((here->MOS6gNode != 0) && (here->MOS6bNode != 0))
                here->MOS6GbPtr = here->MOS6GbStructPtr->CSC ;

            if ((here->MOS6gNode != 0) && (here->MOS6dNodePrime != 0))
                here->MOS6GdpPtr = here->MOS6GdpStructPtr->CSC ;

            if ((here->MOS6gNode != 0) && (here->MOS6sNodePrime != 0))
                here->MOS6GspPtr = here->MOS6GspStructPtr->CSC ;

            if ((here->MOS6sNode != 0) && (here->MOS6sNodePrime != 0))
                here->MOS6SspPtr = here->MOS6SspStructPtr->CSC ;

            if ((here->MOS6bNode != 0) && (here->MOS6dNodePrime != 0))
                here->MOS6BdpPtr = here->MOS6BdpStructPtr->CSC ;

            if ((here->MOS6bNode != 0) && (here->MOS6sNodePrime != 0))
                here->MOS6BspPtr = here->MOS6BspStructPtr->CSC ;

            if ((here->MOS6dNodePrime != 0) && (here->MOS6sNodePrime != 0))
                here->MOS6DPspPtr = here->MOS6DPspStructPtr->CSC ;

            if ((here->MOS6dNodePrime != 0) && (here->MOS6dNode != 0))
                here->MOS6DPdPtr = here->MOS6DPdStructPtr->CSC ;

            if ((here->MOS6bNode != 0) && (here->MOS6gNode != 0))
                here->MOS6BgPtr = here->MOS6BgStructPtr->CSC ;

            if ((here->MOS6dNodePrime != 0) && (here->MOS6gNode != 0))
                here->MOS6DPgPtr = here->MOS6DPgStructPtr->CSC ;

            if ((here->MOS6sNodePrime != 0) && (here->MOS6gNode != 0))
                here->MOS6SPgPtr = here->MOS6SPgStructPtr->CSC ;

            if ((here->MOS6sNodePrime != 0) && (here->MOS6sNode != 0))
                here->MOS6SPsPtr = here->MOS6SPsStructPtr->CSC ;

            if ((here->MOS6dNodePrime != 0) && (here->MOS6bNode != 0))
                here->MOS6DPbPtr = here->MOS6DPbStructPtr->CSC ;

            if ((here->MOS6sNodePrime != 0) && (here->MOS6bNode != 0))
                here->MOS6SPbPtr = here->MOS6SPbStructPtr->CSC ;

            if ((here->MOS6sNodePrime != 0) && (here->MOS6dNodePrime != 0))
                here->MOS6SPdpPtr = here->MOS6SPdpStructPtr->CSC ;
        }
    }

    return (OK) ;
}
