/**********
Author: 2013 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "mos1defs.h"
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
MOS1bindCSC (GENmodel *inModel, CKTcircuit *ckt)
{
    MOS1model *model = (MOS1model *)inModel ;
    MOS1instance *here ;
    double *i ;
    BindElement *matched, *BindStruct ;
    size_t nz ;

    BindStruct = ckt->CKTmatrix->CKTbindStruct ;
    nz = (size_t)ckt->CKTmatrix->CKTklunz ;

    /* loop through all the MOS1 models */
    for ( ; model != NULL ; model = model->MOS1nextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->MOS1instances ; here != NULL ; here = here->MOS1nextInstance)
        {
            if ((here->MOS1dNode != 0) && (here->MOS1dNode != 0))
            {
                i = here->MOS1DdPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MOS1DdStructPtr = matched ;
                here->MOS1DdPtr = matched->CSC ;
            }

            if ((here->MOS1gNode != 0) && (here->MOS1gNode != 0))
            {
                i = here->MOS1GgPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MOS1GgStructPtr = matched ;
                here->MOS1GgPtr = matched->CSC ;
            }

            if ((here->MOS1sNode != 0) && (here->MOS1sNode != 0))
            {
                i = here->MOS1SsPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MOS1SsStructPtr = matched ;
                here->MOS1SsPtr = matched->CSC ;
            }

            if ((here->MOS1bNode != 0) && (here->MOS1bNode != 0))
            {
                i = here->MOS1BbPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MOS1BbStructPtr = matched ;
                here->MOS1BbPtr = matched->CSC ;
            }

            if ((here->MOS1dNodePrime != 0) && (here->MOS1dNodePrime != 0))
            {
                i = here->MOS1DPdpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MOS1DPdpStructPtr = matched ;
                here->MOS1DPdpPtr = matched->CSC ;
            }

            if ((here->MOS1sNodePrime != 0) && (here->MOS1sNodePrime != 0))
            {
                i = here->MOS1SPspPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MOS1SPspStructPtr = matched ;
                here->MOS1SPspPtr = matched->CSC ;
            }

            if ((here->MOS1dNode != 0) && (here->MOS1dNodePrime != 0))
            {
                i = here->MOS1DdpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MOS1DdpStructPtr = matched ;
                here->MOS1DdpPtr = matched->CSC ;
            }

            if ((here->MOS1gNode != 0) && (here->MOS1bNode != 0))
            {
                i = here->MOS1GbPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MOS1GbStructPtr = matched ;
                here->MOS1GbPtr = matched->CSC ;
            }

            if ((here->MOS1gNode != 0) && (here->MOS1dNodePrime != 0))
            {
                i = here->MOS1GdpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MOS1GdpStructPtr = matched ;
                here->MOS1GdpPtr = matched->CSC ;
            }

            if ((here->MOS1gNode != 0) && (here->MOS1sNodePrime != 0))
            {
                i = here->MOS1GspPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MOS1GspStructPtr = matched ;
                here->MOS1GspPtr = matched->CSC ;
            }

            if ((here->MOS1sNode != 0) && (here->MOS1sNodePrime != 0))
            {
                i = here->MOS1SspPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MOS1SspStructPtr = matched ;
                here->MOS1SspPtr = matched->CSC ;
            }

            if ((here->MOS1bNode != 0) && (here->MOS1dNodePrime != 0))
            {
                i = here->MOS1BdpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MOS1BdpStructPtr = matched ;
                here->MOS1BdpPtr = matched->CSC ;
            }

            if ((here->MOS1bNode != 0) && (here->MOS1sNodePrime != 0))
            {
                i = here->MOS1BspPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MOS1BspStructPtr = matched ;
                here->MOS1BspPtr = matched->CSC ;
            }

            if ((here->MOS1dNodePrime != 0) && (here->MOS1sNodePrime != 0))
            {
                i = here->MOS1DPspPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MOS1DPspStructPtr = matched ;
                here->MOS1DPspPtr = matched->CSC ;
            }

            if ((here->MOS1dNodePrime != 0) && (here->MOS1dNode != 0))
            {
                i = here->MOS1DPdPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MOS1DPdStructPtr = matched ;
                here->MOS1DPdPtr = matched->CSC ;
            }

            if ((here->MOS1bNode != 0) && (here->MOS1gNode != 0))
            {
                i = here->MOS1BgPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MOS1BgStructPtr = matched ;
                here->MOS1BgPtr = matched->CSC ;
            }

            if ((here->MOS1dNodePrime != 0) && (here->MOS1gNode != 0))
            {
                i = here->MOS1DPgPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MOS1DPgStructPtr = matched ;
                here->MOS1DPgPtr = matched->CSC ;
            }

            if ((here->MOS1sNodePrime != 0) && (here->MOS1gNode != 0))
            {
                i = here->MOS1SPgPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MOS1SPgStructPtr = matched ;
                here->MOS1SPgPtr = matched->CSC ;
            }

            if ((here->MOS1sNodePrime != 0) && (here->MOS1sNode != 0))
            {
                i = here->MOS1SPsPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MOS1SPsStructPtr = matched ;
                here->MOS1SPsPtr = matched->CSC ;
            }

            if ((here->MOS1dNodePrime != 0) && (here->MOS1bNode != 0))
            {
                i = here->MOS1DPbPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MOS1DPbStructPtr = matched ;
                here->MOS1DPbPtr = matched->CSC ;
            }

            if ((here->MOS1sNodePrime != 0) && (here->MOS1bNode != 0))
            {
                i = here->MOS1SPbPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MOS1SPbStructPtr = matched ;
                here->MOS1SPbPtr = matched->CSC ;
            }

            if ((here->MOS1sNodePrime != 0) && (here->MOS1dNodePrime != 0))
            {
                i = here->MOS1SPdpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MOS1SPdpStructPtr = matched ;
                here->MOS1SPdpPtr = matched->CSC ;
            }

        }
    }

    return (OK) ;
}

int
MOS1bindCSCComplex (GENmodel *inModel, CKTcircuit *ckt)
{
    MOS1model *model = (MOS1model *)inModel ;
    MOS1instance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the MOS1 models */
    for ( ; model != NULL ; model = model->MOS1nextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->MOS1instances ; here != NULL ; here = here->MOS1nextInstance)
        {
            if ((here->MOS1dNode != 0) && (here->MOS1dNode != 0))
                here->MOS1DdPtr = here->MOS1DdStructPtr->CSC_Complex ;

            if ((here->MOS1gNode != 0) && (here->MOS1gNode != 0))
                here->MOS1GgPtr = here->MOS1GgStructPtr->CSC_Complex ;

            if ((here->MOS1sNode != 0) && (here->MOS1sNode != 0))
                here->MOS1SsPtr = here->MOS1SsStructPtr->CSC_Complex ;

            if ((here->MOS1bNode != 0) && (here->MOS1bNode != 0))
                here->MOS1BbPtr = here->MOS1BbStructPtr->CSC_Complex ;

            if ((here->MOS1dNodePrime != 0) && (here->MOS1dNodePrime != 0))
                here->MOS1DPdpPtr = here->MOS1DPdpStructPtr->CSC_Complex ;

            if ((here->MOS1sNodePrime != 0) && (here->MOS1sNodePrime != 0))
                here->MOS1SPspPtr = here->MOS1SPspStructPtr->CSC_Complex ;

            if ((here->MOS1dNode != 0) && (here->MOS1dNodePrime != 0))
                here->MOS1DdpPtr = here->MOS1DdpStructPtr->CSC_Complex ;

            if ((here->MOS1gNode != 0) && (here->MOS1bNode != 0))
                here->MOS1GbPtr = here->MOS1GbStructPtr->CSC_Complex ;

            if ((here->MOS1gNode != 0) && (here->MOS1dNodePrime != 0))
                here->MOS1GdpPtr = here->MOS1GdpStructPtr->CSC_Complex ;

            if ((here->MOS1gNode != 0) && (here->MOS1sNodePrime != 0))
                here->MOS1GspPtr = here->MOS1GspStructPtr->CSC_Complex ;

            if ((here->MOS1sNode != 0) && (here->MOS1sNodePrime != 0))
                here->MOS1SspPtr = here->MOS1SspStructPtr->CSC_Complex ;

            if ((here->MOS1bNode != 0) && (here->MOS1dNodePrime != 0))
                here->MOS1BdpPtr = here->MOS1BdpStructPtr->CSC_Complex ;

            if ((here->MOS1bNode != 0) && (here->MOS1sNodePrime != 0))
                here->MOS1BspPtr = here->MOS1BspStructPtr->CSC_Complex ;

            if ((here->MOS1dNodePrime != 0) && (here->MOS1sNodePrime != 0))
                here->MOS1DPspPtr = here->MOS1DPspStructPtr->CSC_Complex ;

            if ((here->MOS1dNodePrime != 0) && (here->MOS1dNode != 0))
                here->MOS1DPdPtr = here->MOS1DPdStructPtr->CSC_Complex ;

            if ((here->MOS1bNode != 0) && (here->MOS1gNode != 0))
                here->MOS1BgPtr = here->MOS1BgStructPtr->CSC_Complex ;

            if ((here->MOS1dNodePrime != 0) && (here->MOS1gNode != 0))
                here->MOS1DPgPtr = here->MOS1DPgStructPtr->CSC_Complex ;

            if ((here->MOS1sNodePrime != 0) && (here->MOS1gNode != 0))
                here->MOS1SPgPtr = here->MOS1SPgStructPtr->CSC_Complex ;

            if ((here->MOS1sNodePrime != 0) && (here->MOS1sNode != 0))
                here->MOS1SPsPtr = here->MOS1SPsStructPtr->CSC_Complex ;

            if ((here->MOS1dNodePrime != 0) && (here->MOS1bNode != 0))
                here->MOS1DPbPtr = here->MOS1DPbStructPtr->CSC_Complex ;

            if ((here->MOS1sNodePrime != 0) && (here->MOS1bNode != 0))
                here->MOS1SPbPtr = here->MOS1SPbStructPtr->CSC_Complex ;

            if ((here->MOS1sNodePrime != 0) && (here->MOS1dNodePrime != 0))
                here->MOS1SPdpPtr = here->MOS1SPdpStructPtr->CSC_Complex ;

        }
    }

    return (OK) ;
}

int
MOS1bindCSCComplexToReal (GENmodel *inModel, CKTcircuit *ckt)
{
    MOS1model *model = (MOS1model *)inModel ;
    MOS1instance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the MOS1 models */
    for ( ; model != NULL ; model = model->MOS1nextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->MOS1instances ; here != NULL ; here = here->MOS1nextInstance)
        {
            if ((here->MOS1dNode != 0) && (here->MOS1dNode != 0))
                here->MOS1DdPtr = here->MOS1DdStructPtr->CSC ;

            if ((here->MOS1gNode != 0) && (here->MOS1gNode != 0))
                here->MOS1GgPtr = here->MOS1GgStructPtr->CSC ;

            if ((here->MOS1sNode != 0) && (here->MOS1sNode != 0))
                here->MOS1SsPtr = here->MOS1SsStructPtr->CSC ;

            if ((here->MOS1bNode != 0) && (here->MOS1bNode != 0))
                here->MOS1BbPtr = here->MOS1BbStructPtr->CSC ;

            if ((here->MOS1dNodePrime != 0) && (here->MOS1dNodePrime != 0))
                here->MOS1DPdpPtr = here->MOS1DPdpStructPtr->CSC ;

            if ((here->MOS1sNodePrime != 0) && (here->MOS1sNodePrime != 0))
                here->MOS1SPspPtr = here->MOS1SPspStructPtr->CSC ;

            if ((here->MOS1dNode != 0) && (here->MOS1dNodePrime != 0))
                here->MOS1DdpPtr = here->MOS1DdpStructPtr->CSC ;

            if ((here->MOS1gNode != 0) && (here->MOS1bNode != 0))
                here->MOS1GbPtr = here->MOS1GbStructPtr->CSC ;

            if ((here->MOS1gNode != 0) && (here->MOS1dNodePrime != 0))
                here->MOS1GdpPtr = here->MOS1GdpStructPtr->CSC ;

            if ((here->MOS1gNode != 0) && (here->MOS1sNodePrime != 0))
                here->MOS1GspPtr = here->MOS1GspStructPtr->CSC ;

            if ((here->MOS1sNode != 0) && (here->MOS1sNodePrime != 0))
                here->MOS1SspPtr = here->MOS1SspStructPtr->CSC ;

            if ((here->MOS1bNode != 0) && (here->MOS1dNodePrime != 0))
                here->MOS1BdpPtr = here->MOS1BdpStructPtr->CSC ;

            if ((here->MOS1bNode != 0) && (here->MOS1sNodePrime != 0))
                here->MOS1BspPtr = here->MOS1BspStructPtr->CSC ;

            if ((here->MOS1dNodePrime != 0) && (here->MOS1sNodePrime != 0))
                here->MOS1DPspPtr = here->MOS1DPspStructPtr->CSC ;

            if ((here->MOS1dNodePrime != 0) && (here->MOS1dNode != 0))
                here->MOS1DPdPtr = here->MOS1DPdStructPtr->CSC ;

            if ((here->MOS1bNode != 0) && (here->MOS1gNode != 0))
                here->MOS1BgPtr = here->MOS1BgStructPtr->CSC ;

            if ((here->MOS1dNodePrime != 0) && (here->MOS1gNode != 0))
                here->MOS1DPgPtr = here->MOS1DPgStructPtr->CSC ;

            if ((here->MOS1sNodePrime != 0) && (here->MOS1gNode != 0))
                here->MOS1SPgPtr = here->MOS1SPgStructPtr->CSC ;

            if ((here->MOS1sNodePrime != 0) && (here->MOS1sNode != 0))
                here->MOS1SPsPtr = here->MOS1SPsStructPtr->CSC ;

            if ((here->MOS1dNodePrime != 0) && (here->MOS1bNode != 0))
                here->MOS1DPbPtr = here->MOS1DPbStructPtr->CSC ;

            if ((here->MOS1sNodePrime != 0) && (here->MOS1bNode != 0))
                here->MOS1SPbPtr = here->MOS1SPbStructPtr->CSC ;

            if ((here->MOS1sNodePrime != 0) && (here->MOS1dNodePrime != 0))
                here->MOS1SPdpPtr = here->MOS1SPdpStructPtr->CSC ;

        }
    }

    return (OK) ;
}
