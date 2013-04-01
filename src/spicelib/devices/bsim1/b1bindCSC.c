/**********
Author: 2013 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "bsim1def.h"
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
B1bindCSC (GENmodel *inModel, CKTcircuit *ckt)
{
    B1model *model = (B1model *)inModel ;
    B1instance *here ;
    double *i ;
    BindElement *matched, *BindStruct ;
    size_t nz ;

    BindStruct = ckt->CKTmatrix->CKTbindStruct ;
    nz = (size_t)ckt->CKTmatrix->CKTklunz ;

    /* loop through all the B1 models */
    for ( ; model != NULL ; model = model->B1nextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->B1instances ; here != NULL ; here = here->B1nextInstance)
        {
            if ((here-> B1dNode != 0) && (here-> B1dNode != 0))
            {
                i = here->B1DdPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B1DdStructPtr = matched ;
                here->B1DdPtr = matched->CSC ;
            }

            if ((here-> B1gNode != 0) && (here-> B1gNode != 0))
            {
                i = here->B1GgPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B1GgStructPtr = matched ;
                here->B1GgPtr = matched->CSC ;
            }

            if ((here-> B1sNode != 0) && (here-> B1sNode != 0))
            {
                i = here->B1SsPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B1SsStructPtr = matched ;
                here->B1SsPtr = matched->CSC ;
            }

            if ((here-> B1bNode != 0) && (here-> B1bNode != 0))
            {
                i = here->B1BbPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B1BbStructPtr = matched ;
                here->B1BbPtr = matched->CSC ;
            }

            if ((here-> B1dNodePrime != 0) && (here-> B1dNodePrime != 0))
            {
                i = here->B1DPdpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B1DPdpStructPtr = matched ;
                here->B1DPdpPtr = matched->CSC ;
            }

            if ((here-> B1sNodePrime != 0) && (here-> B1sNodePrime != 0))
            {
                i = here->B1SPspPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B1SPspStructPtr = matched ;
                here->B1SPspPtr = matched->CSC ;
            }

            if ((here-> B1dNode != 0) && (here-> B1dNodePrime != 0))
            {
                i = here->B1DdpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B1DdpStructPtr = matched ;
                here->B1DdpPtr = matched->CSC ;
            }

            if ((here-> B1gNode != 0) && (here-> B1bNode != 0))
            {
                i = here->B1GbPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B1GbStructPtr = matched ;
                here->B1GbPtr = matched->CSC ;
            }

            if ((here-> B1gNode != 0) && (here-> B1dNodePrime != 0))
            {
                i = here->B1GdpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B1GdpStructPtr = matched ;
                here->B1GdpPtr = matched->CSC ;
            }

            if ((here-> B1gNode != 0) && (here-> B1sNodePrime != 0))
            {
                i = here->B1GspPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B1GspStructPtr = matched ;
                here->B1GspPtr = matched->CSC ;
            }

            if ((here-> B1sNode != 0) && (here-> B1sNodePrime != 0))
            {
                i = here->B1SspPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B1SspStructPtr = matched ;
                here->B1SspPtr = matched->CSC ;
            }

            if ((here-> B1bNode != 0) && (here-> B1dNodePrime != 0))
            {
                i = here->B1BdpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B1BdpStructPtr = matched ;
                here->B1BdpPtr = matched->CSC ;
            }

            if ((here-> B1bNode != 0) && (here-> B1sNodePrime != 0))
            {
                i = here->B1BspPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B1BspStructPtr = matched ;
                here->B1BspPtr = matched->CSC ;
            }

            if ((here-> B1dNodePrime != 0) && (here-> B1sNodePrime != 0))
            {
                i = here->B1DPspPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B1DPspStructPtr = matched ;
                here->B1DPspPtr = matched->CSC ;
            }

            if ((here-> B1dNodePrime != 0) && (here-> B1dNode != 0))
            {
                i = here->B1DPdPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B1DPdStructPtr = matched ;
                here->B1DPdPtr = matched->CSC ;
            }

            if ((here-> B1bNode != 0) && (here-> B1gNode != 0))
            {
                i = here->B1BgPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B1BgStructPtr = matched ;
                here->B1BgPtr = matched->CSC ;
            }

            if ((here-> B1dNodePrime != 0) && (here-> B1gNode != 0))
            {
                i = here->B1DPgPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B1DPgStructPtr = matched ;
                here->B1DPgPtr = matched->CSC ;
            }

            if ((here-> B1sNodePrime != 0) && (here-> B1gNode != 0))
            {
                i = here->B1SPgPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B1SPgStructPtr = matched ;
                here->B1SPgPtr = matched->CSC ;
            }

            if ((here-> B1sNodePrime != 0) && (here-> B1sNode != 0))
            {
                i = here->B1SPsPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B1SPsStructPtr = matched ;
                here->B1SPsPtr = matched->CSC ;
            }

            if ((here-> B1dNodePrime != 0) && (here-> B1bNode != 0))
            {
                i = here->B1DPbPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B1DPbStructPtr = matched ;
                here->B1DPbPtr = matched->CSC ;
            }

            if ((here-> B1sNodePrime != 0) && (here-> B1bNode != 0))
            {
                i = here->B1SPbPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B1SPbStructPtr = matched ;
                here->B1SPbPtr = matched->CSC ;
            }

            if ((here-> B1sNodePrime != 0) && (here-> B1dNodePrime != 0))
            {
                i = here->B1SPdpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B1SPdpStructPtr = matched ;
                here->B1SPdpPtr = matched->CSC ;
            }

        }
    }

    return (OK) ;
}

int
B1bindCSCComplex (GENmodel *inModel, CKTcircuit *ckt)
{
    B1model *model = (B1model *)inModel ;
    B1instance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the B1 models */
    for ( ; model != NULL ; model = model->B1nextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->B1instances ; here != NULL ; here = here->B1nextInstance)
        {
            if ((here-> B1dNode != 0) && (here-> B1dNode != 0))
                here->B1DdPtr = here->B1DdStructPtr->CSC_Complex ;

            if ((here-> B1gNode != 0) && (here-> B1gNode != 0))
                here->B1GgPtr = here->B1GgStructPtr->CSC_Complex ;

            if ((here-> B1sNode != 0) && (here-> B1sNode != 0))
                here->B1SsPtr = here->B1SsStructPtr->CSC_Complex ;

            if ((here-> B1bNode != 0) && (here-> B1bNode != 0))
                here->B1BbPtr = here->B1BbStructPtr->CSC_Complex ;

            if ((here-> B1dNodePrime != 0) && (here-> B1dNodePrime != 0))
                here->B1DPdpPtr = here->B1DPdpStructPtr->CSC_Complex ;

            if ((here-> B1sNodePrime != 0) && (here-> B1sNodePrime != 0))
                here->B1SPspPtr = here->B1SPspStructPtr->CSC_Complex ;

            if ((here-> B1dNode != 0) && (here-> B1dNodePrime != 0))
                here->B1DdpPtr = here->B1DdpStructPtr->CSC_Complex ;

            if ((here-> B1gNode != 0) && (here-> B1bNode != 0))
                here->B1GbPtr = here->B1GbStructPtr->CSC_Complex ;

            if ((here-> B1gNode != 0) && (here-> B1dNodePrime != 0))
                here->B1GdpPtr = here->B1GdpStructPtr->CSC_Complex ;

            if ((here-> B1gNode != 0) && (here-> B1sNodePrime != 0))
                here->B1GspPtr = here->B1GspStructPtr->CSC_Complex ;

            if ((here-> B1sNode != 0) && (here-> B1sNodePrime != 0))
                here->B1SspPtr = here->B1SspStructPtr->CSC_Complex ;

            if ((here-> B1bNode != 0) && (here-> B1dNodePrime != 0))
                here->B1BdpPtr = here->B1BdpStructPtr->CSC_Complex ;

            if ((here-> B1bNode != 0) && (here-> B1sNodePrime != 0))
                here->B1BspPtr = here->B1BspStructPtr->CSC_Complex ;

            if ((here-> B1dNodePrime != 0) && (here-> B1sNodePrime != 0))
                here->B1DPspPtr = here->B1DPspStructPtr->CSC_Complex ;

            if ((here-> B1dNodePrime != 0) && (here-> B1dNode != 0))
                here->B1DPdPtr = here->B1DPdStructPtr->CSC_Complex ;

            if ((here-> B1bNode != 0) && (here-> B1gNode != 0))
                here->B1BgPtr = here->B1BgStructPtr->CSC_Complex ;

            if ((here-> B1dNodePrime != 0) && (here-> B1gNode != 0))
                here->B1DPgPtr = here->B1DPgStructPtr->CSC_Complex ;

            if ((here-> B1sNodePrime != 0) && (here-> B1gNode != 0))
                here->B1SPgPtr = here->B1SPgStructPtr->CSC_Complex ;

            if ((here-> B1sNodePrime != 0) && (here-> B1sNode != 0))
                here->B1SPsPtr = here->B1SPsStructPtr->CSC_Complex ;

            if ((here-> B1dNodePrime != 0) && (here-> B1bNode != 0))
                here->B1DPbPtr = here->B1DPbStructPtr->CSC_Complex ;

            if ((here-> B1sNodePrime != 0) && (here-> B1bNode != 0))
                here->B1SPbPtr = here->B1SPbStructPtr->CSC_Complex ;

            if ((here-> B1sNodePrime != 0) && (here-> B1dNodePrime != 0))
                here->B1SPdpPtr = here->B1SPdpStructPtr->CSC_Complex ;

        }
    }

    return (OK) ;
}

int
B1bindCSCComplexToReal (GENmodel *inModel, CKTcircuit *ckt)
{
    B1model *model = (B1model *)inModel ;
    B1instance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the B1 models */
    for ( ; model != NULL ; model = model->B1nextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->B1instances ; here != NULL ; here = here->B1nextInstance)
        {
            if ((here-> B1dNode != 0) && (here-> B1dNode != 0))
                here->B1DdPtr = here->B1DdStructPtr->CSC ;

            if ((here-> B1gNode != 0) && (here-> B1gNode != 0))
                here->B1GgPtr = here->B1GgStructPtr->CSC ;

            if ((here-> B1sNode != 0) && (here-> B1sNode != 0))
                here->B1SsPtr = here->B1SsStructPtr->CSC ;

            if ((here-> B1bNode != 0) && (here-> B1bNode != 0))
                here->B1BbPtr = here->B1BbStructPtr->CSC ;

            if ((here-> B1dNodePrime != 0) && (here-> B1dNodePrime != 0))
                here->B1DPdpPtr = here->B1DPdpStructPtr->CSC ;

            if ((here-> B1sNodePrime != 0) && (here-> B1sNodePrime != 0))
                here->B1SPspPtr = here->B1SPspStructPtr->CSC ;

            if ((here-> B1dNode != 0) && (here-> B1dNodePrime != 0))
                here->B1DdpPtr = here->B1DdpStructPtr->CSC ;

            if ((here-> B1gNode != 0) && (here-> B1bNode != 0))
                here->B1GbPtr = here->B1GbStructPtr->CSC ;

            if ((here-> B1gNode != 0) && (here-> B1dNodePrime != 0))
                here->B1GdpPtr = here->B1GdpStructPtr->CSC ;

            if ((here-> B1gNode != 0) && (here-> B1sNodePrime != 0))
                here->B1GspPtr = here->B1GspStructPtr->CSC ;

            if ((here-> B1sNode != 0) && (here-> B1sNodePrime != 0))
                here->B1SspPtr = here->B1SspStructPtr->CSC ;

            if ((here-> B1bNode != 0) && (here-> B1dNodePrime != 0))
                here->B1BdpPtr = here->B1BdpStructPtr->CSC ;

            if ((here-> B1bNode != 0) && (here-> B1sNodePrime != 0))
                here->B1BspPtr = here->B1BspStructPtr->CSC ;

            if ((here-> B1dNodePrime != 0) && (here-> B1sNodePrime != 0))
                here->B1DPspPtr = here->B1DPspStructPtr->CSC ;

            if ((here-> B1dNodePrime != 0) && (here-> B1dNode != 0))
                here->B1DPdPtr = here->B1DPdStructPtr->CSC ;

            if ((here-> B1bNode != 0) && (here-> B1gNode != 0))
                here->B1BgPtr = here->B1BgStructPtr->CSC ;

            if ((here-> B1dNodePrime != 0) && (here-> B1gNode != 0))
                here->B1DPgPtr = here->B1DPgStructPtr->CSC ;

            if ((here-> B1sNodePrime != 0) && (here-> B1gNode != 0))
                here->B1SPgPtr = here->B1SPgStructPtr->CSC ;

            if ((here-> B1sNodePrime != 0) && (here-> B1sNode != 0))
                here->B1SPsPtr = here->B1SPsStructPtr->CSC ;

            if ((here-> B1dNodePrime != 0) && (here-> B1bNode != 0))
                here->B1DPbPtr = here->B1DPbStructPtr->CSC ;

            if ((here-> B1sNodePrime != 0) && (here-> B1bNode != 0))
                here->B1SPbPtr = here->B1SPbStructPtr->CSC ;

            if ((here-> B1sNodePrime != 0) && (here-> B1dNodePrime != 0))
                here->B1SPdpPtr = here->B1SPdpStructPtr->CSC ;

        }
    }

    return (OK) ;
}
