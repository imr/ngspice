/**********
Author: 2013 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "bsim2def.h"
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
B2bindCSC (GENmodel *inModel, CKTcircuit *ckt)
{
    B2model *model = (B2model *)inModel ;
    B2instance *here ;
    double *i ;
    BindElement *matched, *BindStruct ;
    size_t nz ;

    BindStruct = ckt->CKTmatrix->CKTbindStruct ;
    nz = (size_t)ckt->CKTmatrix->CKTklunz ;

    /* loop through all the B2 models */
    for ( ; model != NULL ; model = model->B2nextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->B2instances ; here != NULL ; here = here->B2nextInstance)
        {
            if ((here-> B2dNode != 0) && (here-> B2dNode != 0))
            {
                i = here->B2DdPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B2DdStructPtr = matched ;
                here->B2DdPtr = matched->CSC ;
            }

            if ((here-> B2gNode != 0) && (here-> B2gNode != 0))
            {
                i = here->B2GgPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B2GgStructPtr = matched ;
                here->B2GgPtr = matched->CSC ;
            }

            if ((here-> B2sNode != 0) && (here-> B2sNode != 0))
            {
                i = here->B2SsPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B2SsStructPtr = matched ;
                here->B2SsPtr = matched->CSC ;
            }

            if ((here-> B2bNode != 0) && (here-> B2bNode != 0))
            {
                i = here->B2BbPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B2BbStructPtr = matched ;
                here->B2BbPtr = matched->CSC ;
            }

            if ((here-> B2dNodePrime != 0) && (here-> B2dNodePrime != 0))
            {
                i = here->B2DPdpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B2DPdpStructPtr = matched ;
                here->B2DPdpPtr = matched->CSC ;
            }

            if ((here-> B2sNodePrime != 0) && (here-> B2sNodePrime != 0))
            {
                i = here->B2SPspPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B2SPspStructPtr = matched ;
                here->B2SPspPtr = matched->CSC ;
            }

            if ((here-> B2dNode != 0) && (here-> B2dNodePrime != 0))
            {
                i = here->B2DdpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B2DdpStructPtr = matched ;
                here->B2DdpPtr = matched->CSC ;
            }

            if ((here-> B2gNode != 0) && (here-> B2bNode != 0))
            {
                i = here->B2GbPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B2GbStructPtr = matched ;
                here->B2GbPtr = matched->CSC ;
            }

            if ((here-> B2gNode != 0) && (here-> B2dNodePrime != 0))
            {
                i = here->B2GdpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B2GdpStructPtr = matched ;
                here->B2GdpPtr = matched->CSC ;
            }

            if ((here-> B2gNode != 0) && (here-> B2sNodePrime != 0))
            {
                i = here->B2GspPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B2GspStructPtr = matched ;
                here->B2GspPtr = matched->CSC ;
            }

            if ((here-> B2sNode != 0) && (here-> B2sNodePrime != 0))
            {
                i = here->B2SspPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B2SspStructPtr = matched ;
                here->B2SspPtr = matched->CSC ;
            }

            if ((here-> B2bNode != 0) && (here-> B2dNodePrime != 0))
            {
                i = here->B2BdpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B2BdpStructPtr = matched ;
                here->B2BdpPtr = matched->CSC ;
            }

            if ((here-> B2bNode != 0) && (here-> B2sNodePrime != 0))
            {
                i = here->B2BspPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B2BspStructPtr = matched ;
                here->B2BspPtr = matched->CSC ;
            }

            if ((here-> B2dNodePrime != 0) && (here-> B2sNodePrime != 0))
            {
                i = here->B2DPspPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B2DPspStructPtr = matched ;
                here->B2DPspPtr = matched->CSC ;
            }

            if ((here-> B2dNodePrime != 0) && (here-> B2dNode != 0))
            {
                i = here->B2DPdPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B2DPdStructPtr = matched ;
                here->B2DPdPtr = matched->CSC ;
            }

            if ((here-> B2bNode != 0) && (here-> B2gNode != 0))
            {
                i = here->B2BgPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B2BgStructPtr = matched ;
                here->B2BgPtr = matched->CSC ;
            }

            if ((here-> B2dNodePrime != 0) && (here-> B2gNode != 0))
            {
                i = here->B2DPgPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B2DPgStructPtr = matched ;
                here->B2DPgPtr = matched->CSC ;
            }

            if ((here-> B2sNodePrime != 0) && (here-> B2gNode != 0))
            {
                i = here->B2SPgPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B2SPgStructPtr = matched ;
                here->B2SPgPtr = matched->CSC ;
            }

            if ((here-> B2sNodePrime != 0) && (here-> B2sNode != 0))
            {
                i = here->B2SPsPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B2SPsStructPtr = matched ;
                here->B2SPsPtr = matched->CSC ;
            }

            if ((here-> B2dNodePrime != 0) && (here-> B2bNode != 0))
            {
                i = here->B2DPbPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B2DPbStructPtr = matched ;
                here->B2DPbPtr = matched->CSC ;
            }

            if ((here-> B2sNodePrime != 0) && (here-> B2bNode != 0))
            {
                i = here->B2SPbPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B2SPbStructPtr = matched ;
                here->B2SPbPtr = matched->CSC ;
            }

            if ((here-> B2sNodePrime != 0) && (here-> B2dNodePrime != 0))
            {
                i = here->B2SPdpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->B2SPdpStructPtr = matched ;
                here->B2SPdpPtr = matched->CSC ;
            }

        }
    }

    return (OK) ;
}

int
B2bindCSCComplex (GENmodel *inModel, CKTcircuit *ckt)
{
    B2model *model = (B2model *)inModel ;
    B2instance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the B2 models */
    for ( ; model != NULL ; model = model->B2nextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->B2instances ; here != NULL ; here = here->B2nextInstance)
        {
            if ((here-> B2dNode != 0) && (here-> B2dNode != 0))
                here->B2DdPtr = here->B2DdStructPtr->CSC_Complex ;

            if ((here-> B2gNode != 0) && (here-> B2gNode != 0))
                here->B2GgPtr = here->B2GgStructPtr->CSC_Complex ;

            if ((here-> B2sNode != 0) && (here-> B2sNode != 0))
                here->B2SsPtr = here->B2SsStructPtr->CSC_Complex ;

            if ((here-> B2bNode != 0) && (here-> B2bNode != 0))
                here->B2BbPtr = here->B2BbStructPtr->CSC_Complex ;

            if ((here-> B2dNodePrime != 0) && (here-> B2dNodePrime != 0))
                here->B2DPdpPtr = here->B2DPdpStructPtr->CSC_Complex ;

            if ((here-> B2sNodePrime != 0) && (here-> B2sNodePrime != 0))
                here->B2SPspPtr = here->B2SPspStructPtr->CSC_Complex ;

            if ((here-> B2dNode != 0) && (here-> B2dNodePrime != 0))
                here->B2DdpPtr = here->B2DdpStructPtr->CSC_Complex ;

            if ((here-> B2gNode != 0) && (here-> B2bNode != 0))
                here->B2GbPtr = here->B2GbStructPtr->CSC_Complex ;

            if ((here-> B2gNode != 0) && (here-> B2dNodePrime != 0))
                here->B2GdpPtr = here->B2GdpStructPtr->CSC_Complex ;

            if ((here-> B2gNode != 0) && (here-> B2sNodePrime != 0))
                here->B2GspPtr = here->B2GspStructPtr->CSC_Complex ;

            if ((here-> B2sNode != 0) && (here-> B2sNodePrime != 0))
                here->B2SspPtr = here->B2SspStructPtr->CSC_Complex ;

            if ((here-> B2bNode != 0) && (here-> B2dNodePrime != 0))
                here->B2BdpPtr = here->B2BdpStructPtr->CSC_Complex ;

            if ((here-> B2bNode != 0) && (here-> B2sNodePrime != 0))
                here->B2BspPtr = here->B2BspStructPtr->CSC_Complex ;

            if ((here-> B2dNodePrime != 0) && (here-> B2sNodePrime != 0))
                here->B2DPspPtr = here->B2DPspStructPtr->CSC_Complex ;

            if ((here-> B2dNodePrime != 0) && (here-> B2dNode != 0))
                here->B2DPdPtr = here->B2DPdStructPtr->CSC_Complex ;

            if ((here-> B2bNode != 0) && (here-> B2gNode != 0))
                here->B2BgPtr = here->B2BgStructPtr->CSC_Complex ;

            if ((here-> B2dNodePrime != 0) && (here-> B2gNode != 0))
                here->B2DPgPtr = here->B2DPgStructPtr->CSC_Complex ;

            if ((here-> B2sNodePrime != 0) && (here-> B2gNode != 0))
                here->B2SPgPtr = here->B2SPgStructPtr->CSC_Complex ;

            if ((here-> B2sNodePrime != 0) && (here-> B2sNode != 0))
                here->B2SPsPtr = here->B2SPsStructPtr->CSC_Complex ;

            if ((here-> B2dNodePrime != 0) && (here-> B2bNode != 0))
                here->B2DPbPtr = here->B2DPbStructPtr->CSC_Complex ;

            if ((here-> B2sNodePrime != 0) && (here-> B2bNode != 0))
                here->B2SPbPtr = here->B2SPbStructPtr->CSC_Complex ;

            if ((here-> B2sNodePrime != 0) && (here-> B2dNodePrime != 0))
                here->B2SPdpPtr = here->B2SPdpStructPtr->CSC_Complex ;

        }
    }

    return (OK) ;
}

int
B2bindCSCComplexToReal (GENmodel *inModel, CKTcircuit *ckt)
{
    B2model *model = (B2model *)inModel ;
    B2instance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the B2 models */
    for ( ; model != NULL ; model = model->B2nextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->B2instances ; here != NULL ; here = here->B2nextInstance)
        {
            if ((here-> B2dNode != 0) && (here-> B2dNode != 0))
                here->B2DdPtr = here->B2DdStructPtr->CSC ;

            if ((here-> B2gNode != 0) && (here-> B2gNode != 0))
                here->B2GgPtr = here->B2GgStructPtr->CSC ;

            if ((here-> B2sNode != 0) && (here-> B2sNode != 0))
                here->B2SsPtr = here->B2SsStructPtr->CSC ;

            if ((here-> B2bNode != 0) && (here-> B2bNode != 0))
                here->B2BbPtr = here->B2BbStructPtr->CSC ;

            if ((here-> B2dNodePrime != 0) && (here-> B2dNodePrime != 0))
                here->B2DPdpPtr = here->B2DPdpStructPtr->CSC ;

            if ((here-> B2sNodePrime != 0) && (here-> B2sNodePrime != 0))
                here->B2SPspPtr = here->B2SPspStructPtr->CSC ;

            if ((here-> B2dNode != 0) && (here-> B2dNodePrime != 0))
                here->B2DdpPtr = here->B2DdpStructPtr->CSC ;

            if ((here-> B2gNode != 0) && (here-> B2bNode != 0))
                here->B2GbPtr = here->B2GbStructPtr->CSC ;

            if ((here-> B2gNode != 0) && (here-> B2dNodePrime != 0))
                here->B2GdpPtr = here->B2GdpStructPtr->CSC ;

            if ((here-> B2gNode != 0) && (here-> B2sNodePrime != 0))
                here->B2GspPtr = here->B2GspStructPtr->CSC ;

            if ((here-> B2sNode != 0) && (here-> B2sNodePrime != 0))
                here->B2SspPtr = here->B2SspStructPtr->CSC ;

            if ((here-> B2bNode != 0) && (here-> B2dNodePrime != 0))
                here->B2BdpPtr = here->B2BdpStructPtr->CSC ;

            if ((here-> B2bNode != 0) && (here-> B2sNodePrime != 0))
                here->B2BspPtr = here->B2BspStructPtr->CSC ;

            if ((here-> B2dNodePrime != 0) && (here-> B2sNodePrime != 0))
                here->B2DPspPtr = here->B2DPspStructPtr->CSC ;

            if ((here-> B2dNodePrime != 0) && (here-> B2dNode != 0))
                here->B2DPdPtr = here->B2DPdStructPtr->CSC ;

            if ((here-> B2bNode != 0) && (here-> B2gNode != 0))
                here->B2BgPtr = here->B2BgStructPtr->CSC ;

            if ((here-> B2dNodePrime != 0) && (here-> B2gNode != 0))
                here->B2DPgPtr = here->B2DPgStructPtr->CSC ;

            if ((here-> B2sNodePrime != 0) && (here-> B2gNode != 0))
                here->B2SPgPtr = here->B2SPgStructPtr->CSC ;

            if ((here-> B2sNodePrime != 0) && (here-> B2sNode != 0))
                here->B2SPsPtr = here->B2SPsStructPtr->CSC ;

            if ((here-> B2dNodePrime != 0) && (here-> B2bNode != 0))
                here->B2DPbPtr = here->B2DPbStructPtr->CSC ;

            if ((here-> B2sNodePrime != 0) && (here-> B2bNode != 0))
                here->B2SPbPtr = here->B2SPbStructPtr->CSC ;

            if ((here-> B2sNodePrime != 0) && (here-> B2dNodePrime != 0))
                here->B2SPdpPtr = here->B2SPdpStructPtr->CSC ;

        }
    }

    return (OK) ;
}
