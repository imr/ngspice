/**********
Author: 2013 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "mos2defs.h"
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
MOS2bindCSC (GENmodel *inModel, CKTcircuit *ckt)
{
    MOS2model *model = (MOS2model *)inModel ;
    MOS2instance *here ;
    double *i ;
    BindElement *matched, *BindStruct ;
    size_t nz ;

    BindStruct = ckt->CKTmatrix->CKTbindStruct ;
    nz = (size_t)ckt->CKTmatrix->CKTklunz ;

    /* loop through all the MOS2 models */
    for ( ; model != NULL ; model = model->MOS2nextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->MOS2instances ; here != NULL ; here = here->MOS2nextInstance)
        {
            if ((here-> MOS2dNode != 0) && (here-> MOS2dNode != 0))
            {
                i = here->MOS2DdPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MOS2DdStructPtr = matched ;
                here->MOS2DdPtr = matched->CSC ;
            }

            if ((here-> MOS2gNode != 0) && (here-> MOS2gNode != 0))
            {
                i = here->MOS2GgPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MOS2GgStructPtr = matched ;
                here->MOS2GgPtr = matched->CSC ;
            }

            if ((here-> MOS2sNode != 0) && (here-> MOS2sNode != 0))
            {
                i = here->MOS2SsPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MOS2SsStructPtr = matched ;
                here->MOS2SsPtr = matched->CSC ;
            }

            if ((here-> MOS2bNode != 0) && (here-> MOS2bNode != 0))
            {
                i = here->MOS2BbPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MOS2BbStructPtr = matched ;
                here->MOS2BbPtr = matched->CSC ;
            }

            if ((here-> MOS2dNodePrime != 0) && (here-> MOS2dNodePrime != 0))
            {
                i = here->MOS2DPdpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MOS2DPdpStructPtr = matched ;
                here->MOS2DPdpPtr = matched->CSC ;
            }

            if ((here-> MOS2sNodePrime != 0) && (here-> MOS2sNodePrime != 0))
            {
                i = here->MOS2SPspPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MOS2SPspStructPtr = matched ;
                here->MOS2SPspPtr = matched->CSC ;
            }

            if ((here-> MOS2dNode != 0) && (here-> MOS2dNodePrime != 0))
            {
                i = here->MOS2DdpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MOS2DdpStructPtr = matched ;
                here->MOS2DdpPtr = matched->CSC ;
            }

            if ((here-> MOS2gNode != 0) && (here-> MOS2bNode != 0))
            {
                i = here->MOS2GbPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MOS2GbStructPtr = matched ;
                here->MOS2GbPtr = matched->CSC ;
            }

            if ((here-> MOS2gNode != 0) && (here-> MOS2dNodePrime != 0))
            {
                i = here->MOS2GdpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MOS2GdpStructPtr = matched ;
                here->MOS2GdpPtr = matched->CSC ;
            }

            if ((here-> MOS2gNode != 0) && (here-> MOS2sNodePrime != 0))
            {
                i = here->MOS2GspPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MOS2GspStructPtr = matched ;
                here->MOS2GspPtr = matched->CSC ;
            }

            if ((here-> MOS2sNode != 0) && (here-> MOS2sNodePrime != 0))
            {
                i = here->MOS2SspPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MOS2SspStructPtr = matched ;
                here->MOS2SspPtr = matched->CSC ;
            }

            if ((here-> MOS2bNode != 0) && (here-> MOS2dNodePrime != 0))
            {
                i = here->MOS2BdpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MOS2BdpStructPtr = matched ;
                here->MOS2BdpPtr = matched->CSC ;
            }

            if ((here-> MOS2bNode != 0) && (here-> MOS2sNodePrime != 0))
            {
                i = here->MOS2BspPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MOS2BspStructPtr = matched ;
                here->MOS2BspPtr = matched->CSC ;
            }

            if ((here-> MOS2dNodePrime != 0) && (here-> MOS2sNodePrime != 0))
            {
                i = here->MOS2DPspPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MOS2DPspStructPtr = matched ;
                here->MOS2DPspPtr = matched->CSC ;
            }

            if ((here-> MOS2dNodePrime != 0) && (here-> MOS2dNode != 0))
            {
                i = here->MOS2DPdPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MOS2DPdStructPtr = matched ;
                here->MOS2DPdPtr = matched->CSC ;
            }

            if ((here-> MOS2bNode != 0) && (here-> MOS2gNode != 0))
            {
                i = here->MOS2BgPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MOS2BgStructPtr = matched ;
                here->MOS2BgPtr = matched->CSC ;
            }

            if ((here-> MOS2dNodePrime != 0) && (here-> MOS2gNode != 0))
            {
                i = here->MOS2DPgPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MOS2DPgStructPtr = matched ;
                here->MOS2DPgPtr = matched->CSC ;
            }

            if ((here-> MOS2sNodePrime != 0) && (here-> MOS2gNode != 0))
            {
                i = here->MOS2SPgPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MOS2SPgStructPtr = matched ;
                here->MOS2SPgPtr = matched->CSC ;
            }

            if ((here-> MOS2sNodePrime != 0) && (here-> MOS2sNode != 0))
            {
                i = here->MOS2SPsPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MOS2SPsStructPtr = matched ;
                here->MOS2SPsPtr = matched->CSC ;
            }

            if ((here-> MOS2dNodePrime != 0) && (here-> MOS2bNode != 0))
            {
                i = here->MOS2DPbPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MOS2DPbStructPtr = matched ;
                here->MOS2DPbPtr = matched->CSC ;
            }

            if ((here-> MOS2sNodePrime != 0) && (here-> MOS2bNode != 0))
            {
                i = here->MOS2SPbPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MOS2SPbStructPtr = matched ;
                here->MOS2SPbPtr = matched->CSC ;
            }

            if ((here-> MOS2sNodePrime != 0) && (here-> MOS2dNodePrime != 0))
            {
                i = here->MOS2SPdpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MOS2SPdpStructPtr = matched ;
                here->MOS2SPdpPtr = matched->CSC ;
            }

        }
    }

    return (OK) ;
}

int
MOS2bindCSCComplex (GENmodel *inModel, CKTcircuit *ckt)
{
    MOS2model *model = (MOS2model *)inModel ;
    MOS2instance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the MOS2 models */
    for ( ; model != NULL ; model = model->MOS2nextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->MOS2instances ; here != NULL ; here = here->MOS2nextInstance)
        {
            if ((here-> MOS2dNode != 0) && (here-> MOS2dNode != 0))
                here->MOS2DdPtr = here->MOS2DdStructPtr->CSC_Complex ;

            if ((here-> MOS2gNode != 0) && (here-> MOS2gNode != 0))
                here->MOS2GgPtr = here->MOS2GgStructPtr->CSC_Complex ;

            if ((here-> MOS2sNode != 0) && (here-> MOS2sNode != 0))
                here->MOS2SsPtr = here->MOS2SsStructPtr->CSC_Complex ;

            if ((here-> MOS2bNode != 0) && (here-> MOS2bNode != 0))
                here->MOS2BbPtr = here->MOS2BbStructPtr->CSC_Complex ;

            if ((here-> MOS2dNodePrime != 0) && (here-> MOS2dNodePrime != 0))
                here->MOS2DPdpPtr = here->MOS2DPdpStructPtr->CSC_Complex ;

            if ((here-> MOS2sNodePrime != 0) && (here-> MOS2sNodePrime != 0))
                here->MOS2SPspPtr = here->MOS2SPspStructPtr->CSC_Complex ;

            if ((here-> MOS2dNode != 0) && (here-> MOS2dNodePrime != 0))
                here->MOS2DdpPtr = here->MOS2DdpStructPtr->CSC_Complex ;

            if ((here-> MOS2gNode != 0) && (here-> MOS2bNode != 0))
                here->MOS2GbPtr = here->MOS2GbStructPtr->CSC_Complex ;

            if ((here-> MOS2gNode != 0) && (here-> MOS2dNodePrime != 0))
                here->MOS2GdpPtr = here->MOS2GdpStructPtr->CSC_Complex ;

            if ((here-> MOS2gNode != 0) && (here-> MOS2sNodePrime != 0))
                here->MOS2GspPtr = here->MOS2GspStructPtr->CSC_Complex ;

            if ((here-> MOS2sNode != 0) && (here-> MOS2sNodePrime != 0))
                here->MOS2SspPtr = here->MOS2SspStructPtr->CSC_Complex ;

            if ((here-> MOS2bNode != 0) && (here-> MOS2dNodePrime != 0))
                here->MOS2BdpPtr = here->MOS2BdpStructPtr->CSC_Complex ;

            if ((here-> MOS2bNode != 0) && (here-> MOS2sNodePrime != 0))
                here->MOS2BspPtr = here->MOS2BspStructPtr->CSC_Complex ;

            if ((here-> MOS2dNodePrime != 0) && (here-> MOS2sNodePrime != 0))
                here->MOS2DPspPtr = here->MOS2DPspStructPtr->CSC_Complex ;

            if ((here-> MOS2dNodePrime != 0) && (here-> MOS2dNode != 0))
                here->MOS2DPdPtr = here->MOS2DPdStructPtr->CSC_Complex ;

            if ((here-> MOS2bNode != 0) && (here-> MOS2gNode != 0))
                here->MOS2BgPtr = here->MOS2BgStructPtr->CSC_Complex ;

            if ((here-> MOS2dNodePrime != 0) && (here-> MOS2gNode != 0))
                here->MOS2DPgPtr = here->MOS2DPgStructPtr->CSC_Complex ;

            if ((here-> MOS2sNodePrime != 0) && (here-> MOS2gNode != 0))
                here->MOS2SPgPtr = here->MOS2SPgStructPtr->CSC_Complex ;

            if ((here-> MOS2sNodePrime != 0) && (here-> MOS2sNode != 0))
                here->MOS2SPsPtr = here->MOS2SPsStructPtr->CSC_Complex ;

            if ((here-> MOS2dNodePrime != 0) && (here-> MOS2bNode != 0))
                here->MOS2DPbPtr = here->MOS2DPbStructPtr->CSC_Complex ;

            if ((here-> MOS2sNodePrime != 0) && (here-> MOS2bNode != 0))
                here->MOS2SPbPtr = here->MOS2SPbStructPtr->CSC_Complex ;

            if ((here-> MOS2sNodePrime != 0) && (here-> MOS2dNodePrime != 0))
                here->MOS2SPdpPtr = here->MOS2SPdpStructPtr->CSC_Complex ;

        }
    }

    return (OK) ;
}

int
MOS2bindCSCComplexToReal (GENmodel *inModel, CKTcircuit *ckt)
{
    MOS2model *model = (MOS2model *)inModel ;
    MOS2instance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the MOS2 models */
    for ( ; model != NULL ; model = model->MOS2nextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->MOS2instances ; here != NULL ; here = here->MOS2nextInstance)
        {
            if ((here-> MOS2dNode != 0) && (here-> MOS2dNode != 0))
                here->MOS2DdPtr = here->MOS2DdStructPtr->CSC ;

            if ((here-> MOS2gNode != 0) && (here-> MOS2gNode != 0))
                here->MOS2GgPtr = here->MOS2GgStructPtr->CSC ;

            if ((here-> MOS2sNode != 0) && (here-> MOS2sNode != 0))
                here->MOS2SsPtr = here->MOS2SsStructPtr->CSC ;

            if ((here-> MOS2bNode != 0) && (here-> MOS2bNode != 0))
                here->MOS2BbPtr = here->MOS2BbStructPtr->CSC ;

            if ((here-> MOS2dNodePrime != 0) && (here-> MOS2dNodePrime != 0))
                here->MOS2DPdpPtr = here->MOS2DPdpStructPtr->CSC ;

            if ((here-> MOS2sNodePrime != 0) && (here-> MOS2sNodePrime != 0))
                here->MOS2SPspPtr = here->MOS2SPspStructPtr->CSC ;

            if ((here-> MOS2dNode != 0) && (here-> MOS2dNodePrime != 0))
                here->MOS2DdpPtr = here->MOS2DdpStructPtr->CSC ;

            if ((here-> MOS2gNode != 0) && (here-> MOS2bNode != 0))
                here->MOS2GbPtr = here->MOS2GbStructPtr->CSC ;

            if ((here-> MOS2gNode != 0) && (here-> MOS2dNodePrime != 0))
                here->MOS2GdpPtr = here->MOS2GdpStructPtr->CSC ;

            if ((here-> MOS2gNode != 0) && (here-> MOS2sNodePrime != 0))
                here->MOS2GspPtr = here->MOS2GspStructPtr->CSC ;

            if ((here-> MOS2sNode != 0) && (here-> MOS2sNodePrime != 0))
                here->MOS2SspPtr = here->MOS2SspStructPtr->CSC ;

            if ((here-> MOS2bNode != 0) && (here-> MOS2dNodePrime != 0))
                here->MOS2BdpPtr = here->MOS2BdpStructPtr->CSC ;

            if ((here-> MOS2bNode != 0) && (here-> MOS2sNodePrime != 0))
                here->MOS2BspPtr = here->MOS2BspStructPtr->CSC ;

            if ((here-> MOS2dNodePrime != 0) && (here-> MOS2sNodePrime != 0))
                here->MOS2DPspPtr = here->MOS2DPspStructPtr->CSC ;

            if ((here-> MOS2dNodePrime != 0) && (here-> MOS2dNode != 0))
                here->MOS2DPdPtr = here->MOS2DPdStructPtr->CSC ;

            if ((here-> MOS2bNode != 0) && (here-> MOS2gNode != 0))
                here->MOS2BgPtr = here->MOS2BgStructPtr->CSC ;

            if ((here-> MOS2dNodePrime != 0) && (here-> MOS2gNode != 0))
                here->MOS2DPgPtr = here->MOS2DPgStructPtr->CSC ;

            if ((here-> MOS2sNodePrime != 0) && (here-> MOS2gNode != 0))
                here->MOS2SPgPtr = here->MOS2SPgStructPtr->CSC ;

            if ((here-> MOS2sNodePrime != 0) && (here-> MOS2sNode != 0))
                here->MOS2SPsPtr = here->MOS2SPsStructPtr->CSC ;

            if ((here-> MOS2dNodePrime != 0) && (here-> MOS2bNode != 0))
                here->MOS2DPbPtr = here->MOS2DPbStructPtr->CSC ;

            if ((here-> MOS2sNodePrime != 0) && (here-> MOS2bNode != 0))
                here->MOS2SPbPtr = here->MOS2SPbStructPtr->CSC ;

            if ((here-> MOS2sNodePrime != 0) && (here-> MOS2dNodePrime != 0))
                here->MOS2SPdpPtr = here->MOS2SPdpStructPtr->CSC ;

        }
    }

    return (OK) ;
}
