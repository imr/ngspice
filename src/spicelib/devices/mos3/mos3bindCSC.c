/**********
Author: 2013 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "mos3defs.h"
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
MOS3bindCSC (GENmodel *inModel, CKTcircuit *ckt)
{
    MOS3model *model = (MOS3model *)inModel ;
    MOS3instance *here ;
    double *i ;
    BindElement *matched, *BindStruct ;
    size_t nz ;

    BindStruct = ckt->CKTmatrix->CKTbindStruct ;
    nz = (size_t)ckt->CKTmatrix->CKTklunz ;

    /* loop through all the MOS3 models */
    for ( ; model != NULL ; model = model->MOS3nextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->MOS3instances ; here != NULL ; here = here->MOS3nextInstance)
        {
            if ((here-> MOS3dNode != 0) && (here-> MOS3dNode != 0))
            {
                i = here->MOS3DdPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MOS3DdStructPtr = matched ;
                here->MOS3DdPtr = matched->CSC ;
            }

            if ((here-> MOS3gNode != 0) && (here-> MOS3gNode != 0))
            {
                i = here->MOS3GgPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MOS3GgStructPtr = matched ;
                here->MOS3GgPtr = matched->CSC ;
            }

            if ((here-> MOS3sNode != 0) && (here-> MOS3sNode != 0))
            {
                i = here->MOS3SsPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MOS3SsStructPtr = matched ;
                here->MOS3SsPtr = matched->CSC ;
            }

            if ((here-> MOS3bNode != 0) && (here-> MOS3bNode != 0))
            {
                i = here->MOS3BbPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MOS3BbStructPtr = matched ;
                here->MOS3BbPtr = matched->CSC ;
            }

            if ((here-> MOS3dNodePrime != 0) && (here-> MOS3dNodePrime != 0))
            {
                i = here->MOS3DPdpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MOS3DPdpStructPtr = matched ;
                here->MOS3DPdpPtr = matched->CSC ;
            }

            if ((here-> MOS3sNodePrime != 0) && (here-> MOS3sNodePrime != 0))
            {
                i = here->MOS3SPspPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MOS3SPspStructPtr = matched ;
                here->MOS3SPspPtr = matched->CSC ;
            }

            if ((here-> MOS3dNode != 0) && (here-> MOS3dNodePrime != 0))
            {
                i = here->MOS3DdpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MOS3DdpStructPtr = matched ;
                here->MOS3DdpPtr = matched->CSC ;
            }

            if ((here-> MOS3gNode != 0) && (here-> MOS3bNode != 0))
            {
                i = here->MOS3GbPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MOS3GbStructPtr = matched ;
                here->MOS3GbPtr = matched->CSC ;
            }

            if ((here-> MOS3gNode != 0) && (here-> MOS3dNodePrime != 0))
            {
                i = here->MOS3GdpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MOS3GdpStructPtr = matched ;
                here->MOS3GdpPtr = matched->CSC ;
            }

            if ((here-> MOS3gNode != 0) && (here-> MOS3sNodePrime != 0))
            {
                i = here->MOS3GspPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MOS3GspStructPtr = matched ;
                here->MOS3GspPtr = matched->CSC ;
            }

            if ((here-> MOS3sNode != 0) && (here-> MOS3sNodePrime != 0))
            {
                i = here->MOS3SspPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MOS3SspStructPtr = matched ;
                here->MOS3SspPtr = matched->CSC ;
            }

            if ((here-> MOS3bNode != 0) && (here-> MOS3dNodePrime != 0))
            {
                i = here->MOS3BdpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MOS3BdpStructPtr = matched ;
                here->MOS3BdpPtr = matched->CSC ;
            }

            if ((here-> MOS3bNode != 0) && (here-> MOS3sNodePrime != 0))
            {
                i = here->MOS3BspPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MOS3BspStructPtr = matched ;
                here->MOS3BspPtr = matched->CSC ;
            }

            if ((here-> MOS3dNodePrime != 0) && (here-> MOS3sNodePrime != 0))
            {
                i = here->MOS3DPspPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MOS3DPspStructPtr = matched ;
                here->MOS3DPspPtr = matched->CSC ;
            }

            if ((here-> MOS3dNodePrime != 0) && (here-> MOS3dNode != 0))
            {
                i = here->MOS3DPdPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MOS3DPdStructPtr = matched ;
                here->MOS3DPdPtr = matched->CSC ;
            }

            if ((here-> MOS3bNode != 0) && (here-> MOS3gNode != 0))
            {
                i = here->MOS3BgPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MOS3BgStructPtr = matched ;
                here->MOS3BgPtr = matched->CSC ;
            }

            if ((here-> MOS3dNodePrime != 0) && (here-> MOS3gNode != 0))
            {
                i = here->MOS3DPgPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MOS3DPgStructPtr = matched ;
                here->MOS3DPgPtr = matched->CSC ;
            }

            if ((here-> MOS3sNodePrime != 0) && (here-> MOS3gNode != 0))
            {
                i = here->MOS3SPgPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MOS3SPgStructPtr = matched ;
                here->MOS3SPgPtr = matched->CSC ;
            }

            if ((here-> MOS3sNodePrime != 0) && (here-> MOS3sNode != 0))
            {
                i = here->MOS3SPsPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MOS3SPsStructPtr = matched ;
                here->MOS3SPsPtr = matched->CSC ;
            }

            if ((here-> MOS3dNodePrime != 0) && (here-> MOS3bNode != 0))
            {
                i = here->MOS3DPbPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MOS3DPbStructPtr = matched ;
                here->MOS3DPbPtr = matched->CSC ;
            }

            if ((here-> MOS3sNodePrime != 0) && (here-> MOS3bNode != 0))
            {
                i = here->MOS3SPbPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MOS3SPbStructPtr = matched ;
                here->MOS3SPbPtr = matched->CSC ;
            }

            if ((here-> MOS3sNodePrime != 0) && (here-> MOS3dNodePrime != 0))
            {
                i = here->MOS3SPdpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MOS3SPdpStructPtr = matched ;
                here->MOS3SPdpPtr = matched->CSC ;
            }

        }
    }

    return (OK) ;
}

int
MOS3bindCSCComplex (GENmodel *inModel, CKTcircuit *ckt)
{
    MOS3model *model = (MOS3model *)inModel ;
    MOS3instance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the MOS3 models */
    for ( ; model != NULL ; model = model->MOS3nextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->MOS3instances ; here != NULL ; here = here->MOS3nextInstance)
        {
            if ((here-> MOS3dNode != 0) && (here-> MOS3dNode != 0))
                here->MOS3DdPtr = here->MOS3DdStructPtr->CSC_Complex ;

            if ((here-> MOS3gNode != 0) && (here-> MOS3gNode != 0))
                here->MOS3GgPtr = here->MOS3GgStructPtr->CSC_Complex ;

            if ((here-> MOS3sNode != 0) && (here-> MOS3sNode != 0))
                here->MOS3SsPtr = here->MOS3SsStructPtr->CSC_Complex ;

            if ((here-> MOS3bNode != 0) && (here-> MOS3bNode != 0))
                here->MOS3BbPtr = here->MOS3BbStructPtr->CSC_Complex ;

            if ((here-> MOS3dNodePrime != 0) && (here-> MOS3dNodePrime != 0))
                here->MOS3DPdpPtr = here->MOS3DPdpStructPtr->CSC_Complex ;

            if ((here-> MOS3sNodePrime != 0) && (here-> MOS3sNodePrime != 0))
                here->MOS3SPspPtr = here->MOS3SPspStructPtr->CSC_Complex ;

            if ((here-> MOS3dNode != 0) && (here-> MOS3dNodePrime != 0))
                here->MOS3DdpPtr = here->MOS3DdpStructPtr->CSC_Complex ;

            if ((here-> MOS3gNode != 0) && (here-> MOS3bNode != 0))
                here->MOS3GbPtr = here->MOS3GbStructPtr->CSC_Complex ;

            if ((here-> MOS3gNode != 0) && (here-> MOS3dNodePrime != 0))
                here->MOS3GdpPtr = here->MOS3GdpStructPtr->CSC_Complex ;

            if ((here-> MOS3gNode != 0) && (here-> MOS3sNodePrime != 0))
                here->MOS3GspPtr = here->MOS3GspStructPtr->CSC_Complex ;

            if ((here-> MOS3sNode != 0) && (here-> MOS3sNodePrime != 0))
                here->MOS3SspPtr = here->MOS3SspStructPtr->CSC_Complex ;

            if ((here-> MOS3bNode != 0) && (here-> MOS3dNodePrime != 0))
                here->MOS3BdpPtr = here->MOS3BdpStructPtr->CSC_Complex ;

            if ((here-> MOS3bNode != 0) && (here-> MOS3sNodePrime != 0))
                here->MOS3BspPtr = here->MOS3BspStructPtr->CSC_Complex ;

            if ((here-> MOS3dNodePrime != 0) && (here-> MOS3sNodePrime != 0))
                here->MOS3DPspPtr = here->MOS3DPspStructPtr->CSC_Complex ;

            if ((here-> MOS3dNodePrime != 0) && (here-> MOS3dNode != 0))
                here->MOS3DPdPtr = here->MOS3DPdStructPtr->CSC_Complex ;

            if ((here-> MOS3bNode != 0) && (here-> MOS3gNode != 0))
                here->MOS3BgPtr = here->MOS3BgStructPtr->CSC_Complex ;

            if ((here-> MOS3dNodePrime != 0) && (here-> MOS3gNode != 0))
                here->MOS3DPgPtr = here->MOS3DPgStructPtr->CSC_Complex ;

            if ((here-> MOS3sNodePrime != 0) && (here-> MOS3gNode != 0))
                here->MOS3SPgPtr = here->MOS3SPgStructPtr->CSC_Complex ;

            if ((here-> MOS3sNodePrime != 0) && (here-> MOS3sNode != 0))
                here->MOS3SPsPtr = here->MOS3SPsStructPtr->CSC_Complex ;

            if ((here-> MOS3dNodePrime != 0) && (here-> MOS3bNode != 0))
                here->MOS3DPbPtr = here->MOS3DPbStructPtr->CSC_Complex ;

            if ((here-> MOS3sNodePrime != 0) && (here-> MOS3bNode != 0))
                here->MOS3SPbPtr = here->MOS3SPbStructPtr->CSC_Complex ;

            if ((here-> MOS3sNodePrime != 0) && (here-> MOS3dNodePrime != 0))
                here->MOS3SPdpPtr = here->MOS3SPdpStructPtr->CSC_Complex ;

        }
    }

    return (OK) ;
}

int
MOS3bindCSCComplexToReal (GENmodel *inModel, CKTcircuit *ckt)
{
    MOS3model *model = (MOS3model *)inModel ;
    MOS3instance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the MOS3 models */
    for ( ; model != NULL ; model = model->MOS3nextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->MOS3instances ; here != NULL ; here = here->MOS3nextInstance)
        {
            if ((here-> MOS3dNode != 0) && (here-> MOS3dNode != 0))
                here->MOS3DdPtr = here->MOS3DdStructPtr->CSC ;

            if ((here-> MOS3gNode != 0) && (here-> MOS3gNode != 0))
                here->MOS3GgPtr = here->MOS3GgStructPtr->CSC ;

            if ((here-> MOS3sNode != 0) && (here-> MOS3sNode != 0))
                here->MOS3SsPtr = here->MOS3SsStructPtr->CSC ;

            if ((here-> MOS3bNode != 0) && (here-> MOS3bNode != 0))
                here->MOS3BbPtr = here->MOS3BbStructPtr->CSC ;

            if ((here-> MOS3dNodePrime != 0) && (here-> MOS3dNodePrime != 0))
                here->MOS3DPdpPtr = here->MOS3DPdpStructPtr->CSC ;

            if ((here-> MOS3sNodePrime != 0) && (here-> MOS3sNodePrime != 0))
                here->MOS3SPspPtr = here->MOS3SPspStructPtr->CSC ;

            if ((here-> MOS3dNode != 0) && (here-> MOS3dNodePrime != 0))
                here->MOS3DdpPtr = here->MOS3DdpStructPtr->CSC ;

            if ((here-> MOS3gNode != 0) && (here-> MOS3bNode != 0))
                here->MOS3GbPtr = here->MOS3GbStructPtr->CSC ;

            if ((here-> MOS3gNode != 0) && (here-> MOS3dNodePrime != 0))
                here->MOS3GdpPtr = here->MOS3GdpStructPtr->CSC ;

            if ((here-> MOS3gNode != 0) && (here-> MOS3sNodePrime != 0))
                here->MOS3GspPtr = here->MOS3GspStructPtr->CSC ;

            if ((here-> MOS3sNode != 0) && (here-> MOS3sNodePrime != 0))
                here->MOS3SspPtr = here->MOS3SspStructPtr->CSC ;

            if ((here-> MOS3bNode != 0) && (here-> MOS3dNodePrime != 0))
                here->MOS3BdpPtr = here->MOS3BdpStructPtr->CSC ;

            if ((here-> MOS3bNode != 0) && (here-> MOS3sNodePrime != 0))
                here->MOS3BspPtr = here->MOS3BspStructPtr->CSC ;

            if ((here-> MOS3dNodePrime != 0) && (here-> MOS3sNodePrime != 0))
                here->MOS3DPspPtr = here->MOS3DPspStructPtr->CSC ;

            if ((here-> MOS3dNodePrime != 0) && (here-> MOS3dNode != 0))
                here->MOS3DPdPtr = here->MOS3DPdStructPtr->CSC ;

            if ((here-> MOS3bNode != 0) && (here-> MOS3gNode != 0))
                here->MOS3BgPtr = here->MOS3BgStructPtr->CSC ;

            if ((here-> MOS3dNodePrime != 0) && (here-> MOS3gNode != 0))
                here->MOS3DPgPtr = here->MOS3DPgStructPtr->CSC ;

            if ((here-> MOS3sNodePrime != 0) && (here-> MOS3gNode != 0))
                here->MOS3SPgPtr = here->MOS3SPgStructPtr->CSC ;

            if ((here-> MOS3sNodePrime != 0) && (here-> MOS3sNode != 0))
                here->MOS3SPsPtr = here->MOS3SPsStructPtr->CSC ;

            if ((here-> MOS3dNodePrime != 0) && (here-> MOS3bNode != 0))
                here->MOS3DPbPtr = here->MOS3DPbStructPtr->CSC ;

            if ((here-> MOS3sNodePrime != 0) && (here-> MOS3bNode != 0))
                here->MOS3SPbPtr = here->MOS3SPbStructPtr->CSC ;

            if ((here-> MOS3sNodePrime != 0) && (here-> MOS3dNodePrime != 0))
                here->MOS3SPdpPtr = here->MOS3SPdpStructPtr->CSC ;

        }
    }

    return (OK) ;
}
