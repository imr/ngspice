/**********
Author: 2013 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "mos9defs.h"
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
MOS9bindCSC (GENmodel *inModel, CKTcircuit *ckt)
{
    MOS9model *model = (MOS9model *)inModel ;
    MOS9instance *here ;
    double *i ;
    BindElement *matched, *BindStruct ;
    size_t nz ;

    BindStruct = ckt->CKTmatrix->CKTbindStruct ;
    nz = (size_t)ckt->CKTmatrix->CKTklunz ;

    /* loop through all the MOS9 models */
    for ( ; model != NULL ; model = model->MOS9nextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->MOS9instances ; here != NULL ; here = here->MOS9nextInstance)
        {
            if ((here-> MOS9dNode != 0) && (here-> MOS9dNode != 0))
            {
                i = here->MOS9DdPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MOS9DdStructPtr = matched ;
                here->MOS9DdPtr = matched->CSC ;
            }

            if ((here-> MOS9gNode != 0) && (here-> MOS9gNode != 0))
            {
                i = here->MOS9GgPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MOS9GgStructPtr = matched ;
                here->MOS9GgPtr = matched->CSC ;
            }

            if ((here-> MOS9sNode != 0) && (here-> MOS9sNode != 0))
            {
                i = here->MOS9SsPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MOS9SsStructPtr = matched ;
                here->MOS9SsPtr = matched->CSC ;
            }

            if ((here-> MOS9bNode != 0) && (here-> MOS9bNode != 0))
            {
                i = here->MOS9BbPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MOS9BbStructPtr = matched ;
                here->MOS9BbPtr = matched->CSC ;
            }

            if ((here-> MOS9dNodePrime != 0) && (here-> MOS9dNodePrime != 0))
            {
                i = here->MOS9DPdpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MOS9DPdpStructPtr = matched ;
                here->MOS9DPdpPtr = matched->CSC ;
            }

            if ((here-> MOS9sNodePrime != 0) && (here-> MOS9sNodePrime != 0))
            {
                i = here->MOS9SPspPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MOS9SPspStructPtr = matched ;
                here->MOS9SPspPtr = matched->CSC ;
            }

            if ((here-> MOS9dNode != 0) && (here-> MOS9dNodePrime != 0))
            {
                i = here->MOS9DdpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MOS9DdpStructPtr = matched ;
                here->MOS9DdpPtr = matched->CSC ;
            }

            if ((here-> MOS9gNode != 0) && (here-> MOS9bNode != 0))
            {
                i = here->MOS9GbPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MOS9GbStructPtr = matched ;
                here->MOS9GbPtr = matched->CSC ;
            }

            if ((here-> MOS9gNode != 0) && (here-> MOS9dNodePrime != 0))
            {
                i = here->MOS9GdpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MOS9GdpStructPtr = matched ;
                here->MOS9GdpPtr = matched->CSC ;
            }

            if ((here-> MOS9gNode != 0) && (here-> MOS9sNodePrime != 0))
            {
                i = here->MOS9GspPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MOS9GspStructPtr = matched ;
                here->MOS9GspPtr = matched->CSC ;
            }

            if ((here-> MOS9sNode != 0) && (here-> MOS9sNodePrime != 0))
            {
                i = here->MOS9SspPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MOS9SspStructPtr = matched ;
                here->MOS9SspPtr = matched->CSC ;
            }

            if ((here-> MOS9bNode != 0) && (here-> MOS9dNodePrime != 0))
            {
                i = here->MOS9BdpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MOS9BdpStructPtr = matched ;
                here->MOS9BdpPtr = matched->CSC ;
            }

            if ((here-> MOS9bNode != 0) && (here-> MOS9sNodePrime != 0))
            {
                i = here->MOS9BspPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MOS9BspStructPtr = matched ;
                here->MOS9BspPtr = matched->CSC ;
            }

            if ((here-> MOS9dNodePrime != 0) && (here-> MOS9sNodePrime != 0))
            {
                i = here->MOS9DPspPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MOS9DPspStructPtr = matched ;
                here->MOS9DPspPtr = matched->CSC ;
            }

            if ((here-> MOS9dNodePrime != 0) && (here-> MOS9dNode != 0))
            {
                i = here->MOS9DPdPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MOS9DPdStructPtr = matched ;
                here->MOS9DPdPtr = matched->CSC ;
            }

            if ((here-> MOS9bNode != 0) && (here-> MOS9gNode != 0))
            {
                i = here->MOS9BgPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MOS9BgStructPtr = matched ;
                here->MOS9BgPtr = matched->CSC ;
            }

            if ((here-> MOS9dNodePrime != 0) && (here-> MOS9gNode != 0))
            {
                i = here->MOS9DPgPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MOS9DPgStructPtr = matched ;
                here->MOS9DPgPtr = matched->CSC ;
            }

            if ((here-> MOS9sNodePrime != 0) && (here-> MOS9gNode != 0))
            {
                i = here->MOS9SPgPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MOS9SPgStructPtr = matched ;
                here->MOS9SPgPtr = matched->CSC ;
            }

            if ((here-> MOS9sNodePrime != 0) && (here-> MOS9sNode != 0))
            {
                i = here->MOS9SPsPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MOS9SPsStructPtr = matched ;
                here->MOS9SPsPtr = matched->CSC ;
            }

            if ((here-> MOS9dNodePrime != 0) && (here-> MOS9bNode != 0))
            {
                i = here->MOS9DPbPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MOS9DPbStructPtr = matched ;
                here->MOS9DPbPtr = matched->CSC ;
            }

            if ((here-> MOS9sNodePrime != 0) && (here-> MOS9bNode != 0))
            {
                i = here->MOS9SPbPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MOS9SPbStructPtr = matched ;
                here->MOS9SPbPtr = matched->CSC ;
            }

            if ((here-> MOS9sNodePrime != 0) && (here-> MOS9dNodePrime != 0))
            {
                i = here->MOS9SPdpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MOS9SPdpStructPtr = matched ;
                here->MOS9SPdpPtr = matched->CSC ;
            }

        }
    }

    return (OK) ;
}

int
MOS9bindCSCComplex (GENmodel *inModel, CKTcircuit *ckt)
{
    MOS9model *model = (MOS9model *)inModel ;
    MOS9instance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the MOS9 models */
    for ( ; model != NULL ; model = model->MOS9nextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->MOS9instances ; here != NULL ; here = here->MOS9nextInstance)
        {
            if ((here-> MOS9dNode != 0) && (here-> MOS9dNode != 0))
                here->MOS9DdPtr = here->MOS9DdStructPtr->CSC_Complex ;

            if ((here-> MOS9gNode != 0) && (here-> MOS9gNode != 0))
                here->MOS9GgPtr = here->MOS9GgStructPtr->CSC_Complex ;

            if ((here-> MOS9sNode != 0) && (here-> MOS9sNode != 0))
                here->MOS9SsPtr = here->MOS9SsStructPtr->CSC_Complex ;

            if ((here-> MOS9bNode != 0) && (here-> MOS9bNode != 0))
                here->MOS9BbPtr = here->MOS9BbStructPtr->CSC_Complex ;

            if ((here-> MOS9dNodePrime != 0) && (here-> MOS9dNodePrime != 0))
                here->MOS9DPdpPtr = here->MOS9DPdpStructPtr->CSC_Complex ;

            if ((here-> MOS9sNodePrime != 0) && (here-> MOS9sNodePrime != 0))
                here->MOS9SPspPtr = here->MOS9SPspStructPtr->CSC_Complex ;

            if ((here-> MOS9dNode != 0) && (here-> MOS9dNodePrime != 0))
                here->MOS9DdpPtr = here->MOS9DdpStructPtr->CSC_Complex ;

            if ((here-> MOS9gNode != 0) && (here-> MOS9bNode != 0))
                here->MOS9GbPtr = here->MOS9GbStructPtr->CSC_Complex ;

            if ((here-> MOS9gNode != 0) && (here-> MOS9dNodePrime != 0))
                here->MOS9GdpPtr = here->MOS9GdpStructPtr->CSC_Complex ;

            if ((here-> MOS9gNode != 0) && (here-> MOS9sNodePrime != 0))
                here->MOS9GspPtr = here->MOS9GspStructPtr->CSC_Complex ;

            if ((here-> MOS9sNode != 0) && (here-> MOS9sNodePrime != 0))
                here->MOS9SspPtr = here->MOS9SspStructPtr->CSC_Complex ;

            if ((here-> MOS9bNode != 0) && (here-> MOS9dNodePrime != 0))
                here->MOS9BdpPtr = here->MOS9BdpStructPtr->CSC_Complex ;

            if ((here-> MOS9bNode != 0) && (here-> MOS9sNodePrime != 0))
                here->MOS9BspPtr = here->MOS9BspStructPtr->CSC_Complex ;

            if ((here-> MOS9dNodePrime != 0) && (here-> MOS9sNodePrime != 0))
                here->MOS9DPspPtr = here->MOS9DPspStructPtr->CSC_Complex ;

            if ((here-> MOS9dNodePrime != 0) && (here-> MOS9dNode != 0))
                here->MOS9DPdPtr = here->MOS9DPdStructPtr->CSC_Complex ;

            if ((here-> MOS9bNode != 0) && (here-> MOS9gNode != 0))
                here->MOS9BgPtr = here->MOS9BgStructPtr->CSC_Complex ;

            if ((here-> MOS9dNodePrime != 0) && (here-> MOS9gNode != 0))
                here->MOS9DPgPtr = here->MOS9DPgStructPtr->CSC_Complex ;

            if ((here-> MOS9sNodePrime != 0) && (here-> MOS9gNode != 0))
                here->MOS9SPgPtr = here->MOS9SPgStructPtr->CSC_Complex ;

            if ((here-> MOS9sNodePrime != 0) && (here-> MOS9sNode != 0))
                here->MOS9SPsPtr = here->MOS9SPsStructPtr->CSC_Complex ;

            if ((here-> MOS9dNodePrime != 0) && (here-> MOS9bNode != 0))
                here->MOS9DPbPtr = here->MOS9DPbStructPtr->CSC_Complex ;

            if ((here-> MOS9sNodePrime != 0) && (here-> MOS9bNode != 0))
                here->MOS9SPbPtr = here->MOS9SPbStructPtr->CSC_Complex ;

            if ((here-> MOS9sNodePrime != 0) && (here-> MOS9dNodePrime != 0))
                here->MOS9SPdpPtr = here->MOS9SPdpStructPtr->CSC_Complex ;

        }
    }

    return (OK) ;
}

int
MOS9bindCSCComplexToReal (GENmodel *inModel, CKTcircuit *ckt)
{
    MOS9model *model = (MOS9model *)inModel ;
    MOS9instance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the MOS9 models */
    for ( ; model != NULL ; model = model->MOS9nextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->MOS9instances ; here != NULL ; here = here->MOS9nextInstance)
        {
            if ((here-> MOS9dNode != 0) && (here-> MOS9dNode != 0))
                here->MOS9DdPtr = here->MOS9DdStructPtr->CSC ;

            if ((here-> MOS9gNode != 0) && (here-> MOS9gNode != 0))
                here->MOS9GgPtr = here->MOS9GgStructPtr->CSC ;

            if ((here-> MOS9sNode != 0) && (here-> MOS9sNode != 0))
                here->MOS9SsPtr = here->MOS9SsStructPtr->CSC ;

            if ((here-> MOS9bNode != 0) && (here-> MOS9bNode != 0))
                here->MOS9BbPtr = here->MOS9BbStructPtr->CSC ;

            if ((here-> MOS9dNodePrime != 0) && (here-> MOS9dNodePrime != 0))
                here->MOS9DPdpPtr = here->MOS9DPdpStructPtr->CSC ;

            if ((here-> MOS9sNodePrime != 0) && (here-> MOS9sNodePrime != 0))
                here->MOS9SPspPtr = here->MOS9SPspStructPtr->CSC ;

            if ((here-> MOS9dNode != 0) && (here-> MOS9dNodePrime != 0))
                here->MOS9DdpPtr = here->MOS9DdpStructPtr->CSC ;

            if ((here-> MOS9gNode != 0) && (here-> MOS9bNode != 0))
                here->MOS9GbPtr = here->MOS9GbStructPtr->CSC ;

            if ((here-> MOS9gNode != 0) && (here-> MOS9dNodePrime != 0))
                here->MOS9GdpPtr = here->MOS9GdpStructPtr->CSC ;

            if ((here-> MOS9gNode != 0) && (here-> MOS9sNodePrime != 0))
                here->MOS9GspPtr = here->MOS9GspStructPtr->CSC ;

            if ((here-> MOS9sNode != 0) && (here-> MOS9sNodePrime != 0))
                here->MOS9SspPtr = here->MOS9SspStructPtr->CSC ;

            if ((here-> MOS9bNode != 0) && (here-> MOS9dNodePrime != 0))
                here->MOS9BdpPtr = here->MOS9BdpStructPtr->CSC ;

            if ((here-> MOS9bNode != 0) && (here-> MOS9sNodePrime != 0))
                here->MOS9BspPtr = here->MOS9BspStructPtr->CSC ;

            if ((here-> MOS9dNodePrime != 0) && (here-> MOS9sNodePrime != 0))
                here->MOS9DPspPtr = here->MOS9DPspStructPtr->CSC ;

            if ((here-> MOS9dNodePrime != 0) && (here-> MOS9dNode != 0))
                here->MOS9DPdPtr = here->MOS9DPdStructPtr->CSC ;

            if ((here-> MOS9bNode != 0) && (here-> MOS9gNode != 0))
                here->MOS9BgPtr = here->MOS9BgStructPtr->CSC ;

            if ((here-> MOS9dNodePrime != 0) && (here-> MOS9gNode != 0))
                here->MOS9DPgPtr = here->MOS9DPgStructPtr->CSC ;

            if ((here-> MOS9sNodePrime != 0) && (here-> MOS9gNode != 0))
                here->MOS9SPgPtr = here->MOS9SPgStructPtr->CSC ;

            if ((here-> MOS9sNodePrime != 0) && (here-> MOS9sNode != 0))
                here->MOS9SPsPtr = here->MOS9SPsStructPtr->CSC ;

            if ((here-> MOS9dNodePrime != 0) && (here-> MOS9bNode != 0))
                here->MOS9DPbPtr = here->MOS9DPbStructPtr->CSC ;

            if ((here-> MOS9sNodePrime != 0) && (here-> MOS9bNode != 0))
                here->MOS9SPbPtr = here->MOS9SPbStructPtr->CSC ;

            if ((here-> MOS9sNodePrime != 0) && (here-> MOS9dNodePrime != 0))
                here->MOS9SPdpPtr = here->MOS9SPdpStructPtr->CSC ;

        }
    }

    return (OK) ;
}
