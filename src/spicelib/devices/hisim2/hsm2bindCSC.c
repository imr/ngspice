/**********
Author: 2013 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "hsm2def.h"
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
HSM2bindCSC (GENmodel *inModel, CKTcircuit *ckt)
{
    HSM2model *model = (HSM2model *)inModel ;
    HSM2instance *here ;
    double *i ;
    BindElement *matched, *BindStruct ;
    size_t nz ;

    BindStruct = ckt->CKTmatrix->CKTbindStruct ;
    nz = (size_t)ckt->CKTmatrix->CKTklunz ;

    /* loop through all the HSM2 models */
    for ( ; model != NULL ; model = model->HSM2nextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->HSM2instances ; here != NULL ; here = here->HSM2nextInstance)
        {
            if ((here-> HSM2dNodePrime != 0) && (here-> HSM2bNodePrime != 0))
            {
                i = here->HSM2DPbpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->HSM2DPbpStructPtr = matched ;
                here->HSM2DPbpPtr = matched->CSC ;
            }

            if ((here-> HSM2sNodePrime != 0) && (here-> HSM2bNodePrime != 0))
            {
                i = here->HSM2SPbpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->HSM2SPbpStructPtr = matched ;
                here->HSM2SPbpPtr = matched->CSC ;
            }

            if ((here-> HSM2gNodePrime != 0) && (here-> HSM2bNodePrime != 0))
            {
                i = here->HSM2GPbpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->HSM2GPbpStructPtr = matched ;
                here->HSM2GPbpPtr = matched->CSC ;
            }

            if ((here-> HSM2bNodePrime != 0) && (here-> HSM2dNodePrime != 0))
            {
                i = here->HSM2BPdpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->HSM2BPdpStructPtr = matched ;
                here->HSM2BPdpPtr = matched->CSC ;
            }

            if ((here-> HSM2bNodePrime != 0) && (here-> HSM2sNodePrime != 0))
            {
                i = here->HSM2BPspPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->HSM2BPspStructPtr = matched ;
                here->HSM2BPspPtr = matched->CSC ;
            }

            if ((here-> HSM2bNodePrime != 0) && (here-> HSM2gNodePrime != 0))
            {
                i = here->HSM2BPgpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->HSM2BPgpStructPtr = matched ;
                here->HSM2BPgpPtr = matched->CSC ;
            }

            if ((here-> HSM2bNodePrime != 0) && (here-> HSM2bNodePrime != 0))
            {
                i = here->HSM2BPbpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->HSM2BPbpStructPtr = matched ;
                here->HSM2BPbpPtr = matched->CSC ;
            }

            if ((here-> HSM2dNode != 0) && (here-> HSM2dNode != 0))
            {
                i = here->HSM2DdPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->HSM2DdStructPtr = matched ;
                here->HSM2DdPtr = matched->CSC ;
            }

            if ((here-> HSM2gNodePrime != 0) && (here-> HSM2gNodePrime != 0))
            {
                i = here->HSM2GPgpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->HSM2GPgpStructPtr = matched ;
                here->HSM2GPgpPtr = matched->CSC ;
            }

            if ((here-> HSM2sNode != 0) && (here-> HSM2sNode != 0))
            {
                i = here->HSM2SsPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->HSM2SsStructPtr = matched ;
                here->HSM2SsPtr = matched->CSC ;
            }

            if ((here-> HSM2dNodePrime != 0) && (here-> HSM2dNodePrime != 0))
            {
                i = here->HSM2DPdpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->HSM2DPdpStructPtr = matched ;
                here->HSM2DPdpPtr = matched->CSC ;
            }

            if ((here-> HSM2sNodePrime != 0) && (here-> HSM2sNodePrime != 0))
            {
                i = here->HSM2SPspPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->HSM2SPspStructPtr = matched ;
                here->HSM2SPspPtr = matched->CSC ;
            }

            if ((here-> HSM2dNode != 0) && (here-> HSM2dNodePrime != 0))
            {
                i = here->HSM2DdpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->HSM2DdpStructPtr = matched ;
                here->HSM2DdpPtr = matched->CSC ;
            }

            if ((here-> HSM2gNodePrime != 0) && (here-> HSM2dNodePrime != 0))
            {
                i = here->HSM2GPdpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->HSM2GPdpStructPtr = matched ;
                here->HSM2GPdpPtr = matched->CSC ;
            }

            if ((here-> HSM2gNodePrime != 0) && (here-> HSM2sNodePrime != 0))
            {
                i = here->HSM2GPspPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->HSM2GPspStructPtr = matched ;
                here->HSM2GPspPtr = matched->CSC ;
            }

            if ((here-> HSM2sNode != 0) && (here-> HSM2sNodePrime != 0))
            {
                i = here->HSM2SspPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->HSM2SspStructPtr = matched ;
                here->HSM2SspPtr = matched->CSC ;
            }

            if ((here-> HSM2dNodePrime != 0) && (here-> HSM2sNodePrime != 0))
            {
                i = here->HSM2DPspPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->HSM2DPspStructPtr = matched ;
                here->HSM2DPspPtr = matched->CSC ;
            }

            if ((here-> HSM2dNodePrime != 0) && (here-> HSM2dNode != 0))
            {
                i = here->HSM2DPdPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->HSM2DPdStructPtr = matched ;
                here->HSM2DPdPtr = matched->CSC ;
            }

            if ((here-> HSM2dNodePrime != 0) && (here-> HSM2gNodePrime != 0))
            {
                i = here->HSM2DPgpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->HSM2DPgpStructPtr = matched ;
                here->HSM2DPgpPtr = matched->CSC ;
            }

            if ((here-> HSM2sNodePrime != 0) && (here-> HSM2gNodePrime != 0))
            {
                i = here->HSM2SPgpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->HSM2SPgpStructPtr = matched ;
                here->HSM2SPgpPtr = matched->CSC ;
            }

            if ((here-> HSM2sNodePrime != 0) && (here-> HSM2sNode != 0))
            {
                i = here->HSM2SPsPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->HSM2SPsStructPtr = matched ;
                here->HSM2SPsPtr = matched->CSC ;
            }

            if ((here-> HSM2sNodePrime != 0) && (here-> HSM2dNodePrime != 0))
            {
                i = here->HSM2SPdpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->HSM2SPdpStructPtr = matched ;
                here->HSM2SPdpPtr = matched->CSC ;
            }

            if (here->HSM2_corg == 1)
            {
                if ((here-> HSM2gNode != 0) && (here-> HSM2gNode != 0))
                {
                    i = here->HSM2GgPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->HSM2GgStructPtr = matched ;
                    here->HSM2GgPtr = matched->CSC ;
                }

                if ((here-> HSM2gNode != 0) && (here-> HSM2gNodePrime != 0))
                {
                    i = here->HSM2GgpPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->HSM2GgpStructPtr = matched ;
                    here->HSM2GgpPtr = matched->CSC ;
                }

                if ((here-> HSM2gNodePrime != 0) && (here-> HSM2gNode != 0))
                {
                    i = here->HSM2GPgPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->HSM2GPgStructPtr = matched ;
                    here->HSM2GPgPtr = matched->CSC ;
                }

                if ((here-> HSM2gNode != 0) && (here-> HSM2dNodePrime != 0))
                {
                    i = here->HSM2GdpPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->HSM2GdpStructPtr = matched ;
                    here->HSM2GdpPtr = matched->CSC ;
                }

                if ((here-> HSM2gNode != 0) && (here-> HSM2sNodePrime != 0))
                {
                    i = here->HSM2GspPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->HSM2GspStructPtr = matched ;
                    here->HSM2GspPtr = matched->CSC ;
                }

                if ((here-> HSM2gNode != 0) && (here-> HSM2bNodePrime != 0))
                {
                    i = here->HSM2GbpPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->HSM2GbpStructPtr = matched ;
                    here->HSM2GbpPtr = matched->CSC ;
                }

            }
            if (here->HSM2_corbnet == 1)
            {
                if ((here-> HSM2dNodePrime != 0) && (here-> HSM2dbNode != 0))
                {
                    i = here->HSM2DPdbPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->HSM2DPdbStructPtr = matched ;
                    here->HSM2DPdbPtr = matched->CSC ;
                }

                if ((here-> HSM2sNodePrime != 0) && (here-> HSM2sbNode != 0))
                {
                    i = here->HSM2SPsbPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->HSM2SPsbStructPtr = matched ;
                    here->HSM2SPsbPtr = matched->CSC ;
                }

                if ((here-> HSM2dbNode != 0) && (here-> HSM2dNodePrime != 0))
                {
                    i = here->HSM2DBdpPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->HSM2DBdpStructPtr = matched ;
                    here->HSM2DBdpPtr = matched->CSC ;
                }

                if ((here-> HSM2dbNode != 0) && (here-> HSM2dbNode != 0))
                {
                    i = here->HSM2DBdbPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->HSM2DBdbStructPtr = matched ;
                    here->HSM2DBdbPtr = matched->CSC ;
                }

                if ((here-> HSM2dbNode != 0) && (here-> HSM2bNodePrime != 0))
                {
                    i = here->HSM2DBbpPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->HSM2DBbpStructPtr = matched ;
                    here->HSM2DBbpPtr = matched->CSC ;
                }

                if ((here-> HSM2dbNode != 0) && (here-> HSM2bNode != 0))
                {
                    i = here->HSM2DBbPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->HSM2DBbStructPtr = matched ;
                    here->HSM2DBbPtr = matched->CSC ;
                }

                if ((here-> HSM2bNodePrime != 0) && (here-> HSM2dbNode != 0))
                {
                    i = here->HSM2BPdbPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->HSM2BPdbStructPtr = matched ;
                    here->HSM2BPdbPtr = matched->CSC ;
                }

                if ((here-> HSM2bNodePrime != 0) && (here-> HSM2bNode != 0))
                {
                    i = here->HSM2BPbPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->HSM2BPbStructPtr = matched ;
                    here->HSM2BPbPtr = matched->CSC ;
                }

                if ((here-> HSM2bNodePrime != 0) && (here-> HSM2sbNode != 0))
                {
                    i = here->HSM2BPsbPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->HSM2BPsbStructPtr = matched ;
                    here->HSM2BPsbPtr = matched->CSC ;
                }

                if ((here-> HSM2sbNode != 0) && (here-> HSM2sNodePrime != 0))
                {
                    i = here->HSM2SBspPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->HSM2SBspStructPtr = matched ;
                    here->HSM2SBspPtr = matched->CSC ;
                }

                if ((here-> HSM2sbNode != 0) && (here-> HSM2bNodePrime != 0))
                {
                    i = here->HSM2SBbpPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->HSM2SBbpStructPtr = matched ;
                    here->HSM2SBbpPtr = matched->CSC ;
                }

                if ((here-> HSM2sbNode != 0) && (here-> HSM2bNode != 0))
                {
                    i = here->HSM2SBbPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->HSM2SBbStructPtr = matched ;
                    here->HSM2SBbPtr = matched->CSC ;
                }

                if ((here-> HSM2sbNode != 0) && (here-> HSM2sbNode != 0))
                {
                    i = here->HSM2SBsbPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->HSM2SBsbStructPtr = matched ;
                    here->HSM2SBsbPtr = matched->CSC ;
                }

                if ((here-> HSM2bNode != 0) && (here-> HSM2dbNode != 0))
                {
                    i = here->HSM2BdbPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->HSM2BdbStructPtr = matched ;
                    here->HSM2BdbPtr = matched->CSC ;
                }

                if ((here-> HSM2bNode != 0) && (here-> HSM2bNodePrime != 0))
                {
                    i = here->HSM2BbpPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->HSM2BbpStructPtr = matched ;
                    here->HSM2BbpPtr = matched->CSC ;
                }

                if ((here-> HSM2bNode != 0) && (here-> HSM2sbNode != 0))
                {
                    i = here->HSM2BsbPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->HSM2BsbStructPtr = matched ;
                    here->HSM2BsbPtr = matched->CSC ;
                }

                if ((here-> HSM2bNode != 0) && (here-> HSM2bNode != 0))
                {
                    i = here->HSM2BbPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->HSM2BbStructPtr = matched ;
                    here->HSM2BbPtr = matched->CSC ;
                }

            }
        }
    }

    return (OK) ;
}

int
HSM2bindCSCComplex (GENmodel *inModel, CKTcircuit *ckt)
{
    HSM2model *model = (HSM2model *)inModel ;
    HSM2instance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the HSM2 models */
    for ( ; model != NULL ; model = model->HSM2nextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->HSM2instances ; here != NULL ; here = here->HSM2nextInstance)
        {
            if ((here-> HSM2dNodePrime != 0) && (here-> HSM2bNodePrime != 0))
                here->HSM2DPbpPtr = here->HSM2DPbpStructPtr->CSC_Complex ;

            if ((here-> HSM2sNodePrime != 0) && (here-> HSM2bNodePrime != 0))
                here->HSM2SPbpPtr = here->HSM2SPbpStructPtr->CSC_Complex ;

            if ((here-> HSM2gNodePrime != 0) && (here-> HSM2bNodePrime != 0))
                here->HSM2GPbpPtr = here->HSM2GPbpStructPtr->CSC_Complex ;

            if ((here-> HSM2bNodePrime != 0) && (here-> HSM2dNodePrime != 0))
                here->HSM2BPdpPtr = here->HSM2BPdpStructPtr->CSC_Complex ;

            if ((here-> HSM2bNodePrime != 0) && (here-> HSM2sNodePrime != 0))
                here->HSM2BPspPtr = here->HSM2BPspStructPtr->CSC_Complex ;

            if ((here-> HSM2bNodePrime != 0) && (here-> HSM2gNodePrime != 0))
                here->HSM2BPgpPtr = here->HSM2BPgpStructPtr->CSC_Complex ;

            if ((here-> HSM2bNodePrime != 0) && (here-> HSM2bNodePrime != 0))
                here->HSM2BPbpPtr = here->HSM2BPbpStructPtr->CSC_Complex ;

            if ((here-> HSM2dNode != 0) && (here-> HSM2dNode != 0))
                here->HSM2DdPtr = here->HSM2DdStructPtr->CSC_Complex ;

            if ((here-> HSM2gNodePrime != 0) && (here-> HSM2gNodePrime != 0))
                here->HSM2GPgpPtr = here->HSM2GPgpStructPtr->CSC_Complex ;

            if ((here-> HSM2sNode != 0) && (here-> HSM2sNode != 0))
                here->HSM2SsPtr = here->HSM2SsStructPtr->CSC_Complex ;

            if ((here-> HSM2dNodePrime != 0) && (here-> HSM2dNodePrime != 0))
                here->HSM2DPdpPtr = here->HSM2DPdpStructPtr->CSC_Complex ;

            if ((here-> HSM2sNodePrime != 0) && (here-> HSM2sNodePrime != 0))
                here->HSM2SPspPtr = here->HSM2SPspStructPtr->CSC_Complex ;

            if ((here-> HSM2dNode != 0) && (here-> HSM2dNodePrime != 0))
                here->HSM2DdpPtr = here->HSM2DdpStructPtr->CSC_Complex ;

            if ((here-> HSM2gNodePrime != 0) && (here-> HSM2dNodePrime != 0))
                here->HSM2GPdpPtr = here->HSM2GPdpStructPtr->CSC_Complex ;

            if ((here-> HSM2gNodePrime != 0) && (here-> HSM2sNodePrime != 0))
                here->HSM2GPspPtr = here->HSM2GPspStructPtr->CSC_Complex ;

            if ((here-> HSM2sNode != 0) && (here-> HSM2sNodePrime != 0))
                here->HSM2SspPtr = here->HSM2SspStructPtr->CSC_Complex ;

            if ((here-> HSM2dNodePrime != 0) && (here-> HSM2sNodePrime != 0))
                here->HSM2DPspPtr = here->HSM2DPspStructPtr->CSC_Complex ;

            if ((here-> HSM2dNodePrime != 0) && (here-> HSM2dNode != 0))
                here->HSM2DPdPtr = here->HSM2DPdStructPtr->CSC_Complex ;

            if ((here-> HSM2dNodePrime != 0) && (here-> HSM2gNodePrime != 0))
                here->HSM2DPgpPtr = here->HSM2DPgpStructPtr->CSC_Complex ;

            if ((here-> HSM2sNodePrime != 0) && (here-> HSM2gNodePrime != 0))
                here->HSM2SPgpPtr = here->HSM2SPgpStructPtr->CSC_Complex ;

            if ((here-> HSM2sNodePrime != 0) && (here-> HSM2sNode != 0))
                here->HSM2SPsPtr = here->HSM2SPsStructPtr->CSC_Complex ;

            if ((here-> HSM2sNodePrime != 0) && (here-> HSM2dNodePrime != 0))
                here->HSM2SPdpPtr = here->HSM2SPdpStructPtr->CSC_Complex ;

            if (here->HSM2_corg == 1)
            {
                if ((here-> HSM2gNode != 0) && (here-> HSM2gNode != 0))
                    here->HSM2GgPtr = here->HSM2GgStructPtr->CSC_Complex ;

                if ((here-> HSM2gNode != 0) && (here-> HSM2gNodePrime != 0))
                    here->HSM2GgpPtr = here->HSM2GgpStructPtr->CSC_Complex ;

                if ((here-> HSM2gNodePrime != 0) && (here-> HSM2gNode != 0))
                    here->HSM2GPgPtr = here->HSM2GPgStructPtr->CSC_Complex ;

                if ((here-> HSM2gNode != 0) && (here-> HSM2dNodePrime != 0))
                    here->HSM2GdpPtr = here->HSM2GdpStructPtr->CSC_Complex ;

                if ((here-> HSM2gNode != 0) && (here-> HSM2sNodePrime != 0))
                    here->HSM2GspPtr = here->HSM2GspStructPtr->CSC_Complex ;

                if ((here-> HSM2gNode != 0) && (here-> HSM2bNodePrime != 0))
                    here->HSM2GbpPtr = here->HSM2GbpStructPtr->CSC_Complex ;

            }
            if (here->HSM2_corbnet == 1)
            {
                if ((here-> HSM2dNodePrime != 0) && (here-> HSM2dbNode != 0))
                    here->HSM2DPdbPtr = here->HSM2DPdbStructPtr->CSC_Complex ;

                if ((here-> HSM2sNodePrime != 0) && (here-> HSM2sbNode != 0))
                    here->HSM2SPsbPtr = here->HSM2SPsbStructPtr->CSC_Complex ;

                if ((here-> HSM2dbNode != 0) && (here-> HSM2dNodePrime != 0))
                    here->HSM2DBdpPtr = here->HSM2DBdpStructPtr->CSC_Complex ;

                if ((here-> HSM2dbNode != 0) && (here-> HSM2dbNode != 0))
                    here->HSM2DBdbPtr = here->HSM2DBdbStructPtr->CSC_Complex ;

                if ((here-> HSM2dbNode != 0) && (here-> HSM2bNodePrime != 0))
                    here->HSM2DBbpPtr = here->HSM2DBbpStructPtr->CSC_Complex ;

                if ((here-> HSM2dbNode != 0) && (here-> HSM2bNode != 0))
                    here->HSM2DBbPtr = here->HSM2DBbStructPtr->CSC_Complex ;

                if ((here-> HSM2bNodePrime != 0) && (here-> HSM2dbNode != 0))
                    here->HSM2BPdbPtr = here->HSM2BPdbStructPtr->CSC_Complex ;

                if ((here-> HSM2bNodePrime != 0) && (here-> HSM2bNode != 0))
                    here->HSM2BPbPtr = here->HSM2BPbStructPtr->CSC_Complex ;

                if ((here-> HSM2bNodePrime != 0) && (here-> HSM2sbNode != 0))
                    here->HSM2BPsbPtr = here->HSM2BPsbStructPtr->CSC_Complex ;

                if ((here-> HSM2sbNode != 0) && (here-> HSM2sNodePrime != 0))
                    here->HSM2SBspPtr = here->HSM2SBspStructPtr->CSC_Complex ;

                if ((here-> HSM2sbNode != 0) && (here-> HSM2bNodePrime != 0))
                    here->HSM2SBbpPtr = here->HSM2SBbpStructPtr->CSC_Complex ;

                if ((here-> HSM2sbNode != 0) && (here-> HSM2bNode != 0))
                    here->HSM2SBbPtr = here->HSM2SBbStructPtr->CSC_Complex ;

                if ((here-> HSM2sbNode != 0) && (here-> HSM2sbNode != 0))
                    here->HSM2SBsbPtr = here->HSM2SBsbStructPtr->CSC_Complex ;

                if ((here-> HSM2bNode != 0) && (here-> HSM2dbNode != 0))
                    here->HSM2BdbPtr = here->HSM2BdbStructPtr->CSC_Complex ;

                if ((here-> HSM2bNode != 0) && (here-> HSM2bNodePrime != 0))
                    here->HSM2BbpPtr = here->HSM2BbpStructPtr->CSC_Complex ;

                if ((here-> HSM2bNode != 0) && (here-> HSM2sbNode != 0))
                    here->HSM2BsbPtr = here->HSM2BsbStructPtr->CSC_Complex ;

                if ((here-> HSM2bNode != 0) && (here-> HSM2bNode != 0))
                    here->HSM2BbPtr = here->HSM2BbStructPtr->CSC_Complex ;

            }
        }
    }

    return (OK) ;
}

int
HSM2bindCSCComplexToReal (GENmodel *inModel, CKTcircuit *ckt)
{
    HSM2model *model = (HSM2model *)inModel ;
    HSM2instance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the HSM2 models */
    for ( ; model != NULL ; model = model->HSM2nextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->HSM2instances ; here != NULL ; here = here->HSM2nextInstance)
        {
            if ((here-> HSM2dNodePrime != 0) && (here-> HSM2bNodePrime != 0))
                here->HSM2DPbpPtr = here->HSM2DPbpStructPtr->CSC ;

            if ((here-> HSM2sNodePrime != 0) && (here-> HSM2bNodePrime != 0))
                here->HSM2SPbpPtr = here->HSM2SPbpStructPtr->CSC ;

            if ((here-> HSM2gNodePrime != 0) && (here-> HSM2bNodePrime != 0))
                here->HSM2GPbpPtr = here->HSM2GPbpStructPtr->CSC ;

            if ((here-> HSM2bNodePrime != 0) && (here-> HSM2dNodePrime != 0))
                here->HSM2BPdpPtr = here->HSM2BPdpStructPtr->CSC ;

            if ((here-> HSM2bNodePrime != 0) && (here-> HSM2sNodePrime != 0))
                here->HSM2BPspPtr = here->HSM2BPspStructPtr->CSC ;

            if ((here-> HSM2bNodePrime != 0) && (here-> HSM2gNodePrime != 0))
                here->HSM2BPgpPtr = here->HSM2BPgpStructPtr->CSC ;

            if ((here-> HSM2bNodePrime != 0) && (here-> HSM2bNodePrime != 0))
                here->HSM2BPbpPtr = here->HSM2BPbpStructPtr->CSC ;

            if ((here-> HSM2dNode != 0) && (here-> HSM2dNode != 0))
                here->HSM2DdPtr = here->HSM2DdStructPtr->CSC ;

            if ((here-> HSM2gNodePrime != 0) && (here-> HSM2gNodePrime != 0))
                here->HSM2GPgpPtr = here->HSM2GPgpStructPtr->CSC ;

            if ((here-> HSM2sNode != 0) && (here-> HSM2sNode != 0))
                here->HSM2SsPtr = here->HSM2SsStructPtr->CSC ;

            if ((here-> HSM2dNodePrime != 0) && (here-> HSM2dNodePrime != 0))
                here->HSM2DPdpPtr = here->HSM2DPdpStructPtr->CSC ;

            if ((here-> HSM2sNodePrime != 0) && (here-> HSM2sNodePrime != 0))
                here->HSM2SPspPtr = here->HSM2SPspStructPtr->CSC ;

            if ((here-> HSM2dNode != 0) && (here-> HSM2dNodePrime != 0))
                here->HSM2DdpPtr = here->HSM2DdpStructPtr->CSC ;

            if ((here-> HSM2gNodePrime != 0) && (here-> HSM2dNodePrime != 0))
                here->HSM2GPdpPtr = here->HSM2GPdpStructPtr->CSC ;

            if ((here-> HSM2gNodePrime != 0) && (here-> HSM2sNodePrime != 0))
                here->HSM2GPspPtr = here->HSM2GPspStructPtr->CSC ;

            if ((here-> HSM2sNode != 0) && (here-> HSM2sNodePrime != 0))
                here->HSM2SspPtr = here->HSM2SspStructPtr->CSC ;

            if ((here-> HSM2dNodePrime != 0) && (here-> HSM2sNodePrime != 0))
                here->HSM2DPspPtr = here->HSM2DPspStructPtr->CSC ;

            if ((here-> HSM2dNodePrime != 0) && (here-> HSM2dNode != 0))
                here->HSM2DPdPtr = here->HSM2DPdStructPtr->CSC ;

            if ((here-> HSM2dNodePrime != 0) && (here-> HSM2gNodePrime != 0))
                here->HSM2DPgpPtr = here->HSM2DPgpStructPtr->CSC ;

            if ((here-> HSM2sNodePrime != 0) && (here-> HSM2gNodePrime != 0))
                here->HSM2SPgpPtr = here->HSM2SPgpStructPtr->CSC ;

            if ((here-> HSM2sNodePrime != 0) && (here-> HSM2sNode != 0))
                here->HSM2SPsPtr = here->HSM2SPsStructPtr->CSC ;

            if ((here-> HSM2sNodePrime != 0) && (here-> HSM2dNodePrime != 0))
                here->HSM2SPdpPtr = here->HSM2SPdpStructPtr->CSC ;

            if (here->HSM2_corg == 1)
            {
                if ((here-> HSM2gNode != 0) && (here-> HSM2gNode != 0))
                    here->HSM2GgPtr = here->HSM2GgStructPtr->CSC ;

                if ((here-> HSM2gNode != 0) && (here-> HSM2gNodePrime != 0))
                    here->HSM2GgpPtr = here->HSM2GgpStructPtr->CSC ;

                if ((here-> HSM2gNodePrime != 0) && (here-> HSM2gNode != 0))
                    here->HSM2GPgPtr = here->HSM2GPgStructPtr->CSC ;

                if ((here-> HSM2gNode != 0) && (here-> HSM2dNodePrime != 0))
                    here->HSM2GdpPtr = here->HSM2GdpStructPtr->CSC ;

                if ((here-> HSM2gNode != 0) && (here-> HSM2sNodePrime != 0))
                    here->HSM2GspPtr = here->HSM2GspStructPtr->CSC ;

                if ((here-> HSM2gNode != 0) && (here-> HSM2bNodePrime != 0))
                    here->HSM2GbpPtr = here->HSM2GbpStructPtr->CSC ;

            }
            if (here->HSM2_corbnet == 1)
            {
                if ((here-> HSM2dNodePrime != 0) && (here-> HSM2dbNode != 0))
                    here->HSM2DPdbPtr = here->HSM2DPdbStructPtr->CSC ;

                if ((here-> HSM2sNodePrime != 0) && (here-> HSM2sbNode != 0))
                    here->HSM2SPsbPtr = here->HSM2SPsbStructPtr->CSC ;

                if ((here-> HSM2dbNode != 0) && (here-> HSM2dNodePrime != 0))
                    here->HSM2DBdpPtr = here->HSM2DBdpStructPtr->CSC ;

                if ((here-> HSM2dbNode != 0) && (here-> HSM2dbNode != 0))
                    here->HSM2DBdbPtr = here->HSM2DBdbStructPtr->CSC ;

                if ((here-> HSM2dbNode != 0) && (here-> HSM2bNodePrime != 0))
                    here->HSM2DBbpPtr = here->HSM2DBbpStructPtr->CSC ;

                if ((here-> HSM2dbNode != 0) && (here-> HSM2bNode != 0))
                    here->HSM2DBbPtr = here->HSM2DBbStructPtr->CSC ;

                if ((here-> HSM2bNodePrime != 0) && (here-> HSM2dbNode != 0))
                    here->HSM2BPdbPtr = here->HSM2BPdbStructPtr->CSC ;

                if ((here-> HSM2bNodePrime != 0) && (here-> HSM2bNode != 0))
                    here->HSM2BPbPtr = here->HSM2BPbStructPtr->CSC ;

                if ((here-> HSM2bNodePrime != 0) && (here-> HSM2sbNode != 0))
                    here->HSM2BPsbPtr = here->HSM2BPsbStructPtr->CSC ;

                if ((here-> HSM2sbNode != 0) && (here-> HSM2sNodePrime != 0))
                    here->HSM2SBspPtr = here->HSM2SBspStructPtr->CSC ;

                if ((here-> HSM2sbNode != 0) && (here-> HSM2bNodePrime != 0))
                    here->HSM2SBbpPtr = here->HSM2SBbpStructPtr->CSC ;

                if ((here-> HSM2sbNode != 0) && (here-> HSM2bNode != 0))
                    here->HSM2SBbPtr = here->HSM2SBbStructPtr->CSC ;

                if ((here-> HSM2sbNode != 0) && (here-> HSM2sbNode != 0))
                    here->HSM2SBsbPtr = here->HSM2SBsbStructPtr->CSC ;

                if ((here-> HSM2bNode != 0) && (here-> HSM2dbNode != 0))
                    here->HSM2BdbPtr = here->HSM2BdbStructPtr->CSC ;

                if ((here-> HSM2bNode != 0) && (here-> HSM2bNodePrime != 0))
                    here->HSM2BbpPtr = here->HSM2BbpStructPtr->CSC ;

                if ((here-> HSM2bNode != 0) && (here-> HSM2sbNode != 0))
                    here->HSM2BsbPtr = here->HSM2BsbStructPtr->CSC ;

                if ((here-> HSM2bNode != 0) && (here-> HSM2bNode != 0))
                    here->HSM2BbPtr = here->HSM2BbStructPtr->CSC ;

            }
        }
    }

    return (OK) ;
}
