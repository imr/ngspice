/**********
Author: 2013 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "bsim4v6def.h"
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
BSIM4v6bindCSC (GENmodel *inModel, CKTcircuit *ckt)
{
    BSIM4v6model *model = (BSIM4v6model *)inModel ;
    BSIM4v6instance *here ;
    double *i ;
    BindElement *matched, *BindStruct ;
    size_t nz ;

    BindStruct = ckt->CKTmatrix->CKTbindStruct ;
    nz = (size_t)ckt->CKTmatrix->CKTklunz ;

    /* loop through all the BSIM4v6 models */
    for ( ; model != NULL ; model = model->BSIM4v6nextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->BSIM4v6instances ; here != NULL ; here = here->BSIM4v6nextInstance)
        {
            if ((here-> BSIM4v6dNodePrime != 0) && (here-> BSIM4v6bNodePrime != 0))
            {
                i = here->BSIM4v6DPbpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4v6DPbpStructPtr = matched ;
                here->BSIM4v6DPbpPtr = matched->CSC ;
            }

            if ((here-> BSIM4v6gNodePrime != 0) && (here-> BSIM4v6bNodePrime != 0))
            {
                i = here->BSIM4v6GPbpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4v6GPbpStructPtr = matched ;
                here->BSIM4v6GPbpPtr = matched->CSC ;
            }

            if ((here-> BSIM4v6sNodePrime != 0) && (here-> BSIM4v6bNodePrime != 0))
            {
                i = here->BSIM4v6SPbpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4v6SPbpStructPtr = matched ;
                here->BSIM4v6SPbpPtr = matched->CSC ;
            }

            if ((here-> BSIM4v6bNodePrime != 0) && (here-> BSIM4v6dNodePrime != 0))
            {
                i = here->BSIM4v6BPdpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4v6BPdpStructPtr = matched ;
                here->BSIM4v6BPdpPtr = matched->CSC ;
            }

            if ((here-> BSIM4v6bNodePrime != 0) && (here-> BSIM4v6gNodePrime != 0))
            {
                i = here->BSIM4v6BPgpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4v6BPgpStructPtr = matched ;
                here->BSIM4v6BPgpPtr = matched->CSC ;
            }

            if ((here-> BSIM4v6bNodePrime != 0) && (here-> BSIM4v6sNodePrime != 0))
            {
                i = here->BSIM4v6BPspPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4v6BPspStructPtr = matched ;
                here->BSIM4v6BPspPtr = matched->CSC ;
            }

            if ((here-> BSIM4v6bNodePrime != 0) && (here-> BSIM4v6bNodePrime != 0))
            {
                i = here->BSIM4v6BPbpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4v6BPbpStructPtr = matched ;
                here->BSIM4v6BPbpPtr = matched->CSC ;
            }

            if ((here-> BSIM4v6dNode != 0) && (here-> BSIM4v6dNode != 0))
            {
                i = here->BSIM4v6DdPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4v6DdStructPtr = matched ;
                here->BSIM4v6DdPtr = matched->CSC ;
            }

            if ((here-> BSIM4v6gNodePrime != 0) && (here-> BSIM4v6gNodePrime != 0))
            {
                i = here->BSIM4v6GPgpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4v6GPgpStructPtr = matched ;
                here->BSIM4v6GPgpPtr = matched->CSC ;
            }

            if ((here-> BSIM4v6sNode != 0) && (here-> BSIM4v6sNode != 0))
            {
                i = here->BSIM4v6SsPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4v6SsStructPtr = matched ;
                here->BSIM4v6SsPtr = matched->CSC ;
            }

            if ((here-> BSIM4v6dNodePrime != 0) && (here-> BSIM4v6dNodePrime != 0))
            {
                i = here->BSIM4v6DPdpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4v6DPdpStructPtr = matched ;
                here->BSIM4v6DPdpPtr = matched->CSC ;
            }

            if ((here-> BSIM4v6sNodePrime != 0) && (here-> BSIM4v6sNodePrime != 0))
            {
                i = here->BSIM4v6SPspPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4v6SPspStructPtr = matched ;
                here->BSIM4v6SPspPtr = matched->CSC ;
            }

            if ((here-> BSIM4v6dNode != 0) && (here-> BSIM4v6dNodePrime != 0))
            {
                i = here->BSIM4v6DdpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4v6DdpStructPtr = matched ;
                here->BSIM4v6DdpPtr = matched->CSC ;
            }

            if ((here-> BSIM4v6gNodePrime != 0) && (here-> BSIM4v6dNodePrime != 0))
            {
                i = here->BSIM4v6GPdpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4v6GPdpStructPtr = matched ;
                here->BSIM4v6GPdpPtr = matched->CSC ;
            }

            if ((here-> BSIM4v6gNodePrime != 0) && (here-> BSIM4v6sNodePrime != 0))
            {
                i = here->BSIM4v6GPspPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4v6GPspStructPtr = matched ;
                here->BSIM4v6GPspPtr = matched->CSC ;
            }

            if ((here-> BSIM4v6sNode != 0) && (here-> BSIM4v6sNodePrime != 0))
            {
                i = here->BSIM4v6SspPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4v6SspStructPtr = matched ;
                here->BSIM4v6SspPtr = matched->CSC ;
            }

            if ((here-> BSIM4v6dNodePrime != 0) && (here-> BSIM4v6sNodePrime != 0))
            {
                i = here->BSIM4v6DPspPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4v6DPspStructPtr = matched ;
                here->BSIM4v6DPspPtr = matched->CSC ;
            }

            if ((here-> BSIM4v6dNodePrime != 0) && (here-> BSIM4v6dNode != 0))
            {
                i = here->BSIM4v6DPdPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4v6DPdStructPtr = matched ;
                here->BSIM4v6DPdPtr = matched->CSC ;
            }

            if ((here-> BSIM4v6dNodePrime != 0) && (here-> BSIM4v6gNodePrime != 0))
            {
                i = here->BSIM4v6DPgpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4v6DPgpStructPtr = matched ;
                here->BSIM4v6DPgpPtr = matched->CSC ;
            }

            if ((here-> BSIM4v6sNodePrime != 0) && (here-> BSIM4v6gNodePrime != 0))
            {
                i = here->BSIM4v6SPgpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4v6SPgpStructPtr = matched ;
                here->BSIM4v6SPgpPtr = matched->CSC ;
            }

            if ((here-> BSIM4v6sNodePrime != 0) && (here-> BSIM4v6sNode != 0))
            {
                i = here->BSIM4v6SPsPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4v6SPsStructPtr = matched ;
                here->BSIM4v6SPsPtr = matched->CSC ;
            }

            if ((here-> BSIM4v6sNodePrime != 0) && (here-> BSIM4v6dNodePrime != 0))
            {
                i = here->BSIM4v6SPdpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4v6SPdpStructPtr = matched ;
                here->BSIM4v6SPdpPtr = matched->CSC ;
            }

            if ((here-> BSIM4v6qNode != 0) && (here-> BSIM4v6qNode != 0))
            {
                i = here->BSIM4v6QqPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4v6QqStructPtr = matched ;
                here->BSIM4v6QqPtr = matched->CSC ;
            }

            if ((here-> BSIM4v6qNode != 0) && (here-> BSIM4v6bNodePrime != 0))
            {
                i = here->BSIM4v6QbpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4v6QbpStructPtr = matched ;
                here->BSIM4v6QbpPtr = matched->CSC ;
            }

            if ((here-> BSIM4v6qNode != 0) && (here-> BSIM4v6dNodePrime != 0))
            {
                i = here->BSIM4v6QdpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4v6QdpStructPtr = matched ;
                here->BSIM4v6QdpPtr = matched->CSC ;
            }

            if ((here-> BSIM4v6qNode != 0) && (here-> BSIM4v6sNodePrime != 0))
            {
                i = here->BSIM4v6QspPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4v6QspStructPtr = matched ;
                here->BSIM4v6QspPtr = matched->CSC ;
            }

            if ((here-> BSIM4v6qNode != 0) && (here-> BSIM4v6gNodePrime != 0))
            {
                i = here->BSIM4v6QgpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4v6QgpStructPtr = matched ;
                here->BSIM4v6QgpPtr = matched->CSC ;
            }

            if ((here-> BSIM4v6dNodePrime != 0) && (here-> BSIM4v6qNode != 0))
            {
                i = here->BSIM4v6DPqPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4v6DPqStructPtr = matched ;
                here->BSIM4v6DPqPtr = matched->CSC ;
            }

            if ((here-> BSIM4v6sNodePrime != 0) && (here-> BSIM4v6qNode != 0))
            {
                i = here->BSIM4v6SPqPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4v6SPqStructPtr = matched ;
                here->BSIM4v6SPqPtr = matched->CSC ;
            }

            if ((here-> BSIM4v6gNodePrime != 0) && (here-> BSIM4v6qNode != 0))
            {
                i = here->BSIM4v6GPqPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4v6GPqStructPtr = matched ;
                here->BSIM4v6GPqPtr = matched->CSC ;
            }

            if (here->BSIM4v6rgateMod != 0)
            {
                if ((here-> BSIM4v6gNodeExt != 0) && (here-> BSIM4v6gNodeExt != 0))
                {
                    i = here->BSIM4v6GEgePtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v6GEgeStructPtr = matched ;
                    here->BSIM4v6GEgePtr = matched->CSC ;
                }

                if ((here-> BSIM4v6gNodeExt != 0) && (here-> BSIM4v6gNodePrime != 0))
                {
                    i = here->BSIM4v6GEgpPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v6GEgpStructPtr = matched ;
                    here->BSIM4v6GEgpPtr = matched->CSC ;
                }

                if ((here-> BSIM4v6gNodePrime != 0) && (here-> BSIM4v6gNodeExt != 0))
                {
                    i = here->BSIM4v6GPgePtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v6GPgeStructPtr = matched ;
                    here->BSIM4v6GPgePtr = matched->CSC ;
                }

                if ((here-> BSIM4v6gNodeExt != 0) && (here-> BSIM4v6dNodePrime != 0))
                {
                    i = here->BSIM4v6GEdpPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v6GEdpStructPtr = matched ;
                    here->BSIM4v6GEdpPtr = matched->CSC ;
                }

                if ((here-> BSIM4v6gNodeExt != 0) && (here-> BSIM4v6sNodePrime != 0))
                {
                    i = here->BSIM4v6GEspPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v6GEspStructPtr = matched ;
                    here->BSIM4v6GEspPtr = matched->CSC ;
                }

                if ((here-> BSIM4v6gNodeExt != 0) && (here-> BSIM4v6bNodePrime != 0))
                {
                    i = here->BSIM4v6GEbpPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v6GEbpStructPtr = matched ;
                    here->BSIM4v6GEbpPtr = matched->CSC ;
                }

                if ((here-> BSIM4v6gNodeMid != 0) && (here-> BSIM4v6dNodePrime != 0))
                {
                    i = here->BSIM4v6GMdpPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v6GMdpStructPtr = matched ;
                    here->BSIM4v6GMdpPtr = matched->CSC ;
                }

                if ((here-> BSIM4v6gNodeMid != 0) && (here-> BSIM4v6gNodePrime != 0))
                {
                    i = here->BSIM4v6GMgpPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v6GMgpStructPtr = matched ;
                    here->BSIM4v6GMgpPtr = matched->CSC ;
                }

                if ((here-> BSIM4v6gNodeMid != 0) && (here-> BSIM4v6gNodeMid != 0))
                {
                    i = here->BSIM4v6GMgmPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v6GMgmStructPtr = matched ;
                    here->BSIM4v6GMgmPtr = matched->CSC ;
                }

                if ((here-> BSIM4v6gNodeMid != 0) && (here-> BSIM4v6gNodeExt != 0))
                {
                    i = here->BSIM4v6GMgePtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v6GMgeStructPtr = matched ;
                    here->BSIM4v6GMgePtr = matched->CSC ;
                }

                if ((here-> BSIM4v6gNodeMid != 0) && (here-> BSIM4v6sNodePrime != 0))
                {
                    i = here->BSIM4v6GMspPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v6GMspStructPtr = matched ;
                    here->BSIM4v6GMspPtr = matched->CSC ;
                }

                if ((here-> BSIM4v6gNodeMid != 0) && (here-> BSIM4v6bNodePrime != 0))
                {
                    i = here->BSIM4v6GMbpPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v6GMbpStructPtr = matched ;
                    here->BSIM4v6GMbpPtr = matched->CSC ;
                }

                if ((here-> BSIM4v6dNodePrime != 0) && (here-> BSIM4v6gNodeMid != 0))
                {
                    i = here->BSIM4v6DPgmPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v6DPgmStructPtr = matched ;
                    here->BSIM4v6DPgmPtr = matched->CSC ;
                }

                if ((here-> BSIM4v6gNodePrime != 0) && (here-> BSIM4v6gNodeMid != 0))
                {
                    i = here->BSIM4v6GPgmPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v6GPgmStructPtr = matched ;
                    here->BSIM4v6GPgmPtr = matched->CSC ;
                }

                if ((here-> BSIM4v6gNodeExt != 0) && (here-> BSIM4v6gNodeMid != 0))
                {
                    i = here->BSIM4v6GEgmPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v6GEgmStructPtr = matched ;
                    here->BSIM4v6GEgmPtr = matched->CSC ;
                }

                if ((here-> BSIM4v6sNodePrime != 0) && (here-> BSIM4v6gNodeMid != 0))
                {
                    i = here->BSIM4v6SPgmPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v6SPgmStructPtr = matched ;
                    here->BSIM4v6SPgmPtr = matched->CSC ;
                }

                if ((here-> BSIM4v6bNodePrime != 0) && (here-> BSIM4v6gNodeMid != 0))
                {
                    i = here->BSIM4v6BPgmPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v6BPgmStructPtr = matched ;
                    here->BSIM4v6BPgmPtr = matched->CSC ;
                }

            }
            if ((here->BSIM4v6rbodyMod ==1) || (here->BSIM4v6rbodyMod ==2))
            {
                if ((here-> BSIM4v6dNodePrime != 0) && (here-> BSIM4v6dbNode != 0))
                {
                    i = here->BSIM4v6DPdbPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v6DPdbStructPtr = matched ;
                    here->BSIM4v6DPdbPtr = matched->CSC ;
                }

                if ((here-> BSIM4v6sNodePrime != 0) && (here-> BSIM4v6sbNode != 0))
                {
                    i = here->BSIM4v6SPsbPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v6SPsbStructPtr = matched ;
                    here->BSIM4v6SPsbPtr = matched->CSC ;
                }

                if ((here-> BSIM4v6dbNode != 0) && (here-> BSIM4v6dNodePrime != 0))
                {
                    i = here->BSIM4v6DBdpPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v6DBdpStructPtr = matched ;
                    here->BSIM4v6DBdpPtr = matched->CSC ;
                }

                if ((here-> BSIM4v6dbNode != 0) && (here-> BSIM4v6dbNode != 0))
                {
                    i = here->BSIM4v6DBdbPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v6DBdbStructPtr = matched ;
                    here->BSIM4v6DBdbPtr = matched->CSC ;
                }

                if ((here-> BSIM4v6dbNode != 0) && (here-> BSIM4v6bNodePrime != 0))
                {
                    i = here->BSIM4v6DBbpPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v6DBbpStructPtr = matched ;
                    here->BSIM4v6DBbpPtr = matched->CSC ;
                }

                if ((here-> BSIM4v6dbNode != 0) && (here-> BSIM4v6bNode != 0))
                {
                    i = here->BSIM4v6DBbPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v6DBbStructPtr = matched ;
                    here->BSIM4v6DBbPtr = matched->CSC ;
                }

                if ((here-> BSIM4v6bNodePrime != 0) && (here-> BSIM4v6dbNode != 0))
                {
                    i = here->BSIM4v6BPdbPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v6BPdbStructPtr = matched ;
                    here->BSIM4v6BPdbPtr = matched->CSC ;
                }

                if ((here-> BSIM4v6bNodePrime != 0) && (here-> BSIM4v6bNode != 0))
                {
                    i = here->BSIM4v6BPbPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v6BPbStructPtr = matched ;
                    here->BSIM4v6BPbPtr = matched->CSC ;
                }

                if ((here-> BSIM4v6bNodePrime != 0) && (here-> BSIM4v6sbNode != 0))
                {
                    i = here->BSIM4v6BPsbPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v6BPsbStructPtr = matched ;
                    here->BSIM4v6BPsbPtr = matched->CSC ;
                }

                if ((here-> BSIM4v6sbNode != 0) && (here-> BSIM4v6sNodePrime != 0))
                {
                    i = here->BSIM4v6SBspPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v6SBspStructPtr = matched ;
                    here->BSIM4v6SBspPtr = matched->CSC ;
                }

                if ((here-> BSIM4v6sbNode != 0) && (here-> BSIM4v6bNodePrime != 0))
                {
                    i = here->BSIM4v6SBbpPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v6SBbpStructPtr = matched ;
                    here->BSIM4v6SBbpPtr = matched->CSC ;
                }

                if ((here-> BSIM4v6sbNode != 0) && (here-> BSIM4v6bNode != 0))
                {
                    i = here->BSIM4v6SBbPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v6SBbStructPtr = matched ;
                    here->BSIM4v6SBbPtr = matched->CSC ;
                }

                if ((here-> BSIM4v6sbNode != 0) && (here-> BSIM4v6sbNode != 0))
                {
                    i = here->BSIM4v6SBsbPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v6SBsbStructPtr = matched ;
                    here->BSIM4v6SBsbPtr = matched->CSC ;
                }

                if ((here-> BSIM4v6bNode != 0) && (here-> BSIM4v6dbNode != 0))
                {
                    i = here->BSIM4v6BdbPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v6BdbStructPtr = matched ;
                    here->BSIM4v6BdbPtr = matched->CSC ;
                }

                if ((here-> BSIM4v6bNode != 0) && (here-> BSIM4v6bNodePrime != 0))
                {
                    i = here->BSIM4v6BbpPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v6BbpStructPtr = matched ;
                    here->BSIM4v6BbpPtr = matched->CSC ;
                }

                if ((here-> BSIM4v6bNode != 0) && (here-> BSIM4v6sbNode != 0))
                {
                    i = here->BSIM4v6BsbPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v6BsbStructPtr = matched ;
                    here->BSIM4v6BsbPtr = matched->CSC ;
                }

                if ((here-> BSIM4v6bNode != 0) && (here-> BSIM4v6bNode != 0))
                {
                    i = here->BSIM4v6BbPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v6BbStructPtr = matched ;
                    here->BSIM4v6BbPtr = matched->CSC ;
                }

            }
            if (model->BSIM4v6rdsMod)
            {
                if ((here-> BSIM4v6dNode != 0) && (here-> BSIM4v6gNodePrime != 0))
                {
                    i = here->BSIM4v6DgpPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v6DgpStructPtr = matched ;
                    here->BSIM4v6DgpPtr = matched->CSC ;
                }

                if ((here-> BSIM4v6dNode != 0) && (here-> BSIM4v6sNodePrime != 0))
                {
                    i = here->BSIM4v6DspPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v6DspStructPtr = matched ;
                    here->BSIM4v6DspPtr = matched->CSC ;
                }

                if ((here-> BSIM4v6dNode != 0) && (here-> BSIM4v6bNodePrime != 0))
                {
                    i = here->BSIM4v6DbpPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v6DbpStructPtr = matched ;
                    here->BSIM4v6DbpPtr = matched->CSC ;
                }

                if ((here-> BSIM4v6sNode != 0) && (here-> BSIM4v6dNodePrime != 0))
                {
                    i = here->BSIM4v6SdpPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v6SdpStructPtr = matched ;
                    here->BSIM4v6SdpPtr = matched->CSC ;
                }

                if ((here-> BSIM4v6sNode != 0) && (here-> BSIM4v6gNodePrime != 0))
                {
                    i = here->BSIM4v6SgpPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v6SgpStructPtr = matched ;
                    here->BSIM4v6SgpPtr = matched->CSC ;
                }

                if ((here-> BSIM4v6sNode != 0) && (here-> BSIM4v6bNodePrime != 0))
                {
                    i = here->BSIM4v6SbpPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v6SbpStructPtr = matched ;
                    here->BSIM4v6SbpPtr = matched->CSC ;
                }

            }
        }
    }

    return (OK) ;
}

int
BSIM4v6bindCSCComplex (GENmodel *inModel, CKTcircuit *ckt)
{
    BSIM4v6model *model = (BSIM4v6model *)inModel ;
    BSIM4v6instance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the BSIM4v6 models */
    for ( ; model != NULL ; model = model->BSIM4v6nextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->BSIM4v6instances ; here != NULL ; here = here->BSIM4v6nextInstance)
        {
            if ((here-> BSIM4v6dNodePrime != 0) && (here-> BSIM4v6bNodePrime != 0))
                here->BSIM4v6DPbpPtr = here->BSIM4v6DPbpStructPtr->CSC_Complex ;

            if ((here-> BSIM4v6gNodePrime != 0) && (here-> BSIM4v6bNodePrime != 0))
                here->BSIM4v6GPbpPtr = here->BSIM4v6GPbpStructPtr->CSC_Complex ;

            if ((here-> BSIM4v6sNodePrime != 0) && (here-> BSIM4v6bNodePrime != 0))
                here->BSIM4v6SPbpPtr = here->BSIM4v6SPbpStructPtr->CSC_Complex ;

            if ((here-> BSIM4v6bNodePrime != 0) && (here-> BSIM4v6dNodePrime != 0))
                here->BSIM4v6BPdpPtr = here->BSIM4v6BPdpStructPtr->CSC_Complex ;

            if ((here-> BSIM4v6bNodePrime != 0) && (here-> BSIM4v6gNodePrime != 0))
                here->BSIM4v6BPgpPtr = here->BSIM4v6BPgpStructPtr->CSC_Complex ;

            if ((here-> BSIM4v6bNodePrime != 0) && (here-> BSIM4v6sNodePrime != 0))
                here->BSIM4v6BPspPtr = here->BSIM4v6BPspStructPtr->CSC_Complex ;

            if ((here-> BSIM4v6bNodePrime != 0) && (here-> BSIM4v6bNodePrime != 0))
                here->BSIM4v6BPbpPtr = here->BSIM4v6BPbpStructPtr->CSC_Complex ;

            if ((here-> BSIM4v6dNode != 0) && (here-> BSIM4v6dNode != 0))
                here->BSIM4v6DdPtr = here->BSIM4v6DdStructPtr->CSC_Complex ;

            if ((here-> BSIM4v6gNodePrime != 0) && (here-> BSIM4v6gNodePrime != 0))
                here->BSIM4v6GPgpPtr = here->BSIM4v6GPgpStructPtr->CSC_Complex ;

            if ((here-> BSIM4v6sNode != 0) && (here-> BSIM4v6sNode != 0))
                here->BSIM4v6SsPtr = here->BSIM4v6SsStructPtr->CSC_Complex ;

            if ((here-> BSIM4v6dNodePrime != 0) && (here-> BSIM4v6dNodePrime != 0))
                here->BSIM4v6DPdpPtr = here->BSIM4v6DPdpStructPtr->CSC_Complex ;

            if ((here-> BSIM4v6sNodePrime != 0) && (here-> BSIM4v6sNodePrime != 0))
                here->BSIM4v6SPspPtr = here->BSIM4v6SPspStructPtr->CSC_Complex ;

            if ((here-> BSIM4v6dNode != 0) && (here-> BSIM4v6dNodePrime != 0))
                here->BSIM4v6DdpPtr = here->BSIM4v6DdpStructPtr->CSC_Complex ;

            if ((here-> BSIM4v6gNodePrime != 0) && (here-> BSIM4v6dNodePrime != 0))
                here->BSIM4v6GPdpPtr = here->BSIM4v6GPdpStructPtr->CSC_Complex ;

            if ((here-> BSIM4v6gNodePrime != 0) && (here-> BSIM4v6sNodePrime != 0))
                here->BSIM4v6GPspPtr = here->BSIM4v6GPspStructPtr->CSC_Complex ;

            if ((here-> BSIM4v6sNode != 0) && (here-> BSIM4v6sNodePrime != 0))
                here->BSIM4v6SspPtr = here->BSIM4v6SspStructPtr->CSC_Complex ;

            if ((here-> BSIM4v6dNodePrime != 0) && (here-> BSIM4v6sNodePrime != 0))
                here->BSIM4v6DPspPtr = here->BSIM4v6DPspStructPtr->CSC_Complex ;

            if ((here-> BSIM4v6dNodePrime != 0) && (here-> BSIM4v6dNode != 0))
                here->BSIM4v6DPdPtr = here->BSIM4v6DPdStructPtr->CSC_Complex ;

            if ((here-> BSIM4v6dNodePrime != 0) && (here-> BSIM4v6gNodePrime != 0))
                here->BSIM4v6DPgpPtr = here->BSIM4v6DPgpStructPtr->CSC_Complex ;

            if ((here-> BSIM4v6sNodePrime != 0) && (here-> BSIM4v6gNodePrime != 0))
                here->BSIM4v6SPgpPtr = here->BSIM4v6SPgpStructPtr->CSC_Complex ;

            if ((here-> BSIM4v6sNodePrime != 0) && (here-> BSIM4v6sNode != 0))
                here->BSIM4v6SPsPtr = here->BSIM4v6SPsStructPtr->CSC_Complex ;

            if ((here-> BSIM4v6sNodePrime != 0) && (here-> BSIM4v6dNodePrime != 0))
                here->BSIM4v6SPdpPtr = here->BSIM4v6SPdpStructPtr->CSC_Complex ;

            if ((here-> BSIM4v6qNode != 0) && (here-> BSIM4v6qNode != 0))
                here->BSIM4v6QqPtr = here->BSIM4v6QqStructPtr->CSC_Complex ;

            if ((here-> BSIM4v6qNode != 0) && (here-> BSIM4v6bNodePrime != 0))
                here->BSIM4v6QbpPtr = here->BSIM4v6QbpStructPtr->CSC_Complex ;

            if ((here-> BSIM4v6qNode != 0) && (here-> BSIM4v6dNodePrime != 0))
                here->BSIM4v6QdpPtr = here->BSIM4v6QdpStructPtr->CSC_Complex ;

            if ((here-> BSIM4v6qNode != 0) && (here-> BSIM4v6sNodePrime != 0))
                here->BSIM4v6QspPtr = here->BSIM4v6QspStructPtr->CSC_Complex ;

            if ((here-> BSIM4v6qNode != 0) && (here-> BSIM4v6gNodePrime != 0))
                here->BSIM4v6QgpPtr = here->BSIM4v6QgpStructPtr->CSC_Complex ;

            if ((here-> BSIM4v6dNodePrime != 0) && (here-> BSIM4v6qNode != 0))
                here->BSIM4v6DPqPtr = here->BSIM4v6DPqStructPtr->CSC_Complex ;

            if ((here-> BSIM4v6sNodePrime != 0) && (here-> BSIM4v6qNode != 0))
                here->BSIM4v6SPqPtr = here->BSIM4v6SPqStructPtr->CSC_Complex ;

            if ((here-> BSIM4v6gNodePrime != 0) && (here-> BSIM4v6qNode != 0))
                here->BSIM4v6GPqPtr = here->BSIM4v6GPqStructPtr->CSC_Complex ;

            if (here->BSIM4v6rgateMod != 0)
            {
                if ((here-> BSIM4v6gNodeExt != 0) && (here-> BSIM4v6gNodeExt != 0))
                    here->BSIM4v6GEgePtr = here->BSIM4v6GEgeStructPtr->CSC_Complex ;

                if ((here-> BSIM4v6gNodeExt != 0) && (here-> BSIM4v6gNodePrime != 0))
                    here->BSIM4v6GEgpPtr = here->BSIM4v6GEgpStructPtr->CSC_Complex ;

                if ((here-> BSIM4v6gNodePrime != 0) && (here-> BSIM4v6gNodeExt != 0))
                    here->BSIM4v6GPgePtr = here->BSIM4v6GPgeStructPtr->CSC_Complex ;

                if ((here-> BSIM4v6gNodeExt != 0) && (here-> BSIM4v6dNodePrime != 0))
                    here->BSIM4v6GEdpPtr = here->BSIM4v6GEdpStructPtr->CSC_Complex ;

                if ((here-> BSIM4v6gNodeExt != 0) && (here-> BSIM4v6sNodePrime != 0))
                    here->BSIM4v6GEspPtr = here->BSIM4v6GEspStructPtr->CSC_Complex ;

                if ((here-> BSIM4v6gNodeExt != 0) && (here-> BSIM4v6bNodePrime != 0))
                    here->BSIM4v6GEbpPtr = here->BSIM4v6GEbpStructPtr->CSC_Complex ;

                if ((here-> BSIM4v6gNodeMid != 0) && (here-> BSIM4v6dNodePrime != 0))
                    here->BSIM4v6GMdpPtr = here->BSIM4v6GMdpStructPtr->CSC_Complex ;

                if ((here-> BSIM4v6gNodeMid != 0) && (here-> BSIM4v6gNodePrime != 0))
                    here->BSIM4v6GMgpPtr = here->BSIM4v6GMgpStructPtr->CSC_Complex ;

                if ((here-> BSIM4v6gNodeMid != 0) && (here-> BSIM4v6gNodeMid != 0))
                    here->BSIM4v6GMgmPtr = here->BSIM4v6GMgmStructPtr->CSC_Complex ;

                if ((here-> BSIM4v6gNodeMid != 0) && (here-> BSIM4v6gNodeExt != 0))
                    here->BSIM4v6GMgePtr = here->BSIM4v6GMgeStructPtr->CSC_Complex ;

                if ((here-> BSIM4v6gNodeMid != 0) && (here-> BSIM4v6sNodePrime != 0))
                    here->BSIM4v6GMspPtr = here->BSIM4v6GMspStructPtr->CSC_Complex ;

                if ((here-> BSIM4v6gNodeMid != 0) && (here-> BSIM4v6bNodePrime != 0))
                    here->BSIM4v6GMbpPtr = here->BSIM4v6GMbpStructPtr->CSC_Complex ;

                if ((here-> BSIM4v6dNodePrime != 0) && (here-> BSIM4v6gNodeMid != 0))
                    here->BSIM4v6DPgmPtr = here->BSIM4v6DPgmStructPtr->CSC_Complex ;

                if ((here-> BSIM4v6gNodePrime != 0) && (here-> BSIM4v6gNodeMid != 0))
                    here->BSIM4v6GPgmPtr = here->BSIM4v6GPgmStructPtr->CSC_Complex ;

                if ((here-> BSIM4v6gNodeExt != 0) && (here-> BSIM4v6gNodeMid != 0))
                    here->BSIM4v6GEgmPtr = here->BSIM4v6GEgmStructPtr->CSC_Complex ;

                if ((here-> BSIM4v6sNodePrime != 0) && (here-> BSIM4v6gNodeMid != 0))
                    here->BSIM4v6SPgmPtr = here->BSIM4v6SPgmStructPtr->CSC_Complex ;

                if ((here-> BSIM4v6bNodePrime != 0) && (here-> BSIM4v6gNodeMid != 0))
                    here->BSIM4v6BPgmPtr = here->BSIM4v6BPgmStructPtr->CSC_Complex ;

            }
            if ((here->BSIM4v6rbodyMod ==1) || (here->BSIM4v6rbodyMod ==2))
            {
                if ((here-> BSIM4v6dNodePrime != 0) && (here-> BSIM4v6dbNode != 0))
                    here->BSIM4v6DPdbPtr = here->BSIM4v6DPdbStructPtr->CSC_Complex ;

                if ((here-> BSIM4v6sNodePrime != 0) && (here-> BSIM4v6sbNode != 0))
                    here->BSIM4v6SPsbPtr = here->BSIM4v6SPsbStructPtr->CSC_Complex ;

                if ((here-> BSIM4v6dbNode != 0) && (here-> BSIM4v6dNodePrime != 0))
                    here->BSIM4v6DBdpPtr = here->BSIM4v6DBdpStructPtr->CSC_Complex ;

                if ((here-> BSIM4v6dbNode != 0) && (here-> BSIM4v6dbNode != 0))
                    here->BSIM4v6DBdbPtr = here->BSIM4v6DBdbStructPtr->CSC_Complex ;

                if ((here-> BSIM4v6dbNode != 0) && (here-> BSIM4v6bNodePrime != 0))
                    here->BSIM4v6DBbpPtr = here->BSIM4v6DBbpStructPtr->CSC_Complex ;

                if ((here-> BSIM4v6dbNode != 0) && (here-> BSIM4v6bNode != 0))
                    here->BSIM4v6DBbPtr = here->BSIM4v6DBbStructPtr->CSC_Complex ;

                if ((here-> BSIM4v6bNodePrime != 0) && (here-> BSIM4v6dbNode != 0))
                    here->BSIM4v6BPdbPtr = here->BSIM4v6BPdbStructPtr->CSC_Complex ;

                if ((here-> BSIM4v6bNodePrime != 0) && (here-> BSIM4v6bNode != 0))
                    here->BSIM4v6BPbPtr = here->BSIM4v6BPbStructPtr->CSC_Complex ;

                if ((here-> BSIM4v6bNodePrime != 0) && (here-> BSIM4v6sbNode != 0))
                    here->BSIM4v6BPsbPtr = here->BSIM4v6BPsbStructPtr->CSC_Complex ;

                if ((here-> BSIM4v6sbNode != 0) && (here-> BSIM4v6sNodePrime != 0))
                    here->BSIM4v6SBspPtr = here->BSIM4v6SBspStructPtr->CSC_Complex ;

                if ((here-> BSIM4v6sbNode != 0) && (here-> BSIM4v6bNodePrime != 0))
                    here->BSIM4v6SBbpPtr = here->BSIM4v6SBbpStructPtr->CSC_Complex ;

                if ((here-> BSIM4v6sbNode != 0) && (here-> BSIM4v6bNode != 0))
                    here->BSIM4v6SBbPtr = here->BSIM4v6SBbStructPtr->CSC_Complex ;

                if ((here-> BSIM4v6sbNode != 0) && (here-> BSIM4v6sbNode != 0))
                    here->BSIM4v6SBsbPtr = here->BSIM4v6SBsbStructPtr->CSC_Complex ;

                if ((here-> BSIM4v6bNode != 0) && (here-> BSIM4v6dbNode != 0))
                    here->BSIM4v6BdbPtr = here->BSIM4v6BdbStructPtr->CSC_Complex ;

                if ((here-> BSIM4v6bNode != 0) && (here-> BSIM4v6bNodePrime != 0))
                    here->BSIM4v6BbpPtr = here->BSIM4v6BbpStructPtr->CSC_Complex ;

                if ((here-> BSIM4v6bNode != 0) && (here-> BSIM4v6sbNode != 0))
                    here->BSIM4v6BsbPtr = here->BSIM4v6BsbStructPtr->CSC_Complex ;

                if ((here-> BSIM4v6bNode != 0) && (here-> BSIM4v6bNode != 0))
                    here->BSIM4v6BbPtr = here->BSIM4v6BbStructPtr->CSC_Complex ;

            }
            if (model->BSIM4v6rdsMod)
            {
                if ((here-> BSIM4v6dNode != 0) && (here-> BSIM4v6gNodePrime != 0))
                    here->BSIM4v6DgpPtr = here->BSIM4v6DgpStructPtr->CSC_Complex ;

                if ((here-> BSIM4v6dNode != 0) && (here-> BSIM4v6sNodePrime != 0))
                    here->BSIM4v6DspPtr = here->BSIM4v6DspStructPtr->CSC_Complex ;

                if ((here-> BSIM4v6dNode != 0) && (here-> BSIM4v6bNodePrime != 0))
                    here->BSIM4v6DbpPtr = here->BSIM4v6DbpStructPtr->CSC_Complex ;

                if ((here-> BSIM4v6sNode != 0) && (here-> BSIM4v6dNodePrime != 0))
                    here->BSIM4v6SdpPtr = here->BSIM4v6SdpStructPtr->CSC_Complex ;

                if ((here-> BSIM4v6sNode != 0) && (here-> BSIM4v6gNodePrime != 0))
                    here->BSIM4v6SgpPtr = here->BSIM4v6SgpStructPtr->CSC_Complex ;

                if ((here-> BSIM4v6sNode != 0) && (here-> BSIM4v6bNodePrime != 0))
                    here->BSIM4v6SbpPtr = here->BSIM4v6SbpStructPtr->CSC_Complex ;

            }
        }
    }

    return (OK) ;
}

int
BSIM4v6bindCSCComplexToReal (GENmodel *inModel, CKTcircuit *ckt)
{
    BSIM4v6model *model = (BSIM4v6model *)inModel ;
    BSIM4v6instance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the BSIM4v6 models */
    for ( ; model != NULL ; model = model->BSIM4v6nextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->BSIM4v6instances ; here != NULL ; here = here->BSIM4v6nextInstance)
        {
            if ((here-> BSIM4v6dNodePrime != 0) && (here-> BSIM4v6bNodePrime != 0))
                here->BSIM4v6DPbpPtr = here->BSIM4v6DPbpStructPtr->CSC ;

            if ((here-> BSIM4v6gNodePrime != 0) && (here-> BSIM4v6bNodePrime != 0))
                here->BSIM4v6GPbpPtr = here->BSIM4v6GPbpStructPtr->CSC ;

            if ((here-> BSIM4v6sNodePrime != 0) && (here-> BSIM4v6bNodePrime != 0))
                here->BSIM4v6SPbpPtr = here->BSIM4v6SPbpStructPtr->CSC ;

            if ((here-> BSIM4v6bNodePrime != 0) && (here-> BSIM4v6dNodePrime != 0))
                here->BSIM4v6BPdpPtr = here->BSIM4v6BPdpStructPtr->CSC ;

            if ((here-> BSIM4v6bNodePrime != 0) && (here-> BSIM4v6gNodePrime != 0))
                here->BSIM4v6BPgpPtr = here->BSIM4v6BPgpStructPtr->CSC ;

            if ((here-> BSIM4v6bNodePrime != 0) && (here-> BSIM4v6sNodePrime != 0))
                here->BSIM4v6BPspPtr = here->BSIM4v6BPspStructPtr->CSC ;

            if ((here-> BSIM4v6bNodePrime != 0) && (here-> BSIM4v6bNodePrime != 0))
                here->BSIM4v6BPbpPtr = here->BSIM4v6BPbpStructPtr->CSC ;

            if ((here-> BSIM4v6dNode != 0) && (here-> BSIM4v6dNode != 0))
                here->BSIM4v6DdPtr = here->BSIM4v6DdStructPtr->CSC ;

            if ((here-> BSIM4v6gNodePrime != 0) && (here-> BSIM4v6gNodePrime != 0))
                here->BSIM4v6GPgpPtr = here->BSIM4v6GPgpStructPtr->CSC ;

            if ((here-> BSIM4v6sNode != 0) && (here-> BSIM4v6sNode != 0))
                here->BSIM4v6SsPtr = here->BSIM4v6SsStructPtr->CSC ;

            if ((here-> BSIM4v6dNodePrime != 0) && (here-> BSIM4v6dNodePrime != 0))
                here->BSIM4v6DPdpPtr = here->BSIM4v6DPdpStructPtr->CSC ;

            if ((here-> BSIM4v6sNodePrime != 0) && (here-> BSIM4v6sNodePrime != 0))
                here->BSIM4v6SPspPtr = here->BSIM4v6SPspStructPtr->CSC ;

            if ((here-> BSIM4v6dNode != 0) && (here-> BSIM4v6dNodePrime != 0))
                here->BSIM4v6DdpPtr = here->BSIM4v6DdpStructPtr->CSC ;

            if ((here-> BSIM4v6gNodePrime != 0) && (here-> BSIM4v6dNodePrime != 0))
                here->BSIM4v6GPdpPtr = here->BSIM4v6GPdpStructPtr->CSC ;

            if ((here-> BSIM4v6gNodePrime != 0) && (here-> BSIM4v6sNodePrime != 0))
                here->BSIM4v6GPspPtr = here->BSIM4v6GPspStructPtr->CSC ;

            if ((here-> BSIM4v6sNode != 0) && (here-> BSIM4v6sNodePrime != 0))
                here->BSIM4v6SspPtr = here->BSIM4v6SspStructPtr->CSC ;

            if ((here-> BSIM4v6dNodePrime != 0) && (here-> BSIM4v6sNodePrime != 0))
                here->BSIM4v6DPspPtr = here->BSIM4v6DPspStructPtr->CSC ;

            if ((here-> BSIM4v6dNodePrime != 0) && (here-> BSIM4v6dNode != 0))
                here->BSIM4v6DPdPtr = here->BSIM4v6DPdStructPtr->CSC ;

            if ((here-> BSIM4v6dNodePrime != 0) && (here-> BSIM4v6gNodePrime != 0))
                here->BSIM4v6DPgpPtr = here->BSIM4v6DPgpStructPtr->CSC ;

            if ((here-> BSIM4v6sNodePrime != 0) && (here-> BSIM4v6gNodePrime != 0))
                here->BSIM4v6SPgpPtr = here->BSIM4v6SPgpStructPtr->CSC ;

            if ((here-> BSIM4v6sNodePrime != 0) && (here-> BSIM4v6sNode != 0))
                here->BSIM4v6SPsPtr = here->BSIM4v6SPsStructPtr->CSC ;

            if ((here-> BSIM4v6sNodePrime != 0) && (here-> BSIM4v6dNodePrime != 0))
                here->BSIM4v6SPdpPtr = here->BSIM4v6SPdpStructPtr->CSC ;

            if ((here-> BSIM4v6qNode != 0) && (here-> BSIM4v6qNode != 0))
                here->BSIM4v6QqPtr = here->BSIM4v6QqStructPtr->CSC ;

            if ((here-> BSIM4v6qNode != 0) && (here-> BSIM4v6bNodePrime != 0))
                here->BSIM4v6QbpPtr = here->BSIM4v6QbpStructPtr->CSC ;

            if ((here-> BSIM4v6qNode != 0) && (here-> BSIM4v6dNodePrime != 0))
                here->BSIM4v6QdpPtr = here->BSIM4v6QdpStructPtr->CSC ;

            if ((here-> BSIM4v6qNode != 0) && (here-> BSIM4v6sNodePrime != 0))
                here->BSIM4v6QspPtr = here->BSIM4v6QspStructPtr->CSC ;

            if ((here-> BSIM4v6qNode != 0) && (here-> BSIM4v6gNodePrime != 0))
                here->BSIM4v6QgpPtr = here->BSIM4v6QgpStructPtr->CSC ;

            if ((here-> BSIM4v6dNodePrime != 0) && (here-> BSIM4v6qNode != 0))
                here->BSIM4v6DPqPtr = here->BSIM4v6DPqStructPtr->CSC ;

            if ((here-> BSIM4v6sNodePrime != 0) && (here-> BSIM4v6qNode != 0))
                here->BSIM4v6SPqPtr = here->BSIM4v6SPqStructPtr->CSC ;

            if ((here-> BSIM4v6gNodePrime != 0) && (here-> BSIM4v6qNode != 0))
                here->BSIM4v6GPqPtr = here->BSIM4v6GPqStructPtr->CSC ;

            if (here->BSIM4v6rgateMod != 0)
            {
                if ((here-> BSIM4v6gNodeExt != 0) && (here-> BSIM4v6gNodeExt != 0))
                    here->BSIM4v6GEgePtr = here->BSIM4v6GEgeStructPtr->CSC ;

                if ((here-> BSIM4v6gNodeExt != 0) && (here-> BSIM4v6gNodePrime != 0))
                    here->BSIM4v6GEgpPtr = here->BSIM4v6GEgpStructPtr->CSC ;

                if ((here-> BSIM4v6gNodePrime != 0) && (here-> BSIM4v6gNodeExt != 0))
                    here->BSIM4v6GPgePtr = here->BSIM4v6GPgeStructPtr->CSC ;

                if ((here-> BSIM4v6gNodeExt != 0) && (here-> BSIM4v6dNodePrime != 0))
                    here->BSIM4v6GEdpPtr = here->BSIM4v6GEdpStructPtr->CSC ;

                if ((here-> BSIM4v6gNodeExt != 0) && (here-> BSIM4v6sNodePrime != 0))
                    here->BSIM4v6GEspPtr = here->BSIM4v6GEspStructPtr->CSC ;

                if ((here-> BSIM4v6gNodeExt != 0) && (here-> BSIM4v6bNodePrime != 0))
                    here->BSIM4v6GEbpPtr = here->BSIM4v6GEbpStructPtr->CSC ;

                if ((here-> BSIM4v6gNodeMid != 0) && (here-> BSIM4v6dNodePrime != 0))
                    here->BSIM4v6GMdpPtr = here->BSIM4v6GMdpStructPtr->CSC ;

                if ((here-> BSIM4v6gNodeMid != 0) && (here-> BSIM4v6gNodePrime != 0))
                    here->BSIM4v6GMgpPtr = here->BSIM4v6GMgpStructPtr->CSC ;

                if ((here-> BSIM4v6gNodeMid != 0) && (here-> BSIM4v6gNodeMid != 0))
                    here->BSIM4v6GMgmPtr = here->BSIM4v6GMgmStructPtr->CSC ;

                if ((here-> BSIM4v6gNodeMid != 0) && (here-> BSIM4v6gNodeExt != 0))
                    here->BSIM4v6GMgePtr = here->BSIM4v6GMgeStructPtr->CSC ;

                if ((here-> BSIM4v6gNodeMid != 0) && (here-> BSIM4v6sNodePrime != 0))
                    here->BSIM4v6GMspPtr = here->BSIM4v6GMspStructPtr->CSC ;

                if ((here-> BSIM4v6gNodeMid != 0) && (here-> BSIM4v6bNodePrime != 0))
                    here->BSIM4v6GMbpPtr = here->BSIM4v6GMbpStructPtr->CSC ;

                if ((here-> BSIM4v6dNodePrime != 0) && (here-> BSIM4v6gNodeMid != 0))
                    here->BSIM4v6DPgmPtr = here->BSIM4v6DPgmStructPtr->CSC ;

                if ((here-> BSIM4v6gNodePrime != 0) && (here-> BSIM4v6gNodeMid != 0))
                    here->BSIM4v6GPgmPtr = here->BSIM4v6GPgmStructPtr->CSC ;

                if ((here-> BSIM4v6gNodeExt != 0) && (here-> BSIM4v6gNodeMid != 0))
                    here->BSIM4v6GEgmPtr = here->BSIM4v6GEgmStructPtr->CSC ;

                if ((here-> BSIM4v6sNodePrime != 0) && (here-> BSIM4v6gNodeMid != 0))
                    here->BSIM4v6SPgmPtr = here->BSIM4v6SPgmStructPtr->CSC ;

                if ((here-> BSIM4v6bNodePrime != 0) && (here-> BSIM4v6gNodeMid != 0))
                    here->BSIM4v6BPgmPtr = here->BSIM4v6BPgmStructPtr->CSC ;

            }
            if ((here->BSIM4v6rbodyMod ==1) || (here->BSIM4v6rbodyMod ==2))
            {
                if ((here-> BSIM4v6dNodePrime != 0) && (here-> BSIM4v6dbNode != 0))
                    here->BSIM4v6DPdbPtr = here->BSIM4v6DPdbStructPtr->CSC ;

                if ((here-> BSIM4v6sNodePrime != 0) && (here-> BSIM4v6sbNode != 0))
                    here->BSIM4v6SPsbPtr = here->BSIM4v6SPsbStructPtr->CSC ;

                if ((here-> BSIM4v6dbNode != 0) && (here-> BSIM4v6dNodePrime != 0))
                    here->BSIM4v6DBdpPtr = here->BSIM4v6DBdpStructPtr->CSC ;

                if ((here-> BSIM4v6dbNode != 0) && (here-> BSIM4v6dbNode != 0))
                    here->BSIM4v6DBdbPtr = here->BSIM4v6DBdbStructPtr->CSC ;

                if ((here-> BSIM4v6dbNode != 0) && (here-> BSIM4v6bNodePrime != 0))
                    here->BSIM4v6DBbpPtr = here->BSIM4v6DBbpStructPtr->CSC ;

                if ((here-> BSIM4v6dbNode != 0) && (here-> BSIM4v6bNode != 0))
                    here->BSIM4v6DBbPtr = here->BSIM4v6DBbStructPtr->CSC ;

                if ((here-> BSIM4v6bNodePrime != 0) && (here-> BSIM4v6dbNode != 0))
                    here->BSIM4v6BPdbPtr = here->BSIM4v6BPdbStructPtr->CSC ;

                if ((here-> BSIM4v6bNodePrime != 0) && (here-> BSIM4v6bNode != 0))
                    here->BSIM4v6BPbPtr = here->BSIM4v6BPbStructPtr->CSC ;

                if ((here-> BSIM4v6bNodePrime != 0) && (here-> BSIM4v6sbNode != 0))
                    here->BSIM4v6BPsbPtr = here->BSIM4v6BPsbStructPtr->CSC ;

                if ((here-> BSIM4v6sbNode != 0) && (here-> BSIM4v6sNodePrime != 0))
                    here->BSIM4v6SBspPtr = here->BSIM4v6SBspStructPtr->CSC ;

                if ((here-> BSIM4v6sbNode != 0) && (here-> BSIM4v6bNodePrime != 0))
                    here->BSIM4v6SBbpPtr = here->BSIM4v6SBbpStructPtr->CSC ;

                if ((here-> BSIM4v6sbNode != 0) && (here-> BSIM4v6bNode != 0))
                    here->BSIM4v6SBbPtr = here->BSIM4v6SBbStructPtr->CSC ;

                if ((here-> BSIM4v6sbNode != 0) && (here-> BSIM4v6sbNode != 0))
                    here->BSIM4v6SBsbPtr = here->BSIM4v6SBsbStructPtr->CSC ;

                if ((here-> BSIM4v6bNode != 0) && (here-> BSIM4v6dbNode != 0))
                    here->BSIM4v6BdbPtr = here->BSIM4v6BdbStructPtr->CSC ;

                if ((here-> BSIM4v6bNode != 0) && (here-> BSIM4v6bNodePrime != 0))
                    here->BSIM4v6BbpPtr = here->BSIM4v6BbpStructPtr->CSC ;

                if ((here-> BSIM4v6bNode != 0) && (here-> BSIM4v6sbNode != 0))
                    here->BSIM4v6BsbPtr = here->BSIM4v6BsbStructPtr->CSC ;

                if ((here-> BSIM4v6bNode != 0) && (here-> BSIM4v6bNode != 0))
                    here->BSIM4v6BbPtr = here->BSIM4v6BbStructPtr->CSC ;

            }
            if (model->BSIM4v6rdsMod)
            {
                if ((here-> BSIM4v6dNode != 0) && (here-> BSIM4v6gNodePrime != 0))
                    here->BSIM4v6DgpPtr = here->BSIM4v6DgpStructPtr->CSC ;

                if ((here-> BSIM4v6dNode != 0) && (here-> BSIM4v6sNodePrime != 0))
                    here->BSIM4v6DspPtr = here->BSIM4v6DspStructPtr->CSC ;

                if ((here-> BSIM4v6dNode != 0) && (here-> BSIM4v6bNodePrime != 0))
                    here->BSIM4v6DbpPtr = here->BSIM4v6DbpStructPtr->CSC ;

                if ((here-> BSIM4v6sNode != 0) && (here-> BSIM4v6dNodePrime != 0))
                    here->BSIM4v6SdpPtr = here->BSIM4v6SdpStructPtr->CSC ;

                if ((here-> BSIM4v6sNode != 0) && (here-> BSIM4v6gNodePrime != 0))
                    here->BSIM4v6SgpPtr = here->BSIM4v6SgpStructPtr->CSC ;

                if ((here-> BSIM4v6sNode != 0) && (here-> BSIM4v6bNodePrime != 0))
                    here->BSIM4v6SbpPtr = here->BSIM4v6SbpStructPtr->CSC ;

            }
        }
    }

    return (OK) ;
}
