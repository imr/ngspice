/**********
Author: 2013 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "bsim4v5def.h"
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
BSIM4v5bindCSC (GENmodel *inModel, CKTcircuit *ckt)
{
    BSIM4v5model *model = (BSIM4v5model *)inModel ;
    BSIM4v5instance *here ;
    double *i ;
    BindElement *matched, *BindStruct ;
    size_t nz ;

    BindStruct = ckt->CKTmatrix->CKTbindStruct ;
    nz = (size_t)ckt->CKTmatrix->CKTklunz ;

    /* loop through all the BSIM4v5 models */
    for ( ; model != NULL ; model = model->BSIM4v5nextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->BSIM4v5instances ; here != NULL ; here = here->BSIM4v5nextInstance)
        {
            if ((here-> BSIM4v5dNodePrime != 0) && (here-> BSIM4v5bNodePrime != 0))
            {
                i = here->BSIM4v5DPbpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4v5DPbpStructPtr = matched ;
                here->BSIM4v5DPbpPtr = matched->CSC ;
            }

            if ((here-> BSIM4v5gNodePrime != 0) && (here-> BSIM4v5bNodePrime != 0))
            {
                i = here->BSIM4v5GPbpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4v5GPbpStructPtr = matched ;
                here->BSIM4v5GPbpPtr = matched->CSC ;
            }

            if ((here-> BSIM4v5sNodePrime != 0) && (here-> BSIM4v5bNodePrime != 0))
            {
                i = here->BSIM4v5SPbpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4v5SPbpStructPtr = matched ;
                here->BSIM4v5SPbpPtr = matched->CSC ;
            }

            if ((here-> BSIM4v5bNodePrime != 0) && (here-> BSIM4v5dNodePrime != 0))
            {
                i = here->BSIM4v5BPdpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4v5BPdpStructPtr = matched ;
                here->BSIM4v5BPdpPtr = matched->CSC ;
            }

            if ((here-> BSIM4v5bNodePrime != 0) && (here-> BSIM4v5gNodePrime != 0))
            {
                i = here->BSIM4v5BPgpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4v5BPgpStructPtr = matched ;
                here->BSIM4v5BPgpPtr = matched->CSC ;
            }

            if ((here-> BSIM4v5bNodePrime != 0) && (here-> BSIM4v5sNodePrime != 0))
            {
                i = here->BSIM4v5BPspPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4v5BPspStructPtr = matched ;
                here->BSIM4v5BPspPtr = matched->CSC ;
            }

            if ((here-> BSIM4v5bNodePrime != 0) && (here-> BSIM4v5bNodePrime != 0))
            {
                i = here->BSIM4v5BPbpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4v5BPbpStructPtr = matched ;
                here->BSIM4v5BPbpPtr = matched->CSC ;
            }

            if ((here-> BSIM4v5dNode != 0) && (here-> BSIM4v5dNode != 0))
            {
                i = here->BSIM4v5DdPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4v5DdStructPtr = matched ;
                here->BSIM4v5DdPtr = matched->CSC ;
            }

            if ((here-> BSIM4v5gNodePrime != 0) && (here-> BSIM4v5gNodePrime != 0))
            {
                i = here->BSIM4v5GPgpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4v5GPgpStructPtr = matched ;
                here->BSIM4v5GPgpPtr = matched->CSC ;
            }

            if ((here-> BSIM4v5sNode != 0) && (here-> BSIM4v5sNode != 0))
            {
                i = here->BSIM4v5SsPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4v5SsStructPtr = matched ;
                here->BSIM4v5SsPtr = matched->CSC ;
            }

            if ((here-> BSIM4v5dNodePrime != 0) && (here-> BSIM4v5dNodePrime != 0))
            {
                i = here->BSIM4v5DPdpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4v5DPdpStructPtr = matched ;
                here->BSIM4v5DPdpPtr = matched->CSC ;
            }

            if ((here-> BSIM4v5sNodePrime != 0) && (here-> BSIM4v5sNodePrime != 0))
            {
                i = here->BSIM4v5SPspPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4v5SPspStructPtr = matched ;
                here->BSIM4v5SPspPtr = matched->CSC ;
            }

            if ((here-> BSIM4v5dNode != 0) && (here-> BSIM4v5dNodePrime != 0))
            {
                i = here->BSIM4v5DdpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4v5DdpStructPtr = matched ;
                here->BSIM4v5DdpPtr = matched->CSC ;
            }

            if ((here-> BSIM4v5gNodePrime != 0) && (here-> BSIM4v5dNodePrime != 0))
            {
                i = here->BSIM4v5GPdpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4v5GPdpStructPtr = matched ;
                here->BSIM4v5GPdpPtr = matched->CSC ;
            }

            if ((here-> BSIM4v5gNodePrime != 0) && (here-> BSIM4v5sNodePrime != 0))
            {
                i = here->BSIM4v5GPspPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4v5GPspStructPtr = matched ;
                here->BSIM4v5GPspPtr = matched->CSC ;
            }

            if ((here-> BSIM4v5sNode != 0) && (here-> BSIM4v5sNodePrime != 0))
            {
                i = here->BSIM4v5SspPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4v5SspStructPtr = matched ;
                here->BSIM4v5SspPtr = matched->CSC ;
            }

            if ((here-> BSIM4v5dNodePrime != 0) && (here-> BSIM4v5sNodePrime != 0))
            {
                i = here->BSIM4v5DPspPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4v5DPspStructPtr = matched ;
                here->BSIM4v5DPspPtr = matched->CSC ;
            }

            if ((here-> BSIM4v5dNodePrime != 0) && (here-> BSIM4v5dNode != 0))
            {
                i = here->BSIM4v5DPdPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4v5DPdStructPtr = matched ;
                here->BSIM4v5DPdPtr = matched->CSC ;
            }

            if ((here-> BSIM4v5dNodePrime != 0) && (here-> BSIM4v5gNodePrime != 0))
            {
                i = here->BSIM4v5DPgpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4v5DPgpStructPtr = matched ;
                here->BSIM4v5DPgpPtr = matched->CSC ;
            }

            if ((here-> BSIM4v5sNodePrime != 0) && (here-> BSIM4v5gNodePrime != 0))
            {
                i = here->BSIM4v5SPgpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4v5SPgpStructPtr = matched ;
                here->BSIM4v5SPgpPtr = matched->CSC ;
            }

            if ((here-> BSIM4v5sNodePrime != 0) && (here-> BSIM4v5sNode != 0))
            {
                i = here->BSIM4v5SPsPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4v5SPsStructPtr = matched ;
                here->BSIM4v5SPsPtr = matched->CSC ;
            }

            if ((here-> BSIM4v5sNodePrime != 0) && (here-> BSIM4v5dNodePrime != 0))
            {
                i = here->BSIM4v5SPdpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4v5SPdpStructPtr = matched ;
                here->BSIM4v5SPdpPtr = matched->CSC ;
            }

            if ((here-> BSIM4v5qNode != 0) && (here-> BSIM4v5qNode != 0))
            {
                i = here->BSIM4v5QqPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4v5QqStructPtr = matched ;
                here->BSIM4v5QqPtr = matched->CSC ;
            }

            if ((here-> BSIM4v5qNode != 0) && (here-> BSIM4v5bNodePrime != 0))
            {
                i = here->BSIM4v5QbpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4v5QbpStructPtr = matched ;
                here->BSIM4v5QbpPtr = matched->CSC ;
            }

            if ((here-> BSIM4v5qNode != 0) && (here-> BSIM4v5dNodePrime != 0))
            {
                i = here->BSIM4v5QdpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4v5QdpStructPtr = matched ;
                here->BSIM4v5QdpPtr = matched->CSC ;
            }

            if ((here-> BSIM4v5qNode != 0) && (here-> BSIM4v5sNodePrime != 0))
            {
                i = here->BSIM4v5QspPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4v5QspStructPtr = matched ;
                here->BSIM4v5QspPtr = matched->CSC ;
            }

            if ((here-> BSIM4v5qNode != 0) && (here-> BSIM4v5gNodePrime != 0))
            {
                i = here->BSIM4v5QgpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4v5QgpStructPtr = matched ;
                here->BSIM4v5QgpPtr = matched->CSC ;
            }

            if ((here-> BSIM4v5dNodePrime != 0) && (here-> BSIM4v5qNode != 0))
            {
                i = here->BSIM4v5DPqPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4v5DPqStructPtr = matched ;
                here->BSIM4v5DPqPtr = matched->CSC ;
            }

            if ((here-> BSIM4v5sNodePrime != 0) && (here-> BSIM4v5qNode != 0))
            {
                i = here->BSIM4v5SPqPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4v5SPqStructPtr = matched ;
                here->BSIM4v5SPqPtr = matched->CSC ;
            }

            if ((here-> BSIM4v5gNodePrime != 0) && (here-> BSIM4v5qNode != 0))
            {
                i = here->BSIM4v5GPqPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4v5GPqStructPtr = matched ;
                here->BSIM4v5GPqPtr = matched->CSC ;
            }

            if (here->BSIM4v5rgateMod != 0)
            {
                if ((here-> BSIM4v5gNodeExt != 0) && (here-> BSIM4v5gNodeExt != 0))
                {
                    i = here->BSIM4v5GEgePtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v5GEgeStructPtr = matched ;
                    here->BSIM4v5GEgePtr = matched->CSC ;
                }

                if ((here-> BSIM4v5gNodeExt != 0) && (here-> BSIM4v5gNodePrime != 0))
                {
                    i = here->BSIM4v5GEgpPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v5GEgpStructPtr = matched ;
                    here->BSIM4v5GEgpPtr = matched->CSC ;
                }

                if ((here-> BSIM4v5gNodePrime != 0) && (here-> BSIM4v5gNodeExt != 0))
                {
                    i = here->BSIM4v5GPgePtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v5GPgeStructPtr = matched ;
                    here->BSIM4v5GPgePtr = matched->CSC ;
                }

                if ((here-> BSIM4v5gNodeExt != 0) && (here-> BSIM4v5dNodePrime != 0))
                {
                    i = here->BSIM4v5GEdpPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v5GEdpStructPtr = matched ;
                    here->BSIM4v5GEdpPtr = matched->CSC ;
                }

                if ((here-> BSIM4v5gNodeExt != 0) && (here-> BSIM4v5sNodePrime != 0))
                {
                    i = here->BSIM4v5GEspPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v5GEspStructPtr = matched ;
                    here->BSIM4v5GEspPtr = matched->CSC ;
                }

                if ((here-> BSIM4v5gNodeExt != 0) && (here-> BSIM4v5bNodePrime != 0))
                {
                    i = here->BSIM4v5GEbpPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v5GEbpStructPtr = matched ;
                    here->BSIM4v5GEbpPtr = matched->CSC ;
                }

                if ((here-> BSIM4v5gNodeMid != 0) && (here-> BSIM4v5dNodePrime != 0))
                {
                    i = here->BSIM4v5GMdpPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v5GMdpStructPtr = matched ;
                    here->BSIM4v5GMdpPtr = matched->CSC ;
                }

                if ((here-> BSIM4v5gNodeMid != 0) && (here-> BSIM4v5gNodePrime != 0))
                {
                    i = here->BSIM4v5GMgpPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v5GMgpStructPtr = matched ;
                    here->BSIM4v5GMgpPtr = matched->CSC ;
                }

                if ((here-> BSIM4v5gNodeMid != 0) && (here-> BSIM4v5gNodeMid != 0))
                {
                    i = here->BSIM4v5GMgmPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v5GMgmStructPtr = matched ;
                    here->BSIM4v5GMgmPtr = matched->CSC ;
                }

                if ((here-> BSIM4v5gNodeMid != 0) && (here-> BSIM4v5gNodeExt != 0))
                {
                    i = here->BSIM4v5GMgePtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v5GMgeStructPtr = matched ;
                    here->BSIM4v5GMgePtr = matched->CSC ;
                }

                if ((here-> BSIM4v5gNodeMid != 0) && (here-> BSIM4v5sNodePrime != 0))
                {
                    i = here->BSIM4v5GMspPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v5GMspStructPtr = matched ;
                    here->BSIM4v5GMspPtr = matched->CSC ;
                }

                if ((here-> BSIM4v5gNodeMid != 0) && (here-> BSIM4v5bNodePrime != 0))
                {
                    i = here->BSIM4v5GMbpPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v5GMbpStructPtr = matched ;
                    here->BSIM4v5GMbpPtr = matched->CSC ;
                }

                if ((here-> BSIM4v5dNodePrime != 0) && (here-> BSIM4v5gNodeMid != 0))
                {
                    i = here->BSIM4v5DPgmPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v5DPgmStructPtr = matched ;
                    here->BSIM4v5DPgmPtr = matched->CSC ;
                }

                if ((here-> BSIM4v5gNodePrime != 0) && (here-> BSIM4v5gNodeMid != 0))
                {
                    i = here->BSIM4v5GPgmPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v5GPgmStructPtr = matched ;
                    here->BSIM4v5GPgmPtr = matched->CSC ;
                }

                if ((here-> BSIM4v5gNodeExt != 0) && (here-> BSIM4v5gNodeMid != 0))
                {
                    i = here->BSIM4v5GEgmPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v5GEgmStructPtr = matched ;
                    here->BSIM4v5GEgmPtr = matched->CSC ;
                }

                if ((here-> BSIM4v5sNodePrime != 0) && (here-> BSIM4v5gNodeMid != 0))
                {
                    i = here->BSIM4v5SPgmPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v5SPgmStructPtr = matched ;
                    here->BSIM4v5SPgmPtr = matched->CSC ;
                }

                if ((here-> BSIM4v5bNodePrime != 0) && (here-> BSIM4v5gNodeMid != 0))
                {
                    i = here->BSIM4v5BPgmPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v5BPgmStructPtr = matched ;
                    here->BSIM4v5BPgmPtr = matched->CSC ;
                }

            }
            if ((here->BSIM4v5rbodyMod == 1) || (here->BSIM4v5rbodyMod == 2))
            {
                if ((here-> BSIM4v5dNodePrime != 0) && (here-> BSIM4v5dbNode != 0))
                {
                    i = here->BSIM4v5DPdbPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v5DPdbStructPtr = matched ;
                    here->BSIM4v5DPdbPtr = matched->CSC ;
                }

                if ((here-> BSIM4v5sNodePrime != 0) && (here-> BSIM4v5sbNode != 0))
                {
                    i = here->BSIM4v5SPsbPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v5SPsbStructPtr = matched ;
                    here->BSIM4v5SPsbPtr = matched->CSC ;
                }

                if ((here-> BSIM4v5dbNode != 0) && (here-> BSIM4v5dNodePrime != 0))
                {
                    i = here->BSIM4v5DBdpPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v5DBdpStructPtr = matched ;
                    here->BSIM4v5DBdpPtr = matched->CSC ;
                }

                if ((here-> BSIM4v5dbNode != 0) && (here-> BSIM4v5dbNode != 0))
                {
                    i = here->BSIM4v5DBdbPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v5DBdbStructPtr = matched ;
                    here->BSIM4v5DBdbPtr = matched->CSC ;
                }

                if ((here-> BSIM4v5dbNode != 0) && (here-> BSIM4v5bNodePrime != 0))
                {
                    i = here->BSIM4v5DBbpPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v5DBbpStructPtr = matched ;
                    here->BSIM4v5DBbpPtr = matched->CSC ;
                }

                if ((here-> BSIM4v5dbNode != 0) && (here-> BSIM4v5bNode != 0))
                {
                    i = here->BSIM4v5DBbPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v5DBbStructPtr = matched ;
                    here->BSIM4v5DBbPtr = matched->CSC ;
                }

                if ((here-> BSIM4v5bNodePrime != 0) && (here-> BSIM4v5dbNode != 0))
                {
                    i = here->BSIM4v5BPdbPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v5BPdbStructPtr = matched ;
                    here->BSIM4v5BPdbPtr = matched->CSC ;
                }

                if ((here-> BSIM4v5bNodePrime != 0) && (here-> BSIM4v5bNode != 0))
                {
                    i = here->BSIM4v5BPbPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v5BPbStructPtr = matched ;
                    here->BSIM4v5BPbPtr = matched->CSC ;
                }

                if ((here-> BSIM4v5bNodePrime != 0) && (here-> BSIM4v5sbNode != 0))
                {
                    i = here->BSIM4v5BPsbPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v5BPsbStructPtr = matched ;
                    here->BSIM4v5BPsbPtr = matched->CSC ;
                }

                if ((here-> BSIM4v5sbNode != 0) && (here-> BSIM4v5sNodePrime != 0))
                {
                    i = here->BSIM4v5SBspPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v5SBspStructPtr = matched ;
                    here->BSIM4v5SBspPtr = matched->CSC ;
                }

                if ((here-> BSIM4v5sbNode != 0) && (here-> BSIM4v5bNodePrime != 0))
                {
                    i = here->BSIM4v5SBbpPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v5SBbpStructPtr = matched ;
                    here->BSIM4v5SBbpPtr = matched->CSC ;
                }

                if ((here-> BSIM4v5sbNode != 0) && (here-> BSIM4v5bNode != 0))
                {
                    i = here->BSIM4v5SBbPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v5SBbStructPtr = matched ;
                    here->BSIM4v5SBbPtr = matched->CSC ;
                }

                if ((here-> BSIM4v5sbNode != 0) && (here-> BSIM4v5sbNode != 0))
                {
                    i = here->BSIM4v5SBsbPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v5SBsbStructPtr = matched ;
                    here->BSIM4v5SBsbPtr = matched->CSC ;
                }

                if ((here-> BSIM4v5bNode != 0) && (here-> BSIM4v5dbNode != 0))
                {
                    i = here->BSIM4v5BdbPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v5BdbStructPtr = matched ;
                    here->BSIM4v5BdbPtr = matched->CSC ;
                }

                if ((here-> BSIM4v5bNode != 0) && (here-> BSIM4v5bNodePrime != 0))
                {
                    i = here->BSIM4v5BbpPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v5BbpStructPtr = matched ;
                    here->BSIM4v5BbpPtr = matched->CSC ;
                }

                if ((here-> BSIM4v5bNode != 0) && (here-> BSIM4v5sbNode != 0))
                {
                    i = here->BSIM4v5BsbPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v5BsbStructPtr = matched ;
                    here->BSIM4v5BsbPtr = matched->CSC ;
                }

                if ((here-> BSIM4v5bNode != 0) && (here-> BSIM4v5bNode != 0))
                {
                    i = here->BSIM4v5BbPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v5BbStructPtr = matched ;
                    here->BSIM4v5BbPtr = matched->CSC ;
                }

            }
            if (model->BSIM4v5rdsMod)
            {
                if ((here-> BSIM4v5dNode != 0) && (here-> BSIM4v5gNodePrime != 0))
                {
                    i = here->BSIM4v5DgpPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v5DgpStructPtr = matched ;
                    here->BSIM4v5DgpPtr = matched->CSC ;
                }

                if ((here-> BSIM4v5dNode != 0) && (here-> BSIM4v5sNodePrime != 0))
                {
                    i = here->BSIM4v5DspPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v5DspStructPtr = matched ;
                    here->BSIM4v5DspPtr = matched->CSC ;
                }

                if ((here-> BSIM4v5dNode != 0) && (here-> BSIM4v5bNodePrime != 0))
                {
                    i = here->BSIM4v5DbpPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v5DbpStructPtr = matched ;
                    here->BSIM4v5DbpPtr = matched->CSC ;
                }

                if ((here-> BSIM4v5sNode != 0) && (here-> BSIM4v5dNodePrime != 0))
                {
                    i = here->BSIM4v5SdpPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v5SdpStructPtr = matched ;
                    here->BSIM4v5SdpPtr = matched->CSC ;
                }

                if ((here-> BSIM4v5sNode != 0) && (here-> BSIM4v5gNodePrime != 0))
                {
                    i = here->BSIM4v5SgpPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v5SgpStructPtr = matched ;
                    here->BSIM4v5SgpPtr = matched->CSC ;
                }

                if ((here-> BSIM4v5sNode != 0) && (here-> BSIM4v5bNodePrime != 0))
                {
                    i = here->BSIM4v5SbpPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v5SbpStructPtr = matched ;
                    here->BSIM4v5SbpPtr = matched->CSC ;
                }

            }
        }
    }

    return (OK) ;
}

int
BSIM4v5bindCSCComplex (GENmodel *inModel, CKTcircuit *ckt)
{
    BSIM4v5model *model = (BSIM4v5model *)inModel ;
    BSIM4v5instance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the BSIM4v5 models */
    for ( ; model != NULL ; model = model->BSIM4v5nextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->BSIM4v5instances ; here != NULL ; here = here->BSIM4v5nextInstance)
        {
            if ((here-> BSIM4v5dNodePrime != 0) && (here-> BSIM4v5bNodePrime != 0))
                here->BSIM4v5DPbpPtr = here->BSIM4v5DPbpStructPtr->CSC_Complex ;

            if ((here-> BSIM4v5gNodePrime != 0) && (here-> BSIM4v5bNodePrime != 0))
                here->BSIM4v5GPbpPtr = here->BSIM4v5GPbpStructPtr->CSC_Complex ;

            if ((here-> BSIM4v5sNodePrime != 0) && (here-> BSIM4v5bNodePrime != 0))
                here->BSIM4v5SPbpPtr = here->BSIM4v5SPbpStructPtr->CSC_Complex ;

            if ((here-> BSIM4v5bNodePrime != 0) && (here-> BSIM4v5dNodePrime != 0))
                here->BSIM4v5BPdpPtr = here->BSIM4v5BPdpStructPtr->CSC_Complex ;

            if ((here-> BSIM4v5bNodePrime != 0) && (here-> BSIM4v5gNodePrime != 0))
                here->BSIM4v5BPgpPtr = here->BSIM4v5BPgpStructPtr->CSC_Complex ;

            if ((here-> BSIM4v5bNodePrime != 0) && (here-> BSIM4v5sNodePrime != 0))
                here->BSIM4v5BPspPtr = here->BSIM4v5BPspStructPtr->CSC_Complex ;

            if ((here-> BSIM4v5bNodePrime != 0) && (here-> BSIM4v5bNodePrime != 0))
                here->BSIM4v5BPbpPtr = here->BSIM4v5BPbpStructPtr->CSC_Complex ;

            if ((here-> BSIM4v5dNode != 0) && (here-> BSIM4v5dNode != 0))
                here->BSIM4v5DdPtr = here->BSIM4v5DdStructPtr->CSC_Complex ;

            if ((here-> BSIM4v5gNodePrime != 0) && (here-> BSIM4v5gNodePrime != 0))
                here->BSIM4v5GPgpPtr = here->BSIM4v5GPgpStructPtr->CSC_Complex ;

            if ((here-> BSIM4v5sNode != 0) && (here-> BSIM4v5sNode != 0))
                here->BSIM4v5SsPtr = here->BSIM4v5SsStructPtr->CSC_Complex ;

            if ((here-> BSIM4v5dNodePrime != 0) && (here-> BSIM4v5dNodePrime != 0))
                here->BSIM4v5DPdpPtr = here->BSIM4v5DPdpStructPtr->CSC_Complex ;

            if ((here-> BSIM4v5sNodePrime != 0) && (here-> BSIM4v5sNodePrime != 0))
                here->BSIM4v5SPspPtr = here->BSIM4v5SPspStructPtr->CSC_Complex ;

            if ((here-> BSIM4v5dNode != 0) && (here-> BSIM4v5dNodePrime != 0))
                here->BSIM4v5DdpPtr = here->BSIM4v5DdpStructPtr->CSC_Complex ;

            if ((here-> BSIM4v5gNodePrime != 0) && (here-> BSIM4v5dNodePrime != 0))
                here->BSIM4v5GPdpPtr = here->BSIM4v5GPdpStructPtr->CSC_Complex ;

            if ((here-> BSIM4v5gNodePrime != 0) && (here-> BSIM4v5sNodePrime != 0))
                here->BSIM4v5GPspPtr = here->BSIM4v5GPspStructPtr->CSC_Complex ;

            if ((here-> BSIM4v5sNode != 0) && (here-> BSIM4v5sNodePrime != 0))
                here->BSIM4v5SspPtr = here->BSIM4v5SspStructPtr->CSC_Complex ;

            if ((here-> BSIM4v5dNodePrime != 0) && (here-> BSIM4v5sNodePrime != 0))
                here->BSIM4v5DPspPtr = here->BSIM4v5DPspStructPtr->CSC_Complex ;

            if ((here-> BSIM4v5dNodePrime != 0) && (here-> BSIM4v5dNode != 0))
                here->BSIM4v5DPdPtr = here->BSIM4v5DPdStructPtr->CSC_Complex ;

            if ((here-> BSIM4v5dNodePrime != 0) && (here-> BSIM4v5gNodePrime != 0))
                here->BSIM4v5DPgpPtr = here->BSIM4v5DPgpStructPtr->CSC_Complex ;

            if ((here-> BSIM4v5sNodePrime != 0) && (here-> BSIM4v5gNodePrime != 0))
                here->BSIM4v5SPgpPtr = here->BSIM4v5SPgpStructPtr->CSC_Complex ;

            if ((here-> BSIM4v5sNodePrime != 0) && (here-> BSIM4v5sNode != 0))
                here->BSIM4v5SPsPtr = here->BSIM4v5SPsStructPtr->CSC_Complex ;

            if ((here-> BSIM4v5sNodePrime != 0) && (here-> BSIM4v5dNodePrime != 0))
                here->BSIM4v5SPdpPtr = here->BSIM4v5SPdpStructPtr->CSC_Complex ;

            if ((here-> BSIM4v5qNode != 0) && (here-> BSIM4v5qNode != 0))
                here->BSIM4v5QqPtr = here->BSIM4v5QqStructPtr->CSC_Complex ;

            if ((here-> BSIM4v5qNode != 0) && (here-> BSIM4v5bNodePrime != 0))
                here->BSIM4v5QbpPtr = here->BSIM4v5QbpStructPtr->CSC_Complex ;

            if ((here-> BSIM4v5qNode != 0) && (here-> BSIM4v5dNodePrime != 0))
                here->BSIM4v5QdpPtr = here->BSIM4v5QdpStructPtr->CSC_Complex ;

            if ((here-> BSIM4v5qNode != 0) && (here-> BSIM4v5sNodePrime != 0))
                here->BSIM4v5QspPtr = here->BSIM4v5QspStructPtr->CSC_Complex ;

            if ((here-> BSIM4v5qNode != 0) && (here-> BSIM4v5gNodePrime != 0))
                here->BSIM4v5QgpPtr = here->BSIM4v5QgpStructPtr->CSC_Complex ;

            if ((here-> BSIM4v5dNodePrime != 0) && (here-> BSIM4v5qNode != 0))
                here->BSIM4v5DPqPtr = here->BSIM4v5DPqStructPtr->CSC_Complex ;

            if ((here-> BSIM4v5sNodePrime != 0) && (here-> BSIM4v5qNode != 0))
                here->BSIM4v5SPqPtr = here->BSIM4v5SPqStructPtr->CSC_Complex ;

            if ((here-> BSIM4v5gNodePrime != 0) && (here-> BSIM4v5qNode != 0))
                here->BSIM4v5GPqPtr = here->BSIM4v5GPqStructPtr->CSC_Complex ;

            if (here->BSIM4v5rgateMod != 0)
            {
                if ((here-> BSIM4v5gNodeExt != 0) && (here-> BSIM4v5gNodeExt != 0))
                    here->BSIM4v5GEgePtr = here->BSIM4v5GEgeStructPtr->CSC_Complex ;

                if ((here-> BSIM4v5gNodeExt != 0) && (here-> BSIM4v5gNodePrime != 0))
                    here->BSIM4v5GEgpPtr = here->BSIM4v5GEgpStructPtr->CSC_Complex ;

                if ((here-> BSIM4v5gNodePrime != 0) && (here-> BSIM4v5gNodeExt != 0))
                    here->BSIM4v5GPgePtr = here->BSIM4v5GPgeStructPtr->CSC_Complex ;

                if ((here-> BSIM4v5gNodeExt != 0) && (here-> BSIM4v5dNodePrime != 0))
                    here->BSIM4v5GEdpPtr = here->BSIM4v5GEdpStructPtr->CSC_Complex ;

                if ((here-> BSIM4v5gNodeExt != 0) && (here-> BSIM4v5sNodePrime != 0))
                    here->BSIM4v5GEspPtr = here->BSIM4v5GEspStructPtr->CSC_Complex ;

                if ((here-> BSIM4v5gNodeExt != 0) && (here-> BSIM4v5bNodePrime != 0))
                    here->BSIM4v5GEbpPtr = here->BSIM4v5GEbpStructPtr->CSC_Complex ;

                if ((here-> BSIM4v5gNodeMid != 0) && (here-> BSIM4v5dNodePrime != 0))
                    here->BSIM4v5GMdpPtr = here->BSIM4v5GMdpStructPtr->CSC_Complex ;

                if ((here-> BSIM4v5gNodeMid != 0) && (here-> BSIM4v5gNodePrime != 0))
                    here->BSIM4v5GMgpPtr = here->BSIM4v5GMgpStructPtr->CSC_Complex ;

                if ((here-> BSIM4v5gNodeMid != 0) && (here-> BSIM4v5gNodeMid != 0))
                    here->BSIM4v5GMgmPtr = here->BSIM4v5GMgmStructPtr->CSC_Complex ;

                if ((here-> BSIM4v5gNodeMid != 0) && (here-> BSIM4v5gNodeExt != 0))
                    here->BSIM4v5GMgePtr = here->BSIM4v5GMgeStructPtr->CSC_Complex ;

                if ((here-> BSIM4v5gNodeMid != 0) && (here-> BSIM4v5sNodePrime != 0))
                    here->BSIM4v5GMspPtr = here->BSIM4v5GMspStructPtr->CSC_Complex ;

                if ((here-> BSIM4v5gNodeMid != 0) && (here-> BSIM4v5bNodePrime != 0))
                    here->BSIM4v5GMbpPtr = here->BSIM4v5GMbpStructPtr->CSC_Complex ;

                if ((here-> BSIM4v5dNodePrime != 0) && (here-> BSIM4v5gNodeMid != 0))
                    here->BSIM4v5DPgmPtr = here->BSIM4v5DPgmStructPtr->CSC_Complex ;

                if ((here-> BSIM4v5gNodePrime != 0) && (here-> BSIM4v5gNodeMid != 0))
                    here->BSIM4v5GPgmPtr = here->BSIM4v5GPgmStructPtr->CSC_Complex ;

                if ((here-> BSIM4v5gNodeExt != 0) && (here-> BSIM4v5gNodeMid != 0))
                    here->BSIM4v5GEgmPtr = here->BSIM4v5GEgmStructPtr->CSC_Complex ;

                if ((here-> BSIM4v5sNodePrime != 0) && (here-> BSIM4v5gNodeMid != 0))
                    here->BSIM4v5SPgmPtr = here->BSIM4v5SPgmStructPtr->CSC_Complex ;

                if ((here-> BSIM4v5bNodePrime != 0) && (here-> BSIM4v5gNodeMid != 0))
                    here->BSIM4v5BPgmPtr = here->BSIM4v5BPgmStructPtr->CSC_Complex ;

            }
            if ((here->BSIM4v5rbodyMod == 1) || (here->BSIM4v5rbodyMod == 2))
            {
                if ((here-> BSIM4v5dNodePrime != 0) && (here-> BSIM4v5dbNode != 0))
                    here->BSIM4v5DPdbPtr = here->BSIM4v5DPdbStructPtr->CSC_Complex ;

                if ((here-> BSIM4v5sNodePrime != 0) && (here-> BSIM4v5sbNode != 0))
                    here->BSIM4v5SPsbPtr = here->BSIM4v5SPsbStructPtr->CSC_Complex ;

                if ((here-> BSIM4v5dbNode != 0) && (here-> BSIM4v5dNodePrime != 0))
                    here->BSIM4v5DBdpPtr = here->BSIM4v5DBdpStructPtr->CSC_Complex ;

                if ((here-> BSIM4v5dbNode != 0) && (here-> BSIM4v5dbNode != 0))
                    here->BSIM4v5DBdbPtr = here->BSIM4v5DBdbStructPtr->CSC_Complex ;

                if ((here-> BSIM4v5dbNode != 0) && (here-> BSIM4v5bNodePrime != 0))
                    here->BSIM4v5DBbpPtr = here->BSIM4v5DBbpStructPtr->CSC_Complex ;

                if ((here-> BSIM4v5dbNode != 0) && (here-> BSIM4v5bNode != 0))
                    here->BSIM4v5DBbPtr = here->BSIM4v5DBbStructPtr->CSC_Complex ;

                if ((here-> BSIM4v5bNodePrime != 0) && (here-> BSIM4v5dbNode != 0))
                    here->BSIM4v5BPdbPtr = here->BSIM4v5BPdbStructPtr->CSC_Complex ;

                if ((here-> BSIM4v5bNodePrime != 0) && (here-> BSIM4v5bNode != 0))
                    here->BSIM4v5BPbPtr = here->BSIM4v5BPbStructPtr->CSC_Complex ;

                if ((here-> BSIM4v5bNodePrime != 0) && (here-> BSIM4v5sbNode != 0))
                    here->BSIM4v5BPsbPtr = here->BSIM4v5BPsbStructPtr->CSC_Complex ;

                if ((here-> BSIM4v5sbNode != 0) && (here-> BSIM4v5sNodePrime != 0))
                    here->BSIM4v5SBspPtr = here->BSIM4v5SBspStructPtr->CSC_Complex ;

                if ((here-> BSIM4v5sbNode != 0) && (here-> BSIM4v5bNodePrime != 0))
                    here->BSIM4v5SBbpPtr = here->BSIM4v5SBbpStructPtr->CSC_Complex ;

                if ((here-> BSIM4v5sbNode != 0) && (here-> BSIM4v5bNode != 0))
                    here->BSIM4v5SBbPtr = here->BSIM4v5SBbStructPtr->CSC_Complex ;

                if ((here-> BSIM4v5sbNode != 0) && (here-> BSIM4v5sbNode != 0))
                    here->BSIM4v5SBsbPtr = here->BSIM4v5SBsbStructPtr->CSC_Complex ;

                if ((here-> BSIM4v5bNode != 0) && (here-> BSIM4v5dbNode != 0))
                    here->BSIM4v5BdbPtr = here->BSIM4v5BdbStructPtr->CSC_Complex ;

                if ((here-> BSIM4v5bNode != 0) && (here-> BSIM4v5bNodePrime != 0))
                    here->BSIM4v5BbpPtr = here->BSIM4v5BbpStructPtr->CSC_Complex ;

                if ((here-> BSIM4v5bNode != 0) && (here-> BSIM4v5sbNode != 0))
                    here->BSIM4v5BsbPtr = here->BSIM4v5BsbStructPtr->CSC_Complex ;

                if ((here-> BSIM4v5bNode != 0) && (here-> BSIM4v5bNode != 0))
                    here->BSIM4v5BbPtr = here->BSIM4v5BbStructPtr->CSC_Complex ;

            }
            if (model->BSIM4v5rdsMod)
            {
                if ((here-> BSIM4v5dNode != 0) && (here-> BSIM4v5gNodePrime != 0))
                    here->BSIM4v5DgpPtr = here->BSIM4v5DgpStructPtr->CSC_Complex ;

                if ((here-> BSIM4v5dNode != 0) && (here-> BSIM4v5sNodePrime != 0))
                    here->BSIM4v5DspPtr = here->BSIM4v5DspStructPtr->CSC_Complex ;

                if ((here-> BSIM4v5dNode != 0) && (here-> BSIM4v5bNodePrime != 0))
                    here->BSIM4v5DbpPtr = here->BSIM4v5DbpStructPtr->CSC_Complex ;

                if ((here-> BSIM4v5sNode != 0) && (here-> BSIM4v5dNodePrime != 0))
                    here->BSIM4v5SdpPtr = here->BSIM4v5SdpStructPtr->CSC_Complex ;

                if ((here-> BSIM4v5sNode != 0) && (here-> BSIM4v5gNodePrime != 0))
                    here->BSIM4v5SgpPtr = here->BSIM4v5SgpStructPtr->CSC_Complex ;

                if ((here-> BSIM4v5sNode != 0) && (here-> BSIM4v5bNodePrime != 0))
                    here->BSIM4v5SbpPtr = here->BSIM4v5SbpStructPtr->CSC_Complex ;

            }
        }
    }

    return (OK) ;
}

int
BSIM4v5bindCSCComplexToReal (GENmodel *inModel, CKTcircuit *ckt)
{
    BSIM4v5model *model = (BSIM4v5model *)inModel ;
    BSIM4v5instance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the BSIM4v5 models */
    for ( ; model != NULL ; model = model->BSIM4v5nextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->BSIM4v5instances ; here != NULL ; here = here->BSIM4v5nextInstance)
        {
            if ((here-> BSIM4v5dNodePrime != 0) && (here-> BSIM4v5bNodePrime != 0))
                here->BSIM4v5DPbpPtr = here->BSIM4v5DPbpStructPtr->CSC ;

            if ((here-> BSIM4v5gNodePrime != 0) && (here-> BSIM4v5bNodePrime != 0))
                here->BSIM4v5GPbpPtr = here->BSIM4v5GPbpStructPtr->CSC ;

            if ((here-> BSIM4v5sNodePrime != 0) && (here-> BSIM4v5bNodePrime != 0))
                here->BSIM4v5SPbpPtr = here->BSIM4v5SPbpStructPtr->CSC ;

            if ((here-> BSIM4v5bNodePrime != 0) && (here-> BSIM4v5dNodePrime != 0))
                here->BSIM4v5BPdpPtr = here->BSIM4v5BPdpStructPtr->CSC ;

            if ((here-> BSIM4v5bNodePrime != 0) && (here-> BSIM4v5gNodePrime != 0))
                here->BSIM4v5BPgpPtr = here->BSIM4v5BPgpStructPtr->CSC ;

            if ((here-> BSIM4v5bNodePrime != 0) && (here-> BSIM4v5sNodePrime != 0))
                here->BSIM4v5BPspPtr = here->BSIM4v5BPspStructPtr->CSC ;

            if ((here-> BSIM4v5bNodePrime != 0) && (here-> BSIM4v5bNodePrime != 0))
                here->BSIM4v5BPbpPtr = here->BSIM4v5BPbpStructPtr->CSC ;

            if ((here-> BSIM4v5dNode != 0) && (here-> BSIM4v5dNode != 0))
                here->BSIM4v5DdPtr = here->BSIM4v5DdStructPtr->CSC ;

            if ((here-> BSIM4v5gNodePrime != 0) && (here-> BSIM4v5gNodePrime != 0))
                here->BSIM4v5GPgpPtr = here->BSIM4v5GPgpStructPtr->CSC ;

            if ((here-> BSIM4v5sNode != 0) && (here-> BSIM4v5sNode != 0))
                here->BSIM4v5SsPtr = here->BSIM4v5SsStructPtr->CSC ;

            if ((here-> BSIM4v5dNodePrime != 0) && (here-> BSIM4v5dNodePrime != 0))
                here->BSIM4v5DPdpPtr = here->BSIM4v5DPdpStructPtr->CSC ;

            if ((here-> BSIM4v5sNodePrime != 0) && (here-> BSIM4v5sNodePrime != 0))
                here->BSIM4v5SPspPtr = here->BSIM4v5SPspStructPtr->CSC ;

            if ((here-> BSIM4v5dNode != 0) && (here-> BSIM4v5dNodePrime != 0))
                here->BSIM4v5DdpPtr = here->BSIM4v5DdpStructPtr->CSC ;

            if ((here-> BSIM4v5gNodePrime != 0) && (here-> BSIM4v5dNodePrime != 0))
                here->BSIM4v5GPdpPtr = here->BSIM4v5GPdpStructPtr->CSC ;

            if ((here-> BSIM4v5gNodePrime != 0) && (here-> BSIM4v5sNodePrime != 0))
                here->BSIM4v5GPspPtr = here->BSIM4v5GPspStructPtr->CSC ;

            if ((here-> BSIM4v5sNode != 0) && (here-> BSIM4v5sNodePrime != 0))
                here->BSIM4v5SspPtr = here->BSIM4v5SspStructPtr->CSC ;

            if ((here-> BSIM4v5dNodePrime != 0) && (here-> BSIM4v5sNodePrime != 0))
                here->BSIM4v5DPspPtr = here->BSIM4v5DPspStructPtr->CSC ;

            if ((here-> BSIM4v5dNodePrime != 0) && (here-> BSIM4v5dNode != 0))
                here->BSIM4v5DPdPtr = here->BSIM4v5DPdStructPtr->CSC ;

            if ((here-> BSIM4v5dNodePrime != 0) && (here-> BSIM4v5gNodePrime != 0))
                here->BSIM4v5DPgpPtr = here->BSIM4v5DPgpStructPtr->CSC ;

            if ((here-> BSIM4v5sNodePrime != 0) && (here-> BSIM4v5gNodePrime != 0))
                here->BSIM4v5SPgpPtr = here->BSIM4v5SPgpStructPtr->CSC ;

            if ((here-> BSIM4v5sNodePrime != 0) && (here-> BSIM4v5sNode != 0))
                here->BSIM4v5SPsPtr = here->BSIM4v5SPsStructPtr->CSC ;

            if ((here-> BSIM4v5sNodePrime != 0) && (here-> BSIM4v5dNodePrime != 0))
                here->BSIM4v5SPdpPtr = here->BSIM4v5SPdpStructPtr->CSC ;

            if ((here-> BSIM4v5qNode != 0) && (here-> BSIM4v5qNode != 0))
                here->BSIM4v5QqPtr = here->BSIM4v5QqStructPtr->CSC ;

            if ((here-> BSIM4v5qNode != 0) && (here-> BSIM4v5bNodePrime != 0))
                here->BSIM4v5QbpPtr = here->BSIM4v5QbpStructPtr->CSC ;

            if ((here-> BSIM4v5qNode != 0) && (here-> BSIM4v5dNodePrime != 0))
                here->BSIM4v5QdpPtr = here->BSIM4v5QdpStructPtr->CSC ;

            if ((here-> BSIM4v5qNode != 0) && (here-> BSIM4v5sNodePrime != 0))
                here->BSIM4v5QspPtr = here->BSIM4v5QspStructPtr->CSC ;

            if ((here-> BSIM4v5qNode != 0) && (here-> BSIM4v5gNodePrime != 0))
                here->BSIM4v5QgpPtr = here->BSIM4v5QgpStructPtr->CSC ;

            if ((here-> BSIM4v5dNodePrime != 0) && (here-> BSIM4v5qNode != 0))
                here->BSIM4v5DPqPtr = here->BSIM4v5DPqStructPtr->CSC ;

            if ((here-> BSIM4v5sNodePrime != 0) && (here-> BSIM4v5qNode != 0))
                here->BSIM4v5SPqPtr = here->BSIM4v5SPqStructPtr->CSC ;

            if ((here-> BSIM4v5gNodePrime != 0) && (here-> BSIM4v5qNode != 0))
                here->BSIM4v5GPqPtr = here->BSIM4v5GPqStructPtr->CSC ;

            if (here->BSIM4v5rgateMod != 0)
            {
                if ((here-> BSIM4v5gNodeExt != 0) && (here-> BSIM4v5gNodeExt != 0))
                    here->BSIM4v5GEgePtr = here->BSIM4v5GEgeStructPtr->CSC ;

                if ((here-> BSIM4v5gNodeExt != 0) && (here-> BSIM4v5gNodePrime != 0))
                    here->BSIM4v5GEgpPtr = here->BSIM4v5GEgpStructPtr->CSC ;

                if ((here-> BSIM4v5gNodePrime != 0) && (here-> BSIM4v5gNodeExt != 0))
                    here->BSIM4v5GPgePtr = here->BSIM4v5GPgeStructPtr->CSC ;

                if ((here-> BSIM4v5gNodeExt != 0) && (here-> BSIM4v5dNodePrime != 0))
                    here->BSIM4v5GEdpPtr = here->BSIM4v5GEdpStructPtr->CSC ;

                if ((here-> BSIM4v5gNodeExt != 0) && (here-> BSIM4v5sNodePrime != 0))
                    here->BSIM4v5GEspPtr = here->BSIM4v5GEspStructPtr->CSC ;

                if ((here-> BSIM4v5gNodeExt != 0) && (here-> BSIM4v5bNodePrime != 0))
                    here->BSIM4v5GEbpPtr = here->BSIM4v5GEbpStructPtr->CSC ;

                if ((here-> BSIM4v5gNodeMid != 0) && (here-> BSIM4v5dNodePrime != 0))
                    here->BSIM4v5GMdpPtr = here->BSIM4v5GMdpStructPtr->CSC ;

                if ((here-> BSIM4v5gNodeMid != 0) && (here-> BSIM4v5gNodePrime != 0))
                    here->BSIM4v5GMgpPtr = here->BSIM4v5GMgpStructPtr->CSC ;

                if ((here-> BSIM4v5gNodeMid != 0) && (here-> BSIM4v5gNodeMid != 0))
                    here->BSIM4v5GMgmPtr = here->BSIM4v5GMgmStructPtr->CSC ;

                if ((here-> BSIM4v5gNodeMid != 0) && (here-> BSIM4v5gNodeExt != 0))
                    here->BSIM4v5GMgePtr = here->BSIM4v5GMgeStructPtr->CSC ;

                if ((here-> BSIM4v5gNodeMid != 0) && (here-> BSIM4v5sNodePrime != 0))
                    here->BSIM4v5GMspPtr = here->BSIM4v5GMspStructPtr->CSC ;

                if ((here-> BSIM4v5gNodeMid != 0) && (here-> BSIM4v5bNodePrime != 0))
                    here->BSIM4v5GMbpPtr = here->BSIM4v5GMbpStructPtr->CSC ;

                if ((here-> BSIM4v5dNodePrime != 0) && (here-> BSIM4v5gNodeMid != 0))
                    here->BSIM4v5DPgmPtr = here->BSIM4v5DPgmStructPtr->CSC ;

                if ((here-> BSIM4v5gNodePrime != 0) && (here-> BSIM4v5gNodeMid != 0))
                    here->BSIM4v5GPgmPtr = here->BSIM4v5GPgmStructPtr->CSC ;

                if ((here-> BSIM4v5gNodeExt != 0) && (here-> BSIM4v5gNodeMid != 0))
                    here->BSIM4v5GEgmPtr = here->BSIM4v5GEgmStructPtr->CSC ;

                if ((here-> BSIM4v5sNodePrime != 0) && (here-> BSIM4v5gNodeMid != 0))
                    here->BSIM4v5SPgmPtr = here->BSIM4v5SPgmStructPtr->CSC ;

                if ((here-> BSIM4v5bNodePrime != 0) && (here-> BSIM4v5gNodeMid != 0))
                    here->BSIM4v5BPgmPtr = here->BSIM4v5BPgmStructPtr->CSC ;

            }
            if ((here->BSIM4v5rbodyMod == 1) || (here->BSIM4v5rbodyMod == 2))
            {
                if ((here-> BSIM4v5dNodePrime != 0) && (here-> BSIM4v5dbNode != 0))
                    here->BSIM4v5DPdbPtr = here->BSIM4v5DPdbStructPtr->CSC ;

                if ((here-> BSIM4v5sNodePrime != 0) && (here-> BSIM4v5sbNode != 0))
                    here->BSIM4v5SPsbPtr = here->BSIM4v5SPsbStructPtr->CSC ;

                if ((here-> BSIM4v5dbNode != 0) && (here-> BSIM4v5dNodePrime != 0))
                    here->BSIM4v5DBdpPtr = here->BSIM4v5DBdpStructPtr->CSC ;

                if ((here-> BSIM4v5dbNode != 0) && (here-> BSIM4v5dbNode != 0))
                    here->BSIM4v5DBdbPtr = here->BSIM4v5DBdbStructPtr->CSC ;

                if ((here-> BSIM4v5dbNode != 0) && (here-> BSIM4v5bNodePrime != 0))
                    here->BSIM4v5DBbpPtr = here->BSIM4v5DBbpStructPtr->CSC ;

                if ((here-> BSIM4v5dbNode != 0) && (here-> BSIM4v5bNode != 0))
                    here->BSIM4v5DBbPtr = here->BSIM4v5DBbStructPtr->CSC ;

                if ((here-> BSIM4v5bNodePrime != 0) && (here-> BSIM4v5dbNode != 0))
                    here->BSIM4v5BPdbPtr = here->BSIM4v5BPdbStructPtr->CSC ;

                if ((here-> BSIM4v5bNodePrime != 0) && (here-> BSIM4v5bNode != 0))
                    here->BSIM4v5BPbPtr = here->BSIM4v5BPbStructPtr->CSC ;

                if ((here-> BSIM4v5bNodePrime != 0) && (here-> BSIM4v5sbNode != 0))
                    here->BSIM4v5BPsbPtr = here->BSIM4v5BPsbStructPtr->CSC ;

                if ((here-> BSIM4v5sbNode != 0) && (here-> BSIM4v5sNodePrime != 0))
                    here->BSIM4v5SBspPtr = here->BSIM4v5SBspStructPtr->CSC ;

                if ((here-> BSIM4v5sbNode != 0) && (here-> BSIM4v5bNodePrime != 0))
                    here->BSIM4v5SBbpPtr = here->BSIM4v5SBbpStructPtr->CSC ;

                if ((here-> BSIM4v5sbNode != 0) && (here-> BSIM4v5bNode != 0))
                    here->BSIM4v5SBbPtr = here->BSIM4v5SBbStructPtr->CSC ;

                if ((here-> BSIM4v5sbNode != 0) && (here-> BSIM4v5sbNode != 0))
                    here->BSIM4v5SBsbPtr = here->BSIM4v5SBsbStructPtr->CSC ;

                if ((here-> BSIM4v5bNode != 0) && (here-> BSIM4v5dbNode != 0))
                    here->BSIM4v5BdbPtr = here->BSIM4v5BdbStructPtr->CSC ;

                if ((here-> BSIM4v5bNode != 0) && (here-> BSIM4v5bNodePrime != 0))
                    here->BSIM4v5BbpPtr = here->BSIM4v5BbpStructPtr->CSC ;

                if ((here-> BSIM4v5bNode != 0) && (here-> BSIM4v5sbNode != 0))
                    here->BSIM4v5BsbPtr = here->BSIM4v5BsbStructPtr->CSC ;

                if ((here-> BSIM4v5bNode != 0) && (here-> BSIM4v5bNode != 0))
                    here->BSIM4v5BbPtr = here->BSIM4v5BbStructPtr->CSC ;

            }
            if (model->BSIM4v5rdsMod)
            {
                if ((here-> BSIM4v5dNode != 0) && (here-> BSIM4v5gNodePrime != 0))
                    here->BSIM4v5DgpPtr = here->BSIM4v5DgpStructPtr->CSC ;

                if ((here-> BSIM4v5dNode != 0) && (here-> BSIM4v5sNodePrime != 0))
                    here->BSIM4v5DspPtr = here->BSIM4v5DspStructPtr->CSC ;

                if ((here-> BSIM4v5dNode != 0) && (here-> BSIM4v5bNodePrime != 0))
                    here->BSIM4v5DbpPtr = here->BSIM4v5DbpStructPtr->CSC ;

                if ((here-> BSIM4v5sNode != 0) && (here-> BSIM4v5dNodePrime != 0))
                    here->BSIM4v5SdpPtr = here->BSIM4v5SdpStructPtr->CSC ;

                if ((here-> BSIM4v5sNode != 0) && (here-> BSIM4v5gNodePrime != 0))
                    here->BSIM4v5SgpPtr = here->BSIM4v5SgpStructPtr->CSC ;

                if ((here-> BSIM4v5sNode != 0) && (here-> BSIM4v5bNodePrime != 0))
                    here->BSIM4v5SbpPtr = here->BSIM4v5SbpStructPtr->CSC ;

            }
        }
    }

    return (OK) ;
}
