/**********
Author: 2013 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "bsim4v7def.h"
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
BSIM4v7bindCSC (GENmodel *inModel, CKTcircuit *ckt)
{
    BSIM4v7model *model = (BSIM4v7model *)inModel ;
    BSIM4v7instance *here ;
    double *i ;
    BindElement *matched, *BindStruct ;
    size_t nz ;

    BindStruct = ckt->CKTmatrix->CKTbindStruct ;
    nz = (size_t)ckt->CKTmatrix->CKTklunz ;

    /* loop through all the BSIM4v7 models */
    for ( ; model != NULL ; model = model->BSIM4v7nextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->BSIM4v7instances ; here != NULL ; here = here->BSIM4v7nextInstance)
        {
            if ((here-> BSIM4v7dNodePrime != 0) && (here-> BSIM4v7bNodePrime != 0))
            {
                i = here->BSIM4v7DPbpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4v7DPbpStructPtr = matched ;
                here->BSIM4v7DPbpPtr = matched->CSC ;
            }

            if ((here-> BSIM4v7gNodePrime != 0) && (here-> BSIM4v7bNodePrime != 0))
            {
                i = here->BSIM4v7GPbpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4v7GPbpStructPtr = matched ;
                here->BSIM4v7GPbpPtr = matched->CSC ;
            }

            if ((here-> BSIM4v7sNodePrime != 0) && (here-> BSIM4v7bNodePrime != 0))
            {
                i = here->BSIM4v7SPbpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4v7SPbpStructPtr = matched ;
                here->BSIM4v7SPbpPtr = matched->CSC ;
            }

            if ((here-> BSIM4v7bNodePrime != 0) && (here-> BSIM4v7dNodePrime != 0))
            {
                i = here->BSIM4v7BPdpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4v7BPdpStructPtr = matched ;
                here->BSIM4v7BPdpPtr = matched->CSC ;
            }

            if ((here-> BSIM4v7bNodePrime != 0) && (here-> BSIM4v7gNodePrime != 0))
            {
                i = here->BSIM4v7BPgpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4v7BPgpStructPtr = matched ;
                here->BSIM4v7BPgpPtr = matched->CSC ;
            }

            if ((here-> BSIM4v7bNodePrime != 0) && (here-> BSIM4v7sNodePrime != 0))
            {
                i = here->BSIM4v7BPspPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4v7BPspStructPtr = matched ;
                here->BSIM4v7BPspPtr = matched->CSC ;
            }

            if ((here-> BSIM4v7bNodePrime != 0) && (here-> BSIM4v7bNodePrime != 0))
            {
                i = here->BSIM4v7BPbpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4v7BPbpStructPtr = matched ;
                here->BSIM4v7BPbpPtr = matched->CSC ;
            }

            if ((here-> BSIM4v7dNode != 0) && (here-> BSIM4v7dNode != 0))
            {
                i = here->BSIM4v7DdPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4v7DdStructPtr = matched ;
                here->BSIM4v7DdPtr = matched->CSC ;
            }

            if ((here-> BSIM4v7gNodePrime != 0) && (here-> BSIM4v7gNodePrime != 0))
            {
                i = here->BSIM4v7GPgpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4v7GPgpStructPtr = matched ;
                here->BSIM4v7GPgpPtr = matched->CSC ;
            }

            if ((here-> BSIM4v7sNode != 0) && (here-> BSIM4v7sNode != 0))
            {
                i = here->BSIM4v7SsPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4v7SsStructPtr = matched ;
                here->BSIM4v7SsPtr = matched->CSC ;
            }

            if ((here-> BSIM4v7dNodePrime != 0) && (here-> BSIM4v7dNodePrime != 0))
            {
                i = here->BSIM4v7DPdpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4v7DPdpStructPtr = matched ;
                here->BSIM4v7DPdpPtr = matched->CSC ;
            }

            if ((here-> BSIM4v7sNodePrime != 0) && (here-> BSIM4v7sNodePrime != 0))
            {
                i = here->BSIM4v7SPspPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4v7SPspStructPtr = matched ;
                here->BSIM4v7SPspPtr = matched->CSC ;
            }

            if ((here-> BSIM4v7dNode != 0) && (here-> BSIM4v7dNodePrime != 0))
            {
                i = here->BSIM4v7DdpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4v7DdpStructPtr = matched ;
                here->BSIM4v7DdpPtr = matched->CSC ;
            }

            if ((here-> BSIM4v7gNodePrime != 0) && (here-> BSIM4v7dNodePrime != 0))
            {
                i = here->BSIM4v7GPdpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4v7GPdpStructPtr = matched ;
                here->BSIM4v7GPdpPtr = matched->CSC ;
            }

            if ((here-> BSIM4v7gNodePrime != 0) && (here-> BSIM4v7sNodePrime != 0))
            {
                i = here->BSIM4v7GPspPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4v7GPspStructPtr = matched ;
                here->BSIM4v7GPspPtr = matched->CSC ;
            }

            if ((here-> BSIM4v7sNode != 0) && (here-> BSIM4v7sNodePrime != 0))
            {
                i = here->BSIM4v7SspPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4v7SspStructPtr = matched ;
                here->BSIM4v7SspPtr = matched->CSC ;
            }

            if ((here-> BSIM4v7dNodePrime != 0) && (here-> BSIM4v7sNodePrime != 0))
            {
                i = here->BSIM4v7DPspPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4v7DPspStructPtr = matched ;
                here->BSIM4v7DPspPtr = matched->CSC ;
            }

            if ((here-> BSIM4v7dNodePrime != 0) && (here-> BSIM4v7dNode != 0))
            {
                i = here->BSIM4v7DPdPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4v7DPdStructPtr = matched ;
                here->BSIM4v7DPdPtr = matched->CSC ;
            }

            if ((here-> BSIM4v7dNodePrime != 0) && (here-> BSIM4v7gNodePrime != 0))
            {
                i = here->BSIM4v7DPgpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4v7DPgpStructPtr = matched ;
                here->BSIM4v7DPgpPtr = matched->CSC ;
            }

            if ((here-> BSIM4v7sNodePrime != 0) && (here-> BSIM4v7gNodePrime != 0))
            {
                i = here->BSIM4v7SPgpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4v7SPgpStructPtr = matched ;
                here->BSIM4v7SPgpPtr = matched->CSC ;
            }

            if ((here-> BSIM4v7sNodePrime != 0) && (here-> BSIM4v7sNode != 0))
            {
                i = here->BSIM4v7SPsPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4v7SPsStructPtr = matched ;
                here->BSIM4v7SPsPtr = matched->CSC ;
            }

            if ((here-> BSIM4v7sNodePrime != 0) && (here-> BSIM4v7dNodePrime != 0))
            {
                i = here->BSIM4v7SPdpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4v7SPdpStructPtr = matched ;
                here->BSIM4v7SPdpPtr = matched->CSC ;
            }

            if ((here-> BSIM4v7qNode != 0) && (here-> BSIM4v7qNode != 0))
            {
                i = here->BSIM4v7QqPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4v7QqStructPtr = matched ;
                here->BSIM4v7QqPtr = matched->CSC ;
            }

            if ((here-> BSIM4v7qNode != 0) && (here-> BSIM4v7bNodePrime != 0))
            {
                i = here->BSIM4v7QbpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4v7QbpStructPtr = matched ;
                here->BSIM4v7QbpPtr = matched->CSC ;
            }

            if ((here-> BSIM4v7qNode != 0) && (here-> BSIM4v7dNodePrime != 0))
            {
                i = here->BSIM4v7QdpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4v7QdpStructPtr = matched ;
                here->BSIM4v7QdpPtr = matched->CSC ;
            }

            if ((here-> BSIM4v7qNode != 0) && (here-> BSIM4v7sNodePrime != 0))
            {
                i = here->BSIM4v7QspPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4v7QspStructPtr = matched ;
                here->BSIM4v7QspPtr = matched->CSC ;
            }

            if ((here-> BSIM4v7qNode != 0) && (here-> BSIM4v7gNodePrime != 0))
            {
                i = here->BSIM4v7QgpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4v7QgpStructPtr = matched ;
                here->BSIM4v7QgpPtr = matched->CSC ;
            }

            if ((here-> BSIM4v7dNodePrime != 0) && (here-> BSIM4v7qNode != 0))
            {
                i = here->BSIM4v7DPqPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4v7DPqStructPtr = matched ;
                here->BSIM4v7DPqPtr = matched->CSC ;
            }

            if ((here-> BSIM4v7sNodePrime != 0) && (here-> BSIM4v7qNode != 0))
            {
                i = here->BSIM4v7SPqPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4v7SPqStructPtr = matched ;
                here->BSIM4v7SPqPtr = matched->CSC ;
            }

            if ((here-> BSIM4v7gNodePrime != 0) && (here-> BSIM4v7qNode != 0))
            {
                i = here->BSIM4v7GPqPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4v7GPqStructPtr = matched ;
                here->BSIM4v7GPqPtr = matched->CSC ;
            }

            if (here->BSIM4v7rgateMod != 0)
            {
                if ((here-> BSIM4v7gNodeExt != 0) && (here-> BSIM4v7gNodeExt != 0))
                {
                    i = here->BSIM4v7GEgePtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v7GEgeStructPtr = matched ;
                    here->BSIM4v7GEgePtr = matched->CSC ;
                }

                if ((here-> BSIM4v7gNodeExt != 0) && (here-> BSIM4v7gNodePrime != 0))
                {
                    i = here->BSIM4v7GEgpPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v7GEgpStructPtr = matched ;
                    here->BSIM4v7GEgpPtr = matched->CSC ;
                }

                if ((here-> BSIM4v7gNodePrime != 0) && (here-> BSIM4v7gNodeExt != 0))
                {
                    i = here->BSIM4v7GPgePtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v7GPgeStructPtr = matched ;
                    here->BSIM4v7GPgePtr = matched->CSC ;
                }

                if ((here-> BSIM4v7gNodeExt != 0) && (here-> BSIM4v7dNodePrime != 0))
                {
                    i = here->BSIM4v7GEdpPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v7GEdpStructPtr = matched ;
                    here->BSIM4v7GEdpPtr = matched->CSC ;
                }

                if ((here-> BSIM4v7gNodeExt != 0) && (here-> BSIM4v7sNodePrime != 0))
                {
                    i = here->BSIM4v7GEspPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v7GEspStructPtr = matched ;
                    here->BSIM4v7GEspPtr = matched->CSC ;
                }

                if ((here-> BSIM4v7gNodeExt != 0) && (here-> BSIM4v7bNodePrime != 0))
                {
                    i = here->BSIM4v7GEbpPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v7GEbpStructPtr = matched ;
                    here->BSIM4v7GEbpPtr = matched->CSC ;
                }

                if ((here-> BSIM4v7gNodeMid != 0) && (here-> BSIM4v7dNodePrime != 0))
                {
                    i = here->BSIM4v7GMdpPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v7GMdpStructPtr = matched ;
                    here->BSIM4v7GMdpPtr = matched->CSC ;
                }

                if ((here-> BSIM4v7gNodeMid != 0) && (here-> BSIM4v7gNodePrime != 0))
                {
                    i = here->BSIM4v7GMgpPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v7GMgpStructPtr = matched ;
                    here->BSIM4v7GMgpPtr = matched->CSC ;
                }

                if ((here-> BSIM4v7gNodeMid != 0) && (here-> BSIM4v7gNodeMid != 0))
                {
                    i = here->BSIM4v7GMgmPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v7GMgmStructPtr = matched ;
                    here->BSIM4v7GMgmPtr = matched->CSC ;
                }

                if ((here-> BSIM4v7gNodeMid != 0) && (here-> BSIM4v7gNodeExt != 0))
                {
                    i = here->BSIM4v7GMgePtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v7GMgeStructPtr = matched ;
                    here->BSIM4v7GMgePtr = matched->CSC ;
                }

                if ((here-> BSIM4v7gNodeMid != 0) && (here-> BSIM4v7sNodePrime != 0))
                {
                    i = here->BSIM4v7GMspPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v7GMspStructPtr = matched ;
                    here->BSIM4v7GMspPtr = matched->CSC ;
                }

                if ((here-> BSIM4v7gNodeMid != 0) && (here-> BSIM4v7bNodePrime != 0))
                {
                    i = here->BSIM4v7GMbpPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v7GMbpStructPtr = matched ;
                    here->BSIM4v7GMbpPtr = matched->CSC ;
                }

                if ((here-> BSIM4v7dNodePrime != 0) && (here-> BSIM4v7gNodeMid != 0))
                {
                    i = here->BSIM4v7DPgmPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v7DPgmStructPtr = matched ;
                    here->BSIM4v7DPgmPtr = matched->CSC ;
                }

                if ((here-> BSIM4v7gNodePrime != 0) && (here-> BSIM4v7gNodeMid != 0))
                {
                    i = here->BSIM4v7GPgmPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v7GPgmStructPtr = matched ;
                    here->BSIM4v7GPgmPtr = matched->CSC ;
                }

                if ((here-> BSIM4v7gNodeExt != 0) && (here-> BSIM4v7gNodeMid != 0))
                {
                    i = here->BSIM4v7GEgmPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v7GEgmStructPtr = matched ;
                    here->BSIM4v7GEgmPtr = matched->CSC ;
                }

                if ((here-> BSIM4v7sNodePrime != 0) && (here-> BSIM4v7gNodeMid != 0))
                {
                    i = here->BSIM4v7SPgmPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v7SPgmStructPtr = matched ;
                    here->BSIM4v7SPgmPtr = matched->CSC ;
                }

                if ((here-> BSIM4v7bNodePrime != 0) && (here-> BSIM4v7gNodeMid != 0))
                {
                    i = here->BSIM4v7BPgmPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v7BPgmStructPtr = matched ;
                    here->BSIM4v7BPgmPtr = matched->CSC ;
                }

            }
            if ((here->BSIM4v7rbodyMod == 1) || (here->BSIM4v7rbodyMod == 2))
            {
                if ((here-> BSIM4v7dNodePrime != 0) && (here-> BSIM4v7dbNode != 0))
                {
                    i = here->BSIM4v7DPdbPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v7DPdbStructPtr = matched ;
                    here->BSIM4v7DPdbPtr = matched->CSC ;
                }

                if ((here-> BSIM4v7sNodePrime != 0) && (here-> BSIM4v7sbNode != 0))
                {
                    i = here->BSIM4v7SPsbPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v7SPsbStructPtr = matched ;
                    here->BSIM4v7SPsbPtr = matched->CSC ;
                }

                if ((here-> BSIM4v7dbNode != 0) && (here-> BSIM4v7dNodePrime != 0))
                {
                    i = here->BSIM4v7DBdpPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v7DBdpStructPtr = matched ;
                    here->BSIM4v7DBdpPtr = matched->CSC ;
                }

                if ((here-> BSIM4v7dbNode != 0) && (here-> BSIM4v7dbNode != 0))
                {
                    i = here->BSIM4v7DBdbPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v7DBdbStructPtr = matched ;
                    here->BSIM4v7DBdbPtr = matched->CSC ;
                }

                if ((here-> BSIM4v7dbNode != 0) && (here-> BSIM4v7bNodePrime != 0))
                {
                    i = here->BSIM4v7DBbpPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v7DBbpStructPtr = matched ;
                    here->BSIM4v7DBbpPtr = matched->CSC ;
                }

                if ((here-> BSIM4v7dbNode != 0) && (here-> BSIM4v7bNode != 0))
                {
                    i = here->BSIM4v7DBbPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v7DBbStructPtr = matched ;
                    here->BSIM4v7DBbPtr = matched->CSC ;
                }

                if ((here-> BSIM4v7bNodePrime != 0) && (here-> BSIM4v7dbNode != 0))
                {
                    i = here->BSIM4v7BPdbPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v7BPdbStructPtr = matched ;
                    here->BSIM4v7BPdbPtr = matched->CSC ;
                }

                if ((here-> BSIM4v7bNodePrime != 0) && (here-> BSIM4v7bNode != 0))
                {
                    i = here->BSIM4v7BPbPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v7BPbStructPtr = matched ;
                    here->BSIM4v7BPbPtr = matched->CSC ;
                }

                if ((here-> BSIM4v7bNodePrime != 0) && (here-> BSIM4v7sbNode != 0))
                {
                    i = here->BSIM4v7BPsbPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v7BPsbStructPtr = matched ;
                    here->BSIM4v7BPsbPtr = matched->CSC ;
                }

                if ((here-> BSIM4v7sbNode != 0) && (here-> BSIM4v7sNodePrime != 0))
                {
                    i = here->BSIM4v7SBspPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v7SBspStructPtr = matched ;
                    here->BSIM4v7SBspPtr = matched->CSC ;
                }

                if ((here-> BSIM4v7sbNode != 0) && (here-> BSIM4v7bNodePrime != 0))
                {
                    i = here->BSIM4v7SBbpPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v7SBbpStructPtr = matched ;
                    here->BSIM4v7SBbpPtr = matched->CSC ;
                }

                if ((here-> BSIM4v7sbNode != 0) && (here-> BSIM4v7bNode != 0))
                {
                    i = here->BSIM4v7SBbPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v7SBbStructPtr = matched ;
                    here->BSIM4v7SBbPtr = matched->CSC ;
                }

                if ((here-> BSIM4v7sbNode != 0) && (here-> BSIM4v7sbNode != 0))
                {
                    i = here->BSIM4v7SBsbPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v7SBsbStructPtr = matched ;
                    here->BSIM4v7SBsbPtr = matched->CSC ;
                }

                if ((here-> BSIM4v7bNode != 0) && (here-> BSIM4v7dbNode != 0))
                {
                    i = here->BSIM4v7BdbPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v7BdbStructPtr = matched ;
                    here->BSIM4v7BdbPtr = matched->CSC ;
                }

                if ((here-> BSIM4v7bNode != 0) && (here-> BSIM4v7bNodePrime != 0))
                {
                    i = here->BSIM4v7BbpPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v7BbpStructPtr = matched ;
                    here->BSIM4v7BbpPtr = matched->CSC ;
                }

                if ((here-> BSIM4v7bNode != 0) && (here-> BSIM4v7sbNode != 0))
                {
                    i = here->BSIM4v7BsbPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v7BsbStructPtr = matched ;
                    here->BSIM4v7BsbPtr = matched->CSC ;
                }

                if ((here-> BSIM4v7bNode != 0) && (here-> BSIM4v7bNode != 0))
                {
                    i = here->BSIM4v7BbPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v7BbStructPtr = matched ;
                    here->BSIM4v7BbPtr = matched->CSC ;
                }

            }
            if (model->BSIM4v7rdsMod)
            {
                if ((here-> BSIM4v7dNode != 0) && (here-> BSIM4v7gNodePrime != 0))
                {
                    i = here->BSIM4v7DgpPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v7DgpStructPtr = matched ;
                    here->BSIM4v7DgpPtr = matched->CSC ;
                }

                if ((here-> BSIM4v7dNode != 0) && (here-> BSIM4v7sNodePrime != 0))
                {
                    i = here->BSIM4v7DspPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v7DspStructPtr = matched ;
                    here->BSIM4v7DspPtr = matched->CSC ;
                }

                if ((here-> BSIM4v7dNode != 0) && (here-> BSIM4v7bNodePrime != 0))
                {
                    i = here->BSIM4v7DbpPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v7DbpStructPtr = matched ;
                    here->BSIM4v7DbpPtr = matched->CSC ;
                }

                if ((here-> BSIM4v7sNode != 0) && (here-> BSIM4v7dNodePrime != 0))
                {
                    i = here->BSIM4v7SdpPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v7SdpStructPtr = matched ;
                    here->BSIM4v7SdpPtr = matched->CSC ;
                }

                if ((here-> BSIM4v7sNode != 0) && (here-> BSIM4v7gNodePrime != 0))
                {
                    i = here->BSIM4v7SgpPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v7SgpStructPtr = matched ;
                    here->BSIM4v7SgpPtr = matched->CSC ;
                }

                if ((here-> BSIM4v7sNode != 0) && (here-> BSIM4v7bNodePrime != 0))
                {
                    i = here->BSIM4v7SbpPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v7SbpStructPtr = matched ;
                    here->BSIM4v7SbpPtr = matched->CSC ;
                }

            }
        }
    }

    return (OK) ;
}

int
BSIM4v7bindCSCComplex (GENmodel *inModel, CKTcircuit *ckt)
{
    BSIM4v7model *model = (BSIM4v7model *)inModel ;
    BSIM4v7instance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the BSIM4v7 models */
    for ( ; model != NULL ; model = model->BSIM4v7nextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->BSIM4v7instances ; here != NULL ; here = here->BSIM4v7nextInstance)
        {
            if ((here-> BSIM4v7dNodePrime != 0) && (here-> BSIM4v7bNodePrime != 0))
                here->BSIM4v7DPbpPtr = here->BSIM4v7DPbpStructPtr->CSC_Complex ;

            if ((here-> BSIM4v7gNodePrime != 0) && (here-> BSIM4v7bNodePrime != 0))
                here->BSIM4v7GPbpPtr = here->BSIM4v7GPbpStructPtr->CSC_Complex ;

            if ((here-> BSIM4v7sNodePrime != 0) && (here-> BSIM4v7bNodePrime != 0))
                here->BSIM4v7SPbpPtr = here->BSIM4v7SPbpStructPtr->CSC_Complex ;

            if ((here-> BSIM4v7bNodePrime != 0) && (here-> BSIM4v7dNodePrime != 0))
                here->BSIM4v7BPdpPtr = here->BSIM4v7BPdpStructPtr->CSC_Complex ;

            if ((here-> BSIM4v7bNodePrime != 0) && (here-> BSIM4v7gNodePrime != 0))
                here->BSIM4v7BPgpPtr = here->BSIM4v7BPgpStructPtr->CSC_Complex ;

            if ((here-> BSIM4v7bNodePrime != 0) && (here-> BSIM4v7sNodePrime != 0))
                here->BSIM4v7BPspPtr = here->BSIM4v7BPspStructPtr->CSC_Complex ;

            if ((here-> BSIM4v7bNodePrime != 0) && (here-> BSIM4v7bNodePrime != 0))
                here->BSIM4v7BPbpPtr = here->BSIM4v7BPbpStructPtr->CSC_Complex ;

            if ((here-> BSIM4v7dNode != 0) && (here-> BSIM4v7dNode != 0))
                here->BSIM4v7DdPtr = here->BSIM4v7DdStructPtr->CSC_Complex ;

            if ((here-> BSIM4v7gNodePrime != 0) && (here-> BSIM4v7gNodePrime != 0))
                here->BSIM4v7GPgpPtr = here->BSIM4v7GPgpStructPtr->CSC_Complex ;

            if ((here-> BSIM4v7sNode != 0) && (here-> BSIM4v7sNode != 0))
                here->BSIM4v7SsPtr = here->BSIM4v7SsStructPtr->CSC_Complex ;

            if ((here-> BSIM4v7dNodePrime != 0) && (here-> BSIM4v7dNodePrime != 0))
                here->BSIM4v7DPdpPtr = here->BSIM4v7DPdpStructPtr->CSC_Complex ;

            if ((here-> BSIM4v7sNodePrime != 0) && (here-> BSIM4v7sNodePrime != 0))
                here->BSIM4v7SPspPtr = here->BSIM4v7SPspStructPtr->CSC_Complex ;

            if ((here-> BSIM4v7dNode != 0) && (here-> BSIM4v7dNodePrime != 0))
                here->BSIM4v7DdpPtr = here->BSIM4v7DdpStructPtr->CSC_Complex ;

            if ((here-> BSIM4v7gNodePrime != 0) && (here-> BSIM4v7dNodePrime != 0))
                here->BSIM4v7GPdpPtr = here->BSIM4v7GPdpStructPtr->CSC_Complex ;

            if ((here-> BSIM4v7gNodePrime != 0) && (here-> BSIM4v7sNodePrime != 0))
                here->BSIM4v7GPspPtr = here->BSIM4v7GPspStructPtr->CSC_Complex ;

            if ((here-> BSIM4v7sNode != 0) && (here-> BSIM4v7sNodePrime != 0))
                here->BSIM4v7SspPtr = here->BSIM4v7SspStructPtr->CSC_Complex ;

            if ((here-> BSIM4v7dNodePrime != 0) && (here-> BSIM4v7sNodePrime != 0))
                here->BSIM4v7DPspPtr = here->BSIM4v7DPspStructPtr->CSC_Complex ;

            if ((here-> BSIM4v7dNodePrime != 0) && (here-> BSIM4v7dNode != 0))
                here->BSIM4v7DPdPtr = here->BSIM4v7DPdStructPtr->CSC_Complex ;

            if ((here-> BSIM4v7dNodePrime != 0) && (here-> BSIM4v7gNodePrime != 0))
                here->BSIM4v7DPgpPtr = here->BSIM4v7DPgpStructPtr->CSC_Complex ;

            if ((here-> BSIM4v7sNodePrime != 0) && (here-> BSIM4v7gNodePrime != 0))
                here->BSIM4v7SPgpPtr = here->BSIM4v7SPgpStructPtr->CSC_Complex ;

            if ((here-> BSIM4v7sNodePrime != 0) && (here-> BSIM4v7sNode != 0))
                here->BSIM4v7SPsPtr = here->BSIM4v7SPsStructPtr->CSC_Complex ;

            if ((here-> BSIM4v7sNodePrime != 0) && (here-> BSIM4v7dNodePrime != 0))
                here->BSIM4v7SPdpPtr = here->BSIM4v7SPdpStructPtr->CSC_Complex ;

            if ((here-> BSIM4v7qNode != 0) && (here-> BSIM4v7qNode != 0))
                here->BSIM4v7QqPtr = here->BSIM4v7QqStructPtr->CSC_Complex ;

            if ((here-> BSIM4v7qNode != 0) && (here-> BSIM4v7bNodePrime != 0))
                here->BSIM4v7QbpPtr = here->BSIM4v7QbpStructPtr->CSC_Complex ;

            if ((here-> BSIM4v7qNode != 0) && (here-> BSIM4v7dNodePrime != 0))
                here->BSIM4v7QdpPtr = here->BSIM4v7QdpStructPtr->CSC_Complex ;

            if ((here-> BSIM4v7qNode != 0) && (here-> BSIM4v7sNodePrime != 0))
                here->BSIM4v7QspPtr = here->BSIM4v7QspStructPtr->CSC_Complex ;

            if ((here-> BSIM4v7qNode != 0) && (here-> BSIM4v7gNodePrime != 0))
                here->BSIM4v7QgpPtr = here->BSIM4v7QgpStructPtr->CSC_Complex ;

            if ((here-> BSIM4v7dNodePrime != 0) && (here-> BSIM4v7qNode != 0))
                here->BSIM4v7DPqPtr = here->BSIM4v7DPqStructPtr->CSC_Complex ;

            if ((here-> BSIM4v7sNodePrime != 0) && (here-> BSIM4v7qNode != 0))
                here->BSIM4v7SPqPtr = here->BSIM4v7SPqStructPtr->CSC_Complex ;

            if ((here-> BSIM4v7gNodePrime != 0) && (here-> BSIM4v7qNode != 0))
                here->BSIM4v7GPqPtr = here->BSIM4v7GPqStructPtr->CSC_Complex ;

            if (here->BSIM4v7rgateMod != 0)
            {
                if ((here-> BSIM4v7gNodeExt != 0) && (here-> BSIM4v7gNodeExt != 0))
                    here->BSIM4v7GEgePtr = here->BSIM4v7GEgeStructPtr->CSC_Complex ;

                if ((here-> BSIM4v7gNodeExt != 0) && (here-> BSIM4v7gNodePrime != 0))
                    here->BSIM4v7GEgpPtr = here->BSIM4v7GEgpStructPtr->CSC_Complex ;

                if ((here-> BSIM4v7gNodePrime != 0) && (here-> BSIM4v7gNodeExt != 0))
                    here->BSIM4v7GPgePtr = here->BSIM4v7GPgeStructPtr->CSC_Complex ;

                if ((here-> BSIM4v7gNodeExt != 0) && (here-> BSIM4v7dNodePrime != 0))
                    here->BSIM4v7GEdpPtr = here->BSIM4v7GEdpStructPtr->CSC_Complex ;

                if ((here-> BSIM4v7gNodeExt != 0) && (here-> BSIM4v7sNodePrime != 0))
                    here->BSIM4v7GEspPtr = here->BSIM4v7GEspStructPtr->CSC_Complex ;

                if ((here-> BSIM4v7gNodeExt != 0) && (here-> BSIM4v7bNodePrime != 0))
                    here->BSIM4v7GEbpPtr = here->BSIM4v7GEbpStructPtr->CSC_Complex ;

                if ((here-> BSIM4v7gNodeMid != 0) && (here-> BSIM4v7dNodePrime != 0))
                    here->BSIM4v7GMdpPtr = here->BSIM4v7GMdpStructPtr->CSC_Complex ;

                if ((here-> BSIM4v7gNodeMid != 0) && (here-> BSIM4v7gNodePrime != 0))
                    here->BSIM4v7GMgpPtr = here->BSIM4v7GMgpStructPtr->CSC_Complex ;

                if ((here-> BSIM4v7gNodeMid != 0) && (here-> BSIM4v7gNodeMid != 0))
                    here->BSIM4v7GMgmPtr = here->BSIM4v7GMgmStructPtr->CSC_Complex ;

                if ((here-> BSIM4v7gNodeMid != 0) && (here-> BSIM4v7gNodeExt != 0))
                    here->BSIM4v7GMgePtr = here->BSIM4v7GMgeStructPtr->CSC_Complex ;

                if ((here-> BSIM4v7gNodeMid != 0) && (here-> BSIM4v7sNodePrime != 0))
                    here->BSIM4v7GMspPtr = here->BSIM4v7GMspStructPtr->CSC_Complex ;

                if ((here-> BSIM4v7gNodeMid != 0) && (here-> BSIM4v7bNodePrime != 0))
                    here->BSIM4v7GMbpPtr = here->BSIM4v7GMbpStructPtr->CSC_Complex ;

                if ((here-> BSIM4v7dNodePrime != 0) && (here-> BSIM4v7gNodeMid != 0))
                    here->BSIM4v7DPgmPtr = here->BSIM4v7DPgmStructPtr->CSC_Complex ;

                if ((here-> BSIM4v7gNodePrime != 0) && (here-> BSIM4v7gNodeMid != 0))
                    here->BSIM4v7GPgmPtr = here->BSIM4v7GPgmStructPtr->CSC_Complex ;

                if ((here-> BSIM4v7gNodeExt != 0) && (here-> BSIM4v7gNodeMid != 0))
                    here->BSIM4v7GEgmPtr = here->BSIM4v7GEgmStructPtr->CSC_Complex ;

                if ((here-> BSIM4v7sNodePrime != 0) && (here-> BSIM4v7gNodeMid != 0))
                    here->BSIM4v7SPgmPtr = here->BSIM4v7SPgmStructPtr->CSC_Complex ;

                if ((here-> BSIM4v7bNodePrime != 0) && (here-> BSIM4v7gNodeMid != 0))
                    here->BSIM4v7BPgmPtr = here->BSIM4v7BPgmStructPtr->CSC_Complex ;

            }
            if ((here->BSIM4v7rbodyMod == 1) || (here->BSIM4v7rbodyMod == 2))
            {
                if ((here-> BSIM4v7dNodePrime != 0) && (here-> BSIM4v7dbNode != 0))
                    here->BSIM4v7DPdbPtr = here->BSIM4v7DPdbStructPtr->CSC_Complex ;

                if ((here-> BSIM4v7sNodePrime != 0) && (here-> BSIM4v7sbNode != 0))
                    here->BSIM4v7SPsbPtr = here->BSIM4v7SPsbStructPtr->CSC_Complex ;

                if ((here-> BSIM4v7dbNode != 0) && (here-> BSIM4v7dNodePrime != 0))
                    here->BSIM4v7DBdpPtr = here->BSIM4v7DBdpStructPtr->CSC_Complex ;

                if ((here-> BSIM4v7dbNode != 0) && (here-> BSIM4v7dbNode != 0))
                    here->BSIM4v7DBdbPtr = here->BSIM4v7DBdbStructPtr->CSC_Complex ;

                if ((here-> BSIM4v7dbNode != 0) && (here-> BSIM4v7bNodePrime != 0))
                    here->BSIM4v7DBbpPtr = here->BSIM4v7DBbpStructPtr->CSC_Complex ;

                if ((here-> BSIM4v7dbNode != 0) && (here-> BSIM4v7bNode != 0))
                    here->BSIM4v7DBbPtr = here->BSIM4v7DBbStructPtr->CSC_Complex ;

                if ((here-> BSIM4v7bNodePrime != 0) && (here-> BSIM4v7dbNode != 0))
                    here->BSIM4v7BPdbPtr = here->BSIM4v7BPdbStructPtr->CSC_Complex ;

                if ((here-> BSIM4v7bNodePrime != 0) && (here-> BSIM4v7bNode != 0))
                    here->BSIM4v7BPbPtr = here->BSIM4v7BPbStructPtr->CSC_Complex ;

                if ((here-> BSIM4v7bNodePrime != 0) && (here-> BSIM4v7sbNode != 0))
                    here->BSIM4v7BPsbPtr = here->BSIM4v7BPsbStructPtr->CSC_Complex ;

                if ((here-> BSIM4v7sbNode != 0) && (here-> BSIM4v7sNodePrime != 0))
                    here->BSIM4v7SBspPtr = here->BSIM4v7SBspStructPtr->CSC_Complex ;

                if ((here-> BSIM4v7sbNode != 0) && (here-> BSIM4v7bNodePrime != 0))
                    here->BSIM4v7SBbpPtr = here->BSIM4v7SBbpStructPtr->CSC_Complex ;

                if ((here-> BSIM4v7sbNode != 0) && (here-> BSIM4v7bNode != 0))
                    here->BSIM4v7SBbPtr = here->BSIM4v7SBbStructPtr->CSC_Complex ;

                if ((here-> BSIM4v7sbNode != 0) && (here-> BSIM4v7sbNode != 0))
                    here->BSIM4v7SBsbPtr = here->BSIM4v7SBsbStructPtr->CSC_Complex ;

                if ((here-> BSIM4v7bNode != 0) && (here-> BSIM4v7dbNode != 0))
                    here->BSIM4v7BdbPtr = here->BSIM4v7BdbStructPtr->CSC_Complex ;

                if ((here-> BSIM4v7bNode != 0) && (here-> BSIM4v7bNodePrime != 0))
                    here->BSIM4v7BbpPtr = here->BSIM4v7BbpStructPtr->CSC_Complex ;

                if ((here-> BSIM4v7bNode != 0) && (here-> BSIM4v7sbNode != 0))
                    here->BSIM4v7BsbPtr = here->BSIM4v7BsbStructPtr->CSC_Complex ;

                if ((here-> BSIM4v7bNode != 0) && (here-> BSIM4v7bNode != 0))
                    here->BSIM4v7BbPtr = here->BSIM4v7BbStructPtr->CSC_Complex ;

            }
            if (model->BSIM4v7rdsMod)
            {
                if ((here-> BSIM4v7dNode != 0) && (here-> BSIM4v7gNodePrime != 0))
                    here->BSIM4v7DgpPtr = here->BSIM4v7DgpStructPtr->CSC_Complex ;

                if ((here-> BSIM4v7dNode != 0) && (here-> BSIM4v7sNodePrime != 0))
                    here->BSIM4v7DspPtr = here->BSIM4v7DspStructPtr->CSC_Complex ;

                if ((here-> BSIM4v7dNode != 0) && (here-> BSIM4v7bNodePrime != 0))
                    here->BSIM4v7DbpPtr = here->BSIM4v7DbpStructPtr->CSC_Complex ;

                if ((here-> BSIM4v7sNode != 0) && (here-> BSIM4v7dNodePrime != 0))
                    here->BSIM4v7SdpPtr = here->BSIM4v7SdpStructPtr->CSC_Complex ;

                if ((here-> BSIM4v7sNode != 0) && (here-> BSIM4v7gNodePrime != 0))
                    here->BSIM4v7SgpPtr = here->BSIM4v7SgpStructPtr->CSC_Complex ;

                if ((here-> BSIM4v7sNode != 0) && (here-> BSIM4v7bNodePrime != 0))
                    here->BSIM4v7SbpPtr = here->BSIM4v7SbpStructPtr->CSC_Complex ;

            }
        }
    }

    return (OK) ;
}

int
BSIM4v7bindCSCComplexToReal (GENmodel *inModel, CKTcircuit *ckt)
{
    BSIM4v7model *model = (BSIM4v7model *)inModel ;
    BSIM4v7instance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the BSIM4v7 models */
    for ( ; model != NULL ; model = model->BSIM4v7nextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->BSIM4v7instances ; here != NULL ; here = here->BSIM4v7nextInstance)
        {
            if ((here-> BSIM4v7dNodePrime != 0) && (here-> BSIM4v7bNodePrime != 0))
                here->BSIM4v7DPbpPtr = here->BSIM4v7DPbpStructPtr->CSC ;

            if ((here-> BSIM4v7gNodePrime != 0) && (here-> BSIM4v7bNodePrime != 0))
                here->BSIM4v7GPbpPtr = here->BSIM4v7GPbpStructPtr->CSC ;

            if ((here-> BSIM4v7sNodePrime != 0) && (here-> BSIM4v7bNodePrime != 0))
                here->BSIM4v7SPbpPtr = here->BSIM4v7SPbpStructPtr->CSC ;

            if ((here-> BSIM4v7bNodePrime != 0) && (here-> BSIM4v7dNodePrime != 0))
                here->BSIM4v7BPdpPtr = here->BSIM4v7BPdpStructPtr->CSC ;

            if ((here-> BSIM4v7bNodePrime != 0) && (here-> BSIM4v7gNodePrime != 0))
                here->BSIM4v7BPgpPtr = here->BSIM4v7BPgpStructPtr->CSC ;

            if ((here-> BSIM4v7bNodePrime != 0) && (here-> BSIM4v7sNodePrime != 0))
                here->BSIM4v7BPspPtr = here->BSIM4v7BPspStructPtr->CSC ;

            if ((here-> BSIM4v7bNodePrime != 0) && (here-> BSIM4v7bNodePrime != 0))
                here->BSIM4v7BPbpPtr = here->BSIM4v7BPbpStructPtr->CSC ;

            if ((here-> BSIM4v7dNode != 0) && (here-> BSIM4v7dNode != 0))
                here->BSIM4v7DdPtr = here->BSIM4v7DdStructPtr->CSC ;

            if ((here-> BSIM4v7gNodePrime != 0) && (here-> BSIM4v7gNodePrime != 0))
                here->BSIM4v7GPgpPtr = here->BSIM4v7GPgpStructPtr->CSC ;

            if ((here-> BSIM4v7sNode != 0) && (here-> BSIM4v7sNode != 0))
                here->BSIM4v7SsPtr = here->BSIM4v7SsStructPtr->CSC ;

            if ((here-> BSIM4v7dNodePrime != 0) && (here-> BSIM4v7dNodePrime != 0))
                here->BSIM4v7DPdpPtr = here->BSIM4v7DPdpStructPtr->CSC ;

            if ((here-> BSIM4v7sNodePrime != 0) && (here-> BSIM4v7sNodePrime != 0))
                here->BSIM4v7SPspPtr = here->BSIM4v7SPspStructPtr->CSC ;

            if ((here-> BSIM4v7dNode != 0) && (here-> BSIM4v7dNodePrime != 0))
                here->BSIM4v7DdpPtr = here->BSIM4v7DdpStructPtr->CSC ;

            if ((here-> BSIM4v7gNodePrime != 0) && (here-> BSIM4v7dNodePrime != 0))
                here->BSIM4v7GPdpPtr = here->BSIM4v7GPdpStructPtr->CSC ;

            if ((here-> BSIM4v7gNodePrime != 0) && (here-> BSIM4v7sNodePrime != 0))
                here->BSIM4v7GPspPtr = here->BSIM4v7GPspStructPtr->CSC ;

            if ((here-> BSIM4v7sNode != 0) && (here-> BSIM4v7sNodePrime != 0))
                here->BSIM4v7SspPtr = here->BSIM4v7SspStructPtr->CSC ;

            if ((here-> BSIM4v7dNodePrime != 0) && (here-> BSIM4v7sNodePrime != 0))
                here->BSIM4v7DPspPtr = here->BSIM4v7DPspStructPtr->CSC ;

            if ((here-> BSIM4v7dNodePrime != 0) && (here-> BSIM4v7dNode != 0))
                here->BSIM4v7DPdPtr = here->BSIM4v7DPdStructPtr->CSC ;

            if ((here-> BSIM4v7dNodePrime != 0) && (here-> BSIM4v7gNodePrime != 0))
                here->BSIM4v7DPgpPtr = here->BSIM4v7DPgpStructPtr->CSC ;

            if ((here-> BSIM4v7sNodePrime != 0) && (here-> BSIM4v7gNodePrime != 0))
                here->BSIM4v7SPgpPtr = here->BSIM4v7SPgpStructPtr->CSC ;

            if ((here-> BSIM4v7sNodePrime != 0) && (here-> BSIM4v7sNode != 0))
                here->BSIM4v7SPsPtr = here->BSIM4v7SPsStructPtr->CSC ;

            if ((here-> BSIM4v7sNodePrime != 0) && (here-> BSIM4v7dNodePrime != 0))
                here->BSIM4v7SPdpPtr = here->BSIM4v7SPdpStructPtr->CSC ;

            if ((here-> BSIM4v7qNode != 0) && (here-> BSIM4v7qNode != 0))
                here->BSIM4v7QqPtr = here->BSIM4v7QqStructPtr->CSC ;

            if ((here-> BSIM4v7qNode != 0) && (here-> BSIM4v7bNodePrime != 0))
                here->BSIM4v7QbpPtr = here->BSIM4v7QbpStructPtr->CSC ;

            if ((here-> BSIM4v7qNode != 0) && (here-> BSIM4v7dNodePrime != 0))
                here->BSIM4v7QdpPtr = here->BSIM4v7QdpStructPtr->CSC ;

            if ((here-> BSIM4v7qNode != 0) && (here-> BSIM4v7sNodePrime != 0))
                here->BSIM4v7QspPtr = here->BSIM4v7QspStructPtr->CSC ;

            if ((here-> BSIM4v7qNode != 0) && (here-> BSIM4v7gNodePrime != 0))
                here->BSIM4v7QgpPtr = here->BSIM4v7QgpStructPtr->CSC ;

            if ((here-> BSIM4v7dNodePrime != 0) && (here-> BSIM4v7qNode != 0))
                here->BSIM4v7DPqPtr = here->BSIM4v7DPqStructPtr->CSC ;

            if ((here-> BSIM4v7sNodePrime != 0) && (here-> BSIM4v7qNode != 0))
                here->BSIM4v7SPqPtr = here->BSIM4v7SPqStructPtr->CSC ;

            if ((here-> BSIM4v7gNodePrime != 0) && (here-> BSIM4v7qNode != 0))
                here->BSIM4v7GPqPtr = here->BSIM4v7GPqStructPtr->CSC ;

            if (here->BSIM4v7rgateMod != 0)
            {
                if ((here-> BSIM4v7gNodeExt != 0) && (here-> BSIM4v7gNodeExt != 0))
                    here->BSIM4v7GEgePtr = here->BSIM4v7GEgeStructPtr->CSC ;

                if ((here-> BSIM4v7gNodeExt != 0) && (here-> BSIM4v7gNodePrime != 0))
                    here->BSIM4v7GEgpPtr = here->BSIM4v7GEgpStructPtr->CSC ;

                if ((here-> BSIM4v7gNodePrime != 0) && (here-> BSIM4v7gNodeExt != 0))
                    here->BSIM4v7GPgePtr = here->BSIM4v7GPgeStructPtr->CSC ;

                if ((here-> BSIM4v7gNodeExt != 0) && (here-> BSIM4v7dNodePrime != 0))
                    here->BSIM4v7GEdpPtr = here->BSIM4v7GEdpStructPtr->CSC ;

                if ((here-> BSIM4v7gNodeExt != 0) && (here-> BSIM4v7sNodePrime != 0))
                    here->BSIM4v7GEspPtr = here->BSIM4v7GEspStructPtr->CSC ;

                if ((here-> BSIM4v7gNodeExt != 0) && (here-> BSIM4v7bNodePrime != 0))
                    here->BSIM4v7GEbpPtr = here->BSIM4v7GEbpStructPtr->CSC ;

                if ((here-> BSIM4v7gNodeMid != 0) && (here-> BSIM4v7dNodePrime != 0))
                    here->BSIM4v7GMdpPtr = here->BSIM4v7GMdpStructPtr->CSC ;

                if ((here-> BSIM4v7gNodeMid != 0) && (here-> BSIM4v7gNodePrime != 0))
                    here->BSIM4v7GMgpPtr = here->BSIM4v7GMgpStructPtr->CSC ;

                if ((here-> BSIM4v7gNodeMid != 0) && (here-> BSIM4v7gNodeMid != 0))
                    here->BSIM4v7GMgmPtr = here->BSIM4v7GMgmStructPtr->CSC ;

                if ((here-> BSIM4v7gNodeMid != 0) && (here-> BSIM4v7gNodeExt != 0))
                    here->BSIM4v7GMgePtr = here->BSIM4v7GMgeStructPtr->CSC ;

                if ((here-> BSIM4v7gNodeMid != 0) && (here-> BSIM4v7sNodePrime != 0))
                    here->BSIM4v7GMspPtr = here->BSIM4v7GMspStructPtr->CSC ;

                if ((here-> BSIM4v7gNodeMid != 0) && (here-> BSIM4v7bNodePrime != 0))
                    here->BSIM4v7GMbpPtr = here->BSIM4v7GMbpStructPtr->CSC ;

                if ((here-> BSIM4v7dNodePrime != 0) && (here-> BSIM4v7gNodeMid != 0))
                    here->BSIM4v7DPgmPtr = here->BSIM4v7DPgmStructPtr->CSC ;

                if ((here-> BSIM4v7gNodePrime != 0) && (here-> BSIM4v7gNodeMid != 0))
                    here->BSIM4v7GPgmPtr = here->BSIM4v7GPgmStructPtr->CSC ;

                if ((here-> BSIM4v7gNodeExt != 0) && (here-> BSIM4v7gNodeMid != 0))
                    here->BSIM4v7GEgmPtr = here->BSIM4v7GEgmStructPtr->CSC ;

                if ((here-> BSIM4v7sNodePrime != 0) && (here-> BSIM4v7gNodeMid != 0))
                    here->BSIM4v7SPgmPtr = here->BSIM4v7SPgmStructPtr->CSC ;

                if ((here-> BSIM4v7bNodePrime != 0) && (here-> BSIM4v7gNodeMid != 0))
                    here->BSIM4v7BPgmPtr = here->BSIM4v7BPgmStructPtr->CSC ;

            }
            if ((here->BSIM4v7rbodyMod == 1) || (here->BSIM4v7rbodyMod == 2))
            {
                if ((here-> BSIM4v7dNodePrime != 0) && (here-> BSIM4v7dbNode != 0))
                    here->BSIM4v7DPdbPtr = here->BSIM4v7DPdbStructPtr->CSC ;

                if ((here-> BSIM4v7sNodePrime != 0) && (here-> BSIM4v7sbNode != 0))
                    here->BSIM4v7SPsbPtr = here->BSIM4v7SPsbStructPtr->CSC ;

                if ((here-> BSIM4v7dbNode != 0) && (here-> BSIM4v7dNodePrime != 0))
                    here->BSIM4v7DBdpPtr = here->BSIM4v7DBdpStructPtr->CSC ;

                if ((here-> BSIM4v7dbNode != 0) && (here-> BSIM4v7dbNode != 0))
                    here->BSIM4v7DBdbPtr = here->BSIM4v7DBdbStructPtr->CSC ;

                if ((here-> BSIM4v7dbNode != 0) && (here-> BSIM4v7bNodePrime != 0))
                    here->BSIM4v7DBbpPtr = here->BSIM4v7DBbpStructPtr->CSC ;

                if ((here-> BSIM4v7dbNode != 0) && (here-> BSIM4v7bNode != 0))
                    here->BSIM4v7DBbPtr = here->BSIM4v7DBbStructPtr->CSC ;

                if ((here-> BSIM4v7bNodePrime != 0) && (here-> BSIM4v7dbNode != 0))
                    here->BSIM4v7BPdbPtr = here->BSIM4v7BPdbStructPtr->CSC ;

                if ((here-> BSIM4v7bNodePrime != 0) && (here-> BSIM4v7bNode != 0))
                    here->BSIM4v7BPbPtr = here->BSIM4v7BPbStructPtr->CSC ;

                if ((here-> BSIM4v7bNodePrime != 0) && (here-> BSIM4v7sbNode != 0))
                    here->BSIM4v7BPsbPtr = here->BSIM4v7BPsbStructPtr->CSC ;

                if ((here-> BSIM4v7sbNode != 0) && (here-> BSIM4v7sNodePrime != 0))
                    here->BSIM4v7SBspPtr = here->BSIM4v7SBspStructPtr->CSC ;

                if ((here-> BSIM4v7sbNode != 0) && (here-> BSIM4v7bNodePrime != 0))
                    here->BSIM4v7SBbpPtr = here->BSIM4v7SBbpStructPtr->CSC ;

                if ((here-> BSIM4v7sbNode != 0) && (here-> BSIM4v7bNode != 0))
                    here->BSIM4v7SBbPtr = here->BSIM4v7SBbStructPtr->CSC ;

                if ((here-> BSIM4v7sbNode != 0) && (here-> BSIM4v7sbNode != 0))
                    here->BSIM4v7SBsbPtr = here->BSIM4v7SBsbStructPtr->CSC ;

                if ((here-> BSIM4v7bNode != 0) && (here-> BSIM4v7dbNode != 0))
                    here->BSIM4v7BdbPtr = here->BSIM4v7BdbStructPtr->CSC ;

                if ((here-> BSIM4v7bNode != 0) && (here-> BSIM4v7bNodePrime != 0))
                    here->BSIM4v7BbpPtr = here->BSIM4v7BbpStructPtr->CSC ;

                if ((here-> BSIM4v7bNode != 0) && (here-> BSIM4v7sbNode != 0))
                    here->BSIM4v7BsbPtr = here->BSIM4v7BsbStructPtr->CSC ;

                if ((here-> BSIM4v7bNode != 0) && (here-> BSIM4v7bNode != 0))
                    here->BSIM4v7BbPtr = here->BSIM4v7BbStructPtr->CSC ;

            }
            if (model->BSIM4v7rdsMod)
            {
                if ((here-> BSIM4v7dNode != 0) && (here-> BSIM4v7gNodePrime != 0))
                    here->BSIM4v7DgpPtr = here->BSIM4v7DgpStructPtr->CSC ;

                if ((here-> BSIM4v7dNode != 0) && (here-> BSIM4v7sNodePrime != 0))
                    here->BSIM4v7DspPtr = here->BSIM4v7DspStructPtr->CSC ;

                if ((here-> BSIM4v7dNode != 0) && (here-> BSIM4v7bNodePrime != 0))
                    here->BSIM4v7DbpPtr = here->BSIM4v7DbpStructPtr->CSC ;

                if ((here-> BSIM4v7sNode != 0) && (here-> BSIM4v7dNodePrime != 0))
                    here->BSIM4v7SdpPtr = here->BSIM4v7SdpStructPtr->CSC ;

                if ((here-> BSIM4v7sNode != 0) && (here-> BSIM4v7gNodePrime != 0))
                    here->BSIM4v7SgpPtr = here->BSIM4v7SgpStructPtr->CSC ;

                if ((here-> BSIM4v7sNode != 0) && (here-> BSIM4v7bNodePrime != 0))
                    here->BSIM4v7SbpPtr = here->BSIM4v7SbpStructPtr->CSC ;

            }
        }
    }

    return (OK) ;
}
