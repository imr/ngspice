/**********
Author: 2013 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "bsim4v4def.h"
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
BSIM4v4bindCSC (GENmodel *inModel, CKTcircuit *ckt)
{
    BSIM4v4model *model = (BSIM4v4model *)inModel ;
    BSIM4v4instance *here ;
    double *i ;
    BindElement *matched, *BindStruct ;
    size_t nz ;

    BindStruct = ckt->CKTmatrix->CKTbindStruct ;
    nz = (size_t)ckt->CKTmatrix->CKTklunz ;

    /* loop through all the BSIM4v4 models */
    for ( ; model != NULL ; model = model->BSIM4v4nextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->BSIM4v4instances ; here != NULL ; here = here->BSIM4v4nextInstance)
        {
            if ((here-> BSIM4v4dNodePrime != 0) && (here-> BSIM4v4bNodePrime != 0))
            {
                i = here->BSIM4v4DPbpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4v4DPbpStructPtr = matched ;
                here->BSIM4v4DPbpPtr = matched->CSC ;
            }

            if ((here-> BSIM4v4gNodePrime != 0) && (here-> BSIM4v4bNodePrime != 0))
            {
                i = here->BSIM4v4GPbpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4v4GPbpStructPtr = matched ;
                here->BSIM4v4GPbpPtr = matched->CSC ;
            }

            if ((here-> BSIM4v4sNodePrime != 0) && (here-> BSIM4v4bNodePrime != 0))
            {
                i = here->BSIM4v4SPbpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4v4SPbpStructPtr = matched ;
                here->BSIM4v4SPbpPtr = matched->CSC ;
            }

            if ((here-> BSIM4v4bNodePrime != 0) && (here-> BSIM4v4dNodePrime != 0))
            {
                i = here->BSIM4v4BPdpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4v4BPdpStructPtr = matched ;
                here->BSIM4v4BPdpPtr = matched->CSC ;
            }

            if ((here-> BSIM4v4bNodePrime != 0) && (here-> BSIM4v4gNodePrime != 0))
            {
                i = here->BSIM4v4BPgpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4v4BPgpStructPtr = matched ;
                here->BSIM4v4BPgpPtr = matched->CSC ;
            }

            if ((here-> BSIM4v4bNodePrime != 0) && (here-> BSIM4v4sNodePrime != 0))
            {
                i = here->BSIM4v4BPspPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4v4BPspStructPtr = matched ;
                here->BSIM4v4BPspPtr = matched->CSC ;
            }

            if ((here-> BSIM4v4bNodePrime != 0) && (here-> BSIM4v4bNodePrime != 0))
            {
                i = here->BSIM4v4BPbpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4v4BPbpStructPtr = matched ;
                here->BSIM4v4BPbpPtr = matched->CSC ;
            }

            if ((here-> BSIM4v4dNode != 0) && (here-> BSIM4v4dNode != 0))
            {
                i = here->BSIM4v4DdPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4v4DdStructPtr = matched ;
                here->BSIM4v4DdPtr = matched->CSC ;
            }

            if ((here-> BSIM4v4gNodePrime != 0) && (here-> BSIM4v4gNodePrime != 0))
            {
                i = here->BSIM4v4GPgpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4v4GPgpStructPtr = matched ;
                here->BSIM4v4GPgpPtr = matched->CSC ;
            }

            if ((here-> BSIM4v4sNode != 0) && (here-> BSIM4v4sNode != 0))
            {
                i = here->BSIM4v4SsPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4v4SsStructPtr = matched ;
                here->BSIM4v4SsPtr = matched->CSC ;
            }

            if ((here-> BSIM4v4dNodePrime != 0) && (here-> BSIM4v4dNodePrime != 0))
            {
                i = here->BSIM4v4DPdpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4v4DPdpStructPtr = matched ;
                here->BSIM4v4DPdpPtr = matched->CSC ;
            }

            if ((here-> BSIM4v4sNodePrime != 0) && (here-> BSIM4v4sNodePrime != 0))
            {
                i = here->BSIM4v4SPspPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4v4SPspStructPtr = matched ;
                here->BSIM4v4SPspPtr = matched->CSC ;
            }

            if ((here-> BSIM4v4dNode != 0) && (here-> BSIM4v4dNodePrime != 0))
            {
                i = here->BSIM4v4DdpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4v4DdpStructPtr = matched ;
                here->BSIM4v4DdpPtr = matched->CSC ;
            }

            if ((here-> BSIM4v4gNodePrime != 0) && (here-> BSIM4v4dNodePrime != 0))
            {
                i = here->BSIM4v4GPdpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4v4GPdpStructPtr = matched ;
                here->BSIM4v4GPdpPtr = matched->CSC ;
            }

            if ((here-> BSIM4v4gNodePrime != 0) && (here-> BSIM4v4sNodePrime != 0))
            {
                i = here->BSIM4v4GPspPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4v4GPspStructPtr = matched ;
                here->BSIM4v4GPspPtr = matched->CSC ;
            }

            if ((here-> BSIM4v4sNode != 0) && (here-> BSIM4v4sNodePrime != 0))
            {
                i = here->BSIM4v4SspPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4v4SspStructPtr = matched ;
                here->BSIM4v4SspPtr = matched->CSC ;
            }

            if ((here-> BSIM4v4dNodePrime != 0) && (here-> BSIM4v4sNodePrime != 0))
            {
                i = here->BSIM4v4DPspPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4v4DPspStructPtr = matched ;
                here->BSIM4v4DPspPtr = matched->CSC ;
            }

            if ((here-> BSIM4v4dNodePrime != 0) && (here-> BSIM4v4dNode != 0))
            {
                i = here->BSIM4v4DPdPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4v4DPdStructPtr = matched ;
                here->BSIM4v4DPdPtr = matched->CSC ;
            }

            if ((here-> BSIM4v4dNodePrime != 0) && (here-> BSIM4v4gNodePrime != 0))
            {
                i = here->BSIM4v4DPgpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4v4DPgpStructPtr = matched ;
                here->BSIM4v4DPgpPtr = matched->CSC ;
            }

            if ((here-> BSIM4v4sNodePrime != 0) && (here-> BSIM4v4gNodePrime != 0))
            {
                i = here->BSIM4v4SPgpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4v4SPgpStructPtr = matched ;
                here->BSIM4v4SPgpPtr = matched->CSC ;
            }

            if ((here-> BSIM4v4sNodePrime != 0) && (here-> BSIM4v4sNode != 0))
            {
                i = here->BSIM4v4SPsPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4v4SPsStructPtr = matched ;
                here->BSIM4v4SPsPtr = matched->CSC ;
            }

            if ((here-> BSIM4v4sNodePrime != 0) && (here-> BSIM4v4dNodePrime != 0))
            {
                i = here->BSIM4v4SPdpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4v4SPdpStructPtr = matched ;
                here->BSIM4v4SPdpPtr = matched->CSC ;
            }

            if ((here-> BSIM4v4qNode != 0) && (here-> BSIM4v4qNode != 0))
            {
                i = here->BSIM4v4QqPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4v4QqStructPtr = matched ;
                here->BSIM4v4QqPtr = matched->CSC ;
            }

            if ((here-> BSIM4v4qNode != 0) && (here-> BSIM4v4bNodePrime != 0))
            {
                i = here->BSIM4v4QbpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4v4QbpStructPtr = matched ;
                here->BSIM4v4QbpPtr = matched->CSC ;
            }

            if ((here-> BSIM4v4qNode != 0) && (here-> BSIM4v4dNodePrime != 0))
            {
                i = here->BSIM4v4QdpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4v4QdpStructPtr = matched ;
                here->BSIM4v4QdpPtr = matched->CSC ;
            }

            if ((here-> BSIM4v4qNode != 0) && (here-> BSIM4v4sNodePrime != 0))
            {
                i = here->BSIM4v4QspPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4v4QspStructPtr = matched ;
                here->BSIM4v4QspPtr = matched->CSC ;
            }

            if ((here-> BSIM4v4qNode != 0) && (here-> BSIM4v4gNodePrime != 0))
            {
                i = here->BSIM4v4QgpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4v4QgpStructPtr = matched ;
                here->BSIM4v4QgpPtr = matched->CSC ;
            }

            if ((here-> BSIM4v4dNodePrime != 0) && (here-> BSIM4v4qNode != 0))
            {
                i = here->BSIM4v4DPqPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4v4DPqStructPtr = matched ;
                here->BSIM4v4DPqPtr = matched->CSC ;
            }

            if ((here-> BSIM4v4sNodePrime != 0) && (here-> BSIM4v4qNode != 0))
            {
                i = here->BSIM4v4SPqPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4v4SPqStructPtr = matched ;
                here->BSIM4v4SPqPtr = matched->CSC ;
            }

            if ((here-> BSIM4v4gNodePrime != 0) && (here-> BSIM4v4qNode != 0))
            {
                i = here->BSIM4v4GPqPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4v4GPqStructPtr = matched ;
                here->BSIM4v4GPqPtr = matched->CSC ;
            }

            if (here->BSIM4v4rgateMod != 0)
            {
                if ((here-> BSIM4v4gNodeExt != 0) && (here-> BSIM4v4gNodeExt != 0))
                {
                    i = here->BSIM4v4GEgePtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v4GEgeStructPtr = matched ;
                    here->BSIM4v4GEgePtr = matched->CSC ;
                }

                if ((here-> BSIM4v4gNodeExt != 0) && (here-> BSIM4v4gNodePrime != 0))
                {
                    i = here->BSIM4v4GEgpPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v4GEgpStructPtr = matched ;
                    here->BSIM4v4GEgpPtr = matched->CSC ;
                }

                if ((here-> BSIM4v4gNodePrime != 0) && (here-> BSIM4v4gNodeExt != 0))
                {
                    i = here->BSIM4v4GPgePtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v4GPgeStructPtr = matched ;
                    here->BSIM4v4GPgePtr = matched->CSC ;
                }

                if ((here-> BSIM4v4gNodeExt != 0) && (here-> BSIM4v4dNodePrime != 0))
                {
                    i = here->BSIM4v4GEdpPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v4GEdpStructPtr = matched ;
                    here->BSIM4v4GEdpPtr = matched->CSC ;
                }

                if ((here-> BSIM4v4gNodeExt != 0) && (here-> BSIM4v4sNodePrime != 0))
                {
                    i = here->BSIM4v4GEspPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v4GEspStructPtr = matched ;
                    here->BSIM4v4GEspPtr = matched->CSC ;
                }

                if ((here-> BSIM4v4gNodeExt != 0) && (here-> BSIM4v4bNodePrime != 0))
                {
                    i = here->BSIM4v4GEbpPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v4GEbpStructPtr = matched ;
                    here->BSIM4v4GEbpPtr = matched->CSC ;
                }

                if ((here-> BSIM4v4gNodeMid != 0) && (here-> BSIM4v4dNodePrime != 0))
                {
                    i = here->BSIM4v4GMdpPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v4GMdpStructPtr = matched ;
                    here->BSIM4v4GMdpPtr = matched->CSC ;
                }

                if ((here-> BSIM4v4gNodeMid != 0) && (here-> BSIM4v4gNodePrime != 0))
                {
                    i = here->BSIM4v4GMgpPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v4GMgpStructPtr = matched ;
                    here->BSIM4v4GMgpPtr = matched->CSC ;
                }

                if ((here-> BSIM4v4gNodeMid != 0) && (here-> BSIM4v4gNodeMid != 0))
                {
                    i = here->BSIM4v4GMgmPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v4GMgmStructPtr = matched ;
                    here->BSIM4v4GMgmPtr = matched->CSC ;
                }

                if ((here-> BSIM4v4gNodeMid != 0) && (here-> BSIM4v4gNodeExt != 0))
                {
                    i = here->BSIM4v4GMgePtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v4GMgeStructPtr = matched ;
                    here->BSIM4v4GMgePtr = matched->CSC ;
                }

                if ((here-> BSIM4v4gNodeMid != 0) && (here-> BSIM4v4sNodePrime != 0))
                {
                    i = here->BSIM4v4GMspPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v4GMspStructPtr = matched ;
                    here->BSIM4v4GMspPtr = matched->CSC ;
                }

                if ((here-> BSIM4v4gNodeMid != 0) && (here-> BSIM4v4bNodePrime != 0))
                {
                    i = here->BSIM4v4GMbpPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v4GMbpStructPtr = matched ;
                    here->BSIM4v4GMbpPtr = matched->CSC ;
                }

                if ((here-> BSIM4v4dNodePrime != 0) && (here-> BSIM4v4gNodeMid != 0))
                {
                    i = here->BSIM4v4DPgmPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v4DPgmStructPtr = matched ;
                    here->BSIM4v4DPgmPtr = matched->CSC ;
                }

                if ((here-> BSIM4v4gNodePrime != 0) && (here-> BSIM4v4gNodeMid != 0))
                {
                    i = here->BSIM4v4GPgmPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v4GPgmStructPtr = matched ;
                    here->BSIM4v4GPgmPtr = matched->CSC ;
                }

                if ((here-> BSIM4v4gNodeExt != 0) && (here-> BSIM4v4gNodeMid != 0))
                {
                    i = here->BSIM4v4GEgmPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v4GEgmStructPtr = matched ;
                    here->BSIM4v4GEgmPtr = matched->CSC ;
                }

                if ((here-> BSIM4v4sNodePrime != 0) && (here-> BSIM4v4gNodeMid != 0))
                {
                    i = here->BSIM4v4SPgmPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v4SPgmStructPtr = matched ;
                    here->BSIM4v4SPgmPtr = matched->CSC ;
                }

                if ((here-> BSIM4v4bNodePrime != 0) && (here-> BSIM4v4gNodeMid != 0))
                {
                    i = here->BSIM4v4BPgmPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v4BPgmStructPtr = matched ;
                    here->BSIM4v4BPgmPtr = matched->CSC ;
                }

            }
            if (here->BSIM4v4rbodyMod)
            {
                if ((here-> BSIM4v4dNodePrime != 0) && (here-> BSIM4v4dbNode != 0))
                {
                    i = here->BSIM4v4DPdbPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v4DPdbStructPtr = matched ;
                    here->BSIM4v4DPdbPtr = matched->CSC ;
                }

                if ((here-> BSIM4v4sNodePrime != 0) && (here-> BSIM4v4sbNode != 0))
                {
                    i = here->BSIM4v4SPsbPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v4SPsbStructPtr = matched ;
                    here->BSIM4v4SPsbPtr = matched->CSC ;
                }

                if ((here-> BSIM4v4dbNode != 0) && (here-> BSIM4v4dNodePrime != 0))
                {
                    i = here->BSIM4v4DBdpPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v4DBdpStructPtr = matched ;
                    here->BSIM4v4DBdpPtr = matched->CSC ;
                }

                if ((here-> BSIM4v4dbNode != 0) && (here-> BSIM4v4dbNode != 0))
                {
                    i = here->BSIM4v4DBdbPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v4DBdbStructPtr = matched ;
                    here->BSIM4v4DBdbPtr = matched->CSC ;
                }

                if ((here-> BSIM4v4dbNode != 0) && (here-> BSIM4v4bNodePrime != 0))
                {
                    i = here->BSIM4v4DBbpPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v4DBbpStructPtr = matched ;
                    here->BSIM4v4DBbpPtr = matched->CSC ;
                }

                if ((here-> BSIM4v4dbNode != 0) && (here-> BSIM4v4bNode != 0))
                {
                    i = here->BSIM4v4DBbPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v4DBbStructPtr = matched ;
                    here->BSIM4v4DBbPtr = matched->CSC ;
                }

                if ((here-> BSIM4v4bNodePrime != 0) && (here-> BSIM4v4dbNode != 0))
                {
                    i = here->BSIM4v4BPdbPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v4BPdbStructPtr = matched ;
                    here->BSIM4v4BPdbPtr = matched->CSC ;
                }

                if ((here-> BSIM4v4bNodePrime != 0) && (here-> BSIM4v4bNode != 0))
                {
                    i = here->BSIM4v4BPbPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v4BPbStructPtr = matched ;
                    here->BSIM4v4BPbPtr = matched->CSC ;
                }

                if ((here-> BSIM4v4bNodePrime != 0) && (here-> BSIM4v4sbNode != 0))
                {
                    i = here->BSIM4v4BPsbPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v4BPsbStructPtr = matched ;
                    here->BSIM4v4BPsbPtr = matched->CSC ;
                }

                if ((here-> BSIM4v4sbNode != 0) && (here-> BSIM4v4sNodePrime != 0))
                {
                    i = here->BSIM4v4SBspPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v4SBspStructPtr = matched ;
                    here->BSIM4v4SBspPtr = matched->CSC ;
                }

                if ((here-> BSIM4v4sbNode != 0) && (here-> BSIM4v4bNodePrime != 0))
                {
                    i = here->BSIM4v4SBbpPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v4SBbpStructPtr = matched ;
                    here->BSIM4v4SBbpPtr = matched->CSC ;
                }

                if ((here-> BSIM4v4sbNode != 0) && (here-> BSIM4v4bNode != 0))
                {
                    i = here->BSIM4v4SBbPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v4SBbStructPtr = matched ;
                    here->BSIM4v4SBbPtr = matched->CSC ;
                }

                if ((here-> BSIM4v4sbNode != 0) && (here-> BSIM4v4sbNode != 0))
                {
                    i = here->BSIM4v4SBsbPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v4SBsbStructPtr = matched ;
                    here->BSIM4v4SBsbPtr = matched->CSC ;
                }

                if ((here-> BSIM4v4bNode != 0) && (here-> BSIM4v4dbNode != 0))
                {
                    i = here->BSIM4v4BdbPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v4BdbStructPtr = matched ;
                    here->BSIM4v4BdbPtr = matched->CSC ;
                }

                if ((here-> BSIM4v4bNode != 0) && (here-> BSIM4v4bNodePrime != 0))
                {
                    i = here->BSIM4v4BbpPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v4BbpStructPtr = matched ;
                    here->BSIM4v4BbpPtr = matched->CSC ;
                }

                if ((here-> BSIM4v4bNode != 0) && (here-> BSIM4v4sbNode != 0))
                {
                    i = here->BSIM4v4BsbPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v4BsbStructPtr = matched ;
                    here->BSIM4v4BsbPtr = matched->CSC ;
                }

                if ((here-> BSIM4v4bNode != 0) && (here-> BSIM4v4bNode != 0))
                {
                    i = here->BSIM4v4BbPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v4BbStructPtr = matched ;
                    here->BSIM4v4BbPtr = matched->CSC ;
                }

            }
            if (model->BSIM4v4rdsMod)
            {
                if ((here-> BSIM4v4dNode != 0) && (here-> BSIM4v4gNodePrime != 0))
                {
                    i = here->BSIM4v4DgpPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v4DgpStructPtr = matched ;
                    here->BSIM4v4DgpPtr = matched->CSC ;
                }

                if ((here-> BSIM4v4dNode != 0) && (here-> BSIM4v4sNodePrime != 0))
                {
                    i = here->BSIM4v4DspPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v4DspStructPtr = matched ;
                    here->BSIM4v4DspPtr = matched->CSC ;
                }

                if ((here-> BSIM4v4dNode != 0) && (here-> BSIM4v4bNodePrime != 0))
                {
                    i = here->BSIM4v4DbpPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v4DbpStructPtr = matched ;
                    here->BSIM4v4DbpPtr = matched->CSC ;
                }

                if ((here-> BSIM4v4sNode != 0) && (here-> BSIM4v4dNodePrime != 0))
                {
                    i = here->BSIM4v4SdpPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v4SdpStructPtr = matched ;
                    here->BSIM4v4SdpPtr = matched->CSC ;
                }

                if ((here-> BSIM4v4sNode != 0) && (here-> BSIM4v4gNodePrime != 0))
                {
                    i = here->BSIM4v4SgpPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v4SgpStructPtr = matched ;
                    here->BSIM4v4SgpPtr = matched->CSC ;
                }

                if ((here-> BSIM4v4sNode != 0) && (here-> BSIM4v4bNodePrime != 0))
                {
                    i = here->BSIM4v4SbpPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4v4SbpStructPtr = matched ;
                    here->BSIM4v4SbpPtr = matched->CSC ;
                }

            }
        }
    }

    return (OK) ;
}

int
BSIM4v4bindCSCComplex (GENmodel *inModel, CKTcircuit *ckt)
{
    BSIM4v4model *model = (BSIM4v4model *)inModel ;
    BSIM4v4instance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the BSIM4v4 models */
    for ( ; model != NULL ; model = model->BSIM4v4nextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->BSIM4v4instances ; here != NULL ; here = here->BSIM4v4nextInstance)
        {
            if ((here-> BSIM4v4dNodePrime != 0) && (here-> BSIM4v4bNodePrime != 0))
                here->BSIM4v4DPbpPtr = here->BSIM4v4DPbpStructPtr->CSC_Complex ;

            if ((here-> BSIM4v4gNodePrime != 0) && (here-> BSIM4v4bNodePrime != 0))
                here->BSIM4v4GPbpPtr = here->BSIM4v4GPbpStructPtr->CSC_Complex ;

            if ((here-> BSIM4v4sNodePrime != 0) && (here-> BSIM4v4bNodePrime != 0))
                here->BSIM4v4SPbpPtr = here->BSIM4v4SPbpStructPtr->CSC_Complex ;

            if ((here-> BSIM4v4bNodePrime != 0) && (here-> BSIM4v4dNodePrime != 0))
                here->BSIM4v4BPdpPtr = here->BSIM4v4BPdpStructPtr->CSC_Complex ;

            if ((here-> BSIM4v4bNodePrime != 0) && (here-> BSIM4v4gNodePrime != 0))
                here->BSIM4v4BPgpPtr = here->BSIM4v4BPgpStructPtr->CSC_Complex ;

            if ((here-> BSIM4v4bNodePrime != 0) && (here-> BSIM4v4sNodePrime != 0))
                here->BSIM4v4BPspPtr = here->BSIM4v4BPspStructPtr->CSC_Complex ;

            if ((here-> BSIM4v4bNodePrime != 0) && (here-> BSIM4v4bNodePrime != 0))
                here->BSIM4v4BPbpPtr = here->BSIM4v4BPbpStructPtr->CSC_Complex ;

            if ((here-> BSIM4v4dNode != 0) && (here-> BSIM4v4dNode != 0))
                here->BSIM4v4DdPtr = here->BSIM4v4DdStructPtr->CSC_Complex ;

            if ((here-> BSIM4v4gNodePrime != 0) && (here-> BSIM4v4gNodePrime != 0))
                here->BSIM4v4GPgpPtr = here->BSIM4v4GPgpStructPtr->CSC_Complex ;

            if ((here-> BSIM4v4sNode != 0) && (here-> BSIM4v4sNode != 0))
                here->BSIM4v4SsPtr = here->BSIM4v4SsStructPtr->CSC_Complex ;

            if ((here-> BSIM4v4dNodePrime != 0) && (here-> BSIM4v4dNodePrime != 0))
                here->BSIM4v4DPdpPtr = here->BSIM4v4DPdpStructPtr->CSC_Complex ;

            if ((here-> BSIM4v4sNodePrime != 0) && (here-> BSIM4v4sNodePrime != 0))
                here->BSIM4v4SPspPtr = here->BSIM4v4SPspStructPtr->CSC_Complex ;

            if ((here-> BSIM4v4dNode != 0) && (here-> BSIM4v4dNodePrime != 0))
                here->BSIM4v4DdpPtr = here->BSIM4v4DdpStructPtr->CSC_Complex ;

            if ((here-> BSIM4v4gNodePrime != 0) && (here-> BSIM4v4dNodePrime != 0))
                here->BSIM4v4GPdpPtr = here->BSIM4v4GPdpStructPtr->CSC_Complex ;

            if ((here-> BSIM4v4gNodePrime != 0) && (here-> BSIM4v4sNodePrime != 0))
                here->BSIM4v4GPspPtr = here->BSIM4v4GPspStructPtr->CSC_Complex ;

            if ((here-> BSIM4v4sNode != 0) && (here-> BSIM4v4sNodePrime != 0))
                here->BSIM4v4SspPtr = here->BSIM4v4SspStructPtr->CSC_Complex ;

            if ((here-> BSIM4v4dNodePrime != 0) && (here-> BSIM4v4sNodePrime != 0))
                here->BSIM4v4DPspPtr = here->BSIM4v4DPspStructPtr->CSC_Complex ;

            if ((here-> BSIM4v4dNodePrime != 0) && (here-> BSIM4v4dNode != 0))
                here->BSIM4v4DPdPtr = here->BSIM4v4DPdStructPtr->CSC_Complex ;

            if ((here-> BSIM4v4dNodePrime != 0) && (here-> BSIM4v4gNodePrime != 0))
                here->BSIM4v4DPgpPtr = here->BSIM4v4DPgpStructPtr->CSC_Complex ;

            if ((here-> BSIM4v4sNodePrime != 0) && (here-> BSIM4v4gNodePrime != 0))
                here->BSIM4v4SPgpPtr = here->BSIM4v4SPgpStructPtr->CSC_Complex ;

            if ((here-> BSIM4v4sNodePrime != 0) && (here-> BSIM4v4sNode != 0))
                here->BSIM4v4SPsPtr = here->BSIM4v4SPsStructPtr->CSC_Complex ;

            if ((here-> BSIM4v4sNodePrime != 0) && (here-> BSIM4v4dNodePrime != 0))
                here->BSIM4v4SPdpPtr = here->BSIM4v4SPdpStructPtr->CSC_Complex ;

            if ((here-> BSIM4v4qNode != 0) && (here-> BSIM4v4qNode != 0))
                here->BSIM4v4QqPtr = here->BSIM4v4QqStructPtr->CSC_Complex ;

            if ((here-> BSIM4v4qNode != 0) && (here-> BSIM4v4bNodePrime != 0))
                here->BSIM4v4QbpPtr = here->BSIM4v4QbpStructPtr->CSC_Complex ;

            if ((here-> BSIM4v4qNode != 0) && (here-> BSIM4v4dNodePrime != 0))
                here->BSIM4v4QdpPtr = here->BSIM4v4QdpStructPtr->CSC_Complex ;

            if ((here-> BSIM4v4qNode != 0) && (here-> BSIM4v4sNodePrime != 0))
                here->BSIM4v4QspPtr = here->BSIM4v4QspStructPtr->CSC_Complex ;

            if ((here-> BSIM4v4qNode != 0) && (here-> BSIM4v4gNodePrime != 0))
                here->BSIM4v4QgpPtr = here->BSIM4v4QgpStructPtr->CSC_Complex ;

            if ((here-> BSIM4v4dNodePrime != 0) && (here-> BSIM4v4qNode != 0))
                here->BSIM4v4DPqPtr = here->BSIM4v4DPqStructPtr->CSC_Complex ;

            if ((here-> BSIM4v4sNodePrime != 0) && (here-> BSIM4v4qNode != 0))
                here->BSIM4v4SPqPtr = here->BSIM4v4SPqStructPtr->CSC_Complex ;

            if ((here-> BSIM4v4gNodePrime != 0) && (here-> BSIM4v4qNode != 0))
                here->BSIM4v4GPqPtr = here->BSIM4v4GPqStructPtr->CSC_Complex ;

            if (here->BSIM4v4rgateMod != 0)
            {
                if ((here-> BSIM4v4gNodeExt != 0) && (here-> BSIM4v4gNodeExt != 0))
                    here->BSIM4v4GEgePtr = here->BSIM4v4GEgeStructPtr->CSC_Complex ;

                if ((here-> BSIM4v4gNodeExt != 0) && (here-> BSIM4v4gNodePrime != 0))
                    here->BSIM4v4GEgpPtr = here->BSIM4v4GEgpStructPtr->CSC_Complex ;

                if ((here-> BSIM4v4gNodePrime != 0) && (here-> BSIM4v4gNodeExt != 0))
                    here->BSIM4v4GPgePtr = here->BSIM4v4GPgeStructPtr->CSC_Complex ;

                if ((here-> BSIM4v4gNodeExt != 0) && (here-> BSIM4v4dNodePrime != 0))
                    here->BSIM4v4GEdpPtr = here->BSIM4v4GEdpStructPtr->CSC_Complex ;

                if ((here-> BSIM4v4gNodeExt != 0) && (here-> BSIM4v4sNodePrime != 0))
                    here->BSIM4v4GEspPtr = here->BSIM4v4GEspStructPtr->CSC_Complex ;

                if ((here-> BSIM4v4gNodeExt != 0) && (here-> BSIM4v4bNodePrime != 0))
                    here->BSIM4v4GEbpPtr = here->BSIM4v4GEbpStructPtr->CSC_Complex ;

                if ((here-> BSIM4v4gNodeMid != 0) && (here-> BSIM4v4dNodePrime != 0))
                    here->BSIM4v4GMdpPtr = here->BSIM4v4GMdpStructPtr->CSC_Complex ;

                if ((here-> BSIM4v4gNodeMid != 0) && (here-> BSIM4v4gNodePrime != 0))
                    here->BSIM4v4GMgpPtr = here->BSIM4v4GMgpStructPtr->CSC_Complex ;

                if ((here-> BSIM4v4gNodeMid != 0) && (here-> BSIM4v4gNodeMid != 0))
                    here->BSIM4v4GMgmPtr = here->BSIM4v4GMgmStructPtr->CSC_Complex ;

                if ((here-> BSIM4v4gNodeMid != 0) && (here-> BSIM4v4gNodeExt != 0))
                    here->BSIM4v4GMgePtr = here->BSIM4v4GMgeStructPtr->CSC_Complex ;

                if ((here-> BSIM4v4gNodeMid != 0) && (here-> BSIM4v4sNodePrime != 0))
                    here->BSIM4v4GMspPtr = here->BSIM4v4GMspStructPtr->CSC_Complex ;

                if ((here-> BSIM4v4gNodeMid != 0) && (here-> BSIM4v4bNodePrime != 0))
                    here->BSIM4v4GMbpPtr = here->BSIM4v4GMbpStructPtr->CSC_Complex ;

                if ((here-> BSIM4v4dNodePrime != 0) && (here-> BSIM4v4gNodeMid != 0))
                    here->BSIM4v4DPgmPtr = here->BSIM4v4DPgmStructPtr->CSC_Complex ;

                if ((here-> BSIM4v4gNodePrime != 0) && (here-> BSIM4v4gNodeMid != 0))
                    here->BSIM4v4GPgmPtr = here->BSIM4v4GPgmStructPtr->CSC_Complex ;

                if ((here-> BSIM4v4gNodeExt != 0) && (here-> BSIM4v4gNodeMid != 0))
                    here->BSIM4v4GEgmPtr = here->BSIM4v4GEgmStructPtr->CSC_Complex ;

                if ((here-> BSIM4v4sNodePrime != 0) && (here-> BSIM4v4gNodeMid != 0))
                    here->BSIM4v4SPgmPtr = here->BSIM4v4SPgmStructPtr->CSC_Complex ;

                if ((here-> BSIM4v4bNodePrime != 0) && (here-> BSIM4v4gNodeMid != 0))
                    here->BSIM4v4BPgmPtr = here->BSIM4v4BPgmStructPtr->CSC_Complex ;

            }
            if (here->BSIM4v4rbodyMod)
            {
                if ((here-> BSIM4v4dNodePrime != 0) && (here-> BSIM4v4dbNode != 0))
                    here->BSIM4v4DPdbPtr = here->BSIM4v4DPdbStructPtr->CSC_Complex ;

                if ((here-> BSIM4v4sNodePrime != 0) && (here-> BSIM4v4sbNode != 0))
                    here->BSIM4v4SPsbPtr = here->BSIM4v4SPsbStructPtr->CSC_Complex ;

                if ((here-> BSIM4v4dbNode != 0) && (here-> BSIM4v4dNodePrime != 0))
                    here->BSIM4v4DBdpPtr = here->BSIM4v4DBdpStructPtr->CSC_Complex ;

                if ((here-> BSIM4v4dbNode != 0) && (here-> BSIM4v4dbNode != 0))
                    here->BSIM4v4DBdbPtr = here->BSIM4v4DBdbStructPtr->CSC_Complex ;

                if ((here-> BSIM4v4dbNode != 0) && (here-> BSIM4v4bNodePrime != 0))
                    here->BSIM4v4DBbpPtr = here->BSIM4v4DBbpStructPtr->CSC_Complex ;

                if ((here-> BSIM4v4dbNode != 0) && (here-> BSIM4v4bNode != 0))
                    here->BSIM4v4DBbPtr = here->BSIM4v4DBbStructPtr->CSC_Complex ;

                if ((here-> BSIM4v4bNodePrime != 0) && (here-> BSIM4v4dbNode != 0))
                    here->BSIM4v4BPdbPtr = here->BSIM4v4BPdbStructPtr->CSC_Complex ;

                if ((here-> BSIM4v4bNodePrime != 0) && (here-> BSIM4v4bNode != 0))
                    here->BSIM4v4BPbPtr = here->BSIM4v4BPbStructPtr->CSC_Complex ;

                if ((here-> BSIM4v4bNodePrime != 0) && (here-> BSIM4v4sbNode != 0))
                    here->BSIM4v4BPsbPtr = here->BSIM4v4BPsbStructPtr->CSC_Complex ;

                if ((here-> BSIM4v4sbNode != 0) && (here-> BSIM4v4sNodePrime != 0))
                    here->BSIM4v4SBspPtr = here->BSIM4v4SBspStructPtr->CSC_Complex ;

                if ((here-> BSIM4v4sbNode != 0) && (here-> BSIM4v4bNodePrime != 0))
                    here->BSIM4v4SBbpPtr = here->BSIM4v4SBbpStructPtr->CSC_Complex ;

                if ((here-> BSIM4v4sbNode != 0) && (here-> BSIM4v4bNode != 0))
                    here->BSIM4v4SBbPtr = here->BSIM4v4SBbStructPtr->CSC_Complex ;

                if ((here-> BSIM4v4sbNode != 0) && (here-> BSIM4v4sbNode != 0))
                    here->BSIM4v4SBsbPtr = here->BSIM4v4SBsbStructPtr->CSC_Complex ;

                if ((here-> BSIM4v4bNode != 0) && (here-> BSIM4v4dbNode != 0))
                    here->BSIM4v4BdbPtr = here->BSIM4v4BdbStructPtr->CSC_Complex ;

                if ((here-> BSIM4v4bNode != 0) && (here-> BSIM4v4bNodePrime != 0))
                    here->BSIM4v4BbpPtr = here->BSIM4v4BbpStructPtr->CSC_Complex ;

                if ((here-> BSIM4v4bNode != 0) && (here-> BSIM4v4sbNode != 0))
                    here->BSIM4v4BsbPtr = here->BSIM4v4BsbStructPtr->CSC_Complex ;

                if ((here-> BSIM4v4bNode != 0) && (here-> BSIM4v4bNode != 0))
                    here->BSIM4v4BbPtr = here->BSIM4v4BbStructPtr->CSC_Complex ;

            }
            if (model->BSIM4v4rdsMod)
            {
                if ((here-> BSIM4v4dNode != 0) && (here-> BSIM4v4gNodePrime != 0))
                    here->BSIM4v4DgpPtr = here->BSIM4v4DgpStructPtr->CSC_Complex ;

                if ((here-> BSIM4v4dNode != 0) && (here-> BSIM4v4sNodePrime != 0))
                    here->BSIM4v4DspPtr = here->BSIM4v4DspStructPtr->CSC_Complex ;

                if ((here-> BSIM4v4dNode != 0) && (here-> BSIM4v4bNodePrime != 0))
                    here->BSIM4v4DbpPtr = here->BSIM4v4DbpStructPtr->CSC_Complex ;

                if ((here-> BSIM4v4sNode != 0) && (here-> BSIM4v4dNodePrime != 0))
                    here->BSIM4v4SdpPtr = here->BSIM4v4SdpStructPtr->CSC_Complex ;

                if ((here-> BSIM4v4sNode != 0) && (here-> BSIM4v4gNodePrime != 0))
                    here->BSIM4v4SgpPtr = here->BSIM4v4SgpStructPtr->CSC_Complex ;

                if ((here-> BSIM4v4sNode != 0) && (here-> BSIM4v4bNodePrime != 0))
                    here->BSIM4v4SbpPtr = here->BSIM4v4SbpStructPtr->CSC_Complex ;

            }
        }
    }

    return (OK) ;
}

int
BSIM4v4bindCSCComplexToReal (GENmodel *inModel, CKTcircuit *ckt)
{
    BSIM4v4model *model = (BSIM4v4model *)inModel ;
    BSIM4v4instance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the BSIM4v4 models */
    for ( ; model != NULL ; model = model->BSIM4v4nextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->BSIM4v4instances ; here != NULL ; here = here->BSIM4v4nextInstance)
        {
            if ((here-> BSIM4v4dNodePrime != 0) && (here-> BSIM4v4bNodePrime != 0))
                here->BSIM4v4DPbpPtr = here->BSIM4v4DPbpStructPtr->CSC ;

            if ((here-> BSIM4v4gNodePrime != 0) && (here-> BSIM4v4bNodePrime != 0))
                here->BSIM4v4GPbpPtr = here->BSIM4v4GPbpStructPtr->CSC ;

            if ((here-> BSIM4v4sNodePrime != 0) && (here-> BSIM4v4bNodePrime != 0))
                here->BSIM4v4SPbpPtr = here->BSIM4v4SPbpStructPtr->CSC ;

            if ((here-> BSIM4v4bNodePrime != 0) && (here-> BSIM4v4dNodePrime != 0))
                here->BSIM4v4BPdpPtr = here->BSIM4v4BPdpStructPtr->CSC ;

            if ((here-> BSIM4v4bNodePrime != 0) && (here-> BSIM4v4gNodePrime != 0))
                here->BSIM4v4BPgpPtr = here->BSIM4v4BPgpStructPtr->CSC ;

            if ((here-> BSIM4v4bNodePrime != 0) && (here-> BSIM4v4sNodePrime != 0))
                here->BSIM4v4BPspPtr = here->BSIM4v4BPspStructPtr->CSC ;

            if ((here-> BSIM4v4bNodePrime != 0) && (here-> BSIM4v4bNodePrime != 0))
                here->BSIM4v4BPbpPtr = here->BSIM4v4BPbpStructPtr->CSC ;

            if ((here-> BSIM4v4dNode != 0) && (here-> BSIM4v4dNode != 0))
                here->BSIM4v4DdPtr = here->BSIM4v4DdStructPtr->CSC ;

            if ((here-> BSIM4v4gNodePrime != 0) && (here-> BSIM4v4gNodePrime != 0))
                here->BSIM4v4GPgpPtr = here->BSIM4v4GPgpStructPtr->CSC ;

            if ((here-> BSIM4v4sNode != 0) && (here-> BSIM4v4sNode != 0))
                here->BSIM4v4SsPtr = here->BSIM4v4SsStructPtr->CSC ;

            if ((here-> BSIM4v4dNodePrime != 0) && (here-> BSIM4v4dNodePrime != 0))
                here->BSIM4v4DPdpPtr = here->BSIM4v4DPdpStructPtr->CSC ;

            if ((here-> BSIM4v4sNodePrime != 0) && (here-> BSIM4v4sNodePrime != 0))
                here->BSIM4v4SPspPtr = here->BSIM4v4SPspStructPtr->CSC ;

            if ((here-> BSIM4v4dNode != 0) && (here-> BSIM4v4dNodePrime != 0))
                here->BSIM4v4DdpPtr = here->BSIM4v4DdpStructPtr->CSC ;

            if ((here-> BSIM4v4gNodePrime != 0) && (here-> BSIM4v4dNodePrime != 0))
                here->BSIM4v4GPdpPtr = here->BSIM4v4GPdpStructPtr->CSC ;

            if ((here-> BSIM4v4gNodePrime != 0) && (here-> BSIM4v4sNodePrime != 0))
                here->BSIM4v4GPspPtr = here->BSIM4v4GPspStructPtr->CSC ;

            if ((here-> BSIM4v4sNode != 0) && (here-> BSIM4v4sNodePrime != 0))
                here->BSIM4v4SspPtr = here->BSIM4v4SspStructPtr->CSC ;

            if ((here-> BSIM4v4dNodePrime != 0) && (here-> BSIM4v4sNodePrime != 0))
                here->BSIM4v4DPspPtr = here->BSIM4v4DPspStructPtr->CSC ;

            if ((here-> BSIM4v4dNodePrime != 0) && (here-> BSIM4v4dNode != 0))
                here->BSIM4v4DPdPtr = here->BSIM4v4DPdStructPtr->CSC ;

            if ((here-> BSIM4v4dNodePrime != 0) && (here-> BSIM4v4gNodePrime != 0))
                here->BSIM4v4DPgpPtr = here->BSIM4v4DPgpStructPtr->CSC ;

            if ((here-> BSIM4v4sNodePrime != 0) && (here-> BSIM4v4gNodePrime != 0))
                here->BSIM4v4SPgpPtr = here->BSIM4v4SPgpStructPtr->CSC ;

            if ((here-> BSIM4v4sNodePrime != 0) && (here-> BSIM4v4sNode != 0))
                here->BSIM4v4SPsPtr = here->BSIM4v4SPsStructPtr->CSC ;

            if ((here-> BSIM4v4sNodePrime != 0) && (here-> BSIM4v4dNodePrime != 0))
                here->BSIM4v4SPdpPtr = here->BSIM4v4SPdpStructPtr->CSC ;

            if ((here-> BSIM4v4qNode != 0) && (here-> BSIM4v4qNode != 0))
                here->BSIM4v4QqPtr = here->BSIM4v4QqStructPtr->CSC ;

            if ((here-> BSIM4v4qNode != 0) && (here-> BSIM4v4bNodePrime != 0))
                here->BSIM4v4QbpPtr = here->BSIM4v4QbpStructPtr->CSC ;

            if ((here-> BSIM4v4qNode != 0) && (here-> BSIM4v4dNodePrime != 0))
                here->BSIM4v4QdpPtr = here->BSIM4v4QdpStructPtr->CSC ;

            if ((here-> BSIM4v4qNode != 0) && (here-> BSIM4v4sNodePrime != 0))
                here->BSIM4v4QspPtr = here->BSIM4v4QspStructPtr->CSC ;

            if ((here-> BSIM4v4qNode != 0) && (here-> BSIM4v4gNodePrime != 0))
                here->BSIM4v4QgpPtr = here->BSIM4v4QgpStructPtr->CSC ;

            if ((here-> BSIM4v4dNodePrime != 0) && (here-> BSIM4v4qNode != 0))
                here->BSIM4v4DPqPtr = here->BSIM4v4DPqStructPtr->CSC ;

            if ((here-> BSIM4v4sNodePrime != 0) && (here-> BSIM4v4qNode != 0))
                here->BSIM4v4SPqPtr = here->BSIM4v4SPqStructPtr->CSC ;

            if ((here-> BSIM4v4gNodePrime != 0) && (here-> BSIM4v4qNode != 0))
                here->BSIM4v4GPqPtr = here->BSIM4v4GPqStructPtr->CSC ;

            if (here->BSIM4v4rgateMod != 0)
            {
                if ((here-> BSIM4v4gNodeExt != 0) && (here-> BSIM4v4gNodeExt != 0))
                    here->BSIM4v4GEgePtr = here->BSIM4v4GEgeStructPtr->CSC ;

                if ((here-> BSIM4v4gNodeExt != 0) && (here-> BSIM4v4gNodePrime != 0))
                    here->BSIM4v4GEgpPtr = here->BSIM4v4GEgpStructPtr->CSC ;

                if ((here-> BSIM4v4gNodePrime != 0) && (here-> BSIM4v4gNodeExt != 0))
                    here->BSIM4v4GPgePtr = here->BSIM4v4GPgeStructPtr->CSC ;

                if ((here-> BSIM4v4gNodeExt != 0) && (here-> BSIM4v4dNodePrime != 0))
                    here->BSIM4v4GEdpPtr = here->BSIM4v4GEdpStructPtr->CSC ;

                if ((here-> BSIM4v4gNodeExt != 0) && (here-> BSIM4v4sNodePrime != 0))
                    here->BSIM4v4GEspPtr = here->BSIM4v4GEspStructPtr->CSC ;

                if ((here-> BSIM4v4gNodeExt != 0) && (here-> BSIM4v4bNodePrime != 0))
                    here->BSIM4v4GEbpPtr = here->BSIM4v4GEbpStructPtr->CSC ;

                if ((here-> BSIM4v4gNodeMid != 0) && (here-> BSIM4v4dNodePrime != 0))
                    here->BSIM4v4GMdpPtr = here->BSIM4v4GMdpStructPtr->CSC ;

                if ((here-> BSIM4v4gNodeMid != 0) && (here-> BSIM4v4gNodePrime != 0))
                    here->BSIM4v4GMgpPtr = here->BSIM4v4GMgpStructPtr->CSC ;

                if ((here-> BSIM4v4gNodeMid != 0) && (here-> BSIM4v4gNodeMid != 0))
                    here->BSIM4v4GMgmPtr = here->BSIM4v4GMgmStructPtr->CSC ;

                if ((here-> BSIM4v4gNodeMid != 0) && (here-> BSIM4v4gNodeExt != 0))
                    here->BSIM4v4GMgePtr = here->BSIM4v4GMgeStructPtr->CSC ;

                if ((here-> BSIM4v4gNodeMid != 0) && (here-> BSIM4v4sNodePrime != 0))
                    here->BSIM4v4GMspPtr = here->BSIM4v4GMspStructPtr->CSC ;

                if ((here-> BSIM4v4gNodeMid != 0) && (here-> BSIM4v4bNodePrime != 0))
                    here->BSIM4v4GMbpPtr = here->BSIM4v4GMbpStructPtr->CSC ;

                if ((here-> BSIM4v4dNodePrime != 0) && (here-> BSIM4v4gNodeMid != 0))
                    here->BSIM4v4DPgmPtr = here->BSIM4v4DPgmStructPtr->CSC ;

                if ((here-> BSIM4v4gNodePrime != 0) && (here-> BSIM4v4gNodeMid != 0))
                    here->BSIM4v4GPgmPtr = here->BSIM4v4GPgmStructPtr->CSC ;

                if ((here-> BSIM4v4gNodeExt != 0) && (here-> BSIM4v4gNodeMid != 0))
                    here->BSIM4v4GEgmPtr = here->BSIM4v4GEgmStructPtr->CSC ;

                if ((here-> BSIM4v4sNodePrime != 0) && (here-> BSIM4v4gNodeMid != 0))
                    here->BSIM4v4SPgmPtr = here->BSIM4v4SPgmStructPtr->CSC ;

                if ((here-> BSIM4v4bNodePrime != 0) && (here-> BSIM4v4gNodeMid != 0))
                    here->BSIM4v4BPgmPtr = here->BSIM4v4BPgmStructPtr->CSC ;

            }
            if (here->BSIM4v4rbodyMod)
            {
                if ((here-> BSIM4v4dNodePrime != 0) && (here-> BSIM4v4dbNode != 0))
                    here->BSIM4v4DPdbPtr = here->BSIM4v4DPdbStructPtr->CSC ;

                if ((here-> BSIM4v4sNodePrime != 0) && (here-> BSIM4v4sbNode != 0))
                    here->BSIM4v4SPsbPtr = here->BSIM4v4SPsbStructPtr->CSC ;

                if ((here-> BSIM4v4dbNode != 0) && (here-> BSIM4v4dNodePrime != 0))
                    here->BSIM4v4DBdpPtr = here->BSIM4v4DBdpStructPtr->CSC ;

                if ((here-> BSIM4v4dbNode != 0) && (here-> BSIM4v4dbNode != 0))
                    here->BSIM4v4DBdbPtr = here->BSIM4v4DBdbStructPtr->CSC ;

                if ((here-> BSIM4v4dbNode != 0) && (here-> BSIM4v4bNodePrime != 0))
                    here->BSIM4v4DBbpPtr = here->BSIM4v4DBbpStructPtr->CSC ;

                if ((here-> BSIM4v4dbNode != 0) && (here-> BSIM4v4bNode != 0))
                    here->BSIM4v4DBbPtr = here->BSIM4v4DBbStructPtr->CSC ;

                if ((here-> BSIM4v4bNodePrime != 0) && (here-> BSIM4v4dbNode != 0))
                    here->BSIM4v4BPdbPtr = here->BSIM4v4BPdbStructPtr->CSC ;

                if ((here-> BSIM4v4bNodePrime != 0) && (here-> BSIM4v4bNode != 0))
                    here->BSIM4v4BPbPtr = here->BSIM4v4BPbStructPtr->CSC ;

                if ((here-> BSIM4v4bNodePrime != 0) && (here-> BSIM4v4sbNode != 0))
                    here->BSIM4v4BPsbPtr = here->BSIM4v4BPsbStructPtr->CSC ;

                if ((here-> BSIM4v4sbNode != 0) && (here-> BSIM4v4sNodePrime != 0))
                    here->BSIM4v4SBspPtr = here->BSIM4v4SBspStructPtr->CSC ;

                if ((here-> BSIM4v4sbNode != 0) && (here-> BSIM4v4bNodePrime != 0))
                    here->BSIM4v4SBbpPtr = here->BSIM4v4SBbpStructPtr->CSC ;

                if ((here-> BSIM4v4sbNode != 0) && (here-> BSIM4v4bNode != 0))
                    here->BSIM4v4SBbPtr = here->BSIM4v4SBbStructPtr->CSC ;

                if ((here-> BSIM4v4sbNode != 0) && (here-> BSIM4v4sbNode != 0))
                    here->BSIM4v4SBsbPtr = here->BSIM4v4SBsbStructPtr->CSC ;

                if ((here-> BSIM4v4bNode != 0) && (here-> BSIM4v4dbNode != 0))
                    here->BSIM4v4BdbPtr = here->BSIM4v4BdbStructPtr->CSC ;

                if ((here-> BSIM4v4bNode != 0) && (here-> BSIM4v4bNodePrime != 0))
                    here->BSIM4v4BbpPtr = here->BSIM4v4BbpStructPtr->CSC ;

                if ((here-> BSIM4v4bNode != 0) && (here-> BSIM4v4sbNode != 0))
                    here->BSIM4v4BsbPtr = here->BSIM4v4BsbStructPtr->CSC ;

                if ((here-> BSIM4v4bNode != 0) && (here-> BSIM4v4bNode != 0))
                    here->BSIM4v4BbPtr = here->BSIM4v4BbStructPtr->CSC ;

            }
            if (model->BSIM4v4rdsMod)
            {
                if ((here-> BSIM4v4dNode != 0) && (here-> BSIM4v4gNodePrime != 0))
                    here->BSIM4v4DgpPtr = here->BSIM4v4DgpStructPtr->CSC ;

                if ((here-> BSIM4v4dNode != 0) && (here-> BSIM4v4sNodePrime != 0))
                    here->BSIM4v4DspPtr = here->BSIM4v4DspStructPtr->CSC ;

                if ((here-> BSIM4v4dNode != 0) && (here-> BSIM4v4bNodePrime != 0))
                    here->BSIM4v4DbpPtr = here->BSIM4v4DbpStructPtr->CSC ;

                if ((here-> BSIM4v4sNode != 0) && (here-> BSIM4v4dNodePrime != 0))
                    here->BSIM4v4SdpPtr = here->BSIM4v4SdpStructPtr->CSC ;

                if ((here-> BSIM4v4sNode != 0) && (here-> BSIM4v4gNodePrime != 0))
                    here->BSIM4v4SgpPtr = here->BSIM4v4SgpStructPtr->CSC ;

                if ((here-> BSIM4v4sNode != 0) && (here-> BSIM4v4bNodePrime != 0))
                    here->BSIM4v4SbpPtr = here->BSIM4v4SbpStructPtr->CSC ;

            }
        }
    }

    return (OK) ;
}
