/**********
Author: 2013 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "bsim4def.h"
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
BSIM4bindCSC (GENmodel *inModel, CKTcircuit *ckt)
{
    BSIM4model *model = (BSIM4model *)inModel ;
    BSIM4instance *here ;
    double *i ;
    BindElement *matched, *BindStruct ;
    size_t nz ;

    BindStruct = ckt->CKTmatrix->CKTbindStruct ;
    nz = (size_t)ckt->CKTmatrix->CKTklunz ;

    /* loop through all the BSIM4 models */
    for ( ; model != NULL ; model = model->BSIM4nextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->BSIM4instances ; here != NULL ; here = here->BSIM4nextInstance)
        {
            if ((here-> BSIM4dNodePrime != 0) && (here-> BSIM4bNodePrime != 0))
            {
                i = here->BSIM4DPbpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4DPbpStructPtr = matched ;
                here->BSIM4DPbpPtr = matched->CSC ;
            }

            if ((here-> BSIM4gNodePrime != 0) && (here-> BSIM4bNodePrime != 0))
            {
                i = here->BSIM4GPbpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4GPbpStructPtr = matched ;
                here->BSIM4GPbpPtr = matched->CSC ;
            }

            if ((here-> BSIM4sNodePrime != 0) && (here-> BSIM4bNodePrime != 0))
            {
                i = here->BSIM4SPbpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4SPbpStructPtr = matched ;
                here->BSIM4SPbpPtr = matched->CSC ;
            }

            if ((here-> BSIM4bNodePrime != 0) && (here-> BSIM4dNodePrime != 0))
            {
                i = here->BSIM4BPdpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4BPdpStructPtr = matched ;
                here->BSIM4BPdpPtr = matched->CSC ;
            }

            if ((here-> BSIM4bNodePrime != 0) && (here-> BSIM4gNodePrime != 0))
            {
                i = here->BSIM4BPgpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4BPgpStructPtr = matched ;
                here->BSIM4BPgpPtr = matched->CSC ;
            }

            if ((here-> BSIM4bNodePrime != 0) && (here-> BSIM4sNodePrime != 0))
            {
                i = here->BSIM4BPspPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4BPspStructPtr = matched ;
                here->BSIM4BPspPtr = matched->CSC ;
            }

            if ((here-> BSIM4bNodePrime != 0) && (here-> BSIM4bNodePrime != 0))
            {
                i = here->BSIM4BPbpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4BPbpStructPtr = matched ;
                here->BSIM4BPbpPtr = matched->CSC ;
            }

            if ((here-> BSIM4dNode != 0) && (here-> BSIM4dNode != 0))
            {
                i = here->BSIM4DdPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4DdStructPtr = matched ;
                here->BSIM4DdPtr = matched->CSC ;
            }

            if ((here-> BSIM4gNodePrime != 0) && (here-> BSIM4gNodePrime != 0))
            {
                i = here->BSIM4GPgpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4GPgpStructPtr = matched ;
                here->BSIM4GPgpPtr = matched->CSC ;
            }

            if ((here-> BSIM4sNode != 0) && (here-> BSIM4sNode != 0))
            {
                i = here->BSIM4SsPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4SsStructPtr = matched ;
                here->BSIM4SsPtr = matched->CSC ;
            }

            if ((here-> BSIM4dNodePrime != 0) && (here-> BSIM4dNodePrime != 0))
            {
                i = here->BSIM4DPdpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4DPdpStructPtr = matched ;
                here->BSIM4DPdpPtr = matched->CSC ;
            }

            if ((here-> BSIM4sNodePrime != 0) && (here-> BSIM4sNodePrime != 0))
            {
                i = here->BSIM4SPspPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4SPspStructPtr = matched ;
                here->BSIM4SPspPtr = matched->CSC ;
            }

            if ((here-> BSIM4dNode != 0) && (here-> BSIM4dNodePrime != 0))
            {
                i = here->BSIM4DdpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4DdpStructPtr = matched ;
                here->BSIM4DdpPtr = matched->CSC ;
            }

            if ((here-> BSIM4gNodePrime != 0) && (here-> BSIM4dNodePrime != 0))
            {
                i = here->BSIM4GPdpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4GPdpStructPtr = matched ;
                here->BSIM4GPdpPtr = matched->CSC ;
            }

            if ((here-> BSIM4gNodePrime != 0) && (here-> BSIM4sNodePrime != 0))
            {
                i = here->BSIM4GPspPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4GPspStructPtr = matched ;
                here->BSIM4GPspPtr = matched->CSC ;
            }

            if ((here-> BSIM4sNode != 0) && (here-> BSIM4sNodePrime != 0))
            {
                i = here->BSIM4SspPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4SspStructPtr = matched ;
                here->BSIM4SspPtr = matched->CSC ;
            }

            if ((here-> BSIM4dNodePrime != 0) && (here-> BSIM4sNodePrime != 0))
            {
                i = here->BSIM4DPspPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4DPspStructPtr = matched ;
                here->BSIM4DPspPtr = matched->CSC ;
            }

            if ((here-> BSIM4dNodePrime != 0) && (here-> BSIM4dNode != 0))
            {
                i = here->BSIM4DPdPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4DPdStructPtr = matched ;
                here->BSIM4DPdPtr = matched->CSC ;
            }

            if ((here-> BSIM4dNodePrime != 0) && (here-> BSIM4gNodePrime != 0))
            {
                i = here->BSIM4DPgpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4DPgpStructPtr = matched ;
                here->BSIM4DPgpPtr = matched->CSC ;
            }

            if ((here-> BSIM4sNodePrime != 0) && (here-> BSIM4gNodePrime != 0))
            {
                i = here->BSIM4SPgpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4SPgpStructPtr = matched ;
                here->BSIM4SPgpPtr = matched->CSC ;
            }

            if ((here-> BSIM4sNodePrime != 0) && (here-> BSIM4sNode != 0))
            {
                i = here->BSIM4SPsPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4SPsStructPtr = matched ;
                here->BSIM4SPsPtr = matched->CSC ;
            }

            if ((here-> BSIM4sNodePrime != 0) && (here-> BSIM4dNodePrime != 0))
            {
                i = here->BSIM4SPdpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4SPdpStructPtr = matched ;
                here->BSIM4SPdpPtr = matched->CSC ;
            }

            if ((here-> BSIM4qNode != 0) && (here-> BSIM4qNode != 0))
            {
                i = here->BSIM4QqPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4QqStructPtr = matched ;
                here->BSIM4QqPtr = matched->CSC ;
            }

            if ((here-> BSIM4qNode != 0) && (here-> BSIM4bNodePrime != 0))
            {
                i = here->BSIM4QbpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4QbpStructPtr = matched ;
                here->BSIM4QbpPtr = matched->CSC ;
            }

            if ((here-> BSIM4qNode != 0) && (here-> BSIM4dNodePrime != 0))
            {
                i = here->BSIM4QdpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4QdpStructPtr = matched ;
                here->BSIM4QdpPtr = matched->CSC ;
            }

            if ((here-> BSIM4qNode != 0) && (here-> BSIM4sNodePrime != 0))
            {
                i = here->BSIM4QspPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4QspStructPtr = matched ;
                here->BSIM4QspPtr = matched->CSC ;
            }

            if ((here-> BSIM4qNode != 0) && (here-> BSIM4gNodePrime != 0))
            {
                i = here->BSIM4QgpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4QgpStructPtr = matched ;
                here->BSIM4QgpPtr = matched->CSC ;
            }

            if ((here-> BSIM4dNodePrime != 0) && (here-> BSIM4qNode != 0))
            {
                i = here->BSIM4DPqPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4DPqStructPtr = matched ;
                here->BSIM4DPqPtr = matched->CSC ;
            }

            if ((here-> BSIM4sNodePrime != 0) && (here-> BSIM4qNode != 0))
            {
                i = here->BSIM4SPqPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4SPqStructPtr = matched ;
                here->BSIM4SPqPtr = matched->CSC ;
            }

            if ((here-> BSIM4gNodePrime != 0) && (here-> BSIM4qNode != 0))
            {
                i = here->BSIM4GPqPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->BSIM4GPqStructPtr = matched ;
                here->BSIM4GPqPtr = matched->CSC ;
            }

            if (here->BSIM4rgateMod != 0)
            {
                if ((here-> BSIM4gNodeExt != 0) && (here-> BSIM4gNodeExt != 0))
                {
                    i = here->BSIM4GEgePtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4GEgeStructPtr = matched ;
                    here->BSIM4GEgePtr = matched->CSC ;
                }

                if ((here-> BSIM4gNodeExt != 0) && (here-> BSIM4gNodePrime != 0))
                {
                    i = here->BSIM4GEgpPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4GEgpStructPtr = matched ;
                    here->BSIM4GEgpPtr = matched->CSC ;
                }

                if ((here-> BSIM4gNodePrime != 0) && (here-> BSIM4gNodeExt != 0))
                {
                    i = here->BSIM4GPgePtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4GPgeStructPtr = matched ;
                    here->BSIM4GPgePtr = matched->CSC ;
                }

                if ((here-> BSIM4gNodeExt != 0) && (here-> BSIM4dNodePrime != 0))
                {
                    i = here->BSIM4GEdpPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4GEdpStructPtr = matched ;
                    here->BSIM4GEdpPtr = matched->CSC ;
                }

                if ((here-> BSIM4gNodeExt != 0) && (here-> BSIM4sNodePrime != 0))
                {
                    i = here->BSIM4GEspPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4GEspStructPtr = matched ;
                    here->BSIM4GEspPtr = matched->CSC ;
                }

                if ((here-> BSIM4gNodeExt != 0) && (here-> BSIM4bNodePrime != 0))
                {
                    i = here->BSIM4GEbpPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4GEbpStructPtr = matched ;
                    here->BSIM4GEbpPtr = matched->CSC ;
                }

                if ((here-> BSIM4gNodeMid != 0) && (here-> BSIM4dNodePrime != 0))
                {
                    i = here->BSIM4GMdpPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4GMdpStructPtr = matched ;
                    here->BSIM4GMdpPtr = matched->CSC ;
                }

                if ((here-> BSIM4gNodeMid != 0) && (here-> BSIM4gNodePrime != 0))
                {
                    i = here->BSIM4GMgpPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4GMgpStructPtr = matched ;
                    here->BSIM4GMgpPtr = matched->CSC ;
                }

                if ((here-> BSIM4gNodeMid != 0) && (here-> BSIM4gNodeMid != 0))
                {
                    i = here->BSIM4GMgmPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4GMgmStructPtr = matched ;
                    here->BSIM4GMgmPtr = matched->CSC ;
                }

                if ((here-> BSIM4gNodeMid != 0) && (here-> BSIM4gNodeExt != 0))
                {
                    i = here->BSIM4GMgePtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4GMgeStructPtr = matched ;
                    here->BSIM4GMgePtr = matched->CSC ;
                }

                if ((here-> BSIM4gNodeMid != 0) && (here-> BSIM4sNodePrime != 0))
                {
                    i = here->BSIM4GMspPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4GMspStructPtr = matched ;
                    here->BSIM4GMspPtr = matched->CSC ;
                }

                if ((here-> BSIM4gNodeMid != 0) && (here-> BSIM4bNodePrime != 0))
                {
                    i = here->BSIM4GMbpPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4GMbpStructPtr = matched ;
                    here->BSIM4GMbpPtr = matched->CSC ;
                }

                if ((here-> BSIM4dNodePrime != 0) && (here-> BSIM4gNodeMid != 0))
                {
                    i = here->BSIM4DPgmPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4DPgmStructPtr = matched ;
                    here->BSIM4DPgmPtr = matched->CSC ;
                }

                if ((here-> BSIM4gNodePrime != 0) && (here-> BSIM4gNodeMid != 0))
                {
                    i = here->BSIM4GPgmPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4GPgmStructPtr = matched ;
                    here->BSIM4GPgmPtr = matched->CSC ;
                }

                if ((here-> BSIM4gNodeExt != 0) && (here-> BSIM4gNodeMid != 0))
                {
                    i = here->BSIM4GEgmPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4GEgmStructPtr = matched ;
                    here->BSIM4GEgmPtr = matched->CSC ;
                }

                if ((here-> BSIM4sNodePrime != 0) && (here-> BSIM4gNodeMid != 0))
                {
                    i = here->BSIM4SPgmPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4SPgmStructPtr = matched ;
                    here->BSIM4SPgmPtr = matched->CSC ;
                }

                if ((here-> BSIM4bNodePrime != 0) && (here-> BSIM4gNodeMid != 0))
                {
                    i = here->BSIM4BPgmPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4BPgmStructPtr = matched ;
                    here->BSIM4BPgmPtr = matched->CSC ;
                }

            }
            if ((here->BSIM4rbodyMod == 1) || (here->BSIM4rbodyMod == 2))
            {
                if ((here-> BSIM4dNodePrime != 0) && (here-> BSIM4dbNode != 0))
                {
                    i = here->BSIM4DPdbPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4DPdbStructPtr = matched ;
                    here->BSIM4DPdbPtr = matched->CSC ;
                }

                if ((here-> BSIM4sNodePrime != 0) && (here-> BSIM4sbNode != 0))
                {
                    i = here->BSIM4SPsbPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4SPsbStructPtr = matched ;
                    here->BSIM4SPsbPtr = matched->CSC ;
                }

                if ((here-> BSIM4dbNode != 0) && (here-> BSIM4dNodePrime != 0))
                {
                    i = here->BSIM4DBdpPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4DBdpStructPtr = matched ;
                    here->BSIM4DBdpPtr = matched->CSC ;
                }

                if ((here-> BSIM4dbNode != 0) && (here-> BSIM4dbNode != 0))
                {
                    i = here->BSIM4DBdbPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4DBdbStructPtr = matched ;
                    here->BSIM4DBdbPtr = matched->CSC ;
                }

                if ((here-> BSIM4dbNode != 0) && (here-> BSIM4bNodePrime != 0))
                {
                    i = here->BSIM4DBbpPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4DBbpStructPtr = matched ;
                    here->BSIM4DBbpPtr = matched->CSC ;
                }

                if ((here-> BSIM4dbNode != 0) && (here-> BSIM4bNode != 0))
                {
                    i = here->BSIM4DBbPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4DBbStructPtr = matched ;
                    here->BSIM4DBbPtr = matched->CSC ;
                }

                if ((here-> BSIM4bNodePrime != 0) && (here-> BSIM4dbNode != 0))
                {
                    i = here->BSIM4BPdbPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4BPdbStructPtr = matched ;
                    here->BSIM4BPdbPtr = matched->CSC ;
                }

                if ((here-> BSIM4bNodePrime != 0) && (here-> BSIM4bNode != 0))
                {
                    i = here->BSIM4BPbPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4BPbStructPtr = matched ;
                    here->BSIM4BPbPtr = matched->CSC ;
                }

                if ((here-> BSIM4bNodePrime != 0) && (here-> BSIM4sbNode != 0))
                {
                    i = here->BSIM4BPsbPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4BPsbStructPtr = matched ;
                    here->BSIM4BPsbPtr = matched->CSC ;
                }

                if ((here-> BSIM4sbNode != 0) && (here-> BSIM4sNodePrime != 0))
                {
                    i = here->BSIM4SBspPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4SBspStructPtr = matched ;
                    here->BSIM4SBspPtr = matched->CSC ;
                }

                if ((here-> BSIM4sbNode != 0) && (here-> BSIM4bNodePrime != 0))
                {
                    i = here->BSIM4SBbpPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4SBbpStructPtr = matched ;
                    here->BSIM4SBbpPtr = matched->CSC ;
                }

                if ((here-> BSIM4sbNode != 0) && (here-> BSIM4bNode != 0))
                {
                    i = here->BSIM4SBbPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4SBbStructPtr = matched ;
                    here->BSIM4SBbPtr = matched->CSC ;
                }

                if ((here-> BSIM4sbNode != 0) && (here-> BSIM4sbNode != 0))
                {
                    i = here->BSIM4SBsbPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4SBsbStructPtr = matched ;
                    here->BSIM4SBsbPtr = matched->CSC ;
                }

                if ((here-> BSIM4bNode != 0) && (here-> BSIM4dbNode != 0))
                {
                    i = here->BSIM4BdbPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4BdbStructPtr = matched ;
                    here->BSIM4BdbPtr = matched->CSC ;
                }

                if ((here-> BSIM4bNode != 0) && (here-> BSIM4bNodePrime != 0))
                {
                    i = here->BSIM4BbpPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4BbpStructPtr = matched ;
                    here->BSIM4BbpPtr = matched->CSC ;
                }

                if ((here-> BSIM4bNode != 0) && (here-> BSIM4sbNode != 0))
                {
                    i = here->BSIM4BsbPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4BsbStructPtr = matched ;
                    here->BSIM4BsbPtr = matched->CSC ;
                }

                if ((here-> BSIM4bNode != 0) && (here-> BSIM4bNode != 0))
                {
                    i = here->BSIM4BbPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4BbStructPtr = matched ;
                    here->BSIM4BbPtr = matched->CSC ;
                }

            }
            if (model->BSIM4rdsMod)
            {
                if ((here-> BSIM4dNode != 0) && (here-> BSIM4gNodePrime != 0))
                {
                    i = here->BSIM4DgpPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4DgpStructPtr = matched ;
                    here->BSIM4DgpPtr = matched->CSC ;
                }

                if ((here-> BSIM4dNode != 0) && (here-> BSIM4sNodePrime != 0))
                {
                    i = here->BSIM4DspPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4DspStructPtr = matched ;
                    here->BSIM4DspPtr = matched->CSC ;
                }

                if ((here-> BSIM4dNode != 0) && (here-> BSIM4bNodePrime != 0))
                {
                    i = here->BSIM4DbpPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4DbpStructPtr = matched ;
                    here->BSIM4DbpPtr = matched->CSC ;
                }

                if ((here-> BSIM4sNode != 0) && (here-> BSIM4dNodePrime != 0))
                {
                    i = here->BSIM4SdpPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4SdpStructPtr = matched ;
                    here->BSIM4SdpPtr = matched->CSC ;
                }

                if ((here-> BSIM4sNode != 0) && (here-> BSIM4gNodePrime != 0))
                {
                    i = here->BSIM4SgpPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4SgpStructPtr = matched ;
                    here->BSIM4SgpPtr = matched->CSC ;
                }

                if ((here-> BSIM4sNode != 0) && (here-> BSIM4bNodePrime != 0))
                {
                    i = here->BSIM4SbpPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->BSIM4SbpStructPtr = matched ;
                    here->BSIM4SbpPtr = matched->CSC ;
                }

            }
        }
    }

    return (OK) ;
}

int
BSIM4bindCSCComplex (GENmodel *inModel, CKTcircuit *ckt)
{
    BSIM4model *model = (BSIM4model *)inModel ;
    BSIM4instance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the BSIM4 models */
    for ( ; model != NULL ; model = model->BSIM4nextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->BSIM4instances ; here != NULL ; here = here->BSIM4nextInstance)
        {
            if ((here-> BSIM4dNodePrime != 0) && (here-> BSIM4bNodePrime != 0))
                here->BSIM4DPbpPtr = here->BSIM4DPbpStructPtr->CSC_Complex ;

            if ((here-> BSIM4gNodePrime != 0) && (here-> BSIM4bNodePrime != 0))
                here->BSIM4GPbpPtr = here->BSIM4GPbpStructPtr->CSC_Complex ;

            if ((here-> BSIM4sNodePrime != 0) && (here-> BSIM4bNodePrime != 0))
                here->BSIM4SPbpPtr = here->BSIM4SPbpStructPtr->CSC_Complex ;

            if ((here-> BSIM4bNodePrime != 0) && (here-> BSIM4dNodePrime != 0))
                here->BSIM4BPdpPtr = here->BSIM4BPdpStructPtr->CSC_Complex ;

            if ((here-> BSIM4bNodePrime != 0) && (here-> BSIM4gNodePrime != 0))
                here->BSIM4BPgpPtr = here->BSIM4BPgpStructPtr->CSC_Complex ;

            if ((here-> BSIM4bNodePrime != 0) && (here-> BSIM4sNodePrime != 0))
                here->BSIM4BPspPtr = here->BSIM4BPspStructPtr->CSC_Complex ;

            if ((here-> BSIM4bNodePrime != 0) && (here-> BSIM4bNodePrime != 0))
                here->BSIM4BPbpPtr = here->BSIM4BPbpStructPtr->CSC_Complex ;

            if ((here-> BSIM4dNode != 0) && (here-> BSIM4dNode != 0))
                here->BSIM4DdPtr = here->BSIM4DdStructPtr->CSC_Complex ;

            if ((here-> BSIM4gNodePrime != 0) && (here-> BSIM4gNodePrime != 0))
                here->BSIM4GPgpPtr = here->BSIM4GPgpStructPtr->CSC_Complex ;

            if ((here-> BSIM4sNode != 0) && (here-> BSIM4sNode != 0))
                here->BSIM4SsPtr = here->BSIM4SsStructPtr->CSC_Complex ;

            if ((here-> BSIM4dNodePrime != 0) && (here-> BSIM4dNodePrime != 0))
                here->BSIM4DPdpPtr = here->BSIM4DPdpStructPtr->CSC_Complex ;

            if ((here-> BSIM4sNodePrime != 0) && (here-> BSIM4sNodePrime != 0))
                here->BSIM4SPspPtr = here->BSIM4SPspStructPtr->CSC_Complex ;

            if ((here-> BSIM4dNode != 0) && (here-> BSIM4dNodePrime != 0))
                here->BSIM4DdpPtr = here->BSIM4DdpStructPtr->CSC_Complex ;

            if ((here-> BSIM4gNodePrime != 0) && (here-> BSIM4dNodePrime != 0))
                here->BSIM4GPdpPtr = here->BSIM4GPdpStructPtr->CSC_Complex ;

            if ((here-> BSIM4gNodePrime != 0) && (here-> BSIM4sNodePrime != 0))
                here->BSIM4GPspPtr = here->BSIM4GPspStructPtr->CSC_Complex ;

            if ((here-> BSIM4sNode != 0) && (here-> BSIM4sNodePrime != 0))
                here->BSIM4SspPtr = here->BSIM4SspStructPtr->CSC_Complex ;

            if ((here-> BSIM4dNodePrime != 0) && (here-> BSIM4sNodePrime != 0))
                here->BSIM4DPspPtr = here->BSIM4DPspStructPtr->CSC_Complex ;

            if ((here-> BSIM4dNodePrime != 0) && (here-> BSIM4dNode != 0))
                here->BSIM4DPdPtr = here->BSIM4DPdStructPtr->CSC_Complex ;

            if ((here-> BSIM4dNodePrime != 0) && (here-> BSIM4gNodePrime != 0))
                here->BSIM4DPgpPtr = here->BSIM4DPgpStructPtr->CSC_Complex ;

            if ((here-> BSIM4sNodePrime != 0) && (here-> BSIM4gNodePrime != 0))
                here->BSIM4SPgpPtr = here->BSIM4SPgpStructPtr->CSC_Complex ;

            if ((here-> BSIM4sNodePrime != 0) && (here-> BSIM4sNode != 0))
                here->BSIM4SPsPtr = here->BSIM4SPsStructPtr->CSC_Complex ;

            if ((here-> BSIM4sNodePrime != 0) && (here-> BSIM4dNodePrime != 0))
                here->BSIM4SPdpPtr = here->BSIM4SPdpStructPtr->CSC_Complex ;

            if ((here-> BSIM4qNode != 0) && (here-> BSIM4qNode != 0))
                here->BSIM4QqPtr = here->BSIM4QqStructPtr->CSC_Complex ;

            if ((here-> BSIM4qNode != 0) && (here-> BSIM4bNodePrime != 0))
                here->BSIM4QbpPtr = here->BSIM4QbpStructPtr->CSC_Complex ;

            if ((here-> BSIM4qNode != 0) && (here-> BSIM4dNodePrime != 0))
                here->BSIM4QdpPtr = here->BSIM4QdpStructPtr->CSC_Complex ;

            if ((here-> BSIM4qNode != 0) && (here-> BSIM4sNodePrime != 0))
                here->BSIM4QspPtr = here->BSIM4QspStructPtr->CSC_Complex ;

            if ((here-> BSIM4qNode != 0) && (here-> BSIM4gNodePrime != 0))
                here->BSIM4QgpPtr = here->BSIM4QgpStructPtr->CSC_Complex ;

            if ((here-> BSIM4dNodePrime != 0) && (here-> BSIM4qNode != 0))
                here->BSIM4DPqPtr = here->BSIM4DPqStructPtr->CSC_Complex ;

            if ((here-> BSIM4sNodePrime != 0) && (here-> BSIM4qNode != 0))
                here->BSIM4SPqPtr = here->BSIM4SPqStructPtr->CSC_Complex ;

            if ((here-> BSIM4gNodePrime != 0) && (here-> BSIM4qNode != 0))
                here->BSIM4GPqPtr = here->BSIM4GPqStructPtr->CSC_Complex ;

            if (here->BSIM4rgateMod != 0)
            {
                if ((here-> BSIM4gNodeExt != 0) && (here-> BSIM4gNodeExt != 0))
                    here->BSIM4GEgePtr = here->BSIM4GEgeStructPtr->CSC_Complex ;

                if ((here-> BSIM4gNodeExt != 0) && (here-> BSIM4gNodePrime != 0))
                    here->BSIM4GEgpPtr = here->BSIM4GEgpStructPtr->CSC_Complex ;

                if ((here-> BSIM4gNodePrime != 0) && (here-> BSIM4gNodeExt != 0))
                    here->BSIM4GPgePtr = here->BSIM4GPgeStructPtr->CSC_Complex ;

                if ((here-> BSIM4gNodeExt != 0) && (here-> BSIM4dNodePrime != 0))
                    here->BSIM4GEdpPtr = here->BSIM4GEdpStructPtr->CSC_Complex ;

                if ((here-> BSIM4gNodeExt != 0) && (here-> BSIM4sNodePrime != 0))
                    here->BSIM4GEspPtr = here->BSIM4GEspStructPtr->CSC_Complex ;

                if ((here-> BSIM4gNodeExt != 0) && (here-> BSIM4bNodePrime != 0))
                    here->BSIM4GEbpPtr = here->BSIM4GEbpStructPtr->CSC_Complex ;

                if ((here-> BSIM4gNodeMid != 0) && (here-> BSIM4dNodePrime != 0))
                    here->BSIM4GMdpPtr = here->BSIM4GMdpStructPtr->CSC_Complex ;

                if ((here-> BSIM4gNodeMid != 0) && (here-> BSIM4gNodePrime != 0))
                    here->BSIM4GMgpPtr = here->BSIM4GMgpStructPtr->CSC_Complex ;

                if ((here-> BSIM4gNodeMid != 0) && (here-> BSIM4gNodeMid != 0))
                    here->BSIM4GMgmPtr = here->BSIM4GMgmStructPtr->CSC_Complex ;

                if ((here-> BSIM4gNodeMid != 0) && (here-> BSIM4gNodeExt != 0))
                    here->BSIM4GMgePtr = here->BSIM4GMgeStructPtr->CSC_Complex ;

                if ((here-> BSIM4gNodeMid != 0) && (here-> BSIM4sNodePrime != 0))
                    here->BSIM4GMspPtr = here->BSIM4GMspStructPtr->CSC_Complex ;

                if ((here-> BSIM4gNodeMid != 0) && (here-> BSIM4bNodePrime != 0))
                    here->BSIM4GMbpPtr = here->BSIM4GMbpStructPtr->CSC_Complex ;

                if ((here-> BSIM4dNodePrime != 0) && (here-> BSIM4gNodeMid != 0))
                    here->BSIM4DPgmPtr = here->BSIM4DPgmStructPtr->CSC_Complex ;

                if ((here-> BSIM4gNodePrime != 0) && (here-> BSIM4gNodeMid != 0))
                    here->BSIM4GPgmPtr = here->BSIM4GPgmStructPtr->CSC_Complex ;

                if ((here-> BSIM4gNodeExt != 0) && (here-> BSIM4gNodeMid != 0))
                    here->BSIM4GEgmPtr = here->BSIM4GEgmStructPtr->CSC_Complex ;

                if ((here-> BSIM4sNodePrime != 0) && (here-> BSIM4gNodeMid != 0))
                    here->BSIM4SPgmPtr = here->BSIM4SPgmStructPtr->CSC_Complex ;

                if ((here-> BSIM4bNodePrime != 0) && (here-> BSIM4gNodeMid != 0))
                    here->BSIM4BPgmPtr = here->BSIM4BPgmStructPtr->CSC_Complex ;

            }
            if ((here->BSIM4rbodyMod == 1) || (here->BSIM4rbodyMod == 2))
            {
                if ((here-> BSIM4dNodePrime != 0) && (here-> BSIM4dbNode != 0))
                    here->BSIM4DPdbPtr = here->BSIM4DPdbStructPtr->CSC_Complex ;

                if ((here-> BSIM4sNodePrime != 0) && (here-> BSIM4sbNode != 0))
                    here->BSIM4SPsbPtr = here->BSIM4SPsbStructPtr->CSC_Complex ;

                if ((here-> BSIM4dbNode != 0) && (here-> BSIM4dNodePrime != 0))
                    here->BSIM4DBdpPtr = here->BSIM4DBdpStructPtr->CSC_Complex ;

                if ((here-> BSIM4dbNode != 0) && (here-> BSIM4dbNode != 0))
                    here->BSIM4DBdbPtr = here->BSIM4DBdbStructPtr->CSC_Complex ;

                if ((here-> BSIM4dbNode != 0) && (here-> BSIM4bNodePrime != 0))
                    here->BSIM4DBbpPtr = here->BSIM4DBbpStructPtr->CSC_Complex ;

                if ((here-> BSIM4dbNode != 0) && (here-> BSIM4bNode != 0))
                    here->BSIM4DBbPtr = here->BSIM4DBbStructPtr->CSC_Complex ;

                if ((here-> BSIM4bNodePrime != 0) && (here-> BSIM4dbNode != 0))
                    here->BSIM4BPdbPtr = here->BSIM4BPdbStructPtr->CSC_Complex ;

                if ((here-> BSIM4bNodePrime != 0) && (here-> BSIM4bNode != 0))
                    here->BSIM4BPbPtr = here->BSIM4BPbStructPtr->CSC_Complex ;

                if ((here-> BSIM4bNodePrime != 0) && (here-> BSIM4sbNode != 0))
                    here->BSIM4BPsbPtr = here->BSIM4BPsbStructPtr->CSC_Complex ;

                if ((here-> BSIM4sbNode != 0) && (here-> BSIM4sNodePrime != 0))
                    here->BSIM4SBspPtr = here->BSIM4SBspStructPtr->CSC_Complex ;

                if ((here-> BSIM4sbNode != 0) && (here-> BSIM4bNodePrime != 0))
                    here->BSIM4SBbpPtr = here->BSIM4SBbpStructPtr->CSC_Complex ;

                if ((here-> BSIM4sbNode != 0) && (here-> BSIM4bNode != 0))
                    here->BSIM4SBbPtr = here->BSIM4SBbStructPtr->CSC_Complex ;

                if ((here-> BSIM4sbNode != 0) && (here-> BSIM4sbNode != 0))
                    here->BSIM4SBsbPtr = here->BSIM4SBsbStructPtr->CSC_Complex ;

                if ((here-> BSIM4bNode != 0) && (here-> BSIM4dbNode != 0))
                    here->BSIM4BdbPtr = here->BSIM4BdbStructPtr->CSC_Complex ;

                if ((here-> BSIM4bNode != 0) && (here-> BSIM4bNodePrime != 0))
                    here->BSIM4BbpPtr = here->BSIM4BbpStructPtr->CSC_Complex ;

                if ((here-> BSIM4bNode != 0) && (here-> BSIM4sbNode != 0))
                    here->BSIM4BsbPtr = here->BSIM4BsbStructPtr->CSC_Complex ;

                if ((here-> BSIM4bNode != 0) && (here-> BSIM4bNode != 0))
                    here->BSIM4BbPtr = here->BSIM4BbStructPtr->CSC_Complex ;

            }
            if (model->BSIM4rdsMod)
            {
                if ((here-> BSIM4dNode != 0) && (here-> BSIM4gNodePrime != 0))
                    here->BSIM4DgpPtr = here->BSIM4DgpStructPtr->CSC_Complex ;

                if ((here-> BSIM4dNode != 0) && (here-> BSIM4sNodePrime != 0))
                    here->BSIM4DspPtr = here->BSIM4DspStructPtr->CSC_Complex ;

                if ((here-> BSIM4dNode != 0) && (here-> BSIM4bNodePrime != 0))
                    here->BSIM4DbpPtr = here->BSIM4DbpStructPtr->CSC_Complex ;

                if ((here-> BSIM4sNode != 0) && (here-> BSIM4dNodePrime != 0))
                    here->BSIM4SdpPtr = here->BSIM4SdpStructPtr->CSC_Complex ;

                if ((here-> BSIM4sNode != 0) && (here-> BSIM4gNodePrime != 0))
                    here->BSIM4SgpPtr = here->BSIM4SgpStructPtr->CSC_Complex ;

                if ((here-> BSIM4sNode != 0) && (here-> BSIM4bNodePrime != 0))
                    here->BSIM4SbpPtr = here->BSIM4SbpStructPtr->CSC_Complex ;

            }
        }
    }

    return (OK) ;
}

int
BSIM4bindCSCComplexToReal (GENmodel *inModel, CKTcircuit *ckt)
{
    BSIM4model *model = (BSIM4model *)inModel ;
    BSIM4instance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the BSIM4 models */
    for ( ; model != NULL ; model = model->BSIM4nextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->BSIM4instances ; here != NULL ; here = here->BSIM4nextInstance)
        {
            if ((here-> BSIM4dNodePrime != 0) && (here-> BSIM4bNodePrime != 0))
                here->BSIM4DPbpPtr = here->BSIM4DPbpStructPtr->CSC ;

            if ((here-> BSIM4gNodePrime != 0) && (here-> BSIM4bNodePrime != 0))
                here->BSIM4GPbpPtr = here->BSIM4GPbpStructPtr->CSC ;

            if ((here-> BSIM4sNodePrime != 0) && (here-> BSIM4bNodePrime != 0))
                here->BSIM4SPbpPtr = here->BSIM4SPbpStructPtr->CSC ;

            if ((here-> BSIM4bNodePrime != 0) && (here-> BSIM4dNodePrime != 0))
                here->BSIM4BPdpPtr = here->BSIM4BPdpStructPtr->CSC ;

            if ((here-> BSIM4bNodePrime != 0) && (here-> BSIM4gNodePrime != 0))
                here->BSIM4BPgpPtr = here->BSIM4BPgpStructPtr->CSC ;

            if ((here-> BSIM4bNodePrime != 0) && (here-> BSIM4sNodePrime != 0))
                here->BSIM4BPspPtr = here->BSIM4BPspStructPtr->CSC ;

            if ((here-> BSIM4bNodePrime != 0) && (here-> BSIM4bNodePrime != 0))
                here->BSIM4BPbpPtr = here->BSIM4BPbpStructPtr->CSC ;

            if ((here-> BSIM4dNode != 0) && (here-> BSIM4dNode != 0))
                here->BSIM4DdPtr = here->BSIM4DdStructPtr->CSC ;

            if ((here-> BSIM4gNodePrime != 0) && (here-> BSIM4gNodePrime != 0))
                here->BSIM4GPgpPtr = here->BSIM4GPgpStructPtr->CSC ;

            if ((here-> BSIM4sNode != 0) && (here-> BSIM4sNode != 0))
                here->BSIM4SsPtr = here->BSIM4SsStructPtr->CSC ;

            if ((here-> BSIM4dNodePrime != 0) && (here-> BSIM4dNodePrime != 0))
                here->BSIM4DPdpPtr = here->BSIM4DPdpStructPtr->CSC ;

            if ((here-> BSIM4sNodePrime != 0) && (here-> BSIM4sNodePrime != 0))
                here->BSIM4SPspPtr = here->BSIM4SPspStructPtr->CSC ;

            if ((here-> BSIM4dNode != 0) && (here-> BSIM4dNodePrime != 0))
                here->BSIM4DdpPtr = here->BSIM4DdpStructPtr->CSC ;

            if ((here-> BSIM4gNodePrime != 0) && (here-> BSIM4dNodePrime != 0))
                here->BSIM4GPdpPtr = here->BSIM4GPdpStructPtr->CSC ;

            if ((here-> BSIM4gNodePrime != 0) && (here-> BSIM4sNodePrime != 0))
                here->BSIM4GPspPtr = here->BSIM4GPspStructPtr->CSC ;

            if ((here-> BSIM4sNode != 0) && (here-> BSIM4sNodePrime != 0))
                here->BSIM4SspPtr = here->BSIM4SspStructPtr->CSC ;

            if ((here-> BSIM4dNodePrime != 0) && (here-> BSIM4sNodePrime != 0))
                here->BSIM4DPspPtr = here->BSIM4DPspStructPtr->CSC ;

            if ((here-> BSIM4dNodePrime != 0) && (here-> BSIM4dNode != 0))
                here->BSIM4DPdPtr = here->BSIM4DPdStructPtr->CSC ;

            if ((here-> BSIM4dNodePrime != 0) && (here-> BSIM4gNodePrime != 0))
                here->BSIM4DPgpPtr = here->BSIM4DPgpStructPtr->CSC ;

            if ((here-> BSIM4sNodePrime != 0) && (here-> BSIM4gNodePrime != 0))
                here->BSIM4SPgpPtr = here->BSIM4SPgpStructPtr->CSC ;

            if ((here-> BSIM4sNodePrime != 0) && (here-> BSIM4sNode != 0))
                here->BSIM4SPsPtr = here->BSIM4SPsStructPtr->CSC ;

            if ((here-> BSIM4sNodePrime != 0) && (here-> BSIM4dNodePrime != 0))
                here->BSIM4SPdpPtr = here->BSIM4SPdpStructPtr->CSC ;

            if ((here-> BSIM4qNode != 0) && (here-> BSIM4qNode != 0))
                here->BSIM4QqPtr = here->BSIM4QqStructPtr->CSC ;

            if ((here-> BSIM4qNode != 0) && (here-> BSIM4bNodePrime != 0))
                here->BSIM4QbpPtr = here->BSIM4QbpStructPtr->CSC ;

            if ((here-> BSIM4qNode != 0) && (here-> BSIM4dNodePrime != 0))
                here->BSIM4QdpPtr = here->BSIM4QdpStructPtr->CSC ;

            if ((here-> BSIM4qNode != 0) && (here-> BSIM4sNodePrime != 0))
                here->BSIM4QspPtr = here->BSIM4QspStructPtr->CSC ;

            if ((here-> BSIM4qNode != 0) && (here-> BSIM4gNodePrime != 0))
                here->BSIM4QgpPtr = here->BSIM4QgpStructPtr->CSC ;

            if ((here-> BSIM4dNodePrime != 0) && (here-> BSIM4qNode != 0))
                here->BSIM4DPqPtr = here->BSIM4DPqStructPtr->CSC ;

            if ((here-> BSIM4sNodePrime != 0) && (here-> BSIM4qNode != 0))
                here->BSIM4SPqPtr = here->BSIM4SPqStructPtr->CSC ;

            if ((here-> BSIM4gNodePrime != 0) && (here-> BSIM4qNode != 0))
                here->BSIM4GPqPtr = here->BSIM4GPqStructPtr->CSC ;

            if (here->BSIM4rgateMod != 0)
            {
                if ((here-> BSIM4gNodeExt != 0) && (here-> BSIM4gNodeExt != 0))
                    here->BSIM4GEgePtr = here->BSIM4GEgeStructPtr->CSC ;

                if ((here-> BSIM4gNodeExt != 0) && (here-> BSIM4gNodePrime != 0))
                    here->BSIM4GEgpPtr = here->BSIM4GEgpStructPtr->CSC ;

                if ((here-> BSIM4gNodePrime != 0) && (here-> BSIM4gNodeExt != 0))
                    here->BSIM4GPgePtr = here->BSIM4GPgeStructPtr->CSC ;

                if ((here-> BSIM4gNodeExt != 0) && (here-> BSIM4dNodePrime != 0))
                    here->BSIM4GEdpPtr = here->BSIM4GEdpStructPtr->CSC ;

                if ((here-> BSIM4gNodeExt != 0) && (here-> BSIM4sNodePrime != 0))
                    here->BSIM4GEspPtr = here->BSIM4GEspStructPtr->CSC ;

                if ((here-> BSIM4gNodeExt != 0) && (here-> BSIM4bNodePrime != 0))
                    here->BSIM4GEbpPtr = here->BSIM4GEbpStructPtr->CSC ;

                if ((here-> BSIM4gNodeMid != 0) && (here-> BSIM4dNodePrime != 0))
                    here->BSIM4GMdpPtr = here->BSIM4GMdpStructPtr->CSC ;

                if ((here-> BSIM4gNodeMid != 0) && (here-> BSIM4gNodePrime != 0))
                    here->BSIM4GMgpPtr = here->BSIM4GMgpStructPtr->CSC ;

                if ((here-> BSIM4gNodeMid != 0) && (here-> BSIM4gNodeMid != 0))
                    here->BSIM4GMgmPtr = here->BSIM4GMgmStructPtr->CSC ;

                if ((here-> BSIM4gNodeMid != 0) && (here-> BSIM4gNodeExt != 0))
                    here->BSIM4GMgePtr = here->BSIM4GMgeStructPtr->CSC ;

                if ((here-> BSIM4gNodeMid != 0) && (here-> BSIM4sNodePrime != 0))
                    here->BSIM4GMspPtr = here->BSIM4GMspStructPtr->CSC ;

                if ((here-> BSIM4gNodeMid != 0) && (here-> BSIM4bNodePrime != 0))
                    here->BSIM4GMbpPtr = here->BSIM4GMbpStructPtr->CSC ;

                if ((here-> BSIM4dNodePrime != 0) && (here-> BSIM4gNodeMid != 0))
                    here->BSIM4DPgmPtr = here->BSIM4DPgmStructPtr->CSC ;

                if ((here-> BSIM4gNodePrime != 0) && (here-> BSIM4gNodeMid != 0))
                    here->BSIM4GPgmPtr = here->BSIM4GPgmStructPtr->CSC ;

                if ((here-> BSIM4gNodeExt != 0) && (here-> BSIM4gNodeMid != 0))
                    here->BSIM4GEgmPtr = here->BSIM4GEgmStructPtr->CSC ;

                if ((here-> BSIM4sNodePrime != 0) && (here-> BSIM4gNodeMid != 0))
                    here->BSIM4SPgmPtr = here->BSIM4SPgmStructPtr->CSC ;

                if ((here-> BSIM4bNodePrime != 0) && (here-> BSIM4gNodeMid != 0))
                    here->BSIM4BPgmPtr = here->BSIM4BPgmStructPtr->CSC ;

            }
            if ((here->BSIM4rbodyMod == 1) || (here->BSIM4rbodyMod == 2))
            {
                if ((here-> BSIM4dNodePrime != 0) && (here-> BSIM4dbNode != 0))
                    here->BSIM4DPdbPtr = here->BSIM4DPdbStructPtr->CSC ;

                if ((here-> BSIM4sNodePrime != 0) && (here-> BSIM4sbNode != 0))
                    here->BSIM4SPsbPtr = here->BSIM4SPsbStructPtr->CSC ;

                if ((here-> BSIM4dbNode != 0) && (here-> BSIM4dNodePrime != 0))
                    here->BSIM4DBdpPtr = here->BSIM4DBdpStructPtr->CSC ;

                if ((here-> BSIM4dbNode != 0) && (here-> BSIM4dbNode != 0))
                    here->BSIM4DBdbPtr = here->BSIM4DBdbStructPtr->CSC ;

                if ((here-> BSIM4dbNode != 0) && (here-> BSIM4bNodePrime != 0))
                    here->BSIM4DBbpPtr = here->BSIM4DBbpStructPtr->CSC ;

                if ((here-> BSIM4dbNode != 0) && (here-> BSIM4bNode != 0))
                    here->BSIM4DBbPtr = here->BSIM4DBbStructPtr->CSC ;

                if ((here-> BSIM4bNodePrime != 0) && (here-> BSIM4dbNode != 0))
                    here->BSIM4BPdbPtr = here->BSIM4BPdbStructPtr->CSC ;

                if ((here-> BSIM4bNodePrime != 0) && (here-> BSIM4bNode != 0))
                    here->BSIM4BPbPtr = here->BSIM4BPbStructPtr->CSC ;

                if ((here-> BSIM4bNodePrime != 0) && (here-> BSIM4sbNode != 0))
                    here->BSIM4BPsbPtr = here->BSIM4BPsbStructPtr->CSC ;

                if ((here-> BSIM4sbNode != 0) && (here-> BSIM4sNodePrime != 0))
                    here->BSIM4SBspPtr = here->BSIM4SBspStructPtr->CSC ;

                if ((here-> BSIM4sbNode != 0) && (here-> BSIM4bNodePrime != 0))
                    here->BSIM4SBbpPtr = here->BSIM4SBbpStructPtr->CSC ;

                if ((here-> BSIM4sbNode != 0) && (here-> BSIM4bNode != 0))
                    here->BSIM4SBbPtr = here->BSIM4SBbStructPtr->CSC ;

                if ((here-> BSIM4sbNode != 0) && (here-> BSIM4sbNode != 0))
                    here->BSIM4SBsbPtr = here->BSIM4SBsbStructPtr->CSC ;

                if ((here-> BSIM4bNode != 0) && (here-> BSIM4dbNode != 0))
                    here->BSIM4BdbPtr = here->BSIM4BdbStructPtr->CSC ;

                if ((here-> BSIM4bNode != 0) && (here-> BSIM4bNodePrime != 0))
                    here->BSIM4BbpPtr = here->BSIM4BbpStructPtr->CSC ;

                if ((here-> BSIM4bNode != 0) && (here-> BSIM4sbNode != 0))
                    here->BSIM4BsbPtr = here->BSIM4BsbStructPtr->CSC ;

                if ((here-> BSIM4bNode != 0) && (here-> BSIM4bNode != 0))
                    here->BSIM4BbPtr = here->BSIM4BbStructPtr->CSC ;

            }
            if (model->BSIM4rdsMod)
            {
                if ((here-> BSIM4dNode != 0) && (here-> BSIM4gNodePrime != 0))
                    here->BSIM4DgpPtr = here->BSIM4DgpStructPtr->CSC ;

                if ((here-> BSIM4dNode != 0) && (here-> BSIM4sNodePrime != 0))
                    here->BSIM4DspPtr = here->BSIM4DspStructPtr->CSC ;

                if ((here-> BSIM4dNode != 0) && (here-> BSIM4bNodePrime != 0))
                    here->BSIM4DbpPtr = here->BSIM4DbpStructPtr->CSC ;

                if ((here-> BSIM4sNode != 0) && (here-> BSIM4dNodePrime != 0))
                    here->BSIM4SdpPtr = here->BSIM4SdpStructPtr->CSC ;

                if ((here-> BSIM4sNode != 0) && (here-> BSIM4gNodePrime != 0))
                    here->BSIM4SgpPtr = here->BSIM4SgpStructPtr->CSC ;

                if ((here-> BSIM4sNode != 0) && (here-> BSIM4bNodePrime != 0))
                    here->BSIM4SbpPtr = here->BSIM4SbpStructPtr->CSC ;

            }
        }
    }

    return (OK) ;
}
