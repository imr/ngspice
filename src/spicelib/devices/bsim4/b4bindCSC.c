/**********
Author: 2013 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "bsim4def.h"
#include "ngspice/sperror.h"
#include "ngspice/klu-binding.h"

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
            CREATE_KLU_BINDING_TABLE(BSIM4DPbpPtr, BSIM4DPbpBinding, BSIM4dNodePrime, BSIM4bNodePrime);
            CREATE_KLU_BINDING_TABLE(BSIM4GPbpPtr, BSIM4GPbpBinding, BSIM4gNodePrime, BSIM4bNodePrime);
            CREATE_KLU_BINDING_TABLE(BSIM4SPbpPtr, BSIM4SPbpBinding, BSIM4sNodePrime, BSIM4bNodePrime);
            CREATE_KLU_BINDING_TABLE(BSIM4BPdpPtr, BSIM4BPdpBinding, BSIM4bNodePrime, BSIM4dNodePrime);
            CREATE_KLU_BINDING_TABLE(BSIM4BPgpPtr, BSIM4BPgpBinding, BSIM4bNodePrime, BSIM4gNodePrime);
            CREATE_KLU_BINDING_TABLE(BSIM4BPspPtr, BSIM4BPspBinding, BSIM4bNodePrime, BSIM4sNodePrime);
            CREATE_KLU_BINDING_TABLE(BSIM4BPbpPtr, BSIM4BPbpBinding, BSIM4bNodePrime, BSIM4bNodePrime);
            CREATE_KLU_BINDING_TABLE(BSIM4DdPtr, BSIM4DdBinding, BSIM4dNode, BSIM4dNode);
            CREATE_KLU_BINDING_TABLE(BSIM4GPgpPtr, BSIM4GPgpBinding, BSIM4gNodePrime, BSIM4gNodePrime);
            CREATE_KLU_BINDING_TABLE(BSIM4SsPtr, BSIM4SsBinding, BSIM4sNode, BSIM4sNode);
            CREATE_KLU_BINDING_TABLE(BSIM4DPdpPtr, BSIM4DPdpBinding, BSIM4dNodePrime, BSIM4dNodePrime);
            CREATE_KLU_BINDING_TABLE(BSIM4SPspPtr, BSIM4SPspBinding, BSIM4sNodePrime, BSIM4sNodePrime);
            CREATE_KLU_BINDING_TABLE(BSIM4DdpPtr, BSIM4DdpBinding, BSIM4dNode, BSIM4dNodePrime);
            CREATE_KLU_BINDING_TABLE(BSIM4GPdpPtr, BSIM4GPdpBinding, BSIM4gNodePrime, BSIM4dNodePrime);
            CREATE_KLU_BINDING_TABLE(BSIM4GPspPtr, BSIM4GPspBinding, BSIM4gNodePrime, BSIM4sNodePrime);
            CREATE_KLU_BINDING_TABLE(BSIM4SspPtr, BSIM4SspBinding, BSIM4sNode, BSIM4sNodePrime);
            CREATE_KLU_BINDING_TABLE(BSIM4DPspPtr, BSIM4DPspBinding, BSIM4dNodePrime, BSIM4sNodePrime);
            CREATE_KLU_BINDING_TABLE(BSIM4DPdPtr, BSIM4DPdBinding, BSIM4dNodePrime, BSIM4dNode);
            CREATE_KLU_BINDING_TABLE(BSIM4DPgpPtr, BSIM4DPgpBinding, BSIM4dNodePrime, BSIM4gNodePrime);
            CREATE_KLU_BINDING_TABLE(BSIM4SPgpPtr, BSIM4SPgpBinding, BSIM4sNodePrime, BSIM4gNodePrime);
            CREATE_KLU_BINDING_TABLE(BSIM4SPsPtr, BSIM4SPsBinding, BSIM4sNodePrime, BSIM4sNode);
            CREATE_KLU_BINDING_TABLE(BSIM4SPdpPtr, BSIM4SPdpBinding, BSIM4sNodePrime, BSIM4dNodePrime);
            CREATE_KLU_BINDING_TABLE(BSIM4QqPtr, BSIM4QqBinding, BSIM4qNode, BSIM4qNode);
            CREATE_KLU_BINDING_TABLE(BSIM4QbpPtr, BSIM4QbpBinding, BSIM4qNode, BSIM4bNodePrime);
            CREATE_KLU_BINDING_TABLE(BSIM4QdpPtr, BSIM4QdpBinding, BSIM4qNode, BSIM4dNodePrime);
            CREATE_KLU_BINDING_TABLE(BSIM4QspPtr, BSIM4QspBinding, BSIM4qNode, BSIM4sNodePrime);
            CREATE_KLU_BINDING_TABLE(BSIM4QgpPtr, BSIM4QgpBinding, BSIM4qNode, BSIM4gNodePrime);
            CREATE_KLU_BINDING_TABLE(BSIM4DPqPtr, BSIM4DPqBinding, BSIM4dNodePrime, BSIM4qNode);
            CREATE_KLU_BINDING_TABLE(BSIM4SPqPtr, BSIM4SPqBinding, BSIM4sNodePrime, BSIM4qNode);
            CREATE_KLU_BINDING_TABLE(BSIM4GPqPtr, BSIM4GPqBinding, BSIM4gNodePrime, BSIM4qNode);
            if (here->BSIM4rgateMod != 0)
            {
                CREATE_KLU_BINDING_TABLE(BSIM4GEgePtr, BSIM4GEgeBinding, BSIM4gNodeExt, BSIM4gNodeExt);
                CREATE_KLU_BINDING_TABLE(BSIM4GEgpPtr, BSIM4GEgpBinding, BSIM4gNodeExt, BSIM4gNodePrime);
                CREATE_KLU_BINDING_TABLE(BSIM4GPgePtr, BSIM4GPgeBinding, BSIM4gNodePrime, BSIM4gNodeExt);
                CREATE_KLU_BINDING_TABLE(BSIM4GEdpPtr, BSIM4GEdpBinding, BSIM4gNodeExt, BSIM4dNodePrime);
                CREATE_KLU_BINDING_TABLE(BSIM4GEspPtr, BSIM4GEspBinding, BSIM4gNodeExt, BSIM4sNodePrime);
                CREATE_KLU_BINDING_TABLE(BSIM4GEbpPtr, BSIM4GEbpBinding, BSIM4gNodeExt, BSIM4bNodePrime);
                CREATE_KLU_BINDING_TABLE(BSIM4GMdpPtr, BSIM4GMdpBinding, BSIM4gNodeMid, BSIM4dNodePrime);
                CREATE_KLU_BINDING_TABLE(BSIM4GMgpPtr, BSIM4GMgpBinding, BSIM4gNodeMid, BSIM4gNodePrime);
                CREATE_KLU_BINDING_TABLE(BSIM4GMgmPtr, BSIM4GMgmBinding, BSIM4gNodeMid, BSIM4gNodeMid);
                CREATE_KLU_BINDING_TABLE(BSIM4GMgePtr, BSIM4GMgeBinding, BSIM4gNodeMid, BSIM4gNodeExt);
                CREATE_KLU_BINDING_TABLE(BSIM4GMspPtr, BSIM4GMspBinding, BSIM4gNodeMid, BSIM4sNodePrime);
                CREATE_KLU_BINDING_TABLE(BSIM4GMbpPtr, BSIM4GMbpBinding, BSIM4gNodeMid, BSIM4bNodePrime);
                CREATE_KLU_BINDING_TABLE(BSIM4DPgmPtr, BSIM4DPgmBinding, BSIM4dNodePrime, BSIM4gNodeMid);
                CREATE_KLU_BINDING_TABLE(BSIM4GPgmPtr, BSIM4GPgmBinding, BSIM4gNodePrime, BSIM4gNodeMid);
                CREATE_KLU_BINDING_TABLE(BSIM4GEgmPtr, BSIM4GEgmBinding, BSIM4gNodeExt, BSIM4gNodeMid);
                CREATE_KLU_BINDING_TABLE(BSIM4SPgmPtr, BSIM4SPgmBinding, BSIM4sNodePrime, BSIM4gNodeMid);
                CREATE_KLU_BINDING_TABLE(BSIM4BPgmPtr, BSIM4BPgmBinding, BSIM4bNodePrime, BSIM4gNodeMid);
            }
            if ((here->BSIM4rbodyMod == 1) || (here->BSIM4rbodyMod == 2))
            {
                CREATE_KLU_BINDING_TABLE(BSIM4DPdbPtr, BSIM4DPdbBinding, BSIM4dNodePrime, BSIM4dbNode);
                CREATE_KLU_BINDING_TABLE(BSIM4SPsbPtr, BSIM4SPsbBinding, BSIM4sNodePrime, BSIM4sbNode);
                CREATE_KLU_BINDING_TABLE(BSIM4DBdpPtr, BSIM4DBdpBinding, BSIM4dbNode, BSIM4dNodePrime);
                CREATE_KLU_BINDING_TABLE(BSIM4DBdbPtr, BSIM4DBdbBinding, BSIM4dbNode, BSIM4dbNode);
                CREATE_KLU_BINDING_TABLE(BSIM4DBbpPtr, BSIM4DBbpBinding, BSIM4dbNode, BSIM4bNodePrime);
                CREATE_KLU_BINDING_TABLE(BSIM4DBbPtr, BSIM4DBbBinding, BSIM4dbNode, BSIM4bNode);
                CREATE_KLU_BINDING_TABLE(BSIM4BPdbPtr, BSIM4BPdbBinding, BSIM4bNodePrime, BSIM4dbNode);
                CREATE_KLU_BINDING_TABLE(BSIM4BPbPtr, BSIM4BPbBinding, BSIM4bNodePrime, BSIM4bNode);
                CREATE_KLU_BINDING_TABLE(BSIM4BPsbPtr, BSIM4BPsbBinding, BSIM4bNodePrime, BSIM4sbNode);
                CREATE_KLU_BINDING_TABLE(BSIM4SBspPtr, BSIM4SBspBinding, BSIM4sbNode, BSIM4sNodePrime);
                CREATE_KLU_BINDING_TABLE(BSIM4SBbpPtr, BSIM4SBbpBinding, BSIM4sbNode, BSIM4bNodePrime);
                CREATE_KLU_BINDING_TABLE(BSIM4SBbPtr, BSIM4SBbBinding, BSIM4sbNode, BSIM4bNode);
                CREATE_KLU_BINDING_TABLE(BSIM4SBsbPtr, BSIM4SBsbBinding, BSIM4sbNode, BSIM4sbNode);
                CREATE_KLU_BINDING_TABLE(BSIM4BdbPtr, BSIM4BdbBinding, BSIM4bNode, BSIM4dbNode);
                CREATE_KLU_BINDING_TABLE(BSIM4BbpPtr, BSIM4BbpBinding, BSIM4bNode, BSIM4bNodePrime);
                CREATE_KLU_BINDING_TABLE(BSIM4BsbPtr, BSIM4BsbBinding, BSIM4bNode, BSIM4sbNode);
                CREATE_KLU_BINDING_TABLE(BSIM4BbPtr, BSIM4BbBinding, BSIM4bNode, BSIM4bNode);
            }
            if (model->BSIM4rdsMod)
            {
                CREATE_KLU_BINDING_TABLE(BSIM4DgpPtr, BSIM4DgpBinding, BSIM4dNode, BSIM4gNodePrime);
                CREATE_KLU_BINDING_TABLE(BSIM4DspPtr, BSIM4DspBinding, BSIM4dNode, BSIM4sNodePrime);
                CREATE_KLU_BINDING_TABLE(BSIM4DbpPtr, BSIM4DbpBinding, BSIM4dNode, BSIM4bNodePrime);
                CREATE_KLU_BINDING_TABLE(BSIM4SdpPtr, BSIM4SdpBinding, BSIM4sNode, BSIM4dNodePrime);
                CREATE_KLU_BINDING_TABLE(BSIM4SgpPtr, BSIM4SgpBinding, BSIM4sNode, BSIM4gNodePrime);
                CREATE_KLU_BINDING_TABLE(BSIM4SbpPtr, BSIM4SbpBinding, BSIM4sNode, BSIM4bNodePrime);
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
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4DPbpPtr, BSIM4DPbpBinding, BSIM4dNodePrime, BSIM4bNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4GPbpPtr, BSIM4GPbpBinding, BSIM4gNodePrime, BSIM4bNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4SPbpPtr, BSIM4SPbpBinding, BSIM4sNodePrime, BSIM4bNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4BPdpPtr, BSIM4BPdpBinding, BSIM4bNodePrime, BSIM4dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4BPgpPtr, BSIM4BPgpBinding, BSIM4bNodePrime, BSIM4gNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4BPspPtr, BSIM4BPspBinding, BSIM4bNodePrime, BSIM4sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4BPbpPtr, BSIM4BPbpBinding, BSIM4bNodePrime, BSIM4bNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4DdPtr, BSIM4DdBinding, BSIM4dNode, BSIM4dNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4GPgpPtr, BSIM4GPgpBinding, BSIM4gNodePrime, BSIM4gNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4SsPtr, BSIM4SsBinding, BSIM4sNode, BSIM4sNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4DPdpPtr, BSIM4DPdpBinding, BSIM4dNodePrime, BSIM4dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4SPspPtr, BSIM4SPspBinding, BSIM4sNodePrime, BSIM4sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4DdpPtr, BSIM4DdpBinding, BSIM4dNode, BSIM4dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4GPdpPtr, BSIM4GPdpBinding, BSIM4gNodePrime, BSIM4dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4GPspPtr, BSIM4GPspBinding, BSIM4gNodePrime, BSIM4sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4SspPtr, BSIM4SspBinding, BSIM4sNode, BSIM4sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4DPspPtr, BSIM4DPspBinding, BSIM4dNodePrime, BSIM4sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4DPdPtr, BSIM4DPdBinding, BSIM4dNodePrime, BSIM4dNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4DPgpPtr, BSIM4DPgpBinding, BSIM4dNodePrime, BSIM4gNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4SPgpPtr, BSIM4SPgpBinding, BSIM4sNodePrime, BSIM4gNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4SPsPtr, BSIM4SPsBinding, BSIM4sNodePrime, BSIM4sNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4SPdpPtr, BSIM4SPdpBinding, BSIM4sNodePrime, BSIM4dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4QqPtr, BSIM4QqBinding, BSIM4qNode, BSIM4qNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4QbpPtr, BSIM4QbpBinding, BSIM4qNode, BSIM4bNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4QdpPtr, BSIM4QdpBinding, BSIM4qNode, BSIM4dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4QspPtr, BSIM4QspBinding, BSIM4qNode, BSIM4sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4QgpPtr, BSIM4QgpBinding, BSIM4qNode, BSIM4gNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4DPqPtr, BSIM4DPqBinding, BSIM4dNodePrime, BSIM4qNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4SPqPtr, BSIM4SPqBinding, BSIM4sNodePrime, BSIM4qNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4GPqPtr, BSIM4GPqBinding, BSIM4gNodePrime, BSIM4qNode);
            if (here->BSIM4rgateMod != 0)
            {
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4GEgePtr, BSIM4GEgeBinding, BSIM4gNodeExt, BSIM4gNodeExt);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4GEgpPtr, BSIM4GEgpBinding, BSIM4gNodeExt, BSIM4gNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4GPgePtr, BSIM4GPgeBinding, BSIM4gNodePrime, BSIM4gNodeExt);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4GEdpPtr, BSIM4GEdpBinding, BSIM4gNodeExt, BSIM4dNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4GEspPtr, BSIM4GEspBinding, BSIM4gNodeExt, BSIM4sNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4GEbpPtr, BSIM4GEbpBinding, BSIM4gNodeExt, BSIM4bNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4GMdpPtr, BSIM4GMdpBinding, BSIM4gNodeMid, BSIM4dNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4GMgpPtr, BSIM4GMgpBinding, BSIM4gNodeMid, BSIM4gNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4GMgmPtr, BSIM4GMgmBinding, BSIM4gNodeMid, BSIM4gNodeMid);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4GMgePtr, BSIM4GMgeBinding, BSIM4gNodeMid, BSIM4gNodeExt);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4GMspPtr, BSIM4GMspBinding, BSIM4gNodeMid, BSIM4sNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4GMbpPtr, BSIM4GMbpBinding, BSIM4gNodeMid, BSIM4bNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4DPgmPtr, BSIM4DPgmBinding, BSIM4dNodePrime, BSIM4gNodeMid);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4GPgmPtr, BSIM4GPgmBinding, BSIM4gNodePrime, BSIM4gNodeMid);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4GEgmPtr, BSIM4GEgmBinding, BSIM4gNodeExt, BSIM4gNodeMid);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4SPgmPtr, BSIM4SPgmBinding, BSIM4sNodePrime, BSIM4gNodeMid);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4BPgmPtr, BSIM4BPgmBinding, BSIM4bNodePrime, BSIM4gNodeMid);
            }
            if ((here->BSIM4rbodyMod == 1) || (here->BSIM4rbodyMod == 2))
            {
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4DPdbPtr, BSIM4DPdbBinding, BSIM4dNodePrime, BSIM4dbNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4SPsbPtr, BSIM4SPsbBinding, BSIM4sNodePrime, BSIM4sbNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4DBdpPtr, BSIM4DBdpBinding, BSIM4dbNode, BSIM4dNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4DBdbPtr, BSIM4DBdbBinding, BSIM4dbNode, BSIM4dbNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4DBbpPtr, BSIM4DBbpBinding, BSIM4dbNode, BSIM4bNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4DBbPtr, BSIM4DBbBinding, BSIM4dbNode, BSIM4bNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4BPdbPtr, BSIM4BPdbBinding, BSIM4bNodePrime, BSIM4dbNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4BPbPtr, BSIM4BPbBinding, BSIM4bNodePrime, BSIM4bNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4BPsbPtr, BSIM4BPsbBinding, BSIM4bNodePrime, BSIM4sbNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4SBspPtr, BSIM4SBspBinding, BSIM4sbNode, BSIM4sNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4SBbpPtr, BSIM4SBbpBinding, BSIM4sbNode, BSIM4bNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4SBbPtr, BSIM4SBbBinding, BSIM4sbNode, BSIM4bNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4SBsbPtr, BSIM4SBsbBinding, BSIM4sbNode, BSIM4sbNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4BdbPtr, BSIM4BdbBinding, BSIM4bNode, BSIM4dbNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4BbpPtr, BSIM4BbpBinding, BSIM4bNode, BSIM4bNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4BsbPtr, BSIM4BsbBinding, BSIM4bNode, BSIM4sbNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4BbPtr, BSIM4BbBinding, BSIM4bNode, BSIM4bNode);
            }
            if (model->BSIM4rdsMod)
            {
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4DgpPtr, BSIM4DgpBinding, BSIM4dNode, BSIM4gNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4DspPtr, BSIM4DspBinding, BSIM4dNode, BSIM4sNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4DbpPtr, BSIM4DbpBinding, BSIM4dNode, BSIM4bNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4SdpPtr, BSIM4SdpBinding, BSIM4sNode, BSIM4dNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4SgpPtr, BSIM4SgpBinding, BSIM4sNode, BSIM4gNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4SbpPtr, BSIM4SbpBinding, BSIM4sNode, BSIM4bNodePrime);
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
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4DPbpPtr, BSIM4DPbpBinding, BSIM4dNodePrime, BSIM4bNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4GPbpPtr, BSIM4GPbpBinding, BSIM4gNodePrime, BSIM4bNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4SPbpPtr, BSIM4SPbpBinding, BSIM4sNodePrime, BSIM4bNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4BPdpPtr, BSIM4BPdpBinding, BSIM4bNodePrime, BSIM4dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4BPgpPtr, BSIM4BPgpBinding, BSIM4bNodePrime, BSIM4gNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4BPspPtr, BSIM4BPspBinding, BSIM4bNodePrime, BSIM4sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4BPbpPtr, BSIM4BPbpBinding, BSIM4bNodePrime, BSIM4bNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4DdPtr, BSIM4DdBinding, BSIM4dNode, BSIM4dNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4GPgpPtr, BSIM4GPgpBinding, BSIM4gNodePrime, BSIM4gNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4SsPtr, BSIM4SsBinding, BSIM4sNode, BSIM4sNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4DPdpPtr, BSIM4DPdpBinding, BSIM4dNodePrime, BSIM4dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4SPspPtr, BSIM4SPspBinding, BSIM4sNodePrime, BSIM4sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4DdpPtr, BSIM4DdpBinding, BSIM4dNode, BSIM4dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4GPdpPtr, BSIM4GPdpBinding, BSIM4gNodePrime, BSIM4dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4GPspPtr, BSIM4GPspBinding, BSIM4gNodePrime, BSIM4sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4SspPtr, BSIM4SspBinding, BSIM4sNode, BSIM4sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4DPspPtr, BSIM4DPspBinding, BSIM4dNodePrime, BSIM4sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4DPdPtr, BSIM4DPdBinding, BSIM4dNodePrime, BSIM4dNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4DPgpPtr, BSIM4DPgpBinding, BSIM4dNodePrime, BSIM4gNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4SPgpPtr, BSIM4SPgpBinding, BSIM4sNodePrime, BSIM4gNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4SPsPtr, BSIM4SPsBinding, BSIM4sNodePrime, BSIM4sNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4SPdpPtr, BSIM4SPdpBinding, BSIM4sNodePrime, BSIM4dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4QqPtr, BSIM4QqBinding, BSIM4qNode, BSIM4qNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4QbpPtr, BSIM4QbpBinding, BSIM4qNode, BSIM4bNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4QdpPtr, BSIM4QdpBinding, BSIM4qNode, BSIM4dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4QspPtr, BSIM4QspBinding, BSIM4qNode, BSIM4sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4QgpPtr, BSIM4QgpBinding, BSIM4qNode, BSIM4gNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4DPqPtr, BSIM4DPqBinding, BSIM4dNodePrime, BSIM4qNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4SPqPtr, BSIM4SPqBinding, BSIM4sNodePrime, BSIM4qNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4GPqPtr, BSIM4GPqBinding, BSIM4gNodePrime, BSIM4qNode);
            if (here->BSIM4rgateMod != 0)
            {
                CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4GEgePtr, BSIM4GEgeBinding, BSIM4gNodeExt, BSIM4gNodeExt);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4GEgpPtr, BSIM4GEgpBinding, BSIM4gNodeExt, BSIM4gNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4GPgePtr, BSIM4GPgeBinding, BSIM4gNodePrime, BSIM4gNodeExt);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4GEdpPtr, BSIM4GEdpBinding, BSIM4gNodeExt, BSIM4dNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4GEspPtr, BSIM4GEspBinding, BSIM4gNodeExt, BSIM4sNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4GEbpPtr, BSIM4GEbpBinding, BSIM4gNodeExt, BSIM4bNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4GMdpPtr, BSIM4GMdpBinding, BSIM4gNodeMid, BSIM4dNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4GMgpPtr, BSIM4GMgpBinding, BSIM4gNodeMid, BSIM4gNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4GMgmPtr, BSIM4GMgmBinding, BSIM4gNodeMid, BSIM4gNodeMid);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4GMgePtr, BSIM4GMgeBinding, BSIM4gNodeMid, BSIM4gNodeExt);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4GMspPtr, BSIM4GMspBinding, BSIM4gNodeMid, BSIM4sNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4GMbpPtr, BSIM4GMbpBinding, BSIM4gNodeMid, BSIM4bNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4DPgmPtr, BSIM4DPgmBinding, BSIM4dNodePrime, BSIM4gNodeMid);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4GPgmPtr, BSIM4GPgmBinding, BSIM4gNodePrime, BSIM4gNodeMid);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4GEgmPtr, BSIM4GEgmBinding, BSIM4gNodeExt, BSIM4gNodeMid);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4SPgmPtr, BSIM4SPgmBinding, BSIM4sNodePrime, BSIM4gNodeMid);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4BPgmPtr, BSIM4BPgmBinding, BSIM4bNodePrime, BSIM4gNodeMid);
            }
            if ((here->BSIM4rbodyMod == 1) || (here->BSIM4rbodyMod == 2))
            {
                CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4DPdbPtr, BSIM4DPdbBinding, BSIM4dNodePrime, BSIM4dbNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4SPsbPtr, BSIM4SPsbBinding, BSIM4sNodePrime, BSIM4sbNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4DBdpPtr, BSIM4DBdpBinding, BSIM4dbNode, BSIM4dNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4DBdbPtr, BSIM4DBdbBinding, BSIM4dbNode, BSIM4dbNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4DBbpPtr, BSIM4DBbpBinding, BSIM4dbNode, BSIM4bNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4DBbPtr, BSIM4DBbBinding, BSIM4dbNode, BSIM4bNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4BPdbPtr, BSIM4BPdbBinding, BSIM4bNodePrime, BSIM4dbNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4BPbPtr, BSIM4BPbBinding, BSIM4bNodePrime, BSIM4bNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4BPsbPtr, BSIM4BPsbBinding, BSIM4bNodePrime, BSIM4sbNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4SBspPtr, BSIM4SBspBinding, BSIM4sbNode, BSIM4sNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4SBbpPtr, BSIM4SBbpBinding, BSIM4sbNode, BSIM4bNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4SBbPtr, BSIM4SBbBinding, BSIM4sbNode, BSIM4bNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4SBsbPtr, BSIM4SBsbBinding, BSIM4sbNode, BSIM4sbNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4BdbPtr, BSIM4BdbBinding, BSIM4bNode, BSIM4dbNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4BbpPtr, BSIM4BbpBinding, BSIM4bNode, BSIM4bNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4BsbPtr, BSIM4BsbBinding, BSIM4bNode, BSIM4sbNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4BbPtr, BSIM4BbBinding, BSIM4bNode, BSIM4bNode);
            }
            if (model->BSIM4rdsMod)
            {
                CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4DgpPtr, BSIM4DgpBinding, BSIM4dNode, BSIM4gNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4DspPtr, BSIM4DspBinding, BSIM4dNode, BSIM4sNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4DbpPtr, BSIM4DbpBinding, BSIM4dNode, BSIM4bNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4SdpPtr, BSIM4SdpBinding, BSIM4sNode, BSIM4dNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4SgpPtr, BSIM4SgpBinding, BSIM4sNode, BSIM4gNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4SbpPtr, BSIM4SbpBinding, BSIM4sNode, BSIM4bNodePrime);
            }
        }
    }

    return (OK) ;
}
