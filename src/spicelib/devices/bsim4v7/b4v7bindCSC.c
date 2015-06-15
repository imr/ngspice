/**********
Author: 2013 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "bsim4v7def.h"
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
            CREATE_KLU_BINDING_TABLE(BSIM4v7DPbpPtr, BSIM4v7DPbpBinding, BSIM4v7dNodePrime, BSIM4v7bNodePrime);
            CREATE_KLU_BINDING_TABLE(BSIM4v7GPbpPtr, BSIM4v7GPbpBinding, BSIM4v7gNodePrime, BSIM4v7bNodePrime);
            CREATE_KLU_BINDING_TABLE(BSIM4v7SPbpPtr, BSIM4v7SPbpBinding, BSIM4v7sNodePrime, BSIM4v7bNodePrime);
            CREATE_KLU_BINDING_TABLE(BSIM4v7BPdpPtr, BSIM4v7BPdpBinding, BSIM4v7bNodePrime, BSIM4v7dNodePrime);
            CREATE_KLU_BINDING_TABLE(BSIM4v7BPgpPtr, BSIM4v7BPgpBinding, BSIM4v7bNodePrime, BSIM4v7gNodePrime);
            CREATE_KLU_BINDING_TABLE(BSIM4v7BPspPtr, BSIM4v7BPspBinding, BSIM4v7bNodePrime, BSIM4v7sNodePrime);
            CREATE_KLU_BINDING_TABLE(BSIM4v7BPbpPtr, BSIM4v7BPbpBinding, BSIM4v7bNodePrime, BSIM4v7bNodePrime);
            CREATE_KLU_BINDING_TABLE(BSIM4v7DdPtr, BSIM4v7DdBinding, BSIM4v7dNode, BSIM4v7dNode);
            CREATE_KLU_BINDING_TABLE(BSIM4v7GPgpPtr, BSIM4v7GPgpBinding, BSIM4v7gNodePrime, BSIM4v7gNodePrime);
            CREATE_KLU_BINDING_TABLE(BSIM4v7SsPtr, BSIM4v7SsBinding, BSIM4v7sNode, BSIM4v7sNode);
            CREATE_KLU_BINDING_TABLE(BSIM4v7DPdpPtr, BSIM4v7DPdpBinding, BSIM4v7dNodePrime, BSIM4v7dNodePrime);
            CREATE_KLU_BINDING_TABLE(BSIM4v7SPspPtr, BSIM4v7SPspBinding, BSIM4v7sNodePrime, BSIM4v7sNodePrime);
            CREATE_KLU_BINDING_TABLE(BSIM4v7DdpPtr, BSIM4v7DdpBinding, BSIM4v7dNode, BSIM4v7dNodePrime);
            CREATE_KLU_BINDING_TABLE(BSIM4v7GPdpPtr, BSIM4v7GPdpBinding, BSIM4v7gNodePrime, BSIM4v7dNodePrime);
            CREATE_KLU_BINDING_TABLE(BSIM4v7GPspPtr, BSIM4v7GPspBinding, BSIM4v7gNodePrime, BSIM4v7sNodePrime);
            CREATE_KLU_BINDING_TABLE(BSIM4v7SspPtr, BSIM4v7SspBinding, BSIM4v7sNode, BSIM4v7sNodePrime);
            CREATE_KLU_BINDING_TABLE(BSIM4v7DPspPtr, BSIM4v7DPspBinding, BSIM4v7dNodePrime, BSIM4v7sNodePrime);
            CREATE_KLU_BINDING_TABLE(BSIM4v7DPdPtr, BSIM4v7DPdBinding, BSIM4v7dNodePrime, BSIM4v7dNode);
            CREATE_KLU_BINDING_TABLE(BSIM4v7DPgpPtr, BSIM4v7DPgpBinding, BSIM4v7dNodePrime, BSIM4v7gNodePrime);
            CREATE_KLU_BINDING_TABLE(BSIM4v7SPgpPtr, BSIM4v7SPgpBinding, BSIM4v7sNodePrime, BSIM4v7gNodePrime);
            CREATE_KLU_BINDING_TABLE(BSIM4v7SPsPtr, BSIM4v7SPsBinding, BSIM4v7sNodePrime, BSIM4v7sNode);
            CREATE_KLU_BINDING_TABLE(BSIM4v7SPdpPtr, BSIM4v7SPdpBinding, BSIM4v7sNodePrime, BSIM4v7dNodePrime);
            CREATE_KLU_BINDING_TABLE(BSIM4v7QqPtr, BSIM4v7QqBinding, BSIM4v7qNode, BSIM4v7qNode);
            CREATE_KLU_BINDING_TABLE(BSIM4v7QbpPtr, BSIM4v7QbpBinding, BSIM4v7qNode, BSIM4v7bNodePrime);
            CREATE_KLU_BINDING_TABLE(BSIM4v7QdpPtr, BSIM4v7QdpBinding, BSIM4v7qNode, BSIM4v7dNodePrime);
            CREATE_KLU_BINDING_TABLE(BSIM4v7QspPtr, BSIM4v7QspBinding, BSIM4v7qNode, BSIM4v7sNodePrime);
            CREATE_KLU_BINDING_TABLE(BSIM4v7QgpPtr, BSIM4v7QgpBinding, BSIM4v7qNode, BSIM4v7gNodePrime);
            CREATE_KLU_BINDING_TABLE(BSIM4v7DPqPtr, BSIM4v7DPqBinding, BSIM4v7dNodePrime, BSIM4v7qNode);
            CREATE_KLU_BINDING_TABLE(BSIM4v7SPqPtr, BSIM4v7SPqBinding, BSIM4v7sNodePrime, BSIM4v7qNode);
            CREATE_KLU_BINDING_TABLE(BSIM4v7GPqPtr, BSIM4v7GPqBinding, BSIM4v7gNodePrime, BSIM4v7qNode);
            if (here->BSIM4v7rgateMod != 0)
            {
                CREATE_KLU_BINDING_TABLE(BSIM4v7GEgePtr, BSIM4v7GEgeBinding, BSIM4v7gNodeExt, BSIM4v7gNodeExt);
                CREATE_KLU_BINDING_TABLE(BSIM4v7GEgpPtr, BSIM4v7GEgpBinding, BSIM4v7gNodeExt, BSIM4v7gNodePrime);
                CREATE_KLU_BINDING_TABLE(BSIM4v7GPgePtr, BSIM4v7GPgeBinding, BSIM4v7gNodePrime, BSIM4v7gNodeExt);
                CREATE_KLU_BINDING_TABLE(BSIM4v7GEdpPtr, BSIM4v7GEdpBinding, BSIM4v7gNodeExt, BSIM4v7dNodePrime);
                CREATE_KLU_BINDING_TABLE(BSIM4v7GEspPtr, BSIM4v7GEspBinding, BSIM4v7gNodeExt, BSIM4v7sNodePrime);
                CREATE_KLU_BINDING_TABLE(BSIM4v7GEbpPtr, BSIM4v7GEbpBinding, BSIM4v7gNodeExt, BSIM4v7bNodePrime);
                CREATE_KLU_BINDING_TABLE(BSIM4v7GMdpPtr, BSIM4v7GMdpBinding, BSIM4v7gNodeMid, BSIM4v7dNodePrime);
                CREATE_KLU_BINDING_TABLE(BSIM4v7GMgpPtr, BSIM4v7GMgpBinding, BSIM4v7gNodeMid, BSIM4v7gNodePrime);
                CREATE_KLU_BINDING_TABLE(BSIM4v7GMgmPtr, BSIM4v7GMgmBinding, BSIM4v7gNodeMid, BSIM4v7gNodeMid);
                CREATE_KLU_BINDING_TABLE(BSIM4v7GMgePtr, BSIM4v7GMgeBinding, BSIM4v7gNodeMid, BSIM4v7gNodeExt);
                CREATE_KLU_BINDING_TABLE(BSIM4v7GMspPtr, BSIM4v7GMspBinding, BSIM4v7gNodeMid, BSIM4v7sNodePrime);
                CREATE_KLU_BINDING_TABLE(BSIM4v7GMbpPtr, BSIM4v7GMbpBinding, BSIM4v7gNodeMid, BSIM4v7bNodePrime);
                CREATE_KLU_BINDING_TABLE(BSIM4v7DPgmPtr, BSIM4v7DPgmBinding, BSIM4v7dNodePrime, BSIM4v7gNodeMid);
                CREATE_KLU_BINDING_TABLE(BSIM4v7GPgmPtr, BSIM4v7GPgmBinding, BSIM4v7gNodePrime, BSIM4v7gNodeMid);
                CREATE_KLU_BINDING_TABLE(BSIM4v7GEgmPtr, BSIM4v7GEgmBinding, BSIM4v7gNodeExt, BSIM4v7gNodeMid);
                CREATE_KLU_BINDING_TABLE(BSIM4v7SPgmPtr, BSIM4v7SPgmBinding, BSIM4v7sNodePrime, BSIM4v7gNodeMid);
                CREATE_KLU_BINDING_TABLE(BSIM4v7BPgmPtr, BSIM4v7BPgmBinding, BSIM4v7bNodePrime, BSIM4v7gNodeMid);
            }
            if ((here->BSIM4v7rbodyMod == 1) || (here->BSIM4v7rbodyMod == 2))
            {
                CREATE_KLU_BINDING_TABLE(BSIM4v7DPdbPtr, BSIM4v7DPdbBinding, BSIM4v7dNodePrime, BSIM4v7dbNode);
                CREATE_KLU_BINDING_TABLE(BSIM4v7SPsbPtr, BSIM4v7SPsbBinding, BSIM4v7sNodePrime, BSIM4v7sbNode);
                CREATE_KLU_BINDING_TABLE(BSIM4v7DBdpPtr, BSIM4v7DBdpBinding, BSIM4v7dbNode, BSIM4v7dNodePrime);
                CREATE_KLU_BINDING_TABLE(BSIM4v7DBdbPtr, BSIM4v7DBdbBinding, BSIM4v7dbNode, BSIM4v7dbNode);
                CREATE_KLU_BINDING_TABLE(BSIM4v7DBbpPtr, BSIM4v7DBbpBinding, BSIM4v7dbNode, BSIM4v7bNodePrime);
                CREATE_KLU_BINDING_TABLE(BSIM4v7DBbPtr, BSIM4v7DBbBinding, BSIM4v7dbNode, BSIM4v7bNode);
                CREATE_KLU_BINDING_TABLE(BSIM4v7BPdbPtr, BSIM4v7BPdbBinding, BSIM4v7bNodePrime, BSIM4v7dbNode);
                CREATE_KLU_BINDING_TABLE(BSIM4v7BPbPtr, BSIM4v7BPbBinding, BSIM4v7bNodePrime, BSIM4v7bNode);
                CREATE_KLU_BINDING_TABLE(BSIM4v7BPsbPtr, BSIM4v7BPsbBinding, BSIM4v7bNodePrime, BSIM4v7sbNode);
                CREATE_KLU_BINDING_TABLE(BSIM4v7SBspPtr, BSIM4v7SBspBinding, BSIM4v7sbNode, BSIM4v7sNodePrime);
                CREATE_KLU_BINDING_TABLE(BSIM4v7SBbpPtr, BSIM4v7SBbpBinding, BSIM4v7sbNode, BSIM4v7bNodePrime);
                CREATE_KLU_BINDING_TABLE(BSIM4v7SBbPtr, BSIM4v7SBbBinding, BSIM4v7sbNode, BSIM4v7bNode);
                CREATE_KLU_BINDING_TABLE(BSIM4v7SBsbPtr, BSIM4v7SBsbBinding, BSIM4v7sbNode, BSIM4v7sbNode);
                CREATE_KLU_BINDING_TABLE(BSIM4v7BdbPtr, BSIM4v7BdbBinding, BSIM4v7bNode, BSIM4v7dbNode);
                CREATE_KLU_BINDING_TABLE(BSIM4v7BbpPtr, BSIM4v7BbpBinding, BSIM4v7bNode, BSIM4v7bNodePrime);
                CREATE_KLU_BINDING_TABLE(BSIM4v7BsbPtr, BSIM4v7BsbBinding, BSIM4v7bNode, BSIM4v7sbNode);
                CREATE_KLU_BINDING_TABLE(BSIM4v7BbPtr, BSIM4v7BbBinding, BSIM4v7bNode, BSIM4v7bNode);
            }
            if (model->BSIM4v7rdsMod)
            {
                CREATE_KLU_BINDING_TABLE(BSIM4v7DgpPtr, BSIM4v7DgpBinding, BSIM4v7dNode, BSIM4v7gNodePrime);
                CREATE_KLU_BINDING_TABLE(BSIM4v7DspPtr, BSIM4v7DspBinding, BSIM4v7dNode, BSIM4v7sNodePrime);
                CREATE_KLU_BINDING_TABLE(BSIM4v7DbpPtr, BSIM4v7DbpBinding, BSIM4v7dNode, BSIM4v7bNodePrime);
                CREATE_KLU_BINDING_TABLE(BSIM4v7SdpPtr, BSIM4v7SdpBinding, BSIM4v7sNode, BSIM4v7dNodePrime);
                CREATE_KLU_BINDING_TABLE(BSIM4v7SgpPtr, BSIM4v7SgpBinding, BSIM4v7sNode, BSIM4v7gNodePrime);
                CREATE_KLU_BINDING_TABLE(BSIM4v7SbpPtr, BSIM4v7SbpBinding, BSIM4v7sNode, BSIM4v7bNodePrime);
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
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4v7DPbpPtr, BSIM4v7DPbpBinding, BSIM4v7dNodePrime, BSIM4v7bNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4v7GPbpPtr, BSIM4v7GPbpBinding, BSIM4v7gNodePrime, BSIM4v7bNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4v7SPbpPtr, BSIM4v7SPbpBinding, BSIM4v7sNodePrime, BSIM4v7bNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4v7BPdpPtr, BSIM4v7BPdpBinding, BSIM4v7bNodePrime, BSIM4v7dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4v7BPgpPtr, BSIM4v7BPgpBinding, BSIM4v7bNodePrime, BSIM4v7gNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4v7BPspPtr, BSIM4v7BPspBinding, BSIM4v7bNodePrime, BSIM4v7sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4v7BPbpPtr, BSIM4v7BPbpBinding, BSIM4v7bNodePrime, BSIM4v7bNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4v7DdPtr, BSIM4v7DdBinding, BSIM4v7dNode, BSIM4v7dNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4v7GPgpPtr, BSIM4v7GPgpBinding, BSIM4v7gNodePrime, BSIM4v7gNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4v7SsPtr, BSIM4v7SsBinding, BSIM4v7sNode, BSIM4v7sNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4v7DPdpPtr, BSIM4v7DPdpBinding, BSIM4v7dNodePrime, BSIM4v7dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4v7SPspPtr, BSIM4v7SPspBinding, BSIM4v7sNodePrime, BSIM4v7sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4v7DdpPtr, BSIM4v7DdpBinding, BSIM4v7dNode, BSIM4v7dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4v7GPdpPtr, BSIM4v7GPdpBinding, BSIM4v7gNodePrime, BSIM4v7dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4v7GPspPtr, BSIM4v7GPspBinding, BSIM4v7gNodePrime, BSIM4v7sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4v7SspPtr, BSIM4v7SspBinding, BSIM4v7sNode, BSIM4v7sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4v7DPspPtr, BSIM4v7DPspBinding, BSIM4v7dNodePrime, BSIM4v7sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4v7DPdPtr, BSIM4v7DPdBinding, BSIM4v7dNodePrime, BSIM4v7dNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4v7DPgpPtr, BSIM4v7DPgpBinding, BSIM4v7dNodePrime, BSIM4v7gNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4v7SPgpPtr, BSIM4v7SPgpBinding, BSIM4v7sNodePrime, BSIM4v7gNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4v7SPsPtr, BSIM4v7SPsBinding, BSIM4v7sNodePrime, BSIM4v7sNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4v7SPdpPtr, BSIM4v7SPdpBinding, BSIM4v7sNodePrime, BSIM4v7dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4v7QqPtr, BSIM4v7QqBinding, BSIM4v7qNode, BSIM4v7qNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4v7QbpPtr, BSIM4v7QbpBinding, BSIM4v7qNode, BSIM4v7bNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4v7QdpPtr, BSIM4v7QdpBinding, BSIM4v7qNode, BSIM4v7dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4v7QspPtr, BSIM4v7QspBinding, BSIM4v7qNode, BSIM4v7sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4v7QgpPtr, BSIM4v7QgpBinding, BSIM4v7qNode, BSIM4v7gNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4v7DPqPtr, BSIM4v7DPqBinding, BSIM4v7dNodePrime, BSIM4v7qNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4v7SPqPtr, BSIM4v7SPqBinding, BSIM4v7sNodePrime, BSIM4v7qNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4v7GPqPtr, BSIM4v7GPqBinding, BSIM4v7gNodePrime, BSIM4v7qNode);
            if (here->BSIM4v7rgateMod != 0)
            {
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4v7GEgePtr, BSIM4v7GEgeBinding, BSIM4v7gNodeExt, BSIM4v7gNodeExt);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4v7GEgpPtr, BSIM4v7GEgpBinding, BSIM4v7gNodeExt, BSIM4v7gNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4v7GPgePtr, BSIM4v7GPgeBinding, BSIM4v7gNodePrime, BSIM4v7gNodeExt);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4v7GEdpPtr, BSIM4v7GEdpBinding, BSIM4v7gNodeExt, BSIM4v7dNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4v7GEspPtr, BSIM4v7GEspBinding, BSIM4v7gNodeExt, BSIM4v7sNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4v7GEbpPtr, BSIM4v7GEbpBinding, BSIM4v7gNodeExt, BSIM4v7bNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4v7GMdpPtr, BSIM4v7GMdpBinding, BSIM4v7gNodeMid, BSIM4v7dNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4v7GMgpPtr, BSIM4v7GMgpBinding, BSIM4v7gNodeMid, BSIM4v7gNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4v7GMgmPtr, BSIM4v7GMgmBinding, BSIM4v7gNodeMid, BSIM4v7gNodeMid);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4v7GMgePtr, BSIM4v7GMgeBinding, BSIM4v7gNodeMid, BSIM4v7gNodeExt);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4v7GMspPtr, BSIM4v7GMspBinding, BSIM4v7gNodeMid, BSIM4v7sNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4v7GMbpPtr, BSIM4v7GMbpBinding, BSIM4v7gNodeMid, BSIM4v7bNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4v7DPgmPtr, BSIM4v7DPgmBinding, BSIM4v7dNodePrime, BSIM4v7gNodeMid);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4v7GPgmPtr, BSIM4v7GPgmBinding, BSIM4v7gNodePrime, BSIM4v7gNodeMid);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4v7GEgmPtr, BSIM4v7GEgmBinding, BSIM4v7gNodeExt, BSIM4v7gNodeMid);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4v7SPgmPtr, BSIM4v7SPgmBinding, BSIM4v7sNodePrime, BSIM4v7gNodeMid);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4v7BPgmPtr, BSIM4v7BPgmBinding, BSIM4v7bNodePrime, BSIM4v7gNodeMid);
            }
            if ((here->BSIM4v7rbodyMod == 1) || (here->BSIM4v7rbodyMod == 2))
            {
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4v7DPdbPtr, BSIM4v7DPdbBinding, BSIM4v7dNodePrime, BSIM4v7dbNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4v7SPsbPtr, BSIM4v7SPsbBinding, BSIM4v7sNodePrime, BSIM4v7sbNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4v7DBdpPtr, BSIM4v7DBdpBinding, BSIM4v7dbNode, BSIM4v7dNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4v7DBdbPtr, BSIM4v7DBdbBinding, BSIM4v7dbNode, BSIM4v7dbNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4v7DBbpPtr, BSIM4v7DBbpBinding, BSIM4v7dbNode, BSIM4v7bNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4v7DBbPtr, BSIM4v7DBbBinding, BSIM4v7dbNode, BSIM4v7bNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4v7BPdbPtr, BSIM4v7BPdbBinding, BSIM4v7bNodePrime, BSIM4v7dbNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4v7BPbPtr, BSIM4v7BPbBinding, BSIM4v7bNodePrime, BSIM4v7bNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4v7BPsbPtr, BSIM4v7BPsbBinding, BSIM4v7bNodePrime, BSIM4v7sbNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4v7SBspPtr, BSIM4v7SBspBinding, BSIM4v7sbNode, BSIM4v7sNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4v7SBbpPtr, BSIM4v7SBbpBinding, BSIM4v7sbNode, BSIM4v7bNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4v7SBbPtr, BSIM4v7SBbBinding, BSIM4v7sbNode, BSIM4v7bNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4v7SBsbPtr, BSIM4v7SBsbBinding, BSIM4v7sbNode, BSIM4v7sbNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4v7BdbPtr, BSIM4v7BdbBinding, BSIM4v7bNode, BSIM4v7dbNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4v7BbpPtr, BSIM4v7BbpBinding, BSIM4v7bNode, BSIM4v7bNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4v7BsbPtr, BSIM4v7BsbBinding, BSIM4v7bNode, BSIM4v7sbNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4v7BbPtr, BSIM4v7BbBinding, BSIM4v7bNode, BSIM4v7bNode);
            }
            if (model->BSIM4v7rdsMod)
            {
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4v7DgpPtr, BSIM4v7DgpBinding, BSIM4v7dNode, BSIM4v7gNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4v7DspPtr, BSIM4v7DspBinding, BSIM4v7dNode, BSIM4v7sNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4v7DbpPtr, BSIM4v7DbpBinding, BSIM4v7dNode, BSIM4v7bNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4v7SdpPtr, BSIM4v7SdpBinding, BSIM4v7sNode, BSIM4v7dNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4v7SgpPtr, BSIM4v7SgpBinding, BSIM4v7sNode, BSIM4v7gNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4v7SbpPtr, BSIM4v7SbpBinding, BSIM4v7sNode, BSIM4v7bNodePrime);
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
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4v7DPbpPtr, BSIM4v7DPbpBinding, BSIM4v7dNodePrime, BSIM4v7bNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4v7GPbpPtr, BSIM4v7GPbpBinding, BSIM4v7gNodePrime, BSIM4v7bNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4v7SPbpPtr, BSIM4v7SPbpBinding, BSIM4v7sNodePrime, BSIM4v7bNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4v7BPdpPtr, BSIM4v7BPdpBinding, BSIM4v7bNodePrime, BSIM4v7dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4v7BPgpPtr, BSIM4v7BPgpBinding, BSIM4v7bNodePrime, BSIM4v7gNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4v7BPspPtr, BSIM4v7BPspBinding, BSIM4v7bNodePrime, BSIM4v7sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4v7BPbpPtr, BSIM4v7BPbpBinding, BSIM4v7bNodePrime, BSIM4v7bNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4v7DdPtr, BSIM4v7DdBinding, BSIM4v7dNode, BSIM4v7dNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4v7GPgpPtr, BSIM4v7GPgpBinding, BSIM4v7gNodePrime, BSIM4v7gNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4v7SsPtr, BSIM4v7SsBinding, BSIM4v7sNode, BSIM4v7sNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4v7DPdpPtr, BSIM4v7DPdpBinding, BSIM4v7dNodePrime, BSIM4v7dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4v7SPspPtr, BSIM4v7SPspBinding, BSIM4v7sNodePrime, BSIM4v7sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4v7DdpPtr, BSIM4v7DdpBinding, BSIM4v7dNode, BSIM4v7dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4v7GPdpPtr, BSIM4v7GPdpBinding, BSIM4v7gNodePrime, BSIM4v7dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4v7GPspPtr, BSIM4v7GPspBinding, BSIM4v7gNodePrime, BSIM4v7sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4v7SspPtr, BSIM4v7SspBinding, BSIM4v7sNode, BSIM4v7sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4v7DPspPtr, BSIM4v7DPspBinding, BSIM4v7dNodePrime, BSIM4v7sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4v7DPdPtr, BSIM4v7DPdBinding, BSIM4v7dNodePrime, BSIM4v7dNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4v7DPgpPtr, BSIM4v7DPgpBinding, BSIM4v7dNodePrime, BSIM4v7gNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4v7SPgpPtr, BSIM4v7SPgpBinding, BSIM4v7sNodePrime, BSIM4v7gNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4v7SPsPtr, BSIM4v7SPsBinding, BSIM4v7sNodePrime, BSIM4v7sNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4v7SPdpPtr, BSIM4v7SPdpBinding, BSIM4v7sNodePrime, BSIM4v7dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4v7QqPtr, BSIM4v7QqBinding, BSIM4v7qNode, BSIM4v7qNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4v7QbpPtr, BSIM4v7QbpBinding, BSIM4v7qNode, BSIM4v7bNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4v7QdpPtr, BSIM4v7QdpBinding, BSIM4v7qNode, BSIM4v7dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4v7QspPtr, BSIM4v7QspBinding, BSIM4v7qNode, BSIM4v7sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4v7QgpPtr, BSIM4v7QgpBinding, BSIM4v7qNode, BSIM4v7gNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4v7DPqPtr, BSIM4v7DPqBinding, BSIM4v7dNodePrime, BSIM4v7qNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4v7SPqPtr, BSIM4v7SPqBinding, BSIM4v7sNodePrime, BSIM4v7qNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4v7GPqPtr, BSIM4v7GPqBinding, BSIM4v7gNodePrime, BSIM4v7qNode);
            if (here->BSIM4v7rgateMod != 0)
            {
                CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4v7GEgePtr, BSIM4v7GEgeBinding, BSIM4v7gNodeExt, BSIM4v7gNodeExt);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4v7GEgpPtr, BSIM4v7GEgpBinding, BSIM4v7gNodeExt, BSIM4v7gNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4v7GPgePtr, BSIM4v7GPgeBinding, BSIM4v7gNodePrime, BSIM4v7gNodeExt);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4v7GEdpPtr, BSIM4v7GEdpBinding, BSIM4v7gNodeExt, BSIM4v7dNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4v7GEspPtr, BSIM4v7GEspBinding, BSIM4v7gNodeExt, BSIM4v7sNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4v7GEbpPtr, BSIM4v7GEbpBinding, BSIM4v7gNodeExt, BSIM4v7bNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4v7GMdpPtr, BSIM4v7GMdpBinding, BSIM4v7gNodeMid, BSIM4v7dNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4v7GMgpPtr, BSIM4v7GMgpBinding, BSIM4v7gNodeMid, BSIM4v7gNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4v7GMgmPtr, BSIM4v7GMgmBinding, BSIM4v7gNodeMid, BSIM4v7gNodeMid);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4v7GMgePtr, BSIM4v7GMgeBinding, BSIM4v7gNodeMid, BSIM4v7gNodeExt);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4v7GMspPtr, BSIM4v7GMspBinding, BSIM4v7gNodeMid, BSIM4v7sNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4v7GMbpPtr, BSIM4v7GMbpBinding, BSIM4v7gNodeMid, BSIM4v7bNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4v7DPgmPtr, BSIM4v7DPgmBinding, BSIM4v7dNodePrime, BSIM4v7gNodeMid);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4v7GPgmPtr, BSIM4v7GPgmBinding, BSIM4v7gNodePrime, BSIM4v7gNodeMid);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4v7GEgmPtr, BSIM4v7GEgmBinding, BSIM4v7gNodeExt, BSIM4v7gNodeMid);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4v7SPgmPtr, BSIM4v7SPgmBinding, BSIM4v7sNodePrime, BSIM4v7gNodeMid);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4v7BPgmPtr, BSIM4v7BPgmBinding, BSIM4v7bNodePrime, BSIM4v7gNodeMid);
            }
            if ((here->BSIM4v7rbodyMod == 1) || (here->BSIM4v7rbodyMod == 2))
            {
                CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4v7DPdbPtr, BSIM4v7DPdbBinding, BSIM4v7dNodePrime, BSIM4v7dbNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4v7SPsbPtr, BSIM4v7SPsbBinding, BSIM4v7sNodePrime, BSIM4v7sbNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4v7DBdpPtr, BSIM4v7DBdpBinding, BSIM4v7dbNode, BSIM4v7dNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4v7DBdbPtr, BSIM4v7DBdbBinding, BSIM4v7dbNode, BSIM4v7dbNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4v7DBbpPtr, BSIM4v7DBbpBinding, BSIM4v7dbNode, BSIM4v7bNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4v7DBbPtr, BSIM4v7DBbBinding, BSIM4v7dbNode, BSIM4v7bNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4v7BPdbPtr, BSIM4v7BPdbBinding, BSIM4v7bNodePrime, BSIM4v7dbNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4v7BPbPtr, BSIM4v7BPbBinding, BSIM4v7bNodePrime, BSIM4v7bNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4v7BPsbPtr, BSIM4v7BPsbBinding, BSIM4v7bNodePrime, BSIM4v7sbNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4v7SBspPtr, BSIM4v7SBspBinding, BSIM4v7sbNode, BSIM4v7sNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4v7SBbpPtr, BSIM4v7SBbpBinding, BSIM4v7sbNode, BSIM4v7bNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4v7SBbPtr, BSIM4v7SBbBinding, BSIM4v7sbNode, BSIM4v7bNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4v7SBsbPtr, BSIM4v7SBsbBinding, BSIM4v7sbNode, BSIM4v7sbNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4v7BdbPtr, BSIM4v7BdbBinding, BSIM4v7bNode, BSIM4v7dbNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4v7BbpPtr, BSIM4v7BbpBinding, BSIM4v7bNode, BSIM4v7bNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4v7BsbPtr, BSIM4v7BsbBinding, BSIM4v7bNode, BSIM4v7sbNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4v7BbPtr, BSIM4v7BbBinding, BSIM4v7bNode, BSIM4v7bNode);
            }
            if (model->BSIM4v7rdsMod)
            {
                CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4v7DgpPtr, BSIM4v7DgpBinding, BSIM4v7dNode, BSIM4v7gNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4v7DspPtr, BSIM4v7DspBinding, BSIM4v7dNode, BSIM4v7sNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4v7DbpPtr, BSIM4v7DbpBinding, BSIM4v7dNode, BSIM4v7bNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4v7SdpPtr, BSIM4v7SdpBinding, BSIM4v7sNode, BSIM4v7dNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4v7SgpPtr, BSIM4v7SgpBinding, BSIM4v7sNode, BSIM4v7gNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4v7SbpPtr, BSIM4v7SbpBinding, BSIM4v7sNode, BSIM4v7bNodePrime);
            }
        }
    }

    return (OK) ;
}
