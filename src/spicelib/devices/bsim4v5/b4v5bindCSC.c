/**********
Author: 2013 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "bsim4v5def.h"
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
            CREATE_KLU_BINDING_TABLE(BSIM4v5DPbpPtr, BSIM4v5DPbpBinding, BSIM4v5dNodePrime, BSIM4v5bNodePrime);
            CREATE_KLU_BINDING_TABLE(BSIM4v5GPbpPtr, BSIM4v5GPbpBinding, BSIM4v5gNodePrime, BSIM4v5bNodePrime);
            CREATE_KLU_BINDING_TABLE(BSIM4v5SPbpPtr, BSIM4v5SPbpBinding, BSIM4v5sNodePrime, BSIM4v5bNodePrime);
            CREATE_KLU_BINDING_TABLE(BSIM4v5BPdpPtr, BSIM4v5BPdpBinding, BSIM4v5bNodePrime, BSIM4v5dNodePrime);
            CREATE_KLU_BINDING_TABLE(BSIM4v5BPgpPtr, BSIM4v5BPgpBinding, BSIM4v5bNodePrime, BSIM4v5gNodePrime);
            CREATE_KLU_BINDING_TABLE(BSIM4v5BPspPtr, BSIM4v5BPspBinding, BSIM4v5bNodePrime, BSIM4v5sNodePrime);
            CREATE_KLU_BINDING_TABLE(BSIM4v5BPbpPtr, BSIM4v5BPbpBinding, BSIM4v5bNodePrime, BSIM4v5bNodePrime);
            CREATE_KLU_BINDING_TABLE(BSIM4v5DdPtr, BSIM4v5DdBinding, BSIM4v5dNode, BSIM4v5dNode);
            CREATE_KLU_BINDING_TABLE(BSIM4v5GPgpPtr, BSIM4v5GPgpBinding, BSIM4v5gNodePrime, BSIM4v5gNodePrime);
            CREATE_KLU_BINDING_TABLE(BSIM4v5SsPtr, BSIM4v5SsBinding, BSIM4v5sNode, BSIM4v5sNode);
            CREATE_KLU_BINDING_TABLE(BSIM4v5DPdpPtr, BSIM4v5DPdpBinding, BSIM4v5dNodePrime, BSIM4v5dNodePrime);
            CREATE_KLU_BINDING_TABLE(BSIM4v5SPspPtr, BSIM4v5SPspBinding, BSIM4v5sNodePrime, BSIM4v5sNodePrime);
            CREATE_KLU_BINDING_TABLE(BSIM4v5DdpPtr, BSIM4v5DdpBinding, BSIM4v5dNode, BSIM4v5dNodePrime);
            CREATE_KLU_BINDING_TABLE(BSIM4v5GPdpPtr, BSIM4v5GPdpBinding, BSIM4v5gNodePrime, BSIM4v5dNodePrime);
            CREATE_KLU_BINDING_TABLE(BSIM4v5GPspPtr, BSIM4v5GPspBinding, BSIM4v5gNodePrime, BSIM4v5sNodePrime);
            CREATE_KLU_BINDING_TABLE(BSIM4v5SspPtr, BSIM4v5SspBinding, BSIM4v5sNode, BSIM4v5sNodePrime);
            CREATE_KLU_BINDING_TABLE(BSIM4v5DPspPtr, BSIM4v5DPspBinding, BSIM4v5dNodePrime, BSIM4v5sNodePrime);
            CREATE_KLU_BINDING_TABLE(BSIM4v5DPdPtr, BSIM4v5DPdBinding, BSIM4v5dNodePrime, BSIM4v5dNode);
            CREATE_KLU_BINDING_TABLE(BSIM4v5DPgpPtr, BSIM4v5DPgpBinding, BSIM4v5dNodePrime, BSIM4v5gNodePrime);
            CREATE_KLU_BINDING_TABLE(BSIM4v5SPgpPtr, BSIM4v5SPgpBinding, BSIM4v5sNodePrime, BSIM4v5gNodePrime);
            CREATE_KLU_BINDING_TABLE(BSIM4v5SPsPtr, BSIM4v5SPsBinding, BSIM4v5sNodePrime, BSIM4v5sNode);
            CREATE_KLU_BINDING_TABLE(BSIM4v5SPdpPtr, BSIM4v5SPdpBinding, BSIM4v5sNodePrime, BSIM4v5dNodePrime);
            CREATE_KLU_BINDING_TABLE(BSIM4v5QqPtr, BSIM4v5QqBinding, BSIM4v5qNode, BSIM4v5qNode);
            CREATE_KLU_BINDING_TABLE(BSIM4v5QbpPtr, BSIM4v5QbpBinding, BSIM4v5qNode, BSIM4v5bNodePrime);
            CREATE_KLU_BINDING_TABLE(BSIM4v5QdpPtr, BSIM4v5QdpBinding, BSIM4v5qNode, BSIM4v5dNodePrime);
            CREATE_KLU_BINDING_TABLE(BSIM4v5QspPtr, BSIM4v5QspBinding, BSIM4v5qNode, BSIM4v5sNodePrime);
            CREATE_KLU_BINDING_TABLE(BSIM4v5QgpPtr, BSIM4v5QgpBinding, BSIM4v5qNode, BSIM4v5gNodePrime);
            CREATE_KLU_BINDING_TABLE(BSIM4v5DPqPtr, BSIM4v5DPqBinding, BSIM4v5dNodePrime, BSIM4v5qNode);
            CREATE_KLU_BINDING_TABLE(BSIM4v5SPqPtr, BSIM4v5SPqBinding, BSIM4v5sNodePrime, BSIM4v5qNode);
            CREATE_KLU_BINDING_TABLE(BSIM4v5GPqPtr, BSIM4v5GPqBinding, BSIM4v5gNodePrime, BSIM4v5qNode);
            if (here->BSIM4v5rgateMod != 0)
            {
                CREATE_KLU_BINDING_TABLE(BSIM4v5GEgePtr, BSIM4v5GEgeBinding, BSIM4v5gNodeExt, BSIM4v5gNodeExt);
                CREATE_KLU_BINDING_TABLE(BSIM4v5GEgpPtr, BSIM4v5GEgpBinding, BSIM4v5gNodeExt, BSIM4v5gNodePrime);
                CREATE_KLU_BINDING_TABLE(BSIM4v5GPgePtr, BSIM4v5GPgeBinding, BSIM4v5gNodePrime, BSIM4v5gNodeExt);
                CREATE_KLU_BINDING_TABLE(BSIM4v5GEdpPtr, BSIM4v5GEdpBinding, BSIM4v5gNodeExt, BSIM4v5dNodePrime);
                CREATE_KLU_BINDING_TABLE(BSIM4v5GEspPtr, BSIM4v5GEspBinding, BSIM4v5gNodeExt, BSIM4v5sNodePrime);
                CREATE_KLU_BINDING_TABLE(BSIM4v5GEbpPtr, BSIM4v5GEbpBinding, BSIM4v5gNodeExt, BSIM4v5bNodePrime);
                CREATE_KLU_BINDING_TABLE(BSIM4v5GMdpPtr, BSIM4v5GMdpBinding, BSIM4v5gNodeMid, BSIM4v5dNodePrime);
                CREATE_KLU_BINDING_TABLE(BSIM4v5GMgpPtr, BSIM4v5GMgpBinding, BSIM4v5gNodeMid, BSIM4v5gNodePrime);
                CREATE_KLU_BINDING_TABLE(BSIM4v5GMgmPtr, BSIM4v5GMgmBinding, BSIM4v5gNodeMid, BSIM4v5gNodeMid);
                CREATE_KLU_BINDING_TABLE(BSIM4v5GMgePtr, BSIM4v5GMgeBinding, BSIM4v5gNodeMid, BSIM4v5gNodeExt);
                CREATE_KLU_BINDING_TABLE(BSIM4v5GMspPtr, BSIM4v5GMspBinding, BSIM4v5gNodeMid, BSIM4v5sNodePrime);
                CREATE_KLU_BINDING_TABLE(BSIM4v5GMbpPtr, BSIM4v5GMbpBinding, BSIM4v5gNodeMid, BSIM4v5bNodePrime);
                CREATE_KLU_BINDING_TABLE(BSIM4v5DPgmPtr, BSIM4v5DPgmBinding, BSIM4v5dNodePrime, BSIM4v5gNodeMid);
                CREATE_KLU_BINDING_TABLE(BSIM4v5GPgmPtr, BSIM4v5GPgmBinding, BSIM4v5gNodePrime, BSIM4v5gNodeMid);
                CREATE_KLU_BINDING_TABLE(BSIM4v5GEgmPtr, BSIM4v5GEgmBinding, BSIM4v5gNodeExt, BSIM4v5gNodeMid);
                CREATE_KLU_BINDING_TABLE(BSIM4v5SPgmPtr, BSIM4v5SPgmBinding, BSIM4v5sNodePrime, BSIM4v5gNodeMid);
                CREATE_KLU_BINDING_TABLE(BSIM4v5BPgmPtr, BSIM4v5BPgmBinding, BSIM4v5bNodePrime, BSIM4v5gNodeMid);
            }
            if ((here->BSIM4v5rbodyMod == 1) || (here->BSIM4v5rbodyMod == 2))
            {
                CREATE_KLU_BINDING_TABLE(BSIM4v5DPdbPtr, BSIM4v5DPdbBinding, BSIM4v5dNodePrime, BSIM4v5dbNode);
                CREATE_KLU_BINDING_TABLE(BSIM4v5SPsbPtr, BSIM4v5SPsbBinding, BSIM4v5sNodePrime, BSIM4v5sbNode);
                CREATE_KLU_BINDING_TABLE(BSIM4v5DBdpPtr, BSIM4v5DBdpBinding, BSIM4v5dbNode, BSIM4v5dNodePrime);
                CREATE_KLU_BINDING_TABLE(BSIM4v5DBdbPtr, BSIM4v5DBdbBinding, BSIM4v5dbNode, BSIM4v5dbNode);
                CREATE_KLU_BINDING_TABLE(BSIM4v5DBbpPtr, BSIM4v5DBbpBinding, BSIM4v5dbNode, BSIM4v5bNodePrime);
                CREATE_KLU_BINDING_TABLE(BSIM4v5DBbPtr, BSIM4v5DBbBinding, BSIM4v5dbNode, BSIM4v5bNode);
                CREATE_KLU_BINDING_TABLE(BSIM4v5BPdbPtr, BSIM4v5BPdbBinding, BSIM4v5bNodePrime, BSIM4v5dbNode);
                CREATE_KLU_BINDING_TABLE(BSIM4v5BPbPtr, BSIM4v5BPbBinding, BSIM4v5bNodePrime, BSIM4v5bNode);
                CREATE_KLU_BINDING_TABLE(BSIM4v5BPsbPtr, BSIM4v5BPsbBinding, BSIM4v5bNodePrime, BSIM4v5sbNode);
                CREATE_KLU_BINDING_TABLE(BSIM4v5SBspPtr, BSIM4v5SBspBinding, BSIM4v5sbNode, BSIM4v5sNodePrime);
                CREATE_KLU_BINDING_TABLE(BSIM4v5SBbpPtr, BSIM4v5SBbpBinding, BSIM4v5sbNode, BSIM4v5bNodePrime);
                CREATE_KLU_BINDING_TABLE(BSIM4v5SBbPtr, BSIM4v5SBbBinding, BSIM4v5sbNode, BSIM4v5bNode);
                CREATE_KLU_BINDING_TABLE(BSIM4v5SBsbPtr, BSIM4v5SBsbBinding, BSIM4v5sbNode, BSIM4v5sbNode);
                CREATE_KLU_BINDING_TABLE(BSIM4v5BdbPtr, BSIM4v5BdbBinding, BSIM4v5bNode, BSIM4v5dbNode);
                CREATE_KLU_BINDING_TABLE(BSIM4v5BbpPtr, BSIM4v5BbpBinding, BSIM4v5bNode, BSIM4v5bNodePrime);
                CREATE_KLU_BINDING_TABLE(BSIM4v5BsbPtr, BSIM4v5BsbBinding, BSIM4v5bNode, BSIM4v5sbNode);
                CREATE_KLU_BINDING_TABLE(BSIM4v5BbPtr, BSIM4v5BbBinding, BSIM4v5bNode, BSIM4v5bNode);
            }
            if (model->BSIM4v5rdsMod)
            {
                CREATE_KLU_BINDING_TABLE(BSIM4v5DgpPtr, BSIM4v5DgpBinding, BSIM4v5dNode, BSIM4v5gNodePrime);
                CREATE_KLU_BINDING_TABLE(BSIM4v5DspPtr, BSIM4v5DspBinding, BSIM4v5dNode, BSIM4v5sNodePrime);
                CREATE_KLU_BINDING_TABLE(BSIM4v5DbpPtr, BSIM4v5DbpBinding, BSIM4v5dNode, BSIM4v5bNodePrime);
                CREATE_KLU_BINDING_TABLE(BSIM4v5SdpPtr, BSIM4v5SdpBinding, BSIM4v5sNode, BSIM4v5dNodePrime);
                CREATE_KLU_BINDING_TABLE(BSIM4v5SgpPtr, BSIM4v5SgpBinding, BSIM4v5sNode, BSIM4v5gNodePrime);
                CREATE_KLU_BINDING_TABLE(BSIM4v5SbpPtr, BSIM4v5SbpBinding, BSIM4v5sNode, BSIM4v5bNodePrime);
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
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4v5DPbpPtr, BSIM4v5DPbpBinding, BSIM4v5dNodePrime, BSIM4v5bNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4v5GPbpPtr, BSIM4v5GPbpBinding, BSIM4v5gNodePrime, BSIM4v5bNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4v5SPbpPtr, BSIM4v5SPbpBinding, BSIM4v5sNodePrime, BSIM4v5bNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4v5BPdpPtr, BSIM4v5BPdpBinding, BSIM4v5bNodePrime, BSIM4v5dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4v5BPgpPtr, BSIM4v5BPgpBinding, BSIM4v5bNodePrime, BSIM4v5gNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4v5BPspPtr, BSIM4v5BPspBinding, BSIM4v5bNodePrime, BSIM4v5sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4v5BPbpPtr, BSIM4v5BPbpBinding, BSIM4v5bNodePrime, BSIM4v5bNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4v5DdPtr, BSIM4v5DdBinding, BSIM4v5dNode, BSIM4v5dNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4v5GPgpPtr, BSIM4v5GPgpBinding, BSIM4v5gNodePrime, BSIM4v5gNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4v5SsPtr, BSIM4v5SsBinding, BSIM4v5sNode, BSIM4v5sNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4v5DPdpPtr, BSIM4v5DPdpBinding, BSIM4v5dNodePrime, BSIM4v5dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4v5SPspPtr, BSIM4v5SPspBinding, BSIM4v5sNodePrime, BSIM4v5sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4v5DdpPtr, BSIM4v5DdpBinding, BSIM4v5dNode, BSIM4v5dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4v5GPdpPtr, BSIM4v5GPdpBinding, BSIM4v5gNodePrime, BSIM4v5dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4v5GPspPtr, BSIM4v5GPspBinding, BSIM4v5gNodePrime, BSIM4v5sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4v5SspPtr, BSIM4v5SspBinding, BSIM4v5sNode, BSIM4v5sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4v5DPspPtr, BSIM4v5DPspBinding, BSIM4v5dNodePrime, BSIM4v5sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4v5DPdPtr, BSIM4v5DPdBinding, BSIM4v5dNodePrime, BSIM4v5dNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4v5DPgpPtr, BSIM4v5DPgpBinding, BSIM4v5dNodePrime, BSIM4v5gNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4v5SPgpPtr, BSIM4v5SPgpBinding, BSIM4v5sNodePrime, BSIM4v5gNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4v5SPsPtr, BSIM4v5SPsBinding, BSIM4v5sNodePrime, BSIM4v5sNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4v5SPdpPtr, BSIM4v5SPdpBinding, BSIM4v5sNodePrime, BSIM4v5dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4v5QqPtr, BSIM4v5QqBinding, BSIM4v5qNode, BSIM4v5qNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4v5QbpPtr, BSIM4v5QbpBinding, BSIM4v5qNode, BSIM4v5bNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4v5QdpPtr, BSIM4v5QdpBinding, BSIM4v5qNode, BSIM4v5dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4v5QspPtr, BSIM4v5QspBinding, BSIM4v5qNode, BSIM4v5sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4v5QgpPtr, BSIM4v5QgpBinding, BSIM4v5qNode, BSIM4v5gNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4v5DPqPtr, BSIM4v5DPqBinding, BSIM4v5dNodePrime, BSIM4v5qNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4v5SPqPtr, BSIM4v5SPqBinding, BSIM4v5sNodePrime, BSIM4v5qNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4v5GPqPtr, BSIM4v5GPqBinding, BSIM4v5gNodePrime, BSIM4v5qNode);
            if (here->BSIM4v5rgateMod != 0)
            {
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4v5GEgePtr, BSIM4v5GEgeBinding, BSIM4v5gNodeExt, BSIM4v5gNodeExt);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4v5GEgpPtr, BSIM4v5GEgpBinding, BSIM4v5gNodeExt, BSIM4v5gNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4v5GPgePtr, BSIM4v5GPgeBinding, BSIM4v5gNodePrime, BSIM4v5gNodeExt);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4v5GEdpPtr, BSIM4v5GEdpBinding, BSIM4v5gNodeExt, BSIM4v5dNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4v5GEspPtr, BSIM4v5GEspBinding, BSIM4v5gNodeExt, BSIM4v5sNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4v5GEbpPtr, BSIM4v5GEbpBinding, BSIM4v5gNodeExt, BSIM4v5bNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4v5GMdpPtr, BSIM4v5GMdpBinding, BSIM4v5gNodeMid, BSIM4v5dNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4v5GMgpPtr, BSIM4v5GMgpBinding, BSIM4v5gNodeMid, BSIM4v5gNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4v5GMgmPtr, BSIM4v5GMgmBinding, BSIM4v5gNodeMid, BSIM4v5gNodeMid);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4v5GMgePtr, BSIM4v5GMgeBinding, BSIM4v5gNodeMid, BSIM4v5gNodeExt);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4v5GMspPtr, BSIM4v5GMspBinding, BSIM4v5gNodeMid, BSIM4v5sNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4v5GMbpPtr, BSIM4v5GMbpBinding, BSIM4v5gNodeMid, BSIM4v5bNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4v5DPgmPtr, BSIM4v5DPgmBinding, BSIM4v5dNodePrime, BSIM4v5gNodeMid);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4v5GPgmPtr, BSIM4v5GPgmBinding, BSIM4v5gNodePrime, BSIM4v5gNodeMid);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4v5GEgmPtr, BSIM4v5GEgmBinding, BSIM4v5gNodeExt, BSIM4v5gNodeMid);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4v5SPgmPtr, BSIM4v5SPgmBinding, BSIM4v5sNodePrime, BSIM4v5gNodeMid);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4v5BPgmPtr, BSIM4v5BPgmBinding, BSIM4v5bNodePrime, BSIM4v5gNodeMid);
            }
            if ((here->BSIM4v5rbodyMod == 1) || (here->BSIM4v5rbodyMod == 2))
            {
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4v5DPdbPtr, BSIM4v5DPdbBinding, BSIM4v5dNodePrime, BSIM4v5dbNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4v5SPsbPtr, BSIM4v5SPsbBinding, BSIM4v5sNodePrime, BSIM4v5sbNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4v5DBdpPtr, BSIM4v5DBdpBinding, BSIM4v5dbNode, BSIM4v5dNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4v5DBdbPtr, BSIM4v5DBdbBinding, BSIM4v5dbNode, BSIM4v5dbNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4v5DBbpPtr, BSIM4v5DBbpBinding, BSIM4v5dbNode, BSIM4v5bNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4v5DBbPtr, BSIM4v5DBbBinding, BSIM4v5dbNode, BSIM4v5bNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4v5BPdbPtr, BSIM4v5BPdbBinding, BSIM4v5bNodePrime, BSIM4v5dbNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4v5BPbPtr, BSIM4v5BPbBinding, BSIM4v5bNodePrime, BSIM4v5bNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4v5BPsbPtr, BSIM4v5BPsbBinding, BSIM4v5bNodePrime, BSIM4v5sbNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4v5SBspPtr, BSIM4v5SBspBinding, BSIM4v5sbNode, BSIM4v5sNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4v5SBbpPtr, BSIM4v5SBbpBinding, BSIM4v5sbNode, BSIM4v5bNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4v5SBbPtr, BSIM4v5SBbBinding, BSIM4v5sbNode, BSIM4v5bNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4v5SBsbPtr, BSIM4v5SBsbBinding, BSIM4v5sbNode, BSIM4v5sbNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4v5BdbPtr, BSIM4v5BdbBinding, BSIM4v5bNode, BSIM4v5dbNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4v5BbpPtr, BSIM4v5BbpBinding, BSIM4v5bNode, BSIM4v5bNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4v5BsbPtr, BSIM4v5BsbBinding, BSIM4v5bNode, BSIM4v5sbNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4v5BbPtr, BSIM4v5BbBinding, BSIM4v5bNode, BSIM4v5bNode);
            }
            if (model->BSIM4v5rdsMod)
            {
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4v5DgpPtr, BSIM4v5DgpBinding, BSIM4v5dNode, BSIM4v5gNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4v5DspPtr, BSIM4v5DspBinding, BSIM4v5dNode, BSIM4v5sNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4v5DbpPtr, BSIM4v5DbpBinding, BSIM4v5dNode, BSIM4v5bNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4v5SdpPtr, BSIM4v5SdpBinding, BSIM4v5sNode, BSIM4v5dNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4v5SgpPtr, BSIM4v5SgpBinding, BSIM4v5sNode, BSIM4v5gNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM4v5SbpPtr, BSIM4v5SbpBinding, BSIM4v5sNode, BSIM4v5bNodePrime);
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
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4v5DPbpPtr, BSIM4v5DPbpBinding, BSIM4v5dNodePrime, BSIM4v5bNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4v5GPbpPtr, BSIM4v5GPbpBinding, BSIM4v5gNodePrime, BSIM4v5bNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4v5SPbpPtr, BSIM4v5SPbpBinding, BSIM4v5sNodePrime, BSIM4v5bNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4v5BPdpPtr, BSIM4v5BPdpBinding, BSIM4v5bNodePrime, BSIM4v5dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4v5BPgpPtr, BSIM4v5BPgpBinding, BSIM4v5bNodePrime, BSIM4v5gNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4v5BPspPtr, BSIM4v5BPspBinding, BSIM4v5bNodePrime, BSIM4v5sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4v5BPbpPtr, BSIM4v5BPbpBinding, BSIM4v5bNodePrime, BSIM4v5bNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4v5DdPtr, BSIM4v5DdBinding, BSIM4v5dNode, BSIM4v5dNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4v5GPgpPtr, BSIM4v5GPgpBinding, BSIM4v5gNodePrime, BSIM4v5gNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4v5SsPtr, BSIM4v5SsBinding, BSIM4v5sNode, BSIM4v5sNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4v5DPdpPtr, BSIM4v5DPdpBinding, BSIM4v5dNodePrime, BSIM4v5dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4v5SPspPtr, BSIM4v5SPspBinding, BSIM4v5sNodePrime, BSIM4v5sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4v5DdpPtr, BSIM4v5DdpBinding, BSIM4v5dNode, BSIM4v5dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4v5GPdpPtr, BSIM4v5GPdpBinding, BSIM4v5gNodePrime, BSIM4v5dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4v5GPspPtr, BSIM4v5GPspBinding, BSIM4v5gNodePrime, BSIM4v5sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4v5SspPtr, BSIM4v5SspBinding, BSIM4v5sNode, BSIM4v5sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4v5DPspPtr, BSIM4v5DPspBinding, BSIM4v5dNodePrime, BSIM4v5sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4v5DPdPtr, BSIM4v5DPdBinding, BSIM4v5dNodePrime, BSIM4v5dNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4v5DPgpPtr, BSIM4v5DPgpBinding, BSIM4v5dNodePrime, BSIM4v5gNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4v5SPgpPtr, BSIM4v5SPgpBinding, BSIM4v5sNodePrime, BSIM4v5gNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4v5SPsPtr, BSIM4v5SPsBinding, BSIM4v5sNodePrime, BSIM4v5sNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4v5SPdpPtr, BSIM4v5SPdpBinding, BSIM4v5sNodePrime, BSIM4v5dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4v5QqPtr, BSIM4v5QqBinding, BSIM4v5qNode, BSIM4v5qNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4v5QbpPtr, BSIM4v5QbpBinding, BSIM4v5qNode, BSIM4v5bNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4v5QdpPtr, BSIM4v5QdpBinding, BSIM4v5qNode, BSIM4v5dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4v5QspPtr, BSIM4v5QspBinding, BSIM4v5qNode, BSIM4v5sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4v5QgpPtr, BSIM4v5QgpBinding, BSIM4v5qNode, BSIM4v5gNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4v5DPqPtr, BSIM4v5DPqBinding, BSIM4v5dNodePrime, BSIM4v5qNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4v5SPqPtr, BSIM4v5SPqBinding, BSIM4v5sNodePrime, BSIM4v5qNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4v5GPqPtr, BSIM4v5GPqBinding, BSIM4v5gNodePrime, BSIM4v5qNode);
            if (here->BSIM4v5rgateMod != 0)
            {
                CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4v5GEgePtr, BSIM4v5GEgeBinding, BSIM4v5gNodeExt, BSIM4v5gNodeExt);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4v5GEgpPtr, BSIM4v5GEgpBinding, BSIM4v5gNodeExt, BSIM4v5gNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4v5GPgePtr, BSIM4v5GPgeBinding, BSIM4v5gNodePrime, BSIM4v5gNodeExt);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4v5GEdpPtr, BSIM4v5GEdpBinding, BSIM4v5gNodeExt, BSIM4v5dNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4v5GEspPtr, BSIM4v5GEspBinding, BSIM4v5gNodeExt, BSIM4v5sNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4v5GEbpPtr, BSIM4v5GEbpBinding, BSIM4v5gNodeExt, BSIM4v5bNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4v5GMdpPtr, BSIM4v5GMdpBinding, BSIM4v5gNodeMid, BSIM4v5dNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4v5GMgpPtr, BSIM4v5GMgpBinding, BSIM4v5gNodeMid, BSIM4v5gNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4v5GMgmPtr, BSIM4v5GMgmBinding, BSIM4v5gNodeMid, BSIM4v5gNodeMid);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4v5GMgePtr, BSIM4v5GMgeBinding, BSIM4v5gNodeMid, BSIM4v5gNodeExt);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4v5GMspPtr, BSIM4v5GMspBinding, BSIM4v5gNodeMid, BSIM4v5sNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4v5GMbpPtr, BSIM4v5GMbpBinding, BSIM4v5gNodeMid, BSIM4v5bNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4v5DPgmPtr, BSIM4v5DPgmBinding, BSIM4v5dNodePrime, BSIM4v5gNodeMid);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4v5GPgmPtr, BSIM4v5GPgmBinding, BSIM4v5gNodePrime, BSIM4v5gNodeMid);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4v5GEgmPtr, BSIM4v5GEgmBinding, BSIM4v5gNodeExt, BSIM4v5gNodeMid);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4v5SPgmPtr, BSIM4v5SPgmBinding, BSIM4v5sNodePrime, BSIM4v5gNodeMid);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4v5BPgmPtr, BSIM4v5BPgmBinding, BSIM4v5bNodePrime, BSIM4v5gNodeMid);
            }
            if ((here->BSIM4v5rbodyMod == 1) || (here->BSIM4v5rbodyMod == 2))
            {
                CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4v5DPdbPtr, BSIM4v5DPdbBinding, BSIM4v5dNodePrime, BSIM4v5dbNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4v5SPsbPtr, BSIM4v5SPsbBinding, BSIM4v5sNodePrime, BSIM4v5sbNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4v5DBdpPtr, BSIM4v5DBdpBinding, BSIM4v5dbNode, BSIM4v5dNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4v5DBdbPtr, BSIM4v5DBdbBinding, BSIM4v5dbNode, BSIM4v5dbNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4v5DBbpPtr, BSIM4v5DBbpBinding, BSIM4v5dbNode, BSIM4v5bNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4v5DBbPtr, BSIM4v5DBbBinding, BSIM4v5dbNode, BSIM4v5bNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4v5BPdbPtr, BSIM4v5BPdbBinding, BSIM4v5bNodePrime, BSIM4v5dbNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4v5BPbPtr, BSIM4v5BPbBinding, BSIM4v5bNodePrime, BSIM4v5bNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4v5BPsbPtr, BSIM4v5BPsbBinding, BSIM4v5bNodePrime, BSIM4v5sbNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4v5SBspPtr, BSIM4v5SBspBinding, BSIM4v5sbNode, BSIM4v5sNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4v5SBbpPtr, BSIM4v5SBbpBinding, BSIM4v5sbNode, BSIM4v5bNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4v5SBbPtr, BSIM4v5SBbBinding, BSIM4v5sbNode, BSIM4v5bNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4v5SBsbPtr, BSIM4v5SBsbBinding, BSIM4v5sbNode, BSIM4v5sbNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4v5BdbPtr, BSIM4v5BdbBinding, BSIM4v5bNode, BSIM4v5dbNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4v5BbpPtr, BSIM4v5BbpBinding, BSIM4v5bNode, BSIM4v5bNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4v5BsbPtr, BSIM4v5BsbBinding, BSIM4v5bNode, BSIM4v5sbNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4v5BbPtr, BSIM4v5BbBinding, BSIM4v5bNode, BSIM4v5bNode);
            }
            if (model->BSIM4v5rdsMod)
            {
                CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4v5DgpPtr, BSIM4v5DgpBinding, BSIM4v5dNode, BSIM4v5gNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4v5DspPtr, BSIM4v5DspBinding, BSIM4v5dNode, BSIM4v5sNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4v5DbpPtr, BSIM4v5DbpBinding, BSIM4v5dNode, BSIM4v5bNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4v5SdpPtr, BSIM4v5SdpBinding, BSIM4v5sNode, BSIM4v5dNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4v5SgpPtr, BSIM4v5SgpBinding, BSIM4v5sNode, BSIM4v5gNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM4v5SbpPtr, BSIM4v5SbpBinding, BSIM4v5sNode, BSIM4v5bNodePrime);
            }
        }
    }

    return (OK) ;
}
