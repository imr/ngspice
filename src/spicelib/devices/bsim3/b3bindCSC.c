/**********
Author: 2013 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "bsim3def.h"
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
BSIM3bindCSC (GENmodel *inModel, CKTcircuit *ckt)
{
    BSIM3model *model = (BSIM3model *)inModel ;
    BSIM3instance *here ;
    double *i ;
    BindElement *matched, *BindStruct ;
    size_t nz ;

    BindStruct = ckt->CKTmatrix->CKTbindStruct ;
    nz = (size_t)ckt->CKTmatrix->CKTklunz ;

    /* loop through all the BSIM3 models */
    for ( ; model != NULL ; model = BSIM3nextModel(model))
    {
        /* loop through all the instances of the model */
        for (here = BSIM3instances(model); here != NULL ; here = BSIM3nextInstance(here))
        {
            CREATE_KLU_BINDING_TABLE(BSIM3DdPtr, BSIM3DdBinding, BSIM3dNode, BSIM3dNode);
            CREATE_KLU_BINDING_TABLE(BSIM3GgPtr, BSIM3GgBinding, BSIM3gNode, BSIM3gNode);
            CREATE_KLU_BINDING_TABLE(BSIM3SsPtr, BSIM3SsBinding, BSIM3sNode, BSIM3sNode);
            CREATE_KLU_BINDING_TABLE(BSIM3BbPtr, BSIM3BbBinding, BSIM3bNode, BSIM3bNode);
            CREATE_KLU_BINDING_TABLE(BSIM3DPdpPtr, BSIM3DPdpBinding, BSIM3dNodePrime, BSIM3dNodePrime);
            CREATE_KLU_BINDING_TABLE(BSIM3SPspPtr, BSIM3SPspBinding, BSIM3sNodePrime, BSIM3sNodePrime);
            CREATE_KLU_BINDING_TABLE(BSIM3DdpPtr, BSIM3DdpBinding, BSIM3dNode, BSIM3dNodePrime);
            CREATE_KLU_BINDING_TABLE(BSIM3GbPtr, BSIM3GbBinding, BSIM3gNode, BSIM3bNode);
            CREATE_KLU_BINDING_TABLE(BSIM3GdpPtr, BSIM3GdpBinding, BSIM3gNode, BSIM3dNodePrime);
            CREATE_KLU_BINDING_TABLE(BSIM3GspPtr, BSIM3GspBinding, BSIM3gNode, BSIM3sNodePrime);
            CREATE_KLU_BINDING_TABLE(BSIM3SspPtr, BSIM3SspBinding, BSIM3sNode, BSIM3sNodePrime);
            CREATE_KLU_BINDING_TABLE(BSIM3BdpPtr, BSIM3BdpBinding, BSIM3bNode, BSIM3dNodePrime);
            CREATE_KLU_BINDING_TABLE(BSIM3BspPtr, BSIM3BspBinding, BSIM3bNode, BSIM3sNodePrime);
            CREATE_KLU_BINDING_TABLE(BSIM3DPspPtr, BSIM3DPspBinding, BSIM3dNodePrime, BSIM3sNodePrime);
            CREATE_KLU_BINDING_TABLE(BSIM3DPdPtr, BSIM3DPdBinding, BSIM3dNodePrime, BSIM3dNode);
            CREATE_KLU_BINDING_TABLE(BSIM3BgPtr, BSIM3BgBinding, BSIM3bNode, BSIM3gNode);
            CREATE_KLU_BINDING_TABLE(BSIM3DPgPtr, BSIM3DPgBinding, BSIM3dNodePrime, BSIM3gNode);
            CREATE_KLU_BINDING_TABLE(BSIM3SPgPtr, BSIM3SPgBinding, BSIM3sNodePrime, BSIM3gNode);
            CREATE_KLU_BINDING_TABLE(BSIM3SPsPtr, BSIM3SPsBinding, BSIM3sNodePrime, BSIM3sNode);
            CREATE_KLU_BINDING_TABLE(BSIM3DPbPtr, BSIM3DPbBinding, BSIM3dNodePrime, BSIM3bNode);
            CREATE_KLU_BINDING_TABLE(BSIM3SPbPtr, BSIM3SPbBinding, BSIM3sNodePrime, BSIM3bNode);
            CREATE_KLU_BINDING_TABLE(BSIM3SPdpPtr, BSIM3SPdpBinding, BSIM3sNodePrime, BSIM3dNodePrime);
            CREATE_KLU_BINDING_TABLE(BSIM3QqPtr, BSIM3QqBinding, BSIM3qNode, BSIM3qNode);
            CREATE_KLU_BINDING_TABLE(BSIM3QdpPtr, BSIM3QdpBinding, BSIM3qNode, BSIM3dNodePrime);
            CREATE_KLU_BINDING_TABLE(BSIM3QgPtr, BSIM3QgBinding, BSIM3qNode, BSIM3gNode);
            CREATE_KLU_BINDING_TABLE(BSIM3QspPtr, BSIM3QspBinding, BSIM3qNode, BSIM3sNodePrime);
            CREATE_KLU_BINDING_TABLE(BSIM3QbPtr, BSIM3QbBinding, BSIM3qNode, BSIM3bNode);
            CREATE_KLU_BINDING_TABLE(BSIM3DPqPtr, BSIM3DPqBinding, BSIM3dNodePrime, BSIM3qNode);
            CREATE_KLU_BINDING_TABLE(BSIM3GqPtr, BSIM3GqBinding, BSIM3gNode, BSIM3qNode);
            CREATE_KLU_BINDING_TABLE(BSIM3SPqPtr, BSIM3SPqBinding, BSIM3sNodePrime, BSIM3qNode);
            CREATE_KLU_BINDING_TABLE(BSIM3BqPtr, BSIM3BqBinding, BSIM3bNode, BSIM3qNode);
        }
    }

    return (OK) ;
}

int
BSIM3bindCSCComplex (GENmodel *inModel, CKTcircuit *ckt)
{
    BSIM3model *model = (BSIM3model *)inModel ;
    BSIM3instance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the BSIM3 models */
    for ( ; model != NULL ; model = BSIM3nextModel(model))
    {
        /* loop through all the instances of the model */
        for (here = BSIM3instances(model); here != NULL ; here = BSIM3nextInstance(here))
        {
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM3DdPtr, BSIM3DdBinding, BSIM3dNode, BSIM3dNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM3GgPtr, BSIM3GgBinding, BSIM3gNode, BSIM3gNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM3SsPtr, BSIM3SsBinding, BSIM3sNode, BSIM3sNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM3BbPtr, BSIM3BbBinding, BSIM3bNode, BSIM3bNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM3DPdpPtr, BSIM3DPdpBinding, BSIM3dNodePrime, BSIM3dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM3SPspPtr, BSIM3SPspBinding, BSIM3sNodePrime, BSIM3sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM3DdpPtr, BSIM3DdpBinding, BSIM3dNode, BSIM3dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM3GbPtr, BSIM3GbBinding, BSIM3gNode, BSIM3bNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM3GdpPtr, BSIM3GdpBinding, BSIM3gNode, BSIM3dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM3GspPtr, BSIM3GspBinding, BSIM3gNode, BSIM3sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM3SspPtr, BSIM3SspBinding, BSIM3sNode, BSIM3sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM3BdpPtr, BSIM3BdpBinding, BSIM3bNode, BSIM3dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM3BspPtr, BSIM3BspBinding, BSIM3bNode, BSIM3sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM3DPspPtr, BSIM3DPspBinding, BSIM3dNodePrime, BSIM3sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM3DPdPtr, BSIM3DPdBinding, BSIM3dNodePrime, BSIM3dNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM3BgPtr, BSIM3BgBinding, BSIM3bNode, BSIM3gNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM3DPgPtr, BSIM3DPgBinding, BSIM3dNodePrime, BSIM3gNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM3SPgPtr, BSIM3SPgBinding, BSIM3sNodePrime, BSIM3gNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM3SPsPtr, BSIM3SPsBinding, BSIM3sNodePrime, BSIM3sNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM3DPbPtr, BSIM3DPbBinding, BSIM3dNodePrime, BSIM3bNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM3SPbPtr, BSIM3SPbBinding, BSIM3sNodePrime, BSIM3bNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM3SPdpPtr, BSIM3SPdpBinding, BSIM3sNodePrime, BSIM3dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM3QqPtr, BSIM3QqBinding, BSIM3qNode, BSIM3qNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM3QdpPtr, BSIM3QdpBinding, BSIM3qNode, BSIM3dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM3QgPtr, BSIM3QgBinding, BSIM3qNode, BSIM3gNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM3QspPtr, BSIM3QspBinding, BSIM3qNode, BSIM3sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM3QbPtr, BSIM3QbBinding, BSIM3qNode, BSIM3bNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM3DPqPtr, BSIM3DPqBinding, BSIM3dNodePrime, BSIM3qNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM3GqPtr, BSIM3GqBinding, BSIM3gNode, BSIM3qNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM3SPqPtr, BSIM3SPqBinding, BSIM3sNodePrime, BSIM3qNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM3BqPtr, BSIM3BqBinding, BSIM3bNode, BSIM3qNode);
        }
    }

    return (OK) ;
}

int
BSIM3bindCSCComplexToReal (GENmodel *inModel, CKTcircuit *ckt)
{
    BSIM3model *model = (BSIM3model *)inModel ;
    BSIM3instance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the BSIM3 models */
    for ( ; model != NULL ; model = BSIM3nextModel(model))
    {
        /* loop through all the instances of the model */
        for (here = BSIM3instances(model); here != NULL ; here = BSIM3nextInstance(here))
        {
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM3DdPtr, BSIM3DdBinding, BSIM3dNode, BSIM3dNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM3GgPtr, BSIM3GgBinding, BSIM3gNode, BSIM3gNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM3SsPtr, BSIM3SsBinding, BSIM3sNode, BSIM3sNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM3BbPtr, BSIM3BbBinding, BSIM3bNode, BSIM3bNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM3DPdpPtr, BSIM3DPdpBinding, BSIM3dNodePrime, BSIM3dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM3SPspPtr, BSIM3SPspBinding, BSIM3sNodePrime, BSIM3sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM3DdpPtr, BSIM3DdpBinding, BSIM3dNode, BSIM3dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM3GbPtr, BSIM3GbBinding, BSIM3gNode, BSIM3bNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM3GdpPtr, BSIM3GdpBinding, BSIM3gNode, BSIM3dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM3GspPtr, BSIM3GspBinding, BSIM3gNode, BSIM3sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM3SspPtr, BSIM3SspBinding, BSIM3sNode, BSIM3sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM3BdpPtr, BSIM3BdpBinding, BSIM3bNode, BSIM3dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM3BspPtr, BSIM3BspBinding, BSIM3bNode, BSIM3sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM3DPspPtr, BSIM3DPspBinding, BSIM3dNodePrime, BSIM3sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM3DPdPtr, BSIM3DPdBinding, BSIM3dNodePrime, BSIM3dNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM3BgPtr, BSIM3BgBinding, BSIM3bNode, BSIM3gNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM3DPgPtr, BSIM3DPgBinding, BSIM3dNodePrime, BSIM3gNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM3SPgPtr, BSIM3SPgBinding, BSIM3sNodePrime, BSIM3gNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM3SPsPtr, BSIM3SPsBinding, BSIM3sNodePrime, BSIM3sNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM3DPbPtr, BSIM3DPbBinding, BSIM3dNodePrime, BSIM3bNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM3SPbPtr, BSIM3SPbBinding, BSIM3sNodePrime, BSIM3bNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM3SPdpPtr, BSIM3SPdpBinding, BSIM3sNodePrime, BSIM3dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM3QqPtr, BSIM3QqBinding, BSIM3qNode, BSIM3qNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM3QdpPtr, BSIM3QdpBinding, BSIM3qNode, BSIM3dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM3QgPtr, BSIM3QgBinding, BSIM3qNode, BSIM3gNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM3QspPtr, BSIM3QspBinding, BSIM3qNode, BSIM3sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM3QbPtr, BSIM3QbBinding, BSIM3qNode, BSIM3bNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM3DPqPtr, BSIM3DPqBinding, BSIM3dNodePrime, BSIM3qNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM3GqPtr, BSIM3GqBinding, BSIM3gNode, BSIM3qNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM3SPqPtr, BSIM3SPqBinding, BSIM3sNodePrime, BSIM3qNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM3BqPtr, BSIM3BqBinding, BSIM3bNode, BSIM3qNode);
        }
    }

    return (OK) ;
}
