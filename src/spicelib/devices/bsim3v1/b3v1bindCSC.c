/**********
Author: 2013 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "bsim3v1def.h"
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
BSIM3v1bindCSC (GENmodel *inModel, CKTcircuit *ckt)
{
    BSIM3v1model *model = (BSIM3v1model *)inModel ;
    BSIM3v1instance *here ;
    double *i ;
    BindElement *matched, *BindStruct ;
    size_t nz ;

    BindStruct = ckt->CKTmatrix->CKTbindStruct ;
    nz = (size_t)ckt->CKTmatrix->CKTklunz ;

    /* loop through all the BSIM3v1 models */
    for ( ; model != NULL ; model = model->BSIM3v1nextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->BSIM3v1instances ; here != NULL ; here = here->BSIM3v1nextInstance)
        {
            CREATE_KLU_BINDING_TABLE(BSIM3v1DdPtr, BSIM3v1DdBinding, BSIM3v1dNode, BSIM3v1dNode);
            CREATE_KLU_BINDING_TABLE(BSIM3v1GgPtr, BSIM3v1GgBinding, BSIM3v1gNode, BSIM3v1gNode);
            CREATE_KLU_BINDING_TABLE(BSIM3v1SsPtr, BSIM3v1SsBinding, BSIM3v1sNode, BSIM3v1sNode);
            CREATE_KLU_BINDING_TABLE(BSIM3v1BbPtr, BSIM3v1BbBinding, BSIM3v1bNode, BSIM3v1bNode);
            CREATE_KLU_BINDING_TABLE(BSIM3v1DPdpPtr, BSIM3v1DPdpBinding, BSIM3v1dNodePrime, BSIM3v1dNodePrime);
            CREATE_KLU_BINDING_TABLE(BSIM3v1SPspPtr, BSIM3v1SPspBinding, BSIM3v1sNodePrime, BSIM3v1sNodePrime);
            CREATE_KLU_BINDING_TABLE(BSIM3v1DdpPtr, BSIM3v1DdpBinding, BSIM3v1dNode, BSIM3v1dNodePrime);
            CREATE_KLU_BINDING_TABLE(BSIM3v1GbPtr, BSIM3v1GbBinding, BSIM3v1gNode, BSIM3v1bNode);
            CREATE_KLU_BINDING_TABLE(BSIM3v1GdpPtr, BSIM3v1GdpBinding, BSIM3v1gNode, BSIM3v1dNodePrime);
            CREATE_KLU_BINDING_TABLE(BSIM3v1GspPtr, BSIM3v1GspBinding, BSIM3v1gNode, BSIM3v1sNodePrime);
            CREATE_KLU_BINDING_TABLE(BSIM3v1SspPtr, BSIM3v1SspBinding, BSIM3v1sNode, BSIM3v1sNodePrime);
            CREATE_KLU_BINDING_TABLE(BSIM3v1BdpPtr, BSIM3v1BdpBinding, BSIM3v1bNode, BSIM3v1dNodePrime);
            CREATE_KLU_BINDING_TABLE(BSIM3v1BspPtr, BSIM3v1BspBinding, BSIM3v1bNode, BSIM3v1sNodePrime);
            CREATE_KLU_BINDING_TABLE(BSIM3v1DPspPtr, BSIM3v1DPspBinding, BSIM3v1dNodePrime, BSIM3v1sNodePrime);
            CREATE_KLU_BINDING_TABLE(BSIM3v1DPdPtr, BSIM3v1DPdBinding, BSIM3v1dNodePrime, BSIM3v1dNode);
            CREATE_KLU_BINDING_TABLE(BSIM3v1BgPtr, BSIM3v1BgBinding, BSIM3v1bNode, BSIM3v1gNode);
            CREATE_KLU_BINDING_TABLE(BSIM3v1DPgPtr, BSIM3v1DPgBinding, BSIM3v1dNodePrime, BSIM3v1gNode);
            CREATE_KLU_BINDING_TABLE(BSIM3v1SPgPtr, BSIM3v1SPgBinding, BSIM3v1sNodePrime, BSIM3v1gNode);
            CREATE_KLU_BINDING_TABLE(BSIM3v1SPsPtr, BSIM3v1SPsBinding, BSIM3v1sNodePrime, BSIM3v1sNode);
            CREATE_KLU_BINDING_TABLE(BSIM3v1DPbPtr, BSIM3v1DPbBinding, BSIM3v1dNodePrime, BSIM3v1bNode);
            CREATE_KLU_BINDING_TABLE(BSIM3v1SPbPtr, BSIM3v1SPbBinding, BSIM3v1sNodePrime, BSIM3v1bNode);
            CREATE_KLU_BINDING_TABLE(BSIM3v1SPdpPtr, BSIM3v1SPdpBinding, BSIM3v1sNodePrime, BSIM3v1dNodePrime);
            CREATE_KLU_BINDING_TABLE(BSIM3v1QqPtr, BSIM3v1QqBinding, BSIM3v1qNode, BSIM3v1qNode);
            CREATE_KLU_BINDING_TABLE(BSIM3v1QdpPtr, BSIM3v1QdpBinding, BSIM3v1qNode, BSIM3v1dNodePrime);
            CREATE_KLU_BINDING_TABLE(BSIM3v1QspPtr, BSIM3v1QspBinding, BSIM3v1qNode, BSIM3v1sNodePrime);
            CREATE_KLU_BINDING_TABLE(BSIM3v1QgPtr, BSIM3v1QgBinding, BSIM3v1qNode, BSIM3v1gNode);
            CREATE_KLU_BINDING_TABLE(BSIM3v1QbPtr, BSIM3v1QbBinding, BSIM3v1qNode, BSIM3v1bNode);
            CREATE_KLU_BINDING_TABLE(BSIM3v1DPqPtr, BSIM3v1DPqBinding, BSIM3v1dNodePrime, BSIM3v1qNode);
            CREATE_KLU_BINDING_TABLE(BSIM3v1SPqPtr, BSIM3v1SPqBinding, BSIM3v1sNodePrime, BSIM3v1qNode);
            CREATE_KLU_BINDING_TABLE(BSIM3v1GqPtr, BSIM3v1GqBinding, BSIM3v1gNode, BSIM3v1qNode);
            CREATE_KLU_BINDING_TABLE(BSIM3v1BqPtr, BSIM3v1BqBinding, BSIM3v1bNode, BSIM3v1qNode);
        }
    }

    return (OK) ;
}

int
BSIM3v1bindCSCComplex (GENmodel *inModel, CKTcircuit *ckt)
{
    BSIM3v1model *model = (BSIM3v1model *)inModel ;
    BSIM3v1instance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the BSIM3v1 models */
    for ( ; model != NULL ; model = model->BSIM3v1nextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->BSIM3v1instances ; here != NULL ; here = here->BSIM3v1nextInstance)
        {
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM3v1DdPtr, BSIM3v1DdBinding, BSIM3v1dNode, BSIM3v1dNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM3v1GgPtr, BSIM3v1GgBinding, BSIM3v1gNode, BSIM3v1gNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM3v1SsPtr, BSIM3v1SsBinding, BSIM3v1sNode, BSIM3v1sNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM3v1BbPtr, BSIM3v1BbBinding, BSIM3v1bNode, BSIM3v1bNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM3v1DPdpPtr, BSIM3v1DPdpBinding, BSIM3v1dNodePrime, BSIM3v1dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM3v1SPspPtr, BSIM3v1SPspBinding, BSIM3v1sNodePrime, BSIM3v1sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM3v1DdpPtr, BSIM3v1DdpBinding, BSIM3v1dNode, BSIM3v1dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM3v1GbPtr, BSIM3v1GbBinding, BSIM3v1gNode, BSIM3v1bNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM3v1GdpPtr, BSIM3v1GdpBinding, BSIM3v1gNode, BSIM3v1dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM3v1GspPtr, BSIM3v1GspBinding, BSIM3v1gNode, BSIM3v1sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM3v1SspPtr, BSIM3v1SspBinding, BSIM3v1sNode, BSIM3v1sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM3v1BdpPtr, BSIM3v1BdpBinding, BSIM3v1bNode, BSIM3v1dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM3v1BspPtr, BSIM3v1BspBinding, BSIM3v1bNode, BSIM3v1sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM3v1DPspPtr, BSIM3v1DPspBinding, BSIM3v1dNodePrime, BSIM3v1sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM3v1DPdPtr, BSIM3v1DPdBinding, BSIM3v1dNodePrime, BSIM3v1dNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM3v1BgPtr, BSIM3v1BgBinding, BSIM3v1bNode, BSIM3v1gNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM3v1DPgPtr, BSIM3v1DPgBinding, BSIM3v1dNodePrime, BSIM3v1gNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM3v1SPgPtr, BSIM3v1SPgBinding, BSIM3v1sNodePrime, BSIM3v1gNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM3v1SPsPtr, BSIM3v1SPsBinding, BSIM3v1sNodePrime, BSIM3v1sNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM3v1DPbPtr, BSIM3v1DPbBinding, BSIM3v1dNodePrime, BSIM3v1bNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM3v1SPbPtr, BSIM3v1SPbBinding, BSIM3v1sNodePrime, BSIM3v1bNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM3v1SPdpPtr, BSIM3v1SPdpBinding, BSIM3v1sNodePrime, BSIM3v1dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM3v1QqPtr, BSIM3v1QqBinding, BSIM3v1qNode, BSIM3v1qNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM3v1QdpPtr, BSIM3v1QdpBinding, BSIM3v1qNode, BSIM3v1dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM3v1QspPtr, BSIM3v1QspBinding, BSIM3v1qNode, BSIM3v1sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM3v1QgPtr, BSIM3v1QgBinding, BSIM3v1qNode, BSIM3v1gNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM3v1QbPtr, BSIM3v1QbBinding, BSIM3v1qNode, BSIM3v1bNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM3v1DPqPtr, BSIM3v1DPqBinding, BSIM3v1dNodePrime, BSIM3v1qNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM3v1SPqPtr, BSIM3v1SPqBinding, BSIM3v1sNodePrime, BSIM3v1qNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM3v1GqPtr, BSIM3v1GqBinding, BSIM3v1gNode, BSIM3v1qNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM3v1BqPtr, BSIM3v1BqBinding, BSIM3v1bNode, BSIM3v1qNode);
        }
    }

    return (OK) ;
}

int
BSIM3v1bindCSCComplexToReal (GENmodel *inModel, CKTcircuit *ckt)
{
    BSIM3v1model *model = (BSIM3v1model *)inModel ;
    BSIM3v1instance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the BSIM3v1 models */
    for ( ; model != NULL ; model = model->BSIM3v1nextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->BSIM3v1instances ; here != NULL ; here = here->BSIM3v1nextInstance)
        {
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM3v1DdPtr, BSIM3v1DdBinding, BSIM3v1dNode, BSIM3v1dNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM3v1GgPtr, BSIM3v1GgBinding, BSIM3v1gNode, BSIM3v1gNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM3v1SsPtr, BSIM3v1SsBinding, BSIM3v1sNode, BSIM3v1sNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM3v1BbPtr, BSIM3v1BbBinding, BSIM3v1bNode, BSIM3v1bNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM3v1DPdpPtr, BSIM3v1DPdpBinding, BSIM3v1dNodePrime, BSIM3v1dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM3v1SPspPtr, BSIM3v1SPspBinding, BSIM3v1sNodePrime, BSIM3v1sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM3v1DdpPtr, BSIM3v1DdpBinding, BSIM3v1dNode, BSIM3v1dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM3v1GbPtr, BSIM3v1GbBinding, BSIM3v1gNode, BSIM3v1bNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM3v1GdpPtr, BSIM3v1GdpBinding, BSIM3v1gNode, BSIM3v1dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM3v1GspPtr, BSIM3v1GspBinding, BSIM3v1gNode, BSIM3v1sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM3v1SspPtr, BSIM3v1SspBinding, BSIM3v1sNode, BSIM3v1sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM3v1BdpPtr, BSIM3v1BdpBinding, BSIM3v1bNode, BSIM3v1dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM3v1BspPtr, BSIM3v1BspBinding, BSIM3v1bNode, BSIM3v1sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM3v1DPspPtr, BSIM3v1DPspBinding, BSIM3v1dNodePrime, BSIM3v1sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM3v1DPdPtr, BSIM3v1DPdBinding, BSIM3v1dNodePrime, BSIM3v1dNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM3v1BgPtr, BSIM3v1BgBinding, BSIM3v1bNode, BSIM3v1gNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM3v1DPgPtr, BSIM3v1DPgBinding, BSIM3v1dNodePrime, BSIM3v1gNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM3v1SPgPtr, BSIM3v1SPgBinding, BSIM3v1sNodePrime, BSIM3v1gNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM3v1SPsPtr, BSIM3v1SPsBinding, BSIM3v1sNodePrime, BSIM3v1sNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM3v1DPbPtr, BSIM3v1DPbBinding, BSIM3v1dNodePrime, BSIM3v1bNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM3v1SPbPtr, BSIM3v1SPbBinding, BSIM3v1sNodePrime, BSIM3v1bNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM3v1SPdpPtr, BSIM3v1SPdpBinding, BSIM3v1sNodePrime, BSIM3v1dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM3v1QqPtr, BSIM3v1QqBinding, BSIM3v1qNode, BSIM3v1qNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM3v1QdpPtr, BSIM3v1QdpBinding, BSIM3v1qNode, BSIM3v1dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM3v1QspPtr, BSIM3v1QspBinding, BSIM3v1qNode, BSIM3v1sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM3v1QgPtr, BSIM3v1QgBinding, BSIM3v1qNode, BSIM3v1gNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM3v1QbPtr, BSIM3v1QbBinding, BSIM3v1qNode, BSIM3v1bNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM3v1DPqPtr, BSIM3v1DPqBinding, BSIM3v1dNodePrime, BSIM3v1qNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM3v1SPqPtr, BSIM3v1SPqBinding, BSIM3v1sNodePrime, BSIM3v1qNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM3v1GqPtr, BSIM3v1GqBinding, BSIM3v1gNode, BSIM3v1qNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM3v1BqPtr, BSIM3v1BqBinding, BSIM3v1bNode, BSIM3v1qNode);
        }
    }

    return (OK) ;
}
