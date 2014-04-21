/**********
Author: 2013 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "bsim3v0def.h"
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
BSIM3v0bindCSC (GENmodel *inModel, CKTcircuit *ckt)
{
    BSIM3v0model *model = (BSIM3v0model *)inModel ;
    BSIM3v0instance *here ;
    double *i ;
    BindElement *matched, *BindStruct ;
    size_t nz ;

    BindStruct = ckt->CKTmatrix->CKTbindStruct ;
    nz = (size_t)ckt->CKTmatrix->CKTklunz ;

    /* loop through all the BSIM3v0 models */
    for ( ; model != NULL ; model = model->BSIM3v0nextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->BSIM3v0instances ; here != NULL ; here = here->BSIM3v0nextInstance)
        {
            CREATE_KLU_BINDING_TABLE(BSIM3v0DdPtr, BSIM3v0DdBinding, BSIM3v0dNode, BSIM3v0dNode);
            CREATE_KLU_BINDING_TABLE(BSIM3v0GgPtr, BSIM3v0GgBinding, BSIM3v0gNode, BSIM3v0gNode);
            CREATE_KLU_BINDING_TABLE(BSIM3v0SsPtr, BSIM3v0SsBinding, BSIM3v0sNode, BSIM3v0sNode);
            CREATE_KLU_BINDING_TABLE(BSIM3v0BbPtr, BSIM3v0BbBinding, BSIM3v0bNode, BSIM3v0bNode);
            CREATE_KLU_BINDING_TABLE(BSIM3v0DPdpPtr, BSIM3v0DPdpBinding, BSIM3v0dNodePrime, BSIM3v0dNodePrime);
            CREATE_KLU_BINDING_TABLE(BSIM3v0SPspPtr, BSIM3v0SPspBinding, BSIM3v0sNodePrime, BSIM3v0sNodePrime);
            CREATE_KLU_BINDING_TABLE(BSIM3v0DdpPtr, BSIM3v0DdpBinding, BSIM3v0dNode, BSIM3v0dNodePrime);
            CREATE_KLU_BINDING_TABLE(BSIM3v0GbPtr, BSIM3v0GbBinding, BSIM3v0gNode, BSIM3v0bNode);
            CREATE_KLU_BINDING_TABLE(BSIM3v0GdpPtr, BSIM3v0GdpBinding, BSIM3v0gNode, BSIM3v0dNodePrime);
            CREATE_KLU_BINDING_TABLE(BSIM3v0GspPtr, BSIM3v0GspBinding, BSIM3v0gNode, BSIM3v0sNodePrime);
            CREATE_KLU_BINDING_TABLE(BSIM3v0SspPtr, BSIM3v0SspBinding, BSIM3v0sNode, BSIM3v0sNodePrime);
            CREATE_KLU_BINDING_TABLE(BSIM3v0BdpPtr, BSIM3v0BdpBinding, BSIM3v0bNode, BSIM3v0dNodePrime);
            CREATE_KLU_BINDING_TABLE(BSIM3v0BspPtr, BSIM3v0BspBinding, BSIM3v0bNode, BSIM3v0sNodePrime);
            CREATE_KLU_BINDING_TABLE(BSIM3v0DPspPtr, BSIM3v0DPspBinding, BSIM3v0dNodePrime, BSIM3v0sNodePrime);
            CREATE_KLU_BINDING_TABLE(BSIM3v0DPdPtr, BSIM3v0DPdBinding, BSIM3v0dNodePrime, BSIM3v0dNode);
            CREATE_KLU_BINDING_TABLE(BSIM3v0BgPtr, BSIM3v0BgBinding, BSIM3v0bNode, BSIM3v0gNode);
            CREATE_KLU_BINDING_TABLE(BSIM3v0DPgPtr, BSIM3v0DPgBinding, BSIM3v0dNodePrime, BSIM3v0gNode);
            CREATE_KLU_BINDING_TABLE(BSIM3v0SPgPtr, BSIM3v0SPgBinding, BSIM3v0sNodePrime, BSIM3v0gNode);
            CREATE_KLU_BINDING_TABLE(BSIM3v0SPsPtr, BSIM3v0SPsBinding, BSIM3v0sNodePrime, BSIM3v0sNode);
            CREATE_KLU_BINDING_TABLE(BSIM3v0DPbPtr, BSIM3v0DPbBinding, BSIM3v0dNodePrime, BSIM3v0bNode);
            CREATE_KLU_BINDING_TABLE(BSIM3v0SPbPtr, BSIM3v0SPbBinding, BSIM3v0sNodePrime, BSIM3v0bNode);
            CREATE_KLU_BINDING_TABLE(BSIM3v0SPdpPtr, BSIM3v0SPdpBinding, BSIM3v0sNodePrime, BSIM3v0dNodePrime);
            CREATE_KLU_BINDING_TABLE(BSIM3v0QqPtr, BSIM3v0QqBinding, BSIM3v0qNode, BSIM3v0qNode);
            CREATE_KLU_BINDING_TABLE(BSIM3v0QdpPtr, BSIM3v0QdpBinding, BSIM3v0qNode, BSIM3v0dNodePrime);
            CREATE_KLU_BINDING_TABLE(BSIM3v0QspPtr, BSIM3v0QspBinding, BSIM3v0qNode, BSIM3v0sNodePrime);
            CREATE_KLU_BINDING_TABLE(BSIM3v0QgPtr, BSIM3v0QgBinding, BSIM3v0qNode, BSIM3v0gNode);
            CREATE_KLU_BINDING_TABLE(BSIM3v0QbPtr, BSIM3v0QbBinding, BSIM3v0qNode, BSIM3v0bNode);
            CREATE_KLU_BINDING_TABLE(BSIM3v0DPqPtr, BSIM3v0DPqBinding, BSIM3v0dNodePrime, BSIM3v0qNode);
            CREATE_KLU_BINDING_TABLE(BSIM3v0SPqPtr, BSIM3v0SPqBinding, BSIM3v0sNodePrime, BSIM3v0qNode);
            CREATE_KLU_BINDING_TABLE(BSIM3v0GqPtr, BSIM3v0GqBinding, BSIM3v0gNode, BSIM3v0qNode);
            CREATE_KLU_BINDING_TABLE(BSIM3v0BqPtr, BSIM3v0BqBinding, BSIM3v0bNode, BSIM3v0qNode);
        }
    }

    return (OK) ;
}

int
BSIM3v0bindCSCComplex (GENmodel *inModel, CKTcircuit *ckt)
{
    BSIM3v0model *model = (BSIM3v0model *)inModel ;
    BSIM3v0instance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the BSIM3v0 models */
    for ( ; model != NULL ; model = model->BSIM3v0nextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->BSIM3v0instances ; here != NULL ; here = here->BSIM3v0nextInstance)
        {
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM3v0DdPtr, BSIM3v0DdBinding, BSIM3v0dNode, BSIM3v0dNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM3v0GgPtr, BSIM3v0GgBinding, BSIM3v0gNode, BSIM3v0gNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM3v0SsPtr, BSIM3v0SsBinding, BSIM3v0sNode, BSIM3v0sNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM3v0BbPtr, BSIM3v0BbBinding, BSIM3v0bNode, BSIM3v0bNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM3v0DPdpPtr, BSIM3v0DPdpBinding, BSIM3v0dNodePrime, BSIM3v0dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM3v0SPspPtr, BSIM3v0SPspBinding, BSIM3v0sNodePrime, BSIM3v0sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM3v0DdpPtr, BSIM3v0DdpBinding, BSIM3v0dNode, BSIM3v0dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM3v0GbPtr, BSIM3v0GbBinding, BSIM3v0gNode, BSIM3v0bNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM3v0GdpPtr, BSIM3v0GdpBinding, BSIM3v0gNode, BSIM3v0dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM3v0GspPtr, BSIM3v0GspBinding, BSIM3v0gNode, BSIM3v0sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM3v0SspPtr, BSIM3v0SspBinding, BSIM3v0sNode, BSIM3v0sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM3v0BdpPtr, BSIM3v0BdpBinding, BSIM3v0bNode, BSIM3v0dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM3v0BspPtr, BSIM3v0BspBinding, BSIM3v0bNode, BSIM3v0sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM3v0DPspPtr, BSIM3v0DPspBinding, BSIM3v0dNodePrime, BSIM3v0sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM3v0DPdPtr, BSIM3v0DPdBinding, BSIM3v0dNodePrime, BSIM3v0dNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM3v0BgPtr, BSIM3v0BgBinding, BSIM3v0bNode, BSIM3v0gNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM3v0DPgPtr, BSIM3v0DPgBinding, BSIM3v0dNodePrime, BSIM3v0gNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM3v0SPgPtr, BSIM3v0SPgBinding, BSIM3v0sNodePrime, BSIM3v0gNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM3v0SPsPtr, BSIM3v0SPsBinding, BSIM3v0sNodePrime, BSIM3v0sNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM3v0DPbPtr, BSIM3v0DPbBinding, BSIM3v0dNodePrime, BSIM3v0bNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM3v0SPbPtr, BSIM3v0SPbBinding, BSIM3v0sNodePrime, BSIM3v0bNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM3v0SPdpPtr, BSIM3v0SPdpBinding, BSIM3v0sNodePrime, BSIM3v0dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM3v0QqPtr, BSIM3v0QqBinding, BSIM3v0qNode, BSIM3v0qNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM3v0QdpPtr, BSIM3v0QdpBinding, BSIM3v0qNode, BSIM3v0dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM3v0QspPtr, BSIM3v0QspBinding, BSIM3v0qNode, BSIM3v0sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM3v0QgPtr, BSIM3v0QgBinding, BSIM3v0qNode, BSIM3v0gNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM3v0QbPtr, BSIM3v0QbBinding, BSIM3v0qNode, BSIM3v0bNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM3v0DPqPtr, BSIM3v0DPqBinding, BSIM3v0dNodePrime, BSIM3v0qNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM3v0SPqPtr, BSIM3v0SPqBinding, BSIM3v0sNodePrime, BSIM3v0qNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM3v0GqPtr, BSIM3v0GqBinding, BSIM3v0gNode, BSIM3v0qNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM3v0BqPtr, BSIM3v0BqBinding, BSIM3v0bNode, BSIM3v0qNode);
        }
    }

    return (OK) ;
}

int
BSIM3v0bindCSCComplexToReal (GENmodel *inModel, CKTcircuit *ckt)
{
    BSIM3v0model *model = (BSIM3v0model *)inModel ;
    BSIM3v0instance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the BSIM3v0 models */
    for ( ; model != NULL ; model = model->BSIM3v0nextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->BSIM3v0instances ; here != NULL ; here = here->BSIM3v0nextInstance)
        {
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM3v0DdPtr, BSIM3v0DdBinding, BSIM3v0dNode, BSIM3v0dNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM3v0GgPtr, BSIM3v0GgBinding, BSIM3v0gNode, BSIM3v0gNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM3v0SsPtr, BSIM3v0SsBinding, BSIM3v0sNode, BSIM3v0sNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM3v0BbPtr, BSIM3v0BbBinding, BSIM3v0bNode, BSIM3v0bNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM3v0DPdpPtr, BSIM3v0DPdpBinding, BSIM3v0dNodePrime, BSIM3v0dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM3v0SPspPtr, BSIM3v0SPspBinding, BSIM3v0sNodePrime, BSIM3v0sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM3v0DdpPtr, BSIM3v0DdpBinding, BSIM3v0dNode, BSIM3v0dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM3v0GbPtr, BSIM3v0GbBinding, BSIM3v0gNode, BSIM3v0bNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM3v0GdpPtr, BSIM3v0GdpBinding, BSIM3v0gNode, BSIM3v0dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM3v0GspPtr, BSIM3v0GspBinding, BSIM3v0gNode, BSIM3v0sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM3v0SspPtr, BSIM3v0SspBinding, BSIM3v0sNode, BSIM3v0sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM3v0BdpPtr, BSIM3v0BdpBinding, BSIM3v0bNode, BSIM3v0dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM3v0BspPtr, BSIM3v0BspBinding, BSIM3v0bNode, BSIM3v0sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM3v0DPspPtr, BSIM3v0DPspBinding, BSIM3v0dNodePrime, BSIM3v0sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM3v0DPdPtr, BSIM3v0DPdBinding, BSIM3v0dNodePrime, BSIM3v0dNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM3v0BgPtr, BSIM3v0BgBinding, BSIM3v0bNode, BSIM3v0gNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM3v0DPgPtr, BSIM3v0DPgBinding, BSIM3v0dNodePrime, BSIM3v0gNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM3v0SPgPtr, BSIM3v0SPgBinding, BSIM3v0sNodePrime, BSIM3v0gNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM3v0SPsPtr, BSIM3v0SPsBinding, BSIM3v0sNodePrime, BSIM3v0sNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM3v0DPbPtr, BSIM3v0DPbBinding, BSIM3v0dNodePrime, BSIM3v0bNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM3v0SPbPtr, BSIM3v0SPbBinding, BSIM3v0sNodePrime, BSIM3v0bNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM3v0SPdpPtr, BSIM3v0SPdpBinding, BSIM3v0sNodePrime, BSIM3v0dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM3v0QqPtr, BSIM3v0QqBinding, BSIM3v0qNode, BSIM3v0qNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM3v0QdpPtr, BSIM3v0QdpBinding, BSIM3v0qNode, BSIM3v0dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM3v0QspPtr, BSIM3v0QspBinding, BSIM3v0qNode, BSIM3v0sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM3v0QgPtr, BSIM3v0QgBinding, BSIM3v0qNode, BSIM3v0gNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM3v0QbPtr, BSIM3v0QbBinding, BSIM3v0qNode, BSIM3v0bNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM3v0DPqPtr, BSIM3v0DPqBinding, BSIM3v0dNodePrime, BSIM3v0qNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM3v0SPqPtr, BSIM3v0SPqBinding, BSIM3v0sNodePrime, BSIM3v0qNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM3v0GqPtr, BSIM3v0GqBinding, BSIM3v0gNode, BSIM3v0qNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM3v0BqPtr, BSIM3v0BqBinding, BSIM3v0bNode, BSIM3v0qNode);
        }
    }

    return (OK) ;
}
