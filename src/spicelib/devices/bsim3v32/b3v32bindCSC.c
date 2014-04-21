/**********
Author: 2013 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "bsim3v32def.h"
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
BSIM3v32bindCSC (GENmodel *inModel, CKTcircuit *ckt)
{
    BSIM3v32model *model = (BSIM3v32model *)inModel ;
    BSIM3v32instance *here ;
    double *i ;
    BindElement *matched, *BindStruct ;
    size_t nz ;

    BindStruct = ckt->CKTmatrix->CKTbindStruct ;
    nz = (size_t)ckt->CKTmatrix->CKTklunz ;

    /* loop through all the BSIM3v32 models */
    for ( ; model != NULL ; model = BSIM3v32nextModel(model))
    {
        /* loop through all the instances of the model */
        for (here = BSIM3v32instances(model); here != NULL ; here = BSIM3v32nextInstance(here))
        {
            CREATE_KLU_BINDING_TABLE(BSIM3v32DdPtr, BSIM3v32DdBinding, BSIM3v32dNode, BSIM3v32dNode);
            CREATE_KLU_BINDING_TABLE(BSIM3v32GgPtr, BSIM3v32GgBinding, BSIM3v32gNode, BSIM3v32gNode);
            CREATE_KLU_BINDING_TABLE(BSIM3v32SsPtr, BSIM3v32SsBinding, BSIM3v32sNode, BSIM3v32sNode);
            CREATE_KLU_BINDING_TABLE(BSIM3v32BbPtr, BSIM3v32BbBinding, BSIM3v32bNode, BSIM3v32bNode);
            CREATE_KLU_BINDING_TABLE(BSIM3v32DPdpPtr, BSIM3v32DPdpBinding, BSIM3v32dNodePrime, BSIM3v32dNodePrime);
            CREATE_KLU_BINDING_TABLE(BSIM3v32SPspPtr, BSIM3v32SPspBinding, BSIM3v32sNodePrime, BSIM3v32sNodePrime);
            CREATE_KLU_BINDING_TABLE(BSIM3v32DdpPtr, BSIM3v32DdpBinding, BSIM3v32dNode, BSIM3v32dNodePrime);
            CREATE_KLU_BINDING_TABLE(BSIM3v32GbPtr, BSIM3v32GbBinding, BSIM3v32gNode, BSIM3v32bNode);
            CREATE_KLU_BINDING_TABLE(BSIM3v32GdpPtr, BSIM3v32GdpBinding, BSIM3v32gNode, BSIM3v32dNodePrime);
            CREATE_KLU_BINDING_TABLE(BSIM3v32GspPtr, BSIM3v32GspBinding, BSIM3v32gNode, BSIM3v32sNodePrime);
            CREATE_KLU_BINDING_TABLE(BSIM3v32SspPtr, BSIM3v32SspBinding, BSIM3v32sNode, BSIM3v32sNodePrime);
            CREATE_KLU_BINDING_TABLE(BSIM3v32BdpPtr, BSIM3v32BdpBinding, BSIM3v32bNode, BSIM3v32dNodePrime);
            CREATE_KLU_BINDING_TABLE(BSIM3v32BspPtr, BSIM3v32BspBinding, BSIM3v32bNode, BSIM3v32sNodePrime);
            CREATE_KLU_BINDING_TABLE(BSIM3v32DPspPtr, BSIM3v32DPspBinding, BSIM3v32dNodePrime, BSIM3v32sNodePrime);
            CREATE_KLU_BINDING_TABLE(BSIM3v32DPdPtr, BSIM3v32DPdBinding, BSIM3v32dNodePrime, BSIM3v32dNode);
            CREATE_KLU_BINDING_TABLE(BSIM3v32BgPtr, BSIM3v32BgBinding, BSIM3v32bNode, BSIM3v32gNode);
            CREATE_KLU_BINDING_TABLE(BSIM3v32DPgPtr, BSIM3v32DPgBinding, BSIM3v32dNodePrime, BSIM3v32gNode);
            CREATE_KLU_BINDING_TABLE(BSIM3v32SPgPtr, BSIM3v32SPgBinding, BSIM3v32sNodePrime, BSIM3v32gNode);
            CREATE_KLU_BINDING_TABLE(BSIM3v32SPsPtr, BSIM3v32SPsBinding, BSIM3v32sNodePrime, BSIM3v32sNode);
            CREATE_KLU_BINDING_TABLE(BSIM3v32DPbPtr, BSIM3v32DPbBinding, BSIM3v32dNodePrime, BSIM3v32bNode);
            CREATE_KLU_BINDING_TABLE(BSIM3v32SPbPtr, BSIM3v32SPbBinding, BSIM3v32sNodePrime, BSIM3v32bNode);
            CREATE_KLU_BINDING_TABLE(BSIM3v32SPdpPtr, BSIM3v32SPdpBinding, BSIM3v32sNodePrime, BSIM3v32dNodePrime);
            CREATE_KLU_BINDING_TABLE(BSIM3v32QqPtr, BSIM3v32QqBinding, BSIM3v32qNode, BSIM3v32qNode);
            CREATE_KLU_BINDING_TABLE(BSIM3v32QdpPtr, BSIM3v32QdpBinding, BSIM3v32qNode, BSIM3v32dNodePrime);
            CREATE_KLU_BINDING_TABLE(BSIM3v32QspPtr, BSIM3v32QspBinding, BSIM3v32qNode, BSIM3v32sNodePrime);
            CREATE_KLU_BINDING_TABLE(BSIM3v32QgPtr, BSIM3v32QgBinding, BSIM3v32qNode, BSIM3v32gNode);
            CREATE_KLU_BINDING_TABLE(BSIM3v32QbPtr, BSIM3v32QbBinding, BSIM3v32qNode, BSIM3v32bNode);
            CREATE_KLU_BINDING_TABLE(BSIM3v32DPqPtr, BSIM3v32DPqBinding, BSIM3v32dNodePrime, BSIM3v32qNode);
            CREATE_KLU_BINDING_TABLE(BSIM3v32SPqPtr, BSIM3v32SPqBinding, BSIM3v32sNodePrime, BSIM3v32qNode);
            CREATE_KLU_BINDING_TABLE(BSIM3v32GqPtr, BSIM3v32GqBinding, BSIM3v32gNode, BSIM3v32qNode);
            CREATE_KLU_BINDING_TABLE(BSIM3v32BqPtr, BSIM3v32BqBinding, BSIM3v32bNode, BSIM3v32qNode);
        }
    }

    return (OK) ;
}

int
BSIM3v32bindCSCComplex (GENmodel *inModel, CKTcircuit *ckt)
{
    BSIM3v32model *model = (BSIM3v32model *)inModel ;
    BSIM3v32instance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the BSIM3v32 models */
    for ( ; model != NULL ; model = BSIM3v32nextModel(model))
    {
        /* loop through all the instances of the model */
        for (here = BSIM3v32instances(model); here != NULL ; here = BSIM3v32nextInstance(here))
        {
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM3v32DdPtr, BSIM3v32DdBinding, BSIM3v32dNode, BSIM3v32dNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM3v32GgPtr, BSIM3v32GgBinding, BSIM3v32gNode, BSIM3v32gNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM3v32SsPtr, BSIM3v32SsBinding, BSIM3v32sNode, BSIM3v32sNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM3v32BbPtr, BSIM3v32BbBinding, BSIM3v32bNode, BSIM3v32bNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM3v32DPdpPtr, BSIM3v32DPdpBinding, BSIM3v32dNodePrime, BSIM3v32dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM3v32SPspPtr, BSIM3v32SPspBinding, BSIM3v32sNodePrime, BSIM3v32sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM3v32DdpPtr, BSIM3v32DdpBinding, BSIM3v32dNode, BSIM3v32dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM3v32GbPtr, BSIM3v32GbBinding, BSIM3v32gNode, BSIM3v32bNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM3v32GdpPtr, BSIM3v32GdpBinding, BSIM3v32gNode, BSIM3v32dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM3v32GspPtr, BSIM3v32GspBinding, BSIM3v32gNode, BSIM3v32sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM3v32SspPtr, BSIM3v32SspBinding, BSIM3v32sNode, BSIM3v32sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM3v32BdpPtr, BSIM3v32BdpBinding, BSIM3v32bNode, BSIM3v32dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM3v32BspPtr, BSIM3v32BspBinding, BSIM3v32bNode, BSIM3v32sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM3v32DPspPtr, BSIM3v32DPspBinding, BSIM3v32dNodePrime, BSIM3v32sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM3v32DPdPtr, BSIM3v32DPdBinding, BSIM3v32dNodePrime, BSIM3v32dNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM3v32BgPtr, BSIM3v32BgBinding, BSIM3v32bNode, BSIM3v32gNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM3v32DPgPtr, BSIM3v32DPgBinding, BSIM3v32dNodePrime, BSIM3v32gNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM3v32SPgPtr, BSIM3v32SPgBinding, BSIM3v32sNodePrime, BSIM3v32gNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM3v32SPsPtr, BSIM3v32SPsBinding, BSIM3v32sNodePrime, BSIM3v32sNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM3v32DPbPtr, BSIM3v32DPbBinding, BSIM3v32dNodePrime, BSIM3v32bNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM3v32SPbPtr, BSIM3v32SPbBinding, BSIM3v32sNodePrime, BSIM3v32bNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM3v32SPdpPtr, BSIM3v32SPdpBinding, BSIM3v32sNodePrime, BSIM3v32dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM3v32QqPtr, BSIM3v32QqBinding, BSIM3v32qNode, BSIM3v32qNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM3v32QdpPtr, BSIM3v32QdpBinding, BSIM3v32qNode, BSIM3v32dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM3v32QspPtr, BSIM3v32QspBinding, BSIM3v32qNode, BSIM3v32sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM3v32QgPtr, BSIM3v32QgBinding, BSIM3v32qNode, BSIM3v32gNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM3v32QbPtr, BSIM3v32QbBinding, BSIM3v32qNode, BSIM3v32bNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM3v32DPqPtr, BSIM3v32DPqBinding, BSIM3v32dNodePrime, BSIM3v32qNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM3v32SPqPtr, BSIM3v32SPqBinding, BSIM3v32sNodePrime, BSIM3v32qNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM3v32GqPtr, BSIM3v32GqBinding, BSIM3v32gNode, BSIM3v32qNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(BSIM3v32BqPtr, BSIM3v32BqBinding, BSIM3v32bNode, BSIM3v32qNode);
        }
    }

    return (OK) ;
}

int
BSIM3v32bindCSCComplexToReal (GENmodel *inModel, CKTcircuit *ckt)
{
    BSIM3v32model *model = (BSIM3v32model *)inModel ;
    BSIM3v32instance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the BSIM3v32 models */
    for ( ; model != NULL ; model = BSIM3v32nextModel(model))
    {
        /* loop through all the instances of the model */
        for (here = BSIM3v32instances(model); here != NULL ; here = BSIM3v32nextInstance(here))
        {
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM3v32DdPtr, BSIM3v32DdBinding, BSIM3v32dNode, BSIM3v32dNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM3v32GgPtr, BSIM3v32GgBinding, BSIM3v32gNode, BSIM3v32gNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM3v32SsPtr, BSIM3v32SsBinding, BSIM3v32sNode, BSIM3v32sNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM3v32BbPtr, BSIM3v32BbBinding, BSIM3v32bNode, BSIM3v32bNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM3v32DPdpPtr, BSIM3v32DPdpBinding, BSIM3v32dNodePrime, BSIM3v32dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM3v32SPspPtr, BSIM3v32SPspBinding, BSIM3v32sNodePrime, BSIM3v32sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM3v32DdpPtr, BSIM3v32DdpBinding, BSIM3v32dNode, BSIM3v32dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM3v32GbPtr, BSIM3v32GbBinding, BSIM3v32gNode, BSIM3v32bNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM3v32GdpPtr, BSIM3v32GdpBinding, BSIM3v32gNode, BSIM3v32dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM3v32GspPtr, BSIM3v32GspBinding, BSIM3v32gNode, BSIM3v32sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM3v32SspPtr, BSIM3v32SspBinding, BSIM3v32sNode, BSIM3v32sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM3v32BdpPtr, BSIM3v32BdpBinding, BSIM3v32bNode, BSIM3v32dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM3v32BspPtr, BSIM3v32BspBinding, BSIM3v32bNode, BSIM3v32sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM3v32DPspPtr, BSIM3v32DPspBinding, BSIM3v32dNodePrime, BSIM3v32sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM3v32DPdPtr, BSIM3v32DPdBinding, BSIM3v32dNodePrime, BSIM3v32dNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM3v32BgPtr, BSIM3v32BgBinding, BSIM3v32bNode, BSIM3v32gNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM3v32DPgPtr, BSIM3v32DPgBinding, BSIM3v32dNodePrime, BSIM3v32gNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM3v32SPgPtr, BSIM3v32SPgBinding, BSIM3v32sNodePrime, BSIM3v32gNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM3v32SPsPtr, BSIM3v32SPsBinding, BSIM3v32sNodePrime, BSIM3v32sNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM3v32DPbPtr, BSIM3v32DPbBinding, BSIM3v32dNodePrime, BSIM3v32bNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM3v32SPbPtr, BSIM3v32SPbBinding, BSIM3v32sNodePrime, BSIM3v32bNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM3v32SPdpPtr, BSIM3v32SPdpBinding, BSIM3v32sNodePrime, BSIM3v32dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM3v32QqPtr, BSIM3v32QqBinding, BSIM3v32qNode, BSIM3v32qNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM3v32QdpPtr, BSIM3v32QdpBinding, BSIM3v32qNode, BSIM3v32dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM3v32QspPtr, BSIM3v32QspBinding, BSIM3v32qNode, BSIM3v32sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM3v32QgPtr, BSIM3v32QgBinding, BSIM3v32qNode, BSIM3v32gNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM3v32QbPtr, BSIM3v32QbBinding, BSIM3v32qNode, BSIM3v32bNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM3v32DPqPtr, BSIM3v32DPqBinding, BSIM3v32dNodePrime, BSIM3v32qNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM3v32SPqPtr, BSIM3v32SPqBinding, BSIM3v32sNodePrime, BSIM3v32qNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM3v32GqPtr, BSIM3v32GqBinding, BSIM3v32gNode, BSIM3v32qNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(BSIM3v32BqPtr, BSIM3v32BqBinding, BSIM3v32bNode, BSIM3v32qNode);
        }
    }

    return (OK) ;
}
