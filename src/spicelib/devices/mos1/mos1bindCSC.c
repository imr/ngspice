/**********
Author: 2013 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "mos1defs.h"
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
MOS1bindCSC (GENmodel *inModel, CKTcircuit *ckt)
{
    MOS1model *model = (MOS1model *)inModel ;
    MOS1instance *here ;
    double *i ;
    BindElement *matched, *BindStruct ;
    size_t nz ;

    BindStruct = ckt->CKTmatrix->CKTbindStruct ;
    nz = (size_t)ckt->CKTmatrix->CKTklunz ;

    /* loop through all the MOS1 models */
    for ( ; model != NULL ; model = MOS1nextModel(model))
    {
        /* loop through all the instances of the model */
        for (here = MOS1instances(model); here != NULL ; here = MOS1nextInstance(here))
        {
            CREATE_KLU_BINDING_TABLE(MOS1DdPtr, MOS1DdBinding, MOS1dNode, MOS1dNode);
            CREATE_KLU_BINDING_TABLE(MOS1GgPtr, MOS1GgBinding, MOS1gNode, MOS1gNode);
            CREATE_KLU_BINDING_TABLE(MOS1SsPtr, MOS1SsBinding, MOS1sNode, MOS1sNode);
            CREATE_KLU_BINDING_TABLE(MOS1BbPtr, MOS1BbBinding, MOS1bNode, MOS1bNode);
            CREATE_KLU_BINDING_TABLE(MOS1DPdpPtr, MOS1DPdpBinding, MOS1dNodePrime, MOS1dNodePrime);
            CREATE_KLU_BINDING_TABLE(MOS1SPspPtr, MOS1SPspBinding, MOS1sNodePrime, MOS1sNodePrime);
            CREATE_KLU_BINDING_TABLE(MOS1DdpPtr, MOS1DdpBinding, MOS1dNode, MOS1dNodePrime);
            CREATE_KLU_BINDING_TABLE(MOS1GbPtr, MOS1GbBinding, MOS1gNode, MOS1bNode);
            CREATE_KLU_BINDING_TABLE(MOS1GdpPtr, MOS1GdpBinding, MOS1gNode, MOS1dNodePrime);
            CREATE_KLU_BINDING_TABLE(MOS1GspPtr, MOS1GspBinding, MOS1gNode, MOS1sNodePrime);
            CREATE_KLU_BINDING_TABLE(MOS1SspPtr, MOS1SspBinding, MOS1sNode, MOS1sNodePrime);
            CREATE_KLU_BINDING_TABLE(MOS1BdpPtr, MOS1BdpBinding, MOS1bNode, MOS1dNodePrime);
            CREATE_KLU_BINDING_TABLE(MOS1BspPtr, MOS1BspBinding, MOS1bNode, MOS1sNodePrime);
            CREATE_KLU_BINDING_TABLE(MOS1DPspPtr, MOS1DPspBinding, MOS1dNodePrime, MOS1sNodePrime);
            CREATE_KLU_BINDING_TABLE(MOS1DPdPtr, MOS1DPdBinding, MOS1dNodePrime, MOS1dNode);
            CREATE_KLU_BINDING_TABLE(MOS1BgPtr, MOS1BgBinding, MOS1bNode, MOS1gNode);
            CREATE_KLU_BINDING_TABLE(MOS1DPgPtr, MOS1DPgBinding, MOS1dNodePrime, MOS1gNode);
            CREATE_KLU_BINDING_TABLE(MOS1SPgPtr, MOS1SPgBinding, MOS1sNodePrime, MOS1gNode);
            CREATE_KLU_BINDING_TABLE(MOS1SPsPtr, MOS1SPsBinding, MOS1sNodePrime, MOS1sNode);
            CREATE_KLU_BINDING_TABLE(MOS1DPbPtr, MOS1DPbBinding, MOS1dNodePrime, MOS1bNode);
            CREATE_KLU_BINDING_TABLE(MOS1SPbPtr, MOS1SPbBinding, MOS1sNodePrime, MOS1bNode);
            CREATE_KLU_BINDING_TABLE(MOS1SPdpPtr, MOS1SPdpBinding, MOS1sNodePrime, MOS1dNodePrime);
        }
    }

    return (OK) ;
}

int
MOS1bindCSCComplex (GENmodel *inModel, CKTcircuit *ckt)
{
    MOS1model *model = (MOS1model *)inModel ;
    MOS1instance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the MOS1 models */
    for ( ; model != NULL ; model = MOS1nextModel(model))
    {
        /* loop through all the instances of the model */
        for (here = MOS1instances(model); here != NULL ; here = MOS1nextInstance(here))
        {
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MOS1DdPtr, MOS1DdBinding, MOS1dNode, MOS1dNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MOS1GgPtr, MOS1GgBinding, MOS1gNode, MOS1gNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MOS1SsPtr, MOS1SsBinding, MOS1sNode, MOS1sNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MOS1BbPtr, MOS1BbBinding, MOS1bNode, MOS1bNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MOS1DPdpPtr, MOS1DPdpBinding, MOS1dNodePrime, MOS1dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MOS1SPspPtr, MOS1SPspBinding, MOS1sNodePrime, MOS1sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MOS1DdpPtr, MOS1DdpBinding, MOS1dNode, MOS1dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MOS1GbPtr, MOS1GbBinding, MOS1gNode, MOS1bNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MOS1GdpPtr, MOS1GdpBinding, MOS1gNode, MOS1dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MOS1GspPtr, MOS1GspBinding, MOS1gNode, MOS1sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MOS1SspPtr, MOS1SspBinding, MOS1sNode, MOS1sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MOS1BdpPtr, MOS1BdpBinding, MOS1bNode, MOS1dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MOS1BspPtr, MOS1BspBinding, MOS1bNode, MOS1sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MOS1DPspPtr, MOS1DPspBinding, MOS1dNodePrime, MOS1sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MOS1DPdPtr, MOS1DPdBinding, MOS1dNodePrime, MOS1dNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MOS1BgPtr, MOS1BgBinding, MOS1bNode, MOS1gNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MOS1DPgPtr, MOS1DPgBinding, MOS1dNodePrime, MOS1gNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MOS1SPgPtr, MOS1SPgBinding, MOS1sNodePrime, MOS1gNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MOS1SPsPtr, MOS1SPsBinding, MOS1sNodePrime, MOS1sNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MOS1DPbPtr, MOS1DPbBinding, MOS1dNodePrime, MOS1bNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MOS1SPbPtr, MOS1SPbBinding, MOS1sNodePrime, MOS1bNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MOS1SPdpPtr, MOS1SPdpBinding, MOS1sNodePrime, MOS1dNodePrime);
        }
    }

    return (OK) ;
}

int
MOS1bindCSCComplexToReal (GENmodel *inModel, CKTcircuit *ckt)
{
    MOS1model *model = (MOS1model *)inModel ;
    MOS1instance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the MOS1 models */
    for ( ; model != NULL ; model = MOS1nextModel(model))
    {
        /* loop through all the instances of the model */
        for (here = MOS1instances(model); here != NULL ; here = MOS1nextInstance(here))
        {
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MOS1DdPtr, MOS1DdBinding, MOS1dNode, MOS1dNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MOS1GgPtr, MOS1GgBinding, MOS1gNode, MOS1gNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MOS1SsPtr, MOS1SsBinding, MOS1sNode, MOS1sNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MOS1BbPtr, MOS1BbBinding, MOS1bNode, MOS1bNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MOS1DPdpPtr, MOS1DPdpBinding, MOS1dNodePrime, MOS1dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MOS1SPspPtr, MOS1SPspBinding, MOS1sNodePrime, MOS1sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MOS1DdpPtr, MOS1DdpBinding, MOS1dNode, MOS1dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MOS1GbPtr, MOS1GbBinding, MOS1gNode, MOS1bNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MOS1GdpPtr, MOS1GdpBinding, MOS1gNode, MOS1dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MOS1GspPtr, MOS1GspBinding, MOS1gNode, MOS1sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MOS1SspPtr, MOS1SspBinding, MOS1sNode, MOS1sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MOS1BdpPtr, MOS1BdpBinding, MOS1bNode, MOS1dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MOS1BspPtr, MOS1BspBinding, MOS1bNode, MOS1sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MOS1DPspPtr, MOS1DPspBinding, MOS1dNodePrime, MOS1sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MOS1DPdPtr, MOS1DPdBinding, MOS1dNodePrime, MOS1dNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MOS1BgPtr, MOS1BgBinding, MOS1bNode, MOS1gNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MOS1DPgPtr, MOS1DPgBinding, MOS1dNodePrime, MOS1gNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MOS1SPgPtr, MOS1SPgBinding, MOS1sNodePrime, MOS1gNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MOS1SPsPtr, MOS1SPsBinding, MOS1sNodePrime, MOS1sNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MOS1DPbPtr, MOS1DPbBinding, MOS1dNodePrime, MOS1bNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MOS1SPbPtr, MOS1SPbBinding, MOS1sNodePrime, MOS1bNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MOS1SPdpPtr, MOS1SPdpBinding, MOS1sNodePrime, MOS1dNodePrime);
        }
    }

    return (OK) ;
}
