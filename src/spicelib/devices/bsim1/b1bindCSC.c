/**********
Author: 2013 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "bsim1def.h"
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
B1bindCSC (GENmodel *inModel, CKTcircuit *ckt)
{
    B1model *model = (B1model *)inModel ;
    B1instance *here ;
    double *i ;
    BindElement *matched, *BindStruct ;
    size_t nz ;

    BindStruct = ckt->CKTmatrix->CKTbindStruct ;
    nz = (size_t)ckt->CKTmatrix->CKTklunz ;

    /* loop through all the B1 models */
    for ( ; model != NULL ; model = model->B1nextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->B1instances ; here != NULL ; here = here->B1nextInstance)
        {
            CREATE_KLU_BINDING_TABLE(B1DdPtr, B1DdBinding, B1dNode, B1dNode);
            CREATE_KLU_BINDING_TABLE(B1GgPtr, B1GgBinding, B1gNode, B1gNode);
            CREATE_KLU_BINDING_TABLE(B1SsPtr, B1SsBinding, B1sNode, B1sNode);
            CREATE_KLU_BINDING_TABLE(B1BbPtr, B1BbBinding, B1bNode, B1bNode);
            CREATE_KLU_BINDING_TABLE(B1DPdpPtr, B1DPdpBinding, B1dNodePrime, B1dNodePrime);
            CREATE_KLU_BINDING_TABLE(B1SPspPtr, B1SPspBinding, B1sNodePrime, B1sNodePrime);
            CREATE_KLU_BINDING_TABLE(B1DdpPtr, B1DdpBinding, B1dNode, B1dNodePrime);
            CREATE_KLU_BINDING_TABLE(B1GbPtr, B1GbBinding, B1gNode, B1bNode);
            CREATE_KLU_BINDING_TABLE(B1GdpPtr, B1GdpBinding, B1gNode, B1dNodePrime);
            CREATE_KLU_BINDING_TABLE(B1GspPtr, B1GspBinding, B1gNode, B1sNodePrime);
            CREATE_KLU_BINDING_TABLE(B1SspPtr, B1SspBinding, B1sNode, B1sNodePrime);
            CREATE_KLU_BINDING_TABLE(B1BdpPtr, B1BdpBinding, B1bNode, B1dNodePrime);
            CREATE_KLU_BINDING_TABLE(B1BspPtr, B1BspBinding, B1bNode, B1sNodePrime);
            CREATE_KLU_BINDING_TABLE(B1DPspPtr, B1DPspBinding, B1dNodePrime, B1sNodePrime);
            CREATE_KLU_BINDING_TABLE(B1DPdPtr, B1DPdBinding, B1dNodePrime, B1dNode);
            CREATE_KLU_BINDING_TABLE(B1BgPtr, B1BgBinding, B1bNode, B1gNode);
            CREATE_KLU_BINDING_TABLE(B1DPgPtr, B1DPgBinding, B1dNodePrime, B1gNode);
            CREATE_KLU_BINDING_TABLE(B1SPgPtr, B1SPgBinding, B1sNodePrime, B1gNode);
            CREATE_KLU_BINDING_TABLE(B1SPsPtr, B1SPsBinding, B1sNodePrime, B1sNode);
            CREATE_KLU_BINDING_TABLE(B1DPbPtr, B1DPbBinding, B1dNodePrime, B1bNode);
            CREATE_KLU_BINDING_TABLE(B1SPbPtr, B1SPbBinding, B1sNodePrime, B1bNode);
            CREATE_KLU_BINDING_TABLE(B1SPdpPtr, B1SPdpBinding, B1sNodePrime, B1dNodePrime);
        }
    }

    return (OK) ;
}

int
B1bindCSCComplex (GENmodel *inModel, CKTcircuit *ckt)
{
    B1model *model = (B1model *)inModel ;
    B1instance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the B1 models */
    for ( ; model != NULL ; model = model->B1nextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->B1instances ; here != NULL ; here = here->B1nextInstance)
        {
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B1DdPtr, B1DdBinding, B1dNode, B1dNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B1GgPtr, B1GgBinding, B1gNode, B1gNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B1SsPtr, B1SsBinding, B1sNode, B1sNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B1BbPtr, B1BbBinding, B1bNode, B1bNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B1DPdpPtr, B1DPdpBinding, B1dNodePrime, B1dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B1SPspPtr, B1SPspBinding, B1sNodePrime, B1sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B1DdpPtr, B1DdpBinding, B1dNode, B1dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B1GbPtr, B1GbBinding, B1gNode, B1bNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B1GdpPtr, B1GdpBinding, B1gNode, B1dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B1GspPtr, B1GspBinding, B1gNode, B1sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B1SspPtr, B1SspBinding, B1sNode, B1sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B1BdpPtr, B1BdpBinding, B1bNode, B1dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B1BspPtr, B1BspBinding, B1bNode, B1sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B1DPspPtr, B1DPspBinding, B1dNodePrime, B1sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B1DPdPtr, B1DPdBinding, B1dNodePrime, B1dNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B1BgPtr, B1BgBinding, B1bNode, B1gNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B1DPgPtr, B1DPgBinding, B1dNodePrime, B1gNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B1SPgPtr, B1SPgBinding, B1sNodePrime, B1gNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B1SPsPtr, B1SPsBinding, B1sNodePrime, B1sNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B1DPbPtr, B1DPbBinding, B1dNodePrime, B1bNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B1SPbPtr, B1SPbBinding, B1sNodePrime, B1bNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B1SPdpPtr, B1SPdpBinding, B1sNodePrime, B1dNodePrime);
        }
    }

    return (OK) ;
}

int
B1bindCSCComplexToReal (GENmodel *inModel, CKTcircuit *ckt)
{
    B1model *model = (B1model *)inModel ;
    B1instance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the B1 models */
    for ( ; model != NULL ; model = model->B1nextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->B1instances ; here != NULL ; here = here->B1nextInstance)
        {
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B1DdPtr, B1DdBinding, B1dNode, B1dNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B1GgPtr, B1GgBinding, B1gNode, B1gNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B1SsPtr, B1SsBinding, B1sNode, B1sNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B1BbPtr, B1BbBinding, B1bNode, B1bNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B1DPdpPtr, B1DPdpBinding, B1dNodePrime, B1dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B1SPspPtr, B1SPspBinding, B1sNodePrime, B1sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B1DdpPtr, B1DdpBinding, B1dNode, B1dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B1GbPtr, B1GbBinding, B1gNode, B1bNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B1GdpPtr, B1GdpBinding, B1gNode, B1dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B1GspPtr, B1GspBinding, B1gNode, B1sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B1SspPtr, B1SspBinding, B1sNode, B1sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B1BdpPtr, B1BdpBinding, B1bNode, B1dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B1BspPtr, B1BspBinding, B1bNode, B1sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B1DPspPtr, B1DPspBinding, B1dNodePrime, B1sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B1DPdPtr, B1DPdBinding, B1dNodePrime, B1dNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B1BgPtr, B1BgBinding, B1bNode, B1gNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B1DPgPtr, B1DPgBinding, B1dNodePrime, B1gNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B1SPgPtr, B1SPgBinding, B1sNodePrime, B1gNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B1SPsPtr, B1SPsBinding, B1sNodePrime, B1sNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B1DPbPtr, B1DPbBinding, B1dNodePrime, B1bNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B1SPbPtr, B1SPbBinding, B1sNodePrime, B1bNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B1SPdpPtr, B1SPdpBinding, B1sNodePrime, B1dNodePrime);
        }
    }

    return (OK) ;
}
