/**********
Author: 2012 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "mos6defs.h"
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
MOS6bindCSC (GENmodel *inModel, CKTcircuit *ckt)
{
    MOS6model *model = (MOS6model *)inModel ;
    MOS6instance *here ;
    double *i ;
    BindElement *matched, *BindStruct ;
    size_t nz ;

    BindStruct = ckt->CKTmatrix->CKTbindStruct ;
    nz = (size_t)ckt->CKTmatrix->CKTklunz ;

    /* loop through all the MOS6 models */
    for ( ; model != NULL ; model = model->MOS6nextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->MOS6instances ; here != NULL ; here = here->MOS6nextInstance)
        {
            CREATE_KLU_BINDING_TABLE(MOS6DdPtr, MOS6DdBinding, MOS6dNode, MOS6dNode);
            CREATE_KLU_BINDING_TABLE(MOS6GgPtr, MOS6GgBinding, MOS6gNode, MOS6gNode);
            CREATE_KLU_BINDING_TABLE(MOS6SsPtr, MOS6SsBinding, MOS6sNode, MOS6sNode);
            CREATE_KLU_BINDING_TABLE(MOS6BbPtr, MOS6BbBinding, MOS6bNode, MOS6bNode);
            CREATE_KLU_BINDING_TABLE(MOS6DPdpPtr, MOS6DPdpBinding, MOS6dNodePrime, MOS6dNodePrime);
            CREATE_KLU_BINDING_TABLE(MOS6SPspPtr, MOS6SPspBinding, MOS6sNodePrime, MOS6sNodePrime);
            CREATE_KLU_BINDING_TABLE(MOS6DdpPtr, MOS6DdpBinding, MOS6dNode, MOS6dNodePrime);
            CREATE_KLU_BINDING_TABLE(MOS6GbPtr, MOS6GbBinding, MOS6gNode, MOS6bNode);
            CREATE_KLU_BINDING_TABLE(MOS6GdpPtr, MOS6GdpBinding, MOS6gNode, MOS6dNodePrime);
            CREATE_KLU_BINDING_TABLE(MOS6GspPtr, MOS6GspBinding, MOS6gNode, MOS6sNodePrime);
            CREATE_KLU_BINDING_TABLE(MOS6SspPtr, MOS6SspBinding, MOS6sNode, MOS6sNodePrime);
            CREATE_KLU_BINDING_TABLE(MOS6BdpPtr, MOS6BdpBinding, MOS6bNode, MOS6dNodePrime);
            CREATE_KLU_BINDING_TABLE(MOS6BspPtr, MOS6BspBinding, MOS6bNode, MOS6sNodePrime);
            CREATE_KLU_BINDING_TABLE(MOS6DPspPtr, MOS6DPspBinding, MOS6dNodePrime, MOS6sNodePrime);
            CREATE_KLU_BINDING_TABLE(MOS6DPdPtr, MOS6DPdBinding, MOS6dNodePrime, MOS6dNode);
            CREATE_KLU_BINDING_TABLE(MOS6BgPtr, MOS6BgBinding, MOS6bNode, MOS6gNode);
            CREATE_KLU_BINDING_TABLE(MOS6DPgPtr, MOS6DPgBinding, MOS6dNodePrime, MOS6gNode);
            CREATE_KLU_BINDING_TABLE(MOS6SPgPtr, MOS6SPgBinding, MOS6sNodePrime, MOS6gNode);
            CREATE_KLU_BINDING_TABLE(MOS6SPsPtr, MOS6SPsBinding, MOS6sNodePrime, MOS6sNode);
            CREATE_KLU_BINDING_TABLE(MOS6DPbPtr, MOS6DPbBinding, MOS6dNodePrime, MOS6bNode);
            CREATE_KLU_BINDING_TABLE(MOS6SPbPtr, MOS6SPbBinding, MOS6sNodePrime, MOS6bNode);
            CREATE_KLU_BINDING_TABLE(MOS6SPdpPtr, MOS6SPdpBinding, MOS6sNodePrime, MOS6dNodePrime);
        }
    }

    return (OK) ;
}

int
MOS6bindCSCComplex (GENmodel *inModel, CKTcircuit *ckt)
{
    MOS6model *model = (MOS6model *)inModel ;
    MOS6instance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the MOS6 models */
    for ( ; model != NULL ; model = model->MOS6nextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->MOS6instances ; here != NULL ; here = here->MOS6nextInstance)
        {
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MOS6DdPtr, MOS6DdBinding, MOS6dNode, MOS6dNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MOS6GgPtr, MOS6GgBinding, MOS6gNode, MOS6gNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MOS6SsPtr, MOS6SsBinding, MOS6sNode, MOS6sNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MOS6BbPtr, MOS6BbBinding, MOS6bNode, MOS6bNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MOS6DPdpPtr, MOS6DPdpBinding, MOS6dNodePrime, MOS6dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MOS6SPspPtr, MOS6SPspBinding, MOS6sNodePrime, MOS6sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MOS6DdpPtr, MOS6DdpBinding, MOS6dNode, MOS6dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MOS6GbPtr, MOS6GbBinding, MOS6gNode, MOS6bNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MOS6GdpPtr, MOS6GdpBinding, MOS6gNode, MOS6dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MOS6GspPtr, MOS6GspBinding, MOS6gNode, MOS6sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MOS6SspPtr, MOS6SspBinding, MOS6sNode, MOS6sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MOS6BdpPtr, MOS6BdpBinding, MOS6bNode, MOS6dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MOS6BspPtr, MOS6BspBinding, MOS6bNode, MOS6sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MOS6DPspPtr, MOS6DPspBinding, MOS6dNodePrime, MOS6sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MOS6DPdPtr, MOS6DPdBinding, MOS6dNodePrime, MOS6dNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MOS6BgPtr, MOS6BgBinding, MOS6bNode, MOS6gNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MOS6DPgPtr, MOS6DPgBinding, MOS6dNodePrime, MOS6gNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MOS6SPgPtr, MOS6SPgBinding, MOS6sNodePrime, MOS6gNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MOS6SPsPtr, MOS6SPsBinding, MOS6sNodePrime, MOS6sNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MOS6DPbPtr, MOS6DPbBinding, MOS6dNodePrime, MOS6bNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MOS6SPbPtr, MOS6SPbBinding, MOS6sNodePrime, MOS6bNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MOS6SPdpPtr, MOS6SPdpBinding, MOS6sNodePrime, MOS6dNodePrime);
        }
    }

    return (OK) ;
}

int
MOS6bindCSCComplexToReal (GENmodel *inModel, CKTcircuit *ckt)
{
    MOS6model *model = (MOS6model *)inModel ;
    MOS6instance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the MOS6 models */
    for ( ; model != NULL ; model = model->MOS6nextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->MOS6instances ; here != NULL ; here = here->MOS6nextInstance)
        {
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MOS6DdPtr, MOS6DdBinding, MOS6dNode, MOS6dNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MOS6GgPtr, MOS6GgBinding, MOS6gNode, MOS6gNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MOS6SsPtr, MOS6SsBinding, MOS6sNode, MOS6sNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MOS6BbPtr, MOS6BbBinding, MOS6bNode, MOS6bNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MOS6DPdpPtr, MOS6DPdpBinding, MOS6dNodePrime, MOS6dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MOS6SPspPtr, MOS6SPspBinding, MOS6sNodePrime, MOS6sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MOS6DdpPtr, MOS6DdpBinding, MOS6dNode, MOS6dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MOS6GbPtr, MOS6GbBinding, MOS6gNode, MOS6bNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MOS6GdpPtr, MOS6GdpBinding, MOS6gNode, MOS6dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MOS6GspPtr, MOS6GspBinding, MOS6gNode, MOS6sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MOS6SspPtr, MOS6SspBinding, MOS6sNode, MOS6sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MOS6BdpPtr, MOS6BdpBinding, MOS6bNode, MOS6dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MOS6BspPtr, MOS6BspBinding, MOS6bNode, MOS6sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MOS6DPspPtr, MOS6DPspBinding, MOS6dNodePrime, MOS6sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MOS6DPdPtr, MOS6DPdBinding, MOS6dNodePrime, MOS6dNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MOS6BgPtr, MOS6BgBinding, MOS6bNode, MOS6gNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MOS6DPgPtr, MOS6DPgBinding, MOS6dNodePrime, MOS6gNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MOS6SPgPtr, MOS6SPgBinding, MOS6sNodePrime, MOS6gNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MOS6SPsPtr, MOS6SPsBinding, MOS6sNodePrime, MOS6sNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MOS6DPbPtr, MOS6DPbBinding, MOS6dNodePrime, MOS6bNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MOS6SPbPtr, MOS6SPbBinding, MOS6sNodePrime, MOS6bNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MOS6SPdpPtr, MOS6SPdpBinding, MOS6sNodePrime, MOS6dNodePrime);
        }
    }

    return (OK) ;
}
