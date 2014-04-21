/**********
Author: 2013 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "bsim2def.h"
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
B2bindCSC (GENmodel *inModel, CKTcircuit *ckt)
{
    B2model *model = (B2model *)inModel ;
    B2instance *here ;
    double *i ;
    BindElement *matched, *BindStruct ;
    size_t nz ;

    BindStruct = ckt->CKTmatrix->CKTbindStruct ;
    nz = (size_t)ckt->CKTmatrix->CKTklunz ;

    /* loop through all the B2 models */
    for ( ; model != NULL ; model = B2nextModel(model))
    {
        /* loop through all the instances of the model */
        for (here = B2instances(model); here != NULL ; here = B2nextInstance(here))
        {
            CREATE_KLU_BINDING_TABLE(B2DdPtr, B2DdBinding, B2dNode, B2dNode);
            CREATE_KLU_BINDING_TABLE(B2GgPtr, B2GgBinding, B2gNode, B2gNode);
            CREATE_KLU_BINDING_TABLE(B2SsPtr, B2SsBinding, B2sNode, B2sNode);
            CREATE_KLU_BINDING_TABLE(B2BbPtr, B2BbBinding, B2bNode, B2bNode);
            CREATE_KLU_BINDING_TABLE(B2DPdpPtr, B2DPdpBinding, B2dNodePrime, B2dNodePrime);
            CREATE_KLU_BINDING_TABLE(B2SPspPtr, B2SPspBinding, B2sNodePrime, B2sNodePrime);
            CREATE_KLU_BINDING_TABLE(B2DdpPtr, B2DdpBinding, B2dNode, B2dNodePrime);
            CREATE_KLU_BINDING_TABLE(B2GbPtr, B2GbBinding, B2gNode, B2bNode);
            CREATE_KLU_BINDING_TABLE(B2GdpPtr, B2GdpBinding, B2gNode, B2dNodePrime);
            CREATE_KLU_BINDING_TABLE(B2GspPtr, B2GspBinding, B2gNode, B2sNodePrime);
            CREATE_KLU_BINDING_TABLE(B2SspPtr, B2SspBinding, B2sNode, B2sNodePrime);
            CREATE_KLU_BINDING_TABLE(B2BdpPtr, B2BdpBinding, B2bNode, B2dNodePrime);
            CREATE_KLU_BINDING_TABLE(B2BspPtr, B2BspBinding, B2bNode, B2sNodePrime);
            CREATE_KLU_BINDING_TABLE(B2DPspPtr, B2DPspBinding, B2dNodePrime, B2sNodePrime);
            CREATE_KLU_BINDING_TABLE(B2DPdPtr, B2DPdBinding, B2dNodePrime, B2dNode);
            CREATE_KLU_BINDING_TABLE(B2BgPtr, B2BgBinding, B2bNode, B2gNode);
            CREATE_KLU_BINDING_TABLE(B2DPgPtr, B2DPgBinding, B2dNodePrime, B2gNode);
            CREATE_KLU_BINDING_TABLE(B2SPgPtr, B2SPgBinding, B2sNodePrime, B2gNode);
            CREATE_KLU_BINDING_TABLE(B2SPsPtr, B2SPsBinding, B2sNodePrime, B2sNode);
            CREATE_KLU_BINDING_TABLE(B2DPbPtr, B2DPbBinding, B2dNodePrime, B2bNode);
            CREATE_KLU_BINDING_TABLE(B2SPbPtr, B2SPbBinding, B2sNodePrime, B2bNode);
            CREATE_KLU_BINDING_TABLE(B2SPdpPtr, B2SPdpBinding, B2sNodePrime, B2dNodePrime);
        }
    }

    return (OK) ;
}

int
B2bindCSCComplex (GENmodel *inModel, CKTcircuit *ckt)
{
    B2model *model = (B2model *)inModel ;
    B2instance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the B2 models */
    for ( ; model != NULL ; model = B2nextModel(model))
    {
        /* loop through all the instances of the model */
        for (here = B2instances(model); here != NULL ; here = B2nextInstance(here))
        {
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B2DdPtr, B2DdBinding, B2dNode, B2dNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B2GgPtr, B2GgBinding, B2gNode, B2gNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B2SsPtr, B2SsBinding, B2sNode, B2sNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B2BbPtr, B2BbBinding, B2bNode, B2bNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B2DPdpPtr, B2DPdpBinding, B2dNodePrime, B2dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B2SPspPtr, B2SPspBinding, B2sNodePrime, B2sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B2DdpPtr, B2DdpBinding, B2dNode, B2dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B2GbPtr, B2GbBinding, B2gNode, B2bNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B2GdpPtr, B2GdpBinding, B2gNode, B2dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B2GspPtr, B2GspBinding, B2gNode, B2sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B2SspPtr, B2SspBinding, B2sNode, B2sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B2BdpPtr, B2BdpBinding, B2bNode, B2dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B2BspPtr, B2BspBinding, B2bNode, B2sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B2DPspPtr, B2DPspBinding, B2dNodePrime, B2sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B2DPdPtr, B2DPdBinding, B2dNodePrime, B2dNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B2BgPtr, B2BgBinding, B2bNode, B2gNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B2DPgPtr, B2DPgBinding, B2dNodePrime, B2gNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B2SPgPtr, B2SPgBinding, B2sNodePrime, B2gNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B2SPsPtr, B2SPsBinding, B2sNodePrime, B2sNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B2DPbPtr, B2DPbBinding, B2dNodePrime, B2bNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B2SPbPtr, B2SPbBinding, B2sNodePrime, B2bNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(B2SPdpPtr, B2SPdpBinding, B2sNodePrime, B2dNodePrime);
        }
    }

    return (OK) ;
}

int
B2bindCSCComplexToReal (GENmodel *inModel, CKTcircuit *ckt)
{
    B2model *model = (B2model *)inModel ;
    B2instance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the B2 models */
    for ( ; model != NULL ; model = B2nextModel(model))
    {
        /* loop through all the instances of the model */
        for (here = B2instances(model); here != NULL ; here = B2nextInstance(here))
        {
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B2DdPtr, B2DdBinding, B2dNode, B2dNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B2GgPtr, B2GgBinding, B2gNode, B2gNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B2SsPtr, B2SsBinding, B2sNode, B2sNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B2BbPtr, B2BbBinding, B2bNode, B2bNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B2DPdpPtr, B2DPdpBinding, B2dNodePrime, B2dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B2SPspPtr, B2SPspBinding, B2sNodePrime, B2sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B2DdpPtr, B2DdpBinding, B2dNode, B2dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B2GbPtr, B2GbBinding, B2gNode, B2bNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B2GdpPtr, B2GdpBinding, B2gNode, B2dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B2GspPtr, B2GspBinding, B2gNode, B2sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B2SspPtr, B2SspBinding, B2sNode, B2sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B2BdpPtr, B2BdpBinding, B2bNode, B2dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B2BspPtr, B2BspBinding, B2bNode, B2sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B2DPspPtr, B2DPspBinding, B2dNodePrime, B2sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B2DPdPtr, B2DPdBinding, B2dNodePrime, B2dNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B2BgPtr, B2BgBinding, B2bNode, B2gNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B2DPgPtr, B2DPgBinding, B2dNodePrime, B2gNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B2SPgPtr, B2SPgBinding, B2sNodePrime, B2gNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B2SPsPtr, B2SPsBinding, B2sNodePrime, B2sNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B2DPbPtr, B2DPbBinding, B2dNodePrime, B2bNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B2SPbPtr, B2SPbBinding, B2sNodePrime, B2bNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(B2SPdpPtr, B2SPdpBinding, B2sNodePrime, B2dNodePrime);
        }
    }

    return (OK) ;
}
