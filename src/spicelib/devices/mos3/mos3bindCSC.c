/**********
Author: 2013 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "mos3defs.h"
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
MOS3bindCSC (GENmodel *inModel, CKTcircuit *ckt)
{
    MOS3model *model = (MOS3model *)inModel ;
    MOS3instance *here ;
    double *i ;
    BindElement *matched, *BindStruct ;
    size_t nz ;

    BindStruct = ckt->CKTmatrix->CKTbindStruct ;
    nz = (size_t)ckt->CKTmatrix->CKTklunz ;

    /* loop through all the MOS3 models */
    for ( ; model != NULL ; model = MOS3nextModel(model))
    {
        /* loop through all the instances of the model */
        for (here = MOS3instances(model); here != NULL ; here = MOS3nextInstance(here))
        {
            CREATE_KLU_BINDING_TABLE(MOS3DdPtr, MOS3DdBinding, MOS3dNode, MOS3dNode);
            CREATE_KLU_BINDING_TABLE(MOS3GgPtr, MOS3GgBinding, MOS3gNode, MOS3gNode);
            CREATE_KLU_BINDING_TABLE(MOS3SsPtr, MOS3SsBinding, MOS3sNode, MOS3sNode);
            CREATE_KLU_BINDING_TABLE(MOS3BbPtr, MOS3BbBinding, MOS3bNode, MOS3bNode);
            CREATE_KLU_BINDING_TABLE(MOS3DPdpPtr, MOS3DPdpBinding, MOS3dNodePrime, MOS3dNodePrime);
            CREATE_KLU_BINDING_TABLE(MOS3SPspPtr, MOS3SPspBinding, MOS3sNodePrime, MOS3sNodePrime);
            CREATE_KLU_BINDING_TABLE(MOS3DdpPtr, MOS3DdpBinding, MOS3dNode, MOS3dNodePrime);
            CREATE_KLU_BINDING_TABLE(MOS3GbPtr, MOS3GbBinding, MOS3gNode, MOS3bNode);
            CREATE_KLU_BINDING_TABLE(MOS3GdpPtr, MOS3GdpBinding, MOS3gNode, MOS3dNodePrime);
            CREATE_KLU_BINDING_TABLE(MOS3GspPtr, MOS3GspBinding, MOS3gNode, MOS3sNodePrime);
            CREATE_KLU_BINDING_TABLE(MOS3SspPtr, MOS3SspBinding, MOS3sNode, MOS3sNodePrime);
            CREATE_KLU_BINDING_TABLE(MOS3BdpPtr, MOS3BdpBinding, MOS3bNode, MOS3dNodePrime);
            CREATE_KLU_BINDING_TABLE(MOS3BspPtr, MOS3BspBinding, MOS3bNode, MOS3sNodePrime);
            CREATE_KLU_BINDING_TABLE(MOS3DPspPtr, MOS3DPspBinding, MOS3dNodePrime, MOS3sNodePrime);
            CREATE_KLU_BINDING_TABLE(MOS3DPdPtr, MOS3DPdBinding, MOS3dNodePrime, MOS3dNode);
            CREATE_KLU_BINDING_TABLE(MOS3BgPtr, MOS3BgBinding, MOS3bNode, MOS3gNode);
            CREATE_KLU_BINDING_TABLE(MOS3DPgPtr, MOS3DPgBinding, MOS3dNodePrime, MOS3gNode);
            CREATE_KLU_BINDING_TABLE(MOS3SPgPtr, MOS3SPgBinding, MOS3sNodePrime, MOS3gNode);
            CREATE_KLU_BINDING_TABLE(MOS3SPsPtr, MOS3SPsBinding, MOS3sNodePrime, MOS3sNode);
            CREATE_KLU_BINDING_TABLE(MOS3DPbPtr, MOS3DPbBinding, MOS3dNodePrime, MOS3bNode);
            CREATE_KLU_BINDING_TABLE(MOS3SPbPtr, MOS3SPbBinding, MOS3sNodePrime, MOS3bNode);
            CREATE_KLU_BINDING_TABLE(MOS3SPdpPtr, MOS3SPdpBinding, MOS3sNodePrime, MOS3dNodePrime);
        }
    }

    return (OK) ;
}

int
MOS3bindCSCComplex (GENmodel *inModel, CKTcircuit *ckt)
{
    MOS3model *model = (MOS3model *)inModel ;
    MOS3instance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the MOS3 models */
    for ( ; model != NULL ; model = MOS3nextModel(model))
    {
        /* loop through all the instances of the model */
        for (here = MOS3instances(model); here != NULL ; here = MOS3nextInstance(here))
        {
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MOS3DdPtr, MOS3DdBinding, MOS3dNode, MOS3dNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MOS3GgPtr, MOS3GgBinding, MOS3gNode, MOS3gNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MOS3SsPtr, MOS3SsBinding, MOS3sNode, MOS3sNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MOS3BbPtr, MOS3BbBinding, MOS3bNode, MOS3bNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MOS3DPdpPtr, MOS3DPdpBinding, MOS3dNodePrime, MOS3dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MOS3SPspPtr, MOS3SPspBinding, MOS3sNodePrime, MOS3sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MOS3DdpPtr, MOS3DdpBinding, MOS3dNode, MOS3dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MOS3GbPtr, MOS3GbBinding, MOS3gNode, MOS3bNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MOS3GdpPtr, MOS3GdpBinding, MOS3gNode, MOS3dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MOS3GspPtr, MOS3GspBinding, MOS3gNode, MOS3sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MOS3SspPtr, MOS3SspBinding, MOS3sNode, MOS3sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MOS3BdpPtr, MOS3BdpBinding, MOS3bNode, MOS3dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MOS3BspPtr, MOS3BspBinding, MOS3bNode, MOS3sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MOS3DPspPtr, MOS3DPspBinding, MOS3dNodePrime, MOS3sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MOS3DPdPtr, MOS3DPdBinding, MOS3dNodePrime, MOS3dNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MOS3BgPtr, MOS3BgBinding, MOS3bNode, MOS3gNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MOS3DPgPtr, MOS3DPgBinding, MOS3dNodePrime, MOS3gNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MOS3SPgPtr, MOS3SPgBinding, MOS3sNodePrime, MOS3gNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MOS3SPsPtr, MOS3SPsBinding, MOS3sNodePrime, MOS3sNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MOS3DPbPtr, MOS3DPbBinding, MOS3dNodePrime, MOS3bNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MOS3SPbPtr, MOS3SPbBinding, MOS3sNodePrime, MOS3bNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MOS3SPdpPtr, MOS3SPdpBinding, MOS3sNodePrime, MOS3dNodePrime);
        }
    }

    return (OK) ;
}

int
MOS3bindCSCComplexToReal (GENmodel *inModel, CKTcircuit *ckt)
{
    MOS3model *model = (MOS3model *)inModel ;
    MOS3instance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the MOS3 models */
    for ( ; model != NULL ; model = MOS3nextModel(model))
    {
        /* loop through all the instances of the model */
        for (here = MOS3instances(model); here != NULL ; here = MOS3nextInstance(here))
        {
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MOS3DdPtr, MOS3DdBinding, MOS3dNode, MOS3dNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MOS3GgPtr, MOS3GgBinding, MOS3gNode, MOS3gNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MOS3SsPtr, MOS3SsBinding, MOS3sNode, MOS3sNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MOS3BbPtr, MOS3BbBinding, MOS3bNode, MOS3bNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MOS3DPdpPtr, MOS3DPdpBinding, MOS3dNodePrime, MOS3dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MOS3SPspPtr, MOS3SPspBinding, MOS3sNodePrime, MOS3sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MOS3DdpPtr, MOS3DdpBinding, MOS3dNode, MOS3dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MOS3GbPtr, MOS3GbBinding, MOS3gNode, MOS3bNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MOS3GdpPtr, MOS3GdpBinding, MOS3gNode, MOS3dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MOS3GspPtr, MOS3GspBinding, MOS3gNode, MOS3sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MOS3SspPtr, MOS3SspBinding, MOS3sNode, MOS3sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MOS3BdpPtr, MOS3BdpBinding, MOS3bNode, MOS3dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MOS3BspPtr, MOS3BspBinding, MOS3bNode, MOS3sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MOS3DPspPtr, MOS3DPspBinding, MOS3dNodePrime, MOS3sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MOS3DPdPtr, MOS3DPdBinding, MOS3dNodePrime, MOS3dNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MOS3BgPtr, MOS3BgBinding, MOS3bNode, MOS3gNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MOS3DPgPtr, MOS3DPgBinding, MOS3dNodePrime, MOS3gNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MOS3SPgPtr, MOS3SPgBinding, MOS3sNodePrime, MOS3gNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MOS3SPsPtr, MOS3SPsBinding, MOS3sNodePrime, MOS3sNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MOS3DPbPtr, MOS3DPbBinding, MOS3dNodePrime, MOS3bNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MOS3SPbPtr, MOS3SPbBinding, MOS3sNodePrime, MOS3bNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MOS3SPdpPtr, MOS3SPdpBinding, MOS3sNodePrime, MOS3dNodePrime);
        }
    }

    return (OK) ;
}
