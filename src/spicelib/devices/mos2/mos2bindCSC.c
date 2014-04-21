/**********
Author: 2013 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "mos2defs.h"
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
MOS2bindCSC (GENmodel *inModel, CKTcircuit *ckt)
{
    MOS2model *model = (MOS2model *)inModel ;
    MOS2instance *here ;
    double *i ;
    BindElement *matched, *BindStruct ;
    size_t nz ;

    BindStruct = ckt->CKTmatrix->CKTbindStruct ;
    nz = (size_t)ckt->CKTmatrix->CKTklunz ;

    /* loop through all the MOS2 models */
    for ( ; model != NULL ; model = model->MOS2nextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->MOS2instances ; here != NULL ; here = here->MOS2nextInstance)
        {
            CREATE_KLU_BINDING_TABLE(MOS2DdPtr, MOS2DdBinding, MOS2dNode, MOS2dNode);
            CREATE_KLU_BINDING_TABLE(MOS2GgPtr, MOS2GgBinding, MOS2gNode, MOS2gNode);
            CREATE_KLU_BINDING_TABLE(MOS2SsPtr, MOS2SsBinding, MOS2sNode, MOS2sNode);
            CREATE_KLU_BINDING_TABLE(MOS2BbPtr, MOS2BbBinding, MOS2bNode, MOS2bNode);
            CREATE_KLU_BINDING_TABLE(MOS2DPdpPtr, MOS2DPdpBinding, MOS2dNodePrime, MOS2dNodePrime);
            CREATE_KLU_BINDING_TABLE(MOS2SPspPtr, MOS2SPspBinding, MOS2sNodePrime, MOS2sNodePrime);
            CREATE_KLU_BINDING_TABLE(MOS2DdpPtr, MOS2DdpBinding, MOS2dNode, MOS2dNodePrime);
            CREATE_KLU_BINDING_TABLE(MOS2GbPtr, MOS2GbBinding, MOS2gNode, MOS2bNode);
            CREATE_KLU_BINDING_TABLE(MOS2GdpPtr, MOS2GdpBinding, MOS2gNode, MOS2dNodePrime);
            CREATE_KLU_BINDING_TABLE(MOS2GspPtr, MOS2GspBinding, MOS2gNode, MOS2sNodePrime);
            CREATE_KLU_BINDING_TABLE(MOS2SspPtr, MOS2SspBinding, MOS2sNode, MOS2sNodePrime);
            CREATE_KLU_BINDING_TABLE(MOS2BdpPtr, MOS2BdpBinding, MOS2bNode, MOS2dNodePrime);
            CREATE_KLU_BINDING_TABLE(MOS2BspPtr, MOS2BspBinding, MOS2bNode, MOS2sNodePrime);
            CREATE_KLU_BINDING_TABLE(MOS2DPspPtr, MOS2DPspBinding, MOS2dNodePrime, MOS2sNodePrime);
            CREATE_KLU_BINDING_TABLE(MOS2DPdPtr, MOS2DPdBinding, MOS2dNodePrime, MOS2dNode);
            CREATE_KLU_BINDING_TABLE(MOS2BgPtr, MOS2BgBinding, MOS2bNode, MOS2gNode);
            CREATE_KLU_BINDING_TABLE(MOS2DPgPtr, MOS2DPgBinding, MOS2dNodePrime, MOS2gNode);
            CREATE_KLU_BINDING_TABLE(MOS2SPgPtr, MOS2SPgBinding, MOS2sNodePrime, MOS2gNode);
            CREATE_KLU_BINDING_TABLE(MOS2SPsPtr, MOS2SPsBinding, MOS2sNodePrime, MOS2sNode);
            CREATE_KLU_BINDING_TABLE(MOS2DPbPtr, MOS2DPbBinding, MOS2dNodePrime, MOS2bNode);
            CREATE_KLU_BINDING_TABLE(MOS2SPbPtr, MOS2SPbBinding, MOS2sNodePrime, MOS2bNode);
            CREATE_KLU_BINDING_TABLE(MOS2SPdpPtr, MOS2SPdpBinding, MOS2sNodePrime, MOS2dNodePrime);
        }
    }

    return (OK) ;
}

int
MOS2bindCSCComplex (GENmodel *inModel, CKTcircuit *ckt)
{
    MOS2model *model = (MOS2model *)inModel ;
    MOS2instance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the MOS2 models */
    for ( ; model != NULL ; model = model->MOS2nextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->MOS2instances ; here != NULL ; here = here->MOS2nextInstance)
        {
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MOS2DdPtr, MOS2DdBinding, MOS2dNode, MOS2dNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MOS2GgPtr, MOS2GgBinding, MOS2gNode, MOS2gNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MOS2SsPtr, MOS2SsBinding, MOS2sNode, MOS2sNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MOS2BbPtr, MOS2BbBinding, MOS2bNode, MOS2bNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MOS2DPdpPtr, MOS2DPdpBinding, MOS2dNodePrime, MOS2dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MOS2SPspPtr, MOS2SPspBinding, MOS2sNodePrime, MOS2sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MOS2DdpPtr, MOS2DdpBinding, MOS2dNode, MOS2dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MOS2GbPtr, MOS2GbBinding, MOS2gNode, MOS2bNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MOS2GdpPtr, MOS2GdpBinding, MOS2gNode, MOS2dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MOS2GspPtr, MOS2GspBinding, MOS2gNode, MOS2sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MOS2SspPtr, MOS2SspBinding, MOS2sNode, MOS2sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MOS2BdpPtr, MOS2BdpBinding, MOS2bNode, MOS2dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MOS2BspPtr, MOS2BspBinding, MOS2bNode, MOS2sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MOS2DPspPtr, MOS2DPspBinding, MOS2dNodePrime, MOS2sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MOS2DPdPtr, MOS2DPdBinding, MOS2dNodePrime, MOS2dNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MOS2BgPtr, MOS2BgBinding, MOS2bNode, MOS2gNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MOS2DPgPtr, MOS2DPgBinding, MOS2dNodePrime, MOS2gNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MOS2SPgPtr, MOS2SPgBinding, MOS2sNodePrime, MOS2gNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MOS2SPsPtr, MOS2SPsBinding, MOS2sNodePrime, MOS2sNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MOS2DPbPtr, MOS2DPbBinding, MOS2dNodePrime, MOS2bNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MOS2SPbPtr, MOS2SPbBinding, MOS2sNodePrime, MOS2bNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MOS2SPdpPtr, MOS2SPdpBinding, MOS2sNodePrime, MOS2dNodePrime);
        }
    }

    return (OK) ;
}

int
MOS2bindCSCComplexToReal (GENmodel *inModel, CKTcircuit *ckt)
{
    MOS2model *model = (MOS2model *)inModel ;
    MOS2instance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the MOS2 models */
    for ( ; model != NULL ; model = model->MOS2nextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->MOS2instances ; here != NULL ; here = here->MOS2nextInstance)
        {
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MOS2DdPtr, MOS2DdBinding, MOS2dNode, MOS2dNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MOS2GgPtr, MOS2GgBinding, MOS2gNode, MOS2gNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MOS2SsPtr, MOS2SsBinding, MOS2sNode, MOS2sNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MOS2BbPtr, MOS2BbBinding, MOS2bNode, MOS2bNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MOS2DPdpPtr, MOS2DPdpBinding, MOS2dNodePrime, MOS2dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MOS2SPspPtr, MOS2SPspBinding, MOS2sNodePrime, MOS2sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MOS2DdpPtr, MOS2DdpBinding, MOS2dNode, MOS2dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MOS2GbPtr, MOS2GbBinding, MOS2gNode, MOS2bNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MOS2GdpPtr, MOS2GdpBinding, MOS2gNode, MOS2dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MOS2GspPtr, MOS2GspBinding, MOS2gNode, MOS2sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MOS2SspPtr, MOS2SspBinding, MOS2sNode, MOS2sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MOS2BdpPtr, MOS2BdpBinding, MOS2bNode, MOS2dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MOS2BspPtr, MOS2BspBinding, MOS2bNode, MOS2sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MOS2DPspPtr, MOS2DPspBinding, MOS2dNodePrime, MOS2sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MOS2DPdPtr, MOS2DPdBinding, MOS2dNodePrime, MOS2dNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MOS2BgPtr, MOS2BgBinding, MOS2bNode, MOS2gNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MOS2DPgPtr, MOS2DPgBinding, MOS2dNodePrime, MOS2gNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MOS2SPgPtr, MOS2SPgBinding, MOS2sNodePrime, MOS2gNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MOS2SPsPtr, MOS2SPsBinding, MOS2sNodePrime, MOS2sNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MOS2DPbPtr, MOS2DPbBinding, MOS2dNodePrime, MOS2bNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MOS2SPbPtr, MOS2SPbBinding, MOS2sNodePrime, MOS2bNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MOS2SPdpPtr, MOS2SPdpBinding, MOS2sNodePrime, MOS2dNodePrime);
        }
    }

    return (OK) ;
}
