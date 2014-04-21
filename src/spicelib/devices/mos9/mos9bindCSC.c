/**********
Author: 2013 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "mos9defs.h"
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
MOS9bindCSC (GENmodel *inModel, CKTcircuit *ckt)
{
    MOS9model *model = (MOS9model *)inModel ;
    MOS9instance *here ;
    double *i ;
    BindElement *matched, *BindStruct ;
    size_t nz ;

    BindStruct = ckt->CKTmatrix->CKTbindStruct ;
    nz = (size_t)ckt->CKTmatrix->CKTklunz ;

    /* loop through all the MOS9 models */
    for ( ; model != NULL ; model = model->MOS9nextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->MOS9instances ; here != NULL ; here = here->MOS9nextInstance)
        {
            CREATE_KLU_BINDING_TABLE(MOS9DdPtr, MOS9DdBinding, MOS9dNode, MOS9dNode);
            CREATE_KLU_BINDING_TABLE(MOS9GgPtr, MOS9GgBinding, MOS9gNode, MOS9gNode);
            CREATE_KLU_BINDING_TABLE(MOS9SsPtr, MOS9SsBinding, MOS9sNode, MOS9sNode);
            CREATE_KLU_BINDING_TABLE(MOS9BbPtr, MOS9BbBinding, MOS9bNode, MOS9bNode);
            CREATE_KLU_BINDING_TABLE(MOS9DPdpPtr, MOS9DPdpBinding, MOS9dNodePrime, MOS9dNodePrime);
            CREATE_KLU_BINDING_TABLE(MOS9SPspPtr, MOS9SPspBinding, MOS9sNodePrime, MOS9sNodePrime);
            CREATE_KLU_BINDING_TABLE(MOS9DdpPtr, MOS9DdpBinding, MOS9dNode, MOS9dNodePrime);
            CREATE_KLU_BINDING_TABLE(MOS9GbPtr, MOS9GbBinding, MOS9gNode, MOS9bNode);
            CREATE_KLU_BINDING_TABLE(MOS9GdpPtr, MOS9GdpBinding, MOS9gNode, MOS9dNodePrime);
            CREATE_KLU_BINDING_TABLE(MOS9GspPtr, MOS9GspBinding, MOS9gNode, MOS9sNodePrime);
            CREATE_KLU_BINDING_TABLE(MOS9SspPtr, MOS9SspBinding, MOS9sNode, MOS9sNodePrime);
            CREATE_KLU_BINDING_TABLE(MOS9BdpPtr, MOS9BdpBinding, MOS9bNode, MOS9dNodePrime);
            CREATE_KLU_BINDING_TABLE(MOS9BspPtr, MOS9BspBinding, MOS9bNode, MOS9sNodePrime);
            CREATE_KLU_BINDING_TABLE(MOS9DPspPtr, MOS9DPspBinding, MOS9dNodePrime, MOS9sNodePrime);
            CREATE_KLU_BINDING_TABLE(MOS9DPdPtr, MOS9DPdBinding, MOS9dNodePrime, MOS9dNode);
            CREATE_KLU_BINDING_TABLE(MOS9BgPtr, MOS9BgBinding, MOS9bNode, MOS9gNode);
            CREATE_KLU_BINDING_TABLE(MOS9DPgPtr, MOS9DPgBinding, MOS9dNodePrime, MOS9gNode);
            CREATE_KLU_BINDING_TABLE(MOS9SPgPtr, MOS9SPgBinding, MOS9sNodePrime, MOS9gNode);
            CREATE_KLU_BINDING_TABLE(MOS9SPsPtr, MOS9SPsBinding, MOS9sNodePrime, MOS9sNode);
            CREATE_KLU_BINDING_TABLE(MOS9DPbPtr, MOS9DPbBinding, MOS9dNodePrime, MOS9bNode);
            CREATE_KLU_BINDING_TABLE(MOS9SPbPtr, MOS9SPbBinding, MOS9sNodePrime, MOS9bNode);
            CREATE_KLU_BINDING_TABLE(MOS9SPdpPtr, MOS9SPdpBinding, MOS9sNodePrime, MOS9dNodePrime);
        }
    }

    return (OK) ;
}

int
MOS9bindCSCComplex (GENmodel *inModel, CKTcircuit *ckt)
{
    MOS9model *model = (MOS9model *)inModel ;
    MOS9instance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the MOS9 models */
    for ( ; model != NULL ; model = model->MOS9nextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->MOS9instances ; here != NULL ; here = here->MOS9nextInstance)
        {
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MOS9DdPtr, MOS9DdBinding, MOS9dNode, MOS9dNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MOS9GgPtr, MOS9GgBinding, MOS9gNode, MOS9gNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MOS9SsPtr, MOS9SsBinding, MOS9sNode, MOS9sNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MOS9BbPtr, MOS9BbBinding, MOS9bNode, MOS9bNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MOS9DPdpPtr, MOS9DPdpBinding, MOS9dNodePrime, MOS9dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MOS9SPspPtr, MOS9SPspBinding, MOS9sNodePrime, MOS9sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MOS9DdpPtr, MOS9DdpBinding, MOS9dNode, MOS9dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MOS9GbPtr, MOS9GbBinding, MOS9gNode, MOS9bNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MOS9GdpPtr, MOS9GdpBinding, MOS9gNode, MOS9dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MOS9GspPtr, MOS9GspBinding, MOS9gNode, MOS9sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MOS9SspPtr, MOS9SspBinding, MOS9sNode, MOS9sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MOS9BdpPtr, MOS9BdpBinding, MOS9bNode, MOS9dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MOS9BspPtr, MOS9BspBinding, MOS9bNode, MOS9sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MOS9DPspPtr, MOS9DPspBinding, MOS9dNodePrime, MOS9sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MOS9DPdPtr, MOS9DPdBinding, MOS9dNodePrime, MOS9dNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MOS9BgPtr, MOS9BgBinding, MOS9bNode, MOS9gNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MOS9DPgPtr, MOS9DPgBinding, MOS9dNodePrime, MOS9gNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MOS9SPgPtr, MOS9SPgBinding, MOS9sNodePrime, MOS9gNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MOS9SPsPtr, MOS9SPsBinding, MOS9sNodePrime, MOS9sNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MOS9DPbPtr, MOS9DPbBinding, MOS9dNodePrime, MOS9bNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MOS9SPbPtr, MOS9SPbBinding, MOS9sNodePrime, MOS9bNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MOS9SPdpPtr, MOS9SPdpBinding, MOS9sNodePrime, MOS9dNodePrime);
        }
    }

    return (OK) ;
}

int
MOS9bindCSCComplexToReal (GENmodel *inModel, CKTcircuit *ckt)
{
    MOS9model *model = (MOS9model *)inModel ;
    MOS9instance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the MOS9 models */
    for ( ; model != NULL ; model = model->MOS9nextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->MOS9instances ; here != NULL ; here = here->MOS9nextInstance)
        {
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MOS9DdPtr, MOS9DdBinding, MOS9dNode, MOS9dNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MOS9GgPtr, MOS9GgBinding, MOS9gNode, MOS9gNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MOS9SsPtr, MOS9SsBinding, MOS9sNode, MOS9sNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MOS9BbPtr, MOS9BbBinding, MOS9bNode, MOS9bNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MOS9DPdpPtr, MOS9DPdpBinding, MOS9dNodePrime, MOS9dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MOS9SPspPtr, MOS9SPspBinding, MOS9sNodePrime, MOS9sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MOS9DdpPtr, MOS9DdpBinding, MOS9dNode, MOS9dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MOS9GbPtr, MOS9GbBinding, MOS9gNode, MOS9bNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MOS9GdpPtr, MOS9GdpBinding, MOS9gNode, MOS9dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MOS9GspPtr, MOS9GspBinding, MOS9gNode, MOS9sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MOS9SspPtr, MOS9SspBinding, MOS9sNode, MOS9sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MOS9BdpPtr, MOS9BdpBinding, MOS9bNode, MOS9dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MOS9BspPtr, MOS9BspBinding, MOS9bNode, MOS9sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MOS9DPspPtr, MOS9DPspBinding, MOS9dNodePrime, MOS9sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MOS9DPdPtr, MOS9DPdBinding, MOS9dNodePrime, MOS9dNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MOS9BgPtr, MOS9BgBinding, MOS9bNode, MOS9gNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MOS9DPgPtr, MOS9DPgBinding, MOS9dNodePrime, MOS9gNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MOS9SPgPtr, MOS9SPgBinding, MOS9sNodePrime, MOS9gNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MOS9SPsPtr, MOS9SPsBinding, MOS9sNodePrime, MOS9sNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MOS9DPbPtr, MOS9DPbBinding, MOS9dNodePrime, MOS9bNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MOS9SPbPtr, MOS9SPbBinding, MOS9sNodePrime, MOS9bNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MOS9SPdpPtr, MOS9SPdpBinding, MOS9sNodePrime, MOS9dNodePrime);
        }
    }

    return (OK) ;
}
