/**********
Author: 2013 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "mesdefs.h"
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
MESbindCSC (GENmodel *inModel, CKTcircuit *ckt)
{
    MESmodel *model = (MESmodel *)inModel ;
    MESinstance *here ;
    double *i ;
    BindElement *matched, *BindStruct ;
    size_t nz ;

    BindStruct = ckt->CKTmatrix->CKTbindStruct ;
    nz = (size_t)ckt->CKTmatrix->CKTklunz ;

    /* loop through all the MES models */
    for ( ; model != NULL ; model = model->MESnextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->MESinstances ; here != NULL ; here = here->MESnextInstance)
        {
            CREATE_KLU_BINDING_TABLE(MESdrainDrainPrimePtr, MESdrainDrainPrimeBinding, MESdrainNode, MESdrainPrimeNode);
            CREATE_KLU_BINDING_TABLE(MESgateDrainPrimePtr, MESgateDrainPrimeBinding, MESgateNode, MESdrainPrimeNode);
            CREATE_KLU_BINDING_TABLE(MESgateSourcePrimePtr, MESgateSourcePrimeBinding, MESgateNode, MESsourcePrimeNode);
            CREATE_KLU_BINDING_TABLE(MESsourceSourcePrimePtr, MESsourceSourcePrimeBinding, MESsourceNode, MESsourcePrimeNode);
            CREATE_KLU_BINDING_TABLE(MESdrainPrimeDrainPtr, MESdrainPrimeDrainBinding, MESdrainPrimeNode, MESdrainNode);
            CREATE_KLU_BINDING_TABLE(MESdrainPrimeGatePtr, MESdrainPrimeGateBinding, MESdrainPrimeNode, MESgateNode);
            CREATE_KLU_BINDING_TABLE(MESdrainPrimeSourcePrimePtr, MESdrainPrimeSourcePrimeBinding, MESdrainPrimeNode, MESsourcePrimeNode);
            CREATE_KLU_BINDING_TABLE(MESsourcePrimeGatePtr, MESsourcePrimeGateBinding, MESsourcePrimeNode, MESgateNode);
            CREATE_KLU_BINDING_TABLE(MESsourcePrimeSourcePtr, MESsourcePrimeSourceBinding, MESsourcePrimeNode, MESsourceNode);
            CREATE_KLU_BINDING_TABLE(MESsourcePrimeDrainPrimePtr, MESsourcePrimeDrainPrimeBinding, MESsourcePrimeNode, MESdrainPrimeNode);
            CREATE_KLU_BINDING_TABLE(MESdrainDrainPtr, MESdrainDrainBinding, MESdrainNode, MESdrainNode);
            CREATE_KLU_BINDING_TABLE(MESgateGatePtr, MESgateGateBinding, MESgateNode, MESgateNode);
            CREATE_KLU_BINDING_TABLE(MESsourceSourcePtr, MESsourceSourceBinding, MESsourceNode, MESsourceNode);
            CREATE_KLU_BINDING_TABLE(MESdrainPrimeDrainPrimePtr, MESdrainPrimeDrainPrimeBinding, MESdrainPrimeNode, MESdrainPrimeNode);
            CREATE_KLU_BINDING_TABLE(MESsourcePrimeSourcePrimePtr, MESsourcePrimeSourcePrimeBinding, MESsourcePrimeNode, MESsourcePrimeNode);
        }
    }

    return (OK) ;
}

int
MESbindCSCComplex (GENmodel *inModel, CKTcircuit *ckt)
{
    MESmodel *model = (MESmodel *)inModel ;
    MESinstance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the MES models */
    for ( ; model != NULL ; model = model->MESnextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->MESinstances ; here != NULL ; here = here->MESnextInstance)
        {
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MESdrainDrainPrimePtr, MESdrainDrainPrimeBinding, MESdrainNode, MESdrainPrimeNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MESgateDrainPrimePtr, MESgateDrainPrimeBinding, MESgateNode, MESdrainPrimeNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MESgateSourcePrimePtr, MESgateSourcePrimeBinding, MESgateNode, MESsourcePrimeNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MESsourceSourcePrimePtr, MESsourceSourcePrimeBinding, MESsourceNode, MESsourcePrimeNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MESdrainPrimeDrainPtr, MESdrainPrimeDrainBinding, MESdrainPrimeNode, MESdrainNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MESdrainPrimeGatePtr, MESdrainPrimeGateBinding, MESdrainPrimeNode, MESgateNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MESdrainPrimeSourcePrimePtr, MESdrainPrimeSourcePrimeBinding, MESdrainPrimeNode, MESsourcePrimeNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MESsourcePrimeGatePtr, MESsourcePrimeGateBinding, MESsourcePrimeNode, MESgateNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MESsourcePrimeSourcePtr, MESsourcePrimeSourceBinding, MESsourcePrimeNode, MESsourceNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MESsourcePrimeDrainPrimePtr, MESsourcePrimeDrainPrimeBinding, MESsourcePrimeNode, MESdrainPrimeNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MESdrainDrainPtr, MESdrainDrainBinding, MESdrainNode, MESdrainNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MESgateGatePtr, MESgateGateBinding, MESgateNode, MESgateNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MESsourceSourcePtr, MESsourceSourceBinding, MESsourceNode, MESsourceNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MESdrainPrimeDrainPrimePtr, MESdrainPrimeDrainPrimeBinding, MESdrainPrimeNode, MESdrainPrimeNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MESsourcePrimeSourcePrimePtr, MESsourcePrimeSourcePrimeBinding, MESsourcePrimeNode, MESsourcePrimeNode);
        }
    }

    return (OK) ;
}

int
MESbindCSCComplexToReal (GENmodel *inModel, CKTcircuit *ckt)
{
    MESmodel *model = (MESmodel *)inModel ;
    MESinstance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the MES models */
    for ( ; model != NULL ; model = model->MESnextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->MESinstances ; here != NULL ; here = here->MESnextInstance)
        {
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MESdrainDrainPrimePtr, MESdrainDrainPrimeBinding, MESdrainNode, MESdrainPrimeNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MESgateDrainPrimePtr, MESgateDrainPrimeBinding, MESgateNode, MESdrainPrimeNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MESgateSourcePrimePtr, MESgateSourcePrimeBinding, MESgateNode, MESsourcePrimeNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MESsourceSourcePrimePtr, MESsourceSourcePrimeBinding, MESsourceNode, MESsourcePrimeNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MESdrainPrimeDrainPtr, MESdrainPrimeDrainBinding, MESdrainPrimeNode, MESdrainNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MESdrainPrimeGatePtr, MESdrainPrimeGateBinding, MESdrainPrimeNode, MESgateNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MESdrainPrimeSourcePrimePtr, MESdrainPrimeSourcePrimeBinding, MESdrainPrimeNode, MESsourcePrimeNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MESsourcePrimeGatePtr, MESsourcePrimeGateBinding, MESsourcePrimeNode, MESgateNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MESsourcePrimeSourcePtr, MESsourcePrimeSourceBinding, MESsourcePrimeNode, MESsourceNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MESsourcePrimeDrainPrimePtr, MESsourcePrimeDrainPrimeBinding, MESsourcePrimeNode, MESdrainPrimeNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MESdrainDrainPtr, MESdrainDrainBinding, MESdrainNode, MESdrainNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MESgateGatePtr, MESgateGateBinding, MESgateNode, MESgateNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MESsourceSourcePtr, MESsourceSourceBinding, MESsourceNode, MESsourceNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MESdrainPrimeDrainPrimePtr, MESdrainPrimeDrainPrimeBinding, MESdrainPrimeNode, MESdrainPrimeNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MESsourcePrimeSourcePrimePtr, MESsourcePrimeSourcePrimeBinding, MESsourcePrimeNode, MESsourcePrimeNode);
        }
    }

    return (OK) ;
}
