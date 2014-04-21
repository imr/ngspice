/**********
Author: 2013 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "mesadefs.h"
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
MESAbindCSC (GENmodel *inModel, CKTcircuit *ckt)
{
    MESAmodel *model = (MESAmodel *)inModel ;
    MESAinstance *here ;
    double *i ;
    BindElement *matched, *BindStruct ;
    size_t nz ;

    BindStruct = ckt->CKTmatrix->CKTbindStruct ;
    nz = (size_t)ckt->CKTmatrix->CKTklunz ;

    /* loop through all the MESA models */
    for ( ; model != NULL ; model = MESAnextModel(model))
    {
        /* loop through all the instances of the model */
        for (here = MESAinstances(model); here != NULL ; here = MESAnextInstance(here))
        {
            CREATE_KLU_BINDING_TABLE(MESAdrainDrainPtr, MESAdrainDrainBinding, MESAdrainNode, MESAdrainNode);
            CREATE_KLU_BINDING_TABLE(MESAdrainPrimeDrainPrimePtr, MESAdrainPrimeDrainPrimeBinding, MESAdrainPrimeNode, MESAdrainPrimeNode);
            CREATE_KLU_BINDING_TABLE(MESAdrainPrmPrmDrainPrmPrmPtr, MESAdrainPrmPrmDrainPrmPrmBinding, MESAdrainPrmPrmNode, MESAdrainPrmPrmNode);
            CREATE_KLU_BINDING_TABLE(MESAgateGatePtr, MESAgateGateBinding, MESAgateNode, MESAgateNode);
            CREATE_KLU_BINDING_TABLE(MESAgatePrimeGatePrimePtr, MESAgatePrimeGatePrimeBinding, MESAgatePrimeNode, MESAgatePrimeNode);
            CREATE_KLU_BINDING_TABLE(MESAsourceSourcePtr, MESAsourceSourceBinding, MESAsourceNode, MESAsourceNode);
            CREATE_KLU_BINDING_TABLE(MESAsourcePrimeSourcePrimePtr, MESAsourcePrimeSourcePrimeBinding, MESAsourcePrimeNode, MESAsourcePrimeNode);
            CREATE_KLU_BINDING_TABLE(MESAsourcePrmPrmSourcePrmPrmPtr, MESAsourcePrmPrmSourcePrmPrmBinding, MESAsourcePrmPrmNode, MESAsourcePrmPrmNode);
            CREATE_KLU_BINDING_TABLE(MESAdrainDrainPrimePtr, MESAdrainDrainPrimeBinding, MESAdrainNode, MESAdrainPrimeNode);
            CREATE_KLU_BINDING_TABLE(MESAdrainPrimeDrainPtr, MESAdrainPrimeDrainBinding, MESAdrainPrimeNode, MESAdrainNode);
            CREATE_KLU_BINDING_TABLE(MESAgatePrimeDrainPrimePtr, MESAgatePrimeDrainPrimeBinding, MESAgatePrimeNode, MESAdrainPrimeNode);
            CREATE_KLU_BINDING_TABLE(MESAdrainPrimeGatePrimePtr, MESAdrainPrimeGatePrimeBinding, MESAdrainPrimeNode, MESAgatePrimeNode);
            CREATE_KLU_BINDING_TABLE(MESAgatePrimeSourcePrimePtr, MESAgatePrimeSourcePrimeBinding, MESAgatePrimeNode, MESAsourcePrimeNode);
            CREATE_KLU_BINDING_TABLE(MESAsourcePrimeGatePrimePtr, MESAsourcePrimeGatePrimeBinding, MESAsourcePrimeNode, MESAgatePrimeNode);
            CREATE_KLU_BINDING_TABLE(MESAsourceSourcePrimePtr, MESAsourceSourcePrimeBinding, MESAsourceNode, MESAsourcePrimeNode);
            CREATE_KLU_BINDING_TABLE(MESAsourcePrimeSourcePtr, MESAsourcePrimeSourceBinding, MESAsourcePrimeNode, MESAsourceNode);
            CREATE_KLU_BINDING_TABLE(MESAdrainPrimeSourcePrimePtr, MESAdrainPrimeSourcePrimeBinding, MESAdrainPrimeNode, MESAsourcePrimeNode);
            CREATE_KLU_BINDING_TABLE(MESAsourcePrimeDrainPrimePtr, MESAsourcePrimeDrainPrimeBinding, MESAsourcePrimeNode, MESAdrainPrimeNode);
            CREATE_KLU_BINDING_TABLE(MESAgatePrimeGatePtr, MESAgatePrimeGateBinding, MESAgatePrimeNode, MESAgateNode);
            CREATE_KLU_BINDING_TABLE(MESAgateGatePrimePtr, MESAgateGatePrimeBinding, MESAgateNode, MESAgatePrimeNode);
            CREATE_KLU_BINDING_TABLE(MESAsourcePrmPrmSourcePrimePtr, MESAsourcePrmPrmSourcePrimeBinding, MESAsourcePrmPrmNode, MESAsourcePrimeNode);
            CREATE_KLU_BINDING_TABLE(MESAsourcePrimeSourcePrmPrmPtr, MESAsourcePrimeSourcePrmPrmBinding, MESAsourcePrimeNode, MESAsourcePrmPrmNode);
            CREATE_KLU_BINDING_TABLE(MESAsourcePrmPrmGatePrimePtr, MESAsourcePrmPrmGatePrimeBinding, MESAsourcePrmPrmNode, MESAgatePrimeNode);
            CREATE_KLU_BINDING_TABLE(MESAgatePrimeSourcePrmPrmPtr, MESAgatePrimeSourcePrmPrmBinding, MESAgatePrimeNode, MESAsourcePrmPrmNode);
            CREATE_KLU_BINDING_TABLE(MESAdrainPrmPrmDrainPrimePtr, MESAdrainPrmPrmDrainPrimeBinding, MESAdrainPrmPrmNode, MESAdrainPrimeNode);
            CREATE_KLU_BINDING_TABLE(MESAdrainPrimeDrainPrmPrmPtr, MESAdrainPrimeDrainPrmPrmBinding, MESAdrainPrimeNode, MESAdrainPrmPrmNode);
            CREATE_KLU_BINDING_TABLE(MESAdrainPrmPrmGatePrimePtr, MESAdrainPrmPrmGatePrimeBinding, MESAdrainPrmPrmNode, MESAgatePrimeNode);
            CREATE_KLU_BINDING_TABLE(MESAgatePrimeDrainPrmPrmPtr, MESAgatePrimeDrainPrmPrmBinding, MESAgatePrimeNode, MESAdrainPrmPrmNode);
        }
    }

    return (OK) ;
}

int
MESAbindCSCComplex (GENmodel *inModel, CKTcircuit *ckt)
{
    MESAmodel *model = (MESAmodel *)inModel ;
    MESAinstance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the MESA models */
    for ( ; model != NULL ; model = MESAnextModel(model))
    {
        /* loop through all the instances of the model */
        for (here = MESAinstances(model); here != NULL ; here = MESAnextInstance(here))
        {
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MESAdrainDrainPtr, MESAdrainDrainBinding, MESAdrainNode, MESAdrainNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MESAdrainPrimeDrainPrimePtr, MESAdrainPrimeDrainPrimeBinding, MESAdrainPrimeNode, MESAdrainPrimeNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MESAdrainPrmPrmDrainPrmPrmPtr, MESAdrainPrmPrmDrainPrmPrmBinding, MESAdrainPrmPrmNode, MESAdrainPrmPrmNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MESAgateGatePtr, MESAgateGateBinding, MESAgateNode, MESAgateNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MESAgatePrimeGatePrimePtr, MESAgatePrimeGatePrimeBinding, MESAgatePrimeNode, MESAgatePrimeNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MESAsourceSourcePtr, MESAsourceSourceBinding, MESAsourceNode, MESAsourceNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MESAsourcePrimeSourcePrimePtr, MESAsourcePrimeSourcePrimeBinding, MESAsourcePrimeNode, MESAsourcePrimeNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MESAsourcePrmPrmSourcePrmPrmPtr, MESAsourcePrmPrmSourcePrmPrmBinding, MESAsourcePrmPrmNode, MESAsourcePrmPrmNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MESAdrainDrainPrimePtr, MESAdrainDrainPrimeBinding, MESAdrainNode, MESAdrainPrimeNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MESAdrainPrimeDrainPtr, MESAdrainPrimeDrainBinding, MESAdrainPrimeNode, MESAdrainNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MESAgatePrimeDrainPrimePtr, MESAgatePrimeDrainPrimeBinding, MESAgatePrimeNode, MESAdrainPrimeNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MESAdrainPrimeGatePrimePtr, MESAdrainPrimeGatePrimeBinding, MESAdrainPrimeNode, MESAgatePrimeNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MESAgatePrimeSourcePrimePtr, MESAgatePrimeSourcePrimeBinding, MESAgatePrimeNode, MESAsourcePrimeNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MESAsourcePrimeGatePrimePtr, MESAsourcePrimeGatePrimeBinding, MESAsourcePrimeNode, MESAgatePrimeNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MESAsourceSourcePrimePtr, MESAsourceSourcePrimeBinding, MESAsourceNode, MESAsourcePrimeNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MESAsourcePrimeSourcePtr, MESAsourcePrimeSourceBinding, MESAsourcePrimeNode, MESAsourceNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MESAdrainPrimeSourcePrimePtr, MESAdrainPrimeSourcePrimeBinding, MESAdrainPrimeNode, MESAsourcePrimeNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MESAsourcePrimeDrainPrimePtr, MESAsourcePrimeDrainPrimeBinding, MESAsourcePrimeNode, MESAdrainPrimeNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MESAgatePrimeGatePtr, MESAgatePrimeGateBinding, MESAgatePrimeNode, MESAgateNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MESAgateGatePrimePtr, MESAgateGatePrimeBinding, MESAgateNode, MESAgatePrimeNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MESAsourcePrmPrmSourcePrimePtr, MESAsourcePrmPrmSourcePrimeBinding, MESAsourcePrmPrmNode, MESAsourcePrimeNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MESAsourcePrimeSourcePrmPrmPtr, MESAsourcePrimeSourcePrmPrmBinding, MESAsourcePrimeNode, MESAsourcePrmPrmNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MESAsourcePrmPrmGatePrimePtr, MESAsourcePrmPrmGatePrimeBinding, MESAsourcePrmPrmNode, MESAgatePrimeNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MESAgatePrimeSourcePrmPrmPtr, MESAgatePrimeSourcePrmPrmBinding, MESAgatePrimeNode, MESAsourcePrmPrmNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MESAdrainPrmPrmDrainPrimePtr, MESAdrainPrmPrmDrainPrimeBinding, MESAdrainPrmPrmNode, MESAdrainPrimeNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MESAdrainPrimeDrainPrmPrmPtr, MESAdrainPrimeDrainPrmPrmBinding, MESAdrainPrimeNode, MESAdrainPrmPrmNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MESAdrainPrmPrmGatePrimePtr, MESAdrainPrmPrmGatePrimeBinding, MESAdrainPrmPrmNode, MESAgatePrimeNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(MESAgatePrimeDrainPrmPrmPtr, MESAgatePrimeDrainPrmPrmBinding, MESAgatePrimeNode, MESAdrainPrmPrmNode);
        }
    }

    return (OK) ;
}

int
MESAbindCSCComplexToReal (GENmodel *inModel, CKTcircuit *ckt)
{
    MESAmodel *model = (MESAmodel *)inModel ;
    MESAinstance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the MESA models */
    for ( ; model != NULL ; model = MESAnextModel(model))
    {
        /* loop through all the instances of the model */
        for (here = MESAinstances(model); here != NULL ; here = MESAnextInstance(here))
        {
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MESAdrainDrainPtr, MESAdrainDrainBinding, MESAdrainNode, MESAdrainNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MESAdrainPrimeDrainPrimePtr, MESAdrainPrimeDrainPrimeBinding, MESAdrainPrimeNode, MESAdrainPrimeNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MESAdrainPrmPrmDrainPrmPrmPtr, MESAdrainPrmPrmDrainPrmPrmBinding, MESAdrainPrmPrmNode, MESAdrainPrmPrmNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MESAgateGatePtr, MESAgateGateBinding, MESAgateNode, MESAgateNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MESAgatePrimeGatePrimePtr, MESAgatePrimeGatePrimeBinding, MESAgatePrimeNode, MESAgatePrimeNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MESAsourceSourcePtr, MESAsourceSourceBinding, MESAsourceNode, MESAsourceNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MESAsourcePrimeSourcePrimePtr, MESAsourcePrimeSourcePrimeBinding, MESAsourcePrimeNode, MESAsourcePrimeNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MESAsourcePrmPrmSourcePrmPrmPtr, MESAsourcePrmPrmSourcePrmPrmBinding, MESAsourcePrmPrmNode, MESAsourcePrmPrmNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MESAdrainDrainPrimePtr, MESAdrainDrainPrimeBinding, MESAdrainNode, MESAdrainPrimeNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MESAdrainPrimeDrainPtr, MESAdrainPrimeDrainBinding, MESAdrainPrimeNode, MESAdrainNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MESAgatePrimeDrainPrimePtr, MESAgatePrimeDrainPrimeBinding, MESAgatePrimeNode, MESAdrainPrimeNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MESAdrainPrimeGatePrimePtr, MESAdrainPrimeGatePrimeBinding, MESAdrainPrimeNode, MESAgatePrimeNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MESAgatePrimeSourcePrimePtr, MESAgatePrimeSourcePrimeBinding, MESAgatePrimeNode, MESAsourcePrimeNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MESAsourcePrimeGatePrimePtr, MESAsourcePrimeGatePrimeBinding, MESAsourcePrimeNode, MESAgatePrimeNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MESAsourceSourcePrimePtr, MESAsourceSourcePrimeBinding, MESAsourceNode, MESAsourcePrimeNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MESAsourcePrimeSourcePtr, MESAsourcePrimeSourceBinding, MESAsourcePrimeNode, MESAsourceNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MESAdrainPrimeSourcePrimePtr, MESAdrainPrimeSourcePrimeBinding, MESAdrainPrimeNode, MESAsourcePrimeNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MESAsourcePrimeDrainPrimePtr, MESAsourcePrimeDrainPrimeBinding, MESAsourcePrimeNode, MESAdrainPrimeNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MESAgatePrimeGatePtr, MESAgatePrimeGateBinding, MESAgatePrimeNode, MESAgateNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MESAgateGatePrimePtr, MESAgateGatePrimeBinding, MESAgateNode, MESAgatePrimeNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MESAsourcePrmPrmSourcePrimePtr, MESAsourcePrmPrmSourcePrimeBinding, MESAsourcePrmPrmNode, MESAsourcePrimeNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MESAsourcePrimeSourcePrmPrmPtr, MESAsourcePrimeSourcePrmPrmBinding, MESAsourcePrimeNode, MESAsourcePrmPrmNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MESAsourcePrmPrmGatePrimePtr, MESAsourcePrmPrmGatePrimeBinding, MESAsourcePrmPrmNode, MESAgatePrimeNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MESAgatePrimeSourcePrmPrmPtr, MESAgatePrimeSourcePrmPrmBinding, MESAgatePrimeNode, MESAsourcePrmPrmNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MESAdrainPrmPrmDrainPrimePtr, MESAdrainPrmPrmDrainPrimeBinding, MESAdrainPrmPrmNode, MESAdrainPrimeNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MESAdrainPrimeDrainPrmPrmPtr, MESAdrainPrimeDrainPrmPrmBinding, MESAdrainPrimeNode, MESAdrainPrmPrmNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MESAdrainPrmPrmGatePrimePtr, MESAdrainPrmPrmGatePrimeBinding, MESAdrainPrmPrmNode, MESAgatePrimeNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(MESAgatePrimeDrainPrmPrmPtr, MESAgatePrimeDrainPrmPrmBinding, MESAgatePrimeNode, MESAdrainPrmPrmNode);
        }
    }

    return (OK) ;
}
