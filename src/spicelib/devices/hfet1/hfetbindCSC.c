/**********
Author: 2013 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "hfetdefs.h"
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
HFETAbindCSC (GENmodel *inModel, CKTcircuit *ckt)
{
    HFETAmodel *model = (HFETAmodel *)inModel ;
    HFETAinstance *here ;
    double *i ;
    BindElement *matched, *BindStruct ;
    size_t nz ;

    BindStruct = ckt->CKTmatrix->CKTbindStruct ;
    nz = (size_t)ckt->CKTmatrix->CKTklunz ;

    /* loop through all the HFETA models */
    for ( ; model != NULL ; model = HFETAnextModel(model))
    {
        /* loop through all the instances of the model */
        for (here = HFETAinstances(model); here != NULL ; here = HFETAnextInstance(here))
        {
            CREATE_KLU_BINDING_TABLE(HFETAdrainDrainPrimePtr, HFETAdrainDrainPrimeBinding, HFETAdrainNode, HFETAdrainPrimeNode);
            CREATE_KLU_BINDING_TABLE(HFETAgatePrimeDrainPrimePtr, HFETAgatePrimeDrainPrimeBinding, HFETAgatePrimeNode, HFETAdrainPrimeNode);
            CREATE_KLU_BINDING_TABLE(HFETAgatePrimeSourcePrimePtr, HFETAgatePrimeSourcePrimeBinding, HFETAgatePrimeNode, HFETAsourcePrimeNode);
            CREATE_KLU_BINDING_TABLE(HFETAsourceSourcePrimePtr, HFETAsourceSourcePrimeBinding, HFETAsourceNode, HFETAsourcePrimeNode);
            CREATE_KLU_BINDING_TABLE(HFETAdrainPrimeDrainPtr, HFETAdrainPrimeDrainBinding, HFETAdrainPrimeNode, HFETAdrainNode);
            CREATE_KLU_BINDING_TABLE(HFETAdrainPrimeGatePrimePtr, HFETAdrainPrimeGatePrimeBinding, HFETAdrainPrimeNode, HFETAgatePrimeNode);
            CREATE_KLU_BINDING_TABLE(HFETAdrainPrimeSourcePrimePtr, HFETAdrainPrimeSourcePrimeBinding, HFETAdrainPrimeNode, HFETAsourcePrimeNode);
            CREATE_KLU_BINDING_TABLE(HFETAsourcePrimeGatePrimePtr, HFETAsourcePrimeGatePrimeBinding, HFETAsourcePrimeNode, HFETAgatePrimeNode);
            CREATE_KLU_BINDING_TABLE(HFETAsourcePrimeSourcePtr, HFETAsourcePrimeSourceBinding, HFETAsourcePrimeNode, HFETAsourceNode);
            CREATE_KLU_BINDING_TABLE(HFETAsourcePrimeDrainPrimePtr, HFETAsourcePrimeDrainPrimeBinding, HFETAsourcePrimeNode, HFETAdrainPrimeNode);
            CREATE_KLU_BINDING_TABLE(HFETAdrainDrainPtr, HFETAdrainDrainBinding, HFETAdrainNode, HFETAdrainNode);
            CREATE_KLU_BINDING_TABLE(HFETAgatePrimeGatePrimePtr, HFETAgatePrimeGatePrimeBinding, HFETAgatePrimeNode, HFETAgatePrimeNode);
            CREATE_KLU_BINDING_TABLE(HFETAsourceSourcePtr, HFETAsourceSourceBinding, HFETAsourceNode, HFETAsourceNode);
            CREATE_KLU_BINDING_TABLE(HFETAdrainPrimeDrainPrimePtr, HFETAdrainPrimeDrainPrimeBinding, HFETAdrainPrimeNode, HFETAdrainPrimeNode);
            CREATE_KLU_BINDING_TABLE(HFETAsourcePrimeSourcePrimePtr, HFETAsourcePrimeSourcePrimeBinding, HFETAsourcePrimeNode, HFETAsourcePrimeNode);
            CREATE_KLU_BINDING_TABLE(HFETAdrainPrimeDrainPrmPrmPtr, HFETAdrainPrimeDrainPrmPrmBinding, HFETAdrainPrimeNode, HFETAdrainPrmPrmNode);
            CREATE_KLU_BINDING_TABLE(HFETAdrainPrmPrmDrainPrimePtr, HFETAdrainPrmPrmDrainPrimeBinding, HFETAdrainPrmPrmNode, HFETAdrainPrimeNode);
            CREATE_KLU_BINDING_TABLE(HFETAdrainPrmPrmGatePrimePtr, HFETAdrainPrmPrmGatePrimeBinding, HFETAdrainPrmPrmNode, HFETAgatePrimeNode);
            CREATE_KLU_BINDING_TABLE(HFETAgatePrimeDrainPrmPrmPtr, HFETAgatePrimeDrainPrmPrmBinding, HFETAgatePrimeNode, HFETAdrainPrmPrmNode);
            CREATE_KLU_BINDING_TABLE(HFETAdrainPrmPrmDrainPrmPrmPtr, HFETAdrainPrmPrmDrainPrmPrmBinding, HFETAdrainPrmPrmNode, HFETAdrainPrmPrmNode);
            CREATE_KLU_BINDING_TABLE(HFETAsourcePrimeSourcePrmPrmPtr, HFETAsourcePrimeSourcePrmPrmBinding, HFETAsourcePrimeNode, HFETAsourcePrmPrmNode);
            CREATE_KLU_BINDING_TABLE(HFETAsourcePrmPrmSourcePrimePtr, HFETAsourcePrmPrmSourcePrimeBinding, HFETAsourcePrmPrmNode, HFETAsourcePrimeNode);
            CREATE_KLU_BINDING_TABLE(HFETAsourcePrmPrmGatePrimePtr, HFETAsourcePrmPrmGatePrimeBinding, HFETAsourcePrmPrmNode, HFETAgatePrimeNode);
            CREATE_KLU_BINDING_TABLE(HFETAgatePrimeSourcePrmPrmPtr, HFETAgatePrimeSourcePrmPrmBinding, HFETAgatePrimeNode, HFETAsourcePrmPrmNode);
            CREATE_KLU_BINDING_TABLE(HFETAsourcePrmPrmSourcePrmPrmPtr, HFETAsourcePrmPrmSourcePrmPrmBinding, HFETAsourcePrmPrmNode, HFETAsourcePrmPrmNode);
            CREATE_KLU_BINDING_TABLE(HFETAgateGatePtr, HFETAgateGateBinding, HFETAgateNode, HFETAgateNode);
            CREATE_KLU_BINDING_TABLE(HFETAgateGatePrimePtr, HFETAgateGatePrimeBinding, HFETAgateNode, HFETAgatePrimeNode);
            CREATE_KLU_BINDING_TABLE(HFETAgatePrimeGatePtr, HFETAgatePrimeGateBinding, HFETAgatePrimeNode, HFETAgateNode);
        }
    }

    return (OK) ;
}

int
HFETAbindCSCComplex (GENmodel *inModel, CKTcircuit *ckt)
{
    HFETAmodel *model = (HFETAmodel *)inModel ;
    HFETAinstance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the HFETA models */
    for ( ; model != NULL ; model = HFETAnextModel(model))
    {
        /* loop through all the instances of the model */
        for (here = HFETAinstances(model); here != NULL ; here = HFETAnextInstance(here))
        {
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HFETAdrainDrainPrimePtr, HFETAdrainDrainPrimeBinding, HFETAdrainNode, HFETAdrainPrimeNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HFETAgatePrimeDrainPrimePtr, HFETAgatePrimeDrainPrimeBinding, HFETAgatePrimeNode, HFETAdrainPrimeNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HFETAgatePrimeSourcePrimePtr, HFETAgatePrimeSourcePrimeBinding, HFETAgatePrimeNode, HFETAsourcePrimeNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HFETAsourceSourcePrimePtr, HFETAsourceSourcePrimeBinding, HFETAsourceNode, HFETAsourcePrimeNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HFETAdrainPrimeDrainPtr, HFETAdrainPrimeDrainBinding, HFETAdrainPrimeNode, HFETAdrainNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HFETAdrainPrimeGatePrimePtr, HFETAdrainPrimeGatePrimeBinding, HFETAdrainPrimeNode, HFETAgatePrimeNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HFETAdrainPrimeSourcePrimePtr, HFETAdrainPrimeSourcePrimeBinding, HFETAdrainPrimeNode, HFETAsourcePrimeNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HFETAsourcePrimeGatePrimePtr, HFETAsourcePrimeGatePrimeBinding, HFETAsourcePrimeNode, HFETAgatePrimeNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HFETAsourcePrimeSourcePtr, HFETAsourcePrimeSourceBinding, HFETAsourcePrimeNode, HFETAsourceNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HFETAsourcePrimeDrainPrimePtr, HFETAsourcePrimeDrainPrimeBinding, HFETAsourcePrimeNode, HFETAdrainPrimeNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HFETAdrainDrainPtr, HFETAdrainDrainBinding, HFETAdrainNode, HFETAdrainNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HFETAgatePrimeGatePrimePtr, HFETAgatePrimeGatePrimeBinding, HFETAgatePrimeNode, HFETAgatePrimeNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HFETAsourceSourcePtr, HFETAsourceSourceBinding, HFETAsourceNode, HFETAsourceNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HFETAdrainPrimeDrainPrimePtr, HFETAdrainPrimeDrainPrimeBinding, HFETAdrainPrimeNode, HFETAdrainPrimeNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HFETAsourcePrimeSourcePrimePtr, HFETAsourcePrimeSourcePrimeBinding, HFETAsourcePrimeNode, HFETAsourcePrimeNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HFETAdrainPrimeDrainPrmPrmPtr, HFETAdrainPrimeDrainPrmPrmBinding, HFETAdrainPrimeNode, HFETAdrainPrmPrmNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HFETAdrainPrmPrmDrainPrimePtr, HFETAdrainPrmPrmDrainPrimeBinding, HFETAdrainPrmPrmNode, HFETAdrainPrimeNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HFETAdrainPrmPrmGatePrimePtr, HFETAdrainPrmPrmGatePrimeBinding, HFETAdrainPrmPrmNode, HFETAgatePrimeNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HFETAgatePrimeDrainPrmPrmPtr, HFETAgatePrimeDrainPrmPrmBinding, HFETAgatePrimeNode, HFETAdrainPrmPrmNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HFETAdrainPrmPrmDrainPrmPrmPtr, HFETAdrainPrmPrmDrainPrmPrmBinding, HFETAdrainPrmPrmNode, HFETAdrainPrmPrmNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HFETAsourcePrimeSourcePrmPrmPtr, HFETAsourcePrimeSourcePrmPrmBinding, HFETAsourcePrimeNode, HFETAsourcePrmPrmNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HFETAsourcePrmPrmSourcePrimePtr, HFETAsourcePrmPrmSourcePrimeBinding, HFETAsourcePrmPrmNode, HFETAsourcePrimeNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HFETAsourcePrmPrmGatePrimePtr, HFETAsourcePrmPrmGatePrimeBinding, HFETAsourcePrmPrmNode, HFETAgatePrimeNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HFETAgatePrimeSourcePrmPrmPtr, HFETAgatePrimeSourcePrmPrmBinding, HFETAgatePrimeNode, HFETAsourcePrmPrmNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HFETAsourcePrmPrmSourcePrmPrmPtr, HFETAsourcePrmPrmSourcePrmPrmBinding, HFETAsourcePrmPrmNode, HFETAsourcePrmPrmNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HFETAgateGatePtr, HFETAgateGateBinding, HFETAgateNode, HFETAgateNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HFETAgateGatePrimePtr, HFETAgateGatePrimeBinding, HFETAgateNode, HFETAgatePrimeNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HFETAgatePrimeGatePtr, HFETAgatePrimeGateBinding, HFETAgatePrimeNode, HFETAgateNode);
        }
    }

    return (OK) ;
}

int
HFETAbindCSCComplexToReal (GENmodel *inModel, CKTcircuit *ckt)
{
    HFETAmodel *model = (HFETAmodel *)inModel ;
    HFETAinstance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the HFETA models */
    for ( ; model != NULL ; model = HFETAnextModel(model))
    {
        /* loop through all the instances of the model */
        for (here = HFETAinstances(model); here != NULL ; here = HFETAnextInstance(here))
        {
            CONVERT_KLU_BINDING_TABLE_TO_REAL(HFETAdrainDrainPrimePtr, HFETAdrainDrainPrimeBinding, HFETAdrainNode, HFETAdrainPrimeNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(HFETAgatePrimeDrainPrimePtr, HFETAgatePrimeDrainPrimeBinding, HFETAgatePrimeNode, HFETAdrainPrimeNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(HFETAgatePrimeSourcePrimePtr, HFETAgatePrimeSourcePrimeBinding, HFETAgatePrimeNode, HFETAsourcePrimeNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(HFETAsourceSourcePrimePtr, HFETAsourceSourcePrimeBinding, HFETAsourceNode, HFETAsourcePrimeNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(HFETAdrainPrimeDrainPtr, HFETAdrainPrimeDrainBinding, HFETAdrainPrimeNode, HFETAdrainNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(HFETAdrainPrimeGatePrimePtr, HFETAdrainPrimeGatePrimeBinding, HFETAdrainPrimeNode, HFETAgatePrimeNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(HFETAdrainPrimeSourcePrimePtr, HFETAdrainPrimeSourcePrimeBinding, HFETAdrainPrimeNode, HFETAsourcePrimeNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(HFETAsourcePrimeGatePrimePtr, HFETAsourcePrimeGatePrimeBinding, HFETAsourcePrimeNode, HFETAgatePrimeNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(HFETAsourcePrimeSourcePtr, HFETAsourcePrimeSourceBinding, HFETAsourcePrimeNode, HFETAsourceNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(HFETAsourcePrimeDrainPrimePtr, HFETAsourcePrimeDrainPrimeBinding, HFETAsourcePrimeNode, HFETAdrainPrimeNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(HFETAdrainDrainPtr, HFETAdrainDrainBinding, HFETAdrainNode, HFETAdrainNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(HFETAgatePrimeGatePrimePtr, HFETAgatePrimeGatePrimeBinding, HFETAgatePrimeNode, HFETAgatePrimeNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(HFETAsourceSourcePtr, HFETAsourceSourceBinding, HFETAsourceNode, HFETAsourceNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(HFETAdrainPrimeDrainPrimePtr, HFETAdrainPrimeDrainPrimeBinding, HFETAdrainPrimeNode, HFETAdrainPrimeNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(HFETAsourcePrimeSourcePrimePtr, HFETAsourcePrimeSourcePrimeBinding, HFETAsourcePrimeNode, HFETAsourcePrimeNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(HFETAdrainPrimeDrainPrmPrmPtr, HFETAdrainPrimeDrainPrmPrmBinding, HFETAdrainPrimeNode, HFETAdrainPrmPrmNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(HFETAdrainPrmPrmDrainPrimePtr, HFETAdrainPrmPrmDrainPrimeBinding, HFETAdrainPrmPrmNode, HFETAdrainPrimeNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(HFETAdrainPrmPrmGatePrimePtr, HFETAdrainPrmPrmGatePrimeBinding, HFETAdrainPrmPrmNode, HFETAgatePrimeNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(HFETAgatePrimeDrainPrmPrmPtr, HFETAgatePrimeDrainPrmPrmBinding, HFETAgatePrimeNode, HFETAdrainPrmPrmNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(HFETAdrainPrmPrmDrainPrmPrmPtr, HFETAdrainPrmPrmDrainPrmPrmBinding, HFETAdrainPrmPrmNode, HFETAdrainPrmPrmNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(HFETAsourcePrimeSourcePrmPrmPtr, HFETAsourcePrimeSourcePrmPrmBinding, HFETAsourcePrimeNode, HFETAsourcePrmPrmNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(HFETAsourcePrmPrmSourcePrimePtr, HFETAsourcePrmPrmSourcePrimeBinding, HFETAsourcePrmPrmNode, HFETAsourcePrimeNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(HFETAsourcePrmPrmGatePrimePtr, HFETAsourcePrmPrmGatePrimeBinding, HFETAsourcePrmPrmNode, HFETAgatePrimeNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(HFETAgatePrimeSourcePrmPrmPtr, HFETAgatePrimeSourcePrmPrmBinding, HFETAgatePrimeNode, HFETAsourcePrmPrmNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(HFETAsourcePrmPrmSourcePrmPrmPtr, HFETAsourcePrmPrmSourcePrmPrmBinding, HFETAsourcePrmPrmNode, HFETAsourcePrmPrmNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(HFETAgateGatePtr, HFETAgateGateBinding, HFETAgateNode, HFETAgateNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(HFETAgateGatePrimePtr, HFETAgateGatePrimeBinding, HFETAgateNode, HFETAgatePrimeNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(HFETAgatePrimeGatePtr, HFETAgatePrimeGateBinding, HFETAgatePrimeNode, HFETAgateNode);
        }
    }

    return (OK) ;
}
