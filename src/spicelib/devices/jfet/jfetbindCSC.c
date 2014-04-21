/**********
Author: 2013 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "jfetdefs.h"
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
JFETbindCSC (GENmodel *inModel, CKTcircuit *ckt)
{
    JFETmodel *model = (JFETmodel *)inModel ;
    JFETinstance *here ;
    double *i ;
    BindElement *matched, *BindStruct ;
    size_t nz ;

    BindStruct = ckt->CKTmatrix->CKTbindStruct ;
    nz = (size_t)ckt->CKTmatrix->CKTklunz ;

    /* loop through all the JFET models */
    for ( ; model != NULL ; model = JFETnextModel(model))
    {
        /* loop through all the instances of the model */
        for (here = JFETinstances(model); here != NULL ; here = JFETnextInstance(here))
        {
            CREATE_KLU_BINDING_TABLE(JFETdrainDrainPrimePtr, JFETdrainDrainPrimeBinding, JFETdrainNode, JFETdrainPrimeNode);
            CREATE_KLU_BINDING_TABLE(JFETgateDrainPrimePtr, JFETgateDrainPrimeBinding, JFETgateNode, JFETdrainPrimeNode);
            CREATE_KLU_BINDING_TABLE(JFETgateSourcePrimePtr, JFETgateSourcePrimeBinding, JFETgateNode, JFETsourcePrimeNode);
            CREATE_KLU_BINDING_TABLE(JFETsourceSourcePrimePtr, JFETsourceSourcePrimeBinding, JFETsourceNode, JFETsourcePrimeNode);
            CREATE_KLU_BINDING_TABLE(JFETdrainPrimeDrainPtr, JFETdrainPrimeDrainBinding, JFETdrainPrimeNode, JFETdrainNode);
            CREATE_KLU_BINDING_TABLE(JFETdrainPrimeGatePtr, JFETdrainPrimeGateBinding, JFETdrainPrimeNode, JFETgateNode);
            CREATE_KLU_BINDING_TABLE(JFETdrainPrimeSourcePrimePtr, JFETdrainPrimeSourcePrimeBinding, JFETdrainPrimeNode, JFETsourcePrimeNode);
            CREATE_KLU_BINDING_TABLE(JFETsourcePrimeGatePtr, JFETsourcePrimeGateBinding, JFETsourcePrimeNode, JFETgateNode);
            CREATE_KLU_BINDING_TABLE(JFETsourcePrimeSourcePtr, JFETsourcePrimeSourceBinding, JFETsourcePrimeNode, JFETsourceNode);
            CREATE_KLU_BINDING_TABLE(JFETsourcePrimeDrainPrimePtr, JFETsourcePrimeDrainPrimeBinding, JFETsourcePrimeNode, JFETdrainPrimeNode);
            CREATE_KLU_BINDING_TABLE(JFETdrainDrainPtr, JFETdrainDrainBinding, JFETdrainNode, JFETdrainNode);
            CREATE_KLU_BINDING_TABLE(JFETgateGatePtr, JFETgateGateBinding, JFETgateNode, JFETgateNode);
            CREATE_KLU_BINDING_TABLE(JFETsourceSourcePtr, JFETsourceSourceBinding, JFETsourceNode, JFETsourceNode);
            CREATE_KLU_BINDING_TABLE(JFETdrainPrimeDrainPrimePtr, JFETdrainPrimeDrainPrimeBinding, JFETdrainPrimeNode, JFETdrainPrimeNode);
            CREATE_KLU_BINDING_TABLE(JFETsourcePrimeSourcePrimePtr, JFETsourcePrimeSourcePrimeBinding, JFETsourcePrimeNode, JFETsourcePrimeNode);
        }
    }

    return (OK) ;
}

int
JFETbindCSCComplex (GENmodel *inModel, CKTcircuit *ckt)
{
    JFETmodel *model = (JFETmodel *)inModel ;
    JFETinstance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the JFET models */
    for ( ; model != NULL ; model = JFETnextModel(model))
    {
        /* loop through all the instances of the model */
        for (here = JFETinstances(model); here != NULL ; here = JFETnextInstance(here))
        {
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(JFETdrainDrainPrimePtr, JFETdrainDrainPrimeBinding, JFETdrainNode, JFETdrainPrimeNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(JFETgateDrainPrimePtr, JFETgateDrainPrimeBinding, JFETgateNode, JFETdrainPrimeNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(JFETgateSourcePrimePtr, JFETgateSourcePrimeBinding, JFETgateNode, JFETsourcePrimeNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(JFETsourceSourcePrimePtr, JFETsourceSourcePrimeBinding, JFETsourceNode, JFETsourcePrimeNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(JFETdrainPrimeDrainPtr, JFETdrainPrimeDrainBinding, JFETdrainPrimeNode, JFETdrainNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(JFETdrainPrimeGatePtr, JFETdrainPrimeGateBinding, JFETdrainPrimeNode, JFETgateNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(JFETdrainPrimeSourcePrimePtr, JFETdrainPrimeSourcePrimeBinding, JFETdrainPrimeNode, JFETsourcePrimeNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(JFETsourcePrimeGatePtr, JFETsourcePrimeGateBinding, JFETsourcePrimeNode, JFETgateNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(JFETsourcePrimeSourcePtr, JFETsourcePrimeSourceBinding, JFETsourcePrimeNode, JFETsourceNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(JFETsourcePrimeDrainPrimePtr, JFETsourcePrimeDrainPrimeBinding, JFETsourcePrimeNode, JFETdrainPrimeNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(JFETdrainDrainPtr, JFETdrainDrainBinding, JFETdrainNode, JFETdrainNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(JFETgateGatePtr, JFETgateGateBinding, JFETgateNode, JFETgateNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(JFETsourceSourcePtr, JFETsourceSourceBinding, JFETsourceNode, JFETsourceNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(JFETdrainPrimeDrainPrimePtr, JFETdrainPrimeDrainPrimeBinding, JFETdrainPrimeNode, JFETdrainPrimeNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(JFETsourcePrimeSourcePrimePtr, JFETsourcePrimeSourcePrimeBinding, JFETsourcePrimeNode, JFETsourcePrimeNode);
        }
    }

    return (OK) ;
}

int
JFETbindCSCComplexToReal (GENmodel *inModel, CKTcircuit *ckt)
{
    JFETmodel *model = (JFETmodel *)inModel ;
    JFETinstance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the JFET models */
    for ( ; model != NULL ; model = JFETnextModel(model))
    {
        /* loop through all the instances of the model */
        for (here = JFETinstances(model); here != NULL ; here = JFETnextInstance(here))
        {
            CONVERT_KLU_BINDING_TABLE_TO_REAL(JFETdrainDrainPrimePtr, JFETdrainDrainPrimeBinding, JFETdrainNode, JFETdrainPrimeNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(JFETgateDrainPrimePtr, JFETgateDrainPrimeBinding, JFETgateNode, JFETdrainPrimeNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(JFETgateSourcePrimePtr, JFETgateSourcePrimeBinding, JFETgateNode, JFETsourcePrimeNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(JFETsourceSourcePrimePtr, JFETsourceSourcePrimeBinding, JFETsourceNode, JFETsourcePrimeNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(JFETdrainPrimeDrainPtr, JFETdrainPrimeDrainBinding, JFETdrainPrimeNode, JFETdrainNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(JFETdrainPrimeGatePtr, JFETdrainPrimeGateBinding, JFETdrainPrimeNode, JFETgateNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(JFETdrainPrimeSourcePrimePtr, JFETdrainPrimeSourcePrimeBinding, JFETdrainPrimeNode, JFETsourcePrimeNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(JFETsourcePrimeGatePtr, JFETsourcePrimeGateBinding, JFETsourcePrimeNode, JFETgateNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(JFETsourcePrimeSourcePtr, JFETsourcePrimeSourceBinding, JFETsourcePrimeNode, JFETsourceNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(JFETsourcePrimeDrainPrimePtr, JFETsourcePrimeDrainPrimeBinding, JFETsourcePrimeNode, JFETdrainPrimeNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(JFETdrainDrainPtr, JFETdrainDrainBinding, JFETdrainNode, JFETdrainNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(JFETgateGatePtr, JFETgateGateBinding, JFETgateNode, JFETgateNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(JFETsourceSourcePtr, JFETsourceSourceBinding, JFETsourceNode, JFETsourceNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(JFETdrainPrimeDrainPrimePtr, JFETdrainPrimeDrainPrimeBinding, JFETdrainPrimeNode, JFETdrainPrimeNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(JFETsourcePrimeSourcePrimePtr, JFETsourcePrimeSourcePrimeBinding, JFETsourcePrimeNode, JFETsourcePrimeNode);
        }
    }

    return (OK) ;
}
