/**********
Author: 2013 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "jfet2defs.h"
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
JFET2bindCSC (GENmodel *inModel, CKTcircuit *ckt)
{
    JFET2model *model = (JFET2model *)inModel ;
    JFET2instance *here ;
    double *i ;
    BindElement *matched, *BindStruct ;
    size_t nz ;

    BindStruct = ckt->CKTmatrix->CKTbindStruct ;
    nz = (size_t)ckt->CKTmatrix->CKTklunz ;

    /* loop through all the JFET2 models */
    for ( ; model != NULL ; model = model->JFET2nextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->JFET2instances ; here != NULL ; here = here->JFET2nextInstance)
        {
            CREATE_KLU_BINDING_TABLE(JFET2drainDrainPrimePtr, JFET2drainDrainPrimeBinding, JFET2drainNode, JFET2drainPrimeNode);
            CREATE_KLU_BINDING_TABLE(JFET2gateDrainPrimePtr, JFET2gateDrainPrimeBinding, JFET2gateNode, JFET2drainPrimeNode);
            CREATE_KLU_BINDING_TABLE(JFET2gateSourcePrimePtr, JFET2gateSourcePrimeBinding, JFET2gateNode, JFET2sourcePrimeNode);
            CREATE_KLU_BINDING_TABLE(JFET2sourceSourcePrimePtr, JFET2sourceSourcePrimeBinding, JFET2sourceNode, JFET2sourcePrimeNode);
            CREATE_KLU_BINDING_TABLE(JFET2drainPrimeDrainPtr, JFET2drainPrimeDrainBinding, JFET2drainPrimeNode, JFET2drainNode);
            CREATE_KLU_BINDING_TABLE(JFET2drainPrimeGatePtr, JFET2drainPrimeGateBinding, JFET2drainPrimeNode, JFET2gateNode);
            CREATE_KLU_BINDING_TABLE(JFET2drainPrimeSourcePrimePtr, JFET2drainPrimeSourcePrimeBinding, JFET2drainPrimeNode, JFET2sourcePrimeNode);
            CREATE_KLU_BINDING_TABLE(JFET2sourcePrimeGatePtr, JFET2sourcePrimeGateBinding, JFET2sourcePrimeNode, JFET2gateNode);
            CREATE_KLU_BINDING_TABLE(JFET2sourcePrimeSourcePtr, JFET2sourcePrimeSourceBinding, JFET2sourcePrimeNode, JFET2sourceNode);
            CREATE_KLU_BINDING_TABLE(JFET2sourcePrimeDrainPrimePtr, JFET2sourcePrimeDrainPrimeBinding, JFET2sourcePrimeNode, JFET2drainPrimeNode);
            CREATE_KLU_BINDING_TABLE(JFET2drainDrainPtr, JFET2drainDrainBinding, JFET2drainNode, JFET2drainNode);
            CREATE_KLU_BINDING_TABLE(JFET2gateGatePtr, JFET2gateGateBinding, JFET2gateNode, JFET2gateNode);
            CREATE_KLU_BINDING_TABLE(JFET2sourceSourcePtr, JFET2sourceSourceBinding, JFET2sourceNode, JFET2sourceNode);
            CREATE_KLU_BINDING_TABLE(JFET2drainPrimeDrainPrimePtr, JFET2drainPrimeDrainPrimeBinding, JFET2drainPrimeNode, JFET2drainPrimeNode);
            CREATE_KLU_BINDING_TABLE(JFET2sourcePrimeSourcePrimePtr, JFET2sourcePrimeSourcePrimeBinding, JFET2sourcePrimeNode, JFET2sourcePrimeNode);
        }
    }

    return (OK) ;
}

int
JFET2bindCSCComplex (GENmodel *inModel, CKTcircuit *ckt)
{
    JFET2model *model = (JFET2model *)inModel ;
    JFET2instance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the JFET2 models */
    for ( ; model != NULL ; model = model->JFET2nextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->JFET2instances ; here != NULL ; here = here->JFET2nextInstance)
        {
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(JFET2drainDrainPrimePtr, JFET2drainDrainPrimeBinding, JFET2drainNode, JFET2drainPrimeNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(JFET2gateDrainPrimePtr, JFET2gateDrainPrimeBinding, JFET2gateNode, JFET2drainPrimeNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(JFET2gateSourcePrimePtr, JFET2gateSourcePrimeBinding, JFET2gateNode, JFET2sourcePrimeNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(JFET2sourceSourcePrimePtr, JFET2sourceSourcePrimeBinding, JFET2sourceNode, JFET2sourcePrimeNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(JFET2drainPrimeDrainPtr, JFET2drainPrimeDrainBinding, JFET2drainPrimeNode, JFET2drainNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(JFET2drainPrimeGatePtr, JFET2drainPrimeGateBinding, JFET2drainPrimeNode, JFET2gateNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(JFET2drainPrimeSourcePrimePtr, JFET2drainPrimeSourcePrimeBinding, JFET2drainPrimeNode, JFET2sourcePrimeNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(JFET2sourcePrimeGatePtr, JFET2sourcePrimeGateBinding, JFET2sourcePrimeNode, JFET2gateNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(JFET2sourcePrimeSourcePtr, JFET2sourcePrimeSourceBinding, JFET2sourcePrimeNode, JFET2sourceNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(JFET2sourcePrimeDrainPrimePtr, JFET2sourcePrimeDrainPrimeBinding, JFET2sourcePrimeNode, JFET2drainPrimeNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(JFET2drainDrainPtr, JFET2drainDrainBinding, JFET2drainNode, JFET2drainNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(JFET2gateGatePtr, JFET2gateGateBinding, JFET2gateNode, JFET2gateNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(JFET2sourceSourcePtr, JFET2sourceSourceBinding, JFET2sourceNode, JFET2sourceNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(JFET2drainPrimeDrainPrimePtr, JFET2drainPrimeDrainPrimeBinding, JFET2drainPrimeNode, JFET2drainPrimeNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(JFET2sourcePrimeSourcePrimePtr, JFET2sourcePrimeSourcePrimeBinding, JFET2sourcePrimeNode, JFET2sourcePrimeNode);
        }
    }

    return (OK) ;
}

int
JFET2bindCSCComplexToReal (GENmodel *inModel, CKTcircuit *ckt)
{
    JFET2model *model = (JFET2model *)inModel ;
    JFET2instance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the JFET2 models */
    for ( ; model != NULL ; model = model->JFET2nextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->JFET2instances ; here != NULL ; here = here->JFET2nextInstance)
        {
            CONVERT_KLU_BINDING_TABLE_TO_REAL(JFET2drainDrainPrimePtr, JFET2drainDrainPrimeBinding, JFET2drainNode, JFET2drainPrimeNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(JFET2gateDrainPrimePtr, JFET2gateDrainPrimeBinding, JFET2gateNode, JFET2drainPrimeNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(JFET2gateSourcePrimePtr, JFET2gateSourcePrimeBinding, JFET2gateNode, JFET2sourcePrimeNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(JFET2sourceSourcePrimePtr, JFET2sourceSourcePrimeBinding, JFET2sourceNode, JFET2sourcePrimeNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(JFET2drainPrimeDrainPtr, JFET2drainPrimeDrainBinding, JFET2drainPrimeNode, JFET2drainNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(JFET2drainPrimeGatePtr, JFET2drainPrimeGateBinding, JFET2drainPrimeNode, JFET2gateNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(JFET2drainPrimeSourcePrimePtr, JFET2drainPrimeSourcePrimeBinding, JFET2drainPrimeNode, JFET2sourcePrimeNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(JFET2sourcePrimeGatePtr, JFET2sourcePrimeGateBinding, JFET2sourcePrimeNode, JFET2gateNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(JFET2sourcePrimeSourcePtr, JFET2sourcePrimeSourceBinding, JFET2sourcePrimeNode, JFET2sourceNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(JFET2sourcePrimeDrainPrimePtr, JFET2sourcePrimeDrainPrimeBinding, JFET2sourcePrimeNode, JFET2drainPrimeNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(JFET2drainDrainPtr, JFET2drainDrainBinding, JFET2drainNode, JFET2drainNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(JFET2gateGatePtr, JFET2gateGateBinding, JFET2gateNode, JFET2gateNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(JFET2sourceSourcePtr, JFET2sourceSourceBinding, JFET2sourceNode, JFET2sourceNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(JFET2drainPrimeDrainPrimePtr, JFET2drainPrimeDrainPrimeBinding, JFET2drainPrimeNode, JFET2drainPrimeNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(JFET2sourcePrimeSourcePrimePtr, JFET2sourcePrimeSourcePrimeBinding, JFET2sourcePrimeNode, JFET2sourcePrimeNode);
        }
    }

    return (OK) ;
}
