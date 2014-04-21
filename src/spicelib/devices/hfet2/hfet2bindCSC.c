/**********
Author: 2013 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "hfet2defs.h"
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
HFET2bindCSC (GENmodel *inModel, CKTcircuit *ckt)
{
    HFET2model *model = (HFET2model *)inModel ;
    HFET2instance *here ;
    double *i ;
    BindElement *matched, *BindStruct ;
    size_t nz ;

    BindStruct = ckt->CKTmatrix->CKTbindStruct ;
    nz = (size_t)ckt->CKTmatrix->CKTklunz ;

    /* loop through all the HFET2 models */
    for ( ; model != NULL ; model = model->HFET2nextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->HFET2instances ; here != NULL ; here = here->HFET2nextInstance)
        {
            CREATE_KLU_BINDING_TABLE(HFET2drainDrainPrimePtr, HFET2drainDrainPrimeBinding, HFET2drainNode, HFET2drainPrimeNode);
            CREATE_KLU_BINDING_TABLE(HFET2gateDrainPrimePtr, HFET2gateDrainPrimeBinding, HFET2gateNode, HFET2drainPrimeNode);
            CREATE_KLU_BINDING_TABLE(HFET2gateSourcePrimePtr, HFET2gateSourcePrimeBinding, HFET2gateNode, HFET2sourcePrimeNode);
            CREATE_KLU_BINDING_TABLE(HFET2sourceSourcePrimePtr, HFET2sourceSourcePrimeBinding, HFET2sourceNode, HFET2sourcePrimeNode);
            CREATE_KLU_BINDING_TABLE(HFET2drainPrimeDrainPtr, HFET2drainPrimeDrainBinding, HFET2drainPrimeNode, HFET2drainNode);
            CREATE_KLU_BINDING_TABLE(HFET2drainPrimeGatePtr, HFET2drainPrimeGateBinding, HFET2drainPrimeNode, HFET2gateNode);
            CREATE_KLU_BINDING_TABLE(HFET2drainPriHFET2ourcePrimePtr, HFET2drainPriHFET2ourcePrimeBinding, HFET2drainPrimeNode, HFET2sourcePrimeNode);
            CREATE_KLU_BINDING_TABLE(HFET2sourcePrimeGatePtr, HFET2sourcePrimeGateBinding, HFET2sourcePrimeNode, HFET2gateNode);
            CREATE_KLU_BINDING_TABLE(HFET2sourcePriHFET2ourcePtr, HFET2sourcePriHFET2ourceBinding, HFET2sourcePrimeNode, HFET2sourceNode);
            CREATE_KLU_BINDING_TABLE(HFET2sourcePrimeDrainPrimePtr, HFET2sourcePrimeDrainPrimeBinding, HFET2sourcePrimeNode, HFET2drainPrimeNode);
            CREATE_KLU_BINDING_TABLE(HFET2drainDrainPtr, HFET2drainDrainBinding, HFET2drainNode, HFET2drainNode);
            CREATE_KLU_BINDING_TABLE(HFET2gateGatePtr, HFET2gateGateBinding, HFET2gateNode, HFET2gateNode);
            CREATE_KLU_BINDING_TABLE(HFET2sourceSourcePtr, HFET2sourceSourceBinding, HFET2sourceNode, HFET2sourceNode);
            CREATE_KLU_BINDING_TABLE(HFET2drainPrimeDrainPrimePtr, HFET2drainPrimeDrainPrimeBinding, HFET2drainPrimeNode, HFET2drainPrimeNode);
            CREATE_KLU_BINDING_TABLE(HFET2sourcePriHFET2ourcePrimePtr, HFET2sourcePriHFET2ourcePrimeBinding, HFET2sourcePrimeNode, HFET2sourcePrimeNode);
        }
    }

    return (OK) ;
}

int
HFET2bindCSCComplex (GENmodel *inModel, CKTcircuit *ckt)
{
    HFET2model *model = (HFET2model *)inModel ;
    HFET2instance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the HFET2 models */
    for ( ; model != NULL ; model = model->HFET2nextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->HFET2instances ; here != NULL ; here = here->HFET2nextInstance)
        {
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HFET2drainDrainPrimePtr, HFET2drainDrainPrimeBinding, HFET2drainNode, HFET2drainPrimeNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HFET2gateDrainPrimePtr, HFET2gateDrainPrimeBinding, HFET2gateNode, HFET2drainPrimeNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HFET2gateSourcePrimePtr, HFET2gateSourcePrimeBinding, HFET2gateNode, HFET2sourcePrimeNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HFET2sourceSourcePrimePtr, HFET2sourceSourcePrimeBinding, HFET2sourceNode, HFET2sourcePrimeNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HFET2drainPrimeDrainPtr, HFET2drainPrimeDrainBinding, HFET2drainPrimeNode, HFET2drainNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HFET2drainPrimeGatePtr, HFET2drainPrimeGateBinding, HFET2drainPrimeNode, HFET2gateNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HFET2drainPriHFET2ourcePrimePtr, HFET2drainPriHFET2ourcePrimeBinding, HFET2drainPrimeNode, HFET2sourcePrimeNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HFET2sourcePrimeGatePtr, HFET2sourcePrimeGateBinding, HFET2sourcePrimeNode, HFET2gateNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HFET2sourcePriHFET2ourcePtr, HFET2sourcePriHFET2ourceBinding, HFET2sourcePrimeNode, HFET2sourceNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HFET2sourcePrimeDrainPrimePtr, HFET2sourcePrimeDrainPrimeBinding, HFET2sourcePrimeNode, HFET2drainPrimeNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HFET2drainDrainPtr, HFET2drainDrainBinding, HFET2drainNode, HFET2drainNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HFET2gateGatePtr, HFET2gateGateBinding, HFET2gateNode, HFET2gateNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HFET2sourceSourcePtr, HFET2sourceSourceBinding, HFET2sourceNode, HFET2sourceNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HFET2drainPrimeDrainPrimePtr, HFET2drainPrimeDrainPrimeBinding, HFET2drainPrimeNode, HFET2drainPrimeNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HFET2sourcePriHFET2ourcePrimePtr, HFET2sourcePriHFET2ourcePrimeBinding, HFET2sourcePrimeNode, HFET2sourcePrimeNode);
        }
    }

    return (OK) ;
}

int
HFET2bindCSCComplexToReal (GENmodel *inModel, CKTcircuit *ckt)
{
    HFET2model *model = (HFET2model *)inModel ;
    HFET2instance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the HFET2 models */
    for ( ; model != NULL ; model = model->HFET2nextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->HFET2instances ; here != NULL ; here = here->HFET2nextInstance)
        {
            CONVERT_KLU_BINDING_TABLE_TO_REAL(HFET2drainDrainPrimePtr, HFET2drainDrainPrimeBinding, HFET2drainNode, HFET2drainPrimeNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(HFET2gateDrainPrimePtr, HFET2gateDrainPrimeBinding, HFET2gateNode, HFET2drainPrimeNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(HFET2gateSourcePrimePtr, HFET2gateSourcePrimeBinding, HFET2gateNode, HFET2sourcePrimeNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(HFET2sourceSourcePrimePtr, HFET2sourceSourcePrimeBinding, HFET2sourceNode, HFET2sourcePrimeNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(HFET2drainPrimeDrainPtr, HFET2drainPrimeDrainBinding, HFET2drainPrimeNode, HFET2drainNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(HFET2drainPrimeGatePtr, HFET2drainPrimeGateBinding, HFET2drainPrimeNode, HFET2gateNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(HFET2drainPriHFET2ourcePrimePtr, HFET2drainPriHFET2ourcePrimeBinding, HFET2drainPrimeNode, HFET2sourcePrimeNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(HFET2sourcePrimeGatePtr, HFET2sourcePrimeGateBinding, HFET2sourcePrimeNode, HFET2gateNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(HFET2sourcePriHFET2ourcePtr, HFET2sourcePriHFET2ourceBinding, HFET2sourcePrimeNode, HFET2sourceNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(HFET2sourcePrimeDrainPrimePtr, HFET2sourcePrimeDrainPrimeBinding, HFET2sourcePrimeNode, HFET2drainPrimeNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(HFET2drainDrainPtr, HFET2drainDrainBinding, HFET2drainNode, HFET2drainNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(HFET2gateGatePtr, HFET2gateGateBinding, HFET2gateNode, HFET2gateNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(HFET2sourceSourcePtr, HFET2sourceSourceBinding, HFET2sourceNode, HFET2sourceNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(HFET2drainPrimeDrainPrimePtr, HFET2drainPrimeDrainPrimeBinding, HFET2drainPrimeNode, HFET2drainPrimeNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(HFET2sourcePriHFET2ourcePrimePtr, HFET2sourcePriHFET2ourcePrimeBinding, HFET2sourcePrimeNode, HFET2sourcePrimeNode);
        }
    }

    return (OK) ;
}
