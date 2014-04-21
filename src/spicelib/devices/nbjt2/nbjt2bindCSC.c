/**********
Author: 2013 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "nbjt2def.h"
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
NBJT2bindCSC (GENmodel *inModel, CKTcircuit *ckt)
{
    NBJT2model *model = (NBJT2model *)inModel ;
    NBJT2instance *here ;
    double *i ;
    BindElement *matched, *BindStruct ;
    size_t nz ;

    BindStruct = ckt->CKTmatrix->CKTbindStruct ;
    nz = (size_t)ckt->CKTmatrix->CKTklunz ;

    /* loop through all the NBJT2 models */
    for ( ; model != NULL ; model = model->NBJT2nextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->NBJT2instances ; here != NULL ; here = here->NBJT2nextInstance)
        {
            CREATE_KLU_BINDING_TABLE(NBJT2colColPtr, NBJT2colColBinding, NBJT2colNode, NBJT2colNode);
            CREATE_KLU_BINDING_TABLE(NBJT2colBasePtr, NBJT2colBaseBinding, NBJT2colNode, NBJT2baseNode);
            CREATE_KLU_BINDING_TABLE(NBJT2colEmitPtr, NBJT2colEmitBinding, NBJT2colNode, NBJT2emitNode);
            CREATE_KLU_BINDING_TABLE(NBJT2baseColPtr, NBJT2baseColBinding, NBJT2baseNode, NBJT2colNode);
            CREATE_KLU_BINDING_TABLE(NBJT2baseBasePtr, NBJT2baseBaseBinding, NBJT2baseNode, NBJT2baseNode);
            CREATE_KLU_BINDING_TABLE(NBJT2baseEmitPtr, NBJT2baseEmitBinding, NBJT2baseNode, NBJT2emitNode);
            CREATE_KLU_BINDING_TABLE(NBJT2emitColPtr, NBJT2emitColBinding, NBJT2emitNode, NBJT2colNode);
            CREATE_KLU_BINDING_TABLE(NBJT2emitBasePtr, NBJT2emitBaseBinding, NBJT2emitNode, NBJT2baseNode);
            CREATE_KLU_BINDING_TABLE(NBJT2emitEmitPtr, NBJT2emitEmitBinding, NBJT2emitNode, NBJT2emitNode);
        }
    }

    return (OK) ;
}

int
NBJT2bindCSCComplex (GENmodel *inModel, CKTcircuit *ckt)
{
    NBJT2model *model = (NBJT2model *)inModel ;
    NBJT2instance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the NBJT2 models */
    for ( ; model != NULL ; model = model->NBJT2nextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->NBJT2instances ; here != NULL ; here = here->NBJT2nextInstance)
        {
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(NBJT2colColPtr, NBJT2colColBinding, NBJT2colNode, NBJT2colNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(NBJT2colBasePtr, NBJT2colBaseBinding, NBJT2colNode, NBJT2baseNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(NBJT2colEmitPtr, NBJT2colEmitBinding, NBJT2colNode, NBJT2emitNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(NBJT2baseColPtr, NBJT2baseColBinding, NBJT2baseNode, NBJT2colNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(NBJT2baseBasePtr, NBJT2baseBaseBinding, NBJT2baseNode, NBJT2baseNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(NBJT2baseEmitPtr, NBJT2baseEmitBinding, NBJT2baseNode, NBJT2emitNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(NBJT2emitColPtr, NBJT2emitColBinding, NBJT2emitNode, NBJT2colNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(NBJT2emitBasePtr, NBJT2emitBaseBinding, NBJT2emitNode, NBJT2baseNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(NBJT2emitEmitPtr, NBJT2emitEmitBinding, NBJT2emitNode, NBJT2emitNode);
        }
    }

    return (OK) ;
}

int
NBJT2bindCSCComplexToReal (GENmodel *inModel, CKTcircuit *ckt)
{
    NBJT2model *model = (NBJT2model *)inModel ;
    NBJT2instance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the NBJT2 models */
    for ( ; model != NULL ; model = model->NBJT2nextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->NBJT2instances ; here != NULL ; here = here->NBJT2nextInstance)
        {
            CONVERT_KLU_BINDING_TABLE_TO_REAL(NBJT2colColPtr, NBJT2colColBinding, NBJT2colNode, NBJT2colNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(NBJT2colBasePtr, NBJT2colBaseBinding, NBJT2colNode, NBJT2baseNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(NBJT2colEmitPtr, NBJT2colEmitBinding, NBJT2colNode, NBJT2emitNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(NBJT2baseColPtr, NBJT2baseColBinding, NBJT2baseNode, NBJT2colNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(NBJT2baseBasePtr, NBJT2baseBaseBinding, NBJT2baseNode, NBJT2baseNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(NBJT2baseEmitPtr, NBJT2baseEmitBinding, NBJT2baseNode, NBJT2emitNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(NBJT2emitColPtr, NBJT2emitColBinding, NBJT2emitNode, NBJT2colNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(NBJT2emitBasePtr, NBJT2emitBaseBinding, NBJT2emitNode, NBJT2baseNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(NBJT2emitEmitPtr, NBJT2emitEmitBinding, NBJT2emitNode, NBJT2emitNode);
        }
    }

    return (OK) ;
}
