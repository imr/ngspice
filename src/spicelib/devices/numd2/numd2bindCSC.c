/**********
Author: 2013 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "numd2def.h"
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
NUMD2bindCSC (GENmodel *inModel, CKTcircuit *ckt)
{
    NUMD2model *model = (NUMD2model *)inModel ;
    NUMD2instance *here ;
    double *i ;
    BindElement *matched, *BindStruct ;
    size_t nz ;

    BindStruct = ckt->CKTmatrix->CKTbindStruct ;
    nz = (size_t)ckt->CKTmatrix->CKTklunz ;

    /* loop through all the NUMD2 models */
    for ( ; model != NULL ; model = model->NUMD2nextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->NUMD2instances ; here != NULL ; here = here->NUMD2nextInstance)
        {
            CREATE_KLU_BINDING_TABLE(NUMD2posPosPtr, NUMD2posPosBinding, NUMD2posNode, NUMD2posNode);
            CREATE_KLU_BINDING_TABLE(NUMD2negNegPtr, NUMD2negNegBinding, NUMD2negNode, NUMD2negNode);
            CREATE_KLU_BINDING_TABLE(NUMD2negPosPtr, NUMD2negPosBinding, NUMD2negNode, NUMD2posNode);
            CREATE_KLU_BINDING_TABLE(NUMD2posNegPtr, NUMD2posNegBinding, NUMD2posNode, NUMD2negNode);
        }
    }

    return (OK) ;
}

int
NUMD2bindCSCComplex (GENmodel *inModel, CKTcircuit *ckt)
{
    NUMD2model *model = (NUMD2model *)inModel ;
    NUMD2instance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the NUMD2 models */
    for ( ; model != NULL ; model = model->NUMD2nextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->NUMD2instances ; here != NULL ; here = here->NUMD2nextInstance)
        {
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(NUMD2posPosPtr, NUMD2posPosBinding, NUMD2posNode, NUMD2posNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(NUMD2negNegPtr, NUMD2negNegBinding, NUMD2negNode, NUMD2negNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(NUMD2negPosPtr, NUMD2negPosBinding, NUMD2negNode, NUMD2posNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(NUMD2posNegPtr, NUMD2posNegBinding, NUMD2posNode, NUMD2negNode);
        }
    }

    return (OK) ;
}

int
NUMD2bindCSCComplexToReal (GENmodel *inModel, CKTcircuit *ckt)
{
    NUMD2model *model = (NUMD2model *)inModel ;
    NUMD2instance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the NUMD2 models */
    for ( ; model != NULL ; model = model->NUMD2nextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->NUMD2instances ; here != NULL ; here = here->NUMD2nextInstance)
        {
            CONVERT_KLU_BINDING_TABLE_TO_REAL(NUMD2posPosPtr, NUMD2posPosBinding, NUMD2posNode, NUMD2posNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(NUMD2negNegPtr, NUMD2negNegBinding, NUMD2negNode, NUMD2negNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(NUMD2negPosPtr, NUMD2negPosBinding, NUMD2negNode, NUMD2posNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(NUMD2posNegPtr, NUMD2posNegBinding, NUMD2posNode, NUMD2negNode);
        }
    }

    return (OK) ;
}
