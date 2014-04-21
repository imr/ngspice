/**********
Author: 2013 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "capdefs.h"
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
CAPbindCSC (GENmodel *inModel, CKTcircuit *ckt)
{
    CAPmodel *model = (CAPmodel *)inModel ;
    CAPinstance *here ;
    double *i ;
    BindElement *matched, *BindStruct ;
    size_t nz ;

    BindStruct = ckt->CKTmatrix->CKTbindStruct ;
    nz = (size_t)ckt->CKTmatrix->CKTklunz ;

    /* loop through all the CAP models */
    for ( ; model != NULL ; model = model->CAPnextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->CAPinstances ; here != NULL ; here = here->CAPnextInstance)
        {
            CREATE_KLU_BINDING_TABLE(CAPposPosPtr, CAPposPosBinding, CAPposNode, CAPposNode);
            CREATE_KLU_BINDING_TABLE(CAPnegNegPtr, CAPnegNegBinding, CAPnegNode, CAPnegNode);
            CREATE_KLU_BINDING_TABLE(CAPposNegPtr, CAPposNegBinding, CAPposNode, CAPnegNode);
            CREATE_KLU_BINDING_TABLE(CAPnegPosPtr, CAPnegPosBinding, CAPnegNode, CAPposNode);
        }
    }

    return (OK) ;
}

int
CAPbindCSCComplex (GENmodel *inModel, CKTcircuit *ckt)
{
    CAPmodel *model = (CAPmodel *)inModel ;
    CAPinstance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the CAP models */
    for ( ; model != NULL ; model = model->CAPnextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->CAPinstances ; here != NULL ; here = here->CAPnextInstance)
        {
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(CAPposPosPtr, CAPposPosBinding, CAPposNode, CAPposNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(CAPnegNegPtr, CAPnegNegBinding, CAPnegNode, CAPnegNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(CAPposNegPtr, CAPposNegBinding, CAPposNode, CAPnegNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(CAPnegPosPtr, CAPnegPosBinding, CAPnegNode, CAPposNode);
        }
    }

    return (OK) ;
}

int
CAPbindCSCComplexToReal (GENmodel *inModel, CKTcircuit *ckt)
{
    CAPmodel *model = (CAPmodel *)inModel ;
    CAPinstance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the CAP models */
    for ( ; model != NULL ; model = model->CAPnextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->CAPinstances ; here != NULL ; here = here->CAPnextInstance)
        {
            CONVERT_KLU_BINDING_TABLE_TO_REAL(CAPposPosPtr, CAPposPosBinding, CAPposNode, CAPposNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(CAPnegNegPtr, CAPnegNegBinding, CAPnegNode, CAPnegNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(CAPposNegPtr, CAPposNegBinding, CAPposNode, CAPnegNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(CAPnegPosPtr, CAPnegPosBinding, CAPnegNode, CAPposNode);
        }
    }

    return (OK) ;
}
