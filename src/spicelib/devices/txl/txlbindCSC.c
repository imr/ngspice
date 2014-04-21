/**********
Author: 2013 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "txldefs.h"
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
TXLbindCSC (GENmodel *inModel, CKTcircuit *ckt)
{
    TXLmodel *model = (TXLmodel *)inModel ;
    TXLinstance *here ;
    double *i ;
    BindElement *matched, *BindStruct ;
    size_t nz ;

    BindStruct = ckt->CKTmatrix->CKTbindStruct ;
    nz = (size_t)ckt->CKTmatrix->CKTklunz ;

    /* loop through all the TXL models */
    for ( ; model != NULL ; model = model->TXLnextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->TXLinstances ; here != NULL ; here = here->TXLnextInstance)
        {
            CREATE_KLU_BINDING_TABLE(TXLposPosPtr, TXLposPosBinding, TXLposNode, TXLposNode);
            CREATE_KLU_BINDING_TABLE(TXLposNegPtr, TXLposNegBinding, TXLposNode, TXLnegNode);
            CREATE_KLU_BINDING_TABLE(TXLnegPosPtr, TXLnegPosBinding, TXLnegNode, TXLposNode);
            CREATE_KLU_BINDING_TABLE(TXLnegNegPtr, TXLnegNegBinding, TXLnegNode, TXLnegNode);
            CREATE_KLU_BINDING_TABLE(TXLibr1PosPtr, TXLibr1PosBinding, TXLibr1, TXLposNode);
            CREATE_KLU_BINDING_TABLE(TXLibr2NegPtr, TXLibr2NegBinding, TXLibr2, TXLnegNode);
            CREATE_KLU_BINDING_TABLE(TXLnegIbr2Ptr, TXLnegIbr2Binding, TXLnegNode, TXLibr2);
            CREATE_KLU_BINDING_TABLE(TXLposIbr1Ptr, TXLposIbr1Binding, TXLposNode, TXLibr1);
            CREATE_KLU_BINDING_TABLE(TXLibr1Ibr1Ptr, TXLibr1Ibr1Binding, TXLibr1, TXLibr1);
            CREATE_KLU_BINDING_TABLE(TXLibr2Ibr2Ptr, TXLibr2Ibr2Binding, TXLibr2, TXLibr2);
            CREATE_KLU_BINDING_TABLE(TXLibr1NegPtr, TXLibr1NegBinding, TXLibr1, TXLnegNode);
            CREATE_KLU_BINDING_TABLE(TXLibr2PosPtr, TXLibr2PosBinding, TXLibr2, TXLposNode);
            CREATE_KLU_BINDING_TABLE(TXLibr1Ibr2Ptr, TXLibr1Ibr2Binding, TXLibr1, TXLibr2);
            CREATE_KLU_BINDING_TABLE(TXLibr2Ibr1Ptr, TXLibr2Ibr1Binding, TXLibr2, TXLibr1);
        }
    }

    return (OK) ;
}

int
TXLbindCSCComplex (GENmodel *inModel, CKTcircuit *ckt)
{
    TXLmodel *model = (TXLmodel *)inModel ;
    TXLinstance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the TXL models */
    for ( ; model != NULL ; model = model->TXLnextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->TXLinstances ; here != NULL ; here = here->TXLnextInstance)
        {
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(TXLposPosPtr, TXLposPosBinding, TXLposNode, TXLposNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(TXLposNegPtr, TXLposNegBinding, TXLposNode, TXLnegNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(TXLnegPosPtr, TXLnegPosBinding, TXLnegNode, TXLposNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(TXLnegNegPtr, TXLnegNegBinding, TXLnegNode, TXLnegNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(TXLibr1PosPtr, TXLibr1PosBinding, TXLibr1, TXLposNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(TXLibr2NegPtr, TXLibr2NegBinding, TXLibr2, TXLnegNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(TXLnegIbr2Ptr, TXLnegIbr2Binding, TXLnegNode, TXLibr2);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(TXLposIbr1Ptr, TXLposIbr1Binding, TXLposNode, TXLibr1);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(TXLibr1Ibr1Ptr, TXLibr1Ibr1Binding, TXLibr1, TXLibr1);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(TXLibr2Ibr2Ptr, TXLibr2Ibr2Binding, TXLibr2, TXLibr2);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(TXLibr1NegPtr, TXLibr1NegBinding, TXLibr1, TXLnegNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(TXLibr2PosPtr, TXLibr2PosBinding, TXLibr2, TXLposNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(TXLibr1Ibr2Ptr, TXLibr1Ibr2Binding, TXLibr1, TXLibr2);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(TXLibr2Ibr1Ptr, TXLibr2Ibr1Binding, TXLibr2, TXLibr1);
        }
    }

    return (OK) ;
}

int
TXLbindCSCComplexToReal (GENmodel *inModel, CKTcircuit *ckt)
{
    TXLmodel *model = (TXLmodel *)inModel ;
    TXLinstance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the TXL models */
    for ( ; model != NULL ; model = model->TXLnextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->TXLinstances ; here != NULL ; here = here->TXLnextInstance)
        {
            CONVERT_KLU_BINDING_TABLE_TO_REAL(TXLposPosPtr, TXLposPosBinding, TXLposNode, TXLposNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(TXLposNegPtr, TXLposNegBinding, TXLposNode, TXLnegNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(TXLnegPosPtr, TXLnegPosBinding, TXLnegNode, TXLposNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(TXLnegNegPtr, TXLnegNegBinding, TXLnegNode, TXLnegNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(TXLibr1PosPtr, TXLibr1PosBinding, TXLibr1, TXLposNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(TXLibr2NegPtr, TXLibr2NegBinding, TXLibr2, TXLnegNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(TXLnegIbr2Ptr, TXLnegIbr2Binding, TXLnegNode, TXLibr2);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(TXLposIbr1Ptr, TXLposIbr1Binding, TXLposNode, TXLibr1);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(TXLibr1Ibr1Ptr, TXLibr1Ibr1Binding, TXLibr1, TXLibr1);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(TXLibr2Ibr2Ptr, TXLibr2Ibr2Binding, TXLibr2, TXLibr2);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(TXLibr1NegPtr, TXLibr1NegBinding, TXLibr1, TXLnegNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(TXLibr2PosPtr, TXLibr2PosBinding, TXLibr2, TXLposNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(TXLibr1Ibr2Ptr, TXLibr1Ibr2Binding, TXLibr1, TXLibr2);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(TXLibr2Ibr1Ptr, TXLibr2Ibr1Binding, TXLibr2, TXLibr1);
        }
    }

    return (OK) ;
}
