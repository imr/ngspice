/**********
Author: 2013 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "diodefs.h"
#include "ngspice/sperror.h"
#include "ngspice/klu-binding.h"

int
DIObindCSC (GENmodel *inModel, CKTcircuit *ckt)
{
    DIOmodel *model = (DIOmodel *)inModel ;
    DIOinstance *here ;
    BindElement i, *matched, *BindStruct ;
    size_t nz ;

    BindStruct = ckt->CKTmatrix->SMPkluMatrix->KLUmatrixBindStructCOO ;
    nz = (size_t)ckt->CKTmatrix->SMPkluMatrix->KLUmatrixLinkedListNZ ;

    /* loop through all the DIO models */
    for ( ; model != NULL ; model = DIOnextModel(model))
    {
        /* loop through all the instances of the model */
        for (here = DIOinstances(model); here != NULL ; here = DIOnextInstance(here))
        {
            CREATE_KLU_BINDING_TABLE(DIOposPosPrimePtr, DIOposPosPrimeBinding, DIOposNode, DIOposPrimeNode);
            CREATE_KLU_BINDING_TABLE(DIOnegPosPrimePtr, DIOnegPosPrimeBinding, DIOnegNode, DIOposPrimeNode);
            CREATE_KLU_BINDING_TABLE(DIOposPrimePosPtr, DIOposPrimePosBinding, DIOposPrimeNode, DIOposNode);
            CREATE_KLU_BINDING_TABLE(DIOposPrimeNegPtr, DIOposPrimeNegBinding, DIOposPrimeNode, DIOnegNode);
            CREATE_KLU_BINDING_TABLE(DIOposPosPtr, DIOposPosBinding, DIOposNode, DIOposNode);
            CREATE_KLU_BINDING_TABLE(DIOnegNegPtr, DIOnegNegBinding, DIOnegNode, DIOnegNode);
            CREATE_KLU_BINDING_TABLE(DIOposPrimePosPrimePtr, DIOposPrimePosPrimeBinding, DIOposPrimeNode, DIOposPrimeNode);
        }
    }

    return (OK) ;
}

int
DIObindCSCComplex (GENmodel *inModel, CKTcircuit *ckt)
{
    DIOmodel *model = (DIOmodel *)inModel ;
    DIOinstance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the DIO models */
    for ( ; model != NULL ; model = DIOnextModel(model))
    {
        /* loop through all the instances of the model */
        for (here = DIOinstances(model); here != NULL ; here = DIOnextInstance(here))
        {
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(DIOposPosPrimePtr, DIOposPosPrimeBinding, DIOposNode, DIOposPrimeNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(DIOnegPosPrimePtr, DIOnegPosPrimeBinding, DIOnegNode, DIOposPrimeNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(DIOposPrimePosPtr, DIOposPrimePosBinding, DIOposPrimeNode, DIOposNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(DIOposPrimeNegPtr, DIOposPrimeNegBinding, DIOposPrimeNode, DIOnegNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(DIOposPosPtr, DIOposPosBinding, DIOposNode, DIOposNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(DIOnegNegPtr, DIOnegNegBinding, DIOnegNode, DIOnegNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(DIOposPrimePosPrimePtr, DIOposPrimePosPrimeBinding, DIOposPrimeNode, DIOposPrimeNode);
        }
    }

    return (OK) ;
}

int
DIObindCSCComplexToReal (GENmodel *inModel, CKTcircuit *ckt)
{
    DIOmodel *model = (DIOmodel *)inModel ;
    DIOinstance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the DIO models */
    for ( ; model != NULL ; model = DIOnextModel(model))
    {
        /* loop through all the instances of the model */
        for (here = DIOinstances(model); here != NULL ; here = DIOnextInstance(here))
        {
            CONVERT_KLU_BINDING_TABLE_TO_REAL(DIOposPosPrimePtr, DIOposPosPrimeBinding, DIOposNode, DIOposPrimeNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(DIOnegPosPrimePtr, DIOnegPosPrimeBinding, DIOnegNode, DIOposPrimeNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(DIOposPrimePosPtr, DIOposPrimePosBinding, DIOposPrimeNode, DIOposNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(DIOposPrimeNegPtr, DIOposPrimeNegBinding, DIOposPrimeNode, DIOnegNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(DIOposPosPtr, DIOposPosBinding, DIOposNode, DIOposNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(DIOnegNegPtr, DIOnegNegBinding, DIOnegNode, DIOnegNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(DIOposPrimePosPrimePtr, DIOposPrimePosPrimeBinding, DIOposPrimeNode, DIOposPrimeNode);
        }
    }

    return (OK) ;
}
