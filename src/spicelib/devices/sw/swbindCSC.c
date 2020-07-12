/**********
Author: 2013 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "swdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/klu-binding.h"

int
SWbindCSC (GENmodel *inModel, CKTcircuit *ckt)
{
    SWmodel *model = (SWmodel *)inModel ;
    SWinstance *here ;
    BindElement i, *matched, *BindStruct ;
    size_t nz ;

    BindStruct = ckt->CKTmatrix->SMPkluMatrix->KLUmatrixBindStructCOO ;
    nz = (size_t)ckt->CKTmatrix->SMPkluMatrix->KLUmatrixLinkedListNZ ;

    /* loop through all the SW models */
    for ( ; model != NULL ; model = SWnextModel(model))
    {
        /* loop through all the instances of the model */
        for (here = SWinstances(model); here != NULL ; here = SWnextInstance(here))
        {
            CREATE_KLU_BINDING_TABLE(SWposPosPtr, SWposPosBinding, SWposNode, SWposNode);
            CREATE_KLU_BINDING_TABLE(SWposNegPtr, SWposNegBinding, SWposNode, SWnegNode);
            CREATE_KLU_BINDING_TABLE(SWnegPosPtr, SWnegPosBinding, SWnegNode, SWposNode);
            CREATE_KLU_BINDING_TABLE(SWnegNegPtr, SWnegNegBinding, SWnegNode, SWnegNode);
        }
    }

    return (OK) ;
}

int
SWbindCSCComplex (GENmodel *inModel, CKTcircuit *ckt)
{
    SWmodel *model = (SWmodel *)inModel ;
    SWinstance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the SW models */
    for ( ; model != NULL ; model = SWnextModel(model))
    {
        /* loop through all the instances of the model */
        for (here = SWinstances(model); here != NULL ; here = SWnextInstance(here))
        {
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(SWposPosPtr, SWposPosBinding, SWposNode, SWposNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(SWposNegPtr, SWposNegBinding, SWposNode, SWnegNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(SWnegPosPtr, SWnegPosBinding, SWnegNode, SWposNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(SWnegNegPtr, SWnegNegBinding, SWnegNode, SWnegNode);
        }
    }

    return (OK) ;
}

int
SWbindCSCComplexToReal (GENmodel *inModel, CKTcircuit *ckt)
{
    SWmodel *model = (SWmodel *)inModel ;
    SWinstance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the SW models */
    for ( ; model != NULL ; model = SWnextModel(model))
    {
        /* loop through all the instances of the model */
        for (here = SWinstances(model); here != NULL ; here = SWnextInstance(here))
        {
            CONVERT_KLU_BINDING_TABLE_TO_REAL(SWposPosPtr, SWposPosBinding, SWposNode, SWposNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(SWposNegPtr, SWposNegBinding, SWposNode, SWnegNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(SWnegPosPtr, SWnegPosBinding, SWnegNode, SWposNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(SWnegNegPtr, SWnegNegBinding, SWnegNode, SWnegNode);
        }
    }

    return (OK) ;
}
