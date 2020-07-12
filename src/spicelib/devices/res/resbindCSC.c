/**********
Author: 2013 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "resdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/klu-binding.h"

int
RESbindCSC (GENmodel *inModel, CKTcircuit *ckt)
{
    RESmodel *model = (RESmodel *)inModel ;
    RESinstance *here ;
    BindElement i, *matched, *BindStruct ;
    size_t nz ;

    BindStruct = ckt->CKTmatrix->SMPkluMatrix->KLUmatrixBindStructCOO ;
    nz = (size_t)ckt->CKTmatrix->SMPkluMatrix->KLUmatrixLinkedListNZ ;

    /* loop through all the RES models */
    for ( ; model != NULL ; model = RESnextModel(model))
    {
        /* loop through all the instances of the model */
        for (here = RESinstances(model); here != NULL ; here = RESnextInstance(here))
        {
            CREATE_KLU_BINDING_TABLE(RESposPosPtr, RESposPosBinding, RESposNode, RESposNode);
            CREATE_KLU_BINDING_TABLE(RESnegNegPtr, RESnegNegBinding, RESnegNode, RESnegNode);
            CREATE_KLU_BINDING_TABLE(RESposNegPtr, RESposNegBinding, RESposNode, RESnegNode);
            CREATE_KLU_BINDING_TABLE(RESnegPosPtr, RESnegPosBinding, RESnegNode, RESposNode);
        }
    }

    return (OK) ;
}

int
RESbindCSCComplex (GENmodel *inModel, CKTcircuit *ckt)
{
    RESmodel *model = (RESmodel *)inModel ;
    RESinstance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the RES models */
    for ( ; model != NULL ; model = RESnextModel(model))
    {
        /* loop through all the instances of the model */
        for (here = RESinstances(model); here != NULL ; here = RESnextInstance(here))
        {
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(RESposPosPtr, RESposPosBinding, RESposNode, RESposNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(RESnegNegPtr, RESnegNegBinding, RESnegNode, RESnegNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(RESposNegPtr, RESposNegBinding, RESposNode, RESnegNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(RESnegPosPtr, RESnegPosBinding, RESnegNode, RESposNode);
        }
    }

    return (OK) ;
}

int
RESbindCSCComplexToReal (GENmodel *inModel, CKTcircuit *ckt)
{
    RESmodel *model = (RESmodel *)inModel ;
    RESinstance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the RES models */
    for ( ; model != NULL ; model = RESnextModel(model))
    {
        /* loop through all the instances of the model */
        for (here = RESinstances(model); here != NULL ; here = RESnextInstance(here))
        {
            CONVERT_KLU_BINDING_TABLE_TO_REAL(RESposPosPtr, RESposPosBinding, RESposNode, RESposNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(RESnegNegPtr, RESnegNegBinding, RESnegNode, RESnegNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(RESposNegPtr, RESposNegBinding, RESposNode, RESnegNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(RESnegPosPtr, RESnegPosBinding, RESnegNode, RESposNode);
        }
    }

    return (OK) ;
}
