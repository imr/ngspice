/**********
Author: 2013 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "vccsdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/klu-binding.h"

int
VCCSbindCSC (GENmodel *inModel, CKTcircuit *ckt)
{
    VCCSmodel *model = (VCCSmodel *)inModel ;
    VCCSinstance *here ;
    BindElement i, *matched, *BindStruct ;
    size_t nz ;

    BindStruct = ckt->CKTmatrix->SMPkluMatrix->KLUmatrixBindStructCOO ;
    nz = (size_t)ckt->CKTmatrix->SMPkluMatrix->KLUmatrixLinkedListNZ ;

    /* loop through all the VCCS models */
    for ( ; model != NULL ; model = VCCSnextModel(model))
    {
        /* loop through all the instances of the model */
        for (here = VCCSinstances(model); here != NULL ; here = VCCSnextInstance(here))
        {
            CREATE_KLU_BINDING_TABLE(VCCSposContPosPtr, VCCSposContPosBinding, VCCSposNode, VCCScontPosNode);
            CREATE_KLU_BINDING_TABLE(VCCSposContNegPtr, VCCSposContNegBinding, VCCSposNode, VCCScontNegNode);
            CREATE_KLU_BINDING_TABLE(VCCSnegContPosPtr, VCCSnegContPosBinding, VCCSnegNode, VCCScontPosNode);
            CREATE_KLU_BINDING_TABLE(VCCSnegContNegPtr, VCCSnegContNegBinding, VCCSnegNode, VCCScontNegNode);
        }
    }

    return (OK) ;
}

int
VCCSbindCSCComplex (GENmodel *inModel, CKTcircuit *ckt)
{
    VCCSmodel *model = (VCCSmodel *)inModel ;
    VCCSinstance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the VCCS models */
    for ( ; model != NULL ; model = VCCSnextModel(model))
    {
        /* loop through all the instances of the model */
        for (here = VCCSinstances(model); here != NULL ; here = VCCSnextInstance(here))
        {
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(VCCSposContPosPtr, VCCSposContPosBinding, VCCSposNode, VCCScontPosNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(VCCSposContNegPtr, VCCSposContNegBinding, VCCSposNode, VCCScontNegNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(VCCSnegContPosPtr, VCCSnegContPosBinding, VCCSnegNode, VCCScontPosNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(VCCSnegContNegPtr, VCCSnegContNegBinding, VCCSnegNode, VCCScontNegNode);
        }
    }

    return (OK) ;
}

int
VCCSbindCSCComplexToReal (GENmodel *inModel, CKTcircuit *ckt)
{
    VCCSmodel *model = (VCCSmodel *)inModel ;
    VCCSinstance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the VCCS models */
    for ( ; model != NULL ; model = VCCSnextModel(model))
    {
        /* loop through all the instances of the model */
        for (here = VCCSinstances(model); here != NULL ; here = VCCSnextInstance(here))
        {
            CONVERT_KLU_BINDING_TABLE_TO_REAL(VCCSposContPosPtr, VCCSposContPosBinding, VCCSposNode, VCCScontPosNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(VCCSposContNegPtr, VCCSposContNegBinding, VCCSposNode, VCCScontNegNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(VCCSnegContPosPtr, VCCSnegContPosBinding, VCCSnegNode, VCCScontPosNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(VCCSnegContNegPtr, VCCSnegContNegBinding, VCCSnegNode, VCCScontNegNode);
        }
    }

    return (OK) ;
}
