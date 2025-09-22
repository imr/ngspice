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
            if ((here->DIOtempNode > 0) && (here->DIOthermal) && (model->DIOrth0Given)) {
                CREATE_KLU_BINDING_TABLE(DIOtempPosPtr,      DIOtempPosBinding,      DIOtempNode,     DIOposNode);
                CREATE_KLU_BINDING_TABLE(DIOtempPosPrimePtr, DIOtempPosPrimeBinding, DIOtempNode,     DIOposPrimeNode);
                CREATE_KLU_BINDING_TABLE(DIOtempNegPtr,      DIOtempNegBinding,      DIOtempNode,     DIOnegNode);
                CREATE_KLU_BINDING_TABLE(DIOtempTempPtr,     DIOtempTempBinding,     DIOtempNode,     DIOtempNode);
                CREATE_KLU_BINDING_TABLE(DIOposTempPtr,      DIOposTempBinding,      DIOposNode,      DIOtempNode);
                CREATE_KLU_BINDING_TABLE(DIOposPrimeTempPtr, DIOposPrimeTempBinding, DIOposPrimeNode, DIOtempNode);
                CREATE_KLU_BINDING_TABLE(DIOnegTempPtr,      DIOnegTempBinding,      DIOnegNode,      DIOtempNode);
            }
//            if (model->DIOsoftRevRecParamGiven) {
//                CREATE_KLU_BINDING_TABLE(DIOqpQpPtr      , DIOqpQpBinding      , DIOqpNode      , DIOqpNode);
//                CREATE_KLU_BINDING_TABLE(DIOqpPosPrimePtr, DIOqpPosPrimeBinding, DIOqpNode      , DIOposPrimeNode);
//                CREATE_KLU_BINDING_TABLE(DIOqpNegPtr     , DIOqpNegBinding     , DIOqpNode      , DIOnegNode);
//                CREATE_KLU_BINDING_TABLE(DIOposPrimeQpPtr, DIOposPrimeQpBinding, DIOposPrimeNode, DIOqpNode);
//                CREATE_KLU_BINDING_TABLE(DIOnegQpPtr     , DIOnegQpBinding     , DIOnegNode     , DIOqpNode);
//            }
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
            if ((here->DIOtempNode > 0) && (here->DIOthermal) && (model->DIOrth0Given)) {
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(DIOtempPosPtr,      DIOtempPosBinding,      DIOtempNode,     DIOposNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(DIOtempPosPrimePtr, DIOtempPosPrimeBinding, DIOtempNode,     DIOposPrimeNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(DIOtempNegPtr,      DIOtempNegBinding,      DIOtempNode,     DIOnegNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(DIOtempTempPtr,     DIOtempTempBinding,     DIOtempNode,     DIOtempNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(DIOposTempPtr,      DIOposTempBinding,      DIOposNode,      DIOtempNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(DIOposPrimeTempPtr, DIOposPrimeTempBinding, DIOposPrimeNode, DIOtempNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(DIOnegTempPtr,      DIOnegTempBinding,      DIOnegNode,      DIOtempNode);
            }
//            if (model->DIOsoftRevRecParamGiven) {
//                CREATE_KLU_BINDING_TABLE_TO_COMPLEX(DIOqpQpPtr      , DIOqpQpBinding      , DIOqpNode      , DIOqpNode);
//                CREATE_KLU_BINDING_TABLE_TO_COMPLEX(DIOqpPosPrimePtr, DIOqpPosPrimeBinding, DIOqpNode      , DIOposPrimeNode);
//                CREATE_KLU_BINDING_TABLE_TO_COMPLEX(DIOqpNegPtr     , DIOqpNegBinding     , DIOqpNode      , DIOnegNode);
//                CREATE_KLU_BINDING_TABLE_TO_COMPLEX(DIOposPrimeQpPtr, DIOposPrimeQpBinding, DIOposPrimeNode, DIOqpNode);
//                CREATE_KLU_BINDING_TABLE_TO_COMPLEX(DIOnegQpPtr     , DIOnegQpBinding     , DIOnegNode     , DIOqpNode);
//            }
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
            if ((here->DIOtempNode > 0) && (here->DIOthermal) && (model->DIOrth0Given)) {
                CONVERT_KLU_BINDING_TABLE_TO_REAL(DIOtempPosPtr,      DIOtempPosBinding,      DIOtempNode,     DIOposNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(DIOtempPosPrimePtr, DIOtempPosPrimeBinding, DIOtempNode,     DIOposPrimeNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(DIOtempNegPtr,      DIOtempNegBinding,      DIOtempNode,     DIOnegNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(DIOtempTempPtr,     DIOtempTempBinding,     DIOtempNode,     DIOtempNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(DIOposTempPtr,      DIOposTempBinding,      DIOposNode,      DIOtempNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(DIOposPrimeTempPtr, DIOposPrimeTempBinding, DIOposPrimeNode, DIOtempNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(DIOnegTempPtr,      DIOnegTempBinding,      DIOnegNode,      DIOtempNode);
            }
//            if (model->DIOsoftRevRecParamGiven) {
//                CREATE_KLU_BINDING_TABLE_TO_REAL(DIOqpQpPtr      , DIOqpQpBinding      , DIOqpNode      , DIOqpNode);
//                CREATE_KLU_BINDING_TABLE_TO_REAL(DIOqpPosPrimePtr, DIOqpPosPrimeBinding, DIOqpNode      , DIOposPrimeNode);
//                CREATE_KLU_BINDING_TABLE_TO_REAL(DIOqpNegPtr     , DIOqpNegBinding     , DIOqpNode      , DIOnegNode);
//                CREATE_KLU_BINDING_TABLE_TO_REAL(DIOposPrimeQpPtr, DIOposPrimeQpBinding, DIOposPrimeNode, DIOqpNode);
//                CREATE_KLU_BINDING_TABLE_TO_REAL(DIOnegQpPtr     , DIOnegQpBinding     , DIOnegNode     , DIOqpNode);
//            }
        }
    }

    return (OK) ;
}
