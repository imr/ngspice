/**********
Author: 2013 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "resdefs.h"
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
RESbindCSC (GENmodel *inModel, CKTcircuit *ckt)
{
    RESmodel *model = (RESmodel *)inModel ;
    RESinstance *here ;
    double *i ;
    BindElement *matched, *BindStruct ;
    size_t nz ;

    BindStruct = ckt->CKTmatrix->CKTbindStruct ;
    nz = (size_t)ckt->CKTmatrix->CKTklunz ;

    /* loop through all the RES models */
    for ( ; model != NULL ; model = RESnextModel(model))
    {
        /* loop through all the instances of the model */
        for (here = RESinstances(model); here != NULL ; here = RESnextInstance(here))
        {
            CREATE_KLU_BINDING_TABLE_STATIC(RESposPosPtr, RESposPosBinding, RESposNode, RESposNode);
            CREATE_KLU_BINDING_TABLE_STATIC(RESnegNegPtr, RESnegNegBinding, RESnegNode, RESnegNode);
            CREATE_KLU_BINDING_TABLE_STATIC(RESposNegPtr, RESposNegBinding, RESposNode, RESnegNode);
            CREATE_KLU_BINDING_TABLE_STATIC(RESnegPosPtr, RESnegPosBinding, RESnegNode, RESposNode);
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
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX_STATIC(RESposPosPtr, RESposPosBinding, RESposNode, RESposNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX_STATIC(RESnegNegPtr, RESnegNegBinding, RESnegNode, RESnegNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX_STATIC(RESposNegPtr, RESposNegBinding, RESposNode, RESnegNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX_STATIC(RESnegPosPtr, RESnegPosBinding, RESnegNode, RESposNode);
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
            CONVERT_KLU_BINDING_TABLE_TO_REAL_STATIC(RESposPosPtr, RESposPosBinding, RESposNode, RESposNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL_STATIC(RESnegNegPtr, RESnegNegBinding, RESnegNode, RESnegNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL_STATIC(RESposNegPtr, RESposNegBinding, RESposNode, RESnegNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL_STATIC(RESnegPosPtr, RESnegPosBinding, RESnegNode, RESposNode);
        }
    }

    return (OK) ;
}
