/**********
Author: 2013 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ccvsdefs.h"
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
CCVSbindCSC (GENmodel *inModel, CKTcircuit *ckt)
{
    CCVSmodel *model = (CCVSmodel *)inModel ;
    CCVSinstance *here ;
    double *i ;
    BindElement *matched, *BindStruct ;
    size_t nz ;

    BindStruct = ckt->CKTmatrix->CKTbindStruct ;
    nz = (size_t)ckt->CKTmatrix->CKTklunz ;

    /* loop through all the CCVS models */
    for ( ; model != NULL ; model = CCVSnextModel(model))
    {
        /* loop through all the instances of the model */
        for (here = CCVSinstances(model); here != NULL ; here = CCVSnextInstance(here))
        {
            CREATE_KLU_BINDING_TABLE(CCVSposIbrPtr, CCVSposIbrBinding, CCVSposNode, CCVSbranch);
            CREATE_KLU_BINDING_TABLE(CCVSnegIbrPtr, CCVSnegIbrBinding, CCVSnegNode, CCVSbranch);
            CREATE_KLU_BINDING_TABLE(CCVSibrNegPtr, CCVSibrNegBinding, CCVSbranch, CCVSnegNode);
            CREATE_KLU_BINDING_TABLE(CCVSibrPosPtr, CCVSibrPosBinding, CCVSbranch, CCVSposNode);
            CREATE_KLU_BINDING_TABLE(CCVSibrContBrPtr, CCVSibrContBrBinding, CCVSbranch, CCVScontBranch);
        }
    }

    return (OK) ;
}

int
CCVSbindCSCComplex (GENmodel *inModel, CKTcircuit *ckt)
{
    CCVSmodel *model = (CCVSmodel *)inModel ;
    CCVSinstance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the CCVS models */
    for ( ; model != NULL ; model = CCVSnextModel(model))
    {
        /* loop through all the instances of the model */
        for (here = CCVSinstances(model); here != NULL ; here = CCVSnextInstance(here))
        {
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(CCVSposIbrPtr, CCVSposIbrBinding, CCVSposNode, CCVSbranch);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(CCVSnegIbrPtr, CCVSnegIbrBinding, CCVSnegNode, CCVSbranch);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(CCVSibrNegPtr, CCVSibrNegBinding, CCVSbranch, CCVSnegNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(CCVSibrPosPtr, CCVSibrPosBinding, CCVSbranch, CCVSposNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(CCVSibrContBrPtr, CCVSibrContBrBinding, CCVSbranch, CCVScontBranch);
        }
    }

    return (OK) ;
}

int
CCVSbindCSCComplexToReal (GENmodel *inModel, CKTcircuit *ckt)
{
    CCVSmodel *model = (CCVSmodel *)inModel ;
    CCVSinstance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the CCVS models */
    for ( ; model != NULL ; model = CCVSnextModel(model))
    {
        /* loop through all the instances of the model */
        for (here = CCVSinstances(model); here != NULL ; here = CCVSnextInstance(here))
        {
            CONVERT_KLU_BINDING_TABLE_TO_REAL(CCVSposIbrPtr, CCVSposIbrBinding, CCVSposNode, CCVSbranch);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(CCVSnegIbrPtr, CCVSnegIbrBinding, CCVSnegNode, CCVSbranch);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(CCVSibrNegPtr, CCVSibrNegBinding, CCVSbranch, CCVSnegNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(CCVSibrPosPtr, CCVSibrPosBinding, CCVSbranch, CCVSposNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(CCVSibrContBrPtr, CCVSibrContBrBinding, CCVSbranch, CCVScontBranch);
        }
    }

    return (OK) ;
}
