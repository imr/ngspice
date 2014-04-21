/**********
Author: 2013 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "vcvsdefs.h"
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
VCVSbindCSC (GENmodel *inModel, CKTcircuit *ckt)
{
    VCVSmodel *model = (VCVSmodel *)inModel ;
    VCVSinstance *here ;
    double *i ;
    BindElement *matched, *BindStruct ;
    size_t nz ;

    BindStruct = ckt->CKTmatrix->CKTbindStruct ;
    nz = (size_t)ckt->CKTmatrix->CKTklunz ;

    /* loop through all the VCVS models */
    for ( ; model != NULL ; model = model->VCVSnextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->VCVSinstances ; here != NULL ; here = here->VCVSnextInstance)
        {
            CREATE_KLU_BINDING_TABLE(VCVSposIbrPtr, VCVSposIbrBinding, VCVSposNode, VCVSbranch);
            CREATE_KLU_BINDING_TABLE(VCVSnegIbrPtr, VCVSnegIbrBinding, VCVSnegNode, VCVSbranch);
            CREATE_KLU_BINDING_TABLE(VCVSibrNegPtr, VCVSibrNegBinding, VCVSbranch, VCVSnegNode);
            CREATE_KLU_BINDING_TABLE(VCVSibrPosPtr, VCVSibrPosBinding, VCVSbranch, VCVSposNode);
            CREATE_KLU_BINDING_TABLE(VCVSibrContPosPtr, VCVSibrContPosBinding, VCVSbranch, VCVScontPosNode);
            CREATE_KLU_BINDING_TABLE(VCVSibrContNegPtr, VCVSibrContNegBinding, VCVSbranch, VCVScontNegNode);
        }
    }

    return (OK) ;
}

int
VCVSbindCSCComplex (GENmodel *inModel, CKTcircuit *ckt)
{
    VCVSmodel *model = (VCVSmodel *)inModel ;
    VCVSinstance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the VCVS models */
    for ( ; model != NULL ; model = model->VCVSnextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->VCVSinstances ; here != NULL ; here = here->VCVSnextInstance)
        {
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(VCVSposIbrPtr, VCVSposIbrBinding, VCVSposNode, VCVSbranch);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(VCVSnegIbrPtr, VCVSnegIbrBinding, VCVSnegNode, VCVSbranch);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(VCVSibrNegPtr, VCVSibrNegBinding, VCVSbranch, VCVSnegNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(VCVSibrPosPtr, VCVSibrPosBinding, VCVSbranch, VCVSposNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(VCVSibrContPosPtr, VCVSibrContPosBinding, VCVSbranch, VCVScontPosNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(VCVSibrContNegPtr, VCVSibrContNegBinding, VCVSbranch, VCVScontNegNode);
        }
    }

    return (OK) ;
}

int
VCVSbindCSCComplexToReal (GENmodel *inModel, CKTcircuit *ckt)
{
    VCVSmodel *model = (VCVSmodel *)inModel ;
    VCVSinstance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the VCVS models */
    for ( ; model != NULL ; model = model->VCVSnextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->VCVSinstances ; here != NULL ; here = here->VCVSnextInstance)
        {
            CONVERT_KLU_BINDING_TABLE_TO_REAL(VCVSposIbrPtr, VCVSposIbrBinding, VCVSposNode, VCVSbranch);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(VCVSnegIbrPtr, VCVSnegIbrBinding, VCVSnegNode, VCVSbranch);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(VCVSibrNegPtr, VCVSibrNegBinding, VCVSbranch, VCVSnegNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(VCVSibrPosPtr, VCVSibrPosBinding, VCVSbranch, VCVSposNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(VCVSibrContPosPtr, VCVSibrContPosBinding, VCVSbranch, VCVScontPosNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(VCVSibrContNegPtr, VCVSibrContNegBinding, VCVSbranch, VCVScontNegNode);
        }
    }

    return (OK) ;
}
