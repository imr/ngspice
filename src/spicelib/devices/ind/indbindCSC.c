/**********
Author: 2013 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "inddefs.h"
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
INDbindCSC (GENmodel *inModel, CKTcircuit *ckt)
{
    INDmodel *model = (INDmodel *)inModel ;
    INDinstance *here ;
    double *i ;
    BindElement *matched, *BindStruct ;
    size_t nz ;

    BindStruct = ckt->CKTmatrix->CKTbindStruct ;
    nz = (size_t)ckt->CKTmatrix->CKTklunz ;

    /* loop through all the IND models */
    for ( ; model != NULL ; model = model->INDnextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->INDinstances ; here != NULL ; here = here->INDnextInstance)
        {
            CREATE_KLU_BINDING_TABLE(INDposIbrPtr, INDposIbrBinding, INDposNode, INDbrEq);
            CREATE_KLU_BINDING_TABLE(INDnegIbrPtr, INDnegIbrBinding, INDnegNode, INDbrEq);
            CREATE_KLU_BINDING_TABLE(INDibrNegPtr, INDibrNegBinding, INDbrEq, INDnegNode);
            CREATE_KLU_BINDING_TABLE(INDibrPosPtr, INDibrPosBinding, INDbrEq, INDposNode);
            CREATE_KLU_BINDING_TABLE(INDibrIbrPtr, INDibrIbrBinding, INDbrEq, INDbrEq);
        }
    }

    return (OK) ;
}

int
INDbindCSCComplex (GENmodel *inModel, CKTcircuit *ckt)
{
    INDmodel *model = (INDmodel *)inModel ;
    INDinstance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the IND models */
    for ( ; model != NULL ; model = model->INDnextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->INDinstances ; here != NULL ; here = here->INDnextInstance)
        {
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(INDposIbrPtr, INDposIbrBinding, INDposNode, INDbrEq);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(INDnegIbrPtr, INDnegIbrBinding, INDnegNode, INDbrEq);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(INDibrNegPtr, INDibrNegBinding, INDbrEq, INDnegNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(INDibrPosPtr, INDibrPosBinding, INDbrEq, INDposNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(INDibrIbrPtr, INDibrIbrBinding, INDbrEq, INDbrEq);
        }
    }

    return (OK) ;
}

int
INDbindCSCComplexToReal (GENmodel *inModel, CKTcircuit *ckt)
{
    INDmodel *model = (INDmodel *)inModel ;
    INDinstance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the IND models */
    for ( ; model != NULL ; model = model->INDnextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->INDinstances ; here != NULL ; here = here->INDnextInstance)
        {
            CONVERT_KLU_BINDING_TABLE_TO_REAL(INDposIbrPtr, INDposIbrBinding, INDposNode, INDbrEq);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(INDnegIbrPtr, INDnegIbrBinding, INDnegNode, INDbrEq);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(INDibrNegPtr, INDibrNegBinding, INDbrEq, INDnegNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(INDibrPosPtr, INDibrPosBinding, INDbrEq, INDposNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(INDibrIbrPtr, INDibrIbrBinding, INDbrEq, INDbrEq);
        }
    }

    return (OK) ;
}
