/**********
Author: 2013 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "cccsdefs.h"
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
CCCSbindCSC (GENmodel *inModel, CKTcircuit *ckt)
{
    CCCSmodel *model = (CCCSmodel *)inModel ;
    CCCSinstance *here ;
    double *i ;
    BindElement *matched, *BindStruct ;
    size_t nz ;

    BindStruct = ckt->CKTmatrix->CKTbindStruct ;
    nz = (size_t)ckt->CKTmatrix->CKTklunz ;

    /* loop through all the CCCS models */
    for ( ; model != NULL ; model = model->CCCSnextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->CCCSinstances ; here != NULL ; here = here->CCCSnextInstance)
        {
            CREATE_KLU_BINDING_TABLE(CCCSposContBrPtr, CCCSposContBrBinding, CCCSposNode, CCCScontBranch);
            CREATE_KLU_BINDING_TABLE(CCCSnegContBrPtr, CCCSnegContBrBinding, CCCSnegNode, CCCScontBranch);
        }
    }

    return (OK) ;
}

int
CCCSbindCSCComplex (GENmodel *inModel, CKTcircuit *ckt)
{
    CCCSmodel *model = (CCCSmodel *)inModel ;
    CCCSinstance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the CCCS models */
    for ( ; model != NULL ; model = model->CCCSnextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->CCCSinstances ; here != NULL ; here = here->CCCSnextInstance)
        {
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(CCCSposContBrPtr, CCCSposContBrBinding, CCCSposNode, CCCScontBranch);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(CCCSnegContBrPtr, CCCSnegContBrBinding, CCCSnegNode, CCCScontBranch);
        }
    }

    return (OK) ;
}

int
CCCSbindCSCComplexToReal (GENmodel *inModel, CKTcircuit *ckt)
{
    CCCSmodel *model = (CCCSmodel *)inModel ;
    CCCSinstance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the CCCS models */
    for ( ; model != NULL ; model = model->CCCSnextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->CCCSinstances ; here != NULL ; here = here->CCCSnextInstance)
        {
            CONVERT_KLU_BINDING_TABLE_TO_REAL(CCCSposContBrPtr, CCCSposContBrBinding, CCCSposNode, CCCScontBranch);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(CCCSnegContBrPtr, CCCSnegContBrBinding, CCCSnegNode, CCCScontBranch);
        }
    }

    return (OK) ;
}
