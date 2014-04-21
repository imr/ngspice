/**********
Author: 2013 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "cccsdefs.h"
#include "ngspice/sperror.h"

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
            if ((here->CCCSposNode != 0) && (here->CCCScontBranch != 0))
            {
                i = here->CCCSposContBrptr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->CCCSposContBrptrStructPtr = matched ;
                here->CCCSposContBrptr = matched->CSC ;
            }

            if ((here->CCCSnegNode != 0) && (here->CCCScontBranch != 0))
            {
                i = here->CCCSnegContBrptr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->CCCSnegContBrptrStructPtr = matched ;
                here->CCCSnegContBrptr = matched->CSC ;
            }

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
            if ((here->CCCSposNode != 0) && (here->CCCScontBranch != 0))
                here->CCCSposContBrptr = here->CCCSposContBrptrStructPtr->CSC_Complex ;

            if ((here->CCCSnegNode != 0) && (here->CCCScontBranch != 0))
                here->CCCSnegContBrptr = here->CCCSnegContBrptrStructPtr->CSC_Complex ;

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
            if ((here->CCCSposNode != 0) && (here->CCCScontBranch != 0))
                here->CCCSposContBrptr = here->CCCSposContBrptrStructPtr->CSC ;

            if ((here->CCCSnegNode != 0) && (here->CCCScontBranch != 0))
                here->CCCSnegContBrptr = here->CCCSnegContBrptrStructPtr->CSC ;

        }
    }

    return (OK) ;
}
