/**********
Author: 2013 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "swdefs.h"
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
SWbindCSC (GENmodel *inModel, CKTcircuit *ckt)
{
    SWmodel *model = (SWmodel *)inModel ;
    SWinstance *here ;
    double *i ;
    BindElement *matched, *BindStruct ;
    size_t nz ;

    BindStruct = ckt->CKTmatrix->CKTbindStruct ;
    nz = (size_t)ckt->CKTmatrix->CKTklunz ;

    /* loop through all the SW models */
    for ( ; model != NULL ; model = model->SWnextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->SWinstances ; here != NULL ; here = here->SWnextInstance)
        {
            if ((here-> SWposNode != 0) && (here-> SWposNode != 0))
            {
                i = here->SWposPosptr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->SWposPosptrStructPtr = matched ;
                here->SWposPosptr = matched->CSC ;
            }

            if ((here-> SWposNode != 0) && (here-> SWnegNode != 0))
            {
                i = here->SWposNegptr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->SWposNegptrStructPtr = matched ;
                here->SWposNegptr = matched->CSC ;
            }

            if ((here-> SWnegNode != 0) && (here-> SWposNode != 0))
            {
                i = here->SWnegPosptr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->SWnegPosptrStructPtr = matched ;
                here->SWnegPosptr = matched->CSC ;
            }

            if ((here-> SWnegNode != 0) && (here-> SWnegNode != 0))
            {
                i = here->SWnegNegptr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->SWnegNegptrStructPtr = matched ;
                here->SWnegNegptr = matched->CSC ;
            }

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
    for ( ; model != NULL ; model = model->SWnextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->SWinstances ; here != NULL ; here = here->SWnextInstance)
        {
            if ((here-> SWposNode != 0) && (here-> SWposNode != 0))
                here->SWposPosptr = here->SWposPosptrStructPtr->CSC_Complex ;

            if ((here-> SWposNode != 0) && (here-> SWnegNode != 0))
                here->SWposNegptr = here->SWposNegptrStructPtr->CSC_Complex ;

            if ((here-> SWnegNode != 0) && (here-> SWposNode != 0))
                here->SWnegPosptr = here->SWnegPosptrStructPtr->CSC_Complex ;

            if ((here-> SWnegNode != 0) && (here-> SWnegNode != 0))
                here->SWnegNegptr = here->SWnegNegptrStructPtr->CSC_Complex ;

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
    for ( ; model != NULL ; model = model->SWnextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->SWinstances ; here != NULL ; here = here->SWnextInstance)
        {
            if ((here-> SWposNode != 0) && (here-> SWposNode != 0))
                here->SWposPosptr = here->SWposPosptrStructPtr->CSC ;

            if ((here-> SWposNode != 0) && (here-> SWnegNode != 0))
                here->SWposNegptr = here->SWposNegptrStructPtr->CSC ;

            if ((here-> SWnegNode != 0) && (here-> SWposNode != 0))
                here->SWnegPosptr = here->SWnegPosptrStructPtr->CSC ;

            if ((here-> SWnegNode != 0) && (here-> SWnegNode != 0))
                here->SWnegNegptr = here->SWnegNegptrStructPtr->CSC ;

        }
    }

    return (OK) ;
}
