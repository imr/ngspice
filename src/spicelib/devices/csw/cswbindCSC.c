/**********
Author: 2013 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "cswdefs.h"
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
CSWbindCSC (GENmodel *inModel, CKTcircuit *ckt)
{
    CSWmodel *model = (CSWmodel *)inModel ;
    CSWinstance *here ;
    double *i ;
    BindElement *matched, *BindStruct ;
    size_t nz ;

    BindStruct = ckt->CKTmatrix->CKTbindStruct ;
    nz = (size_t)ckt->CKTmatrix->CKTklunz ;

    /* loop through all the CSW models */
    for ( ; model != NULL ; model = model->CSWnextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->CSWinstances ; here != NULL ; here = here->CSWnextInstance)
        {
            if ((here-> CSWposNode != 0) && (here-> CSWposNode != 0))
            {
                i = here->CSWposPosptr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->CSWposPosptrStructPtr = matched ;
                here->CSWposPosptr = matched->CSC ;
            }

            if ((here-> CSWposNode != 0) && (here-> CSWnegNode != 0))
            {
                i = here->CSWposNegptr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->CSWposNegptrStructPtr = matched ;
                here->CSWposNegptr = matched->CSC ;
            }

            if ((here-> CSWnegNode != 0) && (here-> CSWposNode != 0))
            {
                i = here->CSWnegPosptr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->CSWnegPosptrStructPtr = matched ;
                here->CSWnegPosptr = matched->CSC ;
            }

            if ((here-> CSWnegNode != 0) && (here-> CSWnegNode != 0))
            {
                i = here->CSWnegNegptr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->CSWnegNegptrStructPtr = matched ;
                here->CSWnegNegptr = matched->CSC ;
            }

        }
    }

    return (OK) ;
}

int
CSWbindCSCComplex (GENmodel *inModel, CKTcircuit *ckt)
{
    CSWmodel *model = (CSWmodel *)inModel ;
    CSWinstance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the CSW models */
    for ( ; model != NULL ; model = model->CSWnextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->CSWinstances ; here != NULL ; here = here->CSWnextInstance)
        {
            if ((here-> CSWposNode != 0) && (here-> CSWposNode != 0))
                here->CSWposPosptr = here->CSWposPosptrStructPtr->CSC_Complex ;

            if ((here-> CSWposNode != 0) && (here-> CSWnegNode != 0))
                here->CSWposNegptr = here->CSWposNegptrStructPtr->CSC_Complex ;

            if ((here-> CSWnegNode != 0) && (here-> CSWposNode != 0))
                here->CSWnegPosptr = here->CSWnegPosptrStructPtr->CSC_Complex ;

            if ((here-> CSWnegNode != 0) && (here-> CSWnegNode != 0))
                here->CSWnegNegptr = here->CSWnegNegptrStructPtr->CSC_Complex ;

        }
    }

    return (OK) ;
}

int
CSWbindCSCComplexToReal (GENmodel *inModel, CKTcircuit *ckt)
{
    CSWmodel *model = (CSWmodel *)inModel ;
    CSWinstance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the CSW models */
    for ( ; model != NULL ; model = model->CSWnextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->CSWinstances ; here != NULL ; here = here->CSWnextInstance)
        {
            if ((here-> CSWposNode != 0) && (here-> CSWposNode != 0))
                here->CSWposPosptr = here->CSWposPosptrStructPtr->CSC ;

            if ((here-> CSWposNode != 0) && (here-> CSWnegNode != 0))
                here->CSWposNegptr = here->CSWposNegptrStructPtr->CSC ;

            if ((here-> CSWnegNode != 0) && (here-> CSWposNode != 0))
                here->CSWnegPosptr = here->CSWnegPosptrStructPtr->CSC ;

            if ((here-> CSWnegNode != 0) && (here-> CSWnegNode != 0))
                here->CSWnegNegptr = here->CSWnegNegptrStructPtr->CSC ;

        }
    }

    return (OK) ;
}
