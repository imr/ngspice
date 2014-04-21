/**********
Author: 2013 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "numddefs.h"
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
NUMDbindCSC (GENmodel *inModel, CKTcircuit *ckt)
{
    NUMDmodel *model = (NUMDmodel *)inModel ;
    NUMDinstance *here ;
    double *i ;
    BindElement *matched, *BindStruct ;
    size_t nz ;

    BindStruct = ckt->CKTmatrix->CKTbindStruct ;
    nz = (size_t)ckt->CKTmatrix->CKTklunz ;

    /* loop through all the NUMD models */
    for ( ; model != NULL ; model = model->NUMDnextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->NUMDinstances ; here != NULL ; here = here->NUMDnextInstance)
        {
            if ((here-> NUMDposNode != 0) && (here-> NUMDposNode != 0))
            {
                i = here->NUMDposPosPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->NUMDposPosStructPtr = matched ;
                here->NUMDposPosPtr = matched->CSC ;
            }

            if ((here-> NUMDnegNode != 0) && (here-> NUMDnegNode != 0))
            {
                i = here->NUMDnegNegPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->NUMDnegNegStructPtr = matched ;
                here->NUMDnegNegPtr = matched->CSC ;
            }

            if ((here-> NUMDnegNode != 0) && (here-> NUMDposNode != 0))
            {
                i = here->NUMDnegPosPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->NUMDnegPosStructPtr = matched ;
                here->NUMDnegPosPtr = matched->CSC ;
            }

            if ((here-> NUMDposNode != 0) && (here-> NUMDnegNode != 0))
            {
                i = here->NUMDposNegPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->NUMDposNegStructPtr = matched ;
                here->NUMDposNegPtr = matched->CSC ;
            }

        }
    }

    return (OK) ;
}

int
NUMDbindCSCComplex (GENmodel *inModel, CKTcircuit *ckt)
{
    NUMDmodel *model = (NUMDmodel *)inModel ;
    NUMDinstance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the NUMD models */
    for ( ; model != NULL ; model = model->NUMDnextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->NUMDinstances ; here != NULL ; here = here->NUMDnextInstance)
        {
            if ((here-> NUMDposNode != 0) && (here-> NUMDposNode != 0))
                here->NUMDposPosPtr = here->NUMDposPosStructPtr->CSC_Complex ;

            if ((here-> NUMDnegNode != 0) && (here-> NUMDnegNode != 0))
                here->NUMDnegNegPtr = here->NUMDnegNegStructPtr->CSC_Complex ;

            if ((here-> NUMDnegNode != 0) && (here-> NUMDposNode != 0))
                here->NUMDnegPosPtr = here->NUMDnegPosStructPtr->CSC_Complex ;

            if ((here-> NUMDposNode != 0) && (here-> NUMDnegNode != 0))
                here->NUMDposNegPtr = here->NUMDposNegStructPtr->CSC_Complex ;

        }
    }

    return (OK) ;
}

int
NUMDbindCSCComplexToReal (GENmodel *inModel, CKTcircuit *ckt)
{
    NUMDmodel *model = (NUMDmodel *)inModel ;
    NUMDinstance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the NUMD models */
    for ( ; model != NULL ; model = model->NUMDnextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->NUMDinstances ; here != NULL ; here = here->NUMDnextInstance)
        {
            if ((here-> NUMDposNode != 0) && (here-> NUMDposNode != 0))
                here->NUMDposPosPtr = here->NUMDposPosStructPtr->CSC ;

            if ((here-> NUMDnegNode != 0) && (here-> NUMDnegNode != 0))
                here->NUMDnegNegPtr = here->NUMDnegNegStructPtr->CSC ;

            if ((here-> NUMDnegNode != 0) && (here-> NUMDposNode != 0))
                here->NUMDnegPosPtr = here->NUMDnegPosStructPtr->CSC ;

            if ((here-> NUMDposNode != 0) && (here-> NUMDnegNode != 0))
                here->NUMDposNegPtr = here->NUMDposNegStructPtr->CSC ;

        }
    }

    return (OK) ;
}
