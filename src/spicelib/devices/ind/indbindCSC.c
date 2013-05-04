/**********
Author: 2013 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "inddefs.h"
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
            if ((here->INDposNode != 0) && (here->INDbrEq != 0))
            {
                i = here->INDposIbrptr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->INDposIbrptrStructPtr = matched ;
                here->INDposIbrptr = matched->CSC_LinearDynamic ;
            }

            if ((here->INDnegNode != 0) && (here->INDbrEq != 0))
            {
                i = here->INDnegIbrptr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->INDnegIbrptrStructPtr = matched ;
                here->INDnegIbrptr = matched->CSC_LinearDynamic ;
            }

            if ((here->INDbrEq != 0) && (here->INDnegNode != 0))
            {
                i = here->INDibrNegptr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->INDibrNegptrStructPtr = matched ;
                here->INDibrNegptr = matched->CSC_LinearDynamic ;
            }

            if ((here->INDbrEq != 0) && (here->INDposNode != 0))
            {
                i = here->INDibrPosptr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->INDibrPosptrStructPtr = matched ;
                here->INDibrPosptr = matched->CSC_LinearDynamic ;
            }

            if ((here->INDbrEq != 0) && (here->INDbrEq != 0))
            {
                i = here->INDibrIbrptr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->INDibrIbrptrStructPtr = matched ;
                here->INDibrIbrptr = matched->CSC_LinearDynamic ;
            }

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
            if ((here->INDposNode != 0) && (here->INDbrEq != 0))
                here->INDposIbrptr = here->INDposIbrptrStructPtr->CSC_Complex_LinearDynamic ;

            if ((here->INDnegNode != 0) && (here->INDbrEq != 0))
                here->INDnegIbrptr = here->INDnegIbrptrStructPtr->CSC_Complex_LinearDynamic ;

            if ((here->INDbrEq != 0) && (here->INDnegNode != 0))
                here->INDibrNegptr = here->INDibrNegptrStructPtr->CSC_Complex_LinearDynamic ;

            if ((here->INDbrEq != 0) && (here->INDposNode != 0))
                here->INDibrPosptr = here->INDibrPosptrStructPtr->CSC_Complex_LinearDynamic ;

            if ((here->INDbrEq != 0) && (here->INDbrEq != 0))
                here->INDibrIbrptr = here->INDibrIbrptrStructPtr->CSC_Complex_LinearDynamic ;

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
            if ((here->INDposNode != 0) && (here->INDbrEq != 0))
                here->INDposIbrptr = here->INDposIbrptrStructPtr->CSC_LinearDynamic ;

            if ((here->INDnegNode != 0) && (here->INDbrEq != 0))
                here->INDnegIbrptr = here->INDnegIbrptrStructPtr->CSC_LinearDynamic ;

            if ((here->INDbrEq != 0) && (here->INDnegNode != 0))
                here->INDibrNegptr = here->INDibrNegptrStructPtr->CSC_LinearDynamic ;

            if ((here->INDbrEq != 0) && (here->INDposNode != 0))
                here->INDibrPosptr = here->INDibrPosptrStructPtr->CSC_LinearDynamic ;

            if ((here->INDbrEq != 0) && (here->INDbrEq != 0))
                here->INDibrIbrptr = here->INDibrIbrptrStructPtr->CSC_LinearDynamic ;

        }
    }

    return (OK) ;
}
