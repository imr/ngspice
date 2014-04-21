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
MUTbindCSC (GENmodel *inModel, CKTcircuit *ckt)
{
    MUTmodel *model = (MUTmodel *)inModel ;
    MUTinstance *here ;
    double *i ;
    BindElement *matched, *BindStruct ;
    size_t nz ;

    BindStruct = ckt->CKTmatrix->CKTbindStruct ;
    nz = (size_t)ckt->CKTmatrix->CKTklunz ;

    /* loop through all the MUT models */
    for ( ; model != NULL ; model = model->MUTnextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->MUTinstances ; here != NULL ; here = here->MUTnextInstance)
        {
            if ((here->MUTind1->INDbrEq != 0) && (here->MUTind2->INDbrEq != 0))
            {
                i = here->MUTbr1br2 ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MUTbr1br2StructPtr = matched ;
                here->MUTbr1br2 = matched->CSC ;
            }

            if ((here->MUTind2->INDbrEq != 0) && (here->MUTind1->INDbrEq != 0))
            {
                i = here->MUTbr2br1 ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MUTbr2br1StructPtr = matched ;
                here->MUTbr2br1 = matched->CSC ;
            }

        }
    }

    return (OK) ;
}

int
MUTbindCSCComplex (GENmodel *inModel, CKTcircuit *ckt)
{
    MUTmodel *model = (MUTmodel *)inModel ;
    MUTinstance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the MUT models */
    for ( ; model != NULL ; model = model->MUTnextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->MUTinstances ; here != NULL ; here = here->MUTnextInstance)
        {
            if ((here->MUTind1->INDbrEq != 0) && (here->MUTind2->INDbrEq != 0))
                here->MUTbr1br2 = here->MUTbr1br2StructPtr->CSC_Complex ;

            if ((here->MUTind2->INDbrEq != 0) && (here->MUTind1->INDbrEq != 0))
                here->MUTbr2br1 = here->MUTbr2br1StructPtr->CSC_Complex ;

        }
    }

    return (OK) ;
}

int
MUTbindCSCComplexToReal (GENmodel *inModel, CKTcircuit *ckt)
{
    MUTmodel *model = (MUTmodel *)inModel ;
    MUTinstance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the MUT models */
    for ( ; model != NULL ; model = model->MUTnextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->MUTinstances ; here != NULL ; here = here->MUTnextInstance)
        {
            if ((here->MUTind1->INDbrEq != 0) && (here->MUTind2->INDbrEq != 0))
                here->MUTbr1br2 = here->MUTbr1br2StructPtr->CSC ;

            if ((here->MUTind2->INDbrEq != 0) && (here->MUTind1->INDbrEq != 0))
                here->MUTbr2br1 = here->MUTbr2br1StructPtr->CSC ;

        }
    }

    return (OK) ;
}
