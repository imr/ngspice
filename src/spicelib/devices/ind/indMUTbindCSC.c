/**********
Author: 2012 Francesco Lannutti
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

    /* loop through all the inductor models */
    for ( ; model != NULL ; model = model->INDnextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->INDinstances ; here != NULL ; here = here->INDnextInstance)
        {
            if ((here->INDposNode != 0) && (here->INDbrEq != 0))
            {
                i = here->INDposIbrptr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->INDposIbrStructPtr = matched ;
                here->INDposIbrptr = matched->CSC ;
            }

            if ((here->INDnegNode != 0) && (here->INDbrEq != 0))
            {
                i = here->INDnegIbrptr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->INDnegIbrStructPtr = matched ;
                here->INDnegIbrptr = matched->CSC ;
            }

            if ((here->INDbrEq != 0) && (here->INDnegNode != 0))
            {
                i = here->INDibrNegptr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->INDibrNegStructPtr = matched ;
                here->INDibrNegptr = matched->CSC ;
            }

            if ((here->INDbrEq != 0) && (here->INDposNode != 0))
            {
                i = here->INDibrPosptr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->INDibrPosStructPtr = matched ;
                here->INDibrPosptr = matched->CSC ;
            }

            if ((here->INDbrEq != 0) && (here->INDbrEq != 0))
            {
                i = here->INDibrIbrptr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->INDibrIbrStructPtr = matched ;
                here->INDibrIbrptr = matched->CSC ;
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

    /* loop through all the inductor models */
    for ( ; model != NULL ; model = model->INDnextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->INDinstances ; here != NULL ; here = here->INDnextInstance)
        {
            if ((here->INDposNode != 0) && (here->INDbrEq != 0))
                here->INDposIbrptr = here->INDposIbrStructPtr->CSC_Complex ;

            if ((here->INDnegNode != 0) && (here->INDbrEq != 0))
                here->INDnegIbrptr = here->INDnegIbrStructPtr->CSC_Complex ;

            if ((here->INDbrEq != 0) && (here->INDnegNode != 0))
                here->INDibrNegptr = here->INDibrNegStructPtr->CSC_Complex ;

            if ((here->INDbrEq != 0) && (here->INDposNode != 0))
                here->INDibrPosptr = here->INDibrPosStructPtr->CSC_Complex ;

            if ((here->INDbrEq != 0) && (here->INDbrEq != 0))
                here->INDibrIbrptr = here->INDibrIbrStructPtr->CSC_Complex ;
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

    /* loop through all the inductor models */
    for ( ; model != NULL ; model = model->INDnextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->INDinstances ; here != NULL ; here = here->INDnextInstance)
        {
            if ((here->INDposNode != 0) && (here->INDbrEq != 0))
                here->INDposIbrptr = here->INDposIbrStructPtr->CSC ;

            if ((here->INDnegNode != 0) && (here->INDbrEq != 0))
                here->INDnegIbrptr = here->INDnegIbrStructPtr->CSC ;

            if ((here->INDbrEq != 0) && (here->INDnegNode != 0))
                here->INDibrNegptr = here->INDibrNegStructPtr->CSC ;

            if ((here->INDbrEq != 0) && (here->INDposNode != 0))
                here->INDibrPosptr = here->INDibrPosStructPtr->CSC ;

            if ((here->INDbrEq != 0) && (here->INDbrEq != 0))
                here->INDibrIbrptr = here->INDibrIbrStructPtr->CSC ;
        }
    }

    return (OK) ;
}

#ifdef MUTUAL
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

    /* loop through all the mutual inductor models */
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

    /* loop through all the mutual inductor models */
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

    /* loop through all the mutual inductor models */
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
#endif
