/**********
Author: 2013 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "mesdefs.h"
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
MESbindCSC (GENmodel *inModel, CKTcircuit *ckt)
{
    MESmodel *model = (MESmodel *)inModel ;
    MESinstance *here ;
    double *i ;
    BindElement *matched, *BindStruct ;
    size_t nz ;

    BindStruct = ckt->CKTmatrix->CKTbindStruct ;
    nz = (size_t)ckt->CKTmatrix->CKTklunz ;

    /* loop through all the MES models */
    for ( ; model != NULL ; model = model->MESnextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->MESinstances ; here != NULL ; here = here->MESnextInstance)
        {
            if ((here->MESdrainNode != 0) && (here->MESdrainPrimeNode != 0))
            {
                i = here->MESdrainDrainPrimePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MESdrainDrainPrimeStructPtr = matched ;
                here->MESdrainDrainPrimePtr = matched->CSC ;
            }

            if ((here->MESgateNode != 0) && (here->MESdrainPrimeNode != 0))
            {
                i = here->MESgateDrainPrimePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MESgateDrainPrimeStructPtr = matched ;
                here->MESgateDrainPrimePtr = matched->CSC ;
            }

            if ((here->MESgateNode != 0) && (here->MESsourcePrimeNode != 0))
            {
                i = here->MESgateSourcePrimePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MESgateSourcePrimeStructPtr = matched ;
                here->MESgateSourcePrimePtr = matched->CSC ;
            }

            if ((here->MESsourceNode != 0) && (here->MESsourcePrimeNode != 0))
            {
                i = here->MESsourceSourcePrimePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MESsourceSourcePrimeStructPtr = matched ;
                here->MESsourceSourcePrimePtr = matched->CSC ;
            }

            if ((here->MESdrainPrimeNode != 0) && (here->MESdrainNode != 0))
            {
                i = here->MESdrainPrimeDrainPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MESdrainPrimeDrainStructPtr = matched ;
                here->MESdrainPrimeDrainPtr = matched->CSC ;
            }

            if ((here->MESdrainPrimeNode != 0) && (here->MESgateNode != 0))
            {
                i = here->MESdrainPrimeGatePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MESdrainPrimeGateStructPtr = matched ;
                here->MESdrainPrimeGatePtr = matched->CSC ;
            }

            if ((here->MESdrainPrimeNode != 0) && (here->MESsourcePrimeNode != 0))
            {
                i = here->MESdrainPrimeSourcePrimePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MESdrainPrimeSourcePrimeStructPtr = matched ;
                here->MESdrainPrimeSourcePrimePtr = matched->CSC ;
            }

            if ((here->MESsourcePrimeNode != 0) && (here->MESgateNode != 0))
            {
                i = here->MESsourcePrimeGatePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MESsourcePrimeGateStructPtr = matched ;
                here->MESsourcePrimeGatePtr = matched->CSC ;
            }

            if ((here->MESsourcePrimeNode != 0) && (here->MESsourceNode != 0))
            {
                i = here->MESsourcePrimeSourcePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MESsourcePrimeSourceStructPtr = matched ;
                here->MESsourcePrimeSourcePtr = matched->CSC ;
            }

            if ((here->MESsourcePrimeNode != 0) && (here->MESdrainPrimeNode != 0))
            {
                i = here->MESsourcePrimeDrainPrimePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MESsourcePrimeDrainPrimeStructPtr = matched ;
                here->MESsourcePrimeDrainPrimePtr = matched->CSC ;
            }

            if ((here->MESdrainNode != 0) && (here->MESdrainNode != 0))
            {
                i = here->MESdrainDrainPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MESdrainDrainStructPtr = matched ;
                here->MESdrainDrainPtr = matched->CSC ;
            }

            if ((here->MESgateNode != 0) && (here->MESgateNode != 0))
            {
                i = here->MESgateGatePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MESgateGateStructPtr = matched ;
                here->MESgateGatePtr = matched->CSC ;
            }

            if ((here->MESsourceNode != 0) && (here->MESsourceNode != 0))
            {
                i = here->MESsourceSourcePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MESsourceSourceStructPtr = matched ;
                here->MESsourceSourcePtr = matched->CSC ;
            }

            if ((here->MESdrainPrimeNode != 0) && (here->MESdrainPrimeNode != 0))
            {
                i = here->MESdrainPrimeDrainPrimePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MESdrainPrimeDrainPrimeStructPtr = matched ;
                here->MESdrainPrimeDrainPrimePtr = matched->CSC ;
            }

            if ((here->MESsourcePrimeNode != 0) && (here->MESsourcePrimeNode != 0))
            {
                i = here->MESsourcePrimeSourcePrimePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MESsourcePrimeSourcePrimeStructPtr = matched ;
                here->MESsourcePrimeSourcePrimePtr = matched->CSC ;
            }

        }
    }

    return (OK) ;
}

int
MESbindCSCComplex (GENmodel *inModel, CKTcircuit *ckt)
{
    MESmodel *model = (MESmodel *)inModel ;
    MESinstance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the MES models */
    for ( ; model != NULL ; model = model->MESnextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->MESinstances ; here != NULL ; here = here->MESnextInstance)
        {
            if ((here->MESdrainNode != 0) && (here->MESdrainPrimeNode != 0))
                here->MESdrainDrainPrimePtr = here->MESdrainDrainPrimeStructPtr->CSC_Complex ;

            if ((here->MESgateNode != 0) && (here->MESdrainPrimeNode != 0))
                here->MESgateDrainPrimePtr = here->MESgateDrainPrimeStructPtr->CSC_Complex ;

            if ((here->MESgateNode != 0) && (here->MESsourcePrimeNode != 0))
                here->MESgateSourcePrimePtr = here->MESgateSourcePrimeStructPtr->CSC_Complex ;

            if ((here->MESsourceNode != 0) && (here->MESsourcePrimeNode != 0))
                here->MESsourceSourcePrimePtr = here->MESsourceSourcePrimeStructPtr->CSC_Complex ;

            if ((here->MESdrainPrimeNode != 0) && (here->MESdrainNode != 0))
                here->MESdrainPrimeDrainPtr = here->MESdrainPrimeDrainStructPtr->CSC_Complex ;

            if ((here->MESdrainPrimeNode != 0) && (here->MESgateNode != 0))
                here->MESdrainPrimeGatePtr = here->MESdrainPrimeGateStructPtr->CSC_Complex ;

            if ((here->MESdrainPrimeNode != 0) && (here->MESsourcePrimeNode != 0))
                here->MESdrainPrimeSourcePrimePtr = here->MESdrainPrimeSourcePrimeStructPtr->CSC_Complex ;

            if ((here->MESsourcePrimeNode != 0) && (here->MESgateNode != 0))
                here->MESsourcePrimeGatePtr = here->MESsourcePrimeGateStructPtr->CSC_Complex ;

            if ((here->MESsourcePrimeNode != 0) && (here->MESsourceNode != 0))
                here->MESsourcePrimeSourcePtr = here->MESsourcePrimeSourceStructPtr->CSC_Complex ;

            if ((here->MESsourcePrimeNode != 0) && (here->MESdrainPrimeNode != 0))
                here->MESsourcePrimeDrainPrimePtr = here->MESsourcePrimeDrainPrimeStructPtr->CSC_Complex ;

            if ((here->MESdrainNode != 0) && (here->MESdrainNode != 0))
                here->MESdrainDrainPtr = here->MESdrainDrainStructPtr->CSC_Complex ;

            if ((here->MESgateNode != 0) && (here->MESgateNode != 0))
                here->MESgateGatePtr = here->MESgateGateStructPtr->CSC_Complex ;

            if ((here->MESsourceNode != 0) && (here->MESsourceNode != 0))
                here->MESsourceSourcePtr = here->MESsourceSourceStructPtr->CSC_Complex ;

            if ((here->MESdrainPrimeNode != 0) && (here->MESdrainPrimeNode != 0))
                here->MESdrainPrimeDrainPrimePtr = here->MESdrainPrimeDrainPrimeStructPtr->CSC_Complex ;

            if ((here->MESsourcePrimeNode != 0) && (here->MESsourcePrimeNode != 0))
                here->MESsourcePrimeSourcePrimePtr = here->MESsourcePrimeSourcePrimeStructPtr->CSC_Complex ;

        }
    }

    return (OK) ;
}

int
MESbindCSCComplexToReal (GENmodel *inModel, CKTcircuit *ckt)
{
    MESmodel *model = (MESmodel *)inModel ;
    MESinstance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the MES models */
    for ( ; model != NULL ; model = model->MESnextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->MESinstances ; here != NULL ; here = here->MESnextInstance)
        {
            if ((here->MESdrainNode != 0) && (here->MESdrainPrimeNode != 0))
                here->MESdrainDrainPrimePtr = here->MESdrainDrainPrimeStructPtr->CSC ;

            if ((here->MESgateNode != 0) && (here->MESdrainPrimeNode != 0))
                here->MESgateDrainPrimePtr = here->MESgateDrainPrimeStructPtr->CSC ;

            if ((here->MESgateNode != 0) && (here->MESsourcePrimeNode != 0))
                here->MESgateSourcePrimePtr = here->MESgateSourcePrimeStructPtr->CSC ;

            if ((here->MESsourceNode != 0) && (here->MESsourcePrimeNode != 0))
                here->MESsourceSourcePrimePtr = here->MESsourceSourcePrimeStructPtr->CSC ;

            if ((here->MESdrainPrimeNode != 0) && (here->MESdrainNode != 0))
                here->MESdrainPrimeDrainPtr = here->MESdrainPrimeDrainStructPtr->CSC ;

            if ((here->MESdrainPrimeNode != 0) && (here->MESgateNode != 0))
                here->MESdrainPrimeGatePtr = here->MESdrainPrimeGateStructPtr->CSC ;

            if ((here->MESdrainPrimeNode != 0) && (here->MESsourcePrimeNode != 0))
                here->MESdrainPrimeSourcePrimePtr = here->MESdrainPrimeSourcePrimeStructPtr->CSC ;

            if ((here->MESsourcePrimeNode != 0) && (here->MESgateNode != 0))
                here->MESsourcePrimeGatePtr = here->MESsourcePrimeGateStructPtr->CSC ;

            if ((here->MESsourcePrimeNode != 0) && (here->MESsourceNode != 0))
                here->MESsourcePrimeSourcePtr = here->MESsourcePrimeSourceStructPtr->CSC ;

            if ((here->MESsourcePrimeNode != 0) && (here->MESdrainPrimeNode != 0))
                here->MESsourcePrimeDrainPrimePtr = here->MESsourcePrimeDrainPrimeStructPtr->CSC ;

            if ((here->MESdrainNode != 0) && (here->MESdrainNode != 0))
                here->MESdrainDrainPtr = here->MESdrainDrainStructPtr->CSC ;

            if ((here->MESgateNode != 0) && (here->MESgateNode != 0))
                here->MESgateGatePtr = here->MESgateGateStructPtr->CSC ;

            if ((here->MESsourceNode != 0) && (here->MESsourceNode != 0))
                here->MESsourceSourcePtr = here->MESsourceSourceStructPtr->CSC ;

            if ((here->MESdrainPrimeNode != 0) && (here->MESdrainPrimeNode != 0))
                here->MESdrainPrimeDrainPrimePtr = here->MESdrainPrimeDrainPrimeStructPtr->CSC ;

            if ((here->MESsourcePrimeNode != 0) && (here->MESsourcePrimeNode != 0))
                here->MESsourcePrimeSourcePrimePtr = here->MESsourcePrimeSourcePrimeStructPtr->CSC ;

        }
    }

    return (OK) ;
}
