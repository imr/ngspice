/**********
Author: 2013 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "mesadefs.h"
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
MESAbindCSC (GENmodel *inModel, CKTcircuit *ckt)
{
    MESAmodel *model = (MESAmodel *)inModel ;
    MESAinstance *here ;
    double *i ;
    BindElement *matched, *BindStruct ;
    size_t nz ;

    BindStruct = ckt->CKTmatrix->CKTbindStruct ;
    nz = (size_t)ckt->CKTmatrix->CKTklunz ;

    /* loop through all the MESA models */
    for ( ; model != NULL ; model = model->MESAnextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->MESAinstances ; here != NULL ; here = here->MESAnextInstance)
        {
            if ((here->MESAdrainNode != 0) && (here->MESAdrainNode != 0))
            {
                i = here->MESAdrainDrainPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MESAdrainDrainStructPtr = matched ;
                here->MESAdrainDrainPtr = matched->CSC ;
            }

            if ((here->MESAdrainPrimeNode != 0) && (here->MESAdrainPrimeNode != 0))
            {
                i = here->MESAdrainPrimeDrainPrimePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MESAdrainPrimeDrainPrimeStructPtr = matched ;
                here->MESAdrainPrimeDrainPrimePtr = matched->CSC ;
            }

            if ((here->MESAdrainPrmPrmNode != 0) && (here->MESAdrainPrmPrmNode != 0))
            {
                i = here->MESAdrainPrmPrmDrainPrmPrmPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MESAdrainPrmPrmDrainPrmPrmStructPtr = matched ;
                here->MESAdrainPrmPrmDrainPrmPrmPtr = matched->CSC ;
            }

            if ((here->MESAgateNode != 0) && (here->MESAgateNode != 0))
            {
                i = here->MESAgateGatePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MESAgateGateStructPtr = matched ;
                here->MESAgateGatePtr = matched->CSC ;
            }

            if ((here->MESAgatePrimeNode != 0) && (here->MESAgatePrimeNode != 0))
            {
                i = here->MESAgatePrimeGatePrimePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MESAgatePrimeGatePrimeStructPtr = matched ;
                here->MESAgatePrimeGatePrimePtr = matched->CSC ;
            }

            if ((here->MESAsourceNode != 0) && (here->MESAsourceNode != 0))
            {
                i = here->MESAsourceSourcePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MESAsourceSourceStructPtr = matched ;
                here->MESAsourceSourcePtr = matched->CSC ;
            }

            if ((here->MESAsourcePrimeNode != 0) && (here->MESAsourcePrimeNode != 0))
            {
                i = here->MESAsourcePrimeSourcePrimePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MESAsourcePrimeSourcePrimeStructPtr = matched ;
                here->MESAsourcePrimeSourcePrimePtr = matched->CSC ;
            }

            if ((here->MESAsourcePrmPrmNode != 0) && (here->MESAsourcePrmPrmNode != 0))
            {
                i = here->MESAsourcePrmPrmSourcePrmPrmPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MESAsourcePrmPrmSourcePrmPrmStructPtr = matched ;
                here->MESAsourcePrmPrmSourcePrmPrmPtr = matched->CSC ;
            }

            if ((here->MESAdrainNode != 0) && (here->MESAdrainPrimeNode != 0))
            {
                i = here->MESAdrainDrainPrimePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MESAdrainDrainPrimeStructPtr = matched ;
                here->MESAdrainDrainPrimePtr = matched->CSC ;
            }

            if ((here->MESAdrainPrimeNode != 0) && (here->MESAdrainNode != 0))
            {
                i = here->MESAdrainPrimeDrainPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MESAdrainPrimeDrainStructPtr = matched ;
                here->MESAdrainPrimeDrainPtr = matched->CSC ;
            }

            if ((here->MESAgatePrimeNode != 0) && (here->MESAdrainPrimeNode != 0))
            {
                i = here->MESAgatePrimeDrainPrimePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MESAgatePrimeDrainPrimeStructPtr = matched ;
                here->MESAgatePrimeDrainPrimePtr = matched->CSC ;
            }

            if ((here->MESAdrainPrimeNode != 0) && (here->MESAgatePrimeNode != 0))
            {
                i = here->MESAdrainPrimeGatePrimePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MESAdrainPrimeGatePrimeStructPtr = matched ;
                here->MESAdrainPrimeGatePrimePtr = matched->CSC ;
            }

            if ((here->MESAgatePrimeNode != 0) && (here->MESAsourcePrimeNode != 0))
            {
                i = here->MESAgatePrimeSourcePrimePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MESAgatePrimeSourcePrimeStructPtr = matched ;
                here->MESAgatePrimeSourcePrimePtr = matched->CSC ;
            }

            if ((here->MESAsourcePrimeNode != 0) && (here->MESAgatePrimeNode != 0))
            {
                i = here->MESAsourcePrimeGatePrimePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MESAsourcePrimeGatePrimeStructPtr = matched ;
                here->MESAsourcePrimeGatePrimePtr = matched->CSC ;
            }

            if ((here->MESAsourceNode != 0) && (here->MESAsourcePrimeNode != 0))
            {
                i = here->MESAsourceSourcePrimePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MESAsourceSourcePrimeStructPtr = matched ;
                here->MESAsourceSourcePrimePtr = matched->CSC ;
            }

            if ((here->MESAsourcePrimeNode != 0) && (here->MESAsourceNode != 0))
            {
                i = here->MESAsourcePrimeSourcePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MESAsourcePrimeSourceStructPtr = matched ;
                here->MESAsourcePrimeSourcePtr = matched->CSC ;
            }

            if ((here->MESAdrainPrimeNode != 0) && (here->MESAsourcePrimeNode != 0))
            {
                i = here->MESAdrainPrimeSourcePrimePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MESAdrainPrimeSourcePrimeStructPtr = matched ;
                here->MESAdrainPrimeSourcePrimePtr = matched->CSC ;
            }

            if ((here->MESAsourcePrimeNode != 0) && (here->MESAdrainPrimeNode != 0))
            {
                i = here->MESAsourcePrimeDrainPrimePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MESAsourcePrimeDrainPrimeStructPtr = matched ;
                here->MESAsourcePrimeDrainPrimePtr = matched->CSC ;
            }

            if ((here->MESAgatePrimeNode != 0) && (here->MESAgateNode != 0))
            {
                i = here->MESAgatePrimeGatePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MESAgatePrimeGateStructPtr = matched ;
                here->MESAgatePrimeGatePtr = matched->CSC ;
            }

            if ((here->MESAgateNode != 0) && (here->MESAgatePrimeNode != 0))
            {
                i = here->MESAgateGatePrimePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MESAgateGatePrimeStructPtr = matched ;
                here->MESAgateGatePrimePtr = matched->CSC ;
            }

            if ((here->MESAsourcePrmPrmNode != 0) && (here->MESAsourcePrimeNode != 0))
            {
                i = here->MESAsourcePrmPrmSourcePrimePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MESAsourcePrmPrmSourcePrimeStructPtr = matched ;
                here->MESAsourcePrmPrmSourcePrimePtr = matched->CSC ;
            }

            if ((here->MESAsourcePrimeNode != 0) && (here->MESAsourcePrmPrmNode != 0))
            {
                i = here->MESAsourcePrimeSourcePrmPrmPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MESAsourcePrimeSourcePrmPrmStructPtr = matched ;
                here->MESAsourcePrimeSourcePrmPrmPtr = matched->CSC ;
            }

            if ((here->MESAsourcePrmPrmNode != 0) && (here->MESAgatePrimeNode != 0))
            {
                i = here->MESAsourcePrmPrmGatePrimePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MESAsourcePrmPrmGatePrimeStructPtr = matched ;
                here->MESAsourcePrmPrmGatePrimePtr = matched->CSC ;
            }

            if ((here->MESAgatePrimeNode != 0) && (here->MESAsourcePrmPrmNode != 0))
            {
                i = here->MESAgatePrimeSourcePrmPrmPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MESAgatePrimeSourcePrmPrmStructPtr = matched ;
                here->MESAgatePrimeSourcePrmPrmPtr = matched->CSC ;
            }

            if ((here->MESAdrainPrmPrmNode != 0) && (here->MESAdrainPrimeNode != 0))
            {
                i = here->MESAdrainPrmPrmDrainPrimePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MESAdrainPrmPrmDrainPrimeStructPtr = matched ;
                here->MESAdrainPrmPrmDrainPrimePtr = matched->CSC ;
            }

            if ((here->MESAdrainPrimeNode != 0) && (here->MESAdrainPrmPrmNode != 0))
            {
                i = here->MESAdrainPrimeDrainPrmPrmPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MESAdrainPrimeDrainPrmPrmStructPtr = matched ;
                here->MESAdrainPrimeDrainPrmPrmPtr = matched->CSC ;
            }

            if ((here->MESAdrainPrmPrmNode != 0) && (here->MESAgatePrimeNode != 0))
            {
                i = here->MESAdrainPrmPrmGatePrimePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MESAdrainPrmPrmGatePrimeStructPtr = matched ;
                here->MESAdrainPrmPrmGatePrimePtr = matched->CSC ;
            }

            if ((here->MESAgatePrimeNode != 0) && (here->MESAdrainPrmPrmNode != 0))
            {
                i = here->MESAgatePrimeDrainPrmPrmPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->MESAgatePrimeDrainPrmPrmStructPtr = matched ;
                here->MESAgatePrimeDrainPrmPrmPtr = matched->CSC ;
            }

        }
    }

    return (OK) ;
}

int
MESAbindCSCComplex (GENmodel *inModel, CKTcircuit *ckt)
{
    MESAmodel *model = (MESAmodel *)inModel ;
    MESAinstance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the MESA models */
    for ( ; model != NULL ; model = model->MESAnextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->MESAinstances ; here != NULL ; here = here->MESAnextInstance)
        {
            if ((here->MESAdrainNode != 0) && (here->MESAdrainNode != 0))
                here->MESAdrainDrainPtr = here->MESAdrainDrainStructPtr->CSC_Complex ;

            if ((here->MESAdrainPrimeNode != 0) && (here->MESAdrainPrimeNode != 0))
                here->MESAdrainPrimeDrainPrimePtr = here->MESAdrainPrimeDrainPrimeStructPtr->CSC_Complex ;

            if ((here->MESAdrainPrmPrmNode != 0) && (here->MESAdrainPrmPrmNode != 0))
                here->MESAdrainPrmPrmDrainPrmPrmPtr = here->MESAdrainPrmPrmDrainPrmPrmStructPtr->CSC_Complex ;

            if ((here->MESAgateNode != 0) && (here->MESAgateNode != 0))
                here->MESAgateGatePtr = here->MESAgateGateStructPtr->CSC_Complex ;

            if ((here->MESAgatePrimeNode != 0) && (here->MESAgatePrimeNode != 0))
                here->MESAgatePrimeGatePrimePtr = here->MESAgatePrimeGatePrimeStructPtr->CSC_Complex ;

            if ((here->MESAsourceNode != 0) && (here->MESAsourceNode != 0))
                here->MESAsourceSourcePtr = here->MESAsourceSourceStructPtr->CSC_Complex ;

            if ((here->MESAsourcePrimeNode != 0) && (here->MESAsourcePrimeNode != 0))
                here->MESAsourcePrimeSourcePrimePtr = here->MESAsourcePrimeSourcePrimeStructPtr->CSC_Complex ;

            if ((here->MESAsourcePrmPrmNode != 0) && (here->MESAsourcePrmPrmNode != 0))
                here->MESAsourcePrmPrmSourcePrmPrmPtr = here->MESAsourcePrmPrmSourcePrmPrmStructPtr->CSC_Complex ;

            if ((here->MESAdrainNode != 0) && (here->MESAdrainPrimeNode != 0))
                here->MESAdrainDrainPrimePtr = here->MESAdrainDrainPrimeStructPtr->CSC_Complex ;

            if ((here->MESAdrainPrimeNode != 0) && (here->MESAdrainNode != 0))
                here->MESAdrainPrimeDrainPtr = here->MESAdrainPrimeDrainStructPtr->CSC_Complex ;

            if ((here->MESAgatePrimeNode != 0) && (here->MESAdrainPrimeNode != 0))
                here->MESAgatePrimeDrainPrimePtr = here->MESAgatePrimeDrainPrimeStructPtr->CSC_Complex ;

            if ((here->MESAdrainPrimeNode != 0) && (here->MESAgatePrimeNode != 0))
                here->MESAdrainPrimeGatePrimePtr = here->MESAdrainPrimeGatePrimeStructPtr->CSC_Complex ;

            if ((here->MESAgatePrimeNode != 0) && (here->MESAsourcePrimeNode != 0))
                here->MESAgatePrimeSourcePrimePtr = here->MESAgatePrimeSourcePrimeStructPtr->CSC_Complex ;

            if ((here->MESAsourcePrimeNode != 0) && (here->MESAgatePrimeNode != 0))
                here->MESAsourcePrimeGatePrimePtr = here->MESAsourcePrimeGatePrimeStructPtr->CSC_Complex ;

            if ((here->MESAsourceNode != 0) && (here->MESAsourcePrimeNode != 0))
                here->MESAsourceSourcePrimePtr = here->MESAsourceSourcePrimeStructPtr->CSC_Complex ;

            if ((here->MESAsourcePrimeNode != 0) && (here->MESAsourceNode != 0))
                here->MESAsourcePrimeSourcePtr = here->MESAsourcePrimeSourceStructPtr->CSC_Complex ;

            if ((here->MESAdrainPrimeNode != 0) && (here->MESAsourcePrimeNode != 0))
                here->MESAdrainPrimeSourcePrimePtr = here->MESAdrainPrimeSourcePrimeStructPtr->CSC_Complex ;

            if ((here->MESAsourcePrimeNode != 0) && (here->MESAdrainPrimeNode != 0))
                here->MESAsourcePrimeDrainPrimePtr = here->MESAsourcePrimeDrainPrimeStructPtr->CSC_Complex ;

            if ((here->MESAgatePrimeNode != 0) && (here->MESAgateNode != 0))
                here->MESAgatePrimeGatePtr = here->MESAgatePrimeGateStructPtr->CSC_Complex ;

            if ((here->MESAgateNode != 0) && (here->MESAgatePrimeNode != 0))
                here->MESAgateGatePrimePtr = here->MESAgateGatePrimeStructPtr->CSC_Complex ;

            if ((here->MESAsourcePrmPrmNode != 0) && (here->MESAsourcePrimeNode != 0))
                here->MESAsourcePrmPrmSourcePrimePtr = here->MESAsourcePrmPrmSourcePrimeStructPtr->CSC_Complex ;

            if ((here->MESAsourcePrimeNode != 0) && (here->MESAsourcePrmPrmNode != 0))
                here->MESAsourcePrimeSourcePrmPrmPtr = here->MESAsourcePrimeSourcePrmPrmStructPtr->CSC_Complex ;

            if ((here->MESAsourcePrmPrmNode != 0) && (here->MESAgatePrimeNode != 0))
                here->MESAsourcePrmPrmGatePrimePtr = here->MESAsourcePrmPrmGatePrimeStructPtr->CSC_Complex ;

            if ((here->MESAgatePrimeNode != 0) && (here->MESAsourcePrmPrmNode != 0))
                here->MESAgatePrimeSourcePrmPrmPtr = here->MESAgatePrimeSourcePrmPrmStructPtr->CSC_Complex ;

            if ((here->MESAdrainPrmPrmNode != 0) && (here->MESAdrainPrimeNode != 0))
                here->MESAdrainPrmPrmDrainPrimePtr = here->MESAdrainPrmPrmDrainPrimeStructPtr->CSC_Complex ;

            if ((here->MESAdrainPrimeNode != 0) && (here->MESAdrainPrmPrmNode != 0))
                here->MESAdrainPrimeDrainPrmPrmPtr = here->MESAdrainPrimeDrainPrmPrmStructPtr->CSC_Complex ;

            if ((here->MESAdrainPrmPrmNode != 0) && (here->MESAgatePrimeNode != 0))
                here->MESAdrainPrmPrmGatePrimePtr = here->MESAdrainPrmPrmGatePrimeStructPtr->CSC_Complex ;

            if ((here->MESAgatePrimeNode != 0) && (here->MESAdrainPrmPrmNode != 0))
                here->MESAgatePrimeDrainPrmPrmPtr = here->MESAgatePrimeDrainPrmPrmStructPtr->CSC_Complex ;

        }
    }

    return (OK) ;
}

int
MESAbindCSCComplexToReal (GENmodel *inModel, CKTcircuit *ckt)
{
    MESAmodel *model = (MESAmodel *)inModel ;
    MESAinstance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the MESA models */
    for ( ; model != NULL ; model = model->MESAnextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->MESAinstances ; here != NULL ; here = here->MESAnextInstance)
        {
            if ((here->MESAdrainNode != 0) && (here->MESAdrainNode != 0))
                here->MESAdrainDrainPtr = here->MESAdrainDrainStructPtr->CSC ;

            if ((here->MESAdrainPrimeNode != 0) && (here->MESAdrainPrimeNode != 0))
                here->MESAdrainPrimeDrainPrimePtr = here->MESAdrainPrimeDrainPrimeStructPtr->CSC ;

            if ((here->MESAdrainPrmPrmNode != 0) && (here->MESAdrainPrmPrmNode != 0))
                here->MESAdrainPrmPrmDrainPrmPrmPtr = here->MESAdrainPrmPrmDrainPrmPrmStructPtr->CSC ;

            if ((here->MESAgateNode != 0) && (here->MESAgateNode != 0))
                here->MESAgateGatePtr = here->MESAgateGateStructPtr->CSC ;

            if ((here->MESAgatePrimeNode != 0) && (here->MESAgatePrimeNode != 0))
                here->MESAgatePrimeGatePrimePtr = here->MESAgatePrimeGatePrimeStructPtr->CSC ;

            if ((here->MESAsourceNode != 0) && (here->MESAsourceNode != 0))
                here->MESAsourceSourcePtr = here->MESAsourceSourceStructPtr->CSC ;

            if ((here->MESAsourcePrimeNode != 0) && (here->MESAsourcePrimeNode != 0))
                here->MESAsourcePrimeSourcePrimePtr = here->MESAsourcePrimeSourcePrimeStructPtr->CSC ;

            if ((here->MESAsourcePrmPrmNode != 0) && (here->MESAsourcePrmPrmNode != 0))
                here->MESAsourcePrmPrmSourcePrmPrmPtr = here->MESAsourcePrmPrmSourcePrmPrmStructPtr->CSC ;

            if ((here->MESAdrainNode != 0) && (here->MESAdrainPrimeNode != 0))
                here->MESAdrainDrainPrimePtr = here->MESAdrainDrainPrimeStructPtr->CSC ;

            if ((here->MESAdrainPrimeNode != 0) && (here->MESAdrainNode != 0))
                here->MESAdrainPrimeDrainPtr = here->MESAdrainPrimeDrainStructPtr->CSC ;

            if ((here->MESAgatePrimeNode != 0) && (here->MESAdrainPrimeNode != 0))
                here->MESAgatePrimeDrainPrimePtr = here->MESAgatePrimeDrainPrimeStructPtr->CSC ;

            if ((here->MESAdrainPrimeNode != 0) && (here->MESAgatePrimeNode != 0))
                here->MESAdrainPrimeGatePrimePtr = here->MESAdrainPrimeGatePrimeStructPtr->CSC ;

            if ((here->MESAgatePrimeNode != 0) && (here->MESAsourcePrimeNode != 0))
                here->MESAgatePrimeSourcePrimePtr = here->MESAgatePrimeSourcePrimeStructPtr->CSC ;

            if ((here->MESAsourcePrimeNode != 0) && (here->MESAgatePrimeNode != 0))
                here->MESAsourcePrimeGatePrimePtr = here->MESAsourcePrimeGatePrimeStructPtr->CSC ;

            if ((here->MESAsourceNode != 0) && (here->MESAsourcePrimeNode != 0))
                here->MESAsourceSourcePrimePtr = here->MESAsourceSourcePrimeStructPtr->CSC ;

            if ((here->MESAsourcePrimeNode != 0) && (here->MESAsourceNode != 0))
                here->MESAsourcePrimeSourcePtr = here->MESAsourcePrimeSourceStructPtr->CSC ;

            if ((here->MESAdrainPrimeNode != 0) && (here->MESAsourcePrimeNode != 0))
                here->MESAdrainPrimeSourcePrimePtr = here->MESAdrainPrimeSourcePrimeStructPtr->CSC ;

            if ((here->MESAsourcePrimeNode != 0) && (here->MESAdrainPrimeNode != 0))
                here->MESAsourcePrimeDrainPrimePtr = here->MESAsourcePrimeDrainPrimeStructPtr->CSC ;

            if ((here->MESAgatePrimeNode != 0) && (here->MESAgateNode != 0))
                here->MESAgatePrimeGatePtr = here->MESAgatePrimeGateStructPtr->CSC ;

            if ((here->MESAgateNode != 0) && (here->MESAgatePrimeNode != 0))
                here->MESAgateGatePrimePtr = here->MESAgateGatePrimeStructPtr->CSC ;

            if ((here->MESAsourcePrmPrmNode != 0) && (here->MESAsourcePrimeNode != 0))
                here->MESAsourcePrmPrmSourcePrimePtr = here->MESAsourcePrmPrmSourcePrimeStructPtr->CSC ;

            if ((here->MESAsourcePrimeNode != 0) && (here->MESAsourcePrmPrmNode != 0))
                here->MESAsourcePrimeSourcePrmPrmPtr = here->MESAsourcePrimeSourcePrmPrmStructPtr->CSC ;

            if ((here->MESAsourcePrmPrmNode != 0) && (here->MESAgatePrimeNode != 0))
                here->MESAsourcePrmPrmGatePrimePtr = here->MESAsourcePrmPrmGatePrimeStructPtr->CSC ;

            if ((here->MESAgatePrimeNode != 0) && (here->MESAsourcePrmPrmNode != 0))
                here->MESAgatePrimeSourcePrmPrmPtr = here->MESAgatePrimeSourcePrmPrmStructPtr->CSC ;

            if ((here->MESAdrainPrmPrmNode != 0) && (here->MESAdrainPrimeNode != 0))
                here->MESAdrainPrmPrmDrainPrimePtr = here->MESAdrainPrmPrmDrainPrimeStructPtr->CSC ;

            if ((here->MESAdrainPrimeNode != 0) && (here->MESAdrainPrmPrmNode != 0))
                here->MESAdrainPrimeDrainPrmPrmPtr = here->MESAdrainPrimeDrainPrmPrmStructPtr->CSC ;

            if ((here->MESAdrainPrmPrmNode != 0) && (here->MESAgatePrimeNode != 0))
                here->MESAdrainPrmPrmGatePrimePtr = here->MESAdrainPrmPrmGatePrimeStructPtr->CSC ;

            if ((here->MESAgatePrimeNode != 0) && (here->MESAdrainPrmPrmNode != 0))
                here->MESAgatePrimeDrainPrmPrmPtr = here->MESAgatePrimeDrainPrmPrmStructPtr->CSC ;

        }
    }

    return (OK) ;
}
