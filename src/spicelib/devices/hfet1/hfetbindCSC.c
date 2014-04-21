/**********
Author: 2013 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "hfetdefs.h"
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
HFETAbindCSC (GENmodel *inModel, CKTcircuit *ckt)
{
    HFETAmodel *model = (HFETAmodel *)inModel ;
    HFETAinstance *here ;
    double *i ;
    BindElement *matched, *BindStruct ;
    size_t nz ;

    BindStruct = ckt->CKTmatrix->CKTbindStruct ;
    nz = (size_t)ckt->CKTmatrix->CKTklunz ;

    /* loop through all the HFETA models */
    for ( ; model != NULL ; model = model->HFETAnextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->HFETAinstances ; here != NULL ; here = here->HFETAnextInstance)
        {
            if ((here->HFETAdrainNode != 0) && (here->HFETAdrainPrimeNode != 0))
            {
                i = here->HFETAdrainDrainPrimePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->HFETAdrainDrainPrimeStructPtr = matched ;
                here->HFETAdrainDrainPrimePtr = matched->CSC ;
            }

            if ((here->HFETAgatePrimeNode != 0) && (here->HFETAdrainPrimeNode != 0))
            {
                i = here->HFETAgatePrimeDrainPrimePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->HFETAgatePrimeDrainPrimeStructPtr = matched ;
                here->HFETAgatePrimeDrainPrimePtr = matched->CSC ;
            }

            if ((here->HFETAgatePrimeNode != 0) && (here->HFETAsourcePrimeNode != 0))
            {
                i = here->HFETAgatePrimeSourcePrimePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->HFETAgatePrimeSourcePrimeStructPtr = matched ;
                here->HFETAgatePrimeSourcePrimePtr = matched->CSC ;
            }

            if ((here->HFETAsourceNode != 0) && (here->HFETAsourcePrimeNode != 0))
            {
                i = here->HFETAsourceSourcePrimePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->HFETAsourceSourcePrimeStructPtr = matched ;
                here->HFETAsourceSourcePrimePtr = matched->CSC ;
            }

            if ((here->HFETAdrainPrimeNode != 0) && (here->HFETAdrainNode != 0))
            {
                i = here->HFETAdrainPrimeDrainPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->HFETAdrainPrimeDrainStructPtr = matched ;
                here->HFETAdrainPrimeDrainPtr = matched->CSC ;
            }

            if ((here->HFETAdrainPrimeNode != 0) && (here->HFETAgatePrimeNode != 0))
            {
                i = here->HFETAdrainPrimeGatePrimePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->HFETAdrainPrimeGatePrimeStructPtr = matched ;
                here->HFETAdrainPrimeGatePrimePtr = matched->CSC ;
            }

            if ((here->HFETAdrainPrimeNode != 0) && (here->HFETAsourcePrimeNode != 0))
            {
                i = here->HFETAdrainPrimeSourcePrimePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->HFETAdrainPrimeSourcePrimeStructPtr = matched ;
                here->HFETAdrainPrimeSourcePrimePtr = matched->CSC ;
            }

            if ((here->HFETAsourcePrimeNode != 0) && (here->HFETAgatePrimeNode != 0))
            {
                i = here->HFETAsourcePrimeGatePrimePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->HFETAsourcePrimeGatePrimeStructPtr = matched ;
                here->HFETAsourcePrimeGatePrimePtr = matched->CSC ;
            }

            if ((here->HFETAsourcePrimeNode != 0) && (here->HFETAsourceNode != 0))
            {
                i = here->HFETAsourcePrimeSourcePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->HFETAsourcePrimeSourceStructPtr = matched ;
                here->HFETAsourcePrimeSourcePtr = matched->CSC ;
            }

            if ((here->HFETAsourcePrimeNode != 0) && (here->HFETAdrainPrimeNode != 0))
            {
                i = here->HFETAsourcePrimeDrainPrimePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->HFETAsourcePrimeDrainPrimeStructPtr = matched ;
                here->HFETAsourcePrimeDrainPrimePtr = matched->CSC ;
            }

            if ((here->HFETAdrainNode != 0) && (here->HFETAdrainNode != 0))
            {
                i = here->HFETAdrainDrainPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->HFETAdrainDrainStructPtr = matched ;
                here->HFETAdrainDrainPtr = matched->CSC ;
            }

            if ((here->HFETAgatePrimeNode != 0) && (here->HFETAgatePrimeNode != 0))
            {
                i = here->HFETAgatePrimeGatePrimePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->HFETAgatePrimeGatePrimeStructPtr = matched ;
                here->HFETAgatePrimeGatePrimePtr = matched->CSC ;
            }

            if ((here->HFETAsourceNode != 0) && (here->HFETAsourceNode != 0))
            {
                i = here->HFETAsourceSourcePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->HFETAsourceSourceStructPtr = matched ;
                here->HFETAsourceSourcePtr = matched->CSC ;
            }

            if ((here->HFETAdrainPrimeNode != 0) && (here->HFETAdrainPrimeNode != 0))
            {
                i = here->HFETAdrainPrimeDrainPrimePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->HFETAdrainPrimeDrainPrimeStructPtr = matched ;
                here->HFETAdrainPrimeDrainPrimePtr = matched->CSC ;
            }

            if ((here->HFETAsourcePrimeNode != 0) && (here->HFETAsourcePrimeNode != 0))
            {
                i = here->HFETAsourcePrimeSourcePrimePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->HFETAsourcePrimeSourcePrimeStructPtr = matched ;
                here->HFETAsourcePrimeSourcePrimePtr = matched->CSC ;
            }

            if ((here->HFETAdrainPrimeNode != 0) && (here->HFETAdrainPrmPrmNode != 0))
            {
                i = here->HFETAdrainPrimeDrainPrmPrmPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->HFETAdrainPrimeDrainPrmPrmStructPtr = matched ;
                here->HFETAdrainPrimeDrainPrmPrmPtr = matched->CSC ;
            }

            if ((here->HFETAdrainPrmPrmNode != 0) && (here->HFETAdrainPrimeNode != 0))
            {
                i = here->HFETAdrainPrmPrmDrainPrimePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->HFETAdrainPrmPrmDrainPrimeStructPtr = matched ;
                here->HFETAdrainPrmPrmDrainPrimePtr = matched->CSC ;
            }

            if ((here->HFETAdrainPrmPrmNode != 0) && (here->HFETAgatePrimeNode != 0))
            {
                i = here->HFETAdrainPrmPrmGatePrimePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->HFETAdrainPrmPrmGatePrimeStructPtr = matched ;
                here->HFETAdrainPrmPrmGatePrimePtr = matched->CSC ;
            }

            if ((here->HFETAgatePrimeNode != 0) && (here->HFETAdrainPrmPrmNode != 0))
            {
                i = here->HFETAgatePrimeDrainPrmPrmPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->HFETAgatePrimeDrainPrmPrmStructPtr = matched ;
                here->HFETAgatePrimeDrainPrmPrmPtr = matched->CSC ;
            }

            if ((here->HFETAdrainPrmPrmNode != 0) && (here->HFETAdrainPrmPrmNode != 0))
            {
                i = here->HFETAdrainPrmPrmDrainPrmPrmPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->HFETAdrainPrmPrmDrainPrmPrmStructPtr = matched ;
                here->HFETAdrainPrmPrmDrainPrmPrmPtr = matched->CSC ;
            }

            if ((here->HFETAsourcePrimeNode != 0) && (here->HFETAsourcePrmPrmNode != 0))
            {
                i = here->HFETAsourcePrimeSourcePrmPrmPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->HFETAsourcePrimeSourcePrmPrmStructPtr = matched ;
                here->HFETAsourcePrimeSourcePrmPrmPtr = matched->CSC ;
            }

            if ((here->HFETAsourcePrmPrmNode != 0) && (here->HFETAsourcePrimeNode != 0))
            {
                i = here->HFETAsourcePrmPrmSourcePrimePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->HFETAsourcePrmPrmSourcePrimeStructPtr = matched ;
                here->HFETAsourcePrmPrmSourcePrimePtr = matched->CSC ;
            }

            if ((here->HFETAsourcePrmPrmNode != 0) && (here->HFETAgatePrimeNode != 0))
            {
                i = here->HFETAsourcePrmPrmGatePrimePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->HFETAsourcePrmPrmGatePrimeStructPtr = matched ;
                here->HFETAsourcePrmPrmGatePrimePtr = matched->CSC ;
            }

            if ((here->HFETAgatePrimeNode != 0) && (here->HFETAsourcePrmPrmNode != 0))
            {
                i = here->HFETAgatePrimeSourcePrmPrmPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->HFETAgatePrimeSourcePrmPrmStructPtr = matched ;
                here->HFETAgatePrimeSourcePrmPrmPtr = matched->CSC ;
            }

            if ((here->HFETAsourcePrmPrmNode != 0) && (here->HFETAsourcePrmPrmNode != 0))
            {
                i = here->HFETAsourcePrmPrmSourcePrmPrmPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->HFETAsourcePrmPrmSourcePrmPrmStructPtr = matched ;
                here->HFETAsourcePrmPrmSourcePrmPrmPtr = matched->CSC ;
            }

            if ((here->HFETAgateNode != 0) && (here->HFETAgateNode != 0))
            {
                i = here->HFETAgateGatePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->HFETAgateGateStructPtr = matched ;
                here->HFETAgateGatePtr = matched->CSC ;
            }

            if ((here->HFETAgateNode != 0) && (here->HFETAgatePrimeNode != 0))
            {
                i = here->HFETAgateGatePrimePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->HFETAgateGatePrimeStructPtr = matched ;
                here->HFETAgateGatePrimePtr = matched->CSC ;
            }

            if ((here->HFETAgatePrimeNode != 0) && (here->HFETAgateNode != 0))
            {
                i = here->HFETAgatePrimeGatePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->HFETAgatePrimeGateStructPtr = matched ;
                here->HFETAgatePrimeGatePtr = matched->CSC ;
            }

        }
    }

    return (OK) ;
}

int
HFETAbindCSCComplex (GENmodel *inModel, CKTcircuit *ckt)
{
    HFETAmodel *model = (HFETAmodel *)inModel ;
    HFETAinstance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the HFETA models */
    for ( ; model != NULL ; model = model->HFETAnextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->HFETAinstances ; here != NULL ; here = here->HFETAnextInstance)
        {
            if ((here->HFETAdrainNode != 0) && (here->HFETAdrainPrimeNode != 0))
                here->HFETAdrainDrainPrimePtr = here->HFETAdrainDrainPrimeStructPtr->CSC_Complex ;

            if ((here->HFETAgatePrimeNode != 0) && (here->HFETAdrainPrimeNode != 0))
                here->HFETAgatePrimeDrainPrimePtr = here->HFETAgatePrimeDrainPrimeStructPtr->CSC_Complex ;

            if ((here->HFETAgatePrimeNode != 0) && (here->HFETAsourcePrimeNode != 0))
                here->HFETAgatePrimeSourcePrimePtr = here->HFETAgatePrimeSourcePrimeStructPtr->CSC_Complex ;

            if ((here->HFETAsourceNode != 0) && (here->HFETAsourcePrimeNode != 0))
                here->HFETAsourceSourcePrimePtr = here->HFETAsourceSourcePrimeStructPtr->CSC_Complex ;

            if ((here->HFETAdrainPrimeNode != 0) && (here->HFETAdrainNode != 0))
                here->HFETAdrainPrimeDrainPtr = here->HFETAdrainPrimeDrainStructPtr->CSC_Complex ;

            if ((here->HFETAdrainPrimeNode != 0) && (here->HFETAgatePrimeNode != 0))
                here->HFETAdrainPrimeGatePrimePtr = here->HFETAdrainPrimeGatePrimeStructPtr->CSC_Complex ;

            if ((here->HFETAdrainPrimeNode != 0) && (here->HFETAsourcePrimeNode != 0))
                here->HFETAdrainPrimeSourcePrimePtr = here->HFETAdrainPrimeSourcePrimeStructPtr->CSC_Complex ;

            if ((here->HFETAsourcePrimeNode != 0) && (here->HFETAgatePrimeNode != 0))
                here->HFETAsourcePrimeGatePrimePtr = here->HFETAsourcePrimeGatePrimeStructPtr->CSC_Complex ;

            if ((here->HFETAsourcePrimeNode != 0) && (here->HFETAsourceNode != 0))
                here->HFETAsourcePrimeSourcePtr = here->HFETAsourcePrimeSourceStructPtr->CSC_Complex ;

            if ((here->HFETAsourcePrimeNode != 0) && (here->HFETAdrainPrimeNode != 0))
                here->HFETAsourcePrimeDrainPrimePtr = here->HFETAsourcePrimeDrainPrimeStructPtr->CSC_Complex ;

            if ((here->HFETAdrainNode != 0) && (here->HFETAdrainNode != 0))
                here->HFETAdrainDrainPtr = here->HFETAdrainDrainStructPtr->CSC_Complex ;

            if ((here->HFETAgatePrimeNode != 0) && (here->HFETAgatePrimeNode != 0))
                here->HFETAgatePrimeGatePrimePtr = here->HFETAgatePrimeGatePrimeStructPtr->CSC_Complex ;

            if ((here->HFETAsourceNode != 0) && (here->HFETAsourceNode != 0))
                here->HFETAsourceSourcePtr = here->HFETAsourceSourceStructPtr->CSC_Complex ;

            if ((here->HFETAdrainPrimeNode != 0) && (here->HFETAdrainPrimeNode != 0))
                here->HFETAdrainPrimeDrainPrimePtr = here->HFETAdrainPrimeDrainPrimeStructPtr->CSC_Complex ;

            if ((here->HFETAsourcePrimeNode != 0) && (here->HFETAsourcePrimeNode != 0))
                here->HFETAsourcePrimeSourcePrimePtr = here->HFETAsourcePrimeSourcePrimeStructPtr->CSC_Complex ;

            if ((here->HFETAdrainPrimeNode != 0) && (here->HFETAdrainPrmPrmNode != 0))
                here->HFETAdrainPrimeDrainPrmPrmPtr = here->HFETAdrainPrimeDrainPrmPrmStructPtr->CSC_Complex ;

            if ((here->HFETAdrainPrmPrmNode != 0) && (here->HFETAdrainPrimeNode != 0))
                here->HFETAdrainPrmPrmDrainPrimePtr = here->HFETAdrainPrmPrmDrainPrimeStructPtr->CSC_Complex ;

            if ((here->HFETAdrainPrmPrmNode != 0) && (here->HFETAgatePrimeNode != 0))
                here->HFETAdrainPrmPrmGatePrimePtr = here->HFETAdrainPrmPrmGatePrimeStructPtr->CSC_Complex ;

            if ((here->HFETAgatePrimeNode != 0) && (here->HFETAdrainPrmPrmNode != 0))
                here->HFETAgatePrimeDrainPrmPrmPtr = here->HFETAgatePrimeDrainPrmPrmStructPtr->CSC_Complex ;

            if ((here->HFETAdrainPrmPrmNode != 0) && (here->HFETAdrainPrmPrmNode != 0))
                here->HFETAdrainPrmPrmDrainPrmPrmPtr = here->HFETAdrainPrmPrmDrainPrmPrmStructPtr->CSC_Complex ;

            if ((here->HFETAsourcePrimeNode != 0) && (here->HFETAsourcePrmPrmNode != 0))
                here->HFETAsourcePrimeSourcePrmPrmPtr = here->HFETAsourcePrimeSourcePrmPrmStructPtr->CSC_Complex ;

            if ((here->HFETAsourcePrmPrmNode != 0) && (here->HFETAsourcePrimeNode != 0))
                here->HFETAsourcePrmPrmSourcePrimePtr = here->HFETAsourcePrmPrmSourcePrimeStructPtr->CSC_Complex ;

            if ((here->HFETAsourcePrmPrmNode != 0) && (here->HFETAgatePrimeNode != 0))
                here->HFETAsourcePrmPrmGatePrimePtr = here->HFETAsourcePrmPrmGatePrimeStructPtr->CSC_Complex ;

            if ((here->HFETAgatePrimeNode != 0) && (here->HFETAsourcePrmPrmNode != 0))
                here->HFETAgatePrimeSourcePrmPrmPtr = here->HFETAgatePrimeSourcePrmPrmStructPtr->CSC_Complex ;

            if ((here->HFETAsourcePrmPrmNode != 0) && (here->HFETAsourcePrmPrmNode != 0))
                here->HFETAsourcePrmPrmSourcePrmPrmPtr = here->HFETAsourcePrmPrmSourcePrmPrmStructPtr->CSC_Complex ;

            if ((here->HFETAgateNode != 0) && (here->HFETAgateNode != 0))
                here->HFETAgateGatePtr = here->HFETAgateGateStructPtr->CSC_Complex ;

            if ((here->HFETAgateNode != 0) && (here->HFETAgatePrimeNode != 0))
                here->HFETAgateGatePrimePtr = here->HFETAgateGatePrimeStructPtr->CSC_Complex ;

            if ((here->HFETAgatePrimeNode != 0) && (here->HFETAgateNode != 0))
                here->HFETAgatePrimeGatePtr = here->HFETAgatePrimeGateStructPtr->CSC_Complex ;

        }
    }

    return (OK) ;
}

int
HFETAbindCSCComplexToReal (GENmodel *inModel, CKTcircuit *ckt)
{
    HFETAmodel *model = (HFETAmodel *)inModel ;
    HFETAinstance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the HFETA models */
    for ( ; model != NULL ; model = model->HFETAnextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->HFETAinstances ; here != NULL ; here = here->HFETAnextInstance)
        {
            if ((here->HFETAdrainNode != 0) && (here->HFETAdrainPrimeNode != 0))
                here->HFETAdrainDrainPrimePtr = here->HFETAdrainDrainPrimeStructPtr->CSC ;

            if ((here->HFETAgatePrimeNode != 0) && (here->HFETAdrainPrimeNode != 0))
                here->HFETAgatePrimeDrainPrimePtr = here->HFETAgatePrimeDrainPrimeStructPtr->CSC ;

            if ((here->HFETAgatePrimeNode != 0) && (here->HFETAsourcePrimeNode != 0))
                here->HFETAgatePrimeSourcePrimePtr = here->HFETAgatePrimeSourcePrimeStructPtr->CSC ;

            if ((here->HFETAsourceNode != 0) && (here->HFETAsourcePrimeNode != 0))
                here->HFETAsourceSourcePrimePtr = here->HFETAsourceSourcePrimeStructPtr->CSC ;

            if ((here->HFETAdrainPrimeNode != 0) && (here->HFETAdrainNode != 0))
                here->HFETAdrainPrimeDrainPtr = here->HFETAdrainPrimeDrainStructPtr->CSC ;

            if ((here->HFETAdrainPrimeNode != 0) && (here->HFETAgatePrimeNode != 0))
                here->HFETAdrainPrimeGatePrimePtr = here->HFETAdrainPrimeGatePrimeStructPtr->CSC ;

            if ((here->HFETAdrainPrimeNode != 0) && (here->HFETAsourcePrimeNode != 0))
                here->HFETAdrainPrimeSourcePrimePtr = here->HFETAdrainPrimeSourcePrimeStructPtr->CSC ;

            if ((here->HFETAsourcePrimeNode != 0) && (here->HFETAgatePrimeNode != 0))
                here->HFETAsourcePrimeGatePrimePtr = here->HFETAsourcePrimeGatePrimeStructPtr->CSC ;

            if ((here->HFETAsourcePrimeNode != 0) && (here->HFETAsourceNode != 0))
                here->HFETAsourcePrimeSourcePtr = here->HFETAsourcePrimeSourceStructPtr->CSC ;

            if ((here->HFETAsourcePrimeNode != 0) && (here->HFETAdrainPrimeNode != 0))
                here->HFETAsourcePrimeDrainPrimePtr = here->HFETAsourcePrimeDrainPrimeStructPtr->CSC ;

            if ((here->HFETAdrainNode != 0) && (here->HFETAdrainNode != 0))
                here->HFETAdrainDrainPtr = here->HFETAdrainDrainStructPtr->CSC ;

            if ((here->HFETAgatePrimeNode != 0) && (here->HFETAgatePrimeNode != 0))
                here->HFETAgatePrimeGatePrimePtr = here->HFETAgatePrimeGatePrimeStructPtr->CSC ;

            if ((here->HFETAsourceNode != 0) && (here->HFETAsourceNode != 0))
                here->HFETAsourceSourcePtr = here->HFETAsourceSourceStructPtr->CSC ;

            if ((here->HFETAdrainPrimeNode != 0) && (here->HFETAdrainPrimeNode != 0))
                here->HFETAdrainPrimeDrainPrimePtr = here->HFETAdrainPrimeDrainPrimeStructPtr->CSC ;

            if ((here->HFETAsourcePrimeNode != 0) && (here->HFETAsourcePrimeNode != 0))
                here->HFETAsourcePrimeSourcePrimePtr = here->HFETAsourcePrimeSourcePrimeStructPtr->CSC ;

            if ((here->HFETAdrainPrimeNode != 0) && (here->HFETAdrainPrmPrmNode != 0))
                here->HFETAdrainPrimeDrainPrmPrmPtr = here->HFETAdrainPrimeDrainPrmPrmStructPtr->CSC ;

            if ((here->HFETAdrainPrmPrmNode != 0) && (here->HFETAdrainPrimeNode != 0))
                here->HFETAdrainPrmPrmDrainPrimePtr = here->HFETAdrainPrmPrmDrainPrimeStructPtr->CSC ;

            if ((here->HFETAdrainPrmPrmNode != 0) && (here->HFETAgatePrimeNode != 0))
                here->HFETAdrainPrmPrmGatePrimePtr = here->HFETAdrainPrmPrmGatePrimeStructPtr->CSC ;

            if ((here->HFETAgatePrimeNode != 0) && (here->HFETAdrainPrmPrmNode != 0))
                here->HFETAgatePrimeDrainPrmPrmPtr = here->HFETAgatePrimeDrainPrmPrmStructPtr->CSC ;

            if ((here->HFETAdrainPrmPrmNode != 0) && (here->HFETAdrainPrmPrmNode != 0))
                here->HFETAdrainPrmPrmDrainPrmPrmPtr = here->HFETAdrainPrmPrmDrainPrmPrmStructPtr->CSC ;

            if ((here->HFETAsourcePrimeNode != 0) && (here->HFETAsourcePrmPrmNode != 0))
                here->HFETAsourcePrimeSourcePrmPrmPtr = here->HFETAsourcePrimeSourcePrmPrmStructPtr->CSC ;

            if ((here->HFETAsourcePrmPrmNode != 0) && (here->HFETAsourcePrimeNode != 0))
                here->HFETAsourcePrmPrmSourcePrimePtr = here->HFETAsourcePrmPrmSourcePrimeStructPtr->CSC ;

            if ((here->HFETAsourcePrmPrmNode != 0) && (here->HFETAgatePrimeNode != 0))
                here->HFETAsourcePrmPrmGatePrimePtr = here->HFETAsourcePrmPrmGatePrimeStructPtr->CSC ;

            if ((here->HFETAgatePrimeNode != 0) && (here->HFETAsourcePrmPrmNode != 0))
                here->HFETAgatePrimeSourcePrmPrmPtr = here->HFETAgatePrimeSourcePrmPrmStructPtr->CSC ;

            if ((here->HFETAsourcePrmPrmNode != 0) && (here->HFETAsourcePrmPrmNode != 0))
                here->HFETAsourcePrmPrmSourcePrmPrmPtr = here->HFETAsourcePrmPrmSourcePrmPrmStructPtr->CSC ;

            if ((here->HFETAgateNode != 0) && (here->HFETAgateNode != 0))
                here->HFETAgateGatePtr = here->HFETAgateGateStructPtr->CSC ;

            if ((here->HFETAgateNode != 0) && (here->HFETAgatePrimeNode != 0))
                here->HFETAgateGatePrimePtr = here->HFETAgateGatePrimeStructPtr->CSC ;

            if ((here->HFETAgatePrimeNode != 0) && (here->HFETAgateNode != 0))
                here->HFETAgatePrimeGatePtr = here->HFETAgatePrimeGateStructPtr->CSC ;

        }
    }

    return (OK) ;
}
