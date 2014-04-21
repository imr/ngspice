/**********
Author: 2013 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "jfetdefs.h"
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
JFETbindCSC (GENmodel *inModel, CKTcircuit *ckt)
{
    JFETmodel *model = (JFETmodel *)inModel ;
    JFETinstance *here ;
    double *i ;
    BindElement *matched, *BindStruct ;
    size_t nz ;

    BindStruct = ckt->CKTmatrix->CKTbindStruct ;
    nz = (size_t)ckt->CKTmatrix->CKTklunz ;

    /* loop through all the JFET models */
    for ( ; model != NULL ; model = model->JFETnextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->JFETinstances ; here != NULL ; here = here->JFETnextInstance)
        {
            if ((here->JFETdrainNode != 0) && (here->JFETdrainPrimeNode != 0))
            {
                i = here->JFETdrainDrainPrimePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->JFETdrainDrainPrimeStructPtr = matched ;
                here->JFETdrainDrainPrimePtr = matched->CSC ;
            }

            if ((here->JFETgateNode != 0) && (here->JFETdrainPrimeNode != 0))
            {
                i = here->JFETgateDrainPrimePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->JFETgateDrainPrimeStructPtr = matched ;
                here->JFETgateDrainPrimePtr = matched->CSC ;
            }

            if ((here->JFETgateNode != 0) && (here->JFETsourcePrimeNode != 0))
            {
                i = here->JFETgateSourcePrimePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->JFETgateSourcePrimeStructPtr = matched ;
                here->JFETgateSourcePrimePtr = matched->CSC ;
            }

            if ((here->JFETsourceNode != 0) && (here->JFETsourcePrimeNode != 0))
            {
                i = here->JFETsourceSourcePrimePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->JFETsourceSourcePrimeStructPtr = matched ;
                here->JFETsourceSourcePrimePtr = matched->CSC ;
            }

            if ((here->JFETdrainPrimeNode != 0) && (here->JFETdrainNode != 0))
            {
                i = here->JFETdrainPrimeDrainPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->JFETdrainPrimeDrainStructPtr = matched ;
                here->JFETdrainPrimeDrainPtr = matched->CSC ;
            }

            if ((here->JFETdrainPrimeNode != 0) && (here->JFETgateNode != 0))
            {
                i = here->JFETdrainPrimeGatePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->JFETdrainPrimeGateStructPtr = matched ;
                here->JFETdrainPrimeGatePtr = matched->CSC ;
            }

            if ((here->JFETdrainPrimeNode != 0) && (here->JFETsourcePrimeNode != 0))
            {
                i = here->JFETdrainPrimeSourcePrimePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->JFETdrainPrimeSourcePrimeStructPtr = matched ;
                here->JFETdrainPrimeSourcePrimePtr = matched->CSC ;
            }

            if ((here->JFETsourcePrimeNode != 0) && (here->JFETgateNode != 0))
            {
                i = here->JFETsourcePrimeGatePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->JFETsourcePrimeGateStructPtr = matched ;
                here->JFETsourcePrimeGatePtr = matched->CSC ;
            }

            if ((here->JFETsourcePrimeNode != 0) && (here->JFETsourceNode != 0))
            {
                i = here->JFETsourcePrimeSourcePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->JFETsourcePrimeSourceStructPtr = matched ;
                here->JFETsourcePrimeSourcePtr = matched->CSC ;
            }

            if ((here->JFETsourcePrimeNode != 0) && (here->JFETdrainPrimeNode != 0))
            {
                i = here->JFETsourcePrimeDrainPrimePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->JFETsourcePrimeDrainPrimeStructPtr = matched ;
                here->JFETsourcePrimeDrainPrimePtr = matched->CSC ;
            }

            if ((here->JFETdrainNode != 0) && (here->JFETdrainNode != 0))
            {
                i = here->JFETdrainDrainPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->JFETdrainDrainStructPtr = matched ;
                here->JFETdrainDrainPtr = matched->CSC ;
            }

            if ((here->JFETgateNode != 0) && (here->JFETgateNode != 0))
            {
                i = here->JFETgateGatePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->JFETgateGateStructPtr = matched ;
                here->JFETgateGatePtr = matched->CSC ;
            }

            if ((here->JFETsourceNode != 0) && (here->JFETsourceNode != 0))
            {
                i = here->JFETsourceSourcePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->JFETsourceSourceStructPtr = matched ;
                here->JFETsourceSourcePtr = matched->CSC ;
            }

            if ((here->JFETdrainPrimeNode != 0) && (here->JFETdrainPrimeNode != 0))
            {
                i = here->JFETdrainPrimeDrainPrimePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->JFETdrainPrimeDrainPrimeStructPtr = matched ;
                here->JFETdrainPrimeDrainPrimePtr = matched->CSC ;
            }

            if ((here->JFETsourcePrimeNode != 0) && (here->JFETsourcePrimeNode != 0))
            {
                i = here->JFETsourcePrimeSourcePrimePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->JFETsourcePrimeSourcePrimeStructPtr = matched ;
                here->JFETsourcePrimeSourcePrimePtr = matched->CSC ;
            }

        }
    }

    return (OK) ;
}

int
JFETbindCSCComplex (GENmodel *inModel, CKTcircuit *ckt)
{
    JFETmodel *model = (JFETmodel *)inModel ;
    JFETinstance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the JFET models */
    for ( ; model != NULL ; model = model->JFETnextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->JFETinstances ; here != NULL ; here = here->JFETnextInstance)
        {
            if ((here->JFETdrainNode != 0) && (here->JFETdrainPrimeNode != 0))
                here->JFETdrainDrainPrimePtr = here->JFETdrainDrainPrimeStructPtr->CSC_Complex ;

            if ((here->JFETgateNode != 0) && (here->JFETdrainPrimeNode != 0))
                here->JFETgateDrainPrimePtr = here->JFETgateDrainPrimeStructPtr->CSC_Complex ;

            if ((here->JFETgateNode != 0) && (here->JFETsourcePrimeNode != 0))
                here->JFETgateSourcePrimePtr = here->JFETgateSourcePrimeStructPtr->CSC_Complex ;

            if ((here->JFETsourceNode != 0) && (here->JFETsourcePrimeNode != 0))
                here->JFETsourceSourcePrimePtr = here->JFETsourceSourcePrimeStructPtr->CSC_Complex ;

            if ((here->JFETdrainPrimeNode != 0) && (here->JFETdrainNode != 0))
                here->JFETdrainPrimeDrainPtr = here->JFETdrainPrimeDrainStructPtr->CSC_Complex ;

            if ((here->JFETdrainPrimeNode != 0) && (here->JFETgateNode != 0))
                here->JFETdrainPrimeGatePtr = here->JFETdrainPrimeGateStructPtr->CSC_Complex ;

            if ((here->JFETdrainPrimeNode != 0) && (here->JFETsourcePrimeNode != 0))
                here->JFETdrainPrimeSourcePrimePtr = here->JFETdrainPrimeSourcePrimeStructPtr->CSC_Complex ;

            if ((here->JFETsourcePrimeNode != 0) && (here->JFETgateNode != 0))
                here->JFETsourcePrimeGatePtr = here->JFETsourcePrimeGateStructPtr->CSC_Complex ;

            if ((here->JFETsourcePrimeNode != 0) && (here->JFETsourceNode != 0))
                here->JFETsourcePrimeSourcePtr = here->JFETsourcePrimeSourceStructPtr->CSC_Complex ;

            if ((here->JFETsourcePrimeNode != 0) && (here->JFETdrainPrimeNode != 0))
                here->JFETsourcePrimeDrainPrimePtr = here->JFETsourcePrimeDrainPrimeStructPtr->CSC_Complex ;

            if ((here->JFETdrainNode != 0) && (here->JFETdrainNode != 0))
                here->JFETdrainDrainPtr = here->JFETdrainDrainStructPtr->CSC_Complex ;

            if ((here->JFETgateNode != 0) && (here->JFETgateNode != 0))
                here->JFETgateGatePtr = here->JFETgateGateStructPtr->CSC_Complex ;

            if ((here->JFETsourceNode != 0) && (here->JFETsourceNode != 0))
                here->JFETsourceSourcePtr = here->JFETsourceSourceStructPtr->CSC_Complex ;

            if ((here->JFETdrainPrimeNode != 0) && (here->JFETdrainPrimeNode != 0))
                here->JFETdrainPrimeDrainPrimePtr = here->JFETdrainPrimeDrainPrimeStructPtr->CSC_Complex ;

            if ((here->JFETsourcePrimeNode != 0) && (here->JFETsourcePrimeNode != 0))
                here->JFETsourcePrimeSourcePrimePtr = here->JFETsourcePrimeSourcePrimeStructPtr->CSC_Complex ;

        }
    }

    return (OK) ;
}

int
JFETbindCSCComplexToReal (GENmodel *inModel, CKTcircuit *ckt)
{
    JFETmodel *model = (JFETmodel *)inModel ;
    JFETinstance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the JFET models */
    for ( ; model != NULL ; model = model->JFETnextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->JFETinstances ; here != NULL ; here = here->JFETnextInstance)
        {
            if ((here->JFETdrainNode != 0) && (here->JFETdrainPrimeNode != 0))
                here->JFETdrainDrainPrimePtr = here->JFETdrainDrainPrimeStructPtr->CSC ;

            if ((here->JFETgateNode != 0) && (here->JFETdrainPrimeNode != 0))
                here->JFETgateDrainPrimePtr = here->JFETgateDrainPrimeStructPtr->CSC ;

            if ((here->JFETgateNode != 0) && (here->JFETsourcePrimeNode != 0))
                here->JFETgateSourcePrimePtr = here->JFETgateSourcePrimeStructPtr->CSC ;

            if ((here->JFETsourceNode != 0) && (here->JFETsourcePrimeNode != 0))
                here->JFETsourceSourcePrimePtr = here->JFETsourceSourcePrimeStructPtr->CSC ;

            if ((here->JFETdrainPrimeNode != 0) && (here->JFETdrainNode != 0))
                here->JFETdrainPrimeDrainPtr = here->JFETdrainPrimeDrainStructPtr->CSC ;

            if ((here->JFETdrainPrimeNode != 0) && (here->JFETgateNode != 0))
                here->JFETdrainPrimeGatePtr = here->JFETdrainPrimeGateStructPtr->CSC ;

            if ((here->JFETdrainPrimeNode != 0) && (here->JFETsourcePrimeNode != 0))
                here->JFETdrainPrimeSourcePrimePtr = here->JFETdrainPrimeSourcePrimeStructPtr->CSC ;

            if ((here->JFETsourcePrimeNode != 0) && (here->JFETgateNode != 0))
                here->JFETsourcePrimeGatePtr = here->JFETsourcePrimeGateStructPtr->CSC ;

            if ((here->JFETsourcePrimeNode != 0) && (here->JFETsourceNode != 0))
                here->JFETsourcePrimeSourcePtr = here->JFETsourcePrimeSourceStructPtr->CSC ;

            if ((here->JFETsourcePrimeNode != 0) && (here->JFETdrainPrimeNode != 0))
                here->JFETsourcePrimeDrainPrimePtr = here->JFETsourcePrimeDrainPrimeStructPtr->CSC ;

            if ((here->JFETdrainNode != 0) && (here->JFETdrainNode != 0))
                here->JFETdrainDrainPtr = here->JFETdrainDrainStructPtr->CSC ;

            if ((here->JFETgateNode != 0) && (here->JFETgateNode != 0))
                here->JFETgateGatePtr = here->JFETgateGateStructPtr->CSC ;

            if ((here->JFETsourceNode != 0) && (here->JFETsourceNode != 0))
                here->JFETsourceSourcePtr = here->JFETsourceSourceStructPtr->CSC ;

            if ((here->JFETdrainPrimeNode != 0) && (here->JFETdrainPrimeNode != 0))
                here->JFETdrainPrimeDrainPrimePtr = here->JFETdrainPrimeDrainPrimeStructPtr->CSC ;

            if ((here->JFETsourcePrimeNode != 0) && (here->JFETsourcePrimeNode != 0))
                here->JFETsourcePrimeSourcePrimePtr = here->JFETsourcePrimeSourcePrimeStructPtr->CSC ;

        }
    }

    return (OK) ;
}
