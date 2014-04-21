/**********
Author: 2013 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "jfet2defs.h"
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
JFET2bindCSC (GENmodel *inModel, CKTcircuit *ckt)
{
    JFET2model *model = (JFET2model *)inModel ;
    JFET2instance *here ;
    double *i ;
    BindElement *matched, *BindStruct ;
    size_t nz ;

    BindStruct = ckt->CKTmatrix->CKTbindStruct ;
    nz = (size_t)ckt->CKTmatrix->CKTklunz ;

    /* loop through all the JFET2 models */
    for ( ; model != NULL ; model = model->JFET2nextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->JFET2instances ; here != NULL ; here = here->JFET2nextInstance)
        {
            if ((here->JFET2drainNode != 0) && (here->JFET2drainPrimeNode != 0))
            {
                i = here->JFET2drainDrainPrimePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->JFET2drainDrainPrimeStructPtr = matched ;
                here->JFET2drainDrainPrimePtr = matched->CSC ;
            }

            if ((here->JFET2gateNode != 0) && (here->JFET2drainPrimeNode != 0))
            {
                i = here->JFET2gateDrainPrimePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->JFET2gateDrainPrimeStructPtr = matched ;
                here->JFET2gateDrainPrimePtr = matched->CSC ;
            }

            if ((here->JFET2gateNode != 0) && (here->JFET2sourcePrimeNode != 0))
            {
                i = here->JFET2gateSourcePrimePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->JFET2gateSourcePrimeStructPtr = matched ;
                here->JFET2gateSourcePrimePtr = matched->CSC ;
            }

            if ((here->JFET2sourceNode != 0) && (here->JFET2sourcePrimeNode != 0))
            {
                i = here->JFET2sourceSourcePrimePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->JFET2sourceSourcePrimeStructPtr = matched ;
                here->JFET2sourceSourcePrimePtr = matched->CSC ;
            }

            if ((here->JFET2drainPrimeNode != 0) && (here->JFET2drainNode != 0))
            {
                i = here->JFET2drainPrimeDrainPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->JFET2drainPrimeDrainStructPtr = matched ;
                here->JFET2drainPrimeDrainPtr = matched->CSC ;
            }

            if ((here->JFET2drainPrimeNode != 0) && (here->JFET2gateNode != 0))
            {
                i = here->JFET2drainPrimeGatePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->JFET2drainPrimeGateStructPtr = matched ;
                here->JFET2drainPrimeGatePtr = matched->CSC ;
            }

            if ((here->JFET2drainPrimeNode != 0) && (here->JFET2sourcePrimeNode != 0))
            {
                i = here->JFET2drainPrimeSourcePrimePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->JFET2drainPrimeSourcePrimeStructPtr = matched ;
                here->JFET2drainPrimeSourcePrimePtr = matched->CSC ;
            }

            if ((here->JFET2sourcePrimeNode != 0) && (here->JFET2gateNode != 0))
            {
                i = here->JFET2sourcePrimeGatePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->JFET2sourcePrimeGateStructPtr = matched ;
                here->JFET2sourcePrimeGatePtr = matched->CSC ;
            }

            if ((here->JFET2sourcePrimeNode != 0) && (here->JFET2sourceNode != 0))
            {
                i = here->JFET2sourcePrimeSourcePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->JFET2sourcePrimeSourceStructPtr = matched ;
                here->JFET2sourcePrimeSourcePtr = matched->CSC ;
            }

            if ((here->JFET2sourcePrimeNode != 0) && (here->JFET2drainPrimeNode != 0))
            {
                i = here->JFET2sourcePrimeDrainPrimePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->JFET2sourcePrimeDrainPrimeStructPtr = matched ;
                here->JFET2sourcePrimeDrainPrimePtr = matched->CSC ;
            }

            if ((here->JFET2drainNode != 0) && (here->JFET2drainNode != 0))
            {
                i = here->JFET2drainDrainPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->JFET2drainDrainStructPtr = matched ;
                here->JFET2drainDrainPtr = matched->CSC ;
            }

            if ((here->JFET2gateNode != 0) && (here->JFET2gateNode != 0))
            {
                i = here->JFET2gateGatePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->JFET2gateGateStructPtr = matched ;
                here->JFET2gateGatePtr = matched->CSC ;
            }

            if ((here->JFET2sourceNode != 0) && (here->JFET2sourceNode != 0))
            {
                i = here->JFET2sourceSourcePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->JFET2sourceSourceStructPtr = matched ;
                here->JFET2sourceSourcePtr = matched->CSC ;
            }

            if ((here->JFET2drainPrimeNode != 0) && (here->JFET2drainPrimeNode != 0))
            {
                i = here->JFET2drainPrimeDrainPrimePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->JFET2drainPrimeDrainPrimeStructPtr = matched ;
                here->JFET2drainPrimeDrainPrimePtr = matched->CSC ;
            }

            if ((here->JFET2sourcePrimeNode != 0) && (here->JFET2sourcePrimeNode != 0))
            {
                i = here->JFET2sourcePrimeSourcePrimePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->JFET2sourcePrimeSourcePrimeStructPtr = matched ;
                here->JFET2sourcePrimeSourcePrimePtr = matched->CSC ;
            }

        }
    }

    return (OK) ;
}

int
JFET2bindCSCComplex (GENmodel *inModel, CKTcircuit *ckt)
{
    JFET2model *model = (JFET2model *)inModel ;
    JFET2instance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the JFET2 models */
    for ( ; model != NULL ; model = model->JFET2nextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->JFET2instances ; here != NULL ; here = here->JFET2nextInstance)
        {
            if ((here->JFET2drainNode != 0) && (here->JFET2drainPrimeNode != 0))
                here->JFET2drainDrainPrimePtr = here->JFET2drainDrainPrimeStructPtr->CSC_Complex ;

            if ((here->JFET2gateNode != 0) && (here->JFET2drainPrimeNode != 0))
                here->JFET2gateDrainPrimePtr = here->JFET2gateDrainPrimeStructPtr->CSC_Complex ;

            if ((here->JFET2gateNode != 0) && (here->JFET2sourcePrimeNode != 0))
                here->JFET2gateSourcePrimePtr = here->JFET2gateSourcePrimeStructPtr->CSC_Complex ;

            if ((here->JFET2sourceNode != 0) && (here->JFET2sourcePrimeNode != 0))
                here->JFET2sourceSourcePrimePtr = here->JFET2sourceSourcePrimeStructPtr->CSC_Complex ;

            if ((here->JFET2drainPrimeNode != 0) && (here->JFET2drainNode != 0))
                here->JFET2drainPrimeDrainPtr = here->JFET2drainPrimeDrainStructPtr->CSC_Complex ;

            if ((here->JFET2drainPrimeNode != 0) && (here->JFET2gateNode != 0))
                here->JFET2drainPrimeGatePtr = here->JFET2drainPrimeGateStructPtr->CSC_Complex ;

            if ((here->JFET2drainPrimeNode != 0) && (here->JFET2sourcePrimeNode != 0))
                here->JFET2drainPrimeSourcePrimePtr = here->JFET2drainPrimeSourcePrimeStructPtr->CSC_Complex ;

            if ((here->JFET2sourcePrimeNode != 0) && (here->JFET2gateNode != 0))
                here->JFET2sourcePrimeGatePtr = here->JFET2sourcePrimeGateStructPtr->CSC_Complex ;

            if ((here->JFET2sourcePrimeNode != 0) && (here->JFET2sourceNode != 0))
                here->JFET2sourcePrimeSourcePtr = here->JFET2sourcePrimeSourceStructPtr->CSC_Complex ;

            if ((here->JFET2sourcePrimeNode != 0) && (here->JFET2drainPrimeNode != 0))
                here->JFET2sourcePrimeDrainPrimePtr = here->JFET2sourcePrimeDrainPrimeStructPtr->CSC_Complex ;

            if ((here->JFET2drainNode != 0) && (here->JFET2drainNode != 0))
                here->JFET2drainDrainPtr = here->JFET2drainDrainStructPtr->CSC_Complex ;

            if ((here->JFET2gateNode != 0) && (here->JFET2gateNode != 0))
                here->JFET2gateGatePtr = here->JFET2gateGateStructPtr->CSC_Complex ;

            if ((here->JFET2sourceNode != 0) && (here->JFET2sourceNode != 0))
                here->JFET2sourceSourcePtr = here->JFET2sourceSourceStructPtr->CSC_Complex ;

            if ((here->JFET2drainPrimeNode != 0) && (here->JFET2drainPrimeNode != 0))
                here->JFET2drainPrimeDrainPrimePtr = here->JFET2drainPrimeDrainPrimeStructPtr->CSC_Complex ;

            if ((here->JFET2sourcePrimeNode != 0) && (here->JFET2sourcePrimeNode != 0))
                here->JFET2sourcePrimeSourcePrimePtr = here->JFET2sourcePrimeSourcePrimeStructPtr->CSC_Complex ;

        }
    }

    return (OK) ;
}

int
JFET2bindCSCComplexToReal (GENmodel *inModel, CKTcircuit *ckt)
{
    JFET2model *model = (JFET2model *)inModel ;
    JFET2instance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the JFET2 models */
    for ( ; model != NULL ; model = model->JFET2nextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->JFET2instances ; here != NULL ; here = here->JFET2nextInstance)
        {
            if ((here->JFET2drainNode != 0) && (here->JFET2drainPrimeNode != 0))
                here->JFET2drainDrainPrimePtr = here->JFET2drainDrainPrimeStructPtr->CSC ;

            if ((here->JFET2gateNode != 0) && (here->JFET2drainPrimeNode != 0))
                here->JFET2gateDrainPrimePtr = here->JFET2gateDrainPrimeStructPtr->CSC ;

            if ((here->JFET2gateNode != 0) && (here->JFET2sourcePrimeNode != 0))
                here->JFET2gateSourcePrimePtr = here->JFET2gateSourcePrimeStructPtr->CSC ;

            if ((here->JFET2sourceNode != 0) && (here->JFET2sourcePrimeNode != 0))
                here->JFET2sourceSourcePrimePtr = here->JFET2sourceSourcePrimeStructPtr->CSC ;

            if ((here->JFET2drainPrimeNode != 0) && (here->JFET2drainNode != 0))
                here->JFET2drainPrimeDrainPtr = here->JFET2drainPrimeDrainStructPtr->CSC ;

            if ((here->JFET2drainPrimeNode != 0) && (here->JFET2gateNode != 0))
                here->JFET2drainPrimeGatePtr = here->JFET2drainPrimeGateStructPtr->CSC ;

            if ((here->JFET2drainPrimeNode != 0) && (here->JFET2sourcePrimeNode != 0))
                here->JFET2drainPrimeSourcePrimePtr = here->JFET2drainPrimeSourcePrimeStructPtr->CSC ;

            if ((here->JFET2sourcePrimeNode != 0) && (here->JFET2gateNode != 0))
                here->JFET2sourcePrimeGatePtr = here->JFET2sourcePrimeGateStructPtr->CSC ;

            if ((here->JFET2sourcePrimeNode != 0) && (here->JFET2sourceNode != 0))
                here->JFET2sourcePrimeSourcePtr = here->JFET2sourcePrimeSourceStructPtr->CSC ;

            if ((here->JFET2sourcePrimeNode != 0) && (here->JFET2drainPrimeNode != 0))
                here->JFET2sourcePrimeDrainPrimePtr = here->JFET2sourcePrimeDrainPrimeStructPtr->CSC ;

            if ((here->JFET2drainNode != 0) && (here->JFET2drainNode != 0))
                here->JFET2drainDrainPtr = here->JFET2drainDrainStructPtr->CSC ;

            if ((here->JFET2gateNode != 0) && (here->JFET2gateNode != 0))
                here->JFET2gateGatePtr = here->JFET2gateGateStructPtr->CSC ;

            if ((here->JFET2sourceNode != 0) && (here->JFET2sourceNode != 0))
                here->JFET2sourceSourcePtr = here->JFET2sourceSourceStructPtr->CSC ;

            if ((here->JFET2drainPrimeNode != 0) && (here->JFET2drainPrimeNode != 0))
                here->JFET2drainPrimeDrainPrimePtr = here->JFET2drainPrimeDrainPrimeStructPtr->CSC ;

            if ((here->JFET2sourcePrimeNode != 0) && (here->JFET2sourcePrimeNode != 0))
                here->JFET2sourcePrimeSourcePrimePtr = here->JFET2sourcePrimeSourcePrimeStructPtr->CSC ;

        }
    }

    return (OK) ;
}
