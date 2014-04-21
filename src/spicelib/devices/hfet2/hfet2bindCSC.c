/**********
Author: 2013 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "hfet2defs.h"
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
HFET2bindCSC (GENmodel *inModel, CKTcircuit *ckt)
{
    HFET2model *model = (HFET2model *)inModel ;
    HFET2instance *here ;
    double *i ;
    BindElement *matched, *BindStruct ;
    size_t nz ;

    BindStruct = ckt->CKTmatrix->CKTbindStruct ;
    nz = (size_t)ckt->CKTmatrix->CKTklunz ;

    /* loop through all the HFET2 models */
    for ( ; model != NULL ; model = model->HFET2nextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->HFET2instances ; here != NULL ; here = here->HFET2nextInstance)
        {
            if ((here->HFET2drainNode != 0) && (here->HFET2drainPrimeNode != 0))
            {
                i = here->HFET2drainDrainPrimePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->HFET2drainDrainPrimeStructPtr = matched ;
                here->HFET2drainDrainPrimePtr = matched->CSC ;
            }

            if ((here->HFET2gateNode != 0) && (here->HFET2drainPrimeNode != 0))
            {
                i = here->HFET2gateDrainPrimePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->HFET2gateDrainPrimeStructPtr = matched ;
                here->HFET2gateDrainPrimePtr = matched->CSC ;
            }

            if ((here->HFET2gateNode != 0) && (here->HFET2sourcePrimeNode != 0))
            {
                i = here->HFET2gateSourcePrimePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->HFET2gateSourcePrimeStructPtr = matched ;
                here->HFET2gateSourcePrimePtr = matched->CSC ;
            }

            if ((here->HFET2sourceNode != 0) && (here->HFET2sourcePrimeNode != 0))
            {
                i = here->HFET2sourceSourcePrimePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->HFET2sourceSourcePrimeStructPtr = matched ;
                here->HFET2sourceSourcePrimePtr = matched->CSC ;
            }

            if ((here->HFET2drainPrimeNode != 0) && (here->HFET2drainNode != 0))
            {
                i = here->HFET2drainPrimeDrainPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->HFET2drainPrimeDrainStructPtr = matched ;
                here->HFET2drainPrimeDrainPtr = matched->CSC ;
            }

            if ((here->HFET2drainPrimeNode != 0) && (here->HFET2gateNode != 0))
            {
                i = here->HFET2drainPrimeGatePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->HFET2drainPrimeGateStructPtr = matched ;
                here->HFET2drainPrimeGatePtr = matched->CSC ;
            }

            if ((here->HFET2drainPrimeNode != 0) && (here->HFET2sourcePrimeNode != 0))
            {
                i = here->HFET2drainPriHFET2ourcePrimePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->HFET2drainPriHFET2ourcePrimeStructPtr = matched ;
                here->HFET2drainPriHFET2ourcePrimePtr = matched->CSC ;
            }

            if ((here->HFET2sourcePrimeNode != 0) && (here->HFET2gateNode != 0))
            {
                i = here->HFET2sourcePrimeGatePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->HFET2sourcePrimeGateStructPtr = matched ;
                here->HFET2sourcePrimeGatePtr = matched->CSC ;
            }

            if ((here->HFET2sourcePrimeNode != 0) && (here->HFET2sourceNode != 0))
            {
                i = here->HFET2sourcePriHFET2ourcePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->HFET2sourcePriHFET2ourceStructPtr = matched ;
                here->HFET2sourcePriHFET2ourcePtr = matched->CSC ;
            }

            if ((here->HFET2sourcePrimeNode != 0) && (here->HFET2drainPrimeNode != 0))
            {
                i = here->HFET2sourcePrimeDrainPrimePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->HFET2sourcePrimeDrainPrimeStructPtr = matched ;
                here->HFET2sourcePrimeDrainPrimePtr = matched->CSC ;
            }

            if ((here->HFET2drainNode != 0) && (here->HFET2drainNode != 0))
            {
                i = here->HFET2drainDrainPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->HFET2drainDrainStructPtr = matched ;
                here->HFET2drainDrainPtr = matched->CSC ;
            }

            if ((here->HFET2gateNode != 0) && (here->HFET2gateNode != 0))
            {
                i = here->HFET2gateGatePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->HFET2gateGateStructPtr = matched ;
                here->HFET2gateGatePtr = matched->CSC ;
            }

            if ((here->HFET2sourceNode != 0) && (here->HFET2sourceNode != 0))
            {
                i = here->HFET2sourceSourcePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->HFET2sourceSourceStructPtr = matched ;
                here->HFET2sourceSourcePtr = matched->CSC ;
            }

            if ((here->HFET2drainPrimeNode != 0) && (here->HFET2drainPrimeNode != 0))
            {
                i = here->HFET2drainPrimeDrainPrimePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->HFET2drainPrimeDrainPrimeStructPtr = matched ;
                here->HFET2drainPrimeDrainPrimePtr = matched->CSC ;
            }

            if ((here->HFET2sourcePrimeNode != 0) && (here->HFET2sourcePrimeNode != 0))
            {
                i = here->HFET2sourcePriHFET2ourcePrimePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->HFET2sourcePriHFET2ourcePrimeStructPtr = matched ;
                here->HFET2sourcePriHFET2ourcePrimePtr = matched->CSC ;
            }

        }
    }

    return (OK) ;
}

int
HFET2bindCSCComplex (GENmodel *inModel, CKTcircuit *ckt)
{
    HFET2model *model = (HFET2model *)inModel ;
    HFET2instance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the HFET2 models */
    for ( ; model != NULL ; model = model->HFET2nextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->HFET2instances ; here != NULL ; here = here->HFET2nextInstance)
        {
            if ((here->HFET2drainNode != 0) && (here->HFET2drainPrimeNode != 0))
                here->HFET2drainDrainPrimePtr = here->HFET2drainDrainPrimeStructPtr->CSC_Complex ;

            if ((here->HFET2gateNode != 0) && (here->HFET2drainPrimeNode != 0))
                here->HFET2gateDrainPrimePtr = here->HFET2gateDrainPrimeStructPtr->CSC_Complex ;

            if ((here->HFET2gateNode != 0) && (here->HFET2sourcePrimeNode != 0))
                here->HFET2gateSourcePrimePtr = here->HFET2gateSourcePrimeStructPtr->CSC_Complex ;

            if ((here->HFET2sourceNode != 0) && (here->HFET2sourcePrimeNode != 0))
                here->HFET2sourceSourcePrimePtr = here->HFET2sourceSourcePrimeStructPtr->CSC_Complex ;

            if ((here->HFET2drainPrimeNode != 0) && (here->HFET2drainNode != 0))
                here->HFET2drainPrimeDrainPtr = here->HFET2drainPrimeDrainStructPtr->CSC_Complex ;

            if ((here->HFET2drainPrimeNode != 0) && (here->HFET2gateNode != 0))
                here->HFET2drainPrimeGatePtr = here->HFET2drainPrimeGateStructPtr->CSC_Complex ;

            if ((here->HFET2drainPrimeNode != 0) && (here->HFET2sourcePrimeNode != 0))
                here->HFET2drainPriHFET2ourcePrimePtr = here->HFET2drainPriHFET2ourcePrimeStructPtr->CSC_Complex ;

            if ((here->HFET2sourcePrimeNode != 0) && (here->HFET2gateNode != 0))
                here->HFET2sourcePrimeGatePtr = here->HFET2sourcePrimeGateStructPtr->CSC_Complex ;

            if ((here->HFET2sourcePrimeNode != 0) && (here->HFET2sourceNode != 0))
                here->HFET2sourcePriHFET2ourcePtr = here->HFET2sourcePriHFET2ourceStructPtr->CSC_Complex ;

            if ((here->HFET2sourcePrimeNode != 0) && (here->HFET2drainPrimeNode != 0))
                here->HFET2sourcePrimeDrainPrimePtr = here->HFET2sourcePrimeDrainPrimeStructPtr->CSC_Complex ;

            if ((here->HFET2drainNode != 0) && (here->HFET2drainNode != 0))
                here->HFET2drainDrainPtr = here->HFET2drainDrainStructPtr->CSC_Complex ;

            if ((here->HFET2gateNode != 0) && (here->HFET2gateNode != 0))
                here->HFET2gateGatePtr = here->HFET2gateGateStructPtr->CSC_Complex ;

            if ((here->HFET2sourceNode != 0) && (here->HFET2sourceNode != 0))
                here->HFET2sourceSourcePtr = here->HFET2sourceSourceStructPtr->CSC_Complex ;

            if ((here->HFET2drainPrimeNode != 0) && (here->HFET2drainPrimeNode != 0))
                here->HFET2drainPrimeDrainPrimePtr = here->HFET2drainPrimeDrainPrimeStructPtr->CSC_Complex ;

            if ((here->HFET2sourcePrimeNode != 0) && (here->HFET2sourcePrimeNode != 0))
                here->HFET2sourcePriHFET2ourcePrimePtr = here->HFET2sourcePriHFET2ourcePrimeStructPtr->CSC_Complex ;

        }
    }

    return (OK) ;
}

int
HFET2bindCSCComplexToReal (GENmodel *inModel, CKTcircuit *ckt)
{
    HFET2model *model = (HFET2model *)inModel ;
    HFET2instance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the HFET2 models */
    for ( ; model != NULL ; model = model->HFET2nextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->HFET2instances ; here != NULL ; here = here->HFET2nextInstance)
        {
            if ((here->HFET2drainNode != 0) && (here->HFET2drainPrimeNode != 0))
                here->HFET2drainDrainPrimePtr = here->HFET2drainDrainPrimeStructPtr->CSC ;

            if ((here->HFET2gateNode != 0) && (here->HFET2drainPrimeNode != 0))
                here->HFET2gateDrainPrimePtr = here->HFET2gateDrainPrimeStructPtr->CSC ;

            if ((here->HFET2gateNode != 0) && (here->HFET2sourcePrimeNode != 0))
                here->HFET2gateSourcePrimePtr = here->HFET2gateSourcePrimeStructPtr->CSC ;

            if ((here->HFET2sourceNode != 0) && (here->HFET2sourcePrimeNode != 0))
                here->HFET2sourceSourcePrimePtr = here->HFET2sourceSourcePrimeStructPtr->CSC ;

            if ((here->HFET2drainPrimeNode != 0) && (here->HFET2drainNode != 0))
                here->HFET2drainPrimeDrainPtr = here->HFET2drainPrimeDrainStructPtr->CSC ;

            if ((here->HFET2drainPrimeNode != 0) && (here->HFET2gateNode != 0))
                here->HFET2drainPrimeGatePtr = here->HFET2drainPrimeGateStructPtr->CSC ;

            if ((here->HFET2drainPrimeNode != 0) && (here->HFET2sourcePrimeNode != 0))
                here->HFET2drainPriHFET2ourcePrimePtr = here->HFET2drainPriHFET2ourcePrimeStructPtr->CSC ;

            if ((here->HFET2sourcePrimeNode != 0) && (here->HFET2gateNode != 0))
                here->HFET2sourcePrimeGatePtr = here->HFET2sourcePrimeGateStructPtr->CSC ;

            if ((here->HFET2sourcePrimeNode != 0) && (here->HFET2sourceNode != 0))
                here->HFET2sourcePriHFET2ourcePtr = here->HFET2sourcePriHFET2ourceStructPtr->CSC ;

            if ((here->HFET2sourcePrimeNode != 0) && (here->HFET2drainPrimeNode != 0))
                here->HFET2sourcePrimeDrainPrimePtr = here->HFET2sourcePrimeDrainPrimeStructPtr->CSC ;

            if ((here->HFET2drainNode != 0) && (here->HFET2drainNode != 0))
                here->HFET2drainDrainPtr = here->HFET2drainDrainStructPtr->CSC ;

            if ((here->HFET2gateNode != 0) && (here->HFET2gateNode != 0))
                here->HFET2gateGatePtr = here->HFET2gateGateStructPtr->CSC ;

            if ((here->HFET2sourceNode != 0) && (here->HFET2sourceNode != 0))
                here->HFET2sourceSourcePtr = here->HFET2sourceSourceStructPtr->CSC ;

            if ((here->HFET2drainPrimeNode != 0) && (here->HFET2drainPrimeNode != 0))
                here->HFET2drainPrimeDrainPrimePtr = here->HFET2drainPrimeDrainPrimeStructPtr->CSC ;

            if ((here->HFET2sourcePrimeNode != 0) && (here->HFET2sourcePrimeNode != 0))
                here->HFET2sourcePriHFET2ourcePrimePtr = here->HFET2sourcePriHFET2ourcePrimeStructPtr->CSC ;

        }
    }

    return (OK) ;
}
