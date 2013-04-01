/**********
Author: 2013 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "numosdef.h"
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
NUMOSbindCSC (GENmodel *inModel, CKTcircuit *ckt)
{
    NUMOSmodel *model = (NUMOSmodel *)inModel ;
    NUMOSinstance *here ;
    double *i ;
    BindElement *matched, *BindStruct ;
    size_t nz ;

    BindStruct = ckt->CKTmatrix->CKTbindStruct ;
    nz = (size_t)ckt->CKTmatrix->CKTklunz ;

    /* loop through all the NUMOS models */
    for ( ; model != NULL ; model = model->NUMOSnextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->NUMOSinstances ; here != NULL ; here = here->NUMOSnextInstance)
        {
            if ((here-> NUMOSdrainNode != 0) && (here-> NUMOSdrainNode != 0))
            {
                i = here->NUMOSdrainDrainPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->NUMOSdrainDrainStructPtr = matched ;
                here->NUMOSdrainDrainPtr = matched->CSC ;
            }

            if ((here-> NUMOSdrainNode != 0) && (here-> NUMOSsourceNode != 0))
            {
                i = here->NUMOSdrainSourcePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->NUMOSdrainSourceStructPtr = matched ;
                here->NUMOSdrainSourcePtr = matched->CSC ;
            }

            if ((here-> NUMOSdrainNode != 0) && (here-> NUMOSgateNode != 0))
            {
                i = here->NUMOSdrainGatePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->NUMOSdrainGateStructPtr = matched ;
                here->NUMOSdrainGatePtr = matched->CSC ;
            }

            if ((here-> NUMOSdrainNode != 0) && (here-> NUMOSbulkNode != 0))
            {
                i = here->NUMOSdrainBulkPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->NUMOSdrainBulkStructPtr = matched ;
                here->NUMOSdrainBulkPtr = matched->CSC ;
            }

            if ((here-> NUMOSsourceNode != 0) && (here-> NUMOSdrainNode != 0))
            {
                i = here->NUMOSsourceDrainPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->NUMOSsourceDrainStructPtr = matched ;
                here->NUMOSsourceDrainPtr = matched->CSC ;
            }

            if ((here-> NUMOSsourceNode != 0) && (here-> NUMOSsourceNode != 0))
            {
                i = here->NUMOSsourceSourcePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->NUMOSsourceSourceStructPtr = matched ;
                here->NUMOSsourceSourcePtr = matched->CSC ;
            }

            if ((here-> NUMOSsourceNode != 0) && (here-> NUMOSgateNode != 0))
            {
                i = here->NUMOSsourceGatePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->NUMOSsourceGateStructPtr = matched ;
                here->NUMOSsourceGatePtr = matched->CSC ;
            }

            if ((here-> NUMOSsourceNode != 0) && (here-> NUMOSbulkNode != 0))
            {
                i = here->NUMOSsourceBulkPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->NUMOSsourceBulkStructPtr = matched ;
                here->NUMOSsourceBulkPtr = matched->CSC ;
            }

            if ((here-> NUMOSgateNode != 0) && (here-> NUMOSdrainNode != 0))
            {
                i = here->NUMOSgateDrainPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->NUMOSgateDrainStructPtr = matched ;
                here->NUMOSgateDrainPtr = matched->CSC ;
            }

            if ((here-> NUMOSgateNode != 0) && (here-> NUMOSsourceNode != 0))
            {
                i = here->NUMOSgateSourcePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->NUMOSgateSourceStructPtr = matched ;
                here->NUMOSgateSourcePtr = matched->CSC ;
            }

            if ((here-> NUMOSgateNode != 0) && (here-> NUMOSgateNode != 0))
            {
                i = here->NUMOSgateGatePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->NUMOSgateGateStructPtr = matched ;
                here->NUMOSgateGatePtr = matched->CSC ;
            }

            if ((here-> NUMOSgateNode != 0) && (here-> NUMOSbulkNode != 0))
            {
                i = here->NUMOSgateBulkPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->NUMOSgateBulkStructPtr = matched ;
                here->NUMOSgateBulkPtr = matched->CSC ;
            }

            if ((here-> NUMOSbulkNode != 0) && (here-> NUMOSdrainNode != 0))
            {
                i = here->NUMOSbulkDrainPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->NUMOSbulkDrainStructPtr = matched ;
                here->NUMOSbulkDrainPtr = matched->CSC ;
            }

            if ((here-> NUMOSbulkNode != 0) && (here-> NUMOSsourceNode != 0))
            {
                i = here->NUMOSbulkSourcePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->NUMOSbulkSourceStructPtr = matched ;
                here->NUMOSbulkSourcePtr = matched->CSC ;
            }

            if ((here-> NUMOSbulkNode != 0) && (here-> NUMOSgateNode != 0))
            {
                i = here->NUMOSbulkGatePtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->NUMOSbulkGateStructPtr = matched ;
                here->NUMOSbulkGatePtr = matched->CSC ;
            }

            if ((here-> NUMOSbulkNode != 0) && (here-> NUMOSbulkNode != 0))
            {
                i = here->NUMOSbulkBulkPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->NUMOSbulkBulkStructPtr = matched ;
                here->NUMOSbulkBulkPtr = matched->CSC ;
            }

        }
    }

    return (OK) ;
}

int
NUMOSbindCSCComplex (GENmodel *inModel, CKTcircuit *ckt)
{
    NUMOSmodel *model = (NUMOSmodel *)inModel ;
    NUMOSinstance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the NUMOS models */
    for ( ; model != NULL ; model = model->NUMOSnextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->NUMOSinstances ; here != NULL ; here = here->NUMOSnextInstance)
        {
            if ((here-> NUMOSdrainNode != 0) && (here-> NUMOSdrainNode != 0))
                here->NUMOSdrainDrainPtr = here->NUMOSdrainDrainStructPtr->CSC_Complex ;

            if ((here-> NUMOSdrainNode != 0) && (here-> NUMOSsourceNode != 0))
                here->NUMOSdrainSourcePtr = here->NUMOSdrainSourceStructPtr->CSC_Complex ;

            if ((here-> NUMOSdrainNode != 0) && (here-> NUMOSgateNode != 0))
                here->NUMOSdrainGatePtr = here->NUMOSdrainGateStructPtr->CSC_Complex ;

            if ((here-> NUMOSdrainNode != 0) && (here-> NUMOSbulkNode != 0))
                here->NUMOSdrainBulkPtr = here->NUMOSdrainBulkStructPtr->CSC_Complex ;

            if ((here-> NUMOSsourceNode != 0) && (here-> NUMOSdrainNode != 0))
                here->NUMOSsourceDrainPtr = here->NUMOSsourceDrainStructPtr->CSC_Complex ;

            if ((here-> NUMOSsourceNode != 0) && (here-> NUMOSsourceNode != 0))
                here->NUMOSsourceSourcePtr = here->NUMOSsourceSourceStructPtr->CSC_Complex ;

            if ((here-> NUMOSsourceNode != 0) && (here-> NUMOSgateNode != 0))
                here->NUMOSsourceGatePtr = here->NUMOSsourceGateStructPtr->CSC_Complex ;

            if ((here-> NUMOSsourceNode != 0) && (here-> NUMOSbulkNode != 0))
                here->NUMOSsourceBulkPtr = here->NUMOSsourceBulkStructPtr->CSC_Complex ;

            if ((here-> NUMOSgateNode != 0) && (here-> NUMOSdrainNode != 0))
                here->NUMOSgateDrainPtr = here->NUMOSgateDrainStructPtr->CSC_Complex ;

            if ((here-> NUMOSgateNode != 0) && (here-> NUMOSsourceNode != 0))
                here->NUMOSgateSourcePtr = here->NUMOSgateSourceStructPtr->CSC_Complex ;

            if ((here-> NUMOSgateNode != 0) && (here-> NUMOSgateNode != 0))
                here->NUMOSgateGatePtr = here->NUMOSgateGateStructPtr->CSC_Complex ;

            if ((here-> NUMOSgateNode != 0) && (here-> NUMOSbulkNode != 0))
                here->NUMOSgateBulkPtr = here->NUMOSgateBulkStructPtr->CSC_Complex ;

            if ((here-> NUMOSbulkNode != 0) && (here-> NUMOSdrainNode != 0))
                here->NUMOSbulkDrainPtr = here->NUMOSbulkDrainStructPtr->CSC_Complex ;

            if ((here-> NUMOSbulkNode != 0) && (here-> NUMOSsourceNode != 0))
                here->NUMOSbulkSourcePtr = here->NUMOSbulkSourceStructPtr->CSC_Complex ;

            if ((here-> NUMOSbulkNode != 0) && (here-> NUMOSgateNode != 0))
                here->NUMOSbulkGatePtr = here->NUMOSbulkGateStructPtr->CSC_Complex ;

            if ((here-> NUMOSbulkNode != 0) && (here-> NUMOSbulkNode != 0))
                here->NUMOSbulkBulkPtr = here->NUMOSbulkBulkStructPtr->CSC_Complex ;

        }
    }

    return (OK) ;
}

int
NUMOSbindCSCComplexToReal (GENmodel *inModel, CKTcircuit *ckt)
{
    NUMOSmodel *model = (NUMOSmodel *)inModel ;
    NUMOSinstance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the NUMOS models */
    for ( ; model != NULL ; model = model->NUMOSnextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->NUMOSinstances ; here != NULL ; here = here->NUMOSnextInstance)
        {
            if ((here-> NUMOSdrainNode != 0) && (here-> NUMOSdrainNode != 0))
                here->NUMOSdrainDrainPtr = here->NUMOSdrainDrainStructPtr->CSC ;

            if ((here-> NUMOSdrainNode != 0) && (here-> NUMOSsourceNode != 0))
                here->NUMOSdrainSourcePtr = here->NUMOSdrainSourceStructPtr->CSC ;

            if ((here-> NUMOSdrainNode != 0) && (here-> NUMOSgateNode != 0))
                here->NUMOSdrainGatePtr = here->NUMOSdrainGateStructPtr->CSC ;

            if ((here-> NUMOSdrainNode != 0) && (here-> NUMOSbulkNode != 0))
                here->NUMOSdrainBulkPtr = here->NUMOSdrainBulkStructPtr->CSC ;

            if ((here-> NUMOSsourceNode != 0) && (here-> NUMOSdrainNode != 0))
                here->NUMOSsourceDrainPtr = here->NUMOSsourceDrainStructPtr->CSC ;

            if ((here-> NUMOSsourceNode != 0) && (here-> NUMOSsourceNode != 0))
                here->NUMOSsourceSourcePtr = here->NUMOSsourceSourceStructPtr->CSC ;

            if ((here-> NUMOSsourceNode != 0) && (here-> NUMOSgateNode != 0))
                here->NUMOSsourceGatePtr = here->NUMOSsourceGateStructPtr->CSC ;

            if ((here-> NUMOSsourceNode != 0) && (here-> NUMOSbulkNode != 0))
                here->NUMOSsourceBulkPtr = here->NUMOSsourceBulkStructPtr->CSC ;

            if ((here-> NUMOSgateNode != 0) && (here-> NUMOSdrainNode != 0))
                here->NUMOSgateDrainPtr = here->NUMOSgateDrainStructPtr->CSC ;

            if ((here-> NUMOSgateNode != 0) && (here-> NUMOSsourceNode != 0))
                here->NUMOSgateSourcePtr = here->NUMOSgateSourceStructPtr->CSC ;

            if ((here-> NUMOSgateNode != 0) && (here-> NUMOSgateNode != 0))
                here->NUMOSgateGatePtr = here->NUMOSgateGateStructPtr->CSC ;

            if ((here-> NUMOSgateNode != 0) && (here-> NUMOSbulkNode != 0))
                here->NUMOSgateBulkPtr = here->NUMOSgateBulkStructPtr->CSC ;

            if ((here-> NUMOSbulkNode != 0) && (here-> NUMOSdrainNode != 0))
                here->NUMOSbulkDrainPtr = here->NUMOSbulkDrainStructPtr->CSC ;

            if ((here-> NUMOSbulkNode != 0) && (here-> NUMOSsourceNode != 0))
                here->NUMOSbulkSourcePtr = here->NUMOSbulkSourceStructPtr->CSC ;

            if ((here-> NUMOSbulkNode != 0) && (here-> NUMOSgateNode != 0))
                here->NUMOSbulkGatePtr = here->NUMOSbulkGateStructPtr->CSC ;

            if ((here-> NUMOSbulkNode != 0) && (here-> NUMOSbulkNode != 0))
                here->NUMOSbulkBulkPtr = here->NUMOSbulkBulkStructPtr->CSC ;

        }
    }

    return (OK) ;
}
