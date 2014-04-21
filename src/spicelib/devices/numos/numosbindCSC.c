/**********
Author: 2013 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "numosdef.h"
#include "ngspice/sperror.h"
#include "ngspice/klu-binding.h"

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
    for ( ; model != NULL ; model = NUMOSnextModel(model))
    {
        /* loop through all the instances of the model */
        for (here = NUMOSinstances(model); here != NULL ; here = NUMOSnextInstance(here))
        {
            CREATE_KLU_BINDING_TABLE(NUMOSdrainDrainPtr, NUMOSdrainDrainBinding, NUMOSdrainNode, NUMOSdrainNode);
            CREATE_KLU_BINDING_TABLE(NUMOSdrainSourcePtr, NUMOSdrainSourceBinding, NUMOSdrainNode, NUMOSsourceNode);
            CREATE_KLU_BINDING_TABLE(NUMOSdrainGatePtr, NUMOSdrainGateBinding, NUMOSdrainNode, NUMOSgateNode);
            CREATE_KLU_BINDING_TABLE(NUMOSdrainBulkPtr, NUMOSdrainBulkBinding, NUMOSdrainNode, NUMOSbulkNode);
            CREATE_KLU_BINDING_TABLE(NUMOSsourceDrainPtr, NUMOSsourceDrainBinding, NUMOSsourceNode, NUMOSdrainNode);
            CREATE_KLU_BINDING_TABLE(NUMOSsourceSourcePtr, NUMOSsourceSourceBinding, NUMOSsourceNode, NUMOSsourceNode);
            CREATE_KLU_BINDING_TABLE(NUMOSsourceGatePtr, NUMOSsourceGateBinding, NUMOSsourceNode, NUMOSgateNode);
            CREATE_KLU_BINDING_TABLE(NUMOSsourceBulkPtr, NUMOSsourceBulkBinding, NUMOSsourceNode, NUMOSbulkNode);
            CREATE_KLU_BINDING_TABLE(NUMOSgateDrainPtr, NUMOSgateDrainBinding, NUMOSgateNode, NUMOSdrainNode);
            CREATE_KLU_BINDING_TABLE(NUMOSgateSourcePtr, NUMOSgateSourceBinding, NUMOSgateNode, NUMOSsourceNode);
            CREATE_KLU_BINDING_TABLE(NUMOSgateGatePtr, NUMOSgateGateBinding, NUMOSgateNode, NUMOSgateNode);
            CREATE_KLU_BINDING_TABLE(NUMOSgateBulkPtr, NUMOSgateBulkBinding, NUMOSgateNode, NUMOSbulkNode);
            CREATE_KLU_BINDING_TABLE(NUMOSbulkDrainPtr, NUMOSbulkDrainBinding, NUMOSbulkNode, NUMOSdrainNode);
            CREATE_KLU_BINDING_TABLE(NUMOSbulkSourcePtr, NUMOSbulkSourceBinding, NUMOSbulkNode, NUMOSsourceNode);
            CREATE_KLU_BINDING_TABLE(NUMOSbulkGatePtr, NUMOSbulkGateBinding, NUMOSbulkNode, NUMOSgateNode);
            CREATE_KLU_BINDING_TABLE(NUMOSbulkBulkPtr, NUMOSbulkBulkBinding, NUMOSbulkNode, NUMOSbulkNode);
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
    for ( ; model != NULL ; model = NUMOSnextModel(model))
    {
        /* loop through all the instances of the model */
        for (here = NUMOSinstances(model); here != NULL ; here = NUMOSnextInstance(here))
        {
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(NUMOSdrainDrainPtr, NUMOSdrainDrainBinding, NUMOSdrainNode, NUMOSdrainNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(NUMOSdrainSourcePtr, NUMOSdrainSourceBinding, NUMOSdrainNode, NUMOSsourceNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(NUMOSdrainGatePtr, NUMOSdrainGateBinding, NUMOSdrainNode, NUMOSgateNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(NUMOSdrainBulkPtr, NUMOSdrainBulkBinding, NUMOSdrainNode, NUMOSbulkNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(NUMOSsourceDrainPtr, NUMOSsourceDrainBinding, NUMOSsourceNode, NUMOSdrainNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(NUMOSsourceSourcePtr, NUMOSsourceSourceBinding, NUMOSsourceNode, NUMOSsourceNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(NUMOSsourceGatePtr, NUMOSsourceGateBinding, NUMOSsourceNode, NUMOSgateNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(NUMOSsourceBulkPtr, NUMOSsourceBulkBinding, NUMOSsourceNode, NUMOSbulkNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(NUMOSgateDrainPtr, NUMOSgateDrainBinding, NUMOSgateNode, NUMOSdrainNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(NUMOSgateSourcePtr, NUMOSgateSourceBinding, NUMOSgateNode, NUMOSsourceNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(NUMOSgateGatePtr, NUMOSgateGateBinding, NUMOSgateNode, NUMOSgateNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(NUMOSgateBulkPtr, NUMOSgateBulkBinding, NUMOSgateNode, NUMOSbulkNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(NUMOSbulkDrainPtr, NUMOSbulkDrainBinding, NUMOSbulkNode, NUMOSdrainNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(NUMOSbulkSourcePtr, NUMOSbulkSourceBinding, NUMOSbulkNode, NUMOSsourceNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(NUMOSbulkGatePtr, NUMOSbulkGateBinding, NUMOSbulkNode, NUMOSgateNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(NUMOSbulkBulkPtr, NUMOSbulkBulkBinding, NUMOSbulkNode, NUMOSbulkNode);
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
    for ( ; model != NULL ; model = NUMOSnextModel(model))
    {
        /* loop through all the instances of the model */
        for (here = NUMOSinstances(model); here != NULL ; here = NUMOSnextInstance(here))
        {
            CONVERT_KLU_BINDING_TABLE_TO_REAL(NUMOSdrainDrainPtr, NUMOSdrainDrainBinding, NUMOSdrainNode, NUMOSdrainNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(NUMOSdrainSourcePtr, NUMOSdrainSourceBinding, NUMOSdrainNode, NUMOSsourceNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(NUMOSdrainGatePtr, NUMOSdrainGateBinding, NUMOSdrainNode, NUMOSgateNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(NUMOSdrainBulkPtr, NUMOSdrainBulkBinding, NUMOSdrainNode, NUMOSbulkNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(NUMOSsourceDrainPtr, NUMOSsourceDrainBinding, NUMOSsourceNode, NUMOSdrainNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(NUMOSsourceSourcePtr, NUMOSsourceSourceBinding, NUMOSsourceNode, NUMOSsourceNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(NUMOSsourceGatePtr, NUMOSsourceGateBinding, NUMOSsourceNode, NUMOSgateNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(NUMOSsourceBulkPtr, NUMOSsourceBulkBinding, NUMOSsourceNode, NUMOSbulkNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(NUMOSgateDrainPtr, NUMOSgateDrainBinding, NUMOSgateNode, NUMOSdrainNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(NUMOSgateSourcePtr, NUMOSgateSourceBinding, NUMOSgateNode, NUMOSsourceNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(NUMOSgateGatePtr, NUMOSgateGateBinding, NUMOSgateNode, NUMOSgateNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(NUMOSgateBulkPtr, NUMOSgateBulkBinding, NUMOSgateNode, NUMOSbulkNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(NUMOSbulkDrainPtr, NUMOSbulkDrainBinding, NUMOSbulkNode, NUMOSdrainNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(NUMOSbulkSourcePtr, NUMOSbulkSourceBinding, NUMOSbulkNode, NUMOSsourceNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(NUMOSbulkGatePtr, NUMOSbulkGateBinding, NUMOSbulkNode, NUMOSgateNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(NUMOSbulkBulkPtr, NUMOSbulkBulkBinding, NUMOSbulkNode, NUMOSbulkNode);
        }
    }

    return (OK) ;
}
