/**********
Author: 2013 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "hsm2def.h"
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
HSM2bindCSC (GENmodel *inModel, CKTcircuit *ckt)
{
    HSM2model *model = (HSM2model *)inModel ;
    HSM2instance *here ;
    double *i ;
    BindElement *matched, *BindStruct ;
    size_t nz ;

    BindStruct = ckt->CKTmatrix->CKTbindStruct ;
    nz = (size_t)ckt->CKTmatrix->CKTklunz ;

    /* loop through all the HSM2 models */
    for ( ; model != NULL ; model = model->HSM2nextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->HSM2instances ; here != NULL ; here = here->HSM2nextInstance)
        {
            CREATE_KLU_BINDING_TABLE(HSM2DPbpPtr, HSM2DPbpBinding, HSM2dNodePrime, HSM2bNodePrime);
            CREATE_KLU_BINDING_TABLE(HSM2SPbpPtr, HSM2SPbpBinding, HSM2sNodePrime, HSM2bNodePrime);
            CREATE_KLU_BINDING_TABLE(HSM2GPbpPtr, HSM2GPbpBinding, HSM2gNodePrime, HSM2bNodePrime);
            CREATE_KLU_BINDING_TABLE(HSM2BPdpPtr, HSM2BPdpBinding, HSM2bNodePrime, HSM2dNodePrime);
            CREATE_KLU_BINDING_TABLE(HSM2BPspPtr, HSM2BPspBinding, HSM2bNodePrime, HSM2sNodePrime);
            CREATE_KLU_BINDING_TABLE(HSM2BPgpPtr, HSM2BPgpBinding, HSM2bNodePrime, HSM2gNodePrime);
            CREATE_KLU_BINDING_TABLE(HSM2BPbpPtr, HSM2BPbpBinding, HSM2bNodePrime, HSM2bNodePrime);
            CREATE_KLU_BINDING_TABLE(HSM2DdPtr, HSM2DdBinding, HSM2dNode, HSM2dNode);
            CREATE_KLU_BINDING_TABLE(HSM2GPgpPtr, HSM2GPgpBinding, HSM2gNodePrime, HSM2gNodePrime);
            CREATE_KLU_BINDING_TABLE(HSM2SsPtr, HSM2SsBinding, HSM2sNode, HSM2sNode);
            CREATE_KLU_BINDING_TABLE(HSM2DPdpPtr, HSM2DPdpBinding, HSM2dNodePrime, HSM2dNodePrime);
            CREATE_KLU_BINDING_TABLE(HSM2SPspPtr, HSM2SPspBinding, HSM2sNodePrime, HSM2sNodePrime);
            CREATE_KLU_BINDING_TABLE(HSM2DdpPtr, HSM2DdpBinding, HSM2dNode, HSM2dNodePrime);
            CREATE_KLU_BINDING_TABLE(HSM2GPdpPtr, HSM2GPdpBinding, HSM2gNodePrime, HSM2dNodePrime);
            CREATE_KLU_BINDING_TABLE(HSM2GPspPtr, HSM2GPspBinding, HSM2gNodePrime, HSM2sNodePrime);
            CREATE_KLU_BINDING_TABLE(HSM2SspPtr, HSM2SspBinding, HSM2sNode, HSM2sNodePrime);
            CREATE_KLU_BINDING_TABLE(HSM2DPspPtr, HSM2DPspBinding, HSM2dNodePrime, HSM2sNodePrime);
            CREATE_KLU_BINDING_TABLE(HSM2DPdPtr, HSM2DPdBinding, HSM2dNodePrime, HSM2dNode);
            CREATE_KLU_BINDING_TABLE(HSM2DPgpPtr, HSM2DPgpBinding, HSM2dNodePrime, HSM2gNodePrime);
            CREATE_KLU_BINDING_TABLE(HSM2SPgpPtr, HSM2SPgpBinding, HSM2sNodePrime, HSM2gNodePrime);
            CREATE_KLU_BINDING_TABLE(HSM2SPsPtr, HSM2SPsBinding, HSM2sNodePrime, HSM2sNode);
            CREATE_KLU_BINDING_TABLE(HSM2SPdpPtr, HSM2SPdpBinding, HSM2sNodePrime, HSM2dNodePrime);
            if (here->HSM2_corg == 1)
            {
                CREATE_KLU_BINDING_TABLE(HSM2GgPtr, HSM2GgBinding, HSM2gNode, HSM2gNode);
                CREATE_KLU_BINDING_TABLE(HSM2GgpPtr, HSM2GgpBinding, HSM2gNode, HSM2gNodePrime);
                CREATE_KLU_BINDING_TABLE(HSM2GPgPtr, HSM2GPgBinding, HSM2gNodePrime, HSM2gNode);
                CREATE_KLU_BINDING_TABLE(HSM2GdpPtr, HSM2GdpBinding, HSM2gNode, HSM2dNodePrime);
                CREATE_KLU_BINDING_TABLE(HSM2GspPtr, HSM2GspBinding, HSM2gNode, HSM2sNodePrime);
                CREATE_KLU_BINDING_TABLE(HSM2GbpPtr, HSM2GbpBinding, HSM2gNode, HSM2bNodePrime);
            }
            if (here->HSM2_corbnet == 1)
            {
                CREATE_KLU_BINDING_TABLE(HSM2DPdbPtr, HSM2DPdbBinding, HSM2dNodePrime, HSM2dbNode);
                CREATE_KLU_BINDING_TABLE(HSM2SPsbPtr, HSM2SPsbBinding, HSM2sNodePrime, HSM2sbNode);
                CREATE_KLU_BINDING_TABLE(HSM2DBdpPtr, HSM2DBdpBinding, HSM2dbNode, HSM2dNodePrime);
                CREATE_KLU_BINDING_TABLE(HSM2DBdbPtr, HSM2DBdbBinding, HSM2dbNode, HSM2dbNode);
                CREATE_KLU_BINDING_TABLE(HSM2DBbpPtr, HSM2DBbpBinding, HSM2dbNode, HSM2bNodePrime);
                CREATE_KLU_BINDING_TABLE(HSM2DBbPtr, HSM2DBbBinding, HSM2dbNode, HSM2bNode);
                CREATE_KLU_BINDING_TABLE(HSM2BPdbPtr, HSM2BPdbBinding, HSM2bNodePrime, HSM2dbNode);
                CREATE_KLU_BINDING_TABLE(HSM2BPbPtr, HSM2BPbBinding, HSM2bNodePrime, HSM2bNode);
                CREATE_KLU_BINDING_TABLE(HSM2BPsbPtr, HSM2BPsbBinding, HSM2bNodePrime, HSM2sbNode);
                CREATE_KLU_BINDING_TABLE(HSM2SBspPtr, HSM2SBspBinding, HSM2sbNode, HSM2sNodePrime);
                CREATE_KLU_BINDING_TABLE(HSM2SBbpPtr, HSM2SBbpBinding, HSM2sbNode, HSM2bNodePrime);
                CREATE_KLU_BINDING_TABLE(HSM2SBbPtr, HSM2SBbBinding, HSM2sbNode, HSM2bNode);
                CREATE_KLU_BINDING_TABLE(HSM2SBsbPtr, HSM2SBsbBinding, HSM2sbNode, HSM2sbNode);
                CREATE_KLU_BINDING_TABLE(HSM2BdbPtr, HSM2BdbBinding, HSM2bNode, HSM2dbNode);
                CREATE_KLU_BINDING_TABLE(HSM2BbpPtr, HSM2BbpBinding, HSM2bNode, HSM2bNodePrime);
                CREATE_KLU_BINDING_TABLE(HSM2BsbPtr, HSM2BsbBinding, HSM2bNode, HSM2sbNode);
                CREATE_KLU_BINDING_TABLE(HSM2BbPtr, HSM2BbBinding, HSM2bNode, HSM2bNode);
            }
        }
    }

    return (OK) ;
}

int
HSM2bindCSCComplex (GENmodel *inModel, CKTcircuit *ckt)
{
    HSM2model *model = (HSM2model *)inModel ;
    HSM2instance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the HSM2 models */
    for ( ; model != NULL ; model = model->HSM2nextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->HSM2instances ; here != NULL ; here = here->HSM2nextInstance)
        {
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HSM2DPbpPtr, HSM2DPbpBinding, HSM2dNodePrime, HSM2bNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HSM2SPbpPtr, HSM2SPbpBinding, HSM2sNodePrime, HSM2bNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HSM2GPbpPtr, HSM2GPbpBinding, HSM2gNodePrime, HSM2bNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HSM2BPdpPtr, HSM2BPdpBinding, HSM2bNodePrime, HSM2dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HSM2BPspPtr, HSM2BPspBinding, HSM2bNodePrime, HSM2sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HSM2BPgpPtr, HSM2BPgpBinding, HSM2bNodePrime, HSM2gNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HSM2BPbpPtr, HSM2BPbpBinding, HSM2bNodePrime, HSM2bNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HSM2DdPtr, HSM2DdBinding, HSM2dNode, HSM2dNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HSM2GPgpPtr, HSM2GPgpBinding, HSM2gNodePrime, HSM2gNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HSM2SsPtr, HSM2SsBinding, HSM2sNode, HSM2sNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HSM2DPdpPtr, HSM2DPdpBinding, HSM2dNodePrime, HSM2dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HSM2SPspPtr, HSM2SPspBinding, HSM2sNodePrime, HSM2sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HSM2DdpPtr, HSM2DdpBinding, HSM2dNode, HSM2dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HSM2GPdpPtr, HSM2GPdpBinding, HSM2gNodePrime, HSM2dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HSM2GPspPtr, HSM2GPspBinding, HSM2gNodePrime, HSM2sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HSM2SspPtr, HSM2SspBinding, HSM2sNode, HSM2sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HSM2DPspPtr, HSM2DPspBinding, HSM2dNodePrime, HSM2sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HSM2DPdPtr, HSM2DPdBinding, HSM2dNodePrime, HSM2dNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HSM2DPgpPtr, HSM2DPgpBinding, HSM2dNodePrime, HSM2gNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HSM2SPgpPtr, HSM2SPgpBinding, HSM2sNodePrime, HSM2gNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HSM2SPsPtr, HSM2SPsBinding, HSM2sNodePrime, HSM2sNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HSM2SPdpPtr, HSM2SPdpBinding, HSM2sNodePrime, HSM2dNodePrime);
            if (here->HSM2_corg == 1)
            {
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HSM2GgPtr, HSM2GgBinding, HSM2gNode, HSM2gNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HSM2GgpPtr, HSM2GgpBinding, HSM2gNode, HSM2gNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HSM2GPgPtr, HSM2GPgBinding, HSM2gNodePrime, HSM2gNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HSM2GdpPtr, HSM2GdpBinding, HSM2gNode, HSM2dNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HSM2GspPtr, HSM2GspBinding, HSM2gNode, HSM2sNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HSM2GbpPtr, HSM2GbpBinding, HSM2gNode, HSM2bNodePrime);
            }
            if (here->HSM2_corbnet == 1)
            {
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HSM2DPdbPtr, HSM2DPdbBinding, HSM2dNodePrime, HSM2dbNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HSM2SPsbPtr, HSM2SPsbBinding, HSM2sNodePrime, HSM2sbNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HSM2DBdpPtr, HSM2DBdpBinding, HSM2dbNode, HSM2dNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HSM2DBdbPtr, HSM2DBdbBinding, HSM2dbNode, HSM2dbNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HSM2DBbpPtr, HSM2DBbpBinding, HSM2dbNode, HSM2bNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HSM2DBbPtr, HSM2DBbBinding, HSM2dbNode, HSM2bNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HSM2BPdbPtr, HSM2BPdbBinding, HSM2bNodePrime, HSM2dbNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HSM2BPbPtr, HSM2BPbBinding, HSM2bNodePrime, HSM2bNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HSM2BPsbPtr, HSM2BPsbBinding, HSM2bNodePrime, HSM2sbNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HSM2SBspPtr, HSM2SBspBinding, HSM2sbNode, HSM2sNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HSM2SBbpPtr, HSM2SBbpBinding, HSM2sbNode, HSM2bNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HSM2SBbPtr, HSM2SBbBinding, HSM2sbNode, HSM2bNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HSM2SBsbPtr, HSM2SBsbBinding, HSM2sbNode, HSM2sbNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HSM2BdbPtr, HSM2BdbBinding, HSM2bNode, HSM2dbNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HSM2BbpPtr, HSM2BbpBinding, HSM2bNode, HSM2bNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HSM2BsbPtr, HSM2BsbBinding, HSM2bNode, HSM2sbNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(HSM2BbPtr, HSM2BbBinding, HSM2bNode, HSM2bNode);
            }
        }
    }

    return (OK) ;
}

int
HSM2bindCSCComplexToReal (GENmodel *inModel, CKTcircuit *ckt)
{
    HSM2model *model = (HSM2model *)inModel ;
    HSM2instance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the HSM2 models */
    for ( ; model != NULL ; model = model->HSM2nextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->HSM2instances ; here != NULL ; here = here->HSM2nextInstance)
        {
            CONVERT_KLU_BINDING_TABLE_TO_REAL(HSM2DPbpPtr, HSM2DPbpBinding, HSM2dNodePrime, HSM2bNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(HSM2SPbpPtr, HSM2SPbpBinding, HSM2sNodePrime, HSM2bNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(HSM2GPbpPtr, HSM2GPbpBinding, HSM2gNodePrime, HSM2bNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(HSM2BPdpPtr, HSM2BPdpBinding, HSM2bNodePrime, HSM2dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(HSM2BPspPtr, HSM2BPspBinding, HSM2bNodePrime, HSM2sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(HSM2BPgpPtr, HSM2BPgpBinding, HSM2bNodePrime, HSM2gNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(HSM2BPbpPtr, HSM2BPbpBinding, HSM2bNodePrime, HSM2bNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(HSM2DdPtr, HSM2DdBinding, HSM2dNode, HSM2dNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(HSM2GPgpPtr, HSM2GPgpBinding, HSM2gNodePrime, HSM2gNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(HSM2SsPtr, HSM2SsBinding, HSM2sNode, HSM2sNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(HSM2DPdpPtr, HSM2DPdpBinding, HSM2dNodePrime, HSM2dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(HSM2SPspPtr, HSM2SPspBinding, HSM2sNodePrime, HSM2sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(HSM2DdpPtr, HSM2DdpBinding, HSM2dNode, HSM2dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(HSM2GPdpPtr, HSM2GPdpBinding, HSM2gNodePrime, HSM2dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(HSM2GPspPtr, HSM2GPspBinding, HSM2gNodePrime, HSM2sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(HSM2SspPtr, HSM2SspBinding, HSM2sNode, HSM2sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(HSM2DPspPtr, HSM2DPspBinding, HSM2dNodePrime, HSM2sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(HSM2DPdPtr, HSM2DPdBinding, HSM2dNodePrime, HSM2dNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(HSM2DPgpPtr, HSM2DPgpBinding, HSM2dNodePrime, HSM2gNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(HSM2SPgpPtr, HSM2SPgpBinding, HSM2sNodePrime, HSM2gNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(HSM2SPsPtr, HSM2SPsBinding, HSM2sNodePrime, HSM2sNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(HSM2SPdpPtr, HSM2SPdpBinding, HSM2sNodePrime, HSM2dNodePrime);
            if (here->HSM2_corg == 1)
            {
                CONVERT_KLU_BINDING_TABLE_TO_REAL(HSM2GgPtr, HSM2GgBinding, HSM2gNode, HSM2gNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(HSM2GgpPtr, HSM2GgpBinding, HSM2gNode, HSM2gNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(HSM2GPgPtr, HSM2GPgBinding, HSM2gNodePrime, HSM2gNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(HSM2GdpPtr, HSM2GdpBinding, HSM2gNode, HSM2dNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(HSM2GspPtr, HSM2GspBinding, HSM2gNode, HSM2sNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(HSM2GbpPtr, HSM2GbpBinding, HSM2gNode, HSM2bNodePrime);
            }
            if (here->HSM2_corbnet == 1)
            {
                CONVERT_KLU_BINDING_TABLE_TO_REAL(HSM2DPdbPtr, HSM2DPdbBinding, HSM2dNodePrime, HSM2dbNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(HSM2SPsbPtr, HSM2SPsbBinding, HSM2sNodePrime, HSM2sbNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(HSM2DBdpPtr, HSM2DBdpBinding, HSM2dbNode, HSM2dNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(HSM2DBdbPtr, HSM2DBdbBinding, HSM2dbNode, HSM2dbNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(HSM2DBbpPtr, HSM2DBbpBinding, HSM2dbNode, HSM2bNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(HSM2DBbPtr, HSM2DBbBinding, HSM2dbNode, HSM2bNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(HSM2BPdbPtr, HSM2BPdbBinding, HSM2bNodePrime, HSM2dbNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(HSM2BPbPtr, HSM2BPbBinding, HSM2bNodePrime, HSM2bNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(HSM2BPsbPtr, HSM2BPsbBinding, HSM2bNodePrime, HSM2sbNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(HSM2SBspPtr, HSM2SBspBinding, HSM2sbNode, HSM2sNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(HSM2SBbpPtr, HSM2SBbpBinding, HSM2sbNode, HSM2bNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(HSM2SBbPtr, HSM2SBbBinding, HSM2sbNode, HSM2bNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(HSM2SBsbPtr, HSM2SBsbBinding, HSM2sbNode, HSM2sbNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(HSM2BdbPtr, HSM2BdbBinding, HSM2bNode, HSM2dbNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(HSM2BbpPtr, HSM2BbpBinding, HSM2bNode, HSM2bNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(HSM2BsbPtr, HSM2BsbBinding, HSM2bNode, HSM2sbNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(HSM2BbPtr, HSM2BbBinding, HSM2bNode, HSM2bNode);
            }
        }
    }

    return (OK) ;
}
