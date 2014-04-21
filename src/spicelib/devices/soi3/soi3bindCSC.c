/**********
Author: 2013 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "soi3defs.h"
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
SOI3bindCSC (GENmodel *inModel, CKTcircuit *ckt)
{
    SOI3model *model = (SOI3model *)inModel ;
    SOI3instance *here ;
    double *i ;
    BindElement *matched, *BindStruct ;
    size_t nz ;

    BindStruct = ckt->CKTmatrix->CKTbindStruct ;
    nz = (size_t)ckt->CKTmatrix->CKTklunz ;

    /* loop through all the SOI3 models */
    for ( ; model != NULL ; model = model->SOI3nextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->SOI3instances ; here != NULL ; here = here->SOI3nextInstance)
        {
            CREATE_KLU_BINDING_TABLE(SOI3D_dPtr, SOI3D_dBinding, SOI3dNode, SOI3dNode);
            CREATE_KLU_BINDING_TABLE(SOI3D_dpPtr, SOI3D_dpBinding, SOI3dNode, SOI3dNodePrime);
            CREATE_KLU_BINDING_TABLE(SOI3DP_dPtr, SOI3DP_dBinding, SOI3dNodePrime, SOI3dNode);
            CREATE_KLU_BINDING_TABLE(SOI3S_sPtr, SOI3S_sBinding, SOI3sNode, SOI3sNode);
            CREATE_KLU_BINDING_TABLE(SOI3S_spPtr, SOI3S_spBinding, SOI3sNode, SOI3sNodePrime);
            CREATE_KLU_BINDING_TABLE(SOI3SP_sPtr, SOI3SP_sBinding, SOI3sNodePrime, SOI3sNode);
            CREATE_KLU_BINDING_TABLE(SOI3GF_gfPtr, SOI3GF_gfBinding, SOI3gfNode, SOI3gfNode);
            CREATE_KLU_BINDING_TABLE(SOI3GF_gbPtr, SOI3GF_gbBinding, SOI3gfNode, SOI3gbNode);
            CREATE_KLU_BINDING_TABLE(SOI3GF_dpPtr, SOI3GF_dpBinding, SOI3gfNode, SOI3dNodePrime);
            CREATE_KLU_BINDING_TABLE(SOI3GF_spPtr, SOI3GF_spBinding, SOI3gfNode, SOI3sNodePrime);
            CREATE_KLU_BINDING_TABLE(SOI3GF_bPtr, SOI3GF_bBinding, SOI3gfNode, SOI3bNode);
            CREATE_KLU_BINDING_TABLE(SOI3GB_gfPtr, SOI3GB_gfBinding, SOI3gbNode, SOI3gfNode);
            CREATE_KLU_BINDING_TABLE(SOI3GB_gbPtr, SOI3GB_gbBinding, SOI3gbNode, SOI3gbNode);
            CREATE_KLU_BINDING_TABLE(SOI3GB_dpPtr, SOI3GB_dpBinding, SOI3gbNode, SOI3dNodePrime);
            CREATE_KLU_BINDING_TABLE(SOI3GB_spPtr, SOI3GB_spBinding, SOI3gbNode, SOI3sNodePrime);
            CREATE_KLU_BINDING_TABLE(SOI3GB_bPtr, SOI3GB_bBinding, SOI3gbNode, SOI3bNode);
            CREATE_KLU_BINDING_TABLE(SOI3B_gfPtr, SOI3B_gfBinding, SOI3bNode, SOI3gfNode);
            CREATE_KLU_BINDING_TABLE(SOI3B_gbPtr, SOI3B_gbBinding, SOI3bNode, SOI3gbNode);
            CREATE_KLU_BINDING_TABLE(SOI3B_dpPtr, SOI3B_dpBinding, SOI3bNode, SOI3dNodePrime);
            CREATE_KLU_BINDING_TABLE(SOI3B_spPtr, SOI3B_spBinding, SOI3bNode, SOI3sNodePrime);
            CREATE_KLU_BINDING_TABLE(SOI3B_bPtr, SOI3B_bBinding, SOI3bNode, SOI3bNode);
            CREATE_KLU_BINDING_TABLE(SOI3DP_gfPtr, SOI3DP_gfBinding, SOI3dNodePrime, SOI3gfNode);
            CREATE_KLU_BINDING_TABLE(SOI3DP_gbPtr, SOI3DP_gbBinding, SOI3dNodePrime, SOI3gbNode);
            CREATE_KLU_BINDING_TABLE(SOI3DP_dpPtr, SOI3DP_dpBinding, SOI3dNodePrime, SOI3dNodePrime);
            CREATE_KLU_BINDING_TABLE(SOI3DP_spPtr, SOI3DP_spBinding, SOI3dNodePrime, SOI3sNodePrime);
            CREATE_KLU_BINDING_TABLE(SOI3DP_bPtr, SOI3DP_bBinding, SOI3dNodePrime, SOI3bNode);
            CREATE_KLU_BINDING_TABLE(SOI3SP_gfPtr, SOI3SP_gfBinding, SOI3sNodePrime, SOI3gfNode);
            CREATE_KLU_BINDING_TABLE(SOI3SP_gbPtr, SOI3SP_gbBinding, SOI3sNodePrime, SOI3gbNode);
            CREATE_KLU_BINDING_TABLE(SOI3SP_dpPtr, SOI3SP_dpBinding, SOI3sNodePrime, SOI3dNodePrime);
            CREATE_KLU_BINDING_TABLE(SOI3SP_spPtr, SOI3SP_spBinding, SOI3sNodePrime, SOI3sNodePrime);
            CREATE_KLU_BINDING_TABLE(SOI3SP_bPtr, SOI3SP_bBinding, SOI3sNodePrime, SOI3bNode);
            if (here->SOI3rt == 0)
            {
                CREATE_KLU_BINDING_TABLE(SOI3TOUT_ibrPtr, SOI3TOUT_ibrBinding, SOI3toutNode, SOI3branch);
                CREATE_KLU_BINDING_TABLE(SOI3IBR_toutPtr, SOI3IBR_toutBinding, SOI3branch, SOI3toutNode);
            }
            else
            {
                CREATE_KLU_BINDING_TABLE(SOI3TOUT_toutPtr, SOI3TOUT_toutBinding, SOI3toutNode, SOI3toutNode);
                if (here->SOI3numThermalNodes > 1)
                {
                    CREATE_KLU_BINDING_TABLE(SOI3TOUT_tout1Ptr, SOI3TOUT_tout1Binding, SOI3toutNode, SOI3tout1Node);
                    CREATE_KLU_BINDING_TABLE(SOI3TOUT1_toutPtr, SOI3TOUT1_toutBinding, SOI3tout1Node, SOI3toutNode);
                    CREATE_KLU_BINDING_TABLE(SOI3TOUT1_tout1Ptr, SOI3TOUT1_tout1Binding, SOI3tout1Node, SOI3tout1Node);
                }
                if (here->SOI3numThermalNodes > 2)
                {
                    CREATE_KLU_BINDING_TABLE(SOI3TOUT1_tout2Ptr, SOI3TOUT1_tout2Binding, SOI3tout1Node, SOI3tout2Node);
                    CREATE_KLU_BINDING_TABLE(SOI3TOUT2_tout1Ptr, SOI3TOUT2_tout1Binding, SOI3tout2Node, SOI3tout1Node);
                    CREATE_KLU_BINDING_TABLE(SOI3TOUT2_tout2Ptr, SOI3TOUT2_tout2Binding, SOI3tout2Node, SOI3tout2Node);
                }
                if (here->SOI3numThermalNodes > 3)
                {
                    CREATE_KLU_BINDING_TABLE(SOI3TOUT2_tout3Ptr, SOI3TOUT2_tout3Binding, SOI3tout2Node, SOI3tout3Node);
                    CREATE_KLU_BINDING_TABLE(SOI3TOUT3_tout2Ptr, SOI3TOUT3_tout2Binding, SOI3tout3Node, SOI3tout2Node);
                    CREATE_KLU_BINDING_TABLE(SOI3TOUT3_tout3Ptr, SOI3TOUT3_tout3Binding, SOI3tout3Node, SOI3tout3Node);
                }
                if (here->SOI3numThermalNodes > 4)
                {
                    CREATE_KLU_BINDING_TABLE(SOI3TOUT3_tout4Ptr, SOI3TOUT3_tout4Binding, SOI3tout3Node, SOI3tout4Node);
                    CREATE_KLU_BINDING_TABLE(SOI3TOUT4_tout3Ptr, SOI3TOUT4_tout3Binding, SOI3tout4Node, SOI3tout3Node);
                    CREATE_KLU_BINDING_TABLE(SOI3TOUT4_tout4Ptr, SOI3TOUT4_tout4Binding, SOI3tout4Node, SOI3tout4Node);
                }
                CREATE_KLU_BINDING_TABLE(SOI3TOUT_toutPtr, SOI3TOUT_toutBinding, SOI3toutNode, SOI3toutNode);
                CREATE_KLU_BINDING_TABLE(SOI3TOUT_gfPtr, SOI3TOUT_gfBinding, SOI3toutNode, SOI3gfNode);
                CREATE_KLU_BINDING_TABLE(SOI3TOUT_gbPtr, SOI3TOUT_gbBinding, SOI3toutNode, SOI3gbNode);
                CREATE_KLU_BINDING_TABLE(SOI3TOUT_dpPtr, SOI3TOUT_dpBinding, SOI3toutNode, SOI3dNodePrime);
                CREATE_KLU_BINDING_TABLE(SOI3TOUT_spPtr, SOI3TOUT_spBinding, SOI3toutNode, SOI3sNodePrime);
                CREATE_KLU_BINDING_TABLE(SOI3TOUT_bPtr, SOI3TOUT_bBinding, SOI3toutNode, SOI3bNode);
                CREATE_KLU_BINDING_TABLE(SOI3GF_toutPtr, SOI3GF_toutBinding, SOI3gfNode, SOI3toutNode);
                CREATE_KLU_BINDING_TABLE(SOI3GB_toutPtr, SOI3GB_toutBinding, SOI3gbNode, SOI3toutNode);
                CREATE_KLU_BINDING_TABLE(SOI3DP_toutPtr, SOI3DP_toutBinding, SOI3dNodePrime, SOI3toutNode);
                CREATE_KLU_BINDING_TABLE(SOI3SP_toutPtr, SOI3SP_toutBinding, SOI3sNodePrime, SOI3toutNode);
                CREATE_KLU_BINDING_TABLE(SOI3B_toutPtr, SOI3B_toutBinding, SOI3bNode, SOI3toutNode);
            }
        }
    }

    return (OK) ;
}

int
SOI3bindCSCComplex (GENmodel *inModel, CKTcircuit *ckt)
{
    SOI3model *model = (SOI3model *)inModel ;
    SOI3instance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the SOI3 models */
    for ( ; model != NULL ; model = model->SOI3nextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->SOI3instances ; here != NULL ; here = here->SOI3nextInstance)
        {
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(SOI3D_dPtr, SOI3D_dBinding, SOI3dNode, SOI3dNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(SOI3D_dpPtr, SOI3D_dpBinding, SOI3dNode, SOI3dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(SOI3DP_dPtr, SOI3DP_dBinding, SOI3dNodePrime, SOI3dNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(SOI3S_sPtr, SOI3S_sBinding, SOI3sNode, SOI3sNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(SOI3S_spPtr, SOI3S_spBinding, SOI3sNode, SOI3sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(SOI3SP_sPtr, SOI3SP_sBinding, SOI3sNodePrime, SOI3sNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(SOI3GF_gfPtr, SOI3GF_gfBinding, SOI3gfNode, SOI3gfNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(SOI3GF_gbPtr, SOI3GF_gbBinding, SOI3gfNode, SOI3gbNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(SOI3GF_dpPtr, SOI3GF_dpBinding, SOI3gfNode, SOI3dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(SOI3GF_spPtr, SOI3GF_spBinding, SOI3gfNode, SOI3sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(SOI3GF_bPtr, SOI3GF_bBinding, SOI3gfNode, SOI3bNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(SOI3GB_gfPtr, SOI3GB_gfBinding, SOI3gbNode, SOI3gfNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(SOI3GB_gbPtr, SOI3GB_gbBinding, SOI3gbNode, SOI3gbNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(SOI3GB_dpPtr, SOI3GB_dpBinding, SOI3gbNode, SOI3dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(SOI3GB_spPtr, SOI3GB_spBinding, SOI3gbNode, SOI3sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(SOI3GB_bPtr, SOI3GB_bBinding, SOI3gbNode, SOI3bNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(SOI3B_gfPtr, SOI3B_gfBinding, SOI3bNode, SOI3gfNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(SOI3B_gbPtr, SOI3B_gbBinding, SOI3bNode, SOI3gbNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(SOI3B_dpPtr, SOI3B_dpBinding, SOI3bNode, SOI3dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(SOI3B_spPtr, SOI3B_spBinding, SOI3bNode, SOI3sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(SOI3B_bPtr, SOI3B_bBinding, SOI3bNode, SOI3bNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(SOI3DP_gfPtr, SOI3DP_gfBinding, SOI3dNodePrime, SOI3gfNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(SOI3DP_gbPtr, SOI3DP_gbBinding, SOI3dNodePrime, SOI3gbNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(SOI3DP_dpPtr, SOI3DP_dpBinding, SOI3dNodePrime, SOI3dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(SOI3DP_spPtr, SOI3DP_spBinding, SOI3dNodePrime, SOI3sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(SOI3DP_bPtr, SOI3DP_bBinding, SOI3dNodePrime, SOI3bNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(SOI3SP_gfPtr, SOI3SP_gfBinding, SOI3sNodePrime, SOI3gfNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(SOI3SP_gbPtr, SOI3SP_gbBinding, SOI3sNodePrime, SOI3gbNode);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(SOI3SP_dpPtr, SOI3SP_dpBinding, SOI3sNodePrime, SOI3dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(SOI3SP_spPtr, SOI3SP_spBinding, SOI3sNodePrime, SOI3sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(SOI3SP_bPtr, SOI3SP_bBinding, SOI3sNodePrime, SOI3bNode);
            if (here->SOI3rt == 0)
            {
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(SOI3TOUT_ibrPtr, SOI3TOUT_ibrBinding, SOI3toutNode, SOI3branch);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(SOI3IBR_toutPtr, SOI3IBR_toutBinding, SOI3branch, SOI3toutNode);
            }
            else
            {
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(SOI3TOUT_toutPtr, SOI3TOUT_toutBinding, SOI3toutNode, SOI3toutNode);
                if (here->SOI3numThermalNodes > 1)
                {
                    CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(SOI3TOUT_tout1Ptr, SOI3TOUT_tout1Binding, SOI3toutNode, SOI3tout1Node);
                    CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(SOI3TOUT1_toutPtr, SOI3TOUT1_toutBinding, SOI3tout1Node, SOI3toutNode);
                    CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(SOI3TOUT1_tout1Ptr, SOI3TOUT1_tout1Binding, SOI3tout1Node, SOI3tout1Node);
                }
                if (here->SOI3numThermalNodes > 2)
                {
                    CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(SOI3TOUT1_tout2Ptr, SOI3TOUT1_tout2Binding, SOI3tout1Node, SOI3tout2Node);
                    CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(SOI3TOUT2_tout1Ptr, SOI3TOUT2_tout1Binding, SOI3tout2Node, SOI3tout1Node);
                    CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(SOI3TOUT2_tout2Ptr, SOI3TOUT2_tout2Binding, SOI3tout2Node, SOI3tout2Node);
                }
                if (here->SOI3numThermalNodes > 3)
                {
                    CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(SOI3TOUT2_tout3Ptr, SOI3TOUT2_tout3Binding, SOI3tout2Node, SOI3tout3Node);
                    CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(SOI3TOUT3_tout2Ptr, SOI3TOUT3_tout2Binding, SOI3tout3Node, SOI3tout2Node);
                    CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(SOI3TOUT3_tout3Ptr, SOI3TOUT3_tout3Binding, SOI3tout3Node, SOI3tout3Node);
                }
                if (here->SOI3numThermalNodes > 4)
                {
                    CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(SOI3TOUT3_tout4Ptr, SOI3TOUT3_tout4Binding, SOI3tout3Node, SOI3tout4Node);
                    CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(SOI3TOUT4_tout3Ptr, SOI3TOUT4_tout3Binding, SOI3tout4Node, SOI3tout3Node);
                    CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(SOI3TOUT4_tout4Ptr, SOI3TOUT4_tout4Binding, SOI3tout4Node, SOI3tout4Node);
                }
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(SOI3TOUT_toutPtr, SOI3TOUT_toutBinding, SOI3toutNode, SOI3toutNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(SOI3TOUT_gfPtr, SOI3TOUT_gfBinding, SOI3toutNode, SOI3gfNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(SOI3TOUT_gbPtr, SOI3TOUT_gbBinding, SOI3toutNode, SOI3gbNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(SOI3TOUT_dpPtr, SOI3TOUT_dpBinding, SOI3toutNode, SOI3dNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(SOI3TOUT_spPtr, SOI3TOUT_spBinding, SOI3toutNode, SOI3sNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(SOI3TOUT_bPtr, SOI3TOUT_bBinding, SOI3toutNode, SOI3bNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(SOI3GF_toutPtr, SOI3GF_toutBinding, SOI3gfNode, SOI3toutNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(SOI3GB_toutPtr, SOI3GB_toutBinding, SOI3gbNode, SOI3toutNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(SOI3DP_toutPtr, SOI3DP_toutBinding, SOI3dNodePrime, SOI3toutNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(SOI3SP_toutPtr, SOI3SP_toutBinding, SOI3sNodePrime, SOI3toutNode);
                CONVERT_KLU_BINDING_TABLE_TO_COMPLEX(SOI3B_toutPtr, SOI3B_toutBinding, SOI3bNode, SOI3toutNode);
            }
        }
    }

    return (OK) ;
}

int
SOI3bindCSCComplexToReal (GENmodel *inModel, CKTcircuit *ckt)
{
    SOI3model *model = (SOI3model *)inModel ;
    SOI3instance *here ;

    NG_IGNORE (ckt) ;

    /* loop through all the SOI3 models */
    for ( ; model != NULL ; model = model->SOI3nextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->SOI3instances ; here != NULL ; here = here->SOI3nextInstance)
        {
            CONVERT_KLU_BINDING_TABLE_TO_REAL(SOI3D_dPtr, SOI3D_dBinding, SOI3dNode, SOI3dNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(SOI3D_dpPtr, SOI3D_dpBinding, SOI3dNode, SOI3dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(SOI3DP_dPtr, SOI3DP_dBinding, SOI3dNodePrime, SOI3dNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(SOI3S_sPtr, SOI3S_sBinding, SOI3sNode, SOI3sNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(SOI3S_spPtr, SOI3S_spBinding, SOI3sNode, SOI3sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(SOI3SP_sPtr, SOI3SP_sBinding, SOI3sNodePrime, SOI3sNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(SOI3GF_gfPtr, SOI3GF_gfBinding, SOI3gfNode, SOI3gfNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(SOI3GF_gbPtr, SOI3GF_gbBinding, SOI3gfNode, SOI3gbNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(SOI3GF_dpPtr, SOI3GF_dpBinding, SOI3gfNode, SOI3dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(SOI3GF_spPtr, SOI3GF_spBinding, SOI3gfNode, SOI3sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(SOI3GF_bPtr, SOI3GF_bBinding, SOI3gfNode, SOI3bNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(SOI3GB_gfPtr, SOI3GB_gfBinding, SOI3gbNode, SOI3gfNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(SOI3GB_gbPtr, SOI3GB_gbBinding, SOI3gbNode, SOI3gbNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(SOI3GB_dpPtr, SOI3GB_dpBinding, SOI3gbNode, SOI3dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(SOI3GB_spPtr, SOI3GB_spBinding, SOI3gbNode, SOI3sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(SOI3GB_bPtr, SOI3GB_bBinding, SOI3gbNode, SOI3bNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(SOI3B_gfPtr, SOI3B_gfBinding, SOI3bNode, SOI3gfNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(SOI3B_gbPtr, SOI3B_gbBinding, SOI3bNode, SOI3gbNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(SOI3B_dpPtr, SOI3B_dpBinding, SOI3bNode, SOI3dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(SOI3B_spPtr, SOI3B_spBinding, SOI3bNode, SOI3sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(SOI3B_bPtr, SOI3B_bBinding, SOI3bNode, SOI3bNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(SOI3DP_gfPtr, SOI3DP_gfBinding, SOI3dNodePrime, SOI3gfNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(SOI3DP_gbPtr, SOI3DP_gbBinding, SOI3dNodePrime, SOI3gbNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(SOI3DP_dpPtr, SOI3DP_dpBinding, SOI3dNodePrime, SOI3dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(SOI3DP_spPtr, SOI3DP_spBinding, SOI3dNodePrime, SOI3sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(SOI3DP_bPtr, SOI3DP_bBinding, SOI3dNodePrime, SOI3bNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(SOI3SP_gfPtr, SOI3SP_gfBinding, SOI3sNodePrime, SOI3gfNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(SOI3SP_gbPtr, SOI3SP_gbBinding, SOI3sNodePrime, SOI3gbNode);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(SOI3SP_dpPtr, SOI3SP_dpBinding, SOI3sNodePrime, SOI3dNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(SOI3SP_spPtr, SOI3SP_spBinding, SOI3sNodePrime, SOI3sNodePrime);
            CONVERT_KLU_BINDING_TABLE_TO_REAL(SOI3SP_bPtr, SOI3SP_bBinding, SOI3sNodePrime, SOI3bNode);
            if (here->SOI3rt == 0)
            {
                CONVERT_KLU_BINDING_TABLE_TO_REAL(SOI3TOUT_ibrPtr, SOI3TOUT_ibrBinding, SOI3toutNode, SOI3branch);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(SOI3IBR_toutPtr, SOI3IBR_toutBinding, SOI3branch, SOI3toutNode);
            }
            else
            {
                CONVERT_KLU_BINDING_TABLE_TO_REAL(SOI3TOUT_toutPtr, SOI3TOUT_toutBinding, SOI3toutNode, SOI3toutNode);
                if (here->SOI3numThermalNodes > 1)
                {
                    CONVERT_KLU_BINDING_TABLE_TO_REAL(SOI3TOUT_tout1Ptr, SOI3TOUT_tout1Binding, SOI3toutNode, SOI3tout1Node);
                    CONVERT_KLU_BINDING_TABLE_TO_REAL(SOI3TOUT1_toutPtr, SOI3TOUT1_toutBinding, SOI3tout1Node, SOI3toutNode);
                    CONVERT_KLU_BINDING_TABLE_TO_REAL(SOI3TOUT1_tout1Ptr, SOI3TOUT1_tout1Binding, SOI3tout1Node, SOI3tout1Node);
                }
                if (here->SOI3numThermalNodes > 2)
                {
                    CONVERT_KLU_BINDING_TABLE_TO_REAL(SOI3TOUT1_tout2Ptr, SOI3TOUT1_tout2Binding, SOI3tout1Node, SOI3tout2Node);
                    CONVERT_KLU_BINDING_TABLE_TO_REAL(SOI3TOUT2_tout1Ptr, SOI3TOUT2_tout1Binding, SOI3tout2Node, SOI3tout1Node);
                    CONVERT_KLU_BINDING_TABLE_TO_REAL(SOI3TOUT2_tout2Ptr, SOI3TOUT2_tout2Binding, SOI3tout2Node, SOI3tout2Node);
                }
                if (here->SOI3numThermalNodes > 3)
                {
                    CONVERT_KLU_BINDING_TABLE_TO_REAL(SOI3TOUT2_tout3Ptr, SOI3TOUT2_tout3Binding, SOI3tout2Node, SOI3tout3Node);
                    CONVERT_KLU_BINDING_TABLE_TO_REAL(SOI3TOUT3_tout2Ptr, SOI3TOUT3_tout2Binding, SOI3tout3Node, SOI3tout2Node);
                    CONVERT_KLU_BINDING_TABLE_TO_REAL(SOI3TOUT3_tout3Ptr, SOI3TOUT3_tout3Binding, SOI3tout3Node, SOI3tout3Node);
                }
                if (here->SOI3numThermalNodes > 4)
                {
                    CONVERT_KLU_BINDING_TABLE_TO_REAL(SOI3TOUT3_tout4Ptr, SOI3TOUT3_tout4Binding, SOI3tout3Node, SOI3tout4Node);
                    CONVERT_KLU_BINDING_TABLE_TO_REAL(SOI3TOUT4_tout3Ptr, SOI3TOUT4_tout3Binding, SOI3tout4Node, SOI3tout3Node);
                    CONVERT_KLU_BINDING_TABLE_TO_REAL(SOI3TOUT4_tout4Ptr, SOI3TOUT4_tout4Binding, SOI3tout4Node, SOI3tout4Node);
                }
                CONVERT_KLU_BINDING_TABLE_TO_REAL(SOI3TOUT_toutPtr, SOI3TOUT_toutBinding, SOI3toutNode, SOI3toutNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(SOI3TOUT_gfPtr, SOI3TOUT_gfBinding, SOI3toutNode, SOI3gfNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(SOI3TOUT_gbPtr, SOI3TOUT_gbBinding, SOI3toutNode, SOI3gbNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(SOI3TOUT_dpPtr, SOI3TOUT_dpBinding, SOI3toutNode, SOI3dNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(SOI3TOUT_spPtr, SOI3TOUT_spBinding, SOI3toutNode, SOI3sNodePrime);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(SOI3TOUT_bPtr, SOI3TOUT_bBinding, SOI3toutNode, SOI3bNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(SOI3GF_toutPtr, SOI3GF_toutBinding, SOI3gfNode, SOI3toutNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(SOI3GB_toutPtr, SOI3GB_toutBinding, SOI3gbNode, SOI3toutNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(SOI3DP_toutPtr, SOI3DP_toutBinding, SOI3dNodePrime, SOI3toutNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(SOI3SP_toutPtr, SOI3SP_toutBinding, SOI3sNodePrime, SOI3toutNode);
                CONVERT_KLU_BINDING_TABLE_TO_REAL(SOI3B_toutPtr, SOI3B_toutBinding, SOI3bNode, SOI3toutNode);
            }
        }
    }

    return (OK) ;
}
