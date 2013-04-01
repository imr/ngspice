/**********
Author: 2013 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "soi3defs.h"
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
            if ((here->SOI3dNode != 0) && (here->SOI3dNode != 0))
            {
                i = here->SOI3D_dPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->SOI3D_dStructPtr = matched ;
                here->SOI3D_dPtr = matched->CSC ;
            }

            if ((here->SOI3dNode != 0) && (here->SOI3dNodePrime != 0))
            {
                i = here->SOI3D_dpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->SOI3D_dpStructPtr = matched ;
                here->SOI3D_dpPtr = matched->CSC ;
            }

            if ((here->SOI3dNodePrime != 0) && (here->SOI3dNode != 0))
            {
                i = here->SOI3DP_dPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->SOI3DP_dStructPtr = matched ;
                here->SOI3DP_dPtr = matched->CSC ;
            }

            if ((here->SOI3sNode != 0) && (here->SOI3sNode != 0))
            {
                i = here->SOI3S_sPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->SOI3S_sStructPtr = matched ;
                here->SOI3S_sPtr = matched->CSC ;
            }

            if ((here->SOI3sNode != 0) && (here->SOI3sNodePrime != 0))
            {
                i = here->SOI3S_spPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->SOI3S_spStructPtr = matched ;
                here->SOI3S_spPtr = matched->CSC ;
            }

            if ((here->SOI3sNodePrime != 0) && (here->SOI3sNode != 0))
            {
                i = here->SOI3SP_sPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->SOI3SP_sStructPtr = matched ;
                here->SOI3SP_sPtr = matched->CSC ;
            }

            if ((here->SOI3gfNode != 0) && (here->SOI3gfNode != 0))
            {
                i = here->SOI3GF_gfPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->SOI3GF_gfStructPtr = matched ;
                here->SOI3GF_gfPtr = matched->CSC ;
            }

            if ((here->SOI3gfNode != 0) && (here->SOI3gbNode != 0))
            {
                i = here->SOI3GF_gbPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->SOI3GF_gbStructPtr = matched ;
                here->SOI3GF_gbPtr = matched->CSC ;
            }

            if ((here->SOI3gfNode != 0) && (here->SOI3dNodePrime != 0))
            {
                i = here->SOI3GF_dpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->SOI3GF_dpStructPtr = matched ;
                here->SOI3GF_dpPtr = matched->CSC ;
            }

            if ((here->SOI3gfNode != 0) && (here->SOI3sNodePrime != 0))
            {
                i = here->SOI3GF_spPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->SOI3GF_spStructPtr = matched ;
                here->SOI3GF_spPtr = matched->CSC ;
            }

            if ((here->SOI3gfNode != 0) && (here->SOI3bNode != 0))
            {
                i = here->SOI3GF_bPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->SOI3GF_bStructPtr = matched ;
                here->SOI3GF_bPtr = matched->CSC ;
            }

            if ((here->SOI3gbNode != 0) && (here->SOI3gfNode != 0))
            {
                i = here->SOI3GB_gfPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->SOI3GB_gfStructPtr = matched ;
                here->SOI3GB_gfPtr = matched->CSC ;
            }

            if ((here->SOI3gbNode != 0) && (here->SOI3gbNode != 0))
            {
                i = here->SOI3GB_gbPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->SOI3GB_gbStructPtr = matched ;
                here->SOI3GB_gbPtr = matched->CSC ;
            }

            if ((here->SOI3gbNode != 0) && (here->SOI3dNodePrime != 0))
            {
                i = here->SOI3GB_dpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->SOI3GB_dpStructPtr = matched ;
                here->SOI3GB_dpPtr = matched->CSC ;
            }

            if ((here->SOI3gbNode != 0) && (here->SOI3sNodePrime != 0))
            {
                i = here->SOI3GB_spPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->SOI3GB_spStructPtr = matched ;
                here->SOI3GB_spPtr = matched->CSC ;
            }

            if ((here->SOI3gbNode != 0) && (here->SOI3bNode != 0))
            {
                i = here->SOI3GB_bPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->SOI3GB_bStructPtr = matched ;
                here->SOI3GB_bPtr = matched->CSC ;
            }

            if ((here->SOI3bNode != 0) && (here->SOI3gfNode != 0))
            {
                i = here->SOI3B_gfPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->SOI3B_gfStructPtr = matched ;
                here->SOI3B_gfPtr = matched->CSC ;
            }

            if ((here->SOI3bNode != 0) && (here->SOI3gbNode != 0))
            {
                i = here->SOI3B_gbPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->SOI3B_gbStructPtr = matched ;
                here->SOI3B_gbPtr = matched->CSC ;
            }

            if ((here->SOI3bNode != 0) && (here->SOI3dNodePrime != 0))
            {
                i = here->SOI3B_dpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->SOI3B_dpStructPtr = matched ;
                here->SOI3B_dpPtr = matched->CSC ;
            }

            if ((here->SOI3bNode != 0) && (here->SOI3sNodePrime != 0))
            {
                i = here->SOI3B_spPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->SOI3B_spStructPtr = matched ;
                here->SOI3B_spPtr = matched->CSC ;
            }

            if ((here->SOI3bNode != 0) && (here->SOI3bNode != 0))
            {
                i = here->SOI3B_bPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->SOI3B_bStructPtr = matched ;
                here->SOI3B_bPtr = matched->CSC ;
            }

            if ((here->SOI3dNodePrime != 0) && (here->SOI3gfNode != 0))
            {
                i = here->SOI3DP_gfPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->SOI3DP_gfStructPtr = matched ;
                here->SOI3DP_gfPtr = matched->CSC ;
            }

            if ((here->SOI3dNodePrime != 0) && (here->SOI3gbNode != 0))
            {
                i = here->SOI3DP_gbPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->SOI3DP_gbStructPtr = matched ;
                here->SOI3DP_gbPtr = matched->CSC ;
            }

            if ((here->SOI3dNodePrime != 0) && (here->SOI3dNodePrime != 0))
            {
                i = here->SOI3DP_dpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->SOI3DP_dpStructPtr = matched ;
                here->SOI3DP_dpPtr = matched->CSC ;
            }

            if ((here->SOI3dNodePrime != 0) && (here->SOI3sNodePrime != 0))
            {
                i = here->SOI3DP_spPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->SOI3DP_spStructPtr = matched ;
                here->SOI3DP_spPtr = matched->CSC ;
            }

            if ((here->SOI3dNodePrime != 0) && (here->SOI3bNode != 0))
            {
                i = here->SOI3DP_bPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->SOI3DP_bStructPtr = matched ;
                here->SOI3DP_bPtr = matched->CSC ;
            }

            if ((here->SOI3sNodePrime != 0) && (here->SOI3gfNode != 0))
            {
                i = here->SOI3SP_gfPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->SOI3SP_gfStructPtr = matched ;
                here->SOI3SP_gfPtr = matched->CSC ;
            }

            if ((here->SOI3sNodePrime != 0) && (here->SOI3gbNode != 0))
            {
                i = here->SOI3SP_gbPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->SOI3SP_gbStructPtr = matched ;
                here->SOI3SP_gbPtr = matched->CSC ;
            }

            if ((here->SOI3sNodePrime != 0) && (here->SOI3dNodePrime != 0))
            {
                i = here->SOI3SP_dpPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->SOI3SP_dpStructPtr = matched ;
                here->SOI3SP_dpPtr = matched->CSC ;
            }

            if ((here->SOI3sNodePrime != 0) && (here->SOI3sNodePrime != 0))
            {
                i = here->SOI3SP_spPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->SOI3SP_spStructPtr = matched ;
                here->SOI3SP_spPtr = matched->CSC ;
            }

            if ((here->SOI3sNodePrime != 0) && (here->SOI3bNode != 0))
            {
                i = here->SOI3SP_bPtr ;
                matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                here->SOI3SP_bStructPtr = matched ;
                here->SOI3SP_bPtr = matched->CSC ;
            }

            if (here->SOI3rt == 0)
            {
                if ((here->SOI3toutNode != 0) && (here->SOI3branch != 0))
                {
                    i = here->SOI3TOUT_ibrPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->SOI3TOUT_ibrStructPtr = matched ;
                    here->SOI3TOUT_ibrPtr = matched->CSC ;
                }

                if ((here->SOI3branch != 0) && (here->SOI3toutNode != 0))
                {
                    i = here->SOI3IBR_toutPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->SOI3IBR_toutStructPtr = matched ;
                    here->SOI3IBR_toutPtr = matched->CSC ;
                }

            }
            else
            {
                if ((here->SOI3toutNode != 0) && (here->SOI3toutNode != 0))
                {
                    i = here->SOI3TOUT_toutPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->SOI3TOUT_toutStructPtr = matched ;
                    here->SOI3TOUT_toutPtr = matched->CSC ;
                }

                if (here->SOI3numThermalNodes > 1)
                {
                    if ((here->SOI3toutNode != 0) && (here->SOI3tout1Node != 0))
                    {
                        i = here->SOI3TOUT_tout1Ptr ;
                        matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                        here->SOI3TOUT_tout1StructPtr = matched ;
                        here->SOI3TOUT_tout1Ptr = matched->CSC ;
                    }

                    if ((here->SOI3tout1Node != 0) && (here->SOI3toutNode != 0))
                    {
                        i = here->SOI3TOUT1_toutPtr ;
                        matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                        here->SOI3TOUT1_toutStructPtr = matched ;
                        here->SOI3TOUT1_toutPtr = matched->CSC ;
                    }

                    if ((here->SOI3tout1Node != 0) && (here->SOI3tout1Node != 0))
                    {
                        i = here->SOI3TOUT1_tout1Ptr ;
                        matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                        here->SOI3TOUT1_tout1StructPtr = matched ;
                        here->SOI3TOUT1_tout1Ptr = matched->CSC ;
                    }

                }
                if (here->SOI3numThermalNodes > 2)
                {
                    if ((here->SOI3tout1Node != 0) && (here->SOI3tout2Node != 0))
                    {
                        i = here->SOI3TOUT1_tout2Ptr ;
                        matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                        here->SOI3TOUT1_tout2StructPtr = matched ;
                        here->SOI3TOUT1_tout2Ptr = matched->CSC ;
                    }

                    if ((here->SOI3tout2Node != 0) && (here->SOI3tout1Node != 0))
                    {
                        i = here->SOI3TOUT2_tout1Ptr ;
                        matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                        here->SOI3TOUT2_tout1StructPtr = matched ;
                        here->SOI3TOUT2_tout1Ptr = matched->CSC ;
                    }

                    if ((here->SOI3tout2Node != 0) && (here->SOI3tout2Node != 0))
                    {
                        i = here->SOI3TOUT2_tout2Ptr ;
                        matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                        here->SOI3TOUT2_tout2StructPtr = matched ;
                        here->SOI3TOUT2_tout2Ptr = matched->CSC ;
                    }

                }
                if (here->SOI3numThermalNodes > 3)
                {
                    if ((here->SOI3tout2Node != 0) && (here->SOI3tout3Node != 0))
                    {
                        i = here->SOI3TOUT2_tout3Ptr ;
                        matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                        here->SOI3TOUT2_tout3StructPtr = matched ;
                        here->SOI3TOUT2_tout3Ptr = matched->CSC ;
                    }

                    if ((here->SOI3tout3Node != 0) && (here->SOI3tout2Node != 0))
                    {
                        i = here->SOI3TOUT3_tout2Ptr ;
                        matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                        here->SOI3TOUT3_tout2StructPtr = matched ;
                        here->SOI3TOUT3_tout2Ptr = matched->CSC ;
                    }

                    if ((here->SOI3tout3Node != 0) && (here->SOI3tout3Node != 0))
                    {
                        i = here->SOI3TOUT3_tout3Ptr ;
                        matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                        here->SOI3TOUT3_tout3StructPtr = matched ;
                        here->SOI3TOUT3_tout3Ptr = matched->CSC ;
                    }

                }
                if (here->SOI3numThermalNodes > 4)
                {
                    if ((here->SOI3tout3Node != 0) && (here->SOI3tout4Node != 0))
                    {
                        i = here->SOI3TOUT3_tout4Ptr ;
                        matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                        here->SOI3TOUT3_tout4StructPtr = matched ;
                        here->SOI3TOUT3_tout4Ptr = matched->CSC ;
                    }

                    if ((here->SOI3tout4Node != 0) && (here->SOI3tout3Node != 0))
                    {
                        i = here->SOI3TOUT4_tout3Ptr ;
                        matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                        here->SOI3TOUT4_tout3StructPtr = matched ;
                        here->SOI3TOUT4_tout3Ptr = matched->CSC ;
                    }

                    if ((here->SOI3tout4Node != 0) && (here->SOI3tout4Node != 0))
                    {
                        i = here->SOI3TOUT4_tout4Ptr ;
                        matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                        here->SOI3TOUT4_tout4StructPtr = matched ;
                        here->SOI3TOUT4_tout4Ptr = matched->CSC ;
                    }

                }
                if ((here->SOI3toutNode != 0) && (here->SOI3toutNode != 0))
                {
                    i = here->SOI3TOUT_toutPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->SOI3TOUT_toutStructPtr = matched ;
                    here->SOI3TOUT_toutPtr = matched->CSC ;
                }

                if ((here->SOI3toutNode != 0) && (here->SOI3gfNode != 0))
                {
                    i = here->SOI3TOUT_gfPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->SOI3TOUT_gfStructPtr = matched ;
                    here->SOI3TOUT_gfPtr = matched->CSC ;
                }

                if ((here->SOI3toutNode != 0) && (here->SOI3gbNode != 0))
                {
                    i = here->SOI3TOUT_gbPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->SOI3TOUT_gbStructPtr = matched ;
                    here->SOI3TOUT_gbPtr = matched->CSC ;
                }

                if ((here->SOI3toutNode != 0) && (here->SOI3dNodePrime != 0))
                {
                    i = here->SOI3TOUT_dpPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->SOI3TOUT_dpStructPtr = matched ;
                    here->SOI3TOUT_dpPtr = matched->CSC ;
                }

                if ((here->SOI3toutNode != 0) && (here->SOI3sNodePrime != 0))
                {
                    i = here->SOI3TOUT_spPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->SOI3TOUT_spStructPtr = matched ;
                    here->SOI3TOUT_spPtr = matched->CSC ;
                }

                if ((here->SOI3toutNode != 0) && (here->SOI3bNode != 0))
                {
                    i = here->SOI3TOUT_bPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->SOI3TOUT_bStructPtr = matched ;
                    here->SOI3TOUT_bPtr = matched->CSC ;
                }

                if ((here->SOI3gfNode != 0) && (here->SOI3toutNode != 0))
                {
                    i = here->SOI3GF_toutPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->SOI3GF_toutStructPtr = matched ;
                    here->SOI3GF_toutPtr = matched->CSC ;
                }

                if ((here->SOI3gbNode != 0) && (here->SOI3toutNode != 0))
                {
                    i = here->SOI3GB_toutPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->SOI3GB_toutStructPtr = matched ;
                    here->SOI3GB_toutPtr = matched->CSC ;
                }

                if ((here->SOI3dNodePrime != 0) && (here->SOI3toutNode != 0))
                {
                    i = here->SOI3DP_toutPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->SOI3DP_toutStructPtr = matched ;
                    here->SOI3DP_toutPtr = matched->CSC ;
                }

                if ((here->SOI3sNodePrime != 0) && (here->SOI3toutNode != 0))
                {
                    i = here->SOI3SP_toutPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->SOI3SP_toutStructPtr = matched ;
                    here->SOI3SP_toutPtr = matched->CSC ;
                }

                if ((here->SOI3bNode != 0) && (here->SOI3toutNode != 0))
                {
                    i = here->SOI3B_toutPtr ;
                    matched = (BindElement *) bsearch (&i, BindStruct, nz, sizeof(BindElement), BindCompare) ;
                    here->SOI3B_toutStructPtr = matched ;
                    here->SOI3B_toutPtr = matched->CSC ;
                }

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
            if ((here->SOI3dNode != 0) && (here->SOI3dNode != 0))
                here->SOI3D_dPtr = here->SOI3D_dStructPtr->CSC_Complex ;

            if ((here->SOI3dNode != 0) && (here->SOI3dNodePrime != 0))
                here->SOI3D_dpPtr = here->SOI3D_dpStructPtr->CSC_Complex ;

            if ((here->SOI3dNodePrime != 0) && (here->SOI3dNode != 0))
                here->SOI3DP_dPtr = here->SOI3DP_dStructPtr->CSC_Complex ;

            if ((here->SOI3sNode != 0) && (here->SOI3sNode != 0))
                here->SOI3S_sPtr = here->SOI3S_sStructPtr->CSC_Complex ;

            if ((here->SOI3sNode != 0) && (here->SOI3sNodePrime != 0))
                here->SOI3S_spPtr = here->SOI3S_spStructPtr->CSC_Complex ;

            if ((here->SOI3sNodePrime != 0) && (here->SOI3sNode != 0))
                here->SOI3SP_sPtr = here->SOI3SP_sStructPtr->CSC_Complex ;

            if ((here->SOI3gfNode != 0) && (here->SOI3gfNode != 0))
                here->SOI3GF_gfPtr = here->SOI3GF_gfStructPtr->CSC_Complex ;

            if ((here->SOI3gfNode != 0) && (here->SOI3gbNode != 0))
                here->SOI3GF_gbPtr = here->SOI3GF_gbStructPtr->CSC_Complex ;

            if ((here->SOI3gfNode != 0) && (here->SOI3dNodePrime != 0))
                here->SOI3GF_dpPtr = here->SOI3GF_dpStructPtr->CSC_Complex ;

            if ((here->SOI3gfNode != 0) && (here->SOI3sNodePrime != 0))
                here->SOI3GF_spPtr = here->SOI3GF_spStructPtr->CSC_Complex ;

            if ((here->SOI3gfNode != 0) && (here->SOI3bNode != 0))
                here->SOI3GF_bPtr = here->SOI3GF_bStructPtr->CSC_Complex ;

            if ((here->SOI3gbNode != 0) && (here->SOI3gfNode != 0))
                here->SOI3GB_gfPtr = here->SOI3GB_gfStructPtr->CSC_Complex ;

            if ((here->SOI3gbNode != 0) && (here->SOI3gbNode != 0))
                here->SOI3GB_gbPtr = here->SOI3GB_gbStructPtr->CSC_Complex ;

            if ((here->SOI3gbNode != 0) && (here->SOI3dNodePrime != 0))
                here->SOI3GB_dpPtr = here->SOI3GB_dpStructPtr->CSC_Complex ;

            if ((here->SOI3gbNode != 0) && (here->SOI3sNodePrime != 0))
                here->SOI3GB_spPtr = here->SOI3GB_spStructPtr->CSC_Complex ;

            if ((here->SOI3gbNode != 0) && (here->SOI3bNode != 0))
                here->SOI3GB_bPtr = here->SOI3GB_bStructPtr->CSC_Complex ;

            if ((here->SOI3bNode != 0) && (here->SOI3gfNode != 0))
                here->SOI3B_gfPtr = here->SOI3B_gfStructPtr->CSC_Complex ;

            if ((here->SOI3bNode != 0) && (here->SOI3gbNode != 0))
                here->SOI3B_gbPtr = here->SOI3B_gbStructPtr->CSC_Complex ;

            if ((here->SOI3bNode != 0) && (here->SOI3dNodePrime != 0))
                here->SOI3B_dpPtr = here->SOI3B_dpStructPtr->CSC_Complex ;

            if ((here->SOI3bNode != 0) && (here->SOI3sNodePrime != 0))
                here->SOI3B_spPtr = here->SOI3B_spStructPtr->CSC_Complex ;

            if ((here->SOI3bNode != 0) && (here->SOI3bNode != 0))
                here->SOI3B_bPtr = here->SOI3B_bStructPtr->CSC_Complex ;

            if ((here->SOI3dNodePrime != 0) && (here->SOI3gfNode != 0))
                here->SOI3DP_gfPtr = here->SOI3DP_gfStructPtr->CSC_Complex ;

            if ((here->SOI3dNodePrime != 0) && (here->SOI3gbNode != 0))
                here->SOI3DP_gbPtr = here->SOI3DP_gbStructPtr->CSC_Complex ;

            if ((here->SOI3dNodePrime != 0) && (here->SOI3dNodePrime != 0))
                here->SOI3DP_dpPtr = here->SOI3DP_dpStructPtr->CSC_Complex ;

            if ((here->SOI3dNodePrime != 0) && (here->SOI3sNodePrime != 0))
                here->SOI3DP_spPtr = here->SOI3DP_spStructPtr->CSC_Complex ;

            if ((here->SOI3dNodePrime != 0) && (here->SOI3bNode != 0))
                here->SOI3DP_bPtr = here->SOI3DP_bStructPtr->CSC_Complex ;

            if ((here->SOI3sNodePrime != 0) && (here->SOI3gfNode != 0))
                here->SOI3SP_gfPtr = here->SOI3SP_gfStructPtr->CSC_Complex ;

            if ((here->SOI3sNodePrime != 0) && (here->SOI3gbNode != 0))
                here->SOI3SP_gbPtr = here->SOI3SP_gbStructPtr->CSC_Complex ;

            if ((here->SOI3sNodePrime != 0) && (here->SOI3dNodePrime != 0))
                here->SOI3SP_dpPtr = here->SOI3SP_dpStructPtr->CSC_Complex ;

            if ((here->SOI3sNodePrime != 0) && (here->SOI3sNodePrime != 0))
                here->SOI3SP_spPtr = here->SOI3SP_spStructPtr->CSC_Complex ;

            if ((here->SOI3sNodePrime != 0) && (here->SOI3bNode != 0))
                here->SOI3SP_bPtr = here->SOI3SP_bStructPtr->CSC_Complex ;

            if (here->SOI3rt == 0)
            {
                if ((here->SOI3toutNode != 0) && (here->SOI3branch != 0))
                    here->SOI3TOUT_ibrPtr = here->SOI3TOUT_ibrStructPtr->CSC_Complex ;

                if ((here->SOI3branch != 0) && (here->SOI3toutNode != 0))
                    here->SOI3IBR_toutPtr = here->SOI3IBR_toutStructPtr->CSC_Complex ;

            }
            else
            {
                if ((here->SOI3toutNode != 0) && (here->SOI3toutNode != 0))
                    here->SOI3TOUT_toutPtr = here->SOI3TOUT_toutStructPtr->CSC_Complex ;

                if (here->SOI3numThermalNodes > 1)
                {
                    if ((here->SOI3toutNode != 0) && (here->SOI3tout1Node != 0))
                        here->SOI3TOUT_tout1Ptr = here->SOI3TOUT_tout1StructPtr->CSC_Complex ;

                    if ((here->SOI3tout1Node != 0) && (here->SOI3toutNode != 0))
                        here->SOI3TOUT1_toutPtr = here->SOI3TOUT1_toutStructPtr->CSC_Complex ;

                    if ((here->SOI3tout1Node != 0) && (here->SOI3tout1Node != 0))
                        here->SOI3TOUT1_tout1Ptr = here->SOI3TOUT1_tout1StructPtr->CSC_Complex ;

                }
                if (here->SOI3numThermalNodes > 2)
                {
                    if ((here->SOI3tout1Node != 0) && (here->SOI3tout2Node != 0))
                        here->SOI3TOUT1_tout2Ptr = here->SOI3TOUT1_tout2StructPtr->CSC_Complex ;

                    if ((here->SOI3tout2Node != 0) && (here->SOI3tout1Node != 0))
                        here->SOI3TOUT2_tout1Ptr = here->SOI3TOUT2_tout1StructPtr->CSC_Complex ;

                    if ((here->SOI3tout2Node != 0) && (here->SOI3tout2Node != 0))
                        here->SOI3TOUT2_tout2Ptr = here->SOI3TOUT2_tout2StructPtr->CSC_Complex ;

                }
                if (here->SOI3numThermalNodes > 3)
                {
                    if ((here->SOI3tout2Node != 0) && (here->SOI3tout3Node != 0))
                        here->SOI3TOUT2_tout3Ptr = here->SOI3TOUT2_tout3StructPtr->CSC_Complex ;

                    if ((here->SOI3tout3Node != 0) && (here->SOI3tout2Node != 0))
                        here->SOI3TOUT3_tout2Ptr = here->SOI3TOUT3_tout2StructPtr->CSC_Complex ;

                    if ((here->SOI3tout3Node != 0) && (here->SOI3tout3Node != 0))
                        here->SOI3TOUT3_tout3Ptr = here->SOI3TOUT3_tout3StructPtr->CSC_Complex ;

                }
                if (here->SOI3numThermalNodes > 4)
                {
                    if ((here->SOI3tout3Node != 0) && (here->SOI3tout4Node != 0))
                        here->SOI3TOUT3_tout4Ptr = here->SOI3TOUT3_tout4StructPtr->CSC_Complex ;

                    if ((here->SOI3tout4Node != 0) && (here->SOI3tout3Node != 0))
                        here->SOI3TOUT4_tout3Ptr = here->SOI3TOUT4_tout3StructPtr->CSC_Complex ;

                    if ((here->SOI3tout4Node != 0) && (here->SOI3tout4Node != 0))
                        here->SOI3TOUT4_tout4Ptr = here->SOI3TOUT4_tout4StructPtr->CSC_Complex ;

                }
                if ((here->SOI3toutNode != 0) && (here->SOI3toutNode != 0))
                    here->SOI3TOUT_toutPtr = here->SOI3TOUT_toutStructPtr->CSC_Complex ;

                if ((here->SOI3toutNode != 0) && (here->SOI3gfNode != 0))
                    here->SOI3TOUT_gfPtr = here->SOI3TOUT_gfStructPtr->CSC_Complex ;

                if ((here->SOI3toutNode != 0) && (here->SOI3gbNode != 0))
                    here->SOI3TOUT_gbPtr = here->SOI3TOUT_gbStructPtr->CSC_Complex ;

                if ((here->SOI3toutNode != 0) && (here->SOI3dNodePrime != 0))
                    here->SOI3TOUT_dpPtr = here->SOI3TOUT_dpStructPtr->CSC_Complex ;

                if ((here->SOI3toutNode != 0) && (here->SOI3sNodePrime != 0))
                    here->SOI3TOUT_spPtr = here->SOI3TOUT_spStructPtr->CSC_Complex ;

                if ((here->SOI3toutNode != 0) && (here->SOI3bNode != 0))
                    here->SOI3TOUT_bPtr = here->SOI3TOUT_bStructPtr->CSC_Complex ;

                if ((here->SOI3gfNode != 0) && (here->SOI3toutNode != 0))
                    here->SOI3GF_toutPtr = here->SOI3GF_toutStructPtr->CSC_Complex ;

                if ((here->SOI3gbNode != 0) && (here->SOI3toutNode != 0))
                    here->SOI3GB_toutPtr = here->SOI3GB_toutStructPtr->CSC_Complex ;

                if ((here->SOI3dNodePrime != 0) && (here->SOI3toutNode != 0))
                    here->SOI3DP_toutPtr = here->SOI3DP_toutStructPtr->CSC_Complex ;

                if ((here->SOI3sNodePrime != 0) && (here->SOI3toutNode != 0))
                    here->SOI3SP_toutPtr = here->SOI3SP_toutStructPtr->CSC_Complex ;

                if ((here->SOI3bNode != 0) && (here->SOI3toutNode != 0))
                    here->SOI3B_toutPtr = here->SOI3B_toutStructPtr->CSC_Complex ;

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
            if ((here->SOI3dNode != 0) && (here->SOI3dNode != 0))
                here->SOI3D_dPtr = here->SOI3D_dStructPtr->CSC ;

            if ((here->SOI3dNode != 0) && (here->SOI3dNodePrime != 0))
                here->SOI3D_dpPtr = here->SOI3D_dpStructPtr->CSC ;

            if ((here->SOI3dNodePrime != 0) && (here->SOI3dNode != 0))
                here->SOI3DP_dPtr = here->SOI3DP_dStructPtr->CSC ;

            if ((here->SOI3sNode != 0) && (here->SOI3sNode != 0))
                here->SOI3S_sPtr = here->SOI3S_sStructPtr->CSC ;

            if ((here->SOI3sNode != 0) && (here->SOI3sNodePrime != 0))
                here->SOI3S_spPtr = here->SOI3S_spStructPtr->CSC ;

            if ((here->SOI3sNodePrime != 0) && (here->SOI3sNode != 0))
                here->SOI3SP_sPtr = here->SOI3SP_sStructPtr->CSC ;

            if ((here->SOI3gfNode != 0) && (here->SOI3gfNode != 0))
                here->SOI3GF_gfPtr = here->SOI3GF_gfStructPtr->CSC ;

            if ((here->SOI3gfNode != 0) && (here->SOI3gbNode != 0))
                here->SOI3GF_gbPtr = here->SOI3GF_gbStructPtr->CSC ;

            if ((here->SOI3gfNode != 0) && (here->SOI3dNodePrime != 0))
                here->SOI3GF_dpPtr = here->SOI3GF_dpStructPtr->CSC ;

            if ((here->SOI3gfNode != 0) && (here->SOI3sNodePrime != 0))
                here->SOI3GF_spPtr = here->SOI3GF_spStructPtr->CSC ;

            if ((here->SOI3gfNode != 0) && (here->SOI3bNode != 0))
                here->SOI3GF_bPtr = here->SOI3GF_bStructPtr->CSC ;

            if ((here->SOI3gbNode != 0) && (here->SOI3gfNode != 0))
                here->SOI3GB_gfPtr = here->SOI3GB_gfStructPtr->CSC ;

            if ((here->SOI3gbNode != 0) && (here->SOI3gbNode != 0))
                here->SOI3GB_gbPtr = here->SOI3GB_gbStructPtr->CSC ;

            if ((here->SOI3gbNode != 0) && (here->SOI3dNodePrime != 0))
                here->SOI3GB_dpPtr = here->SOI3GB_dpStructPtr->CSC ;

            if ((here->SOI3gbNode != 0) && (here->SOI3sNodePrime != 0))
                here->SOI3GB_spPtr = here->SOI3GB_spStructPtr->CSC ;

            if ((here->SOI3gbNode != 0) && (here->SOI3bNode != 0))
                here->SOI3GB_bPtr = here->SOI3GB_bStructPtr->CSC ;

            if ((here->SOI3bNode != 0) && (here->SOI3gfNode != 0))
                here->SOI3B_gfPtr = here->SOI3B_gfStructPtr->CSC ;

            if ((here->SOI3bNode != 0) && (here->SOI3gbNode != 0))
                here->SOI3B_gbPtr = here->SOI3B_gbStructPtr->CSC ;

            if ((here->SOI3bNode != 0) && (here->SOI3dNodePrime != 0))
                here->SOI3B_dpPtr = here->SOI3B_dpStructPtr->CSC ;

            if ((here->SOI3bNode != 0) && (here->SOI3sNodePrime != 0))
                here->SOI3B_spPtr = here->SOI3B_spStructPtr->CSC ;

            if ((here->SOI3bNode != 0) && (here->SOI3bNode != 0))
                here->SOI3B_bPtr = here->SOI3B_bStructPtr->CSC ;

            if ((here->SOI3dNodePrime != 0) && (here->SOI3gfNode != 0))
                here->SOI3DP_gfPtr = here->SOI3DP_gfStructPtr->CSC ;

            if ((here->SOI3dNodePrime != 0) && (here->SOI3gbNode != 0))
                here->SOI3DP_gbPtr = here->SOI3DP_gbStructPtr->CSC ;

            if ((here->SOI3dNodePrime != 0) && (here->SOI3dNodePrime != 0))
                here->SOI3DP_dpPtr = here->SOI3DP_dpStructPtr->CSC ;

            if ((here->SOI3dNodePrime != 0) && (here->SOI3sNodePrime != 0))
                here->SOI3DP_spPtr = here->SOI3DP_spStructPtr->CSC ;

            if ((here->SOI3dNodePrime != 0) && (here->SOI3bNode != 0))
                here->SOI3DP_bPtr = here->SOI3DP_bStructPtr->CSC ;

            if ((here->SOI3sNodePrime != 0) && (here->SOI3gfNode != 0))
                here->SOI3SP_gfPtr = here->SOI3SP_gfStructPtr->CSC ;

            if ((here->SOI3sNodePrime != 0) && (here->SOI3gbNode != 0))
                here->SOI3SP_gbPtr = here->SOI3SP_gbStructPtr->CSC ;

            if ((here->SOI3sNodePrime != 0) && (here->SOI3dNodePrime != 0))
                here->SOI3SP_dpPtr = here->SOI3SP_dpStructPtr->CSC ;

            if ((here->SOI3sNodePrime != 0) && (here->SOI3sNodePrime != 0))
                here->SOI3SP_spPtr = here->SOI3SP_spStructPtr->CSC ;

            if ((here->SOI3sNodePrime != 0) && (here->SOI3bNode != 0))
                here->SOI3SP_bPtr = here->SOI3SP_bStructPtr->CSC ;

            if (here->SOI3rt == 0)
            {
                if ((here->SOI3toutNode != 0) && (here->SOI3branch != 0))
                    here->SOI3TOUT_ibrPtr = here->SOI3TOUT_ibrStructPtr->CSC ;

                if ((here->SOI3branch != 0) && (here->SOI3toutNode != 0))
                    here->SOI3IBR_toutPtr = here->SOI3IBR_toutStructPtr->CSC ;

            }
            else
            {
                if ((here->SOI3toutNode != 0) && (here->SOI3toutNode != 0))
                    here->SOI3TOUT_toutPtr = here->SOI3TOUT_toutStructPtr->CSC ;

                if (here->SOI3numThermalNodes > 1)
                {
                    if ((here->SOI3toutNode != 0) && (here->SOI3tout1Node != 0))
                        here->SOI3TOUT_tout1Ptr = here->SOI3TOUT_tout1StructPtr->CSC ;

                    if ((here->SOI3tout1Node != 0) && (here->SOI3toutNode != 0))
                        here->SOI3TOUT1_toutPtr = here->SOI3TOUT1_toutStructPtr->CSC ;

                    if ((here->SOI3tout1Node != 0) && (here->SOI3tout1Node != 0))
                        here->SOI3TOUT1_tout1Ptr = here->SOI3TOUT1_tout1StructPtr->CSC ;

                }
                if (here->SOI3numThermalNodes > 2)
                {
                    if ((here->SOI3tout1Node != 0) && (here->SOI3tout2Node != 0))
                        here->SOI3TOUT1_tout2Ptr = here->SOI3TOUT1_tout2StructPtr->CSC ;

                    if ((here->SOI3tout2Node != 0) && (here->SOI3tout1Node != 0))
                        here->SOI3TOUT2_tout1Ptr = here->SOI3TOUT2_tout1StructPtr->CSC ;

                    if ((here->SOI3tout2Node != 0) && (here->SOI3tout2Node != 0))
                        here->SOI3TOUT2_tout2Ptr = here->SOI3TOUT2_tout2StructPtr->CSC ;

                }
                if (here->SOI3numThermalNodes > 3)
                {
                    if ((here->SOI3tout2Node != 0) && (here->SOI3tout3Node != 0))
                        here->SOI3TOUT2_tout3Ptr = here->SOI3TOUT2_tout3StructPtr->CSC ;

                    if ((here->SOI3tout3Node != 0) && (here->SOI3tout2Node != 0))
                        here->SOI3TOUT3_tout2Ptr = here->SOI3TOUT3_tout2StructPtr->CSC ;

                    if ((here->SOI3tout3Node != 0) && (here->SOI3tout3Node != 0))
                        here->SOI3TOUT3_tout3Ptr = here->SOI3TOUT3_tout3StructPtr->CSC ;

                }
                if (here->SOI3numThermalNodes > 4)
                {
                    if ((here->SOI3tout3Node != 0) && (here->SOI3tout4Node != 0))
                        here->SOI3TOUT3_tout4Ptr = here->SOI3TOUT3_tout4StructPtr->CSC ;

                    if ((here->SOI3tout4Node != 0) && (here->SOI3tout3Node != 0))
                        here->SOI3TOUT4_tout3Ptr = here->SOI3TOUT4_tout3StructPtr->CSC ;

                    if ((here->SOI3tout4Node != 0) && (here->SOI3tout4Node != 0))
                        here->SOI3TOUT4_tout4Ptr = here->SOI3TOUT4_tout4StructPtr->CSC ;

                }
                if ((here->SOI3toutNode != 0) && (here->SOI3toutNode != 0))
                    here->SOI3TOUT_toutPtr = here->SOI3TOUT_toutStructPtr->CSC ;

                if ((here->SOI3toutNode != 0) && (here->SOI3gfNode != 0))
                    here->SOI3TOUT_gfPtr = here->SOI3TOUT_gfStructPtr->CSC ;

                if ((here->SOI3toutNode != 0) && (here->SOI3gbNode != 0))
                    here->SOI3TOUT_gbPtr = here->SOI3TOUT_gbStructPtr->CSC ;

                if ((here->SOI3toutNode != 0) && (here->SOI3dNodePrime != 0))
                    here->SOI3TOUT_dpPtr = here->SOI3TOUT_dpStructPtr->CSC ;

                if ((here->SOI3toutNode != 0) && (here->SOI3sNodePrime != 0))
                    here->SOI3TOUT_spPtr = here->SOI3TOUT_spStructPtr->CSC ;

                if ((here->SOI3toutNode != 0) && (here->SOI3bNode != 0))
                    here->SOI3TOUT_bPtr = here->SOI3TOUT_bStructPtr->CSC ;

                if ((here->SOI3gfNode != 0) && (here->SOI3toutNode != 0))
                    here->SOI3GF_toutPtr = here->SOI3GF_toutStructPtr->CSC ;

                if ((here->SOI3gbNode != 0) && (here->SOI3toutNode != 0))
                    here->SOI3GB_toutPtr = here->SOI3GB_toutStructPtr->CSC ;

                if ((here->SOI3dNodePrime != 0) && (here->SOI3toutNode != 0))
                    here->SOI3DP_toutPtr = here->SOI3DP_toutStructPtr->CSC ;

                if ((here->SOI3sNodePrime != 0) && (here->SOI3toutNode != 0))
                    here->SOI3SP_toutPtr = here->SOI3SP_toutStructPtr->CSC ;

                if ((here->SOI3bNode != 0) && (here->SOI3toutNode != 0))
                    here->SOI3B_toutPtr = here->SOI3B_toutStructPtr->CSC ;

            }
        }
    }

    return (OK) ;
}
