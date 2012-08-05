/**********
Author: 2012 Francesco Lannutti
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "soi3defs.h"
#include "ngspice/sperror.h"

int
SOI3bindCSC(GENmodel *inModel, CKTcircuit *ckt)
{
    SOI3model *model = (SOI3model *)inModel;
    int i ;

    /*  loop through all the soi3 models */
    for( ; model != NULL; model = model->SOI3nextModel ) {
	SOI3instance *here;

        /* loop through all the instances of the model */
        for (here = model->SOI3instances; here != NULL ;
	    here = here->SOI3nextInstance) {

		i = 0 ;
		if ((here->SOI3dNode != 0) && (here->SOI3dNode != 0)) {
			while (here->SOI3D_dPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->SOI3D_dPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here->SOI3dNode != 0) && (here->SOI3dNodePrime != 0)) {
			while (here->SOI3D_dpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->SOI3D_dpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here->SOI3dNodePrime != 0) && (here->SOI3dNode != 0)) {
			while (here->SOI3DP_dPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->SOI3DP_dPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here->SOI3sNode != 0) && (here->SOI3sNode != 0)) {
			while (here->SOI3S_sPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->SOI3S_sPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here->SOI3sNode != 0) && (here->SOI3sNodePrime != 0)) {
			while (here->SOI3S_spPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->SOI3S_spPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here->SOI3sNodePrime != 0) && (here->SOI3sNode != 0)) {
			while (here->SOI3SP_sPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->SOI3SP_sPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here->SOI3gfNode != 0) && (here->SOI3gfNode != 0)) {
			while (here->SOI3GF_gfPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->SOI3GF_gfPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here->SOI3gfNode != 0) && (here->SOI3gbNode != 0)) {
			while (here->SOI3GF_gbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->SOI3GF_gbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here->SOI3gfNode != 0) && (here->SOI3dNodePrime != 0)) {
			while (here->SOI3GF_dpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->SOI3GF_dpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here->SOI3gfNode != 0) && (here->SOI3sNodePrime != 0)) {
			while (here->SOI3GF_spPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->SOI3GF_spPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here->SOI3gfNode != 0) && (here->SOI3bNode != 0)) {
			while (here->SOI3GF_bPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->SOI3GF_bPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here->SOI3gbNode != 0) && (here->SOI3gfNode != 0)) {
			while (here->SOI3GB_gfPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->SOI3GB_gfPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here->SOI3gbNode != 0) && (here->SOI3gbNode != 0)) {
			while (here->SOI3GB_gbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->SOI3GB_gbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here->SOI3gbNode != 0) && (here->SOI3dNodePrime != 0)) {
			while (here->SOI3GB_dpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->SOI3GB_dpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here->SOI3gbNode != 0) && (here->SOI3sNodePrime != 0)) {
			while (here->SOI3GB_spPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->SOI3GB_spPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here->SOI3gbNode != 0) && (here->SOI3bNode != 0)) {
			while (here->SOI3GB_bPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->SOI3GB_bPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here->SOI3bNode != 0) && (here->SOI3gfNode != 0)) {
			while (here->SOI3B_gfPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->SOI3B_gfPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here->SOI3bNode != 0) && (here->SOI3gbNode != 0)) {
			while (here->SOI3B_gbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->SOI3B_gbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here->SOI3bNode != 0) && (here->SOI3dNodePrime != 0)) {
			while (here->SOI3B_dpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->SOI3B_dpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here->SOI3bNode != 0) && (here->SOI3sNodePrime != 0)) {
			while (here->SOI3B_spPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->SOI3B_spPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here->SOI3bNode != 0) && (here->SOI3bNode != 0)) {
			while (here->SOI3B_bPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->SOI3B_bPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here->SOI3dNodePrime != 0) && (here->SOI3gfNode != 0)) {
			while (here->SOI3DP_gfPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->SOI3DP_gfPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here->SOI3dNodePrime != 0) && (here->SOI3gbNode != 0)) {
			while (here->SOI3DP_gbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->SOI3DP_gbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here->SOI3dNodePrime != 0) && (here->SOI3dNodePrime != 0)) {
			while (here->SOI3DP_dpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->SOI3DP_dpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here->SOI3dNodePrime != 0) && (here->SOI3sNodePrime != 0)) {
			while (here->SOI3DP_spPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->SOI3DP_spPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here->SOI3dNodePrime != 0) && (here->SOI3bNode != 0)) {
			while (here->SOI3DP_bPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->SOI3DP_bPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here->SOI3sNodePrime != 0) && (here->SOI3gfNode != 0)) {
			while (here->SOI3SP_gfPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->SOI3SP_gfPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here->SOI3sNodePrime != 0) && (here->SOI3gbNode != 0)) {
			while (here->SOI3SP_gbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->SOI3SP_gbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here->SOI3sNodePrime != 0) && (here->SOI3dNodePrime != 0)) {
			while (here->SOI3SP_dpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->SOI3SP_dpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here->SOI3sNodePrime != 0) && (here->SOI3sNodePrime != 0)) {
			while (here->SOI3SP_spPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->SOI3SP_spPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here->SOI3sNodePrime != 0) && (here->SOI3bNode != 0)) {
			while (here->SOI3SP_bPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->SOI3SP_bPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

                if (here->SOI3rt == 0)
                {

		i = 0 ;
		if ((here->SOI3toutNode != 0) && (here->SOI3branch != 0)) {
			while (here->SOI3TOUT_ibrPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->SOI3TOUT_ibrPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here->SOI3branch != 0) && (here->SOI3toutNode != 0)) {
			while (here->SOI3IBR_toutPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->SOI3IBR_toutPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
                } /* IF */

                else
                {

		i = 0 ;
		if ((here->SOI3toutNode != 0) && (here->SOI3toutNode != 0)) {
			while (here->SOI3TOUT_toutPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->SOI3TOUT_toutPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}

                if (here->SOI3numThermalNodes > 1)
                {

		i = 0 ;
		if ((here->SOI3toutNode != 0) && (here->SOI3tout1Node != 0)) {
			while (here->SOI3TOUT_tout1Ptr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->SOI3TOUT_tout1Ptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here->SOI3tout1Node != 0) && (here->SOI3toutNode != 0)) {
			while (here->SOI3TOUT1_toutPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->SOI3TOUT1_toutPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here->SOI3tout1Node != 0) && (here->SOI3tout1Node != 0)) {
			while (here->SOI3TOUT1_tout1Ptr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->SOI3TOUT1_tout1Ptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
                }

                if (here->SOI3numThermalNodes > 2)
                {

		i = 0 ;
		if ((here->SOI3tout1Node != 0) && (here->SOI3tout2Node != 0)) {
			while (here->SOI3TOUT1_tout2Ptr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->SOI3TOUT1_tout2Ptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here->SOI3tout2Node != 0) && (here->SOI3tout1Node != 0)) {
			while (here->SOI3TOUT2_tout1Ptr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->SOI3TOUT2_tout1Ptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here->SOI3tout2Node != 0) && (here->SOI3tout2Node != 0)) {
			while (here->SOI3TOUT2_tout2Ptr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->SOI3TOUT2_tout2Ptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
                }

                if (here->SOI3numThermalNodes > 3)
                {

		i = 0 ;
		if ((here->SOI3tout2Node != 0) && (here->SOI3tout3Node != 0)) {
			while (here->SOI3TOUT2_tout3Ptr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->SOI3TOUT2_tout3Ptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here->SOI3tout3Node != 0) && (here->SOI3tout2Node != 0)) {
			while (here->SOI3TOUT3_tout2Ptr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->SOI3TOUT3_tout2Ptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here->SOI3tout3Node != 0) && (here->SOI3tout3Node != 0)) {
			while (here->SOI3TOUT3_tout3Ptr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->SOI3TOUT3_tout3Ptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
                }

                if (here->SOI3numThermalNodes > 4)
                {

		i = 0 ;
		if ((here->SOI3tout3Node != 0) && (here->SOI3tout4Node != 0)) {
			while (here->SOI3TOUT3_tout4Ptr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->SOI3TOUT3_tout4Ptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here->SOI3tout4Node != 0) && (here->SOI3tout3Node != 0)) {
			while (here->SOI3TOUT4_tout3Ptr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->SOI3TOUT4_tout3Ptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here->SOI3tout4Node != 0) && (here->SOI3tout4Node != 0)) {
			while (here->SOI3TOUT4_tout4Ptr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->SOI3TOUT4_tout4Ptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
                }

		i = 0 ;
		if ((here->SOI3toutNode != 0) && (here->SOI3toutNode != 0)) {
			while (here->SOI3TOUT_toutPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->SOI3TOUT_toutPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here->SOI3toutNode != 0) && (here->SOI3gfNode != 0)) {
			while (here->SOI3TOUT_gfPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->SOI3TOUT_gfPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here->SOI3toutNode != 0) && (here->SOI3gbNode != 0)) {
			while (here->SOI3TOUT_gbPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->SOI3TOUT_gbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here->SOI3toutNode != 0) && (here->SOI3dNodePrime != 0)) {
			while (here->SOI3TOUT_dpPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->SOI3TOUT_dpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here->SOI3toutNode != 0) && (here->SOI3sNodePrime != 0)) {
			while (here->SOI3TOUT_spPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->SOI3TOUT_spPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here->SOI3toutNode != 0) && (here->SOI3bNode != 0)) {
			while (here->SOI3TOUT_bPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->SOI3TOUT_bPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here->SOI3gfNode != 0) && (here->SOI3toutNode != 0)) {
			while (here->SOI3GF_toutPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->SOI3GF_toutPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here->SOI3gbNode != 0) && (here->SOI3toutNode != 0)) {
			while (here->SOI3GB_toutPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->SOI3GB_toutPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here->SOI3dNodePrime != 0) && (here->SOI3toutNode != 0)) {
			while (here->SOI3DP_toutPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->SOI3DP_toutPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here->SOI3sNodePrime != 0) && (here->SOI3toutNode != 0)) {
			while (here->SOI3SP_toutPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->SOI3SP_toutPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
		i = 0 ;
		if ((here->SOI3bNode != 0) && (here->SOI3toutNode != 0)) {
			while (here->SOI3B_toutPtr != ckt->CKTmatrix->CKTbind_Sparse [i]) i ++ ;
			here->SOI3B_toutPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
		}
                } /* ELSE */
	}
    }
    return(OK);
}

int
SOI3bindCSCComplex(GENmodel *inModel, CKTcircuit *ckt)
{
    SOI3model *model = (SOI3model *)inModel;
    int i ;

    /*  loop through all the soi3 models */
    for( ; model != NULL; model = model->SOI3nextModel ) {
	SOI3instance *here;

        /* loop through all the instances of the model */
        for (here = model->SOI3instances; here != NULL ;
	    here = here->SOI3nextInstance) {

		i = 0 ;
		if ((here->SOI3dNode != 0) && (here->SOI3dNode != 0)) {
			while (here->SOI3D_dPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->SOI3D_dPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here->SOI3dNode != 0) && (here->SOI3dNodePrime != 0)) {
			while (here->SOI3D_dpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->SOI3D_dpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here->SOI3dNodePrime != 0) && (here->SOI3dNode != 0)) {
			while (here->SOI3DP_dPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->SOI3DP_dPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here->SOI3sNode != 0) && (here->SOI3sNode != 0)) {
			while (here->SOI3S_sPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->SOI3S_sPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here->SOI3sNode != 0) && (here->SOI3sNodePrime != 0)) {
			while (here->SOI3S_spPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->SOI3S_spPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here->SOI3sNodePrime != 0) && (here->SOI3sNode != 0)) {
			while (here->SOI3SP_sPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->SOI3SP_sPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here->SOI3gfNode != 0) && (here->SOI3gfNode != 0)) {
			while (here->SOI3GF_gfPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->SOI3GF_gfPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here->SOI3gfNode != 0) && (here->SOI3gbNode != 0)) {
			while (here->SOI3GF_gbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->SOI3GF_gbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here->SOI3gfNode != 0) && (here->SOI3dNodePrime != 0)) {
			while (here->SOI3GF_dpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->SOI3GF_dpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here->SOI3gfNode != 0) && (here->SOI3sNodePrime != 0)) {
			while (here->SOI3GF_spPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->SOI3GF_spPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here->SOI3gfNode != 0) && (here->SOI3bNode != 0)) {
			while (here->SOI3GF_bPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->SOI3GF_bPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here->SOI3gbNode != 0) && (here->SOI3gfNode != 0)) {
			while (here->SOI3GB_gfPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->SOI3GB_gfPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here->SOI3gbNode != 0) && (here->SOI3gbNode != 0)) {
			while (here->SOI3GB_gbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->SOI3GB_gbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here->SOI3gbNode != 0) && (here->SOI3dNodePrime != 0)) {
			while (here->SOI3GB_dpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->SOI3GB_dpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here->SOI3gbNode != 0) && (here->SOI3sNodePrime != 0)) {
			while (here->SOI3GB_spPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->SOI3GB_spPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here->SOI3gbNode != 0) && (here->SOI3bNode != 0)) {
			while (here->SOI3GB_bPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->SOI3GB_bPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here->SOI3bNode != 0) && (here->SOI3gfNode != 0)) {
			while (here->SOI3B_gfPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->SOI3B_gfPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here->SOI3bNode != 0) && (here->SOI3gbNode != 0)) {
			while (here->SOI3B_gbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->SOI3B_gbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here->SOI3bNode != 0) && (here->SOI3dNodePrime != 0)) {
			while (here->SOI3B_dpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->SOI3B_dpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here->SOI3bNode != 0) && (here->SOI3sNodePrime != 0)) {
			while (here->SOI3B_spPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->SOI3B_spPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here->SOI3bNode != 0) && (here->SOI3bNode != 0)) {
			while (here->SOI3B_bPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->SOI3B_bPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here->SOI3dNodePrime != 0) && (here->SOI3gfNode != 0)) {
			while (here->SOI3DP_gfPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->SOI3DP_gfPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here->SOI3dNodePrime != 0) && (here->SOI3gbNode != 0)) {
			while (here->SOI3DP_gbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->SOI3DP_gbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here->SOI3dNodePrime != 0) && (here->SOI3dNodePrime != 0)) {
			while (here->SOI3DP_dpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->SOI3DP_dpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here->SOI3dNodePrime != 0) && (here->SOI3sNodePrime != 0)) {
			while (here->SOI3DP_spPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->SOI3DP_spPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here->SOI3dNodePrime != 0) && (here->SOI3bNode != 0)) {
			while (here->SOI3DP_bPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->SOI3DP_bPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here->SOI3sNodePrime != 0) && (here->SOI3gfNode != 0)) {
			while (here->SOI3SP_gfPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->SOI3SP_gfPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here->SOI3sNodePrime != 0) && (here->SOI3gbNode != 0)) {
			while (here->SOI3SP_gbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->SOI3SP_gbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here->SOI3sNodePrime != 0) && (here->SOI3dNodePrime != 0)) {
			while (here->SOI3SP_dpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->SOI3SP_dpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here->SOI3sNodePrime != 0) && (here->SOI3sNodePrime != 0)) {
			while (here->SOI3SP_spPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->SOI3SP_spPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here->SOI3sNodePrime != 0) && (here->SOI3bNode != 0)) {
			while (here->SOI3SP_bPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->SOI3SP_bPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

                if (here->SOI3rt == 0)
                {

		i = 0 ;
		if ((here->SOI3toutNode != 0) && (here->SOI3branch != 0)) {
			while (here->SOI3TOUT_ibrPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->SOI3TOUT_ibrPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here->SOI3branch != 0) && (here->SOI3toutNode != 0)) {
			while (here->SOI3IBR_toutPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->SOI3IBR_toutPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
                } /* IF */

                else
                {

		i = 0 ;
		if ((here->SOI3toutNode != 0) && (here->SOI3toutNode != 0)) {
			while (here->SOI3TOUT_toutPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->SOI3TOUT_toutPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}

                if (here->SOI3numThermalNodes > 1)
                {

		i = 0 ;
		if ((here->SOI3toutNode != 0) && (here->SOI3tout1Node != 0)) {
			while (here->SOI3TOUT_tout1Ptr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->SOI3TOUT_tout1Ptr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here->SOI3tout1Node != 0) && (here->SOI3toutNode != 0)) {
			while (here->SOI3TOUT1_toutPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->SOI3TOUT1_toutPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here->SOI3tout1Node != 0) && (here->SOI3tout1Node != 0)) {
			while (here->SOI3TOUT1_tout1Ptr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->SOI3TOUT1_tout1Ptr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
                }

                if (here->SOI3numThermalNodes > 2)
                {

		i = 0 ;
		if ((here->SOI3tout1Node != 0) && (here->SOI3tout2Node != 0)) {
			while (here->SOI3TOUT1_tout2Ptr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->SOI3TOUT1_tout2Ptr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here->SOI3tout2Node != 0) && (here->SOI3tout1Node != 0)) {
			while (here->SOI3TOUT2_tout1Ptr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->SOI3TOUT2_tout1Ptr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here->SOI3tout2Node != 0) && (here->SOI3tout2Node != 0)) {
			while (here->SOI3TOUT2_tout2Ptr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->SOI3TOUT2_tout2Ptr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
                }

                if (here->SOI3numThermalNodes > 3)
                {

		i = 0 ;
		if ((here->SOI3tout2Node != 0) && (here->SOI3tout3Node != 0)) {
			while (here->SOI3TOUT2_tout3Ptr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->SOI3TOUT2_tout3Ptr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here->SOI3tout3Node != 0) && (here->SOI3tout2Node != 0)) {
			while (here->SOI3TOUT3_tout2Ptr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->SOI3TOUT3_tout2Ptr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here->SOI3tout3Node != 0) && (here->SOI3tout3Node != 0)) {
			while (here->SOI3TOUT3_tout3Ptr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->SOI3TOUT3_tout3Ptr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
                }

                if (here->SOI3numThermalNodes > 4)
                {

		i = 0 ;
		if ((here->SOI3tout3Node != 0) && (here->SOI3tout4Node != 0)) {
			while (here->SOI3TOUT3_tout4Ptr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->SOI3TOUT3_tout4Ptr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here->SOI3tout4Node != 0) && (here->SOI3tout3Node != 0)) {
			while (here->SOI3TOUT4_tout3Ptr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->SOI3TOUT4_tout3Ptr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here->SOI3tout4Node != 0) && (here->SOI3tout4Node != 0)) {
			while (here->SOI3TOUT4_tout4Ptr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->SOI3TOUT4_tout4Ptr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
                }

		i = 0 ;
		if ((here->SOI3toutNode != 0) && (here->SOI3toutNode != 0)) {
			while (here->SOI3TOUT_toutPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->SOI3TOUT_toutPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here->SOI3toutNode != 0) && (here->SOI3gfNode != 0)) {
			while (here->SOI3TOUT_gfPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->SOI3TOUT_gfPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here->SOI3toutNode != 0) && (here->SOI3gbNode != 0)) {
			while (here->SOI3TOUT_gbPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->SOI3TOUT_gbPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here->SOI3toutNode != 0) && (here->SOI3dNodePrime != 0)) {
			while (here->SOI3TOUT_dpPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->SOI3TOUT_dpPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here->SOI3toutNode != 0) && (here->SOI3sNodePrime != 0)) {
			while (here->SOI3TOUT_spPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->SOI3TOUT_spPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here->SOI3toutNode != 0) && (here->SOI3bNode != 0)) {
			while (here->SOI3TOUT_bPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->SOI3TOUT_bPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here->SOI3gfNode != 0) && (here->SOI3toutNode != 0)) {
			while (here->SOI3GF_toutPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->SOI3GF_toutPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here->SOI3gbNode != 0) && (here->SOI3toutNode != 0)) {
			while (here->SOI3GB_toutPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->SOI3GB_toutPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here->SOI3dNodePrime != 0) && (here->SOI3toutNode != 0)) {
			while (here->SOI3DP_toutPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->SOI3DP_toutPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here->SOI3sNodePrime != 0) && (here->SOI3toutNode != 0)) {
			while (here->SOI3SP_toutPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->SOI3SP_toutPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
		i = 0 ;
		if ((here->SOI3bNode != 0) && (here->SOI3toutNode != 0)) {
			while (here->SOI3B_toutPtr != ckt->CKTmatrix->CKTbind_CSC [i]) i ++ ;
			here->SOI3B_toutPtr = ckt->CKTmatrix->CKTbind_CSC_Complex [i] ;
		}
                } /* ELSE */
	}
    }
    return(OK);
}

int
SOI3bindCSCComplexToReal (GENmodel *inModel, CKTcircuit *ckt)
{
    SOI3model *model = (SOI3model *)inModel ;
    SOI3instance *here ;
    int i ;

    /*  loop through all the SiliconOnInsulator3 models */
    for ( ; model != NULL ; model = model->SOI3nextModel)
    {
        /* loop through all the instances of the model */
        for (here = model->SOI3instances ; here != NULL ; here = here->SOI3nextInstance)
        {
            i = 0 ;
            if ((here->SOI3dNode != 0) && (here->SOI3dNode != 0))
            {
                while (here->SOI3D_dPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->SOI3D_dPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->SOI3dNode != 0) && (here->SOI3dNodePrime != 0))
            {
                while (here->SOI3D_dpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->SOI3D_dpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->SOI3dNodePrime != 0) && (here->SOI3dNode != 0))
            {
                while (here->SOI3DP_dPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->SOI3DP_dPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->SOI3sNode != 0) && (here->SOI3sNode != 0))
            {
                while (here->SOI3S_sPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->SOI3S_sPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->SOI3sNode != 0) && (here->SOI3sNodePrime != 0))
            {
                while (here->SOI3S_spPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->SOI3S_spPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->SOI3sNodePrime != 0) && (here->SOI3sNode != 0))
            {
                while (here->SOI3SP_sPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->SOI3SP_sPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->SOI3gfNode != 0) && (here->SOI3gfNode != 0))
            {
                while (here->SOI3GF_gfPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->SOI3GF_gfPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->SOI3gfNode != 0) && (here->SOI3gbNode != 0))
            {
                while (here->SOI3GF_gbPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->SOI3GF_gbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->SOI3gfNode != 0) && (here->SOI3dNodePrime != 0))
            {
                while (here->SOI3GF_dpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->SOI3GF_dpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->SOI3gfNode != 0) && (here->SOI3sNodePrime != 0))
            {
                while (here->SOI3GF_spPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->SOI3GF_spPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->SOI3gfNode != 0) && (here->SOI3bNode != 0))
            {
                while (here->SOI3GF_bPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->SOI3GF_bPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->SOI3gbNode != 0) && (here->SOI3gfNode != 0))
            {
                while (here->SOI3GB_gfPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->SOI3GB_gfPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->SOI3gbNode != 0) && (here->SOI3gbNode != 0))
            {
                while (here->SOI3GB_gbPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->SOI3GB_gbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->SOI3gbNode != 0) && (here->SOI3dNodePrime != 0))
            {
                while (here->SOI3GB_dpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->SOI3GB_dpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->SOI3gbNode != 0) && (here->SOI3sNodePrime != 0))
            {
                while (here->SOI3GB_spPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->SOI3GB_spPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->SOI3gbNode != 0) && (here->SOI3bNode != 0))
            {
                while (here->SOI3GB_bPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->SOI3GB_bPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->SOI3bNode != 0) && (here->SOI3gfNode != 0))
            {
                while (here->SOI3B_gfPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->SOI3B_gfPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->SOI3bNode != 0) && (here->SOI3gbNode != 0))
            {
                while (here->SOI3B_gbPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->SOI3B_gbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->SOI3bNode != 0) && (here->SOI3dNodePrime != 0))
            {
                while (here->SOI3B_dpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->SOI3B_dpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->SOI3bNode != 0) && (here->SOI3sNodePrime != 0))
            {
                while (here->SOI3B_spPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->SOI3B_spPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->SOI3bNode != 0) && (here->SOI3bNode != 0))
            {
                while (here->SOI3B_bPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->SOI3B_bPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->SOI3dNodePrime != 0) && (here->SOI3gfNode != 0))
            {
                while (here->SOI3DP_gfPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->SOI3DP_gfPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->SOI3dNodePrime != 0) && (here->SOI3gbNode != 0))
            {
                while (here->SOI3DP_gbPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->SOI3DP_gbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->SOI3dNodePrime != 0) && (here->SOI3dNodePrime != 0))
            {
                while (here->SOI3DP_dpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->SOI3DP_dpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->SOI3dNodePrime != 0) && (here->SOI3sNodePrime != 0))
            {
                while (here->SOI3DP_spPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->SOI3DP_spPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->SOI3dNodePrime != 0) && (here->SOI3bNode != 0))
            {
                while (here->SOI3DP_bPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->SOI3DP_bPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->SOI3sNodePrime != 0) && (here->SOI3gfNode != 0))
            {
                while (here->SOI3SP_gfPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->SOI3SP_gfPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->SOI3sNodePrime != 0) && (here->SOI3gbNode != 0))
            {
                while (here->SOI3SP_gbPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->SOI3SP_gbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->SOI3sNodePrime != 0) && (here->SOI3dNodePrime != 0))
            {
                while (here->SOI3SP_dpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->SOI3SP_dpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->SOI3sNodePrime != 0) && (here->SOI3sNodePrime != 0))
            {
                while (here->SOI3SP_spPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->SOI3SP_spPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->SOI3sNodePrime != 0) && (here->SOI3bNode != 0))
            {
                while (here->SOI3SP_bPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->SOI3SP_bPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->SOI3toutNode != 0) && (here->SOI3branch != 0))
            {
                while (here->SOI3TOUT_ibrPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->SOI3TOUT_ibrPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->SOI3branch != 0) && (here->SOI3toutNode != 0))
            {
                while (here->SOI3IBR_toutPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->SOI3IBR_toutPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->SOI3toutNode != 0) && (here->SOI3toutNode != 0))
            {
                while (here->SOI3TOUT_toutPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->SOI3TOUT_toutPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->SOI3toutNode != 0) && (here->SOI3tout1Node != 0))
            {
                while (here->SOI3TOUT_tout1Ptr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->SOI3TOUT_tout1Ptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->SOI3tout1Node != 0) && (here->SOI3toutNode != 0))
            {
                while (here->SOI3TOUT1_toutPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->SOI3TOUT1_toutPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->SOI3tout1Node != 0) && (here->SOI3tout1Node != 0))
            {
                while (here->SOI3TOUT1_tout1Ptr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->SOI3TOUT1_tout1Ptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->SOI3tout1Node != 0) && (here->SOI3tout2Node != 0))
            {
                while (here->SOI3TOUT1_tout2Ptr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->SOI3TOUT1_tout2Ptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->SOI3tout2Node != 0) && (here->SOI3tout1Node != 0))
            {
                while (here->SOI3TOUT2_tout1Ptr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->SOI3TOUT2_tout1Ptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->SOI3tout2Node != 0) && (here->SOI3tout2Node != 0))
            {
                while (here->SOI3TOUT2_tout2Ptr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->SOI3TOUT2_tout2Ptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->SOI3tout2Node != 0) && (here->SOI3tout3Node != 0))
            {
                while (here->SOI3TOUT2_tout3Ptr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->SOI3TOUT2_tout3Ptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->SOI3tout3Node != 0) && (here->SOI3tout2Node != 0))
            {
                while (here->SOI3TOUT3_tout2Ptr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->SOI3TOUT3_tout2Ptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->SOI3tout3Node != 0) && (here->SOI3tout3Node != 0))
            {
                while (here->SOI3TOUT3_tout3Ptr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->SOI3TOUT3_tout3Ptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->SOI3tout3Node != 0) && (here->SOI3tout4Node != 0))
            {
                while (here->SOI3TOUT3_tout4Ptr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->SOI3TOUT3_tout4Ptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->SOI3tout4Node != 0) && (here->SOI3tout3Node != 0))
            {
                while (here->SOI3TOUT4_tout3Ptr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->SOI3TOUT4_tout3Ptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->SOI3tout4Node != 0) && (here->SOI3tout4Node != 0))
            {
                while (here->SOI3TOUT4_tout4Ptr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->SOI3TOUT4_tout4Ptr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->SOI3toutNode != 0) && (here->SOI3toutNode != 0))
            {
                while (here->SOI3TOUT_toutPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->SOI3TOUT_toutPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->SOI3toutNode != 0) && (here->SOI3gfNode != 0))
            {
                while (here->SOI3TOUT_gfPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->SOI3TOUT_gfPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->SOI3toutNode != 0) && (here->SOI3gbNode != 0))
            {
                while (here->SOI3TOUT_gbPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->SOI3TOUT_gbPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->SOI3toutNode != 0) && (here->SOI3dNodePrime != 0))
            {
                while (here->SOI3TOUT_dpPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->SOI3TOUT_dpPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->SOI3toutNode != 0) && (here->SOI3sNodePrime != 0))
            {
                while (here->SOI3TOUT_spPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->SOI3TOUT_spPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->SOI3toutNode != 0) && (here->SOI3bNode != 0))
            {
                while (here->SOI3TOUT_bPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->SOI3TOUT_bPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->SOI3gfNode != 0) && (here->SOI3toutNode != 0))
            {
                while (here->SOI3GF_toutPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->SOI3GF_toutPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->SOI3gbNode != 0) && (here->SOI3toutNode != 0))
            {
                while (here->SOI3GB_toutPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->SOI3GB_toutPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->SOI3dNodePrime != 0) && (here->SOI3toutNode != 0))
            {
                while (here->SOI3DP_toutPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->SOI3DP_toutPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->SOI3sNodePrime != 0) && (here->SOI3toutNode != 0))
            {
                while (here->SOI3SP_toutPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->SOI3SP_toutPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }

            i = 0 ;
            if ((here->SOI3bNode != 0) && (here->SOI3toutNode != 0))
            {
                while (here->SOI3B_toutPtr != ckt->CKTmatrix->CKTbind_CSC_Complex [i]) i ++ ;
                here->SOI3B_toutPtr = ckt->CKTmatrix->CKTbind_CSC [i] ;
            }
        }
    }

    return (OK) ;
}