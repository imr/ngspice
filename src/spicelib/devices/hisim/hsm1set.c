/***********************************************************************
 HiSIM v1.1.0
 File: hsm1set.c of HiSIM v1.1.0

 Copyright (C) 2002 STARC

 June 30, 2002: developed by Hiroshima University and STARC
 June 30, 2002: posted by Keiichi MORIKAWA, STARC Physical Design Group
***********************************************************************/

/*
 * Modified by Paolo Nenzi 2002
 * ngspice integration
 */

#include "ngspice.h"
#include "smpdefs.h"
#include "cktdefs.h"
#include "hsm1def.h"
#include "const.h"
#include "sperror.h"
#include "suffix.h"

int HSM1setup(SMPmatrix *matrix, GENmodel *inModel, CKTcircuit *ckt, 
              int *states)
     /* load the HSM1 device structure with those pointers needed later 
      * for fast matrix loading 
      */
{
  HSM1model *model = (HSM1model*)inModel;
  HSM1instance *here;
  int error;
  CKTnode *tmp;
  
  CKTnode *tmpNode;
  IFuid tmpName;
  
  /*  loop through all the HSM1 device models */
  for ( ;model != NULL ;model = model->HSM1nextModel ) {
    /* Default value Processing for HSM1 MOSFET Models */
    if ( !model->HSM1_type_Given )
      model->HSM1_type = NMOS ;
    /***/
    if ( !model->HSM1_info_Given ) model->HSM1_info = 0 ;
    if ( !model->HSM1_noise_Given) model->HSM1_noise = 5; /* select noise model 5 */
    if ( !model->HSM1_version_Given) model->HSM1_version = 100; /* default 100 */
    if ( !model->HSM1_corsrd_Given ) model->HSM1_corsrd = 0 ;
    if ( !model->HSM1_coiprv_Given ) model->HSM1_coiprv = 1 ;
    if ( !model->HSM1_copprv_Given ) model->HSM1_copprv = 1 ;
    if ( !model->HSM1_cocgso_Given ) model->HSM1_cocgso = 0 ;
    if ( !model->HSM1_cocgdo_Given ) model->HSM1_cocgdo = 0 ;
    if ( !model->HSM1_cocgbo_Given ) model->HSM1_cocgbo = 0 ;
    if ( !model->HSM1_coadov_Given ) model->HSM1_coadov = 1 ;
    if ( !model->HSM1_coxx08_Given ) model->HSM1_coxx08 = 0 ;
    if ( !model->HSM1_coxx09_Given ) model->HSM1_coxx09 = 0 ;
    if ( !model->HSM1_coisub_Given ) model->HSM1_coisub = 0 ;
    if ( !model->HSM1_coiigs_Given ) model->HSM1_coiigs = 0 ;
    if ( !model->HSM1_cogidl_Given ) model->HSM1_cogidl = 0 ;
    if ( !model->HSM1_coovlp_Given ) model->HSM1_coovlp = 0 ;
    if ( !model->HSM1_conois_Given ) model->HSM1_conois = 0 ;
    if ( model->HSM1_version == 110 ) {/* HiSIM1.1 */
      if ( !model->HSM1_coisti_Given ) model->HSM1_coisti = 0 ;
    }
    /***/
    if ( !model->HSM1_vmax_Given ) model->HSM1_vmax = 1.00e+7 ;
    if ( !model->HSM1_bgtmp1_Given ) model->HSM1_bgtmp1 = 9.03e-5 ;
    if ( !model->HSM1_bgtmp2_Given ) model->HSM1_bgtmp2 = 3.05e-7 ;
    if ( !model->HSM1_tox_Given ) model->HSM1_tox = 3.60e-9 ;
    if ( !model->HSM1_dl_Given ) model->HSM1_dl = 0.0 ;
    if ( !model->HSM1_dw_Given ) model->HSM1_dw = 0.0 ;
    if ( model->HSM1_version == 100 ) { /* HiSIM1.0 */
      if ( !model->HSM1_xj_Given ) model->HSM1_xj = 0.0 ;
      if ( model->HSM1_xqy_Given ) {
	printf("warning(HiSIM): the model parameter XQY is only available in VERSION = 110.\n");
	printf("                XQY = %f - ignored\n", model->HSM1_xqy);
      }
    }
    else if ( model->HSM1_version == 110 ) { /* HiSIM1.1 */
      if ( !model->HSM1_xqy_Given ) model->HSM1_xqy = 0.0;
      if ( model->HSM1_xj_Given ) {
	printf("warning(HiSIM): the model parameter XJ is only available in VERSION = 100.\n");
	printf("                 XJ = %f - ignored\n", model->HSM1_xj);
      }
    }
    if ( !model->HSM1_rs_Given ) model->HSM1_rs = 0.0 ;
    if ( !model->HSM1_rd_Given ) model->HSM1_rd = 0.0 ;
    if ( !model->HSM1_vfbc_Given ) model->HSM1_vfbc = -0.722729 ;
    if ( !model->HSM1_nsubc_Given ) model->HSM1_nsubc = 5.94e+17 ;
    if ( !model->HSM1_parl1_Given ) model->HSM1_parl1 = 1.0 ;
    if ( !model->HSM1_parl2_Given ) model->HSM1_parl2 = 2.20e-8 ;
    if ( !model->HSM1_lp_Given ) model->HSM1_lp = 0.0 ;
    if ( !model->HSM1_nsubp_Given ) model->HSM1_nsubp = 5.94e+17 ;
    if ( !model->HSM1_scp1_Given ) model->HSM1_scp1 = 0.0 ;
    if ( !model->HSM1_scp2_Given ) model->HSM1_scp2 = 0.0 ;
    if ( !model->HSM1_scp3_Given ) model->HSM1_scp3 = 0.0 ;
    if ( !model->HSM1_sc1_Given ) model->HSM1_sc1 = 13.5 ;
    if ( !model->HSM1_sc2_Given ) model->HSM1_sc2 = 1.8 ;
    if ( !model->HSM1_sc3_Given ) model->HSM1_sc3 = 0.0 ;
    if ( !model->HSM1_pgd1_Given ) model->HSM1_pgd1 = 0.0 ;
    if ( !model->HSM1_pgd2_Given ) model->HSM1_pgd2 = 0.0 ;
    if ( !model->HSM1_pgd3_Given ) model->HSM1_pgd3 = 0.0 ;
    if ( !model->HSM1_ndep_Given ) model->HSM1_ndep = 1.0 ;
    if ( !model->HSM1_ninv_Given ) model->HSM1_ninv = 0.5 ;
    if ( !model->HSM1_ninvd_Given ) model->HSM1_ninvd = 0.0 ;
    if ( !model->HSM1_muecb0_Given ) model->HSM1_muecb0 = 300.0 ;
    if ( !model->HSM1_muecb1_Given ) model->HSM1_muecb1 = 30.0 ;
    if ( !model->HSM1_mueph1_Given ) model->HSM1_mueph1 = 1.00e+7 ;
    if ( !model->HSM1_mueph0_Given ) model->HSM1_mueph0 = 0.295 ;
    if ( !model->HSM1_mueph2_Given ) model->HSM1_mueph2 = 0.0 ;
    if ( !model->HSM1_w0_Given ) model->HSM1_w0 = 0.0 ;
    if ( !model->HSM1_muesr1_Given ) model->HSM1_muesr1 = 7.00e+8 ;
    if ( !model->HSM1_muesr0_Given ) model->HSM1_muesr0 = 1.0 ;
    if ( !model->HSM1_muetmp_Given ) model->HSM1_muetmp = 0.0 ;
    /***/
    if ( !model->HSM1_bb_Given ) {
      if (model->HSM1_type == NMOS) model->HSM1_bb = 2.0 ;
      else model->HSM1_bb = 1.0 ;
    }
    /***/
    if ( !model->HSM1_vds0_Given ) model->HSM1_vds0 = 0.05 ;
    if ( !model->HSM1_bc0_Given ) model->HSM1_bc0 = 0.0 ;
    if ( !model->HSM1_bc1_Given ) model->HSM1_bc1 = 0.0 ;
    if ( !model->HSM1_sub1_Given ) model->HSM1_sub1 = 0.0 ;
    if ( !model->HSM1_sub2_Given ) model->HSM1_sub2 = -70.0 ;
    if ( !model->HSM1_sub3_Given ) model->HSM1_sub3 = 1.0 ;
    if ( model->HSM1_version == 110)  { /* HiSIM1.1 */
      if ( !model->HSM1_wvthsc_Given ) model->HSM1_wvthsc = 0.0 ;
      if ( !model->HSM1_nsti_Given ) model->HSM1_nsti = 1.0e17 ;
      if ( !model->HSM1_wsti_Given ) model->HSM1_wsti = 0.0 ;
    }
    if ( !model->HSM1_tpoly_Given ) model->HSM1_tpoly = 0.0 ;
    if ( !model->HSM1_js0_Given ) model->HSM1_js0 = 1.0e-4 ;
    if ( !model->HSM1_js0sw_Given ) model->HSM1_js0sw = 0.0 ;
    if ( !model->HSM1_nj_Given ) model->HSM1_nj = 1.0 ;
    if ( !model->HSM1_njsw_Given ) model->HSM1_njsw = 1.0 ;
    if ( !model->HSM1_xti_Given ) model->HSM1_xti = 3.0 ;
    if ( !model->HSM1_cj_Given ) model->HSM1_cj =  8.397247e-04;
    if ( !model->HSM1_cjsw_Given ) model->HSM1_cjsw = 5.0e-10 ;
    if ( !model->HSM1_cjswg_Given ) model->HSM1_cjswg = 5.0e-10 ;
    if ( !model->HSM1_mj_Given ) model->HSM1_mj = 0.5 ;
    if ( !model->HSM1_mjsw_Given ) model->HSM1_mjsw = 0.33 ;
    if ( !model->HSM1_mjswg_Given ) model->HSM1_mjswg = 0.33 ;
    if ( !model->HSM1_pb_Given ) model->HSM1_pb = 1.0 ;
    if ( !model->HSM1_pbsw_Given ) model->HSM1_pbsw = 1.0 ;
    if ( !model->HSM1_pbswg_Given ) model->HSM1_pbswg = 1.0 ;
    if ( !model->HSM1_xpolyd_Given ) model->HSM1_xpolyd = 0.0 ;
    if ( !model->HSM1_clm1_Given ) model->HSM1_clm1 = 0.3e0 ;
    if ( !model->HSM1_clm2_Given ) model->HSM1_clm2 = 0.0 ;
    if ( !model->HSM1_clm3_Given ) model->HSM1_clm3 = 0.0 ;
    if ( !model->HSM1_rpock1_Given ) model->HSM1_rpock1 = 0.0 ;
    if ( !model->HSM1_rpock2_Given ) model->HSM1_rpock2 = 0.0 ;
    if ( model->HSM1_version == 110 ) { /* HiSIM1.1 */
      if ( !model->HSM1_rpocp1_Given ) model->HSM1_rpocp1 = 0.0 ;
      if ( !model->HSM1_rpocp2_Given ) model->HSM1_rpocp2 = 0.0 ;
    }
    if ( !model->HSM1_vover_Given ) model->HSM1_vover = 0.0 ;
    if ( !model->HSM1_voverp_Given ) model->HSM1_voverp = 0.0 ;
    if ( !model->HSM1_wfc_Given ) model->HSM1_wfc = 0.0 ;
    if ( !model->HSM1_qme1_Given ) model->HSM1_qme1 = 0.0 ;
    if ( !model->HSM1_qme2_Given ) model->HSM1_qme2 = 0.0 ;
    if ( !model->HSM1_qme3_Given ) model->HSM1_qme3 = 0.0 ;
    if ( !model->HSM1_gidl1_Given ) model->HSM1_gidl1 = 0.0 ;
    if ( !model->HSM1_gidl2_Given ) model->HSM1_gidl2 = 0.0 ;
    if ( !model->HSM1_gidl3_Given ) model->HSM1_gidl3 = 0.0 ;
    if ( !model->HSM1_gleak1_Given ) model->HSM1_gleak1 = 0.0 ;
    if ( !model->HSM1_gleak2_Given ) model->HSM1_gleak2 = 0.0 ;
    if ( !model->HSM1_gleak3_Given ) model->HSM1_gleak3 = 0.0 ;
    if ( !model->HSM1_vzadd0_Given ) model->HSM1_vzadd0 = 1.0e-2 ;
    if ( !model->HSM1_pzadd0_Given ) model->HSM1_pzadd0 = 1.0e-3 ;
    if ( !model->HSM1_nftrp_Given ) model->HSM1_nftrp = 100e9 ;
    if ( !model->HSM1_nfalp_Given ) model->HSM1_nfalp = 2.00e-15 ;
    if ( !model->HSM1_cit_Given ) model->HSM1_cit = 0.0 ;

    /* for flicker noise the same as BSIM3 */
    if ( !model->HSM1_ef_Given ) model->HSM1_ef = 0.0;
    if ( !model->HSM1_af_Given ) model->HSM1_af = 1.0;
    if ( !model->HSM1_kf_Given ) model->HSM1_kf = 0.0;

    /* loop through all the instances of the model */
    for ( here = model->HSM1instances ;here != NULL ;
	  here = here->HSM1nextInstance ) {
	  
    if (here->HSM1owner == ARCHme)
    {  
      /* allocate a chunk of the state vector */
      here->HSM1states = *states;
      *states += HSM1numStates;
    }
    
      /* perform the parameter defaulting */
      /*
      if ( !here->HSM1_l_Given ) here->HSM1_l = 1.50e-4 ;
      if ( !here->HSM1_w_Given ) here->HSM1_w = 5.55e-4 ;
      */
      if ( !here->HSM1_l_Given ) here->HSM1_l = 5.0e-6 ;
      if ( !here->HSM1_w_Given ) here->HSM1_w = 5.0e-6 ;
      if ( !here->HSM1_m_Given ) here->HSM1_m = 1;
      if ( !here->HSM1_ad_Given ) here->HSM1_ad = 0.0 ;
      if ( !here->HSM1_as_Given ) here->HSM1_as = 0.0 ;
      if ( !here->HSM1_pd_Given ) here->HSM1_pd = 0.0 ;
      if ( !here->HSM1_ps_Given ) here->HSM1_ps = 0.0 ;
      if ( !here->HSM1_nrd_Given ) here->HSM1_nrd = 0.0 ;
      if ( !here->HSM1_nrs_Given ) here->HSM1_nrs = 0.0 ;
      if ( !here->HSM1_temp_Given ) here->HSM1_temp = 300.15 ;
      if ( !here->HSM1_dtemp_Given ) here->HSM1_dtemp = 0.0 ;

      if ( !here->HSM1_icVBS_Given ) here->HSM1_icVBS = 0.0;
      if ( !here->HSM1_icVDS_Given ) here->HSM1_icVDS = 0.0;
      if ( !here->HSM1_icVGS_Given ) here->HSM1_icVGS = 0.0;
      
      /* added by K.M. */
      here->HSM1_weff = here->HSM1_w - 2.0e0 * model->HSM1_dw;
      here->HSM1_leff = here->HSM1_l - 2.0e0 * model->HSM1_dl;

      /* process source/drain series resistance added by K.M. */
      /* Drain and source conductances are always zero,
	 because there is no sheet resistance in HSM1 model param.
      if ( model->HSM1_corsrd ) 
	here->HSM1drainConductance = 0.0;
      else 
	here->HSM1drainConductance = model->HSM1_rs / here->HSM1_weff;
      
      if ( here->HSM1drainConductance > 0.0 )
	here->HSM1drainConductance = 1.0 / here->HSM1drainConductance;
      else
	here->HSM1drainConductance = 0.0;
      
      if ( model->HSM1_corsrd ) 
	here->HSM1sourceConductance = 0.0;
      else 
	here->HSM1sourceConductance = model->HSM1_rd / here->HSM1_weff;

      if ( here->HSM1sourceConductance > 0.0 ) 
	here->HSM1sourceConductance = 1.0 / here->HSM1sourceConductance;
      else
	here->HSM1sourceConductance = 0.0;
      */
      here->HSM1drainConductance = 0.0;
      here->HSM1sourceConductance = 0.0;
      
      /* process drain series resistance */
      if( here->HSM1drainConductance > 0.0 && here->HSM1dNodePrime == 0 ) {
	error = CKTmkVolt(ckt, &tmp, here->HSM1name, "drain");
	if (error) return(error);
	here->HSM1dNodePrime = tmp->number;
	
	if (ckt->CKTcopyNodesets) {
                  if (CKTinst2Node(ckt,here,1,&tmpNode,&tmpName)==OK) {
                     if (tmpNode->nsGiven) {
                       tmp->nodeset=tmpNode->nodeset; 
                       tmp->nsGiven=tmpNode->nsGiven; 
                     }
                  }
                }

      } 
      else {
	here->HSM1dNodePrime = here->HSM1dNode;
      }
      
      /* process source series resistance */
      if( here->HSM1sourceConductance > 0.0 && here->HSM1sNodePrime == 0 ) {
	if ( here->HSM1sNodePrime == 0 ) {
	  error = CKTmkVolt(ckt, &tmp, here->HSM1name, "source");
	  if (error) return(error);
	  here->HSM1sNodePrime = tmp->number;
	  
	  if (ckt->CKTcopyNodesets) {
                  if (CKTinst2Node(ckt,here,3,&tmpNode,&tmpName)==OK) {
                     if (tmpNode->nsGiven) {
                       tmp->nodeset=tmpNode->nodeset; 
                       tmp->nsGiven=tmpNode->nsGiven; 
                     }
                  }
                }

	}
      } 
      else  {
	here->HSM1sNodePrime = here->HSM1sNode;
      }
                   
      /* set Sparse Matrix Pointers */
      
      /* macro to make elements with built in test for out of memory */
#define TSTALLOC(ptr,first,second) \
if((here->ptr = SMPmakeElt(matrix,here->first,here->second))==(double *)NULL){\
    return(E_NOMEM);\
}

      TSTALLOC(HSM1DdPtr, HSM1dNode, HSM1dNode)
      TSTALLOC(HSM1GgPtr, HSM1gNode, HSM1gNode)
      TSTALLOC(HSM1SsPtr, HSM1sNode, HSM1sNode)
      TSTALLOC(HSM1BbPtr, HSM1bNode, HSM1bNode)
      TSTALLOC(HSM1DPdpPtr, HSM1dNodePrime, HSM1dNodePrime)
      TSTALLOC(HSM1SPspPtr, HSM1sNodePrime, HSM1sNodePrime)
      TSTALLOC(HSM1DdpPtr, HSM1dNode, HSM1dNodePrime)
      TSTALLOC(HSM1GbPtr, HSM1gNode, HSM1bNode)
      TSTALLOC(HSM1GdpPtr, HSM1gNode, HSM1dNodePrime)
      TSTALLOC(HSM1GspPtr, HSM1gNode, HSM1sNodePrime)
      TSTALLOC(HSM1SspPtr, HSM1sNode, HSM1sNodePrime)
      TSTALLOC(HSM1BdpPtr, HSM1bNode, HSM1dNodePrime)
      TSTALLOC(HSM1BspPtr, HSM1bNode, HSM1sNodePrime)
      TSTALLOC(HSM1DPspPtr, HSM1dNodePrime, HSM1sNodePrime)
      TSTALLOC(HSM1DPdPtr, HSM1dNodePrime, HSM1dNode)
      TSTALLOC(HSM1BgPtr, HSM1bNode, HSM1gNode)
      TSTALLOC(HSM1DPgPtr, HSM1dNodePrime, HSM1gNode)
      TSTALLOC(HSM1SPgPtr, HSM1sNodePrime, HSM1gNode)
      TSTALLOC(HSM1SPsPtr, HSM1sNodePrime, HSM1sNode)
      TSTALLOC(HSM1DPbPtr, HSM1dNodePrime, HSM1bNode)
      TSTALLOC(HSM1SPbPtr, HSM1sNodePrime, HSM1bNode)
      TSTALLOC(HSM1SPdpPtr, HSM1sNodePrime, HSM1dNodePrime)

    }
  }
  return(OK);
} 


int
HSM1unsetup (GENmodel *inModel, CKTcircuit *ckt)
{
	HSM1model *model;
	HSM1instance *here;

	for (model = (HSM1model *) inModel; model != NULL;
	     model = model->HSM1nextModel)
	{
		for (here = model->HSM1instances; here != NULL;
		     here = here->HSM1nextInstance)
		{
			if (here->HSM1dNodePrime
			    && here->HSM1dNodePrime != here->HSM1dNode)
			{
				CKTdltNNum (ckt, here->HSM1dNodePrime);
				here->HSM1dNodePrime = 0;
			}
			if (here->HSM1sNodePrime
			    && here->HSM1sNodePrime != here->HSM1sNode)
			{
				CKTdltNNum (ckt, here->HSM1sNodePrime);
				here->HSM1sNodePrime = 0;
			}
		}
	}
	return OK;
}
