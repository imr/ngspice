/***********************************************************************

 HiSIM (Hiroshima University STARC IGFET Model)
 Copyright (C) 2012 Hiroshima University & STARC

 MODEL NAME : HiSIM_HV 
 ( VERSION : 1  SUBVERSION : 2  REVISION : 4 )
 Model Parameter VERSION : 1.23
 FILE : hsmhvset.c

 DATE : 2013.04.30

 released by 
                Hiroshima University &
                Semiconductor Technology Academic Research Center (STARC)
***********************************************************************/

#include "ngspice/ngspice.h"
#include "ngspice/smpdefs.h"
#include "ngspice/cktdefs.h"
#include "hsmhvdef.h"
#include "hsmhvevalenv.h"
#include "ngspice/const.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

#define C_m2cm    (1.0e2)
#define C_cm2m_p3 (1.0e-6)
#define C_m2cm_p1o2 (1.0e1)

#define BINNING(param) pParam->HSMHV_##param = model->HSMHV_##param \
  + model->HSMHV_l##param / Lbin + model->HSMHV_w##param / Wbin \
  + model->HSMHV_p##param / LWbin ;

#define RANGECHECK(param, min, max, pname)                              \
  if ( (param) < (min) || (param) > (max) ) {             \
    printf("warning(HiSIMHV): The model/instance parameter %s (= %e) must be in the range [%e , %e].\n", \
           (pname), (param), (double) (min), (double) (max) );                     \
  }
#define MINCHECK(param, min, pname)                              \
  if ( (param) < (min) ) {             \
    printf("warning(HiSIMHV): The model/instance parameter %s (= %e) must be greater than %e.\n", \
           (pname), (param), (min) );                     \
  }

int HSMHVsetup(
     register SMPmatrix *matrix,
     register GENmodel *inModel,
     register CKTcircuit *ckt,
     int *states)
     /* load the HSMHV device structure with those pointers needed later 
      * for fast matrix loading 
      */
{
  register HSMHVmodel *model = (HSMHVmodel*)inModel;
  register HSMHVinstance *here;
  int error=0 ;
  CKTnode *tmp;
  double T2, Rd, Rs ;
  HSMHVbinningParam *pParam ;
  HSMHVmodelMKSParam *modelMKS ;
  HSMHVhereMKSParam  *hereMKS ;
  double LG=0.0, WG =0.0, Lgate =0.0, Wgate =0.0 ;
  double Lbin=0.0, Wbin=0.0, LWbin =0.0; /* binning */
  
  /*  loop through all the HSMHV device models */
  for ( ;model != NULL ;model = HSMHVnextModel(model)) {
    /* Default value Processing for HVMOS Models */
    if ( !model->HSMHV_type_Given )
      model->HSMHV_type = NMOS ;

    if ( !model->HSMHV_info_Given ) model->HSMHV_info = 0 ;

    model->HSMHV_noise = 1;

    if ( !model->HSMHV_version_Given) {
        model->HSMHV_version = "1.24" ;
       printf("          1.24 is selected for VERSION. (default) \n");
    } else {
      if (strncmp(model->HSMHV_version,"1.24", 4) != 0 ) {
       model->HSMHV_version = "1.24" ;
       printf("          1.24 is selected for VERSION. (default) \n");
      } else {
       printf("           %s is selected for VERSION \n", model->HSMHV_version);
      }
    }

    if ( !model->HSMHV_corsrd_Given     ) model->HSMHV_corsrd     = 3 ;
    if ( !model->HSMHV_corg_Given       ) model->HSMHV_corg       = 0 ;
    if ( !model->HSMHV_coiprv_Given     ) model->HSMHV_coiprv     = 1 ;
    if ( !model->HSMHV_copprv_Given     ) model->HSMHV_copprv     = 1 ;
    if ( !model->HSMHV_coadov_Given     ) model->HSMHV_coadov     = 1 ;
    if ( !model->HSMHV_coisub_Given     ) model->HSMHV_coisub     = 0 ;
    if ( !model->HSMHV_coiigs_Given     ) model->HSMHV_coiigs     = 0 ;
    if ( !model->HSMHV_cogidl_Given     ) model->HSMHV_cogidl     = 0 ;
    if ( !model->HSMHV_coovlp_Given     ) model->HSMHV_coovlp     = 1 ;
    if ( !model->HSMHV_coovlps_Given    ) model->HSMHV_coovlps    = 0 ;
    if ( !model->HSMHV_coflick_Given    ) model->HSMHV_coflick    = 0 ;
    if ( !model->HSMHV_coisti_Given     ) model->HSMHV_coisti     = 0 ;
    if ( !model->HSMHV_conqs_Given      ) model->HSMHV_conqs      = 0 ; /* QS (default) */
    if ( !model->HSMHV_cothrml_Given    ) model->HSMHV_cothrml    = 0 ;
    if ( !model->HSMHV_coign_Given      ) model->HSMHV_coign      = 0 ; /* induced gate noise */
    if ( !model->HSMHV_codfm_Given      ) model->HSMHV_codfm      = 0 ; /* DFM */
    if ( !model->HSMHV_coqovsm_Given    ) model->HSMHV_coqovsm    = 1 ; 
    if ( !model->HSMHV_corbnet_Given    ) model->HSMHV_corbnet    = 0 ; 
    else if ( model->HSMHV_corbnet != 0 && model->HSMHV_corbnet != 1 ) {
      model->HSMHV_corbnet = 0;
      printf("warning(HiSIMHV): CORBNET has been set to its default value: %d.\n", model->HSMHV_corbnet);
    }
    if ( !model->HSMHV_coselfheat_Given ) model->HSMHV_coselfheat = 0 ; /* Self-heating model */
    if ( !model->HSMHV_cosubnode_Given  ) model->HSMHV_cosubnode  = 0 ; 
    if ( !model->HSMHV_cosym_Given ) model->HSMHV_cosym = 0 ;           /* Symmetry model for HV */
    if ( !model->HSMHV_cotemp_Given ) model->HSMHV_cotemp = 0 ;
    if ( !model->HSMHV_coldrift_Given ) model->HSMHV_coldrift = 0 ;


    if ( !model->HSMHV_vmax_Given    ) model->HSMHV_vmax    = 1.0e7 ;
    if ( !model->HSMHV_vmaxt1_Given  ) model->HSMHV_vmaxt1  = 0.0 ;
    if ( !model->HSMHV_vmaxt2_Given  ) model->HSMHV_vmaxt2  = 0.0 ;
    if ( !model->HSMHV_bgtmp1_Given  ) model->HSMHV_bgtmp1  = 90.25e-6 ;
    if ( !model->HSMHV_bgtmp2_Given  ) model->HSMHV_bgtmp2  = 1.0e-7 ;
    if ( !model->HSMHV_eg0_Given     ) model->HSMHV_eg0     = 1.1785e0 ;
    if ( !model->HSMHV_tox_Given     ) model->HSMHV_tox     = 30e-9 ;
    if ( !model->HSMHV_xld_Given     ) model->HSMHV_xld     = 0.0 ;
    if ( !model->HSMHV_lovers_Given  ) model->HSMHV_lovers  = 30e-9 ;
    if (  model->HSMHV_lover_Given   ) model->HSMHV_lovers  = model->HSMHV_lover ;
    if ( !model->HSMHV_rdov11_Given  ) model->HSMHV_rdov11   = 0.0 ;
    if ( !model->HSMHV_rdov12_Given  ) model->HSMHV_rdov12   = 1.0 ;
    if ( !model->HSMHV_rdov13_Given  ) model->HSMHV_rdov13   = 1.0 ;
    if ( !model->HSMHV_rdslp1_Given  ) model->HSMHV_rdslp1   = 1.0 ;
    if ( !model->HSMHV_rdict1_Given  ) model->HSMHV_rdict1   = 1.0 ;
    if ( !model->HSMHV_rdslp2_Given  ) model->HSMHV_rdslp2   = 1.0 ;
    if ( !model->HSMHV_rdict2_Given  ) model->HSMHV_rdict2   = 0.0 ;
    if ( !model->HSMHV_loverld_Given ) model->HSMHV_loverld  = 1.0e-6 ;
    if ( !model->HSMHV_ldrift1_Given ) model->HSMHV_ldrift1  = 1.0e-6 ;
    if ( !model->HSMHV_ldrift2_Given ) model->HSMHV_ldrift2  = 1.0e-6 ;
    if ( !model->HSMHV_ldrift1s_Given ) model->HSMHV_ldrift1s  = 0.0 ;
    if ( !model->HSMHV_ldrift2s_Given ) model->HSMHV_ldrift2s  = 1.0e-6 ;
    if ( !model->HSMHV_subld1_Given  ) model->HSMHV_subld1  = 0.0 ;
    if ( !model->HSMHV_subld2_Given  ) model->HSMHV_subld2  = 0.0 ;
    if ( !model->HSMHV_ddltmax_Given ) model->HSMHV_ddltmax = 10.0 ;  /* Vdseff */
    if ( !model->HSMHV_ddltslp_Given ) model->HSMHV_ddltslp = 0.0 ;  /* Vdseff */
    if ( !model->HSMHV_ddltict_Given ) model->HSMHV_ddltict = 10.0 ; /* Vdseff */
    if ( !model->HSMHV_vfbover_Given ) model->HSMHV_vfbover = -0.5 ;
    if ( !model->HSMHV_nover_Given   ) model->HSMHV_nover   = 3.0e16 ;
    if ( !model->HSMHV_novers_Given  ) model->HSMHV_novers  = 0.0 ;
    if ( !model->HSMHV_xwd_Given     ) model->HSMHV_xwd     = 0.0 ;
    if ( !model->HSMHV_xwdc_Given    ) model->HSMHV_xwdc    = model->HSMHV_xwd ;

    if ( !model->HSMHV_xl_Given   ) model->HSMHV_xl   = 0.0 ;
    if ( !model->HSMHV_xw_Given   ) model->HSMHV_xw   = 0.0 ;
    if ( !model->HSMHV_saref_Given   ) model->HSMHV_saref   = 1e-6 ;
    if ( !model->HSMHV_sbref_Given   ) model->HSMHV_sbref   = 1e-6 ;
    if ( !model->HSMHV_ll_Given   ) model->HSMHV_ll   = 0.0 ;
    if ( !model->HSMHV_lld_Given  ) model->HSMHV_lld  = 0.0 ;
    if ( !model->HSMHV_lln_Given  ) model->HSMHV_lln  = 0.0 ;
    if ( !model->HSMHV_wl_Given   ) model->HSMHV_wl   = 0.0 ;
    if ( !model->HSMHV_wl1_Given  ) model->HSMHV_wl1  = 0.0 ;
    if ( !model->HSMHV_wl1p_Given ) model->HSMHV_wl1p = 1.0 ;
    if ( !model->HSMHV_wl2_Given  ) model->HSMHV_wl2  = 0.0 ;
    if ( !model->HSMHV_wl2p_Given ) model->HSMHV_wl2p = 1.0 ;
    if ( !model->HSMHV_wld_Given  ) model->HSMHV_wld  = 0.0 ;
    if ( !model->HSMHV_wln_Given  ) model->HSMHV_wln  = 0.0 ;

    if ( !model->HSMHV_rsh_Given  ) model->HSMHV_rsh  = 0.0 ;
    if ( !model->HSMHV_rshg_Given ) model->HSMHV_rshg = 0.0 ;

    if ( !model->HSMHV_xqy_Given    ) model->HSMHV_xqy   = 0.0 ;
    if ( !model->HSMHV_xqy1_Given   ) model->HSMHV_xqy1  = 0.0 ;
    if ( !model->HSMHV_xqy2_Given   ) model->HSMHV_xqy2  = 0.0 ;
    if ( !model->HSMHV_rs_Given     ) model->HSMHV_rs    = 0.0 ;
    if ( !model->HSMHV_rd_Given     ) model->HSMHV_rd    = 5.0e-3 ;
    if ( !model->HSMHV_vfbc_Given   ) model->HSMHV_vfbc  = -1.0 ;
    if ( !model->HSMHV_vbi_Given    ) model->HSMHV_vbi   = 1.1 ;
    if ( !model->HSMHV_nsubc_Given  ) model->HSMHV_nsubc = 5.0e17 ;
    if ( !model->HSMHV_parl2_Given  ) model->HSMHV_parl2 = 10.0e-9 ;
    if ( !model->HSMHV_lp_Given     ) model->HSMHV_lp    = 0.0 ;
    if ( !model->HSMHV_nsubp_Given  ) model->HSMHV_nsubp = 1.0e18 ;

    if ( !model->HSMHV_nsubp0_Given ) model->HSMHV_nsubp0 = 0.0 ;
    if ( !model->HSMHV_nsubwp_Given ) model->HSMHV_nsubwp = 1.0 ;

    if ( !model->HSMHV_scp1_Given  ) model->HSMHV_scp1 = 1.0 ;
    if ( !model->HSMHV_scp2_Given  ) model->HSMHV_scp2 = 0.0 ;
    if ( !model->HSMHV_scp3_Given  ) model->HSMHV_scp3 = 0.0 ;
    if ( !model->HSMHV_sc1_Given   ) model->HSMHV_sc1  = 1.0 ;
    if ( !model->HSMHV_sc2_Given   ) model->HSMHV_sc2  = 0.0 ;
    if ( !model->HSMHV_sc3_Given   ) model->HSMHV_sc3  = 0.0 ;
    if ( !model->HSMHV_sc4_Given   ) model->HSMHV_sc4  = 0.0 ;
    if ( !model->HSMHV_pgd1_Given  ) model->HSMHV_pgd1 = 0.0 ;
    if ( !model->HSMHV_pgd2_Given  ) model->HSMHV_pgd2 = 1.0 ;
    if ( !model->HSMHV_pgd3_Given  ) model->HSMHV_pgd3 = 0.8 ;
    if ( !model->HSMHV_pgd4_Given  ) model->HSMHV_pgd4 = 0.0 ;

    if ( !model->HSMHV_ndep_Given   ) model->HSMHV_ndep   = 1.0 ;
    if ( !model->HSMHV_ndepl_Given  ) model->HSMHV_ndepl  = 0.0 ;
    if ( !model->HSMHV_ndeplp_Given ) model->HSMHV_ndeplp = 1.0 ;
    if ( !model->HSMHV_ninv_Given   ) model->HSMHV_ninv   = 0.5 ;
    if ( !model->HSMHV_muecb0_Given ) model->HSMHV_muecb0 = 1.0e3 ;
    if ( !model->HSMHV_muecb1_Given ) model->HSMHV_muecb1 = 100.0 ;
    if ( !model->HSMHV_mueph0_Given ) model->HSMHV_mueph0 = 300.0e-3 ;
    if ( !model->HSMHV_mueph1_Given ) {
      if (model->HSMHV_type == NMOS) model->HSMHV_mueph1 = 25.0e3 ;
      else model->HSMHV_mueph1 = 9.0e3 ;
    }
    if ( !model->HSMHV_muephw_Given ) model->HSMHV_muephw = 0.0 ;
    if ( !model->HSMHV_muepwp_Given ) model->HSMHV_muepwp = 1.0 ;
    if ( !model->HSMHV_muephl_Given ) model->HSMHV_muephl = 0.0 ;
    if ( !model->HSMHV_mueplp_Given ) model->HSMHV_mueplp = 1.0 ;
    if ( !model->HSMHV_muephs_Given ) model->HSMHV_muephs = 0.0 ;
    if ( !model->HSMHV_muepsp_Given ) model->HSMHV_muepsp = 1.0 ;

    if ( !model->HSMHV_vtmp_Given  ) model->HSMHV_vtmp  = 0.0 ;

    if ( !model->HSMHV_wvth0_Given ) model->HSMHV_wvth0 = 0.0 ;

    if ( !model->HSMHV_muesr0_Given ) model->HSMHV_muesr0 = 2.0 ;
    if ( !model->HSMHV_muesr1_Given ) model->HSMHV_muesr1 = 1.0e16 ;
    if ( !model->HSMHV_muesrl_Given ) model->HSMHV_muesrl = 0.0 ;
    if ( !model->HSMHV_muesrw_Given ) model->HSMHV_muesrw = 0.0 ;
    if ( !model->HSMHV_mueswp_Given ) model->HSMHV_mueswp = 1.0 ;
    if ( !model->HSMHV_mueslp_Given ) model->HSMHV_mueslp = 1.0 ;

    if ( !model->HSMHV_muetmp_Given  ) model->HSMHV_muetmp = 1.5 ;

    if ( !model->HSMHV_bb_Given ) {
      if (model->HSMHV_type == NMOS) model->HSMHV_bb = 2.0 ;
      else model->HSMHV_bb = 1.0 ;
    }

    if ( !model->HSMHV_sub1_Given  ) model->HSMHV_sub1  = 10.0 ;
    if ( !model->HSMHV_sub2_Given  ) model->HSMHV_sub2  = 25.0 ;
    if ( !model->HSMHV_svgs_Given  ) model->HSMHV_svgs  = 0.8e0 ;
    if ( !model->HSMHV_svbs_Given  ) model->HSMHV_svbs  = 0.5e0 ;
    if ( !model->HSMHV_svbsl_Given ) model->HSMHV_svbsl = 0e0 ;
    if ( !model->HSMHV_svds_Given  ) model->HSMHV_svds  = 0.8e0 ;
    if ( !model->HSMHV_slg_Given   ) model->HSMHV_slg   = 30e-9 ;
    if ( !model->HSMHV_sub1l_Given ) model->HSMHV_sub1l = 2.5e-3 ;
    if ( !model->HSMHV_sub2l_Given ) model->HSMHV_sub2l = 2e-6 ;
    if ( !model->HSMHV_fn1_Given   ) model->HSMHV_fn1   = 50e0 ;
    if ( !model->HSMHV_fn2_Given   ) model->HSMHV_fn2   = 170e-6 ;
    if ( !model->HSMHV_fn3_Given   ) model->HSMHV_fn3   = 0e0 ;
    if ( !model->HSMHV_fvbs_Given  ) model->HSMHV_fvbs  = 12e-3 ;

    if ( !model->HSMHV_svgsl_Given  ) model->HSMHV_svgsl  = 0.0 ;
    if ( !model->HSMHV_svgslp_Given ) model->HSMHV_svgslp = 1.0 ;
    if ( !model->HSMHV_svgswp_Given ) model->HSMHV_svgswp = 1.0 ;
    if ( !model->HSMHV_svgsw_Given  ) model->HSMHV_svgsw  = 0.0 ;
    if ( !model->HSMHV_svbslp_Given ) model->HSMHV_svbslp = 1.0 ;
    if ( !model->HSMHV_slgl_Given   ) model->HSMHV_slgl   = 0.0 ;
    if ( !model->HSMHV_slglp_Given  ) model->HSMHV_slglp  = 1.0 ;
    if ( !model->HSMHV_sub1lp_Given ) model->HSMHV_sub1lp = 1.0 ; 

    if ( !model->HSMHV_nsti_Given      ) model->HSMHV_nsti      = 1.0e17 ;
    if ( !model->HSMHV_wsti_Given      ) model->HSMHV_wsti      = 0.0 ;
    if ( !model->HSMHV_wstil_Given     ) model->HSMHV_wstil     = 0.0 ;
    if ( !model->HSMHV_wstilp_Given    ) model->HSMHV_wstilp    = 1.0 ;
    if ( !model->HSMHV_wstiw_Given     ) model->HSMHV_wstiw     = 0.0 ;
    if ( !model->HSMHV_wstiwp_Given    ) model->HSMHV_wstiwp    = 1.0 ;
    if ( !model->HSMHV_scsti1_Given    ) model->HSMHV_scsti1    = 0.0 ;
    if ( !model->HSMHV_scsti2_Given    ) model->HSMHV_scsti2    = 0.0 ;
    if ( !model->HSMHV_vthsti_Given    ) model->HSMHV_vthsti    = 0.0 ;
    if ( !model->HSMHV_vdsti_Given     ) model->HSMHV_vdsti     = 0.0 ;
    if ( !model->HSMHV_muesti1_Given   ) model->HSMHV_muesti1   = 0.0 ;
    if ( !model->HSMHV_muesti2_Given   ) model->HSMHV_muesti2   = 0.0 ;
    if ( !model->HSMHV_muesti3_Given   ) model->HSMHV_muesti3   = 1.0 ;
    if ( !model->HSMHV_nsubpsti1_Given ) model->HSMHV_nsubpsti1 = 0.0 ;
    if ( !model->HSMHV_nsubpsti2_Given ) model->HSMHV_nsubpsti2 = 0.0 ;
    if ( !model->HSMHV_nsubpsti3_Given ) model->HSMHV_nsubpsti3 = 1.0 ;

    if ( !model->HSMHV_lpext_Given ) model->HSMHV_lpext = 1.0e-50 ;
    if ( !model->HSMHV_npext_Given ) model->HSMHV_npext = 1.0e17 ;
    if ( !model->HSMHV_scp21_Given ) model->HSMHV_scp21 = 0.0 ;
    if ( !model->HSMHV_scp22_Given ) model->HSMHV_scp22 = 0.0 ;
    if ( !model->HSMHV_bs1_Given   ) model->HSMHV_bs1   = 0.0 ;
    if ( !model->HSMHV_bs2_Given   ) model->HSMHV_bs2   = 0.9 ;

    if ( !model->HSMHV_tpoly_Given ) model->HSMHV_tpoly = 200e-9 ;
    if ( !model->HSMHV_cgbo_Given  ) model->HSMHV_cgbo  = 0.0 ;
    if ( !model->HSMHV_js0_Given   ) model->HSMHV_js0   = 0.5e-6 ;
    if ( !model->HSMHV_js0sw_Given ) model->HSMHV_js0sw = 0.0 ;
    if ( !model->HSMHV_nj_Given    ) model->HSMHV_nj    = 1.0 ;
    if ( !model->HSMHV_njsw_Given  ) model->HSMHV_njsw  = 1.0 ;
    if ( !model->HSMHV_xti_Given   ) model->HSMHV_xti   = 2.0 ;
    if ( !model->HSMHV_cj_Given    ) model->HSMHV_cj    = 5.0e-04 ;
    if ( !model->HSMHV_cjsw_Given  ) model->HSMHV_cjsw  = 5.0e-10 ;
    if ( !model->HSMHV_cjswg_Given ) model->HSMHV_cjswg = 5.0e-10 ;
    if ( !model->HSMHV_mj_Given    ) model->HSMHV_mj    = 0.5e0 ;
    if ( !model->HSMHV_mjsw_Given  ) model->HSMHV_mjsw  = 0.33e0 ;
    if ( !model->HSMHV_mjswg_Given ) model->HSMHV_mjswg = 0.33e0 ;
    if ( !model->HSMHV_pb_Given    ) model->HSMHV_pb    = 1.0e0 ;
    if ( !model->HSMHV_pbsw_Given  ) model->HSMHV_pbsw  = 1.0e0 ;
    if ( !model->HSMHV_pbswg_Given ) model->HSMHV_pbswg = 1.0e0 ;
    if ( !model->HSMHV_xti2_Given  ) model->HSMHV_xti2  = 0.0e0 ;
    if ( !model->HSMHV_cisb_Given  ) model->HSMHV_cisb  = 0.0e0 ;
    if ( !model->HSMHV_cvb_Given   ) model->HSMHV_cvb   = 0.0e0 ;
    if ( !model->HSMHV_ctemp_Given ) model->HSMHV_ctemp = 0.0e0 ;
    if ( !model->HSMHV_cisbk_Given ) model->HSMHV_cisbk = 0.0e0 ;
    if ( !model->HSMHV_cvbk_Given  ) model->HSMHV_cvbk  = 0.0e0 ;
    if ( !model->HSMHV_divx_Given  ) model->HSMHV_divx  = 0.0e0 ;

    if ( !model->HSMHV_clm1_Given  ) model->HSMHV_clm1 = 0.7 ;
    if ( !model->HSMHV_clm2_Given  ) model->HSMHV_clm2 = 2.0 ;
    if ( !model->HSMHV_clm3_Given  ) model->HSMHV_clm3 = 1.0 ;
    if ( !model->HSMHV_clm5_Given   ) model->HSMHV_clm5   = 1.0 ;
    if ( !model->HSMHV_clm6_Given   ) model->HSMHV_clm6   = 0.0 ;
    if ( !model->HSMHV_vover_Given  ) model->HSMHV_vover  = 0.3 ;
    if ( !model->HSMHV_voverp_Given ) model->HSMHV_voverp = 0.3 ;
    if ( !model->HSMHV_wfc_Given    ) model->HSMHV_wfc    = 0.0 ;
    if ( !model->HSMHV_nsubcw_Given    ) model->HSMHV_nsubcw  = 0.0 ;
    if ( !model->HSMHV_nsubcwp_Given   ) model->HSMHV_nsubcwp = 1.0 ;
    if ( !model->HSMHV_qme1_Given   ) model->HSMHV_qme1   = 0.0 ;
    if ( !model->HSMHV_qme2_Given   ) model->HSMHV_qme2   = 0.0 ;
    if ( !model->HSMHV_qme3_Given   ) model->HSMHV_qme3   = 0.0 ;

    if ( !model->HSMHV_vovers_Given  ) model->HSMHV_vovers  = 0.0 ;
    if ( !model->HSMHV_voversp_Given ) model->HSMHV_voversp = 0.0 ;

    if ( !model->HSMHV_gidl1_Given ) model->HSMHV_gidl1 = 2e0 ;
    if ( !model->HSMHV_gidl2_Given ) model->HSMHV_gidl2 = 3e7 ;
    if ( !model->HSMHV_gidl3_Given ) model->HSMHV_gidl3 = 0.9e0 ;
    if ( !model->HSMHV_gidl4_Given ) model->HSMHV_gidl4 = 0.0 ;
    if ( !model->HSMHV_gidl5_Given ) model->HSMHV_gidl5 = 0.2e0 ;

    if ( !model->HSMHV_gleak1_Given ) model->HSMHV_gleak1 = 50e0 ;
    if ( !model->HSMHV_gleak2_Given ) model->HSMHV_gleak2 = 10e6 ;
    if ( !model->HSMHV_gleak3_Given ) model->HSMHV_gleak3 = 60e-3 ;
    if ( !model->HSMHV_gleak4_Given ) model->HSMHV_gleak4 = 4e0 ;
    if ( !model->HSMHV_gleak5_Given ) model->HSMHV_gleak5 = 7.5e3 ;
    if ( !model->HSMHV_gleak6_Given ) model->HSMHV_gleak6 = 250e-3 ;
    if ( !model->HSMHV_gleak7_Given ) model->HSMHV_gleak7 = 1e-6 ;

    if ( !model->HSMHV_glpart1_Given ) model->HSMHV_glpart1 = 0.5 ;
    if ( !model->HSMHV_glksd1_Given  ) model->HSMHV_glksd1  = 1.0e-15 ;
    if ( !model->HSMHV_glksd2_Given  ) model->HSMHV_glksd2  = 5e6 ;
    if ( !model->HSMHV_glksd3_Given  ) model->HSMHV_glksd3  = -5e6 ;
    if ( !model->HSMHV_glkb1_Given   ) model->HSMHV_glkb1   = 5e-16 ;
    if ( !model->HSMHV_glkb2_Given   ) model->HSMHV_glkb2   = 1e0 ;
    if ( !model->HSMHV_glkb3_Given   ) model->HSMHV_glkb3   = 0e0 ;
    if ( !model->HSMHV_egig_Given    ) model->HSMHV_egig    = 0e0 ;
    if ( !model->HSMHV_igtemp2_Given ) model->HSMHV_igtemp2 = 0e0 ;
    if ( !model->HSMHV_igtemp3_Given ) model->HSMHV_igtemp3 = 0e0 ;
    if ( !model->HSMHV_vzadd0_Given  ) model->HSMHV_vzadd0  = 10.0e-3 ;
    if ( !model->HSMHV_pzadd0_Given  ) model->HSMHV_pzadd0  = 5.0e-3 ;
    if ( !model->HSMHV_nftrp_Given   ) model->HSMHV_nftrp   = 10e9 ;
    if ( !model->HSMHV_nfalp_Given   ) model->HSMHV_nfalp   = 1.0e-19 ;
    if ( !model->HSMHV_cit_Given     ) model->HSMHV_cit     = 0e0 ;
    if ( !model->HSMHV_falph_Given   ) model->HSMHV_falph   = 1.0 ;

    if ( !model->HSMHV_kappa_Given ) model->HSMHV_kappa = 3.90e0 ;
    if ( !model->HSMHV_cgso_Given  ) model->HSMHV_cgso  = 0.0 ;
    if ( !model->HSMHV_cgdo_Given  ) model->HSMHV_cgdo  = 0.0 ;

    if ( !model->HSMHV_pthrou_Given ) model->HSMHV_pthrou = 0.0 ;

    if ( !model->HSMHV_vdiffj_Given ) model->HSMHV_vdiffj = 0.6e-3 ;
    if ( !model->HSMHV_dly1_Given   ) model->HSMHV_dly1   = 100.0e-12 ;
    if ( !model->HSMHV_dly2_Given   ) model->HSMHV_dly2   = 0.7e0 ;
    if ( !model->HSMHV_dly3_Given   ) model->HSMHV_dly3   = 0.8e-6 ;
    if ( !model->HSMHV_tnom_Given   ) model->HSMHV_tnom   = 27.0 ; /* [C] */

    if ( !model->HSMHV_ovslp_Given  ) model->HSMHV_ovslp = 2.1e-7 ;
    if ( !model->HSMHV_ovmag_Given  ) model->HSMHV_ovmag = 0.6 ;

    if ( !model->HSMHV_gbmin_Given  ) model->HSMHV_gbmin = 1.0e-12; /* in mho */
    if ( !model->HSMHV_rbpb_Given   ) model->HSMHV_rbpb  = 50.0e0 ;
    if ( !model->HSMHV_rbpd_Given   ) model->HSMHV_rbpd  = 50.0e0 ;
    if ( !model->HSMHV_rbps_Given   ) model->HSMHV_rbps  = 50.0e0 ;
    if ( !model->HSMHV_rbdb_Given   ) model->HSMHV_rbdb  = 50.0e0 ;
    if ( !model->HSMHV_rbsb_Given   ) model->HSMHV_rbsb  = 50.0e0 ;

    if ( !model->HSMHV_ibpc1_Given  ) model->HSMHV_ibpc1 = 0.0 ;
    if ( !model->HSMHV_ibpc2_Given  ) model->HSMHV_ibpc2 = 0.0 ;

    if ( !model->HSMHV_mphdfm_Given ) model->HSMHV_mphdfm = -0.3 ;

    if ( !model->HSMHV_rdvg11_Given ) model->HSMHV_rdvg11  = 0.0 ;
    if ( !model->HSMHV_rdvg12_Given ) model->HSMHV_rdvg12  = 100.0 ;
    if ( !model->HSMHV_rth0_Given   ) model->HSMHV_rth0    = 0.1 ;    /* Self-heating model */
    if ( !model->HSMHV_cth0_Given   ) model->HSMHV_cth0    = 1.0e-7 ; /* Self-heating model */
    if ( !model->HSMHV_powrat_Given ) model->HSMHV_powrat  = 1.0 ;    /* Self-heating model */

    if ( !model->HSMHV_tcjbd_Given    ) model->HSMHV_tcjbd    = 0.0 ; /* Self-heating model */    
    if ( !model->HSMHV_tcjbs_Given    ) model->HSMHV_tcjbs    = 0.0 ; /* Self-heating model */    
    if ( !model->HSMHV_tcjbdsw_Given  ) model->HSMHV_tcjbdsw  = 0.0 ; /* Self-heating model */    
    if ( !model->HSMHV_tcjbssw_Given  ) model->HSMHV_tcjbssw  = 0.0 ; /* Self-heating model */    
    if ( !model->HSMHV_tcjbdswg_Given ) model->HSMHV_tcjbdswg = 0.0 ; /* Self-heating model */    
    if ( !model->HSMHV_tcjbsswg_Given ) model->HSMHV_tcjbsswg = 0.0 ; /* Self-heating model */

                                      /* value reset to switch off NQS for QbdLD:           */
                                        model->HSMHV_dlyov    = 0.0 ; /* 1.0e3 ;            */
    if ( !model->HSMHV_qdftvd_Given   ) model->HSMHV_qdftvd   = 1.0 ;
    if ( !model->HSMHV_xldld_Given    ) model->HSMHV_xldld    = 1.0e-6 ;
    if ( !model->HSMHV_xwdld_Given    ) model->HSMHV_xwdld    = model->HSMHV_xwd ;
    if ( !model->HSMHV_rdvd_Given     ) model->HSMHV_rdvd     = 7.0e-2 ;
    if ( !model->HSMHV_qovsm_Given    ) model->HSMHV_qovsm    = 0.2 ;

    if ( !model->HSMHV_rd20_Given    ) model->HSMHV_rd20    = 0.0 ;
    if ( !model->HSMHV_rd21_Given    ) model->HSMHV_rd21    = 1.0 ;
    if ( !model->HSMHV_rd22_Given    ) model->HSMHV_rd22    = 0.0 ;
    if ( !model->HSMHV_rd22d_Given   ) model->HSMHV_rd22d   = 0.0 ;
    if ( !model->HSMHV_rd23_Given    ) model->HSMHV_rd23    = 5e-3 ;
    if ( !model->HSMHV_rd24_Given    ) model->HSMHV_rd24    = 0.0 ;
    if ( !model->HSMHV_rd25_Given    ) model->HSMHV_rd25    = 0.0 ;

    if ( !model->HSMHV_rdvdl_Given    ) model->HSMHV_rdvdl     = 0.0 ;
    if ( !model->HSMHV_rdvdlp_Given   ) model->HSMHV_rdvdlp    = 1.0 ;
    if ( !model->HSMHV_rdvds_Given    ) model->HSMHV_rdvds     = 0.0 ;
    if ( !model->HSMHV_rdvdsp_Given   ) model->HSMHV_rdvdsp    = 1.0 ;
    if ( !model->HSMHV_rd23l_Given    ) model->HSMHV_rd23l     = 0.0 ;
    if ( !model->HSMHV_rd23lp_Given   ) model->HSMHV_rd23lp    = 1.0 ;
    if ( !model->HSMHV_rd23s_Given    ) model->HSMHV_rd23s     = 0.0 ;
    if ( !model->HSMHV_rd23sp_Given   ) model->HSMHV_rd23sp    = 1.0 ;
    if ( !model->HSMHV_rds_Given      ) model->HSMHV_rds       = 0.0 ;
    if ( !model->HSMHV_rdsp_Given     ) model->HSMHV_rdsp      = 1.0 ;
    if ( !model->HSMHV_rdtemp1_Given  ) model->HSMHV_rdtemp1   = 0.0 ;
    if ( !model->HSMHV_rdtemp2_Given  ) model->HSMHV_rdtemp2   = 0.0 ;
    model->HSMHV_rth0r     = 0.0 ; /* not used in this version */
    if ( !model->HSMHV_rdvdtemp1_Given) model->HSMHV_rdvdtemp1 = 0.0 ;
    if ( !model->HSMHV_rdvdtemp2_Given) model->HSMHV_rdvdtemp2 = 0.0 ;
    if ( !model->HSMHV_rth0w_Given    ) model->HSMHV_rth0w     = 0.0 ;
    if ( !model->HSMHV_rth0wp_Given   ) model->HSMHV_rth0wp    = 1.0 ;

    if ( !model->HSMHV_cvdsover_Given ) model->HSMHV_cvdsover  = 0.0 ;

    if ( !model->HSMHV_ninvd_Given    ) model->HSMHV_ninvd     = 0.0 ;
    if ( !model->HSMHV_ninvdw_Given   ) model->HSMHV_ninvdw    = 0.0 ;
    if ( !model->HSMHV_ninvdwp_Given  ) model->HSMHV_ninvdwp   = 1.0 ;
    if ( !model->HSMHV_ninvdt1_Given  ) model->HSMHV_ninvdt1   = 0.0 ;
    if ( !model->HSMHV_ninvdt2_Given  ) model->HSMHV_ninvdt2   = 0.0 ;
    if ( !model->HSMHV_vbsmin_Given   ) model->HSMHV_vbsmin    = -10.5 ;
    if ( !model->HSMHV_rdvb_Given     ) model->HSMHV_rdvb      = 0.0 ;
    if ( !model->HSMHV_rth0nf_Given   ) model->HSMHV_rth0nf    = 0.0 ;

    if ( !model->HSMHV_rthtemp1_Given   ) model->HSMHV_rthtemp1    = 0.0 ;
    if ( !model->HSMHV_rthtemp2_Given   ) model->HSMHV_rthtemp2    = 0.0 ;
    if ( !model->HSMHV_prattemp1_Given  ) model->HSMHV_prattemp1   = 0.0 ;
    if ( !model->HSMHV_prattemp2_Given  ) model->HSMHV_prattemp2   = 0.0 ;

    if ( !model->HSMHV_rdvsub_Given   ) model->HSMHV_rdvsub   = 1.0 ;    /* [-] substrate effect */
    if ( !model->HSMHV_rdvdsub_Given  ) model->HSMHV_rdvdsub  = 0.3 ;    /* [-] substrate effect */
    if ( !model->HSMHV_ddrift_Given   ) model->HSMHV_ddrift   = 1.0e-6 ; /* [m] substrate effect */
    if ( !model->HSMHV_vbisub_Given   ) model->HSMHV_vbisub   = 0.7 ;    /* [V] substrate effect */
    if ( !model->HSMHV_nsubsub_Given  ) model->HSMHV_nsubsub  = 1.0e15 ; /* [cm^-3] substrate effect */
    if ( !model->HSMHV_shemax_Given    ) model->HSMHV_shemax    = 500 ;

    /* binning parameters */
    if ( !model->HSMHV_lmin_Given ) model->HSMHV_lmin = 0.0 ;
    if ( !model->HSMHV_lmax_Given ) model->HSMHV_lmax = 1.0 ;
    if ( !model->HSMHV_wmin_Given ) model->HSMHV_wmin = 0.0 ;
    if ( !model->HSMHV_wmax_Given ) model->HSMHV_wmax = 1.0 ;
    if ( !model->HSMHV_lbinn_Given ) model->HSMHV_lbinn = 1.0 ;
    if ( !model->HSMHV_wbinn_Given ) model->HSMHV_wbinn = 1.0 ;

    /* Length dependence */
    if ( !model->HSMHV_lvmax_Given ) model->HSMHV_lvmax = 0.0 ;
    if ( !model->HSMHV_lbgtmp1_Given ) model->HSMHV_lbgtmp1 = 0.0 ;
    if ( !model->HSMHV_lbgtmp2_Given ) model->HSMHV_lbgtmp2 = 0.0 ;
    if ( !model->HSMHV_leg0_Given ) model->HSMHV_leg0 = 0.0 ;
    if ( !model->HSMHV_lvfbover_Given ) model->HSMHV_lvfbover = 0.0 ;
    if ( !model->HSMHV_lnover_Given ) model->HSMHV_lnover = 0.0 ;
    if ( !model->HSMHV_lnovers_Given ) model->HSMHV_lnovers = 0.0 ;
    if ( !model->HSMHV_lwl2_Given ) model->HSMHV_lwl2 = 0.0 ;
    if ( !model->HSMHV_lvfbc_Given ) model->HSMHV_lvfbc = 0.0 ;
    if ( !model->HSMHV_lnsubc_Given ) model->HSMHV_lnsubc = 0.0 ;
    if ( !model->HSMHV_lnsubp_Given ) model->HSMHV_lnsubp = 0.0 ;
    if ( !model->HSMHV_lscp1_Given ) model->HSMHV_lscp1 = 0.0 ;
    if ( !model->HSMHV_lscp2_Given ) model->HSMHV_lscp2 = 0.0 ;
    if ( !model->HSMHV_lscp3_Given ) model->HSMHV_lscp3 = 0.0 ;
    if ( !model->HSMHV_lsc1_Given ) model->HSMHV_lsc1 = 0.0 ;
    if ( !model->HSMHV_lsc2_Given ) model->HSMHV_lsc2 = 0.0 ;
    if ( !model->HSMHV_lsc3_Given ) model->HSMHV_lsc3 = 0.0 ;
    if ( !model->HSMHV_lpgd1_Given ) model->HSMHV_lpgd1 = 0.0 ;
    if ( !model->HSMHV_lpgd3_Given ) model->HSMHV_lpgd3 = 0.0 ;
    if ( !model->HSMHV_lndep_Given ) model->HSMHV_lndep = 0.0 ;
    if ( !model->HSMHV_lninv_Given ) model->HSMHV_lninv = 0.0 ;
    if ( !model->HSMHV_lmuecb0_Given ) model->HSMHV_lmuecb0 = 0.0 ;
    if ( !model->HSMHV_lmuecb1_Given ) model->HSMHV_lmuecb1 = 0.0 ;
    if ( !model->HSMHV_lmueph1_Given ) model->HSMHV_lmueph1 = 0.0 ;
    if ( !model->HSMHV_lvtmp_Given ) model->HSMHV_lvtmp = 0.0 ;
    if ( !model->HSMHV_lwvth0_Given ) model->HSMHV_lwvth0 = 0.0 ;
    if ( !model->HSMHV_lmuesr1_Given ) model->HSMHV_lmuesr1 = 0.0 ;
    if ( !model->HSMHV_lmuetmp_Given ) model->HSMHV_lmuetmp = 0.0 ;
    if ( !model->HSMHV_lsub1_Given ) model->HSMHV_lsub1 = 0.0 ;
    if ( !model->HSMHV_lsub2_Given ) model->HSMHV_lsub2 = 0.0 ;
    if ( !model->HSMHV_lsvds_Given ) model->HSMHV_lsvds = 0.0 ;
    if ( !model->HSMHV_lsvbs_Given ) model->HSMHV_lsvbs = 0.0 ;
    if ( !model->HSMHV_lsvgs_Given ) model->HSMHV_lsvgs = 0.0 ;
    if ( !model->HSMHV_lfn1_Given ) model->HSMHV_lfn1 = 0.0 ;
    if ( !model->HSMHV_lfn2_Given ) model->HSMHV_lfn2 = 0.0 ;
    if ( !model->HSMHV_lfn3_Given ) model->HSMHV_lfn3 = 0.0 ;
    if ( !model->HSMHV_lfvbs_Given ) model->HSMHV_lfvbs = 0.0 ;
    if ( !model->HSMHV_lnsti_Given ) model->HSMHV_lnsti = 0.0 ;
    if ( !model->HSMHV_lwsti_Given ) model->HSMHV_lwsti = 0.0 ;
    if ( !model->HSMHV_lscsti1_Given ) model->HSMHV_lscsti1 = 0.0 ;
    if ( !model->HSMHV_lscsti2_Given ) model->HSMHV_lscsti2 = 0.0 ;
    if ( !model->HSMHV_lvthsti_Given ) model->HSMHV_lvthsti = 0.0 ;
    if ( !model->HSMHV_lmuesti1_Given ) model->HSMHV_lmuesti1 = 0.0 ;
    if ( !model->HSMHV_lmuesti2_Given ) model->HSMHV_lmuesti2 = 0.0 ;
    if ( !model->HSMHV_lmuesti3_Given ) model->HSMHV_lmuesti3 = 0.0 ;
    if ( !model->HSMHV_lnsubpsti1_Given ) model->HSMHV_lnsubpsti1 = 0.0 ;
    if ( !model->HSMHV_lnsubpsti2_Given ) model->HSMHV_lnsubpsti2 = 0.0 ;
    if ( !model->HSMHV_lnsubpsti3_Given ) model->HSMHV_lnsubpsti3 = 0.0 ;
    if ( !model->HSMHV_lcgso_Given ) model->HSMHV_lcgso = 0.0 ;
    if ( !model->HSMHV_lcgdo_Given ) model->HSMHV_lcgdo = 0.0 ;
    if ( !model->HSMHV_ljs0_Given ) model->HSMHV_ljs0 = 0.0 ;
    if ( !model->HSMHV_ljs0sw_Given ) model->HSMHV_ljs0sw = 0.0 ;
    if ( !model->HSMHV_lnj_Given ) model->HSMHV_lnj = 0.0 ;
    if ( !model->HSMHV_lcisbk_Given ) model->HSMHV_lcisbk = 0.0 ;
    if ( !model->HSMHV_lclm1_Given ) model->HSMHV_lclm1 = 0.0 ;
    if ( !model->HSMHV_lclm2_Given ) model->HSMHV_lclm2 = 0.0 ;
    if ( !model->HSMHV_lclm3_Given ) model->HSMHV_lclm3 = 0.0 ;
    if ( !model->HSMHV_lwfc_Given ) model->HSMHV_lwfc = 0.0 ;
    if ( !model->HSMHV_lgidl1_Given ) model->HSMHV_lgidl1 = 0.0 ;
    if ( !model->HSMHV_lgidl2_Given ) model->HSMHV_lgidl2 = 0.0 ;
    if ( !model->HSMHV_lgleak1_Given ) model->HSMHV_lgleak1 = 0.0 ;
    if ( !model->HSMHV_lgleak2_Given ) model->HSMHV_lgleak2 = 0.0 ;
    if ( !model->HSMHV_lgleak3_Given ) model->HSMHV_lgleak3 = 0.0 ;
    if ( !model->HSMHV_lgleak6_Given ) model->HSMHV_lgleak6 = 0.0 ;
    if ( !model->HSMHV_lglksd1_Given ) model->HSMHV_lglksd1 = 0.0 ;
    if ( !model->HSMHV_lglksd2_Given ) model->HSMHV_lglksd2 = 0.0 ;
    if ( !model->HSMHV_lglkb1_Given ) model->HSMHV_lglkb1 = 0.0 ;
    if ( !model->HSMHV_lglkb2_Given ) model->HSMHV_lglkb2 = 0.0 ;
    if ( !model->HSMHV_lnftrp_Given ) model->HSMHV_lnftrp = 0.0 ;
    if ( !model->HSMHV_lnfalp_Given ) model->HSMHV_lnfalp = 0.0 ;
    if ( !model->HSMHV_lpthrou_Given ) model->HSMHV_lpthrou = 0.0 ;
    if ( !model->HSMHV_lvdiffj_Given ) model->HSMHV_lvdiffj = 0.0 ;
    if ( !model->HSMHV_libpc1_Given ) model->HSMHV_libpc1 = 0.0 ;
    if ( !model->HSMHV_libpc2_Given ) model->HSMHV_libpc2 = 0.0 ;
    if ( !model->HSMHV_lcgbo_Given ) model->HSMHV_lcgbo = 0.0 ;
    if ( !model->HSMHV_lcvdsover_Given ) model->HSMHV_lcvdsover = 0.0 ;
    if ( !model->HSMHV_lfalph_Given ) model->HSMHV_lfalph = 0.0 ;
    if ( !model->HSMHV_lnpext_Given ) model->HSMHV_lnpext = 0.0 ;
    if ( !model->HSMHV_lpowrat_Given ) model->HSMHV_lpowrat = 0.0 ;
    if ( !model->HSMHV_lrd_Given ) model->HSMHV_lrd = 0.0 ;
    if ( !model->HSMHV_lrd22_Given ) model->HSMHV_lrd22 = 0.0 ;
    if ( !model->HSMHV_lrd23_Given ) model->HSMHV_lrd23 = 0.0 ;
    if ( !model->HSMHV_lrd24_Given ) model->HSMHV_lrd24 = 0.0 ;
    if ( !model->HSMHV_lrdict1_Given ) model->HSMHV_lrdict1 = 0.0 ;
    if ( !model->HSMHV_lrdov13_Given ) model->HSMHV_lrdov13 = 0.0 ;
    if ( !model->HSMHV_lrdslp1_Given ) model->HSMHV_lrdslp1 = 0.0 ;
    if ( !model->HSMHV_lrdvb_Given ) model->HSMHV_lrdvb = 0.0 ;
    if ( !model->HSMHV_lrdvd_Given ) model->HSMHV_lrdvd = 0.0 ;
    if ( !model->HSMHV_lrdvg11_Given ) model->HSMHV_lrdvg11 = 0.0 ;
    if ( !model->HSMHV_lrs_Given ) model->HSMHV_lrs = 0.0 ;
    if ( !model->HSMHV_lrth0_Given ) model->HSMHV_lrth0 = 0.0 ;
    if ( !model->HSMHV_lvover_Given ) model->HSMHV_lvover = 0.0 ;

    /* Width dependence */
    if ( !model->HSMHV_wvmax_Given ) model->HSMHV_wvmax = 0.0 ;
    if ( !model->HSMHV_wbgtmp1_Given ) model->HSMHV_wbgtmp1 = 0.0 ;
    if ( !model->HSMHV_wbgtmp2_Given ) model->HSMHV_wbgtmp2 = 0.0 ;
    if ( !model->HSMHV_weg0_Given ) model->HSMHV_weg0 = 0.0 ;
    if ( !model->HSMHV_wvfbover_Given ) model->HSMHV_wvfbover = 0.0 ;
    if ( !model->HSMHV_wnover_Given ) model->HSMHV_wnover = 0.0 ;
    if ( !model->HSMHV_wnovers_Given ) model->HSMHV_wnovers = 0.0 ;
    if ( !model->HSMHV_wwl2_Given ) model->HSMHV_wwl2 = 0.0 ;
    if ( !model->HSMHV_wvfbc_Given ) model->HSMHV_wvfbc = 0.0 ;
    if ( !model->HSMHV_wnsubc_Given ) model->HSMHV_wnsubc = 0.0 ;
    if ( !model->HSMHV_wnsubp_Given ) model->HSMHV_wnsubp = 0.0 ;
    if ( !model->HSMHV_wscp1_Given ) model->HSMHV_wscp1 = 0.0 ;
    if ( !model->HSMHV_wscp2_Given ) model->HSMHV_wscp2 = 0.0 ;
    if ( !model->HSMHV_wscp3_Given ) model->HSMHV_wscp3 = 0.0 ;
    if ( !model->HSMHV_wsc1_Given ) model->HSMHV_wsc1 = 0.0 ;
    if ( !model->HSMHV_wsc2_Given ) model->HSMHV_wsc2 = 0.0 ;
    if ( !model->HSMHV_wsc3_Given ) model->HSMHV_wsc3 = 0.0 ;
    if ( !model->HSMHV_wpgd1_Given ) model->HSMHV_wpgd1 = 0.0 ;
    if ( !model->HSMHV_wpgd3_Given ) model->HSMHV_wpgd3 = 0.0 ;
    if ( !model->HSMHV_wndep_Given ) model->HSMHV_wndep = 0.0 ;
    if ( !model->HSMHV_wninv_Given ) model->HSMHV_wninv = 0.0 ;
    if ( !model->HSMHV_wmuecb0_Given ) model->HSMHV_wmuecb0 = 0.0 ;
    if ( !model->HSMHV_wmuecb1_Given ) model->HSMHV_wmuecb1 = 0.0 ;
    if ( !model->HSMHV_wmueph1_Given ) model->HSMHV_wmueph1 = 0.0 ;
    if ( !model->HSMHV_wvtmp_Given ) model->HSMHV_wvtmp = 0.0 ;
    if ( !model->HSMHV_wwvth0_Given ) model->HSMHV_wwvth0 = 0.0 ;
    if ( !model->HSMHV_wmuesr1_Given ) model->HSMHV_wmuesr1 = 0.0 ;
    if ( !model->HSMHV_wmuetmp_Given ) model->HSMHV_wmuetmp = 0.0 ;
    if ( !model->HSMHV_wsub1_Given ) model->HSMHV_wsub1 = 0.0 ;
    if ( !model->HSMHV_wsub2_Given ) model->HSMHV_wsub2 = 0.0 ;
    if ( !model->HSMHV_wsvds_Given ) model->HSMHV_wsvds = 0.0 ;
    if ( !model->HSMHV_wsvbs_Given ) model->HSMHV_wsvbs = 0.0 ;
    if ( !model->HSMHV_wsvgs_Given ) model->HSMHV_wsvgs = 0.0 ;
    if ( !model->HSMHV_wfn1_Given ) model->HSMHV_wfn1 = 0.0 ;
    if ( !model->HSMHV_wfn2_Given ) model->HSMHV_wfn2 = 0.0 ;
    if ( !model->HSMHV_wfn3_Given ) model->HSMHV_wfn3 = 0.0 ;
    if ( !model->HSMHV_wfvbs_Given ) model->HSMHV_wfvbs = 0.0 ;
    if ( !model->HSMHV_wnsti_Given ) model->HSMHV_wnsti = 0.0 ;
    if ( !model->HSMHV_wwsti_Given ) model->HSMHV_wwsti = 0.0 ;
    if ( !model->HSMHV_wscsti1_Given ) model->HSMHV_wscsti1 = 0.0 ;
    if ( !model->HSMHV_wscsti2_Given ) model->HSMHV_wscsti2 = 0.0 ;
    if ( !model->HSMHV_wvthsti_Given ) model->HSMHV_wvthsti = 0.0 ;
    if ( !model->HSMHV_wmuesti1_Given ) model->HSMHV_wmuesti1 = 0.0 ;
    if ( !model->HSMHV_wmuesti2_Given ) model->HSMHV_wmuesti2 = 0.0 ;
    if ( !model->HSMHV_wmuesti3_Given ) model->HSMHV_wmuesti3 = 0.0 ;
    if ( !model->HSMHV_wnsubpsti1_Given ) model->HSMHV_wnsubpsti1 = 0.0 ;
    if ( !model->HSMHV_wnsubpsti2_Given ) model->HSMHV_wnsubpsti2 = 0.0 ;
    if ( !model->HSMHV_wnsubpsti3_Given ) model->HSMHV_wnsubpsti3 = 0.0 ;
    if ( !model->HSMHV_wcgso_Given ) model->HSMHV_wcgso = 0.0 ;
    if ( !model->HSMHV_wcgdo_Given ) model->HSMHV_wcgdo = 0.0 ;
    if ( !model->HSMHV_wjs0_Given ) model->HSMHV_wjs0 = 0.0 ;
    if ( !model->HSMHV_wjs0sw_Given ) model->HSMHV_wjs0sw = 0.0 ;
    if ( !model->HSMHV_wnj_Given ) model->HSMHV_wnj = 0.0 ;
    if ( !model->HSMHV_wcisbk_Given ) model->HSMHV_wcisbk = 0.0 ;
    if ( !model->HSMHV_wclm1_Given ) model->HSMHV_wclm1 = 0.0 ;
    if ( !model->HSMHV_wclm2_Given ) model->HSMHV_wclm2 = 0.0 ;
    if ( !model->HSMHV_wclm3_Given ) model->HSMHV_wclm3 = 0.0 ;
    if ( !model->HSMHV_wwfc_Given ) model->HSMHV_wwfc = 0.0 ;
    if ( !model->HSMHV_wgidl1_Given ) model->HSMHV_wgidl1 = 0.0 ;
    if ( !model->HSMHV_wgidl2_Given ) model->HSMHV_wgidl2 = 0.0 ;
    if ( !model->HSMHV_wgleak1_Given ) model->HSMHV_wgleak1 = 0.0 ;
    if ( !model->HSMHV_wgleak2_Given ) model->HSMHV_wgleak2 = 0.0 ;
    if ( !model->HSMHV_wgleak3_Given ) model->HSMHV_wgleak3 = 0.0 ;
    if ( !model->HSMHV_wgleak6_Given ) model->HSMHV_wgleak6 = 0.0 ;
    if ( !model->HSMHV_wglksd1_Given ) model->HSMHV_wglksd1 = 0.0 ;
    if ( !model->HSMHV_wglksd2_Given ) model->HSMHV_wglksd2 = 0.0 ;
    if ( !model->HSMHV_wglkb1_Given ) model->HSMHV_wglkb1 = 0.0 ;
    if ( !model->HSMHV_wglkb2_Given ) model->HSMHV_wglkb2 = 0.0 ;
    if ( !model->HSMHV_wnftrp_Given ) model->HSMHV_wnftrp = 0.0 ;
    if ( !model->HSMHV_wnfalp_Given ) model->HSMHV_wnfalp = 0.0 ;
    if ( !model->HSMHV_wpthrou_Given ) model->HSMHV_wpthrou = 0.0 ;
    if ( !model->HSMHV_wvdiffj_Given ) model->HSMHV_wvdiffj = 0.0 ;
    if ( !model->HSMHV_wibpc1_Given ) model->HSMHV_wibpc1 = 0.0 ;
    if ( !model->HSMHV_wibpc2_Given ) model->HSMHV_wibpc2 = 0.0 ;
    if ( !model->HSMHV_wcgbo_Given ) model->HSMHV_wcgbo = 0.0 ;
    if ( !model->HSMHV_wcvdsover_Given ) model->HSMHV_wcvdsover = 0.0 ;
    if ( !model->HSMHV_wfalph_Given ) model->HSMHV_wfalph = 0.0 ;
    if ( !model->HSMHV_wnpext_Given ) model->HSMHV_wnpext = 0.0 ;
    if ( !model->HSMHV_wpowrat_Given ) model->HSMHV_wpowrat = 0.0 ;
    if ( !model->HSMHV_wrd_Given ) model->HSMHV_wrd = 0.0 ;
    if ( !model->HSMHV_wrd22_Given ) model->HSMHV_wrd22 = 0.0 ;
    if ( !model->HSMHV_wrd23_Given ) model->HSMHV_wrd23 = 0.0 ;
    if ( !model->HSMHV_wrd24_Given ) model->HSMHV_wrd24 = 0.0 ;
    if ( !model->HSMHV_wrdict1_Given ) model->HSMHV_wrdict1 = 0.0 ;
    if ( !model->HSMHV_wrdov13_Given ) model->HSMHV_wrdov13 = 0.0 ;
    if ( !model->HSMHV_wrdslp1_Given ) model->HSMHV_wrdslp1 = 0.0 ;
    if ( !model->HSMHV_wrdvb_Given ) model->HSMHV_wrdvb = 0.0 ;
    if ( !model->HSMHV_wrdvd_Given ) model->HSMHV_wrdvd = 0.0 ;
    if ( !model->HSMHV_wrdvg11_Given ) model->HSMHV_wrdvg11 = 0.0 ;
    if ( !model->HSMHV_wrs_Given ) model->HSMHV_wrs = 0.0 ;
    if ( !model->HSMHV_wrth0_Given ) model->HSMHV_wrth0 = 0.0 ;
    if ( !model->HSMHV_wvover_Given ) model->HSMHV_wvover = 0.0 ;

    /* Cross-term dependence */
    if ( !model->HSMHV_pvmax_Given ) model->HSMHV_pvmax = 0.0 ;
    if ( !model->HSMHV_pbgtmp1_Given ) model->HSMHV_pbgtmp1 = 0.0 ;
    if ( !model->HSMHV_pbgtmp2_Given ) model->HSMHV_pbgtmp2 = 0.0 ;
    if ( !model->HSMHV_peg0_Given ) model->HSMHV_peg0 = 0.0 ;
    if ( !model->HSMHV_pvfbover_Given ) model->HSMHV_pvfbover = 0.0 ;
    if ( !model->HSMHV_pnover_Given ) model->HSMHV_pnover = 0.0 ;
    if ( !model->HSMHV_pnovers_Given ) model->HSMHV_pnovers = 0.0 ;
    if ( !model->HSMHV_pwl2_Given ) model->HSMHV_pwl2 = 0.0 ;
    if ( !model->HSMHV_pvfbc_Given ) model->HSMHV_pvfbc = 0.0 ;
    if ( !model->HSMHV_pnsubc_Given ) model->HSMHV_pnsubc = 0.0 ;
    if ( !model->HSMHV_pnsubp_Given ) model->HSMHV_pnsubp = 0.0 ;
    if ( !model->HSMHV_pscp1_Given ) model->HSMHV_pscp1 = 0.0 ;
    if ( !model->HSMHV_pscp2_Given ) model->HSMHV_pscp2 = 0.0 ;
    if ( !model->HSMHV_pscp3_Given ) model->HSMHV_pscp3 = 0.0 ;
    if ( !model->HSMHV_psc1_Given ) model->HSMHV_psc1 = 0.0 ;
    if ( !model->HSMHV_psc2_Given ) model->HSMHV_psc2 = 0.0 ;
    if ( !model->HSMHV_psc3_Given ) model->HSMHV_psc3 = 0.0 ;
    if ( !model->HSMHV_ppgd1_Given ) model->HSMHV_ppgd1 = 0.0 ;
    if ( !model->HSMHV_ppgd3_Given ) model->HSMHV_ppgd3 = 0.0 ;
    if ( !model->HSMHV_pndep_Given ) model->HSMHV_pndep = 0.0 ;
    if ( !model->HSMHV_pninv_Given ) model->HSMHV_pninv = 0.0 ;
    if ( !model->HSMHV_pmuecb0_Given ) model->HSMHV_pmuecb0 = 0.0 ;
    if ( !model->HSMHV_pmuecb1_Given ) model->HSMHV_pmuecb1 = 0.0 ;
    if ( !model->HSMHV_pmueph1_Given ) model->HSMHV_pmueph1 = 0.0 ;
    if ( !model->HSMHV_pvtmp_Given ) model->HSMHV_pvtmp = 0.0 ;
    if ( !model->HSMHV_pwvth0_Given ) model->HSMHV_pwvth0 = 0.0 ;
    if ( !model->HSMHV_pmuesr1_Given ) model->HSMHV_pmuesr1 = 0.0 ;
    if ( !model->HSMHV_pmuetmp_Given ) model->HSMHV_pmuetmp = 0.0 ;
    if ( !model->HSMHV_psub1_Given ) model->HSMHV_psub1 = 0.0 ;
    if ( !model->HSMHV_psub2_Given ) model->HSMHV_psub2 = 0.0 ;
    if ( !model->HSMHV_psvds_Given ) model->HSMHV_psvds = 0.0 ;
    if ( !model->HSMHV_psvbs_Given ) model->HSMHV_psvbs = 0.0 ;
    if ( !model->HSMHV_psvgs_Given ) model->HSMHV_psvgs = 0.0 ;
    if ( !model->HSMHV_pfn1_Given ) model->HSMHV_pfn1 = 0.0 ;
    if ( !model->HSMHV_pfn2_Given ) model->HSMHV_pfn2 = 0.0 ;
    if ( !model->HSMHV_pfn3_Given ) model->HSMHV_pfn3 = 0.0 ;
    if ( !model->HSMHV_pfvbs_Given ) model->HSMHV_pfvbs = 0.0 ;
    if ( !model->HSMHV_pnsti_Given ) model->HSMHV_pnsti = 0.0 ;
    if ( !model->HSMHV_pwsti_Given ) model->HSMHV_pwsti = 0.0 ;
    if ( !model->HSMHV_pscsti1_Given ) model->HSMHV_pscsti1 = 0.0 ;
    if ( !model->HSMHV_pscsti2_Given ) model->HSMHV_pscsti2 = 0.0 ;
    if ( !model->HSMHV_pvthsti_Given ) model->HSMHV_pvthsti = 0.0 ;
    if ( !model->HSMHV_pmuesti1_Given ) model->HSMHV_pmuesti1 = 0.0 ;
    if ( !model->HSMHV_pmuesti2_Given ) model->HSMHV_pmuesti2 = 0.0 ;
    if ( !model->HSMHV_pmuesti3_Given ) model->HSMHV_pmuesti3 = 0.0 ;
    if ( !model->HSMHV_pnsubpsti1_Given ) model->HSMHV_pnsubpsti1 = 0.0 ;
    if ( !model->HSMHV_pnsubpsti2_Given ) model->HSMHV_pnsubpsti2 = 0.0 ;
    if ( !model->HSMHV_pnsubpsti3_Given ) model->HSMHV_pnsubpsti3 = 0.0 ;
    if ( !model->HSMHV_pcgso_Given ) model->HSMHV_pcgso = 0.0 ;
    if ( !model->HSMHV_pcgdo_Given ) model->HSMHV_pcgdo = 0.0 ;
    if ( !model->HSMHV_pjs0_Given ) model->HSMHV_pjs0 = 0.0 ;
    if ( !model->HSMHV_pjs0sw_Given ) model->HSMHV_pjs0sw = 0.0 ;
    if ( !model->HSMHV_pnj_Given ) model->HSMHV_pnj = 0.0 ;
    if ( !model->HSMHV_pcisbk_Given ) model->HSMHV_pcisbk = 0.0 ;
    if ( !model->HSMHV_pclm1_Given ) model->HSMHV_pclm1 = 0.0 ;
    if ( !model->HSMHV_pclm2_Given ) model->HSMHV_pclm2 = 0.0 ;
    if ( !model->HSMHV_pclm3_Given ) model->HSMHV_pclm3 = 0.0 ;
    if ( !model->HSMHV_pwfc_Given ) model->HSMHV_pwfc = 0.0 ;
    if ( !model->HSMHV_pgidl1_Given ) model->HSMHV_pgidl1 = 0.0 ;
    if ( !model->HSMHV_pgidl2_Given ) model->HSMHV_pgidl2 = 0.0 ;
    if ( !model->HSMHV_pgleak1_Given ) model->HSMHV_pgleak1 = 0.0 ;
    if ( !model->HSMHV_pgleak2_Given ) model->HSMHV_pgleak2 = 0.0 ;
    if ( !model->HSMHV_pgleak3_Given ) model->HSMHV_pgleak3 = 0.0 ;
    if ( !model->HSMHV_pgleak6_Given ) model->HSMHV_pgleak6 = 0.0 ;
    if ( !model->HSMHV_pglksd1_Given ) model->HSMHV_pglksd1 = 0.0 ;
    if ( !model->HSMHV_pglksd2_Given ) model->HSMHV_pglksd2 = 0.0 ;
    if ( !model->HSMHV_pglkb1_Given ) model->HSMHV_pglkb1 = 0.0 ;
    if ( !model->HSMHV_pglkb2_Given ) model->HSMHV_pglkb2 = 0.0 ;
    if ( !model->HSMHV_pnftrp_Given ) model->HSMHV_pnftrp = 0.0 ;
    if ( !model->HSMHV_pnfalp_Given ) model->HSMHV_pnfalp = 0.0 ;
    if ( !model->HSMHV_ppthrou_Given ) model->HSMHV_ppthrou = 0.0 ;
    if ( !model->HSMHV_pvdiffj_Given ) model->HSMHV_pvdiffj = 0.0 ;
    if ( !model->HSMHV_pibpc1_Given ) model->HSMHV_pibpc1 = 0.0 ;
    if ( !model->HSMHV_pibpc2_Given ) model->HSMHV_pibpc2 = 0.0 ;
    if ( !model->HSMHV_pcgbo_Given ) model->HSMHV_pcgbo = 0.0 ;
    if ( !model->HSMHV_pcvdsover_Given ) model->HSMHV_pcvdsover = 0.0 ;
    if ( !model->HSMHV_pfalph_Given ) model->HSMHV_pfalph = 0.0 ;
    if ( !model->HSMHV_pnpext_Given ) model->HSMHV_pnpext = 0.0 ;
    if ( !model->HSMHV_ppowrat_Given ) model->HSMHV_ppowrat = 0.0 ;
    if ( !model->HSMHV_prd_Given ) model->HSMHV_prd = 0.0 ;
    if ( !model->HSMHV_prd22_Given ) model->HSMHV_prd22 = 0.0 ;
    if ( !model->HSMHV_prd23_Given ) model->HSMHV_prd23 = 0.0 ;
    if ( !model->HSMHV_prd24_Given ) model->HSMHV_prd24 = 0.0 ;
    if ( !model->HSMHV_prdict1_Given ) model->HSMHV_prdict1 = 0.0 ;
    if ( !model->HSMHV_prdov13_Given ) model->HSMHV_prdov13 = 0.0 ;
    if ( !model->HSMHV_prdslp1_Given ) model->HSMHV_prdslp1 = 0.0 ;
    if ( !model->HSMHV_prdvb_Given ) model->HSMHV_prdvb = 0.0 ;
    if ( !model->HSMHV_prdvd_Given ) model->HSMHV_prdvd = 0.0 ;
    if ( !model->HSMHV_prdvg11_Given ) model->HSMHV_prdvg11 = 0.0 ;
    if ( !model->HSMHV_prs_Given ) model->HSMHV_prs = 0.0 ;
    if ( !model->HSMHV_prth0_Given ) model->HSMHV_prth0 = 0.0 ;
    if ( !model->HSMHV_pvover_Given ) model->HSMHV_pvover = 0.0 ;

    if (  model->HSMHV_rd26_Given   ) model->HSMHV_qovsm   = model->HSMHV_rd26 ;
    if (  model->HSMHV_ldrift_Given ) model->HSMHV_ldrift2 = model->HSMHV_ldrift ;

    if (!model->HSMHVvgsMaxGiven) model->HSMHVvgsMax = 1e99;
    if (!model->HSMHVvgdMaxGiven) model->HSMHVvgdMax = 1e99;
    if (!model->HSMHVvgbMaxGiven) model->HSMHVvgbMax = 1e99;
    if (!model->HSMHVvdsMaxGiven) model->HSMHVvdsMax = 1e99;
    if (!model->HSMHVvbsMaxGiven) model->HSMHVvbsMax = 1e99;
    if (!model->HSMHVvbdMaxGiven) model->HSMHVvbdMax = 1e99;
    if (!model->HSMHVvgsrMaxGiven) model->HSMHVvgsrMax = 1e99;
    if (!model->HSMHVvgdrMaxGiven) model->HSMHVvgdrMax = 1e99;
    if (!model->HSMHVvgbrMaxGiven) model->HSMHVvgbrMax = 1e99;
    if (!model->HSMHVvbsrMaxGiven) model->HSMHVvbsrMax = 1e99;
    if (!model->HSMHVvbdrMaxGiven) model->HSMHVvbdrMax = 1e99;

    /* For Symmetrical Device */
    if (  model->HSMHV_cosym ) {
       if(!model->HSMHV_rs_Given  )
         { model->HSMHV_rs = model->HSMHV_rd  ; }
       if(!model->HSMHV_coovlps_Given  )
         { model->HSMHV_coovlps = model->HSMHV_coovlp  ; }
       if(!model->HSMHV_novers_Given   )
         { model->HSMHV_novers  = model->HSMHV_nover   ; }
/*        if(!model->HSMHV_xld_Given   ) */
/*          { model->HSMHV_xld  = model->HSMHV_xldld   ; } */
       if(!model->HSMHV_lover_Given    ) 
         { model->HSMHV_lover  = model->HSMHV_loverld ; }
       if(!model->HSMHV_lovers_Given   ) 
         { model->HSMHV_lovers  = model->HSMHV_loverld ; }
       if(!model->HSMHV_ldrift1s_Given ) 
         { model->HSMHV_ldrift1s = model->HSMHV_ldrift1 ; }
       if(!model->HSMHV_ldrift2s_Given )  
         { model->HSMHV_ldrift2s = model->HSMHV_ldrift2 ; }
       if(!model->HSMHV_cgso_Given     ) { model->HSMHV_cgso       = model->HSMHV_cgdo ;
                                           model->HSMHV_cgso_Given = model->HSMHV_cgdo_Given ; }
    }

    if ( model->HSMHV_xqy > 0.0 && model->HSMHV_xqy < 1.0e-9 ) {
       fprintf ( stderr , "*** warning(HiSIMHV): XQY (%e[m]) is too small -> reset to 1nm.\n" , model->HSMHV_xqy ) ;
       model->HSMHV_xqy = 1e-9 ;
    }

    modelMKS = &model->modelMKS ;

    /* loop through all the instances of the model */
    for ( here = HSMHVinstances(model);here != NULL ;
         here = HSMHVnextInstance(here)) {
      /* allocate a chunk of the state vector */
      here->HSMHVstates = *states;
      if (model->HSMHV_conqs)
	*states += HSMHVnumStatesNqs;
      else 
	*states += HSMHVnumStates;

      hereMKS  = &here->hereMKS ;
      /* perform the device parameter defaulting */
      if ( !here->HSMHV_coselfheat_Given ) here->HSMHV_coselfheat = model->HSMHV_coselfheat ;
      if ( !here->HSMHV_cosubnode_Given  ) here->HSMHV_cosubnode  = model->HSMHV_cosubnode ;
      if ( !here->HSMHV_l_Given      ) here->HSMHV_l      = 2.0e-6 ;
      if ( !here->HSMHV_w_Given      ) here->HSMHV_w      = 5.0e-6 ;
      if ( !here->HSMHV_ad_Given     ) here->HSMHV_ad     = 0.0 ;
      if ( !here->HSMHV_as_Given     ) here->HSMHV_as     = 0.0 ;
      if ( !here->HSMHV_pd_Given     ) here->HSMHV_pd     = 0.0 ;
      if ( !here->HSMHV_ps_Given     ) here->HSMHV_ps     = 0.0 ;
      if ( !here->HSMHV_nrd_Given    ) here->HSMHV_nrd    = 1.0 ;
      if ( !here->HSMHV_nrs_Given    ) here->HSMHV_nrs    = 1.0 ;
      if ( !here->HSMHV_ngcon_Given  ) here->HSMHV_ngcon  = 1.0 ;
      if ( !here->HSMHV_xgw_Given    ) here->HSMHV_xgw    = 0e0 ;
      if ( !here->HSMHV_xgl_Given    ) here->HSMHV_xgl    = 0e0 ;
      if ( !here->HSMHV_nf_Given     ) here->HSMHV_nf     = 1.0 ;
      if ( !here->HSMHV_sa_Given     ) here->HSMHV_sa     = 0 ;
      if ( !here->HSMHV_sb_Given     ) here->HSMHV_sb     = 0 ;
      if ( !here->HSMHV_sd_Given     ) here->HSMHV_sd     = 0 ;
      if ( !here->HSMHV_dtemp_Given  ) here->HSMHV_dtemp  = 0.0 ;

      if ( !here->HSMHV_icVBS_Given ) here->HSMHV_icVBS = 0.0;
      if ( !here->HSMHV_icVDS_Given ) here->HSMHV_icVDS = 0.0;
      if ( !here->HSMHV_icVGS_Given ) here->HSMHV_icVGS = 0.0;

      if ( !here->HSMHV_corbnet_Given )
	here->HSMHV_corbnet = model->HSMHV_corbnet ;
      else if ( here->HSMHV_corbnet != 0 && here->HSMHV_corbnet != 1 ) {
	here->HSMHV_corbnet = model->HSMHV_corbnet ;
	printf("warning(HiSIMHV): CORBNET has been set to its default value: %d.\n", here->HSMHV_corbnet);
      }
      if ( !here->HSMHV_rbdb_Given) here->HSMHV_rbdb = model->HSMHV_rbdb; /* not used in this version */
      if ( !here->HSMHV_rbsb_Given) here->HSMHV_rbsb = model->HSMHV_rbsb; /* not used in this version */
      if ( !here->HSMHV_rbpb_Given) here->HSMHV_rbpb = model->HSMHV_rbpb;
      if ( !here->HSMHV_rbps_Given) here->HSMHV_rbps = model->HSMHV_rbps;
      if ( !here->HSMHV_rbpd_Given) here->HSMHV_rbpd = model->HSMHV_rbpd;

      if ( !here->HSMHV_corg_Given )
	here->HSMHV_corg = model->HSMHV_corg ;
      else if ( here->HSMHV_corg != 0 && here->HSMHV_corg != 1 ) {
	here->HSMHV_corg = model->HSMHV_corg ;
	printf("warning(HiSIMHV): CORG has been set to its default value: %d.\n", here->HSMHV_corg);
      }

      if ( !here->HSMHV_m_Given      ) here->HSMHV_m      = 1.0 ;
      if ( !here->HSMHV_subld1_Given  ) here->HSMHV_subld1  = model->HSMHV_subld1 ;
      if ( !here->HSMHV_subld2_Given  ) here->HSMHV_subld2  = model->HSMHV_subld2 ;
      if ( !here->HSMHV_lovers_Given  ) here->HSMHV_lovers  = model->HSMHV_lovers ;
      if (  here->HSMHV_lover_Given   ) here->HSMHV_lovers  = here->HSMHV_lover ;
      if ( !here->HSMHV_loverld_Given ) here->HSMHV_loverld = model->HSMHV_loverld ;
      if ( !here->HSMHV_ldrift1_Given  ) here->HSMHV_ldrift1 = model->HSMHV_ldrift1 ;
      if ( !here->HSMHV_ldrift2_Given  ) here->HSMHV_ldrift2 = model->HSMHV_ldrift2 ;
      if ( !here->HSMHV_ldrift1s_Given ) here->HSMHV_ldrift1s = model->HSMHV_ldrift1s ;
      if ( !here->HSMHV_ldrift2s_Given ) here->HSMHV_ldrift2s = model->HSMHV_ldrift2s ;

      if (  model->HSMHV_cosym ) {
         if ( !here->HSMHV_lovers_Given   && !model->HSMHV_lovers_Given ) here->HSMHV_lovers = here->HSMHV_loverld ;
         here->HSMHV_lover  = here->HSMHV_lovers ;
         if ( !here->HSMHV_ldrift1s_Given && !model->HSMHV_ldrift1s_Given ) here->HSMHV_ldrift1s = here->HSMHV_ldrift1 ;
         if ( !here->HSMHV_ldrift2s_Given && !model->HSMHV_ldrift2s_Given ) here->HSMHV_ldrift2s = here->HSMHV_ldrift2 ;
      }




      /* process drain series resistance */
      /* rough check if Rd != 0 *  ****  don't forget to change if Rd processing is changed *******/
      T2 = ( here->HSMHV_ldrift1 * model->HSMHV_rdslp1 * C_m2um  + model->HSMHV_rdict1 )
	   * ( here->HSMHV_ldrift2 * model->HSMHV_rdslp2 * C_m2um  + model->HSMHV_rdict2 ) ;
      Rd = model->HSMHV_rsh * here->HSMHV_nrd * here->HSMHV_nf + (model->HSMHV_rd + model->HSMHV_rdvd) * T2 ;
      if ( (model->HSMHV_corsrd == 1 || model->HSMHV_corsrd == 3)
           && Rd > 0.0 ) {
	if(here->HSMHVdNodePrime <= 0) {
        model->HSMHV_rd   = ( model->HSMHV_rd   == 0.0 ) ? 1e-50 :  model->HSMHV_rd ;
        error = CKTmkVolt(ckt, &tmp, here->HSMHVname, "drain");
	if (error) return(error);
	here->HSMHVdNodePrime = tmp->number;
       }
      } else {
	here->HSMHVdNodePrime = here->HSMHVdNode;
      }
      here->HSMHVdrainConductance = 0.0 ; /* initialized for hsmhvnoi.c */
      
      /* process source series resistance */
      /* rough check if Rs != 0 *  ***** don't forget to change if Rs processing is changed *******/
      T2 = ( here->HSMHV_ldrift1s * model->HSMHV_rdslp1 * C_m2um + model->HSMHV_rdict1 )
	   * ( here->HSMHV_ldrift2s * model->HSMHV_rdslp2 * C_m2um + model->HSMHV_rdict2 ) ;
      Rs = model->HSMHV_rsh * here->HSMHV_nrs * here->HSMHV_nf + model->HSMHV_rs * T2 ;
      if ( (model->HSMHV_corsrd == 1 || model->HSMHV_corsrd == 3)
           && Rs > 0.0 ) {
       if(here->HSMHVsNodePrime == 0) {
        error = CKTmkVolt(ckt, &tmp, here->HSMHVname, "source");
	if (error) return(error);
	here->HSMHVsNodePrime = tmp->number;
       }
      } else {
	here->HSMHVsNodePrime = here->HSMHVsNode;
      }
      here->HSMHVsourceConductance = 0.0 ; /* initialized for hsmhvnoi.c */

      /* process gate resistance */
      if ( (here->HSMHV_corg == 1 && model->HSMHV_rshg > 0.0) ) {
       if(here->HSMHVgNodePrime == 0) {
	error = CKTmkVolt(ckt, &tmp, here->HSMHVname, "gate");
	if (error) return(error);
	here->HSMHVgNodePrime = tmp->number;
       }
      } else {
	here->HSMHVgNodePrime = here->HSMHVgNode;
      }

      /* internal body nodes for body resistance model */
      if ( here->HSMHV_corbnet == 1 ) {
	if (here->HSMHVdbNode == 0) {
	  error = CKTmkVolt(ckt, &tmp, here->HSMHVname, "dbody");
	  if (error) return(error);
	  here->HSMHVdbNode = tmp->number;
	}
	if (here->HSMHVbNodePrime == 0) {
	  error = CKTmkVolt(ckt, &tmp,here->HSMHVname, "body");
	  if (error) return(error);
	  here->HSMHVbNodePrime = tmp->number;
	}
	if (here->HSMHVsbNode == 0) {
	  error = CKTmkVolt(ckt, &tmp, here->HSMHVname,"sbody");
	  if (error) return(error);
	  here->HSMHVsbNode = tmp->number;
	}
      } else {
	here->HSMHVdbNode = here->HSMHVbNodePrime = here->HSMHVsbNode = here->HSMHVbNode;
      }

      here->HSMHVtempNode = here->HSMHVtempNodeExt;
      here->HSMHVsubNode = here->HSMHVsubNodeExt;

      if ( here->HSMHV_cosubnode == 0 && here->HSMHVsubNode >= 0 ) {
        if ( here->HSMHVtempNode >= 0 ) {
       /* FATAL Error when 6th node is defined and COSUBNODE=0 */
          SPfrontEnd->IFerrorf
            (
             ERR_FATAL,
             "HiSIM_HV: MOSFET(%s) MODEL(%s): 6th node is defined and COSUBNODE=0",
             here->HSMHVname, model->HSMHVmodName
             );
          return (E_BADPARM);
        } else {

      /* 5th node is switched to tempNode, if COSUBNODE=0 and 5 external nodes are assigned. */
          if ( here->HSMHVsubNode > 0 ) {
    	    here->HSMHVtempNode = here->HSMHVsubNode ;
	    here->HSMHVsubNode  = -1 ;
          }
        }
      }

      /* self heating*/
      if ( here->HSMHV_coselfheat >  0 && here->HSMHVtempNode <= 0 ){
	error = CKTmkVolt(ckt, &tmp, here->HSMHVname,"temp");
	if(error) return(error);
	here->HSMHVtempNode = tmp->number;
      }
      if ( here->HSMHV_coselfheat <= 0 ) here->HSMHVtempNode = -1;

      /* flat handling of NQS */
      if ( model->HSMHV_conqs ){
	error = CKTmkVolt(ckt, &tmp, here->HSMHVname,"qi_nqs");
	if(error) return(error);
	here->HSMHVqiNode = tmp->number;
	error = CKTmkVolt(ckt, &tmp, here->HSMHVname,"qb_nqs");
	if(error) return(error);
	here->HSMHVqbNode = tmp->number;
      }
      
                   
      /* set Sparse Matrix Pointers */
      
      /* macro to make elements with built in test for out of memory */
#define TSTALLOC(ptr,first,second) \
do { if((here->ptr = SMPmakeElt(matrix,here->first,here->second))==NULL){\
    return(E_NOMEM);\
} } while(0)

      TSTALLOC(HSMHVDPbpPtr, HSMHVdNodePrime, HSMHVbNodePrime);
      TSTALLOC(HSMHVSPbpPtr, HSMHVsNodePrime, HSMHVbNodePrime);
      TSTALLOC(HSMHVGPbpPtr, HSMHVgNodePrime, HSMHVbNodePrime);

      TSTALLOC(HSMHVBPdPtr,  HSMHVbNodePrime, HSMHVdNode);
      TSTALLOC(HSMHVBPsPtr,  HSMHVbNodePrime, HSMHVsNode);
      TSTALLOC(HSMHVBPdpPtr, HSMHVbNodePrime, HSMHVdNodePrime);
      TSTALLOC(HSMHVBPspPtr, HSMHVbNodePrime, HSMHVsNodePrime);
      TSTALLOC(HSMHVBPgpPtr, HSMHVbNodePrime, HSMHVgNodePrime);
      TSTALLOC(HSMHVBPbpPtr, HSMHVbNodePrime, HSMHVbNodePrime);

      TSTALLOC(HSMHVDdPtr, HSMHVdNode, HSMHVdNode);
      TSTALLOC(HSMHVGPgpPtr, HSMHVgNodePrime, HSMHVgNodePrime);
      TSTALLOC(HSMHVSsPtr, HSMHVsNode, HSMHVsNode);
      TSTALLOC(HSMHVDPdpPtr, HSMHVdNodePrime, HSMHVdNodePrime);
      TSTALLOC(HSMHVSPspPtr, HSMHVsNodePrime, HSMHVsNodePrime);
      TSTALLOC(HSMHVDdpPtr, HSMHVdNode, HSMHVdNodePrime);
      TSTALLOC(HSMHVGPdpPtr, HSMHVgNodePrime, HSMHVdNodePrime);
      TSTALLOC(HSMHVGPspPtr, HSMHVgNodePrime, HSMHVsNodePrime);
      TSTALLOC(HSMHVSspPtr, HSMHVsNode, HSMHVsNodePrime);
      TSTALLOC(HSMHVDPspPtr, HSMHVdNodePrime, HSMHVsNodePrime);
      TSTALLOC(HSMHVDPdPtr, HSMHVdNodePrime, HSMHVdNode);
      TSTALLOC(HSMHVDPgpPtr, HSMHVdNodePrime, HSMHVgNodePrime);
      TSTALLOC(HSMHVSPgpPtr, HSMHVsNodePrime, HSMHVgNodePrime);
      TSTALLOC(HSMHVSPsPtr, HSMHVsNodePrime, HSMHVsNode);
      TSTALLOC(HSMHVSPdpPtr, HSMHVsNodePrime, HSMHVdNodePrime);

      TSTALLOC(HSMHVGgPtr, HSMHVgNode, HSMHVgNode);
      TSTALLOC(HSMHVGgpPtr, HSMHVgNode, HSMHVgNodePrime);
      TSTALLOC(HSMHVGPgPtr, HSMHVgNodePrime, HSMHVgNode);
      /* TSTALLOC(HSMHVGdpPtr, HSMHVgNode, HSMHVdNodePrime);	not used */
      /* TSTALLOC(HSMHVGspPtr, HSMHVgNode, HSMHVsNodePrime);	not used */
      /* TSTALLOC(HSMHVGbpPtr, HSMHVgNode, HSMHVbNodePrime);	not used */
      TSTALLOC(HSMHVDdbPtr, HSMHVdNode, HSMHVdbNode);
      TSTALLOC(HSMHVSsbPtr, HSMHVsNode, HSMHVsbNode);

      TSTALLOC(HSMHVDBdPtr, HSMHVdbNode, HSMHVdNode);
      TSTALLOC(HSMHVDBdbPtr, HSMHVdbNode, HSMHVdbNode);
      TSTALLOC(HSMHVDBbpPtr, HSMHVdbNode, HSMHVbNodePrime);
      /* TSTALLOC(HSMHVDBbPtr, HSMHVdbNode, HSMHVbNode);	not used */

      TSTALLOC(HSMHVBPdbPtr, HSMHVbNodePrime, HSMHVdbNode);
      TSTALLOC(HSMHVBPbPtr, HSMHVbNodePrime, HSMHVbNode);
      TSTALLOC(HSMHVBPsbPtr, HSMHVbNodePrime, HSMHVsbNode);

      TSTALLOC(HSMHVSBsPtr, HSMHVsbNode, HSMHVsNode);
      TSTALLOC(HSMHVSBbpPtr, HSMHVsbNode, HSMHVbNodePrime);
      /* TSTALLOC(HSMHVSBbPtr, HSMHVsbNode, HSMHVbNode);	not used */
      TSTALLOC(HSMHVSBsbPtr, HSMHVsbNode, HSMHVsbNode);

      /* TSTALLOC(HSMHVBdbPtr, HSMHVbNode, HSMHVdbNode);	not used */
      TSTALLOC(HSMHVBbpPtr, HSMHVbNode, HSMHVbNodePrime);
      /* TSTALLOC(HSMHVBsbPtr, HSMHVbNode, HSMHVsbNode);	not used */
      TSTALLOC(HSMHVBbPtr, HSMHVbNode, HSMHVbNode);

      TSTALLOC(HSMHVDgpPtr, HSMHVdNode, HSMHVgNodePrime);
      TSTALLOC(HSMHVDsPtr, HSMHVdNode, HSMHVsNode);
      TSTALLOC(HSMHVDbpPtr, HSMHVdNode, HSMHVbNodePrime);
      TSTALLOC(HSMHVDspPtr, HSMHVdNode, HSMHVsNodePrime);
      TSTALLOC(HSMHVDPsPtr, HSMHVdNodePrime, HSMHVsNode);

      TSTALLOC(HSMHVSgpPtr, HSMHVsNode, HSMHVgNodePrime);
      TSTALLOC(HSMHVSdPtr, HSMHVsNode, HSMHVdNode);
      TSTALLOC(HSMHVSbpPtr, HSMHVsNode, HSMHVbNodePrime);
      TSTALLOC(HSMHVSdpPtr, HSMHVsNode, HSMHVdNodePrime);
      TSTALLOC(HSMHVSPdPtr, HSMHVsNodePrime, HSMHVdNode);

      TSTALLOC(HSMHVGPdPtr, HSMHVgNodePrime, HSMHVdNode);
      TSTALLOC(HSMHVGPsPtr, HSMHVgNodePrime, HSMHVsNode);
	
      if ( here->HSMHVsubNode > 0 ) { /* 5th substrate node */
	TSTALLOC(HSMHVDsubPtr,  HSMHVdNode,      HSMHVsubNode);
	TSTALLOC(HSMHVDPsubPtr, HSMHVdNodePrime, HSMHVsubNode);
	TSTALLOC(HSMHVSsubPtr,  HSMHVsNode,      HSMHVsubNode);
	TSTALLOC(HSMHVSPsubPtr, HSMHVsNodePrime, HSMHVsubNode);
      }
      if ( here->HSMHV_coselfheat >  0 ) { /* self heating */
	TSTALLOC(HSMHVTemptempPtr, HSMHVtempNode, HSMHVtempNode);
	TSTALLOC(HSMHVTempdPtr, HSMHVtempNode, HSMHVdNode);
	TSTALLOC(HSMHVTempdpPtr, HSMHVtempNode, HSMHVdNodePrime);
	TSTALLOC(HSMHVTempsPtr, HSMHVtempNode, HSMHVsNode);
	TSTALLOC(HSMHVTempspPtr, HSMHVtempNode, HSMHVsNodePrime);
	TSTALLOC(HSMHVDPtempPtr, HSMHVdNodePrime, HSMHVtempNode);
	TSTALLOC(HSMHVSPtempPtr, HSMHVsNodePrime, HSMHVtempNode);
  
        TSTALLOC(HSMHVTempgpPtr, HSMHVtempNode, HSMHVgNodePrime);
	TSTALLOC(HSMHVTempbpPtr, HSMHVtempNode, HSMHVbNodePrime);

	TSTALLOC(HSMHVGPtempPtr, HSMHVgNodePrime, HSMHVtempNode);
	TSTALLOC(HSMHVBPtempPtr, HSMHVbNodePrime, HSMHVtempNode);

	TSTALLOC(HSMHVDBtempPtr, HSMHVdbNode, HSMHVtempNode);
	TSTALLOC(HSMHVSBtempPtr, HSMHVsbNode, HSMHVtempNode);
	TSTALLOC(HSMHVDtempPtr, HSMHVdNode, HSMHVtempNode);
	TSTALLOC(HSMHVStempPtr, HSMHVsNode, HSMHVtempNode);
      }
      if ( model->HSMHV_conqs ) { /* flat handling of NQS */
	TSTALLOC(HSMHVDPqiPtr, HSMHVdNodePrime, HSMHVqiNode);
	TSTALLOC(HSMHVGPqiPtr, HSMHVgNodePrime, HSMHVqiNode);
	TSTALLOC(HSMHVGPqbPtr, HSMHVgNodePrime, HSMHVqbNode);
	TSTALLOC(HSMHVSPqiPtr, HSMHVsNodePrime, HSMHVqiNode);
	TSTALLOC(HSMHVBPqbPtr, HSMHVbNodePrime, HSMHVqbNode);
	TSTALLOC(HSMHVQIdpPtr, HSMHVqiNode, HSMHVdNodePrime);
	TSTALLOC(HSMHVQIgpPtr, HSMHVqiNode, HSMHVgNodePrime);
	TSTALLOC(HSMHVQIspPtr, HSMHVqiNode, HSMHVsNodePrime);
	TSTALLOC(HSMHVQIbpPtr, HSMHVqiNode, HSMHVbNodePrime);
	TSTALLOC(HSMHVQIqiPtr, HSMHVqiNode, HSMHVqiNode);
	TSTALLOC(HSMHVQBdpPtr, HSMHVqbNode, HSMHVdNodePrime);
	TSTALLOC(HSMHVQBgpPtr, HSMHVqbNode, HSMHVgNodePrime);
	TSTALLOC(HSMHVQBspPtr, HSMHVqbNode, HSMHVsNodePrime);
	TSTALLOC(HSMHVQBbpPtr, HSMHVqbNode, HSMHVbNodePrime);
	TSTALLOC(HSMHVQBqbPtr, HSMHVqbNode, HSMHVqbNode);
        if ( here->HSMHV_coselfheat >  0 ) { /* self heating */
	  TSTALLOC(HSMHVQItempPtr, HSMHVqiNode, HSMHVtempNode);
	  TSTALLOC(HSMHVQBtempPtr, HSMHVqbNode, HSMHVtempNode);
        }
      }




      /*-----------------------------------------------------------*
       * Range check of instance parameters
       *-----------------*/
      RANGECHECK(here->HSMHV_l, model->HSMHV_lmin, model->HSMHV_lmax, "L") ;
      RANGECHECK(here->HSMHV_w/here->HSMHV_nf, model->HSMHV_wmin, model->HSMHV_wmax, "W/NF") ;

      /* binning calculation */
      pParam = &here->pParam ;
      Lgate = here->HSMHV_l + model->HSMHV_xl ;
      Wgate = here->HSMHV_w / here->HSMHV_nf  + model->HSMHV_xw ;
      LG = Lgate * C_m2um ;
      WG = Wgate * C_m2um ;
      Lbin = pow(LG, model->HSMHV_lbinn) ;
      Wbin = pow(WG, model->HSMHV_wbinn) ;
      LWbin = Lbin * Wbin ;

      BINNING(vmax);
      BINNING(bgtmp1);
      BINNING(bgtmp2);
      BINNING(eg0);
      BINNING(vfbover);
      BINNING(nover);
      BINNING(novers);
      BINNING(wl2);
      BINNING(vfbc);
      BINNING(nsubc);
      BINNING(nsubp);
      BINNING(scp1);
      BINNING(scp2);
      BINNING(scp3);
      BINNING(sc1);
      BINNING(sc2);
      BINNING(sc3);
      BINNING(pgd1);
      BINNING(pgd3);
      BINNING(ndep);
      BINNING(ninv);
      BINNING(muecb0);
      BINNING(muecb1);
      BINNING(mueph1);
      BINNING(vtmp);
      BINNING(wvth0);
      BINNING(muesr1);
      BINNING(muetmp);
      BINNING(sub1);
      BINNING(sub2);
      BINNING(svds);
      BINNING(svbs);
      BINNING(svgs);
      BINNING(fn1);
      BINNING(fn2);
      BINNING(fn3);
      BINNING(fvbs);
      BINNING(nsti);
      BINNING(wsti);
      BINNING(scsti1);
      BINNING(scsti2);
      BINNING(vthsti);
      BINNING(muesti1);
      BINNING(muesti2);
      BINNING(muesti3);
      BINNING(nsubpsti1);
      BINNING(nsubpsti2);
      BINNING(nsubpsti3);
      BINNING(cgso);
      BINNING(cgdo);
      BINNING(js0);
      BINNING(js0sw);
      BINNING(nj);
      BINNING(cisbk);
      BINNING(clm1);
      BINNING(clm2);
      BINNING(clm3);
      BINNING(wfc);
      BINNING(gidl1);
      BINNING(gidl2);
      BINNING(gleak1);
      BINNING(gleak2);
      BINNING(gleak3);
      BINNING(gleak6);
      BINNING(glksd1);
      BINNING(glksd2);
      BINNING(glkb1);
      BINNING(glkb2);
      BINNING(nftrp);
      BINNING(nfalp);
      BINNING(pthrou);
      BINNING(vdiffj);
      BINNING(ibpc1);
      BINNING(ibpc2);
      BINNING(cgbo);
      BINNING(cvdsover);
      BINNING(falph);
      BINNING(npext);
      BINNING(powrat);
      BINNING(rd);
      BINNING(rd22);
      BINNING(rd23);
      BINNING(rd24);
      BINNING(rdict1);
      BINNING(rdov13);
      BINNING(rdslp1);
      BINNING(rdvb);
      BINNING(rdvd);
      BINNING(rdvg11);
      BINNING(rs);
      BINNING(rth0);
      BINNING(vover);

      /*-----------------------------------------------------------*
       * Range check of model parameters
       *-----------------*/
      RANGECHECK(pParam->HSMHV_vmax,     1.0e6,   20.0e6, "VMAX") ;
      RANGECHECK(pParam->HSMHV_bgtmp1, 50.0e-6,   1.0e-3, "BGTMP1") ;
      RANGECHECK(pParam->HSMHV_bgtmp2, -1.0e-6,   1.0e-6, "BGTMP2") ;
      RANGECHECK(pParam->HSMHV_eg0,        1.0,      1.3, "EG0") ;
      RANGECHECK(pParam->HSMHV_vfbover,   -1.0,      1.0, "VFBOVER") ;
      RANGECHECK(pParam->HSMHV_vfbc,      -1.2,     -0.8, "VFBC") ;
      RANGECHECK(pParam->HSMHV_nsubc,   1.0e16,   1.0e19, "NSUBC") ;
      RANGECHECK(pParam->HSMHV_nsubp,   1.0e16,   1.0e19, "NSUBP") ;
      RANGECHECK(pParam->HSMHV_scp1,       0.0,     20.0, "SCP1") ;
      RANGECHECK(pParam->HSMHV_scp2,       0.0,      2.0, "SCP2") ;
      RANGECHECK(pParam->HSMHV_scp3,       0.0,   200e-9, "SCP3") ;
      RANGECHECK(pParam->HSMHV_sc1,        0.0,     20.0, "SC1") ;
      RANGECHECK(pParam->HSMHV_sc2,        0.0,      2.0, "SC2") ;
      RANGECHECK(pParam->HSMHV_sc3,        0.0,   200e-9, "SC3") ;
      RANGECHECK(pParam->HSMHV_pgd1,       0.0,  50.0e-3, "PGD1") ;
      RANGECHECK(pParam->HSMHV_pgd3,       0.0,      1.2, "PGD3") ;
      RANGECHECK(pParam->HSMHV_ndep,       0.0,      1.0, "NDEP") ;
      RANGECHECK(pParam->HSMHV_ninv,       0.0,      1.0, "NINV") ;
      RANGECHECK(pParam->HSMHV_muecb0,   100.0,  100.0e3, "MUECB0") ;
      RANGECHECK(pParam->HSMHV_muecb1,     5.0,   10.0e3, "MUECB1") ;
      RANGECHECK(pParam->HSMHV_mueph1,   2.0e3,   30.0e3, "MUEPH1") ;
      RANGECHECK(pParam->HSMHV_vtmp,      -2.0,      1.0, "VTMP") ;
      RANGECHECK(pParam->HSMHV_muesr1,  1.0e14,   1.0e16, "MUESR1") ;
      RANGECHECK(pParam->HSMHV_muetmp,     0.5,      2.0, "MUETMP") ;
      RANGECHECK(pParam->HSMHV_clm1,       0.01,      1.0, "CLM1") ;
      RANGECHECK(pParam->HSMHV_clm2,       1.0,      4.0, "CLM2") ;
      RANGECHECK(pParam->HSMHV_clm3,       0.5,      5.0, "CLM3") ;
      RANGECHECK(pParam->HSMHV_wfc,   -5.0e-15,   1.0e-6, "WFC") ;
      RANGECHECK(pParam->HSMHV_cgso,       0.0, 100e-9 * 100*C_VAC*model->HSMHV_kappa/model->HSMHV_tox*C_m2cm, "CGSO") ;
      RANGECHECK(pParam->HSMHV_cgdo,       0.0, 100e-9 * 100*C_VAC*model->HSMHV_kappa/model->HSMHV_tox*C_m2cm, "CGDO") ;
      RANGECHECK(pParam->HSMHV_pthrou,     0.0,  50.0e-3, "PTHROU") ;
      RANGECHECK(pParam->HSMHV_ibpc1,      0.0,   1.0e12, "IBPC1") ;
      RANGECHECK(pParam->HSMHV_ibpc2,      0.0,   1.0e12, "IBPC2") ;
      RANGECHECK(pParam->HSMHV_cvdsover,   0.0,      1.0, "CVDSOVER") ;
      RANGECHECK(pParam->HSMHV_nsti,    1.0e16,   1.0e19, "NSTI") ;
      if ( pParam->HSMHV_cgbo < 0.0 ) { 
        printf("warning(HiSIMHV): %s = %e\n", "CGBO", pParam->HSMHV_cgbo ); 
        printf("warning(HiSIMHV): The model parameter %s must not be less than %s.\n", "CGBO", "0.0" ); 
       }
      RANGECHECK(pParam->HSMHV_npext,   1.0e16,   1.0e18, "NPEXT") ;
      RANGECHECK(pParam->HSMHV_rd,         0.0,  100.0e-3, "RD") ;
      RANGECHECK(pParam->HSMHV_rd22,      -5.0,      0.0, "RD22") ;
      RANGECHECK(pParam->HSMHV_rd23,       0.0,      2.0, "RD23") ;
      RANGECHECK(pParam->HSMHV_rd24,       0.0,      0.1, "RD24") ;
      RANGECHECK(pParam->HSMHV_rdict1,   -10.0,     10.0, "RDICT1") ;
      RANGECHECK(pParam->HSMHV_rdov13,     0.0,      1.0, "RDOV13") ;
      RANGECHECK(pParam->HSMHV_rdslp1,   -10.0,     10.0, "RDSLP1") ;
      RANGECHECK(pParam->HSMHV_rdvb,       0.0,      2.0, "RDVB") ;
      RANGECHECK(pParam->HSMHV_rdvd,       0.0,      2.0, "RDVD") ;
      MINCHECK(  pParam->HSMHV_rdvg11,     0.0,           "RDVG11") ;
      RANGECHECK(pParam->HSMHV_rs,         0.0,  10.0e-3, "RS") ;
      RANGECHECK(pParam->HSMHV_rth0,       0.0,     10.0, "RTH0") ;
      RANGECHECK(pParam->HSMHV_vover,      0.0,      4.0, "VOVER") ;

      /*-----------------------------------------------------------*
       * Change unit into MKS for instance parameters.
       *-----------------*/

      hereMKS->HSMHV_nsubcdfm  = here->HSMHV_nsubcdfm / C_cm2m_p3 ;
      hereMKS->HSMHV_subld2    = here->HSMHV_subld2   * C_m2cm ;

      pParam->HSMHV_nsubc      = pParam->HSMHV_nsubc  / C_cm2m_p3 ;
      pParam->HSMHV_nsubp      = pParam->HSMHV_nsubp  / C_cm2m_p3 ;
      pParam->HSMHV_nsti       = pParam->HSMHV_nsti   / C_cm2m_p3 ;
      pParam->HSMHV_nover      = pParam->HSMHV_nover  / C_cm2m_p3 ;
      pParam->HSMHV_novers     = pParam->HSMHV_novers / C_cm2m_p3 ;
      pParam->HSMHV_nsubpsti1  = pParam->HSMHV_nsubpsti1 / C_m2cm ;
      pParam->HSMHV_muesti1    = pParam->HSMHV_muesti1 / C_m2cm ;
      pParam->HSMHV_ndep       = pParam->HSMHV_ndep / C_m2cm ;
      pParam->HSMHV_ninv       = pParam->HSMHV_ninv / C_m2cm ;

      pParam->HSMHV_vmax       = pParam->HSMHV_vmax   / C_m2cm ;
      pParam->HSMHV_wfc        = pParam->HSMHV_wfc    * C_m2cm_p2 ;
      pParam->HSMHV_glksd1     = pParam->HSMHV_glksd1 / C_m2cm ;
      pParam->HSMHV_glksd2     = pParam->HSMHV_glksd2 * C_m2cm ;
      pParam->HSMHV_gleak2     = pParam->HSMHV_gleak2 * C_m2cm ;
      pParam->HSMHV_glkb2      = pParam->HSMHV_glkb2  * C_m2cm ;
      pParam->HSMHV_fn2        = pParam->HSMHV_fn2    * C_m2cm ;
      pParam->HSMHV_gidl1      = pParam->HSMHV_gidl1  / C_m2cm_p1o2 ;
      pParam->HSMHV_gidl2      = pParam->HSMHV_gidl2  * C_m2cm ;
      pParam->HSMHV_nfalp      = pParam->HSMHV_nfalp  / C_m2cm ;
      pParam->HSMHV_nftrp      = pParam->HSMHV_nftrp  * C_m2cm_p2 ;

      pParam->HSMHV_npext      = pParam->HSMHV_npext     / C_cm2m_p3 ;
      pParam->HSMHV_rd22       = pParam->HSMHV_rd22      / C_m2cm ;
      pParam->HSMHV_rd23       = pParam->HSMHV_rd23      / C_m2cm ;
      pParam->HSMHV_rd24       = pParam->HSMHV_rd24      / C_m2cm ;
      pParam->HSMHV_rdvd       = pParam->HSMHV_rdvd      / C_m2cm ;
      pParam->HSMHV_rth0       = pParam->HSMHV_rth0      / C_m2cm ;
//    hereMKS->HSMHV_muecb0    = pParam->HSMHV_muecb0 * C_m2cm_p2 ;
//    hereMKS->HSMHV_muecb1    = pParam->HSMHV_muecb1 * C_m2cm_p2 ;
//    hereMKS->HSMHV_muesr1    = pParam->HSMHV_muesr1 * C_m2cm_p2 ;
//    hereMKS->HSMHV_mueph1    = pParam->HSMHV_mueph1 * C_m2cm_p2 ;

      pParam->HSMHV_vfbover    = - pParam->HSMHV_vfbover ; /* For Backward compatibility */

    } /* instance */



    /*-----------------------------------------------------------*
     * Range check of model parameters
     *-----------------*/
     RANGECHECK(model->HSMHV_shemax   ,    300,    600, "SHEMAX");
    if ( model->HSMHV_tox <= 0 ) {
      printf("warning(HiSIMHV): TOX = %e\n ", model->HSMHV_tox);
      printf("warning(HiSIMHV): The model parameter TOX must be positive.\n");
    }
    RANGECHECK(model->HSMHV_xld,        0.0,  50.0e-9, "XLD") ;
    RANGECHECK(model->HSMHV_xwd,   -10.0e-9, 100.0e-9, "XWD") ;
    RANGECHECK(model->HSMHV_xwdc,  -10.0e-9, 100.0e-9, "XWDC") ;
    RANGECHECK(model->HSMHV_rsh,        0.0,      500, "RSH") ;
    RANGECHECK(model->HSMHV_rshg,       0.0,    100.0, "RSHG") ;
    if(model->HSMHV_xqy != 0.0) { MINCHECK  (model->HSMHV_xqy,    10.0e-9,           "XQY") ; }
    MINCHECK  (model->HSMHV_xqy1,       0.0,           "XQY1") ;
    MINCHECK  (model->HSMHV_xqy2,       0.0,           "XQY2") ;
    RANGECHECK(model->HSMHV_vbi,        1.0,      1.2, "VBI") ;
    RANGECHECK(model->HSMHV_parl2,      0.0,  50.0e-9, "PARL2") ;
    RANGECHECK(model->HSMHV_lp,         0.0, 300.0e-9, "LP") ;
    RANGECHECK(model->HSMHV_pgd2,       0.0,      1.5, "PGD2") ;
    RANGECHECK(model->HSMHV_pgd4,       0.0,      3.0, "PGD4") ;
    RANGECHECK(model->HSMHV_mueph0,    0.25,     0.35, "MUEPH0") ;
    RANGECHECK(model->HSMHV_muesr0,     1.8,      2.2, "MUESR0") ;
    RANGECHECK(model->HSMHV_lpext,  1.0e-50,  10.0e-6, "LPEXT") ;
    RANGECHECK(model->HSMHV_scp21,      0.0,      5.0, "SCP21") ;
    RANGECHECK(model->HSMHV_scp22,      0.0,      0.0, "SCP22") ;
    RANGECHECK(model->HSMHV_bs1,        0.0,  50.0e-3, "BS1") ;
    RANGECHECK(model->HSMHV_bs2,        0.5,      1.0, "BS2") ;
    RANGECHECK(model->HSMHV_clm5,       0.0,      2.0, "CLM5") ;
    RANGECHECK(model->HSMHV_clm6,       0.0,     20.0, "CLM6") ;
    MINCHECK  (model->HSMHV_ninvd,      0.0,           "NINVD") ;
    MINCHECK  (model->HSMHV_ninvdw,     0.0,           "NINVDW") ;
    MINCHECK  (model->HSMHV_ninvdwp,    0.0,           "NINVDWP") ;
    MINCHECK  (model->HSMHV_ninvdt1,    0.0,           "NINVDT1") ;
    MINCHECK  (model->HSMHV_ninvdt2,    0.0,           "NINVDT2") ;
    RANGECHECK(model->HSMHV_sub2l,      0.0,      1.0, "SUB2L") ;
    RANGECHECK(model->HSMHV_voverp,     0.0,      2.0, "VOVERP") ;
    RANGECHECK(model->HSMHV_qme1,       0.0, 300.0e-9, "QME1") ;
    RANGECHECK(model->HSMHV_qme2,       0.0,      3.0, "QME2") ;
    RANGECHECK(model->HSMHV_qme3,       0.0,800.0e-12, "QME3") ;
    RANGECHECK(model->HSMHV_glpart1,    0.0,      1.0, "GLPART1") ;
    RANGECHECK(model->HSMHV_tnom,      22.0,     32.0, "TNOM") ;
    RANGECHECK(model->HSMHV_ddltmax,    1.0,     10.0, "DDLTMAX") ;
    RANGECHECK(model->HSMHV_ddltict,   -3.0,     20.0, "DDLTICT") ;
    RANGECHECK(model->HSMHV_ddltslp,    0.0,     20.0, "DDLTSLP") ;
    RANGECHECK(model->HSMHV_mphdfm,    -3.0,      3.0, "MPHDFM") ;
    RANGECHECK(model->HSMHV_cvb,       -0.1,      0.2, "CVB") ;
    RANGECHECK(model->HSMHV_cvbk,      -0.1,      0.2, "CVBK") ;
    RANGECHECK(model->HSMHV_rd20,       0.0,     30.0, "RD20") ;
    RANGECHECK(model->HSMHV_rd21,       0.0,      1.0, "RD21") ;
    RANGECHECK(model->HSMHV_rd22d,      0.0,      2.0, "RD22D") ;
    MINCHECK(  model->HSMHV_rd25,       0.0,           "RD25") ;
    RANGECHECK(model->HSMHV_rdtemp1,  -1e-3,     1e-2, "RDTEMP1") ;
    RANGECHECK(model->HSMHV_rdtemp2,  -1e-5,     1e-5, "RDTEMP2") ;
    RANGECHECK(model->HSMHV_rdvdtemp1,-1e-3,     1e-2, "RDVDTEMP1") ;
    RANGECHECK(model->HSMHV_rdvdtemp2,-1e-5,     1e-5, "RDVDTEMP2") ;
    MINCHECK(  model->HSMHV_rdvg12,     0.0,           "RDVG12") ;
    RANGECHECK(model->HSMHV_rthtemp1,  -1.0,      1.0, "RTHTEMP1") ;
    RANGECHECK(model->HSMHV_rthtemp2,  -1.0,      1.0, "RTHTEMP2") ;
    RANGECHECK(model->HSMHV_rth0w,     -100,      100, "RTH0W") ;
    RANGECHECK(model->HSMHV_rth0wp,     -10,       10, "RTH0WP") ;
    RANGECHECK(model->HSMHV_rth0nf,    -5.0,      5.0, "RTH0NF") ;
    RANGECHECK(model->HSMHV_powrat,     0.0,      1.0, "POWRAT") ;
    RANGECHECK(model->HSMHV_prattemp1, -1.0,      1.0, "PRATTEMP1") ;
    RANGECHECK(model->HSMHV_prattemp2, -1.0,      1.0, "PRATTEMP2") ;
    MINCHECK(  model->HSMHV_xldld,      0.0,           "XLDLD") ;
    MINCHECK(  model->HSMHV_loverld,    0.0,           "LOVERLD") ;
    MINCHECK(  model->HSMHV_lovers,     0.0,           "LOVERS") ;
    MINCHECK(  model->HSMHV_lover,      0.0,           "LOVER") ;
    MINCHECK(  model->HSMHV_ldrift1,    0.0,           "LDRIFT1") ;
    MINCHECK(  model->HSMHV_ldrift1s,   0.0,           "LDRIFT1S") ;
    MINCHECK(  model->HSMHV_ldrift2,    0.0,           "LDRIFT2") ;
    MINCHECK(  model->HSMHV_ldrift2s,   0.0,           "LDRIFT2S") ;
    MINCHECK(  model->HSMHV_ldrift,     0.0,           "LDRIFT") ;
    RANGECHECK(model->HSMHV_rds,       -100,      100, "RDS") ;
    RANGECHECK(model->HSMHV_rdsp,       -10,       10, "RDSP") ;
    RANGECHECK(model->HSMHV_rdvdl,     -100,      100, "RDVDL") ;
    RANGECHECK(model->HSMHV_rdvdlp,     -10,       10, "RDVDLP") ;
    RANGECHECK(model->HSMHV_rdvds,     -100,      100, "RDVDS") ;
    RANGECHECK(model->HSMHV_rdvdsp,     -10,       10, "RDVDSP") ;
    RANGECHECK(model->HSMHV_rd23l,     -100,      100, "RD23L") ;
    RANGECHECK(model->HSMHV_rd23lp,     -10,       10, "RD23LP") ;
    RANGECHECK(model->HSMHV_rd23s,     -100,      100, "RD23S") ;
    RANGECHECK(model->HSMHV_rd23sp,     -10,       10, "RD23SP") ;
    RANGECHECK(model->HSMHV_rdov11,     0.0,       10, "RDOV11") ;
    RANGECHECK(model->HSMHV_rdov12,     0.0,      2.0, "RDOV12") ;
    RANGECHECK(model->HSMHV_rdslp2,   -10.0,     10.0, "RDSLP2") ;
    RANGECHECK(model->HSMHV_rdict2,   -10.0,     10.0, "RDICT2") ;


    /*-----------------------------------------------------------*
     * Change units into MKS.
     *-----------------*/

     modelMKS->HSMHV_vmax      = model->HSMHV_vmax      / C_m2cm ;
     modelMKS->HSMHV_ll        = model->HSMHV_ll        / pow( C_m2cm , model->HSMHV_lln ) ;
     modelMKS->HSMHV_wl        = model->HSMHV_wl        / pow( C_m2cm , model->HSMHV_wln ) ;
     modelMKS->HSMHV_svgsl     = model->HSMHV_svgsl     / pow( C_m2cm , model->HSMHV_svgslp ) ;
     modelMKS->HSMHV_svgsw     = model->HSMHV_svgsw     / pow( C_m2cm , model->HSMHV_svgswp ) ;
     modelMKS->HSMHV_svbsl     = model->HSMHV_svbsl     / pow( C_m2cm , model->HSMHV_svbslp ) ;
     modelMKS->HSMHV_slgl      = model->HSMHV_slgl      / pow( C_m2cm , model->HSMHV_slglp ) ;
     modelMKS->HSMHV_sub1l     = model->HSMHV_sub1l     / pow( C_m2cm , model->HSMHV_sub1lp ) ;
     modelMKS->HSMHV_slg       = model->HSMHV_slg       / C_m2cm ;
     modelMKS->HSMHV_sub2l     = model->HSMHV_sub2l     / C_m2cm ;
     modelMKS->HSMHV_subld2    = model->HSMHV_subld2    * C_m2cm ;
     modelMKS->HSMHV_rdtemp1   = model->HSMHV_rdtemp1   / C_m2cm ;
     modelMKS->HSMHV_rdtemp2   = model->HSMHV_rdtemp2   / C_m2cm ;
     modelMKS->HSMHV_rdvdtemp1 = model->HSMHV_rdvdtemp1 / C_m2cm ;
     modelMKS->HSMHV_rdvdtemp2 = model->HSMHV_rdvdtemp2 / C_m2cm ;
     modelMKS->HSMHV_nsubsub   = model->HSMHV_nsubsub   / C_cm2m_p3 ;
     modelMKS->HSMHV_nsubpsti1 = model->HSMHV_nsubpsti1 / C_m2cm ;
     modelMKS->HSMHV_muesti1   = model->HSMHV_muesti1 / C_m2cm ;
     modelMKS->HSMHV_wfc       = model->HSMHV_wfc       * C_m2cm_p2 ;
     modelMKS->HSMHV_glksd1    = model->HSMHV_glksd1    / C_m2cm ;
     modelMKS->HSMHV_glksd2    = model->HSMHV_glksd2    * C_m2cm ;
     modelMKS->HSMHV_glksd3    = model->HSMHV_glksd3    * C_m2cm ;
     modelMKS->HSMHV_gleak2    = model->HSMHV_gleak2    * C_m2cm ;
     modelMKS->HSMHV_gleak4    = model->HSMHV_gleak4    * C_m2cm ;
     modelMKS->HSMHV_gleak5    = model->HSMHV_gleak5    * C_m2cm ;
     modelMKS->HSMHV_gleak7    = model->HSMHV_gleak7    / C_m2cm_p2 ;
     modelMKS->HSMHV_glkb2     = model->HSMHV_glkb2     * C_m2cm ;
     modelMKS->HSMHV_fn2       = model->HSMHV_fn2       * C_m2cm ;
     modelMKS->HSMHV_gidl1     = model->HSMHV_gidl1     / C_m2cm_p1o2 ;
     modelMKS->HSMHV_gidl2     = model->HSMHV_gidl2     * C_m2cm ;
     modelMKS->HSMHV_nfalp     = model->HSMHV_nfalp     / C_m2cm ;
     modelMKS->HSMHV_nftrp     = model->HSMHV_nftrp     * C_m2cm_p2 ;
     modelMKS->HSMHV_cit       = model->HSMHV_cit       * C_m2cm_p2 ;
     modelMKS->HSMHV_ovslp     = model->HSMHV_ovslp     / C_m2cm ;
     modelMKS->HSMHV_dly3      = model->HSMHV_dly3      / C_m2cm_p2 ;
     modelMKS->HSMHV_cth0      = model->HSMHV_cth0      * C_m2cm ;
//   modelMKS->HSMHV_muecb0    = model->HSMHV_muecb0    * C_cm2m_p2 ;
//   modelMKS->HSMHV_muecb1    = model->HSMHV_muecb1    * C_cm2m_p2 ;
//   modelMKS->HSMHV_muesr1    = model->HSMHV_muesr1    * C_cm2m_p2 ;
//   modelMKS->HSMHV_mueph1    = model->HSMHV_mueph1    * C_cm2m_p2 ;


    /*-----------------------------------------------------------*
     * Change unit into Kelvin.
     *-----------------*/
     model->HSMHV_ktnom =  model->HSMHV_tnom + 273.15 ; /* [C] -> [K] */


  } /* model */

  /* Reset ckt->CKTbypass to 0 */
  if( ckt->CKTbypass == 1 ) {
    fprintf( stderr, "\nwarning(HiSIMHV): The BYPASS option is reset to 0 for reliable simulation.\n");
    ckt->CKTbypass = 0 ;
  }  
  /* check ckt->CKTintegrateMethod */
//  if( ckt->CKTintegrateMethod == TRAPEZOIDAL ) { /* TRAPEZODAL:1 GEAR:2 */
//    fprintf( stderr, "\nwarning(HiSIMHV): Recommend the Gear method for reliable simulation with '.options METHOD=GEAR'.\n");
//  }

  return(OK);
}

int
HSMHVunsetup(
    GENmodel *inModel,
    CKTcircuit *ckt)
{
#ifndef HAS_BATCHSIM
    HSMHVmodel *model;
    HSMHVinstance *here;
 
    for (model = (HSMHVmodel *)inModel; model != NULL;
            model = HSMHVnextModel(model))
    {
        for (here = HSMHVinstances(model); here != NULL;
                here=HSMHVnextInstance(here))
        {
            if (here->HSMHVqbNode > 0)
                CKTdltNNum(ckt, here->HSMHVqbNode);
            here->HSMHVqbNode = 0;

            if (here->HSMHVqiNode > 0)
                CKTdltNNum(ckt, here->HSMHVqiNode);
            here->HSMHVqiNode = 0;

            if (here->HSMHVtempNode > 0 &&
                here->HSMHVtempNode != here->HSMHVtempNodeExt &&
                here->HSMHVtempNode != here->HSMHVsubNodeExt)
                CKTdltNNum(ckt, here->HSMHVtempNode);
            here->HSMHVtempNode = 0;

            here->HSMHVsubNode = 0;

            if (here->HSMHVsbNode > 0
                    && here->HSMHVsbNode != here->HSMHVbNode)
                CKTdltNNum(ckt, here->HSMHVsbNode);
            here->HSMHVsbNode = 0;

            if (here->HSMHVbNodePrime > 0
                    && here->HSMHVbNodePrime != here->HSMHVbNode)
                CKTdltNNum(ckt, here->HSMHVbNodePrime);
            here->HSMHVbNodePrime = 0;

            if (here->HSMHVdbNode > 0
                    && here->HSMHVdbNode != here->HSMHVbNode)
                CKTdltNNum(ckt, here->HSMHVdbNode);
            here->HSMHVdbNode = 0;

            if (here->HSMHVgNodePrime > 0
                    && here->HSMHVgNodePrime != here->HSMHVgNode)
                CKTdltNNum(ckt, here->HSMHVgNodePrime);
            here->HSMHVgNodePrime = 0;

            if (here->HSMHVsNodePrime > 0
                    && here->HSMHVsNodePrime != here->HSMHVsNode)
                CKTdltNNum(ckt, here->HSMHVsNodePrime);
            here->HSMHVsNodePrime = 0;

            if (here->HSMHVdNodePrime > 0
                    && here->HSMHVdNodePrime != here->HSMHVdNode)
                CKTdltNNum(ckt, here->HSMHVdNodePrime);
            here->HSMHVdNodePrime = 0;

        }
    }
#endif
    return OK;
}
