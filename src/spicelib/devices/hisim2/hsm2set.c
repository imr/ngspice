/***********************************************************************

 HiSIM (Hiroshima University STARC IGFET Model)
 Copyright (C) 2012 Hiroshima University & STARC

 VERSION : HiSIM 2.6.1 
 FILE : hsm2set.c

 date : 2012.4.6

 released by 
                Hiroshima University &
                Semiconductor Technology Academic Research Center (STARC)
***********************************************************************/

#include "ngspice/ngspice.h"
#include "ngspice/smpdefs.h"
#include "ngspice/cktdefs.h"
#include "hsm2def.h"
#include "hsm2evalenv.h"
#include "ngspice/const.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int HSM2setup(
     register SMPmatrix *matrix,
     register GENmodel *inModel,
     register CKTcircuit *ckt,
     int *states)
     /* load the HSM2 device structure with those pointers needed later 
      * for fast matrix loading 
      */
{
  register HSM2model *model = (HSM2model*)inModel;
  register HSM2instance *here;
  int error;
  CKTnode *tmp;
  
  /*  loop through all the HSM2 device models */
  for ( ;model != NULL ;model = model->HSM2nextModel ) {
    /* Default value Processing for HSM2 MOSFET Models */
    if ( !model->HSM2_type_Given )
      model->HSM2_type = NMOS ;
    /***/
    if ( !model->HSM2_info_Given ) model->HSM2_info = 0 ;
    /*    if ( !model->HSM2_noise_Given) model->HSM2_noise = 1;*/
    model->HSM2_noise = 1; /* allways noise is set to be 1 */

    if ( !model->HSM2_version_Given) {
        model->HSM2_version = 261; /* default 261 */
	printf("           261 is selected for VERSION. (default) \n");
    } else {
      if (model->HSM2_version != 261) {
	model->HSM2_version = 261; /* default 261 */
	printf("           261 is only available for VERSION. \n");
	printf("           261 is selected for VERSION. (default) \n");
      } else {
	printf("           %d is selected for VERSION \n", (int)model->HSM2_version);
      }
    }

    if ( !model->HSM2_corsrd_Given     ) model->HSM2_corsrd     = 0 ;
    if ( !model->HSM2_corg_Given       ) model->HSM2_corg       = 0 ;
    if ( !model->HSM2_coiprv_Given     ) model->HSM2_coiprv     = 1 ;
    if ( !model->HSM2_copprv_Given     ) model->HSM2_copprv     = 1 ;
    if ( !model->HSM2_coadov_Given     ) model->HSM2_coadov     = 1 ;
    if ( !model->HSM2_coisub_Given     ) model->HSM2_coisub     = 0 ;
    if ( !model->HSM2_coiigs_Given     ) model->HSM2_coiigs     = 0 ;
    if ( !model->HSM2_cogidl_Given     ) model->HSM2_cogidl     = 0 ;
    if ( !model->HSM2_coovlp_Given     ) model->HSM2_coovlp     = 1 ;
    if ( !model->HSM2_coflick_Given    ) model->HSM2_coflick    = 0 ;
    if ( !model->HSM2_coisti_Given     ) model->HSM2_coisti     = 0 ;
    if ( !model->HSM2_conqs_Given      ) model->HSM2_conqs      = 0 ; /* QS (default) */
    if ( !model->HSM2_cothrml_Given    ) model->HSM2_cothrml    = 0 ;
    if ( !model->HSM2_coign_Given      ) model->HSM2_coign      = 0 ; /* induced gate noise */
    if ( !model->HSM2_codfm_Given      ) model->HSM2_codfm      = 0 ; /* DFM */
    if ( !model->HSM2_corbnet_Given    ) model->HSM2_corbnet    = 0 ; 
    else if ( model->HSM2_corbnet != 0 && model->HSM2_corbnet != 1 ) {
      model->HSM2_corbnet = 0;
      printf("warning(HiSIM2): CORBNET has been set to its default value: %d.\n", model->HSM2_corbnet);
    }
    if ( !model->HSM2_corecip_Given    ) model->HSM2_corecip    = 1 ;
    if ( !model->HSM2_coqy_Given       ) model->HSM2_coqy       = 0 ;
    if ( !model->HSM2_coqovsm_Given    ) model->HSM2_coqovsm    = 1 ;


    if ( !model->HSM2_vmax_Given    ) model->HSM2_vmax    = 1.0e7 ;
    if ( !model->HSM2_bgtmp1_Given  ) model->HSM2_bgtmp1  = 90.25e-6 ;
    if ( !model->HSM2_bgtmp2_Given  ) model->HSM2_bgtmp2  = 1.0e-7 ;
    if ( !model->HSM2_eg0_Given     ) model->HSM2_eg0     = 1.1785e0 ;
    if ( !model->HSM2_tox_Given     ) model->HSM2_tox     = 3.0e-9 ;
    if ( !model->HSM2_xld_Given     ) model->HSM2_xld     = 0.0 ;
    if ( !model->HSM2_lover_Given   ) model->HSM2_lover   = 30e-9 ;
    if ( !model->HSM2_ddltmax_Given ) model->HSM2_ddltmax = 10.0 ; /* Vdseff */
    if ( !model->HSM2_ddltslp_Given ) model->HSM2_ddltslp = 0.0 ; /* Vdseff */
    if ( !model->HSM2_ddltict_Given ) model->HSM2_ddltict = 10.0 ; /* Vdseff */
    if ( !model->HSM2_vfbover_Given ) model->HSM2_vfbover = 0.0 ;
    if ( !model->HSM2_nover_Given   ) model->HSM2_nover   = 1E19 ;
    if ( !model->HSM2_xwd_Given     ) model->HSM2_xwd     = 0.0 ;

    if ( !model->HSM2_xl_Given   ) model->HSM2_xl   = 0.0 ;
    if ( !model->HSM2_xw_Given   ) model->HSM2_xw   = 0.0 ;
    if ( !model->HSM2_saref_Given   ) model->HSM2_saref   = 1e-6 ;
    if ( !model->HSM2_sbref_Given   ) model->HSM2_sbref   = 1e-6 ;
    if ( !model->HSM2_ll_Given   ) model->HSM2_ll   = 0.0 ;
    if ( !model->HSM2_lld_Given  ) model->HSM2_lld  = 0.0 ;
    if ( !model->HSM2_lln_Given  ) model->HSM2_lln  = 0.0 ;
    if ( !model->HSM2_wl_Given   ) model->HSM2_wl   = 0.0 ;
    if ( !model->HSM2_wl1_Given  ) model->HSM2_wl1  = 0.0 ;
    if ( !model->HSM2_wl1p_Given ) model->HSM2_wl1p = 1.0 ;
    if ( !model->HSM2_wl2_Given  ) model->HSM2_wl2  = 0.0 ;
    if ( !model->HSM2_wl2p_Given ) model->HSM2_wl2p = 1.0 ;
    if ( !model->HSM2_wld_Given  ) model->HSM2_wld  = 0.0 ;
    if ( !model->HSM2_wln_Given  ) model->HSM2_wln  = 0.0 ;

    if ( !model->HSM2_rsh_Given  ) model->HSM2_rsh  = 0.0 ;
    if ( !model->HSM2_rshg_Given ) model->HSM2_rshg = 0.0 ;

    if ( !model->HSM2_xqy_Given    ) model->HSM2_xqy   = 10e-9 ;
    if ( !model->HSM2_xqy1_Given   ) model->HSM2_xqy1  = 0.0 ;
    if ( !model->HSM2_xqy2_Given   ) model->HSM2_xqy2  = 2.0 ;
    if ( !model->HSM2_qyrat_Given  ) model->HSM2_qyrat = 0.5 ;
    if ( !model->HSM2_rs_Given     ) model->HSM2_rs    = 0.0 ;
    if ( !model->HSM2_rd_Given     ) model->HSM2_rd    = 0.0 ;
    if ( !model->HSM2_vfbc_Given   ) model->HSM2_vfbc  = -1.0 ;
    if ( !model->HSM2_vbi_Given    ) model->HSM2_vbi   = 1.1 ;
    if ( !model->HSM2_nsubc_Given  ) model->HSM2_nsubc = 5.0e17 ;
    if ( !model->HSM2_parl2_Given  ) model->HSM2_parl2 = 10.0e-9 ;
    if ( !model->HSM2_lp_Given     ) model->HSM2_lp    = 15.0e-9 ;
    if ( !model->HSM2_nsubp_Given  ) model->HSM2_nsubp = 1.0e18 ;
    if ( !model->HSM2_nsubpl_Given ) model->HSM2_nsubpl = 0.001 ; /* um */
    if ( !model->HSM2_nsubpfac_Given ) model->HSM2_nsubpfac = 1.0 ;

    if ( !model->HSM2_nsubpw_Given ) model->HSM2_nsubpw = 0.0 ;
    if ( !model->HSM2_nsubpwp_Given ) model->HSM2_nsubpwp = 1.0 ;

    if ( !model->HSM2_scp1_Given  ) model->HSM2_scp1 = 1.0 ;
    if ( !model->HSM2_scp2_Given  ) model->HSM2_scp2 = 0.0 ;
    if ( !model->HSM2_scp3_Given  ) model->HSM2_scp3 = 0.0 ;
    if ( !model->HSM2_sc1_Given   ) model->HSM2_sc1  = 1.0 ;
    if ( !model->HSM2_sc2_Given   ) model->HSM2_sc2  = 0.0 ;
    if ( !model->HSM2_sc3_Given   ) model->HSM2_sc3  = 0.0 ;
    if ( !model->HSM2_sc4_Given   ) model->HSM2_sc4  = 0.0 ;
    if ( !model->HSM2_pgd1_Given  ) model->HSM2_pgd1 = 0.0 ;
    if ( !model->HSM2_pgd2_Given  ) model->HSM2_pgd2 = 0.3 ;
    if ( !model->HSM2_pgd4_Given  ) model->HSM2_pgd4 = 0.0 ;

    if ( !model->HSM2_ndep_Given   ) model->HSM2_ndep   = 1.0 ;
    if ( !model->HSM2_ndepl_Given  ) model->HSM2_ndepl  = 0.0 ;
    if ( !model->HSM2_ndeplp_Given ) model->HSM2_ndeplp = 1.0 ;
    if ( !model->HSM2_ndepw_Given  ) model->HSM2_ndepw  = 0.0 ;
    if ( !model->HSM2_ndepwp_Given ) model->HSM2_ndepwp = 1.0 ;
    if ( !model->HSM2_ninv_Given   ) model->HSM2_ninv   = 0.5 ;
    if ( !model->HSM2_ninvd_Given  ) model->HSM2_ninvd  = 0.0 ;
    if ( !model->HSM2_muecb0_Given ) model->HSM2_muecb0 = 1.0e3 ;
    if ( !model->HSM2_muecb1_Given ) model->HSM2_muecb1 = 100.0 ;
    if ( !model->HSM2_mueph0_Given ) model->HSM2_mueph0 = 300.0e-3 ;
    if ( !model->HSM2_mueph1_Given ) {
      if (model->HSM2_type == NMOS) model->HSM2_mueph1 = 25.0e3 ;
      else model->HSM2_mueph1 = 9.0e3 ;
    }
    if ( !model->HSM2_muephw_Given ) model->HSM2_muephw = 0.0 ;
    if ( !model->HSM2_muepwp_Given ) model->HSM2_muepwp = 1.0 ;
    if ( !model->HSM2_muepwd_Given ) model->HSM2_muepwd = 0.0 ;
    if ( !model->HSM2_muephl_Given ) model->HSM2_muephl = 0.0 ;
    if ( !model->HSM2_mueplp_Given ) model->HSM2_mueplp = 1.0 ;
    if ( !model->HSM2_muepld_Given ) model->HSM2_muepld = 0.0 ;
    if ( !model->HSM2_muephs_Given ) model->HSM2_muephs = 0.0 ;
    if ( !model->HSM2_muepsp_Given ) model->HSM2_muepsp = 1.0 ;

    if ( !model->HSM2_vtmp_Given  ) model->HSM2_vtmp  = 0.0 ;

    if ( !model->HSM2_wvth0_Given ) model->HSM2_wvth0 = 0.0 ;

    if ( !model->HSM2_muesr0_Given ) model->HSM2_muesr0 = 2.0 ;
    if ( !model->HSM2_muesr1_Given ) model->HSM2_muesr1 = 1.0e15 ;
    if ( !model->HSM2_muesrl_Given ) model->HSM2_muesrl = 0.0 ;
    if ( !model->HSM2_muesrw_Given ) model->HSM2_muesrw = 0.0 ;
    if ( !model->HSM2_mueswp_Given ) model->HSM2_mueswp = 1.0 ;
    if ( !model->HSM2_mueslp_Given ) model->HSM2_mueslp = 1.0 ;

    if ( !model->HSM2_muetmp_Given  ) model->HSM2_muetmp = 1.5 ;

    if ( !model->HSM2_bb_Given ) {
      if (model->HSM2_type == NMOS) model->HSM2_bb = 2.0 ;
      else model->HSM2_bb = 1.0 ;
    }

    if ( !model->HSM2_sub1_Given  ) model->HSM2_sub1  = 10e0 ;
    if ( !model->HSM2_sub2_Given  ) model->HSM2_sub2  = 25e0 ;
    if ( !model->HSM2_svgs_Given  ) model->HSM2_svgs  = 0.8e0 ;
    if ( !model->HSM2_svbs_Given  ) model->HSM2_svbs  = 0.5e0 ;
    if ( !model->HSM2_svbsl_Given ) model->HSM2_svbsl = 0e0 ;
    if ( !model->HSM2_svds_Given  ) model->HSM2_svds  = 0.8e0 ;
    if ( !model->HSM2_slg_Given   ) model->HSM2_slg   = 30e-9 ;
    if ( !model->HSM2_sub1l_Given ) model->HSM2_sub1l = 2.5e-3 ;
    if ( !model->HSM2_sub2l_Given ) model->HSM2_sub2l = 2e-6 ;

    if ( !model->HSM2_svgsl_Given  ) model->HSM2_svgsl  = 0.0 ;
    if ( !model->HSM2_svgslp_Given ) model->HSM2_svgslp = 1.0 ;
    if ( !model->HSM2_svgswp_Given ) model->HSM2_svgswp = 1.0 ;
    if ( !model->HSM2_svgsw_Given  ) model->HSM2_svgsw  = 0.0 ;
    if ( !model->HSM2_svbslp_Given ) model->HSM2_svbslp = 1.0 ;
    if ( !model->HSM2_slgl_Given   ) model->HSM2_slgl   = 0.0 ;
    if ( !model->HSM2_slglp_Given  ) model->HSM2_slglp  = 1.0 ;
    if ( !model->HSM2_sub1lp_Given ) model->HSM2_sub1lp = 1.0 ; 

    if ( !model->HSM2_nsti_Given      ) model->HSM2_nsti      = 5.0e17 ;
    if ( !model->HSM2_wsti_Given      ) model->HSM2_wsti      = 0.0 ;
    if ( !model->HSM2_wstil_Given     ) model->HSM2_wstil     = 0.0 ;
    if ( !model->HSM2_wstilp_Given    ) model->HSM2_wstilp    = 1.0 ;
    if ( !model->HSM2_wstiw_Given     ) model->HSM2_wstiw     = 0.0 ;
    if ( !model->HSM2_wstiwp_Given    ) model->HSM2_wstiwp    = 1.0 ;
    if ( !model->HSM2_scsti1_Given    ) model->HSM2_scsti1    = 0.0 ;
    if ( !model->HSM2_scsti2_Given    ) model->HSM2_scsti2    = 0.0 ;
    if ( !model->HSM2_vthsti_Given    ) model->HSM2_vthsti    = 0.0 ;
    if ( !model->HSM2_vdsti_Given     ) model->HSM2_vdsti     = 0.0 ;
    if ( !model->HSM2_muesti1_Given   ) model->HSM2_muesti1   = 0.0 ;
    if ( !model->HSM2_muesti2_Given   ) model->HSM2_muesti2   = 0.0 ;
    if ( !model->HSM2_muesti3_Given   ) model->HSM2_muesti3   = 1.0 ;
    if ( !model->HSM2_nsubpsti1_Given ) model->HSM2_nsubpsti1 = 0.0 ;
    if ( !model->HSM2_nsubpsti2_Given ) model->HSM2_nsubpsti2 = 0.0 ;
    if ( !model->HSM2_nsubpsti3_Given ) model->HSM2_nsubpsti3 = 1.0 ;

    if ( !model->HSM2_lpext_Given ) model->HSM2_lpext = 1.0e-50 ;
    if ( !model->HSM2_npext_Given ) model->HSM2_npext = 5.0e17 ;
    if ( !model->HSM2_npextw_Given ) model->HSM2_npextw = 0.0 ;
    if ( !model->HSM2_npextwp_Given ) model->HSM2_npextwp = 1.0 ;
    if ( !model->HSM2_scp21_Given ) model->HSM2_scp21 = 0.0 ;
    if ( !model->HSM2_scp22_Given ) model->HSM2_scp22 = 0.0 ;
    if ( !model->HSM2_bs1_Given   ) model->HSM2_bs1   = 0.0 ;
    if ( !model->HSM2_bs2_Given   ) model->HSM2_bs2   = 0.9 ;

    if ( !model->HSM2_tpoly_Given ) model->HSM2_tpoly = 200e-9 ;
    if ( !model->HSM2_cgbo_Given  ) model->HSM2_cgbo  = 0.0 ;
    if ( !model->HSM2_js0_Given   ) model->HSM2_js0   = 0.5e-6 ;
    if ( !model->HSM2_js0sw_Given ) model->HSM2_js0sw = 0.0 ;
    if ( !model->HSM2_nj_Given    ) model->HSM2_nj    = 1.0 ;
    if ( !model->HSM2_njsw_Given  ) model->HSM2_njsw  = 1.0 ;
    if ( !model->HSM2_xti_Given   ) model->HSM2_xti   = 2.0 ;
    if ( !model->HSM2_cj_Given    ) model->HSM2_cj    = 5.0e-04 ;
    if ( !model->HSM2_cjsw_Given  ) model->HSM2_cjsw  = 5.0e-10 ;
    if ( !model->HSM2_cjswg_Given ) model->HSM2_cjswg = 5.0e-10 ;
    if ( !model->HSM2_mj_Given    ) model->HSM2_mj    = 0.5e0 ;
    if ( !model->HSM2_mjsw_Given  ) model->HSM2_mjsw  = 0.33e0 ;
    if ( !model->HSM2_mjswg_Given ) model->HSM2_mjswg = 0.33e0 ;
    if ( !model->HSM2_pb_Given    ) model->HSM2_pb    = 1.0e0 ;
    if ( !model->HSM2_pbsw_Given  ) model->HSM2_pbsw  = 1.0e0 ;
    if ( !model->HSM2_pbswg_Given ) model->HSM2_pbswg = 1.0e0 ;

    if ( !model->HSM2_tcjbd_Given    ) model->HSM2_tcjbd    = 0.0 ; 
    if ( !model->HSM2_tcjbs_Given    ) model->HSM2_tcjbs    = 0.0 ; 
    if ( !model->HSM2_tcjbdsw_Given  ) model->HSM2_tcjbdsw  = 0.0 ; 
    if ( !model->HSM2_tcjbssw_Given  ) model->HSM2_tcjbssw  = 0.0 ; 
    if ( !model->HSM2_tcjbdswg_Given ) model->HSM2_tcjbdswg = 0.0 ; 
    if ( !model->HSM2_tcjbsswg_Given ) model->HSM2_tcjbsswg = 0.0 ; 

    if ( !model->HSM2_xti2_Given  ) model->HSM2_xti2  = 0.0e0 ;
    if ( !model->HSM2_cisb_Given  ) model->HSM2_cisb  = 0.0e0 ;
    if ( !model->HSM2_cvb_Given   ) model->HSM2_cvb   = 0.0e0 ;
    if ( !model->HSM2_ctemp_Given ) model->HSM2_ctemp = 0.0e0 ;
    if ( !model->HSM2_cisbk_Given ) model->HSM2_cisbk = 0.0e0 ;
    if ( !model->HSM2_cvbk_Given  ) model->HSM2_cvbk  = 0.0e0 ;
    if ( !model->HSM2_divx_Given  ) model->HSM2_divx  = 0.0e0 ;

    if ( !model->HSM2_clm1_Given  ) model->HSM2_clm1 = 700.0e-3 ;
    if ( !model->HSM2_clm2_Given  ) model->HSM2_clm2 = 2.0 ;
    if ( !model->HSM2_clm3_Given  ) model->HSM2_clm3 = 1.0 ;
    if ( !model->HSM2_clm5_Given   ) model->HSM2_clm5   = 1.0 ;
    if ( !model->HSM2_clm6_Given   ) model->HSM2_clm6   = 0.0 ;
    if ( !model->HSM2_vover_Given  ) model->HSM2_vover  = 0.3 ;
    if ( !model->HSM2_voverp_Given ) model->HSM2_voverp = 0.3 ;
    if ( !model->HSM2_wfc_Given    ) model->HSM2_wfc    = 0.0 ;
    if ( !model->HSM2_nsubcw_Given ) model->HSM2_nsubcw = 0.0 ;
    if ( !model->HSM2_nsubcwp_Given ) model->HSM2_nsubcwp = 1.0 ;
    if ( !model->HSM2_nsubcmax_Given ) model->HSM2_nsubcmax = 5e18 ;

    if ( !model->HSM2_qme1_Given   ) model->HSM2_qme1   = 0.0 ;
    if ( !model->HSM2_qme2_Given   ) model->HSM2_qme2   = 0.0 ;
    if ( !model->HSM2_qme3_Given   ) model->HSM2_qme3   = 0.0 ;

    if ( !model->HSM2_vovers_Given  ) model->HSM2_vovers  = 0.0 ;
    if ( !model->HSM2_voversp_Given ) model->HSM2_voversp = 0.0 ;

    if ( !model->HSM2_gidl1_Given ) model->HSM2_gidl1 = 2e0 ;
    if ( !model->HSM2_gidl2_Given ) model->HSM2_gidl2 = 3e7 ;
    if ( !model->HSM2_gidl3_Given ) model->HSM2_gidl3 = 0.9e0 ;
    if ( !model->HSM2_gidl4_Given ) model->HSM2_gidl4 = 0.0 ;
    if ( !model->HSM2_gidl5_Given ) model->HSM2_gidl5 = 0.2e0 ;

    if ( !model->HSM2_gleak1_Given ) model->HSM2_gleak1 = 50e0 ;
    if ( !model->HSM2_gleak2_Given ) model->HSM2_gleak2 = 10e6 ;
    if ( !model->HSM2_gleak3_Given ) model->HSM2_gleak3 = 60e-3 ;
    if ( !model->HSM2_gleak4_Given ) model->HSM2_gleak4 = 4e0 ;
    if ( !model->HSM2_gleak5_Given ) model->HSM2_gleak5 = 7.5e3 ;
    if ( !model->HSM2_gleak6_Given ) model->HSM2_gleak6 = 250e-3 ;
    if ( !model->HSM2_gleak7_Given ) model->HSM2_gleak7 = 1e-6 ;

    if ( !model->HSM2_glksd1_Given  ) model->HSM2_glksd1  = 1.0e-15 ;
    if ( !model->HSM2_glksd2_Given  ) model->HSM2_glksd2  = 5e6 ;
    if ( !model->HSM2_glksd3_Given  ) model->HSM2_glksd3  = -5e6 ;
    if ( !model->HSM2_glkb1_Given   ) model->HSM2_glkb1   = 5e-16 ;
    if ( !model->HSM2_glkb2_Given   ) model->HSM2_glkb2   = 1e0 ;
    if ( !model->HSM2_glkb3_Given   ) model->HSM2_glkb3   = 0e0 ;
    if ( !model->HSM2_egig_Given    ) model->HSM2_egig    = 0e0 ;
    if ( !model->HSM2_igtemp2_Given ) model->HSM2_igtemp2 = 0e0 ;
    if ( !model->HSM2_igtemp3_Given ) model->HSM2_igtemp3 = 0e0 ;
    if ( !model->HSM2_vzadd0_Given  ) model->HSM2_vzadd0  = 20.0e-3 ;
    if ( !model->HSM2_pzadd0_Given  ) model->HSM2_pzadd0  = 20.0e-3 ;
    if ( !model->HSM2_nftrp_Given   ) model->HSM2_nftrp   = 10e9 ;
    if ( !model->HSM2_nfalp_Given   ) model->HSM2_nfalp   = 1.0e-19 ;
    if ( !model->HSM2_falph_Given   ) model->HSM2_falph   = 1.0 ;
    if ( !model->HSM2_cit_Given     ) model->HSM2_cit     = 0e0 ;

    if ( !model->HSM2_kappa_Given ) model->HSM2_kappa = 3.90e0 ;
    if ( !model->HSM2_cgso_Given  ) model->HSM2_cgso  = 0.0 ;
    if ( !model->HSM2_cgdo_Given  ) model->HSM2_cgdo  = 0.0 ;


    if ( !model->HSM2_vdiffj_Given ) model->HSM2_vdiffj = 0.6e-3 ;
    if ( !model->HSM2_dly1_Given   ) model->HSM2_dly1   = 100.0e-12 ;
    if ( !model->HSM2_dly2_Given   ) model->HSM2_dly2   = 0.7e0 ;
    if ( !model->HSM2_dly3_Given   ) model->HSM2_dly3   = 0.8e-6 ;
    if ( !model->HSM2_tnom_Given   ) model->HSM2_tnom   = 27.0 ; /* [C] */

    if ( !model->HSM2_ovslp_Given  ) model->HSM2_ovslp = 2.1e-7 ;
    if ( !model->HSM2_ovmag_Given  ) model->HSM2_ovmag = 0.6e0 ;

    if ( !model->HSM2_gbmin_Given  ) model->HSM2_gbmin = 1.0e-12; /* in mho */
    if ( !model->HSM2_rbpb_Given   ) model->HSM2_rbpb  = 50.0e0 ;
    if ( !model->HSM2_rbpd_Given   ) model->HSM2_rbpd  = 50.0e0 ;
    if ( !model->HSM2_rbps_Given   ) model->HSM2_rbps  = 50.0e0 ;
    if ( !model->HSM2_rbdb_Given   ) model->HSM2_rbdb  = 50.0e0 ;
    if ( !model->HSM2_rbsb_Given   ) model->HSM2_rbsb  = 50.0e0 ;

    if ( !model->HSM2_ibpc1_Given  ) model->HSM2_ibpc1 = 0.0 ;
    if ( !model->HSM2_ibpc2_Given  ) model->HSM2_ibpc2 = 0.0 ;

    if ( !model->HSM2_mphdfm_Given ) model->HSM2_mphdfm = -0.3 ;


    if ( !model->HSM2_ptl_Given ) model->HSM2_ptl = 0.0 ;
    if ( !model->HSM2_ptp_Given ) model->HSM2_ptp = 3.5 ;
    if ( !model->HSM2_pt2_Given ) model->HSM2_pt2 = 0.0 ;
    if ( !model->HSM2_ptlp_Given ) model->HSM2_ptlp = 1.0 ;
    if ( !model->HSM2_gdl_Given ) model->HSM2_gdl = 0.0 ;
    if ( !model->HSM2_gdlp_Given ) model->HSM2_gdlp = 0.0 ;

    if ( !model->HSM2_gdld_Given ) model->HSM2_gdld = 0.0 ;
    if ( !model->HSM2_pt4_Given ) model->HSM2_pt4 = 0.0 ;
    if ( !model->HSM2_pt4p_Given ) model->HSM2_pt4p = 1.0 ;
    if ( !model->HSM2_muephl2_Given ) model->HSM2_muephl2 = 0.0 ;
    if ( !model->HSM2_mueplp2_Given ) model->HSM2_mueplp2 = 1.0 ;
    if ( !model->HSM2_nsubcw2_Given ) model->HSM2_nsubcw2 = 0.0 ;
    if ( !model->HSM2_nsubcwp2_Given ) model->HSM2_nsubcwp2 = 1.0 ;
    if ( !model->HSM2_muephw2_Given ) model->HSM2_muephw2 = 0.0 ;
    if ( !model->HSM2_muepwp2_Given ) model->HSM2_muepwp2 = 1.0 ;
    /* WPE set default Model parameter value */
    if ( !model->HSM2_web_Given ) model->HSM2_web = 0.0 ;
    if ( !model->HSM2_wec_Given ) model->HSM2_wec = 0.0 ;
    if ( !model->HSM2_nsubcwpe_Given ) model->HSM2_nsubcwpe = 0.0 ; 
    if ( !model->HSM2_npextwpe_Given ) model->HSM2_npextwpe = 0.0 ; 
    if ( !model->HSM2_nsubpwpe_Given ) model->HSM2_nsubpwpe = 0.0 ; 
    if ( !model->HSM2_Vgsmin_Given ) model->HSM2_Vgsmin = -5.0 * model->HSM2_type ;
    if ( !model->HSM2_sc3Vbs_Given ) model->HSM2_sc3Vbs =  0.0 ;
    if ( !model->HSM2_byptol_Given ) model->HSM2_byptol =  0.0 ;
    if ( !model->HSM2_muecb0lp_Given ) model->HSM2_muecb0lp = 0.0;
    if ( !model->HSM2_muecb1lp_Given ) model->HSM2_muecb1lp = 0.0;

    /* binning parameters */
    if ( !model->HSM2_lmin_Given ) model->HSM2_lmin = 0.0 ;
    if ( !model->HSM2_lmax_Given ) model->HSM2_lmax = 1.0 ;
    if ( !model->HSM2_wmin_Given ) model->HSM2_wmin = 0.0 ;
    if ( !model->HSM2_wmax_Given ) model->HSM2_wmax = 1.0 ;
    if ( !model->HSM2_lbinn_Given ) model->HSM2_lbinn = 1.0 ;
    if ( !model->HSM2_wbinn_Given ) model->HSM2_wbinn = 1.0 ;

    /* Length dependence */
    if ( !model->HSM2_lvmax_Given ) model->HSM2_lvmax = 0.0 ;
    if ( !model->HSM2_lbgtmp1_Given ) model->HSM2_lbgtmp1 = 0.0 ;
    if ( !model->HSM2_lbgtmp2_Given ) model->HSM2_lbgtmp2 = 0.0 ;
    if ( !model->HSM2_leg0_Given ) model->HSM2_leg0 = 0.0 ;
    if ( !model->HSM2_llover_Given ) model->HSM2_llover = 0.0 ;
    if ( !model->HSM2_lvfbover_Given ) model->HSM2_lvfbover = 0.0 ;
    if ( !model->HSM2_lnover_Given ) model->HSM2_lnover = 0.0 ;
    if ( !model->HSM2_lwl2_Given ) model->HSM2_lwl2 = 0.0 ;
    if ( !model->HSM2_lvfbc_Given ) model->HSM2_lvfbc = 0.0 ;
    if ( !model->HSM2_lnsubc_Given ) model->HSM2_lnsubc = 0.0 ;
    if ( !model->HSM2_lnsubp_Given ) model->HSM2_lnsubp = 0.0 ;
    if ( !model->HSM2_lscp1_Given ) model->HSM2_lscp1 = 0.0 ;
    if ( !model->HSM2_lscp2_Given ) model->HSM2_lscp2 = 0.0 ;
    if ( !model->HSM2_lscp3_Given ) model->HSM2_lscp3 = 0.0 ;
    if ( !model->HSM2_lsc1_Given ) model->HSM2_lsc1 = 0.0 ;
    if ( !model->HSM2_lsc2_Given ) model->HSM2_lsc2 = 0.0 ;
    if ( !model->HSM2_lsc3_Given ) model->HSM2_lsc3 = 0.0 ;
    if ( !model->HSM2_lsc4_Given ) model->HSM2_lsc4 = 0.0 ;
    if ( !model->HSM2_lpgd1_Given ) model->HSM2_lpgd1 = 0.0 ;
    if ( !model->HSM2_lndep_Given ) model->HSM2_lndep = 0.0 ;
    if ( !model->HSM2_lninv_Given ) model->HSM2_lninv = 0.0 ;
    if ( !model->HSM2_lmuecb0_Given ) model->HSM2_lmuecb0 = 0.0 ;
    if ( !model->HSM2_lmuecb1_Given ) model->HSM2_lmuecb1 = 0.0 ;
    if ( !model->HSM2_lmueph1_Given ) model->HSM2_lmueph1 = 0.0 ;
    if ( !model->HSM2_lvtmp_Given ) model->HSM2_lvtmp = 0.0 ;
    if ( !model->HSM2_lwvth0_Given ) model->HSM2_lwvth0 = 0.0 ;
    if ( !model->HSM2_lmuesr1_Given ) model->HSM2_lmuesr1 = 0.0 ;
    if ( !model->HSM2_lmuetmp_Given ) model->HSM2_lmuetmp = 0.0 ;
    if ( !model->HSM2_lsub1_Given ) model->HSM2_lsub1 = 0.0 ;
    if ( !model->HSM2_lsub2_Given ) model->HSM2_lsub2 = 0.0 ;
    if ( !model->HSM2_lsvds_Given ) model->HSM2_lsvds = 0.0 ;
    if ( !model->HSM2_lsvbs_Given ) model->HSM2_lsvbs = 0.0 ;
    if ( !model->HSM2_lsvgs_Given ) model->HSM2_lsvgs = 0.0 ;
    if ( !model->HSM2_lnsti_Given ) model->HSM2_lnsti = 0.0 ;
    if ( !model->HSM2_lwsti_Given ) model->HSM2_lwsti = 0.0 ;
    if ( !model->HSM2_lscsti1_Given ) model->HSM2_lscsti1 = 0.0 ;
    if ( !model->HSM2_lscsti2_Given ) model->HSM2_lscsti2 = 0.0 ;
    if ( !model->HSM2_lvthsti_Given ) model->HSM2_lvthsti = 0.0 ;
    if ( !model->HSM2_lmuesti1_Given ) model->HSM2_lmuesti1 = 0.0 ;
    if ( !model->HSM2_lmuesti2_Given ) model->HSM2_lmuesti2 = 0.0 ;
    if ( !model->HSM2_lmuesti3_Given ) model->HSM2_lmuesti3 = 0.0 ;
    if ( !model->HSM2_lnsubpsti1_Given ) model->HSM2_lnsubpsti1 = 0.0 ;
    if ( !model->HSM2_lnsubpsti2_Given ) model->HSM2_lnsubpsti2 = 0.0 ;
    if ( !model->HSM2_lnsubpsti3_Given ) model->HSM2_lnsubpsti3 = 0.0 ;
    if ( !model->HSM2_lcgso_Given ) model->HSM2_lcgso = 0.0 ;
    if ( !model->HSM2_lcgdo_Given ) model->HSM2_lcgdo = 0.0 ;
    if ( !model->HSM2_ljs0_Given ) model->HSM2_ljs0 = 0.0 ;
    if ( !model->HSM2_ljs0sw_Given ) model->HSM2_ljs0sw = 0.0 ;
    if ( !model->HSM2_lnj_Given ) model->HSM2_lnj = 0.0 ;
    if ( !model->HSM2_lcisbk_Given ) model->HSM2_lcisbk = 0.0 ;
    if ( !model->HSM2_lclm1_Given ) model->HSM2_lclm1 = 0.0 ;
    if ( !model->HSM2_lclm2_Given ) model->HSM2_lclm2 = 0.0 ;
    if ( !model->HSM2_lclm3_Given ) model->HSM2_lclm3 = 0.0 ;
    if ( !model->HSM2_lwfc_Given ) model->HSM2_lwfc = 0.0 ;
    if ( !model->HSM2_lgidl1_Given ) model->HSM2_lgidl1 = 0.0 ;
    if ( !model->HSM2_lgidl2_Given ) model->HSM2_lgidl2 = 0.0 ;
    if ( !model->HSM2_lgleak1_Given ) model->HSM2_lgleak1 = 0.0 ;
    if ( !model->HSM2_lgleak2_Given ) model->HSM2_lgleak2 = 0.0 ;
    if ( !model->HSM2_lgleak3_Given ) model->HSM2_lgleak3 = 0.0 ;
    if ( !model->HSM2_lgleak6_Given ) model->HSM2_lgleak6 = 0.0 ;
    if ( !model->HSM2_lglksd1_Given ) model->HSM2_lglksd1 = 0.0 ;
    if ( !model->HSM2_lglksd2_Given ) model->HSM2_lglksd2 = 0.0 ;
    if ( !model->HSM2_lglkb1_Given ) model->HSM2_lglkb1 = 0.0 ;
    if ( !model->HSM2_lglkb2_Given ) model->HSM2_lglkb2 = 0.0 ;
    if ( !model->HSM2_lnftrp_Given ) model->HSM2_lnftrp = 0.0 ;
    if ( !model->HSM2_lnfalp_Given ) model->HSM2_lnfalp = 0.0 ;
    if ( !model->HSM2_lvdiffj_Given ) model->HSM2_lvdiffj = 0.0 ;
    if ( !model->HSM2_libpc1_Given ) model->HSM2_libpc1 = 0.0 ;
    if ( !model->HSM2_libpc2_Given ) model->HSM2_libpc2 = 0.0 ;

    /* Width dependence */
    if ( !model->HSM2_wvmax_Given ) model->HSM2_wvmax = 0.0 ;
    if ( !model->HSM2_wbgtmp1_Given ) model->HSM2_wbgtmp1 = 0.0 ;
    if ( !model->HSM2_wbgtmp2_Given ) model->HSM2_wbgtmp2 = 0.0 ;
    if ( !model->HSM2_weg0_Given ) model->HSM2_weg0 = 0.0 ;
    if ( !model->HSM2_wlover_Given ) model->HSM2_wlover = 0.0 ;
    if ( !model->HSM2_wvfbover_Given ) model->HSM2_wvfbover = 0.0 ;
    if ( !model->HSM2_wnover_Given ) model->HSM2_wnover = 0.0 ;
    if ( !model->HSM2_wwl2_Given ) model->HSM2_wwl2 = 0.0 ;
    if ( !model->HSM2_wvfbc_Given ) model->HSM2_wvfbc = 0.0 ;
    if ( !model->HSM2_wnsubc_Given ) model->HSM2_wnsubc = 0.0 ;
    if ( !model->HSM2_wnsubp_Given ) model->HSM2_wnsubp = 0.0 ;
    if ( !model->HSM2_wscp1_Given ) model->HSM2_wscp1 = 0.0 ;
    if ( !model->HSM2_wscp2_Given ) model->HSM2_wscp2 = 0.0 ;
    if ( !model->HSM2_wscp3_Given ) model->HSM2_wscp3 = 0.0 ;
    if ( !model->HSM2_wsc1_Given ) model->HSM2_wsc1 = 0.0 ;
    if ( !model->HSM2_wsc2_Given ) model->HSM2_wsc2 = 0.0 ;
    if ( !model->HSM2_wsc3_Given ) model->HSM2_wsc3 = 0.0 ;
    if ( !model->HSM2_wsc4_Given ) model->HSM2_wsc4 = 0.0 ;
    if ( !model->HSM2_wpgd1_Given ) model->HSM2_wpgd1 = 0.0 ;
    if ( !model->HSM2_wndep_Given ) model->HSM2_wndep = 0.0 ;
    if ( !model->HSM2_wninv_Given ) model->HSM2_wninv = 0.0 ;
    if ( !model->HSM2_wmuecb0_Given ) model->HSM2_wmuecb0 = 0.0 ;
    if ( !model->HSM2_wmuecb1_Given ) model->HSM2_wmuecb1 = 0.0 ;
    if ( !model->HSM2_wmueph1_Given ) model->HSM2_wmueph1 = 0.0 ;
    if ( !model->HSM2_wvtmp_Given ) model->HSM2_wvtmp = 0.0 ;
    if ( !model->HSM2_wwvth0_Given ) model->HSM2_wwvth0 = 0.0 ;
    if ( !model->HSM2_wmuesr1_Given ) model->HSM2_wmuesr1 = 0.0 ;
    if ( !model->HSM2_wmuetmp_Given ) model->HSM2_wmuetmp = 0.0 ;
    if ( !model->HSM2_wsub1_Given ) model->HSM2_wsub1 = 0.0 ;
    if ( !model->HSM2_wsub2_Given ) model->HSM2_wsub2 = 0.0 ;
    if ( !model->HSM2_wsvds_Given ) model->HSM2_wsvds = 0.0 ;
    if ( !model->HSM2_wsvbs_Given ) model->HSM2_wsvbs = 0.0 ;
    if ( !model->HSM2_wsvgs_Given ) model->HSM2_wsvgs = 0.0 ;
    if ( !model->HSM2_wnsti_Given ) model->HSM2_wnsti = 0.0 ;
    if ( !model->HSM2_wwsti_Given ) model->HSM2_wwsti = 0.0 ;
    if ( !model->HSM2_wscsti1_Given ) model->HSM2_wscsti1 = 0.0 ;
    if ( !model->HSM2_wscsti2_Given ) model->HSM2_wscsti2 = 0.0 ;
    if ( !model->HSM2_wvthsti_Given ) model->HSM2_wvthsti = 0.0 ;
    if ( !model->HSM2_wmuesti1_Given ) model->HSM2_wmuesti1 = 0.0 ;
    if ( !model->HSM2_wmuesti2_Given ) model->HSM2_wmuesti2 = 0.0 ;
    if ( !model->HSM2_wmuesti3_Given ) model->HSM2_wmuesti3 = 0.0 ;
    if ( !model->HSM2_wnsubpsti1_Given ) model->HSM2_wnsubpsti1 = 0.0 ;
    if ( !model->HSM2_wnsubpsti2_Given ) model->HSM2_wnsubpsti2 = 0.0 ;
    if ( !model->HSM2_wnsubpsti3_Given ) model->HSM2_wnsubpsti3 = 0.0 ;
    if ( !model->HSM2_wcgso_Given ) model->HSM2_wcgso = 0.0 ;
    if ( !model->HSM2_wcgdo_Given ) model->HSM2_wcgdo = 0.0 ;
    if ( !model->HSM2_wjs0_Given ) model->HSM2_wjs0 = 0.0 ;
    if ( !model->HSM2_wjs0sw_Given ) model->HSM2_wjs0sw = 0.0 ;
    if ( !model->HSM2_wnj_Given ) model->HSM2_wnj = 0.0 ;
    if ( !model->HSM2_wcisbk_Given ) model->HSM2_wcisbk = 0.0 ;
    if ( !model->HSM2_wclm1_Given ) model->HSM2_wclm1 = 0.0 ;
    if ( !model->HSM2_wclm2_Given ) model->HSM2_wclm2 = 0.0 ;
    if ( !model->HSM2_wclm3_Given ) model->HSM2_wclm3 = 0.0 ;
    if ( !model->HSM2_wwfc_Given ) model->HSM2_wwfc = 0.0 ;
    if ( !model->HSM2_wgidl1_Given ) model->HSM2_wgidl1 = 0.0 ;
    if ( !model->HSM2_wgidl2_Given ) model->HSM2_wgidl2 = 0.0 ;
    if ( !model->HSM2_wgleak1_Given ) model->HSM2_wgleak1 = 0.0 ;
    if ( !model->HSM2_wgleak2_Given ) model->HSM2_wgleak2 = 0.0 ;
    if ( !model->HSM2_wgleak3_Given ) model->HSM2_wgleak3 = 0.0 ;
    if ( !model->HSM2_wgleak6_Given ) model->HSM2_wgleak6 = 0.0 ;
    if ( !model->HSM2_wglksd1_Given ) model->HSM2_wglksd1 = 0.0 ;
    if ( !model->HSM2_wglksd2_Given ) model->HSM2_wglksd2 = 0.0 ;
    if ( !model->HSM2_wglkb1_Given ) model->HSM2_wglkb1 = 0.0 ;
    if ( !model->HSM2_wglkb2_Given ) model->HSM2_wglkb2 = 0.0 ;
    if ( !model->HSM2_wnftrp_Given ) model->HSM2_wnftrp = 0.0 ;
    if ( !model->HSM2_wnfalp_Given ) model->HSM2_wnfalp = 0.0 ;
    if ( !model->HSM2_wvdiffj_Given ) model->HSM2_wvdiffj = 0.0 ;
    if ( !model->HSM2_wibpc1_Given ) model->HSM2_wibpc1 = 0.0 ;
    if ( !model->HSM2_wibpc2_Given ) model->HSM2_wibpc2 = 0.0 ;

    /* Cross-term dependence */
    if ( !model->HSM2_pvmax_Given ) model->HSM2_pvmax = 0.0 ;
    if ( !model->HSM2_pbgtmp1_Given ) model->HSM2_pbgtmp1 = 0.0 ;
    if ( !model->HSM2_pbgtmp2_Given ) model->HSM2_pbgtmp2 = 0.0 ;
    if ( !model->HSM2_peg0_Given ) model->HSM2_peg0 = 0.0 ;
    if ( !model->HSM2_plover_Given ) model->HSM2_plover = 0.0 ;
    if ( !model->HSM2_pvfbover_Given ) model->HSM2_pvfbover = 0.0 ;
    if ( !model->HSM2_pnover_Given ) model->HSM2_pnover = 0.0 ;
    if ( !model->HSM2_pwl2_Given ) model->HSM2_pwl2 = 0.0 ;
    if ( !model->HSM2_pvfbc_Given ) model->HSM2_pvfbc = 0.0 ;
    if ( !model->HSM2_pnsubc_Given ) model->HSM2_pnsubc = 0.0 ;
    if ( !model->HSM2_pnsubp_Given ) model->HSM2_pnsubp = 0.0 ;
    if ( !model->HSM2_pscp1_Given ) model->HSM2_pscp1 = 0.0 ;
    if ( !model->HSM2_pscp2_Given ) model->HSM2_pscp2 = 0.0 ;
    if ( !model->HSM2_pscp3_Given ) model->HSM2_pscp3 = 0.0 ;
    if ( !model->HSM2_psc1_Given ) model->HSM2_psc1 = 0.0 ;
    if ( !model->HSM2_psc2_Given ) model->HSM2_psc2 = 0.0 ;
    if ( !model->HSM2_psc3_Given ) model->HSM2_psc3 = 0.0 ;
    if ( !model->HSM2_psc4_Given ) model->HSM2_psc4 = 0.0 ;
    if ( !model->HSM2_ppgd1_Given ) model->HSM2_ppgd1 = 0.0 ;
    if ( !model->HSM2_pndep_Given ) model->HSM2_pndep = 0.0 ;
    if ( !model->HSM2_pninv_Given ) model->HSM2_pninv = 0.0 ;
    if ( !model->HSM2_pmuecb0_Given ) model->HSM2_pmuecb0 = 0.0 ;
    if ( !model->HSM2_pmuecb1_Given ) model->HSM2_pmuecb1 = 0.0 ;
    if ( !model->HSM2_pmueph1_Given ) model->HSM2_pmueph1 = 0.0 ;
    if ( !model->HSM2_pvtmp_Given ) model->HSM2_pvtmp = 0.0 ;
    if ( !model->HSM2_pwvth0_Given ) model->HSM2_pwvth0 = 0.0 ;
    if ( !model->HSM2_pmuesr1_Given ) model->HSM2_pmuesr1 = 0.0 ;
    if ( !model->HSM2_pmuetmp_Given ) model->HSM2_pmuetmp = 0.0 ;
    if ( !model->HSM2_psub1_Given ) model->HSM2_psub1 = 0.0 ;
    if ( !model->HSM2_psub2_Given ) model->HSM2_psub2 = 0.0 ;
    if ( !model->HSM2_psvds_Given ) model->HSM2_psvds = 0.0 ;
    if ( !model->HSM2_psvbs_Given ) model->HSM2_psvbs = 0.0 ;
    if ( !model->HSM2_psvgs_Given ) model->HSM2_psvgs = 0.0 ;
    if ( !model->HSM2_pnsti_Given ) model->HSM2_pnsti = 0.0 ;
    if ( !model->HSM2_pwsti_Given ) model->HSM2_pwsti = 0.0 ;
    if ( !model->HSM2_pscsti1_Given ) model->HSM2_pscsti1 = 0.0 ;
    if ( !model->HSM2_pscsti2_Given ) model->HSM2_pscsti2 = 0.0 ;
    if ( !model->HSM2_pvthsti_Given ) model->HSM2_pvthsti = 0.0 ;
    if ( !model->HSM2_pmuesti1_Given ) model->HSM2_pmuesti1 = 0.0 ;
    if ( !model->HSM2_pmuesti2_Given ) model->HSM2_pmuesti2 = 0.0 ;
    if ( !model->HSM2_pmuesti3_Given ) model->HSM2_pmuesti3 = 0.0 ;
    if ( !model->HSM2_pnsubpsti1_Given ) model->HSM2_pnsubpsti1 = 0.0 ;
    if ( !model->HSM2_pnsubpsti2_Given ) model->HSM2_pnsubpsti2 = 0.0 ;
    if ( !model->HSM2_pnsubpsti3_Given ) model->HSM2_pnsubpsti3 = 0.0 ;
    if ( !model->HSM2_pcgso_Given ) model->HSM2_pcgso = 0.0 ;
    if ( !model->HSM2_pcgdo_Given ) model->HSM2_pcgdo = 0.0 ;
    if ( !model->HSM2_pjs0_Given ) model->HSM2_pjs0 = 0.0 ;
    if ( !model->HSM2_pjs0sw_Given ) model->HSM2_pjs0sw = 0.0 ;
    if ( !model->HSM2_pnj_Given ) model->HSM2_pnj = 0.0 ;
    if ( !model->HSM2_pcisbk_Given ) model->HSM2_pcisbk = 0.0 ;
    if ( !model->HSM2_pclm1_Given ) model->HSM2_pclm1 = 0.0 ;
    if ( !model->HSM2_pclm2_Given ) model->HSM2_pclm2 = 0.0 ;
    if ( !model->HSM2_pclm3_Given ) model->HSM2_pclm3 = 0.0 ;
    if ( !model->HSM2_pwfc_Given ) model->HSM2_pwfc = 0.0 ;
    if ( !model->HSM2_pgidl1_Given ) model->HSM2_pgidl1 = 0.0 ;
    if ( !model->HSM2_pgidl2_Given ) model->HSM2_pgidl2 = 0.0 ;
    if ( !model->HSM2_pgleak1_Given ) model->HSM2_pgleak1 = 0.0 ;
    if ( !model->HSM2_pgleak2_Given ) model->HSM2_pgleak2 = 0.0 ;
    if ( !model->HSM2_pgleak3_Given ) model->HSM2_pgleak3 = 0.0 ;
    if ( !model->HSM2_pgleak6_Given ) model->HSM2_pgleak6 = 0.0 ;
    if ( !model->HSM2_pglksd1_Given ) model->HSM2_pglksd1 = 0.0 ;
    if ( !model->HSM2_pglksd2_Given ) model->HSM2_pglksd2 = 0.0 ;
    if ( !model->HSM2_pglkb1_Given ) model->HSM2_pglkb1 = 0.0 ;
    if ( !model->HSM2_pglkb2_Given ) model->HSM2_pglkb2 = 0.0 ;
    if ( !model->HSM2_pnftrp_Given ) model->HSM2_pnftrp = 0.0 ;
    if ( !model->HSM2_pnfalp_Given ) model->HSM2_pnfalp = 0.0 ;
    if ( !model->HSM2_pvdiffj_Given ) model->HSM2_pvdiffj = 0.0 ;
    if ( !model->HSM2_pibpc1_Given ) model->HSM2_pibpc1 = 0.0 ;
    if ( !model->HSM2_pibpc2_Given ) model->HSM2_pibpc2 = 0.0 ;


    if ( model->HSM2_corecip == 1 ){
      model->HSM2_sc2 =  0.0 ; model->HSM2_lsc2 =  0.0 ; model->HSM2_wsc2 =  0.0 ; model->HSM2_psc2 =  0.0 ;
      model->HSM2_scp2 = 0.0 ; model->HSM2_lscp2 = 0.0 ; model->HSM2_wscp2 = 0.0 ; model->HSM2_pscp2 = 0.0 ;
      model->HSM2_sc4 = 0.0 ;
      model->HSM2_coqy = 0 ;
    }



    /* loop through all the instances of the model */
    for ( here = model->HSM2instances ;here != NULL ;
	  here = here->HSM2nextInstance ) {
      /* allocate a chunk of the state vector */
      here->HSM2states = *states;
      if (model->HSM2_conqs)
	*states += HSM2numStatesNqs;
      else 
	*states += HSM2numStates;

      /* perform the parameter defaulting */
      if ( !here->HSM2_l_Given      ) here->HSM2_l      = 5.0e-6 ;
      if ( !here->HSM2_w_Given      ) here->HSM2_w      = 5.0e-6 ;
      if ( !here->HSM2_ad_Given     ) here->HSM2_ad     = 0.0 ;
      if ( !here->HSM2_as_Given     ) here->HSM2_as     = 0.0 ;
      if ( !here->HSM2_pd_Given     ) here->HSM2_pd     = 0.0 ;
      if ( !here->HSM2_ps_Given     ) here->HSM2_ps     = 0.0 ;
      if ( !here->HSM2_nrd_Given    ) here->HSM2_nrd    = 0.0 ;
      if ( !here->HSM2_nrs_Given    ) here->HSM2_nrs    = 0.0 ;
      if ( !here->HSM2_ngcon_Given  ) here->HSM2_ngcon  = 1.0 ;
      if ( !here->HSM2_xgw_Given    ) here->HSM2_xgw    = 0e0 ;
      if ( !here->HSM2_xgl_Given    ) here->HSM2_xgl    = 0e0 ;
      if ( !here->HSM2_nf_Given     ) here->HSM2_nf     = 1.0 ;
      if ( !here->HSM2_sa_Given     ) here->HSM2_sa     = 0 ;
      if ( !here->HSM2_sb_Given     ) here->HSM2_sb     = 0 ;
      if ( !here->HSM2_sd_Given     ) here->HSM2_sd     = 0 ;
      if ( !here->HSM2_temp_Given   ) here->HSM2_temp   = 27.0 ; /* [C] */
      if ( !here->HSM2_dtemp_Given  ) here->HSM2_dtemp  = 0.0 ;

      if ( !here->HSM2_icVBS_Given ) here->HSM2_icVBS = 0.0;
      if ( !here->HSM2_icVDS_Given ) here->HSM2_icVDS = 0.0;
      if ( !here->HSM2_icVGS_Given ) here->HSM2_icVGS = 0.0;

      if ( !here->HSM2_corbnet_Given )
	here->HSM2_corbnet = model->HSM2_corbnet ;
      else if ( here->HSM2_corbnet != 0 && here->HSM2_corbnet != 1 ) {
	here->HSM2_corbnet = model->HSM2_corbnet ;
	printf("warning(HiSIM2): CORBNET has been set to its default value: %d.\n", here->HSM2_corbnet);
      }
      if ( !here->HSM2_rbdb_Given) here->HSM2_rbdb = model->HSM2_rbdb; /* in ohm */
      if ( !here->HSM2_rbsb_Given) here->HSM2_rbsb = model->HSM2_rbsb;
      if ( !here->HSM2_rbpb_Given) here->HSM2_rbpb = model->HSM2_rbpb;
      if ( !here->HSM2_rbps_Given) here->HSM2_rbps = model->HSM2_rbps;
      if ( !here->HSM2_rbpd_Given) here->HSM2_rbpd = model->HSM2_rbpd;

      if ( !here->HSM2_corg_Given )
	here->HSM2_corg = model->HSM2_corg ;
      else if ( here->HSM2_corg != 0 && here->HSM2_corg != 1 ) {
	here->HSM2_corg = model->HSM2_corg ;
	printf("warning(HiSIM2): CORG has been set to its default value: %d.\n", here->HSM2_corg);
      }

      if ( !here->HSM2_mphdfm_Given ) here->HSM2_mphdfm = model->HSM2_mphdfm ;
      if ( !here->HSM2_m_Given      ) here->HSM2_m      = 1.0 ;

      /* WPE */
      if ( !here->HSM2_sca_Given ) here->HSM2_sca = 0.0 ; /* default value */
      if ( !here->HSM2_scb_Given ) here->HSM2_scb = 0.0 ; /* default value */
      if ( !here->HSM2_scc_Given ) here->HSM2_scc = 0.0 ; /* default value */

      /* process drain series resistance */
      if ((model->HSM2_corsrd < 0 && 
	   (model->HSM2_rsh > 0.0 || model->HSM2_rd > 0.0))) {
	if(here->HSM2dNodePrime <= 0) {
	error = CKTmkVolt(ckt, &tmp, here->HSM2name, "drain");
	if (error) return(error);
	here->HSM2dNodePrime = tmp->number;
       }
      } else {
	here->HSM2dNodePrime = here->HSM2dNode;
      }
      
      /* process source series resistance */
      if ((model->HSM2_corsrd < 0 && 
	   (model->HSM2_rsh > 0.0 || model->HSM2_rs > 0.0))) {
	if(here->HSM2sNodePrime == 0) {
	error = CKTmkVolt(ckt, &tmp, here->HSM2name, "source");
	if (error) return(error);
	here->HSM2sNodePrime = tmp->number;
       }
      } else {
	here->HSM2sNodePrime = here->HSM2sNode;
      }

      /* process gate resistance */
      if ((here->HSM2_corg == 1 && model->HSM2_rshg > 0.0)) {
       if(here->HSM2gNodePrime <= 0) {
	error = CKTmkVolt(ckt, &tmp, here->HSM2name, "gate");
	if (error) return(error);
	here->HSM2gNodePrime = tmp->number;
       }
      } else {
	here->HSM2gNodePrime = here->HSM2gNode;
      }

      /* internal body nodes for body resistance model */
      if ( here->HSM2_corbnet == 1 ) {
	if (here->HSM2dbNode == 0) {
	  error = CKTmkVolt(ckt, &tmp, here->HSM2name, "dbody");
	  if (error) return(error);
	  here->HSM2dbNode = tmp->number;
	}
	if (here->HSM2bNodePrime == 0) {
	  error = CKTmkVolt(ckt, &tmp,here->HSM2name, "body");
	  if (error) return(error);
	  here->HSM2bNodePrime = tmp->number;
	}
	if (here->HSM2sbNode == 0) {
	  error = CKTmkVolt(ckt, &tmp, here->HSM2name,"sbody");
	  if (error) return(error);
	  here->HSM2sbNode = tmp->number;
	}
      } else {
	here->HSM2dbNode = here->HSM2bNodePrime = here->HSM2sbNode = here->HSM2bNode;
      }
                   
      /* set Sparse Matrix Pointers */
      
      /* macro to make elements with built in test for out of memory */
#define TSTALLOC(ptr,first,second) \
if((here->ptr = SMPmakeElt(matrix,here->first,here->second))==(double *)NULL){\
    return(E_NOMEM);\
}

      TSTALLOC(HSM2DPbpPtr, HSM2dNodePrime, HSM2bNodePrime)
      TSTALLOC(HSM2SPbpPtr, HSM2sNodePrime, HSM2bNodePrime)
      TSTALLOC(HSM2GPbpPtr, HSM2gNodePrime, HSM2bNodePrime)

      TSTALLOC(HSM2BPdpPtr, HSM2bNodePrime, HSM2dNodePrime)
      TSTALLOC(HSM2BPspPtr, HSM2bNodePrime, HSM2sNodePrime)
      TSTALLOC(HSM2BPgpPtr, HSM2bNodePrime, HSM2gNodePrime)
      TSTALLOC(HSM2BPbpPtr, HSM2bNodePrime, HSM2bNodePrime)

      TSTALLOC(HSM2DdPtr, HSM2dNode, HSM2dNode)
      TSTALLOC(HSM2GPgpPtr, HSM2gNodePrime, HSM2gNodePrime)
      TSTALLOC(HSM2SsPtr, HSM2sNode, HSM2sNode)
      TSTALLOC(HSM2DPdpPtr, HSM2dNodePrime, HSM2dNodePrime)
      TSTALLOC(HSM2SPspPtr, HSM2sNodePrime, HSM2sNodePrime)
      TSTALLOC(HSM2DdpPtr, HSM2dNode, HSM2dNodePrime)
      TSTALLOC(HSM2GPdpPtr, HSM2gNodePrime, HSM2dNodePrime)
      TSTALLOC(HSM2GPspPtr, HSM2gNodePrime, HSM2sNodePrime)
      TSTALLOC(HSM2SspPtr, HSM2sNode, HSM2sNodePrime)
      TSTALLOC(HSM2DPspPtr, HSM2dNodePrime, HSM2sNodePrime)
      TSTALLOC(HSM2DPdPtr, HSM2dNodePrime, HSM2dNode)
      TSTALLOC(HSM2DPgpPtr, HSM2dNodePrime, HSM2gNodePrime)
      TSTALLOC(HSM2SPgpPtr, HSM2sNodePrime, HSM2gNodePrime)
      TSTALLOC(HSM2SPsPtr, HSM2sNodePrime, HSM2sNode)
      TSTALLOC(HSM2SPdpPtr, HSM2sNodePrime, HSM2dNodePrime);

      if ( here->HSM2_corg == 1 ) {
	TSTALLOC(HSM2GgPtr, HSM2gNode, HSM2gNode);
	TSTALLOC(HSM2GgpPtr, HSM2gNode, HSM2gNodePrime);
	TSTALLOC(HSM2GPgPtr, HSM2gNodePrime, HSM2gNode);
	TSTALLOC(HSM2GdpPtr, HSM2gNode, HSM2dNodePrime);
	TSTALLOC(HSM2GspPtr, HSM2gNode, HSM2sNodePrime);
	TSTALLOC(HSM2GbpPtr, HSM2gNode, HSM2bNodePrime);
      }

      if ( here->HSM2_corbnet == 1 ) { /* consider body resistance net */
	TSTALLOC(HSM2DPdbPtr, HSM2dNodePrime, HSM2dbNode);
	TSTALLOC(HSM2SPsbPtr, HSM2sNodePrime, HSM2sbNode);

	TSTALLOC(HSM2DBdpPtr, HSM2dbNode, HSM2dNodePrime);
	TSTALLOC(HSM2DBdbPtr, HSM2dbNode, HSM2dbNode);
	TSTALLOC(HSM2DBbpPtr, HSM2dbNode, HSM2bNodePrime);
	TSTALLOC(HSM2DBbPtr, HSM2dbNode, HSM2bNode);

	TSTALLOC(HSM2BPdbPtr, HSM2bNodePrime, HSM2dbNode);
	TSTALLOC(HSM2BPbPtr, HSM2bNodePrime, HSM2bNode);
	TSTALLOC(HSM2BPsbPtr, HSM2bNodePrime, HSM2sbNode);

	TSTALLOC(HSM2SBspPtr, HSM2sbNode, HSM2sNodePrime);
	TSTALLOC(HSM2SBbpPtr, HSM2sbNode, HSM2bNodePrime);
	TSTALLOC(HSM2SBbPtr, HSM2sbNode, HSM2bNode);
	TSTALLOC(HSM2SBsbPtr, HSM2sbNode, HSM2sbNode);

	TSTALLOC(HSM2BdbPtr, HSM2bNode, HSM2dbNode);
	TSTALLOC(HSM2BbpPtr, HSM2bNode, HSM2bNodePrime);
	TSTALLOC(HSM2BsbPtr, HSM2bNode, HSM2sbNode);
	TSTALLOC(HSM2BbPtr, HSM2bNode, HSM2bNode);
      }

    }
  }
  return(OK);
} 

int
HSM2unsetup(
    GENmodel *inModel,
    CKTcircuit *ckt)
{
#ifndef HAS_BATCHSIM
    HSM2model *model;
    HSM2instance *here;
 
    for (model = (HSM2model *)inModel; model != NULL;
            model = model->HSM2nextModel)
    {
        for (here = model->HSM2instances; here != NULL;
                here=here->HSM2nextInstance)
        {
            if (here->HSM2dNodePrime
                    && here->HSM2dNodePrime != here->HSM2dNode)
            {
                CKTdltNNum(ckt, here->HSM2dNodePrime);
                here->HSM2dNodePrime = 0;
            }
            if (here->HSM2sNodePrime
                    && here->HSM2sNodePrime != here->HSM2sNode)
            {
                CKTdltNNum(ckt, here->HSM2sNodePrime);
                here->HSM2sNodePrime = 0;
            }
            if (here->HSM2gNodePrime
                    && here->HSM2gNodePrime != here->HSM2gNode)
            {
                CKTdltNNum(ckt, here->HSM2gNodePrime);
                here->HSM2gNodePrime = 0;
            }
            if (here->HSM2bNodePrime
                    && here->HSM2bNodePrime != here->HSM2bNode)
            {
                CKTdltNNum(ckt, here->HSM2bNodePrime);
                here->HSM2bNodePrime = 0;
            }
            if (here->HSM2dbNode
                    && here->HSM2dbNode != here->HSM2bNode)
            {
                CKTdltNNum(ckt, here->HSM2dbNode);
                here->HSM2dbNode = 0;
            }
            if (here->HSM2sbNode
                    && here->HSM2sbNode != here->HSM2bNode)
            {
                CKTdltNNum(ckt, here->HSM2sbNode);
                here->HSM2sbNode = 0;
            }
        }
    }
#endif
    return OK;
}
