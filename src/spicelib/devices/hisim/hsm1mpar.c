/***********************************************************************
 HiSIM (Hiroshima University STARC IGFET Model)
 Copyright (C) 2003 STARC

 VERSION : HiSIM 1.2.0
 FILE : hsm1mpar.c of HiSIM 1.2.0

 April 9, 2003 : released by STARC Physical Design Group
***********************************************************************/

#include "ngspice.h"
#include "hsm1def.h"
#include "ifsim.h"
#include "sperror.h"
#include "suffix.h"

int 
HSM1mParam(int param, IFvalue *value, GENmodel *inMod)
{
  HSM1model *mod = (HSM1model*)inMod;
  switch (param) {
  case  HSM1_MOD_NMOS  :
    if (value->iValue) {
      mod->HSM1_type = 1;
      mod->HSM1_type_Given = TRUE;
    }
    break;
  case  HSM1_MOD_PMOS  :
    if (value->iValue) {
      mod->HSM1_type = - 1;
      mod->HSM1_type_Given = TRUE;
    }
    break;
  case  HSM1_MOD_LEVEL:
    mod->HSM1_level = value->iValue;
    mod->HSM1_level_Given = TRUE;
    break;
  case  HSM1_MOD_INFO:
    mod->HSM1_info = value->iValue;
    mod->HSM1_info_Given = TRUE;
    break;
  case HSM1_MOD_NOISE:
    mod->HSM1_noise = value->iValue;
    mod->HSM1_noise_Given = TRUE;
    break;
  case HSM1_MOD_VERSION:
    mod->HSM1_version = value->iValue;
    mod->HSM1_version_Given = TRUE;
    break;
  case HSM1_MOD_SHOW:
    mod->HSM1_show = value->iValue;
    mod->HSM1_show_Given = TRUE;
    break;
  case  HSM1_MOD_CORSRD:
    mod->HSM1_corsrd = value->iValue;
    mod->HSM1_corsrd_Given = TRUE;
    break;
  case  HSM1_MOD_COIPRV:
    mod->HSM1_coiprv = value->iValue;
    mod->HSM1_coiprv_Given = TRUE;
    break;
  case  HSM1_MOD_COPPRV:
    mod->HSM1_copprv = value->iValue;
    mod->HSM1_copprv_Given = TRUE;
    break;
  case  HSM1_MOD_COCGSO:
    mod->HSM1_cocgso = value->iValue;
    mod->HSM1_cocgso_Given = TRUE;
    break;
  case  HSM1_MOD_COCGDO:
    mod->HSM1_cocgdo = value->iValue;
    mod->HSM1_cocgdo_Given = TRUE;
    break;
  case  HSM1_MOD_COCGBO:
    mod->HSM1_cocgbo = value->iValue;
    mod->HSM1_cocgbo_Given = TRUE;
    break;
  case  HSM1_MOD_COADOV:
    mod->HSM1_coadov = value->iValue;
    mod->HSM1_coadov_Given = TRUE;
    break;
  case  HSM1_MOD_COXX08:
    mod->HSM1_coxx08 = value->iValue;
    mod->HSM1_coxx08_Given = TRUE;
    break;
  case  HSM1_MOD_COXX09:
    mod->HSM1_coxx09 = value->iValue;
    mod->HSM1_coxx09_Given = TRUE;
    break;
  case  HSM1_MOD_COISUB:
    mod->HSM1_coisub = value->iValue;
    mod->HSM1_coisub_Given = TRUE;
    break;
  case  HSM1_MOD_COIIGS:
    mod->HSM1_coiigs = value->iValue;
    mod->HSM1_coiigs_Given = TRUE;
    break;
  case  HSM1_MOD_COGIDL:
    mod->HSM1_cogidl = value->iValue;
    mod->HSM1_cogidl_Given = TRUE;
    break;
  case  HSM1_MOD_COGISL:
    mod->HSM1_cogisl = value->iValue;
    mod->HSM1_cogisl_Given = TRUE;
    break;
  case  HSM1_MOD_COOVLP:
    mod->HSM1_coovlp = value->iValue;
    mod->HSM1_coovlp_Given = TRUE;
    break;
  case  HSM1_MOD_CONOIS:
    mod->HSM1_conois = value->iValue;
    mod->HSM1_conois_Given = TRUE;
    break;
  case  HSM1_MOD_COISTI: /* HiSIM1.1 */
    mod->HSM1_coisti = value->iValue;
    mod->HSM1_coisti_Given = TRUE;
    break;
  case  HSM1_MOD_COSMBI: /* HiSIM1.2 */
    mod->HSM1_cosmbi = value->iValue;
    mod->HSM1_cosmbi_Given = TRUE;
    break;
  case  HSM1_MOD_VMAX:
    mod->HSM1_vmax = value->rValue;
    mod->HSM1_vmax_Given = TRUE;
    break;
  case  HSM1_MOD_BGTMP1:
    mod->HSM1_bgtmp1 = value->rValue;
    mod->HSM1_bgtmp1_Given = TRUE;
    break;
  case  HSM1_MOD_BGTMP2:
    mod->HSM1_bgtmp2 =  value->rValue;
    mod->HSM1_bgtmp2_Given = TRUE;
    break;
  case  HSM1_MOD_TOX:
    mod->HSM1_tox =  value->rValue;
    mod->HSM1_tox_Given = TRUE;
    break;
  case  HSM1_MOD_XLD:
    mod->HSM1_xld = value->rValue;
    mod->HSM1_xld_Given = TRUE;
    break;
  case  HSM1_MOD_XWD:
    mod->HSM1_xwd = value->rValue;
    mod->HSM1_xwd_Given = TRUE;
    break;
  case  HSM1_MOD_XJ: /* HiSIM1.0 */
    mod->HSM1_xj = value->rValue;
    mod->HSM1_xj_Given = TRUE;
    break;
  case  HSM1_MOD_XQY: /* HiSIM1.1 */
    mod->HSM1_xqy = value->rValue;
    mod->HSM1_xqy_Given = TRUE;
    break;
  case  HSM1_MOD_RS:
    mod->HSM1_rs = value->rValue;
    mod->HSM1_rs_Given = TRUE;
    break;
  case  HSM1_MOD_RD:
    mod->HSM1_rd = value->rValue;
    mod->HSM1_rd_Given = TRUE;
    break;
  case  HSM1_MOD_VFBC:
    mod->HSM1_vfbc = value->rValue;
    mod->HSM1_vfbc_Given = TRUE;
    break;
  case  HSM1_MOD_NSUBC:
    mod->HSM1_nsubc = value->rValue;
    mod->HSM1_nsubc_Given = TRUE;
    break;
  case  HSM1_MOD_PARL1:
    mod->HSM1_parl1 = value->rValue;
    mod->HSM1_parl1_Given = TRUE;
    break;
  case  HSM1_MOD_PARL2:
    mod->HSM1_parl2 = value->rValue;
    mod->HSM1_parl2_Given = TRUE;
    break;
  case  HSM1_MOD_LP:
    mod->HSM1_lp = value->rValue;
    mod->HSM1_lp_Given = TRUE;
    break;
  case  HSM1_MOD_NSUBP:
    mod->HSM1_nsubp = value->rValue;
    mod->HSM1_nsubp_Given = TRUE;
    break;
  case  HSM1_MOD_SCP1:
    mod->HSM1_scp1 = value->rValue;
    mod->HSM1_scp1_Given = TRUE;
    break;
  case  HSM1_MOD_SCP2:
    mod->HSM1_scp2 = value->rValue;
    mod->HSM1_scp2_Given = TRUE;
    break;
  case  HSM1_MOD_SCP3:
    mod->HSM1_scp3 = value->rValue;
    mod->HSM1_scp3_Given = TRUE;
    break;
  case  HSM1_MOD_SC1:
    mod->HSM1_sc1 = value->rValue;
    mod->HSM1_sc1_Given = TRUE;
    break;
  case  HSM1_MOD_SC2:
    mod->HSM1_sc2 = value->rValue;
    mod->HSM1_sc2_Given = TRUE;
    break;
  case  HSM1_MOD_SC3:
    mod->HSM1_sc3 = value->rValue;
    mod->HSM1_sc3_Given = TRUE;
    break;
  case  HSM1_MOD_PGD1:
    mod->HSM1_pgd1 = value->rValue;
    mod->HSM1_pgd1_Given = TRUE;
    break;
  case  HSM1_MOD_PGD2:
    mod->HSM1_pgd2 = value->rValue;
    mod->HSM1_pgd2_Given = TRUE;
    break;
  case  HSM1_MOD_PGD3:
    mod->HSM1_pgd3 = value->rValue;
    mod->HSM1_pgd3_Given = TRUE;
    break;
  case  HSM1_MOD_NDEP:
    mod->HSM1_ndep = value->rValue;
    mod->HSM1_ndep_Given = TRUE;
    break;
  case  HSM1_MOD_NINV:
    mod->HSM1_ninv = value->rValue;
    mod->HSM1_ninv_Given = TRUE;
    break;
  case  HSM1_MOD_NINVD:
    mod->HSM1_ninvd = value->rValue;
    mod->HSM1_ninvd_Given = TRUE;
    break;
  case  HSM1_MOD_MUECB0:
    mod->HSM1_muecb0 = value->rValue;
    mod->HSM1_muecb0_Given = TRUE;
    break;
  case  HSM1_MOD_MUECB1:
    mod->HSM1_muecb1 = value->rValue;
    mod->HSM1_muecb1_Given = TRUE;
    break;
  case  HSM1_MOD_MUEPH1:
    mod->HSM1_mueph1 = value->rValue;
    mod->HSM1_mueph1_Given = TRUE;
    break;
  case  HSM1_MOD_MUEPH0:
    mod->HSM1_mueph0 = value->rValue;
    mod->HSM1_mueph0_Given = TRUE;
    break;
  case  HSM1_MOD_MUEPH2:
    mod->HSM1_mueph2 = value->rValue;
    mod->HSM1_mueph2_Given = TRUE;
    break;
  case  HSM1_MOD_W0:
    mod->HSM1_w0 = value->rValue;
    mod->HSM1_w0_Given = TRUE;
    break;
  case  HSM1_MOD_MUESR1:
    mod->HSM1_muesr1 = value->rValue;
    mod->HSM1_muesr1_Given = TRUE;
    break;
  case  HSM1_MOD_MUESR0:
    mod->HSM1_muesr0 = value->rValue;
    mod->HSM1_muesr0_Given = TRUE;
    break;
  case  HSM1_MOD_BB:
    mod->HSM1_bb = value->rValue;
    mod->HSM1_bb_Given = TRUE;
    break;
  case  HSM1_MOD_SUB1:
    mod->HSM1_sub1 = value->rValue;
    mod->HSM1_sub1_Given = TRUE;
    break;
  case  HSM1_MOD_SUB2:
    mod->HSM1_sub2 = value->rValue;
    mod->HSM1_sub2_Given = TRUE;
    break;
  case  HSM1_MOD_SUB3:
    mod->HSM1_sub3 = value->rValue;
    mod->HSM1_sub3_Given = TRUE;
    break;
  case  HSM1_MOD_WVTHSC: /* HiSIM1.1 */
    mod->HSM1_wvthsc = value->rValue;
    mod->HSM1_wvthsc_Given = TRUE;
    break;
  case  HSM1_MOD_NSTI: /* HiSIM1.1 */
    mod->HSM1_nsti = value->rValue;
    mod->HSM1_nsti_Given = TRUE;
    break;
  case  HSM1_MOD_WSTI: /* HiSIM1.1 */
    mod->HSM1_wsti = value->rValue;
    mod->HSM1_wsti_Given = TRUE;
    break;
  case  HSM1_MOD_CGSO:
    mod->HSM1_cgso = value->rValue;
    mod->HSM1_cgso_Given = TRUE;
    break;
  case  HSM1_MOD_CGDO:
    mod->HSM1_cgdo = value->rValue;
    mod->HSM1_cgdo_Given = TRUE;
    break;
  case  HSM1_MOD_CGBO:
    mod->HSM1_cgbo = value->rValue;
    mod->HSM1_cgbo_Given = TRUE;
    break;
  case  HSM1_MOD_TPOLY:
    mod->HSM1_tpoly = value->rValue;
    mod->HSM1_tpoly_Given = TRUE;
    break;
  case  HSM1_MOD_JS0:
    mod->HSM1_js0 = value->rValue;
    mod->HSM1_js0_Given = TRUE;
    break;
  case  HSM1_MOD_JS0SW:
    mod->HSM1_js0sw = value->rValue;
    mod->HSM1_js0sw_Given = TRUE;
    break;
  case  HSM1_MOD_NJ:
    mod->HSM1_nj = value->rValue;
    mod->HSM1_nj_Given = TRUE;
    break;
  case  HSM1_MOD_NJSW:
    mod->HSM1_njsw = value->rValue;
    mod->HSM1_njsw_Given = TRUE;
    break;
  case  HSM1_MOD_XTI:
    mod->HSM1_xti = value->rValue;
    mod->HSM1_xti_Given = TRUE;
    break;
  case  HSM1_MOD_CJ:
    mod->HSM1_cj = value->rValue;
    mod->HSM1_cj_Given = TRUE;
    break;
  case  HSM1_MOD_CJSW:
    mod->HSM1_cjsw = value->rValue;
    mod->HSM1_cjsw_Given = TRUE;
    break;
  case  HSM1_MOD_CJSWG:
    mod->HSM1_cjswg = value->rValue;
    mod->HSM1_cjswg_Given = TRUE;
    break;
  case  HSM1_MOD_MJ:
    mod->HSM1_mj = value->rValue;
    mod->HSM1_mj_Given = TRUE;
    break;
  case  HSM1_MOD_MJSW:
    mod->HSM1_mjsw = value->rValue;
    mod->HSM1_mjsw_Given = TRUE;
    break;
  case  HSM1_MOD_MJSWG:
    mod->HSM1_mjswg = value->rValue;
    mod->HSM1_mjswg_Given = TRUE;
    break;
  case  HSM1_MOD_PB:
    mod->HSM1_pb = value->rValue;
    mod->HSM1_pb_Given = TRUE;
    break;
  case  HSM1_MOD_PBSW:
    mod->HSM1_pbsw = value->rValue;
    mod->HSM1_pbsw_Given = TRUE;
    break;
  case  HSM1_MOD_PBSWG:
    mod->HSM1_pbswg = value->rValue;
    mod->HSM1_pbswg_Given = TRUE;
    break;
  case  HSM1_MOD_XPOLYD:
    mod->HSM1_xpolyd = value->rValue;
    mod->HSM1_xpolyd_Given = TRUE;
    break;
  case  HSM1_MOD_CLM1:
    mod->HSM1_clm1 = value->rValue;
    mod->HSM1_clm1_Given = TRUE;
    break;
  case  HSM1_MOD_CLM2:
    mod->HSM1_clm2 = value->rValue;
    mod->HSM1_clm2_Given = TRUE;
    break;
  case  HSM1_MOD_CLM3:
    mod->HSM1_clm3 = value->rValue;
    mod->HSM1_clm3_Given = TRUE;
    break;
  case  HSM1_MOD_MUETMP:
    mod->HSM1_muetmp = value->rValue;
    mod->HSM1_muetmp_Given = TRUE;
    break;
  case  HSM1_MOD_RPOCK1:
    mod->HSM1_rpock1 = value->rValue;
    mod->HSM1_rpock1_Given = TRUE;
    break;
  case  HSM1_MOD_RPOCK2:
    mod->HSM1_rpock2 = value->rValue;
    mod->HSM1_rpock2_Given = TRUE;
    break;
  case  HSM1_MOD_RPOCP1: /* HiSIM1.1 */
    mod->HSM1_rpocp1 = value->rValue;
    mod->HSM1_rpocp1_Given = TRUE;
    break;
  case  HSM1_MOD_RPOCP2: /* HiSIM1.1 */
    mod->HSM1_rpocp2 = value->rValue;
    mod->HSM1_rpocp2_Given = TRUE;
    break;
  case  HSM1_MOD_VOVER:
    mod->HSM1_vover = value->rValue;
    mod->HSM1_vover_Given = TRUE;
    break;
  case  HSM1_MOD_VOVERP:
    mod->HSM1_voverp = value->rValue;
    mod->HSM1_voverp_Given = TRUE;
    break;
  case  HSM1_MOD_WFC:
    mod->HSM1_wfc = value->rValue;
    mod->HSM1_wfc_Given = TRUE;
    break;
  case  HSM1_MOD_QME1:
    mod->HSM1_qme1 = value->rValue;
    mod->HSM1_qme1_Given = TRUE;
    break;
  case  HSM1_MOD_QME2:
    mod->HSM1_qme2 = value->rValue;
    mod->HSM1_qme2_Given = TRUE;
    break;
  case  HSM1_MOD_QME3:
    mod->HSM1_qme3 = value->rValue;
    mod->HSM1_qme3_Given = TRUE;
    break;
  case  HSM1_MOD_GIDL1:
    mod->HSM1_gidl1 = value->rValue;
    mod->HSM1_gidl1_Given = TRUE;
    break;
  case  HSM1_MOD_GIDL2:
    mod->HSM1_gidl2 = value->rValue;
    mod->HSM1_gidl2_Given = TRUE;
    break;
  case  HSM1_MOD_GIDL3:
    mod->HSM1_gidl3 = value->rValue;
    mod->HSM1_gidl3_Given = TRUE;
    break;
  case  HSM1_MOD_GLEAK1:
    mod->HSM1_gleak1 = value->rValue;
    mod->HSM1_gleak1_Given = TRUE;
    break;
  case  HSM1_MOD_GLEAK2:
    mod->HSM1_gleak2 = value->rValue;
    mod->HSM1_gleak2_Given = TRUE;
    break;
  case  HSM1_MOD_GLEAK3:
    mod->HSM1_gleak3 = value->rValue;
    mod->HSM1_gleak3_Given = TRUE;
    break;
  case  HSM1_MOD_VZADD0:
    mod->HSM1_vzadd0 = value->rValue;
    mod->HSM1_vzadd0_Given = TRUE;
    break;
  case  HSM1_MOD_PZADD0:
    mod->HSM1_pzadd0 = value->rValue;
    mod->HSM1_pzadd0_Given = TRUE;
    break;
  case  HSM1_MOD_NFTRP:
    mod->HSM1_nftrp = value->rValue;
    mod->HSM1_nftrp_Given = TRUE;
    break;
  case  HSM1_MOD_NFALP:
    mod->HSM1_nfalp = value->rValue;
    mod->HSM1_nfalp_Given = TRUE;
    break;
  case  HSM1_MOD_CIT:
    mod->HSM1_cit = value->rValue;
    mod->HSM1_cit_Given = TRUE;
    break;
  case  HSM1_MOD_GLPART1: /* HiSIM1.2 */
    mod->HSM1_glpart1 = value->rValue;
    mod->HSM1_glpart1_Given = TRUE;
    break;
  case  HSM1_MOD_GLPART2: /* HiSIM1.2 */
    mod->HSM1_glpart2 = value->rValue;
    mod->HSM1_glpart2_Given = TRUE;
    break;
  case  HSM1_MOD_KAPPA: /* HiSIM1.2 */
    mod->HSM1_kappa = value->rValue;
    mod->HSM1_kappa_Given = TRUE;
    break;
  case  HSM1_MOD_XDIFFD: /* HiSIM1.2 */
    mod->HSM1_xdiffd = value->rValue;
    mod->HSM1_xdiffd_Given = TRUE;
    break;
  case  HSM1_MOD_PTHROU: /* HiSIM1.2 */
    mod->HSM1_pthrou = value->rValue;
    mod->HSM1_pthrou_Given = TRUE;
    break;
  case  HSM1_MOD_VDIFFJ: /* HiSIM1.2 */
    mod->HSM1_vdiffj = value->rValue;
    mod->HSM1_vdiffj_Given = TRUE;
    break;
  case HSM1_MOD_KF:
    mod->HSM1_kf = value->rValue;
    mod->HSM1_kf_Given = TRUE;
    break;
  case HSM1_MOD_AF:
    mod->HSM1_af = value->rValue;
    mod->HSM1_af_Given = TRUE;
    break;
  case HSM1_MOD_EF:
    mod->HSM1_ef = value->rValue;
    mod->HSM1_ef_Given = TRUE;
    break;
  default:
    return(E_BADPARM);
  }
  return(OK);
}

