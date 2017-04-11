/***********************************************************************

 HiSIM (Hiroshima University STARC IGFET Model)
 Copyright (C) 2012 Hiroshima University & STARC

 MODEL NAME : HiSIM_HV 
 ( VERSION : 1  SUBVERSION : 2  REVISION : 4 )
 Model Parameter VERSION : 1.23
 FILE : hsmhvmpar.c

 DATE : 2013.04.30

 released by
                Hiroshima University &
                Semiconductor Technology Academic Research Center (STARC)
***********************************************************************/

#include "ngspice/ngspice.h"
#include "hsmhvdef.h"
#include "ngspice/ifsim.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

int HSMHVmParam(
     int param,
     IFvalue *value,
     GENmodel *inMod)
{
  HSMHVmodel *mod = (HSMHVmodel*)inMod;
  switch (param) {
  case  HSMHV_MOD_NMOS  :
    if (value->iValue) {
      mod->HSMHV_type = 1;
      mod->HSMHV_type_Given = TRUE;
    }
    break;
  case  HSMHV_MOD_PMOS  :
    if (value->iValue) {
      mod->HSMHV_type = - 1;
      mod->HSMHV_type_Given = TRUE;
    }
    break;
  case  HSMHV_MOD_LEVEL:
    mod->HSMHV_level = value->iValue;
    mod->HSMHV_level_Given = TRUE;
    break;
  case  HSMHV_MOD_INFO:
    mod->HSMHV_info = value->iValue;
    mod->HSMHV_info_Given = TRUE;
    break;
  case HSMHV_MOD_NOISE:
    mod->HSMHV_noise = value->iValue;
    mod->HSMHV_noise_Given = TRUE;
    break;
  case HSMHV_MOD_VERSION:
    mod->HSMHV_version = value->sValue;
    mod->HSMHV_version_Given = TRUE;
    break;
  case HSMHV_MOD_SHOW:
    mod->HSMHV_show = value->iValue;
    mod->HSMHV_show_Given = TRUE;
    break;
  case  HSMHV_MOD_CORSRD:
    mod->HSMHV_corsrd = value->iValue;
    mod->HSMHV_corsrd_Given = TRUE;
    break;
  case  HSMHV_MOD_CORG:
    mod->HSMHV_corg = value->iValue;
    mod->HSMHV_corg_Given = TRUE;
    break;
  case  HSMHV_MOD_COIPRV:
    mod->HSMHV_coiprv = value->iValue;
    mod->HSMHV_coiprv_Given = TRUE;
    break;
  case  HSMHV_MOD_COPPRV:
    mod->HSMHV_copprv = value->iValue;
    mod->HSMHV_copprv_Given = TRUE;
    break;
  case  HSMHV_MOD_COADOV:
    mod->HSMHV_coadov = value->iValue;
    mod->HSMHV_coadov_Given = TRUE;
    break;
  case  HSMHV_MOD_COISUB:
    mod->HSMHV_coisub = value->iValue;
    mod->HSMHV_coisub_Given = TRUE;
    break;
  case  HSMHV_MOD_COIIGS:
    mod->HSMHV_coiigs = value->iValue;
    mod->HSMHV_coiigs_Given = TRUE;
    break;
  case  HSMHV_MOD_COGIDL:
    mod->HSMHV_cogidl = value->iValue;
    mod->HSMHV_cogidl_Given = TRUE;
    break;
  case  HSMHV_MOD_COOVLP:
    mod->HSMHV_coovlp = value->iValue;
    mod->HSMHV_coovlp_Given = TRUE;
    break;
  case  HSMHV_MOD_COOVLPS:
    mod->HSMHV_coovlps = value->iValue;
    mod->HSMHV_coovlps_Given = TRUE;
    break;
  case  HSMHV_MOD_COFLICK:
    mod->HSMHV_coflick = value->iValue;
    mod->HSMHV_coflick_Given = TRUE;
    break;
  case  HSMHV_MOD_COISTI:
    mod->HSMHV_coisti = value->iValue;
    mod->HSMHV_coisti_Given = TRUE;
    break;
  case  HSMHV_MOD_CONQS: /* HiSIMHV */
    mod->HSMHV_conqs = value->iValue;
    mod->HSMHV_conqs_Given = TRUE;
    break;
  case  HSMHV_MOD_CORBNET: 
    mod->HSMHV_corbnet = value->iValue;
    mod->HSMHV_corbnet_Given = TRUE;
    break;
  case  HSMHV_MOD_COTHRML:
    mod->HSMHV_cothrml = value->iValue;
    mod->HSMHV_cothrml_Given = TRUE;
    break;
  case  HSMHV_MOD_COIGN:
    mod->HSMHV_coign = value->iValue;
    mod->HSMHV_coign_Given = TRUE;
    break;
  case  HSMHV_MOD_CODFM:
    mod->HSMHV_codfm = value->iValue;
    mod->HSMHV_codfm_Given = TRUE;
    break;
  case  HSMHV_MOD_COQOVSM:
    mod->HSMHV_coqovsm = value->iValue;
    mod->HSMHV_coqovsm_Given = TRUE;
    break;
  case  HSMHV_MOD_COSELFHEAT: /* Self-heating model */
    mod->HSMHV_coselfheat = value->iValue;
    mod->HSMHV_coselfheat_Given = TRUE;
    break;
  case  HSMHV_MOD_COSUBNODE:
    mod->HSMHV_cosubnode = value->iValue;
    mod->HSMHV_cosubnode_Given = TRUE;
    break;
  case  HSMHV_MOD_COSYM: /* Symmetry model for HV */
    mod->HSMHV_cosym = value->iValue;
    mod->HSMHV_cosym_Given = TRUE;
    break;
  case  HSMHV_MOD_COTEMP:
    mod->HSMHV_cotemp = value->iValue;
    mod->HSMHV_cotemp_Given = TRUE;
    break;
  case  HSMHV_MOD_COLDRIFT:
    mod->HSMHV_coldrift = value->iValue;
    mod->HSMHV_coldrift_Given = TRUE;
    break;
  case  HSMHV_MOD_VMAX:
    mod->HSMHV_vmax = value->rValue;
    mod->HSMHV_vmax_Given = TRUE;
    break;
  case  HSMHV_MOD_VMAXT1:
    mod->HSMHV_vmaxt1 = value->rValue;
    mod->HSMHV_vmaxt1_Given = TRUE;
    break;
  case  HSMHV_MOD_VMAXT2:
    mod->HSMHV_vmaxt2 = value->rValue;
    mod->HSMHV_vmaxt2_Given = TRUE;
    break;
  case  HSMHV_MOD_BGTMP1:
    mod->HSMHV_bgtmp1 = value->rValue;
    mod->HSMHV_bgtmp1_Given = TRUE;
    break;
  case  HSMHV_MOD_BGTMP2:
    mod->HSMHV_bgtmp2 =  value->rValue;
    mod->HSMHV_bgtmp2_Given = TRUE;
    break;
  case  HSMHV_MOD_EG0:
    mod->HSMHV_eg0 =  value->rValue;
    mod->HSMHV_eg0_Given = TRUE;
    break;
  case  HSMHV_MOD_TOX:
    mod->HSMHV_tox =  value->rValue;
    mod->HSMHV_tox_Given = TRUE;
    break;
  case  HSMHV_MOD_XLD:
    mod->HSMHV_xld = value->rValue;
    mod->HSMHV_xld_Given = TRUE;
    break;
  case  HSMHV_MOD_LOVER:
    mod->HSMHV_lover = value->rValue;
    mod->HSMHV_lover_Given = TRUE;
    break;
  case  HSMHV_MOD_LOVERS:
    mod->HSMHV_lovers = value->rValue;
    mod->HSMHV_lovers_Given = TRUE;
    break;
  case  HSMHV_MOD_RDOV11:
    mod->HSMHV_rdov11 = value->rValue;
    mod->HSMHV_rdov11_Given = TRUE;
    break;
  case  HSMHV_MOD_RDOV12:
    mod->HSMHV_rdov12 = value->rValue;
    mod->HSMHV_rdov12_Given = TRUE;
    break;
  case  HSMHV_MOD_RDOV13:
    mod->HSMHV_rdov13 = value->rValue;
    mod->HSMHV_rdov13_Given = TRUE;
    break;
  case  HSMHV_MOD_RDSLP1:
    mod->HSMHV_rdslp1 = value->rValue;
    mod->HSMHV_rdslp1_Given = TRUE;
    break;
  case  HSMHV_MOD_RDICT1:
    mod->HSMHV_rdict1= value->rValue;
    mod->HSMHV_rdict1_Given = TRUE;
    break;
  case  HSMHV_MOD_RDSLP2:
    mod->HSMHV_rdslp2 = value->rValue;
    mod->HSMHV_rdslp2_Given = TRUE;
    break;
  case  HSMHV_MOD_RDICT2:
    mod->HSMHV_rdict2 = value->rValue;
    mod->HSMHV_rdict2_Given = TRUE;
    break;
  case  HSMHV_MOD_LOVERLD:
    mod->HSMHV_loverld = value->rValue;
    mod->HSMHV_loverld_Given = TRUE;
    break;
  case  HSMHV_MOD_LDRIFT1:
    mod->HSMHV_ldrift1 = value->rValue;
    mod->HSMHV_ldrift1_Given = TRUE;
    break;
  case  HSMHV_MOD_LDRIFT2:
    mod->HSMHV_ldrift2 = value->rValue;
    mod->HSMHV_ldrift2_Given = TRUE;
    break;
  case  HSMHV_MOD_LDRIFT1S:
    mod->HSMHV_ldrift1s = value->rValue;
    mod->HSMHV_ldrift1s_Given = TRUE;
    break;
  case  HSMHV_MOD_LDRIFT2S:
    mod->HSMHV_ldrift2s = value->rValue;
    mod->HSMHV_ldrift2s_Given = TRUE;
    break;
  case  HSMHV_MOD_SUBLD1:
    mod->HSMHV_subld1 = value->rValue;
    mod->HSMHV_subld1_Given = TRUE;
    break;
  case  HSMHV_MOD_SUBLD2:
    mod->HSMHV_subld2 = value->rValue;
    mod->HSMHV_subld2_Given = TRUE;
    break;
  case  HSMHV_MOD_DDLTMAX: /* Vdseff */
    mod->HSMHV_ddltmax = value->rValue;
    mod->HSMHV_ddltmax_Given = TRUE;
    break;
  case  HSMHV_MOD_DDLTSLP: /* Vdseff */
    mod->HSMHV_ddltslp = value->rValue;
    mod->HSMHV_ddltslp_Given = TRUE;
    break;
  case  HSMHV_MOD_DDLTICT: /* Vdseff */
    mod->HSMHV_ddltict = value->rValue;
    mod->HSMHV_ddltict_Given = TRUE;
    break;
  case  HSMHV_MOD_VFBOVER:
    mod->HSMHV_vfbover = value->rValue;
    mod->HSMHV_vfbover_Given = TRUE;
    break;
  case  HSMHV_MOD_NOVER:
    mod->HSMHV_nover = value->rValue;
    mod->HSMHV_nover_Given = TRUE;
    break;
  case  HSMHV_MOD_NOVERS:
    mod->HSMHV_novers = value->rValue;
    mod->HSMHV_novers_Given = TRUE;
    break;
  case  HSMHV_MOD_XWD:
    mod->HSMHV_xwd = value->rValue;
    mod->HSMHV_xwd_Given = TRUE;
    break;
  case  HSMHV_MOD_XWDC:
    mod->HSMHV_xwdc = value->rValue;
    mod->HSMHV_xwdc_Given = TRUE;
    break;
  case  HSMHV_MOD_XL:
    mod->HSMHV_xl = value->rValue;
    mod->HSMHV_xl_Given = TRUE;
    break;
  case  HSMHV_MOD_XW:
    mod->HSMHV_xw = value->rValue;
    mod->HSMHV_xw_Given = TRUE;
    break;
  case  HSMHV_MOD_SAREF:
    mod->HSMHV_saref = value->rValue;
    mod->HSMHV_saref_Given = TRUE;
    break;
  case  HSMHV_MOD_SBREF:
    mod->HSMHV_sbref = value->rValue;
    mod->HSMHV_sbref_Given = TRUE;
    break;
  case  HSMHV_MOD_LL:
    mod->HSMHV_ll = value->rValue;
    mod->HSMHV_ll_Given = TRUE;
    break;
  case  HSMHV_MOD_LLD:
    mod->HSMHV_lld = value->rValue;
    mod->HSMHV_lld_Given = TRUE;
    break;
  case  HSMHV_MOD_LLN:
    mod->HSMHV_lln = value->rValue;
    mod->HSMHV_lln_Given = TRUE;
    break;
  case  HSMHV_MOD_WL:
    mod->HSMHV_wl = value->rValue;
    mod->HSMHV_wl_Given = TRUE;
    break;
  case  HSMHV_MOD_WL1:
    mod->HSMHV_wl1 = value->rValue;
    mod->HSMHV_wl1_Given = TRUE;
    break;
  case  HSMHV_MOD_WL1P:
    mod->HSMHV_wl1p = value->rValue;
    mod->HSMHV_wl1p_Given = TRUE;
    break;
  case  HSMHV_MOD_WL2:
    mod->HSMHV_wl2 = value->rValue;
    mod->HSMHV_wl2_Given = TRUE;
    break;
  case  HSMHV_MOD_WL2P:
    mod->HSMHV_wl2p = value->rValue;
    mod->HSMHV_wl2p_Given = TRUE;
    break;
  case  HSMHV_MOD_WLD:
    mod->HSMHV_wld = value->rValue;
    mod->HSMHV_wld_Given = TRUE;
    break;
  case  HSMHV_MOD_WLN:
    mod->HSMHV_wln = value->rValue;
    mod->HSMHV_wln_Given = TRUE;
    break;
  case  HSMHV_MOD_XQY:
    mod->HSMHV_xqy = value->rValue;
    mod->HSMHV_xqy_Given = TRUE;
    break;
  case  HSMHV_MOD_XQY1:
    mod->HSMHV_xqy1 = value->rValue;
    mod->HSMHV_xqy1_Given = TRUE;
    break;
  case  HSMHV_MOD_XQY2:
    mod->HSMHV_xqy2 = value->rValue;
    mod->HSMHV_xqy2_Given = TRUE;
    break;
  case  HSMHV_MOD_RS:
    mod->HSMHV_rs = value->rValue;
    mod->HSMHV_rs_Given = TRUE;
    break;
  case  HSMHV_MOD_RD:
    mod->HSMHV_rd = value->rValue;
    mod->HSMHV_rd_Given = TRUE;
    break;
  case  HSMHV_MOD_RSH:
    mod->HSMHV_rsh = value->rValue;
    mod->HSMHV_rsh_Given = TRUE;
    break;
  case  HSMHV_MOD_RSHG:
    mod->HSMHV_rshg = value->rValue;
    mod->HSMHV_rshg_Given = TRUE;
    break;
  case  HSMHV_MOD_VFBC:
    mod->HSMHV_vfbc = value->rValue;
    mod->HSMHV_vfbc_Given = TRUE;
    break;
  case  HSMHV_MOD_VBI:
    mod->HSMHV_vbi = value->rValue;
    mod->HSMHV_vbi_Given = TRUE;
    break;
  case  HSMHV_MOD_NSUBC:
    mod->HSMHV_nsubc = value->rValue;
    mod->HSMHV_nsubc_Given = TRUE;
    break;
  case  HSMHV_MOD_PARL2:
    mod->HSMHV_parl2 = value->rValue;
    mod->HSMHV_parl2_Given = TRUE;
    break;
  case  HSMHV_MOD_LP:
    mod->HSMHV_lp = value->rValue;
    mod->HSMHV_lp_Given = TRUE;
    break;
  case  HSMHV_MOD_NSUBP:
    mod->HSMHV_nsubp = value->rValue;
    mod->HSMHV_nsubp_Given = TRUE;
    break;
  case  HSMHV_MOD_NSUBP0:
    mod->HSMHV_nsubp0 = value->rValue;
    mod->HSMHV_nsubp0_Given = TRUE;
    break;
  case  HSMHV_MOD_NSUBWP:
    mod->HSMHV_nsubwp = value->rValue;
    mod->HSMHV_nsubwp_Given = TRUE;
    break;
  case  HSMHV_MOD_SCP1:
    mod->HSMHV_scp1 = value->rValue;
    mod->HSMHV_scp1_Given = TRUE;
    break;
  case  HSMHV_MOD_SCP2:
    mod->HSMHV_scp2 = value->rValue;
    mod->HSMHV_scp2_Given = TRUE;
    break;
  case  HSMHV_MOD_SCP3:
    mod->HSMHV_scp3 = value->rValue;
    mod->HSMHV_scp3_Given = TRUE;
    break;
  case  HSMHV_MOD_SC1:
    mod->HSMHV_sc1 = value->rValue;
    mod->HSMHV_sc1_Given = TRUE;
    break;
  case  HSMHV_MOD_SC2:
    mod->HSMHV_sc2 = value->rValue;
    mod->HSMHV_sc2_Given = TRUE;
    break;
  case  HSMHV_MOD_SC3:
    mod->HSMHV_sc3 = value->rValue;
    mod->HSMHV_sc3_Given = TRUE;
    break;
  case  HSMHV_MOD_SC4:
    mod->HSMHV_sc4 = value->rValue;
    mod->HSMHV_sc4_Given = TRUE;
    break;
  case  HSMHV_MOD_PGD1:
    mod->HSMHV_pgd1 = value->rValue;
    mod->HSMHV_pgd1_Given = TRUE;
    break;
  case  HSMHV_MOD_PGD2:
    mod->HSMHV_pgd2 = value->rValue;
    mod->HSMHV_pgd2_Given = TRUE;
    break;
  case  HSMHV_MOD_PGD3:
    mod->HSMHV_pgd3 = value->rValue;
    mod->HSMHV_pgd3_Given = TRUE;
    break;
  case  HSMHV_MOD_PGD4:
    mod->HSMHV_pgd4 = value->rValue;
    mod->HSMHV_pgd4_Given = TRUE;
    break;
  case  HSMHV_MOD_NDEP:
    mod->HSMHV_ndep = value->rValue;
    mod->HSMHV_ndep_Given = TRUE;
    break;
  case  HSMHV_MOD_NDEPL:
    mod->HSMHV_ndepl = value->rValue;
    mod->HSMHV_ndepl_Given = TRUE;
    break;
  case  HSMHV_MOD_NDEPLP:
    mod->HSMHV_ndeplp = value->rValue;
    mod->HSMHV_ndeplp_Given = TRUE;
    break;
  case  HSMHV_MOD_NINV:
    mod->HSMHV_ninv = value->rValue;
    mod->HSMHV_ninv_Given = TRUE;
    break;
  case  HSMHV_MOD_MUECB0:
    mod->HSMHV_muecb0 = value->rValue;
    mod->HSMHV_muecb0_Given = TRUE;
    break;
  case  HSMHV_MOD_MUECB1:
    mod->HSMHV_muecb1 = value->rValue;
    mod->HSMHV_muecb1_Given = TRUE;
    break;
  case  HSMHV_MOD_MUEPH1:
    mod->HSMHV_mueph1 = value->rValue;
    mod->HSMHV_mueph1_Given = TRUE;
    break;
  case  HSMHV_MOD_MUEPH0:
    mod->HSMHV_mueph0 = value->rValue;
    mod->HSMHV_mueph0_Given = TRUE;
    break;
  case  HSMHV_MOD_MUEPHW:
    mod->HSMHV_muephw = value->rValue;
    mod->HSMHV_muephw_Given = TRUE;
    break;
  case  HSMHV_MOD_MUEPWP:
    mod->HSMHV_muepwp = value->rValue;
    mod->HSMHV_muepwp_Given = TRUE;
    break;
  case  HSMHV_MOD_MUEPHL:
    mod->HSMHV_muephl = value->rValue;
    mod->HSMHV_muephl_Given = TRUE;
    break;
  case  HSMHV_MOD_MUEPLP:
    mod->HSMHV_mueplp = value->rValue;
    mod->HSMHV_mueplp_Given = TRUE;
    break;
  case  HSMHV_MOD_MUEPHS:
    mod->HSMHV_muephs = value->rValue;
    mod->HSMHV_muephs_Given = TRUE;
    break;
   case  HSMHV_MOD_MUEPSP:
    mod->HSMHV_muepsp = value->rValue;
    mod->HSMHV_muepsp_Given = TRUE;
    break;
  case  HSMHV_MOD_VTMP:
    mod->HSMHV_vtmp = value->rValue;
    mod->HSMHV_vtmp_Given = TRUE;
    break;
  case  HSMHV_MOD_WVTH0:
    mod->HSMHV_wvth0 = value->rValue;
    mod->HSMHV_wvth0_Given = TRUE;
    break;
  case  HSMHV_MOD_MUESR1:
    mod->HSMHV_muesr1 = value->rValue;
    mod->HSMHV_muesr1_Given = TRUE;
    break;
  case  HSMHV_MOD_MUESR0:
    mod->HSMHV_muesr0 = value->rValue;
    mod->HSMHV_muesr0_Given = TRUE;
    break;
  case  HSMHV_MOD_MUESRL:
    mod->HSMHV_muesrl = value->rValue;
    mod->HSMHV_muesrl_Given = TRUE;
    break;
  case  HSMHV_MOD_MUESLP:
    mod->HSMHV_mueslp = value->rValue;
    mod->HSMHV_mueslp_Given = TRUE;
    break;
  case  HSMHV_MOD_MUESRW:
    mod->HSMHV_muesrw = value->rValue;
    mod->HSMHV_muesrw_Given = TRUE;
    break;
  case  HSMHV_MOD_MUESWP:
    mod->HSMHV_mueswp = value->rValue;
    mod->HSMHV_mueswp_Given = TRUE;
    break;
  case  HSMHV_MOD_BB:
    mod->HSMHV_bb = value->rValue;
    mod->HSMHV_bb_Given = TRUE;
    break;
  case  HSMHV_MOD_SUB1:
    mod->HSMHV_sub1 = value->rValue;
    mod->HSMHV_sub1_Given = TRUE;
    break;
  case  HSMHV_MOD_SUB2:
    mod->HSMHV_sub2 = value->rValue;
    mod->HSMHV_sub2_Given = TRUE;
    break;
  case  HSMHV_MOD_SVGS:
    mod->HSMHV_svgs = value->rValue;
    mod->HSMHV_svgs_Given = TRUE;
    break;
  case  HSMHV_MOD_SVBS:
    mod->HSMHV_svbs = value->rValue;
    mod->HSMHV_svbs_Given = TRUE;
    break;
  case  HSMHV_MOD_SVBSL:
    mod->HSMHV_svbsl = value->rValue;
    mod->HSMHV_svbsl_Given = TRUE;
    break;
  case  HSMHV_MOD_SVDS:
    mod->HSMHV_svds = value->rValue;
    mod->HSMHV_svds_Given = TRUE;
    break;
  case  HSMHV_MOD_SLG:
    mod->HSMHV_slg = value->rValue;
    mod->HSMHV_slg_Given = TRUE;
    break;
  case  HSMHV_MOD_SUB1L:
    mod->HSMHV_sub1l = value->rValue;
    mod->HSMHV_sub1l_Given = TRUE;
    break;
  case  HSMHV_MOD_SUB2L:
    mod->HSMHV_sub2l = value->rValue;
    mod->HSMHV_sub2l_Given = TRUE;
    break;
  case  HSMHV_MOD_FN1:
    mod->HSMHV_fn1 = value->rValue;
    mod->HSMHV_fn1_Given = TRUE;
    break;
  case  HSMHV_MOD_FN2:
    mod->HSMHV_fn2 = value->rValue;
    mod->HSMHV_fn2_Given = TRUE;
    break;
  case  HSMHV_MOD_FN3:
    mod->HSMHV_fn3 = value->rValue;
    mod->HSMHV_fn3_Given = TRUE;
    break;
  case  HSMHV_MOD_FVBS:
    mod->HSMHV_fvbs = value->rValue;
    mod->HSMHV_fvbs_Given = TRUE;
    break;
  case  HSMHV_MOD_SVGSL:
    mod->HSMHV_svgsl = value->rValue;
    mod->HSMHV_svgsl_Given = TRUE;
    break;
  case  HSMHV_MOD_SVGSLP:
    mod->HSMHV_svgslp = value->rValue;
    mod->HSMHV_svgslp_Given = TRUE;
    break;
  case  HSMHV_MOD_SVGSWP:
    mod->HSMHV_svgswp = value->rValue;
    mod->HSMHV_svgswp_Given = TRUE;
    break;
  case  HSMHV_MOD_SVGSW:
    mod->HSMHV_svgsw = value->rValue;
    mod->HSMHV_svgsw_Given = TRUE;
    break;
  case  HSMHV_MOD_SVBSLP:
    mod->HSMHV_svbslp = value->rValue;
    mod->HSMHV_svbslp_Given = TRUE;
    break;
  case  HSMHV_MOD_SLGL:
    mod->HSMHV_slgl = value->rValue;
    mod->HSMHV_slgl_Given = TRUE;
    break;
  case  HSMHV_MOD_SLGLP:
    mod->HSMHV_slglp = value->rValue;
    mod->HSMHV_slglp_Given = TRUE;
    break;
  case  HSMHV_MOD_SUB1LP:
    mod->HSMHV_sub1lp = value->rValue;
    mod->HSMHV_sub1lp_Given = TRUE;
    break;
  case  HSMHV_MOD_NSTI:
    mod->HSMHV_nsti = value->rValue;
    mod->HSMHV_nsti_Given = TRUE;
    break;
  case  HSMHV_MOD_WSTI:
    mod->HSMHV_wsti = value->rValue;
    mod->HSMHV_wsti_Given = TRUE;
    break;
  case  HSMHV_MOD_WSTIL:
    mod->HSMHV_wstil = value->rValue;
    mod->HSMHV_wstil_Given = TRUE;
    break;
  case  HSMHV_MOD_WSTILP:
    mod->HSMHV_wstilp = value->rValue;
    mod->HSMHV_wstilp_Given = TRUE;
    break;
  case  HSMHV_MOD_WSTIW:
    mod->HSMHV_wstiw = value->rValue;
    mod->HSMHV_wstiw_Given = TRUE;
    break;
  case  HSMHV_MOD_WSTIWP:
    mod->HSMHV_wstiwp = value->rValue;
    mod->HSMHV_wstiwp_Given = TRUE;
    break;
  case  HSMHV_MOD_SCSTI1:
    mod->HSMHV_scsti1 = value->rValue;
    mod->HSMHV_scsti1_Given = TRUE;
    break;
  case  HSMHV_MOD_SCSTI2:
    mod->HSMHV_scsti2 = value->rValue;
    mod->HSMHV_scsti2_Given = TRUE;
    break;
  case  HSMHV_MOD_VTHSTI:
    mod->HSMHV_vthsti = value->rValue;
    mod->HSMHV_vthsti_Given = TRUE;
    break;
  case  HSMHV_MOD_VDSTI:
    mod->HSMHV_vdsti = value->rValue;
    mod->HSMHV_vdsti_Given = TRUE;
    break;
  case  HSMHV_MOD_MUESTI1:
    mod->HSMHV_muesti1 = value->rValue;
    mod->HSMHV_muesti1_Given = TRUE;
    break;
  case  HSMHV_MOD_MUESTI2:
    mod->HSMHV_muesti2 = value->rValue;
    mod->HSMHV_muesti2_Given = TRUE;
    break;
  case  HSMHV_MOD_MUESTI3:
    mod->HSMHV_muesti3 = value->rValue;
    mod->HSMHV_muesti3_Given = TRUE;
    break;
  case  HSMHV_MOD_NSUBPSTI1:
    mod->HSMHV_nsubpsti1 = value->rValue;
    mod->HSMHV_nsubpsti1_Given = TRUE;
    break;
  case  HSMHV_MOD_NSUBPSTI2:
    mod->HSMHV_nsubpsti2 = value->rValue;
    mod->HSMHV_nsubpsti2_Given = TRUE;
    break;
  case  HSMHV_MOD_NSUBPSTI3:
    mod->HSMHV_nsubpsti3 = value->rValue;
    mod->HSMHV_nsubpsti3_Given = TRUE;
    break;
  case  HSMHV_MOD_LPEXT:
    mod->HSMHV_lpext = value->rValue;
    mod->HSMHV_lpext_Given = TRUE;
    break;
  case  HSMHV_MOD_NPEXT:
    mod->HSMHV_npext = value->rValue;
    mod->HSMHV_npext_Given = TRUE;
    break;
  case  HSMHV_MOD_SCP22:
    mod->HSMHV_scp22 = value->rValue;
    mod->HSMHV_scp22_Given = TRUE;
    break;
  case  HSMHV_MOD_SCP21:
    mod->HSMHV_scp21 = value->rValue;
    mod->HSMHV_scp21_Given = TRUE;
    break;
  case  HSMHV_MOD_BS1:
    mod->HSMHV_bs1 = value->rValue;
    mod->HSMHV_bs1_Given = TRUE;
    break;
  case  HSMHV_MOD_BS2:
    mod->HSMHV_bs2 = value->rValue;
    mod->HSMHV_bs2_Given = TRUE;
    break;
  case  HSMHV_MOD_CGSO:
    mod->HSMHV_cgso = value->rValue;
    mod->HSMHV_cgso_Given = TRUE;
    break;
  case  HSMHV_MOD_CGDO:
    mod->HSMHV_cgdo = value->rValue;
    mod->HSMHV_cgdo_Given = TRUE;
    break;
  case  HSMHV_MOD_CGBO:
    mod->HSMHV_cgbo = value->rValue;
    mod->HSMHV_cgbo_Given = TRUE;
    break;
  case  HSMHV_MOD_TPOLY:
    mod->HSMHV_tpoly = value->rValue;
    mod->HSMHV_tpoly_Given = TRUE;
    break;
  case  HSMHV_MOD_JS0:
    mod->HSMHV_js0 = value->rValue;
    mod->HSMHV_js0_Given = TRUE;
    break;
  case  HSMHV_MOD_JS0SW:
    mod->HSMHV_js0sw = value->rValue;
    mod->HSMHV_js0sw_Given = TRUE;
    break;
  case  HSMHV_MOD_NJ:
    mod->HSMHV_nj = value->rValue;
    mod->HSMHV_nj_Given = TRUE;
    break;
  case  HSMHV_MOD_NJSW:
    mod->HSMHV_njsw = value->rValue;
    mod->HSMHV_njsw_Given = TRUE;
    break;
  case  HSMHV_MOD_XTI:
    mod->HSMHV_xti = value->rValue;
    mod->HSMHV_xti_Given = TRUE;
    break;
  case  HSMHV_MOD_CJ:
    mod->HSMHV_cj = value->rValue;
    mod->HSMHV_cj_Given = TRUE;
    break;
  case  HSMHV_MOD_CJSW:
    mod->HSMHV_cjsw = value->rValue;
    mod->HSMHV_cjsw_Given = TRUE;
    break;
  case  HSMHV_MOD_CJSWG:
    mod->HSMHV_cjswg = value->rValue;
    mod->HSMHV_cjswg_Given = TRUE;
    break;
  case  HSMHV_MOD_MJ:
    mod->HSMHV_mj = value->rValue;
    mod->HSMHV_mj_Given = TRUE;
    break;
  case  HSMHV_MOD_MJSW:
    mod->HSMHV_mjsw = value->rValue;
    mod->HSMHV_mjsw_Given = TRUE;
    break;
  case  HSMHV_MOD_MJSWG:
    mod->HSMHV_mjswg = value->rValue;
    mod->HSMHV_mjswg_Given = TRUE;
    break;
  case  HSMHV_MOD_PB:
    mod->HSMHV_pb = value->rValue;
    mod->HSMHV_pb_Given = TRUE;
    break;
  case  HSMHV_MOD_PBSW:
    mod->HSMHV_pbsw = value->rValue;
    mod->HSMHV_pbsw_Given = TRUE;
    break;
  case  HSMHV_MOD_PBSWG:
    mod->HSMHV_pbswg = value->rValue;
    mod->HSMHV_pbswg_Given = TRUE;
    break;
  case  HSMHV_MOD_XTI2:
    mod->HSMHV_xti2 = value->rValue;
    mod->HSMHV_xti2_Given = TRUE;
    break;
  case  HSMHV_MOD_CISB:
    mod->HSMHV_cisb = value->rValue;
    mod->HSMHV_cisb_Given = TRUE;
    break;
  case  HSMHV_MOD_CVB:
    mod->HSMHV_cvb = value->rValue;
    mod->HSMHV_cvb_Given = TRUE;
    break;
  case  HSMHV_MOD_CTEMP:
    mod->HSMHV_ctemp = value->rValue;
    mod->HSMHV_ctemp_Given = TRUE;
    break;
  case  HSMHV_MOD_CISBK:
    mod->HSMHV_cisbk = value->rValue;
    mod->HSMHV_cisbk_Given = TRUE;
    break;
  case  HSMHV_MOD_CVBK:
    mod->HSMHV_cvbk = value->rValue;
    mod->HSMHV_cvbk_Given = TRUE;
    break;
  case  HSMHV_MOD_DIVX:
    mod->HSMHV_divx = value->rValue;
    mod->HSMHV_divx_Given = TRUE;
    break;
  case  HSMHV_MOD_CLM1:
    mod->HSMHV_clm1 = value->rValue;
    mod->HSMHV_clm1_Given = TRUE;
    break;
  case  HSMHV_MOD_CLM2:
    mod->HSMHV_clm2 = value->rValue;
    mod->HSMHV_clm2_Given = TRUE;
    break;
  case  HSMHV_MOD_CLM3:
    mod->HSMHV_clm3 = value->rValue;
    mod->HSMHV_clm3_Given = TRUE;
    break;
  case  HSMHV_MOD_CLM5:
    mod->HSMHV_clm5 = value->rValue;
    mod->HSMHV_clm5_Given = TRUE;
    break;
  case  HSMHV_MOD_CLM6:
    mod->HSMHV_clm6 = value->rValue;
    mod->HSMHV_clm6_Given = TRUE;
    break;
  case  HSMHV_MOD_MUETMP:
    mod->HSMHV_muetmp = value->rValue;
    mod->HSMHV_muetmp_Given = TRUE;
    break;
  case  HSMHV_MOD_VOVER:
    mod->HSMHV_vover = value->rValue;
    mod->HSMHV_vover_Given = TRUE;
    break;
  case  HSMHV_MOD_VOVERP:
    mod->HSMHV_voverp = value->rValue;
    mod->HSMHV_voverp_Given = TRUE;
    break;
  case  HSMHV_MOD_VOVERS:
    mod->HSMHV_vovers = value->rValue;
    mod->HSMHV_vovers_Given = TRUE;
    break;
  case  HSMHV_MOD_VOVERSP:
    mod->HSMHV_voversp = value->rValue;
    mod->HSMHV_voversp_Given = TRUE;
    break;
  case  HSMHV_MOD_WFC:
    mod->HSMHV_wfc = value->rValue;
    mod->HSMHV_wfc_Given = TRUE;
    break;
  case  HSMHV_MOD_NSUBCW:
    mod->HSMHV_nsubcw = value->rValue;
    mod->HSMHV_nsubcw_Given = TRUE;
    break;
  case  HSMHV_MOD_NSUBCWP:
    mod->HSMHV_nsubcwp = value->rValue;
    mod->HSMHV_nsubcwp_Given = TRUE;
    break;
  case  HSMHV_MOD_QME1:
    mod->HSMHV_qme1 = value->rValue;
    mod->HSMHV_qme1_Given = TRUE;
    break;
  case  HSMHV_MOD_QME2:
    mod->HSMHV_qme2 = value->rValue;
    mod->HSMHV_qme2_Given = TRUE;
    break;
  case  HSMHV_MOD_QME3:
    mod->HSMHV_qme3 = value->rValue;
    mod->HSMHV_qme3_Given = TRUE;
    break;
  case  HSMHV_MOD_GIDL1:
    mod->HSMHV_gidl1 = value->rValue;
    mod->HSMHV_gidl1_Given = TRUE;
    break;
  case  HSMHV_MOD_GIDL2:
    mod->HSMHV_gidl2 = value->rValue;
    mod->HSMHV_gidl2_Given = TRUE;
    break;
  case  HSMHV_MOD_GIDL3:
    mod->HSMHV_gidl3 = value->rValue;
    mod->HSMHV_gidl3_Given = TRUE;
    break;
  case  HSMHV_MOD_GIDL4:
    mod->HSMHV_gidl4 = value->rValue;
    mod->HSMHV_gidl4_Given = TRUE;
    break;
  case  HSMHV_MOD_GIDL5:
    mod->HSMHV_gidl5 = value->rValue;
    mod->HSMHV_gidl5_Given = TRUE;
    break;
  case  HSMHV_MOD_GLEAK1:
    mod->HSMHV_gleak1 = value->rValue;
    mod->HSMHV_gleak1_Given = TRUE;
    break;
  case  HSMHV_MOD_GLEAK2:
    mod->HSMHV_gleak2 = value->rValue;
    mod->HSMHV_gleak2_Given = TRUE;
    break;
  case  HSMHV_MOD_GLEAK3:
    mod->HSMHV_gleak3 = value->rValue;
    mod->HSMHV_gleak3_Given = TRUE;
    break;
  case  HSMHV_MOD_GLEAK4:
    mod->HSMHV_gleak4 = value->rValue;
    mod->HSMHV_gleak4_Given = TRUE;
    break;
  case  HSMHV_MOD_GLEAK5:
    mod->HSMHV_gleak5 = value->rValue;
    mod->HSMHV_gleak5_Given = TRUE;
    break;
  case  HSMHV_MOD_GLEAK6:
    mod->HSMHV_gleak6 = value->rValue;
    mod->HSMHV_gleak6_Given = TRUE;
    break;
  case  HSMHV_MOD_GLEAK7:
    mod->HSMHV_gleak7 = value->rValue;
    mod->HSMHV_gleak7_Given = TRUE;
    break;
  case  HSMHV_MOD_GLPART1:
    mod->HSMHV_glpart1 = value->rValue;
    mod->HSMHV_glpart1_Given = TRUE;
    break;
  case  HSMHV_MOD_GLKSD1:
    mod->HSMHV_glksd1 = value->rValue;
    mod->HSMHV_glksd1_Given = TRUE;
    break;
  case  HSMHV_MOD_GLKSD2:
    mod->HSMHV_glksd2 = value->rValue;
    mod->HSMHV_glksd2_Given = TRUE;
    break;
  case  HSMHV_MOD_GLKSD3:
    mod->HSMHV_glksd3 = value->rValue;
    mod->HSMHV_glksd3_Given = TRUE;
    break;
  case  HSMHV_MOD_GLKB1:
    mod->HSMHV_glkb1 = value->rValue;
    mod->HSMHV_glkb1_Given = TRUE;
    break;
  case  HSMHV_MOD_GLKB2:
    mod->HSMHV_glkb2 = value->rValue;
    mod->HSMHV_glkb2_Given = TRUE;
    break;
  case  HSMHV_MOD_GLKB3:
    mod->HSMHV_glkb3 = value->rValue;
    mod->HSMHV_glkb3_Given = TRUE;
    break;
  case  HSMHV_MOD_EGIG:
    mod->HSMHV_egig = value->rValue;
    mod->HSMHV_egig_Given = TRUE;
    break;
  case  HSMHV_MOD_IGTEMP2:
    mod->HSMHV_igtemp2 = value->rValue;
    mod->HSMHV_igtemp2_Given = TRUE;
    break;
  case  HSMHV_MOD_IGTEMP3:
    mod->HSMHV_igtemp3 = value->rValue;
    mod->HSMHV_igtemp3_Given = TRUE;
    break;
  case  HSMHV_MOD_VZADD0:
    mod->HSMHV_vzadd0 = value->rValue;
    mod->HSMHV_vzadd0_Given = TRUE;
    break;
  case  HSMHV_MOD_PZADD0:
    mod->HSMHV_pzadd0 = value->rValue;
    mod->HSMHV_pzadd0_Given = TRUE;
    break;
  case  HSMHV_MOD_NFTRP:
    mod->HSMHV_nftrp = value->rValue;
    mod->HSMHV_nftrp_Given = TRUE;
    break;
  case  HSMHV_MOD_NFALP:
    mod->HSMHV_nfalp = value->rValue;
    mod->HSMHV_nfalp_Given = TRUE;
    break;
  case  HSMHV_MOD_CIT:
    mod->HSMHV_cit = value->rValue;
    mod->HSMHV_cit_Given = TRUE;
    break;
  case  HSMHV_MOD_FALPH:
    mod->HSMHV_falph = value->rValue;
    mod->HSMHV_falph_Given = TRUE;
    break;
  case  HSMHV_MOD_KAPPA:
    mod->HSMHV_kappa = value->rValue;
    mod->HSMHV_kappa_Given = TRUE;
    break;
  case  HSMHV_MOD_PTHROU:
    mod->HSMHV_pthrou = value->rValue;
    mod->HSMHV_pthrou_Given = TRUE;
    break;
  case  HSMHV_MOD_VDIFFJ:
    mod->HSMHV_vdiffj = value->rValue;
    mod->HSMHV_vdiffj_Given = TRUE;
    break;
  case  HSMHV_MOD_DLY1:
    mod->HSMHV_dly1 = value->rValue;
    mod->HSMHV_dly1_Given = TRUE;
    break;
  case  HSMHV_MOD_DLY2:
    mod->HSMHV_dly2 = value->rValue;
    mod->HSMHV_dly2_Given = TRUE;
    break;
  case  HSMHV_MOD_DLY3:
    mod->HSMHV_dly3 = value->rValue;
    mod->HSMHV_dly3_Given = TRUE;
    break;
  case  HSMHV_MOD_TNOM:
    mod->HSMHV_tnom = value->rValue;
    mod->HSMHV_tnom_Given = TRUE;
    break;
  case  HSMHV_MOD_OVSLP:
    mod->HSMHV_ovslp = value->rValue;
    mod->HSMHV_ovslp_Given = TRUE;
    break;
  case  HSMHV_MOD_OVMAG:
    mod->HSMHV_ovmag = value->rValue;
    mod->HSMHV_ovmag_Given = TRUE;
    break;
  case  HSMHV_MOD_GBMIN:
    mod->HSMHV_gbmin = value->rValue;
    mod->HSMHV_gbmin_Given = TRUE;
    break;
  case  HSMHV_MOD_RBPB:
    mod->HSMHV_rbpb = value->rValue;
    mod->HSMHV_rbpb_Given = TRUE;
    break;
  case  HSMHV_MOD_RBPD:
    mod->HSMHV_rbpd = value->rValue;
    mod->HSMHV_rbpd_Given = TRUE;
    break;
  case  HSMHV_MOD_RBPS:
    mod->HSMHV_rbps = value->rValue;
    mod->HSMHV_rbps_Given = TRUE;
    break;
  case  HSMHV_MOD_RBDB:
    mod->HSMHV_rbdb = value->rValue;
    mod->HSMHV_rbdb_Given = TRUE;
    break;
  case  HSMHV_MOD_RBSB:
    mod->HSMHV_rbsb = value->rValue;
    mod->HSMHV_rbsb_Given = TRUE;
    break;
  case  HSMHV_MOD_IBPC1:
    mod->HSMHV_ibpc1 = value->rValue;
    mod->HSMHV_ibpc1_Given = TRUE;
    break;
  case  HSMHV_MOD_IBPC2:
    mod->HSMHV_ibpc2 = value->rValue;
    mod->HSMHV_ibpc2_Given = TRUE;
    break;
  case  HSMHV_MOD_MPHDFM:
    mod->HSMHV_mphdfm = value->rValue;
    mod->HSMHV_mphdfm_Given = TRUE;
    break;
  case  HSMHV_MOD_RDVG11:
    mod->HSMHV_rdvg11 = value->rValue;
    mod->HSMHV_rdvg11_Given = TRUE;
    break;
  case  HSMHV_MOD_RDVG12:
    mod->HSMHV_rdvg12 = value->rValue;
    mod->HSMHV_rdvg12_Given = TRUE;
    break;
  case  HSMHV_MOD_RD20:
    mod->HSMHV_rd20 = value->rValue;
    mod->HSMHV_rd20_Given = TRUE;
    break;
  case  HSMHV_MOD_QOVSM: 
    mod->HSMHV_qovsm = value->rValue;
    mod->HSMHV_qovsm_Given = TRUE;
    break;
  case  HSMHV_MOD_LDRIFT: 
    mod->HSMHV_ldrift = value->rValue;
    mod->HSMHV_ldrift_Given = TRUE;
    break;
  case  HSMHV_MOD_RD21:
    mod->HSMHV_rd21 = value->rValue;
    mod->HSMHV_rd21_Given = TRUE;
    break;
  case  HSMHV_MOD_RD22:
    mod->HSMHV_rd22 = value->rValue;
    mod->HSMHV_rd22_Given = TRUE;
    break;
  case  HSMHV_MOD_RD22D:
    mod->HSMHV_rd22d = value->rValue;
    mod->HSMHV_rd22d_Given = TRUE;
    break;
  case  HSMHV_MOD_RD23:
    mod->HSMHV_rd23 = value->rValue;
    mod->HSMHV_rd23_Given = TRUE;
    break;
  case  HSMHV_MOD_RD24:
    mod->HSMHV_rd24 = value->rValue;
    mod->HSMHV_rd24_Given = TRUE;
    break;
  case  HSMHV_MOD_RD25:
    mod->HSMHV_rd25 = value->rValue;
    mod->HSMHV_rd25_Given = TRUE;
    break;
  case  HSMHV_MOD_RD26:
    mod->HSMHV_rd26 = value->rValue;
    mod->HSMHV_rd26_Given = TRUE;
    break;
  case  HSMHV_MOD_RDVDL:
    mod->HSMHV_rdvdl = value->rValue;
    mod->HSMHV_rdvdl_Given = TRUE;
    break;
  case  HSMHV_MOD_RDVDLP:
    mod->HSMHV_rdvdlp = value->rValue;
    mod->HSMHV_rdvdlp_Given = TRUE;
    break;
  case  HSMHV_MOD_RDVDS:
    mod->HSMHV_rdvds = value->rValue;
    mod->HSMHV_rdvds_Given = TRUE;
    break;
  case  HSMHV_MOD_RDVDSP:
    mod->HSMHV_rdvdsp = value->rValue;
    mod->HSMHV_rdvdsp_Given = TRUE;
    break;
  case  HSMHV_MOD_RD23L:
    mod->HSMHV_rd23l = value->rValue;
    mod->HSMHV_rd23l_Given = TRUE;
    break;
  case  HSMHV_MOD_RD23LP:
    mod->HSMHV_rd23lp = value->rValue;
    mod->HSMHV_rd23lp_Given = TRUE;
    break;
  case  HSMHV_MOD_RD23S:
    mod->HSMHV_rd23s = value->rValue;
    mod->HSMHV_rd23s_Given = TRUE;
    break;
  case  HSMHV_MOD_RD23SP:
    mod->HSMHV_rd23sp = value->rValue;
    mod->HSMHV_rd23sp_Given = TRUE;
    break;
  case  HSMHV_MOD_RDS:
    mod->HSMHV_rds = value->rValue;
    mod->HSMHV_rds_Given = TRUE;
    break;
  case  HSMHV_MOD_RDSP:
    mod->HSMHV_rdsp = value->rValue;
    mod->HSMHV_rdsp_Given = TRUE;
    break;
  case  HSMHV_MOD_RTH0: /* Self-heating model */
    mod->HSMHV_rth0 = value->rValue;
    mod->HSMHV_rth0_Given = TRUE;
    break;
  case  HSMHV_MOD_CTH0: /* Self-heating model */
    mod->HSMHV_cth0 = value->rValue;
    mod->HSMHV_cth0_Given = TRUE;
    break;
  case  HSMHV_MOD_POWRAT: /* Self-heating model */
    mod->HSMHV_powrat = value->rValue;
    mod->HSMHV_powrat_Given = TRUE;
    break;
  case  HSMHV_MOD_TCJBD: /* Self-heating model */
    mod->HSMHV_tcjbd = value->rValue;
    mod->HSMHV_tcjbd_Given = TRUE;
    break;
  case  HSMHV_MOD_TCJBS: /* Self-heating model */
    mod->HSMHV_tcjbs = value->rValue;
    mod->HSMHV_tcjbs_Given = TRUE;
    break;
  case  HSMHV_MOD_TCJBDSW: /* Self-heating model */
    mod->HSMHV_tcjbdsw = value->rValue;
    mod->HSMHV_tcjbdsw_Given = TRUE;
    break;
  case  HSMHV_MOD_TCJBSSW: /* Self-heating model */
    mod->HSMHV_tcjbssw = value->rValue;
    mod->HSMHV_tcjbssw_Given = TRUE;
    break;
  case  HSMHV_MOD_TCJBDSWG: /* Self-heating model */
    mod->HSMHV_tcjbdswg = value->rValue;
    mod->HSMHV_tcjbdswg_Given = TRUE;
    break;
  case  HSMHV_MOD_TCJBSSWG: /* Self-heating model */
    mod->HSMHV_tcjbsswg = value->rValue;
    mod->HSMHV_tcjbsswg_Given = TRUE;
    break;

  case  HSMHV_MOD_DLYOV:
    mod->HSMHV_dlyov = value->rValue;
    mod->HSMHV_dlyov_Given = TRUE;
    break;
  case  HSMHV_MOD_QDFTVD:
    mod->HSMHV_qdftvd = value->rValue;
    mod->HSMHV_qdftvd_Given = TRUE;
    break;
  case  HSMHV_MOD_XLDLD:
    mod->HSMHV_xldld = value->rValue;
    mod->HSMHV_xldld_Given = TRUE;
    break;
  case  HSMHV_MOD_XWDLD:
    mod->HSMHV_xwdld = value->rValue;
    mod->HSMHV_xwdld_Given = TRUE;
    break;
  case  HSMHV_MOD_RDVD:
    mod->HSMHV_rdvd = value->rValue;
    mod->HSMHV_rdvd_Given = TRUE;
    break;

  case  HSMHV_MOD_RDTEMP1:
    mod->HSMHV_rdtemp1 = value->rValue;
    mod->HSMHV_rdtemp1_Given = TRUE;
    break;
  case  HSMHV_MOD_RDTEMP2:
    mod->HSMHV_rdtemp2 = value->rValue;
    mod->HSMHV_rdtemp2_Given = TRUE;
    break;
  case  HSMHV_MOD_RTH0R:
    mod->HSMHV_rth0r = value->rValue;
    mod->HSMHV_rth0r_Given = TRUE;
    break;
  case  HSMHV_MOD_RDVDTEMP1:
    mod->HSMHV_rdvdtemp1 = value->rValue;
    mod->HSMHV_rdvdtemp1_Given = TRUE;
    break;
  case  HSMHV_MOD_RDVDTEMP2:
    mod->HSMHV_rdvdtemp2 = value->rValue;
    mod->HSMHV_rdvdtemp2_Given = TRUE;
    break;
  case  HSMHV_MOD_RTH0W:
    mod->HSMHV_rth0w = value->rValue;
    mod->HSMHV_rth0w_Given = TRUE;
    break;
  case  HSMHV_MOD_RTH0WP:
    mod->HSMHV_rth0wp = value->rValue;
    mod->HSMHV_rth0wp_Given = TRUE;
    break;
  case  HSMHV_MOD_CVDSOVER:
    mod->HSMHV_cvdsover = value->rValue;
    mod->HSMHV_cvdsover_Given = TRUE;
    break;

  case  HSMHV_MOD_NINVD:
    mod->HSMHV_ninvd = value->rValue;
    mod->HSMHV_ninvd_Given = TRUE;
    break;
  case  HSMHV_MOD_NINVDW:
    mod->HSMHV_ninvdw = value->rValue;
    mod->HSMHV_ninvdw_Given = TRUE;
    break;
  case  HSMHV_MOD_NINVDWP:
    mod->HSMHV_ninvdwp = value->rValue;
    mod->HSMHV_ninvdwp_Given = TRUE;
    break;
  case  HSMHV_MOD_NINVDT1:
    mod->HSMHV_ninvdt1 = value->rValue;
    mod->HSMHV_ninvdt1_Given = TRUE;
    break;
  case  HSMHV_MOD_NINVDT2:
    mod->HSMHV_ninvdt2 = value->rValue;
    mod->HSMHV_ninvdt2_Given = TRUE;
    break;
  case  HSMHV_MOD_VBSMIN:
    mod->HSMHV_vbsmin = value->rValue;
    mod->HSMHV_vbsmin_Given = TRUE;
    break;
  case  HSMHV_MOD_RDVB:
    mod->HSMHV_rdvb = value->rValue;
    mod->HSMHV_rdvb_Given = TRUE;
    break;
  case  HSMHV_MOD_RTH0NF:
    mod->HSMHV_rth0nf = value->rValue;
    mod->HSMHV_rth0nf_Given = TRUE;
    break;
  case  HSMHV_MOD_RTHTEMP1:
    mod->HSMHV_rthtemp1 = value->rValue;
    mod->HSMHV_rthtemp1_Given = TRUE;
    break;
  case  HSMHV_MOD_RTHTEMP2:
    mod->HSMHV_rthtemp2 = value->rValue;
    mod->HSMHV_rthtemp2_Given = TRUE;
    break;
  case  HSMHV_MOD_PRATTEMP1:
    mod->HSMHV_prattemp1 = value->rValue;
    mod->HSMHV_prattemp1_Given = TRUE;
    break;
  case  HSMHV_MOD_PRATTEMP2:
    mod->HSMHV_prattemp2 = value->rValue;
    mod->HSMHV_prattemp2_Given = TRUE;
    break;

  case  HSMHV_MOD_RDVSUB: /* substrate effect */ 
    mod->HSMHV_rdvsub = value->rValue;
    mod->HSMHV_rdvsub_Given = TRUE;
    break;
  case  HSMHV_MOD_RDVDSUB:
    mod->HSMHV_rdvdsub = value->rValue;
    mod->HSMHV_rdvdsub_Given = TRUE;
    break;
  case  HSMHV_MOD_DDRIFT:
    mod->HSMHV_ddrift = value->rValue;
    mod->HSMHV_ddrift_Given = TRUE;
    break;
  case  HSMHV_MOD_VBISUB:
    mod->HSMHV_vbisub = value->rValue;
    mod->HSMHV_vbisub_Given = TRUE;
    break;
  case  HSMHV_MOD_NSUBSUB:
    mod->HSMHV_nsubsub = value->rValue;
    mod->HSMHV_nsubsub_Given = TRUE;
    break;
  case HSMHV_MOD_SHEMAX:
    mod->HSMHV_shemax = value->rValue;
    mod->HSMHV_shemax_Given = TRUE;
    break;



  /* binning parameters */
  case  HSMHV_MOD_LMIN:
    mod->HSMHV_lmin = value->rValue;
    mod->HSMHV_lmin_Given = TRUE;
    break;
  case  HSMHV_MOD_LMAX:
    mod->HSMHV_lmax = value->rValue;
    mod->HSMHV_lmax_Given = TRUE;
    break;
  case  HSMHV_MOD_WMIN:
    mod->HSMHV_wmin = value->rValue;
    mod->HSMHV_wmin_Given = TRUE;
    break;
  case  HSMHV_MOD_WMAX:
    mod->HSMHV_wmax = value->rValue;
    mod->HSMHV_wmax_Given = TRUE;
    break;
  case  HSMHV_MOD_LBINN:
    mod->HSMHV_lbinn = value->rValue;
    mod->HSMHV_lbinn_Given = TRUE;
    break;
  case  HSMHV_MOD_WBINN:
    mod->HSMHV_wbinn = value->rValue;
    mod->HSMHV_wbinn_Given = TRUE;
    break;

  /* Length dependence */
  case  HSMHV_MOD_LVMAX:
    mod->HSMHV_lvmax = value->rValue;
    mod->HSMHV_lvmax_Given = TRUE;
    break;
  case  HSMHV_MOD_LBGTMP1:
    mod->HSMHV_lbgtmp1 = value->rValue;
    mod->HSMHV_lbgtmp1_Given = TRUE;
    break;
  case  HSMHV_MOD_LBGTMP2:
    mod->HSMHV_lbgtmp2 = value->rValue;
    mod->HSMHV_lbgtmp2_Given = TRUE;
    break;
  case  HSMHV_MOD_LEG0:
    mod->HSMHV_leg0 = value->rValue;
    mod->HSMHV_leg0_Given = TRUE;
    break;
  case  HSMHV_MOD_LVFBOVER:
    mod->HSMHV_lvfbover = value->rValue;
    mod->HSMHV_lvfbover_Given = TRUE;
    break;
  case  HSMHV_MOD_LNOVER:
    mod->HSMHV_lnover = value->rValue;
    mod->HSMHV_lnover_Given = TRUE;
    break;
  case  HSMHV_MOD_LNOVERS:
    mod->HSMHV_lnovers = value->rValue;
    mod->HSMHV_lnovers_Given = TRUE;
    break;
  case  HSMHV_MOD_LWL2:
    mod->HSMHV_lwl2 = value->rValue;
    mod->HSMHV_lwl2_Given = TRUE;
    break;
  case  HSMHV_MOD_LVFBC:
    mod->HSMHV_lvfbc = value->rValue;
    mod->HSMHV_lvfbc_Given = TRUE;
    break;
  case  HSMHV_MOD_LNSUBC:
    mod->HSMHV_lnsubc = value->rValue;
    mod->HSMHV_lnsubc_Given = TRUE;
    break;
  case  HSMHV_MOD_LNSUBP:
    mod->HSMHV_lnsubp = value->rValue;
    mod->HSMHV_lnsubp_Given = TRUE;
    break;
  case  HSMHV_MOD_LSCP1:
    mod->HSMHV_lscp1 = value->rValue;
    mod->HSMHV_lscp1_Given = TRUE;
    break;
  case  HSMHV_MOD_LSCP2:
    mod->HSMHV_lscp2 = value->rValue;
    mod->HSMHV_lscp2_Given = TRUE;
    break;
  case  HSMHV_MOD_LSCP3:
    mod->HSMHV_lscp3 = value->rValue;
    mod->HSMHV_lscp3_Given = TRUE;
    break;
  case  HSMHV_MOD_LSC1:
    mod->HSMHV_lsc1 = value->rValue;
    mod->HSMHV_lsc1_Given = TRUE;
    break;
  case  HSMHV_MOD_LSC2:
    mod->HSMHV_lsc2 = value->rValue;
    mod->HSMHV_lsc2_Given = TRUE;
    break;
  case  HSMHV_MOD_LSC3:
    mod->HSMHV_lsc3 = value->rValue;
    mod->HSMHV_lsc3_Given = TRUE;
    break;
  case  HSMHV_MOD_LPGD1:
    mod->HSMHV_lpgd1 = value->rValue;
    mod->HSMHV_lpgd1_Given = TRUE;
    break;
  case  HSMHV_MOD_LPGD3:
    mod->HSMHV_lpgd3 = value->rValue;
    mod->HSMHV_lpgd3_Given = TRUE;
    break;
  case  HSMHV_MOD_LNDEP:
    mod->HSMHV_lndep = value->rValue;
    mod->HSMHV_lndep_Given = TRUE;
    break;
  case  HSMHV_MOD_LNINV:
    mod->HSMHV_lninv = value->rValue;
    mod->HSMHV_lninv_Given = TRUE;
    break;
  case  HSMHV_MOD_LMUECB0:
    mod->HSMHV_lmuecb0 = value->rValue;
    mod->HSMHV_lmuecb0_Given = TRUE;
    break;
  case  HSMHV_MOD_LMUECB1:
    mod->HSMHV_lmuecb1 = value->rValue;
    mod->HSMHV_lmuecb1_Given = TRUE;
    break;
  case  HSMHV_MOD_LMUEPH1:
    mod->HSMHV_lmueph1 = value->rValue;
    mod->HSMHV_lmueph1_Given = TRUE;
    break;
  case  HSMHV_MOD_LVTMP:
    mod->HSMHV_lvtmp = value->rValue;
    mod->HSMHV_lvtmp_Given = TRUE;
    break;
  case  HSMHV_MOD_LWVTH0:
    mod->HSMHV_lwvth0 = value->rValue;
    mod->HSMHV_lwvth0_Given = TRUE;
    break;
  case  HSMHV_MOD_LMUESR1:
    mod->HSMHV_lmuesr1 = value->rValue;
    mod->HSMHV_lmuesr1_Given = TRUE;
    break;
  case  HSMHV_MOD_LMUETMP:
    mod->HSMHV_lmuetmp = value->rValue;
    mod->HSMHV_lmuetmp_Given = TRUE;
    break;
  case  HSMHV_MOD_LSUB1:
    mod->HSMHV_lsub1 = value->rValue;
    mod->HSMHV_lsub1_Given = TRUE;
    break;
  case  HSMHV_MOD_LSUB2:
    mod->HSMHV_lsub2 = value->rValue;
    mod->HSMHV_lsub2_Given = TRUE;
    break;
  case  HSMHV_MOD_LSVDS:
    mod->HSMHV_lsvds = value->rValue;
    mod->HSMHV_lsvds_Given = TRUE;
    break;
  case  HSMHV_MOD_LSVBS:
    mod->HSMHV_lsvbs = value->rValue;
    mod->HSMHV_lsvbs_Given = TRUE;
    break;
  case  HSMHV_MOD_LSVGS:
    mod->HSMHV_lsvgs = value->rValue;
    mod->HSMHV_lsvgs_Given = TRUE;
    break;
  case  HSMHV_MOD_LFN1:
    mod->HSMHV_lfn1 = value->rValue;
    mod->HSMHV_lfn1_Given = TRUE;
    break;
  case  HSMHV_MOD_LFN2:
    mod->HSMHV_lfn2 = value->rValue;
    mod->HSMHV_lfn2_Given = TRUE;
    break;
  case  HSMHV_MOD_LFN3:
    mod->HSMHV_lfn3 = value->rValue;
    mod->HSMHV_lfn3_Given = TRUE;
    break;
  case  HSMHV_MOD_LFVBS:
    mod->HSMHV_lfvbs = value->rValue;
    mod->HSMHV_lfvbs_Given = TRUE;
    break;
  case  HSMHV_MOD_LNSTI:
    mod->HSMHV_lnsti = value->rValue;
    mod->HSMHV_lnsti_Given = TRUE;
    break;
  case  HSMHV_MOD_LWSTI:
    mod->HSMHV_lwsti = value->rValue;
    mod->HSMHV_lwsti_Given = TRUE;
    break;
  case  HSMHV_MOD_LSCSTI1:
    mod->HSMHV_lscsti1 = value->rValue;
    mod->HSMHV_lscsti1_Given = TRUE;
    break;
  case  HSMHV_MOD_LSCSTI2:
    mod->HSMHV_lscsti2 = value->rValue;
    mod->HSMHV_lscsti2_Given = TRUE;
    break;
  case  HSMHV_MOD_LVTHSTI:
    mod->HSMHV_lvthsti = value->rValue;
    mod->HSMHV_lvthsti_Given = TRUE;
    break;
  case  HSMHV_MOD_LMUESTI1:
    mod->HSMHV_lmuesti1 = value->rValue;
    mod->HSMHV_lmuesti1_Given = TRUE;
    break;
  case  HSMHV_MOD_LMUESTI2:
    mod->HSMHV_lmuesti2 = value->rValue;
    mod->HSMHV_lmuesti2_Given = TRUE;
    break;
  case  HSMHV_MOD_LMUESTI3:
    mod->HSMHV_lmuesti3 = value->rValue;
    mod->HSMHV_lmuesti3_Given = TRUE;
    break;
  case  HSMHV_MOD_LNSUBPSTI1:
    mod->HSMHV_lnsubpsti1 = value->rValue;
    mod->HSMHV_lnsubpsti1_Given = TRUE;
    break;
  case  HSMHV_MOD_LNSUBPSTI2:
    mod->HSMHV_lnsubpsti2 = value->rValue;
    mod->HSMHV_lnsubpsti2_Given = TRUE;
    break;
  case  HSMHV_MOD_LNSUBPSTI3:
    mod->HSMHV_lnsubpsti3 = value->rValue;
    mod->HSMHV_lnsubpsti3_Given = TRUE;
    break;
  case  HSMHV_MOD_LCGSO:
    mod->HSMHV_lcgso = value->rValue;
    mod->HSMHV_lcgso_Given = TRUE;
    break;
  case  HSMHV_MOD_LCGDO:
    mod->HSMHV_lcgdo = value->rValue;
    mod->HSMHV_lcgdo_Given = TRUE;
    break;
  case  HSMHV_MOD_LJS0:
    mod->HSMHV_ljs0 = value->rValue;
    mod->HSMHV_ljs0_Given = TRUE;
    break;
  case  HSMHV_MOD_LJS0SW:
    mod->HSMHV_ljs0sw = value->rValue;
    mod->HSMHV_ljs0sw_Given = TRUE;
    break;
  case  HSMHV_MOD_LNJ:
    mod->HSMHV_lnj = value->rValue;
    mod->HSMHV_lnj_Given = TRUE;
    break;
  case  HSMHV_MOD_LCISBK:
    mod->HSMHV_lcisbk = value->rValue;
    mod->HSMHV_lcisbk_Given = TRUE;
    break;
  case  HSMHV_MOD_LCLM1:
    mod->HSMHV_lclm1 = value->rValue;
    mod->HSMHV_lclm1_Given = TRUE;
    break;
  case  HSMHV_MOD_LCLM2:
    mod->HSMHV_lclm2 = value->rValue;
    mod->HSMHV_lclm2_Given = TRUE;
    break;
  case  HSMHV_MOD_LCLM3:
    mod->HSMHV_lclm3 = value->rValue;
    mod->HSMHV_lclm3_Given = TRUE;
    break;
  case  HSMHV_MOD_LWFC:
    mod->HSMHV_lwfc = value->rValue;
    mod->HSMHV_lwfc_Given = TRUE;
    break;
  case  HSMHV_MOD_LGIDL1:
    mod->HSMHV_lgidl1 = value->rValue;
    mod->HSMHV_lgidl1_Given = TRUE;
    break;
  case  HSMHV_MOD_LGIDL2:
    mod->HSMHV_lgidl2 = value->rValue;
    mod->HSMHV_lgidl2_Given = TRUE;
    break;
  case  HSMHV_MOD_LGLEAK1:
    mod->HSMHV_lgleak1 = value->rValue;
    mod->HSMHV_lgleak1_Given = TRUE;
    break;
  case  HSMHV_MOD_LGLEAK2:
    mod->HSMHV_lgleak2 = value->rValue;
    mod->HSMHV_lgleak2_Given = TRUE;
    break;
  case  HSMHV_MOD_LGLEAK3:
    mod->HSMHV_lgleak3 = value->rValue;
    mod->HSMHV_lgleak3_Given = TRUE;
    break;
  case  HSMHV_MOD_LGLEAK6:
    mod->HSMHV_lgleak6 = value->rValue;
    mod->HSMHV_lgleak6_Given = TRUE;
    break;
  case  HSMHV_MOD_LGLKSD1:
    mod->HSMHV_lglksd1 = value->rValue;
    mod->HSMHV_lglksd1_Given = TRUE;
    break;
  case  HSMHV_MOD_LGLKSD2:
    mod->HSMHV_lglksd2 = value->rValue;
    mod->HSMHV_lglksd2_Given = TRUE;
    break;
  case  HSMHV_MOD_LGLKB1:
    mod->HSMHV_lglkb1 = value->rValue;
    mod->HSMHV_lglkb1_Given = TRUE;
    break;
  case  HSMHV_MOD_LGLKB2:
    mod->HSMHV_lglkb2 = value->rValue;
    mod->HSMHV_lglkb2_Given = TRUE;
    break;
  case  HSMHV_MOD_LNFTRP:
    mod->HSMHV_lnftrp = value->rValue;
    mod->HSMHV_lnftrp_Given = TRUE;
    break;
  case  HSMHV_MOD_LNFALP:
    mod->HSMHV_lnfalp = value->rValue;
    mod->HSMHV_lnfalp_Given = TRUE;
    break;
  case  HSMHV_MOD_LPTHROU:
    mod->HSMHV_lpthrou = value->rValue;
    mod->HSMHV_lpthrou_Given = TRUE;
    break;
  case  HSMHV_MOD_LVDIFFJ:
    mod->HSMHV_lvdiffj = value->rValue;
    mod->HSMHV_lvdiffj_Given = TRUE;
    break;
  case  HSMHV_MOD_LIBPC1:
    mod->HSMHV_libpc1 = value->rValue;
    mod->HSMHV_libpc1_Given = TRUE;
    break;
  case  HSMHV_MOD_LIBPC2:
    mod->HSMHV_libpc2 = value->rValue;
    mod->HSMHV_libpc2_Given = TRUE;
    break;
    break;
  case  HSMHV_MOD_LCGBO:
    mod->HSMHV_lcgbo = value->rValue;
    mod->HSMHV_lcgbo_Given = TRUE;
    break;
  case  HSMHV_MOD_LCVDSOVER:
    mod->HSMHV_lcvdsover = value->rValue;
    mod->HSMHV_lcvdsover_Given = TRUE;
    break;
  case  HSMHV_MOD_LFALPH:
    mod->HSMHV_lfalph = value->rValue;
    mod->HSMHV_lfalph_Given = TRUE;
    break;
  case  HSMHV_MOD_LNPEXT:
    mod->HSMHV_lnpext = value->rValue;
    mod->HSMHV_lnpext_Given = TRUE;
    break;
  case  HSMHV_MOD_LPOWRAT:
    mod->HSMHV_lpowrat = value->rValue;
    mod->HSMHV_lpowrat_Given = TRUE;
    break;
  case  HSMHV_MOD_LRD:
    mod->HSMHV_lrd = value->rValue;
    mod->HSMHV_lrd_Given = TRUE;
    break;
  case  HSMHV_MOD_LRD22:
    mod->HSMHV_lrd22 = value->rValue;
    mod->HSMHV_lrd22_Given = TRUE;
    break;
  case  HSMHV_MOD_LRD23:
    mod->HSMHV_lrd23 = value->rValue;
    mod->HSMHV_lrd23_Given = TRUE;
    break;
  case  HSMHV_MOD_LRD24:
    mod->HSMHV_lrd24 = value->rValue;
    mod->HSMHV_lrd24_Given = TRUE;
    break;
  case  HSMHV_MOD_LRDICT1:
    mod->HSMHV_lrdict1 = value->rValue;
    mod->HSMHV_lrdict1_Given = TRUE;
    break;
  case  HSMHV_MOD_LRDOV13:
    mod->HSMHV_lrdov13 = value->rValue;
    mod->HSMHV_lrdov13_Given = TRUE;
    break;
  case  HSMHV_MOD_LRDSLP1:
    mod->HSMHV_lrdslp1 = value->rValue;
    mod->HSMHV_lrdslp1_Given = TRUE;
    break;
  case  HSMHV_MOD_LRDVB:
    mod->HSMHV_lrdvb = value->rValue;
    mod->HSMHV_lrdvb_Given = TRUE;
    break;
  case  HSMHV_MOD_LRDVD:
    mod->HSMHV_lrdvd = value->rValue;
    mod->HSMHV_lrdvd_Given = TRUE;
    break;
  case  HSMHV_MOD_LRDVG11:
    mod->HSMHV_lrdvg11 = value->rValue;
    mod->HSMHV_lrdvg11_Given = TRUE;
    break;
  case  HSMHV_MOD_LRS:
    mod->HSMHV_lrs = value->rValue;
    mod->HSMHV_lrs_Given = TRUE;
    break;
  case  HSMHV_MOD_LRTH0:
    mod->HSMHV_lrth0 = value->rValue;
    mod->HSMHV_lrth0_Given = TRUE;
    break;
  case  HSMHV_MOD_LVOVER:
    mod->HSMHV_lvover = value->rValue;
    mod->HSMHV_lvover_Given = TRUE;
    break;

  /* Width dependence */
  case  HSMHV_MOD_WVMAX:
    mod->HSMHV_wvmax = value->rValue;
    mod->HSMHV_wvmax_Given = TRUE;
    break;
  case  HSMHV_MOD_WBGTMP1:
    mod->HSMHV_wbgtmp1 = value->rValue;
    mod->HSMHV_wbgtmp1_Given = TRUE;
    break;
  case  HSMHV_MOD_WBGTMP2:
    mod->HSMHV_wbgtmp2 = value->rValue;
    mod->HSMHV_wbgtmp2_Given = TRUE;
    break;
  case  HSMHV_MOD_WEG0:
    mod->HSMHV_weg0 = value->rValue;
    mod->HSMHV_weg0_Given = TRUE;
    break;
  case  HSMHV_MOD_WVFBOVER:
    mod->HSMHV_wvfbover = value->rValue;
    mod->HSMHV_wvfbover_Given = TRUE;
    break;
  case  HSMHV_MOD_WNOVER:
    mod->HSMHV_wnover = value->rValue;
    mod->HSMHV_wnover_Given = TRUE;
    break;
  case  HSMHV_MOD_WNOVERS:
    mod->HSMHV_wnovers = value->rValue;
    mod->HSMHV_wnovers_Given = TRUE;
    break;
  case  HSMHV_MOD_WWL2:
    mod->HSMHV_wwl2 = value->rValue;
    mod->HSMHV_wwl2_Given = TRUE;
    break;
  case  HSMHV_MOD_WVFBC:
    mod->HSMHV_wvfbc = value->rValue;
    mod->HSMHV_wvfbc_Given = TRUE;
    break;
  case  HSMHV_MOD_WNSUBC:
    mod->HSMHV_wnsubc = value->rValue;
    mod->HSMHV_wnsubc_Given = TRUE;
    break;
  case  HSMHV_MOD_WNSUBP:
    mod->HSMHV_wnsubp = value->rValue;
    mod->HSMHV_wnsubp_Given = TRUE;
    break;
  case  HSMHV_MOD_WSCP1:
    mod->HSMHV_wscp1 = value->rValue;
    mod->HSMHV_wscp1_Given = TRUE;
    break;
  case  HSMHV_MOD_WSCP2:
    mod->HSMHV_wscp2 = value->rValue;
    mod->HSMHV_wscp2_Given = TRUE;
    break;
  case  HSMHV_MOD_WSCP3:
    mod->HSMHV_wscp3 = value->rValue;
    mod->HSMHV_wscp3_Given = TRUE;
    break;
  case  HSMHV_MOD_WSC1:
    mod->HSMHV_wsc1 = value->rValue;
    mod->HSMHV_wsc1_Given = TRUE;
    break;
  case  HSMHV_MOD_WSC2:
    mod->HSMHV_wsc2 = value->rValue;
    mod->HSMHV_wsc2_Given = TRUE;
    break;
  case  HSMHV_MOD_WSC3:
    mod->HSMHV_wsc3 = value->rValue;
    mod->HSMHV_wsc3_Given = TRUE;
    break;
  case  HSMHV_MOD_WPGD1:
    mod->HSMHV_wpgd1 = value->rValue;
    mod->HSMHV_wpgd1_Given = TRUE;
    break;
  case  HSMHV_MOD_WPGD3:
    mod->HSMHV_wpgd3 = value->rValue;
    mod->HSMHV_wpgd3_Given = TRUE;
    break;
  case  HSMHV_MOD_WNDEP:
    mod->HSMHV_wndep = value->rValue;
    mod->HSMHV_wndep_Given = TRUE;
    break;
  case  HSMHV_MOD_WNINV:
    mod->HSMHV_wninv = value->rValue;
    mod->HSMHV_wninv_Given = TRUE;
    break;
  case  HSMHV_MOD_WMUECB0:
    mod->HSMHV_wmuecb0 = value->rValue;
    mod->HSMHV_wmuecb0_Given = TRUE;
    break;
  case  HSMHV_MOD_WMUECB1:
    mod->HSMHV_wmuecb1 = value->rValue;
    mod->HSMHV_wmuecb1_Given = TRUE;
    break;
  case  HSMHV_MOD_WMUEPH1:
    mod->HSMHV_wmueph1 = value->rValue;
    mod->HSMHV_wmueph1_Given = TRUE;
    break;
  case  HSMHV_MOD_WVTMP:
    mod->HSMHV_wvtmp = value->rValue;
    mod->HSMHV_wvtmp_Given = TRUE;
    break;
  case  HSMHV_MOD_WWVTH0:
    mod->HSMHV_wwvth0 = value->rValue;
    mod->HSMHV_wwvth0_Given = TRUE;
    break;
  case  HSMHV_MOD_WMUESR1:
    mod->HSMHV_wmuesr1 = value->rValue;
    mod->HSMHV_wmuesr1_Given = TRUE;
    break;
  case  HSMHV_MOD_WMUETMP:
    mod->HSMHV_wmuetmp = value->rValue;
    mod->HSMHV_wmuetmp_Given = TRUE;
    break;
  case  HSMHV_MOD_WSUB1:
    mod->HSMHV_wsub1 = value->rValue;
    mod->HSMHV_wsub1_Given = TRUE;
    break;
  case  HSMHV_MOD_WSUB2:
    mod->HSMHV_wsub2 = value->rValue;
    mod->HSMHV_wsub2_Given = TRUE;
    break;
  case  HSMHV_MOD_WSVDS:
    mod->HSMHV_wsvds = value->rValue;
    mod->HSMHV_wsvds_Given = TRUE;
    break;
  case  HSMHV_MOD_WSVBS:
    mod->HSMHV_wsvbs = value->rValue;
    mod->HSMHV_wsvbs_Given = TRUE;
    break;
  case  HSMHV_MOD_WSVGS:
    mod->HSMHV_wsvgs = value->rValue;
    mod->HSMHV_wsvgs_Given = TRUE;
    break;
  case  HSMHV_MOD_WFN1:
    mod->HSMHV_wfn1 = value->rValue;
    mod->HSMHV_wfn1_Given = TRUE;
    break;
  case  HSMHV_MOD_WFN2:
    mod->HSMHV_wfn2 = value->rValue;
    mod->HSMHV_wfn2_Given = TRUE;
    break;
  case  HSMHV_MOD_WFN3:
    mod->HSMHV_wfn3 = value->rValue;
    mod->HSMHV_wfn3_Given = TRUE;
    break;
  case  HSMHV_MOD_WFVBS:
    mod->HSMHV_wfvbs = value->rValue;
    mod->HSMHV_wfvbs_Given = TRUE;
    break;
  case  HSMHV_MOD_WNSTI:
    mod->HSMHV_wnsti = value->rValue;
    mod->HSMHV_wnsti_Given = TRUE;
    break;
  case  HSMHV_MOD_WWSTI:
    mod->HSMHV_wwsti = value->rValue;
    mod->HSMHV_wwsti_Given = TRUE;
    break;
  case  HSMHV_MOD_WSCSTI1:
    mod->HSMHV_wscsti1 = value->rValue;
    mod->HSMHV_wscsti1_Given = TRUE;
    break;
  case  HSMHV_MOD_WSCSTI2:
    mod->HSMHV_wscsti2 = value->rValue;
    mod->HSMHV_wscsti2_Given = TRUE;
    break;
  case  HSMHV_MOD_WVTHSTI:
    mod->HSMHV_wvthsti = value->rValue;
    mod->HSMHV_wvthsti_Given = TRUE;
    break;
  case  HSMHV_MOD_WMUESTI1:
    mod->HSMHV_wmuesti1 = value->rValue;
    mod->HSMHV_wmuesti1_Given = TRUE;
    break;
  case  HSMHV_MOD_WMUESTI2:
    mod->HSMHV_wmuesti2 = value->rValue;
    mod->HSMHV_wmuesti2_Given = TRUE;
    break;
  case  HSMHV_MOD_WMUESTI3:
    mod->HSMHV_wmuesti3 = value->rValue;
    mod->HSMHV_wmuesti3_Given = TRUE;
    break;
  case  HSMHV_MOD_WNSUBPSTI1:
    mod->HSMHV_wnsubpsti1 = value->rValue;
    mod->HSMHV_wnsubpsti1_Given = TRUE;
    break;
  case  HSMHV_MOD_WNSUBPSTI2:
    mod->HSMHV_wnsubpsti2 = value->rValue;
    mod->HSMHV_wnsubpsti2_Given = TRUE;
    break;
  case  HSMHV_MOD_WNSUBPSTI3:
    mod->HSMHV_wnsubpsti3 = value->rValue;
    mod->HSMHV_wnsubpsti3_Given = TRUE;
    break;
  case  HSMHV_MOD_WCGSO:
    mod->HSMHV_wcgso = value->rValue;
    mod->HSMHV_wcgso_Given = TRUE;
    break;
  case  HSMHV_MOD_WCGDO:
    mod->HSMHV_wcgdo = value->rValue;
    mod->HSMHV_wcgdo_Given = TRUE;
    break;
  case  HSMHV_MOD_WJS0:
    mod->HSMHV_wjs0 = value->rValue;
    mod->HSMHV_wjs0_Given = TRUE;
    break;
  case  HSMHV_MOD_WJS0SW:
    mod->HSMHV_wjs0sw = value->rValue;
    mod->HSMHV_wjs0sw_Given = TRUE;
    break;
  case  HSMHV_MOD_WNJ:
    mod->HSMHV_wnj = value->rValue;
    mod->HSMHV_wnj_Given = TRUE;
    break;
  case  HSMHV_MOD_WCISBK:
    mod->HSMHV_wcisbk = value->rValue;
    mod->HSMHV_wcisbk_Given = TRUE;
    break;
  case  HSMHV_MOD_WCLM1:
    mod->HSMHV_wclm1 = value->rValue;
    mod->HSMHV_wclm1_Given = TRUE;
    break;
  case  HSMHV_MOD_WCLM2:
    mod->HSMHV_wclm2 = value->rValue;
    mod->HSMHV_wclm2_Given = TRUE;
    break;
  case  HSMHV_MOD_WCLM3:
    mod->HSMHV_wclm3 = value->rValue;
    mod->HSMHV_wclm3_Given = TRUE;
    break;
  case  HSMHV_MOD_WWFC:
    mod->HSMHV_wwfc = value->rValue;
    mod->HSMHV_wwfc_Given = TRUE;
    break;
  case  HSMHV_MOD_WGIDL1:
    mod->HSMHV_wgidl1 = value->rValue;
    mod->HSMHV_wgidl1_Given = TRUE;
    break;
  case  HSMHV_MOD_WGIDL2:
    mod->HSMHV_wgidl2 = value->rValue;
    mod->HSMHV_wgidl2_Given = TRUE;
    break;
  case  HSMHV_MOD_WGLEAK1:
    mod->HSMHV_wgleak1 = value->rValue;
    mod->HSMHV_wgleak1_Given = TRUE;
    break;
  case  HSMHV_MOD_WGLEAK2:
    mod->HSMHV_wgleak2 = value->rValue;
    mod->HSMHV_wgleak2_Given = TRUE;
    break;
  case  HSMHV_MOD_WGLEAK3:
    mod->HSMHV_wgleak3 = value->rValue;
    mod->HSMHV_wgleak3_Given = TRUE;
    break;
  case  HSMHV_MOD_WGLEAK6:
    mod->HSMHV_wgleak6 = value->rValue;
    mod->HSMHV_wgleak6_Given = TRUE;
    break;
  case  HSMHV_MOD_WGLKSD1:
    mod->HSMHV_wglksd1 = value->rValue;
    mod->HSMHV_wglksd1_Given = TRUE;
    break;
  case  HSMHV_MOD_WGLKSD2:
    mod->HSMHV_wglksd2 = value->rValue;
    mod->HSMHV_wglksd2_Given = TRUE;
    break;
  case  HSMHV_MOD_WGLKB1:
    mod->HSMHV_wglkb1 = value->rValue;
    mod->HSMHV_wglkb1_Given = TRUE;
    break;
  case  HSMHV_MOD_WGLKB2:
    mod->HSMHV_wglkb2 = value->rValue;
    mod->HSMHV_wglkb2_Given = TRUE;
    break;
  case  HSMHV_MOD_WNFTRP:
    mod->HSMHV_wnftrp = value->rValue;
    mod->HSMHV_wnftrp_Given = TRUE;
    break;
  case  HSMHV_MOD_WNFALP:
    mod->HSMHV_wnfalp = value->rValue;
    mod->HSMHV_wnfalp_Given = TRUE;
    break;
  case  HSMHV_MOD_WPTHROU:
    mod->HSMHV_wpthrou = value->rValue;
    mod->HSMHV_wpthrou_Given = TRUE;
    break;
  case  HSMHV_MOD_WVDIFFJ:
    mod->HSMHV_wvdiffj = value->rValue;
    mod->HSMHV_wvdiffj_Given = TRUE;
    break;
  case  HSMHV_MOD_WIBPC1:
    mod->HSMHV_wibpc1 = value->rValue;
    mod->HSMHV_wibpc1_Given = TRUE;
    break;
  case  HSMHV_MOD_WIBPC2:
    mod->HSMHV_wibpc2 = value->rValue;
    mod->HSMHV_wibpc2_Given = TRUE;
    break;
    break;
  case  HSMHV_MOD_WCGBO:
    mod->HSMHV_wcgbo = value->rValue;
    mod->HSMHV_wcgbo_Given = TRUE;
    break;
  case  HSMHV_MOD_WCVDSOVER:
    mod->HSMHV_wcvdsover = value->rValue;
    mod->HSMHV_wcvdsover_Given = TRUE;
    break;
  case  HSMHV_MOD_WFALPH:
    mod->HSMHV_wfalph = value->rValue;
    mod->HSMHV_wfalph_Given = TRUE;
    break;
  case  HSMHV_MOD_WNPEXT:
    mod->HSMHV_wnpext = value->rValue;
    mod->HSMHV_wnpext_Given = TRUE;
    break;
  case  HSMHV_MOD_WPOWRAT:
    mod->HSMHV_wpowrat = value->rValue;
    mod->HSMHV_wpowrat_Given = TRUE;
    break;
  case  HSMHV_MOD_WRD:
    mod->HSMHV_wrd = value->rValue;
    mod->HSMHV_wrd_Given = TRUE;
    break;
  case  HSMHV_MOD_WRD22:
    mod->HSMHV_wrd22 = value->rValue;
    mod->HSMHV_wrd22_Given = TRUE;
    break;
  case  HSMHV_MOD_WRD23:
    mod->HSMHV_wrd23 = value->rValue;
    mod->HSMHV_wrd23_Given = TRUE;
    break;
  case  HSMHV_MOD_WRD24:
    mod->HSMHV_wrd24 = value->rValue;
    mod->HSMHV_wrd24_Given = TRUE;
    break;
  case  HSMHV_MOD_WRDICT1:
    mod->HSMHV_wrdict1 = value->rValue;
    mod->HSMHV_wrdict1_Given = TRUE;
    break;
  case  HSMHV_MOD_WRDOV13:
    mod->HSMHV_wrdov13 = value->rValue;
    mod->HSMHV_wrdov13_Given = TRUE;
    break;
  case  HSMHV_MOD_WRDSLP1:
    mod->HSMHV_wrdslp1 = value->rValue;
    mod->HSMHV_wrdslp1_Given = TRUE;
    break;
  case  HSMHV_MOD_WRDVB:
    mod->HSMHV_wrdvb = value->rValue;
    mod->HSMHV_wrdvb_Given = TRUE;
    break;
  case  HSMHV_MOD_WRDVD:
    mod->HSMHV_wrdvd = value->rValue;
    mod->HSMHV_wrdvd_Given = TRUE;
    break;
  case  HSMHV_MOD_WRDVG11:
    mod->HSMHV_wrdvg11 = value->rValue;
    mod->HSMHV_wrdvg11_Given = TRUE;
    break;
  case  HSMHV_MOD_WRS:
    mod->HSMHV_wrs = value->rValue;
    mod->HSMHV_wrs_Given = TRUE;
    break;
  case  HSMHV_MOD_WRTH0:
    mod->HSMHV_wrth0 = value->rValue;
    mod->HSMHV_wrth0_Given = TRUE;
    break;
  case  HSMHV_MOD_WVOVER:
    mod->HSMHV_wvover = value->rValue;
    mod->HSMHV_wvover_Given = TRUE;
    break;

  /* Cross-term dependence */
  case  HSMHV_MOD_PVMAX:
    mod->HSMHV_pvmax = value->rValue;
    mod->HSMHV_pvmax_Given = TRUE;
    break;
  case  HSMHV_MOD_PBGTMP1:
    mod->HSMHV_pbgtmp1 = value->rValue;
    mod->HSMHV_pbgtmp1_Given = TRUE;
    break;
  case  HSMHV_MOD_PBGTMP2:
    mod->HSMHV_pbgtmp2 = value->rValue;
    mod->HSMHV_pbgtmp2_Given = TRUE;
    break;
  case  HSMHV_MOD_PEG0:
    mod->HSMHV_peg0 = value->rValue;
    mod->HSMHV_peg0_Given = TRUE;
    break;
  case  HSMHV_MOD_PVFBOVER:
    mod->HSMHV_pvfbover = value->rValue;
    mod->HSMHV_pvfbover_Given = TRUE;
    break;
  case  HSMHV_MOD_PNOVER:
    mod->HSMHV_pnover = value->rValue;
    mod->HSMHV_pnover_Given = TRUE;
    break;
  case  HSMHV_MOD_PNOVERS:
    mod->HSMHV_pnovers = value->rValue;
    mod->HSMHV_pnovers_Given = TRUE;
    break;
  case  HSMHV_MOD_PWL2:
    mod->HSMHV_pwl2 = value->rValue;
    mod->HSMHV_pwl2_Given = TRUE;
    break;
  case  HSMHV_MOD_PVFBC:
    mod->HSMHV_pvfbc = value->rValue;
    mod->HSMHV_pvfbc_Given = TRUE;
    break;
  case  HSMHV_MOD_PNSUBC:
    mod->HSMHV_pnsubc = value->rValue;
    mod->HSMHV_pnsubc_Given = TRUE;
    break;
  case  HSMHV_MOD_PNSUBP:
    mod->HSMHV_pnsubp = value->rValue;
    mod->HSMHV_pnsubp_Given = TRUE;
    break;
  case  HSMHV_MOD_PSCP1:
    mod->HSMHV_pscp1 = value->rValue;
    mod->HSMHV_pscp1_Given = TRUE;
    break;
  case  HSMHV_MOD_PSCP2:
    mod->HSMHV_pscp2 = value->rValue;
    mod->HSMHV_pscp2_Given = TRUE;
    break;
  case  HSMHV_MOD_PSCP3:
    mod->HSMHV_pscp3 = value->rValue;
    mod->HSMHV_pscp3_Given = TRUE;
    break;
  case  HSMHV_MOD_PSC1:
    mod->HSMHV_psc1 = value->rValue;
    mod->HSMHV_psc1_Given = TRUE;
    break;
  case  HSMHV_MOD_PSC2:
    mod->HSMHV_psc2 = value->rValue;
    mod->HSMHV_psc2_Given = TRUE;
    break;
  case  HSMHV_MOD_PSC3:
    mod->HSMHV_psc3 = value->rValue;
    mod->HSMHV_psc3_Given = TRUE;
    break;
  case  HSMHV_MOD_PPGD1:
    mod->HSMHV_ppgd1 = value->rValue;
    mod->HSMHV_ppgd1_Given = TRUE;
    break;
  case  HSMHV_MOD_PPGD3:
    mod->HSMHV_ppgd3 = value->rValue;
    mod->HSMHV_ppgd3_Given = TRUE;
    break;
  case  HSMHV_MOD_PNDEP:
    mod->HSMHV_pndep = value->rValue;
    mod->HSMHV_pndep_Given = TRUE;
    break;
  case  HSMHV_MOD_PNINV:
    mod->HSMHV_pninv = value->rValue;
    mod->HSMHV_pninv_Given = TRUE;
    break;
  case  HSMHV_MOD_PMUECB0:
    mod->HSMHV_pmuecb0 = value->rValue;
    mod->HSMHV_pmuecb0_Given = TRUE;
    break;
  case  HSMHV_MOD_PMUECB1:
    mod->HSMHV_pmuecb1 = value->rValue;
    mod->HSMHV_pmuecb1_Given = TRUE;
    break;
  case  HSMHV_MOD_PMUEPH1:
    mod->HSMHV_pmueph1 = value->rValue;
    mod->HSMHV_pmueph1_Given = TRUE;
    break;
  case  HSMHV_MOD_PVTMP:
    mod->HSMHV_pvtmp = value->rValue;
    mod->HSMHV_pvtmp_Given = TRUE;
    break;
  case  HSMHV_MOD_PWVTH0:
    mod->HSMHV_pwvth0 = value->rValue;
    mod->HSMHV_pwvth0_Given = TRUE;
    break;
  case  HSMHV_MOD_PMUESR1:
    mod->HSMHV_pmuesr1 = value->rValue;
    mod->HSMHV_pmuesr1_Given = TRUE;
    break;
  case  HSMHV_MOD_PMUETMP:
    mod->HSMHV_pmuetmp = value->rValue;
    mod->HSMHV_pmuetmp_Given = TRUE;
    break;
  case  HSMHV_MOD_PSUB1:
    mod->HSMHV_psub1 = value->rValue;
    mod->HSMHV_psub1_Given = TRUE;
    break;
  case  HSMHV_MOD_PSUB2:
    mod->HSMHV_psub2 = value->rValue;
    mod->HSMHV_psub2_Given = TRUE;
    break;
  case  HSMHV_MOD_PSVDS:
    mod->HSMHV_psvds = value->rValue;
    mod->HSMHV_psvds_Given = TRUE;
    break;
  case  HSMHV_MOD_PSVBS:
    mod->HSMHV_psvbs = value->rValue;
    mod->HSMHV_psvbs_Given = TRUE;
    break;
  case  HSMHV_MOD_PSVGS:
    mod->HSMHV_psvgs = value->rValue;
    mod->HSMHV_psvgs_Given = TRUE;
    break;
  case  HSMHV_MOD_PFN1:
    mod->HSMHV_pfn1 = value->rValue;
    mod->HSMHV_pfn1_Given = TRUE;
    break;
  case  HSMHV_MOD_PFN2:
    mod->HSMHV_pfn2 = value->rValue;
    mod->HSMHV_pfn2_Given = TRUE;
    break;
  case  HSMHV_MOD_PFN3:
    mod->HSMHV_pfn3 = value->rValue;
    mod->HSMHV_pfn3_Given = TRUE;
    break;
  case  HSMHV_MOD_PFVBS:
    mod->HSMHV_pfvbs = value->rValue;
    mod->HSMHV_pfvbs_Given = TRUE;
    break;
  case  HSMHV_MOD_PNSTI:
    mod->HSMHV_pnsti = value->rValue;
    mod->HSMHV_pnsti_Given = TRUE;
    break;
  case  HSMHV_MOD_PWSTI:
    mod->HSMHV_pwsti = value->rValue;
    mod->HSMHV_pwsti_Given = TRUE;
    break;
  case  HSMHV_MOD_PSCSTI1:
    mod->HSMHV_pscsti1 = value->rValue;
    mod->HSMHV_pscsti1_Given = TRUE;
    break;
  case  HSMHV_MOD_PSCSTI2:
    mod->HSMHV_pscsti2 = value->rValue;
    mod->HSMHV_pscsti2_Given = TRUE;
    break;
  case  HSMHV_MOD_PVTHSTI:
    mod->HSMHV_pvthsti = value->rValue;
    mod->HSMHV_pvthsti_Given = TRUE;
    break;
  case  HSMHV_MOD_PMUESTI1:
    mod->HSMHV_pmuesti1 = value->rValue;
    mod->HSMHV_pmuesti1_Given = TRUE;
    break;
  case  HSMHV_MOD_PMUESTI2:
    mod->HSMHV_pmuesti2 = value->rValue;
    mod->HSMHV_pmuesti2_Given = TRUE;
    break;
  case  HSMHV_MOD_PMUESTI3:
    mod->HSMHV_pmuesti3 = value->rValue;
    mod->HSMHV_pmuesti3_Given = TRUE;
    break;
  case  HSMHV_MOD_PNSUBPSTI1:
    mod->HSMHV_pnsubpsti1 = value->rValue;
    mod->HSMHV_pnsubpsti1_Given = TRUE;
    break;
  case  HSMHV_MOD_PNSUBPSTI2:
    mod->HSMHV_pnsubpsti2 = value->rValue;
    mod->HSMHV_pnsubpsti2_Given = TRUE;
    break;
  case  HSMHV_MOD_PNSUBPSTI3:
    mod->HSMHV_pnsubpsti3 = value->rValue;
    mod->HSMHV_pnsubpsti3_Given = TRUE;
    break;
  case  HSMHV_MOD_PCGSO:
    mod->HSMHV_pcgso = value->rValue;
    mod->HSMHV_pcgso_Given = TRUE;
    break;
  case  HSMHV_MOD_PCGDO:
    mod->HSMHV_pcgdo = value->rValue;
    mod->HSMHV_pcgdo_Given = TRUE;
    break;
  case  HSMHV_MOD_PJS0:
    mod->HSMHV_pjs0 = value->rValue;
    mod->HSMHV_pjs0_Given = TRUE;
    break;
  case  HSMHV_MOD_PJS0SW:
    mod->HSMHV_pjs0sw = value->rValue;
    mod->HSMHV_pjs0sw_Given = TRUE;
    break;
  case  HSMHV_MOD_PNJ:
    mod->HSMHV_pnj = value->rValue;
    mod->HSMHV_pnj_Given = TRUE;
    break;
  case  HSMHV_MOD_PCISBK:
    mod->HSMHV_pcisbk = value->rValue;
    mod->HSMHV_pcisbk_Given = TRUE;
    break;
  case  HSMHV_MOD_PCLM1:
    mod->HSMHV_pclm1 = value->rValue;
    mod->HSMHV_pclm1_Given = TRUE;
    break;
  case  HSMHV_MOD_PCLM2:
    mod->HSMHV_pclm2 = value->rValue;
    mod->HSMHV_pclm2_Given = TRUE;
    break;
  case  HSMHV_MOD_PCLM3:
    mod->HSMHV_pclm3 = value->rValue;
    mod->HSMHV_pclm3_Given = TRUE;
    break;
  case  HSMHV_MOD_PWFC:
    mod->HSMHV_pwfc = value->rValue;
    mod->HSMHV_pwfc_Given = TRUE;
    break;
  case  HSMHV_MOD_PGIDL1:
    mod->HSMHV_pgidl1 = value->rValue;
    mod->HSMHV_pgidl1_Given = TRUE;
    break;
  case  HSMHV_MOD_PGIDL2:
    mod->HSMHV_pgidl2 = value->rValue;
    mod->HSMHV_pgidl2_Given = TRUE;
    break;
  case  HSMHV_MOD_PGLEAK1:
    mod->HSMHV_pgleak1 = value->rValue;
    mod->HSMHV_pgleak1_Given = TRUE;
    break;
  case  HSMHV_MOD_PGLEAK2:
    mod->HSMHV_pgleak2 = value->rValue;
    mod->HSMHV_pgleak2_Given = TRUE;
    break;
  case  HSMHV_MOD_PGLEAK3:
    mod->HSMHV_pgleak3 = value->rValue;
    mod->HSMHV_pgleak3_Given = TRUE;
    break;
  case  HSMHV_MOD_PGLEAK6:
    mod->HSMHV_pgleak6 = value->rValue;
    mod->HSMHV_pgleak6_Given = TRUE;
    break;
  case  HSMHV_MOD_PGLKSD1:
    mod->HSMHV_pglksd1 = value->rValue;
    mod->HSMHV_pglksd1_Given = TRUE;
    break;
  case  HSMHV_MOD_PGLKSD2:
    mod->HSMHV_pglksd2 = value->rValue;
    mod->HSMHV_pglksd2_Given = TRUE;
    break;
  case  HSMHV_MOD_PGLKB1:
    mod->HSMHV_pglkb1 = value->rValue;
    mod->HSMHV_pglkb1_Given = TRUE;
    break;
  case  HSMHV_MOD_PGLKB2:
    mod->HSMHV_pglkb2 = value->rValue;
    mod->HSMHV_pglkb2_Given = TRUE;
    break;
  case  HSMHV_MOD_PNFTRP:
    mod->HSMHV_pnftrp = value->rValue;
    mod->HSMHV_pnftrp_Given = TRUE;
    break;
  case  HSMHV_MOD_PNFALP:
    mod->HSMHV_pnfalp = value->rValue;
    mod->HSMHV_pnfalp_Given = TRUE;
    break;
  case  HSMHV_MOD_PPTHROU:
    mod->HSMHV_ppthrou = value->rValue;
    mod->HSMHV_ppthrou_Given = TRUE;
    break;
  case  HSMHV_MOD_PVDIFFJ:
    mod->HSMHV_pvdiffj = value->rValue;
    mod->HSMHV_pvdiffj_Given = TRUE;
    break;
  case  HSMHV_MOD_PIBPC1:
    mod->HSMHV_pibpc1 = value->rValue;
    mod->HSMHV_pibpc1_Given = TRUE;
    break;
  case  HSMHV_MOD_PIBPC2:
    mod->HSMHV_pibpc2 = value->rValue;
    mod->HSMHV_pibpc2_Given = TRUE;
    break;
    break;
  case  HSMHV_MOD_PCGBO:
    mod->HSMHV_pcgbo = value->rValue;
    mod->HSMHV_pcgbo_Given = TRUE;
    break;
  case  HSMHV_MOD_PCVDSOVER:
    mod->HSMHV_pcvdsover = value->rValue;
    mod->HSMHV_pcvdsover_Given = TRUE;
    break;
  case  HSMHV_MOD_PFALPH:
    mod->HSMHV_pfalph = value->rValue;
    mod->HSMHV_pfalph_Given = TRUE;
    break;
  case  HSMHV_MOD_PNPEXT:
    mod->HSMHV_pnpext = value->rValue;
    mod->HSMHV_pnpext_Given = TRUE;
    break;
  case  HSMHV_MOD_PPOWRAT:
    mod->HSMHV_ppowrat = value->rValue;
    mod->HSMHV_ppowrat_Given = TRUE;
    break;
  case  HSMHV_MOD_PRD:
    mod->HSMHV_prd = value->rValue;
    mod->HSMHV_prd_Given = TRUE;
    break;
  case  HSMHV_MOD_PRD22:
    mod->HSMHV_prd22 = value->rValue;
    mod->HSMHV_prd22_Given = TRUE;
    break;
  case  HSMHV_MOD_PRD23:
    mod->HSMHV_prd23 = value->rValue;
    mod->HSMHV_prd23_Given = TRUE;
    break;
  case  HSMHV_MOD_PRD24:
    mod->HSMHV_prd24 = value->rValue;
    mod->HSMHV_prd24_Given = TRUE;
    break;
  case  HSMHV_MOD_PRDICT1:
    mod->HSMHV_prdict1 = value->rValue;
    mod->HSMHV_prdict1_Given = TRUE;
    break;
  case  HSMHV_MOD_PRDOV13:
    mod->HSMHV_prdov13 = value->rValue;
    mod->HSMHV_prdov13_Given = TRUE;
    break;
  case  HSMHV_MOD_PRDSLP1:
    mod->HSMHV_prdslp1 = value->rValue;
    mod->HSMHV_prdslp1_Given = TRUE;
    break;
  case  HSMHV_MOD_PRDVB:
    mod->HSMHV_prdvb = value->rValue;
    mod->HSMHV_prdvb_Given = TRUE;
    break;
  case  HSMHV_MOD_PRDVD:
    mod->HSMHV_prdvd = value->rValue;
    mod->HSMHV_prdvd_Given = TRUE;
    break;
  case  HSMHV_MOD_PRDVG11:
    mod->HSMHV_prdvg11 = value->rValue;
    mod->HSMHV_prdvg11_Given = TRUE;
    break;
  case  HSMHV_MOD_PRS:
    mod->HSMHV_prs = value->rValue;
    mod->HSMHV_prs_Given = TRUE;
    break;
  case  HSMHV_MOD_PRTH0:
    mod->HSMHV_prth0 = value->rValue;
    mod->HSMHV_prth0_Given = TRUE;
    break;
  case  HSMHV_MOD_PVOVER:
    mod->HSMHV_pvover = value->rValue;
    mod->HSMHV_pvover_Given = TRUE;
    break;

  case HSMHV_MOD_VGS_MAX:
    mod->HSMHVvgsMax = value->rValue;
    mod->HSMHVvgsMaxGiven = TRUE;
    break;
  case HSMHV_MOD_VGD_MAX:
    mod->HSMHVvgdMax = value->rValue;
    mod->HSMHVvgdMaxGiven = TRUE;
    break;
  case HSMHV_MOD_VGB_MAX:
    mod->HSMHVvgbMax = value->rValue;
    mod->HSMHVvgbMaxGiven = TRUE;
    break;
  case HSMHV_MOD_VDS_MAX:
    mod->HSMHVvdsMax = value->rValue;
    mod->HSMHVvdsMaxGiven = TRUE;
    break;
  case HSMHV_MOD_VBS_MAX:
    mod->HSMHVvbsMax = value->rValue;
    mod->HSMHVvbsMaxGiven = TRUE;
    break;
  case HSMHV_MOD_VBD_MAX:
    mod->HSMHVvbdMax = value->rValue;
    mod->HSMHVvbdMaxGiven = TRUE;
    break;
  case HSMHV_MOD_VGSR_MAX:
      mod->HSMHVvgsrMax = value->rValue;
      mod->HSMHVvgsrMaxGiven = TRUE;
      break;
  case HSMHV_MOD_VGDR_MAX:
      mod->HSMHVvgdrMax = value->rValue;
      mod->HSMHVvgdrMaxGiven = TRUE;
      break;
  case HSMHV_MOD_VGBR_MAX:
      mod->HSMHVvgbrMax = value->rValue;
      mod->HSMHVvgbrMaxGiven = TRUE;
      break;
  case HSMHV_MOD_VBSR_MAX:
      mod->HSMHVvbsrMax = value->rValue;
      mod->HSMHVvbsrMaxGiven = TRUE;
      break;
  case HSMHV_MOD_VBDR_MAX:
      mod->HSMHVvbdrMax = value->rValue;
      mod->HSMHVvbdrMaxGiven = TRUE;
      break;

  default:
    return(E_BADPARM);
  }
  return(OK);
}
