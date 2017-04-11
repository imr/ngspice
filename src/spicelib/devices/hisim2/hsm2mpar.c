/***********************************************************************

 HiSIM (Hiroshima University STARC IGFET Model)
 Copyright (C) 2014 Hiroshima University & STARC

 MODEL NAME : HiSIM
 ( VERSION : 2  SUBVERSION : 8  REVISION : 0 )
 
 FILE : hsm2mpar.c

 Date : 2014.6.5

 released by 
                Hiroshima University &
                Semiconductor Technology Academic Research Center (STARC)
***********************************************************************/

/**********************************************************************

The following source code, and all copyrights, trade secrets or other
intellectual property rights in and to the source code in its entirety,
is owned by the Hiroshima University and the STARC organization.

All users need to follow the "HiSIM2 Distribution Statement and
Copyright Notice" attached to HiSIM2 model.

-----HiSIM2 Distribution Statement and Copyright Notice--------------

Software is distributed as is, completely without warranty or service
support. Hiroshima University or STARC and its employees are not liable
for the condition or performance of the software.

Hiroshima University and STARC own the copyright and grant users a perpetual,
irrevocable, worldwide, non-exclusive, royalty-free license with respect 
to the software as set forth below.   

Hiroshima University and STARC hereby disclaim all implied warranties.

Hiroshima University and STARC grant the users the right to modify, copy,
and redistribute the software and documentation, both within the user's
organization and externally, subject to the following restrictions

1. The users agree not to charge for Hiroshima University and STARC code
itself but may charge for additions, extensions, or support.

2. In any product based on the software, the users agree to acknowledge
Hiroshima University and STARC that developed the software. This
acknowledgment shall appear in the product documentation.

3. The users agree to reproduce any copyright notice which appears on
the software on any copy or modification of such made available
to others."


*************************************************************************/

#include "ngspice/ngspice.h"
#include "hsm2def.h"
#include "ngspice/ifsim.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

int HSM2mParam(
     int param,
     IFvalue *value,
     GENmodel *inMod)
{
  HSM2model *mod = (HSM2model*)inMod;
  switch (param) {
  case  HSM2_MOD_NMOS  :
    if (value->iValue) {
      mod->HSM2_type = 1;
      mod->HSM2_type_Given = TRUE;
    }
    break;
  case  HSM2_MOD_PMOS  :
    if (value->iValue) {
      mod->HSM2_type = - 1;
      mod->HSM2_type_Given = TRUE;
    }
    break;
  case  HSM2_MOD_LEVEL:
    mod->HSM2_level = value->iValue;
    mod->HSM2_level_Given = TRUE;
    break;
  case  HSM2_MOD_INFO:
    mod->HSM2_info = value->iValue;
    mod->HSM2_info_Given = TRUE;
    break;
  case HSM2_MOD_NOISE:
    mod->HSM2_noise = value->iValue;
    mod->HSM2_noise_Given = TRUE;
    break;
  case HSM2_MOD_VERSION:
    mod->HSM2_version = value->iValue;
    mod->HSM2_version_Given = TRUE;
    break;
  case HSM2_MOD_SHOW:
    mod->HSM2_show = value->iValue;
    mod->HSM2_show_Given = TRUE;
    break;
  case  HSM2_MOD_CORSRD:
    mod->HSM2_corsrd = value->iValue;
    mod->HSM2_corsrd_Given = TRUE;
    break;
  case  HSM2_MOD_CORG:
    mod->HSM2_corg = value->iValue;
    mod->HSM2_corg_Given = TRUE;
    break;
  case  HSM2_MOD_COIPRV:
    mod->HSM2_coiprv = value->iValue;
    mod->HSM2_coiprv_Given = TRUE;
    break;
  case  HSM2_MOD_COPPRV:
    mod->HSM2_copprv = value->iValue;
    mod->HSM2_copprv_Given = TRUE;
    break;
  case  HSM2_MOD_COADOV:
    mod->HSM2_coadov = value->iValue;
    mod->HSM2_coadov_Given = TRUE;
    break;
  case  HSM2_MOD_COISUB:
    mod->HSM2_coisub = value->iValue;
    mod->HSM2_coisub_Given = TRUE;
    break;
  case  HSM2_MOD_COIIGS:
    mod->HSM2_coiigs = value->iValue;
    mod->HSM2_coiigs_Given = TRUE;
    break;
  case  HSM2_MOD_COGIDL:
    mod->HSM2_cogidl = value->iValue;
    mod->HSM2_cogidl_Given = TRUE;
    break;
  case  HSM2_MOD_COOVLP:
    mod->HSM2_coovlp = value->iValue;
    mod->HSM2_coovlp_Given = TRUE;
    break;
  case  HSM2_MOD_COFLICK:
    mod->HSM2_coflick = value->iValue;
    mod->HSM2_coflick_Given = TRUE;
    break;
  case  HSM2_MOD_COISTI:
    mod->HSM2_coisti = value->iValue;
    mod->HSM2_coisti_Given = TRUE;
    break;
  case  HSM2_MOD_CONQS: /* HiSIM2 */
    mod->HSM2_conqs = value->iValue;
    mod->HSM2_conqs_Given = TRUE;
    break;
  case  HSM2_MOD_CORBNET: 
    mod->HSM2_corbnet = value->iValue;
    mod->HSM2_corbnet_Given = TRUE;
    break;
  case  HSM2_MOD_COTHRML:
    mod->HSM2_cothrml = value->iValue;
    mod->HSM2_cothrml_Given = TRUE;
    break;
  case  HSM2_MOD_COIGN:
    mod->HSM2_coign = value->iValue;
    mod->HSM2_coign_Given = TRUE;
    break;
  case  HSM2_MOD_CODFM:
    mod->HSM2_codfm = value->iValue;
    mod->HSM2_codfm_Given = TRUE;
    break;
  case  HSM2_MOD_CORECIP:
    mod->HSM2_corecip = value->iValue;
    mod->HSM2_corecip_Given = TRUE;
    break;
  case  HSM2_MOD_COQY:
    mod->HSM2_coqy = value->iValue;
    mod->HSM2_coqy_Given = TRUE;
    break;
  case  HSM2_MOD_COQOVSM:
    mod->HSM2_coqovsm = value->iValue;
    mod->HSM2_coqovsm_Given = TRUE;
    break;
  case HSM2_MOD_COERRREP:
    mod->HSM2_coerrrep = value->iValue;
    mod->HSM2_coerrrep_Given = TRUE;
    break;
  case  HSM2_MOD_CODEP:
    mod->HSM2_codep = value->iValue;
    mod->HSM2_codep_Given = TRUE;
    break;
  case HSM2_MOD_CODDLT:
    mod->HSM2_coddlt = value->iValue;
    mod->HSM2_coddlt_Given = TRUE;
    break;

  case  HSM2_MOD_VMAX:
    mod->HSM2_vmax = value->rValue;
    mod->HSM2_vmax_Given = TRUE;
    break;
  case  HSM2_MOD_BGTMP1:
    mod->HSM2_bgtmp1 = value->rValue;
    mod->HSM2_bgtmp1_Given = TRUE;
    break;
  case  HSM2_MOD_BGTMP2:
    mod->HSM2_bgtmp2 =  value->rValue;
    mod->HSM2_bgtmp2_Given = TRUE;
    break;
  case  HSM2_MOD_EG0:
    mod->HSM2_eg0 =  value->rValue;
    mod->HSM2_eg0_Given = TRUE;
    break;
  case  HSM2_MOD_TOX:
    mod->HSM2_tox =  value->rValue;
    mod->HSM2_tox_Given = TRUE;
    break;
  case  HSM2_MOD_XLD:
    mod->HSM2_xld = value->rValue;
    mod->HSM2_xld_Given = TRUE;
    break;
  case  HSM2_MOD_LOVER:
    mod->HSM2_lover = value->rValue;
    mod->HSM2_lover_Given = TRUE;
    break;
  case  HSM2_MOD_DDLTMAX: /* Vdseff */
    mod->HSM2_ddltmax = value->rValue;
    mod->HSM2_ddltmax_Given = TRUE;
    break;
  case  HSM2_MOD_DDLTSLP: /* Vdseff */
    mod->HSM2_ddltslp = value->rValue;
    mod->HSM2_ddltslp_Given = TRUE;
    break;
  case  HSM2_MOD_DDLTICT: /* Vdseff */
    mod->HSM2_ddltict = value->rValue;
    mod->HSM2_ddltict_Given = TRUE;
    break;
  case  HSM2_MOD_VFBOVER:
    mod->HSM2_vfbover = value->rValue;
    mod->HSM2_vfbover_Given = TRUE;
    break;
  case  HSM2_MOD_NOVER:
    mod->HSM2_nover = value->rValue;
    mod->HSM2_nover_Given = TRUE;
    break;
  case  HSM2_MOD_XWD:
    mod->HSM2_xwd = value->rValue;
    mod->HSM2_xwd_Given = TRUE;
    break;
  case  HSM2_MOD_XL:
    mod->HSM2_xl = value->rValue;
    mod->HSM2_xl_Given = TRUE;
    break;
  case  HSM2_MOD_XW:
    mod->HSM2_xw = value->rValue;
    mod->HSM2_xw_Given = TRUE;
    break;
  case  HSM2_MOD_SAREF:
    mod->HSM2_saref = value->rValue;
    mod->HSM2_saref_Given = TRUE;
    break;
  case  HSM2_MOD_SBREF:
    mod->HSM2_sbref = value->rValue;
    mod->HSM2_sbref_Given = TRUE;
    break;
  case  HSM2_MOD_LL:
    mod->HSM2_ll = value->rValue;
    mod->HSM2_ll_Given = TRUE;
    break;
  case  HSM2_MOD_LLD:
    mod->HSM2_lld = value->rValue;
    mod->HSM2_lld_Given = TRUE;
    break;
  case  HSM2_MOD_LLN:
    mod->HSM2_lln = value->rValue;
    mod->HSM2_lln_Given = TRUE;
    break;
  case  HSM2_MOD_WL:
    mod->HSM2_wl = value->rValue;
    mod->HSM2_wl_Given = TRUE;
    break;
  case  HSM2_MOD_WL1:
    mod->HSM2_wl1 = value->rValue;
    mod->HSM2_wl1_Given = TRUE;
    break;
  case  HSM2_MOD_WL1P:
    mod->HSM2_wl1p = value->rValue;
    mod->HSM2_wl1p_Given = TRUE;
    break;
  case  HSM2_MOD_WL2:
    mod->HSM2_wl2 = value->rValue;
    mod->HSM2_wl2_Given = TRUE;
    break;
  case  HSM2_MOD_WL2P:
    mod->HSM2_wl2p = value->rValue;
    mod->HSM2_wl2p_Given = TRUE;
    break;
  case  HSM2_MOD_WLD:
    mod->HSM2_wld = value->rValue;
    mod->HSM2_wld_Given = TRUE;
    break;
  case  HSM2_MOD_WLN:
    mod->HSM2_wln = value->rValue;
    mod->HSM2_wln_Given = TRUE;
    break;
  case  HSM2_MOD_XQY:
    mod->HSM2_xqy = value->rValue;
    mod->HSM2_xqy_Given = TRUE;
    break;
  case  HSM2_MOD_XQY1:
    mod->HSM2_xqy1 = value->rValue;
    mod->HSM2_xqy1_Given = TRUE;
    break;
  case  HSM2_MOD_XQY2:
    mod->HSM2_xqy2 = value->rValue;
    mod->HSM2_xqy2_Given = TRUE;
    break;
  case  HSM2_MOD_QYRAT:
    mod->HSM2_qyrat = value->rValue;
    mod->HSM2_qyrat_Given = TRUE;
    break;
  case  HSM2_MOD_RS:
    mod->HSM2_rs = value->rValue;
    mod->HSM2_rs_Given = TRUE;
    break;
  case  HSM2_MOD_RD:
    mod->HSM2_rd = value->rValue;
    mod->HSM2_rd_Given = TRUE;
    break;
  case  HSM2_MOD_RSH:
    mod->HSM2_rsh = value->rValue;
    mod->HSM2_rsh_Given = TRUE;
    break;
  case  HSM2_MOD_RSHG:
    mod->HSM2_rshg = value->rValue;
    mod->HSM2_rshg_Given = TRUE;
    break;
/*   case  HSM2_MOD_NGCON: */
/*     mod->HSM2_ngcon = value->rValue; */
/*     mod->HSM2_ngcon_Given = TRUE; */
/*     break; */
/*   case  HSM2_MOD_XGW: */
/*     mod->HSM2_xgw = value->rValue; */
/*     mod->HSM2_xgw_Given = TRUE; */
/*     break; */
/*   case  HSM2_MOD_XGL: */
/*     mod->HSM2_xgl = value->rValue; */
/*     mod->HSM2_xgl_Given = TRUE; */
/*     break; */
/*   case  HSM2_MOD_NF: */
/*     mod->HSM2_nf = value->rValue; */
/*     mod->HSM2_nf_Given = TRUE; */
/*     break; */
  case  HSM2_MOD_VFBC:
    mod->HSM2_vfbc = value->rValue;
    mod->HSM2_vfbc_Given = TRUE;
    break;
  case  HSM2_MOD_VBI:
    mod->HSM2_vbi = value->rValue;
    mod->HSM2_vbi_Given = TRUE;
    break;
  case  HSM2_MOD_NSUBC:
    mod->HSM2_nsubc = value->rValue;
    mod->HSM2_nsubc_Given = TRUE;
    break;
  case HSM2_MOD_VFBCL:
    mod->HSM2_vfbcl = value->rValue;
    mod->HSM2_vfbcl_Given = TRUE;
    break;
  case HSM2_MOD_VFBCLP:
    mod->HSM2_vfbclp = value->rValue;
    mod->HSM2_vfbclp_Given = TRUE;
    break;
  case  HSM2_MOD_PARL2:
    mod->HSM2_parl2 = value->rValue;
    mod->HSM2_parl2_Given = TRUE;
    break;
  case  HSM2_MOD_LP:
    mod->HSM2_lp = value->rValue;
    mod->HSM2_lp_Given = TRUE;
    break;
  case  HSM2_MOD_NSUBP:
    mod->HSM2_nsubp = value->rValue;
    mod->HSM2_nsubp_Given = TRUE;
    break;
  case  HSM2_MOD_NSUBPL:
    mod->HSM2_nsubpl = value->rValue;
    mod->HSM2_nsubpl_Given = TRUE;
    break;
  case  HSM2_MOD_NSUBPFAC:
    mod->HSM2_nsubpfac = value->rValue;
    mod->HSM2_nsubpfac_Given = TRUE;
    break;
  case HSM2_MOD_NSUBPDLT:
    mod->HSM2_nsubpdlt = value->rValue;
    mod->HSM2_nsubpdlt_Given = TRUE;
    break;
  case  HSM2_MOD_NSUBPW:
    mod->HSM2_nsubpw = value->rValue;
    mod->HSM2_nsubpw_Given = TRUE;
    break;
  case  HSM2_MOD_NSUBPWP:
    mod->HSM2_nsubpwp = value->rValue;
    mod->HSM2_nsubpwp_Given = TRUE;
    break;
  case  HSM2_MOD_SCP1:
    mod->HSM2_scp1 = value->rValue;
    mod->HSM2_scp1_Given = TRUE;
    break;
  case  HSM2_MOD_SCP2:
    mod->HSM2_scp2 = value->rValue;
    mod->HSM2_scp2_Given = TRUE;
    break;
  case  HSM2_MOD_SCP3:
    mod->HSM2_scp3 = value->rValue;
    mod->HSM2_scp3_Given = TRUE;
    break;
  case  HSM2_MOD_SC1:
    mod->HSM2_sc1 = value->rValue;
    mod->HSM2_sc1_Given = TRUE;
    break;
  case  HSM2_MOD_SC2:
    mod->HSM2_sc2 = value->rValue;
    mod->HSM2_sc2_Given = TRUE;
    break;
  case  HSM2_MOD_SC3:
    mod->HSM2_sc3 = value->rValue;
    mod->HSM2_sc3_Given = TRUE;
    break;
  case  HSM2_MOD_SC4:
    mod->HSM2_sc4 = value->rValue;
    mod->HSM2_sc4_Given = TRUE;
    break;
  case  HSM2_MOD_PGD1:
    mod->HSM2_pgd1 = value->rValue;
    mod->HSM2_pgd1_Given = TRUE;
    break;
  case  HSM2_MOD_PGD2:
    mod->HSM2_pgd2 = value->rValue;
    mod->HSM2_pgd2_Given = TRUE;
    break;
//case  HSM2_MOD_PGD3:
//  mod->HSM2_pgd3 = value->rValue;
//  mod->HSM2_pgd3_Given = TRUE;
//  break;
  case  HSM2_MOD_PGD4:
    mod->HSM2_pgd4 = value->rValue;
    mod->HSM2_pgd4_Given = TRUE;
    break;
  case  HSM2_MOD_NDEP:
    mod->HSM2_ndep = value->rValue;
    mod->HSM2_ndep_Given = TRUE;
    break;
  case  HSM2_MOD_NDEPL:
    mod->HSM2_ndepl = value->rValue;
    mod->HSM2_ndepl_Given = TRUE;
    break;
  case  HSM2_MOD_NDEPLP:
    mod->HSM2_ndeplp = value->rValue;
    mod->HSM2_ndeplp_Given = TRUE;
    break;
  case  HSM2_MOD_NDEPW:
    mod->HSM2_ndepw = value->rValue;
    mod->HSM2_ndepw_Given = TRUE;
    break;
  case  HSM2_MOD_NDEPWP:
    mod->HSM2_ndepwp = value->rValue;
    mod->HSM2_ndepwp_Given = TRUE;
    break;
  case  HSM2_MOD_NINV:
    mod->HSM2_ninv = value->rValue;
    mod->HSM2_ninv_Given = TRUE;
    break;
  case  HSM2_MOD_NINVD:
    mod->HSM2_ninvd = value->rValue;
    mod->HSM2_ninvd_Given = TRUE;
    break;
  case  HSM2_MOD_NINVDL:
    mod->HSM2_ninvdl = value->rValue;
    mod->HSM2_ninvdl_Given = TRUE;
    break;
  case  HSM2_MOD_NINVDLP:
    mod->HSM2_ninvdlp = value->rValue;
    mod->HSM2_ninvdlp_Given = TRUE;
    break;
  case  HSM2_MOD_MUECB0:
    mod->HSM2_muecb0 = value->rValue;
    mod->HSM2_muecb0_Given = TRUE;
    break;
  case  HSM2_MOD_MUECB1:
    mod->HSM2_muecb1 = value->rValue;
    mod->HSM2_muecb1_Given = TRUE;
    break;
  case  HSM2_MOD_MUEPH1:
    mod->HSM2_mueph1 = value->rValue;
    mod->HSM2_mueph1_Given = TRUE;
    break;
  case  HSM2_MOD_MUEPH0:
    mod->HSM2_mueph0 = value->rValue;
    mod->HSM2_mueph0_Given = TRUE;
    break;
  case  HSM2_MOD_MUEPHW:
    mod->HSM2_muephw = value->rValue;
    mod->HSM2_muephw_Given = TRUE;
    break;
  case  HSM2_MOD_MUEPWP:
    mod->HSM2_muepwp = value->rValue;
    mod->HSM2_muepwp_Given = TRUE;
    break;
  case  HSM2_MOD_MUEPWD:
    mod->HSM2_muepwd = value->rValue;
    mod->HSM2_muepwd_Given = TRUE;
    break;
  case  HSM2_MOD_MUEPHL:
    mod->HSM2_muephl = value->rValue;
    mod->HSM2_muephl_Given = TRUE;
    break;
  case  HSM2_MOD_MUEPLP:
    mod->HSM2_mueplp = value->rValue;
    mod->HSM2_mueplp_Given = TRUE;
    break;
  case  HSM2_MOD_MUEPLD:
    mod->HSM2_muepld = value->rValue;
    mod->HSM2_muepld_Given = TRUE;
    break;
  case  HSM2_MOD_MUEPHS:
    mod->HSM2_muephs = value->rValue;
    mod->HSM2_muephs_Given = TRUE;
    break;
   case  HSM2_MOD_MUEPSP:
    mod->HSM2_muepsp = value->rValue;
    mod->HSM2_muepsp_Given = TRUE;
    break;
  case  HSM2_MOD_VTMP:
    mod->HSM2_vtmp = value->rValue;
    mod->HSM2_vtmp_Given = TRUE;
    break;
  case  HSM2_MOD_WVTH0:
    mod->HSM2_wvth0 = value->rValue;
    mod->HSM2_wvth0_Given = TRUE;
    break;
  case  HSM2_MOD_MUESR1:
    mod->HSM2_muesr1 = value->rValue;
    mod->HSM2_muesr1_Given = TRUE;
    break;
  case  HSM2_MOD_MUESR0:
    mod->HSM2_muesr0 = value->rValue;
    mod->HSM2_muesr0_Given = TRUE;
    break;
  case  HSM2_MOD_MUESRL:
    mod->HSM2_muesrl = value->rValue;
    mod->HSM2_muesrl_Given = TRUE;
    break;
  case  HSM2_MOD_MUESLP:
    mod->HSM2_mueslp = value->rValue;
    mod->HSM2_mueslp_Given = TRUE;
    break;
  case  HSM2_MOD_MUESRW:
    mod->HSM2_muesrw = value->rValue;
    mod->HSM2_muesrw_Given = TRUE;
    break;
  case  HSM2_MOD_MUESWP:
    mod->HSM2_mueswp = value->rValue;
    mod->HSM2_mueswp_Given = TRUE;
    break;
  case  HSM2_MOD_BB:
    mod->HSM2_bb = value->rValue;
    mod->HSM2_bb_Given = TRUE;
    break;
  case  HSM2_MOD_SUB1:
    mod->HSM2_sub1 = value->rValue;
    mod->HSM2_sub1_Given = TRUE;
    break;
  case  HSM2_MOD_SUB2:
    mod->HSM2_sub2 = value->rValue;
    mod->HSM2_sub2_Given = TRUE;
    break;
  case  HSM2_MOD_SVGS:
    mod->HSM2_svgs = value->rValue;
    mod->HSM2_svgs_Given = TRUE;
    break;
  case  HSM2_MOD_SVBS:
    mod->HSM2_svbs = value->rValue;
    mod->HSM2_svbs_Given = TRUE;
    break;
  case  HSM2_MOD_SVBSL:
    mod->HSM2_svbsl = value->rValue;
    mod->HSM2_svbsl_Given = TRUE;
    break;
  case  HSM2_MOD_SVDS:
    mod->HSM2_svds = value->rValue;
    mod->HSM2_svds_Given = TRUE;
    break;
  case  HSM2_MOD_SLG:
    mod->HSM2_slg = value->rValue;
    mod->HSM2_slg_Given = TRUE;
    break;
  case  HSM2_MOD_SUB1L:
    mod->HSM2_sub1l = value->rValue;
    mod->HSM2_sub1l_Given = TRUE;
    break;
  case  HSM2_MOD_SUB2L:
    mod->HSM2_sub2l = value->rValue;
    mod->HSM2_sub2l_Given = TRUE;
    break;
  case  HSM2_MOD_SVGSL:
    mod->HSM2_svgsl = value->rValue;
    mod->HSM2_svgsl_Given = TRUE;
    break;
  case  HSM2_MOD_SVGSLP:
    mod->HSM2_svgslp = value->rValue;
    mod->HSM2_svgslp_Given = TRUE;
    break;
  case  HSM2_MOD_SVGSWP:
    mod->HSM2_svgswp = value->rValue;
    mod->HSM2_svgswp_Given = TRUE;
    break;
  case  HSM2_MOD_SVGSW:
    mod->HSM2_svgsw = value->rValue;
    mod->HSM2_svgsw_Given = TRUE;
    break;
  case  HSM2_MOD_SVBSLP:
    mod->HSM2_svbslp = value->rValue;
    mod->HSM2_svbslp_Given = TRUE;
    break;
  case  HSM2_MOD_SLGL:
    mod->HSM2_slgl = value->rValue;
    mod->HSM2_slgl_Given = TRUE;
    break;
  case  HSM2_MOD_SLGLP:
    mod->HSM2_slglp = value->rValue;
    mod->HSM2_slglp_Given = TRUE;
    break;
  case  HSM2_MOD_SUB1LP:
    mod->HSM2_sub1lp = value->rValue;
    mod->HSM2_sub1lp_Given = TRUE;
    break;
  case  HSM2_MOD_NSTI:
    mod->HSM2_nsti = value->rValue;
    mod->HSM2_nsti_Given = TRUE;
    break;
  case  HSM2_MOD_WSTI:
    mod->HSM2_wsti = value->rValue;
    mod->HSM2_wsti_Given = TRUE;
    break;
  case  HSM2_MOD_WSTIL:
    mod->HSM2_wstil = value->rValue;
    mod->HSM2_wstil_Given = TRUE;
    break;
  case  HSM2_MOD_WSTILP:
    mod->HSM2_wstilp = value->rValue;
    mod->HSM2_wstilp_Given = TRUE;
    break;
  case  HSM2_MOD_WSTIW:
    mod->HSM2_wstiw = value->rValue;
    mod->HSM2_wstiw_Given = TRUE;
    break;
  case  HSM2_MOD_WSTIWP:
    mod->HSM2_wstiwp = value->rValue;
    mod->HSM2_wstiwp_Given = TRUE;
    break;
  case  HSM2_MOD_SCSTI1:
    mod->HSM2_scsti1 = value->rValue;
    mod->HSM2_scsti1_Given = TRUE;
    break;
  case  HSM2_MOD_SCSTI2:
    mod->HSM2_scsti2 = value->rValue;
    mod->HSM2_scsti2_Given = TRUE;
    break;
  case  HSM2_MOD_VTHSTI:
    mod->HSM2_vthsti = value->rValue;
    mod->HSM2_vthsti_Given = TRUE;
    break;
  case  HSM2_MOD_VDSTI:
    mod->HSM2_vdsti = value->rValue;
    mod->HSM2_vdsti_Given = TRUE;
    break;
  case  HSM2_MOD_MUESTI1:
    mod->HSM2_muesti1 = value->rValue;
    mod->HSM2_muesti1_Given = TRUE;
    break;
  case  HSM2_MOD_MUESTI2:
    mod->HSM2_muesti2 = value->rValue;
    mod->HSM2_muesti2_Given = TRUE;
    break;
  case  HSM2_MOD_MUESTI3:
    mod->HSM2_muesti3 = value->rValue;
    mod->HSM2_muesti3_Given = TRUE;
    break;
  case  HSM2_MOD_NSUBPSTI1:
    mod->HSM2_nsubpsti1 = value->rValue;
    mod->HSM2_nsubpsti1_Given = TRUE;
    break;
  case  HSM2_MOD_NSUBPSTI2:
    mod->HSM2_nsubpsti2 = value->rValue;
    mod->HSM2_nsubpsti2_Given = TRUE;
    break;
  case  HSM2_MOD_NSUBPSTI3:
    mod->HSM2_nsubpsti3 = value->rValue;
    mod->HSM2_nsubpsti3_Given = TRUE;
    break;
  case HSM2_MOD_NSUBCSTI1:
    mod->HSM2_nsubcsti1 = value->rValue;
    mod->HSM2_nsubcsti1_Given = TRUE;
    break;
  case HSM2_MOD_NSUBCSTI2:
    mod->HSM2_nsubcsti2 = value->rValue;
    mod->HSM2_nsubcsti2_Given = TRUE;
    break;
  case HSM2_MOD_NSUBCSTI3:
    mod->HSM2_nsubcsti3 = value->rValue;
    mod->HSM2_nsubcsti3_Given = TRUE;
    break;
  case  HSM2_MOD_LPEXT:
    mod->HSM2_lpext = value->rValue;
    mod->HSM2_lpext_Given = TRUE;
    break;
  case  HSM2_MOD_NPEXT:
    mod->HSM2_npext = value->rValue;
    mod->HSM2_npext_Given = TRUE;
    break;
  case  HSM2_MOD_NPEXTW:
    mod->HSM2_npextw = value->rValue;
    mod->HSM2_npextw_Given = TRUE;
    break;
  case  HSM2_MOD_NPEXTWP:
    mod->HSM2_npextwp = value->rValue;
    mod->HSM2_npextwp_Given = TRUE;
    break;
  case  HSM2_MOD_SCP22:
    mod->HSM2_scp22 = value->rValue;
    mod->HSM2_scp22_Given = TRUE;
    break;
  case  HSM2_MOD_SCP21:
    mod->HSM2_scp21 = value->rValue;
    mod->HSM2_scp21_Given = TRUE;
    break;
  case  HSM2_MOD_BS1:
    mod->HSM2_bs1 = value->rValue;
    mod->HSM2_bs1_Given = TRUE;
    break;
  case  HSM2_MOD_BS2:
    mod->HSM2_bs2 = value->rValue;
    mod->HSM2_bs2_Given = TRUE;
    break;
  case  HSM2_MOD_CGSO:
    mod->HSM2_cgso = value->rValue;
    mod->HSM2_cgso_Given = TRUE;
    break;
  case  HSM2_MOD_CGDO:
    mod->HSM2_cgdo = value->rValue;
    mod->HSM2_cgdo_Given = TRUE;
    break;
  case  HSM2_MOD_CGBO:
    mod->HSM2_cgbo = value->rValue;
    mod->HSM2_cgbo_Given = TRUE;
    break;
  case  HSM2_MOD_TPOLY:
    mod->HSM2_tpoly = value->rValue;
    mod->HSM2_tpoly_Given = TRUE;
    break;
  case  HSM2_MOD_JS0:
    mod->HSM2_js0 = value->rValue;
    mod->HSM2_js0_Given = TRUE;
    break;
  case  HSM2_MOD_JS0SW:
    mod->HSM2_js0sw = value->rValue;
    mod->HSM2_js0sw_Given = TRUE;
    break;
  case  HSM2_MOD_NJ:
    mod->HSM2_nj = value->rValue;
    mod->HSM2_nj_Given = TRUE;
    break;
  case  HSM2_MOD_NJSW:
    mod->HSM2_njsw = value->rValue;
    mod->HSM2_njsw_Given = TRUE;
    break;
  case  HSM2_MOD_XTI:
    mod->HSM2_xti = value->rValue;
    mod->HSM2_xti_Given = TRUE;
    break;
  case  HSM2_MOD_CJ:
    mod->HSM2_cj = value->rValue;
    mod->HSM2_cj_Given = TRUE;
    break;
  case  HSM2_MOD_CJSW:
    mod->HSM2_cjsw = value->rValue;
    mod->HSM2_cjsw_Given = TRUE;
    break;
  case  HSM2_MOD_CJSWG:
    mod->HSM2_cjswg = value->rValue;
    mod->HSM2_cjswg_Given = TRUE;
    break;
  case  HSM2_MOD_MJ:
    mod->HSM2_mj = value->rValue;
    mod->HSM2_mj_Given = TRUE;
    break;
  case  HSM2_MOD_MJSW:
    mod->HSM2_mjsw = value->rValue;
    mod->HSM2_mjsw_Given = TRUE;
    break;
  case  HSM2_MOD_MJSWG:
    mod->HSM2_mjswg = value->rValue;
    mod->HSM2_mjswg_Given = TRUE;
    break;
  case  HSM2_MOD_PB:
    mod->HSM2_pb = value->rValue;
    mod->HSM2_pb_Given = TRUE;
    break;
  case  HSM2_MOD_PBSW:
    mod->HSM2_pbsw = value->rValue;
    mod->HSM2_pbsw_Given = TRUE;
    break;
  case  HSM2_MOD_PBSWG:
    mod->HSM2_pbswg = value->rValue;
    mod->HSM2_pbswg_Given = TRUE;
    break;

  case  HSM2_MOD_TCJBD:
    mod->HSM2_tcjbd = value->rValue;
    mod->HSM2_tcjbd_Given = TRUE;
    break;
  case  HSM2_MOD_TCJBS:
    mod->HSM2_tcjbs = value->rValue;
    mod->HSM2_tcjbs_Given = TRUE;
    break;
  case  HSM2_MOD_TCJBDSW:
    mod->HSM2_tcjbdsw = value->rValue;
    mod->HSM2_tcjbdsw_Given = TRUE;
    break;
  case  HSM2_MOD_TCJBSSW:
    mod->HSM2_tcjbssw = value->rValue;
    mod->HSM2_tcjbssw_Given = TRUE;
    break;
  case  HSM2_MOD_TCJBDSWG:
    mod->HSM2_tcjbdswg = value->rValue;
    mod->HSM2_tcjbdswg_Given = TRUE;
    break;
  case  HSM2_MOD_TCJBSSWG:
    mod->HSM2_tcjbsswg = value->rValue;
    mod->HSM2_tcjbsswg_Given = TRUE;
    break;

  case  HSM2_MOD_XTI2:
    mod->HSM2_xti2 = value->rValue;
    mod->HSM2_xti2_Given = TRUE;
    break;
  case  HSM2_MOD_CISB:
    mod->HSM2_cisb = value->rValue;
    mod->HSM2_cisb_Given = TRUE;
    break;
  case  HSM2_MOD_CVB:
    mod->HSM2_cvb = value->rValue;
    mod->HSM2_cvb_Given = TRUE;
    break;
  case  HSM2_MOD_CTEMP:
    mod->HSM2_ctemp = value->rValue;
    mod->HSM2_ctemp_Given = TRUE;
    break;
  case  HSM2_MOD_CISBK:
    mod->HSM2_cisbk = value->rValue;
    mod->HSM2_cisbk_Given = TRUE;
    break;
  case  HSM2_MOD_CVBK:
    mod->HSM2_cvbk = value->rValue;
    mod->HSM2_cvbk_Given = TRUE;
    break;
  case  HSM2_MOD_DIVX:
    mod->HSM2_divx = value->rValue;
    mod->HSM2_divx_Given = TRUE;
    break;
  case  HSM2_MOD_CLM1:
    mod->HSM2_clm1 = value->rValue;
    mod->HSM2_clm1_Given = TRUE;
    break;
  case  HSM2_MOD_CLM2:
    mod->HSM2_clm2 = value->rValue;
    mod->HSM2_clm2_Given = TRUE;
    break;
  case  HSM2_MOD_CLM3:
    mod->HSM2_clm3 = value->rValue;
    mod->HSM2_clm3_Given = TRUE;
    break;
  case  HSM2_MOD_CLM5:
    mod->HSM2_clm5 = value->rValue;
    mod->HSM2_clm5_Given = TRUE;
    break;
  case  HSM2_MOD_CLM6:
    mod->HSM2_clm6 = value->rValue;
    mod->HSM2_clm6_Given = TRUE;
    break;
  case  HSM2_MOD_MUETMP:
    mod->HSM2_muetmp = value->rValue;
    mod->HSM2_muetmp_Given = TRUE;
    break;
  case  HSM2_MOD_VOVER:
    mod->HSM2_vover = value->rValue;
    mod->HSM2_vover_Given = TRUE;
    break;
  case  HSM2_MOD_VOVERP:
    mod->HSM2_voverp = value->rValue;
    mod->HSM2_voverp_Given = TRUE;
    break;
  case  HSM2_MOD_VOVERS:
    mod->HSM2_vovers = value->rValue;
    mod->HSM2_vovers_Given = TRUE;
    break;
  case  HSM2_MOD_VOVERSP:
    mod->HSM2_voversp = value->rValue;
    mod->HSM2_voversp_Given = TRUE;
    break;
  case  HSM2_MOD_WFC:
    mod->HSM2_wfc = value->rValue;
    mod->HSM2_wfc_Given = TRUE;
    break;
  case  HSM2_MOD_NSUBCW:
    mod->HSM2_nsubcw = value->rValue;
    mod->HSM2_nsubcw_Given = TRUE;
    break;
  case  HSM2_MOD_NSUBCWP:
    mod->HSM2_nsubcwp = value->rValue;
    mod->HSM2_nsubcwp_Given = TRUE;
    break;
  case  HSM2_MOD_NSUBCMAX:
    mod->HSM2_nsubcmax = value->rValue;
    mod->HSM2_nsubcmax_Given = TRUE;
    break;
  case  HSM2_MOD_QME1:
    mod->HSM2_qme1 = value->rValue;
    mod->HSM2_qme1_Given = TRUE;
    break;
  case  HSM2_MOD_QME2:
    mod->HSM2_qme2 = value->rValue;
    mod->HSM2_qme2_Given = TRUE;
    break;
  case  HSM2_MOD_QME3:
    mod->HSM2_qme3 = value->rValue;
    mod->HSM2_qme3_Given = TRUE;
    break;
  case  HSM2_MOD_GIDL1:
    mod->HSM2_gidl1 = value->rValue;
    mod->HSM2_gidl1_Given = TRUE;
    break;
  case  HSM2_MOD_GIDL2:
    mod->HSM2_gidl2 = value->rValue;
    mod->HSM2_gidl2_Given = TRUE;
    break;
  case  HSM2_MOD_GIDL3:
    mod->HSM2_gidl3 = value->rValue;
    mod->HSM2_gidl3_Given = TRUE;
    break;
  case  HSM2_MOD_GIDL4:
    mod->HSM2_gidl4 = value->rValue;
    mod->HSM2_gidl4_Given = TRUE;
    break;
  case  HSM2_MOD_GIDL5:
    mod->HSM2_gidl5 = value->rValue;
    mod->HSM2_gidl5_Given = TRUE;
    break;
  case HSM2_MOD_GIDL6:
    mod->HSM2_gidl6 = value->rValue;
    mod->HSM2_gidl6_Given = TRUE;
    break;
  case HSM2_MOD_GIDL7:
    mod->HSM2_gidl7 = value->rValue;
    mod->HSM2_gidl7_Given = TRUE;
    break;
  case  HSM2_MOD_GLEAK1:
    mod->HSM2_gleak1 = value->rValue;
    mod->HSM2_gleak1_Given = TRUE;
    break;
  case  HSM2_MOD_GLEAK2:
    mod->HSM2_gleak2 = value->rValue;
    mod->HSM2_gleak2_Given = TRUE;
    break;
  case  HSM2_MOD_GLEAK3:
    mod->HSM2_gleak3 = value->rValue;
    mod->HSM2_gleak3_Given = TRUE;
    break;
  case  HSM2_MOD_GLEAK4:
    mod->HSM2_gleak4 = value->rValue;
    mod->HSM2_gleak4_Given = TRUE;
    break;
  case  HSM2_MOD_GLEAK5:
    mod->HSM2_gleak5 = value->rValue;
    mod->HSM2_gleak5_Given = TRUE;
    break;
  case  HSM2_MOD_GLEAK6:
    mod->HSM2_gleak6 = value->rValue;
    mod->HSM2_gleak6_Given = TRUE;
    break;
  case  HSM2_MOD_GLEAK7:
    mod->HSM2_gleak7 = value->rValue;
    mod->HSM2_gleak7_Given = TRUE;
    break;
  case  HSM2_MOD_GLKSD1:
    mod->HSM2_glksd1 = value->rValue;
    mod->HSM2_glksd1_Given = TRUE;
    break;
  case  HSM2_MOD_GLKSD2:
    mod->HSM2_glksd2 = value->rValue;
    mod->HSM2_glksd2_Given = TRUE;
    break;
  case  HSM2_MOD_GLKSD3:
    mod->HSM2_glksd3 = value->rValue;
    mod->HSM2_glksd3_Given = TRUE;
    break;
  case  HSM2_MOD_GLKB1:
    mod->HSM2_glkb1 = value->rValue;
    mod->HSM2_glkb1_Given = TRUE;
    break;
  case  HSM2_MOD_GLKB2:
    mod->HSM2_glkb2 = value->rValue;
    mod->HSM2_glkb2_Given = TRUE;
    break;
  case  HSM2_MOD_GLKB3:
    mod->HSM2_glkb3 = value->rValue;
    mod->HSM2_glkb3_Given = TRUE;
    break;
  case  HSM2_MOD_EGIG:
    mod->HSM2_egig = value->rValue;
    mod->HSM2_egig_Given = TRUE;
    break;
  case  HSM2_MOD_IGTEMP2:
    mod->HSM2_igtemp2 = value->rValue;
    mod->HSM2_igtemp2_Given = TRUE;
    break;
  case  HSM2_MOD_IGTEMP3:
    mod->HSM2_igtemp3 = value->rValue;
    mod->HSM2_igtemp3_Given = TRUE;
    break;
  case  HSM2_MOD_VZADD0:
    mod->HSM2_vzadd0 = value->rValue;
    mod->HSM2_vzadd0_Given = TRUE;
    break;
  case  HSM2_MOD_PZADD0:
    mod->HSM2_pzadd0 = value->rValue;
    mod->HSM2_pzadd0_Given = TRUE;
    break;
  case  HSM2_MOD_NFTRP:
    mod->HSM2_nftrp = value->rValue;
    mod->HSM2_nftrp_Given = TRUE;
    break;
  case  HSM2_MOD_NFALP:
    mod->HSM2_nfalp = value->rValue;
    mod->HSM2_nfalp_Given = TRUE;
    break;
  case  HSM2_MOD_CIT:
    mod->HSM2_cit = value->rValue;
    mod->HSM2_cit_Given = TRUE;
    break;
  case  HSM2_MOD_FALPH:
    mod->HSM2_falph = value->rValue;
    mod->HSM2_falph_Given = TRUE;
    break;
  case  HSM2_MOD_KAPPA:
    mod->HSM2_kappa = value->rValue;
    mod->HSM2_kappa_Given = TRUE;
    break;
  case  HSM2_MOD_VDIFFJ:
    mod->HSM2_vdiffj = value->rValue;
    mod->HSM2_vdiffj_Given = TRUE;
    break;
  case  HSM2_MOD_DLY1:
    mod->HSM2_dly1 = value->rValue;
    mod->HSM2_dly1_Given = TRUE;
    break;
  case  HSM2_MOD_DLY2:
    mod->HSM2_dly2 = value->rValue;
    mod->HSM2_dly2_Given = TRUE;
    break;
  case  HSM2_MOD_DLY3:
    mod->HSM2_dly3 = value->rValue;
    mod->HSM2_dly3_Given = TRUE;
    break;
  case  HSM2_MOD_TNOM:
    mod->HSM2_tnom = value->rValue;
    mod->HSM2_tnom_Given = TRUE;
    break;
  case  HSM2_MOD_OVSLP:
    mod->HSM2_ovslp = value->rValue;
    mod->HSM2_ovslp_Given = TRUE;
    break;
  case  HSM2_MOD_OVMAG:
    mod->HSM2_ovmag = value->rValue;
    mod->HSM2_ovmag_Given = TRUE;
    break;
  case  HSM2_MOD_GBMIN:
    mod->HSM2_gbmin = value->rValue;
    mod->HSM2_gbmin_Given = TRUE;
    break;
  case  HSM2_MOD_RBPB:
    mod->HSM2_rbpb = value->rValue;
    mod->HSM2_rbpb_Given = TRUE;
    break;
  case  HSM2_MOD_RBPD:
    mod->HSM2_rbpd = value->rValue;
    mod->HSM2_rbpd_Given = TRUE;
    break;
  case  HSM2_MOD_RBPS:
    mod->HSM2_rbps = value->rValue;
    mod->HSM2_rbps_Given = TRUE;
    break;
  case  HSM2_MOD_RBDB:
    mod->HSM2_rbdb = value->rValue;
    mod->HSM2_rbdb_Given = TRUE;
    break;
  case  HSM2_MOD_RBSB:
    mod->HSM2_rbsb = value->rValue;
    mod->HSM2_rbsb_Given = TRUE;
    break;
  case  HSM2_MOD_IBPC1:
    mod->HSM2_ibpc1 = value->rValue;
    mod->HSM2_ibpc1_Given = TRUE;
    break;
  case  HSM2_MOD_IBPC2:
    mod->HSM2_ibpc2 = value->rValue;
    mod->HSM2_ibpc2_Given = TRUE;
    break;
  case  HSM2_MOD_MPHDFM:
    mod->HSM2_mphdfm = value->rValue;
    mod->HSM2_mphdfm_Given = TRUE;
    break;

  case  HSM2_MOD_PTL:
    mod->HSM2_ptl = value->rValue;
    mod->HSM2_ptl_Given = TRUE;
    break;
  case  HSM2_MOD_PTP:
    mod->HSM2_ptp = value->rValue;
    mod->HSM2_ptp_Given = TRUE;
    break;
  case  HSM2_MOD_PT2:
    mod->HSM2_pt2 = value->rValue;
    mod->HSM2_pt2_Given = TRUE;
    break;
  case  HSM2_MOD_PTLP:
    mod->HSM2_ptlp = value->rValue;
    mod->HSM2_ptlp_Given = TRUE;
    break;
  case  HSM2_MOD_GDL:
    mod->HSM2_gdl = value->rValue;
    mod->HSM2_gdl_Given = TRUE;
    break;
  case  HSM2_MOD_GDLP:
    mod->HSM2_gdlp = value->rValue;
    mod->HSM2_gdlp_Given = TRUE;
    break;

  case  HSM2_MOD_GDLD:
    mod->HSM2_gdld = value->rValue;
    mod->HSM2_gdld_Given = TRUE;
    break;
  case  HSM2_MOD_PT4:
    mod->HSM2_pt4 = value->rValue;
    mod->HSM2_pt4_Given = TRUE;
    break;
  case  HSM2_MOD_PT4P:
    mod->HSM2_pt4p = value->rValue;
    mod->HSM2_pt4p_Given = TRUE;
    break;
  case  HSM2_MOD_MUEPHL2:
    mod->HSM2_muephl2 = value->rValue;
    mod->HSM2_muephl2_Given = TRUE;
    break;
  case  HSM2_MOD_MUEPLP2:
    mod->HSM2_mueplp2 = value->rValue;
    mod->HSM2_mueplp2_Given = TRUE;
    break;
  case  HSM2_MOD_NSUBCW2:
    mod->HSM2_nsubcw2 = value->rValue;
    mod->HSM2_nsubcw2_Given = TRUE;
    break;
  case  HSM2_MOD_NSUBCWP2:
    mod->HSM2_nsubcwp2 = value->rValue;
    mod->HSM2_nsubcwp2_Given = TRUE;
    break;
  case  HSM2_MOD_MUEPHW2:
    mod->HSM2_muephw2 = value->rValue;
    mod->HSM2_muephw2_Given = TRUE;
    break;
  case  HSM2_MOD_MUEPWP2:
    mod->HSM2_muepwp2 = value->rValue;
    mod->HSM2_muepwp2_Given = TRUE;
    break;
/* WPE */
  case HSM2_MOD_WEB:
    mod->HSM2_web = value->rValue;
    mod->HSM2_web_Given = TRUE;
    break;
  case HSM2_MOD_WEC:
    mod->HSM2_wec = value->rValue;
	mod->HSM2_wec_Given = TRUE;
	break;
  case HSM2_MOD_NSUBCWPE:
    mod->HSM2_nsubcwpe = value->rValue;
	mod->HSM2_nsubcwpe_Given = TRUE;
	break;
  case HSM2_MOD_NPEXTWPE:
    mod->HSM2_npextwpe = value->rValue;
	mod->HSM2_npextwpe_Given = TRUE;
	break;
  case HSM2_MOD_NSUBPWPE:
    mod->HSM2_nsubpwpe = value->rValue;
	mod->HSM2_nsubpwpe_Given = TRUE;
	break;
  case  HSM2_MOD_VGSMIN:
    mod->HSM2_Vgsmin = value->rValue;
    mod->HSM2_Vgsmin_Given = TRUE;
    break;
  case  HSM2_MOD_SC3VBS:
    mod->HSM2_sc3Vbs = value->rValue;
    mod->HSM2_sc3Vbs_Given = TRUE;
    break;
  case  HSM2_MOD_BYPTOL:
    mod->HSM2_byptol = value->rValue;
    mod->HSM2_byptol_Given = TRUE;
    break;
  case  HSM2_MOD_MUECB0LP:
    mod->HSM2_muecb0lp = value->rValue;
    mod->HSM2_muecb0lp_Given = TRUE;
    break;
  case  HSM2_MOD_MUECB1LP:
    mod->HSM2_muecb1lp = value->rValue;
    mod->HSM2_muecb1lp_Given = TRUE;
    break;

  /* Depletion Mode MODFET */
  case HSM2_MOD_NDEPM:
    mod->HSM2_ndepm = value->rValue;
    mod->HSM2_ndepm_Given = TRUE;
    break;
  case HSM2_MOD_NDEPML:
    mod->HSM2_ndepml = value->rValue;
    mod->HSM2_ndepml_Given = TRUE;
    break;
  case HSM2_MOD_NDEPMLP:
    mod->HSM2_ndepmlp = value->rValue;
    mod->HSM2_ndepmlp_Given = TRUE;
    break;
  case HSM2_MOD_TNDEP:
    mod->HSM2_tndep = value->rValue;
    mod->HSM2_tndep_Given = TRUE;
    break;
  case HSM2_MOD_DEPLEAK:
    mod->HSM2_depleak = value->rValue;
    mod->HSM2_depleak_Given = TRUE;
    break;
  case HSM2_MOD_DEPLEAKL:
    mod->HSM2_depleakl = value->rValue;
    mod->HSM2_depleakl_Given = TRUE;
    break;
  case HSM2_MOD_DEPLEAKLP:
    mod->HSM2_depleaklp = value->rValue;
    mod->HSM2_depleaklp_Given = TRUE;
    break;
  case HSM2_MOD_DEPETA:
    mod->HSM2_depeta = value->rValue;
    mod->HSM2_depeta_Given = TRUE;
    break;
  case HSM2_MOD_DEPMUE0:
    mod->HSM2_depmue0 = value->rValue;
    mod->HSM2_depmue0_Given = TRUE;
    break;
  case HSM2_MOD_DEPMUE0L:
    mod->HSM2_depmue0l = value->rValue;
    mod->HSM2_depmue0l_Given = TRUE;
    break;
  case HSM2_MOD_DEPMUE0LP:
    mod->HSM2_depmue0lp = value->rValue;
    mod->HSM2_depmue0lp_Given = TRUE;
    break;
  case HSM2_MOD_DEPMUE1:
    mod->HSM2_depmue1 = value->rValue;
    mod->HSM2_depmue1_Given = TRUE;
    break;
  case HSM2_MOD_DEPMUE1L:
    mod->HSM2_depmue1l = value->rValue;
    mod->HSM2_depmue1l_Given = TRUE;
    break;
  case HSM2_MOD_DEPMUE1LP:
    mod->HSM2_depmue1lp = value->rValue;
    mod->HSM2_depmue1lp_Given = TRUE;
    break;
  case HSM2_MOD_DEPMUEBACK0:
    mod->HSM2_depmueback0 = value->rValue;
    mod->HSM2_depmueback0_Given = TRUE;
    break;
  case HSM2_MOD_DEPMUEBACK0L:
    mod->HSM2_depmueback0l = value->rValue;
    mod->HSM2_depmueback0l_Given = TRUE;
    break;
  case HSM2_MOD_DEPMUEBACK0LP:
    mod->HSM2_depmueback0lp = value->rValue;
    mod->HSM2_depmueback0lp_Given = TRUE;
    break;
  case HSM2_MOD_DEPMUEBACK1:
    mod->HSM2_depmueback1 = value->rValue;
    mod->HSM2_depmueback1_Given = TRUE;
    break;
  case HSM2_MOD_DEPMUEBACK1L:
    mod->HSM2_depmueback1l = value->rValue;
    mod->HSM2_depmueback1l_Given = TRUE;
    break;
  case HSM2_MOD_DEPMUEBACK1LP:
    mod->HSM2_depmueback1lp = value->rValue;
    mod->HSM2_depmueback1lp_Given = TRUE;
    break;
  case HSM2_MOD_DEPMUEPH0:
    mod->HSM2_depmueph0 = value->rValue;
    mod->HSM2_depmueph0_Given = TRUE;
    break;
  case HSM2_MOD_DEPMUEPH1:
    mod->HSM2_depmueph1 = value->rValue;
    mod->HSM2_depmueph1_Given = TRUE;
    break;
  case HSM2_MOD_DEPVMAX:
    mod->HSM2_depvmax = value->rValue;
    mod->HSM2_depvmax_Given = TRUE;
    break;
  case HSM2_MOD_DEPVMAXL:
    mod->HSM2_depvmaxl = value->rValue;
    mod->HSM2_depvmaxl_Given = TRUE;
    break;
  case HSM2_MOD_DEPVMAXLP:
    mod->HSM2_depvmaxlp = value->rValue;
    mod->HSM2_depvmaxlp_Given = TRUE;
    break;
  case HSM2_MOD_DEPVDSEF1:
    mod->HSM2_depvdsef1 = value->rValue;
    mod->HSM2_depvdsef1_Given = TRUE;
    break;
  case HSM2_MOD_DEPVDSEF1L:
    mod->HSM2_depvdsef1l = value->rValue;
    mod->HSM2_depvdsef1l_Given = TRUE;
    break;
  case HSM2_MOD_DEPVDSEF1LP:
    mod->HSM2_depvdsef1lp = value->rValue;
    mod->HSM2_depvdsef1lp_Given = TRUE;
    break;
  case HSM2_MOD_DEPVDSEF2:
    mod->HSM2_depvdsef2 = value->rValue;
    mod->HSM2_depvdsef2_Given = TRUE;
    break;
  case HSM2_MOD_DEPVDSEF2L:
    mod->HSM2_depvdsef2l = value->rValue;
    mod->HSM2_depvdsef2l_Given = TRUE;
    break;
  case HSM2_MOD_DEPVDSEF2LP:
    mod->HSM2_depvdsef2lp = value->rValue;
    mod->HSM2_depvdsef2lp_Given = TRUE;
    break;
  case HSM2_MOD_DEPBB:
    mod->HSM2_depbb = value->rValue;
    mod->HSM2_depbb_Given = TRUE;
    break;
  case HSM2_MOD_DEPMUETMP:
    mod->HSM2_depmuetmp = value->rValue;
    mod->HSM2_depmuetmp_Given = TRUE;
    break;

  /* binning parameters */
  case  HSM2_MOD_LMIN:
    mod->HSM2_lmin = value->rValue;
    mod->HSM2_lmin_Given = TRUE;
    break;
  case  HSM2_MOD_LMAX:
    mod->HSM2_lmax = value->rValue;
    mod->HSM2_lmax_Given = TRUE;
    break;
  case  HSM2_MOD_WMIN:
    mod->HSM2_wmin = value->rValue;
    mod->HSM2_wmin_Given = TRUE;
    break;
  case  HSM2_MOD_WMAX:
    mod->HSM2_wmax = value->rValue;
    mod->HSM2_wmax_Given = TRUE;
    break;
  case  HSM2_MOD_LBINN:
    mod->HSM2_lbinn = value->rValue;
    mod->HSM2_lbinn_Given = TRUE;
    break;
  case  HSM2_MOD_WBINN:
    mod->HSM2_wbinn = value->rValue;
    mod->HSM2_wbinn_Given = TRUE;
    break;

  /* Length dependence */
  case  HSM2_MOD_LVMAX:
    mod->HSM2_lvmax = value->rValue;
    mod->HSM2_lvmax_Given = TRUE;
    break;
  case  HSM2_MOD_LBGTMP1:
    mod->HSM2_lbgtmp1 = value->rValue;
    mod->HSM2_lbgtmp1_Given = TRUE;
    break;
  case  HSM2_MOD_LBGTMP2:
    mod->HSM2_lbgtmp2 = value->rValue;
    mod->HSM2_lbgtmp2_Given = TRUE;
    break;
  case  HSM2_MOD_LEG0:
    mod->HSM2_leg0 = value->rValue;
    mod->HSM2_leg0_Given = TRUE;
    break;
  case  HSM2_MOD_LLOVER:
    mod->HSM2_llover = value->rValue;
    mod->HSM2_llover_Given = TRUE;
    break;
  case  HSM2_MOD_LVFBOVER:
    mod->HSM2_lvfbover = value->rValue;
    mod->HSM2_lvfbover_Given = TRUE;
    break;
  case  HSM2_MOD_LNOVER:
    mod->HSM2_lnover = value->rValue;
    mod->HSM2_lnover_Given = TRUE;
    break;
  case  HSM2_MOD_LWL2:
    mod->HSM2_lwl2 = value->rValue;
    mod->HSM2_lwl2_Given = TRUE;
    break;
  case  HSM2_MOD_LVFBC:
    mod->HSM2_lvfbc = value->rValue;
    mod->HSM2_lvfbc_Given = TRUE;
    break;
  case  HSM2_MOD_LNSUBC:
    mod->HSM2_lnsubc = value->rValue;
    mod->HSM2_lnsubc_Given = TRUE;
    break;
  case  HSM2_MOD_LNSUBP:
    mod->HSM2_lnsubp = value->rValue;
    mod->HSM2_lnsubp_Given = TRUE;
    break;
  case  HSM2_MOD_LSCP1:
    mod->HSM2_lscp1 = value->rValue;
    mod->HSM2_lscp1_Given = TRUE;
    break;
  case  HSM2_MOD_LSCP2:
    mod->HSM2_lscp2 = value->rValue;
    mod->HSM2_lscp2_Given = TRUE;
    break;
  case  HSM2_MOD_LSCP3:
    mod->HSM2_lscp3 = value->rValue;
    mod->HSM2_lscp3_Given = TRUE;
    break;
  case  HSM2_MOD_LSC1:
    mod->HSM2_lsc1 = value->rValue;
    mod->HSM2_lsc1_Given = TRUE;
    break;
  case  HSM2_MOD_LSC2:
    mod->HSM2_lsc2 = value->rValue;
    mod->HSM2_lsc2_Given = TRUE;
    break;
  case  HSM2_MOD_LSC3:
    mod->HSM2_lsc3 = value->rValue;
    mod->HSM2_lsc3_Given = TRUE;
    break;
  case  HSM2_MOD_LSC4:
    mod->HSM2_lsc4 = value->rValue;
    mod->HSM2_lsc4_Given = TRUE;
    break;
  case  HSM2_MOD_LPGD1:
    mod->HSM2_lpgd1 = value->rValue;
    mod->HSM2_lpgd1_Given = TRUE;
    break;
//case  HSM2_MOD_LPGD3:
//  mod->HSM2_lpgd3 = value->rValue;
//  mod->HSM2_lpgd3_Given = TRUE;
//  break;
  case  HSM2_MOD_LNDEP:
    mod->HSM2_lndep = value->rValue;
    mod->HSM2_lndep_Given = TRUE;
    break;
  case  HSM2_MOD_LNINV:
    mod->HSM2_lninv = value->rValue;
    mod->HSM2_lninv_Given = TRUE;
    break;
  case  HSM2_MOD_LMUECB0:
    mod->HSM2_lmuecb0 = value->rValue;
    mod->HSM2_lmuecb0_Given = TRUE;
    break;
  case  HSM2_MOD_LMUECB1:
    mod->HSM2_lmuecb1 = value->rValue;
    mod->HSM2_lmuecb1_Given = TRUE;
    break;
  case  HSM2_MOD_LMUEPH1:
    mod->HSM2_lmueph1 = value->rValue;
    mod->HSM2_lmueph1_Given = TRUE;
    break;
  case  HSM2_MOD_LVTMP:
    mod->HSM2_lvtmp = value->rValue;
    mod->HSM2_lvtmp_Given = TRUE;
    break;
  case  HSM2_MOD_LWVTH0:
    mod->HSM2_lwvth0 = value->rValue;
    mod->HSM2_lwvth0_Given = TRUE;
    break;
  case  HSM2_MOD_LMUESR1:
    mod->HSM2_lmuesr1 = value->rValue;
    mod->HSM2_lmuesr1_Given = TRUE;
    break;
  case  HSM2_MOD_LMUETMP:
    mod->HSM2_lmuetmp = value->rValue;
    mod->HSM2_lmuetmp_Given = TRUE;
    break;
  case  HSM2_MOD_LSUB1:
    mod->HSM2_lsub1 = value->rValue;
    mod->HSM2_lsub1_Given = TRUE;
    break;
  case  HSM2_MOD_LSUB2:
    mod->HSM2_lsub2 = value->rValue;
    mod->HSM2_lsub2_Given = TRUE;
    break;
  case  HSM2_MOD_LSVDS:
    mod->HSM2_lsvds = value->rValue;
    mod->HSM2_lsvds_Given = TRUE;
    break;
  case  HSM2_MOD_LSVBS:
    mod->HSM2_lsvbs = value->rValue;
    mod->HSM2_lsvbs_Given = TRUE;
    break;
  case  HSM2_MOD_LSVGS:
    mod->HSM2_lsvgs = value->rValue;
    mod->HSM2_lsvgs_Given = TRUE;
    break;
  case  HSM2_MOD_LNSTI:
    mod->HSM2_lnsti = value->rValue;
    mod->HSM2_lnsti_Given = TRUE;
    break;
  case  HSM2_MOD_LWSTI:
    mod->HSM2_lwsti = value->rValue;
    mod->HSM2_lwsti_Given = TRUE;
    break;
  case  HSM2_MOD_LSCSTI1:
    mod->HSM2_lscsti1 = value->rValue;
    mod->HSM2_lscsti1_Given = TRUE;
    break;
  case  HSM2_MOD_LSCSTI2:
    mod->HSM2_lscsti2 = value->rValue;
    mod->HSM2_lscsti2_Given = TRUE;
    break;
  case  HSM2_MOD_LVTHSTI:
    mod->HSM2_lvthsti = value->rValue;
    mod->HSM2_lvthsti_Given = TRUE;
    break;
  case  HSM2_MOD_LMUESTI1:
    mod->HSM2_lmuesti1 = value->rValue;
    mod->HSM2_lmuesti1_Given = TRUE;
    break;
  case  HSM2_MOD_LMUESTI2:
    mod->HSM2_lmuesti2 = value->rValue;
    mod->HSM2_lmuesti2_Given = TRUE;
    break;
  case  HSM2_MOD_LMUESTI3:
    mod->HSM2_lmuesti3 = value->rValue;
    mod->HSM2_lmuesti3_Given = TRUE;
    break;
  case  HSM2_MOD_LNSUBPSTI1:
    mod->HSM2_lnsubpsti1 = value->rValue;
    mod->HSM2_lnsubpsti1_Given = TRUE;
    break;
  case  HSM2_MOD_LNSUBPSTI2:
    mod->HSM2_lnsubpsti2 = value->rValue;
    mod->HSM2_lnsubpsti2_Given = TRUE;
    break;
  case  HSM2_MOD_LNSUBPSTI3:
    mod->HSM2_lnsubpsti3 = value->rValue;
    mod->HSM2_lnsubpsti3_Given = TRUE;
    break;
  case HSM2_MOD_LNSUBCSTI1:
    mod->HSM2_lnsubcsti1 = value->rValue;
    mod->HSM2_lnsubcsti1_Given = TRUE;
    break;
  case HSM2_MOD_LNSUBCSTI2:
    mod->HSM2_lnsubcsti2 = value->rValue;
    mod->HSM2_lnsubcsti2_Given = TRUE;
    break;
  case HSM2_MOD_LNSUBCSTI3:
    mod->HSM2_lnsubcsti3 = value->rValue;
    mod->HSM2_lnsubcsti3_Given = TRUE;
    break;
  case  HSM2_MOD_LCGSO:
    mod->HSM2_lcgso = value->rValue;
    mod->HSM2_lcgso_Given = TRUE;
    break;
  case  HSM2_MOD_LCGDO:
    mod->HSM2_lcgdo = value->rValue;
    mod->HSM2_lcgdo_Given = TRUE;
    break;
  case  HSM2_MOD_LJS0:
    mod->HSM2_ljs0 = value->rValue;
    mod->HSM2_ljs0_Given = TRUE;
    break;
  case  HSM2_MOD_LJS0SW:
    mod->HSM2_ljs0sw = value->rValue;
    mod->HSM2_ljs0sw_Given = TRUE;
    break;
  case  HSM2_MOD_LNJ:
    mod->HSM2_lnj = value->rValue;
    mod->HSM2_lnj_Given = TRUE;
    break;
  case  HSM2_MOD_LCISBK:
    mod->HSM2_lcisbk = value->rValue;
    mod->HSM2_lcisbk_Given = TRUE;
    break;
  case  HSM2_MOD_LCLM1:
    mod->HSM2_lclm1 = value->rValue;
    mod->HSM2_lclm1_Given = TRUE;
    break;
  case  HSM2_MOD_LCLM2:
    mod->HSM2_lclm2 = value->rValue;
    mod->HSM2_lclm2_Given = TRUE;
    break;
  case  HSM2_MOD_LCLM3:
    mod->HSM2_lclm3 = value->rValue;
    mod->HSM2_lclm3_Given = TRUE;
    break;
  case  HSM2_MOD_LWFC:
    mod->HSM2_lwfc = value->rValue;
    mod->HSM2_lwfc_Given = TRUE;
    break;
  case  HSM2_MOD_LGIDL1:
    mod->HSM2_lgidl1 = value->rValue;
    mod->HSM2_lgidl1_Given = TRUE;
    break;
  case  HSM2_MOD_LGIDL2:
    mod->HSM2_lgidl2 = value->rValue;
    mod->HSM2_lgidl2_Given = TRUE;
    break;
  case  HSM2_MOD_LGLEAK1:
    mod->HSM2_lgleak1 = value->rValue;
    mod->HSM2_lgleak1_Given = TRUE;
    break;
  case  HSM2_MOD_LGLEAK2:
    mod->HSM2_lgleak2 = value->rValue;
    mod->HSM2_lgleak2_Given = TRUE;
    break;
  case  HSM2_MOD_LGLEAK3:
    mod->HSM2_lgleak3 = value->rValue;
    mod->HSM2_lgleak3_Given = TRUE;
    break;
  case  HSM2_MOD_LGLEAK6:
    mod->HSM2_lgleak6 = value->rValue;
    mod->HSM2_lgleak6_Given = TRUE;
    break;
  case  HSM2_MOD_LGLKSD1:
    mod->HSM2_lglksd1 = value->rValue;
    mod->HSM2_lglksd1_Given = TRUE;
    break;
  case  HSM2_MOD_LGLKSD2:
    mod->HSM2_lglksd2 = value->rValue;
    mod->HSM2_lglksd2_Given = TRUE;
    break;
  case  HSM2_MOD_LGLKB1:
    mod->HSM2_lglkb1 = value->rValue;
    mod->HSM2_lglkb1_Given = TRUE;
    break;
  case  HSM2_MOD_LGLKB2:
    mod->HSM2_lglkb2 = value->rValue;
    mod->HSM2_lglkb2_Given = TRUE;
    break;
  case  HSM2_MOD_LNFTRP:
    mod->HSM2_lnftrp = value->rValue;
    mod->HSM2_lnftrp_Given = TRUE;
    break;
  case  HSM2_MOD_LNFALP:
    mod->HSM2_lnfalp = value->rValue;
    mod->HSM2_lnfalp_Given = TRUE;
    break;
  case  HSM2_MOD_LVDIFFJ:
    mod->HSM2_lvdiffj = value->rValue;
    mod->HSM2_lvdiffj_Given = TRUE;
    break;
  case  HSM2_MOD_LIBPC1:
    mod->HSM2_libpc1 = value->rValue;
    mod->HSM2_libpc1_Given = TRUE;
    break;
  case  HSM2_MOD_LIBPC2:
    mod->HSM2_libpc2 = value->rValue;
    mod->HSM2_libpc2_Given = TRUE;
    break;

  /* Width dependence */
  case  HSM2_MOD_WVMAX:
    mod->HSM2_wvmax = value->rValue;
    mod->HSM2_wvmax_Given = TRUE;
    break;
  case  HSM2_MOD_WBGTMP1:
    mod->HSM2_wbgtmp1 = value->rValue;
    mod->HSM2_wbgtmp1_Given = TRUE;
    break;
  case  HSM2_MOD_WBGTMP2:
    mod->HSM2_wbgtmp2 = value->rValue;
    mod->HSM2_wbgtmp2_Given = TRUE;
    break;
  case  HSM2_MOD_WEG0:
    mod->HSM2_weg0 = value->rValue;
    mod->HSM2_weg0_Given = TRUE;
    break;
  case  HSM2_MOD_WLOVER:
    mod->HSM2_wlover = value->rValue;
    mod->HSM2_wlover_Given = TRUE;
    break;
  case  HSM2_MOD_WVFBOVER:
    mod->HSM2_wvfbover = value->rValue;
    mod->HSM2_wvfbover_Given = TRUE;
    break;
  case  HSM2_MOD_WNOVER:
    mod->HSM2_wnover = value->rValue;
    mod->HSM2_wnover_Given = TRUE;
    break;
  case  HSM2_MOD_WWL2:
    mod->HSM2_wwl2 = value->rValue;
    mod->HSM2_wwl2_Given = TRUE;
    break;
  case  HSM2_MOD_WVFBC:
    mod->HSM2_wvfbc = value->rValue;
    mod->HSM2_wvfbc_Given = TRUE;
    break;
  case  HSM2_MOD_WNSUBC:
    mod->HSM2_wnsubc = value->rValue;
    mod->HSM2_wnsubc_Given = TRUE;
    break;
  case  HSM2_MOD_WNSUBP:
    mod->HSM2_wnsubp = value->rValue;
    mod->HSM2_wnsubp_Given = TRUE;
    break;
  case  HSM2_MOD_WSCP1:
    mod->HSM2_wscp1 = value->rValue;
    mod->HSM2_wscp1_Given = TRUE;
    break;
  case  HSM2_MOD_WSCP2:
    mod->HSM2_wscp2 = value->rValue;
    mod->HSM2_wscp2_Given = TRUE;
    break;
  case  HSM2_MOD_WSCP3:
    mod->HSM2_wscp3 = value->rValue;
    mod->HSM2_wscp3_Given = TRUE;
    break;
  case  HSM2_MOD_WSC1:
    mod->HSM2_wsc1 = value->rValue;
    mod->HSM2_wsc1_Given = TRUE;
    break;
  case  HSM2_MOD_WSC2:
    mod->HSM2_wsc2 = value->rValue;
    mod->HSM2_wsc2_Given = TRUE;
    break;
  case  HSM2_MOD_WSC3:
    mod->HSM2_wsc3 = value->rValue;
    mod->HSM2_wsc3_Given = TRUE;
    break;
  case  HSM2_MOD_WSC4:
    mod->HSM2_wsc4 = value->rValue;
    mod->HSM2_wsc4_Given = TRUE;
    break;
  case  HSM2_MOD_WPGD1:
    mod->HSM2_wpgd1 = value->rValue;
    mod->HSM2_wpgd1_Given = TRUE;
    break;
//case  HSM2_MOD_WPGD3:
//  mod->HSM2_wpgd3 = value->rValue;
//  mod->HSM2_wpgd3_Given = TRUE;
//  break;
  case  HSM2_MOD_WNDEP:
    mod->HSM2_wndep = value->rValue;
    mod->HSM2_wndep_Given = TRUE;
    break;
  case  HSM2_MOD_WNINV:
    mod->HSM2_wninv = value->rValue;
    mod->HSM2_wninv_Given = TRUE;
    break;
  case  HSM2_MOD_WMUECB0:
    mod->HSM2_wmuecb0 = value->rValue;
    mod->HSM2_wmuecb0_Given = TRUE;
    break;
  case  HSM2_MOD_WMUECB1:
    mod->HSM2_wmuecb1 = value->rValue;
    mod->HSM2_wmuecb1_Given = TRUE;
    break;
  case  HSM2_MOD_WMUEPH1:
    mod->HSM2_wmueph1 = value->rValue;
    mod->HSM2_wmueph1_Given = TRUE;
    break;
  case  HSM2_MOD_WVTMP:
    mod->HSM2_wvtmp = value->rValue;
    mod->HSM2_wvtmp_Given = TRUE;
    break;
  case  HSM2_MOD_WWVTH0:
    mod->HSM2_wwvth0 = value->rValue;
    mod->HSM2_wwvth0_Given = TRUE;
    break;
  case  HSM2_MOD_WMUESR1:
    mod->HSM2_wmuesr1 = value->rValue;
    mod->HSM2_wmuesr1_Given = TRUE;
    break;
  case  HSM2_MOD_WMUETMP:
    mod->HSM2_wmuetmp = value->rValue;
    mod->HSM2_wmuetmp_Given = TRUE;
    break;
  case  HSM2_MOD_WSUB1:
    mod->HSM2_wsub1 = value->rValue;
    mod->HSM2_wsub1_Given = TRUE;
    break;
  case  HSM2_MOD_WSUB2:
    mod->HSM2_wsub2 = value->rValue;
    mod->HSM2_wsub2_Given = TRUE;
    break;
  case  HSM2_MOD_WSVDS:
    mod->HSM2_wsvds = value->rValue;
    mod->HSM2_wsvds_Given = TRUE;
    break;
  case  HSM2_MOD_WSVBS:
    mod->HSM2_wsvbs = value->rValue;
    mod->HSM2_wsvbs_Given = TRUE;
    break;
  case  HSM2_MOD_WSVGS:
    mod->HSM2_wsvgs = value->rValue;
    mod->HSM2_wsvgs_Given = TRUE;
    break;
  case  HSM2_MOD_WNSTI:
    mod->HSM2_wnsti = value->rValue;
    mod->HSM2_wnsti_Given = TRUE;
    break;
  case  HSM2_MOD_WWSTI:
    mod->HSM2_wwsti = value->rValue;
    mod->HSM2_wwsti_Given = TRUE;
    break;
  case  HSM2_MOD_WSCSTI1:
    mod->HSM2_wscsti1 = value->rValue;
    mod->HSM2_wscsti1_Given = TRUE;
    break;
  case  HSM2_MOD_WSCSTI2:
    mod->HSM2_wscsti2 = value->rValue;
    mod->HSM2_wscsti2_Given = TRUE;
    break;
  case  HSM2_MOD_WVTHSTI:
    mod->HSM2_wvthsti = value->rValue;
    mod->HSM2_wvthsti_Given = TRUE;
    break;
  case  HSM2_MOD_WMUESTI1:
    mod->HSM2_wmuesti1 = value->rValue;
    mod->HSM2_wmuesti1_Given = TRUE;
    break;
  case  HSM2_MOD_WMUESTI2:
    mod->HSM2_wmuesti2 = value->rValue;
    mod->HSM2_wmuesti2_Given = TRUE;
    break;
  case  HSM2_MOD_WMUESTI3:
    mod->HSM2_wmuesti3 = value->rValue;
    mod->HSM2_wmuesti3_Given = TRUE;
    break;
  case  HSM2_MOD_WNSUBPSTI1:
    mod->HSM2_wnsubpsti1 = value->rValue;
    mod->HSM2_wnsubpsti1_Given = TRUE;
    break;
  case  HSM2_MOD_WNSUBPSTI2:
    mod->HSM2_wnsubpsti2 = value->rValue;
    mod->HSM2_wnsubpsti2_Given = TRUE;
    break;
  case  HSM2_MOD_WNSUBPSTI3:
    mod->HSM2_wnsubpsti3 = value->rValue;
    mod->HSM2_wnsubpsti3_Given = TRUE;
    break;
  case HSM2_MOD_WNSUBCSTI1:
    mod->HSM2_wnsubcsti1 = value->rValue;
    mod->HSM2_wnsubcsti1_Given = TRUE;
    break;
  case HSM2_MOD_WNSUBCSTI2:
    mod->HSM2_wnsubcsti2 = value->rValue;
    mod->HSM2_wnsubcsti2_Given = TRUE;
    break;
  case HSM2_MOD_WNSUBCSTI3:
    mod->HSM2_wnsubcsti3 = value->rValue;
    mod->HSM2_wnsubcsti3_Given = TRUE;
    break;
  case  HSM2_MOD_WCGSO:
    mod->HSM2_wcgso = value->rValue;
    mod->HSM2_wcgso_Given = TRUE;
    break;
  case  HSM2_MOD_WCGDO:
    mod->HSM2_wcgdo = value->rValue;
    mod->HSM2_wcgdo_Given = TRUE;
    break;
  case  HSM2_MOD_WJS0:
    mod->HSM2_wjs0 = value->rValue;
    mod->HSM2_wjs0_Given = TRUE;
    break;
  case  HSM2_MOD_WJS0SW:
    mod->HSM2_wjs0sw = value->rValue;
    mod->HSM2_wjs0sw_Given = TRUE;
    break;
  case  HSM2_MOD_WNJ:
    mod->HSM2_wnj = value->rValue;
    mod->HSM2_wnj_Given = TRUE;
    break;
  case  HSM2_MOD_WCISBK:
    mod->HSM2_wcisbk = value->rValue;
    mod->HSM2_wcisbk_Given = TRUE;
    break;
  case  HSM2_MOD_WCLM1:
    mod->HSM2_wclm1 = value->rValue;
    mod->HSM2_wclm1_Given = TRUE;
    break;
  case  HSM2_MOD_WCLM2:
    mod->HSM2_wclm2 = value->rValue;
    mod->HSM2_wclm2_Given = TRUE;
    break;
  case  HSM2_MOD_WCLM3:
    mod->HSM2_wclm3 = value->rValue;
    mod->HSM2_wclm3_Given = TRUE;
    break;
  case  HSM2_MOD_WWFC:
    mod->HSM2_wwfc = value->rValue;
    mod->HSM2_wwfc_Given = TRUE;
    break;
  case  HSM2_MOD_WGIDL1:
    mod->HSM2_wgidl1 = value->rValue;
    mod->HSM2_wgidl1_Given = TRUE;
    break;
  case  HSM2_MOD_WGIDL2:
    mod->HSM2_wgidl2 = value->rValue;
    mod->HSM2_wgidl2_Given = TRUE;
    break;
  case  HSM2_MOD_WGLEAK1:
    mod->HSM2_wgleak1 = value->rValue;
    mod->HSM2_wgleak1_Given = TRUE;
    break;
  case  HSM2_MOD_WGLEAK2:
    mod->HSM2_wgleak2 = value->rValue;
    mod->HSM2_wgleak2_Given = TRUE;
    break;
  case  HSM2_MOD_WGLEAK3:
    mod->HSM2_wgleak3 = value->rValue;
    mod->HSM2_wgleak3_Given = TRUE;
    break;
  case  HSM2_MOD_WGLEAK6:
    mod->HSM2_wgleak6 = value->rValue;
    mod->HSM2_wgleak6_Given = TRUE;
    break;
  case  HSM2_MOD_WGLKSD1:
    mod->HSM2_wglksd1 = value->rValue;
    mod->HSM2_wglksd1_Given = TRUE;
    break;
  case  HSM2_MOD_WGLKSD2:
    mod->HSM2_wglksd2 = value->rValue;
    mod->HSM2_wglksd2_Given = TRUE;
    break;
  case  HSM2_MOD_WGLKB1:
    mod->HSM2_wglkb1 = value->rValue;
    mod->HSM2_wglkb1_Given = TRUE;
    break;
  case  HSM2_MOD_WGLKB2:
    mod->HSM2_wglkb2 = value->rValue;
    mod->HSM2_wglkb2_Given = TRUE;
    break;
  case  HSM2_MOD_WNFTRP:
    mod->HSM2_wnftrp = value->rValue;
    mod->HSM2_wnftrp_Given = TRUE;
    break;
  case  HSM2_MOD_WNFALP:
    mod->HSM2_wnfalp = value->rValue;
    mod->HSM2_wnfalp_Given = TRUE;
    break;
  case  HSM2_MOD_WVDIFFJ:
    mod->HSM2_wvdiffj = value->rValue;
    mod->HSM2_wvdiffj_Given = TRUE;
    break;
  case  HSM2_MOD_WIBPC1:
    mod->HSM2_wibpc1 = value->rValue;
    mod->HSM2_wibpc1_Given = TRUE;
    break;
  case  HSM2_MOD_WIBPC2:
    mod->HSM2_wibpc2 = value->rValue;
    mod->HSM2_wibpc2_Given = TRUE;
    break;

  /* Cross-term dependence */
  case  HSM2_MOD_PVMAX:
    mod->HSM2_pvmax = value->rValue;
    mod->HSM2_pvmax_Given = TRUE;
    break;
  case  HSM2_MOD_PBGTMP1:
    mod->HSM2_pbgtmp1 = value->rValue;
    mod->HSM2_pbgtmp1_Given = TRUE;
    break;
  case  HSM2_MOD_PBGTMP2:
    mod->HSM2_pbgtmp2 = value->rValue;
    mod->HSM2_pbgtmp2_Given = TRUE;
    break;
  case  HSM2_MOD_PEG0:
    mod->HSM2_peg0 = value->rValue;
    mod->HSM2_peg0_Given = TRUE;
    break;
  case  HSM2_MOD_PLOVER:
    mod->HSM2_plover = value->rValue;
    mod->HSM2_plover_Given = TRUE;
    break;
  case  HSM2_MOD_PVFBOVER:
    mod->HSM2_pvfbover = value->rValue;
    mod->HSM2_pvfbover_Given = TRUE;
    break;
  case  HSM2_MOD_PNOVER:
    mod->HSM2_pnover = value->rValue;
    mod->HSM2_pnover_Given = TRUE;
    break;
  case  HSM2_MOD_PWL2:
    mod->HSM2_pwl2 = value->rValue;
    mod->HSM2_pwl2_Given = TRUE;
    break;
  case  HSM2_MOD_PVFBC:
    mod->HSM2_pvfbc = value->rValue;
    mod->HSM2_pvfbc_Given = TRUE;
    break;
  case  HSM2_MOD_PNSUBC:
    mod->HSM2_pnsubc = value->rValue;
    mod->HSM2_pnsubc_Given = TRUE;
    break;
  case  HSM2_MOD_PNSUBP:
    mod->HSM2_pnsubp = value->rValue;
    mod->HSM2_pnsubp_Given = TRUE;
    break;
  case  HSM2_MOD_PSCP1:
    mod->HSM2_pscp1 = value->rValue;
    mod->HSM2_pscp1_Given = TRUE;
    break;
  case  HSM2_MOD_PSCP2:
    mod->HSM2_pscp2 = value->rValue;
    mod->HSM2_pscp2_Given = TRUE;
    break;
  case  HSM2_MOD_PSCP3:
    mod->HSM2_pscp3 = value->rValue;
    mod->HSM2_pscp3_Given = TRUE;
    break;
  case  HSM2_MOD_PSC1:
    mod->HSM2_psc1 = value->rValue;
    mod->HSM2_psc1_Given = TRUE;
    break;
  case  HSM2_MOD_PSC2:
    mod->HSM2_psc2 = value->rValue;
    mod->HSM2_psc2_Given = TRUE;
    break;
  case  HSM2_MOD_PSC3:
    mod->HSM2_psc3 = value->rValue;
    mod->HSM2_psc3_Given = TRUE;
    break;
  case  HSM2_MOD_PSC4:
    mod->HSM2_psc4 = value->rValue;
    mod->HSM2_psc4_Given = TRUE;
    break;
  case  HSM2_MOD_PPGD1:
    mod->HSM2_ppgd1 = value->rValue;
    mod->HSM2_ppgd1_Given = TRUE;
    break;
//case  HSM2_MOD_PPGD3:
//  mod->HSM2_ppgd3 = value->rValue;
//  mod->HSM2_ppgd3_Given = TRUE;
//  break;
  case  HSM2_MOD_PNDEP:
    mod->HSM2_pndep = value->rValue;
    mod->HSM2_pndep_Given = TRUE;
    break;
  case  HSM2_MOD_PNINV:
    mod->HSM2_pninv = value->rValue;
    mod->HSM2_pninv_Given = TRUE;
    break;
  case  HSM2_MOD_PMUECB0:
    mod->HSM2_pmuecb0 = value->rValue;
    mod->HSM2_pmuecb0_Given = TRUE;
    break;
  case  HSM2_MOD_PMUECB1:
    mod->HSM2_pmuecb1 = value->rValue;
    mod->HSM2_pmuecb1_Given = TRUE;
    break;
  case  HSM2_MOD_PMUEPH1:
    mod->HSM2_pmueph1 = value->rValue;
    mod->HSM2_pmueph1_Given = TRUE;
    break;
  case  HSM2_MOD_PVTMP:
    mod->HSM2_pvtmp = value->rValue;
    mod->HSM2_pvtmp_Given = TRUE;
    break;
  case  HSM2_MOD_PWVTH0:
    mod->HSM2_pwvth0 = value->rValue;
    mod->HSM2_pwvth0_Given = TRUE;
    break;
  case  HSM2_MOD_PMUESR1:
    mod->HSM2_pmuesr1 = value->rValue;
    mod->HSM2_pmuesr1_Given = TRUE;
    break;
  case  HSM2_MOD_PMUETMP:
    mod->HSM2_pmuetmp = value->rValue;
    mod->HSM2_pmuetmp_Given = TRUE;
    break;
  case  HSM2_MOD_PSUB1:
    mod->HSM2_psub1 = value->rValue;
    mod->HSM2_psub1_Given = TRUE;
    break;
  case  HSM2_MOD_PSUB2:
    mod->HSM2_psub2 = value->rValue;
    mod->HSM2_psub2_Given = TRUE;
    break;
  case  HSM2_MOD_PSVDS:
    mod->HSM2_psvds = value->rValue;
    mod->HSM2_psvds_Given = TRUE;
    break;
  case  HSM2_MOD_PSVBS:
    mod->HSM2_psvbs = value->rValue;
    mod->HSM2_psvbs_Given = TRUE;
    break;
  case  HSM2_MOD_PSVGS:
    mod->HSM2_psvgs = value->rValue;
    mod->HSM2_psvgs_Given = TRUE;
    break;
  case  HSM2_MOD_PNSTI:
    mod->HSM2_pnsti = value->rValue;
    mod->HSM2_pnsti_Given = TRUE;
    break;
  case  HSM2_MOD_PWSTI:
    mod->HSM2_pwsti = value->rValue;
    mod->HSM2_pwsti_Given = TRUE;
    break;
  case  HSM2_MOD_PSCSTI1:
    mod->HSM2_pscsti1 = value->rValue;
    mod->HSM2_pscsti1_Given = TRUE;
    break;
  case  HSM2_MOD_PSCSTI2:
    mod->HSM2_pscsti2 = value->rValue;
    mod->HSM2_pscsti2_Given = TRUE;
    break;
  case  HSM2_MOD_PVTHSTI:
    mod->HSM2_pvthsti = value->rValue;
    mod->HSM2_pvthsti_Given = TRUE;
    break;
  case  HSM2_MOD_PMUESTI1:
    mod->HSM2_pmuesti1 = value->rValue;
    mod->HSM2_pmuesti1_Given = TRUE;
    break;
  case  HSM2_MOD_PMUESTI2:
    mod->HSM2_pmuesti2 = value->rValue;
    mod->HSM2_pmuesti2_Given = TRUE;
    break;
  case  HSM2_MOD_PMUESTI3:
    mod->HSM2_pmuesti3 = value->rValue;
    mod->HSM2_pmuesti3_Given = TRUE;
    break;
  case  HSM2_MOD_PNSUBPSTI1:
    mod->HSM2_pnsubpsti1 = value->rValue;
    mod->HSM2_pnsubpsti1_Given = TRUE;
    break;
  case  HSM2_MOD_PNSUBPSTI2:
    mod->HSM2_pnsubpsti2 = value->rValue;
    mod->HSM2_pnsubpsti2_Given = TRUE;
    break;
  case  HSM2_MOD_PNSUBPSTI3:
    mod->HSM2_pnsubpsti3 = value->rValue;
    mod->HSM2_pnsubpsti3_Given = TRUE;
    break;
  case HSM2_MOD_PNSUBCSTI1:
    mod->HSM2_pnsubcsti1 = value->rValue;
    mod->HSM2_pnsubcsti1_Given = TRUE;
    break;
  case HSM2_MOD_PNSUBCSTI2:
    mod->HSM2_pnsubcsti2 = value->rValue;
    mod->HSM2_pnsubcsti2_Given = TRUE;
    break;
  case HSM2_MOD_PNSUBCSTI3:
    mod->HSM2_pnsubcsti3 = value->rValue;
    mod->HSM2_pnsubcsti3_Given = TRUE;
    break;
  case  HSM2_MOD_PCGSO:
    mod->HSM2_pcgso = value->rValue;
    mod->HSM2_pcgso_Given = TRUE;
    break;
  case  HSM2_MOD_PCGDO:
    mod->HSM2_pcgdo = value->rValue;
    mod->HSM2_pcgdo_Given = TRUE;
    break;
  case  HSM2_MOD_PJS0:
    mod->HSM2_pjs0 = value->rValue;
    mod->HSM2_pjs0_Given = TRUE;
    break;
  case  HSM2_MOD_PJS0SW:
    mod->HSM2_pjs0sw = value->rValue;
    mod->HSM2_pjs0sw_Given = TRUE;
    break;
  case  HSM2_MOD_PNJ:
    mod->HSM2_pnj = value->rValue;
    mod->HSM2_pnj_Given = TRUE;
    break;
  case  HSM2_MOD_PCISBK:
    mod->HSM2_pcisbk = value->rValue;
    mod->HSM2_pcisbk_Given = TRUE;
    break;
  case  HSM2_MOD_PCLM1:
    mod->HSM2_pclm1 = value->rValue;
    mod->HSM2_pclm1_Given = TRUE;
    break;
  case  HSM2_MOD_PCLM2:
    mod->HSM2_pclm2 = value->rValue;
    mod->HSM2_pclm2_Given = TRUE;
    break;
  case  HSM2_MOD_PCLM3:
    mod->HSM2_pclm3 = value->rValue;
    mod->HSM2_pclm3_Given = TRUE;
    break;
  case  HSM2_MOD_PWFC:
    mod->HSM2_pwfc = value->rValue;
    mod->HSM2_pwfc_Given = TRUE;
    break;
  case  HSM2_MOD_PGIDL1:
    mod->HSM2_pgidl1 = value->rValue;
    mod->HSM2_pgidl1_Given = TRUE;
    break;
  case  HSM2_MOD_PGIDL2:
    mod->HSM2_pgidl2 = value->rValue;
    mod->HSM2_pgidl2_Given = TRUE;
    break;
  case  HSM2_MOD_PGLEAK1:
    mod->HSM2_pgleak1 = value->rValue;
    mod->HSM2_pgleak1_Given = TRUE;
    break;
  case  HSM2_MOD_PGLEAK2:
    mod->HSM2_pgleak2 = value->rValue;
    mod->HSM2_pgleak2_Given = TRUE;
    break;
  case  HSM2_MOD_PGLEAK3:
    mod->HSM2_pgleak3 = value->rValue;
    mod->HSM2_pgleak3_Given = TRUE;
    break;
  case  HSM2_MOD_PGLEAK6:
    mod->HSM2_pgleak6 = value->rValue;
    mod->HSM2_pgleak6_Given = TRUE;
    break;
  case  HSM2_MOD_PGLKSD1:
    mod->HSM2_pglksd1 = value->rValue;
    mod->HSM2_pglksd1_Given = TRUE;
    break;
  case  HSM2_MOD_PGLKSD2:
    mod->HSM2_pglksd2 = value->rValue;
    mod->HSM2_pglksd2_Given = TRUE;
    break;
  case  HSM2_MOD_PGLKB1:
    mod->HSM2_pglkb1 = value->rValue;
    mod->HSM2_pglkb1_Given = TRUE;
    break;
  case  HSM2_MOD_PGLKB2:
    mod->HSM2_pglkb2 = value->rValue;
    mod->HSM2_pglkb2_Given = TRUE;
    break;
  case  HSM2_MOD_PNFTRP:
    mod->HSM2_pnftrp = value->rValue;
    mod->HSM2_pnftrp_Given = TRUE;
    break;
  case  HSM2_MOD_PNFALP:
    mod->HSM2_pnfalp = value->rValue;
    mod->HSM2_pnfalp_Given = TRUE;
    break;
  case  HSM2_MOD_PVDIFFJ:
    mod->HSM2_pvdiffj = value->rValue;
    mod->HSM2_pvdiffj_Given = TRUE;
    break;
  case  HSM2_MOD_PIBPC1:
    mod->HSM2_pibpc1 = value->rValue;
    mod->HSM2_pibpc1_Given = TRUE;
    break;
  case  HSM2_MOD_PIBPC2:
    mod->HSM2_pibpc2 = value->rValue;
    mod->HSM2_pibpc2_Given = TRUE;
    break;

  case HSM2_MOD_VGS_MAX:
      mod->HSM2vgsMax = value->rValue;
      mod->HSM2vgsMaxGiven = TRUE;
      break;
  case HSM2_MOD_VGD_MAX:
      mod->HSM2vgdMax = value->rValue;
      mod->HSM2vgdMaxGiven = TRUE;
      break;
  case HSM2_MOD_VGB_MAX:
      mod->HSM2vgbMax = value->rValue;
      mod->HSM2vgbMaxGiven = TRUE;
      break;
  case HSM2_MOD_VDS_MAX:
      mod->HSM2vdsMax = value->rValue;
      mod->HSM2vdsMaxGiven = TRUE;
      break;
  case HSM2_MOD_VBS_MAX:
      mod->HSM2vbsMax = value->rValue;
      mod->HSM2vbsMaxGiven = TRUE;
      break;
  case HSM2_MOD_VBD_MAX:
      mod->HSM2vbdMax = value->rValue;
      mod->HSM2vbdMaxGiven = TRUE;
      break;
  case HSM2_MOD_VGSR_MAX:
      mod->HSM2vgsrMax = value->rValue;
      mod->HSM2vgsrMaxGiven = TRUE;
      break;
  case HSM2_MOD_VGDR_MAX:
      mod->HSM2vgdrMax = value->rValue;
      mod->HSM2vgdrMaxGiven = TRUE;
      break;
  case HSM2_MOD_VGBR_MAX:
      mod->HSM2vgbrMax = value->rValue;
      mod->HSM2vgbrMaxGiven = TRUE;
      break;
  case HSM2_MOD_VBSR_MAX:
      mod->HSM2vbsrMax = value->rValue;
      mod->HSM2vbsrMaxGiven = TRUE;
      break;
  case HSM2_MOD_VBDR_MAX:
      mod->HSM2vbdrMax = value->rValue;
      mod->HSM2vbdrMaxGiven = TRUE;
      break;

  default:
    return(E_BADPARM);
  }
  return(OK);
}

