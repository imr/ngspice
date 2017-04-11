/***********************************************************************

 HiSIM (Hiroshima University STARC IGFET Model)
 Copyright (C) 2014 Hiroshima University & STARC

 MODEL NAME : HiSIM_HV 
 ( VERSION : 2  SUBVERSION : 2  REVISION : 0 ) 
 Model Parameter 'VERSION' : 2.20
 FILE : hsmhvmpar.c

 DATE : 2014.6.11

 released by
                Hiroshima University &
                Semiconductor Technology Academic Research Center (STARC)
***********************************************************************/

/**********************************************************************

The following source code, and all copyrights, trade secrets or other
intellectual property rights in and to the source code in its entirety,
is owned by the Hiroshima University and the STARC organization.

All users need to follow the "HISIM_HV Distribution Statement and
Copyright Notice" attached to HiSIM_HV model.

-----HISIM_HV Distribution Statement and Copyright Notice--------------

Software is distributed as is, completely without warranty or service
support. Hiroshima University or STARC and its employees are not liable
for the condition or performance of the software.

Hiroshima University and STARC own the copyright and grant users a perpetual,
irrevocable, worldwide, non-exclusive, royalty-free license with respect 
to the software as set forth below.   

Hiroshima University and STARC hereby disclaims all implied warranties.

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

Toshimasa Asahara, President, Hiroshima University
Mitiko Miura-Mattausch, Professor, Hiroshima University
Katsuhiro Shimohigashi, President&CEO, STARC
June 2008 (revised October 2011) 
*************************************************************************/

#include "ngspice/ngspice.h"
#include "hsmhv2def.h"
#include "ngspice/ifsim.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

int HSMHV2mParam(
     int param,
     IFvalue *value,
     GENmodel *inMod)
{
  HSMHV2model *mod = (HSMHV2model*)inMod;
  switch (param) {
  case  HSMHV2_MOD_NMOS  :
    if (value->iValue) {
      mod->HSMHV2_type = 1;
      mod->HSMHV2_type_Given = TRUE;
    }
    break;
  case  HSMHV2_MOD_PMOS  :
    if (value->iValue) {
      mod->HSMHV2_type = - 1;
      mod->HSMHV2_type_Given = TRUE;
    }
    break;
  case  HSMHV2_MOD_LEVEL:
    mod->HSMHV2_level = value->iValue;
    mod->HSMHV2_level_Given = TRUE;
    break;
  case  HSMHV2_MOD_INFO:
    mod->HSMHV2_info = value->iValue;
    mod->HSMHV2_info_Given = TRUE;
    break;
  case HSMHV2_MOD_NOISE:
    mod->HSMHV2_noise = value->iValue;
    mod->HSMHV2_noise_Given = TRUE;
    break;
  case HSMHV2_MOD_VERSION:
    mod->HSMHV2_version = value->sValue;
    mod->HSMHV2_version_Given = TRUE;
    break;
  case HSMHV2_MOD_SHOW:
    mod->HSMHV2_show = value->iValue;
    mod->HSMHV2_show_Given = TRUE;
    break;
  case  HSMHV2_MOD_CORSRD:
    mod->HSMHV2_corsrd = value->iValue;
    mod->HSMHV2_corsrd_Given = TRUE;
    break;
  case  HSMHV2_MOD_CORG:
    mod->HSMHV2_corg = value->iValue;
    mod->HSMHV2_corg_Given = TRUE;
    break;
  case  HSMHV2_MOD_COIPRV:
    mod->HSMHV2_coiprv = value->iValue;
    mod->HSMHV2_coiprv_Given = TRUE;
    break;
  case  HSMHV2_MOD_COPPRV:
    mod->HSMHV2_copprv = value->iValue;
    mod->HSMHV2_copprv_Given = TRUE;
    break;
  case  HSMHV2_MOD_COADOV:
    mod->HSMHV2_coadov = value->iValue;
    mod->HSMHV2_coadov_Given = TRUE;
    break;
  case  HSMHV2_MOD_COISUB:
    mod->HSMHV2_coisub = value->iValue;
    mod->HSMHV2_coisub_Given = TRUE;
    break;
  case  HSMHV2_MOD_COIIGS:
    mod->HSMHV2_coiigs = value->iValue;
    mod->HSMHV2_coiigs_Given = TRUE;
    break;
  case  HSMHV2_MOD_COGIDL:
    mod->HSMHV2_cogidl = value->iValue;
    mod->HSMHV2_cogidl_Given = TRUE;
    break;
  case  HSMHV2_MOD_COOVLP:
    mod->HSMHV2_coovlp = value->iValue;
    mod->HSMHV2_coovlp_Given = TRUE;
    break;
  case  HSMHV2_MOD_COOVLPS:
    mod->HSMHV2_coovlps = value->iValue;
    mod->HSMHV2_coovlps_Given = TRUE;
    break;
  case  HSMHV2_MOD_COFLICK:
    mod->HSMHV2_coflick = value->iValue;
    mod->HSMHV2_coflick_Given = TRUE;
    break;
  case  HSMHV2_MOD_COISTI:
    mod->HSMHV2_coisti = value->iValue;
    mod->HSMHV2_coisti_Given = TRUE;
    break;
  case  HSMHV2_MOD_CONQS: /* HiSIMHV */
    mod->HSMHV2_conqs = value->iValue;
    mod->HSMHV2_conqs_Given = TRUE;
    break;
  case  HSMHV2_MOD_CORBNET: 
    mod->HSMHV2_corbnet = value->iValue;
    mod->HSMHV2_corbnet_Given = TRUE;
    break;
  case  HSMHV2_MOD_COTHRML:
    mod->HSMHV2_cothrml = value->iValue;
    mod->HSMHV2_cothrml_Given = TRUE;
    break;
  case  HSMHV2_MOD_COIGN:
    mod->HSMHV2_coign = value->iValue;
    mod->HSMHV2_coign_Given = TRUE;
    break;
  case  HSMHV2_MOD_CODFM:
    mod->HSMHV2_codfm = value->iValue;
    mod->HSMHV2_codfm_Given = TRUE;
    break;
  case  HSMHV2_MOD_COQOVSM:
    mod->HSMHV2_coqovsm = value->iValue;
    mod->HSMHV2_coqovsm_Given = TRUE;
    break;
  case  HSMHV2_MOD_COSELFHEAT: /* Self-heating model */
    mod->HSMHV2_coselfheat = value->iValue;
    mod->HSMHV2_coselfheat_Given = TRUE;
    break;
  case  HSMHV2_MOD_COSUBNODE:
    mod->HSMHV2_cosubnode = value->iValue;
    mod->HSMHV2_cosubnode_Given = TRUE;
    break;
  case  HSMHV2_MOD_COSYM: /* Symmetry model for HV */
    mod->HSMHV2_cosym = value->iValue;
    mod->HSMHV2_cosym_Given = TRUE;
    break;
  case  HSMHV2_MOD_COTEMP:
    mod->HSMHV2_cotemp = value->iValue;
    mod->HSMHV2_cotemp_Given = TRUE;
    break;
  case  HSMHV2_MOD_COLDRIFT:
    mod->HSMHV2_coldrift = value->iValue;
    mod->HSMHV2_coldrift_Given = TRUE;
    break;
  case  HSMHV2_MOD_CORDRIFT:
    mod->HSMHV2_cordrift = value->iValue;
    mod->HSMHV2_cordrift_Given = TRUE;
    break;
  case  HSMHV2_MOD_COERRREP:
    mod->HSMHV2_coerrrep = value->iValue;
    mod->HSMHV2_coerrrep_Given = TRUE;
    break;
  case  HSMHV2_MOD_CODEP:
    mod->HSMHV2_codep = value->iValue;
    mod->HSMHV2_codep_Given = TRUE;
    break;
  case  HSMHV2_MOD_CODDLT:
    mod->HSMHV2_coddlt = value->iValue;
    mod->HSMHV2_coddlt_Given = TRUE;
    break;
  case  HSMHV2_MOD_VMAX:
    mod->HSMHV2_vmax = value->rValue;
    mod->HSMHV2_vmax_Given = TRUE;
    break;
  case  HSMHV2_MOD_VMAXT1:
    mod->HSMHV2_vmaxt1 = value->rValue;
    mod->HSMHV2_vmaxt1_Given = TRUE;
    break;
  case  HSMHV2_MOD_VMAXT2:
    mod->HSMHV2_vmaxt2 = value->rValue;
    mod->HSMHV2_vmaxt2_Given = TRUE;
    break;
  case  HSMHV2_MOD_BGTMP1:
    mod->HSMHV2_bgtmp1 = value->rValue;
    mod->HSMHV2_bgtmp1_Given = TRUE;
    break;
  case  HSMHV2_MOD_BGTMP2:
    mod->HSMHV2_bgtmp2 =  value->rValue;
    mod->HSMHV2_bgtmp2_Given = TRUE;
    break;
  case  HSMHV2_MOD_EG0:
    mod->HSMHV2_eg0 =  value->rValue;
    mod->HSMHV2_eg0_Given = TRUE;
    break;
  case  HSMHV2_MOD_TOX:
    mod->HSMHV2_tox =  value->rValue;
    mod->HSMHV2_tox_Given = TRUE;
    break;
  case  HSMHV2_MOD_XLD:
    mod->HSMHV2_xld = value->rValue;
    mod->HSMHV2_xld_Given = TRUE;
    break;
  case  HSMHV2_MOD_LOVER:
    mod->HSMHV2_lover = value->rValue;
    mod->HSMHV2_lover_Given = TRUE;
    break;
  case  HSMHV2_MOD_LOVERS:
    mod->HSMHV2_lovers = value->rValue;
    mod->HSMHV2_lovers_Given = TRUE;
    break;
  case  HSMHV2_MOD_RDOV11:
    mod->HSMHV2_rdov11 = value->rValue;
    mod->HSMHV2_rdov11_Given = TRUE;
    break;
  case  HSMHV2_MOD_RDOV12:
    mod->HSMHV2_rdov12 = value->rValue;
    mod->HSMHV2_rdov12_Given = TRUE;
    break;
  case  HSMHV2_MOD_RDOV13:
    mod->HSMHV2_rdov13 = value->rValue;
    mod->HSMHV2_rdov13_Given = TRUE;
    break;
  case  HSMHV2_MOD_RDSLP1:
    mod->HSMHV2_rdslp1 = value->rValue;
    mod->HSMHV2_rdslp1_Given = TRUE;
    break;
  case  HSMHV2_MOD_RDICT1:
    mod->HSMHV2_rdict1= value->rValue;
    mod->HSMHV2_rdict1_Given = TRUE;
    break;
  case  HSMHV2_MOD_RDSLP2:
    mod->HSMHV2_rdslp2 = value->rValue;
    mod->HSMHV2_rdslp2_Given = TRUE;
    break;
  case  HSMHV2_MOD_RDICT2:
    mod->HSMHV2_rdict2 = value->rValue;
    mod->HSMHV2_rdict2_Given = TRUE;
    break;
  case  HSMHV2_MOD_LOVERLD:
    mod->HSMHV2_loverld = value->rValue;
    mod->HSMHV2_loverld_Given = TRUE;
    break;
  case  HSMHV2_MOD_LDRIFT1:
    mod->HSMHV2_ldrift1 = value->rValue;
    mod->HSMHV2_ldrift1_Given = TRUE;
    break;
  case  HSMHV2_MOD_LDRIFT2:
    mod->HSMHV2_ldrift2 = value->rValue;
    mod->HSMHV2_ldrift2_Given = TRUE;
    break;
  case  HSMHV2_MOD_LDRIFT1S:
    mod->HSMHV2_ldrift1s = value->rValue;
    mod->HSMHV2_ldrift1s_Given = TRUE;
    break;
  case  HSMHV2_MOD_LDRIFT2S:
    mod->HSMHV2_ldrift2s = value->rValue;
    mod->HSMHV2_ldrift2s_Given = TRUE;
    break;
  case  HSMHV2_MOD_SUBLD1:
    mod->HSMHV2_subld1 = value->rValue;
    mod->HSMHV2_subld1_Given = TRUE;
    break;
  case  HSMHV2_MOD_SUBLD1L:
    mod->HSMHV2_subld1l = value->rValue;
    mod->HSMHV2_subld1l_Given = TRUE;
    break;
  case  HSMHV2_MOD_SUBLD1LP:
    mod->HSMHV2_subld1lp = value->rValue;
    mod->HSMHV2_subld1lp_Given = TRUE;
    break;
  case  HSMHV2_MOD_SUBLD2:
    mod->HSMHV2_subld2 = value->rValue;
    mod->HSMHV2_subld2_Given = TRUE;
    break;
  case  HSMHV2_MOD_XPDV:
    mod->HSMHV2_xpdv = value->rValue;
    mod->HSMHV2_xpdv_Given = TRUE;
    break;
  case  HSMHV2_MOD_XPVDTH:
    mod->HSMHV2_xpvdth = value->rValue;
    mod->HSMHV2_xpvdth_Given = TRUE;
    break;
  case  HSMHV2_MOD_XPVDTHG:
    mod->HSMHV2_xpvdthg = value->rValue;
    mod->HSMHV2_xpvdthg_Given = TRUE;
    break;
  case  HSMHV2_MOD_DDLTMAX: /* Vdseff */
    mod->HSMHV2_ddltmax = value->rValue;
    mod->HSMHV2_ddltmax_Given = TRUE;
    break;
  case  HSMHV2_MOD_DDLTSLP: /* Vdseff */
    mod->HSMHV2_ddltslp = value->rValue;
    mod->HSMHV2_ddltslp_Given = TRUE;
    break;
  case  HSMHV2_MOD_DDLTICT: /* Vdseff */
    mod->HSMHV2_ddltict = value->rValue;
    mod->HSMHV2_ddltict_Given = TRUE;
    break;
  case  HSMHV2_MOD_VFBOVER:
    mod->HSMHV2_vfbover = value->rValue;
    mod->HSMHV2_vfbover_Given = TRUE;
    break;
  case  HSMHV2_MOD_NOVER:
    mod->HSMHV2_nover = value->rValue;
    mod->HSMHV2_nover_Given = TRUE;
    break;
  case  HSMHV2_MOD_NOVERS:
    mod->HSMHV2_novers = value->rValue;
    mod->HSMHV2_novers_Given = TRUE;
    break;
  case  HSMHV2_MOD_XWD:
    mod->HSMHV2_xwd = value->rValue;
    mod->HSMHV2_xwd_Given = TRUE;
    break;
  case  HSMHV2_MOD_XWDC:
    mod->HSMHV2_xwdc = value->rValue;
    mod->HSMHV2_xwdc_Given = TRUE;
    break;
  case  HSMHV2_MOD_XL:
    mod->HSMHV2_xl = value->rValue;
    mod->HSMHV2_xl_Given = TRUE;
    break;
  case  HSMHV2_MOD_XW:
    mod->HSMHV2_xw = value->rValue;
    mod->HSMHV2_xw_Given = TRUE;
    break;
  case  HSMHV2_MOD_SAREF:
    mod->HSMHV2_saref = value->rValue;
    mod->HSMHV2_saref_Given = TRUE;
    break;
  case  HSMHV2_MOD_SBREF:
    mod->HSMHV2_sbref = value->rValue;
    mod->HSMHV2_sbref_Given = TRUE;
    break;
  case  HSMHV2_MOD_LL:
    mod->HSMHV2_ll = value->rValue;
    mod->HSMHV2_ll_Given = TRUE;
    break;
  case  HSMHV2_MOD_LLD:
    mod->HSMHV2_lld = value->rValue;
    mod->HSMHV2_lld_Given = TRUE;
    break;
  case  HSMHV2_MOD_LLN:
    mod->HSMHV2_lln = value->rValue;
    mod->HSMHV2_lln_Given = TRUE;
    break;
  case  HSMHV2_MOD_WL:
    mod->HSMHV2_wl = value->rValue;
    mod->HSMHV2_wl_Given = TRUE;
    break;
  case  HSMHV2_MOD_WL1:
    mod->HSMHV2_wl1 = value->rValue;
    mod->HSMHV2_wl1_Given = TRUE;
    break;
  case  HSMHV2_MOD_WL1P:
    mod->HSMHV2_wl1p = value->rValue;
    mod->HSMHV2_wl1p_Given = TRUE;
    break;
  case  HSMHV2_MOD_WL2:
    mod->HSMHV2_wl2 = value->rValue;
    mod->HSMHV2_wl2_Given = TRUE;
    break;
  case  HSMHV2_MOD_WL2P:
    mod->HSMHV2_wl2p = value->rValue;
    mod->HSMHV2_wl2p_Given = TRUE;
    break;
  case  HSMHV2_MOD_WLD:
    mod->HSMHV2_wld = value->rValue;
    mod->HSMHV2_wld_Given = TRUE;
    break;
  case  HSMHV2_MOD_WLN:
    mod->HSMHV2_wln = value->rValue;
    mod->HSMHV2_wln_Given = TRUE;
    break;
  case  HSMHV2_MOD_XQY:
    mod->HSMHV2_xqy = value->rValue;
    mod->HSMHV2_xqy_Given = TRUE;
    break;
  case  HSMHV2_MOD_XQY1:
    mod->HSMHV2_xqy1 = value->rValue;
    mod->HSMHV2_xqy1_Given = TRUE;
    break;
  case  HSMHV2_MOD_XQY2:
    mod->HSMHV2_xqy2 = value->rValue;
    mod->HSMHV2_xqy2_Given = TRUE;
    break;
  case  HSMHV2_MOD_RS:
    mod->HSMHV2_rs = value->rValue;
    mod->HSMHV2_rs_Given = TRUE;
    break;
  case  HSMHV2_MOD_RD:
    mod->HSMHV2_rd = value->rValue;
    mod->HSMHV2_rd_Given = TRUE;
    break;
  case  HSMHV2_MOD_RSH:
    mod->HSMHV2_rsh = value->rValue;
    mod->HSMHV2_rsh_Given = TRUE;
    break;
  case  HSMHV2_MOD_RSHG:
    mod->HSMHV2_rshg = value->rValue;
    mod->HSMHV2_rshg_Given = TRUE;
    break;
  case  HSMHV2_MOD_VFBC:
    mod->HSMHV2_vfbc = value->rValue;
    mod->HSMHV2_vfbc_Given = TRUE;
    break;
  case  HSMHV2_MOD_VBI:
    mod->HSMHV2_vbi = value->rValue;
    mod->HSMHV2_vbi_Given = TRUE;
    break;
  case  HSMHV2_MOD_NSUBC:
    mod->HSMHV2_nsubc = value->rValue;
    mod->HSMHV2_nsubc_Given = TRUE;
    break;
  case  HSMHV2_MOD_PARL2:
    mod->HSMHV2_parl2 = value->rValue;
    mod->HSMHV2_parl2_Given = TRUE;
    break;
  case  HSMHV2_MOD_LP:
    mod->HSMHV2_lp = value->rValue;
    mod->HSMHV2_lp_Given = TRUE;
    break;
  case  HSMHV2_MOD_NSUBP:
    mod->HSMHV2_nsubp = value->rValue;
    mod->HSMHV2_nsubp_Given = TRUE;
    break;
  case  HSMHV2_MOD_NSUBP0:
    mod->HSMHV2_nsubp0 = value->rValue;
    mod->HSMHV2_nsubp0_Given = TRUE;
    break;
  case  HSMHV2_MOD_NSUBWP:
    mod->HSMHV2_nsubwp = value->rValue;
    mod->HSMHV2_nsubwp_Given = TRUE;
    break;
  case  HSMHV2_MOD_SCP1:
    mod->HSMHV2_scp1 = value->rValue;
    mod->HSMHV2_scp1_Given = TRUE;
    break;
  case  HSMHV2_MOD_SCP2:
    mod->HSMHV2_scp2 = value->rValue;
    mod->HSMHV2_scp2_Given = TRUE;
    break;
  case  HSMHV2_MOD_SCP3:
    mod->HSMHV2_scp3 = value->rValue;
    mod->HSMHV2_scp3_Given = TRUE;
    break;
  case  HSMHV2_MOD_SC1:
    mod->HSMHV2_sc1 = value->rValue;
    mod->HSMHV2_sc1_Given = TRUE;
    break;
  case  HSMHV2_MOD_SC2:
    mod->HSMHV2_sc2 = value->rValue;
    mod->HSMHV2_sc2_Given = TRUE;
    break;
  case  HSMHV2_MOD_SC3:
    mod->HSMHV2_sc3 = value->rValue;
    mod->HSMHV2_sc3_Given = TRUE;
    break;
  case  HSMHV2_MOD_SC4:
    mod->HSMHV2_sc4 = value->rValue;
    mod->HSMHV2_sc4_Given = TRUE;
    break;
  case  HSMHV2_MOD_PGD1:
    mod->HSMHV2_pgd1 = value->rValue;
    mod->HSMHV2_pgd1_Given = TRUE;
    break;
  case  HSMHV2_MOD_PGD2:
    mod->HSMHV2_pgd2 = value->rValue;
    mod->HSMHV2_pgd2_Given = TRUE;
    break;
  case  HSMHV2_MOD_PGD4:
    mod->HSMHV2_pgd4 = value->rValue;
    mod->HSMHV2_pgd4_Given = TRUE;
    break;
  case  HSMHV2_MOD_NDEP:
    mod->HSMHV2_ndep = value->rValue;
    mod->HSMHV2_ndep_Given = TRUE;
    break;
  case  HSMHV2_MOD_NDEPL:
    mod->HSMHV2_ndepl = value->rValue;
    mod->HSMHV2_ndepl_Given = TRUE;
    break;
  case  HSMHV2_MOD_NDEPLP:
    mod->HSMHV2_ndeplp = value->rValue;
    mod->HSMHV2_ndeplp_Given = TRUE;
    break;
  case  HSMHV2_MOD_NINV:
    mod->HSMHV2_ninv = value->rValue;
    mod->HSMHV2_ninv_Given = TRUE;
    break;
  case  HSMHV2_MOD_MUECB0:
    mod->HSMHV2_muecb0 = value->rValue;
    mod->HSMHV2_muecb0_Given = TRUE;
    break;
  case  HSMHV2_MOD_MUECB1:
    mod->HSMHV2_muecb1 = value->rValue;
    mod->HSMHV2_muecb1_Given = TRUE;
    break;
  case  HSMHV2_MOD_MUEPH1:
    mod->HSMHV2_mueph1 = value->rValue;
    mod->HSMHV2_mueph1_Given = TRUE;
    break;
  case  HSMHV2_MOD_MUEPH0:
    mod->HSMHV2_mueph0 = value->rValue;
    mod->HSMHV2_mueph0_Given = TRUE;
    break;
  case  HSMHV2_MOD_MUEPHW:
    mod->HSMHV2_muephw = value->rValue;
    mod->HSMHV2_muephw_Given = TRUE;
    break;
  case  HSMHV2_MOD_MUEPWP:
    mod->HSMHV2_muepwp = value->rValue;
    mod->HSMHV2_muepwp_Given = TRUE;
    break;
  case  HSMHV2_MOD_MUEPHL:
    mod->HSMHV2_muephl = value->rValue;
    mod->HSMHV2_muephl_Given = TRUE;
    break;
  case  HSMHV2_MOD_MUEPLP:
    mod->HSMHV2_mueplp = value->rValue;
    mod->HSMHV2_mueplp_Given = TRUE;
    break;
  case  HSMHV2_MOD_MUEPHS:
    mod->HSMHV2_muephs = value->rValue;
    mod->HSMHV2_muephs_Given = TRUE;
    break;
   case  HSMHV2_MOD_MUEPSP:
    mod->HSMHV2_muepsp = value->rValue;
    mod->HSMHV2_muepsp_Given = TRUE;
    break;
  case  HSMHV2_MOD_VTMP:
    mod->HSMHV2_vtmp = value->rValue;
    mod->HSMHV2_vtmp_Given = TRUE;
    break;
  case  HSMHV2_MOD_WVTH0:
    mod->HSMHV2_wvth0 = value->rValue;
    mod->HSMHV2_wvth0_Given = TRUE;
    break;
  case  HSMHV2_MOD_MUESR1:
    mod->HSMHV2_muesr1 = value->rValue;
    mod->HSMHV2_muesr1_Given = TRUE;
    break;
  case  HSMHV2_MOD_MUESR0:
    mod->HSMHV2_muesr0 = value->rValue;
    mod->HSMHV2_muesr0_Given = TRUE;
    break;
  case  HSMHV2_MOD_MUESRL:
    mod->HSMHV2_muesrl = value->rValue;
    mod->HSMHV2_muesrl_Given = TRUE;
    break;
  case  HSMHV2_MOD_MUESLP:
    mod->HSMHV2_mueslp = value->rValue;
    mod->HSMHV2_mueslp_Given = TRUE;
    break;
  case  HSMHV2_MOD_MUESRW:
    mod->HSMHV2_muesrw = value->rValue;
    mod->HSMHV2_muesrw_Given = TRUE;
    break;
  case  HSMHV2_MOD_MUESWP:
    mod->HSMHV2_mueswp = value->rValue;
    mod->HSMHV2_mueswp_Given = TRUE;
    break;
  case  HSMHV2_MOD_BB:
    mod->HSMHV2_bb = value->rValue;
    mod->HSMHV2_bb_Given = TRUE;
    break;
  case  HSMHV2_MOD_SUB1:
    mod->HSMHV2_sub1 = value->rValue;
    mod->HSMHV2_sub1_Given = TRUE;
    break;
  case  HSMHV2_MOD_SUB2:
    mod->HSMHV2_sub2 = value->rValue;
    mod->HSMHV2_sub2_Given = TRUE;
    break;
  case  HSMHV2_MOD_SVGS:
    mod->HSMHV2_svgs = value->rValue;
    mod->HSMHV2_svgs_Given = TRUE;
    break;
  case  HSMHV2_MOD_SVBS:
    mod->HSMHV2_svbs = value->rValue;
    mod->HSMHV2_svbs_Given = TRUE;
    break;
  case  HSMHV2_MOD_SVBSL:
    mod->HSMHV2_svbsl = value->rValue;
    mod->HSMHV2_svbsl_Given = TRUE;
    break;
  case  HSMHV2_MOD_SVDS:
    mod->HSMHV2_svds = value->rValue;
    mod->HSMHV2_svds_Given = TRUE;
    break;
  case  HSMHV2_MOD_SLG:
    mod->HSMHV2_slg = value->rValue;
    mod->HSMHV2_slg_Given = TRUE;
    break;
  case  HSMHV2_MOD_SUB1L:
    mod->HSMHV2_sub1l = value->rValue;
    mod->HSMHV2_sub1l_Given = TRUE;
    break;
  case  HSMHV2_MOD_SUB2L:
    mod->HSMHV2_sub2l = value->rValue;
    mod->HSMHV2_sub2l_Given = TRUE;
    break;
  case  HSMHV2_MOD_FN1:
    mod->HSMHV2_fn1 = value->rValue;
    mod->HSMHV2_fn1_Given = TRUE;
    break;
  case  HSMHV2_MOD_FN2:
    mod->HSMHV2_fn2 = value->rValue;
    mod->HSMHV2_fn2_Given = TRUE;
    break;
  case  HSMHV2_MOD_FN3:
    mod->HSMHV2_fn3 = value->rValue;
    mod->HSMHV2_fn3_Given = TRUE;
    break;
  case  HSMHV2_MOD_FVBS:
    mod->HSMHV2_fvbs = value->rValue;
    mod->HSMHV2_fvbs_Given = TRUE;
    break;
  case  HSMHV2_MOD_SVGSL:
    mod->HSMHV2_svgsl = value->rValue;
    mod->HSMHV2_svgsl_Given = TRUE;
    break;
  case  HSMHV2_MOD_SVGSLP:
    mod->HSMHV2_svgslp = value->rValue;
    mod->HSMHV2_svgslp_Given = TRUE;
    break;
  case  HSMHV2_MOD_SVGSWP:
    mod->HSMHV2_svgswp = value->rValue;
    mod->HSMHV2_svgswp_Given = TRUE;
    break;
  case  HSMHV2_MOD_SVGSW:
    mod->HSMHV2_svgsw = value->rValue;
    mod->HSMHV2_svgsw_Given = TRUE;
    break;
  case  HSMHV2_MOD_SVBSLP:
    mod->HSMHV2_svbslp = value->rValue;
    mod->HSMHV2_svbslp_Given = TRUE;
    break;
  case  HSMHV2_MOD_SLGL:
    mod->HSMHV2_slgl = value->rValue;
    mod->HSMHV2_slgl_Given = TRUE;
    break;
  case  HSMHV2_MOD_SLGLP:
    mod->HSMHV2_slglp = value->rValue;
    mod->HSMHV2_slglp_Given = TRUE;
    break;
  case  HSMHV2_MOD_SUB1LP:
    mod->HSMHV2_sub1lp = value->rValue;
    mod->HSMHV2_sub1lp_Given = TRUE;
    break;
  case  HSMHV2_MOD_NSTI:
    mod->HSMHV2_nsti = value->rValue;
    mod->HSMHV2_nsti_Given = TRUE;
    break;
  case  HSMHV2_MOD_WSTI:
    mod->HSMHV2_wsti = value->rValue;
    mod->HSMHV2_wsti_Given = TRUE;
    break;
  case  HSMHV2_MOD_WSTIL:
    mod->HSMHV2_wstil = value->rValue;
    mod->HSMHV2_wstil_Given = TRUE;
    break;
  case  HSMHV2_MOD_WSTILP:
    mod->HSMHV2_wstilp = value->rValue;
    mod->HSMHV2_wstilp_Given = TRUE;
    break;
  case  HSMHV2_MOD_WSTIW:
    mod->HSMHV2_wstiw = value->rValue;
    mod->HSMHV2_wstiw_Given = TRUE;
    break;
  case  HSMHV2_MOD_WSTIWP:
    mod->HSMHV2_wstiwp = value->rValue;
    mod->HSMHV2_wstiwp_Given = TRUE;
    break;
  case  HSMHV2_MOD_SCSTI1:
    mod->HSMHV2_scsti1 = value->rValue;
    mod->HSMHV2_scsti1_Given = TRUE;
    break;
  case  HSMHV2_MOD_SCSTI2:
    mod->HSMHV2_scsti2 = value->rValue;
    mod->HSMHV2_scsti2_Given = TRUE;
    break;
  case  HSMHV2_MOD_VTHSTI:
    mod->HSMHV2_vthsti = value->rValue;
    mod->HSMHV2_vthsti_Given = TRUE;
    break;
  case  HSMHV2_MOD_VDSTI:
    mod->HSMHV2_vdsti = value->rValue;
    mod->HSMHV2_vdsti_Given = TRUE;
    break;
  case  HSMHV2_MOD_MUESTI1:
    mod->HSMHV2_muesti1 = value->rValue;
    mod->HSMHV2_muesti1_Given = TRUE;
    break;
  case  HSMHV2_MOD_MUESTI2:
    mod->HSMHV2_muesti2 = value->rValue;
    mod->HSMHV2_muesti2_Given = TRUE;
    break;
  case  HSMHV2_MOD_MUESTI3:
    mod->HSMHV2_muesti3 = value->rValue;
    mod->HSMHV2_muesti3_Given = TRUE;
    break;
  case  HSMHV2_MOD_NSUBPSTI1:
    mod->HSMHV2_nsubpsti1 = value->rValue;
    mod->HSMHV2_nsubpsti1_Given = TRUE;
    break;
  case  HSMHV2_MOD_NSUBPSTI2:
    mod->HSMHV2_nsubpsti2 = value->rValue;
    mod->HSMHV2_nsubpsti2_Given = TRUE;
    break;
  case  HSMHV2_MOD_NSUBPSTI3:
    mod->HSMHV2_nsubpsti3 = value->rValue;
    mod->HSMHV2_nsubpsti3_Given = TRUE;
    break;
  case  HSMHV2_MOD_LPEXT:
    mod->HSMHV2_lpext = value->rValue;
    mod->HSMHV2_lpext_Given = TRUE;
    break;
  case  HSMHV2_MOD_NPEXT:
    mod->HSMHV2_npext = value->rValue;
    mod->HSMHV2_npext_Given = TRUE;
    break;
  case  HSMHV2_MOD_SCP22:
    mod->HSMHV2_scp22 = value->rValue;
    mod->HSMHV2_scp22_Given = TRUE;
    break;
  case  HSMHV2_MOD_SCP21:
    mod->HSMHV2_scp21 = value->rValue;
    mod->HSMHV2_scp21_Given = TRUE;
    break;
  case  HSMHV2_MOD_BS1:
    mod->HSMHV2_bs1 = value->rValue;
    mod->HSMHV2_bs1_Given = TRUE;
    break;
  case  HSMHV2_MOD_BS2:
    mod->HSMHV2_bs2 = value->rValue;
    mod->HSMHV2_bs2_Given = TRUE;
    break;
  case  HSMHV2_MOD_CGSO:
    mod->HSMHV2_cgso = value->rValue;
    mod->HSMHV2_cgso_Given = TRUE;
    break;
  case  HSMHV2_MOD_CGDO:
    mod->HSMHV2_cgdo = value->rValue;
    mod->HSMHV2_cgdo_Given = TRUE;
    break;
  case  HSMHV2_MOD_CGBO:
    mod->HSMHV2_cgbo = value->rValue;
    mod->HSMHV2_cgbo_Given = TRUE;
    break;
  case  HSMHV2_MOD_TPOLY:
    mod->HSMHV2_tpoly = value->rValue;
    mod->HSMHV2_tpoly_Given = TRUE;
    break;
  case  HSMHV2_MOD_JS0:
    mod->HSMHV2_js0 = value->rValue;
    mod->HSMHV2_js0_Given = TRUE;
    break;
  case  HSMHV2_MOD_JS0SW:
    mod->HSMHV2_js0sw = value->rValue;
    mod->HSMHV2_js0sw_Given = TRUE;
    break;
  case  HSMHV2_MOD_NJ:
    mod->HSMHV2_nj = value->rValue;
    mod->HSMHV2_nj_Given = TRUE;
    break;
  case  HSMHV2_MOD_NJSW:
    mod->HSMHV2_njsw = value->rValue;
    mod->HSMHV2_njsw_Given = TRUE;
    break;
  case  HSMHV2_MOD_XTI:
    mod->HSMHV2_xti = value->rValue;
    mod->HSMHV2_xti_Given = TRUE;
    break;
  case  HSMHV2_MOD_CJ:
    mod->HSMHV2_cj = value->rValue;
    mod->HSMHV2_cj_Given = TRUE;
    break;
  case  HSMHV2_MOD_CJSW:
    mod->HSMHV2_cjsw = value->rValue;
    mod->HSMHV2_cjsw_Given = TRUE;
    break;
  case  HSMHV2_MOD_CJSWG:
    mod->HSMHV2_cjswg = value->rValue;
    mod->HSMHV2_cjswg_Given = TRUE;
    break;
  case  HSMHV2_MOD_MJ:
    mod->HSMHV2_mj = value->rValue;
    mod->HSMHV2_mj_Given = TRUE;
    break;
  case  HSMHV2_MOD_MJSW:
    mod->HSMHV2_mjsw = value->rValue;
    mod->HSMHV2_mjsw_Given = TRUE;
    break;
  case  HSMHV2_MOD_MJSWG:
    mod->HSMHV2_mjswg = value->rValue;
    mod->HSMHV2_mjswg_Given = TRUE;
    break;
  case  HSMHV2_MOD_PB:
    mod->HSMHV2_pb = value->rValue;
    mod->HSMHV2_pb_Given = TRUE;
    break;
  case  HSMHV2_MOD_PBSW:
    mod->HSMHV2_pbsw = value->rValue;
    mod->HSMHV2_pbsw_Given = TRUE;
    break;
  case  HSMHV2_MOD_PBSWG:
    mod->HSMHV2_pbswg = value->rValue;
    mod->HSMHV2_pbswg_Given = TRUE;
    break;
  case  HSMHV2_MOD_XTI2:
    mod->HSMHV2_xti2 = value->rValue;
    mod->HSMHV2_xti2_Given = TRUE;
    break;
  case  HSMHV2_MOD_CISB:
    mod->HSMHV2_cisb = value->rValue;
    mod->HSMHV2_cisb_Given = TRUE;
    break;
  case  HSMHV2_MOD_CVB:
    mod->HSMHV2_cvb = value->rValue;
    mod->HSMHV2_cvb_Given = TRUE;
    break;
  case  HSMHV2_MOD_CTEMP:
    mod->HSMHV2_ctemp = value->rValue;
    mod->HSMHV2_ctemp_Given = TRUE;
    break;
  case  HSMHV2_MOD_CISBK:
    mod->HSMHV2_cisbk = value->rValue;
    mod->HSMHV2_cisbk_Given = TRUE;
    break;
  case  HSMHV2_MOD_CVBK:
    mod->HSMHV2_cvbk = value->rValue;
    mod->HSMHV2_cvbk_Given = TRUE;
    break;
  case  HSMHV2_MOD_DIVX:
    mod->HSMHV2_divx = value->rValue;
    mod->HSMHV2_divx_Given = TRUE;
    break;
  case  HSMHV2_MOD_CLM1:
    mod->HSMHV2_clm1 = value->rValue;
    mod->HSMHV2_clm1_Given = TRUE;
    break;
  case  HSMHV2_MOD_CLM2:
    mod->HSMHV2_clm2 = value->rValue;
    mod->HSMHV2_clm2_Given = TRUE;
    break;
  case  HSMHV2_MOD_CLM3:
    mod->HSMHV2_clm3 = value->rValue;
    mod->HSMHV2_clm3_Given = TRUE;
    break;
  case  HSMHV2_MOD_CLM5:
    mod->HSMHV2_clm5 = value->rValue;
    mod->HSMHV2_clm5_Given = TRUE;
    break;
  case  HSMHV2_MOD_CLM6:
    mod->HSMHV2_clm6 = value->rValue;
    mod->HSMHV2_clm6_Given = TRUE;
    break;
  case  HSMHV2_MOD_MUETMP:
    mod->HSMHV2_muetmp = value->rValue;
    mod->HSMHV2_muetmp_Given = TRUE;
    break;
  case  HSMHV2_MOD_VOVER:
    mod->HSMHV2_vover = value->rValue;
    mod->HSMHV2_vover_Given = TRUE;
    break;
  case  HSMHV2_MOD_VOVERP:
    mod->HSMHV2_voverp = value->rValue;
    mod->HSMHV2_voverp_Given = TRUE;
    break;
  case  HSMHV2_MOD_VOVERS:
    mod->HSMHV2_vovers = value->rValue;
    mod->HSMHV2_vovers_Given = TRUE;
    break;
  case  HSMHV2_MOD_VOVERSP:
    mod->HSMHV2_voversp = value->rValue;
    mod->HSMHV2_voversp_Given = TRUE;
    break;
  case  HSMHV2_MOD_WFC:
    mod->HSMHV2_wfc = value->rValue;
    mod->HSMHV2_wfc_Given = TRUE;
    break;
  case  HSMHV2_MOD_NSUBCW:
    mod->HSMHV2_nsubcw = value->rValue;
    mod->HSMHV2_nsubcw_Given = TRUE;
    break;
  case  HSMHV2_MOD_NSUBCWP:
    mod->HSMHV2_nsubcwp = value->rValue;
    mod->HSMHV2_nsubcwp_Given = TRUE;
    break;
  case  HSMHV2_MOD_QME1:
    mod->HSMHV2_qme1 = value->rValue;
    mod->HSMHV2_qme1_Given = TRUE;
    break;
  case  HSMHV2_MOD_QME2:
    mod->HSMHV2_qme2 = value->rValue;
    mod->HSMHV2_qme2_Given = TRUE;
    break;
  case  HSMHV2_MOD_QME3:
    mod->HSMHV2_qme3 = value->rValue;
    mod->HSMHV2_qme3_Given = TRUE;
    break;
  case  HSMHV2_MOD_GIDL1:
    mod->HSMHV2_gidl1 = value->rValue;
    mod->HSMHV2_gidl1_Given = TRUE;
    break;
  case  HSMHV2_MOD_GIDL2:
    mod->HSMHV2_gidl2 = value->rValue;
    mod->HSMHV2_gidl2_Given = TRUE;
    break;
  case  HSMHV2_MOD_GIDL3:
    mod->HSMHV2_gidl3 = value->rValue;
    mod->HSMHV2_gidl3_Given = TRUE;
    break;
  case  HSMHV2_MOD_GIDL4:
    mod->HSMHV2_gidl4 = value->rValue;
    mod->HSMHV2_gidl4_Given = TRUE;
    break;
  case  HSMHV2_MOD_GIDL5:
    mod->HSMHV2_gidl5 = value->rValue;
    mod->HSMHV2_gidl5_Given = TRUE;
    break;
  case  HSMHV2_MOD_GLEAK1:
    mod->HSMHV2_gleak1 = value->rValue;
    mod->HSMHV2_gleak1_Given = TRUE;
    break;
  case  HSMHV2_MOD_GLEAK2:
    mod->HSMHV2_gleak2 = value->rValue;
    mod->HSMHV2_gleak2_Given = TRUE;
    break;
  case  HSMHV2_MOD_GLEAK3:
    mod->HSMHV2_gleak3 = value->rValue;
    mod->HSMHV2_gleak3_Given = TRUE;
    break;
  case  HSMHV2_MOD_GLEAK4:
    mod->HSMHV2_gleak4 = value->rValue;
    mod->HSMHV2_gleak4_Given = TRUE;
    break;
  case  HSMHV2_MOD_GLEAK5:
    mod->HSMHV2_gleak5 = value->rValue;
    mod->HSMHV2_gleak5_Given = TRUE;
    break;
  case  HSMHV2_MOD_GLEAK6:
    mod->HSMHV2_gleak6 = value->rValue;
    mod->HSMHV2_gleak6_Given = TRUE;
    break;
  case  HSMHV2_MOD_GLEAK7:
    mod->HSMHV2_gleak7 = value->rValue;
    mod->HSMHV2_gleak7_Given = TRUE;
    break;
  case  HSMHV2_MOD_GLPART1:
    mod->HSMHV2_glpart1 = value->rValue;
    mod->HSMHV2_glpart1_Given = TRUE;
    break;
  case  HSMHV2_MOD_GLKSD1:
    mod->HSMHV2_glksd1 = value->rValue;
    mod->HSMHV2_glksd1_Given = TRUE;
    break;
  case  HSMHV2_MOD_GLKSD2:
    mod->HSMHV2_glksd2 = value->rValue;
    mod->HSMHV2_glksd2_Given = TRUE;
    break;
  case  HSMHV2_MOD_GLKSD3:
    mod->HSMHV2_glksd3 = value->rValue;
    mod->HSMHV2_glksd3_Given = TRUE;
    break;
  case  HSMHV2_MOD_GLKB1:
    mod->HSMHV2_glkb1 = value->rValue;
    mod->HSMHV2_glkb1_Given = TRUE;
    break;
  case  HSMHV2_MOD_GLKB2:
    mod->HSMHV2_glkb2 = value->rValue;
    mod->HSMHV2_glkb2_Given = TRUE;
    break;
  case  HSMHV2_MOD_GLKB3:
    mod->HSMHV2_glkb3 = value->rValue;
    mod->HSMHV2_glkb3_Given = TRUE;
    break;
  case  HSMHV2_MOD_EGIG:
    mod->HSMHV2_egig = value->rValue;
    mod->HSMHV2_egig_Given = TRUE;
    break;
  case  HSMHV2_MOD_IGTEMP2:
    mod->HSMHV2_igtemp2 = value->rValue;
    mod->HSMHV2_igtemp2_Given = TRUE;
    break;
  case  HSMHV2_MOD_IGTEMP3:
    mod->HSMHV2_igtemp3 = value->rValue;
    mod->HSMHV2_igtemp3_Given = TRUE;
    break;
  case  HSMHV2_MOD_VZADD0:
    mod->HSMHV2_vzadd0 = value->rValue;
    mod->HSMHV2_vzadd0_Given = TRUE;
    break;
  case  HSMHV2_MOD_PZADD0:
    mod->HSMHV2_pzadd0 = value->rValue;
    mod->HSMHV2_pzadd0_Given = TRUE;
    break;
  case  HSMHV2_MOD_NFTRP:
    mod->HSMHV2_nftrp = value->rValue;
    mod->HSMHV2_nftrp_Given = TRUE;
    break;
  case  HSMHV2_MOD_NFALP:
    mod->HSMHV2_nfalp = value->rValue;
    mod->HSMHV2_nfalp_Given = TRUE;
    break;
  case  HSMHV2_MOD_CIT:
    mod->HSMHV2_cit = value->rValue;
    mod->HSMHV2_cit_Given = TRUE;
    break;
  case  HSMHV2_MOD_FALPH:
    mod->HSMHV2_falph = value->rValue;
    mod->HSMHV2_falph_Given = TRUE;
    break;
  case  HSMHV2_MOD_KAPPA:
    mod->HSMHV2_kappa = value->rValue;
    mod->HSMHV2_kappa_Given = TRUE;
    break;
  case  HSMHV2_MOD_VDIFFJ:
    mod->HSMHV2_vdiffj = value->rValue;
    mod->HSMHV2_vdiffj_Given = TRUE;
    break;
  case  HSMHV2_MOD_DLY1:
    mod->HSMHV2_dly1 = value->rValue;
    mod->HSMHV2_dly1_Given = TRUE;
    break;
  case  HSMHV2_MOD_DLY2:
    mod->HSMHV2_dly2 = value->rValue;
    mod->HSMHV2_dly2_Given = TRUE;
    break;
  case  HSMHV2_MOD_DLY3:
    mod->HSMHV2_dly3 = value->rValue;
    mod->HSMHV2_dly3_Given = TRUE;
    break;
  case  HSMHV2_MOD_TNOM:
    mod->HSMHV2_tnom = value->rValue;
    mod->HSMHV2_tnom_Given = TRUE;
    break;
  case  HSMHV2_MOD_OVSLP:
    mod->HSMHV2_ovslp = value->rValue;
    mod->HSMHV2_ovslp_Given = TRUE;
    break;
  case  HSMHV2_MOD_OVMAG:
    mod->HSMHV2_ovmag = value->rValue;
    mod->HSMHV2_ovmag_Given = TRUE;
    break;
  case  HSMHV2_MOD_GBMIN:
    mod->HSMHV2_gbmin = value->rValue;
    mod->HSMHV2_gbmin_Given = TRUE;
    break;
  case  HSMHV2_MOD_RBPB:
    mod->HSMHV2_rbpb = value->rValue;
    mod->HSMHV2_rbpb_Given = TRUE;
    break;
  case  HSMHV2_MOD_RBPD:
    mod->HSMHV2_rbpd = value->rValue;
    mod->HSMHV2_rbpd_Given = TRUE;
    break;
  case  HSMHV2_MOD_RBPS:
    mod->HSMHV2_rbps = value->rValue;
    mod->HSMHV2_rbps_Given = TRUE;
    break;
  case  HSMHV2_MOD_RBDB:
    mod->HSMHV2_rbdb = value->rValue;
    mod->HSMHV2_rbdb_Given = TRUE;
    break;
  case  HSMHV2_MOD_RBSB:
    mod->HSMHV2_rbsb = value->rValue;
    mod->HSMHV2_rbsb_Given = TRUE;
    break;
  case  HSMHV2_MOD_IBPC1:
    mod->HSMHV2_ibpc1 = value->rValue;
    mod->HSMHV2_ibpc1_Given = TRUE;
    break;
  case  HSMHV2_MOD_IBPC1L:
    mod->HSMHV2_ibpc1l = value->rValue;
    mod->HSMHV2_ibpc1l_Given = TRUE;
    break;
  case  HSMHV2_MOD_IBPC1LP:
    mod->HSMHV2_ibpc1lp = value->rValue;
    mod->HSMHV2_ibpc1lp_Given = TRUE;
    break;
  case  HSMHV2_MOD_IBPC2:
    mod->HSMHV2_ibpc2 = value->rValue;
    mod->HSMHV2_ibpc2_Given = TRUE;
    break;
  case  HSMHV2_MOD_MPHDFM:
    mod->HSMHV2_mphdfm = value->rValue;
    mod->HSMHV2_mphdfm_Given = TRUE;
    break;

  case  HSMHV2_MOD_PTL:
    mod->HSMHV2_ptl = value->rValue;
    mod->HSMHV2_ptl_Given = TRUE;
    break;
  case  HSMHV2_MOD_PTP:
    mod->HSMHV2_ptp = value->rValue;
    mod->HSMHV2_ptp_Given = TRUE;
    break;
  case  HSMHV2_MOD_PT2:
    mod->HSMHV2_pt2 = value->rValue;
    mod->HSMHV2_pt2_Given = TRUE;
    break;
  case  HSMHV2_MOD_PTLP:
    mod->HSMHV2_ptlp = value->rValue;
    mod->HSMHV2_ptlp_Given = TRUE;
    break;
  case  HSMHV2_MOD_GDL:
    mod->HSMHV2_gdl = value->rValue;
    mod->HSMHV2_gdl_Given = TRUE;
    break;
  case  HSMHV2_MOD_GDLP:
    mod->HSMHV2_gdlp = value->rValue;
    mod->HSMHV2_gdlp_Given = TRUE;
    break;

  case  HSMHV2_MOD_GDLD:
    mod->HSMHV2_gdld = value->rValue;
    mod->HSMHV2_gdld_Given = TRUE;
    break;
  case  HSMHV2_MOD_PT4:
    mod->HSMHV2_pt4 = value->rValue;
    mod->HSMHV2_pt4_Given = TRUE;
    break;
  case  HSMHV2_MOD_PT4P:
    mod->HSMHV2_pt4p = value->rValue;
    mod->HSMHV2_pt4p_Given = TRUE;
    break;
  case  HSMHV2_MOD_RDVG11:
    mod->HSMHV2_rdvg11 = value->rValue;
    mod->HSMHV2_rdvg11_Given = TRUE;
    break;
  case  HSMHV2_MOD_RDVG12:
    mod->HSMHV2_rdvg12 = value->rValue;
    mod->HSMHV2_rdvg12_Given = TRUE;
    break;
  case  HSMHV2_MOD_RD20:
    mod->HSMHV2_rd20 = value->rValue;
    mod->HSMHV2_rd20_Given = TRUE;
    break;
  case  HSMHV2_MOD_RD21:
    mod->HSMHV2_rd21 = value->rValue;
    mod->HSMHV2_rd21_Given = TRUE;
    break;
  case  HSMHV2_MOD_RD22:
    mod->HSMHV2_rd22 = value->rValue;
    mod->HSMHV2_rd22_Given = TRUE;
    break;
  case  HSMHV2_MOD_RD22D:
    mod->HSMHV2_rd22d = value->rValue;
    mod->HSMHV2_rd22d_Given = TRUE;
    break;
  case  HSMHV2_MOD_RD23:
    mod->HSMHV2_rd23 = value->rValue;
    mod->HSMHV2_rd23_Given = TRUE;
    break;
  case  HSMHV2_MOD_RD24:
    mod->HSMHV2_rd24 = value->rValue;
    mod->HSMHV2_rd24_Given = TRUE;
    break;
  case  HSMHV2_MOD_RD25:
    mod->HSMHV2_rd25 = value->rValue;
    mod->HSMHV2_rd25_Given = TRUE;
    break;
  case  HSMHV2_MOD_RDVDL:
    mod->HSMHV2_rdvdl = value->rValue;
    mod->HSMHV2_rdvdl_Given = TRUE;
    break;
  case  HSMHV2_MOD_RDVDLP:
    mod->HSMHV2_rdvdlp = value->rValue;
    mod->HSMHV2_rdvdlp_Given = TRUE;
    break;
  case  HSMHV2_MOD_RDVDS:
    mod->HSMHV2_rdvds = value->rValue;
    mod->HSMHV2_rdvds_Given = TRUE;
    break;
  case  HSMHV2_MOD_RDVDSP:
    mod->HSMHV2_rdvdsp = value->rValue;
    mod->HSMHV2_rdvdsp_Given = TRUE;
    break;
  case  HSMHV2_MOD_RD23L:
    mod->HSMHV2_rd23l = value->rValue;
    mod->HSMHV2_rd23l_Given = TRUE;
    break;
  case  HSMHV2_MOD_RD23LP:
    mod->HSMHV2_rd23lp = value->rValue;
    mod->HSMHV2_rd23lp_Given = TRUE;
    break;
  case  HSMHV2_MOD_RD23S:
    mod->HSMHV2_rd23s = value->rValue;
    mod->HSMHV2_rd23s_Given = TRUE;
    break;
  case  HSMHV2_MOD_RD23SP:
    mod->HSMHV2_rd23sp = value->rValue;
    mod->HSMHV2_rd23sp_Given = TRUE;
    break;
  case  HSMHV2_MOD_RDS:
    mod->HSMHV2_rds = value->rValue;
    mod->HSMHV2_rds_Given = TRUE;
    break;
  case  HSMHV2_MOD_RDSP:
    mod->HSMHV2_rdsp = value->rValue;
    mod->HSMHV2_rdsp_Given = TRUE;
    break;
  case  HSMHV2_MOD_RTH0: /* Self-heating model */
    mod->HSMHV2_rth0 = value->rValue;
    mod->HSMHV2_rth0_Given = TRUE;
    break;
  case  HSMHV2_MOD_CTH0: /* Self-heating model */
    mod->HSMHV2_cth0 = value->rValue;
    mod->HSMHV2_cth0_Given = TRUE;
    break;
  case  HSMHV2_MOD_POWRAT: /* Self-heating model */
    mod->HSMHV2_powrat = value->rValue;
    mod->HSMHV2_powrat_Given = TRUE;
    break;
  case  HSMHV2_MOD_TCJBD: /* Self-heating model */
    mod->HSMHV2_tcjbd = value->rValue;
    mod->HSMHV2_tcjbd_Given = TRUE;
    break;
  case  HSMHV2_MOD_TCJBS: /* Self-heating model */
    mod->HSMHV2_tcjbs = value->rValue;
    mod->HSMHV2_tcjbs_Given = TRUE;
    break;
  case  HSMHV2_MOD_TCJBDSW: /* Self-heating model */
    mod->HSMHV2_tcjbdsw = value->rValue;
    mod->HSMHV2_tcjbdsw_Given = TRUE;
    break;
  case  HSMHV2_MOD_TCJBSSW: /* Self-heating model */
    mod->HSMHV2_tcjbssw = value->rValue;
    mod->HSMHV2_tcjbssw_Given = TRUE;
    break;
  case  HSMHV2_MOD_TCJBDSWG: /* Self-heating model */
    mod->HSMHV2_tcjbdswg = value->rValue;
    mod->HSMHV2_tcjbdswg_Given = TRUE;
    break;
  case  HSMHV2_MOD_TCJBSSWG: /* Self-heating model */
    mod->HSMHV2_tcjbsswg = value->rValue;
    mod->HSMHV2_tcjbsswg_Given = TRUE;
    break;

  case  HSMHV2_MOD_DLYOV:
    mod->HSMHV2_dlyov = value->rValue;
    mod->HSMHV2_dlyov_Given = TRUE;
    break;
  case  HSMHV2_MOD_QDFTVD:
    mod->HSMHV2_qdftvd = value->rValue;
    mod->HSMHV2_qdftvd_Given = TRUE;
    break;
  case  HSMHV2_MOD_XLDLD:
    mod->HSMHV2_xldld = value->rValue;
    mod->HSMHV2_xldld_Given = TRUE;
    break;
  case  HSMHV2_MOD_XWDLD:
    mod->HSMHV2_xwdld = value->rValue;
    mod->HSMHV2_xwdld_Given = TRUE;
    break;
  case  HSMHV2_MOD_RDVD:
    mod->HSMHV2_rdvd = value->rValue;
    mod->HSMHV2_rdvd_Given = TRUE;
    break;

  case  HSMHV2_MOD_RDTEMP1:
    mod->HSMHV2_rdtemp1 = value->rValue;
    mod->HSMHV2_rdtemp1_Given = TRUE;
    break;
  case  HSMHV2_MOD_RDTEMP2:
    mod->HSMHV2_rdtemp2 = value->rValue;
    mod->HSMHV2_rdtemp2_Given = TRUE;
    break;
  case  HSMHV2_MOD_RTH0R:
    mod->HSMHV2_rth0r = value->rValue;
    mod->HSMHV2_rth0r_Given = TRUE;
    break;
  case  HSMHV2_MOD_RDVDTEMP1:
    mod->HSMHV2_rdvdtemp1 = value->rValue;
    mod->HSMHV2_rdvdtemp1_Given = TRUE;
    break;
  case  HSMHV2_MOD_RDVDTEMP2:
    mod->HSMHV2_rdvdtemp2 = value->rValue;
    mod->HSMHV2_rdvdtemp2_Given = TRUE;
    break;
  case  HSMHV2_MOD_RTH0W:
    mod->HSMHV2_rth0w = value->rValue;
    mod->HSMHV2_rth0w_Given = TRUE;
    break;
  case  HSMHV2_MOD_RTH0WP:
    mod->HSMHV2_rth0wp = value->rValue;
    mod->HSMHV2_rth0wp_Given = TRUE;
    break;
  case  HSMHV2_MOD_CVDSOVER:
    mod->HSMHV2_cvdsover = value->rValue;
    mod->HSMHV2_cvdsover_Given = TRUE;
    break;

  case  HSMHV2_MOD_NINVD:
    mod->HSMHV2_ninvd = value->rValue;
    mod->HSMHV2_ninvd_Given = TRUE;
    break;
  case  HSMHV2_MOD_NINVDW:
    mod->HSMHV2_ninvdw = value->rValue;
    mod->HSMHV2_ninvdw_Given = TRUE;
    break;
  case  HSMHV2_MOD_NINVDWP:
    mod->HSMHV2_ninvdwp = value->rValue;
    mod->HSMHV2_ninvdwp_Given = TRUE;
    break;
  case  HSMHV2_MOD_NINVDT1:
    mod->HSMHV2_ninvdt1 = value->rValue;
    mod->HSMHV2_ninvdt1_Given = TRUE;
    break;
  case  HSMHV2_MOD_NINVDT2:
    mod->HSMHV2_ninvdt2 = value->rValue;
    mod->HSMHV2_ninvdt2_Given = TRUE;
    break;
  case  HSMHV2_MOD_VBSMIN:
    mod->HSMHV2_vbsmin = value->rValue;
    mod->HSMHV2_vbsmin_Given = TRUE;
    break;
  case  HSMHV2_MOD_RDVB:
    mod->HSMHV2_rdvb = value->rValue;
    mod->HSMHV2_rdvb_Given = TRUE;
    break;
  case  HSMHV2_MOD_RTH0NF:
    mod->HSMHV2_rth0nf = value->rValue;
    mod->HSMHV2_rth0nf_Given = TRUE;
    break;
  case  HSMHV2_MOD_RTHTEMP1:
    mod->HSMHV2_rthtemp1 = value->rValue;
    mod->HSMHV2_rthtemp1_Given = TRUE;
    break;
  case  HSMHV2_MOD_RTHTEMP2:
    mod->HSMHV2_rthtemp2 = value->rValue;
    mod->HSMHV2_rthtemp2_Given = TRUE;
    break;
  case  HSMHV2_MOD_PRATTEMP1:
    mod->HSMHV2_prattemp1 = value->rValue;
    mod->HSMHV2_prattemp1_Given = TRUE;
    break;
  case  HSMHV2_MOD_PRATTEMP2:
    mod->HSMHV2_prattemp2 = value->rValue;
    mod->HSMHV2_prattemp2_Given = TRUE;
    break;

  case  HSMHV2_MOD_RDVSUB: /* substrate effect */ 
    mod->HSMHV2_rdvsub = value->rValue;
    mod->HSMHV2_rdvsub_Given = TRUE;
    break;
  case  HSMHV2_MOD_RDVDSUB:
    mod->HSMHV2_rdvdsub = value->rValue;
    mod->HSMHV2_rdvdsub_Given = TRUE;
    break;
  case  HSMHV2_MOD_DDRIFT:
    mod->HSMHV2_ddrift = value->rValue;
    mod->HSMHV2_ddrift_Given = TRUE;
    break;
  case  HSMHV2_MOD_VBISUB:
    mod->HSMHV2_vbisub = value->rValue;
    mod->HSMHV2_vbisub_Given = TRUE;
    break;
  case  HSMHV2_MOD_NSUBSUB:
    mod->HSMHV2_nsubsub = value->rValue;
    mod->HSMHV2_nsubsub_Given = TRUE;
    break;

  case  HSMHV2_MOD_RDRMUE:
    mod->HSMHV2_rdrmue = value->rValue;
    mod->HSMHV2_rdrmue_Given = TRUE;
    break;
  case  HSMHV2_MOD_RDRVMAX:
    mod->HSMHV2_rdrvmax = value->rValue;
    mod->HSMHV2_rdrvmax_Given = TRUE;
    break;
  case  HSMHV2_MOD_RDRMUETMP:
    mod->HSMHV2_rdrmuetmp = value->rValue;
    mod->HSMHV2_rdrmuetmp_Given = TRUE;
    break;
  case  HSMHV2_MOD_RDRVTMP:
    mod->HSMHV2_rdrvtmp = value->rValue;
    mod->HSMHV2_rdrvtmp_Given = TRUE;
    break;
  case  HSMHV2_MOD_RDRDJUNC:
    mod->HSMHV2_rdrdjunc = value->rValue;
    mod->HSMHV2_rdrdjunc_Given = TRUE;
    break;
  case  HSMHV2_MOD_RDRCX:
    mod->HSMHV2_rdrcx = value->rValue;
    mod->HSMHV2_rdrcx_Given = TRUE;
    break;
  case  HSMHV2_MOD_RDRCAR:
    mod->HSMHV2_rdrcar = value->rValue;
    mod->HSMHV2_rdrcar_Given = TRUE;
    break;
  case  HSMHV2_MOD_RDRDL1:
    mod->HSMHV2_rdrdl1 = value->rValue;
    mod->HSMHV2_rdrdl1_Given = TRUE;
    break;
  case  HSMHV2_MOD_RDRDL2:
    mod->HSMHV2_rdrdl2 = value->rValue;
    mod->HSMHV2_rdrdl2_Given = TRUE;
    break;
  case  HSMHV2_MOD_RDRVMAXW:
    mod->HSMHV2_rdrvmaxw = value->rValue;
    mod->HSMHV2_rdrvmaxw_Given = TRUE;
    break;
  case  HSMHV2_MOD_RDRVMAXWP:
    mod->HSMHV2_rdrvmaxwp = value->rValue;
    mod->HSMHV2_rdrvmaxwp_Given = TRUE;
    break;
  case  HSMHV2_MOD_RDRVMAXL:
    mod->HSMHV2_rdrvmaxl = value->rValue;
    mod->HSMHV2_rdrvmaxl_Given = TRUE;
    break;
  case  HSMHV2_MOD_RDRVMAXLP:
    mod->HSMHV2_rdrvmaxlp = value->rValue;
    mod->HSMHV2_rdrvmaxlp_Given = TRUE;
    break;
  case  HSMHV2_MOD_RDRMUEL:
    mod->HSMHV2_rdrmuel = value->rValue;
    mod->HSMHV2_rdrmuel_Given = TRUE;
    break;
  case  HSMHV2_MOD_RDRMUELP:
    mod->HSMHV2_rdrmuelp = value->rValue;
    mod->HSMHV2_rdrmuelp_Given = TRUE;
    break;
  case  HSMHV2_MOD_RDRQOVER:
    mod->HSMHV2_rdrqover = value->rValue;
    mod->HSMHV2_rdrqover_Given = TRUE;
    break;
  case HSMHV2_MOD_QOVADD:
    mod->HSMHV2_qovadd = value->rValue;
    mod->HSMHV2_qovadd_Given = TRUE;
    break;
  case HSMHV2_MOD_JS0D:
    mod->HSMHV2_js0d = value->rValue;
    mod->HSMHV2_js0d_Given = TRUE;
    break;
  case HSMHV2_MOD_JS0SWD:
    mod->HSMHV2_js0swd = value->rValue;
    mod->HSMHV2_js0swd_Given = TRUE;
    break;
  case HSMHV2_MOD_NJD:
    mod->HSMHV2_njd = value->rValue;
    mod->HSMHV2_njd_Given = TRUE;
    break;
  case HSMHV2_MOD_NJSWD:
    mod->HSMHV2_njswd = value->rValue;
    mod->HSMHV2_njswd_Given = TRUE;
    break;
  case HSMHV2_MOD_XTID:
    mod->HSMHV2_xtid = value->rValue;
    mod->HSMHV2_xtid_Given = TRUE;
    break;
  case HSMHV2_MOD_CJD:
    mod->HSMHV2_cjd = value->rValue;
    mod->HSMHV2_cjd_Given = TRUE;
    break;
  case HSMHV2_MOD_CJSWD:
    mod->HSMHV2_cjswd = value->rValue;
    mod->HSMHV2_cjswd_Given = TRUE;
    break;
  case HSMHV2_MOD_CJSWGD:
    mod->HSMHV2_cjswgd = value->rValue;
    mod->HSMHV2_cjswgd_Given = TRUE;
    break;
  case HSMHV2_MOD_MJD:
    mod->HSMHV2_mjd = value->rValue;
    mod->HSMHV2_mjd_Given = TRUE;
    break;
  case HSMHV2_MOD_MJSWD:
    mod->HSMHV2_mjswd = value->rValue;
    mod->HSMHV2_mjswd_Given = TRUE;
    break;
  case HSMHV2_MOD_MJSWGD:
    mod->HSMHV2_mjswgd = value->rValue;
    mod->HSMHV2_mjswgd_Given = TRUE;
    break;
  case HSMHV2_MOD_PBD:
    mod->HSMHV2_pbd = value->rValue;
    mod->HSMHV2_pbd_Given = TRUE;
    break;
  case HSMHV2_MOD_PBSWD:
    mod->HSMHV2_pbswd = value->rValue;
    mod->HSMHV2_pbswd_Given = TRUE;
    break;
  case HSMHV2_MOD_PBSWDG:
    mod->HSMHV2_pbswgd = value->rValue;
    mod->HSMHV2_pbswgd_Given = TRUE;
    break;
  case HSMHV2_MOD_XTI2D:
    mod->HSMHV2_xti2d = value->rValue;
    mod->HSMHV2_xti2d_Given = TRUE;
    break;
  case HSMHV2_MOD_CISBD:
    mod->HSMHV2_cisbd = value->rValue;
    mod->HSMHV2_cisbd_Given = TRUE;
    break;
  case HSMHV2_MOD_CVBD:
    mod->HSMHV2_cvbd = value->rValue;
    mod->HSMHV2_cvbd_Given = TRUE;
    break;
  case HSMHV2_MOD_CTEMPD:
    mod->HSMHV2_ctempd = value->rValue;
    mod->HSMHV2_ctempd_Given = TRUE;
    break;
  case HSMHV2_MOD_CISBKD:
    mod->HSMHV2_cisbkd = value->rValue;
    mod->HSMHV2_cisbkd_Given = TRUE;
    break;
  case HSMHV2_MOD_DIVXD:
    mod->HSMHV2_divxd = value->rValue;
    mod->HSMHV2_divxd_Given = TRUE;
    break;
  case HSMHV2_MOD_VDIFFJD:
    mod->HSMHV2_vdiffjd = value->rValue;
    mod->HSMHV2_vdiffjd_Given = TRUE;
    break;
  case HSMHV2_MOD_JS0S:
    mod->HSMHV2_js0s = value->rValue;
    mod->HSMHV2_js0s_Given = TRUE;
    break;
  case HSMHV2_MOD_JS0SWS:
    mod->HSMHV2_js0sws = value->rValue;
    mod->HSMHV2_js0sws_Given = TRUE;
    break;
  case HSMHV2_MOD_NJS:
    mod->HSMHV2_njs = value->rValue;
    mod->HSMHV2_njs_Given = TRUE;
    break;
  case HSMHV2_MOD_NJSWS:
    mod->HSMHV2_njsws = value->rValue;
    mod->HSMHV2_njsws_Given = TRUE;
    break;
  case HSMHV2_MOD_XTIS:
    mod->HSMHV2_xtis = value->rValue;
    mod->HSMHV2_xtis_Given = TRUE;
    break;
  case HSMHV2_MOD_CJS:
    mod->HSMHV2_cjs = value->rValue;
    mod->HSMHV2_cjs_Given = TRUE;
    break;
  case HSMHV2_MOD_CJSSW:
    mod->HSMHV2_cjsws = value->rValue;
    mod->HSMHV2_cjsws_Given = TRUE;
    break;
  case HSMHV2_MOD_CJSWGS:
    mod->HSMHV2_cjswgs = value->rValue;
    mod->HSMHV2_cjswgs_Given = TRUE;
    break;
  case HSMHV2_MOD_MJS:
    mod->HSMHV2_mjs = value->rValue;
    mod->HSMHV2_mjs_Given = TRUE;
    break;
  case HSMHV2_MOD_MJSWS:
    mod->HSMHV2_mjsws = value->rValue;
    mod->HSMHV2_mjsws_Given = TRUE;
    break;
  case HSMHV2_MOD_MJSWGS:
    mod->HSMHV2_mjswgs = value->rValue;
    mod->HSMHV2_mjswgs_Given = TRUE;
    break;
  case HSMHV2_MOD_PBS:
    mod->HSMHV2_pbs = value->rValue;
    mod->HSMHV2_pbs_Given = TRUE;
    break;
  case HSMHV2_MOD_PBSWS:
    mod->HSMHV2_pbsws = value->rValue;
    mod->HSMHV2_pbsws_Given = TRUE;
    break;
  case HSMHV2_MOD_PBSWSG:
    mod->HSMHV2_pbswgs = value->rValue;
    mod->HSMHV2_pbswgs_Given = TRUE;
    break;
  case HSMHV2_MOD_XTI2S:
    mod->HSMHV2_xti2s = value->rValue;
    mod->HSMHV2_xti2s_Given = TRUE;
    break;
  case HSMHV2_MOD_CISBS:
    mod->HSMHV2_cisbs = value->rValue;
    mod->HSMHV2_cisbs_Given = TRUE;
    break;
  case HSMHV2_MOD_CVBS:
    mod->HSMHV2_cvbs = value->rValue;
    mod->HSMHV2_cvbs_Given = TRUE;
    break;
  case HSMHV2_MOD_CTEMPS:
    mod->HSMHV2_ctemps = value->rValue;
    mod->HSMHV2_ctemps_Given = TRUE;
    break;
  case HSMHV2_MOD_CISBKS:
    mod->HSMHV2_cisbks = value->rValue;
    mod->HSMHV2_cisbks_Given = TRUE;
    break;
  case HSMHV2_MOD_DIVXS:
    mod->HSMHV2_divxs = value->rValue;
    mod->HSMHV2_divxs_Given = TRUE;
    break;
  case HSMHV2_MOD_VDIFFJS:
    mod->HSMHV2_vdiffjs = value->rValue;
    mod->HSMHV2_vdiffjs_Given = TRUE;
    break;
  case HSMHV2_MOD_SHEMAX:
    mod->HSMHV2_shemax = value->rValue;
    mod->HSMHV2_shemax_Given = TRUE;
    break;
  case HSMHV2_MOD_VGSMIN:
    mod->HSMHV2_vgsmin = value->rValue;
    mod->HSMHV2_vgsmin_Given = TRUE;
    break;
  case HSMHV2_MOD_GDSLEAK:
    mod->HSMHV2_gdsleak = value->rValue;
    mod->HSMHV2_gdsleak_Given = TRUE;
    break;
  case HSMHV2_MOD_RDRBB:
    mod->HSMHV2_rdrbb = value->rValue;
    mod->HSMHV2_rdrbb_Given = TRUE;
    break;
  case HSMHV2_MOD_RDRBBTMP:
    mod->HSMHV2_rdrbbtmp = value->rValue;
    mod->HSMHV2_rdrbbtmp_Given = TRUE;
    break;


  /* binning parameters */
  case  HSMHV2_MOD_LMIN:
    mod->HSMHV2_lmin = value->rValue;
    mod->HSMHV2_lmin_Given = TRUE;
    break;
  case  HSMHV2_MOD_LMAX:
    mod->HSMHV2_lmax = value->rValue;
    mod->HSMHV2_lmax_Given = TRUE;
    break;
  case  HSMHV2_MOD_WMIN:
    mod->HSMHV2_wmin = value->rValue;
    mod->HSMHV2_wmin_Given = TRUE;
    break;
  case  HSMHV2_MOD_WMAX:
    mod->HSMHV2_wmax = value->rValue;
    mod->HSMHV2_wmax_Given = TRUE;
    break;
  case  HSMHV2_MOD_LBINN:
    mod->HSMHV2_lbinn = value->rValue;
    mod->HSMHV2_lbinn_Given = TRUE;
    break;
  case  HSMHV2_MOD_WBINN:
    mod->HSMHV2_wbinn = value->rValue;
    mod->HSMHV2_wbinn_Given = TRUE;
    break;

  /* Length dependence */
  case  HSMHV2_MOD_LVMAX:
    mod->HSMHV2_lvmax = value->rValue;
    mod->HSMHV2_lvmax_Given = TRUE;
    break;
  case  HSMHV2_MOD_LBGTMP1:
    mod->HSMHV2_lbgtmp1 = value->rValue;
    mod->HSMHV2_lbgtmp1_Given = TRUE;
    break;
  case  HSMHV2_MOD_LBGTMP2:
    mod->HSMHV2_lbgtmp2 = value->rValue;
    mod->HSMHV2_lbgtmp2_Given = TRUE;
    break;
  case  HSMHV2_MOD_LEG0:
    mod->HSMHV2_leg0 = value->rValue;
    mod->HSMHV2_leg0_Given = TRUE;
    break;
  case  HSMHV2_MOD_LVFBOVER:
    mod->HSMHV2_lvfbover = value->rValue;
    mod->HSMHV2_lvfbover_Given = TRUE;
    break;
  case  HSMHV2_MOD_LNOVER:
    mod->HSMHV2_lnover = value->rValue;
    mod->HSMHV2_lnover_Given = TRUE;
    break;
  case  HSMHV2_MOD_LNOVERS:
    mod->HSMHV2_lnovers = value->rValue;
    mod->HSMHV2_lnovers_Given = TRUE;
    break;
  case  HSMHV2_MOD_LWL2:
    mod->HSMHV2_lwl2 = value->rValue;
    mod->HSMHV2_lwl2_Given = TRUE;
    break;
  case  HSMHV2_MOD_LVFBC:
    mod->HSMHV2_lvfbc = value->rValue;
    mod->HSMHV2_lvfbc_Given = TRUE;
    break;
  case  HSMHV2_MOD_LNSUBC:
    mod->HSMHV2_lnsubc = value->rValue;
    mod->HSMHV2_lnsubc_Given = TRUE;
    break;
  case  HSMHV2_MOD_LNSUBP:
    mod->HSMHV2_lnsubp = value->rValue;
    mod->HSMHV2_lnsubp_Given = TRUE;
    break;
  case  HSMHV2_MOD_LSCP1:
    mod->HSMHV2_lscp1 = value->rValue;
    mod->HSMHV2_lscp1_Given = TRUE;
    break;
  case  HSMHV2_MOD_LSCP2:
    mod->HSMHV2_lscp2 = value->rValue;
    mod->HSMHV2_lscp2_Given = TRUE;
    break;
  case  HSMHV2_MOD_LSCP3:
    mod->HSMHV2_lscp3 = value->rValue;
    mod->HSMHV2_lscp3_Given = TRUE;
    break;
  case  HSMHV2_MOD_LSC1:
    mod->HSMHV2_lsc1 = value->rValue;
    mod->HSMHV2_lsc1_Given = TRUE;
    break;
  case  HSMHV2_MOD_LSC2:
    mod->HSMHV2_lsc2 = value->rValue;
    mod->HSMHV2_lsc2_Given = TRUE;
    break;
  case  HSMHV2_MOD_LSC3:
    mod->HSMHV2_lsc3 = value->rValue;
    mod->HSMHV2_lsc3_Given = TRUE;
    break;
  case  HSMHV2_MOD_LPGD1:
    mod->HSMHV2_lpgd1 = value->rValue;
    mod->HSMHV2_lpgd1_Given = TRUE;
    break;
  case  HSMHV2_MOD_LNDEP:
    mod->HSMHV2_lndep = value->rValue;
    mod->HSMHV2_lndep_Given = TRUE;
    break;
  case  HSMHV2_MOD_LNINV:
    mod->HSMHV2_lninv = value->rValue;
    mod->HSMHV2_lninv_Given = TRUE;
    break;
  case  HSMHV2_MOD_LMUECB0:
    mod->HSMHV2_lmuecb0 = value->rValue;
    mod->HSMHV2_lmuecb0_Given = TRUE;
    break;
  case  HSMHV2_MOD_LMUECB1:
    mod->HSMHV2_lmuecb1 = value->rValue;
    mod->HSMHV2_lmuecb1_Given = TRUE;
    break;
  case  HSMHV2_MOD_LMUEPH1:
    mod->HSMHV2_lmueph1 = value->rValue;
    mod->HSMHV2_lmueph1_Given = TRUE;
    break;
  case  HSMHV2_MOD_LVTMP:
    mod->HSMHV2_lvtmp = value->rValue;
    mod->HSMHV2_lvtmp_Given = TRUE;
    break;
  case  HSMHV2_MOD_LWVTH0:
    mod->HSMHV2_lwvth0 = value->rValue;
    mod->HSMHV2_lwvth0_Given = TRUE;
    break;
  case  HSMHV2_MOD_LMUESR1:
    mod->HSMHV2_lmuesr1 = value->rValue;
    mod->HSMHV2_lmuesr1_Given = TRUE;
    break;
  case  HSMHV2_MOD_LMUETMP:
    mod->HSMHV2_lmuetmp = value->rValue;
    mod->HSMHV2_lmuetmp_Given = TRUE;
    break;
  case  HSMHV2_MOD_LSUB1:
    mod->HSMHV2_lsub1 = value->rValue;
    mod->HSMHV2_lsub1_Given = TRUE;
    break;
  case  HSMHV2_MOD_LSUB2:
    mod->HSMHV2_lsub2 = value->rValue;
    mod->HSMHV2_lsub2_Given = TRUE;
    break;
  case  HSMHV2_MOD_LSVDS:
    mod->HSMHV2_lsvds = value->rValue;
    mod->HSMHV2_lsvds_Given = TRUE;
    break;
  case  HSMHV2_MOD_LSVBS:
    mod->HSMHV2_lsvbs = value->rValue;
    mod->HSMHV2_lsvbs_Given = TRUE;
    break;
  case  HSMHV2_MOD_LSVGS:
    mod->HSMHV2_lsvgs = value->rValue;
    mod->HSMHV2_lsvgs_Given = TRUE;
    break;
  case  HSMHV2_MOD_LFN1:
    mod->HSMHV2_lfn1 = value->rValue;
    mod->HSMHV2_lfn1_Given = TRUE;
    break;
  case  HSMHV2_MOD_LFN2:
    mod->HSMHV2_lfn2 = value->rValue;
    mod->HSMHV2_lfn2_Given = TRUE;
    break;
  case  HSMHV2_MOD_LFN3:
    mod->HSMHV2_lfn3 = value->rValue;
    mod->HSMHV2_lfn3_Given = TRUE;
    break;
  case  HSMHV2_MOD_LFVBS:
    mod->HSMHV2_lfvbs = value->rValue;
    mod->HSMHV2_lfvbs_Given = TRUE;
    break;
  case  HSMHV2_MOD_LNSTI:
    mod->HSMHV2_lnsti = value->rValue;
    mod->HSMHV2_lnsti_Given = TRUE;
    break;
  case  HSMHV2_MOD_LWSTI:
    mod->HSMHV2_lwsti = value->rValue;
    mod->HSMHV2_lwsti_Given = TRUE;
    break;
  case  HSMHV2_MOD_LSCSTI1:
    mod->HSMHV2_lscsti1 = value->rValue;
    mod->HSMHV2_lscsti1_Given = TRUE;
    break;
  case  HSMHV2_MOD_LSCSTI2:
    mod->HSMHV2_lscsti2 = value->rValue;
    mod->HSMHV2_lscsti2_Given = TRUE;
    break;
  case  HSMHV2_MOD_LVTHSTI:
    mod->HSMHV2_lvthsti = value->rValue;
    mod->HSMHV2_lvthsti_Given = TRUE;
    break;
  case  HSMHV2_MOD_LMUESTI1:
    mod->HSMHV2_lmuesti1 = value->rValue;
    mod->HSMHV2_lmuesti1_Given = TRUE;
    break;
  case  HSMHV2_MOD_LMUESTI2:
    mod->HSMHV2_lmuesti2 = value->rValue;
    mod->HSMHV2_lmuesti2_Given = TRUE;
    break;
  case  HSMHV2_MOD_LMUESTI3:
    mod->HSMHV2_lmuesti3 = value->rValue;
    mod->HSMHV2_lmuesti3_Given = TRUE;
    break;
  case  HSMHV2_MOD_LNSUBPSTI1:
    mod->HSMHV2_lnsubpsti1 = value->rValue;
    mod->HSMHV2_lnsubpsti1_Given = TRUE;
    break;
  case  HSMHV2_MOD_LNSUBPSTI2:
    mod->HSMHV2_lnsubpsti2 = value->rValue;
    mod->HSMHV2_lnsubpsti2_Given = TRUE;
    break;
  case  HSMHV2_MOD_LNSUBPSTI3:
    mod->HSMHV2_lnsubpsti3 = value->rValue;
    mod->HSMHV2_lnsubpsti3_Given = TRUE;
    break;
  case  HSMHV2_MOD_LCGSO:
    mod->HSMHV2_lcgso = value->rValue;
    mod->HSMHV2_lcgso_Given = TRUE;
    break;
  case  HSMHV2_MOD_LCGDO:
    mod->HSMHV2_lcgdo = value->rValue;
    mod->HSMHV2_lcgdo_Given = TRUE;
    break;
  case  HSMHV2_MOD_LJS0:
    mod->HSMHV2_ljs0 = value->rValue;
    mod->HSMHV2_ljs0_Given = TRUE;
    break;
  case  HSMHV2_MOD_LJS0SW:
    mod->HSMHV2_ljs0sw = value->rValue;
    mod->HSMHV2_ljs0sw_Given = TRUE;
    break;
  case  HSMHV2_MOD_LNJ:
    mod->HSMHV2_lnj = value->rValue;
    mod->HSMHV2_lnj_Given = TRUE;
    break;
  case  HSMHV2_MOD_LCISBK:
    mod->HSMHV2_lcisbk = value->rValue;
    mod->HSMHV2_lcisbk_Given = TRUE;
    break;
  case  HSMHV2_MOD_LCLM1:
    mod->HSMHV2_lclm1 = value->rValue;
    mod->HSMHV2_lclm1_Given = TRUE;
    break;
  case  HSMHV2_MOD_LCLM2:
    mod->HSMHV2_lclm2 = value->rValue;
    mod->HSMHV2_lclm2_Given = TRUE;
    break;
  case  HSMHV2_MOD_LCLM3:
    mod->HSMHV2_lclm3 = value->rValue;
    mod->HSMHV2_lclm3_Given = TRUE;
    break;
  case  HSMHV2_MOD_LWFC:
    mod->HSMHV2_lwfc = value->rValue;
    mod->HSMHV2_lwfc_Given = TRUE;
    break;
  case  HSMHV2_MOD_LGIDL1:
    mod->HSMHV2_lgidl1 = value->rValue;
    mod->HSMHV2_lgidl1_Given = TRUE;
    break;
  case  HSMHV2_MOD_LGIDL2:
    mod->HSMHV2_lgidl2 = value->rValue;
    mod->HSMHV2_lgidl2_Given = TRUE;
    break;
  case  HSMHV2_MOD_LGLEAK1:
    mod->HSMHV2_lgleak1 = value->rValue;
    mod->HSMHV2_lgleak1_Given = TRUE;
    break;
  case  HSMHV2_MOD_LGLEAK2:
    mod->HSMHV2_lgleak2 = value->rValue;
    mod->HSMHV2_lgleak2_Given = TRUE;
    break;
  case  HSMHV2_MOD_LGLEAK3:
    mod->HSMHV2_lgleak3 = value->rValue;
    mod->HSMHV2_lgleak3_Given = TRUE;
    break;
  case  HSMHV2_MOD_LGLEAK6:
    mod->HSMHV2_lgleak6 = value->rValue;
    mod->HSMHV2_lgleak6_Given = TRUE;
    break;
  case  HSMHV2_MOD_LGLKSD1:
    mod->HSMHV2_lglksd1 = value->rValue;
    mod->HSMHV2_lglksd1_Given = TRUE;
    break;
  case  HSMHV2_MOD_LGLKSD2:
    mod->HSMHV2_lglksd2 = value->rValue;
    mod->HSMHV2_lglksd2_Given = TRUE;
    break;
  case  HSMHV2_MOD_LGLKB1:
    mod->HSMHV2_lglkb1 = value->rValue;
    mod->HSMHV2_lglkb1_Given = TRUE;
    break;
  case  HSMHV2_MOD_LGLKB2:
    mod->HSMHV2_lglkb2 = value->rValue;
    mod->HSMHV2_lglkb2_Given = TRUE;
    break;
  case  HSMHV2_MOD_LNFTRP:
    mod->HSMHV2_lnftrp = value->rValue;
    mod->HSMHV2_lnftrp_Given = TRUE;
    break;
  case  HSMHV2_MOD_LNFALP:
    mod->HSMHV2_lnfalp = value->rValue;
    mod->HSMHV2_lnfalp_Given = TRUE;
    break;
  case  HSMHV2_MOD_LVDIFFJ:
    mod->HSMHV2_lvdiffj = value->rValue;
    mod->HSMHV2_lvdiffj_Given = TRUE;
    break;
  case  HSMHV2_MOD_LIBPC1:
    mod->HSMHV2_libpc1 = value->rValue;
    mod->HSMHV2_libpc1_Given = TRUE;
    break;
  case  HSMHV2_MOD_LIBPC2:
    mod->HSMHV2_libpc2 = value->rValue;
    mod->HSMHV2_libpc2_Given = TRUE;
    break;
    break;
  case  HSMHV2_MOD_LCGBO:
    mod->HSMHV2_lcgbo = value->rValue;
    mod->HSMHV2_lcgbo_Given = TRUE;
    break;
  case  HSMHV2_MOD_LCVDSOVER:
    mod->HSMHV2_lcvdsover = value->rValue;
    mod->HSMHV2_lcvdsover_Given = TRUE;
    break;
  case  HSMHV2_MOD_LFALPH:
    mod->HSMHV2_lfalph = value->rValue;
    mod->HSMHV2_lfalph_Given = TRUE;
    break;
  case  HSMHV2_MOD_LNPEXT:
    mod->HSMHV2_lnpext = value->rValue;
    mod->HSMHV2_lnpext_Given = TRUE;
    break;
  case  HSMHV2_MOD_LPOWRAT:
    mod->HSMHV2_lpowrat = value->rValue;
    mod->HSMHV2_lpowrat_Given = TRUE;
    break;
  case  HSMHV2_MOD_LRD:
    mod->HSMHV2_lrd = value->rValue;
    mod->HSMHV2_lrd_Given = TRUE;
    break;
  case  HSMHV2_MOD_LRD22:
    mod->HSMHV2_lrd22 = value->rValue;
    mod->HSMHV2_lrd22_Given = TRUE;
    break;
  case  HSMHV2_MOD_LRD23:
    mod->HSMHV2_lrd23 = value->rValue;
    mod->HSMHV2_lrd23_Given = TRUE;
    break;
  case  HSMHV2_MOD_LRD24:
    mod->HSMHV2_lrd24 = value->rValue;
    mod->HSMHV2_lrd24_Given = TRUE;
    break;
  case  HSMHV2_MOD_LRDICT1:
    mod->HSMHV2_lrdict1 = value->rValue;
    mod->HSMHV2_lrdict1_Given = TRUE;
    break;
  case  HSMHV2_MOD_LRDOV13:
    mod->HSMHV2_lrdov13 = value->rValue;
    mod->HSMHV2_lrdov13_Given = TRUE;
    break;
  case  HSMHV2_MOD_LRDSLP1:
    mod->HSMHV2_lrdslp1 = value->rValue;
    mod->HSMHV2_lrdslp1_Given = TRUE;
    break;
  case  HSMHV2_MOD_LRDVB:
    mod->HSMHV2_lrdvb = value->rValue;
    mod->HSMHV2_lrdvb_Given = TRUE;
    break;
  case  HSMHV2_MOD_LRDVD:
    mod->HSMHV2_lrdvd = value->rValue;
    mod->HSMHV2_lrdvd_Given = TRUE;
    break;
  case  HSMHV2_MOD_LRDVG11:
    mod->HSMHV2_lrdvg11 = value->rValue;
    mod->HSMHV2_lrdvg11_Given = TRUE;
    break;
  case  HSMHV2_MOD_LRS:
    mod->HSMHV2_lrs = value->rValue;
    mod->HSMHV2_lrs_Given = TRUE;
    break;
  case  HSMHV2_MOD_LRTH0:
    mod->HSMHV2_lrth0 = value->rValue;
    mod->HSMHV2_lrth0_Given = TRUE;
    break;
  case  HSMHV2_MOD_LVOVER:
    mod->HSMHV2_lvover = value->rValue;
    mod->HSMHV2_lvover_Given = TRUE;
    break;
  case HSMHV2_MOD_LJS0D:
    mod->HSMHV2_ljs0d = value->rValue;
    mod->HSMHV2_ljs0d_Given = TRUE;
    break;
  case HSMHV2_MOD_LJS0SWD:
    mod->HSMHV2_ljs0swd = value->rValue;
    mod->HSMHV2_ljs0swd_Given = TRUE;
    break;
  case HSMHV2_MOD_LNJD:
    mod->HSMHV2_lnjd = value->rValue;
    mod->HSMHV2_lnjd_Given = TRUE;
    break;
  case HSMHV2_MOD_LCISBKD:
    mod->HSMHV2_lcisbkd = value->rValue;
    mod->HSMHV2_lcisbkd_Given = TRUE;
    break;
  case HSMHV2_MOD_LVDIFFJD:
    mod->HSMHV2_lvdiffjd = value->rValue;
    mod->HSMHV2_lvdiffjd_Given = TRUE;
    break;
  case HSMHV2_MOD_LJS0S:
    mod->HSMHV2_ljs0s = value->rValue;
    mod->HSMHV2_ljs0s_Given = TRUE;
    break;
  case HSMHV2_MOD_LJS0SWS:
    mod->HSMHV2_ljs0sws = value->rValue;
    mod->HSMHV2_ljs0sws_Given = TRUE;
    break;
  case HSMHV2_MOD_LNJS:
    mod->HSMHV2_lnjs = value->rValue;
    mod->HSMHV2_lnjs_Given = TRUE;
    break;
  case HSMHV2_MOD_LCISBKS:
    mod->HSMHV2_lcisbks = value->rValue;
    mod->HSMHV2_lcisbks_Given = TRUE;
    break;
  case HSMHV2_MOD_LVDIFFJS:
    mod->HSMHV2_lvdiffjs = value->rValue;
    mod->HSMHV2_lvdiffjs_Given = TRUE;
    break;

  /* Width dependence */
  case  HSMHV2_MOD_WVMAX:
    mod->HSMHV2_wvmax = value->rValue;
    mod->HSMHV2_wvmax_Given = TRUE;
    break;
  case  HSMHV2_MOD_WBGTMP1:
    mod->HSMHV2_wbgtmp1 = value->rValue;
    mod->HSMHV2_wbgtmp1_Given = TRUE;
    break;
  case  HSMHV2_MOD_WBGTMP2:
    mod->HSMHV2_wbgtmp2 = value->rValue;
    mod->HSMHV2_wbgtmp2_Given = TRUE;
    break;
  case  HSMHV2_MOD_WEG0:
    mod->HSMHV2_weg0 = value->rValue;
    mod->HSMHV2_weg0_Given = TRUE;
    break;
  case  HSMHV2_MOD_WVFBOVER:
    mod->HSMHV2_wvfbover = value->rValue;
    mod->HSMHV2_wvfbover_Given = TRUE;
    break;
  case  HSMHV2_MOD_WNOVER:
    mod->HSMHV2_wnover = value->rValue;
    mod->HSMHV2_wnover_Given = TRUE;
    break;
  case  HSMHV2_MOD_WNOVERS:
    mod->HSMHV2_wnovers = value->rValue;
    mod->HSMHV2_wnovers_Given = TRUE;
    break;
  case  HSMHV2_MOD_WWL2:
    mod->HSMHV2_wwl2 = value->rValue;
    mod->HSMHV2_wwl2_Given = TRUE;
    break;
  case  HSMHV2_MOD_WVFBC:
    mod->HSMHV2_wvfbc = value->rValue;
    mod->HSMHV2_wvfbc_Given = TRUE;
    break;
  case  HSMHV2_MOD_WNSUBC:
    mod->HSMHV2_wnsubc = value->rValue;
    mod->HSMHV2_wnsubc_Given = TRUE;
    break;
  case  HSMHV2_MOD_WNSUBP:
    mod->HSMHV2_wnsubp = value->rValue;
    mod->HSMHV2_wnsubp_Given = TRUE;
    break;
  case  HSMHV2_MOD_WSCP1:
    mod->HSMHV2_wscp1 = value->rValue;
    mod->HSMHV2_wscp1_Given = TRUE;
    break;
  case  HSMHV2_MOD_WSCP2:
    mod->HSMHV2_wscp2 = value->rValue;
    mod->HSMHV2_wscp2_Given = TRUE;
    break;
  case  HSMHV2_MOD_WSCP3:
    mod->HSMHV2_wscp3 = value->rValue;
    mod->HSMHV2_wscp3_Given = TRUE;
    break;
  case  HSMHV2_MOD_WSC1:
    mod->HSMHV2_wsc1 = value->rValue;
    mod->HSMHV2_wsc1_Given = TRUE;
    break;
  case  HSMHV2_MOD_WSC2:
    mod->HSMHV2_wsc2 = value->rValue;
    mod->HSMHV2_wsc2_Given = TRUE;
    break;
  case  HSMHV2_MOD_WSC3:
    mod->HSMHV2_wsc3 = value->rValue;
    mod->HSMHV2_wsc3_Given = TRUE;
    break;
  case  HSMHV2_MOD_WPGD1:
    mod->HSMHV2_wpgd1 = value->rValue;
    mod->HSMHV2_wpgd1_Given = TRUE;
    break;
  case  HSMHV2_MOD_WNDEP:
    mod->HSMHV2_wndep = value->rValue;
    mod->HSMHV2_wndep_Given = TRUE;
    break;
  case  HSMHV2_MOD_WNINV:
    mod->HSMHV2_wninv = value->rValue;
    mod->HSMHV2_wninv_Given = TRUE;
    break;
  case  HSMHV2_MOD_WMUECB0:
    mod->HSMHV2_wmuecb0 = value->rValue;
    mod->HSMHV2_wmuecb0_Given = TRUE;
    break;
  case  HSMHV2_MOD_WMUECB1:
    mod->HSMHV2_wmuecb1 = value->rValue;
    mod->HSMHV2_wmuecb1_Given = TRUE;
    break;
  case  HSMHV2_MOD_WMUEPH1:
    mod->HSMHV2_wmueph1 = value->rValue;
    mod->HSMHV2_wmueph1_Given = TRUE;
    break;
  case  HSMHV2_MOD_WVTMP:
    mod->HSMHV2_wvtmp = value->rValue;
    mod->HSMHV2_wvtmp_Given = TRUE;
    break;
  case  HSMHV2_MOD_WWVTH0:
    mod->HSMHV2_wwvth0 = value->rValue;
    mod->HSMHV2_wwvth0_Given = TRUE;
    break;
  case  HSMHV2_MOD_WMUESR1:
    mod->HSMHV2_wmuesr1 = value->rValue;
    mod->HSMHV2_wmuesr1_Given = TRUE;
    break;
  case  HSMHV2_MOD_WMUETMP:
    mod->HSMHV2_wmuetmp = value->rValue;
    mod->HSMHV2_wmuetmp_Given = TRUE;
    break;
  case  HSMHV2_MOD_WSUB1:
    mod->HSMHV2_wsub1 = value->rValue;
    mod->HSMHV2_wsub1_Given = TRUE;
    break;
  case  HSMHV2_MOD_WSUB2:
    mod->HSMHV2_wsub2 = value->rValue;
    mod->HSMHV2_wsub2_Given = TRUE;
    break;
  case  HSMHV2_MOD_WSVDS:
    mod->HSMHV2_wsvds = value->rValue;
    mod->HSMHV2_wsvds_Given = TRUE;
    break;
  case  HSMHV2_MOD_WSVBS:
    mod->HSMHV2_wsvbs = value->rValue;
    mod->HSMHV2_wsvbs_Given = TRUE;
    break;
  case  HSMHV2_MOD_WSVGS:
    mod->HSMHV2_wsvgs = value->rValue;
    mod->HSMHV2_wsvgs_Given = TRUE;
    break;
  case  HSMHV2_MOD_WFN1:
    mod->HSMHV2_wfn1 = value->rValue;
    mod->HSMHV2_wfn1_Given = TRUE;
    break;
  case  HSMHV2_MOD_WFN2:
    mod->HSMHV2_wfn2 = value->rValue;
    mod->HSMHV2_wfn2_Given = TRUE;
    break;
  case  HSMHV2_MOD_WFN3:
    mod->HSMHV2_wfn3 = value->rValue;
    mod->HSMHV2_wfn3_Given = TRUE;
    break;
  case  HSMHV2_MOD_WFVBS:
    mod->HSMHV2_wfvbs = value->rValue;
    mod->HSMHV2_wfvbs_Given = TRUE;
    break;
  case  HSMHV2_MOD_WNSTI:
    mod->HSMHV2_wnsti = value->rValue;
    mod->HSMHV2_wnsti_Given = TRUE;
    break;
  case  HSMHV2_MOD_WWSTI:
    mod->HSMHV2_wwsti = value->rValue;
    mod->HSMHV2_wwsti_Given = TRUE;
    break;
  case  HSMHV2_MOD_WSCSTI1:
    mod->HSMHV2_wscsti1 = value->rValue;
    mod->HSMHV2_wscsti1_Given = TRUE;
    break;
  case  HSMHV2_MOD_WSCSTI2:
    mod->HSMHV2_wscsti2 = value->rValue;
    mod->HSMHV2_wscsti2_Given = TRUE;
    break;
  case  HSMHV2_MOD_WVTHSTI:
    mod->HSMHV2_wvthsti = value->rValue;
    mod->HSMHV2_wvthsti_Given = TRUE;
    break;
  case  HSMHV2_MOD_WMUESTI1:
    mod->HSMHV2_wmuesti1 = value->rValue;
    mod->HSMHV2_wmuesti1_Given = TRUE;
    break;
  case  HSMHV2_MOD_WMUESTI2:
    mod->HSMHV2_wmuesti2 = value->rValue;
    mod->HSMHV2_wmuesti2_Given = TRUE;
    break;
  case  HSMHV2_MOD_WMUESTI3:
    mod->HSMHV2_wmuesti3 = value->rValue;
    mod->HSMHV2_wmuesti3_Given = TRUE;
    break;
  case  HSMHV2_MOD_WNSUBPSTI1:
    mod->HSMHV2_wnsubpsti1 = value->rValue;
    mod->HSMHV2_wnsubpsti1_Given = TRUE;
    break;
  case  HSMHV2_MOD_WNSUBPSTI2:
    mod->HSMHV2_wnsubpsti2 = value->rValue;
    mod->HSMHV2_wnsubpsti2_Given = TRUE;
    break;
  case  HSMHV2_MOD_WNSUBPSTI3:
    mod->HSMHV2_wnsubpsti3 = value->rValue;
    mod->HSMHV2_wnsubpsti3_Given = TRUE;
    break;
  case  HSMHV2_MOD_WCGSO:
    mod->HSMHV2_wcgso = value->rValue;
    mod->HSMHV2_wcgso_Given = TRUE;
    break;
  case  HSMHV2_MOD_WCGDO:
    mod->HSMHV2_wcgdo = value->rValue;
    mod->HSMHV2_wcgdo_Given = TRUE;
    break;
  case  HSMHV2_MOD_WJS0:
    mod->HSMHV2_wjs0 = value->rValue;
    mod->HSMHV2_wjs0_Given = TRUE;
    break;
  case  HSMHV2_MOD_WJS0SW:
    mod->HSMHV2_wjs0sw = value->rValue;
    mod->HSMHV2_wjs0sw_Given = TRUE;
    break;
  case  HSMHV2_MOD_WNJ:
    mod->HSMHV2_wnj = value->rValue;
    mod->HSMHV2_wnj_Given = TRUE;
    break;
  case  HSMHV2_MOD_WCISBK:
    mod->HSMHV2_wcisbk = value->rValue;
    mod->HSMHV2_wcisbk_Given = TRUE;
    break;
  case  HSMHV2_MOD_WCLM1:
    mod->HSMHV2_wclm1 = value->rValue;
    mod->HSMHV2_wclm1_Given = TRUE;
    break;
  case  HSMHV2_MOD_WCLM2:
    mod->HSMHV2_wclm2 = value->rValue;
    mod->HSMHV2_wclm2_Given = TRUE;
    break;
  case  HSMHV2_MOD_WCLM3:
    mod->HSMHV2_wclm3 = value->rValue;
    mod->HSMHV2_wclm3_Given = TRUE;
    break;
  case  HSMHV2_MOD_WWFC:
    mod->HSMHV2_wwfc = value->rValue;
    mod->HSMHV2_wwfc_Given = TRUE;
    break;
  case  HSMHV2_MOD_WGIDL1:
    mod->HSMHV2_wgidl1 = value->rValue;
    mod->HSMHV2_wgidl1_Given = TRUE;
    break;
  case  HSMHV2_MOD_WGIDL2:
    mod->HSMHV2_wgidl2 = value->rValue;
    mod->HSMHV2_wgidl2_Given = TRUE;
    break;
  case  HSMHV2_MOD_WGLEAK1:
    mod->HSMHV2_wgleak1 = value->rValue;
    mod->HSMHV2_wgleak1_Given = TRUE;
    break;
  case  HSMHV2_MOD_WGLEAK2:
    mod->HSMHV2_wgleak2 = value->rValue;
    mod->HSMHV2_wgleak2_Given = TRUE;
    break;
  case  HSMHV2_MOD_WGLEAK3:
    mod->HSMHV2_wgleak3 = value->rValue;
    mod->HSMHV2_wgleak3_Given = TRUE;
    break;
  case  HSMHV2_MOD_WGLEAK6:
    mod->HSMHV2_wgleak6 = value->rValue;
    mod->HSMHV2_wgleak6_Given = TRUE;
    break;
  case  HSMHV2_MOD_WGLKSD1:
    mod->HSMHV2_wglksd1 = value->rValue;
    mod->HSMHV2_wglksd1_Given = TRUE;
    break;
  case  HSMHV2_MOD_WGLKSD2:
    mod->HSMHV2_wglksd2 = value->rValue;
    mod->HSMHV2_wglksd2_Given = TRUE;
    break;
  case  HSMHV2_MOD_WGLKB1:
    mod->HSMHV2_wglkb1 = value->rValue;
    mod->HSMHV2_wglkb1_Given = TRUE;
    break;
  case  HSMHV2_MOD_WGLKB2:
    mod->HSMHV2_wglkb2 = value->rValue;
    mod->HSMHV2_wglkb2_Given = TRUE;
    break;
  case  HSMHV2_MOD_WNFTRP:
    mod->HSMHV2_wnftrp = value->rValue;
    mod->HSMHV2_wnftrp_Given = TRUE;
    break;
  case  HSMHV2_MOD_WNFALP:
    mod->HSMHV2_wnfalp = value->rValue;
    mod->HSMHV2_wnfalp_Given = TRUE;
    break;
  case  HSMHV2_MOD_WVDIFFJ:
    mod->HSMHV2_wvdiffj = value->rValue;
    mod->HSMHV2_wvdiffj_Given = TRUE;
    break;
  case  HSMHV2_MOD_WIBPC1:
    mod->HSMHV2_wibpc1 = value->rValue;
    mod->HSMHV2_wibpc1_Given = TRUE;
    break;
  case  HSMHV2_MOD_WIBPC2:
    mod->HSMHV2_wibpc2 = value->rValue;
    mod->HSMHV2_wibpc2_Given = TRUE;
    break;
    break;
  case  HSMHV2_MOD_WCGBO:
    mod->HSMHV2_wcgbo = value->rValue;
    mod->HSMHV2_wcgbo_Given = TRUE;
    break;
  case  HSMHV2_MOD_WCVDSOVER:
    mod->HSMHV2_wcvdsover = value->rValue;
    mod->HSMHV2_wcvdsover_Given = TRUE;
    break;
  case  HSMHV2_MOD_WFALPH:
    mod->HSMHV2_wfalph = value->rValue;
    mod->HSMHV2_wfalph_Given = TRUE;
    break;
  case  HSMHV2_MOD_WNPEXT:
    mod->HSMHV2_wnpext = value->rValue;
    mod->HSMHV2_wnpext_Given = TRUE;
    break;
  case  HSMHV2_MOD_WPOWRAT:
    mod->HSMHV2_wpowrat = value->rValue;
    mod->HSMHV2_wpowrat_Given = TRUE;
    break;
  case  HSMHV2_MOD_WRD:
    mod->HSMHV2_wrd = value->rValue;
    mod->HSMHV2_wrd_Given = TRUE;
    break;
  case  HSMHV2_MOD_WRD22:
    mod->HSMHV2_wrd22 = value->rValue;
    mod->HSMHV2_wrd22_Given = TRUE;
    break;
  case  HSMHV2_MOD_WRD23:
    mod->HSMHV2_wrd23 = value->rValue;
    mod->HSMHV2_wrd23_Given = TRUE;
    break;
  case  HSMHV2_MOD_WRD24:
    mod->HSMHV2_wrd24 = value->rValue;
    mod->HSMHV2_wrd24_Given = TRUE;
    break;
  case  HSMHV2_MOD_WRDICT1:
    mod->HSMHV2_wrdict1 = value->rValue;
    mod->HSMHV2_wrdict1_Given = TRUE;
    break;
  case  HSMHV2_MOD_WRDOV13:
    mod->HSMHV2_wrdov13 = value->rValue;
    mod->HSMHV2_wrdov13_Given = TRUE;
    break;
  case  HSMHV2_MOD_WRDSLP1:
    mod->HSMHV2_wrdslp1 = value->rValue;
    mod->HSMHV2_wrdslp1_Given = TRUE;
    break;
  case  HSMHV2_MOD_WRDVB:
    mod->HSMHV2_wrdvb = value->rValue;
    mod->HSMHV2_wrdvb_Given = TRUE;
    break;
  case  HSMHV2_MOD_WRDVD:
    mod->HSMHV2_wrdvd = value->rValue;
    mod->HSMHV2_wrdvd_Given = TRUE;
    break;
  case  HSMHV2_MOD_WRDVG11:
    mod->HSMHV2_wrdvg11 = value->rValue;
    mod->HSMHV2_wrdvg11_Given = TRUE;
    break;
  case  HSMHV2_MOD_WRS:
    mod->HSMHV2_wrs = value->rValue;
    mod->HSMHV2_wrs_Given = TRUE;
    break;
  case  HSMHV2_MOD_WRTH0:
    mod->HSMHV2_wrth0 = value->rValue;
    mod->HSMHV2_wrth0_Given = TRUE;
    break;
  case  HSMHV2_MOD_WVOVER:
    mod->HSMHV2_wvover = value->rValue;
    mod->HSMHV2_wvover_Given = TRUE;
    break;
  case HSMHV2_MOD_WJS0D:
    mod->HSMHV2_wjs0d = value->rValue;
    mod->HSMHV2_wjs0d_Given = TRUE;
    break;
  case HSMHV2_MOD_WJS0SWD:
    mod->HSMHV2_wjs0swd = value->rValue;
    mod->HSMHV2_wjs0swd_Given = TRUE;
    break;
  case HSMHV2_MOD_WNJD:
    mod->HSMHV2_wnjd = value->rValue;
    mod->HSMHV2_wnjd_Given = TRUE;
    break;
  case HSMHV2_MOD_WCISBKD:
    mod->HSMHV2_wcisbkd = value->rValue;
    mod->HSMHV2_wcisbkd_Given = TRUE;
    break;
  case HSMHV2_MOD_WVDIFFJD:
    mod->HSMHV2_wvdiffjd = value->rValue;
    mod->HSMHV2_wvdiffjd_Given = TRUE;
    break;
  case HSMHV2_MOD_WJS0S:
    mod->HSMHV2_wjs0s = value->rValue;
    mod->HSMHV2_wjs0s_Given = TRUE;
    break;
  case HSMHV2_MOD_WJS0SWS:
    mod->HSMHV2_wjs0sws = value->rValue;
    mod->HSMHV2_wjs0sws_Given = TRUE;
    break;
  case HSMHV2_MOD_WNJS:
    mod->HSMHV2_wnjs = value->rValue;
    mod->HSMHV2_wnjs_Given = TRUE;
    break;
  case HSMHV2_MOD_WCISBKS:
    mod->HSMHV2_wcisbks = value->rValue;
    mod->HSMHV2_wcisbks_Given = TRUE;
    break;
  case HSMHV2_MOD_WVDIFFJS:
    mod->HSMHV2_wvdiffjs = value->rValue;
    mod->HSMHV2_wvdiffjs_Given = TRUE;
    break;

  /* Cross-term dependence */
  case  HSMHV2_MOD_PVMAX:
    mod->HSMHV2_pvmax = value->rValue;
    mod->HSMHV2_pvmax_Given = TRUE;
    break;
  case  HSMHV2_MOD_PBGTMP1:
    mod->HSMHV2_pbgtmp1 = value->rValue;
    mod->HSMHV2_pbgtmp1_Given = TRUE;
    break;
  case  HSMHV2_MOD_PBGTMP2:
    mod->HSMHV2_pbgtmp2 = value->rValue;
    mod->HSMHV2_pbgtmp2_Given = TRUE;
    break;
  case  HSMHV2_MOD_PEG0:
    mod->HSMHV2_peg0 = value->rValue;
    mod->HSMHV2_peg0_Given = TRUE;
    break;
  case  HSMHV2_MOD_PVFBOVER:
    mod->HSMHV2_pvfbover = value->rValue;
    mod->HSMHV2_pvfbover_Given = TRUE;
    break;
  case  HSMHV2_MOD_PNOVER:
    mod->HSMHV2_pnover = value->rValue;
    mod->HSMHV2_pnover_Given = TRUE;
    break;
  case  HSMHV2_MOD_PNOVERS:
    mod->HSMHV2_pnovers = value->rValue;
    mod->HSMHV2_pnovers_Given = TRUE;
    break;
  case  HSMHV2_MOD_PWL2:
    mod->HSMHV2_pwl2 = value->rValue;
    mod->HSMHV2_pwl2_Given = TRUE;
    break;
  case  HSMHV2_MOD_PVFBC:
    mod->HSMHV2_pvfbc = value->rValue;
    mod->HSMHV2_pvfbc_Given = TRUE;
    break;
  case  HSMHV2_MOD_PNSUBC:
    mod->HSMHV2_pnsubc = value->rValue;
    mod->HSMHV2_pnsubc_Given = TRUE;
    break;
  case  HSMHV2_MOD_PNSUBP:
    mod->HSMHV2_pnsubp = value->rValue;
    mod->HSMHV2_pnsubp_Given = TRUE;
    break;
  case  HSMHV2_MOD_PSCP1:
    mod->HSMHV2_pscp1 = value->rValue;
    mod->HSMHV2_pscp1_Given = TRUE;
    break;
  case  HSMHV2_MOD_PSCP2:
    mod->HSMHV2_pscp2 = value->rValue;
    mod->HSMHV2_pscp2_Given = TRUE;
    break;
  case  HSMHV2_MOD_PSCP3:
    mod->HSMHV2_pscp3 = value->rValue;
    mod->HSMHV2_pscp3_Given = TRUE;
    break;
  case  HSMHV2_MOD_PSC1:
    mod->HSMHV2_psc1 = value->rValue;
    mod->HSMHV2_psc1_Given = TRUE;
    break;
  case  HSMHV2_MOD_PSC2:
    mod->HSMHV2_psc2 = value->rValue;
    mod->HSMHV2_psc2_Given = TRUE;
    break;
  case  HSMHV2_MOD_PSC3:
    mod->HSMHV2_psc3 = value->rValue;
    mod->HSMHV2_psc3_Given = TRUE;
    break;
  case  HSMHV2_MOD_PPGD1:
    mod->HSMHV2_ppgd1 = value->rValue;
    mod->HSMHV2_ppgd1_Given = TRUE;
    break;
  case  HSMHV2_MOD_PNDEP:
    mod->HSMHV2_pndep = value->rValue;
    mod->HSMHV2_pndep_Given = TRUE;
    break;
  case  HSMHV2_MOD_PNINV:
    mod->HSMHV2_pninv = value->rValue;
    mod->HSMHV2_pninv_Given = TRUE;
    break;
  case  HSMHV2_MOD_PMUECB0:
    mod->HSMHV2_pmuecb0 = value->rValue;
    mod->HSMHV2_pmuecb0_Given = TRUE;
    break;
  case  HSMHV2_MOD_PMUECB1:
    mod->HSMHV2_pmuecb1 = value->rValue;
    mod->HSMHV2_pmuecb1_Given = TRUE;
    break;
  case  HSMHV2_MOD_PMUEPH1:
    mod->HSMHV2_pmueph1 = value->rValue;
    mod->HSMHV2_pmueph1_Given = TRUE;
    break;
  case  HSMHV2_MOD_PVTMP:
    mod->HSMHV2_pvtmp = value->rValue;
    mod->HSMHV2_pvtmp_Given = TRUE;
    break;
  case  HSMHV2_MOD_PWVTH0:
    mod->HSMHV2_pwvth0 = value->rValue;
    mod->HSMHV2_pwvth0_Given = TRUE;
    break;
  case  HSMHV2_MOD_PMUESR1:
    mod->HSMHV2_pmuesr1 = value->rValue;
    mod->HSMHV2_pmuesr1_Given = TRUE;
    break;
  case  HSMHV2_MOD_PMUETMP:
    mod->HSMHV2_pmuetmp = value->rValue;
    mod->HSMHV2_pmuetmp_Given = TRUE;
    break;
  case  HSMHV2_MOD_PSUB1:
    mod->HSMHV2_psub1 = value->rValue;
    mod->HSMHV2_psub1_Given = TRUE;
    break;
  case  HSMHV2_MOD_PSUB2:
    mod->HSMHV2_psub2 = value->rValue;
    mod->HSMHV2_psub2_Given = TRUE;
    break;
  case  HSMHV2_MOD_PSVDS:
    mod->HSMHV2_psvds = value->rValue;
    mod->HSMHV2_psvds_Given = TRUE;
    break;
  case  HSMHV2_MOD_PSVBS:
    mod->HSMHV2_psvbs = value->rValue;
    mod->HSMHV2_psvbs_Given = TRUE;
    break;
  case  HSMHV2_MOD_PSVGS:
    mod->HSMHV2_psvgs = value->rValue;
    mod->HSMHV2_psvgs_Given = TRUE;
    break;
  case  HSMHV2_MOD_PFN1:
    mod->HSMHV2_pfn1 = value->rValue;
    mod->HSMHV2_pfn1_Given = TRUE;
    break;
  case  HSMHV2_MOD_PFN2:
    mod->HSMHV2_pfn2 = value->rValue;
    mod->HSMHV2_pfn2_Given = TRUE;
    break;
  case  HSMHV2_MOD_PFN3:
    mod->HSMHV2_pfn3 = value->rValue;
    mod->HSMHV2_pfn3_Given = TRUE;
    break;
  case  HSMHV2_MOD_PFVBS:
    mod->HSMHV2_pfvbs = value->rValue;
    mod->HSMHV2_pfvbs_Given = TRUE;
    break;
  case  HSMHV2_MOD_PNSTI:
    mod->HSMHV2_pnsti = value->rValue;
    mod->HSMHV2_pnsti_Given = TRUE;
    break;
  case  HSMHV2_MOD_PWSTI:
    mod->HSMHV2_pwsti = value->rValue;
    mod->HSMHV2_pwsti_Given = TRUE;
    break;
  case  HSMHV2_MOD_PSCSTI1:
    mod->HSMHV2_pscsti1 = value->rValue;
    mod->HSMHV2_pscsti1_Given = TRUE;
    break;
  case  HSMHV2_MOD_PSCSTI2:
    mod->HSMHV2_pscsti2 = value->rValue;
    mod->HSMHV2_pscsti2_Given = TRUE;
    break;
  case  HSMHV2_MOD_PVTHSTI:
    mod->HSMHV2_pvthsti = value->rValue;
    mod->HSMHV2_pvthsti_Given = TRUE;
    break;
  case  HSMHV2_MOD_PMUESTI1:
    mod->HSMHV2_pmuesti1 = value->rValue;
    mod->HSMHV2_pmuesti1_Given = TRUE;
    break;
  case  HSMHV2_MOD_PMUESTI2:
    mod->HSMHV2_pmuesti2 = value->rValue;
    mod->HSMHV2_pmuesti2_Given = TRUE;
    break;
  case  HSMHV2_MOD_PMUESTI3:
    mod->HSMHV2_pmuesti3 = value->rValue;
    mod->HSMHV2_pmuesti3_Given = TRUE;
    break;
  case  HSMHV2_MOD_PNSUBPSTI1:
    mod->HSMHV2_pnsubpsti1 = value->rValue;
    mod->HSMHV2_pnsubpsti1_Given = TRUE;
    break;
  case  HSMHV2_MOD_PNSUBPSTI2:
    mod->HSMHV2_pnsubpsti2 = value->rValue;
    mod->HSMHV2_pnsubpsti2_Given = TRUE;
    break;
  case  HSMHV2_MOD_PNSUBPSTI3:
    mod->HSMHV2_pnsubpsti3 = value->rValue;
    mod->HSMHV2_pnsubpsti3_Given = TRUE;
    break;
  case  HSMHV2_MOD_PCGSO:
    mod->HSMHV2_pcgso = value->rValue;
    mod->HSMHV2_pcgso_Given = TRUE;
    break;
  case  HSMHV2_MOD_PCGDO:
    mod->HSMHV2_pcgdo = value->rValue;
    mod->HSMHV2_pcgdo_Given = TRUE;
    break;
  case  HSMHV2_MOD_PJS0:
    mod->HSMHV2_pjs0 = value->rValue;
    mod->HSMHV2_pjs0_Given = TRUE;
    break;
  case  HSMHV2_MOD_PJS0SW:
    mod->HSMHV2_pjs0sw = value->rValue;
    mod->HSMHV2_pjs0sw_Given = TRUE;
    break;
  case  HSMHV2_MOD_PNJ:
    mod->HSMHV2_pnj = value->rValue;
    mod->HSMHV2_pnj_Given = TRUE;
    break;
  case  HSMHV2_MOD_PCISBK:
    mod->HSMHV2_pcisbk = value->rValue;
    mod->HSMHV2_pcisbk_Given = TRUE;
    break;
  case  HSMHV2_MOD_PCLM1:
    mod->HSMHV2_pclm1 = value->rValue;
    mod->HSMHV2_pclm1_Given = TRUE;
    break;
  case  HSMHV2_MOD_PCLM2:
    mod->HSMHV2_pclm2 = value->rValue;
    mod->HSMHV2_pclm2_Given = TRUE;
    break;
  case  HSMHV2_MOD_PCLM3:
    mod->HSMHV2_pclm3 = value->rValue;
    mod->HSMHV2_pclm3_Given = TRUE;
    break;
  case  HSMHV2_MOD_PWFC:
    mod->HSMHV2_pwfc = value->rValue;
    mod->HSMHV2_pwfc_Given = TRUE;
    break;
  case  HSMHV2_MOD_PGIDL1:
    mod->HSMHV2_pgidl1 = value->rValue;
    mod->HSMHV2_pgidl1_Given = TRUE;
    break;
  case  HSMHV2_MOD_PGIDL2:
    mod->HSMHV2_pgidl2 = value->rValue;
    mod->HSMHV2_pgidl2_Given = TRUE;
    break;
  case  HSMHV2_MOD_PGLEAK1:
    mod->HSMHV2_pgleak1 = value->rValue;
    mod->HSMHV2_pgleak1_Given = TRUE;
    break;
  case  HSMHV2_MOD_PGLEAK2:
    mod->HSMHV2_pgleak2 = value->rValue;
    mod->HSMHV2_pgleak2_Given = TRUE;
    break;
  case  HSMHV2_MOD_PGLEAK3:
    mod->HSMHV2_pgleak3 = value->rValue;
    mod->HSMHV2_pgleak3_Given = TRUE;
    break;
  case  HSMHV2_MOD_PGLEAK6:
    mod->HSMHV2_pgleak6 = value->rValue;
    mod->HSMHV2_pgleak6_Given = TRUE;
    break;
  case  HSMHV2_MOD_PGLKSD1:
    mod->HSMHV2_pglksd1 = value->rValue;
    mod->HSMHV2_pglksd1_Given = TRUE;
    break;
  case  HSMHV2_MOD_PGLKSD2:
    mod->HSMHV2_pglksd2 = value->rValue;
    mod->HSMHV2_pglksd2_Given = TRUE;
    break;
  case  HSMHV2_MOD_PGLKB1:
    mod->HSMHV2_pglkb1 = value->rValue;
    mod->HSMHV2_pglkb1_Given = TRUE;
    break;
  case  HSMHV2_MOD_PGLKB2:
    mod->HSMHV2_pglkb2 = value->rValue;
    mod->HSMHV2_pglkb2_Given = TRUE;
    break;
  case  HSMHV2_MOD_PNFTRP:
    mod->HSMHV2_pnftrp = value->rValue;
    mod->HSMHV2_pnftrp_Given = TRUE;
    break;
  case  HSMHV2_MOD_PNFALP:
    mod->HSMHV2_pnfalp = value->rValue;
    mod->HSMHV2_pnfalp_Given = TRUE;
    break;
  case  HSMHV2_MOD_PVDIFFJ:
    mod->HSMHV2_pvdiffj = value->rValue;
    mod->HSMHV2_pvdiffj_Given = TRUE;
    break;
  case  HSMHV2_MOD_PIBPC1:
    mod->HSMHV2_pibpc1 = value->rValue;
    mod->HSMHV2_pibpc1_Given = TRUE;
    break;
  case  HSMHV2_MOD_PIBPC2:
    mod->HSMHV2_pibpc2 = value->rValue;
    mod->HSMHV2_pibpc2_Given = TRUE;
    break;
    break;
  case  HSMHV2_MOD_PCGBO:
    mod->HSMHV2_pcgbo = value->rValue;
    mod->HSMHV2_pcgbo_Given = TRUE;
    break;
  case  HSMHV2_MOD_PCVDSOVER:
    mod->HSMHV2_pcvdsover = value->rValue;
    mod->HSMHV2_pcvdsover_Given = TRUE;
    break;
  case  HSMHV2_MOD_PFALPH:
    mod->HSMHV2_pfalph = value->rValue;
    mod->HSMHV2_pfalph_Given = TRUE;
    break;
  case  HSMHV2_MOD_PNPEXT:
    mod->HSMHV2_pnpext = value->rValue;
    mod->HSMHV2_pnpext_Given = TRUE;
    break;
  case  HSMHV2_MOD_PPOWRAT:
    mod->HSMHV2_ppowrat = value->rValue;
    mod->HSMHV2_ppowrat_Given = TRUE;
    break;
  case  HSMHV2_MOD_PRD:
    mod->HSMHV2_prd = value->rValue;
    mod->HSMHV2_prd_Given = TRUE;
    break;
  case  HSMHV2_MOD_PRD22:
    mod->HSMHV2_prd22 = value->rValue;
    mod->HSMHV2_prd22_Given = TRUE;
    break;
  case  HSMHV2_MOD_PRD23:
    mod->HSMHV2_prd23 = value->rValue;
    mod->HSMHV2_prd23_Given = TRUE;
    break;
  case  HSMHV2_MOD_PRD24:
    mod->HSMHV2_prd24 = value->rValue;
    mod->HSMHV2_prd24_Given = TRUE;
    break;
  case  HSMHV2_MOD_PRDICT1:
    mod->HSMHV2_prdict1 = value->rValue;
    mod->HSMHV2_prdict1_Given = TRUE;
    break;
  case  HSMHV2_MOD_PRDOV13:
    mod->HSMHV2_prdov13 = value->rValue;
    mod->HSMHV2_prdov13_Given = TRUE;
    break;
  case  HSMHV2_MOD_PRDSLP1:
    mod->HSMHV2_prdslp1 = value->rValue;
    mod->HSMHV2_prdslp1_Given = TRUE;
    break;
  case  HSMHV2_MOD_PRDVB:
    mod->HSMHV2_prdvb = value->rValue;
    mod->HSMHV2_prdvb_Given = TRUE;
    break;
  case  HSMHV2_MOD_PRDVD:
    mod->HSMHV2_prdvd = value->rValue;
    mod->HSMHV2_prdvd_Given = TRUE;
    break;
  case  HSMHV2_MOD_PRDVG11:
    mod->HSMHV2_prdvg11 = value->rValue;
    mod->HSMHV2_prdvg11_Given = TRUE;
    break;
  case  HSMHV2_MOD_PRS:
    mod->HSMHV2_prs = value->rValue;
    mod->HSMHV2_prs_Given = TRUE;
    break;
  case  HSMHV2_MOD_PRTH0:
    mod->HSMHV2_prth0 = value->rValue;
    mod->HSMHV2_prth0_Given = TRUE;
    break;
  case  HSMHV2_MOD_PVOVER:
    mod->HSMHV2_pvover = value->rValue;
    mod->HSMHV2_pvover_Given = TRUE;
    break;

  case HSMHV2_MOD_PJS0D:
    mod->HSMHV2_pjs0d = value->rValue;
    mod->HSMHV2_pjs0d_Given = TRUE;
    break;
  case HSMHV2_MOD_PJS0SWD:
    mod->HSMHV2_pjs0swd = value->rValue;
    mod->HSMHV2_pjs0swd_Given = TRUE;
    break;
  case HSMHV2_MOD_PNJD:
    mod->HSMHV2_pnjd = value->rValue;
    mod->HSMHV2_pnjd_Given = TRUE;
    break;
  case HSMHV2_MOD_PCISBKD:
    mod->HSMHV2_pcisbkd = value->rValue;
    mod->HSMHV2_pcisbkd_Given = TRUE;
    break;
  case HSMHV2_MOD_PVDIFFJD:
    mod->HSMHV2_pvdiffjd = value->rValue;
    mod->HSMHV2_pvdiffjd_Given = TRUE;
    break;
  case HSMHV2_MOD_PJS0S:
    mod->HSMHV2_pjs0s = value->rValue;
    mod->HSMHV2_pjs0s_Given = TRUE;
    break;
  case HSMHV2_MOD_PJS0SWS:
    mod->HSMHV2_pjs0sws = value->rValue;
    mod->HSMHV2_pjs0sws_Given = TRUE;
    break;
  case HSMHV2_MOD_PNJS:
    mod->HSMHV2_pnjs = value->rValue;
    mod->HSMHV2_pnjs_Given = TRUE;
    break;
  case HSMHV2_MOD_PCISBKS:
    mod->HSMHV2_pcisbks = value->rValue;
    mod->HSMHV2_pcisbks_Given = TRUE;
    break;
  case HSMHV2_MOD_PVDIFFJS:
    mod->HSMHV2_pvdiffjs = value->rValue;
    mod->HSMHV2_pvdiffjs_Given = TRUE;
    break;
  case HSMHV2_MOD_NDEPM:
    mod->HSMHV2_ndepm = value->rValue;
    mod->HSMHV2_ndepm_Given = TRUE;
    break;
  case HSMHV2_MOD_TNDEP:
    mod->HSMHV2_tndep = value->rValue;
    mod->HSMHV2_tndep_Given = TRUE;
    break;
  case HSMHV2_MOD_DEPMUE0:
    mod->HSMHV2_depmue0 = value->rValue;
    mod->HSMHV2_depmue0_Given = TRUE;
    break;
  case HSMHV2_MOD_DEPMUE1:
    mod->HSMHV2_depmue1 = value->rValue;
    mod->HSMHV2_depmue1_Given = TRUE;
    break;
  case HSMHV2_MOD_DEPMUEBACK0:
    mod->HSMHV2_depmueback0 = value->rValue;
    mod->HSMHV2_depmueback0_Given = TRUE;
    break;
  case HSMHV2_MOD_DEPMUEBACK1:
    mod->HSMHV2_depmueback1 = value->rValue;
    mod->HSMHV2_depmueback1_Given = TRUE;
    break;
  case HSMHV2_MOD_DEPLEAK:
    mod->HSMHV2_depleak = value->rValue;
    mod->HSMHV2_depleak_Given = TRUE;
    break;
  case HSMHV2_MOD_DEPETA:
    mod->HSMHV2_depeta = value->rValue;
    mod->HSMHV2_depeta_Given = TRUE;
    break;
  case HSMHV2_MOD_DEPVMAX:
    mod->HSMHV2_depvmax = value->rValue;
    mod->HSMHV2_depvmax_Given = TRUE;
    break;
  case HSMHV2_MOD_DEPVDSEF1:
    mod->HSMHV2_depvdsef1 = value->rValue;
    mod->HSMHV2_depvdsef1_Given = TRUE;
    break;
  case HSMHV2_MOD_DEPVDSEF2:
    mod->HSMHV2_depvdsef2 = value->rValue;
    mod->HSMHV2_depvdsef2_Given = TRUE;
    break;
  case HSMHV2_MOD_DEPMUEPH0:
    mod->HSMHV2_depmueph0 = value->rValue;
    mod->HSMHV2_depmueph0_Given = TRUE;
    break;
  case HSMHV2_MOD_DEPMUEPH1:
    mod->HSMHV2_depmueph1 = value->rValue;
    mod->HSMHV2_depmueph1_Given = TRUE;
    break;
  case HSMHV2_MOD_DEPBB:
    mod->HSMHV2_depbb = value->rValue;
    mod->HSMHV2_depbb_Given = TRUE;
    break;
  case HSMHV2_MOD_DEPVTMP:
    mod->HSMHV2_depvtmp = value->rValue;
    mod->HSMHV2_depvtmp_Given = TRUE;
    break;
  case HSMHV2_MOD_DEPMUETMP:
    mod->HSMHV2_depmuetmp = value->rValue;
    mod->HSMHV2_depmuetmp_Given = TRUE;
    break;
  case HSMHV2_MOD_ISBREAK:
    mod->HSMHV2_isbreak = value->rValue;
    mod->HSMHV2_isbreak_Given = TRUE;
    break;
  case HSMHV2_MOD_RWELL:
    mod->HSMHV2_rwell = value->rValue;
    mod->HSMHV2_rwell_Given = TRUE;
    break;

  case HSMHV2_MOD_VGS_MAX:
    mod->HSMHV2vgsMax = value->rValue;
    mod->HSMHV2vgsMaxGiven = TRUE;
    break;
  case HSMHV2_MOD_VGD_MAX:
    mod->HSMHV2vgdMax = value->rValue;
    mod->HSMHV2vgdMaxGiven = TRUE;
    break;
  case HSMHV2_MOD_VGB_MAX:
    mod->HSMHV2vgbMax = value->rValue;
    mod->HSMHV2vgbMaxGiven = TRUE;
    break;
  case HSMHV2_MOD_VDS_MAX:
    mod->HSMHV2vdsMax = value->rValue;
    mod->HSMHV2vdsMaxGiven = TRUE;
    break;
  case HSMHV2_MOD_VBS_MAX:
    mod->HSMHV2vbsMax = value->rValue;
    mod->HSMHV2vbsMaxGiven = TRUE;
    break;
  case HSMHV2_MOD_VBD_MAX:
    mod->HSMHV2vbdMax = value->rValue;
    mod->HSMHV2vbdMaxGiven = TRUE;
    break;
  case HSMHV2_MOD_VGSR_MAX:
      mod->HSMHV2vgsrMax = value->rValue;
      mod->HSMHV2vgsrMaxGiven = TRUE;
      break;
  case HSMHV2_MOD_VGDR_MAX:
      mod->HSMHV2vgdrMax = value->rValue;
      mod->HSMHV2vgdrMaxGiven = TRUE;
      break;
  case HSMHV2_MOD_VGBR_MAX:
      mod->HSMHV2vgbrMax = value->rValue;
      mod->HSMHV2vgbrMaxGiven = TRUE;
      break;
  case HSMHV2_MOD_VBSR_MAX:
      mod->HSMHV2vbsrMax = value->rValue;
      mod->HSMHV2vbsrMaxGiven = TRUE;
      break;
  case HSMHV2_MOD_VBDR_MAX:
      mod->HSMHV2vbdrMax = value->rValue;
      mod->HSMHV2vbdrMaxGiven = TRUE;
      break;

  default:
    return(E_BADPARM);
  }
  return(OK);
}
