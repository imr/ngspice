/***********************************************************************

 HiSIM (Hiroshima University STARC IGFET Model)
 Copyright (C) 2014 Hiroshima University & STARC

 MODEL NAME : HiSIM_HV 
 ( VERSION : 2  SUBVERSION : 2  REVISION : 0 ) 
 Model Parameter 'VERSION' : 2.20
 FILE : hsmhvpar.c

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
#include "ngspice/ifsim.h"
#include "hsmhvdef.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"
#include "ngspice/fteext.h"

int HSMHVparam(
     int param,
     IFvalue *value,
     GENinstance *inst,
     IFvalue *select)
{
  double scale;

  HSMHVinstance *here = (HSMHVinstance*)inst;

  NG_IGNORE(select);

  if (!cp_getvar("scale", CP_REAL, &scale))
      scale = 1;

  switch (param) {
  case HSMHV_COSELFHEAT:
    here->HSMHV_coselfheat = value->iValue;
    here->HSMHV_coselfheat_Given = TRUE;
    break;
  case HSMHV_COSUBNODE:
    here->HSMHV_cosubnode = value->iValue;
    here->HSMHV_cosubnode_Given = TRUE;
    break;
  case HSMHV_W:
    here->HSMHV_w = value->rValue * scale;
    here->HSMHV_w_Given = TRUE;
    break;
  case HSMHV_L:
    here->HSMHV_l = value->rValue * scale;
    here->HSMHV_l_Given = TRUE;
    break;
  case HSMHV_AS:
    here->HSMHV_as = value->rValue * scale * scale;
    here->HSMHV_as_Given = TRUE;
    break;
  case HSMHV_AD:
    here->HSMHV_ad = value->rValue * scale * scale;
    here->HSMHV_ad_Given = TRUE;
    break;
  case HSMHV_PS:
    here->HSMHV_ps = value->rValue * scale;
    here->HSMHV_ps_Given = TRUE;
    break;
  case HSMHV_PD:
    here->HSMHV_pd = value->rValue * scale;
    here->HSMHV_pd_Given = TRUE;
    break;
  case HSMHV_NRS:
    here->HSMHV_nrs = value->rValue;
    here->HSMHV_nrs_Given = TRUE;
    break;
  case HSMHV_NRD:
    here->HSMHV_nrd = value->rValue;
    here->HSMHV_nrd_Given = TRUE;
    break;
  case HSMHV_DTEMP:
    here->HSMHV_dtemp = value->rValue;
    here->HSMHV_dtemp_Given = TRUE;
    break;
  case HSMHV_OFF:
    here->HSMHV_off = value->iValue;
    break;
  case HSMHV_IC_VBS:
    here->HSMHV_icVBS = value->rValue;
    here->HSMHV_icVBS_Given = TRUE;
    break;
  case HSMHV_IC_VDS:
    here->HSMHV_icVDS = value->rValue;
    here->HSMHV_icVDS_Given = TRUE;
    break;
  case HSMHV_IC_VGS:
    here->HSMHV_icVGS = value->rValue;
    here->HSMHV_icVGS_Given = TRUE;
    break;
  case HSMHV_IC:
    switch (value->v.numValue) {
    case 3:
      here->HSMHV_icVBS = *(value->v.vec.rVec + 2);
      here->HSMHV_icVBS_Given = TRUE;
    case 2:
      here->HSMHV_icVGS = *(value->v.vec.rVec + 1);
      here->HSMHV_icVGS_Given = TRUE;
    case 1:
      here->HSMHV_icVDS = *(value->v.vec.rVec);
      here->HSMHV_icVDS_Given = TRUE;
      break;
    default:
      return(E_BADPARM);
    }
    break;
  case  HSMHV_CORBNET: 
    here->HSMHV_corbnet = value->iValue;
    here->HSMHV_corbnet_Given = TRUE;
    break;
  case  HSMHV_RBPB:
    here->HSMHV_rbpb = value->rValue;
    here->HSMHV_rbpb_Given = TRUE;
    break;
  case  HSMHV_RBPD:
    here->HSMHV_rbpd = value->rValue;
    here->HSMHV_rbpd_Given = TRUE;
    break;
  case  HSMHV_RBPS:
    here->HSMHV_rbps = value->rValue;
    here->HSMHV_rbps_Given = TRUE;
    break;
  case  HSMHV_RBDB:
    here->HSMHV_rbdb = value->rValue;
    here->HSMHV_rbdb_Given = TRUE;
    break;
  case  HSMHV_RBSB:
    here->HSMHV_rbsb = value->rValue;
    here->HSMHV_rbsb_Given = TRUE;
    break;
  case  HSMHV_CORG: 
    here->HSMHV_corg = value->iValue;
    here->HSMHV_corg_Given = TRUE;
    break;
  case  HSMHV_NGCON:
    here->HSMHV_ngcon = value->rValue;
    here->HSMHV_ngcon_Given = TRUE;
    break;
  case  HSMHV_XGW:
    here->HSMHV_xgw = value->rValue;
    here->HSMHV_xgw_Given = TRUE;
    break;
  case  HSMHV_XGL:
    here->HSMHV_xgl = value->rValue;
    here->HSMHV_xgl_Given = TRUE;
    break;
  case  HSMHV_NF:
    here->HSMHV_nf = value->rValue;
    here->HSMHV_nf_Given = TRUE;
    break;
  case  HSMHV_SA:
    here->HSMHV_sa = value->rValue;
    here->HSMHV_sa_Given = TRUE;
    break;
  case  HSMHV_SB:
    here->HSMHV_sb = value->rValue;
    here->HSMHV_sb_Given = TRUE;
    break;
  case  HSMHV_SD:
    here->HSMHV_sd = value->rValue;
    here->HSMHV_sd_Given = TRUE;
    break;
  case  HSMHV_NSUBCDFM:
    here->HSMHV_nsubcdfm = value->rValue;
    here->HSMHV_nsubcdfm_Given = TRUE;
    break;
  case  HSMHV_M:
    here->HSMHV_m = value->rValue;
    here->HSMHV_m_Given = TRUE;
    break;
  case  HSMHV_SUBLD1:
    here->HSMHV_subld1 = value->rValue;
    here->HSMHV_subld1_Given = TRUE;
    break;
  case  HSMHV_SUBLD2:
    here->HSMHV_subld2 = value->rValue;
    here->HSMHV_subld2_Given = TRUE;
    break;
  case  HSMHV_LOVER:
    here->HSMHV_lover = value->rValue;
    here->HSMHV_lover_Given = TRUE;
    break;
  case  HSMHV_LOVERS:
    here->HSMHV_lovers = value->rValue;
    here->HSMHV_lovers_Given = TRUE;
    break;
  case  HSMHV_LOVERLD:
    here->HSMHV_loverld = value->rValue;
    here->HSMHV_loverld_Given = TRUE;
    break;
  case  HSMHV_LDRIFT1:
    here->HSMHV_ldrift1 = value->rValue;
    here->HSMHV_ldrift1_Given = TRUE;
    break;
  case  HSMHV_LDRIFT2:
    here->HSMHV_ldrift2 = value->rValue;
    here->HSMHV_ldrift2_Given = TRUE;
    break;
  case  HSMHV_LDRIFT1S:
    here->HSMHV_ldrift1s = value->rValue;
    here->HSMHV_ldrift1s_Given = TRUE;
    break;
  case  HSMHV_LDRIFT2S:
    here->HSMHV_ldrift2s = value->rValue;
    here->HSMHV_ldrift2s_Given = TRUE;
    break;
  default:
    return(E_BADPARM);
  }
  return(OK);
}
