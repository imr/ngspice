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
#include "hsmhv2def.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"
#include "ngspice/fteext.h"

int HSMHV2param(
     int param,
     IFvalue *value,
     GENinstance *inst,
     IFvalue *select)
{
  double scale;

  HSMHV2instance *here = (HSMHV2instance*)inst;

  NG_IGNORE(select);

  if (!cp_getvar("scale", CP_REAL, &scale, 0))
      scale = 1;

  switch (param) {
  case HSMHV2_COSELFHEAT:
    here->HSMHV2_coselfheat = value->iValue;
    here->HSMHV2_coselfheat_Given = TRUE;
    break;
  case HSMHV2_COSUBNODE:
    here->HSMHV2_cosubnode = value->iValue;
    here->HSMHV2_cosubnode_Given = TRUE;
    break;
  case HSMHV2_W:
    here->HSMHV2_w = value->rValue * scale;
    here->HSMHV2_w_Given = TRUE;
    break;
  case HSMHV2_L:
    here->HSMHV2_l = value->rValue * scale;
    here->HSMHV2_l_Given = TRUE;
    break;
  case HSMHV2_AS:
    here->HSMHV2_as = value->rValue * scale * scale;
    here->HSMHV2_as_Given = TRUE;
    break;
  case HSMHV2_AD:
    here->HSMHV2_ad = value->rValue * scale * scale;
    here->HSMHV2_ad_Given = TRUE;
    break;
  case HSMHV2_PS:
    here->HSMHV2_ps = value->rValue * scale;
    here->HSMHV2_ps_Given = TRUE;
    break;
  case HSMHV2_PD:
    here->HSMHV2_pd = value->rValue * scale;
    here->HSMHV2_pd_Given = TRUE;
    break;
  case HSMHV2_NRS:
    here->HSMHV2_nrs = value->rValue;
    here->HSMHV2_nrs_Given = TRUE;
    break;
  case HSMHV2_NRD:
    here->HSMHV2_nrd = value->rValue;
    here->HSMHV2_nrd_Given = TRUE;
    break;
  case HSMHV2_DTEMP:
    here->HSMHV2_dtemp = value->rValue;
    here->HSMHV2_dtemp_Given = TRUE;
    break;
  case HSMHV2_OFF:
    here->HSMHV2_off = value->iValue;
    break;
  case HSMHV2_IC_VBS:
    here->HSMHV2_icVBS = value->rValue;
    here->HSMHV2_icVBS_Given = TRUE;
    break;
  case HSMHV2_IC_VDS:
    here->HSMHV2_icVDS = value->rValue;
    here->HSMHV2_icVDS_Given = TRUE;
    break;
  case HSMHV2_IC_VGS:
    here->HSMHV2_icVGS = value->rValue;
    here->HSMHV2_icVGS_Given = TRUE;
    break;
  case HSMHV2_IC:
    /* FALLTHROUGH added to suppress GCC warning due to
     * -Wimplicit-fallthrough flag */
    switch (value->v.numValue) {
    case 3:
      here->HSMHV2_icVBS = *(value->v.vec.rVec + 2);
      here->HSMHV2_icVBS_Given = TRUE;
        /* FALLTHROUGH */
    case 2:
      here->HSMHV2_icVGS = *(value->v.vec.rVec + 1);
      here->HSMHV2_icVGS_Given = TRUE;
        /* FALLTHROUGH */
    case 1:
      here->HSMHV2_icVDS = *(value->v.vec.rVec);
      here->HSMHV2_icVDS_Given = TRUE;
      break;
    default:
      return(E_BADPARM);
    }
    break;
  case  HSMHV2_CORBNET: 
    here->HSMHV2_corbnet = value->iValue;
    here->HSMHV2_corbnet_Given = TRUE;
    break;
  case  HSMHV2_RBPB:
    here->HSMHV2_rbpb = value->rValue;
    here->HSMHV2_rbpb_Given = TRUE;
    break;
  case  HSMHV2_RBPD:
    here->HSMHV2_rbpd = value->rValue;
    here->HSMHV2_rbpd_Given = TRUE;
    break;
  case  HSMHV2_RBPS:
    here->HSMHV2_rbps = value->rValue;
    here->HSMHV2_rbps_Given = TRUE;
    break;
  case  HSMHV2_RBDB:
    here->HSMHV2_rbdb = value->rValue;
    here->HSMHV2_rbdb_Given = TRUE;
    break;
  case  HSMHV2_RBSB:
    here->HSMHV2_rbsb = value->rValue;
    here->HSMHV2_rbsb_Given = TRUE;
    break;
  case  HSMHV2_CORG: 
    here->HSMHV2_corg = value->iValue;
    here->HSMHV2_corg_Given = TRUE;
    break;
  case  HSMHV2_NGCON:
    here->HSMHV2_ngcon = value->rValue;
    here->HSMHV2_ngcon_Given = TRUE;
    break;
  case  HSMHV2_XGW:
    here->HSMHV2_xgw = value->rValue;
    here->HSMHV2_xgw_Given = TRUE;
    break;
  case  HSMHV2_XGL:
    here->HSMHV2_xgl = value->rValue;
    here->HSMHV2_xgl_Given = TRUE;
    break;
  case  HSMHV2_NF:
    here->HSMHV2_nf = value->rValue;
    here->HSMHV2_nf_Given = TRUE;
    break;
  case  HSMHV2_SA:
    here->HSMHV2_sa = value->rValue;
    here->HSMHV2_sa_Given = TRUE;
    break;
  case  HSMHV2_SB:
    here->HSMHV2_sb = value->rValue;
    here->HSMHV2_sb_Given = TRUE;
    break;
  case  HSMHV2_SD:
    here->HSMHV2_sd = value->rValue;
    here->HSMHV2_sd_Given = TRUE;
    break;
  case  HSMHV2_NSUBCDFM:
    here->HSMHV2_nsubcdfm = value->rValue;
    here->HSMHV2_nsubcdfm_Given = TRUE;
    break;
  case  HSMHV2_M:
    here->HSMHV2_m = value->rValue;
    here->HSMHV2_m_Given = TRUE;
    break;
  case  HSMHV2_SUBLD1:
    here->HSMHV2_subld1 = value->rValue;
    here->HSMHV2_subld1_Given = TRUE;
    break;
  case  HSMHV2_SUBLD2:
    here->HSMHV2_subld2 = value->rValue;
    here->HSMHV2_subld2_Given = TRUE;
    break;
  case  HSMHV2_LOVER:
    here->HSMHV2_lover = value->rValue;
    here->HSMHV2_lover_Given = TRUE;
    break;
  case  HSMHV2_LOVERS:
    here->HSMHV2_lovers = value->rValue;
    here->HSMHV2_lovers_Given = TRUE;
    break;
  case  HSMHV2_LOVERLD:
    here->HSMHV2_loverld = value->rValue;
    here->HSMHV2_loverld_Given = TRUE;
    break;
  case  HSMHV2_LDRIFT1:
    here->HSMHV2_ldrift1 = value->rValue;
    here->HSMHV2_ldrift1_Given = TRUE;
    break;
  case  HSMHV2_LDRIFT2:
    here->HSMHV2_ldrift2 = value->rValue;
    here->HSMHV2_ldrift2_Given = TRUE;
    break;
  case  HSMHV2_LDRIFT1S:
    here->HSMHV2_ldrift1s = value->rValue;
    here->HSMHV2_ldrift1s_Given = TRUE;
    break;
  case  HSMHV2_LDRIFT2S:
    here->HSMHV2_ldrift2s = value->rValue;
    here->HSMHV2_ldrift2s_Given = TRUE;
    break;
  default:
    return(E_BADPARM);
  }
  return(OK);
}
