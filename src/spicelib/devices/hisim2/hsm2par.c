/***********************************************************************

 HiSIM (Hiroshima University STARC IGFET Model)
 Copyright (C) 2014 Hiroshima University & STARC

 MODEL NAME : HiSIM
 ( VERSION : 2  SUBVERSION : 8  REVISION : 0 )
 
 FILE : hsm2par.c

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
#include "ngspice/ifsim.h"
#include "hsm2def.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"
#include "ngspice/fteext.h"

int HSM2param(
     int param,
     IFvalue *value,
     GENinstance *inst,
     IFvalue *select)
{
  double scale;

  HSM2instance *here = (HSM2instance*)inst;

  NG_IGNORE(select);

  if (!cp_getvar("scale", CP_REAL, &scale, 0))
      scale = 1;

  switch (param) {
  case HSM2_W:
    here->HSM2_w = value->rValue * scale;
    here->HSM2_w_Given = TRUE;
    break;
  case HSM2_L:
    here->HSM2_l = value->rValue * scale;
    here->HSM2_l_Given = TRUE;
    break;
  case HSM2_AS:
    here->HSM2_as = value->rValue * scale * scale;
    here->HSM2_as_Given = TRUE;
    break;
  case HSM2_AD:
    here->HSM2_ad = value->rValue * scale * scale;
    here->HSM2_ad_Given = TRUE;
    break;
  case HSM2_PS:
    here->HSM2_ps = value->rValue * scale;
    here->HSM2_ps_Given = TRUE;
    break;
  case HSM2_PD:
    here->HSM2_pd = value->rValue * scale;
    here->HSM2_pd_Given = TRUE;
    break;
  case HSM2_NRS:
    here->HSM2_nrs = value->rValue;
    here->HSM2_nrs_Given = TRUE;
    break;
  case HSM2_NRD:
    here->HSM2_nrd = value->rValue;
    here->HSM2_nrd_Given = TRUE;
    break;
  case HSM2_TEMP:
    here->HSM2_temp = value->rValue;
    here->HSM2_temp_Given = TRUE;
    break;
  case HSM2_DTEMP:
    here->HSM2_dtemp = value->rValue;
    here->HSM2_dtemp_Given = TRUE;
    break;
  case HSM2_OFF:
    here->HSM2_off = value->iValue;
    break;
  case HSM2_IC_VBS:
    here->HSM2_icVBS = value->rValue;
    here->HSM2_icVBS_Given = TRUE;
    break;
  case HSM2_IC_VDS:
    here->HSM2_icVDS = value->rValue;
    here->HSM2_icVDS_Given = TRUE;
    break;
  case HSM2_IC_VGS:
    here->HSM2_icVGS = value->rValue;
    here->HSM2_icVGS_Given = TRUE;
    break;
  case HSM2_IC:
    /* FALLTHROUGH added to suppress GCC warning due to
     * -Wimplicit-fallthrough flag */
    switch (value->v.numValue) {
    case 3:
      here->HSM2_icVBS = *(value->v.vec.rVec + 2);
      here->HSM2_icVBS_Given = TRUE;
        /* FALLTHROUGH */
    case 2:
      here->HSM2_icVGS = *(value->v.vec.rVec + 1);
      here->HSM2_icVGS_Given = TRUE;
        /* FALLTHROUGH */
    case 1:
      here->HSM2_icVDS = *(value->v.vec.rVec);
      here->HSM2_icVDS_Given = TRUE;
      break;
    default:
      return(E_BADPARM);
    }
    break;
  case  HSM2_CORBNET: 
    here->HSM2_corbnet = value->iValue;
    here->HSM2_corbnet_Given = TRUE;
    break;
  case  HSM2_RBPB:
    here->HSM2_rbpb = value->rValue;
    here->HSM2_rbpb_Given = TRUE;
    break;
  case  HSM2_RBPD:
    here->HSM2_rbpd = value->rValue;
    here->HSM2_rbpd_Given = TRUE;
    break;
  case  HSM2_RBPS:
    here->HSM2_rbps = value->rValue;
    here->HSM2_rbps_Given = TRUE;
    break;
  case  HSM2_RBDB:
    here->HSM2_rbdb = value->rValue;
    here->HSM2_rbdb_Given = TRUE;
    break;
  case  HSM2_RBSB:
    here->HSM2_rbsb = value->rValue;
    here->HSM2_rbsb_Given = TRUE;
    break;
  case  HSM2_CORG: 
    here->HSM2_corg = value->iValue;
    here->HSM2_corg_Given = TRUE;
    break;
/*   case  HSM2_RSHG: */
/*     here->HSM2_rshg = value->rValue; */
/*     here->HSM2_rshg_Given = TRUE; */
/*     break; */
  case  HSM2_NGCON:
    here->HSM2_ngcon = value->rValue;
    here->HSM2_ngcon_Given = TRUE;
    break;
  case  HSM2_XGW:
    here->HSM2_xgw = value->rValue;
    here->HSM2_xgw_Given = TRUE;
    break;
  case  HSM2_XGL:
    here->HSM2_xgl = value->rValue;
    here->HSM2_xgl_Given = TRUE;
    break;
  case  HSM2_NF:
    here->HSM2_nf = value->rValue;
    here->HSM2_nf_Given = TRUE;
    break;
  case  HSM2_SA:
    here->HSM2_sa = value->rValue;
    here->HSM2_sa_Given = TRUE;
    break;
  case  HSM2_SB:
    here->HSM2_sb = value->rValue;
    here->HSM2_sb_Given = TRUE;
    break;
  case  HSM2_SD:
    here->HSM2_sd = value->rValue;
    here->HSM2_sd_Given = TRUE;
    break;
  case  HSM2_NSUBCDFM:
    here->HSM2_nsubcdfm = value->rValue;
    here->HSM2_nsubcdfm_Given = TRUE;
    break;
  case  HSM2_MPHDFM:
    here->HSM2_mphdfm = value->rValue;
    here->HSM2_mphdfm_Given = TRUE;
    break;
  case  HSM2_M:
    here->HSM2_m = value->rValue;
    here->HSM2_m_Given = TRUE;
    break;

/* WPE */
  case HSM2_SCA:
    here->HSM2_sca = value->rValue;
	here->HSM2_sca_Given = TRUE;
	break;
  case HSM2_SCB:
    here->HSM2_scb = value->rValue;
	here->HSM2_scb_Given = TRUE;
	break;
  case HSM2_SCC:
    here->HSM2_scc= value->rValue;
	here->HSM2_scc_Given = TRUE;
	break;
  default:
    return(E_BADPARM);
  }
  return(OK);
}
