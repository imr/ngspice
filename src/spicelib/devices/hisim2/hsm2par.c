/***********************************************************************

 HiSIM (Hiroshima University STARC IGFET Model)
 Copyright (C) 2012 Hiroshima University & STARC

 VERSION : HiSIM 2.6.1 
 FILE : hsm2par.c

 date : 2012.4.6

 released by 
                Hiroshima University &
                Semiconductor Technology Academic Research Center (STARC)
***********************************************************************/

#include "ngspice/ngspice.h"
#include "ngspice/ifsim.h"
#include "hsm2def.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

int HSM2param(
     int param,
     IFvalue *value,
     GENinstance *inst,
     IFvalue *select)
{
  HSM2instance *here = (HSM2instance*)inst;

  NG_IGNORE(select);

  switch (param) {
  case HSM2_W:
    here->HSM2_w = value->rValue;
    here->HSM2_w_Given = TRUE;
    break;
  case HSM2_L:
    here->HSM2_l = value->rValue;
    here->HSM2_l_Given = TRUE;
    break;
  case HSM2_AS:
    here->HSM2_as = value->rValue;
    here->HSM2_as_Given = TRUE;
    break;
  case HSM2_AD:
    here->HSM2_ad = value->rValue;
    here->HSM2_ad_Given = TRUE;
    break;
  case HSM2_PS:
    here->HSM2_ps = value->rValue;
    here->HSM2_ps_Given = TRUE;
    break;
  case HSM2_PD:
    here->HSM2_pd = value->rValue;
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
    switch (value->v.numValue) {
    case 3:
      here->HSM2_icVBS = *(value->v.vec.rVec + 2);
      here->HSM2_icVBS_Given = TRUE;
    case 2:
      here->HSM2_icVGS = *(value->v.vec.rVec + 1);
      here->HSM2_icVGS_Given = TRUE;
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
