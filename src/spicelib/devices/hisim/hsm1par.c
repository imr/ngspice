/***********************************************************************
 HiSIM (Hiroshima University STARC IGFET Model)
 Copyright (C) 2003 STARC

 VERSION : HiSIM 1.2.0
 FILE : hsm1par.c of HiSIM 1.2.0

 April 9, 2003 : released by STARC Physical Design Group
***********************************************************************/

#include "ngspice.h"
#include "ifsim.h"
#include "hsm1def.h"
#include "sperror.h"
#include "suffix.h"

int 
HSM1param(int param, IFvalue *value, GENinstance *inst, IFvalue *select)
{
  HSM1instance *here = (HSM1instance*)inst;
  switch (param) {
  case HSM1_W:
    here->HSM1_w = value->rValue;
    here->HSM1_w_Given = TRUE;
    break;
  case HSM1_L:
    here->HSM1_l = value->rValue;
    here->HSM1_l_Given = TRUE;
    break;
  case HSM1_M:
   here->HSM1_m = value->rValue;
   here->HSM1_m_Given = TRUE;
   break;   
  case HSM1_AS:
    here->HSM1_as = value->rValue;
    here->HSM1_as_Given = TRUE;
    break;
  case HSM1_AD:
    here->HSM1_ad = value->rValue;
    here->HSM1_ad_Given = TRUE;
    break;
  case HSM1_PS:
    here->HSM1_ps = value->rValue;
    here->HSM1_ps_Given = TRUE;
    break;
  case HSM1_PD:
    here->HSM1_pd = value->rValue;
    here->HSM1_pd_Given = TRUE;
    break;
  case HSM1_NRS:
    here->HSM1_nrs = value->rValue;
    here->HSM1_nrs_Given = TRUE;
    break;
  case HSM1_NRD:
    here->HSM1_nrd = value->rValue;
    here->HSM1_nrd_Given = TRUE;
    break;
  case HSM1_TEMP:
    here->HSM1_temp = value->rValue;
    here->HSM1_temp_Given = TRUE;
    break;
  case HSM1_DTEMP:
    here->HSM1_dtemp = value->rValue;
    here->HSM1_dtemp_Given = TRUE;
    break;
  case HSM1_OFF:
    here->HSM1_off = value->iValue;
    break;
  case HSM1_IC_VBS:
    here->HSM1_icVBS = value->rValue;
    here->HSM1_icVBS_Given = TRUE;
    break;
  case HSM1_IC_VDS:
    here->HSM1_icVDS = value->rValue;
    here->HSM1_icVDS_Given = TRUE;
    break;
  case HSM1_IC_VGS:
    here->HSM1_icVGS = value->rValue;
    here->HSM1_icVGS_Given = TRUE;
    break;
  case HSM1_IC:
    switch (value->v.numValue) {
    case 3:
      here->HSM1_icVBS = *(value->v.vec.rVec + 2);
      here->HSM1_icVBS_Given = TRUE;
    case 2:
      here->HSM1_icVGS = *(value->v.vec.rVec + 1);
      here->HSM1_icVGS_Given = TRUE;
    case 1:
      here->HSM1_icVDS = *(value->v.vec.rVec);
      here->HSM1_icVDS_Given = TRUE;
      break;
    default:
      return(E_BADPARM);
    }
    break;
  default:
    return(E_BADPARM);
  }
  return(OK);
}
