/***********************************************************************
 HiSIM (Hiroshima University STARC IGFET Model)
 Copyright (C) 2003 STARC

 VERSION : HiSIM 1.2.0
 FILE : hsm1ask.c of HiSIM 1.2.0

 April 9, 2003 : released by STARC Physical Design Group
***********************************************************************/

#include "ngspice.h"
#include "ifsim.h"
#include "cktdefs.h"
#include "devdefs.h"
#include "hsm1def.h"
#include "sperror.h"
#include "suffix.h"

int 
HSM1ask(CKTcircuit *ckt, GENinstance *inst, int which, 
        IFvalue *value, IFvalue *select)
{
  HSM1instance *here = (HSM1instance*)inst;

  switch (which) {
  case HSM1_L:
    value->rValue = here->HSM1_l;
    return(OK);
  case HSM1_W:
    value->rValue = here->HSM1_w;
    return(OK);
  case HSM1_M:
    value->rValue = here->HSM1_m;
    return(OK);    
  case HSM1_AS:
    value->rValue = here->HSM1_as;
    return(OK);
  case HSM1_AD:
    value->rValue = here->HSM1_ad;
    return(OK);
  case HSM1_PS:
    value->rValue = here->HSM1_ps;
    return(OK);
  case HSM1_PD:
    value->rValue = here->HSM1_pd;
    return(OK);
  case HSM1_NRS:
    value->rValue = here->HSM1_nrs;
    return(OK);
  case HSM1_NRD:
    value->rValue = here->HSM1_nrd;
    return(OK);
  case HSM1_TEMP:
    value->rValue = here->HSM1_temp;
    return(OK);
  case HSM1_DTEMP:
    value->rValue = here->HSM1_dtemp;
    return(OK);
  case HSM1_OFF:
    value->iValue = here->HSM1_off;
    return(OK);
  case HSM1_IC_VBS:
    value->rValue = here->HSM1_icVBS;
    return(OK);
  case HSM1_IC_VDS:
    value->rValue = here->HSM1_icVDS;
    return(OK);
  case HSM1_IC_VGS:
    value->rValue = here->HSM1_icVGS;
    return(OK);
  case HSM1_DNODE:
    value->iValue = here->HSM1dNode;
    return(OK);
  case HSM1_GNODE:
    value->iValue = here->HSM1gNode;
    return(OK);
  case HSM1_SNODE:
    value->iValue = here->HSM1sNode;
    return(OK);
  case HSM1_BNODE:
    value->iValue = here->HSM1bNode;
    return(OK);
  case HSM1_DNODEPRIME:
    value->iValue = here->HSM1dNodePrime;
    return(OK);
  case HSM1_SNODEPRIME:
    value->iValue = here->HSM1sNodePrime;
    return(OK);
  case HSM1_SOURCECONDUCT:
    value->rValue = here->HSM1sourceConductance;
    value->rValue *= here->HSM1_m;
    return(OK);
  case HSM1_DRAINCONDUCT:
    value->rValue = here->HSM1drainConductance;
    value->rValue *= here->HSM1_m;
    return(OK);
  case HSM1_VBD:
    value->rValue = *(ckt->CKTstate0 + here->HSM1vbd);
    return(OK);
  case HSM1_VBS:
    value->rValue = *(ckt->CKTstate0 + here->HSM1vbs);
    return(OK);
  case HSM1_VGS:
    value->rValue = *(ckt->CKTstate0 + here->HSM1vgs);
    return(OK);
  case HSM1_VDS:
    value->rValue = *(ckt->CKTstate0 + here->HSM1vds);
    return(OK);
  case HSM1_CD:
    value->rValue = here->HSM1_ids;
    value->rValue *= here->HSM1_m;
    return(OK);
  case HSM1_CBS:
    value->rValue = here->HSM1_ibs;
    value->rValue *= here->HSM1_m;
    return(OK);
  case HSM1_CBD:
    value->rValue = here->HSM1_ibs;
    value->rValue *= here->HSM1_m;
    return(OK);
  case HSM1_GM:
    value->rValue = here->HSM1_gm;
    value->rValue *= here->HSM1_m;
    return(OK);
  case HSM1_GDS:
    value->rValue = here->HSM1_gds;
    value->rValue *= here->HSM1_m;
    return(OK);
  case HSM1_GMBS:
    value->rValue = here->HSM1_gmbs;
    value->rValue *= here->HSM1_m;
    return(OK);
  case HSM1_GBD:
    value->rValue = here->HSM1_gbd;
    value->rValue *= here->HSM1_m;
    return(OK);
  case HSM1_GBS:
    value->rValue = here->HSM1_gbs;
    value->rValue *= here->HSM1_m;
    return(OK);
  case HSM1_QB:
    value->rValue = *(ckt->CKTstate0 + here->HSM1qb); 
    value->rValue *= here->HSM1_m;
    return(OK);
  case HSM1_CQB:
    value->rValue = *(ckt->CKTstate0 + here->HSM1cqb); 
    value->rValue *= here->HSM1_m;
    return(OK);
  case HSM1_QG:
    value->rValue = *(ckt->CKTstate0 + here->HSM1qg); 
    value->rValue *= here->HSM1_m;
    return(OK);
  case HSM1_CQG:
    value->rValue = *(ckt->CKTstate0 + here->HSM1cqg); 
    value->rValue *= here->HSM1_m;
    return(OK);
  case HSM1_QD:
    value->rValue = *(ckt->CKTstate0 + here->HSM1qd); 
    value->rValue *= here->HSM1_m;
    return(OK);
  case HSM1_CQD:
    value->rValue = *(ckt->CKTstate0 + here->HSM1cqd);
    value->rValue *= here->HSM1_m; 
    return(OK);
  case HSM1_CGG:
    value->rValue = here->HSM1_cggb; 
    value->rValue *= here->HSM1_m;
    return(OK);
  case HSM1_CGD:
    value->rValue = here->HSM1_cgdb;
    value->rValue *= here->HSM1_m;
    return(OK);
  case HSM1_CGS:
    value->rValue = here->HSM1_cgsb;
    value->rValue *= here->HSM1_m;
    return(OK);
  case HSM1_CDG:
    value->rValue = here->HSM1_cdgb; 
    value->rValue *= here->HSM1_m;
    return(OK);
  case HSM1_CDD:
    value->rValue = here->HSM1_cddb; 
    value->rValue *= here->HSM1_m;
    return(OK);
  case HSM1_CDS:
    value->rValue = here->HSM1_cdsb; 
    value->rValue *= here->HSM1_m;
    return(OK);
  case HSM1_CBG:
    value->rValue = here->HSM1_cbgb;
    value->rValue *= here->HSM1_m;
    return(OK);
  case HSM1_CBDB:
    value->rValue = here->HSM1_cbdb;
    value->rValue *= here->HSM1_m;
    return(OK);
  case HSM1_CBSB:
    value->rValue = here->HSM1_cbsb;
    value->rValue *= here->HSM1_m;
    return(OK);
  case HSM1_CAPBD:
    value->rValue = here->HSM1_capbd;
    value->rValue *= here->HSM1_m; 
    return(OK);
  case HSM1_CAPBS:
    value->rValue = here->HSM1_capbs;
    value->rValue *= here->HSM1_m;
    return(OK);
  case HSM1_VON:
    value->rValue = here->HSM1_von; 
    return(OK);
  case HSM1_VDSAT:
    value->rValue = here->HSM1_vdsat; 
    return(OK);
  case HSM1_QBS:
    value->rValue = *(ckt->CKTstate0 + here->HSM1qbs); 
    value->rValue *= here->HSM1_m;
    return(OK);
  case HSM1_QBD:
    value->rValue = *(ckt->CKTstate0 + here->HSM1qbd); 
    value->rValue *= here->HSM1_m;
    return(OK);
  default:
    return(E_BADPARM);
  }
  /* NOTREACHED */
}
