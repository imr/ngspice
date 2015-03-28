/***********************************************************************

 HiSIM (Hiroshima University STARC IGFET Model)
 Copyright (C) 2012 Hiroshima University & STARC

 MODEL NAME : HiSIM_HV 
 ( VERSION : 1  SUBVERSION : 2  REVISION : 4 )
 Model Parameter VERSION : 1.23
 FILE : hsmhvask.c

 DATE : 2013.04.30

 released by 
                Hiroshima University &
                Semiconductor Technology Academic Research Center (STARC)
***********************************************************************/

#include "ngspice/ngspice.h"
#include "ngspice/ifsim.h"
#include "ngspice/cktdefs.h"
#include "ngspice/devdefs.h"
#include "hsmhvdef.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

int HSMHVask(
     CKTcircuit *ckt,
     GENinstance *inst,
     int which,
     IFvalue *value,
     IFvalue *select)
{
  HSMHVinstance *here = (HSMHVinstance*)inst;

  int flg_nqs ;
  double cggb_nqs, cgdb_nqs, cgsb_nqs, cdgb_nqs, cddb_nqs, cdsb_nqs, cbgb_nqs, cbdb_nqs, cbsb_nqs ;
  double Qi_nqs, dQi_nqs_dVds, dQi_nqs_dVgs, dQi_nqs_dVbs,
                 dQb_nqs_dVds, dQb_nqs_dVgs, dQb_nqs_dVbs ;
  double Qdrat,  dQdrat_dVds,  dQdrat_dVgs,  dQdrat_dVbs,
                 dQi_dVds,     dQi_dVgs,     dQi_dVbs,
                 dQbulk_dVds,  dQbulk_dVgs,  dQbulk_dVbs ;
  double         dQd_nqs_dVds, dQd_nqs_dVgs, dQd_nqs_dVbs, dQd_nqs_dQi_nqs ;
  double                                                   dQg_nqs_dQi_nqs, dQg_nqs_dQb_nqs ;

  NG_IGNORE(select);

  here->HSMHV_csdo = - (here->HSMHV_cddo + here->HSMHV_cgdo + here->HSMHV_cbdo) ;
  here->HSMHV_csgo = - (here->HSMHV_cdgo + here->HSMHV_cggo + here->HSMHV_cbgo) ;
  here->HSMHV_csbo = - (here->HSMHV_cdbo + here->HSMHV_cgbo + here->HSMHV_cbbo) ;

  here->HSMHV_cdso = - (here->HSMHV_cddo + here->HSMHV_cdgo + here->HSMHV_cdbo) ;
  here->HSMHV_cgso = - (here->HSMHV_cgdo + here->HSMHV_cggo + here->HSMHV_cgbo) ;
  here->HSMHV_csso = - (here->HSMHV_csdo + here->HSMHV_csgo + here->HSMHV_csbo) ;

  /* NQS? */
  if (here->HSMHVQIqiPtr == NULL) {
    flg_nqs = 0 ;
  } else {
    flg_nqs = 1 ;
  }
  /* printf("HSMHVask: flg_nqs = %d\n", flg_nqs) ; */

  if (flg_nqs) { /* collect data for NQS case (DC operating point only!) */
    Qi_nqs      = *(ckt->CKTstate0 + here->HSMHVqi_nqs) ;
    if ( here->HSMHV_mode > 0 ) { /* forward mode */
      Qdrat       = here->HSMHV_Xd         ;
      dQdrat_dVds = here->HSMHV_Xd_dVdsi   ;
      dQdrat_dVgs = here->HSMHV_Xd_dVgsi   ;
      dQdrat_dVbs = here->HSMHV_Xd_dVbsi   ;
      dQi_dVds    = here->HSMHV_Qi_dVdsi   ;
      dQi_dVgs    = here->HSMHV_Qi_dVgsi   ;
      dQi_dVbs    = here->HSMHV_Qi_dVbsi   ;
      dQbulk_dVds = here->HSMHV_Qbulk_dVdsi ;
      dQbulk_dVgs = here->HSMHV_Qbulk_dVgsi ;
      dQbulk_dVbs = here->HSMHV_Qbulk_dVbsi ;
    } else { /* reverse mode */
      Qdrat       =   1.0 - here->HSMHV_Xd         ;
      dQdrat_dVds = +(here->HSMHV_Xd_dVdsi + here->HSMHV_Xd_dVgsi + here->HSMHV_Xd_dVbsi) ;
      dQdrat_dVgs = - here->HSMHV_Xd_dVgsi   ;
      dQdrat_dVbs = - here->HSMHV_Xd_dVbsi   ;
      dQi_dVds    = -(here->HSMHV_Qi_dVdsi + here->HSMHV_Qi_dVgsi + here->HSMHV_Qi_dVbsi) ;
      dQi_dVgs    =   here->HSMHV_Qi_dVgsi   ;
      dQi_dVbs    =   here->HSMHV_Qi_dVbsi   ;
      dQbulk_dVds = -(here->HSMHV_Qbulk_dVdsi + here->HSMHV_Qbulk_dVgsi + here->HSMHV_Qbulk_dVbsi) ;
      dQbulk_dVgs =   here->HSMHV_Qbulk_dVgsi ;
      dQbulk_dVbs =   here->HSMHV_Qbulk_dVbsi ;
    }
    /* from Qg_nqs = - Qi_nqs - Qb_nqs: */
    dQg_nqs_dQi_nqs = - 1.0 ;
    dQg_nqs_dQb_nqs = - 1.0 ;
    /* from Qd_nqs = Qi_nqs * Qdrat: */
    dQd_nqs_dVds    = Qi_nqs * dQdrat_dVds ;
    dQd_nqs_dVgs    = Qi_nqs * dQdrat_dVgs ;
    dQd_nqs_dVbs    = Qi_nqs * dQdrat_dVbs ;
    dQd_nqs_dQi_nqs = Qdrat ;

    /* by implicit differentiation of the NQS equations (DC operating point only!): */
    dQi_nqs_dVds = dQi_dVds ;
    dQi_nqs_dVgs = dQi_dVgs ; 
    dQi_nqs_dVbs = dQi_dVbs ;
    dQb_nqs_dVds = dQbulk_dVds ; 
    dQb_nqs_dVgs = dQbulk_dVgs ; 
    dQb_nqs_dVbs = dQbulk_dVbs ;

    cggb_nqs =   dQg_nqs_dQi_nqs * dQi_nqs_dVgs + dQg_nqs_dQb_nqs * dQb_nqs_dVgs ;
    cgdb_nqs =   dQg_nqs_dQi_nqs * dQi_nqs_dVds + dQg_nqs_dQb_nqs * dQb_nqs_dVds ;
    cgsb_nqs = - dQg_nqs_dQi_nqs * (dQi_nqs_dVds + dQi_nqs_dVgs + dQi_nqs_dVbs)
                     - dQg_nqs_dQb_nqs * (dQb_nqs_dVds + dQb_nqs_dVgs + dQb_nqs_dVbs) ;
    cdgb_nqs =   dQd_nqs_dVgs + dQd_nqs_dQi_nqs * dQi_nqs_dVgs ; 
    cddb_nqs =   dQd_nqs_dVds + dQd_nqs_dQi_nqs * dQi_nqs_dVds ; 
    cdsb_nqs = -(dQd_nqs_dVds + dQd_nqs_dVgs + dQd_nqs_dVbs) - dQd_nqs_dQi_nqs * (dQi_nqs_dVds + dQi_nqs_dVgs + dQi_nqs_dVbs) ; 
    cbgb_nqs =   dQb_nqs_dVgs ;
    cbdb_nqs =   dQb_nqs_dVds ;
    cbsb_nqs = -(dQb_nqs_dVds + dQb_nqs_dVgs + dQb_nqs_dVbs) ;
  } else { /* QS case */
    cggb_nqs = cgdb_nqs = cgsb_nqs = cdgb_nqs = cddb_nqs = cdsb_nqs = cbgb_nqs = cbdb_nqs = cbsb_nqs = 0.0 ;
  }


  switch (which) {
  case HSMHV_COSELFHEAT:
    value->iValue = here->HSMHV_coselfheat;
    return(OK);
  case HSMHV_COSUBNODE:
    value->iValue = here->HSMHV_cosubnode;
    return(OK);
  case HSMHV_L:
    value->rValue = here->HSMHV_l;
    return(OK);
  case HSMHV_W:
    value->rValue = here->HSMHV_w;
    return(OK);
  case HSMHV_AS:
    value->rValue = here->HSMHV_as;
    return(OK);
  case HSMHV_AD:
    value->rValue = here->HSMHV_ad;
    return(OK);
  case HSMHV_PS:
    value->rValue = here->HSMHV_ps;
    return(OK);
  case HSMHV_PD:
    value->rValue = here->HSMHV_pd;
    return(OK);
  case HSMHV_NRS:
    value->rValue = here->HSMHV_nrs;
    return(OK);
  case HSMHV_NRD:
    value->rValue = here->HSMHV_nrd;
    return(OK);
  case HSMHV_DTEMP:
    value->rValue = here->HSMHV_dtemp;
    return(OK);
  case HSMHV_OFF:
    value->iValue = here->HSMHV_off;
    return(OK);
  case HSMHV_IC_VBS:
    value->rValue = here->HSMHV_icVBS;
    return(OK);
  case HSMHV_IC_VDS:
    value->rValue = here->HSMHV_icVDS;
    return(OK);
  case HSMHV_IC_VGS:
    value->rValue = here->HSMHV_icVGS;
    return(OK);
  case HSMHV_DNODE:
    value->iValue = here->HSMHVdNode;
    return(OK);
  case HSMHV_GNODE:
    value->iValue = here->HSMHVgNode;
    return(OK);
  case HSMHV_SNODE:
    value->iValue = here->HSMHVsNode;
    return(OK);
  case HSMHV_BNODE:
    value->iValue = here->HSMHVbNode;
    return(OK);
  case HSMHV_DNODEPRIME:
    value->iValue = here->HSMHVdNodePrime;
    return(OK);
  case HSMHV_SNODEPRIME:
    value->iValue = here->HSMHVsNodePrime;
    return(OK);
  case HSMHV_SOURCECONDUCT:
    value->rValue = here->HSMHVsourceConductance;
    return(OK);
  case HSMHV_DRAINCONDUCT:
    value->rValue = here->HSMHVdrainConductance;
    return(OK);
  case HSMHV_VBD:
    value->rValue = *(ckt->CKTstate0 + here->HSMHVvbd);
    return(OK);
  case HSMHV_VBS:
    value->rValue = *(ckt->CKTstate0 + here->HSMHVvbs);
    return(OK);
  case HSMHV_VGS:
    value->rValue = *(ckt->CKTstate0 + here->HSMHVvgs);
    return(OK);
  case HSMHV_VDS:
    value->rValue = *(ckt->CKTstate0 + here->HSMHVvds);
    return(OK);
  case HSMHV_CD:
    value->rValue = here->HSMHV_ids;
    return(OK);
  case HSMHV_ISUB:
    value->rValue = here->HSMHV_isub;
    return(OK);
  case HSMHV_IGIDL:
    value->rValue = here->HSMHV_igidl;
    return(OK);
  case HSMHV_IGISL:
    value->rValue = here->HSMHV_igisl;
    return(OK);
  case HSMHV_IGD:
    value->rValue = here->HSMHV_igd;
    return(OK);
  case HSMHV_IGS:
    value->rValue = here->HSMHV_igs;
    return(OK);
  case HSMHV_IGB:
    value->rValue = here->HSMHV_igb;
    return(OK);
  case HSMHV_CBS:
    value->rValue = here->HSMHV_ibs;
    return(OK);
  case HSMHV_CBD:
    value->rValue = here->HSMHV_ibd;
    return(OK);
  case HSMHV_GM:
    value->rValue = here->HSMHV_dIds_dVgsi;
    return(OK);
  case HSMHV_GMT:
    value->rValue = here->HSMHV_dIds_dTi;
    return(OK);
  case HSMHV_GDS:
    value->rValue = here->HSMHV_dIds_dVdsi;
    return(OK);
  case HSMHV_GMBS:
    value->rValue = here->HSMHV_dIds_dVbsi;
    return(OK);
  case HSMHV_GBD:
    value->rValue = here->HSMHV_gbd;
    return(OK);
  case HSMHV_GBS:
    value->rValue = here->HSMHV_gbs;
    return(OK);
  case HSMHV_QB:
    value->rValue = *(ckt->CKTstate0 + here->HSMHVqb); 
    return(OK);
  case HSMHV_CQB:
    value->rValue = *(ckt->CKTstate0 + here->HSMHVcqb); 
    return(OK);
  case HSMHV_QG:
    value->rValue = *(ckt->CKTstate0 + here->HSMHVqg); 
    return(OK);
  case HSMHV_CQG:
    value->rValue = *(ckt->CKTstate0 + here->HSMHVcqg); 
    return(OK);
  case HSMHV_QD:
    value->rValue = *(ckt->CKTstate0 + here->HSMHVqd); 
    return(OK);
  case HSMHV_CQD:
    value->rValue = *(ckt->CKTstate0 + here->HSMHVcqd); 
    return(OK);
  case HSMHV_CGG:
    value->rValue = here->HSMHV_dQg_dVgsi - here->HSMHV_cggo;
    if (flg_nqs) value->rValue += cggb_nqs;
    return(OK);
  case HSMHV_CGD:
    value->rValue = (here->HSMHV_mode > 0) ? here->HSMHV_dQg_dVdsi - here->HSMHV_cgdo 
                                           : - (here->HSMHV_dQg_dVdsi + here->HSMHV_dQg_dVgsi + here->HSMHV_dQg_dVbsi)
                                             - here->HSMHV_cgso;
    if (flg_nqs) value->rValue += cgdb_nqs;
    return(OK);
  case HSMHV_CGS:
    value->rValue = (here->HSMHV_mode > 0) ? - (here->HSMHV_dQg_dVdsi + here->HSMHV_dQg_dVgsi + here->HSMHV_dQg_dVbsi)
                                             - here->HSMHV_cgso
                                           : here->HSMHV_dQg_dVdsi - here->HSMHV_cgdo;
    if (flg_nqs) value->rValue += cgsb_nqs;
    return(OK);
  case HSMHV_CDG:
    value->rValue = (here->HSMHV_mode > 0) ? here->HSMHV_dQdi_dVgsi - here->HSMHV_cdgo
                                           : here->HSMHV_dQsi_dVgsi - here->HSMHV_csgo;
    if (flg_nqs) value->rValue += cdgb_nqs; 
    return(OK);
  case HSMHV_CDD:
    value->rValue = (here->HSMHV_mode > 0) ? here->HSMHV_dQdi_dVdsi - here->HSMHV_cddo
                                           : - (here->HSMHV_dQsi_dVdsi + here->HSMHV_dQsi_dVgsi + here->HSMHV_dQsi_dVbsi)
                                             - here->HSMHV_csso;
    if (flg_nqs) value->rValue += cddb_nqs; 
    return(OK);
  case HSMHV_CDS:
    value->rValue = (here->HSMHV_mode > 0) ? - (here->HSMHV_dQdi_dVdsi + here->HSMHV_dQdi_dVgsi + here->HSMHV_dQdi_dVbsi)
                                             - here->HSMHV_cdso
                                           : here->HSMHV_dQsi_dVdsi - here->HSMHV_csdo;
    if (flg_nqs) value->rValue += cdsb_nqs; 
    return(OK);
  case HSMHV_CBG:
    value->rValue = here->HSMHV_dQb_dVgsi - here->HSMHV_cbgo;
    if (flg_nqs) value->rValue += cbgb_nqs;
    return(OK);
  case HSMHV_CBDB:
    value->rValue = (here->HSMHV_mode > 0) ? here->HSMHV_dQb_dVdsi - here->HSMHV_cbdo
                                           : - (here->HSMHV_dQb_dVdsi + here->HSMHV_dQb_dVgsi + here->HSMHV_dQb_dVbsi)
                                             + (here->HSMHV_cbdo+here->HSMHV_cbgo+here->HSMHV_cbbo);
    if (flg_nqs) value->rValue += cbdb_nqs;
    return(OK);
  case HSMHV_CBSB:
    value->rValue = (here->HSMHV_mode > 0) ? - (here->HSMHV_dQb_dVdsi + here->HSMHV_dQb_dVgsi + here->HSMHV_dQb_dVbsi)
                                             + (here->HSMHV_cbdo + here->HSMHV_cbgo + here->HSMHV_cbbo)
                                           : here->HSMHV_dQb_dVdsi - here->HSMHV_cbdo;
    if (flg_nqs) value->rValue += cbsb_nqs;
    return(OK);
  case HSMHV_CGDO:
    value->rValue = (here->HSMHV_mode > 0) ? here->HSMHV_cgdo : here->HSMHV_cgso;

    return(OK);
  case HSMHV_CGSO:
    value->rValue = (here->HSMHV_mode > 0) ? here->HSMHV_cgso : here->HSMHV_cgdo;
    return(OK);
  case HSMHV_CGBO:
    value->rValue = here->HSMHV_cgbo;
    return(OK);
  case HSMHV_CAPBD:
    value->rValue = here->HSMHV_capbd;
    return(OK);
  case HSMHV_CAPBS:
    value->rValue = here->HSMHV_capbs;
    return(OK);
  case HSMHV_VON:
    value->rValue = here->HSMHV_von; 
    return(OK);
  case HSMHV_VDSAT:
    value->rValue = here->HSMHV_vdsat; 
    return(OK);
  case HSMHV_QBS:
    value->rValue = *(ckt->CKTstate0 + here->HSMHVqbs); 
    return(OK);
  case HSMHV_QBD:
    value->rValue = *(ckt->CKTstate0 + here->HSMHVqbd); 
    return(OK);
  case  HSMHV_CORBNET: 
    value->iValue = here->HSMHV_corbnet;
    return(OK);
  case  HSMHV_RBPB:
    value->rValue = here->HSMHV_rbpb;
    return (OK);
  case  HSMHV_RBPD:
    value->rValue = here->HSMHV_rbpd;
    return(OK);
  case  HSMHV_RBPS:
    value->rValue = here->HSMHV_rbps;
    return(OK);
  case  HSMHV_RBDB:
    value->rValue = here->HSMHV_rbdb;
    return(OK);
  case  HSMHV_RBSB:
    value->rValue = here->HSMHV_rbsb;
    return(OK);
  case  HSMHV_CORG: 
    value->iValue = here->HSMHV_corg;
    return(OK);
  case  HSMHV_NGCON:
    value->rValue = here->HSMHV_ngcon;
    return(OK);
  case  HSMHV_XGW:
    value->rValue = here->HSMHV_xgw;
    return(OK);
  case  HSMHV_XGL:
    value->rValue = here->HSMHV_xgl;
    return(OK);
  case  HSMHV_NF:
    value->rValue = here->HSMHV_nf;
    return(OK);
  case  HSMHV_SA:
    value->rValue = here->HSMHV_sa;
    return(OK);
  case  HSMHV_SB:
    value->rValue = here->HSMHV_sb;
    return(OK);
  case  HSMHV_SD:
    value->rValue = here->HSMHV_sd;
    return(OK);
  case  HSMHV_NSUBCDFM:
    value->rValue = here->HSMHV_nsubcdfm;
    return(OK);
  case  HSMHV_M:
    value->rValue = here->HSMHV_m;
    return(OK);
  case  HSMHV_SUBLD1:
    value->rValue = here->HSMHV_subld1;
    return(OK);
  case  HSMHV_SUBLD2:
    value->rValue = here->HSMHV_subld2;
    return(OK);
  case  HSMHV_LOVER:
    value->rValue = here->HSMHV_lover;
    return(OK);
  case  HSMHV_LOVERS:
    value->rValue = here->HSMHV_lovers;
    return(OK);
  case  HSMHV_LOVERLD:
    value->rValue = here->HSMHV_loverld;
    return(OK);
  case  HSMHV_LDRIFT1:
    value->rValue = here->HSMHV_ldrift1;
    return(OK);
  case  HSMHV_LDRIFT2:
    value->rValue = here->HSMHV_ldrift2;
    return(OK);
  case  HSMHV_LDRIFT1S:
    value->rValue = here->HSMHV_ldrift1s;
    return(OK);
  case  HSMHV_LDRIFT2S:
    value->rValue = here->HSMHV_ldrift2s;
    return(OK);
  default:
    return(E_BADPARM);
  }
  /* NOTREACHED */
}
