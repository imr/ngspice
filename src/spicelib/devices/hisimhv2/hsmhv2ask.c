/***********************************************************************

 HiSIM (Hiroshima University STARC IGFET Model)
 Copyright (C) 2014 Hiroshima University & STARC

 MODEL NAME : HiSIM_HV 
 ( VERSION : 2  SUBVERSION : 2  REVISION : 0 ) 
 Model Parameter 'VERSION' : 2.20
 FILE : hsmhvask.c

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
#include "ngspice/cktdefs.h"
#include "ngspice/devdefs.h"
#include "hsmhv2def.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

int HSMHV2ask(
     CKTcircuit *ckt,
     GENinstance *inst,
     int which,
     IFvalue *value,
     IFvalue *select)
{
  HSMHV2instance *here = (HSMHV2instance*)inst;

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

  here->HSMHV2_csdo = - (here->HSMHV2_cddo + here->HSMHV2_cgdo + here->HSMHV2_cbdo) ;
  here->HSMHV2_csgo = - (here->HSMHV2_cdgo + here->HSMHV2_cggo + here->HSMHV2_cbgo) ;
  here->HSMHV2_csbo = - (here->HSMHV2_cdbo + here->HSMHV2_cgbo + here->HSMHV2_cbbo) ;

  here->HSMHV2_cdso = - (here->HSMHV2_cddo + here->HSMHV2_cdgo + here->HSMHV2_cdbo) ;
  here->HSMHV2_cgso = - (here->HSMHV2_cgdo + here->HSMHV2_cggo + here->HSMHV2_cgbo) ;
  here->HSMHV2_csso = - (here->HSMHV2_csdo + here->HSMHV2_csgo + here->HSMHV2_csbo) ;

  /* NQS? */
  if (here->HSMHV2QIqiPtr == NULL) {
    flg_nqs = 0 ;
  } else {
    flg_nqs = 1 ;
  }
  /* printf("HSMHV2ask: flg_nqs = %d\n", flg_nqs) ; */

  if (flg_nqs) { /* collect data for NQS case (DC operating point only!) */
    Qi_nqs      = *(ckt->CKTstate0 + here->HSMHV2qi_nqs) ;
    if ( here->HSMHV2_mode > 0 ) { /* forward mode */
      Qdrat       = here->HSMHV2_Xd         ;
      dQdrat_dVds = here->HSMHV2_Xd_dVdsi   ;
      dQdrat_dVgs = here->HSMHV2_Xd_dVgsi   ;
      dQdrat_dVbs = here->HSMHV2_Xd_dVbsi   ;
      dQi_dVds    = here->HSMHV2_Qi_dVdsi   ;
      dQi_dVgs    = here->HSMHV2_Qi_dVgsi   ;
      dQi_dVbs    = here->HSMHV2_Qi_dVbsi   ;
      dQbulk_dVds = here->HSMHV2_Qbulk_dVdsi ;
      dQbulk_dVgs = here->HSMHV2_Qbulk_dVgsi ;
      dQbulk_dVbs = here->HSMHV2_Qbulk_dVbsi ;
    } else { /* reverse mode */
      Qdrat       =   1.0 - here->HSMHV2_Xd         ;
      dQdrat_dVds = +(here->HSMHV2_Xd_dVdsi + here->HSMHV2_Xd_dVgsi + here->HSMHV2_Xd_dVbsi) ;
      dQdrat_dVgs = - here->HSMHV2_Xd_dVgsi   ;
      dQdrat_dVbs = - here->HSMHV2_Xd_dVbsi   ;
      dQi_dVds    = -(here->HSMHV2_Qi_dVdsi + here->HSMHV2_Qi_dVgsi + here->HSMHV2_Qi_dVbsi) ;
      dQi_dVgs    =   here->HSMHV2_Qi_dVgsi   ;
      dQi_dVbs    =   here->HSMHV2_Qi_dVbsi   ;
      dQbulk_dVds = -(here->HSMHV2_Qbulk_dVdsi + here->HSMHV2_Qbulk_dVgsi + here->HSMHV2_Qbulk_dVbsi) ;
      dQbulk_dVgs =   here->HSMHV2_Qbulk_dVgsi ;
      dQbulk_dVbs =   here->HSMHV2_Qbulk_dVbsi ;
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
  case HSMHV2_COSELFHEAT:
    value->iValue = here->HSMHV2_coselfheat;
    return(OK);
  case HSMHV2_COSUBNODE:
    value->iValue = here->HSMHV2_cosubnode;
    return(OK);
  case HSMHV2_L:
    value->rValue = here->HSMHV2_l;
    return(OK);
  case HSMHV2_W:
    value->rValue = here->HSMHV2_w;
    return(OK);
  case HSMHV2_AS:
    value->rValue = here->HSMHV2_as;
    return(OK);
  case HSMHV2_AD:
    value->rValue = here->HSMHV2_ad;
    return(OK);
  case HSMHV2_PS:
    value->rValue = here->HSMHV2_ps;
    return(OK);
  case HSMHV2_PD:
    value->rValue = here->HSMHV2_pd;
    return(OK);
  case HSMHV2_NRS:
    value->rValue = here->HSMHV2_nrs;
    return(OK);
  case HSMHV2_NRD:
    value->rValue = here->HSMHV2_nrd;
    return(OK);
  case HSMHV2_DTEMP:
    value->rValue = here->HSMHV2_dtemp;
    return(OK);
  case HSMHV2_OFF:
    value->iValue = here->HSMHV2_off;
    return(OK);
  case HSMHV2_IC_VBS:
    value->rValue = here->HSMHV2_icVBS;
    return(OK);
  case HSMHV2_IC_VDS:
    value->rValue = here->HSMHV2_icVDS;
    return(OK);
  case HSMHV2_IC_VGS:
    value->rValue = here->HSMHV2_icVGS;
    return(OK);
  case HSMHV2_DNODE:
    value->iValue = here->HSMHV2dNode;
    return(OK);
  case HSMHV2_GNODE:
    value->iValue = here->HSMHV2gNode;
    return(OK);
  case HSMHV2_SNODE:
    value->iValue = here->HSMHV2sNode;
    return(OK);
  case HSMHV2_BNODE:
    value->iValue = here->HSMHV2bNode;
    return(OK);
  case HSMHV2_DNODEPRIME:
    value->iValue = here->HSMHV2dNodePrime;
    return(OK);
  case HSMHV2_SNODEPRIME:
    value->iValue = here->HSMHV2sNodePrime;
    return(OK);
  case HSMHV2_SOURCECONDUCT:
    value->rValue = here->HSMHV2sourceConductance;
    return(OK);
  case HSMHV2_DRAINCONDUCT:
    value->rValue = here->HSMHV2drainConductance;
    return(OK);
  case HSMHV2_VBD:
    value->rValue = *(ckt->CKTstate0 + here->HSMHV2vbd);
    return(OK);
  case HSMHV2_VBS:
    value->rValue = *(ckt->CKTstate0 + here->HSMHV2vbs);
    return(OK);
  case HSMHV2_VGS:
    value->rValue = *(ckt->CKTstate0 + here->HSMHV2vgs);
    return(OK);
  case HSMHV2_VDS:
    value->rValue = *(ckt->CKTstate0 + here->HSMHV2vds);
    return(OK);
  case HSMHV2_CD:
    value->rValue = here->HSMHV2_ids;
    return(OK);
  case HSMHV2_ISUB:
    value->rValue = here->HSMHV2_isub;
    return(OK);
  case HSMHV2_ISUBLD:
    value->rValue = here->HSMHV2_isubld;
    return(OK);
  case HSMHV2_IDSIBPC:
    value->rValue = here->HSMHV2_idsibpc;
    return(OK);
  case HSMHV2_IGIDL:
    value->rValue = here->HSMHV2_igidl;
    return(OK);
  case HSMHV2_IGISL:
    value->rValue = here->HSMHV2_igisl;
    return(OK);
  case HSMHV2_IGD:
    value->rValue = here->HSMHV2_igd;
    return(OK);
  case HSMHV2_IGS:
    value->rValue = here->HSMHV2_igs;
    return(OK);
  case HSMHV2_IGB:
    value->rValue = here->HSMHV2_igb;
    return(OK);
  case HSMHV2_CBS:
    value->rValue = here->HSMHV2_ibs;
    return(OK);
  case HSMHV2_CBD:
    value->rValue = here->HSMHV2_ibd;
    return(OK);
  case HSMHV2_GM:
    value->rValue = here->HSMHV2_dIds_dVgsi;
    return(OK);
  case HSMHV2_GMT:
    value->rValue = here->HSMHV2_dIds_dTi;
    return(OK);
  case HSMHV2_GDS:
    value->rValue = here->HSMHV2_dIds_dVdsi;
    return(OK);
  case HSMHV2_GMBS:
    value->rValue = here->HSMHV2_dIds_dVbsi;
    return(OK);
  case HSMHV2_GBD:
    value->rValue = here->HSMHV2_gbd;
    return(OK);
  case HSMHV2_GBS:
    value->rValue = here->HSMHV2_gbs;
    return(OK);
  case HSMHV2_QB:
    value->rValue = *(ckt->CKTstate0 + here->HSMHV2qb); 
    return(OK);
  case HSMHV2_CQB:
    value->rValue = *(ckt->CKTstate0 + here->HSMHV2cqb); 
    return(OK);
  case HSMHV2_QG:
    value->rValue = *(ckt->CKTstate0 + here->HSMHV2qg); 
    return(OK);
  case HSMHV2_CQG:
    value->rValue = *(ckt->CKTstate0 + here->HSMHV2cqg); 
    return(OK);
  case HSMHV2_QD:
    value->rValue = *(ckt->CKTstate0 + here->HSMHV2qd); 
    return(OK);
  case HSMHV2_CQD:
    value->rValue = *(ckt->CKTstate0 + here->HSMHV2cqd); 
    return(OK);
  case HSMHV2_CGG:
    value->rValue = here->HSMHV2_dQg_dVgsi - here->HSMHV2_cggo;
    if (flg_nqs) value->rValue += cggb_nqs;
    return(OK);
  case HSMHV2_CGD:
    value->rValue = (here->HSMHV2_mode > 0) ? here->HSMHV2_dQg_dVdsi - here->HSMHV2_cgdo 
                                           : - (here->HSMHV2_dQg_dVdsi + here->HSMHV2_dQg_dVgsi + here->HSMHV2_dQg_dVbsi)
                                             - here->HSMHV2_cgso;
    if (flg_nqs) value->rValue += cgdb_nqs;
    return(OK);
  case HSMHV2_CGS:
    value->rValue = (here->HSMHV2_mode > 0) ? - (here->HSMHV2_dQg_dVdsi + here->HSMHV2_dQg_dVgsi + here->HSMHV2_dQg_dVbsi)
                                             - here->HSMHV2_cgso
                                           : here->HSMHV2_dQg_dVdsi - here->HSMHV2_cgdo;
    if (flg_nqs) value->rValue += cgsb_nqs;
    return(OK);
  case HSMHV2_CDG:
    value->rValue = (here->HSMHV2_mode > 0) ? here->HSMHV2_dQdi_dVgsi - here->HSMHV2_cdgo
                                           : here->HSMHV2_dQsi_dVgsi - here->HSMHV2_csgo;
    if (flg_nqs) value->rValue += cdgb_nqs; 
    return(OK);
  case HSMHV2_CDD:
    value->rValue = (here->HSMHV2_mode > 0) ? here->HSMHV2_dQdi_dVdsi - here->HSMHV2_cddo
                                           : - (here->HSMHV2_dQsi_dVdsi + here->HSMHV2_dQsi_dVgsi + here->HSMHV2_dQsi_dVbsi)
                                             - here->HSMHV2_csso;
    if (flg_nqs) value->rValue += cddb_nqs; 
    return(OK);
  case HSMHV2_CDS:
    value->rValue = (here->HSMHV2_mode > 0) ? - (here->HSMHV2_dQdi_dVdsi + here->HSMHV2_dQdi_dVgsi + here->HSMHV2_dQdi_dVbsi)
                                             - here->HSMHV2_cdso
                                           : here->HSMHV2_dQsi_dVdsi - here->HSMHV2_csdo;
    if (flg_nqs) value->rValue += cdsb_nqs; 
    return(OK);
  case HSMHV2_CBG:
    value->rValue = here->HSMHV2_dQb_dVgsi - here->HSMHV2_cbgo;
    if (flg_nqs) value->rValue += cbgb_nqs;
    return(OK);
  case HSMHV2_CBDB:
    value->rValue = (here->HSMHV2_mode > 0) ? here->HSMHV2_dQb_dVdsi - here->HSMHV2_cbdo
                                           : - (here->HSMHV2_dQb_dVdsi + here->HSMHV2_dQb_dVgsi + here->HSMHV2_dQb_dVbsi)
                                             + (here->HSMHV2_cbdo+here->HSMHV2_cbgo+here->HSMHV2_cbbo);
    if (flg_nqs) value->rValue += cbdb_nqs;
    return(OK);
  case HSMHV2_CBSB:
    value->rValue = (here->HSMHV2_mode > 0) ? - (here->HSMHV2_dQb_dVdsi + here->HSMHV2_dQb_dVgsi + here->HSMHV2_dQb_dVbsi)
                                             + (here->HSMHV2_cbdo + here->HSMHV2_cbgo + here->HSMHV2_cbbo)
                                           : here->HSMHV2_dQb_dVdsi - here->HSMHV2_cbdo;
    if (flg_nqs) value->rValue += cbsb_nqs;
    return(OK);
  case HSMHV2_CGDO:
    value->rValue = (here->HSMHV2_mode > 0) ? here->HSMHV2_cgdo : here->HSMHV2_cgso;

    return(OK);
  case HSMHV2_CGSO:
    value->rValue = (here->HSMHV2_mode > 0) ? here->HSMHV2_cgso : here->HSMHV2_cgdo;
    return(OK);
  case HSMHV2_CGBO:
    value->rValue = here->HSMHV2_cgbo;
    return(OK);
  case HSMHV2_CAPBD:
    value->rValue = here->HSMHV2_capbd;
    return(OK);
  case HSMHV2_CAPBS:
    value->rValue = here->HSMHV2_capbs;
    return(OK);
  case HSMHV2_VON:
    value->rValue = here->HSMHV2_von; 
    return(OK);
  case HSMHV2_VDSAT:
    value->rValue = here->HSMHV2_vdsat; 
    return(OK);
  case HSMHV2_QBS:
    value->rValue = *(ckt->CKTstate0 + here->HSMHV2qbs); 
    return(OK);
  case HSMHV2_QBD:
    value->rValue = *(ckt->CKTstate0 + here->HSMHV2qbd); 
    return(OK);
  case  HSMHV2_CORBNET: 
    value->iValue = here->HSMHV2_corbnet;
    return(OK);
  case  HSMHV2_RBPB:
    value->rValue = here->HSMHV2_rbpb;
    return (OK);
  case  HSMHV2_RBPD:
    value->rValue = here->HSMHV2_rbpd;
    return(OK);
  case  HSMHV2_RBPS:
    value->rValue = here->HSMHV2_rbps;
    return(OK);
  case  HSMHV2_RBDB:
    value->rValue = here->HSMHV2_rbdb;
    return(OK);
  case  HSMHV2_RBSB:
    value->rValue = here->HSMHV2_rbsb;
    return(OK);
  case  HSMHV2_CORG: 
    value->iValue = here->HSMHV2_corg;
    return(OK);
  case  HSMHV2_NGCON:
    value->rValue = here->HSMHV2_ngcon;
    return(OK);
  case  HSMHV2_XGW:
    value->rValue = here->HSMHV2_xgw;
    return(OK);
  case  HSMHV2_XGL:
    value->rValue = here->HSMHV2_xgl;
    return(OK);
  case  HSMHV2_NF:
    value->rValue = here->HSMHV2_nf;
    return(OK);
  case  HSMHV2_SA:
    value->rValue = here->HSMHV2_sa;
    return(OK);
  case  HSMHV2_SB:
    value->rValue = here->HSMHV2_sb;
    return(OK);
  case  HSMHV2_SD:
    value->rValue = here->HSMHV2_sd;
    return(OK);
  case  HSMHV2_NSUBCDFM:
    value->rValue = here->HSMHV2_nsubcdfm;
    return(OK);
  case  HSMHV2_M:
    value->rValue = here->HSMHV2_m;
    return(OK);
  case  HSMHV2_SUBLD1:
    value->rValue = here->HSMHV2_subld1;
    return(OK);
  case  HSMHV2_SUBLD2:
    value->rValue = here->HSMHV2_subld2;
    return(OK);
  case  HSMHV2_LOVER:
    value->rValue = here->HSMHV2_lover;
    return(OK);
  case  HSMHV2_LOVERS:
    value->rValue = here->HSMHV2_lovers;
    return(OK);
  case  HSMHV2_LOVERLD:
    value->rValue = here->HSMHV2_loverld;
    return(OK);
  case  HSMHV2_LDRIFT1:
    value->rValue = here->HSMHV2_ldrift1;
    return(OK);
  case  HSMHV2_LDRIFT2:
    value->rValue = here->HSMHV2_ldrift2;
    return(OK);
  case  HSMHV2_LDRIFT1S:
    value->rValue = here->HSMHV2_ldrift1s;
    return(OK);
  case  HSMHV2_LDRIFT2S:
    value->rValue = here->HSMHV2_ldrift2s;
    return(OK);
  default:
    return(E_BADPARM);
  }
  /* NOTREACHED */
}
