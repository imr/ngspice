/***********************************************************************
 HiSIM (Hiroshima University STARC IGFET Model)
 Copyright (C) 2003 STARC

 VERSION : HiSIM 1.2.0
 FILE : hsm1mask.c of HiSIM 1.2.0

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
HSM1mAsk(CKTcircuit *ckt, GENmodel *inst, int which, IFvalue *value)
{
  HSM1model *model = (HSM1model *)inst;
  switch (which) {
  case HSM1_MOD_NMOS:
    value->iValue = model->HSM1_type;
    return(OK);
  case  HSM1_MOD_PMOS:
    value->iValue = model->HSM1_type;
    return(OK);
  case  HSM1_MOD_LEVEL:
    value->iValue = model->HSM1_level;
    return(OK);
  case  HSM1_MOD_INFO:
    value->iValue = model->HSM1_info;
    return(OK);
  case HSM1_MOD_NOISE:
    value->iValue = model->HSM1_noise;
    return(OK);
  case HSM1_MOD_VERSION:
    value->iValue = model->HSM1_version;
    return(OK);
  case HSM1_MOD_SHOW:
    value->iValue = model->HSM1_show;
    return(OK);
  case  HSM1_MOD_CORSRD:
    value->iValue = model->HSM1_corsrd;
    return(OK);
  case  HSM1_MOD_COIPRV:
    value->iValue = model->HSM1_coiprv;
    return(OK);
  case  HSM1_MOD_COPPRV:
    value->iValue = model->HSM1_copprv;
    return(OK);
  case  HSM1_MOD_COCGSO:
    value->iValue = model->HSM1_cocgso;
    return(OK);
  case  HSM1_MOD_COCGDO:
    value->iValue = model->HSM1_cocgdo;
    return(OK);
  case  HSM1_MOD_COCGBO:
    value->rValue = model->HSM1_cocgbo;
    return(OK);
  case  HSM1_MOD_COADOV:
    value->iValue = model->HSM1_coadov;
    return(OK);
  case  HSM1_MOD_COXX08:
    value->iValue = model->HSM1_coxx08;
    return(OK);
  case  HSM1_MOD_COXX09:
    value->iValue = model->HSM1_coxx09;
    return(OK);
  case  HSM1_MOD_COISUB:
    value->iValue = model->HSM1_coisub;
    return(OK);
  case  HSM1_MOD_COIIGS:
    value->iValue = model->HSM1_coiigs;
    return(OK);
  case  HSM1_MOD_COGIDL:
    value->iValue = model->HSM1_cogidl;
    return(OK);
  case  HSM1_MOD_COGISL:
    value->iValue = model->HSM1_cogisl;
    return(OK);
  case  HSM1_MOD_COOVLP:
    value->iValue = model->HSM1_coovlp;
    return(OK);
  case  HSM1_MOD_CONOIS:
    value->iValue = model->HSM1_conois;
    return(OK);
  case  HSM1_MOD_COISTI: /* HiSIM1.1 */
    value->iValue = model->HSM1_coisti;
    return(OK);
  case  HSM1_MOD_COSMBI: /* HiSIM1.2 */
    value->iValue = model->HSM1_cosmbi;
    return(OK);
  case  HSM1_MOD_VMAX:
    value->rValue = model->HSM1_vmax;
    return(OK);
  case  HSM1_MOD_BGTMP1:
    value->rValue = model->HSM1_bgtmp1;
    return(OK);
  case  HSM1_MOD_BGTMP2:
    value->rValue = model->HSM1_bgtmp2;
    return(OK);
  case  HSM1_MOD_TOX:
    value->rValue = model->HSM1_tox;
    return(OK);
  case  HSM1_MOD_XLD:
    value->rValue = model->HSM1_xld;
    return(OK);
  case  HSM1_MOD_XWD:
    value->rValue = model->HSM1_xwd;
    return(OK);
  case  HSM1_MOD_XJ: /* HiSIM1.0 */
    value->rValue = model->HSM1_xj;
    return(OK);
  case  HSM1_MOD_XQY: /* HiSIM1.1 */
    value->rValue = model->HSM1_xqy;
    return(OK);
  case  HSM1_MOD_RS:
    value->rValue = model->HSM1_rs;
    return(OK);
  case  HSM1_MOD_RD:
    value->rValue = model->HSM1_rd;
    return(OK);
  case  HSM1_MOD_VFBC:
    value->rValue = model->HSM1_vfbc;
    return(OK);
  case  HSM1_MOD_NSUBC:
    value->rValue = model->HSM1_nsubc;
      return(OK);
  case  HSM1_MOD_PARL1:
    value->rValue = model->HSM1_parl1;
    return(OK);
  case  HSM1_MOD_PARL2:
    value->rValue = model->HSM1_parl2;
    return(OK);
  case  HSM1_MOD_LP:
    value->rValue = model->HSM1_lp;
    return(OK);
  case  HSM1_MOD_NSUBP:
    value->rValue = model->HSM1_nsubp;
    return(OK);
  case  HSM1_MOD_SCP1:
    value->rValue = model->HSM1_scp1;
    return(OK);
  case  HSM1_MOD_SCP2:
    value->rValue = model->HSM1_scp2;
    return(OK);
  case  HSM1_MOD_SCP3:
    value->rValue = model->HSM1_scp3;
    return(OK);
  case  HSM1_MOD_SC1:
    value->rValue = model->HSM1_sc1;
    return(OK);
  case  HSM1_MOD_SC2:
    value->rValue = model->HSM1_sc2;
    return(OK);
  case  HSM1_MOD_SC3:
    value->rValue = model->HSM1_sc3;
    return(OK);
  case  HSM1_MOD_PGD1:
    value->rValue = model->HSM1_pgd1;
    return(OK);
  case  HSM1_MOD_PGD2:
    value->rValue = model->HSM1_pgd2;
    return(OK);
  case  HSM1_MOD_PGD3:
    value->rValue = model->HSM1_pgd3;
    return(OK);
  case  HSM1_MOD_NDEP:
    value->rValue = model->HSM1_ndep;
    return(OK);
  case  HSM1_MOD_NINV:
    value->rValue = model->HSM1_ninv;
    return(OK);
  case  HSM1_MOD_NINVD:
    value->rValue = model->HSM1_ninvd;
    return(OK);
  case  HSM1_MOD_MUECB0:
    value->rValue = model->HSM1_muecb0;
    return(OK);
  case  HSM1_MOD_MUECB1:
    value->rValue = model->HSM1_muecb1;
    return(OK);
  case  HSM1_MOD_MUEPH1:
    value->rValue = model->HSM1_mueph1;
    return(OK);
  case  HSM1_MOD_MUEPH0:
    value->rValue = model->HSM1_mueph0;
    return(OK);
  case  HSM1_MOD_MUEPH2:
    value->rValue = model->HSM1_mueph2;
    return(OK);
  case  HSM1_MOD_W0:
    value->rValue = model->HSM1_w0;
    return(OK);
  case  HSM1_MOD_MUESR1:
    value->rValue = model->HSM1_muesr1;
    return(OK);
  case  HSM1_MOD_MUESR0:
    value->rValue = model->HSM1_muesr0;
    return(OK);
  case  HSM1_MOD_BB:
    value->rValue = model->HSM1_bb;
    return(OK);
  case  HSM1_MOD_SUB1:
    value->rValue = model->HSM1_sub1;
    return(OK);
  case  HSM1_MOD_SUB2:
    value->rValue = model->HSM1_sub2;
    return(OK);
  case  HSM1_MOD_SUB3:
    value->rValue = model->HSM1_sub3;
    return(OK);
  case  HSM1_MOD_WVTHSC: /* HiSIM1.1 */
    value->rValue = model->HSM1_wvthsc;
    return(OK);
  case  HSM1_MOD_NSTI: /* HiSIM1.1 */
    value->rValue = model->HSM1_nsti;
    return(OK);
  case  HSM1_MOD_WSTI: /* HiSIM1.1 */
    value->rValue = model->HSM1_wsti;
    return(OK);
  case  HSM1_MOD_CGSO:
    value->rValue = model->HSM1_cgso;
    return(OK);
  case  HSM1_MOD_CGDO:
    value->rValue = model->HSM1_cgdo;
    return(OK);
  case  HSM1_MOD_CGBO:
    value->rValue = model->HSM1_cgbo;
    return(OK);
  case  HSM1_MOD_TPOLY:
    value->rValue = model->HSM1_tpoly;
    return(OK);
  case  HSM1_MOD_JS0:
    value->rValue = model->HSM1_js0;
    return(OK);
  case  HSM1_MOD_JS0SW:
    value->rValue = model->HSM1_js0sw;
    return(OK);
  case  HSM1_MOD_NJ:
    value->rValue = model->HSM1_nj;
    return(OK);
  case  HSM1_MOD_NJSW:
    value->rValue = model->HSM1_njsw;
    return(OK);
  case  HSM1_MOD_XTI:
    value->rValue = model->HSM1_xti;
    return(OK);
  case  HSM1_MOD_CJ:
    value->rValue = model->HSM1_cj;
    return(OK);
  case  HSM1_MOD_CJSW:
    value->rValue = model->HSM1_cjsw;
    return(OK);
  case  HSM1_MOD_CJSWG:
    value->rValue = model->HSM1_cjswg;
    return(OK);
  case  HSM1_MOD_MJ:
    value->rValue = model->HSM1_mj;
    return(OK);
  case  HSM1_MOD_MJSW:
    value->rValue = model->HSM1_mjsw;
    return(OK);
  case  HSM1_MOD_MJSWG:
    value->rValue = model->HSM1_mjswg;
    return(OK);
  case  HSM1_MOD_PB:
    value->rValue = model->HSM1_pbsw;
    return(OK);
  case  HSM1_MOD_PBSW:
    value->rValue = model->HSM1_pbsw;
    return(OK);
  case  HSM1_MOD_PBSWG:
    value->rValue = model->HSM1_pbswg;
    return(OK);
  case  HSM1_MOD_XPOLYD:
    value->rValue = model->HSM1_xpolyd;
    return(OK);
  case  HSM1_MOD_CLM1:
    value->rValue = model->HSM1_clm1;
    return(OK);
  case  HSM1_MOD_CLM2:
    value->rValue = model->HSM1_clm2;
    return(OK);
  case  HSM1_MOD_CLM3:
    value->rValue = model->HSM1_clm3;
    return(OK);
  case  HSM1_MOD_MUETMP:
    value->rValue = model->HSM1_muetmp;
    return(OK);
  case  HSM1_MOD_RPOCK1:
    value->rValue = model->HSM1_rpock1;
    return(OK);
  case  HSM1_MOD_RPOCK2:
    value->rValue = model->HSM1_rpock2;
    return(OK);
  case  HSM1_MOD_RPOCP1: /* HiSIM1.1 */
    value->rValue = model->HSM1_rpocp1;
    return(OK);
  case  HSM1_MOD_RPOCP2: /* HiSIM1.1 */
    value->rValue = model->HSM1_rpocp2;
    return(OK);
  case  HSM1_MOD_VOVER:
    value->rValue = model->HSM1_vover;
    return(OK);
  case  HSM1_MOD_VOVERP:
    value->rValue = model->HSM1_voverp;
    return(OK);
  case  HSM1_MOD_WFC:
    value->rValue = model->HSM1_wfc;
    return(OK);
  case  HSM1_MOD_QME1:
    value->rValue = model->HSM1_qme1;
    return(OK);
  case  HSM1_MOD_QME2:
    value->rValue = model->HSM1_qme2;
    return(OK);
  case  HSM1_MOD_QME3:
    value->rValue = model->HSM1_qme3;
    return(OK);
  case  HSM1_MOD_GIDL1:
    value->rValue = model->HSM1_gidl1;
    return(OK);
  case  HSM1_MOD_GIDL2:
    value->rValue = model->HSM1_gidl2;
    return(OK);
  case  HSM1_MOD_GIDL3:
    value->rValue = model->HSM1_gidl3;
    return(OK);
  case  HSM1_MOD_GLEAK1:
    value->rValue = model->HSM1_gleak1;
    return(OK);
  case  HSM1_MOD_GLEAK2:
    value->rValue = model->HSM1_gleak2;
    return(OK);
  case  HSM1_MOD_GLEAK3:
    value->rValue = model->HSM1_gleak3;
    return(OK);
  case  HSM1_MOD_VZADD0:
    value->rValue = model->HSM1_vzadd0;
    return(OK);
  case  HSM1_MOD_PZADD0:
    value->rValue = model->HSM1_pzadd0;
    return(OK);
  case  HSM1_MOD_NFTRP:
    value->rValue = model->HSM1_nftrp;
    return(OK);
  case  HSM1_MOD_NFALP:
    value->rValue = model->HSM1_nfalp;
    return(OK);
  case  HSM1_MOD_CIT:
    value->rValue = model->HSM1_cit;
    return(OK);
  case  HSM1_MOD_GLPART1: /* HiSIM1.2 */
    value->rValue = model->HSM1_glpart1;
    return(OK);
  case  HSM1_MOD_GLPART2: /* HiSIM1.2 */
    value->rValue = model->HSM1_glpart2;
    return(OK);
  case  HSM1_MOD_KAPPA: /* HiSIM1.2 */
    value->rValue = model->HSM1_kappa;
    return(OK);
  case  HSM1_MOD_XDIFFD: /* HiSIM1.2 */
    value->rValue = model->HSM1_xdiffd;
    return(OK);
  case  HSM1_MOD_PTHROU: /* HiSIM1.2 */
    value->rValue = model->HSM1_pthrou;
    return(OK);
  case  HSM1_MOD_VDIFFJ: /* HiSIM1.2 */
    value->rValue = model->HSM1_vdiffj;
    return(OK);
  case HSM1_MOD_KF:
    value->rValue = model->HSM1_kf;
    return(OK);
  case HSM1_MOD_AF:
    value->rValue = model->HSM1_af;
    return(OK);
  case HSM1_MOD_EF:
    value->rValue = model->HSM1_ef;
    return(OK);
  default:
    return(E_BADPARM);
  }
  /* NOTREACHED */
}


