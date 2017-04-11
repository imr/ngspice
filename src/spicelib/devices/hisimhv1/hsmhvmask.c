/***********************************************************************

 HiSIM (Hiroshima University STARC IGFET Model)
 Copyright (C) 2012 Hiroshima University & STARC

 MODEL NAME : HiSIM_HV 
 ( VERSION : 1  SUBVERSION : 2  REVISION : 4 )
 Model Parameter VERSION : 1.23
 FILE : hsmhvmask.c

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

int HSMHVmAsk(
     CKTcircuit *ckt,
     GENmodel *inst,
     int which,
     IFvalue *value)
{
  HSMHVmodel *model = (HSMHVmodel *)inst;

  NG_IGNORE(ckt);

  switch (which) {
  case HSMHV_MOD_NMOS:
    value->iValue = model->HSMHV_type;
    return(OK);
  case  HSMHV_MOD_PMOS:
    value->iValue = model->HSMHV_type;
    return(OK);
  case  HSMHV_MOD_LEVEL:
    value->iValue = model->HSMHV_level;
    return(OK);
  case  HSMHV_MOD_INFO:
    value->iValue = model->HSMHV_info;
    return(OK);
  case HSMHV_MOD_NOISE:
    value->iValue = model->HSMHV_noise;
    return(OK);
  case HSMHV_MOD_VERSION:
    value->sValue = model->HSMHV_version;
    return(OK);
  case HSMHV_MOD_SHOW:
    value->iValue = model->HSMHV_show;
    return(OK);
  case  HSMHV_MOD_CORSRD:
    value->iValue = model->HSMHV_corsrd;
    return(OK);
  case  HSMHV_MOD_CORG:
    value->iValue = model->HSMHV_corg;
    return(OK);
  case  HSMHV_MOD_COIPRV:
    value->iValue = model->HSMHV_coiprv;
    return(OK);
  case  HSMHV_MOD_COPPRV:
    value->iValue = model->HSMHV_copprv;
    return(OK);
  case  HSMHV_MOD_COADOV:
    value->iValue = model->HSMHV_coadov;
    return(OK);
  case  HSMHV_MOD_COISUB:
    value->iValue = model->HSMHV_coisub;
    return(OK);
  case  HSMHV_MOD_COIIGS:
    value->iValue = model->HSMHV_coiigs;
    return(OK);
  case  HSMHV_MOD_COGIDL:
    value->iValue = model->HSMHV_cogidl;
    return(OK);
  case  HSMHV_MOD_COOVLP:
    value->iValue = model->HSMHV_coovlp;
    return(OK);
  case  HSMHV_MOD_COOVLPS:
    value->iValue = model->HSMHV_coovlps;
    return(OK);
  case  HSMHV_MOD_COFLICK:
    value->iValue = model->HSMHV_coflick;
    return(OK);
  case  HSMHV_MOD_COISTI:
    value->iValue = model->HSMHV_coisti;
    return(OK);
  case  HSMHV_MOD_CONQS:
    value->iValue = model->HSMHV_conqs;
    return(OK);
  case  HSMHV_MOD_CORBNET:
    value->iValue = model->HSMHV_corbnet;
    return(OK);
  case  HSMHV_MOD_COTHRML:
    value->iValue = model->HSMHV_cothrml;
    return(OK);
  case  HSMHV_MOD_COIGN:
    value->iValue = model->HSMHV_coign;
    return(OK);
  case  HSMHV_MOD_CODFM:
    value->iValue = model->HSMHV_codfm;
    return(OK);
  case  HSMHV_MOD_COQOVSM:
    value->iValue = model->HSMHV_coqovsm;
    return(OK);
  case  HSMHV_MOD_COSELFHEAT: /* Self-heating model */
    value->iValue = model->HSMHV_coselfheat;
    return(OK);
  case  HSMHV_MOD_COSUBNODE: 
    value->iValue = model->HSMHV_cosubnode;
    return(OK);
  case  HSMHV_MOD_COSYM: /* Symmetry model for HV */
    value->iValue = model->HSMHV_cosym;
    return(OK);
  case  HSMHV_MOD_COTEMP:
    value->iValue = model->HSMHV_cotemp;
    return(OK);
  case  HSMHV_MOD_COLDRIFT:
    value->iValue = model->HSMHV_coldrift;
    return(OK);
  case  HSMHV_MOD_VMAX:
    value->rValue = model->HSMHV_vmax;
    return(OK);
  case  HSMHV_MOD_VMAXT1:
    value->rValue = model->HSMHV_vmaxt1;
    return(OK);
  case  HSMHV_MOD_VMAXT2:
    value->rValue = model->HSMHV_vmaxt2;
    return(OK);
  case  HSMHV_MOD_BGTMP1:
    value->rValue = model->HSMHV_bgtmp1;
    return(OK);
  case  HSMHV_MOD_BGTMP2:
    value->rValue = model->HSMHV_bgtmp2;
    return(OK);
  case  HSMHV_MOD_EG0:
    value->rValue = model->HSMHV_eg0;
    return(OK);
  case  HSMHV_MOD_TOX:
    value->rValue = model->HSMHV_tox;
    return(OK);
  case  HSMHV_MOD_XLD:
    value->rValue = model->HSMHV_xld;
    return(OK);
  case  HSMHV_MOD_LOVER:
    value->rValue = model->HSMHV_lover;
    return(OK);
  case  HSMHV_MOD_LOVERS:
    value->rValue = model->HSMHV_lovers;
    return(OK);
  case  HSMHV_MOD_RDOV11:
    value->rValue = model->HSMHV_rdov11;
    return(OK);
  case  HSMHV_MOD_RDOV12:
    value->rValue = model->HSMHV_rdov12;
    return(OK);
  case  HSMHV_MOD_RDOV13:
    value->rValue = model->HSMHV_rdov13;
    return(OK);
  case  HSMHV_MOD_RDSLP1:
    value->rValue = model->HSMHV_rdslp1;
    return(OK);
  case  HSMHV_MOD_RDICT1:
    value->rValue = model->HSMHV_rdict1;
    return(OK);
  case  HSMHV_MOD_RDSLP2:
    value->rValue = model->HSMHV_rdslp2;
    return(OK);
  case  HSMHV_MOD_RDICT2:
    value->rValue = model->HSMHV_rdict2;
    return(OK);
  case  HSMHV_MOD_LOVERLD:
    value->rValue = model->HSMHV_loverld;
    return(OK);
  case  HSMHV_MOD_LDRIFT1:
    value->rValue = model->HSMHV_ldrift1;
    return(OK);
  case  HSMHV_MOD_LDRIFT2:
    value->rValue = model->HSMHV_ldrift2;
    return(OK);
  case  HSMHV_MOD_LDRIFT1S:
    value->rValue = model->HSMHV_ldrift1s;
    return(OK);
  case  HSMHV_MOD_LDRIFT2S:
    value->rValue = model->HSMHV_ldrift2s;
    return(OK);
  case  HSMHV_MOD_SUBLD1:
    value->rValue = model->HSMHV_subld1;
    return(OK);
  case  HSMHV_MOD_SUBLD2:
    value->rValue = model->HSMHV_subld2;
    return(OK);
  case  HSMHV_MOD_DDLTMAX: /* Vdseff */
    value->rValue = model->HSMHV_ddltmax;
    return(OK);
  case  HSMHV_MOD_DDLTSLP: /* Vdseff */
    value->rValue = model->HSMHV_ddltslp;
    return(OK);
  case  HSMHV_MOD_DDLTICT: /* Vdseff */
    value->rValue = model->HSMHV_ddltict;
    return(OK);
  case  HSMHV_MOD_VFBOVER:
    value->rValue = model->HSMHV_vfbover;
    return(OK);
  case  HSMHV_MOD_NOVER:
    value->rValue = model->HSMHV_nover;
    return(OK);
  case  HSMHV_MOD_NOVERS:
    value->rValue = model->HSMHV_novers;
    return(OK);
  case  HSMHV_MOD_XWD:
    value->rValue = model->HSMHV_xwd;
    return(OK);
  case  HSMHV_MOD_XWDC:
    value->rValue = model->HSMHV_xwdc;
    return(OK);
  case  HSMHV_MOD_XL:
    value->rValue = model->HSMHV_xl;
    return(OK);
  case  HSMHV_MOD_XW:
    value->rValue = model->HSMHV_xw;
    return(OK);
  case  HSMHV_MOD_SAREF:
    value->rValue = model->HSMHV_saref;
    return(OK);
  case  HSMHV_MOD_SBREF:
    value->rValue = model->HSMHV_sbref;
    return(OK);
  case  HSMHV_MOD_LL:
    value->rValue = model->HSMHV_ll;
    return(OK);
  case  HSMHV_MOD_LLD:
    value->rValue = model->HSMHV_lld;
    return(OK);
  case  HSMHV_MOD_LLN:
    value->rValue = model->HSMHV_lln;
    return(OK);
  case  HSMHV_MOD_WL:
    value->rValue = model->HSMHV_wl;
    return(OK);
  case  HSMHV_MOD_WL1:
    value->rValue = model->HSMHV_wl1;
    return(OK);
  case  HSMHV_MOD_WL1P:
    value->rValue = model->HSMHV_wl1p;
    return(OK);
  case  HSMHV_MOD_WL2:
    value->rValue = model->HSMHV_wl2;
    return(OK);
  case  HSMHV_MOD_WL2P:
    value->rValue = model->HSMHV_wl2p;
    return(OK);
  case  HSMHV_MOD_WLD:
    value->rValue = model->HSMHV_wld;
    return(OK);
  case  HSMHV_MOD_WLN:
    value->rValue = model->HSMHV_wln;
    return(OK);
  case  HSMHV_MOD_XQY:
    value->rValue = model->HSMHV_xqy;
    return(OK);
  case  HSMHV_MOD_XQY1:
    value->rValue = model->HSMHV_xqy1;
    return(OK);
  case  HSMHV_MOD_XQY2:
    value->rValue = model->HSMHV_xqy2;
    return(OK);
  case  HSMHV_MOD_RS:
    value->rValue = model->HSMHV_rs;
    return(OK);
  case  HSMHV_MOD_RD:
    value->rValue = model->HSMHV_rd;
    return(OK);
  case  HSMHV_MOD_RSH:
    value->rValue = model->HSMHV_rsh;
    return(OK);
  case  HSMHV_MOD_RSHG:
    value->rValue = model->HSMHV_rshg;
    return(OK);
  case  HSMHV_MOD_VFBC:
    value->rValue = model->HSMHV_vfbc;
    return(OK);
  case  HSMHV_MOD_VBI:
    value->rValue = model->HSMHV_vbi;
    return(OK);
  case  HSMHV_MOD_NSUBC:
    value->rValue = model->HSMHV_nsubc;
      return(OK);
  case  HSMHV_MOD_PARL2:
    value->rValue = model->HSMHV_parl2;
    return(OK);
  case  HSMHV_MOD_LP:
    value->rValue = model->HSMHV_lp;
    return(OK);
  case  HSMHV_MOD_NSUBP:
    value->rValue = model->HSMHV_nsubp;
    return(OK);
  case  HSMHV_MOD_NSUBP0:
    value->rValue = model->HSMHV_nsubp0;
    return(OK);
  case  HSMHV_MOD_NSUBWP:
    value->rValue = model->HSMHV_nsubwp;
    return(OK);
  case  HSMHV_MOD_SCP1:
    value->rValue = model->HSMHV_scp1;
    return(OK);
  case  HSMHV_MOD_SCP2:
    value->rValue = model->HSMHV_scp2;
    return(OK);
  case  HSMHV_MOD_SCP3:
    value->rValue = model->HSMHV_scp3;
    return(OK);
  case  HSMHV_MOD_SC1:
    value->rValue = model->HSMHV_sc1;
    return(OK);
  case  HSMHV_MOD_SC2:
    value->rValue = model->HSMHV_sc2;
    return(OK);
  case  HSMHV_MOD_SC3:
    value->rValue = model->HSMHV_sc3;
    return(OK);
  case  HSMHV_MOD_SC4:
    value->rValue = model->HSMHV_sc4;
    return(OK);
  case  HSMHV_MOD_PGD1:
    value->rValue = model->HSMHV_pgd1;
    return(OK);
  case  HSMHV_MOD_PGD2:
    value->rValue = model->HSMHV_pgd2;
    return(OK);
  case  HSMHV_MOD_PGD3:
    value->rValue = model->HSMHV_pgd3;
    return(OK);
  case  HSMHV_MOD_PGD4:
    value->rValue = model->HSMHV_pgd4;
    return(OK);
  case  HSMHV_MOD_NDEP:
    value->rValue = model->HSMHV_ndep;
    return(OK);
  case  HSMHV_MOD_NDEPL:
    value->rValue = model->HSMHV_ndepl;
    return(OK);
  case  HSMHV_MOD_NDEPLP:
    value->rValue = model->HSMHV_ndeplp;
    return(OK);
  case  HSMHV_MOD_NINV:
    value->rValue = model->HSMHV_ninv;
    return(OK);
  case  HSMHV_MOD_MUECB0:
    value->rValue = model->HSMHV_muecb0;
    return(OK);
  case  HSMHV_MOD_MUECB1:
    value->rValue = model->HSMHV_muecb1;
    return(OK);
  case  HSMHV_MOD_MUEPH1:
    value->rValue = model->HSMHV_mueph1;
    return(OK);
  case  HSMHV_MOD_MUEPH0:
    value->rValue = model->HSMHV_mueph0;
    return(OK);
  case  HSMHV_MOD_MUEPHW:
    value->rValue = model->HSMHV_muephw;
    return(OK);
  case  HSMHV_MOD_MUEPWP:
    value->rValue = model->HSMHV_muepwp;
    return(OK);
  case  HSMHV_MOD_MUEPHL:
    value->rValue = model->HSMHV_muephl;
    return(OK);
  case  HSMHV_MOD_MUEPLP:
    value->rValue = model->HSMHV_mueplp;
    return(OK);
  case  HSMHV_MOD_MUEPHS:
    value->rValue = model->HSMHV_muephs;
    return(OK);
  case  HSMHV_MOD_MUEPSP:
    value->rValue = model->HSMHV_muepsp;
    return(OK);
  case  HSMHV_MOD_VTMP:
    value->rValue = model->HSMHV_vtmp;
    return(OK);
  case  HSMHV_MOD_WVTH0:
    value->rValue = model->HSMHV_wvth0;
    return(OK);
  case  HSMHV_MOD_MUESR1:
    value->rValue = model->HSMHV_muesr1;
    return(OK);
  case  HSMHV_MOD_MUESR0:
    value->rValue = model->HSMHV_muesr0;
    return(OK);
  case  HSMHV_MOD_MUESRL:
    value->rValue = model->HSMHV_muesrl;
    return(OK);
  case  HSMHV_MOD_MUESLP:
    value->rValue = model->HSMHV_mueslp;
    return(OK);
  case  HSMHV_MOD_MUESRW:
    value->rValue = model->HSMHV_muesrw;
    return(OK);
  case  HSMHV_MOD_MUESWP:
    value->rValue = model->HSMHV_mueswp;
    return(OK);
  case  HSMHV_MOD_BB:
    value->rValue = model->HSMHV_bb;
    return(OK);
  case  HSMHV_MOD_SUB1:
    value->rValue = model->HSMHV_sub1;
    return(OK);
  case  HSMHV_MOD_SUB2:
    value->rValue = model->HSMHV_sub2;
    return(OK);
  case  HSMHV_MOD_SVGS:
    value->rValue = model->HSMHV_svgs;
    return(OK);
  case  HSMHV_MOD_SVGSL:
    value->rValue = model->HSMHV_svgsl;
    return(OK);
  case  HSMHV_MOD_SVGSLP:
    value->rValue = model->HSMHV_svgslp;
    return(OK);
  case  HSMHV_MOD_SVGSW:
    value->rValue = model->HSMHV_svgsw;
    return(OK);
  case  HSMHV_MOD_SVGSWP:
    value->rValue = model->HSMHV_svgswp;
    return(OK);
  case  HSMHV_MOD_SVBS:
    value->rValue = model->HSMHV_svbs;
    return(OK);
  case  HSMHV_MOD_SVBSL:
    value->rValue = model->HSMHV_svbsl;
    return(OK);
  case  HSMHV_MOD_SVBSLP:
    value->rValue = model->HSMHV_svbslp;
    return(OK);
  case  HSMHV_MOD_SVDS:
    value->rValue = model->HSMHV_svds;
    return(OK);
  case  HSMHV_MOD_SLG:
    value->rValue = model->HSMHV_slg;
    return(OK);
  case  HSMHV_MOD_SLGL:
    value->rValue = model->HSMHV_slgl;
    return(OK);
  case  HSMHV_MOD_SLGLP:
    value->rValue = model->HSMHV_slglp;
    return(OK);
  case  HSMHV_MOD_SUB1L:
    value->rValue = model->HSMHV_sub1l;
    return(OK);
  case  HSMHV_MOD_SUB1LP:
    value->rValue = model->HSMHV_sub1lp;
    return(OK);
  case  HSMHV_MOD_SUB2L:
    value->rValue = model->HSMHV_sub2l;
    return(OK);
  case  HSMHV_MOD_FN1:
    value->rValue = model->HSMHV_fn1;
    return(OK);
  case  HSMHV_MOD_FN2:
    value->rValue = model->HSMHV_fn2;
    return(OK);
  case  HSMHV_MOD_FN3:
    value->rValue = model->HSMHV_fn3;
    return(OK);
  case  HSMHV_MOD_FVBS:
    value->rValue = model->HSMHV_fvbs;
    return(OK);
  case  HSMHV_MOD_NSTI:
    value->rValue = model->HSMHV_nsti;
    return(OK);
  case  HSMHV_MOD_WSTI:
    value->rValue = model->HSMHV_wsti;
    return(OK);
  case  HSMHV_MOD_WSTIL:
    value->rValue = model->HSMHV_wstil;
    return(OK);
  case  HSMHV_MOD_WSTILP:
    value->rValue = model->HSMHV_wstilp;
    return(OK);
  case  HSMHV_MOD_WSTIW:
    value->rValue = model->HSMHV_wstiw;
    return(OK);
  case  HSMHV_MOD_WSTIWP:
    value->rValue = model->HSMHV_wstiwp;
    return(OK);
  case  HSMHV_MOD_SCSTI1:
    value->rValue = model->HSMHV_scsti1;
    return(OK);
  case  HSMHV_MOD_SCSTI2:
    value->rValue = model->HSMHV_scsti2;
    return(OK);
  case  HSMHV_MOD_VTHSTI:
    value->rValue = model->HSMHV_vthsti;
    return(OK);
  case  HSMHV_MOD_VDSTI:
    value->rValue = model->HSMHV_vdsti;
    return(OK);
  case  HSMHV_MOD_MUESTI1:
    value->rValue = model->HSMHV_muesti1;
    return(OK);
  case  HSMHV_MOD_MUESTI2:
    value->rValue = model->HSMHV_muesti2;
    return(OK);
  case  HSMHV_MOD_MUESTI3:
    value->rValue = model->HSMHV_muesti3;
    return(OK);
  case  HSMHV_MOD_NSUBPSTI1:
    value->rValue = model->HSMHV_nsubpsti1;
    return(OK);
  case  HSMHV_MOD_NSUBPSTI2:
    value->rValue = model->HSMHV_nsubpsti2;
    return(OK);
  case  HSMHV_MOD_NSUBPSTI3:
    value->rValue = model->HSMHV_nsubpsti3;
    return(OK);
  case  HSMHV_MOD_LPEXT:
    value->rValue = model->HSMHV_lpext;
    return(OK);
  case  HSMHV_MOD_NPEXT:
    value->rValue = model->HSMHV_npext;
    return(OK);
  case  HSMHV_MOD_SCP22:
    value->rValue = model->HSMHV_scp22;
    return(OK);
  case  HSMHV_MOD_SCP21:
    value->rValue = model->HSMHV_scp21;
    return(OK);
  case  HSMHV_MOD_BS1:
    value->rValue = model->HSMHV_bs1;
    return(OK);
  case  HSMHV_MOD_BS2:
    value->rValue = model->HSMHV_bs2;
    return(OK);
  case  HSMHV_MOD_CGSO:
    value->rValue = model->HSMHV_cgso;
    return(OK);
  case  HSMHV_MOD_CGDO:
    value->rValue = model->HSMHV_cgdo;
    return(OK);
  case  HSMHV_MOD_CGBO:
    value->rValue = model->HSMHV_cgbo;
    return(OK);
  case  HSMHV_MOD_TPOLY:
    value->rValue = model->HSMHV_tpoly;
    return(OK);
  case  HSMHV_MOD_JS0:
    value->rValue = model->HSMHV_js0;
    return(OK);
  case  HSMHV_MOD_JS0SW:
    value->rValue = model->HSMHV_js0sw;
    return(OK);
  case  HSMHV_MOD_NJ:
    value->rValue = model->HSMHV_nj;
    return(OK);
  case  HSMHV_MOD_NJSW:
    value->rValue = model->HSMHV_njsw;
    return(OK);
  case  HSMHV_MOD_XTI:
    value->rValue = model->HSMHV_xti;
    return(OK);
  case  HSMHV_MOD_CJ:
    value->rValue = model->HSMHV_cj;
    return(OK);
  case  HSMHV_MOD_CJSW:
    value->rValue = model->HSMHV_cjsw;
    return(OK);
  case  HSMHV_MOD_CJSWG:
    value->rValue = model->HSMHV_cjswg;
    return(OK);
  case  HSMHV_MOD_MJ:
    value->rValue = model->HSMHV_mj;
    return(OK);
  case  HSMHV_MOD_MJSW:
    value->rValue = model->HSMHV_mjsw;
    return(OK);
  case  HSMHV_MOD_MJSWG:
    value->rValue = model->HSMHV_mjswg;
    return(OK);
  case  HSMHV_MOD_PB:
    value->rValue = model->HSMHV_pb;
    return(OK);
  case  HSMHV_MOD_PBSW:
    value->rValue = model->HSMHV_pbsw;
    return(OK);
  case  HSMHV_MOD_PBSWG:
    value->rValue = model->HSMHV_pbswg;
    return(OK);
  case  HSMHV_MOD_XTI2:
    value->rValue = model->HSMHV_xti2;
    return(OK);
  case  HSMHV_MOD_CISB:
    value->rValue = model->HSMHV_cisb;
    return(OK);
  case  HSMHV_MOD_CVB:
    value->rValue = model->HSMHV_cvb;
    return(OK);
  case  HSMHV_MOD_CTEMP:
    value->rValue = model->HSMHV_ctemp;
    return(OK);
  case  HSMHV_MOD_CISBK:
    value->rValue = model->HSMHV_cisbk;
    return(OK);
  case  HSMHV_MOD_CVBK:
    value->rValue = model->HSMHV_cvbk;
    return(OK);
  case  HSMHV_MOD_DIVX:
    value->rValue = model->HSMHV_divx;
    return(OK);
  case  HSMHV_MOD_CLM1:
    value->rValue = model->HSMHV_clm1;
    return(OK);
  case  HSMHV_MOD_CLM2:
    value->rValue = model->HSMHV_clm2;
    return(OK);
  case  HSMHV_MOD_CLM3:
    value->rValue = model->HSMHV_clm3;
    return(OK);
  case  HSMHV_MOD_CLM5:
    value->rValue = model->HSMHV_clm5;
    return(OK);
  case  HSMHV_MOD_CLM6:
    value->rValue = model->HSMHV_clm6;
    return(OK);
  case  HSMHV_MOD_MUETMP:
    value->rValue = model->HSMHV_muetmp;
    return(OK);
  case  HSMHV_MOD_VOVER:
    value->rValue = model->HSMHV_vover;
    return(OK);
  case  HSMHV_MOD_VOVERP:
    value->rValue = model->HSMHV_voverp;
    return(OK);
  case  HSMHV_MOD_VOVERS:
    value->rValue = model->HSMHV_vovers;
    return(OK);
  case  HSMHV_MOD_VOVERSP:
    value->rValue = model->HSMHV_voversp;
    return(OK);
  case  HSMHV_MOD_WFC:
    value->rValue = model->HSMHV_wfc;
    return(OK);
  case  HSMHV_MOD_NSUBCW:
    value->rValue = model->HSMHV_nsubcw;
    return(OK);
  case  HSMHV_MOD_NSUBCWP:
    value->rValue = model->HSMHV_nsubcwp;
    return(OK);
  case  HSMHV_MOD_QME1:
    value->rValue = model->HSMHV_qme1;
    return(OK);
  case  HSMHV_MOD_QME2:
    value->rValue = model->HSMHV_qme2;
    return(OK);
  case  HSMHV_MOD_QME3:
    value->rValue = model->HSMHV_qme3;
    return(OK);
  case  HSMHV_MOD_GIDL1:
    value->rValue = model->HSMHV_gidl1;
    return(OK);
  case  HSMHV_MOD_GIDL2:
    value->rValue = model->HSMHV_gidl2;
    return(OK);
  case  HSMHV_MOD_GIDL3:
    value->rValue = model->HSMHV_gidl3;
    return(OK);
  case  HSMHV_MOD_GIDL4:
    value->rValue = model->HSMHV_gidl4;
    return(OK);
  case  HSMHV_MOD_GIDL5:
    value->rValue = model->HSMHV_gidl5;
    return(OK);
  case  HSMHV_MOD_GLEAK1:
    value->rValue = model->HSMHV_gleak1;
    return(OK);
  case  HSMHV_MOD_GLEAK2:
    value->rValue = model->HSMHV_gleak2;
    return(OK);
  case  HSMHV_MOD_GLEAK3:
    value->rValue = model->HSMHV_gleak3;
    return(OK);
  case  HSMHV_MOD_GLEAK4:
    value->rValue = model->HSMHV_gleak4;
    return(OK);
  case  HSMHV_MOD_GLEAK5:
    value->rValue = model->HSMHV_gleak5;
    return(OK);
  case  HSMHV_MOD_GLEAK6:
    value->rValue = model->HSMHV_gleak6;
    return(OK);
  case  HSMHV_MOD_GLEAK7:
    value->rValue = model->HSMHV_gleak7;
    return(OK);
  case  HSMHV_MOD_GLPART1:
    value->rValue = model->HSMHV_glpart1;
    return(OK);
  case  HSMHV_MOD_GLKSD1:
    value->rValue = model->HSMHV_glksd1;
    return(OK);
  case  HSMHV_MOD_GLKSD2:
    value->rValue = model->HSMHV_glksd2;
    return(OK);
  case  HSMHV_MOD_GLKSD3:
    value->rValue = model->HSMHV_glksd3;
    return(OK);
  case  HSMHV_MOD_GLKB1:
    value->rValue = model->HSMHV_glkb1;
    return(OK);
  case  HSMHV_MOD_GLKB2:
    value->rValue = model->HSMHV_glkb2;
    return(OK);
  case  HSMHV_MOD_GLKB3:
    value->rValue = model->HSMHV_glkb3;
    return(OK);
  case  HSMHV_MOD_EGIG:
    value->rValue = model->HSMHV_egig;
    return(OK);
  case  HSMHV_MOD_IGTEMP2:
    value->rValue = model->HSMHV_igtemp2;
    return(OK);
  case  HSMHV_MOD_IGTEMP3:
    value->rValue = model->HSMHV_igtemp3;
    return(OK);
  case  HSMHV_MOD_VZADD0:
    value->rValue = model->HSMHV_vzadd0;
    return(OK);
  case  HSMHV_MOD_PZADD0:
    value->rValue = model->HSMHV_pzadd0;
    return(OK);
  case  HSMHV_MOD_NFTRP:
    value->rValue = model->HSMHV_nftrp;
    return(OK);
  case  HSMHV_MOD_NFALP:
    value->rValue = model->HSMHV_nfalp;
    return(OK);
  case  HSMHV_MOD_CIT:
    value->rValue = model->HSMHV_cit;
    return(OK);
  case  HSMHV_MOD_FALPH:
    value->rValue = model->HSMHV_falph;
    return(OK);
  case  HSMHV_MOD_KAPPA:
    value->rValue = model->HSMHV_kappa;
    return(OK);
  case  HSMHV_MOD_PTHROU:
    value->rValue = model->HSMHV_pthrou;
    return(OK);
  case  HSMHV_MOD_VDIFFJ:
    value->rValue = model->HSMHV_vdiffj;
    return(OK);
  case  HSMHV_MOD_DLY1:
    value->rValue = model->HSMHV_dly1;
    return(OK);
  case  HSMHV_MOD_DLY2:
    value->rValue = model->HSMHV_dly2;
    return(OK);
  case  HSMHV_MOD_DLY3:
    value->rValue = model->HSMHV_dly3;
    return(OK);
  case  HSMHV_MOD_DLYOV:
    value->rValue = model->HSMHV_dlyov;
    return(OK);


  case  HSMHV_MOD_TNOM:
    value->rValue = model->HSMHV_tnom;
    return(OK);
  case  HSMHV_MOD_OVSLP:
    value->rValue = model->HSMHV_ovslp;
    return(OK);
  case  HSMHV_MOD_OVMAG:
    value->rValue = model->HSMHV_ovmag;
    return(OK);
  case  HSMHV_MOD_GBMIN:
    value->rValue = model->HSMHV_gbmin;
    return(OK);
  case  HSMHV_MOD_RBPB:
    value->rValue = model->HSMHV_rbpb;
    return(OK);
  case  HSMHV_MOD_RBPD:
    value->rValue = model->HSMHV_rbpd;
    return(OK);
  case  HSMHV_MOD_RBPS:
    value->rValue = model->HSMHV_rbps;
    return(OK);
  case  HSMHV_MOD_RBDB:
    value->rValue = model->HSMHV_rbdb;
    return(OK);
  case  HSMHV_MOD_RBSB:
    value->rValue = model->HSMHV_rbsb;
    return(OK);
  case  HSMHV_MOD_IBPC1:
    value->rValue = model->HSMHV_ibpc1;
    return(OK);
  case  HSMHV_MOD_IBPC2:
    value->rValue = model->HSMHV_ibpc2;
    return(OK);
  case  HSMHV_MOD_MPHDFM:
    value->rValue = model->HSMHV_mphdfm;
    return(OK);
  case  HSMHV_MOD_RDVG11:
    value->rValue = model->HSMHV_rdvg11;
    return(OK);
  case  HSMHV_MOD_RDVG12:
    value->rValue = model->HSMHV_rdvg12;
    return(OK);
  case  HSMHV_MOD_RTH0: /* Self-heating model */
    value->rValue = model->HSMHV_rth0;
    return(OK);
  case  HSMHV_MOD_CTH0: /* Self-heating model */
    value->rValue = model->HSMHV_cth0;
    return(OK);
  case  HSMHV_MOD_POWRAT: /* Self-heating model */
    value->rValue = model->HSMHV_powrat;
    return(OK);
  case  HSMHV_MOD_RTHTEMP1: /* Self-heating model */
    value->rValue = model->HSMHV_rthtemp1;
    return(OK);
  case  HSMHV_MOD_RTHTEMP2: /* Self-heating model */
    value->rValue = model->HSMHV_rthtemp2;
    return(OK);
  case  HSMHV_MOD_PRATTEMP1: /* Self-heating model */
    value->rValue = model->HSMHV_prattemp1;
    return(OK);
  case  HSMHV_MOD_PRATTEMP2: /* Self-heating model */
    value->rValue = model->HSMHV_prattemp2;
    return(OK);



  case  HSMHV_MOD_TCJBD: /* Self-heating model */
    value->rValue = model->HSMHV_tcjbd;
    return(OK);
  case  HSMHV_MOD_TCJBS: /* Self-heating model */
    value->rValue = model->HSMHV_tcjbs;
    return(OK);
  case  HSMHV_MOD_TCJBDSW: /* Self-heating model */
    value->rValue = model->HSMHV_tcjbdsw;
    return(OK);
  case  HSMHV_MOD_TCJBSSW: /* Self-heating model */
    value->rValue = model->HSMHV_tcjbssw;
    return(OK);
  case  HSMHV_MOD_TCJBDSWG: /* Self-heating model */
    value->rValue = model->HSMHV_tcjbdswg;
    return(OK);
  case  HSMHV_MOD_TCJBSSWG: /* Self-heating model */
    value->rValue = model->HSMHV_tcjbsswg;
    return(OK);
/*   case HSMHV_MOD_WTH0:                 */
/*     value->rValue = model->HSMHV_wth0; */
/*     return(OK);                       */
  case  HSMHV_MOD_QDFTVD:
    value->rValue = model->HSMHV_qdftvd;
    return(OK);
  case  HSMHV_MOD_XLDLD:
    value->rValue = model->HSMHV_xldld;
    return(OK);
  case  HSMHV_MOD_XWDLD:
    value->rValue = model->HSMHV_xwdld;
    return(OK);
  case  HSMHV_MOD_RDVD:
    value->rValue = model->HSMHV_rdvd;
    return(OK);
  case  HSMHV_MOD_RD20:
    value->rValue = model->HSMHV_rd20;
    return(OK);
  case  HSMHV_MOD_QOVSM: 
    value->rValue = model->HSMHV_qovsm;
    return(OK);
  case  HSMHV_MOD_LDRIFT: 
    value->rValue = model->HSMHV_ldrift;
    return(OK);
  case  HSMHV_MOD_RD21:
    value->rValue = model->HSMHV_rd21;
    return(OK);
  case  HSMHV_MOD_RD22:
    value->rValue = model->HSMHV_rd22;
    return(OK);
  case  HSMHV_MOD_RD22D:
    value->rValue = model->HSMHV_rd22d;
    return(OK);
  case  HSMHV_MOD_RD23:
    value->rValue = model->HSMHV_rd23;
    return(OK);
  case  HSMHV_MOD_RD24:
    value->rValue = model->HSMHV_rd24;
    return(OK);
  case  HSMHV_MOD_RD25:
    value->rValue = model->HSMHV_rd25;
    return(OK);
  case  HSMHV_MOD_RD26:
    value->rValue = model->HSMHV_rd26;
    return(OK);
  case  HSMHV_MOD_RDVDL:
    value->rValue = model->HSMHV_rdvdl;
    return(OK);
  case  HSMHV_MOD_RDVDLP:
    value->rValue = model->HSMHV_rdvdlp;
    return(OK);
  case  HSMHV_MOD_RDVDS:
    value->rValue = model->HSMHV_rdvds;
    return(OK);
  case  HSMHV_MOD_RDVDSP:
    value->rValue = model->HSMHV_rdvdsp;
    return(OK);
  case  HSMHV_MOD_RD23L:
    value->rValue = model->HSMHV_rd23l;
    return(OK);
  case  HSMHV_MOD_RD23LP:
    value->rValue = model->HSMHV_rd23lp;
    return(OK);
  case  HSMHV_MOD_RD23S:
    value->rValue = model->HSMHV_rd23s;
    return(OK);
  case  HSMHV_MOD_RD23SP:
    value->rValue = model->HSMHV_rd23sp;
    return(OK);
  case  HSMHV_MOD_RDS:
    value->rValue = model->HSMHV_rds;
    return(OK);
  case  HSMHV_MOD_RDSP:
    value->rValue = model->HSMHV_rdsp;
    return(OK);
  case  HSMHV_MOD_RDTEMP1:
    value->rValue = model->HSMHV_rdtemp1;
    return(OK);
  case  HSMHV_MOD_RDTEMP2:
    value->rValue = model->HSMHV_rdtemp2;
    return(OK);
  case  HSMHV_MOD_RTH0R:
    value->rValue = model->HSMHV_rth0r;
    return(OK);
  case  HSMHV_MOD_RDVDTEMP1:
    value->rValue = model->HSMHV_rdvdtemp1;
    return(OK);
  case  HSMHV_MOD_RDVDTEMP2:
    value->rValue = model->HSMHV_rdvdtemp2;
    return(OK);
  case  HSMHV_MOD_RTH0W:
    value->rValue = model->HSMHV_rth0w;
    return(OK);
  case  HSMHV_MOD_RTH0WP:
    value->rValue = model->HSMHV_rth0wp;
    return(OK);
  case  HSMHV_MOD_CVDSOVER:
    value->rValue = model->HSMHV_cvdsover;
    return(OK);

  case  HSMHV_MOD_NINVD:
    value->rValue = model->HSMHV_ninvd;
    return(OK);
  case  HSMHV_MOD_NINVDW:
    value->rValue = model->HSMHV_ninvdw;
    return(OK);
  case  HSMHV_MOD_NINVDWP:
    value->rValue = model->HSMHV_ninvdwp;
    return(OK);
  case  HSMHV_MOD_NINVDT1:
    value->rValue = model->HSMHV_ninvdt1;
    return(OK);
  case  HSMHV_MOD_NINVDT2:
    value->rValue = model->HSMHV_ninvdt2;
    return(OK);
  case  HSMHV_MOD_VBSMIN:
    value->rValue = model->HSMHV_vbsmin;
    return(OK);
  case  HSMHV_MOD_RDVB:
    value->rValue = model->HSMHV_rdvb;
    return(OK);
  case  HSMHV_MOD_RTH0NF:
    value->rValue = model->HSMHV_rth0nf;
    return(OK);

  case  HSMHV_MOD_RDVSUB:
    value->rValue = model->HSMHV_rdvsub;
    return(OK);
  case  HSMHV_MOD_RDVDSUB:
    value->rValue = model->HSMHV_rdvdsub;
    return(OK);
  case  HSMHV_MOD_DDRIFT:
    value->rValue = model->HSMHV_ddrift;
    return(OK);
  case  HSMHV_MOD_VBISUB:
    value->rValue = model->HSMHV_vbisub;
    return(OK);
  case  HSMHV_MOD_NSUBSUB:
    value->rValue = model->HSMHV_nsubsub;
    return(OK);
  case HSMHV_MOD_SHEMAX:
    value->rValue = model->HSMHV_shemax;
    return(OK);

  /* binning parameters */
  case  HSMHV_MOD_LMIN:
    value->rValue = model->HSMHV_lmin;
    return(OK);
  case  HSMHV_MOD_LMAX:
    value->rValue = model->HSMHV_lmax;
    return(OK);
  case  HSMHV_MOD_WMIN:
    value->rValue = model->HSMHV_wmin;
    return(OK);
  case  HSMHV_MOD_WMAX:
    value->rValue = model->HSMHV_wmax;
    return(OK);
  case  HSMHV_MOD_LBINN:
    value->rValue = model->HSMHV_lbinn;
    return(OK);
  case  HSMHV_MOD_WBINN:
    value->rValue = model->HSMHV_wbinn;
    return(OK);

  /* Length dependence */
  case  HSMHV_MOD_LVMAX:
    value->rValue = model->HSMHV_lvmax;
    return(OK);
  case  HSMHV_MOD_LBGTMP1:
    value->rValue = model->HSMHV_lbgtmp1;
    return(OK);
  case  HSMHV_MOD_LBGTMP2:
    value->rValue = model->HSMHV_lbgtmp2;
    return(OK);
  case  HSMHV_MOD_LEG0:
    value->rValue = model->HSMHV_leg0;
    return(OK);
  case  HSMHV_MOD_LVFBOVER:
    value->rValue = model->HSMHV_lvfbover;
    return(OK);
  case  HSMHV_MOD_LNOVER:
    value->rValue = model->HSMHV_lnover;
    return(OK);
  case  HSMHV_MOD_LNOVERS:
    value->rValue = model->HSMHV_lnovers;
    return(OK);
  case  HSMHV_MOD_LWL2:
    value->rValue = model->HSMHV_lwl2;
    return(OK);
  case  HSMHV_MOD_LVFBC:
    value->rValue = model->HSMHV_lvfbc;
    return(OK);
  case  HSMHV_MOD_LNSUBC:
    value->rValue = model->HSMHV_lnsubc;
    return(OK);
  case  HSMHV_MOD_LNSUBP:
    value->rValue = model->HSMHV_lnsubp;
    return(OK);
  case  HSMHV_MOD_LSCP1:
    value->rValue = model->HSMHV_lscp1;
    return(OK);
  case  HSMHV_MOD_LSCP2:
    value->rValue = model->HSMHV_lscp2;
    return(OK);
  case  HSMHV_MOD_LSCP3:
    value->rValue = model->HSMHV_lscp3;
    return(OK);
  case  HSMHV_MOD_LSC1:
    value->rValue = model->HSMHV_lsc1;
    return(OK);
  case  HSMHV_MOD_LSC2:
    value->rValue = model->HSMHV_lsc2;
    return(OK);
  case  HSMHV_MOD_LSC3:
    value->rValue = model->HSMHV_lsc3;
    return(OK);
  case  HSMHV_MOD_LPGD1:
    value->rValue = model->HSMHV_lpgd1;
    return(OK);
  case  HSMHV_MOD_LPGD3:
    value->rValue = model->HSMHV_lpgd3;
    return(OK);
  case  HSMHV_MOD_LNDEP:
    value->rValue = model->HSMHV_lndep;
    return(OK);
  case  HSMHV_MOD_LNINV:
    value->rValue = model->HSMHV_lninv;
    return(OK);
  case  HSMHV_MOD_LMUECB0:
    value->rValue = model->HSMHV_lmuecb0;
    return(OK);
  case  HSMHV_MOD_LMUECB1:
    value->rValue = model->HSMHV_lmuecb1;
    return(OK);
  case  HSMHV_MOD_LMUEPH1:
    value->rValue = model->HSMHV_lmueph1;
    return(OK);
  case  HSMHV_MOD_LVTMP:
    value->rValue = model->HSMHV_lvtmp;
    return(OK);
  case  HSMHV_MOD_LWVTH0:
    value->rValue = model->HSMHV_lwvth0;
    return(OK);
  case  HSMHV_MOD_LMUESR1:
    value->rValue = model->HSMHV_lmuesr1;
    return(OK);
  case  HSMHV_MOD_LMUETMP:
    value->rValue = model->HSMHV_lmuetmp;
    return(OK);
  case  HSMHV_MOD_LSUB1:
    value->rValue = model->HSMHV_lsub1;
    return(OK);
  case  HSMHV_MOD_LSUB2:
    value->rValue = model->HSMHV_lsub2;
    return(OK);
  case  HSMHV_MOD_LSVDS:
    value->rValue = model->HSMHV_lsvds;
    return(OK);
  case  HSMHV_MOD_LSVBS:
    value->rValue = model->HSMHV_lsvbs;
    return(OK);
  case  HSMHV_MOD_LSVGS:
    value->rValue = model->HSMHV_lsvgs;
    return(OK);
  case  HSMHV_MOD_LFN1:
    value->rValue = model->HSMHV_lfn1;
    return(OK);
  case  HSMHV_MOD_LFN2:
    value->rValue = model->HSMHV_lfn2;
    return(OK);
  case  HSMHV_MOD_LFN3:
    value->rValue = model->HSMHV_lfn3;
    return(OK);
  case  HSMHV_MOD_LFVBS:
    value->rValue = model->HSMHV_lfvbs;
    return(OK);
  case  HSMHV_MOD_LNSTI:
    value->rValue = model->HSMHV_lnsti;
    return(OK);
  case  HSMHV_MOD_LWSTI:
    value->rValue = model->HSMHV_lwsti;
    return(OK);
  case  HSMHV_MOD_LSCSTI1:
    value->rValue = model->HSMHV_lscsti1;
    return(OK);
  case  HSMHV_MOD_LSCSTI2:
    value->rValue = model->HSMHV_lscsti2;
    return(OK);
  case  HSMHV_MOD_LVTHSTI:
    value->rValue = model->HSMHV_lvthsti;
    return(OK);
  case  HSMHV_MOD_LMUESTI1:
    value->rValue = model->HSMHV_lmuesti1;
    return(OK);
  case  HSMHV_MOD_LMUESTI2:
    value->rValue = model->HSMHV_lmuesti2;
    return(OK);
  case  HSMHV_MOD_LMUESTI3:
    value->rValue = model->HSMHV_lmuesti3;
    return(OK);
  case  HSMHV_MOD_LNSUBPSTI1:
    value->rValue = model->HSMHV_lnsubpsti1;
    return(OK);
  case  HSMHV_MOD_LNSUBPSTI2:
    value->rValue = model->HSMHV_lnsubpsti2;
    return(OK);
  case  HSMHV_MOD_LNSUBPSTI3:
    value->rValue = model->HSMHV_lnsubpsti3;
    return(OK);
  case  HSMHV_MOD_LCGSO:
    value->rValue = model->HSMHV_lcgso;
    return(OK);
  case  HSMHV_MOD_LCGDO:
    value->rValue = model->HSMHV_lcgdo;
    return(OK);
  case  HSMHV_MOD_LJS0:
    value->rValue = model->HSMHV_ljs0;
    return(OK);
  case  HSMHV_MOD_LJS0SW:
    value->rValue = model->HSMHV_ljs0sw;
    return(OK);
  case  HSMHV_MOD_LNJ:
    value->rValue = model->HSMHV_lnj;
    return(OK);
  case  HSMHV_MOD_LCISBK:
    value->rValue = model->HSMHV_lcisbk;
    return(OK);
  case  HSMHV_MOD_LCLM1:
    value->rValue = model->HSMHV_lclm1;
    return(OK);
  case  HSMHV_MOD_LCLM2:
    value->rValue = model->HSMHV_lclm2;
    return(OK);
  case  HSMHV_MOD_LCLM3:
    value->rValue = model->HSMHV_lclm3;
    return(OK);
  case  HSMHV_MOD_LWFC:
    value->rValue = model->HSMHV_lwfc;
    return(OK);
  case  HSMHV_MOD_LGIDL1:
    value->rValue = model->HSMHV_lgidl1;
    return(OK);
  case  HSMHV_MOD_LGIDL2:
    value->rValue = model->HSMHV_lgidl2;
    return(OK);
  case  HSMHV_MOD_LGLEAK1:
    value->rValue = model->HSMHV_lgleak1;
    return(OK);
  case  HSMHV_MOD_LGLEAK2:
    value->rValue = model->HSMHV_lgleak2;
    return(OK);
  case  HSMHV_MOD_LGLEAK3:
    value->rValue = model->HSMHV_lgleak3;
    return(OK);
  case  HSMHV_MOD_LGLEAK6:
    value->rValue = model->HSMHV_lgleak6;
    return(OK);
  case  HSMHV_MOD_LGLKSD1:
    value->rValue = model->HSMHV_lglksd1;
    return(OK);
  case  HSMHV_MOD_LGLKSD2:
    value->rValue = model->HSMHV_lglksd2;
    return(OK);
  case  HSMHV_MOD_LGLKB1:
    value->rValue = model->HSMHV_lglkb1;
    return(OK);
  case  HSMHV_MOD_LGLKB2:
    value->rValue = model->HSMHV_lglkb2;
    return(OK);
  case  HSMHV_MOD_LNFTRP:
    value->rValue = model->HSMHV_lnftrp;
    return(OK);
  case  HSMHV_MOD_LNFALP:
    value->rValue = model->HSMHV_lnfalp;
    return(OK);
  case  HSMHV_MOD_LPTHROU:
    value->rValue = model->HSMHV_lpthrou;
    return(OK);
  case  HSMHV_MOD_LVDIFFJ:
    value->rValue = model->HSMHV_lvdiffj;
    return(OK);
  case  HSMHV_MOD_LIBPC1:
    value->rValue = model->HSMHV_libpc1;
    return(OK);
  case  HSMHV_MOD_LIBPC2:
    value->rValue = model->HSMHV_libpc2;
    return(OK);
  case  HSMHV_MOD_LCGBO:
    value->rValue = model->HSMHV_lcgbo;
    return(OK);
  case  HSMHV_MOD_LCVDSOVER:
    value->rValue = model->HSMHV_lcvdsover;
    return(OK);
  case  HSMHV_MOD_LFALPH:
    value->rValue = model->HSMHV_lfalph;
    return(OK);
  case  HSMHV_MOD_LNPEXT:
    value->rValue = model->HSMHV_lnpext;
    return(OK);
  case  HSMHV_MOD_LPOWRAT:
    value->rValue = model->HSMHV_lpowrat;
    return(OK);
  case  HSMHV_MOD_LRD:
    value->rValue = model->HSMHV_lrd;
    return(OK);
  case  HSMHV_MOD_LRD22:
    value->rValue = model->HSMHV_lrd22;
    return(OK);
  case  HSMHV_MOD_LRD23:
    value->rValue = model->HSMHV_lrd23;
    return(OK);
  case  HSMHV_MOD_LRD24:
    value->rValue = model->HSMHV_lrd24;
    return(OK);
  case  HSMHV_MOD_LRDICT1:
    value->rValue = model->HSMHV_lrdict1;
    return(OK);
  case  HSMHV_MOD_LRDOV13:
    value->rValue = model->HSMHV_lrdov13;
    return(OK);
  case  HSMHV_MOD_LRDSLP1:
    value->rValue = model->HSMHV_lrdslp1;
    return(OK);
  case  HSMHV_MOD_LRDVB:
    value->rValue = model->HSMHV_lrdvb;
    return(OK);
  case  HSMHV_MOD_LRDVD:
    value->rValue = model->HSMHV_lrdvd;
    return(OK);
  case  HSMHV_MOD_LRDVG11:
    value->rValue = model->HSMHV_lrdvg11;
    return(OK);
  case  HSMHV_MOD_LRS:
    value->rValue = model->HSMHV_lrs;
    return(OK);
  case  HSMHV_MOD_LRTH0:
    value->rValue = model->HSMHV_lrth0;
    return(OK);
  case  HSMHV_MOD_LVOVER:
    value->rValue = model->HSMHV_lvover;
    return(OK);

  /* Width dependence */
  case  HSMHV_MOD_WVMAX:
    value->rValue = model->HSMHV_wvmax;
    return(OK);
  case  HSMHV_MOD_WBGTMP1:
    value->rValue = model->HSMHV_wbgtmp1;
    return(OK);
  case  HSMHV_MOD_WBGTMP2:
    value->rValue = model->HSMHV_wbgtmp2;
    return(OK);
  case  HSMHV_MOD_WEG0:
    value->rValue = model->HSMHV_weg0;
    return(OK);
  case  HSMHV_MOD_WVFBOVER:
    value->rValue = model->HSMHV_wvfbover;
    return(OK);
  case  HSMHV_MOD_WNOVER:
    value->rValue = model->HSMHV_wnover;
    return(OK);
  case  HSMHV_MOD_WNOVERS:
    value->rValue = model->HSMHV_wnovers;
    return(OK);
  case  HSMHV_MOD_WWL2:
    value->rValue = model->HSMHV_wwl2;
    return(OK);
  case  HSMHV_MOD_WVFBC:
    value->rValue = model->HSMHV_wvfbc;
    return(OK);
  case  HSMHV_MOD_WNSUBC:
    value->rValue = model->HSMHV_wnsubc;
    return(OK);
  case  HSMHV_MOD_WNSUBP:
    value->rValue = model->HSMHV_wnsubp;
    return(OK);
  case  HSMHV_MOD_WSCP1:
    value->rValue = model->HSMHV_wscp1;
    return(OK);
  case  HSMHV_MOD_WSCP2:
    value->rValue = model->HSMHV_wscp2;
    return(OK);
  case  HSMHV_MOD_WSCP3:
    value->rValue = model->HSMHV_wscp3;
    return(OK);
  case  HSMHV_MOD_WSC1:
    value->rValue = model->HSMHV_wsc1;
    return(OK);
  case  HSMHV_MOD_WSC2:
    value->rValue = model->HSMHV_wsc2;
    return(OK);
  case  HSMHV_MOD_WSC3:
    value->rValue = model->HSMHV_wsc3;
    return(OK);
  case  HSMHV_MOD_WPGD1:
    value->rValue = model->HSMHV_wpgd1;
    return(OK);
  case  HSMHV_MOD_WPGD3:
    value->rValue = model->HSMHV_wpgd3;
    return(OK);
  case  HSMHV_MOD_WNDEP:
    value->rValue = model->HSMHV_wndep;
    return(OK);
  case  HSMHV_MOD_WNINV:
    value->rValue = model->HSMHV_wninv;
    return(OK);
  case  HSMHV_MOD_WMUECB0:
    value->rValue = model->HSMHV_wmuecb0;
    return(OK);
  case  HSMHV_MOD_WMUECB1:
    value->rValue = model->HSMHV_wmuecb1;
    return(OK);
  case  HSMHV_MOD_WMUEPH1:
    value->rValue = model->HSMHV_wmueph1;
    return(OK);
  case  HSMHV_MOD_WVTMP:
    value->rValue = model->HSMHV_wvtmp;
    return(OK);
  case  HSMHV_MOD_WWVTH0:
    value->rValue = model->HSMHV_wwvth0;
    return(OK);
  case  HSMHV_MOD_WMUESR1:
    value->rValue = model->HSMHV_wmuesr1;
    return(OK);
  case  HSMHV_MOD_WMUETMP:
    value->rValue = model->HSMHV_wmuetmp;
    return(OK);
  case  HSMHV_MOD_WSUB1:
    value->rValue = model->HSMHV_wsub1;
    return(OK);
  case  HSMHV_MOD_WSUB2:
    value->rValue = model->HSMHV_wsub2;
    return(OK);
  case  HSMHV_MOD_WSVDS:
    value->rValue = model->HSMHV_wsvds;
    return(OK);
  case  HSMHV_MOD_WSVBS:
    value->rValue = model->HSMHV_wsvbs;
    return(OK);
  case  HSMHV_MOD_WSVGS:
    value->rValue = model->HSMHV_wsvgs;
    return(OK);
  case  HSMHV_MOD_WFN1:
    value->rValue = model->HSMHV_wfn1;
    return(OK);
  case  HSMHV_MOD_WFN2:
    value->rValue = model->HSMHV_wfn2;
    return(OK);
  case  HSMHV_MOD_WFN3:
    value->rValue = model->HSMHV_wfn3;
    return(OK);
  case  HSMHV_MOD_WFVBS:
    value->rValue = model->HSMHV_wfvbs;
    return(OK);
  case  HSMHV_MOD_WNSTI:
    value->rValue = model->HSMHV_wnsti;
    return(OK);
  case  HSMHV_MOD_WWSTI:
    value->rValue = model->HSMHV_wwsti;
    return(OK);
  case  HSMHV_MOD_WSCSTI1:
    value->rValue = model->HSMHV_wscsti1;
    return(OK);
  case  HSMHV_MOD_WSCSTI2:
    value->rValue = model->HSMHV_wscsti2;
    return(OK);
  case  HSMHV_MOD_WVTHSTI:
    value->rValue = model->HSMHV_wvthsti;
    return(OK);
  case  HSMHV_MOD_WMUESTI1:
    value->rValue = model->HSMHV_wmuesti1;
    return(OK);
  case  HSMHV_MOD_WMUESTI2:
    value->rValue = model->HSMHV_wmuesti2;
    return(OK);
  case  HSMHV_MOD_WMUESTI3:
    value->rValue = model->HSMHV_wmuesti3;
    return(OK);
  case  HSMHV_MOD_WNSUBPSTI1:
    value->rValue = model->HSMHV_wnsubpsti1;
    return(OK);
  case  HSMHV_MOD_WNSUBPSTI2:
    value->rValue = model->HSMHV_wnsubpsti2;
    return(OK);
  case  HSMHV_MOD_WNSUBPSTI3:
    value->rValue = model->HSMHV_wnsubpsti3;
    return(OK);
  case  HSMHV_MOD_WCGSO:
    value->rValue = model->HSMHV_wcgso;
    return(OK);
  case  HSMHV_MOD_WCGDO:
    value->rValue = model->HSMHV_wcgdo;
    return(OK);
  case  HSMHV_MOD_WJS0:
    value->rValue = model->HSMHV_wjs0;
    return(OK);
  case  HSMHV_MOD_WJS0SW:
    value->rValue = model->HSMHV_wjs0sw;
    return(OK);
  case  HSMHV_MOD_WNJ:
    value->rValue = model->HSMHV_wnj;
    return(OK);
  case  HSMHV_MOD_WCISBK:
    value->rValue = model->HSMHV_wcisbk;
    return(OK);
  case  HSMHV_MOD_WCLM1:
    value->rValue = model->HSMHV_wclm1;
    return(OK);
  case  HSMHV_MOD_WCLM2:
    value->rValue = model->HSMHV_wclm2;
    return(OK);
  case  HSMHV_MOD_WCLM3:
    value->rValue = model->HSMHV_wclm3;
    return(OK);
  case  HSMHV_MOD_WWFC:
    value->rValue = model->HSMHV_wwfc;
    return(OK);
  case  HSMHV_MOD_WGIDL1:
    value->rValue = model->HSMHV_wgidl1;
    return(OK);
  case  HSMHV_MOD_WGIDL2:
    value->rValue = model->HSMHV_wgidl2;
    return(OK);
  case  HSMHV_MOD_WGLEAK1:
    value->rValue = model->HSMHV_wgleak1;
    return(OK);
  case  HSMHV_MOD_WGLEAK2:
    value->rValue = model->HSMHV_wgleak2;
    return(OK);
  case  HSMHV_MOD_WGLEAK3:
    value->rValue = model->HSMHV_wgleak3;
    return(OK);
  case  HSMHV_MOD_WGLEAK6:
    value->rValue = model->HSMHV_wgleak6;
    return(OK);
  case  HSMHV_MOD_WGLKSD1:
    value->rValue = model->HSMHV_wglksd1;
    return(OK);
  case  HSMHV_MOD_WGLKSD2:
    value->rValue = model->HSMHV_wglksd2;
    return(OK);
  case  HSMHV_MOD_WGLKB1:
    value->rValue = model->HSMHV_wglkb1;
    return(OK);
  case  HSMHV_MOD_WGLKB2:
    value->rValue = model->HSMHV_wglkb2;
    return(OK);
  case  HSMHV_MOD_WNFTRP:
    value->rValue = model->HSMHV_wnftrp;
    return(OK);
  case  HSMHV_MOD_WNFALP:
    value->rValue = model->HSMHV_wnfalp;
    return(OK);
  case  HSMHV_MOD_WPTHROU:
    value->rValue = model->HSMHV_wpthrou;
    return(OK);
  case  HSMHV_MOD_WVDIFFJ:
    value->rValue = model->HSMHV_wvdiffj;
    return(OK);
  case  HSMHV_MOD_WIBPC1:
    value->rValue = model->HSMHV_wibpc1;
    return(OK);
  case  HSMHV_MOD_WIBPC2:
    value->rValue = model->HSMHV_wibpc2;
    return(OK);
  case  HSMHV_MOD_WCGBO:
    value->rValue = model->HSMHV_wcgbo;
    return(OK);
  case  HSMHV_MOD_WCVDSOVER:
    value->rValue = model->HSMHV_wcvdsover;
    return(OK);
  case  HSMHV_MOD_WFALPH:
    value->rValue = model->HSMHV_wfalph;
    return(OK);
  case  HSMHV_MOD_WNPEXT:
    value->rValue = model->HSMHV_wnpext;
    return(OK);
  case  HSMHV_MOD_WPOWRAT:
    value->rValue = model->HSMHV_wpowrat;
    return(OK);
  case  HSMHV_MOD_WRD:
    value->rValue = model->HSMHV_wrd;
    return(OK);
  case  HSMHV_MOD_WRD22:
    value->rValue = model->HSMHV_wrd22;
    return(OK);
  case  HSMHV_MOD_WRD23:
    value->rValue = model->HSMHV_wrd23;
    return(OK);
  case  HSMHV_MOD_WRD24:
    value->rValue = model->HSMHV_wrd24;
    return(OK);
  case  HSMHV_MOD_WRDICT1:
    value->rValue = model->HSMHV_wrdict1;
    return(OK);
  case  HSMHV_MOD_WRDOV13:
    value->rValue = model->HSMHV_wrdov13;
    return(OK);
  case  HSMHV_MOD_WRDSLP1:
    value->rValue = model->HSMHV_wrdslp1;
    return(OK);
  case  HSMHV_MOD_WRDVB:
    value->rValue = model->HSMHV_wrdvb;
    return(OK);
  case  HSMHV_MOD_WRDVD:
    value->rValue = model->HSMHV_wrdvd;
    return(OK);
  case  HSMHV_MOD_WRDVG11:
    value->rValue = model->HSMHV_wrdvg11;
    return(OK);
  case  HSMHV_MOD_WRS:
    value->rValue = model->HSMHV_wrs;
    return(OK);
  case  HSMHV_MOD_WRTH0:
    value->rValue = model->HSMHV_wrth0;
    return(OK);
  case  HSMHV_MOD_WVOVER:
    value->rValue = model->HSMHV_wvover;
    return(OK);

  /* Cross-term dependence */
  case  HSMHV_MOD_PVMAX:
    value->rValue = model->HSMHV_pvmax;
    return(OK);
  case  HSMHV_MOD_PBGTMP1:
    value->rValue = model->HSMHV_pbgtmp1;
    return(OK);
  case  HSMHV_MOD_PBGTMP2:
    value->rValue = model->HSMHV_pbgtmp2;
    return(OK);
  case  HSMHV_MOD_PEG0:
    value->rValue = model->HSMHV_peg0;
    return(OK);
  case  HSMHV_MOD_PVFBOVER:
    value->rValue = model->HSMHV_pvfbover;
    return(OK);
  case  HSMHV_MOD_PNOVER:
    value->rValue = model->HSMHV_pnover;
    return(OK);
  case  HSMHV_MOD_PNOVERS:
    value->rValue = model->HSMHV_pnovers;
    return(OK);
  case  HSMHV_MOD_PWL2:
    value->rValue = model->HSMHV_pwl2;
    return(OK);
  case  HSMHV_MOD_PVFBC:
    value->rValue = model->HSMHV_pvfbc;
    return(OK);
  case  HSMHV_MOD_PNSUBC:
    value->rValue = model->HSMHV_pnsubc;
    return(OK);
  case  HSMHV_MOD_PNSUBP:
    value->rValue = model->HSMHV_pnsubp;
    return(OK);
  case  HSMHV_MOD_PSCP1:
    value->rValue = model->HSMHV_pscp1;
    return(OK);
  case  HSMHV_MOD_PSCP2:
    value->rValue = model->HSMHV_pscp2;
    return(OK);
  case  HSMHV_MOD_PSCP3:
    value->rValue = model->HSMHV_pscp3;
    return(OK);
  case  HSMHV_MOD_PSC1:
    value->rValue = model->HSMHV_psc1;
    return(OK);
  case  HSMHV_MOD_PSC2:
    value->rValue = model->HSMHV_psc2;
    return(OK);
  case  HSMHV_MOD_PSC3:
    value->rValue = model->HSMHV_psc3;
    return(OK);
  case  HSMHV_MOD_PPGD1:
    value->rValue = model->HSMHV_ppgd1;
    return(OK);
  case  HSMHV_MOD_PPGD3:
    value->rValue = model->HSMHV_ppgd3;
    return(OK);
  case  HSMHV_MOD_PNDEP:
    value->rValue = model->HSMHV_pndep;
    return(OK);
  case  HSMHV_MOD_PNINV:
    value->rValue = model->HSMHV_pninv;
    return(OK);
  case  HSMHV_MOD_PMUECB0:
    value->rValue = model->HSMHV_pmuecb0;
    return(OK);
  case  HSMHV_MOD_PMUECB1:
    value->rValue = model->HSMHV_pmuecb1;
    return(OK);
  case  HSMHV_MOD_PMUEPH1:
    value->rValue = model->HSMHV_pmueph1;
    return(OK);
  case  HSMHV_MOD_PVTMP:
    value->rValue = model->HSMHV_pvtmp;
    return(OK);
  case  HSMHV_MOD_PWVTH0:
    value->rValue = model->HSMHV_pwvth0;
    return(OK);
  case  HSMHV_MOD_PMUESR1:
    value->rValue = model->HSMHV_pmuesr1;
    return(OK);
  case  HSMHV_MOD_PMUETMP:
    value->rValue = model->HSMHV_pmuetmp;
    return(OK);
  case  HSMHV_MOD_PSUB1:
    value->rValue = model->HSMHV_psub1;
    return(OK);
  case  HSMHV_MOD_PSUB2:
    value->rValue = model->HSMHV_psub2;
    return(OK);
  case  HSMHV_MOD_PSVDS:
    value->rValue = model->HSMHV_psvds;
    return(OK);
  case  HSMHV_MOD_PSVBS:
    value->rValue = model->HSMHV_psvbs;
    return(OK);
  case  HSMHV_MOD_PSVGS:
    value->rValue = model->HSMHV_psvgs;
    return(OK);
  case  HSMHV_MOD_PFN1:
    value->rValue = model->HSMHV_pfn1;
    return(OK);
  case  HSMHV_MOD_PFN2:
    value->rValue = model->HSMHV_pfn2;
    return(OK);
  case  HSMHV_MOD_PFN3:
    value->rValue = model->HSMHV_pfn3;
    return(OK);
  case  HSMHV_MOD_PFVBS:
    value->rValue = model->HSMHV_pfvbs;
    return(OK);
  case  HSMHV_MOD_PNSTI:
    value->rValue = model->HSMHV_pnsti;
    return(OK);
  case  HSMHV_MOD_PWSTI:
    value->rValue = model->HSMHV_pwsti;
    return(OK);
  case  HSMHV_MOD_PSCSTI1:
    value->rValue = model->HSMHV_pscsti1;
    return(OK);
  case  HSMHV_MOD_PSCSTI2:
    value->rValue = model->HSMHV_pscsti2;
    return(OK);
  case  HSMHV_MOD_PVTHSTI:
    value->rValue = model->HSMHV_pvthsti;
    return(OK);
  case  HSMHV_MOD_PMUESTI1:
    value->rValue = model->HSMHV_pmuesti1;
    return(OK);
  case  HSMHV_MOD_PMUESTI2:
    value->rValue = model->HSMHV_pmuesti2;
    return(OK);
  case  HSMHV_MOD_PMUESTI3:
    value->rValue = model->HSMHV_pmuesti3;
    return(OK);
  case  HSMHV_MOD_PNSUBPSTI1:
    value->rValue = model->HSMHV_pnsubpsti1;
    return(OK);
  case  HSMHV_MOD_PNSUBPSTI2:
    value->rValue = model->HSMHV_pnsubpsti2;
    return(OK);
  case  HSMHV_MOD_PNSUBPSTI3:
    value->rValue = model->HSMHV_pnsubpsti3;
    return(OK);
  case  HSMHV_MOD_PCGSO:
    value->rValue = model->HSMHV_pcgso;
    return(OK);
  case  HSMHV_MOD_PCGDO:
    value->rValue = model->HSMHV_pcgdo;
    return(OK);
  case  HSMHV_MOD_PJS0:
    value->rValue = model->HSMHV_pjs0;
    return(OK);
  case  HSMHV_MOD_PJS0SW:
    value->rValue = model->HSMHV_pjs0sw;
    return(OK);
  case  HSMHV_MOD_PNJ:
    value->rValue = model->HSMHV_pnj;
    return(OK);
  case  HSMHV_MOD_PCISBK:
    value->rValue = model->HSMHV_pcisbk;
    return(OK);
  case  HSMHV_MOD_PCLM1:
    value->rValue = model->HSMHV_pclm1;
    return(OK);
  case  HSMHV_MOD_PCLM2:
    value->rValue = model->HSMHV_pclm2;
    return(OK);
  case  HSMHV_MOD_PCLM3:
    value->rValue = model->HSMHV_pclm3;
    return(OK);
  case  HSMHV_MOD_PWFC:
    value->rValue = model->HSMHV_pwfc;
    return(OK);
  case  HSMHV_MOD_PGIDL1:
    value->rValue = model->HSMHV_pgidl1;
    return(OK);
  case  HSMHV_MOD_PGIDL2:
    value->rValue = model->HSMHV_pgidl2;
    return(OK);
  case  HSMHV_MOD_PGLEAK1:
    value->rValue = model->HSMHV_pgleak1;
    return(OK);
  case  HSMHV_MOD_PGLEAK2:
    value->rValue = model->HSMHV_pgleak2;
    return(OK);
  case  HSMHV_MOD_PGLEAK3:
    value->rValue = model->HSMHV_pgleak3;
    return(OK);
  case  HSMHV_MOD_PGLEAK6:
    value->rValue = model->HSMHV_pgleak6;
    return(OK);
  case  HSMHV_MOD_PGLKSD1:
    value->rValue = model->HSMHV_pglksd1;
    return(OK);
  case  HSMHV_MOD_PGLKSD2:
    value->rValue = model->HSMHV_pglksd2;
    return(OK);
  case  HSMHV_MOD_PGLKB1:
    value->rValue = model->HSMHV_pglkb1;
    return(OK);
  case  HSMHV_MOD_PGLKB2:
    value->rValue = model->HSMHV_pglkb2;
    return(OK);
  case  HSMHV_MOD_PNFTRP:
    value->rValue = model->HSMHV_pnftrp;
    return(OK);
  case  HSMHV_MOD_PNFALP:
    value->rValue = model->HSMHV_pnfalp;
    return(OK);
  case  HSMHV_MOD_PPTHROU:
    value->rValue = model->HSMHV_ppthrou;
    return(OK);
  case  HSMHV_MOD_PVDIFFJ:
    value->rValue = model->HSMHV_pvdiffj;
    return(OK);
  case  HSMHV_MOD_PIBPC1:
    value->rValue = model->HSMHV_pibpc1;
    return(OK);
  case  HSMHV_MOD_PIBPC2:
    value->rValue = model->HSMHV_pibpc2;
    return(OK);
  case  HSMHV_MOD_PCGBO:
    value->rValue = model->HSMHV_pcgbo;
    return(OK);
  case  HSMHV_MOD_PCVDSOVER:
    value->rValue = model->HSMHV_pcvdsover;
    return(OK);
  case  HSMHV_MOD_PFALPH:
    value->rValue = model->HSMHV_pfalph;
    return(OK);
  case  HSMHV_MOD_PNPEXT:
    value->rValue = model->HSMHV_pnpext;
    return(OK);
  case  HSMHV_MOD_PPOWRAT:
    value->rValue = model->HSMHV_ppowrat;
    return(OK);
  case  HSMHV_MOD_PRD:
    value->rValue = model->HSMHV_prd;
    return(OK);
  case  HSMHV_MOD_PRD22:
    value->rValue = model->HSMHV_prd22;
    return(OK);
  case  HSMHV_MOD_PRD23:
    value->rValue = model->HSMHV_prd23;
    return(OK);
  case  HSMHV_MOD_PRD24:
    value->rValue = model->HSMHV_prd24;
    return(OK);
  case  HSMHV_MOD_PRDICT1:
    value->rValue = model->HSMHV_prdict1;
    return(OK);
  case  HSMHV_MOD_PRDOV13:
    value->rValue = model->HSMHV_prdov13;
    return(OK);
  case  HSMHV_MOD_PRDSLP1:
    value->rValue = model->HSMHV_prdslp1;
    return(OK);
  case  HSMHV_MOD_PRDVB:
    value->rValue = model->HSMHV_prdvb;
    return(OK);
  case  HSMHV_MOD_PRDVD:
    value->rValue = model->HSMHV_prdvd;
    return(OK);
  case  HSMHV_MOD_PRDVG11:
    value->rValue = model->HSMHV_prdvg11;
    return(OK);
  case  HSMHV_MOD_PRS:
    value->rValue = model->HSMHV_prs;
    return(OK);
  case  HSMHV_MOD_PRTH0:
    value->rValue = model->HSMHV_prth0;
    return(OK);
  case  HSMHV_MOD_PVOVER:
    value->rValue = model->HSMHV_pvover;
    return(OK);

  case HSMHV_MOD_VGS_MAX:
    value->rValue = model->HSMHVvgsMax;
    return(OK);
  case HSMHV_MOD_VGD_MAX:
    value->rValue = model->HSMHVvgdMax;
    return(OK);
  case HSMHV_MOD_VGB_MAX:
    value->rValue = model->HSMHVvgbMax;
    return(OK);
  case HSMHV_MOD_VDS_MAX:
    value->rValue = model->HSMHVvdsMax;
    return(OK);
  case HSMHV_MOD_VBS_MAX:
    value->rValue = model->HSMHVvbsMax;
    return(OK);
  case HSMHV_MOD_VBD_MAX:
    value->rValue = model->HSMHVvbdMax;
    return(OK);
  case HSMHV_MOD_VGSR_MAX:
      value->rValue = model->HSMHVvgsrMax;
      return(OK);
  case HSMHV_MOD_VGDR_MAX:
      value->rValue = model->HSMHVvgdrMax;
      return(OK);
  case HSMHV_MOD_VGBR_MAX:
      value->rValue = model->HSMHVvgbrMax;
      return(OK);
  case HSMHV_MOD_VBSR_MAX:
      value->rValue = model->HSMHVvbsrMax;
      return(OK);
  case HSMHV_MOD_VBDR_MAX:
      value->rValue = model->HSMHVvbdrMax;
      return(OK);

  default:
    return(E_BADPARM);
  }
  /* NOTREACHED */
}
