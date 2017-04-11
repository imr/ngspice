/***********************************************************************

 HiSIM (Hiroshima University STARC IGFET Model)
 Copyright (C) 2014 Hiroshima University & STARC

 MODEL NAME : HiSIM
 ( VERSION : 2  SUBVERSION : 8  REVISION : 0 )
 
 FILE : hsm2mask.c

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
#include "ngspice/cktdefs.h"
#include "ngspice/devdefs.h"
#include "hsm2def.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

int HSM2mAsk(
     CKTcircuit *ckt,
     GENmodel *inst,
     int which,
     IFvalue *value)
{
  HSM2model *model = (HSM2model *)inst;

  NG_IGNORE(ckt);

  switch (which) {
  case HSM2_MOD_NMOS:
    value->iValue = model->HSM2_type;
    return(OK);
  case  HSM2_MOD_PMOS:
    value->iValue = model->HSM2_type;
    return(OK);
  case  HSM2_MOD_LEVEL:
    value->iValue = model->HSM2_level;
    return(OK);
  case  HSM2_MOD_INFO:
    value->iValue = model->HSM2_info;
    return(OK);
  case HSM2_MOD_NOISE:
    value->iValue = model->HSM2_noise;
    return(OK);
  case HSM2_MOD_VERSION:
    value->iValue = model->HSM2_version;
    return(OK);
  case HSM2_MOD_SHOW:
    value->iValue = model->HSM2_show;
    return(OK);
  case  HSM2_MOD_CORSRD:
    value->iValue = model->HSM2_corsrd;
    return(OK);
  case  HSM2_MOD_CORG:
    value->iValue = model->HSM2_corg;
    return(OK);
  case  HSM2_MOD_COIPRV:
    value->iValue = model->HSM2_coiprv;
    return(OK);
  case  HSM2_MOD_COPPRV:
    value->iValue = model->HSM2_copprv;
    return(OK);
  case  HSM2_MOD_COADOV:
    value->iValue = model->HSM2_coadov;
    return(OK);
  case  HSM2_MOD_COISUB:
    value->iValue = model->HSM2_coisub;
    return(OK);
  case  HSM2_MOD_COIIGS:
    value->iValue = model->HSM2_coiigs;
    return(OK);
  case  HSM2_MOD_COGIDL:
    value->iValue = model->HSM2_cogidl;
    return(OK);
  case  HSM2_MOD_COOVLP:
    value->iValue = model->HSM2_coovlp;
    return(OK);
  case  HSM2_MOD_COFLICK:
    value->iValue = model->HSM2_coflick;
    return(OK);
  case  HSM2_MOD_COISTI:
    value->iValue = model->HSM2_coisti;
    return(OK);
  case  HSM2_MOD_CONQS:
    value->iValue = model->HSM2_conqs;
    return(OK);
  case  HSM2_MOD_CORBNET:
    value->iValue = model->HSM2_corbnet;
    return(OK);
  case  HSM2_MOD_COTHRML:
    value->iValue = model->HSM2_cothrml;
    return(OK);
  case  HSM2_MOD_COIGN:
    value->iValue = model->HSM2_coign;
    return(OK);
  case  HSM2_MOD_CODFM:
    value->iValue = model->HSM2_codfm;
    return(OK);
  case  HSM2_MOD_CORECIP:
    value->iValue = model->HSM2_corecip;
    return(OK);
  case  HSM2_MOD_COQY:
    value->iValue = model->HSM2_coqy;
    return(OK);
  case  HSM2_MOD_COQOVSM:
    value->iValue = model->HSM2_coqovsm;
    return(OK);
  case HSM2_MOD_COERRREP:
    value->iValue = model->HSM2_coerrrep;
    return(OK);
  case  HSM2_MOD_CODEP:
    value->iValue = model->HSM2_codep;
    return(OK);
  case HSM2_MOD_CODDLT:
    value->iValue = model->HSM2_coddlt;
    return(OK);

  case  HSM2_MOD_VMAX:
    value->rValue = model->HSM2_vmax;
    return(OK);
  case  HSM2_MOD_BGTMP1:
    value->rValue = model->HSM2_bgtmp1;
    return(OK);
  case  HSM2_MOD_BGTMP2:
    value->rValue = model->HSM2_bgtmp2;
    return(OK);
  case  HSM2_MOD_EG0:
    value->rValue = model->HSM2_eg0;
    return(OK);
  case  HSM2_MOD_TOX:
    value->rValue = model->HSM2_tox;
    return(OK);
  case  HSM2_MOD_XLD:
    value->rValue = model->HSM2_xld;
    return(OK);
  case  HSM2_MOD_LOVER:
    value->rValue = model->HSM2_lover;
    return(OK);
  case  HSM2_MOD_DDLTMAX: /* Vdseff */
    value->rValue = model->HSM2_ddltmax;
    return(OK);
  case  HSM2_MOD_DDLTSLP: /* Vdseff */
    value->rValue = model->HSM2_ddltslp;
    return(OK);
  case  HSM2_MOD_DDLTICT: /* Vdseff */
    value->rValue = model->HSM2_ddltict;
    return(OK);
  case  HSM2_MOD_VFBOVER:
    value->rValue = model->HSM2_vfbover;
    return(OK);
  case  HSM2_MOD_NOVER:
    value->rValue = model->HSM2_nover;
    return(OK);
  case  HSM2_MOD_XWD:
    value->rValue = model->HSM2_xwd;
    return(OK);
  case  HSM2_MOD_XL:
    value->rValue = model->HSM2_xl;
    return(OK);
  case  HSM2_MOD_XW:
    value->rValue = model->HSM2_xw;
    return(OK);
  case  HSM2_MOD_SAREF:
    value->rValue = model->HSM2_saref;
    return(OK);
  case  HSM2_MOD_SBREF:
    value->rValue = model->HSM2_sbref;
    return(OK);
  case  HSM2_MOD_LL:
    value->rValue = model->HSM2_ll;
    return(OK);
  case  HSM2_MOD_LLD:
    value->rValue = model->HSM2_lld;
    return(OK);
  case  HSM2_MOD_LLN:
    value->rValue = model->HSM2_lln;
    return(OK);
  case  HSM2_MOD_WL:
    value->rValue = model->HSM2_wl;
    return(OK);
  case  HSM2_MOD_WL1:
    value->rValue = model->HSM2_wl1;
    return(OK);
  case  HSM2_MOD_WL1P:
    value->rValue = model->HSM2_wl1p;
    return(OK);
  case  HSM2_MOD_WL2:
    value->rValue = model->HSM2_wl2;
    return(OK);
  case  HSM2_MOD_WL2P:
    value->rValue = model->HSM2_wl2p;
    return(OK);
  case  HSM2_MOD_WLD:
    value->rValue = model->HSM2_wld;
    return(OK);
  case  HSM2_MOD_WLN:
    value->rValue = model->HSM2_wln;
    return(OK);
  case  HSM2_MOD_XQY:
    value->rValue = model->HSM2_xqy;
    return(OK);
  case  HSM2_MOD_XQY1:
    value->rValue = model->HSM2_xqy1;
    return(OK);
  case  HSM2_MOD_XQY2:
    value->rValue = model->HSM2_xqy2;
    return(OK);
  case  HSM2_MOD_QYRAT:
    value->rValue = model->HSM2_qyrat;
    return(OK);
  case  HSM2_MOD_RS:
    value->rValue = model->HSM2_rs;
    return(OK);
  case  HSM2_MOD_RD:
    value->rValue = model->HSM2_rd;
    return(OK);
  case  HSM2_MOD_RSH:
    value->rValue = model->HSM2_rsh;
    return(OK);
  case  HSM2_MOD_RSHG:
    value->rValue = model->HSM2_rshg;
    return(OK);
/*   case  HSM2_MOD_NGCON: */
/*     value->rValue = model->HSM2_ngcon; */
/*     return(OK); */
/*   case  HSM2_MOD_XGW: */
/*     value->rValue = model->HSM2_xgw; */
/*     return(OK); */
/*   case  HSM2_MOD_XGL: */
/*     value->rValue = model->HSM2_xgl; */
/*     return(OK); */
/*   case  HSM2_MOD_NF: */
/*     value->rValue = model->HSM2_nf; */
/*     return(OK); */
  case  HSM2_MOD_VFBC:
    value->rValue = model->HSM2_vfbc;
    return(OK);
  case  HSM2_MOD_VBI:
    value->rValue = model->HSM2_vbi;
    return(OK);
  case  HSM2_MOD_NSUBC:
    value->rValue = model->HSM2_nsubc;
      return(OK);
  case HSM2_MOD_VFBCL:
    value->rValue = model->HSM2_vfbcl;
    return(OK);
  case HSM2_MOD_VFBCLP:
    value->rValue = model->HSM2_vfbclp;
    return(OK);
  case  HSM2_MOD_PARL2:
    value->rValue = model->HSM2_parl2;
    return(OK);
  case  HSM2_MOD_LP:
    value->rValue = model->HSM2_lp;
    return(OK);
  case  HSM2_MOD_NSUBP:
    value->rValue = model->HSM2_nsubp;
    return(OK);
  case  HSM2_MOD_NSUBPL:
    value->rValue = model->HSM2_nsubpl;
    return(OK);
  case  HSM2_MOD_NSUBPFAC:
    value->rValue = model->HSM2_nsubpfac;
    return(OK);
  case HSM2_MOD_NSUBPDLT:
    value->rValue = model->HSM2_nsubpdlt;
    return(OK);
  case  HSM2_MOD_NSUBPW:
    value->rValue = model->HSM2_nsubpw;
    return(OK);
  case  HSM2_MOD_NSUBPWP:
    value->rValue = model->HSM2_nsubpwp;
    return(OK);
  case  HSM2_MOD_SCP1:
    value->rValue = model->HSM2_scp1;
    return(OK);
  case  HSM2_MOD_SCP2:
    value->rValue = model->HSM2_scp2;
    return(OK);
  case  HSM2_MOD_SCP3:
    value->rValue = model->HSM2_scp3;
    return(OK);
  case  HSM2_MOD_SC1:
    value->rValue = model->HSM2_sc1;
    return(OK);
  case  HSM2_MOD_SC2:
    value->rValue = model->HSM2_sc2;
    return(OK);
  case  HSM2_MOD_SC3:
    value->rValue = model->HSM2_sc3;
    return(OK);
  case  HSM2_MOD_SC4:
    value->rValue = model->HSM2_sc4;
    return(OK);
  case  HSM2_MOD_PGD1:
    value->rValue = model->HSM2_pgd1;
    return(OK);
  case  HSM2_MOD_PGD2:
    value->rValue = model->HSM2_pgd2;
    return(OK);

  case  HSM2_MOD_PGD4:
    value->rValue = model->HSM2_pgd4;
    return(OK);
  case  HSM2_MOD_NDEP:
    value->rValue = model->HSM2_ndep;
    return(OK);
  case  HSM2_MOD_NDEPL:
    value->rValue = model->HSM2_ndepl;
    return(OK);
  case  HSM2_MOD_NDEPLP:
    value->rValue = model->HSM2_ndeplp;
    return(OK);
  case  HSM2_MOD_NDEPW:
    value->rValue = model->HSM2_ndepw;
    return(OK);
  case  HSM2_MOD_NDEPWP:
    value->rValue = model->HSM2_ndepwp;
    return(OK);
  case  HSM2_MOD_NINV:
    value->rValue = model->HSM2_ninv;
    return(OK);
  case  HSM2_MOD_NINVD:
    value->rValue = model->HSM2_ninvd;
    return(OK);
  case  HSM2_MOD_NINVDL:
    value->rValue = model->HSM2_ninvdl;
    return(OK);
  case  HSM2_MOD_NINVDLP:
    value->rValue = model->HSM2_ninvdlp;
    return(OK);
  case  HSM2_MOD_MUECB0:
    value->rValue = model->HSM2_muecb0;
    return(OK);
  case  HSM2_MOD_MUECB1:
    value->rValue = model->HSM2_muecb1;
    return(OK);
  case  HSM2_MOD_MUEPH1:
    value->rValue = model->HSM2_mueph1;
    return(OK);
  case  HSM2_MOD_MUEPH0:
    value->rValue = model->HSM2_mueph0;
    return(OK);
  case  HSM2_MOD_MUEPHW:
    value->rValue = model->HSM2_muephw;
    return(OK);
  case  HSM2_MOD_MUEPWP:
    value->rValue = model->HSM2_muepwp;
    return(OK);
  case  HSM2_MOD_MUEPWD:
    value->rValue = model->HSM2_muepwd;
    return(OK);
  case  HSM2_MOD_MUEPHL:
    value->rValue = model->HSM2_muephl;
    return(OK);
  case  HSM2_MOD_MUEPLP:
    value->rValue = model->HSM2_mueplp;
    return(OK);
  case  HSM2_MOD_MUEPLD:
    value->rValue = model->HSM2_muepld;
    return(OK);
  case  HSM2_MOD_MUEPHS:
    value->rValue = model->HSM2_muephs;
    return(OK);
  case  HSM2_MOD_MUEPSP:
    value->rValue = model->HSM2_muepsp;
    return(OK);
  case  HSM2_MOD_VTMP:
    value->rValue = model->HSM2_vtmp;
    return(OK);
  case  HSM2_MOD_WVTH0:
    value->rValue = model->HSM2_wvth0;
    return(OK);
  case  HSM2_MOD_MUESR1:
    value->rValue = model->HSM2_muesr1;
    return(OK);
  case  HSM2_MOD_MUESR0:
    value->rValue = model->HSM2_muesr0;
    return(OK);
  case  HSM2_MOD_MUESRL:
    value->rValue = model->HSM2_muesrl;
    return(OK);
  case  HSM2_MOD_MUESLP:
    value->rValue = model->HSM2_mueslp;
    return(OK);
  case  HSM2_MOD_MUESRW:
    value->rValue = model->HSM2_muesrw;
    return(OK);
  case  HSM2_MOD_MUESWP:
    value->rValue = model->HSM2_mueswp;
    return(OK);
  case  HSM2_MOD_BB:
    value->rValue = model->HSM2_bb;
    return(OK);
  case  HSM2_MOD_SUB1:
    value->rValue = model->HSM2_sub1;
    return(OK);
  case  HSM2_MOD_SUB2:
    value->rValue = model->HSM2_sub2;
    return(OK);
  case  HSM2_MOD_SVGS:
    value->rValue = model->HSM2_svgs;
    return(OK);
  case  HSM2_MOD_SVGSL:
    value->rValue = model->HSM2_svgsl;
    return(OK);
  case  HSM2_MOD_SVGSLP:
    value->rValue = model->HSM2_svgslp;
    return(OK);
  case  HSM2_MOD_SVGSW:
    value->rValue = model->HSM2_svgsw;
    return(OK);
  case  HSM2_MOD_SVGSWP:
    value->rValue = model->HSM2_svgswp;
    return(OK);
  case  HSM2_MOD_SVBS:
    value->rValue = model->HSM2_svbs;
    return(OK);
  case  HSM2_MOD_SVBSL:
    value->rValue = model->HSM2_svbsl;
    return(OK);
  case  HSM2_MOD_SVBSLP:
    value->rValue = model->HSM2_svbslp;
    return(OK);
  case  HSM2_MOD_SVDS:
    value->rValue = model->HSM2_svds;
    return(OK);
  case  HSM2_MOD_SLG:
    value->rValue = model->HSM2_slg;
    return(OK);
  case  HSM2_MOD_SLGL:
    value->rValue = model->HSM2_slgl;
    return(OK);
  case  HSM2_MOD_SLGLP:
    value->rValue = model->HSM2_slglp;
    return(OK);
  case  HSM2_MOD_SUB1L:
    value->rValue = model->HSM2_sub1l;
    return(OK);
  case  HSM2_MOD_SUB1LP:
    value->rValue = model->HSM2_sub1lp;
    return(OK);
  case  HSM2_MOD_SUB2L:
    value->rValue = model->HSM2_sub2l;
    return(OK);
  case  HSM2_MOD_NSTI:
    value->rValue = model->HSM2_nsti;
    return(OK);
  case  HSM2_MOD_WSTI:
    value->rValue = model->HSM2_wsti;
    return(OK);
  case  HSM2_MOD_WSTIL:
    value->rValue = model->HSM2_wstil;
    return(OK);
  case  HSM2_MOD_WSTILP:
    value->rValue = model->HSM2_wstilp;
    return(OK);
  case  HSM2_MOD_WSTIW:
    value->rValue = model->HSM2_wstiw;
    return(OK);
  case  HSM2_MOD_WSTIWP:
    value->rValue = model->HSM2_wstiwp;
    return(OK);
  case  HSM2_MOD_SCSTI1:
    value->rValue = model->HSM2_scsti1;
    return(OK);
  case  HSM2_MOD_SCSTI2:
    value->rValue = model->HSM2_scsti2;
    return(OK);
  case  HSM2_MOD_VTHSTI:
    value->rValue = model->HSM2_vthsti;
    return(OK);
  case  HSM2_MOD_VDSTI:
    value->rValue = model->HSM2_vdsti;
    return(OK);
  case  HSM2_MOD_MUESTI1:
    value->rValue = model->HSM2_muesti1;
    return(OK);
  case  HSM2_MOD_MUESTI2:
    value->rValue = model->HSM2_muesti2;
    return(OK);
  case  HSM2_MOD_MUESTI3:
    value->rValue = model->HSM2_muesti3;
    return(OK);
  case  HSM2_MOD_NSUBPSTI1:
    value->rValue = model->HSM2_nsubpsti1;
    return(OK);
  case  HSM2_MOD_NSUBPSTI2:
    value->rValue = model->HSM2_nsubpsti2;
    return(OK);
  case  HSM2_MOD_NSUBPSTI3:
    value->rValue = model->HSM2_nsubpsti3;
    return(OK);
  case HSM2_MOD_NSUBCSTI1:
    value->rValue = model->HSM2_nsubcsti1;
    return(OK);
  case HSM2_MOD_NSUBCSTI2:
    value->rValue = model->HSM2_nsubcsti2;
    return(OK);
  case HSM2_MOD_NSUBCSTI3:
    value->rValue = model->HSM2_nsubcsti3;
    return(OK);
  case  HSM2_MOD_LPEXT:
    value->rValue = model->HSM2_lpext;
    return(OK);
  case  HSM2_MOD_NPEXT:
    value->rValue = model->HSM2_npext;
    return(OK);
  case  HSM2_MOD_NPEXTW:
    value->rValue = model->HSM2_npextw;
    return(OK);
  case  HSM2_MOD_NPEXTWP:
    value->rValue = model->HSM2_npextwp;
    return(OK);
  case  HSM2_MOD_SCP22:
    value->rValue = model->HSM2_scp22;
    return(OK);
  case  HSM2_MOD_SCP21:
    value->rValue = model->HSM2_scp21;
    return(OK);
  case  HSM2_MOD_BS1:
    value->rValue = model->HSM2_bs1;
    return(OK);
  case  HSM2_MOD_BS2:
    value->rValue = model->HSM2_bs2;
    return(OK);
  case  HSM2_MOD_CGSO:
    value->rValue = model->HSM2_cgso;
    return(OK);
  case  HSM2_MOD_CGDO:
    value->rValue = model->HSM2_cgdo;
    return(OK);
  case  HSM2_MOD_CGBO:
    value->rValue = model->HSM2_cgbo;
    return(OK);
  case  HSM2_MOD_TPOLY:
    value->rValue = model->HSM2_tpoly;
    return(OK);
  case  HSM2_MOD_JS0:
    value->rValue = model->HSM2_js0;
    return(OK);
  case  HSM2_MOD_JS0SW:
    value->rValue = model->HSM2_js0sw;
    return(OK);
  case  HSM2_MOD_NJ:
    value->rValue = model->HSM2_nj;
    return(OK);
  case  HSM2_MOD_NJSW:
    value->rValue = model->HSM2_njsw;
    return(OK);
  case  HSM2_MOD_XTI:
    value->rValue = model->HSM2_xti;
    return(OK);
  case  HSM2_MOD_CJ:
    value->rValue = model->HSM2_cj;
    return(OK);
  case  HSM2_MOD_CJSW:
    value->rValue = model->HSM2_cjsw;
    return(OK);
  case  HSM2_MOD_CJSWG:
    value->rValue = model->HSM2_cjswg;
    return(OK);
  case  HSM2_MOD_MJ:
    value->rValue = model->HSM2_mj;
    return(OK);
  case  HSM2_MOD_MJSW:
    value->rValue = model->HSM2_mjsw;
    return(OK);
  case  HSM2_MOD_MJSWG:
    value->rValue = model->HSM2_mjswg;
    return(OK);
  case  HSM2_MOD_PB:
    value->rValue = model->HSM2_pb;
    return(OK);
  case  HSM2_MOD_PBSW:
    value->rValue = model->HSM2_pbsw;
    return(OK);
  case  HSM2_MOD_PBSWG:
    value->rValue = model->HSM2_pbswg;
    return(OK);

  case  HSM2_MOD_TCJBD:
    value->rValue = model->HSM2_tcjbd;
    return(OK);
  case  HSM2_MOD_TCJBS:
    value->rValue = model->HSM2_tcjbs;
    return(OK);
  case  HSM2_MOD_TCJBDSW:
    value->rValue = model->HSM2_tcjbdsw;
    return(OK);
  case  HSM2_MOD_TCJBSSW:
    value->rValue = model->HSM2_tcjbssw;
    return(OK);
  case  HSM2_MOD_TCJBDSWG:
    value->rValue = model->HSM2_tcjbdswg;
    return(OK);
  case  HSM2_MOD_TCJBSSWG:
    value->rValue = model->HSM2_tcjbsswg;
    return(OK);
  case  HSM2_MOD_XTI2:
    value->rValue = model->HSM2_xti2;
    return(OK);
  case  HSM2_MOD_CISB:
    value->rValue = model->HSM2_cisb;
    return(OK);
  case  HSM2_MOD_CVB:
    value->rValue = model->HSM2_cvb;
    return(OK);
  case  HSM2_MOD_CTEMP:
    value->rValue = model->HSM2_ctemp;
    return(OK);
  case  HSM2_MOD_CISBK:
    value->rValue = model->HSM2_cisbk;
    return(OK);
  case  HSM2_MOD_CVBK:
    value->rValue = model->HSM2_cvbk;
    return(OK);
  case  HSM2_MOD_DIVX:
    value->rValue = model->HSM2_divx;
    return(OK);
  case  HSM2_MOD_CLM1:
    value->rValue = model->HSM2_clm1;
    return(OK);
  case  HSM2_MOD_CLM2:
    value->rValue = model->HSM2_clm2;
    return(OK);
  case  HSM2_MOD_CLM3:
    value->rValue = model->HSM2_clm3;
    return(OK);
  case  HSM2_MOD_CLM5:
    value->rValue = model->HSM2_clm5;
    return(OK);
  case  HSM2_MOD_CLM6:
    value->rValue = model->HSM2_clm6;
    return(OK);
  case  HSM2_MOD_MUETMP:
    value->rValue = model->HSM2_muetmp;
    return(OK);
  case  HSM2_MOD_VOVER:
    value->rValue = model->HSM2_vover;
    return(OK);
  case  HSM2_MOD_VOVERP:
    value->rValue = model->HSM2_voverp;
    return(OK);
  case  HSM2_MOD_VOVERS:
    value->rValue = model->HSM2_vovers;
    return(OK);
  case  HSM2_MOD_VOVERSP:
    value->rValue = model->HSM2_voversp;
    return(OK);
  case  HSM2_MOD_WFC:
    value->rValue = model->HSM2_wfc;
    return(OK);
  case  HSM2_MOD_NSUBCW:
    value->rValue = model->HSM2_nsubcw;
    return(OK);
  case  HSM2_MOD_NSUBCWP:
    value->rValue = model->HSM2_nsubcwp;
    return(OK);
  case  HSM2_MOD_NSUBCMAX:
    value->rValue = model->HSM2_nsubcmax;
    return(OK);
  case  HSM2_MOD_QME1:
    value->rValue = model->HSM2_qme1;
    return(OK);
  case  HSM2_MOD_QME2:
    value->rValue = model->HSM2_qme2;
    return(OK);
  case  HSM2_MOD_QME3:
    value->rValue = model->HSM2_qme3;
    return(OK);
  case  HSM2_MOD_GIDL1:
    value->rValue = model->HSM2_gidl1;
    return(OK);
  case  HSM2_MOD_GIDL2:
    value->rValue = model->HSM2_gidl2;
    return(OK);
  case  HSM2_MOD_GIDL3:
    value->rValue = model->HSM2_gidl3;
    return(OK);
  case  HSM2_MOD_GIDL4:
    value->rValue = model->HSM2_gidl4;
    return(OK);
  case  HSM2_MOD_GIDL5:
    value->rValue = model->HSM2_gidl5;
    return(OK);
  case HSM2_MOD_GIDL6:
    value->rValue = model->HSM2_gidl6;
    return(OK);
  case HSM2_MOD_GIDL7:
    value->rValue = model->HSM2_gidl7;
    return(OK);
  case  HSM2_MOD_GLEAK1:
    value->rValue = model->HSM2_gleak1;
    return(OK);
  case  HSM2_MOD_GLEAK2:
    value->rValue = model->HSM2_gleak2;
    return(OK);
  case  HSM2_MOD_GLEAK3:
    value->rValue = model->HSM2_gleak3;
    return(OK);
  case  HSM2_MOD_GLEAK4:
    value->rValue = model->HSM2_gleak4;
    return(OK);
  case  HSM2_MOD_GLEAK5:
    value->rValue = model->HSM2_gleak5;
    return(OK);
  case  HSM2_MOD_GLEAK6:
    value->rValue = model->HSM2_gleak6;
    return(OK);
  case  HSM2_MOD_GLEAK7:
    value->rValue = model->HSM2_gleak7;
    return(OK);
  case  HSM2_MOD_GLKSD1:
    value->rValue = model->HSM2_glksd1;
    return(OK);
  case  HSM2_MOD_GLKSD2:
    value->rValue = model->HSM2_glksd2;
    return(OK);
  case  HSM2_MOD_GLKSD3:
    value->rValue = model->HSM2_glksd3;
    return(OK);
  case  HSM2_MOD_GLKB1:
    value->rValue = model->HSM2_glkb1;
    return(OK);
  case  HSM2_MOD_GLKB2:
    value->rValue = model->HSM2_glkb2;
    return(OK);
  case  HSM2_MOD_GLKB3:
    value->rValue = model->HSM2_glkb3;
    return(OK);
  case  HSM2_MOD_EGIG:
    value->rValue = model->HSM2_egig;
    return(OK);
  case  HSM2_MOD_IGTEMP2:
    value->rValue = model->HSM2_igtemp2;
    return(OK);
  case  HSM2_MOD_IGTEMP3:
    value->rValue = model->HSM2_igtemp3;
    return(OK);
  case  HSM2_MOD_VZADD0:
    value->rValue = model->HSM2_vzadd0;
    return(OK);
  case  HSM2_MOD_PZADD0:
    value->rValue = model->HSM2_pzadd0;
    return(OK);
  case  HSM2_MOD_NFTRP:
    value->rValue = model->HSM2_nftrp;
    return(OK);
  case  HSM2_MOD_NFALP:
    value->rValue = model->HSM2_nfalp;
    return(OK);
  case  HSM2_MOD_CIT:
    value->rValue = model->HSM2_cit;
    return(OK);
  case  HSM2_MOD_FALPH:
    value->rValue = model->HSM2_falph;
    return(OK);
  case  HSM2_MOD_KAPPA:
    value->rValue = model->HSM2_kappa;
    return(OK);
  case  HSM2_MOD_VDIFFJ:
    value->rValue = model->HSM2_vdiffj;
    return(OK);
  case  HSM2_MOD_DLY1:
    value->rValue = model->HSM2_dly1;
    return(OK);
  case  HSM2_MOD_DLY2:
    value->rValue = model->HSM2_dly2;
    return(OK);
  case  HSM2_MOD_DLY3:
    value->rValue = model->HSM2_dly3;
    return(OK);
  case  HSM2_MOD_TNOM:
    value->rValue = model->HSM2_tnom;
    return(OK);
  case  HSM2_MOD_OVSLP:
    value->rValue = model->HSM2_ovslp;
    return(OK);
  case  HSM2_MOD_OVMAG:
    value->rValue = model->HSM2_ovmag;
    return(OK);
  case  HSM2_MOD_GBMIN:
    value->rValue = model->HSM2_gbmin;
    return(OK);
  case  HSM2_MOD_RBPB:
    value->rValue = model->HSM2_rbpb;
    return(OK);
  case  HSM2_MOD_RBPD:
    value->rValue = model->HSM2_rbpd;
    return(OK);
  case  HSM2_MOD_RBPS:
    value->rValue = model->HSM2_rbps;
    return(OK);
  case  HSM2_MOD_RBDB:
    value->rValue = model->HSM2_rbdb;
    return(OK);
  case  HSM2_MOD_RBSB:
    value->rValue = model->HSM2_rbsb;
    return(OK);
  case  HSM2_MOD_IBPC1:
    value->rValue = model->HSM2_ibpc1;
    return(OK);
  case  HSM2_MOD_IBPC2:
    value->rValue = model->HSM2_ibpc2;
    return(OK);
  case  HSM2_MOD_MPHDFM:
    value->rValue = model->HSM2_mphdfm;
    return(OK);


  case  HSM2_MOD_PTL:
    value->rValue = model->HSM2_ptl;
    return(OK);
  case  HSM2_MOD_PTP:
    value->rValue = model->HSM2_ptp;
    return(OK);
  case  HSM2_MOD_PT2:
    value->rValue = model->HSM2_pt2;
    return(OK);
  case  HSM2_MOD_PTLP:
    value->rValue = model->HSM2_ptlp;
    return(OK);
  case  HSM2_MOD_GDL:
    value->rValue = model->HSM2_gdl;
    return(OK);
  case  HSM2_MOD_GDLP:
    value->rValue = model->HSM2_gdlp;
    return(OK);

  case  HSM2_MOD_GDLD:
    value->rValue = model->HSM2_gdld;
    return(OK);
  case  HSM2_MOD_PT4:
    value->rValue = model->HSM2_pt4;
    return(OK);
  case  HSM2_MOD_PT4P:
    value->rValue = model->HSM2_pt4p;
    return(OK);
  case  HSM2_MOD_MUEPHL2:
    value->rValue = model->HSM2_muephl2;
    return(OK);
  case  HSM2_MOD_MUEPLP2:
    value->rValue = model->HSM2_mueplp2;
    return(OK);
  case  HSM2_MOD_NSUBCW2:
    value->rValue = model->HSM2_nsubcw2;
    return(OK);
  case  HSM2_MOD_NSUBCWP2:
    value->rValue = model->HSM2_nsubcwp2;
    return(OK);
  case  HSM2_MOD_MUEPHW2:
    value->rValue = model->HSM2_muephw2;
    return(OK);
  case  HSM2_MOD_MUEPWP2:
    value->rValue = model->HSM2_muepwp2;
    return(OK);
/* WPE */
  case HSM2_MOD_WEB:
    value->rValue = model->HSM2_web;
	return(OK);
  case HSM2_MOD_WEC:
    value->rValue = model->HSM2_wec;
    return(OK);
  case HSM2_MOD_NSUBCWPE:
    value->rValue = model->HSM2_nsubcwpe;
    return(OK);
  case HSM2_MOD_NPEXTWPE:
    value->rValue = model->HSM2_npextwpe;
    return(OK);
  case HSM2_MOD_NSUBPWPE:
    value->rValue = model->HSM2_nsubpwpe;
    return(OK);
  case  HSM2_MOD_VGSMIN:
    value->rValue = model->HSM2_Vgsmin;
    return(OK);
  case  HSM2_MOD_SC3VBS:
    value->rValue = model->HSM2_sc3Vbs;
    return(OK);
  case  HSM2_MOD_BYPTOL:
    value->rValue = model->HSM2_byptol;
    return(OK);
  case  HSM2_MOD_MUECB0LP:
    value->rValue = model->HSM2_muecb0lp;
    return(OK);
  case  HSM2_MOD_MUECB1LP:
    value->rValue = model->HSM2_muecb1lp;
    return(OK);

  /* Depletion Mode MOSFET */
  case  HSM2_MOD_NDEPM:
    value->rValue = model->HSM2_ndepm;
    return(OK);
  case  HSM2_MOD_NDEPML:
    value->rValue = model->HSM2_ndepml;
    return(OK);
  case  HSM2_MOD_NDEPMLP:
    value->rValue = model->HSM2_ndepmlp;
    return(OK);
  case  HSM2_MOD_TNDEP:
    value->rValue = model->HSM2_tndep;
    return(OK);
  case  HSM2_MOD_DEPLEAK:
    value->rValue = model->HSM2_depleak;
    return(OK);
  case  HSM2_MOD_DEPLEAKL:
    value->rValue = model->HSM2_depleakl;
    return(OK);
  case  HSM2_MOD_DEPLEAKLP:
    value->rValue = model->HSM2_depleaklp;
    return(OK);
  case  HSM2_MOD_DEPETA:
    value->rValue = model->HSM2_depeta;
    return(OK);
  case HSM2_MOD_DEPMUE0:
    value->rValue = model->HSM2_depmue0;
    return(OK);
  case HSM2_MOD_DEPMUE0L:
    value->rValue = model->HSM2_depmue0l;
    return(OK);
  case HSM2_MOD_DEPMUE0LP:
    value->rValue = model->HSM2_depmue0lp;
    return(OK);
  case HSM2_MOD_DEPMUE1:
    value->rValue = model->HSM2_depmue1;
    return(OK);
  case HSM2_MOD_DEPMUE1L:
    value->rValue = model->HSM2_depmue1l;
    return(OK);
  case HSM2_MOD_DEPMUE1LP:
    value->rValue = model->HSM2_depmue1lp;
    return(OK);
  case HSM2_MOD_DEPMUEBACK0:
    value->rValue = model->HSM2_depmueback0;
    return(OK);
  case HSM2_MOD_DEPMUEBACK0L:
    value->rValue = model->HSM2_depmueback0l;
    return(OK);
  case HSM2_MOD_DEPMUEBACK0LP:
    value->rValue = model->HSM2_depmueback0lp;
    return(OK);
  case HSM2_MOD_DEPMUEBACK1:
    value->rValue = model->HSM2_depmueback1;
    return(OK);
  case HSM2_MOD_DEPMUEBACK1L:
    value->rValue = model->HSM2_depmueback1l;
    return(OK);
  case HSM2_MOD_DEPMUEBACK1LP:
    value->rValue = model->HSM2_depmueback1lp;
    return(OK);
  case HSM2_MOD_DEPMUEPH0:
    value->rValue = model->HSM2_depmueph0;
    return(OK);
  case HSM2_MOD_DEPMUEPH1:
    value->rValue = model->HSM2_depmueph1;
    return(OK);
  case HSM2_MOD_DEPVMAX:
    value->rValue = model->HSM2_depvmax;
    return(OK);
  case HSM2_MOD_DEPVMAXL:
    value->rValue = model->HSM2_depvmaxl;
    return(OK);
  case HSM2_MOD_DEPVMAXLP:
    value->rValue = model->HSM2_depvmaxlp;
    return(OK);
  case HSM2_MOD_DEPVDSEF1:
    value->rValue = model->HSM2_depvdsef1;
    return(OK);
  case HSM2_MOD_DEPVDSEF1L:
    value->rValue = model->HSM2_depvdsef1l;
    return(OK);
  case HSM2_MOD_DEPVDSEF1LP:
    value->rValue = model->HSM2_depvdsef1lp;
    return(OK);
  case HSM2_MOD_DEPVDSEF2:
    value->rValue = model->HSM2_depvdsef2;
    return(OK);
  case HSM2_MOD_DEPVDSEF2L:
    value->rValue = model->HSM2_depvdsef2l;
    return(OK);
  case HSM2_MOD_DEPVDSEF2LP:
    value->rValue = model->HSM2_depvdsef2lp;
    return(OK);
  case HSM2_MOD_DEPBB:
    value->rValue = model->HSM2_depbb;
    return(OK);
  case HSM2_MOD_DEPMUETMP:
    value->rValue = model->HSM2_depmuetmp;
    return(OK);


  /* binning parameters */
  case  HSM2_MOD_LMIN:
    value->rValue = model->HSM2_lmin;
    return(OK);
  case  HSM2_MOD_LMAX:
    value->rValue = model->HSM2_lmax;
    return(OK);
  case  HSM2_MOD_WMIN:
    value->rValue = model->HSM2_wmin;
    return(OK);
  case  HSM2_MOD_WMAX:
    value->rValue = model->HSM2_wmax;
    return(OK);
  case  HSM2_MOD_LBINN:
    value->rValue = model->HSM2_lbinn;
    return(OK);
  case  HSM2_MOD_WBINN:
    value->rValue = model->HSM2_wbinn;
    return(OK);

  /* Length dependence */
  case  HSM2_MOD_LVMAX:
    value->rValue = model->HSM2_lvmax;
    return(OK);
  case  HSM2_MOD_LBGTMP1:
    value->rValue = model->HSM2_lbgtmp1;
    return(OK);
  case  HSM2_MOD_LBGTMP2:
    value->rValue = model->HSM2_lbgtmp2;
    return(OK);
  case  HSM2_MOD_LEG0:
    value->rValue = model->HSM2_leg0;
    return(OK);
  case  HSM2_MOD_LLOVER:
    value->rValue = model->HSM2_llover;
    return(OK);
  case  HSM2_MOD_LVFBOVER:
    value->rValue = model->HSM2_lvfbover;
    return(OK);
  case  HSM2_MOD_LNOVER:
    value->rValue = model->HSM2_lnover;
    return(OK);
  case  HSM2_MOD_LWL2:
    value->rValue = model->HSM2_lwl2;
    return(OK);
  case  HSM2_MOD_LVFBC:
    value->rValue = model->HSM2_lvfbc;
    return(OK);
  case  HSM2_MOD_LNSUBC:
    value->rValue = model->HSM2_lnsubc;
    return(OK);
  case  HSM2_MOD_LNSUBP:
    value->rValue = model->HSM2_lnsubp;
    return(OK);
  case  HSM2_MOD_LSCP1:
    value->rValue = model->HSM2_lscp1;
    return(OK);
  case  HSM2_MOD_LSCP2:
    value->rValue = model->HSM2_lscp2;
    return(OK);
  case  HSM2_MOD_LSCP3:
    value->rValue = model->HSM2_lscp3;
    return(OK);
  case  HSM2_MOD_LSC1:
    value->rValue = model->HSM2_lsc1;
    return(OK);
  case  HSM2_MOD_LSC2:
    value->rValue = model->HSM2_lsc2;
    return(OK);
  case  HSM2_MOD_LSC3:
    value->rValue = model->HSM2_lsc3;
    return(OK);
  case  HSM2_MOD_LSC4:
    value->rValue = model->HSM2_lsc4;
    return(OK);
  case  HSM2_MOD_LPGD1:
    value->rValue = model->HSM2_lpgd1;
    return(OK);
//case  HSM2_MOD_LPGD3:
//  value->rValue = model->HSM2_lpgd3;
//  return(OK);
  case  HSM2_MOD_LNDEP:
    value->rValue = model->HSM2_lndep;
    return(OK);
  case  HSM2_MOD_LNINV:
    value->rValue = model->HSM2_lninv;
    return(OK);
  case  HSM2_MOD_LMUECB0:
    value->rValue = model->HSM2_lmuecb0;
    return(OK);
  case  HSM2_MOD_LMUECB1:
    value->rValue = model->HSM2_lmuecb1;
    return(OK);
  case  HSM2_MOD_LMUEPH1:
    value->rValue = model->HSM2_lmueph1;
    return(OK);
  case  HSM2_MOD_LVTMP:
    value->rValue = model->HSM2_lvtmp;
    return(OK);
  case  HSM2_MOD_LWVTH0:
    value->rValue = model->HSM2_lwvth0;
    return(OK);
  case  HSM2_MOD_LMUESR1:
    value->rValue = model->HSM2_lmuesr1;
    return(OK);
  case  HSM2_MOD_LMUETMP:
    value->rValue = model->HSM2_lmuetmp;
    return(OK);
  case  HSM2_MOD_LSUB1:
    value->rValue = model->HSM2_lsub1;
    return(OK);
  case  HSM2_MOD_LSUB2:
    value->rValue = model->HSM2_lsub2;
    return(OK);
  case  HSM2_MOD_LSVDS:
    value->rValue = model->HSM2_lsvds;
    return(OK);
  case  HSM2_MOD_LSVBS:
    value->rValue = model->HSM2_lsvbs;
    return(OK);
  case  HSM2_MOD_LSVGS:
    value->rValue = model->HSM2_lsvgs;
    return(OK);
  case  HSM2_MOD_LNSTI:
    value->rValue = model->HSM2_lnsti;
    return(OK);
  case  HSM2_MOD_LWSTI:
    value->rValue = model->HSM2_lwsti;
    return(OK);
  case  HSM2_MOD_LSCSTI1:
    value->rValue = model->HSM2_lscsti1;
    return(OK);
  case  HSM2_MOD_LSCSTI2:
    value->rValue = model->HSM2_lscsti2;
    return(OK);
  case  HSM2_MOD_LVTHSTI:
    value->rValue = model->HSM2_lvthsti;
    return(OK);
  case  HSM2_MOD_LMUESTI1:
    value->rValue = model->HSM2_lmuesti1;
    return(OK);
  case  HSM2_MOD_LMUESTI2:
    value->rValue = model->HSM2_lmuesti2;
    return(OK);
  case  HSM2_MOD_LMUESTI3:
    value->rValue = model->HSM2_lmuesti3;
    return(OK);
  case  HSM2_MOD_LNSUBPSTI1:
    value->rValue = model->HSM2_lnsubpsti1;
    return(OK);
  case  HSM2_MOD_LNSUBPSTI2:
    value->rValue = model->HSM2_lnsubpsti2;
    return(OK);
  case  HSM2_MOD_LNSUBPSTI3:
    value->rValue = model->HSM2_lnsubpsti3;
    return(OK);
  case HSM2_MOD_LNSUBCSTI1:
    value->rValue = model->HSM2_lnsubcsti1;
    return(OK);
  case HSM2_MOD_LNSUBCSTI2:
    value->rValue = model->HSM2_lnsubcsti2;
    return(OK);
  case HSM2_MOD_LNSUBCSTI3:
    value->rValue = model->HSM2_lnsubcsti3;
    return(OK);
  case  HSM2_MOD_LCGSO:
    value->rValue = model->HSM2_lcgso;
    return(OK);
  case  HSM2_MOD_LCGDO:
    value->rValue = model->HSM2_lcgdo;
    return(OK);
  case  HSM2_MOD_LJS0:
    value->rValue = model->HSM2_ljs0;
    return(OK);
  case  HSM2_MOD_LJS0SW:
    value->rValue = model->HSM2_ljs0sw;
    return(OK);
  case  HSM2_MOD_LNJ:
    value->rValue = model->HSM2_lnj;
    return(OK);
  case  HSM2_MOD_LCISBK:
    value->rValue = model->HSM2_lcisbk;
    return(OK);
  case  HSM2_MOD_LCLM1:
    value->rValue = model->HSM2_lclm1;
    return(OK);
  case  HSM2_MOD_LCLM2:
    value->rValue = model->HSM2_lclm2;
    return(OK);
  case  HSM2_MOD_LCLM3:
    value->rValue = model->HSM2_lclm3;
    return(OK);
  case  HSM2_MOD_LWFC:
    value->rValue = model->HSM2_lwfc;
    return(OK);
  case  HSM2_MOD_LGIDL1:
    value->rValue = model->HSM2_lgidl1;
    return(OK);
  case  HSM2_MOD_LGIDL2:
    value->rValue = model->HSM2_lgidl2;
    return(OK);
  case  HSM2_MOD_LGLEAK1:
    value->rValue = model->HSM2_lgleak1;
    return(OK);
  case  HSM2_MOD_LGLEAK2:
    value->rValue = model->HSM2_lgleak2;
    return(OK);
  case  HSM2_MOD_LGLEAK3:
    value->rValue = model->HSM2_lgleak3;
    return(OK);
  case  HSM2_MOD_LGLEAK6:
    value->rValue = model->HSM2_lgleak6;
    return(OK);
  case  HSM2_MOD_LGLKSD1:
    value->rValue = model->HSM2_lglksd1;
    return(OK);
  case  HSM2_MOD_LGLKSD2:
    value->rValue = model->HSM2_lglksd2;
    return(OK);
  case  HSM2_MOD_LGLKB1:
    value->rValue = model->HSM2_lglkb1;
    return(OK);
  case  HSM2_MOD_LGLKB2:
    value->rValue = model->HSM2_lglkb2;
    return(OK);
  case  HSM2_MOD_LNFTRP:
    value->rValue = model->HSM2_lnftrp;
    return(OK);
  case  HSM2_MOD_LNFALP:
    value->rValue = model->HSM2_lnfalp;
    return(OK);
  case  HSM2_MOD_LVDIFFJ:
    value->rValue = model->HSM2_lvdiffj;
    return(OK);
  case  HSM2_MOD_LIBPC1:
    value->rValue = model->HSM2_libpc1;
    return(OK);
  case  HSM2_MOD_LIBPC2:
    value->rValue = model->HSM2_libpc2;
    return(OK);

  /* Width dependence */
  case  HSM2_MOD_WVMAX:
    value->rValue = model->HSM2_wvmax;
    return(OK);
  case  HSM2_MOD_WBGTMP1:
    value->rValue = model->HSM2_wbgtmp1;
    return(OK);
  case  HSM2_MOD_WBGTMP2:
    value->rValue = model->HSM2_wbgtmp2;
    return(OK);
  case  HSM2_MOD_WEG0:
    value->rValue = model->HSM2_weg0;
    return(OK);
  case  HSM2_MOD_WLOVER:
    value->rValue = model->HSM2_wlover;
    return(OK);
  case  HSM2_MOD_WVFBOVER:
    value->rValue = model->HSM2_wvfbover;
    return(OK);
  case  HSM2_MOD_WNOVER:
    value->rValue = model->HSM2_wnover;
    return(OK);
  case  HSM2_MOD_WWL2:
    value->rValue = model->HSM2_wwl2;
    return(OK);
  case  HSM2_MOD_WVFBC:
    value->rValue = model->HSM2_wvfbc;
    return(OK);
  case  HSM2_MOD_WNSUBC:
    value->rValue = model->HSM2_wnsubc;
    return(OK);
  case  HSM2_MOD_WNSUBP:
    value->rValue = model->HSM2_wnsubp;
    return(OK);
  case  HSM2_MOD_WSCP1:
    value->rValue = model->HSM2_wscp1;
    return(OK);
  case  HSM2_MOD_WSCP2:
    value->rValue = model->HSM2_wscp2;
    return(OK);
  case  HSM2_MOD_WSCP3:
    value->rValue = model->HSM2_wscp3;
    return(OK);
  case  HSM2_MOD_WSC1:
    value->rValue = model->HSM2_wsc1;
    return(OK);
  case  HSM2_MOD_WSC2:
    value->rValue = model->HSM2_wsc2;
    return(OK);
  case  HSM2_MOD_WSC3:
    value->rValue = model->HSM2_wsc3;
    return(OK);
  case  HSM2_MOD_WSC4:
    value->rValue = model->HSM2_wsc4;
    return(OK);
  case  HSM2_MOD_WPGD1:
    value->rValue = model->HSM2_wpgd1;
    return(OK);
//case  HSM2_MOD_WPGD3:
//  value->rValue = model->HSM2_wpgd3;
//  return(OK);
  case  HSM2_MOD_WNDEP:
    value->rValue = model->HSM2_wndep;
    return(OK);
  case  HSM2_MOD_WNINV:
    value->rValue = model->HSM2_wninv;
    return(OK);
  case  HSM2_MOD_WMUECB0:
    value->rValue = model->HSM2_wmuecb0;
    return(OK);
  case  HSM2_MOD_WMUECB1:
    value->rValue = model->HSM2_wmuecb1;
    return(OK);
  case  HSM2_MOD_WMUEPH1:
    value->rValue = model->HSM2_wmueph1;
    return(OK);
  case  HSM2_MOD_WVTMP:
    value->rValue = model->HSM2_wvtmp;
    return(OK);
  case  HSM2_MOD_WWVTH0:
    value->rValue = model->HSM2_wwvth0;
    return(OK);
  case  HSM2_MOD_WMUESR1:
    value->rValue = model->HSM2_wmuesr1;
    return(OK);
  case  HSM2_MOD_WMUETMP:
    value->rValue = model->HSM2_wmuetmp;
    return(OK);
  case  HSM2_MOD_WSUB1:
    value->rValue = model->HSM2_wsub1;
    return(OK);
  case  HSM2_MOD_WSUB2:
    value->rValue = model->HSM2_wsub2;
    return(OK);
  case  HSM2_MOD_WSVDS:
    value->rValue = model->HSM2_wsvds;
    return(OK);
  case  HSM2_MOD_WSVBS:
    value->rValue = model->HSM2_wsvbs;
    return(OK);
  case  HSM2_MOD_WSVGS:
    value->rValue = model->HSM2_wsvgs;
    return(OK);
  case  HSM2_MOD_WNSTI:
    value->rValue = model->HSM2_wnsti;
    return(OK);
  case  HSM2_MOD_WWSTI:
    value->rValue = model->HSM2_wwsti;
    return(OK);
  case  HSM2_MOD_WSCSTI1:
    value->rValue = model->HSM2_wscsti1;
    return(OK);
  case  HSM2_MOD_WSCSTI2:
    value->rValue = model->HSM2_wscsti2;
    return(OK);
  case  HSM2_MOD_WVTHSTI:
    value->rValue = model->HSM2_wvthsti;
    return(OK);
  case  HSM2_MOD_WMUESTI1:
    value->rValue = model->HSM2_wmuesti1;
    return(OK);
  case  HSM2_MOD_WMUESTI2:
    value->rValue = model->HSM2_wmuesti2;
    return(OK);
  case  HSM2_MOD_WMUESTI3:
    value->rValue = model->HSM2_wmuesti3;
    return(OK);
  case  HSM2_MOD_WNSUBPSTI1:
    value->rValue = model->HSM2_wnsubpsti1;
    return(OK);
  case  HSM2_MOD_WNSUBPSTI2:
    value->rValue = model->HSM2_wnsubpsti2;
    return(OK);
  case  HSM2_MOD_WNSUBPSTI3:
    value->rValue = model->HSM2_wnsubpsti3;
    return(OK);
  case HSM2_MOD_WNSUBCSTI1:
    value->rValue = model->HSM2_wnsubcsti1;
    return(OK);
  case HSM2_MOD_WNSUBCSTI2:
    value->rValue = model->HSM2_wnsubcsti2;
    return(OK);
  case HSM2_MOD_WNSUBCSTI3:
    value->rValue = model->HSM2_wnsubcsti3;
    return(OK);
  case  HSM2_MOD_WCGSO:
    value->rValue = model->HSM2_wcgso;
    return(OK);
  case  HSM2_MOD_WCGDO:
    value->rValue = model->HSM2_wcgdo;
    return(OK);
  case  HSM2_MOD_WJS0:
    value->rValue = model->HSM2_wjs0;
    return(OK);
  case  HSM2_MOD_WJS0SW:
    value->rValue = model->HSM2_wjs0sw;
    return(OK);
  case  HSM2_MOD_WNJ:
    value->rValue = model->HSM2_wnj;
    return(OK);
  case  HSM2_MOD_WCISBK:
    value->rValue = model->HSM2_wcisbk;
    return(OK);
  case  HSM2_MOD_WCLM1:
    value->rValue = model->HSM2_wclm1;
    return(OK);
  case  HSM2_MOD_WCLM2:
    value->rValue = model->HSM2_wclm2;
    return(OK);
  case  HSM2_MOD_WCLM3:
    value->rValue = model->HSM2_wclm3;
    return(OK);
  case  HSM2_MOD_WWFC:
    value->rValue = model->HSM2_wwfc;
    return(OK);
  case  HSM2_MOD_WGIDL1:
    value->rValue = model->HSM2_wgidl1;
    return(OK);
  case  HSM2_MOD_WGIDL2:
    value->rValue = model->HSM2_wgidl2;
    return(OK);
  case  HSM2_MOD_WGLEAK1:
    value->rValue = model->HSM2_wgleak1;
    return(OK);
  case  HSM2_MOD_WGLEAK2:
    value->rValue = model->HSM2_wgleak2;
    return(OK);
  case  HSM2_MOD_WGLEAK3:
    value->rValue = model->HSM2_wgleak3;
    return(OK);
  case  HSM2_MOD_WGLEAK6:
    value->rValue = model->HSM2_wgleak6;
    return(OK);
  case  HSM2_MOD_WGLKSD1:
    value->rValue = model->HSM2_wglksd1;
    return(OK);
  case  HSM2_MOD_WGLKSD2:
    value->rValue = model->HSM2_wglksd2;
    return(OK);
  case  HSM2_MOD_WGLKB1:
    value->rValue = model->HSM2_wglkb1;
    return(OK);
  case  HSM2_MOD_WGLKB2:
    value->rValue = model->HSM2_wglkb2;
    return(OK);
  case  HSM2_MOD_WNFTRP:
    value->rValue = model->HSM2_wnftrp;
    return(OK);
  case  HSM2_MOD_WNFALP:
    value->rValue = model->HSM2_wnfalp;
    return(OK);
  case  HSM2_MOD_WVDIFFJ:
    value->rValue = model->HSM2_wvdiffj;
    return(OK);
  case  HSM2_MOD_WIBPC1:
    value->rValue = model->HSM2_wibpc1;
    return(OK);
  case  HSM2_MOD_WIBPC2:
    value->rValue = model->HSM2_wibpc2;
    return(OK);

  /* Cross-term dependence */
  case  HSM2_MOD_PVMAX:
    value->rValue = model->HSM2_pvmax;
    return(OK);
  case  HSM2_MOD_PBGTMP1:
    value->rValue = model->HSM2_pbgtmp1;
    return(OK);
  case  HSM2_MOD_PBGTMP2:
    value->rValue = model->HSM2_pbgtmp2;
    return(OK);
  case  HSM2_MOD_PEG0:
    value->rValue = model->HSM2_peg0;
    return(OK);
  case  HSM2_MOD_PLOVER:
    value->rValue = model->HSM2_plover;
    return(OK);
  case  HSM2_MOD_PVFBOVER:
    value->rValue = model->HSM2_pvfbover;
    return(OK);
  case  HSM2_MOD_PNOVER:
    value->rValue = model->HSM2_pnover;
    return(OK);
  case  HSM2_MOD_PWL2:
    value->rValue = model->HSM2_pwl2;
    return(OK);
  case  HSM2_MOD_PVFBC:
    value->rValue = model->HSM2_pvfbc;
    return(OK);
  case  HSM2_MOD_PNSUBC:
    value->rValue = model->HSM2_pnsubc;
    return(OK);
  case  HSM2_MOD_PNSUBP:
    value->rValue = model->HSM2_pnsubp;
    return(OK);
  case  HSM2_MOD_PSCP1:
    value->rValue = model->HSM2_pscp1;
    return(OK);
  case  HSM2_MOD_PSCP2:
    value->rValue = model->HSM2_pscp2;
    return(OK);
  case  HSM2_MOD_PSCP3:
    value->rValue = model->HSM2_pscp3;
    return(OK);
  case  HSM2_MOD_PSC1:
    value->rValue = model->HSM2_psc1;
    return(OK);
  case  HSM2_MOD_PSC2:
    value->rValue = model->HSM2_psc2;
    return(OK);
  case  HSM2_MOD_PSC3:
    value->rValue = model->HSM2_psc3;
    return(OK);
  case  HSM2_MOD_PSC4:
    value->rValue = model->HSM2_psc4;
    return(OK);
  case  HSM2_MOD_PPGD1:
    value->rValue = model->HSM2_ppgd1;
    return(OK);
//case  HSM2_MOD_PPGD3:
//  value->rValue = model->HSM2_ppgd3;
//  return(OK);
  case  HSM2_MOD_PNDEP:
    value->rValue = model->HSM2_pndep;
    return(OK);
  case  HSM2_MOD_PNINV:
    value->rValue = model->HSM2_pninv;
    return(OK);
  case  HSM2_MOD_PMUECB0:
    value->rValue = model->HSM2_pmuecb0;
    return(OK);
  case  HSM2_MOD_PMUECB1:
    value->rValue = model->HSM2_pmuecb1;
    return(OK);
  case  HSM2_MOD_PMUEPH1:
    value->rValue = model->HSM2_pmueph1;
    return(OK);
  case  HSM2_MOD_PVTMP:
    value->rValue = model->HSM2_pvtmp;
    return(OK);
  case  HSM2_MOD_PWVTH0:
    value->rValue = model->HSM2_pwvth0;
    return(OK);
  case  HSM2_MOD_PMUESR1:
    value->rValue = model->HSM2_pmuesr1;
    return(OK);
  case  HSM2_MOD_PMUETMP:
    value->rValue = model->HSM2_pmuetmp;
    return(OK);
  case  HSM2_MOD_PSUB1:
    value->rValue = model->HSM2_psub1;
    return(OK);
  case  HSM2_MOD_PSUB2:
    value->rValue = model->HSM2_psub2;
    return(OK);
  case  HSM2_MOD_PSVDS:
    value->rValue = model->HSM2_psvds;
    return(OK);
  case  HSM2_MOD_PSVBS:
    value->rValue = model->HSM2_psvbs;
    return(OK);
  case  HSM2_MOD_PSVGS:
    value->rValue = model->HSM2_psvgs;
    return(OK);
  case  HSM2_MOD_PNSTI:
    value->rValue = model->HSM2_pnsti;
    return(OK);
  case  HSM2_MOD_PWSTI:
    value->rValue = model->HSM2_pwsti;
    return(OK);
  case  HSM2_MOD_PSCSTI1:
    value->rValue = model->HSM2_pscsti1;
    return(OK);
  case  HSM2_MOD_PSCSTI2:
    value->rValue = model->HSM2_pscsti2;
    return(OK);
  case  HSM2_MOD_PVTHSTI:
    value->rValue = model->HSM2_pvthsti;
    return(OK);
  case  HSM2_MOD_PMUESTI1:
    value->rValue = model->HSM2_pmuesti1;
    return(OK);
  case  HSM2_MOD_PMUESTI2:
    value->rValue = model->HSM2_pmuesti2;
    return(OK);
  case  HSM2_MOD_PMUESTI3:
    value->rValue = model->HSM2_pmuesti3;
    return(OK);
  case  HSM2_MOD_PNSUBPSTI1:
    value->rValue = model->HSM2_pnsubpsti1;
    return(OK);
  case  HSM2_MOD_PNSUBPSTI2:
    value->rValue = model->HSM2_pnsubpsti2;
    return(OK);
  case  HSM2_MOD_PNSUBPSTI3:
    value->rValue = model->HSM2_pnsubpsti3;
    return(OK);
  case HSM2_MOD_PNSUBCSTI1:
    value->rValue = model->HSM2_pnsubcsti1;
    return(OK);
  case HSM2_MOD_PNSUBCSTI2:
    value->rValue = model->HSM2_pnsubcsti2;
    return(OK);
  case HSM2_MOD_PNSUBCSTI3:
    value->rValue = model->HSM2_pnsubcsti3;
    return(OK);
  case  HSM2_MOD_PCGSO:
    value->rValue = model->HSM2_pcgso;
    return(OK);
  case  HSM2_MOD_PCGDO:
    value->rValue = model->HSM2_pcgdo;
    return(OK);
  case  HSM2_MOD_PJS0:
    value->rValue = model->HSM2_pjs0;
    return(OK);
  case  HSM2_MOD_PJS0SW:
    value->rValue = model->HSM2_pjs0sw;
    return(OK);
  case  HSM2_MOD_PNJ:
    value->rValue = model->HSM2_pnj;
    return(OK);
  case  HSM2_MOD_PCISBK:
    value->rValue = model->HSM2_pcisbk;
    return(OK);
  case  HSM2_MOD_PCLM1:
    value->rValue = model->HSM2_pclm1;
    return(OK);
  case  HSM2_MOD_PCLM2:
    value->rValue = model->HSM2_pclm2;
    return(OK);
  case  HSM2_MOD_PCLM3:
    value->rValue = model->HSM2_pclm3;
    return(OK);
  case  HSM2_MOD_PWFC:
    value->rValue = model->HSM2_pwfc;
    return(OK);
  case  HSM2_MOD_PGIDL1:
    value->rValue = model->HSM2_pgidl1;
    return(OK);
  case  HSM2_MOD_PGIDL2:
    value->rValue = model->HSM2_pgidl2;
    return(OK);
  case  HSM2_MOD_PGLEAK1:
    value->rValue = model->HSM2_pgleak1;
    return(OK);
  case  HSM2_MOD_PGLEAK2:
    value->rValue = model->HSM2_pgleak2;
    return(OK);
  case  HSM2_MOD_PGLEAK3:
    value->rValue = model->HSM2_pgleak3;
    return(OK);
  case  HSM2_MOD_PGLEAK6:
    value->rValue = model->HSM2_pgleak6;
    return(OK);
  case  HSM2_MOD_PGLKSD1:
    value->rValue = model->HSM2_pglksd1;
    return(OK);
  case  HSM2_MOD_PGLKSD2:
    value->rValue = model->HSM2_pglksd2;
    return(OK);
  case  HSM2_MOD_PGLKB1:
    value->rValue = model->HSM2_pglkb1;
    return(OK);
  case  HSM2_MOD_PGLKB2:
    value->rValue = model->HSM2_pglkb2;
    return(OK);
  case  HSM2_MOD_PNFTRP:
    value->rValue = model->HSM2_pnftrp;
    return(OK);
  case  HSM2_MOD_PNFALP:
    value->rValue = model->HSM2_pnfalp;
    return(OK);
  case  HSM2_MOD_PVDIFFJ:
    value->rValue = model->HSM2_pvdiffj;
    return(OK);
  case  HSM2_MOD_PIBPC1:
    value->rValue = model->HSM2_pibpc1;
    return(OK);
  case  HSM2_MOD_PIBPC2:
    value->rValue = model->HSM2_pibpc2;
    return(OK);

  case HSM2_MOD_VGS_MAX:
      value->rValue = model->HSM2vgsMax;
      return(OK);
  case HSM2_MOD_VGD_MAX:
      value->rValue = model->HSM2vgdMax;
      return(OK);
  case HSM2_MOD_VGB_MAX:
      value->rValue = model->HSM2vgbMax;
      return(OK);
  case HSM2_MOD_VDS_MAX:
      value->rValue = model->HSM2vdsMax;
      return(OK);
  case HSM2_MOD_VBS_MAX:
      value->rValue = model->HSM2vbsMax;
      return(OK);
  case HSM2_MOD_VBD_MAX:
      value->rValue = model->HSM2vbdMax;
      return(OK);
  case HSM2_MOD_VGSR_MAX:
      value->rValue = model->HSM2vgsrMax;
      return(OK);
  case HSM2_MOD_VGDR_MAX:
      value->rValue = model->HSM2vgdrMax;
      return(OK);
  case HSM2_MOD_VGBR_MAX:
      value->rValue = model->HSM2vgbrMax;
      return(OK);
  case HSM2_MOD_VBSR_MAX:
      value->rValue = model->HSM2vbsrMax;
      return(OK);
  case HSM2_MOD_VBDR_MAX:
      value->rValue = model->HSM2vbdrMax;
      return(OK);

  default:
    return(E_BADPARM);
  }
  /* NOTREACHED */
}


