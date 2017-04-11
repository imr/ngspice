/***********************************************************************

 HiSIM (Hiroshima University STARC IGFET Model)
 Copyright (C) 2014 Hiroshima University & STARC

 MODEL NAME : HiSIM_HV 
 ( VERSION : 2  SUBVERSION : 2  REVISION : 0 ) 
 Model Parameter 'VERSION' : 2.20
 FILE : hsmhvmask.c

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

int HSMHV2mAsk(
     CKTcircuit *ckt,
     GENmodel *inst,
     int which,
     IFvalue *value)
{
  HSMHV2model *model = (HSMHV2model *)inst;

  NG_IGNORE(ckt);

  switch (which) {
  case HSMHV2_MOD_NMOS:
    value->iValue = model->HSMHV2_type;
    return(OK);
  case  HSMHV2_MOD_PMOS:
    value->iValue = model->HSMHV2_type;
    return(OK);
  case  HSMHV2_MOD_LEVEL:
    value->iValue = model->HSMHV2_level;
    return(OK);
  case  HSMHV2_MOD_INFO:
    value->iValue = model->HSMHV2_info;
    return(OK);
  case HSMHV2_MOD_NOISE:
    value->iValue = model->HSMHV2_noise;
    return(OK);
  case HSMHV2_MOD_VERSION:
    value->sValue = model->HSMHV2_version;
    return(OK);
  case HSMHV2_MOD_SHOW:
    value->iValue = model->HSMHV2_show;
    return(OK);
  case  HSMHV2_MOD_CORSRD:
    value->iValue = model->HSMHV2_corsrd;
    return(OK);
  case  HSMHV2_MOD_CORG:
    value->iValue = model->HSMHV2_corg;
    return(OK);
  case  HSMHV2_MOD_COIPRV:
    value->iValue = model->HSMHV2_coiprv;
    return(OK);
  case  HSMHV2_MOD_COPPRV:
    value->iValue = model->HSMHV2_copprv;
    return(OK);
  case  HSMHV2_MOD_COADOV:
    value->iValue = model->HSMHV2_coadov;
    return(OK);
  case  HSMHV2_MOD_COISUB:
    value->iValue = model->HSMHV2_coisub;
    return(OK);
  case  HSMHV2_MOD_COIIGS:
    value->iValue = model->HSMHV2_coiigs;
    return(OK);
  case  HSMHV2_MOD_COGIDL:
    value->iValue = model->HSMHV2_cogidl;
    return(OK);
  case  HSMHV2_MOD_COOVLP:
    value->iValue = model->HSMHV2_coovlp;
    return(OK);
  case  HSMHV2_MOD_COOVLPS:
    value->iValue = model->HSMHV2_coovlps;
    return(OK);
  case  HSMHV2_MOD_COFLICK:
    value->iValue = model->HSMHV2_coflick;
    return(OK);
  case  HSMHV2_MOD_COISTI:
    value->iValue = model->HSMHV2_coisti;
    return(OK);
  case  HSMHV2_MOD_CONQS:
    value->iValue = model->HSMHV2_conqs;
    return(OK);
  case  HSMHV2_MOD_CORBNET:
    value->iValue = model->HSMHV2_corbnet;
    return(OK);
  case  HSMHV2_MOD_COTHRML:
    value->iValue = model->HSMHV2_cothrml;
    return(OK);
  case  HSMHV2_MOD_COIGN:
    value->iValue = model->HSMHV2_coign;
    return(OK);
  case  HSMHV2_MOD_CODFM:
    value->iValue = model->HSMHV2_codfm;
    return(OK);
  case  HSMHV2_MOD_COQOVSM:
    value->iValue = model->HSMHV2_coqovsm;
    return(OK);
  case  HSMHV2_MOD_COSELFHEAT: /* Self-heating model */
    value->iValue = model->HSMHV2_coselfheat;
    return(OK);
  case  HSMHV2_MOD_COSUBNODE: 
    value->iValue = model->HSMHV2_cosubnode;
    return(OK);
  case  HSMHV2_MOD_COSYM: /* Symmetry model for HV */
    value->iValue = model->HSMHV2_cosym;
    return(OK);
  case  HSMHV2_MOD_COTEMP:
    value->iValue = model->HSMHV2_cotemp;
    return(OK);
  case  HSMHV2_MOD_COLDRIFT:
    value->iValue = model->HSMHV2_coldrift;
    return(OK);
  case  HSMHV2_MOD_CORDRIFT:
    value->iValue = model->HSMHV2_cordrift;
    return(OK);
  case  HSMHV2_MOD_COERRREP:
    value->iValue = model->HSMHV2_coerrrep;
    return(OK);
  case  HSMHV2_MOD_CODEP:
    value->iValue = model->HSMHV2_codep;
    return(OK);
  case  HSMHV2_MOD_CODDLT:
    value->iValue = model->HSMHV2_coddlt;
    return(OK);
  case  HSMHV2_MOD_VMAX:
    value->rValue = model->HSMHV2_vmax;
    return(OK);
  case  HSMHV2_MOD_VMAXT1:
    value->rValue = model->HSMHV2_vmaxt1;
    return(OK);
  case  HSMHV2_MOD_VMAXT2:
    value->rValue = model->HSMHV2_vmaxt2;
    return(OK);
  case  HSMHV2_MOD_BGTMP1:
    value->rValue = model->HSMHV2_bgtmp1;
    return(OK);
  case  HSMHV2_MOD_BGTMP2:
    value->rValue = model->HSMHV2_bgtmp2;
    return(OK);
  case  HSMHV2_MOD_EG0:
    value->rValue = model->HSMHV2_eg0;
    return(OK);
  case  HSMHV2_MOD_TOX:
    value->rValue = model->HSMHV2_tox;
    return(OK);
  case  HSMHV2_MOD_XLD:
    value->rValue = model->HSMHV2_xld;
    return(OK);
  case  HSMHV2_MOD_LOVER:
    value->rValue = model->HSMHV2_lover;
    return(OK);
  case  HSMHV2_MOD_LOVERS:
    value->rValue = model->HSMHV2_lovers;
    return(OK);
  case  HSMHV2_MOD_RDOV11:
    value->rValue = model->HSMHV2_rdov11;
    return(OK);
  case  HSMHV2_MOD_RDOV12:
    value->rValue = model->HSMHV2_rdov12;
    return(OK);
  case  HSMHV2_MOD_RDOV13:
    value->rValue = model->HSMHV2_rdov13;
    return(OK);
  case  HSMHV2_MOD_RDSLP1:
    value->rValue = model->HSMHV2_rdslp1;
    return(OK);
  case  HSMHV2_MOD_RDICT1:
    value->rValue = model->HSMHV2_rdict1;
    return(OK);
  case  HSMHV2_MOD_RDSLP2:
    value->rValue = model->HSMHV2_rdslp2;
    return(OK);
  case  HSMHV2_MOD_RDICT2:
    value->rValue = model->HSMHV2_rdict2;
    return(OK);
  case  HSMHV2_MOD_LOVERLD:
    value->rValue = model->HSMHV2_loverld;
    return(OK);
  case  HSMHV2_MOD_LDRIFT1:
    value->rValue = model->HSMHV2_ldrift1;
    return(OK);
  case  HSMHV2_MOD_LDRIFT2:
    value->rValue = model->HSMHV2_ldrift2;
    return(OK);
  case  HSMHV2_MOD_LDRIFT1S:
    value->rValue = model->HSMHV2_ldrift1s;
    return(OK);
  case  HSMHV2_MOD_LDRIFT2S:
    value->rValue = model->HSMHV2_ldrift2s;
    return(OK);
  case  HSMHV2_MOD_SUBLD1:
    value->rValue = model->HSMHV2_subld1;
    return(OK);
  case  HSMHV2_MOD_SUBLD1L:
    value->rValue = model->HSMHV2_subld1l;
    return(OK);
  case  HSMHV2_MOD_SUBLD1LP:
    value->rValue = model->HSMHV2_subld1lp;
    return(OK);
  case  HSMHV2_MOD_SUBLD2:
    value->rValue = model->HSMHV2_subld2;
    return(OK);
  case  HSMHV2_MOD_XPDV:
    value->rValue = model->HSMHV2_xpdv;
    return(OK);
  case  HSMHV2_MOD_XPVDTH:
    value->rValue = model->HSMHV2_xpvdth;
    return(OK);
  case  HSMHV2_MOD_XPVDTHG:
    value->rValue = model->HSMHV2_xpvdthg;
    return(OK);
  case  HSMHV2_MOD_DDLTMAX: /* Vdseff */
    value->rValue = model->HSMHV2_ddltmax;
    return(OK);
  case  HSMHV2_MOD_DDLTSLP: /* Vdseff */
    value->rValue = model->HSMHV2_ddltslp;
    return(OK);
  case  HSMHV2_MOD_DDLTICT: /* Vdseff */
    value->rValue = model->HSMHV2_ddltict;
    return(OK);
  case  HSMHV2_MOD_VFBOVER:
    value->rValue = model->HSMHV2_vfbover;
    return(OK);
  case  HSMHV2_MOD_NOVER:
    value->rValue = model->HSMHV2_nover;
    return(OK);
  case  HSMHV2_MOD_NOVERS:
    value->rValue = model->HSMHV2_novers;
    return(OK);
  case  HSMHV2_MOD_XWD:
    value->rValue = model->HSMHV2_xwd;
    return(OK);
  case  HSMHV2_MOD_XWDC:
    value->rValue = model->HSMHV2_xwdc;
    return(OK);
  case  HSMHV2_MOD_XL:
    value->rValue = model->HSMHV2_xl;
    return(OK);
  case  HSMHV2_MOD_XW:
    value->rValue = model->HSMHV2_xw;
    return(OK);
  case  HSMHV2_MOD_SAREF:
    value->rValue = model->HSMHV2_saref;
    return(OK);
  case  HSMHV2_MOD_SBREF:
    value->rValue = model->HSMHV2_sbref;
    return(OK);
  case  HSMHV2_MOD_LL:
    value->rValue = model->HSMHV2_ll;
    return(OK);
  case  HSMHV2_MOD_LLD:
    value->rValue = model->HSMHV2_lld;
    return(OK);
  case  HSMHV2_MOD_LLN:
    value->rValue = model->HSMHV2_lln;
    return(OK);
  case  HSMHV2_MOD_WL:
    value->rValue = model->HSMHV2_wl;
    return(OK);
  case  HSMHV2_MOD_WL1:
    value->rValue = model->HSMHV2_wl1;
    return(OK);
  case  HSMHV2_MOD_WL1P:
    value->rValue = model->HSMHV2_wl1p;
    return(OK);
  case  HSMHV2_MOD_WL2:
    value->rValue = model->HSMHV2_wl2;
    return(OK);
  case  HSMHV2_MOD_WL2P:
    value->rValue = model->HSMHV2_wl2p;
    return(OK);
  case  HSMHV2_MOD_WLD:
    value->rValue = model->HSMHV2_wld;
    return(OK);
  case  HSMHV2_MOD_WLN:
    value->rValue = model->HSMHV2_wln;
    return(OK);
  case  HSMHV2_MOD_XQY:
    value->rValue = model->HSMHV2_xqy;
    return(OK);
  case  HSMHV2_MOD_XQY1:
    value->rValue = model->HSMHV2_xqy1;
    return(OK);
  case  HSMHV2_MOD_XQY2:
    value->rValue = model->HSMHV2_xqy2;
    return(OK);
  case  HSMHV2_MOD_RS:
    value->rValue = model->HSMHV2_rs;
    return(OK);
  case  HSMHV2_MOD_RD:
    value->rValue = model->HSMHV2_rd;
    return(OK);
  case  HSMHV2_MOD_RSH:
    value->rValue = model->HSMHV2_rsh;
    return(OK);
  case  HSMHV2_MOD_RSHG:
    value->rValue = model->HSMHV2_rshg;
    return(OK);
  case  HSMHV2_MOD_VFBC:
    value->rValue = model->HSMHV2_vfbc;
    return(OK);
  case  HSMHV2_MOD_VBI:
    value->rValue = model->HSMHV2_vbi;
    return(OK);
  case  HSMHV2_MOD_NSUBC:
    value->rValue = model->HSMHV2_nsubc;
      return(OK);
  case  HSMHV2_MOD_PARL2:
    value->rValue = model->HSMHV2_parl2;
    return(OK);
  case  HSMHV2_MOD_LP:
    value->rValue = model->HSMHV2_lp;
    return(OK);
  case  HSMHV2_MOD_NSUBP:
    value->rValue = model->HSMHV2_nsubp;
    return(OK);
  case  HSMHV2_MOD_NSUBP0:
    value->rValue = model->HSMHV2_nsubp0;
    return(OK);
  case  HSMHV2_MOD_NSUBWP:
    value->rValue = model->HSMHV2_nsubwp;
    return(OK);
  case  HSMHV2_MOD_SCP1:
    value->rValue = model->HSMHV2_scp1;
    return(OK);
  case  HSMHV2_MOD_SCP2:
    value->rValue = model->HSMHV2_scp2;
    return(OK);
  case  HSMHV2_MOD_SCP3:
    value->rValue = model->HSMHV2_scp3;
    return(OK);
  case  HSMHV2_MOD_SC1:
    value->rValue = model->HSMHV2_sc1;
    return(OK);
  case  HSMHV2_MOD_SC2:
    value->rValue = model->HSMHV2_sc2;
    return(OK);
  case  HSMHV2_MOD_SC3:
    value->rValue = model->HSMHV2_sc3;
    return(OK);
  case  HSMHV2_MOD_SC4:
    value->rValue = model->HSMHV2_sc4;
    return(OK);
  case  HSMHV2_MOD_PGD1:
    value->rValue = model->HSMHV2_pgd1;
    return(OK);
  case  HSMHV2_MOD_PGD2:
    value->rValue = model->HSMHV2_pgd2;
    return(OK);
  case  HSMHV2_MOD_PGD4:
    value->rValue = model->HSMHV2_pgd4;
    return(OK);
  case  HSMHV2_MOD_NDEP:
    value->rValue = model->HSMHV2_ndep;
    return(OK);
  case  HSMHV2_MOD_NDEPL:
    value->rValue = model->HSMHV2_ndepl;
    return(OK);
  case  HSMHV2_MOD_NDEPLP:
    value->rValue = model->HSMHV2_ndeplp;
    return(OK);
  case  HSMHV2_MOD_NINV:
    value->rValue = model->HSMHV2_ninv;
    return(OK);
  case  HSMHV2_MOD_MUECB0:
    value->rValue = model->HSMHV2_muecb0;
    return(OK);
  case  HSMHV2_MOD_MUECB1:
    value->rValue = model->HSMHV2_muecb1;
    return(OK);
  case  HSMHV2_MOD_MUEPH1:
    value->rValue = model->HSMHV2_mueph1;
    return(OK);
  case  HSMHV2_MOD_MUEPH0:
    value->rValue = model->HSMHV2_mueph0;
    return(OK);
  case  HSMHV2_MOD_MUEPHW:
    value->rValue = model->HSMHV2_muephw;
    return(OK);
  case  HSMHV2_MOD_MUEPWP:
    value->rValue = model->HSMHV2_muepwp;
    return(OK);
  case  HSMHV2_MOD_MUEPHL:
    value->rValue = model->HSMHV2_muephl;
    return(OK);
  case  HSMHV2_MOD_MUEPLP:
    value->rValue = model->HSMHV2_mueplp;
    return(OK);
  case  HSMHV2_MOD_MUEPHS:
    value->rValue = model->HSMHV2_muephs;
    return(OK);
  case  HSMHV2_MOD_MUEPSP:
    value->rValue = model->HSMHV2_muepsp;
    return(OK);
  case  HSMHV2_MOD_VTMP:
    value->rValue = model->HSMHV2_vtmp;
    return(OK);
  case  HSMHV2_MOD_WVTH0:
    value->rValue = model->HSMHV2_wvth0;
    return(OK);
  case  HSMHV2_MOD_MUESR1:
    value->rValue = model->HSMHV2_muesr1;
    return(OK);
  case  HSMHV2_MOD_MUESR0:
    value->rValue = model->HSMHV2_muesr0;
    return(OK);
  case  HSMHV2_MOD_MUESRL:
    value->rValue = model->HSMHV2_muesrl;
    return(OK);
  case  HSMHV2_MOD_MUESLP:
    value->rValue = model->HSMHV2_mueslp;
    return(OK);
  case  HSMHV2_MOD_MUESRW:
    value->rValue = model->HSMHV2_muesrw;
    return(OK);
  case  HSMHV2_MOD_MUESWP:
    value->rValue = model->HSMHV2_mueswp;
    return(OK);
  case  HSMHV2_MOD_BB:
    value->rValue = model->HSMHV2_bb;
    return(OK);
  case  HSMHV2_MOD_SUB1:
    value->rValue = model->HSMHV2_sub1;
    return(OK);
  case  HSMHV2_MOD_SUB2:
    value->rValue = model->HSMHV2_sub2;
    return(OK);
  case  HSMHV2_MOD_SVGS:
    value->rValue = model->HSMHV2_svgs;
    return(OK);
  case  HSMHV2_MOD_SVGSL:
    value->rValue = model->HSMHV2_svgsl;
    return(OK);
  case  HSMHV2_MOD_SVGSLP:
    value->rValue = model->HSMHV2_svgslp;
    return(OK);
  case  HSMHV2_MOD_SVGSW:
    value->rValue = model->HSMHV2_svgsw;
    return(OK);
  case  HSMHV2_MOD_SVGSWP:
    value->rValue = model->HSMHV2_svgswp;
    return(OK);
  case  HSMHV2_MOD_SVBS:
    value->rValue = model->HSMHV2_svbs;
    return(OK);
  case  HSMHV2_MOD_SVBSL:
    value->rValue = model->HSMHV2_svbsl;
    return(OK);
  case  HSMHV2_MOD_SVBSLP:
    value->rValue = model->HSMHV2_svbslp;
    return(OK);
  case  HSMHV2_MOD_SVDS:
    value->rValue = model->HSMHV2_svds;
    return(OK);
  case  HSMHV2_MOD_SLG:
    value->rValue = model->HSMHV2_slg;
    return(OK);
  case  HSMHV2_MOD_SLGL:
    value->rValue = model->HSMHV2_slgl;
    return(OK);
  case  HSMHV2_MOD_SLGLP:
    value->rValue = model->HSMHV2_slglp;
    return(OK);
  case  HSMHV2_MOD_SUB1L:
    value->rValue = model->HSMHV2_sub1l;
    return(OK);
  case  HSMHV2_MOD_SUB1LP:
    value->rValue = model->HSMHV2_sub1lp;
    return(OK);
  case  HSMHV2_MOD_SUB2L:
    value->rValue = model->HSMHV2_sub2l;
    return(OK);
  case  HSMHV2_MOD_FN1:
    value->rValue = model->HSMHV2_fn1;
    return(OK);
  case  HSMHV2_MOD_FN2:
    value->rValue = model->HSMHV2_fn2;
    return(OK);
  case  HSMHV2_MOD_FN3:
    value->rValue = model->HSMHV2_fn3;
    return(OK);
  case  HSMHV2_MOD_FVBS:
    value->rValue = model->HSMHV2_fvbs;
    return(OK);
  case  HSMHV2_MOD_NSTI:
    value->rValue = model->HSMHV2_nsti;
    return(OK);
  case  HSMHV2_MOD_WSTI:
    value->rValue = model->HSMHV2_wsti;
    return(OK);
  case  HSMHV2_MOD_WSTIL:
    value->rValue = model->HSMHV2_wstil;
    return(OK);
  case  HSMHV2_MOD_WSTILP:
    value->rValue = model->HSMHV2_wstilp;
    return(OK);
  case  HSMHV2_MOD_WSTIW:
    value->rValue = model->HSMHV2_wstiw;
    return(OK);
  case  HSMHV2_MOD_WSTIWP:
    value->rValue = model->HSMHV2_wstiwp;
    return(OK);
  case  HSMHV2_MOD_SCSTI1:
    value->rValue = model->HSMHV2_scsti1;
    return(OK);
  case  HSMHV2_MOD_SCSTI2:
    value->rValue = model->HSMHV2_scsti2;
    return(OK);
  case  HSMHV2_MOD_VTHSTI:
    value->rValue = model->HSMHV2_vthsti;
    return(OK);
  case  HSMHV2_MOD_VDSTI:
    value->rValue = model->HSMHV2_vdsti;
    return(OK);
  case  HSMHV2_MOD_MUESTI1:
    value->rValue = model->HSMHV2_muesti1;
    return(OK);
  case  HSMHV2_MOD_MUESTI2:
    value->rValue = model->HSMHV2_muesti2;
    return(OK);
  case  HSMHV2_MOD_MUESTI3:
    value->rValue = model->HSMHV2_muesti3;
    return(OK);
  case  HSMHV2_MOD_NSUBPSTI1:
    value->rValue = model->HSMHV2_nsubpsti1;
    return(OK);
  case  HSMHV2_MOD_NSUBPSTI2:
    value->rValue = model->HSMHV2_nsubpsti2;
    return(OK);
  case  HSMHV2_MOD_NSUBPSTI3:
    value->rValue = model->HSMHV2_nsubpsti3;
    return(OK);
  case  HSMHV2_MOD_LPEXT:
    value->rValue = model->HSMHV2_lpext;
    return(OK);
  case  HSMHV2_MOD_NPEXT:
    value->rValue = model->HSMHV2_npext;
    return(OK);
  case  HSMHV2_MOD_SCP22:
    value->rValue = model->HSMHV2_scp22;
    return(OK);
  case  HSMHV2_MOD_SCP21:
    value->rValue = model->HSMHV2_scp21;
    return(OK);
  case  HSMHV2_MOD_BS1:
    value->rValue = model->HSMHV2_bs1;
    return(OK);
  case  HSMHV2_MOD_BS2:
    value->rValue = model->HSMHV2_bs2;
    return(OK);
  case  HSMHV2_MOD_CGSO:
    value->rValue = model->HSMHV2_cgso;
    return(OK);
  case  HSMHV2_MOD_CGDO:
    value->rValue = model->HSMHV2_cgdo;
    return(OK);
  case  HSMHV2_MOD_CGBO:
    value->rValue = model->HSMHV2_cgbo;
    return(OK);
  case  HSMHV2_MOD_TPOLY:
    value->rValue = model->HSMHV2_tpoly;
    return(OK);
  case  HSMHV2_MOD_JS0:
    value->rValue = model->HSMHV2_js0;
    return(OK);
  case  HSMHV2_MOD_JS0SW:
    value->rValue = model->HSMHV2_js0sw;
    return(OK);
  case  HSMHV2_MOD_NJ:
    value->rValue = model->HSMHV2_nj;
    return(OK);
  case  HSMHV2_MOD_NJSW:
    value->rValue = model->HSMHV2_njsw;
    return(OK);
  case  HSMHV2_MOD_XTI:
    value->rValue = model->HSMHV2_xti;
    return(OK);
  case  HSMHV2_MOD_CJ:
    value->rValue = model->HSMHV2_cj;
    return(OK);
  case  HSMHV2_MOD_CJSW:
    value->rValue = model->HSMHV2_cjsw;
    return(OK);
  case  HSMHV2_MOD_CJSWG:
    value->rValue = model->HSMHV2_cjswg;
    return(OK);
  case  HSMHV2_MOD_MJ:
    value->rValue = model->HSMHV2_mj;
    return(OK);
  case  HSMHV2_MOD_MJSW:
    value->rValue = model->HSMHV2_mjsw;
    return(OK);
  case  HSMHV2_MOD_MJSWG:
    value->rValue = model->HSMHV2_mjswg;
    return(OK);
  case  HSMHV2_MOD_PB:
    value->rValue = model->HSMHV2_pb;
    return(OK);
  case  HSMHV2_MOD_PBSW:
    value->rValue = model->HSMHV2_pbsw;
    return(OK);
  case  HSMHV2_MOD_PBSWG:
    value->rValue = model->HSMHV2_pbswg;
    return(OK);
  case  HSMHV2_MOD_XTI2:
    value->rValue = model->HSMHV2_xti2;
    return(OK);
  case  HSMHV2_MOD_CISB:
    value->rValue = model->HSMHV2_cisb;
    return(OK);
  case  HSMHV2_MOD_CVB:
    value->rValue = model->HSMHV2_cvb;
    return(OK);
  case  HSMHV2_MOD_CTEMP:
    value->rValue = model->HSMHV2_ctemp;
    return(OK);
  case  HSMHV2_MOD_CISBK:
    value->rValue = model->HSMHV2_cisbk;
    return(OK);
  case  HSMHV2_MOD_CVBK:
    value->rValue = model->HSMHV2_cvbk;
    return(OK);
  case  HSMHV2_MOD_DIVX:
    value->rValue = model->HSMHV2_divx;
    return(OK);
  case  HSMHV2_MOD_CLM1:
    value->rValue = model->HSMHV2_clm1;
    return(OK);
  case  HSMHV2_MOD_CLM2:
    value->rValue = model->HSMHV2_clm2;
    return(OK);
  case  HSMHV2_MOD_CLM3:
    value->rValue = model->HSMHV2_clm3;
    return(OK);
  case  HSMHV2_MOD_CLM5:
    value->rValue = model->HSMHV2_clm5;
    return(OK);
  case  HSMHV2_MOD_CLM6:
    value->rValue = model->HSMHV2_clm6;
    return(OK);
  case  HSMHV2_MOD_MUETMP:
    value->rValue = model->HSMHV2_muetmp;
    return(OK);
  case  HSMHV2_MOD_VOVER:
    value->rValue = model->HSMHV2_vover;
    return(OK);
  case  HSMHV2_MOD_VOVERP:
    value->rValue = model->HSMHV2_voverp;
    return(OK);
  case  HSMHV2_MOD_VOVERS:
    value->rValue = model->HSMHV2_vovers;
    return(OK);
  case  HSMHV2_MOD_VOVERSP:
    value->rValue = model->HSMHV2_voversp;
    return(OK);
  case  HSMHV2_MOD_WFC:
    value->rValue = model->HSMHV2_wfc;
    return(OK);
  case  HSMHV2_MOD_NSUBCW:
    value->rValue = model->HSMHV2_nsubcw;
    return(OK);
  case  HSMHV2_MOD_NSUBCWP:
    value->rValue = model->HSMHV2_nsubcwp;
    return(OK);
  case  HSMHV2_MOD_QME1:
    value->rValue = model->HSMHV2_qme1;
    return(OK);
  case  HSMHV2_MOD_QME2:
    value->rValue = model->HSMHV2_qme2;
    return(OK);
  case  HSMHV2_MOD_QME3:
    value->rValue = model->HSMHV2_qme3;
    return(OK);
  case  HSMHV2_MOD_GIDL1:
    value->rValue = model->HSMHV2_gidl1;
    return(OK);
  case  HSMHV2_MOD_GIDL2:
    value->rValue = model->HSMHV2_gidl2;
    return(OK);
  case  HSMHV2_MOD_GIDL3:
    value->rValue = model->HSMHV2_gidl3;
    return(OK);
  case  HSMHV2_MOD_GIDL4:
    value->rValue = model->HSMHV2_gidl4;
    return(OK);
  case  HSMHV2_MOD_GIDL5:
    value->rValue = model->HSMHV2_gidl5;
    return(OK);
  case  HSMHV2_MOD_GLEAK1:
    value->rValue = model->HSMHV2_gleak1;
    return(OK);
  case  HSMHV2_MOD_GLEAK2:
    value->rValue = model->HSMHV2_gleak2;
    return(OK);
  case  HSMHV2_MOD_GLEAK3:
    value->rValue = model->HSMHV2_gleak3;
    return(OK);
  case  HSMHV2_MOD_GLEAK4:
    value->rValue = model->HSMHV2_gleak4;
    return(OK);
  case  HSMHV2_MOD_GLEAK5:
    value->rValue = model->HSMHV2_gleak5;
    return(OK);
  case  HSMHV2_MOD_GLEAK6:
    value->rValue = model->HSMHV2_gleak6;
    return(OK);
  case  HSMHV2_MOD_GLEAK7:
    value->rValue = model->HSMHV2_gleak7;
    return(OK);
  case  HSMHV2_MOD_GLPART1:
    value->rValue = model->HSMHV2_glpart1;
    return(OK);
  case  HSMHV2_MOD_GLKSD1:
    value->rValue = model->HSMHV2_glksd1;
    return(OK);
  case  HSMHV2_MOD_GLKSD2:
    value->rValue = model->HSMHV2_glksd2;
    return(OK);
  case  HSMHV2_MOD_GLKSD3:
    value->rValue = model->HSMHV2_glksd3;
    return(OK);
  case  HSMHV2_MOD_GLKB1:
    value->rValue = model->HSMHV2_glkb1;
    return(OK);
  case  HSMHV2_MOD_GLKB2:
    value->rValue = model->HSMHV2_glkb2;
    return(OK);
  case  HSMHV2_MOD_GLKB3:
    value->rValue = model->HSMHV2_glkb3;
    return(OK);
  case  HSMHV2_MOD_EGIG:
    value->rValue = model->HSMHV2_egig;
    return(OK);
  case  HSMHV2_MOD_IGTEMP2:
    value->rValue = model->HSMHV2_igtemp2;
    return(OK);
  case  HSMHV2_MOD_IGTEMP3:
    value->rValue = model->HSMHV2_igtemp3;
    return(OK);
  case  HSMHV2_MOD_VZADD0:
    value->rValue = model->HSMHV2_vzadd0;
    return(OK);
  case  HSMHV2_MOD_PZADD0:
    value->rValue = model->HSMHV2_pzadd0;
    return(OK);
  case  HSMHV2_MOD_NFTRP:
    value->rValue = model->HSMHV2_nftrp;
    return(OK);
  case  HSMHV2_MOD_NFALP:
    value->rValue = model->HSMHV2_nfalp;
    return(OK);
  case  HSMHV2_MOD_CIT:
    value->rValue = model->HSMHV2_cit;
    return(OK);
  case  HSMHV2_MOD_FALPH:
    value->rValue = model->HSMHV2_falph;
    return(OK);
  case  HSMHV2_MOD_KAPPA:
    value->rValue = model->HSMHV2_kappa;
    return(OK);
  case  HSMHV2_MOD_VDIFFJ:
    value->rValue = model->HSMHV2_vdiffj;
    return(OK);
  case  HSMHV2_MOD_DLY1:
    value->rValue = model->HSMHV2_dly1;
    return(OK);
  case  HSMHV2_MOD_DLY2:
    value->rValue = model->HSMHV2_dly2;
    return(OK);
  case  HSMHV2_MOD_DLY3:
    value->rValue = model->HSMHV2_dly3;
    return(OK);
  case  HSMHV2_MOD_DLYOV:
    value->rValue = model->HSMHV2_dlyov;
    return(OK);


  case  HSMHV2_MOD_TNOM:
    value->rValue = model->HSMHV2_tnom;
    return(OK);
  case  HSMHV2_MOD_OVSLP:
    value->rValue = model->HSMHV2_ovslp;
    return(OK);
  case  HSMHV2_MOD_OVMAG:
    value->rValue = model->HSMHV2_ovmag;
    return(OK);
  case  HSMHV2_MOD_GBMIN:
    value->rValue = model->HSMHV2_gbmin;
    return(OK);
  case  HSMHV2_MOD_RBPB:
    value->rValue = model->HSMHV2_rbpb;
    return(OK);
  case  HSMHV2_MOD_RBPD:
    value->rValue = model->HSMHV2_rbpd;
    return(OK);
  case  HSMHV2_MOD_RBPS:
    value->rValue = model->HSMHV2_rbps;
    return(OK);
  case  HSMHV2_MOD_RBDB:
    value->rValue = model->HSMHV2_rbdb;
    return(OK);
  case  HSMHV2_MOD_RBSB:
    value->rValue = model->HSMHV2_rbsb;
    return(OK);
  case  HSMHV2_MOD_IBPC1:
    value->rValue = model->HSMHV2_ibpc1;
    return(OK);
  case  HSMHV2_MOD_IBPC1L:
    value->rValue = model->HSMHV2_ibpc1l;
    return(OK);
  case  HSMHV2_MOD_IBPC1LP:
    value->rValue = model->HSMHV2_ibpc1lp;
    return(OK);
  case  HSMHV2_MOD_IBPC2:
    value->rValue = model->HSMHV2_ibpc2;
    return(OK);
  case  HSMHV2_MOD_MPHDFM:
    value->rValue = model->HSMHV2_mphdfm;
    return(OK);

  case  HSMHV2_MOD_PTL:
    value->rValue = model->HSMHV2_ptl;
    return(OK);
  case  HSMHV2_MOD_PTP:
    value->rValue = model->HSMHV2_ptp;
    return(OK);
  case  HSMHV2_MOD_PT2:
    value->rValue = model->HSMHV2_pt2;
    return(OK);
  case  HSMHV2_MOD_PTLP:
    value->rValue = model->HSMHV2_ptlp;
    return(OK);
  case  HSMHV2_MOD_GDL:
    value->rValue = model->HSMHV2_gdl;
    return(OK);
  case  HSMHV2_MOD_GDLP:
    value->rValue = model->HSMHV2_gdlp;
    return(OK);

  case  HSMHV2_MOD_GDLD:
    value->rValue = model->HSMHV2_gdld;
    return(OK);
  case  HSMHV2_MOD_PT4:
    value->rValue = model->HSMHV2_pt4;
    return(OK);
  case  HSMHV2_MOD_PT4P:
    value->rValue = model->HSMHV2_pt4p;
    return(OK);
  case  HSMHV2_MOD_RDVG11:
    value->rValue = model->HSMHV2_rdvg11;
    return(OK);
  case  HSMHV2_MOD_RDVG12:
    value->rValue = model->HSMHV2_rdvg12;
    return(OK);
  case  HSMHV2_MOD_RTH0: /* Self-heating model */
    value->rValue = model->HSMHV2_rth0;
    return(OK);
  case  HSMHV2_MOD_CTH0: /* Self-heating model */
    value->rValue = model->HSMHV2_cth0;
    return(OK);
  case  HSMHV2_MOD_POWRAT: /* Self-heating model */
    value->rValue = model->HSMHV2_powrat;
    return(OK);
  case  HSMHV2_MOD_RTHTEMP1: /* Self-heating model */
    value->rValue = model->HSMHV2_rthtemp1;
    return(OK);
  case  HSMHV2_MOD_RTHTEMP2: /* Self-heating model */
    value->rValue = model->HSMHV2_rthtemp2;
    return(OK);
  case  HSMHV2_MOD_PRATTEMP1: /* Self-heating model */
    value->rValue = model->HSMHV2_prattemp1;
    return(OK);
  case  HSMHV2_MOD_PRATTEMP2: /* Self-heating model */
    value->rValue = model->HSMHV2_prattemp2;
    return(OK);



  case  HSMHV2_MOD_TCJBD: /* Self-heating model */
    value->rValue = model->HSMHV2_tcjbd;
    return(OK);
  case  HSMHV2_MOD_TCJBS: /* Self-heating model */
    value->rValue = model->HSMHV2_tcjbs;
    return(OK);
  case  HSMHV2_MOD_TCJBDSW: /* Self-heating model */
    value->rValue = model->HSMHV2_tcjbdsw;
    return(OK);
  case  HSMHV2_MOD_TCJBSSW: /* Self-heating model */
    value->rValue = model->HSMHV2_tcjbssw;
    return(OK);
  case  HSMHV2_MOD_TCJBDSWG: /* Self-heating model */
    value->rValue = model->HSMHV2_tcjbdswg;
    return(OK);
  case  HSMHV2_MOD_TCJBSSWG: /* Self-heating model */
    value->rValue = model->HSMHV2_tcjbsswg;
    return(OK);
/*   case HSMHV2_MOD_WTH0:                 */
/*     value->rValue = model->HSMHV2_wth0; */
/*     return(OK);                       */
  case  HSMHV2_MOD_QDFTVD:
    value->rValue = model->HSMHV2_qdftvd;
    return(OK);
  case  HSMHV2_MOD_XLDLD:
    value->rValue = model->HSMHV2_xldld;
    return(OK);
  case  HSMHV2_MOD_XWDLD:
    value->rValue = model->HSMHV2_xwdld;
    return(OK);
  case  HSMHV2_MOD_RDVD:
    value->rValue = model->HSMHV2_rdvd;
    return(OK);
  case  HSMHV2_MOD_RD20:
    value->rValue = model->HSMHV2_rd20;
    return(OK);
  case  HSMHV2_MOD_RD21:
    value->rValue = model->HSMHV2_rd21;
    return(OK);
  case  HSMHV2_MOD_RD22:
    value->rValue = model->HSMHV2_rd22;
    return(OK);
  case  HSMHV2_MOD_RD22D:
    value->rValue = model->HSMHV2_rd22d;
    return(OK);
  case  HSMHV2_MOD_RD23:
    value->rValue = model->HSMHV2_rd23;
    return(OK);
  case  HSMHV2_MOD_RD24:
    value->rValue = model->HSMHV2_rd24;
    return(OK);
  case  HSMHV2_MOD_RD25:
    value->rValue = model->HSMHV2_rd25;
    return(OK);
  case  HSMHV2_MOD_RDVDL:
    value->rValue = model->HSMHV2_rdvdl;
    return(OK);
  case  HSMHV2_MOD_RDVDLP:
    value->rValue = model->HSMHV2_rdvdlp;
    return(OK);
  case  HSMHV2_MOD_RDVDS:
    value->rValue = model->HSMHV2_rdvds;
    return(OK);
  case  HSMHV2_MOD_RDVDSP:
    value->rValue = model->HSMHV2_rdvdsp;
    return(OK);
  case  HSMHV2_MOD_RD23L:
    value->rValue = model->HSMHV2_rd23l;
    return(OK);
  case  HSMHV2_MOD_RD23LP:
    value->rValue = model->HSMHV2_rd23lp;
    return(OK);
  case  HSMHV2_MOD_RD23S:
    value->rValue = model->HSMHV2_rd23s;
    return(OK);
  case  HSMHV2_MOD_RD23SP:
    value->rValue = model->HSMHV2_rd23sp;
    return(OK);
  case  HSMHV2_MOD_RDS:
    value->rValue = model->HSMHV2_rds;
    return(OK);
  case  HSMHV2_MOD_RDSP:
    value->rValue = model->HSMHV2_rdsp;
    return(OK);
  case  HSMHV2_MOD_RDTEMP1:
    value->rValue = model->HSMHV2_rdtemp1;
    return(OK);
  case  HSMHV2_MOD_RDTEMP2:
    value->rValue = model->HSMHV2_rdtemp2;
    return(OK);
  case  HSMHV2_MOD_RTH0R:
    value->rValue = model->HSMHV2_rth0r;
    return(OK);
  case  HSMHV2_MOD_RDVDTEMP1:
    value->rValue = model->HSMHV2_rdvdtemp1;
    return(OK);
  case  HSMHV2_MOD_RDVDTEMP2:
    value->rValue = model->HSMHV2_rdvdtemp2;
    return(OK);
  case  HSMHV2_MOD_RTH0W:
    value->rValue = model->HSMHV2_rth0w;
    return(OK);
  case  HSMHV2_MOD_RTH0WP:
    value->rValue = model->HSMHV2_rth0wp;
    return(OK);
  case  HSMHV2_MOD_CVDSOVER:
    value->rValue = model->HSMHV2_cvdsover;
    return(OK);

  case  HSMHV2_MOD_NINVD:
    value->rValue = model->HSMHV2_ninvd;
    return(OK);
  case  HSMHV2_MOD_NINVDW:
    value->rValue = model->HSMHV2_ninvdw;
    return(OK);
  case  HSMHV2_MOD_NINVDWP:
    value->rValue = model->HSMHV2_ninvdwp;
    return(OK);
  case  HSMHV2_MOD_NINVDT1:
    value->rValue = model->HSMHV2_ninvdt1;
    return(OK);
  case  HSMHV2_MOD_NINVDT2:
    value->rValue = model->HSMHV2_ninvdt2;
    return(OK);
  case  HSMHV2_MOD_VBSMIN:
    value->rValue = model->HSMHV2_vbsmin;
    return(OK);
  case  HSMHV2_MOD_RDVB:
    value->rValue = model->HSMHV2_rdvb;
    return(OK);
  case  HSMHV2_MOD_RTH0NF:
    value->rValue = model->HSMHV2_rth0nf;
    return(OK);

  case  HSMHV2_MOD_RDVSUB:
    value->rValue = model->HSMHV2_rdvsub;
    return(OK);
  case  HSMHV2_MOD_RDVDSUB:
    value->rValue = model->HSMHV2_rdvdsub;
    return(OK);
  case  HSMHV2_MOD_DDRIFT:
    value->rValue = model->HSMHV2_ddrift;
    return(OK);
  case  HSMHV2_MOD_VBISUB:
    value->rValue = model->HSMHV2_vbisub;
    return(OK);
  case  HSMHV2_MOD_NSUBSUB:
    value->rValue = model->HSMHV2_nsubsub;
    return(OK);

  case  HSMHV2_MOD_RDRMUE:
    value->rValue = model->HSMHV2_rdrmue;
    return(OK);
  case  HSMHV2_MOD_RDRVMAX:
    value->rValue = model->HSMHV2_rdrvmax;
    return(OK);
  case  HSMHV2_MOD_RDRMUETMP:
    value->rValue = model->HSMHV2_rdrmuetmp;
    return(OK);
  case  HSMHV2_MOD_RDRVTMP:
    value->rValue = model->HSMHV2_rdrvtmp;
    return(OK);
  case  HSMHV2_MOD_NDEPM:
    value->rValue = model->HSMHV2_ndepm;
    return(OK);
  case  HSMHV2_MOD_TNDEP:
    value->rValue = model->HSMHV2_tndep;
    return(OK);
  case  HSMHV2_MOD_DEPMUE0:
    value->rValue = model->HSMHV2_depmue0;
    return(OK);
  case  HSMHV2_MOD_DEPMUE1:
    value->rValue = model->HSMHV2_depmue1;
    return(OK);
  case  HSMHV2_MOD_DEPMUEBACK0:
    value->rValue = model->HSMHV2_depmueback0;
    return(OK);
  case  HSMHV2_MOD_DEPMUEBACK1:
    value->rValue = model->HSMHV2_depmueback1;
    return(OK);
  case  HSMHV2_MOD_DEPLEAK:
    value->rValue = model->HSMHV2_depleak;
    return(OK);
  case  HSMHV2_MOD_DEPETA:
    value->rValue = model->HSMHV2_depeta;
    return(OK);
  case  HSMHV2_MOD_DEPVMAX:
    value->rValue = model->HSMHV2_depvmax;
    return(OK);
  case  HSMHV2_MOD_DEPVDSEF1:
    value->rValue = model->HSMHV2_depvdsef1;
    return(OK);
  case  HSMHV2_MOD_DEPVDSEF2:
    value->rValue = model->HSMHV2_depvdsef2;
    return(OK);
  case  HSMHV2_MOD_DEPMUEPH0:
    value->rValue = model->HSMHV2_depmueph0;
    return(OK);
  case  HSMHV2_MOD_DEPMUEPH1:
    value->rValue = model->HSMHV2_depmueph1;
    return(OK);
  case  HSMHV2_MOD_DEPBB:
    value->rValue = model->HSMHV2_depbb;
    return(OK);
  case  HSMHV2_MOD_DEPVTMP:
    value->rValue = model->HSMHV2_depvtmp;
    return(OK);
  case  HSMHV2_MOD_DEPMUETMP:
    value->rValue = model->HSMHV2_depmuetmp;
    return(OK);

  case  HSMHV2_MOD_ISBREAK:
    value->rValue = model->HSMHV2_isbreak;
    return(OK);
  case  HSMHV2_MOD_RWELL:
    value->rValue = model->HSMHV2_rwell;
    return(OK);


/*   case  HSMHV2_MOD_RDRVMAXT1: */
/*     value->rValue = model->HSMHV2_rdrvmaxt1; */
/*     return(OK); */
/*   case  HSMHV2_MOD_RDRVMAXT2: */
/*     value->rValue = model->HSMHV2_rdrvmaxt2; */
/*     return(OK); */
  case  HSMHV2_MOD_RDRDJUNC:
    value->rValue = model->HSMHV2_rdrdjunc;
    return(OK);
  case  HSMHV2_MOD_RDRCX:
    value->rValue = model->HSMHV2_rdrcx;
    return(OK);
  case  HSMHV2_MOD_RDRCAR:
    value->rValue = model->HSMHV2_rdrcar;
    return(OK);
  case  HSMHV2_MOD_RDRDL1:
    value->rValue = model->HSMHV2_rdrdl1;
    return(OK);
  case  HSMHV2_MOD_RDRDL2:
    value->rValue = model->HSMHV2_rdrdl2;
    return(OK);
  case  HSMHV2_MOD_RDRVMAXW:
    value->rValue = model->HSMHV2_rdrvmaxw;
    return(OK);
  case  HSMHV2_MOD_RDRVMAXWP:
    value->rValue = model->HSMHV2_rdrvmaxwp;
    return(OK);
  case  HSMHV2_MOD_RDRVMAXL:
    value->rValue = model->HSMHV2_rdrvmaxl;
    return(OK);
  case  HSMHV2_MOD_RDRVMAXLP:
    value->rValue = model->HSMHV2_rdrvmaxlp;
    return(OK);
  case  HSMHV2_MOD_RDRMUEL:
    value->rValue = model->HSMHV2_rdrmuel;
    return(OK);
  case  HSMHV2_MOD_RDRMUELP:
    value->rValue = model->HSMHV2_rdrmuelp;
    return(OK);
  case  HSMHV2_MOD_RDRQOVER:
    value->rValue = model->HSMHV2_rdrqover;
    return(OK);
  case HSMHV2_MOD_QOVADD:
    value->rValue = model->HSMHV2_qovadd;
    return(OK);
  case HSMHV2_MOD_JS0D:
    value->rValue = model->HSMHV2_js0d;
    return(OK);
  case HSMHV2_MOD_JS0SWD:
    value->rValue = model->HSMHV2_js0swd;
    return(OK);
  case HSMHV2_MOD_NJD:
    value->rValue = model->HSMHV2_njd;
    return(OK);
  case HSMHV2_MOD_NJSWD:
    value->rValue = model->HSMHV2_njswd;
    return(OK);
  case HSMHV2_MOD_XTID:
    value->rValue = model->HSMHV2_xtid;
    return(OK);
  case HSMHV2_MOD_CJD:
    value->rValue = model->HSMHV2_cjd;
    return(OK);
  case HSMHV2_MOD_CJSWD:
    value->rValue = model->HSMHV2_cjswd;
    return(OK);
  case HSMHV2_MOD_CJSWGD:
    value->rValue = model->HSMHV2_cjswgd;
    return(OK);
  case HSMHV2_MOD_MJD:
    value->rValue = model->HSMHV2_mjd;
    return(OK);
  case HSMHV2_MOD_MJSWD:
    value->rValue = model->HSMHV2_mjswd;
    return(OK);
  case HSMHV2_MOD_MJSWGD:
    value->rValue = model->HSMHV2_mjswgd;
    return(OK);
  case HSMHV2_MOD_PBD:
    value->rValue = model->HSMHV2_pbd;
    return(OK);
  case HSMHV2_MOD_PBSWD:
    value->rValue = model->HSMHV2_pbswd;
    return(OK);
  case HSMHV2_MOD_PBSWDG:
    value->rValue = model->HSMHV2_pbswgd;
    return(OK);
  case HSMHV2_MOD_XTI2D:
    value->rValue = model->HSMHV2_xti2d;
    return(OK);
  case HSMHV2_MOD_CISBD:
    value->rValue = model->HSMHV2_cisbd;
    return(OK);
  case HSMHV2_MOD_CVBD:
    value->rValue = model->HSMHV2_cvbd;
    return(OK);
  case HSMHV2_MOD_CTEMPD:
    value->rValue = model->HSMHV2_ctempd;
    return(OK);
  case HSMHV2_MOD_CISBKD:
    value->rValue = model->HSMHV2_cisbkd;
    return(OK);
  case HSMHV2_MOD_DIVXD:
    value->rValue = model->HSMHV2_divxd;
    return(OK);
  case HSMHV2_MOD_VDIFFJD:
    value->rValue = model->HSMHV2_vdiffjd;
    return(OK);
  case HSMHV2_MOD_JS0S:
    value->rValue = model->HSMHV2_js0s;
    return(OK);
  case HSMHV2_MOD_JS0SWS:
    value->rValue = model->HSMHV2_js0sws;
    return(OK);
  case HSMHV2_MOD_NJS:
    value->rValue = model->HSMHV2_njs;
    return(OK);
  case HSMHV2_MOD_NJSWS:
    value->rValue = model->HSMHV2_njsws;
    return(OK);
  case HSMHV2_MOD_XTIS:
    value->rValue = model->HSMHV2_xtis;
    return(OK);
  case HSMHV2_MOD_CJS:
    value->rValue = model->HSMHV2_cjs;
    return(OK);
  case HSMHV2_MOD_CJSSW:
    value->rValue = model->HSMHV2_cjsws;
    return(OK);
  case HSMHV2_MOD_CJSWGS:
    value->rValue = model->HSMHV2_cjswgs;
    return(OK);
  case HSMHV2_MOD_MJS:
    value->rValue = model->HSMHV2_mjs;
    return(OK);
  case HSMHV2_MOD_MJSWS:
    value->rValue = model->HSMHV2_mjsws;
    return(OK);
  case HSMHV2_MOD_MJSWGS:
    value->rValue = model->HSMHV2_mjswgs;
    return(OK);
  case HSMHV2_MOD_PBS:
    value->rValue = model->HSMHV2_pbs;
    return(OK);
  case HSMHV2_MOD_PBSWS:
    value->rValue = model->HSMHV2_pbsws;
    return(OK);
  case HSMHV2_MOD_PBSWSG:
    value->rValue = model->HSMHV2_pbswgs;
    return(OK);
  case HSMHV2_MOD_XTI2S:
    value->rValue = model->HSMHV2_xti2s;
    return(OK);
  case HSMHV2_MOD_CISBS:
    value->rValue = model->HSMHV2_cisbs;
    return(OK);
  case HSMHV2_MOD_CVBS:
    value->rValue = model->HSMHV2_cvbs;
    return(OK);
  case HSMHV2_MOD_CTEMPS:
    value->rValue = model->HSMHV2_ctemps;
    return(OK);
  case HSMHV2_MOD_CISBKS:
    value->rValue = model->HSMHV2_cisbks;
    return(OK);
  case HSMHV2_MOD_DIVXS:
    value->rValue = model->HSMHV2_divxs;
    return(OK);
  case HSMHV2_MOD_VDIFFJS:
    value->rValue = model->HSMHV2_vdiffjs;
    return(OK);
  case HSMHV2_MOD_SHEMAX:
    value->rValue = model->HSMHV2_shemax;
    return(OK);
  case HSMHV2_MOD_VGSMIN:
    value->rValue = model->HSMHV2_vgsmin;
    return(OK);
  case HSMHV2_MOD_GDSLEAK:
    value->rValue = model->HSMHV2_gdsleak;
    return(OK);
  case HSMHV2_MOD_RDRBB:
    value->rValue = model->HSMHV2_rdrbb;
    return(OK);
  case HSMHV2_MOD_RDRBBTMP:
    value->rValue = model->HSMHV2_rdrbbtmp;
    return(OK);

  /* binning parameters */
  case  HSMHV2_MOD_LMIN:
    value->rValue = model->HSMHV2_lmin;
    return(OK);
  case  HSMHV2_MOD_LMAX:
    value->rValue = model->HSMHV2_lmax;
    return(OK);
  case  HSMHV2_MOD_WMIN:
    value->rValue = model->HSMHV2_wmin;
    return(OK);
  case  HSMHV2_MOD_WMAX:
    value->rValue = model->HSMHV2_wmax;
    return(OK);
  case  HSMHV2_MOD_LBINN:
    value->rValue = model->HSMHV2_lbinn;
    return(OK);
  case  HSMHV2_MOD_WBINN:
    value->rValue = model->HSMHV2_wbinn;
    return(OK);

  /* Length dependence */
  case  HSMHV2_MOD_LVMAX:
    value->rValue = model->HSMHV2_lvmax;
    return(OK);
  case  HSMHV2_MOD_LBGTMP1:
    value->rValue = model->HSMHV2_lbgtmp1;
    return(OK);
  case  HSMHV2_MOD_LBGTMP2:
    value->rValue = model->HSMHV2_lbgtmp2;
    return(OK);
  case  HSMHV2_MOD_LEG0:
    value->rValue = model->HSMHV2_leg0;
    return(OK);
  case  HSMHV2_MOD_LVFBOVER:
    value->rValue = model->HSMHV2_lvfbover;
    return(OK);
  case  HSMHV2_MOD_LNOVER:
    value->rValue = model->HSMHV2_lnover;
    return(OK);
  case  HSMHV2_MOD_LNOVERS:
    value->rValue = model->HSMHV2_lnovers;
    return(OK);
  case  HSMHV2_MOD_LWL2:
    value->rValue = model->HSMHV2_lwl2;
    return(OK);
  case  HSMHV2_MOD_LVFBC:
    value->rValue = model->HSMHV2_lvfbc;
    return(OK);
  case  HSMHV2_MOD_LNSUBC:
    value->rValue = model->HSMHV2_lnsubc;
    return(OK);
  case  HSMHV2_MOD_LNSUBP:
    value->rValue = model->HSMHV2_lnsubp;
    return(OK);
  case  HSMHV2_MOD_LSCP1:
    value->rValue = model->HSMHV2_lscp1;
    return(OK);
  case  HSMHV2_MOD_LSCP2:
    value->rValue = model->HSMHV2_lscp2;
    return(OK);
  case  HSMHV2_MOD_LSCP3:
    value->rValue = model->HSMHV2_lscp3;
    return(OK);
  case  HSMHV2_MOD_LSC1:
    value->rValue = model->HSMHV2_lsc1;
    return(OK);
  case  HSMHV2_MOD_LSC2:
    value->rValue = model->HSMHV2_lsc2;
    return(OK);
  case  HSMHV2_MOD_LSC3:
    value->rValue = model->HSMHV2_lsc3;
    return(OK);
  case  HSMHV2_MOD_LPGD1:
    value->rValue = model->HSMHV2_lpgd1;
    return(OK);
  case  HSMHV2_MOD_LNDEP:
    value->rValue = model->HSMHV2_lndep;
    return(OK);
  case  HSMHV2_MOD_LNINV:
    value->rValue = model->HSMHV2_lninv;
    return(OK);
  case  HSMHV2_MOD_LMUECB0:
    value->rValue = model->HSMHV2_lmuecb0;
    return(OK);
  case  HSMHV2_MOD_LMUECB1:
    value->rValue = model->HSMHV2_lmuecb1;
    return(OK);
  case  HSMHV2_MOD_LMUEPH1:
    value->rValue = model->HSMHV2_lmueph1;
    return(OK);
  case  HSMHV2_MOD_LVTMP:
    value->rValue = model->HSMHV2_lvtmp;
    return(OK);
  case  HSMHV2_MOD_LWVTH0:
    value->rValue = model->HSMHV2_lwvth0;
    return(OK);
  case  HSMHV2_MOD_LMUESR1:
    value->rValue = model->HSMHV2_lmuesr1;
    return(OK);
  case  HSMHV2_MOD_LMUETMP:
    value->rValue = model->HSMHV2_lmuetmp;
    return(OK);
  case  HSMHV2_MOD_LSUB1:
    value->rValue = model->HSMHV2_lsub1;
    return(OK);
  case  HSMHV2_MOD_LSUB2:
    value->rValue = model->HSMHV2_lsub2;
    return(OK);
  case  HSMHV2_MOD_LSVDS:
    value->rValue = model->HSMHV2_lsvds;
    return(OK);
  case  HSMHV2_MOD_LSVBS:
    value->rValue = model->HSMHV2_lsvbs;
    return(OK);
  case  HSMHV2_MOD_LSVGS:
    value->rValue = model->HSMHV2_lsvgs;
    return(OK);
  case  HSMHV2_MOD_LFN1:
    value->rValue = model->HSMHV2_lfn1;
    return(OK);
  case  HSMHV2_MOD_LFN2:
    value->rValue = model->HSMHV2_lfn2;
    return(OK);
  case  HSMHV2_MOD_LFN3:
    value->rValue = model->HSMHV2_lfn3;
    return(OK);
  case  HSMHV2_MOD_LFVBS:
    value->rValue = model->HSMHV2_lfvbs;
    return(OK);
  case  HSMHV2_MOD_LNSTI:
    value->rValue = model->HSMHV2_lnsti;
    return(OK);
  case  HSMHV2_MOD_LWSTI:
    value->rValue = model->HSMHV2_lwsti;
    return(OK);
  case  HSMHV2_MOD_LSCSTI1:
    value->rValue = model->HSMHV2_lscsti1;
    return(OK);
  case  HSMHV2_MOD_LSCSTI2:
    value->rValue = model->HSMHV2_lscsti2;
    return(OK);
  case  HSMHV2_MOD_LVTHSTI:
    value->rValue = model->HSMHV2_lvthsti;
    return(OK);
  case  HSMHV2_MOD_LMUESTI1:
    value->rValue = model->HSMHV2_lmuesti1;
    return(OK);
  case  HSMHV2_MOD_LMUESTI2:
    value->rValue = model->HSMHV2_lmuesti2;
    return(OK);
  case  HSMHV2_MOD_LMUESTI3:
    value->rValue = model->HSMHV2_lmuesti3;
    return(OK);
  case  HSMHV2_MOD_LNSUBPSTI1:
    value->rValue = model->HSMHV2_lnsubpsti1;
    return(OK);
  case  HSMHV2_MOD_LNSUBPSTI2:
    value->rValue = model->HSMHV2_lnsubpsti2;
    return(OK);
  case  HSMHV2_MOD_LNSUBPSTI3:
    value->rValue = model->HSMHV2_lnsubpsti3;
    return(OK);
  case  HSMHV2_MOD_LCGSO:
    value->rValue = model->HSMHV2_lcgso;
    return(OK);
  case  HSMHV2_MOD_LCGDO:
    value->rValue = model->HSMHV2_lcgdo;
    return(OK);
  case  HSMHV2_MOD_LJS0:
    value->rValue = model->HSMHV2_ljs0;
    return(OK);
  case  HSMHV2_MOD_LJS0SW:
    value->rValue = model->HSMHV2_ljs0sw;
    return(OK);
  case  HSMHV2_MOD_LNJ:
    value->rValue = model->HSMHV2_lnj;
    return(OK);
  case  HSMHV2_MOD_LCISBK:
    value->rValue = model->HSMHV2_lcisbk;
    return(OK);
  case  HSMHV2_MOD_LCLM1:
    value->rValue = model->HSMHV2_lclm1;
    return(OK);
  case  HSMHV2_MOD_LCLM2:
    value->rValue = model->HSMHV2_lclm2;
    return(OK);
  case  HSMHV2_MOD_LCLM3:
    value->rValue = model->HSMHV2_lclm3;
    return(OK);
  case  HSMHV2_MOD_LWFC:
    value->rValue = model->HSMHV2_lwfc;
    return(OK);
  case  HSMHV2_MOD_LGIDL1:
    value->rValue = model->HSMHV2_lgidl1;
    return(OK);
  case  HSMHV2_MOD_LGIDL2:
    value->rValue = model->HSMHV2_lgidl2;
    return(OK);
  case  HSMHV2_MOD_LGLEAK1:
    value->rValue = model->HSMHV2_lgleak1;
    return(OK);
  case  HSMHV2_MOD_LGLEAK2:
    value->rValue = model->HSMHV2_lgleak2;
    return(OK);
  case  HSMHV2_MOD_LGLEAK3:
    value->rValue = model->HSMHV2_lgleak3;
    return(OK);
  case  HSMHV2_MOD_LGLEAK6:
    value->rValue = model->HSMHV2_lgleak6;
    return(OK);
  case  HSMHV2_MOD_LGLKSD1:
    value->rValue = model->HSMHV2_lglksd1;
    return(OK);
  case  HSMHV2_MOD_LGLKSD2:
    value->rValue = model->HSMHV2_lglksd2;
    return(OK);
  case  HSMHV2_MOD_LGLKB1:
    value->rValue = model->HSMHV2_lglkb1;
    return(OK);
  case  HSMHV2_MOD_LGLKB2:
    value->rValue = model->HSMHV2_lglkb2;
    return(OK);
  case  HSMHV2_MOD_LNFTRP:
    value->rValue = model->HSMHV2_lnftrp;
    return(OK);
  case  HSMHV2_MOD_LNFALP:
    value->rValue = model->HSMHV2_lnfalp;
    return(OK);
  case  HSMHV2_MOD_LVDIFFJ:
    value->rValue = model->HSMHV2_lvdiffj;
    return(OK);
  case  HSMHV2_MOD_LIBPC1:
    value->rValue = model->HSMHV2_libpc1;
    return(OK);
  case  HSMHV2_MOD_LIBPC2:
    value->rValue = model->HSMHV2_libpc2;
    return(OK);
  case  HSMHV2_MOD_LCGBO:
    value->rValue = model->HSMHV2_lcgbo;
    return(OK);
  case  HSMHV2_MOD_LCVDSOVER:
    value->rValue = model->HSMHV2_lcvdsover;
    return(OK);
  case  HSMHV2_MOD_LFALPH:
    value->rValue = model->HSMHV2_lfalph;
    return(OK);
  case  HSMHV2_MOD_LNPEXT:
    value->rValue = model->HSMHV2_lnpext;
    return(OK);
  case  HSMHV2_MOD_LPOWRAT:
    value->rValue = model->HSMHV2_lpowrat;
    return(OK);
  case  HSMHV2_MOD_LRD:
    value->rValue = model->HSMHV2_lrd;
    return(OK);
  case  HSMHV2_MOD_LRD22:
    value->rValue = model->HSMHV2_lrd22;
    return(OK);
  case  HSMHV2_MOD_LRD23:
    value->rValue = model->HSMHV2_lrd23;
    return(OK);
  case  HSMHV2_MOD_LRD24:
    value->rValue = model->HSMHV2_lrd24;
    return(OK);
  case  HSMHV2_MOD_LRDICT1:
    value->rValue = model->HSMHV2_lrdict1;
    return(OK);
  case  HSMHV2_MOD_LRDOV13:
    value->rValue = model->HSMHV2_lrdov13;
    return(OK);
  case  HSMHV2_MOD_LRDSLP1:
    value->rValue = model->HSMHV2_lrdslp1;
    return(OK);
  case  HSMHV2_MOD_LRDVB:
    value->rValue = model->HSMHV2_lrdvb;
    return(OK);
  case  HSMHV2_MOD_LRDVD:
    value->rValue = model->HSMHV2_lrdvd;
    return(OK);
  case  HSMHV2_MOD_LRDVG11:
    value->rValue = model->HSMHV2_lrdvg11;
    return(OK);
  case  HSMHV2_MOD_LRS:
    value->rValue = model->HSMHV2_lrs;
    return(OK);
  case  HSMHV2_MOD_LRTH0:
    value->rValue = model->HSMHV2_lrth0;
    return(OK);
  case  HSMHV2_MOD_LVOVER:
    value->rValue = model->HSMHV2_lvover;
    return(OK);
  case HSMHV2_MOD_LJS0D:
    value->rValue = model->HSMHV2_ljs0d;
    return(OK);
  case HSMHV2_MOD_LJS0SWD:
    value->rValue = model->HSMHV2_ljs0swd;
    return(OK);
  case HSMHV2_MOD_LNJD:
    value->rValue = model->HSMHV2_lnjd;
    return(OK);
  case HSMHV2_MOD_LCISBKD:
    value->rValue = model->HSMHV2_lcisbkd;
    return(OK);
  case HSMHV2_MOD_LVDIFFJD:
    value->rValue = model->HSMHV2_lvdiffjd;
    return(OK);
  case HSMHV2_MOD_LJS0S:
    value->rValue = model->HSMHV2_ljs0s;
    return(OK);
  case HSMHV2_MOD_LJS0SWS:
    value->rValue = model->HSMHV2_ljs0sws;
    return(OK);
  case HSMHV2_MOD_LNJS:
    value->rValue = model->HSMHV2_lnjs;
    return(OK);
  case HSMHV2_MOD_LCISBKS:
    value->rValue = model->HSMHV2_lcisbks;
    return(OK);
  case HSMHV2_MOD_LVDIFFJS:
    value->rValue = model->HSMHV2_lvdiffjs;
    return(OK);

  /* Width dependence */
  case  HSMHV2_MOD_WVMAX:
    value->rValue = model->HSMHV2_wvmax;
    return(OK);
  case  HSMHV2_MOD_WBGTMP1:
    value->rValue = model->HSMHV2_wbgtmp1;
    return(OK);
  case  HSMHV2_MOD_WBGTMP2:
    value->rValue = model->HSMHV2_wbgtmp2;
    return(OK);
  case  HSMHV2_MOD_WEG0:
    value->rValue = model->HSMHV2_weg0;
    return(OK);
  case  HSMHV2_MOD_WVFBOVER:
    value->rValue = model->HSMHV2_wvfbover;
    return(OK);
  case  HSMHV2_MOD_WNOVER:
    value->rValue = model->HSMHV2_wnover;
    return(OK);
  case  HSMHV2_MOD_WNOVERS:
    value->rValue = model->HSMHV2_wnovers;
    return(OK);
  case  HSMHV2_MOD_WWL2:
    value->rValue = model->HSMHV2_wwl2;
    return(OK);
  case  HSMHV2_MOD_WVFBC:
    value->rValue = model->HSMHV2_wvfbc;
    return(OK);
  case  HSMHV2_MOD_WNSUBC:
    value->rValue = model->HSMHV2_wnsubc;
    return(OK);
  case  HSMHV2_MOD_WNSUBP:
    value->rValue = model->HSMHV2_wnsubp;
    return(OK);
  case  HSMHV2_MOD_WSCP1:
    value->rValue = model->HSMHV2_wscp1;
    return(OK);
  case  HSMHV2_MOD_WSCP2:
    value->rValue = model->HSMHV2_wscp2;
    return(OK);
  case  HSMHV2_MOD_WSCP3:
    value->rValue = model->HSMHV2_wscp3;
    return(OK);
  case  HSMHV2_MOD_WSC1:
    value->rValue = model->HSMHV2_wsc1;
    return(OK);
  case  HSMHV2_MOD_WSC2:
    value->rValue = model->HSMHV2_wsc2;
    return(OK);
  case  HSMHV2_MOD_WSC3:
    value->rValue = model->HSMHV2_wsc3;
    return(OK);
  case  HSMHV2_MOD_WPGD1:
    value->rValue = model->HSMHV2_wpgd1;
    return(OK);
  case  HSMHV2_MOD_WNDEP:
    value->rValue = model->HSMHV2_wndep;
    return(OK);
  case  HSMHV2_MOD_WNINV:
    value->rValue = model->HSMHV2_wninv;
    return(OK);
  case  HSMHV2_MOD_WMUECB0:
    value->rValue = model->HSMHV2_wmuecb0;
    return(OK);
  case  HSMHV2_MOD_WMUECB1:
    value->rValue = model->HSMHV2_wmuecb1;
    return(OK);
  case  HSMHV2_MOD_WMUEPH1:
    value->rValue = model->HSMHV2_wmueph1;
    return(OK);
  case  HSMHV2_MOD_WVTMP:
    value->rValue = model->HSMHV2_wvtmp;
    return(OK);
  case  HSMHV2_MOD_WWVTH0:
    value->rValue = model->HSMHV2_wwvth0;
    return(OK);
  case  HSMHV2_MOD_WMUESR1:
    value->rValue = model->HSMHV2_wmuesr1;
    return(OK);
  case  HSMHV2_MOD_WMUETMP:
    value->rValue = model->HSMHV2_wmuetmp;
    return(OK);
  case  HSMHV2_MOD_WSUB1:
    value->rValue = model->HSMHV2_wsub1;
    return(OK);
  case  HSMHV2_MOD_WSUB2:
    value->rValue = model->HSMHV2_wsub2;
    return(OK);
  case  HSMHV2_MOD_WSVDS:
    value->rValue = model->HSMHV2_wsvds;
    return(OK);
  case  HSMHV2_MOD_WSVBS:
    value->rValue = model->HSMHV2_wsvbs;
    return(OK);
  case  HSMHV2_MOD_WSVGS:
    value->rValue = model->HSMHV2_wsvgs;
    return(OK);
  case  HSMHV2_MOD_WFN1:
    value->rValue = model->HSMHV2_wfn1;
    return(OK);
  case  HSMHV2_MOD_WFN2:
    value->rValue = model->HSMHV2_wfn2;
    return(OK);
  case  HSMHV2_MOD_WFN3:
    value->rValue = model->HSMHV2_wfn3;
    return(OK);
  case  HSMHV2_MOD_WFVBS:
    value->rValue = model->HSMHV2_wfvbs;
    return(OK);
  case  HSMHV2_MOD_WNSTI:
    value->rValue = model->HSMHV2_wnsti;
    return(OK);
  case  HSMHV2_MOD_WWSTI:
    value->rValue = model->HSMHV2_wwsti;
    return(OK);
  case  HSMHV2_MOD_WSCSTI1:
    value->rValue = model->HSMHV2_wscsti1;
    return(OK);
  case  HSMHV2_MOD_WSCSTI2:
    value->rValue = model->HSMHV2_wscsti2;
    return(OK);
  case  HSMHV2_MOD_WVTHSTI:
    value->rValue = model->HSMHV2_wvthsti;
    return(OK);
  case  HSMHV2_MOD_WMUESTI1:
    value->rValue = model->HSMHV2_wmuesti1;
    return(OK);
  case  HSMHV2_MOD_WMUESTI2:
    value->rValue = model->HSMHV2_wmuesti2;
    return(OK);
  case  HSMHV2_MOD_WMUESTI3:
    value->rValue = model->HSMHV2_wmuesti3;
    return(OK);
  case  HSMHV2_MOD_WNSUBPSTI1:
    value->rValue = model->HSMHV2_wnsubpsti1;
    return(OK);
  case  HSMHV2_MOD_WNSUBPSTI2:
    value->rValue = model->HSMHV2_wnsubpsti2;
    return(OK);
  case  HSMHV2_MOD_WNSUBPSTI3:
    value->rValue = model->HSMHV2_wnsubpsti3;
    return(OK);
  case  HSMHV2_MOD_WCGSO:
    value->rValue = model->HSMHV2_wcgso;
    return(OK);
  case  HSMHV2_MOD_WCGDO:
    value->rValue = model->HSMHV2_wcgdo;
    return(OK);
  case  HSMHV2_MOD_WJS0:
    value->rValue = model->HSMHV2_wjs0;
    return(OK);
  case  HSMHV2_MOD_WJS0SW:
    value->rValue = model->HSMHV2_wjs0sw;
    return(OK);
  case  HSMHV2_MOD_WNJ:
    value->rValue = model->HSMHV2_wnj;
    return(OK);
  case  HSMHV2_MOD_WCISBK:
    value->rValue = model->HSMHV2_wcisbk;
    return(OK);
  case  HSMHV2_MOD_WCLM1:
    value->rValue = model->HSMHV2_wclm1;
    return(OK);
  case  HSMHV2_MOD_WCLM2:
    value->rValue = model->HSMHV2_wclm2;
    return(OK);
  case  HSMHV2_MOD_WCLM3:
    value->rValue = model->HSMHV2_wclm3;
    return(OK);
  case  HSMHV2_MOD_WWFC:
    value->rValue = model->HSMHV2_wwfc;
    return(OK);
  case  HSMHV2_MOD_WGIDL1:
    value->rValue = model->HSMHV2_wgidl1;
    return(OK);
  case  HSMHV2_MOD_WGIDL2:
    value->rValue = model->HSMHV2_wgidl2;
    return(OK);
  case  HSMHV2_MOD_WGLEAK1:
    value->rValue = model->HSMHV2_wgleak1;
    return(OK);
  case  HSMHV2_MOD_WGLEAK2:
    value->rValue = model->HSMHV2_wgleak2;
    return(OK);
  case  HSMHV2_MOD_WGLEAK3:
    value->rValue = model->HSMHV2_wgleak3;
    return(OK);
  case  HSMHV2_MOD_WGLEAK6:
    value->rValue = model->HSMHV2_wgleak6;
    return(OK);
  case  HSMHV2_MOD_WGLKSD1:
    value->rValue = model->HSMHV2_wglksd1;
    return(OK);
  case  HSMHV2_MOD_WGLKSD2:
    value->rValue = model->HSMHV2_wglksd2;
    return(OK);
  case  HSMHV2_MOD_WGLKB1:
    value->rValue = model->HSMHV2_wglkb1;
    return(OK);
  case  HSMHV2_MOD_WGLKB2:
    value->rValue = model->HSMHV2_wglkb2;
    return(OK);
  case  HSMHV2_MOD_WNFTRP:
    value->rValue = model->HSMHV2_wnftrp;
    return(OK);
  case  HSMHV2_MOD_WNFALP:
    value->rValue = model->HSMHV2_wnfalp;
    return(OK);
  case  HSMHV2_MOD_WVDIFFJ:
    value->rValue = model->HSMHV2_wvdiffj;
    return(OK);
  case  HSMHV2_MOD_WIBPC1:
    value->rValue = model->HSMHV2_wibpc1;
    return(OK);
  case  HSMHV2_MOD_WIBPC2:
    value->rValue = model->HSMHV2_wibpc2;
    return(OK);
  case  HSMHV2_MOD_WCGBO:
    value->rValue = model->HSMHV2_wcgbo;
    return(OK);
  case  HSMHV2_MOD_WCVDSOVER:
    value->rValue = model->HSMHV2_wcvdsover;
    return(OK);
  case  HSMHV2_MOD_WFALPH:
    value->rValue = model->HSMHV2_wfalph;
    return(OK);
  case  HSMHV2_MOD_WNPEXT:
    value->rValue = model->HSMHV2_wnpext;
    return(OK);
  case  HSMHV2_MOD_WPOWRAT:
    value->rValue = model->HSMHV2_wpowrat;
    return(OK);
  case  HSMHV2_MOD_WRD:
    value->rValue = model->HSMHV2_wrd;
    return(OK);
  case  HSMHV2_MOD_WRD22:
    value->rValue = model->HSMHV2_wrd22;
    return(OK);
  case  HSMHV2_MOD_WRD23:
    value->rValue = model->HSMHV2_wrd23;
    return(OK);
  case  HSMHV2_MOD_WRD24:
    value->rValue = model->HSMHV2_wrd24;
    return(OK);
  case  HSMHV2_MOD_WRDICT1:
    value->rValue = model->HSMHV2_wrdict1;
    return(OK);
  case  HSMHV2_MOD_WRDOV13:
    value->rValue = model->HSMHV2_wrdov13;
    return(OK);
  case  HSMHV2_MOD_WRDSLP1:
    value->rValue = model->HSMHV2_wrdslp1;
    return(OK);
  case  HSMHV2_MOD_WRDVB:
    value->rValue = model->HSMHV2_wrdvb;
    return(OK);
  case  HSMHV2_MOD_WRDVD:
    value->rValue = model->HSMHV2_wrdvd;
    return(OK);
  case  HSMHV2_MOD_WRDVG11:
    value->rValue = model->HSMHV2_wrdvg11;
    return(OK);
  case  HSMHV2_MOD_WRS:
    value->rValue = model->HSMHV2_wrs;
    return(OK);
  case  HSMHV2_MOD_WRTH0:
    value->rValue = model->HSMHV2_wrth0;
    return(OK);
  case  HSMHV2_MOD_WVOVER:
    value->rValue = model->HSMHV2_wvover;
    return(OK);
  case HSMHV2_MOD_WJS0D:
    value->rValue = model->HSMHV2_wjs0d;
    return(OK);
  case HSMHV2_MOD_WJS0SWD:
    value->rValue = model->HSMHV2_wjs0swd;
    return(OK);
  case HSMHV2_MOD_WNJD:
    value->rValue = model->HSMHV2_wnjd;
    return(OK);
  case HSMHV2_MOD_WCISBKD:
    value->rValue = model->HSMHV2_wcisbkd;
    return(OK);
  case HSMHV2_MOD_WVDIFFJD:
    value->rValue = model->HSMHV2_wvdiffjd;
    return(OK);
  case HSMHV2_MOD_WJS0S:
    value->rValue = model->HSMHV2_wjs0s;
    return(OK);
  case HSMHV2_MOD_WJS0SWS:
    value->rValue = model->HSMHV2_wjs0sws;
    return(OK);
  case HSMHV2_MOD_WNJS:
    value->rValue = model->HSMHV2_wnjs;
    return(OK);
  case HSMHV2_MOD_WCISBKS:
    value->rValue = model->HSMHV2_wcisbks;
    return(OK);
  case HSMHV2_MOD_WVDIFFJS:
    value->rValue = model->HSMHV2_wvdiffjs;
    return(OK);

  /* Cross-term dependence */
  case  HSMHV2_MOD_PVMAX:
    value->rValue = model->HSMHV2_pvmax;
    return(OK);
  case  HSMHV2_MOD_PBGTMP1:
    value->rValue = model->HSMHV2_pbgtmp1;
    return(OK);
  case  HSMHV2_MOD_PBGTMP2:
    value->rValue = model->HSMHV2_pbgtmp2;
    return(OK);
  case  HSMHV2_MOD_PEG0:
    value->rValue = model->HSMHV2_peg0;
    return(OK);
  case  HSMHV2_MOD_PVFBOVER:
    value->rValue = model->HSMHV2_pvfbover;
    return(OK);
  case  HSMHV2_MOD_PNOVER:
    value->rValue = model->HSMHV2_pnover;
    return(OK);
  case  HSMHV2_MOD_PNOVERS:
    value->rValue = model->HSMHV2_pnovers;
    return(OK);
  case  HSMHV2_MOD_PWL2:
    value->rValue = model->HSMHV2_pwl2;
    return(OK);
  case  HSMHV2_MOD_PVFBC:
    value->rValue = model->HSMHV2_pvfbc;
    return(OK);
  case  HSMHV2_MOD_PNSUBC:
    value->rValue = model->HSMHV2_pnsubc;
    return(OK);
  case  HSMHV2_MOD_PNSUBP:
    value->rValue = model->HSMHV2_pnsubp;
    return(OK);
  case  HSMHV2_MOD_PSCP1:
    value->rValue = model->HSMHV2_pscp1;
    return(OK);
  case  HSMHV2_MOD_PSCP2:
    value->rValue = model->HSMHV2_pscp2;
    return(OK);
  case  HSMHV2_MOD_PSCP3:
    value->rValue = model->HSMHV2_pscp3;
    return(OK);
  case  HSMHV2_MOD_PSC1:
    value->rValue = model->HSMHV2_psc1;
    return(OK);
  case  HSMHV2_MOD_PSC2:
    value->rValue = model->HSMHV2_psc2;
    return(OK);
  case  HSMHV2_MOD_PSC3:
    value->rValue = model->HSMHV2_psc3;
    return(OK);
  case  HSMHV2_MOD_PPGD1:
    value->rValue = model->HSMHV2_ppgd1;
    return(OK);
  case  HSMHV2_MOD_PNDEP:
    value->rValue = model->HSMHV2_pndep;
    return(OK);
  case  HSMHV2_MOD_PNINV:
    value->rValue = model->HSMHV2_pninv;
    return(OK);
  case  HSMHV2_MOD_PMUECB0:
    value->rValue = model->HSMHV2_pmuecb0;
    return(OK);
  case  HSMHV2_MOD_PMUECB1:
    value->rValue = model->HSMHV2_pmuecb1;
    return(OK);
  case  HSMHV2_MOD_PMUEPH1:
    value->rValue = model->HSMHV2_pmueph1;
    return(OK);
  case  HSMHV2_MOD_PVTMP:
    value->rValue = model->HSMHV2_pvtmp;
    return(OK);
  case  HSMHV2_MOD_PWVTH0:
    value->rValue = model->HSMHV2_pwvth0;
    return(OK);
  case  HSMHV2_MOD_PMUESR1:
    value->rValue = model->HSMHV2_pmuesr1;
    return(OK);
  case  HSMHV2_MOD_PMUETMP:
    value->rValue = model->HSMHV2_pmuetmp;
    return(OK);
  case  HSMHV2_MOD_PSUB1:
    value->rValue = model->HSMHV2_psub1;
    return(OK);
  case  HSMHV2_MOD_PSUB2:
    value->rValue = model->HSMHV2_psub2;
    return(OK);
  case  HSMHV2_MOD_PSVDS:
    value->rValue = model->HSMHV2_psvds;
    return(OK);
  case  HSMHV2_MOD_PSVBS:
    value->rValue = model->HSMHV2_psvbs;
    return(OK);
  case  HSMHV2_MOD_PSVGS:
    value->rValue = model->HSMHV2_psvgs;
    return(OK);
  case  HSMHV2_MOD_PFN1:
    value->rValue = model->HSMHV2_pfn1;
    return(OK);
  case  HSMHV2_MOD_PFN2:
    value->rValue = model->HSMHV2_pfn2;
    return(OK);
  case  HSMHV2_MOD_PFN3:
    value->rValue = model->HSMHV2_pfn3;
    return(OK);
  case  HSMHV2_MOD_PFVBS:
    value->rValue = model->HSMHV2_pfvbs;
    return(OK);
  case  HSMHV2_MOD_PNSTI:
    value->rValue = model->HSMHV2_pnsti;
    return(OK);
  case  HSMHV2_MOD_PWSTI:
    value->rValue = model->HSMHV2_pwsti;
    return(OK);
  case  HSMHV2_MOD_PSCSTI1:
    value->rValue = model->HSMHV2_pscsti1;
    return(OK);
  case  HSMHV2_MOD_PSCSTI2:
    value->rValue = model->HSMHV2_pscsti2;
    return(OK);
  case  HSMHV2_MOD_PVTHSTI:
    value->rValue = model->HSMHV2_pvthsti;
    return(OK);
  case  HSMHV2_MOD_PMUESTI1:
    value->rValue = model->HSMHV2_pmuesti1;
    return(OK);
  case  HSMHV2_MOD_PMUESTI2:
    value->rValue = model->HSMHV2_pmuesti2;
    return(OK);
  case  HSMHV2_MOD_PMUESTI3:
    value->rValue = model->HSMHV2_pmuesti3;
    return(OK);
  case  HSMHV2_MOD_PNSUBPSTI1:
    value->rValue = model->HSMHV2_pnsubpsti1;
    return(OK);
  case  HSMHV2_MOD_PNSUBPSTI2:
    value->rValue = model->HSMHV2_pnsubpsti2;
    return(OK);
  case  HSMHV2_MOD_PNSUBPSTI3:
    value->rValue = model->HSMHV2_pnsubpsti3;
    return(OK);
  case  HSMHV2_MOD_PCGSO:
    value->rValue = model->HSMHV2_pcgso;
    return(OK);
  case  HSMHV2_MOD_PCGDO:
    value->rValue = model->HSMHV2_pcgdo;
    return(OK);
  case  HSMHV2_MOD_PJS0:
    value->rValue = model->HSMHV2_pjs0;
    return(OK);
  case  HSMHV2_MOD_PJS0SW:
    value->rValue = model->HSMHV2_pjs0sw;
    return(OK);
  case  HSMHV2_MOD_PNJ:
    value->rValue = model->HSMHV2_pnj;
    return(OK);
  case  HSMHV2_MOD_PCISBK:
    value->rValue = model->HSMHV2_pcisbk;
    return(OK);
  case  HSMHV2_MOD_PCLM1:
    value->rValue = model->HSMHV2_pclm1;
    return(OK);
  case  HSMHV2_MOD_PCLM2:
    value->rValue = model->HSMHV2_pclm2;
    return(OK);
  case  HSMHV2_MOD_PCLM3:
    value->rValue = model->HSMHV2_pclm3;
    return(OK);
  case  HSMHV2_MOD_PWFC:
    value->rValue = model->HSMHV2_pwfc;
    return(OK);
  case  HSMHV2_MOD_PGIDL1:
    value->rValue = model->HSMHV2_pgidl1;
    return(OK);
  case  HSMHV2_MOD_PGIDL2:
    value->rValue = model->HSMHV2_pgidl2;
    return(OK);
  case  HSMHV2_MOD_PGLEAK1:
    value->rValue = model->HSMHV2_pgleak1;
    return(OK);
  case  HSMHV2_MOD_PGLEAK2:
    value->rValue = model->HSMHV2_pgleak2;
    return(OK);
  case  HSMHV2_MOD_PGLEAK3:
    value->rValue = model->HSMHV2_pgleak3;
    return(OK);
  case  HSMHV2_MOD_PGLEAK6:
    value->rValue = model->HSMHV2_pgleak6;
    return(OK);
  case  HSMHV2_MOD_PGLKSD1:
    value->rValue = model->HSMHV2_pglksd1;
    return(OK);
  case  HSMHV2_MOD_PGLKSD2:
    value->rValue = model->HSMHV2_pglksd2;
    return(OK);
  case  HSMHV2_MOD_PGLKB1:
    value->rValue = model->HSMHV2_pglkb1;
    return(OK);
  case  HSMHV2_MOD_PGLKB2:
    value->rValue = model->HSMHV2_pglkb2;
    return(OK);
  case  HSMHV2_MOD_PNFTRP:
    value->rValue = model->HSMHV2_pnftrp;
    return(OK);
  case  HSMHV2_MOD_PNFALP:
    value->rValue = model->HSMHV2_pnfalp;
    return(OK);
  case  HSMHV2_MOD_PVDIFFJ:
    value->rValue = model->HSMHV2_pvdiffj;
    return(OK);
  case  HSMHV2_MOD_PIBPC1:
    value->rValue = model->HSMHV2_pibpc1;
    return(OK);
  case  HSMHV2_MOD_PIBPC2:
    value->rValue = model->HSMHV2_pibpc2;
    return(OK);
  case  HSMHV2_MOD_PCGBO:
    value->rValue = model->HSMHV2_pcgbo;
    return(OK);
  case  HSMHV2_MOD_PCVDSOVER:
    value->rValue = model->HSMHV2_pcvdsover;
    return(OK);
  case  HSMHV2_MOD_PFALPH:
    value->rValue = model->HSMHV2_pfalph;
    return(OK);
  case  HSMHV2_MOD_PNPEXT:
    value->rValue = model->HSMHV2_pnpext;
    return(OK);
  case  HSMHV2_MOD_PPOWRAT:
    value->rValue = model->HSMHV2_ppowrat;
    return(OK);
  case  HSMHV2_MOD_PRD:
    value->rValue = model->HSMHV2_prd;
    return(OK);
  case  HSMHV2_MOD_PRD22:
    value->rValue = model->HSMHV2_prd22;
    return(OK);
  case  HSMHV2_MOD_PRD23:
    value->rValue = model->HSMHV2_prd23;
    return(OK);
  case  HSMHV2_MOD_PRD24:
    value->rValue = model->HSMHV2_prd24;
    return(OK);
  case  HSMHV2_MOD_PRDICT1:
    value->rValue = model->HSMHV2_prdict1;
    return(OK);
  case  HSMHV2_MOD_PRDOV13:
    value->rValue = model->HSMHV2_prdov13;
    return(OK);
  case  HSMHV2_MOD_PRDSLP1:
    value->rValue = model->HSMHV2_prdslp1;
    return(OK);
  case  HSMHV2_MOD_PRDVB:
    value->rValue = model->HSMHV2_prdvb;
    return(OK);
  case  HSMHV2_MOD_PRDVD:
    value->rValue = model->HSMHV2_prdvd;
    return(OK);
  case  HSMHV2_MOD_PRDVG11:
    value->rValue = model->HSMHV2_prdvg11;
    return(OK);
  case  HSMHV2_MOD_PRS:
    value->rValue = model->HSMHV2_prs;
    return(OK);
  case  HSMHV2_MOD_PRTH0:
    value->rValue = model->HSMHV2_prth0;
    return(OK);
  case  HSMHV2_MOD_PVOVER:
    value->rValue = model->HSMHV2_pvover;
    return(OK);
  case HSMHV2_MOD_PJS0D:
    value->rValue = model->HSMHV2_pjs0d;
    return(OK);
  case HSMHV2_MOD_PJS0SWD:
    value->rValue = model->HSMHV2_pjs0swd;
    return(OK);
  case HSMHV2_MOD_PNJD:
    value->rValue = model->HSMHV2_pnjd;
    return(OK);
  case HSMHV2_MOD_PCISBKD:
    value->rValue = model->HSMHV2_pcisbkd;
    return(OK);
  case HSMHV2_MOD_PVDIFFJD:
    value->rValue = model->HSMHV2_pvdiffjd;
    return(OK);
  case HSMHV2_MOD_PJS0S:
    value->rValue = model->HSMHV2_pjs0s;
    return(OK);
  case HSMHV2_MOD_PJS0SWS:
    value->rValue = model->HSMHV2_pjs0sws;
    return(OK);
  case HSMHV2_MOD_PNJS:
    value->rValue = model->HSMHV2_pnjs;
    return(OK);
  case HSMHV2_MOD_PCISBKS:
    value->rValue = model->HSMHV2_pcisbks;
    return(OK);
  case HSMHV2_MOD_PVDIFFJS:
    value->rValue = model->HSMHV2_pvdiffjs;
    return(OK);

  case HSMHV2_MOD_VGS_MAX:
    value->rValue = model->HSMHV2vgsMax;
    return(OK);
  case HSMHV2_MOD_VGD_MAX:
    value->rValue = model->HSMHV2vgdMax;
    return(OK);
  case HSMHV2_MOD_VGB_MAX:
    value->rValue = model->HSMHV2vgbMax;
    return(OK);
  case HSMHV2_MOD_VDS_MAX:
    value->rValue = model->HSMHV2vdsMax;
    return(OK);
  case HSMHV2_MOD_VBS_MAX:
    value->rValue = model->HSMHV2vbsMax;
    return(OK);
  case HSMHV2_MOD_VBD_MAX:
    value->rValue = model->HSMHV2vbdMax;
    return(OK);
  case HSMHV2_MOD_VGSR_MAX:
      value->rValue = model->HSMHV2vgsrMax;
      return(OK);
  case HSMHV2_MOD_VGDR_MAX:
      value->rValue = model->HSMHV2vgdrMax;
      return(OK);
  case HSMHV2_MOD_VGBR_MAX:
      value->rValue = model->HSMHV2vgbrMax;
      return(OK);
  case HSMHV2_MOD_VBSR_MAX:
      value->rValue = model->HSMHV2vbsrMax;
      return(OK);
  case HSMHV2_MOD_VBDR_MAX:
      value->rValue = model->HSMHV2vbdrMax;
      return(OK);

  default:
    return(E_BADPARM);
  }
  /* NOTREACHED */
}
