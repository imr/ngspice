/**** BSIM3v3.2.4, Released by Xuemei Xi 12/21/2001 ****/

/**********
 * Copyright 2001 Regents of the University of California. All rights reserved.
 * File: b3mask.c of BSIM3v3.2.4
 * Author: 1995 Min-Chie Jeng and Mansun Chan.
 * Author: 1997-1999 Weidong Liu.
 * Author: 2001  Xuemei Xi
 * Modified by Paolo Nenzi 2002 and Dietmar Warning 2003
 **********/

#include "ngspice/ngspice.h"
#include "ngspice/ifsim.h"
#include "ngspice/cktdefs.h"
#include "ngspice/devdefs.h"
#include "bsim3v32def.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

int
BSIM3v32mAsk (CKTcircuit *ckt, GENmodel *inst, int which, IFvalue *value)
{
    BSIM3v32model *model = (BSIM3v32model *)inst;

    NG_IGNORE(ckt);

    switch(which)
    {   case BSIM3v32_MOD_MOBMOD:
            value->iValue = model->BSIM3v32mobMod;
            return(OK);
        case BSIM3v32_MOD_PARAMCHK:
            value->iValue = model->BSIM3v32paramChk;
            return(OK);
        case BSIM3v32_MOD_BINUNIT:
            value->iValue = model->BSIM3v32binUnit;
            return(OK);
        case BSIM3v32_MOD_CAPMOD:
            value->iValue = model->BSIM3v32capMod;
            return(OK);
        case BSIM3v32_MOD_NOIMOD:
            value->iValue = model->BSIM3v32noiMod;
            return(OK);
        case BSIM3v32_MOD_NQSMOD:
            value->iValue = model->BSIM3v32nqsMod;
            return(OK);
        case BSIM3v32_MOD_ACMMOD:
            value->iValue = model->BSIM3v32acmMod;
            return(OK);
        case BSIM3v32_MOD_CALCACM:
            value->iValue = model->BSIM3v32calcacm;
            return(OK);
        case  BSIM3v32_MOD_VERSION :
          value->sValue = model->BSIM3v32version;
            return(OK);
        case  BSIM3v32_MOD_TOX :
          value->rValue = model->BSIM3v32tox;
            return(OK);
        case  BSIM3v32_MOD_TOXM :
          value->rValue = model->BSIM3v32toxm;
            return(OK);
        case  BSIM3v32_MOD_CDSC :
          value->rValue = model->BSIM3v32cdsc;
            return(OK);
        case  BSIM3v32_MOD_CDSCB :
          value->rValue = model->BSIM3v32cdscb;
            return(OK);

        case  BSIM3v32_MOD_CDSCD :
          value->rValue = model->BSIM3v32cdscd;
            return(OK);

        case  BSIM3v32_MOD_CIT :
          value->rValue = model->BSIM3v32cit;
            return(OK);
        case  BSIM3v32_MOD_NFACTOR :
          value->rValue = model->BSIM3v32nfactor;
            return(OK);
        case BSIM3v32_MOD_XJ:
            value->rValue = model->BSIM3v32xj;
            return(OK);
        case BSIM3v32_MOD_VSAT:
            value->rValue = model->BSIM3v32vsat;
            return(OK);
        case BSIM3v32_MOD_AT:
            value->rValue = model->BSIM3v32at;
            return(OK);
        case BSIM3v32_MOD_A0:
            value->rValue = model->BSIM3v32a0;
            return(OK);

        case BSIM3v32_MOD_AGS:
            value->rValue = model->BSIM3v32ags;
            return(OK);

        case BSIM3v32_MOD_A1:
            value->rValue = model->BSIM3v32a1;
            return(OK);
        case BSIM3v32_MOD_A2:
            value->rValue = model->BSIM3v32a2;
            return(OK);
        case BSIM3v32_MOD_KETA:
            value->rValue = model->BSIM3v32keta;
            return(OK);
        case BSIM3v32_MOD_NSUB:
            value->rValue = model->BSIM3v32nsub;
            return(OK);
        case BSIM3v32_MOD_NPEAK:
            value->rValue = model->BSIM3v32npeak;
            return(OK);
        case BSIM3v32_MOD_NGATE:
            value->rValue = model->BSIM3v32ngate;
            return(OK);
        case BSIM3v32_MOD_GAMMA1:
            value->rValue = model->BSIM3v32gamma1;
            return(OK);
        case BSIM3v32_MOD_GAMMA2:
            value->rValue = model->BSIM3v32gamma2;
            return(OK);
        case BSIM3v32_MOD_VBX:
            value->rValue = model->BSIM3v32vbx;
            return(OK);
        case BSIM3v32_MOD_VBM:
            value->rValue = model->BSIM3v32vbm;
            return(OK);
        case BSIM3v32_MOD_XT:
            value->rValue = model->BSIM3v32xt;
            return(OK);
        case  BSIM3v32_MOD_K1:
          value->rValue = model->BSIM3v32k1;
            return(OK);
        case  BSIM3v32_MOD_KT1:
          value->rValue = model->BSIM3v32kt1;
            return(OK);
        case  BSIM3v32_MOD_KT1L:
          value->rValue = model->BSIM3v32kt1l;
            return(OK);
        case  BSIM3v32_MOD_KT2 :
          value->rValue = model->BSIM3v32kt2;
            return(OK);
        case  BSIM3v32_MOD_K2 :
          value->rValue = model->BSIM3v32k2;
            return(OK);
        case  BSIM3v32_MOD_K3:
          value->rValue = model->BSIM3v32k3;
            return(OK);
        case  BSIM3v32_MOD_K3B:
          value->rValue = model->BSIM3v32k3b;
            return(OK);
        case  BSIM3v32_MOD_W0:
          value->rValue = model->BSIM3v32w0;
            return(OK);
        case  BSIM3v32_MOD_NLX:
          value->rValue = model->BSIM3v32nlx;
            return(OK);
        case  BSIM3v32_MOD_DVT0 :
          value->rValue = model->BSIM3v32dvt0;
            return(OK);
        case  BSIM3v32_MOD_DVT1 :
          value->rValue = model->BSIM3v32dvt1;
            return(OK);
        case  BSIM3v32_MOD_DVT2 :
          value->rValue = model->BSIM3v32dvt2;
            return(OK);
        case  BSIM3v32_MOD_DVT0W :
          value->rValue = model->BSIM3v32dvt0w;
            return(OK);
        case  BSIM3v32_MOD_DVT1W :
          value->rValue = model->BSIM3v32dvt1w;
            return(OK);
        case  BSIM3v32_MOD_DVT2W :
          value->rValue = model->BSIM3v32dvt2w;
            return(OK);
        case  BSIM3v32_MOD_DROUT :
          value->rValue = model->BSIM3v32drout;
            return(OK);
        case  BSIM3v32_MOD_DSUB :
          value->rValue = model->BSIM3v32dsub;
            return(OK);
        case BSIM3v32_MOD_VTH0:
            value->rValue = model->BSIM3v32vth0;
            return(OK);
        case BSIM3v32_MOD_UA:
            value->rValue = model->BSIM3v32ua;
            return(OK);
        case BSIM3v32_MOD_UA1:
            value->rValue = model->BSIM3v32ua1;
            return(OK);
        case BSIM3v32_MOD_UB:
            value->rValue = model->BSIM3v32ub;
            return(OK);
        case BSIM3v32_MOD_UB1:
            value->rValue = model->BSIM3v32ub1;
            return(OK);
        case BSIM3v32_MOD_UC:
            value->rValue = model->BSIM3v32uc;
            return(OK);
        case BSIM3v32_MOD_UC1:
            value->rValue = model->BSIM3v32uc1;
            return(OK);
        case BSIM3v32_MOD_U0:
            value->rValue = model->BSIM3v32u0;
            return(OK);
        case BSIM3v32_MOD_UTE:
            value->rValue = model->BSIM3v32ute;
            return(OK);
        case BSIM3v32_MOD_VOFF:
            value->rValue = model->BSIM3v32voff;
            return(OK);
        case BSIM3v32_MOD_DELTA:
            value->rValue = model->BSIM3v32delta;
            return(OK);
        case BSIM3v32_MOD_RDSW:
            value->rValue = model->BSIM3v32rdsw;
            return(OK);
        case BSIM3v32_MOD_PRWG:
            value->rValue = model->BSIM3v32prwg;
            return(OK);
        case BSIM3v32_MOD_PRWB:
            value->rValue = model->BSIM3v32prwb;
            return(OK);
        case BSIM3v32_MOD_PRT:
            value->rValue = model->BSIM3v32prt;
            return(OK);
        case BSIM3v32_MOD_ETA0:
            value->rValue = model->BSIM3v32eta0;
            return(OK);
        case BSIM3v32_MOD_ETAB:
            value->rValue = model->BSIM3v32etab;
            return(OK);
        case BSIM3v32_MOD_PCLM:
            value->rValue = model->BSIM3v32pclm;
            return(OK);
        case BSIM3v32_MOD_PDIBL1:
            value->rValue = model->BSIM3v32pdibl1;
            return(OK);
        case BSIM3v32_MOD_PDIBL2:
            value->rValue = model->BSIM3v32pdibl2;
            return(OK);
        case BSIM3v32_MOD_PDIBLB:
            value->rValue = model->BSIM3v32pdiblb;
            return(OK);
        case BSIM3v32_MOD_PSCBE1:
            value->rValue = model->BSIM3v32pscbe1;
            return(OK);
        case BSIM3v32_MOD_PSCBE2:
            value->rValue = model->BSIM3v32pscbe2;
            return(OK);
        case BSIM3v32_MOD_PVAG:
            value->rValue = model->BSIM3v32pvag;
            return(OK);
        case BSIM3v32_MOD_WR:
            value->rValue = model->BSIM3v32wr;
            return(OK);
        case BSIM3v32_MOD_DWG:
            value->rValue = model->BSIM3v32dwg;
            return(OK);
        case BSIM3v32_MOD_DWB:
            value->rValue = model->BSIM3v32dwb;
            return(OK);
        case BSIM3v32_MOD_B0:
            value->rValue = model->BSIM3v32b0;
            return(OK);
        case BSIM3v32_MOD_B1:
            value->rValue = model->BSIM3v32b1;
            return(OK);
        case BSIM3v32_MOD_ALPHA0:
            value->rValue = model->BSIM3v32alpha0;
            return(OK);
        case BSIM3v32_MOD_ALPHA1:
            value->rValue = model->BSIM3v32alpha1;
            return(OK);
        case BSIM3v32_MOD_BETA0:
            value->rValue = model->BSIM3v32beta0;
            return(OK);
        case BSIM3v32_MOD_IJTH:
            value->rValue = model->BSIM3v32ijth;
            return(OK);
        case BSIM3v32_MOD_VFB:
            value->rValue = model->BSIM3v32vfb;
            return(OK);

        case BSIM3v32_MOD_ELM:
            value->rValue = model->BSIM3v32elm;
            return(OK);
        case BSIM3v32_MOD_CGSL:
            value->rValue = model->BSIM3v32cgsl;
            return(OK);
        case BSIM3v32_MOD_CGDL:
            value->rValue = model->BSIM3v32cgdl;
            return(OK);
        case BSIM3v32_MOD_CKAPPA:
            value->rValue = model->BSIM3v32ckappa;
            return(OK);
        case BSIM3v32_MOD_CF:
            value->rValue = model->BSIM3v32cf;
            return(OK);
        case BSIM3v32_MOD_CLC:
            value->rValue = model->BSIM3v32clc;
            return(OK);
        case BSIM3v32_MOD_CLE:
            value->rValue = model->BSIM3v32cle;
            return(OK);
        case BSIM3v32_MOD_DWC:
            value->rValue = model->BSIM3v32dwc;
            return(OK);
        case BSIM3v32_MOD_DLC:
            value->rValue = model->BSIM3v32dlc;
            return(OK);
        case BSIM3v32_MOD_VFBCV:
            value->rValue = model->BSIM3v32vfbcv;
            return(OK);
        case BSIM3v32_MOD_ACDE:
            value->rValue = model->BSIM3v32acde;
            return(OK);
        case BSIM3v32_MOD_MOIN:
            value->rValue = model->BSIM3v32moin;
            return(OK);
        case BSIM3v32_MOD_NOFF:
            value->rValue = model->BSIM3v32noff;
            return(OK);
        case BSIM3v32_MOD_VOFFCV:
            value->rValue = model->BSIM3v32voffcv;
            return(OK);
        case BSIM3v32_MOD_TCJ:
            value->rValue = model->BSIM3v32tcj;
            return(OK);
        case BSIM3v32_MOD_TPB:
            value->rValue = model->BSIM3v32tpb;
            return(OK);
        case BSIM3v32_MOD_TCJSW:
            value->rValue = model->BSIM3v32tcjsw;
            return(OK);
        case BSIM3v32_MOD_TPBSW:
            value->rValue = model->BSIM3v32tpbsw;
            return(OK);
        case BSIM3v32_MOD_TCJSWG:
            value->rValue = model->BSIM3v32tcjswg;
            return(OK);
        case BSIM3v32_MOD_TPBSWG:
            value->rValue = model->BSIM3v32tpbswg;
            return(OK);

        /* ACM model */
        case BSIM3v32_MOD_HDIF:
            value->rValue = model->BSIM3v32hdif;
            return(OK);
        case BSIM3v32_MOD_LDIF:
            value->rValue = model->BSIM3v32ldif;
            return(OK);
        case BSIM3v32_MOD_LD:
            value->rValue = model->BSIM3v32ld;
            return(OK);
        case BSIM3v32_MOD_RD:
            value->rValue = model->BSIM3v32rd;
            return(OK);
        case BSIM3v32_MOD_RS:
            value->rValue = model->BSIM3v32rs;
            return(OK);
        case BSIM3v32_MOD_RDC:
            value->rValue = model->BSIM3v32rdc;
            return(OK);
        case BSIM3v32_MOD_RSC:
            value->rValue = model->BSIM3v32rsc;
            return(OK);
        case BSIM3v32_MOD_WMLT:
            value->rValue = model->BSIM3v32wmlt;
            return(OK);

        case BSIM3v32_MOD_LMLT:
            value->rValue = model->BSIM3v32lmlt;
            return(OK);

        /* Length dependence */
        case  BSIM3v32_MOD_LCDSC :
          value->rValue = model->BSIM3v32lcdsc;
            return(OK);
        case  BSIM3v32_MOD_LCDSCB :
          value->rValue = model->BSIM3v32lcdscb;
            return(OK);
        case  BSIM3v32_MOD_LCDSCD :
          value->rValue = model->BSIM3v32lcdscd;
            return(OK);
        case  BSIM3v32_MOD_LCIT :
          value->rValue = model->BSIM3v32lcit;
            return(OK);
        case  BSIM3v32_MOD_LNFACTOR :
          value->rValue = model->BSIM3v32lnfactor;
            return(OK);
        case BSIM3v32_MOD_LXJ:
            value->rValue = model->BSIM3v32lxj;
            return(OK);
        case BSIM3v32_MOD_LVSAT:
            value->rValue = model->BSIM3v32lvsat;
            return(OK);
        case BSIM3v32_MOD_LAT:
            value->rValue = model->BSIM3v32lat;
            return(OK);
        case BSIM3v32_MOD_LA0:
            value->rValue = model->BSIM3v32la0;
            return(OK);
        case BSIM3v32_MOD_LAGS:
            value->rValue = model->BSIM3v32lags;
            return(OK);
        case BSIM3v32_MOD_LA1:
            value->rValue = model->BSIM3v32la1;
            return(OK);
        case BSIM3v32_MOD_LA2:
            value->rValue = model->BSIM3v32la2;
            return(OK);
        case BSIM3v32_MOD_LKETA:
            value->rValue = model->BSIM3v32lketa;
            return(OK);
        case BSIM3v32_MOD_LNSUB:
            value->rValue = model->BSIM3v32lnsub;
            return(OK);
        case BSIM3v32_MOD_LNPEAK:
            value->rValue = model->BSIM3v32lnpeak;
            return(OK);
        case BSIM3v32_MOD_LNGATE:
            value->rValue = model->BSIM3v32lngate;
            return(OK);
        case BSIM3v32_MOD_LGAMMA1:
            value->rValue = model->BSIM3v32lgamma1;
            return(OK);
        case BSIM3v32_MOD_LGAMMA2:
            value->rValue = model->BSIM3v32lgamma2;
            return(OK);
        case BSIM3v32_MOD_LVBX:
            value->rValue = model->BSIM3v32lvbx;
            return(OK);
        case BSIM3v32_MOD_LVBM:
            value->rValue = model->BSIM3v32lvbm;
            return(OK);
        case BSIM3v32_MOD_LXT:
            value->rValue = model->BSIM3v32lxt;
            return(OK);
        case  BSIM3v32_MOD_LK1:
          value->rValue = model->BSIM3v32lk1;
            return(OK);
        case  BSIM3v32_MOD_LKT1:
          value->rValue = model->BSIM3v32lkt1;
            return(OK);
        case  BSIM3v32_MOD_LKT1L:
          value->rValue = model->BSIM3v32lkt1l;
            return(OK);
        case  BSIM3v32_MOD_LKT2 :
          value->rValue = model->BSIM3v32lkt2;
            return(OK);
        case  BSIM3v32_MOD_LK2 :
          value->rValue = model->BSIM3v32lk2;
            return(OK);
        case  BSIM3v32_MOD_LK3:
          value->rValue = model->BSIM3v32lk3;
            return(OK);
        case  BSIM3v32_MOD_LK3B:
          value->rValue = model->BSIM3v32lk3b;
            return(OK);
        case  BSIM3v32_MOD_LW0:
          value->rValue = model->BSIM3v32lw0;
            return(OK);
        case  BSIM3v32_MOD_LNLX:
          value->rValue = model->BSIM3v32lnlx;
            return(OK);
        case  BSIM3v32_MOD_LDVT0:
          value->rValue = model->BSIM3v32ldvt0;
            return(OK);
        case  BSIM3v32_MOD_LDVT1 :
          value->rValue = model->BSIM3v32ldvt1;
            return(OK);
        case  BSIM3v32_MOD_LDVT2 :
          value->rValue = model->BSIM3v32ldvt2;
            return(OK);
        case  BSIM3v32_MOD_LDVT0W :
          value->rValue = model->BSIM3v32ldvt0w;
            return(OK);
        case  BSIM3v32_MOD_LDVT1W :
          value->rValue = model->BSIM3v32ldvt1w;
            return(OK);
        case  BSIM3v32_MOD_LDVT2W :
          value->rValue = model->BSIM3v32ldvt2w;
            return(OK);
        case  BSIM3v32_MOD_LDROUT :
          value->rValue = model->BSIM3v32ldrout;
            return(OK);
        case  BSIM3v32_MOD_LDSUB :
          value->rValue = model->BSIM3v32ldsub;
            return(OK);
        case BSIM3v32_MOD_LVTH0:
            value->rValue = model->BSIM3v32lvth0;
            return(OK);
        case BSIM3v32_MOD_LUA:
            value->rValue = model->BSIM3v32lua;
            return(OK);
        case BSIM3v32_MOD_LUA1:
            value->rValue = model->BSIM3v32lua1;
            return(OK);
        case BSIM3v32_MOD_LUB:
            value->rValue = model->BSIM3v32lub;
            return(OK);
        case BSIM3v32_MOD_LUB1:
            value->rValue = model->BSIM3v32lub1;
            return(OK);
        case BSIM3v32_MOD_LUC:
            value->rValue = model->BSIM3v32luc;
            return(OK);
        case BSIM3v32_MOD_LUC1:
            value->rValue = model->BSIM3v32luc1;
            return(OK);
        case BSIM3v32_MOD_LU0:
            value->rValue = model->BSIM3v32lu0;
            return(OK);
        case BSIM3v32_MOD_LUTE:
            value->rValue = model->BSIM3v32lute;
            return(OK);
        case BSIM3v32_MOD_LVOFF:
            value->rValue = model->BSIM3v32lvoff;
            return(OK);
        case BSIM3v32_MOD_LDELTA:
            value->rValue = model->BSIM3v32ldelta;
            return(OK);
        case BSIM3v32_MOD_LRDSW:
            value->rValue = model->BSIM3v32lrdsw;
            return(OK);
        case BSIM3v32_MOD_LPRWB:
            value->rValue = model->BSIM3v32lprwb;
            return(OK);
        case BSIM3v32_MOD_LPRWG:
            value->rValue = model->BSIM3v32lprwg;
            return(OK);
        case BSIM3v32_MOD_LPRT:
            value->rValue = model->BSIM3v32lprt;
            return(OK);
        case BSIM3v32_MOD_LETA0:
            value->rValue = model->BSIM3v32leta0;
            return(OK);
        case BSIM3v32_MOD_LETAB:
            value->rValue = model->BSIM3v32letab;
            return(OK);
        case BSIM3v32_MOD_LPCLM:
            value->rValue = model->BSIM3v32lpclm;
            return(OK);
        case BSIM3v32_MOD_LPDIBL1:
            value->rValue = model->BSIM3v32lpdibl1;
            return(OK);
        case BSIM3v32_MOD_LPDIBL2:
            value->rValue = model->BSIM3v32lpdibl2;
            return(OK);
        case BSIM3v32_MOD_LPDIBLB:
            value->rValue = model->BSIM3v32lpdiblb;
            return(OK);
        case BSIM3v32_MOD_LPSCBE1:
            value->rValue = model->BSIM3v32lpscbe1;
            return(OK);
        case BSIM3v32_MOD_LPSCBE2:
            value->rValue = model->BSIM3v32lpscbe2;
            return(OK);
        case BSIM3v32_MOD_LPVAG:
            value->rValue = model->BSIM3v32lpvag;
            return(OK);
        case BSIM3v32_MOD_LWR:
            value->rValue = model->BSIM3v32lwr;
            return(OK);
        case BSIM3v32_MOD_LDWG:
            value->rValue = model->BSIM3v32ldwg;
            return(OK);
        case BSIM3v32_MOD_LDWB:
            value->rValue = model->BSIM3v32ldwb;
            return(OK);
        case BSIM3v32_MOD_LB0:
            value->rValue = model->BSIM3v32lb0;
            return(OK);
        case BSIM3v32_MOD_LB1:
            value->rValue = model->BSIM3v32lb1;
            return(OK);
        case BSIM3v32_MOD_LALPHA0:
            value->rValue = model->BSIM3v32lalpha0;
            return(OK);
        case BSIM3v32_MOD_LALPHA1:
            value->rValue = model->BSIM3v32lalpha1;
            return(OK);
        case BSIM3v32_MOD_LBETA0:
            value->rValue = model->BSIM3v32lbeta0;
            return(OK);
        case BSIM3v32_MOD_LVFB:
            value->rValue = model->BSIM3v32lvfb;
            return(OK);

        case BSIM3v32_MOD_LELM:
            value->rValue = model->BSIM3v32lelm;
            return(OK);
        case BSIM3v32_MOD_LCGSL:
            value->rValue = model->BSIM3v32lcgsl;
            return(OK);
        case BSIM3v32_MOD_LCGDL:
            value->rValue = model->BSIM3v32lcgdl;
            return(OK);
        case BSIM3v32_MOD_LCKAPPA:
            value->rValue = model->BSIM3v32lckappa;
            return(OK);
        case BSIM3v32_MOD_LCF:
            value->rValue = model->BSIM3v32lcf;
            return(OK);
        case BSIM3v32_MOD_LCLC:
            value->rValue = model->BSIM3v32lclc;
            return(OK);
        case BSIM3v32_MOD_LCLE:
            value->rValue = model->BSIM3v32lcle;
            return(OK);
        case BSIM3v32_MOD_LVFBCV:
            value->rValue = model->BSIM3v32lvfbcv;
            return(OK);
        case BSIM3v32_MOD_LACDE:
            value->rValue = model->BSIM3v32lacde;
            return(OK);
        case BSIM3v32_MOD_LMOIN:
            value->rValue = model->BSIM3v32lmoin;
            return(OK);
        case BSIM3v32_MOD_LNOFF:
            value->rValue = model->BSIM3v32lnoff;
            return(OK);
        case BSIM3v32_MOD_LVOFFCV:
            value->rValue = model->BSIM3v32lvoffcv;
            return(OK);

        /* Width dependence */
        case  BSIM3v32_MOD_WCDSC :
          value->rValue = model->BSIM3v32wcdsc;
            return(OK);
        case  BSIM3v32_MOD_WCDSCB :
          value->rValue = model->BSIM3v32wcdscb;
            return(OK);
        case  BSIM3v32_MOD_WCDSCD :
          value->rValue = model->BSIM3v32wcdscd;
            return(OK);
        case  BSIM3v32_MOD_WCIT :
          value->rValue = model->BSIM3v32wcit;
            return(OK);
        case  BSIM3v32_MOD_WNFACTOR :
          value->rValue = model->BSIM3v32wnfactor;
            return(OK);
        case BSIM3v32_MOD_WXJ:
            value->rValue = model->BSIM3v32wxj;
            return(OK);
        case BSIM3v32_MOD_WVSAT:
            value->rValue = model->BSIM3v32wvsat;
            return(OK);
        case BSIM3v32_MOD_WAT:
            value->rValue = model->BSIM3v32wat;
            return(OK);
        case BSIM3v32_MOD_WA0:
            value->rValue = model->BSIM3v32wa0;
            return(OK);
        case BSIM3v32_MOD_WAGS:
            value->rValue = model->BSIM3v32wags;
            return(OK);
        case BSIM3v32_MOD_WA1:
            value->rValue = model->BSIM3v32wa1;
            return(OK);
        case BSIM3v32_MOD_WA2:
            value->rValue = model->BSIM3v32wa2;
            return(OK);
        case BSIM3v32_MOD_WKETA:
            value->rValue = model->BSIM3v32wketa;
            return(OK);
        case BSIM3v32_MOD_WNSUB:
            value->rValue = model->BSIM3v32wnsub;
            return(OK);
        case BSIM3v32_MOD_WNPEAK:
            value->rValue = model->BSIM3v32wnpeak;
            return(OK);
        case BSIM3v32_MOD_WNGATE:
            value->rValue = model->BSIM3v32wngate;
            return(OK);
        case BSIM3v32_MOD_WGAMMA1:
            value->rValue = model->BSIM3v32wgamma1;
            return(OK);
        case BSIM3v32_MOD_WGAMMA2:
            value->rValue = model->BSIM3v32wgamma2;
            return(OK);
        case BSIM3v32_MOD_WVBX:
            value->rValue = model->BSIM3v32wvbx;
            return(OK);
        case BSIM3v32_MOD_WVBM:
            value->rValue = model->BSIM3v32wvbm;
            return(OK);
        case BSIM3v32_MOD_WXT:
            value->rValue = model->BSIM3v32wxt;
            return(OK);
        case  BSIM3v32_MOD_WK1:
          value->rValue = model->BSIM3v32wk1;
            return(OK);
        case  BSIM3v32_MOD_WKT1:
          value->rValue = model->BSIM3v32wkt1;
            return(OK);
        case  BSIM3v32_MOD_WKT1L:
          value->rValue = model->BSIM3v32wkt1l;
            return(OK);
        case  BSIM3v32_MOD_WKT2 :
          value->rValue = model->BSIM3v32wkt2;
            return(OK);
        case  BSIM3v32_MOD_WK2 :
          value->rValue = model->BSIM3v32wk2;
            return(OK);
        case  BSIM3v32_MOD_WK3:
          value->rValue = model->BSIM3v32wk3;
            return(OK);
        case  BSIM3v32_MOD_WK3B:
          value->rValue = model->BSIM3v32wk3b;
            return(OK);
        case  BSIM3v32_MOD_WW0:
          value->rValue = model->BSIM3v32ww0;
            return(OK);
        case  BSIM3v32_MOD_WNLX:
          value->rValue = model->BSIM3v32wnlx;
            return(OK);
        case  BSIM3v32_MOD_WDVT0:
          value->rValue = model->BSIM3v32wdvt0;
            return(OK);
        case  BSIM3v32_MOD_WDVT1 :
          value->rValue = model->BSIM3v32wdvt1;
            return(OK);
        case  BSIM3v32_MOD_WDVT2 :
          value->rValue = model->BSIM3v32wdvt2;
            return(OK);
        case  BSIM3v32_MOD_WDVT0W :
          value->rValue = model->BSIM3v32wdvt0w;
            return(OK);
        case  BSIM3v32_MOD_WDVT1W :
          value->rValue = model->BSIM3v32wdvt1w;
            return(OK);
        case  BSIM3v32_MOD_WDVT2W :
          value->rValue = model->BSIM3v32wdvt2w;
            return(OK);
        case  BSIM3v32_MOD_WDROUT :
          value->rValue = model->BSIM3v32wdrout;
            return(OK);
        case  BSIM3v32_MOD_WDSUB :
          value->rValue = model->BSIM3v32wdsub;
            return(OK);
        case BSIM3v32_MOD_WVTH0:
            value->rValue = model->BSIM3v32wvth0;
            return(OK);
        case BSIM3v32_MOD_WUA:
            value->rValue = model->BSIM3v32wua;
            return(OK);
        case BSIM3v32_MOD_WUA1:
            value->rValue = model->BSIM3v32wua1;
            return(OK);
        case BSIM3v32_MOD_WUB:
            value->rValue = model->BSIM3v32wub;
            return(OK);
        case BSIM3v32_MOD_WUB1:
            value->rValue = model->BSIM3v32wub1;
            return(OK);
        case BSIM3v32_MOD_WUC:
            value->rValue = model->BSIM3v32wuc;
            return(OK);
        case BSIM3v32_MOD_WUC1:
            value->rValue = model->BSIM3v32wuc1;
            return(OK);
        case BSIM3v32_MOD_WU0:
            value->rValue = model->BSIM3v32wu0;
            return(OK);
        case BSIM3v32_MOD_WUTE:
            value->rValue = model->BSIM3v32wute;
            return(OK);
        case BSIM3v32_MOD_WVOFF:
            value->rValue = model->BSIM3v32wvoff;
            return(OK);
        case BSIM3v32_MOD_WDELTA:
            value->rValue = model->BSIM3v32wdelta;
            return(OK);
        case BSIM3v32_MOD_WRDSW:
            value->rValue = model->BSIM3v32wrdsw;
            return(OK);
        case BSIM3v32_MOD_WPRWB:
            value->rValue = model->BSIM3v32wprwb;
            return(OK);
        case BSIM3v32_MOD_WPRWG:
            value->rValue = model->BSIM3v32wprwg;
            return(OK);
        case BSIM3v32_MOD_WPRT:
            value->rValue = model->BSIM3v32wprt;
            return(OK);
        case BSIM3v32_MOD_WETA0:
            value->rValue = model->BSIM3v32weta0;
            return(OK);
        case BSIM3v32_MOD_WETAB:
            value->rValue = model->BSIM3v32wetab;
            return(OK);
        case BSIM3v32_MOD_WPCLM:
            value->rValue = model->BSIM3v32wpclm;
            return(OK);
        case BSIM3v32_MOD_WPDIBL1:
            value->rValue = model->BSIM3v32wpdibl1;
            return(OK);
        case BSIM3v32_MOD_WPDIBL2:
            value->rValue = model->BSIM3v32wpdibl2;
            return(OK);
        case BSIM3v32_MOD_WPDIBLB:
            value->rValue = model->BSIM3v32wpdiblb;
            return(OK);
        case BSIM3v32_MOD_WPSCBE1:
            value->rValue = model->BSIM3v32wpscbe1;
            return(OK);
        case BSIM3v32_MOD_WPSCBE2:
            value->rValue = model->BSIM3v32wpscbe2;
            return(OK);
        case BSIM3v32_MOD_WPVAG:
            value->rValue = model->BSIM3v32wpvag;
            return(OK);
        case BSIM3v32_MOD_WWR:
            value->rValue = model->BSIM3v32wwr;
            return(OK);
        case BSIM3v32_MOD_WDWG:
            value->rValue = model->BSIM3v32wdwg;
            return(OK);
        case BSIM3v32_MOD_WDWB:
            value->rValue = model->BSIM3v32wdwb;
            return(OK);
        case BSIM3v32_MOD_WB0:
            value->rValue = model->BSIM3v32wb0;
            return(OK);
        case BSIM3v32_MOD_WB1:
            value->rValue = model->BSIM3v32wb1;
            return(OK);
        case BSIM3v32_MOD_WALPHA0:
            value->rValue = model->BSIM3v32walpha0;
            return(OK);
        case BSIM3v32_MOD_WALPHA1:
            value->rValue = model->BSIM3v32walpha1;
            return(OK);
        case BSIM3v32_MOD_WBETA0:
            value->rValue = model->BSIM3v32wbeta0;
            return(OK);
        case BSIM3v32_MOD_WVFB:
            value->rValue = model->BSIM3v32wvfb;
            return(OK);

        case BSIM3v32_MOD_WELM:
            value->rValue = model->BSIM3v32welm;
            return(OK);
        case BSIM3v32_MOD_WCGSL:
            value->rValue = model->BSIM3v32wcgsl;
            return(OK);
        case BSIM3v32_MOD_WCGDL:
            value->rValue = model->BSIM3v32wcgdl;
            return(OK);
        case BSIM3v32_MOD_WCKAPPA:
            value->rValue = model->BSIM3v32wckappa;
            return(OK);
        case BSIM3v32_MOD_WCF:
            value->rValue = model->BSIM3v32wcf;
            return(OK);
        case BSIM3v32_MOD_WCLC:
            value->rValue = model->BSIM3v32wclc;
            return(OK);
        case BSIM3v32_MOD_WCLE:
            value->rValue = model->BSIM3v32wcle;
            return(OK);
        case BSIM3v32_MOD_WVFBCV:
            value->rValue = model->BSIM3v32wvfbcv;
            return(OK);
        case BSIM3v32_MOD_WACDE:
            value->rValue = model->BSIM3v32wacde;
            return(OK);
        case BSIM3v32_MOD_WMOIN:
            value->rValue = model->BSIM3v32wmoin;
            return(OK);
        case BSIM3v32_MOD_WNOFF:
            value->rValue = model->BSIM3v32wnoff;
            return(OK);
        case BSIM3v32_MOD_WVOFFCV:
            value->rValue = model->BSIM3v32wvoffcv;
            return(OK);

        /* Cross-term dependence */
        case  BSIM3v32_MOD_PCDSC :
          value->rValue = model->BSIM3v32pcdsc;
            return(OK);
        case  BSIM3v32_MOD_PCDSCB :
          value->rValue = model->BSIM3v32pcdscb;
            return(OK);
        case  BSIM3v32_MOD_PCDSCD :
          value->rValue = model->BSIM3v32pcdscd;
            return(OK);
         case  BSIM3v32_MOD_PCIT :
          value->rValue = model->BSIM3v32pcit;
            return(OK);
        case  BSIM3v32_MOD_PNFACTOR :
          value->rValue = model->BSIM3v32pnfactor;
            return(OK);
        case BSIM3v32_MOD_PXJ:
            value->rValue = model->BSIM3v32pxj;
            return(OK);
        case BSIM3v32_MOD_PVSAT:
            value->rValue = model->BSIM3v32pvsat;
            return(OK);
        case BSIM3v32_MOD_PAT:
            value->rValue = model->BSIM3v32pat;
            return(OK);
        case BSIM3v32_MOD_PA0:
            value->rValue = model->BSIM3v32pa0;
            return(OK);
        case BSIM3v32_MOD_PAGS:
            value->rValue = model->BSIM3v32pags;
            return(OK);
        case BSIM3v32_MOD_PA1:
            value->rValue = model->BSIM3v32pa1;
            return(OK);
        case BSIM3v32_MOD_PA2:
            value->rValue = model->BSIM3v32pa2;
            return(OK);
        case BSIM3v32_MOD_PKETA:
            value->rValue = model->BSIM3v32pketa;
            return(OK);
        case BSIM3v32_MOD_PNSUB:
            value->rValue = model->BSIM3v32pnsub;
            return(OK);
        case BSIM3v32_MOD_PNPEAK:
            value->rValue = model->BSIM3v32pnpeak;
            return(OK);
        case BSIM3v32_MOD_PNGATE:
            value->rValue = model->BSIM3v32pngate;
            return(OK);
        case BSIM3v32_MOD_PGAMMA1:
            value->rValue = model->BSIM3v32pgamma1;
            return(OK);
        case BSIM3v32_MOD_PGAMMA2:
            value->rValue = model->BSIM3v32pgamma2;
            return(OK);
        case BSIM3v32_MOD_PVBX:
            value->rValue = model->BSIM3v32pvbx;
            return(OK);
        case BSIM3v32_MOD_PVBM:
            value->rValue = model->BSIM3v32pvbm;
            return(OK);
        case BSIM3v32_MOD_PXT:
            value->rValue = model->BSIM3v32pxt;
            return(OK);
        case  BSIM3v32_MOD_PK1:
          value->rValue = model->BSIM3v32pk1;
            return(OK);
        case  BSIM3v32_MOD_PKT1:
          value->rValue = model->BSIM3v32pkt1;
            return(OK);
        case  BSIM3v32_MOD_PKT1L:
          value->rValue = model->BSIM3v32pkt1l;
            return(OK);
        case  BSIM3v32_MOD_PKT2 :
          value->rValue = model->BSIM3v32pkt2;
            return(OK);
        case  BSIM3v32_MOD_PK2 :
          value->rValue = model->BSIM3v32pk2;
            return(OK);
        case  BSIM3v32_MOD_PK3:
          value->rValue = model->BSIM3v32pk3;
            return(OK);
        case  BSIM3v32_MOD_PK3B:
          value->rValue = model->BSIM3v32pk3b;
            return(OK);
        case  BSIM3v32_MOD_PW0:
          value->rValue = model->BSIM3v32pw0;
            return(OK);
        case  BSIM3v32_MOD_PNLX:
          value->rValue = model->BSIM3v32pnlx;
            return(OK);
        case  BSIM3v32_MOD_PDVT0 :
          value->rValue = model->BSIM3v32pdvt0;
            return(OK);
        case  BSIM3v32_MOD_PDVT1 :
          value->rValue = model->BSIM3v32pdvt1;
            return(OK);
        case  BSIM3v32_MOD_PDVT2 :
          value->rValue = model->BSIM3v32pdvt2;
            return(OK);
        case  BSIM3v32_MOD_PDVT0W :
          value->rValue = model->BSIM3v32pdvt0w;
            return(OK);
        case  BSIM3v32_MOD_PDVT1W :
          value->rValue = model->BSIM3v32pdvt1w;
            return(OK);
        case  BSIM3v32_MOD_PDVT2W :
            value->rValue = model->BSIM3v32pdvt2w;
            return(OK);
        case  BSIM3v32_MOD_PDROUT :
            value->rValue = model->BSIM3v32pdrout;
            return(OK);
        case  BSIM3v32_MOD_PDSUB :
            value->rValue = model->BSIM3v32pdsub;
            return(OK);
        case BSIM3v32_MOD_PVTH0:
            value->rValue = model->BSIM3v32pvth0;
            return(OK);
        case BSIM3v32_MOD_PUA:
            value->rValue = model->BSIM3v32pua;
            return(OK);
        case BSIM3v32_MOD_PUA1:
            value->rValue = model->BSIM3v32pua1;
            return(OK);
        case BSIM3v32_MOD_PUB:
            value->rValue = model->BSIM3v32pub;
            return(OK);
        case BSIM3v32_MOD_PUB1:
            value->rValue = model->BSIM3v32pub1;
            return(OK);
        case BSIM3v32_MOD_PUC:
            value->rValue = model->BSIM3v32puc;
            return(OK);
        case BSIM3v32_MOD_PUC1:
            value->rValue = model->BSIM3v32puc1;
            return(OK);
        case BSIM3v32_MOD_PU0:
            value->rValue = model->BSIM3v32pu0;
            return(OK);
        case BSIM3v32_MOD_PUTE:
            value->rValue = model->BSIM3v32pute;
            return(OK);
        case BSIM3v32_MOD_PVOFF:
            value->rValue = model->BSIM3v32pvoff;
            return(OK);
        case BSIM3v32_MOD_PDELTA:
            value->rValue = model->BSIM3v32pdelta;
            return(OK);
        case BSIM3v32_MOD_PRDSW:
            value->rValue = model->BSIM3v32prdsw;
            return(OK);
        case BSIM3v32_MOD_PPRWB:
            value->rValue = model->BSIM3v32pprwb;
            return(OK);
        case BSIM3v32_MOD_PPRWG:
            value->rValue = model->BSIM3v32pprwg;
            return(OK);
        case BSIM3v32_MOD_PPRT:
            value->rValue = model->BSIM3v32pprt;
            return(OK);
        case BSIM3v32_MOD_PETA0:
            value->rValue = model->BSIM3v32peta0;
            return(OK);
        case BSIM3v32_MOD_PETAB:
            value->rValue = model->BSIM3v32petab;
            return(OK);
        case BSIM3v32_MOD_PPCLM:
            value->rValue = model->BSIM3v32ppclm;
            return(OK);
        case BSIM3v32_MOD_PPDIBL1:
            value->rValue = model->BSIM3v32ppdibl1;
            return(OK);
        case BSIM3v32_MOD_PPDIBL2:
            value->rValue = model->BSIM3v32ppdibl2;
            return(OK);
        case BSIM3v32_MOD_PPDIBLB:
            value->rValue = model->BSIM3v32ppdiblb;
            return(OK);
        case BSIM3v32_MOD_PPSCBE1:
            value->rValue = model->BSIM3v32ppscbe1;
            return(OK);
        case BSIM3v32_MOD_PPSCBE2:
            value->rValue = model->BSIM3v32ppscbe2;
            return(OK);
        case BSIM3v32_MOD_PPVAG:
            value->rValue = model->BSIM3v32ppvag;
            return(OK);
        case BSIM3v32_MOD_PWR:
            value->rValue = model->BSIM3v32pwr;
            return(OK);
        case BSIM3v32_MOD_PDWG:
            value->rValue = model->BSIM3v32pdwg;
            return(OK);
        case BSIM3v32_MOD_PDWB:
            value->rValue = model->BSIM3v32pdwb;
            return(OK);
        case BSIM3v32_MOD_PB0:
            value->rValue = model->BSIM3v32pb0;
            return(OK);
        case BSIM3v32_MOD_PB1:
            value->rValue = model->BSIM3v32pb1;
            return(OK);
        case BSIM3v32_MOD_PALPHA0:
            value->rValue = model->BSIM3v32palpha0;
            return(OK);
        case BSIM3v32_MOD_PALPHA1:
            value->rValue = model->BSIM3v32palpha1;
            return(OK);
        case BSIM3v32_MOD_PBETA0:
            value->rValue = model->BSIM3v32pbeta0;
            return(OK);
        case BSIM3v32_MOD_PVFB:
            value->rValue = model->BSIM3v32pvfb;
            return(OK);

        case BSIM3v32_MOD_PELM:
            value->rValue = model->BSIM3v32pelm;
            return(OK);
        case BSIM3v32_MOD_PCGSL:
            value->rValue = model->BSIM3v32pcgsl;
            return(OK);
        case BSIM3v32_MOD_PCGDL:
            value->rValue = model->BSIM3v32pcgdl;
            return(OK);
        case BSIM3v32_MOD_PCKAPPA:
            value->rValue = model->BSIM3v32pckappa;
            return(OK);
        case BSIM3v32_MOD_PCF:
            value->rValue = model->BSIM3v32pcf;
            return(OK);
        case BSIM3v32_MOD_PCLC:
            value->rValue = model->BSIM3v32pclc;
            return(OK);
        case BSIM3v32_MOD_PCLE:
            value->rValue = model->BSIM3v32pcle;
            return(OK);
        case BSIM3v32_MOD_PVFBCV:
            value->rValue = model->BSIM3v32pvfbcv;
            return(OK);
        case BSIM3v32_MOD_PACDE:
            value->rValue = model->BSIM3v32pacde;
            return(OK);
        case BSIM3v32_MOD_PMOIN:
            value->rValue = model->BSIM3v32pmoin;
            return(OK);
        case BSIM3v32_MOD_PNOFF:
            value->rValue = model->BSIM3v32pnoff;
            return(OK);
        case BSIM3v32_MOD_PVOFFCV:
            value->rValue = model->BSIM3v32pvoffcv;
            return(OK);

        case  BSIM3v32_MOD_TNOM :
            value->rValue = model->BSIM3v32tnom;
            return(OK);
        case BSIM3v32_MOD_CGSO:
            value->rValue = model->BSIM3v32cgso;
            return(OK);
        case BSIM3v32_MOD_CGDO:
            value->rValue = model->BSIM3v32cgdo;
            return(OK);
        case BSIM3v32_MOD_CGBO:
            value->rValue = model->BSIM3v32cgbo;
            return(OK);
        case BSIM3v32_MOD_XPART:
            value->rValue = model->BSIM3v32xpart;
            return(OK);
        case BSIM3v32_MOD_RSH:
            value->rValue = model->BSIM3v32sheetResistance;
            return(OK);
        case BSIM3v32_MOD_JS:
            value->rValue = model->BSIM3v32jctSatCurDensity;
            return(OK);
        case BSIM3v32_MOD_JSW:
            value->rValue = model->BSIM3v32jctSidewallSatCurDensity;
            return(OK);
        case BSIM3v32_MOD_PB:
            value->rValue = model->BSIM3v32bulkJctPotential;
            return(OK);
        case BSIM3v32_MOD_MJ:
            value->rValue = model->BSIM3v32bulkJctBotGradingCoeff;
            return(OK);
        case BSIM3v32_MOD_PBSW:
            value->rValue = model->BSIM3v32sidewallJctPotential;
            return(OK);
        case BSIM3v32_MOD_MJSW:
            value->rValue = model->BSIM3v32bulkJctSideGradingCoeff;
            return(OK);
        case BSIM3v32_MOD_CJ:
            value->rValue = model->BSIM3v32unitAreaJctCap;
            return(OK);
        case BSIM3v32_MOD_CJSW:
            value->rValue = model->BSIM3v32unitLengthSidewallJctCap;
            return(OK);
        case BSIM3v32_MOD_PBSWG:
            value->rValue = model->BSIM3v32GatesidewallJctPotential;
            return(OK);
        case BSIM3v32_MOD_MJSWG:
            value->rValue = model->BSIM3v32bulkJctGateSideGradingCoeff;
            return(OK);
        case BSIM3v32_MOD_CJSWG:
            value->rValue = model->BSIM3v32unitLengthGateSidewallJctCap;
            return(OK);
        case BSIM3v32_MOD_NJ:
            value->rValue = model->BSIM3v32jctEmissionCoeff;
            return(OK);
        case BSIM3v32_MOD_XTI:
            value->rValue = model->BSIM3v32jctTempExponent;
            return(OK);
        case BSIM3v32_MOD_LINT:
            value->rValue = model->BSIM3v32Lint;
            return(OK);
        case BSIM3v32_MOD_LL:
            value->rValue = model->BSIM3v32Ll;
            return(OK);
        case BSIM3v32_MOD_LLC:
            value->rValue = model->BSIM3v32Llc;
            return(OK);
        case BSIM3v32_MOD_LLN:
            value->rValue = model->BSIM3v32Lln;
            return(OK);
        case BSIM3v32_MOD_LW:
            value->rValue = model->BSIM3v32Lw;
            return(OK);
        case BSIM3v32_MOD_LWC:
            value->rValue = model->BSIM3v32Lwc;
            return(OK);
        case BSIM3v32_MOD_LWN:
            value->rValue = model->BSIM3v32Lwn;
            return(OK);
        case BSIM3v32_MOD_LWL:
            value->rValue = model->BSIM3v32Lwl;
            return(OK);
        case BSIM3v32_MOD_LWLC:
            value->rValue = model->BSIM3v32Lwlc;
            return(OK);
        case BSIM3v32_MOD_LMIN:
            value->rValue = model->BSIM3v32Lmin;
            return(OK);
        case BSIM3v32_MOD_LMAX:
            value->rValue = model->BSIM3v32Lmax;
            return(OK);
        case BSIM3v32_MOD_WINT:
            value->rValue = model->BSIM3v32Wint;
            return(OK);
        case BSIM3v32_MOD_WL:
            value->rValue = model->BSIM3v32Wl;
            return(OK);
        case BSIM3v32_MOD_WLC:
            value->rValue = model->BSIM3v32Wlc;
            return(OK);
        case BSIM3v32_MOD_WLN:
            value->rValue = model->BSIM3v32Wln;
            return(OK);
        case BSIM3v32_MOD_WW:
            value->rValue = model->BSIM3v32Ww;
            return(OK);
        case BSIM3v32_MOD_WWC:
            value->rValue = model->BSIM3v32Wwc;
            return(OK);
        case BSIM3v32_MOD_WWN:
            value->rValue = model->BSIM3v32Wwn;
            return(OK);
        case BSIM3v32_MOD_WWL:
            value->rValue = model->BSIM3v32Wwl;
            return(OK);
        case BSIM3v32_MOD_WWLC:
            value->rValue = model->BSIM3v32Wwlc;
            return(OK);
        case BSIM3v32_MOD_WMIN:
            value->rValue = model->BSIM3v32Wmin;
            return(OK);
        case BSIM3v32_MOD_WMAX:
            value->rValue = model->BSIM3v32Wmax;
            return(OK);

        case BSIM3v32_MOD_XL:
            value->rValue = model->BSIM3v32xl;
            return(OK);
        case BSIM3v32_MOD_XW:
            value->rValue = model->BSIM3v32xw;
            return(OK);

        case BSIM3v32_MOD_NOIA:
            value->rValue = model->BSIM3v32oxideTrapDensityA;
            return(OK);
        case BSIM3v32_MOD_NOIB:
            value->rValue = model->BSIM3v32oxideTrapDensityB;
            return(OK);
        case BSIM3v32_MOD_NOIC:
            value->rValue = model->BSIM3v32oxideTrapDensityC;
            return(OK);
        case BSIM3v32_MOD_EM:
            value->rValue = model->BSIM3v32em;
            return(OK);
        case BSIM3v32_MOD_EF:
            value->rValue = model->BSIM3v32ef;
            return(OK);
        case BSIM3v32_MOD_AF:
            value->rValue = model->BSIM3v32af;
            return(OK);
        case BSIM3v32_MOD_KF:
            value->rValue = model->BSIM3v32kf;
            return(OK);

        case BSIM3v32_MOD_VGS_MAX:
            value->rValue = model->BSIM3v32vgsMax;
            return(OK);
        case BSIM3v32_MOD_VGD_MAX:
            value->rValue = model->BSIM3v32vgdMax;
            return(OK);
        case BSIM3v32_MOD_VGB_MAX:
            value->rValue = model->BSIM3v32vgbMax;
            return(OK);
        case BSIM3v32_MOD_VDS_MAX:
            value->rValue = model->BSIM3v32vdsMax;
            return(OK);
        case BSIM3v32_MOD_VBS_MAX:
            value->rValue = model->BSIM3v32vbsMax;
            return(OK);
        case BSIM3v32_MOD_VBD_MAX:
            value->rValue = model->BSIM3v32vbdMax;
            return(OK);
        case BSIM3v32_MOD_VGSR_MAX:
            value->rValue = model->BSIM3v32vgsrMax;
            return(OK);
        case BSIM3v32_MOD_VGDR_MAX:
            value->rValue = model->BSIM3v32vgdrMax;
            return(OK);
        case BSIM3v32_MOD_VGBR_MAX:
            value->rValue = model->BSIM3v32vgbrMax;
            return(OK);
        case BSIM3v32_MOD_VBSR_MAX:
            value->rValue = model->BSIM3v32vbsrMax;
            return(OK);
        case BSIM3v32_MOD_VBDR_MAX:
            value->rValue = model->BSIM3v32vbdrMax;
            return(OK);

        default:
            return(E_BADPARM);
    }
    /* NOTREACHED */
}
