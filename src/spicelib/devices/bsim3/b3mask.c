/**** BSIM3v3.3.0, Released by Xuemei Xi 07/29/2005 ****/

/**********
 * Copyright 2004 Regents of the University of California. All rights reserved.
 * File: b3mask.c of BSIM3v3.3.0
 * Author: 1995 Min-Chie Jeng and Mansun Chan.
 * Author: 1997-1999 Weidong Liu.
 * Author: 2001  Xuemei Xi
 **********/

#include "ngspice/ngspice.h"
#include "ngspice/ifsim.h"
#include "ngspice/cktdefs.h"
#include "ngspice/devdefs.h"
#include "bsim3def.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

int
BSIM3mAsk(
CKTcircuit *ckt,
GENmodel *inst,
int which,
IFvalue *value)
{
    BSIM3model *model = (BSIM3model *)inst;

    NG_IGNORE(ckt);

    switch(which) 
    {   case BSIM3_MOD_MOBMOD:
            value->iValue = model->BSIM3mobMod; 
            return(OK);
        case BSIM3_MOD_PARAMCHK:
            value->iValue = model->BSIM3paramChk; 
            return(OK);
        case BSIM3_MOD_BINUNIT:
            value->iValue = model->BSIM3binUnit; 
            return(OK);
        case BSIM3_MOD_CAPMOD:
            value->iValue = model->BSIM3capMod; 
            return(OK);
        case BSIM3_MOD_NOIMOD:
            value->iValue = model->BSIM3noiMod; 
            return(OK);
        case BSIM3_MOD_NQSMOD:
            value->iValue = model->BSIM3nqsMod;
            return(OK);
        case BSIM3_MOD_ACNQSMOD:
            value->iValue = model->BSIM3acnqsMod; 
            return(OK);
        case BSIM3_MOD_ACMMOD:
            value->iValue = model->BSIM3acmMod;
            return(OK);
        case BSIM3_MOD_CALCACM:
            value->iValue = model->BSIM3calcacm;
            return(OK);
        case  BSIM3_MOD_VERSION :
          value->sValue = model->BSIM3version;
            return(OK);
        case  BSIM3_MOD_TOX :
          value->rValue = model->BSIM3tox;
            return(OK);
        case  BSIM3_MOD_TOXM :
          value->rValue = model->BSIM3toxm;
            return(OK);
        case  BSIM3_MOD_CDSC :
          value->rValue = model->BSIM3cdsc;
            return(OK);
        case  BSIM3_MOD_CDSCB :
          value->rValue = model->BSIM3cdscb;
            return(OK);

        case  BSIM3_MOD_CDSCD :
          value->rValue = model->BSIM3cdscd;
            return(OK);

        case  BSIM3_MOD_CIT :
          value->rValue = model->BSIM3cit;
            return(OK);
        case  BSIM3_MOD_NFACTOR :
          value->rValue = model->BSIM3nfactor;
            return(OK);
        case BSIM3_MOD_XJ:
            value->rValue = model->BSIM3xj;
            return(OK);
        case BSIM3_MOD_VSAT:
            value->rValue = model->BSIM3vsat;
            return(OK);
        case BSIM3_MOD_AT:
            value->rValue = model->BSIM3at;
            return(OK);
        case BSIM3_MOD_A0:
            value->rValue = model->BSIM3a0;
            return(OK);

        case BSIM3_MOD_AGS:
            value->rValue = model->BSIM3ags;
            return(OK);

        case BSIM3_MOD_A1:
            value->rValue = model->BSIM3a1;
            return(OK);
        case BSIM3_MOD_A2:
            value->rValue = model->BSIM3a2;
            return(OK);
        case BSIM3_MOD_KETA:
            value->rValue = model->BSIM3keta;
            return(OK);   
        case BSIM3_MOD_NSUB:
            value->rValue = model->BSIM3nsub;
            return(OK);
        case BSIM3_MOD_NPEAK:
            value->rValue = model->BSIM3npeak;
            return(OK);
        case BSIM3_MOD_NGATE:
            value->rValue = model->BSIM3ngate;
            return(OK);
        case BSIM3_MOD_GAMMA1:
            value->rValue = model->BSIM3gamma1;
            return(OK);
        case BSIM3_MOD_GAMMA2:
            value->rValue = model->BSIM3gamma2;
            return(OK);
        case BSIM3_MOD_VBX:
            value->rValue = model->BSIM3vbx;
            return(OK);
        case BSIM3_MOD_VBM:
            value->rValue = model->BSIM3vbm;
            return(OK);
        case BSIM3_MOD_XT:
            value->rValue = model->BSIM3xt;
            return(OK);
        case  BSIM3_MOD_K1:
          value->rValue = model->BSIM3k1;
            return(OK);
        case  BSIM3_MOD_KT1:
          value->rValue = model->BSIM3kt1;
            return(OK);
        case  BSIM3_MOD_KT1L:
          value->rValue = model->BSIM3kt1l;
            return(OK);
        case  BSIM3_MOD_KT2 :
          value->rValue = model->BSIM3kt2;
            return(OK);
        case  BSIM3_MOD_K2 :
          value->rValue = model->BSIM3k2;
            return(OK);
        case  BSIM3_MOD_K3:
          value->rValue = model->BSIM3k3;
            return(OK);
        case  BSIM3_MOD_K3B:
          value->rValue = model->BSIM3k3b;
            return(OK);
        case  BSIM3_MOD_W0:
          value->rValue = model->BSIM3w0;
            return(OK);
        case  BSIM3_MOD_NLX:
          value->rValue = model->BSIM3nlx;
            return(OK);
        case  BSIM3_MOD_DVT0 :                
          value->rValue = model->BSIM3dvt0;
            return(OK);
        case  BSIM3_MOD_DVT1 :             
          value->rValue = model->BSIM3dvt1;
            return(OK);
        case  BSIM3_MOD_DVT2 :             
          value->rValue = model->BSIM3dvt2;
            return(OK);
        case  BSIM3_MOD_DVT0W :                
          value->rValue = model->BSIM3dvt0w;
            return(OK);
        case  BSIM3_MOD_DVT1W :             
          value->rValue = model->BSIM3dvt1w;
            return(OK);
        case  BSIM3_MOD_DVT2W :             
          value->rValue = model->BSIM3dvt2w;
            return(OK);
        case  BSIM3_MOD_DROUT :           
          value->rValue = model->BSIM3drout;
            return(OK);
        case  BSIM3_MOD_DSUB :           
          value->rValue = model->BSIM3dsub;
            return(OK);
        case BSIM3_MOD_VTH0:
            value->rValue = model->BSIM3vth0; 
            return(OK);
        case BSIM3_MOD_UA:
            value->rValue = model->BSIM3ua; 
            return(OK);
        case BSIM3_MOD_UA1:
            value->rValue = model->BSIM3ua1; 
            return(OK);
        case BSIM3_MOD_UB:
            value->rValue = model->BSIM3ub;  
            return(OK);
        case BSIM3_MOD_UB1:
            value->rValue = model->BSIM3ub1;  
            return(OK);
        case BSIM3_MOD_UC:
            value->rValue = model->BSIM3uc; 
            return(OK);
        case BSIM3_MOD_UC1:
            value->rValue = model->BSIM3uc1; 
            return(OK);
        case BSIM3_MOD_U0:
            value->rValue = model->BSIM3u0;
            return(OK);
        case BSIM3_MOD_UTE:
            value->rValue = model->BSIM3ute;
            return(OK);
        case BSIM3_MOD_VOFF:
            value->rValue = model->BSIM3voff;
            return(OK);
        case BSIM3_MOD_DELTA:
            value->rValue = model->BSIM3delta;
            return(OK);
        case BSIM3_MOD_RDSW:
            value->rValue = model->BSIM3rdsw; 
            return(OK);             
        case BSIM3_MOD_PRWG:
            value->rValue = model->BSIM3prwg; 
            return(OK);             
        case BSIM3_MOD_PRWB:
            value->rValue = model->BSIM3prwb; 
            return(OK);             
        case BSIM3_MOD_PRT:
            value->rValue = model->BSIM3prt; 
            return(OK);              
        case BSIM3_MOD_ETA0:
            value->rValue = model->BSIM3eta0; 
            return(OK);               
        case BSIM3_MOD_ETAB:
            value->rValue = model->BSIM3etab; 
            return(OK);               
        case BSIM3_MOD_PCLM:
            value->rValue = model->BSIM3pclm; 
            return(OK);               
        case BSIM3_MOD_PDIBL1:
            value->rValue = model->BSIM3pdibl1; 
            return(OK);               
        case BSIM3_MOD_PDIBL2:
            value->rValue = model->BSIM3pdibl2; 
            return(OK);               
        case BSIM3_MOD_PDIBLB:
            value->rValue = model->BSIM3pdiblb; 
            return(OK);               
        case BSIM3_MOD_PSCBE1:
            value->rValue = model->BSIM3pscbe1; 
            return(OK);               
        case BSIM3_MOD_PSCBE2:
            value->rValue = model->BSIM3pscbe2; 
            return(OK);               
        case BSIM3_MOD_PVAG:
            value->rValue = model->BSIM3pvag; 
            return(OK);               
        case BSIM3_MOD_WR:
            value->rValue = model->BSIM3wr;
            return(OK);
        case BSIM3_MOD_DWG:
            value->rValue = model->BSIM3dwg;
            return(OK);
        case BSIM3_MOD_DWB:
            value->rValue = model->BSIM3dwb;
            return(OK);
        case BSIM3_MOD_B0:
            value->rValue = model->BSIM3b0;
            return(OK);
        case BSIM3_MOD_B1:
            value->rValue = model->BSIM3b1;
            return(OK);
        case BSIM3_MOD_ALPHA0:
            value->rValue = model->BSIM3alpha0;
            return(OK);
        case BSIM3_MOD_ALPHA1:
            value->rValue = model->BSIM3alpha1;
            return(OK);
        case BSIM3_MOD_BETA0:
            value->rValue = model->BSIM3beta0;
            return(OK);
        case BSIM3_MOD_IJTH:
            value->rValue = model->BSIM3ijth;
            return(OK);
        case BSIM3_MOD_VFB:
            value->rValue = model->BSIM3vfb;
            return(OK);

        case BSIM3_MOD_ELM:
            value->rValue = model->BSIM3elm;
            return(OK);
        case BSIM3_MOD_CGSL:
            value->rValue = model->BSIM3cgsl;
            return(OK);
        case BSIM3_MOD_CGDL:
            value->rValue = model->BSIM3cgdl;
            return(OK);
        case BSIM3_MOD_CKAPPA:
            value->rValue = model->BSIM3ckappa;
            return(OK);
        case BSIM3_MOD_CF:
            value->rValue = model->BSIM3cf;
            return(OK);
        case BSIM3_MOD_CLC:
            value->rValue = model->BSIM3clc;
            return(OK);
        case BSIM3_MOD_CLE:
            value->rValue = model->BSIM3cle;
            return(OK);
        case BSIM3_MOD_DWC:
            value->rValue = model->BSIM3dwc;
            return(OK);
        case BSIM3_MOD_DLC:
            value->rValue = model->BSIM3dlc;
            return(OK);
        case BSIM3_MOD_VFBCV:
            value->rValue = model->BSIM3vfbcv; 
            return(OK);
        case BSIM3_MOD_ACDE:
            value->rValue = model->BSIM3acde;
            return(OK);
        case BSIM3_MOD_MOIN:
            value->rValue = model->BSIM3moin;
            return(OK);
        case BSIM3_MOD_NOFF:
            value->rValue = model->BSIM3noff;
            return(OK);
        case BSIM3_MOD_VOFFCV:
            value->rValue = model->BSIM3voffcv;
            return(OK);
        case BSIM3_MOD_TCJ:
            value->rValue = model->BSIM3tcj;
            return(OK);
        case BSIM3_MOD_TPB:
            value->rValue = model->BSIM3tpb;
            return(OK);
        case BSIM3_MOD_TCJSW:
            value->rValue = model->BSIM3tcjsw;
            return(OK);
        case BSIM3_MOD_TPBSW:
            value->rValue = model->BSIM3tpbsw;
            return(OK);
        case BSIM3_MOD_TCJSWG:
            value->rValue = model->BSIM3tcjswg;
            return(OK);
        case BSIM3_MOD_TPBSWG:
            value->rValue = model->BSIM3tpbswg;
            return(OK);

        /* ACM model */
        case BSIM3_MOD_HDIF:
            value->rValue = model->BSIM3hdif;
            return(OK);
        case BSIM3_MOD_LDIF:
            value->rValue = model->BSIM3ldif;
            return(OK);
        case BSIM3_MOD_LD:
            value->rValue = model->BSIM3ld;
            return(OK);
        case BSIM3_MOD_RD:
            value->rValue = model->BSIM3rd;
            return(OK);
        case BSIM3_MOD_RS:
            value->rValue = model->BSIM3rs;
            return(OK);
        case BSIM3_MOD_RDC:
            value->rValue = model->BSIM3rdc;
            return(OK);
        case BSIM3_MOD_RSC:
            value->rValue = model->BSIM3rsc;
            return(OK);
        case BSIM3_MOD_WMLT:
            value->rValue = model->BSIM3wmlt;
            return(OK);

	/* Length dependence */
        case  BSIM3_MOD_LCDSC :
          value->rValue = model->BSIM3lcdsc;
            return(OK);
        case  BSIM3_MOD_LCDSCB :
          value->rValue = model->BSIM3lcdscb;
            return(OK);
        case  BSIM3_MOD_LCDSCD :
          value->rValue = model->BSIM3lcdscd;
            return(OK);
        case  BSIM3_MOD_LCIT :
          value->rValue = model->BSIM3lcit;
            return(OK);
        case  BSIM3_MOD_LNFACTOR :
          value->rValue = model->BSIM3lnfactor;
            return(OK);
        case BSIM3_MOD_LXJ:
            value->rValue = model->BSIM3lxj;
            return(OK);
        case BSIM3_MOD_LVSAT:
            value->rValue = model->BSIM3lvsat;
            return(OK);
        case BSIM3_MOD_LAT:
            value->rValue = model->BSIM3lat;
            return(OK);
        case BSIM3_MOD_LA0:
            value->rValue = model->BSIM3la0;
            return(OK);
        case BSIM3_MOD_LAGS:
            value->rValue = model->BSIM3lags;
            return(OK);
        case BSIM3_MOD_LA1:
            value->rValue = model->BSIM3la1;
            return(OK);
        case BSIM3_MOD_LA2:
            value->rValue = model->BSIM3la2;
            return(OK);
        case BSIM3_MOD_LKETA:
            value->rValue = model->BSIM3lketa;
            return(OK);   
        case BSIM3_MOD_LNSUB:
            value->rValue = model->BSIM3lnsub;
            return(OK);
        case BSIM3_MOD_LNPEAK:
            value->rValue = model->BSIM3lnpeak;
            return(OK);
        case BSIM3_MOD_LNGATE:
            value->rValue = model->BSIM3lngate;
            return(OK);
        case BSIM3_MOD_LGAMMA1:
            value->rValue = model->BSIM3lgamma1;
            return(OK);
        case BSIM3_MOD_LGAMMA2:
            value->rValue = model->BSIM3lgamma2;
            return(OK);
        case BSIM3_MOD_LVBX:
            value->rValue = model->BSIM3lvbx;
            return(OK);
        case BSIM3_MOD_LVBM:
            value->rValue = model->BSIM3lvbm;
            return(OK);
        case BSIM3_MOD_LXT:
            value->rValue = model->BSIM3lxt;
            return(OK);
        case  BSIM3_MOD_LK1:
          value->rValue = model->BSIM3lk1;
            return(OK);
        case  BSIM3_MOD_LKT1:
          value->rValue = model->BSIM3lkt1;
            return(OK);
        case  BSIM3_MOD_LKT1L:
          value->rValue = model->BSIM3lkt1l;
            return(OK);
        case  BSIM3_MOD_LKT2 :
          value->rValue = model->BSIM3lkt2;
            return(OK);
        case  BSIM3_MOD_LK2 :
          value->rValue = model->BSIM3lk2;
            return(OK);
        case  BSIM3_MOD_LK3:
          value->rValue = model->BSIM3lk3;
            return(OK);
        case  BSIM3_MOD_LK3B:
          value->rValue = model->BSIM3lk3b;
            return(OK);
        case  BSIM3_MOD_LW0:
          value->rValue = model->BSIM3lw0;
            return(OK);
        case  BSIM3_MOD_LNLX:
          value->rValue = model->BSIM3lnlx;
            return(OK);
        case  BSIM3_MOD_LDVT0:                
          value->rValue = model->BSIM3ldvt0;
            return(OK);
        case  BSIM3_MOD_LDVT1 :             
          value->rValue = model->BSIM3ldvt1;
            return(OK);
        case  BSIM3_MOD_LDVT2 :             
          value->rValue = model->BSIM3ldvt2;
            return(OK);
        case  BSIM3_MOD_LDVT0W :                
          value->rValue = model->BSIM3ldvt0w;
            return(OK);
        case  BSIM3_MOD_LDVT1W :             
          value->rValue = model->BSIM3ldvt1w;
            return(OK);
        case  BSIM3_MOD_LDVT2W :             
          value->rValue = model->BSIM3ldvt2w;
            return(OK);
        case  BSIM3_MOD_LDROUT :           
          value->rValue = model->BSIM3ldrout;
            return(OK);
        case  BSIM3_MOD_LDSUB :           
          value->rValue = model->BSIM3ldsub;
            return(OK);
        case BSIM3_MOD_LVTH0:
            value->rValue = model->BSIM3lvth0; 
            return(OK);
        case BSIM3_MOD_LUA:
            value->rValue = model->BSIM3lua; 
            return(OK);
        case BSIM3_MOD_LUA1:
            value->rValue = model->BSIM3lua1; 
            return(OK);
        case BSIM3_MOD_LUB:
            value->rValue = model->BSIM3lub;  
            return(OK);
        case BSIM3_MOD_LUB1:
            value->rValue = model->BSIM3lub1;  
            return(OK);
        case BSIM3_MOD_LUC:
            value->rValue = model->BSIM3luc; 
            return(OK);
        case BSIM3_MOD_LUC1:
            value->rValue = model->BSIM3luc1; 
            return(OK);
        case BSIM3_MOD_LU0:
            value->rValue = model->BSIM3lu0;
            return(OK);
        case BSIM3_MOD_LUTE:
            value->rValue = model->BSIM3lute;
            return(OK);
        case BSIM3_MOD_LVOFF:
            value->rValue = model->BSIM3lvoff;
            return(OK);
        case BSIM3_MOD_LDELTA:
            value->rValue = model->BSIM3ldelta;
            return(OK);
        case BSIM3_MOD_LRDSW:
            value->rValue = model->BSIM3lrdsw; 
            return(OK);             
        case BSIM3_MOD_LPRWB:
            value->rValue = model->BSIM3lprwb; 
            return(OK);             
        case BSIM3_MOD_LPRWG:
            value->rValue = model->BSIM3lprwg; 
            return(OK);             
        case BSIM3_MOD_LPRT:
            value->rValue = model->BSIM3lprt; 
            return(OK);              
        case BSIM3_MOD_LETA0:
            value->rValue = model->BSIM3leta0; 
            return(OK);               
        case BSIM3_MOD_LETAB:
            value->rValue = model->BSIM3letab; 
            return(OK);               
        case BSIM3_MOD_LPCLM:
            value->rValue = model->BSIM3lpclm; 
            return(OK);               
        case BSIM3_MOD_LPDIBL1:
            value->rValue = model->BSIM3lpdibl1; 
            return(OK);               
        case BSIM3_MOD_LPDIBL2:
            value->rValue = model->BSIM3lpdibl2; 
            return(OK);               
        case BSIM3_MOD_LPDIBLB:
            value->rValue = model->BSIM3lpdiblb; 
            return(OK);               
        case BSIM3_MOD_LPSCBE1:
            value->rValue = model->BSIM3lpscbe1; 
            return(OK);               
        case BSIM3_MOD_LPSCBE2:
            value->rValue = model->BSIM3lpscbe2; 
            return(OK);               
        case BSIM3_MOD_LPVAG:
            value->rValue = model->BSIM3lpvag; 
            return(OK);               
        case BSIM3_MOD_LWR:
            value->rValue = model->BSIM3lwr;
            return(OK);
        case BSIM3_MOD_LDWG:
            value->rValue = model->BSIM3ldwg;
            return(OK);
        case BSIM3_MOD_LDWB:
            value->rValue = model->BSIM3ldwb;
            return(OK);
        case BSIM3_MOD_LB0:
            value->rValue = model->BSIM3lb0;
            return(OK);
        case BSIM3_MOD_LB1:
            value->rValue = model->BSIM3lb1;
            return(OK);
        case BSIM3_MOD_LALPHA0:
            value->rValue = model->BSIM3lalpha0;
            return(OK);
        case BSIM3_MOD_LALPHA1:
            value->rValue = model->BSIM3lalpha1;
            return(OK);
        case BSIM3_MOD_LBETA0:
            value->rValue = model->BSIM3lbeta0;
            return(OK);
        case BSIM3_MOD_LVFB:
            value->rValue = model->BSIM3lvfb;
            return(OK);

        case BSIM3_MOD_LELM:
            value->rValue = model->BSIM3lelm;
            return(OK);
        case BSIM3_MOD_LCGSL:
            value->rValue = model->BSIM3lcgsl;
            return(OK);
        case BSIM3_MOD_LCGDL:
            value->rValue = model->BSIM3lcgdl;
            return(OK);
        case BSIM3_MOD_LCKAPPA:
            value->rValue = model->BSIM3lckappa;
            return(OK);
        case BSIM3_MOD_LCF:
            value->rValue = model->BSIM3lcf;
            return(OK);
        case BSIM3_MOD_LCLC:
            value->rValue = model->BSIM3lclc;
            return(OK);
        case BSIM3_MOD_LCLE:
            value->rValue = model->BSIM3lcle;
            return(OK);
        case BSIM3_MOD_LVFBCV:
            value->rValue = model->BSIM3lvfbcv;
            return(OK);
        case BSIM3_MOD_LACDE:
            value->rValue = model->BSIM3lacde;
            return(OK);
        case BSIM3_MOD_LMOIN:
            value->rValue = model->BSIM3lmoin;
            return(OK);
        case BSIM3_MOD_LNOFF:
            value->rValue = model->BSIM3lnoff;
            return(OK);
        case BSIM3_MOD_LVOFFCV:
            value->rValue = model->BSIM3lvoffcv;
            return(OK);

	/* Width dependence */
        case  BSIM3_MOD_WCDSC :
          value->rValue = model->BSIM3wcdsc;
            return(OK);
        case  BSIM3_MOD_WCDSCB :
          value->rValue = model->BSIM3wcdscb;
            return(OK);
        case  BSIM3_MOD_WCDSCD :
          value->rValue = model->BSIM3wcdscd;
            return(OK);
        case  BSIM3_MOD_WCIT :
          value->rValue = model->BSIM3wcit;
            return(OK);
        case  BSIM3_MOD_WNFACTOR :
          value->rValue = model->BSIM3wnfactor;
            return(OK);
        case BSIM3_MOD_WXJ:
            value->rValue = model->BSIM3wxj;
            return(OK);
        case BSIM3_MOD_WVSAT:
            value->rValue = model->BSIM3wvsat;
            return(OK);
        case BSIM3_MOD_WAT:
            value->rValue = model->BSIM3wat;
            return(OK);
        case BSIM3_MOD_WA0:
            value->rValue = model->BSIM3wa0;
            return(OK);
        case BSIM3_MOD_WAGS:
            value->rValue = model->BSIM3wags;
            return(OK);
        case BSIM3_MOD_WA1:
            value->rValue = model->BSIM3wa1;
            return(OK);
        case BSIM3_MOD_WA2:
            value->rValue = model->BSIM3wa2;
            return(OK);
        case BSIM3_MOD_WKETA:
            value->rValue = model->BSIM3wketa;
            return(OK);   
        case BSIM3_MOD_WNSUB:
            value->rValue = model->BSIM3wnsub;
            return(OK);
        case BSIM3_MOD_WNPEAK:
            value->rValue = model->BSIM3wnpeak;
            return(OK);
        case BSIM3_MOD_WNGATE:
            value->rValue = model->BSIM3wngate;
            return(OK);
        case BSIM3_MOD_WGAMMA1:
            value->rValue = model->BSIM3wgamma1;
            return(OK);
        case BSIM3_MOD_WGAMMA2:
            value->rValue = model->BSIM3wgamma2;
            return(OK);
        case BSIM3_MOD_WVBX:
            value->rValue = model->BSIM3wvbx;
            return(OK);
        case BSIM3_MOD_WVBM:
            value->rValue = model->BSIM3wvbm;
            return(OK);
        case BSIM3_MOD_WXT:
            value->rValue = model->BSIM3wxt;
            return(OK);
        case  BSIM3_MOD_WK1:
          value->rValue = model->BSIM3wk1;
            return(OK);
        case  BSIM3_MOD_WKT1:
          value->rValue = model->BSIM3wkt1;
            return(OK);
        case  BSIM3_MOD_WKT1L:
          value->rValue = model->BSIM3wkt1l;
            return(OK);
        case  BSIM3_MOD_WKT2 :
          value->rValue = model->BSIM3wkt2;
            return(OK);
        case  BSIM3_MOD_WK2 :
          value->rValue = model->BSIM3wk2;
            return(OK);
        case  BSIM3_MOD_WK3:
          value->rValue = model->BSIM3wk3;
            return(OK);
        case  BSIM3_MOD_WK3B:
          value->rValue = model->BSIM3wk3b;
            return(OK);
        case  BSIM3_MOD_WW0:
          value->rValue = model->BSIM3ww0;
            return(OK);
        case  BSIM3_MOD_WNLX:
          value->rValue = model->BSIM3wnlx;
            return(OK);
        case  BSIM3_MOD_WDVT0:                
          value->rValue = model->BSIM3wdvt0;
            return(OK);
        case  BSIM3_MOD_WDVT1 :             
          value->rValue = model->BSIM3wdvt1;
            return(OK);
        case  BSIM3_MOD_WDVT2 :             
          value->rValue = model->BSIM3wdvt2;
            return(OK);
        case  BSIM3_MOD_WDVT0W :                
          value->rValue = model->BSIM3wdvt0w;
            return(OK);
        case  BSIM3_MOD_WDVT1W :             
          value->rValue = model->BSIM3wdvt1w;
            return(OK);
        case  BSIM3_MOD_WDVT2W :             
          value->rValue = model->BSIM3wdvt2w;
            return(OK);
        case  BSIM3_MOD_WDROUT :           
          value->rValue = model->BSIM3wdrout;
            return(OK);
        case  BSIM3_MOD_WDSUB :           
          value->rValue = model->BSIM3wdsub;
            return(OK);
        case BSIM3_MOD_WVTH0:
            value->rValue = model->BSIM3wvth0; 
            return(OK);
        case BSIM3_MOD_WUA:
            value->rValue = model->BSIM3wua; 
            return(OK);
        case BSIM3_MOD_WUA1:
            value->rValue = model->BSIM3wua1; 
            return(OK);
        case BSIM3_MOD_WUB:
            value->rValue = model->BSIM3wub;  
            return(OK);
        case BSIM3_MOD_WUB1:
            value->rValue = model->BSIM3wub1;  
            return(OK);
        case BSIM3_MOD_WUC:
            value->rValue = model->BSIM3wuc; 
            return(OK);
        case BSIM3_MOD_WUC1:
            value->rValue = model->BSIM3wuc1; 
            return(OK);
        case BSIM3_MOD_WU0:
            value->rValue = model->BSIM3wu0;
            return(OK);
        case BSIM3_MOD_WUTE:
            value->rValue = model->BSIM3wute;
            return(OK);
        case BSIM3_MOD_WVOFF:
            value->rValue = model->BSIM3wvoff;
            return(OK);
        case BSIM3_MOD_WDELTA:
            value->rValue = model->BSIM3wdelta;
            return(OK);
        case BSIM3_MOD_WRDSW:
            value->rValue = model->BSIM3wrdsw; 
            return(OK);             
        case BSIM3_MOD_WPRWB:
            value->rValue = model->BSIM3wprwb; 
            return(OK);             
        case BSIM3_MOD_WPRWG:
            value->rValue = model->BSIM3wprwg; 
            return(OK);             
        case BSIM3_MOD_WPRT:
            value->rValue = model->BSIM3wprt; 
            return(OK);              
        case BSIM3_MOD_WETA0:
            value->rValue = model->BSIM3weta0; 
            return(OK);               
        case BSIM3_MOD_WETAB:
            value->rValue = model->BSIM3wetab; 
            return(OK);               
        case BSIM3_MOD_WPCLM:
            value->rValue = model->BSIM3wpclm; 
            return(OK);               
        case BSIM3_MOD_WPDIBL1:
            value->rValue = model->BSIM3wpdibl1; 
            return(OK);               
        case BSIM3_MOD_WPDIBL2:
            value->rValue = model->BSIM3wpdibl2; 
            return(OK);               
        case BSIM3_MOD_WPDIBLB:
            value->rValue = model->BSIM3wpdiblb; 
            return(OK);               
        case BSIM3_MOD_WPSCBE1:
            value->rValue = model->BSIM3wpscbe1; 
            return(OK);               
        case BSIM3_MOD_WPSCBE2:
            value->rValue = model->BSIM3wpscbe2; 
            return(OK);               
        case BSIM3_MOD_WPVAG:
            value->rValue = model->BSIM3wpvag; 
            return(OK);               
        case BSIM3_MOD_WWR:
            value->rValue = model->BSIM3wwr;
            return(OK);
        case BSIM3_MOD_WDWG:
            value->rValue = model->BSIM3wdwg;
            return(OK);
        case BSIM3_MOD_WDWB:
            value->rValue = model->BSIM3wdwb;
            return(OK);
        case BSIM3_MOD_WB0:
            value->rValue = model->BSIM3wb0;
            return(OK);
        case BSIM3_MOD_WB1:
            value->rValue = model->BSIM3wb1;
            return(OK);
        case BSIM3_MOD_WALPHA0:
            value->rValue = model->BSIM3walpha0;
            return(OK);
        case BSIM3_MOD_WALPHA1:
            value->rValue = model->BSIM3walpha1;
            return(OK);
        case BSIM3_MOD_WBETA0:
            value->rValue = model->BSIM3wbeta0;
            return(OK);
        case BSIM3_MOD_WVFB:
            value->rValue = model->BSIM3wvfb;
            return(OK);

        case BSIM3_MOD_WELM:
            value->rValue = model->BSIM3welm;
            return(OK);
        case BSIM3_MOD_WCGSL:
            value->rValue = model->BSIM3wcgsl;
            return(OK);
        case BSIM3_MOD_WCGDL:
            value->rValue = model->BSIM3wcgdl;
            return(OK);
        case BSIM3_MOD_WCKAPPA:
            value->rValue = model->BSIM3wckappa;
            return(OK);
        case BSIM3_MOD_WCF:
            value->rValue = model->BSIM3wcf;
            return(OK);
        case BSIM3_MOD_WCLC:
            value->rValue = model->BSIM3wclc;
            return(OK);
        case BSIM3_MOD_WCLE:
            value->rValue = model->BSIM3wcle;
            return(OK);
        case BSIM3_MOD_WVFBCV:
            value->rValue = model->BSIM3wvfbcv;
            return(OK);
        case BSIM3_MOD_WACDE:
            value->rValue = model->BSIM3wacde;
            return(OK);
        case BSIM3_MOD_WMOIN:
            value->rValue = model->BSIM3wmoin;
            return(OK);
        case BSIM3_MOD_WNOFF:
            value->rValue = model->BSIM3wnoff;
            return(OK);
        case BSIM3_MOD_WVOFFCV:
            value->rValue = model->BSIM3wvoffcv;
            return(OK);

	/* Cross-term dependence */
        case  BSIM3_MOD_PCDSC :
          value->rValue = model->BSIM3pcdsc;
            return(OK);
        case  BSIM3_MOD_PCDSCB :
          value->rValue = model->BSIM3pcdscb;
            return(OK);
        case  BSIM3_MOD_PCDSCD :
          value->rValue = model->BSIM3pcdscd;
            return(OK);
         case  BSIM3_MOD_PCIT :
          value->rValue = model->BSIM3pcit;
            return(OK);
        case  BSIM3_MOD_PNFACTOR :
          value->rValue = model->BSIM3pnfactor;
            return(OK);
        case BSIM3_MOD_PXJ:
            value->rValue = model->BSIM3pxj;
            return(OK);
        case BSIM3_MOD_PVSAT:
            value->rValue = model->BSIM3pvsat;
            return(OK);
        case BSIM3_MOD_PAT:
            value->rValue = model->BSIM3pat;
            return(OK);
        case BSIM3_MOD_PA0:
            value->rValue = model->BSIM3pa0;
            return(OK);
        case BSIM3_MOD_PAGS:
            value->rValue = model->BSIM3pags;
            return(OK);
        case BSIM3_MOD_PA1:
            value->rValue = model->BSIM3pa1;
            return(OK);
        case BSIM3_MOD_PA2:
            value->rValue = model->BSIM3pa2;
            return(OK);
        case BSIM3_MOD_PKETA:
            value->rValue = model->BSIM3pketa;
            return(OK);   
        case BSIM3_MOD_PNSUB:
            value->rValue = model->BSIM3pnsub;
            return(OK);
        case BSIM3_MOD_PNPEAK:
            value->rValue = model->BSIM3pnpeak;
            return(OK);
        case BSIM3_MOD_PNGATE:
            value->rValue = model->BSIM3pngate;
            return(OK);
        case BSIM3_MOD_PGAMMA1:
            value->rValue = model->BSIM3pgamma1;
            return(OK);
        case BSIM3_MOD_PGAMMA2:
            value->rValue = model->BSIM3pgamma2;
            return(OK);
        case BSIM3_MOD_PVBX:
            value->rValue = model->BSIM3pvbx;
            return(OK);
        case BSIM3_MOD_PVBM:
            value->rValue = model->BSIM3pvbm;
            return(OK);
        case BSIM3_MOD_PXT:
            value->rValue = model->BSIM3pxt;
            return(OK);
        case  BSIM3_MOD_PK1:
          value->rValue = model->BSIM3pk1;
            return(OK);
        case  BSIM3_MOD_PKT1:
          value->rValue = model->BSIM3pkt1;
            return(OK);
        case  BSIM3_MOD_PKT1L:
          value->rValue = model->BSIM3pkt1l;
            return(OK);
        case  BSIM3_MOD_PKT2 :
          value->rValue = model->BSIM3pkt2;
            return(OK);
        case  BSIM3_MOD_PK2 :
          value->rValue = model->BSIM3pk2;
            return(OK);
        case  BSIM3_MOD_PK3:
          value->rValue = model->BSIM3pk3;
            return(OK);
        case  BSIM3_MOD_PK3B:
          value->rValue = model->BSIM3pk3b;
            return(OK);
        case  BSIM3_MOD_PW0:
          value->rValue = model->BSIM3pw0;
            return(OK);
        case  BSIM3_MOD_PNLX:
          value->rValue = model->BSIM3pnlx;
            return(OK);
        case  BSIM3_MOD_PDVT0 :                
          value->rValue = model->BSIM3pdvt0;
            return(OK);
        case  BSIM3_MOD_PDVT1 :             
          value->rValue = model->BSIM3pdvt1;
            return(OK);
        case  BSIM3_MOD_PDVT2 :             
          value->rValue = model->BSIM3pdvt2;
            return(OK);
        case  BSIM3_MOD_PDVT0W :                
          value->rValue = model->BSIM3pdvt0w;
            return(OK);
        case  BSIM3_MOD_PDVT1W :             
          value->rValue = model->BSIM3pdvt1w;
            return(OK);
        case  BSIM3_MOD_PDVT2W :             
          value->rValue = model->BSIM3pdvt2w;
            return(OK);
        case  BSIM3_MOD_PDROUT :           
          value->rValue = model->BSIM3pdrout;
            return(OK);
        case  BSIM3_MOD_PDSUB :           
          value->rValue = model->BSIM3pdsub;
            return(OK);
        case BSIM3_MOD_PVTH0:
            value->rValue = model->BSIM3pvth0; 
            return(OK);
        case BSIM3_MOD_PUA:
            value->rValue = model->BSIM3pua; 
            return(OK);
        case BSIM3_MOD_PUA1:
            value->rValue = model->BSIM3pua1; 
            return(OK);
        case BSIM3_MOD_PUB:
            value->rValue = model->BSIM3pub;  
            return(OK);
        case BSIM3_MOD_PUB1:
            value->rValue = model->BSIM3pub1;  
            return(OK);
        case BSIM3_MOD_PUC:
            value->rValue = model->BSIM3puc; 
            return(OK);
        case BSIM3_MOD_PUC1:
            value->rValue = model->BSIM3puc1; 
            return(OK);
        case BSIM3_MOD_PU0:
            value->rValue = model->BSIM3pu0;
            return(OK);
        case BSIM3_MOD_PUTE:
            value->rValue = model->BSIM3pute;
            return(OK);
        case BSIM3_MOD_PVOFF:
            value->rValue = model->BSIM3pvoff;
            return(OK);
        case BSIM3_MOD_PDELTA:
            value->rValue = model->BSIM3pdelta;
            return(OK);
        case BSIM3_MOD_PRDSW:
            value->rValue = model->BSIM3prdsw; 
            return(OK);             
        case BSIM3_MOD_PPRWB:
            value->rValue = model->BSIM3pprwb; 
            return(OK);             
        case BSIM3_MOD_PPRWG:
            value->rValue = model->BSIM3pprwg; 
            return(OK);             
        case BSIM3_MOD_PPRT:
            value->rValue = model->BSIM3pprt; 
            return(OK);              
        case BSIM3_MOD_PETA0:
            value->rValue = model->BSIM3peta0; 
            return(OK);               
        case BSIM3_MOD_PETAB:
            value->rValue = model->BSIM3petab; 
            return(OK);               
        case BSIM3_MOD_PPCLM:
            value->rValue = model->BSIM3ppclm; 
            return(OK);               
        case BSIM3_MOD_PPDIBL1:
            value->rValue = model->BSIM3ppdibl1; 
            return(OK);               
        case BSIM3_MOD_PPDIBL2:
            value->rValue = model->BSIM3ppdibl2; 
            return(OK);               
        case BSIM3_MOD_PPDIBLB:
            value->rValue = model->BSIM3ppdiblb; 
            return(OK);               
        case BSIM3_MOD_PPSCBE1:
            value->rValue = model->BSIM3ppscbe1; 
            return(OK);               
        case BSIM3_MOD_PPSCBE2:
            value->rValue = model->BSIM3ppscbe2; 
            return(OK);               
        case BSIM3_MOD_PPVAG:
            value->rValue = model->BSIM3ppvag; 
            return(OK);               
        case BSIM3_MOD_PWR:
            value->rValue = model->BSIM3pwr;
            return(OK);
        case BSIM3_MOD_PDWG:
            value->rValue = model->BSIM3pdwg;
            return(OK);
        case BSIM3_MOD_PDWB:
            value->rValue = model->BSIM3pdwb;
            return(OK);
        case BSIM3_MOD_PB0:
            value->rValue = model->BSIM3pb0;
            return(OK);
        case BSIM3_MOD_PB1:
            value->rValue = model->BSIM3pb1;
            return(OK);
        case BSIM3_MOD_PALPHA0:
            value->rValue = model->BSIM3palpha0;
            return(OK);
        case BSIM3_MOD_PALPHA1:
            value->rValue = model->BSIM3palpha1;
            return(OK);
        case BSIM3_MOD_PBETA0:
            value->rValue = model->BSIM3pbeta0;
            return(OK);
        case BSIM3_MOD_PVFB:
            value->rValue = model->BSIM3pvfb;
            return(OK);

        case BSIM3_MOD_PELM:
            value->rValue = model->BSIM3pelm;
            return(OK);
        case BSIM3_MOD_PCGSL:
            value->rValue = model->BSIM3pcgsl;
            return(OK);
        case BSIM3_MOD_PCGDL:
            value->rValue = model->BSIM3pcgdl;
            return(OK);
        case BSIM3_MOD_PCKAPPA:
            value->rValue = model->BSIM3pckappa;
            return(OK);
        case BSIM3_MOD_PCF:
            value->rValue = model->BSIM3pcf;
            return(OK);
        case BSIM3_MOD_PCLC:
            value->rValue = model->BSIM3pclc;
            return(OK);
        case BSIM3_MOD_PCLE:
            value->rValue = model->BSIM3pcle;
            return(OK);
        case BSIM3_MOD_PVFBCV:
            value->rValue = model->BSIM3pvfbcv;
            return(OK);
        case BSIM3_MOD_PACDE:
            value->rValue = model->BSIM3pacde;
            return(OK);
        case BSIM3_MOD_PMOIN:
            value->rValue = model->BSIM3pmoin;
            return(OK);
        case BSIM3_MOD_PNOFF:
            value->rValue = model->BSIM3pnoff;
            return(OK);
        case BSIM3_MOD_PVOFFCV:
            value->rValue = model->BSIM3pvoffcv;
            return(OK);

        case  BSIM3_MOD_TNOM :
          value->rValue = model->BSIM3tnom;
            return(OK);
        case BSIM3_MOD_CGSO:
            value->rValue = model->BSIM3cgso; 
            return(OK);
        case BSIM3_MOD_CGDO:
            value->rValue = model->BSIM3cgdo; 
            return(OK);
        case BSIM3_MOD_CGBO:
            value->rValue = model->BSIM3cgbo; 
            return(OK);
        case BSIM3_MOD_XPART:
            value->rValue = model->BSIM3xpart; 
            return(OK);
        case BSIM3_MOD_RSH:
            value->rValue = model->BSIM3sheetResistance; 
            return(OK);
        case BSIM3_MOD_JS:
            value->rValue = model->BSIM3jctSatCurDensity; 
            return(OK);
        case BSIM3_MOD_JSW:
            value->rValue = model->BSIM3jctSidewallSatCurDensity; 
            return(OK);
        case BSIM3_MOD_PB:
            value->rValue = model->BSIM3bulkJctPotential; 
            return(OK);
        case BSIM3_MOD_MJ:
            value->rValue = model->BSIM3bulkJctBotGradingCoeff; 
            return(OK);
        case BSIM3_MOD_PBSW:
            value->rValue = model->BSIM3sidewallJctPotential; 
            return(OK);
        case BSIM3_MOD_MJSW:
            value->rValue = model->BSIM3bulkJctSideGradingCoeff; 
            return(OK);
        case BSIM3_MOD_CJ:
            value->rValue = model->BSIM3unitAreaJctCap; 
            return(OK);
        case BSIM3_MOD_CJSW:
            value->rValue = model->BSIM3unitLengthSidewallJctCap; 
            return(OK);
        case BSIM3_MOD_PBSWG:
            value->rValue = model->BSIM3GatesidewallJctPotential; 
            return(OK);
        case BSIM3_MOD_MJSWG:
            value->rValue = model->BSIM3bulkJctGateSideGradingCoeff; 
            return(OK);
        case BSIM3_MOD_CJSWG:
            value->rValue = model->BSIM3unitLengthGateSidewallJctCap; 
            return(OK);
        case BSIM3_MOD_NJ:
            value->rValue = model->BSIM3jctEmissionCoeff; 
            return(OK);
        case BSIM3_MOD_XTI:
            value->rValue = model->BSIM3jctTempExponent; 
            return(OK);
        case BSIM3_MOD_LINTNOI:
            value->rValue = model->BSIM3lintnoi; 
            return(OK);
        case BSIM3_MOD_LINT:
            value->rValue = model->BSIM3Lint; 
            return(OK);
        case BSIM3_MOD_LL:
            value->rValue = model->BSIM3Ll;
            return(OK);
        case BSIM3_MOD_LLC:
            value->rValue = model->BSIM3Llc;
            return(OK);
        case BSIM3_MOD_LLN:
            value->rValue = model->BSIM3Lln;
            return(OK);
        case BSIM3_MOD_LW:
            value->rValue = model->BSIM3Lw;
            return(OK);
        case BSIM3_MOD_LWC:
            value->rValue = model->BSIM3Lwc;
            return(OK);
        case BSIM3_MOD_LWN:
            value->rValue = model->BSIM3Lwn;
            return(OK);
        case BSIM3_MOD_LWL:
            value->rValue = model->BSIM3Lwl;
            return(OK);
        case BSIM3_MOD_LWLC:
            value->rValue = model->BSIM3Lwlc;
            return(OK);
        case BSIM3_MOD_LMIN:
            value->rValue = model->BSIM3Lmin;
            return(OK);
        case BSIM3_MOD_LMAX:
            value->rValue = model->BSIM3Lmax;
            return(OK);
        case BSIM3_MOD_WINT:
            value->rValue = model->BSIM3Wint;
            return(OK);
        case BSIM3_MOD_WL:
            value->rValue = model->BSIM3Wl;
            return(OK);
        case BSIM3_MOD_WLC:
            value->rValue = model->BSIM3Wlc;
            return(OK);
        case BSIM3_MOD_WLN:
            value->rValue = model->BSIM3Wln;
            return(OK);
        case BSIM3_MOD_WW:
            value->rValue = model->BSIM3Ww;
            return(OK);
        case BSIM3_MOD_WWC:
            value->rValue = model->BSIM3Wwc;
            return(OK);
        case BSIM3_MOD_WWN:
            value->rValue = model->BSIM3Wwn;
            return(OK);
        case BSIM3_MOD_WWL:
            value->rValue = model->BSIM3Wwl;
            return(OK);
        case BSIM3_MOD_WWLC:
            value->rValue = model->BSIM3Wwlc;
            return(OK);
        case BSIM3_MOD_WMIN:
            value->rValue = model->BSIM3Wmin;
            return(OK);
        case BSIM3_MOD_WMAX:
            value->rValue = model->BSIM3Wmax;
            return(OK);

        case BSIM3_MOD_XL:
            value->rValue = model->BSIM3xl;
            return(OK);
        case BSIM3_MOD_XW:
            value->rValue = model->BSIM3xw;
            return(OK);

        case BSIM3_MOD_NOIA:
            value->rValue = model->BSIM3oxideTrapDensityA;
            return(OK);
        case BSIM3_MOD_NOIB:
            value->rValue = model->BSIM3oxideTrapDensityB;
            return(OK);
        case BSIM3_MOD_NOIC:
            value->rValue = model->BSIM3oxideTrapDensityC;
            return(OK);
        case BSIM3_MOD_EM:
            value->rValue = model->BSIM3em;
            return(OK);
        case BSIM3_MOD_EF:
            value->rValue = model->BSIM3ef;
            return(OK);
        case BSIM3_MOD_AF:
            value->rValue = model->BSIM3af;
            return(OK);
        case BSIM3_MOD_KF:
            value->rValue = model->BSIM3kf;
            return(OK);

        case BSIM3_MOD_VGS_MAX:
            value->rValue = model->BSIM3vgsMax;
            return(OK);
        case BSIM3_MOD_VGD_MAX:
            value->rValue = model->BSIM3vgdMax;
            return(OK);
        case BSIM3_MOD_VGB_MAX:
            value->rValue = model->BSIM3vgbMax;
            return(OK);
        case BSIM3_MOD_VDS_MAX:
            value->rValue = model->BSIM3vdsMax;
            return(OK);
        case BSIM3_MOD_VBS_MAX:
            value->rValue = model->BSIM3vbsMax;
            return(OK);
        case BSIM3_MOD_VBD_MAX:
            value->rValue = model->BSIM3vbdMax;
            return(OK);
        case BSIM3_MOD_VGSR_MAX:
            value->rValue = model->BSIM3vgsrMax;
            return(OK);
        case BSIM3_MOD_VGDR_MAX:
            value->rValue = model->BSIM3vgdrMax;
            return(OK);
        case BSIM3_MOD_VGBR_MAX:
            value->rValue = model->BSIM3vgbrMax;
            return(OK);
        case BSIM3_MOD_VBSR_MAX:
            value->rValue = model->BSIM3vbsrMax;
            return(OK);
        case BSIM3_MOD_VBDR_MAX:
            value->rValue = model->BSIM3vbdrMax;
            return(OK);

        default:
            return(E_BADPARM);
    }
    /* NOTREACHED */
}



