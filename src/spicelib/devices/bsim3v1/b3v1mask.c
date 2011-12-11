/**********
 * Copyright 1990 Regents of the University of California. All rights reserved.
 * File: b3v1mask.c
 * Author: 1995 Min-Chie Jeng and Mansun Chan. 
 * Modified by Paolo Nenzi 2002
 **********/
 
/* 
 * Release Notes: 
 * BSIM3v3.1,   Released by yuhua  96/12/08
 */

#include "ngspice/ngspice.h"
#include "ngspice/ifsim.h"
#include "ngspice/cktdefs.h"
#include "ngspice/devdefs.h"
#include "bsim3v1def.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

int
BSIM3v1mAsk(CKTcircuit *ckt, GENmodel *inst, int which, IFvalue *value)
{
    BSIM3v1model *model = (BSIM3v1model *)inst;

    NG_IGNORE(ckt);

    switch(which) 
    {   case BSIM3v1_MOD_MOBMOD:
            value->iValue = model->BSIM3v1mobMod; 
            return(OK);
        case BSIM3v1_MOD_PARAMCHK:
            value->iValue = model->BSIM3v1paramChk; 
            return(OK);
        case BSIM3v1_MOD_BINUNIT:
            value->iValue = model->BSIM3v1binUnit; 
            return(OK);
        case BSIM3v1_MOD_CAPMOD:
            value->iValue = model->BSIM3v1capMod; 
            return(OK);
        case BSIM3v1_MOD_NQSMOD:
            value->iValue = model->BSIM3v1nqsMod; 
            return(OK);
        case BSIM3v1_MOD_NOIMOD:
            value->iValue = model->BSIM3v1noiMod; 
            return(OK);
        case  BSIM3v1_MOD_VERSION :
          value->rValue = model->BSIM3v1version;
            return(OK);
        case  BSIM3v1_MOD_TOX :
          value->rValue = model->BSIM3v1tox;
            return(OK);
        case  BSIM3v1_MOD_CDSC :
          value->rValue = model->BSIM3v1cdsc;
            return(OK);
        case  BSIM3v1_MOD_CDSCB :
          value->rValue = model->BSIM3v1cdscb;
            return(OK);

        case  BSIM3v1_MOD_CDSCD :
          value->rValue = model->BSIM3v1cdscd;
            return(OK);

        case  BSIM3v1_MOD_CIT :
          value->rValue = model->BSIM3v1cit;
            return(OK);
        case  BSIM3v1_MOD_NFACTOR :
          value->rValue = model->BSIM3v1nfactor;
            return(OK);
        case BSIM3v1_MOD_XJ:
            value->rValue = model->BSIM3v1xj;
            return(OK);
        case BSIM3v1_MOD_VSAT:
            value->rValue = model->BSIM3v1vsat;
            return(OK);
        case BSIM3v1_MOD_AT:
            value->rValue = model->BSIM3v1at;
            return(OK);
        case BSIM3v1_MOD_A0:
            value->rValue = model->BSIM3v1a0;
            return(OK);

        case BSIM3v1_MOD_AGS:
            value->rValue = model->BSIM3v1ags;
            return(OK);

        case BSIM3v1_MOD_A1:
            value->rValue = model->BSIM3v1a1;
            return(OK);
        case BSIM3v1_MOD_A2:
            value->rValue = model->BSIM3v1a2;
            return(OK);
        case BSIM3v1_MOD_KETA:
            value->rValue = model->BSIM3v1keta;
            return(OK);   
        case BSIM3v1_MOD_NSUB:
            value->rValue = model->BSIM3v1nsub;
            return(OK);
        case BSIM3v1_MOD_NPEAK:
            value->rValue = model->BSIM3v1npeak;
            return(OK);
        case BSIM3v1_MOD_NGATE:
            value->rValue = model->BSIM3v1ngate;
            return(OK);
        case BSIM3v1_MOD_GAMMA1:
            value->rValue = model->BSIM3v1gamma1;
            return(OK);
        case BSIM3v1_MOD_GAMMA2:
            value->rValue = model->BSIM3v1gamma2;
            return(OK);
        case BSIM3v1_MOD_VBX:
            value->rValue = model->BSIM3v1vbx;
            return(OK);
        case BSIM3v1_MOD_VBM:
            value->rValue = model->BSIM3v1vbm;
            return(OK);
        case BSIM3v1_MOD_XT:
            value->rValue = model->BSIM3v1xt;
            return(OK);
        case  BSIM3v1_MOD_K1:
          value->rValue = model->BSIM3v1k1;
            return(OK);
        case  BSIM3v1_MOD_KT1:
          value->rValue = model->BSIM3v1kt1;
            return(OK);
        case  BSIM3v1_MOD_KT1L:
          value->rValue = model->BSIM3v1kt1l;
            return(OK);
        case  BSIM3v1_MOD_KT2 :
          value->rValue = model->BSIM3v1kt2;
            return(OK);
        case  BSIM3v1_MOD_K2 :
          value->rValue = model->BSIM3v1k2;
            return(OK);
        case  BSIM3v1_MOD_K3:
          value->rValue = model->BSIM3v1k3;
            return(OK);
        case  BSIM3v1_MOD_K3B:
          value->rValue = model->BSIM3v1k3b;
            return(OK);
        case  BSIM3v1_MOD_W0:
          value->rValue = model->BSIM3v1w0;
            return(OK);
        case  BSIM3v1_MOD_NLX:
          value->rValue = model->BSIM3v1nlx;
            return(OK);
        case  BSIM3v1_MOD_DVT0 :                
          value->rValue = model->BSIM3v1dvt0;
            return(OK);
        case  BSIM3v1_MOD_DVT1 :             
          value->rValue = model->BSIM3v1dvt1;
            return(OK);
        case  BSIM3v1_MOD_DVT2 :             
          value->rValue = model->BSIM3v1dvt2;
            return(OK);
        case  BSIM3v1_MOD_DVT0W :                
          value->rValue = model->BSIM3v1dvt0w;
            return(OK);
        case  BSIM3v1_MOD_DVT1W :             
          value->rValue = model->BSIM3v1dvt1w;
            return(OK);
        case  BSIM3v1_MOD_DVT2W :             
          value->rValue = model->BSIM3v1dvt2w;
            return(OK);
        case  BSIM3v1_MOD_DROUT :           
          value->rValue = model->BSIM3v1drout;
            return(OK);
        case  BSIM3v1_MOD_DSUB :           
          value->rValue = model->BSIM3v1dsub;
            return(OK);
        case BSIM3v1_MOD_VTH0:
            value->rValue = model->BSIM3v1vth0; 
            return(OK);
        case BSIM3v1_MOD_UA:
            value->rValue = model->BSIM3v1ua; 
            return(OK);
        case BSIM3v1_MOD_UA1:
            value->rValue = model->BSIM3v1ua1; 
            return(OK);
        case BSIM3v1_MOD_UB:
            value->rValue = model->BSIM3v1ub;  
            return(OK);
        case BSIM3v1_MOD_UB1:
            value->rValue = model->BSIM3v1ub1;  
            return(OK);
        case BSIM3v1_MOD_UC:
            value->rValue = model->BSIM3v1uc; 
            return(OK);
        case BSIM3v1_MOD_UC1:
            value->rValue = model->BSIM3v1uc1; 
            return(OK);
        case BSIM3v1_MOD_U0:
            value->rValue = model->BSIM3v1u0;
            return(OK);
        case BSIM3v1_MOD_UTE:
            value->rValue = model->BSIM3v1ute;
            return(OK);
        case BSIM3v1_MOD_VOFF:
            value->rValue = model->BSIM3v1voff;
            return(OK);
        case BSIM3v1_MOD_DELTA:
            value->rValue = model->BSIM3v1delta;
            return(OK);
        case BSIM3v1_MOD_RDSW:
            value->rValue = model->BSIM3v1rdsw; 
            return(OK);             
        case BSIM3v1_MOD_PRWG:
            value->rValue = model->BSIM3v1prwg; 
            return(OK);             
        case BSIM3v1_MOD_PRWB:
            value->rValue = model->BSIM3v1prwb; 
            return(OK);             
        case BSIM3v1_MOD_PRT:
            value->rValue = model->BSIM3v1prt; 
            return(OK);              
        case BSIM3v1_MOD_ETA0:
            value->rValue = model->BSIM3v1eta0; 
            return(OK);               
        case BSIM3v1_MOD_ETAB:
            value->rValue = model->BSIM3v1etab; 
            return(OK);               
        case BSIM3v1_MOD_PCLM:
            value->rValue = model->BSIM3v1pclm; 
            return(OK);               
        case BSIM3v1_MOD_PDIBL1:
            value->rValue = model->BSIM3v1pdibl1; 
            return(OK);               
        case BSIM3v1_MOD_PDIBL2:
            value->rValue = model->BSIM3v1pdibl2; 
            return(OK);               
        case BSIM3v1_MOD_PDIBLB:
            value->rValue = model->BSIM3v1pdiblb; 
            return(OK);               
        case BSIM3v1_MOD_PSCBE1:
            value->rValue = model->BSIM3v1pscbe1; 
            return(OK);               
        case BSIM3v1_MOD_PSCBE2:
            value->rValue = model->BSIM3v1pscbe2; 
            return(OK);               
        case BSIM3v1_MOD_PVAG:
            value->rValue = model->BSIM3v1pvag; 
            return(OK);               
        case BSIM3v1_MOD_WR:
            value->rValue = model->BSIM3v1wr;
            return(OK);
        case BSIM3v1_MOD_DWG:
            value->rValue = model->BSIM3v1dwg;
            return(OK);
        case BSIM3v1_MOD_DWB:
            value->rValue = model->BSIM3v1dwb;
            return(OK);
        case BSIM3v1_MOD_B0:
            value->rValue = model->BSIM3v1b0;
            return(OK);
        case BSIM3v1_MOD_B1:
            value->rValue = model->BSIM3v1b1;
            return(OK);
        case BSIM3v1_MOD_ALPHA0:
            value->rValue = model->BSIM3v1alpha0;
            return(OK);
        case BSIM3v1_MOD_BETA0:
            value->rValue = model->BSIM3v1beta0;
            return(OK);

        case BSIM3v1_MOD_ELM:
            value->rValue = model->BSIM3v1elm;
            return(OK);
        case BSIM3v1_MOD_CGSL:
            value->rValue = model->BSIM3v1cgsl;
            return(OK);
        case BSIM3v1_MOD_CGDL:
            value->rValue = model->BSIM3v1cgdl;
            return(OK);
        case BSIM3v1_MOD_CKAPPA:
            value->rValue = model->BSIM3v1ckappa;
            return(OK);
        case BSIM3v1_MOD_CF:
            value->rValue = model->BSIM3v1cf;
            return(OK);
        case BSIM3v1_MOD_CLC:
            value->rValue = model->BSIM3v1clc;
            return(OK);
        case BSIM3v1_MOD_CLE:
            value->rValue = model->BSIM3v1cle;
            return(OK);
        case BSIM3v1_MOD_DWC:
            value->rValue = model->BSIM3v1dwc;
            return(OK);
        case BSIM3v1_MOD_DLC:
            value->rValue = model->BSIM3v1dlc;
            return(OK);
        case BSIM3v1_MOD_VFBCV:
            value->rValue = model->BSIM3v1vfbcv; 
            return(OK);

	/* Length dependence */
        case  BSIM3v1_MOD_LCDSC :
          value->rValue = model->BSIM3v1lcdsc;
            return(OK);
        case  BSIM3v1_MOD_LCDSCB :
          value->rValue = model->BSIM3v1lcdscb;
            return(OK);
        case  BSIM3v1_MOD_LCDSCD :
          value->rValue = model->BSIM3v1lcdscd;
            return(OK);
        case  BSIM3v1_MOD_LCIT :
          value->rValue = model->BSIM3v1lcit;
            return(OK);
        case  BSIM3v1_MOD_LNFACTOR :
          value->rValue = model->BSIM3v1lnfactor;
            return(OK);
        case BSIM3v1_MOD_LXJ:
            value->rValue = model->BSIM3v1lxj;
            return(OK);
        case BSIM3v1_MOD_LVSAT:
            value->rValue = model->BSIM3v1lvsat;
            return(OK);
        case BSIM3v1_MOD_LAT:
            value->rValue = model->BSIM3v1lat;
            return(OK);
        case BSIM3v1_MOD_LA0:
            value->rValue = model->BSIM3v1la0;
            return(OK);
        case BSIM3v1_MOD_LAGS:
            value->rValue = model->BSIM3v1lags;
            return(OK);
        case BSIM3v1_MOD_LA1:
            value->rValue = model->BSIM3v1la1;
            return(OK);
        case BSIM3v1_MOD_LA2:
            value->rValue = model->BSIM3v1la2;
            return(OK);
        case BSIM3v1_MOD_LKETA:
            value->rValue = model->BSIM3v1lketa;
            return(OK);   
        case BSIM3v1_MOD_LNSUB:
            value->rValue = model->BSIM3v1lnsub;
            return(OK);
        case BSIM3v1_MOD_LNPEAK:
            value->rValue = model->BSIM3v1lnpeak;
            return(OK);
        case BSIM3v1_MOD_LNGATE:
            value->rValue = model->BSIM3v1lngate;
            return(OK);
        case BSIM3v1_MOD_LGAMMA1:
            value->rValue = model->BSIM3v1lgamma1;
            return(OK);
        case BSIM3v1_MOD_LGAMMA2:
            value->rValue = model->BSIM3v1lgamma2;
            return(OK);
        case BSIM3v1_MOD_LVBX:
            value->rValue = model->BSIM3v1lvbx;
            return(OK);
        case BSIM3v1_MOD_LVBM:
            value->rValue = model->BSIM3v1lvbm;
            return(OK);
        case BSIM3v1_MOD_LXT:
            value->rValue = model->BSIM3v1lxt;
            return(OK);
        case  BSIM3v1_MOD_LK1:
          value->rValue = model->BSIM3v1lk1;
            return(OK);
        case  BSIM3v1_MOD_LKT1:
          value->rValue = model->BSIM3v1lkt1;
            return(OK);
        case  BSIM3v1_MOD_LKT1L:
          value->rValue = model->BSIM3v1lkt1l;
            return(OK);
        case  BSIM3v1_MOD_LKT2 :
          value->rValue = model->BSIM3v1lkt2;
            return(OK);
        case  BSIM3v1_MOD_LK2 :
          value->rValue = model->BSIM3v1lk2;
            return(OK);
        case  BSIM3v1_MOD_LK3:
          value->rValue = model->BSIM3v1lk3;
            return(OK);
        case  BSIM3v1_MOD_LK3B:
          value->rValue = model->BSIM3v1lk3b;
            return(OK);
        case  BSIM3v1_MOD_LW0:
          value->rValue = model->BSIM3v1lw0;
            return(OK);
        case  BSIM3v1_MOD_LNLX:
          value->rValue = model->BSIM3v1lnlx;
            return(OK);
        case  BSIM3v1_MOD_LDVT0:                
          value->rValue = model->BSIM3v1ldvt0;
            return(OK);
        case  BSIM3v1_MOD_LDVT1 :             
          value->rValue = model->BSIM3v1ldvt1;
            return(OK);
        case  BSIM3v1_MOD_LDVT2 :             
          value->rValue = model->BSIM3v1ldvt2;
            return(OK);
        case  BSIM3v1_MOD_LDVT0W :                
          value->rValue = model->BSIM3v1ldvt0w;
            return(OK);
        case  BSIM3v1_MOD_LDVT1W :             
          value->rValue = model->BSIM3v1ldvt1w;
            return(OK);
        case  BSIM3v1_MOD_LDVT2W :             
          value->rValue = model->BSIM3v1ldvt2w;
            return(OK);
        case  BSIM3v1_MOD_LDROUT :           
          value->rValue = model->BSIM3v1ldrout;
            return(OK);
        case  BSIM3v1_MOD_LDSUB :           
          value->rValue = model->BSIM3v1ldsub;
            return(OK);
        case BSIM3v1_MOD_LVTH0:
            value->rValue = model->BSIM3v1lvth0; 
            return(OK);
        case BSIM3v1_MOD_LUA:
            value->rValue = model->BSIM3v1lua; 
            return(OK);
        case BSIM3v1_MOD_LUA1:
            value->rValue = model->BSIM3v1lua1; 
            return(OK);
        case BSIM3v1_MOD_LUB:
            value->rValue = model->BSIM3v1lub;  
            return(OK);
        case BSIM3v1_MOD_LUB1:
            value->rValue = model->BSIM3v1lub1;  
            return(OK);
        case BSIM3v1_MOD_LUC:
            value->rValue = model->BSIM3v1luc; 
            return(OK);
        case BSIM3v1_MOD_LUC1:
            value->rValue = model->BSIM3v1luc1; 
            return(OK);
        case BSIM3v1_MOD_LU0:
            value->rValue = model->BSIM3v1lu0;
            return(OK);
        case BSIM3v1_MOD_LUTE:
            value->rValue = model->BSIM3v1lute;
            return(OK);
        case BSIM3v1_MOD_LVOFF:
            value->rValue = model->BSIM3v1lvoff;
            return(OK);
        case BSIM3v1_MOD_LDELTA:
            value->rValue = model->BSIM3v1ldelta;
            return(OK);
        case BSIM3v1_MOD_LRDSW:
            value->rValue = model->BSIM3v1lrdsw; 
            return(OK);             
        case BSIM3v1_MOD_LPRWB:
            value->rValue = model->BSIM3v1lprwb; 
            return(OK);             
        case BSIM3v1_MOD_LPRWG:
            value->rValue = model->BSIM3v1lprwg; 
            return(OK);             
        case BSIM3v1_MOD_LPRT:
            value->rValue = model->BSIM3v1lprt; 
            return(OK);              
        case BSIM3v1_MOD_LETA0:
            value->rValue = model->BSIM3v1leta0; 
            return(OK);               
        case BSIM3v1_MOD_LETAB:
            value->rValue = model->BSIM3v1letab; 
            return(OK);               
        case BSIM3v1_MOD_LPCLM:
            value->rValue = model->BSIM3v1lpclm; 
            return(OK);               
        case BSIM3v1_MOD_LPDIBL1:
            value->rValue = model->BSIM3v1lpdibl1; 
            return(OK);               
        case BSIM3v1_MOD_LPDIBL2:
            value->rValue = model->BSIM3v1lpdibl2; 
            return(OK);               
        case BSIM3v1_MOD_LPDIBLB:
            value->rValue = model->BSIM3v1lpdiblb; 
            return(OK);               
        case BSIM3v1_MOD_LPSCBE1:
            value->rValue = model->BSIM3v1lpscbe1; 
            return(OK);               
        case BSIM3v1_MOD_LPSCBE2:
            value->rValue = model->BSIM3v1lpscbe2; 
            return(OK);               
        case BSIM3v1_MOD_LPVAG:
            value->rValue = model->BSIM3v1lpvag; 
            return(OK);               
        case BSIM3v1_MOD_LWR:
            value->rValue = model->BSIM3v1lwr;
            return(OK);
        case BSIM3v1_MOD_LDWG:
            value->rValue = model->BSIM3v1ldwg;
            return(OK);
        case BSIM3v1_MOD_LDWB:
            value->rValue = model->BSIM3v1ldwb;
            return(OK);
        case BSIM3v1_MOD_LB0:
            value->rValue = model->BSIM3v1lb0;
            return(OK);
        case BSIM3v1_MOD_LB1:
            value->rValue = model->BSIM3v1lb1;
            return(OK);
        case BSIM3v1_MOD_LALPHA0:
            value->rValue = model->BSIM3v1lalpha0;
            return(OK);
        case BSIM3v1_MOD_LBETA0:
            value->rValue = model->BSIM3v1lbeta0;
            return(OK);

        case BSIM3v1_MOD_LELM:
            value->rValue = model->BSIM3v1lelm;
            return(OK);
        case BSIM3v1_MOD_LCGSL:
            value->rValue = model->BSIM3v1lcgsl;
            return(OK);
        case BSIM3v1_MOD_LCGDL:
            value->rValue = model->BSIM3v1lcgdl;
            return(OK);
        case BSIM3v1_MOD_LCKAPPA:
            value->rValue = model->BSIM3v1lckappa;
            return(OK);
        case BSIM3v1_MOD_LCF:
            value->rValue = model->BSIM3v1lcf;
            return(OK);
        case BSIM3v1_MOD_LCLC:
            value->rValue = model->BSIM3v1lclc;
            return(OK);
        case BSIM3v1_MOD_LCLE:
            value->rValue = model->BSIM3v1lcle;
            return(OK);
        case BSIM3v1_MOD_LVFBCV:
            value->rValue = model->BSIM3v1lvfbcv;
            return(OK);

	/* Width dependence */
        case  BSIM3v1_MOD_WCDSC :
          value->rValue = model->BSIM3v1wcdsc;
            return(OK);
        case  BSIM3v1_MOD_WCDSCB :
          value->rValue = model->BSIM3v1wcdscb;
            return(OK);
        case  BSIM3v1_MOD_WCDSCD :
          value->rValue = model->BSIM3v1wcdscd;
            return(OK);
        case  BSIM3v1_MOD_WCIT :
          value->rValue = model->BSIM3v1wcit;
            return(OK);
        case  BSIM3v1_MOD_WNFACTOR :
          value->rValue = model->BSIM3v1wnfactor;
            return(OK);
        case BSIM3v1_MOD_WXJ:
            value->rValue = model->BSIM3v1wxj;
            return(OK);
        case BSIM3v1_MOD_WVSAT:
            value->rValue = model->BSIM3v1wvsat;
            return(OK);
        case BSIM3v1_MOD_WAT:
            value->rValue = model->BSIM3v1wat;
            return(OK);
        case BSIM3v1_MOD_WA0:
            value->rValue = model->BSIM3v1wa0;
            return(OK);
        case BSIM3v1_MOD_WAGS:
            value->rValue = model->BSIM3v1wags;
            return(OK);
        case BSIM3v1_MOD_WA1:
            value->rValue = model->BSIM3v1wa1;
            return(OK);
        case BSIM3v1_MOD_WA2:
            value->rValue = model->BSIM3v1wa2;
            return(OK);
        case BSIM3v1_MOD_WKETA:
            value->rValue = model->BSIM3v1wketa;
            return(OK);   
        case BSIM3v1_MOD_WNSUB:
            value->rValue = model->BSIM3v1wnsub;
            return(OK);
        case BSIM3v1_MOD_WNPEAK:
            value->rValue = model->BSIM3v1wnpeak;
            return(OK);
        case BSIM3v1_MOD_WNGATE:
            value->rValue = model->BSIM3v1wngate;
            return(OK);
        case BSIM3v1_MOD_WGAMMA1:
            value->rValue = model->BSIM3v1wgamma1;
            return(OK);
        case BSIM3v1_MOD_WGAMMA2:
            value->rValue = model->BSIM3v1wgamma2;
            return(OK);
        case BSIM3v1_MOD_WVBX:
            value->rValue = model->BSIM3v1wvbx;
            return(OK);
        case BSIM3v1_MOD_WVBM:
            value->rValue = model->BSIM3v1wvbm;
            return(OK);
        case BSIM3v1_MOD_WXT:
            value->rValue = model->BSIM3v1wxt;
            return(OK);
        case  BSIM3v1_MOD_WK1:
          value->rValue = model->BSIM3v1wk1;
            return(OK);
        case  BSIM3v1_MOD_WKT1:
          value->rValue = model->BSIM3v1wkt1;
            return(OK);
        case  BSIM3v1_MOD_WKT1L:
          value->rValue = model->BSIM3v1wkt1l;
            return(OK);
        case  BSIM3v1_MOD_WKT2 :
          value->rValue = model->BSIM3v1wkt2;
            return(OK);
        case  BSIM3v1_MOD_WK2 :
          value->rValue = model->BSIM3v1wk2;
            return(OK);
        case  BSIM3v1_MOD_WK3:
          value->rValue = model->BSIM3v1wk3;
            return(OK);
        case  BSIM3v1_MOD_WK3B:
          value->rValue = model->BSIM3v1wk3b;
            return(OK);
        case  BSIM3v1_MOD_WW0:
          value->rValue = model->BSIM3v1ww0;
            return(OK);
        case  BSIM3v1_MOD_WNLX:
          value->rValue = model->BSIM3v1wnlx;
            return(OK);
        case  BSIM3v1_MOD_WDVT0:                
          value->rValue = model->BSIM3v1wdvt0;
            return(OK);
        case  BSIM3v1_MOD_WDVT1 :             
          value->rValue = model->BSIM3v1wdvt1;
            return(OK);
        case  BSIM3v1_MOD_WDVT2 :             
          value->rValue = model->BSIM3v1wdvt2;
            return(OK);
        case  BSIM3v1_MOD_WDVT0W :                
          value->rValue = model->BSIM3v1wdvt0w;
            return(OK);
        case  BSIM3v1_MOD_WDVT1W :             
          value->rValue = model->BSIM3v1wdvt1w;
            return(OK);
        case  BSIM3v1_MOD_WDVT2W :             
          value->rValue = model->BSIM3v1wdvt2w;
            return(OK);
        case  BSIM3v1_MOD_WDROUT :           
          value->rValue = model->BSIM3v1wdrout;
            return(OK);
        case  BSIM3v1_MOD_WDSUB :           
          value->rValue = model->BSIM3v1wdsub;
            return(OK);
        case BSIM3v1_MOD_WVTH0:
            value->rValue = model->BSIM3v1wvth0; 
            return(OK);
        case BSIM3v1_MOD_WUA:
            value->rValue = model->BSIM3v1wua; 
            return(OK);
        case BSIM3v1_MOD_WUA1:
            value->rValue = model->BSIM3v1wua1; 
            return(OK);
        case BSIM3v1_MOD_WUB:
            value->rValue = model->BSIM3v1wub;  
            return(OK);
        case BSIM3v1_MOD_WUB1:
            value->rValue = model->BSIM3v1wub1;  
            return(OK);
        case BSIM3v1_MOD_WUC:
            value->rValue = model->BSIM3v1wuc; 
            return(OK);
        case BSIM3v1_MOD_WUC1:
            value->rValue = model->BSIM3v1wuc1; 
            return(OK);
        case BSIM3v1_MOD_WU0:
            value->rValue = model->BSIM3v1wu0;
            return(OK);
        case BSIM3v1_MOD_WUTE:
            value->rValue = model->BSIM3v1wute;
            return(OK);
        case BSIM3v1_MOD_WVOFF:
            value->rValue = model->BSIM3v1wvoff;
            return(OK);
        case BSIM3v1_MOD_WDELTA:
            value->rValue = model->BSIM3v1wdelta;
            return(OK);
        case BSIM3v1_MOD_WRDSW:
            value->rValue = model->BSIM3v1wrdsw; 
            return(OK);             
        case BSIM3v1_MOD_WPRWB:
            value->rValue = model->BSIM3v1wprwb; 
            return(OK);             
        case BSIM3v1_MOD_WPRWG:
            value->rValue = model->BSIM3v1wprwg; 
            return(OK);             
        case BSIM3v1_MOD_WPRT:
            value->rValue = model->BSIM3v1wprt; 
            return(OK);              
        case BSIM3v1_MOD_WETA0:
            value->rValue = model->BSIM3v1weta0; 
            return(OK);               
        case BSIM3v1_MOD_WETAB:
            value->rValue = model->BSIM3v1wetab; 
            return(OK);               
        case BSIM3v1_MOD_WPCLM:
            value->rValue = model->BSIM3v1wpclm; 
            return(OK);               
        case BSIM3v1_MOD_WPDIBL1:
            value->rValue = model->BSIM3v1wpdibl1; 
            return(OK);               
        case BSIM3v1_MOD_WPDIBL2:
            value->rValue = model->BSIM3v1wpdibl2; 
            return(OK);               
        case BSIM3v1_MOD_WPDIBLB:
            value->rValue = model->BSIM3v1wpdiblb; 
            return(OK);               
        case BSIM3v1_MOD_WPSCBE1:
            value->rValue = model->BSIM3v1wpscbe1; 
            return(OK);               
        case BSIM3v1_MOD_WPSCBE2:
            value->rValue = model->BSIM3v1wpscbe2; 
            return(OK);               
        case BSIM3v1_MOD_WPVAG:
            value->rValue = model->BSIM3v1wpvag; 
            return(OK);               
        case BSIM3v1_MOD_WWR:
            value->rValue = model->BSIM3v1wwr;
            return(OK);
        case BSIM3v1_MOD_WDWG:
            value->rValue = model->BSIM3v1wdwg;
            return(OK);
        case BSIM3v1_MOD_WDWB:
            value->rValue = model->BSIM3v1wdwb;
            return(OK);
        case BSIM3v1_MOD_WB0:
            value->rValue = model->BSIM3v1wb0;
            return(OK);
        case BSIM3v1_MOD_WB1:
            value->rValue = model->BSIM3v1wb1;
            return(OK);
        case BSIM3v1_MOD_WALPHA0:
            value->rValue = model->BSIM3v1walpha0;
            return(OK);
        case BSIM3v1_MOD_WBETA0:
            value->rValue = model->BSIM3v1wbeta0;
            return(OK);

        case BSIM3v1_MOD_WELM:
            value->rValue = model->BSIM3v1welm;
            return(OK);
        case BSIM3v1_MOD_WCGSL:
            value->rValue = model->BSIM3v1wcgsl;
            return(OK);
        case BSIM3v1_MOD_WCGDL:
            value->rValue = model->BSIM3v1wcgdl;
            return(OK);
        case BSIM3v1_MOD_WCKAPPA:
            value->rValue = model->BSIM3v1wckappa;
            return(OK);
        case BSIM3v1_MOD_WCF:
            value->rValue = model->BSIM3v1wcf;
            return(OK);
        case BSIM3v1_MOD_WCLC:
            value->rValue = model->BSIM3v1wclc;
            return(OK);
        case BSIM3v1_MOD_WCLE:
            value->rValue = model->BSIM3v1wcle;
            return(OK);
        case BSIM3v1_MOD_WVFBCV:
            value->rValue = model->BSIM3v1wvfbcv;
            return(OK);

	/* Cross-term dependence */
        case  BSIM3v1_MOD_PCDSC :
          value->rValue = model->BSIM3v1pcdsc;
            return(OK);
        case  BSIM3v1_MOD_PCDSCB :
          value->rValue = model->BSIM3v1pcdscb;
            return(OK);
        case  BSIM3v1_MOD_PCDSCD :
          value->rValue = model->BSIM3v1pcdscd;
            return(OK);
         case  BSIM3v1_MOD_PCIT :
          value->rValue = model->BSIM3v1pcit;
            return(OK);
        case  BSIM3v1_MOD_PNFACTOR :
          value->rValue = model->BSIM3v1pnfactor;
            return(OK);
        case BSIM3v1_MOD_PXJ:
            value->rValue = model->BSIM3v1pxj;
            return(OK);
        case BSIM3v1_MOD_PVSAT:
            value->rValue = model->BSIM3v1pvsat;
            return(OK);
        case BSIM3v1_MOD_PAT:
            value->rValue = model->BSIM3v1pat;
            return(OK);
        case BSIM3v1_MOD_PA0:
            value->rValue = model->BSIM3v1pa0;
            return(OK);
        case BSIM3v1_MOD_PAGS:
            value->rValue = model->BSIM3v1pags;
            return(OK);
        case BSIM3v1_MOD_PA1:
            value->rValue = model->BSIM3v1pa1;
            return(OK);
        case BSIM3v1_MOD_PA2:
            value->rValue = model->BSIM3v1pa2;
            return(OK);
        case BSIM3v1_MOD_PKETA:
            value->rValue = model->BSIM3v1pketa;
            return(OK);   
        case BSIM3v1_MOD_PNSUB:
            value->rValue = model->BSIM3v1pnsub;
            return(OK);
        case BSIM3v1_MOD_PNPEAK:
            value->rValue = model->BSIM3v1pnpeak;
            return(OK);
        case BSIM3v1_MOD_PNGATE:
            value->rValue = model->BSIM3v1pngate;
            return(OK);
        case BSIM3v1_MOD_PGAMMA1:
            value->rValue = model->BSIM3v1pgamma1;
            return(OK);
        case BSIM3v1_MOD_PGAMMA2:
            value->rValue = model->BSIM3v1pgamma2;
            return(OK);
        case BSIM3v1_MOD_PVBX:
            value->rValue = model->BSIM3v1pvbx;
            return(OK);
        case BSIM3v1_MOD_PVBM:
            value->rValue = model->BSIM3v1pvbm;
            return(OK);
        case BSIM3v1_MOD_PXT:
            value->rValue = model->BSIM3v1pxt;
            return(OK);
        case  BSIM3v1_MOD_PK1:
          value->rValue = model->BSIM3v1pk1;
            return(OK);
        case  BSIM3v1_MOD_PKT1:
          value->rValue = model->BSIM3v1pkt1;
            return(OK);
        case  BSIM3v1_MOD_PKT1L:
          value->rValue = model->BSIM3v1pkt1l;
            return(OK);
        case  BSIM3v1_MOD_PKT2 :
          value->rValue = model->BSIM3v1pkt2;
            return(OK);
        case  BSIM3v1_MOD_PK2 :
          value->rValue = model->BSIM3v1pk2;
            return(OK);
        case  BSIM3v1_MOD_PK3:
          value->rValue = model->BSIM3v1pk3;
            return(OK);
        case  BSIM3v1_MOD_PK3B:
          value->rValue = model->BSIM3v1pk3b;
            return(OK);
        case  BSIM3v1_MOD_PW0:
          value->rValue = model->BSIM3v1pw0;
            return(OK);
        case  BSIM3v1_MOD_PNLX:
          value->rValue = model->BSIM3v1pnlx;
            return(OK);
        case  BSIM3v1_MOD_PDVT0 :                
          value->rValue = model->BSIM3v1pdvt0;
            return(OK);
        case  BSIM3v1_MOD_PDVT1 :             
          value->rValue = model->BSIM3v1pdvt1;
            return(OK);
        case  BSIM3v1_MOD_PDVT2 :             
          value->rValue = model->BSIM3v1pdvt2;
            return(OK);
        case  BSIM3v1_MOD_PDVT0W :                
          value->rValue = model->BSIM3v1pdvt0w;
            return(OK);
        case  BSIM3v1_MOD_PDVT1W :             
          value->rValue = model->BSIM3v1pdvt1w;
            return(OK);
        case  BSIM3v1_MOD_PDVT2W :             
          value->rValue = model->BSIM3v1pdvt2w;
            return(OK);
        case  BSIM3v1_MOD_PDROUT :           
          value->rValue = model->BSIM3v1pdrout;
            return(OK);
        case  BSIM3v1_MOD_PDSUB :           
          value->rValue = model->BSIM3v1pdsub;
            return(OK);
        case BSIM3v1_MOD_PVTH0:
            value->rValue = model->BSIM3v1pvth0; 
            return(OK);
        case BSIM3v1_MOD_PUA:
            value->rValue = model->BSIM3v1pua; 
            return(OK);
        case BSIM3v1_MOD_PUA1:
            value->rValue = model->BSIM3v1pua1; 
            return(OK);
        case BSIM3v1_MOD_PUB:
            value->rValue = model->BSIM3v1pub;  
            return(OK);
        case BSIM3v1_MOD_PUB1:
            value->rValue = model->BSIM3v1pub1;  
            return(OK);
        case BSIM3v1_MOD_PUC:
            value->rValue = model->BSIM3v1puc; 
            return(OK);
        case BSIM3v1_MOD_PUC1:
            value->rValue = model->BSIM3v1puc1; 
            return(OK);
        case BSIM3v1_MOD_PU0:
            value->rValue = model->BSIM3v1pu0;
            return(OK);
        case BSIM3v1_MOD_PUTE:
            value->rValue = model->BSIM3v1pute;
            return(OK);
        case BSIM3v1_MOD_PVOFF:
            value->rValue = model->BSIM3v1pvoff;
            return(OK);
        case BSIM3v1_MOD_PDELTA:
            value->rValue = model->BSIM3v1pdelta;
            return(OK);
        case BSIM3v1_MOD_PRDSW:
            value->rValue = model->BSIM3v1prdsw; 
            return(OK);             
        case BSIM3v1_MOD_PPRWB:
            value->rValue = model->BSIM3v1pprwb; 
            return(OK);             
        case BSIM3v1_MOD_PPRWG:
            value->rValue = model->BSIM3v1pprwg; 
            return(OK);             
        case BSIM3v1_MOD_PPRT:
            value->rValue = model->BSIM3v1pprt; 
            return(OK);              
        case BSIM3v1_MOD_PETA0:
            value->rValue = model->BSIM3v1peta0; 
            return(OK);               
        case BSIM3v1_MOD_PETAB:
            value->rValue = model->BSIM3v1petab; 
            return(OK);               
        case BSIM3v1_MOD_PPCLM:
            value->rValue = model->BSIM3v1ppclm; 
            return(OK);               
        case BSIM3v1_MOD_PPDIBL1:
            value->rValue = model->BSIM3v1ppdibl1; 
            return(OK);               
        case BSIM3v1_MOD_PPDIBL2:
            value->rValue = model->BSIM3v1ppdibl2; 
            return(OK);               
        case BSIM3v1_MOD_PPDIBLB:
            value->rValue = model->BSIM3v1ppdiblb; 
            return(OK);               
        case BSIM3v1_MOD_PPSCBE1:
            value->rValue = model->BSIM3v1ppscbe1; 
            return(OK);               
        case BSIM3v1_MOD_PPSCBE2:
            value->rValue = model->BSIM3v1ppscbe2; 
            return(OK);               
        case BSIM3v1_MOD_PPVAG:
            value->rValue = model->BSIM3v1ppvag; 
            return(OK);               
        case BSIM3v1_MOD_PWR:
            value->rValue = model->BSIM3v1pwr;
            return(OK);
        case BSIM3v1_MOD_PDWG:
            value->rValue = model->BSIM3v1pdwg;
            return(OK);
        case BSIM3v1_MOD_PDWB:
            value->rValue = model->BSIM3v1pdwb;
            return(OK);
        case BSIM3v1_MOD_PB0:
            value->rValue = model->BSIM3v1pb0;
            return(OK);
        case BSIM3v1_MOD_PB1:
            value->rValue = model->BSIM3v1pb1;
            return(OK);
        case BSIM3v1_MOD_PALPHA0:
            value->rValue = model->BSIM3v1palpha0;
            return(OK);
        case BSIM3v1_MOD_PBETA0:
            value->rValue = model->BSIM3v1pbeta0;
            return(OK);

        case BSIM3v1_MOD_PELM:
            value->rValue = model->BSIM3v1pelm;
            return(OK);
        case BSIM3v1_MOD_PCGSL:
            value->rValue = model->BSIM3v1pcgsl;
            return(OK);
        case BSIM3v1_MOD_PCGDL:
            value->rValue = model->BSIM3v1pcgdl;
            return(OK);
        case BSIM3v1_MOD_PCKAPPA:
            value->rValue = model->BSIM3v1pckappa;
            return(OK);
        case BSIM3v1_MOD_PCF:
            value->rValue = model->BSIM3v1pcf;
            return(OK);
        case BSIM3v1_MOD_PCLC:
            value->rValue = model->BSIM3v1pclc;
            return(OK);
        case BSIM3v1_MOD_PCLE:
            value->rValue = model->BSIM3v1pcle;
            return(OK);
        case BSIM3v1_MOD_PVFBCV:
            value->rValue = model->BSIM3v1pvfbcv;
            return(OK);

        case  BSIM3v1_MOD_TNOM :
          value->rValue = model->BSIM3v1tnom;
            return(OK);
        case BSIM3v1_MOD_CGSO:
            value->rValue = model->BSIM3v1cgso; 
            return(OK);
        case BSIM3v1_MOD_CGDO:
            value->rValue = model->BSIM3v1cgdo; 
            return(OK);
        case BSIM3v1_MOD_CGBO:
            value->rValue = model->BSIM3v1cgbo; 
            return(OK);
        case BSIM3v1_MOD_XPART:
            value->rValue = model->BSIM3v1xpart; 
            return(OK);
        case BSIM3v1_MOD_RSH:
            value->rValue = model->BSIM3v1sheetResistance; 
            return(OK);
        case BSIM3v1_MOD_JS:
            value->rValue = model->BSIM3v1jctSatCurDensity; 
            return(OK);
        case BSIM3v1_MOD_JSW:
            value->rValue = model->BSIM3v1jctSidewallSatCurDensity; 
            return(OK);
        case BSIM3v1_MOD_PB:
            value->rValue = model->BSIM3v1bulkJctPotential; 
            return(OK);
        case BSIM3v1_MOD_MJ:
            value->rValue = model->BSIM3v1bulkJctBotGradingCoeff; 
            return(OK);
        case BSIM3v1_MOD_PBSW:
            value->rValue = model->BSIM3v1sidewallJctPotential; 
            return(OK);
        case BSIM3v1_MOD_MJSW:
            value->rValue = model->BSIM3v1bulkJctSideGradingCoeff; 
            return(OK);
        case BSIM3v1_MOD_CJ:
            value->rValue = model->BSIM3v1unitAreaJctCap; 
            return(OK);
        case BSIM3v1_MOD_CJSW:
            value->rValue = model->BSIM3v1unitLengthSidewallJctCap; 
            return(OK);
        case BSIM3v1_MOD_PBSWG:
            value->rValue = model->BSIM3v1GatesidewallJctPotential; 
            return(OK);
        case BSIM3v1_MOD_MJSWG:
            value->rValue = model->BSIM3v1bulkJctGateSideGradingCoeff; 
            return(OK);
        case BSIM3v1_MOD_CJSWG:
            value->rValue = model->BSIM3v1unitLengthGateSidewallJctCap; 
            return(OK);
        case BSIM3v1_MOD_NJ:
            value->rValue = model->BSIM3v1jctEmissionCoeff; 
            return(OK);
        case BSIM3v1_MOD_XTI:
            value->rValue = model->BSIM3v1jctTempExponent; 
            return(OK);
        case BSIM3v1_MOD_LINT:
            value->rValue = model->BSIM3v1Lint; 
            return(OK);
        case BSIM3v1_MOD_LL:
            value->rValue = model->BSIM3v1Ll;
            return(OK);
        case BSIM3v1_MOD_LLN:
            value->rValue = model->BSIM3v1Lln;
            return(OK);
        case BSIM3v1_MOD_LW:
            value->rValue = model->BSIM3v1Lw;
            return(OK);
        case BSIM3v1_MOD_LWN:
            value->rValue = model->BSIM3v1Lwn;
            return(OK);
        case BSIM3v1_MOD_LWL:
            value->rValue = model->BSIM3v1Lwl;
            return(OK);
        case BSIM3v1_MOD_LMIN:
            value->rValue = model->BSIM3v1Lmin;
            return(OK);
        case BSIM3v1_MOD_LMAX:
            value->rValue = model->BSIM3v1Lmax;
            return(OK);
        case BSIM3v1_MOD_WINT:
            value->rValue = model->BSIM3v1Wint;
            return(OK);
        case BSIM3v1_MOD_WL:
            value->rValue = model->BSIM3v1Wl;
            return(OK);
        case BSIM3v1_MOD_WLN:
            value->rValue = model->BSIM3v1Wln;
            return(OK);
        case BSIM3v1_MOD_WW:
            value->rValue = model->BSIM3v1Ww;
            return(OK);
        case BSIM3v1_MOD_WWN:
            value->rValue = model->BSIM3v1Wwn;
            return(OK);
        case BSIM3v1_MOD_WWL:
            value->rValue = model->BSIM3v1Wwl;
            return(OK);
        case BSIM3v1_MOD_WMIN:
            value->rValue = model->BSIM3v1Wmin;
            return(OK);
        case BSIM3v1_MOD_WMAX:
            value->rValue = model->BSIM3v1Wmax;
            return(OK);
        case BSIM3v1_MOD_NOIA:
            value->rValue = model->BSIM3v1oxideTrapDensityA;
            return(OK);
        case BSIM3v1_MOD_NOIB:
            value->rValue = model->BSIM3v1oxideTrapDensityB;
            return(OK);
        case BSIM3v1_MOD_NOIC:
            value->rValue = model->BSIM3v1oxideTrapDensityC;
            return(OK);
        case BSIM3v1_MOD_EM:
            value->rValue = model->BSIM3v1em;
            return(OK);
        case BSIM3v1_MOD_EF:
            value->rValue = model->BSIM3v1ef;
            return(OK);
        case BSIM3v1_MOD_AF:
            value->rValue = model->BSIM3v1af;
            return(OK);
        case BSIM3v1_MOD_KF:
            value->rValue = model->BSIM3v1kf;
            return(OK);
        default:
            return(E_BADPARM);
    }
    /* NOTREACHED */
}



