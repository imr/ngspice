/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1995 Min-Chie Jeng and Mansun Chan.
File: b3v0mask.c
**********/

#include "ngspice/ngspice.h"
#include "ngspice/ifsim.h"
#include "ngspice/cktdefs.h"
#include "ngspice/devdefs.h"
#include "bsim3v0def.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

int
BSIM3v0mAsk(
CKTcircuit *ckt,
GENmodel *inst,
int which,
IFvalue *value)
{
    BSIM3v0model *model = (BSIM3v0model *)inst;

    NG_IGNORE(ckt);

    switch(which) 
    {   case BSIM3v0_MOD_MOBMOD:
            value->iValue = model->BSIM3v0mobMod; 
            return(OK);
        case BSIM3v0_MOD_BINUNIT:
            value->iValue = model->BSIM3v0binUnit; 
            return(OK);
        case BSIM3v0_MOD_CAPMOD:
            value->iValue = model->BSIM3v0capMod; 
            return(OK);
        case BSIM3v0_MOD_NQSMOD:
            value->iValue = model->BSIM3v0nqsMod; 
            return(OK);
        case BSIM3v0_MOD_NOIMOD:
            value->iValue = model->BSIM3v0noiMod; 
            return(OK);
        case  BSIM3v0_MOD_TOX :
          value->rValue = model->BSIM3v0tox;
            return(OK);
        case  BSIM3v0_MOD_CDSC :
          value->rValue = model->BSIM3v0cdsc;
            return(OK);
        case  BSIM3v0_MOD_CDSCB :
          value->rValue = model->BSIM3v0cdscb;
            return(OK);

        case  BSIM3v0_MOD_CDSCD :
          value->rValue = model->BSIM3v0cdscd;
            return(OK);

        case  BSIM3v0_MOD_CIT :
          value->rValue = model->BSIM3v0cit;
            return(OK);
        case  BSIM3v0_MOD_NFACTOR :
          value->rValue = model->BSIM3v0nfactor;
            return(OK);
        case BSIM3v0_MOD_XJ:
            value->rValue = model->BSIM3v0xj;
            return(OK);
        case BSIM3v0_MOD_VSAT:
            value->rValue = model->BSIM3v0vsat;
            return(OK);
        case BSIM3v0_MOD_AT:
            value->rValue = model->BSIM3v0at;
            return(OK);
        case BSIM3v0_MOD_A0:
            value->rValue = model->BSIM3v0a0;
            return(OK);

        case BSIM3v0_MOD_AGS:
            value->rValue = model->BSIM3v0ags;
            return(OK);

        case BSIM3v0_MOD_A1:
            value->rValue = model->BSIM3v0a1;
            return(OK);
        case BSIM3v0_MOD_A2:
            value->rValue = model->BSIM3v0a2;
            return(OK);
        case BSIM3v0_MOD_KETA:
            value->rValue = model->BSIM3v0keta;
            return(OK);   
        case BSIM3v0_MOD_NSUB:
            value->rValue = model->BSIM3v0nsub;
            return(OK);
        case BSIM3v0_MOD_NPEAK:
            value->rValue = model->BSIM3v0npeak;
            return(OK);
        case BSIM3v0_MOD_NGATE:
            value->rValue = model->BSIM3v0ngate;
            return(OK);
        case BSIM3v0_MOD_GAMMA1:
            value->rValue = model->BSIM3v0gamma1;
            return(OK);
        case BSIM3v0_MOD_GAMMA2:
            value->rValue = model->BSIM3v0gamma2;
            return(OK);
        case BSIM3v0_MOD_VBX:
            value->rValue = model->BSIM3v0vbx;
            return(OK);
        case BSIM3v0_MOD_VBM:
            value->rValue = model->BSIM3v0vbm;
            return(OK);
        case BSIM3v0_MOD_XT:
            value->rValue = model->BSIM3v0xt;
            return(OK);
        case  BSIM3v0_MOD_K1:
          value->rValue = model->BSIM3v0k1;
            return(OK);
        case  BSIM3v0_MOD_KT1:
          value->rValue = model->BSIM3v0kt1;
            return(OK);
        case  BSIM3v0_MOD_KT1L:
          value->rValue = model->BSIM3v0kt1l;
            return(OK);
        case  BSIM3v0_MOD_KT2 :
          value->rValue = model->BSIM3v0kt2;
            return(OK);
        case  BSIM3v0_MOD_K2 :
          value->rValue = model->BSIM3v0k2;
            return(OK);
        case  BSIM3v0_MOD_K3:
          value->rValue = model->BSIM3v0k3;
            return(OK);
        case  BSIM3v0_MOD_K3B:
          value->rValue = model->BSIM3v0k3b;
            return(OK);
        case  BSIM3v0_MOD_W0:
          value->rValue = model->BSIM3v0w0;
            return(OK);
        case  BSIM3v0_MOD_NLX:
          value->rValue = model->BSIM3v0nlx;
            return(OK);
        case  BSIM3v0_MOD_DVT0 :                
          value->rValue = model->BSIM3v0dvt0;
            return(OK);
        case  BSIM3v0_MOD_DVT1 :             
          value->rValue = model->BSIM3v0dvt1;
            return(OK);
        case  BSIM3v0_MOD_DVT2 :             
          value->rValue = model->BSIM3v0dvt2;
            return(OK);
        case  BSIM3v0_MOD_DVT0W :                
          value->rValue = model->BSIM3v0dvt0w;
            return(OK);
        case  BSIM3v0_MOD_DVT1W :             
          value->rValue = model->BSIM3v0dvt1w;
            return(OK);
        case  BSIM3v0_MOD_DVT2W :             
          value->rValue = model->BSIM3v0dvt2w;
            return(OK);
        case  BSIM3v0_MOD_DROUT :           
          value->rValue = model->BSIM3v0drout;
            return(OK);
        case  BSIM3v0_MOD_DSUB :           
          value->rValue = model->BSIM3v0dsub;
            return(OK);
        case BSIM3v0_MOD_VTH0:
            value->rValue = model->BSIM3v0vth0; 
            return(OK);
        case BSIM3v0_MOD_UA:
            value->rValue = model->BSIM3v0ua; 
            return(OK);
        case BSIM3v0_MOD_UA1:
            value->rValue = model->BSIM3v0ua1; 
            return(OK);
        case BSIM3v0_MOD_UB:
            value->rValue = model->BSIM3v0ub;  
            return(OK);
        case BSIM3v0_MOD_UB1:
            value->rValue = model->BSIM3v0ub1;  
            return(OK);
        case BSIM3v0_MOD_UC:
            value->rValue = model->BSIM3v0uc; 
            return(OK);
        case BSIM3v0_MOD_UC1:
            value->rValue = model->BSIM3v0uc1; 
            return(OK);
        case BSIM3v0_MOD_U0:
            value->rValue = model->BSIM3v0u0;
            return(OK);
        case BSIM3v0_MOD_UTE:
            value->rValue = model->BSIM3v0ute;
            return(OK);
        case BSIM3v0_MOD_VOFF:
            value->rValue = model->BSIM3v0voff;
            return(OK);
        case BSIM3v0_MOD_DELTA:
            value->rValue = model->BSIM3v0delta;
            return(OK);
        case BSIM3v0_MOD_RDSW:
            value->rValue = model->BSIM3v0rdsw; 
            return(OK);             
        case BSIM3v0_MOD_PRWG:
            value->rValue = model->BSIM3v0prwg; 
            return(OK);             
        case BSIM3v0_MOD_PRWB:
            value->rValue = model->BSIM3v0prwb; 
            return(OK);             
        case BSIM3v0_MOD_PRT:
            value->rValue = model->BSIM3v0prt; 
            return(OK);              
        case BSIM3v0_MOD_ETA0:
            value->rValue = model->BSIM3v0eta0; 
            return(OK);               
        case BSIM3v0_MOD_ETAB:
            value->rValue = model->BSIM3v0etab; 
            return(OK);               
        case BSIM3v0_MOD_PCLM:
            value->rValue = model->BSIM3v0pclm; 
            return(OK);               
        case BSIM3v0_MOD_PDIBL1:
            value->rValue = model->BSIM3v0pdibl1; 
            return(OK);               
        case BSIM3v0_MOD_PDIBL2:
            value->rValue = model->BSIM3v0pdibl2; 
            return(OK);               
        case BSIM3v0_MOD_PDIBLB:
            value->rValue = model->BSIM3v0pdiblb; 
            return(OK);               
        case BSIM3v0_MOD_PSCBE1:
            value->rValue = model->BSIM3v0pscbe1; 
            return(OK);               
        case BSIM3v0_MOD_PSCBE2:
            value->rValue = model->BSIM3v0pscbe2; 
            return(OK);               
        case BSIM3v0_MOD_PVAG:
            value->rValue = model->BSIM3v0pvag; 
            return(OK);               
        case BSIM3v0_MOD_WR:
            value->rValue = model->BSIM3v0wr;
            return(OK);
        case BSIM3v0_MOD_DWG:
            value->rValue = model->BSIM3v0dwg;
            return(OK);
        case BSIM3v0_MOD_DWB:
            value->rValue = model->BSIM3v0dwb;
            return(OK);
        case BSIM3v0_MOD_B0:
            value->rValue = model->BSIM3v0b0;
            return(OK);
        case BSIM3v0_MOD_B1:
            value->rValue = model->BSIM3v0b1;
            return(OK);
        case BSIM3v0_MOD_ALPHA0:
            value->rValue = model->BSIM3v0alpha0;
            return(OK);
        case BSIM3v0_MOD_BETA0:
            value->rValue = model->BSIM3v0beta0;
            return(OK);

        case BSIM3v0_MOD_ELM:
            value->rValue = model->BSIM3v0elm;
            return(OK);
        case BSIM3v0_MOD_CGSL:
            value->rValue = model->BSIM3v0cgsl;
            return(OK);
        case BSIM3v0_MOD_CGDL:
            value->rValue = model->BSIM3v0cgdl;
            return(OK);
        case BSIM3v0_MOD_CKAPPA:
            value->rValue = model->BSIM3v0ckappa;
            return(OK);
        case BSIM3v0_MOD_CF:
            value->rValue = model->BSIM3v0cf;
            return(OK);
        case BSIM3v0_MOD_CLC:
            value->rValue = model->BSIM3v0clc;
            return(OK);
        case BSIM3v0_MOD_CLE:
            value->rValue = model->BSIM3v0cle;
            return(OK);
        case BSIM3v0_MOD_DWC:
            value->rValue = model->BSIM3v0dwc;
            return(OK);
        case BSIM3v0_MOD_DLC:
            value->rValue = model->BSIM3v0dlc;
            return(OK);

	/* Length dependence */
        case  BSIM3v0_MOD_LCDSC :
          value->rValue = model->BSIM3v0lcdsc;
            return(OK);
        case  BSIM3v0_MOD_LCDSCB :
          value->rValue = model->BSIM3v0lcdscb;
            return(OK);
        case  BSIM3v0_MOD_LCDSCD :
          value->rValue = model->BSIM3v0lcdscd;
            return(OK);
        case  BSIM3v0_MOD_LCIT :
          value->rValue = model->BSIM3v0lcit;
            return(OK);
        case  BSIM3v0_MOD_LNFACTOR :
          value->rValue = model->BSIM3v0lnfactor;
            return(OK);
        case BSIM3v0_MOD_LXJ:
            value->rValue = model->BSIM3v0lxj;
            return(OK);
        case BSIM3v0_MOD_LVSAT:
            value->rValue = model->BSIM3v0lvsat;
            return(OK);
        case BSIM3v0_MOD_LAT:
            value->rValue = model->BSIM3v0lat;
            return(OK);
        case BSIM3v0_MOD_LA0:
            value->rValue = model->BSIM3v0la0;
            return(OK);
        case BSIM3v0_MOD_LAGS:
            value->rValue = model->BSIM3v0lags;
            return(OK);
        case BSIM3v0_MOD_LA1:
            value->rValue = model->BSIM3v0la1;
            return(OK);
        case BSIM3v0_MOD_LA2:
            value->rValue = model->BSIM3v0la2;
            return(OK);
        case BSIM3v0_MOD_LKETA:
            value->rValue = model->BSIM3v0lketa;
            return(OK);   
        case BSIM3v0_MOD_LNSUB:
            value->rValue = model->BSIM3v0lnsub;
            return(OK);
        case BSIM3v0_MOD_LNPEAK:
            value->rValue = model->BSIM3v0lnpeak;
            return(OK);
        case BSIM3v0_MOD_LNGATE:
            value->rValue = model->BSIM3v0lngate;
            return(OK);
        case BSIM3v0_MOD_LGAMMA1:
            value->rValue = model->BSIM3v0lgamma1;
            return(OK);
        case BSIM3v0_MOD_LGAMMA2:
            value->rValue = model->BSIM3v0lgamma2;
            return(OK);
        case BSIM3v0_MOD_LVBX:
            value->rValue = model->BSIM3v0lvbx;
            return(OK);
        case BSIM3v0_MOD_LVBM:
            value->rValue = model->BSIM3v0lvbm;
            return(OK);
        case BSIM3v0_MOD_LXT:
            value->rValue = model->BSIM3v0lxt;
            return(OK);
        case  BSIM3v0_MOD_LK1:
          value->rValue = model->BSIM3v0lk1;
            return(OK);
        case  BSIM3v0_MOD_LKT1:
          value->rValue = model->BSIM3v0lkt1;
            return(OK);
        case  BSIM3v0_MOD_LKT1L:
          value->rValue = model->BSIM3v0lkt1l;
            return(OK);
        case  BSIM3v0_MOD_LKT2 :
          value->rValue = model->BSIM3v0lkt2;
            return(OK);
        case  BSIM3v0_MOD_LK2 :
          value->rValue = model->BSIM3v0lk2;
            return(OK);
        case  BSIM3v0_MOD_LK3:
          value->rValue = model->BSIM3v0lk3;
            return(OK);
        case  BSIM3v0_MOD_LK3B:
          value->rValue = model->BSIM3v0lk3b;
            return(OK);
        case  BSIM3v0_MOD_LW0:
          value->rValue = model->BSIM3v0lw0;
            return(OK);
        case  BSIM3v0_MOD_LNLX:
          value->rValue = model->BSIM3v0lnlx;
            return(OK);
        case  BSIM3v0_MOD_LDVT0:                
          value->rValue = model->BSIM3v0ldvt0;
            return(OK);
        case  BSIM3v0_MOD_LDVT1 :             
          value->rValue = model->BSIM3v0ldvt1;
            return(OK);
        case  BSIM3v0_MOD_LDVT2 :             
          value->rValue = model->BSIM3v0ldvt2;
            return(OK);
        case  BSIM3v0_MOD_LDVT0W :                
          value->rValue = model->BSIM3v0ldvt0w;
            return(OK);
        case  BSIM3v0_MOD_LDVT1W :             
          value->rValue = model->BSIM3v0ldvt1w;
            return(OK);
        case  BSIM3v0_MOD_LDVT2W :             
          value->rValue = model->BSIM3v0ldvt2w;
            return(OK);
        case  BSIM3v0_MOD_LDROUT :           
          value->rValue = model->BSIM3v0ldrout;
            return(OK);
        case  BSIM3v0_MOD_LDSUB :           
          value->rValue = model->BSIM3v0ldsub;
            return(OK);
        case BSIM3v0_MOD_LVTH0:
            value->rValue = model->BSIM3v0lvth0; 
            return(OK);
        case BSIM3v0_MOD_LUA:
            value->rValue = model->BSIM3v0lua; 
            return(OK);
        case BSIM3v0_MOD_LUA1:
            value->rValue = model->BSIM3v0lua1; 
            return(OK);
        case BSIM3v0_MOD_LUB:
            value->rValue = model->BSIM3v0lub;  
            return(OK);
        case BSIM3v0_MOD_LUB1:
            value->rValue = model->BSIM3v0lub1;  
            return(OK);
        case BSIM3v0_MOD_LUC:
            value->rValue = model->BSIM3v0luc; 
            return(OK);
        case BSIM3v0_MOD_LUC1:
            value->rValue = model->BSIM3v0luc1; 
            return(OK);
        case BSIM3v0_MOD_LU0:
            value->rValue = model->BSIM3v0lu0;
            return(OK);
        case BSIM3v0_MOD_LUTE:
            value->rValue = model->BSIM3v0lute;
            return(OK);
        case BSIM3v0_MOD_LVOFF:
            value->rValue = model->BSIM3v0lvoff;
            return(OK);
        case BSIM3v0_MOD_LDELTA:
            value->rValue = model->BSIM3v0ldelta;
            return(OK);
        case BSIM3v0_MOD_LRDSW:
            value->rValue = model->BSIM3v0lrdsw; 
            return(OK);             
        case BSIM3v0_MOD_LPRWB:
            value->rValue = model->BSIM3v0lprwb; 
            return(OK);             
        case BSIM3v0_MOD_LPRWG:
            value->rValue = model->BSIM3v0lprwg; 
            return(OK);             
        case BSIM3v0_MOD_LPRT:
            value->rValue = model->BSIM3v0lprt; 
            return(OK);              
        case BSIM3v0_MOD_LETA0:
            value->rValue = model->BSIM3v0leta0; 
            return(OK);               
        case BSIM3v0_MOD_LETAB:
            value->rValue = model->BSIM3v0letab; 
            return(OK);               
        case BSIM3v0_MOD_LPCLM:
            value->rValue = model->BSIM3v0lpclm; 
            return(OK);               
        case BSIM3v0_MOD_LPDIBL1:
            value->rValue = model->BSIM3v0lpdibl1; 
            return(OK);               
        case BSIM3v0_MOD_LPDIBL2:
            value->rValue = model->BSIM3v0lpdibl2; 
            return(OK);               
        case BSIM3v0_MOD_LPDIBLB:
            value->rValue = model->BSIM3v0lpdiblb; 
            return(OK);               
        case BSIM3v0_MOD_LPSCBE1:
            value->rValue = model->BSIM3v0lpscbe1; 
            return(OK);               
        case BSIM3v0_MOD_LPSCBE2:
            value->rValue = model->BSIM3v0lpscbe2; 
            return(OK);               
        case BSIM3v0_MOD_LPVAG:
            value->rValue = model->BSIM3v0lpvag; 
            return(OK);               
        case BSIM3v0_MOD_LWR:
            value->rValue = model->BSIM3v0lwr;
            return(OK);
        case BSIM3v0_MOD_LDWG:
            value->rValue = model->BSIM3v0ldwg;
            return(OK);
        case BSIM3v0_MOD_LDWB:
            value->rValue = model->BSIM3v0ldwb;
            return(OK);
        case BSIM3v0_MOD_LB0:
            value->rValue = model->BSIM3v0lb0;
            return(OK);
        case BSIM3v0_MOD_LB1:
            value->rValue = model->BSIM3v0lb1;
            return(OK);
        case BSIM3v0_MOD_LALPHA0:
            value->rValue = model->BSIM3v0lalpha0;
            return(OK);
        case BSIM3v0_MOD_LBETA0:
            value->rValue = model->BSIM3v0lbeta0;
            return(OK);

        case BSIM3v0_MOD_LELM:
            value->rValue = model->BSIM3v0lelm;
            return(OK);
        case BSIM3v0_MOD_LCGSL:
            value->rValue = model->BSIM3v0lcgsl;
            return(OK);
        case BSIM3v0_MOD_LCGDL:
            value->rValue = model->BSIM3v0lcgdl;
            return(OK);
        case BSIM3v0_MOD_LCKAPPA:
            value->rValue = model->BSIM3v0lckappa;
            return(OK);
        case BSIM3v0_MOD_LCF:
            value->rValue = model->BSIM3v0lcf;
            return(OK);
        case BSIM3v0_MOD_LCLC:
            value->rValue = model->BSIM3v0lclc;
            return(OK);
        case BSIM3v0_MOD_LCLE:
            value->rValue = model->BSIM3v0lcle;
            return(OK);

	/* Width dependence */
        case  BSIM3v0_MOD_WCDSC :
          value->rValue = model->BSIM3v0wcdsc;
            return(OK);
        case  BSIM3v0_MOD_WCDSCB :
          value->rValue = model->BSIM3v0wcdscb;
            return(OK);
        case  BSIM3v0_MOD_WCDSCD :
          value->rValue = model->BSIM3v0wcdscd;
            return(OK);
        case  BSIM3v0_MOD_WCIT :
          value->rValue = model->BSIM3v0wcit;
            return(OK);
        case  BSIM3v0_MOD_WNFACTOR :
          value->rValue = model->BSIM3v0wnfactor;
            return(OK);
        case BSIM3v0_MOD_WXJ:
            value->rValue = model->BSIM3v0wxj;
            return(OK);
        case BSIM3v0_MOD_WVSAT:
            value->rValue = model->BSIM3v0wvsat;
            return(OK);
        case BSIM3v0_MOD_WAT:
            value->rValue = model->BSIM3v0wat;
            return(OK);
        case BSIM3v0_MOD_WA0:
            value->rValue = model->BSIM3v0wa0;
            return(OK);
        case BSIM3v0_MOD_WAGS:
            value->rValue = model->BSIM3v0wags;
            return(OK);
        case BSIM3v0_MOD_WA1:
            value->rValue = model->BSIM3v0wa1;
            return(OK);
        case BSIM3v0_MOD_WA2:
            value->rValue = model->BSIM3v0wa2;
            return(OK);
        case BSIM3v0_MOD_WKETA:
            value->rValue = model->BSIM3v0wketa;
            return(OK);   
        case BSIM3v0_MOD_WNSUB:
            value->rValue = model->BSIM3v0wnsub;
            return(OK);
        case BSIM3v0_MOD_WNPEAK:
            value->rValue = model->BSIM3v0wnpeak;
            return(OK);
        case BSIM3v0_MOD_WNGATE:
            value->rValue = model->BSIM3v0wngate;
            return(OK);
        case BSIM3v0_MOD_WGAMMA1:
            value->rValue = model->BSIM3v0wgamma1;
            return(OK);
        case BSIM3v0_MOD_WGAMMA2:
            value->rValue = model->BSIM3v0wgamma2;
            return(OK);
        case BSIM3v0_MOD_WVBX:
            value->rValue = model->BSIM3v0wvbx;
            return(OK);
        case BSIM3v0_MOD_WVBM:
            value->rValue = model->BSIM3v0wvbm;
            return(OK);
        case BSIM3v0_MOD_WXT:
            value->rValue = model->BSIM3v0wxt;
            return(OK);
        case  BSIM3v0_MOD_WK1:
          value->rValue = model->BSIM3v0wk1;
            return(OK);
        case  BSIM3v0_MOD_WKT1:
          value->rValue = model->BSIM3v0wkt1;
            return(OK);
        case  BSIM3v0_MOD_WKT1L:
          value->rValue = model->BSIM3v0wkt1l;
            return(OK);
        case  BSIM3v0_MOD_WKT2 :
          value->rValue = model->BSIM3v0wkt2;
            return(OK);
        case  BSIM3v0_MOD_WK2 :
          value->rValue = model->BSIM3v0wk2;
            return(OK);
        case  BSIM3v0_MOD_WK3:
          value->rValue = model->BSIM3v0wk3;
            return(OK);
        case  BSIM3v0_MOD_WK3B:
          value->rValue = model->BSIM3v0wk3b;
            return(OK);
        case  BSIM3v0_MOD_WW0:
          value->rValue = model->BSIM3v0ww0;
            return(OK);
        case  BSIM3v0_MOD_WNLX:
          value->rValue = model->BSIM3v0wnlx;
            return(OK);
        case  BSIM3v0_MOD_WDVT0:                
          value->rValue = model->BSIM3v0wdvt0;
            return(OK);
        case  BSIM3v0_MOD_WDVT1 :             
          value->rValue = model->BSIM3v0wdvt1;
            return(OK);
        case  BSIM3v0_MOD_WDVT2 :             
          value->rValue = model->BSIM3v0wdvt2;
            return(OK);
        case  BSIM3v0_MOD_WDVT0W :                
          value->rValue = model->BSIM3v0wdvt0w;
            return(OK);
        case  BSIM3v0_MOD_WDVT1W :             
          value->rValue = model->BSIM3v0wdvt1w;
            return(OK);
        case  BSIM3v0_MOD_WDVT2W :             
          value->rValue = model->BSIM3v0wdvt2w;
            return(OK);
        case  BSIM3v0_MOD_WDROUT :           
          value->rValue = model->BSIM3v0wdrout;
            return(OK);
        case  BSIM3v0_MOD_WDSUB :           
          value->rValue = model->BSIM3v0wdsub;
            return(OK);
        case BSIM3v0_MOD_WVTH0:
            value->rValue = model->BSIM3v0wvth0; 
            return(OK);
        case BSIM3v0_MOD_WUA:
            value->rValue = model->BSIM3v0wua; 
            return(OK);
        case BSIM3v0_MOD_WUA1:
            value->rValue = model->BSIM3v0wua1; 
            return(OK);
        case BSIM3v0_MOD_WUB:
            value->rValue = model->BSIM3v0wub;  
            return(OK);
        case BSIM3v0_MOD_WUB1:
            value->rValue = model->BSIM3v0wub1;  
            return(OK);
        case BSIM3v0_MOD_WUC:
            value->rValue = model->BSIM3v0wuc; 
            return(OK);
        case BSIM3v0_MOD_WUC1:
            value->rValue = model->BSIM3v0wuc1; 
            return(OK);
        case BSIM3v0_MOD_WU0:
            value->rValue = model->BSIM3v0wu0;
            return(OK);
        case BSIM3v0_MOD_WUTE:
            value->rValue = model->BSIM3v0wute;
            return(OK);
        case BSIM3v0_MOD_WVOFF:
            value->rValue = model->BSIM3v0wvoff;
            return(OK);
        case BSIM3v0_MOD_WDELTA:
            value->rValue = model->BSIM3v0wdelta;
            return(OK);
        case BSIM3v0_MOD_WRDSW:
            value->rValue = model->BSIM3v0wrdsw; 
            return(OK);             
        case BSIM3v0_MOD_WPRWB:
            value->rValue = model->BSIM3v0wprwb; 
            return(OK);             
        case BSIM3v0_MOD_WPRWG:
            value->rValue = model->BSIM3v0wprwg; 
            return(OK);             
        case BSIM3v0_MOD_WPRT:
            value->rValue = model->BSIM3v0wprt; 
            return(OK);              
        case BSIM3v0_MOD_WETA0:
            value->rValue = model->BSIM3v0weta0; 
            return(OK);               
        case BSIM3v0_MOD_WETAB:
            value->rValue = model->BSIM3v0wetab; 
            return(OK);               
        case BSIM3v0_MOD_WPCLM:
            value->rValue = model->BSIM3v0wpclm; 
            return(OK);               
        case BSIM3v0_MOD_WPDIBL1:
            value->rValue = model->BSIM3v0wpdibl1; 
            return(OK);               
        case BSIM3v0_MOD_WPDIBL2:
            value->rValue = model->BSIM3v0wpdibl2; 
            return(OK);               
        case BSIM3v0_MOD_WPDIBLB:
            value->rValue = model->BSIM3v0wpdiblb; 
            return(OK);               
        case BSIM3v0_MOD_WPSCBE1:
            value->rValue = model->BSIM3v0wpscbe1; 
            return(OK);               
        case BSIM3v0_MOD_WPSCBE2:
            value->rValue = model->BSIM3v0wpscbe2; 
            return(OK);               
        case BSIM3v0_MOD_WPVAG:
            value->rValue = model->BSIM3v0wpvag; 
            return(OK);               
        case BSIM3v0_MOD_WWR:
            value->rValue = model->BSIM3v0wwr;
            return(OK);
        case BSIM3v0_MOD_WDWG:
            value->rValue = model->BSIM3v0wdwg;
            return(OK);
        case BSIM3v0_MOD_WDWB:
            value->rValue = model->BSIM3v0wdwb;
            return(OK);
        case BSIM3v0_MOD_WB0:
            value->rValue = model->BSIM3v0wb0;
            return(OK);
        case BSIM3v0_MOD_WB1:
            value->rValue = model->BSIM3v0wb1;
            return(OK);
        case BSIM3v0_MOD_WALPHA0:
            value->rValue = model->BSIM3v0walpha0;
            return(OK);
        case BSIM3v0_MOD_WBETA0:
            value->rValue = model->BSIM3v0wbeta0;
            return(OK);

        case BSIM3v0_MOD_WELM:
            value->rValue = model->BSIM3v0welm;
            return(OK);
        case BSIM3v0_MOD_WCGSL:
            value->rValue = model->BSIM3v0wcgsl;
            return(OK);
        case BSIM3v0_MOD_WCGDL:
            value->rValue = model->BSIM3v0wcgdl;
            return(OK);
        case BSIM3v0_MOD_WCKAPPA:
            value->rValue = model->BSIM3v0wckappa;
            return(OK);
        case BSIM3v0_MOD_WCF:
            value->rValue = model->BSIM3v0wcf;
            return(OK);
        case BSIM3v0_MOD_WCLC:
            value->rValue = model->BSIM3v0wclc;
            return(OK);
        case BSIM3v0_MOD_WCLE:
            value->rValue = model->BSIM3v0wcle;
            return(OK);

	/* Cross-term dependence */
        case  BSIM3v0_MOD_PCDSC :
          value->rValue = model->BSIM3v0pcdsc;
            return(OK);
        case  BSIM3v0_MOD_PCDSCB :
          value->rValue = model->BSIM3v0pcdscb;
            return(OK);
        case  BSIM3v0_MOD_PCDSCD :
          value->rValue = model->BSIM3v0pcdscd;
            return(OK);
         case  BSIM3v0_MOD_PCIT :
          value->rValue = model->BSIM3v0pcit;
            return(OK);
        case  BSIM3v0_MOD_PNFACTOR :
          value->rValue = model->BSIM3v0pnfactor;
            return(OK);
        case BSIM3v0_MOD_PXJ:
            value->rValue = model->BSIM3v0pxj;
            return(OK);
        case BSIM3v0_MOD_PVSAT:
            value->rValue = model->BSIM3v0pvsat;
            return(OK);
        case BSIM3v0_MOD_PAT:
            value->rValue = model->BSIM3v0pat;
            return(OK);
        case BSIM3v0_MOD_PA0:
            value->rValue = model->BSIM3v0pa0;
            return(OK);
        case BSIM3v0_MOD_PAGS:
            value->rValue = model->BSIM3v0pags;
            return(OK);
        case BSIM3v0_MOD_PA1:
            value->rValue = model->BSIM3v0pa1;
            return(OK);
        case BSIM3v0_MOD_PA2:
            value->rValue = model->BSIM3v0pa2;
            return(OK);
        case BSIM3v0_MOD_PKETA:
            value->rValue = model->BSIM3v0pketa;
            return(OK);   
        case BSIM3v0_MOD_PNSUB:
            value->rValue = model->BSIM3v0pnsub;
            return(OK);
        case BSIM3v0_MOD_PNPEAK:
            value->rValue = model->BSIM3v0pnpeak;
            return(OK);
        case BSIM3v0_MOD_PNGATE:
            value->rValue = model->BSIM3v0pngate;
            return(OK);
        case BSIM3v0_MOD_PGAMMA1:
            value->rValue = model->BSIM3v0pgamma1;
            return(OK);
        case BSIM3v0_MOD_PGAMMA2:
            value->rValue = model->BSIM3v0pgamma2;
            return(OK);
        case BSIM3v0_MOD_PVBX:
            value->rValue = model->BSIM3v0pvbx;
            return(OK);
        case BSIM3v0_MOD_PVBM:
            value->rValue = model->BSIM3v0pvbm;
            return(OK);
        case BSIM3v0_MOD_PXT:
            value->rValue = model->BSIM3v0pxt;
            return(OK);
        case  BSIM3v0_MOD_PK1:
          value->rValue = model->BSIM3v0pk1;
            return(OK);
        case  BSIM3v0_MOD_PKT1:
          value->rValue = model->BSIM3v0pkt1;
            return(OK);
        case  BSIM3v0_MOD_PKT1L:
          value->rValue = model->BSIM3v0pkt1l;
            return(OK);
        case  BSIM3v0_MOD_PKT2 :
          value->rValue = model->BSIM3v0pkt2;
            return(OK);
        case  BSIM3v0_MOD_PK2 :
          value->rValue = model->BSIM3v0pk2;
            return(OK);
        case  BSIM3v0_MOD_PK3:
          value->rValue = model->BSIM3v0pk3;
            return(OK);
        case  BSIM3v0_MOD_PK3B:
          value->rValue = model->BSIM3v0pk3b;
            return(OK);
        case  BSIM3v0_MOD_PW0:
          value->rValue = model->BSIM3v0pw0;
            return(OK);
        case  BSIM3v0_MOD_PNLX:
          value->rValue = model->BSIM3v0pnlx;
            return(OK);
        case  BSIM3v0_MOD_PDVT0 :                
          value->rValue = model->BSIM3v0pdvt0;
            return(OK);
        case  BSIM3v0_MOD_PDVT1 :             
          value->rValue = model->BSIM3v0pdvt1;
            return(OK);
        case  BSIM3v0_MOD_PDVT2 :             
          value->rValue = model->BSIM3v0pdvt2;
            return(OK);
        case  BSIM3v0_MOD_PDVT0W :                
          value->rValue = model->BSIM3v0pdvt0w;
            return(OK);
        case  BSIM3v0_MOD_PDVT1W :             
          value->rValue = model->BSIM3v0pdvt1w;
            return(OK);
        case  BSIM3v0_MOD_PDVT2W :             
          value->rValue = model->BSIM3v0pdvt2w;
            return(OK);
        case  BSIM3v0_MOD_PDROUT :           
          value->rValue = model->BSIM3v0pdrout;
            return(OK);
        case  BSIM3v0_MOD_PDSUB :           
          value->rValue = model->BSIM3v0pdsub;
            return(OK);
        case BSIM3v0_MOD_PVTH0:
            value->rValue = model->BSIM3v0pvth0; 
            return(OK);
        case BSIM3v0_MOD_PUA:
            value->rValue = model->BSIM3v0pua; 
            return(OK);
        case BSIM3v0_MOD_PUA1:
            value->rValue = model->BSIM3v0pua1; 
            return(OK);
        case BSIM3v0_MOD_PUB:
            value->rValue = model->BSIM3v0pub;  
            return(OK);
        case BSIM3v0_MOD_PUB1:
            value->rValue = model->BSIM3v0pub1;  
            return(OK);
        case BSIM3v0_MOD_PUC:
            value->rValue = model->BSIM3v0puc; 
            return(OK);
        case BSIM3v0_MOD_PUC1:
            value->rValue = model->BSIM3v0puc1; 
            return(OK);
        case BSIM3v0_MOD_PU0:
            value->rValue = model->BSIM3v0pu0;
            return(OK);
        case BSIM3v0_MOD_PUTE:
            value->rValue = model->BSIM3v0pute;
            return(OK);
        case BSIM3v0_MOD_PVOFF:
            value->rValue = model->BSIM3v0pvoff;
            return(OK);
        case BSIM3v0_MOD_PDELTA:
            value->rValue = model->BSIM3v0pdelta;
            return(OK);
        case BSIM3v0_MOD_PRDSW:
            value->rValue = model->BSIM3v0prdsw; 
            return(OK);             
        case BSIM3v0_MOD_PPRWB:
            value->rValue = model->BSIM3v0pprwb; 
            return(OK);             
        case BSIM3v0_MOD_PPRWG:
            value->rValue = model->BSIM3v0pprwg; 
            return(OK);             
        case BSIM3v0_MOD_PPRT:
            value->rValue = model->BSIM3v0pprt; 
            return(OK);              
        case BSIM3v0_MOD_PETA0:
            value->rValue = model->BSIM3v0peta0; 
            return(OK);               
        case BSIM3v0_MOD_PETAB:
            value->rValue = model->BSIM3v0petab; 
            return(OK);               
        case BSIM3v0_MOD_PPCLM:
            value->rValue = model->BSIM3v0ppclm; 
            return(OK);               
        case BSIM3v0_MOD_PPDIBL1:
            value->rValue = model->BSIM3v0ppdibl1; 
            return(OK);               
        case BSIM3v0_MOD_PPDIBL2:
            value->rValue = model->BSIM3v0ppdibl2; 
            return(OK);               
        case BSIM3v0_MOD_PPDIBLB:
            value->rValue = model->BSIM3v0ppdiblb; 
            return(OK);               
        case BSIM3v0_MOD_PPSCBE1:
            value->rValue = model->BSIM3v0ppscbe1; 
            return(OK);               
        case BSIM3v0_MOD_PPSCBE2:
            value->rValue = model->BSIM3v0ppscbe2; 
            return(OK);               
        case BSIM3v0_MOD_PPVAG:
            value->rValue = model->BSIM3v0ppvag; 
            return(OK);               
        case BSIM3v0_MOD_PWR:
            value->rValue = model->BSIM3v0pwr;
            return(OK);
        case BSIM3v0_MOD_PDWG:
            value->rValue = model->BSIM3v0pdwg;
            return(OK);
        case BSIM3v0_MOD_PDWB:
            value->rValue = model->BSIM3v0pdwb;
            return(OK);
        case BSIM3v0_MOD_PB0:
            value->rValue = model->BSIM3v0pb0;
            return(OK);
        case BSIM3v0_MOD_PB1:
            value->rValue = model->BSIM3v0pb1;
            return(OK);
        case BSIM3v0_MOD_PALPHA0:
            value->rValue = model->BSIM3v0palpha0;
            return(OK);
        case BSIM3v0_MOD_PBETA0:
            value->rValue = model->BSIM3v0pbeta0;
            return(OK);

        case BSIM3v0_MOD_PELM:
            value->rValue = model->BSIM3v0pelm;
            return(OK);
        case BSIM3v0_MOD_PCGSL:
            value->rValue = model->BSIM3v0pcgsl;
            return(OK);
        case BSIM3v0_MOD_PCGDL:
            value->rValue = model->BSIM3v0pcgdl;
            return(OK);
        case BSIM3v0_MOD_PCKAPPA:
            value->rValue = model->BSIM3v0pckappa;
            return(OK);
        case BSIM3v0_MOD_PCF:
            value->rValue = model->BSIM3v0pcf;
            return(OK);
        case BSIM3v0_MOD_PCLC:
            value->rValue = model->BSIM3v0pclc;
            return(OK);
        case BSIM3v0_MOD_PCLE:
            value->rValue = model->BSIM3v0pcle;
            return(OK);

        case  BSIM3v0_MOD_TNOM :
          value->rValue = model->BSIM3v0tnom;
            return(OK);
        case BSIM3v0_MOD_CGSO:
            value->rValue = model->BSIM3v0cgso; 
            return(OK);
        case BSIM3v0_MOD_CGDO:
            value->rValue = model->BSIM3v0cgdo; 
            return(OK);
        case BSIM3v0_MOD_CGBO:
            value->rValue = model->BSIM3v0cgbo; 
            return(OK);
        case BSIM3v0_MOD_XPART:
            value->rValue = model->BSIM3v0xpart; 
            return(OK);
        case BSIM3v0_MOD_RSH:
            value->rValue = model->BSIM3v0sheetResistance; 
            return(OK);
        case BSIM3v0_MOD_JS:
            value->rValue = model->BSIM3v0jctSatCurDensity; 
            return(OK);
        case BSIM3v0_MOD_PB:
            value->rValue = model->BSIM3v0bulkJctPotential; 
            return(OK);
        case BSIM3v0_MOD_MJ:
            value->rValue = model->BSIM3v0bulkJctBotGradingCoeff; 
            return(OK);
        case BSIM3v0_MOD_PBSW:
            value->rValue = model->BSIM3v0sidewallJctPotential; 
            return(OK);
        case BSIM3v0_MOD_MJSW:
            value->rValue = model->BSIM3v0bulkJctSideGradingCoeff; 
            return(OK);
        case BSIM3v0_MOD_CJ:
            value->rValue = model->BSIM3v0unitAreaJctCap; 
            return(OK);
        case BSIM3v0_MOD_CJSW:
            value->rValue = model->BSIM3v0unitLengthSidewallJctCap; 
            return(OK);
        case BSIM3v0_MOD_LINT:
            value->rValue = model->BSIM3v0Lint; 
            return(OK);
        case BSIM3v0_MOD_LL:
            value->rValue = model->BSIM3v0Ll;
            return(OK);
        case BSIM3v0_MOD_LLN:
            value->rValue = model->BSIM3v0Lln;
            return(OK);
        case BSIM3v0_MOD_LW:
            value->rValue = model->BSIM3v0Lw;
            return(OK);
        case BSIM3v0_MOD_LWN:
            value->rValue = model->BSIM3v0Lwn;
            return(OK);
        case BSIM3v0_MOD_LWL:
            value->rValue = model->BSIM3v0Lwl;
            return(OK);
        case BSIM3v0_MOD_LMIN:
            value->rValue = model->BSIM3v0Lmin;
            return(OK);
        case BSIM3v0_MOD_LMAX:
            value->rValue = model->BSIM3v0Lmax;
            return(OK);
        case BSIM3v0_MOD_WINT:
            value->rValue = model->BSIM3v0Wint;
            return(OK);
        case BSIM3v0_MOD_WL:
            value->rValue = model->BSIM3v0Wl;
            return(OK);
        case BSIM3v0_MOD_WLN:
            value->rValue = model->BSIM3v0Wln;
            return(OK);
        case BSIM3v0_MOD_WW:
            value->rValue = model->BSIM3v0Ww;
            return(OK);
        case BSIM3v0_MOD_WWN:
            value->rValue = model->BSIM3v0Wwn;
            return(OK);
        case BSIM3v0_MOD_WWL:
            value->rValue = model->BSIM3v0Wwl;
            return(OK);
        case BSIM3v0_MOD_WMIN:
            value->rValue = model->BSIM3v0Wmin;
            return(OK);
        case BSIM3v0_MOD_WMAX:
            value->rValue = model->BSIM3v0Wmax;
            return(OK);
        case BSIM3v0_MOD_NOIA:
            value->rValue = model->BSIM3v0oxideTrapDensityA;
            return(OK);
        case BSIM3v0_MOD_NOIB:
            value->rValue = model->BSIM3v0oxideTrapDensityB;
            return(OK);
        case BSIM3v0_MOD_NOIC:
            value->rValue = model->BSIM3v0oxideTrapDensityC;
            return(OK);
        case BSIM3v0_MOD_EM:
            value->rValue = model->BSIM3v0em;
            return(OK);
        case BSIM3v0_MOD_EF:
            value->rValue = model->BSIM3v0ef;
            return(OK);
        case BSIM3v0_MOD_AF:
            value->rValue = model->BSIM3v0af;
            return(OK);
        case BSIM3v0_MOD_KF:
            value->rValue = model->BSIM3v0kf;
            return(OK);
        default:
            return(E_BADPARM);
    }
    /* NOTREACHED */
}



