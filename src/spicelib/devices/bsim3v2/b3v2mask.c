/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1995 Min-Chie Jeng and Mansun Chan.
Modified by Weidong Liu (1997-1998).
File: b3v2mask.c
**********/

#include "ngspice.h"
#include <stdio.h>
#include "ifsim.h"
#include "cktdefs.h"
#include "devdefs.h"
#include "bsim3v2def.h"
#include "sperror.h"
#include "suffix.h"

int
BSIM3V2mAsk(ckt,inst,which,value)
CKTcircuit *ckt;
GENmodel *inst;
int which;
IFvalue *value;
{
    BSIM3V2model *model = (BSIM3V2model *)inst;
    switch(which) 
    {   case BSIM3V2_MOD_MOBMOD:
            value->iValue = model->BSIM3V2mobMod; 
            return(OK);
        case BSIM3V2_MOD_PARAMCHK:
            value->iValue = model->BSIM3V2paramChk; 
            return(OK);
        case BSIM3V2_MOD_BINUNIT:
            value->iValue = model->BSIM3V2binUnit; 
            return(OK);
        case BSIM3V2_MOD_CAPMOD:
            value->iValue = model->BSIM3V2capMod; 
            return(OK);
        case BSIM3V2_MOD_NOIMOD:
            value->iValue = model->BSIM3V2noiMod; 
            return(OK);
        case  BSIM3V2_MOD_VERSION :
          value->rValue = model->BSIM3V2version;
            return(OK);
        case  BSIM3V2_MOD_TOX :
          value->rValue = model->BSIM3V2tox;
            return(OK);
        case  BSIM3V2_MOD_TOXM :
          value->rValue = model->BSIM3V2toxm;
            return(OK);
        case  BSIM3V2_MOD_CDSC :
          value->rValue = model->BSIM3V2cdsc;
            return(OK);
        case  BSIM3V2_MOD_CDSCB :
          value->rValue = model->BSIM3V2cdscb;
            return(OK);

        case  BSIM3V2_MOD_CDSCD :
          value->rValue = model->BSIM3V2cdscd;
            return(OK);

        case  BSIM3V2_MOD_CIT :
          value->rValue = model->BSIM3V2cit;
            return(OK);
        case  BSIM3V2_MOD_NFACTOR :
          value->rValue = model->BSIM3V2nfactor;
            return(OK);
        case BSIM3V2_MOD_XJ:
            value->rValue = model->BSIM3V2xj;
            return(OK);
        case BSIM3V2_MOD_VSAT:
            value->rValue = model->BSIM3V2vsat;
            return(OK);
        case BSIM3V2_MOD_AT:
            value->rValue = model->BSIM3V2at;
            return(OK);
        case BSIM3V2_MOD_A0:
            value->rValue = model->BSIM3V2a0;
            return(OK);

        case BSIM3V2_MOD_AGS:
            value->rValue = model->BSIM3V2ags;
            return(OK);

        case BSIM3V2_MOD_A1:
            value->rValue = model->BSIM3V2a1;
            return(OK);
        case BSIM3V2_MOD_A2:
            value->rValue = model->BSIM3V2a2;
            return(OK);
        case BSIM3V2_MOD_KETA:
            value->rValue = model->BSIM3V2keta;
            return(OK);   
        case BSIM3V2_MOD_NSUB:
            value->rValue = model->BSIM3V2nsub;
            return(OK);
        case BSIM3V2_MOD_NPEAK:
            value->rValue = model->BSIM3V2npeak;
            return(OK);
        case BSIM3V2_MOD_NGATE:
            value->rValue = model->BSIM3V2ngate;
            return(OK);
        case BSIM3V2_MOD_GAMMA1:
            value->rValue = model->BSIM3V2gamma1;
            return(OK);
        case BSIM3V2_MOD_GAMMA2:
            value->rValue = model->BSIM3V2gamma2;
            return(OK);
        case BSIM3V2_MOD_VBX:
            value->rValue = model->BSIM3V2vbx;
            return(OK);
        case BSIM3V2_MOD_VBM:
            value->rValue = model->BSIM3V2vbm;
            return(OK);
        case BSIM3V2_MOD_XT:
            value->rValue = model->BSIM3V2xt;
            return(OK);
        case  BSIM3V2_MOD_K1:
          value->rValue = model->BSIM3V2k1;
            return(OK);
        case  BSIM3V2_MOD_KT1:
          value->rValue = model->BSIM3V2kt1;
            return(OK);
        case  BSIM3V2_MOD_KT1L:
          value->rValue = model->BSIM3V2kt1l;
            return(OK);
        case  BSIM3V2_MOD_KT2 :
          value->rValue = model->BSIM3V2kt2;
            return(OK);
        case  BSIM3V2_MOD_K2 :
          value->rValue = model->BSIM3V2k2;
            return(OK);
        case  BSIM3V2_MOD_K3:
          value->rValue = model->BSIM3V2k3;
            return(OK);
        case  BSIM3V2_MOD_K3B:
          value->rValue = model->BSIM3V2k3b;
            return(OK);
        case  BSIM3V2_MOD_W0:
          value->rValue = model->BSIM3V2w0;
            return(OK);
        case  BSIM3V2_MOD_NLX:
          value->rValue = model->BSIM3V2nlx;
            return(OK);
        case  BSIM3V2_MOD_DVT0 :                
          value->rValue = model->BSIM3V2dvt0;
            return(OK);
        case  BSIM3V2_MOD_DVT1 :             
          value->rValue = model->BSIM3V2dvt1;
            return(OK);
        case  BSIM3V2_MOD_DVT2 :             
          value->rValue = model->BSIM3V2dvt2;
            return(OK);
        case  BSIM3V2_MOD_DVT0W :                
          value->rValue = model->BSIM3V2dvt0w;
            return(OK);
        case  BSIM3V2_MOD_DVT1W :             
          value->rValue = model->BSIM3V2dvt1w;
            return(OK);
        case  BSIM3V2_MOD_DVT2W :             
          value->rValue = model->BSIM3V2dvt2w;
            return(OK);
        case  BSIM3V2_MOD_DROUT :           
          value->rValue = model->BSIM3V2drout;
            return(OK);
        case  BSIM3V2_MOD_DSUB :           
          value->rValue = model->BSIM3V2dsub;
            return(OK);
        case BSIM3V2_MOD_VTH0:
            value->rValue = model->BSIM3V2vth0; 
            return(OK);
        case BSIM3V2_MOD_UA:
            value->rValue = model->BSIM3V2ua; 
            return(OK);
        case BSIM3V2_MOD_UA1:
            value->rValue = model->BSIM3V2ua1; 
            return(OK);
        case BSIM3V2_MOD_UB:
            value->rValue = model->BSIM3V2ub;  
            return(OK);
        case BSIM3V2_MOD_UB1:
            value->rValue = model->BSIM3V2ub1;  
            return(OK);
        case BSIM3V2_MOD_UC:
            value->rValue = model->BSIM3V2uc; 
            return(OK);
        case BSIM3V2_MOD_UC1:
            value->rValue = model->BSIM3V2uc1; 
            return(OK);
        case BSIM3V2_MOD_U0:
            value->rValue = model->BSIM3V2u0;
            return(OK);
        case BSIM3V2_MOD_UTE:
            value->rValue = model->BSIM3V2ute;
            return(OK);
        case BSIM3V2_MOD_VOFF:
            value->rValue = model->BSIM3V2voff;
            return(OK);
        case BSIM3V2_MOD_DELTA:
            value->rValue = model->BSIM3V2delta;
            return(OK);
        case BSIM3V2_MOD_RDSW:
            value->rValue = model->BSIM3V2rdsw; 
            return(OK);             
        case BSIM3V2_MOD_PRWG:
            value->rValue = model->BSIM3V2prwg; 
            return(OK);             
        case BSIM3V2_MOD_PRWB:
            value->rValue = model->BSIM3V2prwb; 
            return(OK);             
        case BSIM3V2_MOD_PRT:
            value->rValue = model->BSIM3V2prt; 
            return(OK);              
        case BSIM3V2_MOD_ETA0:
            value->rValue = model->BSIM3V2eta0; 
            return(OK);               
        case BSIM3V2_MOD_ETAB:
            value->rValue = model->BSIM3V2etab; 
            return(OK);               
        case BSIM3V2_MOD_PCLM:
            value->rValue = model->BSIM3V2pclm; 
            return(OK);               
        case BSIM3V2_MOD_PDIBL1:
            value->rValue = model->BSIM3V2pdibl1; 
            return(OK);               
        case BSIM3V2_MOD_PDIBL2:
            value->rValue = model->BSIM3V2pdibl2; 
            return(OK);               
        case BSIM3V2_MOD_PDIBLB:
            value->rValue = model->BSIM3V2pdiblb; 
            return(OK);               
        case BSIM3V2_MOD_PSCBE1:
            value->rValue = model->BSIM3V2pscbe1; 
            return(OK);               
        case BSIM3V2_MOD_PSCBE2:
            value->rValue = model->BSIM3V2pscbe2; 
            return(OK);               
        case BSIM3V2_MOD_PVAG:
            value->rValue = model->BSIM3V2pvag; 
            return(OK);               
        case BSIM3V2_MOD_WR:
            value->rValue = model->BSIM3V2wr;
            return(OK);
        case BSIM3V2_MOD_DWG:
            value->rValue = model->BSIM3V2dwg;
            return(OK);
        case BSIM3V2_MOD_DWB:
            value->rValue = model->BSIM3V2dwb;
            return(OK);
        case BSIM3V2_MOD_B0:
            value->rValue = model->BSIM3V2b0;
            return(OK);
        case BSIM3V2_MOD_B1:
            value->rValue = model->BSIM3V2b1;
            return(OK);
        case BSIM3V2_MOD_ALPHA0:
            value->rValue = model->BSIM3V2alpha0;
            return(OK);
        case BSIM3V2_MOD_ALPHA1:
            value->rValue = model->BSIM3V2alpha1;
            return(OK);
        case BSIM3V2_MOD_BETA0:
            value->rValue = model->BSIM3V2beta0;
            return(OK);
        case BSIM3V2_MOD_IJTH:
            value->rValue = model->BSIM3V2ijth;
            return(OK);
        case BSIM3V2_MOD_VFB:
            value->rValue = model->BSIM3V2vfb;
            return(OK);

        case BSIM3V2_MOD_ELM:
            value->rValue = model->BSIM3V2elm;
            return(OK);
        case BSIM3V2_MOD_CGSL:
            value->rValue = model->BSIM3V2cgsl;
            return(OK);
        case BSIM3V2_MOD_CGDL:
            value->rValue = model->BSIM3V2cgdl;
            return(OK);
        case BSIM3V2_MOD_CKAPPA:
            value->rValue = model->BSIM3V2ckappa;
            return(OK);
        case BSIM3V2_MOD_CF:
            value->rValue = model->BSIM3V2cf;
            return(OK);
        case BSIM3V2_MOD_CLC:
            value->rValue = model->BSIM3V2clc;
            return(OK);
        case BSIM3V2_MOD_CLE:
            value->rValue = model->BSIM3V2cle;
            return(OK);
        case BSIM3V2_MOD_DWC:
            value->rValue = model->BSIM3V2dwc;
            return(OK);
        case BSIM3V2_MOD_DLC:
            value->rValue = model->BSIM3V2dlc;
            return(OK);
        case BSIM3V2_MOD_VFBCV:
            value->rValue = model->BSIM3V2vfbcv; 
            return(OK);
        case BSIM3V2_MOD_ACDE:
            value->rValue = model->BSIM3V2acde;
            return(OK);
        case BSIM3V2_MOD_MOIN:
            value->rValue = model->BSIM3V2moin;
            return(OK);
        case BSIM3V2_MOD_NOFF:
            value->rValue = model->BSIM3V2noff;
            return(OK);
        case BSIM3V2_MOD_VOFFCV:
            value->rValue = model->BSIM3V2voffcv;
            return(OK);
        case BSIM3V2_MOD_TCJ:
            value->rValue = model->BSIM3V2tcj;
            return(OK);
        case BSIM3V2_MOD_TPB:
            value->rValue = model->BSIM3V2tpb;
            return(OK);
        case BSIM3V2_MOD_TCJSW:
            value->rValue = model->BSIM3V2tcjsw;
            return(OK);
        case BSIM3V2_MOD_TPBSW:
            value->rValue = model->BSIM3V2tpbsw;
            return(OK);
        case BSIM3V2_MOD_TCJSWG:
            value->rValue = model->BSIM3V2tcjswg;
            return(OK);
        case BSIM3V2_MOD_TPBSWG:
            value->rValue = model->BSIM3V2tpbswg;
            return(OK);

	/* Length dependence */
        case  BSIM3V2_MOD_LCDSC :
          value->rValue = model->BSIM3V2lcdsc;
            return(OK);
        case  BSIM3V2_MOD_LCDSCB :
          value->rValue = model->BSIM3V2lcdscb;
            return(OK);
        case  BSIM3V2_MOD_LCDSCD :
          value->rValue = model->BSIM3V2lcdscd;
            return(OK);
        case  BSIM3V2_MOD_LCIT :
          value->rValue = model->BSIM3V2lcit;
            return(OK);
        case  BSIM3V2_MOD_LNFACTOR :
          value->rValue = model->BSIM3V2lnfactor;
            return(OK);
        case BSIM3V2_MOD_LXJ:
            value->rValue = model->BSIM3V2lxj;
            return(OK);
        case BSIM3V2_MOD_LVSAT:
            value->rValue = model->BSIM3V2lvsat;
            return(OK);
        case BSIM3V2_MOD_LAT:
            value->rValue = model->BSIM3V2lat;
            return(OK);
        case BSIM3V2_MOD_LA0:
            value->rValue = model->BSIM3V2la0;
            return(OK);
        case BSIM3V2_MOD_LAGS:
            value->rValue = model->BSIM3V2lags;
            return(OK);
        case BSIM3V2_MOD_LA1:
            value->rValue = model->BSIM3V2la1;
            return(OK);
        case BSIM3V2_MOD_LA2:
            value->rValue = model->BSIM3V2la2;
            return(OK);
        case BSIM3V2_MOD_LKETA:
            value->rValue = model->BSIM3V2lketa;
            return(OK);   
        case BSIM3V2_MOD_LNSUB:
            value->rValue = model->BSIM3V2lnsub;
            return(OK);
        case BSIM3V2_MOD_LNPEAK:
            value->rValue = model->BSIM3V2lnpeak;
            return(OK);
        case BSIM3V2_MOD_LNGATE:
            value->rValue = model->BSIM3V2lngate;
            return(OK);
        case BSIM3V2_MOD_LGAMMA1:
            value->rValue = model->BSIM3V2lgamma1;
            return(OK);
        case BSIM3V2_MOD_LGAMMA2:
            value->rValue = model->BSIM3V2lgamma2;
            return(OK);
        case BSIM3V2_MOD_LVBX:
            value->rValue = model->BSIM3V2lvbx;
            return(OK);
        case BSIM3V2_MOD_LVBM:
            value->rValue = model->BSIM3V2lvbm;
            return(OK);
        case BSIM3V2_MOD_LXT:
            value->rValue = model->BSIM3V2lxt;
            return(OK);
        case  BSIM3V2_MOD_LK1:
          value->rValue = model->BSIM3V2lk1;
            return(OK);
        case  BSIM3V2_MOD_LKT1:
          value->rValue = model->BSIM3V2lkt1;
            return(OK);
        case  BSIM3V2_MOD_LKT1L:
          value->rValue = model->BSIM3V2lkt1l;
            return(OK);
        case  BSIM3V2_MOD_LKT2 :
          value->rValue = model->BSIM3V2lkt2;
            return(OK);
        case  BSIM3V2_MOD_LK2 :
          value->rValue = model->BSIM3V2lk2;
            return(OK);
        case  BSIM3V2_MOD_LK3:
          value->rValue = model->BSIM3V2lk3;
            return(OK);
        case  BSIM3V2_MOD_LK3B:
          value->rValue = model->BSIM3V2lk3b;
            return(OK);
        case  BSIM3V2_MOD_LW0:
          value->rValue = model->BSIM3V2lw0;
            return(OK);
        case  BSIM3V2_MOD_LNLX:
          value->rValue = model->BSIM3V2lnlx;
            return(OK);
        case  BSIM3V2_MOD_LDVT0:                
          value->rValue = model->BSIM3V2ldvt0;
            return(OK);
        case  BSIM3V2_MOD_LDVT1 :             
          value->rValue = model->BSIM3V2ldvt1;
            return(OK);
        case  BSIM3V2_MOD_LDVT2 :             
          value->rValue = model->BSIM3V2ldvt2;
            return(OK);
        case  BSIM3V2_MOD_LDVT0W :                
          value->rValue = model->BSIM3V2ldvt0w;
            return(OK);
        case  BSIM3V2_MOD_LDVT1W :             
          value->rValue = model->BSIM3V2ldvt1w;
            return(OK);
        case  BSIM3V2_MOD_LDVT2W :             
          value->rValue = model->BSIM3V2ldvt2w;
            return(OK);
        case  BSIM3V2_MOD_LDROUT :           
          value->rValue = model->BSIM3V2ldrout;
            return(OK);
        case  BSIM3V2_MOD_LDSUB :           
          value->rValue = model->BSIM3V2ldsub;
            return(OK);
        case BSIM3V2_MOD_LVTH0:
            value->rValue = model->BSIM3V2lvth0; 
            return(OK);
        case BSIM3V2_MOD_LUA:
            value->rValue = model->BSIM3V2lua; 
            return(OK);
        case BSIM3V2_MOD_LUA1:
            value->rValue = model->BSIM3V2lua1; 
            return(OK);
        case BSIM3V2_MOD_LUB:
            value->rValue = model->BSIM3V2lub;  
            return(OK);
        case BSIM3V2_MOD_LUB1:
            value->rValue = model->BSIM3V2lub1;  
            return(OK);
        case BSIM3V2_MOD_LUC:
            value->rValue = model->BSIM3V2luc; 
            return(OK);
        case BSIM3V2_MOD_LUC1:
            value->rValue = model->BSIM3V2luc1; 
            return(OK);
        case BSIM3V2_MOD_LU0:
            value->rValue = model->BSIM3V2lu0;
            return(OK);
        case BSIM3V2_MOD_LUTE:
            value->rValue = model->BSIM3V2lute;
            return(OK);
        case BSIM3V2_MOD_LVOFF:
            value->rValue = model->BSIM3V2lvoff;
            return(OK);
        case BSIM3V2_MOD_LDELTA:
            value->rValue = model->BSIM3V2ldelta;
            return(OK);
        case BSIM3V2_MOD_LRDSW:
            value->rValue = model->BSIM3V2lrdsw; 
            return(OK);             
        case BSIM3V2_MOD_LPRWB:
            value->rValue = model->BSIM3V2lprwb; 
            return(OK);             
        case BSIM3V2_MOD_LPRWG:
            value->rValue = model->BSIM3V2lprwg; 
            return(OK);             
        case BSIM3V2_MOD_LPRT:
            value->rValue = model->BSIM3V2lprt; 
            return(OK);              
        case BSIM3V2_MOD_LETA0:
            value->rValue = model->BSIM3V2leta0; 
            return(OK);               
        case BSIM3V2_MOD_LETAB:
            value->rValue = model->BSIM3V2letab; 
            return(OK);               
        case BSIM3V2_MOD_LPCLM:
            value->rValue = model->BSIM3V2lpclm; 
            return(OK);               
        case BSIM3V2_MOD_LPDIBL1:
            value->rValue = model->BSIM3V2lpdibl1; 
            return(OK);               
        case BSIM3V2_MOD_LPDIBL2:
            value->rValue = model->BSIM3V2lpdibl2; 
            return(OK);               
        case BSIM3V2_MOD_LPDIBLB:
            value->rValue = model->BSIM3V2lpdiblb; 
            return(OK);               
        case BSIM3V2_MOD_LPSCBE1:
            value->rValue = model->BSIM3V2lpscbe1; 
            return(OK);               
        case BSIM3V2_MOD_LPSCBE2:
            value->rValue = model->BSIM3V2lpscbe2; 
            return(OK);               
        case BSIM3V2_MOD_LPVAG:
            value->rValue = model->BSIM3V2lpvag; 
            return(OK);               
        case BSIM3V2_MOD_LWR:
            value->rValue = model->BSIM3V2lwr;
            return(OK);
        case BSIM3V2_MOD_LDWG:
            value->rValue = model->BSIM3V2ldwg;
            return(OK);
        case BSIM3V2_MOD_LDWB:
            value->rValue = model->BSIM3V2ldwb;
            return(OK);
        case BSIM3V2_MOD_LB0:
            value->rValue = model->BSIM3V2lb0;
            return(OK);
        case BSIM3V2_MOD_LB1:
            value->rValue = model->BSIM3V2lb1;
            return(OK);
        case BSIM3V2_MOD_LALPHA0:
            value->rValue = model->BSIM3V2lalpha0;
            return(OK);
        case BSIM3V2_MOD_LALPHA1:
            value->rValue = model->BSIM3V2lalpha1;
            return(OK);
        case BSIM3V2_MOD_LBETA0:
            value->rValue = model->BSIM3V2lbeta0;
            return(OK);
        case BSIM3V2_MOD_LVFB:
            value->rValue = model->BSIM3V2lvfb;
            return(OK);

        case BSIM3V2_MOD_LELM:
            value->rValue = model->BSIM3V2lelm;
            return(OK);
        case BSIM3V2_MOD_LCGSL:
            value->rValue = model->BSIM3V2lcgsl;
            return(OK);
        case BSIM3V2_MOD_LCGDL:
            value->rValue = model->BSIM3V2lcgdl;
            return(OK);
        case BSIM3V2_MOD_LCKAPPA:
            value->rValue = model->BSIM3V2lckappa;
            return(OK);
        case BSIM3V2_MOD_LCF:
            value->rValue = model->BSIM3V2lcf;
            return(OK);
        case BSIM3V2_MOD_LCLC:
            value->rValue = model->BSIM3V2lclc;
            return(OK);
        case BSIM3V2_MOD_LCLE:
            value->rValue = model->BSIM3V2lcle;
            return(OK);
        case BSIM3V2_MOD_LVFBCV:
            value->rValue = model->BSIM3V2lvfbcv;
            return(OK);
        case BSIM3V2_MOD_LACDE:
            value->rValue = model->BSIM3V2lacde;
            return(OK);
        case BSIM3V2_MOD_LMOIN:
            value->rValue = model->BSIM3V2lmoin;
            return(OK);
        case BSIM3V2_MOD_LNOFF:
            value->rValue = model->BSIM3V2lnoff;
            return(OK);
        case BSIM3V2_MOD_LVOFFCV:
            value->rValue = model->BSIM3V2lvoffcv;
            return(OK);

	/* Width dependence */
        case  BSIM3V2_MOD_WCDSC :
          value->rValue = model->BSIM3V2wcdsc;
            return(OK);
        case  BSIM3V2_MOD_WCDSCB :
          value->rValue = model->BSIM3V2wcdscb;
            return(OK);
        case  BSIM3V2_MOD_WCDSCD :
          value->rValue = model->BSIM3V2wcdscd;
            return(OK);
        case  BSIM3V2_MOD_WCIT :
          value->rValue = model->BSIM3V2wcit;
            return(OK);
        case  BSIM3V2_MOD_WNFACTOR :
          value->rValue = model->BSIM3V2wnfactor;
            return(OK);
        case BSIM3V2_MOD_WXJ:
            value->rValue = model->BSIM3V2wxj;
            return(OK);
        case BSIM3V2_MOD_WVSAT:
            value->rValue = model->BSIM3V2wvsat;
            return(OK);
        case BSIM3V2_MOD_WAT:
            value->rValue = model->BSIM3V2wat;
            return(OK);
        case BSIM3V2_MOD_WA0:
            value->rValue = model->BSIM3V2wa0;
            return(OK);
        case BSIM3V2_MOD_WAGS:
            value->rValue = model->BSIM3V2wags;
            return(OK);
        case BSIM3V2_MOD_WA1:
            value->rValue = model->BSIM3V2wa1;
            return(OK);
        case BSIM3V2_MOD_WA2:
            value->rValue = model->BSIM3V2wa2;
            return(OK);
        case BSIM3V2_MOD_WKETA:
            value->rValue = model->BSIM3V2wketa;
            return(OK);   
        case BSIM3V2_MOD_WNSUB:
            value->rValue = model->BSIM3V2wnsub;
            return(OK);
        case BSIM3V2_MOD_WNPEAK:
            value->rValue = model->BSIM3V2wnpeak;
            return(OK);
        case BSIM3V2_MOD_WNGATE:
            value->rValue = model->BSIM3V2wngate;
            return(OK);
        case BSIM3V2_MOD_WGAMMA1:
            value->rValue = model->BSIM3V2wgamma1;
            return(OK);
        case BSIM3V2_MOD_WGAMMA2:
            value->rValue = model->BSIM3V2wgamma2;
            return(OK);
        case BSIM3V2_MOD_WVBX:
            value->rValue = model->BSIM3V2wvbx;
            return(OK);
        case BSIM3V2_MOD_WVBM:
            value->rValue = model->BSIM3V2wvbm;
            return(OK);
        case BSIM3V2_MOD_WXT:
            value->rValue = model->BSIM3V2wxt;
            return(OK);
        case  BSIM3V2_MOD_WK1:
          value->rValue = model->BSIM3V2wk1;
            return(OK);
        case  BSIM3V2_MOD_WKT1:
          value->rValue = model->BSIM3V2wkt1;
            return(OK);
        case  BSIM3V2_MOD_WKT1L:
          value->rValue = model->BSIM3V2wkt1l;
            return(OK);
        case  BSIM3V2_MOD_WKT2 :
          value->rValue = model->BSIM3V2wkt2;
            return(OK);
        case  BSIM3V2_MOD_WK2 :
          value->rValue = model->BSIM3V2wk2;
            return(OK);
        case  BSIM3V2_MOD_WK3:
          value->rValue = model->BSIM3V2wk3;
            return(OK);
        case  BSIM3V2_MOD_WK3B:
          value->rValue = model->BSIM3V2wk3b;
            return(OK);
        case  BSIM3V2_MOD_WW0:
          value->rValue = model->BSIM3V2ww0;
            return(OK);
        case  BSIM3V2_MOD_WNLX:
          value->rValue = model->BSIM3V2wnlx;
            return(OK);
        case  BSIM3V2_MOD_WDVT0:                
          value->rValue = model->BSIM3V2wdvt0;
            return(OK);
        case  BSIM3V2_MOD_WDVT1 :             
          value->rValue = model->BSIM3V2wdvt1;
            return(OK);
        case  BSIM3V2_MOD_WDVT2 :             
          value->rValue = model->BSIM3V2wdvt2;
            return(OK);
        case  BSIM3V2_MOD_WDVT0W :                
          value->rValue = model->BSIM3V2wdvt0w;
            return(OK);
        case  BSIM3V2_MOD_WDVT1W :             
          value->rValue = model->BSIM3V2wdvt1w;
            return(OK);
        case  BSIM3V2_MOD_WDVT2W :             
          value->rValue = model->BSIM3V2wdvt2w;
            return(OK);
        case  BSIM3V2_MOD_WDROUT :           
          value->rValue = model->BSIM3V2wdrout;
            return(OK);
        case  BSIM3V2_MOD_WDSUB :           
          value->rValue = model->BSIM3V2wdsub;
            return(OK);
        case BSIM3V2_MOD_WVTH0:
            value->rValue = model->BSIM3V2wvth0; 
            return(OK);
        case BSIM3V2_MOD_WUA:
            value->rValue = model->BSIM3V2wua; 
            return(OK);
        case BSIM3V2_MOD_WUA1:
            value->rValue = model->BSIM3V2wua1; 
            return(OK);
        case BSIM3V2_MOD_WUB:
            value->rValue = model->BSIM3V2wub;  
            return(OK);
        case BSIM3V2_MOD_WUB1:
            value->rValue = model->BSIM3V2wub1;  
            return(OK);
        case BSIM3V2_MOD_WUC:
            value->rValue = model->BSIM3V2wuc; 
            return(OK);
        case BSIM3V2_MOD_WUC1:
            value->rValue = model->BSIM3V2wuc1; 
            return(OK);
        case BSIM3V2_MOD_WU0:
            value->rValue = model->BSIM3V2wu0;
            return(OK);
        case BSIM3V2_MOD_WUTE:
            value->rValue = model->BSIM3V2wute;
            return(OK);
        case BSIM3V2_MOD_WVOFF:
            value->rValue = model->BSIM3V2wvoff;
            return(OK);
        case BSIM3V2_MOD_WDELTA:
            value->rValue = model->BSIM3V2wdelta;
            return(OK);
        case BSIM3V2_MOD_WRDSW:
            value->rValue = model->BSIM3V2wrdsw; 
            return(OK);             
        case BSIM3V2_MOD_WPRWB:
            value->rValue = model->BSIM3V2wprwb; 
            return(OK);             
        case BSIM3V2_MOD_WPRWG:
            value->rValue = model->BSIM3V2wprwg; 
            return(OK);             
        case BSIM3V2_MOD_WPRT:
            value->rValue = model->BSIM3V2wprt; 
            return(OK);              
        case BSIM3V2_MOD_WETA0:
            value->rValue = model->BSIM3V2weta0; 
            return(OK);               
        case BSIM3V2_MOD_WETAB:
            value->rValue = model->BSIM3V2wetab; 
            return(OK);               
        case BSIM3V2_MOD_WPCLM:
            value->rValue = model->BSIM3V2wpclm; 
            return(OK);               
        case BSIM3V2_MOD_WPDIBL1:
            value->rValue = model->BSIM3V2wpdibl1; 
            return(OK);               
        case BSIM3V2_MOD_WPDIBL2:
            value->rValue = model->BSIM3V2wpdibl2; 
            return(OK);               
        case BSIM3V2_MOD_WPDIBLB:
            value->rValue = model->BSIM3V2wpdiblb; 
            return(OK);               
        case BSIM3V2_MOD_WPSCBE1:
            value->rValue = model->BSIM3V2wpscbe1; 
            return(OK);               
        case BSIM3V2_MOD_WPSCBE2:
            value->rValue = model->BSIM3V2wpscbe2; 
            return(OK);               
        case BSIM3V2_MOD_WPVAG:
            value->rValue = model->BSIM3V2wpvag; 
            return(OK);               
        case BSIM3V2_MOD_WWR:
            value->rValue = model->BSIM3V2wwr;
            return(OK);
        case BSIM3V2_MOD_WDWG:
            value->rValue = model->BSIM3V2wdwg;
            return(OK);
        case BSIM3V2_MOD_WDWB:
            value->rValue = model->BSIM3V2wdwb;
            return(OK);
        case BSIM3V2_MOD_WB0:
            value->rValue = model->BSIM3V2wb0;
            return(OK);
        case BSIM3V2_MOD_WB1:
            value->rValue = model->BSIM3V2wb1;
            return(OK);
        case BSIM3V2_MOD_WALPHA0:
            value->rValue = model->BSIM3V2walpha0;
            return(OK);
        case BSIM3V2_MOD_WALPHA1:
            value->rValue = model->BSIM3V2walpha1;
            return(OK);
        case BSIM3V2_MOD_WBETA0:
            value->rValue = model->BSIM3V2wbeta0;
            return(OK);
        case BSIM3V2_MOD_WVFB:
            value->rValue = model->BSIM3V2wvfb;
            return(OK);

        case BSIM3V2_MOD_WELM:
            value->rValue = model->BSIM3V2welm;
            return(OK);
        case BSIM3V2_MOD_WCGSL:
            value->rValue = model->BSIM3V2wcgsl;
            return(OK);
        case BSIM3V2_MOD_WCGDL:
            value->rValue = model->BSIM3V2wcgdl;
            return(OK);
        case BSIM3V2_MOD_WCKAPPA:
            value->rValue = model->BSIM3V2wckappa;
            return(OK);
        case BSIM3V2_MOD_WCF:
            value->rValue = model->BSIM3V2wcf;
            return(OK);
        case BSIM3V2_MOD_WCLC:
            value->rValue = model->BSIM3V2wclc;
            return(OK);
        case BSIM3V2_MOD_WCLE:
            value->rValue = model->BSIM3V2wcle;
            return(OK);
        case BSIM3V2_MOD_WVFBCV:
            value->rValue = model->BSIM3V2wvfbcv;
            return(OK);
        case BSIM3V2_MOD_WACDE:
            value->rValue = model->BSIM3V2wacde;
            return(OK);
        case BSIM3V2_MOD_WMOIN:
            value->rValue = model->BSIM3V2wmoin;
            return(OK);
        case BSIM3V2_MOD_WNOFF:
            value->rValue = model->BSIM3V2wnoff;
            return(OK);
        case BSIM3V2_MOD_WVOFFCV:
            value->rValue = model->BSIM3V2wvoffcv;
            return(OK);

	/* Cross-term dependence */
        case  BSIM3V2_MOD_PCDSC :
          value->rValue = model->BSIM3V2pcdsc;
            return(OK);
        case  BSIM3V2_MOD_PCDSCB :
          value->rValue = model->BSIM3V2pcdscb;
            return(OK);
        case  BSIM3V2_MOD_PCDSCD :
          value->rValue = model->BSIM3V2pcdscd;
            return(OK);
         case  BSIM3V2_MOD_PCIT :
          value->rValue = model->BSIM3V2pcit;
            return(OK);
        case  BSIM3V2_MOD_PNFACTOR :
          value->rValue = model->BSIM3V2pnfactor;
            return(OK);
        case BSIM3V2_MOD_PXJ:
            value->rValue = model->BSIM3V2pxj;
            return(OK);
        case BSIM3V2_MOD_PVSAT:
            value->rValue = model->BSIM3V2pvsat;
            return(OK);
        case BSIM3V2_MOD_PAT:
            value->rValue = model->BSIM3V2pat;
            return(OK);
        case BSIM3V2_MOD_PA0:
            value->rValue = model->BSIM3V2pa0;
            return(OK);
        case BSIM3V2_MOD_PAGS:
            value->rValue = model->BSIM3V2pags;
            return(OK);
        case BSIM3V2_MOD_PA1:
            value->rValue = model->BSIM3V2pa1;
            return(OK);
        case BSIM3V2_MOD_PA2:
            value->rValue = model->BSIM3V2pa2;
            return(OK);
        case BSIM3V2_MOD_PKETA:
            value->rValue = model->BSIM3V2pketa;
            return(OK);   
        case BSIM3V2_MOD_PNSUB:
            value->rValue = model->BSIM3V2pnsub;
            return(OK);
        case BSIM3V2_MOD_PNPEAK:
            value->rValue = model->BSIM3V2pnpeak;
            return(OK);
        case BSIM3V2_MOD_PNGATE:
            value->rValue = model->BSIM3V2pngate;
            return(OK);
        case BSIM3V2_MOD_PGAMMA1:
            value->rValue = model->BSIM3V2pgamma1;
            return(OK);
        case BSIM3V2_MOD_PGAMMA2:
            value->rValue = model->BSIM3V2pgamma2;
            return(OK);
        case BSIM3V2_MOD_PVBX:
            value->rValue = model->BSIM3V2pvbx;
            return(OK);
        case BSIM3V2_MOD_PVBM:
            value->rValue = model->BSIM3V2pvbm;
            return(OK);
        case BSIM3V2_MOD_PXT:
            value->rValue = model->BSIM3V2pxt;
            return(OK);
        case  BSIM3V2_MOD_PK1:
          value->rValue = model->BSIM3V2pk1;
            return(OK);
        case  BSIM3V2_MOD_PKT1:
          value->rValue = model->BSIM3V2pkt1;
            return(OK);
        case  BSIM3V2_MOD_PKT1L:
          value->rValue = model->BSIM3V2pkt1l;
            return(OK);
        case  BSIM3V2_MOD_PKT2 :
          value->rValue = model->BSIM3V2pkt2;
            return(OK);
        case  BSIM3V2_MOD_PK2 :
          value->rValue = model->BSIM3V2pk2;
            return(OK);
        case  BSIM3V2_MOD_PK3:
          value->rValue = model->BSIM3V2pk3;
            return(OK);
        case  BSIM3V2_MOD_PK3B:
          value->rValue = model->BSIM3V2pk3b;
            return(OK);
        case  BSIM3V2_MOD_PW0:
          value->rValue = model->BSIM3V2pw0;
            return(OK);
        case  BSIM3V2_MOD_PNLX:
          value->rValue = model->BSIM3V2pnlx;
            return(OK);
        case  BSIM3V2_MOD_PDVT0 :                
          value->rValue = model->BSIM3V2pdvt0;
            return(OK);
        case  BSIM3V2_MOD_PDVT1 :             
          value->rValue = model->BSIM3V2pdvt1;
            return(OK);
        case  BSIM3V2_MOD_PDVT2 :             
          value->rValue = model->BSIM3V2pdvt2;
            return(OK);
        case  BSIM3V2_MOD_PDVT0W :                
          value->rValue = model->BSIM3V2pdvt0w;
            return(OK);
        case  BSIM3V2_MOD_PDVT1W :             
          value->rValue = model->BSIM3V2pdvt1w;
            return(OK);
        case  BSIM3V2_MOD_PDVT2W :             
          value->rValue = model->BSIM3V2pdvt2w;
            return(OK);
        case  BSIM3V2_MOD_PDROUT :           
          value->rValue = model->BSIM3V2pdrout;
            return(OK);
        case  BSIM3V2_MOD_PDSUB :           
          value->rValue = model->BSIM3V2pdsub;
            return(OK);
        case BSIM3V2_MOD_PVTH0:
            value->rValue = model->BSIM3V2pvth0; 
            return(OK);
        case BSIM3V2_MOD_PUA:
            value->rValue = model->BSIM3V2pua; 
            return(OK);
        case BSIM3V2_MOD_PUA1:
            value->rValue = model->BSIM3V2pua1; 
            return(OK);
        case BSIM3V2_MOD_PUB:
            value->rValue = model->BSIM3V2pub;  
            return(OK);
        case BSIM3V2_MOD_PUB1:
            value->rValue = model->BSIM3V2pub1;  
            return(OK);
        case BSIM3V2_MOD_PUC:
            value->rValue = model->BSIM3V2puc; 
            return(OK);
        case BSIM3V2_MOD_PUC1:
            value->rValue = model->BSIM3V2puc1; 
            return(OK);
        case BSIM3V2_MOD_PU0:
            value->rValue = model->BSIM3V2pu0;
            return(OK);
        case BSIM3V2_MOD_PUTE:
            value->rValue = model->BSIM3V2pute;
            return(OK);
        case BSIM3V2_MOD_PVOFF:
            value->rValue = model->BSIM3V2pvoff;
            return(OK);
        case BSIM3V2_MOD_PDELTA:
            value->rValue = model->BSIM3V2pdelta;
            return(OK);
        case BSIM3V2_MOD_PRDSW:
            value->rValue = model->BSIM3V2prdsw; 
            return(OK);             
        case BSIM3V2_MOD_PPRWB:
            value->rValue = model->BSIM3V2pprwb; 
            return(OK);             
        case BSIM3V2_MOD_PPRWG:
            value->rValue = model->BSIM3V2pprwg; 
            return(OK);             
        case BSIM3V2_MOD_PPRT:
            value->rValue = model->BSIM3V2pprt; 
            return(OK);              
        case BSIM3V2_MOD_PETA0:
            value->rValue = model->BSIM3V2peta0; 
            return(OK);               
        case BSIM3V2_MOD_PETAB:
            value->rValue = model->BSIM3V2petab; 
            return(OK);               
        case BSIM3V2_MOD_PPCLM:
            value->rValue = model->BSIM3V2ppclm; 
            return(OK);               
        case BSIM3V2_MOD_PPDIBL1:
            value->rValue = model->BSIM3V2ppdibl1; 
            return(OK);               
        case BSIM3V2_MOD_PPDIBL2:
            value->rValue = model->BSIM3V2ppdibl2; 
            return(OK);               
        case BSIM3V2_MOD_PPDIBLB:
            value->rValue = model->BSIM3V2ppdiblb; 
            return(OK);               
        case BSIM3V2_MOD_PPSCBE1:
            value->rValue = model->BSIM3V2ppscbe1; 
            return(OK);               
        case BSIM3V2_MOD_PPSCBE2:
            value->rValue = model->BSIM3V2ppscbe2; 
            return(OK);               
        case BSIM3V2_MOD_PPVAG:
            value->rValue = model->BSIM3V2ppvag; 
            return(OK);               
        case BSIM3V2_MOD_PWR:
            value->rValue = model->BSIM3V2pwr;
            return(OK);
        case BSIM3V2_MOD_PDWG:
            value->rValue = model->BSIM3V2pdwg;
            return(OK);
        case BSIM3V2_MOD_PDWB:
            value->rValue = model->BSIM3V2pdwb;
            return(OK);
        case BSIM3V2_MOD_PB0:
            value->rValue = model->BSIM3V2pb0;
            return(OK);
        case BSIM3V2_MOD_PB1:
            value->rValue = model->BSIM3V2pb1;
            return(OK);
        case BSIM3V2_MOD_PALPHA0:
            value->rValue = model->BSIM3V2palpha0;
            return(OK);
        case BSIM3V2_MOD_PALPHA1:
            value->rValue = model->BSIM3V2palpha1;
            return(OK);
        case BSIM3V2_MOD_PBETA0:
            value->rValue = model->BSIM3V2pbeta0;
            return(OK);
        case BSIM3V2_MOD_PVFB:
            value->rValue = model->BSIM3V2pvfb;
            return(OK);

        case BSIM3V2_MOD_PELM:
            value->rValue = model->BSIM3V2pelm;
            return(OK);
        case BSIM3V2_MOD_PCGSL:
            value->rValue = model->BSIM3V2pcgsl;
            return(OK);
        case BSIM3V2_MOD_PCGDL:
            value->rValue = model->BSIM3V2pcgdl;
            return(OK);
        case BSIM3V2_MOD_PCKAPPA:
            value->rValue = model->BSIM3V2pckappa;
            return(OK);
        case BSIM3V2_MOD_PCF:
            value->rValue = model->BSIM3V2pcf;
            return(OK);
        case BSIM3V2_MOD_PCLC:
            value->rValue = model->BSIM3V2pclc;
            return(OK);
        case BSIM3V2_MOD_PCLE:
            value->rValue = model->BSIM3V2pcle;
            return(OK);
        case BSIM3V2_MOD_PVFBCV:
            value->rValue = model->BSIM3V2pvfbcv;
            return(OK);
        case BSIM3V2_MOD_PACDE:
            value->rValue = model->BSIM3V2pacde;
            return(OK);
        case BSIM3V2_MOD_PMOIN:
            value->rValue = model->BSIM3V2pmoin;
            return(OK);
        case BSIM3V2_MOD_PNOFF:
            value->rValue = model->BSIM3V2pnoff;
            return(OK);
        case BSIM3V2_MOD_PVOFFCV:
            value->rValue = model->BSIM3V2pvoffcv;
            return(OK);

        case  BSIM3V2_MOD_TNOM :
          value->rValue = model->BSIM3V2tnom;
            return(OK);
        case BSIM3V2_MOD_CGSO:
            value->rValue = model->BSIM3V2cgso; 
            return(OK);
        case BSIM3V2_MOD_CGDO:
            value->rValue = model->BSIM3V2cgdo; 
            return(OK);
        case BSIM3V2_MOD_CGBO:
            value->rValue = model->BSIM3V2cgbo; 
            return(OK);
        case BSIM3V2_MOD_XPART:
            value->rValue = model->BSIM3V2xpart; 
            return(OK);
        case BSIM3V2_MOD_RSH:
            value->rValue = model->BSIM3V2sheetResistance; 
            return(OK);
        case BSIM3V2_MOD_JS:
            value->rValue = model->BSIM3V2jctSatCurDensity; 
            return(OK);
        case BSIM3V2_MOD_JSW:
            value->rValue = model->BSIM3V2jctSidewallSatCurDensity; 
            return(OK);
        case BSIM3V2_MOD_PB:
            value->rValue = model->BSIM3V2bulkJctPotential; 
            return(OK);
        case BSIM3V2_MOD_MJ:
            value->rValue = model->BSIM3V2bulkJctBotGradingCoeff; 
            return(OK);
        case BSIM3V2_MOD_PBSW:
            value->rValue = model->BSIM3V2sidewallJctPotential; 
            return(OK);
        case BSIM3V2_MOD_MJSW:
            value->rValue = model->BSIM3V2bulkJctSideGradingCoeff; 
            return(OK);
        case BSIM3V2_MOD_CJ:
            value->rValue = model->BSIM3V2unitAreaJctCap; 
            return(OK);
        case BSIM3V2_MOD_CJSW:
            value->rValue = model->BSIM3V2unitLengthSidewallJctCap; 
            return(OK);
        case BSIM3V2_MOD_PBSWG:
            value->rValue = model->BSIM3V2GatesidewallJctPotential; 
            return(OK);
        case BSIM3V2_MOD_MJSWG:
            value->rValue = model->BSIM3V2bulkJctGateSideGradingCoeff; 
            return(OK);
        case BSIM3V2_MOD_CJSWG:
            value->rValue = model->BSIM3V2unitLengthGateSidewallJctCap; 
            return(OK);
        case BSIM3V2_MOD_NJ:
            value->rValue = model->BSIM3V2jctEmissionCoeff; 
            return(OK);
        case BSIM3V2_MOD_XTI:
            value->rValue = model->BSIM3V2jctTempExponent; 
            return(OK);
        case BSIM3V2_MOD_LINT:
            value->rValue = model->BSIM3V2Lint; 
            return(OK);
        case BSIM3V2_MOD_LL:
            value->rValue = model->BSIM3V2Ll;
            return(OK);
        case BSIM3V2_MOD_LLC:
            value->rValue = model->BSIM3V2Llc;
            return(OK);
        case BSIM3V2_MOD_LLN:
            value->rValue = model->BSIM3V2Lln;
            return(OK);
        case BSIM3V2_MOD_LW:
            value->rValue = model->BSIM3V2Lw;
            return(OK);
        case BSIM3V2_MOD_LWC:
            value->rValue = model->BSIM3V2Lwc;
            return(OK);
        case BSIM3V2_MOD_LWN:
            value->rValue = model->BSIM3V2Lwn;
            return(OK);
        case BSIM3V2_MOD_LWL:
            value->rValue = model->BSIM3V2Lwl;
            return(OK);
        case BSIM3V2_MOD_LWLC:
            value->rValue = model->BSIM3V2Lwlc;
            return(OK);
        case BSIM3V2_MOD_LMIN:
            value->rValue = model->BSIM3V2Lmin;
            return(OK);
        case BSIM3V2_MOD_LMAX:
            value->rValue = model->BSIM3V2Lmax;
            return(OK);
        case BSIM3V2_MOD_WINT:
            value->rValue = model->BSIM3V2Wint;
            return(OK);
        case BSIM3V2_MOD_WL:
            value->rValue = model->BSIM3V2Wl;
            return(OK);
        case BSIM3V2_MOD_WLC:
            value->rValue = model->BSIM3V2Wlc;
            return(OK);
        case BSIM3V2_MOD_WLN:
            value->rValue = model->BSIM3V2Wln;
            return(OK);
        case BSIM3V2_MOD_WW:
            value->rValue = model->BSIM3V2Ww;
            return(OK);
        case BSIM3V2_MOD_WWC:
            value->rValue = model->BSIM3V2Wwc;
            return(OK);
        case BSIM3V2_MOD_WWN:
            value->rValue = model->BSIM3V2Wwn;
            return(OK);
        case BSIM3V2_MOD_WWL:
            value->rValue = model->BSIM3V2Wwl;
            return(OK);
        case BSIM3V2_MOD_WWLC:
            value->rValue = model->BSIM3V2Wwlc;
            return(OK);
        case BSIM3V2_MOD_WMIN:
            value->rValue = model->BSIM3V2Wmin;
            return(OK);
        case BSIM3V2_MOD_WMAX:
            value->rValue = model->BSIM3V2Wmax;
            return(OK);
        case BSIM3V2_MOD_NOIA:
            value->rValue = model->BSIM3V2oxideTrapDensityA;
            return(OK);
        case BSIM3V2_MOD_NOIB:
            value->rValue = model->BSIM3V2oxideTrapDensityB;
            return(OK);
        case BSIM3V2_MOD_NOIC:
            value->rValue = model->BSIM3V2oxideTrapDensityC;
            return(OK);
        case BSIM3V2_MOD_EM:
            value->rValue = model->BSIM3V2em;
            return(OK);
        case BSIM3V2_MOD_EF:
            value->rValue = model->BSIM3V2ef;
            return(OK);
        case BSIM3V2_MOD_AF:
            value->rValue = model->BSIM3V2af;
            return(OK);
        case BSIM3V2_MOD_KF:
            value->rValue = model->BSIM3V2kf;
            return(OK);
        default:
            return(E_BADPARM);
    }
    /* NOTREACHED */
}



