/* $Id$  */
/* 
$Log$
Revision 1.1  2000-04-27 20:03:59  pnenzi
Initial revision

 * Revision 3.1  96/12/08  19:55:50  yuhua
 * BSIM3v3.1 release
 * 
*/
static char rcsid[] = "$Id$";

/*************************************/

/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1995 Min-Chie Jeng and Mansun Chan.
File: b3v1mask.c
**********/

#include "ngspice.h"
#include <stdio.h>
#include "ifsim.h"
#include "cktdefs.h"
#include "devdefs.h"
#include "bsim3v1def.h"
#include "sperror.h"
#include "suffix.h"

int
BSIM3V1mAsk(ckt,inst,which,value)
CKTcircuit *ckt;
GENmodel *inst;
int which;
IFvalue *value;
{
    BSIM3V1model *model = (BSIM3V1model *)inst;
    switch(which) 
    {   case BSIM3V1_MOD_MOBMOD:
            value->iValue = model->BSIM3V1mobMod; 
            return(OK);
        case BSIM3V1_MOD_PARAMCHK:
            value->iValue = model->BSIM3V1paramChk; 
            return(OK);
        case BSIM3V1_MOD_BINUNIT:
            value->iValue = model->BSIM3V1binUnit; 
            return(OK);
        case BSIM3V1_MOD_CAPMOD:
            value->iValue = model->BSIM3V1capMod; 
            return(OK);
        case BSIM3V1_MOD_NQSMOD:
            value->iValue = model->BSIM3V1nqsMod; 
            return(OK);
        case BSIM3V1_MOD_NOIMOD:
            value->iValue = model->BSIM3V1noiMod; 
            return(OK);
        case  BSIM3V1_MOD_VERSION :
          value->rValue = model->BSIM3V1version;
            return(OK);
        case  BSIM3V1_MOD_TOX :
          value->rValue = model->BSIM3V1tox;
            return(OK);
        case  BSIM3V1_MOD_CDSC :
          value->rValue = model->BSIM3V1cdsc;
            return(OK);
        case  BSIM3V1_MOD_CDSCB :
          value->rValue = model->BSIM3V1cdscb;
            return(OK);

        case  BSIM3V1_MOD_CDSCD :
          value->rValue = model->BSIM3V1cdscd;
            return(OK);

        case  BSIM3V1_MOD_CIT :
          value->rValue = model->BSIM3V1cit;
            return(OK);
        case  BSIM3V1_MOD_NFACTOR :
          value->rValue = model->BSIM3V1nfactor;
            return(OK);
        case BSIM3V1_MOD_XJ:
            value->rValue = model->BSIM3V1xj;
            return(OK);
        case BSIM3V1_MOD_VSAT:
            value->rValue = model->BSIM3V1vsat;
            return(OK);
        case BSIM3V1_MOD_AT:
            value->rValue = model->BSIM3V1at;
            return(OK);
        case BSIM3V1_MOD_A0:
            value->rValue = model->BSIM3V1a0;
            return(OK);

        case BSIM3V1_MOD_AGS:
            value->rValue = model->BSIM3V1ags;
            return(OK);

        case BSIM3V1_MOD_A1:
            value->rValue = model->BSIM3V1a1;
            return(OK);
        case BSIM3V1_MOD_A2:
            value->rValue = model->BSIM3V1a2;
            return(OK);
        case BSIM3V1_MOD_KETA:
            value->rValue = model->BSIM3V1keta;
            return(OK);   
        case BSIM3V1_MOD_NSUB:
            value->rValue = model->BSIM3V1nsub;
            return(OK);
        case BSIM3V1_MOD_NPEAK:
            value->rValue = model->BSIM3V1npeak;
            return(OK);
        case BSIM3V1_MOD_NGATE:
            value->rValue = model->BSIM3V1ngate;
            return(OK);
        case BSIM3V1_MOD_GAMMA1:
            value->rValue = model->BSIM3V1gamma1;
            return(OK);
        case BSIM3V1_MOD_GAMMA2:
            value->rValue = model->BSIM3V1gamma2;
            return(OK);
        case BSIM3V1_MOD_VBX:
            value->rValue = model->BSIM3V1vbx;
            return(OK);
        case BSIM3V1_MOD_VBM:
            value->rValue = model->BSIM3V1vbm;
            return(OK);
        case BSIM3V1_MOD_XT:
            value->rValue = model->BSIM3V1xt;
            return(OK);
        case  BSIM3V1_MOD_K1:
          value->rValue = model->BSIM3V1k1;
            return(OK);
        case  BSIM3V1_MOD_KT1:
          value->rValue = model->BSIM3V1kt1;
            return(OK);
        case  BSIM3V1_MOD_KT1L:
          value->rValue = model->BSIM3V1kt1l;
            return(OK);
        case  BSIM3V1_MOD_KT2 :
          value->rValue = model->BSIM3V1kt2;
            return(OK);
        case  BSIM3V1_MOD_K2 :
          value->rValue = model->BSIM3V1k2;
            return(OK);
        case  BSIM3V1_MOD_K3:
          value->rValue = model->BSIM3V1k3;
            return(OK);
        case  BSIM3V1_MOD_K3B:
          value->rValue = model->BSIM3V1k3b;
            return(OK);
        case  BSIM3V1_MOD_W0:
          value->rValue = model->BSIM3V1w0;
            return(OK);
        case  BSIM3V1_MOD_NLX:
          value->rValue = model->BSIM3V1nlx;
            return(OK);
        case  BSIM3V1_MOD_DVT0 :                
          value->rValue = model->BSIM3V1dvt0;
            return(OK);
        case  BSIM3V1_MOD_DVT1 :             
          value->rValue = model->BSIM3V1dvt1;
            return(OK);
        case  BSIM3V1_MOD_DVT2 :             
          value->rValue = model->BSIM3V1dvt2;
            return(OK);
        case  BSIM3V1_MOD_DVT0W :                
          value->rValue = model->BSIM3V1dvt0w;
            return(OK);
        case  BSIM3V1_MOD_DVT1W :             
          value->rValue = model->BSIM3V1dvt1w;
            return(OK);
        case  BSIM3V1_MOD_DVT2W :             
          value->rValue = model->BSIM3V1dvt2w;
            return(OK);
        case  BSIM3V1_MOD_DROUT :           
          value->rValue = model->BSIM3V1drout;
            return(OK);
        case  BSIM3V1_MOD_DSUB :           
          value->rValue = model->BSIM3V1dsub;
            return(OK);
        case BSIM3V1_MOD_VTH0:
            value->rValue = model->BSIM3V1vth0; 
            return(OK);
        case BSIM3V1_MOD_UA:
            value->rValue = model->BSIM3V1ua; 
            return(OK);
        case BSIM3V1_MOD_UA1:
            value->rValue = model->BSIM3V1ua1; 
            return(OK);
        case BSIM3V1_MOD_UB:
            value->rValue = model->BSIM3V1ub;  
            return(OK);
        case BSIM3V1_MOD_UB1:
            value->rValue = model->BSIM3V1ub1;  
            return(OK);
        case BSIM3V1_MOD_UC:
            value->rValue = model->BSIM3V1uc; 
            return(OK);
        case BSIM3V1_MOD_UC1:
            value->rValue = model->BSIM3V1uc1; 
            return(OK);
        case BSIM3V1_MOD_U0:
            value->rValue = model->BSIM3V1u0;
            return(OK);
        case BSIM3V1_MOD_UTE:
            value->rValue = model->BSIM3V1ute;
            return(OK);
        case BSIM3V1_MOD_VOFF:
            value->rValue = model->BSIM3V1voff;
            return(OK);
        case BSIM3V1_MOD_DELTA:
            value->rValue = model->BSIM3V1delta;
            return(OK);
        case BSIM3V1_MOD_RDSW:
            value->rValue = model->BSIM3V1rdsw; 
            return(OK);             
        case BSIM3V1_MOD_PRWG:
            value->rValue = model->BSIM3V1prwg; 
            return(OK);             
        case BSIM3V1_MOD_PRWB:
            value->rValue = model->BSIM3V1prwb; 
            return(OK);             
        case BSIM3V1_MOD_PRT:
            value->rValue = model->BSIM3V1prt; 
            return(OK);              
        case BSIM3V1_MOD_ETA0:
            value->rValue = model->BSIM3V1eta0; 
            return(OK);               
        case BSIM3V1_MOD_ETAB:
            value->rValue = model->BSIM3V1etab; 
            return(OK);               
        case BSIM3V1_MOD_PCLM:
            value->rValue = model->BSIM3V1pclm; 
            return(OK);               
        case BSIM3V1_MOD_PDIBL1:
            value->rValue = model->BSIM3V1pdibl1; 
            return(OK);               
        case BSIM3V1_MOD_PDIBL2:
            value->rValue = model->BSIM3V1pdibl2; 
            return(OK);               
        case BSIM3V1_MOD_PDIBLB:
            value->rValue = model->BSIM3V1pdiblb; 
            return(OK);               
        case BSIM3V1_MOD_PSCBE1:
            value->rValue = model->BSIM3V1pscbe1; 
            return(OK);               
        case BSIM3V1_MOD_PSCBE2:
            value->rValue = model->BSIM3V1pscbe2; 
            return(OK);               
        case BSIM3V1_MOD_PVAG:
            value->rValue = model->BSIM3V1pvag; 
            return(OK);               
        case BSIM3V1_MOD_WR:
            value->rValue = model->BSIM3V1wr;
            return(OK);
        case BSIM3V1_MOD_DWG:
            value->rValue = model->BSIM3V1dwg;
            return(OK);
        case BSIM3V1_MOD_DWB:
            value->rValue = model->BSIM3V1dwb;
            return(OK);
        case BSIM3V1_MOD_B0:
            value->rValue = model->BSIM3V1b0;
            return(OK);
        case BSIM3V1_MOD_B1:
            value->rValue = model->BSIM3V1b1;
            return(OK);
        case BSIM3V1_MOD_ALPHA0:
            value->rValue = model->BSIM3V1alpha0;
            return(OK);
        case BSIM3V1_MOD_BETA0:
            value->rValue = model->BSIM3V1beta0;
            return(OK);

        case BSIM3V1_MOD_ELM:
            value->rValue = model->BSIM3V1elm;
            return(OK);
        case BSIM3V1_MOD_CGSL:
            value->rValue = model->BSIM3V1cgsl;
            return(OK);
        case BSIM3V1_MOD_CGDL:
            value->rValue = model->BSIM3V1cgdl;
            return(OK);
        case BSIM3V1_MOD_CKAPPA:
            value->rValue = model->BSIM3V1ckappa;
            return(OK);
        case BSIM3V1_MOD_CF:
            value->rValue = model->BSIM3V1cf;
            return(OK);
        case BSIM3V1_MOD_CLC:
            value->rValue = model->BSIM3V1clc;
            return(OK);
        case BSIM3V1_MOD_CLE:
            value->rValue = model->BSIM3V1cle;
            return(OK);
        case BSIM3V1_MOD_DWC:
            value->rValue = model->BSIM3V1dwc;
            return(OK);
        case BSIM3V1_MOD_DLC:
            value->rValue = model->BSIM3V1dlc;
            return(OK);
        case BSIM3V1_MOD_VFBCV:
            value->rValue = model->BSIM3V1vfbcv; 
            return(OK);

	/* Length dependence */
        case  BSIM3V1_MOD_LCDSC :
          value->rValue = model->BSIM3V1lcdsc;
            return(OK);
        case  BSIM3V1_MOD_LCDSCB :
          value->rValue = model->BSIM3V1lcdscb;
            return(OK);
        case  BSIM3V1_MOD_LCDSCD :
          value->rValue = model->BSIM3V1lcdscd;
            return(OK);
        case  BSIM3V1_MOD_LCIT :
          value->rValue = model->BSIM3V1lcit;
            return(OK);
        case  BSIM3V1_MOD_LNFACTOR :
          value->rValue = model->BSIM3V1lnfactor;
            return(OK);
        case BSIM3V1_MOD_LXJ:
            value->rValue = model->BSIM3V1lxj;
            return(OK);
        case BSIM3V1_MOD_LVSAT:
            value->rValue = model->BSIM3V1lvsat;
            return(OK);
        case BSIM3V1_MOD_LAT:
            value->rValue = model->BSIM3V1lat;
            return(OK);
        case BSIM3V1_MOD_LA0:
            value->rValue = model->BSIM3V1la0;
            return(OK);
        case BSIM3V1_MOD_LAGS:
            value->rValue = model->BSIM3V1lags;
            return(OK);
        case BSIM3V1_MOD_LA1:
            value->rValue = model->BSIM3V1la1;
            return(OK);
        case BSIM3V1_MOD_LA2:
            value->rValue = model->BSIM3V1la2;
            return(OK);
        case BSIM3V1_MOD_LKETA:
            value->rValue = model->BSIM3V1lketa;
            return(OK);   
        case BSIM3V1_MOD_LNSUB:
            value->rValue = model->BSIM3V1lnsub;
            return(OK);
        case BSIM3V1_MOD_LNPEAK:
            value->rValue = model->BSIM3V1lnpeak;
            return(OK);
        case BSIM3V1_MOD_LNGATE:
            value->rValue = model->BSIM3V1lngate;
            return(OK);
        case BSIM3V1_MOD_LGAMMA1:
            value->rValue = model->BSIM3V1lgamma1;
            return(OK);
        case BSIM3V1_MOD_LGAMMA2:
            value->rValue = model->BSIM3V1lgamma2;
            return(OK);
        case BSIM3V1_MOD_LVBX:
            value->rValue = model->BSIM3V1lvbx;
            return(OK);
        case BSIM3V1_MOD_LVBM:
            value->rValue = model->BSIM3V1lvbm;
            return(OK);
        case BSIM3V1_MOD_LXT:
            value->rValue = model->BSIM3V1lxt;
            return(OK);
        case  BSIM3V1_MOD_LK1:
          value->rValue = model->BSIM3V1lk1;
            return(OK);
        case  BSIM3V1_MOD_LKT1:
          value->rValue = model->BSIM3V1lkt1;
            return(OK);
        case  BSIM3V1_MOD_LKT1L:
          value->rValue = model->BSIM3V1lkt1l;
            return(OK);
        case  BSIM3V1_MOD_LKT2 :
          value->rValue = model->BSIM3V1lkt2;
            return(OK);
        case  BSIM3V1_MOD_LK2 :
          value->rValue = model->BSIM3V1lk2;
            return(OK);
        case  BSIM3V1_MOD_LK3:
          value->rValue = model->BSIM3V1lk3;
            return(OK);
        case  BSIM3V1_MOD_LK3B:
          value->rValue = model->BSIM3V1lk3b;
            return(OK);
        case  BSIM3V1_MOD_LW0:
          value->rValue = model->BSIM3V1lw0;
            return(OK);
        case  BSIM3V1_MOD_LNLX:
          value->rValue = model->BSIM3V1lnlx;
            return(OK);
        case  BSIM3V1_MOD_LDVT0:                
          value->rValue = model->BSIM3V1ldvt0;
            return(OK);
        case  BSIM3V1_MOD_LDVT1 :             
          value->rValue = model->BSIM3V1ldvt1;
            return(OK);
        case  BSIM3V1_MOD_LDVT2 :             
          value->rValue = model->BSIM3V1ldvt2;
            return(OK);
        case  BSIM3V1_MOD_LDVT0W :                
          value->rValue = model->BSIM3V1ldvt0w;
            return(OK);
        case  BSIM3V1_MOD_LDVT1W :             
          value->rValue = model->BSIM3V1ldvt1w;
            return(OK);
        case  BSIM3V1_MOD_LDVT2W :             
          value->rValue = model->BSIM3V1ldvt2w;
            return(OK);
        case  BSIM3V1_MOD_LDROUT :           
          value->rValue = model->BSIM3V1ldrout;
            return(OK);
        case  BSIM3V1_MOD_LDSUB :           
          value->rValue = model->BSIM3V1ldsub;
            return(OK);
        case BSIM3V1_MOD_LVTH0:
            value->rValue = model->BSIM3V1lvth0; 
            return(OK);
        case BSIM3V1_MOD_LUA:
            value->rValue = model->BSIM3V1lua; 
            return(OK);
        case BSIM3V1_MOD_LUA1:
            value->rValue = model->BSIM3V1lua1; 
            return(OK);
        case BSIM3V1_MOD_LUB:
            value->rValue = model->BSIM3V1lub;  
            return(OK);
        case BSIM3V1_MOD_LUB1:
            value->rValue = model->BSIM3V1lub1;  
            return(OK);
        case BSIM3V1_MOD_LUC:
            value->rValue = model->BSIM3V1luc; 
            return(OK);
        case BSIM3V1_MOD_LUC1:
            value->rValue = model->BSIM3V1luc1; 
            return(OK);
        case BSIM3V1_MOD_LU0:
            value->rValue = model->BSIM3V1lu0;
            return(OK);
        case BSIM3V1_MOD_LUTE:
            value->rValue = model->BSIM3V1lute;
            return(OK);
        case BSIM3V1_MOD_LVOFF:
            value->rValue = model->BSIM3V1lvoff;
            return(OK);
        case BSIM3V1_MOD_LDELTA:
            value->rValue = model->BSIM3V1ldelta;
            return(OK);
        case BSIM3V1_MOD_LRDSW:
            value->rValue = model->BSIM3V1lrdsw; 
            return(OK);             
        case BSIM3V1_MOD_LPRWB:
            value->rValue = model->BSIM3V1lprwb; 
            return(OK);             
        case BSIM3V1_MOD_LPRWG:
            value->rValue = model->BSIM3V1lprwg; 
            return(OK);             
        case BSIM3V1_MOD_LPRT:
            value->rValue = model->BSIM3V1lprt; 
            return(OK);              
        case BSIM3V1_MOD_LETA0:
            value->rValue = model->BSIM3V1leta0; 
            return(OK);               
        case BSIM3V1_MOD_LETAB:
            value->rValue = model->BSIM3V1letab; 
            return(OK);               
        case BSIM3V1_MOD_LPCLM:
            value->rValue = model->BSIM3V1lpclm; 
            return(OK);               
        case BSIM3V1_MOD_LPDIBL1:
            value->rValue = model->BSIM3V1lpdibl1; 
            return(OK);               
        case BSIM3V1_MOD_LPDIBL2:
            value->rValue = model->BSIM3V1lpdibl2; 
            return(OK);               
        case BSIM3V1_MOD_LPDIBLB:
            value->rValue = model->BSIM3V1lpdiblb; 
            return(OK);               
        case BSIM3V1_MOD_LPSCBE1:
            value->rValue = model->BSIM3V1lpscbe1; 
            return(OK);               
        case BSIM3V1_MOD_LPSCBE2:
            value->rValue = model->BSIM3V1lpscbe2; 
            return(OK);               
        case BSIM3V1_MOD_LPVAG:
            value->rValue = model->BSIM3V1lpvag; 
            return(OK);               
        case BSIM3V1_MOD_LWR:
            value->rValue = model->BSIM3V1lwr;
            return(OK);
        case BSIM3V1_MOD_LDWG:
            value->rValue = model->BSIM3V1ldwg;
            return(OK);
        case BSIM3V1_MOD_LDWB:
            value->rValue = model->BSIM3V1ldwb;
            return(OK);
        case BSIM3V1_MOD_LB0:
            value->rValue = model->BSIM3V1lb0;
            return(OK);
        case BSIM3V1_MOD_LB1:
            value->rValue = model->BSIM3V1lb1;
            return(OK);
        case BSIM3V1_MOD_LALPHA0:
            value->rValue = model->BSIM3V1lalpha0;
            return(OK);
        case BSIM3V1_MOD_LBETA0:
            value->rValue = model->BSIM3V1lbeta0;
            return(OK);

        case BSIM3V1_MOD_LELM:
            value->rValue = model->BSIM3V1lelm;
            return(OK);
        case BSIM3V1_MOD_LCGSL:
            value->rValue = model->BSIM3V1lcgsl;
            return(OK);
        case BSIM3V1_MOD_LCGDL:
            value->rValue = model->BSIM3V1lcgdl;
            return(OK);
        case BSIM3V1_MOD_LCKAPPA:
            value->rValue = model->BSIM3V1lckappa;
            return(OK);
        case BSIM3V1_MOD_LCF:
            value->rValue = model->BSIM3V1lcf;
            return(OK);
        case BSIM3V1_MOD_LCLC:
            value->rValue = model->BSIM3V1lclc;
            return(OK);
        case BSIM3V1_MOD_LCLE:
            value->rValue = model->BSIM3V1lcle;
            return(OK);
        case BSIM3V1_MOD_LVFBCV:
            value->rValue = model->BSIM3V1lvfbcv;
            return(OK);

	/* Width dependence */
        case  BSIM3V1_MOD_WCDSC :
          value->rValue = model->BSIM3V1wcdsc;
            return(OK);
        case  BSIM3V1_MOD_WCDSCB :
          value->rValue = model->BSIM3V1wcdscb;
            return(OK);
        case  BSIM3V1_MOD_WCDSCD :
          value->rValue = model->BSIM3V1wcdscd;
            return(OK);
        case  BSIM3V1_MOD_WCIT :
          value->rValue = model->BSIM3V1wcit;
            return(OK);
        case  BSIM3V1_MOD_WNFACTOR :
          value->rValue = model->BSIM3V1wnfactor;
            return(OK);
        case BSIM3V1_MOD_WXJ:
            value->rValue = model->BSIM3V1wxj;
            return(OK);
        case BSIM3V1_MOD_WVSAT:
            value->rValue = model->BSIM3V1wvsat;
            return(OK);
        case BSIM3V1_MOD_WAT:
            value->rValue = model->BSIM3V1wat;
            return(OK);
        case BSIM3V1_MOD_WA0:
            value->rValue = model->BSIM3V1wa0;
            return(OK);
        case BSIM3V1_MOD_WAGS:
            value->rValue = model->BSIM3V1wags;
            return(OK);
        case BSIM3V1_MOD_WA1:
            value->rValue = model->BSIM3V1wa1;
            return(OK);
        case BSIM3V1_MOD_WA2:
            value->rValue = model->BSIM3V1wa2;
            return(OK);
        case BSIM3V1_MOD_WKETA:
            value->rValue = model->BSIM3V1wketa;
            return(OK);   
        case BSIM3V1_MOD_WNSUB:
            value->rValue = model->BSIM3V1wnsub;
            return(OK);
        case BSIM3V1_MOD_WNPEAK:
            value->rValue = model->BSIM3V1wnpeak;
            return(OK);
        case BSIM3V1_MOD_WNGATE:
            value->rValue = model->BSIM3V1wngate;
            return(OK);
        case BSIM3V1_MOD_WGAMMA1:
            value->rValue = model->BSIM3V1wgamma1;
            return(OK);
        case BSIM3V1_MOD_WGAMMA2:
            value->rValue = model->BSIM3V1wgamma2;
            return(OK);
        case BSIM3V1_MOD_WVBX:
            value->rValue = model->BSIM3V1wvbx;
            return(OK);
        case BSIM3V1_MOD_WVBM:
            value->rValue = model->BSIM3V1wvbm;
            return(OK);
        case BSIM3V1_MOD_WXT:
            value->rValue = model->BSIM3V1wxt;
            return(OK);
        case  BSIM3V1_MOD_WK1:
          value->rValue = model->BSIM3V1wk1;
            return(OK);
        case  BSIM3V1_MOD_WKT1:
          value->rValue = model->BSIM3V1wkt1;
            return(OK);
        case  BSIM3V1_MOD_WKT1L:
          value->rValue = model->BSIM3V1wkt1l;
            return(OK);
        case  BSIM3V1_MOD_WKT2 :
          value->rValue = model->BSIM3V1wkt2;
            return(OK);
        case  BSIM3V1_MOD_WK2 :
          value->rValue = model->BSIM3V1wk2;
            return(OK);
        case  BSIM3V1_MOD_WK3:
          value->rValue = model->BSIM3V1wk3;
            return(OK);
        case  BSIM3V1_MOD_WK3B:
          value->rValue = model->BSIM3V1wk3b;
            return(OK);
        case  BSIM3V1_MOD_WW0:
          value->rValue = model->BSIM3V1ww0;
            return(OK);
        case  BSIM3V1_MOD_WNLX:
          value->rValue = model->BSIM3V1wnlx;
            return(OK);
        case  BSIM3V1_MOD_WDVT0:                
          value->rValue = model->BSIM3V1wdvt0;
            return(OK);
        case  BSIM3V1_MOD_WDVT1 :             
          value->rValue = model->BSIM3V1wdvt1;
            return(OK);
        case  BSIM3V1_MOD_WDVT2 :             
          value->rValue = model->BSIM3V1wdvt2;
            return(OK);
        case  BSIM3V1_MOD_WDVT0W :                
          value->rValue = model->BSIM3V1wdvt0w;
            return(OK);
        case  BSIM3V1_MOD_WDVT1W :             
          value->rValue = model->BSIM3V1wdvt1w;
            return(OK);
        case  BSIM3V1_MOD_WDVT2W :             
          value->rValue = model->BSIM3V1wdvt2w;
            return(OK);
        case  BSIM3V1_MOD_WDROUT :           
          value->rValue = model->BSIM3V1wdrout;
            return(OK);
        case  BSIM3V1_MOD_WDSUB :           
          value->rValue = model->BSIM3V1wdsub;
            return(OK);
        case BSIM3V1_MOD_WVTH0:
            value->rValue = model->BSIM3V1wvth0; 
            return(OK);
        case BSIM3V1_MOD_WUA:
            value->rValue = model->BSIM3V1wua; 
            return(OK);
        case BSIM3V1_MOD_WUA1:
            value->rValue = model->BSIM3V1wua1; 
            return(OK);
        case BSIM3V1_MOD_WUB:
            value->rValue = model->BSIM3V1wub;  
            return(OK);
        case BSIM3V1_MOD_WUB1:
            value->rValue = model->BSIM3V1wub1;  
            return(OK);
        case BSIM3V1_MOD_WUC:
            value->rValue = model->BSIM3V1wuc; 
            return(OK);
        case BSIM3V1_MOD_WUC1:
            value->rValue = model->BSIM3V1wuc1; 
            return(OK);
        case BSIM3V1_MOD_WU0:
            value->rValue = model->BSIM3V1wu0;
            return(OK);
        case BSIM3V1_MOD_WUTE:
            value->rValue = model->BSIM3V1wute;
            return(OK);
        case BSIM3V1_MOD_WVOFF:
            value->rValue = model->BSIM3V1wvoff;
            return(OK);
        case BSIM3V1_MOD_WDELTA:
            value->rValue = model->BSIM3V1wdelta;
            return(OK);
        case BSIM3V1_MOD_WRDSW:
            value->rValue = model->BSIM3V1wrdsw; 
            return(OK);             
        case BSIM3V1_MOD_WPRWB:
            value->rValue = model->BSIM3V1wprwb; 
            return(OK);             
        case BSIM3V1_MOD_WPRWG:
            value->rValue = model->BSIM3V1wprwg; 
            return(OK);             
        case BSIM3V1_MOD_WPRT:
            value->rValue = model->BSIM3V1wprt; 
            return(OK);              
        case BSIM3V1_MOD_WETA0:
            value->rValue = model->BSIM3V1weta0; 
            return(OK);               
        case BSIM3V1_MOD_WETAB:
            value->rValue = model->BSIM3V1wetab; 
            return(OK);               
        case BSIM3V1_MOD_WPCLM:
            value->rValue = model->BSIM3V1wpclm; 
            return(OK);               
        case BSIM3V1_MOD_WPDIBL1:
            value->rValue = model->BSIM3V1wpdibl1; 
            return(OK);               
        case BSIM3V1_MOD_WPDIBL2:
            value->rValue = model->BSIM3V1wpdibl2; 
            return(OK);               
        case BSIM3V1_MOD_WPDIBLB:
            value->rValue = model->BSIM3V1wpdiblb; 
            return(OK);               
        case BSIM3V1_MOD_WPSCBE1:
            value->rValue = model->BSIM3V1wpscbe1; 
            return(OK);               
        case BSIM3V1_MOD_WPSCBE2:
            value->rValue = model->BSIM3V1wpscbe2; 
            return(OK);               
        case BSIM3V1_MOD_WPVAG:
            value->rValue = model->BSIM3V1wpvag; 
            return(OK);               
        case BSIM3V1_MOD_WWR:
            value->rValue = model->BSIM3V1wwr;
            return(OK);
        case BSIM3V1_MOD_WDWG:
            value->rValue = model->BSIM3V1wdwg;
            return(OK);
        case BSIM3V1_MOD_WDWB:
            value->rValue = model->BSIM3V1wdwb;
            return(OK);
        case BSIM3V1_MOD_WB0:
            value->rValue = model->BSIM3V1wb0;
            return(OK);
        case BSIM3V1_MOD_WB1:
            value->rValue = model->BSIM3V1wb1;
            return(OK);
        case BSIM3V1_MOD_WALPHA0:
            value->rValue = model->BSIM3V1walpha0;
            return(OK);
        case BSIM3V1_MOD_WBETA0:
            value->rValue = model->BSIM3V1wbeta0;
            return(OK);

        case BSIM3V1_MOD_WELM:
            value->rValue = model->BSIM3V1welm;
            return(OK);
        case BSIM3V1_MOD_WCGSL:
            value->rValue = model->BSIM3V1wcgsl;
            return(OK);
        case BSIM3V1_MOD_WCGDL:
            value->rValue = model->BSIM3V1wcgdl;
            return(OK);
        case BSIM3V1_MOD_WCKAPPA:
            value->rValue = model->BSIM3V1wckappa;
            return(OK);
        case BSIM3V1_MOD_WCF:
            value->rValue = model->BSIM3V1wcf;
            return(OK);
        case BSIM3V1_MOD_WCLC:
            value->rValue = model->BSIM3V1wclc;
            return(OK);
        case BSIM3V1_MOD_WCLE:
            value->rValue = model->BSIM3V1wcle;
            return(OK);
        case BSIM3V1_MOD_WVFBCV:
            value->rValue = model->BSIM3V1wvfbcv;
            return(OK);

	/* Cross-term dependence */
        case  BSIM3V1_MOD_PCDSC :
          value->rValue = model->BSIM3V1pcdsc;
            return(OK);
        case  BSIM3V1_MOD_PCDSCB :
          value->rValue = model->BSIM3V1pcdscb;
            return(OK);
        case  BSIM3V1_MOD_PCDSCD :
          value->rValue = model->BSIM3V1pcdscd;
            return(OK);
         case  BSIM3V1_MOD_PCIT :
          value->rValue = model->BSIM3V1pcit;
            return(OK);
        case  BSIM3V1_MOD_PNFACTOR :
          value->rValue = model->BSIM3V1pnfactor;
            return(OK);
        case BSIM3V1_MOD_PXJ:
            value->rValue = model->BSIM3V1pxj;
            return(OK);
        case BSIM3V1_MOD_PVSAT:
            value->rValue = model->BSIM3V1pvsat;
            return(OK);
        case BSIM3V1_MOD_PAT:
            value->rValue = model->BSIM3V1pat;
            return(OK);
        case BSIM3V1_MOD_PA0:
            value->rValue = model->BSIM3V1pa0;
            return(OK);
        case BSIM3V1_MOD_PAGS:
            value->rValue = model->BSIM3V1pags;
            return(OK);
        case BSIM3V1_MOD_PA1:
            value->rValue = model->BSIM3V1pa1;
            return(OK);
        case BSIM3V1_MOD_PA2:
            value->rValue = model->BSIM3V1pa2;
            return(OK);
        case BSIM3V1_MOD_PKETA:
            value->rValue = model->BSIM3V1pketa;
            return(OK);   
        case BSIM3V1_MOD_PNSUB:
            value->rValue = model->BSIM3V1pnsub;
            return(OK);
        case BSIM3V1_MOD_PNPEAK:
            value->rValue = model->BSIM3V1pnpeak;
            return(OK);
        case BSIM3V1_MOD_PNGATE:
            value->rValue = model->BSIM3V1pngate;
            return(OK);
        case BSIM3V1_MOD_PGAMMA1:
            value->rValue = model->BSIM3V1pgamma1;
            return(OK);
        case BSIM3V1_MOD_PGAMMA2:
            value->rValue = model->BSIM3V1pgamma2;
            return(OK);
        case BSIM3V1_MOD_PVBX:
            value->rValue = model->BSIM3V1pvbx;
            return(OK);
        case BSIM3V1_MOD_PVBM:
            value->rValue = model->BSIM3V1pvbm;
            return(OK);
        case BSIM3V1_MOD_PXT:
            value->rValue = model->BSIM3V1pxt;
            return(OK);
        case  BSIM3V1_MOD_PK1:
          value->rValue = model->BSIM3V1pk1;
            return(OK);
        case  BSIM3V1_MOD_PKT1:
          value->rValue = model->BSIM3V1pkt1;
            return(OK);
        case  BSIM3V1_MOD_PKT1L:
          value->rValue = model->BSIM3V1pkt1l;
            return(OK);
        case  BSIM3V1_MOD_PKT2 :
          value->rValue = model->BSIM3V1pkt2;
            return(OK);
        case  BSIM3V1_MOD_PK2 :
          value->rValue = model->BSIM3V1pk2;
            return(OK);
        case  BSIM3V1_MOD_PK3:
          value->rValue = model->BSIM3V1pk3;
            return(OK);
        case  BSIM3V1_MOD_PK3B:
          value->rValue = model->BSIM3V1pk3b;
            return(OK);
        case  BSIM3V1_MOD_PW0:
          value->rValue = model->BSIM3V1pw0;
            return(OK);
        case  BSIM3V1_MOD_PNLX:
          value->rValue = model->BSIM3V1pnlx;
            return(OK);
        case  BSIM3V1_MOD_PDVT0 :                
          value->rValue = model->BSIM3V1pdvt0;
            return(OK);
        case  BSIM3V1_MOD_PDVT1 :             
          value->rValue = model->BSIM3V1pdvt1;
            return(OK);
        case  BSIM3V1_MOD_PDVT2 :             
          value->rValue = model->BSIM3V1pdvt2;
            return(OK);
        case  BSIM3V1_MOD_PDVT0W :                
          value->rValue = model->BSIM3V1pdvt0w;
            return(OK);
        case  BSIM3V1_MOD_PDVT1W :             
          value->rValue = model->BSIM3V1pdvt1w;
            return(OK);
        case  BSIM3V1_MOD_PDVT2W :             
          value->rValue = model->BSIM3V1pdvt2w;
            return(OK);
        case  BSIM3V1_MOD_PDROUT :           
          value->rValue = model->BSIM3V1pdrout;
            return(OK);
        case  BSIM3V1_MOD_PDSUB :           
          value->rValue = model->BSIM3V1pdsub;
            return(OK);
        case BSIM3V1_MOD_PVTH0:
            value->rValue = model->BSIM3V1pvth0; 
            return(OK);
        case BSIM3V1_MOD_PUA:
            value->rValue = model->BSIM3V1pua; 
            return(OK);
        case BSIM3V1_MOD_PUA1:
            value->rValue = model->BSIM3V1pua1; 
            return(OK);
        case BSIM3V1_MOD_PUB:
            value->rValue = model->BSIM3V1pub;  
            return(OK);
        case BSIM3V1_MOD_PUB1:
            value->rValue = model->BSIM3V1pub1;  
            return(OK);
        case BSIM3V1_MOD_PUC:
            value->rValue = model->BSIM3V1puc; 
            return(OK);
        case BSIM3V1_MOD_PUC1:
            value->rValue = model->BSIM3V1puc1; 
            return(OK);
        case BSIM3V1_MOD_PU0:
            value->rValue = model->BSIM3V1pu0;
            return(OK);
        case BSIM3V1_MOD_PUTE:
            value->rValue = model->BSIM3V1pute;
            return(OK);
        case BSIM3V1_MOD_PVOFF:
            value->rValue = model->BSIM3V1pvoff;
            return(OK);
        case BSIM3V1_MOD_PDELTA:
            value->rValue = model->BSIM3V1pdelta;
            return(OK);
        case BSIM3V1_MOD_PRDSW:
            value->rValue = model->BSIM3V1prdsw; 
            return(OK);             
        case BSIM3V1_MOD_PPRWB:
            value->rValue = model->BSIM3V1pprwb; 
            return(OK);             
        case BSIM3V1_MOD_PPRWG:
            value->rValue = model->BSIM3V1pprwg; 
            return(OK);             
        case BSIM3V1_MOD_PPRT:
            value->rValue = model->BSIM3V1pprt; 
            return(OK);              
        case BSIM3V1_MOD_PETA0:
            value->rValue = model->BSIM3V1peta0; 
            return(OK);               
        case BSIM3V1_MOD_PETAB:
            value->rValue = model->BSIM3V1petab; 
            return(OK);               
        case BSIM3V1_MOD_PPCLM:
            value->rValue = model->BSIM3V1ppclm; 
            return(OK);               
        case BSIM3V1_MOD_PPDIBL1:
            value->rValue = model->BSIM3V1ppdibl1; 
            return(OK);               
        case BSIM3V1_MOD_PPDIBL2:
            value->rValue = model->BSIM3V1ppdibl2; 
            return(OK);               
        case BSIM3V1_MOD_PPDIBLB:
            value->rValue = model->BSIM3V1ppdiblb; 
            return(OK);               
        case BSIM3V1_MOD_PPSCBE1:
            value->rValue = model->BSIM3V1ppscbe1; 
            return(OK);               
        case BSIM3V1_MOD_PPSCBE2:
            value->rValue = model->BSIM3V1ppscbe2; 
            return(OK);               
        case BSIM3V1_MOD_PPVAG:
            value->rValue = model->BSIM3V1ppvag; 
            return(OK);               
        case BSIM3V1_MOD_PWR:
            value->rValue = model->BSIM3V1pwr;
            return(OK);
        case BSIM3V1_MOD_PDWG:
            value->rValue = model->BSIM3V1pdwg;
            return(OK);
        case BSIM3V1_MOD_PDWB:
            value->rValue = model->BSIM3V1pdwb;
            return(OK);
        case BSIM3V1_MOD_PB0:
            value->rValue = model->BSIM3V1pb0;
            return(OK);
        case BSIM3V1_MOD_PB1:
            value->rValue = model->BSIM3V1pb1;
            return(OK);
        case BSIM3V1_MOD_PALPHA0:
            value->rValue = model->BSIM3V1palpha0;
            return(OK);
        case BSIM3V1_MOD_PBETA0:
            value->rValue = model->BSIM3V1pbeta0;
            return(OK);

        case BSIM3V1_MOD_PELM:
            value->rValue = model->BSIM3V1pelm;
            return(OK);
        case BSIM3V1_MOD_PCGSL:
            value->rValue = model->BSIM3V1pcgsl;
            return(OK);
        case BSIM3V1_MOD_PCGDL:
            value->rValue = model->BSIM3V1pcgdl;
            return(OK);
        case BSIM3V1_MOD_PCKAPPA:
            value->rValue = model->BSIM3V1pckappa;
            return(OK);
        case BSIM3V1_MOD_PCF:
            value->rValue = model->BSIM3V1pcf;
            return(OK);
        case BSIM3V1_MOD_PCLC:
            value->rValue = model->BSIM3V1pclc;
            return(OK);
        case BSIM3V1_MOD_PCLE:
            value->rValue = model->BSIM3V1pcle;
            return(OK);
        case BSIM3V1_MOD_PVFBCV:
            value->rValue = model->BSIM3V1pvfbcv;
            return(OK);

        case  BSIM3V1_MOD_TNOM :
          value->rValue = model->BSIM3V1tnom;
            return(OK);
        case BSIM3V1_MOD_CGSO:
            value->rValue = model->BSIM3V1cgso; 
            return(OK);
        case BSIM3V1_MOD_CGDO:
            value->rValue = model->BSIM3V1cgdo; 
            return(OK);
        case BSIM3V1_MOD_CGBO:
            value->rValue = model->BSIM3V1cgbo; 
            return(OK);
        case BSIM3V1_MOD_XPART:
            value->rValue = model->BSIM3V1xpart; 
            return(OK);
        case BSIM3V1_MOD_RSH:
            value->rValue = model->BSIM3V1sheetResistance; 
            return(OK);
        case BSIM3V1_MOD_JS:
            value->rValue = model->BSIM3V1jctSatCurDensity; 
            return(OK);
        case BSIM3V1_MOD_JSW:
            value->rValue = model->BSIM3V1jctSidewallSatCurDensity; 
            return(OK);
        case BSIM3V1_MOD_PB:
            value->rValue = model->BSIM3V1bulkJctPotential; 
            return(OK);
        case BSIM3V1_MOD_MJ:
            value->rValue = model->BSIM3V1bulkJctBotGradingCoeff; 
            return(OK);
        case BSIM3V1_MOD_PBSW:
            value->rValue = model->BSIM3V1sidewallJctPotential; 
            return(OK);
        case BSIM3V1_MOD_MJSW:
            value->rValue = model->BSIM3V1bulkJctSideGradingCoeff; 
            return(OK);
        case BSIM3V1_MOD_CJ:
            value->rValue = model->BSIM3V1unitAreaJctCap; 
            return(OK);
        case BSIM3V1_MOD_CJSW:
            value->rValue = model->BSIM3V1unitLengthSidewallJctCap; 
            return(OK);
        case BSIM3V1_MOD_PBSWG:
            value->rValue = model->BSIM3V1GatesidewallJctPotential; 
            return(OK);
        case BSIM3V1_MOD_MJSWG:
            value->rValue = model->BSIM3V1bulkJctGateSideGradingCoeff; 
            return(OK);
        case BSIM3V1_MOD_CJSWG:
            value->rValue = model->BSIM3V1unitLengthGateSidewallJctCap; 
            return(OK);
        case BSIM3V1_MOD_NJ:
            value->rValue = model->BSIM3V1jctEmissionCoeff; 
            return(OK);
        case BSIM3V1_MOD_XTI:
            value->rValue = model->BSIM3V1jctTempExponent; 
            return(OK);
        case BSIM3V1_MOD_LINT:
            value->rValue = model->BSIM3V1Lint; 
            return(OK);
        case BSIM3V1_MOD_LL:
            value->rValue = model->BSIM3V1Ll;
            return(OK);
        case BSIM3V1_MOD_LLN:
            value->rValue = model->BSIM3V1Lln;
            return(OK);
        case BSIM3V1_MOD_LW:
            value->rValue = model->BSIM3V1Lw;
            return(OK);
        case BSIM3V1_MOD_LWN:
            value->rValue = model->BSIM3V1Lwn;
            return(OK);
        case BSIM3V1_MOD_LWL:
            value->rValue = model->BSIM3V1Lwl;
            return(OK);
        case BSIM3V1_MOD_LMIN:
            value->rValue = model->BSIM3V1Lmin;
            return(OK);
        case BSIM3V1_MOD_LMAX:
            value->rValue = model->BSIM3V1Lmax;
            return(OK);
        case BSIM3V1_MOD_WINT:
            value->rValue = model->BSIM3V1Wint;
            return(OK);
        case BSIM3V1_MOD_WL:
            value->rValue = model->BSIM3V1Wl;
            return(OK);
        case BSIM3V1_MOD_WLN:
            value->rValue = model->BSIM3V1Wln;
            return(OK);
        case BSIM3V1_MOD_WW:
            value->rValue = model->BSIM3V1Ww;
            return(OK);
        case BSIM3V1_MOD_WWN:
            value->rValue = model->BSIM3V1Wwn;
            return(OK);
        case BSIM3V1_MOD_WWL:
            value->rValue = model->BSIM3V1Wwl;
            return(OK);
        case BSIM3V1_MOD_WMIN:
            value->rValue = model->BSIM3V1Wmin;
            return(OK);
        case BSIM3V1_MOD_WMAX:
            value->rValue = model->BSIM3V1Wmax;
            return(OK);
        case BSIM3V1_MOD_NOIA:
            value->rValue = model->BSIM3V1oxideTrapDensityA;
            return(OK);
        case BSIM3V1_MOD_NOIB:
            value->rValue = model->BSIM3V1oxideTrapDensityB;
            return(OK);
        case BSIM3V1_MOD_NOIC:
            value->rValue = model->BSIM3V1oxideTrapDensityC;
            return(OK);
        case BSIM3V1_MOD_EM:
            value->rValue = model->BSIM3V1em;
            return(OK);
        case BSIM3V1_MOD_EF:
            value->rValue = model->BSIM3V1ef;
            return(OK);
        case BSIM3V1_MOD_AF:
            value->rValue = model->BSIM3V1af;
            return(OK);
        case BSIM3V1_MOD_KF:
            value->rValue = model->BSIM3V1kf;
            return(OK);
        default:
            return(E_BADPARM);
    }
    /* NOTREACHED */
}



