/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1995 Min-Chie Jeng and Mansun Chan.
Modified by Paolo Nenzi 2002
File: b3v1smask.c
**********/

#include "ngspice.h"
#include "ifsim.h"
#include "cktdefs.h"
#include "devdefs.h"
#include "bsim3v1sdef.h"
#include "sperror.h"
#include "suffix.h"

int
BSIM3v1SmAsk(CKTcircuit *ckt, GENmodel *inst, int which, IFvalue *value)
{
    BSIM3v1Smodel *model = (BSIM3v1Smodel *)inst;
    switch(which) 
    {   case BSIM3v1S_MOD_MOBMOD:
            value->iValue = model->BSIM3v1SmobMod; 
            return(OK);
        case BSIM3v1S_MOD_PARAMCHK:
            value->iValue = model->BSIM3v1SparamChk; 
            return(OK);
        case BSIM3v1S_MOD_BINUNIT:
            value->iValue = model->BSIM3v1SbinUnit; 
            return(OK);
        case BSIM3v1S_MOD_CAPMOD:
            value->iValue = model->BSIM3v1ScapMod; 
            return(OK);
        case BSIM3v1S_MOD_NQSMOD:
            value->iValue = model->BSIM3v1SnqsMod; 
            return(OK);
        case BSIM3v1S_MOD_NOIMOD:
            value->iValue = model->BSIM3v1SnoiMod; 
            return(OK);
        case  BSIM3v1S_MOD_VERSION :
          value->rValue = model->BSIM3v1Sversion;
            return(OK);
        case  BSIM3v1S_MOD_TOX :
          value->rValue = model->BSIM3v1Stox;
            return(OK);
        case  BSIM3v1S_MOD_CDSC :
          value->rValue = model->BSIM3v1Scdsc;
            return(OK);
        case  BSIM3v1S_MOD_CDSCB :
          value->rValue = model->BSIM3v1Scdscb;
            return(OK);

        case  BSIM3v1S_MOD_CDSCD :
          value->rValue = model->BSIM3v1Scdscd;
            return(OK);

        case  BSIM3v1S_MOD_CIT :
          value->rValue = model->BSIM3v1Scit;
            return(OK);
        case  BSIM3v1S_MOD_NFACTOR :
          value->rValue = model->BSIM3v1Snfactor;
            return(OK);
        case BSIM3v1S_MOD_XJ:
            value->rValue = model->BSIM3v1Sxj;
            return(OK);
        case BSIM3v1S_MOD_VSAT:
            value->rValue = model->BSIM3v1Svsat;
            return(OK);
        case BSIM3v1S_MOD_AT:
            value->rValue = model->BSIM3v1Sat;
            return(OK);
        case BSIM3v1S_MOD_A0:
            value->rValue = model->BSIM3v1Sa0;
            return(OK);

        case BSIM3v1S_MOD_AGS:
            value->rValue = model->BSIM3v1Sags;
            return(OK);

        case BSIM3v1S_MOD_A1:
            value->rValue = model->BSIM3v1Sa1;
            return(OK);
        case BSIM3v1S_MOD_A2:
            value->rValue = model->BSIM3v1Sa2;
            return(OK);
        case BSIM3v1S_MOD_KETA:
            value->rValue = model->BSIM3v1Sketa;
            return(OK);   
        case BSIM3v1S_MOD_NSUB:
            value->rValue = model->BSIM3v1Snsub;
            return(OK);
        case BSIM3v1S_MOD_NPEAK:
            value->rValue = model->BSIM3v1Snpeak;
            return(OK);
        case BSIM3v1S_MOD_NGATE:
            value->rValue = model->BSIM3v1Sngate;
            return(OK);
        case BSIM3v1S_MOD_GAMMA1:
            value->rValue = model->BSIM3v1Sgamma1;
            return(OK);
        case BSIM3v1S_MOD_GAMMA2:
            value->rValue = model->BSIM3v1Sgamma2;
            return(OK);
        case BSIM3v1S_MOD_VBX:
            value->rValue = model->BSIM3v1Svbx;
            return(OK);
        case BSIM3v1S_MOD_VBM:
            value->rValue = model->BSIM3v1Svbm;
            return(OK);
        case BSIM3v1S_MOD_XT:
            value->rValue = model->BSIM3v1Sxt;
            return(OK);
        case  BSIM3v1S_MOD_K1:
          value->rValue = model->BSIM3v1Sk1;
            return(OK);
        case  BSIM3v1S_MOD_KT1:
          value->rValue = model->BSIM3v1Skt1;
            return(OK);
        case  BSIM3v1S_MOD_KT1L:
          value->rValue = model->BSIM3v1Skt1l;
            return(OK);
        case  BSIM3v1S_MOD_KT2 :
          value->rValue = model->BSIM3v1Skt2;
            return(OK);
        case  BSIM3v1S_MOD_K2 :
          value->rValue = model->BSIM3v1Sk2;
            return(OK);
        case  BSIM3v1S_MOD_K3:
          value->rValue = model->BSIM3v1Sk3;
            return(OK);
        case  BSIM3v1S_MOD_K3B:
          value->rValue = model->BSIM3v1Sk3b;
            return(OK);
        case  BSIM3v1S_MOD_W0:
          value->rValue = model->BSIM3v1Sw0;
            return(OK);
        case  BSIM3v1S_MOD_NLX:
          value->rValue = model->BSIM3v1Snlx;
            return(OK);
        case  BSIM3v1S_MOD_DVT0 :                
          value->rValue = model->BSIM3v1Sdvt0;
            return(OK);
        case  BSIM3v1S_MOD_DVT1 :             
          value->rValue = model->BSIM3v1Sdvt1;
            return(OK);
        case  BSIM3v1S_MOD_DVT2 :             
          value->rValue = model->BSIM3v1Sdvt2;
            return(OK);
        case  BSIM3v1S_MOD_DVT0W :                
          value->rValue = model->BSIM3v1Sdvt0w;
            return(OK);
        case  BSIM3v1S_MOD_DVT1W :             
          value->rValue = model->BSIM3v1Sdvt1w;
            return(OK);
        case  BSIM3v1S_MOD_DVT2W :             
          value->rValue = model->BSIM3v1Sdvt2w;
            return(OK);
        case  BSIM3v1S_MOD_DROUT :           
          value->rValue = model->BSIM3v1Sdrout;
            return(OK);
        case  BSIM3v1S_MOD_DSUB :           
          value->rValue = model->BSIM3v1Sdsub;
            return(OK);
        case BSIM3v1S_MOD_VTH0:
            value->rValue = model->BSIM3v1Svth0; 
            return(OK);
        case BSIM3v1S_MOD_UA:
            value->rValue = model->BSIM3v1Sua; 
            return(OK);
        case BSIM3v1S_MOD_UA1:
            value->rValue = model->BSIM3v1Sua1; 
            return(OK);
        case BSIM3v1S_MOD_UB:
            value->rValue = model->BSIM3v1Sub;  
            return(OK);
        case BSIM3v1S_MOD_UB1:
            value->rValue = model->BSIM3v1Sub1;  
            return(OK);
        case BSIM3v1S_MOD_UC:
            value->rValue = model->BSIM3v1Suc; 
            return(OK);
        case BSIM3v1S_MOD_UC1:
            value->rValue = model->BSIM3v1Suc1; 
            return(OK);
        case BSIM3v1S_MOD_U0:
            value->rValue = model->BSIM3v1Su0;
            return(OK);
        case BSIM3v1S_MOD_UTE:
            value->rValue = model->BSIM3v1Sute;
            return(OK);
        case BSIM3v1S_MOD_VOFF:
            value->rValue = model->BSIM3v1Svoff;
            return(OK);
        case BSIM3v1S_MOD_DELTA:
            value->rValue = model->BSIM3v1Sdelta;
            return(OK);
        case BSIM3v1S_MOD_RDSW:
            value->rValue = model->BSIM3v1Srdsw; 
            return(OK);             
        case BSIM3v1S_MOD_PRWG:
            value->rValue = model->BSIM3v1Sprwg; 
            return(OK);             
        case BSIM3v1S_MOD_PRWB:
            value->rValue = model->BSIM3v1Sprwb; 
            return(OK);             
        case BSIM3v1S_MOD_PRT:
            value->rValue = model->BSIM3v1Sprt; 
            return(OK);              
        case BSIM3v1S_MOD_ETA0:
            value->rValue = model->BSIM3v1Seta0; 
            return(OK);               
        case BSIM3v1S_MOD_ETAB:
            value->rValue = model->BSIM3v1Setab; 
            return(OK);               
        case BSIM3v1S_MOD_PCLM:
            value->rValue = model->BSIM3v1Spclm; 
            return(OK);               
        case BSIM3v1S_MOD_PDIBL1:
            value->rValue = model->BSIM3v1Spdibl1; 
            return(OK);               
        case BSIM3v1S_MOD_PDIBL2:
            value->rValue = model->BSIM3v1Spdibl2; 
            return(OK);               
        case BSIM3v1S_MOD_PDIBLB:
            value->rValue = model->BSIM3v1Spdiblb; 
            return(OK);               
        case BSIM3v1S_MOD_PSCBE1:
            value->rValue = model->BSIM3v1Spscbe1; 
            return(OK);               
        case BSIM3v1S_MOD_PSCBE2:
            value->rValue = model->BSIM3v1Spscbe2; 
            return(OK);               
        case BSIM3v1S_MOD_PVAG:
            value->rValue = model->BSIM3v1Spvag; 
            return(OK);               
        case BSIM3v1S_MOD_WR:
            value->rValue = model->BSIM3v1Swr;
            return(OK);
        case BSIM3v1S_MOD_DWG:
            value->rValue = model->BSIM3v1Sdwg;
            return(OK);
        case BSIM3v1S_MOD_DWB:
            value->rValue = model->BSIM3v1Sdwb;
            return(OK);
        case BSIM3v1S_MOD_B0:
            value->rValue = model->BSIM3v1Sb0;
            return(OK);
        case BSIM3v1S_MOD_B1:
            value->rValue = model->BSIM3v1Sb1;
            return(OK);
        case BSIM3v1S_MOD_ALPHA0:
            value->rValue = model->BSIM3v1Salpha0;
            return(OK);
        case BSIM3v1S_MOD_BETA0:
            value->rValue = model->BSIM3v1Sbeta0;
            return(OK);

        case BSIM3v1S_MOD_ELM:
            value->rValue = model->BSIM3v1Selm;
            return(OK);
        case BSIM3v1S_MOD_CGSL:
            value->rValue = model->BSIM3v1Scgsl;
            return(OK);
        case BSIM3v1S_MOD_CGDL:
            value->rValue = model->BSIM3v1Scgdl;
            return(OK);
        case BSIM3v1S_MOD_CKAPPA:
            value->rValue = model->BSIM3v1Sckappa;
            return(OK);
        case BSIM3v1S_MOD_CF:
            value->rValue = model->BSIM3v1Scf;
            return(OK);
        case BSIM3v1S_MOD_CLC:
            value->rValue = model->BSIM3v1Sclc;
            return(OK);
        case BSIM3v1S_MOD_CLE:
            value->rValue = model->BSIM3v1Scle;
            return(OK);
        case BSIM3v1S_MOD_DWC:
            value->rValue = model->BSIM3v1Sdwc;
            return(OK);
        case BSIM3v1S_MOD_DLC:
            value->rValue = model->BSIM3v1Sdlc;
            return(OK);
        case BSIM3v1S_MOD_VFBCV:
            value->rValue = model->BSIM3v1Svfbcv; 
            return(OK);

	/* Length dependence */
        case  BSIM3v1S_MOD_LCDSC :
          value->rValue = model->BSIM3v1Slcdsc;
            return(OK);
        case  BSIM3v1S_MOD_LCDSCB :
          value->rValue = model->BSIM3v1Slcdscb;
            return(OK);
        case  BSIM3v1S_MOD_LCDSCD :
          value->rValue = model->BSIM3v1Slcdscd;
            return(OK);
        case  BSIM3v1S_MOD_LCIT :
          value->rValue = model->BSIM3v1Slcit;
            return(OK);
        case  BSIM3v1S_MOD_LNFACTOR :
          value->rValue = model->BSIM3v1Slnfactor;
            return(OK);
        case BSIM3v1S_MOD_LXJ:
            value->rValue = model->BSIM3v1Slxj;
            return(OK);
        case BSIM3v1S_MOD_LVSAT:
            value->rValue = model->BSIM3v1Slvsat;
            return(OK);
        case BSIM3v1S_MOD_LAT:
            value->rValue = model->BSIM3v1Slat;
            return(OK);
        case BSIM3v1S_MOD_LA0:
            value->rValue = model->BSIM3v1Sla0;
            return(OK);
        case BSIM3v1S_MOD_LAGS:
            value->rValue = model->BSIM3v1Slags;
            return(OK);
        case BSIM3v1S_MOD_LA1:
            value->rValue = model->BSIM3v1Sla1;
            return(OK);
        case BSIM3v1S_MOD_LA2:
            value->rValue = model->BSIM3v1Sla2;
            return(OK);
        case BSIM3v1S_MOD_LKETA:
            value->rValue = model->BSIM3v1Slketa;
            return(OK);   
        case BSIM3v1S_MOD_LNSUB:
            value->rValue = model->BSIM3v1Slnsub;
            return(OK);
        case BSIM3v1S_MOD_LNPEAK:
            value->rValue = model->BSIM3v1Slnpeak;
            return(OK);
        case BSIM3v1S_MOD_LNGATE:
            value->rValue = model->BSIM3v1Slngate;
            return(OK);
        case BSIM3v1S_MOD_LGAMMA1:
            value->rValue = model->BSIM3v1Slgamma1;
            return(OK);
        case BSIM3v1S_MOD_LGAMMA2:
            value->rValue = model->BSIM3v1Slgamma2;
            return(OK);
        case BSIM3v1S_MOD_LVBX:
            value->rValue = model->BSIM3v1Slvbx;
            return(OK);
        case BSIM3v1S_MOD_LVBM:
            value->rValue = model->BSIM3v1Slvbm;
            return(OK);
        case BSIM3v1S_MOD_LXT:
            value->rValue = model->BSIM3v1Slxt;
            return(OK);
        case  BSIM3v1S_MOD_LK1:
          value->rValue = model->BSIM3v1Slk1;
            return(OK);
        case  BSIM3v1S_MOD_LKT1:
          value->rValue = model->BSIM3v1Slkt1;
            return(OK);
        case  BSIM3v1S_MOD_LKT1L:
          value->rValue = model->BSIM3v1Slkt1l;
            return(OK);
        case  BSIM3v1S_MOD_LKT2 :
          value->rValue = model->BSIM3v1Slkt2;
            return(OK);
        case  BSIM3v1S_MOD_LK2 :
          value->rValue = model->BSIM3v1Slk2;
            return(OK);
        case  BSIM3v1S_MOD_LK3:
          value->rValue = model->BSIM3v1Slk3;
            return(OK);
        case  BSIM3v1S_MOD_LK3B:
          value->rValue = model->BSIM3v1Slk3b;
            return(OK);
        case  BSIM3v1S_MOD_LW0:
          value->rValue = model->BSIM3v1Slw0;
            return(OK);
        case  BSIM3v1S_MOD_LNLX:
          value->rValue = model->BSIM3v1Slnlx;
            return(OK);
        case  BSIM3v1S_MOD_LDVT0:                
          value->rValue = model->BSIM3v1Sldvt0;
            return(OK);
        case  BSIM3v1S_MOD_LDVT1 :             
          value->rValue = model->BSIM3v1Sldvt1;
            return(OK);
        case  BSIM3v1S_MOD_LDVT2 :             
          value->rValue = model->BSIM3v1Sldvt2;
            return(OK);
        case  BSIM3v1S_MOD_LDVT0W :                
          value->rValue = model->BSIM3v1Sldvt0w;
            return(OK);
        case  BSIM3v1S_MOD_LDVT1W :             
          value->rValue = model->BSIM3v1Sldvt1w;
            return(OK);
        case  BSIM3v1S_MOD_LDVT2W :             
          value->rValue = model->BSIM3v1Sldvt2w;
            return(OK);
        case  BSIM3v1S_MOD_LDROUT :           
          value->rValue = model->BSIM3v1Sldrout;
            return(OK);
        case  BSIM3v1S_MOD_LDSUB :           
          value->rValue = model->BSIM3v1Sldsub;
            return(OK);
        case BSIM3v1S_MOD_LVTH0:
            value->rValue = model->BSIM3v1Slvth0; 
            return(OK);
        case BSIM3v1S_MOD_LUA:
            value->rValue = model->BSIM3v1Slua; 
            return(OK);
        case BSIM3v1S_MOD_LUA1:
            value->rValue = model->BSIM3v1Slua1; 
            return(OK);
        case BSIM3v1S_MOD_LUB:
            value->rValue = model->BSIM3v1Slub;  
            return(OK);
        case BSIM3v1S_MOD_LUB1:
            value->rValue = model->BSIM3v1Slub1;  
            return(OK);
        case BSIM3v1S_MOD_LUC:
            value->rValue = model->BSIM3v1Sluc; 
            return(OK);
        case BSIM3v1S_MOD_LUC1:
            value->rValue = model->BSIM3v1Sluc1; 
            return(OK);
        case BSIM3v1S_MOD_LU0:
            value->rValue = model->BSIM3v1Slu0;
            return(OK);
        case BSIM3v1S_MOD_LUTE:
            value->rValue = model->BSIM3v1Slute;
            return(OK);
        case BSIM3v1S_MOD_LVOFF:
            value->rValue = model->BSIM3v1Slvoff;
            return(OK);
        case BSIM3v1S_MOD_LDELTA:
            value->rValue = model->BSIM3v1Sldelta;
            return(OK);
        case BSIM3v1S_MOD_LRDSW:
            value->rValue = model->BSIM3v1Slrdsw; 
            return(OK);             
        case BSIM3v1S_MOD_LPRWB:
            value->rValue = model->BSIM3v1Slprwb; 
            return(OK);             
        case BSIM3v1S_MOD_LPRWG:
            value->rValue = model->BSIM3v1Slprwg; 
            return(OK);             
        case BSIM3v1S_MOD_LPRT:
            value->rValue = model->BSIM3v1Slprt; 
            return(OK);              
        case BSIM3v1S_MOD_LETA0:
            value->rValue = model->BSIM3v1Sleta0; 
            return(OK);               
        case BSIM3v1S_MOD_LETAB:
            value->rValue = model->BSIM3v1Sletab; 
            return(OK);               
        case BSIM3v1S_MOD_LPCLM:
            value->rValue = model->BSIM3v1Slpclm; 
            return(OK);               
        case BSIM3v1S_MOD_LPDIBL1:
            value->rValue = model->BSIM3v1Slpdibl1; 
            return(OK);               
        case BSIM3v1S_MOD_LPDIBL2:
            value->rValue = model->BSIM3v1Slpdibl2; 
            return(OK);               
        case BSIM3v1S_MOD_LPDIBLB:
            value->rValue = model->BSIM3v1Slpdiblb; 
            return(OK);               
        case BSIM3v1S_MOD_LPSCBE1:
            value->rValue = model->BSIM3v1Slpscbe1; 
            return(OK);               
        case BSIM3v1S_MOD_LPSCBE2:
            value->rValue = model->BSIM3v1Slpscbe2; 
            return(OK);               
        case BSIM3v1S_MOD_LPVAG:
            value->rValue = model->BSIM3v1Slpvag; 
            return(OK);               
        case BSIM3v1S_MOD_LWR:
            value->rValue = model->BSIM3v1Slwr;
            return(OK);
        case BSIM3v1S_MOD_LDWG:
            value->rValue = model->BSIM3v1Sldwg;
            return(OK);
        case BSIM3v1S_MOD_LDWB:
            value->rValue = model->BSIM3v1Sldwb;
            return(OK);
        case BSIM3v1S_MOD_LB0:
            value->rValue = model->BSIM3v1Slb0;
            return(OK);
        case BSIM3v1S_MOD_LB1:
            value->rValue = model->BSIM3v1Slb1;
            return(OK);
        case BSIM3v1S_MOD_LALPHA0:
            value->rValue = model->BSIM3v1Slalpha0;
            return(OK);
        case BSIM3v1S_MOD_LBETA0:
            value->rValue = model->BSIM3v1Slbeta0;
            return(OK);

        case BSIM3v1S_MOD_LELM:
            value->rValue = model->BSIM3v1Slelm;
            return(OK);
        case BSIM3v1S_MOD_LCGSL:
            value->rValue = model->BSIM3v1Slcgsl;
            return(OK);
        case BSIM3v1S_MOD_LCGDL:
            value->rValue = model->BSIM3v1Slcgdl;
            return(OK);
        case BSIM3v1S_MOD_LCKAPPA:
            value->rValue = model->BSIM3v1Slckappa;
            return(OK);
        case BSIM3v1S_MOD_LCF:
            value->rValue = model->BSIM3v1Slcf;
            return(OK);
        case BSIM3v1S_MOD_LCLC:
            value->rValue = model->BSIM3v1Slclc;
            return(OK);
        case BSIM3v1S_MOD_LCLE:
            value->rValue = model->BSIM3v1Slcle;
            return(OK);
        case BSIM3v1S_MOD_LVFBCV:
            value->rValue = model->BSIM3v1Slvfbcv;
            return(OK);

	/* Width dependence */
        case  BSIM3v1S_MOD_WCDSC :
          value->rValue = model->BSIM3v1Swcdsc;
            return(OK);
        case  BSIM3v1S_MOD_WCDSCB :
          value->rValue = model->BSIM3v1Swcdscb;
            return(OK);
        case  BSIM3v1S_MOD_WCDSCD :
          value->rValue = model->BSIM3v1Swcdscd;
            return(OK);
        case  BSIM3v1S_MOD_WCIT :
          value->rValue = model->BSIM3v1Swcit;
            return(OK);
        case  BSIM3v1S_MOD_WNFACTOR :
          value->rValue = model->BSIM3v1Swnfactor;
            return(OK);
        case BSIM3v1S_MOD_WXJ:
            value->rValue = model->BSIM3v1Swxj;
            return(OK);
        case BSIM3v1S_MOD_WVSAT:
            value->rValue = model->BSIM3v1Swvsat;
            return(OK);
        case BSIM3v1S_MOD_WAT:
            value->rValue = model->BSIM3v1Swat;
            return(OK);
        case BSIM3v1S_MOD_WA0:
            value->rValue = model->BSIM3v1Swa0;
            return(OK);
        case BSIM3v1S_MOD_WAGS:
            value->rValue = model->BSIM3v1Swags;
            return(OK);
        case BSIM3v1S_MOD_WA1:
            value->rValue = model->BSIM3v1Swa1;
            return(OK);
        case BSIM3v1S_MOD_WA2:
            value->rValue = model->BSIM3v1Swa2;
            return(OK);
        case BSIM3v1S_MOD_WKETA:
            value->rValue = model->BSIM3v1Swketa;
            return(OK);   
        case BSIM3v1S_MOD_WNSUB:
            value->rValue = model->BSIM3v1Swnsub;
            return(OK);
        case BSIM3v1S_MOD_WNPEAK:
            value->rValue = model->BSIM3v1Swnpeak;
            return(OK);
        case BSIM3v1S_MOD_WNGATE:
            value->rValue = model->BSIM3v1Swngate;
            return(OK);
        case BSIM3v1S_MOD_WGAMMA1:
            value->rValue = model->BSIM3v1Swgamma1;
            return(OK);
        case BSIM3v1S_MOD_WGAMMA2:
            value->rValue = model->BSIM3v1Swgamma2;
            return(OK);
        case BSIM3v1S_MOD_WVBX:
            value->rValue = model->BSIM3v1Swvbx;
            return(OK);
        case BSIM3v1S_MOD_WVBM:
            value->rValue = model->BSIM3v1Swvbm;
            return(OK);
        case BSIM3v1S_MOD_WXT:
            value->rValue = model->BSIM3v1Swxt;
            return(OK);
        case  BSIM3v1S_MOD_WK1:
          value->rValue = model->BSIM3v1Swk1;
            return(OK);
        case  BSIM3v1S_MOD_WKT1:
          value->rValue = model->BSIM3v1Swkt1;
            return(OK);
        case  BSIM3v1S_MOD_WKT1L:
          value->rValue = model->BSIM3v1Swkt1l;
            return(OK);
        case  BSIM3v1S_MOD_WKT2 :
          value->rValue = model->BSIM3v1Swkt2;
            return(OK);
        case  BSIM3v1S_MOD_WK2 :
          value->rValue = model->BSIM3v1Swk2;
            return(OK);
        case  BSIM3v1S_MOD_WK3:
          value->rValue = model->BSIM3v1Swk3;
            return(OK);
        case  BSIM3v1S_MOD_WK3B:
          value->rValue = model->BSIM3v1Swk3b;
            return(OK);
        case  BSIM3v1S_MOD_WW0:
          value->rValue = model->BSIM3v1Sww0;
            return(OK);
        case  BSIM3v1S_MOD_WNLX:
          value->rValue = model->BSIM3v1Swnlx;
            return(OK);
        case  BSIM3v1S_MOD_WDVT0:                
          value->rValue = model->BSIM3v1Swdvt0;
            return(OK);
        case  BSIM3v1S_MOD_WDVT1 :             
          value->rValue = model->BSIM3v1Swdvt1;
            return(OK);
        case  BSIM3v1S_MOD_WDVT2 :             
          value->rValue = model->BSIM3v1Swdvt2;
            return(OK);
        case  BSIM3v1S_MOD_WDVT0W :                
          value->rValue = model->BSIM3v1Swdvt0w;
            return(OK);
        case  BSIM3v1S_MOD_WDVT1W :             
          value->rValue = model->BSIM3v1Swdvt1w;
            return(OK);
        case  BSIM3v1S_MOD_WDVT2W :             
          value->rValue = model->BSIM3v1Swdvt2w;
            return(OK);
        case  BSIM3v1S_MOD_WDROUT :           
          value->rValue = model->BSIM3v1Swdrout;
            return(OK);
        case  BSIM3v1S_MOD_WDSUB :           
          value->rValue = model->BSIM3v1Swdsub;
            return(OK);
        case BSIM3v1S_MOD_WVTH0:
            value->rValue = model->BSIM3v1Swvth0; 
            return(OK);
        case BSIM3v1S_MOD_WUA:
            value->rValue = model->BSIM3v1Swua; 
            return(OK);
        case BSIM3v1S_MOD_WUA1:
            value->rValue = model->BSIM3v1Swua1; 
            return(OK);
        case BSIM3v1S_MOD_WUB:
            value->rValue = model->BSIM3v1Swub;  
            return(OK);
        case BSIM3v1S_MOD_WUB1:
            value->rValue = model->BSIM3v1Swub1;  
            return(OK);
        case BSIM3v1S_MOD_WUC:
            value->rValue = model->BSIM3v1Swuc; 
            return(OK);
        case BSIM3v1S_MOD_WUC1:
            value->rValue = model->BSIM3v1Swuc1; 
            return(OK);
        case BSIM3v1S_MOD_WU0:
            value->rValue = model->BSIM3v1Swu0;
            return(OK);
        case BSIM3v1S_MOD_WUTE:
            value->rValue = model->BSIM3v1Swute;
            return(OK);
        case BSIM3v1S_MOD_WVOFF:
            value->rValue = model->BSIM3v1Swvoff;
            return(OK);
        case BSIM3v1S_MOD_WDELTA:
            value->rValue = model->BSIM3v1Swdelta;
            return(OK);
        case BSIM3v1S_MOD_WRDSW:
            value->rValue = model->BSIM3v1Swrdsw; 
            return(OK);             
        case BSIM3v1S_MOD_WPRWB:
            value->rValue = model->BSIM3v1Swprwb; 
            return(OK);             
        case BSIM3v1S_MOD_WPRWG:
            value->rValue = model->BSIM3v1Swprwg; 
            return(OK);             
        case BSIM3v1S_MOD_WPRT:
            value->rValue = model->BSIM3v1Swprt; 
            return(OK);              
        case BSIM3v1S_MOD_WETA0:
            value->rValue = model->BSIM3v1Sweta0; 
            return(OK);               
        case BSIM3v1S_MOD_WETAB:
            value->rValue = model->BSIM3v1Swetab; 
            return(OK);               
        case BSIM3v1S_MOD_WPCLM:
            value->rValue = model->BSIM3v1Swpclm; 
            return(OK);               
        case BSIM3v1S_MOD_WPDIBL1:
            value->rValue = model->BSIM3v1Swpdibl1; 
            return(OK);               
        case BSIM3v1S_MOD_WPDIBL2:
            value->rValue = model->BSIM3v1Swpdibl2; 
            return(OK);               
        case BSIM3v1S_MOD_WPDIBLB:
            value->rValue = model->BSIM3v1Swpdiblb; 
            return(OK);               
        case BSIM3v1S_MOD_WPSCBE1:
            value->rValue = model->BSIM3v1Swpscbe1; 
            return(OK);               
        case BSIM3v1S_MOD_WPSCBE2:
            value->rValue = model->BSIM3v1Swpscbe2; 
            return(OK);               
        case BSIM3v1S_MOD_WPVAG:
            value->rValue = model->BSIM3v1Swpvag; 
            return(OK);               
        case BSIM3v1S_MOD_WWR:
            value->rValue = model->BSIM3v1Swwr;
            return(OK);
        case BSIM3v1S_MOD_WDWG:
            value->rValue = model->BSIM3v1Swdwg;
            return(OK);
        case BSIM3v1S_MOD_WDWB:
            value->rValue = model->BSIM3v1Swdwb;
            return(OK);
        case BSIM3v1S_MOD_WB0:
            value->rValue = model->BSIM3v1Swb0;
            return(OK);
        case BSIM3v1S_MOD_WB1:
            value->rValue = model->BSIM3v1Swb1;
            return(OK);
        case BSIM3v1S_MOD_WALPHA0:
            value->rValue = model->BSIM3v1Swalpha0;
            return(OK);
        case BSIM3v1S_MOD_WBETA0:
            value->rValue = model->BSIM3v1Swbeta0;
            return(OK);

        case BSIM3v1S_MOD_WELM:
            value->rValue = model->BSIM3v1Swelm;
            return(OK);
        case BSIM3v1S_MOD_WCGSL:
            value->rValue = model->BSIM3v1Swcgsl;
            return(OK);
        case BSIM3v1S_MOD_WCGDL:
            value->rValue = model->BSIM3v1Swcgdl;
            return(OK);
        case BSIM3v1S_MOD_WCKAPPA:
            value->rValue = model->BSIM3v1Swckappa;
            return(OK);
        case BSIM3v1S_MOD_WCF:
            value->rValue = model->BSIM3v1Swcf;
            return(OK);
        case BSIM3v1S_MOD_WCLC:
            value->rValue = model->BSIM3v1Swclc;
            return(OK);
        case BSIM3v1S_MOD_WCLE:
            value->rValue = model->BSIM3v1Swcle;
            return(OK);
        case BSIM3v1S_MOD_WVFBCV:
            value->rValue = model->BSIM3v1Swvfbcv;
            return(OK);

	/* Cross-term dependence */
        case  BSIM3v1S_MOD_PCDSC :
          value->rValue = model->BSIM3v1Spcdsc;
            return(OK);
        case  BSIM3v1S_MOD_PCDSCB :
          value->rValue = model->BSIM3v1Spcdscb;
            return(OK);
        case  BSIM3v1S_MOD_PCDSCD :
          value->rValue = model->BSIM3v1Spcdscd;
            return(OK);
         case  BSIM3v1S_MOD_PCIT :
          value->rValue = model->BSIM3v1Spcit;
            return(OK);
        case  BSIM3v1S_MOD_PNFACTOR :
          value->rValue = model->BSIM3v1Spnfactor;
            return(OK);
        case BSIM3v1S_MOD_PXJ:
            value->rValue = model->BSIM3v1Spxj;
            return(OK);
        case BSIM3v1S_MOD_PVSAT:
            value->rValue = model->BSIM3v1Spvsat;
            return(OK);
        case BSIM3v1S_MOD_PAT:
            value->rValue = model->BSIM3v1Spat;
            return(OK);
        case BSIM3v1S_MOD_PA0:
            value->rValue = model->BSIM3v1Spa0;
            return(OK);
        case BSIM3v1S_MOD_PAGS:
            value->rValue = model->BSIM3v1Spags;
            return(OK);
        case BSIM3v1S_MOD_PA1:
            value->rValue = model->BSIM3v1Spa1;
            return(OK);
        case BSIM3v1S_MOD_PA2:
            value->rValue = model->BSIM3v1Spa2;
            return(OK);
        case BSIM3v1S_MOD_PKETA:
            value->rValue = model->BSIM3v1Spketa;
            return(OK);   
        case BSIM3v1S_MOD_PNSUB:
            value->rValue = model->BSIM3v1Spnsub;
            return(OK);
        case BSIM3v1S_MOD_PNPEAK:
            value->rValue = model->BSIM3v1Spnpeak;
            return(OK);
        case BSIM3v1S_MOD_PNGATE:
            value->rValue = model->BSIM3v1Spngate;
            return(OK);
        case BSIM3v1S_MOD_PGAMMA1:
            value->rValue = model->BSIM3v1Spgamma1;
            return(OK);
        case BSIM3v1S_MOD_PGAMMA2:
            value->rValue = model->BSIM3v1Spgamma2;
            return(OK);
        case BSIM3v1S_MOD_PVBX:
            value->rValue = model->BSIM3v1Spvbx;
            return(OK);
        case BSIM3v1S_MOD_PVBM:
            value->rValue = model->BSIM3v1Spvbm;
            return(OK);
        case BSIM3v1S_MOD_PXT:
            value->rValue = model->BSIM3v1Spxt;
            return(OK);
        case  BSIM3v1S_MOD_PK1:
          value->rValue = model->BSIM3v1Spk1;
            return(OK);
        case  BSIM3v1S_MOD_PKT1:
          value->rValue = model->BSIM3v1Spkt1;
            return(OK);
        case  BSIM3v1S_MOD_PKT1L:
          value->rValue = model->BSIM3v1Spkt1l;
            return(OK);
        case  BSIM3v1S_MOD_PKT2 :
          value->rValue = model->BSIM3v1Spkt2;
            return(OK);
        case  BSIM3v1S_MOD_PK2 :
          value->rValue = model->BSIM3v1Spk2;
            return(OK);
        case  BSIM3v1S_MOD_PK3:
          value->rValue = model->BSIM3v1Spk3;
            return(OK);
        case  BSIM3v1S_MOD_PK3B:
          value->rValue = model->BSIM3v1Spk3b;
            return(OK);
        case  BSIM3v1S_MOD_PW0:
          value->rValue = model->BSIM3v1Spw0;
            return(OK);
        case  BSIM3v1S_MOD_PNLX:
          value->rValue = model->BSIM3v1Spnlx;
            return(OK);
        case  BSIM3v1S_MOD_PDVT0 :                
          value->rValue = model->BSIM3v1Spdvt0;
            return(OK);
        case  BSIM3v1S_MOD_PDVT1 :             
          value->rValue = model->BSIM3v1Spdvt1;
            return(OK);
        case  BSIM3v1S_MOD_PDVT2 :             
          value->rValue = model->BSIM3v1Spdvt2;
            return(OK);
        case  BSIM3v1S_MOD_PDVT0W :                
          value->rValue = model->BSIM3v1Spdvt0w;
            return(OK);
        case  BSIM3v1S_MOD_PDVT1W :             
          value->rValue = model->BSIM3v1Spdvt1w;
            return(OK);
        case  BSIM3v1S_MOD_PDVT2W :             
          value->rValue = model->BSIM3v1Spdvt2w;
            return(OK);
        case  BSIM3v1S_MOD_PDROUT :           
          value->rValue = model->BSIM3v1Spdrout;
            return(OK);
        case  BSIM3v1S_MOD_PDSUB :           
          value->rValue = model->BSIM3v1Spdsub;
            return(OK);
        case BSIM3v1S_MOD_PVTH0:
            value->rValue = model->BSIM3v1Spvth0; 
            return(OK);
        case BSIM3v1S_MOD_PUA:
            value->rValue = model->BSIM3v1Spua; 
            return(OK);
        case BSIM3v1S_MOD_PUA1:
            value->rValue = model->BSIM3v1Spua1; 
            return(OK);
        case BSIM3v1S_MOD_PUB:
            value->rValue = model->BSIM3v1Spub;  
            return(OK);
        case BSIM3v1S_MOD_PUB1:
            value->rValue = model->BSIM3v1Spub1;  
            return(OK);
        case BSIM3v1S_MOD_PUC:
            value->rValue = model->BSIM3v1Spuc; 
            return(OK);
        case BSIM3v1S_MOD_PUC1:
            value->rValue = model->BSIM3v1Spuc1; 
            return(OK);
        case BSIM3v1S_MOD_PU0:
            value->rValue = model->BSIM3v1Spu0;
            return(OK);
        case BSIM3v1S_MOD_PUTE:
            value->rValue = model->BSIM3v1Spute;
            return(OK);
        case BSIM3v1S_MOD_PVOFF:
            value->rValue = model->BSIM3v1Spvoff;
            return(OK);
        case BSIM3v1S_MOD_PDELTA:
            value->rValue = model->BSIM3v1Spdelta;
            return(OK);
        case BSIM3v1S_MOD_PRDSW:
            value->rValue = model->BSIM3v1Sprdsw; 
            return(OK);             
        case BSIM3v1S_MOD_PPRWB:
            value->rValue = model->BSIM3v1Spprwb; 
            return(OK);             
        case BSIM3v1S_MOD_PPRWG:
            value->rValue = model->BSIM3v1Spprwg; 
            return(OK);             
        case BSIM3v1S_MOD_PPRT:
            value->rValue = model->BSIM3v1Spprt; 
            return(OK);              
        case BSIM3v1S_MOD_PETA0:
            value->rValue = model->BSIM3v1Speta0; 
            return(OK);               
        case BSIM3v1S_MOD_PETAB:
            value->rValue = model->BSIM3v1Spetab; 
            return(OK);               
        case BSIM3v1S_MOD_PPCLM:
            value->rValue = model->BSIM3v1Sppclm; 
            return(OK);               
        case BSIM3v1S_MOD_PPDIBL1:
            value->rValue = model->BSIM3v1Sppdibl1; 
            return(OK);               
        case BSIM3v1S_MOD_PPDIBL2:
            value->rValue = model->BSIM3v1Sppdibl2; 
            return(OK);               
        case BSIM3v1S_MOD_PPDIBLB:
            value->rValue = model->BSIM3v1Sppdiblb; 
            return(OK);               
        case BSIM3v1S_MOD_PPSCBE1:
            value->rValue = model->BSIM3v1Sppscbe1; 
            return(OK);               
        case BSIM3v1S_MOD_PPSCBE2:
            value->rValue = model->BSIM3v1Sppscbe2; 
            return(OK);               
        case BSIM3v1S_MOD_PPVAG:
            value->rValue = model->BSIM3v1Sppvag; 
            return(OK);               
        case BSIM3v1S_MOD_PWR:
            value->rValue = model->BSIM3v1Spwr;
            return(OK);
        case BSIM3v1S_MOD_PDWG:
            value->rValue = model->BSIM3v1Spdwg;
            return(OK);
        case BSIM3v1S_MOD_PDWB:
            value->rValue = model->BSIM3v1Spdwb;
            return(OK);
        case BSIM3v1S_MOD_PB0:
            value->rValue = model->BSIM3v1Spb0;
            return(OK);
        case BSIM3v1S_MOD_PB1:
            value->rValue = model->BSIM3v1Spb1;
            return(OK);
        case BSIM3v1S_MOD_PALPHA0:
            value->rValue = model->BSIM3v1Spalpha0;
            return(OK);
        case BSIM3v1S_MOD_PBETA0:
            value->rValue = model->BSIM3v1Spbeta0;
            return(OK);

        case BSIM3v1S_MOD_PELM:
            value->rValue = model->BSIM3v1Spelm;
            return(OK);
        case BSIM3v1S_MOD_PCGSL:
            value->rValue = model->BSIM3v1Spcgsl;
            return(OK);
        case BSIM3v1S_MOD_PCGDL:
            value->rValue = model->BSIM3v1Spcgdl;
            return(OK);
        case BSIM3v1S_MOD_PCKAPPA:
            value->rValue = model->BSIM3v1Spckappa;
            return(OK);
        case BSIM3v1S_MOD_PCF:
            value->rValue = model->BSIM3v1Spcf;
            return(OK);
        case BSIM3v1S_MOD_PCLC:
            value->rValue = model->BSIM3v1Spclc;
            return(OK);
        case BSIM3v1S_MOD_PCLE:
            value->rValue = model->BSIM3v1Spcle;
            return(OK);
        case BSIM3v1S_MOD_PVFBCV:
            value->rValue = model->BSIM3v1Spvfbcv;
            return(OK);

        case  BSIM3v1S_MOD_TNOM :
          value->rValue = model->BSIM3v1Stnom;
            return(OK);
        case BSIM3v1S_MOD_CGSO:
            value->rValue = model->BSIM3v1Scgso; 
            return(OK);
        case BSIM3v1S_MOD_CGDO:
            value->rValue = model->BSIM3v1Scgdo; 
            return(OK);
        case BSIM3v1S_MOD_CGBO:
            value->rValue = model->BSIM3v1Scgbo; 
            return(OK);
        case BSIM3v1S_MOD_XPART:
            value->rValue = model->BSIM3v1Sxpart; 
            return(OK);
        case BSIM3v1S_MOD_RSH:
            value->rValue = model->BSIM3v1SsheetResistance; 
            return(OK);
        case BSIM3v1S_MOD_JS:
            value->rValue = model->BSIM3v1SjctSatCurDensity; 
            return(OK);
        case BSIM3v1S_MOD_JSW:
            value->rValue = model->BSIM3v1SjctSidewallSatCurDensity; 
            return(OK);
        case BSIM3v1S_MOD_PB:
            value->rValue = model->BSIM3v1SbulkJctPotential; 
            return(OK);
        case BSIM3v1S_MOD_MJ:
            value->rValue = model->BSIM3v1SbulkJctBotGradingCoeff; 
            return(OK);
        case BSIM3v1S_MOD_PBSW:
            value->rValue = model->BSIM3v1SsidewallJctPotential; 
            return(OK);
        case BSIM3v1S_MOD_MJSW:
            value->rValue = model->BSIM3v1SbulkJctSideGradingCoeff; 
            return(OK);
        case BSIM3v1S_MOD_CJ:
            value->rValue = model->BSIM3v1SunitAreaJctCap; 
            return(OK);
        case BSIM3v1S_MOD_CJSW:
            value->rValue = model->BSIM3v1SunitLengthSidewallJctCap; 
            return(OK);
        case BSIM3v1S_MOD_PBSWG:
            value->rValue = model->BSIM3v1SGatesidewallJctPotential; 
            return(OK);
        case BSIM3v1S_MOD_MJSWG:
            value->rValue = model->BSIM3v1SbulkJctGateSideGradingCoeff; 
            return(OK);
        case BSIM3v1S_MOD_CJSWG:
            value->rValue = model->BSIM3v1SunitLengthGateSidewallJctCap; 
            return(OK);
        case BSIM3v1S_MOD_NJ:
            value->rValue = model->BSIM3v1SjctEmissionCoeff; 
            return(OK);
        case BSIM3v1S_MOD_XTI:
            value->rValue = model->BSIM3v1SjctTempExponent; 
            return(OK);
        case BSIM3v1S_MOD_LINT:
            value->rValue = model->BSIM3v1SLint; 
            return(OK);
        case BSIM3v1S_MOD_LL:
            value->rValue = model->BSIM3v1SLl;
            return(OK);
        case BSIM3v1S_MOD_LLN:
            value->rValue = model->BSIM3v1SLln;
            return(OK);
        case BSIM3v1S_MOD_LW:
            value->rValue = model->BSIM3v1SLw;
            return(OK);
        case BSIM3v1S_MOD_LWN:
            value->rValue = model->BSIM3v1SLwn;
            return(OK);
        case BSIM3v1S_MOD_LWL:
            value->rValue = model->BSIM3v1SLwl;
            return(OK);
        case BSIM3v1S_MOD_LMIN:
            value->rValue = model->BSIM3v1SLmin;
            return(OK);
        case BSIM3v1S_MOD_LMAX:
            value->rValue = model->BSIM3v1SLmax;
            return(OK);
        case BSIM3v1S_MOD_WINT:
            value->rValue = model->BSIM3v1SWint;
            return(OK);
        case BSIM3v1S_MOD_WL:
            value->rValue = model->BSIM3v1SWl;
            return(OK);
        case BSIM3v1S_MOD_WLN:
            value->rValue = model->BSIM3v1SWln;
            return(OK);
        case BSIM3v1S_MOD_WW:
            value->rValue = model->BSIM3v1SWw;
            return(OK);
        case BSIM3v1S_MOD_WWN:
            value->rValue = model->BSIM3v1SWwn;
            return(OK);
        case BSIM3v1S_MOD_WWL:
            value->rValue = model->BSIM3v1SWwl;
            return(OK);
        case BSIM3v1S_MOD_WMIN:
            value->rValue = model->BSIM3v1SWmin;
            return(OK);
        case BSIM3v1S_MOD_WMAX:
            value->rValue = model->BSIM3v1SWmax;
            return(OK);
        case BSIM3v1S_MOD_NOIA:
            value->rValue = model->BSIM3v1SoxideTrapDensityA;
            return(OK);
        case BSIM3v1S_MOD_NOIB:
            value->rValue = model->BSIM3v1SoxideTrapDensityB;
            return(OK);
        case BSIM3v1S_MOD_NOIC:
            value->rValue = model->BSIM3v1SoxideTrapDensityC;
            return(OK);
        case BSIM3v1S_MOD_EM:
            value->rValue = model->BSIM3v1Sem;
            return(OK);
        case BSIM3v1S_MOD_EF:
            value->rValue = model->BSIM3v1Sef;
            return(OK);
        case BSIM3v1S_MOD_AF:
            value->rValue = model->BSIM3v1Saf;
            return(OK);
        case BSIM3v1S_MOD_KF:
            value->rValue = model->BSIM3v1Skf;
            return(OK);
        default:
            return(E_BADPARM);
    }
    /* NOTREACHED */
}



