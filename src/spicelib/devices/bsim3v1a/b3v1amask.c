/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1995 Min-Chie Jeng and Mansun Chan.
Modified by Paolo Nenzi 2002
File: b3v1amask.c
**********/

#include "ngspice.h"
#include "ifsim.h"
#include "cktdefs.h"
#include "devdefs.h"
#include "bsim3v1adef.h"
#include "sperror.h"
#include "suffix.h"

int
BSIM3v1AmAsk(CKTcircuit *ckt, GENmodel *inst, int which, IFvalue *value)
{
    BSIM3v1Amodel *model = (BSIM3v1Amodel *)inst;
    switch(which) 
    {   case BSIM3v1A_MOD_MOBMOD:
            value->iValue = model->BSIM3v1AmobMod; 
            return(OK);
        case BSIM3v1A_MOD_BINUNIT:
            value->iValue = model->BSIM3v1AbinUnit; 
            return(OK);
        case BSIM3v1A_MOD_CAPMOD:
            value->iValue = model->BSIM3v1AcapMod; 
            return(OK);
        case BSIM3v1A_MOD_NQSMOD:
            value->iValue = model->BSIM3v1AnqsMod; 
            return(OK);
        case BSIM3v1A_MOD_NOIMOD:
            value->iValue = model->BSIM3v1AnoiMod; 
            return(OK);
        case  BSIM3v1A_MOD_TOX :
          value->rValue = model->BSIM3v1Atox;
            return(OK);
        case  BSIM3v1A_MOD_CDSC :
          value->rValue = model->BSIM3v1Acdsc;
            return(OK);
        case  BSIM3v1A_MOD_CDSCB :
          value->rValue = model->BSIM3v1Acdscb;
            return(OK);

        case  BSIM3v1A_MOD_CDSCD :
          value->rValue = model->BSIM3v1Acdscd;
            return(OK);

        case  BSIM3v1A_MOD_CIT :
          value->rValue = model->BSIM3v1Acit;
            return(OK);
        case  BSIM3v1A_MOD_NFACTOR :
          value->rValue = model->BSIM3v1Anfactor;
            return(OK);
        case BSIM3v1A_MOD_XJ:
            value->rValue = model->BSIM3v1Axj;
            return(OK);
        case BSIM3v1A_MOD_VSAT:
            value->rValue = model->BSIM3v1Avsat;
            return(OK);
        case BSIM3v1A_MOD_AT:
            value->rValue = model->BSIM3v1Aat;
            return(OK);
        case BSIM3v1A_MOD_A0:
            value->rValue = model->BSIM3v1Aa0;
            return(OK);

        case BSIM3v1A_MOD_AGS:
            value->rValue = model->BSIM3v1Aags;
            return(OK);

        case BSIM3v1A_MOD_A1:
            value->rValue = model->BSIM3v1Aa1;
            return(OK);
        case BSIM3v1A_MOD_A2:
            value->rValue = model->BSIM3v1Aa2;
            return(OK);
        case BSIM3v1A_MOD_KETA:
            value->rValue = model->BSIM3v1Aketa;
            return(OK);   
        case BSIM3v1A_MOD_NSUB:
            value->rValue = model->BSIM3v1Ansub;
            return(OK);
        case BSIM3v1A_MOD_NPEAK:
            value->rValue = model->BSIM3v1Anpeak;
            return(OK);
        case BSIM3v1A_MOD_NGATE:
            value->rValue = model->BSIM3v1Angate;
            return(OK);
        case BSIM3v1A_MOD_GAMMA1:
            value->rValue = model->BSIM3v1Agamma1;
            return(OK);
        case BSIM3v1A_MOD_GAMMA2:
            value->rValue = model->BSIM3v1Agamma2;
            return(OK);
        case BSIM3v1A_MOD_VBX:
            value->rValue = model->BSIM3v1Avbx;
            return(OK);
        case BSIM3v1A_MOD_VBM:
            value->rValue = model->BSIM3v1Avbm;
            return(OK);
        case BSIM3v1A_MOD_XT:
            value->rValue = model->BSIM3v1Axt;
            return(OK);
        case  BSIM3v1A_MOD_K1:
          value->rValue = model->BSIM3v1Ak1;
            return(OK);
        case  BSIM3v1A_MOD_KT1:
          value->rValue = model->BSIM3v1Akt1;
            return(OK);
        case  BSIM3v1A_MOD_KT1L:
          value->rValue = model->BSIM3v1Akt1l;
            return(OK);
        case  BSIM3v1A_MOD_KT2 :
          value->rValue = model->BSIM3v1Akt2;
            return(OK);
        case  BSIM3v1A_MOD_K2 :
          value->rValue = model->BSIM3v1Ak2;
            return(OK);
        case  BSIM3v1A_MOD_K3:
          value->rValue = model->BSIM3v1Ak3;
            return(OK);
        case  BSIM3v1A_MOD_K3B:
          value->rValue = model->BSIM3v1Ak3b;
            return(OK);
        case  BSIM3v1A_MOD_W0:
          value->rValue = model->BSIM3v1Aw0;
            return(OK);
        case  BSIM3v1A_MOD_NLX:
          value->rValue = model->BSIM3v1Anlx;
            return(OK);
        case  BSIM3v1A_MOD_DVT0 :                
          value->rValue = model->BSIM3v1Advt0;
            return(OK);
        case  BSIM3v1A_MOD_DVT1 :             
          value->rValue = model->BSIM3v1Advt1;
            return(OK);
        case  BSIM3v1A_MOD_DVT2 :             
          value->rValue = model->BSIM3v1Advt2;
            return(OK);
        case  BSIM3v1A_MOD_DVT0W :                
          value->rValue = model->BSIM3v1Advt0w;
            return(OK);
        case  BSIM3v1A_MOD_DVT1W :             
          value->rValue = model->BSIM3v1Advt1w;
            return(OK);
        case  BSIM3v1A_MOD_DVT2W :             
          value->rValue = model->BSIM3v1Advt2w;
            return(OK);
        case  BSIM3v1A_MOD_DROUT :           
          value->rValue = model->BSIM3v1Adrout;
            return(OK);
        case  BSIM3v1A_MOD_DSUB :           
          value->rValue = model->BSIM3v1Adsub;
            return(OK);
        case BSIM3v1A_MOD_VTH0:
            value->rValue = model->BSIM3v1Avth0; 
            return(OK);
        case BSIM3v1A_MOD_UA:
            value->rValue = model->BSIM3v1Aua; 
            return(OK);
        case BSIM3v1A_MOD_UA1:
            value->rValue = model->BSIM3v1Aua1; 
            return(OK);
        case BSIM3v1A_MOD_UB:
            value->rValue = model->BSIM3v1Aub;  
            return(OK);
        case BSIM3v1A_MOD_UB1:
            value->rValue = model->BSIM3v1Aub1;  
            return(OK);
        case BSIM3v1A_MOD_UC:
            value->rValue = model->BSIM3v1Auc; 
            return(OK);
        case BSIM3v1A_MOD_UC1:
            value->rValue = model->BSIM3v1Auc1; 
            return(OK);
        case BSIM3v1A_MOD_U0:
            value->rValue = model->BSIM3v1Au0;
            return(OK);
        case BSIM3v1A_MOD_UTE:
            value->rValue = model->BSIM3v1Aute;
            return(OK);
        case BSIM3v1A_MOD_VOFF:
            value->rValue = model->BSIM3v1Avoff;
            return(OK);
        case BSIM3v1A_MOD_DELTA:
            value->rValue = model->BSIM3v1Adelta;
            return(OK);
        case BSIM3v1A_MOD_RDSW:
            value->rValue = model->BSIM3v1Ardsw; 
            return(OK);             
        case BSIM3v1A_MOD_PRWG:
            value->rValue = model->BSIM3v1Aprwg; 
            return(OK);             
        case BSIM3v1A_MOD_PRWB:
            value->rValue = model->BSIM3v1Aprwb; 
            return(OK);             
        case BSIM3v1A_MOD_PRT:
            value->rValue = model->BSIM3v1Aprt; 
            return(OK);              
        case BSIM3v1A_MOD_ETA0:
            value->rValue = model->BSIM3v1Aeta0; 
            return(OK);               
        case BSIM3v1A_MOD_ETAB:
            value->rValue = model->BSIM3v1Aetab; 
            return(OK);               
        case BSIM3v1A_MOD_PCLM:
            value->rValue = model->BSIM3v1Apclm; 
            return(OK);               
        case BSIM3v1A_MOD_PDIBL1:
            value->rValue = model->BSIM3v1Apdibl1; 
            return(OK);               
        case BSIM3v1A_MOD_PDIBL2:
            value->rValue = model->BSIM3v1Apdibl2; 
            return(OK);               
        case BSIM3v1A_MOD_PDIBLB:
            value->rValue = model->BSIM3v1Apdiblb; 
            return(OK);               
        case BSIM3v1A_MOD_PSCBE1:
            value->rValue = model->BSIM3v1Apscbe1; 
            return(OK);               
        case BSIM3v1A_MOD_PSCBE2:
            value->rValue = model->BSIM3v1Apscbe2; 
            return(OK);               
        case BSIM3v1A_MOD_PVAG:
            value->rValue = model->BSIM3v1Apvag; 
            return(OK);               
        case BSIM3v1A_MOD_WR:
            value->rValue = model->BSIM3v1Awr;
            return(OK);
        case BSIM3v1A_MOD_DWG:
            value->rValue = model->BSIM3v1Adwg;
            return(OK);
        case BSIM3v1A_MOD_DWB:
            value->rValue = model->BSIM3v1Adwb;
            return(OK);
        case BSIM3v1A_MOD_B0:
            value->rValue = model->BSIM3v1Ab0;
            return(OK);
        case BSIM3v1A_MOD_B1:
            value->rValue = model->BSIM3v1Ab1;
            return(OK);
        case BSIM3v1A_MOD_ALPHA0:
            value->rValue = model->BSIM3v1Aalpha0;
            return(OK);
        case BSIM3v1A_MOD_BETA0:
            value->rValue = model->BSIM3v1Abeta0;
            return(OK);

        case BSIM3v1A_MOD_ELM:
            value->rValue = model->BSIM3v1Aelm;
            return(OK);
        case BSIM3v1A_MOD_CGSL:
            value->rValue = model->BSIM3v1Acgsl;
            return(OK);
        case BSIM3v1A_MOD_CGDL:
            value->rValue = model->BSIM3v1Acgdl;
            return(OK);
        case BSIM3v1A_MOD_CKAPPA:
            value->rValue = model->BSIM3v1Ackappa;
            return(OK);
        case BSIM3v1A_MOD_CF:
            value->rValue = model->BSIM3v1Acf;
            return(OK);
        case BSIM3v1A_MOD_CLC:
            value->rValue = model->BSIM3v1Aclc;
            return(OK);
        case BSIM3v1A_MOD_CLE:
            value->rValue = model->BSIM3v1Acle;
            return(OK);
        case BSIM3v1A_MOD_DWC:
            value->rValue = model->BSIM3v1Adwc;
            return(OK);
        case BSIM3v1A_MOD_DLC:
            value->rValue = model->BSIM3v1Adlc;
            return(OK);

	/* Length dependence */
        case  BSIM3v1A_MOD_LCDSC :
          value->rValue = model->BSIM3v1Alcdsc;
            return(OK);
        case  BSIM3v1A_MOD_LCDSCB :
          value->rValue = model->BSIM3v1Alcdscb;
            return(OK);
        case  BSIM3v1A_MOD_LCDSCD :
          value->rValue = model->BSIM3v1Alcdscd;
            return(OK);
        case  BSIM3v1A_MOD_LCIT :
          value->rValue = model->BSIM3v1Alcit;
            return(OK);
        case  BSIM3v1A_MOD_LNFACTOR :
          value->rValue = model->BSIM3v1Alnfactor;
            return(OK);
        case BSIM3v1A_MOD_LXJ:
            value->rValue = model->BSIM3v1Alxj;
            return(OK);
        case BSIM3v1A_MOD_LVSAT:
            value->rValue = model->BSIM3v1Alvsat;
            return(OK);
        case BSIM3v1A_MOD_LAT:
            value->rValue = model->BSIM3v1Alat;
            return(OK);
        case BSIM3v1A_MOD_LA0:
            value->rValue = model->BSIM3v1Ala0;
            return(OK);
        case BSIM3v1A_MOD_LAGS:
            value->rValue = model->BSIM3v1Alags;
            return(OK);
        case BSIM3v1A_MOD_LA1:
            value->rValue = model->BSIM3v1Ala1;
            return(OK);
        case BSIM3v1A_MOD_LA2:
            value->rValue = model->BSIM3v1Ala2;
            return(OK);
        case BSIM3v1A_MOD_LKETA:
            value->rValue = model->BSIM3v1Alketa;
            return(OK);   
        case BSIM3v1A_MOD_LNSUB:
            value->rValue = model->BSIM3v1Alnsub;
            return(OK);
        case BSIM3v1A_MOD_LNPEAK:
            value->rValue = model->BSIM3v1Alnpeak;
            return(OK);
        case BSIM3v1A_MOD_LNGATE:
            value->rValue = model->BSIM3v1Alngate;
            return(OK);
        case BSIM3v1A_MOD_LGAMMA1:
            value->rValue = model->BSIM3v1Algamma1;
            return(OK);
        case BSIM3v1A_MOD_LGAMMA2:
            value->rValue = model->BSIM3v1Algamma2;
            return(OK);
        case BSIM3v1A_MOD_LVBX:
            value->rValue = model->BSIM3v1Alvbx;
            return(OK);
        case BSIM3v1A_MOD_LVBM:
            value->rValue = model->BSIM3v1Alvbm;
            return(OK);
        case BSIM3v1A_MOD_LXT:
            value->rValue = model->BSIM3v1Alxt;
            return(OK);
        case  BSIM3v1A_MOD_LK1:
          value->rValue = model->BSIM3v1Alk1;
            return(OK);
        case  BSIM3v1A_MOD_LKT1:
          value->rValue = model->BSIM3v1Alkt1;
            return(OK);
        case  BSIM3v1A_MOD_LKT1L:
          value->rValue = model->BSIM3v1Alkt1l;
            return(OK);
        case  BSIM3v1A_MOD_LKT2 :
          value->rValue = model->BSIM3v1Alkt2;
            return(OK);
        case  BSIM3v1A_MOD_LK2 :
          value->rValue = model->BSIM3v1Alk2;
            return(OK);
        case  BSIM3v1A_MOD_LK3:
          value->rValue = model->BSIM3v1Alk3;
            return(OK);
        case  BSIM3v1A_MOD_LK3B:
          value->rValue = model->BSIM3v1Alk3b;
            return(OK);
        case  BSIM3v1A_MOD_LW0:
          value->rValue = model->BSIM3v1Alw0;
            return(OK);
        case  BSIM3v1A_MOD_LNLX:
          value->rValue = model->BSIM3v1Alnlx;
            return(OK);
        case  BSIM3v1A_MOD_LDVT0:                
          value->rValue = model->BSIM3v1Aldvt0;
            return(OK);
        case  BSIM3v1A_MOD_LDVT1 :             
          value->rValue = model->BSIM3v1Aldvt1;
            return(OK);
        case  BSIM3v1A_MOD_LDVT2 :             
          value->rValue = model->BSIM3v1Aldvt2;
            return(OK);
        case  BSIM3v1A_MOD_LDVT0W :                
          value->rValue = model->BSIM3v1Aldvt0w;
            return(OK);
        case  BSIM3v1A_MOD_LDVT1W :             
          value->rValue = model->BSIM3v1Aldvt1w;
            return(OK);
        case  BSIM3v1A_MOD_LDVT2W :             
          value->rValue = model->BSIM3v1Aldvt2w;
            return(OK);
        case  BSIM3v1A_MOD_LDROUT :           
          value->rValue = model->BSIM3v1Aldrout;
            return(OK);
        case  BSIM3v1A_MOD_LDSUB :           
          value->rValue = model->BSIM3v1Aldsub;
            return(OK);
        case BSIM3v1A_MOD_LVTH0:
            value->rValue = model->BSIM3v1Alvth0; 
            return(OK);
        case BSIM3v1A_MOD_LUA:
            value->rValue = model->BSIM3v1Alua; 
            return(OK);
        case BSIM3v1A_MOD_LUA1:
            value->rValue = model->BSIM3v1Alua1; 
            return(OK);
        case BSIM3v1A_MOD_LUB:
            value->rValue = model->BSIM3v1Alub;  
            return(OK);
        case BSIM3v1A_MOD_LUB1:
            value->rValue = model->BSIM3v1Alub1;  
            return(OK);
        case BSIM3v1A_MOD_LUC:
            value->rValue = model->BSIM3v1Aluc; 
            return(OK);
        case BSIM3v1A_MOD_LUC1:
            value->rValue = model->BSIM3v1Aluc1; 
            return(OK);
        case BSIM3v1A_MOD_LU0:
            value->rValue = model->BSIM3v1Alu0;
            return(OK);
        case BSIM3v1A_MOD_LUTE:
            value->rValue = model->BSIM3v1Alute;
            return(OK);
        case BSIM3v1A_MOD_LVOFF:
            value->rValue = model->BSIM3v1Alvoff;
            return(OK);
        case BSIM3v1A_MOD_LDELTA:
            value->rValue = model->BSIM3v1Aldelta;
            return(OK);
        case BSIM3v1A_MOD_LRDSW:
            value->rValue = model->BSIM3v1Alrdsw; 
            return(OK);             
        case BSIM3v1A_MOD_LPRWB:
            value->rValue = model->BSIM3v1Alprwb; 
            return(OK);             
        case BSIM3v1A_MOD_LPRWG:
            value->rValue = model->BSIM3v1Alprwg; 
            return(OK);             
        case BSIM3v1A_MOD_LPRT:
            value->rValue = model->BSIM3v1Alprt; 
            return(OK);              
        case BSIM3v1A_MOD_LETA0:
            value->rValue = model->BSIM3v1Aleta0; 
            return(OK);               
        case BSIM3v1A_MOD_LETAB:
            value->rValue = model->BSIM3v1Aletab; 
            return(OK);               
        case BSIM3v1A_MOD_LPCLM:
            value->rValue = model->BSIM3v1Alpclm; 
            return(OK);               
        case BSIM3v1A_MOD_LPDIBL1:
            value->rValue = model->BSIM3v1Alpdibl1; 
            return(OK);               
        case BSIM3v1A_MOD_LPDIBL2:
            value->rValue = model->BSIM3v1Alpdibl2; 
            return(OK);               
        case BSIM3v1A_MOD_LPDIBLB:
            value->rValue = model->BSIM3v1Alpdiblb; 
            return(OK);               
        case BSIM3v1A_MOD_LPSCBE1:
            value->rValue = model->BSIM3v1Alpscbe1; 
            return(OK);               
        case BSIM3v1A_MOD_LPSCBE2:
            value->rValue = model->BSIM3v1Alpscbe2; 
            return(OK);               
        case BSIM3v1A_MOD_LPVAG:
            value->rValue = model->BSIM3v1Alpvag; 
            return(OK);               
        case BSIM3v1A_MOD_LWR:
            value->rValue = model->BSIM3v1Alwr;
            return(OK);
        case BSIM3v1A_MOD_LDWG:
            value->rValue = model->BSIM3v1Aldwg;
            return(OK);
        case BSIM3v1A_MOD_LDWB:
            value->rValue = model->BSIM3v1Aldwb;
            return(OK);
        case BSIM3v1A_MOD_LB0:
            value->rValue = model->BSIM3v1Alb0;
            return(OK);
        case BSIM3v1A_MOD_LB1:
            value->rValue = model->BSIM3v1Alb1;
            return(OK);
        case BSIM3v1A_MOD_LALPHA0:
            value->rValue = model->BSIM3v1Alalpha0;
            return(OK);
        case BSIM3v1A_MOD_LBETA0:
            value->rValue = model->BSIM3v1Albeta0;
            return(OK);

        case BSIM3v1A_MOD_LELM:
            value->rValue = model->BSIM3v1Alelm;
            return(OK);
        case BSIM3v1A_MOD_LCGSL:
            value->rValue = model->BSIM3v1Alcgsl;
            return(OK);
        case BSIM3v1A_MOD_LCGDL:
            value->rValue = model->BSIM3v1Alcgdl;
            return(OK);
        case BSIM3v1A_MOD_LCKAPPA:
            value->rValue = model->BSIM3v1Alckappa;
            return(OK);
        case BSIM3v1A_MOD_LCF:
            value->rValue = model->BSIM3v1Alcf;
            return(OK);
        case BSIM3v1A_MOD_LCLC:
            value->rValue = model->BSIM3v1Alclc;
            return(OK);
        case BSIM3v1A_MOD_LCLE:
            value->rValue = model->BSIM3v1Alcle;
            return(OK);

	/* Width dependence */
        case  BSIM3v1A_MOD_WCDSC :
          value->rValue = model->BSIM3v1Awcdsc;
            return(OK);
        case  BSIM3v1A_MOD_WCDSCB :
          value->rValue = model->BSIM3v1Awcdscb;
            return(OK);
        case  BSIM3v1A_MOD_WCDSCD :
          value->rValue = model->BSIM3v1Awcdscd;
            return(OK);
        case  BSIM3v1A_MOD_WCIT :
          value->rValue = model->BSIM3v1Awcit;
            return(OK);
        case  BSIM3v1A_MOD_WNFACTOR :
          value->rValue = model->BSIM3v1Awnfactor;
            return(OK);
        case BSIM3v1A_MOD_WXJ:
            value->rValue = model->BSIM3v1Awxj;
            return(OK);
        case BSIM3v1A_MOD_WVSAT:
            value->rValue = model->BSIM3v1Awvsat;
            return(OK);
        case BSIM3v1A_MOD_WAT:
            value->rValue = model->BSIM3v1Awat;
            return(OK);
        case BSIM3v1A_MOD_WA0:
            value->rValue = model->BSIM3v1Awa0;
            return(OK);
        case BSIM3v1A_MOD_WAGS:
            value->rValue = model->BSIM3v1Awags;
            return(OK);
        case BSIM3v1A_MOD_WA1:
            value->rValue = model->BSIM3v1Awa1;
            return(OK);
        case BSIM3v1A_MOD_WA2:
            value->rValue = model->BSIM3v1Awa2;
            return(OK);
        case BSIM3v1A_MOD_WKETA:
            value->rValue = model->BSIM3v1Awketa;
            return(OK);   
        case BSIM3v1A_MOD_WNSUB:
            value->rValue = model->BSIM3v1Awnsub;
            return(OK);
        case BSIM3v1A_MOD_WNPEAK:
            value->rValue = model->BSIM3v1Awnpeak;
            return(OK);
        case BSIM3v1A_MOD_WNGATE:
            value->rValue = model->BSIM3v1Awngate;
            return(OK);
        case BSIM3v1A_MOD_WGAMMA1:
            value->rValue = model->BSIM3v1Awgamma1;
            return(OK);
        case BSIM3v1A_MOD_WGAMMA2:
            value->rValue = model->BSIM3v1Awgamma2;
            return(OK);
        case BSIM3v1A_MOD_WVBX:
            value->rValue = model->BSIM3v1Awvbx;
            return(OK);
        case BSIM3v1A_MOD_WVBM:
            value->rValue = model->BSIM3v1Awvbm;
            return(OK);
        case BSIM3v1A_MOD_WXT:
            value->rValue = model->BSIM3v1Awxt;
            return(OK);
        case  BSIM3v1A_MOD_WK1:
          value->rValue = model->BSIM3v1Awk1;
            return(OK);
        case  BSIM3v1A_MOD_WKT1:
          value->rValue = model->BSIM3v1Awkt1;
            return(OK);
        case  BSIM3v1A_MOD_WKT1L:
          value->rValue = model->BSIM3v1Awkt1l;
            return(OK);
        case  BSIM3v1A_MOD_WKT2 :
          value->rValue = model->BSIM3v1Awkt2;
            return(OK);
        case  BSIM3v1A_MOD_WK2 :
          value->rValue = model->BSIM3v1Awk2;
            return(OK);
        case  BSIM3v1A_MOD_WK3:
          value->rValue = model->BSIM3v1Awk3;
            return(OK);
        case  BSIM3v1A_MOD_WK3B:
          value->rValue = model->BSIM3v1Awk3b;
            return(OK);
        case  BSIM3v1A_MOD_WW0:
          value->rValue = model->BSIM3v1Aww0;
            return(OK);
        case  BSIM3v1A_MOD_WNLX:
          value->rValue = model->BSIM3v1Awnlx;
            return(OK);
        case  BSIM3v1A_MOD_WDVT0:                
          value->rValue = model->BSIM3v1Awdvt0;
            return(OK);
        case  BSIM3v1A_MOD_WDVT1 :             
          value->rValue = model->BSIM3v1Awdvt1;
            return(OK);
        case  BSIM3v1A_MOD_WDVT2 :             
          value->rValue = model->BSIM3v1Awdvt2;
            return(OK);
        case  BSIM3v1A_MOD_WDVT0W :                
          value->rValue = model->BSIM3v1Awdvt0w;
            return(OK);
        case  BSIM3v1A_MOD_WDVT1W :             
          value->rValue = model->BSIM3v1Awdvt1w;
            return(OK);
        case  BSIM3v1A_MOD_WDVT2W :             
          value->rValue = model->BSIM3v1Awdvt2w;
            return(OK);
        case  BSIM3v1A_MOD_WDROUT :           
          value->rValue = model->BSIM3v1Awdrout;
            return(OK);
        case  BSIM3v1A_MOD_WDSUB :           
          value->rValue = model->BSIM3v1Awdsub;
            return(OK);
        case BSIM3v1A_MOD_WVTH0:
            value->rValue = model->BSIM3v1Awvth0; 
            return(OK);
        case BSIM3v1A_MOD_WUA:
            value->rValue = model->BSIM3v1Awua; 
            return(OK);
        case BSIM3v1A_MOD_WUA1:
            value->rValue = model->BSIM3v1Awua1; 
            return(OK);
        case BSIM3v1A_MOD_WUB:
            value->rValue = model->BSIM3v1Awub;  
            return(OK);
        case BSIM3v1A_MOD_WUB1:
            value->rValue = model->BSIM3v1Awub1;  
            return(OK);
        case BSIM3v1A_MOD_WUC:
            value->rValue = model->BSIM3v1Awuc; 
            return(OK);
        case BSIM3v1A_MOD_WUC1:
            value->rValue = model->BSIM3v1Awuc1; 
            return(OK);
        case BSIM3v1A_MOD_WU0:
            value->rValue = model->BSIM3v1Awu0;
            return(OK);
        case BSIM3v1A_MOD_WUTE:
            value->rValue = model->BSIM3v1Awute;
            return(OK);
        case BSIM3v1A_MOD_WVOFF:
            value->rValue = model->BSIM3v1Awvoff;
            return(OK);
        case BSIM3v1A_MOD_WDELTA:
            value->rValue = model->BSIM3v1Awdelta;
            return(OK);
        case BSIM3v1A_MOD_WRDSW:
            value->rValue = model->BSIM3v1Awrdsw; 
            return(OK);             
        case BSIM3v1A_MOD_WPRWB:
            value->rValue = model->BSIM3v1Awprwb; 
            return(OK);             
        case BSIM3v1A_MOD_WPRWG:
            value->rValue = model->BSIM3v1Awprwg; 
            return(OK);             
        case BSIM3v1A_MOD_WPRT:
            value->rValue = model->BSIM3v1Awprt; 
            return(OK);              
        case BSIM3v1A_MOD_WETA0:
            value->rValue = model->BSIM3v1Aweta0; 
            return(OK);               
        case BSIM3v1A_MOD_WETAB:
            value->rValue = model->BSIM3v1Awetab; 
            return(OK);               
        case BSIM3v1A_MOD_WPCLM:
            value->rValue = model->BSIM3v1Awpclm; 
            return(OK);               
        case BSIM3v1A_MOD_WPDIBL1:
            value->rValue = model->BSIM3v1Awpdibl1; 
            return(OK);               
        case BSIM3v1A_MOD_WPDIBL2:
            value->rValue = model->BSIM3v1Awpdibl2; 
            return(OK);               
        case BSIM3v1A_MOD_WPDIBLB:
            value->rValue = model->BSIM3v1Awpdiblb; 
            return(OK);               
        case BSIM3v1A_MOD_WPSCBE1:
            value->rValue = model->BSIM3v1Awpscbe1; 
            return(OK);               
        case BSIM3v1A_MOD_WPSCBE2:
            value->rValue = model->BSIM3v1Awpscbe2; 
            return(OK);               
        case BSIM3v1A_MOD_WPVAG:
            value->rValue = model->BSIM3v1Awpvag; 
            return(OK);               
        case BSIM3v1A_MOD_WWR:
            value->rValue = model->BSIM3v1Awwr;
            return(OK);
        case BSIM3v1A_MOD_WDWG:
            value->rValue = model->BSIM3v1Awdwg;
            return(OK);
        case BSIM3v1A_MOD_WDWB:
            value->rValue = model->BSIM3v1Awdwb;
            return(OK);
        case BSIM3v1A_MOD_WB0:
            value->rValue = model->BSIM3v1Awb0;
            return(OK);
        case BSIM3v1A_MOD_WB1:
            value->rValue = model->BSIM3v1Awb1;
            return(OK);
        case BSIM3v1A_MOD_WALPHA0:
            value->rValue = model->BSIM3v1Awalpha0;
            return(OK);
        case BSIM3v1A_MOD_WBETA0:
            value->rValue = model->BSIM3v1Awbeta0;
            return(OK);

        case BSIM3v1A_MOD_WELM:
            value->rValue = model->BSIM3v1Awelm;
            return(OK);
        case BSIM3v1A_MOD_WCGSL:
            value->rValue = model->BSIM3v1Awcgsl;
            return(OK);
        case BSIM3v1A_MOD_WCGDL:
            value->rValue = model->BSIM3v1Awcgdl;
            return(OK);
        case BSIM3v1A_MOD_WCKAPPA:
            value->rValue = model->BSIM3v1Awckappa;
            return(OK);
        case BSIM3v1A_MOD_WCF:
            value->rValue = model->BSIM3v1Awcf;
            return(OK);
        case BSIM3v1A_MOD_WCLC:
            value->rValue = model->BSIM3v1Awclc;
            return(OK);
        case BSIM3v1A_MOD_WCLE:
            value->rValue = model->BSIM3v1Awcle;
            return(OK);

	/* Cross-term dependence */
        case  BSIM3v1A_MOD_PCDSC :
          value->rValue = model->BSIM3v1Apcdsc;
            return(OK);
        case  BSIM3v1A_MOD_PCDSCB :
          value->rValue = model->BSIM3v1Apcdscb;
            return(OK);
        case  BSIM3v1A_MOD_PCDSCD :
          value->rValue = model->BSIM3v1Apcdscd;
            return(OK);
         case  BSIM3v1A_MOD_PCIT :
          value->rValue = model->BSIM3v1Apcit;
            return(OK);
        case  BSIM3v1A_MOD_PNFACTOR :
          value->rValue = model->BSIM3v1Apnfactor;
            return(OK);
        case BSIM3v1A_MOD_PXJ:
            value->rValue = model->BSIM3v1Apxj;
            return(OK);
        case BSIM3v1A_MOD_PVSAT:
            value->rValue = model->BSIM3v1Apvsat;
            return(OK);
        case BSIM3v1A_MOD_PAT:
            value->rValue = model->BSIM3v1Apat;
            return(OK);
        case BSIM3v1A_MOD_PA0:
            value->rValue = model->BSIM3v1Apa0;
            return(OK);
        case BSIM3v1A_MOD_PAGS:
            value->rValue = model->BSIM3v1Apags;
            return(OK);
        case BSIM3v1A_MOD_PA1:
            value->rValue = model->BSIM3v1Apa1;
            return(OK);
        case BSIM3v1A_MOD_PA2:
            value->rValue = model->BSIM3v1Apa2;
            return(OK);
        case BSIM3v1A_MOD_PKETA:
            value->rValue = model->BSIM3v1Apketa;
            return(OK);   
        case BSIM3v1A_MOD_PNSUB:
            value->rValue = model->BSIM3v1Apnsub;
            return(OK);
        case BSIM3v1A_MOD_PNPEAK:
            value->rValue = model->BSIM3v1Apnpeak;
            return(OK);
        case BSIM3v1A_MOD_PNGATE:
            value->rValue = model->BSIM3v1Apngate;
            return(OK);
        case BSIM3v1A_MOD_PGAMMA1:
            value->rValue = model->BSIM3v1Apgamma1;
            return(OK);
        case BSIM3v1A_MOD_PGAMMA2:
            value->rValue = model->BSIM3v1Apgamma2;
            return(OK);
        case BSIM3v1A_MOD_PVBX:
            value->rValue = model->BSIM3v1Apvbx;
            return(OK);
        case BSIM3v1A_MOD_PVBM:
            value->rValue = model->BSIM3v1Apvbm;
            return(OK);
        case BSIM3v1A_MOD_PXT:
            value->rValue = model->BSIM3v1Apxt;
            return(OK);
        case  BSIM3v1A_MOD_PK1:
          value->rValue = model->BSIM3v1Apk1;
            return(OK);
        case  BSIM3v1A_MOD_PKT1:
          value->rValue = model->BSIM3v1Apkt1;
            return(OK);
        case  BSIM3v1A_MOD_PKT1L:
          value->rValue = model->BSIM3v1Apkt1l;
            return(OK);
        case  BSIM3v1A_MOD_PKT2 :
          value->rValue = model->BSIM3v1Apkt2;
            return(OK);
        case  BSIM3v1A_MOD_PK2 :
          value->rValue = model->BSIM3v1Apk2;
            return(OK);
        case  BSIM3v1A_MOD_PK3:
          value->rValue = model->BSIM3v1Apk3;
            return(OK);
        case  BSIM3v1A_MOD_PK3B:
          value->rValue = model->BSIM3v1Apk3b;
            return(OK);
        case  BSIM3v1A_MOD_PW0:
          value->rValue = model->BSIM3v1Apw0;
            return(OK);
        case  BSIM3v1A_MOD_PNLX:
          value->rValue = model->BSIM3v1Apnlx;
            return(OK);
        case  BSIM3v1A_MOD_PDVT0 :                
          value->rValue = model->BSIM3v1Apdvt0;
            return(OK);
        case  BSIM3v1A_MOD_PDVT1 :             
          value->rValue = model->BSIM3v1Apdvt1;
            return(OK);
        case  BSIM3v1A_MOD_PDVT2 :             
          value->rValue = model->BSIM3v1Apdvt2;
            return(OK);
        case  BSIM3v1A_MOD_PDVT0W :                
          value->rValue = model->BSIM3v1Apdvt0w;
            return(OK);
        case  BSIM3v1A_MOD_PDVT1W :             
          value->rValue = model->BSIM3v1Apdvt1w;
            return(OK);
        case  BSIM3v1A_MOD_PDVT2W :             
          value->rValue = model->BSIM3v1Apdvt2w;
            return(OK);
        case  BSIM3v1A_MOD_PDROUT :           
          value->rValue = model->BSIM3v1Apdrout;
            return(OK);
        case  BSIM3v1A_MOD_PDSUB :           
          value->rValue = model->BSIM3v1Apdsub;
            return(OK);
        case BSIM3v1A_MOD_PVTH0:
            value->rValue = model->BSIM3v1Apvth0; 
            return(OK);
        case BSIM3v1A_MOD_PUA:
            value->rValue = model->BSIM3v1Apua; 
            return(OK);
        case BSIM3v1A_MOD_PUA1:
            value->rValue = model->BSIM3v1Apua1; 
            return(OK);
        case BSIM3v1A_MOD_PUB:
            value->rValue = model->BSIM3v1Apub;  
            return(OK);
        case BSIM3v1A_MOD_PUB1:
            value->rValue = model->BSIM3v1Apub1;  
            return(OK);
        case BSIM3v1A_MOD_PUC:
            value->rValue = model->BSIM3v1Apuc; 
            return(OK);
        case BSIM3v1A_MOD_PUC1:
            value->rValue = model->BSIM3v1Apuc1; 
            return(OK);
        case BSIM3v1A_MOD_PU0:
            value->rValue = model->BSIM3v1Apu0;
            return(OK);
        case BSIM3v1A_MOD_PUTE:
            value->rValue = model->BSIM3v1Apute;
            return(OK);
        case BSIM3v1A_MOD_PVOFF:
            value->rValue = model->BSIM3v1Apvoff;
            return(OK);
        case BSIM3v1A_MOD_PDELTA:
            value->rValue = model->BSIM3v1Apdelta;
            return(OK);
        case BSIM3v1A_MOD_PRDSW:
            value->rValue = model->BSIM3v1Aprdsw; 
            return(OK);             
        case BSIM3v1A_MOD_PPRWB:
            value->rValue = model->BSIM3v1Apprwb; 
            return(OK);             
        case BSIM3v1A_MOD_PPRWG:
            value->rValue = model->BSIM3v1Apprwg; 
            return(OK);             
        case BSIM3v1A_MOD_PPRT:
            value->rValue = model->BSIM3v1Apprt; 
            return(OK);              
        case BSIM3v1A_MOD_PETA0:
            value->rValue = model->BSIM3v1Apeta0; 
            return(OK);               
        case BSIM3v1A_MOD_PETAB:
            value->rValue = model->BSIM3v1Apetab; 
            return(OK);               
        case BSIM3v1A_MOD_PPCLM:
            value->rValue = model->BSIM3v1Appclm; 
            return(OK);               
        case BSIM3v1A_MOD_PPDIBL1:
            value->rValue = model->BSIM3v1Appdibl1; 
            return(OK);               
        case BSIM3v1A_MOD_PPDIBL2:
            value->rValue = model->BSIM3v1Appdibl2; 
            return(OK);               
        case BSIM3v1A_MOD_PPDIBLB:
            value->rValue = model->BSIM3v1Appdiblb; 
            return(OK);               
        case BSIM3v1A_MOD_PPSCBE1:
            value->rValue = model->BSIM3v1Appscbe1; 
            return(OK);               
        case BSIM3v1A_MOD_PPSCBE2:
            value->rValue = model->BSIM3v1Appscbe2; 
            return(OK);               
        case BSIM3v1A_MOD_PPVAG:
            value->rValue = model->BSIM3v1Appvag; 
            return(OK);               
        case BSIM3v1A_MOD_PWR:
            value->rValue = model->BSIM3v1Apwr;
            return(OK);
        case BSIM3v1A_MOD_PDWG:
            value->rValue = model->BSIM3v1Apdwg;
            return(OK);
        case BSIM3v1A_MOD_PDWB:
            value->rValue = model->BSIM3v1Apdwb;
            return(OK);
        case BSIM3v1A_MOD_PB0:
            value->rValue = model->BSIM3v1Apb0;
            return(OK);
        case BSIM3v1A_MOD_PB1:
            value->rValue = model->BSIM3v1Apb1;
            return(OK);
        case BSIM3v1A_MOD_PALPHA0:
            value->rValue = model->BSIM3v1Apalpha0;
            return(OK);
        case BSIM3v1A_MOD_PBETA0:
            value->rValue = model->BSIM3v1Apbeta0;
            return(OK);

        case BSIM3v1A_MOD_PELM:
            value->rValue = model->BSIM3v1Apelm;
            return(OK);
        case BSIM3v1A_MOD_PCGSL:
            value->rValue = model->BSIM3v1Apcgsl;
            return(OK);
        case BSIM3v1A_MOD_PCGDL:
            value->rValue = model->BSIM3v1Apcgdl;
            return(OK);
        case BSIM3v1A_MOD_PCKAPPA:
            value->rValue = model->BSIM3v1Apckappa;
            return(OK);
        case BSIM3v1A_MOD_PCF:
            value->rValue = model->BSIM3v1Apcf;
            return(OK);
        case BSIM3v1A_MOD_PCLC:
            value->rValue = model->BSIM3v1Apclc;
            return(OK);
        case BSIM3v1A_MOD_PCLE:
            value->rValue = model->BSIM3v1Apcle;
            return(OK);

        case  BSIM3v1A_MOD_TNOM :
          value->rValue = model->BSIM3v1Atnom;
            return(OK);
        case BSIM3v1A_MOD_CGSO:
            value->rValue = model->BSIM3v1Acgso; 
            return(OK);
        case BSIM3v1A_MOD_CGDO:
            value->rValue = model->BSIM3v1Acgdo; 
            return(OK);
        case BSIM3v1A_MOD_CGBO:
            value->rValue = model->BSIM3v1Acgbo; 
            return(OK);
        case BSIM3v1A_MOD_XPART:
            value->rValue = model->BSIM3v1Axpart; 
            return(OK);
        case BSIM3v1A_MOD_RSH:
            value->rValue = model->BSIM3v1AsheetResistance; 
            return(OK);
        case BSIM3v1A_MOD_JS:
            value->rValue = model->BSIM3v1AjctSatCurDensity; 
            return(OK);
        case BSIM3v1A_MOD_PB:
            value->rValue = model->BSIM3v1AbulkJctPotential; 
            return(OK);
        case BSIM3v1A_MOD_MJ:
            value->rValue = model->BSIM3v1AbulkJctBotGradingCoeff; 
            return(OK);
        case BSIM3v1A_MOD_PBSW:
            value->rValue = model->BSIM3v1AsidewallJctPotential; 
            return(OK);
        case BSIM3v1A_MOD_MJSW:
            value->rValue = model->BSIM3v1AbulkJctSideGradingCoeff; 
            return(OK);
        case BSIM3v1A_MOD_CJ:
            value->rValue = model->BSIM3v1AunitAreaJctCap; 
            return(OK);
        case BSIM3v1A_MOD_CJSW:
            value->rValue = model->BSIM3v1AunitLengthSidewallJctCap; 
            return(OK);
        case BSIM3v1A_MOD_LINT:
            value->rValue = model->BSIM3v1ALint; 
            return(OK);
        case BSIM3v1A_MOD_LL:
            value->rValue = model->BSIM3v1ALl;
            return(OK);
        case BSIM3v1A_MOD_LLN:
            value->rValue = model->BSIM3v1ALln;
            return(OK);
        case BSIM3v1A_MOD_LW:
            value->rValue = model->BSIM3v1ALw;
            return(OK);
        case BSIM3v1A_MOD_LWN:
            value->rValue = model->BSIM3v1ALwn;
            return(OK);
        case BSIM3v1A_MOD_LWL:
            value->rValue = model->BSIM3v1ALwl;
            return(OK);
        case BSIM3v1A_MOD_LMIN:
            value->rValue = model->BSIM3v1ALmin;
            return(OK);
        case BSIM3v1A_MOD_LMAX:
            value->rValue = model->BSIM3v1ALmax;
            return(OK);
        case BSIM3v1A_MOD_WINT:
            value->rValue = model->BSIM3v1AWint;
            return(OK);
        case BSIM3v1A_MOD_WL:
            value->rValue = model->BSIM3v1AWl;
            return(OK);
        case BSIM3v1A_MOD_WLN:
            value->rValue = model->BSIM3v1AWln;
            return(OK);
        case BSIM3v1A_MOD_WW:
            value->rValue = model->BSIM3v1AWw;
            return(OK);
        case BSIM3v1A_MOD_WWN:
            value->rValue = model->BSIM3v1AWwn;
            return(OK);
        case BSIM3v1A_MOD_WWL:
            value->rValue = model->BSIM3v1AWwl;
            return(OK);
        case BSIM3v1A_MOD_WMIN:
            value->rValue = model->BSIM3v1AWmin;
            return(OK);
        case BSIM3v1A_MOD_WMAX:
            value->rValue = model->BSIM3v1AWmax;
            return(OK);
        case BSIM3v1A_MOD_NOIA:
            value->rValue = model->BSIM3v1AoxideTrapDensityA;
            return(OK);
        case BSIM3v1A_MOD_NOIB:
            value->rValue = model->BSIM3v1AoxideTrapDensityB;
            return(OK);
        case BSIM3v1A_MOD_NOIC:
            value->rValue = model->BSIM3v1AoxideTrapDensityC;
            return(OK);
        case BSIM3v1A_MOD_EM:
            value->rValue = model->BSIM3v1Aem;
            return(OK);
        case BSIM3v1A_MOD_EF:
            value->rValue = model->BSIM3v1Aef;
            return(OK);
        case BSIM3v1A_MOD_AF:
            value->rValue = model->BSIM3v1Aaf;
            return(OK);
        case BSIM3v1A_MOD_KF:
            value->rValue = model->BSIM3v1Akf;
            return(OK);
        default:
            return(E_BADPARM);
    }
    /* NOTREACHED */
}



