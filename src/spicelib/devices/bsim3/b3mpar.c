/**** BSIM3v3.3.0, Released by Xuemei Xi 07/29/2005 ****/

/**********
 * Copyright 2004 Regents of the University of California. All rights reserved.
 * File: b3mpar.c of BSIM3v3.3.0
 * Author: 1995 Min-Chie Jeng and Mansun Chan. 
 * Author: 1997-1999 Weidong Liu.
 * Author: 2001  Xuemei Xi
 **********/

#include "ngspice/ngspice.h"
#include "bsim3def.h"
#include "ngspice/ifsim.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
BSIM3mParam(
int param,
IFvalue *value,
GENmodel *inMod)
{
    BSIM3model *mod = (BSIM3model*)inMod;
    switch(param)
    {   case  BSIM3_MOD_MOBMOD :
            mod->BSIM3mobMod = value->iValue;
            mod->BSIM3mobModGiven = TRUE;
            break;
        case  BSIM3_MOD_BINUNIT :
            mod->BSIM3binUnit = value->iValue;
            mod->BSIM3binUnitGiven = TRUE;
            break;
        case  BSIM3_MOD_PARAMCHK :
            mod->BSIM3paramChk = value->iValue;
            mod->BSIM3paramChkGiven = TRUE;
            break;
        case  BSIM3_MOD_CAPMOD :
            mod->BSIM3capMod = value->iValue;
            mod->BSIM3capModGiven = TRUE;
            break;
        case BSIM3_MOD_ACMMOD:
            mod->BSIM3acmMod = value->iValue;
            mod->BSIM3acmModGiven = TRUE;
            break;
        case BSIM3_MOD_CALCACM:
            mod->BSIM3calcacm = value->iValue;
            mod->BSIM3calcacmGiven = TRUE;
            break;
        case  BSIM3_MOD_NOIMOD :
            mod->BSIM3noiMod = value->iValue;
            mod->BSIM3noiModGiven = TRUE;
            break;
        case  BSIM3_MOD_NQSMOD :
            mod->BSIM3nqsMod = value->iValue;
            mod->BSIM3nqsModGiven = TRUE;
            break;
        case  BSIM3_MOD_ACNQSMOD :
            mod->BSIM3acnqsMod = value->iValue;
            mod->BSIM3acnqsModGiven = TRUE;
            break;
        case  BSIM3_MOD_VERSION :
            mod->BSIM3version = value->sValue;
            mod->BSIM3versionGiven = TRUE;
            break;
        case  BSIM3_MOD_TOX :
            mod->BSIM3tox = value->rValue;
            mod->BSIM3toxGiven = TRUE;
            break;
        case  BSIM3_MOD_TOXM :
            mod->BSIM3toxm = value->rValue;
            mod->BSIM3toxmGiven = TRUE;
            break;

        case  BSIM3_MOD_CDSC :
            mod->BSIM3cdsc = value->rValue;
            mod->BSIM3cdscGiven = TRUE;
            break;
        case  BSIM3_MOD_CDSCB :
            mod->BSIM3cdscb = value->rValue;
            mod->BSIM3cdscbGiven = TRUE;
            break;

        case  BSIM3_MOD_CDSCD :
            mod->BSIM3cdscd = value->rValue;
            mod->BSIM3cdscdGiven = TRUE;
            break;

        case  BSIM3_MOD_CIT :
            mod->BSIM3cit = value->rValue;
            mod->BSIM3citGiven = TRUE;
            break;
        case  BSIM3_MOD_NFACTOR :
            mod->BSIM3nfactor = value->rValue;
            mod->BSIM3nfactorGiven = TRUE;
            break;
        case BSIM3_MOD_XJ:
            mod->BSIM3xj = value->rValue;
            mod->BSIM3xjGiven = TRUE;
            break;
        case BSIM3_MOD_VSAT:
            mod->BSIM3vsat = value->rValue;
            mod->BSIM3vsatGiven = TRUE;
            break;
        case BSIM3_MOD_A0:
            mod->BSIM3a0 = value->rValue;
            mod->BSIM3a0Given = TRUE;
            break;
        
        case BSIM3_MOD_AGS:
            mod->BSIM3ags= value->rValue;
            mod->BSIM3agsGiven = TRUE;
            break;
        
        case BSIM3_MOD_A1:
            mod->BSIM3a1 = value->rValue;
            mod->BSIM3a1Given = TRUE;
            break;
        case BSIM3_MOD_A2:
            mod->BSIM3a2 = value->rValue;
            mod->BSIM3a2Given = TRUE;
            break;
        case BSIM3_MOD_AT:
            mod->BSIM3at = value->rValue;
            mod->BSIM3atGiven = TRUE;
            break;
        case BSIM3_MOD_KETA:
            mod->BSIM3keta = value->rValue;
            mod->BSIM3ketaGiven = TRUE;
            break;    
        case BSIM3_MOD_NSUB:
            mod->BSIM3nsub = value->rValue;
            mod->BSIM3nsubGiven = TRUE;
            break;
        case BSIM3_MOD_NPEAK:
            mod->BSIM3npeak = value->rValue;
            mod->BSIM3npeakGiven = TRUE;
            if (mod->BSIM3npeak > 1.0e20)
                mod->BSIM3npeak *= 1.0e-6;
            break;
        case BSIM3_MOD_NGATE:
            mod->BSIM3ngate = value->rValue;
            mod->BSIM3ngateGiven = TRUE;
            if (mod->BSIM3ngate > 1.000001e24)
                mod->BSIM3ngate *= 1.0e-6;
            break;
        case BSIM3_MOD_GAMMA1:
            mod->BSIM3gamma1 = value->rValue;
            mod->BSIM3gamma1Given = TRUE;
            break;
        case BSIM3_MOD_GAMMA2:
            mod->BSIM3gamma2 = value->rValue;
            mod->BSIM3gamma2Given = TRUE;
            break;
        case BSIM3_MOD_VBX:
            mod->BSIM3vbx = value->rValue;
            mod->BSIM3vbxGiven = TRUE;
            break;
        case BSIM3_MOD_VBM:
            mod->BSIM3vbm = value->rValue;
            mod->BSIM3vbmGiven = TRUE;
            break;
        case BSIM3_MOD_XT:
            mod->BSIM3xt = value->rValue;
            mod->BSIM3xtGiven = TRUE;
            break;
        case  BSIM3_MOD_K1:
            mod->BSIM3k1 = value->rValue;
            mod->BSIM3k1Given = TRUE;
            break;
        case  BSIM3_MOD_KT1:
            mod->BSIM3kt1 = value->rValue;
            mod->BSIM3kt1Given = TRUE;
            break;
        case  BSIM3_MOD_KT1L:
            mod->BSIM3kt1l = value->rValue;
            mod->BSIM3kt1lGiven = TRUE;
            break;
        case  BSIM3_MOD_KT2:
            mod->BSIM3kt2 = value->rValue;
            mod->BSIM3kt2Given = TRUE;
            break;
        case  BSIM3_MOD_K2:
            mod->BSIM3k2 = value->rValue;
            mod->BSIM3k2Given = TRUE;
            break;
        case  BSIM3_MOD_K3:
            mod->BSIM3k3 = value->rValue;
            mod->BSIM3k3Given = TRUE;
            break;
        case  BSIM3_MOD_K3B:
            mod->BSIM3k3b = value->rValue;
            mod->BSIM3k3bGiven = TRUE;
            break;
        case  BSIM3_MOD_NLX:
            mod->BSIM3nlx = value->rValue;
            mod->BSIM3nlxGiven = TRUE;
            break;
        case  BSIM3_MOD_W0:
            mod->BSIM3w0 = value->rValue;
            mod->BSIM3w0Given = TRUE;
            break;
        case  BSIM3_MOD_DVT0:               
            mod->BSIM3dvt0 = value->rValue;
            mod->BSIM3dvt0Given = TRUE;
            break;
        case  BSIM3_MOD_DVT1:             
            mod->BSIM3dvt1 = value->rValue;
            mod->BSIM3dvt1Given = TRUE;
            break;
        case  BSIM3_MOD_DVT2:             
            mod->BSIM3dvt2 = value->rValue;
            mod->BSIM3dvt2Given = TRUE;
            break;
        case  BSIM3_MOD_DVT0W:               
            mod->BSIM3dvt0w = value->rValue;
            mod->BSIM3dvt0wGiven = TRUE;
            break;
        case  BSIM3_MOD_DVT1W:             
            mod->BSIM3dvt1w = value->rValue;
            mod->BSIM3dvt1wGiven = TRUE;
            break;
        case  BSIM3_MOD_DVT2W:             
            mod->BSIM3dvt2w = value->rValue;
            mod->BSIM3dvt2wGiven = TRUE;
            break;
        case  BSIM3_MOD_DROUT:             
            mod->BSIM3drout = value->rValue;
            mod->BSIM3droutGiven = TRUE;
            break;
        case  BSIM3_MOD_DSUB:             
            mod->BSIM3dsub = value->rValue;
            mod->BSIM3dsubGiven = TRUE;
            break;
        case BSIM3_MOD_VTH0:
            mod->BSIM3vth0 = value->rValue;
            mod->BSIM3vth0Given = TRUE;
            break;
        case BSIM3_MOD_UA:
            mod->BSIM3ua = value->rValue;
            mod->BSIM3uaGiven = TRUE;
            break;
        case BSIM3_MOD_UA1:
            mod->BSIM3ua1 = value->rValue;
            mod->BSIM3ua1Given = TRUE;
            break;
        case BSIM3_MOD_UB:
            mod->BSIM3ub = value->rValue;
            mod->BSIM3ubGiven = TRUE;
            break;
        case BSIM3_MOD_UB1:
            mod->BSIM3ub1 = value->rValue;
            mod->BSIM3ub1Given = TRUE;
            break;
        case BSIM3_MOD_UC:
            mod->BSIM3uc = value->rValue;
            mod->BSIM3ucGiven = TRUE;
            break;
        case BSIM3_MOD_UC1:
            mod->BSIM3uc1 = value->rValue;
            mod->BSIM3uc1Given = TRUE;
            break;
        case  BSIM3_MOD_U0 :
            mod->BSIM3u0 = value->rValue;
            mod->BSIM3u0Given = TRUE;
            break;
        case  BSIM3_MOD_UTE :
            mod->BSIM3ute = value->rValue;
            mod->BSIM3uteGiven = TRUE;
            break;
        case BSIM3_MOD_VOFF:
            mod->BSIM3voff = value->rValue;
            mod->BSIM3voffGiven = TRUE;
            break;
        case  BSIM3_MOD_DELTA :
            mod->BSIM3delta = value->rValue;
            mod->BSIM3deltaGiven = TRUE;
            break;
        case BSIM3_MOD_RDSW:
            mod->BSIM3rdsw = value->rValue;
            mod->BSIM3rdswGiven = TRUE;
            break;                     
        case BSIM3_MOD_PRWG:
            mod->BSIM3prwg = value->rValue;
            mod->BSIM3prwgGiven = TRUE;
            break;                     
        case BSIM3_MOD_PRWB:
            mod->BSIM3prwb = value->rValue;
            mod->BSIM3prwbGiven = TRUE;
            break;                     
        case BSIM3_MOD_PRT:
            mod->BSIM3prt = value->rValue;
            mod->BSIM3prtGiven = TRUE;
            break;                     
        case BSIM3_MOD_ETA0:
            mod->BSIM3eta0 = value->rValue;
            mod->BSIM3eta0Given = TRUE;
            break;                 
        case BSIM3_MOD_ETAB:
            mod->BSIM3etab = value->rValue;
            mod->BSIM3etabGiven = TRUE;
            break;                 
        case BSIM3_MOD_PCLM:
            mod->BSIM3pclm = value->rValue;
            mod->BSIM3pclmGiven = TRUE;
            break;                 
        case BSIM3_MOD_PDIBL1:
            mod->BSIM3pdibl1 = value->rValue;
            mod->BSIM3pdibl1Given = TRUE;
            break;                 
        case BSIM3_MOD_PDIBL2:
            mod->BSIM3pdibl2 = value->rValue;
            mod->BSIM3pdibl2Given = TRUE;
            break;                 
        case BSIM3_MOD_PDIBLB:
            mod->BSIM3pdiblb = value->rValue;
            mod->BSIM3pdiblbGiven = TRUE;
            break;                 
        case BSIM3_MOD_PSCBE1:
            mod->BSIM3pscbe1 = value->rValue;
            mod->BSIM3pscbe1Given = TRUE;
            break;                 
        case BSIM3_MOD_PSCBE2:
            mod->BSIM3pscbe2 = value->rValue;
            mod->BSIM3pscbe2Given = TRUE;
            break;                 
        case BSIM3_MOD_PVAG:
            mod->BSIM3pvag = value->rValue;
            mod->BSIM3pvagGiven = TRUE;
            break;                 
        case  BSIM3_MOD_WR :
            mod->BSIM3wr = value->rValue;
            mod->BSIM3wrGiven = TRUE;
            break;
        case  BSIM3_MOD_DWG :
            mod->BSIM3dwg = value->rValue;
            mod->BSIM3dwgGiven = TRUE;
            break;
        case  BSIM3_MOD_DWB :
            mod->BSIM3dwb = value->rValue;
            mod->BSIM3dwbGiven = TRUE;
            break;
        case  BSIM3_MOD_B0 :
            mod->BSIM3b0 = value->rValue;
            mod->BSIM3b0Given = TRUE;
            break;
        case  BSIM3_MOD_B1 :
            mod->BSIM3b1 = value->rValue;
            mod->BSIM3b1Given = TRUE;
            break;
        case  BSIM3_MOD_ALPHA0 :
            mod->BSIM3alpha0 = value->rValue;
            mod->BSIM3alpha0Given = TRUE;
            break;
        case  BSIM3_MOD_ALPHA1 :
            mod->BSIM3alpha1 = value->rValue;
            mod->BSIM3alpha1Given = TRUE;
            break;
        case  BSIM3_MOD_BETA0 :
            mod->BSIM3beta0 = value->rValue;
            mod->BSIM3beta0Given = TRUE;
            break;
        case  BSIM3_MOD_IJTH :
            mod->BSIM3ijth = value->rValue;
            mod->BSIM3ijthGiven = TRUE;
            break;
        case  BSIM3_MOD_VFB :
            mod->BSIM3vfb = value->rValue;
            mod->BSIM3vfbGiven = TRUE;
            break;

        case  BSIM3_MOD_ELM :
            mod->BSIM3elm = value->rValue;
            mod->BSIM3elmGiven = TRUE;
            break;
        case  BSIM3_MOD_CGSL :
            mod->BSIM3cgsl = value->rValue;
            mod->BSIM3cgslGiven = TRUE;
            break;
        case  BSIM3_MOD_CGDL :
            mod->BSIM3cgdl = value->rValue;
            mod->BSIM3cgdlGiven = TRUE;
            break;
        case  BSIM3_MOD_CKAPPA :
            mod->BSIM3ckappa = value->rValue;
            mod->BSIM3ckappaGiven = TRUE;
            break;
        case  BSIM3_MOD_CF :
            mod->BSIM3cf = value->rValue;
            mod->BSIM3cfGiven = TRUE;
            break;
        case  BSIM3_MOD_CLC :
            mod->BSIM3clc = value->rValue;
            mod->BSIM3clcGiven = TRUE;
            break;
        case  BSIM3_MOD_CLE :
            mod->BSIM3cle = value->rValue;
            mod->BSIM3cleGiven = TRUE;
            break;
        case  BSIM3_MOD_DWC :
            mod->BSIM3dwc = value->rValue;
            mod->BSIM3dwcGiven = TRUE;
            break;
        case  BSIM3_MOD_DLC :
            mod->BSIM3dlc = value->rValue;
            mod->BSIM3dlcGiven = TRUE;
            break;
        case  BSIM3_MOD_VFBCV :
            mod->BSIM3vfbcv = value->rValue;
            mod->BSIM3vfbcvGiven = TRUE;
            break;
        case  BSIM3_MOD_ACDE :
            mod->BSIM3acde = value->rValue;
            mod->BSIM3acdeGiven = TRUE;
            break;
        case  BSIM3_MOD_MOIN :
            mod->BSIM3moin = value->rValue;
            mod->BSIM3moinGiven = TRUE;
            break;
        case  BSIM3_MOD_NOFF :
            mod->BSIM3noff = value->rValue;
            mod->BSIM3noffGiven = TRUE;
            break;
        case  BSIM3_MOD_VOFFCV :
            mod->BSIM3voffcv = value->rValue;
            mod->BSIM3voffcvGiven = TRUE;
            break;
        case  BSIM3_MOD_TCJ :
            mod->BSIM3tcj = value->rValue;
            mod->BSIM3tcjGiven = TRUE;
            break;
        case  BSIM3_MOD_TPB :
            mod->BSIM3tpb = value->rValue;
            mod->BSIM3tpbGiven = TRUE;
            break;
        case  BSIM3_MOD_TCJSW :
            mod->BSIM3tcjsw = value->rValue;
            mod->BSIM3tcjswGiven = TRUE;
            break;
        case  BSIM3_MOD_TPBSW :
            mod->BSIM3tpbsw = value->rValue;
            mod->BSIM3tpbswGiven = TRUE;
            break;
        case  BSIM3_MOD_TCJSWG :
            mod->BSIM3tcjswg = value->rValue;
            mod->BSIM3tcjswgGiven = TRUE;
            break;
        case  BSIM3_MOD_TPBSWG :
            mod->BSIM3tpbswg = value->rValue;
            mod->BSIM3tpbswgGiven = TRUE;
            break;

          /* acm model */
        case BSIM3_MOD_HDIF:
            mod->BSIM3hdif = value->rValue;
            mod->BSIM3hdifGiven = TRUE;
            break;
        case BSIM3_MOD_LDIF:
            mod->BSIM3ldif = value->rValue;
            mod->BSIM3ldifGiven = TRUE;
            break;
        case BSIM3_MOD_LD:
            mod->BSIM3ld = value->rValue;
            mod->BSIM3ldGiven = TRUE;
            break;
        case BSIM3_MOD_RD:
            mod->BSIM3rd = value->rValue;
            mod->BSIM3rdGiven = TRUE;
            break;
        case BSIM3_MOD_RS:
            mod->BSIM3rs = value->rValue;
            mod->BSIM3rsGiven = TRUE;
            break;
        case BSIM3_MOD_RDC:
            mod->BSIM3rdc = value->rValue;
            mod->BSIM3rdcGiven = TRUE;
            break;
        case BSIM3_MOD_RSC:
            mod->BSIM3rsc = value->rValue;
            mod->BSIM3rscGiven = TRUE;
            break;
        case BSIM3_MOD_WMLT:
            mod->BSIM3wmlt = value->rValue;
            mod->BSIM3wmltGiven = TRUE;
            break;

        /* Length dependence */
        case  BSIM3_MOD_LCDSC :
            mod->BSIM3lcdsc = value->rValue;
            mod->BSIM3lcdscGiven = TRUE;
            break;


        case  BSIM3_MOD_LCDSCB :
            mod->BSIM3lcdscb = value->rValue;
            mod->BSIM3lcdscbGiven = TRUE;
            break;
        case  BSIM3_MOD_LCDSCD :
            mod->BSIM3lcdscd = value->rValue;
            mod->BSIM3lcdscdGiven = TRUE;
            break;
        case  BSIM3_MOD_LCIT :
            mod->BSIM3lcit = value->rValue;
            mod->BSIM3lcitGiven = TRUE;
            break;
        case  BSIM3_MOD_LNFACTOR :
            mod->BSIM3lnfactor = value->rValue;
            mod->BSIM3lnfactorGiven = TRUE;
            break;
        case BSIM3_MOD_LXJ:
            mod->BSIM3lxj = value->rValue;
            mod->BSIM3lxjGiven = TRUE;
            break;
        case BSIM3_MOD_LVSAT:
            mod->BSIM3lvsat = value->rValue;
            mod->BSIM3lvsatGiven = TRUE;
            break;
        
        
        case BSIM3_MOD_LA0:
            mod->BSIM3la0 = value->rValue;
            mod->BSIM3la0Given = TRUE;
            break;
        case BSIM3_MOD_LAGS:
            mod->BSIM3lags = value->rValue;
            mod->BSIM3lagsGiven = TRUE;
            break;
        case BSIM3_MOD_LA1:
            mod->BSIM3la1 = value->rValue;
            mod->BSIM3la1Given = TRUE;
            break;
        case BSIM3_MOD_LA2:
            mod->BSIM3la2 = value->rValue;
            mod->BSIM3la2Given = TRUE;
            break;
        case BSIM3_MOD_LAT:
            mod->BSIM3lat = value->rValue;
            mod->BSIM3latGiven = TRUE;
            break;
        case BSIM3_MOD_LKETA:
            mod->BSIM3lketa = value->rValue;
            mod->BSIM3lketaGiven = TRUE;
            break;    
        case BSIM3_MOD_LNSUB:
            mod->BSIM3lnsub = value->rValue;
            mod->BSIM3lnsubGiven = TRUE;
            break;
        case BSIM3_MOD_LNPEAK:
            mod->BSIM3lnpeak = value->rValue;
            mod->BSIM3lnpeakGiven = TRUE;
            if (mod->BSIM3lnpeak > 1.0e20)
                mod->BSIM3lnpeak *= 1.0e-6;
            break;
        case BSIM3_MOD_LNGATE:
            mod->BSIM3lngate = value->rValue;
            mod->BSIM3lngateGiven = TRUE;
            if (mod->BSIM3lngate > 1.0e23)
                mod->BSIM3lngate *= 1.0e-6;
            break;
        case BSIM3_MOD_LGAMMA1:
            mod->BSIM3lgamma1 = value->rValue;
            mod->BSIM3lgamma1Given = TRUE;
            break;
        case BSIM3_MOD_LGAMMA2:
            mod->BSIM3lgamma2 = value->rValue;
            mod->BSIM3lgamma2Given = TRUE;
            break;
        case BSIM3_MOD_LVBX:
            mod->BSIM3lvbx = value->rValue;
            mod->BSIM3lvbxGiven = TRUE;
            break;
        case BSIM3_MOD_LVBM:
            mod->BSIM3lvbm = value->rValue;
            mod->BSIM3lvbmGiven = TRUE;
            break;
        case BSIM3_MOD_LXT:
            mod->BSIM3lxt = value->rValue;
            mod->BSIM3lxtGiven = TRUE;
            break;
        case  BSIM3_MOD_LK1:
            mod->BSIM3lk1 = value->rValue;
            mod->BSIM3lk1Given = TRUE;
            break;
        case  BSIM3_MOD_LKT1:
            mod->BSIM3lkt1 = value->rValue;
            mod->BSIM3lkt1Given = TRUE;
            break;
        case  BSIM3_MOD_LKT1L:
            mod->BSIM3lkt1l = value->rValue;
            mod->BSIM3lkt1lGiven = TRUE;
            break;
        case  BSIM3_MOD_LKT2:
            mod->BSIM3lkt2 = value->rValue;
            mod->BSIM3lkt2Given = TRUE;
            break;
        case  BSIM3_MOD_LK2:
            mod->BSIM3lk2 = value->rValue;
            mod->BSIM3lk2Given = TRUE;
            break;
        case  BSIM3_MOD_LK3:
            mod->BSIM3lk3 = value->rValue;
            mod->BSIM3lk3Given = TRUE;
            break;
        case  BSIM3_MOD_LK3B:
            mod->BSIM3lk3b = value->rValue;
            mod->BSIM3lk3bGiven = TRUE;
            break;
        case  BSIM3_MOD_LNLX:
            mod->BSIM3lnlx = value->rValue;
            mod->BSIM3lnlxGiven = TRUE;
            break;
        case  BSIM3_MOD_LW0:
            mod->BSIM3lw0 = value->rValue;
            mod->BSIM3lw0Given = TRUE;
            break;
        case  BSIM3_MOD_LDVT0:               
            mod->BSIM3ldvt0 = value->rValue;
            mod->BSIM3ldvt0Given = TRUE;
            break;
        case  BSIM3_MOD_LDVT1:             
            mod->BSIM3ldvt1 = value->rValue;
            mod->BSIM3ldvt1Given = TRUE;
            break;
        case  BSIM3_MOD_LDVT2:             
            mod->BSIM3ldvt2 = value->rValue;
            mod->BSIM3ldvt2Given = TRUE;
            break;
        case  BSIM3_MOD_LDVT0W:               
            mod->BSIM3ldvt0w = value->rValue;
            mod->BSIM3ldvt0wGiven = TRUE;
            break;
        case  BSIM3_MOD_LDVT1W:             
            mod->BSIM3ldvt1w = value->rValue;
            mod->BSIM3ldvt1wGiven = TRUE;
            break;
        case  BSIM3_MOD_LDVT2W:             
            mod->BSIM3ldvt2w = value->rValue;
            mod->BSIM3ldvt2wGiven = TRUE;
            break;
        case  BSIM3_MOD_LDROUT:             
            mod->BSIM3ldrout = value->rValue;
            mod->BSIM3ldroutGiven = TRUE;
            break;
        case  BSIM3_MOD_LDSUB:             
            mod->BSIM3ldsub = value->rValue;
            mod->BSIM3ldsubGiven = TRUE;
            break;
        case BSIM3_MOD_LVTH0:
            mod->BSIM3lvth0 = value->rValue;
            mod->BSIM3lvth0Given = TRUE;
            break;
        case BSIM3_MOD_LUA:
            mod->BSIM3lua = value->rValue;
            mod->BSIM3luaGiven = TRUE;
            break;
        case BSIM3_MOD_LUA1:
            mod->BSIM3lua1 = value->rValue;
            mod->BSIM3lua1Given = TRUE;
            break;
        case BSIM3_MOD_LUB:
            mod->BSIM3lub = value->rValue;
            mod->BSIM3lubGiven = TRUE;
            break;
        case BSIM3_MOD_LUB1:
            mod->BSIM3lub1 = value->rValue;
            mod->BSIM3lub1Given = TRUE;
            break;
        case BSIM3_MOD_LUC:
            mod->BSIM3luc = value->rValue;
            mod->BSIM3lucGiven = TRUE;
            break;
        case BSIM3_MOD_LUC1:
            mod->BSIM3luc1 = value->rValue;
            mod->BSIM3luc1Given = TRUE;
            break;
        case  BSIM3_MOD_LU0 :
            mod->BSIM3lu0 = value->rValue;
            mod->BSIM3lu0Given = TRUE;
            break;
        case  BSIM3_MOD_LUTE :
            mod->BSIM3lute = value->rValue;
            mod->BSIM3luteGiven = TRUE;
            break;
        case BSIM3_MOD_LVOFF:
            mod->BSIM3lvoff = value->rValue;
            mod->BSIM3lvoffGiven = TRUE;
            break;
        case  BSIM3_MOD_LDELTA :
            mod->BSIM3ldelta = value->rValue;
            mod->BSIM3ldeltaGiven = TRUE;
            break;
        case BSIM3_MOD_LRDSW:
            mod->BSIM3lrdsw = value->rValue;
            mod->BSIM3lrdswGiven = TRUE;
            break;                     
        case BSIM3_MOD_LPRWB:
            mod->BSIM3lprwb = value->rValue;
            mod->BSIM3lprwbGiven = TRUE;
            break;                     
        case BSIM3_MOD_LPRWG:
            mod->BSIM3lprwg = value->rValue;
            mod->BSIM3lprwgGiven = TRUE;
            break;                     
        case BSIM3_MOD_LPRT:
            mod->BSIM3lprt = value->rValue;
            mod->BSIM3lprtGiven = TRUE;
            break;                     
        case BSIM3_MOD_LETA0:
            mod->BSIM3leta0 = value->rValue;
            mod->BSIM3leta0Given = TRUE;
            break;                 
        case BSIM3_MOD_LETAB:
            mod->BSIM3letab = value->rValue;
            mod->BSIM3letabGiven = TRUE;
            break;                 
        case BSIM3_MOD_LPCLM:
            mod->BSIM3lpclm = value->rValue;
            mod->BSIM3lpclmGiven = TRUE;
            break;                 
        case BSIM3_MOD_LPDIBL1:
            mod->BSIM3lpdibl1 = value->rValue;
            mod->BSIM3lpdibl1Given = TRUE;
            break;                 
        case BSIM3_MOD_LPDIBL2:
            mod->BSIM3lpdibl2 = value->rValue;
            mod->BSIM3lpdibl2Given = TRUE;
            break;                 
        case BSIM3_MOD_LPDIBLB:
            mod->BSIM3lpdiblb = value->rValue;
            mod->BSIM3lpdiblbGiven = TRUE;
            break;                 
        case BSIM3_MOD_LPSCBE1:
            mod->BSIM3lpscbe1 = value->rValue;
            mod->BSIM3lpscbe1Given = TRUE;
            break;                 
        case BSIM3_MOD_LPSCBE2:
            mod->BSIM3lpscbe2 = value->rValue;
            mod->BSIM3lpscbe2Given = TRUE;
            break;                 
        case BSIM3_MOD_LPVAG:
            mod->BSIM3lpvag = value->rValue;
            mod->BSIM3lpvagGiven = TRUE;
            break;                 
        case  BSIM3_MOD_LWR :
            mod->BSIM3lwr = value->rValue;
            mod->BSIM3lwrGiven = TRUE;
            break;
        case  BSIM3_MOD_LDWG :
            mod->BSIM3ldwg = value->rValue;
            mod->BSIM3ldwgGiven = TRUE;
            break;
        case  BSIM3_MOD_LDWB :
            mod->BSIM3ldwb = value->rValue;
            mod->BSIM3ldwbGiven = TRUE;
            break;
        case  BSIM3_MOD_LB0 :
            mod->BSIM3lb0 = value->rValue;
            mod->BSIM3lb0Given = TRUE;
            break;
        case  BSIM3_MOD_LB1 :
            mod->BSIM3lb1 = value->rValue;
            mod->BSIM3lb1Given = TRUE;
            break;
        case  BSIM3_MOD_LALPHA0 :
            mod->BSIM3lalpha0 = value->rValue;
            mod->BSIM3lalpha0Given = TRUE;
            break;
        case  BSIM3_MOD_LALPHA1 :
            mod->BSIM3lalpha1 = value->rValue;
            mod->BSIM3lalpha1Given = TRUE;
            break;
        case  BSIM3_MOD_LBETA0 :
            mod->BSIM3lbeta0 = value->rValue;
            mod->BSIM3lbeta0Given = TRUE;
            break;
        case  BSIM3_MOD_LVFB :
            mod->BSIM3lvfb = value->rValue;
            mod->BSIM3lvfbGiven = TRUE;
            break;

        case  BSIM3_MOD_LELM :
            mod->BSIM3lelm = value->rValue;
            mod->BSIM3lelmGiven = TRUE;
            break;
        case  BSIM3_MOD_LCGSL :
            mod->BSIM3lcgsl = value->rValue;
            mod->BSIM3lcgslGiven = TRUE;
            break;
        case  BSIM3_MOD_LCGDL :
            mod->BSIM3lcgdl = value->rValue;
            mod->BSIM3lcgdlGiven = TRUE;
            break;
        case  BSIM3_MOD_LCKAPPA :
            mod->BSIM3lckappa = value->rValue;
            mod->BSIM3lckappaGiven = TRUE;
            break;
        case  BSIM3_MOD_LCF :
            mod->BSIM3lcf = value->rValue;
            mod->BSIM3lcfGiven = TRUE;
            break;
        case  BSIM3_MOD_LCLC :
            mod->BSIM3lclc = value->rValue;
            mod->BSIM3lclcGiven = TRUE;
            break;
        case  BSIM3_MOD_LCLE :
            mod->BSIM3lcle = value->rValue;
            mod->BSIM3lcleGiven = TRUE;
            break;
        case  BSIM3_MOD_LVFBCV :
            mod->BSIM3lvfbcv = value->rValue;
            mod->BSIM3lvfbcvGiven = TRUE;
            break;
        case  BSIM3_MOD_LACDE :
            mod->BSIM3lacde = value->rValue;
            mod->BSIM3lacdeGiven = TRUE;
            break;
        case  BSIM3_MOD_LMOIN :
            mod->BSIM3lmoin = value->rValue;
            mod->BSIM3lmoinGiven = TRUE;
            break;
        case  BSIM3_MOD_LNOFF :
            mod->BSIM3lnoff = value->rValue;
            mod->BSIM3lnoffGiven = TRUE;
            break;
        case  BSIM3_MOD_LVOFFCV :
            mod->BSIM3lvoffcv = value->rValue;
            mod->BSIM3lvoffcvGiven = TRUE;
            break;

        /* Width dependence */
        case  BSIM3_MOD_WCDSC :
            mod->BSIM3wcdsc = value->rValue;
            mod->BSIM3wcdscGiven = TRUE;
            break;
       
       
         case  BSIM3_MOD_WCDSCB :
            mod->BSIM3wcdscb = value->rValue;
            mod->BSIM3wcdscbGiven = TRUE;
            break;
         case  BSIM3_MOD_WCDSCD :
            mod->BSIM3wcdscd = value->rValue;
            mod->BSIM3wcdscdGiven = TRUE;
            break;
        case  BSIM3_MOD_WCIT :
            mod->BSIM3wcit = value->rValue;
            mod->BSIM3wcitGiven = TRUE;
            break;
        case  BSIM3_MOD_WNFACTOR :
            mod->BSIM3wnfactor = value->rValue;
            mod->BSIM3wnfactorGiven = TRUE;
            break;
        case BSIM3_MOD_WXJ:
            mod->BSIM3wxj = value->rValue;
            mod->BSIM3wxjGiven = TRUE;
            break;
        case BSIM3_MOD_WVSAT:
            mod->BSIM3wvsat = value->rValue;
            mod->BSIM3wvsatGiven = TRUE;
            break;


        case BSIM3_MOD_WA0:
            mod->BSIM3wa0 = value->rValue;
            mod->BSIM3wa0Given = TRUE;
            break;
        case BSIM3_MOD_WAGS:
            mod->BSIM3wags = value->rValue;
            mod->BSIM3wagsGiven = TRUE;
            break;
        case BSIM3_MOD_WA1:
            mod->BSIM3wa1 = value->rValue;
            mod->BSIM3wa1Given = TRUE;
            break;
        case BSIM3_MOD_WA2:
            mod->BSIM3wa2 = value->rValue;
            mod->BSIM3wa2Given = TRUE;
            break;
        case BSIM3_MOD_WAT:
            mod->BSIM3wat = value->rValue;
            mod->BSIM3watGiven = TRUE;
            break;
        case BSIM3_MOD_WKETA:
            mod->BSIM3wketa = value->rValue;
            mod->BSIM3wketaGiven = TRUE;
            break;    
        case BSIM3_MOD_WNSUB:
            mod->BSIM3wnsub = value->rValue;
            mod->BSIM3wnsubGiven = TRUE;
            break;
        case BSIM3_MOD_WNPEAK:
            mod->BSIM3wnpeak = value->rValue;
            mod->BSIM3wnpeakGiven = TRUE;
            if (mod->BSIM3wnpeak > 1.0e20)
                mod->BSIM3wnpeak *= 1.0e-6;
            break;
        case BSIM3_MOD_WNGATE:
            mod->BSIM3wngate = value->rValue;
            mod->BSIM3wngateGiven = TRUE;
            if (mod->BSIM3wngate > 1.0e23)
                mod->BSIM3wngate *= 1.0e-6;
            break;
        case BSIM3_MOD_WGAMMA1:
            mod->BSIM3wgamma1 = value->rValue;
            mod->BSIM3wgamma1Given = TRUE;
            break;
        case BSIM3_MOD_WGAMMA2:
            mod->BSIM3wgamma2 = value->rValue;
            mod->BSIM3wgamma2Given = TRUE;
            break;
        case BSIM3_MOD_WVBX:
            mod->BSIM3wvbx = value->rValue;
            mod->BSIM3wvbxGiven = TRUE;
            break;
        case BSIM3_MOD_WVBM:
            mod->BSIM3wvbm = value->rValue;
            mod->BSIM3wvbmGiven = TRUE;
            break;
        case BSIM3_MOD_WXT:
            mod->BSIM3wxt = value->rValue;
            mod->BSIM3wxtGiven = TRUE;
            break;
        case  BSIM3_MOD_WK1:
            mod->BSIM3wk1 = value->rValue;
            mod->BSIM3wk1Given = TRUE;
            break;
        case  BSIM3_MOD_WKT1:
            mod->BSIM3wkt1 = value->rValue;
            mod->BSIM3wkt1Given = TRUE;
            break;
        case  BSIM3_MOD_WKT1L:
            mod->BSIM3wkt1l = value->rValue;
            mod->BSIM3wkt1lGiven = TRUE;
            break;
        case  BSIM3_MOD_WKT2:
            mod->BSIM3wkt2 = value->rValue;
            mod->BSIM3wkt2Given = TRUE;
            break;
        case  BSIM3_MOD_WK2:
            mod->BSIM3wk2 = value->rValue;
            mod->BSIM3wk2Given = TRUE;
            break;
        case  BSIM3_MOD_WK3:
            mod->BSIM3wk3 = value->rValue;
            mod->BSIM3wk3Given = TRUE;
            break;
        case  BSIM3_MOD_WK3B:
            mod->BSIM3wk3b = value->rValue;
            mod->BSIM3wk3bGiven = TRUE;
            break;
        case  BSIM3_MOD_WNLX:
            mod->BSIM3wnlx = value->rValue;
            mod->BSIM3wnlxGiven = TRUE;
            break;
        case  BSIM3_MOD_WW0:
            mod->BSIM3ww0 = value->rValue;
            mod->BSIM3ww0Given = TRUE;
            break;
        case  BSIM3_MOD_WDVT0:               
            mod->BSIM3wdvt0 = value->rValue;
            mod->BSIM3wdvt0Given = TRUE;
            break;
        case  BSIM3_MOD_WDVT1:             
            mod->BSIM3wdvt1 = value->rValue;
            mod->BSIM3wdvt1Given = TRUE;
            break;
        case  BSIM3_MOD_WDVT2:             
            mod->BSIM3wdvt2 = value->rValue;
            mod->BSIM3wdvt2Given = TRUE;
            break;
        case  BSIM3_MOD_WDVT0W:               
            mod->BSIM3wdvt0w = value->rValue;
            mod->BSIM3wdvt0wGiven = TRUE;
            break;
        case  BSIM3_MOD_WDVT1W:             
            mod->BSIM3wdvt1w = value->rValue;
            mod->BSIM3wdvt1wGiven = TRUE;
            break;
        case  BSIM3_MOD_WDVT2W:             
            mod->BSIM3wdvt2w = value->rValue;
            mod->BSIM3wdvt2wGiven = TRUE;
            break;
        case  BSIM3_MOD_WDROUT:             
            mod->BSIM3wdrout = value->rValue;
            mod->BSIM3wdroutGiven = TRUE;
            break;
        case  BSIM3_MOD_WDSUB:             
            mod->BSIM3wdsub = value->rValue;
            mod->BSIM3wdsubGiven = TRUE;
            break;
        case BSIM3_MOD_WVTH0:
            mod->BSIM3wvth0 = value->rValue;
            mod->BSIM3wvth0Given = TRUE;
            break;
        case BSIM3_MOD_WUA:
            mod->BSIM3wua = value->rValue;
            mod->BSIM3wuaGiven = TRUE;
            break;
        case BSIM3_MOD_WUA1:
            mod->BSIM3wua1 = value->rValue;
            mod->BSIM3wua1Given = TRUE;
            break;
        case BSIM3_MOD_WUB:
            mod->BSIM3wub = value->rValue;
            mod->BSIM3wubGiven = TRUE;
            break;
        case BSIM3_MOD_WUB1:
            mod->BSIM3wub1 = value->rValue;
            mod->BSIM3wub1Given = TRUE;
            break;
        case BSIM3_MOD_WUC:
            mod->BSIM3wuc = value->rValue;
            mod->BSIM3wucGiven = TRUE;
            break;
        case BSIM3_MOD_WUC1:
            mod->BSIM3wuc1 = value->rValue;
            mod->BSIM3wuc1Given = TRUE;
            break;
        case  BSIM3_MOD_WU0 :
            mod->BSIM3wu0 = value->rValue;
            mod->BSIM3wu0Given = TRUE;
            break;
        case  BSIM3_MOD_WUTE :
            mod->BSIM3wute = value->rValue;
            mod->BSIM3wuteGiven = TRUE;
            break;
        case BSIM3_MOD_WVOFF:
            mod->BSIM3wvoff = value->rValue;
            mod->BSIM3wvoffGiven = TRUE;
            break;
        case  BSIM3_MOD_WDELTA :
            mod->BSIM3wdelta = value->rValue;
            mod->BSIM3wdeltaGiven = TRUE;
            break;
        case BSIM3_MOD_WRDSW:
            mod->BSIM3wrdsw = value->rValue;
            mod->BSIM3wrdswGiven = TRUE;
            break;                     
        case BSIM3_MOD_WPRWB:
            mod->BSIM3wprwb = value->rValue;
            mod->BSIM3wprwbGiven = TRUE;
            break;                     
        case BSIM3_MOD_WPRWG:
            mod->BSIM3wprwg = value->rValue;
            mod->BSIM3wprwgGiven = TRUE;
            break;                     
        case BSIM3_MOD_WPRT:
            mod->BSIM3wprt = value->rValue;
            mod->BSIM3wprtGiven = TRUE;
            break;                     
        case BSIM3_MOD_WETA0:
            mod->BSIM3weta0 = value->rValue;
            mod->BSIM3weta0Given = TRUE;
            break;                 
        case BSIM3_MOD_WETAB:
            mod->BSIM3wetab = value->rValue;
            mod->BSIM3wetabGiven = TRUE;
            break;                 
        case BSIM3_MOD_WPCLM:
            mod->BSIM3wpclm = value->rValue;
            mod->BSIM3wpclmGiven = TRUE;
            break;                 
        case BSIM3_MOD_WPDIBL1:
            mod->BSIM3wpdibl1 = value->rValue;
            mod->BSIM3wpdibl1Given = TRUE;
            break;                 
        case BSIM3_MOD_WPDIBL2:
            mod->BSIM3wpdibl2 = value->rValue;
            mod->BSIM3wpdibl2Given = TRUE;
            break;                 
        case BSIM3_MOD_WPDIBLB:
            mod->BSIM3wpdiblb = value->rValue;
            mod->BSIM3wpdiblbGiven = TRUE;
            break;                 
        case BSIM3_MOD_WPSCBE1:
            mod->BSIM3wpscbe1 = value->rValue;
            mod->BSIM3wpscbe1Given = TRUE;
            break;                 
        case BSIM3_MOD_WPSCBE2:
            mod->BSIM3wpscbe2 = value->rValue;
            mod->BSIM3wpscbe2Given = TRUE;
            break;                 
        case BSIM3_MOD_WPVAG:
            mod->BSIM3wpvag = value->rValue;
            mod->BSIM3wpvagGiven = TRUE;
            break;                 
        case  BSIM3_MOD_WWR :
            mod->BSIM3wwr = value->rValue;
            mod->BSIM3wwrGiven = TRUE;
            break;
        case  BSIM3_MOD_WDWG :
            mod->BSIM3wdwg = value->rValue;
            mod->BSIM3wdwgGiven = TRUE;
            break;
        case  BSIM3_MOD_WDWB :
            mod->BSIM3wdwb = value->rValue;
            mod->BSIM3wdwbGiven = TRUE;
            break;
        case  BSIM3_MOD_WB0 :
            mod->BSIM3wb0 = value->rValue;
            mod->BSIM3wb0Given = TRUE;
            break;
        case  BSIM3_MOD_WB1 :
            mod->BSIM3wb1 = value->rValue;
            mod->BSIM3wb1Given = TRUE;
            break;
        case  BSIM3_MOD_WALPHA0 :
            mod->BSIM3walpha0 = value->rValue;
            mod->BSIM3walpha0Given = TRUE;
            break;
        case  BSIM3_MOD_WALPHA1 :
            mod->BSIM3walpha1 = value->rValue;
            mod->BSIM3walpha1Given = TRUE;
            break;
        case  BSIM3_MOD_WBETA0 :
            mod->BSIM3wbeta0 = value->rValue;
            mod->BSIM3wbeta0Given = TRUE;
            break;
        case  BSIM3_MOD_WVFB :
            mod->BSIM3wvfb = value->rValue;
            mod->BSIM3wvfbGiven = TRUE;
            break;

        case  BSIM3_MOD_WELM :
            mod->BSIM3welm = value->rValue;
            mod->BSIM3welmGiven = TRUE;
            break;
        case  BSIM3_MOD_WCGSL :
            mod->BSIM3wcgsl = value->rValue;
            mod->BSIM3wcgslGiven = TRUE;
            break;
        case  BSIM3_MOD_WCGDL :
            mod->BSIM3wcgdl = value->rValue;
            mod->BSIM3wcgdlGiven = TRUE;
            break;
        case  BSIM3_MOD_WCKAPPA :
            mod->BSIM3wckappa = value->rValue;
            mod->BSIM3wckappaGiven = TRUE;
            break;
        case  BSIM3_MOD_WCF :
            mod->BSIM3wcf = value->rValue;
            mod->BSIM3wcfGiven = TRUE;
            break;
        case  BSIM3_MOD_WCLC :
            mod->BSIM3wclc = value->rValue;
            mod->BSIM3wclcGiven = TRUE;
            break;
        case  BSIM3_MOD_WCLE :
            mod->BSIM3wcle = value->rValue;
            mod->BSIM3wcleGiven = TRUE;
            break;
        case  BSIM3_MOD_WVFBCV :
            mod->BSIM3wvfbcv = value->rValue;
            mod->BSIM3wvfbcvGiven = TRUE;
            break;
        case  BSIM3_MOD_WACDE :
            mod->BSIM3wacde = value->rValue;
            mod->BSIM3wacdeGiven = TRUE;
            break;
        case  BSIM3_MOD_WMOIN :
            mod->BSIM3wmoin = value->rValue;
            mod->BSIM3wmoinGiven = TRUE;
            break;
        case  BSIM3_MOD_WNOFF :
            mod->BSIM3wnoff = value->rValue;
            mod->BSIM3wnoffGiven = TRUE;
            break;
        case  BSIM3_MOD_WVOFFCV :
            mod->BSIM3wvoffcv = value->rValue;
            mod->BSIM3wvoffcvGiven = TRUE;
            break;

        /* Cross-term dependence */
        case  BSIM3_MOD_PCDSC :
            mod->BSIM3pcdsc = value->rValue;
            mod->BSIM3pcdscGiven = TRUE;
            break;


        case  BSIM3_MOD_PCDSCB :
            mod->BSIM3pcdscb = value->rValue;
            mod->BSIM3pcdscbGiven = TRUE;
            break;
        case  BSIM3_MOD_PCDSCD :
            mod->BSIM3pcdscd = value->rValue;
            mod->BSIM3pcdscdGiven = TRUE;
            break;
        case  BSIM3_MOD_PCIT :
            mod->BSIM3pcit = value->rValue;
            mod->BSIM3pcitGiven = TRUE;
            break;
        case  BSIM3_MOD_PNFACTOR :
            mod->BSIM3pnfactor = value->rValue;
            mod->BSIM3pnfactorGiven = TRUE;
            break;
        case BSIM3_MOD_PXJ:
            mod->BSIM3pxj = value->rValue;
            mod->BSIM3pxjGiven = TRUE;
            break;
        case BSIM3_MOD_PVSAT:
            mod->BSIM3pvsat = value->rValue;
            mod->BSIM3pvsatGiven = TRUE;
            break;


        case BSIM3_MOD_PA0:
            mod->BSIM3pa0 = value->rValue;
            mod->BSIM3pa0Given = TRUE;
            break;
        case BSIM3_MOD_PAGS:
            mod->BSIM3pags = value->rValue;
            mod->BSIM3pagsGiven = TRUE;
            break;
        case BSIM3_MOD_PA1:
            mod->BSIM3pa1 = value->rValue;
            mod->BSIM3pa1Given = TRUE;
            break;
        case BSIM3_MOD_PA2:
            mod->BSIM3pa2 = value->rValue;
            mod->BSIM3pa2Given = TRUE;
            break;
        case BSIM3_MOD_PAT:
            mod->BSIM3pat = value->rValue;
            mod->BSIM3patGiven = TRUE;
            break;
        case BSIM3_MOD_PKETA:
            mod->BSIM3pketa = value->rValue;
            mod->BSIM3pketaGiven = TRUE;
            break;    
        case BSIM3_MOD_PNSUB:
            mod->BSIM3pnsub = value->rValue;
            mod->BSIM3pnsubGiven = TRUE;
            break;
        case BSIM3_MOD_PNPEAK:
            mod->BSIM3pnpeak = value->rValue;
            mod->BSIM3pnpeakGiven = TRUE;
            if (mod->BSIM3pnpeak > 1.0e20)
                mod->BSIM3pnpeak *= 1.0e-6;
            break;
        case BSIM3_MOD_PNGATE:
            mod->BSIM3pngate = value->rValue;
            mod->BSIM3pngateGiven = TRUE;
            if (mod->BSIM3pngate > 1.0e23)
                mod->BSIM3pngate *= 1.0e-6;
            break;
        case BSIM3_MOD_PGAMMA1:
            mod->BSIM3pgamma1 = value->rValue;
            mod->BSIM3pgamma1Given = TRUE;
            break;
        case BSIM3_MOD_PGAMMA2:
            mod->BSIM3pgamma2 = value->rValue;
            mod->BSIM3pgamma2Given = TRUE;
            break;
        case BSIM3_MOD_PVBX:
            mod->BSIM3pvbx = value->rValue;
            mod->BSIM3pvbxGiven = TRUE;
            break;
        case BSIM3_MOD_PVBM:
            mod->BSIM3pvbm = value->rValue;
            mod->BSIM3pvbmGiven = TRUE;
            break;
        case BSIM3_MOD_PXT:
            mod->BSIM3pxt = value->rValue;
            mod->BSIM3pxtGiven = TRUE;
            break;
        case  BSIM3_MOD_PK1:
            mod->BSIM3pk1 = value->rValue;
            mod->BSIM3pk1Given = TRUE;
            break;
        case  BSIM3_MOD_PKT1:
            mod->BSIM3pkt1 = value->rValue;
            mod->BSIM3pkt1Given = TRUE;
            break;
        case  BSIM3_MOD_PKT1L:
            mod->BSIM3pkt1l = value->rValue;
            mod->BSIM3pkt1lGiven = TRUE;
            break;
        case  BSIM3_MOD_PKT2:
            mod->BSIM3pkt2 = value->rValue;
            mod->BSIM3pkt2Given = TRUE;
            break;
        case  BSIM3_MOD_PK2:
            mod->BSIM3pk2 = value->rValue;
            mod->BSIM3pk2Given = TRUE;
            break;
        case  BSIM3_MOD_PK3:
            mod->BSIM3pk3 = value->rValue;
            mod->BSIM3pk3Given = TRUE;
            break;
        case  BSIM3_MOD_PK3B:
            mod->BSIM3pk3b = value->rValue;
            mod->BSIM3pk3bGiven = TRUE;
            break;
        case  BSIM3_MOD_PNLX:
            mod->BSIM3pnlx = value->rValue;
            mod->BSIM3pnlxGiven = TRUE;
            break;
        case  BSIM3_MOD_PW0:
            mod->BSIM3pw0 = value->rValue;
            mod->BSIM3pw0Given = TRUE;
            break;
        case  BSIM3_MOD_PDVT0:               
            mod->BSIM3pdvt0 = value->rValue;
            mod->BSIM3pdvt0Given = TRUE;
            break;
        case  BSIM3_MOD_PDVT1:             
            mod->BSIM3pdvt1 = value->rValue;
            mod->BSIM3pdvt1Given = TRUE;
            break;
        case  BSIM3_MOD_PDVT2:             
            mod->BSIM3pdvt2 = value->rValue;
            mod->BSIM3pdvt2Given = TRUE;
            break;
        case  BSIM3_MOD_PDVT0W:               
            mod->BSIM3pdvt0w = value->rValue;
            mod->BSIM3pdvt0wGiven = TRUE;
            break;
        case  BSIM3_MOD_PDVT1W:             
            mod->BSIM3pdvt1w = value->rValue;
            mod->BSIM3pdvt1wGiven = TRUE;
            break;
        case  BSIM3_MOD_PDVT2W:             
            mod->BSIM3pdvt2w = value->rValue;
            mod->BSIM3pdvt2wGiven = TRUE;
            break;
        case  BSIM3_MOD_PDROUT:             
            mod->BSIM3pdrout = value->rValue;
            mod->BSIM3pdroutGiven = TRUE;
            break;
        case  BSIM3_MOD_PDSUB:             
            mod->BSIM3pdsub = value->rValue;
            mod->BSIM3pdsubGiven = TRUE;
            break;
        case BSIM3_MOD_PVTH0:
            mod->BSIM3pvth0 = value->rValue;
            mod->BSIM3pvth0Given = TRUE;
            break;
        case BSIM3_MOD_PUA:
            mod->BSIM3pua = value->rValue;
            mod->BSIM3puaGiven = TRUE;
            break;
        case BSIM3_MOD_PUA1:
            mod->BSIM3pua1 = value->rValue;
            mod->BSIM3pua1Given = TRUE;
            break;
        case BSIM3_MOD_PUB:
            mod->BSIM3pub = value->rValue;
            mod->BSIM3pubGiven = TRUE;
            break;
        case BSIM3_MOD_PUB1:
            mod->BSIM3pub1 = value->rValue;
            mod->BSIM3pub1Given = TRUE;
            break;
        case BSIM3_MOD_PUC:
            mod->BSIM3puc = value->rValue;
            mod->BSIM3pucGiven = TRUE;
            break;
        case BSIM3_MOD_PUC1:
            mod->BSIM3puc1 = value->rValue;
            mod->BSIM3puc1Given = TRUE;
            break;
        case  BSIM3_MOD_PU0 :
            mod->BSIM3pu0 = value->rValue;
            mod->BSIM3pu0Given = TRUE;
            break;
        case  BSIM3_MOD_PUTE :
            mod->BSIM3pute = value->rValue;
            mod->BSIM3puteGiven = TRUE;
            break;
        case BSIM3_MOD_PVOFF:
            mod->BSIM3pvoff = value->rValue;
            mod->BSIM3pvoffGiven = TRUE;
            break;
        case  BSIM3_MOD_PDELTA :
            mod->BSIM3pdelta = value->rValue;
            mod->BSIM3pdeltaGiven = TRUE;
            break;
        case BSIM3_MOD_PRDSW:
            mod->BSIM3prdsw = value->rValue;
            mod->BSIM3prdswGiven = TRUE;
            break;                     
        case BSIM3_MOD_PPRWB:
            mod->BSIM3pprwb = value->rValue;
            mod->BSIM3pprwbGiven = TRUE;
            break;                     
        case BSIM3_MOD_PPRWG:
            mod->BSIM3pprwg = value->rValue;
            mod->BSIM3pprwgGiven = TRUE;
            break;                     
        case BSIM3_MOD_PPRT:
            mod->BSIM3pprt = value->rValue;
            mod->BSIM3pprtGiven = TRUE;
            break;                     
        case BSIM3_MOD_PETA0:
            mod->BSIM3peta0 = value->rValue;
            mod->BSIM3peta0Given = TRUE;
            break;                 
        case BSIM3_MOD_PETAB:
            mod->BSIM3petab = value->rValue;
            mod->BSIM3petabGiven = TRUE;
            break;                 
        case BSIM3_MOD_PPCLM:
            mod->BSIM3ppclm = value->rValue;
            mod->BSIM3ppclmGiven = TRUE;
            break;                 
        case BSIM3_MOD_PPDIBL1:
            mod->BSIM3ppdibl1 = value->rValue;
            mod->BSIM3ppdibl1Given = TRUE;
            break;                 
        case BSIM3_MOD_PPDIBL2:
            mod->BSIM3ppdibl2 = value->rValue;
            mod->BSIM3ppdibl2Given = TRUE;
            break;                 
        case BSIM3_MOD_PPDIBLB:
            mod->BSIM3ppdiblb = value->rValue;
            mod->BSIM3ppdiblbGiven = TRUE;
            break;                 
        case BSIM3_MOD_PPSCBE1:
            mod->BSIM3ppscbe1 = value->rValue;
            mod->BSIM3ppscbe1Given = TRUE;
            break;                 
        case BSIM3_MOD_PPSCBE2:
            mod->BSIM3ppscbe2 = value->rValue;
            mod->BSIM3ppscbe2Given = TRUE;
            break;                 
        case BSIM3_MOD_PPVAG:
            mod->BSIM3ppvag = value->rValue;
            mod->BSIM3ppvagGiven = TRUE;
            break;                 
        case  BSIM3_MOD_PWR :
            mod->BSIM3pwr = value->rValue;
            mod->BSIM3pwrGiven = TRUE;
            break;
        case  BSIM3_MOD_PDWG :
            mod->BSIM3pdwg = value->rValue;
            mod->BSIM3pdwgGiven = TRUE;
            break;
        case  BSIM3_MOD_PDWB :
            mod->BSIM3pdwb = value->rValue;
            mod->BSIM3pdwbGiven = TRUE;
            break;
        case  BSIM3_MOD_PB0 :
            mod->BSIM3pb0 = value->rValue;
            mod->BSIM3pb0Given = TRUE;
            break;
        case  BSIM3_MOD_PB1 :
            mod->BSIM3pb1 = value->rValue;
            mod->BSIM3pb1Given = TRUE;
            break;
        case  BSIM3_MOD_PALPHA0 :
            mod->BSIM3palpha0 = value->rValue;
            mod->BSIM3palpha0Given = TRUE;
            break;
        case  BSIM3_MOD_PALPHA1 :
            mod->BSIM3palpha1 = value->rValue;
            mod->BSIM3palpha1Given = TRUE;
            break;
        case  BSIM3_MOD_PBETA0 :
            mod->BSIM3pbeta0 = value->rValue;
            mod->BSIM3pbeta0Given = TRUE;
            break;
        case  BSIM3_MOD_PVFB :
            mod->BSIM3pvfb = value->rValue;
            mod->BSIM3pvfbGiven = TRUE;
            break;

        case  BSIM3_MOD_PELM :
            mod->BSIM3pelm = value->rValue;
            mod->BSIM3pelmGiven = TRUE;
            break;
        case  BSIM3_MOD_PCGSL :
            mod->BSIM3pcgsl = value->rValue;
            mod->BSIM3pcgslGiven = TRUE;
            break;
        case  BSIM3_MOD_PCGDL :
            mod->BSIM3pcgdl = value->rValue;
            mod->BSIM3pcgdlGiven = TRUE;
            break;
        case  BSIM3_MOD_PCKAPPA :
            mod->BSIM3pckappa = value->rValue;
            mod->BSIM3pckappaGiven = TRUE;
            break;
        case  BSIM3_MOD_PCF :
            mod->BSIM3pcf = value->rValue;
            mod->BSIM3pcfGiven = TRUE;
            break;
        case  BSIM3_MOD_PCLC :
            mod->BSIM3pclc = value->rValue;
            mod->BSIM3pclcGiven = TRUE;
            break;
        case  BSIM3_MOD_PCLE :
            mod->BSIM3pcle = value->rValue;
            mod->BSIM3pcleGiven = TRUE;
            break;
        case  BSIM3_MOD_PVFBCV :
            mod->BSIM3pvfbcv = value->rValue;
            mod->BSIM3pvfbcvGiven = TRUE;
            break;
        case  BSIM3_MOD_PACDE :
            mod->BSIM3pacde = value->rValue;
            mod->BSIM3pacdeGiven = TRUE;
            break;
        case  BSIM3_MOD_PMOIN :
            mod->BSIM3pmoin = value->rValue;
            mod->BSIM3pmoinGiven = TRUE;
            break;
        case  BSIM3_MOD_PNOFF :
            mod->BSIM3pnoff = value->rValue;
            mod->BSIM3pnoffGiven = TRUE;
            break;
        case  BSIM3_MOD_PVOFFCV :
            mod->BSIM3pvoffcv = value->rValue;
            mod->BSIM3pvoffcvGiven = TRUE;
            break;

        case  BSIM3_MOD_TNOM :
            mod->BSIM3tnom = value->rValue + CONSTCtoK;
            mod->BSIM3tnomGiven = TRUE;
            break;
        case  BSIM3_MOD_CGSO :
            mod->BSIM3cgso = value->rValue;
            mod->BSIM3cgsoGiven = TRUE;
            break;
        case  BSIM3_MOD_CGDO :
            mod->BSIM3cgdo = value->rValue;
            mod->BSIM3cgdoGiven = TRUE;
            break;
        case  BSIM3_MOD_CGBO :
            mod->BSIM3cgbo = value->rValue;
            mod->BSIM3cgboGiven = TRUE;
            break;
        case  BSIM3_MOD_XPART :
            mod->BSIM3xpart = value->rValue;
            mod->BSIM3xpartGiven = TRUE;
            break;
        case  BSIM3_MOD_RSH :
            mod->BSIM3sheetResistance = value->rValue;
            mod->BSIM3sheetResistanceGiven = TRUE;
            break;
        case  BSIM3_MOD_JS :
            mod->BSIM3jctSatCurDensity = value->rValue;
            mod->BSIM3jctSatCurDensityGiven = TRUE;
            break;
        case  BSIM3_MOD_JSW :
            mod->BSIM3jctSidewallSatCurDensity = value->rValue;
            mod->BSIM3jctSidewallSatCurDensityGiven = TRUE;
            break;
        case  BSIM3_MOD_PB :
            mod->BSIM3bulkJctPotential = value->rValue;
            mod->BSIM3bulkJctPotentialGiven = TRUE;
            break;
        case  BSIM3_MOD_MJ :
            mod->BSIM3bulkJctBotGradingCoeff = value->rValue;
            mod->BSIM3bulkJctBotGradingCoeffGiven = TRUE;
            break;
        case  BSIM3_MOD_PBSW :
            mod->BSIM3sidewallJctPotential = value->rValue;
            mod->BSIM3sidewallJctPotentialGiven = TRUE;
            break;
        case  BSIM3_MOD_MJSW :
            mod->BSIM3bulkJctSideGradingCoeff = value->rValue;
            mod->BSIM3bulkJctSideGradingCoeffGiven = TRUE;
            break;
        case  BSIM3_MOD_CJ :
            mod->BSIM3unitAreaJctCap = value->rValue;
            mod->BSIM3unitAreaJctCapGiven = TRUE;
            break;
        case  BSIM3_MOD_CJSW :
            mod->BSIM3unitLengthSidewallJctCap = value->rValue;
            mod->BSIM3unitLengthSidewallJctCapGiven = TRUE;
            break;
        case  BSIM3_MOD_NJ :
            mod->BSIM3jctEmissionCoeff = value->rValue;
            mod->BSIM3jctEmissionCoeffGiven = TRUE;
            break;
        case  BSIM3_MOD_PBSWG :
            mod->BSIM3GatesidewallJctPotential = value->rValue;
            mod->BSIM3GatesidewallJctPotentialGiven = TRUE;
            break;
        case  BSIM3_MOD_MJSWG :
            mod->BSIM3bulkJctGateSideGradingCoeff = value->rValue;
            mod->BSIM3bulkJctGateSideGradingCoeffGiven = TRUE;
            break;
        case  BSIM3_MOD_CJSWG :
            mod->BSIM3unitLengthGateSidewallJctCap = value->rValue;
            mod->BSIM3unitLengthGateSidewallJctCapGiven = TRUE;
            break;
        case  BSIM3_MOD_XTI :
            mod->BSIM3jctTempExponent = value->rValue;
            mod->BSIM3jctTempExponentGiven = TRUE;
            break;
        case  BSIM3_MOD_LINTNOI:
            mod->BSIM3lintnoi = value->rValue;
            mod->BSIM3lintnoiGiven = TRUE;
            break;
        case  BSIM3_MOD_LINT :
            mod->BSIM3Lint = value->rValue;
            mod->BSIM3LintGiven = TRUE;
            break;
        case  BSIM3_MOD_LL :
            mod->BSIM3Ll = value->rValue;
            mod->BSIM3LlGiven = TRUE;
            break;
        case  BSIM3_MOD_LLC :
            mod->BSIM3Llc = value->rValue;
            mod->BSIM3LlcGiven = TRUE;
            break;
        case  BSIM3_MOD_LLN :
            mod->BSIM3Lln = value->rValue;
            mod->BSIM3LlnGiven = TRUE;
            break;
        case  BSIM3_MOD_LW :
            mod->BSIM3Lw = value->rValue;
            mod->BSIM3LwGiven = TRUE;
            break;
        case  BSIM3_MOD_LWC :
            mod->BSIM3Lwc = value->rValue;
            mod->BSIM3LwcGiven = TRUE;
            break;
        case  BSIM3_MOD_LWN :
            mod->BSIM3Lwn = value->rValue;
            mod->BSIM3LwnGiven = TRUE;
            break;
        case  BSIM3_MOD_LWL :
            mod->BSIM3Lwl = value->rValue;
            mod->BSIM3LwlGiven = TRUE;
            break;
        case  BSIM3_MOD_LWLC :
            mod->BSIM3Lwlc = value->rValue;
            mod->BSIM3LwlcGiven = TRUE;
            break;
        case  BSIM3_MOD_LMIN :
            mod->BSIM3Lmin = value->rValue;
            mod->BSIM3LminGiven = TRUE;
            break;
        case  BSIM3_MOD_LMAX :
            mod->BSIM3Lmax = value->rValue;
            mod->BSIM3LmaxGiven = TRUE;
            break;
        case  BSIM3_MOD_WINT :
            mod->BSIM3Wint = value->rValue;
            mod->BSIM3WintGiven = TRUE;
            break;
        case  BSIM3_MOD_WL :
            mod->BSIM3Wl = value->rValue;
            mod->BSIM3WlGiven = TRUE;
            break;
        case  BSIM3_MOD_WLC :
            mod->BSIM3Wlc = value->rValue;
            mod->BSIM3WlcGiven = TRUE;
            break;
        case  BSIM3_MOD_WLN :
            mod->BSIM3Wln = value->rValue;
            mod->BSIM3WlnGiven = TRUE;
            break;
        case  BSIM3_MOD_WW :
            mod->BSIM3Ww = value->rValue;
            mod->BSIM3WwGiven = TRUE;
            break;
        case  BSIM3_MOD_WWC :
            mod->BSIM3Wwc = value->rValue;
            mod->BSIM3WwcGiven = TRUE;
            break;
        case  BSIM3_MOD_WWN :
            mod->BSIM3Wwn = value->rValue;
            mod->BSIM3WwnGiven = TRUE;
            break;
        case  BSIM3_MOD_WWL :
            mod->BSIM3Wwl = value->rValue;
            mod->BSIM3WwlGiven = TRUE;
            break;
        case  BSIM3_MOD_WWLC :
            mod->BSIM3Wwlc = value->rValue;
            mod->BSIM3WwlcGiven = TRUE;
            break;
        case  BSIM3_MOD_WMIN :
            mod->BSIM3Wmin = value->rValue;
            mod->BSIM3WminGiven = TRUE;
            break;
        case  BSIM3_MOD_WMAX :
            mod->BSIM3Wmax = value->rValue;
            mod->BSIM3WmaxGiven = TRUE;
            break;

       case BSIM3_MOD_XL:
            mod->BSIM3xl = value->rValue;
            mod->BSIM3xlGiven = TRUE;
            break;
       case BSIM3_MOD_XW:
            mod->BSIM3xw = value->rValue;
            mod->BSIM3xwGiven = TRUE;
            break;

        case  BSIM3_MOD_NOIA :
            mod->BSIM3oxideTrapDensityA = value->rValue;
            mod->BSIM3oxideTrapDensityAGiven = TRUE;
            break;
        case  BSIM3_MOD_NOIB :
            mod->BSIM3oxideTrapDensityB = value->rValue;
            mod->BSIM3oxideTrapDensityBGiven = TRUE;
            break;
        case  BSIM3_MOD_NOIC :
            mod->BSIM3oxideTrapDensityC = value->rValue;
            mod->BSIM3oxideTrapDensityCGiven = TRUE;
            break;
        case  BSIM3_MOD_EM :
            mod->BSIM3em = value->rValue;
            mod->BSIM3emGiven = TRUE;
            break;
        case  BSIM3_MOD_EF :
            mod->BSIM3ef = value->rValue;
            mod->BSIM3efGiven = TRUE;
            break;
        case  BSIM3_MOD_AF :
            mod->BSIM3af = value->rValue;
            mod->BSIM3afGiven = TRUE;
            break;
        case  BSIM3_MOD_KF :
            mod->BSIM3kf = value->rValue;
            mod->BSIM3kfGiven = TRUE;
            break;

        case BSIM3_MOD_VGS_MAX:
            mod->BSIM3vgsMax = value->rValue;
            mod->BSIM3vgsMaxGiven = TRUE;
            break;
        case BSIM3_MOD_VGD_MAX:
            mod->BSIM3vgdMax = value->rValue;
            mod->BSIM3vgdMaxGiven = TRUE;
            break;
        case BSIM3_MOD_VGB_MAX:
            mod->BSIM3vgbMax = value->rValue;
            mod->BSIM3vgbMaxGiven = TRUE;
            break;
        case BSIM3_MOD_VDS_MAX:
            mod->BSIM3vdsMax = value->rValue;
            mod->BSIM3vdsMaxGiven = TRUE;
            break;
        case BSIM3_MOD_VBS_MAX:
            mod->BSIM3vbsMax = value->rValue;
            mod->BSIM3vbsMaxGiven = TRUE;
            break;
        case BSIM3_MOD_VBD_MAX:
            mod->BSIM3vbdMax = value->rValue;
            mod->BSIM3vbdMaxGiven = TRUE;
            break;
        case BSIM3_MOD_VGSR_MAX:
            mod->BSIM3vgsrMax = value->rValue;
            mod->BSIM3vgsrMaxGiven = TRUE;
            break;
        case BSIM3_MOD_VGDR_MAX:
            mod->BSIM3vgdrMax = value->rValue;
            mod->BSIM3vgdrMaxGiven = TRUE;
            break;
        case BSIM3_MOD_VGBR_MAX:
            mod->BSIM3vgbrMax = value->rValue;
            mod->BSIM3vgbrMaxGiven = TRUE;
            break;
        case BSIM3_MOD_VBSR_MAX:
            mod->BSIM3vbsrMax = value->rValue;
            mod->BSIM3vbsrMaxGiven = TRUE;
            break;
        case BSIM3_MOD_VBDR_MAX:
            mod->BSIM3vbdrMax = value->rValue;
            mod->BSIM3vbdrMaxGiven = TRUE;
            break;

        case  BSIM3_MOD_NMOS  :
            if(value->iValue) {
                mod->BSIM3type = 1;
                mod->BSIM3typeGiven = TRUE;
            }
            break;
        case  BSIM3_MOD_PMOS  :
            if(value->iValue) {
                mod->BSIM3type = - 1;
                mod->BSIM3typeGiven = TRUE;
            }
            break;
        default:
            return(E_BADPARM);
    }
    return(OK);
}


