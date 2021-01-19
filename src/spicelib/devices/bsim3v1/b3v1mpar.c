/**********
 * Copyright 1990 Regents of the University of California. All rights reserved.
 * File: b3mpar.c
 * Author: 1995 Min-Chie Jeng and Mansun Chan. 
 * Modified by Paolo Nenzi 2002
 **********/
 
/* 
 * Release Notes: 
 * BSIM3v1v3.1,   Released by yuhua  96/12/08
 */

#include "ngspice/ngspice.h"
#include "bsim3v1def.h"
#include "ngspice/ifsim.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
BSIM3v1mParam(int param, IFvalue *value, GENmodel *inMod)
{
    BSIM3v1model *mod = (BSIM3v1model*)inMod;
    switch(param)
    {   case  BSIM3v1_MOD_MOBMOD :
            mod->BSIM3v1mobMod = value->iValue;
            mod->BSIM3v1mobModGiven = TRUE;
            break;
        case  BSIM3v1_MOD_BINUNIT :
            mod->BSIM3v1binUnit = value->iValue;
            mod->BSIM3v1binUnitGiven = TRUE;
            break;
        case  BSIM3v1_MOD_PARAMCHK :
            mod->BSIM3v1paramChk = value->iValue;
            mod->BSIM3v1paramChkGiven = TRUE;
            break;
        case  BSIM3v1_MOD_CAPMOD :
            mod->BSIM3v1capMod = value->iValue;
            mod->BSIM3v1capModGiven = TRUE;
            break;
        case  BSIM3v1_MOD_NQSMOD :
            mod->BSIM3v1nqsMod = value->iValue;
            mod->BSIM3v1nqsModGiven = TRUE;
            break;
        case  BSIM3v1_MOD_NOIMOD :
            mod->BSIM3v1noiMod = value->iValue;
            mod->BSIM3v1noiModGiven = TRUE;
            break;
        case  BSIM3v1_MOD_VERSION :
            mod->BSIM3v1version = value->rValue;
            mod->BSIM3v1versionGiven = TRUE;
            break;
        case  BSIM3v1_MOD_TOX :
            mod->BSIM3v1tox = value->rValue;
            mod->BSIM3v1toxGiven = TRUE;
            break;

        case  BSIM3v1_MOD_CDSC :
            mod->BSIM3v1cdsc = value->rValue;
            mod->BSIM3v1cdscGiven = TRUE;
            break;
        case  BSIM3v1_MOD_CDSCB :
            mod->BSIM3v1cdscb = value->rValue;
            mod->BSIM3v1cdscbGiven = TRUE;
            break;

        case  BSIM3v1_MOD_CDSCD :
            mod->BSIM3v1cdscd = value->rValue;
            mod->BSIM3v1cdscdGiven = TRUE;
            break;

        case  BSIM3v1_MOD_CIT :
            mod->BSIM3v1cit = value->rValue;
            mod->BSIM3v1citGiven = TRUE;
            break;
        case  BSIM3v1_MOD_NFACTOR :
            mod->BSIM3v1nfactor = value->rValue;
            mod->BSIM3v1nfactorGiven = TRUE;
            break;
        case BSIM3v1_MOD_XJ:
            mod->BSIM3v1xj = value->rValue;
            mod->BSIM3v1xjGiven = TRUE;
            break;
        case BSIM3v1_MOD_VSAT:
            mod->BSIM3v1vsat = value->rValue;
            mod->BSIM3v1vsatGiven = TRUE;
            break;
        case BSIM3v1_MOD_A0:
            mod->BSIM3v1a0 = value->rValue;
            mod->BSIM3v1a0Given = TRUE;
            break;
        
        case BSIM3v1_MOD_AGS:
            mod->BSIM3v1ags= value->rValue;
            mod->BSIM3v1agsGiven = TRUE;
            break;
        
        case BSIM3v1_MOD_A1:
            mod->BSIM3v1a1 = value->rValue;
            mod->BSIM3v1a1Given = TRUE;
            break;
        case BSIM3v1_MOD_A2:
            mod->BSIM3v1a2 = value->rValue;
            mod->BSIM3v1a2Given = TRUE;
            break;
        case BSIM3v1_MOD_AT:
            mod->BSIM3v1at = value->rValue;
            mod->BSIM3v1atGiven = TRUE;
            break;
        case BSIM3v1_MOD_KETA:
            mod->BSIM3v1keta = value->rValue;
            mod->BSIM3v1ketaGiven = TRUE;
            break;    
        case BSIM3v1_MOD_NSUB:
            mod->BSIM3v1nsub = value->rValue;
            mod->BSIM3v1nsubGiven = TRUE;
            break;
        case BSIM3v1_MOD_NPEAK:
            mod->BSIM3v1npeak = value->rValue;
            mod->BSIM3v1npeakGiven = TRUE;
            if (mod->BSIM3v1npeak > 1.0e20)
              mod->BSIM3v1npeak *= 1.0e-6;
            break;
        case BSIM3v1_MOD_NGATE:
            mod->BSIM3v1ngate = value->rValue;
            mod->BSIM3v1ngateGiven = TRUE;
            if (mod->BSIM3v1ngate > 1.000001e24)
              mod->BSIM3v1ngate *= 1.0e-6;
            break;
        case BSIM3v1_MOD_GAMMA1:
            mod->BSIM3v1gamma1 = value->rValue;
            mod->BSIM3v1gamma1Given = TRUE;
            break;
        case BSIM3v1_MOD_GAMMA2:
            mod->BSIM3v1gamma2 = value->rValue;
            mod->BSIM3v1gamma2Given = TRUE;
            break;
        case BSIM3v1_MOD_VBX:
            mod->BSIM3v1vbx = value->rValue;
            mod->BSIM3v1vbxGiven = TRUE;
            break;
        case BSIM3v1_MOD_VBM:
            mod->BSIM3v1vbm = value->rValue;
            mod->BSIM3v1vbmGiven = TRUE;
            break;
        case BSIM3v1_MOD_XT:
            mod->BSIM3v1xt = value->rValue;
            mod->BSIM3v1xtGiven = TRUE;
            break;
        case  BSIM3v1_MOD_K1:
            mod->BSIM3v1k1 = value->rValue;
            mod->BSIM3v1k1Given = TRUE;
            break;
        case  BSIM3v1_MOD_KT1:
            mod->BSIM3v1kt1 = value->rValue;
            mod->BSIM3v1kt1Given = TRUE;
            break;
        case  BSIM3v1_MOD_KT1L:
            mod->BSIM3v1kt1l = value->rValue;
            mod->BSIM3v1kt1lGiven = TRUE;
            break;
        case  BSIM3v1_MOD_KT2:
            mod->BSIM3v1kt2 = value->rValue;
            mod->BSIM3v1kt2Given = TRUE;
            break;
        case  BSIM3v1_MOD_K2:
            mod->BSIM3v1k2 = value->rValue;
            mod->BSIM3v1k2Given = TRUE;
            break;
        case  BSIM3v1_MOD_K3:
            mod->BSIM3v1k3 = value->rValue;
            mod->BSIM3v1k3Given = TRUE;
            break;
        case  BSIM3v1_MOD_K3B:
            mod->BSIM3v1k3b = value->rValue;
            mod->BSIM3v1k3bGiven = TRUE;
            break;
        case  BSIM3v1_MOD_NLX:
            mod->BSIM3v1nlx = value->rValue;
            mod->BSIM3v1nlxGiven = TRUE;
            break;
        case  BSIM3v1_MOD_W0:
            mod->BSIM3v1w0 = value->rValue;
            mod->BSIM3v1w0Given = TRUE;
            break;
        case  BSIM3v1_MOD_DVT0:               
            mod->BSIM3v1dvt0 = value->rValue;
            mod->BSIM3v1dvt0Given = TRUE;
            break;
        case  BSIM3v1_MOD_DVT1:             
            mod->BSIM3v1dvt1 = value->rValue;
            mod->BSIM3v1dvt1Given = TRUE;
            break;
        case  BSIM3v1_MOD_DVT2:             
            mod->BSIM3v1dvt2 = value->rValue;
            mod->BSIM3v1dvt2Given = TRUE;
            break;
        case  BSIM3v1_MOD_DVT0W:               
            mod->BSIM3v1dvt0w = value->rValue;
            mod->BSIM3v1dvt0wGiven = TRUE;
            break;
        case  BSIM3v1_MOD_DVT1W:             
            mod->BSIM3v1dvt1w = value->rValue;
            mod->BSIM3v1dvt1wGiven = TRUE;
            break;
        case  BSIM3v1_MOD_DVT2W:             
            mod->BSIM3v1dvt2w = value->rValue;
            mod->BSIM3v1dvt2wGiven = TRUE;
            break;
        case  BSIM3v1_MOD_DROUT:             
            mod->BSIM3v1drout = value->rValue;
            mod->BSIM3v1droutGiven = TRUE;
            break;
        case  BSIM3v1_MOD_DSUB:             
            mod->BSIM3v1dsub = value->rValue;
            mod->BSIM3v1dsubGiven = TRUE;
            break;
        case BSIM3v1_MOD_VTH0:
            mod->BSIM3v1vth0 = value->rValue;
            mod->BSIM3v1vth0Given = TRUE;
            break;
        case BSIM3v1_MOD_UA:
            mod->BSIM3v1ua = value->rValue;
            mod->BSIM3v1uaGiven = TRUE;
            break;
        case BSIM3v1_MOD_UA1:
            mod->BSIM3v1ua1 = value->rValue;
            mod->BSIM3v1ua1Given = TRUE;
            break;
        case BSIM3v1_MOD_UB:
            mod->BSIM3v1ub = value->rValue;
            mod->BSIM3v1ubGiven = TRUE;
            break;
        case BSIM3v1_MOD_UB1:
            mod->BSIM3v1ub1 = value->rValue;
            mod->BSIM3v1ub1Given = TRUE;
            break;
        case BSIM3v1_MOD_UC:
            mod->BSIM3v1uc = value->rValue;
            mod->BSIM3v1ucGiven = TRUE;
            break;
        case BSIM3v1_MOD_UC1:
            mod->BSIM3v1uc1 = value->rValue;
            mod->BSIM3v1uc1Given = TRUE;
            break;
        case  BSIM3v1_MOD_U0 :
            mod->BSIM3v1u0 = value->rValue;
            mod->BSIM3v1u0Given = TRUE;
            break;
        case  BSIM3v1_MOD_UTE :
            mod->BSIM3v1ute = value->rValue;
            mod->BSIM3v1uteGiven = TRUE;
            break;
        case BSIM3v1_MOD_VOFF:
            mod->BSIM3v1voff = value->rValue;
            mod->BSIM3v1voffGiven = TRUE;
            break;
        case  BSIM3v1_MOD_DELTA :
            mod->BSIM3v1delta = value->rValue;
            mod->BSIM3v1deltaGiven = TRUE;
            break;
        case BSIM3v1_MOD_RDSW:
            mod->BSIM3v1rdsw = value->rValue;
            mod->BSIM3v1rdswGiven = TRUE;
            break;                     
        case BSIM3v1_MOD_PRWG:
            mod->BSIM3v1prwg = value->rValue;
            mod->BSIM3v1prwgGiven = TRUE;
            break;                     
        case BSIM3v1_MOD_PRWB:
            mod->BSIM3v1prwb = value->rValue;
            mod->BSIM3v1prwbGiven = TRUE;
            break;                     
        case BSIM3v1_MOD_PRT:
            mod->BSIM3v1prt = value->rValue;
            mod->BSIM3v1prtGiven = TRUE;
            break;                     
        case BSIM3v1_MOD_ETA0:
            mod->BSIM3v1eta0 = value->rValue;
            mod->BSIM3v1eta0Given = TRUE;
            break;                 
        case BSIM3v1_MOD_ETAB:
            mod->BSIM3v1etab = value->rValue;
            mod->BSIM3v1etabGiven = TRUE;
            break;                 
        case BSIM3v1_MOD_PCLM:
            mod->BSIM3v1pclm = value->rValue;
            mod->BSIM3v1pclmGiven = TRUE;
            break;                 
        case BSIM3v1_MOD_PDIBL1:
            mod->BSIM3v1pdibl1 = value->rValue;
            mod->BSIM3v1pdibl1Given = TRUE;
            break;                 
        case BSIM3v1_MOD_PDIBL2:
            mod->BSIM3v1pdibl2 = value->rValue;
            mod->BSIM3v1pdibl2Given = TRUE;
            break;                 
        case BSIM3v1_MOD_PDIBLB:
            mod->BSIM3v1pdiblb = value->rValue;
            mod->BSIM3v1pdiblbGiven = TRUE;
            break;                 
        case BSIM3v1_MOD_PSCBE1:
            mod->BSIM3v1pscbe1 = value->rValue;
            mod->BSIM3v1pscbe1Given = TRUE;
            break;                 
        case BSIM3v1_MOD_PSCBE2:
            mod->BSIM3v1pscbe2 = value->rValue;
            mod->BSIM3v1pscbe2Given = TRUE;
            break;                 
        case BSIM3v1_MOD_PVAG:
            mod->BSIM3v1pvag = value->rValue;
            mod->BSIM3v1pvagGiven = TRUE;
            break;                 
        case  BSIM3v1_MOD_WR :
            mod->BSIM3v1wr = value->rValue;
            mod->BSIM3v1wrGiven = TRUE;
            break;
        case  BSIM3v1_MOD_DWG :
            mod->BSIM3v1dwg = value->rValue;
            mod->BSIM3v1dwgGiven = TRUE;
            break;
        case  BSIM3v1_MOD_DWB :
            mod->BSIM3v1dwb = value->rValue;
            mod->BSIM3v1dwbGiven = TRUE;
            break;
        case  BSIM3v1_MOD_B0 :
            mod->BSIM3v1b0 = value->rValue;
            mod->BSIM3v1b0Given = TRUE;
            break;
        case  BSIM3v1_MOD_B1 :
            mod->BSIM3v1b1 = value->rValue;
            mod->BSIM3v1b1Given = TRUE;
            break;
        case  BSIM3v1_MOD_ALPHA0 :
            mod->BSIM3v1alpha0 = value->rValue;
            mod->BSIM3v1alpha0Given = TRUE;
            break;
        case  BSIM3v1_MOD_BETA0 :
            mod->BSIM3v1beta0 = value->rValue;
            mod->BSIM3v1beta0Given = TRUE;
            break;

        case  BSIM3v1_MOD_ELM :
            mod->BSIM3v1elm = value->rValue;
            mod->BSIM3v1elmGiven = TRUE;
            break;
        case  BSIM3v1_MOD_CGSL :
            mod->BSIM3v1cgsl = value->rValue;
            mod->BSIM3v1cgslGiven = TRUE;
            break;
        case  BSIM3v1_MOD_CGDL :
            mod->BSIM3v1cgdl = value->rValue;
            mod->BSIM3v1cgdlGiven = TRUE;
            break;
        case  BSIM3v1_MOD_CKAPPA :
            mod->BSIM3v1ckappa = value->rValue;
            mod->BSIM3v1ckappaGiven = TRUE;
            break;
        case  BSIM3v1_MOD_CF :
            mod->BSIM3v1cf = value->rValue;
            mod->BSIM3v1cfGiven = TRUE;
            break;
        case  BSIM3v1_MOD_CLC :
            mod->BSIM3v1clc = value->rValue;
            mod->BSIM3v1clcGiven = TRUE;
            break;
        case  BSIM3v1_MOD_CLE :
            mod->BSIM3v1cle = value->rValue;
            mod->BSIM3v1cleGiven = TRUE;
            break;
        case  BSIM3v1_MOD_DWC :
            mod->BSIM3v1dwc = value->rValue;
            mod->BSIM3v1dwcGiven = TRUE;
            break;
        case  BSIM3v1_MOD_DLC :
            mod->BSIM3v1dlc = value->rValue;
            mod->BSIM3v1dlcGiven = TRUE;
            break;
        case  BSIM3v1_MOD_VFBCV :
            mod->BSIM3v1vfbcv = value->rValue;
            mod->BSIM3v1vfbcvGiven = TRUE;
            break;

	/* Length dependence */
        case  BSIM3v1_MOD_LCDSC :
            mod->BSIM3v1lcdsc = value->rValue;
            mod->BSIM3v1lcdscGiven = TRUE;
            break;


        case  BSIM3v1_MOD_LCDSCB :
            mod->BSIM3v1lcdscb = value->rValue;
            mod->BSIM3v1lcdscbGiven = TRUE;
            break;
        case  BSIM3v1_MOD_LCDSCD :
            mod->BSIM3v1lcdscd = value->rValue;
            mod->BSIM3v1lcdscdGiven = TRUE;
            break;
        case  BSIM3v1_MOD_LCIT :
            mod->BSIM3v1lcit = value->rValue;
            mod->BSIM3v1lcitGiven = TRUE;
            break;
        case  BSIM3v1_MOD_LNFACTOR :
            mod->BSIM3v1lnfactor = value->rValue;
            mod->BSIM3v1lnfactorGiven = TRUE;
            break;
        case BSIM3v1_MOD_LXJ:
            mod->BSIM3v1lxj = value->rValue;
            mod->BSIM3v1lxjGiven = TRUE;
            break;
        case BSIM3v1_MOD_LVSAT:
            mod->BSIM3v1lvsat = value->rValue;
            mod->BSIM3v1lvsatGiven = TRUE;
            break;
        
        
        case BSIM3v1_MOD_LA0:
            mod->BSIM3v1la0 = value->rValue;
            mod->BSIM3v1la0Given = TRUE;
            break;
        case BSIM3v1_MOD_LAGS:
            mod->BSIM3v1lags = value->rValue;
            mod->BSIM3v1lagsGiven = TRUE;
            break;
        case BSIM3v1_MOD_LA1:
            mod->BSIM3v1la1 = value->rValue;
            mod->BSIM3v1la1Given = TRUE;
            break;
        case BSIM3v1_MOD_LA2:
            mod->BSIM3v1la2 = value->rValue;
            mod->BSIM3v1la2Given = TRUE;
            break;
        case BSIM3v1_MOD_LAT:
            mod->BSIM3v1lat = value->rValue;
            mod->BSIM3v1latGiven = TRUE;
            break;
        case BSIM3v1_MOD_LKETA:
            mod->BSIM3v1lketa = value->rValue;
            mod->BSIM3v1lketaGiven = TRUE;
            break;    
        case BSIM3v1_MOD_LNSUB:
            mod->BSIM3v1lnsub = value->rValue;
            mod->BSIM3v1lnsubGiven = TRUE;
            break;
        case BSIM3v1_MOD_LNPEAK:
            mod->BSIM3v1lnpeak = value->rValue;
            mod->BSIM3v1lnpeakGiven = TRUE;
            if (mod->BSIM3v1lnpeak > 1.0e20)
              mod->BSIM3v1lnpeak *= 1.0e-6;
            break;
        case BSIM3v1_MOD_LNGATE:
            mod->BSIM3v1lngate = value->rValue;
            mod->BSIM3v1lngateGiven = TRUE;
            if (mod->BSIM3v1lngate > 1.0e23)
              mod->BSIM3v1lngate *= 1.0e-6;
            break;
        case BSIM3v1_MOD_LGAMMA1:
            mod->BSIM3v1lgamma1 = value->rValue;
            mod->BSIM3v1lgamma1Given = TRUE;
            break;
        case BSIM3v1_MOD_LGAMMA2:
            mod->BSIM3v1lgamma2 = value->rValue;
            mod->BSIM3v1lgamma2Given = TRUE;
            break;
        case BSIM3v1_MOD_LVBX:
            mod->BSIM3v1lvbx = value->rValue;
            mod->BSIM3v1lvbxGiven = TRUE;
            break;
        case BSIM3v1_MOD_LVBM:
            mod->BSIM3v1lvbm = value->rValue;
            mod->BSIM3v1lvbmGiven = TRUE;
            break;
        case BSIM3v1_MOD_LXT:
            mod->BSIM3v1lxt = value->rValue;
            mod->BSIM3v1lxtGiven = TRUE;
            break;
        case  BSIM3v1_MOD_LK1:
            mod->BSIM3v1lk1 = value->rValue;
            mod->BSIM3v1lk1Given = TRUE;
            break;
        case  BSIM3v1_MOD_LKT1:
            mod->BSIM3v1lkt1 = value->rValue;
            mod->BSIM3v1lkt1Given = TRUE;
            break;
        case  BSIM3v1_MOD_LKT1L:
            mod->BSIM3v1lkt1l = value->rValue;
            mod->BSIM3v1lkt1lGiven = TRUE;
            break;
        case  BSIM3v1_MOD_LKT2:
            mod->BSIM3v1lkt2 = value->rValue;
            mod->BSIM3v1lkt2Given = TRUE;
            break;
        case  BSIM3v1_MOD_LK2:
            mod->BSIM3v1lk2 = value->rValue;
            mod->BSIM3v1lk2Given = TRUE;
            break;
        case  BSIM3v1_MOD_LK3:
            mod->BSIM3v1lk3 = value->rValue;
            mod->BSIM3v1lk3Given = TRUE;
            break;
        case  BSIM3v1_MOD_LK3B:
            mod->BSIM3v1lk3b = value->rValue;
            mod->BSIM3v1lk3bGiven = TRUE;
            break;
        case  BSIM3v1_MOD_LNLX:
            mod->BSIM3v1lnlx = value->rValue;
            mod->BSIM3v1lnlxGiven = TRUE;
            break;
        case  BSIM3v1_MOD_LW0:
            mod->BSIM3v1lw0 = value->rValue;
            mod->BSIM3v1lw0Given = TRUE;
            break;
        case  BSIM3v1_MOD_LDVT0:               
            mod->BSIM3v1ldvt0 = value->rValue;
            mod->BSIM3v1ldvt0Given = TRUE;
            break;
        case  BSIM3v1_MOD_LDVT1:             
            mod->BSIM3v1ldvt1 = value->rValue;
            mod->BSIM3v1ldvt1Given = TRUE;
            break;
        case  BSIM3v1_MOD_LDVT2:             
            mod->BSIM3v1ldvt2 = value->rValue;
            mod->BSIM3v1ldvt2Given = TRUE;
            break;
        case  BSIM3v1_MOD_LDVT0W:               
            mod->BSIM3v1ldvt0w = value->rValue;
            mod->BSIM3v1ldvt0Given = TRUE;
            break;
        case  BSIM3v1_MOD_LDVT1W:             
            mod->BSIM3v1ldvt1w = value->rValue;
            mod->BSIM3v1ldvt1wGiven = TRUE;
            break;
        case  BSIM3v1_MOD_LDVT2W:             
            mod->BSIM3v1ldvt2w = value->rValue;
            mod->BSIM3v1ldvt2wGiven = TRUE;
            break;
        case  BSIM3v1_MOD_LDROUT:             
            mod->BSIM3v1ldrout = value->rValue;
            mod->BSIM3v1ldroutGiven = TRUE;
            break;
        case  BSIM3v1_MOD_LDSUB:             
            mod->BSIM3v1ldsub = value->rValue;
            mod->BSIM3v1ldsubGiven = TRUE;
            break;
        case BSIM3v1_MOD_LVTH0:
            mod->BSIM3v1lvth0 = value->rValue;
            mod->BSIM3v1lvth0Given = TRUE;
            break;
        case BSIM3v1_MOD_LUA:
            mod->BSIM3v1lua = value->rValue;
            mod->BSIM3v1luaGiven = TRUE;
            break;
        case BSIM3v1_MOD_LUA1:
            mod->BSIM3v1lua1 = value->rValue;
            mod->BSIM3v1lua1Given = TRUE;
            break;
        case BSIM3v1_MOD_LUB:
            mod->BSIM3v1lub = value->rValue;
            mod->BSIM3v1lubGiven = TRUE;
            break;
        case BSIM3v1_MOD_LUB1:
            mod->BSIM3v1lub1 = value->rValue;
            mod->BSIM3v1lub1Given = TRUE;
            break;
        case BSIM3v1_MOD_LUC:
            mod->BSIM3v1luc = value->rValue;
            mod->BSIM3v1lucGiven = TRUE;
            break;
        case BSIM3v1_MOD_LUC1:
            mod->BSIM3v1luc1 = value->rValue;
            mod->BSIM3v1luc1Given = TRUE;
            break;
        case  BSIM3v1_MOD_LU0 :
            mod->BSIM3v1lu0 = value->rValue;
            mod->BSIM3v1lu0Given = TRUE;
            break;
        case  BSIM3v1_MOD_LUTE :
            mod->BSIM3v1lute = value->rValue;
            mod->BSIM3v1luteGiven = TRUE;
            break;
        case BSIM3v1_MOD_LVOFF:
            mod->BSIM3v1lvoff = value->rValue;
            mod->BSIM3v1lvoffGiven = TRUE;
            break;
        case  BSIM3v1_MOD_LDELTA :
            mod->BSIM3v1ldelta = value->rValue;
            mod->BSIM3v1ldeltaGiven = TRUE;
            break;
        case BSIM3v1_MOD_LRDSW:
            mod->BSIM3v1lrdsw = value->rValue;
            mod->BSIM3v1lrdswGiven = TRUE;
            break;                     
        case BSIM3v1_MOD_LPRWB:
            mod->BSIM3v1lprwb = value->rValue;
            mod->BSIM3v1lprwbGiven = TRUE;
            break;                     
        case BSIM3v1_MOD_LPRWG:
            mod->BSIM3v1lprwg = value->rValue;
            mod->BSIM3v1lprwgGiven = TRUE;
            break;                     
        case BSIM3v1_MOD_LPRT:
            mod->BSIM3v1lprt = value->rValue;
            mod->BSIM3v1lprtGiven = TRUE;
            break;                     
        case BSIM3v1_MOD_LETA0:
            mod->BSIM3v1leta0 = value->rValue;
            mod->BSIM3v1leta0Given = TRUE;
            break;                 
        case BSIM3v1_MOD_LETAB:
            mod->BSIM3v1letab = value->rValue;
            mod->BSIM3v1letabGiven = TRUE;
            break;                 
        case BSIM3v1_MOD_LPCLM:
            mod->BSIM3v1lpclm = value->rValue;
            mod->BSIM3v1lpclmGiven = TRUE;
            break;                 
        case BSIM3v1_MOD_LPDIBL1:
            mod->BSIM3v1lpdibl1 = value->rValue;
            mod->BSIM3v1lpdibl1Given = TRUE;
            break;                 
        case BSIM3v1_MOD_LPDIBL2:
            mod->BSIM3v1lpdibl2 = value->rValue;
            mod->BSIM3v1lpdibl2Given = TRUE;
            break;                 
        case BSIM3v1_MOD_LPDIBLB:
            mod->BSIM3v1lpdiblb = value->rValue;
            mod->BSIM3v1lpdiblbGiven = TRUE;
            break;                 
        case BSIM3v1_MOD_LPSCBE1:
            mod->BSIM3v1lpscbe1 = value->rValue;
            mod->BSIM3v1lpscbe1Given = TRUE;
            break;                 
        case BSIM3v1_MOD_LPSCBE2:
            mod->BSIM3v1lpscbe2 = value->rValue;
            mod->BSIM3v1lpscbe2Given = TRUE;
            break;                 
        case BSIM3v1_MOD_LPVAG:
            mod->BSIM3v1lpvag = value->rValue;
            mod->BSIM3v1lpvagGiven = TRUE;
            break;                 
        case  BSIM3v1_MOD_LWR :
            mod->BSIM3v1lwr = value->rValue;
            mod->BSIM3v1lwrGiven = TRUE;
            break;
        case  BSIM3v1_MOD_LDWG :
            mod->BSIM3v1ldwg = value->rValue;
            mod->BSIM3v1ldwgGiven = TRUE;
            break;
        case  BSIM3v1_MOD_LDWB :
            mod->BSIM3v1ldwb = value->rValue;
            mod->BSIM3v1ldwbGiven = TRUE;
            break;
        case  BSIM3v1_MOD_LB0 :
            mod->BSIM3v1lb0 = value->rValue;
            mod->BSIM3v1lb0Given = TRUE;
            break;
        case  BSIM3v1_MOD_LB1 :
            mod->BSIM3v1lb1 = value->rValue;
            mod->BSIM3v1lb1Given = TRUE;
            break;
        case  BSIM3v1_MOD_LALPHA0 :
            mod->BSIM3v1lalpha0 = value->rValue;
            mod->BSIM3v1lalpha0Given = TRUE;
            break;
        case  BSIM3v1_MOD_LBETA0 :
            mod->BSIM3v1lbeta0 = value->rValue;
            mod->BSIM3v1lbeta0Given = TRUE;
            break;

        case  BSIM3v1_MOD_LELM :
            mod->BSIM3v1lelm = value->rValue;
            mod->BSIM3v1lelmGiven = TRUE;
            break;
        case  BSIM3v1_MOD_LCGSL :
            mod->BSIM3v1lcgsl = value->rValue;
            mod->BSIM3v1lcgslGiven = TRUE;
            break;
        case  BSIM3v1_MOD_LCGDL :
            mod->BSIM3v1lcgdl = value->rValue;
            mod->BSIM3v1lcgdlGiven = TRUE;
            break;
        case  BSIM3v1_MOD_LCKAPPA :
            mod->BSIM3v1lckappa = value->rValue;
            mod->BSIM3v1lckappaGiven = TRUE;
            break;
        case  BSIM3v1_MOD_LCF :
            mod->BSIM3v1lcf = value->rValue;
            mod->BSIM3v1lcfGiven = TRUE;
            break;
        case  BSIM3v1_MOD_LCLC :
            mod->BSIM3v1lclc = value->rValue;
            mod->BSIM3v1lclcGiven = TRUE;
            break;
        case  BSIM3v1_MOD_LCLE :
            mod->BSIM3v1lcle = value->rValue;
            mod->BSIM3v1lcleGiven = TRUE;
            break;
        case  BSIM3v1_MOD_LVFBCV :
            mod->BSIM3v1lvfbcv = value->rValue;
            mod->BSIM3v1lvfbcvGiven = TRUE;
            break;

	/* Width dependence */
        case  BSIM3v1_MOD_WCDSC :
            mod->BSIM3v1wcdsc = value->rValue;
            mod->BSIM3v1wcdscGiven = TRUE;
            break;
       
       
         case  BSIM3v1_MOD_WCDSCB :
            mod->BSIM3v1wcdscb = value->rValue;
            mod->BSIM3v1wcdscbGiven = TRUE;
            break;
         case  BSIM3v1_MOD_WCDSCD :
            mod->BSIM3v1wcdscd = value->rValue;
            mod->BSIM3v1wcdscdGiven = TRUE;
            break;
        case  BSIM3v1_MOD_WCIT :
            mod->BSIM3v1wcit = value->rValue;
            mod->BSIM3v1wcitGiven = TRUE;
            break;
        case  BSIM3v1_MOD_WNFACTOR :
            mod->BSIM3v1wnfactor = value->rValue;
            mod->BSIM3v1wnfactorGiven = TRUE;
            break;
        case BSIM3v1_MOD_WXJ:
            mod->BSIM3v1wxj = value->rValue;
            mod->BSIM3v1wxjGiven = TRUE;
            break;
        case BSIM3v1_MOD_WVSAT:
            mod->BSIM3v1wvsat = value->rValue;
            mod->BSIM3v1wvsatGiven = TRUE;
            break;


        case BSIM3v1_MOD_WA0:
            mod->BSIM3v1wa0 = value->rValue;
            mod->BSIM3v1wa0Given = TRUE;
            break;
        case BSIM3v1_MOD_WAGS:
            mod->BSIM3v1wags = value->rValue;
            mod->BSIM3v1wagsGiven = TRUE;
            break;
        case BSIM3v1_MOD_WA1:
            mod->BSIM3v1wa1 = value->rValue;
            mod->BSIM3v1wa1Given = TRUE;
            break;
        case BSIM3v1_MOD_WA2:
            mod->BSIM3v1wa2 = value->rValue;
            mod->BSIM3v1wa2Given = TRUE;
            break;
        case BSIM3v1_MOD_WAT:
            mod->BSIM3v1wat = value->rValue;
            mod->BSIM3v1watGiven = TRUE;
            break;
        case BSIM3v1_MOD_WKETA:
            mod->BSIM3v1wketa = value->rValue;
            mod->BSIM3v1wketaGiven = TRUE;
            break;    
        case BSIM3v1_MOD_WNSUB:
            mod->BSIM3v1wnsub = value->rValue;
            mod->BSIM3v1wnsubGiven = TRUE;
            break;
        case BSIM3v1_MOD_WNPEAK:
            mod->BSIM3v1wnpeak = value->rValue;
            mod->BSIM3v1wnpeakGiven = TRUE;
            if (mod->BSIM3v1wnpeak > 1.0e20)
              mod->BSIM3v1wnpeak *= 1.0e-6;
            break;
        case BSIM3v1_MOD_WNGATE:
            mod->BSIM3v1wngate = value->rValue;
            mod->BSIM3v1wngateGiven = TRUE;
            if (mod->BSIM3v1wngate > 1.0e23)
              mod->BSIM3v1wngate *= 1.0e-6;
            break;
        case BSIM3v1_MOD_WGAMMA1:
            mod->BSIM3v1wgamma1 = value->rValue;
            mod->BSIM3v1wgamma1Given = TRUE;
            break;
        case BSIM3v1_MOD_WGAMMA2:
            mod->BSIM3v1wgamma2 = value->rValue;
            mod->BSIM3v1wgamma2Given = TRUE;
            break;
        case BSIM3v1_MOD_WVBX:
            mod->BSIM3v1wvbx = value->rValue;
            mod->BSIM3v1wvbxGiven = TRUE;
            break;
        case BSIM3v1_MOD_WVBM:
            mod->BSIM3v1wvbm = value->rValue;
            mod->BSIM3v1wvbmGiven = TRUE;
            break;
        case BSIM3v1_MOD_WXT:
            mod->BSIM3v1wxt = value->rValue;
            mod->BSIM3v1wxtGiven = TRUE;
            break;
        case  BSIM3v1_MOD_WK1:
            mod->BSIM3v1wk1 = value->rValue;
            mod->BSIM3v1wk1Given = TRUE;
            break;
        case  BSIM3v1_MOD_WKT1:
            mod->BSIM3v1wkt1 = value->rValue;
            mod->BSIM3v1wkt1Given = TRUE;
            break;
        case  BSIM3v1_MOD_WKT1L:
            mod->BSIM3v1wkt1l = value->rValue;
            mod->BSIM3v1wkt1lGiven = TRUE;
            break;
        case  BSIM3v1_MOD_WKT2:
            mod->BSIM3v1wkt2 = value->rValue;
            mod->BSIM3v1wkt2Given = TRUE;
            break;
        case  BSIM3v1_MOD_WK2:
            mod->BSIM3v1wk2 = value->rValue;
            mod->BSIM3v1wk2Given = TRUE;
            break;
        case  BSIM3v1_MOD_WK3:
            mod->BSIM3v1wk3 = value->rValue;
            mod->BSIM3v1wk3Given = TRUE;
            break;
        case  BSIM3v1_MOD_WK3B:
            mod->BSIM3v1wk3b = value->rValue;
            mod->BSIM3v1wk3bGiven = TRUE;
            break;
        case  BSIM3v1_MOD_WNLX:
            mod->BSIM3v1wnlx = value->rValue;
            mod->BSIM3v1wnlxGiven = TRUE;
            break;
        case  BSIM3v1_MOD_WW0:
            mod->BSIM3v1ww0 = value->rValue;
            mod->BSIM3v1ww0Given = TRUE;
            break;
        case  BSIM3v1_MOD_WDVT0:               
            mod->BSIM3v1wdvt0 = value->rValue;
            mod->BSIM3v1wdvt0Given = TRUE;
            break;
        case  BSIM3v1_MOD_WDVT1:             
            mod->BSIM3v1wdvt1 = value->rValue;
            mod->BSIM3v1wdvt1Given = TRUE;
            break;
        case  BSIM3v1_MOD_WDVT2:             
            mod->BSIM3v1wdvt2 = value->rValue;
            mod->BSIM3v1wdvt2Given = TRUE;
            break;
        case  BSIM3v1_MOD_WDVT0W:               
            mod->BSIM3v1wdvt0w = value->rValue;
            mod->BSIM3v1wdvt0wGiven = TRUE;
            break;
        case  BSIM3v1_MOD_WDVT1W:             
            mod->BSIM3v1wdvt1w = value->rValue;
            mod->BSIM3v1wdvt1wGiven = TRUE;
            break;
        case  BSIM3v1_MOD_WDVT2W:             
            mod->BSIM3v1wdvt2w = value->rValue;
            mod->BSIM3v1wdvt2wGiven = TRUE;
            break;
        case  BSIM3v1_MOD_WDROUT:             
            mod->BSIM3v1wdrout = value->rValue;
            mod->BSIM3v1wdroutGiven = TRUE;
            break;
        case  BSIM3v1_MOD_WDSUB:             
            mod->BSIM3v1wdsub = value->rValue;
            mod->BSIM3v1wdsubGiven = TRUE;
            break;
        case BSIM3v1_MOD_WVTH0:
            mod->BSIM3v1wvth0 = value->rValue;
            mod->BSIM3v1wvth0Given = TRUE;
            break;
        case BSIM3v1_MOD_WUA:
            mod->BSIM3v1wua = value->rValue;
            mod->BSIM3v1wuaGiven = TRUE;
            break;
        case BSIM3v1_MOD_WUA1:
            mod->BSIM3v1wua1 = value->rValue;
            mod->BSIM3v1wua1Given = TRUE;
            break;
        case BSIM3v1_MOD_WUB:
            mod->BSIM3v1wub = value->rValue;
            mod->BSIM3v1wubGiven = TRUE;
            break;
        case BSIM3v1_MOD_WUB1:
            mod->BSIM3v1wub1 = value->rValue;
            mod->BSIM3v1wub1Given = TRUE;
            break;
        case BSIM3v1_MOD_WUC:
            mod->BSIM3v1wuc = value->rValue;
            mod->BSIM3v1wucGiven = TRUE;
            break;
        case BSIM3v1_MOD_WUC1:
            mod->BSIM3v1wuc1 = value->rValue;
            mod->BSIM3v1wuc1Given = TRUE;
            break;
        case  BSIM3v1_MOD_WU0 :
            mod->BSIM3v1wu0 = value->rValue;
            mod->BSIM3v1wu0Given = TRUE;
            break;
        case  BSIM3v1_MOD_WUTE :
            mod->BSIM3v1wute = value->rValue;
            mod->BSIM3v1wuteGiven = TRUE;
            break;
        case BSIM3v1_MOD_WVOFF:
            mod->BSIM3v1wvoff = value->rValue;
            mod->BSIM3v1wvoffGiven = TRUE;
            break;
        case  BSIM3v1_MOD_WDELTA :
            mod->BSIM3v1wdelta = value->rValue;
            mod->BSIM3v1wdeltaGiven = TRUE;
            break;
        case BSIM3v1_MOD_WRDSW:
            mod->BSIM3v1wrdsw = value->rValue;
            mod->BSIM3v1wrdswGiven = TRUE;
            break;                     
        case BSIM3v1_MOD_WPRWB:
            mod->BSIM3v1wprwb = value->rValue;
            mod->BSIM3v1wprwbGiven = TRUE;
            break;                     
        case BSIM3v1_MOD_WPRWG:
            mod->BSIM3v1wprwg = value->rValue;
            mod->BSIM3v1wprwgGiven = TRUE;
            break;                     
        case BSIM3v1_MOD_WPRT:
            mod->BSIM3v1wprt = value->rValue;
            mod->BSIM3v1wprtGiven = TRUE;
            break;                     
        case BSIM3v1_MOD_WETA0:
            mod->BSIM3v1weta0 = value->rValue;
            mod->BSIM3v1weta0Given = TRUE;
            break;                 
        case BSIM3v1_MOD_WETAB:
            mod->BSIM3v1wetab = value->rValue;
            mod->BSIM3v1wetabGiven = TRUE;
            break;                 
        case BSIM3v1_MOD_WPCLM:
            mod->BSIM3v1wpclm = value->rValue;
            mod->BSIM3v1wpclmGiven = TRUE;
            break;                 
        case BSIM3v1_MOD_WPDIBL1:
            mod->BSIM3v1wpdibl1 = value->rValue;
            mod->BSIM3v1wpdibl1Given = TRUE;
            break;                 
        case BSIM3v1_MOD_WPDIBL2:
            mod->BSIM3v1wpdibl2 = value->rValue;
            mod->BSIM3v1wpdibl2Given = TRUE;
            break;                 
        case BSIM3v1_MOD_WPDIBLB:
            mod->BSIM3v1wpdiblb = value->rValue;
            mod->BSIM3v1wpdiblbGiven = TRUE;
            break;                 
        case BSIM3v1_MOD_WPSCBE1:
            mod->BSIM3v1wpscbe1 = value->rValue;
            mod->BSIM3v1wpscbe1Given = TRUE;
            break;                 
        case BSIM3v1_MOD_WPSCBE2:
            mod->BSIM3v1wpscbe2 = value->rValue;
            mod->BSIM3v1wpscbe2Given = TRUE;
            break;                 
        case BSIM3v1_MOD_WPVAG:
            mod->BSIM3v1wpvag = value->rValue;
            mod->BSIM3v1wpvagGiven = TRUE;
            break;                 
        case  BSIM3v1_MOD_WWR :
            mod->BSIM3v1wwr = value->rValue;
            mod->BSIM3v1wwrGiven = TRUE;
            break;
        case  BSIM3v1_MOD_WDWG :
            mod->BSIM3v1wdwg = value->rValue;
            mod->BSIM3v1wdwgGiven = TRUE;
            break;
        case  BSIM3v1_MOD_WDWB :
            mod->BSIM3v1wdwb = value->rValue;
            mod->BSIM3v1wdwbGiven = TRUE;
            break;
        case  BSIM3v1_MOD_WB0 :
            mod->BSIM3v1wb0 = value->rValue;
            mod->BSIM3v1wb0Given = TRUE;
            break;
        case  BSIM3v1_MOD_WB1 :
            mod->BSIM3v1wb1 = value->rValue;
            mod->BSIM3v1wb1Given = TRUE;
            break;
        case  BSIM3v1_MOD_WALPHA0 :
            mod->BSIM3v1walpha0 = value->rValue;
            mod->BSIM3v1walpha0Given = TRUE;
            break;
        case  BSIM3v1_MOD_WBETA0 :
            mod->BSIM3v1wbeta0 = value->rValue;
            mod->BSIM3v1wbeta0Given = TRUE;
            break;

        case  BSIM3v1_MOD_WELM :
            mod->BSIM3v1welm = value->rValue;
            mod->BSIM3v1welmGiven = TRUE;
            break;
        case  BSIM3v1_MOD_WCGSL :
            mod->BSIM3v1wcgsl = value->rValue;
            mod->BSIM3v1wcgslGiven = TRUE;
            break;
        case  BSIM3v1_MOD_WCGDL :
            mod->BSIM3v1wcgdl = value->rValue;
            mod->BSIM3v1wcgdlGiven = TRUE;
            break;
        case  BSIM3v1_MOD_WCKAPPA :
            mod->BSIM3v1wckappa = value->rValue;
            mod->BSIM3v1wckappaGiven = TRUE;
            break;
        case  BSIM3v1_MOD_WCF :
            mod->BSIM3v1wcf = value->rValue;
            mod->BSIM3v1wcfGiven = TRUE;
            break;
        case  BSIM3v1_MOD_WCLC :
            mod->BSIM3v1wclc = value->rValue;
            mod->BSIM3v1wclcGiven = TRUE;
            break;
        case  BSIM3v1_MOD_WCLE :
            mod->BSIM3v1wcle = value->rValue;
            mod->BSIM3v1wcleGiven = TRUE;
            break;
        case  BSIM3v1_MOD_WVFBCV :
            mod->BSIM3v1wvfbcv = value->rValue;
            mod->BSIM3v1wvfbcvGiven = TRUE;
            break;

	/* Cross-term dependence */
        case  BSIM3v1_MOD_PCDSC :
            mod->BSIM3v1pcdsc = value->rValue;
            mod->BSIM3v1pcdscGiven = TRUE;
            break;


        case  BSIM3v1_MOD_PCDSCB :
            mod->BSIM3v1pcdscb = value->rValue;
            mod->BSIM3v1pcdscbGiven = TRUE;
            break;
        case  BSIM3v1_MOD_PCDSCD :
            mod->BSIM3v1pcdscd = value->rValue;
            mod->BSIM3v1pcdscdGiven = TRUE;
            break;
        case  BSIM3v1_MOD_PCIT :
            mod->BSIM3v1pcit = value->rValue;
            mod->BSIM3v1pcitGiven = TRUE;
            break;
        case  BSIM3v1_MOD_PNFACTOR :
            mod->BSIM3v1pnfactor = value->rValue;
            mod->BSIM3v1pnfactorGiven = TRUE;
            break;
        case BSIM3v1_MOD_PXJ:
            mod->BSIM3v1pxj = value->rValue;
            mod->BSIM3v1pxjGiven = TRUE;
            break;
        case BSIM3v1_MOD_PVSAT:
            mod->BSIM3v1pvsat = value->rValue;
            mod->BSIM3v1pvsatGiven = TRUE;
            break;


        case BSIM3v1_MOD_PA0:
            mod->BSIM3v1pa0 = value->rValue;
            mod->BSIM3v1pa0Given = TRUE;
            break;
        case BSIM3v1_MOD_PAGS:
            mod->BSIM3v1pags = value->rValue;
            mod->BSIM3v1pagsGiven = TRUE;
            break;
        case BSIM3v1_MOD_PA1:
            mod->BSIM3v1pa1 = value->rValue;
            mod->BSIM3v1pa1Given = TRUE;
            break;
        case BSIM3v1_MOD_PA2:
            mod->BSIM3v1pa2 = value->rValue;
            mod->BSIM3v1pa2Given = TRUE;
            break;
        case BSIM3v1_MOD_PAT:
            mod->BSIM3v1pat = value->rValue;
            mod->BSIM3v1patGiven = TRUE;
            break;
        case BSIM3v1_MOD_PKETA:
            mod->BSIM3v1pketa = value->rValue;
            mod->BSIM3v1pketaGiven = TRUE;
            break;    
        case BSIM3v1_MOD_PNSUB:
            mod->BSIM3v1pnsub = value->rValue;
            mod->BSIM3v1pnsubGiven = TRUE;
            break;
        case BSIM3v1_MOD_PNPEAK:
            mod->BSIM3v1pnpeak = value->rValue;
            mod->BSIM3v1pnpeakGiven = TRUE;
            if (mod->BSIM3v1pnpeak > 1.0e20)
              mod->BSIM3v1pnpeak *= 1.0e-6;
            break;
        case BSIM3v1_MOD_PNGATE:
            mod->BSIM3v1pngate = value->rValue;
            mod->BSIM3v1pngateGiven = TRUE;
            if (mod->BSIM3v1pngate > 1.0e23)
              mod->BSIM3v1pngate *= 1.0e-6;
            break;
        case BSIM3v1_MOD_PGAMMA1:
            mod->BSIM3v1pgamma1 = value->rValue;
            mod->BSIM3v1pgamma1Given = TRUE;
            break;
        case BSIM3v1_MOD_PGAMMA2:
            mod->BSIM3v1pgamma2 = value->rValue;
            mod->BSIM3v1pgamma2Given = TRUE;
            break;
        case BSIM3v1_MOD_PVBX:
            mod->BSIM3v1pvbx = value->rValue;
            mod->BSIM3v1pvbxGiven = TRUE;
            break;
        case BSIM3v1_MOD_PVBM:
            mod->BSIM3v1pvbm = value->rValue;
            mod->BSIM3v1pvbmGiven = TRUE;
            break;
        case BSIM3v1_MOD_PXT:
            mod->BSIM3v1pxt = value->rValue;
            mod->BSIM3v1pxtGiven = TRUE;
            break;
        case  BSIM3v1_MOD_PK1:
            mod->BSIM3v1pk1 = value->rValue;
            mod->BSIM3v1pk1Given = TRUE;
            break;
        case  BSIM3v1_MOD_PKT1:
            mod->BSIM3v1pkt1 = value->rValue;
            mod->BSIM3v1pkt1Given = TRUE;
            break;
        case  BSIM3v1_MOD_PKT1L:
            mod->BSIM3v1pkt1l = value->rValue;
            mod->BSIM3v1pkt1lGiven = TRUE;
            break;
        case  BSIM3v1_MOD_PKT2:
            mod->BSIM3v1pkt2 = value->rValue;
            mod->BSIM3v1pkt2Given = TRUE;
            break;
        case  BSIM3v1_MOD_PK2:
            mod->BSIM3v1pk2 = value->rValue;
            mod->BSIM3v1pk2Given = TRUE;
            break;
        case  BSIM3v1_MOD_PK3:
            mod->BSIM3v1pk3 = value->rValue;
            mod->BSIM3v1pk3Given = TRUE;
            break;
        case  BSIM3v1_MOD_PK3B:
            mod->BSIM3v1pk3b = value->rValue;
            mod->BSIM3v1pk3bGiven = TRUE;
            break;
        case  BSIM3v1_MOD_PNLX:
            mod->BSIM3v1pnlx = value->rValue;
            mod->BSIM3v1pnlxGiven = TRUE;
            break;
        case  BSIM3v1_MOD_PW0:
            mod->BSIM3v1pw0 = value->rValue;
            mod->BSIM3v1pw0Given = TRUE;
            break;
        case  BSIM3v1_MOD_PDVT0:               
            mod->BSIM3v1pdvt0 = value->rValue;
            mod->BSIM3v1pdvt0Given = TRUE;
            break;
        case  BSIM3v1_MOD_PDVT1:             
            mod->BSIM3v1pdvt1 = value->rValue;
            mod->BSIM3v1pdvt1Given = TRUE;
            break;
        case  BSIM3v1_MOD_PDVT2:             
            mod->BSIM3v1pdvt2 = value->rValue;
            mod->BSIM3v1pdvt2Given = TRUE;
            break;
        case  BSIM3v1_MOD_PDVT0W:               
            mod->BSIM3v1pdvt0w = value->rValue;
            mod->BSIM3v1pdvt0wGiven = TRUE;
            break;
        case  BSIM3v1_MOD_PDVT1W:             
            mod->BSIM3v1pdvt1w = value->rValue;
            mod->BSIM3v1pdvt1wGiven = TRUE;
            break;
        case  BSIM3v1_MOD_PDVT2W:             
            mod->BSIM3v1pdvt2w = value->rValue;
            mod->BSIM3v1pdvt2wGiven = TRUE;
            break;
        case  BSIM3v1_MOD_PDROUT:             
            mod->BSIM3v1pdrout = value->rValue;
            mod->BSIM3v1pdroutGiven = TRUE;
            break;
        case  BSIM3v1_MOD_PDSUB:             
            mod->BSIM3v1pdsub = value->rValue;
            mod->BSIM3v1pdsubGiven = TRUE;
            break;
        case BSIM3v1_MOD_PVTH0:
            mod->BSIM3v1pvth0 = value->rValue;
            mod->BSIM3v1pvth0Given = TRUE;
            break;
        case BSIM3v1_MOD_PUA:
            mod->BSIM3v1pua = value->rValue;
            mod->BSIM3v1puaGiven = TRUE;
            break;
        case BSIM3v1_MOD_PUA1:
            mod->BSIM3v1pua1 = value->rValue;
            mod->BSIM3v1pua1Given = TRUE;
            break;
        case BSIM3v1_MOD_PUB:
            mod->BSIM3v1pub = value->rValue;
            mod->BSIM3v1pubGiven = TRUE;
            break;
        case BSIM3v1_MOD_PUB1:
            mod->BSIM3v1pub1 = value->rValue;
            mod->BSIM3v1pub1Given = TRUE;
            break;
        case BSIM3v1_MOD_PUC:
            mod->BSIM3v1puc = value->rValue;
            mod->BSIM3v1pucGiven = TRUE;
            break;
        case BSIM3v1_MOD_PUC1:
            mod->BSIM3v1puc1 = value->rValue;
            mod->BSIM3v1puc1Given = TRUE;
            break;
        case  BSIM3v1_MOD_PU0 :
            mod->BSIM3v1pu0 = value->rValue;
            mod->BSIM3v1pu0Given = TRUE;
            break;
        case  BSIM3v1_MOD_PUTE :
            mod->BSIM3v1pute = value->rValue;
            mod->BSIM3v1puteGiven = TRUE;
            break;
        case BSIM3v1_MOD_PVOFF:
            mod->BSIM3v1pvoff = value->rValue;
            mod->BSIM3v1pvoffGiven = TRUE;
            break;
        case  BSIM3v1_MOD_PDELTA :
            mod->BSIM3v1pdelta = value->rValue;
            mod->BSIM3v1pdeltaGiven = TRUE;
            break;
        case BSIM3v1_MOD_PRDSW:
            mod->BSIM3v1prdsw = value->rValue;
            mod->BSIM3v1prdswGiven = TRUE;
            break;                     
        case BSIM3v1_MOD_PPRWB:
            mod->BSIM3v1pprwb = value->rValue;
            mod->BSIM3v1pprwbGiven = TRUE;
            break;                     
        case BSIM3v1_MOD_PPRWG:
            mod->BSIM3v1pprwg = value->rValue;
            mod->BSIM3v1pprwgGiven = TRUE;
            break;                     
        case BSIM3v1_MOD_PPRT:
            mod->BSIM3v1pprt = value->rValue;
            mod->BSIM3v1pprtGiven = TRUE;
            break;                     
        case BSIM3v1_MOD_PETA0:
            mod->BSIM3v1peta0 = value->rValue;
            mod->BSIM3v1peta0Given = TRUE;
            break;                 
        case BSIM3v1_MOD_PETAB:
            mod->BSIM3v1petab = value->rValue;
            mod->BSIM3v1petabGiven = TRUE;
            break;                 
        case BSIM3v1_MOD_PPCLM:
            mod->BSIM3v1ppclm = value->rValue;
            mod->BSIM3v1ppclmGiven = TRUE;
            break;                 
        case BSIM3v1_MOD_PPDIBL1:
            mod->BSIM3v1ppdibl1 = value->rValue;
            mod->BSIM3v1ppdibl1Given = TRUE;
            break;                 
        case BSIM3v1_MOD_PPDIBL2:
            mod->BSIM3v1ppdibl2 = value->rValue;
            mod->BSIM3v1ppdibl2Given = TRUE;
            break;                 
        case BSIM3v1_MOD_PPDIBLB:
            mod->BSIM3v1ppdiblb = value->rValue;
            mod->BSIM3v1ppdiblbGiven = TRUE;
            break;                 
        case BSIM3v1_MOD_PPSCBE1:
            mod->BSIM3v1ppscbe1 = value->rValue;
            mod->BSIM3v1ppscbe1Given = TRUE;
            break;                 
        case BSIM3v1_MOD_PPSCBE2:
            mod->BSIM3v1ppscbe2 = value->rValue;
            mod->BSIM3v1ppscbe2Given = TRUE;
            break;                 
        case BSIM3v1_MOD_PPVAG:
            mod->BSIM3v1ppvag = value->rValue;
            mod->BSIM3v1ppvagGiven = TRUE;
            break;                 
        case  BSIM3v1_MOD_PWR :
            mod->BSIM3v1pwr = value->rValue;
            mod->BSIM3v1pwrGiven = TRUE;
            break;
        case  BSIM3v1_MOD_PDWG :
            mod->BSIM3v1pdwg = value->rValue;
            mod->BSIM3v1pdwgGiven = TRUE;
            break;
        case  BSIM3v1_MOD_PDWB :
            mod->BSIM3v1pdwb = value->rValue;
            mod->BSIM3v1pdwbGiven = TRUE;
            break;
        case  BSIM3v1_MOD_PB0 :
            mod->BSIM3v1pb0 = value->rValue;
            mod->BSIM3v1pb0Given = TRUE;
            break;
        case  BSIM3v1_MOD_PB1 :
            mod->BSIM3v1pb1 = value->rValue;
            mod->BSIM3v1pb1Given = TRUE;
            break;
        case  BSIM3v1_MOD_PALPHA0 :
            mod->BSIM3v1palpha0 = value->rValue;
            mod->BSIM3v1palpha0Given = TRUE;
            break;
        case  BSIM3v1_MOD_PBETA0 :
            mod->BSIM3v1pbeta0 = value->rValue;
            mod->BSIM3v1pbeta0Given = TRUE;
            break;

        case  BSIM3v1_MOD_PELM :
            mod->BSIM3v1pelm = value->rValue;
            mod->BSIM3v1pelmGiven = TRUE;
            break;
        case  BSIM3v1_MOD_PCGSL :
            mod->BSIM3v1pcgsl = value->rValue;
            mod->BSIM3v1pcgslGiven = TRUE;
            break;
        case  BSIM3v1_MOD_PCGDL :
            mod->BSIM3v1pcgdl = value->rValue;
            mod->BSIM3v1pcgdlGiven = TRUE;
            break;
        case  BSIM3v1_MOD_PCKAPPA :
            mod->BSIM3v1pckappa = value->rValue;
            mod->BSIM3v1pckappaGiven = TRUE;
            break;
        case  BSIM3v1_MOD_PCF :
            mod->BSIM3v1pcf = value->rValue;
            mod->BSIM3v1pcfGiven = TRUE;
            break;
        case  BSIM3v1_MOD_PCLC :
            mod->BSIM3v1pclc = value->rValue;
            mod->BSIM3v1pclcGiven = TRUE;
            break;
        case  BSIM3v1_MOD_PCLE :
            mod->BSIM3v1pcle = value->rValue;
            mod->BSIM3v1pcleGiven = TRUE;
            break;
        case  BSIM3v1_MOD_PVFBCV :
            mod->BSIM3v1pvfbcv = value->rValue;
            mod->BSIM3v1pvfbcvGiven = TRUE;
            break;

        case  BSIM3v1_MOD_TNOM :
            mod->BSIM3v1tnom = value->rValue + 273.15;
            mod->BSIM3v1tnomGiven = TRUE;
            break;
        case  BSIM3v1_MOD_CGSO :
            mod->BSIM3v1cgso = value->rValue;
            mod->BSIM3v1cgsoGiven = TRUE;
            break;
        case  BSIM3v1_MOD_CGDO :
            mod->BSIM3v1cgdo = value->rValue;
            mod->BSIM3v1cgdoGiven = TRUE;
            break;
        case  BSIM3v1_MOD_CGBO :
            mod->BSIM3v1cgbo = value->rValue;
            mod->BSIM3v1cgboGiven = TRUE;
            break;
        case  BSIM3v1_MOD_XPART :
            mod->BSIM3v1xpart = value->rValue;
            mod->BSIM3v1xpartGiven = TRUE;
            break;
        case  BSIM3v1_MOD_RSH :
            mod->BSIM3v1sheetResistance = value->rValue;
            mod->BSIM3v1sheetResistanceGiven = TRUE;
            break;
        case  BSIM3v1_MOD_JS :
            mod->BSIM3v1jctSatCurDensity = value->rValue;
            mod->BSIM3v1jctSatCurDensityGiven = TRUE;
            break;
        case  BSIM3v1_MOD_JSW :
            mod->BSIM3v1jctSidewallSatCurDensity = value->rValue;
            mod->BSIM3v1jctSidewallSatCurDensityGiven = TRUE;
            break;
        case  BSIM3v1_MOD_PB :
            mod->BSIM3v1bulkJctPotential = value->rValue;
            mod->BSIM3v1bulkJctPotentialGiven = TRUE;
            break;
        case  BSIM3v1_MOD_MJ :
            mod->BSIM3v1bulkJctBotGradingCoeff = value->rValue;
            mod->BSIM3v1bulkJctBotGradingCoeffGiven = TRUE;
            break;
        case  BSIM3v1_MOD_PBSW :
            mod->BSIM3v1sidewallJctPotential = value->rValue;
            mod->BSIM3v1sidewallJctPotentialGiven = TRUE;
            break;
        case  BSIM3v1_MOD_MJSW :
            mod->BSIM3v1bulkJctSideGradingCoeff = value->rValue;
            mod->BSIM3v1bulkJctSideGradingCoeffGiven = TRUE;
            break;
        case  BSIM3v1_MOD_CJ :
            mod->BSIM3v1unitAreaJctCap = value->rValue;
            mod->BSIM3v1unitAreaJctCapGiven = TRUE;
            break;
        case  BSIM3v1_MOD_CJSW :
            mod->BSIM3v1unitLengthSidewallJctCap = value->rValue;
            mod->BSIM3v1unitLengthSidewallJctCapGiven = TRUE;
            break;
        case  BSIM3v1_MOD_NJ :
            mod->BSIM3v1jctEmissionCoeff = value->rValue;
            mod->BSIM3v1jctEmissionCoeffGiven = TRUE;
            break;
        case  BSIM3v1_MOD_PBSWG :
            mod->BSIM3v1GatesidewallJctPotential = value->rValue;
            mod->BSIM3v1GatesidewallJctPotentialGiven = TRUE;
            break;
        case  BSIM3v1_MOD_MJSWG :
            mod->BSIM3v1bulkJctGateSideGradingCoeff = value->rValue;
            mod->BSIM3v1bulkJctGateSideGradingCoeffGiven = TRUE;
            break;
        case  BSIM3v1_MOD_CJSWG :
            mod->BSIM3v1unitLengthGateSidewallJctCap = value->rValue;
            mod->BSIM3v1unitLengthGateSidewallJctCapGiven = TRUE;
            break;
        case  BSIM3v1_MOD_XTI :
            mod->BSIM3v1jctTempExponent = value->rValue;
            mod->BSIM3v1jctTempExponentGiven = TRUE;
            break;
        case  BSIM3v1_MOD_LINT :
            mod->BSIM3v1Lint = value->rValue;
            mod->BSIM3v1LintGiven = TRUE;
            break;
        case  BSIM3v1_MOD_LL :
            mod->BSIM3v1Ll = value->rValue;
            mod->BSIM3v1LlGiven = TRUE;
            break;
        case  BSIM3v1_MOD_LLN :
            mod->BSIM3v1Lln = value->rValue;
            mod->BSIM3v1LlnGiven = TRUE;
            break;
        case  BSIM3v1_MOD_LW :
            mod->BSIM3v1Lw = value->rValue;
            mod->BSIM3v1LwGiven = TRUE;
            break;
        case  BSIM3v1_MOD_LWN :
            mod->BSIM3v1Lwn = value->rValue;
            mod->BSIM3v1LwnGiven = TRUE;
            break;
        case  BSIM3v1_MOD_LWL :
            mod->BSIM3v1Lwl = value->rValue;
            mod->BSIM3v1LwlGiven = TRUE;
            break;
        case  BSIM3v1_MOD_LMIN :
            mod->BSIM3v1Lmin = value->rValue;
            mod->BSIM3v1LminGiven = TRUE;
            break;
        case  BSIM3v1_MOD_LMAX :
            mod->BSIM3v1Lmax = value->rValue;
            mod->BSIM3v1LmaxGiven = TRUE;
            break;
        case  BSIM3v1_MOD_WINT :
            mod->BSIM3v1Wint = value->rValue;
            mod->BSIM3v1WintGiven = TRUE;
            break;
        case  BSIM3v1_MOD_WL :
            mod->BSIM3v1Wl = value->rValue;
            mod->BSIM3v1WlGiven = TRUE;
            break;
        case  BSIM3v1_MOD_WLN :
            mod->BSIM3v1Wln = value->rValue;
            mod->BSIM3v1WlnGiven = TRUE;
            break;
        case  BSIM3v1_MOD_WW :
            mod->BSIM3v1Ww = value->rValue;
            mod->BSIM3v1WwGiven = TRUE;
            break;
        case  BSIM3v1_MOD_WWN :
            mod->BSIM3v1Wwn = value->rValue;
            mod->BSIM3v1WwnGiven = TRUE;
            break;
        case  BSIM3v1_MOD_WWL :
            mod->BSIM3v1Wwl = value->rValue;
            mod->BSIM3v1WwlGiven = TRUE;
            break;
        case  BSIM3v1_MOD_WMIN :
            mod->BSIM3v1Wmin = value->rValue;
            mod->BSIM3v1WminGiven = TRUE;
            break;
        case  BSIM3v1_MOD_WMAX :
            mod->BSIM3v1Wmax = value->rValue;
            mod->BSIM3v1WmaxGiven = TRUE;
            break;

        case  BSIM3v1_MOD_NOIA :
            mod->BSIM3v1oxideTrapDensityA = value->rValue;
            mod->BSIM3v1oxideTrapDensityAGiven = TRUE;
            break;
        case  BSIM3v1_MOD_NOIB :
            mod->BSIM3v1oxideTrapDensityB = value->rValue;
            mod->BSIM3v1oxideTrapDensityBGiven = TRUE;
            break;
        case  BSIM3v1_MOD_NOIC :
            mod->BSIM3v1oxideTrapDensityC = value->rValue;
            mod->BSIM3v1oxideTrapDensityCGiven = TRUE;
            break;
        case  BSIM3v1_MOD_EM :
            mod->BSIM3v1em = value->rValue;
            mod->BSIM3v1emGiven = TRUE;
            break;
        case  BSIM3v1_MOD_EF :
            mod->BSIM3v1ef = value->rValue;
            mod->BSIM3v1efGiven = TRUE;
            break;
        case  BSIM3v1_MOD_AF :
            mod->BSIM3v1af = value->rValue;
            mod->BSIM3v1afGiven = TRUE;
            break;
        case  BSIM3v1_MOD_KF :
            mod->BSIM3v1kf = value->rValue;
            mod->BSIM3v1kfGiven = TRUE;
            break;
        case  BSIM3v1_MOD_NMOS  :
            if(value->iValue) {
                mod->BSIM3v1type = 1;
                mod->BSIM3v1typeGiven = TRUE;
            }
            break;
        case  BSIM3v1_MOD_PMOS  :
            if(value->iValue) {
                mod->BSIM3v1type = - 1;
                mod->BSIM3v1typeGiven = TRUE;
            }
            break;
/* serban */
        case  BSIM3v1_MOD_HDIF  :
            mod->BSIM3v1hdif = value->rValue;
            mod->BSIM3v1hdifGiven = TRUE;
            break;
        default:
            return(E_BADPARM);
    }
    return(OK);
}


