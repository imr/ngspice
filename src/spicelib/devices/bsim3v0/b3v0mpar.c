/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1995 Min-Chie Jeng and Mansun Chan.
File: b3v0mpar.c
**********/

#include "ngspice/ngspice.h"
#include "bsim3v0def.h"
#include "ngspice/ifsim.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
BSIM3v0mParam(int param, IFvalue *value, GENmodel *inMod)
{
    BSIM3v0model *mod = (BSIM3v0model*)inMod;
    switch(param)
    {   case  BSIM3v0_MOD_MOBMOD :
            mod->BSIM3v0mobMod = value->iValue;
            mod->BSIM3v0mobModGiven = TRUE;
            break;
        case  BSIM3v0_MOD_BINUNIT :
            mod->BSIM3v0binUnit = value->iValue;
            mod->BSIM3v0binUnitGiven = TRUE;
            break;
        case  BSIM3v0_MOD_CAPMOD :
            mod->BSIM3v0capMod = value->iValue;
            mod->BSIM3v0capModGiven = TRUE;
            break;
        case  BSIM3v0_MOD_NQSMOD :
            mod->BSIM3v0nqsMod = value->iValue;
            mod->BSIM3v0nqsModGiven = TRUE;
            break;
        case  BSIM3v0_MOD_NOIMOD :
            mod->BSIM3v0noiMod = value->iValue;
            mod->BSIM3v0noiModGiven = TRUE;
            break;
        case  BSIM3v0_MOD_TOX :
            mod->BSIM3v0tox = value->rValue;
            mod->BSIM3v0toxGiven = TRUE;
            break;

        case  BSIM3v0_MOD_CDSC :
            mod->BSIM3v0cdsc = value->rValue;
            mod->BSIM3v0cdscGiven = TRUE;
            break;
        case  BSIM3v0_MOD_CDSCB :
            mod->BSIM3v0cdscb = value->rValue;
            mod->BSIM3v0cdscbGiven = TRUE;
            break;

        case  BSIM3v0_MOD_CDSCD :
            mod->BSIM3v0cdscd = value->rValue;
            mod->BSIM3v0cdscdGiven = TRUE;
            break;

        case  BSIM3v0_MOD_CIT :
            mod->BSIM3v0cit = value->rValue;
            mod->BSIM3v0citGiven = TRUE;
            break;
        case  BSIM3v0_MOD_NFACTOR :
            mod->BSIM3v0nfactor = value->rValue;
            mod->BSIM3v0nfactorGiven = TRUE;
            break;
        case BSIM3v0_MOD_XJ:
            mod->BSIM3v0xj = value->rValue;
            mod->BSIM3v0xjGiven = TRUE;
            break;
        case BSIM3v0_MOD_VSAT:
            mod->BSIM3v0vsat = value->rValue;
            mod->BSIM3v0vsatGiven = TRUE;
            break;
        case BSIM3v0_MOD_A0:
            mod->BSIM3v0a0 = value->rValue;
            mod->BSIM3v0a0Given = TRUE;
            break;
        
        case BSIM3v0_MOD_AGS:
            mod->BSIM3v0ags= value->rValue;
            mod->BSIM3v0agsGiven = TRUE;
            break;
        
        case BSIM3v0_MOD_A1:
            mod->BSIM3v0a1 = value->rValue;
            mod->BSIM3v0a1Given = TRUE;
            break;
        case BSIM3v0_MOD_A2:
            mod->BSIM3v0a2 = value->rValue;
            mod->BSIM3v0a2Given = TRUE;
            break;
        case BSIM3v0_MOD_AT:
            mod->BSIM3v0at = value->rValue;
            mod->BSIM3v0atGiven = TRUE;
            break;
        case BSIM3v0_MOD_KETA:
            mod->BSIM3v0keta = value->rValue;
            mod->BSIM3v0ketaGiven = TRUE;
            break;    
        case BSIM3v0_MOD_NSUB:
            mod->BSIM3v0nsub = value->rValue;
            mod->BSIM3v0nsubGiven = TRUE;
            break;
        case BSIM3v0_MOD_NPEAK:
            mod->BSIM3v0npeak = value->rValue;
            mod->BSIM3v0npeakGiven = TRUE;
	    if (mod->BSIM3v0npeak > 1.0e20)
		mod->BSIM3v0npeak *= 1.0e-6;
            break;
        case BSIM3v0_MOD_NGATE:
            mod->BSIM3v0ngate = value->rValue;
            mod->BSIM3v0ngateGiven = TRUE;
	    if (mod->BSIM3v0ngate > 1.0e23)
		mod->BSIM3v0ngate *= 1.0e-6;
            break;
        case BSIM3v0_MOD_GAMMA1:
            mod->BSIM3v0gamma1 = value->rValue;
            mod->BSIM3v0gamma1Given = TRUE;
            break;
        case BSIM3v0_MOD_GAMMA2:
            mod->BSIM3v0gamma2 = value->rValue;
            mod->BSIM3v0gamma2Given = TRUE;
            break;
        case BSIM3v0_MOD_VBX:
            mod->BSIM3v0vbx = value->rValue;
            mod->BSIM3v0vbxGiven = TRUE;
            break;
        case BSIM3v0_MOD_VBM:
            mod->BSIM3v0vbm = value->rValue;
            mod->BSIM3v0vbmGiven = TRUE;
            break;
        case BSIM3v0_MOD_XT:
            mod->BSIM3v0xt = value->rValue;
            mod->BSIM3v0xtGiven = TRUE;
            break;
        case  BSIM3v0_MOD_K1:
            mod->BSIM3v0k1 = value->rValue;
            mod->BSIM3v0k1Given = TRUE;
            break;
        case  BSIM3v0_MOD_KT1:
            mod->BSIM3v0kt1 = value->rValue;
            mod->BSIM3v0kt1Given = TRUE;
            break;
        case  BSIM3v0_MOD_KT1L:
            mod->BSIM3v0kt1l = value->rValue;
            mod->BSIM3v0kt1lGiven = TRUE;
            break;
        case  BSIM3v0_MOD_KT2:
            mod->BSIM3v0kt2 = value->rValue;
            mod->BSIM3v0kt2Given = TRUE;
            break;
        case  BSIM3v0_MOD_K2:
            mod->BSIM3v0k2 = value->rValue;
            mod->BSIM3v0k2Given = TRUE;
            break;
        case  BSIM3v0_MOD_K3:
            mod->BSIM3v0k3 = value->rValue;
            mod->BSIM3v0k3Given = TRUE;
            break;
        case  BSIM3v0_MOD_K3B:
            mod->BSIM3v0k3b = value->rValue;
            mod->BSIM3v0k3bGiven = TRUE;
            break;
        case  BSIM3v0_MOD_NLX:
            mod->BSIM3v0nlx = value->rValue;
            mod->BSIM3v0nlxGiven = TRUE;
            break;
        case  BSIM3v0_MOD_W0:
            mod->BSIM3v0w0 = value->rValue;
            mod->BSIM3v0w0Given = TRUE;
            break;
        case  BSIM3v0_MOD_DVT0:               
            mod->BSIM3v0dvt0 = value->rValue;
            mod->BSIM3v0dvt0Given = TRUE;
            break;
        case  BSIM3v0_MOD_DVT1:             
            mod->BSIM3v0dvt1 = value->rValue;
            mod->BSIM3v0dvt1Given = TRUE;
            break;
        case  BSIM3v0_MOD_DVT2:             
            mod->BSIM3v0dvt2 = value->rValue;
            mod->BSIM3v0dvt2Given = TRUE;
            break;
        case  BSIM3v0_MOD_DVT0W:               
            mod->BSIM3v0dvt0w = value->rValue;
            mod->BSIM3v0dvt0wGiven = TRUE;
            break;
        case  BSIM3v0_MOD_DVT1W:             
            mod->BSIM3v0dvt1w = value->rValue;
            mod->BSIM3v0dvt1wGiven = TRUE;
            break;
        case  BSIM3v0_MOD_DVT2W:             
            mod->BSIM3v0dvt2w = value->rValue;
            mod->BSIM3v0dvt2wGiven = TRUE;
            break;
        case  BSIM3v0_MOD_DROUT:             
            mod->BSIM3v0drout = value->rValue;
            mod->BSIM3v0droutGiven = TRUE;
            break;
        case  BSIM3v0_MOD_DSUB:             
            mod->BSIM3v0dsub = value->rValue;
            mod->BSIM3v0dsubGiven = TRUE;
            break;
        case BSIM3v0_MOD_VTH0:
            mod->BSIM3v0vth0 = value->rValue;
            mod->BSIM3v0vth0Given = TRUE;
            break;
        case BSIM3v0_MOD_UA:
            mod->BSIM3v0ua = value->rValue;
            mod->BSIM3v0uaGiven = TRUE;
            break;
        case BSIM3v0_MOD_UA1:
            mod->BSIM3v0ua1 = value->rValue;
            mod->BSIM3v0ua1Given = TRUE;
            break;
        case BSIM3v0_MOD_UB:
            mod->BSIM3v0ub = value->rValue;
            mod->BSIM3v0ubGiven = TRUE;
            break;
        case BSIM3v0_MOD_UB1:
            mod->BSIM3v0ub1 = value->rValue;
            mod->BSIM3v0ub1Given = TRUE;
            break;
        case BSIM3v0_MOD_UC:
            mod->BSIM3v0uc = value->rValue;
            mod->BSIM3v0ucGiven = TRUE;
            break;
        case BSIM3v0_MOD_UC1:
            mod->BSIM3v0uc1 = value->rValue;
            mod->BSIM3v0uc1Given = TRUE;
            break;
        case  BSIM3v0_MOD_U0 :
            mod->BSIM3v0u0 = value->rValue;
            mod->BSIM3v0u0Given = TRUE;
	    if (mod->BSIM3v0u0 > 1.0)
		mod->BSIM3v0u0 *= 1.0e-4;
            break;
        case  BSIM3v0_MOD_UTE :
            mod->BSIM3v0ute = value->rValue;
            mod->BSIM3v0uteGiven = TRUE;
            break;
        case BSIM3v0_MOD_VOFF:
            mod->BSIM3v0voff = value->rValue;
            mod->BSIM3v0voffGiven = TRUE;
            break;
        case  BSIM3v0_MOD_DELTA :
            mod->BSIM3v0delta = value->rValue;
            mod->BSIM3v0deltaGiven = TRUE;
            break;
        case BSIM3v0_MOD_RDSW:
            mod->BSIM3v0rdsw = value->rValue;
            mod->BSIM3v0rdswGiven = TRUE;
            break;                     
        case BSIM3v0_MOD_PRWG:
            mod->BSIM3v0prwg = value->rValue;
            mod->BSIM3v0prwgGiven = TRUE;
            break;                     
        case BSIM3v0_MOD_PRWB:
            mod->BSIM3v0prwb = value->rValue;
            mod->BSIM3v0prwbGiven = TRUE;
            break;                     
        case BSIM3v0_MOD_PRT:
            mod->BSIM3v0prt = value->rValue;
            mod->BSIM3v0prtGiven = TRUE;
            break;                     
        case BSIM3v0_MOD_ETA0:
            mod->BSIM3v0eta0 = value->rValue;
            mod->BSIM3v0eta0Given = TRUE;
            break;                 
        case BSIM3v0_MOD_ETAB:
            mod->BSIM3v0etab = value->rValue;
            mod->BSIM3v0etabGiven = TRUE;
            break;                 
        case BSIM3v0_MOD_PCLM:
            mod->BSIM3v0pclm = value->rValue;
            mod->BSIM3v0pclmGiven = TRUE;
            break;                 
        case BSIM3v0_MOD_PDIBL1:
            mod->BSIM3v0pdibl1 = value->rValue;
            mod->BSIM3v0pdibl1Given = TRUE;
            break;                 
        case BSIM3v0_MOD_PDIBL2:
            mod->BSIM3v0pdibl2 = value->rValue;
            mod->BSIM3v0pdibl2Given = TRUE;
            break;                 
        case BSIM3v0_MOD_PDIBLB:
            mod->BSIM3v0pdiblb = value->rValue;
            mod->BSIM3v0pdiblbGiven = TRUE;
            break;                 
        case BSIM3v0_MOD_PSCBE1:
            mod->BSIM3v0pscbe1 = value->rValue;
            mod->BSIM3v0pscbe1Given = TRUE;
            break;                 
        case BSIM3v0_MOD_PSCBE2:
            mod->BSIM3v0pscbe2 = value->rValue;
            mod->BSIM3v0pscbe2Given = TRUE;
            break;                 
        case BSIM3v0_MOD_PVAG:
            mod->BSIM3v0pvag = value->rValue;
            mod->BSIM3v0pvagGiven = TRUE;
            break;                 
        case  BSIM3v0_MOD_WR :
            mod->BSIM3v0wr = value->rValue;
            mod->BSIM3v0wrGiven = TRUE;
            break;
        case  BSIM3v0_MOD_DWG :
            mod->BSIM3v0dwg = value->rValue;
            mod->BSIM3v0dwgGiven = TRUE;
            break;
        case  BSIM3v0_MOD_DWB :
            mod->BSIM3v0dwb = value->rValue;
            mod->BSIM3v0dwbGiven = TRUE;
            break;
        case  BSIM3v0_MOD_B0 :
            mod->BSIM3v0b0 = value->rValue;
            mod->BSIM3v0b0Given = TRUE;
            break;
        case  BSIM3v0_MOD_B1 :
            mod->BSIM3v0b1 = value->rValue;
            mod->BSIM3v0b1Given = TRUE;
            break;
        case  BSIM3v0_MOD_ALPHA0 :
            mod->BSIM3v0alpha0 = value->rValue;
            mod->BSIM3v0alpha0Given = TRUE;
            break;
        case  BSIM3v0_MOD_BETA0 :
            mod->BSIM3v0beta0 = value->rValue;
            mod->BSIM3v0beta0Given = TRUE;
            break;

        case  BSIM3v0_MOD_ELM :
            mod->BSIM3v0elm = value->rValue;
            mod->BSIM3v0elmGiven = TRUE;
            break;
        case  BSIM3v0_MOD_CGSL :
            mod->BSIM3v0cgsl = value->rValue;
            mod->BSIM3v0cgslGiven = TRUE;
            break;
        case  BSIM3v0_MOD_CGDL :
            mod->BSIM3v0cgdl = value->rValue;
            mod->BSIM3v0cgdlGiven = TRUE;
            break;
        case  BSIM3v0_MOD_CKAPPA :
            mod->BSIM3v0ckappa = value->rValue;
            mod->BSIM3v0ckappaGiven = TRUE;
            break;
        case  BSIM3v0_MOD_CF :
            mod->BSIM3v0cf = value->rValue;
            mod->BSIM3v0cfGiven = TRUE;
            break;
        case  BSIM3v0_MOD_CLC :
            mod->BSIM3v0clc = value->rValue;
            mod->BSIM3v0clcGiven = TRUE;
            break;
        case  BSIM3v0_MOD_CLE :
            mod->BSIM3v0cle = value->rValue;
            mod->BSIM3v0cleGiven = TRUE;
            break;
        case  BSIM3v0_MOD_DWC :
            mod->BSIM3v0dwc = value->rValue;
            mod->BSIM3v0dwcGiven = TRUE;
            break;
        case  BSIM3v0_MOD_DLC :
            mod->BSIM3v0dlc = value->rValue;
            mod->BSIM3v0dlcGiven = TRUE;
            break;

	/* Length dependence */
        case  BSIM3v0_MOD_LCDSC :
            mod->BSIM3v0lcdsc = value->rValue;
            mod->BSIM3v0lcdscGiven = TRUE;
            break;


        case  BSIM3v0_MOD_LCDSCB :
            mod->BSIM3v0lcdscb = value->rValue;
            mod->BSIM3v0lcdscbGiven = TRUE;
            break;
        case  BSIM3v0_MOD_LCDSCD :
            mod->BSIM3v0lcdscd = value->rValue;
            mod->BSIM3v0lcdscdGiven = TRUE;
            break;
        case  BSIM3v0_MOD_LCIT :
            mod->BSIM3v0lcit = value->rValue;
            mod->BSIM3v0lcitGiven = TRUE;
            break;
        case  BSIM3v0_MOD_LNFACTOR :
            mod->BSIM3v0lnfactor = value->rValue;
            mod->BSIM3v0lnfactorGiven = TRUE;
            break;
        case BSIM3v0_MOD_LXJ:
            mod->BSIM3v0lxj = value->rValue;
            mod->BSIM3v0lxjGiven = TRUE;
            break;
        case BSIM3v0_MOD_LVSAT:
            mod->BSIM3v0lvsat = value->rValue;
            mod->BSIM3v0lvsatGiven = TRUE;
            break;
        
        
        case BSIM3v0_MOD_LA0:
            mod->BSIM3v0la0 = value->rValue;
            mod->BSIM3v0la0Given = TRUE;
            break;
        case BSIM3v0_MOD_LAGS:
            mod->BSIM3v0lags = value->rValue;
            mod->BSIM3v0lagsGiven = TRUE;
            break;
        case BSIM3v0_MOD_LA1:
            mod->BSIM3v0la1 = value->rValue;
            mod->BSIM3v0la1Given = TRUE;
            break;
        case BSIM3v0_MOD_LA2:
            mod->BSIM3v0la2 = value->rValue;
            mod->BSIM3v0la2Given = TRUE;
            break;
        case BSIM3v0_MOD_LAT:
            mod->BSIM3v0lat = value->rValue;
            mod->BSIM3v0latGiven = TRUE;
            break;
        case BSIM3v0_MOD_LKETA:
            mod->BSIM3v0lketa = value->rValue;
            mod->BSIM3v0lketaGiven = TRUE;
            break;    
        case BSIM3v0_MOD_LNSUB:
            mod->BSIM3v0lnsub = value->rValue;
            mod->BSIM3v0lnsubGiven = TRUE;
            break;
        case BSIM3v0_MOD_LNPEAK:
            mod->BSIM3v0lnpeak = value->rValue;
            mod->BSIM3v0lnpeakGiven = TRUE;
	    if (mod->BSIM3v0lnpeak > 1.0e20)
		mod->BSIM3v0lnpeak *= 1.0e-6;
            break;
        case BSIM3v0_MOD_LNGATE:
            mod->BSIM3v0lngate = value->rValue;
            mod->BSIM3v0lngateGiven = TRUE;
	    if (mod->BSIM3v0lngate > 1.0e23)
		mod->BSIM3v0lngate *= 1.0e-6;
            break;
        case BSIM3v0_MOD_LGAMMA1:
            mod->BSIM3v0lgamma1 = value->rValue;
            mod->BSIM3v0lgamma1Given = TRUE;
            break;
        case BSIM3v0_MOD_LGAMMA2:
            mod->BSIM3v0lgamma2 = value->rValue;
            mod->BSIM3v0lgamma2Given = TRUE;
            break;
        case BSIM3v0_MOD_LVBX:
            mod->BSIM3v0lvbx = value->rValue;
            mod->BSIM3v0lvbxGiven = TRUE;
            break;
        case BSIM3v0_MOD_LVBM:
            mod->BSIM3v0lvbm = value->rValue;
            mod->BSIM3v0lvbmGiven = TRUE;
            break;
        case BSIM3v0_MOD_LXT:
            mod->BSIM3v0lxt = value->rValue;
            mod->BSIM3v0lxtGiven = TRUE;
            break;
        case  BSIM3v0_MOD_LK1:
            mod->BSIM3v0lk1 = value->rValue;
            mod->BSIM3v0lk1Given = TRUE;
            break;
        case  BSIM3v0_MOD_LKT1:
            mod->BSIM3v0lkt1 = value->rValue;
            mod->BSIM3v0lkt1Given = TRUE;
            break;
        case  BSIM3v0_MOD_LKT1L:
            mod->BSIM3v0lkt1l = value->rValue;
            mod->BSIM3v0lkt1lGiven = TRUE;
            break;
        case  BSIM3v0_MOD_LKT2:
            mod->BSIM3v0lkt2 = value->rValue;
            mod->BSIM3v0lkt2Given = TRUE;
            break;
        case  BSIM3v0_MOD_LK2:
            mod->BSIM3v0lk2 = value->rValue;
            mod->BSIM3v0lk2Given = TRUE;
            break;
        case  BSIM3v0_MOD_LK3:
            mod->BSIM3v0lk3 = value->rValue;
            mod->BSIM3v0lk3Given = TRUE;
            break;
        case  BSIM3v0_MOD_LK3B:
            mod->BSIM3v0lk3b = value->rValue;
            mod->BSIM3v0lk3bGiven = TRUE;
            break;
        case  BSIM3v0_MOD_LNLX:
            mod->BSIM3v0lnlx = value->rValue;
            mod->BSIM3v0lnlxGiven = TRUE;
            break;
        case  BSIM3v0_MOD_LW0:
            mod->BSIM3v0lw0 = value->rValue;
            mod->BSIM3v0lw0Given = TRUE;
            break;
        case  BSIM3v0_MOD_LDVT0:               
            mod->BSIM3v0ldvt0 = value->rValue;
            mod->BSIM3v0ldvt0Given = TRUE;
            break;
        case  BSIM3v0_MOD_LDVT1:             
            mod->BSIM3v0ldvt1 = value->rValue;
            mod->BSIM3v0ldvt1Given = TRUE;
            break;
        case  BSIM3v0_MOD_LDVT2:             
            mod->BSIM3v0ldvt2 = value->rValue;
            mod->BSIM3v0ldvt2Given = TRUE;
            break;
        case  BSIM3v0_MOD_LDVT0W:               
            mod->BSIM3v0ldvt0w = value->rValue;
            mod->BSIM3v0ldvt0Given = TRUE;
            break;
        case  BSIM3v0_MOD_LDVT1W:             
            mod->BSIM3v0ldvt1 = value->rValue;
            mod->BSIM3v0ldvt1wGiven = TRUE;
            break;
        case  BSIM3v0_MOD_LDVT2W:             
            mod->BSIM3v0ldvt2 = value->rValue;
            mod->BSIM3v0ldvt2wGiven = TRUE;
            break;
        case  BSIM3v0_MOD_LDROUT:             
            mod->BSIM3v0ldrout = value->rValue;
            mod->BSIM3v0ldroutGiven = TRUE;
            break;
        case  BSIM3v0_MOD_LDSUB:             
            mod->BSIM3v0ldsub = value->rValue;
            mod->BSIM3v0ldsubGiven = TRUE;
            break;
        case BSIM3v0_MOD_LVTH0:
            mod->BSIM3v0lvth0 = value->rValue;
            mod->BSIM3v0lvth0Given = TRUE;
            break;
        case BSIM3v0_MOD_LUA:
            mod->BSIM3v0lua = value->rValue;
            mod->BSIM3v0luaGiven = TRUE;
            break;
        case BSIM3v0_MOD_LUA1:
            mod->BSIM3v0lua1 = value->rValue;
            mod->BSIM3v0lua1Given = TRUE;
            break;
        case BSIM3v0_MOD_LUB:
            mod->BSIM3v0lub = value->rValue;
            mod->BSIM3v0lubGiven = TRUE;
            break;
        case BSIM3v0_MOD_LUB1:
            mod->BSIM3v0lub1 = value->rValue;
            mod->BSIM3v0lub1Given = TRUE;
            break;
        case BSIM3v0_MOD_LUC:
            mod->BSIM3v0luc = value->rValue;
            mod->BSIM3v0lucGiven = TRUE;
            break;
        case BSIM3v0_MOD_LUC1:
            mod->BSIM3v0luc1 = value->rValue;
            mod->BSIM3v0luc1Given = TRUE;
            break;
        case  BSIM3v0_MOD_LU0 :
            mod->BSIM3v0lu0 = value->rValue;
            mod->BSIM3v0lu0Given = TRUE;
	    if (mod->BSIM3v0lu0 > 1.0)
		mod->BSIM3v0lu0 *= 1.0e-4;
            break;
        case  BSIM3v0_MOD_LUTE :
            mod->BSIM3v0lute = value->rValue;
            mod->BSIM3v0luteGiven = TRUE;
            break;
        case BSIM3v0_MOD_LVOFF:
            mod->BSIM3v0lvoff = value->rValue;
            mod->BSIM3v0lvoffGiven = TRUE;
            break;
        case  BSIM3v0_MOD_LDELTA :
            mod->BSIM3v0ldelta = value->rValue;
            mod->BSIM3v0ldeltaGiven = TRUE;
            break;
        case BSIM3v0_MOD_LRDSW:
            mod->BSIM3v0lrdsw = value->rValue;
            mod->BSIM3v0lrdswGiven = TRUE;
            break;                     
        case BSIM3v0_MOD_LPRWB:
            mod->BSIM3v0lprwb = value->rValue;
            mod->BSIM3v0lprwbGiven = TRUE;
            break;                     
        case BSIM3v0_MOD_LPRWG:
            mod->BSIM3v0lprwg = value->rValue;
            mod->BSIM3v0lprwgGiven = TRUE;
            break;                     
        case BSIM3v0_MOD_LPRT:
            mod->BSIM3v0lprt = value->rValue;
            mod->BSIM3v0lprtGiven = TRUE;
            break;                     
        case BSIM3v0_MOD_LETA0:
            mod->BSIM3v0leta0 = value->rValue;
            mod->BSIM3v0leta0Given = TRUE;
            break;                 
        case BSIM3v0_MOD_LETAB:
            mod->BSIM3v0letab = value->rValue;
            mod->BSIM3v0letabGiven = TRUE;
            break;                 
        case BSIM3v0_MOD_LPCLM:
            mod->BSIM3v0lpclm = value->rValue;
            mod->BSIM3v0lpclmGiven = TRUE;
            break;                 
        case BSIM3v0_MOD_LPDIBL1:
            mod->BSIM3v0lpdibl1 = value->rValue;
            mod->BSIM3v0lpdibl1Given = TRUE;
            break;                 
        case BSIM3v0_MOD_LPDIBL2:
            mod->BSIM3v0lpdibl2 = value->rValue;
            mod->BSIM3v0lpdibl2Given = TRUE;
            break;                 
        case BSIM3v0_MOD_LPDIBLB:
            mod->BSIM3v0lpdiblb = value->rValue;
            mod->BSIM3v0lpdiblbGiven = TRUE;
            break;                 
        case BSIM3v0_MOD_LPSCBE1:
            mod->BSIM3v0lpscbe1 = value->rValue;
            mod->BSIM3v0lpscbe1Given = TRUE;
            break;                 
        case BSIM3v0_MOD_LPSCBE2:
            mod->BSIM3v0lpscbe2 = value->rValue;
            mod->BSIM3v0lpscbe2Given = TRUE;
            break;                 
        case BSIM3v0_MOD_LPVAG:
            mod->BSIM3v0lpvag = value->rValue;
            mod->BSIM3v0lpvagGiven = TRUE;
            break;                 
        case  BSIM3v0_MOD_LWR :
            mod->BSIM3v0lwr = value->rValue;
            mod->BSIM3v0lwrGiven = TRUE;
            break;
        case  BSIM3v0_MOD_LDWG :
            mod->BSIM3v0ldwg = value->rValue;
            mod->BSIM3v0ldwgGiven = TRUE;
            break;
        case  BSIM3v0_MOD_LDWB :
            mod->BSIM3v0ldwb = value->rValue;
            mod->BSIM3v0ldwbGiven = TRUE;
            break;
        case  BSIM3v0_MOD_LB0 :
            mod->BSIM3v0lb0 = value->rValue;
            mod->BSIM3v0lb0Given = TRUE;
            break;
        case  BSIM3v0_MOD_LB1 :
            mod->BSIM3v0lb1 = value->rValue;
            mod->BSIM3v0lb1Given = TRUE;
            break;
        case  BSIM3v0_MOD_LALPHA0 :
            mod->BSIM3v0lalpha0 = value->rValue;
            mod->BSIM3v0lalpha0Given = TRUE;
            break;
        case  BSIM3v0_MOD_LBETA0 :
            mod->BSIM3v0lbeta0 = value->rValue;
            mod->BSIM3v0lbeta0Given = TRUE;
            break;

        case  BSIM3v0_MOD_LELM :
            mod->BSIM3v0lelm = value->rValue;
            mod->BSIM3v0lelmGiven = TRUE;
            break;
        case  BSIM3v0_MOD_LCGSL :
            mod->BSIM3v0lcgsl = value->rValue;
            mod->BSIM3v0lcgslGiven = TRUE;
            break;
        case  BSIM3v0_MOD_LCGDL :
            mod->BSIM3v0lcgdl = value->rValue;
            mod->BSIM3v0lcgdlGiven = TRUE;
            break;
        case  BSIM3v0_MOD_LCKAPPA :
            mod->BSIM3v0lckappa = value->rValue;
            mod->BSIM3v0lckappaGiven = TRUE;
            break;
        case  BSIM3v0_MOD_LCF :
            mod->BSIM3v0lcf = value->rValue;
            mod->BSIM3v0lcfGiven = TRUE;
            break;
        case  BSIM3v0_MOD_LCLC :
            mod->BSIM3v0lclc = value->rValue;
            mod->BSIM3v0lclcGiven = TRUE;
            break;
        case  BSIM3v0_MOD_LCLE :
            mod->BSIM3v0lcle = value->rValue;
            mod->BSIM3v0lcleGiven = TRUE;
            break;

	/* Width dependence */
        case  BSIM3v0_MOD_WCDSC :
            mod->BSIM3v0wcdsc = value->rValue;
            mod->BSIM3v0wcdscGiven = TRUE;
            break;
       
       
         case  BSIM3v0_MOD_WCDSCB :
            mod->BSIM3v0wcdscb = value->rValue;
            mod->BSIM3v0wcdscbGiven = TRUE;
            break;
         case  BSIM3v0_MOD_WCDSCD :
            mod->BSIM3v0wcdscd = value->rValue;
            mod->BSIM3v0wcdscdGiven = TRUE;
            break;
        case  BSIM3v0_MOD_WCIT :
            mod->BSIM3v0wcit = value->rValue;
            mod->BSIM3v0wcitGiven = TRUE;
            break;
        case  BSIM3v0_MOD_WNFACTOR :
            mod->BSIM3v0wnfactor = value->rValue;
            mod->BSIM3v0wnfactorGiven = TRUE;
            break;
        case BSIM3v0_MOD_WXJ:
            mod->BSIM3v0wxj = value->rValue;
            mod->BSIM3v0wxjGiven = TRUE;
            break;
        case BSIM3v0_MOD_WVSAT:
            mod->BSIM3v0wvsat = value->rValue;
            mod->BSIM3v0wvsatGiven = TRUE;
            break;


        case BSIM3v0_MOD_WA0:
            mod->BSIM3v0wa0 = value->rValue;
            mod->BSIM3v0wa0Given = TRUE;
            break;
        case BSIM3v0_MOD_WAGS:
            mod->BSIM3v0wags = value->rValue;
            mod->BSIM3v0wagsGiven = TRUE;
            break;
        case BSIM3v0_MOD_WA1:
            mod->BSIM3v0wa1 = value->rValue;
            mod->BSIM3v0wa1Given = TRUE;
            break;
        case BSIM3v0_MOD_WA2:
            mod->BSIM3v0wa2 = value->rValue;
            mod->BSIM3v0wa2Given = TRUE;
            break;
        case BSIM3v0_MOD_WAT:
            mod->BSIM3v0wat = value->rValue;
            mod->BSIM3v0watGiven = TRUE;
            break;
        case BSIM3v0_MOD_WKETA:
            mod->BSIM3v0wketa = value->rValue;
            mod->BSIM3v0wketaGiven = TRUE;
            break;    
        case BSIM3v0_MOD_WNSUB:
            mod->BSIM3v0wnsub = value->rValue;
            mod->BSIM3v0wnsubGiven = TRUE;
            break;
        case BSIM3v0_MOD_WNPEAK:
            mod->BSIM3v0wnpeak = value->rValue;
            mod->BSIM3v0wnpeakGiven = TRUE;
	    if (mod->BSIM3v0wnpeak > 1.0e20)
		mod->BSIM3v0wnpeak *= 1.0e-6;
            break;
        case BSIM3v0_MOD_WNGATE:
            mod->BSIM3v0wngate = value->rValue;
            mod->BSIM3v0wngateGiven = TRUE;
	    if (mod->BSIM3v0wngate > 1.0e23)
		mod->BSIM3v0wngate *= 1.0e-6;
            break;
        case BSIM3v0_MOD_WGAMMA1:
            mod->BSIM3v0wgamma1 = value->rValue;
            mod->BSIM3v0wgamma1Given = TRUE;
            break;
        case BSIM3v0_MOD_WGAMMA2:
            mod->BSIM3v0wgamma2 = value->rValue;
            mod->BSIM3v0wgamma2Given = TRUE;
            break;
        case BSIM3v0_MOD_WVBX:
            mod->BSIM3v0wvbx = value->rValue;
            mod->BSIM3v0wvbxGiven = TRUE;
            break;
        case BSIM3v0_MOD_WVBM:
            mod->BSIM3v0wvbm = value->rValue;
            mod->BSIM3v0wvbmGiven = TRUE;
            break;
        case BSIM3v0_MOD_WXT:
            mod->BSIM3v0wxt = value->rValue;
            mod->BSIM3v0wxtGiven = TRUE;
            break;
        case  BSIM3v0_MOD_WK1:
            mod->BSIM3v0wk1 = value->rValue;
            mod->BSIM3v0wk1Given = TRUE;
            break;
        case  BSIM3v0_MOD_WKT1:
            mod->BSIM3v0wkt1 = value->rValue;
            mod->BSIM3v0wkt1Given = TRUE;
            break;
        case  BSIM3v0_MOD_WKT1L:
            mod->BSIM3v0wkt1l = value->rValue;
            mod->BSIM3v0wkt1lGiven = TRUE;
            break;
        case  BSIM3v0_MOD_WKT2:
            mod->BSIM3v0wkt2 = value->rValue;
            mod->BSIM3v0wkt2Given = TRUE;
            break;
        case  BSIM3v0_MOD_WK2:
            mod->BSIM3v0wk2 = value->rValue;
            mod->BSIM3v0wk2Given = TRUE;
            break;
        case  BSIM3v0_MOD_WK3:
            mod->BSIM3v0wk3 = value->rValue;
            mod->BSIM3v0wk3Given = TRUE;
            break;
        case  BSIM3v0_MOD_WK3B:
            mod->BSIM3v0wk3b = value->rValue;
            mod->BSIM3v0wk3bGiven = TRUE;
            break;
        case  BSIM3v0_MOD_WNLX:
            mod->BSIM3v0wnlx = value->rValue;
            mod->BSIM3v0wnlxGiven = TRUE;
            break;
        case  BSIM3v0_MOD_WW0:
            mod->BSIM3v0ww0 = value->rValue;
            mod->BSIM3v0ww0Given = TRUE;
            break;
        case  BSIM3v0_MOD_WDVT0:               
            mod->BSIM3v0wdvt0 = value->rValue;
            mod->BSIM3v0wdvt0Given = TRUE;
            break;
        case  BSIM3v0_MOD_WDVT1:             
            mod->BSIM3v0wdvt1 = value->rValue;
            mod->BSIM3v0wdvt1Given = TRUE;
            break;
        case  BSIM3v0_MOD_WDVT2:             
            mod->BSIM3v0wdvt2 = value->rValue;
            mod->BSIM3v0wdvt2Given = TRUE;
            break;
        case  BSIM3v0_MOD_WDVT0W:               
            mod->BSIM3v0wdvt0w = value->rValue;
            mod->BSIM3v0wdvt0wGiven = TRUE;
            break;
        case  BSIM3v0_MOD_WDVT1W:             
            mod->BSIM3v0wdvt1w = value->rValue;
            mod->BSIM3v0wdvt1wGiven = TRUE;
            break;
        case  BSIM3v0_MOD_WDVT2W:             
            mod->BSIM3v0wdvt2w = value->rValue;
            mod->BSIM3v0wdvt2wGiven = TRUE;
            break;
        case  BSIM3v0_MOD_WDROUT:             
            mod->BSIM3v0wdrout = value->rValue;
            mod->BSIM3v0wdroutGiven = TRUE;
            break;
        case  BSIM3v0_MOD_WDSUB:             
            mod->BSIM3v0wdsub = value->rValue;
            mod->BSIM3v0wdsubGiven = TRUE;
            break;
        case BSIM3v0_MOD_WVTH0:
            mod->BSIM3v0wvth0 = value->rValue;
            mod->BSIM3v0wvth0Given = TRUE;
            break;
        case BSIM3v0_MOD_WUA:
            mod->BSIM3v0wua = value->rValue;
            mod->BSIM3v0wuaGiven = TRUE;
            break;
        case BSIM3v0_MOD_WUA1:
            mod->BSIM3v0wua1 = value->rValue;
            mod->BSIM3v0wua1Given = TRUE;
            break;
        case BSIM3v0_MOD_WUB:
            mod->BSIM3v0wub = value->rValue;
            mod->BSIM3v0wubGiven = TRUE;
            break;
        case BSIM3v0_MOD_WUB1:
            mod->BSIM3v0wub1 = value->rValue;
            mod->BSIM3v0wub1Given = TRUE;
            break;
        case BSIM3v0_MOD_WUC:
            mod->BSIM3v0wuc = value->rValue;
            mod->BSIM3v0wucGiven = TRUE;
            break;
        case BSIM3v0_MOD_WUC1:
            mod->BSIM3v0wuc1 = value->rValue;
            mod->BSIM3v0wuc1Given = TRUE;
            break;
        case  BSIM3v0_MOD_WU0 :
            mod->BSIM3v0wu0 = value->rValue;
            mod->BSIM3v0wu0Given = TRUE;
	    if (mod->BSIM3v0wu0 > 1.0)
		mod->BSIM3v0wu0 *= 1.0e-4;
            break;
        case  BSIM3v0_MOD_WUTE :
            mod->BSIM3v0wute = value->rValue;
            mod->BSIM3v0wuteGiven = TRUE;
            break;
        case BSIM3v0_MOD_WVOFF:
            mod->BSIM3v0wvoff = value->rValue;
            mod->BSIM3v0wvoffGiven = TRUE;
            break;
        case  BSIM3v0_MOD_WDELTA :
            mod->BSIM3v0wdelta = value->rValue;
            mod->BSIM3v0wdeltaGiven = TRUE;
            break;
        case BSIM3v0_MOD_WRDSW:
            mod->BSIM3v0wrdsw = value->rValue;
            mod->BSIM3v0wrdswGiven = TRUE;
            break;                     
        case BSIM3v0_MOD_WPRWB:
            mod->BSIM3v0wprwb = value->rValue;
            mod->BSIM3v0wprwbGiven = TRUE;
            break;                     
        case BSIM3v0_MOD_WPRWG:
            mod->BSIM3v0wprwg = value->rValue;
            mod->BSIM3v0wprwgGiven = TRUE;
            break;                     
        case BSIM3v0_MOD_WPRT:
            mod->BSIM3v0wprt = value->rValue;
            mod->BSIM3v0wprtGiven = TRUE;
            break;                     
        case BSIM3v0_MOD_WETA0:
            mod->BSIM3v0weta0 = value->rValue;
            mod->BSIM3v0weta0Given = TRUE;
            break;                 
        case BSIM3v0_MOD_WETAB:
            mod->BSIM3v0wetab = value->rValue;
            mod->BSIM3v0wetabGiven = TRUE;
            break;                 
        case BSIM3v0_MOD_WPCLM:
            mod->BSIM3v0wpclm = value->rValue;
            mod->BSIM3v0wpclmGiven = TRUE;
            break;                 
        case BSIM3v0_MOD_WPDIBL1:
            mod->BSIM3v0wpdibl1 = value->rValue;
            mod->BSIM3v0wpdibl1Given = TRUE;
            break;                 
        case BSIM3v0_MOD_WPDIBL2:
            mod->BSIM3v0wpdibl2 = value->rValue;
            mod->BSIM3v0wpdibl2Given = TRUE;
            break;                 
        case BSIM3v0_MOD_WPDIBLB:
            mod->BSIM3v0wpdiblb = value->rValue;
            mod->BSIM3v0wpdiblbGiven = TRUE;
            break;                 
        case BSIM3v0_MOD_WPSCBE1:
            mod->BSIM3v0wpscbe1 = value->rValue;
            mod->BSIM3v0wpscbe1Given = TRUE;
            break;                 
        case BSIM3v0_MOD_WPSCBE2:
            mod->BSIM3v0wpscbe2 = value->rValue;
            mod->BSIM3v0wpscbe2Given = TRUE;
            break;                 
        case BSIM3v0_MOD_WPVAG:
            mod->BSIM3v0wpvag = value->rValue;
            mod->BSIM3v0wpvagGiven = TRUE;
            break;                 
        case  BSIM3v0_MOD_WWR :
            mod->BSIM3v0wwr = value->rValue;
            mod->BSIM3v0wwrGiven = TRUE;
            break;
        case  BSIM3v0_MOD_WDWG :
            mod->BSIM3v0wdwg = value->rValue;
            mod->BSIM3v0wdwgGiven = TRUE;
            break;
        case  BSIM3v0_MOD_WDWB :
            mod->BSIM3v0wdwb = value->rValue;
            mod->BSIM3v0wdwbGiven = TRUE;
            break;
        case  BSIM3v0_MOD_WB0 :
            mod->BSIM3v0wb0 = value->rValue;
            mod->BSIM3v0wb0Given = TRUE;
            break;
        case  BSIM3v0_MOD_WB1 :
            mod->BSIM3v0wb1 = value->rValue;
            mod->BSIM3v0wb1Given = TRUE;
            break;
        case  BSIM3v0_MOD_WALPHA0 :
            mod->BSIM3v0walpha0 = value->rValue;
            mod->BSIM3v0walpha0Given = TRUE;
            break;
        case  BSIM3v0_MOD_WBETA0 :
            mod->BSIM3v0wbeta0 = value->rValue;
            mod->BSIM3v0wbeta0Given = TRUE;
            break;

        case  BSIM3v0_MOD_WELM :
            mod->BSIM3v0welm = value->rValue;
            mod->BSIM3v0welmGiven = TRUE;
            break;
        case  BSIM3v0_MOD_WCGSL :
            mod->BSIM3v0wcgsl = value->rValue;
            mod->BSIM3v0wcgslGiven = TRUE;
            break;
        case  BSIM3v0_MOD_WCGDL :
            mod->BSIM3v0wcgdl = value->rValue;
            mod->BSIM3v0wcgdlGiven = TRUE;
            break;
        case  BSIM3v0_MOD_WCKAPPA :
            mod->BSIM3v0wckappa = value->rValue;
            mod->BSIM3v0wckappaGiven = TRUE;
            break;
        case  BSIM3v0_MOD_WCF :
            mod->BSIM3v0wcf = value->rValue;
            mod->BSIM3v0wcfGiven = TRUE;
            break;
        case  BSIM3v0_MOD_WCLC :
            mod->BSIM3v0wclc = value->rValue;
            mod->BSIM3v0wclcGiven = TRUE;
            break;
        case  BSIM3v0_MOD_WCLE :
            mod->BSIM3v0wcle = value->rValue;
            mod->BSIM3v0wcleGiven = TRUE;
            break;

	/* Cross-term dependence */
        case  BSIM3v0_MOD_PCDSC :
            mod->BSIM3v0pcdsc = value->rValue;
            mod->BSIM3v0pcdscGiven = TRUE;
            break;


        case  BSIM3v0_MOD_PCDSCB :
            mod->BSIM3v0pcdscb = value->rValue;
            mod->BSIM3v0pcdscbGiven = TRUE;
            break;
        case  BSIM3v0_MOD_PCDSCD :
            mod->BSIM3v0pcdscd = value->rValue;
            mod->BSIM3v0pcdscdGiven = TRUE;
            break;
        case  BSIM3v0_MOD_PCIT :
            mod->BSIM3v0pcit = value->rValue;
            mod->BSIM3v0pcitGiven = TRUE;
            break;
        case  BSIM3v0_MOD_PNFACTOR :
            mod->BSIM3v0pnfactor = value->rValue;
            mod->BSIM3v0pnfactorGiven = TRUE;
            break;
        case BSIM3v0_MOD_PXJ:
            mod->BSIM3v0pxj = value->rValue;
            mod->BSIM3v0pxjGiven = TRUE;
            break;
        case BSIM3v0_MOD_PVSAT:
            mod->BSIM3v0pvsat = value->rValue;
            mod->BSIM3v0pvsatGiven = TRUE;
            break;


        case BSIM3v0_MOD_PA0:
            mod->BSIM3v0pa0 = value->rValue;
            mod->BSIM3v0pa0Given = TRUE;
            break;
        case BSIM3v0_MOD_PAGS:
            mod->BSIM3v0pags = value->rValue;
            mod->BSIM3v0pagsGiven = TRUE;
            break;
        case BSIM3v0_MOD_PA1:
            mod->BSIM3v0pa1 = value->rValue;
            mod->BSIM3v0pa1Given = TRUE;
            break;
        case BSIM3v0_MOD_PA2:
            mod->BSIM3v0pa2 = value->rValue;
            mod->BSIM3v0pa2Given = TRUE;
            break;
        case BSIM3v0_MOD_PAT:
            mod->BSIM3v0pat = value->rValue;
            mod->BSIM3v0patGiven = TRUE;
            break;
        case BSIM3v0_MOD_PKETA:
            mod->BSIM3v0pketa = value->rValue;
            mod->BSIM3v0pketaGiven = TRUE;
            break;    
        case BSIM3v0_MOD_PNSUB:
            mod->BSIM3v0pnsub = value->rValue;
            mod->BSIM3v0pnsubGiven = TRUE;
            break;
        case BSIM3v0_MOD_PNPEAK:
            mod->BSIM3v0pnpeak = value->rValue;
            mod->BSIM3v0pnpeakGiven = TRUE;
	    if (mod->BSIM3v0pnpeak > 1.0e20)
		mod->BSIM3v0pnpeak *= 1.0e-6;
            break;
        case BSIM3v0_MOD_PNGATE:
            mod->BSIM3v0pngate = value->rValue;
            mod->BSIM3v0pngateGiven = TRUE;
	    if (mod->BSIM3v0pngate > 1.0e23)
		mod->BSIM3v0pngate *= 1.0e-6;
            break;
        case BSIM3v0_MOD_PGAMMA1:
            mod->BSIM3v0pgamma1 = value->rValue;
            mod->BSIM3v0pgamma1Given = TRUE;
            break;
        case BSIM3v0_MOD_PGAMMA2:
            mod->BSIM3v0pgamma2 = value->rValue;
            mod->BSIM3v0pgamma2Given = TRUE;
            break;
        case BSIM3v0_MOD_PVBX:
            mod->BSIM3v0pvbx = value->rValue;
            mod->BSIM3v0pvbxGiven = TRUE;
            break;
        case BSIM3v0_MOD_PVBM:
            mod->BSIM3v0pvbm = value->rValue;
            mod->BSIM3v0pvbmGiven = TRUE;
            break;
        case BSIM3v0_MOD_PXT:
            mod->BSIM3v0pxt = value->rValue;
            mod->BSIM3v0pxtGiven = TRUE;
            break;
        case  BSIM3v0_MOD_PK1:
            mod->BSIM3v0pk1 = value->rValue;
            mod->BSIM3v0pk1Given = TRUE;
            break;
        case  BSIM3v0_MOD_PKT1:
            mod->BSIM3v0pkt1 = value->rValue;
            mod->BSIM3v0pkt1Given = TRUE;
            break;
        case  BSIM3v0_MOD_PKT1L:
            mod->BSIM3v0pkt1l = value->rValue;
            mod->BSIM3v0pkt1lGiven = TRUE;
            break;
        case  BSIM3v0_MOD_PKT2:
            mod->BSIM3v0pkt2 = value->rValue;
            mod->BSIM3v0pkt2Given = TRUE;
            break;
        case  BSIM3v0_MOD_PK2:
            mod->BSIM3v0pk2 = value->rValue;
            mod->BSIM3v0pk2Given = TRUE;
            break;
        case  BSIM3v0_MOD_PK3:
            mod->BSIM3v0pk3 = value->rValue;
            mod->BSIM3v0pk3Given = TRUE;
            break;
        case  BSIM3v0_MOD_PK3B:
            mod->BSIM3v0pk3b = value->rValue;
            mod->BSIM3v0pk3bGiven = TRUE;
            break;
        case  BSIM3v0_MOD_PNLX:
            mod->BSIM3v0pnlx = value->rValue;
            mod->BSIM3v0pnlxGiven = TRUE;
            break;
        case  BSIM3v0_MOD_PW0:
            mod->BSIM3v0pw0 = value->rValue;
            mod->BSIM3v0pw0Given = TRUE;
            break;
        case  BSIM3v0_MOD_PDVT0:               
            mod->BSIM3v0pdvt0 = value->rValue;
            mod->BSIM3v0pdvt0Given = TRUE;
            break;
        case  BSIM3v0_MOD_PDVT1:             
            mod->BSIM3v0pdvt1 = value->rValue;
            mod->BSIM3v0pdvt1Given = TRUE;
            break;
        case  BSIM3v0_MOD_PDVT2:             
            mod->BSIM3v0pdvt2 = value->rValue;
            mod->BSIM3v0pdvt2Given = TRUE;
            break;
        case  BSIM3v0_MOD_PDVT0W:               
            mod->BSIM3v0pdvt0w = value->rValue;
            mod->BSIM3v0pdvt0wGiven = TRUE;
            break;
        case  BSIM3v0_MOD_PDVT1W:             
            mod->BSIM3v0pdvt1w = value->rValue;
            mod->BSIM3v0pdvt1wGiven = TRUE;
            break;
        case  BSIM3v0_MOD_PDVT2W:             
            mod->BSIM3v0pdvt2w = value->rValue;
            mod->BSIM3v0pdvt2wGiven = TRUE;
            break;
        case  BSIM3v0_MOD_PDROUT:             
            mod->BSIM3v0pdrout = value->rValue;
            mod->BSIM3v0pdroutGiven = TRUE;
            break;
        case  BSIM3v0_MOD_PDSUB:             
            mod->BSIM3v0pdsub = value->rValue;
            mod->BSIM3v0pdsubGiven = TRUE;
            break;
        case BSIM3v0_MOD_PVTH0:
            mod->BSIM3v0pvth0 = value->rValue;
            mod->BSIM3v0pvth0Given = TRUE;
            break;
        case BSIM3v0_MOD_PUA:
            mod->BSIM3v0pua = value->rValue;
            mod->BSIM3v0puaGiven = TRUE;
            break;
        case BSIM3v0_MOD_PUA1:
            mod->BSIM3v0pua1 = value->rValue;
            mod->BSIM3v0pua1Given = TRUE;
            break;
        case BSIM3v0_MOD_PUB:
            mod->BSIM3v0pub = value->rValue;
            mod->BSIM3v0pubGiven = TRUE;
            break;
        case BSIM3v0_MOD_PUB1:
            mod->BSIM3v0pub1 = value->rValue;
            mod->BSIM3v0pub1Given = TRUE;
            break;
        case BSIM3v0_MOD_PUC:
            mod->BSIM3v0puc = value->rValue;
            mod->BSIM3v0pucGiven = TRUE;
            break;
        case BSIM3v0_MOD_PUC1:
            mod->BSIM3v0puc1 = value->rValue;
            mod->BSIM3v0puc1Given = TRUE;
            break;
        case  BSIM3v0_MOD_PU0 :
            mod->BSIM3v0pu0 = value->rValue;
            mod->BSIM3v0pu0Given = TRUE;
	    if (mod->BSIM3v0pu0 > 1.0)
		mod->BSIM3v0pu0 *= 1.0e-4;
            break;
        case  BSIM3v0_MOD_PUTE :
            mod->BSIM3v0pute = value->rValue;
            mod->BSIM3v0puteGiven = TRUE;
            break;
        case BSIM3v0_MOD_PVOFF:
            mod->BSIM3v0pvoff = value->rValue;
            mod->BSIM3v0pvoffGiven = TRUE;
            break;
        case  BSIM3v0_MOD_PDELTA :
            mod->BSIM3v0pdelta = value->rValue;
            mod->BSIM3v0pdeltaGiven = TRUE;
            break;
        case BSIM3v0_MOD_PRDSW:
            mod->BSIM3v0prdsw = value->rValue;
            mod->BSIM3v0prdswGiven = TRUE;
            break;                     
        case BSIM3v0_MOD_PPRWB:
            mod->BSIM3v0pprwb = value->rValue;
            mod->BSIM3v0pprwbGiven = TRUE;
            break;                     
        case BSIM3v0_MOD_PPRWG:
            mod->BSIM3v0pprwg = value->rValue;
            mod->BSIM3v0pprwgGiven = TRUE;
            break;                     
        case BSIM3v0_MOD_PPRT:
            mod->BSIM3v0pprt = value->rValue;
            mod->BSIM3v0pprtGiven = TRUE;
            break;                     
        case BSIM3v0_MOD_PETA0:
            mod->BSIM3v0peta0 = value->rValue;
            mod->BSIM3v0peta0Given = TRUE;
            break;                 
        case BSIM3v0_MOD_PETAB:
            mod->BSIM3v0petab = value->rValue;
            mod->BSIM3v0petabGiven = TRUE;
            break;                 
        case BSIM3v0_MOD_PPCLM:
            mod->BSIM3v0ppclm = value->rValue;
            mod->BSIM3v0ppclmGiven = TRUE;
            break;                 
        case BSIM3v0_MOD_PPDIBL1:
            mod->BSIM3v0ppdibl1 = value->rValue;
            mod->BSIM3v0ppdibl1Given = TRUE;
            break;                 
        case BSIM3v0_MOD_PPDIBL2:
            mod->BSIM3v0ppdibl2 = value->rValue;
            mod->BSIM3v0ppdibl2Given = TRUE;
            break;                 
        case BSIM3v0_MOD_PPDIBLB:
            mod->BSIM3v0ppdiblb = value->rValue;
            mod->BSIM3v0ppdiblbGiven = TRUE;
            break;                 
        case BSIM3v0_MOD_PPSCBE1:
            mod->BSIM3v0ppscbe1 = value->rValue;
            mod->BSIM3v0ppscbe1Given = TRUE;
            break;                 
        case BSIM3v0_MOD_PPSCBE2:
            mod->BSIM3v0ppscbe2 = value->rValue;
            mod->BSIM3v0ppscbe2Given = TRUE;
            break;                 
        case BSIM3v0_MOD_PPVAG:
            mod->BSIM3v0ppvag = value->rValue;
            mod->BSIM3v0ppvagGiven = TRUE;
            break;                 
        case  BSIM3v0_MOD_PWR :
            mod->BSIM3v0pwr = value->rValue;
            mod->BSIM3v0pwrGiven = TRUE;
            break;
        case  BSIM3v0_MOD_PDWG :
            mod->BSIM3v0pdwg = value->rValue;
            mod->BSIM3v0pdwgGiven = TRUE;
            break;
        case  BSIM3v0_MOD_PDWB :
            mod->BSIM3v0pdwb = value->rValue;
            mod->BSIM3v0pdwbGiven = TRUE;
            break;
        case  BSIM3v0_MOD_PB0 :
            mod->BSIM3v0pb0 = value->rValue;
            mod->BSIM3v0pb0Given = TRUE;
            break;
        case  BSIM3v0_MOD_PB1 :
            mod->BSIM3v0pb1 = value->rValue;
            mod->BSIM3v0pb1Given = TRUE;
            break;
        case  BSIM3v0_MOD_PALPHA0 :
            mod->BSIM3v0palpha0 = value->rValue;
            mod->BSIM3v0palpha0Given = TRUE;
            break;
        case  BSIM3v0_MOD_PBETA0 :
            mod->BSIM3v0pbeta0 = value->rValue;
            mod->BSIM3v0pbeta0Given = TRUE;
            break;

        case  BSIM3v0_MOD_PELM :
            mod->BSIM3v0pelm = value->rValue;
            mod->BSIM3v0pelmGiven = TRUE;
            break;
        case  BSIM3v0_MOD_PCGSL :
            mod->BSIM3v0pcgsl = value->rValue;
            mod->BSIM3v0pcgslGiven = TRUE;
            break;
        case  BSIM3v0_MOD_PCGDL :
            mod->BSIM3v0pcgdl = value->rValue;
            mod->BSIM3v0pcgdlGiven = TRUE;
            break;
        case  BSIM3v0_MOD_PCKAPPA :
            mod->BSIM3v0pckappa = value->rValue;
            mod->BSIM3v0pckappaGiven = TRUE;
            break;
        case  BSIM3v0_MOD_PCF :
            mod->BSIM3v0pcf = value->rValue;
            mod->BSIM3v0pcfGiven = TRUE;
            break;
        case  BSIM3v0_MOD_PCLC :
            mod->BSIM3v0pclc = value->rValue;
            mod->BSIM3v0pclcGiven = TRUE;
            break;
        case  BSIM3v0_MOD_PCLE :
            mod->BSIM3v0pcle = value->rValue;
            mod->BSIM3v0pcleGiven = TRUE;
            break;

        case  BSIM3v0_MOD_TNOM :
            mod->BSIM3v0tnom = value->rValue + 273.15;
            mod->BSIM3v0tnomGiven = TRUE;
            break;
        case  BSIM3v0_MOD_CGSO :
            mod->BSIM3v0cgso = value->rValue;
            mod->BSIM3v0cgsoGiven = TRUE;
            break;
        case  BSIM3v0_MOD_CGDO :
            mod->BSIM3v0cgdo = value->rValue;
            mod->BSIM3v0cgdoGiven = TRUE;
            break;
        case  BSIM3v0_MOD_CGBO :
            mod->BSIM3v0cgbo = value->rValue;
            mod->BSIM3v0cgboGiven = TRUE;
            break;
        case  BSIM3v0_MOD_XPART :
            mod->BSIM3v0xpart = value->rValue;
            mod->BSIM3v0xpartGiven = TRUE;
            break;
        case  BSIM3v0_MOD_RSH :
            mod->BSIM3v0sheetResistance = value->rValue;
            mod->BSIM3v0sheetResistanceGiven = TRUE;
            break;
        case  BSIM3v0_MOD_JS :
            mod->BSIM3v0jctSatCurDensity = value->rValue;
            mod->BSIM3v0jctSatCurDensityGiven = TRUE;
            break;
        case  BSIM3v0_MOD_PB :
            mod->BSIM3v0bulkJctPotential = value->rValue;
            mod->BSIM3v0bulkJctPotentialGiven = TRUE;
            break;
        case  BSIM3v0_MOD_MJ :
            mod->BSIM3v0bulkJctBotGradingCoeff = value->rValue;
            mod->BSIM3v0bulkJctBotGradingCoeffGiven = TRUE;
            break;
        case  BSIM3v0_MOD_PBSW :
            mod->BSIM3v0sidewallJctPotential = value->rValue;
            mod->BSIM3v0sidewallJctPotentialGiven = TRUE;
            break;
        case  BSIM3v0_MOD_MJSW :
            mod->BSIM3v0bulkJctSideGradingCoeff = value->rValue;
            mod->BSIM3v0bulkJctSideGradingCoeffGiven = TRUE;
            break;
        case  BSIM3v0_MOD_CJ :
            mod->BSIM3v0unitAreaJctCap = value->rValue;
            mod->BSIM3v0unitAreaJctCapGiven = TRUE;
            break;
        case  BSIM3v0_MOD_CJSW :
            mod->BSIM3v0unitLengthSidewallJctCap = value->rValue;
            mod->BSIM3v0unitLengthSidewallJctCapGiven = TRUE;
            break;
        case  BSIM3v0_MOD_LINT :
            mod->BSIM3v0Lint = value->rValue;
            mod->BSIM3v0LintGiven = TRUE;
            break;
        case  BSIM3v0_MOD_LL :
            mod->BSIM3v0Ll = value->rValue;
            mod->BSIM3v0LlGiven = TRUE;
            break;
        case  BSIM3v0_MOD_LLN :
            mod->BSIM3v0Lln = value->rValue;
            mod->BSIM3v0LlnGiven = TRUE;
            break;
        case  BSIM3v0_MOD_LW :
            mod->BSIM3v0Lw = value->rValue;
            mod->BSIM3v0LwGiven = TRUE;
            break;
        case  BSIM3v0_MOD_LWN :
            mod->BSIM3v0Lwn = value->rValue;
            mod->BSIM3v0LwnGiven = TRUE;
            break;
        case  BSIM3v0_MOD_LWL :
            mod->BSIM3v0Lwl = value->rValue;
            mod->BSIM3v0LwlGiven = TRUE;
            break;
        case  BSIM3v0_MOD_LMIN :
            mod->BSIM3v0Lmin = value->rValue;
            mod->BSIM3v0LminGiven = TRUE;
            break;
        case  BSIM3v0_MOD_LMAX :
            mod->BSIM3v0Lmax = value->rValue;
            mod->BSIM3v0LmaxGiven = TRUE;
            break;
        case  BSIM3v0_MOD_WINT :
            mod->BSIM3v0Wint = value->rValue;
            mod->BSIM3v0WintGiven = TRUE;
            break;
        case  BSIM3v0_MOD_WL :
            mod->BSIM3v0Wl = value->rValue;
            mod->BSIM3v0WlGiven = TRUE;
            break;
        case  BSIM3v0_MOD_WLN :
            mod->BSIM3v0Wln = value->rValue;
            mod->BSIM3v0WlnGiven = TRUE;
            break;
        case  BSIM3v0_MOD_WW :
            mod->BSIM3v0Ww = value->rValue;
            mod->BSIM3v0WwGiven = TRUE;
            break;
        case  BSIM3v0_MOD_WWN :
            mod->BSIM3v0Wwn = value->rValue;
            mod->BSIM3v0WwnGiven = TRUE;
            break;
        case  BSIM3v0_MOD_WWL :
            mod->BSIM3v0Wwl = value->rValue;
            mod->BSIM3v0WwlGiven = TRUE;
            break;
        case  BSIM3v0_MOD_WMIN :
            mod->BSIM3v0Wmin = value->rValue;
            mod->BSIM3v0WminGiven = TRUE;
            break;
        case  BSIM3v0_MOD_WMAX :
            mod->BSIM3v0Wmax = value->rValue;
            mod->BSIM3v0WmaxGiven = TRUE;
            break;

        case  BSIM3v0_MOD_NOIA :
            mod->BSIM3v0oxideTrapDensityA = value->rValue;
            mod->BSIM3v0oxideTrapDensityAGiven = TRUE;
            break;
        case  BSIM3v0_MOD_NOIB :
            mod->BSIM3v0oxideTrapDensityB = value->rValue;
            mod->BSIM3v0oxideTrapDensityBGiven = TRUE;
            break;
        case  BSIM3v0_MOD_NOIC :
            mod->BSIM3v0oxideTrapDensityC = value->rValue;
            mod->BSIM3v0oxideTrapDensityCGiven = TRUE;
            break;
        case  BSIM3v0_MOD_EM :
            mod->BSIM3v0em = value->rValue;
            mod->BSIM3v0emGiven = TRUE;
            break;
        case  BSIM3v0_MOD_EF :
            mod->BSIM3v0ef = value->rValue;
            mod->BSIM3v0efGiven = TRUE;
            break;
        case  BSIM3v0_MOD_AF :
            mod->BSIM3v0af = value->rValue;
            mod->BSIM3v0afGiven = TRUE;
            break;
        case  BSIM3v0_MOD_KF :
            mod->BSIM3v0kf = value->rValue;
            mod->BSIM3v0kfGiven = TRUE;
            break;
        case  BSIM3v0_MOD_NMOS  :
            if(value->iValue) {
                mod->BSIM3v0type = 1;
                mod->BSIM3v0typeGiven = TRUE;
            }
            break;
        case  BSIM3v0_MOD_PMOS  :
            if(value->iValue) {
                mod->BSIM3v0type = - 1;
                mod->BSIM3v0typeGiven = TRUE;
            }
            break;
        default:
            return(E_BADPARM);
    }
    return(OK);
}


