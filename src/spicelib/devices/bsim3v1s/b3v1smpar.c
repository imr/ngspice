/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1995 Min-Chie Jeng and Mansun Chan.
Modified by Paolo Nenzi 2002
File: b3v1smpar.c
**********/

#include "ngspice.h"
#include "bsim3v1sdef.h"
#include "ifsim.h"
#include "sperror.h"
#include "suffix.h"


int
BSIM3v1SmParam(int param, IFvalue *value, GENmodel *inMod)
{
    BSIM3v1Smodel *mod = (BSIM3v1Smodel*)inMod;
    switch(param)
    {   case  BSIM3v1S_MOD_MOBMOD :
            mod->BSIM3v1SmobMod = value->iValue;
            mod->BSIM3v1SmobModGiven = TRUE;
            break;
        case  BSIM3v1S_MOD_BINUNIT :
            mod->BSIM3v1SbinUnit = value->iValue;
            mod->BSIM3v1SbinUnitGiven = TRUE;
            break;
        case  BSIM3v1S_MOD_PARAMCHK :
            mod->BSIM3v1SparamChk = value->iValue;
            mod->BSIM3v1SparamChkGiven = TRUE;
            break;
        case  BSIM3v1S_MOD_CAPMOD :
            mod->BSIM3v1ScapMod = value->iValue;
            mod->BSIM3v1ScapModGiven = TRUE;
            break;
        case  BSIM3v1S_MOD_NQSMOD :
            mod->BSIM3v1SnqsMod = value->iValue;
            mod->BSIM3v1SnqsModGiven = TRUE;
            break;
        case  BSIM3v1S_MOD_NOIMOD :
            mod->BSIM3v1SnoiMod = value->iValue;
            mod->BSIM3v1SnoiModGiven = TRUE;
            break;
        case  BSIM3v1S_MOD_VERSION :
            mod->BSIM3v1Sversion = value->rValue;
            mod->BSIM3v1SversionGiven = TRUE;
            break;
        case  BSIM3v1S_MOD_TOX :
            mod->BSIM3v1Stox = value->rValue;
            mod->BSIM3v1StoxGiven = TRUE;
            break;

        case  BSIM3v1S_MOD_CDSC :
            mod->BSIM3v1Scdsc = value->rValue;
            mod->BSIM3v1ScdscGiven = TRUE;
            break;
        case  BSIM3v1S_MOD_CDSCB :
            mod->BSIM3v1Scdscb = value->rValue;
            mod->BSIM3v1ScdscbGiven = TRUE;
            break;

        case  BSIM3v1S_MOD_CDSCD :
            mod->BSIM3v1Scdscd = value->rValue;
            mod->BSIM3v1ScdscdGiven = TRUE;
            break;

        case  BSIM3v1S_MOD_CIT :
            mod->BSIM3v1Scit = value->rValue;
            mod->BSIM3v1ScitGiven = TRUE;
            break;
        case  BSIM3v1S_MOD_NFACTOR :
            mod->BSIM3v1Snfactor = value->rValue;
            mod->BSIM3v1SnfactorGiven = TRUE;
            break;
        case BSIM3v1S_MOD_XJ:
            mod->BSIM3v1Sxj = value->rValue;
            mod->BSIM3v1SxjGiven = TRUE;
            break;
        case BSIM3v1S_MOD_VSAT:
            mod->BSIM3v1Svsat = value->rValue;
            mod->BSIM3v1SvsatGiven = TRUE;
            break;
        case BSIM3v1S_MOD_A0:
            mod->BSIM3v1Sa0 = value->rValue;
            mod->BSIM3v1Sa0Given = TRUE;
            break;
        
        case BSIM3v1S_MOD_AGS:
            mod->BSIM3v1Sags= value->rValue;
            mod->BSIM3v1SagsGiven = TRUE;
            break;
        
        case BSIM3v1S_MOD_A1:
            mod->BSIM3v1Sa1 = value->rValue;
            mod->BSIM3v1Sa1Given = TRUE;
            break;
        case BSIM3v1S_MOD_A2:
            mod->BSIM3v1Sa2 = value->rValue;
            mod->BSIM3v1Sa2Given = TRUE;
            break;
        case BSIM3v1S_MOD_AT:
            mod->BSIM3v1Sat = value->rValue;
            mod->BSIM3v1SatGiven = TRUE;
            break;
        case BSIM3v1S_MOD_KETA:
            mod->BSIM3v1Sketa = value->rValue;
            mod->BSIM3v1SketaGiven = TRUE;
            break;    
        case BSIM3v1S_MOD_NSUB:
            mod->BSIM3v1Snsub = value->rValue;
            mod->BSIM3v1SnsubGiven = TRUE;
            break;
        case BSIM3v1S_MOD_NPEAK:
            mod->BSIM3v1Snpeak = value->rValue;
            mod->BSIM3v1SnpeakGiven = TRUE;
	    if (mod->BSIM3v1Snpeak > 1.0e20)
		mod->BSIM3v1Snpeak *= 1.0e-6;
            break;
        case BSIM3v1S_MOD_NGATE:
            mod->BSIM3v1Sngate = value->rValue;
            mod->BSIM3v1SngateGiven = TRUE;
	    if (mod->BSIM3v1Sngate > 1.0e23)
		mod->BSIM3v1Sngate *= 1.0e-6;
            break;
        case BSIM3v1S_MOD_GAMMA1:
            mod->BSIM3v1Sgamma1 = value->rValue;
            mod->BSIM3v1Sgamma1Given = TRUE;
            break;
        case BSIM3v1S_MOD_GAMMA2:
            mod->BSIM3v1Sgamma2 = value->rValue;
            mod->BSIM3v1Sgamma2Given = TRUE;
            break;
        case BSIM3v1S_MOD_VBX:
            mod->BSIM3v1Svbx = value->rValue;
            mod->BSIM3v1SvbxGiven = TRUE;
            break;
        case BSIM3v1S_MOD_VBM:
            mod->BSIM3v1Svbm = value->rValue;
            mod->BSIM3v1SvbmGiven = TRUE;
            break;
        case BSIM3v1S_MOD_XT:
            mod->BSIM3v1Sxt = value->rValue;
            mod->BSIM3v1SxtGiven = TRUE;
            break;
        case  BSIM3v1S_MOD_K1:
            mod->BSIM3v1Sk1 = value->rValue;
            mod->BSIM3v1Sk1Given = TRUE;
            break;
        case  BSIM3v1S_MOD_KT1:
            mod->BSIM3v1Skt1 = value->rValue;
            mod->BSIM3v1Skt1Given = TRUE;
            break;
        case  BSIM3v1S_MOD_KT1L:
            mod->BSIM3v1Skt1l = value->rValue;
            mod->BSIM3v1Skt1lGiven = TRUE;
            break;
        case  BSIM3v1S_MOD_KT2:
            mod->BSIM3v1Skt2 = value->rValue;
            mod->BSIM3v1Skt2Given = TRUE;
            break;
        case  BSIM3v1S_MOD_K2:
            mod->BSIM3v1Sk2 = value->rValue;
            mod->BSIM3v1Sk2Given = TRUE;
            break;
        case  BSIM3v1S_MOD_K3:
            mod->BSIM3v1Sk3 = value->rValue;
            mod->BSIM3v1Sk3Given = TRUE;
            break;
        case  BSIM3v1S_MOD_K3B:
            mod->BSIM3v1Sk3b = value->rValue;
            mod->BSIM3v1Sk3bGiven = TRUE;
            break;
        case  BSIM3v1S_MOD_NLX:
            mod->BSIM3v1Snlx = value->rValue;
            mod->BSIM3v1SnlxGiven = TRUE;
            break;
        case  BSIM3v1S_MOD_W0:
            mod->BSIM3v1Sw0 = value->rValue;
            mod->BSIM3v1Sw0Given = TRUE;
            break;
        case  BSIM3v1S_MOD_DVT0:               
            mod->BSIM3v1Sdvt0 = value->rValue;
            mod->BSIM3v1Sdvt0Given = TRUE;
            break;
        case  BSIM3v1S_MOD_DVT1:             
            mod->BSIM3v1Sdvt1 = value->rValue;
            mod->BSIM3v1Sdvt1Given = TRUE;
            break;
        case  BSIM3v1S_MOD_DVT2:             
            mod->BSIM3v1Sdvt2 = value->rValue;
            mod->BSIM3v1Sdvt2Given = TRUE;
            break;
        case  BSIM3v1S_MOD_DVT0W:               
            mod->BSIM3v1Sdvt0w = value->rValue;
            mod->BSIM3v1Sdvt0wGiven = TRUE;
            break;
        case  BSIM3v1S_MOD_DVT1W:             
            mod->BSIM3v1Sdvt1w = value->rValue;
            mod->BSIM3v1Sdvt1wGiven = TRUE;
            break;
        case  BSIM3v1S_MOD_DVT2W:             
            mod->BSIM3v1Sdvt2w = value->rValue;
            mod->BSIM3v1Sdvt2wGiven = TRUE;
            break;
        case  BSIM3v1S_MOD_DROUT:             
            mod->BSIM3v1Sdrout = value->rValue;
            mod->BSIM3v1SdroutGiven = TRUE;
            break;
        case  BSIM3v1S_MOD_DSUB:             
            mod->BSIM3v1Sdsub = value->rValue;
            mod->BSIM3v1SdsubGiven = TRUE;
            break;
        case BSIM3v1S_MOD_VTH0:
            mod->BSIM3v1Svth0 = value->rValue;
            mod->BSIM3v1Svth0Given = TRUE;
            break;
        case BSIM3v1S_MOD_UA:
            mod->BSIM3v1Sua = value->rValue;
            mod->BSIM3v1SuaGiven = TRUE;
            break;
        case BSIM3v1S_MOD_UA1:
            mod->BSIM3v1Sua1 = value->rValue;
            mod->BSIM3v1Sua1Given = TRUE;
            break;
        case BSIM3v1S_MOD_UB:
            mod->BSIM3v1Sub = value->rValue;
            mod->BSIM3v1SubGiven = TRUE;
            break;
        case BSIM3v1S_MOD_UB1:
            mod->BSIM3v1Sub1 = value->rValue;
            mod->BSIM3v1Sub1Given = TRUE;
            break;
        case BSIM3v1S_MOD_UC:
            mod->BSIM3v1Suc = value->rValue;
            mod->BSIM3v1SucGiven = TRUE;
            break;
        case BSIM3v1S_MOD_UC1:
            mod->BSIM3v1Suc1 = value->rValue;
            mod->BSIM3v1Suc1Given = TRUE;
            break;
        case  BSIM3v1S_MOD_U0 :
            mod->BSIM3v1Su0 = value->rValue;
            mod->BSIM3v1Su0Given = TRUE;
            break;
        case  BSIM3v1S_MOD_UTE :
            mod->BSIM3v1Sute = value->rValue;
            mod->BSIM3v1SuteGiven = TRUE;
            break;
        case BSIM3v1S_MOD_VOFF:
            mod->BSIM3v1Svoff = value->rValue;
            mod->BSIM3v1SvoffGiven = TRUE;
            break;
        case  BSIM3v1S_MOD_DELTA :
            mod->BSIM3v1Sdelta = value->rValue;
            mod->BSIM3v1SdeltaGiven = TRUE;
            break;
        case BSIM3v1S_MOD_RDSW:
            mod->BSIM3v1Srdsw = value->rValue;
            mod->BSIM3v1SrdswGiven = TRUE;
            break;                     
        case BSIM3v1S_MOD_PRWG:
            mod->BSIM3v1Sprwg = value->rValue;
            mod->BSIM3v1SprwgGiven = TRUE;
            break;                     
        case BSIM3v1S_MOD_PRWB:
            mod->BSIM3v1Sprwb = value->rValue;
            mod->BSIM3v1SprwbGiven = TRUE;
            break;                     
        case BSIM3v1S_MOD_PRT:
            mod->BSIM3v1Sprt = value->rValue;
            mod->BSIM3v1SprtGiven = TRUE;
            break;                     
        case BSIM3v1S_MOD_ETA0:
            mod->BSIM3v1Seta0 = value->rValue;
            mod->BSIM3v1Seta0Given = TRUE;
            break;                 
        case BSIM3v1S_MOD_ETAB:
            mod->BSIM3v1Setab = value->rValue;
            mod->BSIM3v1SetabGiven = TRUE;
            break;                 
        case BSIM3v1S_MOD_PCLM:
            mod->BSIM3v1Spclm = value->rValue;
            mod->BSIM3v1SpclmGiven = TRUE;
            break;                 
        case BSIM3v1S_MOD_PDIBL1:
            mod->BSIM3v1Spdibl1 = value->rValue;
            mod->BSIM3v1Spdibl1Given = TRUE;
            break;                 
        case BSIM3v1S_MOD_PDIBL2:
            mod->BSIM3v1Spdibl2 = value->rValue;
            mod->BSIM3v1Spdibl2Given = TRUE;
            break;                 
        case BSIM3v1S_MOD_PDIBLB:
            mod->BSIM3v1Spdiblb = value->rValue;
            mod->BSIM3v1SpdiblbGiven = TRUE;
            break;                 
        case BSIM3v1S_MOD_PSCBE1:
            mod->BSIM3v1Spscbe1 = value->rValue;
            mod->BSIM3v1Spscbe1Given = TRUE;
            break;                 
        case BSIM3v1S_MOD_PSCBE2:
            mod->BSIM3v1Spscbe2 = value->rValue;
            mod->BSIM3v1Spscbe2Given = TRUE;
            break;                 
        case BSIM3v1S_MOD_PVAG:
            mod->BSIM3v1Spvag = value->rValue;
            mod->BSIM3v1SpvagGiven = TRUE;
            break;                 
        case  BSIM3v1S_MOD_WR :
            mod->BSIM3v1Swr = value->rValue;
            mod->BSIM3v1SwrGiven = TRUE;
            break;
        case  BSIM3v1S_MOD_DWG :
            mod->BSIM3v1Sdwg = value->rValue;
            mod->BSIM3v1SdwgGiven = TRUE;
            break;
        case  BSIM3v1S_MOD_DWB :
            mod->BSIM3v1Sdwb = value->rValue;
            mod->BSIM3v1SdwbGiven = TRUE;
            break;
        case  BSIM3v1S_MOD_B0 :
            mod->BSIM3v1Sb0 = value->rValue;
            mod->BSIM3v1Sb0Given = TRUE;
            break;
        case  BSIM3v1S_MOD_B1 :
            mod->BSIM3v1Sb1 = value->rValue;
            mod->BSIM3v1Sb1Given = TRUE;
            break;
        case  BSIM3v1S_MOD_ALPHA0 :
            mod->BSIM3v1Salpha0 = value->rValue;
            mod->BSIM3v1Salpha0Given = TRUE;
            break;
        case  BSIM3v1S_MOD_BETA0 :
            mod->BSIM3v1Sbeta0 = value->rValue;
            mod->BSIM3v1Sbeta0Given = TRUE;
            break;

        case  BSIM3v1S_MOD_ELM :
            mod->BSIM3v1Selm = value->rValue;
            mod->BSIM3v1SelmGiven = TRUE;
            break;
        case  BSIM3v1S_MOD_CGSL :
            mod->BSIM3v1Scgsl = value->rValue;
            mod->BSIM3v1ScgslGiven = TRUE;
            break;
        case  BSIM3v1S_MOD_CGDL :
            mod->BSIM3v1Scgdl = value->rValue;
            mod->BSIM3v1ScgdlGiven = TRUE;
            break;
        case  BSIM3v1S_MOD_CKAPPA :
            mod->BSIM3v1Sckappa = value->rValue;
            mod->BSIM3v1SckappaGiven = TRUE;
            break;
        case  BSIM3v1S_MOD_CF :
            mod->BSIM3v1Scf = value->rValue;
            mod->BSIM3v1ScfGiven = TRUE;
            break;
        case  BSIM3v1S_MOD_CLC :
            mod->BSIM3v1Sclc = value->rValue;
            mod->BSIM3v1SclcGiven = TRUE;
            break;
        case  BSIM3v1S_MOD_CLE :
            mod->BSIM3v1Scle = value->rValue;
            mod->BSIM3v1ScleGiven = TRUE;
            break;
        case  BSIM3v1S_MOD_DWC :
            mod->BSIM3v1Sdwc = value->rValue;
            mod->BSIM3v1SdwcGiven = TRUE;
            break;
        case  BSIM3v1S_MOD_DLC :
            mod->BSIM3v1Sdlc = value->rValue;
            mod->BSIM3v1SdlcGiven = TRUE;
            break;
        case  BSIM3v1S_MOD_VFBCV :
            mod->BSIM3v1Svfbcv = value->rValue;
            mod->BSIM3v1SvfbcvGiven = TRUE;
            break;

	/* Length dependence */
        case  BSIM3v1S_MOD_LCDSC :
            mod->BSIM3v1Slcdsc = value->rValue;
            mod->BSIM3v1SlcdscGiven = TRUE;
            break;


        case  BSIM3v1S_MOD_LCDSCB :
            mod->BSIM3v1Slcdscb = value->rValue;
            mod->BSIM3v1SlcdscbGiven = TRUE;
            break;
        case  BSIM3v1S_MOD_LCDSCD :
            mod->BSIM3v1Slcdscd = value->rValue;
            mod->BSIM3v1SlcdscdGiven = TRUE;
            break;
        case  BSIM3v1S_MOD_LCIT :
            mod->BSIM3v1Slcit = value->rValue;
            mod->BSIM3v1SlcitGiven = TRUE;
            break;
        case  BSIM3v1S_MOD_LNFACTOR :
            mod->BSIM3v1Slnfactor = value->rValue;
            mod->BSIM3v1SlnfactorGiven = TRUE;
            break;
        case BSIM3v1S_MOD_LXJ:
            mod->BSIM3v1Slxj = value->rValue;
            mod->BSIM3v1SlxjGiven = TRUE;
            break;
        case BSIM3v1S_MOD_LVSAT:
            mod->BSIM3v1Slvsat = value->rValue;
            mod->BSIM3v1SlvsatGiven = TRUE;
            break;
        
        
        case BSIM3v1S_MOD_LA0:
            mod->BSIM3v1Sla0 = value->rValue;
            mod->BSIM3v1Sla0Given = TRUE;
            break;
        case BSIM3v1S_MOD_LAGS:
            mod->BSIM3v1Slags = value->rValue;
            mod->BSIM3v1SlagsGiven = TRUE;
            break;
        case BSIM3v1S_MOD_LA1:
            mod->BSIM3v1Sla1 = value->rValue;
            mod->BSIM3v1Sla1Given = TRUE;
            break;
        case BSIM3v1S_MOD_LA2:
            mod->BSIM3v1Sla2 = value->rValue;
            mod->BSIM3v1Sla2Given = TRUE;
            break;
        case BSIM3v1S_MOD_LAT:
            mod->BSIM3v1Slat = value->rValue;
            mod->BSIM3v1SlatGiven = TRUE;
            break;
        case BSIM3v1S_MOD_LKETA:
            mod->BSIM3v1Slketa = value->rValue;
            mod->BSIM3v1SlketaGiven = TRUE;
            break;    
        case BSIM3v1S_MOD_LNSUB:
            mod->BSIM3v1Slnsub = value->rValue;
            mod->BSIM3v1SlnsubGiven = TRUE;
            break;
        case BSIM3v1S_MOD_LNPEAK:
            mod->BSIM3v1Slnpeak = value->rValue;
            mod->BSIM3v1SlnpeakGiven = TRUE;
	    if (mod->BSIM3v1Slnpeak > 1.0e20)
		mod->BSIM3v1Slnpeak *= 1.0e-6;
            break;
        case BSIM3v1S_MOD_LNGATE:
            mod->BSIM3v1Slngate = value->rValue;
            mod->BSIM3v1SlngateGiven = TRUE;
	    if (mod->BSIM3v1Slngate > 1.0e23)
		mod->BSIM3v1Slngate *= 1.0e-6;
            break;
        case BSIM3v1S_MOD_LGAMMA1:
            mod->BSIM3v1Slgamma1 = value->rValue;
            mod->BSIM3v1Slgamma1Given = TRUE;
            break;
        case BSIM3v1S_MOD_LGAMMA2:
            mod->BSIM3v1Slgamma2 = value->rValue;
            mod->BSIM3v1Slgamma2Given = TRUE;
            break;
        case BSIM3v1S_MOD_LVBX:
            mod->BSIM3v1Slvbx = value->rValue;
            mod->BSIM3v1SlvbxGiven = TRUE;
            break;
        case BSIM3v1S_MOD_LVBM:
            mod->BSIM3v1Slvbm = value->rValue;
            mod->BSIM3v1SlvbmGiven = TRUE;
            break;
        case BSIM3v1S_MOD_LXT:
            mod->BSIM3v1Slxt = value->rValue;
            mod->BSIM3v1SlxtGiven = TRUE;
            break;
        case  BSIM3v1S_MOD_LK1:
            mod->BSIM3v1Slk1 = value->rValue;
            mod->BSIM3v1Slk1Given = TRUE;
            break;
        case  BSIM3v1S_MOD_LKT1:
            mod->BSIM3v1Slkt1 = value->rValue;
            mod->BSIM3v1Slkt1Given = TRUE;
            break;
        case  BSIM3v1S_MOD_LKT1L:
            mod->BSIM3v1Slkt1l = value->rValue;
            mod->BSIM3v1Slkt1lGiven = TRUE;
            break;
        case  BSIM3v1S_MOD_LKT2:
            mod->BSIM3v1Slkt2 = value->rValue;
            mod->BSIM3v1Slkt2Given = TRUE;
            break;
        case  BSIM3v1S_MOD_LK2:
            mod->BSIM3v1Slk2 = value->rValue;
            mod->BSIM3v1Slk2Given = TRUE;
            break;
        case  BSIM3v1S_MOD_LK3:
            mod->BSIM3v1Slk3 = value->rValue;
            mod->BSIM3v1Slk3Given = TRUE;
            break;
        case  BSIM3v1S_MOD_LK3B:
            mod->BSIM3v1Slk3b = value->rValue;
            mod->BSIM3v1Slk3bGiven = TRUE;
            break;
        case  BSIM3v1S_MOD_LNLX:
            mod->BSIM3v1Slnlx = value->rValue;
            mod->BSIM3v1SlnlxGiven = TRUE;
            break;
        case  BSIM3v1S_MOD_LW0:
            mod->BSIM3v1Slw0 = value->rValue;
            mod->BSIM3v1Slw0Given = TRUE;
            break;
        case  BSIM3v1S_MOD_LDVT0:               
            mod->BSIM3v1Sldvt0 = value->rValue;
            mod->BSIM3v1Sldvt0Given = TRUE;
            break;
        case  BSIM3v1S_MOD_LDVT1:             
            mod->BSIM3v1Sldvt1 = value->rValue;
            mod->BSIM3v1Sldvt1Given = TRUE;
            break;
        case  BSIM3v1S_MOD_LDVT2:             
            mod->BSIM3v1Sldvt2 = value->rValue;
            mod->BSIM3v1Sldvt2Given = TRUE;
            break;
        case  BSIM3v1S_MOD_LDVT0W:               
            mod->BSIM3v1Sldvt0w = value->rValue;
            mod->BSIM3v1Sldvt0Given = TRUE;
            break;
        case  BSIM3v1S_MOD_LDVT1W:             
            mod->BSIM3v1Sldvt1w = value->rValue;
            mod->BSIM3v1Sldvt1wGiven = TRUE;
            break;
        case  BSIM3v1S_MOD_LDVT2W:             
            mod->BSIM3v1Sldvt2w = value->rValue;
            mod->BSIM3v1Sldvt2wGiven = TRUE;
            break;
        case  BSIM3v1S_MOD_LDROUT:             
            mod->BSIM3v1Sldrout = value->rValue;
            mod->BSIM3v1SldroutGiven = TRUE;
            break;
        case  BSIM3v1S_MOD_LDSUB:             
            mod->BSIM3v1Sldsub = value->rValue;
            mod->BSIM3v1SldsubGiven = TRUE;
            break;
        case BSIM3v1S_MOD_LVTH0:
            mod->BSIM3v1Slvth0 = value->rValue;
            mod->BSIM3v1Slvth0Given = TRUE;
            break;
        case BSIM3v1S_MOD_LUA:
            mod->BSIM3v1Slua = value->rValue;
            mod->BSIM3v1SluaGiven = TRUE;
            break;
        case BSIM3v1S_MOD_LUA1:
            mod->BSIM3v1Slua1 = value->rValue;
            mod->BSIM3v1Slua1Given = TRUE;
            break;
        case BSIM3v1S_MOD_LUB:
            mod->BSIM3v1Slub = value->rValue;
            mod->BSIM3v1SlubGiven = TRUE;
            break;
        case BSIM3v1S_MOD_LUB1:
            mod->BSIM3v1Slub1 = value->rValue;
            mod->BSIM3v1Slub1Given = TRUE;
            break;
        case BSIM3v1S_MOD_LUC:
            mod->BSIM3v1Sluc = value->rValue;
            mod->BSIM3v1SlucGiven = TRUE;
            break;
        case BSIM3v1S_MOD_LUC1:
            mod->BSIM3v1Sluc1 = value->rValue;
            mod->BSIM3v1Sluc1Given = TRUE;
            break;
        case  BSIM3v1S_MOD_LU0 :
            mod->BSIM3v1Slu0 = value->rValue;
            mod->BSIM3v1Slu0Given = TRUE;
            break;
        case  BSIM3v1S_MOD_LUTE :
            mod->BSIM3v1Slute = value->rValue;
            mod->BSIM3v1SluteGiven = TRUE;
            break;
        case BSIM3v1S_MOD_LVOFF:
            mod->BSIM3v1Slvoff = value->rValue;
            mod->BSIM3v1SlvoffGiven = TRUE;
            break;
        case  BSIM3v1S_MOD_LDELTA :
            mod->BSIM3v1Sldelta = value->rValue;
            mod->BSIM3v1SldeltaGiven = TRUE;
            break;
        case BSIM3v1S_MOD_LRDSW:
            mod->BSIM3v1Slrdsw = value->rValue;
            mod->BSIM3v1SlrdswGiven = TRUE;
            break;                     
        case BSIM3v1S_MOD_LPRWB:
            mod->BSIM3v1Slprwb = value->rValue;
            mod->BSIM3v1SlprwbGiven = TRUE;
            break;                     
        case BSIM3v1S_MOD_LPRWG:
            mod->BSIM3v1Slprwg = value->rValue;
            mod->BSIM3v1SlprwgGiven = TRUE;
            break;                     
        case BSIM3v1S_MOD_LPRT:
            mod->BSIM3v1Slprt = value->rValue;
            mod->BSIM3v1SlprtGiven = TRUE;
            break;                     
        case BSIM3v1S_MOD_LETA0:
            mod->BSIM3v1Sleta0 = value->rValue;
            mod->BSIM3v1Sleta0Given = TRUE;
            break;                 
        case BSIM3v1S_MOD_LETAB:
            mod->BSIM3v1Sletab = value->rValue;
            mod->BSIM3v1SletabGiven = TRUE;
            break;                 
        case BSIM3v1S_MOD_LPCLM:
            mod->BSIM3v1Slpclm = value->rValue;
            mod->BSIM3v1SlpclmGiven = TRUE;
            break;                 
        case BSIM3v1S_MOD_LPDIBL1:
            mod->BSIM3v1Slpdibl1 = value->rValue;
            mod->BSIM3v1Slpdibl1Given = TRUE;
            break;                 
        case BSIM3v1S_MOD_LPDIBL2:
            mod->BSIM3v1Slpdibl2 = value->rValue;
            mod->BSIM3v1Slpdibl2Given = TRUE;
            break;                 
        case BSIM3v1S_MOD_LPDIBLB:
            mod->BSIM3v1Slpdiblb = value->rValue;
            mod->BSIM3v1SlpdiblbGiven = TRUE;
            break;                 
        case BSIM3v1S_MOD_LPSCBE1:
            mod->BSIM3v1Slpscbe1 = value->rValue;
            mod->BSIM3v1Slpscbe1Given = TRUE;
            break;                 
        case BSIM3v1S_MOD_LPSCBE2:
            mod->BSIM3v1Slpscbe2 = value->rValue;
            mod->BSIM3v1Slpscbe2Given = TRUE;
            break;                 
        case BSIM3v1S_MOD_LPVAG:
            mod->BSIM3v1Slpvag = value->rValue;
            mod->BSIM3v1SlpvagGiven = TRUE;
            break;                 
        case  BSIM3v1S_MOD_LWR :
            mod->BSIM3v1Slwr = value->rValue;
            mod->BSIM3v1SlwrGiven = TRUE;
            break;
        case  BSIM3v1S_MOD_LDWG :
            mod->BSIM3v1Sldwg = value->rValue;
            mod->BSIM3v1SldwgGiven = TRUE;
            break;
        case  BSIM3v1S_MOD_LDWB :
            mod->BSIM3v1Sldwb = value->rValue;
            mod->BSIM3v1SldwbGiven = TRUE;
            break;
        case  BSIM3v1S_MOD_LB0 :
            mod->BSIM3v1Slb0 = value->rValue;
            mod->BSIM3v1Slb0Given = TRUE;
            break;
        case  BSIM3v1S_MOD_LB1 :
            mod->BSIM3v1Slb1 = value->rValue;
            mod->BSIM3v1Slb1Given = TRUE;
            break;
        case  BSIM3v1S_MOD_LALPHA0 :
            mod->BSIM3v1Slalpha0 = value->rValue;
            mod->BSIM3v1Slalpha0Given = TRUE;
            break;
        case  BSIM3v1S_MOD_LBETA0 :
            mod->BSIM3v1Slbeta0 = value->rValue;
            mod->BSIM3v1Slbeta0Given = TRUE;
            break;

        case  BSIM3v1S_MOD_LELM :
            mod->BSIM3v1Slelm = value->rValue;
            mod->BSIM3v1SlelmGiven = TRUE;
            break;
        case  BSIM3v1S_MOD_LCGSL :
            mod->BSIM3v1Slcgsl = value->rValue;
            mod->BSIM3v1SlcgslGiven = TRUE;
            break;
        case  BSIM3v1S_MOD_LCGDL :
            mod->BSIM3v1Slcgdl = value->rValue;
            mod->BSIM3v1SlcgdlGiven = TRUE;
            break;
        case  BSIM3v1S_MOD_LCKAPPA :
            mod->BSIM3v1Slckappa = value->rValue;
            mod->BSIM3v1SlckappaGiven = TRUE;
            break;
        case  BSIM3v1S_MOD_LCF :
            mod->BSIM3v1Slcf = value->rValue;
            mod->BSIM3v1SlcfGiven = TRUE;
            break;
        case  BSIM3v1S_MOD_LCLC :
            mod->BSIM3v1Slclc = value->rValue;
            mod->BSIM3v1SlclcGiven = TRUE;
            break;
        case  BSIM3v1S_MOD_LCLE :
            mod->BSIM3v1Slcle = value->rValue;
            mod->BSIM3v1SlcleGiven = TRUE;
            break;
        case  BSIM3v1S_MOD_LVFBCV :
            mod->BSIM3v1Slvfbcv = value->rValue;
            mod->BSIM3v1SlvfbcvGiven = TRUE;
            break;

	/* Width dependence */
        case  BSIM3v1S_MOD_WCDSC :
            mod->BSIM3v1Swcdsc = value->rValue;
            mod->BSIM3v1SwcdscGiven = TRUE;
            break;
       
       
         case  BSIM3v1S_MOD_WCDSCB :
            mod->BSIM3v1Swcdscb = value->rValue;
            mod->BSIM3v1SwcdscbGiven = TRUE;
            break;
         case  BSIM3v1S_MOD_WCDSCD :
            mod->BSIM3v1Swcdscd = value->rValue;
            mod->BSIM3v1SwcdscdGiven = TRUE;
            break;
        case  BSIM3v1S_MOD_WCIT :
            mod->BSIM3v1Swcit = value->rValue;
            mod->BSIM3v1SwcitGiven = TRUE;
            break;
        case  BSIM3v1S_MOD_WNFACTOR :
            mod->BSIM3v1Swnfactor = value->rValue;
            mod->BSIM3v1SwnfactorGiven = TRUE;
            break;
        case BSIM3v1S_MOD_WXJ:
            mod->BSIM3v1Swxj = value->rValue;
            mod->BSIM3v1SwxjGiven = TRUE;
            break;
        case BSIM3v1S_MOD_WVSAT:
            mod->BSIM3v1Swvsat = value->rValue;
            mod->BSIM3v1SwvsatGiven = TRUE;
            break;


        case BSIM3v1S_MOD_WA0:
            mod->BSIM3v1Swa0 = value->rValue;
            mod->BSIM3v1Swa0Given = TRUE;
            break;
        case BSIM3v1S_MOD_WAGS:
            mod->BSIM3v1Swags = value->rValue;
            mod->BSIM3v1SwagsGiven = TRUE;
            break;
        case BSIM3v1S_MOD_WA1:
            mod->BSIM3v1Swa1 = value->rValue;
            mod->BSIM3v1Swa1Given = TRUE;
            break;
        case BSIM3v1S_MOD_WA2:
            mod->BSIM3v1Swa2 = value->rValue;
            mod->BSIM3v1Swa2Given = TRUE;
            break;
        case BSIM3v1S_MOD_WAT:
            mod->BSIM3v1Swat = value->rValue;
            mod->BSIM3v1SwatGiven = TRUE;
            break;
        case BSIM3v1S_MOD_WKETA:
            mod->BSIM3v1Swketa = value->rValue;
            mod->BSIM3v1SwketaGiven = TRUE;
            break;    
        case BSIM3v1S_MOD_WNSUB:
            mod->BSIM3v1Swnsub = value->rValue;
            mod->BSIM3v1SwnsubGiven = TRUE;
            break;
        case BSIM3v1S_MOD_WNPEAK:
            mod->BSIM3v1Swnpeak = value->rValue;
            mod->BSIM3v1SwnpeakGiven = TRUE;
	    if (mod->BSIM3v1Swnpeak > 1.0e20)
		mod->BSIM3v1Swnpeak *= 1.0e-6;
            break;
        case BSIM3v1S_MOD_WNGATE:
            mod->BSIM3v1Swngate = value->rValue;
            mod->BSIM3v1SwngateGiven = TRUE;
	    if (mod->BSIM3v1Swngate > 1.0e23)
		mod->BSIM3v1Swngate *= 1.0e-6;
            break;
        case BSIM3v1S_MOD_WGAMMA1:
            mod->BSIM3v1Swgamma1 = value->rValue;
            mod->BSIM3v1Swgamma1Given = TRUE;
            break;
        case BSIM3v1S_MOD_WGAMMA2:
            mod->BSIM3v1Swgamma2 = value->rValue;
            mod->BSIM3v1Swgamma2Given = TRUE;
            break;
        case BSIM3v1S_MOD_WVBX:
            mod->BSIM3v1Swvbx = value->rValue;
            mod->BSIM3v1SwvbxGiven = TRUE;
            break;
        case BSIM3v1S_MOD_WVBM:
            mod->BSIM3v1Swvbm = value->rValue;
            mod->BSIM3v1SwvbmGiven = TRUE;
            break;
        case BSIM3v1S_MOD_WXT:
            mod->BSIM3v1Swxt = value->rValue;
            mod->BSIM3v1SwxtGiven = TRUE;
            break;
        case  BSIM3v1S_MOD_WK1:
            mod->BSIM3v1Swk1 = value->rValue;
            mod->BSIM3v1Swk1Given = TRUE;
            break;
        case  BSIM3v1S_MOD_WKT1:
            mod->BSIM3v1Swkt1 = value->rValue;
            mod->BSIM3v1Swkt1Given = TRUE;
            break;
        case  BSIM3v1S_MOD_WKT1L:
            mod->BSIM3v1Swkt1l = value->rValue;
            mod->BSIM3v1Swkt1lGiven = TRUE;
            break;
        case  BSIM3v1S_MOD_WKT2:
            mod->BSIM3v1Swkt2 = value->rValue;
            mod->BSIM3v1Swkt2Given = TRUE;
            break;
        case  BSIM3v1S_MOD_WK2:
            mod->BSIM3v1Swk2 = value->rValue;
            mod->BSIM3v1Swk2Given = TRUE;
            break;
        case  BSIM3v1S_MOD_WK3:
            mod->BSIM3v1Swk3 = value->rValue;
            mod->BSIM3v1Swk3Given = TRUE;
            break;
        case  BSIM3v1S_MOD_WK3B:
            mod->BSIM3v1Swk3b = value->rValue;
            mod->BSIM3v1Swk3bGiven = TRUE;
            break;
        case  BSIM3v1S_MOD_WNLX:
            mod->BSIM3v1Swnlx = value->rValue;
            mod->BSIM3v1SwnlxGiven = TRUE;
            break;
        case  BSIM3v1S_MOD_WW0:
            mod->BSIM3v1Sww0 = value->rValue;
            mod->BSIM3v1Sww0Given = TRUE;
            break;
        case  BSIM3v1S_MOD_WDVT0:               
            mod->BSIM3v1Swdvt0 = value->rValue;
            mod->BSIM3v1Swdvt0Given = TRUE;
            break;
        case  BSIM3v1S_MOD_WDVT1:             
            mod->BSIM3v1Swdvt1 = value->rValue;
            mod->BSIM3v1Swdvt1Given = TRUE;
            break;
        case  BSIM3v1S_MOD_WDVT2:             
            mod->BSIM3v1Swdvt2 = value->rValue;
            mod->BSIM3v1Swdvt2Given = TRUE;
            break;
        case  BSIM3v1S_MOD_WDVT0W:               
            mod->BSIM3v1Swdvt0w = value->rValue;
            mod->BSIM3v1Swdvt0wGiven = TRUE;
            break;
        case  BSIM3v1S_MOD_WDVT1W:             
            mod->BSIM3v1Swdvt1w = value->rValue;
            mod->BSIM3v1Swdvt1wGiven = TRUE;
            break;
        case  BSIM3v1S_MOD_WDVT2W:             
            mod->BSIM3v1Swdvt2w = value->rValue;
            mod->BSIM3v1Swdvt2wGiven = TRUE;
            break;
        case  BSIM3v1S_MOD_WDROUT:             
            mod->BSIM3v1Swdrout = value->rValue;
            mod->BSIM3v1SwdroutGiven = TRUE;
            break;
        case  BSIM3v1S_MOD_WDSUB:             
            mod->BSIM3v1Swdsub = value->rValue;
            mod->BSIM3v1SwdsubGiven = TRUE;
            break;
        case BSIM3v1S_MOD_WVTH0:
            mod->BSIM3v1Swvth0 = value->rValue;
            mod->BSIM3v1Swvth0Given = TRUE;
            break;
        case BSIM3v1S_MOD_WUA:
            mod->BSIM3v1Swua = value->rValue;
            mod->BSIM3v1SwuaGiven = TRUE;
            break;
        case BSIM3v1S_MOD_WUA1:
            mod->BSIM3v1Swua1 = value->rValue;
            mod->BSIM3v1Swua1Given = TRUE;
            break;
        case BSIM3v1S_MOD_WUB:
            mod->BSIM3v1Swub = value->rValue;
            mod->BSIM3v1SwubGiven = TRUE;
            break;
        case BSIM3v1S_MOD_WUB1:
            mod->BSIM3v1Swub1 = value->rValue;
            mod->BSIM3v1Swub1Given = TRUE;
            break;
        case BSIM3v1S_MOD_WUC:
            mod->BSIM3v1Swuc = value->rValue;
            mod->BSIM3v1SwucGiven = TRUE;
            break;
        case BSIM3v1S_MOD_WUC1:
            mod->BSIM3v1Swuc1 = value->rValue;
            mod->BSIM3v1Swuc1Given = TRUE;
            break;
        case  BSIM3v1S_MOD_WU0 :
            mod->BSIM3v1Swu0 = value->rValue;
            mod->BSIM3v1Swu0Given = TRUE;
            break;
        case  BSIM3v1S_MOD_WUTE :
            mod->BSIM3v1Swute = value->rValue;
            mod->BSIM3v1SwuteGiven = TRUE;
            break;
        case BSIM3v1S_MOD_WVOFF:
            mod->BSIM3v1Swvoff = value->rValue;
            mod->BSIM3v1SwvoffGiven = TRUE;
            break;
        case  BSIM3v1S_MOD_WDELTA :
            mod->BSIM3v1Swdelta = value->rValue;
            mod->BSIM3v1SwdeltaGiven = TRUE;
            break;
        case BSIM3v1S_MOD_WRDSW:
            mod->BSIM3v1Swrdsw = value->rValue;
            mod->BSIM3v1SwrdswGiven = TRUE;
            break;                     
        case BSIM3v1S_MOD_WPRWB:
            mod->BSIM3v1Swprwb = value->rValue;
            mod->BSIM3v1SwprwbGiven = TRUE;
            break;                     
        case BSIM3v1S_MOD_WPRWG:
            mod->BSIM3v1Swprwg = value->rValue;
            mod->BSIM3v1SwprwgGiven = TRUE;
            break;                     
        case BSIM3v1S_MOD_WPRT:
            mod->BSIM3v1Swprt = value->rValue;
            mod->BSIM3v1SwprtGiven = TRUE;
            break;                     
        case BSIM3v1S_MOD_WETA0:
            mod->BSIM3v1Sweta0 = value->rValue;
            mod->BSIM3v1Sweta0Given = TRUE;
            break;                 
        case BSIM3v1S_MOD_WETAB:
            mod->BSIM3v1Swetab = value->rValue;
            mod->BSIM3v1SwetabGiven = TRUE;
            break;                 
        case BSIM3v1S_MOD_WPCLM:
            mod->BSIM3v1Swpclm = value->rValue;
            mod->BSIM3v1SwpclmGiven = TRUE;
            break;                 
        case BSIM3v1S_MOD_WPDIBL1:
            mod->BSIM3v1Swpdibl1 = value->rValue;
            mod->BSIM3v1Swpdibl1Given = TRUE;
            break;                 
        case BSIM3v1S_MOD_WPDIBL2:
            mod->BSIM3v1Swpdibl2 = value->rValue;
            mod->BSIM3v1Swpdibl2Given = TRUE;
            break;                 
        case BSIM3v1S_MOD_WPDIBLB:
            mod->BSIM3v1Swpdiblb = value->rValue;
            mod->BSIM3v1SwpdiblbGiven = TRUE;
            break;                 
        case BSIM3v1S_MOD_WPSCBE1:
            mod->BSIM3v1Swpscbe1 = value->rValue;
            mod->BSIM3v1Swpscbe1Given = TRUE;
            break;                 
        case BSIM3v1S_MOD_WPSCBE2:
            mod->BSIM3v1Swpscbe2 = value->rValue;
            mod->BSIM3v1Swpscbe2Given = TRUE;
            break;                 
        case BSIM3v1S_MOD_WPVAG:
            mod->BSIM3v1Swpvag = value->rValue;
            mod->BSIM3v1SwpvagGiven = TRUE;
            break;                 
        case  BSIM3v1S_MOD_WWR :
            mod->BSIM3v1Swwr = value->rValue;
            mod->BSIM3v1SwwrGiven = TRUE;
            break;
        case  BSIM3v1S_MOD_WDWG :
            mod->BSIM3v1Swdwg = value->rValue;
            mod->BSIM3v1SwdwgGiven = TRUE;
            break;
        case  BSIM3v1S_MOD_WDWB :
            mod->BSIM3v1Swdwb = value->rValue;
            mod->BSIM3v1SwdwbGiven = TRUE;
            break;
        case  BSIM3v1S_MOD_WB0 :
            mod->BSIM3v1Swb0 = value->rValue;
            mod->BSIM3v1Swb0Given = TRUE;
            break;
        case  BSIM3v1S_MOD_WB1 :
            mod->BSIM3v1Swb1 = value->rValue;
            mod->BSIM3v1Swb1Given = TRUE;
            break;
        case  BSIM3v1S_MOD_WALPHA0 :
            mod->BSIM3v1Swalpha0 = value->rValue;
            mod->BSIM3v1Swalpha0Given = TRUE;
            break;
        case  BSIM3v1S_MOD_WBETA0 :
            mod->BSIM3v1Swbeta0 = value->rValue;
            mod->BSIM3v1Swbeta0Given = TRUE;
            break;

        case  BSIM3v1S_MOD_WELM :
            mod->BSIM3v1Swelm = value->rValue;
            mod->BSIM3v1SwelmGiven = TRUE;
            break;
        case  BSIM3v1S_MOD_WCGSL :
            mod->BSIM3v1Swcgsl = value->rValue;
            mod->BSIM3v1SwcgslGiven = TRUE;
            break;
        case  BSIM3v1S_MOD_WCGDL :
            mod->BSIM3v1Swcgdl = value->rValue;
            mod->BSIM3v1SwcgdlGiven = TRUE;
            break;
        case  BSIM3v1S_MOD_WCKAPPA :
            mod->BSIM3v1Swckappa = value->rValue;
            mod->BSIM3v1SwckappaGiven = TRUE;
            break;
        case  BSIM3v1S_MOD_WCF :
            mod->BSIM3v1Swcf = value->rValue;
            mod->BSIM3v1SwcfGiven = TRUE;
            break;
        case  BSIM3v1S_MOD_WCLC :
            mod->BSIM3v1Swclc = value->rValue;
            mod->BSIM3v1SwclcGiven = TRUE;
            break;
        case  BSIM3v1S_MOD_WCLE :
            mod->BSIM3v1Swcle = value->rValue;
            mod->BSIM3v1SwcleGiven = TRUE;
            break;
        case  BSIM3v1S_MOD_WVFBCV :
            mod->BSIM3v1Swvfbcv = value->rValue;
            mod->BSIM3v1SwvfbcvGiven = TRUE;
            break;

	/* Cross-term dependence */
        case  BSIM3v1S_MOD_PCDSC :
            mod->BSIM3v1Spcdsc = value->rValue;
            mod->BSIM3v1SpcdscGiven = TRUE;
            break;


        case  BSIM3v1S_MOD_PCDSCB :
            mod->BSIM3v1Spcdscb = value->rValue;
            mod->BSIM3v1SpcdscbGiven = TRUE;
            break;
        case  BSIM3v1S_MOD_PCDSCD :
            mod->BSIM3v1Spcdscd = value->rValue;
            mod->BSIM3v1SpcdscdGiven = TRUE;
            break;
        case  BSIM3v1S_MOD_PCIT :
            mod->BSIM3v1Spcit = value->rValue;
            mod->BSIM3v1SpcitGiven = TRUE;
            break;
        case  BSIM3v1S_MOD_PNFACTOR :
            mod->BSIM3v1Spnfactor = value->rValue;
            mod->BSIM3v1SpnfactorGiven = TRUE;
            break;
        case BSIM3v1S_MOD_PXJ:
            mod->BSIM3v1Spxj = value->rValue;
            mod->BSIM3v1SpxjGiven = TRUE;
            break;
        case BSIM3v1S_MOD_PVSAT:
            mod->BSIM3v1Spvsat = value->rValue;
            mod->BSIM3v1SpvsatGiven = TRUE;
            break;


        case BSIM3v1S_MOD_PA0:
            mod->BSIM3v1Spa0 = value->rValue;
            mod->BSIM3v1Spa0Given = TRUE;
            break;
        case BSIM3v1S_MOD_PAGS:
            mod->BSIM3v1Spags = value->rValue;
            mod->BSIM3v1SpagsGiven = TRUE;
            break;
        case BSIM3v1S_MOD_PA1:
            mod->BSIM3v1Spa1 = value->rValue;
            mod->BSIM3v1Spa1Given = TRUE;
            break;
        case BSIM3v1S_MOD_PA2:
            mod->BSIM3v1Spa2 = value->rValue;
            mod->BSIM3v1Spa2Given = TRUE;
            break;
        case BSIM3v1S_MOD_PAT:
            mod->BSIM3v1Spat = value->rValue;
            mod->BSIM3v1SpatGiven = TRUE;
            break;
        case BSIM3v1S_MOD_PKETA:
            mod->BSIM3v1Spketa = value->rValue;
            mod->BSIM3v1SpketaGiven = TRUE;
            break;    
        case BSIM3v1S_MOD_PNSUB:
            mod->BSIM3v1Spnsub = value->rValue;
            mod->BSIM3v1SpnsubGiven = TRUE;
            break;
        case BSIM3v1S_MOD_PNPEAK:
            mod->BSIM3v1Spnpeak = value->rValue;
            mod->BSIM3v1SpnpeakGiven = TRUE;
	    if (mod->BSIM3v1Spnpeak > 1.0e20)
		mod->BSIM3v1Spnpeak *= 1.0e-6;
            break;
        case BSIM3v1S_MOD_PNGATE:
            mod->BSIM3v1Spngate = value->rValue;
            mod->BSIM3v1SpngateGiven = TRUE;
	    if (mod->BSIM3v1Spngate > 1.0e23)
		mod->BSIM3v1Spngate *= 1.0e-6;
            break;
        case BSIM3v1S_MOD_PGAMMA1:
            mod->BSIM3v1Spgamma1 = value->rValue;
            mod->BSIM3v1Spgamma1Given = TRUE;
            break;
        case BSIM3v1S_MOD_PGAMMA2:
            mod->BSIM3v1Spgamma2 = value->rValue;
            mod->BSIM3v1Spgamma2Given = TRUE;
            break;
        case BSIM3v1S_MOD_PVBX:
            mod->BSIM3v1Spvbx = value->rValue;
            mod->BSIM3v1SpvbxGiven = TRUE;
            break;
        case BSIM3v1S_MOD_PVBM:
            mod->BSIM3v1Spvbm = value->rValue;
            mod->BSIM3v1SpvbmGiven = TRUE;
            break;
        case BSIM3v1S_MOD_PXT:
            mod->BSIM3v1Spxt = value->rValue;
            mod->BSIM3v1SpxtGiven = TRUE;
            break;
        case  BSIM3v1S_MOD_PK1:
            mod->BSIM3v1Spk1 = value->rValue;
            mod->BSIM3v1Spk1Given = TRUE;
            break;
        case  BSIM3v1S_MOD_PKT1:
            mod->BSIM3v1Spkt1 = value->rValue;
            mod->BSIM3v1Spkt1Given = TRUE;
            break;
        case  BSIM3v1S_MOD_PKT1L:
            mod->BSIM3v1Spkt1l = value->rValue;
            mod->BSIM3v1Spkt1lGiven = TRUE;
            break;
        case  BSIM3v1S_MOD_PKT2:
            mod->BSIM3v1Spkt2 = value->rValue;
            mod->BSIM3v1Spkt2Given = TRUE;
            break;
        case  BSIM3v1S_MOD_PK2:
            mod->BSIM3v1Spk2 = value->rValue;
            mod->BSIM3v1Spk2Given = TRUE;
            break;
        case  BSIM3v1S_MOD_PK3:
            mod->BSIM3v1Spk3 = value->rValue;
            mod->BSIM3v1Spk3Given = TRUE;
            break;
        case  BSIM3v1S_MOD_PK3B:
            mod->BSIM3v1Spk3b = value->rValue;
            mod->BSIM3v1Spk3bGiven = TRUE;
            break;
        case  BSIM3v1S_MOD_PNLX:
            mod->BSIM3v1Spnlx = value->rValue;
            mod->BSIM3v1SpnlxGiven = TRUE;
            break;
        case  BSIM3v1S_MOD_PW0:
            mod->BSIM3v1Spw0 = value->rValue;
            mod->BSIM3v1Spw0Given = TRUE;
            break;
        case  BSIM3v1S_MOD_PDVT0:               
            mod->BSIM3v1Spdvt0 = value->rValue;
            mod->BSIM3v1Spdvt0Given = TRUE;
            break;
        case  BSIM3v1S_MOD_PDVT1:             
            mod->BSIM3v1Spdvt1 = value->rValue;
            mod->BSIM3v1Spdvt1Given = TRUE;
            break;
        case  BSIM3v1S_MOD_PDVT2:             
            mod->BSIM3v1Spdvt2 = value->rValue;
            mod->BSIM3v1Spdvt2Given = TRUE;
            break;
        case  BSIM3v1S_MOD_PDVT0W:               
            mod->BSIM3v1Spdvt0w = value->rValue;
            mod->BSIM3v1Spdvt0wGiven = TRUE;
            break;
        case  BSIM3v1S_MOD_PDVT1W:             
            mod->BSIM3v1Spdvt1w = value->rValue;
            mod->BSIM3v1Spdvt1wGiven = TRUE;
            break;
        case  BSIM3v1S_MOD_PDVT2W:             
            mod->BSIM3v1Spdvt2w = value->rValue;
            mod->BSIM3v1Spdvt2wGiven = TRUE;
            break;
        case  BSIM3v1S_MOD_PDROUT:             
            mod->BSIM3v1Spdrout = value->rValue;
            mod->BSIM3v1SpdroutGiven = TRUE;
            break;
        case  BSIM3v1S_MOD_PDSUB:             
            mod->BSIM3v1Spdsub = value->rValue;
            mod->BSIM3v1SpdsubGiven = TRUE;
            break;
        case BSIM3v1S_MOD_PVTH0:
            mod->BSIM3v1Spvth0 = value->rValue;
            mod->BSIM3v1Spvth0Given = TRUE;
            break;
        case BSIM3v1S_MOD_PUA:
            mod->BSIM3v1Spua = value->rValue;
            mod->BSIM3v1SpuaGiven = TRUE;
            break;
        case BSIM3v1S_MOD_PUA1:
            mod->BSIM3v1Spua1 = value->rValue;
            mod->BSIM3v1Spua1Given = TRUE;
            break;
        case BSIM3v1S_MOD_PUB:
            mod->BSIM3v1Spub = value->rValue;
            mod->BSIM3v1SpubGiven = TRUE;
            break;
        case BSIM3v1S_MOD_PUB1:
            mod->BSIM3v1Spub1 = value->rValue;
            mod->BSIM3v1Spub1Given = TRUE;
            break;
        case BSIM3v1S_MOD_PUC:
            mod->BSIM3v1Spuc = value->rValue;
            mod->BSIM3v1SpucGiven = TRUE;
            break;
        case BSIM3v1S_MOD_PUC1:
            mod->BSIM3v1Spuc1 = value->rValue;
            mod->BSIM3v1Spuc1Given = TRUE;
            break;
        case  BSIM3v1S_MOD_PU0 :
            mod->BSIM3v1Spu0 = value->rValue;
            mod->BSIM3v1Spu0Given = TRUE;
            break;
        case  BSIM3v1S_MOD_PUTE :
            mod->BSIM3v1Spute = value->rValue;
            mod->BSIM3v1SputeGiven = TRUE;
            break;
        case BSIM3v1S_MOD_PVOFF:
            mod->BSIM3v1Spvoff = value->rValue;
            mod->BSIM3v1SpvoffGiven = TRUE;
            break;
        case  BSIM3v1S_MOD_PDELTA :
            mod->BSIM3v1Spdelta = value->rValue;
            mod->BSIM3v1SpdeltaGiven = TRUE;
            break;
        case BSIM3v1S_MOD_PRDSW:
            mod->BSIM3v1Sprdsw = value->rValue;
            mod->BSIM3v1SprdswGiven = TRUE;
            break;                     
        case BSIM3v1S_MOD_PPRWB:
            mod->BSIM3v1Spprwb = value->rValue;
            mod->BSIM3v1SpprwbGiven = TRUE;
            break;                     
        case BSIM3v1S_MOD_PPRWG:
            mod->BSIM3v1Spprwg = value->rValue;
            mod->BSIM3v1SpprwgGiven = TRUE;
            break;                     
        case BSIM3v1S_MOD_PPRT:
            mod->BSIM3v1Spprt = value->rValue;
            mod->BSIM3v1SpprtGiven = TRUE;
            break;                     
        case BSIM3v1S_MOD_PETA0:
            mod->BSIM3v1Speta0 = value->rValue;
            mod->BSIM3v1Speta0Given = TRUE;
            break;                 
        case BSIM3v1S_MOD_PETAB:
            mod->BSIM3v1Spetab = value->rValue;
            mod->BSIM3v1SpetabGiven = TRUE;
            break;                 
        case BSIM3v1S_MOD_PPCLM:
            mod->BSIM3v1Sppclm = value->rValue;
            mod->BSIM3v1SppclmGiven = TRUE;
            break;                 
        case BSIM3v1S_MOD_PPDIBL1:
            mod->BSIM3v1Sppdibl1 = value->rValue;
            mod->BSIM3v1Sppdibl1Given = TRUE;
            break;                 
        case BSIM3v1S_MOD_PPDIBL2:
            mod->BSIM3v1Sppdibl2 = value->rValue;
            mod->BSIM3v1Sppdibl2Given = TRUE;
            break;                 
        case BSIM3v1S_MOD_PPDIBLB:
            mod->BSIM3v1Sppdiblb = value->rValue;
            mod->BSIM3v1SppdiblbGiven = TRUE;
            break;                 
        case BSIM3v1S_MOD_PPSCBE1:
            mod->BSIM3v1Sppscbe1 = value->rValue;
            mod->BSIM3v1Sppscbe1Given = TRUE;
            break;                 
        case BSIM3v1S_MOD_PPSCBE2:
            mod->BSIM3v1Sppscbe2 = value->rValue;
            mod->BSIM3v1Sppscbe2Given = TRUE;
            break;                 
        case BSIM3v1S_MOD_PPVAG:
            mod->BSIM3v1Sppvag = value->rValue;
            mod->BSIM3v1SppvagGiven = TRUE;
            break;                 
        case  BSIM3v1S_MOD_PWR :
            mod->BSIM3v1Spwr = value->rValue;
            mod->BSIM3v1SpwrGiven = TRUE;
            break;
        case  BSIM3v1S_MOD_PDWG :
            mod->BSIM3v1Spdwg = value->rValue;
            mod->BSIM3v1SpdwgGiven = TRUE;
            break;
        case  BSIM3v1S_MOD_PDWB :
            mod->BSIM3v1Spdwb = value->rValue;
            mod->BSIM3v1SpdwbGiven = TRUE;
            break;
        case  BSIM3v1S_MOD_PB0 :
            mod->BSIM3v1Spb0 = value->rValue;
            mod->BSIM3v1Spb0Given = TRUE;
            break;
        case  BSIM3v1S_MOD_PB1 :
            mod->BSIM3v1Spb1 = value->rValue;
            mod->BSIM3v1Spb1Given = TRUE;
            break;
        case  BSIM3v1S_MOD_PALPHA0 :
            mod->BSIM3v1Spalpha0 = value->rValue;
            mod->BSIM3v1Spalpha0Given = TRUE;
            break;
        case  BSIM3v1S_MOD_PBETA0 :
            mod->BSIM3v1Spbeta0 = value->rValue;
            mod->BSIM3v1Spbeta0Given = TRUE;
            break;

        case  BSIM3v1S_MOD_PELM :
            mod->BSIM3v1Spelm = value->rValue;
            mod->BSIM3v1SpelmGiven = TRUE;
            break;
        case  BSIM3v1S_MOD_PCGSL :
            mod->BSIM3v1Spcgsl = value->rValue;
            mod->BSIM3v1SpcgslGiven = TRUE;
            break;
        case  BSIM3v1S_MOD_PCGDL :
            mod->BSIM3v1Spcgdl = value->rValue;
            mod->BSIM3v1SpcgdlGiven = TRUE;
            break;
        case  BSIM3v1S_MOD_PCKAPPA :
            mod->BSIM3v1Spckappa = value->rValue;
            mod->BSIM3v1SpckappaGiven = TRUE;
            break;
        case  BSIM3v1S_MOD_PCF :
            mod->BSIM3v1Spcf = value->rValue;
            mod->BSIM3v1SpcfGiven = TRUE;
            break;
        case  BSIM3v1S_MOD_PCLC :
            mod->BSIM3v1Spclc = value->rValue;
            mod->BSIM3v1SpclcGiven = TRUE;
            break;
        case  BSIM3v1S_MOD_PCLE :
            mod->BSIM3v1Spcle = value->rValue;
            mod->BSIM3v1SpcleGiven = TRUE;
            break;
        case  BSIM3v1S_MOD_PVFBCV :
            mod->BSIM3v1Spvfbcv = value->rValue;
            mod->BSIM3v1SpvfbcvGiven = TRUE;
            break;

        case  BSIM3v1S_MOD_TNOM :
            mod->BSIM3v1Stnom = value->rValue + 273.15;
            mod->BSIM3v1StnomGiven = TRUE;
            break;
        case  BSIM3v1S_MOD_CGSO :
            mod->BSIM3v1Scgso = value->rValue;
            mod->BSIM3v1ScgsoGiven = TRUE;
            break;
        case  BSIM3v1S_MOD_CGDO :
            mod->BSIM3v1Scgdo = value->rValue;
            mod->BSIM3v1ScgdoGiven = TRUE;
            break;
        case  BSIM3v1S_MOD_CGBO :
            mod->BSIM3v1Scgbo = value->rValue;
            mod->BSIM3v1ScgboGiven = TRUE;
            break;
        case  BSIM3v1S_MOD_XPART :
            mod->BSIM3v1Sxpart = value->rValue;
            mod->BSIM3v1SxpartGiven = TRUE;
            break;
        case  BSIM3v1S_MOD_RSH :
            mod->BSIM3v1SsheetResistance = value->rValue;
            mod->BSIM3v1SsheetResistanceGiven = TRUE;
            break;
        case  BSIM3v1S_MOD_JS :
            mod->BSIM3v1SjctSatCurDensity = value->rValue;
            mod->BSIM3v1SjctSatCurDensityGiven = TRUE;
            break;
        case  BSIM3v1S_MOD_JSW :
            mod->BSIM3v1SjctSidewallSatCurDensity = value->rValue;
            mod->BSIM3v1SjctSidewallSatCurDensityGiven = TRUE;
            break;
        case  BSIM3v1S_MOD_PB :
            mod->BSIM3v1SbulkJctPotential = value->rValue;
            mod->BSIM3v1SbulkJctPotentialGiven = TRUE;
            break;
        case  BSIM3v1S_MOD_MJ :
            mod->BSIM3v1SbulkJctBotGradingCoeff = value->rValue;
            mod->BSIM3v1SbulkJctBotGradingCoeffGiven = TRUE;
            break;
        case  BSIM3v1S_MOD_PBSW :
            mod->BSIM3v1SsidewallJctPotential = value->rValue;
            mod->BSIM3v1SsidewallJctPotentialGiven = TRUE;
            break;
        case  BSIM3v1S_MOD_MJSW :
            mod->BSIM3v1SbulkJctSideGradingCoeff = value->rValue;
            mod->BSIM3v1SbulkJctSideGradingCoeffGiven = TRUE;
            break;
        case  BSIM3v1S_MOD_CJ :
            mod->BSIM3v1SunitAreaJctCap = value->rValue;
            mod->BSIM3v1SunitAreaJctCapGiven = TRUE;
            break;
        case  BSIM3v1S_MOD_CJSW :
            mod->BSIM3v1SunitLengthSidewallJctCap = value->rValue;
            mod->BSIM3v1SunitLengthSidewallJctCapGiven = TRUE;
            break;
        case  BSIM3v1S_MOD_NJ :
            mod->BSIM3v1SjctEmissionCoeff = value->rValue;
            mod->BSIM3v1SjctEmissionCoeffGiven = TRUE;
            break;
        case  BSIM3v1S_MOD_PBSWG :
            mod->BSIM3v1SGatesidewallJctPotential = value->rValue;
            mod->BSIM3v1SGatesidewallJctPotentialGiven = TRUE;
            break;
        case  BSIM3v1S_MOD_MJSWG :
            mod->BSIM3v1SbulkJctGateSideGradingCoeff = value->rValue;
            mod->BSIM3v1SbulkJctGateSideGradingCoeffGiven = TRUE;
            break;
        case  BSIM3v1S_MOD_CJSWG :
            mod->BSIM3v1SunitLengthGateSidewallJctCap = value->rValue;
            mod->BSIM3v1SunitLengthGateSidewallJctCapGiven = TRUE;
            break;
        case  BSIM3v1S_MOD_XTI :
            mod->BSIM3v1SjctTempExponent = value->rValue;
            mod->BSIM3v1SjctTempExponentGiven = TRUE;
            break;
        case  BSIM3v1S_MOD_LINT :
            mod->BSIM3v1SLint = value->rValue;
            mod->BSIM3v1SLintGiven = TRUE;
            break;
        case  BSIM3v1S_MOD_LL :
            mod->BSIM3v1SLl = value->rValue;
            mod->BSIM3v1SLlGiven = TRUE;
            break;
        case  BSIM3v1S_MOD_LLN :
            mod->BSIM3v1SLln = value->rValue;
            mod->BSIM3v1SLlnGiven = TRUE;
            break;
        case  BSIM3v1S_MOD_LW :
            mod->BSIM3v1SLw = value->rValue;
            mod->BSIM3v1SLwGiven = TRUE;
            break;
        case  BSIM3v1S_MOD_LWN :
            mod->BSIM3v1SLwn = value->rValue;
            mod->BSIM3v1SLwnGiven = TRUE;
            break;
        case  BSIM3v1S_MOD_LWL :
            mod->BSIM3v1SLwl = value->rValue;
            mod->BSIM3v1SLwlGiven = TRUE;
            break;
        case  BSIM3v1S_MOD_LMIN :
            mod->BSIM3v1SLmin = value->rValue;
            mod->BSIM3v1SLminGiven = TRUE;
            break;
        case  BSIM3v1S_MOD_LMAX :
            mod->BSIM3v1SLmax = value->rValue;
            mod->BSIM3v1SLmaxGiven = TRUE;
            break;
        case  BSIM3v1S_MOD_WINT :
            mod->BSIM3v1SWint = value->rValue;
            mod->BSIM3v1SWintGiven = TRUE;
            break;
        case  BSIM3v1S_MOD_WL :
            mod->BSIM3v1SWl = value->rValue;
            mod->BSIM3v1SWlGiven = TRUE;
            break;
        case  BSIM3v1S_MOD_WLN :
            mod->BSIM3v1SWln = value->rValue;
            mod->BSIM3v1SWlnGiven = TRUE;
            break;
        case  BSIM3v1S_MOD_WW :
            mod->BSIM3v1SWw = value->rValue;
            mod->BSIM3v1SWwGiven = TRUE;
            break;
        case  BSIM3v1S_MOD_WWN :
            mod->BSIM3v1SWwn = value->rValue;
            mod->BSIM3v1SWwnGiven = TRUE;
            break;
        case  BSIM3v1S_MOD_WWL :
            mod->BSIM3v1SWwl = value->rValue;
            mod->BSIM3v1SWwlGiven = TRUE;
            break;
        case  BSIM3v1S_MOD_WMIN :
            mod->BSIM3v1SWmin = value->rValue;
            mod->BSIM3v1SWminGiven = TRUE;
            break;
        case  BSIM3v1S_MOD_WMAX :
            mod->BSIM3v1SWmax = value->rValue;
            mod->BSIM3v1SWmaxGiven = TRUE;
            break;

        case  BSIM3v1S_MOD_NOIA :
            mod->BSIM3v1SoxideTrapDensityA = value->rValue;
            mod->BSIM3v1SoxideTrapDensityAGiven = TRUE;
            break;
        case  BSIM3v1S_MOD_NOIB :
            mod->BSIM3v1SoxideTrapDensityB = value->rValue;
            mod->BSIM3v1SoxideTrapDensityBGiven = TRUE;
            break;
        case  BSIM3v1S_MOD_NOIC :
            mod->BSIM3v1SoxideTrapDensityC = value->rValue;
            mod->BSIM3v1SoxideTrapDensityCGiven = TRUE;
            break;
        case  BSIM3v1S_MOD_EM :
            mod->BSIM3v1Sem = value->rValue;
            mod->BSIM3v1SemGiven = TRUE;
            break;
        case  BSIM3v1S_MOD_EF :
            mod->BSIM3v1Sef = value->rValue;
            mod->BSIM3v1SefGiven = TRUE;
            break;
        case  BSIM3v1S_MOD_AF :
            mod->BSIM3v1Saf = value->rValue;
            mod->BSIM3v1SafGiven = TRUE;
            break;
        case  BSIM3v1S_MOD_KF :
            mod->BSIM3v1Skf = value->rValue;
            mod->BSIM3v1SkfGiven = TRUE;
            break;
        case  BSIM3v1S_MOD_NMOS  :
            if(value->iValue) {
                mod->BSIM3v1Stype = 1;
                mod->BSIM3v1StypeGiven = TRUE;
            }
            break;
        case  BSIM3v1S_MOD_PMOS  :
            if(value->iValue) {
                mod->BSIM3v1Stype = - 1;
                mod->BSIM3v1StypeGiven = TRUE;
            }
            break;
/* serban */
        case  BSIM3v1S_MOD_HDIF  :
            mod->BSIM3v1Shdif = value->rValue;
            mod->BSIM3v1ShdifGiven = TRUE;
            break;
        default:
            return(E_BADPARM);
    }
    return(OK);
}


