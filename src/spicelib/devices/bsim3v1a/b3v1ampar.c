/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1995 Min-Chie Jeng and Mansun Chan.
Modified by Paolo Nenzi 2002
File: b3v1ampar.c
**********/

#include "ngspice.h"
#include "bsim3v1adef.h"
#include "ifsim.h"
#include "sperror.h"
#include "suffix.h"


int
BSIM3v1AmParam(int param, IFvalue *value, GENmodel *inMod)
{
    BSIM3v1Amodel *mod = (BSIM3v1Amodel*)inMod;
    switch(param)
    {   case  BSIM3v1A_MOD_MOBMOD :
            mod->BSIM3v1AmobMod = value->iValue;
            mod->BSIM3v1AmobModGiven = TRUE;
            break;
        case  BSIM3v1A_MOD_BINUNIT :
            mod->BSIM3v1AbinUnit = value->iValue;
            mod->BSIM3v1AbinUnitGiven = TRUE;
            break;
        case  BSIM3v1A_MOD_CAPMOD :
            mod->BSIM3v1AcapMod = value->iValue;
            mod->BSIM3v1AcapModGiven = TRUE;
            break;
        case  BSIM3v1A_MOD_NQSMOD :
            mod->BSIM3v1AnqsMod = value->iValue;
            mod->BSIM3v1AnqsModGiven = TRUE;
            break;
        case  BSIM3v1A_MOD_NOIMOD :
            mod->BSIM3v1AnoiMod = value->iValue;
            mod->BSIM3v1AnoiModGiven = TRUE;
            break;
        case  BSIM3v1A_MOD_TOX :
            mod->BSIM3v1Atox = value->rValue;
            mod->BSIM3v1AtoxGiven = TRUE;
            break;

        case  BSIM3v1A_MOD_CDSC :
            mod->BSIM3v1Acdsc = value->rValue;
            mod->BSIM3v1AcdscGiven = TRUE;
            break;
        case  BSIM3v1A_MOD_CDSCB :
            mod->BSIM3v1Acdscb = value->rValue;
            mod->BSIM3v1AcdscbGiven = TRUE;
            break;

        case  BSIM3v1A_MOD_CDSCD :
            mod->BSIM3v1Acdscd = value->rValue;
            mod->BSIM3v1AcdscdGiven = TRUE;
            break;

        case  BSIM3v1A_MOD_CIT :
            mod->BSIM3v1Acit = value->rValue;
            mod->BSIM3v1AcitGiven = TRUE;
            break;
        case  BSIM3v1A_MOD_NFACTOR :
            mod->BSIM3v1Anfactor = value->rValue;
            mod->BSIM3v1AnfactorGiven = TRUE;
            break;
        case BSIM3v1A_MOD_XJ:
            mod->BSIM3v1Axj = value->rValue;
            mod->BSIM3v1AxjGiven = TRUE;
            break;
        case BSIM3v1A_MOD_VSAT:
            mod->BSIM3v1Avsat = value->rValue;
            mod->BSIM3v1AvsatGiven = TRUE;
            break;
        case BSIM3v1A_MOD_A0:
            mod->BSIM3v1Aa0 = value->rValue;
            mod->BSIM3v1Aa0Given = TRUE;
            break;
        
        case BSIM3v1A_MOD_AGS:
            mod->BSIM3v1Aags= value->rValue;
            mod->BSIM3v1AagsGiven = TRUE;
            break;
        
        case BSIM3v1A_MOD_A1:
            mod->BSIM3v1Aa1 = value->rValue;
            mod->BSIM3v1Aa1Given = TRUE;
            break;
        case BSIM3v1A_MOD_A2:
            mod->BSIM3v1Aa2 = value->rValue;
            mod->BSIM3v1Aa2Given = TRUE;
            break;
        case BSIM3v1A_MOD_AT:
            mod->BSIM3v1Aat = value->rValue;
            mod->BSIM3v1AatGiven = TRUE;
            break;
        case BSIM3v1A_MOD_KETA:
            mod->BSIM3v1Aketa = value->rValue;
            mod->BSIM3v1AketaGiven = TRUE;
            break;    
        case BSIM3v1A_MOD_NSUB:
            mod->BSIM3v1Ansub = value->rValue;
            mod->BSIM3v1AnsubGiven = TRUE;
            break;
        case BSIM3v1A_MOD_NPEAK:
            mod->BSIM3v1Anpeak = value->rValue;
            mod->BSIM3v1AnpeakGiven = TRUE;
	    if (mod->BSIM3v1Anpeak > 1.0e20)
		mod->BSIM3v1Anpeak *= 1.0e-6;
            break;
        case BSIM3v1A_MOD_NGATE:
            mod->BSIM3v1Angate = value->rValue;
            mod->BSIM3v1AngateGiven = TRUE;
	    if (mod->BSIM3v1Angate > 1.0e23)
		mod->BSIM3v1Angate *= 1.0e-6;
            break;
        case BSIM3v1A_MOD_GAMMA1:
            mod->BSIM3v1Agamma1 = value->rValue;
            mod->BSIM3v1Agamma1Given = TRUE;
            break;
        case BSIM3v1A_MOD_GAMMA2:
            mod->BSIM3v1Agamma2 = value->rValue;
            mod->BSIM3v1Agamma2Given = TRUE;
            break;
        case BSIM3v1A_MOD_VBX:
            mod->BSIM3v1Avbx = value->rValue;
            mod->BSIM3v1AvbxGiven = TRUE;
            break;
        case BSIM3v1A_MOD_VBM:
            mod->BSIM3v1Avbm = value->rValue;
            mod->BSIM3v1AvbmGiven = TRUE;
            break;
        case BSIM3v1A_MOD_XT:
            mod->BSIM3v1Axt = value->rValue;
            mod->BSIM3v1AxtGiven = TRUE;
            break;
        case  BSIM3v1A_MOD_K1:
            mod->BSIM3v1Ak1 = value->rValue;
            mod->BSIM3v1Ak1Given = TRUE;
            break;
        case  BSIM3v1A_MOD_KT1:
            mod->BSIM3v1Akt1 = value->rValue;
            mod->BSIM3v1Akt1Given = TRUE;
            break;
        case  BSIM3v1A_MOD_KT1L:
            mod->BSIM3v1Akt1l = value->rValue;
            mod->BSIM3v1Akt1lGiven = TRUE;
            break;
        case  BSIM3v1A_MOD_KT2:
            mod->BSIM3v1Akt2 = value->rValue;
            mod->BSIM3v1Akt2Given = TRUE;
            break;
        case  BSIM3v1A_MOD_K2:
            mod->BSIM3v1Ak2 = value->rValue;
            mod->BSIM3v1Ak2Given = TRUE;
            break;
        case  BSIM3v1A_MOD_K3:
            mod->BSIM3v1Ak3 = value->rValue;
            mod->BSIM3v1Ak3Given = TRUE;
            break;
        case  BSIM3v1A_MOD_K3B:
            mod->BSIM3v1Ak3b = value->rValue;
            mod->BSIM3v1Ak3bGiven = TRUE;
            break;
        case  BSIM3v1A_MOD_NLX:
            mod->BSIM3v1Anlx = value->rValue;
            mod->BSIM3v1AnlxGiven = TRUE;
            break;
        case  BSIM3v1A_MOD_W0:
            mod->BSIM3v1Aw0 = value->rValue;
            mod->BSIM3v1Aw0Given = TRUE;
            break;
        case  BSIM3v1A_MOD_DVT0:               
            mod->BSIM3v1Advt0 = value->rValue;
            mod->BSIM3v1Advt0Given = TRUE;
            break;
        case  BSIM3v1A_MOD_DVT1:             
            mod->BSIM3v1Advt1 = value->rValue;
            mod->BSIM3v1Advt1Given = TRUE;
            break;
        case  BSIM3v1A_MOD_DVT2:             
            mod->BSIM3v1Advt2 = value->rValue;
            mod->BSIM3v1Advt2Given = TRUE;
            break;
        case  BSIM3v1A_MOD_DVT0W:               
            mod->BSIM3v1Advt0w = value->rValue;
            mod->BSIM3v1Advt0wGiven = TRUE;
            break;
        case  BSIM3v1A_MOD_DVT1W:             
            mod->BSIM3v1Advt1w = value->rValue;
            mod->BSIM3v1Advt1wGiven = TRUE;
            break;
        case  BSIM3v1A_MOD_DVT2W:             
            mod->BSIM3v1Advt2w = value->rValue;
            mod->BSIM3v1Advt2wGiven = TRUE;
            break;
        case  BSIM3v1A_MOD_DROUT:             
            mod->BSIM3v1Adrout = value->rValue;
            mod->BSIM3v1AdroutGiven = TRUE;
            break;
        case  BSIM3v1A_MOD_DSUB:             
            mod->BSIM3v1Adsub = value->rValue;
            mod->BSIM3v1AdsubGiven = TRUE;
            break;
        case BSIM3v1A_MOD_VTH0:
            mod->BSIM3v1Avth0 = value->rValue;
            mod->BSIM3v1Avth0Given = TRUE;
            break;
        case BSIM3v1A_MOD_UA:
            mod->BSIM3v1Aua = value->rValue;
            mod->BSIM3v1AuaGiven = TRUE;
            break;
        case BSIM3v1A_MOD_UA1:
            mod->BSIM3v1Aua1 = value->rValue;
            mod->BSIM3v1Aua1Given = TRUE;
            break;
        case BSIM3v1A_MOD_UB:
            mod->BSIM3v1Aub = value->rValue;
            mod->BSIM3v1AubGiven = TRUE;
            break;
        case BSIM3v1A_MOD_UB1:
            mod->BSIM3v1Aub1 = value->rValue;
            mod->BSIM3v1Aub1Given = TRUE;
            break;
        case BSIM3v1A_MOD_UC:
            mod->BSIM3v1Auc = value->rValue;
            mod->BSIM3v1AucGiven = TRUE;
            break;
        case BSIM3v1A_MOD_UC1:
            mod->BSIM3v1Auc1 = value->rValue;
            mod->BSIM3v1Auc1Given = TRUE;
            break;
        case  BSIM3v1A_MOD_U0 :
            mod->BSIM3v1Au0 = value->rValue;
            mod->BSIM3v1Au0Given = TRUE;
	    if (mod->BSIM3v1Au0 > 1.0)
		mod->BSIM3v1Au0 *= 1.0e-4;
            break;
        case  BSIM3v1A_MOD_UTE :
            mod->BSIM3v1Aute = value->rValue;
            mod->BSIM3v1AuteGiven = TRUE;
            break;
        case BSIM3v1A_MOD_VOFF:
            mod->BSIM3v1Avoff = value->rValue;
            mod->BSIM3v1AvoffGiven = TRUE;
            break;
        case  BSIM3v1A_MOD_DELTA :
            mod->BSIM3v1Adelta = value->rValue;
            mod->BSIM3v1AdeltaGiven = TRUE;
            break;
        case BSIM3v1A_MOD_RDSW:
            mod->BSIM3v1Ardsw = value->rValue;
            mod->BSIM3v1ArdswGiven = TRUE;
            break;                     
        case BSIM3v1A_MOD_PRWG:
            mod->BSIM3v1Aprwg = value->rValue;
            mod->BSIM3v1AprwgGiven = TRUE;
            break;                     
        case BSIM3v1A_MOD_PRWB:
            mod->BSIM3v1Aprwb = value->rValue;
            mod->BSIM3v1AprwbGiven = TRUE;
            break;                     
        case BSIM3v1A_MOD_PRT:
            mod->BSIM3v1Aprt = value->rValue;
            mod->BSIM3v1AprtGiven = TRUE;
            break;                     
        case BSIM3v1A_MOD_ETA0:
            mod->BSIM3v1Aeta0 = value->rValue;
            mod->BSIM3v1Aeta0Given = TRUE;
            break;                 
        case BSIM3v1A_MOD_ETAB:
            mod->BSIM3v1Aetab = value->rValue;
            mod->BSIM3v1AetabGiven = TRUE;
            break;                 
        case BSIM3v1A_MOD_PCLM:
            mod->BSIM3v1Apclm = value->rValue;
            mod->BSIM3v1ApclmGiven = TRUE;
            break;                 
        case BSIM3v1A_MOD_PDIBL1:
            mod->BSIM3v1Apdibl1 = value->rValue;
            mod->BSIM3v1Apdibl1Given = TRUE;
            break;                 
        case BSIM3v1A_MOD_PDIBL2:
            mod->BSIM3v1Apdibl2 = value->rValue;
            mod->BSIM3v1Apdibl2Given = TRUE;
            break;                 
        case BSIM3v1A_MOD_PDIBLB:
            mod->BSIM3v1Apdiblb = value->rValue;
            mod->BSIM3v1ApdiblbGiven = TRUE;
            break;                 
        case BSIM3v1A_MOD_PSCBE1:
            mod->BSIM3v1Apscbe1 = value->rValue;
            mod->BSIM3v1Apscbe1Given = TRUE;
            break;                 
        case BSIM3v1A_MOD_PSCBE2:
            mod->BSIM3v1Apscbe2 = value->rValue;
            mod->BSIM3v1Apscbe2Given = TRUE;
            break;                 
        case BSIM3v1A_MOD_PVAG:
            mod->BSIM3v1Apvag = value->rValue;
            mod->BSIM3v1ApvagGiven = TRUE;
            break;                 
        case  BSIM3v1A_MOD_WR :
            mod->BSIM3v1Awr = value->rValue;
            mod->BSIM3v1AwrGiven = TRUE;
            break;
        case  BSIM3v1A_MOD_DWG :
            mod->BSIM3v1Adwg = value->rValue;
            mod->BSIM3v1AdwgGiven = TRUE;
            break;
        case  BSIM3v1A_MOD_DWB :
            mod->BSIM3v1Adwb = value->rValue;
            mod->BSIM3v1AdwbGiven = TRUE;
            break;
        case  BSIM3v1A_MOD_B0 :
            mod->BSIM3v1Ab0 = value->rValue;
            mod->BSIM3v1Ab0Given = TRUE;
            break;
        case  BSIM3v1A_MOD_B1 :
            mod->BSIM3v1Ab1 = value->rValue;
            mod->BSIM3v1Ab1Given = TRUE;
            break;
        case  BSIM3v1A_MOD_ALPHA0 :
            mod->BSIM3v1Aalpha0 = value->rValue;
            mod->BSIM3v1Aalpha0Given = TRUE;
            break;
        case  BSIM3v1A_MOD_BETA0 :
            mod->BSIM3v1Abeta0 = value->rValue;
            mod->BSIM3v1Abeta0Given = TRUE;
            break;

        case  BSIM3v1A_MOD_ELM :
            mod->BSIM3v1Aelm = value->rValue;
            mod->BSIM3v1AelmGiven = TRUE;
            break;
        case  BSIM3v1A_MOD_CGSL :
            mod->BSIM3v1Acgsl = value->rValue;
            mod->BSIM3v1AcgslGiven = TRUE;
            break;
        case  BSIM3v1A_MOD_CGDL :
            mod->BSIM3v1Acgdl = value->rValue;
            mod->BSIM3v1AcgdlGiven = TRUE;
            break;
        case  BSIM3v1A_MOD_CKAPPA :
            mod->BSIM3v1Ackappa = value->rValue;
            mod->BSIM3v1AckappaGiven = TRUE;
            break;
        case  BSIM3v1A_MOD_CF :
            mod->BSIM3v1Acf = value->rValue;
            mod->BSIM3v1AcfGiven = TRUE;
            break;
        case  BSIM3v1A_MOD_CLC :
            mod->BSIM3v1Aclc = value->rValue;
            mod->BSIM3v1AclcGiven = TRUE;
            break;
        case  BSIM3v1A_MOD_CLE :
            mod->BSIM3v1Acle = value->rValue;
            mod->BSIM3v1AcleGiven = TRUE;
            break;
        case  BSIM3v1A_MOD_DWC :
            mod->BSIM3v1Adwc = value->rValue;
            mod->BSIM3v1AdwcGiven = TRUE;
            break;
        case  BSIM3v1A_MOD_DLC :
            mod->BSIM3v1Adlc = value->rValue;
            mod->BSIM3v1AdlcGiven = TRUE;
            break;

	/* Length dependence */
        case  BSIM3v1A_MOD_LCDSC :
            mod->BSIM3v1Alcdsc = value->rValue;
            mod->BSIM3v1AlcdscGiven = TRUE;
            break;


        case  BSIM3v1A_MOD_LCDSCB :
            mod->BSIM3v1Alcdscb = value->rValue;
            mod->BSIM3v1AlcdscbGiven = TRUE;
            break;
        case  BSIM3v1A_MOD_LCDSCD :
            mod->BSIM3v1Alcdscd = value->rValue;
            mod->BSIM3v1AlcdscdGiven = TRUE;
            break;
        case  BSIM3v1A_MOD_LCIT :
            mod->BSIM3v1Alcit = value->rValue;
            mod->BSIM3v1AlcitGiven = TRUE;
            break;
        case  BSIM3v1A_MOD_LNFACTOR :
            mod->BSIM3v1Alnfactor = value->rValue;
            mod->BSIM3v1AlnfactorGiven = TRUE;
            break;
        case BSIM3v1A_MOD_LXJ:
            mod->BSIM3v1Alxj = value->rValue;
            mod->BSIM3v1AlxjGiven = TRUE;
            break;
        case BSIM3v1A_MOD_LVSAT:
            mod->BSIM3v1Alvsat = value->rValue;
            mod->BSIM3v1AlvsatGiven = TRUE;
            break;
        
        
        case BSIM3v1A_MOD_LA0:
            mod->BSIM3v1Ala0 = value->rValue;
            mod->BSIM3v1Ala0Given = TRUE;
            break;
        case BSIM3v1A_MOD_LAGS:
            mod->BSIM3v1Alags = value->rValue;
            mod->BSIM3v1AlagsGiven = TRUE;
            break;
        case BSIM3v1A_MOD_LA1:
            mod->BSIM3v1Ala1 = value->rValue;
            mod->BSIM3v1Ala1Given = TRUE;
            break;
        case BSIM3v1A_MOD_LA2:
            mod->BSIM3v1Ala2 = value->rValue;
            mod->BSIM3v1Ala2Given = TRUE;
            break;
        case BSIM3v1A_MOD_LAT:
            mod->BSIM3v1Alat = value->rValue;
            mod->BSIM3v1AlatGiven = TRUE;
            break;
        case BSIM3v1A_MOD_LKETA:
            mod->BSIM3v1Alketa = value->rValue;
            mod->BSIM3v1AlketaGiven = TRUE;
            break;    
        case BSIM3v1A_MOD_LNSUB:
            mod->BSIM3v1Alnsub = value->rValue;
            mod->BSIM3v1AlnsubGiven = TRUE;
            break;
        case BSIM3v1A_MOD_LNPEAK:
            mod->BSIM3v1Alnpeak = value->rValue;
            mod->BSIM3v1AlnpeakGiven = TRUE;
	    if (mod->BSIM3v1Alnpeak > 1.0e20)
		mod->BSIM3v1Alnpeak *= 1.0e-6;
            break;
        case BSIM3v1A_MOD_LNGATE:
            mod->BSIM3v1Alngate = value->rValue;
            mod->BSIM3v1AlngateGiven = TRUE;
	    if (mod->BSIM3v1Alngate > 1.0e23)
		mod->BSIM3v1Alngate *= 1.0e-6;
            break;
        case BSIM3v1A_MOD_LGAMMA1:
            mod->BSIM3v1Algamma1 = value->rValue;
            mod->BSIM3v1Algamma1Given = TRUE;
            break;
        case BSIM3v1A_MOD_LGAMMA2:
            mod->BSIM3v1Algamma2 = value->rValue;
            mod->BSIM3v1Algamma2Given = TRUE;
            break;
        case BSIM3v1A_MOD_LVBX:
            mod->BSIM3v1Alvbx = value->rValue;
            mod->BSIM3v1AlvbxGiven = TRUE;
            break;
        case BSIM3v1A_MOD_LVBM:
            mod->BSIM3v1Alvbm = value->rValue;
            mod->BSIM3v1AlvbmGiven = TRUE;
            break;
        case BSIM3v1A_MOD_LXT:
            mod->BSIM3v1Alxt = value->rValue;
            mod->BSIM3v1AlxtGiven = TRUE;
            break;
        case  BSIM3v1A_MOD_LK1:
            mod->BSIM3v1Alk1 = value->rValue;
            mod->BSIM3v1Alk1Given = TRUE;
            break;
        case  BSIM3v1A_MOD_LKT1:
            mod->BSIM3v1Alkt1 = value->rValue;
            mod->BSIM3v1Alkt1Given = TRUE;
            break;
        case  BSIM3v1A_MOD_LKT1L:
            mod->BSIM3v1Alkt1l = value->rValue;
            mod->BSIM3v1Alkt1lGiven = TRUE;
            break;
        case  BSIM3v1A_MOD_LKT2:
            mod->BSIM3v1Alkt2 = value->rValue;
            mod->BSIM3v1Alkt2Given = TRUE;
            break;
        case  BSIM3v1A_MOD_LK2:
            mod->BSIM3v1Alk2 = value->rValue;
            mod->BSIM3v1Alk2Given = TRUE;
            break;
        case  BSIM3v1A_MOD_LK3:
            mod->BSIM3v1Alk3 = value->rValue;
            mod->BSIM3v1Alk3Given = TRUE;
            break;
        case  BSIM3v1A_MOD_LK3B:
            mod->BSIM3v1Alk3b = value->rValue;
            mod->BSIM3v1Alk3bGiven = TRUE;
            break;
        case  BSIM3v1A_MOD_LNLX:
            mod->BSIM3v1Alnlx = value->rValue;
            mod->BSIM3v1AlnlxGiven = TRUE;
            break;
        case  BSIM3v1A_MOD_LW0:
            mod->BSIM3v1Alw0 = value->rValue;
            mod->BSIM3v1Alw0Given = TRUE;
            break;
        case  BSIM3v1A_MOD_LDVT0:               
            mod->BSIM3v1Aldvt0 = value->rValue;
            mod->BSIM3v1Aldvt0Given = TRUE;
            break;
        case  BSIM3v1A_MOD_LDVT1:             
            mod->BSIM3v1Aldvt1 = value->rValue;
            mod->BSIM3v1Aldvt1Given = TRUE;
            break;
        case  BSIM3v1A_MOD_LDVT2:             
            mod->BSIM3v1Aldvt2 = value->rValue;
            mod->BSIM3v1Aldvt2Given = TRUE;
            break;
        case  BSIM3v1A_MOD_LDVT0W:               
            mod->BSIM3v1Aldvt0w = value->rValue;
            mod->BSIM3v1Aldvt0Given = TRUE;
            break;
        case  BSIM3v1A_MOD_LDVT1W:             
            mod->BSIM3v1Aldvt1 = value->rValue;
            mod->BSIM3v1Aldvt1wGiven = TRUE;
            break;
        case  BSIM3v1A_MOD_LDVT2W:             
            mod->BSIM3v1Aldvt2 = value->rValue;
            mod->BSIM3v1Aldvt2wGiven = TRUE;
            break;
        case  BSIM3v1A_MOD_LDROUT:             
            mod->BSIM3v1Aldrout = value->rValue;
            mod->BSIM3v1AldroutGiven = TRUE;
            break;
        case  BSIM3v1A_MOD_LDSUB:             
            mod->BSIM3v1Aldsub = value->rValue;
            mod->BSIM3v1AldsubGiven = TRUE;
            break;
        case BSIM3v1A_MOD_LVTH0:
            mod->BSIM3v1Alvth0 = value->rValue;
            mod->BSIM3v1Alvth0Given = TRUE;
            break;
        case BSIM3v1A_MOD_LUA:
            mod->BSIM3v1Alua = value->rValue;
            mod->BSIM3v1AluaGiven = TRUE;
            break;
        case BSIM3v1A_MOD_LUA1:
            mod->BSIM3v1Alua1 = value->rValue;
            mod->BSIM3v1Alua1Given = TRUE;
            break;
        case BSIM3v1A_MOD_LUB:
            mod->BSIM3v1Alub = value->rValue;
            mod->BSIM3v1AlubGiven = TRUE;
            break;
        case BSIM3v1A_MOD_LUB1:
            mod->BSIM3v1Alub1 = value->rValue;
            mod->BSIM3v1Alub1Given = TRUE;
            break;
        case BSIM3v1A_MOD_LUC:
            mod->BSIM3v1Aluc = value->rValue;
            mod->BSIM3v1AlucGiven = TRUE;
            break;
        case BSIM3v1A_MOD_LUC1:
            mod->BSIM3v1Aluc1 = value->rValue;
            mod->BSIM3v1Aluc1Given = TRUE;
            break;
        case  BSIM3v1A_MOD_LU0 :
            mod->BSIM3v1Alu0 = value->rValue;
            mod->BSIM3v1Alu0Given = TRUE;
	    if (mod->BSIM3v1Alu0 > 1.0)
		mod->BSIM3v1Alu0 *= 1.0e-4;
            break;
        case  BSIM3v1A_MOD_LUTE :
            mod->BSIM3v1Alute = value->rValue;
            mod->BSIM3v1AluteGiven = TRUE;
            break;
        case BSIM3v1A_MOD_LVOFF:
            mod->BSIM3v1Alvoff = value->rValue;
            mod->BSIM3v1AlvoffGiven = TRUE;
            break;
        case  BSIM3v1A_MOD_LDELTA :
            mod->BSIM3v1Aldelta = value->rValue;
            mod->BSIM3v1AldeltaGiven = TRUE;
            break;
        case BSIM3v1A_MOD_LRDSW:
            mod->BSIM3v1Alrdsw = value->rValue;
            mod->BSIM3v1AlrdswGiven = TRUE;
            break;                     
        case BSIM3v1A_MOD_LPRWB:
            mod->BSIM3v1Alprwb = value->rValue;
            mod->BSIM3v1AlprwbGiven = TRUE;
            break;                     
        case BSIM3v1A_MOD_LPRWG:
            mod->BSIM3v1Alprwg = value->rValue;
            mod->BSIM3v1AlprwgGiven = TRUE;
            break;                     
        case BSIM3v1A_MOD_LPRT:
            mod->BSIM3v1Alprt = value->rValue;
            mod->BSIM3v1AlprtGiven = TRUE;
            break;                     
        case BSIM3v1A_MOD_LETA0:
            mod->BSIM3v1Aleta0 = value->rValue;
            mod->BSIM3v1Aleta0Given = TRUE;
            break;                 
        case BSIM3v1A_MOD_LETAB:
            mod->BSIM3v1Aletab = value->rValue;
            mod->BSIM3v1AletabGiven = TRUE;
            break;                 
        case BSIM3v1A_MOD_LPCLM:
            mod->BSIM3v1Alpclm = value->rValue;
            mod->BSIM3v1AlpclmGiven = TRUE;
            break;                 
        case BSIM3v1A_MOD_LPDIBL1:
            mod->BSIM3v1Alpdibl1 = value->rValue;
            mod->BSIM3v1Alpdibl1Given = TRUE;
            break;                 
        case BSIM3v1A_MOD_LPDIBL2:
            mod->BSIM3v1Alpdibl2 = value->rValue;
            mod->BSIM3v1Alpdibl2Given = TRUE;
            break;                 
        case BSIM3v1A_MOD_LPDIBLB:
            mod->BSIM3v1Alpdiblb = value->rValue;
            mod->BSIM3v1AlpdiblbGiven = TRUE;
            break;                 
        case BSIM3v1A_MOD_LPSCBE1:
            mod->BSIM3v1Alpscbe1 = value->rValue;
            mod->BSIM3v1Alpscbe1Given = TRUE;
            break;                 
        case BSIM3v1A_MOD_LPSCBE2:
            mod->BSIM3v1Alpscbe2 = value->rValue;
            mod->BSIM3v1Alpscbe2Given = TRUE;
            break;                 
        case BSIM3v1A_MOD_LPVAG:
            mod->BSIM3v1Alpvag = value->rValue;
            mod->BSIM3v1AlpvagGiven = TRUE;
            break;                 
        case  BSIM3v1A_MOD_LWR :
            mod->BSIM3v1Alwr = value->rValue;
            mod->BSIM3v1AlwrGiven = TRUE;
            break;
        case  BSIM3v1A_MOD_LDWG :
            mod->BSIM3v1Aldwg = value->rValue;
            mod->BSIM3v1AldwgGiven = TRUE;
            break;
        case  BSIM3v1A_MOD_LDWB :
            mod->BSIM3v1Aldwb = value->rValue;
            mod->BSIM3v1AldwbGiven = TRUE;
            break;
        case  BSIM3v1A_MOD_LB0 :
            mod->BSIM3v1Alb0 = value->rValue;
            mod->BSIM3v1Alb0Given = TRUE;
            break;
        case  BSIM3v1A_MOD_LB1 :
            mod->BSIM3v1Alb1 = value->rValue;
            mod->BSIM3v1Alb1Given = TRUE;
            break;
        case  BSIM3v1A_MOD_LALPHA0 :
            mod->BSIM3v1Alalpha0 = value->rValue;
            mod->BSIM3v1Alalpha0Given = TRUE;
            break;
        case  BSIM3v1A_MOD_LBETA0 :
            mod->BSIM3v1Albeta0 = value->rValue;
            mod->BSIM3v1Albeta0Given = TRUE;
            break;

        case  BSIM3v1A_MOD_LELM :
            mod->BSIM3v1Alelm = value->rValue;
            mod->BSIM3v1AlelmGiven = TRUE;
            break;
        case  BSIM3v1A_MOD_LCGSL :
            mod->BSIM3v1Alcgsl = value->rValue;
            mod->BSIM3v1AlcgslGiven = TRUE;
            break;
        case  BSIM3v1A_MOD_LCGDL :
            mod->BSIM3v1Alcgdl = value->rValue;
            mod->BSIM3v1AlcgdlGiven = TRUE;
            break;
        case  BSIM3v1A_MOD_LCKAPPA :
            mod->BSIM3v1Alckappa = value->rValue;
            mod->BSIM3v1AlckappaGiven = TRUE;
            break;
        case  BSIM3v1A_MOD_LCF :
            mod->BSIM3v1Alcf = value->rValue;
            mod->BSIM3v1AlcfGiven = TRUE;
            break;
        case  BSIM3v1A_MOD_LCLC :
            mod->BSIM3v1Alclc = value->rValue;
            mod->BSIM3v1AlclcGiven = TRUE;
            break;
        case  BSIM3v1A_MOD_LCLE :
            mod->BSIM3v1Alcle = value->rValue;
            mod->BSIM3v1AlcleGiven = TRUE;
            break;

	/* Width dependence */
        case  BSIM3v1A_MOD_WCDSC :
            mod->BSIM3v1Awcdsc = value->rValue;
            mod->BSIM3v1AwcdscGiven = TRUE;
            break;
       
       
         case  BSIM3v1A_MOD_WCDSCB :
            mod->BSIM3v1Awcdscb = value->rValue;
            mod->BSIM3v1AwcdscbGiven = TRUE;
            break;
         case  BSIM3v1A_MOD_WCDSCD :
            mod->BSIM3v1Awcdscd = value->rValue;
            mod->BSIM3v1AwcdscdGiven = TRUE;
            break;
        case  BSIM3v1A_MOD_WCIT :
            mod->BSIM3v1Awcit = value->rValue;
            mod->BSIM3v1AwcitGiven = TRUE;
            break;
        case  BSIM3v1A_MOD_WNFACTOR :
            mod->BSIM3v1Awnfactor = value->rValue;
            mod->BSIM3v1AwnfactorGiven = TRUE;
            break;
        case BSIM3v1A_MOD_WXJ:
            mod->BSIM3v1Awxj = value->rValue;
            mod->BSIM3v1AwxjGiven = TRUE;
            break;
        case BSIM3v1A_MOD_WVSAT:
            mod->BSIM3v1Awvsat = value->rValue;
            mod->BSIM3v1AwvsatGiven = TRUE;
            break;


        case BSIM3v1A_MOD_WA0:
            mod->BSIM3v1Awa0 = value->rValue;
            mod->BSIM3v1Awa0Given = TRUE;
            break;
        case BSIM3v1A_MOD_WAGS:
            mod->BSIM3v1Awags = value->rValue;
            mod->BSIM3v1AwagsGiven = TRUE;
            break;
        case BSIM3v1A_MOD_WA1:
            mod->BSIM3v1Awa1 = value->rValue;
            mod->BSIM3v1Awa1Given = TRUE;
            break;
        case BSIM3v1A_MOD_WA2:
            mod->BSIM3v1Awa2 = value->rValue;
            mod->BSIM3v1Awa2Given = TRUE;
            break;
        case BSIM3v1A_MOD_WAT:
            mod->BSIM3v1Awat = value->rValue;
            mod->BSIM3v1AwatGiven = TRUE;
            break;
        case BSIM3v1A_MOD_WKETA:
            mod->BSIM3v1Awketa = value->rValue;
            mod->BSIM3v1AwketaGiven = TRUE;
            break;    
        case BSIM3v1A_MOD_WNSUB:
            mod->BSIM3v1Awnsub = value->rValue;
            mod->BSIM3v1AwnsubGiven = TRUE;
            break;
        case BSIM3v1A_MOD_WNPEAK:
            mod->BSIM3v1Awnpeak = value->rValue;
            mod->BSIM3v1AwnpeakGiven = TRUE;
	    if (mod->BSIM3v1Awnpeak > 1.0e20)
		mod->BSIM3v1Awnpeak *= 1.0e-6;
            break;
        case BSIM3v1A_MOD_WNGATE:
            mod->BSIM3v1Awngate = value->rValue;
            mod->BSIM3v1AwngateGiven = TRUE;
	    if (mod->BSIM3v1Awngate > 1.0e23)
		mod->BSIM3v1Awngate *= 1.0e-6;
            break;
        case BSIM3v1A_MOD_WGAMMA1:
            mod->BSIM3v1Awgamma1 = value->rValue;
            mod->BSIM3v1Awgamma1Given = TRUE;
            break;
        case BSIM3v1A_MOD_WGAMMA2:
            mod->BSIM3v1Awgamma2 = value->rValue;
            mod->BSIM3v1Awgamma2Given = TRUE;
            break;
        case BSIM3v1A_MOD_WVBX:
            mod->BSIM3v1Awvbx = value->rValue;
            mod->BSIM3v1AwvbxGiven = TRUE;
            break;
        case BSIM3v1A_MOD_WVBM:
            mod->BSIM3v1Awvbm = value->rValue;
            mod->BSIM3v1AwvbmGiven = TRUE;
            break;
        case BSIM3v1A_MOD_WXT:
            mod->BSIM3v1Awxt = value->rValue;
            mod->BSIM3v1AwxtGiven = TRUE;
            break;
        case  BSIM3v1A_MOD_WK1:
            mod->BSIM3v1Awk1 = value->rValue;
            mod->BSIM3v1Awk1Given = TRUE;
            break;
        case  BSIM3v1A_MOD_WKT1:
            mod->BSIM3v1Awkt1 = value->rValue;
            mod->BSIM3v1Awkt1Given = TRUE;
            break;
        case  BSIM3v1A_MOD_WKT1L:
            mod->BSIM3v1Awkt1l = value->rValue;
            mod->BSIM3v1Awkt1lGiven = TRUE;
            break;
        case  BSIM3v1A_MOD_WKT2:
            mod->BSIM3v1Awkt2 = value->rValue;
            mod->BSIM3v1Awkt2Given = TRUE;
            break;
        case  BSIM3v1A_MOD_WK2:
            mod->BSIM3v1Awk2 = value->rValue;
            mod->BSIM3v1Awk2Given = TRUE;
            break;
        case  BSIM3v1A_MOD_WK3:
            mod->BSIM3v1Awk3 = value->rValue;
            mod->BSIM3v1Awk3Given = TRUE;
            break;
        case  BSIM3v1A_MOD_WK3B:
            mod->BSIM3v1Awk3b = value->rValue;
            mod->BSIM3v1Awk3bGiven = TRUE;
            break;
        case  BSIM3v1A_MOD_WNLX:
            mod->BSIM3v1Awnlx = value->rValue;
            mod->BSIM3v1AwnlxGiven = TRUE;
            break;
        case  BSIM3v1A_MOD_WW0:
            mod->BSIM3v1Aww0 = value->rValue;
            mod->BSIM3v1Aww0Given = TRUE;
            break;
        case  BSIM3v1A_MOD_WDVT0:               
            mod->BSIM3v1Awdvt0 = value->rValue;
            mod->BSIM3v1Awdvt0Given = TRUE;
            break;
        case  BSIM3v1A_MOD_WDVT1:             
            mod->BSIM3v1Awdvt1 = value->rValue;
            mod->BSIM3v1Awdvt1Given = TRUE;
            break;
        case  BSIM3v1A_MOD_WDVT2:             
            mod->BSIM3v1Awdvt2 = value->rValue;
            mod->BSIM3v1Awdvt2Given = TRUE;
            break;
        case  BSIM3v1A_MOD_WDVT0W:               
            mod->BSIM3v1Awdvt0w = value->rValue;
            mod->BSIM3v1Awdvt0wGiven = TRUE;
            break;
        case  BSIM3v1A_MOD_WDVT1W:             
            mod->BSIM3v1Awdvt1w = value->rValue;
            mod->BSIM3v1Awdvt1wGiven = TRUE;
            break;
        case  BSIM3v1A_MOD_WDVT2W:             
            mod->BSIM3v1Awdvt2w = value->rValue;
            mod->BSIM3v1Awdvt2wGiven = TRUE;
            break;
        case  BSIM3v1A_MOD_WDROUT:             
            mod->BSIM3v1Awdrout = value->rValue;
            mod->BSIM3v1AwdroutGiven = TRUE;
            break;
        case  BSIM3v1A_MOD_WDSUB:             
            mod->BSIM3v1Awdsub = value->rValue;
            mod->BSIM3v1AwdsubGiven = TRUE;
            break;
        case BSIM3v1A_MOD_WVTH0:
            mod->BSIM3v1Awvth0 = value->rValue;
            mod->BSIM3v1Awvth0Given = TRUE;
            break;
        case BSIM3v1A_MOD_WUA:
            mod->BSIM3v1Awua = value->rValue;
            mod->BSIM3v1AwuaGiven = TRUE;
            break;
        case BSIM3v1A_MOD_WUA1:
            mod->BSIM3v1Awua1 = value->rValue;
            mod->BSIM3v1Awua1Given = TRUE;
            break;
        case BSIM3v1A_MOD_WUB:
            mod->BSIM3v1Awub = value->rValue;
            mod->BSIM3v1AwubGiven = TRUE;
            break;
        case BSIM3v1A_MOD_WUB1:
            mod->BSIM3v1Awub1 = value->rValue;
            mod->BSIM3v1Awub1Given = TRUE;
            break;
        case BSIM3v1A_MOD_WUC:
            mod->BSIM3v1Awuc = value->rValue;
            mod->BSIM3v1AwucGiven = TRUE;
            break;
        case BSIM3v1A_MOD_WUC1:
            mod->BSIM3v1Awuc1 = value->rValue;
            mod->BSIM3v1Awuc1Given = TRUE;
            break;
        case  BSIM3v1A_MOD_WU0 :
            mod->BSIM3v1Awu0 = value->rValue;
            mod->BSIM3v1Awu0Given = TRUE;
	    if (mod->BSIM3v1Awu0 > 1.0)
		mod->BSIM3v1Awu0 *= 1.0e-4;
            break;
        case  BSIM3v1A_MOD_WUTE :
            mod->BSIM3v1Awute = value->rValue;
            mod->BSIM3v1AwuteGiven = TRUE;
            break;
        case BSIM3v1A_MOD_WVOFF:
            mod->BSIM3v1Awvoff = value->rValue;
            mod->BSIM3v1AwvoffGiven = TRUE;
            break;
        case  BSIM3v1A_MOD_WDELTA :
            mod->BSIM3v1Awdelta = value->rValue;
            mod->BSIM3v1AwdeltaGiven = TRUE;
            break;
        case BSIM3v1A_MOD_WRDSW:
            mod->BSIM3v1Awrdsw = value->rValue;
            mod->BSIM3v1AwrdswGiven = TRUE;
            break;                     
        case BSIM3v1A_MOD_WPRWB:
            mod->BSIM3v1Awprwb = value->rValue;
            mod->BSIM3v1AwprwbGiven = TRUE;
            break;                     
        case BSIM3v1A_MOD_WPRWG:
            mod->BSIM3v1Awprwg = value->rValue;
            mod->BSIM3v1AwprwgGiven = TRUE;
            break;                     
        case BSIM3v1A_MOD_WPRT:
            mod->BSIM3v1Awprt = value->rValue;
            mod->BSIM3v1AwprtGiven = TRUE;
            break;                     
        case BSIM3v1A_MOD_WETA0:
            mod->BSIM3v1Aweta0 = value->rValue;
            mod->BSIM3v1Aweta0Given = TRUE;
            break;                 
        case BSIM3v1A_MOD_WETAB:
            mod->BSIM3v1Awetab = value->rValue;
            mod->BSIM3v1AwetabGiven = TRUE;
            break;                 
        case BSIM3v1A_MOD_WPCLM:
            mod->BSIM3v1Awpclm = value->rValue;
            mod->BSIM3v1AwpclmGiven = TRUE;
            break;                 
        case BSIM3v1A_MOD_WPDIBL1:
            mod->BSIM3v1Awpdibl1 = value->rValue;
            mod->BSIM3v1Awpdibl1Given = TRUE;
            break;                 
        case BSIM3v1A_MOD_WPDIBL2:
            mod->BSIM3v1Awpdibl2 = value->rValue;
            mod->BSIM3v1Awpdibl2Given = TRUE;
            break;                 
        case BSIM3v1A_MOD_WPDIBLB:
            mod->BSIM3v1Awpdiblb = value->rValue;
            mod->BSIM3v1AwpdiblbGiven = TRUE;
            break;                 
        case BSIM3v1A_MOD_WPSCBE1:
            mod->BSIM3v1Awpscbe1 = value->rValue;
            mod->BSIM3v1Awpscbe1Given = TRUE;
            break;                 
        case BSIM3v1A_MOD_WPSCBE2:
            mod->BSIM3v1Awpscbe2 = value->rValue;
            mod->BSIM3v1Awpscbe2Given = TRUE;
            break;                 
        case BSIM3v1A_MOD_WPVAG:
            mod->BSIM3v1Awpvag = value->rValue;
            mod->BSIM3v1AwpvagGiven = TRUE;
            break;                 
        case  BSIM3v1A_MOD_WWR :
            mod->BSIM3v1Awwr = value->rValue;
            mod->BSIM3v1AwwrGiven = TRUE;
            break;
        case  BSIM3v1A_MOD_WDWG :
            mod->BSIM3v1Awdwg = value->rValue;
            mod->BSIM3v1AwdwgGiven = TRUE;
            break;
        case  BSIM3v1A_MOD_WDWB :
            mod->BSIM3v1Awdwb = value->rValue;
            mod->BSIM3v1AwdwbGiven = TRUE;
            break;
        case  BSIM3v1A_MOD_WB0 :
            mod->BSIM3v1Awb0 = value->rValue;
            mod->BSIM3v1Awb0Given = TRUE;
            break;
        case  BSIM3v1A_MOD_WB1 :
            mod->BSIM3v1Awb1 = value->rValue;
            mod->BSIM3v1Awb1Given = TRUE;
            break;
        case  BSIM3v1A_MOD_WALPHA0 :
            mod->BSIM3v1Awalpha0 = value->rValue;
            mod->BSIM3v1Awalpha0Given = TRUE;
            break;
        case  BSIM3v1A_MOD_WBETA0 :
            mod->BSIM3v1Awbeta0 = value->rValue;
            mod->BSIM3v1Awbeta0Given = TRUE;
            break;

        case  BSIM3v1A_MOD_WELM :
            mod->BSIM3v1Awelm = value->rValue;
            mod->BSIM3v1AwelmGiven = TRUE;
            break;
        case  BSIM3v1A_MOD_WCGSL :
            mod->BSIM3v1Awcgsl = value->rValue;
            mod->BSIM3v1AwcgslGiven = TRUE;
            break;
        case  BSIM3v1A_MOD_WCGDL :
            mod->BSIM3v1Awcgdl = value->rValue;
            mod->BSIM3v1AwcgdlGiven = TRUE;
            break;
        case  BSIM3v1A_MOD_WCKAPPA :
            mod->BSIM3v1Awckappa = value->rValue;
            mod->BSIM3v1AwckappaGiven = TRUE;
            break;
        case  BSIM3v1A_MOD_WCF :
            mod->BSIM3v1Awcf = value->rValue;
            mod->BSIM3v1AwcfGiven = TRUE;
            break;
        case  BSIM3v1A_MOD_WCLC :
            mod->BSIM3v1Awclc = value->rValue;
            mod->BSIM3v1AwclcGiven = TRUE;
            break;
        case  BSIM3v1A_MOD_WCLE :
            mod->BSIM3v1Awcle = value->rValue;
            mod->BSIM3v1AwcleGiven = TRUE;
            break;

	/* Cross-term dependence */
        case  BSIM3v1A_MOD_PCDSC :
            mod->BSIM3v1Apcdsc = value->rValue;
            mod->BSIM3v1ApcdscGiven = TRUE;
            break;


        case  BSIM3v1A_MOD_PCDSCB :
            mod->BSIM3v1Apcdscb = value->rValue;
            mod->BSIM3v1ApcdscbGiven = TRUE;
            break;
        case  BSIM3v1A_MOD_PCDSCD :
            mod->BSIM3v1Apcdscd = value->rValue;
            mod->BSIM3v1ApcdscdGiven = TRUE;
            break;
        case  BSIM3v1A_MOD_PCIT :
            mod->BSIM3v1Apcit = value->rValue;
            mod->BSIM3v1ApcitGiven = TRUE;
            break;
        case  BSIM3v1A_MOD_PNFACTOR :
            mod->BSIM3v1Apnfactor = value->rValue;
            mod->BSIM3v1ApnfactorGiven = TRUE;
            break;
        case BSIM3v1A_MOD_PXJ:
            mod->BSIM3v1Apxj = value->rValue;
            mod->BSIM3v1ApxjGiven = TRUE;
            break;
        case BSIM3v1A_MOD_PVSAT:
            mod->BSIM3v1Apvsat = value->rValue;
            mod->BSIM3v1ApvsatGiven = TRUE;
            break;


        case BSIM3v1A_MOD_PA0:
            mod->BSIM3v1Apa0 = value->rValue;
            mod->BSIM3v1Apa0Given = TRUE;
            break;
        case BSIM3v1A_MOD_PAGS:
            mod->BSIM3v1Apags = value->rValue;
            mod->BSIM3v1ApagsGiven = TRUE;
            break;
        case BSIM3v1A_MOD_PA1:
            mod->BSIM3v1Apa1 = value->rValue;
            mod->BSIM3v1Apa1Given = TRUE;
            break;
        case BSIM3v1A_MOD_PA2:
            mod->BSIM3v1Apa2 = value->rValue;
            mod->BSIM3v1Apa2Given = TRUE;
            break;
        case BSIM3v1A_MOD_PAT:
            mod->BSIM3v1Apat = value->rValue;
            mod->BSIM3v1ApatGiven = TRUE;
            break;
        case BSIM3v1A_MOD_PKETA:
            mod->BSIM3v1Apketa = value->rValue;
            mod->BSIM3v1ApketaGiven = TRUE;
            break;    
        case BSIM3v1A_MOD_PNSUB:
            mod->BSIM3v1Apnsub = value->rValue;
            mod->BSIM3v1ApnsubGiven = TRUE;
            break;
        case BSIM3v1A_MOD_PNPEAK:
            mod->BSIM3v1Apnpeak = value->rValue;
            mod->BSIM3v1ApnpeakGiven = TRUE;
	    if (mod->BSIM3v1Apnpeak > 1.0e20)
		mod->BSIM3v1Apnpeak *= 1.0e-6;
            break;
        case BSIM3v1A_MOD_PNGATE:
            mod->BSIM3v1Apngate = value->rValue;
            mod->BSIM3v1ApngateGiven = TRUE;
	    if (mod->BSIM3v1Apngate > 1.0e23)
		mod->BSIM3v1Apngate *= 1.0e-6;
            break;
        case BSIM3v1A_MOD_PGAMMA1:
            mod->BSIM3v1Apgamma1 = value->rValue;
            mod->BSIM3v1Apgamma1Given = TRUE;
            break;
        case BSIM3v1A_MOD_PGAMMA2:
            mod->BSIM3v1Apgamma2 = value->rValue;
            mod->BSIM3v1Apgamma2Given = TRUE;
            break;
        case BSIM3v1A_MOD_PVBX:
            mod->BSIM3v1Apvbx = value->rValue;
            mod->BSIM3v1ApvbxGiven = TRUE;
            break;
        case BSIM3v1A_MOD_PVBM:
            mod->BSIM3v1Apvbm = value->rValue;
            mod->BSIM3v1ApvbmGiven = TRUE;
            break;
        case BSIM3v1A_MOD_PXT:
            mod->BSIM3v1Apxt = value->rValue;
            mod->BSIM3v1ApxtGiven = TRUE;
            break;
        case  BSIM3v1A_MOD_PK1:
            mod->BSIM3v1Apk1 = value->rValue;
            mod->BSIM3v1Apk1Given = TRUE;
            break;
        case  BSIM3v1A_MOD_PKT1:
            mod->BSIM3v1Apkt1 = value->rValue;
            mod->BSIM3v1Apkt1Given = TRUE;
            break;
        case  BSIM3v1A_MOD_PKT1L:
            mod->BSIM3v1Apkt1l = value->rValue;
            mod->BSIM3v1Apkt1lGiven = TRUE;
            break;
        case  BSIM3v1A_MOD_PKT2:
            mod->BSIM3v1Apkt2 = value->rValue;
            mod->BSIM3v1Apkt2Given = TRUE;
            break;
        case  BSIM3v1A_MOD_PK2:
            mod->BSIM3v1Apk2 = value->rValue;
            mod->BSIM3v1Apk2Given = TRUE;
            break;
        case  BSIM3v1A_MOD_PK3:
            mod->BSIM3v1Apk3 = value->rValue;
            mod->BSIM3v1Apk3Given = TRUE;
            break;
        case  BSIM3v1A_MOD_PK3B:
            mod->BSIM3v1Apk3b = value->rValue;
            mod->BSIM3v1Apk3bGiven = TRUE;
            break;
        case  BSIM3v1A_MOD_PNLX:
            mod->BSIM3v1Apnlx = value->rValue;
            mod->BSIM3v1ApnlxGiven = TRUE;
            break;
        case  BSIM3v1A_MOD_PW0:
            mod->BSIM3v1Apw0 = value->rValue;
            mod->BSIM3v1Apw0Given = TRUE;
            break;
        case  BSIM3v1A_MOD_PDVT0:               
            mod->BSIM3v1Apdvt0 = value->rValue;
            mod->BSIM3v1Apdvt0Given = TRUE;
            break;
        case  BSIM3v1A_MOD_PDVT1:             
            mod->BSIM3v1Apdvt1 = value->rValue;
            mod->BSIM3v1Apdvt1Given = TRUE;
            break;
        case  BSIM3v1A_MOD_PDVT2:             
            mod->BSIM3v1Apdvt2 = value->rValue;
            mod->BSIM3v1Apdvt2Given = TRUE;
            break;
        case  BSIM3v1A_MOD_PDVT0W:               
            mod->BSIM3v1Apdvt0w = value->rValue;
            mod->BSIM3v1Apdvt0wGiven = TRUE;
            break;
        case  BSIM3v1A_MOD_PDVT1W:             
            mod->BSIM3v1Apdvt1w = value->rValue;
            mod->BSIM3v1Apdvt1wGiven = TRUE;
            break;
        case  BSIM3v1A_MOD_PDVT2W:             
            mod->BSIM3v1Apdvt2w = value->rValue;
            mod->BSIM3v1Apdvt2wGiven = TRUE;
            break;
        case  BSIM3v1A_MOD_PDROUT:             
            mod->BSIM3v1Apdrout = value->rValue;
            mod->BSIM3v1ApdroutGiven = TRUE;
            break;
        case  BSIM3v1A_MOD_PDSUB:             
            mod->BSIM3v1Apdsub = value->rValue;
            mod->BSIM3v1ApdsubGiven = TRUE;
            break;
        case BSIM3v1A_MOD_PVTH0:
            mod->BSIM3v1Apvth0 = value->rValue;
            mod->BSIM3v1Apvth0Given = TRUE;
            break;
        case BSIM3v1A_MOD_PUA:
            mod->BSIM3v1Apua = value->rValue;
            mod->BSIM3v1ApuaGiven = TRUE;
            break;
        case BSIM3v1A_MOD_PUA1:
            mod->BSIM3v1Apua1 = value->rValue;
            mod->BSIM3v1Apua1Given = TRUE;
            break;
        case BSIM3v1A_MOD_PUB:
            mod->BSIM3v1Apub = value->rValue;
            mod->BSIM3v1ApubGiven = TRUE;
            break;
        case BSIM3v1A_MOD_PUB1:
            mod->BSIM3v1Apub1 = value->rValue;
            mod->BSIM3v1Apub1Given = TRUE;
            break;
        case BSIM3v1A_MOD_PUC:
            mod->BSIM3v1Apuc = value->rValue;
            mod->BSIM3v1ApucGiven = TRUE;
            break;
        case BSIM3v1A_MOD_PUC1:
            mod->BSIM3v1Apuc1 = value->rValue;
            mod->BSIM3v1Apuc1Given = TRUE;
            break;
        case  BSIM3v1A_MOD_PU0 :
            mod->BSIM3v1Apu0 = value->rValue;
            mod->BSIM3v1Apu0Given = TRUE;
	    if (mod->BSIM3v1Apu0 > 1.0)
		mod->BSIM3v1Apu0 *= 1.0e-4;
            break;
        case  BSIM3v1A_MOD_PUTE :
            mod->BSIM3v1Apute = value->rValue;
            mod->BSIM3v1AputeGiven = TRUE;
            break;
        case BSIM3v1A_MOD_PVOFF:
            mod->BSIM3v1Apvoff = value->rValue;
            mod->BSIM3v1ApvoffGiven = TRUE;
            break;
        case  BSIM3v1A_MOD_PDELTA :
            mod->BSIM3v1Apdelta = value->rValue;
            mod->BSIM3v1ApdeltaGiven = TRUE;
            break;
        case BSIM3v1A_MOD_PRDSW:
            mod->BSIM3v1Aprdsw = value->rValue;
            mod->BSIM3v1AprdswGiven = TRUE;
            break;                     
        case BSIM3v1A_MOD_PPRWB:
            mod->BSIM3v1Apprwb = value->rValue;
            mod->BSIM3v1ApprwbGiven = TRUE;
            break;                     
        case BSIM3v1A_MOD_PPRWG:
            mod->BSIM3v1Apprwg = value->rValue;
            mod->BSIM3v1ApprwgGiven = TRUE;
            break;                     
        case BSIM3v1A_MOD_PPRT:
            mod->BSIM3v1Apprt = value->rValue;
            mod->BSIM3v1ApprtGiven = TRUE;
            break;                     
        case BSIM3v1A_MOD_PETA0:
            mod->BSIM3v1Apeta0 = value->rValue;
            mod->BSIM3v1Apeta0Given = TRUE;
            break;                 
        case BSIM3v1A_MOD_PETAB:
            mod->BSIM3v1Apetab = value->rValue;
            mod->BSIM3v1ApetabGiven = TRUE;
            break;                 
        case BSIM3v1A_MOD_PPCLM:
            mod->BSIM3v1Appclm = value->rValue;
            mod->BSIM3v1AppclmGiven = TRUE;
            break;                 
        case BSIM3v1A_MOD_PPDIBL1:
            mod->BSIM3v1Appdibl1 = value->rValue;
            mod->BSIM3v1Appdibl1Given = TRUE;
            break;                 
        case BSIM3v1A_MOD_PPDIBL2:
            mod->BSIM3v1Appdibl2 = value->rValue;
            mod->BSIM3v1Appdibl2Given = TRUE;
            break;                 
        case BSIM3v1A_MOD_PPDIBLB:
            mod->BSIM3v1Appdiblb = value->rValue;
            mod->BSIM3v1AppdiblbGiven = TRUE;
            break;                 
        case BSIM3v1A_MOD_PPSCBE1:
            mod->BSIM3v1Appscbe1 = value->rValue;
            mod->BSIM3v1Appscbe1Given = TRUE;
            break;                 
        case BSIM3v1A_MOD_PPSCBE2:
            mod->BSIM3v1Appscbe2 = value->rValue;
            mod->BSIM3v1Appscbe2Given = TRUE;
            break;                 
        case BSIM3v1A_MOD_PPVAG:
            mod->BSIM3v1Appvag = value->rValue;
            mod->BSIM3v1AppvagGiven = TRUE;
            break;                 
        case  BSIM3v1A_MOD_PWR :
            mod->BSIM3v1Apwr = value->rValue;
            mod->BSIM3v1ApwrGiven = TRUE;
            break;
        case  BSIM3v1A_MOD_PDWG :
            mod->BSIM3v1Apdwg = value->rValue;
            mod->BSIM3v1ApdwgGiven = TRUE;
            break;
        case  BSIM3v1A_MOD_PDWB :
            mod->BSIM3v1Apdwb = value->rValue;
            mod->BSIM3v1ApdwbGiven = TRUE;
            break;
        case  BSIM3v1A_MOD_PB0 :
            mod->BSIM3v1Apb0 = value->rValue;
            mod->BSIM3v1Apb0Given = TRUE;
            break;
        case  BSIM3v1A_MOD_PB1 :
            mod->BSIM3v1Apb1 = value->rValue;
            mod->BSIM3v1Apb1Given = TRUE;
            break;
        case  BSIM3v1A_MOD_PALPHA0 :
            mod->BSIM3v1Apalpha0 = value->rValue;
            mod->BSIM3v1Apalpha0Given = TRUE;
            break;
        case  BSIM3v1A_MOD_PBETA0 :
            mod->BSIM3v1Apbeta0 = value->rValue;
            mod->BSIM3v1Apbeta0Given = TRUE;
            break;

        case  BSIM3v1A_MOD_PELM :
            mod->BSIM3v1Apelm = value->rValue;
            mod->BSIM3v1ApelmGiven = TRUE;
            break;
        case  BSIM3v1A_MOD_PCGSL :
            mod->BSIM3v1Apcgsl = value->rValue;
            mod->BSIM3v1ApcgslGiven = TRUE;
            break;
        case  BSIM3v1A_MOD_PCGDL :
            mod->BSIM3v1Apcgdl = value->rValue;
            mod->BSIM3v1ApcgdlGiven = TRUE;
            break;
        case  BSIM3v1A_MOD_PCKAPPA :
            mod->BSIM3v1Apckappa = value->rValue;
            mod->BSIM3v1ApckappaGiven = TRUE;
            break;
        case  BSIM3v1A_MOD_PCF :
            mod->BSIM3v1Apcf = value->rValue;
            mod->BSIM3v1ApcfGiven = TRUE;
            break;
        case  BSIM3v1A_MOD_PCLC :
            mod->BSIM3v1Apclc = value->rValue;
            mod->BSIM3v1ApclcGiven = TRUE;
            break;
        case  BSIM3v1A_MOD_PCLE :
            mod->BSIM3v1Apcle = value->rValue;
            mod->BSIM3v1ApcleGiven = TRUE;
            break;

        case  BSIM3v1A_MOD_TNOM :
            mod->BSIM3v1Atnom = value->rValue + 273.15;
            mod->BSIM3v1AtnomGiven = TRUE;
            break;
        case  BSIM3v1A_MOD_CGSO :
            mod->BSIM3v1Acgso = value->rValue;
            mod->BSIM3v1AcgsoGiven = TRUE;
            break;
        case  BSIM3v1A_MOD_CGDO :
            mod->BSIM3v1Acgdo = value->rValue;
            mod->BSIM3v1AcgdoGiven = TRUE;
            break;
        case  BSIM3v1A_MOD_CGBO :
            mod->BSIM3v1Acgbo = value->rValue;
            mod->BSIM3v1AcgboGiven = TRUE;
            break;
        case  BSIM3v1A_MOD_XPART :
            mod->BSIM3v1Axpart = value->rValue;
            mod->BSIM3v1AxpartGiven = TRUE;
            break;
        case  BSIM3v1A_MOD_RSH :
            mod->BSIM3v1AsheetResistance = value->rValue;
            mod->BSIM3v1AsheetResistanceGiven = TRUE;
            break;
        case  BSIM3v1A_MOD_JS :
            mod->BSIM3v1AjctSatCurDensity = value->rValue;
            mod->BSIM3v1AjctSatCurDensityGiven = TRUE;
            break;
        case  BSIM3v1A_MOD_PB :
            mod->BSIM3v1AbulkJctPotential = value->rValue;
            mod->BSIM3v1AbulkJctPotentialGiven = TRUE;
            break;
        case  BSIM3v1A_MOD_MJ :
            mod->BSIM3v1AbulkJctBotGradingCoeff = value->rValue;
            mod->BSIM3v1AbulkJctBotGradingCoeffGiven = TRUE;
            break;
        case  BSIM3v1A_MOD_PBSW :
            mod->BSIM3v1AsidewallJctPotential = value->rValue;
            mod->BSIM3v1AsidewallJctPotentialGiven = TRUE;
            break;
        case  BSIM3v1A_MOD_MJSW :
            mod->BSIM3v1AbulkJctSideGradingCoeff = value->rValue;
            mod->BSIM3v1AbulkJctSideGradingCoeffGiven = TRUE;
            break;
        case  BSIM3v1A_MOD_CJ :
            mod->BSIM3v1AunitAreaJctCap = value->rValue;
            mod->BSIM3v1AunitAreaJctCapGiven = TRUE;
            break;
        case  BSIM3v1A_MOD_CJSW :
            mod->BSIM3v1AunitLengthSidewallJctCap = value->rValue;
            mod->BSIM3v1AunitLengthSidewallJctCapGiven = TRUE;
            break;
        case  BSIM3v1A_MOD_LINT :
            mod->BSIM3v1ALint = value->rValue;
            mod->BSIM3v1ALintGiven = TRUE;
            break;
        case  BSIM3v1A_MOD_LL :
            mod->BSIM3v1ALl = value->rValue;
            mod->BSIM3v1ALlGiven = TRUE;
            break;
        case  BSIM3v1A_MOD_LLN :
            mod->BSIM3v1ALln = value->rValue;
            mod->BSIM3v1ALlnGiven = TRUE;
            break;
        case  BSIM3v1A_MOD_LW :
            mod->BSIM3v1ALw = value->rValue;
            mod->BSIM3v1ALwGiven = TRUE;
            break;
        case  BSIM3v1A_MOD_LWN :
            mod->BSIM3v1ALwn = value->rValue;
            mod->BSIM3v1ALwnGiven = TRUE;
            break;
        case  BSIM3v1A_MOD_LWL :
            mod->BSIM3v1ALwl = value->rValue;
            mod->BSIM3v1ALwlGiven = TRUE;
            break;
        case  BSIM3v1A_MOD_LMIN :
            mod->BSIM3v1ALmin = value->rValue;
            mod->BSIM3v1ALminGiven = TRUE;
            break;
        case  BSIM3v1A_MOD_LMAX :
            mod->BSIM3v1ALmax = value->rValue;
            mod->BSIM3v1ALmaxGiven = TRUE;
            break;
        case  BSIM3v1A_MOD_WINT :
            mod->BSIM3v1AWint = value->rValue;
            mod->BSIM3v1AWintGiven = TRUE;
            break;
        case  BSIM3v1A_MOD_WL :
            mod->BSIM3v1AWl = value->rValue;
            mod->BSIM3v1AWlGiven = TRUE;
            break;
        case  BSIM3v1A_MOD_WLN :
            mod->BSIM3v1AWln = value->rValue;
            mod->BSIM3v1AWlnGiven = TRUE;
            break;
        case  BSIM3v1A_MOD_WW :
            mod->BSIM3v1AWw = value->rValue;
            mod->BSIM3v1AWwGiven = TRUE;
            break;
        case  BSIM3v1A_MOD_WWN :
            mod->BSIM3v1AWwn = value->rValue;
            mod->BSIM3v1AWwnGiven = TRUE;
            break;
        case  BSIM3v1A_MOD_WWL :
            mod->BSIM3v1AWwl = value->rValue;
            mod->BSIM3v1AWwlGiven = TRUE;
            break;
        case  BSIM3v1A_MOD_WMIN :
            mod->BSIM3v1AWmin = value->rValue;
            mod->BSIM3v1AWminGiven = TRUE;
            break;
        case  BSIM3v1A_MOD_WMAX :
            mod->BSIM3v1AWmax = value->rValue;
            mod->BSIM3v1AWmaxGiven = TRUE;
            break;

        case  BSIM3v1A_MOD_NOIA :
            mod->BSIM3v1AoxideTrapDensityA = value->rValue;
            mod->BSIM3v1AoxideTrapDensityAGiven = TRUE;
            break;
        case  BSIM3v1A_MOD_NOIB :
            mod->BSIM3v1AoxideTrapDensityB = value->rValue;
            mod->BSIM3v1AoxideTrapDensityBGiven = TRUE;
            break;
        case  BSIM3v1A_MOD_NOIC :
            mod->BSIM3v1AoxideTrapDensityC = value->rValue;
            mod->BSIM3v1AoxideTrapDensityCGiven = TRUE;
            break;
        case  BSIM3v1A_MOD_EM :
            mod->BSIM3v1Aem = value->rValue;
            mod->BSIM3v1AemGiven = TRUE;
            break;
        case  BSIM3v1A_MOD_EF :
            mod->BSIM3v1Aef = value->rValue;
            mod->BSIM3v1AefGiven = TRUE;
            break;
        case  BSIM3v1A_MOD_AF :
            mod->BSIM3v1Aaf = value->rValue;
            mod->BSIM3v1AafGiven = TRUE;
            break;
        case  BSIM3v1A_MOD_KF :
            mod->BSIM3v1Akf = value->rValue;
            mod->BSIM3v1AkfGiven = TRUE;
            break;
        case  BSIM3v1A_MOD_NMOS  :
            if(value->iValue) {
                mod->BSIM3v1Atype = 1;
                mod->BSIM3v1AtypeGiven = TRUE;
            }
            break;
        case  BSIM3v1A_MOD_PMOS  :
            if(value->iValue) {
                mod->BSIM3v1Atype = - 1;
                mod->BSIM3v1AtypeGiven = TRUE;
            }
            break;
        default:
            return(E_BADPARM);
    }
    return(OK);
}


