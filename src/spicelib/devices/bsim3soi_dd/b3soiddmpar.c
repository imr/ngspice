/**********
Copyright 1999 Regents of the University of California.  All rights reserved.
Author: Weidong Liu and Pin Su         Feb 1999
Author: 1998 Samuel Fung, Dennis Sinitsky and Stephen Tang
Modified by Wei Jin 99/9/27
File: b3soiddmpar.c          98/5/01
Modified by Paolo Nenzi 2002
**********/

/*
 * Revision 2.1  99/9/27 Pin Su 
 * BSIMDD2.1 release
 */

#include "ngspice/ngspice.h"
#include "b3soidddef.h"
#include "ngspice/ifsim.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
B3SOIDDmParam(int param, IFvalue *value, GENmodel *inMod)
{
    B3SOIDDmodel *mod = (B3SOIDDmodel*)inMod;
    switch(param)
    {  
 
	case  B3SOIDD_MOD_MOBMOD :
            mod->B3SOIDDmobMod = value->iValue;
            mod->B3SOIDDmobModGiven = TRUE;
            break;
        case  B3SOIDD_MOD_BINUNIT :
            mod->B3SOIDDbinUnit = value->iValue;
            mod->B3SOIDDbinUnitGiven = TRUE;
            break;
        case  B3SOIDD_MOD_PARAMCHK :
            mod->B3SOIDDparamChk = value->iValue;
            mod->B3SOIDDparamChkGiven = TRUE;
            break;
        case  B3SOIDD_MOD_CAPMOD :
            mod->B3SOIDDcapMod = value->iValue;
            mod->B3SOIDDcapModGiven = TRUE;
            break;
        case  B3SOIDD_MOD_SHMOD :
            mod->B3SOIDDshMod = value->iValue;
            mod->B3SOIDDshModGiven = TRUE;
            break;
        case  B3SOIDD_MOD_NOIMOD :
            mod->B3SOIDDnoiMod = value->iValue;
            mod->B3SOIDDnoiModGiven = TRUE;
            break;
        case  B3SOIDD_MOD_VERSION :
            mod->B3SOIDDversion = value->rValue;
            mod->B3SOIDDversionGiven = TRUE;
            break;
        case  B3SOIDD_MOD_TOX :
            mod->B3SOIDDtox = value->rValue;
            mod->B3SOIDDtoxGiven = TRUE;
            break;

        case  B3SOIDD_MOD_CDSC :
            mod->B3SOIDDcdsc = value->rValue;
            mod->B3SOIDDcdscGiven = TRUE;
            break;
        case  B3SOIDD_MOD_CDSCB :
            mod->B3SOIDDcdscb = value->rValue;
            mod->B3SOIDDcdscbGiven = TRUE;
            break;

        case  B3SOIDD_MOD_CDSCD :
            mod->B3SOIDDcdscd = value->rValue;
            mod->B3SOIDDcdscdGiven = TRUE;
            break;

        case  B3SOIDD_MOD_CIT :
            mod->B3SOIDDcit = value->rValue;
            mod->B3SOIDDcitGiven = TRUE;
            break;
        case  B3SOIDD_MOD_NFACTOR :
            mod->B3SOIDDnfactor = value->rValue;
            mod->B3SOIDDnfactorGiven = TRUE;
            break;
        case B3SOIDD_MOD_VSAT:
            mod->B3SOIDDvsat = value->rValue;
            mod->B3SOIDDvsatGiven = TRUE;
            break;
        case B3SOIDD_MOD_A0:
            mod->B3SOIDDa0 = value->rValue;
            mod->B3SOIDDa0Given = TRUE;
            break;
        
        case B3SOIDD_MOD_AGS:
            mod->B3SOIDDags= value->rValue;
            mod->B3SOIDDagsGiven = TRUE;
            break;
        
        case B3SOIDD_MOD_A1:
            mod->B3SOIDDa1 = value->rValue;
            mod->B3SOIDDa1Given = TRUE;
            break;
        case B3SOIDD_MOD_A2:
            mod->B3SOIDDa2 = value->rValue;
            mod->B3SOIDDa2Given = TRUE;
            break;
        case B3SOIDD_MOD_AT:
            mod->B3SOIDDat = value->rValue;
            mod->B3SOIDDatGiven = TRUE;
            break;
        case B3SOIDD_MOD_KETA:
            mod->B3SOIDDketa = value->rValue;
            mod->B3SOIDDketaGiven = TRUE;
            break;    
        case B3SOIDD_MOD_NSUB:
            mod->B3SOIDDnsub = value->rValue;
            mod->B3SOIDDnsubGiven = TRUE;
            break;
        case B3SOIDD_MOD_NPEAK:
            mod->B3SOIDDnpeak = value->rValue;
            mod->B3SOIDDnpeakGiven = TRUE;
	    if (mod->B3SOIDDnpeak > 1.0e20)
		mod->B3SOIDDnpeak *= 1.0e-6;
            break;
        case B3SOIDD_MOD_NGATE:
            mod->B3SOIDDngate = value->rValue;
            mod->B3SOIDDngateGiven = TRUE;
	    if (mod->B3SOIDDngate > 1.0e23)
		mod->B3SOIDDngate *= 1.0e-6;
            break;
        case B3SOIDD_MOD_GAMMA1:
            mod->B3SOIDDgamma1 = value->rValue;
            mod->B3SOIDDgamma1Given = TRUE;
            break;
        case B3SOIDD_MOD_GAMMA2:
            mod->B3SOIDDgamma2 = value->rValue;
            mod->B3SOIDDgamma2Given = TRUE;
            break;
        case B3SOIDD_MOD_VBX:
            mod->B3SOIDDvbx = value->rValue;
            mod->B3SOIDDvbxGiven = TRUE;
            break;
        case B3SOIDD_MOD_VBM:
            mod->B3SOIDDvbm = value->rValue;
            mod->B3SOIDDvbmGiven = TRUE;
            break;
        case B3SOIDD_MOD_XT:
            mod->B3SOIDDxt = value->rValue;
            mod->B3SOIDDxtGiven = TRUE;
            break;
        case  B3SOIDD_MOD_K1:
            mod->B3SOIDDk1 = value->rValue;
            mod->B3SOIDDk1Given = TRUE;
            break;
        case  B3SOIDD_MOD_KT1:
            mod->B3SOIDDkt1 = value->rValue;
            mod->B3SOIDDkt1Given = TRUE;
            break;
        case  B3SOIDD_MOD_KT1L:
            mod->B3SOIDDkt1l = value->rValue;
            mod->B3SOIDDkt1lGiven = TRUE;
            break;
        case  B3SOIDD_MOD_KT2:
            mod->B3SOIDDkt2 = value->rValue;
            mod->B3SOIDDkt2Given = TRUE;
            break;
        case  B3SOIDD_MOD_K2:
            mod->B3SOIDDk2 = value->rValue;
            mod->B3SOIDDk2Given = TRUE;
            break;
        case  B3SOIDD_MOD_K3:
            mod->B3SOIDDk3 = value->rValue;
            mod->B3SOIDDk3Given = TRUE;
            break;
        case  B3SOIDD_MOD_K3B:
            mod->B3SOIDDk3b = value->rValue;
            mod->B3SOIDDk3bGiven = TRUE;
            break;
        case  B3SOIDD_MOD_NLX:
            mod->B3SOIDDnlx = value->rValue;
            mod->B3SOIDDnlxGiven = TRUE;
            break;
        case  B3SOIDD_MOD_W0:
            mod->B3SOIDDw0 = value->rValue;
            mod->B3SOIDDw0Given = TRUE;
            break;
        case  B3SOIDD_MOD_DVT0:               
            mod->B3SOIDDdvt0 = value->rValue;
            mod->B3SOIDDdvt0Given = TRUE;
            break;
        case  B3SOIDD_MOD_DVT1:             
            mod->B3SOIDDdvt1 = value->rValue;
            mod->B3SOIDDdvt1Given = TRUE;
            break;
        case  B3SOIDD_MOD_DVT2:             
            mod->B3SOIDDdvt2 = value->rValue;
            mod->B3SOIDDdvt2Given = TRUE;
            break;
        case  B3SOIDD_MOD_DVT0W:               
            mod->B3SOIDDdvt0w = value->rValue;
            mod->B3SOIDDdvt0wGiven = TRUE;
            break;
        case  B3SOIDD_MOD_DVT1W:             
            mod->B3SOIDDdvt1w = value->rValue;
            mod->B3SOIDDdvt1wGiven = TRUE;
            break;
        case  B3SOIDD_MOD_DVT2W:             
            mod->B3SOIDDdvt2w = value->rValue;
            mod->B3SOIDDdvt2wGiven = TRUE;
            break;
        case  B3SOIDD_MOD_DROUT:             
            mod->B3SOIDDdrout = value->rValue;
            mod->B3SOIDDdroutGiven = TRUE;
            break;
        case  B3SOIDD_MOD_DSUB:             
            mod->B3SOIDDdsub = value->rValue;
            mod->B3SOIDDdsubGiven = TRUE;
            break;
        case B3SOIDD_MOD_VTH0:
            mod->B3SOIDDvth0 = value->rValue;
            mod->B3SOIDDvth0Given = TRUE;
            break;
        case B3SOIDD_MOD_UA:
            mod->B3SOIDDua = value->rValue;
            mod->B3SOIDDuaGiven = TRUE;
            break;
        case B3SOIDD_MOD_UA1:
            mod->B3SOIDDua1 = value->rValue;
            mod->B3SOIDDua1Given = TRUE;
            break;
        case B3SOIDD_MOD_UB:
            mod->B3SOIDDub = value->rValue;
            mod->B3SOIDDubGiven = TRUE;
            break;
        case B3SOIDD_MOD_UB1:
            mod->B3SOIDDub1 = value->rValue;
            mod->B3SOIDDub1Given = TRUE;
            break;
        case B3SOIDD_MOD_UC:
            mod->B3SOIDDuc = value->rValue;
            mod->B3SOIDDucGiven = TRUE;
            break;
        case B3SOIDD_MOD_UC1:
            mod->B3SOIDDuc1 = value->rValue;
            mod->B3SOIDDuc1Given = TRUE;
            break;
        case  B3SOIDD_MOD_U0 :
            mod->B3SOIDDu0 = value->rValue;
            mod->B3SOIDDu0Given = TRUE;
            break;
        case  B3SOIDD_MOD_UTE :
            mod->B3SOIDDute = value->rValue;
            mod->B3SOIDDuteGiven = TRUE;
            break;
        case B3SOIDD_MOD_VOFF:
            mod->B3SOIDDvoff = value->rValue;
            mod->B3SOIDDvoffGiven = TRUE;
            break;
        case  B3SOIDD_MOD_DELTA :
            mod->B3SOIDDdelta = value->rValue;
            mod->B3SOIDDdeltaGiven = TRUE;
            break;
        case B3SOIDD_MOD_RDSW:
            mod->B3SOIDDrdsw = value->rValue;
            mod->B3SOIDDrdswGiven = TRUE;
            break;                     
        case B3SOIDD_MOD_PRWG:
            mod->B3SOIDDprwg = value->rValue;
            mod->B3SOIDDprwgGiven = TRUE;
            break;                     
        case B3SOIDD_MOD_PRWB:
            mod->B3SOIDDprwb = value->rValue;
            mod->B3SOIDDprwbGiven = TRUE;
            break;                     
        case B3SOIDD_MOD_PRT:
            mod->B3SOIDDprt = value->rValue;
            mod->B3SOIDDprtGiven = TRUE;
            break;                     
        case B3SOIDD_MOD_ETA0:
            mod->B3SOIDDeta0 = value->rValue;
            mod->B3SOIDDeta0Given = TRUE;
            break;                 
        case B3SOIDD_MOD_ETAB:
            mod->B3SOIDDetab = value->rValue;
            mod->B3SOIDDetabGiven = TRUE;
            break;                 
        case B3SOIDD_MOD_PCLM:
            mod->B3SOIDDpclm = value->rValue;
            mod->B3SOIDDpclmGiven = TRUE;
            break;                 
        case B3SOIDD_MOD_PDIBL1:
            mod->B3SOIDDpdibl1 = value->rValue;
            mod->B3SOIDDpdibl1Given = TRUE;
            break;                 
        case B3SOIDD_MOD_PDIBL2:
            mod->B3SOIDDpdibl2 = value->rValue;
            mod->B3SOIDDpdibl2Given = TRUE;
            break;                 
        case B3SOIDD_MOD_PDIBLB:
            mod->B3SOIDDpdiblb = value->rValue;
            mod->B3SOIDDpdiblbGiven = TRUE;
            break;                 
        case B3SOIDD_MOD_PVAG:
            mod->B3SOIDDpvag = value->rValue;
            mod->B3SOIDDpvagGiven = TRUE;
            break;                 
        case  B3SOIDD_MOD_WR :
            mod->B3SOIDDwr = value->rValue;
            mod->B3SOIDDwrGiven = TRUE;
            break;
        case  B3SOIDD_MOD_DWG :
            mod->B3SOIDDdwg = value->rValue;
            mod->B3SOIDDdwgGiven = TRUE;
            break;
        case  B3SOIDD_MOD_DWB :
            mod->B3SOIDDdwb = value->rValue;
            mod->B3SOIDDdwbGiven = TRUE;
            break;
        case  B3SOIDD_MOD_B0 :
            mod->B3SOIDDb0 = value->rValue;
            mod->B3SOIDDb0Given = TRUE;
            break;
        case  B3SOIDD_MOD_B1 :
            mod->B3SOIDDb1 = value->rValue;
            mod->B3SOIDDb1Given = TRUE;
            break;
        case  B3SOIDD_MOD_ALPHA0 :
            mod->B3SOIDDalpha0 = value->rValue;
            mod->B3SOIDDalpha0Given = TRUE;
            break;
        case  B3SOIDD_MOD_ALPHA1 :
            mod->B3SOIDDalpha1 = value->rValue;
            mod->B3SOIDDalpha1Given = TRUE;
            break;
        case  B3SOIDD_MOD_BETA0 :
            mod->B3SOIDDbeta0 = value->rValue;
            mod->B3SOIDDbeta0Given = TRUE;
            break;

        case  B3SOIDD_MOD_CGSL :
            mod->B3SOIDDcgsl = value->rValue;
            mod->B3SOIDDcgslGiven = TRUE;
            break;
        case  B3SOIDD_MOD_CGDL :
            mod->B3SOIDDcgdl = value->rValue;
            mod->B3SOIDDcgdlGiven = TRUE;
            break;
        case  B3SOIDD_MOD_CKAPPA :
            mod->B3SOIDDckappa = value->rValue;
            mod->B3SOIDDckappaGiven = TRUE;
            break;
        case  B3SOIDD_MOD_CF :
            mod->B3SOIDDcf = value->rValue;
            mod->B3SOIDDcfGiven = TRUE;
            break;
        case  B3SOIDD_MOD_CLC :
            mod->B3SOIDDclc = value->rValue;
            mod->B3SOIDDclcGiven = TRUE;
            break;
        case  B3SOIDD_MOD_CLE :
            mod->B3SOIDDcle = value->rValue;
            mod->B3SOIDDcleGiven = TRUE;
            break;
        case  B3SOIDD_MOD_DWC :
            mod->B3SOIDDdwc = value->rValue;
            mod->B3SOIDDdwcGiven = TRUE;
            break;
        case  B3SOIDD_MOD_DLC :
            mod->B3SOIDDdlc = value->rValue;
            mod->B3SOIDDdlcGiven = TRUE;
            break;
        case  B3SOIDD_MOD_TBOX :
            mod->B3SOIDDtbox = value->rValue;
            mod->B3SOIDDtboxGiven = TRUE;
            break;
        case  B3SOIDD_MOD_TSI :
            mod->B3SOIDDtsi = value->rValue;
            mod->B3SOIDDtsiGiven = TRUE;
            break;
        case  B3SOIDD_MOD_XJ :
            mod->B3SOIDDxj = value->rValue;
            mod->B3SOIDDxjGiven = TRUE;
            break;
        case  B3SOIDD_MOD_KB1 :
            mod->B3SOIDDkb1 = value->rValue;
            mod->B3SOIDDkb1Given = TRUE;
            break;
        case  B3SOIDD_MOD_KB3 :
            mod->B3SOIDDkb3 = value->rValue;
            mod->B3SOIDDkb3Given = TRUE;
            break;
        case  B3SOIDD_MOD_DVBD0 :
            mod->B3SOIDDdvbd0 = value->rValue;
            mod->B3SOIDDdvbd0Given = TRUE;
            break;
        case  B3SOIDD_MOD_DVBD1 :
            mod->B3SOIDDdvbd1 = value->rValue;
            mod->B3SOIDDdvbd1Given = TRUE;
            break;
        case  B3SOIDD_MOD_DELP :
            mod->B3SOIDDdelp = value->rValue;
            mod->B3SOIDDdelpGiven = TRUE;
            break;
        case  B3SOIDD_MOD_VBSA :
            mod->B3SOIDDvbsa = value->rValue;
            mod->B3SOIDDvbsaGiven = TRUE;
            break;
        case  B3SOIDD_MOD_RBODY :
            mod->B3SOIDDrbody = value->rValue;
            mod->B3SOIDDrbodyGiven = TRUE;
            break;
        case  B3SOIDD_MOD_RBSH :
            mod->B3SOIDDrbsh = value->rValue;
            mod->B3SOIDDrbshGiven = TRUE;
            break;
        case  B3SOIDD_MOD_ADICE0 :
            mod->B3SOIDDadice0 = value->rValue;
            mod->B3SOIDDadice0Given = TRUE;
            break;
        case  B3SOIDD_MOD_ABP :
            mod->B3SOIDDabp = value->rValue;
            mod->B3SOIDDabpGiven = TRUE;
            break;
        case  B3SOIDD_MOD_MXC :
            mod->B3SOIDDmxc = value->rValue;
            mod->B3SOIDDmxcGiven = TRUE;
            break;
        case  B3SOIDD_MOD_RTH0 :
            mod->B3SOIDDrth0 = value->rValue;
            mod->B3SOIDDrth0Given = TRUE;
            break;
        case  B3SOIDD_MOD_CTH0 :
            mod->B3SOIDDcth0 = value->rValue;
            mod->B3SOIDDcth0Given = TRUE;
            break;
        case  B3SOIDD_MOD_AII :
            mod->B3SOIDDaii = value->rValue;
            mod->B3SOIDDaiiGiven = TRUE;
            break;
        case  B3SOIDD_MOD_BII :
            mod->B3SOIDDbii = value->rValue;
            mod->B3SOIDDbiiGiven = TRUE;
            break;
        case  B3SOIDD_MOD_CII :
            mod->B3SOIDDcii = value->rValue;
            mod->B3SOIDDciiGiven = TRUE;
            break;
        case  B3SOIDD_MOD_DII :
            mod->B3SOIDDdii = value->rValue;
            mod->B3SOIDDdiiGiven = TRUE;
            break;
        case  B3SOIDD_MOD_NGIDL :
            mod->B3SOIDDngidl = value->rValue;
            mod->B3SOIDDngidlGiven = TRUE;
            break;
        case  B3SOIDD_MOD_AGIDL :
            mod->B3SOIDDagidl = value->rValue;
            mod->B3SOIDDagidlGiven = TRUE;
            break;
        case  B3SOIDD_MOD_BGIDL :
            mod->B3SOIDDbgidl = value->rValue;
            mod->B3SOIDDbgidlGiven = TRUE;
            break;
        case  B3SOIDD_MOD_NDIODE :
            mod->B3SOIDDndiode = value->rValue;
            mod->B3SOIDDndiodeGiven = TRUE;
            break;
        case  B3SOIDD_MOD_NTUN :
            mod->B3SOIDDntun = value->rValue;
            mod->B3SOIDDntunGiven = TRUE;
            break;
        case  B3SOIDD_MOD_ISBJT :
            mod->B3SOIDDisbjt = value->rValue;
            mod->B3SOIDDisbjtGiven = TRUE;
            break;
        case  B3SOIDD_MOD_ISDIF :
            mod->B3SOIDDisdif = value->rValue;
            mod->B3SOIDDisdifGiven = TRUE;
            break;
        case  B3SOIDD_MOD_ISREC :
            mod->B3SOIDDisrec = value->rValue;
            mod->B3SOIDDisrecGiven = TRUE;
            break;
        case  B3SOIDD_MOD_ISTUN :
            mod->B3SOIDDistun = value->rValue;
            mod->B3SOIDDistunGiven = TRUE;
            break;
        case  B3SOIDD_MOD_XBJT :
            mod->B3SOIDDxbjt = value->rValue;
            mod->B3SOIDDxbjtGiven = TRUE;
            break;
        case  B3SOIDD_MOD_XREC :
            mod->B3SOIDDxrec = value->rValue;
            mod->B3SOIDDxrecGiven = TRUE;
            break;
        case  B3SOIDD_MOD_XTUN :
            mod->B3SOIDDxtun = value->rValue;
            mod->B3SOIDDxtunGiven = TRUE;
            break;
        case  B3SOIDD_MOD_EDL :
            mod->B3SOIDDedl = value->rValue;
            mod->B3SOIDDedlGiven = TRUE;
            break;
        case  B3SOIDD_MOD_KBJT1 :
            mod->B3SOIDDkbjt1 = value->rValue;
            mod->B3SOIDDkbjt1Given = TRUE;
            break;
        case  B3SOIDD_MOD_TT :
            mod->B3SOIDDtt = value->rValue;
            mod->B3SOIDDttGiven = TRUE;
            break;
        case  B3SOIDD_MOD_VSDTH :
            mod->B3SOIDDvsdth = value->rValue;
            mod->B3SOIDDvsdthGiven = TRUE;
            break;
        case  B3SOIDD_MOD_VSDFB :
            mod->B3SOIDDvsdfb = value->rValue;
            mod->B3SOIDDvsdfbGiven = TRUE;
            break;
        case  B3SOIDD_MOD_CSDMIN :
            mod->B3SOIDDcsdmin = value->rValue;
            mod->B3SOIDDcsdminGiven = TRUE;
            break;
        case  B3SOIDD_MOD_ASD :
            mod->B3SOIDDasd = value->rValue;
            mod->B3SOIDDasdGiven = TRUE;
            break;


        case  B3SOIDD_MOD_TNOM :
            mod->B3SOIDDtnom = value->rValue + 273.15;
            mod->B3SOIDDtnomGiven = TRUE;
            break;
        case  B3SOIDD_MOD_CGSO :
            mod->B3SOIDDcgso = value->rValue;
            mod->B3SOIDDcgsoGiven = TRUE;
            break;
        case  B3SOIDD_MOD_CGDO :
            mod->B3SOIDDcgdo = value->rValue;
            mod->B3SOIDDcgdoGiven = TRUE;
            break;
        case  B3SOIDD_MOD_CGEO :
            mod->B3SOIDDcgeo = value->rValue;
            mod->B3SOIDDcgeoGiven = TRUE;
            break;
        case  B3SOIDD_MOD_XPART :
            mod->B3SOIDDxpart = value->rValue;
            mod->B3SOIDDxpartGiven = TRUE;
            break;
        case  B3SOIDD_MOD_RSH :
            mod->B3SOIDDsheetResistance = value->rValue;
            mod->B3SOIDDsheetResistanceGiven = TRUE;
            break;
        case  B3SOIDD_MOD_PBSWG :
            mod->B3SOIDDGatesidewallJctPotential = value->rValue;
            mod->B3SOIDDGatesidewallJctPotentialGiven = TRUE;
            break;
        case  B3SOIDD_MOD_MJSWG :
            mod->B3SOIDDbodyJctGateSideGradingCoeff = value->rValue;
            mod->B3SOIDDbodyJctGateSideGradingCoeffGiven = TRUE;
            break;
        case  B3SOIDD_MOD_CJSWG :
            mod->B3SOIDDunitLengthGateSidewallJctCap = value->rValue;
            mod->B3SOIDDunitLengthGateSidewallJctCapGiven = TRUE;
            break;
        case  B3SOIDD_MOD_CSDESW :
            mod->B3SOIDDcsdesw = value->rValue;
            mod->B3SOIDDcsdeswGiven = TRUE;
            break;
        case  B3SOIDD_MOD_LINT :
            mod->B3SOIDDLint = value->rValue;
            mod->B3SOIDDLintGiven = TRUE;
            break;
        case  B3SOIDD_MOD_LL :
            mod->B3SOIDDLl = value->rValue;
            mod->B3SOIDDLlGiven = TRUE;
            break;
        case  B3SOIDD_MOD_LLN :
            mod->B3SOIDDLln = value->rValue;
            mod->B3SOIDDLlnGiven = TRUE;
            break;
        case  B3SOIDD_MOD_LW :
            mod->B3SOIDDLw = value->rValue;
            mod->B3SOIDDLwGiven = TRUE;
            break;
        case  B3SOIDD_MOD_LWN :
            mod->B3SOIDDLwn = value->rValue;
            mod->B3SOIDDLwnGiven = TRUE;
            break;
        case  B3SOIDD_MOD_LWL :
            mod->B3SOIDDLwl = value->rValue;
            mod->B3SOIDDLwlGiven = TRUE;
            break;
        case  B3SOIDD_MOD_WINT :
            mod->B3SOIDDWint = value->rValue;
            mod->B3SOIDDWintGiven = TRUE;
            break;
        case  B3SOIDD_MOD_WL :
            mod->B3SOIDDWl = value->rValue;
            mod->B3SOIDDWlGiven = TRUE;
            break;
        case  B3SOIDD_MOD_WLN :
            mod->B3SOIDDWln = value->rValue;
            mod->B3SOIDDWlnGiven = TRUE;
            break;
        case  B3SOIDD_MOD_WW :
            mod->B3SOIDDWw = value->rValue;
            mod->B3SOIDDWwGiven = TRUE;
            break;
        case  B3SOIDD_MOD_WWN :
            mod->B3SOIDDWwn = value->rValue;
            mod->B3SOIDDWwnGiven = TRUE;
            break;
        case  B3SOIDD_MOD_WWL :
            mod->B3SOIDDWwl = value->rValue;
            mod->B3SOIDDWwlGiven = TRUE;
            break;

        case  B3SOIDD_MOD_NOIA :
            mod->B3SOIDDoxideTrapDensityA = value->rValue;
            mod->B3SOIDDoxideTrapDensityAGiven = TRUE;
            break;
        case  B3SOIDD_MOD_NOIB :
            mod->B3SOIDDoxideTrapDensityB = value->rValue;
            mod->B3SOIDDoxideTrapDensityBGiven = TRUE;
            break;
        case  B3SOIDD_MOD_NOIC :
            mod->B3SOIDDoxideTrapDensityC = value->rValue;
            mod->B3SOIDDoxideTrapDensityCGiven = TRUE;
            break;
        case  B3SOIDD_MOD_NOIF :
            mod->B3SOIDDnoif = value->rValue;
            mod->B3SOIDDnoifGiven = TRUE;
            break;
        case  B3SOIDD_MOD_EM :
            mod->B3SOIDDem = value->rValue;
            mod->B3SOIDDemGiven = TRUE;
            break;
        case  B3SOIDD_MOD_EF :
            mod->B3SOIDDef = value->rValue;
            mod->B3SOIDDefGiven = TRUE;
            break;
        case  B3SOIDD_MOD_AF :
            mod->B3SOIDDaf = value->rValue;
            mod->B3SOIDDafGiven = TRUE;
            break;
        case  B3SOIDD_MOD_KF :
            mod->B3SOIDDkf = value->rValue;
            mod->B3SOIDDkfGiven = TRUE;
            break;

/* Added for binning - START */
        /* Length Dependence */
        case B3SOIDD_MOD_LNPEAK:
            mod->B3SOIDDlnpeak = value->rValue;
            mod->B3SOIDDlnpeakGiven = TRUE;
            break;
        case B3SOIDD_MOD_LNSUB:
            mod->B3SOIDDlnsub = value->rValue;
            mod->B3SOIDDlnsubGiven = TRUE;
            break;
        case B3SOIDD_MOD_LNGATE:
            mod->B3SOIDDlngate = value->rValue;
            mod->B3SOIDDlngateGiven = TRUE;
            break;
        case B3SOIDD_MOD_LVTH0:
            mod->B3SOIDDlvth0 = value->rValue;
            mod->B3SOIDDlvth0Given = TRUE;
            break;
        case  B3SOIDD_MOD_LK1:
            mod->B3SOIDDlk1 = value->rValue;
            mod->B3SOIDDlk1Given = TRUE;
            break;
        case  B3SOIDD_MOD_LK2:
            mod->B3SOIDDlk2 = value->rValue;
            mod->B3SOIDDlk2Given = TRUE;
            break;
        case  B3SOIDD_MOD_LK3:
            mod->B3SOIDDlk3 = value->rValue;
            mod->B3SOIDDlk3Given = TRUE;
            break;
        case  B3SOIDD_MOD_LK3B:
            mod->B3SOIDDlk3b = value->rValue;
            mod->B3SOIDDlk3bGiven = TRUE;
            break;
        case  B3SOIDD_MOD_LVBSA:
            mod->B3SOIDDlvbsa = value->rValue;
            mod->B3SOIDDlvbsaGiven = TRUE;
            break;
        case  B3SOIDD_MOD_LDELP:
            mod->B3SOIDDldelp = value->rValue;
            mod->B3SOIDDldelpGiven = TRUE;
            break;
        case  B3SOIDD_MOD_LKB1 :
            mod->B3SOIDDlkb1 = value->rValue;
            mod->B3SOIDDlkb1Given = TRUE;
            break;
        case  B3SOIDD_MOD_LKB3 :
            mod->B3SOIDDlkb3 = value->rValue;
            mod->B3SOIDDlkb3Given = TRUE;
            break;
        case  B3SOIDD_MOD_LDVBD0 :
            mod->B3SOIDDldvbd0 = value->rValue;
            mod->B3SOIDDldvbd0Given = TRUE;
            break;
        case  B3SOIDD_MOD_LDVBD1 :
            mod->B3SOIDDldvbd1 = value->rValue;
            mod->B3SOIDDldvbd1Given = TRUE;
            break;
        case  B3SOIDD_MOD_LW0:
            mod->B3SOIDDlw0 = value->rValue;
            mod->B3SOIDDlw0Given = TRUE;
            break;
        case  B3SOIDD_MOD_LNLX:
            mod->B3SOIDDlnlx = value->rValue;
            mod->B3SOIDDlnlxGiven = TRUE;
            break;
        case  B3SOIDD_MOD_LDVT0:               
            mod->B3SOIDDldvt0 = value->rValue;
            mod->B3SOIDDldvt0Given = TRUE;
            break;
        case  B3SOIDD_MOD_LDVT1:             
            mod->B3SOIDDldvt1 = value->rValue;
            mod->B3SOIDDldvt1Given = TRUE;
            break;
        case  B3SOIDD_MOD_LDVT2:             
            mod->B3SOIDDldvt2 = value->rValue;
            mod->B3SOIDDldvt2Given = TRUE;
            break;
        case  B3SOIDD_MOD_LDVT0W:               
            mod->B3SOIDDldvt0w = value->rValue;
            mod->B3SOIDDldvt0wGiven = TRUE;
            break;
        case  B3SOIDD_MOD_LDVT1W:             
            mod->B3SOIDDldvt1w = value->rValue;
            mod->B3SOIDDldvt1wGiven = TRUE;
            break;
        case  B3SOIDD_MOD_LDVT2W:             
            mod->B3SOIDDldvt2w = value->rValue;
            mod->B3SOIDDldvt2wGiven = TRUE;
            break;
        case  B3SOIDD_MOD_LU0 :
            mod->B3SOIDDlu0 = value->rValue;
            mod->B3SOIDDlu0Given = TRUE;
            break;
        case B3SOIDD_MOD_LUA:
            mod->B3SOIDDlua = value->rValue;
            mod->B3SOIDDluaGiven = TRUE;
            break;
        case B3SOIDD_MOD_LUB:
            mod->B3SOIDDlub = value->rValue;
            mod->B3SOIDDlubGiven = TRUE;
            break;
        case B3SOIDD_MOD_LUC:
            mod->B3SOIDDluc = value->rValue;
            mod->B3SOIDDlucGiven = TRUE;
            break;
        case B3SOIDD_MOD_LVSAT:
            mod->B3SOIDDlvsat = value->rValue;
            mod->B3SOIDDlvsatGiven = TRUE;
            break;
        case B3SOIDD_MOD_LA0:
            mod->B3SOIDDla0 = value->rValue;
            mod->B3SOIDDla0Given = TRUE;
            break;
        case B3SOIDD_MOD_LAGS:
            mod->B3SOIDDlags= value->rValue;
            mod->B3SOIDDlagsGiven = TRUE;
            break;
        case  B3SOIDD_MOD_LB0 :
            mod->B3SOIDDlb0 = value->rValue;
            mod->B3SOIDDlb0Given = TRUE;
            break;
        case  B3SOIDD_MOD_LB1 :
            mod->B3SOIDDlb1 = value->rValue;
            mod->B3SOIDDlb1Given = TRUE;
            break;
        case B3SOIDD_MOD_LKETA:
            mod->B3SOIDDlketa = value->rValue;
            mod->B3SOIDDlketaGiven = TRUE;
            break;    
        case B3SOIDD_MOD_LABP:
            mod->B3SOIDDlabp = value->rValue;
            mod->B3SOIDDlabpGiven = TRUE;
            break;    
        case B3SOIDD_MOD_LMXC:
            mod->B3SOIDDlmxc = value->rValue;
            mod->B3SOIDDlmxcGiven = TRUE;
            break;    
        case B3SOIDD_MOD_LADICE0:
            mod->B3SOIDDladice0 = value->rValue;
            mod->B3SOIDDladice0Given = TRUE;
            break;    
        case B3SOIDD_MOD_LA1:
            mod->B3SOIDDla1 = value->rValue;
            mod->B3SOIDDla1Given = TRUE;
            break;
        case B3SOIDD_MOD_LA2:
            mod->B3SOIDDla2 = value->rValue;
            mod->B3SOIDDla2Given = TRUE;
            break;
        case B3SOIDD_MOD_LRDSW:
            mod->B3SOIDDlrdsw = value->rValue;
            mod->B3SOIDDlrdswGiven = TRUE;
            break;                     
        case B3SOIDD_MOD_LPRWB:
            mod->B3SOIDDlprwb = value->rValue;
            mod->B3SOIDDlprwbGiven = TRUE;
            break;                     
        case B3SOIDD_MOD_LPRWG:
            mod->B3SOIDDlprwg = value->rValue;
            mod->B3SOIDDlprwgGiven = TRUE;
            break;                     
        case  B3SOIDD_MOD_LWR :
            mod->B3SOIDDlwr = value->rValue;
            mod->B3SOIDDlwrGiven = TRUE;
            break;
        case  B3SOIDD_MOD_LNFACTOR :
            mod->B3SOIDDlnfactor = value->rValue;
            mod->B3SOIDDlnfactorGiven = TRUE;
            break;
        case  B3SOIDD_MOD_LDWG :
            mod->B3SOIDDldwg = value->rValue;
            mod->B3SOIDDldwgGiven = TRUE;
            break;
        case  B3SOIDD_MOD_LDWB :
            mod->B3SOIDDldwb = value->rValue;
            mod->B3SOIDDldwbGiven = TRUE;
            break;
        case B3SOIDD_MOD_LVOFF:
            mod->B3SOIDDlvoff = value->rValue;
            mod->B3SOIDDlvoffGiven = TRUE;
            break;
        case B3SOIDD_MOD_LETA0:
            mod->B3SOIDDleta0 = value->rValue;
            mod->B3SOIDDleta0Given = TRUE;
            break;                 
        case B3SOIDD_MOD_LETAB:
            mod->B3SOIDDletab = value->rValue;
            mod->B3SOIDDletabGiven = TRUE;
            break;                 
        case  B3SOIDD_MOD_LDSUB:             
            mod->B3SOIDDldsub = value->rValue;
            mod->B3SOIDDldsubGiven = TRUE;
            break;
        case  B3SOIDD_MOD_LCIT :
            mod->B3SOIDDlcit = value->rValue;
            mod->B3SOIDDlcitGiven = TRUE;
            break;
        case  B3SOIDD_MOD_LCDSC :
            mod->B3SOIDDlcdsc = value->rValue;
            mod->B3SOIDDlcdscGiven = TRUE;
            break;
        case  B3SOIDD_MOD_LCDSCB :
            mod->B3SOIDDlcdscb = value->rValue;
            mod->B3SOIDDlcdscbGiven = TRUE;
            break;
        case  B3SOIDD_MOD_LCDSCD :
            mod->B3SOIDDlcdscd = value->rValue;
            mod->B3SOIDDlcdscdGiven = TRUE;
            break;
        case B3SOIDD_MOD_LPCLM:
            mod->B3SOIDDlpclm = value->rValue;
            mod->B3SOIDDlpclmGiven = TRUE;
            break;                 
        case B3SOIDD_MOD_LPDIBL1:
            mod->B3SOIDDlpdibl1 = value->rValue;
            mod->B3SOIDDlpdibl1Given = TRUE;
            break;                 
        case B3SOIDD_MOD_LPDIBL2:
            mod->B3SOIDDlpdibl2 = value->rValue;
            mod->B3SOIDDlpdibl2Given = TRUE;
            break;                 
        case B3SOIDD_MOD_LPDIBLB:
            mod->B3SOIDDlpdiblb = value->rValue;
            mod->B3SOIDDlpdiblbGiven = TRUE;
            break;                 
        case  B3SOIDD_MOD_LDROUT:             
            mod->B3SOIDDldrout = value->rValue;
            mod->B3SOIDDldroutGiven = TRUE;
            break;
        case B3SOIDD_MOD_LPVAG:
            mod->B3SOIDDlpvag = value->rValue;
            mod->B3SOIDDlpvagGiven = TRUE;
            break;                 
        case  B3SOIDD_MOD_LDELTA :
            mod->B3SOIDDldelta = value->rValue;
            mod->B3SOIDDldeltaGiven = TRUE;
            break;
        case  B3SOIDD_MOD_LAII :
            mod->B3SOIDDlaii = value->rValue;
            mod->B3SOIDDlaiiGiven = TRUE;
            break;
        case  B3SOIDD_MOD_LBII :
            mod->B3SOIDDlbii = value->rValue;
            mod->B3SOIDDlbiiGiven = TRUE;
            break;
        case  B3SOIDD_MOD_LCII :
            mod->B3SOIDDlcii = value->rValue;
            mod->B3SOIDDlciiGiven = TRUE;
            break;
        case  B3SOIDD_MOD_LDII :
            mod->B3SOIDDldii = value->rValue;
            mod->B3SOIDDldiiGiven = TRUE;
            break;
        case  B3SOIDD_MOD_LALPHA0 :
            mod->B3SOIDDlalpha0 = value->rValue;
            mod->B3SOIDDlalpha0Given = TRUE;
            break;
        case  B3SOIDD_MOD_LALPHA1 :
            mod->B3SOIDDlalpha1 = value->rValue;
            mod->B3SOIDDlalpha1Given = TRUE;
            break;
        case  B3SOIDD_MOD_LBETA0 :
            mod->B3SOIDDlbeta0 = value->rValue;
            mod->B3SOIDDlbeta0Given = TRUE;
            break;
        case  B3SOIDD_MOD_LAGIDL :
            mod->B3SOIDDlagidl = value->rValue;
            mod->B3SOIDDlagidlGiven = TRUE;
            break;
        case  B3SOIDD_MOD_LBGIDL :
            mod->B3SOIDDlbgidl = value->rValue;
            mod->B3SOIDDlbgidlGiven = TRUE;
            break;
        case  B3SOIDD_MOD_LNGIDL :
            mod->B3SOIDDlngidl = value->rValue;
            mod->B3SOIDDlngidlGiven = TRUE;
            break;
        case  B3SOIDD_MOD_LNTUN :
            mod->B3SOIDDlntun = value->rValue;
            mod->B3SOIDDlntunGiven = TRUE;
            break;
        case  B3SOIDD_MOD_LNDIODE :
            mod->B3SOIDDlndiode = value->rValue;
            mod->B3SOIDDlndiodeGiven = TRUE;
            break;
        case  B3SOIDD_MOD_LISBJT :
            mod->B3SOIDDlisbjt = value->rValue;
            mod->B3SOIDDlisbjtGiven = TRUE;
            break;
        case  B3SOIDD_MOD_LISDIF :
            mod->B3SOIDDlisdif = value->rValue;
            mod->B3SOIDDlisdifGiven = TRUE;
            break;
        case  B3SOIDD_MOD_LISREC :
            mod->B3SOIDDlisrec = value->rValue;
            mod->B3SOIDDlisrecGiven = TRUE;
            break;
        case  B3SOIDD_MOD_LISTUN :
            mod->B3SOIDDlistun = value->rValue;
            mod->B3SOIDDlistunGiven = TRUE;
            break;
        case  B3SOIDD_MOD_LEDL :
            mod->B3SOIDDledl = value->rValue;
            mod->B3SOIDDledlGiven = TRUE;
            break;
        case  B3SOIDD_MOD_LKBJT1 :
            mod->B3SOIDDlkbjt1 = value->rValue;
            mod->B3SOIDDlkbjt1Given = TRUE;
            break;
	/* CV Model */
        case  B3SOIDD_MOD_LVSDFB :
            mod->B3SOIDDlvsdfb = value->rValue;
            mod->B3SOIDDlvsdfbGiven = TRUE;
            break;
        case  B3SOIDD_MOD_LVSDTH :
            mod->B3SOIDDlvsdth = value->rValue;
            mod->B3SOIDDlvsdthGiven = TRUE;
            break;
        /* Width Dependence */
        case B3SOIDD_MOD_WNPEAK:
            mod->B3SOIDDwnpeak = value->rValue;
            mod->B3SOIDDwnpeakGiven = TRUE;
            break;
        case B3SOIDD_MOD_WNSUB:
            mod->B3SOIDDwnsub = value->rValue;
            mod->B3SOIDDwnsubGiven = TRUE;
            break;
        case B3SOIDD_MOD_WNGATE:
            mod->B3SOIDDwngate = value->rValue;
            mod->B3SOIDDwngateGiven = TRUE;
            break;
        case B3SOIDD_MOD_WVTH0:
            mod->B3SOIDDwvth0 = value->rValue;
            mod->B3SOIDDwvth0Given = TRUE;
            break;
        case  B3SOIDD_MOD_WK1:
            mod->B3SOIDDwk1 = value->rValue;
            mod->B3SOIDDwk1Given = TRUE;
            break;
        case  B3SOIDD_MOD_WK2:
            mod->B3SOIDDwk2 = value->rValue;
            mod->B3SOIDDwk2Given = TRUE;
            break;
        case  B3SOIDD_MOD_WK3:
            mod->B3SOIDDwk3 = value->rValue;
            mod->B3SOIDDwk3Given = TRUE;
            break;
        case  B3SOIDD_MOD_WK3B:
            mod->B3SOIDDwk3b = value->rValue;
            mod->B3SOIDDwk3bGiven = TRUE;
            break;
        case  B3SOIDD_MOD_WVBSA:
            mod->B3SOIDDwvbsa = value->rValue;
            mod->B3SOIDDwvbsaGiven = TRUE;
            break;
        case  B3SOIDD_MOD_WDELP:
            mod->B3SOIDDwdelp = value->rValue;
            mod->B3SOIDDwdelpGiven = TRUE;
            break;
        case  B3SOIDD_MOD_WKB1 :
            mod->B3SOIDDwkb1 = value->rValue;
            mod->B3SOIDDwkb1Given = TRUE;
            break;
        case  B3SOIDD_MOD_WKB3 :
            mod->B3SOIDDwkb3 = value->rValue;
            mod->B3SOIDDwkb3Given = TRUE;
            break;
        case  B3SOIDD_MOD_WDVBD0 :
            mod->B3SOIDDwdvbd0 = value->rValue;
            mod->B3SOIDDwdvbd0Given = TRUE;
            break;
        case  B3SOIDD_MOD_WDVBD1 :
            mod->B3SOIDDwdvbd1 = value->rValue;
            mod->B3SOIDDwdvbd1Given = TRUE;
            break;
        case  B3SOIDD_MOD_WW0:
            mod->B3SOIDDww0 = value->rValue;
            mod->B3SOIDDww0Given = TRUE;
            break;
        case  B3SOIDD_MOD_WNLX:
            mod->B3SOIDDwnlx = value->rValue;
            mod->B3SOIDDwnlxGiven = TRUE;
            break;
        case  B3SOIDD_MOD_WDVT0:               
            mod->B3SOIDDwdvt0 = value->rValue;
            mod->B3SOIDDwdvt0Given = TRUE;
            break;
        case  B3SOIDD_MOD_WDVT1:             
            mod->B3SOIDDwdvt1 = value->rValue;
            mod->B3SOIDDwdvt1Given = TRUE;
            break;
        case  B3SOIDD_MOD_WDVT2:             
            mod->B3SOIDDwdvt2 = value->rValue;
            mod->B3SOIDDwdvt2Given = TRUE;
            break;
        case  B3SOIDD_MOD_WDVT0W:               
            mod->B3SOIDDwdvt0w = value->rValue;
            mod->B3SOIDDwdvt0wGiven = TRUE;
            break;
        case  B3SOIDD_MOD_WDVT1W:             
            mod->B3SOIDDwdvt1w = value->rValue;
            mod->B3SOIDDwdvt1wGiven = TRUE;
            break;
        case  B3SOIDD_MOD_WDVT2W:             
            mod->B3SOIDDwdvt2w = value->rValue;
            mod->B3SOIDDwdvt2wGiven = TRUE;
            break;
        case  B3SOIDD_MOD_WU0 :
            mod->B3SOIDDwu0 = value->rValue;
            mod->B3SOIDDwu0Given = TRUE;
            break;
        case B3SOIDD_MOD_WUA:
            mod->B3SOIDDwua = value->rValue;
            mod->B3SOIDDwuaGiven = TRUE;
            break;
        case B3SOIDD_MOD_WUB:
            mod->B3SOIDDwub = value->rValue;
            mod->B3SOIDDwubGiven = TRUE;
            break;
        case B3SOIDD_MOD_WUC:
            mod->B3SOIDDwuc = value->rValue;
            mod->B3SOIDDwucGiven = TRUE;
            break;
        case B3SOIDD_MOD_WVSAT:
            mod->B3SOIDDwvsat = value->rValue;
            mod->B3SOIDDwvsatGiven = TRUE;
            break;
        case B3SOIDD_MOD_WA0:
            mod->B3SOIDDwa0 = value->rValue;
            mod->B3SOIDDwa0Given = TRUE;
            break;
        case B3SOIDD_MOD_WAGS:
            mod->B3SOIDDwags= value->rValue;
            mod->B3SOIDDwagsGiven = TRUE;
            break;
        case  B3SOIDD_MOD_WB0 :
            mod->B3SOIDDwb0 = value->rValue;
            mod->B3SOIDDwb0Given = TRUE;
            break;
        case  B3SOIDD_MOD_WB1 :
            mod->B3SOIDDwb1 = value->rValue;
            mod->B3SOIDDwb1Given = TRUE;
            break;
        case B3SOIDD_MOD_WKETA:
            mod->B3SOIDDwketa = value->rValue;
            mod->B3SOIDDwketaGiven = TRUE;
            break;    
        case B3SOIDD_MOD_WABP:
            mod->B3SOIDDwabp = value->rValue;
            mod->B3SOIDDwabpGiven = TRUE;
            break;    
        case B3SOIDD_MOD_WMXC:
            mod->B3SOIDDwmxc = value->rValue;
            mod->B3SOIDDwmxcGiven = TRUE;
            break;    
        case B3SOIDD_MOD_WADICE0:
            mod->B3SOIDDwadice0 = value->rValue;
            mod->B3SOIDDwadice0Given = TRUE;
            break;    
        case B3SOIDD_MOD_WA1:
            mod->B3SOIDDwa1 = value->rValue;
            mod->B3SOIDDwa1Given = TRUE;
            break;
        case B3SOIDD_MOD_WA2:
            mod->B3SOIDDwa2 = value->rValue;
            mod->B3SOIDDwa2Given = TRUE;
            break;
        case B3SOIDD_MOD_WRDSW:
            mod->B3SOIDDwrdsw = value->rValue;
            mod->B3SOIDDwrdswGiven = TRUE;
            break;                     
        case B3SOIDD_MOD_WPRWB:
            mod->B3SOIDDwprwb = value->rValue;
            mod->B3SOIDDwprwbGiven = TRUE;
            break;                     
        case B3SOIDD_MOD_WPRWG:
            mod->B3SOIDDwprwg = value->rValue;
            mod->B3SOIDDwprwgGiven = TRUE;
            break;                     
        case  B3SOIDD_MOD_WWR :
            mod->B3SOIDDwwr = value->rValue;
            mod->B3SOIDDwwrGiven = TRUE;
            break;
        case  B3SOIDD_MOD_WNFACTOR :
            mod->B3SOIDDwnfactor = value->rValue;
            mod->B3SOIDDwnfactorGiven = TRUE;
            break;
        case  B3SOIDD_MOD_WDWG :
            mod->B3SOIDDwdwg = value->rValue;
            mod->B3SOIDDwdwgGiven = TRUE;
            break;
        case  B3SOIDD_MOD_WDWB :
            mod->B3SOIDDwdwb = value->rValue;
            mod->B3SOIDDwdwbGiven = TRUE;
            break;
        case B3SOIDD_MOD_WVOFF:
            mod->B3SOIDDwvoff = value->rValue;
            mod->B3SOIDDwvoffGiven = TRUE;
            break;
        case B3SOIDD_MOD_WETA0:
            mod->B3SOIDDweta0 = value->rValue;
            mod->B3SOIDDweta0Given = TRUE;
            break;                 
        case B3SOIDD_MOD_WETAB:
            mod->B3SOIDDwetab = value->rValue;
            mod->B3SOIDDwetabGiven = TRUE;
            break;                 
        case  B3SOIDD_MOD_WDSUB:             
            mod->B3SOIDDwdsub = value->rValue;
            mod->B3SOIDDwdsubGiven = TRUE;
            break;
        case  B3SOIDD_MOD_WCIT :
            mod->B3SOIDDwcit = value->rValue;
            mod->B3SOIDDwcitGiven = TRUE;
            break;
        case  B3SOIDD_MOD_WCDSC :
            mod->B3SOIDDwcdsc = value->rValue;
            mod->B3SOIDDwcdscGiven = TRUE;
            break;
        case  B3SOIDD_MOD_WCDSCB :
            mod->B3SOIDDwcdscb = value->rValue;
            mod->B3SOIDDwcdscbGiven = TRUE;
            break;
        case  B3SOIDD_MOD_WCDSCD :
            mod->B3SOIDDwcdscd = value->rValue;
            mod->B3SOIDDwcdscdGiven = TRUE;
            break;
        case B3SOIDD_MOD_WPCLM:
            mod->B3SOIDDwpclm = value->rValue;
            mod->B3SOIDDwpclmGiven = TRUE;
            break;                 
        case B3SOIDD_MOD_WPDIBL1:
            mod->B3SOIDDwpdibl1 = value->rValue;
            mod->B3SOIDDwpdibl1Given = TRUE;
            break;                 
        case B3SOIDD_MOD_WPDIBL2:
            mod->B3SOIDDwpdibl2 = value->rValue;
            mod->B3SOIDDwpdibl2Given = TRUE;
            break;                 
        case B3SOIDD_MOD_WPDIBLB:
            mod->B3SOIDDwpdiblb = value->rValue;
            mod->B3SOIDDwpdiblbGiven = TRUE;
            break;                 
        case  B3SOIDD_MOD_WDROUT:             
            mod->B3SOIDDwdrout = value->rValue;
            mod->B3SOIDDwdroutGiven = TRUE;
            break;
        case B3SOIDD_MOD_WPVAG:
            mod->B3SOIDDwpvag = value->rValue;
            mod->B3SOIDDwpvagGiven = TRUE;
            break;                 
        case  B3SOIDD_MOD_WDELTA :
            mod->B3SOIDDwdelta = value->rValue;
            mod->B3SOIDDwdeltaGiven = TRUE;
            break;
        case  B3SOIDD_MOD_WAII :
            mod->B3SOIDDwaii = value->rValue;
            mod->B3SOIDDwaiiGiven = TRUE;
            break;
        case  B3SOIDD_MOD_WBII :
            mod->B3SOIDDwbii = value->rValue;
            mod->B3SOIDDwbiiGiven = TRUE;
            break;
        case  B3SOIDD_MOD_WCII :
            mod->B3SOIDDwcii = value->rValue;
            mod->B3SOIDDwciiGiven = TRUE;
            break;
        case  B3SOIDD_MOD_WDII :
            mod->B3SOIDDwdii = value->rValue;
            mod->B3SOIDDwdiiGiven = TRUE;
            break;
        case  B3SOIDD_MOD_WALPHA0 :
            mod->B3SOIDDwalpha0 = value->rValue;
            mod->B3SOIDDwalpha0Given = TRUE;
            break;
        case  B3SOIDD_MOD_WALPHA1 :
            mod->B3SOIDDwalpha1 = value->rValue;
            mod->B3SOIDDwalpha1Given = TRUE;
            break;
        case  B3SOIDD_MOD_WBETA0 :
            mod->B3SOIDDwbeta0 = value->rValue;
            mod->B3SOIDDwbeta0Given = TRUE;
            break;
        case  B3SOIDD_MOD_WAGIDL :
            mod->B3SOIDDwagidl = value->rValue;
            mod->B3SOIDDwagidlGiven = TRUE;
            break;
        case  B3SOIDD_MOD_WBGIDL :
            mod->B3SOIDDwbgidl = value->rValue;
            mod->B3SOIDDwbgidlGiven = TRUE;
            break;
        case  B3SOIDD_MOD_WNGIDL :
            mod->B3SOIDDwngidl = value->rValue;
            mod->B3SOIDDwngidlGiven = TRUE;
            break;
        case  B3SOIDD_MOD_WNTUN :
            mod->B3SOIDDwntun = value->rValue;
            mod->B3SOIDDwntunGiven = TRUE;
            break;
        case  B3SOIDD_MOD_WNDIODE :
            mod->B3SOIDDwndiode = value->rValue;
            mod->B3SOIDDwndiodeGiven = TRUE;
            break;
        case  B3SOIDD_MOD_WISBJT :
            mod->B3SOIDDwisbjt = value->rValue;
            mod->B3SOIDDwisbjtGiven = TRUE;
            break;
        case  B3SOIDD_MOD_WISDIF :
            mod->B3SOIDDwisdif = value->rValue;
            mod->B3SOIDDwisdifGiven = TRUE;
            break;
        case  B3SOIDD_MOD_WISREC :
            mod->B3SOIDDwisrec = value->rValue;
            mod->B3SOIDDwisrecGiven = TRUE;
            break;
        case  B3SOIDD_MOD_WISTUN :
            mod->B3SOIDDwistun = value->rValue;
            mod->B3SOIDDwistunGiven = TRUE;
            break;
        case  B3SOIDD_MOD_WEDL :
            mod->B3SOIDDwedl = value->rValue;
            mod->B3SOIDDwedlGiven = TRUE;
            break;
        case  B3SOIDD_MOD_WKBJT1 :
            mod->B3SOIDDwkbjt1 = value->rValue;
            mod->B3SOIDDwkbjt1Given = TRUE;
            break;
	/* CV Model */
        case  B3SOIDD_MOD_WVSDFB :
            mod->B3SOIDDwvsdfb = value->rValue;
            mod->B3SOIDDwvsdfbGiven = TRUE;
            break;
        case  B3SOIDD_MOD_WVSDTH :
            mod->B3SOIDDwvsdth = value->rValue;
            mod->B3SOIDDwvsdthGiven = TRUE;
            break;
        /* Cross-term Dependence */
        case B3SOIDD_MOD_PNPEAK:
            mod->B3SOIDDpnpeak = value->rValue;
            mod->B3SOIDDpnpeakGiven = TRUE;
            break;
        case B3SOIDD_MOD_PNSUB:
            mod->B3SOIDDpnsub = value->rValue;
            mod->B3SOIDDpnsubGiven = TRUE;
            break;
        case B3SOIDD_MOD_PNGATE:
            mod->B3SOIDDpngate = value->rValue;
            mod->B3SOIDDpngateGiven = TRUE;
            break;
        case B3SOIDD_MOD_PVTH0:
            mod->B3SOIDDpvth0 = value->rValue;
            mod->B3SOIDDpvth0Given = TRUE;
            break;
        case  B3SOIDD_MOD_PK1:
            mod->B3SOIDDpk1 = value->rValue;
            mod->B3SOIDDpk1Given = TRUE;
            break;
        case  B3SOIDD_MOD_PK2:
            mod->B3SOIDDpk2 = value->rValue;
            mod->B3SOIDDpk2Given = TRUE;
            break;
        case  B3SOIDD_MOD_PK3:
            mod->B3SOIDDpk3 = value->rValue;
            mod->B3SOIDDpk3Given = TRUE;
            break;
        case  B3SOIDD_MOD_PK3B:
            mod->B3SOIDDpk3b = value->rValue;
            mod->B3SOIDDpk3bGiven = TRUE;
            break;
        case  B3SOIDD_MOD_PVBSA:
            mod->B3SOIDDpvbsa = value->rValue;
            mod->B3SOIDDpvbsaGiven = TRUE;
            break;
        case  B3SOIDD_MOD_PDELP:
            mod->B3SOIDDpdelp = value->rValue;
            mod->B3SOIDDpdelpGiven = TRUE;
            break;
        case  B3SOIDD_MOD_PKB1 :
            mod->B3SOIDDpkb1 = value->rValue;
            mod->B3SOIDDpkb1Given = TRUE;
            break;
        case  B3SOIDD_MOD_PKB3 :
            mod->B3SOIDDpkb3 = value->rValue;
            mod->B3SOIDDpkb3Given = TRUE;
            break;
        case  B3SOIDD_MOD_PDVBD0 :
            mod->B3SOIDDpdvbd0 = value->rValue;
            mod->B3SOIDDpdvbd0Given = TRUE;
            break;
        case  B3SOIDD_MOD_PDVBD1 :
            mod->B3SOIDDpdvbd1 = value->rValue;
            mod->B3SOIDDpdvbd1Given = TRUE;
            break;
        case  B3SOIDD_MOD_PW0:
            mod->B3SOIDDpw0 = value->rValue;
            mod->B3SOIDDpw0Given = TRUE;
            break;
        case  B3SOIDD_MOD_PNLX:
            mod->B3SOIDDpnlx = value->rValue;
            mod->B3SOIDDpnlxGiven = TRUE;
            break;
        case  B3SOIDD_MOD_PDVT0:               
            mod->B3SOIDDpdvt0 = value->rValue;
            mod->B3SOIDDpdvt0Given = TRUE;
            break;
        case  B3SOIDD_MOD_PDVT1:             
            mod->B3SOIDDpdvt1 = value->rValue;
            mod->B3SOIDDpdvt1Given = TRUE;
            break;
        case  B3SOIDD_MOD_PDVT2:             
            mod->B3SOIDDpdvt2 = value->rValue;
            mod->B3SOIDDpdvt2Given = TRUE;
            break;
        case  B3SOIDD_MOD_PDVT0W:               
            mod->B3SOIDDpdvt0w = value->rValue;
            mod->B3SOIDDpdvt0wGiven = TRUE;
            break;
        case  B3SOIDD_MOD_PDVT1W:             
            mod->B3SOIDDpdvt1w = value->rValue;
            mod->B3SOIDDpdvt1wGiven = TRUE;
            break;
        case  B3SOIDD_MOD_PDVT2W:             
            mod->B3SOIDDpdvt2w = value->rValue;
            mod->B3SOIDDpdvt2wGiven = TRUE;
            break;
        case  B3SOIDD_MOD_PU0 :
            mod->B3SOIDDpu0 = value->rValue;
            mod->B3SOIDDpu0Given = TRUE;
            break;
        case B3SOIDD_MOD_PUA:
            mod->B3SOIDDpua = value->rValue;
            mod->B3SOIDDpuaGiven = TRUE;
            break;
        case B3SOIDD_MOD_PUB:
            mod->B3SOIDDpub = value->rValue;
            mod->B3SOIDDpubGiven = TRUE;
            break;
        case B3SOIDD_MOD_PUC:
            mod->B3SOIDDpuc = value->rValue;
            mod->B3SOIDDpucGiven = TRUE;
            break;
        case B3SOIDD_MOD_PVSAT:
            mod->B3SOIDDpvsat = value->rValue;
            mod->B3SOIDDpvsatGiven = TRUE;
            break;
        case B3SOIDD_MOD_PA0:
            mod->B3SOIDDpa0 = value->rValue;
            mod->B3SOIDDpa0Given = TRUE;
            break;
        case B3SOIDD_MOD_PAGS:
            mod->B3SOIDDpags= value->rValue;
            mod->B3SOIDDpagsGiven = TRUE;
            break;
        case  B3SOIDD_MOD_PB0 :
            mod->B3SOIDDpb0 = value->rValue;
            mod->B3SOIDDpb0Given = TRUE;
            break;
        case  B3SOIDD_MOD_PB1 :
            mod->B3SOIDDpb1 = value->rValue;
            mod->B3SOIDDpb1Given = TRUE;
            break;
        case B3SOIDD_MOD_PKETA:
            mod->B3SOIDDpketa = value->rValue;
            mod->B3SOIDDpketaGiven = TRUE;
            break;    
        case B3SOIDD_MOD_PABP:
            mod->B3SOIDDpabp = value->rValue;
            mod->B3SOIDDpabpGiven = TRUE;
            break;    
        case B3SOIDD_MOD_PMXC:
            mod->B3SOIDDpmxc = value->rValue;
            mod->B3SOIDDpmxcGiven = TRUE;
            break;    
        case B3SOIDD_MOD_PADICE0:
            mod->B3SOIDDpadice0 = value->rValue;
            mod->B3SOIDDpadice0Given = TRUE;
            break;    
        case B3SOIDD_MOD_PA1:
            mod->B3SOIDDpa1 = value->rValue;
            mod->B3SOIDDpa1Given = TRUE;
            break;
        case B3SOIDD_MOD_PA2:
            mod->B3SOIDDpa2 = value->rValue;
            mod->B3SOIDDpa2Given = TRUE;
            break;
        case B3SOIDD_MOD_PRDSW:
            mod->B3SOIDDprdsw = value->rValue;
            mod->B3SOIDDprdswGiven = TRUE;
            break;                     
        case B3SOIDD_MOD_PPRWB:
            mod->B3SOIDDpprwb = value->rValue;
            mod->B3SOIDDpprwbGiven = TRUE;
            break;                     
        case B3SOIDD_MOD_PPRWG:
            mod->B3SOIDDpprwg = value->rValue;
            mod->B3SOIDDpprwgGiven = TRUE;
            break;                     
        case  B3SOIDD_MOD_PWR :
            mod->B3SOIDDpwr = value->rValue;
            mod->B3SOIDDpwrGiven = TRUE;
            break;
        case  B3SOIDD_MOD_PNFACTOR :
            mod->B3SOIDDpnfactor = value->rValue;
            mod->B3SOIDDpnfactorGiven = TRUE;
            break;
        case  B3SOIDD_MOD_PDWG :
            mod->B3SOIDDpdwg = value->rValue;
            mod->B3SOIDDpdwgGiven = TRUE;
            break;
        case  B3SOIDD_MOD_PDWB :
            mod->B3SOIDDpdwb = value->rValue;
            mod->B3SOIDDpdwbGiven = TRUE;
            break;
        case B3SOIDD_MOD_PVOFF:
            mod->B3SOIDDpvoff = value->rValue;
            mod->B3SOIDDpvoffGiven = TRUE;
            break;
        case B3SOIDD_MOD_PETA0:
            mod->B3SOIDDpeta0 = value->rValue;
            mod->B3SOIDDpeta0Given = TRUE;
            break;                 
        case B3SOIDD_MOD_PETAB:
            mod->B3SOIDDpetab = value->rValue;
            mod->B3SOIDDpetabGiven = TRUE;
            break;                 
        case  B3SOIDD_MOD_PDSUB:             
            mod->B3SOIDDpdsub = value->rValue;
            mod->B3SOIDDpdsubGiven = TRUE;
            break;
        case  B3SOIDD_MOD_PCIT :
            mod->B3SOIDDpcit = value->rValue;
            mod->B3SOIDDpcitGiven = TRUE;
            break;
        case  B3SOIDD_MOD_PCDSC :
            mod->B3SOIDDpcdsc = value->rValue;
            mod->B3SOIDDpcdscGiven = TRUE;
            break;
        case  B3SOIDD_MOD_PCDSCB :
            mod->B3SOIDDpcdscb = value->rValue;
            mod->B3SOIDDpcdscbGiven = TRUE;
            break;
        case  B3SOIDD_MOD_PCDSCD :
            mod->B3SOIDDpcdscd = value->rValue;
            mod->B3SOIDDpcdscdGiven = TRUE;
            break;
        case B3SOIDD_MOD_PPCLM:
            mod->B3SOIDDppclm = value->rValue;
            mod->B3SOIDDppclmGiven = TRUE;
            break;                 
        case B3SOIDD_MOD_PPDIBL1:
            mod->B3SOIDDppdibl1 = value->rValue;
            mod->B3SOIDDppdibl1Given = TRUE;
            break;                 
        case B3SOIDD_MOD_PPDIBL2:
            mod->B3SOIDDppdibl2 = value->rValue;
            mod->B3SOIDDppdibl2Given = TRUE;
            break;                 
        case B3SOIDD_MOD_PPDIBLB:
            mod->B3SOIDDppdiblb = value->rValue;
            mod->B3SOIDDppdiblbGiven = TRUE;
            break;                 
        case  B3SOIDD_MOD_PDROUT:             
            mod->B3SOIDDpdrout = value->rValue;
            mod->B3SOIDDpdroutGiven = TRUE;
            break;
        case B3SOIDD_MOD_PPVAG:
            mod->B3SOIDDppvag = value->rValue;
            mod->B3SOIDDppvagGiven = TRUE;
            break;                 
        case  B3SOIDD_MOD_PDELTA :
            mod->B3SOIDDpdelta = value->rValue;
            mod->B3SOIDDpdeltaGiven = TRUE;
            break;
        case  B3SOIDD_MOD_PAII :
            mod->B3SOIDDpaii = value->rValue;
            mod->B3SOIDDpaiiGiven = TRUE;
            break;
        case  B3SOIDD_MOD_PBII :
            mod->B3SOIDDpbii = value->rValue;
            mod->B3SOIDDpbiiGiven = TRUE;
            break;
        case  B3SOIDD_MOD_PCII :
            mod->B3SOIDDpcii = value->rValue;
            mod->B3SOIDDpciiGiven = TRUE;
            break;
        case  B3SOIDD_MOD_PDII :
            mod->B3SOIDDpdii = value->rValue;
            mod->B3SOIDDpdiiGiven = TRUE;
            break;
        case  B3SOIDD_MOD_PALPHA0 :
            mod->B3SOIDDpalpha0 = value->rValue;
            mod->B3SOIDDpalpha0Given = TRUE;
            break;
        case  B3SOIDD_MOD_PALPHA1 :
            mod->B3SOIDDpalpha1 = value->rValue;
            mod->B3SOIDDpalpha1Given = TRUE;
            break;
        case  B3SOIDD_MOD_PBETA0 :
            mod->B3SOIDDpbeta0 = value->rValue;
            mod->B3SOIDDpbeta0Given = TRUE;
            break;
        case  B3SOIDD_MOD_PAGIDL :
            mod->B3SOIDDpagidl = value->rValue;
            mod->B3SOIDDpagidlGiven = TRUE;
            break;
        case  B3SOIDD_MOD_PBGIDL :
            mod->B3SOIDDpbgidl = value->rValue;
            mod->B3SOIDDpbgidlGiven = TRUE;
            break;
        case  B3SOIDD_MOD_PNGIDL :
            mod->B3SOIDDpngidl = value->rValue;
            mod->B3SOIDDpngidlGiven = TRUE;
            break;
        case  B3SOIDD_MOD_PNTUN :
            mod->B3SOIDDpntun = value->rValue;
            mod->B3SOIDDpntunGiven = TRUE;
            break;
        case  B3SOIDD_MOD_PNDIODE :
            mod->B3SOIDDpndiode = value->rValue;
            mod->B3SOIDDpndiodeGiven = TRUE;
            break;
        case  B3SOIDD_MOD_PISBJT :
            mod->B3SOIDDpisbjt = value->rValue;
            mod->B3SOIDDpisbjtGiven = TRUE;
            break;
        case  B3SOIDD_MOD_PISDIF :
            mod->B3SOIDDpisdif = value->rValue;
            mod->B3SOIDDpisdifGiven = TRUE;
            break;
        case  B3SOIDD_MOD_PISREC :
            mod->B3SOIDDpisrec = value->rValue;
            mod->B3SOIDDpisrecGiven = TRUE;
            break;
        case  B3SOIDD_MOD_PISTUN :
            mod->B3SOIDDpistun = value->rValue;
            mod->B3SOIDDpistunGiven = TRUE;
            break;
        case  B3SOIDD_MOD_PEDL :
            mod->B3SOIDDpedl = value->rValue;
            mod->B3SOIDDpedlGiven = TRUE;
            break;
        case  B3SOIDD_MOD_PKBJT1 :
            mod->B3SOIDDpkbjt1 = value->rValue;
            mod->B3SOIDDpkbjt1Given = TRUE;
            break;
	/* CV Model */
        case  B3SOIDD_MOD_PVSDFB :
            mod->B3SOIDDpvsdfb = value->rValue;
            mod->B3SOIDDpvsdfbGiven = TRUE;
            break;
        case  B3SOIDD_MOD_PVSDTH :
            mod->B3SOIDDpvsdth = value->rValue;
            mod->B3SOIDDpvsdthGiven = TRUE;
            break;
/* Added for binning - END */

        case  B3SOIDD_MOD_NMOS  :
            if(value->iValue) {
                mod->B3SOIDDtype = 1;
                mod->B3SOIDDtypeGiven = TRUE;
            }
            break;
        case  B3SOIDD_MOD_PMOS  :
            if(value->iValue) {
                mod->B3SOIDDtype = - 1;
                mod->B3SOIDDtypeGiven = TRUE;
            }
            break;
        default:
            return(E_BADPARM);
    }
    return(OK);
}


