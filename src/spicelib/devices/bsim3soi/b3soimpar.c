/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1998 Samuel Fung, Dennis Sinitsky and Stephen Tang
File: b3soimpar.c          98/5/01
Modified by Pin Su and Jan Feng	99/2/15
Modified by Pin Su 99/4/30
Modified by Wei Jin 99/9/27
Modified by Pin Su 00/3/1
Modified by Pin Su 01/2/15
Modified by Pin Su 02/3/5
Modified by Pin Su 02/5/20
Modified by Paolo Nenzi 2002
**********/


#include "ngspice.h"
#include "b3soidef.h"
#include "ifsim.h"
#include "sperror.h"
#include "suffix.h"


int
B3SOImParam(int param, IFvalue *value, GENmodel *inMod)
{
    B3SOImodel *mod = (B3SOImodel*)inMod;
    switch(param)
    {  
 
	case  B3SOI_MOD_MOBMOD :
            mod->B3SOImobMod = value->iValue;
            mod->B3SOImobModGiven = TRUE;
            break;
        case  B3SOI_MOD_BINUNIT :
            mod->B3SOIbinUnit = value->iValue;
            mod->B3SOIbinUnitGiven = TRUE;
            break;
        case  B3SOI_MOD_PARAMCHK :
            mod->B3SOIparamChk = value->iValue;
            mod->B3SOIparamChkGiven = TRUE;
            break;
        case  B3SOI_MOD_CAPMOD :
            mod->B3SOIcapMod = value->iValue;
            mod->B3SOIcapModGiven = TRUE;
            break;
        case  B3SOI_MOD_SHMOD :
            mod->B3SOIshMod = value->iValue;
            mod->B3SOIshModGiven = TRUE;
            break;
        case  B3SOI_MOD_NOIMOD :
            mod->B3SOInoiMod = value->iValue;
            mod->B3SOInoiModGiven = TRUE;
            break;
        case  B3SOI_MOD_VERSION :
            mod->B3SOIversion = value->rValue;
            mod->B3SOIversionGiven = TRUE;
            break;
        case  B3SOI_MOD_TOX :
            mod->B3SOItox = value->rValue;
            mod->B3SOItoxGiven = TRUE;
            break;
/* v2.2.3 */
        case  B3SOI_MOD_DTOXCV :
            mod->B3SOIdtoxcv = value->rValue;
            mod->B3SOIdtoxcvGiven = TRUE;
            break;

        case  B3SOI_MOD_CDSC :
            mod->B3SOIcdsc = value->rValue;
            mod->B3SOIcdscGiven = TRUE;
            break;
        case  B3SOI_MOD_CDSCB :
            mod->B3SOIcdscb = value->rValue;
            mod->B3SOIcdscbGiven = TRUE;
            break;

        case  B3SOI_MOD_CDSCD :
            mod->B3SOIcdscd = value->rValue;
            mod->B3SOIcdscdGiven = TRUE;
            break;

        case  B3SOI_MOD_CIT :
            mod->B3SOIcit = value->rValue;
            mod->B3SOIcitGiven = TRUE;
            break;
        case  B3SOI_MOD_NFACTOR :
            mod->B3SOInfactor = value->rValue;
            mod->B3SOInfactorGiven = TRUE;
            break;
        case B3SOI_MOD_VSAT:
            mod->B3SOIvsat = value->rValue;
            mod->B3SOIvsatGiven = TRUE;
            break;
        case B3SOI_MOD_A0:
            mod->B3SOIa0 = value->rValue;
            mod->B3SOIa0Given = TRUE;
            break;
        
        case B3SOI_MOD_AGS:
            mod->B3SOIags= value->rValue;
            mod->B3SOIagsGiven = TRUE;
            break;
        
        case B3SOI_MOD_A1:
            mod->B3SOIa1 = value->rValue;
            mod->B3SOIa1Given = TRUE;
            break;
        case B3SOI_MOD_A2:
            mod->B3SOIa2 = value->rValue;
            mod->B3SOIa2Given = TRUE;
            break;
        case B3SOI_MOD_AT:
            mod->B3SOIat = value->rValue;
            mod->B3SOIatGiven = TRUE;
            break;
        case B3SOI_MOD_KETA:
            mod->B3SOIketa = value->rValue;
            mod->B3SOIketaGiven = TRUE;
            break;    
        case B3SOI_MOD_NSUB:
            mod->B3SOInsub = value->rValue;
            mod->B3SOInsubGiven = TRUE;
            break;
        case B3SOI_MOD_NPEAK:
            mod->B3SOInpeak = value->rValue;
            mod->B3SOInpeakGiven = TRUE;
	    if (mod->B3SOInpeak > 1.0e20)
		mod->B3SOInpeak *= 1.0e-6;
            break;
        case B3SOI_MOD_NGATE:
            mod->B3SOIngate = value->rValue;
            mod->B3SOIngateGiven = TRUE;
	    if (mod->B3SOIngate > 1.0e23)
		mod->B3SOIngate *= 1.0e-6;
            break;
        case B3SOI_MOD_GAMMA1:
            mod->B3SOIgamma1 = value->rValue;
            mod->B3SOIgamma1Given = TRUE;
            break;
        case B3SOI_MOD_GAMMA2:
            mod->B3SOIgamma2 = value->rValue;
            mod->B3SOIgamma2Given = TRUE;
            break;
        case B3SOI_MOD_VBX:
            mod->B3SOIvbx = value->rValue;
            mod->B3SOIvbxGiven = TRUE;
            break;
        case B3SOI_MOD_VBM:
            mod->B3SOIvbm = value->rValue;
            mod->B3SOIvbmGiven = TRUE;
            break;
        case B3SOI_MOD_XT:
            mod->B3SOIxt = value->rValue;
            mod->B3SOIxtGiven = TRUE;
            break;
        case  B3SOI_MOD_K1:
            mod->B3SOIk1 = value->rValue;
            mod->B3SOIk1Given = TRUE;
            break;
        case  B3SOI_MOD_KT1:
            mod->B3SOIkt1 = value->rValue;
            mod->B3SOIkt1Given = TRUE;
            break;
        case  B3SOI_MOD_KT1L:
            mod->B3SOIkt1l = value->rValue;
            mod->B3SOIkt1lGiven = TRUE;
            break;
        case  B3SOI_MOD_KT2:
            mod->B3SOIkt2 = value->rValue;
            mod->B3SOIkt2Given = TRUE;
            break;
        case  B3SOI_MOD_K2:
            mod->B3SOIk2 = value->rValue;
            mod->B3SOIk2Given = TRUE;
            break;
        case  B3SOI_MOD_K3:
            mod->B3SOIk3 = value->rValue;
            mod->B3SOIk3Given = TRUE;
            break;
        case  B3SOI_MOD_K3B:
            mod->B3SOIk3b = value->rValue;
            mod->B3SOIk3bGiven = TRUE;
            break;
        case  B3SOI_MOD_NLX:
            mod->B3SOInlx = value->rValue;
            mod->B3SOInlxGiven = TRUE;
            break;
        case  B3SOI_MOD_W0:
            mod->B3SOIw0 = value->rValue;
            mod->B3SOIw0Given = TRUE;
            break;
        case  B3SOI_MOD_DVT0:               
            mod->B3SOIdvt0 = value->rValue;
            mod->B3SOIdvt0Given = TRUE;
            break;
        case  B3SOI_MOD_DVT1:             
            mod->B3SOIdvt1 = value->rValue;
            mod->B3SOIdvt1Given = TRUE;
            break;
        case  B3SOI_MOD_DVT2:             
            mod->B3SOIdvt2 = value->rValue;
            mod->B3SOIdvt2Given = TRUE;
            break;
        case  B3SOI_MOD_DVT0W:               
            mod->B3SOIdvt0w = value->rValue;
            mod->B3SOIdvt0wGiven = TRUE;
            break;
        case  B3SOI_MOD_DVT1W:             
            mod->B3SOIdvt1w = value->rValue;
            mod->B3SOIdvt1wGiven = TRUE;
            break;
        case  B3SOI_MOD_DVT2W:             
            mod->B3SOIdvt2w = value->rValue;
            mod->B3SOIdvt2wGiven = TRUE;
            break;
        case  B3SOI_MOD_DROUT:             
            mod->B3SOIdrout = value->rValue;
            mod->B3SOIdroutGiven = TRUE;
            break;
        case  B3SOI_MOD_DSUB:             
            mod->B3SOIdsub = value->rValue;
            mod->B3SOIdsubGiven = TRUE;
            break;
        case B3SOI_MOD_VTH0:
            mod->B3SOIvth0 = value->rValue;
            mod->B3SOIvth0Given = TRUE;
            break;
        case B3SOI_MOD_UA:
            mod->B3SOIua = value->rValue;
            mod->B3SOIuaGiven = TRUE;
            break;
        case B3SOI_MOD_UA1:
            mod->B3SOIua1 = value->rValue;
            mod->B3SOIua1Given = TRUE;
            break;
        case B3SOI_MOD_UB:
            mod->B3SOIub = value->rValue;
            mod->B3SOIubGiven = TRUE;
            break;
        case B3SOI_MOD_UB1:
            mod->B3SOIub1 = value->rValue;
            mod->B3SOIub1Given = TRUE;
            break;
        case B3SOI_MOD_UC:
            mod->B3SOIuc = value->rValue;
            mod->B3SOIucGiven = TRUE;
            break;
        case B3SOI_MOD_UC1:
            mod->B3SOIuc1 = value->rValue;
            mod->B3SOIuc1Given = TRUE;
            break;
        case  B3SOI_MOD_U0 :
            mod->B3SOIu0 = value->rValue;
            mod->B3SOIu0Given = TRUE;
            break;
        case  B3SOI_MOD_UTE :
            mod->B3SOIute = value->rValue;
            mod->B3SOIuteGiven = TRUE;
            break;
        case B3SOI_MOD_VOFF:
            mod->B3SOIvoff = value->rValue;
            mod->B3SOIvoffGiven = TRUE;
            break;
        case  B3SOI_MOD_DELTA :
            mod->B3SOIdelta = value->rValue;
            mod->B3SOIdeltaGiven = TRUE;
            break;
        case B3SOI_MOD_RDSW:
            mod->B3SOIrdsw = value->rValue;
            mod->B3SOIrdswGiven = TRUE;
            break;                     
        case B3SOI_MOD_PRWG:
            mod->B3SOIprwg = value->rValue;
            mod->B3SOIprwgGiven = TRUE;
            break;                     
        case B3SOI_MOD_PRWB:
            mod->B3SOIprwb = value->rValue;
            mod->B3SOIprwbGiven = TRUE;
            break;                     
        case B3SOI_MOD_PRT:
            mod->B3SOIprt = value->rValue;
            mod->B3SOIprtGiven = TRUE;
            break;                     
        case B3SOI_MOD_ETA0:
            mod->B3SOIeta0 = value->rValue;
            mod->B3SOIeta0Given = TRUE;
            break;                 
        case B3SOI_MOD_ETAB:
            mod->B3SOIetab = value->rValue;
            mod->B3SOIetabGiven = TRUE;
            break;                 
        case B3SOI_MOD_PCLM:
            mod->B3SOIpclm = value->rValue;
            mod->B3SOIpclmGiven = TRUE;
            break;                 
        case B3SOI_MOD_PDIBL1:
            mod->B3SOIpdibl1 = value->rValue;
            mod->B3SOIpdibl1Given = TRUE;
            break;                 
        case B3SOI_MOD_PDIBL2:
            mod->B3SOIpdibl2 = value->rValue;
            mod->B3SOIpdibl2Given = TRUE;
            break;                 
        case B3SOI_MOD_PDIBLB:
            mod->B3SOIpdiblb = value->rValue;
            mod->B3SOIpdiblbGiven = TRUE;
            break;                 
        case B3SOI_MOD_PVAG:
            mod->B3SOIpvag = value->rValue;
            mod->B3SOIpvagGiven = TRUE;
            break;                 
        case  B3SOI_MOD_WR :
            mod->B3SOIwr = value->rValue;
            mod->B3SOIwrGiven = TRUE;
            break;
        case  B3SOI_MOD_DWG :
            mod->B3SOIdwg = value->rValue;
            mod->B3SOIdwgGiven = TRUE;
            break;
        case  B3SOI_MOD_DWB :
            mod->B3SOIdwb = value->rValue;
            mod->B3SOIdwbGiven = TRUE;
            break;
        case  B3SOI_MOD_B0 :
            mod->B3SOIb0 = value->rValue;
            mod->B3SOIb0Given = TRUE;
            break;
        case  B3SOI_MOD_B1 :
            mod->B3SOIb1 = value->rValue;
            mod->B3SOIb1Given = TRUE;
            break;
        case  B3SOI_MOD_ALPHA0 :
            mod->B3SOIalpha0 = value->rValue;
            mod->B3SOIalpha0Given = TRUE;
            break;

        case  B3SOI_MOD_CGSL :
            mod->B3SOIcgsl = value->rValue;
            mod->B3SOIcgslGiven = TRUE;
            break;
        case  B3SOI_MOD_CGDL :
            mod->B3SOIcgdl = value->rValue;
            mod->B3SOIcgdlGiven = TRUE;
            break;
        case  B3SOI_MOD_CKAPPA :
            mod->B3SOIckappa = value->rValue;
            mod->B3SOIckappaGiven = TRUE;
            break;
        case  B3SOI_MOD_CF :
            mod->B3SOIcf = value->rValue;
            mod->B3SOIcfGiven = TRUE;
            break;
        case  B3SOI_MOD_CLC :
            mod->B3SOIclc = value->rValue;
            mod->B3SOIclcGiven = TRUE;
            break;
        case  B3SOI_MOD_CLE :
            mod->B3SOIcle = value->rValue;
            mod->B3SOIcleGiven = TRUE;
            break;
        case  B3SOI_MOD_DWC :
            mod->B3SOIdwc = value->rValue;
            mod->B3SOIdwcGiven = TRUE;
            break;
        case  B3SOI_MOD_DLC :
            mod->B3SOIdlc = value->rValue;
            mod->B3SOIdlcGiven = TRUE;
            break;
        case  B3SOI_MOD_TBOX :
            mod->B3SOItbox = value->rValue;
            mod->B3SOItboxGiven = TRUE;
            break;
        case  B3SOI_MOD_TSI :
            mod->B3SOItsi = value->rValue;
            mod->B3SOItsiGiven = TRUE;
            break;
        case  B3SOI_MOD_XJ :
            mod->B3SOIxj = value->rValue;
            mod->B3SOIxjGiven = TRUE;
            break;
        case  B3SOI_MOD_RBODY :
            mod->B3SOIrbody = value->rValue;
            mod->B3SOIrbodyGiven = TRUE;
            break;
        case  B3SOI_MOD_RBSH :
            mod->B3SOIrbsh = value->rValue;
            mod->B3SOIrbshGiven = TRUE;
            break;
        case  B3SOI_MOD_RTH0 :
            mod->B3SOIrth0 = value->rValue;
            mod->B3SOIrth0Given = TRUE;
            break;
        case  B3SOI_MOD_CTH0 :
            mod->B3SOIcth0 = value->rValue;
            mod->B3SOIcth0Given = TRUE;
            break;
        case  B3SOI_MOD_NGIDL :
            mod->B3SOIngidl = value->rValue;
            mod->B3SOIngidlGiven = TRUE;
            break;
        case  B3SOI_MOD_AGIDL :
            mod->B3SOIagidl = value->rValue;
            mod->B3SOIagidlGiven = TRUE;
            break;
        case  B3SOI_MOD_BGIDL :
            mod->B3SOIbgidl = value->rValue;
            mod->B3SOIbgidlGiven = TRUE;
            break;
        case  B3SOI_MOD_NDIODE :
            mod->B3SOIndiode = value->rValue;
            mod->B3SOIndiodeGiven = TRUE;
            break;
        case  B3SOI_MOD_XBJT :
            mod->B3SOIxbjt = value->rValue;
            mod->B3SOIxbjtGiven = TRUE;
            break;

        case  B3SOI_MOD_XDIF :
            mod->B3SOIxdif = value->rValue;
            mod->B3SOIxdifGiven = TRUE;
            break;

        case  B3SOI_MOD_XREC :
            mod->B3SOIxrec = value->rValue;
            mod->B3SOIxrecGiven = TRUE;
            break;
        case  B3SOI_MOD_XTUN :
            mod->B3SOIxtun = value->rValue;
            mod->B3SOIxtunGiven = TRUE;
            break;
        case  B3SOI_MOD_TT :
            mod->B3SOItt = value->rValue;
            mod->B3SOIttGiven = TRUE;
            break;
        case  B3SOI_MOD_VSDTH :
            mod->B3SOIvsdth = value->rValue;
            mod->B3SOIvsdthGiven = TRUE;
            break;
        case  B3SOI_MOD_VSDFB :
            mod->B3SOIvsdfb = value->rValue;
            mod->B3SOIvsdfbGiven = TRUE;
            break;
        case  B3SOI_MOD_CSDMIN :
            mod->B3SOIcsdmin = value->rValue;
            mod->B3SOIcsdminGiven = TRUE;
            break;
        case  B3SOI_MOD_ASD :
            mod->B3SOIasd = value->rValue;
            mod->B3SOIasdGiven = TRUE;
            break;


        case  B3SOI_MOD_TNOM :
            mod->B3SOItnom = value->rValue + 273.15;
            mod->B3SOItnomGiven = TRUE;
            break;
        case  B3SOI_MOD_CGSO :
            mod->B3SOIcgso = value->rValue;
            mod->B3SOIcgsoGiven = TRUE;
            break;
        case  B3SOI_MOD_CGDO :
            mod->B3SOIcgdo = value->rValue;
            mod->B3SOIcgdoGiven = TRUE;
            break;
        case  B3SOI_MOD_CGEO :
            mod->B3SOIcgeo = value->rValue;
            mod->B3SOIcgeoGiven = TRUE;
            break;
        case  B3SOI_MOD_XPART :
            mod->B3SOIxpart = value->rValue;
            mod->B3SOIxpartGiven = TRUE;
            break;
        case  B3SOI_MOD_RSH :
            mod->B3SOIsheetResistance = value->rValue;
            mod->B3SOIsheetResistanceGiven = TRUE;
            break;
        case  B3SOI_MOD_PBSWG :
            mod->B3SOIGatesidewallJctPotential = value->rValue;
            mod->B3SOIGatesidewallJctPotentialGiven = TRUE;
            break;
        case  B3SOI_MOD_MJSWG :
            mod->B3SOIbodyJctGateSideGradingCoeff = value->rValue;
            mod->B3SOIbodyJctGateSideGradingCoeffGiven = TRUE;
            break;
        case  B3SOI_MOD_CJSWG :
            mod->B3SOIunitLengthGateSidewallJctCap = value->rValue;
            mod->B3SOIunitLengthGateSidewallJctCapGiven = TRUE;
            break;
        case  B3SOI_MOD_CSDESW :
            mod->B3SOIcsdesw = value->rValue;
            mod->B3SOIcsdeswGiven = TRUE;
            break;
        case  B3SOI_MOD_LINT :
            mod->B3SOILint = value->rValue;
            mod->B3SOILintGiven = TRUE;
            break;
        case  B3SOI_MOD_LL :
            mod->B3SOILl = value->rValue;
            mod->B3SOILlGiven = TRUE;
            break;
/* v2.2.3 */
        case  B3SOI_MOD_LLC :
            mod->B3SOILlc = value->rValue;
            mod->B3SOILlcGiven = TRUE;
            break;

        case  B3SOI_MOD_LLN :
            mod->B3SOILln = value->rValue;
            mod->B3SOILlnGiven = TRUE;
            break;
        case  B3SOI_MOD_LW :
            mod->B3SOILw = value->rValue;
            mod->B3SOILwGiven = TRUE;
            break;
/* v2.2.3 */
        case  B3SOI_MOD_LWC :
            mod->B3SOILwc = value->rValue;
            mod->B3SOILwcGiven = TRUE;
            break;

        case  B3SOI_MOD_LWN :
            mod->B3SOILwn = value->rValue;
            mod->B3SOILwnGiven = TRUE;
            break;
        case  B3SOI_MOD_LWL :
            mod->B3SOILwl = value->rValue;
            mod->B3SOILwlGiven = TRUE;
            break;
/* v2.2.3 */
        case  B3SOI_MOD_LWLC :
            mod->B3SOILwlc = value->rValue;
            mod->B3SOILwlcGiven = TRUE;
            break;

        case  B3SOI_MOD_WINT :
            mod->B3SOIWint = value->rValue;
            mod->B3SOIWintGiven = TRUE;
            break;
        case  B3SOI_MOD_WL :
            mod->B3SOIWl = value->rValue;
            mod->B3SOIWlGiven = TRUE;
            break;
/* v2.2.3 */
        case  B3SOI_MOD_WLC :
            mod->B3SOIWlc = value->rValue;
            mod->B3SOIWlcGiven = TRUE;
            break;

        case  B3SOI_MOD_WLN :
            mod->B3SOIWln = value->rValue;
            mod->B3SOIWlnGiven = TRUE;
            break;
        case  B3SOI_MOD_WW :
            mod->B3SOIWw = value->rValue;
            mod->B3SOIWwGiven = TRUE;
            break;
/* v2.2.3 */
        case  B3SOI_MOD_WWC :
            mod->B3SOIWwc = value->rValue;
            mod->B3SOIWwcGiven = TRUE;
            break;

        case  B3SOI_MOD_WWN :
            mod->B3SOIWwn = value->rValue;
            mod->B3SOIWwnGiven = TRUE;
            break;
        case  B3SOI_MOD_WWL :
            mod->B3SOIWwl = value->rValue;
            mod->B3SOIWwlGiven = TRUE;
            break;
/* v2.2.3 */
        case  B3SOI_MOD_WWLC :
            mod->B3SOIWwlc = value->rValue;
            mod->B3SOIWwlcGiven = TRUE;
            break;

        case  B3SOI_MOD_NOIA :
            mod->B3SOIoxideTrapDensityA = value->rValue;
            mod->B3SOIoxideTrapDensityAGiven = TRUE;
            break;
        case  B3SOI_MOD_NOIB :
            mod->B3SOIoxideTrapDensityB = value->rValue;
            mod->B3SOIoxideTrapDensityBGiven = TRUE;
            break;
        case  B3SOI_MOD_NOIC :
            mod->B3SOIoxideTrapDensityC = value->rValue;
            mod->B3SOIoxideTrapDensityCGiven = TRUE;
            break;
        case  B3SOI_MOD_NOIF :
            mod->B3SOInoif = value->rValue;
            mod->B3SOInoifGiven = TRUE;
            break;
        case  B3SOI_MOD_EM :
            mod->B3SOIem = value->rValue;
            mod->B3SOIemGiven = TRUE;
            break;
        case  B3SOI_MOD_EF :
            mod->B3SOIef = value->rValue;
            mod->B3SOIefGiven = TRUE;
            break;
        case  B3SOI_MOD_AF :
            mod->B3SOIaf = value->rValue;
            mod->B3SOIafGiven = TRUE;
            break;
        case  B3SOI_MOD_KF :
            mod->B3SOIkf = value->rValue;
            mod->B3SOIkfGiven = TRUE;
            break;

/* v3.0 */
        case  B3SOI_MOD_SOIMOD:
            mod->B3SOIsoiMod = value->rValue;
            mod->B3SOIsoimodGiven = TRUE;
            break;
        case  B3SOI_MOD_VBSA:
            mod->B3SOIvbsa = value->rValue;
            mod->B3SOIvbsaGiven = TRUE;
            break;
        case  B3SOI_MOD_NOFFFD :
            mod->B3SOInofffd = value->rValue;
            mod->B3SOInofffdGiven = TRUE;
            break;
        case  B3SOI_MOD_VOFFFD:
            mod->B3SOIvofffd = value->rValue;
            mod->B3SOIvofffdGiven = TRUE;
            break;
        case  B3SOI_MOD_K1B:
            mod->B3SOIk1b = value->rValue;
            mod->B3SOIk1bGiven = TRUE;
            break;
        case  B3SOI_MOD_K2B:
            mod->B3SOIk2b = value->rValue;
            mod->B3SOIk2bGiven = TRUE;
            break;
        case  B3SOI_MOD_DK2B:
            mod->B3SOIdk2b = value->rValue;
            mod->B3SOIdk2bGiven = TRUE;
            break;
        case  B3SOI_MOD_DVBD0:
            mod->B3SOIdvbd0 = value->rValue;
            mod->B3SOIdvbd0Given = TRUE;
            break;
        case  B3SOI_MOD_DVBD1:
            mod->B3SOIdvbd1 = value->rValue;
            mod->B3SOIdvbd1Given = TRUE;
            break;
        case  B3SOI_MOD_MOINFD:
            mod->B3SOImoinFD = value->rValue;
            mod->B3SOImoinFDGiven = TRUE;
            break;


/* v2.2 release */
        case  B3SOI_MOD_WTH0 :
            mod->B3SOIwth0 = value->rValue;
            mod->B3SOIwth0Given = TRUE;
            break;
        case  B3SOI_MOD_RHALO :
            mod->B3SOIrhalo = value->rValue;
            mod->B3SOIrhaloGiven = TRUE;
            break;
        case  B3SOI_MOD_NTOX :
            mod->B3SOIntox = value->rValue;
            mod->B3SOIntoxGiven = TRUE;
            break;
        case  B3SOI_MOD_TOXREF :
            mod->B3SOItoxref = value->rValue;
            mod->B3SOItoxrefGiven = TRUE;
            break;
        case  B3SOI_MOD_EBG :
            mod->B3SOIebg = value->rValue;
            mod->B3SOIebgGiven = TRUE;
            break;
        case  B3SOI_MOD_VEVB :
            mod->B3SOIvevb = value->rValue;
            mod->B3SOIvevbGiven = TRUE;
            break;
        case  B3SOI_MOD_ALPHAGB1 :
            mod->B3SOIalphaGB1 = value->rValue;
            mod->B3SOIalphaGB1Given = TRUE;
            break;
        case  B3SOI_MOD_BETAGB1 :
            mod->B3SOIbetaGB1 = value->rValue;
            mod->B3SOIbetaGB1Given = TRUE;
            break;
        case  B3SOI_MOD_VGB1 :
            mod->B3SOIvgb1 = value->rValue;
            mod->B3SOIvgb1Given = TRUE;
            break;
        case  B3SOI_MOD_VECB :
            mod->B3SOIvecb = value->rValue;
            mod->B3SOIvecbGiven = TRUE;
            break;
        case  B3SOI_MOD_ALPHAGB2 :
            mod->B3SOIalphaGB2 = value->rValue;
            mod->B3SOIalphaGB2Given = TRUE;
            break;
        case  B3SOI_MOD_BETAGB2 :
            mod->B3SOIbetaGB2 = value->rValue;
            mod->B3SOIbetaGB2Given = TRUE;
            break;
        case  B3SOI_MOD_VGB2 :
            mod->B3SOIvgb2 = value->rValue;
            mod->B3SOIvgb2Given = TRUE;
            break;
        case  B3SOI_MOD_TOXQM :
            mod->B3SOItoxqm = value->rValue;
            mod->B3SOItoxqmGiven = TRUE;
            break;
        case  B3SOI_MOD_VOXH :
            mod->B3SOIvoxh = value->rValue;
            mod->B3SOIvoxhGiven = TRUE;
            break;
        case  B3SOI_MOD_DELTAVOX :
            mod->B3SOIdeltavox = value->rValue;
            mod->B3SOIdeltavoxGiven = TRUE;
            break;

/* v3.0 */
        case  B3SOI_MOD_IGBMOD :
            mod->B3SOIigbMod = value->iValue;
            mod->B3SOIigbModGiven = TRUE;
            break;
        case  B3SOI_MOD_IGCMOD :
            mod->B3SOIigcMod = value->iValue;
            mod->B3SOIigcModGiven = TRUE;
            break;
        case  B3SOI_MOD_AIGC :
            mod->B3SOIaigc = value->rValue;
            mod->B3SOIaigcGiven = TRUE;
            break;
        case  B3SOI_MOD_BIGC :
            mod->B3SOIbigc = value->rValue;
            mod->B3SOIbigcGiven = TRUE;
            break;
        case  B3SOI_MOD_CIGC :
            mod->B3SOIcigc = value->rValue;
            mod->B3SOIcigcGiven = TRUE;
            break;
        case  B3SOI_MOD_AIGSD :
            mod->B3SOIaigsd = value->rValue;
            mod->B3SOIaigsdGiven = TRUE;
            break;
        case  B3SOI_MOD_BIGSD :
            mod->B3SOIbigsd = value->rValue;
            mod->B3SOIbigsdGiven = TRUE;
            break;
        case  B3SOI_MOD_CIGSD :
            mod->B3SOIcigsd = value->rValue;
            mod->B3SOIcigsdGiven = TRUE;
            break;
        case  B3SOI_MOD_NIGC :
            mod->B3SOInigc = value->rValue;
            mod->B3SOInigcGiven = TRUE;
            break;
        case  B3SOI_MOD_PIGCD :
            mod->B3SOIpigcd = value->rValue;
            mod->B3SOIpigcdGiven = TRUE;
            break;
        case  B3SOI_MOD_POXEDGE :
            mod->B3SOIpoxedge = value->rValue;
            mod->B3SOIpoxedgeGiven = TRUE;
            break;
        case  B3SOI_MOD_DLCIG :
            mod->B3SOIdlcig = value->rValue;
            mod->B3SOIdlcigGiven = TRUE;
            break;

/* v2.0 release */
        case  B3SOI_MOD_K1W1 :         
            mod->B3SOIk1w1 = value->rValue;
            mod->B3SOIk1w1Given = TRUE;
            break;
        case  B3SOI_MOD_K1W2 :
            mod->B3SOIk1w2 = value->rValue;
            mod->B3SOIk1w2Given = TRUE;
            break;
        case  B3SOI_MOD_KETAS :
            mod->B3SOIketas = value->rValue;
            mod->B3SOIketasGiven = TRUE;
            break;
        case  B3SOI_MOD_DWBC :
            mod->B3SOIdwbc = value->rValue;
            mod->B3SOIdwbcGiven = TRUE;
            break;
        case  B3SOI_MOD_BETA0 :
            mod->B3SOIbeta0 = value->rValue;
            mod->B3SOIbeta0Given = TRUE;
            break;
        case  B3SOI_MOD_BETA1 :
            mod->B3SOIbeta1 = value->rValue;
            mod->B3SOIbeta1Given = TRUE;
            break;
        case  B3SOI_MOD_BETA2 :
            mod->B3SOIbeta2 = value->rValue;
            mod->B3SOIbeta2Given = TRUE;
            break;
        case  B3SOI_MOD_VDSATII0 :
            mod->B3SOIvdsatii0 = value->rValue;
            mod->B3SOIvdsatii0Given = TRUE;
            break;
        case  B3SOI_MOD_TII :
            mod->B3SOItii = value->rValue;
            mod->B3SOItiiGiven = TRUE;
            break;
        case  B3SOI_MOD_LII :
            mod->B3SOIlii = value->rValue;
            mod->B3SOIliiGiven = TRUE;
            break;
        case  B3SOI_MOD_SII0 :
            mod->B3SOIsii0 = value->rValue;
            mod->B3SOIsii0Given = TRUE;
            break;
        case  B3SOI_MOD_SII1 :
            mod->B3SOIsii1 = value->rValue;
            mod->B3SOIsii1Given = TRUE;
            break;
        case  B3SOI_MOD_SII2 :
            mod->B3SOIsii2 = value->rValue;
            mod->B3SOIsii2Given = TRUE;
            break;
        case  B3SOI_MOD_SIID :
            mod->B3SOIsiid = value->rValue;
            mod->B3SOIsiidGiven = TRUE;
            break;
        case  B3SOI_MOD_FBJTII :
            mod->B3SOIfbjtii = value->rValue;
            mod->B3SOIfbjtiiGiven = TRUE;
            break;
        case  B3SOI_MOD_ESATII :
            mod->B3SOIesatii = value->rValue;
            mod->B3SOIesatiiGiven = TRUE;
            break;
        case  B3SOI_MOD_NTUN :
            mod->B3SOIntun = value->rValue;
            mod->B3SOIntunGiven = TRUE;
            break;
        case  B3SOI_MOD_NRECF0 :
            mod->B3SOInrecf0 = value->rValue;
            mod->B3SOInrecf0Given = TRUE;
            break;
        case  B3SOI_MOD_NRECR0 :
            mod->B3SOInrecr0 = value->rValue;
            mod->B3SOInrecr0Given = TRUE;
            break;
        case  B3SOI_MOD_ISBJT :
            mod->B3SOIisbjt = value->rValue;
            mod->B3SOIisbjtGiven = TRUE;
            break;
        case  B3SOI_MOD_ISDIF :
            mod->B3SOIisdif = value->rValue;
            mod->B3SOIisdifGiven = TRUE;
            break;
        case  B3SOI_MOD_ISREC :
            mod->B3SOIisrec = value->rValue;
            mod->B3SOIisrecGiven = TRUE;
            break;
        case  B3SOI_MOD_ISTUN :
            mod->B3SOIistun = value->rValue;
            mod->B3SOIistunGiven = TRUE;
            break;
        case  B3SOI_MOD_LN :
            mod->B3SOIln = value->rValue;
            mod->B3SOIlnGiven = TRUE;
            break;
        case  B3SOI_MOD_VREC0 :
            mod->B3SOIvrec0 = value->rValue;
            mod->B3SOIvrec0Given = TRUE;
            break;
        case  B3SOI_MOD_VTUN0 :
            mod->B3SOIvtun0 = value->rValue;
            mod->B3SOIvtun0Given = TRUE;
            break;
        case  B3SOI_MOD_NBJT :
            mod->B3SOInbjt = value->rValue;
            mod->B3SOInbjtGiven = TRUE;
            break;
        case  B3SOI_MOD_LBJT0 :
            mod->B3SOIlbjt0 = value->rValue;
            mod->B3SOIlbjt0Given = TRUE;
            break;
        case  B3SOI_MOD_LDIF0 :
            mod->B3SOIldif0 = value->rValue;
            mod->B3SOIldif0Given = TRUE;
            break;
        case  B3SOI_MOD_VABJT :
            mod->B3SOIvabjt = value->rValue;
            mod->B3SOIvabjtGiven = TRUE;
            break;
        case  B3SOI_MOD_AELY :
            mod->B3SOIaely = value->rValue;
            mod->B3SOIaelyGiven = TRUE;
            break;
        case  B3SOI_MOD_AHLI :
            mod->B3SOIahli = value->rValue;
            mod->B3SOIahliGiven = TRUE;
            break;
        case  B3SOI_MOD_NDIF :
            mod->B3SOIndif = value->rValue;
            mod->B3SOIndifGiven = TRUE;
            break;
        case  B3SOI_MOD_NTRECF :
            mod->B3SOIntrecf = value->rValue;
            mod->B3SOIntrecfGiven = TRUE;
            break;
        case  B3SOI_MOD_NTRECR :
            mod->B3SOIntrecr = value->rValue;
            mod->B3SOIntrecrGiven = TRUE;
            break;
        case  B3SOI_MOD_DLCB :
            mod->B3SOIdlcb = value->rValue;
            mod->B3SOIdlcbGiven = TRUE;
            break;
        case  B3SOI_MOD_FBODY :
            mod->B3SOIfbody = value->rValue;
            mod->B3SOIfbodyGiven = TRUE;
            break;
        case  B3SOI_MOD_TCJSWG :
            mod->B3SOItcjswg = value->rValue;
            mod->B3SOItcjswgGiven = TRUE;
            break;
        case  B3SOI_MOD_TPBSWG :
            mod->B3SOItpbswg = value->rValue;
            mod->B3SOItpbswgGiven = TRUE;
            break;
        case  B3SOI_MOD_ACDE :
            mod->B3SOIacde = value->rValue;
            mod->B3SOIacdeGiven = TRUE;
            break;
        case  B3SOI_MOD_MOIN :
            mod->B3SOImoin = value->rValue;
            mod->B3SOImoinGiven = TRUE;
            break;
        case  B3SOI_MOD_DELVT :
            mod->B3SOIdelvt = value->rValue;
            mod->B3SOIdelvtGiven = TRUE;
            break;
        case  B3SOI_MOD_KB1 :
            mod->B3SOIkb1 = value->rValue;
            mod->B3SOIkb1Given = TRUE;
            break;
        case  B3SOI_MOD_DLBG :
            mod->B3SOIdlbg = value->rValue;
            mod->B3SOIdlbgGiven = TRUE;
            break;

/* Added for binning - START */
        /* Length Dependence */
/* v3.0 */
        case  B3SOI_MOD_LAIGC :
            mod->B3SOIlaigc = value->rValue;
            mod->B3SOIlaigcGiven = TRUE;
            break;
        case  B3SOI_MOD_LBIGC :
            mod->B3SOIlbigc = value->rValue;
            mod->B3SOIlbigcGiven = TRUE;
            break;
        case  B3SOI_MOD_LCIGC :
            mod->B3SOIlcigc = value->rValue;
            mod->B3SOIlcigcGiven = TRUE;
            break;
        case  B3SOI_MOD_LAIGSD :
            mod->B3SOIlaigsd = value->rValue;
            mod->B3SOIlaigsdGiven = TRUE;
            break;
        case  B3SOI_MOD_LBIGSD :
            mod->B3SOIlbigsd = value->rValue;
            mod->B3SOIlbigsdGiven = TRUE;
            break;
        case  B3SOI_MOD_LCIGSD :
            mod->B3SOIlcigsd = value->rValue;
            mod->B3SOIlcigsdGiven = TRUE;
            break;
        case  B3SOI_MOD_LNIGC :
            mod->B3SOIlnigc = value->rValue;
            mod->B3SOIlnigcGiven = TRUE;
            break;
        case  B3SOI_MOD_LPIGCD :
            mod->B3SOIlpigcd = value->rValue;
            mod->B3SOIlpigcdGiven = TRUE;
            break;
        case  B3SOI_MOD_LPOXEDGE :
            mod->B3SOIlpoxedge = value->rValue;
            mod->B3SOIlpoxedgeGiven = TRUE;
            break;

        case B3SOI_MOD_LNPEAK:
            mod->B3SOIlnpeak = value->rValue;
            mod->B3SOIlnpeakGiven = TRUE;
            break;
        case B3SOI_MOD_LNSUB:
            mod->B3SOIlnsub = value->rValue;
            mod->B3SOIlnsubGiven = TRUE;
            break;
        case B3SOI_MOD_LNGATE:
            mod->B3SOIlngate = value->rValue;
            mod->B3SOIlngateGiven = TRUE;
            break;
        case B3SOI_MOD_LVTH0:
            mod->B3SOIlvth0 = value->rValue;
            mod->B3SOIlvth0Given = TRUE;
            break;
        case  B3SOI_MOD_LK1:
            mod->B3SOIlk1 = value->rValue;
            mod->B3SOIlk1Given = TRUE;
            break;
        case  B3SOI_MOD_LK1W1:
            mod->B3SOIlk1w1 = value->rValue;
            mod->B3SOIlk1w1Given = TRUE;
            break;
        case  B3SOI_MOD_LK1W2:
            mod->B3SOIlk1w2 = value->rValue;
            mod->B3SOIlk1w2Given = TRUE;
            break;
        case  B3SOI_MOD_LK2:
            mod->B3SOIlk2 = value->rValue;
            mod->B3SOIlk2Given = TRUE;
            break;
        case  B3SOI_MOD_LK3:
            mod->B3SOIlk3 = value->rValue;
            mod->B3SOIlk3Given = TRUE;
            break;
        case  B3SOI_MOD_LK3B:
            mod->B3SOIlk3b = value->rValue;
            mod->B3SOIlk3bGiven = TRUE;
            break;
        case  B3SOI_MOD_LKB1 :
            mod->B3SOIlkb1 = value->rValue;
            mod->B3SOIlkb1Given = TRUE;
            break;
        case  B3SOI_MOD_LW0:
            mod->B3SOIlw0 = value->rValue;
            mod->B3SOIlw0Given = TRUE;
            break;
        case  B3SOI_MOD_LNLX:
            mod->B3SOIlnlx = value->rValue;
            mod->B3SOIlnlxGiven = TRUE;
            break;
        case  B3SOI_MOD_LDVT0:               
            mod->B3SOIldvt0 = value->rValue;
            mod->B3SOIldvt0Given = TRUE;
            break;
        case  B3SOI_MOD_LDVT1:             
            mod->B3SOIldvt1 = value->rValue;
            mod->B3SOIldvt1Given = TRUE;
            break;
        case  B3SOI_MOD_LDVT2:             
            mod->B3SOIldvt2 = value->rValue;
            mod->B3SOIldvt2Given = TRUE;
            break;
        case  B3SOI_MOD_LDVT0W:               
            mod->B3SOIldvt0w = value->rValue;
            mod->B3SOIldvt0wGiven = TRUE;
            break;
        case  B3SOI_MOD_LDVT1W:             
            mod->B3SOIldvt1w = value->rValue;
            mod->B3SOIldvt1wGiven = TRUE;
            break;
        case  B3SOI_MOD_LDVT2W:             
            mod->B3SOIldvt2w = value->rValue;
            mod->B3SOIldvt2wGiven = TRUE;
            break;
        case  B3SOI_MOD_LU0 :
            mod->B3SOIlu0 = value->rValue;
            mod->B3SOIlu0Given = TRUE;
            break;
        case B3SOI_MOD_LUA:
            mod->B3SOIlua = value->rValue;
            mod->B3SOIluaGiven = TRUE;
            break;
        case B3SOI_MOD_LUB:
            mod->B3SOIlub = value->rValue;
            mod->B3SOIlubGiven = TRUE;
            break;
        case B3SOI_MOD_LUC:
            mod->B3SOIluc = value->rValue;
            mod->B3SOIlucGiven = TRUE;
            break;
        case B3SOI_MOD_LVSAT:
            mod->B3SOIlvsat = value->rValue;
            mod->B3SOIlvsatGiven = TRUE;
            break;
        case B3SOI_MOD_LA0:
            mod->B3SOIla0 = value->rValue;
            mod->B3SOIla0Given = TRUE;
            break;
        case B3SOI_MOD_LAGS:
            mod->B3SOIlags= value->rValue;
            mod->B3SOIlagsGiven = TRUE;
            break;
        case  B3SOI_MOD_LB0 :
            mod->B3SOIlb0 = value->rValue;
            mod->B3SOIlb0Given = TRUE;
            break;
        case  B3SOI_MOD_LB1 :
            mod->B3SOIlb1 = value->rValue;
            mod->B3SOIlb1Given = TRUE;
            break;
        case B3SOI_MOD_LKETA:
            mod->B3SOIlketa = value->rValue;
            mod->B3SOIlketaGiven = TRUE;
            break;    
        case B3SOI_MOD_LKETAS:
            mod->B3SOIlketas = value->rValue;
            mod->B3SOIlketasGiven = TRUE;
            break;    
        case B3SOI_MOD_LA1:
            mod->B3SOIla1 = value->rValue;
            mod->B3SOIla1Given = TRUE;
            break;
        case B3SOI_MOD_LA2:
            mod->B3SOIla2 = value->rValue;
            mod->B3SOIla2Given = TRUE;
            break;
        case B3SOI_MOD_LRDSW:
            mod->B3SOIlrdsw = value->rValue;
            mod->B3SOIlrdswGiven = TRUE;
            break;                     
        case B3SOI_MOD_LPRWB:
            mod->B3SOIlprwb = value->rValue;
            mod->B3SOIlprwbGiven = TRUE;
            break;                     
        case B3SOI_MOD_LPRWG:
            mod->B3SOIlprwg = value->rValue;
            mod->B3SOIlprwgGiven = TRUE;
            break;                     
        case  B3SOI_MOD_LWR :
            mod->B3SOIlwr = value->rValue;
            mod->B3SOIlwrGiven = TRUE;
            break;
        case  B3SOI_MOD_LNFACTOR :
            mod->B3SOIlnfactor = value->rValue;
            mod->B3SOIlnfactorGiven = TRUE;
            break;
        case  B3SOI_MOD_LDWG :
            mod->B3SOIldwg = value->rValue;
            mod->B3SOIldwgGiven = TRUE;
            break;
        case  B3SOI_MOD_LDWB :
            mod->B3SOIldwb = value->rValue;
            mod->B3SOIldwbGiven = TRUE;
            break;
        case B3SOI_MOD_LVOFF:
            mod->B3SOIlvoff = value->rValue;
            mod->B3SOIlvoffGiven = TRUE;
            break;
        case B3SOI_MOD_LETA0:
            mod->B3SOIleta0 = value->rValue;
            mod->B3SOIleta0Given = TRUE;
            break;                 
        case B3SOI_MOD_LETAB:
            mod->B3SOIletab = value->rValue;
            mod->B3SOIletabGiven = TRUE;
            break;                 
        case  B3SOI_MOD_LDSUB:             
            mod->B3SOIldsub = value->rValue;
            mod->B3SOIldsubGiven = TRUE;
            break;
        case  B3SOI_MOD_LCIT :
            mod->B3SOIlcit = value->rValue;
            mod->B3SOIlcitGiven = TRUE;
            break;
        case  B3SOI_MOD_LCDSC :
            mod->B3SOIlcdsc = value->rValue;
            mod->B3SOIlcdscGiven = TRUE;
            break;
        case  B3SOI_MOD_LCDSCB :
            mod->B3SOIlcdscb = value->rValue;
            mod->B3SOIlcdscbGiven = TRUE;
            break;
        case  B3SOI_MOD_LCDSCD :
            mod->B3SOIlcdscd = value->rValue;
            mod->B3SOIlcdscdGiven = TRUE;
            break;
        case B3SOI_MOD_LPCLM:
            mod->B3SOIlpclm = value->rValue;
            mod->B3SOIlpclmGiven = TRUE;
            break;                 
        case B3SOI_MOD_LPDIBL1:
            mod->B3SOIlpdibl1 = value->rValue;
            mod->B3SOIlpdibl1Given = TRUE;
            break;                 
        case B3SOI_MOD_LPDIBL2:
            mod->B3SOIlpdibl2 = value->rValue;
            mod->B3SOIlpdibl2Given = TRUE;
            break;                 
        case B3SOI_MOD_LPDIBLB:
            mod->B3SOIlpdiblb = value->rValue;
            mod->B3SOIlpdiblbGiven = TRUE;
            break;                 
        case  B3SOI_MOD_LDROUT:             
            mod->B3SOIldrout = value->rValue;
            mod->B3SOIldroutGiven = TRUE;
            break;
        case B3SOI_MOD_LPVAG:
            mod->B3SOIlpvag = value->rValue;
            mod->B3SOIlpvagGiven = TRUE;
            break;                 
        case  B3SOI_MOD_LDELTA :
            mod->B3SOIldelta = value->rValue;
            mod->B3SOIldeltaGiven = TRUE;
            break;
        case  B3SOI_MOD_LALPHA0 :
            mod->B3SOIlalpha0 = value->rValue;
            mod->B3SOIlalpha0Given = TRUE;
            break;
        case  B3SOI_MOD_LFBJTII :
            mod->B3SOIlfbjtii = value->rValue;
            mod->B3SOIlfbjtiiGiven = TRUE;
            break;
        case  B3SOI_MOD_LBETA0 :
            mod->B3SOIlbeta0 = value->rValue;
            mod->B3SOIlbeta0Given = TRUE;
            break;
        case  B3SOI_MOD_LBETA1 :
            mod->B3SOIlbeta1 = value->rValue;
            mod->B3SOIlbeta1Given = TRUE;
            break;
        case  B3SOI_MOD_LBETA2 :
            mod->B3SOIlbeta2 = value->rValue;
            mod->B3SOIlbeta2Given = TRUE;
            break;
        case  B3SOI_MOD_LVDSATII0 :
            mod->B3SOIlvdsatii0 = value->rValue;
            mod->B3SOIlvdsatii0Given = TRUE;
            break;
        case  B3SOI_MOD_LLII :
            mod->B3SOIllii = value->rValue;
            mod->B3SOIlliiGiven = TRUE;
            break;
        case  B3SOI_MOD_LESATII :
            mod->B3SOIlesatii = value->rValue;
            mod->B3SOIlesatiiGiven = TRUE;
            break;
        case  B3SOI_MOD_LSII0 :
            mod->B3SOIlsii0 = value->rValue;
            mod->B3SOIlsii0Given = TRUE;
            break;
        case  B3SOI_MOD_LSII1 :
            mod->B3SOIlsii1 = value->rValue;
            mod->B3SOIlsii1Given = TRUE;
            break;
        case  B3SOI_MOD_LSII2 :
            mod->B3SOIlsii2 = value->rValue;
            mod->B3SOIlsii2Given = TRUE;
            break;
        case  B3SOI_MOD_LSIID :
            mod->B3SOIlsiid = value->rValue;
            mod->B3SOIlsiidGiven = TRUE;
            break;
        case  B3SOI_MOD_LAGIDL :
            mod->B3SOIlagidl = value->rValue;
            mod->B3SOIlagidlGiven = TRUE;
            break;
        case  B3SOI_MOD_LBGIDL :
            mod->B3SOIlbgidl = value->rValue;
            mod->B3SOIlbgidlGiven = TRUE;
            break;
        case  B3SOI_MOD_LNGIDL :
            mod->B3SOIlngidl = value->rValue;
            mod->B3SOIlngidlGiven = TRUE;
            break;
        case  B3SOI_MOD_LNTUN :
            mod->B3SOIlntun = value->rValue;
            mod->B3SOIlntunGiven = TRUE;
            break;
        case  B3SOI_MOD_LNDIODE :
            mod->B3SOIlndiode = value->rValue;
            mod->B3SOIlndiodeGiven = TRUE;
            break;
        case  B3SOI_MOD_LNRECF0 :
            mod->B3SOIlnrecf0 = value->rValue;
            mod->B3SOIlnrecf0Given = TRUE;
            break;
        case  B3SOI_MOD_LNRECR0 :
            mod->B3SOIlnrecr0 = value->rValue;
            mod->B3SOIlnrecr0Given = TRUE;
            break;
        case  B3SOI_MOD_LISBJT :
            mod->B3SOIlisbjt = value->rValue;
            mod->B3SOIlisbjtGiven = TRUE;
            break;
        case  B3SOI_MOD_LISDIF :
            mod->B3SOIlisdif = value->rValue;
            mod->B3SOIlisdifGiven = TRUE;
            break;
        case  B3SOI_MOD_LISREC :
            mod->B3SOIlisrec = value->rValue;
            mod->B3SOIlisrecGiven = TRUE;
            break;
        case  B3SOI_MOD_LISTUN :
            mod->B3SOIlistun = value->rValue;
            mod->B3SOIlistunGiven = TRUE;
            break;
        case  B3SOI_MOD_LVREC0 :
            mod->B3SOIlvrec0 = value->rValue;
            mod->B3SOIlvrec0Given = TRUE;
            break;
        case  B3SOI_MOD_LVTUN0 :
            mod->B3SOIlvtun0 = value->rValue;
            mod->B3SOIlvtun0Given = TRUE;
            break;
        case  B3SOI_MOD_LNBJT :
            mod->B3SOIlnbjt = value->rValue;
            mod->B3SOIlnbjtGiven = TRUE;
            break;
        case  B3SOI_MOD_LLBJT0 :
            mod->B3SOIllbjt0 = value->rValue;
            mod->B3SOIllbjt0Given = TRUE;
            break;
        case  B3SOI_MOD_LVABJT :
            mod->B3SOIlvabjt = value->rValue;
            mod->B3SOIlvabjtGiven = TRUE;
            break;
        case  B3SOI_MOD_LAELY :
            mod->B3SOIlaely = value->rValue;
            mod->B3SOIlaelyGiven = TRUE;
            break;
        case  B3SOI_MOD_LAHLI :
            mod->B3SOIlahli = value->rValue;
            mod->B3SOIlahliGiven = TRUE;
            break;
	/* CV Model */
        case  B3SOI_MOD_LVSDFB :
            mod->B3SOIlvsdfb = value->rValue;
            mod->B3SOIlvsdfbGiven = TRUE;
            break;
        case  B3SOI_MOD_LVSDTH :
            mod->B3SOIlvsdth = value->rValue;
            mod->B3SOIlvsdthGiven = TRUE;
            break;
        case  B3SOI_MOD_LDELVT :
            mod->B3SOIldelvt = value->rValue;
            mod->B3SOIldelvtGiven = TRUE;
            break;
        case  B3SOI_MOD_LACDE :
            mod->B3SOIlacde = value->rValue;
            mod->B3SOIlacdeGiven = TRUE;
            break;
        case  B3SOI_MOD_LMOIN :
            mod->B3SOIlmoin = value->rValue;
            mod->B3SOIlmoinGiven = TRUE;
            break;

        /* Width Dependence */
/* v3.0 */
        case  B3SOI_MOD_WAIGC :
            mod->B3SOIwaigc = value->rValue;
            mod->B3SOIwaigcGiven = TRUE;
            break;
        case  B3SOI_MOD_WBIGC :
            mod->B3SOIwbigc = value->rValue;
            mod->B3SOIwbigcGiven = TRUE;
            break;
        case  B3SOI_MOD_WCIGC :
            mod->B3SOIwcigc = value->rValue;
            mod->B3SOIwcigcGiven = TRUE;
            break;
        case  B3SOI_MOD_WAIGSD :
            mod->B3SOIwaigsd = value->rValue;
            mod->B3SOIwaigsdGiven = TRUE;
            break;
        case  B3SOI_MOD_WBIGSD :
            mod->B3SOIwbigsd = value->rValue;
            mod->B3SOIwbigsdGiven = TRUE;
            break;
        case  B3SOI_MOD_WCIGSD :
            mod->B3SOIwcigsd = value->rValue;
            mod->B3SOIwcigsdGiven = TRUE;
            break;
        case  B3SOI_MOD_WNIGC :
            mod->B3SOIwnigc = value->rValue;
            mod->B3SOIwnigcGiven = TRUE;
            break;
        case  B3SOI_MOD_WPIGCD :
            mod->B3SOIwpigcd = value->rValue;
            mod->B3SOIwpigcdGiven = TRUE;
            break;
        case  B3SOI_MOD_WPOXEDGE :
            mod->B3SOIwpoxedge = value->rValue;
            mod->B3SOIwpoxedgeGiven = TRUE;
            break;

        case B3SOI_MOD_WNPEAK:
            mod->B3SOIwnpeak = value->rValue;
            mod->B3SOIwnpeakGiven = TRUE;
            break;
        case B3SOI_MOD_WNSUB:
            mod->B3SOIwnsub = value->rValue;
            mod->B3SOIwnsubGiven = TRUE;
            break;
        case B3SOI_MOD_WNGATE:
            mod->B3SOIwngate = value->rValue;
            mod->B3SOIwngateGiven = TRUE;
            break;
        case B3SOI_MOD_WVTH0:
            mod->B3SOIwvth0 = value->rValue;
            mod->B3SOIwvth0Given = TRUE;
            break;
        case  B3SOI_MOD_WK1:
            mod->B3SOIwk1 = value->rValue;
            mod->B3SOIwk1Given = TRUE;
            break;
        case  B3SOI_MOD_WK1W1:
            mod->B3SOIwk1w1 = value->rValue;
            mod->B3SOIwk1w1Given = TRUE;
            break;
        case  B3SOI_MOD_WK1W2:
            mod->B3SOIwk1w2 = value->rValue;
            mod->B3SOIwk1w2Given = TRUE;
            break;
        case  B3SOI_MOD_WK2:
            mod->B3SOIwk2 = value->rValue;
            mod->B3SOIwk2Given = TRUE;
            break;
        case  B3SOI_MOD_WK3:
            mod->B3SOIwk3 = value->rValue;
            mod->B3SOIwk3Given = TRUE;
            break;
        case  B3SOI_MOD_WK3B:
            mod->B3SOIwk3b = value->rValue;
            mod->B3SOIwk3bGiven = TRUE;
            break;
        case  B3SOI_MOD_WKB1 :
            mod->B3SOIwkb1 = value->rValue;
            mod->B3SOIwkb1Given = TRUE;
            break;
        case  B3SOI_MOD_WW0:
            mod->B3SOIww0 = value->rValue;
            mod->B3SOIww0Given = TRUE;
            break;
        case  B3SOI_MOD_WNLX:
            mod->B3SOIwnlx = value->rValue;
            mod->B3SOIwnlxGiven = TRUE;
            break;
        case  B3SOI_MOD_WDVT0:               
            mod->B3SOIwdvt0 = value->rValue;
            mod->B3SOIwdvt0Given = TRUE;
            break;
        case  B3SOI_MOD_WDVT1:             
            mod->B3SOIwdvt1 = value->rValue;
            mod->B3SOIwdvt1Given = TRUE;
            break;
        case  B3SOI_MOD_WDVT2:             
            mod->B3SOIwdvt2 = value->rValue;
            mod->B3SOIwdvt2Given = TRUE;
            break;
        case  B3SOI_MOD_WDVT0W:               
            mod->B3SOIwdvt0w = value->rValue;
            mod->B3SOIwdvt0wGiven = TRUE;
            break;
        case  B3SOI_MOD_WDVT1W:             
            mod->B3SOIwdvt1w = value->rValue;
            mod->B3SOIwdvt1wGiven = TRUE;
            break;
        case  B3SOI_MOD_WDVT2W:             
            mod->B3SOIwdvt2w = value->rValue;
            mod->B3SOIwdvt2wGiven = TRUE;
            break;
        case  B3SOI_MOD_WU0 :
            mod->B3SOIwu0 = value->rValue;
            mod->B3SOIwu0Given = TRUE;
            break;
        case B3SOI_MOD_WUA:
            mod->B3SOIwua = value->rValue;
            mod->B3SOIwuaGiven = TRUE;
            break;
        case B3SOI_MOD_WUB:
            mod->B3SOIwub = value->rValue;
            mod->B3SOIwubGiven = TRUE;
            break;
        case B3SOI_MOD_WUC:
            mod->B3SOIwuc = value->rValue;
            mod->B3SOIwucGiven = TRUE;
            break;
        case B3SOI_MOD_WVSAT:
            mod->B3SOIwvsat = value->rValue;
            mod->B3SOIwvsatGiven = TRUE;
            break;
        case B3SOI_MOD_WA0:
            mod->B3SOIwa0 = value->rValue;
            mod->B3SOIwa0Given = TRUE;
            break;
        case B3SOI_MOD_WAGS:
            mod->B3SOIwags= value->rValue;
            mod->B3SOIwagsGiven = TRUE;
            break;
        case  B3SOI_MOD_WB0 :
            mod->B3SOIwb0 = value->rValue;
            mod->B3SOIwb0Given = TRUE;
            break;
        case  B3SOI_MOD_WB1 :
            mod->B3SOIwb1 = value->rValue;
            mod->B3SOIwb1Given = TRUE;
            break;
        case B3SOI_MOD_WKETA:
            mod->B3SOIwketa = value->rValue;
            mod->B3SOIwketaGiven = TRUE;
            break;    
        case B3SOI_MOD_WKETAS:
            mod->B3SOIwketas = value->rValue;
            mod->B3SOIwketasGiven = TRUE;
            break;    
        case B3SOI_MOD_WA1:
            mod->B3SOIwa1 = value->rValue;
            mod->B3SOIwa1Given = TRUE;
            break;
        case B3SOI_MOD_WA2:
            mod->B3SOIwa2 = value->rValue;
            mod->B3SOIwa2Given = TRUE;
            break;
        case B3SOI_MOD_WRDSW:
            mod->B3SOIwrdsw = value->rValue;
            mod->B3SOIwrdswGiven = TRUE;
            break;                     
        case B3SOI_MOD_WPRWB:
            mod->B3SOIwprwb = value->rValue;
            mod->B3SOIwprwbGiven = TRUE;
            break;                     
        case B3SOI_MOD_WPRWG:
            mod->B3SOIwprwg = value->rValue;
            mod->B3SOIwprwgGiven = TRUE;
            break;                     
        case  B3SOI_MOD_WWR :
            mod->B3SOIwwr = value->rValue;
            mod->B3SOIwwrGiven = TRUE;
            break;
        case  B3SOI_MOD_WNFACTOR :
            mod->B3SOIwnfactor = value->rValue;
            mod->B3SOIwnfactorGiven = TRUE;
            break;
        case  B3SOI_MOD_WDWG :
            mod->B3SOIwdwg = value->rValue;
            mod->B3SOIwdwgGiven = TRUE;
            break;
        case  B3SOI_MOD_WDWB :
            mod->B3SOIwdwb = value->rValue;
            mod->B3SOIwdwbGiven = TRUE;
            break;
        case B3SOI_MOD_WVOFF:
            mod->B3SOIwvoff = value->rValue;
            mod->B3SOIwvoffGiven = TRUE;
            break;
        case B3SOI_MOD_WETA0:
            mod->B3SOIweta0 = value->rValue;
            mod->B3SOIweta0Given = TRUE;
            break;                 
        case B3SOI_MOD_WETAB:
            mod->B3SOIwetab = value->rValue;
            mod->B3SOIwetabGiven = TRUE;
            break;                 
        case  B3SOI_MOD_WDSUB:             
            mod->B3SOIwdsub = value->rValue;
            mod->B3SOIwdsubGiven = TRUE;
            break;
        case  B3SOI_MOD_WCIT :
            mod->B3SOIwcit = value->rValue;
            mod->B3SOIwcitGiven = TRUE;
            break;
        case  B3SOI_MOD_WCDSC :
            mod->B3SOIwcdsc = value->rValue;
            mod->B3SOIwcdscGiven = TRUE;
            break;
        case  B3SOI_MOD_WCDSCB :
            mod->B3SOIwcdscb = value->rValue;
            mod->B3SOIwcdscbGiven = TRUE;
            break;
        case  B3SOI_MOD_WCDSCD :
            mod->B3SOIwcdscd = value->rValue;
            mod->B3SOIwcdscdGiven = TRUE;
            break;
        case B3SOI_MOD_WPCLM:
            mod->B3SOIwpclm = value->rValue;
            mod->B3SOIwpclmGiven = TRUE;
            break;                 
        case B3SOI_MOD_WPDIBL1:
            mod->B3SOIwpdibl1 = value->rValue;
            mod->B3SOIwpdibl1Given = TRUE;
            break;                 
        case B3SOI_MOD_WPDIBL2:
            mod->B3SOIwpdibl2 = value->rValue;
            mod->B3SOIwpdibl2Given = TRUE;
            break;                 
        case B3SOI_MOD_WPDIBLB:
            mod->B3SOIwpdiblb = value->rValue;
            mod->B3SOIwpdiblbGiven = TRUE;
            break;                 
        case  B3SOI_MOD_WDROUT:             
            mod->B3SOIwdrout = value->rValue;
            mod->B3SOIwdroutGiven = TRUE;
            break;
        case B3SOI_MOD_WPVAG:
            mod->B3SOIwpvag = value->rValue;
            mod->B3SOIwpvagGiven = TRUE;
            break;                 
        case  B3SOI_MOD_WDELTA :
            mod->B3SOIwdelta = value->rValue;
            mod->B3SOIwdeltaGiven = TRUE;
            break;
        case  B3SOI_MOD_WALPHA0 :
            mod->B3SOIwalpha0 = value->rValue;
            mod->B3SOIwalpha0Given = TRUE;
            break;
        case  B3SOI_MOD_WFBJTII :
            mod->B3SOIwfbjtii = value->rValue;
            mod->B3SOIwfbjtiiGiven = TRUE;
            break;
        case  B3SOI_MOD_WBETA0 :
            mod->B3SOIwbeta0 = value->rValue;
            mod->B3SOIwbeta0Given = TRUE;
            break;
        case  B3SOI_MOD_WBETA1 :
            mod->B3SOIwbeta1 = value->rValue;
            mod->B3SOIwbeta1Given = TRUE;
            break;
        case  B3SOI_MOD_WBETA2 :
            mod->B3SOIwbeta2 = value->rValue;
            mod->B3SOIwbeta2Given = TRUE;
            break;
        case  B3SOI_MOD_WVDSATII0 :
            mod->B3SOIwvdsatii0 = value->rValue;
            mod->B3SOIwvdsatii0Given = TRUE;
            break;
        case  B3SOI_MOD_WLII :
            mod->B3SOIwlii = value->rValue;
            mod->B3SOIwliiGiven = TRUE;
            break;
        case  B3SOI_MOD_WESATII :
            mod->B3SOIwesatii = value->rValue;
            mod->B3SOIwesatiiGiven = TRUE;
            break;
        case  B3SOI_MOD_WSII0 :
            mod->B3SOIwsii0 = value->rValue;
            mod->B3SOIwsii0Given = TRUE;
            break;
        case  B3SOI_MOD_WSII1 :
            mod->B3SOIwsii1 = value->rValue;
            mod->B3SOIwsii1Given = TRUE;
            break;
        case  B3SOI_MOD_WSII2 :
            mod->B3SOIwsii2 = value->rValue;
            mod->B3SOIwsii2Given = TRUE;
            break;
        case  B3SOI_MOD_WSIID :
            mod->B3SOIwsiid = value->rValue;
            mod->B3SOIwsiidGiven = TRUE;
            break;
        case  B3SOI_MOD_WAGIDL :
            mod->B3SOIwagidl = value->rValue;
            mod->B3SOIwagidlGiven = TRUE;
            break;
        case  B3SOI_MOD_WBGIDL :
            mod->B3SOIwbgidl = value->rValue;
            mod->B3SOIwbgidlGiven = TRUE;
            break;
        case  B3SOI_MOD_WNGIDL :
            mod->B3SOIwngidl = value->rValue;
            mod->B3SOIwngidlGiven = TRUE;
            break;
        case  B3SOI_MOD_WNTUN :
            mod->B3SOIwntun = value->rValue;
            mod->B3SOIwntunGiven = TRUE;
            break;
        case  B3SOI_MOD_WNDIODE :
            mod->B3SOIwndiode = value->rValue;
            mod->B3SOIwndiodeGiven = TRUE;
            break;
        case  B3SOI_MOD_WNRECF0 :
            mod->B3SOIwnrecf0 = value->rValue;
            mod->B3SOIwnrecf0Given = TRUE;
            break;
        case  B3SOI_MOD_WNRECR0 :
            mod->B3SOIwnrecr0 = value->rValue;
            mod->B3SOIwnrecr0Given = TRUE;
            break;
        case  B3SOI_MOD_WISBJT :
            mod->B3SOIwisbjt = value->rValue;
            mod->B3SOIwisbjtGiven = TRUE;
            break;
        case  B3SOI_MOD_WISDIF :
            mod->B3SOIwisdif = value->rValue;
            mod->B3SOIwisdifGiven = TRUE;
            break;
        case  B3SOI_MOD_WISREC :
            mod->B3SOIwisrec = value->rValue;
            mod->B3SOIwisrecGiven = TRUE;
            break;
        case  B3SOI_MOD_WISTUN :
            mod->B3SOIwistun = value->rValue;
            mod->B3SOIwistunGiven = TRUE;
            break;
        case  B3SOI_MOD_WVREC0 :
            mod->B3SOIwvrec0 = value->rValue;
            mod->B3SOIwvrec0Given = TRUE;
            break;
        case  B3SOI_MOD_WVTUN0 :
            mod->B3SOIwvtun0 = value->rValue;
            mod->B3SOIwvtun0Given = TRUE;
            break;
        case  B3SOI_MOD_WNBJT :
            mod->B3SOIwnbjt = value->rValue;
            mod->B3SOIwnbjtGiven = TRUE;
            break;
        case  B3SOI_MOD_WLBJT0 :
            mod->B3SOIwlbjt0 = value->rValue;
            mod->B3SOIwlbjt0Given = TRUE;
            break;
        case  B3SOI_MOD_WVABJT :
            mod->B3SOIwvabjt = value->rValue;
            mod->B3SOIwvabjtGiven = TRUE;
            break;
        case  B3SOI_MOD_WAELY :
            mod->B3SOIwaely = value->rValue;
            mod->B3SOIwaelyGiven = TRUE;
            break;
        case  B3SOI_MOD_WAHLI :
            mod->B3SOIwahli = value->rValue;
            mod->B3SOIwahliGiven = TRUE;
            break;
	/* CV Model */
        case  B3SOI_MOD_WVSDFB :
            mod->B3SOIwvsdfb = value->rValue;
            mod->B3SOIwvsdfbGiven = TRUE;
            break;
        case  B3SOI_MOD_WVSDTH :
            mod->B3SOIwvsdth = value->rValue;
            mod->B3SOIwvsdthGiven = TRUE;
            break;
        case  B3SOI_MOD_WDELVT :
            mod->B3SOIwdelvt = value->rValue;
            mod->B3SOIwdelvtGiven = TRUE;
            break;
        case  B3SOI_MOD_WACDE :
            mod->B3SOIwacde = value->rValue;
            mod->B3SOIwacdeGiven = TRUE;
            break;
        case  B3SOI_MOD_WMOIN :
            mod->B3SOIwmoin = value->rValue;
            mod->B3SOIwmoinGiven = TRUE;
            break;

        /* Cross-term Dependence */
/* v3.0 */
        case  B3SOI_MOD_PAIGC :
            mod->B3SOIpaigc = value->rValue;
            mod->B3SOIpaigcGiven = TRUE;
            break;
        case  B3SOI_MOD_PBIGC :
            mod->B3SOIpbigc = value->rValue;
            mod->B3SOIpbigcGiven = TRUE;
            break;
        case  B3SOI_MOD_PCIGC :
            mod->B3SOIpcigc = value->rValue;
            mod->B3SOIpcigcGiven = TRUE;
            break;
        case  B3SOI_MOD_PAIGSD :
            mod->B3SOIpaigsd = value->rValue;
            mod->B3SOIpaigsdGiven = TRUE;
            break;
        case  B3SOI_MOD_PBIGSD :
            mod->B3SOIpbigsd = value->rValue;
            mod->B3SOIpbigsdGiven = TRUE;
            break;
        case  B3SOI_MOD_PCIGSD :
            mod->B3SOIpcigsd = value->rValue;
            mod->B3SOIpcigsdGiven = TRUE;
            break;
        case  B3SOI_MOD_PNIGC :
            mod->B3SOIpnigc = value->rValue;
            mod->B3SOIpnigcGiven = TRUE;
            break;
        case  B3SOI_MOD_PPIGCD :
            mod->B3SOIppigcd = value->rValue;
            mod->B3SOIppigcdGiven = TRUE;
            break;
        case  B3SOI_MOD_PPOXEDGE :
            mod->B3SOIppoxedge = value->rValue;
            mod->B3SOIppoxedgeGiven = TRUE;
            break;

        case B3SOI_MOD_PNPEAK:
            mod->B3SOIpnpeak = value->rValue;
            mod->B3SOIpnpeakGiven = TRUE;
            break;
        case B3SOI_MOD_PNSUB:
            mod->B3SOIpnsub = value->rValue;
            mod->B3SOIpnsubGiven = TRUE;
            break;
        case B3SOI_MOD_PNGATE:
            mod->B3SOIpngate = value->rValue;
            mod->B3SOIpngateGiven = TRUE;
            break;
        case B3SOI_MOD_PVTH0:
            mod->B3SOIpvth0 = value->rValue;
            mod->B3SOIpvth0Given = TRUE;
            break;
        case  B3SOI_MOD_PK1:
            mod->B3SOIpk1 = value->rValue;
            mod->B3SOIpk1Given = TRUE;
            break;
        case  B3SOI_MOD_PK1W1:
            mod->B3SOIpk1w1 = value->rValue;
            mod->B3SOIpk1w1Given = TRUE;
            break;
        case  B3SOI_MOD_PK1W2:
            mod->B3SOIpk1w2 = value->rValue;
            mod->B3SOIpk1w2Given = TRUE;
            break;
        case  B3SOI_MOD_PK2:
            mod->B3SOIpk2 = value->rValue;
            mod->B3SOIpk2Given = TRUE;
            break;
        case  B3SOI_MOD_PK3:
            mod->B3SOIpk3 = value->rValue;
            mod->B3SOIpk3Given = TRUE;
            break;
        case  B3SOI_MOD_PK3B:
            mod->B3SOIpk3b = value->rValue;
            mod->B3SOIpk3bGiven = TRUE;
            break;
        case  B3SOI_MOD_PKB1 :
            mod->B3SOIpkb1 = value->rValue;
            mod->B3SOIpkb1Given = TRUE;
            break;
        case  B3SOI_MOD_PW0:
            mod->B3SOIpw0 = value->rValue;
            mod->B3SOIpw0Given = TRUE;
            break;
        case  B3SOI_MOD_PNLX:
            mod->B3SOIpnlx = value->rValue;
            mod->B3SOIpnlxGiven = TRUE;
            break;
        case  B3SOI_MOD_PDVT0:               
            mod->B3SOIpdvt0 = value->rValue;
            mod->B3SOIpdvt0Given = TRUE;
            break;
        case  B3SOI_MOD_PDVT1:             
            mod->B3SOIpdvt1 = value->rValue;
            mod->B3SOIpdvt1Given = TRUE;
            break;
        case  B3SOI_MOD_PDVT2:             
            mod->B3SOIpdvt2 = value->rValue;
            mod->B3SOIpdvt2Given = TRUE;
            break;
        case  B3SOI_MOD_PDVT0W:               
            mod->B3SOIpdvt0w = value->rValue;
            mod->B3SOIpdvt0wGiven = TRUE;
            break;
        case  B3SOI_MOD_PDVT1W:             
            mod->B3SOIpdvt1w = value->rValue;
            mod->B3SOIpdvt1wGiven = TRUE;
            break;
        case  B3SOI_MOD_PDVT2W:             
            mod->B3SOIpdvt2w = value->rValue;
            mod->B3SOIpdvt2wGiven = TRUE;
            break;
        case  B3SOI_MOD_PU0 :
            mod->B3SOIpu0 = value->rValue;
            mod->B3SOIpu0Given = TRUE;
            break;
        case B3SOI_MOD_PUA:
            mod->B3SOIpua = value->rValue;
            mod->B3SOIpuaGiven = TRUE;
            break;
        case B3SOI_MOD_PUB:
            mod->B3SOIpub = value->rValue;
            mod->B3SOIpubGiven = TRUE;
            break;
        case B3SOI_MOD_PUC:
            mod->B3SOIpuc = value->rValue;
            mod->B3SOIpucGiven = TRUE;
            break;
        case B3SOI_MOD_PVSAT:
            mod->B3SOIpvsat = value->rValue;
            mod->B3SOIpvsatGiven = TRUE;
            break;
        case B3SOI_MOD_PA0:
            mod->B3SOIpa0 = value->rValue;
            mod->B3SOIpa0Given = TRUE;
            break;
        case B3SOI_MOD_PAGS:
            mod->B3SOIpags= value->rValue;
            mod->B3SOIpagsGiven = TRUE;
            break;
        case  B3SOI_MOD_PB0 :
            mod->B3SOIpb0 = value->rValue;
            mod->B3SOIpb0Given = TRUE;
            break;
        case  B3SOI_MOD_PB1 :
            mod->B3SOIpb1 = value->rValue;
            mod->B3SOIpb1Given = TRUE;
            break;
        case B3SOI_MOD_PKETA:
            mod->B3SOIpketa = value->rValue;
            mod->B3SOIpketaGiven = TRUE;
            break;    
        case B3SOI_MOD_PKETAS:
            mod->B3SOIpketas = value->rValue;
            mod->B3SOIpketasGiven = TRUE;
            break;    
        case B3SOI_MOD_PA1:
            mod->B3SOIpa1 = value->rValue;
            mod->B3SOIpa1Given = TRUE;
            break;
        case B3SOI_MOD_PA2:
            mod->B3SOIpa2 = value->rValue;
            mod->B3SOIpa2Given = TRUE;
            break;
        case B3SOI_MOD_PRDSW:
            mod->B3SOIprdsw = value->rValue;
            mod->B3SOIprdswGiven = TRUE;
            break;                     
        case B3SOI_MOD_PPRWB:
            mod->B3SOIpprwb = value->rValue;
            mod->B3SOIpprwbGiven = TRUE;
            break;                     
        case B3SOI_MOD_PPRWG:
            mod->B3SOIpprwg = value->rValue;
            mod->B3SOIpprwgGiven = TRUE;
            break;                     
        case  B3SOI_MOD_PWR :
            mod->B3SOIpwr = value->rValue;
            mod->B3SOIpwrGiven = TRUE;
            break;
        case  B3SOI_MOD_PNFACTOR :
            mod->B3SOIpnfactor = value->rValue;
            mod->B3SOIpnfactorGiven = TRUE;
            break;
        case  B3SOI_MOD_PDWG :
            mod->B3SOIpdwg = value->rValue;
            mod->B3SOIpdwgGiven = TRUE;
            break;
        case  B3SOI_MOD_PDWB :
            mod->B3SOIpdwb = value->rValue;
            mod->B3SOIpdwbGiven = TRUE;
            break;
        case B3SOI_MOD_PVOFF:
            mod->B3SOIpvoff = value->rValue;
            mod->B3SOIpvoffGiven = TRUE;
            break;
        case B3SOI_MOD_PETA0:
            mod->B3SOIpeta0 = value->rValue;
            mod->B3SOIpeta0Given = TRUE;
            break;                 
        case B3SOI_MOD_PETAB:
            mod->B3SOIpetab = value->rValue;
            mod->B3SOIpetabGiven = TRUE;
            break;                 
        case  B3SOI_MOD_PDSUB:             
            mod->B3SOIpdsub = value->rValue;
            mod->B3SOIpdsubGiven = TRUE;
            break;
        case  B3SOI_MOD_PCIT :
            mod->B3SOIpcit = value->rValue;
            mod->B3SOIpcitGiven = TRUE;
            break;
        case  B3SOI_MOD_PCDSC :
            mod->B3SOIpcdsc = value->rValue;
            mod->B3SOIpcdscGiven = TRUE;
            break;
        case  B3SOI_MOD_PCDSCB :
            mod->B3SOIpcdscb = value->rValue;
            mod->B3SOIpcdscbGiven = TRUE;
            break;
        case  B3SOI_MOD_PCDSCD :
            mod->B3SOIpcdscd = value->rValue;
            mod->B3SOIpcdscdGiven = TRUE;
            break;
        case B3SOI_MOD_PPCLM:
            mod->B3SOIppclm = value->rValue;
            mod->B3SOIppclmGiven = TRUE;
            break;                 
        case B3SOI_MOD_PPDIBL1:
            mod->B3SOIppdibl1 = value->rValue;
            mod->B3SOIppdibl1Given = TRUE;
            break;                 
        case B3SOI_MOD_PPDIBL2:
            mod->B3SOIppdibl2 = value->rValue;
            mod->B3SOIppdibl2Given = TRUE;
            break;                 
        case B3SOI_MOD_PPDIBLB:
            mod->B3SOIppdiblb = value->rValue;
            mod->B3SOIppdiblbGiven = TRUE;
            break;                 
        case  B3SOI_MOD_PDROUT:             
            mod->B3SOIpdrout = value->rValue;
            mod->B3SOIpdroutGiven = TRUE;
            break;
        case B3SOI_MOD_PPVAG:
            mod->B3SOIppvag = value->rValue;
            mod->B3SOIppvagGiven = TRUE;
            break;                 
        case  B3SOI_MOD_PDELTA :
            mod->B3SOIpdelta = value->rValue;
            mod->B3SOIpdeltaGiven = TRUE;
            break;
        case  B3SOI_MOD_PALPHA0 :
            mod->B3SOIpalpha0 = value->rValue;
            mod->B3SOIpalpha0Given = TRUE;
            break;
        case  B3SOI_MOD_PFBJTII :
            mod->B3SOIpfbjtii = value->rValue;
            mod->B3SOIpfbjtiiGiven = TRUE;
            break;
        case  B3SOI_MOD_PBETA0 :
            mod->B3SOIpbeta0 = value->rValue;
            mod->B3SOIpbeta0Given = TRUE;
            break;
        case  B3SOI_MOD_PBETA1 :
            mod->B3SOIpbeta1 = value->rValue;
            mod->B3SOIpbeta1Given = TRUE;
            break;
        case  B3SOI_MOD_PBETA2 :
            mod->B3SOIpbeta2 = value->rValue;
            mod->B3SOIpbeta2Given = TRUE;
            break;
        case  B3SOI_MOD_PVDSATII0 :
            mod->B3SOIpvdsatii0 = value->rValue;
            mod->B3SOIpvdsatii0Given = TRUE;
            break;
        case  B3SOI_MOD_PLII :
            mod->B3SOIplii = value->rValue;
            mod->B3SOIpliiGiven = TRUE;
            break;
        case  B3SOI_MOD_PESATII :
            mod->B3SOIpesatii = value->rValue;
            mod->B3SOIpesatiiGiven = TRUE;
            break;
        case  B3SOI_MOD_PSII0 :
            mod->B3SOIpsii0 = value->rValue;
            mod->B3SOIpsii0Given = TRUE;
            break;
        case  B3SOI_MOD_PSII1 :
            mod->B3SOIpsii1 = value->rValue;
            mod->B3SOIpsii1Given = TRUE;
            break;
        case  B3SOI_MOD_PSII2 :
            mod->B3SOIpsii2 = value->rValue;
            mod->B3SOIpsii2Given = TRUE;
            break;
        case  B3SOI_MOD_PSIID :
            mod->B3SOIpsiid = value->rValue;
            mod->B3SOIpsiidGiven = TRUE;
            break;
        case  B3SOI_MOD_PAGIDL :
            mod->B3SOIpagidl = value->rValue;
            mod->B3SOIpagidlGiven = TRUE;
            break;
        case  B3SOI_MOD_PBGIDL :
            mod->B3SOIpbgidl = value->rValue;
            mod->B3SOIpbgidlGiven = TRUE;
            break;
        case  B3SOI_MOD_PNGIDL :
            mod->B3SOIpngidl = value->rValue;
            mod->B3SOIpngidlGiven = TRUE;
            break;
        case  B3SOI_MOD_PNTUN :
            mod->B3SOIpntun = value->rValue;
            mod->B3SOIpntunGiven = TRUE;
            break;
        case  B3SOI_MOD_PNDIODE :
            mod->B3SOIpndiode = value->rValue;
            mod->B3SOIpndiodeGiven = TRUE;
            break;
        case  B3SOI_MOD_PNRECF0 :
            mod->B3SOIpnrecf0 = value->rValue;
            mod->B3SOIpnrecf0Given = TRUE;
            break;
        case  B3SOI_MOD_PNRECR0 :
            mod->B3SOIpnrecr0 = value->rValue;
            mod->B3SOIpnrecr0Given = TRUE;
            break;
        case  B3SOI_MOD_PISBJT :
            mod->B3SOIpisbjt = value->rValue;
            mod->B3SOIpisbjtGiven = TRUE;
            break;
        case  B3SOI_MOD_PISDIF :
            mod->B3SOIpisdif = value->rValue;
            mod->B3SOIpisdifGiven = TRUE;
            break;
        case  B3SOI_MOD_PISREC :
            mod->B3SOIpisrec = value->rValue;
            mod->B3SOIpisrecGiven = TRUE;
            break;
        case  B3SOI_MOD_PISTUN :
            mod->B3SOIpistun = value->rValue;
            mod->B3SOIpistunGiven = TRUE;
            break;
        case  B3SOI_MOD_PVREC0 :
            mod->B3SOIpvrec0 = value->rValue;
            mod->B3SOIpvrec0Given = TRUE;
            break;
        case  B3SOI_MOD_PVTUN0 :
            mod->B3SOIpvtun0 = value->rValue;
            mod->B3SOIpvtun0Given = TRUE;
            break;
        case  B3SOI_MOD_PNBJT :
            mod->B3SOIpnbjt = value->rValue;
            mod->B3SOIpnbjtGiven = TRUE;
            break;
        case  B3SOI_MOD_PLBJT0 :
            mod->B3SOIplbjt0 = value->rValue;
            mod->B3SOIplbjt0Given = TRUE;
            break;
        case  B3SOI_MOD_PVABJT :
            mod->B3SOIpvabjt = value->rValue;
            mod->B3SOIpvabjtGiven = TRUE;
            break;
        case  B3SOI_MOD_PAELY :
            mod->B3SOIpaely = value->rValue;
            mod->B3SOIpaelyGiven = TRUE;
            break;
        case  B3SOI_MOD_PAHLI :
            mod->B3SOIpahli = value->rValue;
            mod->B3SOIpahliGiven = TRUE;
            break;
	/* CV Model */
        case  B3SOI_MOD_PVSDFB :
            mod->B3SOIpvsdfb = value->rValue;
            mod->B3SOIpvsdfbGiven = TRUE;
            break;
        case  B3SOI_MOD_PVSDTH :
            mod->B3SOIpvsdth = value->rValue;
            mod->B3SOIpvsdthGiven = TRUE;
            break;
        case  B3SOI_MOD_PDELVT :
            mod->B3SOIpdelvt = value->rValue;
            mod->B3SOIpdelvtGiven = TRUE;
            break;
        case  B3SOI_MOD_PACDE :
            mod->B3SOIpacde = value->rValue;
            mod->B3SOIpacdeGiven = TRUE;
            break;
        case  B3SOI_MOD_PMOIN :
            mod->B3SOIpmoin = value->rValue;
            mod->B3SOIpmoinGiven = TRUE;
            break;
/* Added for binning - END */

        case  B3SOI_MOD_NMOS  :
            if(value->iValue) {
                mod->B3SOItype = 1;
                mod->B3SOItypeGiven = TRUE;
            }
            break;
        case  B3SOI_MOD_PMOS  :
            if(value->iValue) {
                mod->B3SOItype = - 1;
                mod->B3SOItypeGiven = TRUE;
            }
            break;
        default:
            return(E_BADPARM);
    }
    return(OK);
}


