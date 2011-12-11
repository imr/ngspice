/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1998 Samuel Fung, Dennis Sinitsky and Stephen Tang
File: b3soipdmpar.c          98/5/01
Modified by Pin Su and Jan Feng	99/2/15
Modified by Pin Su 99/4/30
Modified by Wei Jin 99/9/27
Modified by Pin Su 00/3/1
Modified by Pin Su 01/2/15
Modified by Pin Su 02/3/5
Modified by Paolo Nenzi 2002
**********/

/*
 * Revision 2.2.3  02/3/5  Pin Su 
 * BSIMPD2.2.3 release
 */

#include "ngspice/ngspice.h"
#include "b3soipddef.h"
#include "ngspice/ifsim.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
B3SOIPDmParam(int param, IFvalue *value, GENmodel *inMod)
{
    B3SOIPDmodel *mod = (B3SOIPDmodel*)inMod;
    switch(param)
    {  
 
	case  B3SOIPD_MOD_MOBMOD :
            mod->B3SOIPDmobMod = value->iValue;
            mod->B3SOIPDmobModGiven = TRUE;
            break;
        case  B3SOIPD_MOD_BINUNIT :
            mod->B3SOIPDbinUnit = value->iValue;
            mod->B3SOIPDbinUnitGiven = TRUE;
            break;
        case  B3SOIPD_MOD_PARAMCHK :
            mod->B3SOIPDparamChk = value->iValue;
            mod->B3SOIPDparamChkGiven = TRUE;
            break;
        case  B3SOIPD_MOD_CAPMOD :
            mod->B3SOIPDcapMod = value->iValue;
            mod->B3SOIPDcapModGiven = TRUE;
            break;
        case  B3SOIPD_MOD_SHMOD :
            mod->B3SOIPDshMod = value->iValue;
            mod->B3SOIPDshModGiven = TRUE;
            break;
        case  B3SOIPD_MOD_NOIMOD :
            mod->B3SOIPDnoiMod = value->iValue;
            mod->B3SOIPDnoiModGiven = TRUE;
            break;
        case  B3SOIPD_MOD_VERSION :
            mod->B3SOIPDversion = value->rValue;
            mod->B3SOIPDversionGiven = TRUE;
            break;
        case  B3SOIPD_MOD_TOX :
            mod->B3SOIPDtox = value->rValue;
            mod->B3SOIPDtoxGiven = TRUE;
            break;
/* v2.2.3 */
        case  B3SOIPD_MOD_DTOXCV :
            mod->B3SOIPDdtoxcv = value->rValue;
            mod->B3SOIPDdtoxcvGiven = TRUE;
            break;

        case  B3SOIPD_MOD_CDSC :
            mod->B3SOIPDcdsc = value->rValue;
            mod->B3SOIPDcdscGiven = TRUE;
            break;
        case  B3SOIPD_MOD_CDSCB :
            mod->B3SOIPDcdscb = value->rValue;
            mod->B3SOIPDcdscbGiven = TRUE;
            break;

        case  B3SOIPD_MOD_CDSCD :
            mod->B3SOIPDcdscd = value->rValue;
            mod->B3SOIPDcdscdGiven = TRUE;
            break;

        case  B3SOIPD_MOD_CIT :
            mod->B3SOIPDcit = value->rValue;
            mod->B3SOIPDcitGiven = TRUE;
            break;
        case  B3SOIPD_MOD_NFACTOR :
            mod->B3SOIPDnfactor = value->rValue;
            mod->B3SOIPDnfactorGiven = TRUE;
            break;
        case B3SOIPD_MOD_VSAT:
            mod->B3SOIPDvsat = value->rValue;
            mod->B3SOIPDvsatGiven = TRUE;
            break;
        case B3SOIPD_MOD_A0:
            mod->B3SOIPDa0 = value->rValue;
            mod->B3SOIPDa0Given = TRUE;
            break;
        
        case B3SOIPD_MOD_AGS:
            mod->B3SOIPDags= value->rValue;
            mod->B3SOIPDagsGiven = TRUE;
            break;
        
        case B3SOIPD_MOD_A1:
            mod->B3SOIPDa1 = value->rValue;
            mod->B3SOIPDa1Given = TRUE;
            break;
        case B3SOIPD_MOD_A2:
            mod->B3SOIPDa2 = value->rValue;
            mod->B3SOIPDa2Given = TRUE;
            break;
        case B3SOIPD_MOD_AT:
            mod->B3SOIPDat = value->rValue;
            mod->B3SOIPDatGiven = TRUE;
            break;
        case B3SOIPD_MOD_KETA:
            mod->B3SOIPDketa = value->rValue;
            mod->B3SOIPDketaGiven = TRUE;
            break;    
        case B3SOIPD_MOD_NSUB:
            mod->B3SOIPDnsub = value->rValue;
            mod->B3SOIPDnsubGiven = TRUE;
            break;
        case B3SOIPD_MOD_NPEAK:
            mod->B3SOIPDnpeak = value->rValue;
            mod->B3SOIPDnpeakGiven = TRUE;
	    if (mod->B3SOIPDnpeak > 1.0e20)
		mod->B3SOIPDnpeak *= 1.0e-6;
            break;
        case B3SOIPD_MOD_NGATE:
            mod->B3SOIPDngate = value->rValue;
            mod->B3SOIPDngateGiven = TRUE;
	    if (mod->B3SOIPDngate > 1.0e23)
		mod->B3SOIPDngate *= 1.0e-6;
            break;
        case B3SOIPD_MOD_GAMMA1:
            mod->B3SOIPDgamma1 = value->rValue;
            mod->B3SOIPDgamma1Given = TRUE;
            break;
        case B3SOIPD_MOD_GAMMA2:
            mod->B3SOIPDgamma2 = value->rValue;
            mod->B3SOIPDgamma2Given = TRUE;
            break;
        case B3SOIPD_MOD_VBX:
            mod->B3SOIPDvbx = value->rValue;
            mod->B3SOIPDvbxGiven = TRUE;
            break;
        case B3SOIPD_MOD_VBM:
            mod->B3SOIPDvbm = value->rValue;
            mod->B3SOIPDvbmGiven = TRUE;
            break;
        case B3SOIPD_MOD_XT:
            mod->B3SOIPDxt = value->rValue;
            mod->B3SOIPDxtGiven = TRUE;
            break;
        case  B3SOIPD_MOD_K1:
            mod->B3SOIPDk1 = value->rValue;
            mod->B3SOIPDk1Given = TRUE;
            break;
        case  B3SOIPD_MOD_KT1:
            mod->B3SOIPDkt1 = value->rValue;
            mod->B3SOIPDkt1Given = TRUE;
            break;
        case  B3SOIPD_MOD_KT1L:
            mod->B3SOIPDkt1l = value->rValue;
            mod->B3SOIPDkt1lGiven = TRUE;
            break;
        case  B3SOIPD_MOD_KT2:
            mod->B3SOIPDkt2 = value->rValue;
            mod->B3SOIPDkt2Given = TRUE;
            break;
        case  B3SOIPD_MOD_K2:
            mod->B3SOIPDk2 = value->rValue;
            mod->B3SOIPDk2Given = TRUE;
            break;
        case  B3SOIPD_MOD_K3:
            mod->B3SOIPDk3 = value->rValue;
            mod->B3SOIPDk3Given = TRUE;
            break;
        case  B3SOIPD_MOD_K3B:
            mod->B3SOIPDk3b = value->rValue;
            mod->B3SOIPDk3bGiven = TRUE;
            break;
        case  B3SOIPD_MOD_NLX:
            mod->B3SOIPDnlx = value->rValue;
            mod->B3SOIPDnlxGiven = TRUE;
            break;
        case  B3SOIPD_MOD_W0:
            mod->B3SOIPDw0 = value->rValue;
            mod->B3SOIPDw0Given = TRUE;
            break;
        case  B3SOIPD_MOD_DVT0:               
            mod->B3SOIPDdvt0 = value->rValue;
            mod->B3SOIPDdvt0Given = TRUE;
            break;
        case  B3SOIPD_MOD_DVT1:             
            mod->B3SOIPDdvt1 = value->rValue;
            mod->B3SOIPDdvt1Given = TRUE;
            break;
        case  B3SOIPD_MOD_DVT2:             
            mod->B3SOIPDdvt2 = value->rValue;
            mod->B3SOIPDdvt2Given = TRUE;
            break;
        case  B3SOIPD_MOD_DVT0W:               
            mod->B3SOIPDdvt0w = value->rValue;
            mod->B3SOIPDdvt0wGiven = TRUE;
            break;
        case  B3SOIPD_MOD_DVT1W:             
            mod->B3SOIPDdvt1w = value->rValue;
            mod->B3SOIPDdvt1wGiven = TRUE;
            break;
        case  B3SOIPD_MOD_DVT2W:             
            mod->B3SOIPDdvt2w = value->rValue;
            mod->B3SOIPDdvt2wGiven = TRUE;
            break;
        case  B3SOIPD_MOD_DROUT:             
            mod->B3SOIPDdrout = value->rValue;
            mod->B3SOIPDdroutGiven = TRUE;
            break;
        case  B3SOIPD_MOD_DSUB:             
            mod->B3SOIPDdsub = value->rValue;
            mod->B3SOIPDdsubGiven = TRUE;
            break;
        case B3SOIPD_MOD_VTH0:
            mod->B3SOIPDvth0 = value->rValue;
            mod->B3SOIPDvth0Given = TRUE;
            break;
        case B3SOIPD_MOD_UA:
            mod->B3SOIPDua = value->rValue;
            mod->B3SOIPDuaGiven = TRUE;
            break;
        case B3SOIPD_MOD_UA1:
            mod->B3SOIPDua1 = value->rValue;
            mod->B3SOIPDua1Given = TRUE;
            break;
        case B3SOIPD_MOD_UB:
            mod->B3SOIPDub = value->rValue;
            mod->B3SOIPDubGiven = TRUE;
            break;
        case B3SOIPD_MOD_UB1:
            mod->B3SOIPDub1 = value->rValue;
            mod->B3SOIPDub1Given = TRUE;
            break;
        case B3SOIPD_MOD_UC:
            mod->B3SOIPDuc = value->rValue;
            mod->B3SOIPDucGiven = TRUE;
            break;
        case B3SOIPD_MOD_UC1:
            mod->B3SOIPDuc1 = value->rValue;
            mod->B3SOIPDuc1Given = TRUE;
            break;
        case  B3SOIPD_MOD_U0 :
            mod->B3SOIPDu0 = value->rValue;
            mod->B3SOIPDu0Given = TRUE;
            break;
        case  B3SOIPD_MOD_UTE :
            mod->B3SOIPDute = value->rValue;
            mod->B3SOIPDuteGiven = TRUE;
            break;
        case B3SOIPD_MOD_VOFF:
            mod->B3SOIPDvoff = value->rValue;
            mod->B3SOIPDvoffGiven = TRUE;
            break;
        case  B3SOIPD_MOD_DELTA :
            mod->B3SOIPDdelta = value->rValue;
            mod->B3SOIPDdeltaGiven = TRUE;
            break;
        case B3SOIPD_MOD_RDSW:
            mod->B3SOIPDrdsw = value->rValue;
            mod->B3SOIPDrdswGiven = TRUE;
            break;                     
        case B3SOIPD_MOD_PRWG:
            mod->B3SOIPDprwg = value->rValue;
            mod->B3SOIPDprwgGiven = TRUE;
            break;                     
        case B3SOIPD_MOD_PRWB:
            mod->B3SOIPDprwb = value->rValue;
            mod->B3SOIPDprwbGiven = TRUE;
            break;                     
        case B3SOIPD_MOD_PRT:
            mod->B3SOIPDprt = value->rValue;
            mod->B3SOIPDprtGiven = TRUE;
            break;                     
        case B3SOIPD_MOD_ETA0:
            mod->B3SOIPDeta0 = value->rValue;
            mod->B3SOIPDeta0Given = TRUE;
            break;                 
        case B3SOIPD_MOD_ETAB:
            mod->B3SOIPDetab = value->rValue;
            mod->B3SOIPDetabGiven = TRUE;
            break;                 
        case B3SOIPD_MOD_PCLM:
            mod->B3SOIPDpclm = value->rValue;
            mod->B3SOIPDpclmGiven = TRUE;
            break;                 
        case B3SOIPD_MOD_PDIBL1:
            mod->B3SOIPDpdibl1 = value->rValue;
            mod->B3SOIPDpdibl1Given = TRUE;
            break;                 
        case B3SOIPD_MOD_PDIBL2:
            mod->B3SOIPDpdibl2 = value->rValue;
            mod->B3SOIPDpdibl2Given = TRUE;
            break;                 
        case B3SOIPD_MOD_PDIBLB:
            mod->B3SOIPDpdiblb = value->rValue;
            mod->B3SOIPDpdiblbGiven = TRUE;
            break;                 
        case B3SOIPD_MOD_PVAG:
            mod->B3SOIPDpvag = value->rValue;
            mod->B3SOIPDpvagGiven = TRUE;
            break;                 
        case  B3SOIPD_MOD_WR :
            mod->B3SOIPDwr = value->rValue;
            mod->B3SOIPDwrGiven = TRUE;
            break;
        case  B3SOIPD_MOD_DWG :
            mod->B3SOIPDdwg = value->rValue;
            mod->B3SOIPDdwgGiven = TRUE;
            break;
        case  B3SOIPD_MOD_DWB :
            mod->B3SOIPDdwb = value->rValue;
            mod->B3SOIPDdwbGiven = TRUE;
            break;
        case  B3SOIPD_MOD_B0 :
            mod->B3SOIPDb0 = value->rValue;
            mod->B3SOIPDb0Given = TRUE;
            break;
        case  B3SOIPD_MOD_B1 :
            mod->B3SOIPDb1 = value->rValue;
            mod->B3SOIPDb1Given = TRUE;
            break;
        case  B3SOIPD_MOD_ALPHA0 :
            mod->B3SOIPDalpha0 = value->rValue;
            mod->B3SOIPDalpha0Given = TRUE;
            break;

        case  B3SOIPD_MOD_CGSL :
            mod->B3SOIPDcgsl = value->rValue;
            mod->B3SOIPDcgslGiven = TRUE;
            break;
        case  B3SOIPD_MOD_CGDL :
            mod->B3SOIPDcgdl = value->rValue;
            mod->B3SOIPDcgdlGiven = TRUE;
            break;
        case  B3SOIPD_MOD_CKAPPA :
            mod->B3SOIPDckappa = value->rValue;
            mod->B3SOIPDckappaGiven = TRUE;
            break;
        case  B3SOIPD_MOD_CF :
            mod->B3SOIPDcf = value->rValue;
            mod->B3SOIPDcfGiven = TRUE;
            break;
        case  B3SOIPD_MOD_CLC :
            mod->B3SOIPDclc = value->rValue;
            mod->B3SOIPDclcGiven = TRUE;
            break;
        case  B3SOIPD_MOD_CLE :
            mod->B3SOIPDcle = value->rValue;
            mod->B3SOIPDcleGiven = TRUE;
            break;
        case  B3SOIPD_MOD_DWC :
            mod->B3SOIPDdwc = value->rValue;
            mod->B3SOIPDdwcGiven = TRUE;
            break;
        case  B3SOIPD_MOD_DLC :
            mod->B3SOIPDdlc = value->rValue;
            mod->B3SOIPDdlcGiven = TRUE;
            break;
        case  B3SOIPD_MOD_TBOX :
            mod->B3SOIPDtbox = value->rValue;
            mod->B3SOIPDtboxGiven = TRUE;
            break;
        case  B3SOIPD_MOD_TSI :
            mod->B3SOIPDtsi = value->rValue;
            mod->B3SOIPDtsiGiven = TRUE;
            break;
        case  B3SOIPD_MOD_XJ :
            mod->B3SOIPDxj = value->rValue;
            mod->B3SOIPDxjGiven = TRUE;
            break;
        case  B3SOIPD_MOD_RBODY :
            mod->B3SOIPDrbody = value->rValue;
            mod->B3SOIPDrbodyGiven = TRUE;
            break;
        case  B3SOIPD_MOD_RBSH :
            mod->B3SOIPDrbsh = value->rValue;
            mod->B3SOIPDrbshGiven = TRUE;
            break;
        case  B3SOIPD_MOD_RTH0 :
            mod->B3SOIPDrth0 = value->rValue;
            mod->B3SOIPDrth0Given = TRUE;
            break;
        case  B3SOIPD_MOD_CTH0 :
            mod->B3SOIPDcth0 = value->rValue;
            mod->B3SOIPDcth0Given = TRUE;
            break;
        case  B3SOIPD_MOD_NGIDL :
            mod->B3SOIPDngidl = value->rValue;
            mod->B3SOIPDngidlGiven = TRUE;
            break;
        case  B3SOIPD_MOD_AGIDL :
            mod->B3SOIPDagidl = value->rValue;
            mod->B3SOIPDagidlGiven = TRUE;
            break;
        case  B3SOIPD_MOD_BGIDL :
            mod->B3SOIPDbgidl = value->rValue;
            mod->B3SOIPDbgidlGiven = TRUE;
            break;
        case  B3SOIPD_MOD_NDIODE :
            mod->B3SOIPDndiode = value->rValue;
            mod->B3SOIPDndiodeGiven = TRUE;
            break;
        case  B3SOIPD_MOD_XBJT :
            mod->B3SOIPDxbjt = value->rValue;
            mod->B3SOIPDxbjtGiven = TRUE;
            break;

        case  B3SOIPD_MOD_XDIF :
            mod->B3SOIPDxdif = value->rValue;
            mod->B3SOIPDxdifGiven = TRUE;
            break;

        case  B3SOIPD_MOD_XREC :
            mod->B3SOIPDxrec = value->rValue;
            mod->B3SOIPDxrecGiven = TRUE;
            break;
        case  B3SOIPD_MOD_XTUN :
            mod->B3SOIPDxtun = value->rValue;
            mod->B3SOIPDxtunGiven = TRUE;
            break;
        case  B3SOIPD_MOD_TT :
            mod->B3SOIPDtt = value->rValue;
            mod->B3SOIPDttGiven = TRUE;
            break;
        case  B3SOIPD_MOD_VSDTH :
            mod->B3SOIPDvsdth = value->rValue;
            mod->B3SOIPDvsdthGiven = TRUE;
            break;
        case  B3SOIPD_MOD_VSDFB :
            mod->B3SOIPDvsdfb = value->rValue;
            mod->B3SOIPDvsdfbGiven = TRUE;
            break;
        case  B3SOIPD_MOD_CSDMIN :
            mod->B3SOIPDcsdmin = value->rValue;
            mod->B3SOIPDcsdminGiven = TRUE;
            break;
        case  B3SOIPD_MOD_ASD :
            mod->B3SOIPDasd = value->rValue;
            mod->B3SOIPDasdGiven = TRUE;
            break;


        case  B3SOIPD_MOD_TNOM :
            mod->B3SOIPDtnom = value->rValue + 273.15;
            mod->B3SOIPDtnomGiven = TRUE;
            break;
        case  B3SOIPD_MOD_CGSO :
            mod->B3SOIPDcgso = value->rValue;
            mod->B3SOIPDcgsoGiven = TRUE;
            break;
        case  B3SOIPD_MOD_CGDO :
            mod->B3SOIPDcgdo = value->rValue;
            mod->B3SOIPDcgdoGiven = TRUE;
            break;
        case  B3SOIPD_MOD_CGEO :
            mod->B3SOIPDcgeo = value->rValue;
            mod->B3SOIPDcgeoGiven = TRUE;
            break;
        case  B3SOIPD_MOD_XPART :
            mod->B3SOIPDxpart = value->rValue;
            mod->B3SOIPDxpartGiven = TRUE;
            break;
        case  B3SOIPD_MOD_RSH :
            mod->B3SOIPDsheetResistance = value->rValue;
            mod->B3SOIPDsheetResistanceGiven = TRUE;
            break;
        case  B3SOIPD_MOD_PBSWG :
            mod->B3SOIPDGatesidewallJctPotential = value->rValue;
            mod->B3SOIPDGatesidewallJctPotentialGiven = TRUE;
            break;
        case  B3SOIPD_MOD_MJSWG :
            mod->B3SOIPDbodyJctGateSideGradingCoeff = value->rValue;
            mod->B3SOIPDbodyJctGateSideGradingCoeffGiven = TRUE;
            break;
        case  B3SOIPD_MOD_CJSWG :
            mod->B3SOIPDunitLengthGateSidewallJctCap = value->rValue;
            mod->B3SOIPDunitLengthGateSidewallJctCapGiven = TRUE;
            break;
        case  B3SOIPD_MOD_CSDESW :
            mod->B3SOIPDcsdesw = value->rValue;
            mod->B3SOIPDcsdeswGiven = TRUE;
            break;
        case  B3SOIPD_MOD_LINT :
            mod->B3SOIPDLint = value->rValue;
            mod->B3SOIPDLintGiven = TRUE;
            break;
        case  B3SOIPD_MOD_LL :
            mod->B3SOIPDLl = value->rValue;
            mod->B3SOIPDLlGiven = TRUE;
            break;
/* v2.2.3 */
        case  B3SOIPD_MOD_LLC :
            mod->B3SOIPDLlc = value->rValue;
            mod->B3SOIPDLlcGiven = TRUE;
            break;

        case  B3SOIPD_MOD_LLN :
            mod->B3SOIPDLln = value->rValue;
            mod->B3SOIPDLlnGiven = TRUE;
            break;
        case  B3SOIPD_MOD_LW :
            mod->B3SOIPDLw = value->rValue;
            mod->B3SOIPDLwGiven = TRUE;
            break;
/* v2.2.3 */
        case  B3SOIPD_MOD_LWC :
            mod->B3SOIPDLwc = value->rValue;
            mod->B3SOIPDLwcGiven = TRUE;
            break;

        case  B3SOIPD_MOD_LWN :
            mod->B3SOIPDLwn = value->rValue;
            mod->B3SOIPDLwnGiven = TRUE;
            break;
        case  B3SOIPD_MOD_LWL :
            mod->B3SOIPDLwl = value->rValue;
            mod->B3SOIPDLwlGiven = TRUE;
            break;
/* v2.2.3 */
        case  B3SOIPD_MOD_LWLC :
            mod->B3SOIPDLwlc = value->rValue;
            mod->B3SOIPDLwlcGiven = TRUE;
            break;

        case  B3SOIPD_MOD_WINT :
            mod->B3SOIPDWint = value->rValue;
            mod->B3SOIPDWintGiven = TRUE;
            break;
        case  B3SOIPD_MOD_WL :
            mod->B3SOIPDWl = value->rValue;
            mod->B3SOIPDWlGiven = TRUE;
            break;
/* v2.2.3 */
        case  B3SOIPD_MOD_WLC :
            mod->B3SOIPDWlc = value->rValue;
            mod->B3SOIPDWlcGiven = TRUE;
            break;

        case  B3SOIPD_MOD_WLN :
            mod->B3SOIPDWln = value->rValue;
            mod->B3SOIPDWlnGiven = TRUE;
            break;
        case  B3SOIPD_MOD_WW :
            mod->B3SOIPDWw = value->rValue;
            mod->B3SOIPDWwGiven = TRUE;
            break;
/* v2.2.3 */
        case  B3SOIPD_MOD_WWC :
            mod->B3SOIPDWwc = value->rValue;
            mod->B3SOIPDWwcGiven = TRUE;
            break;

        case  B3SOIPD_MOD_WWN :
            mod->B3SOIPDWwn = value->rValue;
            mod->B3SOIPDWwnGiven = TRUE;
            break;
        case  B3SOIPD_MOD_WWL :
            mod->B3SOIPDWwl = value->rValue;
            mod->B3SOIPDWwlGiven = TRUE;
            break;
/* v2.2.3 */
        case  B3SOIPD_MOD_WWLC :
            mod->B3SOIPDWwlc = value->rValue;
            mod->B3SOIPDWwlcGiven = TRUE;
            break;

        case  B3SOIPD_MOD_NOIA :
            mod->B3SOIPDoxideTrapDensityA = value->rValue;
            mod->B3SOIPDoxideTrapDensityAGiven = TRUE;
            break;
        case  B3SOIPD_MOD_NOIB :
            mod->B3SOIPDoxideTrapDensityB = value->rValue;
            mod->B3SOIPDoxideTrapDensityBGiven = TRUE;
            break;
        case  B3SOIPD_MOD_NOIC :
            mod->B3SOIPDoxideTrapDensityC = value->rValue;
            mod->B3SOIPDoxideTrapDensityCGiven = TRUE;
            break;
        case  B3SOIPD_MOD_NOIF :
            mod->B3SOIPDnoif = value->rValue;
            mod->B3SOIPDnoifGiven = TRUE;
            break;
        case  B3SOIPD_MOD_EM :
            mod->B3SOIPDem = value->rValue;
            mod->B3SOIPDemGiven = TRUE;
            break;
        case  B3SOIPD_MOD_EF :
            mod->B3SOIPDef = value->rValue;
            mod->B3SOIPDefGiven = TRUE;
            break;
        case  B3SOIPD_MOD_AF :
            mod->B3SOIPDaf = value->rValue;
            mod->B3SOIPDafGiven = TRUE;
            break;
        case  B3SOIPD_MOD_KF :
            mod->B3SOIPDkf = value->rValue;
            mod->B3SOIPDkfGiven = TRUE;
            break;


/* v2.2 release */
        case  B3SOIPD_MOD_WTH0 :
            mod->B3SOIPDwth0 = value->rValue;
            mod->B3SOIPDwth0Given = TRUE;
            break;
        case  B3SOIPD_MOD_RHALO :
            mod->B3SOIPDrhalo = value->rValue;
            mod->B3SOIPDrhaloGiven = TRUE;
            break;
        case  B3SOIPD_MOD_NTOX :
            mod->B3SOIPDntox = value->rValue;
            mod->B3SOIPDntoxGiven = TRUE;
            break;
        case  B3SOIPD_MOD_TOXREF :
            mod->B3SOIPDtoxref = value->rValue;
            mod->B3SOIPDtoxrefGiven = TRUE;
            break;
        case  B3SOIPD_MOD_EBG :
            mod->B3SOIPDebg = value->rValue;
            mod->B3SOIPDebgGiven = TRUE;
            break;
        case  B3SOIPD_MOD_VEVB :
            mod->B3SOIPDvevb = value->rValue;
            mod->B3SOIPDvevbGiven = TRUE;
            break;
        case  B3SOIPD_MOD_ALPHAGB1 :
            mod->B3SOIPDalphaGB1 = value->rValue;
            mod->B3SOIPDalphaGB1Given = TRUE;
            break;
        case  B3SOIPD_MOD_BETAGB1 :
            mod->B3SOIPDbetaGB1 = value->rValue;
            mod->B3SOIPDbetaGB1Given = TRUE;
            break;
        case  B3SOIPD_MOD_VGB1 :
            mod->B3SOIPDvgb1 = value->rValue;
            mod->B3SOIPDvgb1Given = TRUE;
            break;
        case  B3SOIPD_MOD_VECB :
            mod->B3SOIPDvecb = value->rValue;
            mod->B3SOIPDvecbGiven = TRUE;
            break;
        case  B3SOIPD_MOD_ALPHAGB2 :
            mod->B3SOIPDalphaGB2 = value->rValue;
            mod->B3SOIPDalphaGB2Given = TRUE;
            break;
        case  B3SOIPD_MOD_BETAGB2 :
            mod->B3SOIPDbetaGB2 = value->rValue;
            mod->B3SOIPDbetaGB2Given = TRUE;
            break;
        case  B3SOIPD_MOD_VGB2 :
            mod->B3SOIPDvgb2 = value->rValue;
            mod->B3SOIPDvgb2Given = TRUE;
            break;
        case  B3SOIPD_MOD_TOXQM :
            mod->B3SOIPDtoxqm = value->rValue;
            mod->B3SOIPDtoxqmGiven = TRUE;
            break;
        case  B3SOIPD_MOD_VOXH :
            mod->B3SOIPDvoxh = value->rValue;
            mod->B3SOIPDvoxhGiven = TRUE;
            break;
        case  B3SOIPD_MOD_DELTAVOX :
            mod->B3SOIPDdeltavox = value->rValue;
            mod->B3SOIPDdeltavoxGiven = TRUE;
            break;
        case  B3SOIPD_MOD_IGMOD :
            mod->B3SOIPDigMod = value->iValue;
            mod->B3SOIPDigModGiven = TRUE;
            break;


/* v2.0 release */
        case  B3SOIPD_MOD_K1W1 :         
            mod->B3SOIPDk1w1 = value->rValue;
            mod->B3SOIPDk1w1Given = TRUE;
            break;
        case  B3SOIPD_MOD_K1W2 :
            mod->B3SOIPDk1w2 = value->rValue;
            mod->B3SOIPDk1w2Given = TRUE;
            break;
        case  B3SOIPD_MOD_KETAS :
            mod->B3SOIPDketas = value->rValue;
            mod->B3SOIPDketasGiven = TRUE;
            break;
        case  B3SOIPD_MOD_DWBC :
            mod->B3SOIPDdwbc = value->rValue;
            mod->B3SOIPDdwbcGiven = TRUE;
            break;
        case  B3SOIPD_MOD_BETA0 :
            mod->B3SOIPDbeta0 = value->rValue;
            mod->B3SOIPDbeta0Given = TRUE;
            break;
        case  B3SOIPD_MOD_BETA1 :
            mod->B3SOIPDbeta1 = value->rValue;
            mod->B3SOIPDbeta1Given = TRUE;
            break;
        case  B3SOIPD_MOD_BETA2 :
            mod->B3SOIPDbeta2 = value->rValue;
            mod->B3SOIPDbeta2Given = TRUE;
            break;
        case  B3SOIPD_MOD_VDSATII0 :
            mod->B3SOIPDvdsatii0 = value->rValue;
            mod->B3SOIPDvdsatii0Given = TRUE;
            break;
        case  B3SOIPD_MOD_TII :
            mod->B3SOIPDtii = value->rValue;
            mod->B3SOIPDtiiGiven = TRUE;
            break;
        case  B3SOIPD_MOD_LII :
            mod->B3SOIPDlii = value->rValue;
            mod->B3SOIPDliiGiven = TRUE;
            break;
        case  B3SOIPD_MOD_SII0 :
            mod->B3SOIPDsii0 = value->rValue;
            mod->B3SOIPDsii0Given = TRUE;
            break;
        case  B3SOIPD_MOD_SII1 :
            mod->B3SOIPDsii1 = value->rValue;
            mod->B3SOIPDsii1Given = TRUE;
            break;
        case  B3SOIPD_MOD_SII2 :
            mod->B3SOIPDsii2 = value->rValue;
            mod->B3SOIPDsii2Given = TRUE;
            break;
        case  B3SOIPD_MOD_SIID :
            mod->B3SOIPDsiid = value->rValue;
            mod->B3SOIPDsiidGiven = TRUE;
            break;
        case  B3SOIPD_MOD_FBJTII :
            mod->B3SOIPDfbjtii = value->rValue;
            mod->B3SOIPDfbjtiiGiven = TRUE;
            break;
        case  B3SOIPD_MOD_ESATII :
            mod->B3SOIPDesatii = value->rValue;
            mod->B3SOIPDesatiiGiven = TRUE;
            break;
        case  B3SOIPD_MOD_NTUN :
            mod->B3SOIPDntun = value->rValue;
            mod->B3SOIPDntunGiven = TRUE;
            break;
        case  B3SOIPD_MOD_NRECF0 :
            mod->B3SOIPDnrecf0 = value->rValue;
            mod->B3SOIPDnrecf0Given = TRUE;
            break;
        case  B3SOIPD_MOD_NRECR0 :
            mod->B3SOIPDnrecr0 = value->rValue;
            mod->B3SOIPDnrecr0Given = TRUE;
            break;
        case  B3SOIPD_MOD_ISBJT :
            mod->B3SOIPDisbjt = value->rValue;
            mod->B3SOIPDisbjtGiven = TRUE;
            break;
        case  B3SOIPD_MOD_ISDIF :
            mod->B3SOIPDisdif = value->rValue;
            mod->B3SOIPDisdifGiven = TRUE;
            break;
        case  B3SOIPD_MOD_ISREC :
            mod->B3SOIPDisrec = value->rValue;
            mod->B3SOIPDisrecGiven = TRUE;
            break;
        case  B3SOIPD_MOD_ISTUN :
            mod->B3SOIPDistun = value->rValue;
            mod->B3SOIPDistunGiven = TRUE;
            break;
        case  B3SOIPD_MOD_LN :
            mod->B3SOIPDln = value->rValue;
            mod->B3SOIPDlnGiven = TRUE;
            break;
        case  B3SOIPD_MOD_VREC0 :
            mod->B3SOIPDvrec0 = value->rValue;
            mod->B3SOIPDvrec0Given = TRUE;
            break;
        case  B3SOIPD_MOD_VTUN0 :
            mod->B3SOIPDvtun0 = value->rValue;
            mod->B3SOIPDvtun0Given = TRUE;
            break;
        case  B3SOIPD_MOD_NBJT :
            mod->B3SOIPDnbjt = value->rValue;
            mod->B3SOIPDnbjtGiven = TRUE;
            break;
        case  B3SOIPD_MOD_LBJT0 :
            mod->B3SOIPDlbjt0 = value->rValue;
            mod->B3SOIPDlbjt0Given = TRUE;
            break;
        case  B3SOIPD_MOD_LDIF0 :
            mod->B3SOIPDldif0 = value->rValue;
            mod->B3SOIPDldif0Given = TRUE;
            break;
        case  B3SOIPD_MOD_VABJT :
            mod->B3SOIPDvabjt = value->rValue;
            mod->B3SOIPDvabjtGiven = TRUE;
            break;
        case  B3SOIPD_MOD_AELY :
            mod->B3SOIPDaely = value->rValue;
            mod->B3SOIPDaelyGiven = TRUE;
            break;
        case  B3SOIPD_MOD_AHLI :
            mod->B3SOIPDahli = value->rValue;
            mod->B3SOIPDahliGiven = TRUE;
            break;
        case  B3SOIPD_MOD_NDIF :
            mod->B3SOIPDndif = value->rValue;
            mod->B3SOIPDndifGiven = TRUE;
            break;
        case  B3SOIPD_MOD_NTRECF :
            mod->B3SOIPDntrecf = value->rValue;
            mod->B3SOIPDntrecfGiven = TRUE;
            break;
        case  B3SOIPD_MOD_NTRECR :
            mod->B3SOIPDntrecr = value->rValue;
            mod->B3SOIPDntrecrGiven = TRUE;
            break;
        case  B3SOIPD_MOD_DLCB :
            mod->B3SOIPDdlcb = value->rValue;
            mod->B3SOIPDdlcbGiven = TRUE;
            break;
        case  B3SOIPD_MOD_FBODY :
            mod->B3SOIPDfbody = value->rValue;
            mod->B3SOIPDfbodyGiven = TRUE;
            break;
        case  B3SOIPD_MOD_TCJSWG :
            mod->B3SOIPDtcjswg = value->rValue;
            mod->B3SOIPDtcjswgGiven = TRUE;
            break;
        case  B3SOIPD_MOD_TPBSWG :
            mod->B3SOIPDtpbswg = value->rValue;
            mod->B3SOIPDtpbswgGiven = TRUE;
            break;
        case  B3SOIPD_MOD_ACDE :
            mod->B3SOIPDacde = value->rValue;
            mod->B3SOIPDacdeGiven = TRUE;
            break;
        case  B3SOIPD_MOD_MOIN :
            mod->B3SOIPDmoin = value->rValue;
            mod->B3SOIPDmoinGiven = TRUE;
            break;
        case  B3SOIPD_MOD_DELVT :
            mod->B3SOIPDdelvt = value->rValue;
            mod->B3SOIPDdelvtGiven = TRUE;
            break;
        case  B3SOIPD_MOD_KB1 :
            mod->B3SOIPDkb1 = value->rValue;
            mod->B3SOIPDkb1Given = TRUE;
            break;
        case  B3SOIPD_MOD_DLBG :
            mod->B3SOIPDdlbg = value->rValue;
            mod->B3SOIPDdlbgGiven = TRUE;
            break;

/* Added for binning - START */
        /* Length Dependence */
        case B3SOIPD_MOD_LNPEAK:
            mod->B3SOIPDlnpeak = value->rValue;
            mod->B3SOIPDlnpeakGiven = TRUE;
            break;
        case B3SOIPD_MOD_LNSUB:
            mod->B3SOIPDlnsub = value->rValue;
            mod->B3SOIPDlnsubGiven = TRUE;
            break;
        case B3SOIPD_MOD_LNGATE:
            mod->B3SOIPDlngate = value->rValue;
            mod->B3SOIPDlngateGiven = TRUE;
            break;
        case B3SOIPD_MOD_LVTH0:
            mod->B3SOIPDlvth0 = value->rValue;
            mod->B3SOIPDlvth0Given = TRUE;
            break;
        case  B3SOIPD_MOD_LK1:
            mod->B3SOIPDlk1 = value->rValue;
            mod->B3SOIPDlk1Given = TRUE;
            break;
        case  B3SOIPD_MOD_LK1W1:
            mod->B3SOIPDlk1w1 = value->rValue;
            mod->B3SOIPDlk1w1Given = TRUE;
            break;
        case  B3SOIPD_MOD_LK1W2:
            mod->B3SOIPDlk1w2 = value->rValue;
            mod->B3SOIPDlk1w2Given = TRUE;
            break;
        case  B3SOIPD_MOD_LK2:
            mod->B3SOIPDlk2 = value->rValue;
            mod->B3SOIPDlk2Given = TRUE;
            break;
        case  B3SOIPD_MOD_LK3:
            mod->B3SOIPDlk3 = value->rValue;
            mod->B3SOIPDlk3Given = TRUE;
            break;
        case  B3SOIPD_MOD_LK3B:
            mod->B3SOIPDlk3b = value->rValue;
            mod->B3SOIPDlk3bGiven = TRUE;
            break;
        case  B3SOIPD_MOD_LKB1 :
            mod->B3SOIPDlkb1 = value->rValue;
            mod->B3SOIPDlkb1Given = TRUE;
            break;
        case  B3SOIPD_MOD_LW0:
            mod->B3SOIPDlw0 = value->rValue;
            mod->B3SOIPDlw0Given = TRUE;
            break;
        case  B3SOIPD_MOD_LNLX:
            mod->B3SOIPDlnlx = value->rValue;
            mod->B3SOIPDlnlxGiven = TRUE;
            break;
        case  B3SOIPD_MOD_LDVT0:               
            mod->B3SOIPDldvt0 = value->rValue;
            mod->B3SOIPDldvt0Given = TRUE;
            break;
        case  B3SOIPD_MOD_LDVT1:             
            mod->B3SOIPDldvt1 = value->rValue;
            mod->B3SOIPDldvt1Given = TRUE;
            break;
        case  B3SOIPD_MOD_LDVT2:             
            mod->B3SOIPDldvt2 = value->rValue;
            mod->B3SOIPDldvt2Given = TRUE;
            break;
        case  B3SOIPD_MOD_LDVT0W:               
            mod->B3SOIPDldvt0w = value->rValue;
            mod->B3SOIPDldvt0wGiven = TRUE;
            break;
        case  B3SOIPD_MOD_LDVT1W:             
            mod->B3SOIPDldvt1w = value->rValue;
            mod->B3SOIPDldvt1wGiven = TRUE;
            break;
        case  B3SOIPD_MOD_LDVT2W:             
            mod->B3SOIPDldvt2w = value->rValue;
            mod->B3SOIPDldvt2wGiven = TRUE;
            break;
        case  B3SOIPD_MOD_LU0 :
            mod->B3SOIPDlu0 = value->rValue;
            mod->B3SOIPDlu0Given = TRUE;
            break;
        case B3SOIPD_MOD_LUA:
            mod->B3SOIPDlua = value->rValue;
            mod->B3SOIPDluaGiven = TRUE;
            break;
        case B3SOIPD_MOD_LUB:
            mod->B3SOIPDlub = value->rValue;
            mod->B3SOIPDlubGiven = TRUE;
            break;
        case B3SOIPD_MOD_LUC:
            mod->B3SOIPDluc = value->rValue;
            mod->B3SOIPDlucGiven = TRUE;
            break;
        case B3SOIPD_MOD_LVSAT:
            mod->B3SOIPDlvsat = value->rValue;
            mod->B3SOIPDlvsatGiven = TRUE;
            break;
        case B3SOIPD_MOD_LA0:
            mod->B3SOIPDla0 = value->rValue;
            mod->B3SOIPDla0Given = TRUE;
            break;
        case B3SOIPD_MOD_LAGS:
            mod->B3SOIPDlags= value->rValue;
            mod->B3SOIPDlagsGiven = TRUE;
            break;
        case  B3SOIPD_MOD_LB0 :
            mod->B3SOIPDlb0 = value->rValue;
            mod->B3SOIPDlb0Given = TRUE;
            break;
        case  B3SOIPD_MOD_LB1 :
            mod->B3SOIPDlb1 = value->rValue;
            mod->B3SOIPDlb1Given = TRUE;
            break;
        case B3SOIPD_MOD_LKETA:
            mod->B3SOIPDlketa = value->rValue;
            mod->B3SOIPDlketaGiven = TRUE;
            break;    
        case B3SOIPD_MOD_LKETAS:
            mod->B3SOIPDlketas = value->rValue;
            mod->B3SOIPDlketasGiven = TRUE;
            break;    
        case B3SOIPD_MOD_LA1:
            mod->B3SOIPDla1 = value->rValue;
            mod->B3SOIPDla1Given = TRUE;
            break;
        case B3SOIPD_MOD_LA2:
            mod->B3SOIPDla2 = value->rValue;
            mod->B3SOIPDla2Given = TRUE;
            break;
        case B3SOIPD_MOD_LRDSW:
            mod->B3SOIPDlrdsw = value->rValue;
            mod->B3SOIPDlrdswGiven = TRUE;
            break;                     
        case B3SOIPD_MOD_LPRWB:
            mod->B3SOIPDlprwb = value->rValue;
            mod->B3SOIPDlprwbGiven = TRUE;
            break;                     
        case B3SOIPD_MOD_LPRWG:
            mod->B3SOIPDlprwg = value->rValue;
            mod->B3SOIPDlprwgGiven = TRUE;
            break;                     
        case  B3SOIPD_MOD_LWR :
            mod->B3SOIPDlwr = value->rValue;
            mod->B3SOIPDlwrGiven = TRUE;
            break;
        case  B3SOIPD_MOD_LNFACTOR :
            mod->B3SOIPDlnfactor = value->rValue;
            mod->B3SOIPDlnfactorGiven = TRUE;
            break;
        case  B3SOIPD_MOD_LDWG :
            mod->B3SOIPDldwg = value->rValue;
            mod->B3SOIPDldwgGiven = TRUE;
            break;
        case  B3SOIPD_MOD_LDWB :
            mod->B3SOIPDldwb = value->rValue;
            mod->B3SOIPDldwbGiven = TRUE;
            break;
        case B3SOIPD_MOD_LVOFF:
            mod->B3SOIPDlvoff = value->rValue;
            mod->B3SOIPDlvoffGiven = TRUE;
            break;
        case B3SOIPD_MOD_LETA0:
            mod->B3SOIPDleta0 = value->rValue;
            mod->B3SOIPDleta0Given = TRUE;
            break;                 
        case B3SOIPD_MOD_LETAB:
            mod->B3SOIPDletab = value->rValue;
            mod->B3SOIPDletabGiven = TRUE;
            break;                 
        case  B3SOIPD_MOD_LDSUB:             
            mod->B3SOIPDldsub = value->rValue;
            mod->B3SOIPDldsubGiven = TRUE;
            break;
        case  B3SOIPD_MOD_LCIT :
            mod->B3SOIPDlcit = value->rValue;
            mod->B3SOIPDlcitGiven = TRUE;
            break;
        case  B3SOIPD_MOD_LCDSC :
            mod->B3SOIPDlcdsc = value->rValue;
            mod->B3SOIPDlcdscGiven = TRUE;
            break;
        case  B3SOIPD_MOD_LCDSCB :
            mod->B3SOIPDlcdscb = value->rValue;
            mod->B3SOIPDlcdscbGiven = TRUE;
            break;
        case  B3SOIPD_MOD_LCDSCD :
            mod->B3SOIPDlcdscd = value->rValue;
            mod->B3SOIPDlcdscdGiven = TRUE;
            break;
        case B3SOIPD_MOD_LPCLM:
            mod->B3SOIPDlpclm = value->rValue;
            mod->B3SOIPDlpclmGiven = TRUE;
            break;                 
        case B3SOIPD_MOD_LPDIBL1:
            mod->B3SOIPDlpdibl1 = value->rValue;
            mod->B3SOIPDlpdibl1Given = TRUE;
            break;                 
        case B3SOIPD_MOD_LPDIBL2:
            mod->B3SOIPDlpdibl2 = value->rValue;
            mod->B3SOIPDlpdibl2Given = TRUE;
            break;                 
        case B3SOIPD_MOD_LPDIBLB:
            mod->B3SOIPDlpdiblb = value->rValue;
            mod->B3SOIPDlpdiblbGiven = TRUE;
            break;                 
        case  B3SOIPD_MOD_LDROUT:             
            mod->B3SOIPDldrout = value->rValue;
            mod->B3SOIPDldroutGiven = TRUE;
            break;
        case B3SOIPD_MOD_LPVAG:
            mod->B3SOIPDlpvag = value->rValue;
            mod->B3SOIPDlpvagGiven = TRUE;
            break;                 
        case  B3SOIPD_MOD_LDELTA :
            mod->B3SOIPDldelta = value->rValue;
            mod->B3SOIPDldeltaGiven = TRUE;
            break;
        case  B3SOIPD_MOD_LALPHA0 :
            mod->B3SOIPDlalpha0 = value->rValue;
            mod->B3SOIPDlalpha0Given = TRUE;
            break;
        case  B3SOIPD_MOD_LFBJTII :
            mod->B3SOIPDlfbjtii = value->rValue;
            mod->B3SOIPDlfbjtiiGiven = TRUE;
            break;
        case  B3SOIPD_MOD_LBETA0 :
            mod->B3SOIPDlbeta0 = value->rValue;
            mod->B3SOIPDlbeta0Given = TRUE;
            break;
        case  B3SOIPD_MOD_LBETA1 :
            mod->B3SOIPDlbeta1 = value->rValue;
            mod->B3SOIPDlbeta1Given = TRUE;
            break;
        case  B3SOIPD_MOD_LBETA2 :
            mod->B3SOIPDlbeta2 = value->rValue;
            mod->B3SOIPDlbeta2Given = TRUE;
            break;
        case  B3SOIPD_MOD_LVDSATII0 :
            mod->B3SOIPDlvdsatii0 = value->rValue;
            mod->B3SOIPDlvdsatii0Given = TRUE;
            break;
        case  B3SOIPD_MOD_LLII :
            mod->B3SOIPDllii = value->rValue;
            mod->B3SOIPDlliiGiven = TRUE;
            break;
        case  B3SOIPD_MOD_LESATII :
            mod->B3SOIPDlesatii = value->rValue;
            mod->B3SOIPDlesatiiGiven = TRUE;
            break;
        case  B3SOIPD_MOD_LSII0 :
            mod->B3SOIPDlsii0 = value->rValue;
            mod->B3SOIPDlsii0Given = TRUE;
            break;
        case  B3SOIPD_MOD_LSII1 :
            mod->B3SOIPDlsii1 = value->rValue;
            mod->B3SOIPDlsii1Given = TRUE;
            break;
        case  B3SOIPD_MOD_LSII2 :
            mod->B3SOIPDlsii2 = value->rValue;
            mod->B3SOIPDlsii2Given = TRUE;
            break;
        case  B3SOIPD_MOD_LSIID :
            mod->B3SOIPDlsiid = value->rValue;
            mod->B3SOIPDlsiidGiven = TRUE;
            break;
        case  B3SOIPD_MOD_LAGIDL :
            mod->B3SOIPDlagidl = value->rValue;
            mod->B3SOIPDlagidlGiven = TRUE;
            break;
        case  B3SOIPD_MOD_LBGIDL :
            mod->B3SOIPDlbgidl = value->rValue;
            mod->B3SOIPDlbgidlGiven = TRUE;
            break;
        case  B3SOIPD_MOD_LNGIDL :
            mod->B3SOIPDlngidl = value->rValue;
            mod->B3SOIPDlngidlGiven = TRUE;
            break;
        case  B3SOIPD_MOD_LNTUN :
            mod->B3SOIPDlntun = value->rValue;
            mod->B3SOIPDlntunGiven = TRUE;
            break;
        case  B3SOIPD_MOD_LNDIODE :
            mod->B3SOIPDlndiode = value->rValue;
            mod->B3SOIPDlndiodeGiven = TRUE;
            break;
        case  B3SOIPD_MOD_LNRECF0 :
            mod->B3SOIPDlnrecf0 = value->rValue;
            mod->B3SOIPDlnrecf0Given = TRUE;
            break;
        case  B3SOIPD_MOD_LNRECR0 :
            mod->B3SOIPDlnrecr0 = value->rValue;
            mod->B3SOIPDlnrecr0Given = TRUE;
            break;
        case  B3SOIPD_MOD_LISBJT :
            mod->B3SOIPDlisbjt = value->rValue;
            mod->B3SOIPDlisbjtGiven = TRUE;
            break;
        case  B3SOIPD_MOD_LISDIF :
            mod->B3SOIPDlisdif = value->rValue;
            mod->B3SOIPDlisdifGiven = TRUE;
            break;
        case  B3SOIPD_MOD_LISREC :
            mod->B3SOIPDlisrec = value->rValue;
            mod->B3SOIPDlisrecGiven = TRUE;
            break;
        case  B3SOIPD_MOD_LISTUN :
            mod->B3SOIPDlistun = value->rValue;
            mod->B3SOIPDlistunGiven = TRUE;
            break;
        case  B3SOIPD_MOD_LVREC0 :
            mod->B3SOIPDlvrec0 = value->rValue;
            mod->B3SOIPDlvrec0Given = TRUE;
            break;
        case  B3SOIPD_MOD_LVTUN0 :
            mod->B3SOIPDlvtun0 = value->rValue;
            mod->B3SOIPDlvtun0Given = TRUE;
            break;
        case  B3SOIPD_MOD_LNBJT :
            mod->B3SOIPDlnbjt = value->rValue;
            mod->B3SOIPDlnbjtGiven = TRUE;
            break;
        case  B3SOIPD_MOD_LLBJT0 :
            mod->B3SOIPDllbjt0 = value->rValue;
            mod->B3SOIPDllbjt0Given = TRUE;
            break;
        case  B3SOIPD_MOD_LVABJT :
            mod->B3SOIPDlvabjt = value->rValue;
            mod->B3SOIPDlvabjtGiven = TRUE;
            break;
        case  B3SOIPD_MOD_LAELY :
            mod->B3SOIPDlaely = value->rValue;
            mod->B3SOIPDlaelyGiven = TRUE;
            break;
        case  B3SOIPD_MOD_LAHLI :
            mod->B3SOIPDlahli = value->rValue;
            mod->B3SOIPDlahliGiven = TRUE;
            break;
	/* CV Model */
        case  B3SOIPD_MOD_LVSDFB :
            mod->B3SOIPDlvsdfb = value->rValue;
            mod->B3SOIPDlvsdfbGiven = TRUE;
            break;
        case  B3SOIPD_MOD_LVSDTH :
            mod->B3SOIPDlvsdth = value->rValue;
            mod->B3SOIPDlvsdthGiven = TRUE;
            break;
        case  B3SOIPD_MOD_LDELVT :
            mod->B3SOIPDldelvt = value->rValue;
            mod->B3SOIPDldelvtGiven = TRUE;
            break;
        case  B3SOIPD_MOD_LACDE :
            mod->B3SOIPDlacde = value->rValue;
            mod->B3SOIPDlacdeGiven = TRUE;
            break;
        case  B3SOIPD_MOD_LMOIN :
            mod->B3SOIPDlmoin = value->rValue;
            mod->B3SOIPDlmoinGiven = TRUE;
            break;

        /* Width Dependence */
        case B3SOIPD_MOD_WNPEAK:
            mod->B3SOIPDwnpeak = value->rValue;
            mod->B3SOIPDwnpeakGiven = TRUE;
            break;
        case B3SOIPD_MOD_WNSUB:
            mod->B3SOIPDwnsub = value->rValue;
            mod->B3SOIPDwnsubGiven = TRUE;
            break;
        case B3SOIPD_MOD_WNGATE:
            mod->B3SOIPDwngate = value->rValue;
            mod->B3SOIPDwngateGiven = TRUE;
            break;
        case B3SOIPD_MOD_WVTH0:
            mod->B3SOIPDwvth0 = value->rValue;
            mod->B3SOIPDwvth0Given = TRUE;
            break;
        case  B3SOIPD_MOD_WK1:
            mod->B3SOIPDwk1 = value->rValue;
            mod->B3SOIPDwk1Given = TRUE;
            break;
        case  B3SOIPD_MOD_WK1W1:
            mod->B3SOIPDwk1w1 = value->rValue;
            mod->B3SOIPDwk1w1Given = TRUE;
            break;
        case  B3SOIPD_MOD_WK1W2:
            mod->B3SOIPDwk1w2 = value->rValue;
            mod->B3SOIPDwk1w2Given = TRUE;
            break;
        case  B3SOIPD_MOD_WK2:
            mod->B3SOIPDwk2 = value->rValue;
            mod->B3SOIPDwk2Given = TRUE;
            break;
        case  B3SOIPD_MOD_WK3:
            mod->B3SOIPDwk3 = value->rValue;
            mod->B3SOIPDwk3Given = TRUE;
            break;
        case  B3SOIPD_MOD_WK3B:
            mod->B3SOIPDwk3b = value->rValue;
            mod->B3SOIPDwk3bGiven = TRUE;
            break;
        case  B3SOIPD_MOD_WKB1 :
            mod->B3SOIPDwkb1 = value->rValue;
            mod->B3SOIPDwkb1Given = TRUE;
            break;
        case  B3SOIPD_MOD_WW0:
            mod->B3SOIPDww0 = value->rValue;
            mod->B3SOIPDww0Given = TRUE;
            break;
        case  B3SOIPD_MOD_WNLX:
            mod->B3SOIPDwnlx = value->rValue;
            mod->B3SOIPDwnlxGiven = TRUE;
            break;
        case  B3SOIPD_MOD_WDVT0:               
            mod->B3SOIPDwdvt0 = value->rValue;
            mod->B3SOIPDwdvt0Given = TRUE;
            break;
        case  B3SOIPD_MOD_WDVT1:             
            mod->B3SOIPDwdvt1 = value->rValue;
            mod->B3SOIPDwdvt1Given = TRUE;
            break;
        case  B3SOIPD_MOD_WDVT2:             
            mod->B3SOIPDwdvt2 = value->rValue;
            mod->B3SOIPDwdvt2Given = TRUE;
            break;
        case  B3SOIPD_MOD_WDVT0W:               
            mod->B3SOIPDwdvt0w = value->rValue;
            mod->B3SOIPDwdvt0wGiven = TRUE;
            break;
        case  B3SOIPD_MOD_WDVT1W:             
            mod->B3SOIPDwdvt1w = value->rValue;
            mod->B3SOIPDwdvt1wGiven = TRUE;
            break;
        case  B3SOIPD_MOD_WDVT2W:             
            mod->B3SOIPDwdvt2w = value->rValue;
            mod->B3SOIPDwdvt2wGiven = TRUE;
            break;
        case  B3SOIPD_MOD_WU0 :
            mod->B3SOIPDwu0 = value->rValue;
            mod->B3SOIPDwu0Given = TRUE;
            break;
        case B3SOIPD_MOD_WUA:
            mod->B3SOIPDwua = value->rValue;
            mod->B3SOIPDwuaGiven = TRUE;
            break;
        case B3SOIPD_MOD_WUB:
            mod->B3SOIPDwub = value->rValue;
            mod->B3SOIPDwubGiven = TRUE;
            break;
        case B3SOIPD_MOD_WUC:
            mod->B3SOIPDwuc = value->rValue;
            mod->B3SOIPDwucGiven = TRUE;
            break;
        case B3SOIPD_MOD_WVSAT:
            mod->B3SOIPDwvsat = value->rValue;
            mod->B3SOIPDwvsatGiven = TRUE;
            break;
        case B3SOIPD_MOD_WA0:
            mod->B3SOIPDwa0 = value->rValue;
            mod->B3SOIPDwa0Given = TRUE;
            break;
        case B3SOIPD_MOD_WAGS:
            mod->B3SOIPDwags= value->rValue;
            mod->B3SOIPDwagsGiven = TRUE;
            break;
        case  B3SOIPD_MOD_WB0 :
            mod->B3SOIPDwb0 = value->rValue;
            mod->B3SOIPDwb0Given = TRUE;
            break;
        case  B3SOIPD_MOD_WB1 :
            mod->B3SOIPDwb1 = value->rValue;
            mod->B3SOIPDwb1Given = TRUE;
            break;
        case B3SOIPD_MOD_WKETA:
            mod->B3SOIPDwketa = value->rValue;
            mod->B3SOIPDwketaGiven = TRUE;
            break;    
        case B3SOIPD_MOD_WKETAS:
            mod->B3SOIPDwketas = value->rValue;
            mod->B3SOIPDwketasGiven = TRUE;
            break;    
        case B3SOIPD_MOD_WA1:
            mod->B3SOIPDwa1 = value->rValue;
            mod->B3SOIPDwa1Given = TRUE;
            break;
        case B3SOIPD_MOD_WA2:
            mod->B3SOIPDwa2 = value->rValue;
            mod->B3SOIPDwa2Given = TRUE;
            break;
        case B3SOIPD_MOD_WRDSW:
            mod->B3SOIPDwrdsw = value->rValue;
            mod->B3SOIPDwrdswGiven = TRUE;
            break;                     
        case B3SOIPD_MOD_WPRWB:
            mod->B3SOIPDwprwb = value->rValue;
            mod->B3SOIPDwprwbGiven = TRUE;
            break;                     
        case B3SOIPD_MOD_WPRWG:
            mod->B3SOIPDwprwg = value->rValue;
            mod->B3SOIPDwprwgGiven = TRUE;
            break;                     
        case  B3SOIPD_MOD_WWR :
            mod->B3SOIPDwwr = value->rValue;
            mod->B3SOIPDwwrGiven = TRUE;
            break;
        case  B3SOIPD_MOD_WNFACTOR :
            mod->B3SOIPDwnfactor = value->rValue;
            mod->B3SOIPDwnfactorGiven = TRUE;
            break;
        case  B3SOIPD_MOD_WDWG :
            mod->B3SOIPDwdwg = value->rValue;
            mod->B3SOIPDwdwgGiven = TRUE;
            break;
        case  B3SOIPD_MOD_WDWB :
            mod->B3SOIPDwdwb = value->rValue;
            mod->B3SOIPDwdwbGiven = TRUE;
            break;
        case B3SOIPD_MOD_WVOFF:
            mod->B3SOIPDwvoff = value->rValue;
            mod->B3SOIPDwvoffGiven = TRUE;
            break;
        case B3SOIPD_MOD_WETA0:
            mod->B3SOIPDweta0 = value->rValue;
            mod->B3SOIPDweta0Given = TRUE;
            break;                 
        case B3SOIPD_MOD_WETAB:
            mod->B3SOIPDwetab = value->rValue;
            mod->B3SOIPDwetabGiven = TRUE;
            break;                 
        case  B3SOIPD_MOD_WDSUB:             
            mod->B3SOIPDwdsub = value->rValue;
            mod->B3SOIPDwdsubGiven = TRUE;
            break;
        case  B3SOIPD_MOD_WCIT :
            mod->B3SOIPDwcit = value->rValue;
            mod->B3SOIPDwcitGiven = TRUE;
            break;
        case  B3SOIPD_MOD_WCDSC :
            mod->B3SOIPDwcdsc = value->rValue;
            mod->B3SOIPDwcdscGiven = TRUE;
            break;
        case  B3SOIPD_MOD_WCDSCB :
            mod->B3SOIPDwcdscb = value->rValue;
            mod->B3SOIPDwcdscbGiven = TRUE;
            break;
        case  B3SOIPD_MOD_WCDSCD :
            mod->B3SOIPDwcdscd = value->rValue;
            mod->B3SOIPDwcdscdGiven = TRUE;
            break;
        case B3SOIPD_MOD_WPCLM:
            mod->B3SOIPDwpclm = value->rValue;
            mod->B3SOIPDwpclmGiven = TRUE;
            break;                 
        case B3SOIPD_MOD_WPDIBL1:
            mod->B3SOIPDwpdibl1 = value->rValue;
            mod->B3SOIPDwpdibl1Given = TRUE;
            break;                 
        case B3SOIPD_MOD_WPDIBL2:
            mod->B3SOIPDwpdibl2 = value->rValue;
            mod->B3SOIPDwpdibl2Given = TRUE;
            break;                 
        case B3SOIPD_MOD_WPDIBLB:
            mod->B3SOIPDwpdiblb = value->rValue;
            mod->B3SOIPDwpdiblbGiven = TRUE;
            break;                 
        case  B3SOIPD_MOD_WDROUT:             
            mod->B3SOIPDwdrout = value->rValue;
            mod->B3SOIPDwdroutGiven = TRUE;
            break;
        case B3SOIPD_MOD_WPVAG:
            mod->B3SOIPDwpvag = value->rValue;
            mod->B3SOIPDwpvagGiven = TRUE;
            break;                 
        case  B3SOIPD_MOD_WDELTA :
            mod->B3SOIPDwdelta = value->rValue;
            mod->B3SOIPDwdeltaGiven = TRUE;
            break;
        case  B3SOIPD_MOD_WALPHA0 :
            mod->B3SOIPDwalpha0 = value->rValue;
            mod->B3SOIPDwalpha0Given = TRUE;
            break;
        case  B3SOIPD_MOD_WFBJTII :
            mod->B3SOIPDwfbjtii = value->rValue;
            mod->B3SOIPDwfbjtiiGiven = TRUE;
            break;
        case  B3SOIPD_MOD_WBETA0 :
            mod->B3SOIPDwbeta0 = value->rValue;
            mod->B3SOIPDwbeta0Given = TRUE;
            break;
        case  B3SOIPD_MOD_WBETA1 :
            mod->B3SOIPDwbeta1 = value->rValue;
            mod->B3SOIPDwbeta1Given = TRUE;
            break;
        case  B3SOIPD_MOD_WBETA2 :
            mod->B3SOIPDwbeta2 = value->rValue;
            mod->B3SOIPDwbeta2Given = TRUE;
            break;
        case  B3SOIPD_MOD_WVDSATII0 :
            mod->B3SOIPDwvdsatii0 = value->rValue;
            mod->B3SOIPDwvdsatii0Given = TRUE;
            break;
        case  B3SOIPD_MOD_WLII :
            mod->B3SOIPDwlii = value->rValue;
            mod->B3SOIPDwliiGiven = TRUE;
            break;
        case  B3SOIPD_MOD_WESATII :
            mod->B3SOIPDwesatii = value->rValue;
            mod->B3SOIPDwesatiiGiven = TRUE;
            break;
        case  B3SOIPD_MOD_WSII0 :
            mod->B3SOIPDwsii0 = value->rValue;
            mod->B3SOIPDwsii0Given = TRUE;
            break;
        case  B3SOIPD_MOD_WSII1 :
            mod->B3SOIPDwsii1 = value->rValue;
            mod->B3SOIPDwsii1Given = TRUE;
            break;
        case  B3SOIPD_MOD_WSII2 :
            mod->B3SOIPDwsii2 = value->rValue;
            mod->B3SOIPDwsii2Given = TRUE;
            break;
        case  B3SOIPD_MOD_WSIID :
            mod->B3SOIPDwsiid = value->rValue;
            mod->B3SOIPDwsiidGiven = TRUE;
            break;
        case  B3SOIPD_MOD_WAGIDL :
            mod->B3SOIPDwagidl = value->rValue;
            mod->B3SOIPDwagidlGiven = TRUE;
            break;
        case  B3SOIPD_MOD_WBGIDL :
            mod->B3SOIPDwbgidl = value->rValue;
            mod->B3SOIPDwbgidlGiven = TRUE;
            break;
        case  B3SOIPD_MOD_WNGIDL :
            mod->B3SOIPDwngidl = value->rValue;
            mod->B3SOIPDwngidlGiven = TRUE;
            break;
        case  B3SOIPD_MOD_WNTUN :
            mod->B3SOIPDwntun = value->rValue;
            mod->B3SOIPDwntunGiven = TRUE;
            break;
        case  B3SOIPD_MOD_WNDIODE :
            mod->B3SOIPDwndiode = value->rValue;
            mod->B3SOIPDwndiodeGiven = TRUE;
            break;
        case  B3SOIPD_MOD_WNRECF0 :
            mod->B3SOIPDwnrecf0 = value->rValue;
            mod->B3SOIPDwnrecf0Given = TRUE;
            break;
        case  B3SOIPD_MOD_WNRECR0 :
            mod->B3SOIPDwnrecr0 = value->rValue;
            mod->B3SOIPDwnrecr0Given = TRUE;
            break;
        case  B3SOIPD_MOD_WISBJT :
            mod->B3SOIPDwisbjt = value->rValue;
            mod->B3SOIPDwisbjtGiven = TRUE;
            break;
        case  B3SOIPD_MOD_WISDIF :
            mod->B3SOIPDwisdif = value->rValue;
            mod->B3SOIPDwisdifGiven = TRUE;
            break;
        case  B3SOIPD_MOD_WISREC :
            mod->B3SOIPDwisrec = value->rValue;
            mod->B3SOIPDwisrecGiven = TRUE;
            break;
        case  B3SOIPD_MOD_WISTUN :
            mod->B3SOIPDwistun = value->rValue;
            mod->B3SOIPDwistunGiven = TRUE;
            break;
        case  B3SOIPD_MOD_WVREC0 :
            mod->B3SOIPDwvrec0 = value->rValue;
            mod->B3SOIPDwvrec0Given = TRUE;
            break;
        case  B3SOIPD_MOD_WVTUN0 :
            mod->B3SOIPDwvtun0 = value->rValue;
            mod->B3SOIPDwvtun0Given = TRUE;
            break;
        case  B3SOIPD_MOD_WNBJT :
            mod->B3SOIPDwnbjt = value->rValue;
            mod->B3SOIPDwnbjtGiven = TRUE;
            break;
        case  B3SOIPD_MOD_WLBJT0 :
            mod->B3SOIPDwlbjt0 = value->rValue;
            mod->B3SOIPDwlbjt0Given = TRUE;
            break;
        case  B3SOIPD_MOD_WVABJT :
            mod->B3SOIPDwvabjt = value->rValue;
            mod->B3SOIPDwvabjtGiven = TRUE;
            break;
        case  B3SOIPD_MOD_WAELY :
            mod->B3SOIPDwaely = value->rValue;
            mod->B3SOIPDwaelyGiven = TRUE;
            break;
        case  B3SOIPD_MOD_WAHLI :
            mod->B3SOIPDwahli = value->rValue;
            mod->B3SOIPDwahliGiven = TRUE;
            break;
	/* CV Model */
        case  B3SOIPD_MOD_WVSDFB :
            mod->B3SOIPDwvsdfb = value->rValue;
            mod->B3SOIPDwvsdfbGiven = TRUE;
            break;
        case  B3SOIPD_MOD_WVSDTH :
            mod->B3SOIPDwvsdth = value->rValue;
            mod->B3SOIPDwvsdthGiven = TRUE;
            break;
        case  B3SOIPD_MOD_WDELVT :
            mod->B3SOIPDwdelvt = value->rValue;
            mod->B3SOIPDwdelvtGiven = TRUE;
            break;
        case  B3SOIPD_MOD_WACDE :
            mod->B3SOIPDwacde = value->rValue;
            mod->B3SOIPDwacdeGiven = TRUE;
            break;
        case  B3SOIPD_MOD_WMOIN :
            mod->B3SOIPDwmoin = value->rValue;
            mod->B3SOIPDwmoinGiven = TRUE;
            break;

        /* Cross-term Dependence */
        case B3SOIPD_MOD_PNPEAK:
            mod->B3SOIPDpnpeak = value->rValue;
            mod->B3SOIPDpnpeakGiven = TRUE;
            break;
        case B3SOIPD_MOD_PNSUB:
            mod->B3SOIPDpnsub = value->rValue;
            mod->B3SOIPDpnsubGiven = TRUE;
            break;
        case B3SOIPD_MOD_PNGATE:
            mod->B3SOIPDpngate = value->rValue;
            mod->B3SOIPDpngateGiven = TRUE;
            break;
        case B3SOIPD_MOD_PVTH0:
            mod->B3SOIPDpvth0 = value->rValue;
            mod->B3SOIPDpvth0Given = TRUE;
            break;
        case  B3SOIPD_MOD_PK1:
            mod->B3SOIPDpk1 = value->rValue;
            mod->B3SOIPDpk1Given = TRUE;
            break;
        case  B3SOIPD_MOD_PK1W1:
            mod->B3SOIPDpk1w1 = value->rValue;
            mod->B3SOIPDpk1w1Given = TRUE;
            break;
        case  B3SOIPD_MOD_PK1W2:
            mod->B3SOIPDpk1w2 = value->rValue;
            mod->B3SOIPDpk1w2Given = TRUE;
            break;
        case  B3SOIPD_MOD_PK2:
            mod->B3SOIPDpk2 = value->rValue;
            mod->B3SOIPDpk2Given = TRUE;
            break;
        case  B3SOIPD_MOD_PK3:
            mod->B3SOIPDpk3 = value->rValue;
            mod->B3SOIPDpk3Given = TRUE;
            break;
        case  B3SOIPD_MOD_PK3B:
            mod->B3SOIPDpk3b = value->rValue;
            mod->B3SOIPDpk3bGiven = TRUE;
            break;
        case  B3SOIPD_MOD_PKB1 :
            mod->B3SOIPDpkb1 = value->rValue;
            mod->B3SOIPDpkb1Given = TRUE;
            break;
        case  B3SOIPD_MOD_PW0:
            mod->B3SOIPDpw0 = value->rValue;
            mod->B3SOIPDpw0Given = TRUE;
            break;
        case  B3SOIPD_MOD_PNLX:
            mod->B3SOIPDpnlx = value->rValue;
            mod->B3SOIPDpnlxGiven = TRUE;
            break;
        case  B3SOIPD_MOD_PDVT0:               
            mod->B3SOIPDpdvt0 = value->rValue;
            mod->B3SOIPDpdvt0Given = TRUE;
            break;
        case  B3SOIPD_MOD_PDVT1:             
            mod->B3SOIPDpdvt1 = value->rValue;
            mod->B3SOIPDpdvt1Given = TRUE;
            break;
        case  B3SOIPD_MOD_PDVT2:             
            mod->B3SOIPDpdvt2 = value->rValue;
            mod->B3SOIPDpdvt2Given = TRUE;
            break;
        case  B3SOIPD_MOD_PDVT0W:               
            mod->B3SOIPDpdvt0w = value->rValue;
            mod->B3SOIPDpdvt0wGiven = TRUE;
            break;
        case  B3SOIPD_MOD_PDVT1W:             
            mod->B3SOIPDpdvt1w = value->rValue;
            mod->B3SOIPDpdvt1wGiven = TRUE;
            break;
        case  B3SOIPD_MOD_PDVT2W:             
            mod->B3SOIPDpdvt2w = value->rValue;
            mod->B3SOIPDpdvt2wGiven = TRUE;
            break;
        case  B3SOIPD_MOD_PU0 :
            mod->B3SOIPDpu0 = value->rValue;
            mod->B3SOIPDpu0Given = TRUE;
            break;
        case B3SOIPD_MOD_PUA:
            mod->B3SOIPDpua = value->rValue;
            mod->B3SOIPDpuaGiven = TRUE;
            break;
        case B3SOIPD_MOD_PUB:
            mod->B3SOIPDpub = value->rValue;
            mod->B3SOIPDpubGiven = TRUE;
            break;
        case B3SOIPD_MOD_PUC:
            mod->B3SOIPDpuc = value->rValue;
            mod->B3SOIPDpucGiven = TRUE;
            break;
        case B3SOIPD_MOD_PVSAT:
            mod->B3SOIPDpvsat = value->rValue;
            mod->B3SOIPDpvsatGiven = TRUE;
            break;
        case B3SOIPD_MOD_PA0:
            mod->B3SOIPDpa0 = value->rValue;
            mod->B3SOIPDpa0Given = TRUE;
            break;
        case B3SOIPD_MOD_PAGS:
            mod->B3SOIPDpags= value->rValue;
            mod->B3SOIPDpagsGiven = TRUE;
            break;
        case  B3SOIPD_MOD_PB0 :
            mod->B3SOIPDpb0 = value->rValue;
            mod->B3SOIPDpb0Given = TRUE;
            break;
        case  B3SOIPD_MOD_PB1 :
            mod->B3SOIPDpb1 = value->rValue;
            mod->B3SOIPDpb1Given = TRUE;
            break;
        case B3SOIPD_MOD_PKETA:
            mod->B3SOIPDpketa = value->rValue;
            mod->B3SOIPDpketaGiven = TRUE;
            break;    
        case B3SOIPD_MOD_PKETAS:
            mod->B3SOIPDpketas = value->rValue;
            mod->B3SOIPDpketasGiven = TRUE;
            break;    
        case B3SOIPD_MOD_PA1:
            mod->B3SOIPDpa1 = value->rValue;
            mod->B3SOIPDpa1Given = TRUE;
            break;
        case B3SOIPD_MOD_PA2:
            mod->B3SOIPDpa2 = value->rValue;
            mod->B3SOIPDpa2Given = TRUE;
            break;
        case B3SOIPD_MOD_PRDSW:
            mod->B3SOIPDprdsw = value->rValue;
            mod->B3SOIPDprdswGiven = TRUE;
            break;                     
        case B3SOIPD_MOD_PPRWB:
            mod->B3SOIPDpprwb = value->rValue;
            mod->B3SOIPDpprwbGiven = TRUE;
            break;                     
        case B3SOIPD_MOD_PPRWG:
            mod->B3SOIPDpprwg = value->rValue;
            mod->B3SOIPDpprwgGiven = TRUE;
            break;                     
        case  B3SOIPD_MOD_PWR :
            mod->B3SOIPDpwr = value->rValue;
            mod->B3SOIPDpwrGiven = TRUE;
            break;
        case  B3SOIPD_MOD_PNFACTOR :
            mod->B3SOIPDpnfactor = value->rValue;
            mod->B3SOIPDpnfactorGiven = TRUE;
            break;
        case  B3SOIPD_MOD_PDWG :
            mod->B3SOIPDpdwg = value->rValue;
            mod->B3SOIPDpdwgGiven = TRUE;
            break;
        case  B3SOIPD_MOD_PDWB :
            mod->B3SOIPDpdwb = value->rValue;
            mod->B3SOIPDpdwbGiven = TRUE;
            break;
        case B3SOIPD_MOD_PVOFF:
            mod->B3SOIPDpvoff = value->rValue;
            mod->B3SOIPDpvoffGiven = TRUE;
            break;
        case B3SOIPD_MOD_PETA0:
            mod->B3SOIPDpeta0 = value->rValue;
            mod->B3SOIPDpeta0Given = TRUE;
            break;                 
        case B3SOIPD_MOD_PETAB:
            mod->B3SOIPDpetab = value->rValue;
            mod->B3SOIPDpetabGiven = TRUE;
            break;                 
        case  B3SOIPD_MOD_PDSUB:             
            mod->B3SOIPDpdsub = value->rValue;
            mod->B3SOIPDpdsubGiven = TRUE;
            break;
        case  B3SOIPD_MOD_PCIT :
            mod->B3SOIPDpcit = value->rValue;
            mod->B3SOIPDpcitGiven = TRUE;
            break;
        case  B3SOIPD_MOD_PCDSC :
            mod->B3SOIPDpcdsc = value->rValue;
            mod->B3SOIPDpcdscGiven = TRUE;
            break;
        case  B3SOIPD_MOD_PCDSCB :
            mod->B3SOIPDpcdscb = value->rValue;
            mod->B3SOIPDpcdscbGiven = TRUE;
            break;
        case  B3SOIPD_MOD_PCDSCD :
            mod->B3SOIPDpcdscd = value->rValue;
            mod->B3SOIPDpcdscdGiven = TRUE;
            break;
        case B3SOIPD_MOD_PPCLM:
            mod->B3SOIPDppclm = value->rValue;
            mod->B3SOIPDppclmGiven = TRUE;
            break;                 
        case B3SOIPD_MOD_PPDIBL1:
            mod->B3SOIPDppdibl1 = value->rValue;
            mod->B3SOIPDppdibl1Given = TRUE;
            break;                 
        case B3SOIPD_MOD_PPDIBL2:
            mod->B3SOIPDppdibl2 = value->rValue;
            mod->B3SOIPDppdibl2Given = TRUE;
            break;                 
        case B3SOIPD_MOD_PPDIBLB:
            mod->B3SOIPDppdiblb = value->rValue;
            mod->B3SOIPDppdiblbGiven = TRUE;
            break;                 
        case  B3SOIPD_MOD_PDROUT:             
            mod->B3SOIPDpdrout = value->rValue;
            mod->B3SOIPDpdroutGiven = TRUE;
            break;
        case B3SOIPD_MOD_PPVAG:
            mod->B3SOIPDppvag = value->rValue;
            mod->B3SOIPDppvagGiven = TRUE;
            break;                 
        case  B3SOIPD_MOD_PDELTA :
            mod->B3SOIPDpdelta = value->rValue;
            mod->B3SOIPDpdeltaGiven = TRUE;
            break;
        case  B3SOIPD_MOD_PALPHA0 :
            mod->B3SOIPDpalpha0 = value->rValue;
            mod->B3SOIPDpalpha0Given = TRUE;
            break;
        case  B3SOIPD_MOD_PFBJTII :
            mod->B3SOIPDpfbjtii = value->rValue;
            mod->B3SOIPDpfbjtiiGiven = TRUE;
            break;
        case  B3SOIPD_MOD_PBETA0 :
            mod->B3SOIPDpbeta0 = value->rValue;
            mod->B3SOIPDpbeta0Given = TRUE;
            break;
        case  B3SOIPD_MOD_PBETA1 :
            mod->B3SOIPDpbeta1 = value->rValue;
            mod->B3SOIPDpbeta1Given = TRUE;
            break;
        case  B3SOIPD_MOD_PBETA2 :
            mod->B3SOIPDpbeta2 = value->rValue;
            mod->B3SOIPDpbeta2Given = TRUE;
            break;
        case  B3SOIPD_MOD_PVDSATII0 :
            mod->B3SOIPDpvdsatii0 = value->rValue;
            mod->B3SOIPDpvdsatii0Given = TRUE;
            break;
        case  B3SOIPD_MOD_PLII :
            mod->B3SOIPDplii = value->rValue;
            mod->B3SOIPDpliiGiven = TRUE;
            break;
        case  B3SOIPD_MOD_PESATII :
            mod->B3SOIPDpesatii = value->rValue;
            mod->B3SOIPDpesatiiGiven = TRUE;
            break;
        case  B3SOIPD_MOD_PSII0 :
            mod->B3SOIPDpsii0 = value->rValue;
            mod->B3SOIPDpsii0Given = TRUE;
            break;
        case  B3SOIPD_MOD_PSII1 :
            mod->B3SOIPDpsii1 = value->rValue;
            mod->B3SOIPDpsii1Given = TRUE;
            break;
        case  B3SOIPD_MOD_PSII2 :
            mod->B3SOIPDpsii2 = value->rValue;
            mod->B3SOIPDpsii2Given = TRUE;
            break;
        case  B3SOIPD_MOD_PSIID :
            mod->B3SOIPDpsiid = value->rValue;
            mod->B3SOIPDpsiidGiven = TRUE;
            break;
        case  B3SOIPD_MOD_PAGIDL :
            mod->B3SOIPDpagidl = value->rValue;
            mod->B3SOIPDpagidlGiven = TRUE;
            break;
        case  B3SOIPD_MOD_PBGIDL :
            mod->B3SOIPDpbgidl = value->rValue;
            mod->B3SOIPDpbgidlGiven = TRUE;
            break;
        case  B3SOIPD_MOD_PNGIDL :
            mod->B3SOIPDpngidl = value->rValue;
            mod->B3SOIPDpngidlGiven = TRUE;
            break;
        case  B3SOIPD_MOD_PNTUN :
            mod->B3SOIPDpntun = value->rValue;
            mod->B3SOIPDpntunGiven = TRUE;
            break;
        case  B3SOIPD_MOD_PNDIODE :
            mod->B3SOIPDpndiode = value->rValue;
            mod->B3SOIPDpndiodeGiven = TRUE;
            break;
        case  B3SOIPD_MOD_PNRECF0 :
            mod->B3SOIPDpnrecf0 = value->rValue;
            mod->B3SOIPDpnrecf0Given = TRUE;
            break;
        case  B3SOIPD_MOD_PNRECR0 :
            mod->B3SOIPDpnrecr0 = value->rValue;
            mod->B3SOIPDpnrecr0Given = TRUE;
            break;
        case  B3SOIPD_MOD_PISBJT :
            mod->B3SOIPDpisbjt = value->rValue;
            mod->B3SOIPDpisbjtGiven = TRUE;
            break;
        case  B3SOIPD_MOD_PISDIF :
            mod->B3SOIPDpisdif = value->rValue;
            mod->B3SOIPDpisdifGiven = TRUE;
            break;
        case  B3SOIPD_MOD_PISREC :
            mod->B3SOIPDpisrec = value->rValue;
            mod->B3SOIPDpisrecGiven = TRUE;
            break;
        case  B3SOIPD_MOD_PISTUN :
            mod->B3SOIPDpistun = value->rValue;
            mod->B3SOIPDpistunGiven = TRUE;
            break;
        case  B3SOIPD_MOD_PVREC0 :
            mod->B3SOIPDpvrec0 = value->rValue;
            mod->B3SOIPDpvrec0Given = TRUE;
            break;
        case  B3SOIPD_MOD_PVTUN0 :
            mod->B3SOIPDpvtun0 = value->rValue;
            mod->B3SOIPDpvtun0Given = TRUE;
            break;
        case  B3SOIPD_MOD_PNBJT :
            mod->B3SOIPDpnbjt = value->rValue;
            mod->B3SOIPDpnbjtGiven = TRUE;
            break;
        case  B3SOIPD_MOD_PLBJT0 :
            mod->B3SOIPDplbjt0 = value->rValue;
            mod->B3SOIPDplbjt0Given = TRUE;
            break;
        case  B3SOIPD_MOD_PVABJT :
            mod->B3SOIPDpvabjt = value->rValue;
            mod->B3SOIPDpvabjtGiven = TRUE;
            break;
        case  B3SOIPD_MOD_PAELY :
            mod->B3SOIPDpaely = value->rValue;
            mod->B3SOIPDpaelyGiven = TRUE;
            break;
        case  B3SOIPD_MOD_PAHLI :
            mod->B3SOIPDpahli = value->rValue;
            mod->B3SOIPDpahliGiven = TRUE;
            break;
	/* CV Model */
        case  B3SOIPD_MOD_PVSDFB :
            mod->B3SOIPDpvsdfb = value->rValue;
            mod->B3SOIPDpvsdfbGiven = TRUE;
            break;
        case  B3SOIPD_MOD_PVSDTH :
            mod->B3SOIPDpvsdth = value->rValue;
            mod->B3SOIPDpvsdthGiven = TRUE;
            break;
        case  B3SOIPD_MOD_PDELVT :
            mod->B3SOIPDpdelvt = value->rValue;
            mod->B3SOIPDpdelvtGiven = TRUE;
            break;
        case  B3SOIPD_MOD_PACDE :
            mod->B3SOIPDpacde = value->rValue;
            mod->B3SOIPDpacdeGiven = TRUE;
            break;
        case  B3SOIPD_MOD_PMOIN :
            mod->B3SOIPDpmoin = value->rValue;
            mod->B3SOIPDpmoinGiven = TRUE;
            break;
/* Added for binning - END */

        case  B3SOIPD_MOD_NMOS  :
            if(value->iValue) {
                mod->B3SOIPDtype = 1;
                mod->B3SOIPDtypeGiven = TRUE;
            }
            break;
        case  B3SOIPD_MOD_PMOS  :
            if(value->iValue) {
                mod->B3SOIPDtype = - 1;
                mod->B3SOIPDtypeGiven = TRUE;
            }
            break;
        default:
            return(E_BADPARM);
    }
    return(OK);
}


