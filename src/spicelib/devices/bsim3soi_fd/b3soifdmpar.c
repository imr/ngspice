/**********
Copyright 1999 Regents of the University of California.  All rights reserved.
Author: Weidong Liu and Pin Su         Feb 1999
Author: 1998 Samuel Fung, Dennis Sinitsky and Stephen Tang
File: b3soifdmpar.c          98/5/01
Modified by Wei Jin 99/9/27
Modified by Paolo Nenzi 2002
**********/

/*
 * Revision 2.1  99/9/27 Pin Su 
 * BSIMFD2.1 release
 */

#include "ngspice/ngspice.h"
#include "b3soifddef.h"
#include "ngspice/ifsim.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
B3SOIFDmParam(int param, IFvalue *value, GENmodel *inMod)
{
    B3SOIFDmodel *mod = (B3SOIFDmodel*)inMod;
    switch(param)
    {  
 
	case  B3SOIFD_MOD_MOBMOD :
            mod->B3SOIFDmobMod = value->iValue;
            mod->B3SOIFDmobModGiven = TRUE;
            break;
        case  B3SOIFD_MOD_BINUNIT :
            mod->B3SOIFDbinUnit = value->iValue;
            mod->B3SOIFDbinUnitGiven = TRUE;
            break;
        case  B3SOIFD_MOD_PARAMCHK :
            mod->B3SOIFDparamChk = value->iValue;
            mod->B3SOIFDparamChkGiven = TRUE;
            break;
        case  B3SOIFD_MOD_CAPMOD :
            mod->B3SOIFDcapMod = value->iValue;
            mod->B3SOIFDcapModGiven = TRUE;
            break;
        case  B3SOIFD_MOD_SHMOD :
            mod->B3SOIFDshMod = value->iValue;
            mod->B3SOIFDshModGiven = TRUE;
            break;
        case  B3SOIFD_MOD_NOIMOD :
            mod->B3SOIFDnoiMod = value->iValue;
            mod->B3SOIFDnoiModGiven = TRUE;
            break;
        case  B3SOIFD_MOD_VERSION :
            mod->B3SOIFDversion = value->rValue;
            mod->B3SOIFDversionGiven = TRUE;
            break;
        case  B3SOIFD_MOD_TOX :
            mod->B3SOIFDtox = value->rValue;
            mod->B3SOIFDtoxGiven = TRUE;
            break;

        case  B3SOIFD_MOD_CDSC :
            mod->B3SOIFDcdsc = value->rValue;
            mod->B3SOIFDcdscGiven = TRUE;
            break;
        case  B3SOIFD_MOD_CDSCB :
            mod->B3SOIFDcdscb = value->rValue;
            mod->B3SOIFDcdscbGiven = TRUE;
            break;

        case  B3SOIFD_MOD_CDSCD :
            mod->B3SOIFDcdscd = value->rValue;
            mod->B3SOIFDcdscdGiven = TRUE;
            break;

        case  B3SOIFD_MOD_CIT :
            mod->B3SOIFDcit = value->rValue;
            mod->B3SOIFDcitGiven = TRUE;
            break;
        case  B3SOIFD_MOD_NFACTOR :
            mod->B3SOIFDnfactor = value->rValue;
            mod->B3SOIFDnfactorGiven = TRUE;
            break;
        case B3SOIFD_MOD_VSAT:
            mod->B3SOIFDvsat = value->rValue;
            mod->B3SOIFDvsatGiven = TRUE;
            break;
        case B3SOIFD_MOD_A0:
            mod->B3SOIFDa0 = value->rValue;
            mod->B3SOIFDa0Given = TRUE;
            break;
        
        case B3SOIFD_MOD_AGS:
            mod->B3SOIFDags= value->rValue;
            mod->B3SOIFDagsGiven = TRUE;
            break;
        
        case B3SOIFD_MOD_A1:
            mod->B3SOIFDa1 = value->rValue;
            mod->B3SOIFDa1Given = TRUE;
            break;
        case B3SOIFD_MOD_A2:
            mod->B3SOIFDa2 = value->rValue;
            mod->B3SOIFDa2Given = TRUE;
            break;
        case B3SOIFD_MOD_AT:
            mod->B3SOIFDat = value->rValue;
            mod->B3SOIFDatGiven = TRUE;
            break;
        case B3SOIFD_MOD_KETA:
            mod->B3SOIFDketa = value->rValue;
            mod->B3SOIFDketaGiven = TRUE;
            break;    
        case B3SOIFD_MOD_NSUB:
            mod->B3SOIFDnsub = value->rValue;
            mod->B3SOIFDnsubGiven = TRUE;
            break;
        case B3SOIFD_MOD_NPEAK:
            mod->B3SOIFDnpeak = value->rValue;
            mod->B3SOIFDnpeakGiven = TRUE;
	    if (mod->B3SOIFDnpeak > 1.0e20)
		mod->B3SOIFDnpeak *= 1.0e-6;
            break;
        case B3SOIFD_MOD_NGATE:
            mod->B3SOIFDngate = value->rValue;
            mod->B3SOIFDngateGiven = TRUE;
	    if (mod->B3SOIFDngate > 1.0e23)
		mod->B3SOIFDngate *= 1.0e-6;
            break;
        case B3SOIFD_MOD_GAMMA1:
            mod->B3SOIFDgamma1 = value->rValue;
            mod->B3SOIFDgamma1Given = TRUE;
            break;
        case B3SOIFD_MOD_GAMMA2:
            mod->B3SOIFDgamma2 = value->rValue;
            mod->B3SOIFDgamma2Given = TRUE;
            break;
        case B3SOIFD_MOD_VBX:
            mod->B3SOIFDvbx = value->rValue;
            mod->B3SOIFDvbxGiven = TRUE;
            break;
        case B3SOIFD_MOD_VBM:
            mod->B3SOIFDvbm = value->rValue;
            mod->B3SOIFDvbmGiven = TRUE;
            break;
        case B3SOIFD_MOD_XT:
            mod->B3SOIFDxt = value->rValue;
            mod->B3SOIFDxtGiven = TRUE;
            break;
        case  B3SOIFD_MOD_K1:
            mod->B3SOIFDk1 = value->rValue;
            mod->B3SOIFDk1Given = TRUE;
            break;
        case  B3SOIFD_MOD_KT1:
            mod->B3SOIFDkt1 = value->rValue;
            mod->B3SOIFDkt1Given = TRUE;
            break;
        case  B3SOIFD_MOD_KT1L:
            mod->B3SOIFDkt1l = value->rValue;
            mod->B3SOIFDkt1lGiven = TRUE;
            break;
        case  B3SOIFD_MOD_KT2:
            mod->B3SOIFDkt2 = value->rValue;
            mod->B3SOIFDkt2Given = TRUE;
            break;
        case  B3SOIFD_MOD_K2:
            mod->B3SOIFDk2 = value->rValue;
            mod->B3SOIFDk2Given = TRUE;
            break;
        case  B3SOIFD_MOD_K3:
            mod->B3SOIFDk3 = value->rValue;
            mod->B3SOIFDk3Given = TRUE;
            break;
        case  B3SOIFD_MOD_K3B:
            mod->B3SOIFDk3b = value->rValue;
            mod->B3SOIFDk3bGiven = TRUE;
            break;
        case  B3SOIFD_MOD_NLX:
            mod->B3SOIFDnlx = value->rValue;
            mod->B3SOIFDnlxGiven = TRUE;
            break;
        case  B3SOIFD_MOD_W0:
            mod->B3SOIFDw0 = value->rValue;
            mod->B3SOIFDw0Given = TRUE;
            break;
        case  B3SOIFD_MOD_DVT0:               
            mod->B3SOIFDdvt0 = value->rValue;
            mod->B3SOIFDdvt0Given = TRUE;
            break;
        case  B3SOIFD_MOD_DVT1:             
            mod->B3SOIFDdvt1 = value->rValue;
            mod->B3SOIFDdvt1Given = TRUE;
            break;
        case  B3SOIFD_MOD_DVT2:             
            mod->B3SOIFDdvt2 = value->rValue;
            mod->B3SOIFDdvt2Given = TRUE;
            break;
        case  B3SOIFD_MOD_DVT0W:               
            mod->B3SOIFDdvt0w = value->rValue;
            mod->B3SOIFDdvt0wGiven = TRUE;
            break;
        case  B3SOIFD_MOD_DVT1W:             
            mod->B3SOIFDdvt1w = value->rValue;
            mod->B3SOIFDdvt1wGiven = TRUE;
            break;
        case  B3SOIFD_MOD_DVT2W:             
            mod->B3SOIFDdvt2w = value->rValue;
            mod->B3SOIFDdvt2wGiven = TRUE;
            break;
        case  B3SOIFD_MOD_DROUT:             
            mod->B3SOIFDdrout = value->rValue;
            mod->B3SOIFDdroutGiven = TRUE;
            break;
        case  B3SOIFD_MOD_DSUB:             
            mod->B3SOIFDdsub = value->rValue;
            mod->B3SOIFDdsubGiven = TRUE;
            break;
        case B3SOIFD_MOD_VTH0:
            mod->B3SOIFDvth0 = value->rValue;
            mod->B3SOIFDvth0Given = TRUE;
            break;
        case B3SOIFD_MOD_UA:
            mod->B3SOIFDua = value->rValue;
            mod->B3SOIFDuaGiven = TRUE;
            break;
        case B3SOIFD_MOD_UA1:
            mod->B3SOIFDua1 = value->rValue;
            mod->B3SOIFDua1Given = TRUE;
            break;
        case B3SOIFD_MOD_UB:
            mod->B3SOIFDub = value->rValue;
            mod->B3SOIFDubGiven = TRUE;
            break;
        case B3SOIFD_MOD_UB1:
            mod->B3SOIFDub1 = value->rValue;
            mod->B3SOIFDub1Given = TRUE;
            break;
        case B3SOIFD_MOD_UC:
            mod->B3SOIFDuc = value->rValue;
            mod->B3SOIFDucGiven = TRUE;
            break;
        case B3SOIFD_MOD_UC1:
            mod->B3SOIFDuc1 = value->rValue;
            mod->B3SOIFDuc1Given = TRUE;
            break;
        case  B3SOIFD_MOD_U0 :
            mod->B3SOIFDu0 = value->rValue;
            mod->B3SOIFDu0Given = TRUE;
            break;
        case  B3SOIFD_MOD_UTE :
            mod->B3SOIFDute = value->rValue;
            mod->B3SOIFDuteGiven = TRUE;
            break;
        case B3SOIFD_MOD_VOFF:
            mod->B3SOIFDvoff = value->rValue;
            mod->B3SOIFDvoffGiven = TRUE;
            break;
        case  B3SOIFD_MOD_DELTA :
            mod->B3SOIFDdelta = value->rValue;
            mod->B3SOIFDdeltaGiven = TRUE;
            break;
        case B3SOIFD_MOD_RDSW:
            mod->B3SOIFDrdsw = value->rValue;
            mod->B3SOIFDrdswGiven = TRUE;
            break;                     
        case B3SOIFD_MOD_PRWG:
            mod->B3SOIFDprwg = value->rValue;
            mod->B3SOIFDprwgGiven = TRUE;
            break;                     
        case B3SOIFD_MOD_PRWB:
            mod->B3SOIFDprwb = value->rValue;
            mod->B3SOIFDprwbGiven = TRUE;
            break;                     
        case B3SOIFD_MOD_PRT:
            mod->B3SOIFDprt = value->rValue;
            mod->B3SOIFDprtGiven = TRUE;
            break;                     
        case B3SOIFD_MOD_ETA0:
            mod->B3SOIFDeta0 = value->rValue;
            mod->B3SOIFDeta0Given = TRUE;
            break;                 
        case B3SOIFD_MOD_ETAB:
            mod->B3SOIFDetab = value->rValue;
            mod->B3SOIFDetabGiven = TRUE;
            break;                 
        case B3SOIFD_MOD_PCLM:
            mod->B3SOIFDpclm = value->rValue;
            mod->B3SOIFDpclmGiven = TRUE;
            break;                 
        case B3SOIFD_MOD_PDIBL1:
            mod->B3SOIFDpdibl1 = value->rValue;
            mod->B3SOIFDpdibl1Given = TRUE;
            break;                 
        case B3SOIFD_MOD_PDIBL2:
            mod->B3SOIFDpdibl2 = value->rValue;
            mod->B3SOIFDpdibl2Given = TRUE;
            break;                 
        case B3SOIFD_MOD_PDIBLB:
            mod->B3SOIFDpdiblb = value->rValue;
            mod->B3SOIFDpdiblbGiven = TRUE;
            break;                 
        case B3SOIFD_MOD_PVAG:
            mod->B3SOIFDpvag = value->rValue;
            mod->B3SOIFDpvagGiven = TRUE;
            break;                 
        case  B3SOIFD_MOD_WR :
            mod->B3SOIFDwr = value->rValue;
            mod->B3SOIFDwrGiven = TRUE;
            break;
        case  B3SOIFD_MOD_DWG :
            mod->B3SOIFDdwg = value->rValue;
            mod->B3SOIFDdwgGiven = TRUE;
            break;
        case  B3SOIFD_MOD_DWB :
            mod->B3SOIFDdwb = value->rValue;
            mod->B3SOIFDdwbGiven = TRUE;
            break;
        case  B3SOIFD_MOD_B0 :
            mod->B3SOIFDb0 = value->rValue;
            mod->B3SOIFDb0Given = TRUE;
            break;
        case  B3SOIFD_MOD_B1 :
            mod->B3SOIFDb1 = value->rValue;
            mod->B3SOIFDb1Given = TRUE;
            break;
        case  B3SOIFD_MOD_ALPHA0 :
            mod->B3SOIFDalpha0 = value->rValue;
            mod->B3SOIFDalpha0Given = TRUE;
            break;
        case  B3SOIFD_MOD_ALPHA1 :
            mod->B3SOIFDalpha1 = value->rValue;
            mod->B3SOIFDalpha1Given = TRUE;
            break;
        case  B3SOIFD_MOD_BETA0 :
            mod->B3SOIFDbeta0 = value->rValue;
            mod->B3SOIFDbeta0Given = TRUE;
            break;

        case  B3SOIFD_MOD_CGSL :
            mod->B3SOIFDcgsl = value->rValue;
            mod->B3SOIFDcgslGiven = TRUE;
            break;
        case  B3SOIFD_MOD_CGDL :
            mod->B3SOIFDcgdl = value->rValue;
            mod->B3SOIFDcgdlGiven = TRUE;
            break;
        case  B3SOIFD_MOD_CKAPPA :
            mod->B3SOIFDckappa = value->rValue;
            mod->B3SOIFDckappaGiven = TRUE;
            break;
        case  B3SOIFD_MOD_CF :
            mod->B3SOIFDcf = value->rValue;
            mod->B3SOIFDcfGiven = TRUE;
            break;
        case  B3SOIFD_MOD_CLC :
            mod->B3SOIFDclc = value->rValue;
            mod->B3SOIFDclcGiven = TRUE;
            break;
        case  B3SOIFD_MOD_CLE :
            mod->B3SOIFDcle = value->rValue;
            mod->B3SOIFDcleGiven = TRUE;
            break;
        case  B3SOIFD_MOD_DWC :
            mod->B3SOIFDdwc = value->rValue;
            mod->B3SOIFDdwcGiven = TRUE;
            break;
        case  B3SOIFD_MOD_DLC :
            mod->B3SOIFDdlc = value->rValue;
            mod->B3SOIFDdlcGiven = TRUE;
            break;
        case  B3SOIFD_MOD_TBOX :
            mod->B3SOIFDtbox = value->rValue;
            mod->B3SOIFDtboxGiven = TRUE;
            break;
        case  B3SOIFD_MOD_TSI :
            mod->B3SOIFDtsi = value->rValue;
            mod->B3SOIFDtsiGiven = TRUE;
            break;
        case  B3SOIFD_MOD_XJ :
            mod->B3SOIFDxj = value->rValue;
            mod->B3SOIFDxjGiven = TRUE;
            break;
        case  B3SOIFD_MOD_KB1 :
            mod->B3SOIFDkb1 = value->rValue;
            mod->B3SOIFDkb1Given = TRUE;
            break;
        case  B3SOIFD_MOD_KB3 :
            mod->B3SOIFDkb3 = value->rValue;
            mod->B3SOIFDkb3Given = TRUE;
            break;
        case  B3SOIFD_MOD_DVBD0 :
            mod->B3SOIFDdvbd0 = value->rValue;
            mod->B3SOIFDdvbd0Given = TRUE;
            break;
        case  B3SOIFD_MOD_DVBD1 :
            mod->B3SOIFDdvbd1 = value->rValue;
            mod->B3SOIFDdvbd1Given = TRUE;
            break;
        case  B3SOIFD_MOD_DELP :
            mod->B3SOIFDdelp = value->rValue;
            mod->B3SOIFDdelpGiven = TRUE;
            break;
        case  B3SOIFD_MOD_VBSA :
            mod->B3SOIFDvbsa = value->rValue;
            mod->B3SOIFDvbsaGiven = TRUE;
            break;
        case  B3SOIFD_MOD_RBODY :
            mod->B3SOIFDrbody = value->rValue;
            mod->B3SOIFDrbodyGiven = TRUE;
            break;
        case  B3SOIFD_MOD_RBSH :
            mod->B3SOIFDrbsh = value->rValue;
            mod->B3SOIFDrbshGiven = TRUE;
            break;
        case  B3SOIFD_MOD_ADICE0 :
            mod->B3SOIFDadice0 = value->rValue;
            mod->B3SOIFDadice0Given = TRUE;
            break;
        case  B3SOIFD_MOD_ABP :
            mod->B3SOIFDabp = value->rValue;
            mod->B3SOIFDabpGiven = TRUE;
            break;
        case  B3SOIFD_MOD_MXC :
            mod->B3SOIFDmxc = value->rValue;
            mod->B3SOIFDmxcGiven = TRUE;
            break;
        case  B3SOIFD_MOD_RTH0 :
            mod->B3SOIFDrth0 = value->rValue;
            mod->B3SOIFDrth0Given = TRUE;
            break;
        case  B3SOIFD_MOD_CTH0 :
            mod->B3SOIFDcth0 = value->rValue;
            mod->B3SOIFDcth0Given = TRUE;
            break;
        case  B3SOIFD_MOD_AII :
            mod->B3SOIFDaii = value->rValue;
            mod->B3SOIFDaiiGiven = TRUE;
            break;
        case  B3SOIFD_MOD_BII :
            mod->B3SOIFDbii = value->rValue;
            mod->B3SOIFDbiiGiven = TRUE;
            break;
        case  B3SOIFD_MOD_CII :
            mod->B3SOIFDcii = value->rValue;
            mod->B3SOIFDciiGiven = TRUE;
            break;
        case  B3SOIFD_MOD_DII :
            mod->B3SOIFDdii = value->rValue;
            mod->B3SOIFDdiiGiven = TRUE;
            break;
        case  B3SOIFD_MOD_NGIDL :
            mod->B3SOIFDngidl = value->rValue;
            mod->B3SOIFDngidlGiven = TRUE;
            break;
        case  B3SOIFD_MOD_AGIDL :
            mod->B3SOIFDagidl = value->rValue;
            mod->B3SOIFDagidlGiven = TRUE;
            break;
        case  B3SOIFD_MOD_BGIDL :
            mod->B3SOIFDbgidl = value->rValue;
            mod->B3SOIFDbgidlGiven = TRUE;
            break;
        case  B3SOIFD_MOD_NDIODE :
            mod->B3SOIFDndiode = value->rValue;
            mod->B3SOIFDndiodeGiven = TRUE;
            break;
        case  B3SOIFD_MOD_NTUN :
            mod->B3SOIFDntun = value->rValue;
            mod->B3SOIFDntunGiven = TRUE;
            break;
        case  B3SOIFD_MOD_ISBJT :
            mod->B3SOIFDisbjt = value->rValue;
            mod->B3SOIFDisbjtGiven = TRUE;
            break;
        case  B3SOIFD_MOD_ISDIF :
            mod->B3SOIFDisdif = value->rValue;
            mod->B3SOIFDisdifGiven = TRUE;
            break;
        case  B3SOIFD_MOD_ISREC :
            mod->B3SOIFDisrec = value->rValue;
            mod->B3SOIFDisrecGiven = TRUE;
            break;
        case  B3SOIFD_MOD_ISTUN :
            mod->B3SOIFDistun = value->rValue;
            mod->B3SOIFDistunGiven = TRUE;
            break;
        case  B3SOIFD_MOD_XBJT :
            mod->B3SOIFDxbjt = value->rValue;
            mod->B3SOIFDxbjtGiven = TRUE;
            break;
        case  B3SOIFD_MOD_XREC :
            mod->B3SOIFDxrec = value->rValue;
            mod->B3SOIFDxrecGiven = TRUE;
            break;
        case  B3SOIFD_MOD_XTUN :
            mod->B3SOIFDxtun = value->rValue;
            mod->B3SOIFDxtunGiven = TRUE;
            break;
        case  B3SOIFD_MOD_EDL :
            mod->B3SOIFDedl = value->rValue;
            mod->B3SOIFDedlGiven = TRUE;
            break;
        case  B3SOIFD_MOD_KBJT1 :
            mod->B3SOIFDkbjt1 = value->rValue;
            mod->B3SOIFDkbjt1Given = TRUE;
            break;
        case  B3SOIFD_MOD_TT :
            mod->B3SOIFDtt = value->rValue;
            mod->B3SOIFDttGiven = TRUE;
            break;
        case  B3SOIFD_MOD_VSDTH :
            mod->B3SOIFDvsdth = value->rValue;
            mod->B3SOIFDvsdthGiven = TRUE;
            break;
        case  B3SOIFD_MOD_VSDFB :
            mod->B3SOIFDvsdfb = value->rValue;
            mod->B3SOIFDvsdfbGiven = TRUE;
            break;
        case  B3SOIFD_MOD_CSDMIN :
            mod->B3SOIFDcsdmin = value->rValue;
            mod->B3SOIFDcsdminGiven = TRUE;
            break;
        case  B3SOIFD_MOD_ASD :
            mod->B3SOIFDasd = value->rValue;
            mod->B3SOIFDasdGiven = TRUE;
            break;


        case  B3SOIFD_MOD_TNOM :
            mod->B3SOIFDtnom = value->rValue + 273.15;
            mod->B3SOIFDtnomGiven = TRUE;
            break;
        case  B3SOIFD_MOD_CGSO :
            mod->B3SOIFDcgso = value->rValue;
            mod->B3SOIFDcgsoGiven = TRUE;
            break;
        case  B3SOIFD_MOD_CGDO :
            mod->B3SOIFDcgdo = value->rValue;
            mod->B3SOIFDcgdoGiven = TRUE;
            break;
        case  B3SOIFD_MOD_CGEO :
            mod->B3SOIFDcgeo = value->rValue;
            mod->B3SOIFDcgeoGiven = TRUE;
            break;
        case  B3SOIFD_MOD_XPART :
            mod->B3SOIFDxpart = value->rValue;
            mod->B3SOIFDxpartGiven = TRUE;
            break;
        case  B3SOIFD_MOD_RSH :
            mod->B3SOIFDsheetResistance = value->rValue;
            mod->B3SOIFDsheetResistanceGiven = TRUE;
            break;
        case  B3SOIFD_MOD_PBSWG :
            mod->B3SOIFDGatesidewallJctPotential = value->rValue;
            mod->B3SOIFDGatesidewallJctPotentialGiven = TRUE;
            break;
        case  B3SOIFD_MOD_MJSWG :
            mod->B3SOIFDbodyJctGateSideGradingCoeff = value->rValue;
            mod->B3SOIFDbodyJctGateSideGradingCoeffGiven = TRUE;
            break;
        case  B3SOIFD_MOD_CJSWG :
            mod->B3SOIFDunitLengthGateSidewallJctCap = value->rValue;
            mod->B3SOIFDunitLengthGateSidewallJctCapGiven = TRUE;
            break;
        case  B3SOIFD_MOD_CSDESW :
            mod->B3SOIFDcsdesw = value->rValue;
            mod->B3SOIFDcsdeswGiven = TRUE;
            break;
        case  B3SOIFD_MOD_LINT :
            mod->B3SOIFDLint = value->rValue;
            mod->B3SOIFDLintGiven = TRUE;
            break;
        case  B3SOIFD_MOD_LL :
            mod->B3SOIFDLl = value->rValue;
            mod->B3SOIFDLlGiven = TRUE;
            break;
        case  B3SOIFD_MOD_LLN :
            mod->B3SOIFDLln = value->rValue;
            mod->B3SOIFDLlnGiven = TRUE;
            break;
        case  B3SOIFD_MOD_LW :
            mod->B3SOIFDLw = value->rValue;
            mod->B3SOIFDLwGiven = TRUE;
            break;
        case  B3SOIFD_MOD_LWN :
            mod->B3SOIFDLwn = value->rValue;
            mod->B3SOIFDLwnGiven = TRUE;
            break;
        case  B3SOIFD_MOD_LWL :
            mod->B3SOIFDLwl = value->rValue;
            mod->B3SOIFDLwlGiven = TRUE;
            break;
        case  B3SOIFD_MOD_WINT :
            mod->B3SOIFDWint = value->rValue;
            mod->B3SOIFDWintGiven = TRUE;
            break;
        case  B3SOIFD_MOD_WL :
            mod->B3SOIFDWl = value->rValue;
            mod->B3SOIFDWlGiven = TRUE;
            break;
        case  B3SOIFD_MOD_WLN :
            mod->B3SOIFDWln = value->rValue;
            mod->B3SOIFDWlnGiven = TRUE;
            break;
        case  B3SOIFD_MOD_WW :
            mod->B3SOIFDWw = value->rValue;
            mod->B3SOIFDWwGiven = TRUE;
            break;
        case  B3SOIFD_MOD_WWN :
            mod->B3SOIFDWwn = value->rValue;
            mod->B3SOIFDWwnGiven = TRUE;
            break;
        case  B3SOIFD_MOD_WWL :
            mod->B3SOIFDWwl = value->rValue;
            mod->B3SOIFDWwlGiven = TRUE;
            break;

        case  B3SOIFD_MOD_NOIA :
            mod->B3SOIFDoxideTrapDensityA = value->rValue;
            mod->B3SOIFDoxideTrapDensityAGiven = TRUE;
            break;
        case  B3SOIFD_MOD_NOIB :
            mod->B3SOIFDoxideTrapDensityB = value->rValue;
            mod->B3SOIFDoxideTrapDensityBGiven = TRUE;
            break;
        case  B3SOIFD_MOD_NOIC :
            mod->B3SOIFDoxideTrapDensityC = value->rValue;
            mod->B3SOIFDoxideTrapDensityCGiven = TRUE;
            break;
        case  B3SOIFD_MOD_NOIF :
            mod->B3SOIFDnoif = value->rValue;
            mod->B3SOIFDnoifGiven = TRUE;
            break;
        case  B3SOIFD_MOD_EM :
            mod->B3SOIFDem = value->rValue;
            mod->B3SOIFDemGiven = TRUE;
            break;
        case  B3SOIFD_MOD_EF :
            mod->B3SOIFDef = value->rValue;
            mod->B3SOIFDefGiven = TRUE;
            break;
        case  B3SOIFD_MOD_AF :
            mod->B3SOIFDaf = value->rValue;
            mod->B3SOIFDafGiven = TRUE;
            break;
        case  B3SOIFD_MOD_KF :
            mod->B3SOIFDkf = value->rValue;
            mod->B3SOIFDkfGiven = TRUE;
            break;

/* Added for binning - START */
        /* Length Dependence */
        case B3SOIFD_MOD_LNPEAK:
            mod->B3SOIFDlnpeak = value->rValue;
            mod->B3SOIFDlnpeakGiven = TRUE;
            break;
        case B3SOIFD_MOD_LNSUB:
            mod->B3SOIFDlnsub = value->rValue;
            mod->B3SOIFDlnsubGiven = TRUE;
            break;
        case B3SOIFD_MOD_LNGATE:
            mod->B3SOIFDlngate = value->rValue;
            mod->B3SOIFDlngateGiven = TRUE;
            break;
        case B3SOIFD_MOD_LVTH0:
            mod->B3SOIFDlvth0 = value->rValue;
            mod->B3SOIFDlvth0Given = TRUE;
            break;
        case  B3SOIFD_MOD_LK1:
            mod->B3SOIFDlk1 = value->rValue;
            mod->B3SOIFDlk1Given = TRUE;
            break;
        case  B3SOIFD_MOD_LK2:
            mod->B3SOIFDlk2 = value->rValue;
            mod->B3SOIFDlk2Given = TRUE;
            break;
        case  B3SOIFD_MOD_LK3:
            mod->B3SOIFDlk3 = value->rValue;
            mod->B3SOIFDlk3Given = TRUE;
            break;
        case  B3SOIFD_MOD_LK3B:
            mod->B3SOIFDlk3b = value->rValue;
            mod->B3SOIFDlk3bGiven = TRUE;
            break;
        case  B3SOIFD_MOD_LVBSA:
            mod->B3SOIFDlvbsa = value->rValue;
            mod->B3SOIFDlvbsaGiven = TRUE;
            break;
        case  B3SOIFD_MOD_LDELP:
            mod->B3SOIFDldelp = value->rValue;
            mod->B3SOIFDldelpGiven = TRUE;
            break;
        case  B3SOIFD_MOD_LKB1 :
            mod->B3SOIFDlkb1 = value->rValue;
            mod->B3SOIFDlkb1Given = TRUE;
            break;
        case  B3SOIFD_MOD_LKB3 :
            mod->B3SOIFDlkb3 = value->rValue;
            mod->B3SOIFDlkb3Given = TRUE;
            break;
        case  B3SOIFD_MOD_LDVBD0 :
            mod->B3SOIFDldvbd0 = value->rValue;
            mod->B3SOIFDldvbd0Given = TRUE;
            break;
        case  B3SOIFD_MOD_LDVBD1 :
            mod->B3SOIFDldvbd1 = value->rValue;
            mod->B3SOIFDldvbd1Given = TRUE;
            break;
        case  B3SOIFD_MOD_LW0:
            mod->B3SOIFDlw0 = value->rValue;
            mod->B3SOIFDlw0Given = TRUE;
            break;
        case  B3SOIFD_MOD_LNLX:
            mod->B3SOIFDlnlx = value->rValue;
            mod->B3SOIFDlnlxGiven = TRUE;
            break;
        case  B3SOIFD_MOD_LDVT0:               
            mod->B3SOIFDldvt0 = value->rValue;
            mod->B3SOIFDldvt0Given = TRUE;
            break;
        case  B3SOIFD_MOD_LDVT1:             
            mod->B3SOIFDldvt1 = value->rValue;
            mod->B3SOIFDldvt1Given = TRUE;
            break;
        case  B3SOIFD_MOD_LDVT2:             
            mod->B3SOIFDldvt2 = value->rValue;
            mod->B3SOIFDldvt2Given = TRUE;
            break;
        case  B3SOIFD_MOD_LDVT0W:               
            mod->B3SOIFDldvt0w = value->rValue;
            mod->B3SOIFDldvt0wGiven = TRUE;
            break;
        case  B3SOIFD_MOD_LDVT1W:             
            mod->B3SOIFDldvt1w = value->rValue;
            mod->B3SOIFDldvt1wGiven = TRUE;
            break;
        case  B3SOIFD_MOD_LDVT2W:             
            mod->B3SOIFDldvt2w = value->rValue;
            mod->B3SOIFDldvt2wGiven = TRUE;
            break;
        case  B3SOIFD_MOD_LU0 :
            mod->B3SOIFDlu0 = value->rValue;
            mod->B3SOIFDlu0Given = TRUE;
            break;
        case B3SOIFD_MOD_LUA:
            mod->B3SOIFDlua = value->rValue;
            mod->B3SOIFDluaGiven = TRUE;
            break;
        case B3SOIFD_MOD_LUB:
            mod->B3SOIFDlub = value->rValue;
            mod->B3SOIFDlubGiven = TRUE;
            break;
        case B3SOIFD_MOD_LUC:
            mod->B3SOIFDluc = value->rValue;
            mod->B3SOIFDlucGiven = TRUE;
            break;
        case B3SOIFD_MOD_LVSAT:
            mod->B3SOIFDlvsat = value->rValue;
            mod->B3SOIFDlvsatGiven = TRUE;
            break;
        case B3SOIFD_MOD_LA0:
            mod->B3SOIFDla0 = value->rValue;
            mod->B3SOIFDla0Given = TRUE;
            break;
        case B3SOIFD_MOD_LAGS:
            mod->B3SOIFDlags= value->rValue;
            mod->B3SOIFDlagsGiven = TRUE;
            break;
        case  B3SOIFD_MOD_LB0 :
            mod->B3SOIFDlb0 = value->rValue;
            mod->B3SOIFDlb0Given = TRUE;
            break;
        case  B3SOIFD_MOD_LB1 :
            mod->B3SOIFDlb1 = value->rValue;
            mod->B3SOIFDlb1Given = TRUE;
            break;
        case B3SOIFD_MOD_LKETA:
            mod->B3SOIFDlketa = value->rValue;
            mod->B3SOIFDlketaGiven = TRUE;
            break;    
        case B3SOIFD_MOD_LABP:
            mod->B3SOIFDlabp = value->rValue;
            mod->B3SOIFDlabpGiven = TRUE;
            break;    
        case B3SOIFD_MOD_LMXC:
            mod->B3SOIFDlmxc = value->rValue;
            mod->B3SOIFDlmxcGiven = TRUE;
            break;    
        case B3SOIFD_MOD_LADICE0:
            mod->B3SOIFDladice0 = value->rValue;
            mod->B3SOIFDladice0Given = TRUE;
            break;    
        case B3SOIFD_MOD_LA1:
            mod->B3SOIFDla1 = value->rValue;
            mod->B3SOIFDla1Given = TRUE;
            break;
        case B3SOIFD_MOD_LA2:
            mod->B3SOIFDla2 = value->rValue;
            mod->B3SOIFDla2Given = TRUE;
            break;
        case B3SOIFD_MOD_LRDSW:
            mod->B3SOIFDlrdsw = value->rValue;
            mod->B3SOIFDlrdswGiven = TRUE;
            break;                     
        case B3SOIFD_MOD_LPRWB:
            mod->B3SOIFDlprwb = value->rValue;
            mod->B3SOIFDlprwbGiven = TRUE;
            break;                     
        case B3SOIFD_MOD_LPRWG:
            mod->B3SOIFDlprwg = value->rValue;
            mod->B3SOIFDlprwgGiven = TRUE;
            break;                     
        case  B3SOIFD_MOD_LWR :
            mod->B3SOIFDlwr = value->rValue;
            mod->B3SOIFDlwrGiven = TRUE;
            break;
        case  B3SOIFD_MOD_LNFACTOR :
            mod->B3SOIFDlnfactor = value->rValue;
            mod->B3SOIFDlnfactorGiven = TRUE;
            break;
        case  B3SOIFD_MOD_LDWG :
            mod->B3SOIFDldwg = value->rValue;
            mod->B3SOIFDldwgGiven = TRUE;
            break;
        case  B3SOIFD_MOD_LDWB :
            mod->B3SOIFDldwb = value->rValue;
            mod->B3SOIFDldwbGiven = TRUE;
            break;
        case B3SOIFD_MOD_LVOFF:
            mod->B3SOIFDlvoff = value->rValue;
            mod->B3SOIFDlvoffGiven = TRUE;
            break;
        case B3SOIFD_MOD_LETA0:
            mod->B3SOIFDleta0 = value->rValue;
            mod->B3SOIFDleta0Given = TRUE;
            break;                 
        case B3SOIFD_MOD_LETAB:
            mod->B3SOIFDletab = value->rValue;
            mod->B3SOIFDletabGiven = TRUE;
            break;                 
        case  B3SOIFD_MOD_LDSUB:             
            mod->B3SOIFDldsub = value->rValue;
            mod->B3SOIFDldsubGiven = TRUE;
            break;
        case  B3SOIFD_MOD_LCIT :
            mod->B3SOIFDlcit = value->rValue;
            mod->B3SOIFDlcitGiven = TRUE;
            break;
        case  B3SOIFD_MOD_LCDSC :
            mod->B3SOIFDlcdsc = value->rValue;
            mod->B3SOIFDlcdscGiven = TRUE;
            break;
        case  B3SOIFD_MOD_LCDSCB :
            mod->B3SOIFDlcdscb = value->rValue;
            mod->B3SOIFDlcdscbGiven = TRUE;
            break;
        case  B3SOIFD_MOD_LCDSCD :
            mod->B3SOIFDlcdscd = value->rValue;
            mod->B3SOIFDlcdscdGiven = TRUE;
            break;
        case B3SOIFD_MOD_LPCLM:
            mod->B3SOIFDlpclm = value->rValue;
            mod->B3SOIFDlpclmGiven = TRUE;
            break;                 
        case B3SOIFD_MOD_LPDIBL1:
            mod->B3SOIFDlpdibl1 = value->rValue;
            mod->B3SOIFDlpdibl1Given = TRUE;
            break;                 
        case B3SOIFD_MOD_LPDIBL2:
            mod->B3SOIFDlpdibl2 = value->rValue;
            mod->B3SOIFDlpdibl2Given = TRUE;
            break;                 
        case B3SOIFD_MOD_LPDIBLB:
            mod->B3SOIFDlpdiblb = value->rValue;
            mod->B3SOIFDlpdiblbGiven = TRUE;
            break;                 
        case  B3SOIFD_MOD_LDROUT:             
            mod->B3SOIFDldrout = value->rValue;
            mod->B3SOIFDldroutGiven = TRUE;
            break;
        case B3SOIFD_MOD_LPVAG:
            mod->B3SOIFDlpvag = value->rValue;
            mod->B3SOIFDlpvagGiven = TRUE;
            break;                 
        case  B3SOIFD_MOD_LDELTA :
            mod->B3SOIFDldelta = value->rValue;
            mod->B3SOIFDldeltaGiven = TRUE;
            break;
        case  B3SOIFD_MOD_LAII :
            mod->B3SOIFDlaii = value->rValue;
            mod->B3SOIFDlaiiGiven = TRUE;
            break;
        case  B3SOIFD_MOD_LBII :
            mod->B3SOIFDlbii = value->rValue;
            mod->B3SOIFDlbiiGiven = TRUE;
            break;
        case  B3SOIFD_MOD_LCII :
            mod->B3SOIFDlcii = value->rValue;
            mod->B3SOIFDlciiGiven = TRUE;
            break;
        case  B3SOIFD_MOD_LDII :
            mod->B3SOIFDldii = value->rValue;
            mod->B3SOIFDldiiGiven = TRUE;
            break;
        case  B3SOIFD_MOD_LALPHA0 :
            mod->B3SOIFDlalpha0 = value->rValue;
            mod->B3SOIFDlalpha0Given = TRUE;
            break;
        case  B3SOIFD_MOD_LALPHA1 :
            mod->B3SOIFDlalpha1 = value->rValue;
            mod->B3SOIFDlalpha1Given = TRUE;
            break;
        case  B3SOIFD_MOD_LBETA0 :
            mod->B3SOIFDlbeta0 = value->rValue;
            mod->B3SOIFDlbeta0Given = TRUE;
            break;
        case  B3SOIFD_MOD_LAGIDL :
            mod->B3SOIFDlagidl = value->rValue;
            mod->B3SOIFDlagidlGiven = TRUE;
            break;
        case  B3SOIFD_MOD_LBGIDL :
            mod->B3SOIFDlbgidl = value->rValue;
            mod->B3SOIFDlbgidlGiven = TRUE;
            break;
        case  B3SOIFD_MOD_LNGIDL :
            mod->B3SOIFDlngidl = value->rValue;
            mod->B3SOIFDlngidlGiven = TRUE;
            break;
        case  B3SOIFD_MOD_LNTUN :
            mod->B3SOIFDlntun = value->rValue;
            mod->B3SOIFDlntunGiven = TRUE;
            break;
        case  B3SOIFD_MOD_LNDIODE :
            mod->B3SOIFDlndiode = value->rValue;
            mod->B3SOIFDlndiodeGiven = TRUE;
            break;
        case  B3SOIFD_MOD_LISBJT :
            mod->B3SOIFDlisbjt = value->rValue;
            mod->B3SOIFDlisbjtGiven = TRUE;
            break;
        case  B3SOIFD_MOD_LISDIF :
            mod->B3SOIFDlisdif = value->rValue;
            mod->B3SOIFDlisdifGiven = TRUE;
            break;
        case  B3SOIFD_MOD_LISREC :
            mod->B3SOIFDlisrec = value->rValue;
            mod->B3SOIFDlisrecGiven = TRUE;
            break;
        case  B3SOIFD_MOD_LISTUN :
            mod->B3SOIFDlistun = value->rValue;
            mod->B3SOIFDlistunGiven = TRUE;
            break;
        case  B3SOIFD_MOD_LEDL :
            mod->B3SOIFDledl = value->rValue;
            mod->B3SOIFDledlGiven = TRUE;
            break;
        case  B3SOIFD_MOD_LKBJT1 :
            mod->B3SOIFDlkbjt1 = value->rValue;
            mod->B3SOIFDlkbjt1Given = TRUE;
            break;
	/* CV Model */
        case  B3SOIFD_MOD_LVSDFB :
            mod->B3SOIFDlvsdfb = value->rValue;
            mod->B3SOIFDlvsdfbGiven = TRUE;
            break;
        case  B3SOIFD_MOD_LVSDTH :
            mod->B3SOIFDlvsdth = value->rValue;
            mod->B3SOIFDlvsdthGiven = TRUE;
            break;
        /* Width Dependence */
        case B3SOIFD_MOD_WNPEAK:
            mod->B3SOIFDwnpeak = value->rValue;
            mod->B3SOIFDwnpeakGiven = TRUE;
            break;
        case B3SOIFD_MOD_WNSUB:
            mod->B3SOIFDwnsub = value->rValue;
            mod->B3SOIFDwnsubGiven = TRUE;
            break;
        case B3SOIFD_MOD_WNGATE:
            mod->B3SOIFDwngate = value->rValue;
            mod->B3SOIFDwngateGiven = TRUE;
            break;
        case B3SOIFD_MOD_WVTH0:
            mod->B3SOIFDwvth0 = value->rValue;
            mod->B3SOIFDwvth0Given = TRUE;
            break;
        case  B3SOIFD_MOD_WK1:
            mod->B3SOIFDwk1 = value->rValue;
            mod->B3SOIFDwk1Given = TRUE;
            break;
        case  B3SOIFD_MOD_WK2:
            mod->B3SOIFDwk2 = value->rValue;
            mod->B3SOIFDwk2Given = TRUE;
            break;
        case  B3SOIFD_MOD_WK3:
            mod->B3SOIFDwk3 = value->rValue;
            mod->B3SOIFDwk3Given = TRUE;
            break;
        case  B3SOIFD_MOD_WK3B:
            mod->B3SOIFDwk3b = value->rValue;
            mod->B3SOIFDwk3bGiven = TRUE;
            break;
        case  B3SOIFD_MOD_WVBSA:
            mod->B3SOIFDwvbsa = value->rValue;
            mod->B3SOIFDwvbsaGiven = TRUE;
            break;
        case  B3SOIFD_MOD_WDELP:
            mod->B3SOIFDwdelp = value->rValue;
            mod->B3SOIFDwdelpGiven = TRUE;
            break;
        case  B3SOIFD_MOD_WKB1 :
            mod->B3SOIFDwkb1 = value->rValue;
            mod->B3SOIFDwkb1Given = TRUE;
            break;
        case  B3SOIFD_MOD_WKB3 :
            mod->B3SOIFDwkb3 = value->rValue;
            mod->B3SOIFDwkb3Given = TRUE;
            break;
        case  B3SOIFD_MOD_WDVBD0 :
            mod->B3SOIFDwdvbd0 = value->rValue;
            mod->B3SOIFDwdvbd0Given = TRUE;
            break;
        case  B3SOIFD_MOD_WDVBD1 :
            mod->B3SOIFDwdvbd1 = value->rValue;
            mod->B3SOIFDwdvbd1Given = TRUE;
            break;
        case  B3SOIFD_MOD_WW0:
            mod->B3SOIFDww0 = value->rValue;
            mod->B3SOIFDww0Given = TRUE;
            break;
        case  B3SOIFD_MOD_WNLX:
            mod->B3SOIFDwnlx = value->rValue;
            mod->B3SOIFDwnlxGiven = TRUE;
            break;
        case  B3SOIFD_MOD_WDVT0:               
            mod->B3SOIFDwdvt0 = value->rValue;
            mod->B3SOIFDwdvt0Given = TRUE;
            break;
        case  B3SOIFD_MOD_WDVT1:             
            mod->B3SOIFDwdvt1 = value->rValue;
            mod->B3SOIFDwdvt1Given = TRUE;
            break;
        case  B3SOIFD_MOD_WDVT2:             
            mod->B3SOIFDwdvt2 = value->rValue;
            mod->B3SOIFDwdvt2Given = TRUE;
            break;
        case  B3SOIFD_MOD_WDVT0W:               
            mod->B3SOIFDwdvt0w = value->rValue;
            mod->B3SOIFDwdvt0wGiven = TRUE;
            break;
        case  B3SOIFD_MOD_WDVT1W:             
            mod->B3SOIFDwdvt1w = value->rValue;
            mod->B3SOIFDwdvt1wGiven = TRUE;
            break;
        case  B3SOIFD_MOD_WDVT2W:             
            mod->B3SOIFDwdvt2w = value->rValue;
            mod->B3SOIFDwdvt2wGiven = TRUE;
            break;
        case  B3SOIFD_MOD_WU0 :
            mod->B3SOIFDwu0 = value->rValue;
            mod->B3SOIFDwu0Given = TRUE;
            break;
        case B3SOIFD_MOD_WUA:
            mod->B3SOIFDwua = value->rValue;
            mod->B3SOIFDwuaGiven = TRUE;
            break;
        case B3SOIFD_MOD_WUB:
            mod->B3SOIFDwub = value->rValue;
            mod->B3SOIFDwubGiven = TRUE;
            break;
        case B3SOIFD_MOD_WUC:
            mod->B3SOIFDwuc = value->rValue;
            mod->B3SOIFDwucGiven = TRUE;
            break;
        case B3SOIFD_MOD_WVSAT:
            mod->B3SOIFDwvsat = value->rValue;
            mod->B3SOIFDwvsatGiven = TRUE;
            break;
        case B3SOIFD_MOD_WA0:
            mod->B3SOIFDwa0 = value->rValue;
            mod->B3SOIFDwa0Given = TRUE;
            break;
        case B3SOIFD_MOD_WAGS:
            mod->B3SOIFDwags= value->rValue;
            mod->B3SOIFDwagsGiven = TRUE;
            break;
        case  B3SOIFD_MOD_WB0 :
            mod->B3SOIFDwb0 = value->rValue;
            mod->B3SOIFDwb0Given = TRUE;
            break;
        case  B3SOIFD_MOD_WB1 :
            mod->B3SOIFDwb1 = value->rValue;
            mod->B3SOIFDwb1Given = TRUE;
            break;
        case B3SOIFD_MOD_WKETA:
            mod->B3SOIFDwketa = value->rValue;
            mod->B3SOIFDwketaGiven = TRUE;
            break;    
        case B3SOIFD_MOD_WABP:
            mod->B3SOIFDwabp = value->rValue;
            mod->B3SOIFDwabpGiven = TRUE;
            break;    
        case B3SOIFD_MOD_WMXC:
            mod->B3SOIFDwmxc = value->rValue;
            mod->B3SOIFDwmxcGiven = TRUE;
            break;    
        case B3SOIFD_MOD_WADICE0:
            mod->B3SOIFDwadice0 = value->rValue;
            mod->B3SOIFDwadice0Given = TRUE;
            break;    
        case B3SOIFD_MOD_WA1:
            mod->B3SOIFDwa1 = value->rValue;
            mod->B3SOIFDwa1Given = TRUE;
            break;
        case B3SOIFD_MOD_WA2:
            mod->B3SOIFDwa2 = value->rValue;
            mod->B3SOIFDwa2Given = TRUE;
            break;
        case B3SOIFD_MOD_WRDSW:
            mod->B3SOIFDwrdsw = value->rValue;
            mod->B3SOIFDwrdswGiven = TRUE;
            break;                     
        case B3SOIFD_MOD_WPRWB:
            mod->B3SOIFDwprwb = value->rValue;
            mod->B3SOIFDwprwbGiven = TRUE;
            break;                     
        case B3SOIFD_MOD_WPRWG:
            mod->B3SOIFDwprwg = value->rValue;
            mod->B3SOIFDwprwgGiven = TRUE;
            break;                     
        case  B3SOIFD_MOD_WWR :
            mod->B3SOIFDwwr = value->rValue;
            mod->B3SOIFDwwrGiven = TRUE;
            break;
        case  B3SOIFD_MOD_WNFACTOR :
            mod->B3SOIFDwnfactor = value->rValue;
            mod->B3SOIFDwnfactorGiven = TRUE;
            break;
        case  B3SOIFD_MOD_WDWG :
            mod->B3SOIFDwdwg = value->rValue;
            mod->B3SOIFDwdwgGiven = TRUE;
            break;
        case  B3SOIFD_MOD_WDWB :
            mod->B3SOIFDwdwb = value->rValue;
            mod->B3SOIFDwdwbGiven = TRUE;
            break;
        case B3SOIFD_MOD_WVOFF:
            mod->B3SOIFDwvoff = value->rValue;
            mod->B3SOIFDwvoffGiven = TRUE;
            break;
        case B3SOIFD_MOD_WETA0:
            mod->B3SOIFDweta0 = value->rValue;
            mod->B3SOIFDweta0Given = TRUE;
            break;                 
        case B3SOIFD_MOD_WETAB:
            mod->B3SOIFDwetab = value->rValue;
            mod->B3SOIFDwetabGiven = TRUE;
            break;                 
        case  B3SOIFD_MOD_WDSUB:             
            mod->B3SOIFDwdsub = value->rValue;
            mod->B3SOIFDwdsubGiven = TRUE;
            break;
        case  B3SOIFD_MOD_WCIT :
            mod->B3SOIFDwcit = value->rValue;
            mod->B3SOIFDwcitGiven = TRUE;
            break;
        case  B3SOIFD_MOD_WCDSC :
            mod->B3SOIFDwcdsc = value->rValue;
            mod->B3SOIFDwcdscGiven = TRUE;
            break;
        case  B3SOIFD_MOD_WCDSCB :
            mod->B3SOIFDwcdscb = value->rValue;
            mod->B3SOIFDwcdscbGiven = TRUE;
            break;
        case  B3SOIFD_MOD_WCDSCD :
            mod->B3SOIFDwcdscd = value->rValue;
            mod->B3SOIFDwcdscdGiven = TRUE;
            break;
        case B3SOIFD_MOD_WPCLM:
            mod->B3SOIFDwpclm = value->rValue;
            mod->B3SOIFDwpclmGiven = TRUE;
            break;                 
        case B3SOIFD_MOD_WPDIBL1:
            mod->B3SOIFDwpdibl1 = value->rValue;
            mod->B3SOIFDwpdibl1Given = TRUE;
            break;                 
        case B3SOIFD_MOD_WPDIBL2:
            mod->B3SOIFDwpdibl2 = value->rValue;
            mod->B3SOIFDwpdibl2Given = TRUE;
            break;                 
        case B3SOIFD_MOD_WPDIBLB:
            mod->B3SOIFDwpdiblb = value->rValue;
            mod->B3SOIFDwpdiblbGiven = TRUE;
            break;                 
        case  B3SOIFD_MOD_WDROUT:             
            mod->B3SOIFDwdrout = value->rValue;
            mod->B3SOIFDwdroutGiven = TRUE;
            break;
        case B3SOIFD_MOD_WPVAG:
            mod->B3SOIFDwpvag = value->rValue;
            mod->B3SOIFDwpvagGiven = TRUE;
            break;                 
        case  B3SOIFD_MOD_WDELTA :
            mod->B3SOIFDwdelta = value->rValue;
            mod->B3SOIFDwdeltaGiven = TRUE;
            break;
        case  B3SOIFD_MOD_WAII :
            mod->B3SOIFDwaii = value->rValue;
            mod->B3SOIFDwaiiGiven = TRUE;
            break;
        case  B3SOIFD_MOD_WBII :
            mod->B3SOIFDwbii = value->rValue;
            mod->B3SOIFDwbiiGiven = TRUE;
            break;
        case  B3SOIFD_MOD_WCII :
            mod->B3SOIFDwcii = value->rValue;
            mod->B3SOIFDwciiGiven = TRUE;
            break;
        case  B3SOIFD_MOD_WDII :
            mod->B3SOIFDwdii = value->rValue;
            mod->B3SOIFDwdiiGiven = TRUE;
            break;
        case  B3SOIFD_MOD_WALPHA0 :
            mod->B3SOIFDwalpha0 = value->rValue;
            mod->B3SOIFDwalpha0Given = TRUE;
            break;
        case  B3SOIFD_MOD_WALPHA1 :
            mod->B3SOIFDwalpha1 = value->rValue;
            mod->B3SOIFDwalpha1Given = TRUE;
            break;
        case  B3SOIFD_MOD_WBETA0 :
            mod->B3SOIFDwbeta0 = value->rValue;
            mod->B3SOIFDwbeta0Given = TRUE;
            break;
        case  B3SOIFD_MOD_WAGIDL :
            mod->B3SOIFDwagidl = value->rValue;
            mod->B3SOIFDwagidlGiven = TRUE;
            break;
        case  B3SOIFD_MOD_WBGIDL :
            mod->B3SOIFDwbgidl = value->rValue;
            mod->B3SOIFDwbgidlGiven = TRUE;
            break;
        case  B3SOIFD_MOD_WNGIDL :
            mod->B3SOIFDwngidl = value->rValue;
            mod->B3SOIFDwngidlGiven = TRUE;
            break;
        case  B3SOIFD_MOD_WNTUN :
            mod->B3SOIFDwntun = value->rValue;
            mod->B3SOIFDwntunGiven = TRUE;
            break;
        case  B3SOIFD_MOD_WNDIODE :
            mod->B3SOIFDwndiode = value->rValue;
            mod->B3SOIFDwndiodeGiven = TRUE;
            break;
        case  B3SOIFD_MOD_WISBJT :
            mod->B3SOIFDwisbjt = value->rValue;
            mod->B3SOIFDwisbjtGiven = TRUE;
            break;
        case  B3SOIFD_MOD_WISDIF :
            mod->B3SOIFDwisdif = value->rValue;
            mod->B3SOIFDwisdifGiven = TRUE;
            break;
        case  B3SOIFD_MOD_WISREC :
            mod->B3SOIFDwisrec = value->rValue;
            mod->B3SOIFDwisrecGiven = TRUE;
            break;
        case  B3SOIFD_MOD_WISTUN :
            mod->B3SOIFDwistun = value->rValue;
            mod->B3SOIFDwistunGiven = TRUE;
            break;
        case  B3SOIFD_MOD_WEDL :
            mod->B3SOIFDwedl = value->rValue;
            mod->B3SOIFDwedlGiven = TRUE;
            break;
        case  B3SOIFD_MOD_WKBJT1 :
            mod->B3SOIFDwkbjt1 = value->rValue;
            mod->B3SOIFDwkbjt1Given = TRUE;
            break;
	/* CV Model */
        case  B3SOIFD_MOD_WVSDFB :
            mod->B3SOIFDwvsdfb = value->rValue;
            mod->B3SOIFDwvsdfbGiven = TRUE;
            break;
        case  B3SOIFD_MOD_WVSDTH :
            mod->B3SOIFDwvsdth = value->rValue;
            mod->B3SOIFDwvsdthGiven = TRUE;
            break;
        /* Cross-term Dependence */
        case B3SOIFD_MOD_PNPEAK:
            mod->B3SOIFDpnpeak = value->rValue;
            mod->B3SOIFDpnpeakGiven = TRUE;
            break;
        case B3SOIFD_MOD_PNSUB:
            mod->B3SOIFDpnsub = value->rValue;
            mod->B3SOIFDpnsubGiven = TRUE;
            break;
        case B3SOIFD_MOD_PNGATE:
            mod->B3SOIFDpngate = value->rValue;
            mod->B3SOIFDpngateGiven = TRUE;
            break;
        case B3SOIFD_MOD_PVTH0:
            mod->B3SOIFDpvth0 = value->rValue;
            mod->B3SOIFDpvth0Given = TRUE;
            break;
        case  B3SOIFD_MOD_PK1:
            mod->B3SOIFDpk1 = value->rValue;
            mod->B3SOIFDpk1Given = TRUE;
            break;
        case  B3SOIFD_MOD_PK2:
            mod->B3SOIFDpk2 = value->rValue;
            mod->B3SOIFDpk2Given = TRUE;
            break;
        case  B3SOIFD_MOD_PK3:
            mod->B3SOIFDpk3 = value->rValue;
            mod->B3SOIFDpk3Given = TRUE;
            break;
        case  B3SOIFD_MOD_PK3B:
            mod->B3SOIFDpk3b = value->rValue;
            mod->B3SOIFDpk3bGiven = TRUE;
            break;
        case  B3SOIFD_MOD_PVBSA:
            mod->B3SOIFDpvbsa = value->rValue;
            mod->B3SOIFDpvbsaGiven = TRUE;
            break;
        case  B3SOIFD_MOD_PDELP:
            mod->B3SOIFDpdelp = value->rValue;
            mod->B3SOIFDpdelpGiven = TRUE;
            break;
        case  B3SOIFD_MOD_PKB1 :
            mod->B3SOIFDpkb1 = value->rValue;
            mod->B3SOIFDpkb1Given = TRUE;
            break;
        case  B3SOIFD_MOD_PKB3 :
            mod->B3SOIFDpkb3 = value->rValue;
            mod->B3SOIFDpkb3Given = TRUE;
            break;
        case  B3SOIFD_MOD_PDVBD0 :
            mod->B3SOIFDpdvbd0 = value->rValue;
            mod->B3SOIFDpdvbd0Given = TRUE;
            break;
        case  B3SOIFD_MOD_PDVBD1 :
            mod->B3SOIFDpdvbd1 = value->rValue;
            mod->B3SOIFDpdvbd1Given = TRUE;
            break;
        case  B3SOIFD_MOD_PW0:
            mod->B3SOIFDpw0 = value->rValue;
            mod->B3SOIFDpw0Given = TRUE;
            break;
        case  B3SOIFD_MOD_PNLX:
            mod->B3SOIFDpnlx = value->rValue;
            mod->B3SOIFDpnlxGiven = TRUE;
            break;
        case  B3SOIFD_MOD_PDVT0:               
            mod->B3SOIFDpdvt0 = value->rValue;
            mod->B3SOIFDpdvt0Given = TRUE;
            break;
        case  B3SOIFD_MOD_PDVT1:             
            mod->B3SOIFDpdvt1 = value->rValue;
            mod->B3SOIFDpdvt1Given = TRUE;
            break;
        case  B3SOIFD_MOD_PDVT2:             
            mod->B3SOIFDpdvt2 = value->rValue;
            mod->B3SOIFDpdvt2Given = TRUE;
            break;
        case  B3SOIFD_MOD_PDVT0W:               
            mod->B3SOIFDpdvt0w = value->rValue;
            mod->B3SOIFDpdvt0wGiven = TRUE;
            break;
        case  B3SOIFD_MOD_PDVT1W:             
            mod->B3SOIFDpdvt1w = value->rValue;
            mod->B3SOIFDpdvt1wGiven = TRUE;
            break;
        case  B3SOIFD_MOD_PDVT2W:             
            mod->B3SOIFDpdvt2w = value->rValue;
            mod->B3SOIFDpdvt2wGiven = TRUE;
            break;
        case  B3SOIFD_MOD_PU0 :
            mod->B3SOIFDpu0 = value->rValue;
            mod->B3SOIFDpu0Given = TRUE;
            break;
        case B3SOIFD_MOD_PUA:
            mod->B3SOIFDpua = value->rValue;
            mod->B3SOIFDpuaGiven = TRUE;
            break;
        case B3SOIFD_MOD_PUB:
            mod->B3SOIFDpub = value->rValue;
            mod->B3SOIFDpubGiven = TRUE;
            break;
        case B3SOIFD_MOD_PUC:
            mod->B3SOIFDpuc = value->rValue;
            mod->B3SOIFDpucGiven = TRUE;
            break;
        case B3SOIFD_MOD_PVSAT:
            mod->B3SOIFDpvsat = value->rValue;
            mod->B3SOIFDpvsatGiven = TRUE;
            break;
        case B3SOIFD_MOD_PA0:
            mod->B3SOIFDpa0 = value->rValue;
            mod->B3SOIFDpa0Given = TRUE;
            break;
        case B3SOIFD_MOD_PAGS:
            mod->B3SOIFDpags= value->rValue;
            mod->B3SOIFDpagsGiven = TRUE;
            break;
        case  B3SOIFD_MOD_PB0 :
            mod->B3SOIFDpb0 = value->rValue;
            mod->B3SOIFDpb0Given = TRUE;
            break;
        case  B3SOIFD_MOD_PB1 :
            mod->B3SOIFDpb1 = value->rValue;
            mod->B3SOIFDpb1Given = TRUE;
            break;
        case B3SOIFD_MOD_PKETA:
            mod->B3SOIFDpketa = value->rValue;
            mod->B3SOIFDpketaGiven = TRUE;
            break;    
        case B3SOIFD_MOD_PABP:
            mod->B3SOIFDpabp = value->rValue;
            mod->B3SOIFDpabpGiven = TRUE;
            break;    
        case B3SOIFD_MOD_PMXC:
            mod->B3SOIFDpmxc = value->rValue;
            mod->B3SOIFDpmxcGiven = TRUE;
            break;    
        case B3SOIFD_MOD_PADICE0:
            mod->B3SOIFDpadice0 = value->rValue;
            mod->B3SOIFDpadice0Given = TRUE;
            break;    
        case B3SOIFD_MOD_PA1:
            mod->B3SOIFDpa1 = value->rValue;
            mod->B3SOIFDpa1Given = TRUE;
            break;
        case B3SOIFD_MOD_PA2:
            mod->B3SOIFDpa2 = value->rValue;
            mod->B3SOIFDpa2Given = TRUE;
            break;
        case B3SOIFD_MOD_PRDSW:
            mod->B3SOIFDprdsw = value->rValue;
            mod->B3SOIFDprdswGiven = TRUE;
            break;                     
        case B3SOIFD_MOD_PPRWB:
            mod->B3SOIFDpprwb = value->rValue;
            mod->B3SOIFDpprwbGiven = TRUE;
            break;                     
        case B3SOIFD_MOD_PPRWG:
            mod->B3SOIFDpprwg = value->rValue;
            mod->B3SOIFDpprwgGiven = TRUE;
            break;                     
        case  B3SOIFD_MOD_PWR :
            mod->B3SOIFDpwr = value->rValue;
            mod->B3SOIFDpwrGiven = TRUE;
            break;
        case  B3SOIFD_MOD_PNFACTOR :
            mod->B3SOIFDpnfactor = value->rValue;
            mod->B3SOIFDpnfactorGiven = TRUE;
            break;
        case  B3SOIFD_MOD_PDWG :
            mod->B3SOIFDpdwg = value->rValue;
            mod->B3SOIFDpdwgGiven = TRUE;
            break;
        case  B3SOIFD_MOD_PDWB :
            mod->B3SOIFDpdwb = value->rValue;
            mod->B3SOIFDpdwbGiven = TRUE;
            break;
        case B3SOIFD_MOD_PVOFF:
            mod->B3SOIFDpvoff = value->rValue;
            mod->B3SOIFDpvoffGiven = TRUE;
            break;
        case B3SOIFD_MOD_PETA0:
            mod->B3SOIFDpeta0 = value->rValue;
            mod->B3SOIFDpeta0Given = TRUE;
            break;                 
        case B3SOIFD_MOD_PETAB:
            mod->B3SOIFDpetab = value->rValue;
            mod->B3SOIFDpetabGiven = TRUE;
            break;                 
        case  B3SOIFD_MOD_PDSUB:             
            mod->B3SOIFDpdsub = value->rValue;
            mod->B3SOIFDpdsubGiven = TRUE;
            break;
        case  B3SOIFD_MOD_PCIT :
            mod->B3SOIFDpcit = value->rValue;
            mod->B3SOIFDpcitGiven = TRUE;
            break;
        case  B3SOIFD_MOD_PCDSC :
            mod->B3SOIFDpcdsc = value->rValue;
            mod->B3SOIFDpcdscGiven = TRUE;
            break;
        case  B3SOIFD_MOD_PCDSCB :
            mod->B3SOIFDpcdscb = value->rValue;
            mod->B3SOIFDpcdscbGiven = TRUE;
            break;
        case  B3SOIFD_MOD_PCDSCD :
            mod->B3SOIFDpcdscd = value->rValue;
            mod->B3SOIFDpcdscdGiven = TRUE;
            break;
        case B3SOIFD_MOD_PPCLM:
            mod->B3SOIFDppclm = value->rValue;
            mod->B3SOIFDppclmGiven = TRUE;
            break;                 
        case B3SOIFD_MOD_PPDIBL1:
            mod->B3SOIFDppdibl1 = value->rValue;
            mod->B3SOIFDppdibl1Given = TRUE;
            break;                 
        case B3SOIFD_MOD_PPDIBL2:
            mod->B3SOIFDppdibl2 = value->rValue;
            mod->B3SOIFDppdibl2Given = TRUE;
            break;                 
        case B3SOIFD_MOD_PPDIBLB:
            mod->B3SOIFDppdiblb = value->rValue;
            mod->B3SOIFDppdiblbGiven = TRUE;
            break;                 
        case  B3SOIFD_MOD_PDROUT:             
            mod->B3SOIFDpdrout = value->rValue;
            mod->B3SOIFDpdroutGiven = TRUE;
            break;
        case B3SOIFD_MOD_PPVAG:
            mod->B3SOIFDppvag = value->rValue;
            mod->B3SOIFDppvagGiven = TRUE;
            break;                 
        case  B3SOIFD_MOD_PDELTA :
            mod->B3SOIFDpdelta = value->rValue;
            mod->B3SOIFDpdeltaGiven = TRUE;
            break;
        case  B3SOIFD_MOD_PAII :
            mod->B3SOIFDpaii = value->rValue;
            mod->B3SOIFDpaiiGiven = TRUE;
            break;
        case  B3SOIFD_MOD_PBII :
            mod->B3SOIFDpbii = value->rValue;
            mod->B3SOIFDpbiiGiven = TRUE;
            break;
        case  B3SOIFD_MOD_PCII :
            mod->B3SOIFDpcii = value->rValue;
            mod->B3SOIFDpciiGiven = TRUE;
            break;
        case  B3SOIFD_MOD_PDII :
            mod->B3SOIFDpdii = value->rValue;
            mod->B3SOIFDpdiiGiven = TRUE;
            break;
        case  B3SOIFD_MOD_PALPHA0 :
            mod->B3SOIFDpalpha0 = value->rValue;
            mod->B3SOIFDpalpha0Given = TRUE;
            break;
        case  B3SOIFD_MOD_PALPHA1 :
            mod->B3SOIFDpalpha1 = value->rValue;
            mod->B3SOIFDpalpha1Given = TRUE;
            break;
        case  B3SOIFD_MOD_PBETA0 :
            mod->B3SOIFDpbeta0 = value->rValue;
            mod->B3SOIFDpbeta0Given = TRUE;
            break;
        case  B3SOIFD_MOD_PAGIDL :
            mod->B3SOIFDpagidl = value->rValue;
            mod->B3SOIFDpagidlGiven = TRUE;
            break;
        case  B3SOIFD_MOD_PBGIDL :
            mod->B3SOIFDpbgidl = value->rValue;
            mod->B3SOIFDpbgidlGiven = TRUE;
            break;
        case  B3SOIFD_MOD_PNGIDL :
            mod->B3SOIFDpngidl = value->rValue;
            mod->B3SOIFDpngidlGiven = TRUE;
            break;
        case  B3SOIFD_MOD_PNTUN :
            mod->B3SOIFDpntun = value->rValue;
            mod->B3SOIFDpntunGiven = TRUE;
            break;
        case  B3SOIFD_MOD_PNDIODE :
            mod->B3SOIFDpndiode = value->rValue;
            mod->B3SOIFDpndiodeGiven = TRUE;
            break;
        case  B3SOIFD_MOD_PISBJT :
            mod->B3SOIFDpisbjt = value->rValue;
            mod->B3SOIFDpisbjtGiven = TRUE;
            break;
        case  B3SOIFD_MOD_PISDIF :
            mod->B3SOIFDpisdif = value->rValue;
            mod->B3SOIFDpisdifGiven = TRUE;
            break;
        case  B3SOIFD_MOD_PISREC :
            mod->B3SOIFDpisrec = value->rValue;
            mod->B3SOIFDpisrecGiven = TRUE;
            break;
        case  B3SOIFD_MOD_PISTUN :
            mod->B3SOIFDpistun = value->rValue;
            mod->B3SOIFDpistunGiven = TRUE;
            break;
        case  B3SOIFD_MOD_PEDL :
            mod->B3SOIFDpedl = value->rValue;
            mod->B3SOIFDpedlGiven = TRUE;
            break;
        case  B3SOIFD_MOD_PKBJT1 :
            mod->B3SOIFDpkbjt1 = value->rValue;
            mod->B3SOIFDpkbjt1Given = TRUE;
            break;
	/* CV Model */
        case  B3SOIFD_MOD_PVSDFB :
            mod->B3SOIFDpvsdfb = value->rValue;
            mod->B3SOIFDpvsdfbGiven = TRUE;
            break;
        case  B3SOIFD_MOD_PVSDTH :
            mod->B3SOIFDpvsdth = value->rValue;
            mod->B3SOIFDpvsdthGiven = TRUE;
            break;
/* Added for binning - END */

        case  B3SOIFD_MOD_NMOS  :
            if(value->iValue) {
                mod->B3SOIFDtype = 1;
                mod->B3SOIFDtypeGiven = TRUE;
            }
            break;
        case  B3SOIFD_MOD_PMOS  :
            if(value->iValue) {
                mod->B3SOIFDtype = - 1;
                mod->B3SOIFDtypeGiven = TRUE;
            }
            break;
        default:
            return(E_BADPARM);
    }
    return(OK);
}


