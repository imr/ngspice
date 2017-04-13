/**** BSIM4.0.0, Released by Weidong Liu 3/24/2000 ****/

/**********
 * Copyright 2000 Regents of the University of California. All rights reserved.
 * File: b4mpar.c of BSIM4.0.0.
 * Authors: Weidong Liu, Kanyu M. Cao, Xiaodong Jin, Chenming Hu.
 * Project Director: Prof. Chenming Hu.
 **********/

#include "ngspice/ngspice.h"
#include "bsim4v0def.h"
#include "ngspice/ifsim.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"
#include "ngspice/const.h"

int
BSIM4v0mParam(
int param,
IFvalue *value,
GENmodel *inMod)
{
    BSIM4v0model *mod = (BSIM4v0model*)inMod;
    switch(param)
    {   case  BSIM4v0_MOD_MOBMOD :
            mod->BSIM4v0mobMod = value->iValue;
            mod->BSIM4v0mobModGiven = TRUE;
            break;
        case  BSIM4v0_MOD_BINUNIT :
            mod->BSIM4v0binUnit = value->iValue;
            mod->BSIM4v0binUnitGiven = TRUE;
            break;
        case  BSIM4v0_MOD_PARAMCHK :
            mod->BSIM4v0paramChk = value->iValue;
            mod->BSIM4v0paramChkGiven = TRUE;
            break;
        case  BSIM4v0_MOD_CAPMOD :
            mod->BSIM4v0capMod = value->iValue;
            mod->BSIM4v0capModGiven = TRUE;
            break;
        case  BSIM4v0_MOD_DIOMOD :
            mod->BSIM4v0dioMod = value->iValue;
            mod->BSIM4v0dioModGiven = TRUE;
            break;
        case  BSIM4v0_MOD_RDSMOD :
            mod->BSIM4v0rdsMod = value->iValue;
            mod->BSIM4v0rdsModGiven = TRUE;
            break;
        case  BSIM4v0_MOD_TRNQSMOD :
            mod->BSIM4v0trnqsMod = value->iValue;
            mod->BSIM4v0trnqsModGiven = TRUE;
            break;
        case  BSIM4v0_MOD_ACNQSMOD :
            mod->BSIM4v0acnqsMod = value->iValue;
            mod->BSIM4v0acnqsModGiven = TRUE;
            break;
        case  BSIM4v0_MOD_RBODYMOD :
            mod->BSIM4v0rbodyMod = value->iValue;
            mod->BSIM4v0rbodyModGiven = TRUE;
            break;
        case  BSIM4v0_MOD_RGATEMOD :
            mod->BSIM4v0rgateMod = value->iValue;
            mod->BSIM4v0rgateModGiven = TRUE;
            break;
        case  BSIM4v0_MOD_PERMOD :
            mod->BSIM4v0perMod = value->iValue;
            mod->BSIM4v0perModGiven = TRUE;
            break;
        case  BSIM4v0_MOD_GEOMOD :
            mod->BSIM4v0geoMod = value->iValue;
            mod->BSIM4v0geoModGiven = TRUE;
            break;
        case  BSIM4v0_MOD_FNOIMOD :
            mod->BSIM4v0fnoiMod = value->iValue;
            mod->BSIM4v0fnoiModGiven = TRUE;
            break;
        case  BSIM4v0_MOD_TNOIMOD :
            mod->BSIM4v0tnoiMod = value->iValue;
            mod->BSIM4v0tnoiModGiven = TRUE;
            break;
        case  BSIM4v0_MOD_IGCMOD :
            mod->BSIM4v0igcMod = value->iValue;
            mod->BSIM4v0igcModGiven = TRUE;
            break;
        case  BSIM4v0_MOD_IGBMOD :
            mod->BSIM4v0igbMod = value->iValue;
            mod->BSIM4v0igbModGiven = TRUE;
            break;
        case  BSIM4v0_MOD_VERSION :
            mod->BSIM4v0version = value->sValue;
            mod->BSIM4v0versionGiven = TRUE;
            break;
        case  BSIM4v0_MOD_TOXREF :
            mod->BSIM4v0toxref = value->rValue;
            mod->BSIM4v0toxrefGiven = TRUE;
            break;
        case  BSIM4v0_MOD_TOXE :
            mod->BSIM4v0toxe = value->rValue;
            mod->BSIM4v0toxeGiven = TRUE;
            break;
        case  BSIM4v0_MOD_TOXP :
            mod->BSIM4v0toxp = value->rValue;
            mod->BSIM4v0toxpGiven = TRUE;
            break;
        case  BSIM4v0_MOD_TOXM :
            mod->BSIM4v0toxm = value->rValue;
            mod->BSIM4v0toxmGiven = TRUE;
            break;
        case  BSIM4v0_MOD_DTOX :
            mod->BSIM4v0dtox = value->rValue;
            mod->BSIM4v0dtoxGiven = TRUE;
            break;
        case  BSIM4v0_MOD_EPSROX :
            mod->BSIM4v0epsrox = value->rValue;
            mod->BSIM4v0epsroxGiven = TRUE;
            break;

        case  BSIM4v0_MOD_CDSC :
            mod->BSIM4v0cdsc = value->rValue;
            mod->BSIM4v0cdscGiven = TRUE;
            break;
        case  BSIM4v0_MOD_CDSCB :
            mod->BSIM4v0cdscb = value->rValue;
            mod->BSIM4v0cdscbGiven = TRUE;
            break;

        case  BSIM4v0_MOD_CDSCD :
            mod->BSIM4v0cdscd = value->rValue;
            mod->BSIM4v0cdscdGiven = TRUE;
            break;

        case  BSIM4v0_MOD_CIT :
            mod->BSIM4v0cit = value->rValue;
            mod->BSIM4v0citGiven = TRUE;
            break;
        case  BSIM4v0_MOD_NFACTOR :
            mod->BSIM4v0nfactor = value->rValue;
            mod->BSIM4v0nfactorGiven = TRUE;
            break;
        case BSIM4v0_MOD_XJ:
            mod->BSIM4v0xj = value->rValue;
            mod->BSIM4v0xjGiven = TRUE;
            break;
        case BSIM4v0_MOD_VSAT:
            mod->BSIM4v0vsat = value->rValue;
            mod->BSIM4v0vsatGiven = TRUE;
            break;
        case BSIM4v0_MOD_A0:
            mod->BSIM4v0a0 = value->rValue;
            mod->BSIM4v0a0Given = TRUE;
            break;
        
        case BSIM4v0_MOD_AGS:
            mod->BSIM4v0ags= value->rValue;
            mod->BSIM4v0agsGiven = TRUE;
            break;
        
        case BSIM4v0_MOD_A1:
            mod->BSIM4v0a1 = value->rValue;
            mod->BSIM4v0a1Given = TRUE;
            break;
        case BSIM4v0_MOD_A2:
            mod->BSIM4v0a2 = value->rValue;
            mod->BSIM4v0a2Given = TRUE;
            break;
        case BSIM4v0_MOD_AT:
            mod->BSIM4v0at = value->rValue;
            mod->BSIM4v0atGiven = TRUE;
            break;
        case BSIM4v0_MOD_KETA:
            mod->BSIM4v0keta = value->rValue;
            mod->BSIM4v0ketaGiven = TRUE;
            break;    
        case BSIM4v0_MOD_NSUB:
            mod->BSIM4v0nsub = value->rValue;
            mod->BSIM4v0nsubGiven = TRUE;
            break;
        case BSIM4v0_MOD_NDEP:
            mod->BSIM4v0ndep = value->rValue;
            mod->BSIM4v0ndepGiven = TRUE;
	    if (mod->BSIM4v0ndep > 1.0e20)
		mod->BSIM4v0ndep *= 1.0e-6;
            break;
        case BSIM4v0_MOD_NSD:
            mod->BSIM4v0nsd = value->rValue;
            mod->BSIM4v0nsdGiven = TRUE;
            if (mod->BSIM4v0nsd > 1.0e23)
                mod->BSIM4v0nsd *= 1.0e-6;
            break;
        case BSIM4v0_MOD_NGATE:
            mod->BSIM4v0ngate = value->rValue;
            mod->BSIM4v0ngateGiven = TRUE;
	    if (mod->BSIM4v0ngate > 1.0e23)
		mod->BSIM4v0ngate *= 1.0e-6;
            break;
        case BSIM4v0_MOD_GAMMA1:
            mod->BSIM4v0gamma1 = value->rValue;
            mod->BSIM4v0gamma1Given = TRUE;
            break;
        case BSIM4v0_MOD_GAMMA2:
            mod->BSIM4v0gamma2 = value->rValue;
            mod->BSIM4v0gamma2Given = TRUE;
            break;
        case BSIM4v0_MOD_VBX:
            mod->BSIM4v0vbx = value->rValue;
            mod->BSIM4v0vbxGiven = TRUE;
            break;
        case BSIM4v0_MOD_VBM:
            mod->BSIM4v0vbm = value->rValue;
            mod->BSIM4v0vbmGiven = TRUE;
            break;
        case BSIM4v0_MOD_XT:
            mod->BSIM4v0xt = value->rValue;
            mod->BSIM4v0xtGiven = TRUE;
            break;
        case  BSIM4v0_MOD_K1:
            mod->BSIM4v0k1 = value->rValue;
            mod->BSIM4v0k1Given = TRUE;
            break;
        case  BSIM4v0_MOD_KT1:
            mod->BSIM4v0kt1 = value->rValue;
            mod->BSIM4v0kt1Given = TRUE;
            break;
        case  BSIM4v0_MOD_KT1L:
            mod->BSIM4v0kt1l = value->rValue;
            mod->BSIM4v0kt1lGiven = TRUE;
            break;
        case  BSIM4v0_MOD_KT2:
            mod->BSIM4v0kt2 = value->rValue;
            mod->BSIM4v0kt2Given = TRUE;
            break;
        case  BSIM4v0_MOD_K2:
            mod->BSIM4v0k2 = value->rValue;
            mod->BSIM4v0k2Given = TRUE;
            break;
        case  BSIM4v0_MOD_K3:
            mod->BSIM4v0k3 = value->rValue;
            mod->BSIM4v0k3Given = TRUE;
            break;
        case  BSIM4v0_MOD_K3B:
            mod->BSIM4v0k3b = value->rValue;
            mod->BSIM4v0k3bGiven = TRUE;
            break;
        case  BSIM4v0_MOD_LPE0:
            mod->BSIM4v0lpe0 = value->rValue;
            mod->BSIM4v0lpe0Given = TRUE;
            break;
        case  BSIM4v0_MOD_LPEB:
            mod->BSIM4v0lpeb = value->rValue;
            mod->BSIM4v0lpebGiven = TRUE;
            break;
        case  BSIM4v0_MOD_DVTP0:
            mod->BSIM4v0dvtp0 = value->rValue;
            mod->BSIM4v0dvtp0Given = TRUE;
            break;
        case  BSIM4v0_MOD_DVTP1:
            mod->BSIM4v0dvtp1 = value->rValue;
            mod->BSIM4v0dvtp1Given = TRUE;
            break;
        case  BSIM4v0_MOD_W0:
            mod->BSIM4v0w0 = value->rValue;
            mod->BSIM4v0w0Given = TRUE;
            break;
        case  BSIM4v0_MOD_DVT0:               
            mod->BSIM4v0dvt0 = value->rValue;
            mod->BSIM4v0dvt0Given = TRUE;
            break;
        case  BSIM4v0_MOD_DVT1:             
            mod->BSIM4v0dvt1 = value->rValue;
            mod->BSIM4v0dvt1Given = TRUE;
            break;
        case  BSIM4v0_MOD_DVT2:             
            mod->BSIM4v0dvt2 = value->rValue;
            mod->BSIM4v0dvt2Given = TRUE;
            break;
        case  BSIM4v0_MOD_DVT0W:               
            mod->BSIM4v0dvt0w = value->rValue;
            mod->BSIM4v0dvt0wGiven = TRUE;
            break;
        case  BSIM4v0_MOD_DVT1W:             
            mod->BSIM4v0dvt1w = value->rValue;
            mod->BSIM4v0dvt1wGiven = TRUE;
            break;
        case  BSIM4v0_MOD_DVT2W:             
            mod->BSIM4v0dvt2w = value->rValue;
            mod->BSIM4v0dvt2wGiven = TRUE;
            break;
        case  BSIM4v0_MOD_DROUT:             
            mod->BSIM4v0drout = value->rValue;
            mod->BSIM4v0droutGiven = TRUE;
            break;
        case  BSIM4v0_MOD_DSUB:             
            mod->BSIM4v0dsub = value->rValue;
            mod->BSIM4v0dsubGiven = TRUE;
            break;
        case BSIM4v0_MOD_VTH0:
            mod->BSIM4v0vth0 = value->rValue;
            mod->BSIM4v0vth0Given = TRUE;
            break;
        case BSIM4v0_MOD_EU:
            mod->BSIM4v0eu = value->rValue;
            mod->BSIM4v0euGiven = TRUE;
            break;
        case BSIM4v0_MOD_UA:
            mod->BSIM4v0ua = value->rValue;
            mod->BSIM4v0uaGiven = TRUE;
            break;
        case BSIM4v0_MOD_UA1:
            mod->BSIM4v0ua1 = value->rValue;
            mod->BSIM4v0ua1Given = TRUE;
            break;
        case BSIM4v0_MOD_UB:
            mod->BSIM4v0ub = value->rValue;
            mod->BSIM4v0ubGiven = TRUE;
            break;
        case BSIM4v0_MOD_UB1:
            mod->BSIM4v0ub1 = value->rValue;
            mod->BSIM4v0ub1Given = TRUE;
            break;
        case BSIM4v0_MOD_UC:
            mod->BSIM4v0uc = value->rValue;
            mod->BSIM4v0ucGiven = TRUE;
            break;
        case BSIM4v0_MOD_UC1:
            mod->BSIM4v0uc1 = value->rValue;
            mod->BSIM4v0uc1Given = TRUE;
            break;
        case  BSIM4v0_MOD_U0 :
            mod->BSIM4v0u0 = value->rValue;
            mod->BSIM4v0u0Given = TRUE;
            break;
        case  BSIM4v0_MOD_UTE :
            mod->BSIM4v0ute = value->rValue;
            mod->BSIM4v0uteGiven = TRUE;
            break;
        case BSIM4v0_MOD_VOFF:
            mod->BSIM4v0voff = value->rValue;
            mod->BSIM4v0voffGiven = TRUE;
            break;
        case BSIM4v0_MOD_VOFFL:
            mod->BSIM4v0voffl = value->rValue;
            mod->BSIM4v0vofflGiven = TRUE;
            break;
        case BSIM4v0_MOD_MINV:
            mod->BSIM4v0minv = value->rValue;
            mod->BSIM4v0minvGiven = TRUE;
            break;
        case BSIM4v0_MOD_FPROUT:
            mod->BSIM4v0fprout = value->rValue;
            mod->BSIM4v0fproutGiven = TRUE;
            break;
        case BSIM4v0_MOD_PDITS:
            mod->BSIM4v0pdits = value->rValue;
            mod->BSIM4v0pditsGiven = TRUE;
            break;
        case BSIM4v0_MOD_PDITSD:
            mod->BSIM4v0pditsd = value->rValue;
            mod->BSIM4v0pditsdGiven = TRUE;
            break;
        case BSIM4v0_MOD_PDITSL:
            mod->BSIM4v0pditsl = value->rValue;
            mod->BSIM4v0pditslGiven = TRUE;
            break;
        case  BSIM4v0_MOD_DELTA :
            mod->BSIM4v0delta = value->rValue;
            mod->BSIM4v0deltaGiven = TRUE;
            break;
        case BSIM4v0_MOD_RDSW:
            mod->BSIM4v0rdsw = value->rValue;
            mod->BSIM4v0rdswGiven = TRUE;
            break;                     
        case BSIM4v0_MOD_RDSWMIN:
            mod->BSIM4v0rdswmin = value->rValue;
            mod->BSIM4v0rdswminGiven = TRUE;
            break;
        case BSIM4v0_MOD_RDWMIN:
            mod->BSIM4v0rdwmin = value->rValue;
            mod->BSIM4v0rdwminGiven = TRUE;
            break;
        case BSIM4v0_MOD_RSWMIN:
            mod->BSIM4v0rswmin = value->rValue;
            mod->BSIM4v0rswminGiven = TRUE;
            break;
        case BSIM4v0_MOD_RDW:
            mod->BSIM4v0rdw = value->rValue;
            mod->BSIM4v0rdwGiven = TRUE;
            break;
        case BSIM4v0_MOD_RSW:
            mod->BSIM4v0rsw = value->rValue;
            mod->BSIM4v0rswGiven = TRUE;
            break;
        case BSIM4v0_MOD_PRWG:
            mod->BSIM4v0prwg = value->rValue;
            mod->BSIM4v0prwgGiven = TRUE;
            break;                     
        case BSIM4v0_MOD_PRWB:
            mod->BSIM4v0prwb = value->rValue;
            mod->BSIM4v0prwbGiven = TRUE;
            break;                     
        case BSIM4v0_MOD_PRT:
            mod->BSIM4v0prt = value->rValue;
            mod->BSIM4v0prtGiven = TRUE;
            break;                     
        case BSIM4v0_MOD_ETA0:
            mod->BSIM4v0eta0 = value->rValue;
            mod->BSIM4v0eta0Given = TRUE;
            break;                 
        case BSIM4v0_MOD_ETAB:
            mod->BSIM4v0etab = value->rValue;
            mod->BSIM4v0etabGiven = TRUE;
            break;                 
        case BSIM4v0_MOD_PCLM:
            mod->BSIM4v0pclm = value->rValue;
            mod->BSIM4v0pclmGiven = TRUE;
            break;                 
        case BSIM4v0_MOD_PDIBL1:
            mod->BSIM4v0pdibl1 = value->rValue;
            mod->BSIM4v0pdibl1Given = TRUE;
            break;                 
        case BSIM4v0_MOD_PDIBL2:
            mod->BSIM4v0pdibl2 = value->rValue;
            mod->BSIM4v0pdibl2Given = TRUE;
            break;                 
        case BSIM4v0_MOD_PDIBLB:
            mod->BSIM4v0pdiblb = value->rValue;
            mod->BSIM4v0pdiblbGiven = TRUE;
            break;                 
        case BSIM4v0_MOD_PSCBE1:
            mod->BSIM4v0pscbe1 = value->rValue;
            mod->BSIM4v0pscbe1Given = TRUE;
            break;                 
        case BSIM4v0_MOD_PSCBE2:
            mod->BSIM4v0pscbe2 = value->rValue;
            mod->BSIM4v0pscbe2Given = TRUE;
            break;                 
        case BSIM4v0_MOD_PVAG:
            mod->BSIM4v0pvag = value->rValue;
            mod->BSIM4v0pvagGiven = TRUE;
            break;                 
        case  BSIM4v0_MOD_WR :
            mod->BSIM4v0wr = value->rValue;
            mod->BSIM4v0wrGiven = TRUE;
            break;
        case  BSIM4v0_MOD_DWG :
            mod->BSIM4v0dwg = value->rValue;
            mod->BSIM4v0dwgGiven = TRUE;
            break;
        case  BSIM4v0_MOD_DWB :
            mod->BSIM4v0dwb = value->rValue;
            mod->BSIM4v0dwbGiven = TRUE;
            break;
        case  BSIM4v0_MOD_B0 :
            mod->BSIM4v0b0 = value->rValue;
            mod->BSIM4v0b0Given = TRUE;
            break;
        case  BSIM4v0_MOD_B1 :
            mod->BSIM4v0b1 = value->rValue;
            mod->BSIM4v0b1Given = TRUE;
            break;
        case  BSIM4v0_MOD_ALPHA0 :
            mod->BSIM4v0alpha0 = value->rValue;
            mod->BSIM4v0alpha0Given = TRUE;
            break;
        case  BSIM4v0_MOD_ALPHA1 :
            mod->BSIM4v0alpha1 = value->rValue;
            mod->BSIM4v0alpha1Given = TRUE;
            break;
        case  BSIM4v0_MOD_AGIDL :
            mod->BSIM4v0agidl = value->rValue;
            mod->BSIM4v0agidlGiven = TRUE;
            break;
        case  BSIM4v0_MOD_BGIDL :
            mod->BSIM4v0bgidl = value->rValue;
            mod->BSIM4v0bgidlGiven = TRUE;
            break;
        case  BSIM4v0_MOD_CGIDL :
            mod->BSIM4v0cgidl = value->rValue;
            mod->BSIM4v0cgidlGiven = TRUE;
            break;
        case  BSIM4v0_MOD_PHIN :
            mod->BSIM4v0phin = value->rValue;
            mod->BSIM4v0phinGiven = TRUE;
            break;
        case  BSIM4v0_MOD_EGIDL :
            mod->BSIM4v0egidl = value->rValue;
            mod->BSIM4v0egidlGiven = TRUE;
            break;
        case  BSIM4v0_MOD_AIGC :
            mod->BSIM4v0aigc = value->rValue;
            mod->BSIM4v0aigcGiven = TRUE;
            break;
        case  BSIM4v0_MOD_BIGC :
            mod->BSIM4v0bigc = value->rValue;
            mod->BSIM4v0bigcGiven = TRUE;
            break;
        case  BSIM4v0_MOD_CIGC :
            mod->BSIM4v0cigc = value->rValue;
            mod->BSIM4v0cigcGiven = TRUE;
            break;
        case  BSIM4v0_MOD_AIGSD :
            mod->BSIM4v0aigsd = value->rValue;
            mod->BSIM4v0aigsdGiven = TRUE;
            break;
        case  BSIM4v0_MOD_BIGSD :
            mod->BSIM4v0bigsd = value->rValue;
            mod->BSIM4v0bigsdGiven = TRUE;
            break;
        case  BSIM4v0_MOD_CIGSD :
            mod->BSIM4v0cigsd = value->rValue;
            mod->BSIM4v0cigsdGiven = TRUE;
            break;
        case  BSIM4v0_MOD_AIGBACC :
            mod->BSIM4v0aigbacc = value->rValue;
            mod->BSIM4v0aigbaccGiven = TRUE;
            break;
        case  BSIM4v0_MOD_BIGBACC :
            mod->BSIM4v0bigbacc = value->rValue;
            mod->BSIM4v0bigbaccGiven = TRUE;
            break;
        case  BSIM4v0_MOD_CIGBACC :
            mod->BSIM4v0cigbacc = value->rValue;
            mod->BSIM4v0cigbaccGiven = TRUE;
            break;
        case  BSIM4v0_MOD_AIGBINV :
            mod->BSIM4v0aigbinv = value->rValue;
            mod->BSIM4v0aigbinvGiven = TRUE;
            break;
        case  BSIM4v0_MOD_BIGBINV :
            mod->BSIM4v0bigbinv = value->rValue;
            mod->BSIM4v0bigbinvGiven = TRUE;
            break;
        case  BSIM4v0_MOD_CIGBINV :
            mod->BSIM4v0cigbinv = value->rValue;
            mod->BSIM4v0cigbinvGiven = TRUE;
            break;
        case  BSIM4v0_MOD_NIGC :
            mod->BSIM4v0nigc = value->rValue;
            mod->BSIM4v0nigcGiven = TRUE;
            break;
        case  BSIM4v0_MOD_NIGBINV :
            mod->BSIM4v0nigbinv = value->rValue;
            mod->BSIM4v0nigbinvGiven = TRUE;
            break;
        case  BSIM4v0_MOD_NIGBACC :
            mod->BSIM4v0nigbacc = value->rValue;
            mod->BSIM4v0nigbaccGiven = TRUE;
            break;
        case  BSIM4v0_MOD_NTOX :
            mod->BSIM4v0ntox = value->rValue;
            mod->BSIM4v0ntoxGiven = TRUE;
            break;
        case  BSIM4v0_MOD_EIGBINV :
            mod->BSIM4v0eigbinv = value->rValue;
            mod->BSIM4v0eigbinvGiven = TRUE;
            break;
        case  BSIM4v0_MOD_PIGCD :
            mod->BSIM4v0pigcd = value->rValue;
            mod->BSIM4v0pigcdGiven = TRUE;
            break;
        case  BSIM4v0_MOD_POXEDGE :
            mod->BSIM4v0poxedge = value->rValue;
            mod->BSIM4v0poxedgeGiven = TRUE;
            break;
        case  BSIM4v0_MOD_XRCRG1 :
            mod->BSIM4v0xrcrg1 = value->rValue;
            mod->BSIM4v0xrcrg1Given = TRUE;
            break;
        case  BSIM4v0_MOD_TNOIA :
            mod->BSIM4v0tnoia = value->rValue;
            mod->BSIM4v0tnoiaGiven = TRUE;
            break;
        case  BSIM4v0_MOD_TNOIB :
            mod->BSIM4v0tnoib = value->rValue;
            mod->BSIM4v0tnoibGiven = TRUE;
            break;
        case  BSIM4v0_MOD_NTNOI :
            mod->BSIM4v0ntnoi = value->rValue;
            mod->BSIM4v0ntnoiGiven = TRUE;
            break;
        case  BSIM4v0_MOD_XRCRG2 :
            mod->BSIM4v0xrcrg2 = value->rValue;
            mod->BSIM4v0xrcrg2Given = TRUE;
            break;
        case  BSIM4v0_MOD_BETA0 :
            mod->BSIM4v0beta0 = value->rValue;
            mod->BSIM4v0beta0Given = TRUE;
            break;
        case  BSIM4v0_MOD_IJTHDFWD :
            mod->BSIM4v0ijthdfwd = value->rValue;
            mod->BSIM4v0ijthdfwdGiven = TRUE;
            break;
        case  BSIM4v0_MOD_IJTHSFWD :
            mod->BSIM4v0ijthsfwd = value->rValue;
            mod->BSIM4v0ijthsfwdGiven = TRUE;
            break;
        case  BSIM4v0_MOD_IJTHDREV :
            mod->BSIM4v0ijthdrev = value->rValue;
            mod->BSIM4v0ijthdrevGiven = TRUE;
            break;
        case  BSIM4v0_MOD_IJTHSREV :
            mod->BSIM4v0ijthsrev = value->rValue;
            mod->BSIM4v0ijthsrevGiven = TRUE;
            break;
        case  BSIM4v0_MOD_XJBVD :
            mod->BSIM4v0xjbvd = value->rValue;
            mod->BSIM4v0xjbvdGiven = TRUE;
            break;
        case  BSIM4v0_MOD_XJBVS :
            mod->BSIM4v0xjbvs = value->rValue;
            mod->BSIM4v0xjbvsGiven = TRUE;
            break;
        case  BSIM4v0_MOD_BVD :
            mod->BSIM4v0bvd = value->rValue;
            mod->BSIM4v0bvdGiven = TRUE;
            break;
        case  BSIM4v0_MOD_BVS :
            mod->BSIM4v0bvs = value->rValue;
            mod->BSIM4v0bvsGiven = TRUE;
            break;
        case  BSIM4v0_MOD_VFB :
            mod->BSIM4v0vfb = value->rValue;
            mod->BSIM4v0vfbGiven = TRUE;
            break;

        case  BSIM4v0_MOD_GBMIN :
            mod->BSIM4v0gbmin = value->rValue;
            mod->BSIM4v0gbminGiven = TRUE;
            break;
        case  BSIM4v0_MOD_RBDB :
            mod->BSIM4v0rbdb = value->rValue;
            mod->BSIM4v0rbdbGiven = TRUE;
            break;
        case  BSIM4v0_MOD_RBPB :
            mod->BSIM4v0rbpb = value->rValue;
            mod->BSIM4v0rbpbGiven = TRUE;
            break;
        case  BSIM4v0_MOD_RBSB :
            mod->BSIM4v0rbsb = value->rValue;
            mod->BSIM4v0rbsbGiven = TRUE;
            break;
        case  BSIM4v0_MOD_RBPS :
            mod->BSIM4v0rbps = value->rValue;
            mod->BSIM4v0rbpsGiven = TRUE;
            break;
        case  BSIM4v0_MOD_RBPD :
            mod->BSIM4v0rbpd = value->rValue;
            mod->BSIM4v0rbpdGiven = TRUE;
            break;

        case  BSIM4v0_MOD_CGSL :
            mod->BSIM4v0cgsl = value->rValue;
            mod->BSIM4v0cgslGiven = TRUE;
            break;
        case  BSIM4v0_MOD_CGDL :
            mod->BSIM4v0cgdl = value->rValue;
            mod->BSIM4v0cgdlGiven = TRUE;
            break;
        case  BSIM4v0_MOD_CKAPPAS :
            mod->BSIM4v0ckappas = value->rValue;
            mod->BSIM4v0ckappasGiven = TRUE;
            break;
        case  BSIM4v0_MOD_CKAPPAD :
            mod->BSIM4v0ckappad = value->rValue;
            mod->BSIM4v0ckappadGiven = TRUE;
            break;
        case  BSIM4v0_MOD_CF :
            mod->BSIM4v0cf = value->rValue;
            mod->BSIM4v0cfGiven = TRUE;
            break;
        case  BSIM4v0_MOD_CLC :
            mod->BSIM4v0clc = value->rValue;
            mod->BSIM4v0clcGiven = TRUE;
            break;
        case  BSIM4v0_MOD_CLE :
            mod->BSIM4v0cle = value->rValue;
            mod->BSIM4v0cleGiven = TRUE;
            break;
        case  BSIM4v0_MOD_DWC :
            mod->BSIM4v0dwc = value->rValue;
            mod->BSIM4v0dwcGiven = TRUE;
            break;
        case  BSIM4v0_MOD_DLC :
            mod->BSIM4v0dlc = value->rValue;
            mod->BSIM4v0dlcGiven = TRUE;
            break;
        case  BSIM4v0_MOD_DLCIG :
            mod->BSIM4v0dlcig = value->rValue;
            mod->BSIM4v0dlcigGiven = TRUE;
            break;
        case  BSIM4v0_MOD_DWJ :
            mod->BSIM4v0dwj = value->rValue;
            mod->BSIM4v0dwjGiven = TRUE;
            break;
        case  BSIM4v0_MOD_VFBCV :
            mod->BSIM4v0vfbcv = value->rValue;
            mod->BSIM4v0vfbcvGiven = TRUE;
            break;
        case  BSIM4v0_MOD_ACDE :
            mod->BSIM4v0acde = value->rValue;
            mod->BSIM4v0acdeGiven = TRUE;
            break;
        case  BSIM4v0_MOD_MOIN :
            mod->BSIM4v0moin = value->rValue;
            mod->BSIM4v0moinGiven = TRUE;
            break;
        case  BSIM4v0_MOD_NOFF :
            mod->BSIM4v0noff = value->rValue;
            mod->BSIM4v0noffGiven = TRUE;
            break;
        case  BSIM4v0_MOD_VOFFCV :
            mod->BSIM4v0voffcv = value->rValue;
            mod->BSIM4v0voffcvGiven = TRUE;
            break;
        case  BSIM4v0_MOD_DMCG :
            mod->BSIM4v0dmcg = value->rValue;
            mod->BSIM4v0dmcgGiven = TRUE;
            break;
        case  BSIM4v0_MOD_DMCI :
            mod->BSIM4v0dmci = value->rValue;
            mod->BSIM4v0dmciGiven = TRUE;
            break;
        case  BSIM4v0_MOD_DMDG :
            mod->BSIM4v0dmdg = value->rValue;
            mod->BSIM4v0dmdgGiven = TRUE;
            break;
        case  BSIM4v0_MOD_DMCGT :
            mod->BSIM4v0dmcgt = value->rValue;
            mod->BSIM4v0dmcgtGiven = TRUE;
            break;
        case  BSIM4v0_MOD_XGW :
            mod->BSIM4v0xgw = value->rValue;
            mod->BSIM4v0xgwGiven = TRUE;
            break;
        case  BSIM4v0_MOD_XGL :
            mod->BSIM4v0xgl = value->rValue;
            mod->BSIM4v0xglGiven = TRUE;
            break;
        case  BSIM4v0_MOD_RSHG :
            mod->BSIM4v0rshg = value->rValue;
            mod->BSIM4v0rshgGiven = TRUE;
            break;
        case  BSIM4v0_MOD_NGCON :
            mod->BSIM4v0ngcon = value->rValue;
            mod->BSIM4v0ngconGiven = TRUE;
            break;
        case  BSIM4v0_MOD_TCJ :
            mod->BSIM4v0tcj = value->rValue;
            mod->BSIM4v0tcjGiven = TRUE;
            break;
        case  BSIM4v0_MOD_TPB :
            mod->BSIM4v0tpb = value->rValue;
            mod->BSIM4v0tpbGiven = TRUE;
            break;
        case  BSIM4v0_MOD_TCJSW :
            mod->BSIM4v0tcjsw = value->rValue;
            mod->BSIM4v0tcjswGiven = TRUE;
            break;
        case  BSIM4v0_MOD_TPBSW :
            mod->BSIM4v0tpbsw = value->rValue;
            mod->BSIM4v0tpbswGiven = TRUE;
            break;
        case  BSIM4v0_MOD_TCJSWG :
            mod->BSIM4v0tcjswg = value->rValue;
            mod->BSIM4v0tcjswgGiven = TRUE;
            break;
        case  BSIM4v0_MOD_TPBSWG :
            mod->BSIM4v0tpbswg = value->rValue;
            mod->BSIM4v0tpbswgGiven = TRUE;
            break;

	/* Length dependence */
        case  BSIM4v0_MOD_LCDSC :
            mod->BSIM4v0lcdsc = value->rValue;
            mod->BSIM4v0lcdscGiven = TRUE;
            break;


        case  BSIM4v0_MOD_LCDSCB :
            mod->BSIM4v0lcdscb = value->rValue;
            mod->BSIM4v0lcdscbGiven = TRUE;
            break;
        case  BSIM4v0_MOD_LCDSCD :
            mod->BSIM4v0lcdscd = value->rValue;
            mod->BSIM4v0lcdscdGiven = TRUE;
            break;
        case  BSIM4v0_MOD_LCIT :
            mod->BSIM4v0lcit = value->rValue;
            mod->BSIM4v0lcitGiven = TRUE;
            break;
        case  BSIM4v0_MOD_LNFACTOR :
            mod->BSIM4v0lnfactor = value->rValue;
            mod->BSIM4v0lnfactorGiven = TRUE;
            break;
        case BSIM4v0_MOD_LXJ:
            mod->BSIM4v0lxj = value->rValue;
            mod->BSIM4v0lxjGiven = TRUE;
            break;
        case BSIM4v0_MOD_LVSAT:
            mod->BSIM4v0lvsat = value->rValue;
            mod->BSIM4v0lvsatGiven = TRUE;
            break;
        
        
        case BSIM4v0_MOD_LA0:
            mod->BSIM4v0la0 = value->rValue;
            mod->BSIM4v0la0Given = TRUE;
            break;
        case BSIM4v0_MOD_LAGS:
            mod->BSIM4v0lags = value->rValue;
            mod->BSIM4v0lagsGiven = TRUE;
            break;
        case BSIM4v0_MOD_LA1:
            mod->BSIM4v0la1 = value->rValue;
            mod->BSIM4v0la1Given = TRUE;
            break;
        case BSIM4v0_MOD_LA2:
            mod->BSIM4v0la2 = value->rValue;
            mod->BSIM4v0la2Given = TRUE;
            break;
        case BSIM4v0_MOD_LAT:
            mod->BSIM4v0lat = value->rValue;
            mod->BSIM4v0latGiven = TRUE;
            break;
        case BSIM4v0_MOD_LKETA:
            mod->BSIM4v0lketa = value->rValue;
            mod->BSIM4v0lketaGiven = TRUE;
            break;    
        case BSIM4v0_MOD_LNSUB:
            mod->BSIM4v0lnsub = value->rValue;
            mod->BSIM4v0lnsubGiven = TRUE;
            break;
        case BSIM4v0_MOD_LNDEP:
            mod->BSIM4v0lndep = value->rValue;
            mod->BSIM4v0lndepGiven = TRUE;
	    if (mod->BSIM4v0lndep > 1.0e20)
		mod->BSIM4v0lndep *= 1.0e-6;
            break;
        case BSIM4v0_MOD_LNSD:
            mod->BSIM4v0lnsd = value->rValue;
            mod->BSIM4v0lnsdGiven = TRUE;
            if (mod->BSIM4v0lnsd > 1.0e23)
                mod->BSIM4v0lnsd *= 1.0e-6;
            break;
        case BSIM4v0_MOD_LNGATE:
            mod->BSIM4v0lngate = value->rValue;
            mod->BSIM4v0lngateGiven = TRUE;
	    if (mod->BSIM4v0lngate > 1.0e23)
		mod->BSIM4v0lngate *= 1.0e-6;
            break;
        case BSIM4v0_MOD_LGAMMA1:
            mod->BSIM4v0lgamma1 = value->rValue;
            mod->BSIM4v0lgamma1Given = TRUE;
            break;
        case BSIM4v0_MOD_LGAMMA2:
            mod->BSIM4v0lgamma2 = value->rValue;
            mod->BSIM4v0lgamma2Given = TRUE;
            break;
        case BSIM4v0_MOD_LVBX:
            mod->BSIM4v0lvbx = value->rValue;
            mod->BSIM4v0lvbxGiven = TRUE;
            break;
        case BSIM4v0_MOD_LVBM:
            mod->BSIM4v0lvbm = value->rValue;
            mod->BSIM4v0lvbmGiven = TRUE;
            break;
        case BSIM4v0_MOD_LXT:
            mod->BSIM4v0lxt = value->rValue;
            mod->BSIM4v0lxtGiven = TRUE;
            break;
        case  BSIM4v0_MOD_LK1:
            mod->BSIM4v0lk1 = value->rValue;
            mod->BSIM4v0lk1Given = TRUE;
            break;
        case  BSIM4v0_MOD_LKT1:
            mod->BSIM4v0lkt1 = value->rValue;
            mod->BSIM4v0lkt1Given = TRUE;
            break;
        case  BSIM4v0_MOD_LKT1L:
            mod->BSIM4v0lkt1l = value->rValue;
            mod->BSIM4v0lkt1lGiven = TRUE;
            break;
        case  BSIM4v0_MOD_LKT2:
            mod->BSIM4v0lkt2 = value->rValue;
            mod->BSIM4v0lkt2Given = TRUE;
            break;
        case  BSIM4v0_MOD_LK2:
            mod->BSIM4v0lk2 = value->rValue;
            mod->BSIM4v0lk2Given = TRUE;
            break;
        case  BSIM4v0_MOD_LK3:
            mod->BSIM4v0lk3 = value->rValue;
            mod->BSIM4v0lk3Given = TRUE;
            break;
        case  BSIM4v0_MOD_LK3B:
            mod->BSIM4v0lk3b = value->rValue;
            mod->BSIM4v0lk3bGiven = TRUE;
            break;
        case  BSIM4v0_MOD_LLPE0:
            mod->BSIM4v0llpe0 = value->rValue;
            mod->BSIM4v0llpe0Given = TRUE;
            break;
        case  BSIM4v0_MOD_LLPEB:
            mod->BSIM4v0llpeb = value->rValue;
            mod->BSIM4v0llpebGiven = TRUE;
            break;
        case  BSIM4v0_MOD_LDVTP0:
            mod->BSIM4v0ldvtp0 = value->rValue;
            mod->BSIM4v0ldvtp0Given = TRUE;
            break;
        case  BSIM4v0_MOD_LDVTP1:
            mod->BSIM4v0ldvtp1 = value->rValue;
            mod->BSIM4v0ldvtp1Given = TRUE;
            break;
        case  BSIM4v0_MOD_LW0:
            mod->BSIM4v0lw0 = value->rValue;
            mod->BSIM4v0lw0Given = TRUE;
            break;
        case  BSIM4v0_MOD_LDVT0:               
            mod->BSIM4v0ldvt0 = value->rValue;
            mod->BSIM4v0ldvt0Given = TRUE;
            break;
        case  BSIM4v0_MOD_LDVT1:             
            mod->BSIM4v0ldvt1 = value->rValue;
            mod->BSIM4v0ldvt1Given = TRUE;
            break;
        case  BSIM4v0_MOD_LDVT2:             
            mod->BSIM4v0ldvt2 = value->rValue;
            mod->BSIM4v0ldvt2Given = TRUE;
            break;
        case  BSIM4v0_MOD_LDVT0W:               
            mod->BSIM4v0ldvt0w = value->rValue;
            mod->BSIM4v0ldvt0wGiven = TRUE;
            break;
        case  BSIM4v0_MOD_LDVT1W:             
            mod->BSIM4v0ldvt1w = value->rValue;
            mod->BSIM4v0ldvt1wGiven = TRUE;
            break;
        case  BSIM4v0_MOD_LDVT2W:             
            mod->BSIM4v0ldvt2w = value->rValue;
            mod->BSIM4v0ldvt2wGiven = TRUE;
            break;
        case  BSIM4v0_MOD_LDROUT:             
            mod->BSIM4v0ldrout = value->rValue;
            mod->BSIM4v0ldroutGiven = TRUE;
            break;
        case  BSIM4v0_MOD_LDSUB:             
            mod->BSIM4v0ldsub = value->rValue;
            mod->BSIM4v0ldsubGiven = TRUE;
            break;
        case BSIM4v0_MOD_LVTH0:
            mod->BSIM4v0lvth0 = value->rValue;
            mod->BSIM4v0lvth0Given = TRUE;
            break;
        case BSIM4v0_MOD_LUA:
            mod->BSIM4v0lua = value->rValue;
            mod->BSIM4v0luaGiven = TRUE;
            break;
        case BSIM4v0_MOD_LUA1:
            mod->BSIM4v0lua1 = value->rValue;
            mod->BSIM4v0lua1Given = TRUE;
            break;
        case BSIM4v0_MOD_LUB:
            mod->BSIM4v0lub = value->rValue;
            mod->BSIM4v0lubGiven = TRUE;
            break;
        case BSIM4v0_MOD_LUB1:
            mod->BSIM4v0lub1 = value->rValue;
            mod->BSIM4v0lub1Given = TRUE;
            break;
        case BSIM4v0_MOD_LUC:
            mod->BSIM4v0luc = value->rValue;
            mod->BSIM4v0lucGiven = TRUE;
            break;
        case BSIM4v0_MOD_LUC1:
            mod->BSIM4v0luc1 = value->rValue;
            mod->BSIM4v0luc1Given = TRUE;
            break;
        case  BSIM4v0_MOD_LU0 :
            mod->BSIM4v0lu0 = value->rValue;
            mod->BSIM4v0lu0Given = TRUE;
            break;
        case  BSIM4v0_MOD_LUTE :
            mod->BSIM4v0lute = value->rValue;
            mod->BSIM4v0luteGiven = TRUE;
            break;
        case BSIM4v0_MOD_LVOFF:
            mod->BSIM4v0lvoff = value->rValue;
            mod->BSIM4v0lvoffGiven = TRUE;
            break;
        case BSIM4v0_MOD_LMINV:
            mod->BSIM4v0lminv = value->rValue;
            mod->BSIM4v0lminvGiven = TRUE;
            break;
        case BSIM4v0_MOD_LFPROUT:
            mod->BSIM4v0lfprout = value->rValue;
            mod->BSIM4v0lfproutGiven = TRUE;
            break;
        case BSIM4v0_MOD_LPDITS:
            mod->BSIM4v0lpdits = value->rValue;
            mod->BSIM4v0lpditsGiven = TRUE;
            break;
        case BSIM4v0_MOD_LPDITSD:
            mod->BSIM4v0lpditsd = value->rValue;
            mod->BSIM4v0lpditsdGiven = TRUE;
            break;
        case  BSIM4v0_MOD_LDELTA :
            mod->BSIM4v0ldelta = value->rValue;
            mod->BSIM4v0ldeltaGiven = TRUE;
            break;
        case BSIM4v0_MOD_LRDSW:
            mod->BSIM4v0lrdsw = value->rValue;
            mod->BSIM4v0lrdswGiven = TRUE;
            break;                     
        case BSIM4v0_MOD_LRDW:
            mod->BSIM4v0lrdw = value->rValue;
            mod->BSIM4v0lrdwGiven = TRUE;
            break;
        case BSIM4v0_MOD_LRSW:
            mod->BSIM4v0lrsw = value->rValue;
            mod->BSIM4v0lrswGiven = TRUE;
            break;
        case BSIM4v0_MOD_LPRWB:
            mod->BSIM4v0lprwb = value->rValue;
            mod->BSIM4v0lprwbGiven = TRUE;
            break;                     
        case BSIM4v0_MOD_LPRWG:
            mod->BSIM4v0lprwg = value->rValue;
            mod->BSIM4v0lprwgGiven = TRUE;
            break;                     
        case BSIM4v0_MOD_LPRT:
            mod->BSIM4v0lprt = value->rValue;
            mod->BSIM4v0lprtGiven = TRUE;
            break;                     
        case BSIM4v0_MOD_LETA0:
            mod->BSIM4v0leta0 = value->rValue;
            mod->BSIM4v0leta0Given = TRUE;
            break;                 
        case BSIM4v0_MOD_LETAB:
            mod->BSIM4v0letab = value->rValue;
            mod->BSIM4v0letabGiven = TRUE;
            break;                 
        case BSIM4v0_MOD_LPCLM:
            mod->BSIM4v0lpclm = value->rValue;
            mod->BSIM4v0lpclmGiven = TRUE;
            break;                 
        case BSIM4v0_MOD_LPDIBL1:
            mod->BSIM4v0lpdibl1 = value->rValue;
            mod->BSIM4v0lpdibl1Given = TRUE;
            break;                 
        case BSIM4v0_MOD_LPDIBL2:
            mod->BSIM4v0lpdibl2 = value->rValue;
            mod->BSIM4v0lpdibl2Given = TRUE;
            break;                 
        case BSIM4v0_MOD_LPDIBLB:
            mod->BSIM4v0lpdiblb = value->rValue;
            mod->BSIM4v0lpdiblbGiven = TRUE;
            break;                 
        case BSIM4v0_MOD_LPSCBE1:
            mod->BSIM4v0lpscbe1 = value->rValue;
            mod->BSIM4v0lpscbe1Given = TRUE;
            break;                 
        case BSIM4v0_MOD_LPSCBE2:
            mod->BSIM4v0lpscbe2 = value->rValue;
            mod->BSIM4v0lpscbe2Given = TRUE;
            break;                 
        case BSIM4v0_MOD_LPVAG:
            mod->BSIM4v0lpvag = value->rValue;
            mod->BSIM4v0lpvagGiven = TRUE;
            break;                 
        case  BSIM4v0_MOD_LWR :
            mod->BSIM4v0lwr = value->rValue;
            mod->BSIM4v0lwrGiven = TRUE;
            break;
        case  BSIM4v0_MOD_LDWG :
            mod->BSIM4v0ldwg = value->rValue;
            mod->BSIM4v0ldwgGiven = TRUE;
            break;
        case  BSIM4v0_MOD_LDWB :
            mod->BSIM4v0ldwb = value->rValue;
            mod->BSIM4v0ldwbGiven = TRUE;
            break;
        case  BSIM4v0_MOD_LB0 :
            mod->BSIM4v0lb0 = value->rValue;
            mod->BSIM4v0lb0Given = TRUE;
            break;
        case  BSIM4v0_MOD_LB1 :
            mod->BSIM4v0lb1 = value->rValue;
            mod->BSIM4v0lb1Given = TRUE;
            break;
        case  BSIM4v0_MOD_LALPHA0 :
            mod->BSIM4v0lalpha0 = value->rValue;
            mod->BSIM4v0lalpha0Given = TRUE;
            break;
        case  BSIM4v0_MOD_LALPHA1 :
            mod->BSIM4v0lalpha1 = value->rValue;
            mod->BSIM4v0lalpha1Given = TRUE;
            break;
        case  BSIM4v0_MOD_LBETA0 :
            mod->BSIM4v0lbeta0 = value->rValue;
            mod->BSIM4v0lbeta0Given = TRUE;
            break;
        case  BSIM4v0_MOD_LAGIDL :
            mod->BSIM4v0lagidl = value->rValue;
            mod->BSIM4v0lagidlGiven = TRUE;
            break;
        case  BSIM4v0_MOD_LBGIDL :
            mod->BSIM4v0lbgidl = value->rValue;
            mod->BSIM4v0lbgidlGiven = TRUE;
            break;
        case  BSIM4v0_MOD_LCGIDL :
            mod->BSIM4v0lcgidl = value->rValue;
            mod->BSIM4v0lcgidlGiven = TRUE;
            break;
        case  BSIM4v0_MOD_LPHIN :
            mod->BSIM4v0lphin = value->rValue;
            mod->BSIM4v0lphinGiven = TRUE;
            break;
        case  BSIM4v0_MOD_LEGIDL :
            mod->BSIM4v0legidl = value->rValue;
            mod->BSIM4v0legidlGiven = TRUE;
            break;
        case  BSIM4v0_MOD_LAIGC :
            mod->BSIM4v0laigc = value->rValue;
            mod->BSIM4v0laigcGiven = TRUE;
            break;
        case  BSIM4v0_MOD_LBIGC :
            mod->BSIM4v0lbigc = value->rValue;
            mod->BSIM4v0lbigcGiven = TRUE;
            break;
        case  BSIM4v0_MOD_LCIGC :
            mod->BSIM4v0lcigc = value->rValue;
            mod->BSIM4v0lcigcGiven = TRUE;
            break;
        case  BSIM4v0_MOD_LAIGSD :
            mod->BSIM4v0laigsd = value->rValue;
            mod->BSIM4v0laigsdGiven = TRUE;
            break;
        case  BSIM4v0_MOD_LBIGSD :
            mod->BSIM4v0lbigsd = value->rValue;
            mod->BSIM4v0lbigsdGiven = TRUE;
            break;
        case  BSIM4v0_MOD_LCIGSD :
            mod->BSIM4v0lcigsd = value->rValue;
            mod->BSIM4v0lcigsdGiven = TRUE;
            break;
        case  BSIM4v0_MOD_LAIGBACC :
            mod->BSIM4v0laigbacc = value->rValue;
            mod->BSIM4v0laigbaccGiven = TRUE;
            break;
        case  BSIM4v0_MOD_LBIGBACC :
            mod->BSIM4v0lbigbacc = value->rValue;
            mod->BSIM4v0lbigbaccGiven = TRUE;
            break;
        case  BSIM4v0_MOD_LCIGBACC :
            mod->BSIM4v0lcigbacc = value->rValue;
            mod->BSIM4v0lcigbaccGiven = TRUE;
            break;
        case  BSIM4v0_MOD_LAIGBINV :
            mod->BSIM4v0laigbinv = value->rValue;
            mod->BSIM4v0laigbinvGiven = TRUE;
            break;
        case  BSIM4v0_MOD_LBIGBINV :
            mod->BSIM4v0lbigbinv = value->rValue;
            mod->BSIM4v0lbigbinvGiven = TRUE;
            break;
        case  BSIM4v0_MOD_LCIGBINV :
            mod->BSIM4v0lcigbinv = value->rValue;
            mod->BSIM4v0lcigbinvGiven = TRUE;
            break;
        case  BSIM4v0_MOD_LNIGC :
            mod->BSIM4v0lnigc = value->rValue;
            mod->BSIM4v0lnigcGiven = TRUE;
            break;
        case  BSIM4v0_MOD_LNIGBINV :
            mod->BSIM4v0lnigbinv = value->rValue;
            mod->BSIM4v0lnigbinvGiven = TRUE;
            break;
        case  BSIM4v0_MOD_LNIGBACC :
            mod->BSIM4v0lnigbacc = value->rValue;
            mod->BSIM4v0lnigbaccGiven = TRUE;
            break;
        case  BSIM4v0_MOD_LNTOX :
            mod->BSIM4v0lntox = value->rValue;
            mod->BSIM4v0lntoxGiven = TRUE;
            break;
        case  BSIM4v0_MOD_LEIGBINV :
            mod->BSIM4v0leigbinv = value->rValue;
            mod->BSIM4v0leigbinvGiven = TRUE;
            break;
        case  BSIM4v0_MOD_LPIGCD :
            mod->BSIM4v0lpigcd = value->rValue;
            mod->BSIM4v0lpigcdGiven = TRUE;
            break;
        case  BSIM4v0_MOD_LPOXEDGE :
            mod->BSIM4v0lpoxedge = value->rValue;
            mod->BSIM4v0lpoxedgeGiven = TRUE;
            break;
        case  BSIM4v0_MOD_LXRCRG1 :
            mod->BSIM4v0lxrcrg1 = value->rValue;
            mod->BSIM4v0lxrcrg1Given = TRUE;
            break;
        case  BSIM4v0_MOD_LXRCRG2 :
            mod->BSIM4v0lxrcrg2 = value->rValue;
            mod->BSIM4v0lxrcrg2Given = TRUE;
            break;
        case  BSIM4v0_MOD_LEU :
            mod->BSIM4v0leu = value->rValue;
            mod->BSIM4v0leuGiven = TRUE;
            break;
        case  BSIM4v0_MOD_LVFB :
            mod->BSIM4v0lvfb = value->rValue;
            mod->BSIM4v0lvfbGiven = TRUE;
            break;

        case  BSIM4v0_MOD_LCGSL :
            mod->BSIM4v0lcgsl = value->rValue;
            mod->BSIM4v0lcgslGiven = TRUE;
            break;
        case  BSIM4v0_MOD_LCGDL :
            mod->BSIM4v0lcgdl = value->rValue;
            mod->BSIM4v0lcgdlGiven = TRUE;
            break;
        case  BSIM4v0_MOD_LCKAPPAS :
            mod->BSIM4v0lckappas = value->rValue;
            mod->BSIM4v0lckappasGiven = TRUE;
            break;
        case  BSIM4v0_MOD_LCKAPPAD :
            mod->BSIM4v0lckappad = value->rValue;
            mod->BSIM4v0lckappadGiven = TRUE;
            break;
        case  BSIM4v0_MOD_LCF :
            mod->BSIM4v0lcf = value->rValue;
            mod->BSIM4v0lcfGiven = TRUE;
            break;
        case  BSIM4v0_MOD_LCLC :
            mod->BSIM4v0lclc = value->rValue;
            mod->BSIM4v0lclcGiven = TRUE;
            break;
        case  BSIM4v0_MOD_LCLE :
            mod->BSIM4v0lcle = value->rValue;
            mod->BSIM4v0lcleGiven = TRUE;
            break;
        case  BSIM4v0_MOD_LVFBCV :
            mod->BSIM4v0lvfbcv = value->rValue;
            mod->BSIM4v0lvfbcvGiven = TRUE;
            break;
        case  BSIM4v0_MOD_LACDE :
            mod->BSIM4v0lacde = value->rValue;
            mod->BSIM4v0lacdeGiven = TRUE;
            break;
        case  BSIM4v0_MOD_LMOIN :
            mod->BSIM4v0lmoin = value->rValue;
            mod->BSIM4v0lmoinGiven = TRUE;
            break;
        case  BSIM4v0_MOD_LNOFF :
            mod->BSIM4v0lnoff = value->rValue;
            mod->BSIM4v0lnoffGiven = TRUE;
            break;
        case  BSIM4v0_MOD_LVOFFCV :
            mod->BSIM4v0lvoffcv = value->rValue;
            mod->BSIM4v0lvoffcvGiven = TRUE;
            break;

	/* Width dependence */
        case  BSIM4v0_MOD_WCDSC :
            mod->BSIM4v0wcdsc = value->rValue;
            mod->BSIM4v0wcdscGiven = TRUE;
            break;
       
       
         case  BSIM4v0_MOD_WCDSCB :
            mod->BSIM4v0wcdscb = value->rValue;
            mod->BSIM4v0wcdscbGiven = TRUE;
            break;
         case  BSIM4v0_MOD_WCDSCD :
            mod->BSIM4v0wcdscd = value->rValue;
            mod->BSIM4v0wcdscdGiven = TRUE;
            break;
        case  BSIM4v0_MOD_WCIT :
            mod->BSIM4v0wcit = value->rValue;
            mod->BSIM4v0wcitGiven = TRUE;
            break;
        case  BSIM4v0_MOD_WNFACTOR :
            mod->BSIM4v0wnfactor = value->rValue;
            mod->BSIM4v0wnfactorGiven = TRUE;
            break;
        case BSIM4v0_MOD_WXJ:
            mod->BSIM4v0wxj = value->rValue;
            mod->BSIM4v0wxjGiven = TRUE;
            break;
        case BSIM4v0_MOD_WVSAT:
            mod->BSIM4v0wvsat = value->rValue;
            mod->BSIM4v0wvsatGiven = TRUE;
            break;


        case BSIM4v0_MOD_WA0:
            mod->BSIM4v0wa0 = value->rValue;
            mod->BSIM4v0wa0Given = TRUE;
            break;
        case BSIM4v0_MOD_WAGS:
            mod->BSIM4v0wags = value->rValue;
            mod->BSIM4v0wagsGiven = TRUE;
            break;
        case BSIM4v0_MOD_WA1:
            mod->BSIM4v0wa1 = value->rValue;
            mod->BSIM4v0wa1Given = TRUE;
            break;
        case BSIM4v0_MOD_WA2:
            mod->BSIM4v0wa2 = value->rValue;
            mod->BSIM4v0wa2Given = TRUE;
            break;
        case BSIM4v0_MOD_WAT:
            mod->BSIM4v0wat = value->rValue;
            mod->BSIM4v0watGiven = TRUE;
            break;
        case BSIM4v0_MOD_WKETA:
            mod->BSIM4v0wketa = value->rValue;
            mod->BSIM4v0wketaGiven = TRUE;
            break;    
        case BSIM4v0_MOD_WNSUB:
            mod->BSIM4v0wnsub = value->rValue;
            mod->BSIM4v0wnsubGiven = TRUE;
            break;
        case BSIM4v0_MOD_WNDEP:
            mod->BSIM4v0wndep = value->rValue;
            mod->BSIM4v0wndepGiven = TRUE;
	    if (mod->BSIM4v0wndep > 1.0e20)
		mod->BSIM4v0wndep *= 1.0e-6;
            break;
        case BSIM4v0_MOD_WNSD:
            mod->BSIM4v0wnsd = value->rValue;
            mod->BSIM4v0wnsdGiven = TRUE;
            if (mod->BSIM4v0wnsd > 1.0e23)
                mod->BSIM4v0wnsd *= 1.0e-6;
            break;
        case BSIM4v0_MOD_WNGATE:
            mod->BSIM4v0wngate = value->rValue;
            mod->BSIM4v0wngateGiven = TRUE;
	    if (mod->BSIM4v0wngate > 1.0e23)
		mod->BSIM4v0wngate *= 1.0e-6;
            break;
        case BSIM4v0_MOD_WGAMMA1:
            mod->BSIM4v0wgamma1 = value->rValue;
            mod->BSIM4v0wgamma1Given = TRUE;
            break;
        case BSIM4v0_MOD_WGAMMA2:
            mod->BSIM4v0wgamma2 = value->rValue;
            mod->BSIM4v0wgamma2Given = TRUE;
            break;
        case BSIM4v0_MOD_WVBX:
            mod->BSIM4v0wvbx = value->rValue;
            mod->BSIM4v0wvbxGiven = TRUE;
            break;
        case BSIM4v0_MOD_WVBM:
            mod->BSIM4v0wvbm = value->rValue;
            mod->BSIM4v0wvbmGiven = TRUE;
            break;
        case BSIM4v0_MOD_WXT:
            mod->BSIM4v0wxt = value->rValue;
            mod->BSIM4v0wxtGiven = TRUE;
            break;
        case  BSIM4v0_MOD_WK1:
            mod->BSIM4v0wk1 = value->rValue;
            mod->BSIM4v0wk1Given = TRUE;
            break;
        case  BSIM4v0_MOD_WKT1:
            mod->BSIM4v0wkt1 = value->rValue;
            mod->BSIM4v0wkt1Given = TRUE;
            break;
        case  BSIM4v0_MOD_WKT1L:
            mod->BSIM4v0wkt1l = value->rValue;
            mod->BSIM4v0wkt1lGiven = TRUE;
            break;
        case  BSIM4v0_MOD_WKT2:
            mod->BSIM4v0wkt2 = value->rValue;
            mod->BSIM4v0wkt2Given = TRUE;
            break;
        case  BSIM4v0_MOD_WK2:
            mod->BSIM4v0wk2 = value->rValue;
            mod->BSIM4v0wk2Given = TRUE;
            break;
        case  BSIM4v0_MOD_WK3:
            mod->BSIM4v0wk3 = value->rValue;
            mod->BSIM4v0wk3Given = TRUE;
            break;
        case  BSIM4v0_MOD_WK3B:
            mod->BSIM4v0wk3b = value->rValue;
            mod->BSIM4v0wk3bGiven = TRUE;
            break;
        case  BSIM4v0_MOD_WLPE0:
            mod->BSIM4v0wlpe0 = value->rValue;
            mod->BSIM4v0wlpe0Given = TRUE;
            break;
        case  BSIM4v0_MOD_WLPEB:
            mod->BSIM4v0wlpeb = value->rValue;
            mod->BSIM4v0wlpebGiven = TRUE;
            break;
        case  BSIM4v0_MOD_WDVTP0:
            mod->BSIM4v0wdvtp0 = value->rValue;
            mod->BSIM4v0wdvtp0Given = TRUE;
            break;
        case  BSIM4v0_MOD_WDVTP1:
            mod->BSIM4v0wdvtp1 = value->rValue;
            mod->BSIM4v0wdvtp1Given = TRUE;
            break;
        case  BSIM4v0_MOD_WW0:
            mod->BSIM4v0ww0 = value->rValue;
            mod->BSIM4v0ww0Given = TRUE;
            break;
        case  BSIM4v0_MOD_WDVT0:               
            mod->BSIM4v0wdvt0 = value->rValue;
            mod->BSIM4v0wdvt0Given = TRUE;
            break;
        case  BSIM4v0_MOD_WDVT1:             
            mod->BSIM4v0wdvt1 = value->rValue;
            mod->BSIM4v0wdvt1Given = TRUE;
            break;
        case  BSIM4v0_MOD_WDVT2:             
            mod->BSIM4v0wdvt2 = value->rValue;
            mod->BSIM4v0wdvt2Given = TRUE;
            break;
        case  BSIM4v0_MOD_WDVT0W:               
            mod->BSIM4v0wdvt0w = value->rValue;
            mod->BSIM4v0wdvt0wGiven = TRUE;
            break;
        case  BSIM4v0_MOD_WDVT1W:             
            mod->BSIM4v0wdvt1w = value->rValue;
            mod->BSIM4v0wdvt1wGiven = TRUE;
            break;
        case  BSIM4v0_MOD_WDVT2W:             
            mod->BSIM4v0wdvt2w = value->rValue;
            mod->BSIM4v0wdvt2wGiven = TRUE;
            break;
        case  BSIM4v0_MOD_WDROUT:             
            mod->BSIM4v0wdrout = value->rValue;
            mod->BSIM4v0wdroutGiven = TRUE;
            break;
        case  BSIM4v0_MOD_WDSUB:             
            mod->BSIM4v0wdsub = value->rValue;
            mod->BSIM4v0wdsubGiven = TRUE;
            break;
        case BSIM4v0_MOD_WVTH0:
            mod->BSIM4v0wvth0 = value->rValue;
            mod->BSIM4v0wvth0Given = TRUE;
            break;
        case BSIM4v0_MOD_WUA:
            mod->BSIM4v0wua = value->rValue;
            mod->BSIM4v0wuaGiven = TRUE;
            break;
        case BSIM4v0_MOD_WUA1:
            mod->BSIM4v0wua1 = value->rValue;
            mod->BSIM4v0wua1Given = TRUE;
            break;
        case BSIM4v0_MOD_WUB:
            mod->BSIM4v0wub = value->rValue;
            mod->BSIM4v0wubGiven = TRUE;
            break;
        case BSIM4v0_MOD_WUB1:
            mod->BSIM4v0wub1 = value->rValue;
            mod->BSIM4v0wub1Given = TRUE;
            break;
        case BSIM4v0_MOD_WUC:
            mod->BSIM4v0wuc = value->rValue;
            mod->BSIM4v0wucGiven = TRUE;
            break;
        case BSIM4v0_MOD_WUC1:
            mod->BSIM4v0wuc1 = value->rValue;
            mod->BSIM4v0wuc1Given = TRUE;
            break;
        case  BSIM4v0_MOD_WU0 :
            mod->BSIM4v0wu0 = value->rValue;
            mod->BSIM4v0wu0Given = TRUE;
            break;
        case  BSIM4v0_MOD_WUTE :
            mod->BSIM4v0wute = value->rValue;
            mod->BSIM4v0wuteGiven = TRUE;
            break;
        case BSIM4v0_MOD_WVOFF:
            mod->BSIM4v0wvoff = value->rValue;
            mod->BSIM4v0wvoffGiven = TRUE;
            break;
        case BSIM4v0_MOD_WMINV:
            mod->BSIM4v0wminv = value->rValue;
            mod->BSIM4v0wminvGiven = TRUE;
            break;
        case BSIM4v0_MOD_WFPROUT:
            mod->BSIM4v0wfprout = value->rValue;
            mod->BSIM4v0wfproutGiven = TRUE;
            break;
        case BSIM4v0_MOD_WPDITS:
            mod->BSIM4v0wpdits = value->rValue;
            mod->BSIM4v0wpditsGiven = TRUE;
            break;
        case BSIM4v0_MOD_WPDITSD:
            mod->BSIM4v0wpditsd = value->rValue;
            mod->BSIM4v0wpditsdGiven = TRUE;
            break;
        case  BSIM4v0_MOD_WDELTA :
            mod->BSIM4v0wdelta = value->rValue;
            mod->BSIM4v0wdeltaGiven = TRUE;
            break;
        case BSIM4v0_MOD_WRDSW:
            mod->BSIM4v0wrdsw = value->rValue;
            mod->BSIM4v0wrdswGiven = TRUE;
            break;                     
        case BSIM4v0_MOD_WRDW:
            mod->BSIM4v0wrdw = value->rValue;
            mod->BSIM4v0wrdwGiven = TRUE;
            break;
        case BSIM4v0_MOD_WRSW:
            mod->BSIM4v0wrsw = value->rValue;
            mod->BSIM4v0wrswGiven = TRUE;
            break;
        case BSIM4v0_MOD_WPRWB:
            mod->BSIM4v0wprwb = value->rValue;
            mod->BSIM4v0wprwbGiven = TRUE;
            break;                     
        case BSIM4v0_MOD_WPRWG:
            mod->BSIM4v0wprwg = value->rValue;
            mod->BSIM4v0wprwgGiven = TRUE;
            break;                     
        case BSIM4v0_MOD_WPRT:
            mod->BSIM4v0wprt = value->rValue;
            mod->BSIM4v0wprtGiven = TRUE;
            break;                     
        case BSIM4v0_MOD_WETA0:
            mod->BSIM4v0weta0 = value->rValue;
            mod->BSIM4v0weta0Given = TRUE;
            break;                 
        case BSIM4v0_MOD_WETAB:
            mod->BSIM4v0wetab = value->rValue;
            mod->BSIM4v0wetabGiven = TRUE;
            break;                 
        case BSIM4v0_MOD_WPCLM:
            mod->BSIM4v0wpclm = value->rValue;
            mod->BSIM4v0wpclmGiven = TRUE;
            break;                 
        case BSIM4v0_MOD_WPDIBL1:
            mod->BSIM4v0wpdibl1 = value->rValue;
            mod->BSIM4v0wpdibl1Given = TRUE;
            break;                 
        case BSIM4v0_MOD_WPDIBL2:
            mod->BSIM4v0wpdibl2 = value->rValue;
            mod->BSIM4v0wpdibl2Given = TRUE;
            break;                 
        case BSIM4v0_MOD_WPDIBLB:
            mod->BSIM4v0wpdiblb = value->rValue;
            mod->BSIM4v0wpdiblbGiven = TRUE;
            break;                 
        case BSIM4v0_MOD_WPSCBE1:
            mod->BSIM4v0wpscbe1 = value->rValue;
            mod->BSIM4v0wpscbe1Given = TRUE;
            break;                 
        case BSIM4v0_MOD_WPSCBE2:
            mod->BSIM4v0wpscbe2 = value->rValue;
            mod->BSIM4v0wpscbe2Given = TRUE;
            break;                 
        case BSIM4v0_MOD_WPVAG:
            mod->BSIM4v0wpvag = value->rValue;
            mod->BSIM4v0wpvagGiven = TRUE;
            break;                 
        case  BSIM4v0_MOD_WWR :
            mod->BSIM4v0wwr = value->rValue;
            mod->BSIM4v0wwrGiven = TRUE;
            break;
        case  BSIM4v0_MOD_WDWG :
            mod->BSIM4v0wdwg = value->rValue;
            mod->BSIM4v0wdwgGiven = TRUE;
            break;
        case  BSIM4v0_MOD_WDWB :
            mod->BSIM4v0wdwb = value->rValue;
            mod->BSIM4v0wdwbGiven = TRUE;
            break;
        case  BSIM4v0_MOD_WB0 :
            mod->BSIM4v0wb0 = value->rValue;
            mod->BSIM4v0wb0Given = TRUE;
            break;
        case  BSIM4v0_MOD_WB1 :
            mod->BSIM4v0wb1 = value->rValue;
            mod->BSIM4v0wb1Given = TRUE;
            break;
        case  BSIM4v0_MOD_WALPHA0 :
            mod->BSIM4v0walpha0 = value->rValue;
            mod->BSIM4v0walpha0Given = TRUE;
            break;
        case  BSIM4v0_MOD_WALPHA1 :
            mod->BSIM4v0walpha1 = value->rValue;
            mod->BSIM4v0walpha1Given = TRUE;
            break;
        case  BSIM4v0_MOD_WBETA0 :
            mod->BSIM4v0wbeta0 = value->rValue;
            mod->BSIM4v0wbeta0Given = TRUE;
            break;
        case  BSIM4v0_MOD_WAGIDL :
            mod->BSIM4v0wagidl = value->rValue;
            mod->BSIM4v0wagidlGiven = TRUE;
            break;
        case  BSIM4v0_MOD_WBGIDL :
            mod->BSIM4v0wbgidl = value->rValue;
            mod->BSIM4v0wbgidlGiven = TRUE;
            break;
        case  BSIM4v0_MOD_WCGIDL :
            mod->BSIM4v0wcgidl = value->rValue;
            mod->BSIM4v0wcgidlGiven = TRUE;
            break;
        case  BSIM4v0_MOD_WPHIN :
            mod->BSIM4v0wphin = value->rValue;
            mod->BSIM4v0wphinGiven = TRUE;
            break;
        case  BSIM4v0_MOD_WEGIDL :
            mod->BSIM4v0wegidl = value->rValue;
            mod->BSIM4v0wegidlGiven = TRUE;
            break;
        case  BSIM4v0_MOD_WAIGC :
            mod->BSIM4v0waigc = value->rValue;
            mod->BSIM4v0waigcGiven = TRUE;
            break;
        case  BSIM4v0_MOD_WBIGC :
            mod->BSIM4v0wbigc = value->rValue;
            mod->BSIM4v0wbigcGiven = TRUE;
            break;
        case  BSIM4v0_MOD_WCIGC :
            mod->BSIM4v0wcigc = value->rValue;
            mod->BSIM4v0wcigcGiven = TRUE;
            break;
        case  BSIM4v0_MOD_WAIGSD :
            mod->BSIM4v0waigsd = value->rValue;
            mod->BSIM4v0waigsdGiven = TRUE;
            break;
        case  BSIM4v0_MOD_WBIGSD :
            mod->BSIM4v0wbigsd = value->rValue;
            mod->BSIM4v0wbigsdGiven = TRUE;
            break;
        case  BSIM4v0_MOD_WCIGSD :
            mod->BSIM4v0wcigsd = value->rValue;
            mod->BSIM4v0wcigsdGiven = TRUE;
            break;
        case  BSIM4v0_MOD_WAIGBACC :
            mod->BSIM4v0waigbacc = value->rValue;
            mod->BSIM4v0waigbaccGiven = TRUE;
            break;
        case  BSIM4v0_MOD_WBIGBACC :
            mod->BSIM4v0wbigbacc = value->rValue;
            mod->BSIM4v0wbigbaccGiven = TRUE;
            break;
        case  BSIM4v0_MOD_WCIGBACC :
            mod->BSIM4v0wcigbacc = value->rValue;
            mod->BSIM4v0wcigbaccGiven = TRUE;
            break;
        case  BSIM4v0_MOD_WAIGBINV :
            mod->BSIM4v0waigbinv = value->rValue;
            mod->BSIM4v0waigbinvGiven = TRUE;
            break;
        case  BSIM4v0_MOD_WBIGBINV :
            mod->BSIM4v0wbigbinv = value->rValue;
            mod->BSIM4v0wbigbinvGiven = TRUE;
            break;
        case  BSIM4v0_MOD_WCIGBINV :
            mod->BSIM4v0wcigbinv = value->rValue;
            mod->BSIM4v0wcigbinvGiven = TRUE;
            break;
        case  BSIM4v0_MOD_WNIGC :
            mod->BSIM4v0wnigc = value->rValue;
            mod->BSIM4v0wnigcGiven = TRUE;
            break;
        case  BSIM4v0_MOD_WNIGBINV :
            mod->BSIM4v0wnigbinv = value->rValue;
            mod->BSIM4v0wnigbinvGiven = TRUE;
            break;
        case  BSIM4v0_MOD_WNIGBACC :
            mod->BSIM4v0wnigbacc = value->rValue;
            mod->BSIM4v0wnigbaccGiven = TRUE;
            break;
        case  BSIM4v0_MOD_WNTOX :
            mod->BSIM4v0wntox = value->rValue;
            mod->BSIM4v0wntoxGiven = TRUE;
            break;
        case  BSIM4v0_MOD_WEIGBINV :
            mod->BSIM4v0weigbinv = value->rValue;
            mod->BSIM4v0weigbinvGiven = TRUE;
            break;
        case  BSIM4v0_MOD_WPIGCD :
            mod->BSIM4v0wpigcd = value->rValue;
            mod->BSIM4v0wpigcdGiven = TRUE;
            break;
        case  BSIM4v0_MOD_WPOXEDGE :
            mod->BSIM4v0wpoxedge = value->rValue;
            mod->BSIM4v0wpoxedgeGiven = TRUE;
            break;
        case  BSIM4v0_MOD_WXRCRG1 :
            mod->BSIM4v0wxrcrg1 = value->rValue;
            mod->BSIM4v0wxrcrg1Given = TRUE;
            break;
        case  BSIM4v0_MOD_WXRCRG2 :
            mod->BSIM4v0wxrcrg2 = value->rValue;
            mod->BSIM4v0wxrcrg2Given = TRUE;
            break;
        case  BSIM4v0_MOD_WEU :
            mod->BSIM4v0weu = value->rValue;
            mod->BSIM4v0weuGiven = TRUE;
            break;
        case  BSIM4v0_MOD_WVFB :
            mod->BSIM4v0wvfb = value->rValue;
            mod->BSIM4v0wvfbGiven = TRUE;
            break;

        case  BSIM4v0_MOD_WCGSL :
            mod->BSIM4v0wcgsl = value->rValue;
            mod->BSIM4v0wcgslGiven = TRUE;
            break;
        case  BSIM4v0_MOD_WCGDL :
            mod->BSIM4v0wcgdl = value->rValue;
            mod->BSIM4v0wcgdlGiven = TRUE;
            break;
        case  BSIM4v0_MOD_WCKAPPAS :
            mod->BSIM4v0wckappas = value->rValue;
            mod->BSIM4v0wckappasGiven = TRUE;
            break;
        case  BSIM4v0_MOD_WCKAPPAD :
            mod->BSIM4v0wckappad = value->rValue;
            mod->BSIM4v0wckappadGiven = TRUE;
            break;
        case  BSIM4v0_MOD_WCF :
            mod->BSIM4v0wcf = value->rValue;
            mod->BSIM4v0wcfGiven = TRUE;
            break;
        case  BSIM4v0_MOD_WCLC :
            mod->BSIM4v0wclc = value->rValue;
            mod->BSIM4v0wclcGiven = TRUE;
            break;
        case  BSIM4v0_MOD_WCLE :
            mod->BSIM4v0wcle = value->rValue;
            mod->BSIM4v0wcleGiven = TRUE;
            break;
        case  BSIM4v0_MOD_WVFBCV :
            mod->BSIM4v0wvfbcv = value->rValue;
            mod->BSIM4v0wvfbcvGiven = TRUE;
            break;
        case  BSIM4v0_MOD_WACDE :
            mod->BSIM4v0wacde = value->rValue;
            mod->BSIM4v0wacdeGiven = TRUE;
            break;
        case  BSIM4v0_MOD_WMOIN :
            mod->BSIM4v0wmoin = value->rValue;
            mod->BSIM4v0wmoinGiven = TRUE;
            break;
        case  BSIM4v0_MOD_WNOFF :
            mod->BSIM4v0wnoff = value->rValue;
            mod->BSIM4v0wnoffGiven = TRUE;
            break;
        case  BSIM4v0_MOD_WVOFFCV :
            mod->BSIM4v0wvoffcv = value->rValue;
            mod->BSIM4v0wvoffcvGiven = TRUE;
            break;

	/* Cross-term dependence */
        case  BSIM4v0_MOD_PCDSC :
            mod->BSIM4v0pcdsc = value->rValue;
            mod->BSIM4v0pcdscGiven = TRUE;
            break;


        case  BSIM4v0_MOD_PCDSCB :
            mod->BSIM4v0pcdscb = value->rValue;
            mod->BSIM4v0pcdscbGiven = TRUE;
            break;
        case  BSIM4v0_MOD_PCDSCD :
            mod->BSIM4v0pcdscd = value->rValue;
            mod->BSIM4v0pcdscdGiven = TRUE;
            break;
        case  BSIM4v0_MOD_PCIT :
            mod->BSIM4v0pcit = value->rValue;
            mod->BSIM4v0pcitGiven = TRUE;
            break;
        case  BSIM4v0_MOD_PNFACTOR :
            mod->BSIM4v0pnfactor = value->rValue;
            mod->BSIM4v0pnfactorGiven = TRUE;
            break;
        case BSIM4v0_MOD_PXJ:
            mod->BSIM4v0pxj = value->rValue;
            mod->BSIM4v0pxjGiven = TRUE;
            break;
        case BSIM4v0_MOD_PVSAT:
            mod->BSIM4v0pvsat = value->rValue;
            mod->BSIM4v0pvsatGiven = TRUE;
            break;


        case BSIM4v0_MOD_PA0:
            mod->BSIM4v0pa0 = value->rValue;
            mod->BSIM4v0pa0Given = TRUE;
            break;
        case BSIM4v0_MOD_PAGS:
            mod->BSIM4v0pags = value->rValue;
            mod->BSIM4v0pagsGiven = TRUE;
            break;
        case BSIM4v0_MOD_PA1:
            mod->BSIM4v0pa1 = value->rValue;
            mod->BSIM4v0pa1Given = TRUE;
            break;
        case BSIM4v0_MOD_PA2:
            mod->BSIM4v0pa2 = value->rValue;
            mod->BSIM4v0pa2Given = TRUE;
            break;
        case BSIM4v0_MOD_PAT:
            mod->BSIM4v0pat = value->rValue;
            mod->BSIM4v0patGiven = TRUE;
            break;
        case BSIM4v0_MOD_PKETA:
            mod->BSIM4v0pketa = value->rValue;
            mod->BSIM4v0pketaGiven = TRUE;
            break;    
        case BSIM4v0_MOD_PNSUB:
            mod->BSIM4v0pnsub = value->rValue;
            mod->BSIM4v0pnsubGiven = TRUE;
            break;
        case BSIM4v0_MOD_PNDEP:
            mod->BSIM4v0pndep = value->rValue;
            mod->BSIM4v0pndepGiven = TRUE;
	    if (mod->BSIM4v0pndep > 1.0e20)
		mod->BSIM4v0pndep *= 1.0e-6;
            break;
        case BSIM4v0_MOD_PNSD:
            mod->BSIM4v0pnsd = value->rValue;
            mod->BSIM4v0pnsdGiven = TRUE;
            if (mod->BSIM4v0pnsd > 1.0e23)
                mod->BSIM4v0pnsd *= 1.0e-6;
            break;
        case BSIM4v0_MOD_PNGATE:
            mod->BSIM4v0pngate = value->rValue;
            mod->BSIM4v0pngateGiven = TRUE;
	    if (mod->BSIM4v0pngate > 1.0e23)
		mod->BSIM4v0pngate *= 1.0e-6;
            break;
        case BSIM4v0_MOD_PGAMMA1:
            mod->BSIM4v0pgamma1 = value->rValue;
            mod->BSIM4v0pgamma1Given = TRUE;
            break;
        case BSIM4v0_MOD_PGAMMA2:
            mod->BSIM4v0pgamma2 = value->rValue;
            mod->BSIM4v0pgamma2Given = TRUE;
            break;
        case BSIM4v0_MOD_PVBX:
            mod->BSIM4v0pvbx = value->rValue;
            mod->BSIM4v0pvbxGiven = TRUE;
            break;
        case BSIM4v0_MOD_PVBM:
            mod->BSIM4v0pvbm = value->rValue;
            mod->BSIM4v0pvbmGiven = TRUE;
            break;
        case BSIM4v0_MOD_PXT:
            mod->BSIM4v0pxt = value->rValue;
            mod->BSIM4v0pxtGiven = TRUE;
            break;
        case  BSIM4v0_MOD_PK1:
            mod->BSIM4v0pk1 = value->rValue;
            mod->BSIM4v0pk1Given = TRUE;
            break;
        case  BSIM4v0_MOD_PKT1:
            mod->BSIM4v0pkt1 = value->rValue;
            mod->BSIM4v0pkt1Given = TRUE;
            break;
        case  BSIM4v0_MOD_PKT1L:
            mod->BSIM4v0pkt1l = value->rValue;
            mod->BSIM4v0pkt1lGiven = TRUE;
            break;
        case  BSIM4v0_MOD_PKT2:
            mod->BSIM4v0pkt2 = value->rValue;
            mod->BSIM4v0pkt2Given = TRUE;
            break;
        case  BSIM4v0_MOD_PK2:
            mod->BSIM4v0pk2 = value->rValue;
            mod->BSIM4v0pk2Given = TRUE;
            break;
        case  BSIM4v0_MOD_PK3:
            mod->BSIM4v0pk3 = value->rValue;
            mod->BSIM4v0pk3Given = TRUE;
            break;
        case  BSIM4v0_MOD_PK3B:
            mod->BSIM4v0pk3b = value->rValue;
            mod->BSIM4v0pk3bGiven = TRUE;
            break;
        case  BSIM4v0_MOD_PLPE0:
            mod->BSIM4v0plpe0 = value->rValue;
            mod->BSIM4v0plpe0Given = TRUE;
            break;
        case  BSIM4v0_MOD_PLPEB:
            mod->BSIM4v0plpeb = value->rValue;
            mod->BSIM4v0plpebGiven = TRUE;
            break;
        case  BSIM4v0_MOD_PDVTP0:
            mod->BSIM4v0pdvtp0 = value->rValue;
            mod->BSIM4v0pdvtp0Given = TRUE;
            break;
        case  BSIM4v0_MOD_PDVTP1:
            mod->BSIM4v0pdvtp1 = value->rValue;
            mod->BSIM4v0pdvtp1Given = TRUE;
            break;
        case  BSIM4v0_MOD_PW0:
            mod->BSIM4v0pw0 = value->rValue;
            mod->BSIM4v0pw0Given = TRUE;
            break;
        case  BSIM4v0_MOD_PDVT0:               
            mod->BSIM4v0pdvt0 = value->rValue;
            mod->BSIM4v0pdvt0Given = TRUE;
            break;
        case  BSIM4v0_MOD_PDVT1:             
            mod->BSIM4v0pdvt1 = value->rValue;
            mod->BSIM4v0pdvt1Given = TRUE;
            break;
        case  BSIM4v0_MOD_PDVT2:             
            mod->BSIM4v0pdvt2 = value->rValue;
            mod->BSIM4v0pdvt2Given = TRUE;
            break;
        case  BSIM4v0_MOD_PDVT0W:               
            mod->BSIM4v0pdvt0w = value->rValue;
            mod->BSIM4v0pdvt0wGiven = TRUE;
            break;
        case  BSIM4v0_MOD_PDVT1W:             
            mod->BSIM4v0pdvt1w = value->rValue;
            mod->BSIM4v0pdvt1wGiven = TRUE;
            break;
        case  BSIM4v0_MOD_PDVT2W:             
            mod->BSIM4v0pdvt2w = value->rValue;
            mod->BSIM4v0pdvt2wGiven = TRUE;
            break;
        case  BSIM4v0_MOD_PDROUT:             
            mod->BSIM4v0pdrout = value->rValue;
            mod->BSIM4v0pdroutGiven = TRUE;
            break;
        case  BSIM4v0_MOD_PDSUB:             
            mod->BSIM4v0pdsub = value->rValue;
            mod->BSIM4v0pdsubGiven = TRUE;
            break;
        case BSIM4v0_MOD_PVTH0:
            mod->BSIM4v0pvth0 = value->rValue;
            mod->BSIM4v0pvth0Given = TRUE;
            break;
        case BSIM4v0_MOD_PUA:
            mod->BSIM4v0pua = value->rValue;
            mod->BSIM4v0puaGiven = TRUE;
            break;
        case BSIM4v0_MOD_PUA1:
            mod->BSIM4v0pua1 = value->rValue;
            mod->BSIM4v0pua1Given = TRUE;
            break;
        case BSIM4v0_MOD_PUB:
            mod->BSIM4v0pub = value->rValue;
            mod->BSIM4v0pubGiven = TRUE;
            break;
        case BSIM4v0_MOD_PUB1:
            mod->BSIM4v0pub1 = value->rValue;
            mod->BSIM4v0pub1Given = TRUE;
            break;
        case BSIM4v0_MOD_PUC:
            mod->BSIM4v0puc = value->rValue;
            mod->BSIM4v0pucGiven = TRUE;
            break;
        case BSIM4v0_MOD_PUC1:
            mod->BSIM4v0puc1 = value->rValue;
            mod->BSIM4v0puc1Given = TRUE;
            break;
        case  BSIM4v0_MOD_PU0 :
            mod->BSIM4v0pu0 = value->rValue;
            mod->BSIM4v0pu0Given = TRUE;
            break;
        case  BSIM4v0_MOD_PUTE :
            mod->BSIM4v0pute = value->rValue;
            mod->BSIM4v0puteGiven = TRUE;
            break;
        case BSIM4v0_MOD_PVOFF:
            mod->BSIM4v0pvoff = value->rValue;
            mod->BSIM4v0pvoffGiven = TRUE;
            break;
        case BSIM4v0_MOD_PMINV:
            mod->BSIM4v0pminv = value->rValue;
            mod->BSIM4v0pminvGiven = TRUE;
            break;
        case BSIM4v0_MOD_PFPROUT:
            mod->BSIM4v0pfprout = value->rValue;
            mod->BSIM4v0pfproutGiven = TRUE;
            break;
        case BSIM4v0_MOD_PPDITS:
            mod->BSIM4v0ppdits = value->rValue;
            mod->BSIM4v0ppditsGiven = TRUE;
            break;
        case BSIM4v0_MOD_PPDITSD:
            mod->BSIM4v0ppditsd = value->rValue;
            mod->BSIM4v0ppditsdGiven = TRUE;
            break;
        case  BSIM4v0_MOD_PDELTA :
            mod->BSIM4v0pdelta = value->rValue;
            mod->BSIM4v0pdeltaGiven = TRUE;
            break;
        case BSIM4v0_MOD_PRDSW:
            mod->BSIM4v0prdsw = value->rValue;
            mod->BSIM4v0prdswGiven = TRUE;
            break;                     
        case BSIM4v0_MOD_PRDW:
            mod->BSIM4v0prdw = value->rValue;
            mod->BSIM4v0prdwGiven = TRUE;
            break;
        case BSIM4v0_MOD_PRSW:
            mod->BSIM4v0prsw = value->rValue;
            mod->BSIM4v0prswGiven = TRUE;
            break;
        case BSIM4v0_MOD_PPRWB:
            mod->BSIM4v0pprwb = value->rValue;
            mod->BSIM4v0pprwbGiven = TRUE;
            break;                     
        case BSIM4v0_MOD_PPRWG:
            mod->BSIM4v0pprwg = value->rValue;
            mod->BSIM4v0pprwgGiven = TRUE;
            break;                     
        case BSIM4v0_MOD_PPRT:
            mod->BSIM4v0pprt = value->rValue;
            mod->BSIM4v0pprtGiven = TRUE;
            break;                     
        case BSIM4v0_MOD_PETA0:
            mod->BSIM4v0peta0 = value->rValue;
            mod->BSIM4v0peta0Given = TRUE;
            break;                 
        case BSIM4v0_MOD_PETAB:
            mod->BSIM4v0petab = value->rValue;
            mod->BSIM4v0petabGiven = TRUE;
            break;                 
        case BSIM4v0_MOD_PPCLM:
            mod->BSIM4v0ppclm = value->rValue;
            mod->BSIM4v0ppclmGiven = TRUE;
            break;                 
        case BSIM4v0_MOD_PPDIBL1:
            mod->BSIM4v0ppdibl1 = value->rValue;
            mod->BSIM4v0ppdibl1Given = TRUE;
            break;                 
        case BSIM4v0_MOD_PPDIBL2:
            mod->BSIM4v0ppdibl2 = value->rValue;
            mod->BSIM4v0ppdibl2Given = TRUE;
            break;                 
        case BSIM4v0_MOD_PPDIBLB:
            mod->BSIM4v0ppdiblb = value->rValue;
            mod->BSIM4v0ppdiblbGiven = TRUE;
            break;                 
        case BSIM4v0_MOD_PPSCBE1:
            mod->BSIM4v0ppscbe1 = value->rValue;
            mod->BSIM4v0ppscbe1Given = TRUE;
            break;                 
        case BSIM4v0_MOD_PPSCBE2:
            mod->BSIM4v0ppscbe2 = value->rValue;
            mod->BSIM4v0ppscbe2Given = TRUE;
            break;                 
        case BSIM4v0_MOD_PPVAG:
            mod->BSIM4v0ppvag = value->rValue;
            mod->BSIM4v0ppvagGiven = TRUE;
            break;                 
        case  BSIM4v0_MOD_PWR :
            mod->BSIM4v0pwr = value->rValue;
            mod->BSIM4v0pwrGiven = TRUE;
            break;
        case  BSIM4v0_MOD_PDWG :
            mod->BSIM4v0pdwg = value->rValue;
            mod->BSIM4v0pdwgGiven = TRUE;
            break;
        case  BSIM4v0_MOD_PDWB :
            mod->BSIM4v0pdwb = value->rValue;
            mod->BSIM4v0pdwbGiven = TRUE;
            break;
        case  BSIM4v0_MOD_PB0 :
            mod->BSIM4v0pb0 = value->rValue;
            mod->BSIM4v0pb0Given = TRUE;
            break;
        case  BSIM4v0_MOD_PB1 :
            mod->BSIM4v0pb1 = value->rValue;
            mod->BSIM4v0pb1Given = TRUE;
            break;
        case  BSIM4v0_MOD_PALPHA0 :
            mod->BSIM4v0palpha0 = value->rValue;
            mod->BSIM4v0palpha0Given = TRUE;
            break;
        case  BSIM4v0_MOD_PALPHA1 :
            mod->BSIM4v0palpha1 = value->rValue;
            mod->BSIM4v0palpha1Given = TRUE;
            break;
        case  BSIM4v0_MOD_PBETA0 :
            mod->BSIM4v0pbeta0 = value->rValue;
            mod->BSIM4v0pbeta0Given = TRUE;
            break;
        case  BSIM4v0_MOD_PAGIDL :
            mod->BSIM4v0pagidl = value->rValue;
            mod->BSIM4v0pagidlGiven = TRUE;
            break;
        case  BSIM4v0_MOD_PBGIDL :
            mod->BSIM4v0pbgidl = value->rValue;
            mod->BSIM4v0pbgidlGiven = TRUE;
            break;
        case  BSIM4v0_MOD_PCGIDL :
            mod->BSIM4v0pcgidl = value->rValue;
            mod->BSIM4v0pcgidlGiven = TRUE;
            break;
        case  BSIM4v0_MOD_PPHIN :
            mod->BSIM4v0pphin = value->rValue;
            mod->BSIM4v0pphinGiven = TRUE;
            break;
        case  BSIM4v0_MOD_PEGIDL :
            mod->BSIM4v0pegidl = value->rValue;
            mod->BSIM4v0pegidlGiven = TRUE;
            break;
        case  BSIM4v0_MOD_PAIGC :
            mod->BSIM4v0paigc = value->rValue;
            mod->BSIM4v0paigcGiven = TRUE;
            break;
        case  BSIM4v0_MOD_PBIGC :
            mod->BSIM4v0pbigc = value->rValue;
            mod->BSIM4v0pbigcGiven = TRUE;
            break;
        case  BSIM4v0_MOD_PCIGC :
            mod->BSIM4v0pcigc = value->rValue;
            mod->BSIM4v0pcigcGiven = TRUE;
            break;
        case  BSIM4v0_MOD_PAIGSD :
            mod->BSIM4v0paigsd = value->rValue;
            mod->BSIM4v0paigsdGiven = TRUE;
            break;
        case  BSIM4v0_MOD_PBIGSD :
            mod->BSIM4v0pbigsd = value->rValue;
            mod->BSIM4v0pbigsdGiven = TRUE;
            break;
        case  BSIM4v0_MOD_PCIGSD :
            mod->BSIM4v0pcigsd = value->rValue;
            mod->BSIM4v0pcigsdGiven = TRUE;
            break;
        case  BSIM4v0_MOD_PAIGBACC :
            mod->BSIM4v0paigbacc = value->rValue;
            mod->BSIM4v0paigbaccGiven = TRUE;
            break;
        case  BSIM4v0_MOD_PBIGBACC :
            mod->BSIM4v0pbigbacc = value->rValue;
            mod->BSIM4v0pbigbaccGiven = TRUE;
            break;
        case  BSIM4v0_MOD_PCIGBACC :
            mod->BSIM4v0pcigbacc = value->rValue;
            mod->BSIM4v0pcigbaccGiven = TRUE;
            break;
        case  BSIM4v0_MOD_PAIGBINV :
            mod->BSIM4v0paigbinv = value->rValue;
            mod->BSIM4v0paigbinvGiven = TRUE;
            break;
        case  BSIM4v0_MOD_PBIGBINV :
            mod->BSIM4v0pbigbinv = value->rValue;
            mod->BSIM4v0pbigbinvGiven = TRUE;
            break;
        case  BSIM4v0_MOD_PCIGBINV :
            mod->BSIM4v0pcigbinv = value->rValue;
            mod->BSIM4v0pcigbinvGiven = TRUE;
            break;
        case  BSIM4v0_MOD_PNIGC :
            mod->BSIM4v0pnigc = value->rValue;
            mod->BSIM4v0pnigcGiven = TRUE;
            break;
        case  BSIM4v0_MOD_PNIGBINV :
            mod->BSIM4v0pnigbinv = value->rValue;
            mod->BSIM4v0pnigbinvGiven = TRUE;
            break;
        case  BSIM4v0_MOD_PNIGBACC :
            mod->BSIM4v0pnigbacc = value->rValue;
            mod->BSIM4v0pnigbaccGiven = TRUE;
            break;
        case  BSIM4v0_MOD_PNTOX :
            mod->BSIM4v0pntox = value->rValue;
            mod->BSIM4v0pntoxGiven = TRUE;
            break;
        case  BSIM4v0_MOD_PEIGBINV :
            mod->BSIM4v0peigbinv = value->rValue;
            mod->BSIM4v0peigbinvGiven = TRUE;
            break;
        case  BSIM4v0_MOD_PPIGCD :
            mod->BSIM4v0ppigcd = value->rValue;
            mod->BSIM4v0ppigcdGiven = TRUE;
            break;
        case  BSIM4v0_MOD_PPOXEDGE :
            mod->BSIM4v0ppoxedge = value->rValue;
            mod->BSIM4v0ppoxedgeGiven = TRUE;
            break;
        case  BSIM4v0_MOD_PXRCRG1 :
            mod->BSIM4v0pxrcrg1 = value->rValue;
            mod->BSIM4v0pxrcrg1Given = TRUE;
            break;
        case  BSIM4v0_MOD_PXRCRG2 :
            mod->BSIM4v0pxrcrg2 = value->rValue;
            mod->BSIM4v0pxrcrg2Given = TRUE;
            break;
        case  BSIM4v0_MOD_PEU :
            mod->BSIM4v0peu = value->rValue;
            mod->BSIM4v0peuGiven = TRUE;
            break;
        case  BSIM4v0_MOD_PVFB :
            mod->BSIM4v0pvfb = value->rValue;
            mod->BSIM4v0pvfbGiven = TRUE;
            break;

        case  BSIM4v0_MOD_PCGSL :
            mod->BSIM4v0pcgsl = value->rValue;
            mod->BSIM4v0pcgslGiven = TRUE;
            break;
        case  BSIM4v0_MOD_PCGDL :
            mod->BSIM4v0pcgdl = value->rValue;
            mod->BSIM4v0pcgdlGiven = TRUE;
            break;
        case  BSIM4v0_MOD_PCKAPPAS :
            mod->BSIM4v0pckappas = value->rValue;
            mod->BSIM4v0pckappasGiven = TRUE;
            break;
        case  BSIM4v0_MOD_PCKAPPAD :
            mod->BSIM4v0pckappad = value->rValue;
            mod->BSIM4v0pckappadGiven = TRUE;
            break;
        case  BSIM4v0_MOD_PCF :
            mod->BSIM4v0pcf = value->rValue;
            mod->BSIM4v0pcfGiven = TRUE;
            break;
        case  BSIM4v0_MOD_PCLC :
            mod->BSIM4v0pclc = value->rValue;
            mod->BSIM4v0pclcGiven = TRUE;
            break;
        case  BSIM4v0_MOD_PCLE :
            mod->BSIM4v0pcle = value->rValue;
            mod->BSIM4v0pcleGiven = TRUE;
            break;
        case  BSIM4v0_MOD_PVFBCV :
            mod->BSIM4v0pvfbcv = value->rValue;
            mod->BSIM4v0pvfbcvGiven = TRUE;
            break;
        case  BSIM4v0_MOD_PACDE :
            mod->BSIM4v0pacde = value->rValue;
            mod->BSIM4v0pacdeGiven = TRUE;
            break;
        case  BSIM4v0_MOD_PMOIN :
            mod->BSIM4v0pmoin = value->rValue;
            mod->BSIM4v0pmoinGiven = TRUE;
            break;
        case  BSIM4v0_MOD_PNOFF :
            mod->BSIM4v0pnoff = value->rValue;
            mod->BSIM4v0pnoffGiven = TRUE;
            break;
        case  BSIM4v0_MOD_PVOFFCV :
            mod->BSIM4v0pvoffcv = value->rValue;
            mod->BSIM4v0pvoffcvGiven = TRUE;
            break;

        case  BSIM4v0_MOD_TNOM :
            mod->BSIM4v0tnom = value->rValue + CONSTCtoK;
            mod->BSIM4v0tnomGiven = TRUE;
            break;
        case  BSIM4v0_MOD_CGSO :
            mod->BSIM4v0cgso = value->rValue;
            mod->BSIM4v0cgsoGiven = TRUE;
            break;
        case  BSIM4v0_MOD_CGDO :
            mod->BSIM4v0cgdo = value->rValue;
            mod->BSIM4v0cgdoGiven = TRUE;
            break;
        case  BSIM4v0_MOD_CGBO :
            mod->BSIM4v0cgbo = value->rValue;
            mod->BSIM4v0cgboGiven = TRUE;
            break;
        case  BSIM4v0_MOD_XPART :
            mod->BSIM4v0xpart = value->rValue;
            mod->BSIM4v0xpartGiven = TRUE;
            break;
        case  BSIM4v0_MOD_RSH :
            mod->BSIM4v0sheetResistance = value->rValue;
            mod->BSIM4v0sheetResistanceGiven = TRUE;
            break;
        case  BSIM4v0_MOD_JSS :
            mod->BSIM4v0SjctSatCurDensity = value->rValue;
            mod->BSIM4v0SjctSatCurDensityGiven = TRUE;
            break;
        case  BSIM4v0_MOD_JSWS :
            mod->BSIM4v0SjctSidewallSatCurDensity = value->rValue;
            mod->BSIM4v0SjctSidewallSatCurDensityGiven = TRUE;
            break;
        case  BSIM4v0_MOD_JSWGS :
            mod->BSIM4v0SjctGateSidewallSatCurDensity = value->rValue;
            mod->BSIM4v0SjctGateSidewallSatCurDensityGiven = TRUE;
            break;
        case  BSIM4v0_MOD_PBS :
            mod->BSIM4v0SbulkJctPotential = value->rValue;
            mod->BSIM4v0SbulkJctPotentialGiven = TRUE;
            break;
        case  BSIM4v0_MOD_MJS :
            mod->BSIM4v0SbulkJctBotGradingCoeff = value->rValue;
            mod->BSIM4v0SbulkJctBotGradingCoeffGiven = TRUE;
            break;
        case  BSIM4v0_MOD_PBSWS :
            mod->BSIM4v0SsidewallJctPotential = value->rValue;
            mod->BSIM4v0SsidewallJctPotentialGiven = TRUE;
            break;
        case  BSIM4v0_MOD_MJSWS :
            mod->BSIM4v0SbulkJctSideGradingCoeff = value->rValue;
            mod->BSIM4v0SbulkJctSideGradingCoeffGiven = TRUE;
            break;
        case  BSIM4v0_MOD_CJS :
            mod->BSIM4v0SunitAreaJctCap = value->rValue;
            mod->BSIM4v0SunitAreaJctCapGiven = TRUE;
            break;
        case  BSIM4v0_MOD_CJSWS :
            mod->BSIM4v0SunitLengthSidewallJctCap = value->rValue;
            mod->BSIM4v0SunitLengthSidewallJctCapGiven = TRUE;
            break;
        case  BSIM4v0_MOD_NJS :
            mod->BSIM4v0SjctEmissionCoeff = value->rValue;
            mod->BSIM4v0SjctEmissionCoeffGiven = TRUE;
            break;
        case  BSIM4v0_MOD_PBSWGS :
            mod->BSIM4v0SGatesidewallJctPotential = value->rValue;
            mod->BSIM4v0SGatesidewallJctPotentialGiven = TRUE;
            break;
        case  BSIM4v0_MOD_MJSWGS :
            mod->BSIM4v0SbulkJctGateSideGradingCoeff = value->rValue;
            mod->BSIM4v0SbulkJctGateSideGradingCoeffGiven = TRUE;
            break;
        case  BSIM4v0_MOD_CJSWGS :
            mod->BSIM4v0SunitLengthGateSidewallJctCap = value->rValue;
            mod->BSIM4v0SunitLengthGateSidewallJctCapGiven = TRUE;
            break;
        case  BSIM4v0_MOD_XTIS :
            mod->BSIM4v0SjctTempExponent = value->rValue;
            mod->BSIM4v0SjctTempExponentGiven = TRUE;
            break;
        case  BSIM4v0_MOD_JSD :
            mod->BSIM4v0DjctSatCurDensity = value->rValue;
            mod->BSIM4v0DjctSatCurDensityGiven = TRUE;
            break;
        case  BSIM4v0_MOD_JSWD :
            mod->BSIM4v0DjctSidewallSatCurDensity = value->rValue;
            mod->BSIM4v0DjctSidewallSatCurDensityGiven = TRUE;
            break;
        case  BSIM4v0_MOD_JSWGD :
            mod->BSIM4v0DjctGateSidewallSatCurDensity = value->rValue;
            mod->BSIM4v0DjctGateSidewallSatCurDensityGiven = TRUE;
            break;
        case  BSIM4v0_MOD_PBD :
            mod->BSIM4v0DbulkJctPotential = value->rValue;
            mod->BSIM4v0DbulkJctPotentialGiven = TRUE;
            break;
        case  BSIM4v0_MOD_MJD :
            mod->BSIM4v0DbulkJctBotGradingCoeff = value->rValue;
            mod->BSIM4v0DbulkJctBotGradingCoeffGiven = TRUE;
            break;
        case  BSIM4v0_MOD_PBSWD :
            mod->BSIM4v0DsidewallJctPotential = value->rValue;
            mod->BSIM4v0DsidewallJctPotentialGiven = TRUE;
            break;
        case  BSIM4v0_MOD_MJSWD :
            mod->BSIM4v0DbulkJctSideGradingCoeff = value->rValue;
            mod->BSIM4v0DbulkJctSideGradingCoeffGiven = TRUE;
            break;
        case  BSIM4v0_MOD_CJD :
            mod->BSIM4v0DunitAreaJctCap = value->rValue;
            mod->BSIM4v0DunitAreaJctCapGiven = TRUE;
            break;
        case  BSIM4v0_MOD_CJSWD :
            mod->BSIM4v0DunitLengthSidewallJctCap = value->rValue;
            mod->BSIM4v0DunitLengthSidewallJctCapGiven = TRUE;
            break;
        case  BSIM4v0_MOD_NJD :
            mod->BSIM4v0DjctEmissionCoeff = value->rValue;
            mod->BSIM4v0DjctEmissionCoeffGiven = TRUE;
            break;
        case  BSIM4v0_MOD_PBSWGD :
            mod->BSIM4v0DGatesidewallJctPotential = value->rValue;
            mod->BSIM4v0DGatesidewallJctPotentialGiven = TRUE;
            break;
        case  BSIM4v0_MOD_MJSWGD :
            mod->BSIM4v0DbulkJctGateSideGradingCoeff = value->rValue;
            mod->BSIM4v0DbulkJctGateSideGradingCoeffGiven = TRUE;
            break;
        case  BSIM4v0_MOD_CJSWGD :
            mod->BSIM4v0DunitLengthGateSidewallJctCap = value->rValue;
            mod->BSIM4v0DunitLengthGateSidewallJctCapGiven = TRUE;
            break;
        case  BSIM4v0_MOD_XTID :
            mod->BSIM4v0DjctTempExponent = value->rValue;
            mod->BSIM4v0DjctTempExponentGiven = TRUE;
            break;
        case  BSIM4v0_MOD_LINT :
            mod->BSIM4v0Lint = value->rValue;
            mod->BSIM4v0LintGiven = TRUE;
            break;
        case  BSIM4v0_MOD_LL :
            mod->BSIM4v0Ll = value->rValue;
            mod->BSIM4v0LlGiven = TRUE;
            break;
        case  BSIM4v0_MOD_LLC :
            mod->BSIM4v0Llc = value->rValue;
            mod->BSIM4v0LlcGiven = TRUE;
            break;
        case  BSIM4v0_MOD_LLN :
            mod->BSIM4v0Lln = value->rValue;
            mod->BSIM4v0LlnGiven = TRUE;
            break;
        case  BSIM4v0_MOD_LW :
            mod->BSIM4v0Lw = value->rValue;
            mod->BSIM4v0LwGiven = TRUE;
            break;
        case  BSIM4v0_MOD_LWC :
            mod->BSIM4v0Lwc = value->rValue;
            mod->BSIM4v0LwcGiven = TRUE;
            break;
        case  BSIM4v0_MOD_LWN :
            mod->BSIM4v0Lwn = value->rValue;
            mod->BSIM4v0LwnGiven = TRUE;
            break;
        case  BSIM4v0_MOD_LWL :
            mod->BSIM4v0Lwl = value->rValue;
            mod->BSIM4v0LwlGiven = TRUE;
            break;
        case  BSIM4v0_MOD_LWLC :
            mod->BSIM4v0Lwlc = value->rValue;
            mod->BSIM4v0LwlcGiven = TRUE;
            break;
        case  BSIM4v0_MOD_LMIN :
            mod->BSIM4v0Lmin = value->rValue;
            mod->BSIM4v0LminGiven = TRUE;
            break;
        case  BSIM4v0_MOD_LMAX :
            mod->BSIM4v0Lmax = value->rValue;
            mod->BSIM4v0LmaxGiven = TRUE;
            break;
        case  BSIM4v0_MOD_WINT :
            mod->BSIM4v0Wint = value->rValue;
            mod->BSIM4v0WintGiven = TRUE;
            break;
        case  BSIM4v0_MOD_WL :
            mod->BSIM4v0Wl = value->rValue;
            mod->BSIM4v0WlGiven = TRUE;
            break;
        case  BSIM4v0_MOD_WLC :
            mod->BSIM4v0Wlc = value->rValue;
            mod->BSIM4v0WlcGiven = TRUE;
            break;
        case  BSIM4v0_MOD_WLN :
            mod->BSIM4v0Wln = value->rValue;
            mod->BSIM4v0WlnGiven = TRUE;
            break;
        case  BSIM4v0_MOD_WW :
            mod->BSIM4v0Ww = value->rValue;
            mod->BSIM4v0WwGiven = TRUE;
            break;
        case  BSIM4v0_MOD_WWC :
            mod->BSIM4v0Wwc = value->rValue;
            mod->BSIM4v0WwcGiven = TRUE;
            break;
        case  BSIM4v0_MOD_WWN :
            mod->BSIM4v0Wwn = value->rValue;
            mod->BSIM4v0WwnGiven = TRUE;
            break;
        case  BSIM4v0_MOD_WWL :
            mod->BSIM4v0Wwl = value->rValue;
            mod->BSIM4v0WwlGiven = TRUE;
            break;
        case  BSIM4v0_MOD_WWLC :
            mod->BSIM4v0Wwlc = value->rValue;
            mod->BSIM4v0WwlcGiven = TRUE;
            break;
        case  BSIM4v0_MOD_WMIN :
            mod->BSIM4v0Wmin = value->rValue;
            mod->BSIM4v0WminGiven = TRUE;
            break;
        case  BSIM4v0_MOD_WMAX :
            mod->BSIM4v0Wmax = value->rValue;
            mod->BSIM4v0WmaxGiven = TRUE;
            break;

        case  BSIM4v0_MOD_NOIA :
            mod->BSIM4v0oxideTrapDensityA = value->rValue;
            mod->BSIM4v0oxideTrapDensityAGiven = TRUE;
            break;
        case  BSIM4v0_MOD_NOIB :
            mod->BSIM4v0oxideTrapDensityB = value->rValue;
            mod->BSIM4v0oxideTrapDensityBGiven = TRUE;
            break;
        case  BSIM4v0_MOD_NOIC :
            mod->BSIM4v0oxideTrapDensityC = value->rValue;
            mod->BSIM4v0oxideTrapDensityCGiven = TRUE;
            break;
        case  BSIM4v0_MOD_EM :
            mod->BSIM4v0em = value->rValue;
            mod->BSIM4v0emGiven = TRUE;
            break;
        case  BSIM4v0_MOD_EF :
            mod->BSIM4v0ef = value->rValue;
            mod->BSIM4v0efGiven = TRUE;
            break;
        case  BSIM4v0_MOD_AF :
            mod->BSIM4v0af = value->rValue;
            mod->BSIM4v0afGiven = TRUE;
            break;
        case  BSIM4v0_MOD_KF :
            mod->BSIM4v0kf = value->rValue;
            mod->BSIM4v0kfGiven = TRUE;
            break;
        case  BSIM4v0_MOD_NMOS  :
            if(value->iValue) {
                mod->BSIM4v0type = 1;
                mod->BSIM4v0typeGiven = TRUE;
            }
            break;
        case  BSIM4v0_MOD_PMOS  :
            if(value->iValue) {
                mod->BSIM4v0type = - 1;
                mod->BSIM4v0typeGiven = TRUE;
            }
            break;
        default:
            return(E_BADPARM);
    }
    return(OK);
}


