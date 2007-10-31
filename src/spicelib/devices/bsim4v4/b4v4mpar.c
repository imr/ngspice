/**** BSIM4.4.0  Released by Xuemei (Jane) Xi 03/04/2004 ****/

/**********
 * Copyright 2004 Regents of the University of California. All rights reserved.
 * File: b4mpar.c of BSIM4.4.0.
 * Author: 2000 Weidong Liu
 * Authors: 2001- Xuemei Xi, Jin He, Kanyu Cao, Mohan Dunga, Mansun Chan, Ali Niknejad, Chenming Hu.
 * Project Director: Prof. Chenming Hu.
 * Modified by Xuemei Xi, 04/06/2001.
 * Modified by Xuemei Xi, 10/05/2001.
 * Modified by Xuemei Xi, 11/15/2002.
 * Modified by Xuemei Xi, 05/09/2003.
 * Modified by Xuemei Xi, 03/04/2004.
 **********/

#include "ngspice.h"
#include "bsim4v4def.h"
#include "ifsim.h"
#include "sperror.h"
#include "suffix.h"

int
BSIM4V4mParam(param,value,inMod)
int param;
IFvalue *value;
GENmodel *inMod;
{
    BSIM4V4model *mod = (BSIM4V4model*)inMod;
    switch(param)
    {   case  BSIM4V4_MOD_MOBMOD :
            mod->BSIM4V4mobMod = value->iValue;
            mod->BSIM4V4mobModGiven = TRUE;
            break;
        case  BSIM4V4_MOD_BINUNIT :
            mod->BSIM4V4binUnit = value->iValue;
            mod->BSIM4V4binUnitGiven = TRUE;
            break;
        case  BSIM4V4_MOD_PARAMCHK :
            mod->BSIM4V4paramChk = value->iValue;
            mod->BSIM4V4paramChkGiven = TRUE;
            break;
        case  BSIM4V4_MOD_CAPMOD :
            mod->BSIM4V4capMod = value->iValue;
            mod->BSIM4V4capModGiven = TRUE;
            break;
        case  BSIM4V4_MOD_DIOMOD :
            mod->BSIM4V4dioMod = value->iValue;
            mod->BSIM4V4dioModGiven = TRUE;
            break;
        case  BSIM4V4_MOD_RDSMOD :
            mod->BSIM4V4rdsMod = value->iValue;
            mod->BSIM4V4rdsModGiven = TRUE;
            break;
        case  BSIM4V4_MOD_TRNQSMOD :
            mod->BSIM4V4trnqsMod = value->iValue;
            mod->BSIM4V4trnqsModGiven = TRUE;
            break;
        case  BSIM4V4_MOD_ACNQSMOD :
            mod->BSIM4V4acnqsMod = value->iValue;
            mod->BSIM4V4acnqsModGiven = TRUE;
            break;
        case  BSIM4V4_MOD_RBODYMOD :
            mod->BSIM4V4rbodyMod = value->iValue;
            mod->BSIM4V4rbodyModGiven = TRUE;
            break;
        case  BSIM4V4_MOD_RGATEMOD :
            mod->BSIM4V4rgateMod = value->iValue;
            mod->BSIM4V4rgateModGiven = TRUE;
            break;
        case  BSIM4V4_MOD_PERMOD :
            mod->BSIM4V4perMod = value->iValue;
            mod->BSIM4V4perModGiven = TRUE;
            break;
        case  BSIM4V4_MOD_GEOMOD :
            mod->BSIM4V4geoMod = value->iValue;
            mod->BSIM4V4geoModGiven = TRUE;
            break;
        case  BSIM4V4_MOD_FNOIMOD :
            mod->BSIM4V4fnoiMod = value->iValue;
            mod->BSIM4V4fnoiModGiven = TRUE;
            break;
        case  BSIM4V4_MOD_TNOIMOD :
            mod->BSIM4V4tnoiMod = value->iValue;
            mod->BSIM4V4tnoiModGiven = TRUE;
            break;
        case  BSIM4V4_MOD_IGCMOD :
            mod->BSIM4V4igcMod = value->iValue;
            mod->BSIM4V4igcModGiven = TRUE;
            break;
        case  BSIM4V4_MOD_IGBMOD :
            mod->BSIM4V4igbMod = value->iValue;
            mod->BSIM4V4igbModGiven = TRUE;
            break;
        case  BSIM4V4_MOD_TEMPMOD :
            mod->BSIM4V4tempMod = value->iValue;
            mod->BSIM4V4tempModGiven = TRUE;
            break;
        case  BSIM4V4_MOD_VERSION :
            mod->BSIM4V4version = value->sValue;
            mod->BSIM4V4versionGiven = TRUE;
            break;
        case  BSIM4V4_MOD_TOXREF :
            mod->BSIM4V4toxref = value->rValue;
            mod->BSIM4V4toxrefGiven = TRUE;
            break;
        case  BSIM4V4_MOD_TOXE :
            mod->BSIM4V4toxe = value->rValue;
            mod->BSIM4V4toxeGiven = TRUE;
            break;
        case  BSIM4V4_MOD_TOXP :
            mod->BSIM4V4toxp = value->rValue;
            mod->BSIM4V4toxpGiven = TRUE;
            break;
        case  BSIM4V4_MOD_TOXM :
            mod->BSIM4V4toxm = value->rValue;
            mod->BSIM4V4toxmGiven = TRUE;
            break;
        case  BSIM4V4_MOD_DTOX :
            mod->BSIM4V4dtox = value->rValue;
            mod->BSIM4V4dtoxGiven = TRUE;
            break;
        case  BSIM4V4_MOD_EPSROX :
            mod->BSIM4V4epsrox = value->rValue;
            mod->BSIM4V4epsroxGiven = TRUE;
            break;

        case  BSIM4V4_MOD_CDSC :
            mod->BSIM4V4cdsc = value->rValue;
            mod->BSIM4V4cdscGiven = TRUE;
            break;
        case  BSIM4V4_MOD_CDSCB :
            mod->BSIM4V4cdscb = value->rValue;
            mod->BSIM4V4cdscbGiven = TRUE;
            break;

        case  BSIM4V4_MOD_CDSCD :
            mod->BSIM4V4cdscd = value->rValue;
            mod->BSIM4V4cdscdGiven = TRUE;
            break;

        case  BSIM4V4_MOD_CIT :
            mod->BSIM4V4cit = value->rValue;
            mod->BSIM4V4citGiven = TRUE;
            break;
        case  BSIM4V4_MOD_NFACTOR :
            mod->BSIM4V4nfactor = value->rValue;
            mod->BSIM4V4nfactorGiven = TRUE;
            break;
        case BSIM4V4_MOD_XJ:
            mod->BSIM4V4xj = value->rValue;
            mod->BSIM4V4xjGiven = TRUE;
            break;
        case BSIM4V4_MOD_VSAT:
            mod->BSIM4V4vsat = value->rValue;
            mod->BSIM4V4vsatGiven = TRUE;
            break;
        case BSIM4V4_MOD_A0:
            mod->BSIM4V4a0 = value->rValue;
            mod->BSIM4V4a0Given = TRUE;
            break;
        
        case BSIM4V4_MOD_AGS:
            mod->BSIM4V4ags= value->rValue;
            mod->BSIM4V4agsGiven = TRUE;
            break;
        
        case BSIM4V4_MOD_A1:
            mod->BSIM4V4a1 = value->rValue;
            mod->BSIM4V4a1Given = TRUE;
            break;
        case BSIM4V4_MOD_A2:
            mod->BSIM4V4a2 = value->rValue;
            mod->BSIM4V4a2Given = TRUE;
            break;
        case BSIM4V4_MOD_AT:
            mod->BSIM4V4at = value->rValue;
            mod->BSIM4V4atGiven = TRUE;
            break;
        case BSIM4V4_MOD_KETA:
            mod->BSIM4V4keta = value->rValue;
            mod->BSIM4V4ketaGiven = TRUE;
            break;    
        case BSIM4V4_MOD_NSUB:
            mod->BSIM4V4nsub = value->rValue;
            mod->BSIM4V4nsubGiven = TRUE;
            break;
        case BSIM4V4_MOD_NDEP:
            mod->BSIM4V4ndep = value->rValue;
            mod->BSIM4V4ndepGiven = TRUE;
	    if (mod->BSIM4V4ndep > 1.0e20)
		mod->BSIM4V4ndep *= 1.0e-6;
            break;
        case BSIM4V4_MOD_NSD:
            mod->BSIM4V4nsd = value->rValue;
            mod->BSIM4V4nsdGiven = TRUE;
            if (mod->BSIM4V4nsd > 1.0e23)
                mod->BSIM4V4nsd *= 1.0e-6;
            break;
        case BSIM4V4_MOD_NGATE:
            mod->BSIM4V4ngate = value->rValue;
            mod->BSIM4V4ngateGiven = TRUE;
	    if (mod->BSIM4V4ngate > 1.0e23)
		mod->BSIM4V4ngate *= 1.0e-6;
            break;
        case BSIM4V4_MOD_GAMMA1:
            mod->BSIM4V4gamma1 = value->rValue;
            mod->BSIM4V4gamma1Given = TRUE;
            break;
        case BSIM4V4_MOD_GAMMA2:
            mod->BSIM4V4gamma2 = value->rValue;
            mod->BSIM4V4gamma2Given = TRUE;
            break;
        case BSIM4V4_MOD_VBX:
            mod->BSIM4V4vbx = value->rValue;
            mod->BSIM4V4vbxGiven = TRUE;
            break;
        case BSIM4V4_MOD_VBM:
            mod->BSIM4V4vbm = value->rValue;
            mod->BSIM4V4vbmGiven = TRUE;
            break;
        case BSIM4V4_MOD_XT:
            mod->BSIM4V4xt = value->rValue;
            mod->BSIM4V4xtGiven = TRUE;
            break;
        case  BSIM4V4_MOD_K1:
            mod->BSIM4V4k1 = value->rValue;
            mod->BSIM4V4k1Given = TRUE;
            break;
        case  BSIM4V4_MOD_KT1:
            mod->BSIM4V4kt1 = value->rValue;
            mod->BSIM4V4kt1Given = TRUE;
            break;
        case  BSIM4V4_MOD_KT1L:
            mod->BSIM4V4kt1l = value->rValue;
            mod->BSIM4V4kt1lGiven = TRUE;
            break;
        case  BSIM4V4_MOD_KT2:
            mod->BSIM4V4kt2 = value->rValue;
            mod->BSIM4V4kt2Given = TRUE;
            break;
        case  BSIM4V4_MOD_K2:
            mod->BSIM4V4k2 = value->rValue;
            mod->BSIM4V4k2Given = TRUE;
            break;
        case  BSIM4V4_MOD_K3:
            mod->BSIM4V4k3 = value->rValue;
            mod->BSIM4V4k3Given = TRUE;
            break;
        case  BSIM4V4_MOD_K3B:
            mod->BSIM4V4k3b = value->rValue;
            mod->BSIM4V4k3bGiven = TRUE;
            break;
        case  BSIM4V4_MOD_LPE0:
            mod->BSIM4V4lpe0 = value->rValue;
            mod->BSIM4V4lpe0Given = TRUE;
            break;
        case  BSIM4V4_MOD_LPEB:
            mod->BSIM4V4lpeb = value->rValue;
            mod->BSIM4V4lpebGiven = TRUE;
            break;
        case  BSIM4V4_MOD_DVTP0:
            mod->BSIM4V4dvtp0 = value->rValue;
            mod->BSIM4V4dvtp0Given = TRUE;
            break;
        case  BSIM4V4_MOD_DVTP1:
            mod->BSIM4V4dvtp1 = value->rValue;
            mod->BSIM4V4dvtp1Given = TRUE;
            break;
        case  BSIM4V4_MOD_W0:
            mod->BSIM4V4w0 = value->rValue;
            mod->BSIM4V4w0Given = TRUE;
            break;
        case  BSIM4V4_MOD_DVT0:               
            mod->BSIM4V4dvt0 = value->rValue;
            mod->BSIM4V4dvt0Given = TRUE;
            break;
        case  BSIM4V4_MOD_DVT1:             
            mod->BSIM4V4dvt1 = value->rValue;
            mod->BSIM4V4dvt1Given = TRUE;
            break;
        case  BSIM4V4_MOD_DVT2:             
            mod->BSIM4V4dvt2 = value->rValue;
            mod->BSIM4V4dvt2Given = TRUE;
            break;
        case  BSIM4V4_MOD_DVT0W:               
            mod->BSIM4V4dvt0w = value->rValue;
            mod->BSIM4V4dvt0wGiven = TRUE;
            break;
        case  BSIM4V4_MOD_DVT1W:             
            mod->BSIM4V4dvt1w = value->rValue;
            mod->BSIM4V4dvt1wGiven = TRUE;
            break;
        case  BSIM4V4_MOD_DVT2W:             
            mod->BSIM4V4dvt2w = value->rValue;
            mod->BSIM4V4dvt2wGiven = TRUE;
            break;
        case  BSIM4V4_MOD_DROUT:             
            mod->BSIM4V4drout = value->rValue;
            mod->BSIM4V4droutGiven = TRUE;
            break;
        case  BSIM4V4_MOD_DSUB:             
            mod->BSIM4V4dsub = value->rValue;
            mod->BSIM4V4dsubGiven = TRUE;
            break;
        case BSIM4V4_MOD_VTH0:
            mod->BSIM4V4vth0 = value->rValue;
            mod->BSIM4V4vth0Given = TRUE;
            break;
        case BSIM4V4_MOD_EU:
            mod->BSIM4V4eu = value->rValue;
            mod->BSIM4V4euGiven = TRUE;
            break;
        case BSIM4V4_MOD_UA:
            mod->BSIM4V4ua = value->rValue;
            mod->BSIM4V4uaGiven = TRUE;
            break;
        case BSIM4V4_MOD_UA1:
            mod->BSIM4V4ua1 = value->rValue;
            mod->BSIM4V4ua1Given = TRUE;
            break;
        case BSIM4V4_MOD_UB:
            mod->BSIM4V4ub = value->rValue;
            mod->BSIM4V4ubGiven = TRUE;
            break;
        case BSIM4V4_MOD_UB1:
            mod->BSIM4V4ub1 = value->rValue;
            mod->BSIM4V4ub1Given = TRUE;
            break;
        case BSIM4V4_MOD_UC:
            mod->BSIM4V4uc = value->rValue;
            mod->BSIM4V4ucGiven = TRUE;
            break;
        case BSIM4V4_MOD_UC1:
            mod->BSIM4V4uc1 = value->rValue;
            mod->BSIM4V4uc1Given = TRUE;
            break;
        case  BSIM4V4_MOD_U0 :
            mod->BSIM4V4u0 = value->rValue;
            mod->BSIM4V4u0Given = TRUE;
            break;
        case  BSIM4V4_MOD_UTE :
            mod->BSIM4V4ute = value->rValue;
            mod->BSIM4V4uteGiven = TRUE;
            break;
        case BSIM4V4_MOD_VOFF:
            mod->BSIM4V4voff = value->rValue;
            mod->BSIM4V4voffGiven = TRUE;
            break;
        case BSIM4V4_MOD_VOFFL:
            mod->BSIM4V4voffl = value->rValue;
            mod->BSIM4V4vofflGiven = TRUE;
            break;
        case BSIM4V4_MOD_MINV:
            mod->BSIM4V4minv = value->rValue;
            mod->BSIM4V4minvGiven = TRUE;
            break;
        case BSIM4V4_MOD_FPROUT:
            mod->BSIM4V4fprout = value->rValue;
            mod->BSIM4V4fproutGiven = TRUE;
            break;
        case BSIM4V4_MOD_PDITS:
            mod->BSIM4V4pdits = value->rValue;
            mod->BSIM4V4pditsGiven = TRUE;
            break;
        case BSIM4V4_MOD_PDITSD:
            mod->BSIM4V4pditsd = value->rValue;
            mod->BSIM4V4pditsdGiven = TRUE;
            break;
        case BSIM4V4_MOD_PDITSL:
            mod->BSIM4V4pditsl = value->rValue;
            mod->BSIM4V4pditslGiven = TRUE;
            break;
        case  BSIM4V4_MOD_DELTA :
            mod->BSIM4V4delta = value->rValue;
            mod->BSIM4V4deltaGiven = TRUE;
            break;
        case BSIM4V4_MOD_RDSW:
            mod->BSIM4V4rdsw = value->rValue;
            mod->BSIM4V4rdswGiven = TRUE;
            break;                     
        case BSIM4V4_MOD_RDSWMIN:
            mod->BSIM4V4rdswmin = value->rValue;
            mod->BSIM4V4rdswminGiven = TRUE;
            break;
        case BSIM4V4_MOD_RDWMIN:
            mod->BSIM4V4rdwmin = value->rValue;
            mod->BSIM4V4rdwminGiven = TRUE;
            break;
        case BSIM4V4_MOD_RSWMIN:
            mod->BSIM4V4rswmin = value->rValue;
            mod->BSIM4V4rswminGiven = TRUE;
            break;
        case BSIM4V4_MOD_RDW:
            mod->BSIM4V4rdw = value->rValue;
            mod->BSIM4V4rdwGiven = TRUE;
            break;
        case BSIM4V4_MOD_RSW:
            mod->BSIM4V4rsw = value->rValue;
            mod->BSIM4V4rswGiven = TRUE;
            break;
        case BSIM4V4_MOD_PRWG:
            mod->BSIM4V4prwg = value->rValue;
            mod->BSIM4V4prwgGiven = TRUE;
            break;                     
        case BSIM4V4_MOD_PRWB:
            mod->BSIM4V4prwb = value->rValue;
            mod->BSIM4V4prwbGiven = TRUE;
            break;                     
        case BSIM4V4_MOD_PRT:
            mod->BSIM4V4prt = value->rValue;
            mod->BSIM4V4prtGiven = TRUE;
            break;                     
        case BSIM4V4_MOD_ETA0:
            mod->BSIM4V4eta0 = value->rValue;
            mod->BSIM4V4eta0Given = TRUE;
            break;                 
        case BSIM4V4_MOD_ETAB:
            mod->BSIM4V4etab = value->rValue;
            mod->BSIM4V4etabGiven = TRUE;
            break;                 
        case BSIM4V4_MOD_PCLM:
            mod->BSIM4V4pclm = value->rValue;
            mod->BSIM4V4pclmGiven = TRUE;
            break;                 
        case BSIM4V4_MOD_PDIBL1:
            mod->BSIM4V4pdibl1 = value->rValue;
            mod->BSIM4V4pdibl1Given = TRUE;
            break;                 
        case BSIM4V4_MOD_PDIBL2:
            mod->BSIM4V4pdibl2 = value->rValue;
            mod->BSIM4V4pdibl2Given = TRUE;
            break;                 
        case BSIM4V4_MOD_PDIBLB:
            mod->BSIM4V4pdiblb = value->rValue;
            mod->BSIM4V4pdiblbGiven = TRUE;
            break;                 
        case BSIM4V4_MOD_PSCBE1:
            mod->BSIM4V4pscbe1 = value->rValue;
            mod->BSIM4V4pscbe1Given = TRUE;
            break;                 
        case BSIM4V4_MOD_PSCBE2:
            mod->BSIM4V4pscbe2 = value->rValue;
            mod->BSIM4V4pscbe2Given = TRUE;
            break;                 
        case BSIM4V4_MOD_PVAG:
            mod->BSIM4V4pvag = value->rValue;
            mod->BSIM4V4pvagGiven = TRUE;
            break;                 
        case  BSIM4V4_MOD_WR :
            mod->BSIM4V4wr = value->rValue;
            mod->BSIM4V4wrGiven = TRUE;
            break;
        case  BSIM4V4_MOD_DWG :
            mod->BSIM4V4dwg = value->rValue;
            mod->BSIM4V4dwgGiven = TRUE;
            break;
        case  BSIM4V4_MOD_DWB :
            mod->BSIM4V4dwb = value->rValue;
            mod->BSIM4V4dwbGiven = TRUE;
            break;
        case  BSIM4V4_MOD_B0 :
            mod->BSIM4V4b0 = value->rValue;
            mod->BSIM4V4b0Given = TRUE;
            break;
        case  BSIM4V4_MOD_B1 :
            mod->BSIM4V4b1 = value->rValue;
            mod->BSIM4V4b1Given = TRUE;
            break;
        case  BSIM4V4_MOD_ALPHA0 :
            mod->BSIM4V4alpha0 = value->rValue;
            mod->BSIM4V4alpha0Given = TRUE;
            break;
        case  BSIM4V4_MOD_ALPHA1 :
            mod->BSIM4V4alpha1 = value->rValue;
            mod->BSIM4V4alpha1Given = TRUE;
            break;
        case  BSIM4V4_MOD_AGIDL :
            mod->BSIM4V4agidl = value->rValue;
            mod->BSIM4V4agidlGiven = TRUE;
            break;
        case  BSIM4V4_MOD_BGIDL :
            mod->BSIM4V4bgidl = value->rValue;
            mod->BSIM4V4bgidlGiven = TRUE;
            break;
        case  BSIM4V4_MOD_CGIDL :
            mod->BSIM4V4cgidl = value->rValue;
            mod->BSIM4V4cgidlGiven = TRUE;
            break;
        case  BSIM4V4_MOD_PHIN :
            mod->BSIM4V4phin = value->rValue;
            mod->BSIM4V4phinGiven = TRUE;
            break;
        case  BSIM4V4_MOD_EGIDL :
            mod->BSIM4V4egidl = value->rValue;
            mod->BSIM4V4egidlGiven = TRUE;
            break;
        case  BSIM4V4_MOD_AIGC :
            mod->BSIM4V4aigc = value->rValue;
            mod->BSIM4V4aigcGiven = TRUE;
            break;
        case  BSIM4V4_MOD_BIGC :
            mod->BSIM4V4bigc = value->rValue;
            mod->BSIM4V4bigcGiven = TRUE;
            break;
        case  BSIM4V4_MOD_CIGC :
            mod->BSIM4V4cigc = value->rValue;
            mod->BSIM4V4cigcGiven = TRUE;
            break;
        case  BSIM4V4_MOD_AIGSD :
            mod->BSIM4V4aigsd = value->rValue;
            mod->BSIM4V4aigsdGiven = TRUE;
            break;
        case  BSIM4V4_MOD_BIGSD :
            mod->BSIM4V4bigsd = value->rValue;
            mod->BSIM4V4bigsdGiven = TRUE;
            break;
        case  BSIM4V4_MOD_CIGSD :
            mod->BSIM4V4cigsd = value->rValue;
            mod->BSIM4V4cigsdGiven = TRUE;
            break;
        case  BSIM4V4_MOD_AIGBACC :
            mod->BSIM4V4aigbacc = value->rValue;
            mod->BSIM4V4aigbaccGiven = TRUE;
            break;
        case  BSIM4V4_MOD_BIGBACC :
            mod->BSIM4V4bigbacc = value->rValue;
            mod->BSIM4V4bigbaccGiven = TRUE;
            break;
        case  BSIM4V4_MOD_CIGBACC :
            mod->BSIM4V4cigbacc = value->rValue;
            mod->BSIM4V4cigbaccGiven = TRUE;
            break;
        case  BSIM4V4_MOD_AIGBINV :
            mod->BSIM4V4aigbinv = value->rValue;
            mod->BSIM4V4aigbinvGiven = TRUE;
            break;
        case  BSIM4V4_MOD_BIGBINV :
            mod->BSIM4V4bigbinv = value->rValue;
            mod->BSIM4V4bigbinvGiven = TRUE;
            break;
        case  BSIM4V4_MOD_CIGBINV :
            mod->BSIM4V4cigbinv = value->rValue;
            mod->BSIM4V4cigbinvGiven = TRUE;
            break;
        case  BSIM4V4_MOD_NIGC :
            mod->BSIM4V4nigc = value->rValue;
            mod->BSIM4V4nigcGiven = TRUE;
            break;
        case  BSIM4V4_MOD_NIGBINV :
            mod->BSIM4V4nigbinv = value->rValue;
            mod->BSIM4V4nigbinvGiven = TRUE;
            break;
        case  BSIM4V4_MOD_NIGBACC :
            mod->BSIM4V4nigbacc = value->rValue;
            mod->BSIM4V4nigbaccGiven = TRUE;
            break;
        case  BSIM4V4_MOD_NTOX :
            mod->BSIM4V4ntox = value->rValue;
            mod->BSIM4V4ntoxGiven = TRUE;
            break;
        case  BSIM4V4_MOD_EIGBINV :
            mod->BSIM4V4eigbinv = value->rValue;
            mod->BSIM4V4eigbinvGiven = TRUE;
            break;
        case  BSIM4V4_MOD_PIGCD :
            mod->BSIM4V4pigcd = value->rValue;
            mod->BSIM4V4pigcdGiven = TRUE;
            break;
        case  BSIM4V4_MOD_POXEDGE :
            mod->BSIM4V4poxedge = value->rValue;
            mod->BSIM4V4poxedgeGiven = TRUE;
            break;
        case  BSIM4V4_MOD_XRCRG1 :
            mod->BSIM4V4xrcrg1 = value->rValue;
            mod->BSIM4V4xrcrg1Given = TRUE;
            break;
        case  BSIM4V4_MOD_XRCRG2 :
            mod->BSIM4V4xrcrg2 = value->rValue;
            mod->BSIM4V4xrcrg2Given = TRUE;
            break;
        case  BSIM4V4_MOD_LAMBDA :
            mod->BSIM4V4lambda = value->rValue;
            mod->BSIM4V4lambdaGiven = TRUE;
            break;
        case  BSIM4V4_MOD_VTL :
            mod->BSIM4V4vtl = value->rValue;
            mod->BSIM4V4vtlGiven = TRUE;
            break;
        case  BSIM4V4_MOD_XN:
            mod->BSIM4V4xn = value->rValue;
            mod->BSIM4V4xnGiven = TRUE;
            break;
        case  BSIM4V4_MOD_LC:
            mod->BSIM4V4lc = value->rValue;
            mod->BSIM4V4lcGiven = TRUE;
            break;
        case  BSIM4V4_MOD_TNOIA :
            mod->BSIM4V4tnoia = value->rValue;
            mod->BSIM4V4tnoiaGiven = TRUE;
            break;
        case  BSIM4V4_MOD_TNOIB :
            mod->BSIM4V4tnoib = value->rValue;
            mod->BSIM4V4tnoibGiven = TRUE;
            break;
        case  BSIM4V4_MOD_RNOIA :
            mod->BSIM4V4rnoia = value->rValue;
            mod->BSIM4V4rnoiaGiven = TRUE;
            break;
        case  BSIM4V4_MOD_RNOIB :
            mod->BSIM4V4rnoib = value->rValue;
            mod->BSIM4V4rnoibGiven = TRUE;
            break;
        case  BSIM4V4_MOD_NTNOI :
            mod->BSIM4V4ntnoi = value->rValue;
            mod->BSIM4V4ntnoiGiven = TRUE;
            break;
        case  BSIM4V4_MOD_VFBSDOFF:
            mod->BSIM4V4vfbsdoff = value->rValue;
            mod->BSIM4V4vfbsdoffGiven = TRUE;
            break;
        case  BSIM4V4_MOD_LINTNOI:
            mod->BSIM4V4lintnoi = value->rValue;
            mod->BSIM4V4lintnoiGiven = TRUE;
            break;

        /* stress effect */
        case  BSIM4V4_MOD_SAREF :
            mod->BSIM4V4saref = value->rValue;
            mod->BSIM4V4sarefGiven = TRUE;
            break;
        case  BSIM4V4_MOD_SBREF :
            mod->BSIM4V4sbref = value->rValue;
            mod->BSIM4V4sbrefGiven = TRUE;
            break;
        case  BSIM4V4_MOD_WLOD :
            mod->BSIM4V4wlod = value->rValue;
            mod->BSIM4V4wlodGiven = TRUE;
            break;
        case  BSIM4V4_MOD_KU0 :
            mod->BSIM4V4ku0 = value->rValue;
            mod->BSIM4V4ku0Given = TRUE;
            break;
        case  BSIM4V4_MOD_KVSAT :
            mod->BSIM4V4kvsat = value->rValue;
            mod->BSIM4V4kvsatGiven = TRUE;
            break;
        case  BSIM4V4_MOD_KVTH0 :
            mod->BSIM4V4kvth0 = value->rValue;
            mod->BSIM4V4kvth0Given = TRUE;
            break;
        case  BSIM4V4_MOD_TKU0 :
            mod->BSIM4V4tku0 = value->rValue;
            mod->BSIM4V4tku0Given = TRUE;
            break;
        case  BSIM4V4_MOD_LLODKU0 :
            mod->BSIM4V4llodku0 = value->rValue;
            mod->BSIM4V4llodku0Given = TRUE;
            break;
        case  BSIM4V4_MOD_WLODKU0 :
            mod->BSIM4V4wlodku0 = value->rValue;
            mod->BSIM4V4wlodku0Given = TRUE;
            break;
        case  BSIM4V4_MOD_LLODVTH :
            mod->BSIM4V4llodvth = value->rValue;
            mod->BSIM4V4llodvthGiven = TRUE;
            break;
        case  BSIM4V4_MOD_WLODVTH :
            mod->BSIM4V4wlodvth = value->rValue;
            mod->BSIM4V4wlodvthGiven = TRUE;
            break;
        case  BSIM4V4_MOD_LKU0 :
            mod->BSIM4V4lku0 = value->rValue;
            mod->BSIM4V4lku0Given = TRUE;
            break;
        case  BSIM4V4_MOD_WKU0 :
            mod->BSIM4V4wku0 = value->rValue;
            mod->BSIM4V4wku0Given = TRUE;
            break;
        case  BSIM4V4_MOD_PKU0 :
            mod->BSIM4V4pku0 = value->rValue;
            mod->BSIM4V4pku0Given = TRUE;
            break;
        case  BSIM4V4_MOD_LKVTH0 :
            mod->BSIM4V4lkvth0 = value->rValue;
            mod->BSIM4V4lkvth0Given = TRUE;
            break;
        case  BSIM4V4_MOD_WKVTH0 :
            mod->BSIM4V4wkvth0 = value->rValue;
            mod->BSIM4V4wkvth0Given = TRUE;
            break;
        case  BSIM4V4_MOD_PKVTH0 :
            mod->BSIM4V4pkvth0 = value->rValue;
            mod->BSIM4V4pkvth0Given = TRUE;
            break;
        case  BSIM4V4_MOD_STK2 :
            mod->BSIM4V4stk2 = value->rValue;
            mod->BSIM4V4stk2Given = TRUE;
            break;
        case  BSIM4V4_MOD_LODK2 :
            mod->BSIM4V4lodk2 = value->rValue;
            mod->BSIM4V4lodk2Given = TRUE;
            break;
        case  BSIM4V4_MOD_STETA0 :
            mod->BSIM4V4steta0 = value->rValue;
            mod->BSIM4V4steta0Given = TRUE;
            break;
        case  BSIM4V4_MOD_LODETA0 :
            mod->BSIM4V4lodeta0 = value->rValue;
            mod->BSIM4V4lodeta0Given = TRUE;
            break;

        case  BSIM4V4_MOD_BETA0 :
            mod->BSIM4V4beta0 = value->rValue;
            mod->BSIM4V4beta0Given = TRUE;
            break;
        case  BSIM4V4_MOD_IJTHDFWD :
            mod->BSIM4V4ijthdfwd = value->rValue;
            mod->BSIM4V4ijthdfwdGiven = TRUE;
            break;
        case  BSIM4V4_MOD_IJTHSFWD :
            mod->BSIM4V4ijthsfwd = value->rValue;
            mod->BSIM4V4ijthsfwdGiven = TRUE;
            break;
        case  BSIM4V4_MOD_IJTHDREV :
            mod->BSIM4V4ijthdrev = value->rValue;
            mod->BSIM4V4ijthdrevGiven = TRUE;
            break;
        case  BSIM4V4_MOD_IJTHSREV :
            mod->BSIM4V4ijthsrev = value->rValue;
            mod->BSIM4V4ijthsrevGiven = TRUE;
            break;
        case  BSIM4V4_MOD_XJBVD :
            mod->BSIM4V4xjbvd = value->rValue;
            mod->BSIM4V4xjbvdGiven = TRUE;
            break;
        case  BSIM4V4_MOD_XJBVS :
            mod->BSIM4V4xjbvs = value->rValue;
            mod->BSIM4V4xjbvsGiven = TRUE;
            break;
        case  BSIM4V4_MOD_BVD :
            mod->BSIM4V4bvd = value->rValue;
            mod->BSIM4V4bvdGiven = TRUE;
            break;
        case  BSIM4V4_MOD_BVS :
            mod->BSIM4V4bvs = value->rValue;
            mod->BSIM4V4bvsGiven = TRUE;
            break;
        
        /* reverse diode */
        case  BSIM4V4_MOD_JTSS :
            mod->BSIM4V4jtss = value->rValue;
            mod->BSIM4V4jtssGiven = TRUE;
            break;
        case  BSIM4V4_MOD_JTSD :
            mod->BSIM4V4jtsd = value->rValue;
            mod->BSIM4V4jtsdGiven = TRUE;
            break;
        case  BSIM4V4_MOD_JTSSWS :
            mod->BSIM4V4jtssws = value->rValue;
            mod->BSIM4V4jtsswsGiven = TRUE;
            break;
        case  BSIM4V4_MOD_JTSSWD :
            mod->BSIM4V4jtsswd = value->rValue;
            mod->BSIM4V4jtsswdGiven = TRUE;
            break;
        case  BSIM4V4_MOD_JTSSWGS :
            mod->BSIM4V4jtsswgs = value->rValue;
            mod->BSIM4V4jtsswgsGiven = TRUE;
            break;
        case  BSIM4V4_MOD_JTSSWGD :
            mod->BSIM4V4jtsswgd = value->rValue;
            mod->BSIM4V4jtsswgdGiven = TRUE;
            break;
        case  BSIM4V4_MOD_NJTS :
            mod->BSIM4V4njts = value->rValue;
            mod->BSIM4V4njtsGiven = TRUE;
            break;
        case  BSIM4V4_MOD_NJTSSW :
            mod->BSIM4V4njtssw = value->rValue;
            mod->BSIM4V4njtsswGiven = TRUE;
            break;
        case  BSIM4V4_MOD_NJTSSWG :
            mod->BSIM4V4njtsswg = value->rValue;
            mod->BSIM4V4njtsswgGiven = TRUE;
            break;
        case  BSIM4V4_MOD_XTSS :
            mod->BSIM4V4xtss = value->rValue;
            mod->BSIM4V4xtssGiven = TRUE;
            break;
        case  BSIM4V4_MOD_XTSD :
            mod->BSIM4V4xtsd = value->rValue;
            mod->BSIM4V4xtsdGiven = TRUE;
            break;
        case  BSIM4V4_MOD_XTSSWS :
            mod->BSIM4V4xtssws = value->rValue;
            mod->BSIM4V4xtsswsGiven = TRUE;
            break;
        case  BSIM4V4_MOD_XTSSWD :
            mod->BSIM4V4xtsswd = value->rValue;
            mod->BSIM4V4xtsswdGiven = TRUE;
            break;
        case  BSIM4V4_MOD_XTSSWGS :
            mod->BSIM4V4xtsswgs = value->rValue;
            mod->BSIM4V4xtsswgsGiven = TRUE;
            break;
        case  BSIM4V4_MOD_XTSSWGD :
            mod->BSIM4V4xtsswgd = value->rValue;
            mod->BSIM4V4xtsswgdGiven = TRUE;
            break;
        case  BSIM4V4_MOD_TNJTS :
            mod->BSIM4V4tnjts = value->rValue;
            mod->BSIM4V4tnjtsGiven = TRUE;
            break;
        case  BSIM4V4_MOD_TNJTSSW :
            mod->BSIM4V4tnjtssw = value->rValue;
            mod->BSIM4V4tnjtsswGiven = TRUE;
            break;
        case  BSIM4V4_MOD_TNJTSSWG :
            mod->BSIM4V4tnjtsswg = value->rValue;
            mod->BSIM4V4tnjtsswgGiven = TRUE;
            break;
        case  BSIM4V4_MOD_VTSS :
            mod->BSIM4V4vtss = value->rValue;
            mod->BSIM4V4vtssGiven = TRUE;
            break;
        case  BSIM4V4_MOD_VTSD :
            mod->BSIM4V4vtsd = value->rValue;
            mod->BSIM4V4vtsdGiven = TRUE;
            break;
        case  BSIM4V4_MOD_VTSSWS :
            mod->BSIM4V4vtssws = value->rValue;
            mod->BSIM4V4vtsswsGiven = TRUE;
            break;
        case  BSIM4V4_MOD_VTSSWD :
            mod->BSIM4V4vtsswd = value->rValue;
            mod->BSIM4V4vtsswdGiven = TRUE;
            break;
        case  BSIM4V4_MOD_VTSSWGS :
            mod->BSIM4V4vtsswgs = value->rValue;
            mod->BSIM4V4vtsswgsGiven = TRUE;
            break;
        case  BSIM4V4_MOD_VTSSWGD :
            mod->BSIM4V4vtsswgd = value->rValue;
            mod->BSIM4V4vtsswgdGiven = TRUE;
            break;

        case  BSIM4V4_MOD_VFB :
            mod->BSIM4V4vfb = value->rValue;
            mod->BSIM4V4vfbGiven = TRUE;
            break;

        case  BSIM4V4_MOD_GBMIN :
            mod->BSIM4V4gbmin = value->rValue;
            mod->BSIM4V4gbminGiven = TRUE;
            break;
        case  BSIM4V4_MOD_RBDB :
            mod->BSIM4V4rbdb = value->rValue;
            mod->BSIM4V4rbdbGiven = TRUE;
            break;
        case  BSIM4V4_MOD_RBPB :
            mod->BSIM4V4rbpb = value->rValue;
            mod->BSIM4V4rbpbGiven = TRUE;
            break;
        case  BSIM4V4_MOD_RBSB :
            mod->BSIM4V4rbsb = value->rValue;
            mod->BSIM4V4rbsbGiven = TRUE;
            break;
        case  BSIM4V4_MOD_RBPS :
            mod->BSIM4V4rbps = value->rValue;
            mod->BSIM4V4rbpsGiven = TRUE;
            break;
        case  BSIM4V4_MOD_RBPD :
            mod->BSIM4V4rbpd = value->rValue;
            mod->BSIM4V4rbpdGiven = TRUE;
            break;

        case  BSIM4V4_MOD_CGSL :
            mod->BSIM4V4cgsl = value->rValue;
            mod->BSIM4V4cgslGiven = TRUE;
            break;
        case  BSIM4V4_MOD_CGDL :
            mod->BSIM4V4cgdl = value->rValue;
            mod->BSIM4V4cgdlGiven = TRUE;
            break;
        case  BSIM4V4_MOD_CKAPPAS :
            mod->BSIM4V4ckappas = value->rValue;
            mod->BSIM4V4ckappasGiven = TRUE;
            break;
        case  BSIM4V4_MOD_CKAPPAD :
            mod->BSIM4V4ckappad = value->rValue;
            mod->BSIM4V4ckappadGiven = TRUE;
            break;
        case  BSIM4V4_MOD_CF :
            mod->BSIM4V4cf = value->rValue;
            mod->BSIM4V4cfGiven = TRUE;
            break;
        case  BSIM4V4_MOD_CLC :
            mod->BSIM4V4clc = value->rValue;
            mod->BSIM4V4clcGiven = TRUE;
            break;
        case  BSIM4V4_MOD_CLE :
            mod->BSIM4V4cle = value->rValue;
            mod->BSIM4V4cleGiven = TRUE;
            break;
        case  BSIM4V4_MOD_DWC :
            mod->BSIM4V4dwc = value->rValue;
            mod->BSIM4V4dwcGiven = TRUE;
            break;
        case  BSIM4V4_MOD_DLC :
            mod->BSIM4V4dlc = value->rValue;
            mod->BSIM4V4dlcGiven = TRUE;
            break;
        case  BSIM4V4_MOD_XW :
            mod->BSIM4V4xw = value->rValue;
            mod->BSIM4V4xwGiven = TRUE;
            break;
        case  BSIM4V4_MOD_XL :
            mod->BSIM4V4xl = value->rValue;
            mod->BSIM4V4xlGiven = TRUE;
            break;
        case  BSIM4V4_MOD_DLCIG :
            mod->BSIM4V4dlcig = value->rValue;
            mod->BSIM4V4dlcigGiven = TRUE;
            break;
        case  BSIM4V4_MOD_DWJ :
            mod->BSIM4V4dwj = value->rValue;
            mod->BSIM4V4dwjGiven = TRUE;
            break;
        case  BSIM4V4_MOD_VFBCV :
            mod->BSIM4V4vfbcv = value->rValue;
            mod->BSIM4V4vfbcvGiven = TRUE;
            break;
        case  BSIM4V4_MOD_ACDE :
            mod->BSIM4V4acde = value->rValue;
            mod->BSIM4V4acdeGiven = TRUE;
            break;
        case  BSIM4V4_MOD_MOIN :
            mod->BSIM4V4moin = value->rValue;
            mod->BSIM4V4moinGiven = TRUE;
            break;
        case  BSIM4V4_MOD_NOFF :
            mod->BSIM4V4noff = value->rValue;
            mod->BSIM4V4noffGiven = TRUE;
            break;
        case  BSIM4V4_MOD_VOFFCV :
            mod->BSIM4V4voffcv = value->rValue;
            mod->BSIM4V4voffcvGiven = TRUE;
            break;
        case  BSIM4V4_MOD_DMCG :
            mod->BSIM4V4dmcg = value->rValue;
            mod->BSIM4V4dmcgGiven = TRUE;
            break;
        case  BSIM4V4_MOD_DMCI :
            mod->BSIM4V4dmci = value->rValue;
            mod->BSIM4V4dmciGiven = TRUE;
            break;
        case  BSIM4V4_MOD_DMDG :
            mod->BSIM4V4dmdg = value->rValue;
            mod->BSIM4V4dmdgGiven = TRUE;
            break;
        case  BSIM4V4_MOD_DMCGT :
            mod->BSIM4V4dmcgt = value->rValue;
            mod->BSIM4V4dmcgtGiven = TRUE;
            break;
        case  BSIM4V4_MOD_XGW :
            mod->BSIM4V4xgw = value->rValue;
            mod->BSIM4V4xgwGiven = TRUE;
            break;
        case  BSIM4V4_MOD_XGL :
            mod->BSIM4V4xgl = value->rValue;
            mod->BSIM4V4xglGiven = TRUE;
            break;
        case  BSIM4V4_MOD_RSHG :
            mod->BSIM4V4rshg = value->rValue;
            mod->BSIM4V4rshgGiven = TRUE;
            break;
        case  BSIM4V4_MOD_NGCON :
            mod->BSIM4V4ngcon = value->rValue;
            mod->BSIM4V4ngconGiven = TRUE;
            break;
        case  BSIM4V4_MOD_TCJ :
            mod->BSIM4V4tcj = value->rValue;
            mod->BSIM4V4tcjGiven = TRUE;
            break;
        case  BSIM4V4_MOD_TPB :
            mod->BSIM4V4tpb = value->rValue;
            mod->BSIM4V4tpbGiven = TRUE;
            break;
        case  BSIM4V4_MOD_TCJSW :
            mod->BSIM4V4tcjsw = value->rValue;
            mod->BSIM4V4tcjswGiven = TRUE;
            break;
        case  BSIM4V4_MOD_TPBSW :
            mod->BSIM4V4tpbsw = value->rValue;
            mod->BSIM4V4tpbswGiven = TRUE;
            break;
        case  BSIM4V4_MOD_TCJSWG :
            mod->BSIM4V4tcjswg = value->rValue;
            mod->BSIM4V4tcjswgGiven = TRUE;
            break;
        case  BSIM4V4_MOD_TPBSWG :
            mod->BSIM4V4tpbswg = value->rValue;
            mod->BSIM4V4tpbswgGiven = TRUE;
            break;

	/* Length dependence */
        case  BSIM4V4_MOD_LCDSC :
            mod->BSIM4V4lcdsc = value->rValue;
            mod->BSIM4V4lcdscGiven = TRUE;
            break;


        case  BSIM4V4_MOD_LCDSCB :
            mod->BSIM4V4lcdscb = value->rValue;
            mod->BSIM4V4lcdscbGiven = TRUE;
            break;
        case  BSIM4V4_MOD_LCDSCD :
            mod->BSIM4V4lcdscd = value->rValue;
            mod->BSIM4V4lcdscdGiven = TRUE;
            break;
        case  BSIM4V4_MOD_LCIT :
            mod->BSIM4V4lcit = value->rValue;
            mod->BSIM4V4lcitGiven = TRUE;
            break;
        case  BSIM4V4_MOD_LNFACTOR :
            mod->BSIM4V4lnfactor = value->rValue;
            mod->BSIM4V4lnfactorGiven = TRUE;
            break;
        case BSIM4V4_MOD_LXJ:
            mod->BSIM4V4lxj = value->rValue;
            mod->BSIM4V4lxjGiven = TRUE;
            break;
        case BSIM4V4_MOD_LVSAT:
            mod->BSIM4V4lvsat = value->rValue;
            mod->BSIM4V4lvsatGiven = TRUE;
            break;
        
        
        case BSIM4V4_MOD_LA0:
            mod->BSIM4V4la0 = value->rValue;
            mod->BSIM4V4la0Given = TRUE;
            break;
        case BSIM4V4_MOD_LAGS:
            mod->BSIM4V4lags = value->rValue;
            mod->BSIM4V4lagsGiven = TRUE;
            break;
        case BSIM4V4_MOD_LA1:
            mod->BSIM4V4la1 = value->rValue;
            mod->BSIM4V4la1Given = TRUE;
            break;
        case BSIM4V4_MOD_LA2:
            mod->BSIM4V4la2 = value->rValue;
            mod->BSIM4V4la2Given = TRUE;
            break;
        case BSIM4V4_MOD_LAT:
            mod->BSIM4V4lat = value->rValue;
            mod->BSIM4V4latGiven = TRUE;
            break;
        case BSIM4V4_MOD_LKETA:
            mod->BSIM4V4lketa = value->rValue;
            mod->BSIM4V4lketaGiven = TRUE;
            break;    
        case BSIM4V4_MOD_LNSUB:
            mod->BSIM4V4lnsub = value->rValue;
            mod->BSIM4V4lnsubGiven = TRUE;
            break;
        case BSIM4V4_MOD_LNDEP:
            mod->BSIM4V4lndep = value->rValue;
            mod->BSIM4V4lndepGiven = TRUE;
	    if (mod->BSIM4V4lndep > 1.0e20)
		mod->BSIM4V4lndep *= 1.0e-6;
            break;
        case BSIM4V4_MOD_LNSD:
            mod->BSIM4V4lnsd = value->rValue;
            mod->BSIM4V4lnsdGiven = TRUE;
            if (mod->BSIM4V4lnsd > 1.0e23)
                mod->BSIM4V4lnsd *= 1.0e-6;
            break;
        case BSIM4V4_MOD_LNGATE:
            mod->BSIM4V4lngate = value->rValue;
            mod->BSIM4V4lngateGiven = TRUE;
	    if (mod->BSIM4V4lngate > 1.0e23)
		mod->BSIM4V4lngate *= 1.0e-6;
            break;
        case BSIM4V4_MOD_LGAMMA1:
            mod->BSIM4V4lgamma1 = value->rValue;
            mod->BSIM4V4lgamma1Given = TRUE;
            break;
        case BSIM4V4_MOD_LGAMMA2:
            mod->BSIM4V4lgamma2 = value->rValue;
            mod->BSIM4V4lgamma2Given = TRUE;
            break;
        case BSIM4V4_MOD_LVBX:
            mod->BSIM4V4lvbx = value->rValue;
            mod->BSIM4V4lvbxGiven = TRUE;
            break;
        case BSIM4V4_MOD_LVBM:
            mod->BSIM4V4lvbm = value->rValue;
            mod->BSIM4V4lvbmGiven = TRUE;
            break;
        case BSIM4V4_MOD_LXT:
            mod->BSIM4V4lxt = value->rValue;
            mod->BSIM4V4lxtGiven = TRUE;
            break;
        case  BSIM4V4_MOD_LK1:
            mod->BSIM4V4lk1 = value->rValue;
            mod->BSIM4V4lk1Given = TRUE;
            break;
        case  BSIM4V4_MOD_LKT1:
            mod->BSIM4V4lkt1 = value->rValue;
            mod->BSIM4V4lkt1Given = TRUE;
            break;
        case  BSIM4V4_MOD_LKT1L:
            mod->BSIM4V4lkt1l = value->rValue;
            mod->BSIM4V4lkt1lGiven = TRUE;
            break;
        case  BSIM4V4_MOD_LKT2:
            mod->BSIM4V4lkt2 = value->rValue;
            mod->BSIM4V4lkt2Given = TRUE;
            break;
        case  BSIM4V4_MOD_LK2:
            mod->BSIM4V4lk2 = value->rValue;
            mod->BSIM4V4lk2Given = TRUE;
            break;
        case  BSIM4V4_MOD_LK3:
            mod->BSIM4V4lk3 = value->rValue;
            mod->BSIM4V4lk3Given = TRUE;
            break;
        case  BSIM4V4_MOD_LK3B:
            mod->BSIM4V4lk3b = value->rValue;
            mod->BSIM4V4lk3bGiven = TRUE;
            break;
        case  BSIM4V4_MOD_LLPE0:
            mod->BSIM4V4llpe0 = value->rValue;
            mod->BSIM4V4llpe0Given = TRUE;
            break;
        case  BSIM4V4_MOD_LLPEB:
            mod->BSIM4V4llpeb = value->rValue;
            mod->BSIM4V4llpebGiven = TRUE;
            break;
        case  BSIM4V4_MOD_LDVTP0:
            mod->BSIM4V4ldvtp0 = value->rValue;
            mod->BSIM4V4ldvtp0Given = TRUE;
            break;
        case  BSIM4V4_MOD_LDVTP1:
            mod->BSIM4V4ldvtp1 = value->rValue;
            mod->BSIM4V4ldvtp1Given = TRUE;
            break;
        case  BSIM4V4_MOD_LW0:
            mod->BSIM4V4lw0 = value->rValue;
            mod->BSIM4V4lw0Given = TRUE;
            break;
        case  BSIM4V4_MOD_LDVT0:               
            mod->BSIM4V4ldvt0 = value->rValue;
            mod->BSIM4V4ldvt0Given = TRUE;
            break;
        case  BSIM4V4_MOD_LDVT1:             
            mod->BSIM4V4ldvt1 = value->rValue;
            mod->BSIM4V4ldvt1Given = TRUE;
            break;
        case  BSIM4V4_MOD_LDVT2:             
            mod->BSIM4V4ldvt2 = value->rValue;
            mod->BSIM4V4ldvt2Given = TRUE;
            break;
        case  BSIM4V4_MOD_LDVT0W:               
            mod->BSIM4V4ldvt0w = value->rValue;
            mod->BSIM4V4ldvt0wGiven = TRUE;
            break;
        case  BSIM4V4_MOD_LDVT1W:             
            mod->BSIM4V4ldvt1w = value->rValue;
            mod->BSIM4V4ldvt1wGiven = TRUE;
            break;
        case  BSIM4V4_MOD_LDVT2W:             
            mod->BSIM4V4ldvt2w = value->rValue;
            mod->BSIM4V4ldvt2wGiven = TRUE;
            break;
        case  BSIM4V4_MOD_LDROUT:             
            mod->BSIM4V4ldrout = value->rValue;
            mod->BSIM4V4ldroutGiven = TRUE;
            break;
        case  BSIM4V4_MOD_LDSUB:             
            mod->BSIM4V4ldsub = value->rValue;
            mod->BSIM4V4ldsubGiven = TRUE;
            break;
        case BSIM4V4_MOD_LVTH0:
            mod->BSIM4V4lvth0 = value->rValue;
            mod->BSIM4V4lvth0Given = TRUE;
            break;
        case BSIM4V4_MOD_LUA:
            mod->BSIM4V4lua = value->rValue;
            mod->BSIM4V4luaGiven = TRUE;
            break;
        case BSIM4V4_MOD_LUA1:
            mod->BSIM4V4lua1 = value->rValue;
            mod->BSIM4V4lua1Given = TRUE;
            break;
        case BSIM4V4_MOD_LUB:
            mod->BSIM4V4lub = value->rValue;
            mod->BSIM4V4lubGiven = TRUE;
            break;
        case BSIM4V4_MOD_LUB1:
            mod->BSIM4V4lub1 = value->rValue;
            mod->BSIM4V4lub1Given = TRUE;
            break;
        case BSIM4V4_MOD_LUC:
            mod->BSIM4V4luc = value->rValue;
            mod->BSIM4V4lucGiven = TRUE;
            break;
        case BSIM4V4_MOD_LUC1:
            mod->BSIM4V4luc1 = value->rValue;
            mod->BSIM4V4luc1Given = TRUE;
            break;
        case  BSIM4V4_MOD_LU0 :
            mod->BSIM4V4lu0 = value->rValue;
            mod->BSIM4V4lu0Given = TRUE;
            break;
        case  BSIM4V4_MOD_LUTE :
            mod->BSIM4V4lute = value->rValue;
            mod->BSIM4V4luteGiven = TRUE;
            break;
        case BSIM4V4_MOD_LVOFF:
            mod->BSIM4V4lvoff = value->rValue;
            mod->BSIM4V4lvoffGiven = TRUE;
            break;
        case BSIM4V4_MOD_LMINV:
            mod->BSIM4V4lminv = value->rValue;
            mod->BSIM4V4lminvGiven = TRUE;
            break;
        case BSIM4V4_MOD_LFPROUT:
            mod->BSIM4V4lfprout = value->rValue;
            mod->BSIM4V4lfproutGiven = TRUE;
            break;
        case BSIM4V4_MOD_LPDITS:
            mod->BSIM4V4lpdits = value->rValue;
            mod->BSIM4V4lpditsGiven = TRUE;
            break;
        case BSIM4V4_MOD_LPDITSD:
            mod->BSIM4V4lpditsd = value->rValue;
            mod->BSIM4V4lpditsdGiven = TRUE;
            break;
        case  BSIM4V4_MOD_LDELTA :
            mod->BSIM4V4ldelta = value->rValue;
            mod->BSIM4V4ldeltaGiven = TRUE;
            break;
        case BSIM4V4_MOD_LRDSW:
            mod->BSIM4V4lrdsw = value->rValue;
            mod->BSIM4V4lrdswGiven = TRUE;
            break;                     
        case BSIM4V4_MOD_LRDW:
            mod->BSIM4V4lrdw = value->rValue;
            mod->BSIM4V4lrdwGiven = TRUE;
            break;
        case BSIM4V4_MOD_LRSW:
            mod->BSIM4V4lrsw = value->rValue;
            mod->BSIM4V4lrswGiven = TRUE;
            break;
        case BSIM4V4_MOD_LPRWB:
            mod->BSIM4V4lprwb = value->rValue;
            mod->BSIM4V4lprwbGiven = TRUE;
            break;                     
        case BSIM4V4_MOD_LPRWG:
            mod->BSIM4V4lprwg = value->rValue;
            mod->BSIM4V4lprwgGiven = TRUE;
            break;                     
        case BSIM4V4_MOD_LPRT:
            mod->BSIM4V4lprt = value->rValue;
            mod->BSIM4V4lprtGiven = TRUE;
            break;                     
        case BSIM4V4_MOD_LETA0:
            mod->BSIM4V4leta0 = value->rValue;
            mod->BSIM4V4leta0Given = TRUE;
            break;                 
        case BSIM4V4_MOD_LETAB:
            mod->BSIM4V4letab = value->rValue;
            mod->BSIM4V4letabGiven = TRUE;
            break;                 
        case BSIM4V4_MOD_LPCLM:
            mod->BSIM4V4lpclm = value->rValue;
            mod->BSIM4V4lpclmGiven = TRUE;
            break;                 
        case BSIM4V4_MOD_LPDIBL1:
            mod->BSIM4V4lpdibl1 = value->rValue;
            mod->BSIM4V4lpdibl1Given = TRUE;
            break;                 
        case BSIM4V4_MOD_LPDIBL2:
            mod->BSIM4V4lpdibl2 = value->rValue;
            mod->BSIM4V4lpdibl2Given = TRUE;
            break;                 
        case BSIM4V4_MOD_LPDIBLB:
            mod->BSIM4V4lpdiblb = value->rValue;
            mod->BSIM4V4lpdiblbGiven = TRUE;
            break;                 
        case BSIM4V4_MOD_LPSCBE1:
            mod->BSIM4V4lpscbe1 = value->rValue;
            mod->BSIM4V4lpscbe1Given = TRUE;
            break;                 
        case BSIM4V4_MOD_LPSCBE2:
            mod->BSIM4V4lpscbe2 = value->rValue;
            mod->BSIM4V4lpscbe2Given = TRUE;
            break;                 
        case BSIM4V4_MOD_LPVAG:
            mod->BSIM4V4lpvag = value->rValue;
            mod->BSIM4V4lpvagGiven = TRUE;
            break;                 
        case  BSIM4V4_MOD_LWR :
            mod->BSIM4V4lwr = value->rValue;
            mod->BSIM4V4lwrGiven = TRUE;
            break;
        case  BSIM4V4_MOD_LDWG :
            mod->BSIM4V4ldwg = value->rValue;
            mod->BSIM4V4ldwgGiven = TRUE;
            break;
        case  BSIM4V4_MOD_LDWB :
            mod->BSIM4V4ldwb = value->rValue;
            mod->BSIM4V4ldwbGiven = TRUE;
            break;
        case  BSIM4V4_MOD_LB0 :
            mod->BSIM4V4lb0 = value->rValue;
            mod->BSIM4V4lb0Given = TRUE;
            break;
        case  BSIM4V4_MOD_LB1 :
            mod->BSIM4V4lb1 = value->rValue;
            mod->BSIM4V4lb1Given = TRUE;
            break;
        case  BSIM4V4_MOD_LALPHA0 :
            mod->BSIM4V4lalpha0 = value->rValue;
            mod->BSIM4V4lalpha0Given = TRUE;
            break;
        case  BSIM4V4_MOD_LALPHA1 :
            mod->BSIM4V4lalpha1 = value->rValue;
            mod->BSIM4V4lalpha1Given = TRUE;
            break;
        case  BSIM4V4_MOD_LBETA0 :
            mod->BSIM4V4lbeta0 = value->rValue;
            mod->BSIM4V4lbeta0Given = TRUE;
            break;
        case  BSIM4V4_MOD_LAGIDL :
            mod->BSIM4V4lagidl = value->rValue;
            mod->BSIM4V4lagidlGiven = TRUE;
            break;
        case  BSIM4V4_MOD_LBGIDL :
            mod->BSIM4V4lbgidl = value->rValue;
            mod->BSIM4V4lbgidlGiven = TRUE;
            break;
        case  BSIM4V4_MOD_LCGIDL :
            mod->BSIM4V4lcgidl = value->rValue;
            mod->BSIM4V4lcgidlGiven = TRUE;
            break;
        case  BSIM4V4_MOD_LPHIN :
            mod->BSIM4V4lphin = value->rValue;
            mod->BSIM4V4lphinGiven = TRUE;
            break;
        case  BSIM4V4_MOD_LEGIDL :
            mod->BSIM4V4legidl = value->rValue;
            mod->BSIM4V4legidlGiven = TRUE;
            break;
        case  BSIM4V4_MOD_LAIGC :
            mod->BSIM4V4laigc = value->rValue;
            mod->BSIM4V4laigcGiven = TRUE;
            break;
        case  BSIM4V4_MOD_LBIGC :
            mod->BSIM4V4lbigc = value->rValue;
            mod->BSIM4V4lbigcGiven = TRUE;
            break;
        case  BSIM4V4_MOD_LCIGC :
            mod->BSIM4V4lcigc = value->rValue;
            mod->BSIM4V4lcigcGiven = TRUE;
            break;
        case  BSIM4V4_MOD_LAIGSD :
            mod->BSIM4V4laigsd = value->rValue;
            mod->BSIM4V4laigsdGiven = TRUE;
            break;
        case  BSIM4V4_MOD_LBIGSD :
            mod->BSIM4V4lbigsd = value->rValue;
            mod->BSIM4V4lbigsdGiven = TRUE;
            break;
        case  BSIM4V4_MOD_LCIGSD :
            mod->BSIM4V4lcigsd = value->rValue;
            mod->BSIM4V4lcigsdGiven = TRUE;
            break;
        case  BSIM4V4_MOD_LAIGBACC :
            mod->BSIM4V4laigbacc = value->rValue;
            mod->BSIM4V4laigbaccGiven = TRUE;
            break;
        case  BSIM4V4_MOD_LBIGBACC :
            mod->BSIM4V4lbigbacc = value->rValue;
            mod->BSIM4V4lbigbaccGiven = TRUE;
            break;
        case  BSIM4V4_MOD_LCIGBACC :
            mod->BSIM4V4lcigbacc = value->rValue;
            mod->BSIM4V4lcigbaccGiven = TRUE;
            break;
        case  BSIM4V4_MOD_LAIGBINV :
            mod->BSIM4V4laigbinv = value->rValue;
            mod->BSIM4V4laigbinvGiven = TRUE;
            break;
        case  BSIM4V4_MOD_LBIGBINV :
            mod->BSIM4V4lbigbinv = value->rValue;
            mod->BSIM4V4lbigbinvGiven = TRUE;
            break;
        case  BSIM4V4_MOD_LCIGBINV :
            mod->BSIM4V4lcigbinv = value->rValue;
            mod->BSIM4V4lcigbinvGiven = TRUE;
            break;
        case  BSIM4V4_MOD_LNIGC :
            mod->BSIM4V4lnigc = value->rValue;
            mod->BSIM4V4lnigcGiven = TRUE;
            break;
        case  BSIM4V4_MOD_LNIGBINV :
            mod->BSIM4V4lnigbinv = value->rValue;
            mod->BSIM4V4lnigbinvGiven = TRUE;
            break;
        case  BSIM4V4_MOD_LNIGBACC :
            mod->BSIM4V4lnigbacc = value->rValue;
            mod->BSIM4V4lnigbaccGiven = TRUE;
            break;
        case  BSIM4V4_MOD_LNTOX :
            mod->BSIM4V4lntox = value->rValue;
            mod->BSIM4V4lntoxGiven = TRUE;
            break;
        case  BSIM4V4_MOD_LEIGBINV :
            mod->BSIM4V4leigbinv = value->rValue;
            mod->BSIM4V4leigbinvGiven = TRUE;
            break;
        case  BSIM4V4_MOD_LPIGCD :
            mod->BSIM4V4lpigcd = value->rValue;
            mod->BSIM4V4lpigcdGiven = TRUE;
            break;
        case  BSIM4V4_MOD_LPOXEDGE :
            mod->BSIM4V4lpoxedge = value->rValue;
            mod->BSIM4V4lpoxedgeGiven = TRUE;
            break;
        case  BSIM4V4_MOD_LXRCRG1 :
            mod->BSIM4V4lxrcrg1 = value->rValue;
            mod->BSIM4V4lxrcrg1Given = TRUE;
            break;
        case  BSIM4V4_MOD_LXRCRG2 :
            mod->BSIM4V4lxrcrg2 = value->rValue;
            mod->BSIM4V4lxrcrg2Given = TRUE;
            break;
        case  BSIM4V4_MOD_LLAMBDA :
            mod->BSIM4V4llambda = value->rValue;
            mod->BSIM4V4llambdaGiven = TRUE;
            break;
        case  BSIM4V4_MOD_LVTL :
            mod->BSIM4V4lvtl = value->rValue;
            mod->BSIM4V4lvtlGiven = TRUE;
            break;
        case  BSIM4V4_MOD_LXN:
            mod->BSIM4V4lxn = value->rValue;
            mod->BSIM4V4lxnGiven = TRUE;
            break;
        case  BSIM4V4_MOD_LVFBSDOFF:
            mod->BSIM4V4lvfbsdoff = value->rValue;
            mod->BSIM4V4lvfbsdoffGiven = TRUE;
            break;
        case  BSIM4V4_MOD_LEU :
            mod->BSIM4V4leu = value->rValue;
            mod->BSIM4V4leuGiven = TRUE;
            break;
        case  BSIM4V4_MOD_LVFB :
            mod->BSIM4V4lvfb = value->rValue;
            mod->BSIM4V4lvfbGiven = TRUE;
            break;
        case  BSIM4V4_MOD_LCGSL :
            mod->BSIM4V4lcgsl = value->rValue;
            mod->BSIM4V4lcgslGiven = TRUE;
            break;
        case  BSIM4V4_MOD_LCGDL :
            mod->BSIM4V4lcgdl = value->rValue;
            mod->BSIM4V4lcgdlGiven = TRUE;
            break;
        case  BSIM4V4_MOD_LCKAPPAS :
            mod->BSIM4V4lckappas = value->rValue;
            mod->BSIM4V4lckappasGiven = TRUE;
            break;
        case  BSIM4V4_MOD_LCKAPPAD :
            mod->BSIM4V4lckappad = value->rValue;
            mod->BSIM4V4lckappadGiven = TRUE;
            break;
        case  BSIM4V4_MOD_LCF :
            mod->BSIM4V4lcf = value->rValue;
            mod->BSIM4V4lcfGiven = TRUE;
            break;
        case  BSIM4V4_MOD_LCLC :
            mod->BSIM4V4lclc = value->rValue;
            mod->BSIM4V4lclcGiven = TRUE;
            break;
        case  BSIM4V4_MOD_LCLE :
            mod->BSIM4V4lcle = value->rValue;
            mod->BSIM4V4lcleGiven = TRUE;
            break;
        case  BSIM4V4_MOD_LVFBCV :
            mod->BSIM4V4lvfbcv = value->rValue;
            mod->BSIM4V4lvfbcvGiven = TRUE;
            break;
        case  BSIM4V4_MOD_LACDE :
            mod->BSIM4V4lacde = value->rValue;
            mod->BSIM4V4lacdeGiven = TRUE;
            break;
        case  BSIM4V4_MOD_LMOIN :
            mod->BSIM4V4lmoin = value->rValue;
            mod->BSIM4V4lmoinGiven = TRUE;
            break;
        case  BSIM4V4_MOD_LNOFF :
            mod->BSIM4V4lnoff = value->rValue;
            mod->BSIM4V4lnoffGiven = TRUE;
            break;
        case  BSIM4V4_MOD_LVOFFCV :
            mod->BSIM4V4lvoffcv = value->rValue;
            mod->BSIM4V4lvoffcvGiven = TRUE;
            break;

	/* Width dependence */
        case  BSIM4V4_MOD_WCDSC :
            mod->BSIM4V4wcdsc = value->rValue;
            mod->BSIM4V4wcdscGiven = TRUE;
            break;
       
       
         case  BSIM4V4_MOD_WCDSCB :
            mod->BSIM4V4wcdscb = value->rValue;
            mod->BSIM4V4wcdscbGiven = TRUE;
            break;
         case  BSIM4V4_MOD_WCDSCD :
            mod->BSIM4V4wcdscd = value->rValue;
            mod->BSIM4V4wcdscdGiven = TRUE;
            break;
        case  BSIM4V4_MOD_WCIT :
            mod->BSIM4V4wcit = value->rValue;
            mod->BSIM4V4wcitGiven = TRUE;
            break;
        case  BSIM4V4_MOD_WNFACTOR :
            mod->BSIM4V4wnfactor = value->rValue;
            mod->BSIM4V4wnfactorGiven = TRUE;
            break;
        case BSIM4V4_MOD_WXJ:
            mod->BSIM4V4wxj = value->rValue;
            mod->BSIM4V4wxjGiven = TRUE;
            break;
        case BSIM4V4_MOD_WVSAT:
            mod->BSIM4V4wvsat = value->rValue;
            mod->BSIM4V4wvsatGiven = TRUE;
            break;


        case BSIM4V4_MOD_WA0:
            mod->BSIM4V4wa0 = value->rValue;
            mod->BSIM4V4wa0Given = TRUE;
            break;
        case BSIM4V4_MOD_WAGS:
            mod->BSIM4V4wags = value->rValue;
            mod->BSIM4V4wagsGiven = TRUE;
            break;
        case BSIM4V4_MOD_WA1:
            mod->BSIM4V4wa1 = value->rValue;
            mod->BSIM4V4wa1Given = TRUE;
            break;
        case BSIM4V4_MOD_WA2:
            mod->BSIM4V4wa2 = value->rValue;
            mod->BSIM4V4wa2Given = TRUE;
            break;
        case BSIM4V4_MOD_WAT:
            mod->BSIM4V4wat = value->rValue;
            mod->BSIM4V4watGiven = TRUE;
            break;
        case BSIM4V4_MOD_WKETA:
            mod->BSIM4V4wketa = value->rValue;
            mod->BSIM4V4wketaGiven = TRUE;
            break;    
        case BSIM4V4_MOD_WNSUB:
            mod->BSIM4V4wnsub = value->rValue;
            mod->BSIM4V4wnsubGiven = TRUE;
            break;
        case BSIM4V4_MOD_WNDEP:
            mod->BSIM4V4wndep = value->rValue;
            mod->BSIM4V4wndepGiven = TRUE;
	    if (mod->BSIM4V4wndep > 1.0e20)
		mod->BSIM4V4wndep *= 1.0e-6;
            break;
        case BSIM4V4_MOD_WNSD:
            mod->BSIM4V4wnsd = value->rValue;
            mod->BSIM4V4wnsdGiven = TRUE;
            if (mod->BSIM4V4wnsd > 1.0e23)
                mod->BSIM4V4wnsd *= 1.0e-6;
            break;
        case BSIM4V4_MOD_WNGATE:
            mod->BSIM4V4wngate = value->rValue;
            mod->BSIM4V4wngateGiven = TRUE;
	    if (mod->BSIM4V4wngate > 1.0e23)
		mod->BSIM4V4wngate *= 1.0e-6;
            break;
        case BSIM4V4_MOD_WGAMMA1:
            mod->BSIM4V4wgamma1 = value->rValue;
            mod->BSIM4V4wgamma1Given = TRUE;
            break;
        case BSIM4V4_MOD_WGAMMA2:
            mod->BSIM4V4wgamma2 = value->rValue;
            mod->BSIM4V4wgamma2Given = TRUE;
            break;
        case BSIM4V4_MOD_WVBX:
            mod->BSIM4V4wvbx = value->rValue;
            mod->BSIM4V4wvbxGiven = TRUE;
            break;
        case BSIM4V4_MOD_WVBM:
            mod->BSIM4V4wvbm = value->rValue;
            mod->BSIM4V4wvbmGiven = TRUE;
            break;
        case BSIM4V4_MOD_WXT:
            mod->BSIM4V4wxt = value->rValue;
            mod->BSIM4V4wxtGiven = TRUE;
            break;
        case  BSIM4V4_MOD_WK1:
            mod->BSIM4V4wk1 = value->rValue;
            mod->BSIM4V4wk1Given = TRUE;
            break;
        case  BSIM4V4_MOD_WKT1:
            mod->BSIM4V4wkt1 = value->rValue;
            mod->BSIM4V4wkt1Given = TRUE;
            break;
        case  BSIM4V4_MOD_WKT1L:
            mod->BSIM4V4wkt1l = value->rValue;
            mod->BSIM4V4wkt1lGiven = TRUE;
            break;
        case  BSIM4V4_MOD_WKT2:
            mod->BSIM4V4wkt2 = value->rValue;
            mod->BSIM4V4wkt2Given = TRUE;
            break;
        case  BSIM4V4_MOD_WK2:
            mod->BSIM4V4wk2 = value->rValue;
            mod->BSIM4V4wk2Given = TRUE;
            break;
        case  BSIM4V4_MOD_WK3:
            mod->BSIM4V4wk3 = value->rValue;
            mod->BSIM4V4wk3Given = TRUE;
            break;
        case  BSIM4V4_MOD_WK3B:
            mod->BSIM4V4wk3b = value->rValue;
            mod->BSIM4V4wk3bGiven = TRUE;
            break;
        case  BSIM4V4_MOD_WLPE0:
            mod->BSIM4V4wlpe0 = value->rValue;
            mod->BSIM4V4wlpe0Given = TRUE;
            break;
        case  BSIM4V4_MOD_WLPEB:
            mod->BSIM4V4wlpeb = value->rValue;
            mod->BSIM4V4wlpebGiven = TRUE;
            break;
        case  BSIM4V4_MOD_WDVTP0:
            mod->BSIM4V4wdvtp0 = value->rValue;
            mod->BSIM4V4wdvtp0Given = TRUE;
            break;
        case  BSIM4V4_MOD_WDVTP1:
            mod->BSIM4V4wdvtp1 = value->rValue;
            mod->BSIM4V4wdvtp1Given = TRUE;
            break;
        case  BSIM4V4_MOD_WW0:
            mod->BSIM4V4ww0 = value->rValue;
            mod->BSIM4V4ww0Given = TRUE;
            break;
        case  BSIM4V4_MOD_WDVT0:               
            mod->BSIM4V4wdvt0 = value->rValue;
            mod->BSIM4V4wdvt0Given = TRUE;
            break;
        case  BSIM4V4_MOD_WDVT1:             
            mod->BSIM4V4wdvt1 = value->rValue;
            mod->BSIM4V4wdvt1Given = TRUE;
            break;
        case  BSIM4V4_MOD_WDVT2:             
            mod->BSIM4V4wdvt2 = value->rValue;
            mod->BSIM4V4wdvt2Given = TRUE;
            break;
        case  BSIM4V4_MOD_WDVT0W:               
            mod->BSIM4V4wdvt0w = value->rValue;
            mod->BSIM4V4wdvt0wGiven = TRUE;
            break;
        case  BSIM4V4_MOD_WDVT1W:             
            mod->BSIM4V4wdvt1w = value->rValue;
            mod->BSIM4V4wdvt1wGiven = TRUE;
            break;
        case  BSIM4V4_MOD_WDVT2W:             
            mod->BSIM4V4wdvt2w = value->rValue;
            mod->BSIM4V4wdvt2wGiven = TRUE;
            break;
        case  BSIM4V4_MOD_WDROUT:             
            mod->BSIM4V4wdrout = value->rValue;
            mod->BSIM4V4wdroutGiven = TRUE;
            break;
        case  BSIM4V4_MOD_WDSUB:             
            mod->BSIM4V4wdsub = value->rValue;
            mod->BSIM4V4wdsubGiven = TRUE;
            break;
        case BSIM4V4_MOD_WVTH0:
            mod->BSIM4V4wvth0 = value->rValue;
            mod->BSIM4V4wvth0Given = TRUE;
            break;
        case BSIM4V4_MOD_WUA:
            mod->BSIM4V4wua = value->rValue;
            mod->BSIM4V4wuaGiven = TRUE;
            break;
        case BSIM4V4_MOD_WUA1:
            mod->BSIM4V4wua1 = value->rValue;
            mod->BSIM4V4wua1Given = TRUE;
            break;
        case BSIM4V4_MOD_WUB:
            mod->BSIM4V4wub = value->rValue;
            mod->BSIM4V4wubGiven = TRUE;
            break;
        case BSIM4V4_MOD_WUB1:
            mod->BSIM4V4wub1 = value->rValue;
            mod->BSIM4V4wub1Given = TRUE;
            break;
        case BSIM4V4_MOD_WUC:
            mod->BSIM4V4wuc = value->rValue;
            mod->BSIM4V4wucGiven = TRUE;
            break;
        case BSIM4V4_MOD_WUC1:
            mod->BSIM4V4wuc1 = value->rValue;
            mod->BSIM4V4wuc1Given = TRUE;
            break;
        case  BSIM4V4_MOD_WU0 :
            mod->BSIM4V4wu0 = value->rValue;
            mod->BSIM4V4wu0Given = TRUE;
            break;
        case  BSIM4V4_MOD_WUTE :
            mod->BSIM4V4wute = value->rValue;
            mod->BSIM4V4wuteGiven = TRUE;
            break;
        case BSIM4V4_MOD_WVOFF:
            mod->BSIM4V4wvoff = value->rValue;
            mod->BSIM4V4wvoffGiven = TRUE;
            break;
        case BSIM4V4_MOD_WMINV:
            mod->BSIM4V4wminv = value->rValue;
            mod->BSIM4V4wminvGiven = TRUE;
            break;
        case BSIM4V4_MOD_WFPROUT:
            mod->BSIM4V4wfprout = value->rValue;
            mod->BSIM4V4wfproutGiven = TRUE;
            break;
        case BSIM4V4_MOD_WPDITS:
            mod->BSIM4V4wpdits = value->rValue;
            mod->BSIM4V4wpditsGiven = TRUE;
            break;
        case BSIM4V4_MOD_WPDITSD:
            mod->BSIM4V4wpditsd = value->rValue;
            mod->BSIM4V4wpditsdGiven = TRUE;
            break;
        case  BSIM4V4_MOD_WDELTA :
            mod->BSIM4V4wdelta = value->rValue;
            mod->BSIM4V4wdeltaGiven = TRUE;
            break;
        case BSIM4V4_MOD_WRDSW:
            mod->BSIM4V4wrdsw = value->rValue;
            mod->BSIM4V4wrdswGiven = TRUE;
            break;                     
        case BSIM4V4_MOD_WRDW:
            mod->BSIM4V4wrdw = value->rValue;
            mod->BSIM4V4wrdwGiven = TRUE;
            break;
        case BSIM4V4_MOD_WRSW:
            mod->BSIM4V4wrsw = value->rValue;
            mod->BSIM4V4wrswGiven = TRUE;
            break;
        case BSIM4V4_MOD_WPRWB:
            mod->BSIM4V4wprwb = value->rValue;
            mod->BSIM4V4wprwbGiven = TRUE;
            break;                     
        case BSIM4V4_MOD_WPRWG:
            mod->BSIM4V4wprwg = value->rValue;
            mod->BSIM4V4wprwgGiven = TRUE;
            break;                     
        case BSIM4V4_MOD_WPRT:
            mod->BSIM4V4wprt = value->rValue;
            mod->BSIM4V4wprtGiven = TRUE;
            break;                     
        case BSIM4V4_MOD_WETA0:
            mod->BSIM4V4weta0 = value->rValue;
            mod->BSIM4V4weta0Given = TRUE;
            break;                 
        case BSIM4V4_MOD_WETAB:
            mod->BSIM4V4wetab = value->rValue;
            mod->BSIM4V4wetabGiven = TRUE;
            break;                 
        case BSIM4V4_MOD_WPCLM:
            mod->BSIM4V4wpclm = value->rValue;
            mod->BSIM4V4wpclmGiven = TRUE;
            break;                 
        case BSIM4V4_MOD_WPDIBL1:
            mod->BSIM4V4wpdibl1 = value->rValue;
            mod->BSIM4V4wpdibl1Given = TRUE;
            break;                 
        case BSIM4V4_MOD_WPDIBL2:
            mod->BSIM4V4wpdibl2 = value->rValue;
            mod->BSIM4V4wpdibl2Given = TRUE;
            break;                 
        case BSIM4V4_MOD_WPDIBLB:
            mod->BSIM4V4wpdiblb = value->rValue;
            mod->BSIM4V4wpdiblbGiven = TRUE;
            break;                 
        case BSIM4V4_MOD_WPSCBE1:
            mod->BSIM4V4wpscbe1 = value->rValue;
            mod->BSIM4V4wpscbe1Given = TRUE;
            break;                 
        case BSIM4V4_MOD_WPSCBE2:
            mod->BSIM4V4wpscbe2 = value->rValue;
            mod->BSIM4V4wpscbe2Given = TRUE;
            break;                 
        case BSIM4V4_MOD_WPVAG:
            mod->BSIM4V4wpvag = value->rValue;
            mod->BSIM4V4wpvagGiven = TRUE;
            break;                 
        case  BSIM4V4_MOD_WWR :
            mod->BSIM4V4wwr = value->rValue;
            mod->BSIM4V4wwrGiven = TRUE;
            break;
        case  BSIM4V4_MOD_WDWG :
            mod->BSIM4V4wdwg = value->rValue;
            mod->BSIM4V4wdwgGiven = TRUE;
            break;
        case  BSIM4V4_MOD_WDWB :
            mod->BSIM4V4wdwb = value->rValue;
            mod->BSIM4V4wdwbGiven = TRUE;
            break;
        case  BSIM4V4_MOD_WB0 :
            mod->BSIM4V4wb0 = value->rValue;
            mod->BSIM4V4wb0Given = TRUE;
            break;
        case  BSIM4V4_MOD_WB1 :
            mod->BSIM4V4wb1 = value->rValue;
            mod->BSIM4V4wb1Given = TRUE;
            break;
        case  BSIM4V4_MOD_WALPHA0 :
            mod->BSIM4V4walpha0 = value->rValue;
            mod->BSIM4V4walpha0Given = TRUE;
            break;
        case  BSIM4V4_MOD_WALPHA1 :
            mod->BSIM4V4walpha1 = value->rValue;
            mod->BSIM4V4walpha1Given = TRUE;
            break;
        case  BSIM4V4_MOD_WBETA0 :
            mod->BSIM4V4wbeta0 = value->rValue;
            mod->BSIM4V4wbeta0Given = TRUE;
            break;
        case  BSIM4V4_MOD_WAGIDL :
            mod->BSIM4V4wagidl = value->rValue;
            mod->BSIM4V4wagidlGiven = TRUE;
            break;
        case  BSIM4V4_MOD_WBGIDL :
            mod->BSIM4V4wbgidl = value->rValue;
            mod->BSIM4V4wbgidlGiven = TRUE;
            break;
        case  BSIM4V4_MOD_WCGIDL :
            mod->BSIM4V4wcgidl = value->rValue;
            mod->BSIM4V4wcgidlGiven = TRUE;
            break;
        case  BSIM4V4_MOD_WPHIN :
            mod->BSIM4V4wphin = value->rValue;
            mod->BSIM4V4wphinGiven = TRUE;
            break;
        case  BSIM4V4_MOD_WEGIDL :
            mod->BSIM4V4wegidl = value->rValue;
            mod->BSIM4V4wegidlGiven = TRUE;
            break;
        case  BSIM4V4_MOD_WAIGC :
            mod->BSIM4V4waigc = value->rValue;
            mod->BSIM4V4waigcGiven = TRUE;
            break;
        case  BSIM4V4_MOD_WBIGC :
            mod->BSIM4V4wbigc = value->rValue;
            mod->BSIM4V4wbigcGiven = TRUE;
            break;
        case  BSIM4V4_MOD_WCIGC :
            mod->BSIM4V4wcigc = value->rValue;
            mod->BSIM4V4wcigcGiven = TRUE;
            break;
        case  BSIM4V4_MOD_WAIGSD :
            mod->BSIM4V4waigsd = value->rValue;
            mod->BSIM4V4waigsdGiven = TRUE;
            break;
        case  BSIM4V4_MOD_WBIGSD :
            mod->BSIM4V4wbigsd = value->rValue;
            mod->BSIM4V4wbigsdGiven = TRUE;
            break;
        case  BSIM4V4_MOD_WCIGSD :
            mod->BSIM4V4wcigsd = value->rValue;
            mod->BSIM4V4wcigsdGiven = TRUE;
            break;
        case  BSIM4V4_MOD_WAIGBACC :
            mod->BSIM4V4waigbacc = value->rValue;
            mod->BSIM4V4waigbaccGiven = TRUE;
            break;
        case  BSIM4V4_MOD_WBIGBACC :
            mod->BSIM4V4wbigbacc = value->rValue;
            mod->BSIM4V4wbigbaccGiven = TRUE;
            break;
        case  BSIM4V4_MOD_WCIGBACC :
            mod->BSIM4V4wcigbacc = value->rValue;
            mod->BSIM4V4wcigbaccGiven = TRUE;
            break;
        case  BSIM4V4_MOD_WAIGBINV :
            mod->BSIM4V4waigbinv = value->rValue;
            mod->BSIM4V4waigbinvGiven = TRUE;
            break;
        case  BSIM4V4_MOD_WBIGBINV :
            mod->BSIM4V4wbigbinv = value->rValue;
            mod->BSIM4V4wbigbinvGiven = TRUE;
            break;
        case  BSIM4V4_MOD_WCIGBINV :
            mod->BSIM4V4wcigbinv = value->rValue;
            mod->BSIM4V4wcigbinvGiven = TRUE;
            break;
        case  BSIM4V4_MOD_WNIGC :
            mod->BSIM4V4wnigc = value->rValue;
            mod->BSIM4V4wnigcGiven = TRUE;
            break;
        case  BSIM4V4_MOD_WNIGBINV :
            mod->BSIM4V4wnigbinv = value->rValue;
            mod->BSIM4V4wnigbinvGiven = TRUE;
            break;
        case  BSIM4V4_MOD_WNIGBACC :
            mod->BSIM4V4wnigbacc = value->rValue;
            mod->BSIM4V4wnigbaccGiven = TRUE;
            break;
        case  BSIM4V4_MOD_WNTOX :
            mod->BSIM4V4wntox = value->rValue;
            mod->BSIM4V4wntoxGiven = TRUE;
            break;
        case  BSIM4V4_MOD_WEIGBINV :
            mod->BSIM4V4weigbinv = value->rValue;
            mod->BSIM4V4weigbinvGiven = TRUE;
            break;
        case  BSIM4V4_MOD_WPIGCD :
            mod->BSIM4V4wpigcd = value->rValue;
            mod->BSIM4V4wpigcdGiven = TRUE;
            break;
        case  BSIM4V4_MOD_WPOXEDGE :
            mod->BSIM4V4wpoxedge = value->rValue;
            mod->BSIM4V4wpoxedgeGiven = TRUE;
            break;
        case  BSIM4V4_MOD_WXRCRG1 :
            mod->BSIM4V4wxrcrg1 = value->rValue;
            mod->BSIM4V4wxrcrg1Given = TRUE;
            break;
        case  BSIM4V4_MOD_WXRCRG2 :
            mod->BSIM4V4wxrcrg2 = value->rValue;
            mod->BSIM4V4wxrcrg2Given = TRUE;
            break;
        case  BSIM4V4_MOD_WLAMBDA :
            mod->BSIM4V4wlambda = value->rValue;
            mod->BSIM4V4wlambdaGiven = TRUE;
            break;
        case  BSIM4V4_MOD_WVTL :
            mod->BSIM4V4wvtl = value->rValue;
            mod->BSIM4V4wvtlGiven = TRUE;
            break;
        case  BSIM4V4_MOD_WXN:
            mod->BSIM4V4wxn = value->rValue;
            mod->BSIM4V4wxnGiven = TRUE;
            break;
        case  BSIM4V4_MOD_WVFBSDOFF:
            mod->BSIM4V4wvfbsdoff = value->rValue;
            mod->BSIM4V4wvfbsdoffGiven = TRUE;
            break;
        case  BSIM4V4_MOD_WEU :
            mod->BSIM4V4weu = value->rValue;
            mod->BSIM4V4weuGiven = TRUE;
            break;
        case  BSIM4V4_MOD_WVFB :
            mod->BSIM4V4wvfb = value->rValue;
            mod->BSIM4V4wvfbGiven = TRUE;
            break;
        case  BSIM4V4_MOD_WCGSL :
            mod->BSIM4V4wcgsl = value->rValue;
            mod->BSIM4V4wcgslGiven = TRUE;
            break;
        case  BSIM4V4_MOD_WCGDL :
            mod->BSIM4V4wcgdl = value->rValue;
            mod->BSIM4V4wcgdlGiven = TRUE;
            break;
        case  BSIM4V4_MOD_WCKAPPAS :
            mod->BSIM4V4wckappas = value->rValue;
            mod->BSIM4V4wckappasGiven = TRUE;
            break;
        case  BSIM4V4_MOD_WCKAPPAD :
            mod->BSIM4V4wckappad = value->rValue;
            mod->BSIM4V4wckappadGiven = TRUE;
            break;
        case  BSIM4V4_MOD_WCF :
            mod->BSIM4V4wcf = value->rValue;
            mod->BSIM4V4wcfGiven = TRUE;
            break;
        case  BSIM4V4_MOD_WCLC :
            mod->BSIM4V4wclc = value->rValue;
            mod->BSIM4V4wclcGiven = TRUE;
            break;
        case  BSIM4V4_MOD_WCLE :
            mod->BSIM4V4wcle = value->rValue;
            mod->BSIM4V4wcleGiven = TRUE;
            break;
        case  BSIM4V4_MOD_WVFBCV :
            mod->BSIM4V4wvfbcv = value->rValue;
            mod->BSIM4V4wvfbcvGiven = TRUE;
            break;
        case  BSIM4V4_MOD_WACDE :
            mod->BSIM4V4wacde = value->rValue;
            mod->BSIM4V4wacdeGiven = TRUE;
            break;
        case  BSIM4V4_MOD_WMOIN :
            mod->BSIM4V4wmoin = value->rValue;
            mod->BSIM4V4wmoinGiven = TRUE;
            break;
        case  BSIM4V4_MOD_WNOFF :
            mod->BSIM4V4wnoff = value->rValue;
            mod->BSIM4V4wnoffGiven = TRUE;
            break;
        case  BSIM4V4_MOD_WVOFFCV :
            mod->BSIM4V4wvoffcv = value->rValue;
            mod->BSIM4V4wvoffcvGiven = TRUE;
            break;

	/* Cross-term dependence */
        case  BSIM4V4_MOD_PCDSC :
            mod->BSIM4V4pcdsc = value->rValue;
            mod->BSIM4V4pcdscGiven = TRUE;
            break;


        case  BSIM4V4_MOD_PCDSCB :
            mod->BSIM4V4pcdscb = value->rValue;
            mod->BSIM4V4pcdscbGiven = TRUE;
            break;
        case  BSIM4V4_MOD_PCDSCD :
            mod->BSIM4V4pcdscd = value->rValue;
            mod->BSIM4V4pcdscdGiven = TRUE;
            break;
        case  BSIM4V4_MOD_PCIT :
            mod->BSIM4V4pcit = value->rValue;
            mod->BSIM4V4pcitGiven = TRUE;
            break;
        case  BSIM4V4_MOD_PNFACTOR :
            mod->BSIM4V4pnfactor = value->rValue;
            mod->BSIM4V4pnfactorGiven = TRUE;
            break;
        case BSIM4V4_MOD_PXJ:
            mod->BSIM4V4pxj = value->rValue;
            mod->BSIM4V4pxjGiven = TRUE;
            break;
        case BSIM4V4_MOD_PVSAT:
            mod->BSIM4V4pvsat = value->rValue;
            mod->BSIM4V4pvsatGiven = TRUE;
            break;


        case BSIM4V4_MOD_PA0:
            mod->BSIM4V4pa0 = value->rValue;
            mod->BSIM4V4pa0Given = TRUE;
            break;
        case BSIM4V4_MOD_PAGS:
            mod->BSIM4V4pags = value->rValue;
            mod->BSIM4V4pagsGiven = TRUE;
            break;
        case BSIM4V4_MOD_PA1:
            mod->BSIM4V4pa1 = value->rValue;
            mod->BSIM4V4pa1Given = TRUE;
            break;
        case BSIM4V4_MOD_PA2:
            mod->BSIM4V4pa2 = value->rValue;
            mod->BSIM4V4pa2Given = TRUE;
            break;
        case BSIM4V4_MOD_PAT:
            mod->BSIM4V4pat = value->rValue;
            mod->BSIM4V4patGiven = TRUE;
            break;
        case BSIM4V4_MOD_PKETA:
            mod->BSIM4V4pketa = value->rValue;
            mod->BSIM4V4pketaGiven = TRUE;
            break;    
        case BSIM4V4_MOD_PNSUB:
            mod->BSIM4V4pnsub = value->rValue;
            mod->BSIM4V4pnsubGiven = TRUE;
            break;
        case BSIM4V4_MOD_PNDEP:
            mod->BSIM4V4pndep = value->rValue;
            mod->BSIM4V4pndepGiven = TRUE;
	    if (mod->BSIM4V4pndep > 1.0e20)
		mod->BSIM4V4pndep *= 1.0e-6;
            break;
        case BSIM4V4_MOD_PNSD:
            mod->BSIM4V4pnsd = value->rValue;
            mod->BSIM4V4pnsdGiven = TRUE;
            if (mod->BSIM4V4pnsd > 1.0e23)
                mod->BSIM4V4pnsd *= 1.0e-6;
            break;
        case BSIM4V4_MOD_PNGATE:
            mod->BSIM4V4pngate = value->rValue;
            mod->BSIM4V4pngateGiven = TRUE;
	    if (mod->BSIM4V4pngate > 1.0e23)
		mod->BSIM4V4pngate *= 1.0e-6;
            break;
        case BSIM4V4_MOD_PGAMMA1:
            mod->BSIM4V4pgamma1 = value->rValue;
            mod->BSIM4V4pgamma1Given = TRUE;
            break;
        case BSIM4V4_MOD_PGAMMA2:
            mod->BSIM4V4pgamma2 = value->rValue;
            mod->BSIM4V4pgamma2Given = TRUE;
            break;
        case BSIM4V4_MOD_PVBX:
            mod->BSIM4V4pvbx = value->rValue;
            mod->BSIM4V4pvbxGiven = TRUE;
            break;
        case BSIM4V4_MOD_PVBM:
            mod->BSIM4V4pvbm = value->rValue;
            mod->BSIM4V4pvbmGiven = TRUE;
            break;
        case BSIM4V4_MOD_PXT:
            mod->BSIM4V4pxt = value->rValue;
            mod->BSIM4V4pxtGiven = TRUE;
            break;
        case  BSIM4V4_MOD_PK1:
            mod->BSIM4V4pk1 = value->rValue;
            mod->BSIM4V4pk1Given = TRUE;
            break;
        case  BSIM4V4_MOD_PKT1:
            mod->BSIM4V4pkt1 = value->rValue;
            mod->BSIM4V4pkt1Given = TRUE;
            break;
        case  BSIM4V4_MOD_PKT1L:
            mod->BSIM4V4pkt1l = value->rValue;
            mod->BSIM4V4pkt1lGiven = TRUE;
            break;
        case  BSIM4V4_MOD_PKT2:
            mod->BSIM4V4pkt2 = value->rValue;
            mod->BSIM4V4pkt2Given = TRUE;
            break;
        case  BSIM4V4_MOD_PK2:
            mod->BSIM4V4pk2 = value->rValue;
            mod->BSIM4V4pk2Given = TRUE;
            break;
        case  BSIM4V4_MOD_PK3:
            mod->BSIM4V4pk3 = value->rValue;
            mod->BSIM4V4pk3Given = TRUE;
            break;
        case  BSIM4V4_MOD_PK3B:
            mod->BSIM4V4pk3b = value->rValue;
            mod->BSIM4V4pk3bGiven = TRUE;
            break;
        case  BSIM4V4_MOD_PLPE0:
            mod->BSIM4V4plpe0 = value->rValue;
            mod->BSIM4V4plpe0Given = TRUE;
            break;
        case  BSIM4V4_MOD_PLPEB:
            mod->BSIM4V4plpeb = value->rValue;
            mod->BSIM4V4plpebGiven = TRUE;
            break;
        case  BSIM4V4_MOD_PDVTP0:
            mod->BSIM4V4pdvtp0 = value->rValue;
            mod->BSIM4V4pdvtp0Given = TRUE;
            break;
        case  BSIM4V4_MOD_PDVTP1:
            mod->BSIM4V4pdvtp1 = value->rValue;
            mod->BSIM4V4pdvtp1Given = TRUE;
            break;
        case  BSIM4V4_MOD_PW0:
            mod->BSIM4V4pw0 = value->rValue;
            mod->BSIM4V4pw0Given = TRUE;
            break;
        case  BSIM4V4_MOD_PDVT0:               
            mod->BSIM4V4pdvt0 = value->rValue;
            mod->BSIM4V4pdvt0Given = TRUE;
            break;
        case  BSIM4V4_MOD_PDVT1:             
            mod->BSIM4V4pdvt1 = value->rValue;
            mod->BSIM4V4pdvt1Given = TRUE;
            break;
        case  BSIM4V4_MOD_PDVT2:             
            mod->BSIM4V4pdvt2 = value->rValue;
            mod->BSIM4V4pdvt2Given = TRUE;
            break;
        case  BSIM4V4_MOD_PDVT0W:               
            mod->BSIM4V4pdvt0w = value->rValue;
            mod->BSIM4V4pdvt0wGiven = TRUE;
            break;
        case  BSIM4V4_MOD_PDVT1W:             
            mod->BSIM4V4pdvt1w = value->rValue;
            mod->BSIM4V4pdvt1wGiven = TRUE;
            break;
        case  BSIM4V4_MOD_PDVT2W:             
            mod->BSIM4V4pdvt2w = value->rValue;
            mod->BSIM4V4pdvt2wGiven = TRUE;
            break;
        case  BSIM4V4_MOD_PDROUT:             
            mod->BSIM4V4pdrout = value->rValue;
            mod->BSIM4V4pdroutGiven = TRUE;
            break;
        case  BSIM4V4_MOD_PDSUB:             
            mod->BSIM4V4pdsub = value->rValue;
            mod->BSIM4V4pdsubGiven = TRUE;
            break;
        case BSIM4V4_MOD_PVTH0:
            mod->BSIM4V4pvth0 = value->rValue;
            mod->BSIM4V4pvth0Given = TRUE;
            break;
        case BSIM4V4_MOD_PUA:
            mod->BSIM4V4pua = value->rValue;
            mod->BSIM4V4puaGiven = TRUE;
            break;
        case BSIM4V4_MOD_PUA1:
            mod->BSIM4V4pua1 = value->rValue;
            mod->BSIM4V4pua1Given = TRUE;
            break;
        case BSIM4V4_MOD_PUB:
            mod->BSIM4V4pub = value->rValue;
            mod->BSIM4V4pubGiven = TRUE;
            break;
        case BSIM4V4_MOD_PUB1:
            mod->BSIM4V4pub1 = value->rValue;
            mod->BSIM4V4pub1Given = TRUE;
            break;
        case BSIM4V4_MOD_PUC:
            mod->BSIM4V4puc = value->rValue;
            mod->BSIM4V4pucGiven = TRUE;
            break;
        case BSIM4V4_MOD_PUC1:
            mod->BSIM4V4puc1 = value->rValue;
            mod->BSIM4V4puc1Given = TRUE;
            break;
        case  BSIM4V4_MOD_PU0 :
            mod->BSIM4V4pu0 = value->rValue;
            mod->BSIM4V4pu0Given = TRUE;
            break;
        case  BSIM4V4_MOD_PUTE :
            mod->BSIM4V4pute = value->rValue;
            mod->BSIM4V4puteGiven = TRUE;
            break;
        case BSIM4V4_MOD_PVOFF:
            mod->BSIM4V4pvoff = value->rValue;
            mod->BSIM4V4pvoffGiven = TRUE;
            break;
        case BSIM4V4_MOD_PMINV:
            mod->BSIM4V4pminv = value->rValue;
            mod->BSIM4V4pminvGiven = TRUE;
            break;
        case BSIM4V4_MOD_PFPROUT:
            mod->BSIM4V4pfprout = value->rValue;
            mod->BSIM4V4pfproutGiven = TRUE;
            break;
        case BSIM4V4_MOD_PPDITS:
            mod->BSIM4V4ppdits = value->rValue;
            mod->BSIM4V4ppditsGiven = TRUE;
            break;
        case BSIM4V4_MOD_PPDITSD:
            mod->BSIM4V4ppditsd = value->rValue;
            mod->BSIM4V4ppditsdGiven = TRUE;
            break;
        case  BSIM4V4_MOD_PDELTA :
            mod->BSIM4V4pdelta = value->rValue;
            mod->BSIM4V4pdeltaGiven = TRUE;
            break;
        case BSIM4V4_MOD_PRDSW:
            mod->BSIM4V4prdsw = value->rValue;
            mod->BSIM4V4prdswGiven = TRUE;
            break;                     
        case BSIM4V4_MOD_PRDW:
            mod->BSIM4V4prdw = value->rValue;
            mod->BSIM4V4prdwGiven = TRUE;
            break;
        case BSIM4V4_MOD_PRSW:
            mod->BSIM4V4prsw = value->rValue;
            mod->BSIM4V4prswGiven = TRUE;
            break;
        case BSIM4V4_MOD_PPRWB:
            mod->BSIM4V4pprwb = value->rValue;
            mod->BSIM4V4pprwbGiven = TRUE;
            break;                     
        case BSIM4V4_MOD_PPRWG:
            mod->BSIM4V4pprwg = value->rValue;
            mod->BSIM4V4pprwgGiven = TRUE;
            break;                     
        case BSIM4V4_MOD_PPRT:
            mod->BSIM4V4pprt = value->rValue;
            mod->BSIM4V4pprtGiven = TRUE;
            break;                     
        case BSIM4V4_MOD_PETA0:
            mod->BSIM4V4peta0 = value->rValue;
            mod->BSIM4V4peta0Given = TRUE;
            break;                 
        case BSIM4V4_MOD_PETAB:
            mod->BSIM4V4petab = value->rValue;
            mod->BSIM4V4petabGiven = TRUE;
            break;                 
        case BSIM4V4_MOD_PPCLM:
            mod->BSIM4V4ppclm = value->rValue;
            mod->BSIM4V4ppclmGiven = TRUE;
            break;                 
        case BSIM4V4_MOD_PPDIBL1:
            mod->BSIM4V4ppdibl1 = value->rValue;
            mod->BSIM4V4ppdibl1Given = TRUE;
            break;                 
        case BSIM4V4_MOD_PPDIBL2:
            mod->BSIM4V4ppdibl2 = value->rValue;
            mod->BSIM4V4ppdibl2Given = TRUE;
            break;                 
        case BSIM4V4_MOD_PPDIBLB:
            mod->BSIM4V4ppdiblb = value->rValue;
            mod->BSIM4V4ppdiblbGiven = TRUE;
            break;                 
        case BSIM4V4_MOD_PPSCBE1:
            mod->BSIM4V4ppscbe1 = value->rValue;
            mod->BSIM4V4ppscbe1Given = TRUE;
            break;                 
        case BSIM4V4_MOD_PPSCBE2:
            mod->BSIM4V4ppscbe2 = value->rValue;
            mod->BSIM4V4ppscbe2Given = TRUE;
            break;                 
        case BSIM4V4_MOD_PPVAG:
            mod->BSIM4V4ppvag = value->rValue;
            mod->BSIM4V4ppvagGiven = TRUE;
            break;                 
        case  BSIM4V4_MOD_PWR :
            mod->BSIM4V4pwr = value->rValue;
            mod->BSIM4V4pwrGiven = TRUE;
            break;
        case  BSIM4V4_MOD_PDWG :
            mod->BSIM4V4pdwg = value->rValue;
            mod->BSIM4V4pdwgGiven = TRUE;
            break;
        case  BSIM4V4_MOD_PDWB :
            mod->BSIM4V4pdwb = value->rValue;
            mod->BSIM4V4pdwbGiven = TRUE;
            break;
        case  BSIM4V4_MOD_PB0 :
            mod->BSIM4V4pb0 = value->rValue;
            mod->BSIM4V4pb0Given = TRUE;
            break;
        case  BSIM4V4_MOD_PB1 :
            mod->BSIM4V4pb1 = value->rValue;
            mod->BSIM4V4pb1Given = TRUE;
            break;
        case  BSIM4V4_MOD_PALPHA0 :
            mod->BSIM4V4palpha0 = value->rValue;
            mod->BSIM4V4palpha0Given = TRUE;
            break;
        case  BSIM4V4_MOD_PALPHA1 :
            mod->BSIM4V4palpha1 = value->rValue;
            mod->BSIM4V4palpha1Given = TRUE;
            break;
        case  BSIM4V4_MOD_PBETA0 :
            mod->BSIM4V4pbeta0 = value->rValue;
            mod->BSIM4V4pbeta0Given = TRUE;
            break;
        case  BSIM4V4_MOD_PAGIDL :
            mod->BSIM4V4pagidl = value->rValue;
            mod->BSIM4V4pagidlGiven = TRUE;
            break;
        case  BSIM4V4_MOD_PBGIDL :
            mod->BSIM4V4pbgidl = value->rValue;
            mod->BSIM4V4pbgidlGiven = TRUE;
            break;
        case  BSIM4V4_MOD_PCGIDL :
            mod->BSIM4V4pcgidl = value->rValue;
            mod->BSIM4V4pcgidlGiven = TRUE;
            break;
        case  BSIM4V4_MOD_PPHIN :
            mod->BSIM4V4pphin = value->rValue;
            mod->BSIM4V4pphinGiven = TRUE;
            break;
        case  BSIM4V4_MOD_PEGIDL :
            mod->BSIM4V4pegidl = value->rValue;
            mod->BSIM4V4pegidlGiven = TRUE;
            break;
        case  BSIM4V4_MOD_PAIGC :
            mod->BSIM4V4paigc = value->rValue;
            mod->BSIM4V4paigcGiven = TRUE;
            break;
        case  BSIM4V4_MOD_PBIGC :
            mod->BSIM4V4pbigc = value->rValue;
            mod->BSIM4V4pbigcGiven = TRUE;
            break;
        case  BSIM4V4_MOD_PCIGC :
            mod->BSIM4V4pcigc = value->rValue;
            mod->BSIM4V4pcigcGiven = TRUE;
            break;
        case  BSIM4V4_MOD_PAIGSD :
            mod->BSIM4V4paigsd = value->rValue;
            mod->BSIM4V4paigsdGiven = TRUE;
            break;
        case  BSIM4V4_MOD_PBIGSD :
            mod->BSIM4V4pbigsd = value->rValue;
            mod->BSIM4V4pbigsdGiven = TRUE;
            break;
        case  BSIM4V4_MOD_PCIGSD :
            mod->BSIM4V4pcigsd = value->rValue;
            mod->BSIM4V4pcigsdGiven = TRUE;
            break;
        case  BSIM4V4_MOD_PAIGBACC :
            mod->BSIM4V4paigbacc = value->rValue;
            mod->BSIM4V4paigbaccGiven = TRUE;
            break;
        case  BSIM4V4_MOD_PBIGBACC :
            mod->BSIM4V4pbigbacc = value->rValue;
            mod->BSIM4V4pbigbaccGiven = TRUE;
            break;
        case  BSIM4V4_MOD_PCIGBACC :
            mod->BSIM4V4pcigbacc = value->rValue;
            mod->BSIM4V4pcigbaccGiven = TRUE;
            break;
        case  BSIM4V4_MOD_PAIGBINV :
            mod->BSIM4V4paigbinv = value->rValue;
            mod->BSIM4V4paigbinvGiven = TRUE;
            break;
        case  BSIM4V4_MOD_PBIGBINV :
            mod->BSIM4V4pbigbinv = value->rValue;
            mod->BSIM4V4pbigbinvGiven = TRUE;
            break;
        case  BSIM4V4_MOD_PCIGBINV :
            mod->BSIM4V4pcigbinv = value->rValue;
            mod->BSIM4V4pcigbinvGiven = TRUE;
            break;
        case  BSIM4V4_MOD_PNIGC :
            mod->BSIM4V4pnigc = value->rValue;
            mod->BSIM4V4pnigcGiven = TRUE;
            break;
        case  BSIM4V4_MOD_PNIGBINV :
            mod->BSIM4V4pnigbinv = value->rValue;
            mod->BSIM4V4pnigbinvGiven = TRUE;
            break;
        case  BSIM4V4_MOD_PNIGBACC :
            mod->BSIM4V4pnigbacc = value->rValue;
            mod->BSIM4V4pnigbaccGiven = TRUE;
            break;
        case  BSIM4V4_MOD_PNTOX :
            mod->BSIM4V4pntox = value->rValue;
            mod->BSIM4V4pntoxGiven = TRUE;
            break;
        case  BSIM4V4_MOD_PEIGBINV :
            mod->BSIM4V4peigbinv = value->rValue;
            mod->BSIM4V4peigbinvGiven = TRUE;
            break;
        case  BSIM4V4_MOD_PPIGCD :
            mod->BSIM4V4ppigcd = value->rValue;
            mod->BSIM4V4ppigcdGiven = TRUE;
            break;
        case  BSIM4V4_MOD_PPOXEDGE :
            mod->BSIM4V4ppoxedge = value->rValue;
            mod->BSIM4V4ppoxedgeGiven = TRUE;
            break;
        case  BSIM4V4_MOD_PXRCRG1 :
            mod->BSIM4V4pxrcrg1 = value->rValue;
            mod->BSIM4V4pxrcrg1Given = TRUE;
            break;
        case  BSIM4V4_MOD_PXRCRG2 :
            mod->BSIM4V4pxrcrg2 = value->rValue;
            mod->BSIM4V4pxrcrg2Given = TRUE;
            break;
        case  BSIM4V4_MOD_PLAMBDA :
            mod->BSIM4V4plambda = value->rValue;
            mod->BSIM4V4plambdaGiven = TRUE;
            break;
        case  BSIM4V4_MOD_PVTL :
            mod->BSIM4V4pvtl = value->rValue;
            mod->BSIM4V4pvtlGiven = TRUE;
            break;
        case  BSIM4V4_MOD_PXN:
            mod->BSIM4V4pxn = value->rValue;
            mod->BSIM4V4pxnGiven = TRUE;
            break;
        case  BSIM4V4_MOD_PVFBSDOFF:
            mod->BSIM4V4pvfbsdoff = value->rValue;
            mod->BSIM4V4pvfbsdoffGiven = TRUE;
            break;
        case  BSIM4V4_MOD_PEU :
            mod->BSIM4V4peu = value->rValue;
            mod->BSIM4V4peuGiven = TRUE;
            break;
        case  BSIM4V4_MOD_PVFB :
            mod->BSIM4V4pvfb = value->rValue;
            mod->BSIM4V4pvfbGiven = TRUE;
            break;
        case  BSIM4V4_MOD_PCGSL :
            mod->BSIM4V4pcgsl = value->rValue;
            mod->BSIM4V4pcgslGiven = TRUE;
            break;
        case  BSIM4V4_MOD_PCGDL :
            mod->BSIM4V4pcgdl = value->rValue;
            mod->BSIM4V4pcgdlGiven = TRUE;
            break;
        case  BSIM4V4_MOD_PCKAPPAS :
            mod->BSIM4V4pckappas = value->rValue;
            mod->BSIM4V4pckappasGiven = TRUE;
            break;
        case  BSIM4V4_MOD_PCKAPPAD :
            mod->BSIM4V4pckappad = value->rValue;
            mod->BSIM4V4pckappadGiven = TRUE;
            break;
        case  BSIM4V4_MOD_PCF :
            mod->BSIM4V4pcf = value->rValue;
            mod->BSIM4V4pcfGiven = TRUE;
            break;
        case  BSIM4V4_MOD_PCLC :
            mod->BSIM4V4pclc = value->rValue;
            mod->BSIM4V4pclcGiven = TRUE;
            break;
        case  BSIM4V4_MOD_PCLE :
            mod->BSIM4V4pcle = value->rValue;
            mod->BSIM4V4pcleGiven = TRUE;
            break;
        case  BSIM4V4_MOD_PVFBCV :
            mod->BSIM4V4pvfbcv = value->rValue;
            mod->BSIM4V4pvfbcvGiven = TRUE;
            break;
        case  BSIM4V4_MOD_PACDE :
            mod->BSIM4V4pacde = value->rValue;
            mod->BSIM4V4pacdeGiven = TRUE;
            break;
        case  BSIM4V4_MOD_PMOIN :
            mod->BSIM4V4pmoin = value->rValue;
            mod->BSIM4V4pmoinGiven = TRUE;
            break;
        case  BSIM4V4_MOD_PNOFF :
            mod->BSIM4V4pnoff = value->rValue;
            mod->BSIM4V4pnoffGiven = TRUE;
            break;
        case  BSIM4V4_MOD_PVOFFCV :
            mod->BSIM4V4pvoffcv = value->rValue;
            mod->BSIM4V4pvoffcvGiven = TRUE;
            break;

        case  BSIM4V4_MOD_TNOM :
            mod->BSIM4V4tnom = value->rValue + CONSTCtoK;
            mod->BSIM4V4tnomGiven = TRUE;
            break;
        case  BSIM4V4_MOD_CGSO :
            mod->BSIM4V4cgso = value->rValue;
            mod->BSIM4V4cgsoGiven = TRUE;
            break;
        case  BSIM4V4_MOD_CGDO :
            mod->BSIM4V4cgdo = value->rValue;
            mod->BSIM4V4cgdoGiven = TRUE;
            break;
        case  BSIM4V4_MOD_CGBO :
            mod->BSIM4V4cgbo = value->rValue;
            mod->BSIM4V4cgboGiven = TRUE;
            break;
        case  BSIM4V4_MOD_XPART :
            mod->BSIM4V4xpart = value->rValue;
            mod->BSIM4V4xpartGiven = TRUE;
            break;
        case  BSIM4V4_MOD_RSH :
            mod->BSIM4V4sheetResistance = value->rValue;
            mod->BSIM4V4sheetResistanceGiven = TRUE;
            break;
        case  BSIM4V4_MOD_JSS :
            mod->BSIM4V4SjctSatCurDensity = value->rValue;
            mod->BSIM4V4SjctSatCurDensityGiven = TRUE;
            break;
        case  BSIM4V4_MOD_JSWS :
            mod->BSIM4V4SjctSidewallSatCurDensity = value->rValue;
            mod->BSIM4V4SjctSidewallSatCurDensityGiven = TRUE;
            break;
        case  BSIM4V4_MOD_JSWGS :
            mod->BSIM4V4SjctGateSidewallSatCurDensity = value->rValue;
            mod->BSIM4V4SjctGateSidewallSatCurDensityGiven = TRUE;
            break;
        case  BSIM4V4_MOD_PBS :
            mod->BSIM4V4SbulkJctPotential = value->rValue;
            mod->BSIM4V4SbulkJctPotentialGiven = TRUE;
            break;
        case  BSIM4V4_MOD_MJS :
            mod->BSIM4V4SbulkJctBotGradingCoeff = value->rValue;
            mod->BSIM4V4SbulkJctBotGradingCoeffGiven = TRUE;
            break;
        case  BSIM4V4_MOD_PBSWS :
            mod->BSIM4V4SsidewallJctPotential = value->rValue;
            mod->BSIM4V4SsidewallJctPotentialGiven = TRUE;
            break;
        case  BSIM4V4_MOD_MJSWS :
            mod->BSIM4V4SbulkJctSideGradingCoeff = value->rValue;
            mod->BSIM4V4SbulkJctSideGradingCoeffGiven = TRUE;
            break;
        case  BSIM4V4_MOD_CJS :
            mod->BSIM4V4SunitAreaJctCap = value->rValue;
            mod->BSIM4V4SunitAreaJctCapGiven = TRUE;
            break;
        case  BSIM4V4_MOD_CJSWS :
            mod->BSIM4V4SunitLengthSidewallJctCap = value->rValue;
            mod->BSIM4V4SunitLengthSidewallJctCapGiven = TRUE;
            break;
        case  BSIM4V4_MOD_NJS :
            mod->BSIM4V4SjctEmissionCoeff = value->rValue;
            mod->BSIM4V4SjctEmissionCoeffGiven = TRUE;
            break;
        case  BSIM4V4_MOD_PBSWGS :
            mod->BSIM4V4SGatesidewallJctPotential = value->rValue;
            mod->BSIM4V4SGatesidewallJctPotentialGiven = TRUE;
            break;
        case  BSIM4V4_MOD_MJSWGS :
            mod->BSIM4V4SbulkJctGateSideGradingCoeff = value->rValue;
            mod->BSIM4V4SbulkJctGateSideGradingCoeffGiven = TRUE;
            break;
        case  BSIM4V4_MOD_CJSWGS :
            mod->BSIM4V4SunitLengthGateSidewallJctCap = value->rValue;
            mod->BSIM4V4SunitLengthGateSidewallJctCapGiven = TRUE;
            break;
        case  BSIM4V4_MOD_XTIS :
            mod->BSIM4V4SjctTempExponent = value->rValue;
            mod->BSIM4V4SjctTempExponentGiven = TRUE;
            break;
        case  BSIM4V4_MOD_JSD :
            mod->BSIM4V4DjctSatCurDensity = value->rValue;
            mod->BSIM4V4DjctSatCurDensityGiven = TRUE;
            break;
        case  BSIM4V4_MOD_JSWD :
            mod->BSIM4V4DjctSidewallSatCurDensity = value->rValue;
            mod->BSIM4V4DjctSidewallSatCurDensityGiven = TRUE;
            break;
        case  BSIM4V4_MOD_JSWGD :
            mod->BSIM4V4DjctGateSidewallSatCurDensity = value->rValue;
            mod->BSIM4V4DjctGateSidewallSatCurDensityGiven = TRUE;
            break;
        case  BSIM4V4_MOD_PBD :
            mod->BSIM4V4DbulkJctPotential = value->rValue;
            mod->BSIM4V4DbulkJctPotentialGiven = TRUE;
            break;
        case  BSIM4V4_MOD_MJD :
            mod->BSIM4V4DbulkJctBotGradingCoeff = value->rValue;
            mod->BSIM4V4DbulkJctBotGradingCoeffGiven = TRUE;
            break;
        case  BSIM4V4_MOD_PBSWD :
            mod->BSIM4V4DsidewallJctPotential = value->rValue;
            mod->BSIM4V4DsidewallJctPotentialGiven = TRUE;
            break;
        case  BSIM4V4_MOD_MJSWD :
            mod->BSIM4V4DbulkJctSideGradingCoeff = value->rValue;
            mod->BSIM4V4DbulkJctSideGradingCoeffGiven = TRUE;
            break;
        case  BSIM4V4_MOD_CJD :
            mod->BSIM4V4DunitAreaJctCap = value->rValue;
            mod->BSIM4V4DunitAreaJctCapGiven = TRUE;
            break;
        case  BSIM4V4_MOD_CJSWD :
            mod->BSIM4V4DunitLengthSidewallJctCap = value->rValue;
            mod->BSIM4V4DunitLengthSidewallJctCapGiven = TRUE;
            break;
        case  BSIM4V4_MOD_NJD :
            mod->BSIM4V4DjctEmissionCoeff = value->rValue;
            mod->BSIM4V4DjctEmissionCoeffGiven = TRUE;
            break;
        case  BSIM4V4_MOD_PBSWGD :
            mod->BSIM4V4DGatesidewallJctPotential = value->rValue;
            mod->BSIM4V4DGatesidewallJctPotentialGiven = TRUE;
            break;
        case  BSIM4V4_MOD_MJSWGD :
            mod->BSIM4V4DbulkJctGateSideGradingCoeff = value->rValue;
            mod->BSIM4V4DbulkJctGateSideGradingCoeffGiven = TRUE;
            break;
        case  BSIM4V4_MOD_CJSWGD :
            mod->BSIM4V4DunitLengthGateSidewallJctCap = value->rValue;
            mod->BSIM4V4DunitLengthGateSidewallJctCapGiven = TRUE;
            break;
        case  BSIM4V4_MOD_XTID :
            mod->BSIM4V4DjctTempExponent = value->rValue;
            mod->BSIM4V4DjctTempExponentGiven = TRUE;
            break;
        case  BSIM4V4_MOD_LINT :
            mod->BSIM4V4Lint = value->rValue;
            mod->BSIM4V4LintGiven = TRUE;
            break;
        case  BSIM4V4_MOD_LL :
            mod->BSIM4V4Ll = value->rValue;
            mod->BSIM4V4LlGiven = TRUE;
            break;
        case  BSIM4V4_MOD_LLC :
            mod->BSIM4V4Llc = value->rValue;
            mod->BSIM4V4LlcGiven = TRUE;
            break;
        case  BSIM4V4_MOD_LLN :
            mod->BSIM4V4Lln = value->rValue;
            mod->BSIM4V4LlnGiven = TRUE;
            break;
        case  BSIM4V4_MOD_LW :
            mod->BSIM4V4Lw = value->rValue;
            mod->BSIM4V4LwGiven = TRUE;
            break;
        case  BSIM4V4_MOD_LWC :
            mod->BSIM4V4Lwc = value->rValue;
            mod->BSIM4V4LwcGiven = TRUE;
            break;
        case  BSIM4V4_MOD_LWN :
            mod->BSIM4V4Lwn = value->rValue;
            mod->BSIM4V4LwnGiven = TRUE;
            break;
        case  BSIM4V4_MOD_LWL :
            mod->BSIM4V4Lwl = value->rValue;
            mod->BSIM4V4LwlGiven = TRUE;
            break;
        case  BSIM4V4_MOD_LWLC :
            mod->BSIM4V4Lwlc = value->rValue;
            mod->BSIM4V4LwlcGiven = TRUE;
            break;
        case  BSIM4V4_MOD_LMIN :
            mod->BSIM4V4Lmin = value->rValue;
            mod->BSIM4V4LminGiven = TRUE;
            break;
        case  BSIM4V4_MOD_LMAX :
            mod->BSIM4V4Lmax = value->rValue;
            mod->BSIM4V4LmaxGiven = TRUE;
            break;
        case  BSIM4V4_MOD_WINT :
            mod->BSIM4V4Wint = value->rValue;
            mod->BSIM4V4WintGiven = TRUE;
            break;
        case  BSIM4V4_MOD_WL :
            mod->BSIM4V4Wl = value->rValue;
            mod->BSIM4V4WlGiven = TRUE;
            break;
        case  BSIM4V4_MOD_WLC :
            mod->BSIM4V4Wlc = value->rValue;
            mod->BSIM4V4WlcGiven = TRUE;
            break;
        case  BSIM4V4_MOD_WLN :
            mod->BSIM4V4Wln = value->rValue;
            mod->BSIM4V4WlnGiven = TRUE;
            break;
        case  BSIM4V4_MOD_WW :
            mod->BSIM4V4Ww = value->rValue;
            mod->BSIM4V4WwGiven = TRUE;
            break;
        case  BSIM4V4_MOD_WWC :
            mod->BSIM4V4Wwc = value->rValue;
            mod->BSIM4V4WwcGiven = TRUE;
            break;
        case  BSIM4V4_MOD_WWN :
            mod->BSIM4V4Wwn = value->rValue;
            mod->BSIM4V4WwnGiven = TRUE;
            break;
        case  BSIM4V4_MOD_WWL :
            mod->BSIM4V4Wwl = value->rValue;
            mod->BSIM4V4WwlGiven = TRUE;
            break;
        case  BSIM4V4_MOD_WWLC :
            mod->BSIM4V4Wwlc = value->rValue;
            mod->BSIM4V4WwlcGiven = TRUE;
            break;
        case  BSIM4V4_MOD_WMIN :
            mod->BSIM4V4Wmin = value->rValue;
            mod->BSIM4V4WminGiven = TRUE;
            break;
        case  BSIM4V4_MOD_WMAX :
            mod->BSIM4V4Wmax = value->rValue;
            mod->BSIM4V4WmaxGiven = TRUE;
            break;

        case  BSIM4V4_MOD_NOIA :
            mod->BSIM4V4oxideTrapDensityA = value->rValue;
            mod->BSIM4V4oxideTrapDensityAGiven = TRUE;
            break;
        case  BSIM4V4_MOD_NOIB :
            mod->BSIM4V4oxideTrapDensityB = value->rValue;
            mod->BSIM4V4oxideTrapDensityBGiven = TRUE;
            break;
        case  BSIM4V4_MOD_NOIC :
            mod->BSIM4V4oxideTrapDensityC = value->rValue;
            mod->BSIM4V4oxideTrapDensityCGiven = TRUE;
            break;
        case  BSIM4V4_MOD_EM :
            mod->BSIM4V4em = value->rValue;
            mod->BSIM4V4emGiven = TRUE;
            break;
        case  BSIM4V4_MOD_EF :
            mod->BSIM4V4ef = value->rValue;
            mod->BSIM4V4efGiven = TRUE;
            break;
        case  BSIM4V4_MOD_AF :
            mod->BSIM4V4af = value->rValue;
            mod->BSIM4V4afGiven = TRUE;
            break;
        case  BSIM4V4_MOD_KF :
            mod->BSIM4V4kf = value->rValue;
            mod->BSIM4V4kfGiven = TRUE;
            break;
        case  BSIM4V4_MOD_STIMOD :
            mod->BSIM4V4stimod = value->rValue;
            mod->BSIM4V4stimodGiven = TRUE;
            break;
        case  BSIM4V4_MOD_RGEOMOD :
            mod->BSIM4V4rgeomod = value->rValue;
            mod->BSIM4V4rgeomodGiven = TRUE;
            break;
        case  BSIM4V4_MOD_SA0 :
            mod->BSIM4V4sa0 = value->rValue;
            mod->BSIM4V4sa0Given = TRUE;
            break;
        case  BSIM4V4_MOD_SB0 :
            mod->BSIM4V4sb0 = value->rValue;
            mod->BSIM4V4sb0Given = TRUE;
            break;
        case  BSIM4V4_MOD_NMOS  :
            if(value->iValue) {
                mod->BSIM4V4type = 1;
                mod->BSIM4V4typeGiven = TRUE;
            }
            break;
        case  BSIM4V4_MOD_PMOS  :
            if(value->iValue) {
                mod->BSIM4V4type = - 1;
                mod->BSIM4V4typeGiven = TRUE;
            }
            break;
        default:
            return(E_BADPARM);
    }
    return(OK);
}


