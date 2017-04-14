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

#include "ngspice/ngspice.h"
#include "bsim4v4def.h"
#include "ngspice/ifsim.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"
#include "ngspice/const.h"

int
BSIM4v4mParam(
int param,
IFvalue *value,
GENmodel *inMod)
{
    BSIM4v4model *mod = (BSIM4v4model*)inMod;
    switch(param)
    {   case  BSIM4v4_MOD_MOBMOD :
            mod->BSIM4v4mobMod = value->iValue;
            mod->BSIM4v4mobModGiven = TRUE;
            break;
        case  BSIM4v4_MOD_BINUNIT :
            mod->BSIM4v4binUnit = value->iValue;
            mod->BSIM4v4binUnitGiven = TRUE;
            break;
        case  BSIM4v4_MOD_PARAMCHK :
            mod->BSIM4v4paramChk = value->iValue;
            mod->BSIM4v4paramChkGiven = TRUE;
            break;
        case  BSIM4v4_MOD_CAPMOD :
            mod->BSIM4v4capMod = value->iValue;
            mod->BSIM4v4capModGiven = TRUE;
            break;
        case  BSIM4v4_MOD_DIOMOD :
            mod->BSIM4v4dioMod = value->iValue;
            mod->BSIM4v4dioModGiven = TRUE;
            break;
        case  BSIM4v4_MOD_RDSMOD :
            mod->BSIM4v4rdsMod = value->iValue;
            mod->BSIM4v4rdsModGiven = TRUE;
            break;
        case  BSIM4v4_MOD_TRNQSMOD :
            mod->BSIM4v4trnqsMod = value->iValue;
            mod->BSIM4v4trnqsModGiven = TRUE;
            break;
        case  BSIM4v4_MOD_ACNQSMOD :
            mod->BSIM4v4acnqsMod = value->iValue;
            mod->BSIM4v4acnqsModGiven = TRUE;
            break;
        case  BSIM4v4_MOD_RBODYMOD :
            mod->BSIM4v4rbodyMod = value->iValue;
            mod->BSIM4v4rbodyModGiven = TRUE;
            break;
        case  BSIM4v4_MOD_RGATEMOD :
            mod->BSIM4v4rgateMod = value->iValue;
            mod->BSIM4v4rgateModGiven = TRUE;
            break;
        case  BSIM4v4_MOD_PERMOD :
            mod->BSIM4v4perMod = value->iValue;
            mod->BSIM4v4perModGiven = TRUE;
            break;
        case  BSIM4v4_MOD_GEOMOD :
            mod->BSIM4v4geoMod = value->iValue;
            mod->BSIM4v4geoModGiven = TRUE;
            break;
        case  BSIM4v4_MOD_FNOIMOD :
            mod->BSIM4v4fnoiMod = value->iValue;
            mod->BSIM4v4fnoiModGiven = TRUE;
            break;
        case  BSIM4v4_MOD_TNOIMOD :
            mod->BSIM4v4tnoiMod = value->iValue;
            mod->BSIM4v4tnoiModGiven = TRUE;
            break;
        case  BSIM4v4_MOD_IGCMOD :
            mod->BSIM4v4igcMod = value->iValue;
            mod->BSIM4v4igcModGiven = TRUE;
            break;
        case  BSIM4v4_MOD_IGBMOD :
            mod->BSIM4v4igbMod = value->iValue;
            mod->BSIM4v4igbModGiven = TRUE;
            break;
        case  BSIM4v4_MOD_TEMPMOD :
            mod->BSIM4v4tempMod = value->iValue;
            mod->BSIM4v4tempModGiven = TRUE;
            break;
        case  BSIM4v4_MOD_VERSION :
            mod->BSIM4v4version = value->sValue;
            mod->BSIM4v4versionGiven = TRUE;
            break;
        case  BSIM4v4_MOD_TOXREF :
            mod->BSIM4v4toxref = value->rValue;
            mod->BSIM4v4toxrefGiven = TRUE;
            break;
        case  BSIM4v4_MOD_TOXE :
            mod->BSIM4v4toxe = value->rValue;
            mod->BSIM4v4toxeGiven = TRUE;
            break;
        case  BSIM4v4_MOD_TOXP :
            mod->BSIM4v4toxp = value->rValue;
            mod->BSIM4v4toxpGiven = TRUE;
            break;
        case  BSIM4v4_MOD_TOXM :
            mod->BSIM4v4toxm = value->rValue;
            mod->BSIM4v4toxmGiven = TRUE;
            break;
        case  BSIM4v4_MOD_DTOX :
            mod->BSIM4v4dtox = value->rValue;
            mod->BSIM4v4dtoxGiven = TRUE;
            break;
        case  BSIM4v4_MOD_EPSROX :
            mod->BSIM4v4epsrox = value->rValue;
            mod->BSIM4v4epsroxGiven = TRUE;
            break;

        case  BSIM4v4_MOD_CDSC :
            mod->BSIM4v4cdsc = value->rValue;
            mod->BSIM4v4cdscGiven = TRUE;
            break;
        case  BSIM4v4_MOD_CDSCB :
            mod->BSIM4v4cdscb = value->rValue;
            mod->BSIM4v4cdscbGiven = TRUE;
            break;

        case  BSIM4v4_MOD_CDSCD :
            mod->BSIM4v4cdscd = value->rValue;
            mod->BSIM4v4cdscdGiven = TRUE;
            break;

        case  BSIM4v4_MOD_CIT :
            mod->BSIM4v4cit = value->rValue;
            mod->BSIM4v4citGiven = TRUE;
            break;
        case  BSIM4v4_MOD_NFACTOR :
            mod->BSIM4v4nfactor = value->rValue;
            mod->BSIM4v4nfactorGiven = TRUE;
            break;
        case BSIM4v4_MOD_XJ:
            mod->BSIM4v4xj = value->rValue;
            mod->BSIM4v4xjGiven = TRUE;
            break;
        case BSIM4v4_MOD_VSAT:
            mod->BSIM4v4vsat = value->rValue;
            mod->BSIM4v4vsatGiven = TRUE;
            break;
        case BSIM4v4_MOD_A0:
            mod->BSIM4v4a0 = value->rValue;
            mod->BSIM4v4a0Given = TRUE;
            break;

        case BSIM4v4_MOD_AGS:
            mod->BSIM4v4ags= value->rValue;
            mod->BSIM4v4agsGiven = TRUE;
            break;

        case BSIM4v4_MOD_A1:
            mod->BSIM4v4a1 = value->rValue;
            mod->BSIM4v4a1Given = TRUE;
            break;
        case BSIM4v4_MOD_A2:
            mod->BSIM4v4a2 = value->rValue;
            mod->BSIM4v4a2Given = TRUE;
            break;
        case BSIM4v4_MOD_AT:
            mod->BSIM4v4at = value->rValue;
            mod->BSIM4v4atGiven = TRUE;
            break;
        case BSIM4v4_MOD_KETA:
            mod->BSIM4v4keta = value->rValue;
            mod->BSIM4v4ketaGiven = TRUE;
            break;
        case BSIM4v4_MOD_NSUB:
            mod->BSIM4v4nsub = value->rValue;
            mod->BSIM4v4nsubGiven = TRUE;
            break;
        case BSIM4v4_MOD_NDEP:
            mod->BSIM4v4ndep = value->rValue;
            mod->BSIM4v4ndepGiven = TRUE;
	    if (mod->BSIM4v4ndep > 1.0e20)
		mod->BSIM4v4ndep *= 1.0e-6;
            break;
        case BSIM4v4_MOD_NSD:
            mod->BSIM4v4nsd = value->rValue;
            mod->BSIM4v4nsdGiven = TRUE;
            if (mod->BSIM4v4nsd > 1.0e23)
                mod->BSIM4v4nsd *= 1.0e-6;
            break;
        case BSIM4v4_MOD_NGATE:
            mod->BSIM4v4ngate = value->rValue;
            mod->BSIM4v4ngateGiven = TRUE;
	    if (mod->BSIM4v4ngate > 1.0e23)
		mod->BSIM4v4ngate *= 1.0e-6;
            break;
        case BSIM4v4_MOD_GAMMA1:
            mod->BSIM4v4gamma1 = value->rValue;
            mod->BSIM4v4gamma1Given = TRUE;
            break;
        case BSIM4v4_MOD_GAMMA2:
            mod->BSIM4v4gamma2 = value->rValue;
            mod->BSIM4v4gamma2Given = TRUE;
            break;
        case BSIM4v4_MOD_VBX:
            mod->BSIM4v4vbx = value->rValue;
            mod->BSIM4v4vbxGiven = TRUE;
            break;
        case BSIM4v4_MOD_VBM:
            mod->BSIM4v4vbm = value->rValue;
            mod->BSIM4v4vbmGiven = TRUE;
            break;
        case BSIM4v4_MOD_XT:
            mod->BSIM4v4xt = value->rValue;
            mod->BSIM4v4xtGiven = TRUE;
            break;
        case  BSIM4v4_MOD_K1:
            mod->BSIM4v4k1 = value->rValue;
            mod->BSIM4v4k1Given = TRUE;
            break;
        case  BSIM4v4_MOD_KT1:
            mod->BSIM4v4kt1 = value->rValue;
            mod->BSIM4v4kt1Given = TRUE;
            break;
        case  BSIM4v4_MOD_KT1L:
            mod->BSIM4v4kt1l = value->rValue;
            mod->BSIM4v4kt1lGiven = TRUE;
            break;
        case  BSIM4v4_MOD_KT2:
            mod->BSIM4v4kt2 = value->rValue;
            mod->BSIM4v4kt2Given = TRUE;
            break;
        case  BSIM4v4_MOD_K2:
            mod->BSIM4v4k2 = value->rValue;
            mod->BSIM4v4k2Given = TRUE;
            break;
        case  BSIM4v4_MOD_K3:
            mod->BSIM4v4k3 = value->rValue;
            mod->BSIM4v4k3Given = TRUE;
            break;
        case  BSIM4v4_MOD_K3B:
            mod->BSIM4v4k3b = value->rValue;
            mod->BSIM4v4k3bGiven = TRUE;
            break;
        case  BSIM4v4_MOD_LPE0:
            mod->BSIM4v4lpe0 = value->rValue;
            mod->BSIM4v4lpe0Given = TRUE;
            break;
        case  BSIM4v4_MOD_LPEB:
            mod->BSIM4v4lpeb = value->rValue;
            mod->BSIM4v4lpebGiven = TRUE;
            break;
        case  BSIM4v4_MOD_DVTP0:
            mod->BSIM4v4dvtp0 = value->rValue;
            mod->BSIM4v4dvtp0Given = TRUE;
            break;
        case  BSIM4v4_MOD_DVTP1:
            mod->BSIM4v4dvtp1 = value->rValue;
            mod->BSIM4v4dvtp1Given = TRUE;
            break;
        case  BSIM4v4_MOD_W0:
            mod->BSIM4v4w0 = value->rValue;
            mod->BSIM4v4w0Given = TRUE;
            break;
        case  BSIM4v4_MOD_DVT0:
            mod->BSIM4v4dvt0 = value->rValue;
            mod->BSIM4v4dvt0Given = TRUE;
            break;
        case  BSIM4v4_MOD_DVT1:
            mod->BSIM4v4dvt1 = value->rValue;
            mod->BSIM4v4dvt1Given = TRUE;
            break;
        case  BSIM4v4_MOD_DVT2:
            mod->BSIM4v4dvt2 = value->rValue;
            mod->BSIM4v4dvt2Given = TRUE;
            break;
        case  BSIM4v4_MOD_DVT0W:
            mod->BSIM4v4dvt0w = value->rValue;
            mod->BSIM4v4dvt0wGiven = TRUE;
            break;
        case  BSIM4v4_MOD_DVT1W:
            mod->BSIM4v4dvt1w = value->rValue;
            mod->BSIM4v4dvt1wGiven = TRUE;
            break;
        case  BSIM4v4_MOD_DVT2W:
            mod->BSIM4v4dvt2w = value->rValue;
            mod->BSIM4v4dvt2wGiven = TRUE;
            break;
        case  BSIM4v4_MOD_DROUT:
            mod->BSIM4v4drout = value->rValue;
            mod->BSIM4v4droutGiven = TRUE;
            break;
        case  BSIM4v4_MOD_DSUB:
            mod->BSIM4v4dsub = value->rValue;
            mod->BSIM4v4dsubGiven = TRUE;
            break;
        case BSIM4v4_MOD_VTH0:
            mod->BSIM4v4vth0 = value->rValue;
            mod->BSIM4v4vth0Given = TRUE;
            break;
        case BSIM4v4_MOD_EU:
            mod->BSIM4v4eu = value->rValue;
            mod->BSIM4v4euGiven = TRUE;
            break;
        case BSIM4v4_MOD_UA:
            mod->BSIM4v4ua = value->rValue;
            mod->BSIM4v4uaGiven = TRUE;
            break;
        case BSIM4v4_MOD_UA1:
            mod->BSIM4v4ua1 = value->rValue;
            mod->BSIM4v4ua1Given = TRUE;
            break;
        case BSIM4v4_MOD_UB:
            mod->BSIM4v4ub = value->rValue;
            mod->BSIM4v4ubGiven = TRUE;
            break;
        case BSIM4v4_MOD_UB1:
            mod->BSIM4v4ub1 = value->rValue;
            mod->BSIM4v4ub1Given = TRUE;
            break;
        case BSIM4v4_MOD_UC:
            mod->BSIM4v4uc = value->rValue;
            mod->BSIM4v4ucGiven = TRUE;
            break;
        case BSIM4v4_MOD_UC1:
            mod->BSIM4v4uc1 = value->rValue;
            mod->BSIM4v4uc1Given = TRUE;
            break;
        case  BSIM4v4_MOD_U0 :
            mod->BSIM4v4u0 = value->rValue;
            mod->BSIM4v4u0Given = TRUE;
            break;
        case  BSIM4v4_MOD_UTE :
            mod->BSIM4v4ute = value->rValue;
            mod->BSIM4v4uteGiven = TRUE;
            break;
        case BSIM4v4_MOD_VOFF:
            mod->BSIM4v4voff = value->rValue;
            mod->BSIM4v4voffGiven = TRUE;
            break;
        case BSIM4v4_MOD_VOFFL:
            mod->BSIM4v4voffl = value->rValue;
            mod->BSIM4v4vofflGiven = TRUE;
            break;
        case BSIM4v4_MOD_MINV:
            mod->BSIM4v4minv = value->rValue;
            mod->BSIM4v4minvGiven = TRUE;
            break;
        case BSIM4v4_MOD_FPROUT:
            mod->BSIM4v4fprout = value->rValue;
            mod->BSIM4v4fproutGiven = TRUE;
            break;
        case BSIM4v4_MOD_PDITS:
            mod->BSIM4v4pdits = value->rValue;
            mod->BSIM4v4pditsGiven = TRUE;
            break;
        case BSIM4v4_MOD_PDITSD:
            mod->BSIM4v4pditsd = value->rValue;
            mod->BSIM4v4pditsdGiven = TRUE;
            break;
        case BSIM4v4_MOD_PDITSL:
            mod->BSIM4v4pditsl = value->rValue;
            mod->BSIM4v4pditslGiven = TRUE;
            break;
        case  BSIM4v4_MOD_DELTA :
            mod->BSIM4v4delta = value->rValue;
            mod->BSIM4v4deltaGiven = TRUE;
            break;
        case BSIM4v4_MOD_RDSW:
            mod->BSIM4v4rdsw = value->rValue;
            mod->BSIM4v4rdswGiven = TRUE;
            break;
        case BSIM4v4_MOD_RDSWMIN:
            mod->BSIM4v4rdswmin = value->rValue;
            mod->BSIM4v4rdswminGiven = TRUE;
            break;
        case BSIM4v4_MOD_RDWMIN:
            mod->BSIM4v4rdwmin = value->rValue;
            mod->BSIM4v4rdwminGiven = TRUE;
            break;
        case BSIM4v4_MOD_RSWMIN:
            mod->BSIM4v4rswmin = value->rValue;
            mod->BSIM4v4rswminGiven = TRUE;
            break;
        case BSIM4v4_MOD_RDW:
            mod->BSIM4v4rdw = value->rValue;
            mod->BSIM4v4rdwGiven = TRUE;
            break;
        case BSIM4v4_MOD_RSW:
            mod->BSIM4v4rsw = value->rValue;
            mod->BSIM4v4rswGiven = TRUE;
            break;
        case BSIM4v4_MOD_PRWG:
            mod->BSIM4v4prwg = value->rValue;
            mod->BSIM4v4prwgGiven = TRUE;
            break;
        case BSIM4v4_MOD_PRWB:
            mod->BSIM4v4prwb = value->rValue;
            mod->BSIM4v4prwbGiven = TRUE;
            break;
        case BSIM4v4_MOD_PRT:
            mod->BSIM4v4prt = value->rValue;
            mod->BSIM4v4prtGiven = TRUE;
            break;
        case BSIM4v4_MOD_ETA0:
            mod->BSIM4v4eta0 = value->rValue;
            mod->BSIM4v4eta0Given = TRUE;
            break;
        case BSIM4v4_MOD_ETAB:
            mod->BSIM4v4etab = value->rValue;
            mod->BSIM4v4etabGiven = TRUE;
            break;
        case BSIM4v4_MOD_PCLM:
            mod->BSIM4v4pclm = value->rValue;
            mod->BSIM4v4pclmGiven = TRUE;
            break;
        case BSIM4v4_MOD_PDIBL1:
            mod->BSIM4v4pdibl1 = value->rValue;
            mod->BSIM4v4pdibl1Given = TRUE;
            break;
        case BSIM4v4_MOD_PDIBL2:
            mod->BSIM4v4pdibl2 = value->rValue;
            mod->BSIM4v4pdibl2Given = TRUE;
            break;
        case BSIM4v4_MOD_PDIBLB:
            mod->BSIM4v4pdiblb = value->rValue;
            mod->BSIM4v4pdiblbGiven = TRUE;
            break;
        case BSIM4v4_MOD_PSCBE1:
            mod->BSIM4v4pscbe1 = value->rValue;
            mod->BSIM4v4pscbe1Given = TRUE;
            break;
        case BSIM4v4_MOD_PSCBE2:
            mod->BSIM4v4pscbe2 = value->rValue;
            mod->BSIM4v4pscbe2Given = TRUE;
            break;
        case BSIM4v4_MOD_PVAG:
            mod->BSIM4v4pvag = value->rValue;
            mod->BSIM4v4pvagGiven = TRUE;
            break;
        case  BSIM4v4_MOD_WR :
            mod->BSIM4v4wr = value->rValue;
            mod->BSIM4v4wrGiven = TRUE;
            break;
        case  BSIM4v4_MOD_DWG :
            mod->BSIM4v4dwg = value->rValue;
            mod->BSIM4v4dwgGiven = TRUE;
            break;
        case  BSIM4v4_MOD_DWB :
            mod->BSIM4v4dwb = value->rValue;
            mod->BSIM4v4dwbGiven = TRUE;
            break;
        case  BSIM4v4_MOD_B0 :
            mod->BSIM4v4b0 = value->rValue;
            mod->BSIM4v4b0Given = TRUE;
            break;
        case  BSIM4v4_MOD_B1 :
            mod->BSIM4v4b1 = value->rValue;
            mod->BSIM4v4b1Given = TRUE;
            break;
        case  BSIM4v4_MOD_ALPHA0 :
            mod->BSIM4v4alpha0 = value->rValue;
            mod->BSIM4v4alpha0Given = TRUE;
            break;
        case  BSIM4v4_MOD_ALPHA1 :
            mod->BSIM4v4alpha1 = value->rValue;
            mod->BSIM4v4alpha1Given = TRUE;
            break;
        case  BSIM4v4_MOD_AGIDL :
            mod->BSIM4v4agidl = value->rValue;
            mod->BSIM4v4agidlGiven = TRUE;
            break;
        case  BSIM4v4_MOD_BGIDL :
            mod->BSIM4v4bgidl = value->rValue;
            mod->BSIM4v4bgidlGiven = TRUE;
            break;
        case  BSIM4v4_MOD_CGIDL :
            mod->BSIM4v4cgidl = value->rValue;
            mod->BSIM4v4cgidlGiven = TRUE;
            break;
        case  BSIM4v4_MOD_PHIN :
            mod->BSIM4v4phin = value->rValue;
            mod->BSIM4v4phinGiven = TRUE;
            break;
        case  BSIM4v4_MOD_EGIDL :
            mod->BSIM4v4egidl = value->rValue;
            mod->BSIM4v4egidlGiven = TRUE;
            break;
        case  BSIM4v4_MOD_AIGC :
            mod->BSIM4v4aigc = value->rValue;
            mod->BSIM4v4aigcGiven = TRUE;
            break;
        case  BSIM4v4_MOD_BIGC :
            mod->BSIM4v4bigc = value->rValue;
            mod->BSIM4v4bigcGiven = TRUE;
            break;
        case  BSIM4v4_MOD_CIGC :
            mod->BSIM4v4cigc = value->rValue;
            mod->BSIM4v4cigcGiven = TRUE;
            break;
        case  BSIM4v4_MOD_AIGSD :
            mod->BSIM4v4aigsd = value->rValue;
            mod->BSIM4v4aigsdGiven = TRUE;
            break;
        case  BSIM4v4_MOD_BIGSD :
            mod->BSIM4v4bigsd = value->rValue;
            mod->BSIM4v4bigsdGiven = TRUE;
            break;
        case  BSIM4v4_MOD_CIGSD :
            mod->BSIM4v4cigsd = value->rValue;
            mod->BSIM4v4cigsdGiven = TRUE;
            break;
        case  BSIM4v4_MOD_AIGBACC :
            mod->BSIM4v4aigbacc = value->rValue;
            mod->BSIM4v4aigbaccGiven = TRUE;
            break;
        case  BSIM4v4_MOD_BIGBACC :
            mod->BSIM4v4bigbacc = value->rValue;
            mod->BSIM4v4bigbaccGiven = TRUE;
            break;
        case  BSIM4v4_MOD_CIGBACC :
            mod->BSIM4v4cigbacc = value->rValue;
            mod->BSIM4v4cigbaccGiven = TRUE;
            break;
        case  BSIM4v4_MOD_AIGBINV :
            mod->BSIM4v4aigbinv = value->rValue;
            mod->BSIM4v4aigbinvGiven = TRUE;
            break;
        case  BSIM4v4_MOD_BIGBINV :
            mod->BSIM4v4bigbinv = value->rValue;
            mod->BSIM4v4bigbinvGiven = TRUE;
            break;
        case  BSIM4v4_MOD_CIGBINV :
            mod->BSIM4v4cigbinv = value->rValue;
            mod->BSIM4v4cigbinvGiven = TRUE;
            break;
        case  BSIM4v4_MOD_NIGC :
            mod->BSIM4v4nigc = value->rValue;
            mod->BSIM4v4nigcGiven = TRUE;
            break;
        case  BSIM4v4_MOD_NIGBINV :
            mod->BSIM4v4nigbinv = value->rValue;
            mod->BSIM4v4nigbinvGiven = TRUE;
            break;
        case  BSIM4v4_MOD_NIGBACC :
            mod->BSIM4v4nigbacc = value->rValue;
            mod->BSIM4v4nigbaccGiven = TRUE;
            break;
        case  BSIM4v4_MOD_NTOX :
            mod->BSIM4v4ntox = value->rValue;
            mod->BSIM4v4ntoxGiven = TRUE;
            break;
        case  BSIM4v4_MOD_EIGBINV :
            mod->BSIM4v4eigbinv = value->rValue;
            mod->BSIM4v4eigbinvGiven = TRUE;
            break;
        case  BSIM4v4_MOD_PIGCD :
            mod->BSIM4v4pigcd = value->rValue;
            mod->BSIM4v4pigcdGiven = TRUE;
            break;
        case  BSIM4v4_MOD_POXEDGE :
            mod->BSIM4v4poxedge = value->rValue;
            mod->BSIM4v4poxedgeGiven = TRUE;
            break;
        case  BSIM4v4_MOD_XRCRG1 :
            mod->BSIM4v4xrcrg1 = value->rValue;
            mod->BSIM4v4xrcrg1Given = TRUE;
            break;
        case  BSIM4v4_MOD_XRCRG2 :
            mod->BSIM4v4xrcrg2 = value->rValue;
            mod->BSIM4v4xrcrg2Given = TRUE;
            break;
        case  BSIM4v4_MOD_LAMBDA :
            mod->BSIM4v4lambda = value->rValue;
            mod->BSIM4v4lambdaGiven = TRUE;
            break;
        case  BSIM4v4_MOD_VTL :
            mod->BSIM4v4vtl = value->rValue;
            mod->BSIM4v4vtlGiven = TRUE;
            break;
        case  BSIM4v4_MOD_XN:
            mod->BSIM4v4xn = value->rValue;
            mod->BSIM4v4xnGiven = TRUE;
            break;
        case  BSIM4v4_MOD_LC:
            mod->BSIM4v4lc = value->rValue;
            mod->BSIM4v4lcGiven = TRUE;
            break;
        case  BSIM4v4_MOD_TNOIA :
            mod->BSIM4v4tnoia = value->rValue;
            mod->BSIM4v4tnoiaGiven = TRUE;
            break;
        case  BSIM4v4_MOD_TNOIB :
            mod->BSIM4v4tnoib = value->rValue;
            mod->BSIM4v4tnoibGiven = TRUE;
            break;
        case  BSIM4v4_MOD_RNOIA :
            mod->BSIM4v4rnoia = value->rValue;
            mod->BSIM4v4rnoiaGiven = TRUE;
            break;
        case  BSIM4v4_MOD_RNOIB :
            mod->BSIM4v4rnoib = value->rValue;
            mod->BSIM4v4rnoibGiven = TRUE;
            break;
        case  BSIM4v4_MOD_NTNOI :
            mod->BSIM4v4ntnoi = value->rValue;
            mod->BSIM4v4ntnoiGiven = TRUE;
            break;
        case  BSIM4v4_MOD_VFBSDOFF:
            mod->BSIM4v4vfbsdoff = value->rValue;
            mod->BSIM4v4vfbsdoffGiven = TRUE;
            break;
        case  BSIM4v4_MOD_LINTNOI:
            mod->BSIM4v4lintnoi = value->rValue;
            mod->BSIM4v4lintnoiGiven = TRUE;
            break;

        /* stress effect */
        case  BSIM4v4_MOD_SAREF :
            mod->BSIM4v4saref = value->rValue;
            mod->BSIM4v4sarefGiven = TRUE;
            break;
        case  BSIM4v4_MOD_SBREF :
            mod->BSIM4v4sbref = value->rValue;
            mod->BSIM4v4sbrefGiven = TRUE;
            break;
        case  BSIM4v4_MOD_WLOD :
            mod->BSIM4v4wlod = value->rValue;
            mod->BSIM4v4wlodGiven = TRUE;
            break;
        case  BSIM4v4_MOD_KU0 :
            mod->BSIM4v4ku0 = value->rValue;
            mod->BSIM4v4ku0Given = TRUE;
            break;
        case  BSIM4v4_MOD_KVSAT :
            mod->BSIM4v4kvsat = value->rValue;
            mod->BSIM4v4kvsatGiven = TRUE;
            break;
        case  BSIM4v4_MOD_KVTH0 :
            mod->BSIM4v4kvth0 = value->rValue;
            mod->BSIM4v4kvth0Given = TRUE;
            break;
        case  BSIM4v4_MOD_TKU0 :
            mod->BSIM4v4tku0 = value->rValue;
            mod->BSIM4v4tku0Given = TRUE;
            break;
        case  BSIM4v4_MOD_LLODKU0 :
            mod->BSIM4v4llodku0 = value->rValue;
            mod->BSIM4v4llodku0Given = TRUE;
            break;
        case  BSIM4v4_MOD_WLODKU0 :
            mod->BSIM4v4wlodku0 = value->rValue;
            mod->BSIM4v4wlodku0Given = TRUE;
            break;
        case  BSIM4v4_MOD_LLODVTH :
            mod->BSIM4v4llodvth = value->rValue;
            mod->BSIM4v4llodvthGiven = TRUE;
            break;
        case  BSIM4v4_MOD_WLODVTH :
            mod->BSIM4v4wlodvth = value->rValue;
            mod->BSIM4v4wlodvthGiven = TRUE;
            break;
        case  BSIM4v4_MOD_LKU0 :
            mod->BSIM4v4lku0 = value->rValue;
            mod->BSIM4v4lku0Given = TRUE;
            break;
        case  BSIM4v4_MOD_WKU0 :
            mod->BSIM4v4wku0 = value->rValue;
            mod->BSIM4v4wku0Given = TRUE;
            break;
        case  BSIM4v4_MOD_PKU0 :
            mod->BSIM4v4pku0 = value->rValue;
            mod->BSIM4v4pku0Given = TRUE;
            break;
        case  BSIM4v4_MOD_LKVTH0 :
            mod->BSIM4v4lkvth0 = value->rValue;
            mod->BSIM4v4lkvth0Given = TRUE;
            break;
        case  BSIM4v4_MOD_WKVTH0 :
            mod->BSIM4v4wkvth0 = value->rValue;
            mod->BSIM4v4wkvth0Given = TRUE;
            break;
        case  BSIM4v4_MOD_PKVTH0 :
            mod->BSIM4v4pkvth0 = value->rValue;
            mod->BSIM4v4pkvth0Given = TRUE;
            break;
        case  BSIM4v4_MOD_STK2 :
            mod->BSIM4v4stk2 = value->rValue;
            mod->BSIM4v4stk2Given = TRUE;
            break;
        case  BSIM4v4_MOD_LODK2 :
            mod->BSIM4v4lodk2 = value->rValue;
            mod->BSIM4v4lodk2Given = TRUE;
            break;
        case  BSIM4v4_MOD_STETA0 :
            mod->BSIM4v4steta0 = value->rValue;
            mod->BSIM4v4steta0Given = TRUE;
            break;
        case  BSIM4v4_MOD_LODETA0 :
            mod->BSIM4v4lodeta0 = value->rValue;
            mod->BSIM4v4lodeta0Given = TRUE;
            break;

        case  BSIM4v4_MOD_BETA0 :
            mod->BSIM4v4beta0 = value->rValue;
            mod->BSIM4v4beta0Given = TRUE;
            break;
        case  BSIM4v4_MOD_IJTHDFWD :
            mod->BSIM4v4ijthdfwd = value->rValue;
            mod->BSIM4v4ijthdfwdGiven = TRUE;
            break;
        case  BSIM4v4_MOD_IJTHSFWD :
            mod->BSIM4v4ijthsfwd = value->rValue;
            mod->BSIM4v4ijthsfwdGiven = TRUE;
            break;
        case  BSIM4v4_MOD_IJTHDREV :
            mod->BSIM4v4ijthdrev = value->rValue;
            mod->BSIM4v4ijthdrevGiven = TRUE;
            break;
        case  BSIM4v4_MOD_IJTHSREV :
            mod->BSIM4v4ijthsrev = value->rValue;
            mod->BSIM4v4ijthsrevGiven = TRUE;
            break;
        case  BSIM4v4_MOD_XJBVD :
            mod->BSIM4v4xjbvd = value->rValue;
            mod->BSIM4v4xjbvdGiven = TRUE;
            break;
        case  BSIM4v4_MOD_XJBVS :
            mod->BSIM4v4xjbvs = value->rValue;
            mod->BSIM4v4xjbvsGiven = TRUE;
            break;
        case  BSIM4v4_MOD_BVD :
            mod->BSIM4v4bvd = value->rValue;
            mod->BSIM4v4bvdGiven = TRUE;
            break;
        case  BSIM4v4_MOD_BVS :
            mod->BSIM4v4bvs = value->rValue;
            mod->BSIM4v4bvsGiven = TRUE;
            break;

        /* reverse diode */
        case  BSIM4v4_MOD_JTSS :
            mod->BSIM4v4jtss = value->rValue;
            mod->BSIM4v4jtssGiven = TRUE;
            break;
        case  BSIM4v4_MOD_JTSD :
            mod->BSIM4v4jtsd = value->rValue;
            mod->BSIM4v4jtsdGiven = TRUE;
            break;
        case  BSIM4v4_MOD_JTSSWS :
            mod->BSIM4v4jtssws = value->rValue;
            mod->BSIM4v4jtsswsGiven = TRUE;
            break;
        case  BSIM4v4_MOD_JTSSWD :
            mod->BSIM4v4jtsswd = value->rValue;
            mod->BSIM4v4jtsswdGiven = TRUE;
            break;
        case  BSIM4v4_MOD_JTSSWGS :
            mod->BSIM4v4jtsswgs = value->rValue;
            mod->BSIM4v4jtsswgsGiven = TRUE;
            break;
        case  BSIM4v4_MOD_JTSSWGD :
            mod->BSIM4v4jtsswgd = value->rValue;
            mod->BSIM4v4jtsswgdGiven = TRUE;
            break;
        case  BSIM4v4_MOD_NJTS :
            mod->BSIM4v4njts = value->rValue;
            mod->BSIM4v4njtsGiven = TRUE;
            break;
        case  BSIM4v4_MOD_NJTSSW :
            mod->BSIM4v4njtssw = value->rValue;
            mod->BSIM4v4njtsswGiven = TRUE;
            break;
        case  BSIM4v4_MOD_NJTSSWG :
            mod->BSIM4v4njtsswg = value->rValue;
            mod->BSIM4v4njtsswgGiven = TRUE;
            break;
        case  BSIM4v4_MOD_XTSS :
            mod->BSIM4v4xtss = value->rValue;
            mod->BSIM4v4xtssGiven = TRUE;
            break;
        case  BSIM4v4_MOD_XTSD :
            mod->BSIM4v4xtsd = value->rValue;
            mod->BSIM4v4xtsdGiven = TRUE;
            break;
        case  BSIM4v4_MOD_XTSSWS :
            mod->BSIM4v4xtssws = value->rValue;
            mod->BSIM4v4xtsswsGiven = TRUE;
            break;
        case  BSIM4v4_MOD_XTSSWD :
            mod->BSIM4v4xtsswd = value->rValue;
            mod->BSIM4v4xtsswdGiven = TRUE;
            break;
        case  BSIM4v4_MOD_XTSSWGS :
            mod->BSIM4v4xtsswgs = value->rValue;
            mod->BSIM4v4xtsswgsGiven = TRUE;
            break;
        case  BSIM4v4_MOD_XTSSWGD :
            mod->BSIM4v4xtsswgd = value->rValue;
            mod->BSIM4v4xtsswgdGiven = TRUE;
            break;
        case  BSIM4v4_MOD_TNJTS :
            mod->BSIM4v4tnjts = value->rValue;
            mod->BSIM4v4tnjtsGiven = TRUE;
            break;
        case  BSIM4v4_MOD_TNJTSSW :
            mod->BSIM4v4tnjtssw = value->rValue;
            mod->BSIM4v4tnjtsswGiven = TRUE;
            break;
        case  BSIM4v4_MOD_TNJTSSWG :
            mod->BSIM4v4tnjtsswg = value->rValue;
            mod->BSIM4v4tnjtsswgGiven = TRUE;
            break;
        case  BSIM4v4_MOD_VTSS :
            mod->BSIM4v4vtss = value->rValue;
            mod->BSIM4v4vtssGiven = TRUE;
            break;
        case  BSIM4v4_MOD_VTSD :
            mod->BSIM4v4vtsd = value->rValue;
            mod->BSIM4v4vtsdGiven = TRUE;
            break;
        case  BSIM4v4_MOD_VTSSWS :
            mod->BSIM4v4vtssws = value->rValue;
            mod->BSIM4v4vtsswsGiven = TRUE;
            break;
        case  BSIM4v4_MOD_VTSSWD :
            mod->BSIM4v4vtsswd = value->rValue;
            mod->BSIM4v4vtsswdGiven = TRUE;
            break;
        case  BSIM4v4_MOD_VTSSWGS :
            mod->BSIM4v4vtsswgs = value->rValue;
            mod->BSIM4v4vtsswgsGiven = TRUE;
            break;
        case  BSIM4v4_MOD_VTSSWGD :
            mod->BSIM4v4vtsswgd = value->rValue;
            mod->BSIM4v4vtsswgdGiven = TRUE;
            break;

        case  BSIM4v4_MOD_VFB :
            mod->BSIM4v4vfb = value->rValue;
            mod->BSIM4v4vfbGiven = TRUE;
            break;

        case  BSIM4v4_MOD_GBMIN :
            mod->BSIM4v4gbmin = value->rValue;
            mod->BSIM4v4gbminGiven = TRUE;
            break;
        case  BSIM4v4_MOD_RBDB :
            mod->BSIM4v4rbdb = value->rValue;
            mod->BSIM4v4rbdbGiven = TRUE;
            break;
        case  BSIM4v4_MOD_RBPB :
            mod->BSIM4v4rbpb = value->rValue;
            mod->BSIM4v4rbpbGiven = TRUE;
            break;
        case  BSIM4v4_MOD_RBSB :
            mod->BSIM4v4rbsb = value->rValue;
            mod->BSIM4v4rbsbGiven = TRUE;
            break;
        case  BSIM4v4_MOD_RBPS :
            mod->BSIM4v4rbps = value->rValue;
            mod->BSIM4v4rbpsGiven = TRUE;
            break;
        case  BSIM4v4_MOD_RBPD :
            mod->BSIM4v4rbpd = value->rValue;
            mod->BSIM4v4rbpdGiven = TRUE;
            break;

        case  BSIM4v4_MOD_CGSL :
            mod->BSIM4v4cgsl = value->rValue;
            mod->BSIM4v4cgslGiven = TRUE;
            break;
        case  BSIM4v4_MOD_CGDL :
            mod->BSIM4v4cgdl = value->rValue;
            mod->BSIM4v4cgdlGiven = TRUE;
            break;
        case  BSIM4v4_MOD_CKAPPAS :
            mod->BSIM4v4ckappas = value->rValue;
            mod->BSIM4v4ckappasGiven = TRUE;
            break;
        case  BSIM4v4_MOD_CKAPPAD :
            mod->BSIM4v4ckappad = value->rValue;
            mod->BSIM4v4ckappadGiven = TRUE;
            break;
        case  BSIM4v4_MOD_CF :
            mod->BSIM4v4cf = value->rValue;
            mod->BSIM4v4cfGiven = TRUE;
            break;
        case  BSIM4v4_MOD_CLC :
            mod->BSIM4v4clc = value->rValue;
            mod->BSIM4v4clcGiven = TRUE;
            break;
        case  BSIM4v4_MOD_CLE :
            mod->BSIM4v4cle = value->rValue;
            mod->BSIM4v4cleGiven = TRUE;
            break;
        case  BSIM4v4_MOD_DWC :
            mod->BSIM4v4dwc = value->rValue;
            mod->BSIM4v4dwcGiven = TRUE;
            break;
        case  BSIM4v4_MOD_DLC :
            mod->BSIM4v4dlc = value->rValue;
            mod->BSIM4v4dlcGiven = TRUE;
            break;
        case  BSIM4v4_MOD_XW :
            mod->BSIM4v4xw = value->rValue;
            mod->BSIM4v4xwGiven = TRUE;
            break;
        case  BSIM4v4_MOD_XL :
            mod->BSIM4v4xl = value->rValue;
            mod->BSIM4v4xlGiven = TRUE;
            break;
        case  BSIM4v4_MOD_DLCIG :
            mod->BSIM4v4dlcig = value->rValue;
            mod->BSIM4v4dlcigGiven = TRUE;
            break;
        case  BSIM4v4_MOD_DWJ :
            mod->BSIM4v4dwj = value->rValue;
            mod->BSIM4v4dwjGiven = TRUE;
            break;
        case  BSIM4v4_MOD_VFBCV :
            mod->BSIM4v4vfbcv = value->rValue;
            mod->BSIM4v4vfbcvGiven = TRUE;
            break;
        case  BSIM4v4_MOD_ACDE :
            mod->BSIM4v4acde = value->rValue;
            mod->BSIM4v4acdeGiven = TRUE;
            break;
        case  BSIM4v4_MOD_MOIN :
            mod->BSIM4v4moin = value->rValue;
            mod->BSIM4v4moinGiven = TRUE;
            break;
        case  BSIM4v4_MOD_NOFF :
            mod->BSIM4v4noff = value->rValue;
            mod->BSIM4v4noffGiven = TRUE;
            break;
        case  BSIM4v4_MOD_VOFFCV :
            mod->BSIM4v4voffcv = value->rValue;
            mod->BSIM4v4voffcvGiven = TRUE;
            break;
        case  BSIM4v4_MOD_DMCG :
            mod->BSIM4v4dmcg = value->rValue;
            mod->BSIM4v4dmcgGiven = TRUE;
            break;
        case  BSIM4v4_MOD_DMCI :
            mod->BSIM4v4dmci = value->rValue;
            mod->BSIM4v4dmciGiven = TRUE;
            break;
        case  BSIM4v4_MOD_DMDG :
            mod->BSIM4v4dmdg = value->rValue;
            mod->BSIM4v4dmdgGiven = TRUE;
            break;
        case  BSIM4v4_MOD_DMCGT :
            mod->BSIM4v4dmcgt = value->rValue;
            mod->BSIM4v4dmcgtGiven = TRUE;
            break;
        case  BSIM4v4_MOD_XGW :
            mod->BSIM4v4xgw = value->rValue;
            mod->BSIM4v4xgwGiven = TRUE;
            break;
        case  BSIM4v4_MOD_XGL :
            mod->BSIM4v4xgl = value->rValue;
            mod->BSIM4v4xglGiven = TRUE;
            break;
        case  BSIM4v4_MOD_RSHG :
            mod->BSIM4v4rshg = value->rValue;
            mod->BSIM4v4rshgGiven = TRUE;
            break;
        case  BSIM4v4_MOD_NGCON :
            mod->BSIM4v4ngcon = value->rValue;
            mod->BSIM4v4ngconGiven = TRUE;
            break;
        case  BSIM4v4_MOD_TCJ :
            mod->BSIM4v4tcj = value->rValue;
            mod->BSIM4v4tcjGiven = TRUE;
            break;
        case  BSIM4v4_MOD_TPB :
            mod->BSIM4v4tpb = value->rValue;
            mod->BSIM4v4tpbGiven = TRUE;
            break;
        case  BSIM4v4_MOD_TCJSW :
            mod->BSIM4v4tcjsw = value->rValue;
            mod->BSIM4v4tcjswGiven = TRUE;
            break;
        case  BSIM4v4_MOD_TPBSW :
            mod->BSIM4v4tpbsw = value->rValue;
            mod->BSIM4v4tpbswGiven = TRUE;
            break;
        case  BSIM4v4_MOD_TCJSWG :
            mod->BSIM4v4tcjswg = value->rValue;
            mod->BSIM4v4tcjswgGiven = TRUE;
            break;
        case  BSIM4v4_MOD_TPBSWG :
            mod->BSIM4v4tpbswg = value->rValue;
            mod->BSIM4v4tpbswgGiven = TRUE;
            break;

	/* Length dependence */
        case  BSIM4v4_MOD_LCDSC :
            mod->BSIM4v4lcdsc = value->rValue;
            mod->BSIM4v4lcdscGiven = TRUE;
            break;


        case  BSIM4v4_MOD_LCDSCB :
            mod->BSIM4v4lcdscb = value->rValue;
            mod->BSIM4v4lcdscbGiven = TRUE;
            break;
        case  BSIM4v4_MOD_LCDSCD :
            mod->BSIM4v4lcdscd = value->rValue;
            mod->BSIM4v4lcdscdGiven = TRUE;
            break;
        case  BSIM4v4_MOD_LCIT :
            mod->BSIM4v4lcit = value->rValue;
            mod->BSIM4v4lcitGiven = TRUE;
            break;
        case  BSIM4v4_MOD_LNFACTOR :
            mod->BSIM4v4lnfactor = value->rValue;
            mod->BSIM4v4lnfactorGiven = TRUE;
            break;
        case BSIM4v4_MOD_LXJ:
            mod->BSIM4v4lxj = value->rValue;
            mod->BSIM4v4lxjGiven = TRUE;
            break;
        case BSIM4v4_MOD_LVSAT:
            mod->BSIM4v4lvsat = value->rValue;
            mod->BSIM4v4lvsatGiven = TRUE;
            break;


        case BSIM4v4_MOD_LA0:
            mod->BSIM4v4la0 = value->rValue;
            mod->BSIM4v4la0Given = TRUE;
            break;
        case BSIM4v4_MOD_LAGS:
            mod->BSIM4v4lags = value->rValue;
            mod->BSIM4v4lagsGiven = TRUE;
            break;
        case BSIM4v4_MOD_LA1:
            mod->BSIM4v4la1 = value->rValue;
            mod->BSIM4v4la1Given = TRUE;
            break;
        case BSIM4v4_MOD_LA2:
            mod->BSIM4v4la2 = value->rValue;
            mod->BSIM4v4la2Given = TRUE;
            break;
        case BSIM4v4_MOD_LAT:
            mod->BSIM4v4lat = value->rValue;
            mod->BSIM4v4latGiven = TRUE;
            break;
        case BSIM4v4_MOD_LKETA:
            mod->BSIM4v4lketa = value->rValue;
            mod->BSIM4v4lketaGiven = TRUE;
            break;
        case BSIM4v4_MOD_LNSUB:
            mod->BSIM4v4lnsub = value->rValue;
            mod->BSIM4v4lnsubGiven = TRUE;
            break;
        case BSIM4v4_MOD_LNDEP:
            mod->BSIM4v4lndep = value->rValue;
            mod->BSIM4v4lndepGiven = TRUE;
	    if (mod->BSIM4v4lndep > 1.0e20)
		mod->BSIM4v4lndep *= 1.0e-6;
            break;
        case BSIM4v4_MOD_LNSD:
            mod->BSIM4v4lnsd = value->rValue;
            mod->BSIM4v4lnsdGiven = TRUE;
            if (mod->BSIM4v4lnsd > 1.0e23)
                mod->BSIM4v4lnsd *= 1.0e-6;
            break;
        case BSIM4v4_MOD_LNGATE:
            mod->BSIM4v4lngate = value->rValue;
            mod->BSIM4v4lngateGiven = TRUE;
	    if (mod->BSIM4v4lngate > 1.0e23)
		mod->BSIM4v4lngate *= 1.0e-6;
            break;
        case BSIM4v4_MOD_LGAMMA1:
            mod->BSIM4v4lgamma1 = value->rValue;
            mod->BSIM4v4lgamma1Given = TRUE;
            break;
        case BSIM4v4_MOD_LGAMMA2:
            mod->BSIM4v4lgamma2 = value->rValue;
            mod->BSIM4v4lgamma2Given = TRUE;
            break;
        case BSIM4v4_MOD_LVBX:
            mod->BSIM4v4lvbx = value->rValue;
            mod->BSIM4v4lvbxGiven = TRUE;
            break;
        case BSIM4v4_MOD_LVBM:
            mod->BSIM4v4lvbm = value->rValue;
            mod->BSIM4v4lvbmGiven = TRUE;
            break;
        case BSIM4v4_MOD_LXT:
            mod->BSIM4v4lxt = value->rValue;
            mod->BSIM4v4lxtGiven = TRUE;
            break;
        case  BSIM4v4_MOD_LK1:
            mod->BSIM4v4lk1 = value->rValue;
            mod->BSIM4v4lk1Given = TRUE;
            break;
        case  BSIM4v4_MOD_LKT1:
            mod->BSIM4v4lkt1 = value->rValue;
            mod->BSIM4v4lkt1Given = TRUE;
            break;
        case  BSIM4v4_MOD_LKT1L:
            mod->BSIM4v4lkt1l = value->rValue;
            mod->BSIM4v4lkt1lGiven = TRUE;
            break;
        case  BSIM4v4_MOD_LKT2:
            mod->BSIM4v4lkt2 = value->rValue;
            mod->BSIM4v4lkt2Given = TRUE;
            break;
        case  BSIM4v4_MOD_LK2:
            mod->BSIM4v4lk2 = value->rValue;
            mod->BSIM4v4lk2Given = TRUE;
            break;
        case  BSIM4v4_MOD_LK3:
            mod->BSIM4v4lk3 = value->rValue;
            mod->BSIM4v4lk3Given = TRUE;
            break;
        case  BSIM4v4_MOD_LK3B:
            mod->BSIM4v4lk3b = value->rValue;
            mod->BSIM4v4lk3bGiven = TRUE;
            break;
        case  BSIM4v4_MOD_LLPE0:
            mod->BSIM4v4llpe0 = value->rValue;
            mod->BSIM4v4llpe0Given = TRUE;
            break;
        case  BSIM4v4_MOD_LLPEB:
            mod->BSIM4v4llpeb = value->rValue;
            mod->BSIM4v4llpebGiven = TRUE;
            break;
        case  BSIM4v4_MOD_LDVTP0:
            mod->BSIM4v4ldvtp0 = value->rValue;
            mod->BSIM4v4ldvtp0Given = TRUE;
            break;
        case  BSIM4v4_MOD_LDVTP1:
            mod->BSIM4v4ldvtp1 = value->rValue;
            mod->BSIM4v4ldvtp1Given = TRUE;
            break;
        case  BSIM4v4_MOD_LW0:
            mod->BSIM4v4lw0 = value->rValue;
            mod->BSIM4v4lw0Given = TRUE;
            break;
        case  BSIM4v4_MOD_LDVT0:
            mod->BSIM4v4ldvt0 = value->rValue;
            mod->BSIM4v4ldvt0Given = TRUE;
            break;
        case  BSIM4v4_MOD_LDVT1:
            mod->BSIM4v4ldvt1 = value->rValue;
            mod->BSIM4v4ldvt1Given = TRUE;
            break;
        case  BSIM4v4_MOD_LDVT2:
            mod->BSIM4v4ldvt2 = value->rValue;
            mod->BSIM4v4ldvt2Given = TRUE;
            break;
        case  BSIM4v4_MOD_LDVT0W:
            mod->BSIM4v4ldvt0w = value->rValue;
            mod->BSIM4v4ldvt0wGiven = TRUE;
            break;
        case  BSIM4v4_MOD_LDVT1W:
            mod->BSIM4v4ldvt1w = value->rValue;
            mod->BSIM4v4ldvt1wGiven = TRUE;
            break;
        case  BSIM4v4_MOD_LDVT2W:
            mod->BSIM4v4ldvt2w = value->rValue;
            mod->BSIM4v4ldvt2wGiven = TRUE;
            break;
        case  BSIM4v4_MOD_LDROUT:
            mod->BSIM4v4ldrout = value->rValue;
            mod->BSIM4v4ldroutGiven = TRUE;
            break;
        case  BSIM4v4_MOD_LDSUB:
            mod->BSIM4v4ldsub = value->rValue;
            mod->BSIM4v4ldsubGiven = TRUE;
            break;
        case BSIM4v4_MOD_LVTH0:
            mod->BSIM4v4lvth0 = value->rValue;
            mod->BSIM4v4lvth0Given = TRUE;
            break;
        case BSIM4v4_MOD_LUA:
            mod->BSIM4v4lua = value->rValue;
            mod->BSIM4v4luaGiven = TRUE;
            break;
        case BSIM4v4_MOD_LUA1:
            mod->BSIM4v4lua1 = value->rValue;
            mod->BSIM4v4lua1Given = TRUE;
            break;
        case BSIM4v4_MOD_LUB:
            mod->BSIM4v4lub = value->rValue;
            mod->BSIM4v4lubGiven = TRUE;
            break;
        case BSIM4v4_MOD_LUB1:
            mod->BSIM4v4lub1 = value->rValue;
            mod->BSIM4v4lub1Given = TRUE;
            break;
        case BSIM4v4_MOD_LUC:
            mod->BSIM4v4luc = value->rValue;
            mod->BSIM4v4lucGiven = TRUE;
            break;
        case BSIM4v4_MOD_LUC1:
            mod->BSIM4v4luc1 = value->rValue;
            mod->BSIM4v4luc1Given = TRUE;
            break;
        case  BSIM4v4_MOD_LU0 :
            mod->BSIM4v4lu0 = value->rValue;
            mod->BSIM4v4lu0Given = TRUE;
            break;
        case  BSIM4v4_MOD_LUTE :
            mod->BSIM4v4lute = value->rValue;
            mod->BSIM4v4luteGiven = TRUE;
            break;
        case BSIM4v4_MOD_LVOFF:
            mod->BSIM4v4lvoff = value->rValue;
            mod->BSIM4v4lvoffGiven = TRUE;
            break;
        case BSIM4v4_MOD_LMINV:
            mod->BSIM4v4lminv = value->rValue;
            mod->BSIM4v4lminvGiven = TRUE;
            break;
        case BSIM4v4_MOD_LFPROUT:
            mod->BSIM4v4lfprout = value->rValue;
            mod->BSIM4v4lfproutGiven = TRUE;
            break;
        case BSIM4v4_MOD_LPDITS:
            mod->BSIM4v4lpdits = value->rValue;
            mod->BSIM4v4lpditsGiven = TRUE;
            break;
        case BSIM4v4_MOD_LPDITSD:
            mod->BSIM4v4lpditsd = value->rValue;
            mod->BSIM4v4lpditsdGiven = TRUE;
            break;
        case  BSIM4v4_MOD_LDELTA :
            mod->BSIM4v4ldelta = value->rValue;
            mod->BSIM4v4ldeltaGiven = TRUE;
            break;
        case BSIM4v4_MOD_LRDSW:
            mod->BSIM4v4lrdsw = value->rValue;
            mod->BSIM4v4lrdswGiven = TRUE;
            break;
        case BSIM4v4_MOD_LRDW:
            mod->BSIM4v4lrdw = value->rValue;
            mod->BSIM4v4lrdwGiven = TRUE;
            break;
        case BSIM4v4_MOD_LRSW:
            mod->BSIM4v4lrsw = value->rValue;
            mod->BSIM4v4lrswGiven = TRUE;
            break;
        case BSIM4v4_MOD_LPRWB:
            mod->BSIM4v4lprwb = value->rValue;
            mod->BSIM4v4lprwbGiven = TRUE;
            break;
        case BSIM4v4_MOD_LPRWG:
            mod->BSIM4v4lprwg = value->rValue;
            mod->BSIM4v4lprwgGiven = TRUE;
            break;
        case BSIM4v4_MOD_LPRT:
            mod->BSIM4v4lprt = value->rValue;
            mod->BSIM4v4lprtGiven = TRUE;
            break;
        case BSIM4v4_MOD_LETA0:
            mod->BSIM4v4leta0 = value->rValue;
            mod->BSIM4v4leta0Given = TRUE;
            break;
        case BSIM4v4_MOD_LETAB:
            mod->BSIM4v4letab = value->rValue;
            mod->BSIM4v4letabGiven = TRUE;
            break;
        case BSIM4v4_MOD_LPCLM:
            mod->BSIM4v4lpclm = value->rValue;
            mod->BSIM4v4lpclmGiven = TRUE;
            break;
        case BSIM4v4_MOD_LPDIBL1:
            mod->BSIM4v4lpdibl1 = value->rValue;
            mod->BSIM4v4lpdibl1Given = TRUE;
            break;
        case BSIM4v4_MOD_LPDIBL2:
            mod->BSIM4v4lpdibl2 = value->rValue;
            mod->BSIM4v4lpdibl2Given = TRUE;
            break;
        case BSIM4v4_MOD_LPDIBLB:
            mod->BSIM4v4lpdiblb = value->rValue;
            mod->BSIM4v4lpdiblbGiven = TRUE;
            break;
        case BSIM4v4_MOD_LPSCBE1:
            mod->BSIM4v4lpscbe1 = value->rValue;
            mod->BSIM4v4lpscbe1Given = TRUE;
            break;
        case BSIM4v4_MOD_LPSCBE2:
            mod->BSIM4v4lpscbe2 = value->rValue;
            mod->BSIM4v4lpscbe2Given = TRUE;
            break;
        case BSIM4v4_MOD_LPVAG:
            mod->BSIM4v4lpvag = value->rValue;
            mod->BSIM4v4lpvagGiven = TRUE;
            break;
        case  BSIM4v4_MOD_LWR :
            mod->BSIM4v4lwr = value->rValue;
            mod->BSIM4v4lwrGiven = TRUE;
            break;
        case  BSIM4v4_MOD_LDWG :
            mod->BSIM4v4ldwg = value->rValue;
            mod->BSIM4v4ldwgGiven = TRUE;
            break;
        case  BSIM4v4_MOD_LDWB :
            mod->BSIM4v4ldwb = value->rValue;
            mod->BSIM4v4ldwbGiven = TRUE;
            break;
        case  BSIM4v4_MOD_LB0 :
            mod->BSIM4v4lb0 = value->rValue;
            mod->BSIM4v4lb0Given = TRUE;
            break;
        case  BSIM4v4_MOD_LB1 :
            mod->BSIM4v4lb1 = value->rValue;
            mod->BSIM4v4lb1Given = TRUE;
            break;
        case  BSIM4v4_MOD_LALPHA0 :
            mod->BSIM4v4lalpha0 = value->rValue;
            mod->BSIM4v4lalpha0Given = TRUE;
            break;
        case  BSIM4v4_MOD_LALPHA1 :
            mod->BSIM4v4lalpha1 = value->rValue;
            mod->BSIM4v4lalpha1Given = TRUE;
            break;
        case  BSIM4v4_MOD_LBETA0 :
            mod->BSIM4v4lbeta0 = value->rValue;
            mod->BSIM4v4lbeta0Given = TRUE;
            break;
        case  BSIM4v4_MOD_LAGIDL :
            mod->BSIM4v4lagidl = value->rValue;
            mod->BSIM4v4lagidlGiven = TRUE;
            break;
        case  BSIM4v4_MOD_LBGIDL :
            mod->BSIM4v4lbgidl = value->rValue;
            mod->BSIM4v4lbgidlGiven = TRUE;
            break;
        case  BSIM4v4_MOD_LCGIDL :
            mod->BSIM4v4lcgidl = value->rValue;
            mod->BSIM4v4lcgidlGiven = TRUE;
            break;
        case  BSIM4v4_MOD_LPHIN :
            mod->BSIM4v4lphin = value->rValue;
            mod->BSIM4v4lphinGiven = TRUE;
            break;
        case  BSIM4v4_MOD_LEGIDL :
            mod->BSIM4v4legidl = value->rValue;
            mod->BSIM4v4legidlGiven = TRUE;
            break;
        case  BSIM4v4_MOD_LAIGC :
            mod->BSIM4v4laigc = value->rValue;
            mod->BSIM4v4laigcGiven = TRUE;
            break;
        case  BSIM4v4_MOD_LBIGC :
            mod->BSIM4v4lbigc = value->rValue;
            mod->BSIM4v4lbigcGiven = TRUE;
            break;
        case  BSIM4v4_MOD_LCIGC :
            mod->BSIM4v4lcigc = value->rValue;
            mod->BSIM4v4lcigcGiven = TRUE;
            break;
        case  BSIM4v4_MOD_LAIGSD :
            mod->BSIM4v4laigsd = value->rValue;
            mod->BSIM4v4laigsdGiven = TRUE;
            break;
        case  BSIM4v4_MOD_LBIGSD :
            mod->BSIM4v4lbigsd = value->rValue;
            mod->BSIM4v4lbigsdGiven = TRUE;
            break;
        case  BSIM4v4_MOD_LCIGSD :
            mod->BSIM4v4lcigsd = value->rValue;
            mod->BSIM4v4lcigsdGiven = TRUE;
            break;
        case  BSIM4v4_MOD_LAIGBACC :
            mod->BSIM4v4laigbacc = value->rValue;
            mod->BSIM4v4laigbaccGiven = TRUE;
            break;
        case  BSIM4v4_MOD_LBIGBACC :
            mod->BSIM4v4lbigbacc = value->rValue;
            mod->BSIM4v4lbigbaccGiven = TRUE;
            break;
        case  BSIM4v4_MOD_LCIGBACC :
            mod->BSIM4v4lcigbacc = value->rValue;
            mod->BSIM4v4lcigbaccGiven = TRUE;
            break;
        case  BSIM4v4_MOD_LAIGBINV :
            mod->BSIM4v4laigbinv = value->rValue;
            mod->BSIM4v4laigbinvGiven = TRUE;
            break;
        case  BSIM4v4_MOD_LBIGBINV :
            mod->BSIM4v4lbigbinv = value->rValue;
            mod->BSIM4v4lbigbinvGiven = TRUE;
            break;
        case  BSIM4v4_MOD_LCIGBINV :
            mod->BSIM4v4lcigbinv = value->rValue;
            mod->BSIM4v4lcigbinvGiven = TRUE;
            break;
        case  BSIM4v4_MOD_LNIGC :
            mod->BSIM4v4lnigc = value->rValue;
            mod->BSIM4v4lnigcGiven = TRUE;
            break;
        case  BSIM4v4_MOD_LNIGBINV :
            mod->BSIM4v4lnigbinv = value->rValue;
            mod->BSIM4v4lnigbinvGiven = TRUE;
            break;
        case  BSIM4v4_MOD_LNIGBACC :
            mod->BSIM4v4lnigbacc = value->rValue;
            mod->BSIM4v4lnigbaccGiven = TRUE;
            break;
        case  BSIM4v4_MOD_LNTOX :
            mod->BSIM4v4lntox = value->rValue;
            mod->BSIM4v4lntoxGiven = TRUE;
            break;
        case  BSIM4v4_MOD_LEIGBINV :
            mod->BSIM4v4leigbinv = value->rValue;
            mod->BSIM4v4leigbinvGiven = TRUE;
            break;
        case  BSIM4v4_MOD_LPIGCD :
            mod->BSIM4v4lpigcd = value->rValue;
            mod->BSIM4v4lpigcdGiven = TRUE;
            break;
        case  BSIM4v4_MOD_LPOXEDGE :
            mod->BSIM4v4lpoxedge = value->rValue;
            mod->BSIM4v4lpoxedgeGiven = TRUE;
            break;
        case  BSIM4v4_MOD_LXRCRG1 :
            mod->BSIM4v4lxrcrg1 = value->rValue;
            mod->BSIM4v4lxrcrg1Given = TRUE;
            break;
        case  BSIM4v4_MOD_LXRCRG2 :
            mod->BSIM4v4lxrcrg2 = value->rValue;
            mod->BSIM4v4lxrcrg2Given = TRUE;
            break;
        case  BSIM4v4_MOD_LLAMBDA :
            mod->BSIM4v4llambda = value->rValue;
            mod->BSIM4v4llambdaGiven = TRUE;
            break;
        case  BSIM4v4_MOD_LVTL :
            mod->BSIM4v4lvtl = value->rValue;
            mod->BSIM4v4lvtlGiven = TRUE;
            break;
        case  BSIM4v4_MOD_LXN:
            mod->BSIM4v4lxn = value->rValue;
            mod->BSIM4v4lxnGiven = TRUE;
            break;
        case  BSIM4v4_MOD_LVFBSDOFF:
            mod->BSIM4v4lvfbsdoff = value->rValue;
            mod->BSIM4v4lvfbsdoffGiven = TRUE;
            break;
        case  BSIM4v4_MOD_LEU :
            mod->BSIM4v4leu = value->rValue;
            mod->BSIM4v4leuGiven = TRUE;
            break;
        case  BSIM4v4_MOD_LVFB :
            mod->BSIM4v4lvfb = value->rValue;
            mod->BSIM4v4lvfbGiven = TRUE;
            break;
        case  BSIM4v4_MOD_LCGSL :
            mod->BSIM4v4lcgsl = value->rValue;
            mod->BSIM4v4lcgslGiven = TRUE;
            break;
        case  BSIM4v4_MOD_LCGDL :
            mod->BSIM4v4lcgdl = value->rValue;
            mod->BSIM4v4lcgdlGiven = TRUE;
            break;
        case  BSIM4v4_MOD_LCKAPPAS :
            mod->BSIM4v4lckappas = value->rValue;
            mod->BSIM4v4lckappasGiven = TRUE;
            break;
        case  BSIM4v4_MOD_LCKAPPAD :
            mod->BSIM4v4lckappad = value->rValue;
            mod->BSIM4v4lckappadGiven = TRUE;
            break;
        case  BSIM4v4_MOD_LCF :
            mod->BSIM4v4lcf = value->rValue;
            mod->BSIM4v4lcfGiven = TRUE;
            break;
        case  BSIM4v4_MOD_LCLC :
            mod->BSIM4v4lclc = value->rValue;
            mod->BSIM4v4lclcGiven = TRUE;
            break;
        case  BSIM4v4_MOD_LCLE :
            mod->BSIM4v4lcle = value->rValue;
            mod->BSIM4v4lcleGiven = TRUE;
            break;
        case  BSIM4v4_MOD_LVFBCV :
            mod->BSIM4v4lvfbcv = value->rValue;
            mod->BSIM4v4lvfbcvGiven = TRUE;
            break;
        case  BSIM4v4_MOD_LACDE :
            mod->BSIM4v4lacde = value->rValue;
            mod->BSIM4v4lacdeGiven = TRUE;
            break;
        case  BSIM4v4_MOD_LMOIN :
            mod->BSIM4v4lmoin = value->rValue;
            mod->BSIM4v4lmoinGiven = TRUE;
            break;
        case  BSIM4v4_MOD_LNOFF :
            mod->BSIM4v4lnoff = value->rValue;
            mod->BSIM4v4lnoffGiven = TRUE;
            break;
        case  BSIM4v4_MOD_LVOFFCV :
            mod->BSIM4v4lvoffcv = value->rValue;
            mod->BSIM4v4lvoffcvGiven = TRUE;
            break;

	/* Width dependence */
        case  BSIM4v4_MOD_WCDSC :
            mod->BSIM4v4wcdsc = value->rValue;
            mod->BSIM4v4wcdscGiven = TRUE;
            break;


         case  BSIM4v4_MOD_WCDSCB :
            mod->BSIM4v4wcdscb = value->rValue;
            mod->BSIM4v4wcdscbGiven = TRUE;
            break;
         case  BSIM4v4_MOD_WCDSCD :
            mod->BSIM4v4wcdscd = value->rValue;
            mod->BSIM4v4wcdscdGiven = TRUE;
            break;
        case  BSIM4v4_MOD_WCIT :
            mod->BSIM4v4wcit = value->rValue;
            mod->BSIM4v4wcitGiven = TRUE;
            break;
        case  BSIM4v4_MOD_WNFACTOR :
            mod->BSIM4v4wnfactor = value->rValue;
            mod->BSIM4v4wnfactorGiven = TRUE;
            break;
        case BSIM4v4_MOD_WXJ:
            mod->BSIM4v4wxj = value->rValue;
            mod->BSIM4v4wxjGiven = TRUE;
            break;
        case BSIM4v4_MOD_WVSAT:
            mod->BSIM4v4wvsat = value->rValue;
            mod->BSIM4v4wvsatGiven = TRUE;
            break;


        case BSIM4v4_MOD_WA0:
            mod->BSIM4v4wa0 = value->rValue;
            mod->BSIM4v4wa0Given = TRUE;
            break;
        case BSIM4v4_MOD_WAGS:
            mod->BSIM4v4wags = value->rValue;
            mod->BSIM4v4wagsGiven = TRUE;
            break;
        case BSIM4v4_MOD_WA1:
            mod->BSIM4v4wa1 = value->rValue;
            mod->BSIM4v4wa1Given = TRUE;
            break;
        case BSIM4v4_MOD_WA2:
            mod->BSIM4v4wa2 = value->rValue;
            mod->BSIM4v4wa2Given = TRUE;
            break;
        case BSIM4v4_MOD_WAT:
            mod->BSIM4v4wat = value->rValue;
            mod->BSIM4v4watGiven = TRUE;
            break;
        case BSIM4v4_MOD_WKETA:
            mod->BSIM4v4wketa = value->rValue;
            mod->BSIM4v4wketaGiven = TRUE;
            break;
        case BSIM4v4_MOD_WNSUB:
            mod->BSIM4v4wnsub = value->rValue;
            mod->BSIM4v4wnsubGiven = TRUE;
            break;
        case BSIM4v4_MOD_WNDEP:
            mod->BSIM4v4wndep = value->rValue;
            mod->BSIM4v4wndepGiven = TRUE;
	    if (mod->BSIM4v4wndep > 1.0e20)
		mod->BSIM4v4wndep *= 1.0e-6;
            break;
        case BSIM4v4_MOD_WNSD:
            mod->BSIM4v4wnsd = value->rValue;
            mod->BSIM4v4wnsdGiven = TRUE;
            if (mod->BSIM4v4wnsd > 1.0e23)
                mod->BSIM4v4wnsd *= 1.0e-6;
            break;
        case BSIM4v4_MOD_WNGATE:
            mod->BSIM4v4wngate = value->rValue;
            mod->BSIM4v4wngateGiven = TRUE;
	    if (mod->BSIM4v4wngate > 1.0e23)
		mod->BSIM4v4wngate *= 1.0e-6;
            break;
        case BSIM4v4_MOD_WGAMMA1:
            mod->BSIM4v4wgamma1 = value->rValue;
            mod->BSIM4v4wgamma1Given = TRUE;
            break;
        case BSIM4v4_MOD_WGAMMA2:
            mod->BSIM4v4wgamma2 = value->rValue;
            mod->BSIM4v4wgamma2Given = TRUE;
            break;
        case BSIM4v4_MOD_WVBX:
            mod->BSIM4v4wvbx = value->rValue;
            mod->BSIM4v4wvbxGiven = TRUE;
            break;
        case BSIM4v4_MOD_WVBM:
            mod->BSIM4v4wvbm = value->rValue;
            mod->BSIM4v4wvbmGiven = TRUE;
            break;
        case BSIM4v4_MOD_WXT:
            mod->BSIM4v4wxt = value->rValue;
            mod->BSIM4v4wxtGiven = TRUE;
            break;
        case  BSIM4v4_MOD_WK1:
            mod->BSIM4v4wk1 = value->rValue;
            mod->BSIM4v4wk1Given = TRUE;
            break;
        case  BSIM4v4_MOD_WKT1:
            mod->BSIM4v4wkt1 = value->rValue;
            mod->BSIM4v4wkt1Given = TRUE;
            break;
        case  BSIM4v4_MOD_WKT1L:
            mod->BSIM4v4wkt1l = value->rValue;
            mod->BSIM4v4wkt1lGiven = TRUE;
            break;
        case  BSIM4v4_MOD_WKT2:
            mod->BSIM4v4wkt2 = value->rValue;
            mod->BSIM4v4wkt2Given = TRUE;
            break;
        case  BSIM4v4_MOD_WK2:
            mod->BSIM4v4wk2 = value->rValue;
            mod->BSIM4v4wk2Given = TRUE;
            break;
        case  BSIM4v4_MOD_WK3:
            mod->BSIM4v4wk3 = value->rValue;
            mod->BSIM4v4wk3Given = TRUE;
            break;
        case  BSIM4v4_MOD_WK3B:
            mod->BSIM4v4wk3b = value->rValue;
            mod->BSIM4v4wk3bGiven = TRUE;
            break;
        case  BSIM4v4_MOD_WLPE0:
            mod->BSIM4v4wlpe0 = value->rValue;
            mod->BSIM4v4wlpe0Given = TRUE;
            break;
        case  BSIM4v4_MOD_WLPEB:
            mod->BSIM4v4wlpeb = value->rValue;
            mod->BSIM4v4wlpebGiven = TRUE;
            break;
        case  BSIM4v4_MOD_WDVTP0:
            mod->BSIM4v4wdvtp0 = value->rValue;
            mod->BSIM4v4wdvtp0Given = TRUE;
            break;
        case  BSIM4v4_MOD_WDVTP1:
            mod->BSIM4v4wdvtp1 = value->rValue;
            mod->BSIM4v4wdvtp1Given = TRUE;
            break;
        case  BSIM4v4_MOD_WW0:
            mod->BSIM4v4ww0 = value->rValue;
            mod->BSIM4v4ww0Given = TRUE;
            break;
        case  BSIM4v4_MOD_WDVT0:
            mod->BSIM4v4wdvt0 = value->rValue;
            mod->BSIM4v4wdvt0Given = TRUE;
            break;
        case  BSIM4v4_MOD_WDVT1:
            mod->BSIM4v4wdvt1 = value->rValue;
            mod->BSIM4v4wdvt1Given = TRUE;
            break;
        case  BSIM4v4_MOD_WDVT2:
            mod->BSIM4v4wdvt2 = value->rValue;
            mod->BSIM4v4wdvt2Given = TRUE;
            break;
        case  BSIM4v4_MOD_WDVT0W:
            mod->BSIM4v4wdvt0w = value->rValue;
            mod->BSIM4v4wdvt0wGiven = TRUE;
            break;
        case  BSIM4v4_MOD_WDVT1W:
            mod->BSIM4v4wdvt1w = value->rValue;
            mod->BSIM4v4wdvt1wGiven = TRUE;
            break;
        case  BSIM4v4_MOD_WDVT2W:
            mod->BSIM4v4wdvt2w = value->rValue;
            mod->BSIM4v4wdvt2wGiven = TRUE;
            break;
        case  BSIM4v4_MOD_WDROUT:
            mod->BSIM4v4wdrout = value->rValue;
            mod->BSIM4v4wdroutGiven = TRUE;
            break;
        case  BSIM4v4_MOD_WDSUB:
            mod->BSIM4v4wdsub = value->rValue;
            mod->BSIM4v4wdsubGiven = TRUE;
            break;
        case BSIM4v4_MOD_WVTH0:
            mod->BSIM4v4wvth0 = value->rValue;
            mod->BSIM4v4wvth0Given = TRUE;
            break;
        case BSIM4v4_MOD_WUA:
            mod->BSIM4v4wua = value->rValue;
            mod->BSIM4v4wuaGiven = TRUE;
            break;
        case BSIM4v4_MOD_WUA1:
            mod->BSIM4v4wua1 = value->rValue;
            mod->BSIM4v4wua1Given = TRUE;
            break;
        case BSIM4v4_MOD_WUB:
            mod->BSIM4v4wub = value->rValue;
            mod->BSIM4v4wubGiven = TRUE;
            break;
        case BSIM4v4_MOD_WUB1:
            mod->BSIM4v4wub1 = value->rValue;
            mod->BSIM4v4wub1Given = TRUE;
            break;
        case BSIM4v4_MOD_WUC:
            mod->BSIM4v4wuc = value->rValue;
            mod->BSIM4v4wucGiven = TRUE;
            break;
        case BSIM4v4_MOD_WUC1:
            mod->BSIM4v4wuc1 = value->rValue;
            mod->BSIM4v4wuc1Given = TRUE;
            break;
        case  BSIM4v4_MOD_WU0 :
            mod->BSIM4v4wu0 = value->rValue;
            mod->BSIM4v4wu0Given = TRUE;
            break;
        case  BSIM4v4_MOD_WUTE :
            mod->BSIM4v4wute = value->rValue;
            mod->BSIM4v4wuteGiven = TRUE;
            break;
        case BSIM4v4_MOD_WVOFF:
            mod->BSIM4v4wvoff = value->rValue;
            mod->BSIM4v4wvoffGiven = TRUE;
            break;
        case BSIM4v4_MOD_WMINV:
            mod->BSIM4v4wminv = value->rValue;
            mod->BSIM4v4wminvGiven = TRUE;
            break;
        case BSIM4v4_MOD_WFPROUT:
            mod->BSIM4v4wfprout = value->rValue;
            mod->BSIM4v4wfproutGiven = TRUE;
            break;
        case BSIM4v4_MOD_WPDITS:
            mod->BSIM4v4wpdits = value->rValue;
            mod->BSIM4v4wpditsGiven = TRUE;
            break;
        case BSIM4v4_MOD_WPDITSD:
            mod->BSIM4v4wpditsd = value->rValue;
            mod->BSIM4v4wpditsdGiven = TRUE;
            break;
        case  BSIM4v4_MOD_WDELTA :
            mod->BSIM4v4wdelta = value->rValue;
            mod->BSIM4v4wdeltaGiven = TRUE;
            break;
        case BSIM4v4_MOD_WRDSW:
            mod->BSIM4v4wrdsw = value->rValue;
            mod->BSIM4v4wrdswGiven = TRUE;
            break;
        case BSIM4v4_MOD_WRDW:
            mod->BSIM4v4wrdw = value->rValue;
            mod->BSIM4v4wrdwGiven = TRUE;
            break;
        case BSIM4v4_MOD_WRSW:
            mod->BSIM4v4wrsw = value->rValue;
            mod->BSIM4v4wrswGiven = TRUE;
            break;
        case BSIM4v4_MOD_WPRWB:
            mod->BSIM4v4wprwb = value->rValue;
            mod->BSIM4v4wprwbGiven = TRUE;
            break;
        case BSIM4v4_MOD_WPRWG:
            mod->BSIM4v4wprwg = value->rValue;
            mod->BSIM4v4wprwgGiven = TRUE;
            break;
        case BSIM4v4_MOD_WPRT:
            mod->BSIM4v4wprt = value->rValue;
            mod->BSIM4v4wprtGiven = TRUE;
            break;
        case BSIM4v4_MOD_WETA0:
            mod->BSIM4v4weta0 = value->rValue;
            mod->BSIM4v4weta0Given = TRUE;
            break;
        case BSIM4v4_MOD_WETAB:
            mod->BSIM4v4wetab = value->rValue;
            mod->BSIM4v4wetabGiven = TRUE;
            break;
        case BSIM4v4_MOD_WPCLM:
            mod->BSIM4v4wpclm = value->rValue;
            mod->BSIM4v4wpclmGiven = TRUE;
            break;
        case BSIM4v4_MOD_WPDIBL1:
            mod->BSIM4v4wpdibl1 = value->rValue;
            mod->BSIM4v4wpdibl1Given = TRUE;
            break;
        case BSIM4v4_MOD_WPDIBL2:
            mod->BSIM4v4wpdibl2 = value->rValue;
            mod->BSIM4v4wpdibl2Given = TRUE;
            break;
        case BSIM4v4_MOD_WPDIBLB:
            mod->BSIM4v4wpdiblb = value->rValue;
            mod->BSIM4v4wpdiblbGiven = TRUE;
            break;
        case BSIM4v4_MOD_WPSCBE1:
            mod->BSIM4v4wpscbe1 = value->rValue;
            mod->BSIM4v4wpscbe1Given = TRUE;
            break;
        case BSIM4v4_MOD_WPSCBE2:
            mod->BSIM4v4wpscbe2 = value->rValue;
            mod->BSIM4v4wpscbe2Given = TRUE;
            break;
        case BSIM4v4_MOD_WPVAG:
            mod->BSIM4v4wpvag = value->rValue;
            mod->BSIM4v4wpvagGiven = TRUE;
            break;
        case  BSIM4v4_MOD_WWR :
            mod->BSIM4v4wwr = value->rValue;
            mod->BSIM4v4wwrGiven = TRUE;
            break;
        case  BSIM4v4_MOD_WDWG :
            mod->BSIM4v4wdwg = value->rValue;
            mod->BSIM4v4wdwgGiven = TRUE;
            break;
        case  BSIM4v4_MOD_WDWB :
            mod->BSIM4v4wdwb = value->rValue;
            mod->BSIM4v4wdwbGiven = TRUE;
            break;
        case  BSIM4v4_MOD_WB0 :
            mod->BSIM4v4wb0 = value->rValue;
            mod->BSIM4v4wb0Given = TRUE;
            break;
        case  BSIM4v4_MOD_WB1 :
            mod->BSIM4v4wb1 = value->rValue;
            mod->BSIM4v4wb1Given = TRUE;
            break;
        case  BSIM4v4_MOD_WALPHA0 :
            mod->BSIM4v4walpha0 = value->rValue;
            mod->BSIM4v4walpha0Given = TRUE;
            break;
        case  BSIM4v4_MOD_WALPHA1 :
            mod->BSIM4v4walpha1 = value->rValue;
            mod->BSIM4v4walpha1Given = TRUE;
            break;
        case  BSIM4v4_MOD_WBETA0 :
            mod->BSIM4v4wbeta0 = value->rValue;
            mod->BSIM4v4wbeta0Given = TRUE;
            break;
        case  BSIM4v4_MOD_WAGIDL :
            mod->BSIM4v4wagidl = value->rValue;
            mod->BSIM4v4wagidlGiven = TRUE;
            break;
        case  BSIM4v4_MOD_WBGIDL :
            mod->BSIM4v4wbgidl = value->rValue;
            mod->BSIM4v4wbgidlGiven = TRUE;
            break;
        case  BSIM4v4_MOD_WCGIDL :
            mod->BSIM4v4wcgidl = value->rValue;
            mod->BSIM4v4wcgidlGiven = TRUE;
            break;
        case  BSIM4v4_MOD_WPHIN :
            mod->BSIM4v4wphin = value->rValue;
            mod->BSIM4v4wphinGiven = TRUE;
            break;
        case  BSIM4v4_MOD_WEGIDL :
            mod->BSIM4v4wegidl = value->rValue;
            mod->BSIM4v4wegidlGiven = TRUE;
            break;
        case  BSIM4v4_MOD_WAIGC :
            mod->BSIM4v4waigc = value->rValue;
            mod->BSIM4v4waigcGiven = TRUE;
            break;
        case  BSIM4v4_MOD_WBIGC :
            mod->BSIM4v4wbigc = value->rValue;
            mod->BSIM4v4wbigcGiven = TRUE;
            break;
        case  BSIM4v4_MOD_WCIGC :
            mod->BSIM4v4wcigc = value->rValue;
            mod->BSIM4v4wcigcGiven = TRUE;
            break;
        case  BSIM4v4_MOD_WAIGSD :
            mod->BSIM4v4waigsd = value->rValue;
            mod->BSIM4v4waigsdGiven = TRUE;
            break;
        case  BSIM4v4_MOD_WBIGSD :
            mod->BSIM4v4wbigsd = value->rValue;
            mod->BSIM4v4wbigsdGiven = TRUE;
            break;
        case  BSIM4v4_MOD_WCIGSD :
            mod->BSIM4v4wcigsd = value->rValue;
            mod->BSIM4v4wcigsdGiven = TRUE;
            break;
        case  BSIM4v4_MOD_WAIGBACC :
            mod->BSIM4v4waigbacc = value->rValue;
            mod->BSIM4v4waigbaccGiven = TRUE;
            break;
        case  BSIM4v4_MOD_WBIGBACC :
            mod->BSIM4v4wbigbacc = value->rValue;
            mod->BSIM4v4wbigbaccGiven = TRUE;
            break;
        case  BSIM4v4_MOD_WCIGBACC :
            mod->BSIM4v4wcigbacc = value->rValue;
            mod->BSIM4v4wcigbaccGiven = TRUE;
            break;
        case  BSIM4v4_MOD_WAIGBINV :
            mod->BSIM4v4waigbinv = value->rValue;
            mod->BSIM4v4waigbinvGiven = TRUE;
            break;
        case  BSIM4v4_MOD_WBIGBINV :
            mod->BSIM4v4wbigbinv = value->rValue;
            mod->BSIM4v4wbigbinvGiven = TRUE;
            break;
        case  BSIM4v4_MOD_WCIGBINV :
            mod->BSIM4v4wcigbinv = value->rValue;
            mod->BSIM4v4wcigbinvGiven = TRUE;
            break;
        case  BSIM4v4_MOD_WNIGC :
            mod->BSIM4v4wnigc = value->rValue;
            mod->BSIM4v4wnigcGiven = TRUE;
            break;
        case  BSIM4v4_MOD_WNIGBINV :
            mod->BSIM4v4wnigbinv = value->rValue;
            mod->BSIM4v4wnigbinvGiven = TRUE;
            break;
        case  BSIM4v4_MOD_WNIGBACC :
            mod->BSIM4v4wnigbacc = value->rValue;
            mod->BSIM4v4wnigbaccGiven = TRUE;
            break;
        case  BSIM4v4_MOD_WNTOX :
            mod->BSIM4v4wntox = value->rValue;
            mod->BSIM4v4wntoxGiven = TRUE;
            break;
        case  BSIM4v4_MOD_WEIGBINV :
            mod->BSIM4v4weigbinv = value->rValue;
            mod->BSIM4v4weigbinvGiven = TRUE;
            break;
        case  BSIM4v4_MOD_WPIGCD :
            mod->BSIM4v4wpigcd = value->rValue;
            mod->BSIM4v4wpigcdGiven = TRUE;
            break;
        case  BSIM4v4_MOD_WPOXEDGE :
            mod->BSIM4v4wpoxedge = value->rValue;
            mod->BSIM4v4wpoxedgeGiven = TRUE;
            break;
        case  BSIM4v4_MOD_WXRCRG1 :
            mod->BSIM4v4wxrcrg1 = value->rValue;
            mod->BSIM4v4wxrcrg1Given = TRUE;
            break;
        case  BSIM4v4_MOD_WXRCRG2 :
            mod->BSIM4v4wxrcrg2 = value->rValue;
            mod->BSIM4v4wxrcrg2Given = TRUE;
            break;
        case  BSIM4v4_MOD_WLAMBDA :
            mod->BSIM4v4wlambda = value->rValue;
            mod->BSIM4v4wlambdaGiven = TRUE;
            break;
        case  BSIM4v4_MOD_WVTL :
            mod->BSIM4v4wvtl = value->rValue;
            mod->BSIM4v4wvtlGiven = TRUE;
            break;
        case  BSIM4v4_MOD_WXN:
            mod->BSIM4v4wxn = value->rValue;
            mod->BSIM4v4wxnGiven = TRUE;
            break;
        case  BSIM4v4_MOD_WVFBSDOFF:
            mod->BSIM4v4wvfbsdoff = value->rValue;
            mod->BSIM4v4wvfbsdoffGiven = TRUE;
            break;
        case  BSIM4v4_MOD_WEU :
            mod->BSIM4v4weu = value->rValue;
            mod->BSIM4v4weuGiven = TRUE;
            break;
        case  BSIM4v4_MOD_WVFB :
            mod->BSIM4v4wvfb = value->rValue;
            mod->BSIM4v4wvfbGiven = TRUE;
            break;
        case  BSIM4v4_MOD_WCGSL :
            mod->BSIM4v4wcgsl = value->rValue;
            mod->BSIM4v4wcgslGiven = TRUE;
            break;
        case  BSIM4v4_MOD_WCGDL :
            mod->BSIM4v4wcgdl = value->rValue;
            mod->BSIM4v4wcgdlGiven = TRUE;
            break;
        case  BSIM4v4_MOD_WCKAPPAS :
            mod->BSIM4v4wckappas = value->rValue;
            mod->BSIM4v4wckappasGiven = TRUE;
            break;
        case  BSIM4v4_MOD_WCKAPPAD :
            mod->BSIM4v4wckappad = value->rValue;
            mod->BSIM4v4wckappadGiven = TRUE;
            break;
        case  BSIM4v4_MOD_WCF :
            mod->BSIM4v4wcf = value->rValue;
            mod->BSIM4v4wcfGiven = TRUE;
            break;
        case  BSIM4v4_MOD_WCLC :
            mod->BSIM4v4wclc = value->rValue;
            mod->BSIM4v4wclcGiven = TRUE;
            break;
        case  BSIM4v4_MOD_WCLE :
            mod->BSIM4v4wcle = value->rValue;
            mod->BSIM4v4wcleGiven = TRUE;
            break;
        case  BSIM4v4_MOD_WVFBCV :
            mod->BSIM4v4wvfbcv = value->rValue;
            mod->BSIM4v4wvfbcvGiven = TRUE;
            break;
        case  BSIM4v4_MOD_WACDE :
            mod->BSIM4v4wacde = value->rValue;
            mod->BSIM4v4wacdeGiven = TRUE;
            break;
        case  BSIM4v4_MOD_WMOIN :
            mod->BSIM4v4wmoin = value->rValue;
            mod->BSIM4v4wmoinGiven = TRUE;
            break;
        case  BSIM4v4_MOD_WNOFF :
            mod->BSIM4v4wnoff = value->rValue;
            mod->BSIM4v4wnoffGiven = TRUE;
            break;
        case  BSIM4v4_MOD_WVOFFCV :
            mod->BSIM4v4wvoffcv = value->rValue;
            mod->BSIM4v4wvoffcvGiven = TRUE;
            break;

	/* Cross-term dependence */
        case  BSIM4v4_MOD_PCDSC :
            mod->BSIM4v4pcdsc = value->rValue;
            mod->BSIM4v4pcdscGiven = TRUE;
            break;


        case  BSIM4v4_MOD_PCDSCB :
            mod->BSIM4v4pcdscb = value->rValue;
            mod->BSIM4v4pcdscbGiven = TRUE;
            break;
        case  BSIM4v4_MOD_PCDSCD :
            mod->BSIM4v4pcdscd = value->rValue;
            mod->BSIM4v4pcdscdGiven = TRUE;
            break;
        case  BSIM4v4_MOD_PCIT :
            mod->BSIM4v4pcit = value->rValue;
            mod->BSIM4v4pcitGiven = TRUE;
            break;
        case  BSIM4v4_MOD_PNFACTOR :
            mod->BSIM4v4pnfactor = value->rValue;
            mod->BSIM4v4pnfactorGiven = TRUE;
            break;
        case BSIM4v4_MOD_PXJ:
            mod->BSIM4v4pxj = value->rValue;
            mod->BSIM4v4pxjGiven = TRUE;
            break;
        case BSIM4v4_MOD_PVSAT:
            mod->BSIM4v4pvsat = value->rValue;
            mod->BSIM4v4pvsatGiven = TRUE;
            break;


        case BSIM4v4_MOD_PA0:
            mod->BSIM4v4pa0 = value->rValue;
            mod->BSIM4v4pa0Given = TRUE;
            break;
        case BSIM4v4_MOD_PAGS:
            mod->BSIM4v4pags = value->rValue;
            mod->BSIM4v4pagsGiven = TRUE;
            break;
        case BSIM4v4_MOD_PA1:
            mod->BSIM4v4pa1 = value->rValue;
            mod->BSIM4v4pa1Given = TRUE;
            break;
        case BSIM4v4_MOD_PA2:
            mod->BSIM4v4pa2 = value->rValue;
            mod->BSIM4v4pa2Given = TRUE;
            break;
        case BSIM4v4_MOD_PAT:
            mod->BSIM4v4pat = value->rValue;
            mod->BSIM4v4patGiven = TRUE;
            break;
        case BSIM4v4_MOD_PKETA:
            mod->BSIM4v4pketa = value->rValue;
            mod->BSIM4v4pketaGiven = TRUE;
            break;
        case BSIM4v4_MOD_PNSUB:
            mod->BSIM4v4pnsub = value->rValue;
            mod->BSIM4v4pnsubGiven = TRUE;
            break;
        case BSIM4v4_MOD_PNDEP:
            mod->BSIM4v4pndep = value->rValue;
            mod->BSIM4v4pndepGiven = TRUE;
	    if (mod->BSIM4v4pndep > 1.0e20)
		mod->BSIM4v4pndep *= 1.0e-6;
            break;
        case BSIM4v4_MOD_PNSD:
            mod->BSIM4v4pnsd = value->rValue;
            mod->BSIM4v4pnsdGiven = TRUE;
            if (mod->BSIM4v4pnsd > 1.0e23)
                mod->BSIM4v4pnsd *= 1.0e-6;
            break;
        case BSIM4v4_MOD_PNGATE:
            mod->BSIM4v4pngate = value->rValue;
            mod->BSIM4v4pngateGiven = TRUE;
	    if (mod->BSIM4v4pngate > 1.0e23)
		mod->BSIM4v4pngate *= 1.0e-6;
            break;
        case BSIM4v4_MOD_PGAMMA1:
            mod->BSIM4v4pgamma1 = value->rValue;
            mod->BSIM4v4pgamma1Given = TRUE;
            break;
        case BSIM4v4_MOD_PGAMMA2:
            mod->BSIM4v4pgamma2 = value->rValue;
            mod->BSIM4v4pgamma2Given = TRUE;
            break;
        case BSIM4v4_MOD_PVBX:
            mod->BSIM4v4pvbx = value->rValue;
            mod->BSIM4v4pvbxGiven = TRUE;
            break;
        case BSIM4v4_MOD_PVBM:
            mod->BSIM4v4pvbm = value->rValue;
            mod->BSIM4v4pvbmGiven = TRUE;
            break;
        case BSIM4v4_MOD_PXT:
            mod->BSIM4v4pxt = value->rValue;
            mod->BSIM4v4pxtGiven = TRUE;
            break;
        case  BSIM4v4_MOD_PK1:
            mod->BSIM4v4pk1 = value->rValue;
            mod->BSIM4v4pk1Given = TRUE;
            break;
        case  BSIM4v4_MOD_PKT1:
            mod->BSIM4v4pkt1 = value->rValue;
            mod->BSIM4v4pkt1Given = TRUE;
            break;
        case  BSIM4v4_MOD_PKT1L:
            mod->BSIM4v4pkt1l = value->rValue;
            mod->BSIM4v4pkt1lGiven = TRUE;
            break;
        case  BSIM4v4_MOD_PKT2:
            mod->BSIM4v4pkt2 = value->rValue;
            mod->BSIM4v4pkt2Given = TRUE;
            break;
        case  BSIM4v4_MOD_PK2:
            mod->BSIM4v4pk2 = value->rValue;
            mod->BSIM4v4pk2Given = TRUE;
            break;
        case  BSIM4v4_MOD_PK3:
            mod->BSIM4v4pk3 = value->rValue;
            mod->BSIM4v4pk3Given = TRUE;
            break;
        case  BSIM4v4_MOD_PK3B:
            mod->BSIM4v4pk3b = value->rValue;
            mod->BSIM4v4pk3bGiven = TRUE;
            break;
        case  BSIM4v4_MOD_PLPE0:
            mod->BSIM4v4plpe0 = value->rValue;
            mod->BSIM4v4plpe0Given = TRUE;
            break;
        case  BSIM4v4_MOD_PLPEB:
            mod->BSIM4v4plpeb = value->rValue;
            mod->BSIM4v4plpebGiven = TRUE;
            break;
        case  BSIM4v4_MOD_PDVTP0:
            mod->BSIM4v4pdvtp0 = value->rValue;
            mod->BSIM4v4pdvtp0Given = TRUE;
            break;
        case  BSIM4v4_MOD_PDVTP1:
            mod->BSIM4v4pdvtp1 = value->rValue;
            mod->BSIM4v4pdvtp1Given = TRUE;
            break;
        case  BSIM4v4_MOD_PW0:
            mod->BSIM4v4pw0 = value->rValue;
            mod->BSIM4v4pw0Given = TRUE;
            break;
        case  BSIM4v4_MOD_PDVT0:
            mod->BSIM4v4pdvt0 = value->rValue;
            mod->BSIM4v4pdvt0Given = TRUE;
            break;
        case  BSIM4v4_MOD_PDVT1:
            mod->BSIM4v4pdvt1 = value->rValue;
            mod->BSIM4v4pdvt1Given = TRUE;
            break;
        case  BSIM4v4_MOD_PDVT2:
            mod->BSIM4v4pdvt2 = value->rValue;
            mod->BSIM4v4pdvt2Given = TRUE;
            break;
        case  BSIM4v4_MOD_PDVT0W:
            mod->BSIM4v4pdvt0w = value->rValue;
            mod->BSIM4v4pdvt0wGiven = TRUE;
            break;
        case  BSIM4v4_MOD_PDVT1W:
            mod->BSIM4v4pdvt1w = value->rValue;
            mod->BSIM4v4pdvt1wGiven = TRUE;
            break;
        case  BSIM4v4_MOD_PDVT2W:
            mod->BSIM4v4pdvt2w = value->rValue;
            mod->BSIM4v4pdvt2wGiven = TRUE;
            break;
        case  BSIM4v4_MOD_PDROUT:
            mod->BSIM4v4pdrout = value->rValue;
            mod->BSIM4v4pdroutGiven = TRUE;
            break;
        case  BSIM4v4_MOD_PDSUB:
            mod->BSIM4v4pdsub = value->rValue;
            mod->BSIM4v4pdsubGiven = TRUE;
            break;
        case BSIM4v4_MOD_PVTH0:
            mod->BSIM4v4pvth0 = value->rValue;
            mod->BSIM4v4pvth0Given = TRUE;
            break;
        case BSIM4v4_MOD_PUA:
            mod->BSIM4v4pua = value->rValue;
            mod->BSIM4v4puaGiven = TRUE;
            break;
        case BSIM4v4_MOD_PUA1:
            mod->BSIM4v4pua1 = value->rValue;
            mod->BSIM4v4pua1Given = TRUE;
            break;
        case BSIM4v4_MOD_PUB:
            mod->BSIM4v4pub = value->rValue;
            mod->BSIM4v4pubGiven = TRUE;
            break;
        case BSIM4v4_MOD_PUB1:
            mod->BSIM4v4pub1 = value->rValue;
            mod->BSIM4v4pub1Given = TRUE;
            break;
        case BSIM4v4_MOD_PUC:
            mod->BSIM4v4puc = value->rValue;
            mod->BSIM4v4pucGiven = TRUE;
            break;
        case BSIM4v4_MOD_PUC1:
            mod->BSIM4v4puc1 = value->rValue;
            mod->BSIM4v4puc1Given = TRUE;
            break;
        case  BSIM4v4_MOD_PU0 :
            mod->BSIM4v4pu0 = value->rValue;
            mod->BSIM4v4pu0Given = TRUE;
            break;
        case  BSIM4v4_MOD_PUTE :
            mod->BSIM4v4pute = value->rValue;
            mod->BSIM4v4puteGiven = TRUE;
            break;
        case BSIM4v4_MOD_PVOFF:
            mod->BSIM4v4pvoff = value->rValue;
            mod->BSIM4v4pvoffGiven = TRUE;
            break;
        case BSIM4v4_MOD_PMINV:
            mod->BSIM4v4pminv = value->rValue;
            mod->BSIM4v4pminvGiven = TRUE;
            break;
        case BSIM4v4_MOD_PFPROUT:
            mod->BSIM4v4pfprout = value->rValue;
            mod->BSIM4v4pfproutGiven = TRUE;
            break;
        case BSIM4v4_MOD_PPDITS:
            mod->BSIM4v4ppdits = value->rValue;
            mod->BSIM4v4ppditsGiven = TRUE;
            break;
        case BSIM4v4_MOD_PPDITSD:
            mod->BSIM4v4ppditsd = value->rValue;
            mod->BSIM4v4ppditsdGiven = TRUE;
            break;
        case  BSIM4v4_MOD_PDELTA :
            mod->BSIM4v4pdelta = value->rValue;
            mod->BSIM4v4pdeltaGiven = TRUE;
            break;
        case BSIM4v4_MOD_PRDSW:
            mod->BSIM4v4prdsw = value->rValue;
            mod->BSIM4v4prdswGiven = TRUE;
            break;
        case BSIM4v4_MOD_PRDW:
            mod->BSIM4v4prdw = value->rValue;
            mod->BSIM4v4prdwGiven = TRUE;
            break;
        case BSIM4v4_MOD_PRSW:
            mod->BSIM4v4prsw = value->rValue;
            mod->BSIM4v4prswGiven = TRUE;
            break;
        case BSIM4v4_MOD_PPRWB:
            mod->BSIM4v4pprwb = value->rValue;
            mod->BSIM4v4pprwbGiven = TRUE;
            break;
        case BSIM4v4_MOD_PPRWG:
            mod->BSIM4v4pprwg = value->rValue;
            mod->BSIM4v4pprwgGiven = TRUE;
            break;
        case BSIM4v4_MOD_PPRT:
            mod->BSIM4v4pprt = value->rValue;
            mod->BSIM4v4pprtGiven = TRUE;
            break;
        case BSIM4v4_MOD_PETA0:
            mod->BSIM4v4peta0 = value->rValue;
            mod->BSIM4v4peta0Given = TRUE;
            break;
        case BSIM4v4_MOD_PETAB:
            mod->BSIM4v4petab = value->rValue;
            mod->BSIM4v4petabGiven = TRUE;
            break;
        case BSIM4v4_MOD_PPCLM:
            mod->BSIM4v4ppclm = value->rValue;
            mod->BSIM4v4ppclmGiven = TRUE;
            break;
        case BSIM4v4_MOD_PPDIBL1:
            mod->BSIM4v4ppdibl1 = value->rValue;
            mod->BSIM4v4ppdibl1Given = TRUE;
            break;
        case BSIM4v4_MOD_PPDIBL2:
            mod->BSIM4v4ppdibl2 = value->rValue;
            mod->BSIM4v4ppdibl2Given = TRUE;
            break;
        case BSIM4v4_MOD_PPDIBLB:
            mod->BSIM4v4ppdiblb = value->rValue;
            mod->BSIM4v4ppdiblbGiven = TRUE;
            break;
        case BSIM4v4_MOD_PPSCBE1:
            mod->BSIM4v4ppscbe1 = value->rValue;
            mod->BSIM4v4ppscbe1Given = TRUE;
            break;
        case BSIM4v4_MOD_PPSCBE2:
            mod->BSIM4v4ppscbe2 = value->rValue;
            mod->BSIM4v4ppscbe2Given = TRUE;
            break;
        case BSIM4v4_MOD_PPVAG:
            mod->BSIM4v4ppvag = value->rValue;
            mod->BSIM4v4ppvagGiven = TRUE;
            break;
        case  BSIM4v4_MOD_PWR :
            mod->BSIM4v4pwr = value->rValue;
            mod->BSIM4v4pwrGiven = TRUE;
            break;
        case  BSIM4v4_MOD_PDWG :
            mod->BSIM4v4pdwg = value->rValue;
            mod->BSIM4v4pdwgGiven = TRUE;
            break;
        case  BSIM4v4_MOD_PDWB :
            mod->BSIM4v4pdwb = value->rValue;
            mod->BSIM4v4pdwbGiven = TRUE;
            break;
        case  BSIM4v4_MOD_PB0 :
            mod->BSIM4v4pb0 = value->rValue;
            mod->BSIM4v4pb0Given = TRUE;
            break;
        case  BSIM4v4_MOD_PB1 :
            mod->BSIM4v4pb1 = value->rValue;
            mod->BSIM4v4pb1Given = TRUE;
            break;
        case  BSIM4v4_MOD_PALPHA0 :
            mod->BSIM4v4palpha0 = value->rValue;
            mod->BSIM4v4palpha0Given = TRUE;
            break;
        case  BSIM4v4_MOD_PALPHA1 :
            mod->BSIM4v4palpha1 = value->rValue;
            mod->BSIM4v4palpha1Given = TRUE;
            break;
        case  BSIM4v4_MOD_PBETA0 :
            mod->BSIM4v4pbeta0 = value->rValue;
            mod->BSIM4v4pbeta0Given = TRUE;
            break;
        case  BSIM4v4_MOD_PAGIDL :
            mod->BSIM4v4pagidl = value->rValue;
            mod->BSIM4v4pagidlGiven = TRUE;
            break;
        case  BSIM4v4_MOD_PBGIDL :
            mod->BSIM4v4pbgidl = value->rValue;
            mod->BSIM4v4pbgidlGiven = TRUE;
            break;
        case  BSIM4v4_MOD_PCGIDL :
            mod->BSIM4v4pcgidl = value->rValue;
            mod->BSIM4v4pcgidlGiven = TRUE;
            break;
        case  BSIM4v4_MOD_PPHIN :
            mod->BSIM4v4pphin = value->rValue;
            mod->BSIM4v4pphinGiven = TRUE;
            break;
        case  BSIM4v4_MOD_PEGIDL :
            mod->BSIM4v4pegidl = value->rValue;
            mod->BSIM4v4pegidlGiven = TRUE;
            break;
        case  BSIM4v4_MOD_PAIGC :
            mod->BSIM4v4paigc = value->rValue;
            mod->BSIM4v4paigcGiven = TRUE;
            break;
        case  BSIM4v4_MOD_PBIGC :
            mod->BSIM4v4pbigc = value->rValue;
            mod->BSIM4v4pbigcGiven = TRUE;
            break;
        case  BSIM4v4_MOD_PCIGC :
            mod->BSIM4v4pcigc = value->rValue;
            mod->BSIM4v4pcigcGiven = TRUE;
            break;
        case  BSIM4v4_MOD_PAIGSD :
            mod->BSIM4v4paigsd = value->rValue;
            mod->BSIM4v4paigsdGiven = TRUE;
            break;
        case  BSIM4v4_MOD_PBIGSD :
            mod->BSIM4v4pbigsd = value->rValue;
            mod->BSIM4v4pbigsdGiven = TRUE;
            break;
        case  BSIM4v4_MOD_PCIGSD :
            mod->BSIM4v4pcigsd = value->rValue;
            mod->BSIM4v4pcigsdGiven = TRUE;
            break;
        case  BSIM4v4_MOD_PAIGBACC :
            mod->BSIM4v4paigbacc = value->rValue;
            mod->BSIM4v4paigbaccGiven = TRUE;
            break;
        case  BSIM4v4_MOD_PBIGBACC :
            mod->BSIM4v4pbigbacc = value->rValue;
            mod->BSIM4v4pbigbaccGiven = TRUE;
            break;
        case  BSIM4v4_MOD_PCIGBACC :
            mod->BSIM4v4pcigbacc = value->rValue;
            mod->BSIM4v4pcigbaccGiven = TRUE;
            break;
        case  BSIM4v4_MOD_PAIGBINV :
            mod->BSIM4v4paigbinv = value->rValue;
            mod->BSIM4v4paigbinvGiven = TRUE;
            break;
        case  BSIM4v4_MOD_PBIGBINV :
            mod->BSIM4v4pbigbinv = value->rValue;
            mod->BSIM4v4pbigbinvGiven = TRUE;
            break;
        case  BSIM4v4_MOD_PCIGBINV :
            mod->BSIM4v4pcigbinv = value->rValue;
            mod->BSIM4v4pcigbinvGiven = TRUE;
            break;
        case  BSIM4v4_MOD_PNIGC :
            mod->BSIM4v4pnigc = value->rValue;
            mod->BSIM4v4pnigcGiven = TRUE;
            break;
        case  BSIM4v4_MOD_PNIGBINV :
            mod->BSIM4v4pnigbinv = value->rValue;
            mod->BSIM4v4pnigbinvGiven = TRUE;
            break;
        case  BSIM4v4_MOD_PNIGBACC :
            mod->BSIM4v4pnigbacc = value->rValue;
            mod->BSIM4v4pnigbaccGiven = TRUE;
            break;
        case  BSIM4v4_MOD_PNTOX :
            mod->BSIM4v4pntox = value->rValue;
            mod->BSIM4v4pntoxGiven = TRUE;
            break;
        case  BSIM4v4_MOD_PEIGBINV :
            mod->BSIM4v4peigbinv = value->rValue;
            mod->BSIM4v4peigbinvGiven = TRUE;
            break;
        case  BSIM4v4_MOD_PPIGCD :
            mod->BSIM4v4ppigcd = value->rValue;
            mod->BSIM4v4ppigcdGiven = TRUE;
            break;
        case  BSIM4v4_MOD_PPOXEDGE :
            mod->BSIM4v4ppoxedge = value->rValue;
            mod->BSIM4v4ppoxedgeGiven = TRUE;
            break;
        case  BSIM4v4_MOD_PXRCRG1 :
            mod->BSIM4v4pxrcrg1 = value->rValue;
            mod->BSIM4v4pxrcrg1Given = TRUE;
            break;
        case  BSIM4v4_MOD_PXRCRG2 :
            mod->BSIM4v4pxrcrg2 = value->rValue;
            mod->BSIM4v4pxrcrg2Given = TRUE;
            break;
        case  BSIM4v4_MOD_PLAMBDA :
            mod->BSIM4v4plambda = value->rValue;
            mod->BSIM4v4plambdaGiven = TRUE;
            break;
        case  BSIM4v4_MOD_PVTL :
            mod->BSIM4v4pvtl = value->rValue;
            mod->BSIM4v4pvtlGiven = TRUE;
            break;
        case  BSIM4v4_MOD_PXN:
            mod->BSIM4v4pxn = value->rValue;
            mod->BSIM4v4pxnGiven = TRUE;
            break;
        case  BSIM4v4_MOD_PVFBSDOFF:
            mod->BSIM4v4pvfbsdoff = value->rValue;
            mod->BSIM4v4pvfbsdoffGiven = TRUE;
            break;
        case  BSIM4v4_MOD_PEU :
            mod->BSIM4v4peu = value->rValue;
            mod->BSIM4v4peuGiven = TRUE;
            break;
        case  BSIM4v4_MOD_PVFB :
            mod->BSIM4v4pvfb = value->rValue;
            mod->BSIM4v4pvfbGiven = TRUE;
            break;
        case  BSIM4v4_MOD_PCGSL :
            mod->BSIM4v4pcgsl = value->rValue;
            mod->BSIM4v4pcgslGiven = TRUE;
            break;
        case  BSIM4v4_MOD_PCGDL :
            mod->BSIM4v4pcgdl = value->rValue;
            mod->BSIM4v4pcgdlGiven = TRUE;
            break;
        case  BSIM4v4_MOD_PCKAPPAS :
            mod->BSIM4v4pckappas = value->rValue;
            mod->BSIM4v4pckappasGiven = TRUE;
            break;
        case  BSIM4v4_MOD_PCKAPPAD :
            mod->BSIM4v4pckappad = value->rValue;
            mod->BSIM4v4pckappadGiven = TRUE;
            break;
        case  BSIM4v4_MOD_PCF :
            mod->BSIM4v4pcf = value->rValue;
            mod->BSIM4v4pcfGiven = TRUE;
            break;
        case  BSIM4v4_MOD_PCLC :
            mod->BSIM4v4pclc = value->rValue;
            mod->BSIM4v4pclcGiven = TRUE;
            break;
        case  BSIM4v4_MOD_PCLE :
            mod->BSIM4v4pcle = value->rValue;
            mod->BSIM4v4pcleGiven = TRUE;
            break;
        case  BSIM4v4_MOD_PVFBCV :
            mod->BSIM4v4pvfbcv = value->rValue;
            mod->BSIM4v4pvfbcvGiven = TRUE;
            break;
        case  BSIM4v4_MOD_PACDE :
            mod->BSIM4v4pacde = value->rValue;
            mod->BSIM4v4pacdeGiven = TRUE;
            break;
        case  BSIM4v4_MOD_PMOIN :
            mod->BSIM4v4pmoin = value->rValue;
            mod->BSIM4v4pmoinGiven = TRUE;
            break;
        case  BSIM4v4_MOD_PNOFF :
            mod->BSIM4v4pnoff = value->rValue;
            mod->BSIM4v4pnoffGiven = TRUE;
            break;
        case  BSIM4v4_MOD_PVOFFCV :
            mod->BSIM4v4pvoffcv = value->rValue;
            mod->BSIM4v4pvoffcvGiven = TRUE;
            break;

        case  BSIM4v4_MOD_TNOM :
            mod->BSIM4v4tnom = value->rValue + CONSTCtoK;
            mod->BSIM4v4tnomGiven = TRUE;
            break;
        case  BSIM4v4_MOD_CGSO :
            mod->BSIM4v4cgso = value->rValue;
            mod->BSIM4v4cgsoGiven = TRUE;
            break;
        case  BSIM4v4_MOD_CGDO :
            mod->BSIM4v4cgdo = value->rValue;
            mod->BSIM4v4cgdoGiven = TRUE;
            break;
        case  BSIM4v4_MOD_CGBO :
            mod->BSIM4v4cgbo = value->rValue;
            mod->BSIM4v4cgboGiven = TRUE;
            break;
        case  BSIM4v4_MOD_XPART :
            mod->BSIM4v4xpart = value->rValue;
            mod->BSIM4v4xpartGiven = TRUE;
            break;
        case  BSIM4v4_MOD_RSH :
            mod->BSIM4v4sheetResistance = value->rValue;
            mod->BSIM4v4sheetResistanceGiven = TRUE;
            break;
        case  BSIM4v4_MOD_JSS :
            mod->BSIM4v4SjctSatCurDensity = value->rValue;
            mod->BSIM4v4SjctSatCurDensityGiven = TRUE;
            break;
        case  BSIM4v4_MOD_JSWS :
            mod->BSIM4v4SjctSidewallSatCurDensity = value->rValue;
            mod->BSIM4v4SjctSidewallSatCurDensityGiven = TRUE;
            break;
        case  BSIM4v4_MOD_JSWGS :
            mod->BSIM4v4SjctGateSidewallSatCurDensity = value->rValue;
            mod->BSIM4v4SjctGateSidewallSatCurDensityGiven = TRUE;
            break;
        case  BSIM4v4_MOD_PBS :
            mod->BSIM4v4SbulkJctPotential = value->rValue;
            mod->BSIM4v4SbulkJctPotentialGiven = TRUE;
            break;
        case  BSIM4v4_MOD_MJS :
            mod->BSIM4v4SbulkJctBotGradingCoeff = value->rValue;
            mod->BSIM4v4SbulkJctBotGradingCoeffGiven = TRUE;
            break;
        case  BSIM4v4_MOD_PBSWS :
            mod->BSIM4v4SsidewallJctPotential = value->rValue;
            mod->BSIM4v4SsidewallJctPotentialGiven = TRUE;
            break;
        case  BSIM4v4_MOD_MJSWS :
            mod->BSIM4v4SbulkJctSideGradingCoeff = value->rValue;
            mod->BSIM4v4SbulkJctSideGradingCoeffGiven = TRUE;
            break;
        case  BSIM4v4_MOD_CJS :
            mod->BSIM4v4SunitAreaJctCap = value->rValue;
            mod->BSIM4v4SunitAreaJctCapGiven = TRUE;
            break;
        case  BSIM4v4_MOD_CJSWS :
            mod->BSIM4v4SunitLengthSidewallJctCap = value->rValue;
            mod->BSIM4v4SunitLengthSidewallJctCapGiven = TRUE;
            break;
        case  BSIM4v4_MOD_NJS :
            mod->BSIM4v4SjctEmissionCoeff = value->rValue;
            mod->BSIM4v4SjctEmissionCoeffGiven = TRUE;
            break;
        case  BSIM4v4_MOD_PBSWGS :
            mod->BSIM4v4SGatesidewallJctPotential = value->rValue;
            mod->BSIM4v4SGatesidewallJctPotentialGiven = TRUE;
            break;
        case  BSIM4v4_MOD_MJSWGS :
            mod->BSIM4v4SbulkJctGateSideGradingCoeff = value->rValue;
            mod->BSIM4v4SbulkJctGateSideGradingCoeffGiven = TRUE;
            break;
        case  BSIM4v4_MOD_CJSWGS :
            mod->BSIM4v4SunitLengthGateSidewallJctCap = value->rValue;
            mod->BSIM4v4SunitLengthGateSidewallJctCapGiven = TRUE;
            break;
        case  BSIM4v4_MOD_XTIS :
            mod->BSIM4v4SjctTempExponent = value->rValue;
            mod->BSIM4v4SjctTempExponentGiven = TRUE;
            break;
        case  BSIM4v4_MOD_JSD :
            mod->BSIM4v4DjctSatCurDensity = value->rValue;
            mod->BSIM4v4DjctSatCurDensityGiven = TRUE;
            break;
        case  BSIM4v4_MOD_JSWD :
            mod->BSIM4v4DjctSidewallSatCurDensity = value->rValue;
            mod->BSIM4v4DjctSidewallSatCurDensityGiven = TRUE;
            break;
        case  BSIM4v4_MOD_JSWGD :
            mod->BSIM4v4DjctGateSidewallSatCurDensity = value->rValue;
            mod->BSIM4v4DjctGateSidewallSatCurDensityGiven = TRUE;
            break;
        case  BSIM4v4_MOD_PBD :
            mod->BSIM4v4DbulkJctPotential = value->rValue;
            mod->BSIM4v4DbulkJctPotentialGiven = TRUE;
            break;
        case  BSIM4v4_MOD_MJD :
            mod->BSIM4v4DbulkJctBotGradingCoeff = value->rValue;
            mod->BSIM4v4DbulkJctBotGradingCoeffGiven = TRUE;
            break;
        case  BSIM4v4_MOD_PBSWD :
            mod->BSIM4v4DsidewallJctPotential = value->rValue;
            mod->BSIM4v4DsidewallJctPotentialGiven = TRUE;
            break;
        case  BSIM4v4_MOD_MJSWD :
            mod->BSIM4v4DbulkJctSideGradingCoeff = value->rValue;
            mod->BSIM4v4DbulkJctSideGradingCoeffGiven = TRUE;
            break;
        case  BSIM4v4_MOD_CJD :
            mod->BSIM4v4DunitAreaJctCap = value->rValue;
            mod->BSIM4v4DunitAreaJctCapGiven = TRUE;
            break;
        case  BSIM4v4_MOD_CJSWD :
            mod->BSIM4v4DunitLengthSidewallJctCap = value->rValue;
            mod->BSIM4v4DunitLengthSidewallJctCapGiven = TRUE;
            break;
        case  BSIM4v4_MOD_NJD :
            mod->BSIM4v4DjctEmissionCoeff = value->rValue;
            mod->BSIM4v4DjctEmissionCoeffGiven = TRUE;
            break;
        case  BSIM4v4_MOD_PBSWGD :
            mod->BSIM4v4DGatesidewallJctPotential = value->rValue;
            mod->BSIM4v4DGatesidewallJctPotentialGiven = TRUE;
            break;
        case  BSIM4v4_MOD_MJSWGD :
            mod->BSIM4v4DbulkJctGateSideGradingCoeff = value->rValue;
            mod->BSIM4v4DbulkJctGateSideGradingCoeffGiven = TRUE;
            break;
        case  BSIM4v4_MOD_CJSWGD :
            mod->BSIM4v4DunitLengthGateSidewallJctCap = value->rValue;
            mod->BSIM4v4DunitLengthGateSidewallJctCapGiven = TRUE;
            break;
        case  BSIM4v4_MOD_XTID :
            mod->BSIM4v4DjctTempExponent = value->rValue;
            mod->BSIM4v4DjctTempExponentGiven = TRUE;
            break;
        case  BSIM4v4_MOD_LINT :
            mod->BSIM4v4Lint = value->rValue;
            mod->BSIM4v4LintGiven = TRUE;
            break;
        case  BSIM4v4_MOD_LL :
            mod->BSIM4v4Ll = value->rValue;
            mod->BSIM4v4LlGiven = TRUE;
            break;
        case  BSIM4v4_MOD_LLC :
            mod->BSIM4v4Llc = value->rValue;
            mod->BSIM4v4LlcGiven = TRUE;
            break;
        case  BSIM4v4_MOD_LLN :
            mod->BSIM4v4Lln = value->rValue;
            mod->BSIM4v4LlnGiven = TRUE;
            break;
        case  BSIM4v4_MOD_LW :
            mod->BSIM4v4Lw = value->rValue;
            mod->BSIM4v4LwGiven = TRUE;
            break;
        case  BSIM4v4_MOD_LWC :
            mod->BSIM4v4Lwc = value->rValue;
            mod->BSIM4v4LwcGiven = TRUE;
            break;
        case  BSIM4v4_MOD_LWN :
            mod->BSIM4v4Lwn = value->rValue;
            mod->BSIM4v4LwnGiven = TRUE;
            break;
        case  BSIM4v4_MOD_LWL :
            mod->BSIM4v4Lwl = value->rValue;
            mod->BSIM4v4LwlGiven = TRUE;
            break;
        case  BSIM4v4_MOD_LWLC :
            mod->BSIM4v4Lwlc = value->rValue;
            mod->BSIM4v4LwlcGiven = TRUE;
            break;
        case  BSIM4v4_MOD_LMIN :
            mod->BSIM4v4Lmin = value->rValue;
            mod->BSIM4v4LminGiven = TRUE;
            break;
        case  BSIM4v4_MOD_LMAX :
            mod->BSIM4v4Lmax = value->rValue;
            mod->BSIM4v4LmaxGiven = TRUE;
            break;
        case  BSIM4v4_MOD_WINT :
            mod->BSIM4v4Wint = value->rValue;
            mod->BSIM4v4WintGiven = TRUE;
            break;
        case  BSIM4v4_MOD_WL :
            mod->BSIM4v4Wl = value->rValue;
            mod->BSIM4v4WlGiven = TRUE;
            break;
        case  BSIM4v4_MOD_WLC :
            mod->BSIM4v4Wlc = value->rValue;
            mod->BSIM4v4WlcGiven = TRUE;
            break;
        case  BSIM4v4_MOD_WLN :
            mod->BSIM4v4Wln = value->rValue;
            mod->BSIM4v4WlnGiven = TRUE;
            break;
        case  BSIM4v4_MOD_WW :
            mod->BSIM4v4Ww = value->rValue;
            mod->BSIM4v4WwGiven = TRUE;
            break;
        case  BSIM4v4_MOD_WWC :
            mod->BSIM4v4Wwc = value->rValue;
            mod->BSIM4v4WwcGiven = TRUE;
            break;
        case  BSIM4v4_MOD_WWN :
            mod->BSIM4v4Wwn = value->rValue;
            mod->BSIM4v4WwnGiven = TRUE;
            break;
        case  BSIM4v4_MOD_WWL :
            mod->BSIM4v4Wwl = value->rValue;
            mod->BSIM4v4WwlGiven = TRUE;
            break;
        case  BSIM4v4_MOD_WWLC :
            mod->BSIM4v4Wwlc = value->rValue;
            mod->BSIM4v4WwlcGiven = TRUE;
            break;
        case  BSIM4v4_MOD_WMIN :
            mod->BSIM4v4Wmin = value->rValue;
            mod->BSIM4v4WminGiven = TRUE;
            break;
        case  BSIM4v4_MOD_WMAX :
            mod->BSIM4v4Wmax = value->rValue;
            mod->BSIM4v4WmaxGiven = TRUE;
            break;

        case  BSIM4v4_MOD_NOIA :
            mod->BSIM4v4oxideTrapDensityA = value->rValue;
            mod->BSIM4v4oxideTrapDensityAGiven = TRUE;
            break;
        case  BSIM4v4_MOD_NOIB :
            mod->BSIM4v4oxideTrapDensityB = value->rValue;
            mod->BSIM4v4oxideTrapDensityBGiven = TRUE;
            break;
        case  BSIM4v4_MOD_NOIC :
            mod->BSIM4v4oxideTrapDensityC = value->rValue;
            mod->BSIM4v4oxideTrapDensityCGiven = TRUE;
            break;
        case  BSIM4v4_MOD_EM :
            mod->BSIM4v4em = value->rValue;
            mod->BSIM4v4emGiven = TRUE;
            break;
        case  BSIM4v4_MOD_EF :
            mod->BSIM4v4ef = value->rValue;
            mod->BSIM4v4efGiven = TRUE;
            break;
        case  BSIM4v4_MOD_AF :
            mod->BSIM4v4af = value->rValue;
            mod->BSIM4v4afGiven = TRUE;
            break;
        case  BSIM4v4_MOD_KF :
            mod->BSIM4v4kf = value->rValue;
            mod->BSIM4v4kfGiven = TRUE;
            break;

        case BSIM4v4_MOD_VGS_MAX:
            mod->BSIM4v4vgsMax = value->rValue;
            mod->BSIM4v4vgsMaxGiven = TRUE;
            break;
        case BSIM4v4_MOD_VGD_MAX:
            mod->BSIM4v4vgdMax = value->rValue;
            mod->BSIM4v4vgdMaxGiven = TRUE;
            break;
        case BSIM4v4_MOD_VGB_MAX:
            mod->BSIM4v4vgbMax = value->rValue;
            mod->BSIM4v4vgbMaxGiven = TRUE;
            break;
        case BSIM4v4_MOD_VDS_MAX:
            mod->BSIM4v4vdsMax = value->rValue;
            mod->BSIM4v4vdsMaxGiven = TRUE;
            break;
        case BSIM4v4_MOD_VBS_MAX:
            mod->BSIM4v4vbsMax = value->rValue;
            mod->BSIM4v4vbsMaxGiven = TRUE;
            break;
        case BSIM4v4_MOD_VBD_MAX:
            mod->BSIM4v4vbdMax = value->rValue;
            mod->BSIM4v4vbdMaxGiven = TRUE;
            break;
        case BSIM4v4_MOD_VGSR_MAX:
            mod->BSIM4v4vgsrMax = value->rValue;
            mod->BSIM4v4vgsrMaxGiven = TRUE;
            break;
        case BSIM4v4_MOD_VGDR_MAX:
            mod->BSIM4v4vgdrMax = value->rValue;
            mod->BSIM4v4vgdrMaxGiven = TRUE;
            break;
        case BSIM4v4_MOD_VGBR_MAX:
            mod->BSIM4v4vgbrMax = value->rValue;
            mod->BSIM4v4vgbrMaxGiven = TRUE;
            break;
        case BSIM4v4_MOD_VBSR_MAX:
            mod->BSIM4v4vbsrMax = value->rValue;
            mod->BSIM4v4vbsrMaxGiven = TRUE;
            break;
        case BSIM4v4_MOD_VBDR_MAX:
            mod->BSIM4v4vbdrMax = value->rValue;
            mod->BSIM4v4vbdrMaxGiven = TRUE;
            break;

        case  BSIM4v4_MOD_NMOS  :
            if(value->iValue) {
                mod->BSIM4v4type = 1;
                mod->BSIM4v4typeGiven = TRUE;
            }
            break;
        case  BSIM4v4_MOD_PMOS  :
            if(value->iValue) {
                mod->BSIM4v4type = - 1;
                mod->BSIM4v4typeGiven = TRUE;
            }
            break;
        default:
            return(E_BADPARM);
    }
    return(OK);
}


