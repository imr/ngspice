/**** BSIM4.2.1, Released by Xuemei Xi 10/05/2001 ****/

/**********
 * Copyright 2001 Regents of the University of California. All rights reserved.
 * File: b4mpar.c of BSIM4.2.1.
 * Author: 2000 Weidong Liu
 * Authors: Xuemei Xi, Kanyu M. Cao, Hui Wan, Mansun Chan, Chenming Hu.
 * Project Director: Prof. Chenming Hu.
 *
 * Modified by Xuemei Xi 04/06/2001
 **********/

#include "ngspice.h"
#include <stdio.h>
#include "bsim4v2def.h"
#include "ifsim.h"
#include "sperror.h"
#include "const.h"

int
BSIM4v2mParam(
int param,
IFvalue *value,
GENmodel *inMod)
{
    BSIM4v2model *mod = (BSIM4v2model*)inMod;
    switch(param)
    {   case  BSIM4v2_MOD_MOBMOD :
            mod->BSIM4v2mobMod = value->iValue;
            mod->BSIM4v2mobModGiven = TRUE;
            break;
        case  BSIM4v2_MOD_BINUNIT :
            mod->BSIM4v2binUnit = value->iValue;
            mod->BSIM4v2binUnitGiven = TRUE;
            break;
        case  BSIM4v2_MOD_PARAMCHK :
            mod->BSIM4v2paramChk = value->iValue;
            mod->BSIM4v2paramChkGiven = TRUE;
            break;
        case  BSIM4v2_MOD_CAPMOD :
            mod->BSIM4v2capMod = value->iValue;
            mod->BSIM4v2capModGiven = TRUE;
            break;
        case  BSIM4v2_MOD_DIOMOD :
            mod->BSIM4v2dioMod = value->iValue;
            mod->BSIM4v2dioModGiven = TRUE;
            break;
        case  BSIM4v2_MOD_RDSMOD :
            mod->BSIM4v2rdsMod = value->iValue;
            mod->BSIM4v2rdsModGiven = TRUE;
            break;
        case  BSIM4v2_MOD_TRNQSMOD :
            mod->BSIM4v2trnqsMod = value->iValue;
            mod->BSIM4v2trnqsModGiven = TRUE;
            break;
        case  BSIM4v2_MOD_ACNQSMOD :
            mod->BSIM4v2acnqsMod = value->iValue;
            mod->BSIM4v2acnqsModGiven = TRUE;
            break;
        case  BSIM4v2_MOD_RBODYMOD :
            mod->BSIM4v2rbodyMod = value->iValue;
            mod->BSIM4v2rbodyModGiven = TRUE;
            break;
        case  BSIM4v2_MOD_RGATEMOD :
            mod->BSIM4v2rgateMod = value->iValue;
            mod->BSIM4v2rgateModGiven = TRUE;
            break;
        case  BSIM4v2_MOD_PERMOD :
            mod->BSIM4v2perMod = value->iValue;
            mod->BSIM4v2perModGiven = TRUE;
            break;
        case  BSIM4v2_MOD_GEOMOD :
            mod->BSIM4v2geoMod = value->iValue;
            mod->BSIM4v2geoModGiven = TRUE;
            break;
        case  BSIM4v2_MOD_FNOIMOD :
            mod->BSIM4v2fnoiMod = value->iValue;
            mod->BSIM4v2fnoiModGiven = TRUE;
            break;
        case  BSIM4v2_MOD_TNOIMOD :
            mod->BSIM4v2tnoiMod = value->iValue;
            mod->BSIM4v2tnoiModGiven = TRUE;
            break;
        case  BSIM4v2_MOD_IGCMOD :
            mod->BSIM4v2igcMod = value->iValue;
            mod->BSIM4v2igcModGiven = TRUE;
            break;
        case  BSIM4v2_MOD_IGBMOD :
            mod->BSIM4v2igbMod = value->iValue;
            mod->BSIM4v2igbModGiven = TRUE;
            break;
        case  BSIM4v2_MOD_VERSION :
            mod->BSIM4v2version = value->sValue;
            mod->BSIM4v2versionGiven = TRUE;
            break;
        case  BSIM4v2_MOD_TOXREF :
            mod->BSIM4v2toxref = value->rValue;
            mod->BSIM4v2toxrefGiven = TRUE;
            break;
        case  BSIM4v2_MOD_TOXE :
            mod->BSIM4v2toxe = value->rValue;
            mod->BSIM4v2toxeGiven = TRUE;
            break;
        case  BSIM4v2_MOD_TOXP :
            mod->BSIM4v2toxp = value->rValue;
            mod->BSIM4v2toxpGiven = TRUE;
            break;
        case  BSIM4v2_MOD_TOXM :
            mod->BSIM4v2toxm = value->rValue;
            mod->BSIM4v2toxmGiven = TRUE;
            break;
        case  BSIM4v2_MOD_DTOX :
            mod->BSIM4v2dtox = value->rValue;
            mod->BSIM4v2dtoxGiven = TRUE;
            break;
        case  BSIM4v2_MOD_EPSROX :
            mod->BSIM4v2epsrox = value->rValue;
            mod->BSIM4v2epsroxGiven = TRUE;
            break;

        case  BSIM4v2_MOD_CDSC :
            mod->BSIM4v2cdsc = value->rValue;
            mod->BSIM4v2cdscGiven = TRUE;
            break;
        case  BSIM4v2_MOD_CDSCB :
            mod->BSIM4v2cdscb = value->rValue;
            mod->BSIM4v2cdscbGiven = TRUE;
            break;

        case  BSIM4v2_MOD_CDSCD :
            mod->BSIM4v2cdscd = value->rValue;
            mod->BSIM4v2cdscdGiven = TRUE;
            break;

        case  BSIM4v2_MOD_CIT :
            mod->BSIM4v2cit = value->rValue;
            mod->BSIM4v2citGiven = TRUE;
            break;
        case  BSIM4v2_MOD_NFACTOR :
            mod->BSIM4v2nfactor = value->rValue;
            mod->BSIM4v2nfactorGiven = TRUE;
            break;
        case BSIM4v2_MOD_XJ:
            mod->BSIM4v2xj = value->rValue;
            mod->BSIM4v2xjGiven = TRUE;
            break;
        case BSIM4v2_MOD_VSAT:
            mod->BSIM4v2vsat = value->rValue;
            mod->BSIM4v2vsatGiven = TRUE;
            break;
        case BSIM4v2_MOD_A0:
            mod->BSIM4v2a0 = value->rValue;
            mod->BSIM4v2a0Given = TRUE;
            break;
        
        case BSIM4v2_MOD_AGS:
            mod->BSIM4v2ags= value->rValue;
            mod->BSIM4v2agsGiven = TRUE;
            break;
        
        case BSIM4v2_MOD_A1:
            mod->BSIM4v2a1 = value->rValue;
            mod->BSIM4v2a1Given = TRUE;
            break;
        case BSIM4v2_MOD_A2:
            mod->BSIM4v2a2 = value->rValue;
            mod->BSIM4v2a2Given = TRUE;
            break;
        case BSIM4v2_MOD_AT:
            mod->BSIM4v2at = value->rValue;
            mod->BSIM4v2atGiven = TRUE;
            break;
        case BSIM4v2_MOD_KETA:
            mod->BSIM4v2keta = value->rValue;
            mod->BSIM4v2ketaGiven = TRUE;
            break;    
        case BSIM4v2_MOD_NSUB:
            mod->BSIM4v2nsub = value->rValue;
            mod->BSIM4v2nsubGiven = TRUE;
            break;
        case BSIM4v2_MOD_NDEP:
            mod->BSIM4v2ndep = value->rValue;
            mod->BSIM4v2ndepGiven = TRUE;
	    if (mod->BSIM4v2ndep > 1.0e20)
		mod->BSIM4v2ndep *= 1.0e-6;
            break;
        case BSIM4v2_MOD_NSD:
            mod->BSIM4v2nsd = value->rValue;
            mod->BSIM4v2nsdGiven = TRUE;
            if (mod->BSIM4v2nsd > 1.0e23)
                mod->BSIM4v2nsd *= 1.0e-6;
            break;
        case BSIM4v2_MOD_NGATE:
            mod->BSIM4v2ngate = value->rValue;
            mod->BSIM4v2ngateGiven = TRUE;
	    if (mod->BSIM4v2ngate > 1.0e23)
		mod->BSIM4v2ngate *= 1.0e-6;
            break;
        case BSIM4v2_MOD_GAMMA1:
            mod->BSIM4v2gamma1 = value->rValue;
            mod->BSIM4v2gamma1Given = TRUE;
            break;
        case BSIM4v2_MOD_GAMMA2:
            mod->BSIM4v2gamma2 = value->rValue;
            mod->BSIM4v2gamma2Given = TRUE;
            break;
        case BSIM4v2_MOD_VBX:
            mod->BSIM4v2vbx = value->rValue;
            mod->BSIM4v2vbxGiven = TRUE;
            break;
        case BSIM4v2_MOD_VBM:
            mod->BSIM4v2vbm = value->rValue;
            mod->BSIM4v2vbmGiven = TRUE;
            break;
        case BSIM4v2_MOD_XT:
            mod->BSIM4v2xt = value->rValue;
            mod->BSIM4v2xtGiven = TRUE;
            break;
        case  BSIM4v2_MOD_K1:
            mod->BSIM4v2k1 = value->rValue;
            mod->BSIM4v2k1Given = TRUE;
            break;
        case  BSIM4v2_MOD_KT1:
            mod->BSIM4v2kt1 = value->rValue;
            mod->BSIM4v2kt1Given = TRUE;
            break;
        case  BSIM4v2_MOD_KT1L:
            mod->BSIM4v2kt1l = value->rValue;
            mod->BSIM4v2kt1lGiven = TRUE;
            break;
        case  BSIM4v2_MOD_KT2:
            mod->BSIM4v2kt2 = value->rValue;
            mod->BSIM4v2kt2Given = TRUE;
            break;
        case  BSIM4v2_MOD_K2:
            mod->BSIM4v2k2 = value->rValue;
            mod->BSIM4v2k2Given = TRUE;
            break;
        case  BSIM4v2_MOD_K3:
            mod->BSIM4v2k3 = value->rValue;
            mod->BSIM4v2k3Given = TRUE;
            break;
        case  BSIM4v2_MOD_K3B:
            mod->BSIM4v2k3b = value->rValue;
            mod->BSIM4v2k3bGiven = TRUE;
            break;
        case  BSIM4v2_MOD_LPE0:
            mod->BSIM4v2lpe0 = value->rValue;
            mod->BSIM4v2lpe0Given = TRUE;
            break;
        case  BSIM4v2_MOD_LPEB:
            mod->BSIM4v2lpeb = value->rValue;
            mod->BSIM4v2lpebGiven = TRUE;
            break;
        case  BSIM4v2_MOD_DVTP0:
            mod->BSIM4v2dvtp0 = value->rValue;
            mod->BSIM4v2dvtp0Given = TRUE;
            break;
        case  BSIM4v2_MOD_DVTP1:
            mod->BSIM4v2dvtp1 = value->rValue;
            mod->BSIM4v2dvtp1Given = TRUE;
            break;
        case  BSIM4v2_MOD_W0:
            mod->BSIM4v2w0 = value->rValue;
            mod->BSIM4v2w0Given = TRUE;
            break;
        case  BSIM4v2_MOD_DVT0:               
            mod->BSIM4v2dvt0 = value->rValue;
            mod->BSIM4v2dvt0Given = TRUE;
            break;
        case  BSIM4v2_MOD_DVT1:             
            mod->BSIM4v2dvt1 = value->rValue;
            mod->BSIM4v2dvt1Given = TRUE;
            break;
        case  BSIM4v2_MOD_DVT2:             
            mod->BSIM4v2dvt2 = value->rValue;
            mod->BSIM4v2dvt2Given = TRUE;
            break;
        case  BSIM4v2_MOD_DVT0W:               
            mod->BSIM4v2dvt0w = value->rValue;
            mod->BSIM4v2dvt0wGiven = TRUE;
            break;
        case  BSIM4v2_MOD_DVT1W:             
            mod->BSIM4v2dvt1w = value->rValue;
            mod->BSIM4v2dvt1wGiven = TRUE;
            break;
        case  BSIM4v2_MOD_DVT2W:             
            mod->BSIM4v2dvt2w = value->rValue;
            mod->BSIM4v2dvt2wGiven = TRUE;
            break;
        case  BSIM4v2_MOD_DROUT:             
            mod->BSIM4v2drout = value->rValue;
            mod->BSIM4v2droutGiven = TRUE;
            break;
        case  BSIM4v2_MOD_DSUB:             
            mod->BSIM4v2dsub = value->rValue;
            mod->BSIM4v2dsubGiven = TRUE;
            break;
        case BSIM4v2_MOD_VTH0:
            mod->BSIM4v2vth0 = value->rValue;
            mod->BSIM4v2vth0Given = TRUE;
            break;
        case BSIM4v2_MOD_EU:
            mod->BSIM4v2eu = value->rValue;
            mod->BSIM4v2euGiven = TRUE;
            break;
        case BSIM4v2_MOD_UA:
            mod->BSIM4v2ua = value->rValue;
            mod->BSIM4v2uaGiven = TRUE;
            break;
        case BSIM4v2_MOD_UA1:
            mod->BSIM4v2ua1 = value->rValue;
            mod->BSIM4v2ua1Given = TRUE;
            break;
        case BSIM4v2_MOD_UB:
            mod->BSIM4v2ub = value->rValue;
            mod->BSIM4v2ubGiven = TRUE;
            break;
        case BSIM4v2_MOD_UB1:
            mod->BSIM4v2ub1 = value->rValue;
            mod->BSIM4v2ub1Given = TRUE;
            break;
        case BSIM4v2_MOD_UC:
            mod->BSIM4v2uc = value->rValue;
            mod->BSIM4v2ucGiven = TRUE;
            break;
        case BSIM4v2_MOD_UC1:
            mod->BSIM4v2uc1 = value->rValue;
            mod->BSIM4v2uc1Given = TRUE;
            break;
        case  BSIM4v2_MOD_U0 :
            mod->BSIM4v2u0 = value->rValue;
            mod->BSIM4v2u0Given = TRUE;
            break;
        case  BSIM4v2_MOD_UTE :
            mod->BSIM4v2ute = value->rValue;
            mod->BSIM4v2uteGiven = TRUE;
            break;
        case BSIM4v2_MOD_VOFF:
            mod->BSIM4v2voff = value->rValue;
            mod->BSIM4v2voffGiven = TRUE;
            break;
        case BSIM4v2_MOD_VOFFL:
            mod->BSIM4v2voffl = value->rValue;
            mod->BSIM4v2vofflGiven = TRUE;
            break;
        case BSIM4v2_MOD_MINV:
            mod->BSIM4v2minv = value->rValue;
            mod->BSIM4v2minvGiven = TRUE;
            break;
        case BSIM4v2_MOD_FPROUT:
            mod->BSIM4v2fprout = value->rValue;
            mod->BSIM4v2fproutGiven = TRUE;
            break;
        case BSIM4v2_MOD_PDITS:
            mod->BSIM4v2pdits = value->rValue;
            mod->BSIM4v2pditsGiven = TRUE;
            break;
        case BSIM4v2_MOD_PDITSD:
            mod->BSIM4v2pditsd = value->rValue;
            mod->BSIM4v2pditsdGiven = TRUE;
            break;
        case BSIM4v2_MOD_PDITSL:
            mod->BSIM4v2pditsl = value->rValue;
            mod->BSIM4v2pditslGiven = TRUE;
            break;
        case  BSIM4v2_MOD_DELTA :
            mod->BSIM4v2delta = value->rValue;
            mod->BSIM4v2deltaGiven = TRUE;
            break;
        case BSIM4v2_MOD_RDSW:
            mod->BSIM4v2rdsw = value->rValue;
            mod->BSIM4v2rdswGiven = TRUE;
            break;                     
        case BSIM4v2_MOD_RDSWMIN:
            mod->BSIM4v2rdswmin = value->rValue;
            mod->BSIM4v2rdswminGiven = TRUE;
            break;
        case BSIM4v2_MOD_RDWMIN:
            mod->BSIM4v2rdwmin = value->rValue;
            mod->BSIM4v2rdwminGiven = TRUE;
            break;
        case BSIM4v2_MOD_RSWMIN:
            mod->BSIM4v2rswmin = value->rValue;
            mod->BSIM4v2rswminGiven = TRUE;
            break;
        case BSIM4v2_MOD_RDW:
            mod->BSIM4v2rdw = value->rValue;
            mod->BSIM4v2rdwGiven = TRUE;
            break;
        case BSIM4v2_MOD_RSW:
            mod->BSIM4v2rsw = value->rValue;
            mod->BSIM4v2rswGiven = TRUE;
            break;
        case BSIM4v2_MOD_PRWG:
            mod->BSIM4v2prwg = value->rValue;
            mod->BSIM4v2prwgGiven = TRUE;
            break;                     
        case BSIM4v2_MOD_PRWB:
            mod->BSIM4v2prwb = value->rValue;
            mod->BSIM4v2prwbGiven = TRUE;
            break;                     
        case BSIM4v2_MOD_PRT:
            mod->BSIM4v2prt = value->rValue;
            mod->BSIM4v2prtGiven = TRUE;
            break;                     
        case BSIM4v2_MOD_ETA0:
            mod->BSIM4v2eta0 = value->rValue;
            mod->BSIM4v2eta0Given = TRUE;
            break;                 
        case BSIM4v2_MOD_ETAB:
            mod->BSIM4v2etab = value->rValue;
            mod->BSIM4v2etabGiven = TRUE;
            break;                 
        case BSIM4v2_MOD_PCLM:
            mod->BSIM4v2pclm = value->rValue;
            mod->BSIM4v2pclmGiven = TRUE;
            break;                 
        case BSIM4v2_MOD_PDIBL1:
            mod->BSIM4v2pdibl1 = value->rValue;
            mod->BSIM4v2pdibl1Given = TRUE;
            break;                 
        case BSIM4v2_MOD_PDIBL2:
            mod->BSIM4v2pdibl2 = value->rValue;
            mod->BSIM4v2pdibl2Given = TRUE;
            break;                 
        case BSIM4v2_MOD_PDIBLB:
            mod->BSIM4v2pdiblb = value->rValue;
            mod->BSIM4v2pdiblbGiven = TRUE;
            break;                 
        case BSIM4v2_MOD_PSCBE1:
            mod->BSIM4v2pscbe1 = value->rValue;
            mod->BSIM4v2pscbe1Given = TRUE;
            break;                 
        case BSIM4v2_MOD_PSCBE2:
            mod->BSIM4v2pscbe2 = value->rValue;
            mod->BSIM4v2pscbe2Given = TRUE;
            break;                 
        case BSIM4v2_MOD_PVAG:
            mod->BSIM4v2pvag = value->rValue;
            mod->BSIM4v2pvagGiven = TRUE;
            break;                 
        case  BSIM4v2_MOD_WR :
            mod->BSIM4v2wr = value->rValue;
            mod->BSIM4v2wrGiven = TRUE;
            break;
        case  BSIM4v2_MOD_DWG :
            mod->BSIM4v2dwg = value->rValue;
            mod->BSIM4v2dwgGiven = TRUE;
            break;
        case  BSIM4v2_MOD_DWB :
            mod->BSIM4v2dwb = value->rValue;
            mod->BSIM4v2dwbGiven = TRUE;
            break;
        case  BSIM4v2_MOD_B0 :
            mod->BSIM4v2b0 = value->rValue;
            mod->BSIM4v2b0Given = TRUE;
            break;
        case  BSIM4v2_MOD_B1 :
            mod->BSIM4v2b1 = value->rValue;
            mod->BSIM4v2b1Given = TRUE;
            break;
        case  BSIM4v2_MOD_ALPHA0 :
            mod->BSIM4v2alpha0 = value->rValue;
            mod->BSIM4v2alpha0Given = TRUE;
            break;
        case  BSIM4v2_MOD_ALPHA1 :
            mod->BSIM4v2alpha1 = value->rValue;
            mod->BSIM4v2alpha1Given = TRUE;
            break;
        case  BSIM4v2_MOD_AGIDL :
            mod->BSIM4v2agidl = value->rValue;
            mod->BSIM4v2agidlGiven = TRUE;
            break;
        case  BSIM4v2_MOD_BGIDL :
            mod->BSIM4v2bgidl = value->rValue;
            mod->BSIM4v2bgidlGiven = TRUE;
            break;
        case  BSIM4v2_MOD_CGIDL :
            mod->BSIM4v2cgidl = value->rValue;
            mod->BSIM4v2cgidlGiven = TRUE;
            break;
        case  BSIM4v2_MOD_PHIN :
            mod->BSIM4v2phin = value->rValue;
            mod->BSIM4v2phinGiven = TRUE;
            break;
        case  BSIM4v2_MOD_EGIDL :
            mod->BSIM4v2egidl = value->rValue;
            mod->BSIM4v2egidlGiven = TRUE;
            break;
        case  BSIM4v2_MOD_AIGC :
            mod->BSIM4v2aigc = value->rValue;
            mod->BSIM4v2aigcGiven = TRUE;
            break;
        case  BSIM4v2_MOD_BIGC :
            mod->BSIM4v2bigc = value->rValue;
            mod->BSIM4v2bigcGiven = TRUE;
            break;
        case  BSIM4v2_MOD_CIGC :
            mod->BSIM4v2cigc = value->rValue;
            mod->BSIM4v2cigcGiven = TRUE;
            break;
        case  BSIM4v2_MOD_AIGSD :
            mod->BSIM4v2aigsd = value->rValue;
            mod->BSIM4v2aigsdGiven = TRUE;
            break;
        case  BSIM4v2_MOD_BIGSD :
            mod->BSIM4v2bigsd = value->rValue;
            mod->BSIM4v2bigsdGiven = TRUE;
            break;
        case  BSIM4v2_MOD_CIGSD :
            mod->BSIM4v2cigsd = value->rValue;
            mod->BSIM4v2cigsdGiven = TRUE;
            break;
        case  BSIM4v2_MOD_AIGBACC :
            mod->BSIM4v2aigbacc = value->rValue;
            mod->BSIM4v2aigbaccGiven = TRUE;
            break;
        case  BSIM4v2_MOD_BIGBACC :
            mod->BSIM4v2bigbacc = value->rValue;
            mod->BSIM4v2bigbaccGiven = TRUE;
            break;
        case  BSIM4v2_MOD_CIGBACC :
            mod->BSIM4v2cigbacc = value->rValue;
            mod->BSIM4v2cigbaccGiven = TRUE;
            break;
        case  BSIM4v2_MOD_AIGBINV :
            mod->BSIM4v2aigbinv = value->rValue;
            mod->BSIM4v2aigbinvGiven = TRUE;
            break;
        case  BSIM4v2_MOD_BIGBINV :
            mod->BSIM4v2bigbinv = value->rValue;
            mod->BSIM4v2bigbinvGiven = TRUE;
            break;
        case  BSIM4v2_MOD_CIGBINV :
            mod->BSIM4v2cigbinv = value->rValue;
            mod->BSIM4v2cigbinvGiven = TRUE;
            break;
        case  BSIM4v2_MOD_NIGC :
            mod->BSIM4v2nigc = value->rValue;
            mod->BSIM4v2nigcGiven = TRUE;
            break;
        case  BSIM4v2_MOD_NIGBINV :
            mod->BSIM4v2nigbinv = value->rValue;
            mod->BSIM4v2nigbinvGiven = TRUE;
            break;
        case  BSIM4v2_MOD_NIGBACC :
            mod->BSIM4v2nigbacc = value->rValue;
            mod->BSIM4v2nigbaccGiven = TRUE;
            break;
        case  BSIM4v2_MOD_NTOX :
            mod->BSIM4v2ntox = value->rValue;
            mod->BSIM4v2ntoxGiven = TRUE;
            break;
        case  BSIM4v2_MOD_EIGBINV :
            mod->BSIM4v2eigbinv = value->rValue;
            mod->BSIM4v2eigbinvGiven = TRUE;
            break;
        case  BSIM4v2_MOD_PIGCD :
            mod->BSIM4v2pigcd = value->rValue;
            mod->BSIM4v2pigcdGiven = TRUE;
            break;
        case  BSIM4v2_MOD_POXEDGE :
            mod->BSIM4v2poxedge = value->rValue;
            mod->BSIM4v2poxedgeGiven = TRUE;
            break;
        case  BSIM4v2_MOD_XRCRG1 :
            mod->BSIM4v2xrcrg1 = value->rValue;
            mod->BSIM4v2xrcrg1Given = TRUE;
            break;
        case  BSIM4v2_MOD_TNOIA :
            mod->BSIM4v2tnoia = value->rValue;
            mod->BSIM4v2tnoiaGiven = TRUE;
            break;
        case  BSIM4v2_MOD_TNOIB :
            mod->BSIM4v2tnoib = value->rValue;
            mod->BSIM4v2tnoibGiven = TRUE;
            break;
        case  BSIM4v2_MOD_NTNOI :
            mod->BSIM4v2ntnoi = value->rValue;
            mod->BSIM4v2ntnoiGiven = TRUE;
            break;
        case  BSIM4v2_MOD_XRCRG2 :
            mod->BSIM4v2xrcrg2 = value->rValue;
            mod->BSIM4v2xrcrg2Given = TRUE;
            break;
        case  BSIM4v2_MOD_BETA0 :
            mod->BSIM4v2beta0 = value->rValue;
            mod->BSIM4v2beta0Given = TRUE;
            break;
        case  BSIM4v2_MOD_IJTHDFWD :
            mod->BSIM4v2ijthdfwd = value->rValue;
            mod->BSIM4v2ijthdfwdGiven = TRUE;
            break;
        case  BSIM4v2_MOD_IJTHSFWD :
            mod->BSIM4v2ijthsfwd = value->rValue;
            mod->BSIM4v2ijthsfwdGiven = TRUE;
            break;
        case  BSIM4v2_MOD_IJTHDREV :
            mod->BSIM4v2ijthdrev = value->rValue;
            mod->BSIM4v2ijthdrevGiven = TRUE;
            break;
        case  BSIM4v2_MOD_IJTHSREV :
            mod->BSIM4v2ijthsrev = value->rValue;
            mod->BSIM4v2ijthsrevGiven = TRUE;
            break;
        case  BSIM4v2_MOD_XJBVD :
            mod->BSIM4v2xjbvd = value->rValue;
            mod->BSIM4v2xjbvdGiven = TRUE;
            break;
        case  BSIM4v2_MOD_XJBVS :
            mod->BSIM4v2xjbvs = value->rValue;
            mod->BSIM4v2xjbvsGiven = TRUE;
            break;
        case  BSIM4v2_MOD_BVD :
            mod->BSIM4v2bvd = value->rValue;
            mod->BSIM4v2bvdGiven = TRUE;
            break;
        case  BSIM4v2_MOD_BVS :
            mod->BSIM4v2bvs = value->rValue;
            mod->BSIM4v2bvsGiven = TRUE;
            break;
        case  BSIM4v2_MOD_VFB :
            mod->BSIM4v2vfb = value->rValue;
            mod->BSIM4v2vfbGiven = TRUE;
            break;

        case  BSIM4v2_MOD_GBMIN :
            mod->BSIM4v2gbmin = value->rValue;
            mod->BSIM4v2gbminGiven = TRUE;
            break;
        case  BSIM4v2_MOD_RBDB :
            mod->BSIM4v2rbdb = value->rValue;
            mod->BSIM4v2rbdbGiven = TRUE;
            break;
        case  BSIM4v2_MOD_RBPB :
            mod->BSIM4v2rbpb = value->rValue;
            mod->BSIM4v2rbpbGiven = TRUE;
            break;
        case  BSIM4v2_MOD_RBSB :
            mod->BSIM4v2rbsb = value->rValue;
            mod->BSIM4v2rbsbGiven = TRUE;
            break;
        case  BSIM4v2_MOD_RBPS :
            mod->BSIM4v2rbps = value->rValue;
            mod->BSIM4v2rbpsGiven = TRUE;
            break;
        case  BSIM4v2_MOD_RBPD :
            mod->BSIM4v2rbpd = value->rValue;
            mod->BSIM4v2rbpdGiven = TRUE;
            break;

        case  BSIM4v2_MOD_CGSL :
            mod->BSIM4v2cgsl = value->rValue;
            mod->BSIM4v2cgslGiven = TRUE;
            break;
        case  BSIM4v2_MOD_CGDL :
            mod->BSIM4v2cgdl = value->rValue;
            mod->BSIM4v2cgdlGiven = TRUE;
            break;
        case  BSIM4v2_MOD_CKAPPAS :
            mod->BSIM4v2ckappas = value->rValue;
            mod->BSIM4v2ckappasGiven = TRUE;
            break;
        case  BSIM4v2_MOD_CKAPPAD :
            mod->BSIM4v2ckappad = value->rValue;
            mod->BSIM4v2ckappadGiven = TRUE;
            break;
        case  BSIM4v2_MOD_CF :
            mod->BSIM4v2cf = value->rValue;
            mod->BSIM4v2cfGiven = TRUE;
            break;
        case  BSIM4v2_MOD_CLC :
            mod->BSIM4v2clc = value->rValue;
            mod->BSIM4v2clcGiven = TRUE;
            break;
        case  BSIM4v2_MOD_CLE :
            mod->BSIM4v2cle = value->rValue;
            mod->BSIM4v2cleGiven = TRUE;
            break;
        case  BSIM4v2_MOD_DWC :
            mod->BSIM4v2dwc = value->rValue;
            mod->BSIM4v2dwcGiven = TRUE;
            break;
        case  BSIM4v2_MOD_DLC :
            mod->BSIM4v2dlc = value->rValue;
            mod->BSIM4v2dlcGiven = TRUE;
            break;
        case  BSIM4v2_MOD_XW :
            mod->BSIM4v2xw = value->rValue;
            mod->BSIM4v2xwGiven = TRUE;
            break;
        case  BSIM4v2_MOD_XL :
            mod->BSIM4v2xl = value->rValue;
            mod->BSIM4v2xlGiven = TRUE;
            break;
        case  BSIM4v2_MOD_DLCIG :
            mod->BSIM4v2dlcig = value->rValue;
            mod->BSIM4v2dlcigGiven = TRUE;
            break;
        case  BSIM4v2_MOD_DWJ :
            mod->BSIM4v2dwj = value->rValue;
            mod->BSIM4v2dwjGiven = TRUE;
            break;
        case  BSIM4v2_MOD_VFBCV :
            mod->BSIM4v2vfbcv = value->rValue;
            mod->BSIM4v2vfbcvGiven = TRUE;
            break;
        case  BSIM4v2_MOD_ACDE :
            mod->BSIM4v2acde = value->rValue;
            mod->BSIM4v2acdeGiven = TRUE;
            break;
        case  BSIM4v2_MOD_MOIN :
            mod->BSIM4v2moin = value->rValue;
            mod->BSIM4v2moinGiven = TRUE;
            break;
        case  BSIM4v2_MOD_NOFF :
            mod->BSIM4v2noff = value->rValue;
            mod->BSIM4v2noffGiven = TRUE;
            break;
        case  BSIM4v2_MOD_VOFFCV :
            mod->BSIM4v2voffcv = value->rValue;
            mod->BSIM4v2voffcvGiven = TRUE;
            break;
        case  BSIM4v2_MOD_DMCG :
            mod->BSIM4v2dmcg = value->rValue;
            mod->BSIM4v2dmcgGiven = TRUE;
            break;
        case  BSIM4v2_MOD_DMCI :
            mod->BSIM4v2dmci = value->rValue;
            mod->BSIM4v2dmciGiven = TRUE;
            break;
        case  BSIM4v2_MOD_DMDG :
            mod->BSIM4v2dmdg = value->rValue;
            mod->BSIM4v2dmdgGiven = TRUE;
            break;
        case  BSIM4v2_MOD_DMCGT :
            mod->BSIM4v2dmcgt = value->rValue;
            mod->BSIM4v2dmcgtGiven = TRUE;
            break;
        case  BSIM4v2_MOD_XGW :
            mod->BSIM4v2xgw = value->rValue;
            mod->BSIM4v2xgwGiven = TRUE;
            break;
        case  BSIM4v2_MOD_XGL :
            mod->BSIM4v2xgl = value->rValue;
            mod->BSIM4v2xglGiven = TRUE;
            break;
        case  BSIM4v2_MOD_RSHG :
            mod->BSIM4v2rshg = value->rValue;
            mod->BSIM4v2rshgGiven = TRUE;
            break;
        case  BSIM4v2_MOD_NGCON :
            mod->BSIM4v2ngcon = value->rValue;
            mod->BSIM4v2ngconGiven = TRUE;
            break;
        case  BSIM4v2_MOD_TCJ :
            mod->BSIM4v2tcj = value->rValue;
            mod->BSIM4v2tcjGiven = TRUE;
            break;
        case  BSIM4v2_MOD_TPB :
            mod->BSIM4v2tpb = value->rValue;
            mod->BSIM4v2tpbGiven = TRUE;
            break;
        case  BSIM4v2_MOD_TCJSW :
            mod->BSIM4v2tcjsw = value->rValue;
            mod->BSIM4v2tcjswGiven = TRUE;
            break;
        case  BSIM4v2_MOD_TPBSW :
            mod->BSIM4v2tpbsw = value->rValue;
            mod->BSIM4v2tpbswGiven = TRUE;
            break;
        case  BSIM4v2_MOD_TCJSWG :
            mod->BSIM4v2tcjswg = value->rValue;
            mod->BSIM4v2tcjswgGiven = TRUE;
            break;
        case  BSIM4v2_MOD_TPBSWG :
            mod->BSIM4v2tpbswg = value->rValue;
            mod->BSIM4v2tpbswgGiven = TRUE;
            break;

	/* Length dependence */
        case  BSIM4v2_MOD_LCDSC :
            mod->BSIM4v2lcdsc = value->rValue;
            mod->BSIM4v2lcdscGiven = TRUE;
            break;


        case  BSIM4v2_MOD_LCDSCB :
            mod->BSIM4v2lcdscb = value->rValue;
            mod->BSIM4v2lcdscbGiven = TRUE;
            break;
        case  BSIM4v2_MOD_LCDSCD :
            mod->BSIM4v2lcdscd = value->rValue;
            mod->BSIM4v2lcdscdGiven = TRUE;
            break;
        case  BSIM4v2_MOD_LCIT :
            mod->BSIM4v2lcit = value->rValue;
            mod->BSIM4v2lcitGiven = TRUE;
            break;
        case  BSIM4v2_MOD_LNFACTOR :
            mod->BSIM4v2lnfactor = value->rValue;
            mod->BSIM4v2lnfactorGiven = TRUE;
            break;
        case BSIM4v2_MOD_LXJ:
            mod->BSIM4v2lxj = value->rValue;
            mod->BSIM4v2lxjGiven = TRUE;
            break;
        case BSIM4v2_MOD_LVSAT:
            mod->BSIM4v2lvsat = value->rValue;
            mod->BSIM4v2lvsatGiven = TRUE;
            break;
        
        
        case BSIM4v2_MOD_LA0:
            mod->BSIM4v2la0 = value->rValue;
            mod->BSIM4v2la0Given = TRUE;
            break;
        case BSIM4v2_MOD_LAGS:
            mod->BSIM4v2lags = value->rValue;
            mod->BSIM4v2lagsGiven = TRUE;
            break;
        case BSIM4v2_MOD_LA1:
            mod->BSIM4v2la1 = value->rValue;
            mod->BSIM4v2la1Given = TRUE;
            break;
        case BSIM4v2_MOD_LA2:
            mod->BSIM4v2la2 = value->rValue;
            mod->BSIM4v2la2Given = TRUE;
            break;
        case BSIM4v2_MOD_LAT:
            mod->BSIM4v2lat = value->rValue;
            mod->BSIM4v2latGiven = TRUE;
            break;
        case BSIM4v2_MOD_LKETA:
            mod->BSIM4v2lketa = value->rValue;
            mod->BSIM4v2lketaGiven = TRUE;
            break;    
        case BSIM4v2_MOD_LNSUB:
            mod->BSIM4v2lnsub = value->rValue;
            mod->BSIM4v2lnsubGiven = TRUE;
            break;
        case BSIM4v2_MOD_LNDEP:
            mod->BSIM4v2lndep = value->rValue;
            mod->BSIM4v2lndepGiven = TRUE;
	    if (mod->BSIM4v2lndep > 1.0e20)
		mod->BSIM4v2lndep *= 1.0e-6;
            break;
        case BSIM4v2_MOD_LNSD:
            mod->BSIM4v2lnsd = value->rValue;
            mod->BSIM4v2lnsdGiven = TRUE;
            if (mod->BSIM4v2lnsd > 1.0e23)
                mod->BSIM4v2lnsd *= 1.0e-6;
            break;
        case BSIM4v2_MOD_LNGATE:
            mod->BSIM4v2lngate = value->rValue;
            mod->BSIM4v2lngateGiven = TRUE;
	    if (mod->BSIM4v2lngate > 1.0e23)
		mod->BSIM4v2lngate *= 1.0e-6;
            break;
        case BSIM4v2_MOD_LGAMMA1:
            mod->BSIM4v2lgamma1 = value->rValue;
            mod->BSIM4v2lgamma1Given = TRUE;
            break;
        case BSIM4v2_MOD_LGAMMA2:
            mod->BSIM4v2lgamma2 = value->rValue;
            mod->BSIM4v2lgamma2Given = TRUE;
            break;
        case BSIM4v2_MOD_LVBX:
            mod->BSIM4v2lvbx = value->rValue;
            mod->BSIM4v2lvbxGiven = TRUE;
            break;
        case BSIM4v2_MOD_LVBM:
            mod->BSIM4v2lvbm = value->rValue;
            mod->BSIM4v2lvbmGiven = TRUE;
            break;
        case BSIM4v2_MOD_LXT:
            mod->BSIM4v2lxt = value->rValue;
            mod->BSIM4v2lxtGiven = TRUE;
            break;
        case  BSIM4v2_MOD_LK1:
            mod->BSIM4v2lk1 = value->rValue;
            mod->BSIM4v2lk1Given = TRUE;
            break;
        case  BSIM4v2_MOD_LKT1:
            mod->BSIM4v2lkt1 = value->rValue;
            mod->BSIM4v2lkt1Given = TRUE;
            break;
        case  BSIM4v2_MOD_LKT1L:
            mod->BSIM4v2lkt1l = value->rValue;
            mod->BSIM4v2lkt1lGiven = TRUE;
            break;
        case  BSIM4v2_MOD_LKT2:
            mod->BSIM4v2lkt2 = value->rValue;
            mod->BSIM4v2lkt2Given = TRUE;
            break;
        case  BSIM4v2_MOD_LK2:
            mod->BSIM4v2lk2 = value->rValue;
            mod->BSIM4v2lk2Given = TRUE;
            break;
        case  BSIM4v2_MOD_LK3:
            mod->BSIM4v2lk3 = value->rValue;
            mod->BSIM4v2lk3Given = TRUE;
            break;
        case  BSIM4v2_MOD_LK3B:
            mod->BSIM4v2lk3b = value->rValue;
            mod->BSIM4v2lk3bGiven = TRUE;
            break;
        case  BSIM4v2_MOD_LLPE0:
            mod->BSIM4v2llpe0 = value->rValue;
            mod->BSIM4v2llpe0Given = TRUE;
            break;
        case  BSIM4v2_MOD_LLPEB:
            mod->BSIM4v2llpeb = value->rValue;
            mod->BSIM4v2llpebGiven = TRUE;
            break;
        case  BSIM4v2_MOD_LDVTP0:
            mod->BSIM4v2ldvtp0 = value->rValue;
            mod->BSIM4v2ldvtp0Given = TRUE;
            break;
        case  BSIM4v2_MOD_LDVTP1:
            mod->BSIM4v2ldvtp1 = value->rValue;
            mod->BSIM4v2ldvtp1Given = TRUE;
            break;
        case  BSIM4v2_MOD_LW0:
            mod->BSIM4v2lw0 = value->rValue;
            mod->BSIM4v2lw0Given = TRUE;
            break;
        case  BSIM4v2_MOD_LDVT0:               
            mod->BSIM4v2ldvt0 = value->rValue;
            mod->BSIM4v2ldvt0Given = TRUE;
            break;
        case  BSIM4v2_MOD_LDVT1:             
            mod->BSIM4v2ldvt1 = value->rValue;
            mod->BSIM4v2ldvt1Given = TRUE;
            break;
        case  BSIM4v2_MOD_LDVT2:             
            mod->BSIM4v2ldvt2 = value->rValue;
            mod->BSIM4v2ldvt2Given = TRUE;
            break;
        case  BSIM4v2_MOD_LDVT0W:               
            mod->BSIM4v2ldvt0w = value->rValue;
            mod->BSIM4v2ldvt0wGiven = TRUE;
            break;
        case  BSIM4v2_MOD_LDVT1W:             
            mod->BSIM4v2ldvt1w = value->rValue;
            mod->BSIM4v2ldvt1wGiven = TRUE;
            break;
        case  BSIM4v2_MOD_LDVT2W:             
            mod->BSIM4v2ldvt2w = value->rValue;
            mod->BSIM4v2ldvt2wGiven = TRUE;
            break;
        case  BSIM4v2_MOD_LDROUT:             
            mod->BSIM4v2ldrout = value->rValue;
            mod->BSIM4v2ldroutGiven = TRUE;
            break;
        case  BSIM4v2_MOD_LDSUB:             
            mod->BSIM4v2ldsub = value->rValue;
            mod->BSIM4v2ldsubGiven = TRUE;
            break;
        case BSIM4v2_MOD_LVTH0:
            mod->BSIM4v2lvth0 = value->rValue;
            mod->BSIM4v2lvth0Given = TRUE;
            break;
        case BSIM4v2_MOD_LUA:
            mod->BSIM4v2lua = value->rValue;
            mod->BSIM4v2luaGiven = TRUE;
            break;
        case BSIM4v2_MOD_LUA1:
            mod->BSIM4v2lua1 = value->rValue;
            mod->BSIM4v2lua1Given = TRUE;
            break;
        case BSIM4v2_MOD_LUB:
            mod->BSIM4v2lub = value->rValue;
            mod->BSIM4v2lubGiven = TRUE;
            break;
        case BSIM4v2_MOD_LUB1:
            mod->BSIM4v2lub1 = value->rValue;
            mod->BSIM4v2lub1Given = TRUE;
            break;
        case BSIM4v2_MOD_LUC:
            mod->BSIM4v2luc = value->rValue;
            mod->BSIM4v2lucGiven = TRUE;
            break;
        case BSIM4v2_MOD_LUC1:
            mod->BSIM4v2luc1 = value->rValue;
            mod->BSIM4v2luc1Given = TRUE;
            break;
        case  BSIM4v2_MOD_LU0 :
            mod->BSIM4v2lu0 = value->rValue;
            mod->BSIM4v2lu0Given = TRUE;
            break;
        case  BSIM4v2_MOD_LUTE :
            mod->BSIM4v2lute = value->rValue;
            mod->BSIM4v2luteGiven = TRUE;
            break;
        case BSIM4v2_MOD_LVOFF:
            mod->BSIM4v2lvoff = value->rValue;
            mod->BSIM4v2lvoffGiven = TRUE;
            break;
        case BSIM4v2_MOD_LMINV:
            mod->BSIM4v2lminv = value->rValue;
            mod->BSIM4v2lminvGiven = TRUE;
            break;
        case BSIM4v2_MOD_LFPROUT:
            mod->BSIM4v2lfprout = value->rValue;
            mod->BSIM4v2lfproutGiven = TRUE;
            break;
        case BSIM4v2_MOD_LPDITS:
            mod->BSIM4v2lpdits = value->rValue;
            mod->BSIM4v2lpditsGiven = TRUE;
            break;
        case BSIM4v2_MOD_LPDITSD:
            mod->BSIM4v2lpditsd = value->rValue;
            mod->BSIM4v2lpditsdGiven = TRUE;
            break;
        case  BSIM4v2_MOD_LDELTA :
            mod->BSIM4v2ldelta = value->rValue;
            mod->BSIM4v2ldeltaGiven = TRUE;
            break;
        case BSIM4v2_MOD_LRDSW:
            mod->BSIM4v2lrdsw = value->rValue;
            mod->BSIM4v2lrdswGiven = TRUE;
            break;                     
        case BSIM4v2_MOD_LRDW:
            mod->BSIM4v2lrdw = value->rValue;
            mod->BSIM4v2lrdwGiven = TRUE;
            break;
        case BSIM4v2_MOD_LRSW:
            mod->BSIM4v2lrsw = value->rValue;
            mod->BSIM4v2lrswGiven = TRUE;
            break;
        case BSIM4v2_MOD_LPRWB:
            mod->BSIM4v2lprwb = value->rValue;
            mod->BSIM4v2lprwbGiven = TRUE;
            break;                     
        case BSIM4v2_MOD_LPRWG:
            mod->BSIM4v2lprwg = value->rValue;
            mod->BSIM4v2lprwgGiven = TRUE;
            break;                     
        case BSIM4v2_MOD_LPRT:
            mod->BSIM4v2lprt = value->rValue;
            mod->BSIM4v2lprtGiven = TRUE;
            break;                     
        case BSIM4v2_MOD_LETA0:
            mod->BSIM4v2leta0 = value->rValue;
            mod->BSIM4v2leta0Given = TRUE;
            break;                 
        case BSIM4v2_MOD_LETAB:
            mod->BSIM4v2letab = value->rValue;
            mod->BSIM4v2letabGiven = TRUE;
            break;                 
        case BSIM4v2_MOD_LPCLM:
            mod->BSIM4v2lpclm = value->rValue;
            mod->BSIM4v2lpclmGiven = TRUE;
            break;                 
        case BSIM4v2_MOD_LPDIBL1:
            mod->BSIM4v2lpdibl1 = value->rValue;
            mod->BSIM4v2lpdibl1Given = TRUE;
            break;                 
        case BSIM4v2_MOD_LPDIBL2:
            mod->BSIM4v2lpdibl2 = value->rValue;
            mod->BSIM4v2lpdibl2Given = TRUE;
            break;                 
        case BSIM4v2_MOD_LPDIBLB:
            mod->BSIM4v2lpdiblb = value->rValue;
            mod->BSIM4v2lpdiblbGiven = TRUE;
            break;                 
        case BSIM4v2_MOD_LPSCBE1:
            mod->BSIM4v2lpscbe1 = value->rValue;
            mod->BSIM4v2lpscbe1Given = TRUE;
            break;                 
        case BSIM4v2_MOD_LPSCBE2:
            mod->BSIM4v2lpscbe2 = value->rValue;
            mod->BSIM4v2lpscbe2Given = TRUE;
            break;                 
        case BSIM4v2_MOD_LPVAG:
            mod->BSIM4v2lpvag = value->rValue;
            mod->BSIM4v2lpvagGiven = TRUE;
            break;                 
        case  BSIM4v2_MOD_LWR :
            mod->BSIM4v2lwr = value->rValue;
            mod->BSIM4v2lwrGiven = TRUE;
            break;
        case  BSIM4v2_MOD_LDWG :
            mod->BSIM4v2ldwg = value->rValue;
            mod->BSIM4v2ldwgGiven = TRUE;
            break;
        case  BSIM4v2_MOD_LDWB :
            mod->BSIM4v2ldwb = value->rValue;
            mod->BSIM4v2ldwbGiven = TRUE;
            break;
        case  BSIM4v2_MOD_LB0 :
            mod->BSIM4v2lb0 = value->rValue;
            mod->BSIM4v2lb0Given = TRUE;
            break;
        case  BSIM4v2_MOD_LB1 :
            mod->BSIM4v2lb1 = value->rValue;
            mod->BSIM4v2lb1Given = TRUE;
            break;
        case  BSIM4v2_MOD_LALPHA0 :
            mod->BSIM4v2lalpha0 = value->rValue;
            mod->BSIM4v2lalpha0Given = TRUE;
            break;
        case  BSIM4v2_MOD_LALPHA1 :
            mod->BSIM4v2lalpha1 = value->rValue;
            mod->BSIM4v2lalpha1Given = TRUE;
            break;
        case  BSIM4v2_MOD_LBETA0 :
            mod->BSIM4v2lbeta0 = value->rValue;
            mod->BSIM4v2lbeta0Given = TRUE;
            break;
        case  BSIM4v2_MOD_LAGIDL :
            mod->BSIM4v2lagidl = value->rValue;
            mod->BSIM4v2lagidlGiven = TRUE;
            break;
        case  BSIM4v2_MOD_LBGIDL :
            mod->BSIM4v2lbgidl = value->rValue;
            mod->BSIM4v2lbgidlGiven = TRUE;
            break;
        case  BSIM4v2_MOD_LCGIDL :
            mod->BSIM4v2lcgidl = value->rValue;
            mod->BSIM4v2lcgidlGiven = TRUE;
            break;
        case  BSIM4v2_MOD_LPHIN :
            mod->BSIM4v2lphin = value->rValue;
            mod->BSIM4v2lphinGiven = TRUE;
            break;
        case  BSIM4v2_MOD_LEGIDL :
            mod->BSIM4v2legidl = value->rValue;
            mod->BSIM4v2legidlGiven = TRUE;
            break;
        case  BSIM4v2_MOD_LAIGC :
            mod->BSIM4v2laigc = value->rValue;
            mod->BSIM4v2laigcGiven = TRUE;
            break;
        case  BSIM4v2_MOD_LBIGC :
            mod->BSIM4v2lbigc = value->rValue;
            mod->BSIM4v2lbigcGiven = TRUE;
            break;
        case  BSIM4v2_MOD_LCIGC :
            mod->BSIM4v2lcigc = value->rValue;
            mod->BSIM4v2lcigcGiven = TRUE;
            break;
        case  BSIM4v2_MOD_LAIGSD :
            mod->BSIM4v2laigsd = value->rValue;
            mod->BSIM4v2laigsdGiven = TRUE;
            break;
        case  BSIM4v2_MOD_LBIGSD :
            mod->BSIM4v2lbigsd = value->rValue;
            mod->BSIM4v2lbigsdGiven = TRUE;
            break;
        case  BSIM4v2_MOD_LCIGSD :
            mod->BSIM4v2lcigsd = value->rValue;
            mod->BSIM4v2lcigsdGiven = TRUE;
            break;
        case  BSIM4v2_MOD_LAIGBACC :
            mod->BSIM4v2laigbacc = value->rValue;
            mod->BSIM4v2laigbaccGiven = TRUE;
            break;
        case  BSIM4v2_MOD_LBIGBACC :
            mod->BSIM4v2lbigbacc = value->rValue;
            mod->BSIM4v2lbigbaccGiven = TRUE;
            break;
        case  BSIM4v2_MOD_LCIGBACC :
            mod->BSIM4v2lcigbacc = value->rValue;
            mod->BSIM4v2lcigbaccGiven = TRUE;
            break;
        case  BSIM4v2_MOD_LAIGBINV :
            mod->BSIM4v2laigbinv = value->rValue;
            mod->BSIM4v2laigbinvGiven = TRUE;
            break;
        case  BSIM4v2_MOD_LBIGBINV :
            mod->BSIM4v2lbigbinv = value->rValue;
            mod->BSIM4v2lbigbinvGiven = TRUE;
            break;
        case  BSIM4v2_MOD_LCIGBINV :
            mod->BSIM4v2lcigbinv = value->rValue;
            mod->BSIM4v2lcigbinvGiven = TRUE;
            break;
        case  BSIM4v2_MOD_LNIGC :
            mod->BSIM4v2lnigc = value->rValue;
            mod->BSIM4v2lnigcGiven = TRUE;
            break;
        case  BSIM4v2_MOD_LNIGBINV :
            mod->BSIM4v2lnigbinv = value->rValue;
            mod->BSIM4v2lnigbinvGiven = TRUE;
            break;
        case  BSIM4v2_MOD_LNIGBACC :
            mod->BSIM4v2lnigbacc = value->rValue;
            mod->BSIM4v2lnigbaccGiven = TRUE;
            break;
        case  BSIM4v2_MOD_LNTOX :
            mod->BSIM4v2lntox = value->rValue;
            mod->BSIM4v2lntoxGiven = TRUE;
            break;
        case  BSIM4v2_MOD_LEIGBINV :
            mod->BSIM4v2leigbinv = value->rValue;
            mod->BSIM4v2leigbinvGiven = TRUE;
            break;
        case  BSIM4v2_MOD_LPIGCD :
            mod->BSIM4v2lpigcd = value->rValue;
            mod->BSIM4v2lpigcdGiven = TRUE;
            break;
        case  BSIM4v2_MOD_LPOXEDGE :
            mod->BSIM4v2lpoxedge = value->rValue;
            mod->BSIM4v2lpoxedgeGiven = TRUE;
            break;
        case  BSIM4v2_MOD_LXRCRG1 :
            mod->BSIM4v2lxrcrg1 = value->rValue;
            mod->BSIM4v2lxrcrg1Given = TRUE;
            break;
        case  BSIM4v2_MOD_LXRCRG2 :
            mod->BSIM4v2lxrcrg2 = value->rValue;
            mod->BSIM4v2lxrcrg2Given = TRUE;
            break;
        case  BSIM4v2_MOD_LEU :
            mod->BSIM4v2leu = value->rValue;
            mod->BSIM4v2leuGiven = TRUE;
            break;
        case  BSIM4v2_MOD_LVFB :
            mod->BSIM4v2lvfb = value->rValue;
            mod->BSIM4v2lvfbGiven = TRUE;
            break;

        case  BSIM4v2_MOD_LCGSL :
            mod->BSIM4v2lcgsl = value->rValue;
            mod->BSIM4v2lcgslGiven = TRUE;
            break;
        case  BSIM4v2_MOD_LCGDL :
            mod->BSIM4v2lcgdl = value->rValue;
            mod->BSIM4v2lcgdlGiven = TRUE;
            break;
        case  BSIM4v2_MOD_LCKAPPAS :
            mod->BSIM4v2lckappas = value->rValue;
            mod->BSIM4v2lckappasGiven = TRUE;
            break;
        case  BSIM4v2_MOD_LCKAPPAD :
            mod->BSIM4v2lckappad = value->rValue;
            mod->BSIM4v2lckappadGiven = TRUE;
            break;
        case  BSIM4v2_MOD_LCF :
            mod->BSIM4v2lcf = value->rValue;
            mod->BSIM4v2lcfGiven = TRUE;
            break;
        case  BSIM4v2_MOD_LCLC :
            mod->BSIM4v2lclc = value->rValue;
            mod->BSIM4v2lclcGiven = TRUE;
            break;
        case  BSIM4v2_MOD_LCLE :
            mod->BSIM4v2lcle = value->rValue;
            mod->BSIM4v2lcleGiven = TRUE;
            break;
        case  BSIM4v2_MOD_LVFBCV :
            mod->BSIM4v2lvfbcv = value->rValue;
            mod->BSIM4v2lvfbcvGiven = TRUE;
            break;
        case  BSIM4v2_MOD_LACDE :
            mod->BSIM4v2lacde = value->rValue;
            mod->BSIM4v2lacdeGiven = TRUE;
            break;
        case  BSIM4v2_MOD_LMOIN :
            mod->BSIM4v2lmoin = value->rValue;
            mod->BSIM4v2lmoinGiven = TRUE;
            break;
        case  BSIM4v2_MOD_LNOFF :
            mod->BSIM4v2lnoff = value->rValue;
            mod->BSIM4v2lnoffGiven = TRUE;
            break;
        case  BSIM4v2_MOD_LVOFFCV :
            mod->BSIM4v2lvoffcv = value->rValue;
            mod->BSIM4v2lvoffcvGiven = TRUE;
            break;

	/* Width dependence */
        case  BSIM4v2_MOD_WCDSC :
            mod->BSIM4v2wcdsc = value->rValue;
            mod->BSIM4v2wcdscGiven = TRUE;
            break;
       
       
         case  BSIM4v2_MOD_WCDSCB :
            mod->BSIM4v2wcdscb = value->rValue;
            mod->BSIM4v2wcdscbGiven = TRUE;
            break;
         case  BSIM4v2_MOD_WCDSCD :
            mod->BSIM4v2wcdscd = value->rValue;
            mod->BSIM4v2wcdscdGiven = TRUE;
            break;
        case  BSIM4v2_MOD_WCIT :
            mod->BSIM4v2wcit = value->rValue;
            mod->BSIM4v2wcitGiven = TRUE;
            break;
        case  BSIM4v2_MOD_WNFACTOR :
            mod->BSIM4v2wnfactor = value->rValue;
            mod->BSIM4v2wnfactorGiven = TRUE;
            break;
        case BSIM4v2_MOD_WXJ:
            mod->BSIM4v2wxj = value->rValue;
            mod->BSIM4v2wxjGiven = TRUE;
            break;
        case BSIM4v2_MOD_WVSAT:
            mod->BSIM4v2wvsat = value->rValue;
            mod->BSIM4v2wvsatGiven = TRUE;
            break;


        case BSIM4v2_MOD_WA0:
            mod->BSIM4v2wa0 = value->rValue;
            mod->BSIM4v2wa0Given = TRUE;
            break;
        case BSIM4v2_MOD_WAGS:
            mod->BSIM4v2wags = value->rValue;
            mod->BSIM4v2wagsGiven = TRUE;
            break;
        case BSIM4v2_MOD_WA1:
            mod->BSIM4v2wa1 = value->rValue;
            mod->BSIM4v2wa1Given = TRUE;
            break;
        case BSIM4v2_MOD_WA2:
            mod->BSIM4v2wa2 = value->rValue;
            mod->BSIM4v2wa2Given = TRUE;
            break;
        case BSIM4v2_MOD_WAT:
            mod->BSIM4v2wat = value->rValue;
            mod->BSIM4v2watGiven = TRUE;
            break;
        case BSIM4v2_MOD_WKETA:
            mod->BSIM4v2wketa = value->rValue;
            mod->BSIM4v2wketaGiven = TRUE;
            break;    
        case BSIM4v2_MOD_WNSUB:
            mod->BSIM4v2wnsub = value->rValue;
            mod->BSIM4v2wnsubGiven = TRUE;
            break;
        case BSIM4v2_MOD_WNDEP:
            mod->BSIM4v2wndep = value->rValue;
            mod->BSIM4v2wndepGiven = TRUE;
	    if (mod->BSIM4v2wndep > 1.0e20)
		mod->BSIM4v2wndep *= 1.0e-6;
            break;
        case BSIM4v2_MOD_WNSD:
            mod->BSIM4v2wnsd = value->rValue;
            mod->BSIM4v2wnsdGiven = TRUE;
            if (mod->BSIM4v2wnsd > 1.0e23)
                mod->BSIM4v2wnsd *= 1.0e-6;
            break;
        case BSIM4v2_MOD_WNGATE:
            mod->BSIM4v2wngate = value->rValue;
            mod->BSIM4v2wngateGiven = TRUE;
	    if (mod->BSIM4v2wngate > 1.0e23)
		mod->BSIM4v2wngate *= 1.0e-6;
            break;
        case BSIM4v2_MOD_WGAMMA1:
            mod->BSIM4v2wgamma1 = value->rValue;
            mod->BSIM4v2wgamma1Given = TRUE;
            break;
        case BSIM4v2_MOD_WGAMMA2:
            mod->BSIM4v2wgamma2 = value->rValue;
            mod->BSIM4v2wgamma2Given = TRUE;
            break;
        case BSIM4v2_MOD_WVBX:
            mod->BSIM4v2wvbx = value->rValue;
            mod->BSIM4v2wvbxGiven = TRUE;
            break;
        case BSIM4v2_MOD_WVBM:
            mod->BSIM4v2wvbm = value->rValue;
            mod->BSIM4v2wvbmGiven = TRUE;
            break;
        case BSIM4v2_MOD_WXT:
            mod->BSIM4v2wxt = value->rValue;
            mod->BSIM4v2wxtGiven = TRUE;
            break;
        case  BSIM4v2_MOD_WK1:
            mod->BSIM4v2wk1 = value->rValue;
            mod->BSIM4v2wk1Given = TRUE;
            break;
        case  BSIM4v2_MOD_WKT1:
            mod->BSIM4v2wkt1 = value->rValue;
            mod->BSIM4v2wkt1Given = TRUE;
            break;
        case  BSIM4v2_MOD_WKT1L:
            mod->BSIM4v2wkt1l = value->rValue;
            mod->BSIM4v2wkt1lGiven = TRUE;
            break;
        case  BSIM4v2_MOD_WKT2:
            mod->BSIM4v2wkt2 = value->rValue;
            mod->BSIM4v2wkt2Given = TRUE;
            break;
        case  BSIM4v2_MOD_WK2:
            mod->BSIM4v2wk2 = value->rValue;
            mod->BSIM4v2wk2Given = TRUE;
            break;
        case  BSIM4v2_MOD_WK3:
            mod->BSIM4v2wk3 = value->rValue;
            mod->BSIM4v2wk3Given = TRUE;
            break;
        case  BSIM4v2_MOD_WK3B:
            mod->BSIM4v2wk3b = value->rValue;
            mod->BSIM4v2wk3bGiven = TRUE;
            break;
        case  BSIM4v2_MOD_WLPE0:
            mod->BSIM4v2wlpe0 = value->rValue;
            mod->BSIM4v2wlpe0Given = TRUE;
            break;
        case  BSIM4v2_MOD_WLPEB:
            mod->BSIM4v2wlpeb = value->rValue;
            mod->BSIM4v2wlpebGiven = TRUE;
            break;
        case  BSIM4v2_MOD_WDVTP0:
            mod->BSIM4v2wdvtp0 = value->rValue;
            mod->BSIM4v2wdvtp0Given = TRUE;
            break;
        case  BSIM4v2_MOD_WDVTP1:
            mod->BSIM4v2wdvtp1 = value->rValue;
            mod->BSIM4v2wdvtp1Given = TRUE;
            break;
        case  BSIM4v2_MOD_WW0:
            mod->BSIM4v2ww0 = value->rValue;
            mod->BSIM4v2ww0Given = TRUE;
            break;
        case  BSIM4v2_MOD_WDVT0:               
            mod->BSIM4v2wdvt0 = value->rValue;
            mod->BSIM4v2wdvt0Given = TRUE;
            break;
        case  BSIM4v2_MOD_WDVT1:             
            mod->BSIM4v2wdvt1 = value->rValue;
            mod->BSIM4v2wdvt1Given = TRUE;
            break;
        case  BSIM4v2_MOD_WDVT2:             
            mod->BSIM4v2wdvt2 = value->rValue;
            mod->BSIM4v2wdvt2Given = TRUE;
            break;
        case  BSIM4v2_MOD_WDVT0W:               
            mod->BSIM4v2wdvt0w = value->rValue;
            mod->BSIM4v2wdvt0wGiven = TRUE;
            break;
        case  BSIM4v2_MOD_WDVT1W:             
            mod->BSIM4v2wdvt1w = value->rValue;
            mod->BSIM4v2wdvt1wGiven = TRUE;
            break;
        case  BSIM4v2_MOD_WDVT2W:             
            mod->BSIM4v2wdvt2w = value->rValue;
            mod->BSIM4v2wdvt2wGiven = TRUE;
            break;
        case  BSIM4v2_MOD_WDROUT:             
            mod->BSIM4v2wdrout = value->rValue;
            mod->BSIM4v2wdroutGiven = TRUE;
            break;
        case  BSIM4v2_MOD_WDSUB:             
            mod->BSIM4v2wdsub = value->rValue;
            mod->BSIM4v2wdsubGiven = TRUE;
            break;
        case BSIM4v2_MOD_WVTH0:
            mod->BSIM4v2wvth0 = value->rValue;
            mod->BSIM4v2wvth0Given = TRUE;
            break;
        case BSIM4v2_MOD_WUA:
            mod->BSIM4v2wua = value->rValue;
            mod->BSIM4v2wuaGiven = TRUE;
            break;
        case BSIM4v2_MOD_WUA1:
            mod->BSIM4v2wua1 = value->rValue;
            mod->BSIM4v2wua1Given = TRUE;
            break;
        case BSIM4v2_MOD_WUB:
            mod->BSIM4v2wub = value->rValue;
            mod->BSIM4v2wubGiven = TRUE;
            break;
        case BSIM4v2_MOD_WUB1:
            mod->BSIM4v2wub1 = value->rValue;
            mod->BSIM4v2wub1Given = TRUE;
            break;
        case BSIM4v2_MOD_WUC:
            mod->BSIM4v2wuc = value->rValue;
            mod->BSIM4v2wucGiven = TRUE;
            break;
        case BSIM4v2_MOD_WUC1:
            mod->BSIM4v2wuc1 = value->rValue;
            mod->BSIM4v2wuc1Given = TRUE;
            break;
        case  BSIM4v2_MOD_WU0 :
            mod->BSIM4v2wu0 = value->rValue;
            mod->BSIM4v2wu0Given = TRUE;
            break;
        case  BSIM4v2_MOD_WUTE :
            mod->BSIM4v2wute = value->rValue;
            mod->BSIM4v2wuteGiven = TRUE;
            break;
        case BSIM4v2_MOD_WVOFF:
            mod->BSIM4v2wvoff = value->rValue;
            mod->BSIM4v2wvoffGiven = TRUE;
            break;
        case BSIM4v2_MOD_WMINV:
            mod->BSIM4v2wminv = value->rValue;
            mod->BSIM4v2wminvGiven = TRUE;
            break;
        case BSIM4v2_MOD_WFPROUT:
            mod->BSIM4v2wfprout = value->rValue;
            mod->BSIM4v2wfproutGiven = TRUE;
            break;
        case BSIM4v2_MOD_WPDITS:
            mod->BSIM4v2wpdits = value->rValue;
            mod->BSIM4v2wpditsGiven = TRUE;
            break;
        case BSIM4v2_MOD_WPDITSD:
            mod->BSIM4v2wpditsd = value->rValue;
            mod->BSIM4v2wpditsdGiven = TRUE;
            break;
        case  BSIM4v2_MOD_WDELTA :
            mod->BSIM4v2wdelta = value->rValue;
            mod->BSIM4v2wdeltaGiven = TRUE;
            break;
        case BSIM4v2_MOD_WRDSW:
            mod->BSIM4v2wrdsw = value->rValue;
            mod->BSIM4v2wrdswGiven = TRUE;
            break;                     
        case BSIM4v2_MOD_WRDW:
            mod->BSIM4v2wrdw = value->rValue;
            mod->BSIM4v2wrdwGiven = TRUE;
            break;
        case BSIM4v2_MOD_WRSW:
            mod->BSIM4v2wrsw = value->rValue;
            mod->BSIM4v2wrswGiven = TRUE;
            break;
        case BSIM4v2_MOD_WPRWB:
            mod->BSIM4v2wprwb = value->rValue;
            mod->BSIM4v2wprwbGiven = TRUE;
            break;                     
        case BSIM4v2_MOD_WPRWG:
            mod->BSIM4v2wprwg = value->rValue;
            mod->BSIM4v2wprwgGiven = TRUE;
            break;                     
        case BSIM4v2_MOD_WPRT:
            mod->BSIM4v2wprt = value->rValue;
            mod->BSIM4v2wprtGiven = TRUE;
            break;                     
        case BSIM4v2_MOD_WETA0:
            mod->BSIM4v2weta0 = value->rValue;
            mod->BSIM4v2weta0Given = TRUE;
            break;                 
        case BSIM4v2_MOD_WETAB:
            mod->BSIM4v2wetab = value->rValue;
            mod->BSIM4v2wetabGiven = TRUE;
            break;                 
        case BSIM4v2_MOD_WPCLM:
            mod->BSIM4v2wpclm = value->rValue;
            mod->BSIM4v2wpclmGiven = TRUE;
            break;                 
        case BSIM4v2_MOD_WPDIBL1:
            mod->BSIM4v2wpdibl1 = value->rValue;
            mod->BSIM4v2wpdibl1Given = TRUE;
            break;                 
        case BSIM4v2_MOD_WPDIBL2:
            mod->BSIM4v2wpdibl2 = value->rValue;
            mod->BSIM4v2wpdibl2Given = TRUE;
            break;                 
        case BSIM4v2_MOD_WPDIBLB:
            mod->BSIM4v2wpdiblb = value->rValue;
            mod->BSIM4v2wpdiblbGiven = TRUE;
            break;                 
        case BSIM4v2_MOD_WPSCBE1:
            mod->BSIM4v2wpscbe1 = value->rValue;
            mod->BSIM4v2wpscbe1Given = TRUE;
            break;                 
        case BSIM4v2_MOD_WPSCBE2:
            mod->BSIM4v2wpscbe2 = value->rValue;
            mod->BSIM4v2wpscbe2Given = TRUE;
            break;                 
        case BSIM4v2_MOD_WPVAG:
            mod->BSIM4v2wpvag = value->rValue;
            mod->BSIM4v2wpvagGiven = TRUE;
            break;                 
        case  BSIM4v2_MOD_WWR :
            mod->BSIM4v2wwr = value->rValue;
            mod->BSIM4v2wwrGiven = TRUE;
            break;
        case  BSIM4v2_MOD_WDWG :
            mod->BSIM4v2wdwg = value->rValue;
            mod->BSIM4v2wdwgGiven = TRUE;
            break;
        case  BSIM4v2_MOD_WDWB :
            mod->BSIM4v2wdwb = value->rValue;
            mod->BSIM4v2wdwbGiven = TRUE;
            break;
        case  BSIM4v2_MOD_WB0 :
            mod->BSIM4v2wb0 = value->rValue;
            mod->BSIM4v2wb0Given = TRUE;
            break;
        case  BSIM4v2_MOD_WB1 :
            mod->BSIM4v2wb1 = value->rValue;
            mod->BSIM4v2wb1Given = TRUE;
            break;
        case  BSIM4v2_MOD_WALPHA0 :
            mod->BSIM4v2walpha0 = value->rValue;
            mod->BSIM4v2walpha0Given = TRUE;
            break;
        case  BSIM4v2_MOD_WALPHA1 :
            mod->BSIM4v2walpha1 = value->rValue;
            mod->BSIM4v2walpha1Given = TRUE;
            break;
        case  BSIM4v2_MOD_WBETA0 :
            mod->BSIM4v2wbeta0 = value->rValue;
            mod->BSIM4v2wbeta0Given = TRUE;
            break;
        case  BSIM4v2_MOD_WAGIDL :
            mod->BSIM4v2wagidl = value->rValue;
            mod->BSIM4v2wagidlGiven = TRUE;
            break;
        case  BSIM4v2_MOD_WBGIDL :
            mod->BSIM4v2wbgidl = value->rValue;
            mod->BSIM4v2wbgidlGiven = TRUE;
            break;
        case  BSIM4v2_MOD_WCGIDL :
            mod->BSIM4v2wcgidl = value->rValue;
            mod->BSIM4v2wcgidlGiven = TRUE;
            break;
        case  BSIM4v2_MOD_WPHIN :
            mod->BSIM4v2wphin = value->rValue;
            mod->BSIM4v2wphinGiven = TRUE;
            break;
        case  BSIM4v2_MOD_WEGIDL :
            mod->BSIM4v2wegidl = value->rValue;
            mod->BSIM4v2wegidlGiven = TRUE;
            break;
        case  BSIM4v2_MOD_WAIGC :
            mod->BSIM4v2waigc = value->rValue;
            mod->BSIM4v2waigcGiven = TRUE;
            break;
        case  BSIM4v2_MOD_WBIGC :
            mod->BSIM4v2wbigc = value->rValue;
            mod->BSIM4v2wbigcGiven = TRUE;
            break;
        case  BSIM4v2_MOD_WCIGC :
            mod->BSIM4v2wcigc = value->rValue;
            mod->BSIM4v2wcigcGiven = TRUE;
            break;
        case  BSIM4v2_MOD_WAIGSD :
            mod->BSIM4v2waigsd = value->rValue;
            mod->BSIM4v2waigsdGiven = TRUE;
            break;
        case  BSIM4v2_MOD_WBIGSD :
            mod->BSIM4v2wbigsd = value->rValue;
            mod->BSIM4v2wbigsdGiven = TRUE;
            break;
        case  BSIM4v2_MOD_WCIGSD :
            mod->BSIM4v2wcigsd = value->rValue;
            mod->BSIM4v2wcigsdGiven = TRUE;
            break;
        case  BSIM4v2_MOD_WAIGBACC :
            mod->BSIM4v2waigbacc = value->rValue;
            mod->BSIM4v2waigbaccGiven = TRUE;
            break;
        case  BSIM4v2_MOD_WBIGBACC :
            mod->BSIM4v2wbigbacc = value->rValue;
            mod->BSIM4v2wbigbaccGiven = TRUE;
            break;
        case  BSIM4v2_MOD_WCIGBACC :
            mod->BSIM4v2wcigbacc = value->rValue;
            mod->BSIM4v2wcigbaccGiven = TRUE;
            break;
        case  BSIM4v2_MOD_WAIGBINV :
            mod->BSIM4v2waigbinv = value->rValue;
            mod->BSIM4v2waigbinvGiven = TRUE;
            break;
        case  BSIM4v2_MOD_WBIGBINV :
            mod->BSIM4v2wbigbinv = value->rValue;
            mod->BSIM4v2wbigbinvGiven = TRUE;
            break;
        case  BSIM4v2_MOD_WCIGBINV :
            mod->BSIM4v2wcigbinv = value->rValue;
            mod->BSIM4v2wcigbinvGiven = TRUE;
            break;
        case  BSIM4v2_MOD_WNIGC :
            mod->BSIM4v2wnigc = value->rValue;
            mod->BSIM4v2wnigcGiven = TRUE;
            break;
        case  BSIM4v2_MOD_WNIGBINV :
            mod->BSIM4v2wnigbinv = value->rValue;
            mod->BSIM4v2wnigbinvGiven = TRUE;
            break;
        case  BSIM4v2_MOD_WNIGBACC :
            mod->BSIM4v2wnigbacc = value->rValue;
            mod->BSIM4v2wnigbaccGiven = TRUE;
            break;
        case  BSIM4v2_MOD_WNTOX :
            mod->BSIM4v2wntox = value->rValue;
            mod->BSIM4v2wntoxGiven = TRUE;
            break;
        case  BSIM4v2_MOD_WEIGBINV :
            mod->BSIM4v2weigbinv = value->rValue;
            mod->BSIM4v2weigbinvGiven = TRUE;
            break;
        case  BSIM4v2_MOD_WPIGCD :
            mod->BSIM4v2wpigcd = value->rValue;
            mod->BSIM4v2wpigcdGiven = TRUE;
            break;
        case  BSIM4v2_MOD_WPOXEDGE :
            mod->BSIM4v2wpoxedge = value->rValue;
            mod->BSIM4v2wpoxedgeGiven = TRUE;
            break;
        case  BSIM4v2_MOD_WXRCRG1 :
            mod->BSIM4v2wxrcrg1 = value->rValue;
            mod->BSIM4v2wxrcrg1Given = TRUE;
            break;
        case  BSIM4v2_MOD_WXRCRG2 :
            mod->BSIM4v2wxrcrg2 = value->rValue;
            mod->BSIM4v2wxrcrg2Given = TRUE;
            break;
        case  BSIM4v2_MOD_WEU :
            mod->BSIM4v2weu = value->rValue;
            mod->BSIM4v2weuGiven = TRUE;
            break;
        case  BSIM4v2_MOD_WVFB :
            mod->BSIM4v2wvfb = value->rValue;
            mod->BSIM4v2wvfbGiven = TRUE;
            break;

        case  BSIM4v2_MOD_WCGSL :
            mod->BSIM4v2wcgsl = value->rValue;
            mod->BSIM4v2wcgslGiven = TRUE;
            break;
        case  BSIM4v2_MOD_WCGDL :
            mod->BSIM4v2wcgdl = value->rValue;
            mod->BSIM4v2wcgdlGiven = TRUE;
            break;
        case  BSIM4v2_MOD_WCKAPPAS :
            mod->BSIM4v2wckappas = value->rValue;
            mod->BSIM4v2wckappasGiven = TRUE;
            break;
        case  BSIM4v2_MOD_WCKAPPAD :
            mod->BSIM4v2wckappad = value->rValue;
            mod->BSIM4v2wckappadGiven = TRUE;
            break;
        case  BSIM4v2_MOD_WCF :
            mod->BSIM4v2wcf = value->rValue;
            mod->BSIM4v2wcfGiven = TRUE;
            break;
        case  BSIM4v2_MOD_WCLC :
            mod->BSIM4v2wclc = value->rValue;
            mod->BSIM4v2wclcGiven = TRUE;
            break;
        case  BSIM4v2_MOD_WCLE :
            mod->BSIM4v2wcle = value->rValue;
            mod->BSIM4v2wcleGiven = TRUE;
            break;
        case  BSIM4v2_MOD_WVFBCV :
            mod->BSIM4v2wvfbcv = value->rValue;
            mod->BSIM4v2wvfbcvGiven = TRUE;
            break;
        case  BSIM4v2_MOD_WACDE :
            mod->BSIM4v2wacde = value->rValue;
            mod->BSIM4v2wacdeGiven = TRUE;
            break;
        case  BSIM4v2_MOD_WMOIN :
            mod->BSIM4v2wmoin = value->rValue;
            mod->BSIM4v2wmoinGiven = TRUE;
            break;
        case  BSIM4v2_MOD_WNOFF :
            mod->BSIM4v2wnoff = value->rValue;
            mod->BSIM4v2wnoffGiven = TRUE;
            break;
        case  BSIM4v2_MOD_WVOFFCV :
            mod->BSIM4v2wvoffcv = value->rValue;
            mod->BSIM4v2wvoffcvGiven = TRUE;
            break;

	/* Cross-term dependence */
        case  BSIM4v2_MOD_PCDSC :
            mod->BSIM4v2pcdsc = value->rValue;
            mod->BSIM4v2pcdscGiven = TRUE;
            break;


        case  BSIM4v2_MOD_PCDSCB :
            mod->BSIM4v2pcdscb = value->rValue;
            mod->BSIM4v2pcdscbGiven = TRUE;
            break;
        case  BSIM4v2_MOD_PCDSCD :
            mod->BSIM4v2pcdscd = value->rValue;
            mod->BSIM4v2pcdscdGiven = TRUE;
            break;
        case  BSIM4v2_MOD_PCIT :
            mod->BSIM4v2pcit = value->rValue;
            mod->BSIM4v2pcitGiven = TRUE;
            break;
        case  BSIM4v2_MOD_PNFACTOR :
            mod->BSIM4v2pnfactor = value->rValue;
            mod->BSIM4v2pnfactorGiven = TRUE;
            break;
        case BSIM4v2_MOD_PXJ:
            mod->BSIM4v2pxj = value->rValue;
            mod->BSIM4v2pxjGiven = TRUE;
            break;
        case BSIM4v2_MOD_PVSAT:
            mod->BSIM4v2pvsat = value->rValue;
            mod->BSIM4v2pvsatGiven = TRUE;
            break;


        case BSIM4v2_MOD_PA0:
            mod->BSIM4v2pa0 = value->rValue;
            mod->BSIM4v2pa0Given = TRUE;
            break;
        case BSIM4v2_MOD_PAGS:
            mod->BSIM4v2pags = value->rValue;
            mod->BSIM4v2pagsGiven = TRUE;
            break;
        case BSIM4v2_MOD_PA1:
            mod->BSIM4v2pa1 = value->rValue;
            mod->BSIM4v2pa1Given = TRUE;
            break;
        case BSIM4v2_MOD_PA2:
            mod->BSIM4v2pa2 = value->rValue;
            mod->BSIM4v2pa2Given = TRUE;
            break;
        case BSIM4v2_MOD_PAT:
            mod->BSIM4v2pat = value->rValue;
            mod->BSIM4v2patGiven = TRUE;
            break;
        case BSIM4v2_MOD_PKETA:
            mod->BSIM4v2pketa = value->rValue;
            mod->BSIM4v2pketaGiven = TRUE;
            break;    
        case BSIM4v2_MOD_PNSUB:
            mod->BSIM4v2pnsub = value->rValue;
            mod->BSIM4v2pnsubGiven = TRUE;
            break;
        case BSIM4v2_MOD_PNDEP:
            mod->BSIM4v2pndep = value->rValue;
            mod->BSIM4v2pndepGiven = TRUE;
	    if (mod->BSIM4v2pndep > 1.0e20)
		mod->BSIM4v2pndep *= 1.0e-6;
            break;
        case BSIM4v2_MOD_PNSD:
            mod->BSIM4v2pnsd = value->rValue;
            mod->BSIM4v2pnsdGiven = TRUE;
            if (mod->BSIM4v2pnsd > 1.0e23)
                mod->BSIM4v2pnsd *= 1.0e-6;
            break;
        case BSIM4v2_MOD_PNGATE:
            mod->BSIM4v2pngate = value->rValue;
            mod->BSIM4v2pngateGiven = TRUE;
	    if (mod->BSIM4v2pngate > 1.0e23)
		mod->BSIM4v2pngate *= 1.0e-6;
            break;
        case BSIM4v2_MOD_PGAMMA1:
            mod->BSIM4v2pgamma1 = value->rValue;
            mod->BSIM4v2pgamma1Given = TRUE;
            break;
        case BSIM4v2_MOD_PGAMMA2:
            mod->BSIM4v2pgamma2 = value->rValue;
            mod->BSIM4v2pgamma2Given = TRUE;
            break;
        case BSIM4v2_MOD_PVBX:
            mod->BSIM4v2pvbx = value->rValue;
            mod->BSIM4v2pvbxGiven = TRUE;
            break;
        case BSIM4v2_MOD_PVBM:
            mod->BSIM4v2pvbm = value->rValue;
            mod->BSIM4v2pvbmGiven = TRUE;
            break;
        case BSIM4v2_MOD_PXT:
            mod->BSIM4v2pxt = value->rValue;
            mod->BSIM4v2pxtGiven = TRUE;
            break;
        case  BSIM4v2_MOD_PK1:
            mod->BSIM4v2pk1 = value->rValue;
            mod->BSIM4v2pk1Given = TRUE;
            break;
        case  BSIM4v2_MOD_PKT1:
            mod->BSIM4v2pkt1 = value->rValue;
            mod->BSIM4v2pkt1Given = TRUE;
            break;
        case  BSIM4v2_MOD_PKT1L:
            mod->BSIM4v2pkt1l = value->rValue;
            mod->BSIM4v2pkt1lGiven = TRUE;
            break;
        case  BSIM4v2_MOD_PKT2:
            mod->BSIM4v2pkt2 = value->rValue;
            mod->BSIM4v2pkt2Given = TRUE;
            break;
        case  BSIM4v2_MOD_PK2:
            mod->BSIM4v2pk2 = value->rValue;
            mod->BSIM4v2pk2Given = TRUE;
            break;
        case  BSIM4v2_MOD_PK3:
            mod->BSIM4v2pk3 = value->rValue;
            mod->BSIM4v2pk3Given = TRUE;
            break;
        case  BSIM4v2_MOD_PK3B:
            mod->BSIM4v2pk3b = value->rValue;
            mod->BSIM4v2pk3bGiven = TRUE;
            break;
        case  BSIM4v2_MOD_PLPE0:
            mod->BSIM4v2plpe0 = value->rValue;
            mod->BSIM4v2plpe0Given = TRUE;
            break;
        case  BSIM4v2_MOD_PLPEB:
            mod->BSIM4v2plpeb = value->rValue;
            mod->BSIM4v2plpebGiven = TRUE;
            break;
        case  BSIM4v2_MOD_PDVTP0:
            mod->BSIM4v2pdvtp0 = value->rValue;
            mod->BSIM4v2pdvtp0Given = TRUE;
            break;
        case  BSIM4v2_MOD_PDVTP1:
            mod->BSIM4v2pdvtp1 = value->rValue;
            mod->BSIM4v2pdvtp1Given = TRUE;
            break;
        case  BSIM4v2_MOD_PW0:
            mod->BSIM4v2pw0 = value->rValue;
            mod->BSIM4v2pw0Given = TRUE;
            break;
        case  BSIM4v2_MOD_PDVT0:               
            mod->BSIM4v2pdvt0 = value->rValue;
            mod->BSIM4v2pdvt0Given = TRUE;
            break;
        case  BSIM4v2_MOD_PDVT1:             
            mod->BSIM4v2pdvt1 = value->rValue;
            mod->BSIM4v2pdvt1Given = TRUE;
            break;
        case  BSIM4v2_MOD_PDVT2:             
            mod->BSIM4v2pdvt2 = value->rValue;
            mod->BSIM4v2pdvt2Given = TRUE;
            break;
        case  BSIM4v2_MOD_PDVT0W:               
            mod->BSIM4v2pdvt0w = value->rValue;
            mod->BSIM4v2pdvt0wGiven = TRUE;
            break;
        case  BSIM4v2_MOD_PDVT1W:             
            mod->BSIM4v2pdvt1w = value->rValue;
            mod->BSIM4v2pdvt1wGiven = TRUE;
            break;
        case  BSIM4v2_MOD_PDVT2W:             
            mod->BSIM4v2pdvt2w = value->rValue;
            mod->BSIM4v2pdvt2wGiven = TRUE;
            break;
        case  BSIM4v2_MOD_PDROUT:             
            mod->BSIM4v2pdrout = value->rValue;
            mod->BSIM4v2pdroutGiven = TRUE;
            break;
        case  BSIM4v2_MOD_PDSUB:             
            mod->BSIM4v2pdsub = value->rValue;
            mod->BSIM4v2pdsubGiven = TRUE;
            break;
        case BSIM4v2_MOD_PVTH0:
            mod->BSIM4v2pvth0 = value->rValue;
            mod->BSIM4v2pvth0Given = TRUE;
            break;
        case BSIM4v2_MOD_PUA:
            mod->BSIM4v2pua = value->rValue;
            mod->BSIM4v2puaGiven = TRUE;
            break;
        case BSIM4v2_MOD_PUA1:
            mod->BSIM4v2pua1 = value->rValue;
            mod->BSIM4v2pua1Given = TRUE;
            break;
        case BSIM4v2_MOD_PUB:
            mod->BSIM4v2pub = value->rValue;
            mod->BSIM4v2pubGiven = TRUE;
            break;
        case BSIM4v2_MOD_PUB1:
            mod->BSIM4v2pub1 = value->rValue;
            mod->BSIM4v2pub1Given = TRUE;
            break;
        case BSIM4v2_MOD_PUC:
            mod->BSIM4v2puc = value->rValue;
            mod->BSIM4v2pucGiven = TRUE;
            break;
        case BSIM4v2_MOD_PUC1:
            mod->BSIM4v2puc1 = value->rValue;
            mod->BSIM4v2puc1Given = TRUE;
            break;
        case  BSIM4v2_MOD_PU0 :
            mod->BSIM4v2pu0 = value->rValue;
            mod->BSIM4v2pu0Given = TRUE;
            break;
        case  BSIM4v2_MOD_PUTE :
            mod->BSIM4v2pute = value->rValue;
            mod->BSIM4v2puteGiven = TRUE;
            break;
        case BSIM4v2_MOD_PVOFF:
            mod->BSIM4v2pvoff = value->rValue;
            mod->BSIM4v2pvoffGiven = TRUE;
            break;
        case BSIM4v2_MOD_PMINV:
            mod->BSIM4v2pminv = value->rValue;
            mod->BSIM4v2pminvGiven = TRUE;
            break;
        case BSIM4v2_MOD_PFPROUT:
            mod->BSIM4v2pfprout = value->rValue;
            mod->BSIM4v2pfproutGiven = TRUE;
            break;
        case BSIM4v2_MOD_PPDITS:
            mod->BSIM4v2ppdits = value->rValue;
            mod->BSIM4v2ppditsGiven = TRUE;
            break;
        case BSIM4v2_MOD_PPDITSD:
            mod->BSIM4v2ppditsd = value->rValue;
            mod->BSIM4v2ppditsdGiven = TRUE;
            break;
        case  BSIM4v2_MOD_PDELTA :
            mod->BSIM4v2pdelta = value->rValue;
            mod->BSIM4v2pdeltaGiven = TRUE;
            break;
        case BSIM4v2_MOD_PRDSW:
            mod->BSIM4v2prdsw = value->rValue;
            mod->BSIM4v2prdswGiven = TRUE;
            break;                     
        case BSIM4v2_MOD_PRDW:
            mod->BSIM4v2prdw = value->rValue;
            mod->BSIM4v2prdwGiven = TRUE;
            break;
        case BSIM4v2_MOD_PRSW:
            mod->BSIM4v2prsw = value->rValue;
            mod->BSIM4v2prswGiven = TRUE;
            break;
        case BSIM4v2_MOD_PPRWB:
            mod->BSIM4v2pprwb = value->rValue;
            mod->BSIM4v2pprwbGiven = TRUE;
            break;                     
        case BSIM4v2_MOD_PPRWG:
            mod->BSIM4v2pprwg = value->rValue;
            mod->BSIM4v2pprwgGiven = TRUE;
            break;                     
        case BSIM4v2_MOD_PPRT:
            mod->BSIM4v2pprt = value->rValue;
            mod->BSIM4v2pprtGiven = TRUE;
            break;                     
        case BSIM4v2_MOD_PETA0:
            mod->BSIM4v2peta0 = value->rValue;
            mod->BSIM4v2peta0Given = TRUE;
            break;                 
        case BSIM4v2_MOD_PETAB:
            mod->BSIM4v2petab = value->rValue;
            mod->BSIM4v2petabGiven = TRUE;
            break;                 
        case BSIM4v2_MOD_PPCLM:
            mod->BSIM4v2ppclm = value->rValue;
            mod->BSIM4v2ppclmGiven = TRUE;
            break;                 
        case BSIM4v2_MOD_PPDIBL1:
            mod->BSIM4v2ppdibl1 = value->rValue;
            mod->BSIM4v2ppdibl1Given = TRUE;
            break;                 
        case BSIM4v2_MOD_PPDIBL2:
            mod->BSIM4v2ppdibl2 = value->rValue;
            mod->BSIM4v2ppdibl2Given = TRUE;
            break;                 
        case BSIM4v2_MOD_PPDIBLB:
            mod->BSIM4v2ppdiblb = value->rValue;
            mod->BSIM4v2ppdiblbGiven = TRUE;
            break;                 
        case BSIM4v2_MOD_PPSCBE1:
            mod->BSIM4v2ppscbe1 = value->rValue;
            mod->BSIM4v2ppscbe1Given = TRUE;
            break;                 
        case BSIM4v2_MOD_PPSCBE2:
            mod->BSIM4v2ppscbe2 = value->rValue;
            mod->BSIM4v2ppscbe2Given = TRUE;
            break;                 
        case BSIM4v2_MOD_PPVAG:
            mod->BSIM4v2ppvag = value->rValue;
            mod->BSIM4v2ppvagGiven = TRUE;
            break;                 
        case  BSIM4v2_MOD_PWR :
            mod->BSIM4v2pwr = value->rValue;
            mod->BSIM4v2pwrGiven = TRUE;
            break;
        case  BSIM4v2_MOD_PDWG :
            mod->BSIM4v2pdwg = value->rValue;
            mod->BSIM4v2pdwgGiven = TRUE;
            break;
        case  BSIM4v2_MOD_PDWB :
            mod->BSIM4v2pdwb = value->rValue;
            mod->BSIM4v2pdwbGiven = TRUE;
            break;
        case  BSIM4v2_MOD_PB0 :
            mod->BSIM4v2pb0 = value->rValue;
            mod->BSIM4v2pb0Given = TRUE;
            break;
        case  BSIM4v2_MOD_PB1 :
            mod->BSIM4v2pb1 = value->rValue;
            mod->BSIM4v2pb1Given = TRUE;
            break;
        case  BSIM4v2_MOD_PALPHA0 :
            mod->BSIM4v2palpha0 = value->rValue;
            mod->BSIM4v2palpha0Given = TRUE;
            break;
        case  BSIM4v2_MOD_PALPHA1 :
            mod->BSIM4v2palpha1 = value->rValue;
            mod->BSIM4v2palpha1Given = TRUE;
            break;
        case  BSIM4v2_MOD_PBETA0 :
            mod->BSIM4v2pbeta0 = value->rValue;
            mod->BSIM4v2pbeta0Given = TRUE;
            break;
        case  BSIM4v2_MOD_PAGIDL :
            mod->BSIM4v2pagidl = value->rValue;
            mod->BSIM4v2pagidlGiven = TRUE;
            break;
        case  BSIM4v2_MOD_PBGIDL :
            mod->BSIM4v2pbgidl = value->rValue;
            mod->BSIM4v2pbgidlGiven = TRUE;
            break;
        case  BSIM4v2_MOD_PCGIDL :
            mod->BSIM4v2pcgidl = value->rValue;
            mod->BSIM4v2pcgidlGiven = TRUE;
            break;
        case  BSIM4v2_MOD_PPHIN :
            mod->BSIM4v2pphin = value->rValue;
            mod->BSIM4v2pphinGiven = TRUE;
            break;
        case  BSIM4v2_MOD_PEGIDL :
            mod->BSIM4v2pegidl = value->rValue;
            mod->BSIM4v2pegidlGiven = TRUE;
            break;
        case  BSIM4v2_MOD_PAIGC :
            mod->BSIM4v2paigc = value->rValue;
            mod->BSIM4v2paigcGiven = TRUE;
            break;
        case  BSIM4v2_MOD_PBIGC :
            mod->BSIM4v2pbigc = value->rValue;
            mod->BSIM4v2pbigcGiven = TRUE;
            break;
        case  BSIM4v2_MOD_PCIGC :
            mod->BSIM4v2pcigc = value->rValue;
            mod->BSIM4v2pcigcGiven = TRUE;
            break;
        case  BSIM4v2_MOD_PAIGSD :
            mod->BSIM4v2paigsd = value->rValue;
            mod->BSIM4v2paigsdGiven = TRUE;
            break;
        case  BSIM4v2_MOD_PBIGSD :
            mod->BSIM4v2pbigsd = value->rValue;
            mod->BSIM4v2pbigsdGiven = TRUE;
            break;
        case  BSIM4v2_MOD_PCIGSD :
            mod->BSIM4v2pcigsd = value->rValue;
            mod->BSIM4v2pcigsdGiven = TRUE;
            break;
        case  BSIM4v2_MOD_PAIGBACC :
            mod->BSIM4v2paigbacc = value->rValue;
            mod->BSIM4v2paigbaccGiven = TRUE;
            break;
        case  BSIM4v2_MOD_PBIGBACC :
            mod->BSIM4v2pbigbacc = value->rValue;
            mod->BSIM4v2pbigbaccGiven = TRUE;
            break;
        case  BSIM4v2_MOD_PCIGBACC :
            mod->BSIM4v2pcigbacc = value->rValue;
            mod->BSIM4v2pcigbaccGiven = TRUE;
            break;
        case  BSIM4v2_MOD_PAIGBINV :
            mod->BSIM4v2paigbinv = value->rValue;
            mod->BSIM4v2paigbinvGiven = TRUE;
            break;
        case  BSIM4v2_MOD_PBIGBINV :
            mod->BSIM4v2pbigbinv = value->rValue;
            mod->BSIM4v2pbigbinvGiven = TRUE;
            break;
        case  BSIM4v2_MOD_PCIGBINV :
            mod->BSIM4v2pcigbinv = value->rValue;
            mod->BSIM4v2pcigbinvGiven = TRUE;
            break;
        case  BSIM4v2_MOD_PNIGC :
            mod->BSIM4v2pnigc = value->rValue;
            mod->BSIM4v2pnigcGiven = TRUE;
            break;
        case  BSIM4v2_MOD_PNIGBINV :
            mod->BSIM4v2pnigbinv = value->rValue;
            mod->BSIM4v2pnigbinvGiven = TRUE;
            break;
        case  BSIM4v2_MOD_PNIGBACC :
            mod->BSIM4v2pnigbacc = value->rValue;
            mod->BSIM4v2pnigbaccGiven = TRUE;
            break;
        case  BSIM4v2_MOD_PNTOX :
            mod->BSIM4v2pntox = value->rValue;
            mod->BSIM4v2pntoxGiven = TRUE;
            break;
        case  BSIM4v2_MOD_PEIGBINV :
            mod->BSIM4v2peigbinv = value->rValue;
            mod->BSIM4v2peigbinvGiven = TRUE;
            break;
        case  BSIM4v2_MOD_PPIGCD :
            mod->BSIM4v2ppigcd = value->rValue;
            mod->BSIM4v2ppigcdGiven = TRUE;
            break;
        case  BSIM4v2_MOD_PPOXEDGE :
            mod->BSIM4v2ppoxedge = value->rValue;
            mod->BSIM4v2ppoxedgeGiven = TRUE;
            break;
        case  BSIM4v2_MOD_PXRCRG1 :
            mod->BSIM4v2pxrcrg1 = value->rValue;
            mod->BSIM4v2pxrcrg1Given = TRUE;
            break;
        case  BSIM4v2_MOD_PXRCRG2 :
            mod->BSIM4v2pxrcrg2 = value->rValue;
            mod->BSIM4v2pxrcrg2Given = TRUE;
            break;
        case  BSIM4v2_MOD_PEU :
            mod->BSIM4v2peu = value->rValue;
            mod->BSIM4v2peuGiven = TRUE;
            break;
        case  BSIM4v2_MOD_PVFB :
            mod->BSIM4v2pvfb = value->rValue;
            mod->BSIM4v2pvfbGiven = TRUE;
            break;

        case  BSIM4v2_MOD_PCGSL :
            mod->BSIM4v2pcgsl = value->rValue;
            mod->BSIM4v2pcgslGiven = TRUE;
            break;
        case  BSIM4v2_MOD_PCGDL :
            mod->BSIM4v2pcgdl = value->rValue;
            mod->BSIM4v2pcgdlGiven = TRUE;
            break;
        case  BSIM4v2_MOD_PCKAPPAS :
            mod->BSIM4v2pckappas = value->rValue;
            mod->BSIM4v2pckappasGiven = TRUE;
            break;
        case  BSIM4v2_MOD_PCKAPPAD :
            mod->BSIM4v2pckappad = value->rValue;
            mod->BSIM4v2pckappadGiven = TRUE;
            break;
        case  BSIM4v2_MOD_PCF :
            mod->BSIM4v2pcf = value->rValue;
            mod->BSIM4v2pcfGiven = TRUE;
            break;
        case  BSIM4v2_MOD_PCLC :
            mod->BSIM4v2pclc = value->rValue;
            mod->BSIM4v2pclcGiven = TRUE;
            break;
        case  BSIM4v2_MOD_PCLE :
            mod->BSIM4v2pcle = value->rValue;
            mod->BSIM4v2pcleGiven = TRUE;
            break;
        case  BSIM4v2_MOD_PVFBCV :
            mod->BSIM4v2pvfbcv = value->rValue;
            mod->BSIM4v2pvfbcvGiven = TRUE;
            break;
        case  BSIM4v2_MOD_PACDE :
            mod->BSIM4v2pacde = value->rValue;
            mod->BSIM4v2pacdeGiven = TRUE;
            break;
        case  BSIM4v2_MOD_PMOIN :
            mod->BSIM4v2pmoin = value->rValue;
            mod->BSIM4v2pmoinGiven = TRUE;
            break;
        case  BSIM4v2_MOD_PNOFF :
            mod->BSIM4v2pnoff = value->rValue;
            mod->BSIM4v2pnoffGiven = TRUE;
            break;
        case  BSIM4v2_MOD_PVOFFCV :
            mod->BSIM4v2pvoffcv = value->rValue;
            mod->BSIM4v2pvoffcvGiven = TRUE;
            break;

        case  BSIM4v2_MOD_TNOM :
            mod->BSIM4v2tnom = value->rValue + CONSTCtoK;
            mod->BSIM4v2tnomGiven = TRUE;
            break;
        case  BSIM4v2_MOD_CGSO :
            mod->BSIM4v2cgso = value->rValue;
            mod->BSIM4v2cgsoGiven = TRUE;
            break;
        case  BSIM4v2_MOD_CGDO :
            mod->BSIM4v2cgdo = value->rValue;
            mod->BSIM4v2cgdoGiven = TRUE;
            break;
        case  BSIM4v2_MOD_CGBO :
            mod->BSIM4v2cgbo = value->rValue;
            mod->BSIM4v2cgboGiven = TRUE;
            break;
        case  BSIM4v2_MOD_XPART :
            mod->BSIM4v2xpart = value->rValue;
            mod->BSIM4v2xpartGiven = TRUE;
            break;
        case  BSIM4v2_MOD_RSH :
            mod->BSIM4v2sheetResistance = value->rValue;
            mod->BSIM4v2sheetResistanceGiven = TRUE;
            break;
        case  BSIM4v2_MOD_JSS :
            mod->BSIM4v2SjctSatCurDensity = value->rValue;
            mod->BSIM4v2SjctSatCurDensityGiven = TRUE;
            break;
        case  BSIM4v2_MOD_JSWS :
            mod->BSIM4v2SjctSidewallSatCurDensity = value->rValue;
            mod->BSIM4v2SjctSidewallSatCurDensityGiven = TRUE;
            break;
        case  BSIM4v2_MOD_JSWGS :
            mod->BSIM4v2SjctGateSidewallSatCurDensity = value->rValue;
            mod->BSIM4v2SjctGateSidewallSatCurDensityGiven = TRUE;
            break;
        case  BSIM4v2_MOD_PBS :
            mod->BSIM4v2SbulkJctPotential = value->rValue;
            mod->BSIM4v2SbulkJctPotentialGiven = TRUE;
            break;
        case  BSIM4v2_MOD_MJS :
            mod->BSIM4v2SbulkJctBotGradingCoeff = value->rValue;
            mod->BSIM4v2SbulkJctBotGradingCoeffGiven = TRUE;
            break;
        case  BSIM4v2_MOD_PBSWS :
            mod->BSIM4v2SsidewallJctPotential = value->rValue;
            mod->BSIM4v2SsidewallJctPotentialGiven = TRUE;
            break;
        case  BSIM4v2_MOD_MJSWS :
            mod->BSIM4v2SbulkJctSideGradingCoeff = value->rValue;
            mod->BSIM4v2SbulkJctSideGradingCoeffGiven = TRUE;
            break;
        case  BSIM4v2_MOD_CJS :
            mod->BSIM4v2SunitAreaJctCap = value->rValue;
            mod->BSIM4v2SunitAreaJctCapGiven = TRUE;
            break;
        case  BSIM4v2_MOD_CJSWS :
            mod->BSIM4v2SunitLengthSidewallJctCap = value->rValue;
            mod->BSIM4v2SunitLengthSidewallJctCapGiven = TRUE;
            break;
        case  BSIM4v2_MOD_NJS :
            mod->BSIM4v2SjctEmissionCoeff = value->rValue;
            mod->BSIM4v2SjctEmissionCoeffGiven = TRUE;
            break;
        case  BSIM4v2_MOD_PBSWGS :
            mod->BSIM4v2SGatesidewallJctPotential = value->rValue;
            mod->BSIM4v2SGatesidewallJctPotentialGiven = TRUE;
            break;
        case  BSIM4v2_MOD_MJSWGS :
            mod->BSIM4v2SbulkJctGateSideGradingCoeff = value->rValue;
            mod->BSIM4v2SbulkJctGateSideGradingCoeffGiven = TRUE;
            break;
        case  BSIM4v2_MOD_CJSWGS :
            mod->BSIM4v2SunitLengthGateSidewallJctCap = value->rValue;
            mod->BSIM4v2SunitLengthGateSidewallJctCapGiven = TRUE;
            break;
        case  BSIM4v2_MOD_XTIS :
            mod->BSIM4v2SjctTempExponent = value->rValue;
            mod->BSIM4v2SjctTempExponentGiven = TRUE;
            break;
        case  BSIM4v2_MOD_JSD :
            mod->BSIM4v2DjctSatCurDensity = value->rValue;
            mod->BSIM4v2DjctSatCurDensityGiven = TRUE;
            break;
        case  BSIM4v2_MOD_JSWD :
            mod->BSIM4v2DjctSidewallSatCurDensity = value->rValue;
            mod->BSIM4v2DjctSidewallSatCurDensityGiven = TRUE;
            break;
        case  BSIM4v2_MOD_JSWGD :
            mod->BSIM4v2DjctGateSidewallSatCurDensity = value->rValue;
            mod->BSIM4v2DjctGateSidewallSatCurDensityGiven = TRUE;
            break;
        case  BSIM4v2_MOD_PBD :
            mod->BSIM4v2DbulkJctPotential = value->rValue;
            mod->BSIM4v2DbulkJctPotentialGiven = TRUE;
            break;
        case  BSIM4v2_MOD_MJD :
            mod->BSIM4v2DbulkJctBotGradingCoeff = value->rValue;
            mod->BSIM4v2DbulkJctBotGradingCoeffGiven = TRUE;
            break;
        case  BSIM4v2_MOD_PBSWD :
            mod->BSIM4v2DsidewallJctPotential = value->rValue;
            mod->BSIM4v2DsidewallJctPotentialGiven = TRUE;
            break;
        case  BSIM4v2_MOD_MJSWD :
            mod->BSIM4v2DbulkJctSideGradingCoeff = value->rValue;
            mod->BSIM4v2DbulkJctSideGradingCoeffGiven = TRUE;
            break;
        case  BSIM4v2_MOD_CJD :
            mod->BSIM4v2DunitAreaJctCap = value->rValue;
            mod->BSIM4v2DunitAreaJctCapGiven = TRUE;
            break;
        case  BSIM4v2_MOD_CJSWD :
            mod->BSIM4v2DunitLengthSidewallJctCap = value->rValue;
            mod->BSIM4v2DunitLengthSidewallJctCapGiven = TRUE;
            break;
        case  BSIM4v2_MOD_NJD :
            mod->BSIM4v2DjctEmissionCoeff = value->rValue;
            mod->BSIM4v2DjctEmissionCoeffGiven = TRUE;
            break;
        case  BSIM4v2_MOD_PBSWGD :
            mod->BSIM4v2DGatesidewallJctPotential = value->rValue;
            mod->BSIM4v2DGatesidewallJctPotentialGiven = TRUE;
            break;
        case  BSIM4v2_MOD_MJSWGD :
            mod->BSIM4v2DbulkJctGateSideGradingCoeff = value->rValue;
            mod->BSIM4v2DbulkJctGateSideGradingCoeffGiven = TRUE;
            break;
        case  BSIM4v2_MOD_CJSWGD :
            mod->BSIM4v2DunitLengthGateSidewallJctCap = value->rValue;
            mod->BSIM4v2DunitLengthGateSidewallJctCapGiven = TRUE;
            break;
        case  BSIM4v2_MOD_XTID :
            mod->BSIM4v2DjctTempExponent = value->rValue;
            mod->BSIM4v2DjctTempExponentGiven = TRUE;
            break;
        case  BSIM4v2_MOD_LINT :
            mod->BSIM4v2Lint = value->rValue;
            mod->BSIM4v2LintGiven = TRUE;
            break;
        case  BSIM4v2_MOD_LL :
            mod->BSIM4v2Ll = value->rValue;
            mod->BSIM4v2LlGiven = TRUE;
            break;
        case  BSIM4v2_MOD_LLC :
            mod->BSIM4v2Llc = value->rValue;
            mod->BSIM4v2LlcGiven = TRUE;
            break;
        case  BSIM4v2_MOD_LLN :
            mod->BSIM4v2Lln = value->rValue;
            mod->BSIM4v2LlnGiven = TRUE;
            break;
        case  BSIM4v2_MOD_LW :
            mod->BSIM4v2Lw = value->rValue;
            mod->BSIM4v2LwGiven = TRUE;
            break;
        case  BSIM4v2_MOD_LWC :
            mod->BSIM4v2Lwc = value->rValue;
            mod->BSIM4v2LwcGiven = TRUE;
            break;
        case  BSIM4v2_MOD_LWN :
            mod->BSIM4v2Lwn = value->rValue;
            mod->BSIM4v2LwnGiven = TRUE;
            break;
        case  BSIM4v2_MOD_LWL :
            mod->BSIM4v2Lwl = value->rValue;
            mod->BSIM4v2LwlGiven = TRUE;
            break;
        case  BSIM4v2_MOD_LWLC :
            mod->BSIM4v2Lwlc = value->rValue;
            mod->BSIM4v2LwlcGiven = TRUE;
            break;
        case  BSIM4v2_MOD_LMIN :
            mod->BSIM4v2Lmin = value->rValue;
            mod->BSIM4v2LminGiven = TRUE;
            break;
        case  BSIM4v2_MOD_LMAX :
            mod->BSIM4v2Lmax = value->rValue;
            mod->BSIM4v2LmaxGiven = TRUE;
            break;
        case  BSIM4v2_MOD_WINT :
            mod->BSIM4v2Wint = value->rValue;
            mod->BSIM4v2WintGiven = TRUE;
            break;
        case  BSIM4v2_MOD_WL :
            mod->BSIM4v2Wl = value->rValue;
            mod->BSIM4v2WlGiven = TRUE;
            break;
        case  BSIM4v2_MOD_WLC :
            mod->BSIM4v2Wlc = value->rValue;
            mod->BSIM4v2WlcGiven = TRUE;
            break;
        case  BSIM4v2_MOD_WLN :
            mod->BSIM4v2Wln = value->rValue;
            mod->BSIM4v2WlnGiven = TRUE;
            break;
        case  BSIM4v2_MOD_WW :
            mod->BSIM4v2Ww = value->rValue;
            mod->BSIM4v2WwGiven = TRUE;
            break;
        case  BSIM4v2_MOD_WWC :
            mod->BSIM4v2Wwc = value->rValue;
            mod->BSIM4v2WwcGiven = TRUE;
            break;
        case  BSIM4v2_MOD_WWN :
            mod->BSIM4v2Wwn = value->rValue;
            mod->BSIM4v2WwnGiven = TRUE;
            break;
        case  BSIM4v2_MOD_WWL :
            mod->BSIM4v2Wwl = value->rValue;
            mod->BSIM4v2WwlGiven = TRUE;
            break;
        case  BSIM4v2_MOD_WWLC :
            mod->BSIM4v2Wwlc = value->rValue;
            mod->BSIM4v2WwlcGiven = TRUE;
            break;
        case  BSIM4v2_MOD_WMIN :
            mod->BSIM4v2Wmin = value->rValue;
            mod->BSIM4v2WminGiven = TRUE;
            break;
        case  BSIM4v2_MOD_WMAX :
            mod->BSIM4v2Wmax = value->rValue;
            mod->BSIM4v2WmaxGiven = TRUE;
            break;

        case  BSIM4v2_MOD_NOIA :
            mod->BSIM4v2oxideTrapDensityA = value->rValue;
            mod->BSIM4v2oxideTrapDensityAGiven = TRUE;
            break;
        case  BSIM4v2_MOD_NOIB :
            mod->BSIM4v2oxideTrapDensityB = value->rValue;
            mod->BSIM4v2oxideTrapDensityBGiven = TRUE;
            break;
        case  BSIM4v2_MOD_NOIC :
            mod->BSIM4v2oxideTrapDensityC = value->rValue;
            mod->BSIM4v2oxideTrapDensityCGiven = TRUE;
            break;
        case  BSIM4v2_MOD_EM :
            mod->BSIM4v2em = value->rValue;
            mod->BSIM4v2emGiven = TRUE;
            break;
        case  BSIM4v2_MOD_EF :
            mod->BSIM4v2ef = value->rValue;
            mod->BSIM4v2efGiven = TRUE;
            break;
        case  BSIM4v2_MOD_AF :
            mod->BSIM4v2af = value->rValue;
            mod->BSIM4v2afGiven = TRUE;
            break;
        case  BSIM4v2_MOD_KF :
            mod->BSIM4v2kf = value->rValue;
            mod->BSIM4v2kfGiven = TRUE;
            break;
        case  BSIM4v2_MOD_NMOS  :
            if(value->iValue) {
                mod->BSIM4v2type = 1;
                mod->BSIM4v2typeGiven = TRUE;
            }
            break;
        case  BSIM4v2_MOD_PMOS  :
            if(value->iValue) {
                mod->BSIM4v2type = - 1;
                mod->BSIM4v2typeGiven = TRUE;
            }
            break;
        default:
            return(E_BADPARM);
    }
    return(OK);
}


