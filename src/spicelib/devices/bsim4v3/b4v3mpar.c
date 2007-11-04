/**** BSIM4.3.0 Released by Xuemei(Jane) Xi 05/09/2003 ****/

/**********
 * Copyright 2003 Regents of the University of California. All rights reserved.
 * File: b4v3mpar.c of BSIM4.3.0.
 * Author: 2000 Weidong Liu
 * Authors: 2001- Xuemei Xi, Jin He, Kanyu Cao, Mohan Dunga, Mansun Chan, Ali Niknejad, Chenming Hu.
 * Project Director: Prof. Chenming Hu.
 * Modified by Xuemei Xi, 04/06/2001.
 * Modified by Xuemei Xi, 10/05/2001.
 * Modified by Xuemei Xi, 11/15/2002.
 * Modified by Xuemei Xi, 05/09/2003.
 **********/

#include "ngspice.h"
#include <stdio.h>
#include "bsim4v3def.h"
#include "ifsim.h"
#include "sperror.h"
#include "const.h"

int
BSIM4v3mParam(param,value,inMod)
int param;
IFvalue *value;
GENmodel *inMod;
{
    BSIM4v3model *mod = (BSIM4v3model*)inMod;
    switch(param)
    {   case  BSIM4v3_MOD_MOBMOD :
            mod->BSIM4v3mobMod = value->iValue;
            mod->BSIM4v3mobModGiven = TRUE;
            break;
        case  BSIM4v3_MOD_BINUNIT :
            mod->BSIM4v3binUnit = value->iValue;
            mod->BSIM4v3binUnitGiven = TRUE;
            break;
        case  BSIM4v3_MOD_PARAMCHK :
            mod->BSIM4v3paramChk = value->iValue;
            mod->BSIM4v3paramChkGiven = TRUE;
            break;
        case  BSIM4v3_MOD_CAPMOD :
            mod->BSIM4v3capMod = value->iValue;
            mod->BSIM4v3capModGiven = TRUE;
            break;
        case  BSIM4v3_MOD_DIOMOD :
            mod->BSIM4v3dioMod = value->iValue;
            mod->BSIM4v3dioModGiven = TRUE;
            break;
        case  BSIM4v3_MOD_RDSMOD :
            mod->BSIM4v3rdsMod = value->iValue;
            mod->BSIM4v3rdsModGiven = TRUE;
            break;
        case  BSIM4v3_MOD_TRNQSMOD :
            mod->BSIM4v3trnqsMod = value->iValue;
            mod->BSIM4v3trnqsModGiven = TRUE;
            break;
        case  BSIM4v3_MOD_ACNQSMOD :
            mod->BSIM4v3acnqsMod = value->iValue;
            mod->BSIM4v3acnqsModGiven = TRUE;
            break;
        case  BSIM4v3_MOD_RBODYMOD :
            mod->BSIM4v3rbodyMod = value->iValue;
            mod->BSIM4v3rbodyModGiven = TRUE;
            break;
        case  BSIM4v3_MOD_RGATEMOD :
            mod->BSIM4v3rgateMod = value->iValue;
            mod->BSIM4v3rgateModGiven = TRUE;
            break;
        case  BSIM4v3_MOD_PERMOD :
            mod->BSIM4v3perMod = value->iValue;
            mod->BSIM4v3perModGiven = TRUE;
            break;
        case  BSIM4v3_MOD_GEOMOD :
            mod->BSIM4v3geoMod = value->iValue;
            mod->BSIM4v3geoModGiven = TRUE;
            break;
        case  BSIM4v3_MOD_FNOIMOD :
            mod->BSIM4v3fnoiMod = value->iValue;
            mod->BSIM4v3fnoiModGiven = TRUE;
            break;
        case  BSIM4v3_MOD_TNOIMOD :
            mod->BSIM4v3tnoiMod = value->iValue;
            mod->BSIM4v3tnoiModGiven = TRUE;
            break;
        case  BSIM4v3_MOD_IGCMOD :
            mod->BSIM4v3igcMod = value->iValue;
            mod->BSIM4v3igcModGiven = TRUE;
            break;
        case  BSIM4v3_MOD_IGBMOD :
            mod->BSIM4v3igbMod = value->iValue;
            mod->BSIM4v3igbModGiven = TRUE;
            break;
        case  BSIM4v3_MOD_TEMPMOD :
            mod->BSIM4v3tempMod = value->iValue;
            mod->BSIM4v3tempModGiven = TRUE;
            break;
        case  BSIM4v3_MOD_VERSION :
            mod->BSIM4v3version = value->sValue;
            mod->BSIM4v3versionGiven = TRUE;
            break;
        case  BSIM4v3_MOD_TOXREF :
            mod->BSIM4v3toxref = value->rValue;
            mod->BSIM4v3toxrefGiven = TRUE;
            break;
        case  BSIM4v3_MOD_TOXE :
            mod->BSIM4v3toxe = value->rValue;
            mod->BSIM4v3toxeGiven = TRUE;
            break;
        case  BSIM4v3_MOD_TOXP :
            mod->BSIM4v3toxp = value->rValue;
            mod->BSIM4v3toxpGiven = TRUE;
            break;
        case  BSIM4v3_MOD_TOXM :
            mod->BSIM4v3toxm = value->rValue;
            mod->BSIM4v3toxmGiven = TRUE;
            break;
        case  BSIM4v3_MOD_DTOX :
            mod->BSIM4v3dtox = value->rValue;
            mod->BSIM4v3dtoxGiven = TRUE;
            break;
        case  BSIM4v3_MOD_EPSROX :
            mod->BSIM4v3epsrox = value->rValue;
            mod->BSIM4v3epsroxGiven = TRUE;
            break;

        case  BSIM4v3_MOD_CDSC :
            mod->BSIM4v3cdsc = value->rValue;
            mod->BSIM4v3cdscGiven = TRUE;
            break;
        case  BSIM4v3_MOD_CDSCB :
            mod->BSIM4v3cdscb = value->rValue;
            mod->BSIM4v3cdscbGiven = TRUE;
            break;

        case  BSIM4v3_MOD_CDSCD :
            mod->BSIM4v3cdscd = value->rValue;
            mod->BSIM4v3cdscdGiven = TRUE;
            break;

        case  BSIM4v3_MOD_CIT :
            mod->BSIM4v3cit = value->rValue;
            mod->BSIM4v3citGiven = TRUE;
            break;
        case  BSIM4v3_MOD_NFACTOR :
            mod->BSIM4v3nfactor = value->rValue;
            mod->BSIM4v3nfactorGiven = TRUE;
            break;
        case BSIM4v3_MOD_XJ:
            mod->BSIM4v3xj = value->rValue;
            mod->BSIM4v3xjGiven = TRUE;
            break;
        case BSIM4v3_MOD_VSAT:
            mod->BSIM4v3vsat = value->rValue;
            mod->BSIM4v3vsatGiven = TRUE;
            break;
        case BSIM4v3_MOD_A0:
            mod->BSIM4v3a0 = value->rValue;
            mod->BSIM4v3a0Given = TRUE;
            break;
        
        case BSIM4v3_MOD_AGS:
            mod->BSIM4v3ags= value->rValue;
            mod->BSIM4v3agsGiven = TRUE;
            break;
        
        case BSIM4v3_MOD_A1:
            mod->BSIM4v3a1 = value->rValue;
            mod->BSIM4v3a1Given = TRUE;
            break;
        case BSIM4v3_MOD_A2:
            mod->BSIM4v3a2 = value->rValue;
            mod->BSIM4v3a2Given = TRUE;
            break;
        case BSIM4v3_MOD_AT:
            mod->BSIM4v3at = value->rValue;
            mod->BSIM4v3atGiven = TRUE;
            break;
        case BSIM4v3_MOD_KETA:
            mod->BSIM4v3keta = value->rValue;
            mod->BSIM4v3ketaGiven = TRUE;
            break;    
        case BSIM4v3_MOD_NSUB:
            mod->BSIM4v3nsub = value->rValue;
            mod->BSIM4v3nsubGiven = TRUE;
            break;
        case BSIM4v3_MOD_NDEP:
            mod->BSIM4v3ndep = value->rValue;
            mod->BSIM4v3ndepGiven = TRUE;
	    if (mod->BSIM4v3ndep > 1.0e20)
		mod->BSIM4v3ndep *= 1.0e-6;
            break;
        case BSIM4v3_MOD_NSD:
            mod->BSIM4v3nsd = value->rValue;
            mod->BSIM4v3nsdGiven = TRUE;
            if (mod->BSIM4v3nsd > 1.0e23)
                mod->BSIM4v3nsd *= 1.0e-6;
            break;
        case BSIM4v3_MOD_NGATE:
            mod->BSIM4v3ngate = value->rValue;
            mod->BSIM4v3ngateGiven = TRUE;
	    if (mod->BSIM4v3ngate > 1.0e23)
		mod->BSIM4v3ngate *= 1.0e-6;
            break;
        case BSIM4v3_MOD_GAMMA1:
            mod->BSIM4v3gamma1 = value->rValue;
            mod->BSIM4v3gamma1Given = TRUE;
            break;
        case BSIM4v3_MOD_GAMMA2:
            mod->BSIM4v3gamma2 = value->rValue;
            mod->BSIM4v3gamma2Given = TRUE;
            break;
        case BSIM4v3_MOD_VBX:
            mod->BSIM4v3vbx = value->rValue;
            mod->BSIM4v3vbxGiven = TRUE;
            break;
        case BSIM4v3_MOD_VBM:
            mod->BSIM4v3vbm = value->rValue;
            mod->BSIM4v3vbmGiven = TRUE;
            break;
        case BSIM4v3_MOD_XT:
            mod->BSIM4v3xt = value->rValue;
            mod->BSIM4v3xtGiven = TRUE;
            break;
        case  BSIM4v3_MOD_K1:
            mod->BSIM4v3k1 = value->rValue;
            mod->BSIM4v3k1Given = TRUE;
            break;
        case  BSIM4v3_MOD_KT1:
            mod->BSIM4v3kt1 = value->rValue;
            mod->BSIM4v3kt1Given = TRUE;
            break;
        case  BSIM4v3_MOD_KT1L:
            mod->BSIM4v3kt1l = value->rValue;
            mod->BSIM4v3kt1lGiven = TRUE;
            break;
        case  BSIM4v3_MOD_KT2:
            mod->BSIM4v3kt2 = value->rValue;
            mod->BSIM4v3kt2Given = TRUE;
            break;
        case  BSIM4v3_MOD_K2:
            mod->BSIM4v3k2 = value->rValue;
            mod->BSIM4v3k2Given = TRUE;
            break;
        case  BSIM4v3_MOD_K3:
            mod->BSIM4v3k3 = value->rValue;
            mod->BSIM4v3k3Given = TRUE;
            break;
        case  BSIM4v3_MOD_K3B:
            mod->BSIM4v3k3b = value->rValue;
            mod->BSIM4v3k3bGiven = TRUE;
            break;
        case  BSIM4v3_MOD_LPE0:
            mod->BSIM4v3lpe0 = value->rValue;
            mod->BSIM4v3lpe0Given = TRUE;
            break;
        case  BSIM4v3_MOD_LPEB:
            mod->BSIM4v3lpeb = value->rValue;
            mod->BSIM4v3lpebGiven = TRUE;
            break;
        case  BSIM4v3_MOD_DVTP0:
            mod->BSIM4v3dvtp0 = value->rValue;
            mod->BSIM4v3dvtp0Given = TRUE;
            break;
        case  BSIM4v3_MOD_DVTP1:
            mod->BSIM4v3dvtp1 = value->rValue;
            mod->BSIM4v3dvtp1Given = TRUE;
            break;
        case  BSIM4v3_MOD_W0:
            mod->BSIM4v3w0 = value->rValue;
            mod->BSIM4v3w0Given = TRUE;
            break;
        case  BSIM4v3_MOD_DVT0:               
            mod->BSIM4v3dvt0 = value->rValue;
            mod->BSIM4v3dvt0Given = TRUE;
            break;
        case  BSIM4v3_MOD_DVT1:             
            mod->BSIM4v3dvt1 = value->rValue;
            mod->BSIM4v3dvt1Given = TRUE;
            break;
        case  BSIM4v3_MOD_DVT2:             
            mod->BSIM4v3dvt2 = value->rValue;
            mod->BSIM4v3dvt2Given = TRUE;
            break;
        case  BSIM4v3_MOD_DVT0W:               
            mod->BSIM4v3dvt0w = value->rValue;
            mod->BSIM4v3dvt0wGiven = TRUE;
            break;
        case  BSIM4v3_MOD_DVT1W:             
            mod->BSIM4v3dvt1w = value->rValue;
            mod->BSIM4v3dvt1wGiven = TRUE;
            break;
        case  BSIM4v3_MOD_DVT2W:             
            mod->BSIM4v3dvt2w = value->rValue;
            mod->BSIM4v3dvt2wGiven = TRUE;
            break;
        case  BSIM4v3_MOD_DROUT:             
            mod->BSIM4v3drout = value->rValue;
            mod->BSIM4v3droutGiven = TRUE;
            break;
        case  BSIM4v3_MOD_DSUB:             
            mod->BSIM4v3dsub = value->rValue;
            mod->BSIM4v3dsubGiven = TRUE;
            break;
        case BSIM4v3_MOD_VTH0:
            mod->BSIM4v3vth0 = value->rValue;
            mod->BSIM4v3vth0Given = TRUE;
            break;
        case BSIM4v3_MOD_EU:
            mod->BSIM4v3eu = value->rValue;
            mod->BSIM4v3euGiven = TRUE;
            break;
        case BSIM4v3_MOD_UA:
            mod->BSIM4v3ua = value->rValue;
            mod->BSIM4v3uaGiven = TRUE;
            break;
        case BSIM4v3_MOD_UA1:
            mod->BSIM4v3ua1 = value->rValue;
            mod->BSIM4v3ua1Given = TRUE;
            break;
        case BSIM4v3_MOD_UB:
            mod->BSIM4v3ub = value->rValue;
            mod->BSIM4v3ubGiven = TRUE;
            break;
        case BSIM4v3_MOD_UB1:
            mod->BSIM4v3ub1 = value->rValue;
            mod->BSIM4v3ub1Given = TRUE;
            break;
        case BSIM4v3_MOD_UC:
            mod->BSIM4v3uc = value->rValue;
            mod->BSIM4v3ucGiven = TRUE;
            break;
        case BSIM4v3_MOD_UC1:
            mod->BSIM4v3uc1 = value->rValue;
            mod->BSIM4v3uc1Given = TRUE;
            break;
        case  BSIM4v3_MOD_U0 :
            mod->BSIM4v3u0 = value->rValue;
            mod->BSIM4v3u0Given = TRUE;
            break;
        case  BSIM4v3_MOD_UTE :
            mod->BSIM4v3ute = value->rValue;
            mod->BSIM4v3uteGiven = TRUE;
            break;
        case BSIM4v3_MOD_VOFF:
            mod->BSIM4v3voff = value->rValue;
            mod->BSIM4v3voffGiven = TRUE;
            break;
        case BSIM4v3_MOD_VOFFL:
            mod->BSIM4v3voffl = value->rValue;
            mod->BSIM4v3vofflGiven = TRUE;
            break;
        case BSIM4v3_MOD_MINV:
            mod->BSIM4v3minv = value->rValue;
            mod->BSIM4v3minvGiven = TRUE;
            break;
        case BSIM4v3_MOD_FPROUT:
            mod->BSIM4v3fprout = value->rValue;
            mod->BSIM4v3fproutGiven = TRUE;
            break;
        case BSIM4v3_MOD_PDITS:
            mod->BSIM4v3pdits = value->rValue;
            mod->BSIM4v3pditsGiven = TRUE;
            break;
        case BSIM4v3_MOD_PDITSD:
            mod->BSIM4v3pditsd = value->rValue;
            mod->BSIM4v3pditsdGiven = TRUE;
            break;
        case BSIM4v3_MOD_PDITSL:
            mod->BSIM4v3pditsl = value->rValue;
            mod->BSIM4v3pditslGiven = TRUE;
            break;
        case  BSIM4v3_MOD_DELTA :
            mod->BSIM4v3delta = value->rValue;
            mod->BSIM4v3deltaGiven = TRUE;
            break;
        case BSIM4v3_MOD_RDSW:
            mod->BSIM4v3rdsw = value->rValue;
            mod->BSIM4v3rdswGiven = TRUE;
            break;                     
        case BSIM4v3_MOD_RDSWMIN:
            mod->BSIM4v3rdswmin = value->rValue;
            mod->BSIM4v3rdswminGiven = TRUE;
            break;
        case BSIM4v3_MOD_RDWMIN:
            mod->BSIM4v3rdwmin = value->rValue;
            mod->BSIM4v3rdwminGiven = TRUE;
            break;
        case BSIM4v3_MOD_RSWMIN:
            mod->BSIM4v3rswmin = value->rValue;
            mod->BSIM4v3rswminGiven = TRUE;
            break;
        case BSIM4v3_MOD_RDW:
            mod->BSIM4v3rdw = value->rValue;
            mod->BSIM4v3rdwGiven = TRUE;
            break;
        case BSIM4v3_MOD_RSW:
            mod->BSIM4v3rsw = value->rValue;
            mod->BSIM4v3rswGiven = TRUE;
            break;
        case BSIM4v3_MOD_PRWG:
            mod->BSIM4v3prwg = value->rValue;
            mod->BSIM4v3prwgGiven = TRUE;
            break;                     
        case BSIM4v3_MOD_PRWB:
            mod->BSIM4v3prwb = value->rValue;
            mod->BSIM4v3prwbGiven = TRUE;
            break;                     
        case BSIM4v3_MOD_PRT:
            mod->BSIM4v3prt = value->rValue;
            mod->BSIM4v3prtGiven = TRUE;
            break;                     
        case BSIM4v3_MOD_ETA0:
            mod->BSIM4v3eta0 = value->rValue;
            mod->BSIM4v3eta0Given = TRUE;
            break;                 
        case BSIM4v3_MOD_ETAB:
            mod->BSIM4v3etab = value->rValue;
            mod->BSIM4v3etabGiven = TRUE;
            break;                 
        case BSIM4v3_MOD_PCLM:
            mod->BSIM4v3pclm = value->rValue;
            mod->BSIM4v3pclmGiven = TRUE;
            break;                 
        case BSIM4v3_MOD_PDIBL1:
            mod->BSIM4v3pdibl1 = value->rValue;
            mod->BSIM4v3pdibl1Given = TRUE;
            break;                 
        case BSIM4v3_MOD_PDIBL2:
            mod->BSIM4v3pdibl2 = value->rValue;
            mod->BSIM4v3pdibl2Given = TRUE;
            break;                 
        case BSIM4v3_MOD_PDIBLB:
            mod->BSIM4v3pdiblb = value->rValue;
            mod->BSIM4v3pdiblbGiven = TRUE;
            break;                 
        case BSIM4v3_MOD_PSCBE1:
            mod->BSIM4v3pscbe1 = value->rValue;
            mod->BSIM4v3pscbe1Given = TRUE;
            break;                 
        case BSIM4v3_MOD_PSCBE2:
            mod->BSIM4v3pscbe2 = value->rValue;
            mod->BSIM4v3pscbe2Given = TRUE;
            break;                 
        case BSIM4v3_MOD_PVAG:
            mod->BSIM4v3pvag = value->rValue;
            mod->BSIM4v3pvagGiven = TRUE;
            break;                 
        case  BSIM4v3_MOD_WR :
            mod->BSIM4v3wr = value->rValue;
            mod->BSIM4v3wrGiven = TRUE;
            break;
        case  BSIM4v3_MOD_DWG :
            mod->BSIM4v3dwg = value->rValue;
            mod->BSIM4v3dwgGiven = TRUE;
            break;
        case  BSIM4v3_MOD_DWB :
            mod->BSIM4v3dwb = value->rValue;
            mod->BSIM4v3dwbGiven = TRUE;
            break;
        case  BSIM4v3_MOD_B0 :
            mod->BSIM4v3b0 = value->rValue;
            mod->BSIM4v3b0Given = TRUE;
            break;
        case  BSIM4v3_MOD_B1 :
            mod->BSIM4v3b1 = value->rValue;
            mod->BSIM4v3b1Given = TRUE;
            break;
        case  BSIM4v3_MOD_ALPHA0 :
            mod->BSIM4v3alpha0 = value->rValue;
            mod->BSIM4v3alpha0Given = TRUE;
            break;
        case  BSIM4v3_MOD_ALPHA1 :
            mod->BSIM4v3alpha1 = value->rValue;
            mod->BSIM4v3alpha1Given = TRUE;
            break;
        case  BSIM4v3_MOD_AGIDL :
            mod->BSIM4v3agidl = value->rValue;
            mod->BSIM4v3agidlGiven = TRUE;
            break;
        case  BSIM4v3_MOD_BGIDL :
            mod->BSIM4v3bgidl = value->rValue;
            mod->BSIM4v3bgidlGiven = TRUE;
            break;
        case  BSIM4v3_MOD_CGIDL :
            mod->BSIM4v3cgidl = value->rValue;
            mod->BSIM4v3cgidlGiven = TRUE;
            break;
        case  BSIM4v3_MOD_PHIN :
            mod->BSIM4v3phin = value->rValue;
            mod->BSIM4v3phinGiven = TRUE;
            break;
        case  BSIM4v3_MOD_EGIDL :
            mod->BSIM4v3egidl = value->rValue;
            mod->BSIM4v3egidlGiven = TRUE;
            break;
        case  BSIM4v3_MOD_AIGC :
            mod->BSIM4v3aigc = value->rValue;
            mod->BSIM4v3aigcGiven = TRUE;
            break;
        case  BSIM4v3_MOD_BIGC :
            mod->BSIM4v3bigc = value->rValue;
            mod->BSIM4v3bigcGiven = TRUE;
            break;
        case  BSIM4v3_MOD_CIGC :
            mod->BSIM4v3cigc = value->rValue;
            mod->BSIM4v3cigcGiven = TRUE;
            break;
        case  BSIM4v3_MOD_AIGSD :
            mod->BSIM4v3aigsd = value->rValue;
            mod->BSIM4v3aigsdGiven = TRUE;
            break;
        case  BSIM4v3_MOD_BIGSD :
            mod->BSIM4v3bigsd = value->rValue;
            mod->BSIM4v3bigsdGiven = TRUE;
            break;
        case  BSIM4v3_MOD_CIGSD :
            mod->BSIM4v3cigsd = value->rValue;
            mod->BSIM4v3cigsdGiven = TRUE;
            break;
        case  BSIM4v3_MOD_AIGBACC :
            mod->BSIM4v3aigbacc = value->rValue;
            mod->BSIM4v3aigbaccGiven = TRUE;
            break;
        case  BSIM4v3_MOD_BIGBACC :
            mod->BSIM4v3bigbacc = value->rValue;
            mod->BSIM4v3bigbaccGiven = TRUE;
            break;
        case  BSIM4v3_MOD_CIGBACC :
            mod->BSIM4v3cigbacc = value->rValue;
            mod->BSIM4v3cigbaccGiven = TRUE;
            break;
        case  BSIM4v3_MOD_AIGBINV :
            mod->BSIM4v3aigbinv = value->rValue;
            mod->BSIM4v3aigbinvGiven = TRUE;
            break;
        case  BSIM4v3_MOD_BIGBINV :
            mod->BSIM4v3bigbinv = value->rValue;
            mod->BSIM4v3bigbinvGiven = TRUE;
            break;
        case  BSIM4v3_MOD_CIGBINV :
            mod->BSIM4v3cigbinv = value->rValue;
            mod->BSIM4v3cigbinvGiven = TRUE;
            break;
        case  BSIM4v3_MOD_NIGC :
            mod->BSIM4v3nigc = value->rValue;
            mod->BSIM4v3nigcGiven = TRUE;
            break;
        case  BSIM4v3_MOD_NIGBINV :
            mod->BSIM4v3nigbinv = value->rValue;
            mod->BSIM4v3nigbinvGiven = TRUE;
            break;
        case  BSIM4v3_MOD_NIGBACC :
            mod->BSIM4v3nigbacc = value->rValue;
            mod->BSIM4v3nigbaccGiven = TRUE;
            break;
        case  BSIM4v3_MOD_NTOX :
            mod->BSIM4v3ntox = value->rValue;
            mod->BSIM4v3ntoxGiven = TRUE;
            break;
        case  BSIM4v3_MOD_EIGBINV :
            mod->BSIM4v3eigbinv = value->rValue;
            mod->BSIM4v3eigbinvGiven = TRUE;
            break;
        case  BSIM4v3_MOD_PIGCD :
            mod->BSIM4v3pigcd = value->rValue;
            mod->BSIM4v3pigcdGiven = TRUE;
            break;
        case  BSIM4v3_MOD_POXEDGE :
            mod->BSIM4v3poxedge = value->rValue;
            mod->BSIM4v3poxedgeGiven = TRUE;
            break;
        case  BSIM4v3_MOD_XRCRG1 :
            mod->BSIM4v3xrcrg1 = value->rValue;
            mod->BSIM4v3xrcrg1Given = TRUE;
            break;
        case  BSIM4v3_MOD_XRCRG2 :
            mod->BSIM4v3xrcrg2 = value->rValue;
            mod->BSIM4v3xrcrg2Given = TRUE;
            break;
        case  BSIM4v3_MOD_LAMBDA :
            mod->BSIM4v3lambda = value->rValue;
            mod->BSIM4v3lambdaGiven = TRUE;
            break;
        case  BSIM4v3_MOD_VTL :
            mod->BSIM4v3vtl = value->rValue;
            mod->BSIM4v3vtlGiven = TRUE;
            break;
        case  BSIM4v3_MOD_XN:
            mod->BSIM4v3xn = value->rValue;
            mod->BSIM4v3xnGiven = TRUE;
            break;
        case  BSIM4v3_MOD_LC:
            mod->BSIM4v3lc = value->rValue;
            mod->BSIM4v3lcGiven = TRUE;
            break;
        case  BSIM4v3_MOD_TNOIA :
            mod->BSIM4v3tnoia = value->rValue;
            mod->BSIM4v3tnoiaGiven = TRUE;
            break;
        case  BSIM4v3_MOD_TNOIB :
            mod->BSIM4v3tnoib = value->rValue;
            mod->BSIM4v3tnoibGiven = TRUE;
            break;
        case  BSIM4v3_MOD_RNOIA :
            mod->BSIM4v3rnoia = value->rValue;
            mod->BSIM4v3rnoiaGiven = TRUE;
            break;
        case  BSIM4v3_MOD_RNOIB :
            mod->BSIM4v3rnoib = value->rValue;
            mod->BSIM4v3rnoibGiven = TRUE;
            break;
        case  BSIM4v3_MOD_NTNOI :
            mod->BSIM4v3ntnoi = value->rValue;
            mod->BSIM4v3ntnoiGiven = TRUE;
            break;

        /* stress effect */
        case  BSIM4v3_MOD_SAREF :
            mod->BSIM4v3saref = value->rValue;
            mod->BSIM4v3sarefGiven = TRUE;
            break;
        case  BSIM4v3_MOD_SBREF :
            mod->BSIM4v3sbref = value->rValue;
            mod->BSIM4v3sbrefGiven = TRUE;
            break;
        case  BSIM4v3_MOD_WLOD :
            mod->BSIM4v3wlod = value->rValue;
            mod->BSIM4v3wlodGiven = TRUE;
            break;
        case  BSIM4v3_MOD_KU0 :
            mod->BSIM4v3ku0 = value->rValue;
            mod->BSIM4v3ku0Given = TRUE;
            break;
        case  BSIM4v3_MOD_KVSAT :
            mod->BSIM4v3kvsat = value->rValue;
            mod->BSIM4v3kvsatGiven = TRUE;
            break;
        case  BSIM4v3_MOD_KVTH0 :
            mod->BSIM4v3kvth0 = value->rValue;
            mod->BSIM4v3kvth0Given = TRUE;
            break;
        case  BSIM4v3_MOD_TKU0 :
            mod->BSIM4v3tku0 = value->rValue;
            mod->BSIM4v3tku0Given = TRUE;
            break;
        case  BSIM4v3_MOD_LLODKU0 :
            mod->BSIM4v3llodku0 = value->rValue;
            mod->BSIM4v3llodku0Given = TRUE;
            break;
        case  BSIM4v3_MOD_WLODKU0 :
            mod->BSIM4v3wlodku0 = value->rValue;
            mod->BSIM4v3wlodku0Given = TRUE;
            break;
        case  BSIM4v3_MOD_LLODVTH :
            mod->BSIM4v3llodvth = value->rValue;
            mod->BSIM4v3llodvthGiven = TRUE;
            break;
        case  BSIM4v3_MOD_WLODVTH :
            mod->BSIM4v3wlodvth = value->rValue;
            mod->BSIM4v3wlodvthGiven = TRUE;
            break;
        case  BSIM4v3_MOD_LKU0 :
            mod->BSIM4v3lku0 = value->rValue;
            mod->BSIM4v3lku0Given = TRUE;
            break;
        case  BSIM4v3_MOD_WKU0 :
            mod->BSIM4v3wku0 = value->rValue;
            mod->BSIM4v3wku0Given = TRUE;
            break;
        case  BSIM4v3_MOD_PKU0 :
            mod->BSIM4v3pku0 = value->rValue;
            mod->BSIM4v3pku0Given = TRUE;
            break;
        case  BSIM4v3_MOD_LKVTH0 :
            mod->BSIM4v3lkvth0 = value->rValue;
            mod->BSIM4v3lkvth0Given = TRUE;
            break;
        case  BSIM4v3_MOD_WKVTH0 :
            mod->BSIM4v3wkvth0 = value->rValue;
            mod->BSIM4v3wkvth0Given = TRUE;
            break;
        case  BSIM4v3_MOD_PKVTH0 :
            mod->BSIM4v3pkvth0 = value->rValue;
            mod->BSIM4v3pkvth0Given = TRUE;
            break;
        case  BSIM4v3_MOD_STK2 :
            mod->BSIM4v3stk2 = value->rValue;
            mod->BSIM4v3stk2Given = TRUE;
            break;
        case  BSIM4v3_MOD_LODK2 :
            mod->BSIM4v3lodk2 = value->rValue;
            mod->BSIM4v3lodk2Given = TRUE;
            break;
        case  BSIM4v3_MOD_STETA0 :
            mod->BSIM4v3steta0 = value->rValue;
            mod->BSIM4v3steta0Given = TRUE;
            break;
        case  BSIM4v3_MOD_LODETA0 :
            mod->BSIM4v3lodeta0 = value->rValue;
            mod->BSIM4v3lodeta0Given = TRUE;
            break;

        case  BSIM4v3_MOD_BETA0 :
            mod->BSIM4v3beta0 = value->rValue;
            mod->BSIM4v3beta0Given = TRUE;
            break;
        case  BSIM4v3_MOD_IJTHDFWD :
            mod->BSIM4v3ijthdfwd = value->rValue;
            mod->BSIM4v3ijthdfwdGiven = TRUE;
            break;
        case  BSIM4v3_MOD_IJTHSFWD :
            mod->BSIM4v3ijthsfwd = value->rValue;
            mod->BSIM4v3ijthsfwdGiven = TRUE;
            break;
        case  BSIM4v3_MOD_IJTHDREV :
            mod->BSIM4v3ijthdrev = value->rValue;
            mod->BSIM4v3ijthdrevGiven = TRUE;
            break;
        case  BSIM4v3_MOD_IJTHSREV :
            mod->BSIM4v3ijthsrev = value->rValue;
            mod->BSIM4v3ijthsrevGiven = TRUE;
            break;
        case  BSIM4v3_MOD_XJBVD :
            mod->BSIM4v3xjbvd = value->rValue;
            mod->BSIM4v3xjbvdGiven = TRUE;
            break;
        case  BSIM4v3_MOD_XJBVS :
            mod->BSIM4v3xjbvs = value->rValue;
            mod->BSIM4v3xjbvsGiven = TRUE;
            break;
        case  BSIM4v3_MOD_BVD :
            mod->BSIM4v3bvd = value->rValue;
            mod->BSIM4v3bvdGiven = TRUE;
            break;
        case  BSIM4v3_MOD_BVS :
            mod->BSIM4v3bvs = value->rValue;
            mod->BSIM4v3bvsGiven = TRUE;
            break;
        case  BSIM4v3_MOD_VFB :
            mod->BSIM4v3vfb = value->rValue;
            mod->BSIM4v3vfbGiven = TRUE;
            break;

        case  BSIM4v3_MOD_GBMIN :
            mod->BSIM4v3gbmin = value->rValue;
            mod->BSIM4v3gbminGiven = TRUE;
            break;
        case  BSIM4v3_MOD_RBDB :
            mod->BSIM4v3rbdb = value->rValue;
            mod->BSIM4v3rbdbGiven = TRUE;
            break;
        case  BSIM4v3_MOD_RBPB :
            mod->BSIM4v3rbpb = value->rValue;
            mod->BSIM4v3rbpbGiven = TRUE;
            break;
        case  BSIM4v3_MOD_RBSB :
            mod->BSIM4v3rbsb = value->rValue;
            mod->BSIM4v3rbsbGiven = TRUE;
            break;
        case  BSIM4v3_MOD_RBPS :
            mod->BSIM4v3rbps = value->rValue;
            mod->BSIM4v3rbpsGiven = TRUE;
            break;
        case  BSIM4v3_MOD_RBPD :
            mod->BSIM4v3rbpd = value->rValue;
            mod->BSIM4v3rbpdGiven = TRUE;
            break;

        case  BSIM4v3_MOD_CGSL :
            mod->BSIM4v3cgsl = value->rValue;
            mod->BSIM4v3cgslGiven = TRUE;
            break;
        case  BSIM4v3_MOD_CGDL :
            mod->BSIM4v3cgdl = value->rValue;
            mod->BSIM4v3cgdlGiven = TRUE;
            break;
        case  BSIM4v3_MOD_CKAPPAS :
            mod->BSIM4v3ckappas = value->rValue;
            mod->BSIM4v3ckappasGiven = TRUE;
            break;
        case  BSIM4v3_MOD_CKAPPAD :
            mod->BSIM4v3ckappad = value->rValue;
            mod->BSIM4v3ckappadGiven = TRUE;
            break;
        case  BSIM4v3_MOD_CF :
            mod->BSIM4v3cf = value->rValue;
            mod->BSIM4v3cfGiven = TRUE;
            break;
        case  BSIM4v3_MOD_CLC :
            mod->BSIM4v3clc = value->rValue;
            mod->BSIM4v3clcGiven = TRUE;
            break;
        case  BSIM4v3_MOD_CLE :
            mod->BSIM4v3cle = value->rValue;
            mod->BSIM4v3cleGiven = TRUE;
            break;
        case  BSIM4v3_MOD_DWC :
            mod->BSIM4v3dwc = value->rValue;
            mod->BSIM4v3dwcGiven = TRUE;
            break;
        case  BSIM4v3_MOD_DLC :
            mod->BSIM4v3dlc = value->rValue;
            mod->BSIM4v3dlcGiven = TRUE;
            break;
        case  BSIM4v3_MOD_XW :
            mod->BSIM4v3xw = value->rValue;
            mod->BSIM4v3xwGiven = TRUE;
            break;
        case  BSIM4v3_MOD_XL :
            mod->BSIM4v3xl = value->rValue;
            mod->BSIM4v3xlGiven = TRUE;
            break;
        case  BSIM4v3_MOD_DLCIG :
            mod->BSIM4v3dlcig = value->rValue;
            mod->BSIM4v3dlcigGiven = TRUE;
            break;
        case  BSIM4v3_MOD_DWJ :
            mod->BSIM4v3dwj = value->rValue;
            mod->BSIM4v3dwjGiven = TRUE;
            break;
        case  BSIM4v3_MOD_VFBCV :
            mod->BSIM4v3vfbcv = value->rValue;
            mod->BSIM4v3vfbcvGiven = TRUE;
            break;
        case  BSIM4v3_MOD_ACDE :
            mod->BSIM4v3acde = value->rValue;
            mod->BSIM4v3acdeGiven = TRUE;
            break;
        case  BSIM4v3_MOD_MOIN :
            mod->BSIM4v3moin = value->rValue;
            mod->BSIM4v3moinGiven = TRUE;
            break;
        case  BSIM4v3_MOD_NOFF :
            mod->BSIM4v3noff = value->rValue;
            mod->BSIM4v3noffGiven = TRUE;
            break;
        case  BSIM4v3_MOD_VOFFCV :
            mod->BSIM4v3voffcv = value->rValue;
            mod->BSIM4v3voffcvGiven = TRUE;
            break;
        case  BSIM4v3_MOD_DMCG :
            mod->BSIM4v3dmcg = value->rValue;
            mod->BSIM4v3dmcgGiven = TRUE;
            break;
        case  BSIM4v3_MOD_DMCI :
            mod->BSIM4v3dmci = value->rValue;
            mod->BSIM4v3dmciGiven = TRUE;
            break;
        case  BSIM4v3_MOD_DMDG :
            mod->BSIM4v3dmdg = value->rValue;
            mod->BSIM4v3dmdgGiven = TRUE;
            break;
        case  BSIM4v3_MOD_DMCGT :
            mod->BSIM4v3dmcgt = value->rValue;
            mod->BSIM4v3dmcgtGiven = TRUE;
            break;
        case  BSIM4v3_MOD_XGW :
            mod->BSIM4v3xgw = value->rValue;
            mod->BSIM4v3xgwGiven = TRUE;
            break;
        case  BSIM4v3_MOD_XGL :
            mod->BSIM4v3xgl = value->rValue;
            mod->BSIM4v3xglGiven = TRUE;
            break;
        case  BSIM4v3_MOD_RSHG :
            mod->BSIM4v3rshg = value->rValue;
            mod->BSIM4v3rshgGiven = TRUE;
            break;
        case  BSIM4v3_MOD_NGCON :
            mod->BSIM4v3ngcon = value->rValue;
            mod->BSIM4v3ngconGiven = TRUE;
            break;
        case  BSIM4v3_MOD_TCJ :
            mod->BSIM4v3tcj = value->rValue;
            mod->BSIM4v3tcjGiven = TRUE;
            break;
        case  BSIM4v3_MOD_TPB :
            mod->BSIM4v3tpb = value->rValue;
            mod->BSIM4v3tpbGiven = TRUE;
            break;
        case  BSIM4v3_MOD_TCJSW :
            mod->BSIM4v3tcjsw = value->rValue;
            mod->BSIM4v3tcjswGiven = TRUE;
            break;
        case  BSIM4v3_MOD_TPBSW :
            mod->BSIM4v3tpbsw = value->rValue;
            mod->BSIM4v3tpbswGiven = TRUE;
            break;
        case  BSIM4v3_MOD_TCJSWG :
            mod->BSIM4v3tcjswg = value->rValue;
            mod->BSIM4v3tcjswgGiven = TRUE;
            break;
        case  BSIM4v3_MOD_TPBSWG :
            mod->BSIM4v3tpbswg = value->rValue;
            mod->BSIM4v3tpbswgGiven = TRUE;
            break;

	/* Length dependence */
        case  BSIM4v3_MOD_LCDSC :
            mod->BSIM4v3lcdsc = value->rValue;
            mod->BSIM4v3lcdscGiven = TRUE;
            break;


        case  BSIM4v3_MOD_LCDSCB :
            mod->BSIM4v3lcdscb = value->rValue;
            mod->BSIM4v3lcdscbGiven = TRUE;
            break;
        case  BSIM4v3_MOD_LCDSCD :
            mod->BSIM4v3lcdscd = value->rValue;
            mod->BSIM4v3lcdscdGiven = TRUE;
            break;
        case  BSIM4v3_MOD_LCIT :
            mod->BSIM4v3lcit = value->rValue;
            mod->BSIM4v3lcitGiven = TRUE;
            break;
        case  BSIM4v3_MOD_LNFACTOR :
            mod->BSIM4v3lnfactor = value->rValue;
            mod->BSIM4v3lnfactorGiven = TRUE;
            break;
        case BSIM4v3_MOD_LXJ:
            mod->BSIM4v3lxj = value->rValue;
            mod->BSIM4v3lxjGiven = TRUE;
            break;
        case BSIM4v3_MOD_LVSAT:
            mod->BSIM4v3lvsat = value->rValue;
            mod->BSIM4v3lvsatGiven = TRUE;
            break;
        
        
        case BSIM4v3_MOD_LA0:
            mod->BSIM4v3la0 = value->rValue;
            mod->BSIM4v3la0Given = TRUE;
            break;
        case BSIM4v3_MOD_LAGS:
            mod->BSIM4v3lags = value->rValue;
            mod->BSIM4v3lagsGiven = TRUE;
            break;
        case BSIM4v3_MOD_LA1:
            mod->BSIM4v3la1 = value->rValue;
            mod->BSIM4v3la1Given = TRUE;
            break;
        case BSIM4v3_MOD_LA2:
            mod->BSIM4v3la2 = value->rValue;
            mod->BSIM4v3la2Given = TRUE;
            break;
        case BSIM4v3_MOD_LAT:
            mod->BSIM4v3lat = value->rValue;
            mod->BSIM4v3latGiven = TRUE;
            break;
        case BSIM4v3_MOD_LKETA:
            mod->BSIM4v3lketa = value->rValue;
            mod->BSIM4v3lketaGiven = TRUE;
            break;    
        case BSIM4v3_MOD_LNSUB:
            mod->BSIM4v3lnsub = value->rValue;
            mod->BSIM4v3lnsubGiven = TRUE;
            break;
        case BSIM4v3_MOD_LNDEP:
            mod->BSIM4v3lndep = value->rValue;
            mod->BSIM4v3lndepGiven = TRUE;
	    if (mod->BSIM4v3lndep > 1.0e20)
		mod->BSIM4v3lndep *= 1.0e-6;
            break;
        case BSIM4v3_MOD_LNSD:
            mod->BSIM4v3lnsd = value->rValue;
            mod->BSIM4v3lnsdGiven = TRUE;
            if (mod->BSIM4v3lnsd > 1.0e23)
                mod->BSIM4v3lnsd *= 1.0e-6;
            break;
        case BSIM4v3_MOD_LNGATE:
            mod->BSIM4v3lngate = value->rValue;
            mod->BSIM4v3lngateGiven = TRUE;
	    if (mod->BSIM4v3lngate > 1.0e23)
		mod->BSIM4v3lngate *= 1.0e-6;
            break;
        case BSIM4v3_MOD_LGAMMA1:
            mod->BSIM4v3lgamma1 = value->rValue;
            mod->BSIM4v3lgamma1Given = TRUE;
            break;
        case BSIM4v3_MOD_LGAMMA2:
            mod->BSIM4v3lgamma2 = value->rValue;
            mod->BSIM4v3lgamma2Given = TRUE;
            break;
        case BSIM4v3_MOD_LVBX:
            mod->BSIM4v3lvbx = value->rValue;
            mod->BSIM4v3lvbxGiven = TRUE;
            break;
        case BSIM4v3_MOD_LVBM:
            mod->BSIM4v3lvbm = value->rValue;
            mod->BSIM4v3lvbmGiven = TRUE;
            break;
        case BSIM4v3_MOD_LXT:
            mod->BSIM4v3lxt = value->rValue;
            mod->BSIM4v3lxtGiven = TRUE;
            break;
        case  BSIM4v3_MOD_LK1:
            mod->BSIM4v3lk1 = value->rValue;
            mod->BSIM4v3lk1Given = TRUE;
            break;
        case  BSIM4v3_MOD_LKT1:
            mod->BSIM4v3lkt1 = value->rValue;
            mod->BSIM4v3lkt1Given = TRUE;
            break;
        case  BSIM4v3_MOD_LKT1L:
            mod->BSIM4v3lkt1l = value->rValue;
            mod->BSIM4v3lkt1lGiven = TRUE;
            break;
        case  BSIM4v3_MOD_LKT2:
            mod->BSIM4v3lkt2 = value->rValue;
            mod->BSIM4v3lkt2Given = TRUE;
            break;
        case  BSIM4v3_MOD_LK2:
            mod->BSIM4v3lk2 = value->rValue;
            mod->BSIM4v3lk2Given = TRUE;
            break;
        case  BSIM4v3_MOD_LK3:
            mod->BSIM4v3lk3 = value->rValue;
            mod->BSIM4v3lk3Given = TRUE;
            break;
        case  BSIM4v3_MOD_LK3B:
            mod->BSIM4v3lk3b = value->rValue;
            mod->BSIM4v3lk3bGiven = TRUE;
            break;
        case  BSIM4v3_MOD_LLPE0:
            mod->BSIM4v3llpe0 = value->rValue;
            mod->BSIM4v3llpe0Given = TRUE;
            break;
        case  BSIM4v3_MOD_LLPEB:
            mod->BSIM4v3llpeb = value->rValue;
            mod->BSIM4v3llpebGiven = TRUE;
            break;
        case  BSIM4v3_MOD_LDVTP0:
            mod->BSIM4v3ldvtp0 = value->rValue;
            mod->BSIM4v3ldvtp0Given = TRUE;
            break;
        case  BSIM4v3_MOD_LDVTP1:
            mod->BSIM4v3ldvtp1 = value->rValue;
            mod->BSIM4v3ldvtp1Given = TRUE;
            break;
        case  BSIM4v3_MOD_LW0:
            mod->BSIM4v3lw0 = value->rValue;
            mod->BSIM4v3lw0Given = TRUE;
            break;
        case  BSIM4v3_MOD_LDVT0:               
            mod->BSIM4v3ldvt0 = value->rValue;
            mod->BSIM4v3ldvt0Given = TRUE;
            break;
        case  BSIM4v3_MOD_LDVT1:             
            mod->BSIM4v3ldvt1 = value->rValue;
            mod->BSIM4v3ldvt1Given = TRUE;
            break;
        case  BSIM4v3_MOD_LDVT2:             
            mod->BSIM4v3ldvt2 = value->rValue;
            mod->BSIM4v3ldvt2Given = TRUE;
            break;
        case  BSIM4v3_MOD_LDVT0W:               
            mod->BSIM4v3ldvt0w = value->rValue;
            mod->BSIM4v3ldvt0wGiven = TRUE;
            break;
        case  BSIM4v3_MOD_LDVT1W:             
            mod->BSIM4v3ldvt1w = value->rValue;
            mod->BSIM4v3ldvt1wGiven = TRUE;
            break;
        case  BSIM4v3_MOD_LDVT2W:             
            mod->BSIM4v3ldvt2w = value->rValue;
            mod->BSIM4v3ldvt2wGiven = TRUE;
            break;
        case  BSIM4v3_MOD_LDROUT:             
            mod->BSIM4v3ldrout = value->rValue;
            mod->BSIM4v3ldroutGiven = TRUE;
            break;
        case  BSIM4v3_MOD_LDSUB:             
            mod->BSIM4v3ldsub = value->rValue;
            mod->BSIM4v3ldsubGiven = TRUE;
            break;
        case BSIM4v3_MOD_LVTH0:
            mod->BSIM4v3lvth0 = value->rValue;
            mod->BSIM4v3lvth0Given = TRUE;
            break;
        case BSIM4v3_MOD_LUA:
            mod->BSIM4v3lua = value->rValue;
            mod->BSIM4v3luaGiven = TRUE;
            break;
        case BSIM4v3_MOD_LUA1:
            mod->BSIM4v3lua1 = value->rValue;
            mod->BSIM4v3lua1Given = TRUE;
            break;
        case BSIM4v3_MOD_LUB:
            mod->BSIM4v3lub = value->rValue;
            mod->BSIM4v3lubGiven = TRUE;
            break;
        case BSIM4v3_MOD_LUB1:
            mod->BSIM4v3lub1 = value->rValue;
            mod->BSIM4v3lub1Given = TRUE;
            break;
        case BSIM4v3_MOD_LUC:
            mod->BSIM4v3luc = value->rValue;
            mod->BSIM4v3lucGiven = TRUE;
            break;
        case BSIM4v3_MOD_LUC1:
            mod->BSIM4v3luc1 = value->rValue;
            mod->BSIM4v3luc1Given = TRUE;
            break;
        case  BSIM4v3_MOD_LU0 :
            mod->BSIM4v3lu0 = value->rValue;
            mod->BSIM4v3lu0Given = TRUE;
            break;
        case  BSIM4v3_MOD_LUTE :
            mod->BSIM4v3lute = value->rValue;
            mod->BSIM4v3luteGiven = TRUE;
            break;
        case BSIM4v3_MOD_LVOFF:
            mod->BSIM4v3lvoff = value->rValue;
            mod->BSIM4v3lvoffGiven = TRUE;
            break;
        case BSIM4v3_MOD_LMINV:
            mod->BSIM4v3lminv = value->rValue;
            mod->BSIM4v3lminvGiven = TRUE;
            break;
        case BSIM4v3_MOD_LFPROUT:
            mod->BSIM4v3lfprout = value->rValue;
            mod->BSIM4v3lfproutGiven = TRUE;
            break;
        case BSIM4v3_MOD_LPDITS:
            mod->BSIM4v3lpdits = value->rValue;
            mod->BSIM4v3lpditsGiven = TRUE;
            break;
        case BSIM4v3_MOD_LPDITSD:
            mod->BSIM4v3lpditsd = value->rValue;
            mod->BSIM4v3lpditsdGiven = TRUE;
            break;
        case  BSIM4v3_MOD_LDELTA :
            mod->BSIM4v3ldelta = value->rValue;
            mod->BSIM4v3ldeltaGiven = TRUE;
            break;
        case BSIM4v3_MOD_LRDSW:
            mod->BSIM4v3lrdsw = value->rValue;
            mod->BSIM4v3lrdswGiven = TRUE;
            break;                     
        case BSIM4v3_MOD_LRDW:
            mod->BSIM4v3lrdw = value->rValue;
            mod->BSIM4v3lrdwGiven = TRUE;
            break;
        case BSIM4v3_MOD_LRSW:
            mod->BSIM4v3lrsw = value->rValue;
            mod->BSIM4v3lrswGiven = TRUE;
            break;
        case BSIM4v3_MOD_LPRWB:
            mod->BSIM4v3lprwb = value->rValue;
            mod->BSIM4v3lprwbGiven = TRUE;
            break;                     
        case BSIM4v3_MOD_LPRWG:
            mod->BSIM4v3lprwg = value->rValue;
            mod->BSIM4v3lprwgGiven = TRUE;
            break;                     
        case BSIM4v3_MOD_LPRT:
            mod->BSIM4v3lprt = value->rValue;
            mod->BSIM4v3lprtGiven = TRUE;
            break;                     
        case BSIM4v3_MOD_LETA0:
            mod->BSIM4v3leta0 = value->rValue;
            mod->BSIM4v3leta0Given = TRUE;
            break;                 
        case BSIM4v3_MOD_LETAB:
            mod->BSIM4v3letab = value->rValue;
            mod->BSIM4v3letabGiven = TRUE;
            break;                 
        case BSIM4v3_MOD_LPCLM:
            mod->BSIM4v3lpclm = value->rValue;
            mod->BSIM4v3lpclmGiven = TRUE;
            break;                 
        case BSIM4v3_MOD_LPDIBL1:
            mod->BSIM4v3lpdibl1 = value->rValue;
            mod->BSIM4v3lpdibl1Given = TRUE;
            break;                 
        case BSIM4v3_MOD_LPDIBL2:
            mod->BSIM4v3lpdibl2 = value->rValue;
            mod->BSIM4v3lpdibl2Given = TRUE;
            break;                 
        case BSIM4v3_MOD_LPDIBLB:
            mod->BSIM4v3lpdiblb = value->rValue;
            mod->BSIM4v3lpdiblbGiven = TRUE;
            break;                 
        case BSIM4v3_MOD_LPSCBE1:
            mod->BSIM4v3lpscbe1 = value->rValue;
            mod->BSIM4v3lpscbe1Given = TRUE;
            break;                 
        case BSIM4v3_MOD_LPSCBE2:
            mod->BSIM4v3lpscbe2 = value->rValue;
            mod->BSIM4v3lpscbe2Given = TRUE;
            break;                 
        case BSIM4v3_MOD_LPVAG:
            mod->BSIM4v3lpvag = value->rValue;
            mod->BSIM4v3lpvagGiven = TRUE;
            break;                 
        case  BSIM4v3_MOD_LWR :
            mod->BSIM4v3lwr = value->rValue;
            mod->BSIM4v3lwrGiven = TRUE;
            break;
        case  BSIM4v3_MOD_LDWG :
            mod->BSIM4v3ldwg = value->rValue;
            mod->BSIM4v3ldwgGiven = TRUE;
            break;
        case  BSIM4v3_MOD_LDWB :
            mod->BSIM4v3ldwb = value->rValue;
            mod->BSIM4v3ldwbGiven = TRUE;
            break;
        case  BSIM4v3_MOD_LB0 :
            mod->BSIM4v3lb0 = value->rValue;
            mod->BSIM4v3lb0Given = TRUE;
            break;
        case  BSIM4v3_MOD_LB1 :
            mod->BSIM4v3lb1 = value->rValue;
            mod->BSIM4v3lb1Given = TRUE;
            break;
        case  BSIM4v3_MOD_LALPHA0 :
            mod->BSIM4v3lalpha0 = value->rValue;
            mod->BSIM4v3lalpha0Given = TRUE;
            break;
        case  BSIM4v3_MOD_LALPHA1 :
            mod->BSIM4v3lalpha1 = value->rValue;
            mod->BSIM4v3lalpha1Given = TRUE;
            break;
        case  BSIM4v3_MOD_LBETA0 :
            mod->BSIM4v3lbeta0 = value->rValue;
            mod->BSIM4v3lbeta0Given = TRUE;
            break;
        case  BSIM4v3_MOD_LAGIDL :
            mod->BSIM4v3lagidl = value->rValue;
            mod->BSIM4v3lagidlGiven = TRUE;
            break;
        case  BSIM4v3_MOD_LBGIDL :
            mod->BSIM4v3lbgidl = value->rValue;
            mod->BSIM4v3lbgidlGiven = TRUE;
            break;
        case  BSIM4v3_MOD_LCGIDL :
            mod->BSIM4v3lcgidl = value->rValue;
            mod->BSIM4v3lcgidlGiven = TRUE;
            break;
        case  BSIM4v3_MOD_LPHIN :
            mod->BSIM4v3lphin = value->rValue;
            mod->BSIM4v3lphinGiven = TRUE;
            break;
        case  BSIM4v3_MOD_LEGIDL :
            mod->BSIM4v3legidl = value->rValue;
            mod->BSIM4v3legidlGiven = TRUE;
            break;
        case  BSIM4v3_MOD_LAIGC :
            mod->BSIM4v3laigc = value->rValue;
            mod->BSIM4v3laigcGiven = TRUE;
            break;
        case  BSIM4v3_MOD_LBIGC :
            mod->BSIM4v3lbigc = value->rValue;
            mod->BSIM4v3lbigcGiven = TRUE;
            break;
        case  BSIM4v3_MOD_LCIGC :
            mod->BSIM4v3lcigc = value->rValue;
            mod->BSIM4v3lcigcGiven = TRUE;
            break;
        case  BSIM4v3_MOD_LAIGSD :
            mod->BSIM4v3laigsd = value->rValue;
            mod->BSIM4v3laigsdGiven = TRUE;
            break;
        case  BSIM4v3_MOD_LBIGSD :
            mod->BSIM4v3lbigsd = value->rValue;
            mod->BSIM4v3lbigsdGiven = TRUE;
            break;
        case  BSIM4v3_MOD_LCIGSD :
            mod->BSIM4v3lcigsd = value->rValue;
            mod->BSIM4v3lcigsdGiven = TRUE;
            break;
        case  BSIM4v3_MOD_LAIGBACC :
            mod->BSIM4v3laigbacc = value->rValue;
            mod->BSIM4v3laigbaccGiven = TRUE;
            break;
        case  BSIM4v3_MOD_LBIGBACC :
            mod->BSIM4v3lbigbacc = value->rValue;
            mod->BSIM4v3lbigbaccGiven = TRUE;
            break;
        case  BSIM4v3_MOD_LCIGBACC :
            mod->BSIM4v3lcigbacc = value->rValue;
            mod->BSIM4v3lcigbaccGiven = TRUE;
            break;
        case  BSIM4v3_MOD_LAIGBINV :
            mod->BSIM4v3laigbinv = value->rValue;
            mod->BSIM4v3laigbinvGiven = TRUE;
            break;
        case  BSIM4v3_MOD_LBIGBINV :
            mod->BSIM4v3lbigbinv = value->rValue;
            mod->BSIM4v3lbigbinvGiven = TRUE;
            break;
        case  BSIM4v3_MOD_LCIGBINV :
            mod->BSIM4v3lcigbinv = value->rValue;
            mod->BSIM4v3lcigbinvGiven = TRUE;
            break;
        case  BSIM4v3_MOD_LNIGC :
            mod->BSIM4v3lnigc = value->rValue;
            mod->BSIM4v3lnigcGiven = TRUE;
            break;
        case  BSIM4v3_MOD_LNIGBINV :
            mod->BSIM4v3lnigbinv = value->rValue;
            mod->BSIM4v3lnigbinvGiven = TRUE;
            break;
        case  BSIM4v3_MOD_LNIGBACC :
            mod->BSIM4v3lnigbacc = value->rValue;
            mod->BSIM4v3lnigbaccGiven = TRUE;
            break;
        case  BSIM4v3_MOD_LNTOX :
            mod->BSIM4v3lntox = value->rValue;
            mod->BSIM4v3lntoxGiven = TRUE;
            break;
        case  BSIM4v3_MOD_LEIGBINV :
            mod->BSIM4v3leigbinv = value->rValue;
            mod->BSIM4v3leigbinvGiven = TRUE;
            break;
        case  BSIM4v3_MOD_LPIGCD :
            mod->BSIM4v3lpigcd = value->rValue;
            mod->BSIM4v3lpigcdGiven = TRUE;
            break;
        case  BSIM4v3_MOD_LPOXEDGE :
            mod->BSIM4v3lpoxedge = value->rValue;
            mod->BSIM4v3lpoxedgeGiven = TRUE;
            break;
        case  BSIM4v3_MOD_LXRCRG1 :
            mod->BSIM4v3lxrcrg1 = value->rValue;
            mod->BSIM4v3lxrcrg1Given = TRUE;
            break;
        case  BSIM4v3_MOD_LXRCRG2 :
            mod->BSIM4v3lxrcrg2 = value->rValue;
            mod->BSIM4v3lxrcrg2Given = TRUE;
            break;
        case  BSIM4v3_MOD_LLAMBDA :
            mod->BSIM4v3llambda = value->rValue;
            mod->BSIM4v3llambdaGiven = TRUE;
            break;
        case  BSIM4v3_MOD_LVTL :
            mod->BSIM4v3lvtl = value->rValue;
            mod->BSIM4v3lvtlGiven = TRUE;
            break;
        case  BSIM4v3_MOD_LXN:
            mod->BSIM4v3lxn = value->rValue;
            mod->BSIM4v3lxnGiven = TRUE;
            break;
        case  BSIM4v3_MOD_LEU :
            mod->BSIM4v3leu = value->rValue;
            mod->BSIM4v3leuGiven = TRUE;
            break;
        case  BSIM4v3_MOD_LVFB :
            mod->BSIM4v3lvfb = value->rValue;
            mod->BSIM4v3lvfbGiven = TRUE;
            break;
        case  BSIM4v3_MOD_LCGSL :
            mod->BSIM4v3lcgsl = value->rValue;
            mod->BSIM4v3lcgslGiven = TRUE;
            break;
        case  BSIM4v3_MOD_LCGDL :
            mod->BSIM4v3lcgdl = value->rValue;
            mod->BSIM4v3lcgdlGiven = TRUE;
            break;
        case  BSIM4v3_MOD_LCKAPPAS :
            mod->BSIM4v3lckappas = value->rValue;
            mod->BSIM4v3lckappasGiven = TRUE;
            break;
        case  BSIM4v3_MOD_LCKAPPAD :
            mod->BSIM4v3lckappad = value->rValue;
            mod->BSIM4v3lckappadGiven = TRUE;
            break;
        case  BSIM4v3_MOD_LCF :
            mod->BSIM4v3lcf = value->rValue;
            mod->BSIM4v3lcfGiven = TRUE;
            break;
        case  BSIM4v3_MOD_LCLC :
            mod->BSIM4v3lclc = value->rValue;
            mod->BSIM4v3lclcGiven = TRUE;
            break;
        case  BSIM4v3_MOD_LCLE :
            mod->BSIM4v3lcle = value->rValue;
            mod->BSIM4v3lcleGiven = TRUE;
            break;
        case  BSIM4v3_MOD_LVFBCV :
            mod->BSIM4v3lvfbcv = value->rValue;
            mod->BSIM4v3lvfbcvGiven = TRUE;
            break;
        case  BSIM4v3_MOD_LACDE :
            mod->BSIM4v3lacde = value->rValue;
            mod->BSIM4v3lacdeGiven = TRUE;
            break;
        case  BSIM4v3_MOD_LMOIN :
            mod->BSIM4v3lmoin = value->rValue;
            mod->BSIM4v3lmoinGiven = TRUE;
            break;
        case  BSIM4v3_MOD_LNOFF :
            mod->BSIM4v3lnoff = value->rValue;
            mod->BSIM4v3lnoffGiven = TRUE;
            break;
        case  BSIM4v3_MOD_LVOFFCV :
            mod->BSIM4v3lvoffcv = value->rValue;
            mod->BSIM4v3lvoffcvGiven = TRUE;
            break;

	/* Width dependence */
        case  BSIM4v3_MOD_WCDSC :
            mod->BSIM4v3wcdsc = value->rValue;
            mod->BSIM4v3wcdscGiven = TRUE;
            break;
       
       
         case  BSIM4v3_MOD_WCDSCB :
            mod->BSIM4v3wcdscb = value->rValue;
            mod->BSIM4v3wcdscbGiven = TRUE;
            break;
         case  BSIM4v3_MOD_WCDSCD :
            mod->BSIM4v3wcdscd = value->rValue;
            mod->BSIM4v3wcdscdGiven = TRUE;
            break;
        case  BSIM4v3_MOD_WCIT :
            mod->BSIM4v3wcit = value->rValue;
            mod->BSIM4v3wcitGiven = TRUE;
            break;
        case  BSIM4v3_MOD_WNFACTOR :
            mod->BSIM4v3wnfactor = value->rValue;
            mod->BSIM4v3wnfactorGiven = TRUE;
            break;
        case BSIM4v3_MOD_WXJ:
            mod->BSIM4v3wxj = value->rValue;
            mod->BSIM4v3wxjGiven = TRUE;
            break;
        case BSIM4v3_MOD_WVSAT:
            mod->BSIM4v3wvsat = value->rValue;
            mod->BSIM4v3wvsatGiven = TRUE;
            break;


        case BSIM4v3_MOD_WA0:
            mod->BSIM4v3wa0 = value->rValue;
            mod->BSIM4v3wa0Given = TRUE;
            break;
        case BSIM4v3_MOD_WAGS:
            mod->BSIM4v3wags = value->rValue;
            mod->BSIM4v3wagsGiven = TRUE;
            break;
        case BSIM4v3_MOD_WA1:
            mod->BSIM4v3wa1 = value->rValue;
            mod->BSIM4v3wa1Given = TRUE;
            break;
        case BSIM4v3_MOD_WA2:
            mod->BSIM4v3wa2 = value->rValue;
            mod->BSIM4v3wa2Given = TRUE;
            break;
        case BSIM4v3_MOD_WAT:
            mod->BSIM4v3wat = value->rValue;
            mod->BSIM4v3watGiven = TRUE;
            break;
        case BSIM4v3_MOD_WKETA:
            mod->BSIM4v3wketa = value->rValue;
            mod->BSIM4v3wketaGiven = TRUE;
            break;    
        case BSIM4v3_MOD_WNSUB:
            mod->BSIM4v3wnsub = value->rValue;
            mod->BSIM4v3wnsubGiven = TRUE;
            break;
        case BSIM4v3_MOD_WNDEP:
            mod->BSIM4v3wndep = value->rValue;
            mod->BSIM4v3wndepGiven = TRUE;
	    if (mod->BSIM4v3wndep > 1.0e20)
		mod->BSIM4v3wndep *= 1.0e-6;
            break;
        case BSIM4v3_MOD_WNSD:
            mod->BSIM4v3wnsd = value->rValue;
            mod->BSIM4v3wnsdGiven = TRUE;
            if (mod->BSIM4v3wnsd > 1.0e23)
                mod->BSIM4v3wnsd *= 1.0e-6;
            break;
        case BSIM4v3_MOD_WNGATE:
            mod->BSIM4v3wngate = value->rValue;
            mod->BSIM4v3wngateGiven = TRUE;
	    if (mod->BSIM4v3wngate > 1.0e23)
		mod->BSIM4v3wngate *= 1.0e-6;
            break;
        case BSIM4v3_MOD_WGAMMA1:
            mod->BSIM4v3wgamma1 = value->rValue;
            mod->BSIM4v3wgamma1Given = TRUE;
            break;
        case BSIM4v3_MOD_WGAMMA2:
            mod->BSIM4v3wgamma2 = value->rValue;
            mod->BSIM4v3wgamma2Given = TRUE;
            break;
        case BSIM4v3_MOD_WVBX:
            mod->BSIM4v3wvbx = value->rValue;
            mod->BSIM4v3wvbxGiven = TRUE;
            break;
        case BSIM4v3_MOD_WVBM:
            mod->BSIM4v3wvbm = value->rValue;
            mod->BSIM4v3wvbmGiven = TRUE;
            break;
        case BSIM4v3_MOD_WXT:
            mod->BSIM4v3wxt = value->rValue;
            mod->BSIM4v3wxtGiven = TRUE;
            break;
        case  BSIM4v3_MOD_WK1:
            mod->BSIM4v3wk1 = value->rValue;
            mod->BSIM4v3wk1Given = TRUE;
            break;
        case  BSIM4v3_MOD_WKT1:
            mod->BSIM4v3wkt1 = value->rValue;
            mod->BSIM4v3wkt1Given = TRUE;
            break;
        case  BSIM4v3_MOD_WKT1L:
            mod->BSIM4v3wkt1l = value->rValue;
            mod->BSIM4v3wkt1lGiven = TRUE;
            break;
        case  BSIM4v3_MOD_WKT2:
            mod->BSIM4v3wkt2 = value->rValue;
            mod->BSIM4v3wkt2Given = TRUE;
            break;
        case  BSIM4v3_MOD_WK2:
            mod->BSIM4v3wk2 = value->rValue;
            mod->BSIM4v3wk2Given = TRUE;
            break;
        case  BSIM4v3_MOD_WK3:
            mod->BSIM4v3wk3 = value->rValue;
            mod->BSIM4v3wk3Given = TRUE;
            break;
        case  BSIM4v3_MOD_WK3B:
            mod->BSIM4v3wk3b = value->rValue;
            mod->BSIM4v3wk3bGiven = TRUE;
            break;
        case  BSIM4v3_MOD_WLPE0:
            mod->BSIM4v3wlpe0 = value->rValue;
            mod->BSIM4v3wlpe0Given = TRUE;
            break;
        case  BSIM4v3_MOD_WLPEB:
            mod->BSIM4v3wlpeb = value->rValue;
            mod->BSIM4v3wlpebGiven = TRUE;
            break;
        case  BSIM4v3_MOD_WDVTP0:
            mod->BSIM4v3wdvtp0 = value->rValue;
            mod->BSIM4v3wdvtp0Given = TRUE;
            break;
        case  BSIM4v3_MOD_WDVTP1:
            mod->BSIM4v3wdvtp1 = value->rValue;
            mod->BSIM4v3wdvtp1Given = TRUE;
            break;
        case  BSIM4v3_MOD_WW0:
            mod->BSIM4v3ww0 = value->rValue;
            mod->BSIM4v3ww0Given = TRUE;
            break;
        case  BSIM4v3_MOD_WDVT0:               
            mod->BSIM4v3wdvt0 = value->rValue;
            mod->BSIM4v3wdvt0Given = TRUE;
            break;
        case  BSIM4v3_MOD_WDVT1:             
            mod->BSIM4v3wdvt1 = value->rValue;
            mod->BSIM4v3wdvt1Given = TRUE;
            break;
        case  BSIM4v3_MOD_WDVT2:             
            mod->BSIM4v3wdvt2 = value->rValue;
            mod->BSIM4v3wdvt2Given = TRUE;
            break;
        case  BSIM4v3_MOD_WDVT0W:               
            mod->BSIM4v3wdvt0w = value->rValue;
            mod->BSIM4v3wdvt0wGiven = TRUE;
            break;
        case  BSIM4v3_MOD_WDVT1W:             
            mod->BSIM4v3wdvt1w = value->rValue;
            mod->BSIM4v3wdvt1wGiven = TRUE;
            break;
        case  BSIM4v3_MOD_WDVT2W:             
            mod->BSIM4v3wdvt2w = value->rValue;
            mod->BSIM4v3wdvt2wGiven = TRUE;
            break;
        case  BSIM4v3_MOD_WDROUT:             
            mod->BSIM4v3wdrout = value->rValue;
            mod->BSIM4v3wdroutGiven = TRUE;
            break;
        case  BSIM4v3_MOD_WDSUB:             
            mod->BSIM4v3wdsub = value->rValue;
            mod->BSIM4v3wdsubGiven = TRUE;
            break;
        case BSIM4v3_MOD_WVTH0:
            mod->BSIM4v3wvth0 = value->rValue;
            mod->BSIM4v3wvth0Given = TRUE;
            break;
        case BSIM4v3_MOD_WUA:
            mod->BSIM4v3wua = value->rValue;
            mod->BSIM4v3wuaGiven = TRUE;
            break;
        case BSIM4v3_MOD_WUA1:
            mod->BSIM4v3wua1 = value->rValue;
            mod->BSIM4v3wua1Given = TRUE;
            break;
        case BSIM4v3_MOD_WUB:
            mod->BSIM4v3wub = value->rValue;
            mod->BSIM4v3wubGiven = TRUE;
            break;
        case BSIM4v3_MOD_WUB1:
            mod->BSIM4v3wub1 = value->rValue;
            mod->BSIM4v3wub1Given = TRUE;
            break;
        case BSIM4v3_MOD_WUC:
            mod->BSIM4v3wuc = value->rValue;
            mod->BSIM4v3wucGiven = TRUE;
            break;
        case BSIM4v3_MOD_WUC1:
            mod->BSIM4v3wuc1 = value->rValue;
            mod->BSIM4v3wuc1Given = TRUE;
            break;
        case  BSIM4v3_MOD_WU0 :
            mod->BSIM4v3wu0 = value->rValue;
            mod->BSIM4v3wu0Given = TRUE;
            break;
        case  BSIM4v3_MOD_WUTE :
            mod->BSIM4v3wute = value->rValue;
            mod->BSIM4v3wuteGiven = TRUE;
            break;
        case BSIM4v3_MOD_WVOFF:
            mod->BSIM4v3wvoff = value->rValue;
            mod->BSIM4v3wvoffGiven = TRUE;
            break;
        case BSIM4v3_MOD_WMINV:
            mod->BSIM4v3wminv = value->rValue;
            mod->BSIM4v3wminvGiven = TRUE;
            break;
        case BSIM4v3_MOD_WFPROUT:
            mod->BSIM4v3wfprout = value->rValue;
            mod->BSIM4v3wfproutGiven = TRUE;
            break;
        case BSIM4v3_MOD_WPDITS:
            mod->BSIM4v3wpdits = value->rValue;
            mod->BSIM4v3wpditsGiven = TRUE;
            break;
        case BSIM4v3_MOD_WPDITSD:
            mod->BSIM4v3wpditsd = value->rValue;
            mod->BSIM4v3wpditsdGiven = TRUE;
            break;
        case  BSIM4v3_MOD_WDELTA :
            mod->BSIM4v3wdelta = value->rValue;
            mod->BSIM4v3wdeltaGiven = TRUE;
            break;
        case BSIM4v3_MOD_WRDSW:
            mod->BSIM4v3wrdsw = value->rValue;
            mod->BSIM4v3wrdswGiven = TRUE;
            break;                     
        case BSIM4v3_MOD_WRDW:
            mod->BSIM4v3wrdw = value->rValue;
            mod->BSIM4v3wrdwGiven = TRUE;
            break;
        case BSIM4v3_MOD_WRSW:
            mod->BSIM4v3wrsw = value->rValue;
            mod->BSIM4v3wrswGiven = TRUE;
            break;
        case BSIM4v3_MOD_WPRWB:
            mod->BSIM4v3wprwb = value->rValue;
            mod->BSIM4v3wprwbGiven = TRUE;
            break;                     
        case BSIM4v3_MOD_WPRWG:
            mod->BSIM4v3wprwg = value->rValue;
            mod->BSIM4v3wprwgGiven = TRUE;
            break;                     
        case BSIM4v3_MOD_WPRT:
            mod->BSIM4v3wprt = value->rValue;
            mod->BSIM4v3wprtGiven = TRUE;
            break;                     
        case BSIM4v3_MOD_WETA0:
            mod->BSIM4v3weta0 = value->rValue;
            mod->BSIM4v3weta0Given = TRUE;
            break;                 
        case BSIM4v3_MOD_WETAB:
            mod->BSIM4v3wetab = value->rValue;
            mod->BSIM4v3wetabGiven = TRUE;
            break;                 
        case BSIM4v3_MOD_WPCLM:
            mod->BSIM4v3wpclm = value->rValue;
            mod->BSIM4v3wpclmGiven = TRUE;
            break;                 
        case BSIM4v3_MOD_WPDIBL1:
            mod->BSIM4v3wpdibl1 = value->rValue;
            mod->BSIM4v3wpdibl1Given = TRUE;
            break;                 
        case BSIM4v3_MOD_WPDIBL2:
            mod->BSIM4v3wpdibl2 = value->rValue;
            mod->BSIM4v3wpdibl2Given = TRUE;
            break;                 
        case BSIM4v3_MOD_WPDIBLB:
            mod->BSIM4v3wpdiblb = value->rValue;
            mod->BSIM4v3wpdiblbGiven = TRUE;
            break;                 
        case BSIM4v3_MOD_WPSCBE1:
            mod->BSIM4v3wpscbe1 = value->rValue;
            mod->BSIM4v3wpscbe1Given = TRUE;
            break;                 
        case BSIM4v3_MOD_WPSCBE2:
            mod->BSIM4v3wpscbe2 = value->rValue;
            mod->BSIM4v3wpscbe2Given = TRUE;
            break;                 
        case BSIM4v3_MOD_WPVAG:
            mod->BSIM4v3wpvag = value->rValue;
            mod->BSIM4v3wpvagGiven = TRUE;
            break;                 
        case  BSIM4v3_MOD_WWR :
            mod->BSIM4v3wwr = value->rValue;
            mod->BSIM4v3wwrGiven = TRUE;
            break;
        case  BSIM4v3_MOD_WDWG :
            mod->BSIM4v3wdwg = value->rValue;
            mod->BSIM4v3wdwgGiven = TRUE;
            break;
        case  BSIM4v3_MOD_WDWB :
            mod->BSIM4v3wdwb = value->rValue;
            mod->BSIM4v3wdwbGiven = TRUE;
            break;
        case  BSIM4v3_MOD_WB0 :
            mod->BSIM4v3wb0 = value->rValue;
            mod->BSIM4v3wb0Given = TRUE;
            break;
        case  BSIM4v3_MOD_WB1 :
            mod->BSIM4v3wb1 = value->rValue;
            mod->BSIM4v3wb1Given = TRUE;
            break;
        case  BSIM4v3_MOD_WALPHA0 :
            mod->BSIM4v3walpha0 = value->rValue;
            mod->BSIM4v3walpha0Given = TRUE;
            break;
        case  BSIM4v3_MOD_WALPHA1 :
            mod->BSIM4v3walpha1 = value->rValue;
            mod->BSIM4v3walpha1Given = TRUE;
            break;
        case  BSIM4v3_MOD_WBETA0 :
            mod->BSIM4v3wbeta0 = value->rValue;
            mod->BSIM4v3wbeta0Given = TRUE;
            break;
        case  BSIM4v3_MOD_WAGIDL :
            mod->BSIM4v3wagidl = value->rValue;
            mod->BSIM4v3wagidlGiven = TRUE;
            break;
        case  BSIM4v3_MOD_WBGIDL :
            mod->BSIM4v3wbgidl = value->rValue;
            mod->BSIM4v3wbgidlGiven = TRUE;
            break;
        case  BSIM4v3_MOD_WCGIDL :
            mod->BSIM4v3wcgidl = value->rValue;
            mod->BSIM4v3wcgidlGiven = TRUE;
            break;
        case  BSIM4v3_MOD_WPHIN :
            mod->BSIM4v3wphin = value->rValue;
            mod->BSIM4v3wphinGiven = TRUE;
            break;
        case  BSIM4v3_MOD_WEGIDL :
            mod->BSIM4v3wegidl = value->rValue;
            mod->BSIM4v3wegidlGiven = TRUE;
            break;
        case  BSIM4v3_MOD_WAIGC :
            mod->BSIM4v3waigc = value->rValue;
            mod->BSIM4v3waigcGiven = TRUE;
            break;
        case  BSIM4v3_MOD_WBIGC :
            mod->BSIM4v3wbigc = value->rValue;
            mod->BSIM4v3wbigcGiven = TRUE;
            break;
        case  BSIM4v3_MOD_WCIGC :
            mod->BSIM4v3wcigc = value->rValue;
            mod->BSIM4v3wcigcGiven = TRUE;
            break;
        case  BSIM4v3_MOD_WAIGSD :
            mod->BSIM4v3waigsd = value->rValue;
            mod->BSIM4v3waigsdGiven = TRUE;
            break;
        case  BSIM4v3_MOD_WBIGSD :
            mod->BSIM4v3wbigsd = value->rValue;
            mod->BSIM4v3wbigsdGiven = TRUE;
            break;
        case  BSIM4v3_MOD_WCIGSD :
            mod->BSIM4v3wcigsd = value->rValue;
            mod->BSIM4v3wcigsdGiven = TRUE;
            break;
        case  BSIM4v3_MOD_WAIGBACC :
            mod->BSIM4v3waigbacc = value->rValue;
            mod->BSIM4v3waigbaccGiven = TRUE;
            break;
        case  BSIM4v3_MOD_WBIGBACC :
            mod->BSIM4v3wbigbacc = value->rValue;
            mod->BSIM4v3wbigbaccGiven = TRUE;
            break;
        case  BSIM4v3_MOD_WCIGBACC :
            mod->BSIM4v3wcigbacc = value->rValue;
            mod->BSIM4v3wcigbaccGiven = TRUE;
            break;
        case  BSIM4v3_MOD_WAIGBINV :
            mod->BSIM4v3waigbinv = value->rValue;
            mod->BSIM4v3waigbinvGiven = TRUE;
            break;
        case  BSIM4v3_MOD_WBIGBINV :
            mod->BSIM4v3wbigbinv = value->rValue;
            mod->BSIM4v3wbigbinvGiven = TRUE;
            break;
        case  BSIM4v3_MOD_WCIGBINV :
            mod->BSIM4v3wcigbinv = value->rValue;
            mod->BSIM4v3wcigbinvGiven = TRUE;
            break;
        case  BSIM4v3_MOD_WNIGC :
            mod->BSIM4v3wnigc = value->rValue;
            mod->BSIM4v3wnigcGiven = TRUE;
            break;
        case  BSIM4v3_MOD_WNIGBINV :
            mod->BSIM4v3wnigbinv = value->rValue;
            mod->BSIM4v3wnigbinvGiven = TRUE;
            break;
        case  BSIM4v3_MOD_WNIGBACC :
            mod->BSIM4v3wnigbacc = value->rValue;
            mod->BSIM4v3wnigbaccGiven = TRUE;
            break;
        case  BSIM4v3_MOD_WNTOX :
            mod->BSIM4v3wntox = value->rValue;
            mod->BSIM4v3wntoxGiven = TRUE;
            break;
        case  BSIM4v3_MOD_WEIGBINV :
            mod->BSIM4v3weigbinv = value->rValue;
            mod->BSIM4v3weigbinvGiven = TRUE;
            break;
        case  BSIM4v3_MOD_WPIGCD :
            mod->BSIM4v3wpigcd = value->rValue;
            mod->BSIM4v3wpigcdGiven = TRUE;
            break;
        case  BSIM4v3_MOD_WPOXEDGE :
            mod->BSIM4v3wpoxedge = value->rValue;
            mod->BSIM4v3wpoxedgeGiven = TRUE;
            break;
        case  BSIM4v3_MOD_WXRCRG1 :
            mod->BSIM4v3wxrcrg1 = value->rValue;
            mod->BSIM4v3wxrcrg1Given = TRUE;
            break;
        case  BSIM4v3_MOD_WXRCRG2 :
            mod->BSIM4v3wxrcrg2 = value->rValue;
            mod->BSIM4v3wxrcrg2Given = TRUE;
            break;
        case  BSIM4v3_MOD_WLAMBDA :
            mod->BSIM4v3wlambda = value->rValue;
            mod->BSIM4v3wlambdaGiven = TRUE;
            break;
        case  BSIM4v3_MOD_WVTL :
            mod->BSIM4v3wvtl = value->rValue;
            mod->BSIM4v3wvtlGiven = TRUE;
            break;
        case  BSIM4v3_MOD_WXN:
            mod->BSIM4v3wxn = value->rValue;
            mod->BSIM4v3wxnGiven = TRUE;
            break;
        case  BSIM4v3_MOD_WEU :
            mod->BSIM4v3weu = value->rValue;
            mod->BSIM4v3weuGiven = TRUE;
            break;
        case  BSIM4v3_MOD_WVFB :
            mod->BSIM4v3wvfb = value->rValue;
            mod->BSIM4v3wvfbGiven = TRUE;
            break;
        case  BSIM4v3_MOD_WCGSL :
            mod->BSIM4v3wcgsl = value->rValue;
            mod->BSIM4v3wcgslGiven = TRUE;
            break;
        case  BSIM4v3_MOD_WCGDL :
            mod->BSIM4v3wcgdl = value->rValue;
            mod->BSIM4v3wcgdlGiven = TRUE;
            break;
        case  BSIM4v3_MOD_WCKAPPAS :
            mod->BSIM4v3wckappas = value->rValue;
            mod->BSIM4v3wckappasGiven = TRUE;
            break;
        case  BSIM4v3_MOD_WCKAPPAD :
            mod->BSIM4v3wckappad = value->rValue;
            mod->BSIM4v3wckappadGiven = TRUE;
            break;
        case  BSIM4v3_MOD_WCF :
            mod->BSIM4v3wcf = value->rValue;
            mod->BSIM4v3wcfGiven = TRUE;
            break;
        case  BSIM4v3_MOD_WCLC :
            mod->BSIM4v3wclc = value->rValue;
            mod->BSIM4v3wclcGiven = TRUE;
            break;
        case  BSIM4v3_MOD_WCLE :
            mod->BSIM4v3wcle = value->rValue;
            mod->BSIM4v3wcleGiven = TRUE;
            break;
        case  BSIM4v3_MOD_WVFBCV :
            mod->BSIM4v3wvfbcv = value->rValue;
            mod->BSIM4v3wvfbcvGiven = TRUE;
            break;
        case  BSIM4v3_MOD_WACDE :
            mod->BSIM4v3wacde = value->rValue;
            mod->BSIM4v3wacdeGiven = TRUE;
            break;
        case  BSIM4v3_MOD_WMOIN :
            mod->BSIM4v3wmoin = value->rValue;
            mod->BSIM4v3wmoinGiven = TRUE;
            break;
        case  BSIM4v3_MOD_WNOFF :
            mod->BSIM4v3wnoff = value->rValue;
            mod->BSIM4v3wnoffGiven = TRUE;
            break;
        case  BSIM4v3_MOD_WVOFFCV :
            mod->BSIM4v3wvoffcv = value->rValue;
            mod->BSIM4v3wvoffcvGiven = TRUE;
            break;

	/* Cross-term dependence */
        case  BSIM4v3_MOD_PCDSC :
            mod->BSIM4v3pcdsc = value->rValue;
            mod->BSIM4v3pcdscGiven = TRUE;
            break;


        case  BSIM4v3_MOD_PCDSCB :
            mod->BSIM4v3pcdscb = value->rValue;
            mod->BSIM4v3pcdscbGiven = TRUE;
            break;
        case  BSIM4v3_MOD_PCDSCD :
            mod->BSIM4v3pcdscd = value->rValue;
            mod->BSIM4v3pcdscdGiven = TRUE;
            break;
        case  BSIM4v3_MOD_PCIT :
            mod->BSIM4v3pcit = value->rValue;
            mod->BSIM4v3pcitGiven = TRUE;
            break;
        case  BSIM4v3_MOD_PNFACTOR :
            mod->BSIM4v3pnfactor = value->rValue;
            mod->BSIM4v3pnfactorGiven = TRUE;
            break;
        case BSIM4v3_MOD_PXJ:
            mod->BSIM4v3pxj = value->rValue;
            mod->BSIM4v3pxjGiven = TRUE;
            break;
        case BSIM4v3_MOD_PVSAT:
            mod->BSIM4v3pvsat = value->rValue;
            mod->BSIM4v3pvsatGiven = TRUE;
            break;


        case BSIM4v3_MOD_PA0:
            mod->BSIM4v3pa0 = value->rValue;
            mod->BSIM4v3pa0Given = TRUE;
            break;
        case BSIM4v3_MOD_PAGS:
            mod->BSIM4v3pags = value->rValue;
            mod->BSIM4v3pagsGiven = TRUE;
            break;
        case BSIM4v3_MOD_PA1:
            mod->BSIM4v3pa1 = value->rValue;
            mod->BSIM4v3pa1Given = TRUE;
            break;
        case BSIM4v3_MOD_PA2:
            mod->BSIM4v3pa2 = value->rValue;
            mod->BSIM4v3pa2Given = TRUE;
            break;
        case BSIM4v3_MOD_PAT:
            mod->BSIM4v3pat = value->rValue;
            mod->BSIM4v3patGiven = TRUE;
            break;
        case BSIM4v3_MOD_PKETA:
            mod->BSIM4v3pketa = value->rValue;
            mod->BSIM4v3pketaGiven = TRUE;
            break;    
        case BSIM4v3_MOD_PNSUB:
            mod->BSIM4v3pnsub = value->rValue;
            mod->BSIM4v3pnsubGiven = TRUE;
            break;
        case BSIM4v3_MOD_PNDEP:
            mod->BSIM4v3pndep = value->rValue;
            mod->BSIM4v3pndepGiven = TRUE;
	    if (mod->BSIM4v3pndep > 1.0e20)
		mod->BSIM4v3pndep *= 1.0e-6;
            break;
        case BSIM4v3_MOD_PNSD:
            mod->BSIM4v3pnsd = value->rValue;
            mod->BSIM4v3pnsdGiven = TRUE;
            if (mod->BSIM4v3pnsd > 1.0e23)
                mod->BSIM4v3pnsd *= 1.0e-6;
            break;
        case BSIM4v3_MOD_PNGATE:
            mod->BSIM4v3pngate = value->rValue;
            mod->BSIM4v3pngateGiven = TRUE;
	    if (mod->BSIM4v3pngate > 1.0e23)
		mod->BSIM4v3pngate *= 1.0e-6;
            break;
        case BSIM4v3_MOD_PGAMMA1:
            mod->BSIM4v3pgamma1 = value->rValue;
            mod->BSIM4v3pgamma1Given = TRUE;
            break;
        case BSIM4v3_MOD_PGAMMA2:
            mod->BSIM4v3pgamma2 = value->rValue;
            mod->BSIM4v3pgamma2Given = TRUE;
            break;
        case BSIM4v3_MOD_PVBX:
            mod->BSIM4v3pvbx = value->rValue;
            mod->BSIM4v3pvbxGiven = TRUE;
            break;
        case BSIM4v3_MOD_PVBM:
            mod->BSIM4v3pvbm = value->rValue;
            mod->BSIM4v3pvbmGiven = TRUE;
            break;
        case BSIM4v3_MOD_PXT:
            mod->BSIM4v3pxt = value->rValue;
            mod->BSIM4v3pxtGiven = TRUE;
            break;
        case  BSIM4v3_MOD_PK1:
            mod->BSIM4v3pk1 = value->rValue;
            mod->BSIM4v3pk1Given = TRUE;
            break;
        case  BSIM4v3_MOD_PKT1:
            mod->BSIM4v3pkt1 = value->rValue;
            mod->BSIM4v3pkt1Given = TRUE;
            break;
        case  BSIM4v3_MOD_PKT1L:
            mod->BSIM4v3pkt1l = value->rValue;
            mod->BSIM4v3pkt1lGiven = TRUE;
            break;
        case  BSIM4v3_MOD_PKT2:
            mod->BSIM4v3pkt2 = value->rValue;
            mod->BSIM4v3pkt2Given = TRUE;
            break;
        case  BSIM4v3_MOD_PK2:
            mod->BSIM4v3pk2 = value->rValue;
            mod->BSIM4v3pk2Given = TRUE;
            break;
        case  BSIM4v3_MOD_PK3:
            mod->BSIM4v3pk3 = value->rValue;
            mod->BSIM4v3pk3Given = TRUE;
            break;
        case  BSIM4v3_MOD_PK3B:
            mod->BSIM4v3pk3b = value->rValue;
            mod->BSIM4v3pk3bGiven = TRUE;
            break;
        case  BSIM4v3_MOD_PLPE0:
            mod->BSIM4v3plpe0 = value->rValue;
            mod->BSIM4v3plpe0Given = TRUE;
            break;
        case  BSIM4v3_MOD_PLPEB:
            mod->BSIM4v3plpeb = value->rValue;
            mod->BSIM4v3plpebGiven = TRUE;
            break;
        case  BSIM4v3_MOD_PDVTP0:
            mod->BSIM4v3pdvtp0 = value->rValue;
            mod->BSIM4v3pdvtp0Given = TRUE;
            break;
        case  BSIM4v3_MOD_PDVTP1:
            mod->BSIM4v3pdvtp1 = value->rValue;
            mod->BSIM4v3pdvtp1Given = TRUE;
            break;
        case  BSIM4v3_MOD_PW0:
            mod->BSIM4v3pw0 = value->rValue;
            mod->BSIM4v3pw0Given = TRUE;
            break;
        case  BSIM4v3_MOD_PDVT0:               
            mod->BSIM4v3pdvt0 = value->rValue;
            mod->BSIM4v3pdvt0Given = TRUE;
            break;
        case  BSIM4v3_MOD_PDVT1:             
            mod->BSIM4v3pdvt1 = value->rValue;
            mod->BSIM4v3pdvt1Given = TRUE;
            break;
        case  BSIM4v3_MOD_PDVT2:             
            mod->BSIM4v3pdvt2 = value->rValue;
            mod->BSIM4v3pdvt2Given = TRUE;
            break;
        case  BSIM4v3_MOD_PDVT0W:               
            mod->BSIM4v3pdvt0w = value->rValue;
            mod->BSIM4v3pdvt0wGiven = TRUE;
            break;
        case  BSIM4v3_MOD_PDVT1W:             
            mod->BSIM4v3pdvt1w = value->rValue;
            mod->BSIM4v3pdvt1wGiven = TRUE;
            break;
        case  BSIM4v3_MOD_PDVT2W:             
            mod->BSIM4v3pdvt2w = value->rValue;
            mod->BSIM4v3pdvt2wGiven = TRUE;
            break;
        case  BSIM4v3_MOD_PDROUT:             
            mod->BSIM4v3pdrout = value->rValue;
            mod->BSIM4v3pdroutGiven = TRUE;
            break;
        case  BSIM4v3_MOD_PDSUB:             
            mod->BSIM4v3pdsub = value->rValue;
            mod->BSIM4v3pdsubGiven = TRUE;
            break;
        case BSIM4v3_MOD_PVTH0:
            mod->BSIM4v3pvth0 = value->rValue;
            mod->BSIM4v3pvth0Given = TRUE;
            break;
        case BSIM4v3_MOD_PUA:
            mod->BSIM4v3pua = value->rValue;
            mod->BSIM4v3puaGiven = TRUE;
            break;
        case BSIM4v3_MOD_PUA1:
            mod->BSIM4v3pua1 = value->rValue;
            mod->BSIM4v3pua1Given = TRUE;
            break;
        case BSIM4v3_MOD_PUB:
            mod->BSIM4v3pub = value->rValue;
            mod->BSIM4v3pubGiven = TRUE;
            break;
        case BSIM4v3_MOD_PUB1:
            mod->BSIM4v3pub1 = value->rValue;
            mod->BSIM4v3pub1Given = TRUE;
            break;
        case BSIM4v3_MOD_PUC:
            mod->BSIM4v3puc = value->rValue;
            mod->BSIM4v3pucGiven = TRUE;
            break;
        case BSIM4v3_MOD_PUC1:
            mod->BSIM4v3puc1 = value->rValue;
            mod->BSIM4v3puc1Given = TRUE;
            break;
        case  BSIM4v3_MOD_PU0 :
            mod->BSIM4v3pu0 = value->rValue;
            mod->BSIM4v3pu0Given = TRUE;
            break;
        case  BSIM4v3_MOD_PUTE :
            mod->BSIM4v3pute = value->rValue;
            mod->BSIM4v3puteGiven = TRUE;
            break;
        case BSIM4v3_MOD_PVOFF:
            mod->BSIM4v3pvoff = value->rValue;
            mod->BSIM4v3pvoffGiven = TRUE;
            break;
        case BSIM4v3_MOD_PMINV:
            mod->BSIM4v3pminv = value->rValue;
            mod->BSIM4v3pminvGiven = TRUE;
            break;
        case BSIM4v3_MOD_PFPROUT:
            mod->BSIM4v3pfprout = value->rValue;
            mod->BSIM4v3pfproutGiven = TRUE;
            break;
        case BSIM4v3_MOD_PPDITS:
            mod->BSIM4v3ppdits = value->rValue;
            mod->BSIM4v3ppditsGiven = TRUE;
            break;
        case BSIM4v3_MOD_PPDITSD:
            mod->BSIM4v3ppditsd = value->rValue;
            mod->BSIM4v3ppditsdGiven = TRUE;
            break;
        case  BSIM4v3_MOD_PDELTA :
            mod->BSIM4v3pdelta = value->rValue;
            mod->BSIM4v3pdeltaGiven = TRUE;
            break;
        case BSIM4v3_MOD_PRDSW:
            mod->BSIM4v3prdsw = value->rValue;
            mod->BSIM4v3prdswGiven = TRUE;
            break;                     
        case BSIM4v3_MOD_PRDW:
            mod->BSIM4v3prdw = value->rValue;
            mod->BSIM4v3prdwGiven = TRUE;
            break;
        case BSIM4v3_MOD_PRSW:
            mod->BSIM4v3prsw = value->rValue;
            mod->BSIM4v3prswGiven = TRUE;
            break;
        case BSIM4v3_MOD_PPRWB:
            mod->BSIM4v3pprwb = value->rValue;
            mod->BSIM4v3pprwbGiven = TRUE;
            break;                     
        case BSIM4v3_MOD_PPRWG:
            mod->BSIM4v3pprwg = value->rValue;
            mod->BSIM4v3pprwgGiven = TRUE;
            break;                     
        case BSIM4v3_MOD_PPRT:
            mod->BSIM4v3pprt = value->rValue;
            mod->BSIM4v3pprtGiven = TRUE;
            break;                     
        case BSIM4v3_MOD_PETA0:
            mod->BSIM4v3peta0 = value->rValue;
            mod->BSIM4v3peta0Given = TRUE;
            break;                 
        case BSIM4v3_MOD_PETAB:
            mod->BSIM4v3petab = value->rValue;
            mod->BSIM4v3petabGiven = TRUE;
            break;                 
        case BSIM4v3_MOD_PPCLM:
            mod->BSIM4v3ppclm = value->rValue;
            mod->BSIM4v3ppclmGiven = TRUE;
            break;                 
        case BSIM4v3_MOD_PPDIBL1:
            mod->BSIM4v3ppdibl1 = value->rValue;
            mod->BSIM4v3ppdibl1Given = TRUE;
            break;                 
        case BSIM4v3_MOD_PPDIBL2:
            mod->BSIM4v3ppdibl2 = value->rValue;
            mod->BSIM4v3ppdibl2Given = TRUE;
            break;                 
        case BSIM4v3_MOD_PPDIBLB:
            mod->BSIM4v3ppdiblb = value->rValue;
            mod->BSIM4v3ppdiblbGiven = TRUE;
            break;                 
        case BSIM4v3_MOD_PPSCBE1:
            mod->BSIM4v3ppscbe1 = value->rValue;
            mod->BSIM4v3ppscbe1Given = TRUE;
            break;                 
        case BSIM4v3_MOD_PPSCBE2:
            mod->BSIM4v3ppscbe2 = value->rValue;
            mod->BSIM4v3ppscbe2Given = TRUE;
            break;                 
        case BSIM4v3_MOD_PPVAG:
            mod->BSIM4v3ppvag = value->rValue;
            mod->BSIM4v3ppvagGiven = TRUE;
            break;                 
        case  BSIM4v3_MOD_PWR :
            mod->BSIM4v3pwr = value->rValue;
            mod->BSIM4v3pwrGiven = TRUE;
            break;
        case  BSIM4v3_MOD_PDWG :
            mod->BSIM4v3pdwg = value->rValue;
            mod->BSIM4v3pdwgGiven = TRUE;
            break;
        case  BSIM4v3_MOD_PDWB :
            mod->BSIM4v3pdwb = value->rValue;
            mod->BSIM4v3pdwbGiven = TRUE;
            break;
        case  BSIM4v3_MOD_PB0 :
            mod->BSIM4v3pb0 = value->rValue;
            mod->BSIM4v3pb0Given = TRUE;
            break;
        case  BSIM4v3_MOD_PB1 :
            mod->BSIM4v3pb1 = value->rValue;
            mod->BSIM4v3pb1Given = TRUE;
            break;
        case  BSIM4v3_MOD_PALPHA0 :
            mod->BSIM4v3palpha0 = value->rValue;
            mod->BSIM4v3palpha0Given = TRUE;
            break;
        case  BSIM4v3_MOD_PALPHA1 :
            mod->BSIM4v3palpha1 = value->rValue;
            mod->BSIM4v3palpha1Given = TRUE;
            break;
        case  BSIM4v3_MOD_PBETA0 :
            mod->BSIM4v3pbeta0 = value->rValue;
            mod->BSIM4v3pbeta0Given = TRUE;
            break;
        case  BSIM4v3_MOD_PAGIDL :
            mod->BSIM4v3pagidl = value->rValue;
            mod->BSIM4v3pagidlGiven = TRUE;
            break;
        case  BSIM4v3_MOD_PBGIDL :
            mod->BSIM4v3pbgidl = value->rValue;
            mod->BSIM4v3pbgidlGiven = TRUE;
            break;
        case  BSIM4v3_MOD_PCGIDL :
            mod->BSIM4v3pcgidl = value->rValue;
            mod->BSIM4v3pcgidlGiven = TRUE;
            break;
        case  BSIM4v3_MOD_PPHIN :
            mod->BSIM4v3pphin = value->rValue;
            mod->BSIM4v3pphinGiven = TRUE;
            break;
        case  BSIM4v3_MOD_PEGIDL :
            mod->BSIM4v3pegidl = value->rValue;
            mod->BSIM4v3pegidlGiven = TRUE;
            break;
        case  BSIM4v3_MOD_PAIGC :
            mod->BSIM4v3paigc = value->rValue;
            mod->BSIM4v3paigcGiven = TRUE;
            break;
        case  BSIM4v3_MOD_PBIGC :
            mod->BSIM4v3pbigc = value->rValue;
            mod->BSIM4v3pbigcGiven = TRUE;
            break;
        case  BSIM4v3_MOD_PCIGC :
            mod->BSIM4v3pcigc = value->rValue;
            mod->BSIM4v3pcigcGiven = TRUE;
            break;
        case  BSIM4v3_MOD_PAIGSD :
            mod->BSIM4v3paigsd = value->rValue;
            mod->BSIM4v3paigsdGiven = TRUE;
            break;
        case  BSIM4v3_MOD_PBIGSD :
            mod->BSIM4v3pbigsd = value->rValue;
            mod->BSIM4v3pbigsdGiven = TRUE;
            break;
        case  BSIM4v3_MOD_PCIGSD :
            mod->BSIM4v3pcigsd = value->rValue;
            mod->BSIM4v3pcigsdGiven = TRUE;
            break;
        case  BSIM4v3_MOD_PAIGBACC :
            mod->BSIM4v3paigbacc = value->rValue;
            mod->BSIM4v3paigbaccGiven = TRUE;
            break;
        case  BSIM4v3_MOD_PBIGBACC :
            mod->BSIM4v3pbigbacc = value->rValue;
            mod->BSIM4v3pbigbaccGiven = TRUE;
            break;
        case  BSIM4v3_MOD_PCIGBACC :
            mod->BSIM4v3pcigbacc = value->rValue;
            mod->BSIM4v3pcigbaccGiven = TRUE;
            break;
        case  BSIM4v3_MOD_PAIGBINV :
            mod->BSIM4v3paigbinv = value->rValue;
            mod->BSIM4v3paigbinvGiven = TRUE;
            break;
        case  BSIM4v3_MOD_PBIGBINV :
            mod->BSIM4v3pbigbinv = value->rValue;
            mod->BSIM4v3pbigbinvGiven = TRUE;
            break;
        case  BSIM4v3_MOD_PCIGBINV :
            mod->BSIM4v3pcigbinv = value->rValue;
            mod->BSIM4v3pcigbinvGiven = TRUE;
            break;
        case  BSIM4v3_MOD_PNIGC :
            mod->BSIM4v3pnigc = value->rValue;
            mod->BSIM4v3pnigcGiven = TRUE;
            break;
        case  BSIM4v3_MOD_PNIGBINV :
            mod->BSIM4v3pnigbinv = value->rValue;
            mod->BSIM4v3pnigbinvGiven = TRUE;
            break;
        case  BSIM4v3_MOD_PNIGBACC :
            mod->BSIM4v3pnigbacc = value->rValue;
            mod->BSIM4v3pnigbaccGiven = TRUE;
            break;
        case  BSIM4v3_MOD_PNTOX :
            mod->BSIM4v3pntox = value->rValue;
            mod->BSIM4v3pntoxGiven = TRUE;
            break;
        case  BSIM4v3_MOD_PEIGBINV :
            mod->BSIM4v3peigbinv = value->rValue;
            mod->BSIM4v3peigbinvGiven = TRUE;
            break;
        case  BSIM4v3_MOD_PPIGCD :
            mod->BSIM4v3ppigcd = value->rValue;
            mod->BSIM4v3ppigcdGiven = TRUE;
            break;
        case  BSIM4v3_MOD_PPOXEDGE :
            mod->BSIM4v3ppoxedge = value->rValue;
            mod->BSIM4v3ppoxedgeGiven = TRUE;
            break;
        case  BSIM4v3_MOD_PXRCRG1 :
            mod->BSIM4v3pxrcrg1 = value->rValue;
            mod->BSIM4v3pxrcrg1Given = TRUE;
            break;
        case  BSIM4v3_MOD_PXRCRG2 :
            mod->BSIM4v3pxrcrg2 = value->rValue;
            mod->BSIM4v3pxrcrg2Given = TRUE;
            break;
        case  BSIM4v3_MOD_PLAMBDA :
            mod->BSIM4v3plambda = value->rValue;
            mod->BSIM4v3plambdaGiven = TRUE;
            break;
        case  BSIM4v3_MOD_PVTL :
            mod->BSIM4v3pvtl = value->rValue;
            mod->BSIM4v3pvtlGiven = TRUE;
            break;
        case  BSIM4v3_MOD_PXN:
            mod->BSIM4v3pxn = value->rValue;
            mod->BSIM4v3pxnGiven = TRUE;
            break;
        case  BSIM4v3_MOD_PEU :
            mod->BSIM4v3peu = value->rValue;
            mod->BSIM4v3peuGiven = TRUE;
            break;
        case  BSIM4v3_MOD_PVFB :
            mod->BSIM4v3pvfb = value->rValue;
            mod->BSIM4v3pvfbGiven = TRUE;
            break;
        case  BSIM4v3_MOD_PCGSL :
            mod->BSIM4v3pcgsl = value->rValue;
            mod->BSIM4v3pcgslGiven = TRUE;
            break;
        case  BSIM4v3_MOD_PCGDL :
            mod->BSIM4v3pcgdl = value->rValue;
            mod->BSIM4v3pcgdlGiven = TRUE;
            break;
        case  BSIM4v3_MOD_PCKAPPAS :
            mod->BSIM4v3pckappas = value->rValue;
            mod->BSIM4v3pckappasGiven = TRUE;
            break;
        case  BSIM4v3_MOD_PCKAPPAD :
            mod->BSIM4v3pckappad = value->rValue;
            mod->BSIM4v3pckappadGiven = TRUE;
            break;
        case  BSIM4v3_MOD_PCF :
            mod->BSIM4v3pcf = value->rValue;
            mod->BSIM4v3pcfGiven = TRUE;
            break;
        case  BSIM4v3_MOD_PCLC :
            mod->BSIM4v3pclc = value->rValue;
            mod->BSIM4v3pclcGiven = TRUE;
            break;
        case  BSIM4v3_MOD_PCLE :
            mod->BSIM4v3pcle = value->rValue;
            mod->BSIM4v3pcleGiven = TRUE;
            break;
        case  BSIM4v3_MOD_PVFBCV :
            mod->BSIM4v3pvfbcv = value->rValue;
            mod->BSIM4v3pvfbcvGiven = TRUE;
            break;
        case  BSIM4v3_MOD_PACDE :
            mod->BSIM4v3pacde = value->rValue;
            mod->BSIM4v3pacdeGiven = TRUE;
            break;
        case  BSIM4v3_MOD_PMOIN :
            mod->BSIM4v3pmoin = value->rValue;
            mod->BSIM4v3pmoinGiven = TRUE;
            break;
        case  BSIM4v3_MOD_PNOFF :
            mod->BSIM4v3pnoff = value->rValue;
            mod->BSIM4v3pnoffGiven = TRUE;
            break;
        case  BSIM4v3_MOD_PVOFFCV :
            mod->BSIM4v3pvoffcv = value->rValue;
            mod->BSIM4v3pvoffcvGiven = TRUE;
            break;

        case  BSIM4v3_MOD_TNOM :
            mod->BSIM4v3tnom = value->rValue + CONSTCtoK;
            mod->BSIM4v3tnomGiven = TRUE;
            break;
        case  BSIM4v3_MOD_CGSO :
            mod->BSIM4v3cgso = value->rValue;
            mod->BSIM4v3cgsoGiven = TRUE;
            break;
        case  BSIM4v3_MOD_CGDO :
            mod->BSIM4v3cgdo = value->rValue;
            mod->BSIM4v3cgdoGiven = TRUE;
            break;
        case  BSIM4v3_MOD_CGBO :
            mod->BSIM4v3cgbo = value->rValue;
            mod->BSIM4v3cgboGiven = TRUE;
            break;
        case  BSIM4v3_MOD_XPART :
            mod->BSIM4v3xpart = value->rValue;
            mod->BSIM4v3xpartGiven = TRUE;
            break;
        case  BSIM4v3_MOD_RSH :
            mod->BSIM4v3sheetResistance = value->rValue;
            mod->BSIM4v3sheetResistanceGiven = TRUE;
            break;
        case  BSIM4v3_MOD_JSS :
            mod->BSIM4v3SjctSatCurDensity = value->rValue;
            mod->BSIM4v3SjctSatCurDensityGiven = TRUE;
            break;
        case  BSIM4v3_MOD_JSWS :
            mod->BSIM4v3SjctSidewallSatCurDensity = value->rValue;
            mod->BSIM4v3SjctSidewallSatCurDensityGiven = TRUE;
            break;
        case  BSIM4v3_MOD_JSWGS :
            mod->BSIM4v3SjctGateSidewallSatCurDensity = value->rValue;
            mod->BSIM4v3SjctGateSidewallSatCurDensityGiven = TRUE;
            break;
        case  BSIM4v3_MOD_PBS :
            mod->BSIM4v3SbulkJctPotential = value->rValue;
            mod->BSIM4v3SbulkJctPotentialGiven = TRUE;
            break;
        case  BSIM4v3_MOD_MJS :
            mod->BSIM4v3SbulkJctBotGradingCoeff = value->rValue;
            mod->BSIM4v3SbulkJctBotGradingCoeffGiven = TRUE;
            break;
        case  BSIM4v3_MOD_PBSWS :
            mod->BSIM4v3SsidewallJctPotential = value->rValue;
            mod->BSIM4v3SsidewallJctPotentialGiven = TRUE;
            break;
        case  BSIM4v3_MOD_MJSWS :
            mod->BSIM4v3SbulkJctSideGradingCoeff = value->rValue;
            mod->BSIM4v3SbulkJctSideGradingCoeffGiven = TRUE;
            break;
        case  BSIM4v3_MOD_CJS :
            mod->BSIM4v3SunitAreaJctCap = value->rValue;
            mod->BSIM4v3SunitAreaJctCapGiven = TRUE;
            break;
        case  BSIM4v3_MOD_CJSWS :
            mod->BSIM4v3SunitLengthSidewallJctCap = value->rValue;
            mod->BSIM4v3SunitLengthSidewallJctCapGiven = TRUE;
            break;
        case  BSIM4v3_MOD_NJS :
            mod->BSIM4v3SjctEmissionCoeff = value->rValue;
            mod->BSIM4v3SjctEmissionCoeffGiven = TRUE;
            break;
        case  BSIM4v3_MOD_PBSWGS :
            mod->BSIM4v3SGatesidewallJctPotential = value->rValue;
            mod->BSIM4v3SGatesidewallJctPotentialGiven = TRUE;
            break;
        case  BSIM4v3_MOD_MJSWGS :
            mod->BSIM4v3SbulkJctGateSideGradingCoeff = value->rValue;
            mod->BSIM4v3SbulkJctGateSideGradingCoeffGiven = TRUE;
            break;
        case  BSIM4v3_MOD_CJSWGS :
            mod->BSIM4v3SunitLengthGateSidewallJctCap = value->rValue;
            mod->BSIM4v3SunitLengthGateSidewallJctCapGiven = TRUE;
            break;
        case  BSIM4v3_MOD_XTIS :
            mod->BSIM4v3SjctTempExponent = value->rValue;
            mod->BSIM4v3SjctTempExponentGiven = TRUE;
            break;
        case  BSIM4v3_MOD_JSD :
            mod->BSIM4v3DjctSatCurDensity = value->rValue;
            mod->BSIM4v3DjctSatCurDensityGiven = TRUE;
            break;
        case  BSIM4v3_MOD_JSWD :
            mod->BSIM4v3DjctSidewallSatCurDensity = value->rValue;
            mod->BSIM4v3DjctSidewallSatCurDensityGiven = TRUE;
            break;
        case  BSIM4v3_MOD_JSWGD :
            mod->BSIM4v3DjctGateSidewallSatCurDensity = value->rValue;
            mod->BSIM4v3DjctGateSidewallSatCurDensityGiven = TRUE;
            break;
        case  BSIM4v3_MOD_PBD :
            mod->BSIM4v3DbulkJctPotential = value->rValue;
            mod->BSIM4v3DbulkJctPotentialGiven = TRUE;
            break;
        case  BSIM4v3_MOD_MJD :
            mod->BSIM4v3DbulkJctBotGradingCoeff = value->rValue;
            mod->BSIM4v3DbulkJctBotGradingCoeffGiven = TRUE;
            break;
        case  BSIM4v3_MOD_PBSWD :
            mod->BSIM4v3DsidewallJctPotential = value->rValue;
            mod->BSIM4v3DsidewallJctPotentialGiven = TRUE;
            break;
        case  BSIM4v3_MOD_MJSWD :
            mod->BSIM4v3DbulkJctSideGradingCoeff = value->rValue;
            mod->BSIM4v3DbulkJctSideGradingCoeffGiven = TRUE;
            break;
        case  BSIM4v3_MOD_CJD :
            mod->BSIM4v3DunitAreaJctCap = value->rValue;
            mod->BSIM4v3DunitAreaJctCapGiven = TRUE;
            break;
        case  BSIM4v3_MOD_CJSWD :
            mod->BSIM4v3DunitLengthSidewallJctCap = value->rValue;
            mod->BSIM4v3DunitLengthSidewallJctCapGiven = TRUE;
            break;
        case  BSIM4v3_MOD_NJD :
            mod->BSIM4v3DjctEmissionCoeff = value->rValue;
            mod->BSIM4v3DjctEmissionCoeffGiven = TRUE;
            break;
        case  BSIM4v3_MOD_PBSWGD :
            mod->BSIM4v3DGatesidewallJctPotential = value->rValue;
            mod->BSIM4v3DGatesidewallJctPotentialGiven = TRUE;
            break;
        case  BSIM4v3_MOD_MJSWGD :
            mod->BSIM4v3DbulkJctGateSideGradingCoeff = value->rValue;
            mod->BSIM4v3DbulkJctGateSideGradingCoeffGiven = TRUE;
            break;
        case  BSIM4v3_MOD_CJSWGD :
            mod->BSIM4v3DunitLengthGateSidewallJctCap = value->rValue;
            mod->BSIM4v3DunitLengthGateSidewallJctCapGiven = TRUE;
            break;
        case  BSIM4v3_MOD_XTID :
            mod->BSIM4v3DjctTempExponent = value->rValue;
            mod->BSIM4v3DjctTempExponentGiven = TRUE;
            break;
        case  BSIM4v3_MOD_LINT :
            mod->BSIM4v3Lint = value->rValue;
            mod->BSIM4v3LintGiven = TRUE;
            break;
        case  BSIM4v3_MOD_LL :
            mod->BSIM4v3Ll = value->rValue;
            mod->BSIM4v3LlGiven = TRUE;
            break;
        case  BSIM4v3_MOD_LLC :
            mod->BSIM4v3Llc = value->rValue;
            mod->BSIM4v3LlcGiven = TRUE;
            break;
        case  BSIM4v3_MOD_LLN :
            mod->BSIM4v3Lln = value->rValue;
            mod->BSIM4v3LlnGiven = TRUE;
            break;
        case  BSIM4v3_MOD_LW :
            mod->BSIM4v3Lw = value->rValue;
            mod->BSIM4v3LwGiven = TRUE;
            break;
        case  BSIM4v3_MOD_LWC :
            mod->BSIM4v3Lwc = value->rValue;
            mod->BSIM4v3LwcGiven = TRUE;
            break;
        case  BSIM4v3_MOD_LWN :
            mod->BSIM4v3Lwn = value->rValue;
            mod->BSIM4v3LwnGiven = TRUE;
            break;
        case  BSIM4v3_MOD_LWL :
            mod->BSIM4v3Lwl = value->rValue;
            mod->BSIM4v3LwlGiven = TRUE;
            break;
        case  BSIM4v3_MOD_LWLC :
            mod->BSIM4v3Lwlc = value->rValue;
            mod->BSIM4v3LwlcGiven = TRUE;
            break;
        case  BSIM4v3_MOD_LMIN :
            mod->BSIM4v3Lmin = value->rValue;
            mod->BSIM4v3LminGiven = TRUE;
            break;
        case  BSIM4v3_MOD_LMAX :
            mod->BSIM4v3Lmax = value->rValue;
            mod->BSIM4v3LmaxGiven = TRUE;
            break;
        case  BSIM4v3_MOD_WINT :
            mod->BSIM4v3Wint = value->rValue;
            mod->BSIM4v3WintGiven = TRUE;
            break;
        case  BSIM4v3_MOD_WL :
            mod->BSIM4v3Wl = value->rValue;
            mod->BSIM4v3WlGiven = TRUE;
            break;
        case  BSIM4v3_MOD_WLC :
            mod->BSIM4v3Wlc = value->rValue;
            mod->BSIM4v3WlcGiven = TRUE;
            break;
        case  BSIM4v3_MOD_WLN :
            mod->BSIM4v3Wln = value->rValue;
            mod->BSIM4v3WlnGiven = TRUE;
            break;
        case  BSIM4v3_MOD_WW :
            mod->BSIM4v3Ww = value->rValue;
            mod->BSIM4v3WwGiven = TRUE;
            break;
        case  BSIM4v3_MOD_WWC :
            mod->BSIM4v3Wwc = value->rValue;
            mod->BSIM4v3WwcGiven = TRUE;
            break;
        case  BSIM4v3_MOD_WWN :
            mod->BSIM4v3Wwn = value->rValue;
            mod->BSIM4v3WwnGiven = TRUE;
            break;
        case  BSIM4v3_MOD_WWL :
            mod->BSIM4v3Wwl = value->rValue;
            mod->BSIM4v3WwlGiven = TRUE;
            break;
        case  BSIM4v3_MOD_WWLC :
            mod->BSIM4v3Wwlc = value->rValue;
            mod->BSIM4v3WwlcGiven = TRUE;
            break;
        case  BSIM4v3_MOD_WMIN :
            mod->BSIM4v3Wmin = value->rValue;
            mod->BSIM4v3WminGiven = TRUE;
            break;
        case  BSIM4v3_MOD_WMAX :
            mod->BSIM4v3Wmax = value->rValue;
            mod->BSIM4v3WmaxGiven = TRUE;
            break;

        case  BSIM4v3_MOD_NOIA :
            mod->BSIM4v3oxideTrapDensityA = value->rValue;
            mod->BSIM4v3oxideTrapDensityAGiven = TRUE;
            break;
        case  BSIM4v3_MOD_NOIB :
            mod->BSIM4v3oxideTrapDensityB = value->rValue;
            mod->BSIM4v3oxideTrapDensityBGiven = TRUE;
            break;
        case  BSIM4v3_MOD_NOIC :
            mod->BSIM4v3oxideTrapDensityC = value->rValue;
            mod->BSIM4v3oxideTrapDensityCGiven = TRUE;
            break;
        case  BSIM4v3_MOD_EM :
            mod->BSIM4v3em = value->rValue;
            mod->BSIM4v3emGiven = TRUE;
            break;
        case  BSIM4v3_MOD_EF :
            mod->BSIM4v3ef = value->rValue;
            mod->BSIM4v3efGiven = TRUE;
            break;
        case  BSIM4v3_MOD_AF :
            mod->BSIM4v3af = value->rValue;
            mod->BSIM4v3afGiven = TRUE;
            break;
        case  BSIM4v3_MOD_KF :
            mod->BSIM4v3kf = value->rValue;
            mod->BSIM4v3kfGiven = TRUE;
            break;
        case  BSIM4v3_MOD_NMOS  :
            if(value->iValue) {
                mod->BSIM4v3type = 1;
                mod->BSIM4v3typeGiven = TRUE;
            }
            break;
        case  BSIM4v3_MOD_PMOS  :
            if(value->iValue) {
                mod->BSIM4v3type = - 1;
                mod->BSIM4v3typeGiven = TRUE;
            }
            break;
        default:
            return(E_BADPARM);
    }
    return(OK);
}


