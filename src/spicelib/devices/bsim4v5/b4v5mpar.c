/**** BSIM4.5.0 Released by Xuemei (Jane) Xi 07/29/2005 ****/

/**********
 * Copyright 2005 Regents of the University of California. All rights reserved.
 * File: b4mpar.c of BSIM4.5.0.
 * Author: 2000 Weidong Liu
 * Authors: 2001- Xuemei Xi, Mohan Dunga, Ali Niknejad, Chenming Hu.
 * Project Director: Prof. Chenming Hu.
 * Modified by Xuemei Xi, 04/06/2001.
 * Modified by Xuemei Xi, 10/05/2001.
 * Modified by Xuemei Xi, 11/15/2002.
 * Modified by Xuemei Xi, 05/09/2003.
 * Modified by Xuemei Xi, 03/04/2004.
 * Modified by Xuemei Xi, Mohan Dunga, 07/29/2005.
 **********/

#include "ngspice/ngspice.h"
#include "bsim4v5def.h"
#include "ngspice/ifsim.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"
#include "ngspice/const.h"

int
BSIM4v5mParam(
int param,
IFvalue *value,
GENmodel *inMod)
{
    BSIM4v5model *mod = (BSIM4v5model*)inMod;
    switch(param)
    {   case  BSIM4v5_MOD_MOBMOD :
            mod->BSIM4v5mobMod = value->iValue;
            mod->BSIM4v5mobModGiven = TRUE;
            break;
        case  BSIM4v5_MOD_BINUNIT :
            mod->BSIM4v5binUnit = value->iValue;
            mod->BSIM4v5binUnitGiven = TRUE;
            break;
        case  BSIM4v5_MOD_PARAMCHK :
            mod->BSIM4v5paramChk = value->iValue;
            mod->BSIM4v5paramChkGiven = TRUE;
            break;
        case  BSIM4v5_MOD_CAPMOD :
            mod->BSIM4v5capMod = value->iValue;
            mod->BSIM4v5capModGiven = TRUE;
            break;
        case  BSIM4v5_MOD_DIOMOD :
            mod->BSIM4v5dioMod = value->iValue;
            mod->BSIM4v5dioModGiven = TRUE;
            break;
        case  BSIM4v5_MOD_RDSMOD :
            mod->BSIM4v5rdsMod = value->iValue;
            mod->BSIM4v5rdsModGiven = TRUE;
            break;
        case  BSIM4v5_MOD_TRNQSMOD :
            mod->BSIM4v5trnqsMod = value->iValue;
            mod->BSIM4v5trnqsModGiven = TRUE;
            break;
        case  BSIM4v5_MOD_ACNQSMOD :
            mod->BSIM4v5acnqsMod = value->iValue;
            mod->BSIM4v5acnqsModGiven = TRUE;
            break;
        case  BSIM4v5_MOD_RBODYMOD :
            mod->BSIM4v5rbodyMod = value->iValue;
            mod->BSIM4v5rbodyModGiven = TRUE;
            break;
        case  BSIM4v5_MOD_RGATEMOD :
            mod->BSIM4v5rgateMod = value->iValue;
            mod->BSIM4v5rgateModGiven = TRUE;
            break;
        case  BSIM4v5_MOD_PERMOD :
            mod->BSIM4v5perMod = value->iValue;
            mod->BSIM4v5perModGiven = TRUE;
            break;
        case  BSIM4v5_MOD_GEOMOD :
            mod->BSIM4v5geoMod = value->iValue;
            mod->BSIM4v5geoModGiven = TRUE;
            break;
        case  BSIM4v5_MOD_RGEOMOD :
            mod->BSIM4v5rgeoMod = value->iValue;
            mod->BSIM4v5rgeoModGiven = TRUE;
            break;
        case  BSIM4v5_MOD_FNOIMOD :
            mod->BSIM4v5fnoiMod = value->iValue;
            mod->BSIM4v5fnoiModGiven = TRUE;
            break;
        case  BSIM4v5_MOD_TNOIMOD :
            mod->BSIM4v5tnoiMod = value->iValue;
            mod->BSIM4v5tnoiModGiven = TRUE;
            break;
        case  BSIM4v5_MOD_IGCMOD :
            mod->BSIM4v5igcMod = value->iValue;
            mod->BSIM4v5igcModGiven = TRUE;
            break;
        case  BSIM4v5_MOD_IGBMOD :
            mod->BSIM4v5igbMod = value->iValue;
            mod->BSIM4v5igbModGiven = TRUE;
            break;
        case  BSIM4v5_MOD_TEMPMOD :
            mod->BSIM4v5tempMod = value->iValue;
            mod->BSIM4v5tempModGiven = TRUE;
            break;

        case  BSIM4v5_MOD_VERSION :
            if (mod->BSIM4v5version)
                free(mod->BSIM4v5version);
            mod->BSIM4v5version = value->sValue;
            mod->BSIM4v5versionGiven = TRUE;
            break;
        case  BSIM4v5_MOD_TOXREF :
            mod->BSIM4v5toxref = value->rValue;
            mod->BSIM4v5toxrefGiven = TRUE;
            break;
        case  BSIM4v5_MOD_TOXE :
            mod->BSIM4v5toxe = value->rValue;
            mod->BSIM4v5toxeGiven = TRUE;
            break;
        case  BSIM4v5_MOD_TOXP :
            mod->BSIM4v5toxp = value->rValue;
            mod->BSIM4v5toxpGiven = TRUE;
            break;
        case  BSIM4v5_MOD_TOXM :
            mod->BSIM4v5toxm = value->rValue;
            mod->BSIM4v5toxmGiven = TRUE;
            break;
        case  BSIM4v5_MOD_DTOX :
            mod->BSIM4v5dtox = value->rValue;
            mod->BSIM4v5dtoxGiven = TRUE;
            break;
        case  BSIM4v5_MOD_EPSROX :
            mod->BSIM4v5epsrox = value->rValue;
            mod->BSIM4v5epsroxGiven = TRUE;
            break;

        case  BSIM4v5_MOD_CDSC :
            mod->BSIM4v5cdsc = value->rValue;
            mod->BSIM4v5cdscGiven = TRUE;
            break;
        case  BSIM4v5_MOD_CDSCB :
            mod->BSIM4v5cdscb = value->rValue;
            mod->BSIM4v5cdscbGiven = TRUE;
            break;

        case  BSIM4v5_MOD_CDSCD :
            mod->BSIM4v5cdscd = value->rValue;
            mod->BSIM4v5cdscdGiven = TRUE;
            break;

        case  BSIM4v5_MOD_CIT :
            mod->BSIM4v5cit = value->rValue;
            mod->BSIM4v5citGiven = TRUE;
            break;
        case  BSIM4v5_MOD_NFACTOR :
            mod->BSIM4v5nfactor = value->rValue;
            mod->BSIM4v5nfactorGiven = TRUE;
            break;
        case BSIM4v5_MOD_XJ:
            mod->BSIM4v5xj = value->rValue;
            mod->BSIM4v5xjGiven = TRUE;
            break;
        case BSIM4v5_MOD_VSAT:
            mod->BSIM4v5vsat = value->rValue;
            mod->BSIM4v5vsatGiven = TRUE;
            break;
        case BSIM4v5_MOD_A0:
            mod->BSIM4v5a0 = value->rValue;
            mod->BSIM4v5a0Given = TRUE;
            break;
        
        case BSIM4v5_MOD_AGS:
            mod->BSIM4v5ags= value->rValue;
            mod->BSIM4v5agsGiven = TRUE;
            break;
        
        case BSIM4v5_MOD_A1:
            mod->BSIM4v5a1 = value->rValue;
            mod->BSIM4v5a1Given = TRUE;
            break;
        case BSIM4v5_MOD_A2:
            mod->BSIM4v5a2 = value->rValue;
            mod->BSIM4v5a2Given = TRUE;
            break;
        case BSIM4v5_MOD_AT:
            mod->BSIM4v5at = value->rValue;
            mod->BSIM4v5atGiven = TRUE;
            break;
        case BSIM4v5_MOD_KETA:
            mod->BSIM4v5keta = value->rValue;
            mod->BSIM4v5ketaGiven = TRUE;
            break;    
        case BSIM4v5_MOD_NSUB:
            mod->BSIM4v5nsub = value->rValue;
            mod->BSIM4v5nsubGiven = TRUE;
            break;
        case BSIM4v5_MOD_NDEP:
            mod->BSIM4v5ndep = value->rValue;
            mod->BSIM4v5ndepGiven = TRUE;
	    if (mod->BSIM4v5ndep > 1.0e20)
		mod->BSIM4v5ndep *= 1.0e-6;
            break;
        case BSIM4v5_MOD_NSD:
            mod->BSIM4v5nsd = value->rValue;
            mod->BSIM4v5nsdGiven = TRUE;
            if (mod->BSIM4v5nsd > 1.000001e24)
                mod->BSIM4v5nsd *= 1.0e-6;
            break;
        case BSIM4v5_MOD_NGATE:
            mod->BSIM4v5ngate = value->rValue;
            mod->BSIM4v5ngateGiven = TRUE;
            if (mod->BSIM4v5ngate > 1.000001e24)
                mod->BSIM4v5ngate *= 1.0e-6;
            break;
        case BSIM4v5_MOD_GAMMA1:
            mod->BSIM4v5gamma1 = value->rValue;
            mod->BSIM4v5gamma1Given = TRUE;
            break;
        case BSIM4v5_MOD_GAMMA2:
            mod->BSIM4v5gamma2 = value->rValue;
            mod->BSIM4v5gamma2Given = TRUE;
            break;
        case BSIM4v5_MOD_VBX:
            mod->BSIM4v5vbx = value->rValue;
            mod->BSIM4v5vbxGiven = TRUE;
            break;
        case BSIM4v5_MOD_VBM:
            mod->BSIM4v5vbm = value->rValue;
            mod->BSIM4v5vbmGiven = TRUE;
            break;
        case BSIM4v5_MOD_XT:
            mod->BSIM4v5xt = value->rValue;
            mod->BSIM4v5xtGiven = TRUE;
            break;
        case  BSIM4v5_MOD_K1:
            mod->BSIM4v5k1 = value->rValue;
            mod->BSIM4v5k1Given = TRUE;
            break;
        case  BSIM4v5_MOD_KT1:
            mod->BSIM4v5kt1 = value->rValue;
            mod->BSIM4v5kt1Given = TRUE;
            break;
        case  BSIM4v5_MOD_KT1L:
            mod->BSIM4v5kt1l = value->rValue;
            mod->BSIM4v5kt1lGiven = TRUE;
            break;
        case  BSIM4v5_MOD_KT2:
            mod->BSIM4v5kt2 = value->rValue;
            mod->BSIM4v5kt2Given = TRUE;
            break;
        case  BSIM4v5_MOD_K2:
            mod->BSIM4v5k2 = value->rValue;
            mod->BSIM4v5k2Given = TRUE;
            break;
        case  BSIM4v5_MOD_K3:
            mod->BSIM4v5k3 = value->rValue;
            mod->BSIM4v5k3Given = TRUE;
            break;
        case  BSIM4v5_MOD_K3B:
            mod->BSIM4v5k3b = value->rValue;
            mod->BSIM4v5k3bGiven = TRUE;
            break;
        case  BSIM4v5_MOD_LPE0:
            mod->BSIM4v5lpe0 = value->rValue;
            mod->BSIM4v5lpe0Given = TRUE;
            break;
        case  BSIM4v5_MOD_LPEB:
            mod->BSIM4v5lpeb = value->rValue;
            mod->BSIM4v5lpebGiven = TRUE;
            break;
        case  BSIM4v5_MOD_DVTP0:
            mod->BSIM4v5dvtp0 = value->rValue;
            mod->BSIM4v5dvtp0Given = TRUE;
            break;
        case  BSIM4v5_MOD_DVTP1:
            mod->BSIM4v5dvtp1 = value->rValue;
            mod->BSIM4v5dvtp1Given = TRUE;
            break;
        case  BSIM4v5_MOD_W0:
            mod->BSIM4v5w0 = value->rValue;
            mod->BSIM4v5w0Given = TRUE;
            break;
        case  BSIM4v5_MOD_DVT0:               
            mod->BSIM4v5dvt0 = value->rValue;
            mod->BSIM4v5dvt0Given = TRUE;
            break;
        case  BSIM4v5_MOD_DVT1:             
            mod->BSIM4v5dvt1 = value->rValue;
            mod->BSIM4v5dvt1Given = TRUE;
            break;
        case  BSIM4v5_MOD_DVT2:             
            mod->BSIM4v5dvt2 = value->rValue;
            mod->BSIM4v5dvt2Given = TRUE;
            break;
        case  BSIM4v5_MOD_DVT0W:               
            mod->BSIM4v5dvt0w = value->rValue;
            mod->BSIM4v5dvt0wGiven = TRUE;
            break;
        case  BSIM4v5_MOD_DVT1W:             
            mod->BSIM4v5dvt1w = value->rValue;
            mod->BSIM4v5dvt1wGiven = TRUE;
            break;
        case  BSIM4v5_MOD_DVT2W:             
            mod->BSIM4v5dvt2w = value->rValue;
            mod->BSIM4v5dvt2wGiven = TRUE;
            break;
        case  BSIM4v5_MOD_DROUT:             
            mod->BSIM4v5drout = value->rValue;
            mod->BSIM4v5droutGiven = TRUE;
            break;
        case  BSIM4v5_MOD_DSUB:             
            mod->BSIM4v5dsub = value->rValue;
            mod->BSIM4v5dsubGiven = TRUE;
            break;
        case BSIM4v5_MOD_VTH0:
            mod->BSIM4v5vth0 = value->rValue;
            mod->BSIM4v5vth0Given = TRUE;
            break;
        case BSIM4v5_MOD_EU:
            mod->BSIM4v5eu = value->rValue;
            mod->BSIM4v5euGiven = TRUE;
            break;
        case BSIM4v5_MOD_UA:
            mod->BSIM4v5ua = value->rValue;
            mod->BSIM4v5uaGiven = TRUE;
            break;
        case BSIM4v5_MOD_UA1:
            mod->BSIM4v5ua1 = value->rValue;
            mod->BSIM4v5ua1Given = TRUE;
            break;
        case BSIM4v5_MOD_UB:
            mod->BSIM4v5ub = value->rValue;
            mod->BSIM4v5ubGiven = TRUE;
            break;
        case BSIM4v5_MOD_UB1:
            mod->BSIM4v5ub1 = value->rValue;
            mod->BSIM4v5ub1Given = TRUE;
            break;
        case BSIM4v5_MOD_UC:
            mod->BSIM4v5uc = value->rValue;
            mod->BSIM4v5ucGiven = TRUE;
            break;
        case BSIM4v5_MOD_UC1:
            mod->BSIM4v5uc1 = value->rValue;
            mod->BSIM4v5uc1Given = TRUE;
            break;
        case  BSIM4v5_MOD_U0 :
            mod->BSIM4v5u0 = value->rValue;
            mod->BSIM4v5u0Given = TRUE;
            break;
        case  BSIM4v5_MOD_UTE :
            mod->BSIM4v5ute = value->rValue;
            mod->BSIM4v5uteGiven = TRUE;
            break;

        case BSIM4v5_MOD_UD:
            mod->BSIM4v5ud = value->rValue;
            mod->BSIM4v5udGiven = TRUE;
            break;
        case BSIM4v5_MOD_UD1:
            mod->BSIM4v5ud1 = value->rValue;
            mod->BSIM4v5ud1Given = TRUE;
            break;
        case BSIM4v5_MOD_UP:
            mod->BSIM4v5up = value->rValue;
            mod->BSIM4v5upGiven = TRUE;
            break;
        case BSIM4v5_MOD_LP:
            mod->BSIM4v5lp = value->rValue;
            mod->BSIM4v5lpGiven = TRUE;
            break;
        case BSIM4v5_MOD_LUD:
            mod->BSIM4v5lud = value->rValue;
            mod->BSIM4v5ludGiven = TRUE;
            break;
        case BSIM4v5_MOD_LUD1:
            mod->BSIM4v5lud1 = value->rValue;
            mod->BSIM4v5lud1Given = TRUE;
            break;
        case BSIM4v5_MOD_LUP:
            mod->BSIM4v5lup = value->rValue;
            mod->BSIM4v5lupGiven = TRUE;
            break;
        case BSIM4v5_MOD_LLP:
            mod->BSIM4v5llp = value->rValue;
            mod->BSIM4v5llpGiven = TRUE;
            break;
        case BSIM4v5_MOD_WUD:
            mod->BSIM4v5wud = value->rValue;
            mod->BSIM4v5wudGiven = TRUE;
            break;
        case BSIM4v5_MOD_WUD1:
            mod->BSIM4v5wud1 = value->rValue;
            mod->BSIM4v5wud1Given = TRUE;
            break;
        case BSIM4v5_MOD_WUP:
            mod->BSIM4v5wup = value->rValue;
            mod->BSIM4v5wupGiven = TRUE;
            break;
        case BSIM4v5_MOD_WLP:
            mod->BSIM4v5wlp = value->rValue;
            mod->BSIM4v5wlpGiven = TRUE;
            break;
        case BSIM4v5_MOD_PUD:
            mod->BSIM4v5pud = value->rValue;
            mod->BSIM4v5pudGiven = TRUE;
            break;
        case BSIM4v5_MOD_PUD1:
            mod->BSIM4v5pud1 = value->rValue;
            mod->BSIM4v5pud1Given = TRUE;
            break;
        case BSIM4v5_MOD_PUP:
            mod->BSIM4v5pup = value->rValue;
            mod->BSIM4v5pupGiven = TRUE;
            break;
        case BSIM4v5_MOD_PLP:
            mod->BSIM4v5plp = value->rValue;
            mod->BSIM4v5plpGiven = TRUE;
            break;


        case BSIM4v5_MOD_VOFF:
            mod->BSIM4v5voff = value->rValue;
            mod->BSIM4v5voffGiven = TRUE;
            break;
        case BSIM4v5_MOD_TVOFF:
            mod->BSIM4v5tvoff = value->rValue;
            mod->BSIM4v5tvoffGiven = TRUE;
            break;
        case BSIM4v5_MOD_VOFFL:
            mod->BSIM4v5voffl = value->rValue;
            mod->BSIM4v5vofflGiven = TRUE;
            break;
        case BSIM4v5_MOD_MINV:
            mod->BSIM4v5minv = value->rValue;
            mod->BSIM4v5minvGiven = TRUE;
            break;
        case BSIM4v5_MOD_FPROUT:
            mod->BSIM4v5fprout = value->rValue;
            mod->BSIM4v5fproutGiven = TRUE;
            break;
        case BSIM4v5_MOD_PDITS:
            mod->BSIM4v5pdits = value->rValue;
            mod->BSIM4v5pditsGiven = TRUE;
            break;
        case BSIM4v5_MOD_PDITSD:
            mod->BSIM4v5pditsd = value->rValue;
            mod->BSIM4v5pditsdGiven = TRUE;
            break;
        case BSIM4v5_MOD_PDITSL:
            mod->BSIM4v5pditsl = value->rValue;
            mod->BSIM4v5pditslGiven = TRUE;
            break;
        case  BSIM4v5_MOD_DELTA :
            mod->BSIM4v5delta = value->rValue;
            mod->BSIM4v5deltaGiven = TRUE;
            break;
        case BSIM4v5_MOD_RDSW:
            mod->BSIM4v5rdsw = value->rValue;
            mod->BSIM4v5rdswGiven = TRUE;
            break;                     
        case BSIM4v5_MOD_RDSWMIN:
            mod->BSIM4v5rdswmin = value->rValue;
            mod->BSIM4v5rdswminGiven = TRUE;
            break;
        case BSIM4v5_MOD_RDWMIN:
            mod->BSIM4v5rdwmin = value->rValue;
            mod->BSIM4v5rdwminGiven = TRUE;
            break;
        case BSIM4v5_MOD_RSWMIN:
            mod->BSIM4v5rswmin = value->rValue;
            mod->BSIM4v5rswminGiven = TRUE;
            break;
        case BSIM4v5_MOD_RDW:
            mod->BSIM4v5rdw = value->rValue;
            mod->BSIM4v5rdwGiven = TRUE;
            break;
        case BSIM4v5_MOD_RSW:
            mod->BSIM4v5rsw = value->rValue;
            mod->BSIM4v5rswGiven = TRUE;
            break;
        case BSIM4v5_MOD_PRWG:
            mod->BSIM4v5prwg = value->rValue;
            mod->BSIM4v5prwgGiven = TRUE;
            break;                     
        case BSIM4v5_MOD_PRWB:
            mod->BSIM4v5prwb = value->rValue;
            mod->BSIM4v5prwbGiven = TRUE;
            break;                     
        case BSIM4v5_MOD_PRT:
            mod->BSIM4v5prt = value->rValue;
            mod->BSIM4v5prtGiven = TRUE;
            break;                     
        case BSIM4v5_MOD_ETA0:
            mod->BSIM4v5eta0 = value->rValue;
            mod->BSIM4v5eta0Given = TRUE;
            break;                 
        case BSIM4v5_MOD_ETAB:
            mod->BSIM4v5etab = value->rValue;
            mod->BSIM4v5etabGiven = TRUE;
            break;                 
        case BSIM4v5_MOD_PCLM:
            mod->BSIM4v5pclm = value->rValue;
            mod->BSIM4v5pclmGiven = TRUE;
            break;                 
        case BSIM4v5_MOD_PDIBL1:
            mod->BSIM4v5pdibl1 = value->rValue;
            mod->BSIM4v5pdibl1Given = TRUE;
            break;                 
        case BSIM4v5_MOD_PDIBL2:
            mod->BSIM4v5pdibl2 = value->rValue;
            mod->BSIM4v5pdibl2Given = TRUE;
            break;                 
        case BSIM4v5_MOD_PDIBLB:
            mod->BSIM4v5pdiblb = value->rValue;
            mod->BSIM4v5pdiblbGiven = TRUE;
            break;                 
        case BSIM4v5_MOD_PSCBE1:
            mod->BSIM4v5pscbe1 = value->rValue;
            mod->BSIM4v5pscbe1Given = TRUE;
            break;                 
        case BSIM4v5_MOD_PSCBE2:
            mod->BSIM4v5pscbe2 = value->rValue;
            mod->BSIM4v5pscbe2Given = TRUE;
            break;                 
        case BSIM4v5_MOD_PVAG:
            mod->BSIM4v5pvag = value->rValue;
            mod->BSIM4v5pvagGiven = TRUE;
            break;                 
        case  BSIM4v5_MOD_WR :
            mod->BSIM4v5wr = value->rValue;
            mod->BSIM4v5wrGiven = TRUE;
            break;
        case  BSIM4v5_MOD_DWG :
            mod->BSIM4v5dwg = value->rValue;
            mod->BSIM4v5dwgGiven = TRUE;
            break;
        case  BSIM4v5_MOD_DWB :
            mod->BSIM4v5dwb = value->rValue;
            mod->BSIM4v5dwbGiven = TRUE;
            break;
        case  BSIM4v5_MOD_B0 :
            mod->BSIM4v5b0 = value->rValue;
            mod->BSIM4v5b0Given = TRUE;
            break;
        case  BSIM4v5_MOD_B1 :
            mod->BSIM4v5b1 = value->rValue;
            mod->BSIM4v5b1Given = TRUE;
            break;
        case  BSIM4v5_MOD_ALPHA0 :
            mod->BSIM4v5alpha0 = value->rValue;
            mod->BSIM4v5alpha0Given = TRUE;
            break;
        case  BSIM4v5_MOD_ALPHA1 :
            mod->BSIM4v5alpha1 = value->rValue;
            mod->BSIM4v5alpha1Given = TRUE;
            break;
        case  BSIM4v5_MOD_AGIDL :
            mod->BSIM4v5agidl = value->rValue;
            mod->BSIM4v5agidlGiven = TRUE;
            break;
        case  BSIM4v5_MOD_BGIDL :
            mod->BSIM4v5bgidl = value->rValue;
            mod->BSIM4v5bgidlGiven = TRUE;
            break;
        case  BSIM4v5_MOD_CGIDL :
            mod->BSIM4v5cgidl = value->rValue;
            mod->BSIM4v5cgidlGiven = TRUE;
            break;
        case  BSIM4v5_MOD_PHIN :
            mod->BSIM4v5phin = value->rValue;
            mod->BSIM4v5phinGiven = TRUE;
            break;
        case  BSIM4v5_MOD_EGIDL :
            mod->BSIM4v5egidl = value->rValue;
            mod->BSIM4v5egidlGiven = TRUE;
            break;
        case  BSIM4v5_MOD_AIGC :
            mod->BSIM4v5aigc = value->rValue;
            mod->BSIM4v5aigcGiven = TRUE;
            break;
        case  BSIM4v5_MOD_BIGC :
            mod->BSIM4v5bigc = value->rValue;
            mod->BSIM4v5bigcGiven = TRUE;
            break;
        case  BSIM4v5_MOD_CIGC :
            mod->BSIM4v5cigc = value->rValue;
            mod->BSIM4v5cigcGiven = TRUE;
            break;
        case  BSIM4v5_MOD_AIGSD :
            mod->BSIM4v5aigsd = value->rValue;
            mod->BSIM4v5aigsdGiven = TRUE;
            break;
        case  BSIM4v5_MOD_BIGSD :
            mod->BSIM4v5bigsd = value->rValue;
            mod->BSIM4v5bigsdGiven = TRUE;
            break;
        case  BSIM4v5_MOD_CIGSD :
            mod->BSIM4v5cigsd = value->rValue;
            mod->BSIM4v5cigsdGiven = TRUE;
            break;
        case  BSIM4v5_MOD_AIGBACC :
            mod->BSIM4v5aigbacc = value->rValue;
            mod->BSIM4v5aigbaccGiven = TRUE;
            break;
        case  BSIM4v5_MOD_BIGBACC :
            mod->BSIM4v5bigbacc = value->rValue;
            mod->BSIM4v5bigbaccGiven = TRUE;
            break;
        case  BSIM4v5_MOD_CIGBACC :
            mod->BSIM4v5cigbacc = value->rValue;
            mod->BSIM4v5cigbaccGiven = TRUE;
            break;
        case  BSIM4v5_MOD_AIGBINV :
            mod->BSIM4v5aigbinv = value->rValue;
            mod->BSIM4v5aigbinvGiven = TRUE;
            break;
        case  BSIM4v5_MOD_BIGBINV :
            mod->BSIM4v5bigbinv = value->rValue;
            mod->BSIM4v5bigbinvGiven = TRUE;
            break;
        case  BSIM4v5_MOD_CIGBINV :
            mod->BSIM4v5cigbinv = value->rValue;
            mod->BSIM4v5cigbinvGiven = TRUE;
            break;
        case  BSIM4v5_MOD_NIGC :
            mod->BSIM4v5nigc = value->rValue;
            mod->BSIM4v5nigcGiven = TRUE;
            break;
        case  BSIM4v5_MOD_NIGBINV :
            mod->BSIM4v5nigbinv = value->rValue;
            mod->BSIM4v5nigbinvGiven = TRUE;
            break;
        case  BSIM4v5_MOD_NIGBACC :
            mod->BSIM4v5nigbacc = value->rValue;
            mod->BSIM4v5nigbaccGiven = TRUE;
            break;
        case  BSIM4v5_MOD_NTOX :
            mod->BSIM4v5ntox = value->rValue;
            mod->BSIM4v5ntoxGiven = TRUE;
            break;
        case  BSIM4v5_MOD_EIGBINV :
            mod->BSIM4v5eigbinv = value->rValue;
            mod->BSIM4v5eigbinvGiven = TRUE;
            break;
        case  BSIM4v5_MOD_PIGCD :
            mod->BSIM4v5pigcd = value->rValue;
            mod->BSIM4v5pigcdGiven = TRUE;
            break;
        case  BSIM4v5_MOD_POXEDGE :
            mod->BSIM4v5poxedge = value->rValue;
            mod->BSIM4v5poxedgeGiven = TRUE;
            break;
        case  BSIM4v5_MOD_XRCRG1 :
            mod->BSIM4v5xrcrg1 = value->rValue;
            mod->BSIM4v5xrcrg1Given = TRUE;
            break;
        case  BSIM4v5_MOD_XRCRG2 :
            mod->BSIM4v5xrcrg2 = value->rValue;
            mod->BSIM4v5xrcrg2Given = TRUE;
            break;
        case  BSIM4v5_MOD_LAMBDA :
            mod->BSIM4v5lambda = value->rValue;
            mod->BSIM4v5lambdaGiven = TRUE;
            break;
        case  BSIM4v5_MOD_VTL :
            mod->BSIM4v5vtl = value->rValue;
            mod->BSIM4v5vtlGiven = TRUE;
            break;
        case  BSIM4v5_MOD_XN:
            mod->BSIM4v5xn = value->rValue;
            mod->BSIM4v5xnGiven = TRUE;
            break;
        case  BSIM4v5_MOD_LC:
            mod->BSIM4v5lc = value->rValue;
            mod->BSIM4v5lcGiven = TRUE;
            break;
        case  BSIM4v5_MOD_TNOIA :
            mod->BSIM4v5tnoia = value->rValue;
            mod->BSIM4v5tnoiaGiven = TRUE;
            break;
        case  BSIM4v5_MOD_TNOIB :
            mod->BSIM4v5tnoib = value->rValue;
            mod->BSIM4v5tnoibGiven = TRUE;
            break;
        case  BSIM4v5_MOD_RNOIA :
            mod->BSIM4v5rnoia = value->rValue;
            mod->BSIM4v5rnoiaGiven = TRUE;
            break;
        case  BSIM4v5_MOD_RNOIB :
            mod->BSIM4v5rnoib = value->rValue;
            mod->BSIM4v5rnoibGiven = TRUE;
            break;
        case  BSIM4v5_MOD_NTNOI :
            mod->BSIM4v5ntnoi = value->rValue;
            mod->BSIM4v5ntnoiGiven = TRUE;
            break;
        case  BSIM4v5_MOD_VFBSDOFF:
            mod->BSIM4v5vfbsdoff = value->rValue;
            mod->BSIM4v5vfbsdoffGiven = TRUE;
            break;
        case  BSIM4v5_MOD_TVFBSDOFF:
            mod->BSIM4v5tvfbsdoff = value->rValue;
            mod->BSIM4v5tvfbsdoffGiven = TRUE;
            break;
        case  BSIM4v5_MOD_LINTNOI:
            mod->BSIM4v5lintnoi = value->rValue;
            mod->BSIM4v5lintnoiGiven = TRUE;
            break;

        /* stress effect */
        case  BSIM4v5_MOD_SAREF :
            mod->BSIM4v5saref = value->rValue;
            mod->BSIM4v5sarefGiven = TRUE;
            break;
        case  BSIM4v5_MOD_SBREF :
            mod->BSIM4v5sbref = value->rValue;
            mod->BSIM4v5sbrefGiven = TRUE;
            break;
        case  BSIM4v5_MOD_WLOD :
            mod->BSIM4v5wlod = value->rValue;
            mod->BSIM4v5wlodGiven = TRUE;
            break;
        case  BSIM4v5_MOD_KU0 :
            mod->BSIM4v5ku0 = value->rValue;
            mod->BSIM4v5ku0Given = TRUE;
            break;
        case  BSIM4v5_MOD_KVSAT :
            mod->BSIM4v5kvsat = value->rValue;
            mod->BSIM4v5kvsatGiven = TRUE;
            break;
        case  BSIM4v5_MOD_KVTH0 :
            mod->BSIM4v5kvth0 = value->rValue;
            mod->BSIM4v5kvth0Given = TRUE;
            break;
        case  BSIM4v5_MOD_TKU0 :
            mod->BSIM4v5tku0 = value->rValue;
            mod->BSIM4v5tku0Given = TRUE;
            break;
        case  BSIM4v5_MOD_LLODKU0 :
            mod->BSIM4v5llodku0 = value->rValue;
            mod->BSIM4v5llodku0Given = TRUE;
            break;
        case  BSIM4v5_MOD_WLODKU0 :
            mod->BSIM4v5wlodku0 = value->rValue;
            mod->BSIM4v5wlodku0Given = TRUE;
            break;
        case  BSIM4v5_MOD_LLODVTH :
            mod->BSIM4v5llodvth = value->rValue;
            mod->BSIM4v5llodvthGiven = TRUE;
            break;
        case  BSIM4v5_MOD_WLODVTH :
            mod->BSIM4v5wlodvth = value->rValue;
            mod->BSIM4v5wlodvthGiven = TRUE;
            break;
        case  BSIM4v5_MOD_LKU0 :
            mod->BSIM4v5lku0 = value->rValue;
            mod->BSIM4v5lku0Given = TRUE;
            break;
        case  BSIM4v5_MOD_WKU0 :
            mod->BSIM4v5wku0 = value->rValue;
            mod->BSIM4v5wku0Given = TRUE;
            break;
        case  BSIM4v5_MOD_PKU0 :
            mod->BSIM4v5pku0 = value->rValue;
            mod->BSIM4v5pku0Given = TRUE;
            break;
        case  BSIM4v5_MOD_LKVTH0 :
            mod->BSIM4v5lkvth0 = value->rValue;
            mod->BSIM4v5lkvth0Given = TRUE;
            break;
        case  BSIM4v5_MOD_WKVTH0 :
            mod->BSIM4v5wkvth0 = value->rValue;
            mod->BSIM4v5wkvth0Given = TRUE;
            break;
        case  BSIM4v5_MOD_PKVTH0 :
            mod->BSIM4v5pkvth0 = value->rValue;
            mod->BSIM4v5pkvth0Given = TRUE;
            break;
        case  BSIM4v5_MOD_STK2 :
            mod->BSIM4v5stk2 = value->rValue;
            mod->BSIM4v5stk2Given = TRUE;
            break;
        case  BSIM4v5_MOD_LODK2 :
            mod->BSIM4v5lodk2 = value->rValue;
            mod->BSIM4v5lodk2Given = TRUE;
            break;
        case  BSIM4v5_MOD_STETA0 :
            mod->BSIM4v5steta0 = value->rValue;
            mod->BSIM4v5steta0Given = TRUE;
            break;
        case  BSIM4v5_MOD_LODETA0 :
            mod->BSIM4v5lodeta0 = value->rValue;
            mod->BSIM4v5lodeta0Given = TRUE;
            break;

        case  BSIM4v5_MOD_WEB :
            mod->BSIM4v5web = value->rValue;
            mod->BSIM4v5webGiven = TRUE;
            break;
	case BSIM4v5_MOD_WEC :
            mod->BSIM4v5wec = value->rValue;
            mod->BSIM4v5wecGiven = TRUE;
            break;
        case  BSIM4v5_MOD_KVTH0WE :
            mod->BSIM4v5kvth0we = value->rValue;
            mod->BSIM4v5kvth0weGiven = TRUE;
            break;
        case  BSIM4v5_MOD_K2WE :
            mod->BSIM4v5k2we = value->rValue;
            mod->BSIM4v5k2weGiven = TRUE;
            break;
        case  BSIM4v5_MOD_KU0WE :
            mod->BSIM4v5ku0we = value->rValue;
            mod->BSIM4v5ku0weGiven = TRUE;
            break;
        case  BSIM4v5_MOD_SCREF :
            mod->BSIM4v5scref = value->rValue;
            mod->BSIM4v5screfGiven = TRUE;
            break;
        case  BSIM4v5_MOD_WPEMOD :
            mod->BSIM4v5wpemod = value->rValue;
            mod->BSIM4v5wpemodGiven = TRUE;
            break;
        case  BSIM4v5_MOD_LKVTH0WE :
            mod->BSIM4v5lkvth0we = value->rValue;
            mod->BSIM4v5lkvth0weGiven = TRUE;
            break;
        case  BSIM4v5_MOD_LK2WE :
            mod->BSIM4v5lk2we = value->rValue;
            mod->BSIM4v5lk2weGiven = TRUE;
            break;
        case  BSIM4v5_MOD_LKU0WE :
            mod->BSIM4v5lku0we = value->rValue;
            mod->BSIM4v5lku0weGiven = TRUE;
            break;
        case  BSIM4v5_MOD_WKVTH0WE :
            mod->BSIM4v5wkvth0we = value->rValue;
            mod->BSIM4v5wkvth0weGiven = TRUE;
            break;
        case  BSIM4v5_MOD_WK2WE :
            mod->BSIM4v5wk2we = value->rValue;
            mod->BSIM4v5wk2weGiven = TRUE;
            break;
        case  BSIM4v5_MOD_WKU0WE :
            mod->BSIM4v5wku0we = value->rValue;
            mod->BSIM4v5wku0weGiven = TRUE;
            break;
        case  BSIM4v5_MOD_PKVTH0WE :
            mod->BSIM4v5pkvth0we = value->rValue;
            mod->BSIM4v5pkvth0weGiven = TRUE;
            break;
        case  BSIM4v5_MOD_PK2WE :
            mod->BSIM4v5pk2we = value->rValue;
            mod->BSIM4v5pk2weGiven = TRUE;
            break;
        case  BSIM4v5_MOD_PKU0WE :
            mod->BSIM4v5pku0we = value->rValue;
            mod->BSIM4v5pku0weGiven = TRUE;
            break;

        case  BSIM4v5_MOD_BETA0 :
            mod->BSIM4v5beta0 = value->rValue;
            mod->BSIM4v5beta0Given = TRUE;
            break;
        case  BSIM4v5_MOD_IJTHDFWD :
            mod->BSIM4v5ijthdfwd = value->rValue;
            mod->BSIM4v5ijthdfwdGiven = TRUE;
            break;
        case  BSIM4v5_MOD_IJTHSFWD :
            mod->BSIM4v5ijthsfwd = value->rValue;
            mod->BSIM4v5ijthsfwdGiven = TRUE;
            break;
        case  BSIM4v5_MOD_IJTHDREV :
            mod->BSIM4v5ijthdrev = value->rValue;
            mod->BSIM4v5ijthdrevGiven = TRUE;
            break;
        case  BSIM4v5_MOD_IJTHSREV :
            mod->BSIM4v5ijthsrev = value->rValue;
            mod->BSIM4v5ijthsrevGiven = TRUE;
            break;
        case  BSIM4v5_MOD_XJBVD :
            mod->BSIM4v5xjbvd = value->rValue;
            mod->BSIM4v5xjbvdGiven = TRUE;
            break;
        case  BSIM4v5_MOD_XJBVS :
            mod->BSIM4v5xjbvs = value->rValue;
            mod->BSIM4v5xjbvsGiven = TRUE;
            break;
        case  BSIM4v5_MOD_BVD :
            mod->BSIM4v5bvd = value->rValue;
            mod->BSIM4v5bvdGiven = TRUE;
            break;
        case  BSIM4v5_MOD_BVS :
            mod->BSIM4v5bvs = value->rValue;
            mod->BSIM4v5bvsGiven = TRUE;
            break;
        
        /* reverse diode */
        case  BSIM4v5_MOD_JTSS :
            mod->BSIM4v5jtss = value->rValue;
            mod->BSIM4v5jtssGiven = TRUE;
            break;
        case  BSIM4v5_MOD_JTSD :
            mod->BSIM4v5jtsd = value->rValue;
            mod->BSIM4v5jtsdGiven = TRUE;
            break;
        case  BSIM4v5_MOD_JTSSWS :
            mod->BSIM4v5jtssws = value->rValue;
            mod->BSIM4v5jtsswsGiven = TRUE;
            break;
        case  BSIM4v5_MOD_JTSSWD :
            mod->BSIM4v5jtsswd = value->rValue;
            mod->BSIM4v5jtsswdGiven = TRUE;
            break;
        case  BSIM4v5_MOD_JTSSWGS :
            mod->BSIM4v5jtsswgs = value->rValue;
            mod->BSIM4v5jtsswgsGiven = TRUE;
            break;
        case  BSIM4v5_MOD_JTSSWGD :
            mod->BSIM4v5jtsswgd = value->rValue;
            mod->BSIM4v5jtsswgdGiven = TRUE;
            break;
        case  BSIM4v5_MOD_NJTS :
            mod->BSIM4v5njts = value->rValue;
            mod->BSIM4v5njtsGiven = TRUE;
            break;
        case  BSIM4v5_MOD_NJTSSW :
            mod->BSIM4v5njtssw = value->rValue;
            mod->BSIM4v5njtsswGiven = TRUE;
            break;
        case  BSIM4v5_MOD_NJTSSWG :
            mod->BSIM4v5njtsswg = value->rValue;
            mod->BSIM4v5njtsswgGiven = TRUE;
            break;
        case  BSIM4v5_MOD_XTSS :
            mod->BSIM4v5xtss = value->rValue;
            mod->BSIM4v5xtssGiven = TRUE;
            break;
        case  BSIM4v5_MOD_XTSD :
            mod->BSIM4v5xtsd = value->rValue;
            mod->BSIM4v5xtsdGiven = TRUE;
            break;
        case  BSIM4v5_MOD_XTSSWS :
            mod->BSIM4v5xtssws = value->rValue;
            mod->BSIM4v5xtsswsGiven = TRUE;
            break;
        case  BSIM4v5_MOD_XTSSWD :
            mod->BSIM4v5xtsswd = value->rValue;
            mod->BSIM4v5xtsswdGiven = TRUE;
            break;
        case  BSIM4v5_MOD_XTSSWGS :
            mod->BSIM4v5xtsswgs = value->rValue;
            mod->BSIM4v5xtsswgsGiven = TRUE;
            break;
        case  BSIM4v5_MOD_XTSSWGD :
            mod->BSIM4v5xtsswgd = value->rValue;
            mod->BSIM4v5xtsswgdGiven = TRUE;
            break;
        case  BSIM4v5_MOD_TNJTS :
            mod->BSIM4v5tnjts = value->rValue;
            mod->BSIM4v5tnjtsGiven = TRUE;
            break;
        case  BSIM4v5_MOD_TNJTSSW :
            mod->BSIM4v5tnjtssw = value->rValue;
            mod->BSIM4v5tnjtsswGiven = TRUE;
            break;
        case  BSIM4v5_MOD_TNJTSSWG :
            mod->BSIM4v5tnjtsswg = value->rValue;
            mod->BSIM4v5tnjtsswgGiven = TRUE;
            break;
        case  BSIM4v5_MOD_VTSS :
            mod->BSIM4v5vtss = value->rValue;
            mod->BSIM4v5vtssGiven = TRUE;
            break;
        case  BSIM4v5_MOD_VTSD :
            mod->BSIM4v5vtsd = value->rValue;
            mod->BSIM4v5vtsdGiven = TRUE;
            break;
        case  BSIM4v5_MOD_VTSSWS :
            mod->BSIM4v5vtssws = value->rValue;
            mod->BSIM4v5vtsswsGiven = TRUE;
            break;
        case  BSIM4v5_MOD_VTSSWD :
            mod->BSIM4v5vtsswd = value->rValue;
            mod->BSIM4v5vtsswdGiven = TRUE;
            break;
        case  BSIM4v5_MOD_VTSSWGS :
            mod->BSIM4v5vtsswgs = value->rValue;
            mod->BSIM4v5vtsswgsGiven = TRUE;
            break;
        case  BSIM4v5_MOD_VTSSWGD :
            mod->BSIM4v5vtsswgd = value->rValue;
            mod->BSIM4v5vtsswgdGiven = TRUE;
            break;

        case  BSIM4v5_MOD_VFB :
            mod->BSIM4v5vfb = value->rValue;
            mod->BSIM4v5vfbGiven = TRUE;
            break;

        case  BSIM4v5_MOD_GBMIN :
            mod->BSIM4v5gbmin = value->rValue;
            mod->BSIM4v5gbminGiven = TRUE;
            break;
        case  BSIM4v5_MOD_RBDB :
            mod->BSIM4v5rbdb = value->rValue;
            mod->BSIM4v5rbdbGiven = TRUE;
            break;
        case  BSIM4v5_MOD_RBPB :
            mod->BSIM4v5rbpb = value->rValue;
            mod->BSIM4v5rbpbGiven = TRUE;
            break;
        case  BSIM4v5_MOD_RBSB :
            mod->BSIM4v5rbsb = value->rValue;
            mod->BSIM4v5rbsbGiven = TRUE;
            break;
        case  BSIM4v5_MOD_RBPS :
            mod->BSIM4v5rbps = value->rValue;
            mod->BSIM4v5rbpsGiven = TRUE;
            break;
        case  BSIM4v5_MOD_RBPD :
            mod->BSIM4v5rbpd = value->rValue;
            mod->BSIM4v5rbpdGiven = TRUE;
            break;

        case  BSIM4v5_MOD_RBPS0 :
            mod->BSIM4v5rbps0 = value->rValue;
            mod->BSIM4v5rbps0Given = TRUE;
            break;
        case  BSIM4v5_MOD_RBPSL :
            mod->BSIM4v5rbpsl = value->rValue;
            mod->BSIM4v5rbpslGiven = TRUE;
            break;
        case  BSIM4v5_MOD_RBPSW :
            mod->BSIM4v5rbpsw = value->rValue;
            mod->BSIM4v5rbpswGiven = TRUE;
            break;
        case  BSIM4v5_MOD_RBPSNF :
            mod->BSIM4v5rbpsnf = value->rValue;
            mod->BSIM4v5rbpsnfGiven = TRUE;
            break;

        case  BSIM4v5_MOD_RBPD0 :
            mod->BSIM4v5rbpd0 = value->rValue;
            mod->BSIM4v5rbpd0Given = TRUE;
            break;
        case  BSIM4v5_MOD_RBPDL :
            mod->BSIM4v5rbpdl = value->rValue;
            mod->BSIM4v5rbpdlGiven = TRUE;
            break;
        case  BSIM4v5_MOD_RBPDW :
            mod->BSIM4v5rbpdw = value->rValue;
            mod->BSIM4v5rbpdwGiven = TRUE;
            break;
        case  BSIM4v5_MOD_RBPDNF :
            mod->BSIM4v5rbpdnf = value->rValue;
            mod->BSIM4v5rbpdnfGiven = TRUE;
            break;

        case  BSIM4v5_MOD_RBPBX0 :
            mod->BSIM4v5rbpbx0 = value->rValue;
            mod->BSIM4v5rbpbx0Given = TRUE;
            break;
        case  BSIM4v5_MOD_RBPBXL :
            mod->BSIM4v5rbpbxl = value->rValue;
            mod->BSIM4v5rbpbxlGiven = TRUE;
            break;
        case  BSIM4v5_MOD_RBPBXW :
            mod->BSIM4v5rbpbxw = value->rValue;
            mod->BSIM4v5rbpbxwGiven = TRUE;
            break;
        case  BSIM4v5_MOD_RBPBXNF :
            mod->BSIM4v5rbpbxnf = value->rValue;
            mod->BSIM4v5rbpbxnfGiven = TRUE;
            break;
        case  BSIM4v5_MOD_RBPBY0 :
            mod->BSIM4v5rbpby0 = value->rValue;
            mod->BSIM4v5rbpby0Given = TRUE;
            break;
        case  BSIM4v5_MOD_RBPBYL :
            mod->BSIM4v5rbpbyl = value->rValue;
            mod->BSIM4v5rbpbylGiven = TRUE;
            break;
        case  BSIM4v5_MOD_RBPBYW :
            mod->BSIM4v5rbpbyw = value->rValue;
            mod->BSIM4v5rbpbywGiven = TRUE;
            break;
        case  BSIM4v5_MOD_RBPBYNF :
            mod->BSIM4v5rbpbynf = value->rValue;
            mod->BSIM4v5rbpbynfGiven = TRUE;
            break;
       case  BSIM4v5_MOD_RBSBX0 :
            mod->BSIM4v5rbsbx0 = value->rValue;
            mod->BSIM4v5rbsbx0Given = TRUE;
            break;
       case  BSIM4v5_MOD_RBSBY0 :
            mod->BSIM4v5rbsby0 = value->rValue;
            mod->BSIM4v5rbsby0Given = TRUE;
            break;
       case  BSIM4v5_MOD_RBDBX0 :
            mod->BSIM4v5rbdbx0 = value->rValue;
            mod->BSIM4v5rbdbx0Given = TRUE;
            break;
       case  BSIM4v5_MOD_RBDBY0 :
            mod->BSIM4v5rbdby0 = value->rValue;
            mod->BSIM4v5rbdby0Given = TRUE;
            break;


       case  BSIM4v5_MOD_RBSDBXL :
            mod->BSIM4v5rbsdbxl = value->rValue;
            mod->BSIM4v5rbsdbxlGiven = TRUE;
            break;
       case  BSIM4v5_MOD_RBSDBXW :
            mod->BSIM4v5rbsdbxw = value->rValue;
            mod->BSIM4v5rbsdbxwGiven = TRUE;
            break;
       case  BSIM4v5_MOD_RBSDBXNF :
            mod->BSIM4v5rbsdbxnf = value->rValue;
            mod->BSIM4v5rbsdbxnfGiven = TRUE;
            break;
       case  BSIM4v5_MOD_RBSDBYL :
            mod->BSIM4v5rbsdbyl = value->rValue;
            mod->BSIM4v5rbsdbylGiven = TRUE;
            break;
       case  BSIM4v5_MOD_RBSDBYW :
            mod->BSIM4v5rbsdbyw = value->rValue;
            mod->BSIM4v5rbsdbywGiven = TRUE;
            break;
       case  BSIM4v5_MOD_RBSDBYNF :
            mod->BSIM4v5rbsdbynf = value->rValue;
            mod->BSIM4v5rbsdbynfGiven = TRUE;
            break;
 
        case  BSIM4v5_MOD_CGSL :
            mod->BSIM4v5cgsl = value->rValue;
            mod->BSIM4v5cgslGiven = TRUE;
            break;
        case  BSIM4v5_MOD_CGDL :
            mod->BSIM4v5cgdl = value->rValue;
            mod->BSIM4v5cgdlGiven = TRUE;
            break;
        case  BSIM4v5_MOD_CKAPPAS :
            mod->BSIM4v5ckappas = value->rValue;
            mod->BSIM4v5ckappasGiven = TRUE;
            break;
        case  BSIM4v5_MOD_CKAPPAD :
            mod->BSIM4v5ckappad = value->rValue;
            mod->BSIM4v5ckappadGiven = TRUE;
            break;
        case  BSIM4v5_MOD_CF :
            mod->BSIM4v5cf = value->rValue;
            mod->BSIM4v5cfGiven = TRUE;
            break;
        case  BSIM4v5_MOD_CLC :
            mod->BSIM4v5clc = value->rValue;
            mod->BSIM4v5clcGiven = TRUE;
            break;
        case  BSIM4v5_MOD_CLE :
            mod->BSIM4v5cle = value->rValue;
            mod->BSIM4v5cleGiven = TRUE;
            break;
        case  BSIM4v5_MOD_DWC :
            mod->BSIM4v5dwc = value->rValue;
            mod->BSIM4v5dwcGiven = TRUE;
            break;
        case  BSIM4v5_MOD_DLC :
            mod->BSIM4v5dlc = value->rValue;
            mod->BSIM4v5dlcGiven = TRUE;
            break;
        case  BSIM4v5_MOD_XW :
            mod->BSIM4v5xw = value->rValue;
            mod->BSIM4v5xwGiven = TRUE;
            break;
        case  BSIM4v5_MOD_XL :
            mod->BSIM4v5xl = value->rValue;
            mod->BSIM4v5xlGiven = TRUE;
            break;
        case  BSIM4v5_MOD_DLCIG :
            mod->BSIM4v5dlcig = value->rValue;
            mod->BSIM4v5dlcigGiven = TRUE;
            break;
        case  BSIM4v5_MOD_DWJ :
            mod->BSIM4v5dwj = value->rValue;
            mod->BSIM4v5dwjGiven = TRUE;
            break;
        case  BSIM4v5_MOD_VFBCV :
            mod->BSIM4v5vfbcv = value->rValue;
            mod->BSIM4v5vfbcvGiven = TRUE;
            break;
        case  BSIM4v5_MOD_ACDE :
            mod->BSIM4v5acde = value->rValue;
            mod->BSIM4v5acdeGiven = TRUE;
            break;
        case  BSIM4v5_MOD_MOIN :
            mod->BSIM4v5moin = value->rValue;
            mod->BSIM4v5moinGiven = TRUE;
            break;
        case  BSIM4v5_MOD_NOFF :
            mod->BSIM4v5noff = value->rValue;
            mod->BSIM4v5noffGiven = TRUE;
            break;
        case  BSIM4v5_MOD_VOFFCV :
            mod->BSIM4v5voffcv = value->rValue;
            mod->BSIM4v5voffcvGiven = TRUE;
            break;
        case  BSIM4v5_MOD_DMCG :
            mod->BSIM4v5dmcg = value->rValue;
            mod->BSIM4v5dmcgGiven = TRUE;
            break;
        case  BSIM4v5_MOD_DMCI :
            mod->BSIM4v5dmci = value->rValue;
            mod->BSIM4v5dmciGiven = TRUE;
            break;
        case  BSIM4v5_MOD_DMDG :
            mod->BSIM4v5dmdg = value->rValue;
            mod->BSIM4v5dmdgGiven = TRUE;
            break;
        case  BSIM4v5_MOD_DMCGT :
            mod->BSIM4v5dmcgt = value->rValue;
            mod->BSIM4v5dmcgtGiven = TRUE;
            break;
        case  BSIM4v5_MOD_XGW :
            mod->BSIM4v5xgw = value->rValue;
            mod->BSIM4v5xgwGiven = TRUE;
            break;
        case  BSIM4v5_MOD_XGL :
            mod->BSIM4v5xgl = value->rValue;
            mod->BSIM4v5xglGiven = TRUE;
            break;
        case  BSIM4v5_MOD_RSHG :
            mod->BSIM4v5rshg = value->rValue;
            mod->BSIM4v5rshgGiven = TRUE;
            break;
        case  BSIM4v5_MOD_NGCON :
            mod->BSIM4v5ngcon = value->rValue;
            mod->BSIM4v5ngconGiven = TRUE;
            break;
        case  BSIM4v5_MOD_TCJ :
            mod->BSIM4v5tcj = value->rValue;
            mod->BSIM4v5tcjGiven = TRUE;
            break;
        case  BSIM4v5_MOD_TPB :
            mod->BSIM4v5tpb = value->rValue;
            mod->BSIM4v5tpbGiven = TRUE;
            break;
        case  BSIM4v5_MOD_TCJSW :
            mod->BSIM4v5tcjsw = value->rValue;
            mod->BSIM4v5tcjswGiven = TRUE;
            break;
        case  BSIM4v5_MOD_TPBSW :
            mod->BSIM4v5tpbsw = value->rValue;
            mod->BSIM4v5tpbswGiven = TRUE;
            break;
        case  BSIM4v5_MOD_TCJSWG :
            mod->BSIM4v5tcjswg = value->rValue;
            mod->BSIM4v5tcjswgGiven = TRUE;
            break;
        case  BSIM4v5_MOD_TPBSWG :
            mod->BSIM4v5tpbswg = value->rValue;
            mod->BSIM4v5tpbswgGiven = TRUE;
            break;

	/* Length dependence */
        case  BSIM4v5_MOD_LCDSC :
            mod->BSIM4v5lcdsc = value->rValue;
            mod->BSIM4v5lcdscGiven = TRUE;
            break;


        case  BSIM4v5_MOD_LCDSCB :
            mod->BSIM4v5lcdscb = value->rValue;
            mod->BSIM4v5lcdscbGiven = TRUE;
            break;
        case  BSIM4v5_MOD_LCDSCD :
            mod->BSIM4v5lcdscd = value->rValue;
            mod->BSIM4v5lcdscdGiven = TRUE;
            break;
        case  BSIM4v5_MOD_LCIT :
            mod->BSIM4v5lcit = value->rValue;
            mod->BSIM4v5lcitGiven = TRUE;
            break;
        case  BSIM4v5_MOD_LNFACTOR :
            mod->BSIM4v5lnfactor = value->rValue;
            mod->BSIM4v5lnfactorGiven = TRUE;
            break;
        case BSIM4v5_MOD_LXJ:
            mod->BSIM4v5lxj = value->rValue;
            mod->BSIM4v5lxjGiven = TRUE;
            break;
        case BSIM4v5_MOD_LVSAT:
            mod->BSIM4v5lvsat = value->rValue;
            mod->BSIM4v5lvsatGiven = TRUE;
            break;
        
        
        case BSIM4v5_MOD_LA0:
            mod->BSIM4v5la0 = value->rValue;
            mod->BSIM4v5la0Given = TRUE;
            break;
        case BSIM4v5_MOD_LAGS:
            mod->BSIM4v5lags = value->rValue;
            mod->BSIM4v5lagsGiven = TRUE;
            break;
        case BSIM4v5_MOD_LA1:
            mod->BSIM4v5la1 = value->rValue;
            mod->BSIM4v5la1Given = TRUE;
            break;
        case BSIM4v5_MOD_LA2:
            mod->BSIM4v5la2 = value->rValue;
            mod->BSIM4v5la2Given = TRUE;
            break;
        case BSIM4v5_MOD_LAT:
            mod->BSIM4v5lat = value->rValue;
            mod->BSIM4v5latGiven = TRUE;
            break;
        case BSIM4v5_MOD_LKETA:
            mod->BSIM4v5lketa = value->rValue;
            mod->BSIM4v5lketaGiven = TRUE;
            break;    
        case BSIM4v5_MOD_LNSUB:
            mod->BSIM4v5lnsub = value->rValue;
            mod->BSIM4v5lnsubGiven = TRUE;
            break;
        case BSIM4v5_MOD_LNDEP:
            mod->BSIM4v5lndep = value->rValue;
            mod->BSIM4v5lndepGiven = TRUE;
	    if (mod->BSIM4v5lndep > 1.0e20)
		mod->BSIM4v5lndep *= 1.0e-6;
            break;
        case BSIM4v5_MOD_LNSD:
            mod->BSIM4v5lnsd = value->rValue;
            mod->BSIM4v5lnsdGiven = TRUE;
            if (mod->BSIM4v5lnsd > 1.0e23)
                mod->BSIM4v5lnsd *= 1.0e-6;
            break;
        case BSIM4v5_MOD_LNGATE:
            mod->BSIM4v5lngate = value->rValue;
            mod->BSIM4v5lngateGiven = TRUE;
	    if (mod->BSIM4v5lngate > 1.0e23)
		mod->BSIM4v5lngate *= 1.0e-6;
            break;
        case BSIM4v5_MOD_LGAMMA1:
            mod->BSIM4v5lgamma1 = value->rValue;
            mod->BSIM4v5lgamma1Given = TRUE;
            break;
        case BSIM4v5_MOD_LGAMMA2:
            mod->BSIM4v5lgamma2 = value->rValue;
            mod->BSIM4v5lgamma2Given = TRUE;
            break;
        case BSIM4v5_MOD_LVBX:
            mod->BSIM4v5lvbx = value->rValue;
            mod->BSIM4v5lvbxGiven = TRUE;
            break;
        case BSIM4v5_MOD_LVBM:
            mod->BSIM4v5lvbm = value->rValue;
            mod->BSIM4v5lvbmGiven = TRUE;
            break;
        case BSIM4v5_MOD_LXT:
            mod->BSIM4v5lxt = value->rValue;
            mod->BSIM4v5lxtGiven = TRUE;
            break;
        case  BSIM4v5_MOD_LK1:
            mod->BSIM4v5lk1 = value->rValue;
            mod->BSIM4v5lk1Given = TRUE;
            break;
        case  BSIM4v5_MOD_LKT1:
            mod->BSIM4v5lkt1 = value->rValue;
            mod->BSIM4v5lkt1Given = TRUE;
            break;
        case  BSIM4v5_MOD_LKT1L:
            mod->BSIM4v5lkt1l = value->rValue;
            mod->BSIM4v5lkt1lGiven = TRUE;
            break;
        case  BSIM4v5_MOD_LKT2:
            mod->BSIM4v5lkt2 = value->rValue;
            mod->BSIM4v5lkt2Given = TRUE;
            break;
        case  BSIM4v5_MOD_LK2:
            mod->BSIM4v5lk2 = value->rValue;
            mod->BSIM4v5lk2Given = TRUE;
            break;
        case  BSIM4v5_MOD_LK3:
            mod->BSIM4v5lk3 = value->rValue;
            mod->BSIM4v5lk3Given = TRUE;
            break;
        case  BSIM4v5_MOD_LK3B:
            mod->BSIM4v5lk3b = value->rValue;
            mod->BSIM4v5lk3bGiven = TRUE;
            break;
        case  BSIM4v5_MOD_LLPE0:
            mod->BSIM4v5llpe0 = value->rValue;
            mod->BSIM4v5llpe0Given = TRUE;
            break;
        case  BSIM4v5_MOD_LLPEB:
            mod->BSIM4v5llpeb = value->rValue;
            mod->BSIM4v5llpebGiven = TRUE;
            break;
        case  BSIM4v5_MOD_LDVTP0:
            mod->BSIM4v5ldvtp0 = value->rValue;
            mod->BSIM4v5ldvtp0Given = TRUE;
            break;
        case  BSIM4v5_MOD_LDVTP1:
            mod->BSIM4v5ldvtp1 = value->rValue;
            mod->BSIM4v5ldvtp1Given = TRUE;
            break;
        case  BSIM4v5_MOD_LW0:
            mod->BSIM4v5lw0 = value->rValue;
            mod->BSIM4v5lw0Given = TRUE;
            break;
        case  BSIM4v5_MOD_LDVT0:               
            mod->BSIM4v5ldvt0 = value->rValue;
            mod->BSIM4v5ldvt0Given = TRUE;
            break;
        case  BSIM4v5_MOD_LDVT1:             
            mod->BSIM4v5ldvt1 = value->rValue;
            mod->BSIM4v5ldvt1Given = TRUE;
            break;
        case  BSIM4v5_MOD_LDVT2:             
            mod->BSIM4v5ldvt2 = value->rValue;
            mod->BSIM4v5ldvt2Given = TRUE;
            break;
        case  BSIM4v5_MOD_LDVT0W:               
            mod->BSIM4v5ldvt0w = value->rValue;
            mod->BSIM4v5ldvt0wGiven = TRUE;
            break;
        case  BSIM4v5_MOD_LDVT1W:             
            mod->BSIM4v5ldvt1w = value->rValue;
            mod->BSIM4v5ldvt1wGiven = TRUE;
            break;
        case  BSIM4v5_MOD_LDVT2W:             
            mod->BSIM4v5ldvt2w = value->rValue;
            mod->BSIM4v5ldvt2wGiven = TRUE;
            break;
        case  BSIM4v5_MOD_LDROUT:             
            mod->BSIM4v5ldrout = value->rValue;
            mod->BSIM4v5ldroutGiven = TRUE;
            break;
        case  BSIM4v5_MOD_LDSUB:             
            mod->BSIM4v5ldsub = value->rValue;
            mod->BSIM4v5ldsubGiven = TRUE;
            break;
        case BSIM4v5_MOD_LVTH0:
            mod->BSIM4v5lvth0 = value->rValue;
            mod->BSIM4v5lvth0Given = TRUE;
            break;
        case BSIM4v5_MOD_LUA:
            mod->BSIM4v5lua = value->rValue;
            mod->BSIM4v5luaGiven = TRUE;
            break;
        case BSIM4v5_MOD_LUA1:
            mod->BSIM4v5lua1 = value->rValue;
            mod->BSIM4v5lua1Given = TRUE;
            break;
        case BSIM4v5_MOD_LUB:
            mod->BSIM4v5lub = value->rValue;
            mod->BSIM4v5lubGiven = TRUE;
            break;
        case BSIM4v5_MOD_LUB1:
            mod->BSIM4v5lub1 = value->rValue;
            mod->BSIM4v5lub1Given = TRUE;
            break;
        case BSIM4v5_MOD_LUC:
            mod->BSIM4v5luc = value->rValue;
            mod->BSIM4v5lucGiven = TRUE;
            break;
        case BSIM4v5_MOD_LUC1:
            mod->BSIM4v5luc1 = value->rValue;
            mod->BSIM4v5luc1Given = TRUE;
            break;
        case  BSIM4v5_MOD_LU0 :
            mod->BSIM4v5lu0 = value->rValue;
            mod->BSIM4v5lu0Given = TRUE;
            break;
        case  BSIM4v5_MOD_LUTE :
            mod->BSIM4v5lute = value->rValue;
            mod->BSIM4v5luteGiven = TRUE;
            break;
        case BSIM4v5_MOD_LVOFF:
            mod->BSIM4v5lvoff = value->rValue;
            mod->BSIM4v5lvoffGiven = TRUE;
            break;
        case BSIM4v5_MOD_LTVOFF:
            mod->BSIM4v5ltvoff = value->rValue;
            mod->BSIM4v5ltvoffGiven = TRUE;
            break;
        case BSIM4v5_MOD_LMINV:
            mod->BSIM4v5lminv = value->rValue;
            mod->BSIM4v5lminvGiven = TRUE;
            break;
        case BSIM4v5_MOD_LFPROUT:
            mod->BSIM4v5lfprout = value->rValue;
            mod->BSIM4v5lfproutGiven = TRUE;
            break;
        case BSIM4v5_MOD_LPDITS:
            mod->BSIM4v5lpdits = value->rValue;
            mod->BSIM4v5lpditsGiven = TRUE;
            break;
        case BSIM4v5_MOD_LPDITSD:
            mod->BSIM4v5lpditsd = value->rValue;
            mod->BSIM4v5lpditsdGiven = TRUE;
            break;
        case  BSIM4v5_MOD_LDELTA :
            mod->BSIM4v5ldelta = value->rValue;
            mod->BSIM4v5ldeltaGiven = TRUE;
            break;
        case BSIM4v5_MOD_LRDSW:
            mod->BSIM4v5lrdsw = value->rValue;
            mod->BSIM4v5lrdswGiven = TRUE;
            break;                     
        case BSIM4v5_MOD_LRDW:
            mod->BSIM4v5lrdw = value->rValue;
            mod->BSIM4v5lrdwGiven = TRUE;
            break;
        case BSIM4v5_MOD_LRSW:
            mod->BSIM4v5lrsw = value->rValue;
            mod->BSIM4v5lrswGiven = TRUE;
            break;
        case BSIM4v5_MOD_LPRWB:
            mod->BSIM4v5lprwb = value->rValue;
            mod->BSIM4v5lprwbGiven = TRUE;
            break;                     
        case BSIM4v5_MOD_LPRWG:
            mod->BSIM4v5lprwg = value->rValue;
            mod->BSIM4v5lprwgGiven = TRUE;
            break;                     
        case BSIM4v5_MOD_LPRT:
            mod->BSIM4v5lprt = value->rValue;
            mod->BSIM4v5lprtGiven = TRUE;
            break;                     
        case BSIM4v5_MOD_LETA0:
            mod->BSIM4v5leta0 = value->rValue;
            mod->BSIM4v5leta0Given = TRUE;
            break;                 
        case BSIM4v5_MOD_LETAB:
            mod->BSIM4v5letab = value->rValue;
            mod->BSIM4v5letabGiven = TRUE;
            break;                 
        case BSIM4v5_MOD_LPCLM:
            mod->BSIM4v5lpclm = value->rValue;
            mod->BSIM4v5lpclmGiven = TRUE;
            break;                 
        case BSIM4v5_MOD_LPDIBL1:
            mod->BSIM4v5lpdibl1 = value->rValue;
            mod->BSIM4v5lpdibl1Given = TRUE;
            break;                 
        case BSIM4v5_MOD_LPDIBL2:
            mod->BSIM4v5lpdibl2 = value->rValue;
            mod->BSIM4v5lpdibl2Given = TRUE;
            break;                 
        case BSIM4v5_MOD_LPDIBLB:
            mod->BSIM4v5lpdiblb = value->rValue;
            mod->BSIM4v5lpdiblbGiven = TRUE;
            break;                 
        case BSIM4v5_MOD_LPSCBE1:
            mod->BSIM4v5lpscbe1 = value->rValue;
            mod->BSIM4v5lpscbe1Given = TRUE;
            break;                 
        case BSIM4v5_MOD_LPSCBE2:
            mod->BSIM4v5lpscbe2 = value->rValue;
            mod->BSIM4v5lpscbe2Given = TRUE;
            break;                 
        case BSIM4v5_MOD_LPVAG:
            mod->BSIM4v5lpvag = value->rValue;
            mod->BSIM4v5lpvagGiven = TRUE;
            break;                 
        case  BSIM4v5_MOD_LWR :
            mod->BSIM4v5lwr = value->rValue;
            mod->BSIM4v5lwrGiven = TRUE;
            break;
        case  BSIM4v5_MOD_LDWG :
            mod->BSIM4v5ldwg = value->rValue;
            mod->BSIM4v5ldwgGiven = TRUE;
            break;
        case  BSIM4v5_MOD_LDWB :
            mod->BSIM4v5ldwb = value->rValue;
            mod->BSIM4v5ldwbGiven = TRUE;
            break;
        case  BSIM4v5_MOD_LB0 :
            mod->BSIM4v5lb0 = value->rValue;
            mod->BSIM4v5lb0Given = TRUE;
            break;
        case  BSIM4v5_MOD_LB1 :
            mod->BSIM4v5lb1 = value->rValue;
            mod->BSIM4v5lb1Given = TRUE;
            break;
        case  BSIM4v5_MOD_LALPHA0 :
            mod->BSIM4v5lalpha0 = value->rValue;
            mod->BSIM4v5lalpha0Given = TRUE;
            break;
        case  BSIM4v5_MOD_LALPHA1 :
            mod->BSIM4v5lalpha1 = value->rValue;
            mod->BSIM4v5lalpha1Given = TRUE;
            break;
        case  BSIM4v5_MOD_LBETA0 :
            mod->BSIM4v5lbeta0 = value->rValue;
            mod->BSIM4v5lbeta0Given = TRUE;
            break;
        case  BSIM4v5_MOD_LAGIDL :
            mod->BSIM4v5lagidl = value->rValue;
            mod->BSIM4v5lagidlGiven = TRUE;
            break;
        case  BSIM4v5_MOD_LBGIDL :
            mod->BSIM4v5lbgidl = value->rValue;
            mod->BSIM4v5lbgidlGiven = TRUE;
            break;
        case  BSIM4v5_MOD_LCGIDL :
            mod->BSIM4v5lcgidl = value->rValue;
            mod->BSIM4v5lcgidlGiven = TRUE;
            break;
        case  BSIM4v5_MOD_LPHIN :
            mod->BSIM4v5lphin = value->rValue;
            mod->BSIM4v5lphinGiven = TRUE;
            break;
        case  BSIM4v5_MOD_LEGIDL :
            mod->BSIM4v5legidl = value->rValue;
            mod->BSIM4v5legidlGiven = TRUE;
            break;
        case  BSIM4v5_MOD_LAIGC :
            mod->BSIM4v5laigc = value->rValue;
            mod->BSIM4v5laigcGiven = TRUE;
            break;
        case  BSIM4v5_MOD_LBIGC :
            mod->BSIM4v5lbigc = value->rValue;
            mod->BSIM4v5lbigcGiven = TRUE;
            break;
        case  BSIM4v5_MOD_LCIGC :
            mod->BSIM4v5lcigc = value->rValue;
            mod->BSIM4v5lcigcGiven = TRUE;
            break;
        case  BSIM4v5_MOD_LAIGSD :
            mod->BSIM4v5laigsd = value->rValue;
            mod->BSIM4v5laigsdGiven = TRUE;
            break;
        case  BSIM4v5_MOD_LBIGSD :
            mod->BSIM4v5lbigsd = value->rValue;
            mod->BSIM4v5lbigsdGiven = TRUE;
            break;
        case  BSIM4v5_MOD_LCIGSD :
            mod->BSIM4v5lcigsd = value->rValue;
            mod->BSIM4v5lcigsdGiven = TRUE;
            break;
        case  BSIM4v5_MOD_LAIGBACC :
            mod->BSIM4v5laigbacc = value->rValue;
            mod->BSIM4v5laigbaccGiven = TRUE;
            break;
        case  BSIM4v5_MOD_LBIGBACC :
            mod->BSIM4v5lbigbacc = value->rValue;
            mod->BSIM4v5lbigbaccGiven = TRUE;
            break;
        case  BSIM4v5_MOD_LCIGBACC :
            mod->BSIM4v5lcigbacc = value->rValue;
            mod->BSIM4v5lcigbaccGiven = TRUE;
            break;
        case  BSIM4v5_MOD_LAIGBINV :
            mod->BSIM4v5laigbinv = value->rValue;
            mod->BSIM4v5laigbinvGiven = TRUE;
            break;
        case  BSIM4v5_MOD_LBIGBINV :
            mod->BSIM4v5lbigbinv = value->rValue;
            mod->BSIM4v5lbigbinvGiven = TRUE;
            break;
        case  BSIM4v5_MOD_LCIGBINV :
            mod->BSIM4v5lcigbinv = value->rValue;
            mod->BSIM4v5lcigbinvGiven = TRUE;
            break;
        case  BSIM4v5_MOD_LNIGC :
            mod->BSIM4v5lnigc = value->rValue;
            mod->BSIM4v5lnigcGiven = TRUE;
            break;
        case  BSIM4v5_MOD_LNIGBINV :
            mod->BSIM4v5lnigbinv = value->rValue;
            mod->BSIM4v5lnigbinvGiven = TRUE;
            break;
        case  BSIM4v5_MOD_LNIGBACC :
            mod->BSIM4v5lnigbacc = value->rValue;
            mod->BSIM4v5lnigbaccGiven = TRUE;
            break;
        case  BSIM4v5_MOD_LNTOX :
            mod->BSIM4v5lntox = value->rValue;
            mod->BSIM4v5lntoxGiven = TRUE;
            break;
        case  BSIM4v5_MOD_LEIGBINV :
            mod->BSIM4v5leigbinv = value->rValue;
            mod->BSIM4v5leigbinvGiven = TRUE;
            break;
        case  BSIM4v5_MOD_LPIGCD :
            mod->BSIM4v5lpigcd = value->rValue;
            mod->BSIM4v5lpigcdGiven = TRUE;
            break;
        case  BSIM4v5_MOD_LPOXEDGE :
            mod->BSIM4v5lpoxedge = value->rValue;
            mod->BSIM4v5lpoxedgeGiven = TRUE;
            break;
        case  BSIM4v5_MOD_LXRCRG1 :
            mod->BSIM4v5lxrcrg1 = value->rValue;
            mod->BSIM4v5lxrcrg1Given = TRUE;
            break;
        case  BSIM4v5_MOD_LXRCRG2 :
            mod->BSIM4v5lxrcrg2 = value->rValue;
            mod->BSIM4v5lxrcrg2Given = TRUE;
            break;
        case  BSIM4v5_MOD_LLAMBDA :
            mod->BSIM4v5llambda = value->rValue;
            mod->BSIM4v5llambdaGiven = TRUE;
            break;
        case  BSIM4v5_MOD_LVTL :
            mod->BSIM4v5lvtl = value->rValue;
            mod->BSIM4v5lvtlGiven = TRUE;
            break;
        case  BSIM4v5_MOD_LXN:
            mod->BSIM4v5lxn = value->rValue;
            mod->BSIM4v5lxnGiven = TRUE;
            break;
        case  BSIM4v5_MOD_LVFBSDOFF:
            mod->BSIM4v5lvfbsdoff = value->rValue;
            mod->BSIM4v5lvfbsdoffGiven = TRUE;
            break;
        case  BSIM4v5_MOD_LTVFBSDOFF:
            mod->BSIM4v5ltvfbsdoff = value->rValue;
            mod->BSIM4v5ltvfbsdoffGiven = TRUE;
            break;
        case  BSIM4v5_MOD_LEU :
            mod->BSIM4v5leu = value->rValue;
            mod->BSIM4v5leuGiven = TRUE;
            break;
        case  BSIM4v5_MOD_LVFB :
            mod->BSIM4v5lvfb = value->rValue;
            mod->BSIM4v5lvfbGiven = TRUE;
            break;
        case  BSIM4v5_MOD_LCGSL :
            mod->BSIM4v5lcgsl = value->rValue;
            mod->BSIM4v5lcgslGiven = TRUE;
            break;
        case  BSIM4v5_MOD_LCGDL :
            mod->BSIM4v5lcgdl = value->rValue;
            mod->BSIM4v5lcgdlGiven = TRUE;
            break;
        case  BSIM4v5_MOD_LCKAPPAS :
            mod->BSIM4v5lckappas = value->rValue;
            mod->BSIM4v5lckappasGiven = TRUE;
            break;
        case  BSIM4v5_MOD_LCKAPPAD :
            mod->BSIM4v5lckappad = value->rValue;
            mod->BSIM4v5lckappadGiven = TRUE;
            break;
        case  BSIM4v5_MOD_LCF :
            mod->BSIM4v5lcf = value->rValue;
            mod->BSIM4v5lcfGiven = TRUE;
            break;
        case  BSIM4v5_MOD_LCLC :
            mod->BSIM4v5lclc = value->rValue;
            mod->BSIM4v5lclcGiven = TRUE;
            break;
        case  BSIM4v5_MOD_LCLE :
            mod->BSIM4v5lcle = value->rValue;
            mod->BSIM4v5lcleGiven = TRUE;
            break;
        case  BSIM4v5_MOD_LVFBCV :
            mod->BSIM4v5lvfbcv = value->rValue;
            mod->BSIM4v5lvfbcvGiven = TRUE;
            break;
        case  BSIM4v5_MOD_LACDE :
            mod->BSIM4v5lacde = value->rValue;
            mod->BSIM4v5lacdeGiven = TRUE;
            break;
        case  BSIM4v5_MOD_LMOIN :
            mod->BSIM4v5lmoin = value->rValue;
            mod->BSIM4v5lmoinGiven = TRUE;
            break;
        case  BSIM4v5_MOD_LNOFF :
            mod->BSIM4v5lnoff = value->rValue;
            mod->BSIM4v5lnoffGiven = TRUE;
            break;
        case  BSIM4v5_MOD_LVOFFCV :
            mod->BSIM4v5lvoffcv = value->rValue;
            mod->BSIM4v5lvoffcvGiven = TRUE;
            break;

	/* Width dependence */
        case  BSIM4v5_MOD_WCDSC :
            mod->BSIM4v5wcdsc = value->rValue;
            mod->BSIM4v5wcdscGiven = TRUE;
            break;
       
       
         case  BSIM4v5_MOD_WCDSCB :
            mod->BSIM4v5wcdscb = value->rValue;
            mod->BSIM4v5wcdscbGiven = TRUE;
            break;
         case  BSIM4v5_MOD_WCDSCD :
            mod->BSIM4v5wcdscd = value->rValue;
            mod->BSIM4v5wcdscdGiven = TRUE;
            break;
        case  BSIM4v5_MOD_WCIT :
            mod->BSIM4v5wcit = value->rValue;
            mod->BSIM4v5wcitGiven = TRUE;
            break;
        case  BSIM4v5_MOD_WNFACTOR :
            mod->BSIM4v5wnfactor = value->rValue;
            mod->BSIM4v5wnfactorGiven = TRUE;
            break;
        case BSIM4v5_MOD_WXJ:
            mod->BSIM4v5wxj = value->rValue;
            mod->BSIM4v5wxjGiven = TRUE;
            break;
        case BSIM4v5_MOD_WVSAT:
            mod->BSIM4v5wvsat = value->rValue;
            mod->BSIM4v5wvsatGiven = TRUE;
            break;


        case BSIM4v5_MOD_WA0:
            mod->BSIM4v5wa0 = value->rValue;
            mod->BSIM4v5wa0Given = TRUE;
            break;
        case BSIM4v5_MOD_WAGS:
            mod->BSIM4v5wags = value->rValue;
            mod->BSIM4v5wagsGiven = TRUE;
            break;
        case BSIM4v5_MOD_WA1:
            mod->BSIM4v5wa1 = value->rValue;
            mod->BSIM4v5wa1Given = TRUE;
            break;
        case BSIM4v5_MOD_WA2:
            mod->BSIM4v5wa2 = value->rValue;
            mod->BSIM4v5wa2Given = TRUE;
            break;
        case BSIM4v5_MOD_WAT:
            mod->BSIM4v5wat = value->rValue;
            mod->BSIM4v5watGiven = TRUE;
            break;
        case BSIM4v5_MOD_WKETA:
            mod->BSIM4v5wketa = value->rValue;
            mod->BSIM4v5wketaGiven = TRUE;
            break;    
        case BSIM4v5_MOD_WNSUB:
            mod->BSIM4v5wnsub = value->rValue;
            mod->BSIM4v5wnsubGiven = TRUE;
            break;
        case BSIM4v5_MOD_WNDEP:
            mod->BSIM4v5wndep = value->rValue;
            mod->BSIM4v5wndepGiven = TRUE;
	    if (mod->BSIM4v5wndep > 1.0e20)
		mod->BSIM4v5wndep *= 1.0e-6;
            break;
        case BSIM4v5_MOD_WNSD:
            mod->BSIM4v5wnsd = value->rValue;
            mod->BSIM4v5wnsdGiven = TRUE;
            if (mod->BSIM4v5wnsd > 1.0e23)
                mod->BSIM4v5wnsd *= 1.0e-6;
            break;
        case BSIM4v5_MOD_WNGATE:
            mod->BSIM4v5wngate = value->rValue;
            mod->BSIM4v5wngateGiven = TRUE;
	    if (mod->BSIM4v5wngate > 1.0e23)
		mod->BSIM4v5wngate *= 1.0e-6;
            break;
        case BSIM4v5_MOD_WGAMMA1:
            mod->BSIM4v5wgamma1 = value->rValue;
            mod->BSIM4v5wgamma1Given = TRUE;
            break;
        case BSIM4v5_MOD_WGAMMA2:
            mod->BSIM4v5wgamma2 = value->rValue;
            mod->BSIM4v5wgamma2Given = TRUE;
            break;
        case BSIM4v5_MOD_WVBX:
            mod->BSIM4v5wvbx = value->rValue;
            mod->BSIM4v5wvbxGiven = TRUE;
            break;
        case BSIM4v5_MOD_WVBM:
            mod->BSIM4v5wvbm = value->rValue;
            mod->BSIM4v5wvbmGiven = TRUE;
            break;
        case BSIM4v5_MOD_WXT:
            mod->BSIM4v5wxt = value->rValue;
            mod->BSIM4v5wxtGiven = TRUE;
            break;
        case  BSIM4v5_MOD_WK1:
            mod->BSIM4v5wk1 = value->rValue;
            mod->BSIM4v5wk1Given = TRUE;
            break;
        case  BSIM4v5_MOD_WKT1:
            mod->BSIM4v5wkt1 = value->rValue;
            mod->BSIM4v5wkt1Given = TRUE;
            break;
        case  BSIM4v5_MOD_WKT1L:
            mod->BSIM4v5wkt1l = value->rValue;
            mod->BSIM4v5wkt1lGiven = TRUE;
            break;
        case  BSIM4v5_MOD_WKT2:
            mod->BSIM4v5wkt2 = value->rValue;
            mod->BSIM4v5wkt2Given = TRUE;
            break;
        case  BSIM4v5_MOD_WK2:
            mod->BSIM4v5wk2 = value->rValue;
            mod->BSIM4v5wk2Given = TRUE;
            break;
        case  BSIM4v5_MOD_WK3:
            mod->BSIM4v5wk3 = value->rValue;
            mod->BSIM4v5wk3Given = TRUE;
            break;
        case  BSIM4v5_MOD_WK3B:
            mod->BSIM4v5wk3b = value->rValue;
            mod->BSIM4v5wk3bGiven = TRUE;
            break;
        case  BSIM4v5_MOD_WLPE0:
            mod->BSIM4v5wlpe0 = value->rValue;
            mod->BSIM4v5wlpe0Given = TRUE;
            break;
        case  BSIM4v5_MOD_WLPEB:
            mod->BSIM4v5wlpeb = value->rValue;
            mod->BSIM4v5wlpebGiven = TRUE;
            break;
        case  BSIM4v5_MOD_WDVTP0:
            mod->BSIM4v5wdvtp0 = value->rValue;
            mod->BSIM4v5wdvtp0Given = TRUE;
            break;
        case  BSIM4v5_MOD_WDVTP1:
            mod->BSIM4v5wdvtp1 = value->rValue;
            mod->BSIM4v5wdvtp1Given = TRUE;
            break;
        case  BSIM4v5_MOD_WW0:
            mod->BSIM4v5ww0 = value->rValue;
            mod->BSIM4v5ww0Given = TRUE;
            break;
        case  BSIM4v5_MOD_WDVT0:               
            mod->BSIM4v5wdvt0 = value->rValue;
            mod->BSIM4v5wdvt0Given = TRUE;
            break;
        case  BSIM4v5_MOD_WDVT1:             
            mod->BSIM4v5wdvt1 = value->rValue;
            mod->BSIM4v5wdvt1Given = TRUE;
            break;
        case  BSIM4v5_MOD_WDVT2:             
            mod->BSIM4v5wdvt2 = value->rValue;
            mod->BSIM4v5wdvt2Given = TRUE;
            break;
        case  BSIM4v5_MOD_WDVT0W:               
            mod->BSIM4v5wdvt0w = value->rValue;
            mod->BSIM4v5wdvt0wGiven = TRUE;
            break;
        case  BSIM4v5_MOD_WDVT1W:             
            mod->BSIM4v5wdvt1w = value->rValue;
            mod->BSIM4v5wdvt1wGiven = TRUE;
            break;
        case  BSIM4v5_MOD_WDVT2W:             
            mod->BSIM4v5wdvt2w = value->rValue;
            mod->BSIM4v5wdvt2wGiven = TRUE;
            break;
        case  BSIM4v5_MOD_WDROUT:             
            mod->BSIM4v5wdrout = value->rValue;
            mod->BSIM4v5wdroutGiven = TRUE;
            break;
        case  BSIM4v5_MOD_WDSUB:             
            mod->BSIM4v5wdsub = value->rValue;
            mod->BSIM4v5wdsubGiven = TRUE;
            break;
        case BSIM4v5_MOD_WVTH0:
            mod->BSIM4v5wvth0 = value->rValue;
            mod->BSIM4v5wvth0Given = TRUE;
            break;
        case BSIM4v5_MOD_WUA:
            mod->BSIM4v5wua = value->rValue;
            mod->BSIM4v5wuaGiven = TRUE;
            break;
        case BSIM4v5_MOD_WUA1:
            mod->BSIM4v5wua1 = value->rValue;
            mod->BSIM4v5wua1Given = TRUE;
            break;
        case BSIM4v5_MOD_WUB:
            mod->BSIM4v5wub = value->rValue;
            mod->BSIM4v5wubGiven = TRUE;
            break;
        case BSIM4v5_MOD_WUB1:
            mod->BSIM4v5wub1 = value->rValue;
            mod->BSIM4v5wub1Given = TRUE;
            break;
        case BSIM4v5_MOD_WUC:
            mod->BSIM4v5wuc = value->rValue;
            mod->BSIM4v5wucGiven = TRUE;
            break;
        case BSIM4v5_MOD_WUC1:
            mod->BSIM4v5wuc1 = value->rValue;
            mod->BSIM4v5wuc1Given = TRUE;
            break;
        case  BSIM4v5_MOD_WU0 :
            mod->BSIM4v5wu0 = value->rValue;
            mod->BSIM4v5wu0Given = TRUE;
            break;
        case  BSIM4v5_MOD_WUTE :
            mod->BSIM4v5wute = value->rValue;
            mod->BSIM4v5wuteGiven = TRUE;
            break;
        case BSIM4v5_MOD_WVOFF:
            mod->BSIM4v5wvoff = value->rValue;
            mod->BSIM4v5wvoffGiven = TRUE;
            break;
        case BSIM4v5_MOD_WTVOFF:
            mod->BSIM4v5wtvoff = value->rValue;
            mod->BSIM4v5wtvoffGiven = TRUE;
            break;
        case BSIM4v5_MOD_WMINV:
            mod->BSIM4v5wminv = value->rValue;
            mod->BSIM4v5wminvGiven = TRUE;
            break;
        case BSIM4v5_MOD_WFPROUT:
            mod->BSIM4v5wfprout = value->rValue;
            mod->BSIM4v5wfproutGiven = TRUE;
            break;
        case BSIM4v5_MOD_WPDITS:
            mod->BSIM4v5wpdits = value->rValue;
            mod->BSIM4v5wpditsGiven = TRUE;
            break;
        case BSIM4v5_MOD_WPDITSD:
            mod->BSIM4v5wpditsd = value->rValue;
            mod->BSIM4v5wpditsdGiven = TRUE;
            break;
        case  BSIM4v5_MOD_WDELTA :
            mod->BSIM4v5wdelta = value->rValue;
            mod->BSIM4v5wdeltaGiven = TRUE;
            break;
        case BSIM4v5_MOD_WRDSW:
            mod->BSIM4v5wrdsw = value->rValue;
            mod->BSIM4v5wrdswGiven = TRUE;
            break;                     
        case BSIM4v5_MOD_WRDW:
            mod->BSIM4v5wrdw = value->rValue;
            mod->BSIM4v5wrdwGiven = TRUE;
            break;
        case BSIM4v5_MOD_WRSW:
            mod->BSIM4v5wrsw = value->rValue;
            mod->BSIM4v5wrswGiven = TRUE;
            break;
        case BSIM4v5_MOD_WPRWB:
            mod->BSIM4v5wprwb = value->rValue;
            mod->BSIM4v5wprwbGiven = TRUE;
            break;                     
        case BSIM4v5_MOD_WPRWG:
            mod->BSIM4v5wprwg = value->rValue;
            mod->BSIM4v5wprwgGiven = TRUE;
            break;                     
        case BSIM4v5_MOD_WPRT:
            mod->BSIM4v5wprt = value->rValue;
            mod->BSIM4v5wprtGiven = TRUE;
            break;                     
        case BSIM4v5_MOD_WETA0:
            mod->BSIM4v5weta0 = value->rValue;
            mod->BSIM4v5weta0Given = TRUE;
            break;                 
        case BSIM4v5_MOD_WETAB:
            mod->BSIM4v5wetab = value->rValue;
            mod->BSIM4v5wetabGiven = TRUE;
            break;                 
        case BSIM4v5_MOD_WPCLM:
            mod->BSIM4v5wpclm = value->rValue;
            mod->BSIM4v5wpclmGiven = TRUE;
            break;                 
        case BSIM4v5_MOD_WPDIBL1:
            mod->BSIM4v5wpdibl1 = value->rValue;
            mod->BSIM4v5wpdibl1Given = TRUE;
            break;                 
        case BSIM4v5_MOD_WPDIBL2:
            mod->BSIM4v5wpdibl2 = value->rValue;
            mod->BSIM4v5wpdibl2Given = TRUE;
            break;                 
        case BSIM4v5_MOD_WPDIBLB:
            mod->BSIM4v5wpdiblb = value->rValue;
            mod->BSIM4v5wpdiblbGiven = TRUE;
            break;                 
        case BSIM4v5_MOD_WPSCBE1:
            mod->BSIM4v5wpscbe1 = value->rValue;
            mod->BSIM4v5wpscbe1Given = TRUE;
            break;                 
        case BSIM4v5_MOD_WPSCBE2:
            mod->BSIM4v5wpscbe2 = value->rValue;
            mod->BSIM4v5wpscbe2Given = TRUE;
            break;                 
        case BSIM4v5_MOD_WPVAG:
            mod->BSIM4v5wpvag = value->rValue;
            mod->BSIM4v5wpvagGiven = TRUE;
            break;                 
        case  BSIM4v5_MOD_WWR :
            mod->BSIM4v5wwr = value->rValue;
            mod->BSIM4v5wwrGiven = TRUE;
            break;
        case  BSIM4v5_MOD_WDWG :
            mod->BSIM4v5wdwg = value->rValue;
            mod->BSIM4v5wdwgGiven = TRUE;
            break;
        case  BSIM4v5_MOD_WDWB :
            mod->BSIM4v5wdwb = value->rValue;
            mod->BSIM4v5wdwbGiven = TRUE;
            break;
        case  BSIM4v5_MOD_WB0 :
            mod->BSIM4v5wb0 = value->rValue;
            mod->BSIM4v5wb0Given = TRUE;
            break;
        case  BSIM4v5_MOD_WB1 :
            mod->BSIM4v5wb1 = value->rValue;
            mod->BSIM4v5wb1Given = TRUE;
            break;
        case  BSIM4v5_MOD_WALPHA0 :
            mod->BSIM4v5walpha0 = value->rValue;
            mod->BSIM4v5walpha0Given = TRUE;
            break;
        case  BSIM4v5_MOD_WALPHA1 :
            mod->BSIM4v5walpha1 = value->rValue;
            mod->BSIM4v5walpha1Given = TRUE;
            break;
        case  BSIM4v5_MOD_WBETA0 :
            mod->BSIM4v5wbeta0 = value->rValue;
            mod->BSIM4v5wbeta0Given = TRUE;
            break;
        case  BSIM4v5_MOD_WAGIDL :
            mod->BSIM4v5wagidl = value->rValue;
            mod->BSIM4v5wagidlGiven = TRUE;
            break;
        case  BSIM4v5_MOD_WBGIDL :
            mod->BSIM4v5wbgidl = value->rValue;
            mod->BSIM4v5wbgidlGiven = TRUE;
            break;
        case  BSIM4v5_MOD_WCGIDL :
            mod->BSIM4v5wcgidl = value->rValue;
            mod->BSIM4v5wcgidlGiven = TRUE;
            break;
        case  BSIM4v5_MOD_WPHIN :
            mod->BSIM4v5wphin = value->rValue;
            mod->BSIM4v5wphinGiven = TRUE;
            break;
        case  BSIM4v5_MOD_WEGIDL :
            mod->BSIM4v5wegidl = value->rValue;
            mod->BSIM4v5wegidlGiven = TRUE;
            break;
        case  BSIM4v5_MOD_WAIGC :
            mod->BSIM4v5waigc = value->rValue;
            mod->BSIM4v5waigcGiven = TRUE;
            break;
        case  BSIM4v5_MOD_WBIGC :
            mod->BSIM4v5wbigc = value->rValue;
            mod->BSIM4v5wbigcGiven = TRUE;
            break;
        case  BSIM4v5_MOD_WCIGC :
            mod->BSIM4v5wcigc = value->rValue;
            mod->BSIM4v5wcigcGiven = TRUE;
            break;
        case  BSIM4v5_MOD_WAIGSD :
            mod->BSIM4v5waigsd = value->rValue;
            mod->BSIM4v5waigsdGiven = TRUE;
            break;
        case  BSIM4v5_MOD_WBIGSD :
            mod->BSIM4v5wbigsd = value->rValue;
            mod->BSIM4v5wbigsdGiven = TRUE;
            break;
        case  BSIM4v5_MOD_WCIGSD :
            mod->BSIM4v5wcigsd = value->rValue;
            mod->BSIM4v5wcigsdGiven = TRUE;
            break;
        case  BSIM4v5_MOD_WAIGBACC :
            mod->BSIM4v5waigbacc = value->rValue;
            mod->BSIM4v5waigbaccGiven = TRUE;
            break;
        case  BSIM4v5_MOD_WBIGBACC :
            mod->BSIM4v5wbigbacc = value->rValue;
            mod->BSIM4v5wbigbaccGiven = TRUE;
            break;
        case  BSIM4v5_MOD_WCIGBACC :
            mod->BSIM4v5wcigbacc = value->rValue;
            mod->BSIM4v5wcigbaccGiven = TRUE;
            break;
        case  BSIM4v5_MOD_WAIGBINV :
            mod->BSIM4v5waigbinv = value->rValue;
            mod->BSIM4v5waigbinvGiven = TRUE;
            break;
        case  BSIM4v5_MOD_WBIGBINV :
            mod->BSIM4v5wbigbinv = value->rValue;
            mod->BSIM4v5wbigbinvGiven = TRUE;
            break;
        case  BSIM4v5_MOD_WCIGBINV :
            mod->BSIM4v5wcigbinv = value->rValue;
            mod->BSIM4v5wcigbinvGiven = TRUE;
            break;
        case  BSIM4v5_MOD_WNIGC :
            mod->BSIM4v5wnigc = value->rValue;
            mod->BSIM4v5wnigcGiven = TRUE;
            break;
        case  BSIM4v5_MOD_WNIGBINV :
            mod->BSIM4v5wnigbinv = value->rValue;
            mod->BSIM4v5wnigbinvGiven = TRUE;
            break;
        case  BSIM4v5_MOD_WNIGBACC :
            mod->BSIM4v5wnigbacc = value->rValue;
            mod->BSIM4v5wnigbaccGiven = TRUE;
            break;
        case  BSIM4v5_MOD_WNTOX :
            mod->BSIM4v5wntox = value->rValue;
            mod->BSIM4v5wntoxGiven = TRUE;
            break;
        case  BSIM4v5_MOD_WEIGBINV :
            mod->BSIM4v5weigbinv = value->rValue;
            mod->BSIM4v5weigbinvGiven = TRUE;
            break;
        case  BSIM4v5_MOD_WPIGCD :
            mod->BSIM4v5wpigcd = value->rValue;
            mod->BSIM4v5wpigcdGiven = TRUE;
            break;
        case  BSIM4v5_MOD_WPOXEDGE :
            mod->BSIM4v5wpoxedge = value->rValue;
            mod->BSIM4v5wpoxedgeGiven = TRUE;
            break;
        case  BSIM4v5_MOD_WXRCRG1 :
            mod->BSIM4v5wxrcrg1 = value->rValue;
            mod->BSIM4v5wxrcrg1Given = TRUE;
            break;
        case  BSIM4v5_MOD_WXRCRG2 :
            mod->BSIM4v5wxrcrg2 = value->rValue;
            mod->BSIM4v5wxrcrg2Given = TRUE;
            break;
        case  BSIM4v5_MOD_WLAMBDA :
            mod->BSIM4v5wlambda = value->rValue;
            mod->BSIM4v5wlambdaGiven = TRUE;
            break;
        case  BSIM4v5_MOD_WVTL :
            mod->BSIM4v5wvtl = value->rValue;
            mod->BSIM4v5wvtlGiven = TRUE;
            break;
        case  BSIM4v5_MOD_WXN:
            mod->BSIM4v5wxn = value->rValue;
            mod->BSIM4v5wxnGiven = TRUE;
            break;
        case  BSIM4v5_MOD_WVFBSDOFF:
            mod->BSIM4v5wvfbsdoff = value->rValue;
            mod->BSIM4v5wvfbsdoffGiven = TRUE;
            break;
        case  BSIM4v5_MOD_WTVFBSDOFF:
            mod->BSIM4v5wtvfbsdoff = value->rValue;
            mod->BSIM4v5wtvfbsdoffGiven = TRUE;
            break;
        case  BSIM4v5_MOD_WEU :
            mod->BSIM4v5weu = value->rValue;
            mod->BSIM4v5weuGiven = TRUE;
            break;
        case  BSIM4v5_MOD_WVFB :
            mod->BSIM4v5wvfb = value->rValue;
            mod->BSIM4v5wvfbGiven = TRUE;
            break;
        case  BSIM4v5_MOD_WCGSL :
            mod->BSIM4v5wcgsl = value->rValue;
            mod->BSIM4v5wcgslGiven = TRUE;
            break;
        case  BSIM4v5_MOD_WCGDL :
            mod->BSIM4v5wcgdl = value->rValue;
            mod->BSIM4v5wcgdlGiven = TRUE;
            break;
        case  BSIM4v5_MOD_WCKAPPAS :
            mod->BSIM4v5wckappas = value->rValue;
            mod->BSIM4v5wckappasGiven = TRUE;
            break;
        case  BSIM4v5_MOD_WCKAPPAD :
            mod->BSIM4v5wckappad = value->rValue;
            mod->BSIM4v5wckappadGiven = TRUE;
            break;
        case  BSIM4v5_MOD_WCF :
            mod->BSIM4v5wcf = value->rValue;
            mod->BSIM4v5wcfGiven = TRUE;
            break;
        case  BSIM4v5_MOD_WCLC :
            mod->BSIM4v5wclc = value->rValue;
            mod->BSIM4v5wclcGiven = TRUE;
            break;
        case  BSIM4v5_MOD_WCLE :
            mod->BSIM4v5wcle = value->rValue;
            mod->BSIM4v5wcleGiven = TRUE;
            break;
        case  BSIM4v5_MOD_WVFBCV :
            mod->BSIM4v5wvfbcv = value->rValue;
            mod->BSIM4v5wvfbcvGiven = TRUE;
            break;
        case  BSIM4v5_MOD_WACDE :
            mod->BSIM4v5wacde = value->rValue;
            mod->BSIM4v5wacdeGiven = TRUE;
            break;
        case  BSIM4v5_MOD_WMOIN :
            mod->BSIM4v5wmoin = value->rValue;
            mod->BSIM4v5wmoinGiven = TRUE;
            break;
        case  BSIM4v5_MOD_WNOFF :
            mod->BSIM4v5wnoff = value->rValue;
            mod->BSIM4v5wnoffGiven = TRUE;
            break;
        case  BSIM4v5_MOD_WVOFFCV :
            mod->BSIM4v5wvoffcv = value->rValue;
            mod->BSIM4v5wvoffcvGiven = TRUE;
            break;

	/* Cross-term dependence */
        case  BSIM4v5_MOD_PCDSC :
            mod->BSIM4v5pcdsc = value->rValue;
            mod->BSIM4v5pcdscGiven = TRUE;
            break;


        case  BSIM4v5_MOD_PCDSCB :
            mod->BSIM4v5pcdscb = value->rValue;
            mod->BSIM4v5pcdscbGiven = TRUE;
            break;
        case  BSIM4v5_MOD_PCDSCD :
            mod->BSIM4v5pcdscd = value->rValue;
            mod->BSIM4v5pcdscdGiven = TRUE;
            break;
        case  BSIM4v5_MOD_PCIT :
            mod->BSIM4v5pcit = value->rValue;
            mod->BSIM4v5pcitGiven = TRUE;
            break;
        case  BSIM4v5_MOD_PNFACTOR :
            mod->BSIM4v5pnfactor = value->rValue;
            mod->BSIM4v5pnfactorGiven = TRUE;
            break;
        case BSIM4v5_MOD_PXJ:
            mod->BSIM4v5pxj = value->rValue;
            mod->BSIM4v5pxjGiven = TRUE;
            break;
        case BSIM4v5_MOD_PVSAT:
            mod->BSIM4v5pvsat = value->rValue;
            mod->BSIM4v5pvsatGiven = TRUE;
            break;


        case BSIM4v5_MOD_PA0:
            mod->BSIM4v5pa0 = value->rValue;
            mod->BSIM4v5pa0Given = TRUE;
            break;
        case BSIM4v5_MOD_PAGS:
            mod->BSIM4v5pags = value->rValue;
            mod->BSIM4v5pagsGiven = TRUE;
            break;
        case BSIM4v5_MOD_PA1:
            mod->BSIM4v5pa1 = value->rValue;
            mod->BSIM4v5pa1Given = TRUE;
            break;
        case BSIM4v5_MOD_PA2:
            mod->BSIM4v5pa2 = value->rValue;
            mod->BSIM4v5pa2Given = TRUE;
            break;
        case BSIM4v5_MOD_PAT:
            mod->BSIM4v5pat = value->rValue;
            mod->BSIM4v5patGiven = TRUE;
            break;
        case BSIM4v5_MOD_PKETA:
            mod->BSIM4v5pketa = value->rValue;
            mod->BSIM4v5pketaGiven = TRUE;
            break;    
        case BSIM4v5_MOD_PNSUB:
            mod->BSIM4v5pnsub = value->rValue;
            mod->BSIM4v5pnsubGiven = TRUE;
            break;
        case BSIM4v5_MOD_PNDEP:
            mod->BSIM4v5pndep = value->rValue;
            mod->BSIM4v5pndepGiven = TRUE;
	    if (mod->BSIM4v5pndep > 1.0e20)
		mod->BSIM4v5pndep *= 1.0e-6;
            break;
        case BSIM4v5_MOD_PNSD:
            mod->BSIM4v5pnsd = value->rValue;
            mod->BSIM4v5pnsdGiven = TRUE;
            if (mod->BSIM4v5pnsd > 1.0e23)
                mod->BSIM4v5pnsd *= 1.0e-6;
            break;
        case BSIM4v5_MOD_PNGATE:
            mod->BSIM4v5pngate = value->rValue;
            mod->BSIM4v5pngateGiven = TRUE;
	    if (mod->BSIM4v5pngate > 1.0e23)
		mod->BSIM4v5pngate *= 1.0e-6;
            break;
        case BSIM4v5_MOD_PGAMMA1:
            mod->BSIM4v5pgamma1 = value->rValue;
            mod->BSIM4v5pgamma1Given = TRUE;
            break;
        case BSIM4v5_MOD_PGAMMA2:
            mod->BSIM4v5pgamma2 = value->rValue;
            mod->BSIM4v5pgamma2Given = TRUE;
            break;
        case BSIM4v5_MOD_PVBX:
            mod->BSIM4v5pvbx = value->rValue;
            mod->BSIM4v5pvbxGiven = TRUE;
            break;
        case BSIM4v5_MOD_PVBM:
            mod->BSIM4v5pvbm = value->rValue;
            mod->BSIM4v5pvbmGiven = TRUE;
            break;
        case BSIM4v5_MOD_PXT:
            mod->BSIM4v5pxt = value->rValue;
            mod->BSIM4v5pxtGiven = TRUE;
            break;
        case  BSIM4v5_MOD_PK1:
            mod->BSIM4v5pk1 = value->rValue;
            mod->BSIM4v5pk1Given = TRUE;
            break;
        case  BSIM4v5_MOD_PKT1:
            mod->BSIM4v5pkt1 = value->rValue;
            mod->BSIM4v5pkt1Given = TRUE;
            break;
        case  BSIM4v5_MOD_PKT1L:
            mod->BSIM4v5pkt1l = value->rValue;
            mod->BSIM4v5pkt1lGiven = TRUE;
            break;
        case  BSIM4v5_MOD_PKT2:
            mod->BSIM4v5pkt2 = value->rValue;
            mod->BSIM4v5pkt2Given = TRUE;
            break;
        case  BSIM4v5_MOD_PK2:
            mod->BSIM4v5pk2 = value->rValue;
            mod->BSIM4v5pk2Given = TRUE;
            break;
        case  BSIM4v5_MOD_PK3:
            mod->BSIM4v5pk3 = value->rValue;
            mod->BSIM4v5pk3Given = TRUE;
            break;
        case  BSIM4v5_MOD_PK3B:
            mod->BSIM4v5pk3b = value->rValue;
            mod->BSIM4v5pk3bGiven = TRUE;
            break;
        case  BSIM4v5_MOD_PLPE0:
            mod->BSIM4v5plpe0 = value->rValue;
            mod->BSIM4v5plpe0Given = TRUE;
            break;
        case  BSIM4v5_MOD_PLPEB:
            mod->BSIM4v5plpeb = value->rValue;
            mod->BSIM4v5plpebGiven = TRUE;
            break;
        case  BSIM4v5_MOD_PDVTP0:
            mod->BSIM4v5pdvtp0 = value->rValue;
            mod->BSIM4v5pdvtp0Given = TRUE;
            break;
        case  BSIM4v5_MOD_PDVTP1:
            mod->BSIM4v5pdvtp1 = value->rValue;
            mod->BSIM4v5pdvtp1Given = TRUE;
            break;
        case  BSIM4v5_MOD_PW0:
            mod->BSIM4v5pw0 = value->rValue;
            mod->BSIM4v5pw0Given = TRUE;
            break;
        case  BSIM4v5_MOD_PDVT0:               
            mod->BSIM4v5pdvt0 = value->rValue;
            mod->BSIM4v5pdvt0Given = TRUE;
            break;
        case  BSIM4v5_MOD_PDVT1:             
            mod->BSIM4v5pdvt1 = value->rValue;
            mod->BSIM4v5pdvt1Given = TRUE;
            break;
        case  BSIM4v5_MOD_PDVT2:             
            mod->BSIM4v5pdvt2 = value->rValue;
            mod->BSIM4v5pdvt2Given = TRUE;
            break;
        case  BSIM4v5_MOD_PDVT0W:               
            mod->BSIM4v5pdvt0w = value->rValue;
            mod->BSIM4v5pdvt0wGiven = TRUE;
            break;
        case  BSIM4v5_MOD_PDVT1W:             
            mod->BSIM4v5pdvt1w = value->rValue;
            mod->BSIM4v5pdvt1wGiven = TRUE;
            break;
        case  BSIM4v5_MOD_PDVT2W:             
            mod->BSIM4v5pdvt2w = value->rValue;
            mod->BSIM4v5pdvt2wGiven = TRUE;
            break;
        case  BSIM4v5_MOD_PDROUT:             
            mod->BSIM4v5pdrout = value->rValue;
            mod->BSIM4v5pdroutGiven = TRUE;
            break;
        case  BSIM4v5_MOD_PDSUB:             
            mod->BSIM4v5pdsub = value->rValue;
            mod->BSIM4v5pdsubGiven = TRUE;
            break;
        case BSIM4v5_MOD_PVTH0:
            mod->BSIM4v5pvth0 = value->rValue;
            mod->BSIM4v5pvth0Given = TRUE;
            break;
        case BSIM4v5_MOD_PUA:
            mod->BSIM4v5pua = value->rValue;
            mod->BSIM4v5puaGiven = TRUE;
            break;
        case BSIM4v5_MOD_PUA1:
            mod->BSIM4v5pua1 = value->rValue;
            mod->BSIM4v5pua1Given = TRUE;
            break;
        case BSIM4v5_MOD_PUB:
            mod->BSIM4v5pub = value->rValue;
            mod->BSIM4v5pubGiven = TRUE;
            break;
        case BSIM4v5_MOD_PUB1:
            mod->BSIM4v5pub1 = value->rValue;
            mod->BSIM4v5pub1Given = TRUE;
            break;
        case BSIM4v5_MOD_PUC:
            mod->BSIM4v5puc = value->rValue;
            mod->BSIM4v5pucGiven = TRUE;
            break;
        case BSIM4v5_MOD_PUC1:
            mod->BSIM4v5puc1 = value->rValue;
            mod->BSIM4v5puc1Given = TRUE;
            break;
        case  BSIM4v5_MOD_PU0 :
            mod->BSIM4v5pu0 = value->rValue;
            mod->BSIM4v5pu0Given = TRUE;
            break;
        case  BSIM4v5_MOD_PUTE :
            mod->BSIM4v5pute = value->rValue;
            mod->BSIM4v5puteGiven = TRUE;
            break;
        case BSIM4v5_MOD_PVOFF:
            mod->BSIM4v5pvoff = value->rValue;
            mod->BSIM4v5pvoffGiven = TRUE;
            break;
        case BSIM4v5_MOD_PTVOFF:
            mod->BSIM4v5ptvoff = value->rValue;
            mod->BSIM4v5ptvoffGiven = TRUE;
            break;
        case BSIM4v5_MOD_PMINV:
            mod->BSIM4v5pminv = value->rValue;
            mod->BSIM4v5pminvGiven = TRUE;
            break;
        case BSIM4v5_MOD_PFPROUT:
            mod->BSIM4v5pfprout = value->rValue;
            mod->BSIM4v5pfproutGiven = TRUE;
            break;
        case BSIM4v5_MOD_PPDITS:
            mod->BSIM4v5ppdits = value->rValue;
            mod->BSIM4v5ppditsGiven = TRUE;
            break;
        case BSIM4v5_MOD_PPDITSD:
            mod->BSIM4v5ppditsd = value->rValue;
            mod->BSIM4v5ppditsdGiven = TRUE;
            break;
        case  BSIM4v5_MOD_PDELTA :
            mod->BSIM4v5pdelta = value->rValue;
            mod->BSIM4v5pdeltaGiven = TRUE;
            break;
        case BSIM4v5_MOD_PRDSW:
            mod->BSIM4v5prdsw = value->rValue;
            mod->BSIM4v5prdswGiven = TRUE;
            break;                     
        case BSIM4v5_MOD_PRDW:
            mod->BSIM4v5prdw = value->rValue;
            mod->BSIM4v5prdwGiven = TRUE;
            break;
        case BSIM4v5_MOD_PRSW:
            mod->BSIM4v5prsw = value->rValue;
            mod->BSIM4v5prswGiven = TRUE;
            break;
        case BSIM4v5_MOD_PPRWB:
            mod->BSIM4v5pprwb = value->rValue;
            mod->BSIM4v5pprwbGiven = TRUE;
            break;                     
        case BSIM4v5_MOD_PPRWG:
            mod->BSIM4v5pprwg = value->rValue;
            mod->BSIM4v5pprwgGiven = TRUE;
            break;                     
        case BSIM4v5_MOD_PPRT:
            mod->BSIM4v5pprt = value->rValue;
            mod->BSIM4v5pprtGiven = TRUE;
            break;                     
        case BSIM4v5_MOD_PETA0:
            mod->BSIM4v5peta0 = value->rValue;
            mod->BSIM4v5peta0Given = TRUE;
            break;                 
        case BSIM4v5_MOD_PETAB:
            mod->BSIM4v5petab = value->rValue;
            mod->BSIM4v5petabGiven = TRUE;
            break;                 
        case BSIM4v5_MOD_PPCLM:
            mod->BSIM4v5ppclm = value->rValue;
            mod->BSIM4v5ppclmGiven = TRUE;
            break;                 
        case BSIM4v5_MOD_PPDIBL1:
            mod->BSIM4v5ppdibl1 = value->rValue;
            mod->BSIM4v5ppdibl1Given = TRUE;
            break;                 
        case BSIM4v5_MOD_PPDIBL2:
            mod->BSIM4v5ppdibl2 = value->rValue;
            mod->BSIM4v5ppdibl2Given = TRUE;
            break;                 
        case BSIM4v5_MOD_PPDIBLB:
            mod->BSIM4v5ppdiblb = value->rValue;
            mod->BSIM4v5ppdiblbGiven = TRUE;
            break;                 
        case BSIM4v5_MOD_PPSCBE1:
            mod->BSIM4v5ppscbe1 = value->rValue;
            mod->BSIM4v5ppscbe1Given = TRUE;
            break;                 
        case BSIM4v5_MOD_PPSCBE2:
            mod->BSIM4v5ppscbe2 = value->rValue;
            mod->BSIM4v5ppscbe2Given = TRUE;
            break;                 
        case BSIM4v5_MOD_PPVAG:
            mod->BSIM4v5ppvag = value->rValue;
            mod->BSIM4v5ppvagGiven = TRUE;
            break;                 
        case  BSIM4v5_MOD_PWR :
            mod->BSIM4v5pwr = value->rValue;
            mod->BSIM4v5pwrGiven = TRUE;
            break;
        case  BSIM4v5_MOD_PDWG :
            mod->BSIM4v5pdwg = value->rValue;
            mod->BSIM4v5pdwgGiven = TRUE;
            break;
        case  BSIM4v5_MOD_PDWB :
            mod->BSIM4v5pdwb = value->rValue;
            mod->BSIM4v5pdwbGiven = TRUE;
            break;
        case  BSIM4v5_MOD_PB0 :
            mod->BSIM4v5pb0 = value->rValue;
            mod->BSIM4v5pb0Given = TRUE;
            break;
        case  BSIM4v5_MOD_PB1 :
            mod->BSIM4v5pb1 = value->rValue;
            mod->BSIM4v5pb1Given = TRUE;
            break;
        case  BSIM4v5_MOD_PALPHA0 :
            mod->BSIM4v5palpha0 = value->rValue;
            mod->BSIM4v5palpha0Given = TRUE;
            break;
        case  BSIM4v5_MOD_PALPHA1 :
            mod->BSIM4v5palpha1 = value->rValue;
            mod->BSIM4v5palpha1Given = TRUE;
            break;
        case  BSIM4v5_MOD_PBETA0 :
            mod->BSIM4v5pbeta0 = value->rValue;
            mod->BSIM4v5pbeta0Given = TRUE;
            break;
        case  BSIM4v5_MOD_PAGIDL :
            mod->BSIM4v5pagidl = value->rValue;
            mod->BSIM4v5pagidlGiven = TRUE;
            break;
        case  BSIM4v5_MOD_PBGIDL :
            mod->BSIM4v5pbgidl = value->rValue;
            mod->BSIM4v5pbgidlGiven = TRUE;
            break;
        case  BSIM4v5_MOD_PCGIDL :
            mod->BSIM4v5pcgidl = value->rValue;
            mod->BSIM4v5pcgidlGiven = TRUE;
            break;
        case  BSIM4v5_MOD_PPHIN :
            mod->BSIM4v5pphin = value->rValue;
            mod->BSIM4v5pphinGiven = TRUE;
            break;
        case  BSIM4v5_MOD_PEGIDL :
            mod->BSIM4v5pegidl = value->rValue;
            mod->BSIM4v5pegidlGiven = TRUE;
            break;
        case  BSIM4v5_MOD_PAIGC :
            mod->BSIM4v5paigc = value->rValue;
            mod->BSIM4v5paigcGiven = TRUE;
            break;
        case  BSIM4v5_MOD_PBIGC :
            mod->BSIM4v5pbigc = value->rValue;
            mod->BSIM4v5pbigcGiven = TRUE;
            break;
        case  BSIM4v5_MOD_PCIGC :
            mod->BSIM4v5pcigc = value->rValue;
            mod->BSIM4v5pcigcGiven = TRUE;
            break;
        case  BSIM4v5_MOD_PAIGSD :
            mod->BSIM4v5paigsd = value->rValue;
            mod->BSIM4v5paigsdGiven = TRUE;
            break;
        case  BSIM4v5_MOD_PBIGSD :
            mod->BSIM4v5pbigsd = value->rValue;
            mod->BSIM4v5pbigsdGiven = TRUE;
            break;
        case  BSIM4v5_MOD_PCIGSD :
            mod->BSIM4v5pcigsd = value->rValue;
            mod->BSIM4v5pcigsdGiven = TRUE;
            break;
        case  BSIM4v5_MOD_PAIGBACC :
            mod->BSIM4v5paigbacc = value->rValue;
            mod->BSIM4v5paigbaccGiven = TRUE;
            break;
        case  BSIM4v5_MOD_PBIGBACC :
            mod->BSIM4v5pbigbacc = value->rValue;
            mod->BSIM4v5pbigbaccGiven = TRUE;
            break;
        case  BSIM4v5_MOD_PCIGBACC :
            mod->BSIM4v5pcigbacc = value->rValue;
            mod->BSIM4v5pcigbaccGiven = TRUE;
            break;
        case  BSIM4v5_MOD_PAIGBINV :
            mod->BSIM4v5paigbinv = value->rValue;
            mod->BSIM4v5paigbinvGiven = TRUE;
            break;
        case  BSIM4v5_MOD_PBIGBINV :
            mod->BSIM4v5pbigbinv = value->rValue;
            mod->BSIM4v5pbigbinvGiven = TRUE;
            break;
        case  BSIM4v5_MOD_PCIGBINV :
            mod->BSIM4v5pcigbinv = value->rValue;
            mod->BSIM4v5pcigbinvGiven = TRUE;
            break;
        case  BSIM4v5_MOD_PNIGC :
            mod->BSIM4v5pnigc = value->rValue;
            mod->BSIM4v5pnigcGiven = TRUE;
            break;
        case  BSIM4v5_MOD_PNIGBINV :
            mod->BSIM4v5pnigbinv = value->rValue;
            mod->BSIM4v5pnigbinvGiven = TRUE;
            break;
        case  BSIM4v5_MOD_PNIGBACC :
            mod->BSIM4v5pnigbacc = value->rValue;
            mod->BSIM4v5pnigbaccGiven = TRUE;
            break;
        case  BSIM4v5_MOD_PNTOX :
            mod->BSIM4v5pntox = value->rValue;
            mod->BSIM4v5pntoxGiven = TRUE;
            break;
        case  BSIM4v5_MOD_PEIGBINV :
            mod->BSIM4v5peigbinv = value->rValue;
            mod->BSIM4v5peigbinvGiven = TRUE;
            break;
        case  BSIM4v5_MOD_PPIGCD :
            mod->BSIM4v5ppigcd = value->rValue;
            mod->BSIM4v5ppigcdGiven = TRUE;
            break;
        case  BSIM4v5_MOD_PPOXEDGE :
            mod->BSIM4v5ppoxedge = value->rValue;
            mod->BSIM4v5ppoxedgeGiven = TRUE;
            break;
        case  BSIM4v5_MOD_PXRCRG1 :
            mod->BSIM4v5pxrcrg1 = value->rValue;
            mod->BSIM4v5pxrcrg1Given = TRUE;
            break;
        case  BSIM4v5_MOD_PXRCRG2 :
            mod->BSIM4v5pxrcrg2 = value->rValue;
            mod->BSIM4v5pxrcrg2Given = TRUE;
            break;
        case  BSIM4v5_MOD_PLAMBDA :
            mod->BSIM4v5plambda = value->rValue;
            mod->BSIM4v5plambdaGiven = TRUE;
            break;
        case  BSIM4v5_MOD_PVTL :
            mod->BSIM4v5pvtl = value->rValue;
            mod->BSIM4v5pvtlGiven = TRUE;
            break;
        case  BSIM4v5_MOD_PXN:
            mod->BSIM4v5pxn = value->rValue;
            mod->BSIM4v5pxnGiven = TRUE;
            break;
        case  BSIM4v5_MOD_PVFBSDOFF:
            mod->BSIM4v5pvfbsdoff = value->rValue;
            mod->BSIM4v5pvfbsdoffGiven = TRUE;
            break;
        case  BSIM4v5_MOD_PTVFBSDOFF:
            mod->BSIM4v5ptvfbsdoff = value->rValue;
            mod->BSIM4v5ptvfbsdoffGiven = TRUE;
            break;
        case  BSIM4v5_MOD_PEU :
            mod->BSIM4v5peu = value->rValue;
            mod->BSIM4v5peuGiven = TRUE;
            break;
        case  BSIM4v5_MOD_PVFB :
            mod->BSIM4v5pvfb = value->rValue;
            mod->BSIM4v5pvfbGiven = TRUE;
            break;
        case  BSIM4v5_MOD_PCGSL :
            mod->BSIM4v5pcgsl = value->rValue;
            mod->BSIM4v5pcgslGiven = TRUE;
            break;
        case  BSIM4v5_MOD_PCGDL :
            mod->BSIM4v5pcgdl = value->rValue;
            mod->BSIM4v5pcgdlGiven = TRUE;
            break;
        case  BSIM4v5_MOD_PCKAPPAS :
            mod->BSIM4v5pckappas = value->rValue;
            mod->BSIM4v5pckappasGiven = TRUE;
            break;
        case  BSIM4v5_MOD_PCKAPPAD :
            mod->BSIM4v5pckappad = value->rValue;
            mod->BSIM4v5pckappadGiven = TRUE;
            break;
        case  BSIM4v5_MOD_PCF :
            mod->BSIM4v5pcf = value->rValue;
            mod->BSIM4v5pcfGiven = TRUE;
            break;
        case  BSIM4v5_MOD_PCLC :
            mod->BSIM4v5pclc = value->rValue;
            mod->BSIM4v5pclcGiven = TRUE;
            break;
        case  BSIM4v5_MOD_PCLE :
            mod->BSIM4v5pcle = value->rValue;
            mod->BSIM4v5pcleGiven = TRUE;
            break;
        case  BSIM4v5_MOD_PVFBCV :
            mod->BSIM4v5pvfbcv = value->rValue;
            mod->BSIM4v5pvfbcvGiven = TRUE;
            break;
        case  BSIM4v5_MOD_PACDE :
            mod->BSIM4v5pacde = value->rValue;
            mod->BSIM4v5pacdeGiven = TRUE;
            break;
        case  BSIM4v5_MOD_PMOIN :
            mod->BSIM4v5pmoin = value->rValue;
            mod->BSIM4v5pmoinGiven = TRUE;
            break;
        case  BSIM4v5_MOD_PNOFF :
            mod->BSIM4v5pnoff = value->rValue;
            mod->BSIM4v5pnoffGiven = TRUE;
            break;
        case  BSIM4v5_MOD_PVOFFCV :
            mod->BSIM4v5pvoffcv = value->rValue;
            mod->BSIM4v5pvoffcvGiven = TRUE;
            break;

        case  BSIM4v5_MOD_TNOM :
            mod->BSIM4v5tnom = value->rValue + CONSTCtoK;
            mod->BSIM4v5tnomGiven = TRUE;
            break;
        case  BSIM4v5_MOD_CGSO :
            mod->BSIM4v5cgso = value->rValue;
            mod->BSIM4v5cgsoGiven = TRUE;
            break;
        case  BSIM4v5_MOD_CGDO :
            mod->BSIM4v5cgdo = value->rValue;
            mod->BSIM4v5cgdoGiven = TRUE;
            break;
        case  BSIM4v5_MOD_CGBO :
            mod->BSIM4v5cgbo = value->rValue;
            mod->BSIM4v5cgboGiven = TRUE;
            break;
        case  BSIM4v5_MOD_XPART :
            mod->BSIM4v5xpart = value->rValue;
            mod->BSIM4v5xpartGiven = TRUE;
            break;
        case  BSIM4v5_MOD_RSH :
            mod->BSIM4v5sheetResistance = value->rValue;
            mod->BSIM4v5sheetResistanceGiven = TRUE;
            break;
        case  BSIM4v5_MOD_JSS :
            mod->BSIM4v5SjctSatCurDensity = value->rValue;
            mod->BSIM4v5SjctSatCurDensityGiven = TRUE;
            break;
        case  BSIM4v5_MOD_JSWS :
            mod->BSIM4v5SjctSidewallSatCurDensity = value->rValue;
            mod->BSIM4v5SjctSidewallSatCurDensityGiven = TRUE;
            break;
        case  BSIM4v5_MOD_JSWGS :
            mod->BSIM4v5SjctGateSidewallSatCurDensity = value->rValue;
            mod->BSIM4v5SjctGateSidewallSatCurDensityGiven = TRUE;
            break;
        case  BSIM4v5_MOD_PBS :
            mod->BSIM4v5SbulkJctPotential = value->rValue;
            mod->BSIM4v5SbulkJctPotentialGiven = TRUE;
            break;
        case  BSIM4v5_MOD_MJS :
            mod->BSIM4v5SbulkJctBotGradingCoeff = value->rValue;
            mod->BSIM4v5SbulkJctBotGradingCoeffGiven = TRUE;
            break;
        case  BSIM4v5_MOD_PBSWS :
            mod->BSIM4v5SsidewallJctPotential = value->rValue;
            mod->BSIM4v5SsidewallJctPotentialGiven = TRUE;
            break;
        case  BSIM4v5_MOD_MJSWS :
            mod->BSIM4v5SbulkJctSideGradingCoeff = value->rValue;
            mod->BSIM4v5SbulkJctSideGradingCoeffGiven = TRUE;
            break;
        case  BSIM4v5_MOD_CJS :
            mod->BSIM4v5SunitAreaJctCap = value->rValue;
            mod->BSIM4v5SunitAreaJctCapGiven = TRUE;
            break;
        case  BSIM4v5_MOD_CJSWS :
            mod->BSIM4v5SunitLengthSidewallJctCap = value->rValue;
            mod->BSIM4v5SunitLengthSidewallJctCapGiven = TRUE;
            break;
        case  BSIM4v5_MOD_NJS :
            mod->BSIM4v5SjctEmissionCoeff = value->rValue;
            mod->BSIM4v5SjctEmissionCoeffGiven = TRUE;
            break;
        case  BSIM4v5_MOD_PBSWGS :
            mod->BSIM4v5SGatesidewallJctPotential = value->rValue;
            mod->BSIM4v5SGatesidewallJctPotentialGiven = TRUE;
            break;
        case  BSIM4v5_MOD_MJSWGS :
            mod->BSIM4v5SbulkJctGateSideGradingCoeff = value->rValue;
            mod->BSIM4v5SbulkJctGateSideGradingCoeffGiven = TRUE;
            break;
        case  BSIM4v5_MOD_CJSWGS :
            mod->BSIM4v5SunitLengthGateSidewallJctCap = value->rValue;
            mod->BSIM4v5SunitLengthGateSidewallJctCapGiven = TRUE;
            break;
        case  BSIM4v5_MOD_XTIS :
            mod->BSIM4v5SjctTempExponent = value->rValue;
            mod->BSIM4v5SjctTempExponentGiven = TRUE;
            break;
        case  BSIM4v5_MOD_JSD :
            mod->BSIM4v5DjctSatCurDensity = value->rValue;
            mod->BSIM4v5DjctSatCurDensityGiven = TRUE;
            break;
        case  BSIM4v5_MOD_JSWD :
            mod->BSIM4v5DjctSidewallSatCurDensity = value->rValue;
            mod->BSIM4v5DjctSidewallSatCurDensityGiven = TRUE;
            break;
        case  BSIM4v5_MOD_JSWGD :
            mod->BSIM4v5DjctGateSidewallSatCurDensity = value->rValue;
            mod->BSIM4v5DjctGateSidewallSatCurDensityGiven = TRUE;
            break;
        case  BSIM4v5_MOD_PBD :
            mod->BSIM4v5DbulkJctPotential = value->rValue;
            mod->BSIM4v5DbulkJctPotentialGiven = TRUE;
            break;
        case  BSIM4v5_MOD_MJD :
            mod->BSIM4v5DbulkJctBotGradingCoeff = value->rValue;
            mod->BSIM4v5DbulkJctBotGradingCoeffGiven = TRUE;
            break;
        case  BSIM4v5_MOD_PBSWD :
            mod->BSIM4v5DsidewallJctPotential = value->rValue;
            mod->BSIM4v5DsidewallJctPotentialGiven = TRUE;
            break;
        case  BSIM4v5_MOD_MJSWD :
            mod->BSIM4v5DbulkJctSideGradingCoeff = value->rValue;
            mod->BSIM4v5DbulkJctSideGradingCoeffGiven = TRUE;
            break;
        case  BSIM4v5_MOD_CJD :
            mod->BSIM4v5DunitAreaJctCap = value->rValue;
            mod->BSIM4v5DunitAreaJctCapGiven = TRUE;
            break;
        case  BSIM4v5_MOD_CJSWD :
            mod->BSIM4v5DunitLengthSidewallJctCap = value->rValue;
            mod->BSIM4v5DunitLengthSidewallJctCapGiven = TRUE;
            break;
        case  BSIM4v5_MOD_NJD :
            mod->BSIM4v5DjctEmissionCoeff = value->rValue;
            mod->BSIM4v5DjctEmissionCoeffGiven = TRUE;
            break;
        case  BSIM4v5_MOD_PBSWGD :
            mod->BSIM4v5DGatesidewallJctPotential = value->rValue;
            mod->BSIM4v5DGatesidewallJctPotentialGiven = TRUE;
            break;
        case  BSIM4v5_MOD_MJSWGD :
            mod->BSIM4v5DbulkJctGateSideGradingCoeff = value->rValue;
            mod->BSIM4v5DbulkJctGateSideGradingCoeffGiven = TRUE;
            break;
        case  BSIM4v5_MOD_CJSWGD :
            mod->BSIM4v5DunitLengthGateSidewallJctCap = value->rValue;
            mod->BSIM4v5DunitLengthGateSidewallJctCapGiven = TRUE;
            break;
        case  BSIM4v5_MOD_XTID :
            mod->BSIM4v5DjctTempExponent = value->rValue;
            mod->BSIM4v5DjctTempExponentGiven = TRUE;
            break;
        case  BSIM4v5_MOD_LINT :
            mod->BSIM4v5Lint = value->rValue;
            mod->BSIM4v5LintGiven = TRUE;
            break;
        case  BSIM4v5_MOD_LL :
            mod->BSIM4v5Ll = value->rValue;
            mod->BSIM4v5LlGiven = TRUE;
            break;
        case  BSIM4v5_MOD_LLC :
            mod->BSIM4v5Llc = value->rValue;
            mod->BSIM4v5LlcGiven = TRUE;
            break;
        case  BSIM4v5_MOD_LLN :
            mod->BSIM4v5Lln = value->rValue;
            mod->BSIM4v5LlnGiven = TRUE;
            break;
        case  BSIM4v5_MOD_LW :
            mod->BSIM4v5Lw = value->rValue;
            mod->BSIM4v5LwGiven = TRUE;
            break;
        case  BSIM4v5_MOD_LWC :
            mod->BSIM4v5Lwc = value->rValue;
            mod->BSIM4v5LwcGiven = TRUE;
            break;
        case  BSIM4v5_MOD_LWN :
            mod->BSIM4v5Lwn = value->rValue;
            mod->BSIM4v5LwnGiven = TRUE;
            break;
        case  BSIM4v5_MOD_LWL :
            mod->BSIM4v5Lwl = value->rValue;
            mod->BSIM4v5LwlGiven = TRUE;
            break;
        case  BSIM4v5_MOD_LWLC :
            mod->BSIM4v5Lwlc = value->rValue;
            mod->BSIM4v5LwlcGiven = TRUE;
            break;
        case  BSIM4v5_MOD_LMIN :
            mod->BSIM4v5Lmin = value->rValue;
            mod->BSIM4v5LminGiven = TRUE;
            break;
        case  BSIM4v5_MOD_LMAX :
            mod->BSIM4v5Lmax = value->rValue;
            mod->BSIM4v5LmaxGiven = TRUE;
            break;
        case  BSIM4v5_MOD_WINT :
            mod->BSIM4v5Wint = value->rValue;
            mod->BSIM4v5WintGiven = TRUE;
            break;
        case  BSIM4v5_MOD_WL :
            mod->BSIM4v5Wl = value->rValue;
            mod->BSIM4v5WlGiven = TRUE;
            break;
        case  BSIM4v5_MOD_WLC :
            mod->BSIM4v5Wlc = value->rValue;
            mod->BSIM4v5WlcGiven = TRUE;
            break;
        case  BSIM4v5_MOD_WLN :
            mod->BSIM4v5Wln = value->rValue;
            mod->BSIM4v5WlnGiven = TRUE;
            break;
        case  BSIM4v5_MOD_WW :
            mod->BSIM4v5Ww = value->rValue;
            mod->BSIM4v5WwGiven = TRUE;
            break;
        case  BSIM4v5_MOD_WWC :
            mod->BSIM4v5Wwc = value->rValue;
            mod->BSIM4v5WwcGiven = TRUE;
            break;
        case  BSIM4v5_MOD_WWN :
            mod->BSIM4v5Wwn = value->rValue;
            mod->BSIM4v5WwnGiven = TRUE;
            break;
        case  BSIM4v5_MOD_WWL :
            mod->BSIM4v5Wwl = value->rValue;
            mod->BSIM4v5WwlGiven = TRUE;
            break;
        case  BSIM4v5_MOD_WWLC :
            mod->BSIM4v5Wwlc = value->rValue;
            mod->BSIM4v5WwlcGiven = TRUE;
            break;
        case  BSIM4v5_MOD_WMIN :
            mod->BSIM4v5Wmin = value->rValue;
            mod->BSIM4v5WminGiven = TRUE;
            break;
        case  BSIM4v5_MOD_WMAX :
            mod->BSIM4v5Wmax = value->rValue;
            mod->BSIM4v5WmaxGiven = TRUE;
            break;

        case  BSIM4v5_MOD_NOIA :
            mod->BSIM4v5oxideTrapDensityA = value->rValue;
            mod->BSIM4v5oxideTrapDensityAGiven = TRUE;
            break;
        case  BSIM4v5_MOD_NOIB :
            mod->BSIM4v5oxideTrapDensityB = value->rValue;
            mod->BSIM4v5oxideTrapDensityBGiven = TRUE;
            break;
        case  BSIM4v5_MOD_NOIC :
            mod->BSIM4v5oxideTrapDensityC = value->rValue;
            mod->BSIM4v5oxideTrapDensityCGiven = TRUE;
            break;
        case  BSIM4v5_MOD_EM :
            mod->BSIM4v5em = value->rValue;
            mod->BSIM4v5emGiven = TRUE;
            break;
        case  BSIM4v5_MOD_EF :
            mod->BSIM4v5ef = value->rValue;
            mod->BSIM4v5efGiven = TRUE;
            break;
        case  BSIM4v5_MOD_AF :
            mod->BSIM4v5af = value->rValue;
            mod->BSIM4v5afGiven = TRUE;
            break;
        case  BSIM4v5_MOD_KF :
            mod->BSIM4v5kf = value->rValue;
            mod->BSIM4v5kfGiven = TRUE;
            break;

        case BSIM4v5_MOD_VGS_MAX:
            mod->BSIM4v5vgsMax = value->rValue;
            mod->BSIM4v5vgsMaxGiven = TRUE;
            break;
        case BSIM4v5_MOD_VGD_MAX:
            mod->BSIM4v5vgdMax = value->rValue;
            mod->BSIM4v5vgdMaxGiven = TRUE;
            break;
        case BSIM4v5_MOD_VGB_MAX:
            mod->BSIM4v5vgbMax = value->rValue;
            mod->BSIM4v5vgbMaxGiven = TRUE;
            break;
        case BSIM4v5_MOD_VDS_MAX:
            mod->BSIM4v5vdsMax = value->rValue;
            mod->BSIM4v5vdsMaxGiven = TRUE;
            break;
        case BSIM4v5_MOD_VBS_MAX:
            mod->BSIM4v5vbsMax = value->rValue;
            mod->BSIM4v5vbsMaxGiven = TRUE;
            break;
        case BSIM4v5_MOD_VBD_MAX:
            mod->BSIM4v5vbdMax = value->rValue;
            mod->BSIM4v5vbdMaxGiven = TRUE;
            break;
        case BSIM4v5_MOD_VGSR_MAX:
            mod->BSIM4v5vgsrMax = value->rValue;
            mod->BSIM4v5vgsrMaxGiven = TRUE;
            break;
        case BSIM4v5_MOD_VGDR_MAX:
            mod->BSIM4v5vgdrMax = value->rValue;
            mod->BSIM4v5vgdrMaxGiven = TRUE;
            break;
        case BSIM4v5_MOD_VGBR_MAX:
            mod->BSIM4v5vgbrMax = value->rValue;
            mod->BSIM4v5vgbrMaxGiven = TRUE;
            break;
        case BSIM4v5_MOD_VBSR_MAX:
            mod->BSIM4v5vbsrMax = value->rValue;
            mod->BSIM4v5vbsrMaxGiven = TRUE;
            break;
        case BSIM4v5_MOD_VBDR_MAX:
            mod->BSIM4v5vbdrMax = value->rValue;
            mod->BSIM4v5vbdrMaxGiven = TRUE;
            break;

        case  BSIM4v5_MOD_NMOS  :
            if(value->iValue) {
                mod->BSIM4v5type = 1;
                mod->BSIM4v5typeGiven = TRUE;
            }
            break;
        case  BSIM4v5_MOD_PMOS  :
            if(value->iValue) {
                mod->BSIM4v5type = - 1;
                mod->BSIM4v5typeGiven = TRUE;
            }
            break;
        default:
            return(E_BADPARM);
    }
    return(OK);
}


