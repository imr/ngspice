/**** BSIM4.6.2 Released by Wenwei Yang 07/31/2008 ****/

/**********
 * Copyright 2006 Regents of the University of California. All rights reserved.
 * File: b4mpar.c of BSIM4.6.1.
 * Author: 2000 Weidong Liu
 * Authors: 2001- Xuemei Xi, Mohan Dunga, Ali Niknejad, Chenming Hu.
 * Authors: 2006- Mohan Dunga, Ali Niknejad, Chenming Hu
 * Authors: 2007- Mohan Dunga, Wenwei Yang, Ali Niknejad, Chenming Hu
 * Authors: 2008- Wenwei Yang, Ali Niknejad, Chenming Hu 
 * Project Director: Prof. Chenming Hu.
 * Modified by Xuemei Xi, 04/06/2001.
 * Modified by Xuemei Xi, 10/05/2001.
 * Modified by Xuemei Xi, 11/15/2002.
 * Modified by Xuemei Xi, 05/09/2003.
 * Modified by Xuemei Xi, 03/04/2004.
 * Modified by Xuemei Xi, Mohan Dunga, 07/29/2005.
 * Modified by Mohan Dunga, 12/13/2006
 * Modified by Mohan Dunga, Wenwei Yang, 05/18/2007.
 * Modified by Wenwei Yang, 07/31/2008.
 **********/

#include "ngspice/ngspice.h"
#include "bsim4v6def.h"
#include "ngspice/ifsim.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"
#include "ngspice/const.h"

int
BSIM4v6mParam(
int param,
IFvalue *value,
GENmodel *inMod)
{
    BSIM4v6model *mod = (BSIM4v6model*)inMod;
    switch(param)
    {   case  BSIM4v6_MOD_MOBMOD :
            mod->BSIM4v6mobMod = value->iValue;
            mod->BSIM4v6mobModGiven = TRUE;
            break;
        case  BSIM4v6_MOD_BINUNIT :
            mod->BSIM4v6binUnit = value->iValue;
            mod->BSIM4v6binUnitGiven = TRUE;
            break;
        case  BSIM4v6_MOD_PARAMCHK :
            mod->BSIM4v6paramChk = value->iValue;
            mod->BSIM4v6paramChkGiven = TRUE;
            break;
        case  BSIM4v6_MOD_CVCHARGEMOD :
            mod->BSIM4v6cvchargeMod = value->iValue;
            mod->BSIM4v6cvchargeModGiven = TRUE;
            break;
        case  BSIM4v6_MOD_CAPMOD :
            mod->BSIM4v6capMod = value->iValue;
            mod->BSIM4v6capModGiven = TRUE;
            break;
        case  BSIM4v6_MOD_DIOMOD :
            mod->BSIM4v6dioMod = value->iValue;
            mod->BSIM4v6dioModGiven = TRUE;
            break;
        case  BSIM4v6_MOD_RDSMOD :
            mod->BSIM4v6rdsMod = value->iValue;
            mod->BSIM4v6rdsModGiven = TRUE;
            break;
        case  BSIM4v6_MOD_TRNQSMOD :
            mod->BSIM4v6trnqsMod = value->iValue;
            mod->BSIM4v6trnqsModGiven = TRUE;
            break;
        case  BSIM4v6_MOD_ACNQSMOD :
            mod->BSIM4v6acnqsMod = value->iValue;
            mod->BSIM4v6acnqsModGiven = TRUE;
            break;
        case  BSIM4v6_MOD_RBODYMOD :
            mod->BSIM4v6rbodyMod = value->iValue;
            mod->BSIM4v6rbodyModGiven = TRUE;
            break;
        case  BSIM4v6_MOD_RGATEMOD :
            mod->BSIM4v6rgateMod = value->iValue;
            mod->BSIM4v6rgateModGiven = TRUE;
            break;
        case  BSIM4v6_MOD_PERMOD :
            mod->BSIM4v6perMod = value->iValue;
            mod->BSIM4v6perModGiven = TRUE;
            break;
        case  BSIM4v6_MOD_GEOMOD :
            mod->BSIM4v6geoMod = value->iValue;
            mod->BSIM4v6geoModGiven = TRUE;
            break;
        case  BSIM4v6_MOD_RGEOMOD :
            mod->BSIM4v6rgeoMod = value->iValue;
            mod->BSIM4v6rgeoModGiven = TRUE;
            break;
        case  BSIM4v6_MOD_FNOIMOD :
            mod->BSIM4v6fnoiMod = value->iValue;
            mod->BSIM4v6fnoiModGiven = TRUE;
            break;
        case  BSIM4v6_MOD_TNOIMOD :
            mod->BSIM4v6tnoiMod = value->iValue;
            mod->BSIM4v6tnoiModGiven = TRUE;
            break;
        case  BSIM4v6_MOD_MTRLMOD :
            mod->BSIM4v6mtrlMod = value->iValue;
            mod->BSIM4v6mtrlModGiven = TRUE;
            break;
        case  BSIM4v6_MOD_IGCMOD :
            mod->BSIM4v6igcMod = value->iValue;
            mod->BSIM4v6igcModGiven = TRUE;
            break;
        case  BSIM4v6_MOD_IGBMOD :
            mod->BSIM4v6igbMod = value->iValue;
            mod->BSIM4v6igbModGiven = TRUE;
            break;
        case  BSIM4v6_MOD_TEMPMOD :
            mod->BSIM4v6tempMod = value->iValue;
            mod->BSIM4v6tempModGiven = TRUE;
            break;

        case  BSIM4v6_MOD_VERSION :
            mod->BSIM4v6version = value->sValue;
            mod->BSIM4v6versionGiven = TRUE;
            break;
        case  BSIM4v6_MOD_TOXREF :
            mod->BSIM4v6toxref = value->rValue;
            mod->BSIM4v6toxrefGiven = TRUE;
            break;
        case  BSIM4v6_MOD_EOT :
            mod->BSIM4v6eot = value->rValue;
            mod->BSIM4v6eotGiven = TRUE;
            break;
        case  BSIM4v6_MOD_VDDEOT :
            mod->BSIM4v6vddeot = value->rValue;
            mod->BSIM4v6vddeotGiven = TRUE;
            break;
		  case  BSIM4v6_MOD_TEMPEOT :
            mod->BSIM4v6tempeot = value->rValue;
            mod->BSIM4v6tempeotGiven = TRUE;
            break;
		  case  BSIM4v6_MOD_LEFFEOT :
            mod->BSIM4v6leffeot = value->rValue;
            mod->BSIM4v6leffeotGiven = TRUE;
            break;
		  case  BSIM4v6_MOD_WEFFEOT :
            mod->BSIM4v6weffeot = value->rValue;
            mod->BSIM4v6weffeotGiven = TRUE;
            break;
         case  BSIM4v6_MOD_ADOS :
            mod->BSIM4v6ados = value->rValue;
            mod->BSIM4v6adosGiven = TRUE;
            break;
        case  BSIM4v6_MOD_BDOS :
            mod->BSIM4v6bdos = value->rValue;
            mod->BSIM4v6bdosGiven = TRUE;
            break;
       case  BSIM4v6_MOD_TOXE :
            mod->BSIM4v6toxe = value->rValue;
            mod->BSIM4v6toxeGiven = TRUE;
            break;
        case  BSIM4v6_MOD_TOXP :
            mod->BSIM4v6toxp = value->rValue;
            mod->BSIM4v6toxpGiven = TRUE;
            break;
        case  BSIM4v6_MOD_TOXM :
            mod->BSIM4v6toxm = value->rValue;
            mod->BSIM4v6toxmGiven = TRUE;
            break;
        case  BSIM4v6_MOD_DTOX :
            mod->BSIM4v6dtox = value->rValue;
            mod->BSIM4v6dtoxGiven = TRUE;
            break;
        case  BSIM4v6_MOD_EPSROX :
            mod->BSIM4v6epsrox = value->rValue;
            mod->BSIM4v6epsroxGiven = TRUE;
            break;

        case  BSIM4v6_MOD_CDSC :
            mod->BSIM4v6cdsc = value->rValue;
            mod->BSIM4v6cdscGiven = TRUE;
            break;
        case  BSIM4v6_MOD_CDSCB :
            mod->BSIM4v6cdscb = value->rValue;
            mod->BSIM4v6cdscbGiven = TRUE;
            break;

        case  BSIM4v6_MOD_CDSCD :
            mod->BSIM4v6cdscd = value->rValue;
            mod->BSIM4v6cdscdGiven = TRUE;
            break;

        case  BSIM4v6_MOD_CIT :
            mod->BSIM4v6cit = value->rValue;
            mod->BSIM4v6citGiven = TRUE;
            break;
        case  BSIM4v6_MOD_NFACTOR :
            mod->BSIM4v6nfactor = value->rValue;
            mod->BSIM4v6nfactorGiven = TRUE;
            break;
        case BSIM4v6_MOD_XJ:
            mod->BSIM4v6xj = value->rValue;
            mod->BSIM4v6xjGiven = TRUE;
            break;
        case BSIM4v6_MOD_VSAT:
            mod->BSIM4v6vsat = value->rValue;
            mod->BSIM4v6vsatGiven = TRUE;
            break;
        case BSIM4v6_MOD_A0:
            mod->BSIM4v6a0 = value->rValue;
            mod->BSIM4v6a0Given = TRUE;
            break;
        
        case BSIM4v6_MOD_AGS:
            mod->BSIM4v6ags= value->rValue;
            mod->BSIM4v6agsGiven = TRUE;
            break;
        
        case BSIM4v6_MOD_A1:
            mod->BSIM4v6a1 = value->rValue;
            mod->BSIM4v6a1Given = TRUE;
            break;
        case BSIM4v6_MOD_A2:
            mod->BSIM4v6a2 = value->rValue;
            mod->BSIM4v6a2Given = TRUE;
            break;
        case BSIM4v6_MOD_AT:
            mod->BSIM4v6at = value->rValue;
            mod->BSIM4v6atGiven = TRUE;
            break;
        case BSIM4v6_MOD_KETA:
            mod->BSIM4v6keta = value->rValue;
            mod->BSIM4v6ketaGiven = TRUE;
            break;    
        case BSIM4v6_MOD_NSUB:
            mod->BSIM4v6nsub = value->rValue;
            mod->BSIM4v6nsubGiven = TRUE;
            break;
        case BSIM4v6_MOD_PHIG:
	    mod->BSIM4v6phig = value->rValue;
	    mod->BSIM4v6phigGiven = TRUE;
	    break;
        case BSIM4v6_MOD_EPSRGATE:
	    mod->BSIM4v6epsrgate = value->rValue;
	    mod->BSIM4v6epsrgateGiven = TRUE;
	    break;
        case BSIM4v6_MOD_EASUB:
            mod->BSIM4v6easub = value->rValue;
            mod->BSIM4v6easubGiven = TRUE;
            break;
        case BSIM4v6_MOD_EPSRSUB:
            mod->BSIM4v6epsrsub = value->rValue;
            mod->BSIM4v6epsrsubGiven = TRUE;
            break;
        case BSIM4v6_MOD_NI0SUB:
            mod->BSIM4v6ni0sub = value->rValue;
            mod->BSIM4v6ni0subGiven = TRUE;
            break;
        case BSIM4v6_MOD_BG0SUB:
            mod->BSIM4v6bg0sub = value->rValue;
            mod->BSIM4v6bg0subGiven = TRUE;
            break;
        case BSIM4v6_MOD_TBGASUB:
            mod->BSIM4v6tbgasub = value->rValue;
            mod->BSIM4v6tbgasubGiven = TRUE;
            break;
        case BSIM4v6_MOD_TBGBSUB:
            mod->BSIM4v6tbgbsub = value->rValue;
            mod->BSIM4v6tbgbsubGiven = TRUE;
            break;
        case BSIM4v6_MOD_NDEP:
            mod->BSIM4v6ndep = value->rValue;
            mod->BSIM4v6ndepGiven = TRUE;
	    if (mod->BSIM4v6ndep > 1.0e20)
		mod->BSIM4v6ndep *= 1.0e-6;
            break;
        case BSIM4v6_MOD_NSD:
            mod->BSIM4v6nsd = value->rValue;
            mod->BSIM4v6nsdGiven = TRUE;
            if (mod->BSIM4v6nsd > 1.000001e24)
                mod->BSIM4v6nsd *= 1.0e-6;
            break;
        case BSIM4v6_MOD_NGATE:
            mod->BSIM4v6ngate = value->rValue;
            mod->BSIM4v6ngateGiven = TRUE;
            if (mod->BSIM4v6ngate > 1.000001e24)
                mod->BSIM4v6ngate *= 1.0e-6;
            break;
        case BSIM4v6_MOD_GAMMA1:
            mod->BSIM4v6gamma1 = value->rValue;
            mod->BSIM4v6gamma1Given = TRUE;
            break;
        case BSIM4v6_MOD_GAMMA2:
            mod->BSIM4v6gamma2 = value->rValue;
            mod->BSIM4v6gamma2Given = TRUE;
            break;
        case BSIM4v6_MOD_VBX:
            mod->BSIM4v6vbx = value->rValue;
            mod->BSIM4v6vbxGiven = TRUE;
            break;
        case BSIM4v6_MOD_VBM:
            mod->BSIM4v6vbm = value->rValue;
            mod->BSIM4v6vbmGiven = TRUE;
            break;
        case BSIM4v6_MOD_XT:
            mod->BSIM4v6xt = value->rValue;
            mod->BSIM4v6xtGiven = TRUE;
            break;
        case  BSIM4v6_MOD_K1:
            mod->BSIM4v6k1 = value->rValue;
            mod->BSIM4v6k1Given = TRUE;
            break;
        case  BSIM4v6_MOD_KT1:
            mod->BSIM4v6kt1 = value->rValue;
            mod->BSIM4v6kt1Given = TRUE;
            break;
        case  BSIM4v6_MOD_KT1L:
            mod->BSIM4v6kt1l = value->rValue;
            mod->BSIM4v6kt1lGiven = TRUE;
            break;
        case  BSIM4v6_MOD_KT2:
            mod->BSIM4v6kt2 = value->rValue;
            mod->BSIM4v6kt2Given = TRUE;
            break;
        case  BSIM4v6_MOD_K2:
            mod->BSIM4v6k2 = value->rValue;
            mod->BSIM4v6k2Given = TRUE;
            break;
        case  BSIM4v6_MOD_K3:
            mod->BSIM4v6k3 = value->rValue;
            mod->BSIM4v6k3Given = TRUE;
            break;
        case  BSIM4v6_MOD_K3B:
            mod->BSIM4v6k3b = value->rValue;
            mod->BSIM4v6k3bGiven = TRUE;
            break;
        case  BSIM4v6_MOD_LPE0:
            mod->BSIM4v6lpe0 = value->rValue;
            mod->BSIM4v6lpe0Given = TRUE;
            break;
        case  BSIM4v6_MOD_LPEB:
            mod->BSIM4v6lpeb = value->rValue;
            mod->BSIM4v6lpebGiven = TRUE;
            break;
        case  BSIM4v6_MOD_DVTP0:
            mod->BSIM4v6dvtp0 = value->rValue;
            mod->BSIM4v6dvtp0Given = TRUE;
            break;
        case  BSIM4v6_MOD_DVTP1:
            mod->BSIM4v6dvtp1 = value->rValue;
            mod->BSIM4v6dvtp1Given = TRUE;
            break;
        case  BSIM4v6_MOD_W0:
            mod->BSIM4v6w0 = value->rValue;
            mod->BSIM4v6w0Given = TRUE;
            break;
        case  BSIM4v6_MOD_DVT0:               
            mod->BSIM4v6dvt0 = value->rValue;
            mod->BSIM4v6dvt0Given = TRUE;
            break;
        case  BSIM4v6_MOD_DVT1:             
            mod->BSIM4v6dvt1 = value->rValue;
            mod->BSIM4v6dvt1Given = TRUE;
            break;
        case  BSIM4v6_MOD_DVT2:             
            mod->BSIM4v6dvt2 = value->rValue;
            mod->BSIM4v6dvt2Given = TRUE;
            break;
        case  BSIM4v6_MOD_DVT0W:               
            mod->BSIM4v6dvt0w = value->rValue;
            mod->BSIM4v6dvt0wGiven = TRUE;
            break;
        case  BSIM4v6_MOD_DVT1W:             
            mod->BSIM4v6dvt1w = value->rValue;
            mod->BSIM4v6dvt1wGiven = TRUE;
            break;
        case  BSIM4v6_MOD_DVT2W:             
            mod->BSIM4v6dvt2w = value->rValue;
            mod->BSIM4v6dvt2wGiven = TRUE;
            break;
        case  BSIM4v6_MOD_DROUT:             
            mod->BSIM4v6drout = value->rValue;
            mod->BSIM4v6droutGiven = TRUE;
            break;
        case  BSIM4v6_MOD_DSUB:             
            mod->BSIM4v6dsub = value->rValue;
            mod->BSIM4v6dsubGiven = TRUE;
            break;
        case BSIM4v6_MOD_VTH0:
            mod->BSIM4v6vth0 = value->rValue;
            mod->BSIM4v6vth0Given = TRUE;
            break;
        case BSIM4v6_MOD_EU:
            mod->BSIM4v6eu = value->rValue;
            mod->BSIM4v6euGiven = TRUE;
            break;
		case BSIM4v6_MOD_UCS:
            mod->BSIM4v6ucs = value->rValue;
            mod->BSIM4v6ucsGiven = TRUE;
            break;
        case BSIM4v6_MOD_UA:
            mod->BSIM4v6ua = value->rValue;
            mod->BSIM4v6uaGiven = TRUE;
            break;
        case BSIM4v6_MOD_UA1:
            mod->BSIM4v6ua1 = value->rValue;
            mod->BSIM4v6ua1Given = TRUE;
            break;
        case BSIM4v6_MOD_UB:
            mod->BSIM4v6ub = value->rValue;
            mod->BSIM4v6ubGiven = TRUE;
            break;
        case BSIM4v6_MOD_UB1:
            mod->BSIM4v6ub1 = value->rValue;
            mod->BSIM4v6ub1Given = TRUE;
            break;
        case BSIM4v6_MOD_UC:
            mod->BSIM4v6uc = value->rValue;
            mod->BSIM4v6ucGiven = TRUE;
            break;
        case BSIM4v6_MOD_UC1:
            mod->BSIM4v6uc1 = value->rValue;
            mod->BSIM4v6uc1Given = TRUE;
            break;
        case  BSIM4v6_MOD_U0 :
            mod->BSIM4v6u0 = value->rValue;
            mod->BSIM4v6u0Given = TRUE;
            break;
        case  BSIM4v6_MOD_UTE :
            mod->BSIM4v6ute = value->rValue;
            mod->BSIM4v6uteGiven = TRUE;
            break;
        case  BSIM4v6_MOD_UCSTE :
            mod->BSIM4v6ucste = value->rValue;
            mod->BSIM4v6ucsteGiven = TRUE;
            break;
        case BSIM4v6_MOD_UD:
            mod->BSIM4v6ud = value->rValue;
            mod->BSIM4v6udGiven = TRUE;
            break;
        case BSIM4v6_MOD_UD1:
            mod->BSIM4v6ud1 = value->rValue;
            mod->BSIM4v6ud1Given = TRUE;
            break;
        case BSIM4v6_MOD_UP:
            mod->BSIM4v6up = value->rValue;
            mod->BSIM4v6upGiven = TRUE;
            break;
        case BSIM4v6_MOD_LP:
            mod->BSIM4v6lp = value->rValue;
            mod->BSIM4v6lpGiven = TRUE;
            break;
        case BSIM4v6_MOD_LUD:
            mod->BSIM4v6lud = value->rValue;
            mod->BSIM4v6ludGiven = TRUE;
            break;
        case BSIM4v6_MOD_LUD1:
            mod->BSIM4v6lud1 = value->rValue;
            mod->BSIM4v6lud1Given = TRUE;
            break;
        case BSIM4v6_MOD_LUP:
            mod->BSIM4v6lup = value->rValue;
            mod->BSIM4v6lupGiven = TRUE;
            break;
        case BSIM4v6_MOD_LLP:
            mod->BSIM4v6llp = value->rValue;
            mod->BSIM4v6llpGiven = TRUE;
            break;
        case BSIM4v6_MOD_WUD:
            mod->BSIM4v6wud = value->rValue;
            mod->BSIM4v6wudGiven = TRUE;
            break;
        case BSIM4v6_MOD_WUD1:
            mod->BSIM4v6wud1 = value->rValue;
            mod->BSIM4v6wud1Given = TRUE;
            break;
        case BSIM4v6_MOD_WUP:
            mod->BSIM4v6wup = value->rValue;
            mod->BSIM4v6wupGiven = TRUE;
            break;
        case BSIM4v6_MOD_WLP:
            mod->BSIM4v6wlp = value->rValue;
            mod->BSIM4v6wlpGiven = TRUE;
            break;
        case BSIM4v6_MOD_PUD:
            mod->BSIM4v6pud = value->rValue;
            mod->BSIM4v6pudGiven = TRUE;
            break;
        case BSIM4v6_MOD_PUD1:
            mod->BSIM4v6pud1 = value->rValue;
            mod->BSIM4v6pud1Given = TRUE;
            break;
        case BSIM4v6_MOD_PUP:
            mod->BSIM4v6pup = value->rValue;
            mod->BSIM4v6pupGiven = TRUE;
            break;
        case BSIM4v6_MOD_PLP:
            mod->BSIM4v6plp = value->rValue;
            mod->BSIM4v6plpGiven = TRUE;
            break;


        case BSIM4v6_MOD_VOFF:
            mod->BSIM4v6voff = value->rValue;
            mod->BSIM4v6voffGiven = TRUE;
            break;
        case BSIM4v6_MOD_TVOFF:
            mod->BSIM4v6tvoff = value->rValue;
            mod->BSIM4v6tvoffGiven = TRUE;
            break;
        case BSIM4v6_MOD_VOFFL:
            mod->BSIM4v6voffl = value->rValue;
            mod->BSIM4v6vofflGiven = TRUE;
            break;
        case BSIM4v6_MOD_VOFFCVL:
            mod->BSIM4v6voffcvl = value->rValue;
            mod->BSIM4v6voffcvlGiven = TRUE;
            break;
        case BSIM4v6_MOD_MINV:
            mod->BSIM4v6minv = value->rValue;
            mod->BSIM4v6minvGiven = TRUE;
            break;
        case BSIM4v6_MOD_MINVCV:
            mod->BSIM4v6minvcv = value->rValue;
            mod->BSIM4v6minvcvGiven = TRUE;
            break;
        case BSIM4v6_MOD_FPROUT:
            mod->BSIM4v6fprout = value->rValue;
            mod->BSIM4v6fproutGiven = TRUE;
            break;
        case BSIM4v6_MOD_PDITS:
            mod->BSIM4v6pdits = value->rValue;
            mod->BSIM4v6pditsGiven = TRUE;
            break;
        case BSIM4v6_MOD_PDITSD:
            mod->BSIM4v6pditsd = value->rValue;
            mod->BSIM4v6pditsdGiven = TRUE;
            break;
        case BSIM4v6_MOD_PDITSL:
            mod->BSIM4v6pditsl = value->rValue;
            mod->BSIM4v6pditslGiven = TRUE;
            break;
        case  BSIM4v6_MOD_DELTA :
            mod->BSIM4v6delta = value->rValue;
            mod->BSIM4v6deltaGiven = TRUE;
            break;
        case BSIM4v6_MOD_RDSW:
            mod->BSIM4v6rdsw = value->rValue;
            mod->BSIM4v6rdswGiven = TRUE;
            break;                     
        case BSIM4v6_MOD_RDSWMIN:
            mod->BSIM4v6rdswmin = value->rValue;
            mod->BSIM4v6rdswminGiven = TRUE;
            break;
        case BSIM4v6_MOD_RDWMIN:
            mod->BSIM4v6rdwmin = value->rValue;
            mod->BSIM4v6rdwminGiven = TRUE;
            break;
        case BSIM4v6_MOD_RSWMIN:
            mod->BSIM4v6rswmin = value->rValue;
            mod->BSIM4v6rswminGiven = TRUE;
            break;
        case BSIM4v6_MOD_RDW:
            mod->BSIM4v6rdw = value->rValue;
            mod->BSIM4v6rdwGiven = TRUE;
            break;
        case BSIM4v6_MOD_RSW:
            mod->BSIM4v6rsw = value->rValue;
            mod->BSIM4v6rswGiven = TRUE;
            break;
        case BSIM4v6_MOD_PRWG:
            mod->BSIM4v6prwg = value->rValue;
            mod->BSIM4v6prwgGiven = TRUE;
            break;                     
        case BSIM4v6_MOD_PRWB:
            mod->BSIM4v6prwb = value->rValue;
            mod->BSIM4v6prwbGiven = TRUE;
            break;                     
        case BSIM4v6_MOD_PRT:
            mod->BSIM4v6prt = value->rValue;
            mod->BSIM4v6prtGiven = TRUE;
            break;                     
        case BSIM4v6_MOD_ETA0:
            mod->BSIM4v6eta0 = value->rValue;
            mod->BSIM4v6eta0Given = TRUE;
            break;                 
        case BSIM4v6_MOD_ETAB:
            mod->BSIM4v6etab = value->rValue;
            mod->BSIM4v6etabGiven = TRUE;
            break;                 
        case BSIM4v6_MOD_PCLM:
            mod->BSIM4v6pclm = value->rValue;
            mod->BSIM4v6pclmGiven = TRUE;
            break;                 
        case BSIM4v6_MOD_PDIBL1:
            mod->BSIM4v6pdibl1 = value->rValue;
            mod->BSIM4v6pdibl1Given = TRUE;
            break;                 
        case BSIM4v6_MOD_PDIBL2:
            mod->BSIM4v6pdibl2 = value->rValue;
            mod->BSIM4v6pdibl2Given = TRUE;
            break;                 
        case BSIM4v6_MOD_PDIBLB:
            mod->BSIM4v6pdiblb = value->rValue;
            mod->BSIM4v6pdiblbGiven = TRUE;
            break;                 
        case BSIM4v6_MOD_PSCBE1:
            mod->BSIM4v6pscbe1 = value->rValue;
            mod->BSIM4v6pscbe1Given = TRUE;
            break;                 
        case BSIM4v6_MOD_PSCBE2:
            mod->BSIM4v6pscbe2 = value->rValue;
            mod->BSIM4v6pscbe2Given = TRUE;
            break;                 
        case BSIM4v6_MOD_PVAG:
            mod->BSIM4v6pvag = value->rValue;
            mod->BSIM4v6pvagGiven = TRUE;
            break;                 
        case  BSIM4v6_MOD_WR :
            mod->BSIM4v6wr = value->rValue;
            mod->BSIM4v6wrGiven = TRUE;
            break;
        case  BSIM4v6_MOD_DWG :
            mod->BSIM4v6dwg = value->rValue;
            mod->BSIM4v6dwgGiven = TRUE;
            break;
        case  BSIM4v6_MOD_DWB :
            mod->BSIM4v6dwb = value->rValue;
            mod->BSIM4v6dwbGiven = TRUE;
            break;
        case  BSIM4v6_MOD_B0 :
            mod->BSIM4v6b0 = value->rValue;
            mod->BSIM4v6b0Given = TRUE;
            break;
        case  BSIM4v6_MOD_B1 :
            mod->BSIM4v6b1 = value->rValue;
            mod->BSIM4v6b1Given = TRUE;
            break;
        case  BSIM4v6_MOD_ALPHA0 :
            mod->BSIM4v6alpha0 = value->rValue;
            mod->BSIM4v6alpha0Given = TRUE;
            break;
        case  BSIM4v6_MOD_ALPHA1 :
            mod->BSIM4v6alpha1 = value->rValue;
            mod->BSIM4v6alpha1Given = TRUE;
            break;
        case  BSIM4v6_MOD_PHIN :
            mod->BSIM4v6phin = value->rValue;
            mod->BSIM4v6phinGiven = TRUE;
            break;
        case  BSIM4v6_MOD_AGIDL :
            mod->BSIM4v6agidl = value->rValue;
            mod->BSIM4v6agidlGiven = TRUE;
            break;
        case  BSIM4v6_MOD_BGIDL :
            mod->BSIM4v6bgidl = value->rValue;
            mod->BSIM4v6bgidlGiven = TRUE;
            break;
        case  BSIM4v6_MOD_CGIDL :
            mod->BSIM4v6cgidl = value->rValue;
            mod->BSIM4v6cgidlGiven = TRUE;
            break;
        case  BSIM4v6_MOD_EGIDL :
            mod->BSIM4v6egidl = value->rValue;
            mod->BSIM4v6egidlGiven = TRUE;
            break;
        case  BSIM4v6_MOD_AGISL :
            mod->BSIM4v6agisl = value->rValue;
            mod->BSIM4v6agislGiven = TRUE;
            break;
        case  BSIM4v6_MOD_BGISL :
            mod->BSIM4v6bgisl = value->rValue;
            mod->BSIM4v6bgislGiven = TRUE;
            break;
        case  BSIM4v6_MOD_CGISL :
            mod->BSIM4v6cgisl = value->rValue;
            mod->BSIM4v6cgislGiven = TRUE;
            break;
        case  BSIM4v6_MOD_EGISL :
            mod->BSIM4v6egisl = value->rValue;
            mod->BSIM4v6egislGiven = TRUE;
            break;
        case  BSIM4v6_MOD_AIGC :
            mod->BSIM4v6aigc = value->rValue;
            mod->BSIM4v6aigcGiven = TRUE;
            break;
        case  BSIM4v6_MOD_BIGC :
            mod->BSIM4v6bigc = value->rValue;
            mod->BSIM4v6bigcGiven = TRUE;
            break;
        case  BSIM4v6_MOD_CIGC :
            mod->BSIM4v6cigc = value->rValue;
            mod->BSIM4v6cigcGiven = TRUE;
            break;
        case  BSIM4v6_MOD_AIGSD :
            mod->BSIM4v6aigsd = value->rValue;
            mod->BSIM4v6aigsdGiven = TRUE;
            break;
        case  BSIM4v6_MOD_BIGSD :
            mod->BSIM4v6bigsd = value->rValue;
            mod->BSIM4v6bigsdGiven = TRUE;
            break;
        case  BSIM4v6_MOD_CIGSD :
            mod->BSIM4v6cigsd = value->rValue;
            mod->BSIM4v6cigsdGiven = TRUE;
            break;
        case  BSIM4v6_MOD_AIGS :
            mod->BSIM4v6aigs = value->rValue;
            mod->BSIM4v6aigsGiven = TRUE;
            break;
        case  BSIM4v6_MOD_BIGS :
            mod->BSIM4v6bigs = value->rValue;
            mod->BSIM4v6bigsGiven = TRUE;
            break;
        case  BSIM4v6_MOD_CIGS :
            mod->BSIM4v6cigs = value->rValue;
            mod->BSIM4v6cigsGiven = TRUE;
            break;
        case  BSIM4v6_MOD_AIGD :
            mod->BSIM4v6aigd = value->rValue;
            mod->BSIM4v6aigdGiven = TRUE;
            break;
        case  BSIM4v6_MOD_BIGD :
            mod->BSIM4v6bigd = value->rValue;
            mod->BSIM4v6bigdGiven = TRUE;
            break;
        case  BSIM4v6_MOD_CIGD :
            mod->BSIM4v6cigd = value->rValue;
            mod->BSIM4v6cigdGiven = TRUE;
            break;
        case  BSIM4v6_MOD_AIGBACC :
            mod->BSIM4v6aigbacc = value->rValue;
            mod->BSIM4v6aigbaccGiven = TRUE;
            break;
        case  BSIM4v6_MOD_BIGBACC :
            mod->BSIM4v6bigbacc = value->rValue;
            mod->BSIM4v6bigbaccGiven = TRUE;
            break;
        case  BSIM4v6_MOD_CIGBACC :
            mod->BSIM4v6cigbacc = value->rValue;
            mod->BSIM4v6cigbaccGiven = TRUE;
            break;
        case  BSIM4v6_MOD_AIGBINV :
            mod->BSIM4v6aigbinv = value->rValue;
            mod->BSIM4v6aigbinvGiven = TRUE;
            break;
        case  BSIM4v6_MOD_BIGBINV :
            mod->BSIM4v6bigbinv = value->rValue;
            mod->BSIM4v6bigbinvGiven = TRUE;
            break;
        case  BSIM4v6_MOD_CIGBINV :
            mod->BSIM4v6cigbinv = value->rValue;
            mod->BSIM4v6cigbinvGiven = TRUE;
            break;
        case  BSIM4v6_MOD_NIGC :
            mod->BSIM4v6nigc = value->rValue;
            mod->BSIM4v6nigcGiven = TRUE;
            break;
        case  BSIM4v6_MOD_NIGBINV :
            mod->BSIM4v6nigbinv = value->rValue;
            mod->BSIM4v6nigbinvGiven = TRUE;
            break;
        case  BSIM4v6_MOD_NIGBACC :
            mod->BSIM4v6nigbacc = value->rValue;
            mod->BSIM4v6nigbaccGiven = TRUE;
            break;
        case  BSIM4v6_MOD_NTOX :
            mod->BSIM4v6ntox = value->rValue;
            mod->BSIM4v6ntoxGiven = TRUE;
            break;
        case  BSIM4v6_MOD_EIGBINV :
            mod->BSIM4v6eigbinv = value->rValue;
            mod->BSIM4v6eigbinvGiven = TRUE;
            break;
        case  BSIM4v6_MOD_PIGCD :
            mod->BSIM4v6pigcd = value->rValue;
            mod->BSIM4v6pigcdGiven = TRUE;
            break;
        case  BSIM4v6_MOD_POXEDGE :
            mod->BSIM4v6poxedge = value->rValue;
            mod->BSIM4v6poxedgeGiven = TRUE;
            break;
        case  BSIM4v6_MOD_XRCRG1 :
            mod->BSIM4v6xrcrg1 = value->rValue;
            mod->BSIM4v6xrcrg1Given = TRUE;
            break;
        case  BSIM4v6_MOD_XRCRG2 :
            mod->BSIM4v6xrcrg2 = value->rValue;
            mod->BSIM4v6xrcrg2Given = TRUE;
            break;
        case  BSIM4v6_MOD_LAMBDA :
            mod->BSIM4v6lambda = value->rValue;
            mod->BSIM4v6lambdaGiven = TRUE;
            break;
        case  BSIM4v6_MOD_VTL :
            mod->BSIM4v6vtl = value->rValue;
            mod->BSIM4v6vtlGiven = TRUE;
            break;
        case  BSIM4v6_MOD_XN:
            mod->BSIM4v6xn = value->rValue;
            mod->BSIM4v6xnGiven = TRUE;
            break;
        case  BSIM4v6_MOD_LC:
            mod->BSIM4v6lc = value->rValue;
            mod->BSIM4v6lcGiven = TRUE;
            break;
        case  BSIM4v6_MOD_TNOIA :
            mod->BSIM4v6tnoia = value->rValue;
            mod->BSIM4v6tnoiaGiven = TRUE;
            break;
        case  BSIM4v6_MOD_TNOIB :
            mod->BSIM4v6tnoib = value->rValue;
            mod->BSIM4v6tnoibGiven = TRUE;
            break;
        case  BSIM4v6_MOD_RNOIA :
            mod->BSIM4v6rnoia = value->rValue;
            mod->BSIM4v6rnoiaGiven = TRUE;
            break;
        case  BSIM4v6_MOD_RNOIB :
            mod->BSIM4v6rnoib = value->rValue;
            mod->BSIM4v6rnoibGiven = TRUE;
            break;
        case  BSIM4v6_MOD_NTNOI :
            mod->BSIM4v6ntnoi = value->rValue;
            mod->BSIM4v6ntnoiGiven = TRUE;
            break;
        case  BSIM4v6_MOD_VFBSDOFF:
            mod->BSIM4v6vfbsdoff = value->rValue;
            mod->BSIM4v6vfbsdoffGiven = TRUE;
            break;
        case  BSIM4v6_MOD_TVFBSDOFF:
            mod->BSIM4v6tvfbsdoff = value->rValue;
            mod->BSIM4v6tvfbsdoffGiven = TRUE;
            break;
        case  BSIM4v6_MOD_LINTNOI:
            mod->BSIM4v6lintnoi = value->rValue;
            mod->BSIM4v6lintnoiGiven = TRUE;
            break;

        /* stress effect */
        case  BSIM4v6_MOD_SAREF :
            mod->BSIM4v6saref = value->rValue;
            mod->BSIM4v6sarefGiven = TRUE;
            break;
        case  BSIM4v6_MOD_SBREF :
            mod->BSIM4v6sbref = value->rValue;
            mod->BSIM4v6sbrefGiven = TRUE;
            break;
        case  BSIM4v6_MOD_WLOD :
            mod->BSIM4v6wlod = value->rValue;
            mod->BSIM4v6wlodGiven = TRUE;
            break;
        case  BSIM4v6_MOD_KU0 :
            mod->BSIM4v6ku0 = value->rValue;
            mod->BSIM4v6ku0Given = TRUE;
            break;
        case  BSIM4v6_MOD_KVSAT :
            mod->BSIM4v6kvsat = value->rValue;
            mod->BSIM4v6kvsatGiven = TRUE;
            break;
        case  BSIM4v6_MOD_KVTH0 :
            mod->BSIM4v6kvth0 = value->rValue;
            mod->BSIM4v6kvth0Given = TRUE;
            break;
        case  BSIM4v6_MOD_TKU0 :
            mod->BSIM4v6tku0 = value->rValue;
            mod->BSIM4v6tku0Given = TRUE;
            break;
        case  BSIM4v6_MOD_LLODKU0 :
            mod->BSIM4v6llodku0 = value->rValue;
            mod->BSIM4v6llodku0Given = TRUE;
            break;
        case  BSIM4v6_MOD_WLODKU0 :
            mod->BSIM4v6wlodku0 = value->rValue;
            mod->BSIM4v6wlodku0Given = TRUE;
            break;
        case  BSIM4v6_MOD_LLODVTH :
            mod->BSIM4v6llodvth = value->rValue;
            mod->BSIM4v6llodvthGiven = TRUE;
            break;
        case  BSIM4v6_MOD_WLODVTH :
            mod->BSIM4v6wlodvth = value->rValue;
            mod->BSIM4v6wlodvthGiven = TRUE;
            break;
        case  BSIM4v6_MOD_LKU0 :
            mod->BSIM4v6lku0 = value->rValue;
            mod->BSIM4v6lku0Given = TRUE;
            break;
        case  BSIM4v6_MOD_WKU0 :
            mod->BSIM4v6wku0 = value->rValue;
            mod->BSIM4v6wku0Given = TRUE;
            break;
        case  BSIM4v6_MOD_PKU0 :
            mod->BSIM4v6pku0 = value->rValue;
            mod->BSIM4v6pku0Given = TRUE;
            break;
        case  BSIM4v6_MOD_LKVTH0 :
            mod->BSIM4v6lkvth0 = value->rValue;
            mod->BSIM4v6lkvth0Given = TRUE;
            break;
        case  BSIM4v6_MOD_WKVTH0 :
            mod->BSIM4v6wkvth0 = value->rValue;
            mod->BSIM4v6wkvth0Given = TRUE;
            break;
        case  BSIM4v6_MOD_PKVTH0 :
            mod->BSIM4v6pkvth0 = value->rValue;
            mod->BSIM4v6pkvth0Given = TRUE;
            break;
        case  BSIM4v6_MOD_STK2 :
            mod->BSIM4v6stk2 = value->rValue;
            mod->BSIM4v6stk2Given = TRUE;
            break;
        case  BSIM4v6_MOD_LODK2 :
            mod->BSIM4v6lodk2 = value->rValue;
            mod->BSIM4v6lodk2Given = TRUE;
            break;
        case  BSIM4v6_MOD_STETA0 :
            mod->BSIM4v6steta0 = value->rValue;
            mod->BSIM4v6steta0Given = TRUE;
            break;
        case  BSIM4v6_MOD_LODETA0 :
            mod->BSIM4v6lodeta0 = value->rValue;
            mod->BSIM4v6lodeta0Given = TRUE;
            break;

        case  BSIM4v6_MOD_WEB :
            mod->BSIM4v6web = value->rValue;
            mod->BSIM4v6webGiven = TRUE;
            break;
	case BSIM4v6_MOD_WEC :
            mod->BSIM4v6wec = value->rValue;
            mod->BSIM4v6wecGiven = TRUE;
            break;
        case  BSIM4v6_MOD_KVTH0WE :
            mod->BSIM4v6kvth0we = value->rValue;
            mod->BSIM4v6kvth0weGiven = TRUE;
            break;
        case  BSIM4v6_MOD_K2WE :
            mod->BSIM4v6k2we = value->rValue;
            mod->BSIM4v6k2weGiven = TRUE;
            break;
        case  BSIM4v6_MOD_KU0WE :
            mod->BSIM4v6ku0we = value->rValue;
            mod->BSIM4v6ku0weGiven = TRUE;
            break;
        case  BSIM4v6_MOD_SCREF :
            mod->BSIM4v6scref = value->rValue;
            mod->BSIM4v6screfGiven = TRUE;
            break;
        case  BSIM4v6_MOD_WPEMOD :
            mod->BSIM4v6wpemod = value->rValue;
            mod->BSIM4v6wpemodGiven = TRUE;
            break;
        case  BSIM4v6_MOD_LKVTH0WE :
            mod->BSIM4v6lkvth0we = value->rValue;
            mod->BSIM4v6lkvth0weGiven = TRUE;
            break;
        case  BSIM4v6_MOD_LK2WE :
            mod->BSIM4v6lk2we = value->rValue;
            mod->BSIM4v6lk2weGiven = TRUE;
            break;
        case  BSIM4v6_MOD_LKU0WE :
            mod->BSIM4v6lku0we = value->rValue;
            mod->BSIM4v6lku0weGiven = TRUE;
            break;
        case  BSIM4v6_MOD_WKVTH0WE :
            mod->BSIM4v6wkvth0we = value->rValue;
            mod->BSIM4v6wkvth0weGiven = TRUE;
            break;
        case  BSIM4v6_MOD_WK2WE :
            mod->BSIM4v6wk2we = value->rValue;
            mod->BSIM4v6wk2weGiven = TRUE;
            break;
        case  BSIM4v6_MOD_WKU0WE :
            mod->BSIM4v6wku0we = value->rValue;
            mod->BSIM4v6wku0weGiven = TRUE;
            break;
        case  BSIM4v6_MOD_PKVTH0WE :
            mod->BSIM4v6pkvth0we = value->rValue;
            mod->BSIM4v6pkvth0weGiven = TRUE;
            break;
        case  BSIM4v6_MOD_PK2WE :
            mod->BSIM4v6pk2we = value->rValue;
            mod->BSIM4v6pk2weGiven = TRUE;
            break;
        case  BSIM4v6_MOD_PKU0WE :
            mod->BSIM4v6pku0we = value->rValue;
            mod->BSIM4v6pku0weGiven = TRUE;
            break;

        case  BSIM4v6_MOD_BETA0 :
            mod->BSIM4v6beta0 = value->rValue;
            mod->BSIM4v6beta0Given = TRUE;
            break;
        case  BSIM4v6_MOD_IJTHDFWD :
            mod->BSIM4v6ijthdfwd = value->rValue;
            mod->BSIM4v6ijthdfwdGiven = TRUE;
            break;
        case  BSIM4v6_MOD_IJTHSFWD :
            mod->BSIM4v6ijthsfwd = value->rValue;
            mod->BSIM4v6ijthsfwdGiven = TRUE;
            break;
        case  BSIM4v6_MOD_IJTHDREV :
            mod->BSIM4v6ijthdrev = value->rValue;
            mod->BSIM4v6ijthdrevGiven = TRUE;
            break;
        case  BSIM4v6_MOD_IJTHSREV :
            mod->BSIM4v6ijthsrev = value->rValue;
            mod->BSIM4v6ijthsrevGiven = TRUE;
            break;
        case  BSIM4v6_MOD_XJBVD :
            mod->BSIM4v6xjbvd = value->rValue;
            mod->BSIM4v6xjbvdGiven = TRUE;
            break;
        case  BSIM4v6_MOD_XJBVS :
            mod->BSIM4v6xjbvs = value->rValue;
            mod->BSIM4v6xjbvsGiven = TRUE;
            break;
        case  BSIM4v6_MOD_BVD :
            mod->BSIM4v6bvd = value->rValue;
            mod->BSIM4v6bvdGiven = TRUE;
            break;
        case  BSIM4v6_MOD_BVS :
            mod->BSIM4v6bvs = value->rValue;
            mod->BSIM4v6bvsGiven = TRUE;
            break;
        
        /* reverse diode */
        case  BSIM4v6_MOD_JTSS :
            mod->BSIM4v6jtss = value->rValue;
            mod->BSIM4v6jtssGiven = TRUE;
            break;
        case  BSIM4v6_MOD_JTSD :
            mod->BSIM4v6jtsd = value->rValue;
            mod->BSIM4v6jtsdGiven = TRUE;
            break;
        case  BSIM4v6_MOD_JTSSWS :
            mod->BSIM4v6jtssws = value->rValue;
            mod->BSIM4v6jtsswsGiven = TRUE;
            break;
        case  BSIM4v6_MOD_JTSSWD :
            mod->BSIM4v6jtsswd = value->rValue;
            mod->BSIM4v6jtsswdGiven = TRUE;
            break;
        case  BSIM4v6_MOD_JTSSWGS :
            mod->BSIM4v6jtsswgs = value->rValue;
            mod->BSIM4v6jtsswgsGiven = TRUE;
            break;
        case  BSIM4v6_MOD_JTSSWGD :
            mod->BSIM4v6jtsswgd = value->rValue;
            mod->BSIM4v6jtsswgdGiven = TRUE;
            break;
	case BSIM4v6_MOD_JTWEFF :
	    mod->BSIM4v6jtweff = value->rValue;
	    mod->BSIM4v6jtweffGiven = TRUE;
            break;
        case  BSIM4v6_MOD_NJTS :
            mod->BSIM4v6njts = value->rValue;
            mod->BSIM4v6njtsGiven = TRUE;
            break;
        case  BSIM4v6_MOD_NJTSSW :
            mod->BSIM4v6njtssw = value->rValue;
            mod->BSIM4v6njtsswGiven = TRUE;
            break;
        case  BSIM4v6_MOD_NJTSSWG :
            mod->BSIM4v6njtsswg = value->rValue;
            mod->BSIM4v6njtsswgGiven = TRUE;
            break;
        case  BSIM4v6_MOD_NJTSD :
            mod->BSIM4v6njtsd = value->rValue;
            mod->BSIM4v6njtsdGiven = TRUE;
            break;
        case  BSIM4v6_MOD_NJTSSWD :
            mod->BSIM4v6njtsswd = value->rValue;
            mod->BSIM4v6njtsswdGiven = TRUE;
            break;
        case  BSIM4v6_MOD_NJTSSWGD :
            mod->BSIM4v6njtsswgd = value->rValue;
            mod->BSIM4v6njtsswgdGiven = TRUE;
            break;
        case  BSIM4v6_MOD_XTSS :
            mod->BSIM4v6xtss = value->rValue;
            mod->BSIM4v6xtssGiven = TRUE;
            break;
        case  BSIM4v6_MOD_XTSD :
            mod->BSIM4v6xtsd = value->rValue;
            mod->BSIM4v6xtsdGiven = TRUE;
            break;
        case  BSIM4v6_MOD_XTSSWS :
            mod->BSIM4v6xtssws = value->rValue;
            mod->BSIM4v6xtsswsGiven = TRUE;
            break;
        case  BSIM4v6_MOD_XTSSWD :
            mod->BSIM4v6xtsswd = value->rValue;
            mod->BSIM4v6xtsswdGiven = TRUE;
            break;
        case  BSIM4v6_MOD_XTSSWGS :
            mod->BSIM4v6xtsswgs = value->rValue;
            mod->BSIM4v6xtsswgsGiven = TRUE;
            break;
        case  BSIM4v6_MOD_XTSSWGD :
            mod->BSIM4v6xtsswgd = value->rValue;
            mod->BSIM4v6xtsswgdGiven = TRUE;
            break;
        case  BSIM4v6_MOD_TNJTS :
            mod->BSIM4v6tnjts = value->rValue;
            mod->BSIM4v6tnjtsGiven = TRUE;
            break;
        case  BSIM4v6_MOD_TNJTSSW :
            mod->BSIM4v6tnjtssw = value->rValue;
            mod->BSIM4v6tnjtsswGiven = TRUE;
            break;
        case  BSIM4v6_MOD_TNJTSSWG :
            mod->BSIM4v6tnjtsswg = value->rValue;
            mod->BSIM4v6tnjtsswgGiven = TRUE;
            break;
        case  BSIM4v6_MOD_TNJTSD :
            mod->BSIM4v6tnjtsd = value->rValue;
            mod->BSIM4v6tnjtsdGiven = TRUE;
            break;
        case  BSIM4v6_MOD_TNJTSSWD :
            mod->BSIM4v6tnjtsswd = value->rValue;
            mod->BSIM4v6tnjtsswdGiven = TRUE;
            break;
        case  BSIM4v6_MOD_TNJTSSWGD :
            mod->BSIM4v6tnjtsswgd = value->rValue;
            mod->BSIM4v6tnjtsswgdGiven = TRUE;
            break;
        case  BSIM4v6_MOD_VTSS :
            mod->BSIM4v6vtss = value->rValue;
            mod->BSIM4v6vtssGiven = TRUE;
            break;
        case  BSIM4v6_MOD_VTSD :
            mod->BSIM4v6vtsd = value->rValue;
            mod->BSIM4v6vtsdGiven = TRUE;
            break;
        case  BSIM4v6_MOD_VTSSWS :
            mod->BSIM4v6vtssws = value->rValue;
            mod->BSIM4v6vtsswsGiven = TRUE;
            break;
        case  BSIM4v6_MOD_VTSSWD :
            mod->BSIM4v6vtsswd = value->rValue;
            mod->BSIM4v6vtsswdGiven = TRUE;
            break;
        case  BSIM4v6_MOD_VTSSWGS :
            mod->BSIM4v6vtsswgs = value->rValue;
            mod->BSIM4v6vtsswgsGiven = TRUE;
            break;
        case  BSIM4v6_MOD_VTSSWGD :
            mod->BSIM4v6vtsswgd = value->rValue;
            mod->BSIM4v6vtsswgdGiven = TRUE;
            break;

        case  BSIM4v6_MOD_VFB :
            mod->BSIM4v6vfb = value->rValue;
            mod->BSIM4v6vfbGiven = TRUE;
            break;

        case  BSIM4v6_MOD_GBMIN :
            mod->BSIM4v6gbmin = value->rValue;
            mod->BSIM4v6gbminGiven = TRUE;
            break;
        case  BSIM4v6_MOD_RBDB :
            mod->BSIM4v6rbdb = value->rValue;
            mod->BSIM4v6rbdbGiven = TRUE;
            break;
        case  BSIM4v6_MOD_RBPB :
            mod->BSIM4v6rbpb = value->rValue;
            mod->BSIM4v6rbpbGiven = TRUE;
            break;
        case  BSIM4v6_MOD_RBSB :
            mod->BSIM4v6rbsb = value->rValue;
            mod->BSIM4v6rbsbGiven = TRUE;
            break;
        case  BSIM4v6_MOD_RBPS :
            mod->BSIM4v6rbps = value->rValue;
            mod->BSIM4v6rbpsGiven = TRUE;
            break;
        case  BSIM4v6_MOD_RBPD :
            mod->BSIM4v6rbpd = value->rValue;
            mod->BSIM4v6rbpdGiven = TRUE;
            break;

        case  BSIM4v6_MOD_RBPS0 :
            mod->BSIM4v6rbps0 = value->rValue;
            mod->BSIM4v6rbps0Given = TRUE;
            break;
        case  BSIM4v6_MOD_RBPSL :
            mod->BSIM4v6rbpsl = value->rValue;
            mod->BSIM4v6rbpslGiven = TRUE;
            break;
        case  BSIM4v6_MOD_RBPSW :
            mod->BSIM4v6rbpsw = value->rValue;
            mod->BSIM4v6rbpswGiven = TRUE;
            break;
        case  BSIM4v6_MOD_RBPSNF :
            mod->BSIM4v6rbpsnf = value->rValue;
            mod->BSIM4v6rbpsnfGiven = TRUE;
            break;

        case  BSIM4v6_MOD_RBPD0 :
            mod->BSIM4v6rbpd0 = value->rValue;
            mod->BSIM4v6rbpd0Given = TRUE;
            break;
        case  BSIM4v6_MOD_RBPDL :
            mod->BSIM4v6rbpdl = value->rValue;
            mod->BSIM4v6rbpdlGiven = TRUE;
            break;
        case  BSIM4v6_MOD_RBPDW :
            mod->BSIM4v6rbpdw = value->rValue;
            mod->BSIM4v6rbpdwGiven = TRUE;
            break;
        case  BSIM4v6_MOD_RBPDNF :
            mod->BSIM4v6rbpdnf = value->rValue;
            mod->BSIM4v6rbpdnfGiven = TRUE;
            break;

        case  BSIM4v6_MOD_RBPBX0 :
            mod->BSIM4v6rbpbx0 = value->rValue;
            mod->BSIM4v6rbpbx0Given = TRUE;
            break;
        case  BSIM4v6_MOD_RBPBXL :
            mod->BSIM4v6rbpbxl = value->rValue;
            mod->BSIM4v6rbpbxlGiven = TRUE;
            break;
        case  BSIM4v6_MOD_RBPBXW :
            mod->BSIM4v6rbpbxw = value->rValue;
            mod->BSIM4v6rbpbxwGiven = TRUE;
            break;
        case  BSIM4v6_MOD_RBPBXNF :
            mod->BSIM4v6rbpbxnf = value->rValue;
            mod->BSIM4v6rbpbxnfGiven = TRUE;
            break;
        case  BSIM4v6_MOD_RBPBY0 :
            mod->BSIM4v6rbpby0 = value->rValue;
            mod->BSIM4v6rbpby0Given = TRUE;
            break;
        case  BSIM4v6_MOD_RBPBYL :
            mod->BSIM4v6rbpbyl = value->rValue;
            mod->BSIM4v6rbpbylGiven = TRUE;
            break;
        case  BSIM4v6_MOD_RBPBYW :
            mod->BSIM4v6rbpbyw = value->rValue;
            mod->BSIM4v6rbpbywGiven = TRUE;
            break;
        case  BSIM4v6_MOD_RBPBYNF :
            mod->BSIM4v6rbpbynf = value->rValue;
            mod->BSIM4v6rbpbynfGiven = TRUE;
            break;
       case  BSIM4v6_MOD_RBSBX0 :
            mod->BSIM4v6rbsbx0 = value->rValue;
            mod->BSIM4v6rbsbx0Given = TRUE;
            break;
       case  BSIM4v6_MOD_RBSBY0 :
            mod->BSIM4v6rbsby0 = value->rValue;
            mod->BSIM4v6rbsby0Given = TRUE;
            break;
       case  BSIM4v6_MOD_RBDBX0 :
            mod->BSIM4v6rbdbx0 = value->rValue;
            mod->BSIM4v6rbdbx0Given = TRUE;
            break;
       case  BSIM4v6_MOD_RBDBY0 :
            mod->BSIM4v6rbdby0 = value->rValue;
            mod->BSIM4v6rbdby0Given = TRUE;
            break;


       case  BSIM4v6_MOD_RBSDBXL :
            mod->BSIM4v6rbsdbxl = value->rValue;
            mod->BSIM4v6rbsdbxlGiven = TRUE;
            break;
       case  BSIM4v6_MOD_RBSDBXW :
            mod->BSIM4v6rbsdbxw = value->rValue;
            mod->BSIM4v6rbsdbxwGiven = TRUE;
            break;
       case  BSIM4v6_MOD_RBSDBXNF :
            mod->BSIM4v6rbsdbxnf = value->rValue;
            mod->BSIM4v6rbsdbxnfGiven = TRUE;
            break;
       case  BSIM4v6_MOD_RBSDBYL :
            mod->BSIM4v6rbsdbyl = value->rValue;
            mod->BSIM4v6rbsdbylGiven = TRUE;
            break;
       case  BSIM4v6_MOD_RBSDBYW :
            mod->BSIM4v6rbsdbyw = value->rValue;
            mod->BSIM4v6rbsdbywGiven = TRUE;
            break;
       case  BSIM4v6_MOD_RBSDBYNF :
            mod->BSIM4v6rbsdbynf = value->rValue;
            mod->BSIM4v6rbsdbynfGiven = TRUE;
            break;
 
        case  BSIM4v6_MOD_CGSL :
            mod->BSIM4v6cgsl = value->rValue;
            mod->BSIM4v6cgslGiven = TRUE;
            break;
        case  BSIM4v6_MOD_CGDL :
            mod->BSIM4v6cgdl = value->rValue;
            mod->BSIM4v6cgdlGiven = TRUE;
            break;
        case  BSIM4v6_MOD_CKAPPAS :
            mod->BSIM4v6ckappas = value->rValue;
            mod->BSIM4v6ckappasGiven = TRUE;
            break;
        case  BSIM4v6_MOD_CKAPPAD :
            mod->BSIM4v6ckappad = value->rValue;
            mod->BSIM4v6ckappadGiven = TRUE;
            break;
        case  BSIM4v6_MOD_CF :
            mod->BSIM4v6cf = value->rValue;
            mod->BSIM4v6cfGiven = TRUE;
            break;
        case  BSIM4v6_MOD_CLC :
            mod->BSIM4v6clc = value->rValue;
            mod->BSIM4v6clcGiven = TRUE;
            break;
        case  BSIM4v6_MOD_CLE :
            mod->BSIM4v6cle = value->rValue;
            mod->BSIM4v6cleGiven = TRUE;
            break;
        case  BSIM4v6_MOD_DWC :
            mod->BSIM4v6dwc = value->rValue;
            mod->BSIM4v6dwcGiven = TRUE;
            break;
        case  BSIM4v6_MOD_DLC :
            mod->BSIM4v6dlc = value->rValue;
            mod->BSIM4v6dlcGiven = TRUE;
            break;
        case  BSIM4v6_MOD_XW :
            mod->BSIM4v6xw = value->rValue;
            mod->BSIM4v6xwGiven = TRUE;
            break;
        case  BSIM4v6_MOD_XL :
            mod->BSIM4v6xl = value->rValue;
            mod->BSIM4v6xlGiven = TRUE;
            break;
        case  BSIM4v6_MOD_DLCIG :
            mod->BSIM4v6dlcig = value->rValue;
            mod->BSIM4v6dlcigGiven = TRUE;
            break;
        case  BSIM4v6_MOD_DLCIGD :
            mod->BSIM4v6dlcigd = value->rValue;
            mod->BSIM4v6dlcigdGiven = TRUE;
            break;
        case  BSIM4v6_MOD_DWJ :
            mod->BSIM4v6dwj = value->rValue;
            mod->BSIM4v6dwjGiven = TRUE;
            break;
        case  BSIM4v6_MOD_VFBCV :
            mod->BSIM4v6vfbcv = value->rValue;
            mod->BSIM4v6vfbcvGiven = TRUE;
            break;
        case  BSIM4v6_MOD_ACDE :
            mod->BSIM4v6acde = value->rValue;
            mod->BSIM4v6acdeGiven = TRUE;
            break;
        case  BSIM4v6_MOD_MOIN :
            mod->BSIM4v6moin = value->rValue;
            mod->BSIM4v6moinGiven = TRUE;
            break;
        case  BSIM4v6_MOD_NOFF :
            mod->BSIM4v6noff = value->rValue;
            mod->BSIM4v6noffGiven = TRUE;
            break;
        case  BSIM4v6_MOD_VOFFCV :
            mod->BSIM4v6voffcv = value->rValue;
            mod->BSIM4v6voffcvGiven = TRUE;
            break;
        case  BSIM4v6_MOD_DMCG :
            mod->BSIM4v6dmcg = value->rValue;
            mod->BSIM4v6dmcgGiven = TRUE;
            break;
        case  BSIM4v6_MOD_DMCI :
            mod->BSIM4v6dmci = value->rValue;
            mod->BSIM4v6dmciGiven = TRUE;
            break;
        case  BSIM4v6_MOD_DMDG :
            mod->BSIM4v6dmdg = value->rValue;
            mod->BSIM4v6dmdgGiven = TRUE;
            break;
        case  BSIM4v6_MOD_DMCGT :
            mod->BSIM4v6dmcgt = value->rValue;
            mod->BSIM4v6dmcgtGiven = TRUE;
            break;
        case  BSIM4v6_MOD_XGW :
            mod->BSIM4v6xgw = value->rValue;
            mod->BSIM4v6xgwGiven = TRUE;
            break;
        case  BSIM4v6_MOD_XGL :
            mod->BSIM4v6xgl = value->rValue;
            mod->BSIM4v6xglGiven = TRUE;
            break;
        case  BSIM4v6_MOD_RSHG :
            mod->BSIM4v6rshg = value->rValue;
            mod->BSIM4v6rshgGiven = TRUE;
            break;
        case  BSIM4v6_MOD_NGCON :
            mod->BSIM4v6ngcon = value->rValue;
            mod->BSIM4v6ngconGiven = TRUE;
            break;
        case  BSIM4v6_MOD_TCJ :
            mod->BSIM4v6tcj = value->rValue;
            mod->BSIM4v6tcjGiven = TRUE;
            break;
        case  BSIM4v6_MOD_TPB :
            mod->BSIM4v6tpb = value->rValue;
            mod->BSIM4v6tpbGiven = TRUE;
            break;
        case  BSIM4v6_MOD_TCJSW :
            mod->BSIM4v6tcjsw = value->rValue;
            mod->BSIM4v6tcjswGiven = TRUE;
            break;
        case  BSIM4v6_MOD_TPBSW :
            mod->BSIM4v6tpbsw = value->rValue;
            mod->BSIM4v6tpbswGiven = TRUE;
            break;
        case  BSIM4v6_MOD_TCJSWG :
            mod->BSIM4v6tcjswg = value->rValue;
            mod->BSIM4v6tcjswgGiven = TRUE;
            break;
        case  BSIM4v6_MOD_TPBSWG :
            mod->BSIM4v6tpbswg = value->rValue;
            mod->BSIM4v6tpbswgGiven = TRUE;
            break;

	/* Length dependence */
        case  BSIM4v6_MOD_LCDSC :
            mod->BSIM4v6lcdsc = value->rValue;
            mod->BSIM4v6lcdscGiven = TRUE;
            break;


        case  BSIM4v6_MOD_LCDSCB :
            mod->BSIM4v6lcdscb = value->rValue;
            mod->BSIM4v6lcdscbGiven = TRUE;
            break;
        case  BSIM4v6_MOD_LCDSCD :
            mod->BSIM4v6lcdscd = value->rValue;
            mod->BSIM4v6lcdscdGiven = TRUE;
            break;
        case  BSIM4v6_MOD_LCIT :
            mod->BSIM4v6lcit = value->rValue;
            mod->BSIM4v6lcitGiven = TRUE;
            break;
        case  BSIM4v6_MOD_LNFACTOR :
            mod->BSIM4v6lnfactor = value->rValue;
            mod->BSIM4v6lnfactorGiven = TRUE;
            break;
        case BSIM4v6_MOD_LXJ:
            mod->BSIM4v6lxj = value->rValue;
            mod->BSIM4v6lxjGiven = TRUE;
            break;
        case BSIM4v6_MOD_LVSAT:
            mod->BSIM4v6lvsat = value->rValue;
            mod->BSIM4v6lvsatGiven = TRUE;
            break;
        
        
        case BSIM4v6_MOD_LA0:
            mod->BSIM4v6la0 = value->rValue;
            mod->BSIM4v6la0Given = TRUE;
            break;
        case BSIM4v6_MOD_LAGS:
            mod->BSIM4v6lags = value->rValue;
            mod->BSIM4v6lagsGiven = TRUE;
            break;
        case BSIM4v6_MOD_LA1:
            mod->BSIM4v6la1 = value->rValue;
            mod->BSIM4v6la1Given = TRUE;
            break;
        case BSIM4v6_MOD_LA2:
            mod->BSIM4v6la2 = value->rValue;
            mod->BSIM4v6la2Given = TRUE;
            break;
        case BSIM4v6_MOD_LAT:
            mod->BSIM4v6lat = value->rValue;
            mod->BSIM4v6latGiven = TRUE;
            break;
        case BSIM4v6_MOD_LKETA:
            mod->BSIM4v6lketa = value->rValue;
            mod->BSIM4v6lketaGiven = TRUE;
            break;    
        case BSIM4v6_MOD_LNSUB:
            mod->BSIM4v6lnsub = value->rValue;
            mod->BSIM4v6lnsubGiven = TRUE;
            break;
        case BSIM4v6_MOD_LNDEP:
            mod->BSIM4v6lndep = value->rValue;
            mod->BSIM4v6lndepGiven = TRUE;
	    if (mod->BSIM4v6lndep > 1.0e20)
		mod->BSIM4v6lndep *= 1.0e-6;
            break;
        case BSIM4v6_MOD_LNSD:
            mod->BSIM4v6lnsd = value->rValue;
            mod->BSIM4v6lnsdGiven = TRUE;
            if (mod->BSIM4v6lnsd > 1.0e23)
                mod->BSIM4v6lnsd *= 1.0e-6;
            break;
        case BSIM4v6_MOD_LNGATE:
            mod->BSIM4v6lngate = value->rValue;
            mod->BSIM4v6lngateGiven = TRUE;
	    if (mod->BSIM4v6lngate > 1.0e23)
		mod->BSIM4v6lngate *= 1.0e-6;
            break;
        case BSIM4v6_MOD_LGAMMA1:
            mod->BSIM4v6lgamma1 = value->rValue;
            mod->BSIM4v6lgamma1Given = TRUE;
            break;
        case BSIM4v6_MOD_LGAMMA2:
            mod->BSIM4v6lgamma2 = value->rValue;
            mod->BSIM4v6lgamma2Given = TRUE;
            break;
        case BSIM4v6_MOD_LVBX:
            mod->BSIM4v6lvbx = value->rValue;
            mod->BSIM4v6lvbxGiven = TRUE;
            break;
        case BSIM4v6_MOD_LVBM:
            mod->BSIM4v6lvbm = value->rValue;
            mod->BSIM4v6lvbmGiven = TRUE;
            break;
        case BSIM4v6_MOD_LXT:
            mod->BSIM4v6lxt = value->rValue;
            mod->BSIM4v6lxtGiven = TRUE;
            break;
        case  BSIM4v6_MOD_LK1:
            mod->BSIM4v6lk1 = value->rValue;
            mod->BSIM4v6lk1Given = TRUE;
            break;
        case  BSIM4v6_MOD_LKT1:
            mod->BSIM4v6lkt1 = value->rValue;
            mod->BSIM4v6lkt1Given = TRUE;
            break;
        case  BSIM4v6_MOD_LKT1L:
            mod->BSIM4v6lkt1l = value->rValue;
            mod->BSIM4v6lkt1lGiven = TRUE;
            break;
        case  BSIM4v6_MOD_LKT2:
            mod->BSIM4v6lkt2 = value->rValue;
            mod->BSIM4v6lkt2Given = TRUE;
            break;
        case  BSIM4v6_MOD_LK2:
            mod->BSIM4v6lk2 = value->rValue;
            mod->BSIM4v6lk2Given = TRUE;
            break;
        case  BSIM4v6_MOD_LK3:
            mod->BSIM4v6lk3 = value->rValue;
            mod->BSIM4v6lk3Given = TRUE;
            break;
        case  BSIM4v6_MOD_LK3B:
            mod->BSIM4v6lk3b = value->rValue;
            mod->BSIM4v6lk3bGiven = TRUE;
            break;
        case  BSIM4v6_MOD_LLPE0:
            mod->BSIM4v6llpe0 = value->rValue;
            mod->BSIM4v6llpe0Given = TRUE;
            break;
        case  BSIM4v6_MOD_LLPEB:
            mod->BSIM4v6llpeb = value->rValue;
            mod->BSIM4v6llpebGiven = TRUE;
            break;
        case  BSIM4v6_MOD_LDVTP0:
            mod->BSIM4v6ldvtp0 = value->rValue;
            mod->BSIM4v6ldvtp0Given = TRUE;
            break;
        case  BSIM4v6_MOD_LDVTP1:
            mod->BSIM4v6ldvtp1 = value->rValue;
            mod->BSIM4v6ldvtp1Given = TRUE;
            break;
        case  BSIM4v6_MOD_LW0:
            mod->BSIM4v6lw0 = value->rValue;
            mod->BSIM4v6lw0Given = TRUE;
            break;
        case  BSIM4v6_MOD_LDVT0:               
            mod->BSIM4v6ldvt0 = value->rValue;
            mod->BSIM4v6ldvt0Given = TRUE;
            break;
        case  BSIM4v6_MOD_LDVT1:             
            mod->BSIM4v6ldvt1 = value->rValue;
            mod->BSIM4v6ldvt1Given = TRUE;
            break;
        case  BSIM4v6_MOD_LDVT2:             
            mod->BSIM4v6ldvt2 = value->rValue;
            mod->BSIM4v6ldvt2Given = TRUE;
            break;
        case  BSIM4v6_MOD_LDVT0W:               
            mod->BSIM4v6ldvt0w = value->rValue;
            mod->BSIM4v6ldvt0wGiven = TRUE;
            break;
        case  BSIM4v6_MOD_LDVT1W:             
            mod->BSIM4v6ldvt1w = value->rValue;
            mod->BSIM4v6ldvt1wGiven = TRUE;
            break;
        case  BSIM4v6_MOD_LDVT2W:             
            mod->BSIM4v6ldvt2w = value->rValue;
            mod->BSIM4v6ldvt2wGiven = TRUE;
            break;
        case  BSIM4v6_MOD_LDROUT:             
            mod->BSIM4v6ldrout = value->rValue;
            mod->BSIM4v6ldroutGiven = TRUE;
            break;
        case  BSIM4v6_MOD_LDSUB:             
            mod->BSIM4v6ldsub = value->rValue;
            mod->BSIM4v6ldsubGiven = TRUE;
            break;
        case BSIM4v6_MOD_LVTH0:
            mod->BSIM4v6lvth0 = value->rValue;
            mod->BSIM4v6lvth0Given = TRUE;
            break;
        case BSIM4v6_MOD_LUA:
            mod->BSIM4v6lua = value->rValue;
            mod->BSIM4v6luaGiven = TRUE;
            break;
        case BSIM4v6_MOD_LUA1:
            mod->BSIM4v6lua1 = value->rValue;
            mod->BSIM4v6lua1Given = TRUE;
            break;
        case BSIM4v6_MOD_LUB:
            mod->BSIM4v6lub = value->rValue;
            mod->BSIM4v6lubGiven = TRUE;
            break;
        case BSIM4v6_MOD_LUB1:
            mod->BSIM4v6lub1 = value->rValue;
            mod->BSIM4v6lub1Given = TRUE;
            break;
        case BSIM4v6_MOD_LUC:
            mod->BSIM4v6luc = value->rValue;
            mod->BSIM4v6lucGiven = TRUE;
            break;
        case BSIM4v6_MOD_LUC1:
            mod->BSIM4v6luc1 = value->rValue;
            mod->BSIM4v6luc1Given = TRUE;
            break;
        case  BSIM4v6_MOD_LU0 :
            mod->BSIM4v6lu0 = value->rValue;
            mod->BSIM4v6lu0Given = TRUE;
            break;
        case  BSIM4v6_MOD_LUTE :
            mod->BSIM4v6lute = value->rValue;
            mod->BSIM4v6luteGiven = TRUE;
            break;
		case  BSIM4v6_MOD_LUCSTE :
            mod->BSIM4v6lucste = value->rValue;
            mod->BSIM4v6lucsteGiven = TRUE;
            break;
        case BSIM4v6_MOD_LVOFF:
            mod->BSIM4v6lvoff = value->rValue;
            mod->BSIM4v6lvoffGiven = TRUE;
            break;
        case BSIM4v6_MOD_LTVOFF:
            mod->BSIM4v6ltvoff = value->rValue;
            mod->BSIM4v6ltvoffGiven = TRUE;
            break;
        case BSIM4v6_MOD_LMINV:
            mod->BSIM4v6lminv = value->rValue;
            mod->BSIM4v6lminvGiven = TRUE;
            break;
        case BSIM4v6_MOD_LMINVCV:
            mod->BSIM4v6lminvcv = value->rValue;
            mod->BSIM4v6lminvcvGiven = TRUE;
            break;
        case BSIM4v6_MOD_LFPROUT:
            mod->BSIM4v6lfprout = value->rValue;
            mod->BSIM4v6lfproutGiven = TRUE;
            break;
        case BSIM4v6_MOD_LPDITS:
            mod->BSIM4v6lpdits = value->rValue;
            mod->BSIM4v6lpditsGiven = TRUE;
            break;
        case BSIM4v6_MOD_LPDITSD:
            mod->BSIM4v6lpditsd = value->rValue;
            mod->BSIM4v6lpditsdGiven = TRUE;
            break;
        case  BSIM4v6_MOD_LDELTA :
            mod->BSIM4v6ldelta = value->rValue;
            mod->BSIM4v6ldeltaGiven = TRUE;
            break;
        case BSIM4v6_MOD_LRDSW:
            mod->BSIM4v6lrdsw = value->rValue;
            mod->BSIM4v6lrdswGiven = TRUE;
            break;                     
        case BSIM4v6_MOD_LRDW:
            mod->BSIM4v6lrdw = value->rValue;
            mod->BSIM4v6lrdwGiven = TRUE;
            break;
        case BSIM4v6_MOD_LRSW:
            mod->BSIM4v6lrsw = value->rValue;
            mod->BSIM4v6lrswGiven = TRUE;
            break;
        case BSIM4v6_MOD_LPRWB:
            mod->BSIM4v6lprwb = value->rValue;
            mod->BSIM4v6lprwbGiven = TRUE;
            break;                     
        case BSIM4v6_MOD_LPRWG:
            mod->BSIM4v6lprwg = value->rValue;
            mod->BSIM4v6lprwgGiven = TRUE;
            break;                     
        case BSIM4v6_MOD_LPRT:
            mod->BSIM4v6lprt = value->rValue;
            mod->BSIM4v6lprtGiven = TRUE;
            break;                     
        case BSIM4v6_MOD_LETA0:
            mod->BSIM4v6leta0 = value->rValue;
            mod->BSIM4v6leta0Given = TRUE;
            break;                 
        case BSIM4v6_MOD_LETAB:
            mod->BSIM4v6letab = value->rValue;
            mod->BSIM4v6letabGiven = TRUE;
            break;                 
        case BSIM4v6_MOD_LPCLM:
            mod->BSIM4v6lpclm = value->rValue;
            mod->BSIM4v6lpclmGiven = TRUE;
            break;                 
        case BSIM4v6_MOD_LPDIBL1:
            mod->BSIM4v6lpdibl1 = value->rValue;
            mod->BSIM4v6lpdibl1Given = TRUE;
            break;                 
        case BSIM4v6_MOD_LPDIBL2:
            mod->BSIM4v6lpdibl2 = value->rValue;
            mod->BSIM4v6lpdibl2Given = TRUE;
            break;                 
        case BSIM4v6_MOD_LPDIBLB:
            mod->BSIM4v6lpdiblb = value->rValue;
            mod->BSIM4v6lpdiblbGiven = TRUE;
            break;                 
        case BSIM4v6_MOD_LPSCBE1:
            mod->BSIM4v6lpscbe1 = value->rValue;
            mod->BSIM4v6lpscbe1Given = TRUE;
            break;                 
        case BSIM4v6_MOD_LPSCBE2:
            mod->BSIM4v6lpscbe2 = value->rValue;
            mod->BSIM4v6lpscbe2Given = TRUE;
            break;                 
        case BSIM4v6_MOD_LPVAG:
            mod->BSIM4v6lpvag = value->rValue;
            mod->BSIM4v6lpvagGiven = TRUE;
            break;                 
        case  BSIM4v6_MOD_LWR :
            mod->BSIM4v6lwr = value->rValue;
            mod->BSIM4v6lwrGiven = TRUE;
            break;
        case  BSIM4v6_MOD_LDWG :
            mod->BSIM4v6ldwg = value->rValue;
            mod->BSIM4v6ldwgGiven = TRUE;
            break;
        case  BSIM4v6_MOD_LDWB :
            mod->BSIM4v6ldwb = value->rValue;
            mod->BSIM4v6ldwbGiven = TRUE;
            break;
        case  BSIM4v6_MOD_LB0 :
            mod->BSIM4v6lb0 = value->rValue;
            mod->BSIM4v6lb0Given = TRUE;
            break;
        case  BSIM4v6_MOD_LB1 :
            mod->BSIM4v6lb1 = value->rValue;
            mod->BSIM4v6lb1Given = TRUE;
            break;
        case  BSIM4v6_MOD_LALPHA0 :
            mod->BSIM4v6lalpha0 = value->rValue;
            mod->BSIM4v6lalpha0Given = TRUE;
            break;
        case  BSIM4v6_MOD_LALPHA1 :
            mod->BSIM4v6lalpha1 = value->rValue;
            mod->BSIM4v6lalpha1Given = TRUE;
            break;
        case  BSIM4v6_MOD_LBETA0 :
            mod->BSIM4v6lbeta0 = value->rValue;
            mod->BSIM4v6lbeta0Given = TRUE;
            break;
        case  BSIM4v6_MOD_LPHIN :
            mod->BSIM4v6lphin = value->rValue;
            mod->BSIM4v6lphinGiven = TRUE;
            break;
        case  BSIM4v6_MOD_LAGIDL :
            mod->BSIM4v6lagidl = value->rValue;
            mod->BSIM4v6lagidlGiven = TRUE;
            break;
        case  BSIM4v6_MOD_LBGIDL :
            mod->BSIM4v6lbgidl = value->rValue;
            mod->BSIM4v6lbgidlGiven = TRUE;
            break;
        case  BSIM4v6_MOD_LCGIDL :
            mod->BSIM4v6lcgidl = value->rValue;
            mod->BSIM4v6lcgidlGiven = TRUE;
            break;
        case  BSIM4v6_MOD_LEGIDL :
            mod->BSIM4v6legidl = value->rValue;
            mod->BSIM4v6legidlGiven = TRUE;
            break;
        case  BSIM4v6_MOD_LAGISL :
            mod->BSIM4v6lagisl = value->rValue;
            mod->BSIM4v6lagislGiven = TRUE;
            break;
        case  BSIM4v6_MOD_LBGISL :
            mod->BSIM4v6lbgisl = value->rValue;
            mod->BSIM4v6lbgislGiven = TRUE;
            break;
        case  BSIM4v6_MOD_LCGISL :
            mod->BSIM4v6lcgisl = value->rValue;
            mod->BSIM4v6lcgislGiven = TRUE;
            break;
        case  BSIM4v6_MOD_LEGISL :
            mod->BSIM4v6legisl = value->rValue;
            mod->BSIM4v6legislGiven = TRUE;
            break;
        case  BSIM4v6_MOD_LAIGC :
            mod->BSIM4v6laigc = value->rValue;
            mod->BSIM4v6laigcGiven = TRUE;
            break;
        case  BSIM4v6_MOD_LBIGC :
            mod->BSIM4v6lbigc = value->rValue;
            mod->BSIM4v6lbigcGiven = TRUE;
            break;
        case  BSIM4v6_MOD_LCIGC :
            mod->BSIM4v6lcigc = value->rValue;
            mod->BSIM4v6lcigcGiven = TRUE;
            break;
        case  BSIM4v6_MOD_LAIGSD :
            mod->BSIM4v6laigsd = value->rValue;
            mod->BSIM4v6laigsdGiven = TRUE;
            break;
        case  BSIM4v6_MOD_LBIGSD :
            mod->BSIM4v6lbigsd = value->rValue;
            mod->BSIM4v6lbigsdGiven = TRUE;
            break;
        case  BSIM4v6_MOD_LCIGSD :
            mod->BSIM4v6lcigsd = value->rValue;
            mod->BSIM4v6lcigsdGiven = TRUE;
            break;
        case  BSIM4v6_MOD_LAIGS :
            mod->BSIM4v6laigs = value->rValue;
            mod->BSIM4v6laigsGiven = TRUE;
            break;
        case  BSIM4v6_MOD_LBIGS :
            mod->BSIM4v6lbigs = value->rValue;
            mod->BSIM4v6lbigsGiven = TRUE;
            break;
        case  BSIM4v6_MOD_LCIGS :
            mod->BSIM4v6lcigs = value->rValue;
            mod->BSIM4v6lcigsGiven = TRUE;
            break;
        case  BSIM4v6_MOD_LAIGD :
            mod->BSIM4v6laigd = value->rValue;
            mod->BSIM4v6laigdGiven = TRUE;
            break;
        case  BSIM4v6_MOD_LBIGD :
            mod->BSIM4v6lbigd = value->rValue;
            mod->BSIM4v6lbigdGiven = TRUE;
            break;
        case  BSIM4v6_MOD_LCIGD :
            mod->BSIM4v6lcigd = value->rValue;
            mod->BSIM4v6lcigdGiven = TRUE;
            break;
        case  BSIM4v6_MOD_LAIGBACC :
            mod->BSIM4v6laigbacc = value->rValue;
            mod->BSIM4v6laigbaccGiven = TRUE;
            break;
        case  BSIM4v6_MOD_LBIGBACC :
            mod->BSIM4v6lbigbacc = value->rValue;
            mod->BSIM4v6lbigbaccGiven = TRUE;
            break;
        case  BSIM4v6_MOD_LCIGBACC :
            mod->BSIM4v6lcigbacc = value->rValue;
            mod->BSIM4v6lcigbaccGiven = TRUE;
            break;
        case  BSIM4v6_MOD_LAIGBINV :
            mod->BSIM4v6laigbinv = value->rValue;
            mod->BSIM4v6laigbinvGiven = TRUE;
            break;
        case  BSIM4v6_MOD_LBIGBINV :
            mod->BSIM4v6lbigbinv = value->rValue;
            mod->BSIM4v6lbigbinvGiven = TRUE;
            break;
        case  BSIM4v6_MOD_LCIGBINV :
            mod->BSIM4v6lcigbinv = value->rValue;
            mod->BSIM4v6lcigbinvGiven = TRUE;
            break;
        case  BSIM4v6_MOD_LNIGC :
            mod->BSIM4v6lnigc = value->rValue;
            mod->BSIM4v6lnigcGiven = TRUE;
            break;
        case  BSIM4v6_MOD_LNIGBINV :
            mod->BSIM4v6lnigbinv = value->rValue;
            mod->BSIM4v6lnigbinvGiven = TRUE;
            break;
        case  BSIM4v6_MOD_LNIGBACC :
            mod->BSIM4v6lnigbacc = value->rValue;
            mod->BSIM4v6lnigbaccGiven = TRUE;
            break;
        case  BSIM4v6_MOD_LNTOX :
            mod->BSIM4v6lntox = value->rValue;
            mod->BSIM4v6lntoxGiven = TRUE;
            break;
        case  BSIM4v6_MOD_LEIGBINV :
            mod->BSIM4v6leigbinv = value->rValue;
            mod->BSIM4v6leigbinvGiven = TRUE;
            break;
        case  BSIM4v6_MOD_LPIGCD :
            mod->BSIM4v6lpigcd = value->rValue;
            mod->BSIM4v6lpigcdGiven = TRUE;
            break;
        case  BSIM4v6_MOD_LPOXEDGE :
            mod->BSIM4v6lpoxedge = value->rValue;
            mod->BSIM4v6lpoxedgeGiven = TRUE;
            break;
        case  BSIM4v6_MOD_LXRCRG1 :
            mod->BSIM4v6lxrcrg1 = value->rValue;
            mod->BSIM4v6lxrcrg1Given = TRUE;
            break;
        case  BSIM4v6_MOD_LXRCRG2 :
            mod->BSIM4v6lxrcrg2 = value->rValue;
            mod->BSIM4v6lxrcrg2Given = TRUE;
            break;
        case  BSIM4v6_MOD_LLAMBDA :
            mod->BSIM4v6llambda = value->rValue;
            mod->BSIM4v6llambdaGiven = TRUE;
            break;
        case  BSIM4v6_MOD_LVTL :
            mod->BSIM4v6lvtl = value->rValue;
            mod->BSIM4v6lvtlGiven = TRUE;
            break;
        case  BSIM4v6_MOD_LXN:
            mod->BSIM4v6lxn = value->rValue;
            mod->BSIM4v6lxnGiven = TRUE;
            break;
        case  BSIM4v6_MOD_LVFBSDOFF:
            mod->BSIM4v6lvfbsdoff = value->rValue;
            mod->BSIM4v6lvfbsdoffGiven = TRUE;
            break;
        case  BSIM4v6_MOD_LTVFBSDOFF:
            mod->BSIM4v6ltvfbsdoff = value->rValue;
            mod->BSIM4v6ltvfbsdoffGiven = TRUE;
            break;
        case  BSIM4v6_MOD_LEU :
            mod->BSIM4v6leu = value->rValue;
            mod->BSIM4v6leuGiven = TRUE;
            break;
		case  BSIM4v6_MOD_LUCS :
            mod->BSIM4v6lucs = value->rValue;
            mod->BSIM4v6lucsGiven = TRUE;
            break;
        case  BSIM4v6_MOD_LVFB :
            mod->BSIM4v6lvfb = value->rValue;
            mod->BSIM4v6lvfbGiven = TRUE;
            break;
        case  BSIM4v6_MOD_LCGSL :
            mod->BSIM4v6lcgsl = value->rValue;
            mod->BSIM4v6lcgslGiven = TRUE;
            break;
        case  BSIM4v6_MOD_LCGDL :
            mod->BSIM4v6lcgdl = value->rValue;
            mod->BSIM4v6lcgdlGiven = TRUE;
            break;
        case  BSIM4v6_MOD_LCKAPPAS :
            mod->BSIM4v6lckappas = value->rValue;
            mod->BSIM4v6lckappasGiven = TRUE;
            break;
        case  BSIM4v6_MOD_LCKAPPAD :
            mod->BSIM4v6lckappad = value->rValue;
            mod->BSIM4v6lckappadGiven = TRUE;
            break;
        case  BSIM4v6_MOD_LCF :
            mod->BSIM4v6lcf = value->rValue;
            mod->BSIM4v6lcfGiven = TRUE;
            break;
        case  BSIM4v6_MOD_LCLC :
            mod->BSIM4v6lclc = value->rValue;
            mod->BSIM4v6lclcGiven = TRUE;
            break;
        case  BSIM4v6_MOD_LCLE :
            mod->BSIM4v6lcle = value->rValue;
            mod->BSIM4v6lcleGiven = TRUE;
            break;
        case  BSIM4v6_MOD_LVFBCV :
            mod->BSIM4v6lvfbcv = value->rValue;
            mod->BSIM4v6lvfbcvGiven = TRUE;
            break;
        case  BSIM4v6_MOD_LACDE :
            mod->BSIM4v6lacde = value->rValue;
            mod->BSIM4v6lacdeGiven = TRUE;
            break;
        case  BSIM4v6_MOD_LMOIN :
            mod->BSIM4v6lmoin = value->rValue;
            mod->BSIM4v6lmoinGiven = TRUE;
            break;
        case  BSIM4v6_MOD_LNOFF :
            mod->BSIM4v6lnoff = value->rValue;
            mod->BSIM4v6lnoffGiven = TRUE;
            break;
        case  BSIM4v6_MOD_LVOFFCV :
            mod->BSIM4v6lvoffcv = value->rValue;
            mod->BSIM4v6lvoffcvGiven = TRUE;
            break;

	/* Width dependence */
        case  BSIM4v6_MOD_WCDSC :
            mod->BSIM4v6wcdsc = value->rValue;
            mod->BSIM4v6wcdscGiven = TRUE;
            break;
       
       
         case  BSIM4v6_MOD_WCDSCB :
            mod->BSIM4v6wcdscb = value->rValue;
            mod->BSIM4v6wcdscbGiven = TRUE;
            break;
         case  BSIM4v6_MOD_WCDSCD :
            mod->BSIM4v6wcdscd = value->rValue;
            mod->BSIM4v6wcdscdGiven = TRUE;
            break;
        case  BSIM4v6_MOD_WCIT :
            mod->BSIM4v6wcit = value->rValue;
            mod->BSIM4v6wcitGiven = TRUE;
            break;
        case  BSIM4v6_MOD_WNFACTOR :
            mod->BSIM4v6wnfactor = value->rValue;
            mod->BSIM4v6wnfactorGiven = TRUE;
            break;
        case BSIM4v6_MOD_WXJ:
            mod->BSIM4v6wxj = value->rValue;
            mod->BSIM4v6wxjGiven = TRUE;
            break;
        case BSIM4v6_MOD_WVSAT:
            mod->BSIM4v6wvsat = value->rValue;
            mod->BSIM4v6wvsatGiven = TRUE;
            break;


        case BSIM4v6_MOD_WA0:
            mod->BSIM4v6wa0 = value->rValue;
            mod->BSIM4v6wa0Given = TRUE;
            break;
        case BSIM4v6_MOD_WAGS:
            mod->BSIM4v6wags = value->rValue;
            mod->BSIM4v6wagsGiven = TRUE;
            break;
        case BSIM4v6_MOD_WA1:
            mod->BSIM4v6wa1 = value->rValue;
            mod->BSIM4v6wa1Given = TRUE;
            break;
        case BSIM4v6_MOD_WA2:
            mod->BSIM4v6wa2 = value->rValue;
            mod->BSIM4v6wa2Given = TRUE;
            break;
        case BSIM4v6_MOD_WAT:
            mod->BSIM4v6wat = value->rValue;
            mod->BSIM4v6watGiven = TRUE;
            break;
        case BSIM4v6_MOD_WKETA:
            mod->BSIM4v6wketa = value->rValue;
            mod->BSIM4v6wketaGiven = TRUE;
            break;    
        case BSIM4v6_MOD_WNSUB:
            mod->BSIM4v6wnsub = value->rValue;
            mod->BSIM4v6wnsubGiven = TRUE;
            break;
        case BSIM4v6_MOD_WNDEP:
            mod->BSIM4v6wndep = value->rValue;
            mod->BSIM4v6wndepGiven = TRUE;
	    if (mod->BSIM4v6wndep > 1.0e20)
		mod->BSIM4v6wndep *= 1.0e-6;
            break;
        case BSIM4v6_MOD_WNSD:
            mod->BSIM4v6wnsd = value->rValue;
            mod->BSIM4v6wnsdGiven = TRUE;
            if (mod->BSIM4v6wnsd > 1.0e23)
                mod->BSIM4v6wnsd *= 1.0e-6;
            break;
        case BSIM4v6_MOD_WNGATE:
            mod->BSIM4v6wngate = value->rValue;
            mod->BSIM4v6wngateGiven = TRUE;
	    if (mod->BSIM4v6wngate > 1.0e23)
		mod->BSIM4v6wngate *= 1.0e-6;
            break;
        case BSIM4v6_MOD_WGAMMA1:
            mod->BSIM4v6wgamma1 = value->rValue;
            mod->BSIM4v6wgamma1Given = TRUE;
            break;
        case BSIM4v6_MOD_WGAMMA2:
            mod->BSIM4v6wgamma2 = value->rValue;
            mod->BSIM4v6wgamma2Given = TRUE;
            break;
        case BSIM4v6_MOD_WVBX:
            mod->BSIM4v6wvbx = value->rValue;
            mod->BSIM4v6wvbxGiven = TRUE;
            break;
        case BSIM4v6_MOD_WVBM:
            mod->BSIM4v6wvbm = value->rValue;
            mod->BSIM4v6wvbmGiven = TRUE;
            break;
        case BSIM4v6_MOD_WXT:
            mod->BSIM4v6wxt = value->rValue;
            mod->BSIM4v6wxtGiven = TRUE;
            break;
        case  BSIM4v6_MOD_WK1:
            mod->BSIM4v6wk1 = value->rValue;
            mod->BSIM4v6wk1Given = TRUE;
            break;
        case  BSIM4v6_MOD_WKT1:
            mod->BSIM4v6wkt1 = value->rValue;
            mod->BSIM4v6wkt1Given = TRUE;
            break;
        case  BSIM4v6_MOD_WKT1L:
            mod->BSIM4v6wkt1l = value->rValue;
            mod->BSIM4v6wkt1lGiven = TRUE;
            break;
        case  BSIM4v6_MOD_WKT2:
            mod->BSIM4v6wkt2 = value->rValue;
            mod->BSIM4v6wkt2Given = TRUE;
            break;
        case  BSIM4v6_MOD_WK2:
            mod->BSIM4v6wk2 = value->rValue;
            mod->BSIM4v6wk2Given = TRUE;
            break;
        case  BSIM4v6_MOD_WK3:
            mod->BSIM4v6wk3 = value->rValue;
            mod->BSIM4v6wk3Given = TRUE;
            break;
        case  BSIM4v6_MOD_WK3B:
            mod->BSIM4v6wk3b = value->rValue;
            mod->BSIM4v6wk3bGiven = TRUE;
            break;
        case  BSIM4v6_MOD_WLPE0:
            mod->BSIM4v6wlpe0 = value->rValue;
            mod->BSIM4v6wlpe0Given = TRUE;
            break;
        case  BSIM4v6_MOD_WLPEB:
            mod->BSIM4v6wlpeb = value->rValue;
            mod->BSIM4v6wlpebGiven = TRUE;
            break;
        case  BSIM4v6_MOD_WDVTP0:
            mod->BSIM4v6wdvtp0 = value->rValue;
            mod->BSIM4v6wdvtp0Given = TRUE;
            break;
        case  BSIM4v6_MOD_WDVTP1:
            mod->BSIM4v6wdvtp1 = value->rValue;
            mod->BSIM4v6wdvtp1Given = TRUE;
            break;
        case  BSIM4v6_MOD_WW0:
            mod->BSIM4v6ww0 = value->rValue;
            mod->BSIM4v6ww0Given = TRUE;
            break;
        case  BSIM4v6_MOD_WDVT0:               
            mod->BSIM4v6wdvt0 = value->rValue;
            mod->BSIM4v6wdvt0Given = TRUE;
            break;
        case  BSIM4v6_MOD_WDVT1:             
            mod->BSIM4v6wdvt1 = value->rValue;
            mod->BSIM4v6wdvt1Given = TRUE;
            break;
        case  BSIM4v6_MOD_WDVT2:             
            mod->BSIM4v6wdvt2 = value->rValue;
            mod->BSIM4v6wdvt2Given = TRUE;
            break;
        case  BSIM4v6_MOD_WDVT0W:               
            mod->BSIM4v6wdvt0w = value->rValue;
            mod->BSIM4v6wdvt0wGiven = TRUE;
            break;
        case  BSIM4v6_MOD_WDVT1W:             
            mod->BSIM4v6wdvt1w = value->rValue;
            mod->BSIM4v6wdvt1wGiven = TRUE;
            break;
        case  BSIM4v6_MOD_WDVT2W:             
            mod->BSIM4v6wdvt2w = value->rValue;
            mod->BSIM4v6wdvt2wGiven = TRUE;
            break;
        case  BSIM4v6_MOD_WDROUT:             
            mod->BSIM4v6wdrout = value->rValue;
            mod->BSIM4v6wdroutGiven = TRUE;
            break;
        case  BSIM4v6_MOD_WDSUB:             
            mod->BSIM4v6wdsub = value->rValue;
            mod->BSIM4v6wdsubGiven = TRUE;
            break;
        case BSIM4v6_MOD_WVTH0:
            mod->BSIM4v6wvth0 = value->rValue;
            mod->BSIM4v6wvth0Given = TRUE;
            break;
        case BSIM4v6_MOD_WUA:
            mod->BSIM4v6wua = value->rValue;
            mod->BSIM4v6wuaGiven = TRUE;
            break;
        case BSIM4v6_MOD_WUA1:
            mod->BSIM4v6wua1 = value->rValue;
            mod->BSIM4v6wua1Given = TRUE;
            break;
        case BSIM4v6_MOD_WUB:
            mod->BSIM4v6wub = value->rValue;
            mod->BSIM4v6wubGiven = TRUE;
            break;
        case BSIM4v6_MOD_WUB1:
            mod->BSIM4v6wub1 = value->rValue;
            mod->BSIM4v6wub1Given = TRUE;
            break;
        case BSIM4v6_MOD_WUC:
            mod->BSIM4v6wuc = value->rValue;
            mod->BSIM4v6wucGiven = TRUE;
            break;
        case BSIM4v6_MOD_WUC1:
            mod->BSIM4v6wuc1 = value->rValue;
            mod->BSIM4v6wuc1Given = TRUE;
            break;
        case  BSIM4v6_MOD_WU0 :
            mod->BSIM4v6wu0 = value->rValue;
            mod->BSIM4v6wu0Given = TRUE;
            break;
        case  BSIM4v6_MOD_WUTE :
            mod->BSIM4v6wute = value->rValue;
            mod->BSIM4v6wuteGiven = TRUE;
            break;
		case  BSIM4v6_MOD_WUCSTE :
            mod->BSIM4v6wucste = value->rValue;
            mod->BSIM4v6wucsteGiven = TRUE;
            break;
        case BSIM4v6_MOD_WVOFF:
            mod->BSIM4v6wvoff = value->rValue;
            mod->BSIM4v6wvoffGiven = TRUE;
            break;
        case BSIM4v6_MOD_WTVOFF:
            mod->BSIM4v6wtvoff = value->rValue;
            mod->BSIM4v6wtvoffGiven = TRUE;
            break;
        case BSIM4v6_MOD_WMINV:
            mod->BSIM4v6wminv = value->rValue;
            mod->BSIM4v6wminvGiven = TRUE;
            break;
        case BSIM4v6_MOD_WMINVCV:
            mod->BSIM4v6wminvcv = value->rValue;
            mod->BSIM4v6wminvcvGiven = TRUE;
            break;
        case BSIM4v6_MOD_WFPROUT:
            mod->BSIM4v6wfprout = value->rValue;
            mod->BSIM4v6wfproutGiven = TRUE;
            break;
        case BSIM4v6_MOD_WPDITS:
            mod->BSIM4v6wpdits = value->rValue;
            mod->BSIM4v6wpditsGiven = TRUE;
            break;
        case BSIM4v6_MOD_WPDITSD:
            mod->BSIM4v6wpditsd = value->rValue;
            mod->BSIM4v6wpditsdGiven = TRUE;
            break;
        case  BSIM4v6_MOD_WDELTA :
            mod->BSIM4v6wdelta = value->rValue;
            mod->BSIM4v6wdeltaGiven = TRUE;
            break;
        case BSIM4v6_MOD_WRDSW:
            mod->BSIM4v6wrdsw = value->rValue;
            mod->BSIM4v6wrdswGiven = TRUE;
            break;                     
        case BSIM4v6_MOD_WRDW:
            mod->BSIM4v6wrdw = value->rValue;
            mod->BSIM4v6wrdwGiven = TRUE;
            break;
        case BSIM4v6_MOD_WRSW:
            mod->BSIM4v6wrsw = value->rValue;
            mod->BSIM4v6wrswGiven = TRUE;
            break;
        case BSIM4v6_MOD_WPRWB:
            mod->BSIM4v6wprwb = value->rValue;
            mod->BSIM4v6wprwbGiven = TRUE;
            break;                     
        case BSIM4v6_MOD_WPRWG:
            mod->BSIM4v6wprwg = value->rValue;
            mod->BSIM4v6wprwgGiven = TRUE;
            break;                     
        case BSIM4v6_MOD_WPRT:
            mod->BSIM4v6wprt = value->rValue;
            mod->BSIM4v6wprtGiven = TRUE;
            break;                     
        case BSIM4v6_MOD_WETA0:
            mod->BSIM4v6weta0 = value->rValue;
            mod->BSIM4v6weta0Given = TRUE;
            break;                 
        case BSIM4v6_MOD_WETAB:
            mod->BSIM4v6wetab = value->rValue;
            mod->BSIM4v6wetabGiven = TRUE;
            break;                 
        case BSIM4v6_MOD_WPCLM:
            mod->BSIM4v6wpclm = value->rValue;
            mod->BSIM4v6wpclmGiven = TRUE;
            break;                 
        case BSIM4v6_MOD_WPDIBL1:
            mod->BSIM4v6wpdibl1 = value->rValue;
            mod->BSIM4v6wpdibl1Given = TRUE;
            break;                 
        case BSIM4v6_MOD_WPDIBL2:
            mod->BSIM4v6wpdibl2 = value->rValue;
            mod->BSIM4v6wpdibl2Given = TRUE;
            break;                 
        case BSIM4v6_MOD_WPDIBLB:
            mod->BSIM4v6wpdiblb = value->rValue;
            mod->BSIM4v6wpdiblbGiven = TRUE;
            break;                 
        case BSIM4v6_MOD_WPSCBE1:
            mod->BSIM4v6wpscbe1 = value->rValue;
            mod->BSIM4v6wpscbe1Given = TRUE;
            break;                 
        case BSIM4v6_MOD_WPSCBE2:
            mod->BSIM4v6wpscbe2 = value->rValue;
            mod->BSIM4v6wpscbe2Given = TRUE;
            break;                 
        case BSIM4v6_MOD_WPVAG:
            mod->BSIM4v6wpvag = value->rValue;
            mod->BSIM4v6wpvagGiven = TRUE;
            break;                 
        case  BSIM4v6_MOD_WWR :
            mod->BSIM4v6wwr = value->rValue;
            mod->BSIM4v6wwrGiven = TRUE;
            break;
        case  BSIM4v6_MOD_WDWG :
            mod->BSIM4v6wdwg = value->rValue;
            mod->BSIM4v6wdwgGiven = TRUE;
            break;
        case  BSIM4v6_MOD_WDWB :
            mod->BSIM4v6wdwb = value->rValue;
            mod->BSIM4v6wdwbGiven = TRUE;
            break;
        case  BSIM4v6_MOD_WB0 :
            mod->BSIM4v6wb0 = value->rValue;
            mod->BSIM4v6wb0Given = TRUE;
            break;
        case  BSIM4v6_MOD_WB1 :
            mod->BSIM4v6wb1 = value->rValue;
            mod->BSIM4v6wb1Given = TRUE;
            break;
        case  BSIM4v6_MOD_WALPHA0 :
            mod->BSIM4v6walpha0 = value->rValue;
            mod->BSIM4v6walpha0Given = TRUE;
            break;
        case  BSIM4v6_MOD_WALPHA1 :
            mod->BSIM4v6walpha1 = value->rValue;
            mod->BSIM4v6walpha1Given = TRUE;
            break;
        case  BSIM4v6_MOD_WBETA0 :
            mod->BSIM4v6wbeta0 = value->rValue;
            mod->BSIM4v6wbeta0Given = TRUE;
            break;
        case  BSIM4v6_MOD_WPHIN :
            mod->BSIM4v6wphin = value->rValue;
            mod->BSIM4v6wphinGiven = TRUE;
            break;
        case  BSIM4v6_MOD_WAGIDL :
            mod->BSIM4v6wagidl = value->rValue;
            mod->BSIM4v6wagidlGiven = TRUE;
            break;
        case  BSIM4v6_MOD_WBGIDL :
            mod->BSIM4v6wbgidl = value->rValue;
            mod->BSIM4v6wbgidlGiven = TRUE;
            break;
        case  BSIM4v6_MOD_WCGIDL :
            mod->BSIM4v6wcgidl = value->rValue;
            mod->BSIM4v6wcgidlGiven = TRUE;
            break;
        case  BSIM4v6_MOD_WEGIDL :
            mod->BSIM4v6wegidl = value->rValue;
            mod->BSIM4v6wegidlGiven = TRUE;
            break;
        case  BSIM4v6_MOD_WAGISL :
            mod->BSIM4v6wagisl = value->rValue;
            mod->BSIM4v6wagislGiven = TRUE;
            break;
        case  BSIM4v6_MOD_WBGISL :
            mod->BSIM4v6wbgisl = value->rValue;
            mod->BSIM4v6wbgislGiven = TRUE;
            break;
        case  BSIM4v6_MOD_WCGISL :
            mod->BSIM4v6wcgisl = value->rValue;
            mod->BSIM4v6wcgislGiven = TRUE;
            break;
        case  BSIM4v6_MOD_WEGISL :
            mod->BSIM4v6wegisl = value->rValue;
            mod->BSIM4v6wegislGiven = TRUE;
            break;
        case  BSIM4v6_MOD_WAIGC :
            mod->BSIM4v6waigc = value->rValue;
            mod->BSIM4v6waigcGiven = TRUE;
            break;
        case  BSIM4v6_MOD_WBIGC :
            mod->BSIM4v6wbigc = value->rValue;
            mod->BSIM4v6wbigcGiven = TRUE;
            break;
        case  BSIM4v6_MOD_WCIGC :
            mod->BSIM4v6wcigc = value->rValue;
            mod->BSIM4v6wcigcGiven = TRUE;
            break;
        case  BSIM4v6_MOD_WAIGSD :
            mod->BSIM4v6waigsd = value->rValue;
            mod->BSIM4v6waigsdGiven = TRUE;
            break;
        case  BSIM4v6_MOD_WBIGSD :
            mod->BSIM4v6wbigsd = value->rValue;
            mod->BSIM4v6wbigsdGiven = TRUE;
            break;
        case  BSIM4v6_MOD_WCIGSD :
            mod->BSIM4v6wcigsd = value->rValue;
            mod->BSIM4v6wcigsdGiven = TRUE;
            break;
        case  BSIM4v6_MOD_WAIGS :
            mod->BSIM4v6waigs = value->rValue;
            mod->BSIM4v6waigsGiven = TRUE;
            break;
        case  BSIM4v6_MOD_WBIGS :
            mod->BSIM4v6wbigs = value->rValue;
            mod->BSIM4v6wbigsGiven = TRUE;
            break;
        case  BSIM4v6_MOD_WCIGS :
            mod->BSIM4v6wcigs = value->rValue;
            mod->BSIM4v6wcigsGiven = TRUE;
            break;
        case  BSIM4v6_MOD_WAIGD :
            mod->BSIM4v6waigd = value->rValue;
            mod->BSIM4v6waigdGiven = TRUE;
            break;
        case  BSIM4v6_MOD_WBIGD :
            mod->BSIM4v6wbigd = value->rValue;
            mod->BSIM4v6wbigdGiven = TRUE;
            break;
        case  BSIM4v6_MOD_WCIGD :
            mod->BSIM4v6wcigd = value->rValue;
            mod->BSIM4v6wcigdGiven = TRUE;
            break;
        case  BSIM4v6_MOD_WAIGBACC :
            mod->BSIM4v6waigbacc = value->rValue;
            mod->BSIM4v6waigbaccGiven = TRUE;
            break;
        case  BSIM4v6_MOD_WBIGBACC :
            mod->BSIM4v6wbigbacc = value->rValue;
            mod->BSIM4v6wbigbaccGiven = TRUE;
            break;
        case  BSIM4v6_MOD_WCIGBACC :
            mod->BSIM4v6wcigbacc = value->rValue;
            mod->BSIM4v6wcigbaccGiven = TRUE;
            break;
        case  BSIM4v6_MOD_WAIGBINV :
            mod->BSIM4v6waigbinv = value->rValue;
            mod->BSIM4v6waigbinvGiven = TRUE;
            break;
        case  BSIM4v6_MOD_WBIGBINV :
            mod->BSIM4v6wbigbinv = value->rValue;
            mod->BSIM4v6wbigbinvGiven = TRUE;
            break;
        case  BSIM4v6_MOD_WCIGBINV :
            mod->BSIM4v6wcigbinv = value->rValue;
            mod->BSIM4v6wcigbinvGiven = TRUE;
            break;
        case  BSIM4v6_MOD_WNIGC :
            mod->BSIM4v6wnigc = value->rValue;
            mod->BSIM4v6wnigcGiven = TRUE;
            break;
        case  BSIM4v6_MOD_WNIGBINV :
            mod->BSIM4v6wnigbinv = value->rValue;
            mod->BSIM4v6wnigbinvGiven = TRUE;
            break;
        case  BSIM4v6_MOD_WNIGBACC :
            mod->BSIM4v6wnigbacc = value->rValue;
            mod->BSIM4v6wnigbaccGiven = TRUE;
            break;
        case  BSIM4v6_MOD_WNTOX :
            mod->BSIM4v6wntox = value->rValue;
            mod->BSIM4v6wntoxGiven = TRUE;
            break;
        case  BSIM4v6_MOD_WEIGBINV :
            mod->BSIM4v6weigbinv = value->rValue;
            mod->BSIM4v6weigbinvGiven = TRUE;
            break;
        case  BSIM4v6_MOD_WPIGCD :
            mod->BSIM4v6wpigcd = value->rValue;
            mod->BSIM4v6wpigcdGiven = TRUE;
            break;
        case  BSIM4v6_MOD_WPOXEDGE :
            mod->BSIM4v6wpoxedge = value->rValue;
            mod->BSIM4v6wpoxedgeGiven = TRUE;
            break;
        case  BSIM4v6_MOD_WXRCRG1 :
            mod->BSIM4v6wxrcrg1 = value->rValue;
            mod->BSIM4v6wxrcrg1Given = TRUE;
            break;
        case  BSIM4v6_MOD_WXRCRG2 :
            mod->BSIM4v6wxrcrg2 = value->rValue;
            mod->BSIM4v6wxrcrg2Given = TRUE;
            break;
        case  BSIM4v6_MOD_WLAMBDA :
            mod->BSIM4v6wlambda = value->rValue;
            mod->BSIM4v6wlambdaGiven = TRUE;
            break;
        case  BSIM4v6_MOD_WVTL :
            mod->BSIM4v6wvtl = value->rValue;
            mod->BSIM4v6wvtlGiven = TRUE;
            break;
        case  BSIM4v6_MOD_WXN:
            mod->BSIM4v6wxn = value->rValue;
            mod->BSIM4v6wxnGiven = TRUE;
            break;
        case  BSIM4v6_MOD_WVFBSDOFF:
            mod->BSIM4v6wvfbsdoff = value->rValue;
            mod->BSIM4v6wvfbsdoffGiven = TRUE;
            break;
        case  BSIM4v6_MOD_WTVFBSDOFF:
            mod->BSIM4v6wtvfbsdoff = value->rValue;
            mod->BSIM4v6wtvfbsdoffGiven = TRUE;
            break;
        case  BSIM4v6_MOD_WEU :
            mod->BSIM4v6weu = value->rValue;
            mod->BSIM4v6weuGiven = TRUE;
            break;
		 case  BSIM4v6_MOD_WUCS :
            mod->BSIM4v6wucs = value->rValue;
            mod->BSIM4v6wucsGiven = TRUE;
            break;
        case  BSIM4v6_MOD_WVFB :
            mod->BSIM4v6wvfb = value->rValue;
            mod->BSIM4v6wvfbGiven = TRUE;
            break;
        case  BSIM4v6_MOD_WCGSL :
            mod->BSIM4v6wcgsl = value->rValue;
            mod->BSIM4v6wcgslGiven = TRUE;
            break;
        case  BSIM4v6_MOD_WCGDL :
            mod->BSIM4v6wcgdl = value->rValue;
            mod->BSIM4v6wcgdlGiven = TRUE;
            break;
        case  BSIM4v6_MOD_WCKAPPAS :
            mod->BSIM4v6wckappas = value->rValue;
            mod->BSIM4v6wckappasGiven = TRUE;
            break;
        case  BSIM4v6_MOD_WCKAPPAD :
            mod->BSIM4v6wckappad = value->rValue;
            mod->BSIM4v6wckappadGiven = TRUE;
            break;
        case  BSIM4v6_MOD_WCF :
            mod->BSIM4v6wcf = value->rValue;
            mod->BSIM4v6wcfGiven = TRUE;
            break;
        case  BSIM4v6_MOD_WCLC :
            mod->BSIM4v6wclc = value->rValue;
            mod->BSIM4v6wclcGiven = TRUE;
            break;
        case  BSIM4v6_MOD_WCLE :
            mod->BSIM4v6wcle = value->rValue;
            mod->BSIM4v6wcleGiven = TRUE;
            break;
        case  BSIM4v6_MOD_WVFBCV :
            mod->BSIM4v6wvfbcv = value->rValue;
            mod->BSIM4v6wvfbcvGiven = TRUE;
            break;
        case  BSIM4v6_MOD_WACDE :
            mod->BSIM4v6wacde = value->rValue;
            mod->BSIM4v6wacdeGiven = TRUE;
            break;
        case  BSIM4v6_MOD_WMOIN :
            mod->BSIM4v6wmoin = value->rValue;
            mod->BSIM4v6wmoinGiven = TRUE;
            break;
        case  BSIM4v6_MOD_WNOFF :
            mod->BSIM4v6wnoff = value->rValue;
            mod->BSIM4v6wnoffGiven = TRUE;
            break;
        case  BSIM4v6_MOD_WVOFFCV :
            mod->BSIM4v6wvoffcv = value->rValue;
            mod->BSIM4v6wvoffcvGiven = TRUE;
            break;

	/* Cross-term dependence */
        case  BSIM4v6_MOD_PCDSC :
            mod->BSIM4v6pcdsc = value->rValue;
            mod->BSIM4v6pcdscGiven = TRUE;
            break;


        case  BSIM4v6_MOD_PCDSCB :
            mod->BSIM4v6pcdscb = value->rValue;
            mod->BSIM4v6pcdscbGiven = TRUE;
            break;
        case  BSIM4v6_MOD_PCDSCD :
            mod->BSIM4v6pcdscd = value->rValue;
            mod->BSIM4v6pcdscdGiven = TRUE;
            break;
        case  BSIM4v6_MOD_PCIT :
            mod->BSIM4v6pcit = value->rValue;
            mod->BSIM4v6pcitGiven = TRUE;
            break;
        case  BSIM4v6_MOD_PNFACTOR :
            mod->BSIM4v6pnfactor = value->rValue;
            mod->BSIM4v6pnfactorGiven = TRUE;
            break;
        case BSIM4v6_MOD_PXJ:
            mod->BSIM4v6pxj = value->rValue;
            mod->BSIM4v6pxjGiven = TRUE;
            break;
        case BSIM4v6_MOD_PVSAT:
            mod->BSIM4v6pvsat = value->rValue;
            mod->BSIM4v6pvsatGiven = TRUE;
            break;


        case BSIM4v6_MOD_PA0:
            mod->BSIM4v6pa0 = value->rValue;
            mod->BSIM4v6pa0Given = TRUE;
            break;
        case BSIM4v6_MOD_PAGS:
            mod->BSIM4v6pags = value->rValue;
            mod->BSIM4v6pagsGiven = TRUE;
            break;
        case BSIM4v6_MOD_PA1:
            mod->BSIM4v6pa1 = value->rValue;
            mod->BSIM4v6pa1Given = TRUE;
            break;
        case BSIM4v6_MOD_PA2:
            mod->BSIM4v6pa2 = value->rValue;
            mod->BSIM4v6pa2Given = TRUE;
            break;
        case BSIM4v6_MOD_PAT:
            mod->BSIM4v6pat = value->rValue;
            mod->BSIM4v6patGiven = TRUE;
            break;
        case BSIM4v6_MOD_PKETA:
            mod->BSIM4v6pketa = value->rValue;
            mod->BSIM4v6pketaGiven = TRUE;
            break;    
        case BSIM4v6_MOD_PNSUB:
            mod->BSIM4v6pnsub = value->rValue;
            mod->BSIM4v6pnsubGiven = TRUE;
            break;
        case BSIM4v6_MOD_PNDEP:
            mod->BSIM4v6pndep = value->rValue;
            mod->BSIM4v6pndepGiven = TRUE;
	    if (mod->BSIM4v6pndep > 1.0e20)
		mod->BSIM4v6pndep *= 1.0e-6;
            break;
        case BSIM4v6_MOD_PNSD:
            mod->BSIM4v6pnsd = value->rValue;
            mod->BSIM4v6pnsdGiven = TRUE;
            if (mod->BSIM4v6pnsd > 1.0e23)
                mod->BSIM4v6pnsd *= 1.0e-6;
            break;
        case BSIM4v6_MOD_PNGATE:
            mod->BSIM4v6pngate = value->rValue;
            mod->BSIM4v6pngateGiven = TRUE;
	    if (mod->BSIM4v6pngate > 1.0e23)
		mod->BSIM4v6pngate *= 1.0e-6;
            break;
        case BSIM4v6_MOD_PGAMMA1:
            mod->BSIM4v6pgamma1 = value->rValue;
            mod->BSIM4v6pgamma1Given = TRUE;
            break;
        case BSIM4v6_MOD_PGAMMA2:
            mod->BSIM4v6pgamma2 = value->rValue;
            mod->BSIM4v6pgamma2Given = TRUE;
            break;
        case BSIM4v6_MOD_PVBX:
            mod->BSIM4v6pvbx = value->rValue;
            mod->BSIM4v6pvbxGiven = TRUE;
            break;
        case BSIM4v6_MOD_PVBM:
            mod->BSIM4v6pvbm = value->rValue;
            mod->BSIM4v6pvbmGiven = TRUE;
            break;
        case BSIM4v6_MOD_PXT:
            mod->BSIM4v6pxt = value->rValue;
            mod->BSIM4v6pxtGiven = TRUE;
            break;
        case  BSIM4v6_MOD_PK1:
            mod->BSIM4v6pk1 = value->rValue;
            mod->BSIM4v6pk1Given = TRUE;
            break;
        case  BSIM4v6_MOD_PKT1:
            mod->BSIM4v6pkt1 = value->rValue;
            mod->BSIM4v6pkt1Given = TRUE;
            break;
        case  BSIM4v6_MOD_PKT1L:
            mod->BSIM4v6pkt1l = value->rValue;
            mod->BSIM4v6pkt1lGiven = TRUE;
            break;
        case  BSIM4v6_MOD_PKT2:
            mod->BSIM4v6pkt2 = value->rValue;
            mod->BSIM4v6pkt2Given = TRUE;
            break;
        case  BSIM4v6_MOD_PK2:
            mod->BSIM4v6pk2 = value->rValue;
            mod->BSIM4v6pk2Given = TRUE;
            break;
        case  BSIM4v6_MOD_PK3:
            mod->BSIM4v6pk3 = value->rValue;
            mod->BSIM4v6pk3Given = TRUE;
            break;
        case  BSIM4v6_MOD_PK3B:
            mod->BSIM4v6pk3b = value->rValue;
            mod->BSIM4v6pk3bGiven = TRUE;
            break;
        case  BSIM4v6_MOD_PLPE0:
            mod->BSIM4v6plpe0 = value->rValue;
            mod->BSIM4v6plpe0Given = TRUE;
            break;
        case  BSIM4v6_MOD_PLPEB:
            mod->BSIM4v6plpeb = value->rValue;
            mod->BSIM4v6plpebGiven = TRUE;
            break;
        case  BSIM4v6_MOD_PDVTP0:
            mod->BSIM4v6pdvtp0 = value->rValue;
            mod->BSIM4v6pdvtp0Given = TRUE;
            break;
        case  BSIM4v6_MOD_PDVTP1:
            mod->BSIM4v6pdvtp1 = value->rValue;
            mod->BSIM4v6pdvtp1Given = TRUE;
            break;
        case  BSIM4v6_MOD_PW0:
            mod->BSIM4v6pw0 = value->rValue;
            mod->BSIM4v6pw0Given = TRUE;
            break;
        case  BSIM4v6_MOD_PDVT0:               
            mod->BSIM4v6pdvt0 = value->rValue;
            mod->BSIM4v6pdvt0Given = TRUE;
            break;
        case  BSIM4v6_MOD_PDVT1:             
            mod->BSIM4v6pdvt1 = value->rValue;
            mod->BSIM4v6pdvt1Given = TRUE;
            break;
        case  BSIM4v6_MOD_PDVT2:             
            mod->BSIM4v6pdvt2 = value->rValue;
            mod->BSIM4v6pdvt2Given = TRUE;
            break;
        case  BSIM4v6_MOD_PDVT0W:               
            mod->BSIM4v6pdvt0w = value->rValue;
            mod->BSIM4v6pdvt0wGiven = TRUE;
            break;
        case  BSIM4v6_MOD_PDVT1W:             
            mod->BSIM4v6pdvt1w = value->rValue;
            mod->BSIM4v6pdvt1wGiven = TRUE;
            break;
        case  BSIM4v6_MOD_PDVT2W:             
            mod->BSIM4v6pdvt2w = value->rValue;
            mod->BSIM4v6pdvt2wGiven = TRUE;
            break;
        case  BSIM4v6_MOD_PDROUT:             
            mod->BSIM4v6pdrout = value->rValue;
            mod->BSIM4v6pdroutGiven = TRUE;
            break;
        case  BSIM4v6_MOD_PDSUB:             
            mod->BSIM4v6pdsub = value->rValue;
            mod->BSIM4v6pdsubGiven = TRUE;
            break;
        case BSIM4v6_MOD_PVTH0:
            mod->BSIM4v6pvth0 = value->rValue;
            mod->BSIM4v6pvth0Given = TRUE;
            break;
        case BSIM4v6_MOD_PUA:
            mod->BSIM4v6pua = value->rValue;
            mod->BSIM4v6puaGiven = TRUE;
            break;
        case BSIM4v6_MOD_PUA1:
            mod->BSIM4v6pua1 = value->rValue;
            mod->BSIM4v6pua1Given = TRUE;
            break;
        case BSIM4v6_MOD_PUB:
            mod->BSIM4v6pub = value->rValue;
            mod->BSIM4v6pubGiven = TRUE;
            break;
        case BSIM4v6_MOD_PUB1:
            mod->BSIM4v6pub1 = value->rValue;
            mod->BSIM4v6pub1Given = TRUE;
            break;
        case BSIM4v6_MOD_PUC:
            mod->BSIM4v6puc = value->rValue;
            mod->BSIM4v6pucGiven = TRUE;
            break;
        case BSIM4v6_MOD_PUC1:
            mod->BSIM4v6puc1 = value->rValue;
            mod->BSIM4v6puc1Given = TRUE;
            break;
        case  BSIM4v6_MOD_PU0 :
            mod->BSIM4v6pu0 = value->rValue;
            mod->BSIM4v6pu0Given = TRUE;
            break;
        case  BSIM4v6_MOD_PUTE :
            mod->BSIM4v6pute = value->rValue;
            mod->BSIM4v6puteGiven = TRUE;
            break;
		 case  BSIM4v6_MOD_PUCSTE :
            mod->BSIM4v6pucste = value->rValue;
            mod->BSIM4v6pucsteGiven = TRUE;
            break;
        case BSIM4v6_MOD_PVOFF:
            mod->BSIM4v6pvoff = value->rValue;
            mod->BSIM4v6pvoffGiven = TRUE;
            break;
        case BSIM4v6_MOD_PTVOFF:
            mod->BSIM4v6ptvoff = value->rValue;
            mod->BSIM4v6ptvoffGiven = TRUE;
            break;
        case BSIM4v6_MOD_PMINV:
            mod->BSIM4v6pminv = value->rValue;
            mod->BSIM4v6pminvGiven = TRUE;
            break;
        case BSIM4v6_MOD_PMINVCV:
            mod->BSIM4v6pminvcv = value->rValue;
            mod->BSIM4v6pminvcvGiven = TRUE;
            break;
        case BSIM4v6_MOD_PFPROUT:
            mod->BSIM4v6pfprout = value->rValue;
            mod->BSIM4v6pfproutGiven = TRUE;
            break;
        case BSIM4v6_MOD_PPDITS:
            mod->BSIM4v6ppdits = value->rValue;
            mod->BSIM4v6ppditsGiven = TRUE;
            break;
        case BSIM4v6_MOD_PPDITSD:
            mod->BSIM4v6ppditsd = value->rValue;
            mod->BSIM4v6ppditsdGiven = TRUE;
            break;
        case  BSIM4v6_MOD_PDELTA :
            mod->BSIM4v6pdelta = value->rValue;
            mod->BSIM4v6pdeltaGiven = TRUE;
            break;
        case BSIM4v6_MOD_PRDSW:
            mod->BSIM4v6prdsw = value->rValue;
            mod->BSIM4v6prdswGiven = TRUE;
            break;                     
        case BSIM4v6_MOD_PRDW:
            mod->BSIM4v6prdw = value->rValue;
            mod->BSIM4v6prdwGiven = TRUE;
            break;
        case BSIM4v6_MOD_PRSW:
            mod->BSIM4v6prsw = value->rValue;
            mod->BSIM4v6prswGiven = TRUE;
            break;
        case BSIM4v6_MOD_PPRWB:
            mod->BSIM4v6pprwb = value->rValue;
            mod->BSIM4v6pprwbGiven = TRUE;
            break;                     
        case BSIM4v6_MOD_PPRWG:
            mod->BSIM4v6pprwg = value->rValue;
            mod->BSIM4v6pprwgGiven = TRUE;
            break;                     
        case BSIM4v6_MOD_PPRT:
            mod->BSIM4v6pprt = value->rValue;
            mod->BSIM4v6pprtGiven = TRUE;
            break;                     
        case BSIM4v6_MOD_PETA0:
            mod->BSIM4v6peta0 = value->rValue;
            mod->BSIM4v6peta0Given = TRUE;
            break;                 
        case BSIM4v6_MOD_PETAB:
            mod->BSIM4v6petab = value->rValue;
            mod->BSIM4v6petabGiven = TRUE;
            break;                 
        case BSIM4v6_MOD_PPCLM:
            mod->BSIM4v6ppclm = value->rValue;
            mod->BSIM4v6ppclmGiven = TRUE;
            break;                 
        case BSIM4v6_MOD_PPDIBL1:
            mod->BSIM4v6ppdibl1 = value->rValue;
            mod->BSIM4v6ppdibl1Given = TRUE;
            break;                 
        case BSIM4v6_MOD_PPDIBL2:
            mod->BSIM4v6ppdibl2 = value->rValue;
            mod->BSIM4v6ppdibl2Given = TRUE;
            break;                 
        case BSIM4v6_MOD_PPDIBLB:
            mod->BSIM4v6ppdiblb = value->rValue;
            mod->BSIM4v6ppdiblbGiven = TRUE;
            break;                 
        case BSIM4v6_MOD_PPSCBE1:
            mod->BSIM4v6ppscbe1 = value->rValue;
            mod->BSIM4v6ppscbe1Given = TRUE;
            break;                 
        case BSIM4v6_MOD_PPSCBE2:
            mod->BSIM4v6ppscbe2 = value->rValue;
            mod->BSIM4v6ppscbe2Given = TRUE;
            break;                 
        case BSIM4v6_MOD_PPVAG:
            mod->BSIM4v6ppvag = value->rValue;
            mod->BSIM4v6ppvagGiven = TRUE;
            break;                 
        case  BSIM4v6_MOD_PWR :
            mod->BSIM4v6pwr = value->rValue;
            mod->BSIM4v6pwrGiven = TRUE;
            break;
        case  BSIM4v6_MOD_PDWG :
            mod->BSIM4v6pdwg = value->rValue;
            mod->BSIM4v6pdwgGiven = TRUE;
            break;
        case  BSIM4v6_MOD_PDWB :
            mod->BSIM4v6pdwb = value->rValue;
            mod->BSIM4v6pdwbGiven = TRUE;
            break;
        case  BSIM4v6_MOD_PB0 :
            mod->BSIM4v6pb0 = value->rValue;
            mod->BSIM4v6pb0Given = TRUE;
            break;
        case  BSIM4v6_MOD_PB1 :
            mod->BSIM4v6pb1 = value->rValue;
            mod->BSIM4v6pb1Given = TRUE;
            break;
        case  BSIM4v6_MOD_PALPHA0 :
            mod->BSIM4v6palpha0 = value->rValue;
            mod->BSIM4v6palpha0Given = TRUE;
            break;
        case  BSIM4v6_MOD_PALPHA1 :
            mod->BSIM4v6palpha1 = value->rValue;
            mod->BSIM4v6palpha1Given = TRUE;
            break;
        case  BSIM4v6_MOD_PBETA0 :
            mod->BSIM4v6pbeta0 = value->rValue;
            mod->BSIM4v6pbeta0Given = TRUE;
            break;
        case  BSIM4v6_MOD_PPHIN :
            mod->BSIM4v6pphin = value->rValue;
            mod->BSIM4v6pphinGiven = TRUE;
            break;
        case  BSIM4v6_MOD_PAGIDL :
            mod->BSIM4v6pagidl = value->rValue;
            mod->BSIM4v6pagidlGiven = TRUE;
            break;
        case  BSIM4v6_MOD_PBGIDL :
            mod->BSIM4v6pbgidl = value->rValue;
            mod->BSIM4v6pbgidlGiven = TRUE;
            break;
        case  BSIM4v6_MOD_PCGIDL :
            mod->BSIM4v6pcgidl = value->rValue;
            mod->BSIM4v6pcgidlGiven = TRUE;
            break;
        case  BSIM4v6_MOD_PEGIDL :
            mod->BSIM4v6pegidl = value->rValue;
            mod->BSIM4v6pegidlGiven = TRUE;
            break;
        case  BSIM4v6_MOD_PAGISL :
            mod->BSIM4v6pagisl = value->rValue;
            mod->BSIM4v6pagislGiven = TRUE;
            break;
        case  BSIM4v6_MOD_PBGISL :
            mod->BSIM4v6pbgisl = value->rValue;
            mod->BSIM4v6pbgislGiven = TRUE;
            break;
        case  BSIM4v6_MOD_PCGISL :
            mod->BSIM4v6pcgisl = value->rValue;
            mod->BSIM4v6pcgislGiven = TRUE;
            break;
        case  BSIM4v6_MOD_PEGISL :
            mod->BSIM4v6pegisl = value->rValue;
            mod->BSIM4v6pegislGiven = TRUE;
            break;
        case  BSIM4v6_MOD_PAIGC :
            mod->BSIM4v6paigc = value->rValue;
            mod->BSIM4v6paigcGiven = TRUE;
            break;
        case  BSIM4v6_MOD_PBIGC :
            mod->BSIM4v6pbigc = value->rValue;
            mod->BSIM4v6pbigcGiven = TRUE;
            break;
        case  BSIM4v6_MOD_PCIGC :
            mod->BSIM4v6pcigc = value->rValue;
            mod->BSIM4v6pcigcGiven = TRUE;
            break;
        case  BSIM4v6_MOD_PAIGSD :
            mod->BSIM4v6paigsd = value->rValue;
            mod->BSIM4v6paigsdGiven = TRUE;
            break;
        case  BSIM4v6_MOD_PBIGSD :
            mod->BSIM4v6pbigsd = value->rValue;
            mod->BSIM4v6pbigsdGiven = TRUE;
            break;
        case  BSIM4v6_MOD_PCIGSD :
            mod->BSIM4v6pcigsd = value->rValue;
            mod->BSIM4v6pcigsdGiven = TRUE;
            break;
        case  BSIM4v6_MOD_PAIGS :
            mod->BSIM4v6paigs = value->rValue;
            mod->BSIM4v6paigsGiven = TRUE;
            break;
        case  BSIM4v6_MOD_PBIGS :
            mod->BSIM4v6pbigs = value->rValue;
            mod->BSIM4v6pbigsGiven = TRUE;
            break;
        case  BSIM4v6_MOD_PCIGS :
            mod->BSIM4v6pcigs = value->rValue;
            mod->BSIM4v6pcigsGiven = TRUE;
            break;
        case  BSIM4v6_MOD_PAIGD :
            mod->BSIM4v6paigd = value->rValue;
            mod->BSIM4v6paigdGiven = TRUE;
            break;
        case  BSIM4v6_MOD_PBIGD :
            mod->BSIM4v6pbigd = value->rValue;
            mod->BSIM4v6pbigdGiven = TRUE;
            break;
        case  BSIM4v6_MOD_PCIGD :
            mod->BSIM4v6pcigd = value->rValue;
            mod->BSIM4v6pcigdGiven = TRUE;
            break;
        case  BSIM4v6_MOD_PAIGBACC :
            mod->BSIM4v6paigbacc = value->rValue;
            mod->BSIM4v6paigbaccGiven = TRUE;
            break;
        case  BSIM4v6_MOD_PBIGBACC :
            mod->BSIM4v6pbigbacc = value->rValue;
            mod->BSIM4v6pbigbaccGiven = TRUE;
            break;
        case  BSIM4v6_MOD_PCIGBACC :
            mod->BSIM4v6pcigbacc = value->rValue;
            mod->BSIM4v6pcigbaccGiven = TRUE;
            break;
        case  BSIM4v6_MOD_PAIGBINV :
            mod->BSIM4v6paigbinv = value->rValue;
            mod->BSIM4v6paigbinvGiven = TRUE;
            break;
        case  BSIM4v6_MOD_PBIGBINV :
            mod->BSIM4v6pbigbinv = value->rValue;
            mod->BSIM4v6pbigbinvGiven = TRUE;
            break;
        case  BSIM4v6_MOD_PCIGBINV :
            mod->BSIM4v6pcigbinv = value->rValue;
            mod->BSIM4v6pcigbinvGiven = TRUE;
            break;
        case  BSIM4v6_MOD_PNIGC :
            mod->BSIM4v6pnigc = value->rValue;
            mod->BSIM4v6pnigcGiven = TRUE;
            break;
        case  BSIM4v6_MOD_PNIGBINV :
            mod->BSIM4v6pnigbinv = value->rValue;
            mod->BSIM4v6pnigbinvGiven = TRUE;
            break;
        case  BSIM4v6_MOD_PNIGBACC :
            mod->BSIM4v6pnigbacc = value->rValue;
            mod->BSIM4v6pnigbaccGiven = TRUE;
            break;
        case  BSIM4v6_MOD_PNTOX :
            mod->BSIM4v6pntox = value->rValue;
            mod->BSIM4v6pntoxGiven = TRUE;
            break;
        case  BSIM4v6_MOD_PEIGBINV :
            mod->BSIM4v6peigbinv = value->rValue;
            mod->BSIM4v6peigbinvGiven = TRUE;
            break;
        case  BSIM4v6_MOD_PPIGCD :
            mod->BSIM4v6ppigcd = value->rValue;
            mod->BSIM4v6ppigcdGiven = TRUE;
            break;
        case  BSIM4v6_MOD_PPOXEDGE :
            mod->BSIM4v6ppoxedge = value->rValue;
            mod->BSIM4v6ppoxedgeGiven = TRUE;
            break;
        case  BSIM4v6_MOD_PXRCRG1 :
            mod->BSIM4v6pxrcrg1 = value->rValue;
            mod->BSIM4v6pxrcrg1Given = TRUE;
            break;
        case  BSIM4v6_MOD_PXRCRG2 :
            mod->BSIM4v6pxrcrg2 = value->rValue;
            mod->BSIM4v6pxrcrg2Given = TRUE;
            break;
        case  BSIM4v6_MOD_PLAMBDA :
            mod->BSIM4v6plambda = value->rValue;
            mod->BSIM4v6plambdaGiven = TRUE;
            break;
        case  BSIM4v6_MOD_PVTL :
            mod->BSIM4v6pvtl = value->rValue;
            mod->BSIM4v6pvtlGiven = TRUE;
            break;
        case  BSIM4v6_MOD_PXN:
            mod->BSIM4v6pxn = value->rValue;
            mod->BSIM4v6pxnGiven = TRUE;
            break;
        case  BSIM4v6_MOD_PVFBSDOFF:
            mod->BSIM4v6pvfbsdoff = value->rValue;
            mod->BSIM4v6pvfbsdoffGiven = TRUE;
            break;
        case  BSIM4v6_MOD_PTVFBSDOFF:
            mod->BSIM4v6ptvfbsdoff = value->rValue;
            mod->BSIM4v6ptvfbsdoffGiven = TRUE;
            break;
        case  BSIM4v6_MOD_PEU :
            mod->BSIM4v6peu = value->rValue;
            mod->BSIM4v6peuGiven = TRUE;
            break;
		case  BSIM4v6_MOD_PUCS :
            mod->BSIM4v6pucs = value->rValue;
            mod->BSIM4v6pucsGiven = TRUE;
            break;
        case  BSIM4v6_MOD_PVFB :
            mod->BSIM4v6pvfb = value->rValue;
            mod->BSIM4v6pvfbGiven = TRUE;
            break;
        case  BSIM4v6_MOD_PCGSL :
            mod->BSIM4v6pcgsl = value->rValue;
            mod->BSIM4v6pcgslGiven = TRUE;
            break;
        case  BSIM4v6_MOD_PCGDL :
            mod->BSIM4v6pcgdl = value->rValue;
            mod->BSIM4v6pcgdlGiven = TRUE;
            break;
        case  BSIM4v6_MOD_PCKAPPAS :
            mod->BSIM4v6pckappas = value->rValue;
            mod->BSIM4v6pckappasGiven = TRUE;
            break;
        case  BSIM4v6_MOD_PCKAPPAD :
            mod->BSIM4v6pckappad = value->rValue;
            mod->BSIM4v6pckappadGiven = TRUE;
            break;
        case  BSIM4v6_MOD_PCF :
            mod->BSIM4v6pcf = value->rValue;
            mod->BSIM4v6pcfGiven = TRUE;
            break;
        case  BSIM4v6_MOD_PCLC :
            mod->BSIM4v6pclc = value->rValue;
            mod->BSIM4v6pclcGiven = TRUE;
            break;
        case  BSIM4v6_MOD_PCLE :
            mod->BSIM4v6pcle = value->rValue;
            mod->BSIM4v6pcleGiven = TRUE;
            break;
        case  BSIM4v6_MOD_PVFBCV :
            mod->BSIM4v6pvfbcv = value->rValue;
            mod->BSIM4v6pvfbcvGiven = TRUE;
            break;
        case  BSIM4v6_MOD_PACDE :
            mod->BSIM4v6pacde = value->rValue;
            mod->BSIM4v6pacdeGiven = TRUE;
            break;
        case  BSIM4v6_MOD_PMOIN :
            mod->BSIM4v6pmoin = value->rValue;
            mod->BSIM4v6pmoinGiven = TRUE;
            break;
        case  BSIM4v6_MOD_PNOFF :
            mod->BSIM4v6pnoff = value->rValue;
            mod->BSIM4v6pnoffGiven = TRUE;
            break;
        case  BSIM4v6_MOD_PVOFFCV :
            mod->BSIM4v6pvoffcv = value->rValue;
            mod->BSIM4v6pvoffcvGiven = TRUE;
            break;

        case  BSIM4v6_MOD_TNOM :
            mod->BSIM4v6tnom = value->rValue + CONSTCtoK;
            mod->BSIM4v6tnomGiven = TRUE;
            break;
        case  BSIM4v6_MOD_CGSO :
            mod->BSIM4v6cgso = value->rValue;
            mod->BSIM4v6cgsoGiven = TRUE;
            break;
        case  BSIM4v6_MOD_CGDO :
            mod->BSIM4v6cgdo = value->rValue;
            mod->BSIM4v6cgdoGiven = TRUE;
            break;
        case  BSIM4v6_MOD_CGBO :
            mod->BSIM4v6cgbo = value->rValue;
            mod->BSIM4v6cgboGiven = TRUE;
            break;
        case  BSIM4v6_MOD_XPART :
            mod->BSIM4v6xpart = value->rValue;
            mod->BSIM4v6xpartGiven = TRUE;
            break;
        case  BSIM4v6_MOD_RSH :
            mod->BSIM4v6sheetResistance = value->rValue;
            mod->BSIM4v6sheetResistanceGiven = TRUE;
            break;
        case  BSIM4v6_MOD_JSS :
            mod->BSIM4v6SjctSatCurDensity = value->rValue;
            mod->BSIM4v6SjctSatCurDensityGiven = TRUE;
            break;
        case  BSIM4v6_MOD_JSWS :
            mod->BSIM4v6SjctSidewallSatCurDensity = value->rValue;
            mod->BSIM4v6SjctSidewallSatCurDensityGiven = TRUE;
            break;
        case  BSIM4v6_MOD_JSWGS :
            mod->BSIM4v6SjctGateSidewallSatCurDensity = value->rValue;
            mod->BSIM4v6SjctGateSidewallSatCurDensityGiven = TRUE;
            break;
        case  BSIM4v6_MOD_PBS :
            mod->BSIM4v6SbulkJctPotential = value->rValue;
            mod->BSIM4v6SbulkJctPotentialGiven = TRUE;
            break;
        case  BSIM4v6_MOD_MJS :
            mod->BSIM4v6SbulkJctBotGradingCoeff = value->rValue;
            mod->BSIM4v6SbulkJctBotGradingCoeffGiven = TRUE;
            break;
        case  BSIM4v6_MOD_PBSWS :
            mod->BSIM4v6SsidewallJctPotential = value->rValue;
            mod->BSIM4v6SsidewallJctPotentialGiven = TRUE;
            break;
        case  BSIM4v6_MOD_MJSWS :
            mod->BSIM4v6SbulkJctSideGradingCoeff = value->rValue;
            mod->BSIM4v6SbulkJctSideGradingCoeffGiven = TRUE;
            break;
        case  BSIM4v6_MOD_CJS :
            mod->BSIM4v6SunitAreaJctCap = value->rValue;
            mod->BSIM4v6SunitAreaJctCapGiven = TRUE;
            break;
        case  BSIM4v6_MOD_CJSWS :
            mod->BSIM4v6SunitLengthSidewallJctCap = value->rValue;
            mod->BSIM4v6SunitLengthSidewallJctCapGiven = TRUE;
            break;
        case  BSIM4v6_MOD_NJS :
            mod->BSIM4v6SjctEmissionCoeff = value->rValue;
            mod->BSIM4v6SjctEmissionCoeffGiven = TRUE;
            break;
        case  BSIM4v6_MOD_PBSWGS :
            mod->BSIM4v6SGatesidewallJctPotential = value->rValue;
            mod->BSIM4v6SGatesidewallJctPotentialGiven = TRUE;
            break;
        case  BSIM4v6_MOD_MJSWGS :
            mod->BSIM4v6SbulkJctGateSideGradingCoeff = value->rValue;
            mod->BSIM4v6SbulkJctGateSideGradingCoeffGiven = TRUE;
            break;
        case  BSIM4v6_MOD_CJSWGS :
            mod->BSIM4v6SunitLengthGateSidewallJctCap = value->rValue;
            mod->BSIM4v6SunitLengthGateSidewallJctCapGiven = TRUE;
            break;
        case  BSIM4v6_MOD_XTIS :
            mod->BSIM4v6SjctTempExponent = value->rValue;
            mod->BSIM4v6SjctTempExponentGiven = TRUE;
            break;
        case  BSIM4v6_MOD_JSD :
            mod->BSIM4v6DjctSatCurDensity = value->rValue;
            mod->BSIM4v6DjctSatCurDensityGiven = TRUE;
            break;
        case  BSIM4v6_MOD_JSWD :
            mod->BSIM4v6DjctSidewallSatCurDensity = value->rValue;
            mod->BSIM4v6DjctSidewallSatCurDensityGiven = TRUE;
            break;
        case  BSIM4v6_MOD_JSWGD :
            mod->BSIM4v6DjctGateSidewallSatCurDensity = value->rValue;
            mod->BSIM4v6DjctGateSidewallSatCurDensityGiven = TRUE;
            break;
        case  BSIM4v6_MOD_PBD :
            mod->BSIM4v6DbulkJctPotential = value->rValue;
            mod->BSIM4v6DbulkJctPotentialGiven = TRUE;
            break;
        case  BSIM4v6_MOD_MJD :
            mod->BSIM4v6DbulkJctBotGradingCoeff = value->rValue;
            mod->BSIM4v6DbulkJctBotGradingCoeffGiven = TRUE;
            break;
        case  BSIM4v6_MOD_PBSWD :
            mod->BSIM4v6DsidewallJctPotential = value->rValue;
            mod->BSIM4v6DsidewallJctPotentialGiven = TRUE;
            break;
        case  BSIM4v6_MOD_MJSWD :
            mod->BSIM4v6DbulkJctSideGradingCoeff = value->rValue;
            mod->BSIM4v6DbulkJctSideGradingCoeffGiven = TRUE;
            break;
        case  BSIM4v6_MOD_CJD :
            mod->BSIM4v6DunitAreaJctCap = value->rValue;
            mod->BSIM4v6DunitAreaJctCapGiven = TRUE;
            break;
        case  BSIM4v6_MOD_CJSWD :
            mod->BSIM4v6DunitLengthSidewallJctCap = value->rValue;
            mod->BSIM4v6DunitLengthSidewallJctCapGiven = TRUE;
            break;
        case  BSIM4v6_MOD_NJD :
            mod->BSIM4v6DjctEmissionCoeff = value->rValue;
            mod->BSIM4v6DjctEmissionCoeffGiven = TRUE;
            break;
        case  BSIM4v6_MOD_PBSWGD :
            mod->BSIM4v6DGatesidewallJctPotential = value->rValue;
            mod->BSIM4v6DGatesidewallJctPotentialGiven = TRUE;
            break;
        case  BSIM4v6_MOD_MJSWGD :
            mod->BSIM4v6DbulkJctGateSideGradingCoeff = value->rValue;
            mod->BSIM4v6DbulkJctGateSideGradingCoeffGiven = TRUE;
            break;
        case  BSIM4v6_MOD_CJSWGD :
            mod->BSIM4v6DunitLengthGateSidewallJctCap = value->rValue;
            mod->BSIM4v6DunitLengthGateSidewallJctCapGiven = TRUE;
            break;
        case  BSIM4v6_MOD_XTID :
            mod->BSIM4v6DjctTempExponent = value->rValue;
            mod->BSIM4v6DjctTempExponentGiven = TRUE;
            break;
        case  BSIM4v6_MOD_LINT :
            mod->BSIM4v6Lint = value->rValue;
            mod->BSIM4v6LintGiven = TRUE;
            break;
        case  BSIM4v6_MOD_LL :
            mod->BSIM4v6Ll = value->rValue;
            mod->BSIM4v6LlGiven = TRUE;
            break;
        case  BSIM4v6_MOD_LLC :
            mod->BSIM4v6Llc = value->rValue;
            mod->BSIM4v6LlcGiven = TRUE;
            break;
        case  BSIM4v6_MOD_LLN :
            mod->BSIM4v6Lln = value->rValue;
            mod->BSIM4v6LlnGiven = TRUE;
            break;
        case  BSIM4v6_MOD_LW :
            mod->BSIM4v6Lw = value->rValue;
            mod->BSIM4v6LwGiven = TRUE;
            break;
        case  BSIM4v6_MOD_LWC :
            mod->BSIM4v6Lwc = value->rValue;
            mod->BSIM4v6LwcGiven = TRUE;
            break;
        case  BSIM4v6_MOD_LWN :
            mod->BSIM4v6Lwn = value->rValue;
            mod->BSIM4v6LwnGiven = TRUE;
            break;
        case  BSIM4v6_MOD_LWL :
            mod->BSIM4v6Lwl = value->rValue;
            mod->BSIM4v6LwlGiven = TRUE;
            break;
        case  BSIM4v6_MOD_LWLC :
            mod->BSIM4v6Lwlc = value->rValue;
            mod->BSIM4v6LwlcGiven = TRUE;
            break;
        case  BSIM4v6_MOD_LMIN :
            mod->BSIM4v6Lmin = value->rValue;
            mod->BSIM4v6LminGiven = TRUE;
            break;
        case  BSIM4v6_MOD_LMAX :
            mod->BSIM4v6Lmax = value->rValue;
            mod->BSIM4v6LmaxGiven = TRUE;
            break;
        case  BSIM4v6_MOD_WINT :
            mod->BSIM4v6Wint = value->rValue;
            mod->BSIM4v6WintGiven = TRUE;
            break;
        case  BSIM4v6_MOD_WL :
            mod->BSIM4v6Wl = value->rValue;
            mod->BSIM4v6WlGiven = TRUE;
            break;
        case  BSIM4v6_MOD_WLC :
            mod->BSIM4v6Wlc = value->rValue;
            mod->BSIM4v6WlcGiven = TRUE;
            break;
        case  BSIM4v6_MOD_WLN :
            mod->BSIM4v6Wln = value->rValue;
            mod->BSIM4v6WlnGiven = TRUE;
            break;
        case  BSIM4v6_MOD_WW :
            mod->BSIM4v6Ww = value->rValue;
            mod->BSIM4v6WwGiven = TRUE;
            break;
        case  BSIM4v6_MOD_WWC :
            mod->BSIM4v6Wwc = value->rValue;
            mod->BSIM4v6WwcGiven = TRUE;
            break;
        case  BSIM4v6_MOD_WWN :
            mod->BSIM4v6Wwn = value->rValue;
            mod->BSIM4v6WwnGiven = TRUE;
            break;
        case  BSIM4v6_MOD_WWL :
            mod->BSIM4v6Wwl = value->rValue;
            mod->BSIM4v6WwlGiven = TRUE;
            break;
        case  BSIM4v6_MOD_WWLC :
            mod->BSIM4v6Wwlc = value->rValue;
            mod->BSIM4v6WwlcGiven = TRUE;
            break;
        case  BSIM4v6_MOD_WMIN :
            mod->BSIM4v6Wmin = value->rValue;
            mod->BSIM4v6WminGiven = TRUE;
            break;
        case  BSIM4v6_MOD_WMAX :
            mod->BSIM4v6Wmax = value->rValue;
            mod->BSIM4v6WmaxGiven = TRUE;
            break;

        case  BSIM4v6_MOD_NOIA :
            mod->BSIM4v6oxideTrapDensityA = value->rValue;
            mod->BSIM4v6oxideTrapDensityAGiven = TRUE;
            break;
        case  BSIM4v6_MOD_NOIB :
            mod->BSIM4v6oxideTrapDensityB = value->rValue;
            mod->BSIM4v6oxideTrapDensityBGiven = TRUE;
            break;
        case  BSIM4v6_MOD_NOIC :
            mod->BSIM4v6oxideTrapDensityC = value->rValue;
            mod->BSIM4v6oxideTrapDensityCGiven = TRUE;
            break;
        case  BSIM4v6_MOD_EM :
            mod->BSIM4v6em = value->rValue;
            mod->BSIM4v6emGiven = TRUE;
            break;
        case  BSIM4v6_MOD_EF :
            mod->BSIM4v6ef = value->rValue;
            mod->BSIM4v6efGiven = TRUE;
            break;
        case  BSIM4v6_MOD_AF :
            mod->BSIM4v6af = value->rValue;
            mod->BSIM4v6afGiven = TRUE;
            break;
        case  BSIM4v6_MOD_KF :
            mod->BSIM4v6kf = value->rValue;
            mod->BSIM4v6kfGiven = TRUE;
            break;
        case BSIM4v6_MOD_VGS_MAX:
            mod->BSIM4v6vgsMax = value->rValue;
            mod->BSIM4v6vgsMaxGiven = TRUE;
            break;
        case BSIM4v6_MOD_VGD_MAX:
            mod->BSIM4v6vgdMax = value->rValue;
            mod->BSIM4v6vgdMaxGiven = TRUE;
            break;
        case BSIM4v6_MOD_VGB_MAX:
            mod->BSIM4v6vgbMax = value->rValue;
            mod->BSIM4v6vgbMaxGiven = TRUE;
            break;
        case BSIM4v6_MOD_VDS_MAX:
            mod->BSIM4v6vdsMax = value->rValue;
            mod->BSIM4v6vdsMaxGiven = TRUE;
            break;
        case BSIM4v6_MOD_VBS_MAX:
            mod->BSIM4v6vbsMax = value->rValue;
            mod->BSIM4v6vbsMaxGiven = TRUE;
            break;
        case BSIM4v6_MOD_VBD_MAX:
            mod->BSIM4v6vbdMax = value->rValue;
            mod->BSIM4v6vbdMaxGiven = TRUE;
            break;
        case BSIM4v6_MOD_VGSR_MAX:
            mod->BSIM4v6vgsrMax = value->rValue;
            mod->BSIM4v6vgsrMaxGiven = TRUE;
            break;
        case BSIM4v6_MOD_VGDR_MAX:
            mod->BSIM4v6vgdrMax = value->rValue;
            mod->BSIM4v6vgdrMaxGiven = TRUE;
            break;
        case BSIM4v6_MOD_VGBR_MAX:
            mod->BSIM4v6vgbrMax = value->rValue;
            mod->BSIM4v6vgbrMaxGiven = TRUE;
            break;
        case BSIM4v6_MOD_VBSR_MAX:
            mod->BSIM4v6vbsrMax = value->rValue;
            mod->BSIM4v6vbsrMaxGiven = TRUE;
            break;
        case BSIM4v6_MOD_VBDR_MAX:
            mod->BSIM4v6vbdrMax = value->rValue;
            mod->BSIM4v6vbdrMaxGiven = TRUE;
            break;

        case  BSIM4v6_MOD_NMOS  :
            if(value->iValue) {
                mod->BSIM4v6type = 1;
                mod->BSIM4v6typeGiven = TRUE;
            }
            break;
        case  BSIM4v6_MOD_PMOS  :
            if(value->iValue) {
                mod->BSIM4v6type = - 1;
                mod->BSIM4v6typeGiven = TRUE;
            }
            break;
        default:
            return(E_BADPARM);
    }
    return(OK);
}


