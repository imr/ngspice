/* $Id$  */
/*
 $Log$
 Revision 1.1.1.1  2000-04-27 20:03:59  pnenzi
 Imported sources

 * Revision 3.2 1998/6/16  18:00:00  Weidong 
 * BSIM3v3.2 release
 *
*/
static char rcsid[] = "$Id$";

/*************************************/

/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1995 Min-Chie Jeng and Mansun Chan.
Modified by Weidong Liu (1997-1998).
File: b3v2mpar.c
**********/

#include "ngspice.h"
#include <stdio.h>
#include "bsim3v2def.h"
#include "ifsim.h"
#include "sperror.h"
#include "suffix.h"


int
BSIM3V2mParam(param,value,inMod)
int param;
IFvalue *value;
GENmodel *inMod;
{
    BSIM3V2model *mod = (BSIM3V2model*)inMod;
    switch(param)
    {   case  BSIM3V2_MOD_MOBMOD :
            mod->BSIM3V2mobMod = value->iValue;
            mod->BSIM3V2mobModGiven = TRUE;
            break;
        case  BSIM3V2_MOD_BINUNIT :
            mod->BSIM3V2binUnit = value->iValue;
            mod->BSIM3V2binUnitGiven = TRUE;
            break;
        case  BSIM3V2_MOD_PARAMCHK :
            mod->BSIM3V2paramChk = value->iValue;
            mod->BSIM3V2paramChkGiven = TRUE;
            break;
        case  BSIM3V2_MOD_CAPMOD :
            mod->BSIM3V2capMod = value->iValue;
            mod->BSIM3V2capModGiven = TRUE;
            break;
        case  BSIM3V2_MOD_NOIMOD :
            mod->BSIM3V2noiMod = value->iValue;
            mod->BSIM3V2noiModGiven = TRUE;
            break;
        case  BSIM3V2_MOD_VERSION :
            mod->BSIM3V2version = value->rValue;
            mod->BSIM3V2versionGiven = TRUE;
            break;
        case  BSIM3V2_MOD_TOX :
            mod->BSIM3V2tox = value->rValue;
            mod->BSIM3V2toxGiven = TRUE;
            break;
        case  BSIM3V2_MOD_TOXM :
            mod->BSIM3V2toxm = value->rValue;
            mod->BSIM3V2toxmGiven = TRUE;
            break;

        case  BSIM3V2_MOD_CDSC :
            mod->BSIM3V2cdsc = value->rValue;
            mod->BSIM3V2cdscGiven = TRUE;
            break;
        case  BSIM3V2_MOD_CDSCB :
            mod->BSIM3V2cdscb = value->rValue;
            mod->BSIM3V2cdscbGiven = TRUE;
            break;

        case  BSIM3V2_MOD_CDSCD :
            mod->BSIM3V2cdscd = value->rValue;
            mod->BSIM3V2cdscdGiven = TRUE;
            break;

        case  BSIM3V2_MOD_CIT :
            mod->BSIM3V2cit = value->rValue;
            mod->BSIM3V2citGiven = TRUE;
            break;
        case  BSIM3V2_MOD_NFACTOR :
            mod->BSIM3V2nfactor = value->rValue;
            mod->BSIM3V2nfactorGiven = TRUE;
            break;
        case BSIM3V2_MOD_XJ:
            mod->BSIM3V2xj = value->rValue;
            mod->BSIM3V2xjGiven = TRUE;
            break;
        case BSIM3V2_MOD_VSAT:
            mod->BSIM3V2vsat = value->rValue;
            mod->BSIM3V2vsatGiven = TRUE;
            break;
        case BSIM3V2_MOD_A0:
            mod->BSIM3V2a0 = value->rValue;
            mod->BSIM3V2a0Given = TRUE;
            break;
        
        case BSIM3V2_MOD_AGS:
            mod->BSIM3V2ags= value->rValue;
            mod->BSIM3V2agsGiven = TRUE;
            break;
        
        case BSIM3V2_MOD_A1:
            mod->BSIM3V2a1 = value->rValue;
            mod->BSIM3V2a1Given = TRUE;
            break;
        case BSIM3V2_MOD_A2:
            mod->BSIM3V2a2 = value->rValue;
            mod->BSIM3V2a2Given = TRUE;
            break;
        case BSIM3V2_MOD_AT:
            mod->BSIM3V2at = value->rValue;
            mod->BSIM3V2atGiven = TRUE;
            break;
        case BSIM3V2_MOD_KETA:
            mod->BSIM3V2keta = value->rValue;
            mod->BSIM3V2ketaGiven = TRUE;
            break;    
        case BSIM3V2_MOD_NSUB:
            mod->BSIM3V2nsub = value->rValue;
            mod->BSIM3V2nsubGiven = TRUE;
            break;
        case BSIM3V2_MOD_NPEAK:
            mod->BSIM3V2npeak = value->rValue;
            mod->BSIM3V2npeakGiven = TRUE;
	    if (mod->BSIM3V2npeak > 1.0e20)
		mod->BSIM3V2npeak *= 1.0e-6;
            break;
        case BSIM3V2_MOD_NGATE:
            mod->BSIM3V2ngate = value->rValue;
            mod->BSIM3V2ngateGiven = TRUE;
	    if (mod->BSIM3V2ngate > 1.0e23)
		mod->BSIM3V2ngate *= 1.0e-6;
            break;
        case BSIM3V2_MOD_GAMMA1:
            mod->BSIM3V2gamma1 = value->rValue;
            mod->BSIM3V2gamma1Given = TRUE;
            break;
        case BSIM3V2_MOD_GAMMA2:
            mod->BSIM3V2gamma2 = value->rValue;
            mod->BSIM3V2gamma2Given = TRUE;
            break;
        case BSIM3V2_MOD_VBX:
            mod->BSIM3V2vbx = value->rValue;
            mod->BSIM3V2vbxGiven = TRUE;
            break;
        case BSIM3V2_MOD_VBM:
            mod->BSIM3V2vbm = value->rValue;
            mod->BSIM3V2vbmGiven = TRUE;
            break;
        case BSIM3V2_MOD_XT:
            mod->BSIM3V2xt = value->rValue;
            mod->BSIM3V2xtGiven = TRUE;
            break;
        case  BSIM3V2_MOD_K1:
            mod->BSIM3V2k1 = value->rValue;
            mod->BSIM3V2k1Given = TRUE;
            break;
        case  BSIM3V2_MOD_KT1:
            mod->BSIM3V2kt1 = value->rValue;
            mod->BSIM3V2kt1Given = TRUE;
            break;
        case  BSIM3V2_MOD_KT1L:
            mod->BSIM3V2kt1l = value->rValue;
            mod->BSIM3V2kt1lGiven = TRUE;
            break;
        case  BSIM3V2_MOD_KT2:
            mod->BSIM3V2kt2 = value->rValue;
            mod->BSIM3V2kt2Given = TRUE;
            break;
        case  BSIM3V2_MOD_K2:
            mod->BSIM3V2k2 = value->rValue;
            mod->BSIM3V2k2Given = TRUE;
            break;
        case  BSIM3V2_MOD_K3:
            mod->BSIM3V2k3 = value->rValue;
            mod->BSIM3V2k3Given = TRUE;
            break;
        case  BSIM3V2_MOD_K3B:
            mod->BSIM3V2k3b = value->rValue;
            mod->BSIM3V2k3bGiven = TRUE;
            break;
        case  BSIM3V2_MOD_NLX:
            mod->BSIM3V2nlx = value->rValue;
            mod->BSIM3V2nlxGiven = TRUE;
            break;
        case  BSIM3V2_MOD_W0:
            mod->BSIM3V2w0 = value->rValue;
            mod->BSIM3V2w0Given = TRUE;
            break;
        case  BSIM3V2_MOD_DVT0:               
            mod->BSIM3V2dvt0 = value->rValue;
            mod->BSIM3V2dvt0Given = TRUE;
            break;
        case  BSIM3V2_MOD_DVT1:             
            mod->BSIM3V2dvt1 = value->rValue;
            mod->BSIM3V2dvt1Given = TRUE;
            break;
        case  BSIM3V2_MOD_DVT2:             
            mod->BSIM3V2dvt2 = value->rValue;
            mod->BSIM3V2dvt2Given = TRUE;
            break;
        case  BSIM3V2_MOD_DVT0W:               
            mod->BSIM3V2dvt0w = value->rValue;
            mod->BSIM3V2dvt0wGiven = TRUE;
            break;
        case  BSIM3V2_MOD_DVT1W:             
            mod->BSIM3V2dvt1w = value->rValue;
            mod->BSIM3V2dvt1wGiven = TRUE;
            break;
        case  BSIM3V2_MOD_DVT2W:             
            mod->BSIM3V2dvt2w = value->rValue;
            mod->BSIM3V2dvt2wGiven = TRUE;
            break;
        case  BSIM3V2_MOD_DROUT:             
            mod->BSIM3V2drout = value->rValue;
            mod->BSIM3V2droutGiven = TRUE;
            break;
        case  BSIM3V2_MOD_DSUB:             
            mod->BSIM3V2dsub = value->rValue;
            mod->BSIM3V2dsubGiven = TRUE;
            break;
        case BSIM3V2_MOD_VTH0:
            mod->BSIM3V2vth0 = value->rValue;
            mod->BSIM3V2vth0Given = TRUE;
            break;
        case BSIM3V2_MOD_UA:
            mod->BSIM3V2ua = value->rValue;
            mod->BSIM3V2uaGiven = TRUE;
            break;
        case BSIM3V2_MOD_UA1:
            mod->BSIM3V2ua1 = value->rValue;
            mod->BSIM3V2ua1Given = TRUE;
            break;
        case BSIM3V2_MOD_UB:
            mod->BSIM3V2ub = value->rValue;
            mod->BSIM3V2ubGiven = TRUE;
            break;
        case BSIM3V2_MOD_UB1:
            mod->BSIM3V2ub1 = value->rValue;
            mod->BSIM3V2ub1Given = TRUE;
            break;
        case BSIM3V2_MOD_UC:
            mod->BSIM3V2uc = value->rValue;
            mod->BSIM3V2ucGiven = TRUE;
            break;
        case BSIM3V2_MOD_UC1:
            mod->BSIM3V2uc1 = value->rValue;
            mod->BSIM3V2uc1Given = TRUE;
            break;
        case  BSIM3V2_MOD_U0 :
            mod->BSIM3V2u0 = value->rValue;
            mod->BSIM3V2u0Given = TRUE;
            break;
        case  BSIM3V2_MOD_UTE :
            mod->BSIM3V2ute = value->rValue;
            mod->BSIM3V2uteGiven = TRUE;
            break;
        case BSIM3V2_MOD_VOFF:
            mod->BSIM3V2voff = value->rValue;
            mod->BSIM3V2voffGiven = TRUE;
            break;
        case  BSIM3V2_MOD_DELTA :
            mod->BSIM3V2delta = value->rValue;
            mod->BSIM3V2deltaGiven = TRUE;
            break;
        case BSIM3V2_MOD_RDSW:
            mod->BSIM3V2rdsw = value->rValue;
            mod->BSIM3V2rdswGiven = TRUE;
            break;                     
        case BSIM3V2_MOD_PRWG:
            mod->BSIM3V2prwg = value->rValue;
            mod->BSIM3V2prwgGiven = TRUE;
            break;                     
        case BSIM3V2_MOD_PRWB:
            mod->BSIM3V2prwb = value->rValue;
            mod->BSIM3V2prwbGiven = TRUE;
            break;                     
        case BSIM3V2_MOD_PRT:
            mod->BSIM3V2prt = value->rValue;
            mod->BSIM3V2prtGiven = TRUE;
            break;                     
        case BSIM3V2_MOD_ETA0:
            mod->BSIM3V2eta0 = value->rValue;
            mod->BSIM3V2eta0Given = TRUE;
            break;                 
        case BSIM3V2_MOD_ETAB:
            mod->BSIM3V2etab = value->rValue;
            mod->BSIM3V2etabGiven = TRUE;
            break;                 
        case BSIM3V2_MOD_PCLM:
            mod->BSIM3V2pclm = value->rValue;
            mod->BSIM3V2pclmGiven = TRUE;
            break;                 
        case BSIM3V2_MOD_PDIBL1:
            mod->BSIM3V2pdibl1 = value->rValue;
            mod->BSIM3V2pdibl1Given = TRUE;
            break;                 
        case BSIM3V2_MOD_PDIBL2:
            mod->BSIM3V2pdibl2 = value->rValue;
            mod->BSIM3V2pdibl2Given = TRUE;
            break;                 
        case BSIM3V2_MOD_PDIBLB:
            mod->BSIM3V2pdiblb = value->rValue;
            mod->BSIM3V2pdiblbGiven = TRUE;
            break;                 
        case BSIM3V2_MOD_PSCBE1:
            mod->BSIM3V2pscbe1 = value->rValue;
            mod->BSIM3V2pscbe1Given = TRUE;
            break;                 
        case BSIM3V2_MOD_PSCBE2:
            mod->BSIM3V2pscbe2 = value->rValue;
            mod->BSIM3V2pscbe2Given = TRUE;
            break;                 
        case BSIM3V2_MOD_PVAG:
            mod->BSIM3V2pvag = value->rValue;
            mod->BSIM3V2pvagGiven = TRUE;
            break;                 
        case  BSIM3V2_MOD_WR :
            mod->BSIM3V2wr = value->rValue;
            mod->BSIM3V2wrGiven = TRUE;
            break;
        case  BSIM3V2_MOD_DWG :
            mod->BSIM3V2dwg = value->rValue;
            mod->BSIM3V2dwgGiven = TRUE;
            break;
        case  BSIM3V2_MOD_DWB :
            mod->BSIM3V2dwb = value->rValue;
            mod->BSIM3V2dwbGiven = TRUE;
            break;
        case  BSIM3V2_MOD_B0 :
            mod->BSIM3V2b0 = value->rValue;
            mod->BSIM3V2b0Given = TRUE;
            break;
        case  BSIM3V2_MOD_B1 :
            mod->BSIM3V2b1 = value->rValue;
            mod->BSIM3V2b1Given = TRUE;
            break;
        case  BSIM3V2_MOD_ALPHA0 :
            mod->BSIM3V2alpha0 = value->rValue;
            mod->BSIM3V2alpha0Given = TRUE;
            break;
        case  BSIM3V2_MOD_ALPHA1 :
            mod->BSIM3V2alpha1 = value->rValue;
            mod->BSIM3V2alpha1Given = TRUE;
            break;
        case  BSIM3V2_MOD_BETA0 :
            mod->BSIM3V2beta0 = value->rValue;
            mod->BSIM3V2beta0Given = TRUE;
            break;
        case  BSIM3V2_MOD_IJTH :
            mod->BSIM3V2ijth = value->rValue;
            mod->BSIM3V2ijthGiven = TRUE;
            break;
        case  BSIM3V2_MOD_VFB :
            mod->BSIM3V2vfb = value->rValue;
            mod->BSIM3V2vfbGiven = TRUE;
            break;

        case  BSIM3V2_MOD_ELM :
            mod->BSIM3V2elm = value->rValue;
            mod->BSIM3V2elmGiven = TRUE;
            break;
        case  BSIM3V2_MOD_CGSL :
            mod->BSIM3V2cgsl = value->rValue;
            mod->BSIM3V2cgslGiven = TRUE;
            break;
        case  BSIM3V2_MOD_CGDL :
            mod->BSIM3V2cgdl = value->rValue;
            mod->BSIM3V2cgdlGiven = TRUE;
            break;
        case  BSIM3V2_MOD_CKAPPA :
            mod->BSIM3V2ckappa = value->rValue;
            mod->BSIM3V2ckappaGiven = TRUE;
            break;
        case  BSIM3V2_MOD_CF :
            mod->BSIM3V2cf = value->rValue;
            mod->BSIM3V2cfGiven = TRUE;
            break;
        case  BSIM3V2_MOD_CLC :
            mod->BSIM3V2clc = value->rValue;
            mod->BSIM3V2clcGiven = TRUE;
            break;
        case  BSIM3V2_MOD_CLE :
            mod->BSIM3V2cle = value->rValue;
            mod->BSIM3V2cleGiven = TRUE;
            break;
        case  BSIM3V2_MOD_DWC :
            mod->BSIM3V2dwc = value->rValue;
            mod->BSIM3V2dwcGiven = TRUE;
            break;
        case  BSIM3V2_MOD_DLC :
            mod->BSIM3V2dlc = value->rValue;
            mod->BSIM3V2dlcGiven = TRUE;
            break;
        case  BSIM3V2_MOD_VFBCV :
            mod->BSIM3V2vfbcv = value->rValue;
            mod->BSIM3V2vfbcvGiven = TRUE;
            break;
        case  BSIM3V2_MOD_ACDE :
            mod->BSIM3V2acde = value->rValue;
            mod->BSIM3V2acdeGiven = TRUE;
            break;
        case  BSIM3V2_MOD_MOIN :
            mod->BSIM3V2moin = value->rValue;
            mod->BSIM3V2moinGiven = TRUE;
            break;
        case  BSIM3V2_MOD_NOFF :
            mod->BSIM3V2noff = value->rValue;
            mod->BSIM3V2noffGiven = TRUE;
            break;
        case  BSIM3V2_MOD_VOFFCV :
            mod->BSIM3V2voffcv = value->rValue;
            mod->BSIM3V2voffcvGiven = TRUE;
            break;
        case  BSIM3V2_MOD_TCJ :
            mod->BSIM3V2tcj = value->rValue;
            mod->BSIM3V2tcjGiven = TRUE;
            break;
        case  BSIM3V2_MOD_TPB :
            mod->BSIM3V2tpb = value->rValue;
            mod->BSIM3V2tpbGiven = TRUE;
            break;
        case  BSIM3V2_MOD_TCJSW :
            mod->BSIM3V2tcjsw = value->rValue;
            mod->BSIM3V2tcjswGiven = TRUE;
            break;
        case  BSIM3V2_MOD_TPBSW :
            mod->BSIM3V2tpbsw = value->rValue;
            mod->BSIM3V2tpbswGiven = TRUE;
            break;
        case  BSIM3V2_MOD_TCJSWG :
            mod->BSIM3V2tcjswg = value->rValue;
            mod->BSIM3V2tcjswgGiven = TRUE;
            break;
        case  BSIM3V2_MOD_TPBSWG :
            mod->BSIM3V2tpbswg = value->rValue;
            mod->BSIM3V2tpbswgGiven = TRUE;
            break;

	/* Length dependence */
        case  BSIM3V2_MOD_LCDSC :
            mod->BSIM3V2lcdsc = value->rValue;
            mod->BSIM3V2lcdscGiven = TRUE;
            break;


        case  BSIM3V2_MOD_LCDSCB :
            mod->BSIM3V2lcdscb = value->rValue;
            mod->BSIM3V2lcdscbGiven = TRUE;
            break;
        case  BSIM3V2_MOD_LCDSCD :
            mod->BSIM3V2lcdscd = value->rValue;
            mod->BSIM3V2lcdscdGiven = TRUE;
            break;
        case  BSIM3V2_MOD_LCIT :
            mod->BSIM3V2lcit = value->rValue;
            mod->BSIM3V2lcitGiven = TRUE;
            break;
        case  BSIM3V2_MOD_LNFACTOR :
            mod->BSIM3V2lnfactor = value->rValue;
            mod->BSIM3V2lnfactorGiven = TRUE;
            break;
        case BSIM3V2_MOD_LXJ:
            mod->BSIM3V2lxj = value->rValue;
            mod->BSIM3V2lxjGiven = TRUE;
            break;
        case BSIM3V2_MOD_LVSAT:
            mod->BSIM3V2lvsat = value->rValue;
            mod->BSIM3V2lvsatGiven = TRUE;
            break;
        
        
        case BSIM3V2_MOD_LA0:
            mod->BSIM3V2la0 = value->rValue;
            mod->BSIM3V2la0Given = TRUE;
            break;
        case BSIM3V2_MOD_LAGS:
            mod->BSIM3V2lags = value->rValue;
            mod->BSIM3V2lagsGiven = TRUE;
            break;
        case BSIM3V2_MOD_LA1:
            mod->BSIM3V2la1 = value->rValue;
            mod->BSIM3V2la1Given = TRUE;
            break;
        case BSIM3V2_MOD_LA2:
            mod->BSIM3V2la2 = value->rValue;
            mod->BSIM3V2la2Given = TRUE;
            break;
        case BSIM3V2_MOD_LAT:
            mod->BSIM3V2lat = value->rValue;
            mod->BSIM3V2latGiven = TRUE;
            break;
        case BSIM3V2_MOD_LKETA:
            mod->BSIM3V2lketa = value->rValue;
            mod->BSIM3V2lketaGiven = TRUE;
            break;    
        case BSIM3V2_MOD_LNSUB:
            mod->BSIM3V2lnsub = value->rValue;
            mod->BSIM3V2lnsubGiven = TRUE;
            break;
        case BSIM3V2_MOD_LNPEAK:
            mod->BSIM3V2lnpeak = value->rValue;
            mod->BSIM3V2lnpeakGiven = TRUE;
	    if (mod->BSIM3V2lnpeak > 1.0e20)
		mod->BSIM3V2lnpeak *= 1.0e-6;
            break;
        case BSIM3V2_MOD_LNGATE:
            mod->BSIM3V2lngate = value->rValue;
            mod->BSIM3V2lngateGiven = TRUE;
	    if (mod->BSIM3V2lngate > 1.0e23)
		mod->BSIM3V2lngate *= 1.0e-6;
            break;
        case BSIM3V2_MOD_LGAMMA1:
            mod->BSIM3V2lgamma1 = value->rValue;
            mod->BSIM3V2lgamma1Given = TRUE;
            break;
        case BSIM3V2_MOD_LGAMMA2:
            mod->BSIM3V2lgamma2 = value->rValue;
            mod->BSIM3V2lgamma2Given = TRUE;
            break;
        case BSIM3V2_MOD_LVBX:
            mod->BSIM3V2lvbx = value->rValue;
            mod->BSIM3V2lvbxGiven = TRUE;
            break;
        case BSIM3V2_MOD_LVBM:
            mod->BSIM3V2lvbm = value->rValue;
            mod->BSIM3V2lvbmGiven = TRUE;
            break;
        case BSIM3V2_MOD_LXT:
            mod->BSIM3V2lxt = value->rValue;
            mod->BSIM3V2lxtGiven = TRUE;
            break;
        case  BSIM3V2_MOD_LK1:
            mod->BSIM3V2lk1 = value->rValue;
            mod->BSIM3V2lk1Given = TRUE;
            break;
        case  BSIM3V2_MOD_LKT1:
            mod->BSIM3V2lkt1 = value->rValue;
            mod->BSIM3V2lkt1Given = TRUE;
            break;
        case  BSIM3V2_MOD_LKT1L:
            mod->BSIM3V2lkt1l = value->rValue;
            mod->BSIM3V2lkt1lGiven = TRUE;
            break;
        case  BSIM3V2_MOD_LKT2:
            mod->BSIM3V2lkt2 = value->rValue;
            mod->BSIM3V2lkt2Given = TRUE;
            break;
        case  BSIM3V2_MOD_LK2:
            mod->BSIM3V2lk2 = value->rValue;
            mod->BSIM3V2lk2Given = TRUE;
            break;
        case  BSIM3V2_MOD_LK3:
            mod->BSIM3V2lk3 = value->rValue;
            mod->BSIM3V2lk3Given = TRUE;
            break;
        case  BSIM3V2_MOD_LK3B:
            mod->BSIM3V2lk3b = value->rValue;
            mod->BSIM3V2lk3bGiven = TRUE;
            break;
        case  BSIM3V2_MOD_LNLX:
            mod->BSIM3V2lnlx = value->rValue;
            mod->BSIM3V2lnlxGiven = TRUE;
            break;
        case  BSIM3V2_MOD_LW0:
            mod->BSIM3V2lw0 = value->rValue;
            mod->BSIM3V2lw0Given = TRUE;
            break;
        case  BSIM3V2_MOD_LDVT0:               
            mod->BSIM3V2ldvt0 = value->rValue;
            mod->BSIM3V2ldvt0Given = TRUE;
            break;
        case  BSIM3V2_MOD_LDVT1:             
            mod->BSIM3V2ldvt1 = value->rValue;
            mod->BSIM3V2ldvt1Given = TRUE;
            break;
        case  BSIM3V2_MOD_LDVT2:             
            mod->BSIM3V2ldvt2 = value->rValue;
            mod->BSIM3V2ldvt2Given = TRUE;
            break;
        case  BSIM3V2_MOD_LDVT0W:               
            mod->BSIM3V2ldvt0w = value->rValue;
            mod->BSIM3V2ldvt0wGiven = TRUE;
            break;
        case  BSIM3V2_MOD_LDVT1W:             
            mod->BSIM3V2ldvt1w = value->rValue;
            mod->BSIM3V2ldvt1wGiven = TRUE;
            break;
        case  BSIM3V2_MOD_LDVT2W:             
            mod->BSIM3V2ldvt2w = value->rValue;
            mod->BSIM3V2ldvt2wGiven = TRUE;
            break;
        case  BSIM3V2_MOD_LDROUT:             
            mod->BSIM3V2ldrout = value->rValue;
            mod->BSIM3V2ldroutGiven = TRUE;
            break;
        case  BSIM3V2_MOD_LDSUB:             
            mod->BSIM3V2ldsub = value->rValue;
            mod->BSIM3V2ldsubGiven = TRUE;
            break;
        case BSIM3V2_MOD_LVTH0:
            mod->BSIM3V2lvth0 = value->rValue;
            mod->BSIM3V2lvth0Given = TRUE;
            break;
        case BSIM3V2_MOD_LUA:
            mod->BSIM3V2lua = value->rValue;
            mod->BSIM3V2luaGiven = TRUE;
            break;
        case BSIM3V2_MOD_LUA1:
            mod->BSIM3V2lua1 = value->rValue;
            mod->BSIM3V2lua1Given = TRUE;
            break;
        case BSIM3V2_MOD_LUB:
            mod->BSIM3V2lub = value->rValue;
            mod->BSIM3V2lubGiven = TRUE;
            break;
        case BSIM3V2_MOD_LUB1:
            mod->BSIM3V2lub1 = value->rValue;
            mod->BSIM3V2lub1Given = TRUE;
            break;
        case BSIM3V2_MOD_LUC:
            mod->BSIM3V2luc = value->rValue;
            mod->BSIM3V2lucGiven = TRUE;
            break;
        case BSIM3V2_MOD_LUC1:
            mod->BSIM3V2luc1 = value->rValue;
            mod->BSIM3V2luc1Given = TRUE;
            break;
        case  BSIM3V2_MOD_LU0 :
            mod->BSIM3V2lu0 = value->rValue;
            mod->BSIM3V2lu0Given = TRUE;
            break;
        case  BSIM3V2_MOD_LUTE :
            mod->BSIM3V2lute = value->rValue;
            mod->BSIM3V2luteGiven = TRUE;
            break;
        case BSIM3V2_MOD_LVOFF:
            mod->BSIM3V2lvoff = value->rValue;
            mod->BSIM3V2lvoffGiven = TRUE;
            break;
        case  BSIM3V2_MOD_LDELTA :
            mod->BSIM3V2ldelta = value->rValue;
            mod->BSIM3V2ldeltaGiven = TRUE;
            break;
        case BSIM3V2_MOD_LRDSW:
            mod->BSIM3V2lrdsw = value->rValue;
            mod->BSIM3V2lrdswGiven = TRUE;
            break;                     
        case BSIM3V2_MOD_LPRWB:
            mod->BSIM3V2lprwb = value->rValue;
            mod->BSIM3V2lprwbGiven = TRUE;
            break;                     
        case BSIM3V2_MOD_LPRWG:
            mod->BSIM3V2lprwg = value->rValue;
            mod->BSIM3V2lprwgGiven = TRUE;
            break;                     
        case BSIM3V2_MOD_LPRT:
            mod->BSIM3V2lprt = value->rValue;
            mod->BSIM3V2lprtGiven = TRUE;
            break;                     
        case BSIM3V2_MOD_LETA0:
            mod->BSIM3V2leta0 = value->rValue;
            mod->BSIM3V2leta0Given = TRUE;
            break;                 
        case BSIM3V2_MOD_LETAB:
            mod->BSIM3V2letab = value->rValue;
            mod->BSIM3V2letabGiven = TRUE;
            break;                 
        case BSIM3V2_MOD_LPCLM:
            mod->BSIM3V2lpclm = value->rValue;
            mod->BSIM3V2lpclmGiven = TRUE;
            break;                 
        case BSIM3V2_MOD_LPDIBL1:
            mod->BSIM3V2lpdibl1 = value->rValue;
            mod->BSIM3V2lpdibl1Given = TRUE;
            break;                 
        case BSIM3V2_MOD_LPDIBL2:
            mod->BSIM3V2lpdibl2 = value->rValue;
            mod->BSIM3V2lpdibl2Given = TRUE;
            break;                 
        case BSIM3V2_MOD_LPDIBLB:
            mod->BSIM3V2lpdiblb = value->rValue;
            mod->BSIM3V2lpdiblbGiven = TRUE;
            break;                 
        case BSIM3V2_MOD_LPSCBE1:
            mod->BSIM3V2lpscbe1 = value->rValue;
            mod->BSIM3V2lpscbe1Given = TRUE;
            break;                 
        case BSIM3V2_MOD_LPSCBE2:
            mod->BSIM3V2lpscbe2 = value->rValue;
            mod->BSIM3V2lpscbe2Given = TRUE;
            break;                 
        case BSIM3V2_MOD_LPVAG:
            mod->BSIM3V2lpvag = value->rValue;
            mod->BSIM3V2lpvagGiven = TRUE;
            break;                 
        case  BSIM3V2_MOD_LWR :
            mod->BSIM3V2lwr = value->rValue;
            mod->BSIM3V2lwrGiven = TRUE;
            break;
        case  BSIM3V2_MOD_LDWG :
            mod->BSIM3V2ldwg = value->rValue;
            mod->BSIM3V2ldwgGiven = TRUE;
            break;
        case  BSIM3V2_MOD_LDWB :
            mod->BSIM3V2ldwb = value->rValue;
            mod->BSIM3V2ldwbGiven = TRUE;
            break;
        case  BSIM3V2_MOD_LB0 :
            mod->BSIM3V2lb0 = value->rValue;
            mod->BSIM3V2lb0Given = TRUE;
            break;
        case  BSIM3V2_MOD_LB1 :
            mod->BSIM3V2lb1 = value->rValue;
            mod->BSIM3V2lb1Given = TRUE;
            break;
        case  BSIM3V2_MOD_LALPHA0 :
            mod->BSIM3V2lalpha0 = value->rValue;
            mod->BSIM3V2lalpha0Given = TRUE;
            break;
        case  BSIM3V2_MOD_LALPHA1 :
            mod->BSIM3V2lalpha1 = value->rValue;
            mod->BSIM3V2lalpha1Given = TRUE;
            break;
        case  BSIM3V2_MOD_LBETA0 :
            mod->BSIM3V2lbeta0 = value->rValue;
            mod->BSIM3V2lbeta0Given = TRUE;
            break;
        case  BSIM3V2_MOD_LVFB :
            mod->BSIM3V2lvfb = value->rValue;
            mod->BSIM3V2lvfbGiven = TRUE;
            break;

        case  BSIM3V2_MOD_LELM :
            mod->BSIM3V2lelm = value->rValue;
            mod->BSIM3V2lelmGiven = TRUE;
            break;
        case  BSIM3V2_MOD_LCGSL :
            mod->BSIM3V2lcgsl = value->rValue;
            mod->BSIM3V2lcgslGiven = TRUE;
            break;
        case  BSIM3V2_MOD_LCGDL :
            mod->BSIM3V2lcgdl = value->rValue;
            mod->BSIM3V2lcgdlGiven = TRUE;
            break;
        case  BSIM3V2_MOD_LCKAPPA :
            mod->BSIM3V2lckappa = value->rValue;
            mod->BSIM3V2lckappaGiven = TRUE;
            break;
        case  BSIM3V2_MOD_LCF :
            mod->BSIM3V2lcf = value->rValue;
            mod->BSIM3V2lcfGiven = TRUE;
            break;
        case  BSIM3V2_MOD_LCLC :
            mod->BSIM3V2lclc = value->rValue;
            mod->BSIM3V2lclcGiven = TRUE;
            break;
        case  BSIM3V2_MOD_LCLE :
            mod->BSIM3V2lcle = value->rValue;
            mod->BSIM3V2lcleGiven = TRUE;
            break;
        case  BSIM3V2_MOD_LVFBCV :
            mod->BSIM3V2lvfbcv = value->rValue;
            mod->BSIM3V2lvfbcvGiven = TRUE;
            break;
        case  BSIM3V2_MOD_LACDE :
            mod->BSIM3V2lacde = value->rValue;
            mod->BSIM3V2lacdeGiven = TRUE;
            break;
        case  BSIM3V2_MOD_LMOIN :
            mod->BSIM3V2lmoin = value->rValue;
            mod->BSIM3V2lmoinGiven = TRUE;
            break;
        case  BSIM3V2_MOD_LNOFF :
            mod->BSIM3V2lnoff = value->rValue;
            mod->BSIM3V2lnoffGiven = TRUE;
            break;
        case  BSIM3V2_MOD_LVOFFCV :
            mod->BSIM3V2lvoffcv = value->rValue;
            mod->BSIM3V2lvoffcvGiven = TRUE;
            break;

	/* Width dependence */
        case  BSIM3V2_MOD_WCDSC :
            mod->BSIM3V2wcdsc = value->rValue;
            mod->BSIM3V2wcdscGiven = TRUE;
            break;
       
       
         case  BSIM3V2_MOD_WCDSCB :
            mod->BSIM3V2wcdscb = value->rValue;
            mod->BSIM3V2wcdscbGiven = TRUE;
            break;
         case  BSIM3V2_MOD_WCDSCD :
            mod->BSIM3V2wcdscd = value->rValue;
            mod->BSIM3V2wcdscdGiven = TRUE;
            break;
        case  BSIM3V2_MOD_WCIT :
            mod->BSIM3V2wcit = value->rValue;
            mod->BSIM3V2wcitGiven = TRUE;
            break;
        case  BSIM3V2_MOD_WNFACTOR :
            mod->BSIM3V2wnfactor = value->rValue;
            mod->BSIM3V2wnfactorGiven = TRUE;
            break;
        case BSIM3V2_MOD_WXJ:
            mod->BSIM3V2wxj = value->rValue;
            mod->BSIM3V2wxjGiven = TRUE;
            break;
        case BSIM3V2_MOD_WVSAT:
            mod->BSIM3V2wvsat = value->rValue;
            mod->BSIM3V2wvsatGiven = TRUE;
            break;


        case BSIM3V2_MOD_WA0:
            mod->BSIM3V2wa0 = value->rValue;
            mod->BSIM3V2wa0Given = TRUE;
            break;
        case BSIM3V2_MOD_WAGS:
            mod->BSIM3V2wags = value->rValue;
            mod->BSIM3V2wagsGiven = TRUE;
            break;
        case BSIM3V2_MOD_WA1:
            mod->BSIM3V2wa1 = value->rValue;
            mod->BSIM3V2wa1Given = TRUE;
            break;
        case BSIM3V2_MOD_WA2:
            mod->BSIM3V2wa2 = value->rValue;
            mod->BSIM3V2wa2Given = TRUE;
            break;
        case BSIM3V2_MOD_WAT:
            mod->BSIM3V2wat = value->rValue;
            mod->BSIM3V2watGiven = TRUE;
            break;
        case BSIM3V2_MOD_WKETA:
            mod->BSIM3V2wketa = value->rValue;
            mod->BSIM3V2wketaGiven = TRUE;
            break;    
        case BSIM3V2_MOD_WNSUB:
            mod->BSIM3V2wnsub = value->rValue;
            mod->BSIM3V2wnsubGiven = TRUE;
            break;
        case BSIM3V2_MOD_WNPEAK:
            mod->BSIM3V2wnpeak = value->rValue;
            mod->BSIM3V2wnpeakGiven = TRUE;
	    if (mod->BSIM3V2wnpeak > 1.0e20)
		mod->BSIM3V2wnpeak *= 1.0e-6;
            break;
        case BSIM3V2_MOD_WNGATE:
            mod->BSIM3V2wngate = value->rValue;
            mod->BSIM3V2wngateGiven = TRUE;
	    if (mod->BSIM3V2wngate > 1.0e23)
		mod->BSIM3V2wngate *= 1.0e-6;
            break;
        case BSIM3V2_MOD_WGAMMA1:
            mod->BSIM3V2wgamma1 = value->rValue;
            mod->BSIM3V2wgamma1Given = TRUE;
            break;
        case BSIM3V2_MOD_WGAMMA2:
            mod->BSIM3V2wgamma2 = value->rValue;
            mod->BSIM3V2wgamma2Given = TRUE;
            break;
        case BSIM3V2_MOD_WVBX:
            mod->BSIM3V2wvbx = value->rValue;
            mod->BSIM3V2wvbxGiven = TRUE;
            break;
        case BSIM3V2_MOD_WVBM:
            mod->BSIM3V2wvbm = value->rValue;
            mod->BSIM3V2wvbmGiven = TRUE;
            break;
        case BSIM3V2_MOD_WXT:
            mod->BSIM3V2wxt = value->rValue;
            mod->BSIM3V2wxtGiven = TRUE;
            break;
        case  BSIM3V2_MOD_WK1:
            mod->BSIM3V2wk1 = value->rValue;
            mod->BSIM3V2wk1Given = TRUE;
            break;
        case  BSIM3V2_MOD_WKT1:
            mod->BSIM3V2wkt1 = value->rValue;
            mod->BSIM3V2wkt1Given = TRUE;
            break;
        case  BSIM3V2_MOD_WKT1L:
            mod->BSIM3V2wkt1l = value->rValue;
            mod->BSIM3V2wkt1lGiven = TRUE;
            break;
        case  BSIM3V2_MOD_WKT2:
            mod->BSIM3V2wkt2 = value->rValue;
            mod->BSIM3V2wkt2Given = TRUE;
            break;
        case  BSIM3V2_MOD_WK2:
            mod->BSIM3V2wk2 = value->rValue;
            mod->BSIM3V2wk2Given = TRUE;
            break;
        case  BSIM3V2_MOD_WK3:
            mod->BSIM3V2wk3 = value->rValue;
            mod->BSIM3V2wk3Given = TRUE;
            break;
        case  BSIM3V2_MOD_WK3B:
            mod->BSIM3V2wk3b = value->rValue;
            mod->BSIM3V2wk3bGiven = TRUE;
            break;
        case  BSIM3V2_MOD_WNLX:
            mod->BSIM3V2wnlx = value->rValue;
            mod->BSIM3V2wnlxGiven = TRUE;
            break;
        case  BSIM3V2_MOD_WW0:
            mod->BSIM3V2ww0 = value->rValue;
            mod->BSIM3V2ww0Given = TRUE;
            break;
        case  BSIM3V2_MOD_WDVT0:               
            mod->BSIM3V2wdvt0 = value->rValue;
            mod->BSIM3V2wdvt0Given = TRUE;
            break;
        case  BSIM3V2_MOD_WDVT1:             
            mod->BSIM3V2wdvt1 = value->rValue;
            mod->BSIM3V2wdvt1Given = TRUE;
            break;
        case  BSIM3V2_MOD_WDVT2:             
            mod->BSIM3V2wdvt2 = value->rValue;
            mod->BSIM3V2wdvt2Given = TRUE;
            break;
        case  BSIM3V2_MOD_WDVT0W:               
            mod->BSIM3V2wdvt0w = value->rValue;
            mod->BSIM3V2wdvt0wGiven = TRUE;
            break;
        case  BSIM3V2_MOD_WDVT1W:             
            mod->BSIM3V2wdvt1w = value->rValue;
            mod->BSIM3V2wdvt1wGiven = TRUE;
            break;
        case  BSIM3V2_MOD_WDVT2W:             
            mod->BSIM3V2wdvt2w = value->rValue;
            mod->BSIM3V2wdvt2wGiven = TRUE;
            break;
        case  BSIM3V2_MOD_WDROUT:             
            mod->BSIM3V2wdrout = value->rValue;
            mod->BSIM3V2wdroutGiven = TRUE;
            break;
        case  BSIM3V2_MOD_WDSUB:             
            mod->BSIM3V2wdsub = value->rValue;
            mod->BSIM3V2wdsubGiven = TRUE;
            break;
        case BSIM3V2_MOD_WVTH0:
            mod->BSIM3V2wvth0 = value->rValue;
            mod->BSIM3V2wvth0Given = TRUE;
            break;
        case BSIM3V2_MOD_WUA:
            mod->BSIM3V2wua = value->rValue;
            mod->BSIM3V2wuaGiven = TRUE;
            break;
        case BSIM3V2_MOD_WUA1:
            mod->BSIM3V2wua1 = value->rValue;
            mod->BSIM3V2wua1Given = TRUE;
            break;
        case BSIM3V2_MOD_WUB:
            mod->BSIM3V2wub = value->rValue;
            mod->BSIM3V2wubGiven = TRUE;
            break;
        case BSIM3V2_MOD_WUB1:
            mod->BSIM3V2wub1 = value->rValue;
            mod->BSIM3V2wub1Given = TRUE;
            break;
        case BSIM3V2_MOD_WUC:
            mod->BSIM3V2wuc = value->rValue;
            mod->BSIM3V2wucGiven = TRUE;
            break;
        case BSIM3V2_MOD_WUC1:
            mod->BSIM3V2wuc1 = value->rValue;
            mod->BSIM3V2wuc1Given = TRUE;
            break;
        case  BSIM3V2_MOD_WU0 :
            mod->BSIM3V2wu0 = value->rValue;
            mod->BSIM3V2wu0Given = TRUE;
            break;
        case  BSIM3V2_MOD_WUTE :
            mod->BSIM3V2wute = value->rValue;
            mod->BSIM3V2wuteGiven = TRUE;
            break;
        case BSIM3V2_MOD_WVOFF:
            mod->BSIM3V2wvoff = value->rValue;
            mod->BSIM3V2wvoffGiven = TRUE;
            break;
        case  BSIM3V2_MOD_WDELTA :
            mod->BSIM3V2wdelta = value->rValue;
            mod->BSIM3V2wdeltaGiven = TRUE;
            break;
        case BSIM3V2_MOD_WRDSW:
            mod->BSIM3V2wrdsw = value->rValue;
            mod->BSIM3V2wrdswGiven = TRUE;
            break;                     
        case BSIM3V2_MOD_WPRWB:
            mod->BSIM3V2wprwb = value->rValue;
            mod->BSIM3V2wprwbGiven = TRUE;
            break;                     
        case BSIM3V2_MOD_WPRWG:
            mod->BSIM3V2wprwg = value->rValue;
            mod->BSIM3V2wprwgGiven = TRUE;
            break;                     
        case BSIM3V2_MOD_WPRT:
            mod->BSIM3V2wprt = value->rValue;
            mod->BSIM3V2wprtGiven = TRUE;
            break;                     
        case BSIM3V2_MOD_WETA0:
            mod->BSIM3V2weta0 = value->rValue;
            mod->BSIM3V2weta0Given = TRUE;
            break;                 
        case BSIM3V2_MOD_WETAB:
            mod->BSIM3V2wetab = value->rValue;
            mod->BSIM3V2wetabGiven = TRUE;
            break;                 
        case BSIM3V2_MOD_WPCLM:
            mod->BSIM3V2wpclm = value->rValue;
            mod->BSIM3V2wpclmGiven = TRUE;
            break;                 
        case BSIM3V2_MOD_WPDIBL1:
            mod->BSIM3V2wpdibl1 = value->rValue;
            mod->BSIM3V2wpdibl1Given = TRUE;
            break;                 
        case BSIM3V2_MOD_WPDIBL2:
            mod->BSIM3V2wpdibl2 = value->rValue;
            mod->BSIM3V2wpdibl2Given = TRUE;
            break;                 
        case BSIM3V2_MOD_WPDIBLB:
            mod->BSIM3V2wpdiblb = value->rValue;
            mod->BSIM3V2wpdiblbGiven = TRUE;
            break;                 
        case BSIM3V2_MOD_WPSCBE1:
            mod->BSIM3V2wpscbe1 = value->rValue;
            mod->BSIM3V2wpscbe1Given = TRUE;
            break;                 
        case BSIM3V2_MOD_WPSCBE2:
            mod->BSIM3V2wpscbe2 = value->rValue;
            mod->BSIM3V2wpscbe2Given = TRUE;
            break;                 
        case BSIM3V2_MOD_WPVAG:
            mod->BSIM3V2wpvag = value->rValue;
            mod->BSIM3V2wpvagGiven = TRUE;
            break;                 
        case  BSIM3V2_MOD_WWR :
            mod->BSIM3V2wwr = value->rValue;
            mod->BSIM3V2wwrGiven = TRUE;
            break;
        case  BSIM3V2_MOD_WDWG :
            mod->BSIM3V2wdwg = value->rValue;
            mod->BSIM3V2wdwgGiven = TRUE;
            break;
        case  BSIM3V2_MOD_WDWB :
            mod->BSIM3V2wdwb = value->rValue;
            mod->BSIM3V2wdwbGiven = TRUE;
            break;
        case  BSIM3V2_MOD_WB0 :
            mod->BSIM3V2wb0 = value->rValue;
            mod->BSIM3V2wb0Given = TRUE;
            break;
        case  BSIM3V2_MOD_WB1 :
            mod->BSIM3V2wb1 = value->rValue;
            mod->BSIM3V2wb1Given = TRUE;
            break;
        case  BSIM3V2_MOD_WALPHA0 :
            mod->BSIM3V2walpha0 = value->rValue;
            mod->BSIM3V2walpha0Given = TRUE;
            break;
        case  BSIM3V2_MOD_WALPHA1 :
            mod->BSIM3V2walpha1 = value->rValue;
            mod->BSIM3V2walpha1Given = TRUE;
            break;
        case  BSIM3V2_MOD_WBETA0 :
            mod->BSIM3V2wbeta0 = value->rValue;
            mod->BSIM3V2wbeta0Given = TRUE;
            break;
        case  BSIM3V2_MOD_WVFB :
            mod->BSIM3V2wvfb = value->rValue;
            mod->BSIM3V2wvfbGiven = TRUE;
            break;

        case  BSIM3V2_MOD_WELM :
            mod->BSIM3V2welm = value->rValue;
            mod->BSIM3V2welmGiven = TRUE;
            break;
        case  BSIM3V2_MOD_WCGSL :
            mod->BSIM3V2wcgsl = value->rValue;
            mod->BSIM3V2wcgslGiven = TRUE;
            break;
        case  BSIM3V2_MOD_WCGDL :
            mod->BSIM3V2wcgdl = value->rValue;
            mod->BSIM3V2wcgdlGiven = TRUE;
            break;
        case  BSIM3V2_MOD_WCKAPPA :
            mod->BSIM3V2wckappa = value->rValue;
            mod->BSIM3V2wckappaGiven = TRUE;
            break;
        case  BSIM3V2_MOD_WCF :
            mod->BSIM3V2wcf = value->rValue;
            mod->BSIM3V2wcfGiven = TRUE;
            break;
        case  BSIM3V2_MOD_WCLC :
            mod->BSIM3V2wclc = value->rValue;
            mod->BSIM3V2wclcGiven = TRUE;
            break;
        case  BSIM3V2_MOD_WCLE :
            mod->BSIM3V2wcle = value->rValue;
            mod->BSIM3V2wcleGiven = TRUE;
            break;
        case  BSIM3V2_MOD_WVFBCV :
            mod->BSIM3V2wvfbcv = value->rValue;
            mod->BSIM3V2wvfbcvGiven = TRUE;
            break;
        case  BSIM3V2_MOD_WACDE :
            mod->BSIM3V2wacde = value->rValue;
            mod->BSIM3V2wacdeGiven = TRUE;
            break;
        case  BSIM3V2_MOD_WMOIN :
            mod->BSIM3V2wmoin = value->rValue;
            mod->BSIM3V2wmoinGiven = TRUE;
            break;
        case  BSIM3V2_MOD_WNOFF :
            mod->BSIM3V2wnoff = value->rValue;
            mod->BSIM3V2wnoffGiven = TRUE;
            break;
        case  BSIM3V2_MOD_WVOFFCV :
            mod->BSIM3V2wvoffcv = value->rValue;
            mod->BSIM3V2wvoffcvGiven = TRUE;
            break;

	/* Cross-term dependence */
        case  BSIM3V2_MOD_PCDSC :
            mod->BSIM3V2pcdsc = value->rValue;
            mod->BSIM3V2pcdscGiven = TRUE;
            break;


        case  BSIM3V2_MOD_PCDSCB :
            mod->BSIM3V2pcdscb = value->rValue;
            mod->BSIM3V2pcdscbGiven = TRUE;
            break;
        case  BSIM3V2_MOD_PCDSCD :
            mod->BSIM3V2pcdscd = value->rValue;
            mod->BSIM3V2pcdscdGiven = TRUE;
            break;
        case  BSIM3V2_MOD_PCIT :
            mod->BSIM3V2pcit = value->rValue;
            mod->BSIM3V2pcitGiven = TRUE;
            break;
        case  BSIM3V2_MOD_PNFACTOR :
            mod->BSIM3V2pnfactor = value->rValue;
            mod->BSIM3V2pnfactorGiven = TRUE;
            break;
        case BSIM3V2_MOD_PXJ:
            mod->BSIM3V2pxj = value->rValue;
            mod->BSIM3V2pxjGiven = TRUE;
            break;
        case BSIM3V2_MOD_PVSAT:
            mod->BSIM3V2pvsat = value->rValue;
            mod->BSIM3V2pvsatGiven = TRUE;
            break;


        case BSIM3V2_MOD_PA0:
            mod->BSIM3V2pa0 = value->rValue;
            mod->BSIM3V2pa0Given = TRUE;
            break;
        case BSIM3V2_MOD_PAGS:
            mod->BSIM3V2pags = value->rValue;
            mod->BSIM3V2pagsGiven = TRUE;
            break;
        case BSIM3V2_MOD_PA1:
            mod->BSIM3V2pa1 = value->rValue;
            mod->BSIM3V2pa1Given = TRUE;
            break;
        case BSIM3V2_MOD_PA2:
            mod->BSIM3V2pa2 = value->rValue;
            mod->BSIM3V2pa2Given = TRUE;
            break;
        case BSIM3V2_MOD_PAT:
            mod->BSIM3V2pat = value->rValue;
            mod->BSIM3V2patGiven = TRUE;
            break;
        case BSIM3V2_MOD_PKETA:
            mod->BSIM3V2pketa = value->rValue;
            mod->BSIM3V2pketaGiven = TRUE;
            break;    
        case BSIM3V2_MOD_PNSUB:
            mod->BSIM3V2pnsub = value->rValue;
            mod->BSIM3V2pnsubGiven = TRUE;
            break;
        case BSIM3V2_MOD_PNPEAK:
            mod->BSIM3V2pnpeak = value->rValue;
            mod->BSIM3V2pnpeakGiven = TRUE;
	    if (mod->BSIM3V2pnpeak > 1.0e20)
		mod->BSIM3V2pnpeak *= 1.0e-6;
            break;
        case BSIM3V2_MOD_PNGATE:
            mod->BSIM3V2pngate = value->rValue;
            mod->BSIM3V2pngateGiven = TRUE;
	    if (mod->BSIM3V2pngate > 1.0e23)
		mod->BSIM3V2pngate *= 1.0e-6;
            break;
        case BSIM3V2_MOD_PGAMMA1:
            mod->BSIM3V2pgamma1 = value->rValue;
            mod->BSIM3V2pgamma1Given = TRUE;
            break;
        case BSIM3V2_MOD_PGAMMA2:
            mod->BSIM3V2pgamma2 = value->rValue;
            mod->BSIM3V2pgamma2Given = TRUE;
            break;
        case BSIM3V2_MOD_PVBX:
            mod->BSIM3V2pvbx = value->rValue;
            mod->BSIM3V2pvbxGiven = TRUE;
            break;
        case BSIM3V2_MOD_PVBM:
            mod->BSIM3V2pvbm = value->rValue;
            mod->BSIM3V2pvbmGiven = TRUE;
            break;
        case BSIM3V2_MOD_PXT:
            mod->BSIM3V2pxt = value->rValue;
            mod->BSIM3V2pxtGiven = TRUE;
            break;
        case  BSIM3V2_MOD_PK1:
            mod->BSIM3V2pk1 = value->rValue;
            mod->BSIM3V2pk1Given = TRUE;
            break;
        case  BSIM3V2_MOD_PKT1:
            mod->BSIM3V2pkt1 = value->rValue;
            mod->BSIM3V2pkt1Given = TRUE;
            break;
        case  BSIM3V2_MOD_PKT1L:
            mod->BSIM3V2pkt1l = value->rValue;
            mod->BSIM3V2pkt1lGiven = TRUE;
            break;
        case  BSIM3V2_MOD_PKT2:
            mod->BSIM3V2pkt2 = value->rValue;
            mod->BSIM3V2pkt2Given = TRUE;
            break;
        case  BSIM3V2_MOD_PK2:
            mod->BSIM3V2pk2 = value->rValue;
            mod->BSIM3V2pk2Given = TRUE;
            break;
        case  BSIM3V2_MOD_PK3:
            mod->BSIM3V2pk3 = value->rValue;
            mod->BSIM3V2pk3Given = TRUE;
            break;
        case  BSIM3V2_MOD_PK3B:
            mod->BSIM3V2pk3b = value->rValue;
            mod->BSIM3V2pk3bGiven = TRUE;
            break;
        case  BSIM3V2_MOD_PNLX:
            mod->BSIM3V2pnlx = value->rValue;
            mod->BSIM3V2pnlxGiven = TRUE;
            break;
        case  BSIM3V2_MOD_PW0:
            mod->BSIM3V2pw0 = value->rValue;
            mod->BSIM3V2pw0Given = TRUE;
            break;
        case  BSIM3V2_MOD_PDVT0:               
            mod->BSIM3V2pdvt0 = value->rValue;
            mod->BSIM3V2pdvt0Given = TRUE;
            break;
        case  BSIM3V2_MOD_PDVT1:             
            mod->BSIM3V2pdvt1 = value->rValue;
            mod->BSIM3V2pdvt1Given = TRUE;
            break;
        case  BSIM3V2_MOD_PDVT2:             
            mod->BSIM3V2pdvt2 = value->rValue;
            mod->BSIM3V2pdvt2Given = TRUE;
            break;
        case  BSIM3V2_MOD_PDVT0W:               
            mod->BSIM3V2pdvt0w = value->rValue;
            mod->BSIM3V2pdvt0wGiven = TRUE;
            break;
        case  BSIM3V2_MOD_PDVT1W:             
            mod->BSIM3V2pdvt1w = value->rValue;
            mod->BSIM3V2pdvt1wGiven = TRUE;
            break;
        case  BSIM3V2_MOD_PDVT2W:             
            mod->BSIM3V2pdvt2w = value->rValue;
            mod->BSIM3V2pdvt2wGiven = TRUE;
            break;
        case  BSIM3V2_MOD_PDROUT:             
            mod->BSIM3V2pdrout = value->rValue;
            mod->BSIM3V2pdroutGiven = TRUE;
            break;
        case  BSIM3V2_MOD_PDSUB:             
            mod->BSIM3V2pdsub = value->rValue;
            mod->BSIM3V2pdsubGiven = TRUE;
            break;
        case BSIM3V2_MOD_PVTH0:
            mod->BSIM3V2pvth0 = value->rValue;
            mod->BSIM3V2pvth0Given = TRUE;
            break;
        case BSIM3V2_MOD_PUA:
            mod->BSIM3V2pua = value->rValue;
            mod->BSIM3V2puaGiven = TRUE;
            break;
        case BSIM3V2_MOD_PUA1:
            mod->BSIM3V2pua1 = value->rValue;
            mod->BSIM3V2pua1Given = TRUE;
            break;
        case BSIM3V2_MOD_PUB:
            mod->BSIM3V2pub = value->rValue;
            mod->BSIM3V2pubGiven = TRUE;
            break;
        case BSIM3V2_MOD_PUB1:
            mod->BSIM3V2pub1 = value->rValue;
            mod->BSIM3V2pub1Given = TRUE;
            break;
        case BSIM3V2_MOD_PUC:
            mod->BSIM3V2puc = value->rValue;
            mod->BSIM3V2pucGiven = TRUE;
            break;
        case BSIM3V2_MOD_PUC1:
            mod->BSIM3V2puc1 = value->rValue;
            mod->BSIM3V2puc1Given = TRUE;
            break;
        case  BSIM3V2_MOD_PU0 :
            mod->BSIM3V2pu0 = value->rValue;
            mod->BSIM3V2pu0Given = TRUE;
            break;
        case  BSIM3V2_MOD_PUTE :
            mod->BSIM3V2pute = value->rValue;
            mod->BSIM3V2puteGiven = TRUE;
            break;
        case BSIM3V2_MOD_PVOFF:
            mod->BSIM3V2pvoff = value->rValue;
            mod->BSIM3V2pvoffGiven = TRUE;
            break;
        case  BSIM3V2_MOD_PDELTA :
            mod->BSIM3V2pdelta = value->rValue;
            mod->BSIM3V2pdeltaGiven = TRUE;
            break;
        case BSIM3V2_MOD_PRDSW:
            mod->BSIM3V2prdsw = value->rValue;
            mod->BSIM3V2prdswGiven = TRUE;
            break;                     
        case BSIM3V2_MOD_PPRWB:
            mod->BSIM3V2pprwb = value->rValue;
            mod->BSIM3V2pprwbGiven = TRUE;
            break;                     
        case BSIM3V2_MOD_PPRWG:
            mod->BSIM3V2pprwg = value->rValue;
            mod->BSIM3V2pprwgGiven = TRUE;
            break;                     
        case BSIM3V2_MOD_PPRT:
            mod->BSIM3V2pprt = value->rValue;
            mod->BSIM3V2pprtGiven = TRUE;
            break;                     
        case BSIM3V2_MOD_PETA0:
            mod->BSIM3V2peta0 = value->rValue;
            mod->BSIM3V2peta0Given = TRUE;
            break;                 
        case BSIM3V2_MOD_PETAB:
            mod->BSIM3V2petab = value->rValue;
            mod->BSIM3V2petabGiven = TRUE;
            break;                 
        case BSIM3V2_MOD_PPCLM:
            mod->BSIM3V2ppclm = value->rValue;
            mod->BSIM3V2ppclmGiven = TRUE;
            break;                 
        case BSIM3V2_MOD_PPDIBL1:
            mod->BSIM3V2ppdibl1 = value->rValue;
            mod->BSIM3V2ppdibl1Given = TRUE;
            break;                 
        case BSIM3V2_MOD_PPDIBL2:
            mod->BSIM3V2ppdibl2 = value->rValue;
            mod->BSIM3V2ppdibl2Given = TRUE;
            break;                 
        case BSIM3V2_MOD_PPDIBLB:
            mod->BSIM3V2ppdiblb = value->rValue;
            mod->BSIM3V2ppdiblbGiven = TRUE;
            break;                 
        case BSIM3V2_MOD_PPSCBE1:
            mod->BSIM3V2ppscbe1 = value->rValue;
            mod->BSIM3V2ppscbe1Given = TRUE;
            break;                 
        case BSIM3V2_MOD_PPSCBE2:
            mod->BSIM3V2ppscbe2 = value->rValue;
            mod->BSIM3V2ppscbe2Given = TRUE;
            break;                 
        case BSIM3V2_MOD_PPVAG:
            mod->BSIM3V2ppvag = value->rValue;
            mod->BSIM3V2ppvagGiven = TRUE;
            break;                 
        case  BSIM3V2_MOD_PWR :
            mod->BSIM3V2pwr = value->rValue;
            mod->BSIM3V2pwrGiven = TRUE;
            break;
        case  BSIM3V2_MOD_PDWG :
            mod->BSIM3V2pdwg = value->rValue;
            mod->BSIM3V2pdwgGiven = TRUE;
            break;
        case  BSIM3V2_MOD_PDWB :
            mod->BSIM3V2pdwb = value->rValue;
            mod->BSIM3V2pdwbGiven = TRUE;
            break;
        case  BSIM3V2_MOD_PB0 :
            mod->BSIM3V2pb0 = value->rValue;
            mod->BSIM3V2pb0Given = TRUE;
            break;
        case  BSIM3V2_MOD_PB1 :
            mod->BSIM3V2pb1 = value->rValue;
            mod->BSIM3V2pb1Given = TRUE;
            break;
        case  BSIM3V2_MOD_PALPHA0 :
            mod->BSIM3V2palpha0 = value->rValue;
            mod->BSIM3V2palpha0Given = TRUE;
            break;
        case  BSIM3V2_MOD_PALPHA1 :
            mod->BSIM3V2palpha1 = value->rValue;
            mod->BSIM3V2palpha1Given = TRUE;
            break;
        case  BSIM3V2_MOD_PBETA0 :
            mod->BSIM3V2pbeta0 = value->rValue;
            mod->BSIM3V2pbeta0Given = TRUE;
            break;
        case  BSIM3V2_MOD_PVFB :
            mod->BSIM3V2pvfb = value->rValue;
            mod->BSIM3V2pvfbGiven = TRUE;
            break;

        case  BSIM3V2_MOD_PELM :
            mod->BSIM3V2pelm = value->rValue;
            mod->BSIM3V2pelmGiven = TRUE;
            break;
        case  BSIM3V2_MOD_PCGSL :
            mod->BSIM3V2pcgsl = value->rValue;
            mod->BSIM3V2pcgslGiven = TRUE;
            break;
        case  BSIM3V2_MOD_PCGDL :
            mod->BSIM3V2pcgdl = value->rValue;
            mod->BSIM3V2pcgdlGiven = TRUE;
            break;
        case  BSIM3V2_MOD_PCKAPPA :
            mod->BSIM3V2pckappa = value->rValue;
            mod->BSIM3V2pckappaGiven = TRUE;
            break;
        case  BSIM3V2_MOD_PCF :
            mod->BSIM3V2pcf = value->rValue;
            mod->BSIM3V2pcfGiven = TRUE;
            break;
        case  BSIM3V2_MOD_PCLC :
            mod->BSIM3V2pclc = value->rValue;
            mod->BSIM3V2pclcGiven = TRUE;
            break;
        case  BSIM3V2_MOD_PCLE :
            mod->BSIM3V2pcle = value->rValue;
            mod->BSIM3V2pcleGiven = TRUE;
            break;
        case  BSIM3V2_MOD_PVFBCV :
            mod->BSIM3V2pvfbcv = value->rValue;
            mod->BSIM3V2pvfbcvGiven = TRUE;
            break;
        case  BSIM3V2_MOD_PACDE :
            mod->BSIM3V2pacde = value->rValue;
            mod->BSIM3V2pacdeGiven = TRUE;
            break;
        case  BSIM3V2_MOD_PMOIN :
            mod->BSIM3V2pmoin = value->rValue;
            mod->BSIM3V2pmoinGiven = TRUE;
            break;
        case  BSIM3V2_MOD_PNOFF :
            mod->BSIM3V2pnoff = value->rValue;
            mod->BSIM3V2pnoffGiven = TRUE;
            break;
        case  BSIM3V2_MOD_PVOFFCV :
            mod->BSIM3V2pvoffcv = value->rValue;
            mod->BSIM3V2pvoffcvGiven = TRUE;
            break;

        case  BSIM3V2_MOD_TNOM :
            mod->BSIM3V2tnom = value->rValue + 273.15;
            mod->BSIM3V2tnomGiven = TRUE;
            break;
        case  BSIM3V2_MOD_CGSO :
            mod->BSIM3V2cgso = value->rValue;
            mod->BSIM3V2cgsoGiven = TRUE;
            break;
        case  BSIM3V2_MOD_CGDO :
            mod->BSIM3V2cgdo = value->rValue;
            mod->BSIM3V2cgdoGiven = TRUE;
            break;
        case  BSIM3V2_MOD_CGBO :
            mod->BSIM3V2cgbo = value->rValue;
            mod->BSIM3V2cgboGiven = TRUE;
            break;
        case  BSIM3V2_MOD_XPART :
            mod->BSIM3V2xpart = value->rValue;
            mod->BSIM3V2xpartGiven = TRUE;
            break;
        case  BSIM3V2_MOD_RSH :
            mod->BSIM3V2sheetResistance = value->rValue;
            mod->BSIM3V2sheetResistanceGiven = TRUE;
            break;
        case  BSIM3V2_MOD_JS :
            mod->BSIM3V2jctSatCurDensity = value->rValue;
            mod->BSIM3V2jctSatCurDensityGiven = TRUE;
            break;
        case  BSIM3V2_MOD_JSW :
            mod->BSIM3V2jctSidewallSatCurDensity = value->rValue;
            mod->BSIM3V2jctSidewallSatCurDensityGiven = TRUE;
            break;
        case  BSIM3V2_MOD_PB :
            mod->BSIM3V2bulkJctPotential = value->rValue;
            mod->BSIM3V2bulkJctPotentialGiven = TRUE;
            break;
        case  BSIM3V2_MOD_MJ :
            mod->BSIM3V2bulkJctBotGradingCoeff = value->rValue;
            mod->BSIM3V2bulkJctBotGradingCoeffGiven = TRUE;
            break;
        case  BSIM3V2_MOD_PBSW :
            mod->BSIM3V2sidewallJctPotential = value->rValue;
            mod->BSIM3V2sidewallJctPotentialGiven = TRUE;
            break;
        case  BSIM3V2_MOD_MJSW :
            mod->BSIM3V2bulkJctSideGradingCoeff = value->rValue;
            mod->BSIM3V2bulkJctSideGradingCoeffGiven = TRUE;
            break;
        case  BSIM3V2_MOD_CJ :
            mod->BSIM3V2unitAreaJctCap = value->rValue;
            mod->BSIM3V2unitAreaJctCapGiven = TRUE;
            break;
        case  BSIM3V2_MOD_CJSW :
            mod->BSIM3V2unitLengthSidewallJctCap = value->rValue;
            mod->BSIM3V2unitLengthSidewallJctCapGiven = TRUE;
            break;
        case  BSIM3V2_MOD_NJ :
            mod->BSIM3V2jctEmissionCoeff = value->rValue;
            mod->BSIM3V2jctEmissionCoeffGiven = TRUE;
            break;
        case  BSIM3V2_MOD_PBSWG :
            mod->BSIM3V2GatesidewallJctPotential = value->rValue;
            mod->BSIM3V2GatesidewallJctPotentialGiven = TRUE;
            break;
        case  BSIM3V2_MOD_MJSWG :
            mod->BSIM3V2bulkJctGateSideGradingCoeff = value->rValue;
            mod->BSIM3V2bulkJctGateSideGradingCoeffGiven = TRUE;
            break;
        case  BSIM3V2_MOD_CJSWG :
            mod->BSIM3V2unitLengthGateSidewallJctCap = value->rValue;
            mod->BSIM3V2unitLengthGateSidewallJctCapGiven = TRUE;
            break;
        case  BSIM3V2_MOD_XTI :
            mod->BSIM3V2jctTempExponent = value->rValue;
            mod->BSIM3V2jctTempExponentGiven = TRUE;
            break;
        case  BSIM3V2_MOD_LINT :
            mod->BSIM3V2Lint = value->rValue;
            mod->BSIM3V2LintGiven = TRUE;
            break;
        case  BSIM3V2_MOD_LL :
            mod->BSIM3V2Ll = value->rValue;
            mod->BSIM3V2LlGiven = TRUE;
            break;
        case  BSIM3V2_MOD_LLC :
            mod->BSIM3V2Llc = value->rValue;
            mod->BSIM3V2LlcGiven = TRUE;
            break;
        case  BSIM3V2_MOD_LLN :
            mod->BSIM3V2Lln = value->rValue;
            mod->BSIM3V2LlnGiven = TRUE;
            break;
        case  BSIM3V2_MOD_LW :
            mod->BSIM3V2Lw = value->rValue;
            mod->BSIM3V2LwGiven = TRUE;
            break;
        case  BSIM3V2_MOD_LWC :
            mod->BSIM3V2Lwc = value->rValue;
            mod->BSIM3V2LwcGiven = TRUE;
            break;
        case  BSIM3V2_MOD_LWN :
            mod->BSIM3V2Lwn = value->rValue;
            mod->BSIM3V2LwnGiven = TRUE;
            break;
        case  BSIM3V2_MOD_LWL :
            mod->BSIM3V2Lwl = value->rValue;
            mod->BSIM3V2LwlGiven = TRUE;
            break;
        case  BSIM3V2_MOD_LWLC :
            mod->BSIM3V2Lwlc = value->rValue;
            mod->BSIM3V2LwlcGiven = TRUE;
            break;
        case  BSIM3V2_MOD_LMIN :
            mod->BSIM3V2Lmin = value->rValue;
            mod->BSIM3V2LminGiven = TRUE;
            break;
        case  BSIM3V2_MOD_LMAX :
            mod->BSIM3V2Lmax = value->rValue;
            mod->BSIM3V2LmaxGiven = TRUE;
            break;
        case  BSIM3V2_MOD_WINT :
            mod->BSIM3V2Wint = value->rValue;
            mod->BSIM3V2WintGiven = TRUE;
            break;
        case  BSIM3V2_MOD_WL :
            mod->BSIM3V2Wl = value->rValue;
            mod->BSIM3V2WlGiven = TRUE;
            break;
        case  BSIM3V2_MOD_WLC :
            mod->BSIM3V2Wlc = value->rValue;
            mod->BSIM3V2WlcGiven = TRUE;
            break;
        case  BSIM3V2_MOD_WLN :
            mod->BSIM3V2Wln = value->rValue;
            mod->BSIM3V2WlnGiven = TRUE;
            break;
        case  BSIM3V2_MOD_WW :
            mod->BSIM3V2Ww = value->rValue;
            mod->BSIM3V2WwGiven = TRUE;
            break;
        case  BSIM3V2_MOD_WWC :
            mod->BSIM3V2Wwc = value->rValue;
            mod->BSIM3V2WwcGiven = TRUE;
            break;
        case  BSIM3V2_MOD_WWN :
            mod->BSIM3V2Wwn = value->rValue;
            mod->BSIM3V2WwnGiven = TRUE;
            break;
        case  BSIM3V2_MOD_WWL :
            mod->BSIM3V2Wwl = value->rValue;
            mod->BSIM3V2WwlGiven = TRUE;
            break;
        case  BSIM3V2_MOD_WWLC :
            mod->BSIM3V2Wwlc = value->rValue;
            mod->BSIM3V2WwlcGiven = TRUE;
            break;
        case  BSIM3V2_MOD_WMIN :
            mod->BSIM3V2Wmin = value->rValue;
            mod->BSIM3V2WminGiven = TRUE;
            break;
        case  BSIM3V2_MOD_WMAX :
            mod->BSIM3V2Wmax = value->rValue;
            mod->BSIM3V2WmaxGiven = TRUE;
            break;

        case  BSIM3V2_MOD_NOIA :
            mod->BSIM3V2oxideTrapDensityA = value->rValue;
            mod->BSIM3V2oxideTrapDensityAGiven = TRUE;
            break;
        case  BSIM3V2_MOD_NOIB :
            mod->BSIM3V2oxideTrapDensityB = value->rValue;
            mod->BSIM3V2oxideTrapDensityBGiven = TRUE;
            break;
        case  BSIM3V2_MOD_NOIC :
            mod->BSIM3V2oxideTrapDensityC = value->rValue;
            mod->BSIM3V2oxideTrapDensityCGiven = TRUE;
            break;
        case  BSIM3V2_MOD_EM :
            mod->BSIM3V2em = value->rValue;
            mod->BSIM3V2emGiven = TRUE;
            break;
        case  BSIM3V2_MOD_EF :
            mod->BSIM3V2ef = value->rValue;
            mod->BSIM3V2efGiven = TRUE;
            break;
        case  BSIM3V2_MOD_AF :
            mod->BSIM3V2af = value->rValue;
            mod->BSIM3V2afGiven = TRUE;
            break;
        case  BSIM3V2_MOD_KF :
            mod->BSIM3V2kf = value->rValue;
            mod->BSIM3V2kfGiven = TRUE;
            break;
        case  BSIM3V2_MOD_NMOS  :
            if(value->iValue) {
                mod->BSIM3V2type = 1;
                mod->BSIM3V2typeGiven = TRUE;
            }
            break;
        case  BSIM3V2_MOD_PMOS  :
            if(value->iValue) {
                mod->BSIM3V2type = - 1;
                mod->BSIM3V2typeGiven = TRUE;
            }
            break;
        default:
            return(E_BADPARM);
    }
    return(OK);
}


