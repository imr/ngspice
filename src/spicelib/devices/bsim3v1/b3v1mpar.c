/* $Id$  */
/* 
$Log$
Revision 1.1.1.1  2000-04-27 20:03:59  pnenzi
Imported sources

 * Revision 3.1  96/12/08  19:56:49  yuhua
 * 	
 * BSIM3v3.1 release
 * 
*/
static char rcsid[] = "$Id$";

/*************************************/

/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1995 Min-Chie Jeng and Mansun Chan.
File: b3v1mpar.c
**********/

#include "ngspice.h"
#include <stdio.h>
#include "bsim3v1def.h"
#include "ifsim.h"
#include "sperror.h"
#include "suffix.h"


int
BSIM3V1mParam(param,value,inMod)
int param;
IFvalue *value;
GENmodel *inMod;
{
    BSIM3V1model *mod = (BSIM3V1model*)inMod;
    switch(param)
    {   case  BSIM3V1_MOD_MOBMOD :
            mod->BSIM3V1mobMod = value->iValue;
            mod->BSIM3V1mobModGiven = TRUE;
            break;
        case  BSIM3V1_MOD_BINUNIT :
            mod->BSIM3V1binUnit = value->iValue;
            mod->BSIM3V1binUnitGiven = TRUE;
            break;
        case  BSIM3V1_MOD_PARAMCHK :
            mod->BSIM3V1paramChk = value->iValue;
            mod->BSIM3V1paramChkGiven = TRUE;
            break;
        case  BSIM3V1_MOD_CAPMOD :
            mod->BSIM3V1capMod = value->iValue;
            mod->BSIM3V1capModGiven = TRUE;
            break;
        case  BSIM3V1_MOD_NQSMOD :
            mod->BSIM3V1nqsMod = value->iValue;
            mod->BSIM3V1nqsModGiven = TRUE;
            break;
        case  BSIM3V1_MOD_NOIMOD :
            mod->BSIM3V1noiMod = value->iValue;
            mod->BSIM3V1noiModGiven = TRUE;
            break;
        case  BSIM3V1_MOD_VERSION :
            mod->BSIM3V1version = value->rValue;
            mod->BSIM3V1versionGiven = TRUE;
            break;
        case  BSIM3V1_MOD_TOX :
            mod->BSIM3V1tox = value->rValue;
            mod->BSIM3V1toxGiven = TRUE;
            break;

        case  BSIM3V1_MOD_CDSC :
            mod->BSIM3V1cdsc = value->rValue;
            mod->BSIM3V1cdscGiven = TRUE;
            break;
        case  BSIM3V1_MOD_CDSCB :
            mod->BSIM3V1cdscb = value->rValue;
            mod->BSIM3V1cdscbGiven = TRUE;
            break;

        case  BSIM3V1_MOD_CDSCD :
            mod->BSIM3V1cdscd = value->rValue;
            mod->BSIM3V1cdscdGiven = TRUE;
            break;

        case  BSIM3V1_MOD_CIT :
            mod->BSIM3V1cit = value->rValue;
            mod->BSIM3V1citGiven = TRUE;
            break;
        case  BSIM3V1_MOD_NFACTOR :
            mod->BSIM3V1nfactor = value->rValue;
            mod->BSIM3V1nfactorGiven = TRUE;
            break;
        case BSIM3V1_MOD_XJ:
            mod->BSIM3V1xj = value->rValue;
            mod->BSIM3V1xjGiven = TRUE;
            break;
        case BSIM3V1_MOD_VSAT:
            mod->BSIM3V1vsat = value->rValue;
            mod->BSIM3V1vsatGiven = TRUE;
            break;
        case BSIM3V1_MOD_A0:
            mod->BSIM3V1a0 = value->rValue;
            mod->BSIM3V1a0Given = TRUE;
            break;
        
        case BSIM3V1_MOD_AGS:
            mod->BSIM3V1ags= value->rValue;
            mod->BSIM3V1agsGiven = TRUE;
            break;
        
        case BSIM3V1_MOD_A1:
            mod->BSIM3V1a1 = value->rValue;
            mod->BSIM3V1a1Given = TRUE;
            break;
        case BSIM3V1_MOD_A2:
            mod->BSIM3V1a2 = value->rValue;
            mod->BSIM3V1a2Given = TRUE;
            break;
        case BSIM3V1_MOD_AT:
            mod->BSIM3V1at = value->rValue;
            mod->BSIM3V1atGiven = TRUE;
            break;
        case BSIM3V1_MOD_KETA:
            mod->BSIM3V1keta = value->rValue;
            mod->BSIM3V1ketaGiven = TRUE;
            break;    
        case BSIM3V1_MOD_NSUB:
            mod->BSIM3V1nsub = value->rValue;
            mod->BSIM3V1nsubGiven = TRUE;
            break;
        case BSIM3V1_MOD_NPEAK:
            mod->BSIM3V1npeak = value->rValue;
            mod->BSIM3V1npeakGiven = TRUE;
	    if (mod->BSIM3V1npeak > 1.0e20)
		mod->BSIM3V1npeak *= 1.0e-6;
            break;
        case BSIM3V1_MOD_NGATE:
            mod->BSIM3V1ngate = value->rValue;
            mod->BSIM3V1ngateGiven = TRUE;
	    if (mod->BSIM3V1ngate > 1.0e23)
		mod->BSIM3V1ngate *= 1.0e-6;
            break;
        case BSIM3V1_MOD_GAMMA1:
            mod->BSIM3V1gamma1 = value->rValue;
            mod->BSIM3V1gamma1Given = TRUE;
            break;
        case BSIM3V1_MOD_GAMMA2:
            mod->BSIM3V1gamma2 = value->rValue;
            mod->BSIM3V1gamma2Given = TRUE;
            break;
        case BSIM3V1_MOD_VBX:
            mod->BSIM3V1vbx = value->rValue;
            mod->BSIM3V1vbxGiven = TRUE;
            break;
        case BSIM3V1_MOD_VBM:
            mod->BSIM3V1vbm = value->rValue;
            mod->BSIM3V1vbmGiven = TRUE;
            break;
        case BSIM3V1_MOD_XT:
            mod->BSIM3V1xt = value->rValue;
            mod->BSIM3V1xtGiven = TRUE;
            break;
        case  BSIM3V1_MOD_K1:
            mod->BSIM3V1k1 = value->rValue;
            mod->BSIM3V1k1Given = TRUE;
            break;
        case  BSIM3V1_MOD_KT1:
            mod->BSIM3V1kt1 = value->rValue;
            mod->BSIM3V1kt1Given = TRUE;
            break;
        case  BSIM3V1_MOD_KT1L:
            mod->BSIM3V1kt1l = value->rValue;
            mod->BSIM3V1kt1lGiven = TRUE;
            break;
        case  BSIM3V1_MOD_KT2:
            mod->BSIM3V1kt2 = value->rValue;
            mod->BSIM3V1kt2Given = TRUE;
            break;
        case  BSIM3V1_MOD_K2:
            mod->BSIM3V1k2 = value->rValue;
            mod->BSIM3V1k2Given = TRUE;
            break;
        case  BSIM3V1_MOD_K3:
            mod->BSIM3V1k3 = value->rValue;
            mod->BSIM3V1k3Given = TRUE;
            break;
        case  BSIM3V1_MOD_K3B:
            mod->BSIM3V1k3b = value->rValue;
            mod->BSIM3V1k3bGiven = TRUE;
            break;
        case  BSIM3V1_MOD_NLX:
            mod->BSIM3V1nlx = value->rValue;
            mod->BSIM3V1nlxGiven = TRUE;
            break;
        case  BSIM3V1_MOD_W0:
            mod->BSIM3V1w0 = value->rValue;
            mod->BSIM3V1w0Given = TRUE;
            break;
        case  BSIM3V1_MOD_DVT0:               
            mod->BSIM3V1dvt0 = value->rValue;
            mod->BSIM3V1dvt0Given = TRUE;
            break;
        case  BSIM3V1_MOD_DVT1:             
            mod->BSIM3V1dvt1 = value->rValue;
            mod->BSIM3V1dvt1Given = TRUE;
            break;
        case  BSIM3V1_MOD_DVT2:             
            mod->BSIM3V1dvt2 = value->rValue;
            mod->BSIM3V1dvt2Given = TRUE;
            break;
        case  BSIM3V1_MOD_DVT0W:               
            mod->BSIM3V1dvt0w = value->rValue;
            mod->BSIM3V1dvt0wGiven = TRUE;
            break;
        case  BSIM3V1_MOD_DVT1W:             
            mod->BSIM3V1dvt1w = value->rValue;
            mod->BSIM3V1dvt1wGiven = TRUE;
            break;
        case  BSIM3V1_MOD_DVT2W:             
            mod->BSIM3V1dvt2w = value->rValue;
            mod->BSIM3V1dvt2wGiven = TRUE;
            break;
        case  BSIM3V1_MOD_DROUT:             
            mod->BSIM3V1drout = value->rValue;
            mod->BSIM3V1droutGiven = TRUE;
            break;
        case  BSIM3V1_MOD_DSUB:             
            mod->BSIM3V1dsub = value->rValue;
            mod->BSIM3V1dsubGiven = TRUE;
            break;
        case BSIM3V1_MOD_VTH0:
            mod->BSIM3V1vth0 = value->rValue;
            mod->BSIM3V1vth0Given = TRUE;
            break;
        case BSIM3V1_MOD_UA:
            mod->BSIM3V1ua = value->rValue;
            mod->BSIM3V1uaGiven = TRUE;
            break;
        case BSIM3V1_MOD_UA1:
            mod->BSIM3V1ua1 = value->rValue;
            mod->BSIM3V1ua1Given = TRUE;
            break;
        case BSIM3V1_MOD_UB:
            mod->BSIM3V1ub = value->rValue;
            mod->BSIM3V1ubGiven = TRUE;
            break;
        case BSIM3V1_MOD_UB1:
            mod->BSIM3V1ub1 = value->rValue;
            mod->BSIM3V1ub1Given = TRUE;
            break;
        case BSIM3V1_MOD_UC:
            mod->BSIM3V1uc = value->rValue;
            mod->BSIM3V1ucGiven = TRUE;
            break;
        case BSIM3V1_MOD_UC1:
            mod->BSIM3V1uc1 = value->rValue;
            mod->BSIM3V1uc1Given = TRUE;
            break;
        case  BSIM3V1_MOD_U0 :
            mod->BSIM3V1u0 = value->rValue;
            mod->BSIM3V1u0Given = TRUE;
            break;
        case  BSIM3V1_MOD_UTE :
            mod->BSIM3V1ute = value->rValue;
            mod->BSIM3V1uteGiven = TRUE;
            break;
        case BSIM3V1_MOD_VOFF:
            mod->BSIM3V1voff = value->rValue;
            mod->BSIM3V1voffGiven = TRUE;
            break;
        case  BSIM3V1_MOD_DELTA :
            mod->BSIM3V1delta = value->rValue;
            mod->BSIM3V1deltaGiven = TRUE;
            break;
        case BSIM3V1_MOD_RDSW:
            mod->BSIM3V1rdsw = value->rValue;
            mod->BSIM3V1rdswGiven = TRUE;
            break;                     
        case BSIM3V1_MOD_PRWG:
            mod->BSIM3V1prwg = value->rValue;
            mod->BSIM3V1prwgGiven = TRUE;
            break;                     
        case BSIM3V1_MOD_PRWB:
            mod->BSIM3V1prwb = value->rValue;
            mod->BSIM3V1prwbGiven = TRUE;
            break;                     
        case BSIM3V1_MOD_PRT:
            mod->BSIM3V1prt = value->rValue;
            mod->BSIM3V1prtGiven = TRUE;
            break;                     
        case BSIM3V1_MOD_ETA0:
            mod->BSIM3V1eta0 = value->rValue;
            mod->BSIM3V1eta0Given = TRUE;
            break;                 
        case BSIM3V1_MOD_ETAB:
            mod->BSIM3V1etab = value->rValue;
            mod->BSIM3V1etabGiven = TRUE;
            break;                 
        case BSIM3V1_MOD_PCLM:
            mod->BSIM3V1pclm = value->rValue;
            mod->BSIM3V1pclmGiven = TRUE;
            break;                 
        case BSIM3V1_MOD_PDIBL1:
            mod->BSIM3V1pdibl1 = value->rValue;
            mod->BSIM3V1pdibl1Given = TRUE;
            break;                 
        case BSIM3V1_MOD_PDIBL2:
            mod->BSIM3V1pdibl2 = value->rValue;
            mod->BSIM3V1pdibl2Given = TRUE;
            break;                 
        case BSIM3V1_MOD_PDIBLB:
            mod->BSIM3V1pdiblb = value->rValue;
            mod->BSIM3V1pdiblbGiven = TRUE;
            break;                 
        case BSIM3V1_MOD_PSCBE1:
            mod->BSIM3V1pscbe1 = value->rValue;
            mod->BSIM3V1pscbe1Given = TRUE;
            break;                 
        case BSIM3V1_MOD_PSCBE2:
            mod->BSIM3V1pscbe2 = value->rValue;
            mod->BSIM3V1pscbe2Given = TRUE;
            break;                 
        case BSIM3V1_MOD_PVAG:
            mod->BSIM3V1pvag = value->rValue;
            mod->BSIM3V1pvagGiven = TRUE;
            break;                 
        case  BSIM3V1_MOD_WR :
            mod->BSIM3V1wr = value->rValue;
            mod->BSIM3V1wrGiven = TRUE;
            break;
        case  BSIM3V1_MOD_DWG :
            mod->BSIM3V1dwg = value->rValue;
            mod->BSIM3V1dwgGiven = TRUE;
            break;
        case  BSIM3V1_MOD_DWB :
            mod->BSIM3V1dwb = value->rValue;
            mod->BSIM3V1dwbGiven = TRUE;
            break;
        case  BSIM3V1_MOD_B0 :
            mod->BSIM3V1b0 = value->rValue;
            mod->BSIM3V1b0Given = TRUE;
            break;
        case  BSIM3V1_MOD_B1 :
            mod->BSIM3V1b1 = value->rValue;
            mod->BSIM3V1b1Given = TRUE;
            break;
        case  BSIM3V1_MOD_ALPHA0 :
            mod->BSIM3V1alpha0 = value->rValue;
            mod->BSIM3V1alpha0Given = TRUE;
            break;
        case  BSIM3V1_MOD_BETA0 :
            mod->BSIM3V1beta0 = value->rValue;
            mod->BSIM3V1beta0Given = TRUE;
            break;

        case  BSIM3V1_MOD_ELM :
            mod->BSIM3V1elm = value->rValue;
            mod->BSIM3V1elmGiven = TRUE;
            break;
        case  BSIM3V1_MOD_CGSL :
            mod->BSIM3V1cgsl = value->rValue;
            mod->BSIM3V1cgslGiven = TRUE;
            break;
        case  BSIM3V1_MOD_CGDL :
            mod->BSIM3V1cgdl = value->rValue;
            mod->BSIM3V1cgdlGiven = TRUE;
            break;
        case  BSIM3V1_MOD_CKAPPA :
            mod->BSIM3V1ckappa = value->rValue;
            mod->BSIM3V1ckappaGiven = TRUE;
            break;
        case  BSIM3V1_MOD_CF :
            mod->BSIM3V1cf = value->rValue;
            mod->BSIM3V1cfGiven = TRUE;
            break;
        case  BSIM3V1_MOD_CLC :
            mod->BSIM3V1clc = value->rValue;
            mod->BSIM3V1clcGiven = TRUE;
            break;
        case  BSIM3V1_MOD_CLE :
            mod->BSIM3V1cle = value->rValue;
            mod->BSIM3V1cleGiven = TRUE;
            break;
        case  BSIM3V1_MOD_DWC :
            mod->BSIM3V1dwc = value->rValue;
            mod->BSIM3V1dwcGiven = TRUE;
            break;
        case  BSIM3V1_MOD_DLC :
            mod->BSIM3V1dlc = value->rValue;
            mod->BSIM3V1dlcGiven = TRUE;
            break;
        case  BSIM3V1_MOD_VFBCV :
            mod->BSIM3V1vfbcv = value->rValue;
            mod->BSIM3V1vfbcvGiven = TRUE;
            break;

	/* Length dependence */
        case  BSIM3V1_MOD_LCDSC :
            mod->BSIM3V1lcdsc = value->rValue;
            mod->BSIM3V1lcdscGiven = TRUE;
            break;


        case  BSIM3V1_MOD_LCDSCB :
            mod->BSIM3V1lcdscb = value->rValue;
            mod->BSIM3V1lcdscbGiven = TRUE;
            break;
        case  BSIM3V1_MOD_LCDSCD :
            mod->BSIM3V1lcdscd = value->rValue;
            mod->BSIM3V1lcdscdGiven = TRUE;
            break;
        case  BSIM3V1_MOD_LCIT :
            mod->BSIM3V1lcit = value->rValue;
            mod->BSIM3V1lcitGiven = TRUE;
            break;
        case  BSIM3V1_MOD_LNFACTOR :
            mod->BSIM3V1lnfactor = value->rValue;
            mod->BSIM3V1lnfactorGiven = TRUE;
            break;
        case BSIM3V1_MOD_LXJ:
            mod->BSIM3V1lxj = value->rValue;
            mod->BSIM3V1lxjGiven = TRUE;
            break;
        case BSIM3V1_MOD_LVSAT:
            mod->BSIM3V1lvsat = value->rValue;
            mod->BSIM3V1lvsatGiven = TRUE;
            break;
        
        
        case BSIM3V1_MOD_LA0:
            mod->BSIM3V1la0 = value->rValue;
            mod->BSIM3V1la0Given = TRUE;
            break;
        case BSIM3V1_MOD_LAGS:
            mod->BSIM3V1lags = value->rValue;
            mod->BSIM3V1lagsGiven = TRUE;
            break;
        case BSIM3V1_MOD_LA1:
            mod->BSIM3V1la1 = value->rValue;
            mod->BSIM3V1la1Given = TRUE;
            break;
        case BSIM3V1_MOD_LA2:
            mod->BSIM3V1la2 = value->rValue;
            mod->BSIM3V1la2Given = TRUE;
            break;
        case BSIM3V1_MOD_LAT:
            mod->BSIM3V1lat = value->rValue;
            mod->BSIM3V1latGiven = TRUE;
            break;
        case BSIM3V1_MOD_LKETA:
            mod->BSIM3V1lketa = value->rValue;
            mod->BSIM3V1lketaGiven = TRUE;
            break;    
        case BSIM3V1_MOD_LNSUB:
            mod->BSIM3V1lnsub = value->rValue;
            mod->BSIM3V1lnsubGiven = TRUE;
            break;
        case BSIM3V1_MOD_LNPEAK:
            mod->BSIM3V1lnpeak = value->rValue;
            mod->BSIM3V1lnpeakGiven = TRUE;
	    if (mod->BSIM3V1lnpeak > 1.0e20)
		mod->BSIM3V1lnpeak *= 1.0e-6;
            break;
        case BSIM3V1_MOD_LNGATE:
            mod->BSIM3V1lngate = value->rValue;
            mod->BSIM3V1lngateGiven = TRUE;
	    if (mod->BSIM3V1lngate > 1.0e23)
		mod->BSIM3V1lngate *= 1.0e-6;
            break;
        case BSIM3V1_MOD_LGAMMA1:
            mod->BSIM3V1lgamma1 = value->rValue;
            mod->BSIM3V1lgamma1Given = TRUE;
            break;
        case BSIM3V1_MOD_LGAMMA2:
            mod->BSIM3V1lgamma2 = value->rValue;
            mod->BSIM3V1lgamma2Given = TRUE;
            break;
        case BSIM3V1_MOD_LVBX:
            mod->BSIM3V1lvbx = value->rValue;
            mod->BSIM3V1lvbxGiven = TRUE;
            break;
        case BSIM3V1_MOD_LVBM:
            mod->BSIM3V1lvbm = value->rValue;
            mod->BSIM3V1lvbmGiven = TRUE;
            break;
        case BSIM3V1_MOD_LXT:
            mod->BSIM3V1lxt = value->rValue;
            mod->BSIM3V1lxtGiven = TRUE;
            break;
        case  BSIM3V1_MOD_LK1:
            mod->BSIM3V1lk1 = value->rValue;
            mod->BSIM3V1lk1Given = TRUE;
            break;
        case  BSIM3V1_MOD_LKT1:
            mod->BSIM3V1lkt1 = value->rValue;
            mod->BSIM3V1lkt1Given = TRUE;
            break;
        case  BSIM3V1_MOD_LKT1L:
            mod->BSIM3V1lkt1l = value->rValue;
            mod->BSIM3V1lkt1lGiven = TRUE;
            break;
        case  BSIM3V1_MOD_LKT2:
            mod->BSIM3V1lkt2 = value->rValue;
            mod->BSIM3V1lkt2Given = TRUE;
            break;
        case  BSIM3V1_MOD_LK2:
            mod->BSIM3V1lk2 = value->rValue;
            mod->BSIM3V1lk2Given = TRUE;
            break;
        case  BSIM3V1_MOD_LK3:
            mod->BSIM3V1lk3 = value->rValue;
            mod->BSIM3V1lk3Given = TRUE;
            break;
        case  BSIM3V1_MOD_LK3B:
            mod->BSIM3V1lk3b = value->rValue;
            mod->BSIM3V1lk3bGiven = TRUE;
            break;
        case  BSIM3V1_MOD_LNLX:
            mod->BSIM3V1lnlx = value->rValue;
            mod->BSIM3V1lnlxGiven = TRUE;
            break;
        case  BSIM3V1_MOD_LW0:
            mod->BSIM3V1lw0 = value->rValue;
            mod->BSIM3V1lw0Given = TRUE;
            break;
        case  BSIM3V1_MOD_LDVT0:               
            mod->BSIM3V1ldvt0 = value->rValue;
            mod->BSIM3V1ldvt0Given = TRUE;
            break;
        case  BSIM3V1_MOD_LDVT1:             
            mod->BSIM3V1ldvt1 = value->rValue;
            mod->BSIM3V1ldvt1Given = TRUE;
            break;
        case  BSIM3V1_MOD_LDVT2:             
            mod->BSIM3V1ldvt2 = value->rValue;
            mod->BSIM3V1ldvt2Given = TRUE;
            break;
        case  BSIM3V1_MOD_LDVT0W:               
            mod->BSIM3V1ldvt0w = value->rValue;
            mod->BSIM3V1ldvt0Given = TRUE;
            break;
        case  BSIM3V1_MOD_LDVT1W:             
            mod->BSIM3V1ldvt1w = value->rValue;
            mod->BSIM3V1ldvt1wGiven = TRUE;
            break;
        case  BSIM3V1_MOD_LDVT2W:             
            mod->BSIM3V1ldvt2w = value->rValue;
            mod->BSIM3V1ldvt2wGiven = TRUE;
            break;
        case  BSIM3V1_MOD_LDROUT:             
            mod->BSIM3V1ldrout = value->rValue;
            mod->BSIM3V1ldroutGiven = TRUE;
            break;
        case  BSIM3V1_MOD_LDSUB:             
            mod->BSIM3V1ldsub = value->rValue;
            mod->BSIM3V1ldsubGiven = TRUE;
            break;
        case BSIM3V1_MOD_LVTH0:
            mod->BSIM3V1lvth0 = value->rValue;
            mod->BSIM3V1lvth0Given = TRUE;
            break;
        case BSIM3V1_MOD_LUA:
            mod->BSIM3V1lua = value->rValue;
            mod->BSIM3V1luaGiven = TRUE;
            break;
        case BSIM3V1_MOD_LUA1:
            mod->BSIM3V1lua1 = value->rValue;
            mod->BSIM3V1lua1Given = TRUE;
            break;
        case BSIM3V1_MOD_LUB:
            mod->BSIM3V1lub = value->rValue;
            mod->BSIM3V1lubGiven = TRUE;
            break;
        case BSIM3V1_MOD_LUB1:
            mod->BSIM3V1lub1 = value->rValue;
            mod->BSIM3V1lub1Given = TRUE;
            break;
        case BSIM3V1_MOD_LUC:
            mod->BSIM3V1luc = value->rValue;
            mod->BSIM3V1lucGiven = TRUE;
            break;
        case BSIM3V1_MOD_LUC1:
            mod->BSIM3V1luc1 = value->rValue;
            mod->BSIM3V1luc1Given = TRUE;
            break;
        case  BSIM3V1_MOD_LU0 :
            mod->BSIM3V1lu0 = value->rValue;
            mod->BSIM3V1lu0Given = TRUE;
            break;
        case  BSIM3V1_MOD_LUTE :
            mod->BSIM3V1lute = value->rValue;
            mod->BSIM3V1luteGiven = TRUE;
            break;
        case BSIM3V1_MOD_LVOFF:
            mod->BSIM3V1lvoff = value->rValue;
            mod->BSIM3V1lvoffGiven = TRUE;
            break;
        case  BSIM3V1_MOD_LDELTA :
            mod->BSIM3V1ldelta = value->rValue;
            mod->BSIM3V1ldeltaGiven = TRUE;
            break;
        case BSIM3V1_MOD_LRDSW:
            mod->BSIM3V1lrdsw = value->rValue;
            mod->BSIM3V1lrdswGiven = TRUE;
            break;                     
        case BSIM3V1_MOD_LPRWB:
            mod->BSIM3V1lprwb = value->rValue;
            mod->BSIM3V1lprwbGiven = TRUE;
            break;                     
        case BSIM3V1_MOD_LPRWG:
            mod->BSIM3V1lprwg = value->rValue;
            mod->BSIM3V1lprwgGiven = TRUE;
            break;                     
        case BSIM3V1_MOD_LPRT:
            mod->BSIM3V1lprt = value->rValue;
            mod->BSIM3V1lprtGiven = TRUE;
            break;                     
        case BSIM3V1_MOD_LETA0:
            mod->BSIM3V1leta0 = value->rValue;
            mod->BSIM3V1leta0Given = TRUE;
            break;                 
        case BSIM3V1_MOD_LETAB:
            mod->BSIM3V1letab = value->rValue;
            mod->BSIM3V1letabGiven = TRUE;
            break;                 
        case BSIM3V1_MOD_LPCLM:
            mod->BSIM3V1lpclm = value->rValue;
            mod->BSIM3V1lpclmGiven = TRUE;
            break;                 
        case BSIM3V1_MOD_LPDIBL1:
            mod->BSIM3V1lpdibl1 = value->rValue;
            mod->BSIM3V1lpdibl1Given = TRUE;
            break;                 
        case BSIM3V1_MOD_LPDIBL2:
            mod->BSIM3V1lpdibl2 = value->rValue;
            mod->BSIM3V1lpdibl2Given = TRUE;
            break;                 
        case BSIM3V1_MOD_LPDIBLB:
            mod->BSIM3V1lpdiblb = value->rValue;
            mod->BSIM3V1lpdiblbGiven = TRUE;
            break;                 
        case BSIM3V1_MOD_LPSCBE1:
            mod->BSIM3V1lpscbe1 = value->rValue;
            mod->BSIM3V1lpscbe1Given = TRUE;
            break;                 
        case BSIM3V1_MOD_LPSCBE2:
            mod->BSIM3V1lpscbe2 = value->rValue;
            mod->BSIM3V1lpscbe2Given = TRUE;
            break;                 
        case BSIM3V1_MOD_LPVAG:
            mod->BSIM3V1lpvag = value->rValue;
            mod->BSIM3V1lpvagGiven = TRUE;
            break;                 
        case  BSIM3V1_MOD_LWR :
            mod->BSIM3V1lwr = value->rValue;
            mod->BSIM3V1lwrGiven = TRUE;
            break;
        case  BSIM3V1_MOD_LDWG :
            mod->BSIM3V1ldwg = value->rValue;
            mod->BSIM3V1ldwgGiven = TRUE;
            break;
        case  BSIM3V1_MOD_LDWB :
            mod->BSIM3V1ldwb = value->rValue;
            mod->BSIM3V1ldwbGiven = TRUE;
            break;
        case  BSIM3V1_MOD_LB0 :
            mod->BSIM3V1lb0 = value->rValue;
            mod->BSIM3V1lb0Given = TRUE;
            break;
        case  BSIM3V1_MOD_LB1 :
            mod->BSIM3V1lb1 = value->rValue;
            mod->BSIM3V1lb1Given = TRUE;
            break;
        case  BSIM3V1_MOD_LALPHA0 :
            mod->BSIM3V1lalpha0 = value->rValue;
            mod->BSIM3V1lalpha0Given = TRUE;
            break;
        case  BSIM3V1_MOD_LBETA0 :
            mod->BSIM3V1lbeta0 = value->rValue;
            mod->BSIM3V1lbeta0Given = TRUE;
            break;

        case  BSIM3V1_MOD_LELM :
            mod->BSIM3V1lelm = value->rValue;
            mod->BSIM3V1lelmGiven = TRUE;
            break;
        case  BSIM3V1_MOD_LCGSL :
            mod->BSIM3V1lcgsl = value->rValue;
            mod->BSIM3V1lcgslGiven = TRUE;
            break;
        case  BSIM3V1_MOD_LCGDL :
            mod->BSIM3V1lcgdl = value->rValue;
            mod->BSIM3V1lcgdlGiven = TRUE;
            break;
        case  BSIM3V1_MOD_LCKAPPA :
            mod->BSIM3V1lckappa = value->rValue;
            mod->BSIM3V1lckappaGiven = TRUE;
            break;
        case  BSIM3V1_MOD_LCF :
            mod->BSIM3V1lcf = value->rValue;
            mod->BSIM3V1lcfGiven = TRUE;
            break;
        case  BSIM3V1_MOD_LCLC :
            mod->BSIM3V1lclc = value->rValue;
            mod->BSIM3V1lclcGiven = TRUE;
            break;
        case  BSIM3V1_MOD_LCLE :
            mod->BSIM3V1lcle = value->rValue;
            mod->BSIM3V1lcleGiven = TRUE;
            break;
        case  BSIM3V1_MOD_LVFBCV :
            mod->BSIM3V1lvfbcv = value->rValue;
            mod->BSIM3V1lvfbcvGiven = TRUE;
            break;

	/* Width dependence */
        case  BSIM3V1_MOD_WCDSC :
            mod->BSIM3V1wcdsc = value->rValue;
            mod->BSIM3V1wcdscGiven = TRUE;
            break;
       
       
         case  BSIM3V1_MOD_WCDSCB :
            mod->BSIM3V1wcdscb = value->rValue;
            mod->BSIM3V1wcdscbGiven = TRUE;
            break;
         case  BSIM3V1_MOD_WCDSCD :
            mod->BSIM3V1wcdscd = value->rValue;
            mod->BSIM3V1wcdscdGiven = TRUE;
            break;
        case  BSIM3V1_MOD_WCIT :
            mod->BSIM3V1wcit = value->rValue;
            mod->BSIM3V1wcitGiven = TRUE;
            break;
        case  BSIM3V1_MOD_WNFACTOR :
            mod->BSIM3V1wnfactor = value->rValue;
            mod->BSIM3V1wnfactorGiven = TRUE;
            break;
        case BSIM3V1_MOD_WXJ:
            mod->BSIM3V1wxj = value->rValue;
            mod->BSIM3V1wxjGiven = TRUE;
            break;
        case BSIM3V1_MOD_WVSAT:
            mod->BSIM3V1wvsat = value->rValue;
            mod->BSIM3V1wvsatGiven = TRUE;
            break;


        case BSIM3V1_MOD_WA0:
            mod->BSIM3V1wa0 = value->rValue;
            mod->BSIM3V1wa0Given = TRUE;
            break;
        case BSIM3V1_MOD_WAGS:
            mod->BSIM3V1wags = value->rValue;
            mod->BSIM3V1wagsGiven = TRUE;
            break;
        case BSIM3V1_MOD_WA1:
            mod->BSIM3V1wa1 = value->rValue;
            mod->BSIM3V1wa1Given = TRUE;
            break;
        case BSIM3V1_MOD_WA2:
            mod->BSIM3V1wa2 = value->rValue;
            mod->BSIM3V1wa2Given = TRUE;
            break;
        case BSIM3V1_MOD_WAT:
            mod->BSIM3V1wat = value->rValue;
            mod->BSIM3V1watGiven = TRUE;
            break;
        case BSIM3V1_MOD_WKETA:
            mod->BSIM3V1wketa = value->rValue;
            mod->BSIM3V1wketaGiven = TRUE;
            break;    
        case BSIM3V1_MOD_WNSUB:
            mod->BSIM3V1wnsub = value->rValue;
            mod->BSIM3V1wnsubGiven = TRUE;
            break;
        case BSIM3V1_MOD_WNPEAK:
            mod->BSIM3V1wnpeak = value->rValue;
            mod->BSIM3V1wnpeakGiven = TRUE;
	    if (mod->BSIM3V1wnpeak > 1.0e20)
		mod->BSIM3V1wnpeak *= 1.0e-6;
            break;
        case BSIM3V1_MOD_WNGATE:
            mod->BSIM3V1wngate = value->rValue;
            mod->BSIM3V1wngateGiven = TRUE;
	    if (mod->BSIM3V1wngate > 1.0e23)
		mod->BSIM3V1wngate *= 1.0e-6;
            break;
        case BSIM3V1_MOD_WGAMMA1:
            mod->BSIM3V1wgamma1 = value->rValue;
            mod->BSIM3V1wgamma1Given = TRUE;
            break;
        case BSIM3V1_MOD_WGAMMA2:
            mod->BSIM3V1wgamma2 = value->rValue;
            mod->BSIM3V1wgamma2Given = TRUE;
            break;
        case BSIM3V1_MOD_WVBX:
            mod->BSIM3V1wvbx = value->rValue;
            mod->BSIM3V1wvbxGiven = TRUE;
            break;
        case BSIM3V1_MOD_WVBM:
            mod->BSIM3V1wvbm = value->rValue;
            mod->BSIM3V1wvbmGiven = TRUE;
            break;
        case BSIM3V1_MOD_WXT:
            mod->BSIM3V1wxt = value->rValue;
            mod->BSIM3V1wxtGiven = TRUE;
            break;
        case  BSIM3V1_MOD_WK1:
            mod->BSIM3V1wk1 = value->rValue;
            mod->BSIM3V1wk1Given = TRUE;
            break;
        case  BSIM3V1_MOD_WKT1:
            mod->BSIM3V1wkt1 = value->rValue;
            mod->BSIM3V1wkt1Given = TRUE;
            break;
        case  BSIM3V1_MOD_WKT1L:
            mod->BSIM3V1wkt1l = value->rValue;
            mod->BSIM3V1wkt1lGiven = TRUE;
            break;
        case  BSIM3V1_MOD_WKT2:
            mod->BSIM3V1wkt2 = value->rValue;
            mod->BSIM3V1wkt2Given = TRUE;
            break;
        case  BSIM3V1_MOD_WK2:
            mod->BSIM3V1wk2 = value->rValue;
            mod->BSIM3V1wk2Given = TRUE;
            break;
        case  BSIM3V1_MOD_WK3:
            mod->BSIM3V1wk3 = value->rValue;
            mod->BSIM3V1wk3Given = TRUE;
            break;
        case  BSIM3V1_MOD_WK3B:
            mod->BSIM3V1wk3b = value->rValue;
            mod->BSIM3V1wk3bGiven = TRUE;
            break;
        case  BSIM3V1_MOD_WNLX:
            mod->BSIM3V1wnlx = value->rValue;
            mod->BSIM3V1wnlxGiven = TRUE;
            break;
        case  BSIM3V1_MOD_WW0:
            mod->BSIM3V1ww0 = value->rValue;
            mod->BSIM3V1ww0Given = TRUE;
            break;
        case  BSIM3V1_MOD_WDVT0:               
            mod->BSIM3V1wdvt0 = value->rValue;
            mod->BSIM3V1wdvt0Given = TRUE;
            break;
        case  BSIM3V1_MOD_WDVT1:             
            mod->BSIM3V1wdvt1 = value->rValue;
            mod->BSIM3V1wdvt1Given = TRUE;
            break;
        case  BSIM3V1_MOD_WDVT2:             
            mod->BSIM3V1wdvt2 = value->rValue;
            mod->BSIM3V1wdvt2Given = TRUE;
            break;
        case  BSIM3V1_MOD_WDVT0W:               
            mod->BSIM3V1wdvt0w = value->rValue;
            mod->BSIM3V1wdvt0wGiven = TRUE;
            break;
        case  BSIM3V1_MOD_WDVT1W:             
            mod->BSIM3V1wdvt1w = value->rValue;
            mod->BSIM3V1wdvt1wGiven = TRUE;
            break;
        case  BSIM3V1_MOD_WDVT2W:             
            mod->BSIM3V1wdvt2w = value->rValue;
            mod->BSIM3V1wdvt2wGiven = TRUE;
            break;
        case  BSIM3V1_MOD_WDROUT:             
            mod->BSIM3V1wdrout = value->rValue;
            mod->BSIM3V1wdroutGiven = TRUE;
            break;
        case  BSIM3V1_MOD_WDSUB:             
            mod->BSIM3V1wdsub = value->rValue;
            mod->BSIM3V1wdsubGiven = TRUE;
            break;
        case BSIM3V1_MOD_WVTH0:
            mod->BSIM3V1wvth0 = value->rValue;
            mod->BSIM3V1wvth0Given = TRUE;
            break;
        case BSIM3V1_MOD_WUA:
            mod->BSIM3V1wua = value->rValue;
            mod->BSIM3V1wuaGiven = TRUE;
            break;
        case BSIM3V1_MOD_WUA1:
            mod->BSIM3V1wua1 = value->rValue;
            mod->BSIM3V1wua1Given = TRUE;
            break;
        case BSIM3V1_MOD_WUB:
            mod->BSIM3V1wub = value->rValue;
            mod->BSIM3V1wubGiven = TRUE;
            break;
        case BSIM3V1_MOD_WUB1:
            mod->BSIM3V1wub1 = value->rValue;
            mod->BSIM3V1wub1Given = TRUE;
            break;
        case BSIM3V1_MOD_WUC:
            mod->BSIM3V1wuc = value->rValue;
            mod->BSIM3V1wucGiven = TRUE;
            break;
        case BSIM3V1_MOD_WUC1:
            mod->BSIM3V1wuc1 = value->rValue;
            mod->BSIM3V1wuc1Given = TRUE;
            break;
        case  BSIM3V1_MOD_WU0 :
            mod->BSIM3V1wu0 = value->rValue;
            mod->BSIM3V1wu0Given = TRUE;
            break;
        case  BSIM3V1_MOD_WUTE :
            mod->BSIM3V1wute = value->rValue;
            mod->BSIM3V1wuteGiven = TRUE;
            break;
        case BSIM3V1_MOD_WVOFF:
            mod->BSIM3V1wvoff = value->rValue;
            mod->BSIM3V1wvoffGiven = TRUE;
            break;
        case  BSIM3V1_MOD_WDELTA :
            mod->BSIM3V1wdelta = value->rValue;
            mod->BSIM3V1wdeltaGiven = TRUE;
            break;
        case BSIM3V1_MOD_WRDSW:
            mod->BSIM3V1wrdsw = value->rValue;
            mod->BSIM3V1wrdswGiven = TRUE;
            break;                     
        case BSIM3V1_MOD_WPRWB:
            mod->BSIM3V1wprwb = value->rValue;
            mod->BSIM3V1wprwbGiven = TRUE;
            break;                     
        case BSIM3V1_MOD_WPRWG:
            mod->BSIM3V1wprwg = value->rValue;
            mod->BSIM3V1wprwgGiven = TRUE;
            break;                     
        case BSIM3V1_MOD_WPRT:
            mod->BSIM3V1wprt = value->rValue;
            mod->BSIM3V1wprtGiven = TRUE;
            break;                     
        case BSIM3V1_MOD_WETA0:
            mod->BSIM3V1weta0 = value->rValue;
            mod->BSIM3V1weta0Given = TRUE;
            break;                 
        case BSIM3V1_MOD_WETAB:
            mod->BSIM3V1wetab = value->rValue;
            mod->BSIM3V1wetabGiven = TRUE;
            break;                 
        case BSIM3V1_MOD_WPCLM:
            mod->BSIM3V1wpclm = value->rValue;
            mod->BSIM3V1wpclmGiven = TRUE;
            break;                 
        case BSIM3V1_MOD_WPDIBL1:
            mod->BSIM3V1wpdibl1 = value->rValue;
            mod->BSIM3V1wpdibl1Given = TRUE;
            break;                 
        case BSIM3V1_MOD_WPDIBL2:
            mod->BSIM3V1wpdibl2 = value->rValue;
            mod->BSIM3V1wpdibl2Given = TRUE;
            break;                 
        case BSIM3V1_MOD_WPDIBLB:
            mod->BSIM3V1wpdiblb = value->rValue;
            mod->BSIM3V1wpdiblbGiven = TRUE;
            break;                 
        case BSIM3V1_MOD_WPSCBE1:
            mod->BSIM3V1wpscbe1 = value->rValue;
            mod->BSIM3V1wpscbe1Given = TRUE;
            break;                 
        case BSIM3V1_MOD_WPSCBE2:
            mod->BSIM3V1wpscbe2 = value->rValue;
            mod->BSIM3V1wpscbe2Given = TRUE;
            break;                 
        case BSIM3V1_MOD_WPVAG:
            mod->BSIM3V1wpvag = value->rValue;
            mod->BSIM3V1wpvagGiven = TRUE;
            break;                 
        case  BSIM3V1_MOD_WWR :
            mod->BSIM3V1wwr = value->rValue;
            mod->BSIM3V1wwrGiven = TRUE;
            break;
        case  BSIM3V1_MOD_WDWG :
            mod->BSIM3V1wdwg = value->rValue;
            mod->BSIM3V1wdwgGiven = TRUE;
            break;
        case  BSIM3V1_MOD_WDWB :
            mod->BSIM3V1wdwb = value->rValue;
            mod->BSIM3V1wdwbGiven = TRUE;
            break;
        case  BSIM3V1_MOD_WB0 :
            mod->BSIM3V1wb0 = value->rValue;
            mod->BSIM3V1wb0Given = TRUE;
            break;
        case  BSIM3V1_MOD_WB1 :
            mod->BSIM3V1wb1 = value->rValue;
            mod->BSIM3V1wb1Given = TRUE;
            break;
        case  BSIM3V1_MOD_WALPHA0 :
            mod->BSIM3V1walpha0 = value->rValue;
            mod->BSIM3V1walpha0Given = TRUE;
            break;
        case  BSIM3V1_MOD_WBETA0 :
            mod->BSIM3V1wbeta0 = value->rValue;
            mod->BSIM3V1wbeta0Given = TRUE;
            break;

        case  BSIM3V1_MOD_WELM :
            mod->BSIM3V1welm = value->rValue;
            mod->BSIM3V1welmGiven = TRUE;
            break;
        case  BSIM3V1_MOD_WCGSL :
            mod->BSIM3V1wcgsl = value->rValue;
            mod->BSIM3V1wcgslGiven = TRUE;
            break;
        case  BSIM3V1_MOD_WCGDL :
            mod->BSIM3V1wcgdl = value->rValue;
            mod->BSIM3V1wcgdlGiven = TRUE;
            break;
        case  BSIM3V1_MOD_WCKAPPA :
            mod->BSIM3V1wckappa = value->rValue;
            mod->BSIM3V1wckappaGiven = TRUE;
            break;
        case  BSIM3V1_MOD_WCF :
            mod->BSIM3V1wcf = value->rValue;
            mod->BSIM3V1wcfGiven = TRUE;
            break;
        case  BSIM3V1_MOD_WCLC :
            mod->BSIM3V1wclc = value->rValue;
            mod->BSIM3V1wclcGiven = TRUE;
            break;
        case  BSIM3V1_MOD_WCLE :
            mod->BSIM3V1wcle = value->rValue;
            mod->BSIM3V1wcleGiven = TRUE;
            break;
        case  BSIM3V1_MOD_WVFBCV :
            mod->BSIM3V1wvfbcv = value->rValue;
            mod->BSIM3V1wvfbcvGiven = TRUE;
            break;

	/* Cross-term dependence */
        case  BSIM3V1_MOD_PCDSC :
            mod->BSIM3V1pcdsc = value->rValue;
            mod->BSIM3V1pcdscGiven = TRUE;
            break;


        case  BSIM3V1_MOD_PCDSCB :
            mod->BSIM3V1pcdscb = value->rValue;
            mod->BSIM3V1pcdscbGiven = TRUE;
            break;
        case  BSIM3V1_MOD_PCDSCD :
            mod->BSIM3V1pcdscd = value->rValue;
            mod->BSIM3V1pcdscdGiven = TRUE;
            break;
        case  BSIM3V1_MOD_PCIT :
            mod->BSIM3V1pcit = value->rValue;
            mod->BSIM3V1pcitGiven = TRUE;
            break;
        case  BSIM3V1_MOD_PNFACTOR :
            mod->BSIM3V1pnfactor = value->rValue;
            mod->BSIM3V1pnfactorGiven = TRUE;
            break;
        case BSIM3V1_MOD_PXJ:
            mod->BSIM3V1pxj = value->rValue;
            mod->BSIM3V1pxjGiven = TRUE;
            break;
        case BSIM3V1_MOD_PVSAT:
            mod->BSIM3V1pvsat = value->rValue;
            mod->BSIM3V1pvsatGiven = TRUE;
            break;


        case BSIM3V1_MOD_PA0:
            mod->BSIM3V1pa0 = value->rValue;
            mod->BSIM3V1pa0Given = TRUE;
            break;
        case BSIM3V1_MOD_PAGS:
            mod->BSIM3V1pags = value->rValue;
            mod->BSIM3V1pagsGiven = TRUE;
            break;
        case BSIM3V1_MOD_PA1:
            mod->BSIM3V1pa1 = value->rValue;
            mod->BSIM3V1pa1Given = TRUE;
            break;
        case BSIM3V1_MOD_PA2:
            mod->BSIM3V1pa2 = value->rValue;
            mod->BSIM3V1pa2Given = TRUE;
            break;
        case BSIM3V1_MOD_PAT:
            mod->BSIM3V1pat = value->rValue;
            mod->BSIM3V1patGiven = TRUE;
            break;
        case BSIM3V1_MOD_PKETA:
            mod->BSIM3V1pketa = value->rValue;
            mod->BSIM3V1pketaGiven = TRUE;
            break;    
        case BSIM3V1_MOD_PNSUB:
            mod->BSIM3V1pnsub = value->rValue;
            mod->BSIM3V1pnsubGiven = TRUE;
            break;
        case BSIM3V1_MOD_PNPEAK:
            mod->BSIM3V1pnpeak = value->rValue;
            mod->BSIM3V1pnpeakGiven = TRUE;
	    if (mod->BSIM3V1pnpeak > 1.0e20)
		mod->BSIM3V1pnpeak *= 1.0e-6;
            break;
        case BSIM3V1_MOD_PNGATE:
            mod->BSIM3V1pngate = value->rValue;
            mod->BSIM3V1pngateGiven = TRUE;
	    if (mod->BSIM3V1pngate > 1.0e23)
		mod->BSIM3V1pngate *= 1.0e-6;
            break;
        case BSIM3V1_MOD_PGAMMA1:
            mod->BSIM3V1pgamma1 = value->rValue;
            mod->BSIM3V1pgamma1Given = TRUE;
            break;
        case BSIM3V1_MOD_PGAMMA2:
            mod->BSIM3V1pgamma2 = value->rValue;
            mod->BSIM3V1pgamma2Given = TRUE;
            break;
        case BSIM3V1_MOD_PVBX:
            mod->BSIM3V1pvbx = value->rValue;
            mod->BSIM3V1pvbxGiven = TRUE;
            break;
        case BSIM3V1_MOD_PVBM:
            mod->BSIM3V1pvbm = value->rValue;
            mod->BSIM3V1pvbmGiven = TRUE;
            break;
        case BSIM3V1_MOD_PXT:
            mod->BSIM3V1pxt = value->rValue;
            mod->BSIM3V1pxtGiven = TRUE;
            break;
        case  BSIM3V1_MOD_PK1:
            mod->BSIM3V1pk1 = value->rValue;
            mod->BSIM3V1pk1Given = TRUE;
            break;
        case  BSIM3V1_MOD_PKT1:
            mod->BSIM3V1pkt1 = value->rValue;
            mod->BSIM3V1pkt1Given = TRUE;
            break;
        case  BSIM3V1_MOD_PKT1L:
            mod->BSIM3V1pkt1l = value->rValue;
            mod->BSIM3V1pkt1lGiven = TRUE;
            break;
        case  BSIM3V1_MOD_PKT2:
            mod->BSIM3V1pkt2 = value->rValue;
            mod->BSIM3V1pkt2Given = TRUE;
            break;
        case  BSIM3V1_MOD_PK2:
            mod->BSIM3V1pk2 = value->rValue;
            mod->BSIM3V1pk2Given = TRUE;
            break;
        case  BSIM3V1_MOD_PK3:
            mod->BSIM3V1pk3 = value->rValue;
            mod->BSIM3V1pk3Given = TRUE;
            break;
        case  BSIM3V1_MOD_PK3B:
            mod->BSIM3V1pk3b = value->rValue;
            mod->BSIM3V1pk3bGiven = TRUE;
            break;
        case  BSIM3V1_MOD_PNLX:
            mod->BSIM3V1pnlx = value->rValue;
            mod->BSIM3V1pnlxGiven = TRUE;
            break;
        case  BSIM3V1_MOD_PW0:
            mod->BSIM3V1pw0 = value->rValue;
            mod->BSIM3V1pw0Given = TRUE;
            break;
        case  BSIM3V1_MOD_PDVT0:               
            mod->BSIM3V1pdvt0 = value->rValue;
            mod->BSIM3V1pdvt0Given = TRUE;
            break;
        case  BSIM3V1_MOD_PDVT1:             
            mod->BSIM3V1pdvt1 = value->rValue;
            mod->BSIM3V1pdvt1Given = TRUE;
            break;
        case  BSIM3V1_MOD_PDVT2:             
            mod->BSIM3V1pdvt2 = value->rValue;
            mod->BSIM3V1pdvt2Given = TRUE;
            break;
        case  BSIM3V1_MOD_PDVT0W:               
            mod->BSIM3V1pdvt0w = value->rValue;
            mod->BSIM3V1pdvt0wGiven = TRUE;
            break;
        case  BSIM3V1_MOD_PDVT1W:             
            mod->BSIM3V1pdvt1w = value->rValue;
            mod->BSIM3V1pdvt1wGiven = TRUE;
            break;
        case  BSIM3V1_MOD_PDVT2W:             
            mod->BSIM3V1pdvt2w = value->rValue;
            mod->BSIM3V1pdvt2wGiven = TRUE;
            break;
        case  BSIM3V1_MOD_PDROUT:             
            mod->BSIM3V1pdrout = value->rValue;
            mod->BSIM3V1pdroutGiven = TRUE;
            break;
        case  BSIM3V1_MOD_PDSUB:             
            mod->BSIM3V1pdsub = value->rValue;
            mod->BSIM3V1pdsubGiven = TRUE;
            break;
        case BSIM3V1_MOD_PVTH0:
            mod->BSIM3V1pvth0 = value->rValue;
            mod->BSIM3V1pvth0Given = TRUE;
            break;
        case BSIM3V1_MOD_PUA:
            mod->BSIM3V1pua = value->rValue;
            mod->BSIM3V1puaGiven = TRUE;
            break;
        case BSIM3V1_MOD_PUA1:
            mod->BSIM3V1pua1 = value->rValue;
            mod->BSIM3V1pua1Given = TRUE;
            break;
        case BSIM3V1_MOD_PUB:
            mod->BSIM3V1pub = value->rValue;
            mod->BSIM3V1pubGiven = TRUE;
            break;
        case BSIM3V1_MOD_PUB1:
            mod->BSIM3V1pub1 = value->rValue;
            mod->BSIM3V1pub1Given = TRUE;
            break;
        case BSIM3V1_MOD_PUC:
            mod->BSIM3V1puc = value->rValue;
            mod->BSIM3V1pucGiven = TRUE;
            break;
        case BSIM3V1_MOD_PUC1:
            mod->BSIM3V1puc1 = value->rValue;
            mod->BSIM3V1puc1Given = TRUE;
            break;
        case  BSIM3V1_MOD_PU0 :
            mod->BSIM3V1pu0 = value->rValue;
            mod->BSIM3V1pu0Given = TRUE;
            break;
        case  BSIM3V1_MOD_PUTE :
            mod->BSIM3V1pute = value->rValue;
            mod->BSIM3V1puteGiven = TRUE;
            break;
        case BSIM3V1_MOD_PVOFF:
            mod->BSIM3V1pvoff = value->rValue;
            mod->BSIM3V1pvoffGiven = TRUE;
            break;
        case  BSIM3V1_MOD_PDELTA :
            mod->BSIM3V1pdelta = value->rValue;
            mod->BSIM3V1pdeltaGiven = TRUE;
            break;
        case BSIM3V1_MOD_PRDSW:
            mod->BSIM3V1prdsw = value->rValue;
            mod->BSIM3V1prdswGiven = TRUE;
            break;                     
        case BSIM3V1_MOD_PPRWB:
            mod->BSIM3V1pprwb = value->rValue;
            mod->BSIM3V1pprwbGiven = TRUE;
            break;                     
        case BSIM3V1_MOD_PPRWG:
            mod->BSIM3V1pprwg = value->rValue;
            mod->BSIM3V1pprwgGiven = TRUE;
            break;                     
        case BSIM3V1_MOD_PPRT:
            mod->BSIM3V1pprt = value->rValue;
            mod->BSIM3V1pprtGiven = TRUE;
            break;                     
        case BSIM3V1_MOD_PETA0:
            mod->BSIM3V1peta0 = value->rValue;
            mod->BSIM3V1peta0Given = TRUE;
            break;                 
        case BSIM3V1_MOD_PETAB:
            mod->BSIM3V1petab = value->rValue;
            mod->BSIM3V1petabGiven = TRUE;
            break;                 
        case BSIM3V1_MOD_PPCLM:
            mod->BSIM3V1ppclm = value->rValue;
            mod->BSIM3V1ppclmGiven = TRUE;
            break;                 
        case BSIM3V1_MOD_PPDIBL1:
            mod->BSIM3V1ppdibl1 = value->rValue;
            mod->BSIM3V1ppdibl1Given = TRUE;
            break;                 
        case BSIM3V1_MOD_PPDIBL2:
            mod->BSIM3V1ppdibl2 = value->rValue;
            mod->BSIM3V1ppdibl2Given = TRUE;
            break;                 
        case BSIM3V1_MOD_PPDIBLB:
            mod->BSIM3V1ppdiblb = value->rValue;
            mod->BSIM3V1ppdiblbGiven = TRUE;
            break;                 
        case BSIM3V1_MOD_PPSCBE1:
            mod->BSIM3V1ppscbe1 = value->rValue;
            mod->BSIM3V1ppscbe1Given = TRUE;
            break;                 
        case BSIM3V1_MOD_PPSCBE2:
            mod->BSIM3V1ppscbe2 = value->rValue;
            mod->BSIM3V1ppscbe2Given = TRUE;
            break;                 
        case BSIM3V1_MOD_PPVAG:
            mod->BSIM3V1ppvag = value->rValue;
            mod->BSIM3V1ppvagGiven = TRUE;
            break;                 
        case  BSIM3V1_MOD_PWR :
            mod->BSIM3V1pwr = value->rValue;
            mod->BSIM3V1pwrGiven = TRUE;
            break;
        case  BSIM3V1_MOD_PDWG :
            mod->BSIM3V1pdwg = value->rValue;
            mod->BSIM3V1pdwgGiven = TRUE;
            break;
        case  BSIM3V1_MOD_PDWB :
            mod->BSIM3V1pdwb = value->rValue;
            mod->BSIM3V1pdwbGiven = TRUE;
            break;
        case  BSIM3V1_MOD_PB0 :
            mod->BSIM3V1pb0 = value->rValue;
            mod->BSIM3V1pb0Given = TRUE;
            break;
        case  BSIM3V1_MOD_PB1 :
            mod->BSIM3V1pb1 = value->rValue;
            mod->BSIM3V1pb1Given = TRUE;
            break;
        case  BSIM3V1_MOD_PALPHA0 :
            mod->BSIM3V1palpha0 = value->rValue;
            mod->BSIM3V1palpha0Given = TRUE;
            break;
        case  BSIM3V1_MOD_PBETA0 :
            mod->BSIM3V1pbeta0 = value->rValue;
            mod->BSIM3V1pbeta0Given = TRUE;
            break;

        case  BSIM3V1_MOD_PELM :
            mod->BSIM3V1pelm = value->rValue;
            mod->BSIM3V1pelmGiven = TRUE;
            break;
        case  BSIM3V1_MOD_PCGSL :
            mod->BSIM3V1pcgsl = value->rValue;
            mod->BSIM3V1pcgslGiven = TRUE;
            break;
        case  BSIM3V1_MOD_PCGDL :
            mod->BSIM3V1pcgdl = value->rValue;
            mod->BSIM3V1pcgdlGiven = TRUE;
            break;
        case  BSIM3V1_MOD_PCKAPPA :
            mod->BSIM3V1pckappa = value->rValue;
            mod->BSIM3V1pckappaGiven = TRUE;
            break;
        case  BSIM3V1_MOD_PCF :
            mod->BSIM3V1pcf = value->rValue;
            mod->BSIM3V1pcfGiven = TRUE;
            break;
        case  BSIM3V1_MOD_PCLC :
            mod->BSIM3V1pclc = value->rValue;
            mod->BSIM3V1pclcGiven = TRUE;
            break;
        case  BSIM3V1_MOD_PCLE :
            mod->BSIM3V1pcle = value->rValue;
            mod->BSIM3V1pcleGiven = TRUE;
            break;
        case  BSIM3V1_MOD_PVFBCV :
            mod->BSIM3V1pvfbcv = value->rValue;
            mod->BSIM3V1pvfbcvGiven = TRUE;
            break;

        case  BSIM3V1_MOD_TNOM :
            mod->BSIM3V1tnom = value->rValue + 273.15;
            mod->BSIM3V1tnomGiven = TRUE;
            break;
        case  BSIM3V1_MOD_CGSO :
            mod->BSIM3V1cgso = value->rValue;
            mod->BSIM3V1cgsoGiven = TRUE;
            break;
        case  BSIM3V1_MOD_CGDO :
            mod->BSIM3V1cgdo = value->rValue;
            mod->BSIM3V1cgdoGiven = TRUE;
            break;
        case  BSIM3V1_MOD_CGBO :
            mod->BSIM3V1cgbo = value->rValue;
            mod->BSIM3V1cgboGiven = TRUE;
            break;
        case  BSIM3V1_MOD_XPART :
            mod->BSIM3V1xpart = value->rValue;
            mod->BSIM3V1xpartGiven = TRUE;
            break;
        case  BSIM3V1_MOD_RSH :
            mod->BSIM3V1sheetResistance = value->rValue;
            mod->BSIM3V1sheetResistanceGiven = TRUE;
            break;
        case  BSIM3V1_MOD_JS :
            mod->BSIM3V1jctSatCurDensity = value->rValue;
            mod->BSIM3V1jctSatCurDensityGiven = TRUE;
            break;
        case  BSIM3V1_MOD_JSW :
            mod->BSIM3V1jctSidewallSatCurDensity = value->rValue;
            mod->BSIM3V1jctSidewallSatCurDensityGiven = TRUE;
            break;
        case  BSIM3V1_MOD_PB :
            mod->BSIM3V1bulkJctPotential = value->rValue;
            mod->BSIM3V1bulkJctPotentialGiven = TRUE;
            break;
        case  BSIM3V1_MOD_MJ :
            mod->BSIM3V1bulkJctBotGradingCoeff = value->rValue;
            mod->BSIM3V1bulkJctBotGradingCoeffGiven = TRUE;
            break;
        case  BSIM3V1_MOD_PBSW :
            mod->BSIM3V1sidewallJctPotential = value->rValue;
            mod->BSIM3V1sidewallJctPotentialGiven = TRUE;
            break;
        case  BSIM3V1_MOD_MJSW :
            mod->BSIM3V1bulkJctSideGradingCoeff = value->rValue;
            mod->BSIM3V1bulkJctSideGradingCoeffGiven = TRUE;
            break;
        case  BSIM3V1_MOD_CJ :
            mod->BSIM3V1unitAreaJctCap = value->rValue;
            mod->BSIM3V1unitAreaJctCapGiven = TRUE;
            break;
        case  BSIM3V1_MOD_CJSW :
            mod->BSIM3V1unitLengthSidewallJctCap = value->rValue;
            mod->BSIM3V1unitLengthSidewallJctCapGiven = TRUE;
            break;
        case  BSIM3V1_MOD_NJ :
            mod->BSIM3V1jctEmissionCoeff = value->rValue;
            mod->BSIM3V1jctEmissionCoeffGiven = TRUE;
            break;
        case  BSIM3V1_MOD_PBSWG :
            mod->BSIM3V1GatesidewallJctPotential = value->rValue;
            mod->BSIM3V1GatesidewallJctPotentialGiven = TRUE;
            break;
        case  BSIM3V1_MOD_MJSWG :
            mod->BSIM3V1bulkJctGateSideGradingCoeff = value->rValue;
            mod->BSIM3V1bulkJctGateSideGradingCoeffGiven = TRUE;
            break;
        case  BSIM3V1_MOD_CJSWG :
            mod->BSIM3V1unitLengthGateSidewallJctCap = value->rValue;
            mod->BSIM3V1unitLengthGateSidewallJctCapGiven = TRUE;
            break;
        case  BSIM3V1_MOD_XTI :
            mod->BSIM3V1jctTempExponent = value->rValue;
            mod->BSIM3V1jctTempExponentGiven = TRUE;
            break;
        case  BSIM3V1_MOD_LINT :
            mod->BSIM3V1Lint = value->rValue;
            mod->BSIM3V1LintGiven = TRUE;
            break;
        case  BSIM3V1_MOD_LL :
            mod->BSIM3V1Ll = value->rValue;
            mod->BSIM3V1LlGiven = TRUE;
            break;
        case  BSIM3V1_MOD_LLN :
            mod->BSIM3V1Lln = value->rValue;
            mod->BSIM3V1LlnGiven = TRUE;
            break;
        case  BSIM3V1_MOD_LW :
            mod->BSIM3V1Lw = value->rValue;
            mod->BSIM3V1LwGiven = TRUE;
            break;
        case  BSIM3V1_MOD_LWN :
            mod->BSIM3V1Lwn = value->rValue;
            mod->BSIM3V1LwnGiven = TRUE;
            break;
        case  BSIM3V1_MOD_LWL :
            mod->BSIM3V1Lwl = value->rValue;
            mod->BSIM3V1LwlGiven = TRUE;
            break;
        case  BSIM3V1_MOD_LMIN :
            mod->BSIM3V1Lmin = value->rValue;
            mod->BSIM3V1LminGiven = TRUE;
            break;
        case  BSIM3V1_MOD_LMAX :
            mod->BSIM3V1Lmax = value->rValue;
            mod->BSIM3V1LmaxGiven = TRUE;
            break;
        case  BSIM3V1_MOD_WINT :
            mod->BSIM3V1Wint = value->rValue;
            mod->BSIM3V1WintGiven = TRUE;
            break;
        case  BSIM3V1_MOD_WL :
            mod->BSIM3V1Wl = value->rValue;
            mod->BSIM3V1WlGiven = TRUE;
            break;
        case  BSIM3V1_MOD_WLN :
            mod->BSIM3V1Wln = value->rValue;
            mod->BSIM3V1WlnGiven = TRUE;
            break;
        case  BSIM3V1_MOD_WW :
            mod->BSIM3V1Ww = value->rValue;
            mod->BSIM3V1WwGiven = TRUE;
            break;
        case  BSIM3V1_MOD_WWN :
            mod->BSIM3V1Wwn = value->rValue;
            mod->BSIM3V1WwnGiven = TRUE;
            break;
        case  BSIM3V1_MOD_WWL :
            mod->BSIM3V1Wwl = value->rValue;
            mod->BSIM3V1WwlGiven = TRUE;
            break;
        case  BSIM3V1_MOD_WMIN :
            mod->BSIM3V1Wmin = value->rValue;
            mod->BSIM3V1WminGiven = TRUE;
            break;
        case  BSIM3V1_MOD_WMAX :
            mod->BSIM3V1Wmax = value->rValue;
            mod->BSIM3V1WmaxGiven = TRUE;
            break;

        case  BSIM3V1_MOD_NOIA :
            mod->BSIM3V1oxideTrapDensityA = value->rValue;
            mod->BSIM3V1oxideTrapDensityAGiven = TRUE;
            break;
        case  BSIM3V1_MOD_NOIB :
            mod->BSIM3V1oxideTrapDensityB = value->rValue;
            mod->BSIM3V1oxideTrapDensityBGiven = TRUE;
            break;
        case  BSIM3V1_MOD_NOIC :
            mod->BSIM3V1oxideTrapDensityC = value->rValue;
            mod->BSIM3V1oxideTrapDensityCGiven = TRUE;
            break;
        case  BSIM3V1_MOD_EM :
            mod->BSIM3V1em = value->rValue;
            mod->BSIM3V1emGiven = TRUE;
            break;
        case  BSIM3V1_MOD_EF :
            mod->BSIM3V1ef = value->rValue;
            mod->BSIM3V1efGiven = TRUE;
            break;
        case  BSIM3V1_MOD_AF :
            mod->BSIM3V1af = value->rValue;
            mod->BSIM3V1afGiven = TRUE;
            break;
        case  BSIM3V1_MOD_KF :
            mod->BSIM3V1kf = value->rValue;
            mod->BSIM3V1kfGiven = TRUE;
            break;
        case  BSIM3V1_MOD_NMOS  :
            if(value->iValue) {
                mod->BSIM3V1type = 1;
                mod->BSIM3V1typeGiven = TRUE;
            }
            break;
        case  BSIM3V1_MOD_PMOS  :
            if(value->iValue) {
                mod->BSIM3V1type = - 1;
                mod->BSIM3V1typeGiven = TRUE;
            }
            break;
/* serban */
        case  BSIM3V1_MOD_HDIF  :
            mod->BSIM3V1hdif = value->rValue;
            mod->BSIM3V1hdifGiven = TRUE;
            break;
        default:
            return(E_BADPARM);
    }
    return(OK);
}


