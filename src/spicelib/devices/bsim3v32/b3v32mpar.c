/**** BSIM3v3.2.4, Released by Xuemei Xi 12/21/2001 ****/

/**********
 * Copyright 2001 Regents of the University of California. All rights reserved.
 * File: b3mpar.c of BSIM3v3.2.4
 * Author: 1995 Min-Chie Jeng and Mansun Chan.
 * Author: 1997-1999 Weidong Liu.
 * Author: 2001  Xuemei Xi
 * Modified by Paolo Nenzi 2002 and Dietmar Warning 2003
 **********/

#include "ngspice/ngspice.h"
#include "bsim3v32def.h"
#include "ngspice/ifsim.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
BSIM3v32mParam(int param, IFvalue *value, GENmodel *inMod)
{
    BSIM3v32model *mod = (BSIM3v32model*)inMod;
    switch(param)
    {   case  BSIM3v32_MOD_MOBMOD :
            mod->BSIM3v32mobMod = value->iValue;
            mod->BSIM3v32mobModGiven = TRUE;
            break;
        case  BSIM3v32_MOD_BINUNIT :
            mod->BSIM3v32binUnit = value->iValue;
            mod->BSIM3v32binUnitGiven = TRUE;
            break;
        case  BSIM3v32_MOD_PARAMCHK :
            mod->BSIM3v32paramChk = value->iValue;
            mod->BSIM3v32paramChkGiven = TRUE;
            break;
        case  BSIM3v32_MOD_CAPMOD :
            mod->BSIM3v32capMod = value->iValue;
            mod->BSIM3v32capModGiven = TRUE;
            break;
        case BSIM3v32_MOD_ACMMOD:
            mod->BSIM3v32acmMod = value->iValue;
            mod->BSIM3v32acmModGiven = TRUE;
            break;
        case BSIM3v32_MOD_CALCACM:
            mod->BSIM3v32calcacm = value->iValue;
            mod->BSIM3v32calcacmGiven = TRUE;
            break;
        case  BSIM3v32_MOD_NOIMOD :
            mod->BSIM3v32noiMod = value->iValue;
            mod->BSIM3v32noiModGiven = TRUE;
            break;
        case  BSIM3v32_MOD_NQSMOD :
            mod->BSIM3v32nqsMod = value->iValue;
            mod->BSIM3v32nqsModGiven = TRUE;
            break;
        case  BSIM3v32_MOD_VERSION :
            mod->BSIM3v32version = value->sValue;
            mod->BSIM3v32versionGiven = TRUE;
            break;
        case  BSIM3v32_MOD_TOX :
            mod->BSIM3v32tox = value->rValue;
            mod->BSIM3v32toxGiven = TRUE;
            break;
        case  BSIM3v32_MOD_TOXM :
            mod->BSIM3v32toxm = value->rValue;
            mod->BSIM3v32toxmGiven = TRUE;
            break;

        case  BSIM3v32_MOD_CDSC :
            mod->BSIM3v32cdsc = value->rValue;
            mod->BSIM3v32cdscGiven = TRUE;
            break;
        case  BSIM3v32_MOD_CDSCB :
            mod->BSIM3v32cdscb = value->rValue;
            mod->BSIM3v32cdscbGiven = TRUE;
            break;

        case  BSIM3v32_MOD_CDSCD :
            mod->BSIM3v32cdscd = value->rValue;
            mod->BSIM3v32cdscdGiven = TRUE;
            break;

        case  BSIM3v32_MOD_CIT :
            mod->BSIM3v32cit = value->rValue;
            mod->BSIM3v32citGiven = TRUE;
            break;
        case  BSIM3v32_MOD_NFACTOR :
            mod->BSIM3v32nfactor = value->rValue;
            mod->BSIM3v32nfactorGiven = TRUE;
            break;
        case BSIM3v32_MOD_XJ:
            mod->BSIM3v32xj = value->rValue;
            mod->BSIM3v32xjGiven = TRUE;
            break;
        case BSIM3v32_MOD_VSAT:
            mod->BSIM3v32vsat = value->rValue;
            mod->BSIM3v32vsatGiven = TRUE;
            break;
        case BSIM3v32_MOD_A0:
            mod->BSIM3v32a0 = value->rValue;
            mod->BSIM3v32a0Given = TRUE;
            break;

        case BSIM3v32_MOD_AGS:
            mod->BSIM3v32ags= value->rValue;
            mod->BSIM3v32agsGiven = TRUE;
            break;

        case BSIM3v32_MOD_A1:
            mod->BSIM3v32a1 = value->rValue;
            mod->BSIM3v32a1Given = TRUE;
            break;
        case BSIM3v32_MOD_A2:
            mod->BSIM3v32a2 = value->rValue;
            mod->BSIM3v32a2Given = TRUE;
            break;
        case BSIM3v32_MOD_AT:
            mod->BSIM3v32at = value->rValue;
            mod->BSIM3v32atGiven = TRUE;
            break;
        case BSIM3v32_MOD_KETA:
            mod->BSIM3v32keta = value->rValue;
            mod->BSIM3v32ketaGiven = TRUE;
            break;
        case BSIM3v32_MOD_NSUB:
            mod->BSIM3v32nsub = value->rValue;
            mod->BSIM3v32nsubGiven = TRUE;
            break;
        case BSIM3v32_MOD_NPEAK:
            mod->BSIM3v32npeak = value->rValue;
            mod->BSIM3v32npeakGiven = TRUE;
            if (mod->BSIM3v32npeak > 1.0e20)
                mod->BSIM3v32npeak *= 1.0e-6;
            break;
        case BSIM3v32_MOD_NGATE:
            mod->BSIM3v32ngate = value->rValue;
            mod->BSIM3v32ngateGiven = TRUE;
            if (mod->BSIM3v32ngate > 1.000001e24)
                mod->BSIM3v32ngate *= 1.0e-6;
            break;
        case BSIM3v32_MOD_GAMMA1:
            mod->BSIM3v32gamma1 = value->rValue;
            mod->BSIM3v32gamma1Given = TRUE;
            break;
        case BSIM3v32_MOD_GAMMA2:
            mod->BSIM3v32gamma2 = value->rValue;
            mod->BSIM3v32gamma2Given = TRUE;
            break;
        case BSIM3v32_MOD_VBX:
            mod->BSIM3v32vbx = value->rValue;
            mod->BSIM3v32vbxGiven = TRUE;
            break;
        case BSIM3v32_MOD_VBM:
            mod->BSIM3v32vbm = value->rValue;
            mod->BSIM3v32vbmGiven = TRUE;
            break;
        case BSIM3v32_MOD_XT:
            mod->BSIM3v32xt = value->rValue;
            mod->BSIM3v32xtGiven = TRUE;
            break;
        case  BSIM3v32_MOD_K1:
            mod->BSIM3v32k1 = value->rValue;
            mod->BSIM3v32k1Given = TRUE;
            break;
        case  BSIM3v32_MOD_KT1:
            mod->BSIM3v32kt1 = value->rValue;
            mod->BSIM3v32kt1Given = TRUE;
            break;
        case  BSIM3v32_MOD_KT1L:
            mod->BSIM3v32kt1l = value->rValue;
            mod->BSIM3v32kt1lGiven = TRUE;
            break;
        case  BSIM3v32_MOD_KT2:
            mod->BSIM3v32kt2 = value->rValue;
            mod->BSIM3v32kt2Given = TRUE;
            break;
        case  BSIM3v32_MOD_K2:
            mod->BSIM3v32k2 = value->rValue;
            mod->BSIM3v32k2Given = TRUE;
            break;
        case  BSIM3v32_MOD_K3:
            mod->BSIM3v32k3 = value->rValue;
            mod->BSIM3v32k3Given = TRUE;
            break;
        case  BSIM3v32_MOD_K3B:
            mod->BSIM3v32k3b = value->rValue;
            mod->BSIM3v32k3bGiven = TRUE;
            break;
        case  BSIM3v32_MOD_NLX:
            mod->BSIM3v32nlx = value->rValue;
            mod->BSIM3v32nlxGiven = TRUE;
            break;
        case  BSIM3v32_MOD_W0:
            mod->BSIM3v32w0 = value->rValue;
            mod->BSIM3v32w0Given = TRUE;
            break;
        case  BSIM3v32_MOD_DVT0:
            mod->BSIM3v32dvt0 = value->rValue;
            mod->BSIM3v32dvt0Given = TRUE;
            break;
        case  BSIM3v32_MOD_DVT1:
            mod->BSIM3v32dvt1 = value->rValue;
            mod->BSIM3v32dvt1Given = TRUE;
            break;
        case  BSIM3v32_MOD_DVT2:
            mod->BSIM3v32dvt2 = value->rValue;
            mod->BSIM3v32dvt2Given = TRUE;
            break;
        case  BSIM3v32_MOD_DVT0W:
            mod->BSIM3v32dvt0w = value->rValue;
            mod->BSIM3v32dvt0wGiven = TRUE;
            break;
        case  BSIM3v32_MOD_DVT1W:
            mod->BSIM3v32dvt1w = value->rValue;
            mod->BSIM3v32dvt1wGiven = TRUE;
            break;
        case  BSIM3v32_MOD_DVT2W:
            mod->BSIM3v32dvt2w = value->rValue;
            mod->BSIM3v32dvt2wGiven = TRUE;
            break;
        case  BSIM3v32_MOD_DROUT:
            mod->BSIM3v32drout = value->rValue;
            mod->BSIM3v32droutGiven = TRUE;
            break;
        case  BSIM3v32_MOD_DSUB:
            mod->BSIM3v32dsub = value->rValue;
            mod->BSIM3v32dsubGiven = TRUE;
            break;
        case BSIM3v32_MOD_VTH0:
            mod->BSIM3v32vth0 = value->rValue;
            mod->BSIM3v32vth0Given = TRUE;
            break;
        case BSIM3v32_MOD_UA:
            mod->BSIM3v32ua = value->rValue;
            mod->BSIM3v32uaGiven = TRUE;
            break;
        case BSIM3v32_MOD_UA1:
            mod->BSIM3v32ua1 = value->rValue;
            mod->BSIM3v32ua1Given = TRUE;
            break;
        case BSIM3v32_MOD_UB:
            mod->BSIM3v32ub = value->rValue;
            mod->BSIM3v32ubGiven = TRUE;
            break;
        case BSIM3v32_MOD_UB1:
            mod->BSIM3v32ub1 = value->rValue;
            mod->BSIM3v32ub1Given = TRUE;
            break;
        case BSIM3v32_MOD_UC:
            mod->BSIM3v32uc = value->rValue;
            mod->BSIM3v32ucGiven = TRUE;
            break;
        case BSIM3v32_MOD_UC1:
            mod->BSIM3v32uc1 = value->rValue;
            mod->BSIM3v32uc1Given = TRUE;
            break;
        case  BSIM3v32_MOD_U0 :
            mod->BSIM3v32u0 = value->rValue;
            mod->BSIM3v32u0Given = TRUE;
            break;
        case  BSIM3v32_MOD_UTE :
            mod->BSIM3v32ute = value->rValue;
            mod->BSIM3v32uteGiven = TRUE;
            break;
        case BSIM3v32_MOD_VOFF:
            mod->BSIM3v32voff = value->rValue;
            mod->BSIM3v32voffGiven = TRUE;
            break;
        case  BSIM3v32_MOD_DELTA :
            mod->BSIM3v32delta = value->rValue;
            mod->BSIM3v32deltaGiven = TRUE;
            break;
        case BSIM3v32_MOD_RDSW:
            mod->BSIM3v32rdsw = value->rValue;
            mod->BSIM3v32rdswGiven = TRUE;
            break;
        case BSIM3v32_MOD_PRWG:
            mod->BSIM3v32prwg = value->rValue;
            mod->BSIM3v32prwgGiven = TRUE;
            break;
        case BSIM3v32_MOD_PRWB:
            mod->BSIM3v32prwb = value->rValue;
            mod->BSIM3v32prwbGiven = TRUE;
            break;
        case BSIM3v32_MOD_PRT:
            mod->BSIM3v32prt = value->rValue;
            mod->BSIM3v32prtGiven = TRUE;
            break;
        case BSIM3v32_MOD_ETA0:
            mod->BSIM3v32eta0 = value->rValue;
            mod->BSIM3v32eta0Given = TRUE;
            break;
        case BSIM3v32_MOD_ETAB:
            mod->BSIM3v32etab = value->rValue;
            mod->BSIM3v32etabGiven = TRUE;
            break;
        case BSIM3v32_MOD_PCLM:
            mod->BSIM3v32pclm = value->rValue;
            mod->BSIM3v32pclmGiven = TRUE;
            break;
        case BSIM3v32_MOD_PDIBL1:
            mod->BSIM3v32pdibl1 = value->rValue;
            mod->BSIM3v32pdibl1Given = TRUE;
            break;
        case BSIM3v32_MOD_PDIBL2:
            mod->BSIM3v32pdibl2 = value->rValue;
            mod->BSIM3v32pdibl2Given = TRUE;
            break;
        case BSIM3v32_MOD_PDIBLB:
            mod->BSIM3v32pdiblb = value->rValue;
            mod->BSIM3v32pdiblbGiven = TRUE;
            break;
        case BSIM3v32_MOD_PSCBE1:
            mod->BSIM3v32pscbe1 = value->rValue;
            mod->BSIM3v32pscbe1Given = TRUE;
            break;
        case BSIM3v32_MOD_PSCBE2:
            mod->BSIM3v32pscbe2 = value->rValue;
            mod->BSIM3v32pscbe2Given = TRUE;
            break;
        case BSIM3v32_MOD_PVAG:
            mod->BSIM3v32pvag = value->rValue;
            mod->BSIM3v32pvagGiven = TRUE;
            break;
        case  BSIM3v32_MOD_WR :
            mod->BSIM3v32wr = value->rValue;
            mod->BSIM3v32wrGiven = TRUE;
            break;
        case  BSIM3v32_MOD_DWG :
            mod->BSIM3v32dwg = value->rValue;
            mod->BSIM3v32dwgGiven = TRUE;
            break;
        case  BSIM3v32_MOD_DWB :
            mod->BSIM3v32dwb = value->rValue;
            mod->BSIM3v32dwbGiven = TRUE;
            break;
        case  BSIM3v32_MOD_B0 :
            mod->BSIM3v32b0 = value->rValue;
            mod->BSIM3v32b0Given = TRUE;
            break;
        case  BSIM3v32_MOD_B1 :
            mod->BSIM3v32b1 = value->rValue;
            mod->BSIM3v32b1Given = TRUE;
            break;
        case  BSIM3v32_MOD_ALPHA0 :
            mod->BSIM3v32alpha0 = value->rValue;
            mod->BSIM3v32alpha0Given = TRUE;
            break;
        case  BSIM3v32_MOD_ALPHA1 :
            mod->BSIM3v32alpha1 = value->rValue;
            mod->BSIM3v32alpha1Given = TRUE;
            break;
        case  BSIM3v32_MOD_BETA0 :
            mod->BSIM3v32beta0 = value->rValue;
            mod->BSIM3v32beta0Given = TRUE;
            break;
        case  BSIM3v32_MOD_IJTH :
            mod->BSIM3v32ijth = value->rValue;
            mod->BSIM3v32ijthGiven = TRUE;
            break;
        case  BSIM3v32_MOD_VFB :
            mod->BSIM3v32vfb = value->rValue;
            mod->BSIM3v32vfbGiven = TRUE;
            break;

        case  BSIM3v32_MOD_ELM :
            mod->BSIM3v32elm = value->rValue;
            mod->BSIM3v32elmGiven = TRUE;
            break;
        case  BSIM3v32_MOD_CGSL :
            mod->BSIM3v32cgsl = value->rValue;
            mod->BSIM3v32cgslGiven = TRUE;
            break;
        case  BSIM3v32_MOD_CGDL :
            mod->BSIM3v32cgdl = value->rValue;
            mod->BSIM3v32cgdlGiven = TRUE;
            break;
        case  BSIM3v32_MOD_CKAPPA :
            mod->BSIM3v32ckappa = value->rValue;
            mod->BSIM3v32ckappaGiven = TRUE;
            break;
        case  BSIM3v32_MOD_CF :
            mod->BSIM3v32cf = value->rValue;
            mod->BSIM3v32cfGiven = TRUE;
            break;
        case  BSIM3v32_MOD_CLC :
            mod->BSIM3v32clc = value->rValue;
            mod->BSIM3v32clcGiven = TRUE;
            break;
        case  BSIM3v32_MOD_CLE :
            mod->BSIM3v32cle = value->rValue;
            mod->BSIM3v32cleGiven = TRUE;
            break;
        case  BSIM3v32_MOD_DWC :
            mod->BSIM3v32dwc = value->rValue;
            mod->BSIM3v32dwcGiven = TRUE;
            break;
        case  BSIM3v32_MOD_DLC :
            mod->BSIM3v32dlc = value->rValue;
            mod->BSIM3v32dlcGiven = TRUE;
            break;
        case  BSIM3v32_MOD_VFBCV :
            mod->BSIM3v32vfbcv = value->rValue;
            mod->BSIM3v32vfbcvGiven = TRUE;
            break;
        case  BSIM3v32_MOD_ACDE :
            mod->BSIM3v32acde = value->rValue;
            mod->BSIM3v32acdeGiven = TRUE;
            break;
        case  BSIM3v32_MOD_MOIN :
            mod->BSIM3v32moin = value->rValue;
            mod->BSIM3v32moinGiven = TRUE;
            break;
        case  BSIM3v32_MOD_NOFF :
            mod->BSIM3v32noff = value->rValue;
            mod->BSIM3v32noffGiven = TRUE;
            break;
        case  BSIM3v32_MOD_VOFFCV :
            mod->BSIM3v32voffcv = value->rValue;
            mod->BSIM3v32voffcvGiven = TRUE;
            break;
        case  BSIM3v32_MOD_TCJ :
            mod->BSIM3v32tcj = value->rValue;
            mod->BSIM3v32tcjGiven = TRUE;
            break;
        case  BSIM3v32_MOD_TPB :
            mod->BSIM3v32tpb = value->rValue;
            mod->BSIM3v32tpbGiven = TRUE;
            break;
        case  BSIM3v32_MOD_TCJSW :
            mod->BSIM3v32tcjsw = value->rValue;
            mod->BSIM3v32tcjswGiven = TRUE;
            break;
        case  BSIM3v32_MOD_TPBSW :
            mod->BSIM3v32tpbsw = value->rValue;
            mod->BSIM3v32tpbswGiven = TRUE;
            break;
        case  BSIM3v32_MOD_TCJSWG :
            mod->BSIM3v32tcjswg = value->rValue;
            mod->BSIM3v32tcjswgGiven = TRUE;
            break;
        case  BSIM3v32_MOD_TPBSWG :
            mod->BSIM3v32tpbswg = value->rValue;
            mod->BSIM3v32tpbswgGiven = TRUE;
            break;

          /* acm model */
        case BSIM3v32_MOD_HDIF:
            mod->BSIM3v32hdif = value->rValue;
            mod->BSIM3v32hdifGiven = TRUE;
            break;
        case BSIM3v32_MOD_LDIF:
            mod->BSIM3v32ldif = value->rValue;
            mod->BSIM3v32ldifGiven = TRUE;
            break;
        case BSIM3v32_MOD_LD:
            mod->BSIM3v32ld = value->rValue;
            mod->BSIM3v32ldGiven = TRUE;
            break;
        case BSIM3v32_MOD_RD:
            mod->BSIM3v32rd = value->rValue;
            mod->BSIM3v32rdGiven = TRUE;
            break;
        case BSIM3v32_MOD_RS:
            mod->BSIM3v32rs = value->rValue;
            mod->BSIM3v32rsGiven = TRUE;
            break;
        case BSIM3v32_MOD_RDC:
            mod->BSIM3v32rdc = value->rValue;
            mod->BSIM3v32rdcGiven = TRUE;
            break;
        case BSIM3v32_MOD_RSC:
            mod->BSIM3v32rsc = value->rValue;
            mod->BSIM3v32rscGiven = TRUE;
            break;
        case BSIM3v32_MOD_WMLT:
            mod->BSIM3v32wmlt = value->rValue;
            mod->BSIM3v32wmltGiven = TRUE;
            break;

            /* Length shrink */
        case  BSIM3v32_MOD_LMLT:
            mod->BSIM3v32lmlt = value->rValue;
            mod->BSIM3v32lmltGiven = TRUE;
            break;

        /* Length dependence */
        case  BSIM3v32_MOD_LCDSC :
            mod->BSIM3v32lcdsc = value->rValue;
            mod->BSIM3v32lcdscGiven = TRUE;
            break;


        case  BSIM3v32_MOD_LCDSCB :
            mod->BSIM3v32lcdscb = value->rValue;
            mod->BSIM3v32lcdscbGiven = TRUE;
            break;
        case  BSIM3v32_MOD_LCDSCD :
            mod->BSIM3v32lcdscd = value->rValue;
            mod->BSIM3v32lcdscdGiven = TRUE;
            break;
        case  BSIM3v32_MOD_LCIT :
            mod->BSIM3v32lcit = value->rValue;
            mod->BSIM3v32lcitGiven = TRUE;
            break;
        case  BSIM3v32_MOD_LNFACTOR :
            mod->BSIM3v32lnfactor = value->rValue;
            mod->BSIM3v32lnfactorGiven = TRUE;
            break;
        case BSIM3v32_MOD_LXJ:
            mod->BSIM3v32lxj = value->rValue;
            mod->BSIM3v32lxjGiven = TRUE;
            break;
        case BSIM3v32_MOD_LVSAT:
            mod->BSIM3v32lvsat = value->rValue;
            mod->BSIM3v32lvsatGiven = TRUE;
            break;


        case BSIM3v32_MOD_LA0:
            mod->BSIM3v32la0 = value->rValue;
            mod->BSIM3v32la0Given = TRUE;
            break;
        case BSIM3v32_MOD_LAGS:
            mod->BSIM3v32lags = value->rValue;
            mod->BSIM3v32lagsGiven = TRUE;
            break;
        case BSIM3v32_MOD_LA1:
            mod->BSIM3v32la1 = value->rValue;
            mod->BSIM3v32la1Given = TRUE;
            break;
        case BSIM3v32_MOD_LA2:
            mod->BSIM3v32la2 = value->rValue;
            mod->BSIM3v32la2Given = TRUE;
            break;
        case BSIM3v32_MOD_LAT:
            mod->BSIM3v32lat = value->rValue;
            mod->BSIM3v32latGiven = TRUE;
            break;
        case BSIM3v32_MOD_LKETA:
            mod->BSIM3v32lketa = value->rValue;
            mod->BSIM3v32lketaGiven = TRUE;
            break;
        case BSIM3v32_MOD_LNSUB:
            mod->BSIM3v32lnsub = value->rValue;
            mod->BSIM3v32lnsubGiven = TRUE;
            break;
        case BSIM3v32_MOD_LNPEAK:
            mod->BSIM3v32lnpeak = value->rValue;
            mod->BSIM3v32lnpeakGiven = TRUE;
            if (mod->BSIM3v32lnpeak > 1.0e20)
                mod->BSIM3v32lnpeak *= 1.0e-6;
            break;
        case BSIM3v32_MOD_LNGATE:
            mod->BSIM3v32lngate = value->rValue;
            mod->BSIM3v32lngateGiven = TRUE;
            if (mod->BSIM3v32lngate > 1.0e23)
                mod->BSIM3v32lngate *= 1.0e-6;
            break;
        case BSIM3v32_MOD_LGAMMA1:
            mod->BSIM3v32lgamma1 = value->rValue;
            mod->BSIM3v32lgamma1Given = TRUE;
            break;
        case BSIM3v32_MOD_LGAMMA2:
            mod->BSIM3v32lgamma2 = value->rValue;
            mod->BSIM3v32lgamma2Given = TRUE;
            break;
        case BSIM3v32_MOD_LVBX:
            mod->BSIM3v32lvbx = value->rValue;
            mod->BSIM3v32lvbxGiven = TRUE;
            break;
        case BSIM3v32_MOD_LVBM:
            mod->BSIM3v32lvbm = value->rValue;
            mod->BSIM3v32lvbmGiven = TRUE;
            break;
        case BSIM3v32_MOD_LXT:
            mod->BSIM3v32lxt = value->rValue;
            mod->BSIM3v32lxtGiven = TRUE;
            break;
        case  BSIM3v32_MOD_LK1:
            mod->BSIM3v32lk1 = value->rValue;
            mod->BSIM3v32lk1Given = TRUE;
            break;
        case  BSIM3v32_MOD_LKT1:
            mod->BSIM3v32lkt1 = value->rValue;
            mod->BSIM3v32lkt1Given = TRUE;
            break;
        case  BSIM3v32_MOD_LKT1L:
            mod->BSIM3v32lkt1l = value->rValue;
            mod->BSIM3v32lkt1lGiven = TRUE;
            break;
        case  BSIM3v32_MOD_LKT2:
            mod->BSIM3v32lkt2 = value->rValue;
            mod->BSIM3v32lkt2Given = TRUE;
            break;
        case  BSIM3v32_MOD_LK2:
            mod->BSIM3v32lk2 = value->rValue;
            mod->BSIM3v32lk2Given = TRUE;
            break;
        case  BSIM3v32_MOD_LK3:
            mod->BSIM3v32lk3 = value->rValue;
            mod->BSIM3v32lk3Given = TRUE;
            break;
        case  BSIM3v32_MOD_LK3B:
            mod->BSIM3v32lk3b = value->rValue;
            mod->BSIM3v32lk3bGiven = TRUE;
            break;
        case  BSIM3v32_MOD_LNLX:
            mod->BSIM3v32lnlx = value->rValue;
            mod->BSIM3v32lnlxGiven = TRUE;
            break;
        case  BSIM3v32_MOD_LW0:
            mod->BSIM3v32lw0 = value->rValue;
            mod->BSIM3v32lw0Given = TRUE;
            break;
        case  BSIM3v32_MOD_LDVT0:
            mod->BSIM3v32ldvt0 = value->rValue;
            mod->BSIM3v32ldvt0Given = TRUE;
            break;
        case  BSIM3v32_MOD_LDVT1:
            mod->BSIM3v32ldvt1 = value->rValue;
            mod->BSIM3v32ldvt1Given = TRUE;
            break;
        case  BSIM3v32_MOD_LDVT2:
            mod->BSIM3v32ldvt2 = value->rValue;
            mod->BSIM3v32ldvt2Given = TRUE;
            break;
        case  BSIM3v32_MOD_LDVT0W:
            mod->BSIM3v32ldvt0w = value->rValue;
            mod->BSIM3v32ldvt0wGiven = TRUE;
            break;
        case  BSIM3v32_MOD_LDVT1W:
            mod->BSIM3v32ldvt1w = value->rValue;
            mod->BSIM3v32ldvt1wGiven = TRUE;
            break;
        case  BSIM3v32_MOD_LDVT2W:
            mod->BSIM3v32ldvt2w = value->rValue;
            mod->BSIM3v32ldvt2wGiven = TRUE;
            break;
        case  BSIM3v32_MOD_LDROUT:
            mod->BSIM3v32ldrout = value->rValue;
            mod->BSIM3v32ldroutGiven = TRUE;
            break;
        case  BSIM3v32_MOD_LDSUB:
            mod->BSIM3v32ldsub = value->rValue;
            mod->BSIM3v32ldsubGiven = TRUE;
            break;
        case BSIM3v32_MOD_LVTH0:
            mod->BSIM3v32lvth0 = value->rValue;
            mod->BSIM3v32lvth0Given = TRUE;
            break;
        case BSIM3v32_MOD_LUA:
            mod->BSIM3v32lua = value->rValue;
            mod->BSIM3v32luaGiven = TRUE;
            break;
        case BSIM3v32_MOD_LUA1:
            mod->BSIM3v32lua1 = value->rValue;
            mod->BSIM3v32lua1Given = TRUE;
            break;
        case BSIM3v32_MOD_LUB:
            mod->BSIM3v32lub = value->rValue;
            mod->BSIM3v32lubGiven = TRUE;
            break;
        case BSIM3v32_MOD_LUB1:
            mod->BSIM3v32lub1 = value->rValue;
            mod->BSIM3v32lub1Given = TRUE;
            break;
        case BSIM3v32_MOD_LUC:
            mod->BSIM3v32luc = value->rValue;
            mod->BSIM3v32lucGiven = TRUE;
            break;
        case BSIM3v32_MOD_LUC1:
            mod->BSIM3v32luc1 = value->rValue;
            mod->BSIM3v32luc1Given = TRUE;
            break;
        case  BSIM3v32_MOD_LU0 :
            mod->BSIM3v32lu0 = value->rValue;
            mod->BSIM3v32lu0Given = TRUE;
            break;
        case  BSIM3v32_MOD_LUTE :
            mod->BSIM3v32lute = value->rValue;
            mod->BSIM3v32luteGiven = TRUE;
            break;
        case BSIM3v32_MOD_LVOFF:
            mod->BSIM3v32lvoff = value->rValue;
            mod->BSIM3v32lvoffGiven = TRUE;
            break;
        case  BSIM3v32_MOD_LDELTA :
            mod->BSIM3v32ldelta = value->rValue;
            mod->BSIM3v32ldeltaGiven = TRUE;
            break;
        case BSIM3v32_MOD_LRDSW:
            mod->BSIM3v32lrdsw = value->rValue;
            mod->BSIM3v32lrdswGiven = TRUE;
            break;
        case BSIM3v32_MOD_LPRWB:
            mod->BSIM3v32lprwb = value->rValue;
            mod->BSIM3v32lprwbGiven = TRUE;
            break;
        case BSIM3v32_MOD_LPRWG:
            mod->BSIM3v32lprwg = value->rValue;
            mod->BSIM3v32lprwgGiven = TRUE;
            break;
        case BSIM3v32_MOD_LPRT:
            mod->BSIM3v32lprt = value->rValue;
            mod->BSIM3v32lprtGiven = TRUE;
            break;
        case BSIM3v32_MOD_LETA0:
            mod->BSIM3v32leta0 = value->rValue;
            mod->BSIM3v32leta0Given = TRUE;
            break;
        case BSIM3v32_MOD_LETAB:
            mod->BSIM3v32letab = value->rValue;
            mod->BSIM3v32letabGiven = TRUE;
            break;
        case BSIM3v32_MOD_LPCLM:
            mod->BSIM3v32lpclm = value->rValue;
            mod->BSIM3v32lpclmGiven = TRUE;
            break;
        case BSIM3v32_MOD_LPDIBL1:
            mod->BSIM3v32lpdibl1 = value->rValue;
            mod->BSIM3v32lpdibl1Given = TRUE;
            break;
        case BSIM3v32_MOD_LPDIBL2:
            mod->BSIM3v32lpdibl2 = value->rValue;
            mod->BSIM3v32lpdibl2Given = TRUE;
            break;
        case BSIM3v32_MOD_LPDIBLB:
            mod->BSIM3v32lpdiblb = value->rValue;
            mod->BSIM3v32lpdiblbGiven = TRUE;
            break;
        case BSIM3v32_MOD_LPSCBE1:
            mod->BSIM3v32lpscbe1 = value->rValue;
            mod->BSIM3v32lpscbe1Given = TRUE;
            break;
        case BSIM3v32_MOD_LPSCBE2:
            mod->BSIM3v32lpscbe2 = value->rValue;
            mod->BSIM3v32lpscbe2Given = TRUE;
            break;
        case BSIM3v32_MOD_LPVAG:
            mod->BSIM3v32lpvag = value->rValue;
            mod->BSIM3v32lpvagGiven = TRUE;
            break;
        case  BSIM3v32_MOD_LWR :
            mod->BSIM3v32lwr = value->rValue;
            mod->BSIM3v32lwrGiven = TRUE;
            break;
        case  BSIM3v32_MOD_LDWG :
            mod->BSIM3v32ldwg = value->rValue;
            mod->BSIM3v32ldwgGiven = TRUE;
            break;
        case  BSIM3v32_MOD_LDWB :
            mod->BSIM3v32ldwb = value->rValue;
            mod->BSIM3v32ldwbGiven = TRUE;
            break;
        case  BSIM3v32_MOD_LB0 :
            mod->BSIM3v32lb0 = value->rValue;
            mod->BSIM3v32lb0Given = TRUE;
            break;
        case  BSIM3v32_MOD_LB1 :
            mod->BSIM3v32lb1 = value->rValue;
            mod->BSIM3v32lb1Given = TRUE;
            break;
        case  BSIM3v32_MOD_LALPHA0 :
            mod->BSIM3v32lalpha0 = value->rValue;
            mod->BSIM3v32lalpha0Given = TRUE;
            break;
        case  BSIM3v32_MOD_LALPHA1 :
            mod->BSIM3v32lalpha1 = value->rValue;
            mod->BSIM3v32lalpha1Given = TRUE;
            break;
        case  BSIM3v32_MOD_LBETA0 :
            mod->BSIM3v32lbeta0 = value->rValue;
            mod->BSIM3v32lbeta0Given = TRUE;
            break;
        case  BSIM3v32_MOD_LVFB :
            mod->BSIM3v32lvfb = value->rValue;
            mod->BSIM3v32lvfbGiven = TRUE;
            break;

        case  BSIM3v32_MOD_LELM :
            mod->BSIM3v32lelm = value->rValue;
            mod->BSIM3v32lelmGiven = TRUE;
            break;
        case  BSIM3v32_MOD_LCGSL :
            mod->BSIM3v32lcgsl = value->rValue;
            mod->BSIM3v32lcgslGiven = TRUE;
            break;
        case  BSIM3v32_MOD_LCGDL :
            mod->BSIM3v32lcgdl = value->rValue;
            mod->BSIM3v32lcgdlGiven = TRUE;
            break;
        case  BSIM3v32_MOD_LCKAPPA :
            mod->BSIM3v32lckappa = value->rValue;
            mod->BSIM3v32lckappaGiven = TRUE;
            break;
        case  BSIM3v32_MOD_LCF :
            mod->BSIM3v32lcf = value->rValue;
            mod->BSIM3v32lcfGiven = TRUE;
            break;
        case  BSIM3v32_MOD_LCLC :
            mod->BSIM3v32lclc = value->rValue;
            mod->BSIM3v32lclcGiven = TRUE;
            break;
        case  BSIM3v32_MOD_LCLE :
            mod->BSIM3v32lcle = value->rValue;
            mod->BSIM3v32lcleGiven = TRUE;
            break;
        case  BSIM3v32_MOD_LVFBCV :
            mod->BSIM3v32lvfbcv = value->rValue;
            mod->BSIM3v32lvfbcvGiven = TRUE;
            break;
        case  BSIM3v32_MOD_LACDE :
            mod->BSIM3v32lacde = value->rValue;
            mod->BSIM3v32lacdeGiven = TRUE;
            break;
        case  BSIM3v32_MOD_LMOIN :
            mod->BSIM3v32lmoin = value->rValue;
            mod->BSIM3v32lmoinGiven = TRUE;
            break;
        case  BSIM3v32_MOD_LNOFF :
            mod->BSIM3v32lnoff = value->rValue;
            mod->BSIM3v32lnoffGiven = TRUE;
            break;
        case  BSIM3v32_MOD_LVOFFCV :
            mod->BSIM3v32lvoffcv = value->rValue;
            mod->BSIM3v32lvoffcvGiven = TRUE;
            break;

        /* Width dependence */
        case  BSIM3v32_MOD_WCDSC :
            mod->BSIM3v32wcdsc = value->rValue;
            mod->BSIM3v32wcdscGiven = TRUE;
            break;


         case  BSIM3v32_MOD_WCDSCB :
            mod->BSIM3v32wcdscb = value->rValue;
            mod->BSIM3v32wcdscbGiven = TRUE;
            break;
         case  BSIM3v32_MOD_WCDSCD :
            mod->BSIM3v32wcdscd = value->rValue;
            mod->BSIM3v32wcdscdGiven = TRUE;
            break;
        case  BSIM3v32_MOD_WCIT :
            mod->BSIM3v32wcit = value->rValue;
            mod->BSIM3v32wcitGiven = TRUE;
            break;
        case  BSIM3v32_MOD_WNFACTOR :
            mod->BSIM3v32wnfactor = value->rValue;
            mod->BSIM3v32wnfactorGiven = TRUE;
            break;
        case BSIM3v32_MOD_WXJ:
            mod->BSIM3v32wxj = value->rValue;
            mod->BSIM3v32wxjGiven = TRUE;
            break;
        case BSIM3v32_MOD_WVSAT:
            mod->BSIM3v32wvsat = value->rValue;
            mod->BSIM3v32wvsatGiven = TRUE;
            break;


        case BSIM3v32_MOD_WA0:
            mod->BSIM3v32wa0 = value->rValue;
            mod->BSIM3v32wa0Given = TRUE;
            break;
        case BSIM3v32_MOD_WAGS:
            mod->BSIM3v32wags = value->rValue;
            mod->BSIM3v32wagsGiven = TRUE;
            break;
        case BSIM3v32_MOD_WA1:
            mod->BSIM3v32wa1 = value->rValue;
            mod->BSIM3v32wa1Given = TRUE;
            break;
        case BSIM3v32_MOD_WA2:
            mod->BSIM3v32wa2 = value->rValue;
            mod->BSIM3v32wa2Given = TRUE;
            break;
        case BSIM3v32_MOD_WAT:
            mod->BSIM3v32wat = value->rValue;
            mod->BSIM3v32watGiven = TRUE;
            break;
        case BSIM3v32_MOD_WKETA:
            mod->BSIM3v32wketa = value->rValue;
            mod->BSIM3v32wketaGiven = TRUE;
            break;
        case BSIM3v32_MOD_WNSUB:
            mod->BSIM3v32wnsub = value->rValue;
            mod->BSIM3v32wnsubGiven = TRUE;
            break;
        case BSIM3v32_MOD_WNPEAK:
            mod->BSIM3v32wnpeak = value->rValue;
            mod->BSIM3v32wnpeakGiven = TRUE;
            if (mod->BSIM3v32wnpeak > 1.0e20)
                mod->BSIM3v32wnpeak *= 1.0e-6;
            break;
        case BSIM3v32_MOD_WNGATE:
            mod->BSIM3v32wngate = value->rValue;
            mod->BSIM3v32wngateGiven = TRUE;
            if (mod->BSIM3v32wngate > 1.0e23)
                mod->BSIM3v32wngate *= 1.0e-6;
            break;
        case BSIM3v32_MOD_WGAMMA1:
            mod->BSIM3v32wgamma1 = value->rValue;
            mod->BSIM3v32wgamma1Given = TRUE;
            break;
        case BSIM3v32_MOD_WGAMMA2:
            mod->BSIM3v32wgamma2 = value->rValue;
            mod->BSIM3v32wgamma2Given = TRUE;
            break;
        case BSIM3v32_MOD_WVBX:
            mod->BSIM3v32wvbx = value->rValue;
            mod->BSIM3v32wvbxGiven = TRUE;
            break;
        case BSIM3v32_MOD_WVBM:
            mod->BSIM3v32wvbm = value->rValue;
            mod->BSIM3v32wvbmGiven = TRUE;
            break;
        case BSIM3v32_MOD_WXT:
            mod->BSIM3v32wxt = value->rValue;
            mod->BSIM3v32wxtGiven = TRUE;
            break;
        case  BSIM3v32_MOD_WK1:
            mod->BSIM3v32wk1 = value->rValue;
            mod->BSIM3v32wk1Given = TRUE;
            break;
        case  BSIM3v32_MOD_WKT1:
            mod->BSIM3v32wkt1 = value->rValue;
            mod->BSIM3v32wkt1Given = TRUE;
            break;
        case  BSIM3v32_MOD_WKT1L:
            mod->BSIM3v32wkt1l = value->rValue;
            mod->BSIM3v32wkt1lGiven = TRUE;
            break;
        case  BSIM3v32_MOD_WKT2:
            mod->BSIM3v32wkt2 = value->rValue;
            mod->BSIM3v32wkt2Given = TRUE;
            break;
        case  BSIM3v32_MOD_WK2:
            mod->BSIM3v32wk2 = value->rValue;
            mod->BSIM3v32wk2Given = TRUE;
            break;
        case  BSIM3v32_MOD_WK3:
            mod->BSIM3v32wk3 = value->rValue;
            mod->BSIM3v32wk3Given = TRUE;
            break;
        case  BSIM3v32_MOD_WK3B:
            mod->BSIM3v32wk3b = value->rValue;
            mod->BSIM3v32wk3bGiven = TRUE;
            break;
        case  BSIM3v32_MOD_WNLX:
            mod->BSIM3v32wnlx = value->rValue;
            mod->BSIM3v32wnlxGiven = TRUE;
            break;
        case  BSIM3v32_MOD_WW0:
            mod->BSIM3v32ww0 = value->rValue;
            mod->BSIM3v32ww0Given = TRUE;
            break;
        case  BSIM3v32_MOD_WDVT0:
            mod->BSIM3v32wdvt0 = value->rValue;
            mod->BSIM3v32wdvt0Given = TRUE;
            break;
        case  BSIM3v32_MOD_WDVT1:
            mod->BSIM3v32wdvt1 = value->rValue;
            mod->BSIM3v32wdvt1Given = TRUE;
            break;
        case  BSIM3v32_MOD_WDVT2:
            mod->BSIM3v32wdvt2 = value->rValue;
            mod->BSIM3v32wdvt2Given = TRUE;
            break;
        case  BSIM3v32_MOD_WDVT0W:
            mod->BSIM3v32wdvt0w = value->rValue;
            mod->BSIM3v32wdvt0wGiven = TRUE;
            break;
        case  BSIM3v32_MOD_WDVT1W:
            mod->BSIM3v32wdvt1w = value->rValue;
            mod->BSIM3v32wdvt1wGiven = TRUE;
            break;
        case  BSIM3v32_MOD_WDVT2W:
            mod->BSIM3v32wdvt2w = value->rValue;
            mod->BSIM3v32wdvt2wGiven = TRUE;
            break;
        case  BSIM3v32_MOD_WDROUT:
            mod->BSIM3v32wdrout = value->rValue;
            mod->BSIM3v32wdroutGiven = TRUE;
            break;
        case  BSIM3v32_MOD_WDSUB:
            mod->BSIM3v32wdsub = value->rValue;
            mod->BSIM3v32wdsubGiven = TRUE;
            break;
        case BSIM3v32_MOD_WVTH0:
            mod->BSIM3v32wvth0 = value->rValue;
            mod->BSIM3v32wvth0Given = TRUE;
            break;
        case BSIM3v32_MOD_WUA:
            mod->BSIM3v32wua = value->rValue;
            mod->BSIM3v32wuaGiven = TRUE;
            break;
        case BSIM3v32_MOD_WUA1:
            mod->BSIM3v32wua1 = value->rValue;
            mod->BSIM3v32wua1Given = TRUE;
            break;
        case BSIM3v32_MOD_WUB:
            mod->BSIM3v32wub = value->rValue;
            mod->BSIM3v32wubGiven = TRUE;
            break;
        case BSIM3v32_MOD_WUB1:
            mod->BSIM3v32wub1 = value->rValue;
            mod->BSIM3v32wub1Given = TRUE;
            break;
        case BSIM3v32_MOD_WUC:
            mod->BSIM3v32wuc = value->rValue;
            mod->BSIM3v32wucGiven = TRUE;
            break;
        case BSIM3v32_MOD_WUC1:
            mod->BSIM3v32wuc1 = value->rValue;
            mod->BSIM3v32wuc1Given = TRUE;
            break;
        case  BSIM3v32_MOD_WU0 :
            mod->BSIM3v32wu0 = value->rValue;
            mod->BSIM3v32wu0Given = TRUE;
            break;
        case  BSIM3v32_MOD_WUTE :
            mod->BSIM3v32wute = value->rValue;
            mod->BSIM3v32wuteGiven = TRUE;
            break;
        case BSIM3v32_MOD_WVOFF:
            mod->BSIM3v32wvoff = value->rValue;
            mod->BSIM3v32wvoffGiven = TRUE;
            break;
        case  BSIM3v32_MOD_WDELTA :
            mod->BSIM3v32wdelta = value->rValue;
            mod->BSIM3v32wdeltaGiven = TRUE;
            break;
        case BSIM3v32_MOD_WRDSW:
            mod->BSIM3v32wrdsw = value->rValue;
            mod->BSIM3v32wrdswGiven = TRUE;
            break;
        case BSIM3v32_MOD_WPRWB:
            mod->BSIM3v32wprwb = value->rValue;
            mod->BSIM3v32wprwbGiven = TRUE;
            break;
        case BSIM3v32_MOD_WPRWG:
            mod->BSIM3v32wprwg = value->rValue;
            mod->BSIM3v32wprwgGiven = TRUE;
            break;
        case BSIM3v32_MOD_WPRT:
            mod->BSIM3v32wprt = value->rValue;
            mod->BSIM3v32wprtGiven = TRUE;
            break;
        case BSIM3v32_MOD_WETA0:
            mod->BSIM3v32weta0 = value->rValue;
            mod->BSIM3v32weta0Given = TRUE;
            break;
        case BSIM3v32_MOD_WETAB:
            mod->BSIM3v32wetab = value->rValue;
            mod->BSIM3v32wetabGiven = TRUE;
            break;
        case BSIM3v32_MOD_WPCLM:
            mod->BSIM3v32wpclm = value->rValue;
            mod->BSIM3v32wpclmGiven = TRUE;
            break;
        case BSIM3v32_MOD_WPDIBL1:
            mod->BSIM3v32wpdibl1 = value->rValue;
            mod->BSIM3v32wpdibl1Given = TRUE;
            break;
        case BSIM3v32_MOD_WPDIBL2:
            mod->BSIM3v32wpdibl2 = value->rValue;
            mod->BSIM3v32wpdibl2Given = TRUE;
            break;
        case BSIM3v32_MOD_WPDIBLB:
            mod->BSIM3v32wpdiblb = value->rValue;
            mod->BSIM3v32wpdiblbGiven = TRUE;
            break;
        case BSIM3v32_MOD_WPSCBE1:
            mod->BSIM3v32wpscbe1 = value->rValue;
            mod->BSIM3v32wpscbe1Given = TRUE;
            break;
        case BSIM3v32_MOD_WPSCBE2:
            mod->BSIM3v32wpscbe2 = value->rValue;
            mod->BSIM3v32wpscbe2Given = TRUE;
            break;
        case BSIM3v32_MOD_WPVAG:
            mod->BSIM3v32wpvag = value->rValue;
            mod->BSIM3v32wpvagGiven = TRUE;
            break;
        case  BSIM3v32_MOD_WWR :
            mod->BSIM3v32wwr = value->rValue;
            mod->BSIM3v32wwrGiven = TRUE;
            break;
        case  BSIM3v32_MOD_WDWG :
            mod->BSIM3v32wdwg = value->rValue;
            mod->BSIM3v32wdwgGiven = TRUE;
            break;
        case  BSIM3v32_MOD_WDWB :
            mod->BSIM3v32wdwb = value->rValue;
            mod->BSIM3v32wdwbGiven = TRUE;
            break;
        case  BSIM3v32_MOD_WB0 :
            mod->BSIM3v32wb0 = value->rValue;
            mod->BSIM3v32wb0Given = TRUE;
            break;
        case  BSIM3v32_MOD_WB1 :
            mod->BSIM3v32wb1 = value->rValue;
            mod->BSIM3v32wb1Given = TRUE;
            break;
        case  BSIM3v32_MOD_WALPHA0 :
            mod->BSIM3v32walpha0 = value->rValue;
            mod->BSIM3v32walpha0Given = TRUE;
            break;
        case  BSIM3v32_MOD_WALPHA1 :
            mod->BSIM3v32walpha1 = value->rValue;
            mod->BSIM3v32walpha1Given = TRUE;
            break;
        case  BSIM3v32_MOD_WBETA0 :
            mod->BSIM3v32wbeta0 = value->rValue;
            mod->BSIM3v32wbeta0Given = TRUE;
            break;
        case  BSIM3v32_MOD_WVFB :
            mod->BSIM3v32wvfb = value->rValue;
            mod->BSIM3v32wvfbGiven = TRUE;
            break;

        case  BSIM3v32_MOD_WELM :
            mod->BSIM3v32welm = value->rValue;
            mod->BSIM3v32welmGiven = TRUE;
            break;
        case  BSIM3v32_MOD_WCGSL :
            mod->BSIM3v32wcgsl = value->rValue;
            mod->BSIM3v32wcgslGiven = TRUE;
            break;
        case  BSIM3v32_MOD_WCGDL :
            mod->BSIM3v32wcgdl = value->rValue;
            mod->BSIM3v32wcgdlGiven = TRUE;
            break;
        case  BSIM3v32_MOD_WCKAPPA :
            mod->BSIM3v32wckappa = value->rValue;
            mod->BSIM3v32wckappaGiven = TRUE;
            break;
        case  BSIM3v32_MOD_WCF :
            mod->BSIM3v32wcf = value->rValue;
            mod->BSIM3v32wcfGiven = TRUE;
            break;
        case  BSIM3v32_MOD_WCLC :
            mod->BSIM3v32wclc = value->rValue;
            mod->BSIM3v32wclcGiven = TRUE;
            break;
        case  BSIM3v32_MOD_WCLE :
            mod->BSIM3v32wcle = value->rValue;
            mod->BSIM3v32wcleGiven = TRUE;
            break;
        case  BSIM3v32_MOD_WVFBCV :
            mod->BSIM3v32wvfbcv = value->rValue;
            mod->BSIM3v32wvfbcvGiven = TRUE;
            break;
        case  BSIM3v32_MOD_WACDE :
            mod->BSIM3v32wacde = value->rValue;
            mod->BSIM3v32wacdeGiven = TRUE;
            break;
        case  BSIM3v32_MOD_WMOIN :
            mod->BSIM3v32wmoin = value->rValue;
            mod->BSIM3v32wmoinGiven = TRUE;
            break;
        case  BSIM3v32_MOD_WNOFF :
            mod->BSIM3v32wnoff = value->rValue;
            mod->BSIM3v32wnoffGiven = TRUE;
            break;
        case  BSIM3v32_MOD_WVOFFCV :
            mod->BSIM3v32wvoffcv = value->rValue;
            mod->BSIM3v32wvoffcvGiven = TRUE;
            break;

        /* Cross-term dependence */
        case  BSIM3v32_MOD_PCDSC :
            mod->BSIM3v32pcdsc = value->rValue;
            mod->BSIM3v32pcdscGiven = TRUE;
            break;


        case  BSIM3v32_MOD_PCDSCB :
            mod->BSIM3v32pcdscb = value->rValue;
            mod->BSIM3v32pcdscbGiven = TRUE;
            break;
        case  BSIM3v32_MOD_PCDSCD :
            mod->BSIM3v32pcdscd = value->rValue;
            mod->BSIM3v32pcdscdGiven = TRUE;
            break;
        case  BSIM3v32_MOD_PCIT :
            mod->BSIM3v32pcit = value->rValue;
            mod->BSIM3v32pcitGiven = TRUE;
            break;
        case  BSIM3v32_MOD_PNFACTOR :
            mod->BSIM3v32pnfactor = value->rValue;
            mod->BSIM3v32pnfactorGiven = TRUE;
            break;
        case BSIM3v32_MOD_PXJ:
            mod->BSIM3v32pxj = value->rValue;
            mod->BSIM3v32pxjGiven = TRUE;
            break;
        case BSIM3v32_MOD_PVSAT:
            mod->BSIM3v32pvsat = value->rValue;
            mod->BSIM3v32pvsatGiven = TRUE;
            break;


        case BSIM3v32_MOD_PA0:
            mod->BSIM3v32pa0 = value->rValue;
            mod->BSIM3v32pa0Given = TRUE;
            break;
        case BSIM3v32_MOD_PAGS:
            mod->BSIM3v32pags = value->rValue;
            mod->BSIM3v32pagsGiven = TRUE;
            break;
        case BSIM3v32_MOD_PA1:
            mod->BSIM3v32pa1 = value->rValue;
            mod->BSIM3v32pa1Given = TRUE;
            break;
        case BSIM3v32_MOD_PA2:
            mod->BSIM3v32pa2 = value->rValue;
            mod->BSIM3v32pa2Given = TRUE;
            break;
        case BSIM3v32_MOD_PAT:
            mod->BSIM3v32pat = value->rValue;
            mod->BSIM3v32patGiven = TRUE;
            break;
        case BSIM3v32_MOD_PKETA:
            mod->BSIM3v32pketa = value->rValue;
            mod->BSIM3v32pketaGiven = TRUE;
            break;
        case BSIM3v32_MOD_PNSUB:
            mod->BSIM3v32pnsub = value->rValue;
            mod->BSIM3v32pnsubGiven = TRUE;
            break;
        case BSIM3v32_MOD_PNPEAK:
            mod->BSIM3v32pnpeak = value->rValue;
            mod->BSIM3v32pnpeakGiven = TRUE;
            if (mod->BSIM3v32pnpeak > 1.0e20)
                mod->BSIM3v32pnpeak *= 1.0e-6;
            break;
        case BSIM3v32_MOD_PNGATE:
            mod->BSIM3v32pngate = value->rValue;
            mod->BSIM3v32pngateGiven = TRUE;
            if (mod->BSIM3v32pngate > 1.0e23)
                mod->BSIM3v32pngate *= 1.0e-6;
            break;
        case BSIM3v32_MOD_PGAMMA1:
            mod->BSIM3v32pgamma1 = value->rValue;
            mod->BSIM3v32pgamma1Given = TRUE;
            break;
        case BSIM3v32_MOD_PGAMMA2:
            mod->BSIM3v32pgamma2 = value->rValue;
            mod->BSIM3v32pgamma2Given = TRUE;
            break;
        case BSIM3v32_MOD_PVBX:
            mod->BSIM3v32pvbx = value->rValue;
            mod->BSIM3v32pvbxGiven = TRUE;
            break;
        case BSIM3v32_MOD_PVBM:
            mod->BSIM3v32pvbm = value->rValue;
            mod->BSIM3v32pvbmGiven = TRUE;
            break;
        case BSIM3v32_MOD_PXT:
            mod->BSIM3v32pxt = value->rValue;
            mod->BSIM3v32pxtGiven = TRUE;
            break;
        case  BSIM3v32_MOD_PK1:
            mod->BSIM3v32pk1 = value->rValue;
            mod->BSIM3v32pk1Given = TRUE;
            break;
        case  BSIM3v32_MOD_PKT1:
            mod->BSIM3v32pkt1 = value->rValue;
            mod->BSIM3v32pkt1Given = TRUE;
            break;
        case  BSIM3v32_MOD_PKT1L:
            mod->BSIM3v32pkt1l = value->rValue;
            mod->BSIM3v32pkt1lGiven = TRUE;
            break;
        case  BSIM3v32_MOD_PKT2:
            mod->BSIM3v32pkt2 = value->rValue;
            mod->BSIM3v32pkt2Given = TRUE;
            break;
        case  BSIM3v32_MOD_PK2:
            mod->BSIM3v32pk2 = value->rValue;
            mod->BSIM3v32pk2Given = TRUE;
            break;
        case  BSIM3v32_MOD_PK3:
            mod->BSIM3v32pk3 = value->rValue;
            mod->BSIM3v32pk3Given = TRUE;
            break;
        case  BSIM3v32_MOD_PK3B:
            mod->BSIM3v32pk3b = value->rValue;
            mod->BSIM3v32pk3bGiven = TRUE;
            break;
        case  BSIM3v32_MOD_PNLX:
            mod->BSIM3v32pnlx = value->rValue;
            mod->BSIM3v32pnlxGiven = TRUE;
            break;
        case  BSIM3v32_MOD_PW0:
            mod->BSIM3v32pw0 = value->rValue;
            mod->BSIM3v32pw0Given = TRUE;
            break;
        case  BSIM3v32_MOD_PDVT0:
            mod->BSIM3v32pdvt0 = value->rValue;
            mod->BSIM3v32pdvt0Given = TRUE;
            break;
        case  BSIM3v32_MOD_PDVT1:
            mod->BSIM3v32pdvt1 = value->rValue;
            mod->BSIM3v32pdvt1Given = TRUE;
            break;
        case  BSIM3v32_MOD_PDVT2:
            mod->BSIM3v32pdvt2 = value->rValue;
            mod->BSIM3v32pdvt2Given = TRUE;
            break;
        case  BSIM3v32_MOD_PDVT0W:
            mod->BSIM3v32pdvt0w = value->rValue;
            mod->BSIM3v32pdvt0wGiven = TRUE;
            break;
        case  BSIM3v32_MOD_PDVT1W:
            mod->BSIM3v32pdvt1w = value->rValue;
            mod->BSIM3v32pdvt1wGiven = TRUE;
            break;
        case  BSIM3v32_MOD_PDVT2W:
            mod->BSIM3v32pdvt2w = value->rValue;
            mod->BSIM3v32pdvt2wGiven = TRUE;
            break;
        case  BSIM3v32_MOD_PDROUT:
            mod->BSIM3v32pdrout = value->rValue;
            mod->BSIM3v32pdroutGiven = TRUE;
            break;
        case  BSIM3v32_MOD_PDSUB:
            mod->BSIM3v32pdsub = value->rValue;
            mod->BSIM3v32pdsubGiven = TRUE;
            break;
        case BSIM3v32_MOD_PVTH0:
            mod->BSIM3v32pvth0 = value->rValue;
            mod->BSIM3v32pvth0Given = TRUE;
            break;
        case BSIM3v32_MOD_PUA:
            mod->BSIM3v32pua = value->rValue;
            mod->BSIM3v32puaGiven = TRUE;
            break;
        case BSIM3v32_MOD_PUA1:
            mod->BSIM3v32pua1 = value->rValue;
            mod->BSIM3v32pua1Given = TRUE;
            break;
        case BSIM3v32_MOD_PUB:
            mod->BSIM3v32pub = value->rValue;
            mod->BSIM3v32pubGiven = TRUE;
            break;
        case BSIM3v32_MOD_PUB1:
            mod->BSIM3v32pub1 = value->rValue;
            mod->BSIM3v32pub1Given = TRUE;
            break;
        case BSIM3v32_MOD_PUC:
            mod->BSIM3v32puc = value->rValue;
            mod->BSIM3v32pucGiven = TRUE;
            break;
        case BSIM3v32_MOD_PUC1:
            mod->BSIM3v32puc1 = value->rValue;
            mod->BSIM3v32puc1Given = TRUE;
            break;
        case  BSIM3v32_MOD_PU0 :
            mod->BSIM3v32pu0 = value->rValue;
            mod->BSIM3v32pu0Given = TRUE;
            break;
        case  BSIM3v32_MOD_PUTE :
            mod->BSIM3v32pute = value->rValue;
            mod->BSIM3v32puteGiven = TRUE;
            break;
        case BSIM3v32_MOD_PVOFF:
            mod->BSIM3v32pvoff = value->rValue;
            mod->BSIM3v32pvoffGiven = TRUE;
            break;
        case  BSIM3v32_MOD_PDELTA :
            mod->BSIM3v32pdelta = value->rValue;
            mod->BSIM3v32pdeltaGiven = TRUE;
            break;
        case BSIM3v32_MOD_PRDSW:
            mod->BSIM3v32prdsw = value->rValue;
            mod->BSIM3v32prdswGiven = TRUE;
            break;
        case BSIM3v32_MOD_PPRWB:
            mod->BSIM3v32pprwb = value->rValue;
            mod->BSIM3v32pprwbGiven = TRUE;
            break;
        case BSIM3v32_MOD_PPRWG:
            mod->BSIM3v32pprwg = value->rValue;
            mod->BSIM3v32pprwgGiven = TRUE;
            break;
        case BSIM3v32_MOD_PPRT:
            mod->BSIM3v32pprt = value->rValue;
            mod->BSIM3v32pprtGiven = TRUE;
            break;
        case BSIM3v32_MOD_PETA0:
            mod->BSIM3v32peta0 = value->rValue;
            mod->BSIM3v32peta0Given = TRUE;
            break;
        case BSIM3v32_MOD_PETAB:
            mod->BSIM3v32petab = value->rValue;
            mod->BSIM3v32petabGiven = TRUE;
            break;
        case BSIM3v32_MOD_PPCLM:
            mod->BSIM3v32ppclm = value->rValue;
            mod->BSIM3v32ppclmGiven = TRUE;
            break;
        case BSIM3v32_MOD_PPDIBL1:
            mod->BSIM3v32ppdibl1 = value->rValue;
            mod->BSIM3v32ppdibl1Given = TRUE;
            break;
        case BSIM3v32_MOD_PPDIBL2:
            mod->BSIM3v32ppdibl2 = value->rValue;
            mod->BSIM3v32ppdibl2Given = TRUE;
            break;
        case BSIM3v32_MOD_PPDIBLB:
            mod->BSIM3v32ppdiblb = value->rValue;
            mod->BSIM3v32ppdiblbGiven = TRUE;
            break;
        case BSIM3v32_MOD_PPSCBE1:
            mod->BSIM3v32ppscbe1 = value->rValue;
            mod->BSIM3v32ppscbe1Given = TRUE;
            break;
        case BSIM3v32_MOD_PPSCBE2:
            mod->BSIM3v32ppscbe2 = value->rValue;
            mod->BSIM3v32ppscbe2Given = TRUE;
            break;
        case BSIM3v32_MOD_PPVAG:
            mod->BSIM3v32ppvag = value->rValue;
            mod->BSIM3v32ppvagGiven = TRUE;
            break;
        case  BSIM3v32_MOD_PWR :
            mod->BSIM3v32pwr = value->rValue;
            mod->BSIM3v32pwrGiven = TRUE;
            break;
        case  BSIM3v32_MOD_PDWG :
            mod->BSIM3v32pdwg = value->rValue;
            mod->BSIM3v32pdwgGiven = TRUE;
            break;
        case  BSIM3v32_MOD_PDWB :
            mod->BSIM3v32pdwb = value->rValue;
            mod->BSIM3v32pdwbGiven = TRUE;
            break;
        case  BSIM3v32_MOD_PB0 :
            mod->BSIM3v32pb0 = value->rValue;
            mod->BSIM3v32pb0Given = TRUE;
            break;
        case  BSIM3v32_MOD_PB1 :
            mod->BSIM3v32pb1 = value->rValue;
            mod->BSIM3v32pb1Given = TRUE;
            break;
        case  BSIM3v32_MOD_PALPHA0 :
            mod->BSIM3v32palpha0 = value->rValue;
            mod->BSIM3v32palpha0Given = TRUE;
            break;
        case  BSIM3v32_MOD_PALPHA1 :
            mod->BSIM3v32palpha1 = value->rValue;
            mod->BSIM3v32palpha1Given = TRUE;
            break;
        case  BSIM3v32_MOD_PBETA0 :
            mod->BSIM3v32pbeta0 = value->rValue;
            mod->BSIM3v32pbeta0Given = TRUE;
            break;
        case  BSIM3v32_MOD_PVFB :
            mod->BSIM3v32pvfb = value->rValue;
            mod->BSIM3v32pvfbGiven = TRUE;
            break;

        case  BSIM3v32_MOD_PELM :
            mod->BSIM3v32pelm = value->rValue;
            mod->BSIM3v32pelmGiven = TRUE;
            break;
        case  BSIM3v32_MOD_PCGSL :
            mod->BSIM3v32pcgsl = value->rValue;
            mod->BSIM3v32pcgslGiven = TRUE;
            break;
        case  BSIM3v32_MOD_PCGDL :
            mod->BSIM3v32pcgdl = value->rValue;
            mod->BSIM3v32pcgdlGiven = TRUE;
            break;
        case  BSIM3v32_MOD_PCKAPPA :
            mod->BSIM3v32pckappa = value->rValue;
            mod->BSIM3v32pckappaGiven = TRUE;
            break;
        case  BSIM3v32_MOD_PCF :
            mod->BSIM3v32pcf = value->rValue;
            mod->BSIM3v32pcfGiven = TRUE;
            break;
        case  BSIM3v32_MOD_PCLC :
            mod->BSIM3v32pclc = value->rValue;
            mod->BSIM3v32pclcGiven = TRUE;
            break;
        case  BSIM3v32_MOD_PCLE :
            mod->BSIM3v32pcle = value->rValue;
            mod->BSIM3v32pcleGiven = TRUE;
            break;
        case  BSIM3v32_MOD_PVFBCV :
            mod->BSIM3v32pvfbcv = value->rValue;
            mod->BSIM3v32pvfbcvGiven = TRUE;
            break;
        case  BSIM3v32_MOD_PACDE :
            mod->BSIM3v32pacde = value->rValue;
            mod->BSIM3v32pacdeGiven = TRUE;
            break;
        case  BSIM3v32_MOD_PMOIN :
            mod->BSIM3v32pmoin = value->rValue;
            mod->BSIM3v32pmoinGiven = TRUE;
            break;
        case  BSIM3v32_MOD_PNOFF :
            mod->BSIM3v32pnoff = value->rValue;
            mod->BSIM3v32pnoffGiven = TRUE;
            break;
        case  BSIM3v32_MOD_PVOFFCV :
            mod->BSIM3v32pvoffcv = value->rValue;
            mod->BSIM3v32pvoffcvGiven = TRUE;
            break;

        case  BSIM3v32_MOD_TNOM :
            mod->BSIM3v32tnom = value->rValue + CONSTCtoK;
            mod->BSIM3v32tnomGiven = TRUE;
            break;
        case  BSIM3v32_MOD_CGSO :
            mod->BSIM3v32cgso = value->rValue;
            mod->BSIM3v32cgsoGiven = TRUE;
            break;
        case  BSIM3v32_MOD_CGDO :
            mod->BSIM3v32cgdo = value->rValue;
            mod->BSIM3v32cgdoGiven = TRUE;
            break;
        case  BSIM3v32_MOD_CGBO :
            mod->BSIM3v32cgbo = value->rValue;
            mod->BSIM3v32cgboGiven = TRUE;
            break;
        case  BSIM3v32_MOD_XPART :
            mod->BSIM3v32xpart = value->rValue;
            mod->BSIM3v32xpartGiven = TRUE;
            break;
        case  BSIM3v32_MOD_RSH :
            mod->BSIM3v32sheetResistance = value->rValue;
            mod->BSIM3v32sheetResistanceGiven = TRUE;
            break;
        case  BSIM3v32_MOD_JS :
            mod->BSIM3v32jctSatCurDensity = value->rValue;
            mod->BSIM3v32jctSatCurDensityGiven = TRUE;
            break;
        case  BSIM3v32_MOD_JSW :
            mod->BSIM3v32jctSidewallSatCurDensity = value->rValue;
            mod->BSIM3v32jctSidewallSatCurDensityGiven = TRUE;
            break;
        case  BSIM3v32_MOD_PB :
            mod->BSIM3v32bulkJctPotential = value->rValue;
            mod->BSIM3v32bulkJctPotentialGiven = TRUE;
            break;
        case  BSIM3v32_MOD_MJ :
            mod->BSIM3v32bulkJctBotGradingCoeff = value->rValue;
            mod->BSIM3v32bulkJctBotGradingCoeffGiven = TRUE;
            break;
        case  BSIM3v32_MOD_PBSW :
            mod->BSIM3v32sidewallJctPotential = value->rValue;
            mod->BSIM3v32sidewallJctPotentialGiven = TRUE;
            break;
        case  BSIM3v32_MOD_MJSW :
            mod->BSIM3v32bulkJctSideGradingCoeff = value->rValue;
            mod->BSIM3v32bulkJctSideGradingCoeffGiven = TRUE;
            break;
        case  BSIM3v32_MOD_CJ :
            mod->BSIM3v32unitAreaJctCap = value->rValue;
            mod->BSIM3v32unitAreaJctCapGiven = TRUE;
            break;
        case  BSIM3v32_MOD_CJSW :
            mod->BSIM3v32unitLengthSidewallJctCap = value->rValue;
            mod->BSIM3v32unitLengthSidewallJctCapGiven = TRUE;
            break;
        case  BSIM3v32_MOD_NJ :
            mod->BSIM3v32jctEmissionCoeff = value->rValue;
            mod->BSIM3v32jctEmissionCoeffGiven = TRUE;
            break;
        case  BSIM3v32_MOD_PBSWG :
            mod->BSIM3v32GatesidewallJctPotential = value->rValue;
            mod->BSIM3v32GatesidewallJctPotentialGiven = TRUE;
            break;
        case  BSIM3v32_MOD_MJSWG :
            mod->BSIM3v32bulkJctGateSideGradingCoeff = value->rValue;
            mod->BSIM3v32bulkJctGateSideGradingCoeffGiven = TRUE;
            break;
        case  BSIM3v32_MOD_CJSWG :
            mod->BSIM3v32unitLengthGateSidewallJctCap = value->rValue;
            mod->BSIM3v32unitLengthGateSidewallJctCapGiven = TRUE;
            break;
        case  BSIM3v32_MOD_XTI :
            mod->BSIM3v32jctTempExponent = value->rValue;
            mod->BSIM3v32jctTempExponentGiven = TRUE;
            break;
        case  BSIM3v32_MOD_LINT :
            mod->BSIM3v32Lint = value->rValue;
            mod->BSIM3v32LintGiven = TRUE;
            break;
        case  BSIM3v32_MOD_LL :
            mod->BSIM3v32Ll = value->rValue;
            mod->BSIM3v32LlGiven = TRUE;
            break;
        case  BSIM3v32_MOD_LLC :
            mod->BSIM3v32Llc = value->rValue;
            mod->BSIM3v32LlcGiven = TRUE;
            break;
        case  BSIM3v32_MOD_LLN :
            mod->BSIM3v32Lln = value->rValue;
            mod->BSIM3v32LlnGiven = TRUE;
            break;
        case  BSIM3v32_MOD_LW :
            mod->BSIM3v32Lw = value->rValue;
            mod->BSIM3v32LwGiven = TRUE;
            break;
        case  BSIM3v32_MOD_LWC :
            mod->BSIM3v32Lwc = value->rValue;
            mod->BSIM3v32LwcGiven = TRUE;
            break;
        case  BSIM3v32_MOD_LWN :
            mod->BSIM3v32Lwn = value->rValue;
            mod->BSIM3v32LwnGiven = TRUE;
            break;
        case  BSIM3v32_MOD_LWL :
            mod->BSIM3v32Lwl = value->rValue;
            mod->BSIM3v32LwlGiven = TRUE;
            break;
        case  BSIM3v32_MOD_LWLC :
            mod->BSIM3v32Lwlc = value->rValue;
            mod->BSIM3v32LwlcGiven = TRUE;
            break;
        case  BSIM3v32_MOD_LMIN :
            mod->BSIM3v32Lmin = value->rValue;
            mod->BSIM3v32LminGiven = TRUE;
            break;
        case  BSIM3v32_MOD_LMAX :
            mod->BSIM3v32Lmax = value->rValue;
            mod->BSIM3v32LmaxGiven = TRUE;
            break;
        case  BSIM3v32_MOD_WINT :
            mod->BSIM3v32Wint = value->rValue;
            mod->BSIM3v32WintGiven = TRUE;
            break;
        case  BSIM3v32_MOD_WL :
            mod->BSIM3v32Wl = value->rValue;
            mod->BSIM3v32WlGiven = TRUE;
            break;
        case  BSIM3v32_MOD_WLC :
            mod->BSIM3v32Wlc = value->rValue;
            mod->BSIM3v32WlcGiven = TRUE;
            break;
        case  BSIM3v32_MOD_WLN :
            mod->BSIM3v32Wln = value->rValue;
            mod->BSIM3v32WlnGiven = TRUE;
            break;
        case  BSIM3v32_MOD_WW :
            mod->BSIM3v32Ww = value->rValue;
            mod->BSIM3v32WwGiven = TRUE;
            break;
        case  BSIM3v32_MOD_WWC :
            mod->BSIM3v32Wwc = value->rValue;
            mod->BSIM3v32WwcGiven = TRUE;
            break;
        case  BSIM3v32_MOD_WWN :
            mod->BSIM3v32Wwn = value->rValue;
            mod->BSIM3v32WwnGiven = TRUE;
            break;
        case  BSIM3v32_MOD_WWL :
            mod->BSIM3v32Wwl = value->rValue;
            mod->BSIM3v32WwlGiven = TRUE;
            break;
        case  BSIM3v32_MOD_WWLC :
            mod->BSIM3v32Wwlc = value->rValue;
            mod->BSIM3v32WwlcGiven = TRUE;
            break;
        case  BSIM3v32_MOD_WMIN :
            mod->BSIM3v32Wmin = value->rValue;
            mod->BSIM3v32WminGiven = TRUE;
            break;
        case  BSIM3v32_MOD_WMAX :
            mod->BSIM3v32Wmax = value->rValue;
            mod->BSIM3v32WmaxGiven = TRUE;
            break;

       case BSIM3v32_MOD_XL:
            mod->BSIM3v32xl = value->rValue;
            mod->BSIM3v32xlGiven = TRUE;
            break;
       case BSIM3v32_MOD_XW:
            mod->BSIM3v32xw = value->rValue;
            mod->BSIM3v32xwGiven = TRUE;
            break;

        case  BSIM3v32_MOD_NOIA :
            mod->BSIM3v32oxideTrapDensityA = value->rValue;
            mod->BSIM3v32oxideTrapDensityAGiven = TRUE;
            break;
        case  BSIM3v32_MOD_NOIB :
            mod->BSIM3v32oxideTrapDensityB = value->rValue;
            mod->BSIM3v32oxideTrapDensityBGiven = TRUE;
            break;
        case  BSIM3v32_MOD_NOIC :
            mod->BSIM3v32oxideTrapDensityC = value->rValue;
            mod->BSIM3v32oxideTrapDensityCGiven = TRUE;
            break;
        case  BSIM3v32_MOD_EM :
            mod->BSIM3v32em = value->rValue;
            mod->BSIM3v32emGiven = TRUE;
            break;
        case  BSIM3v32_MOD_EF :
            mod->BSIM3v32ef = value->rValue;
            mod->BSIM3v32efGiven = TRUE;
            break;
        case  BSIM3v32_MOD_AF :
            mod->BSIM3v32af = value->rValue;
            mod->BSIM3v32afGiven = TRUE;
            break;
        case  BSIM3v32_MOD_KF :
            mod->BSIM3v32kf = value->rValue;
            mod->BSIM3v32kfGiven = TRUE;
            break;

        case BSIM3v32_MOD_VGS_MAX:
            mod->BSIM3v32vgsMax = value->rValue;
            mod->BSIM3v32vgsMaxGiven = TRUE;
            break;
        case BSIM3v32_MOD_VGD_MAX:
            mod->BSIM3v32vgdMax = value->rValue;
            mod->BSIM3v32vgdMaxGiven = TRUE;
            break;
        case BSIM3v32_MOD_VGB_MAX:
            mod->BSIM3v32vgbMax = value->rValue;
            mod->BSIM3v32vgbMaxGiven = TRUE;
            break;
        case BSIM3v32_MOD_VDS_MAX:
            mod->BSIM3v32vdsMax = value->rValue;
            mod->BSIM3v32vdsMaxGiven = TRUE;
            break;
        case BSIM3v32_MOD_VBS_MAX:
            mod->BSIM3v32vbsMax = value->rValue;
            mod->BSIM3v32vbsMaxGiven = TRUE;
            break;
        case BSIM3v32_MOD_VBD_MAX:
            mod->BSIM3v32vbdMax = value->rValue;
            mod->BSIM3v32vbdMaxGiven = TRUE;
            break;
        case BSIM3v32_MOD_VGSR_MAX:
            mod->BSIM3v32vgsrMax = value->rValue;
            mod->BSIM3v32vgsrMaxGiven = TRUE;
            break;
        case BSIM3v32_MOD_VGDR_MAX:
            mod->BSIM3v32vgdrMax = value->rValue;
            mod->BSIM3v32vgdrMaxGiven = TRUE;
            break;
        case BSIM3v32_MOD_VGBR_MAX:
            mod->BSIM3v32vgbrMax = value->rValue;
            mod->BSIM3v32vgbrMaxGiven = TRUE;
            break;
        case BSIM3v32_MOD_VBSR_MAX:
            mod->BSIM3v32vbsrMax = value->rValue;
            mod->BSIM3v32vbsrMaxGiven = TRUE;
            break;
        case BSIM3v32_MOD_VBDR_MAX:
            mod->BSIM3v32vbdrMax = value->rValue;
            mod->BSIM3v32vbdrMaxGiven = TRUE;
            break;

        case  BSIM3v32_MOD_NMOS  :
            if(value->iValue) {
                mod->BSIM3v32type = 1;
                mod->BSIM3v32typeGiven = TRUE;
            }
            break;
        case  BSIM3v32_MOD_PMOS  :
            if(value->iValue) {
                mod->BSIM3v32type = - 1;
                mod->BSIM3v32typeGiven = TRUE;
            }
            break;
        default:
            return(E_BADPARM);
    }
    return(OK);
}


