/**** BSIM3v3.2.4, Released by Xuemei Xi 12/21/2001 ****/

/**********
 * Copyright 2001 Regents of the University of California. All rights reserved.
 * File: b3set.c of BSIM3v3.2.4
 * Author: 1995 Min-Chie Jeng and Mansun Chan.
 * Author: 1997-1999 Weidong Liu.
 * Author: 2001  Xuemei Xi
 **********/

#include "ngspice/ngspice.h"
#include "ngspice/smpdefs.h"
#include "ngspice/cktdefs.h"
#include "bsim3v32def.h"
#include "ngspice/const.h"
#include "ngspice/sperror.h"
#include "ngspice/devdefs.h"
#include "ngspice/suffix.h"

#define MAX_EXP 5.834617425e14
#define MIN_EXP 1.713908431e-15
#define EXP_THRESHOLD 34.0
#define SMOOTHFACTOR 0.1
#define EPSOX 3.453133e-11
#define EPSSI 1.03594e-10
#define PI 3.141592654
#define Charge_q 1.60219e-19
#define Meter2Micron 1.0e6

int
BSIM3v32setup (SMPmatrix *matrix, GENmodel *inModel, CKTcircuit *ckt,
            int *states)
{
BSIM3v32model *model = (BSIM3v32model*)inModel;
BSIM3v32instance *here;
int error;
CKTnode *tmp;

CKTnode *tmpNode;
IFuid tmpName;

#ifdef USE_OMP
int idx, InstCount;
BSIM3v32instance **InstArray;
#endif

    /*  loop through all the BSIM3v32 device models */
    for( ; model != NULL; model = BSIM3v32nextModel(model))
    {
/* Default value Processing for BSIM3v32 MOSFET Models */
        if (!model->BSIM3v32typeGiven)
            model->BSIM3v32type = NMOS;
        if (!model->BSIM3v32mobModGiven)
            model->BSIM3v32mobMod = 1;
        if (!model->BSIM3v32binUnitGiven)
            model->BSIM3v32binUnit = 1;
        if (!model->BSIM3v32paramChkGiven)
            model->BSIM3v32paramChk = 0;
        if (!model->BSIM3v32capModGiven)
            model->BSIM3v32capMod = 3;
        if (!model->BSIM3v32acmModGiven)
            model->BSIM3v32acmMod = 0;
        if (!model->BSIM3v32calcacmGiven)
            model->BSIM3v32calcacm = 0;
        if (!model->BSIM3v32noiModGiven)
            model->BSIM3v32noiMod = 1;
        if (!model->BSIM3v32nqsModGiven)
            model->BSIM3v32nqsMod = 0;
        else if ((model->BSIM3v32nqsMod != 0) && (model->BSIM3v32nqsMod != 1))
        {   model->BSIM3v32nqsMod = 0;
            printf("Warning: nqsMod has been set to its default value: 0.\n");
        }

        /* If the user does not provide the model revision,
         * we always choose the most recent.
         */
        if (!model->BSIM3v32versionGiven)
                model->BSIM3v32version = copy("3.2.4");

        /* I have added below the code that translate model string
         * into an integer. This trick is meant to speed up the
         * revision testing instruction, since comparing integer
         * is faster than comparing strings.
         * Paolo Nenzi 2002
         */
        if ((!strcmp(model->BSIM3v32version, "3.2.4"))||(!strncmp(model->BSIM3v32version, "3.24", 4)))
                model->BSIM3v32intVersion = BSIM3v32V324;
        else if ((!strcmp(model->BSIM3v32version, "3.2.3"))||(!strncmp(model->BSIM3v32version, "3.23", 4)))
                model->BSIM3v32intVersion = BSIM3v32V323;
        else if ((!strcmp(model->BSIM3v32version, "3.2.2"))||(!strncmp(model->BSIM3v32version, "3.22", 4)))
                model->BSIM3v32intVersion = BSIM3v32V322;
        else if ((!strncmp(model->BSIM3v32version, "3.2", 3))||(!strncmp(model->BSIM3v32version, "3.20", 4)))
                model->BSIM3v32intVersion = BSIM3v32V32;
        else
                model->BSIM3v32intVersion = BSIM3v32V3OLD;
        /* BSIM3v32V3OLD is a placeholder for pre 3.2 revision
         * This model should not be used for pre 3.2 models.
         */

        if (!model->BSIM3v32toxGiven)
            model->BSIM3v32tox = 150.0e-10;
        model->BSIM3v32cox = 3.453133e-11 / model->BSIM3v32tox;
        if (!model->BSIM3v32toxmGiven)
            model->BSIM3v32toxm = model->BSIM3v32tox;

        if (!model->BSIM3v32cdscGiven)
            model->BSIM3v32cdsc = 2.4e-4;   /* unit Q/V/m^2  */
        if (!model->BSIM3v32cdscbGiven)
            model->BSIM3v32cdscb = 0.0;   /* unit Q/V/m^2  */
        if (!model->BSIM3v32cdscdGiven)
            model->BSIM3v32cdscd = 0.0;   /* unit Q/V/m^2  */
        if (!model->BSIM3v32citGiven)
            model->BSIM3v32cit = 0.0;   /* unit Q/V/m^2  */
        if (!model->BSIM3v32nfactorGiven)
            model->BSIM3v32nfactor = 1;
        if (!model->BSIM3v32xjGiven)
            model->BSIM3v32xj = .15e-6;
        if (!model->BSIM3v32vsatGiven)
            model->BSIM3v32vsat = 8.0e4;    /* unit m/s */
        if (!model->BSIM3v32atGiven)
            model->BSIM3v32at = 3.3e4;    /* unit m/s */
        if (!model->BSIM3v32a0Given)
            model->BSIM3v32a0 = 1.0;
        if (!model->BSIM3v32agsGiven)
            model->BSIM3v32ags = 0.0;
        if (!model->BSIM3v32a1Given)
            model->BSIM3v32a1 = 0.0;
        if (!model->BSIM3v32a2Given)
            model->BSIM3v32a2 = 1.0;
        if (!model->BSIM3v32ketaGiven)
            model->BSIM3v32keta = -0.047;    /* unit  / V */
        if (!model->BSIM3v32nsubGiven)
            model->BSIM3v32nsub = 6.0e16;   /* unit 1/cm3 */
        if (!model->BSIM3v32npeakGiven)
            model->BSIM3v32npeak = 1.7e17;   /* unit 1/cm3 */
        if (!model->BSIM3v32ngateGiven)
            model->BSIM3v32ngate = 0;   /* unit 1/cm3 */
        if (!model->BSIM3v32vbmGiven)
            model->BSIM3v32vbm = -3.0;
        if (!model->BSIM3v32xtGiven)
            model->BSIM3v32xt = 1.55e-7;
        if (!model->BSIM3v32kt1Given)
            model->BSIM3v32kt1 = -0.11;      /* unit V */
        if (!model->BSIM3v32kt1lGiven)
            model->BSIM3v32kt1l = 0.0;      /* unit V*m */
        if (!model->BSIM3v32kt2Given)
            model->BSIM3v32kt2 = 0.022;      /* No unit */
        if (!model->BSIM3v32k3Given)
            model->BSIM3v32k3 = 80.0;
        if (!model->BSIM3v32k3bGiven)
            model->BSIM3v32k3b = 0.0;
        if (!model->BSIM3v32w0Given)
            model->BSIM3v32w0 = 2.5e-6;
        if (!model->BSIM3v32nlxGiven)
            model->BSIM3v32nlx = 1.74e-7;
        if (!model->BSIM3v32dvt0Given)
            model->BSIM3v32dvt0 = 2.2;
        if (!model->BSIM3v32dvt1Given)
            model->BSIM3v32dvt1 = 0.53;
        if (!model->BSIM3v32dvt2Given)
            model->BSIM3v32dvt2 = -0.032;   /* unit 1 / V */

        if (!model->BSIM3v32dvt0wGiven)
            model->BSIM3v32dvt0w = 0.0;
        if (!model->BSIM3v32dvt1wGiven)
            model->BSIM3v32dvt1w = 5.3e6;
        if (!model->BSIM3v32dvt2wGiven)
            model->BSIM3v32dvt2w = -0.032;

        if (!model->BSIM3v32droutGiven)
            model->BSIM3v32drout = 0.56;
        if (!model->BSIM3v32dsubGiven)
            model->BSIM3v32dsub = model->BSIM3v32drout;
        if (!model->BSIM3v32vth0Given)
            model->BSIM3v32vth0 = (model->BSIM3v32type == NMOS) ? 0.7 : -0.7;
        if (!model->BSIM3v32uaGiven)
            model->BSIM3v32ua = 2.25e-9;      /* unit m/V */
        if (!model->BSIM3v32ua1Given)
            model->BSIM3v32ua1 = 4.31e-9;      /* unit m/V */
        if (!model->BSIM3v32ubGiven)
            model->BSIM3v32ub = 5.87e-19;     /* unit (m/V)**2 */
        if (!model->BSIM3v32ub1Given)
            model->BSIM3v32ub1 = -7.61e-18;     /* unit (m/V)**2 */
        if (!model->BSIM3v32ucGiven)
            model->BSIM3v32uc = (model->BSIM3v32mobMod == 3) ? -0.0465 : -0.0465e-9;
        if (!model->BSIM3v32uc1Given)
            model->BSIM3v32uc1 = (model->BSIM3v32mobMod == 3) ? -0.056 : -0.056e-9;
        if (!model->BSIM3v32u0Given)
            model->BSIM3v32u0 = (model->BSIM3v32type == NMOS) ? 0.067 : 0.025;
        if (!model->BSIM3v32uteGiven)
            model->BSIM3v32ute = -1.5;
        if (!model->BSIM3v32voffGiven)
            model->BSIM3v32voff = -0.08;
        if (!model->BSIM3v32deltaGiven)
           model->BSIM3v32delta = 0.01;
        if (!model->BSIM3v32rdswGiven)
            model->BSIM3v32rdsw = 0;
        if (!model->BSIM3v32prwgGiven)
            model->BSIM3v32prwg = 0.0;      /* unit 1/V */
        if (!model->BSIM3v32prwbGiven)
            model->BSIM3v32prwb = 0.0;
        if (!model->BSIM3v32prtGiven)
            model->BSIM3v32prt = 0.0;
        if (!model->BSIM3v32eta0Given)
            model->BSIM3v32eta0 = 0.08;      /* no unit  */
        if (!model->BSIM3v32etabGiven)
            model->BSIM3v32etab = -0.07;      /* unit  1/V */
        if (!model->BSIM3v32pclmGiven)
            model->BSIM3v32pclm = 1.3;      /* no unit  */
        if (!model->BSIM3v32pdibl1Given)
            model->BSIM3v32pdibl1 = .39;    /* no unit  */
        if (!model->BSIM3v32pdibl2Given)
            model->BSIM3v32pdibl2 = 0.0086;    /* no unit  */
        if (!model->BSIM3v32pdiblbGiven)
            model->BSIM3v32pdiblb = 0.0;    /* 1/V  */
        if (!model->BSIM3v32pscbe1Given)
            model->BSIM3v32pscbe1 = 4.24e8;
        if (!model->BSIM3v32pscbe2Given)
            model->BSIM3v32pscbe2 = 1.0e-5;
        if (!model->BSIM3v32pvagGiven)
            model->BSIM3v32pvag = 0.0;
        if (!model->BSIM3v32wrGiven)
            model->BSIM3v32wr = 1.0;
        if (!model->BSIM3v32dwgGiven)
            model->BSIM3v32dwg = 0.0;
        if (!model->BSIM3v32dwbGiven)
            model->BSIM3v32dwb = 0.0;
        if (!model->BSIM3v32b0Given)
            model->BSIM3v32b0 = 0.0;
        if (!model->BSIM3v32b1Given)
            model->BSIM3v32b1 = 0.0;
        if (!model->BSIM3v32alpha0Given)
            model->BSIM3v32alpha0 = 0.0;
        if (!model->BSIM3v32alpha1Given)
            model->BSIM3v32alpha1 = 0.0;
        if (!model->BSIM3v32beta0Given)
            model->BSIM3v32beta0 = 30.0;
        if (!model->BSIM3v32ijthGiven)
            model->BSIM3v32ijth = 0.1; /* unit A */

        if (!model->BSIM3v32elmGiven)
            model->BSIM3v32elm = 5.0;
        if (!model->BSIM3v32cgslGiven)
            model->BSIM3v32cgsl = 0.0;
        if (!model->BSIM3v32cgdlGiven)
            model->BSIM3v32cgdl = 0.0;
        if (!model->BSIM3v32ckappaGiven)
            model->BSIM3v32ckappa = 0.6;
        if (!model->BSIM3v32clcGiven)
            model->BSIM3v32clc = 0.1e-6;
        if (!model->BSIM3v32cleGiven)
            model->BSIM3v32cle = 0.6;
        if (!model->BSIM3v32vfbcvGiven)
            model->BSIM3v32vfbcv = -1.0;
        if (!model->BSIM3v32acdeGiven)
            model->BSIM3v32acde = 1.0;
        if (!model->BSIM3v32moinGiven)
            model->BSIM3v32moin = 15.0;
        if (!model->BSIM3v32noffGiven)
            model->BSIM3v32noff = 1.0;
        if (!model->BSIM3v32voffcvGiven)
            model->BSIM3v32voffcv = 0.0;
        if (!model->BSIM3v32tcjGiven)
            model->BSIM3v32tcj = 0.0;
        if (!model->BSIM3v32tpbGiven)
            model->BSIM3v32tpb = 0.0;
        if (!model->BSIM3v32tcjswGiven)
            model->BSIM3v32tcjsw = 0.0;
        if (!model->BSIM3v32tpbswGiven)
            model->BSIM3v32tpbsw = 0.0;
        if (!model->BSIM3v32tcjswgGiven)
            model->BSIM3v32tcjswg = 0.0;
        if (!model->BSIM3v32tpbswgGiven)
            model->BSIM3v32tpbswg = 0.0;

        /* ACM model */
        if (!model->BSIM3v32hdifGiven)
          model->BSIM3v32hdif = 0.0;
        if (!model->BSIM3v32ldifGiven)
          model->BSIM3v32ldif = 0.0;
        if (!model->BSIM3v32ldGiven)
          model->BSIM3v32ld = 0.0;
        if (!model->BSIM3v32rdGiven)
          model->BSIM3v32rd = 0.0;
        if (!model->BSIM3v32rsGiven)
          model->BSIM3v32rs = 0.0;
        if (!model->BSIM3v32rdcGiven)
          model->BSIM3v32rdc = 0.0;
        if (!model->BSIM3v32rscGiven)
          model->BSIM3v32rsc = 0.0;
        if (!model->BSIM3v32wmltGiven)
          model->BSIM3v32wmlt = 1.0;

        if (!model->BSIM3v32lmltGiven)
          model->BSIM3v32lmlt = 1.0;

        /* Length dependence */
        if (!model->BSIM3v32lcdscGiven)
            model->BSIM3v32lcdsc = 0.0;
        if (!model->BSIM3v32lcdscbGiven)
            model->BSIM3v32lcdscb = 0.0;
        if (!model->BSIM3v32lcdscdGiven)
            model->BSIM3v32lcdscd = 0.0;
        if (!model->BSIM3v32lcitGiven)
            model->BSIM3v32lcit = 0.0;
        if (!model->BSIM3v32lnfactorGiven)
            model->BSIM3v32lnfactor = 0.0;
        if (!model->BSIM3v32lxjGiven)
            model->BSIM3v32lxj = 0.0;
        if (!model->BSIM3v32lvsatGiven)
            model->BSIM3v32lvsat = 0.0;
        if (!model->BSIM3v32latGiven)
            model->BSIM3v32lat = 0.0;
        if (!model->BSIM3v32la0Given)
            model->BSIM3v32la0 = 0.0;
        if (!model->BSIM3v32lagsGiven)
            model->BSIM3v32lags = 0.0;
        if (!model->BSIM3v32la1Given)
            model->BSIM3v32la1 = 0.0;
        if (!model->BSIM3v32la2Given)
            model->BSIM3v32la2 = 0.0;
        if (!model->BSIM3v32lketaGiven)
            model->BSIM3v32lketa = 0.0;
        if (!model->BSIM3v32lnsubGiven)
            model->BSIM3v32lnsub = 0.0;
        if (!model->BSIM3v32lnpeakGiven)
            model->BSIM3v32lnpeak = 0.0;
        if (!model->BSIM3v32lngateGiven)
            model->BSIM3v32lngate = 0.0;
        if (!model->BSIM3v32lvbmGiven)
            model->BSIM3v32lvbm = 0.0;
        if (!model->BSIM3v32lxtGiven)
            model->BSIM3v32lxt = 0.0;
        if (!model->BSIM3v32lkt1Given)
            model->BSIM3v32lkt1 = 0.0;
        if (!model->BSIM3v32lkt1lGiven)
            model->BSIM3v32lkt1l = 0.0;
        if (!model->BSIM3v32lkt2Given)
            model->BSIM3v32lkt2 = 0.0;
        if (!model->BSIM3v32lk3Given)
            model->BSIM3v32lk3 = 0.0;
        if (!model->BSIM3v32lk3bGiven)
            model->BSIM3v32lk3b = 0.0;
        if (!model->BSIM3v32lw0Given)
            model->BSIM3v32lw0 = 0.0;
        if (!model->BSIM3v32lnlxGiven)
            model->BSIM3v32lnlx = 0.0;
        if (!model->BSIM3v32ldvt0Given)
            model->BSIM3v32ldvt0 = 0.0;
        if (!model->BSIM3v32ldvt1Given)
            model->BSIM3v32ldvt1 = 0.0;
        if (!model->BSIM3v32ldvt2Given)
            model->BSIM3v32ldvt2 = 0.0;
        if (!model->BSIM3v32ldvt0wGiven)
            model->BSIM3v32ldvt0w = 0.0;
        if (!model->BSIM3v32ldvt1wGiven)
            model->BSIM3v32ldvt1w = 0.0;
        if (!model->BSIM3v32ldvt2wGiven)
            model->BSIM3v32ldvt2w = 0.0;
        if (!model->BSIM3v32ldroutGiven)
            model->BSIM3v32ldrout = 0.0;
        if (!model->BSIM3v32ldsubGiven)
            model->BSIM3v32ldsub = 0.0;
        if (!model->BSIM3v32lvth0Given)
           model->BSIM3v32lvth0 = 0.0;
        if (!model->BSIM3v32luaGiven)
            model->BSIM3v32lua = 0.0;
        if (!model->BSIM3v32lua1Given)
            model->BSIM3v32lua1 = 0.0;
        if (!model->BSIM3v32lubGiven)
            model->BSIM3v32lub = 0.0;
        if (!model->BSIM3v32lub1Given)
            model->BSIM3v32lub1 = 0.0;
        if (!model->BSIM3v32lucGiven)
            model->BSIM3v32luc = 0.0;
        if (!model->BSIM3v32luc1Given)
            model->BSIM3v32luc1 = 0.0;
        if (!model->BSIM3v32lu0Given)
            model->BSIM3v32lu0 = 0.0;
        if (!model->BSIM3v32luteGiven)
            model->BSIM3v32lute = 0.0;
        if (!model->BSIM3v32lvoffGiven)
            model->BSIM3v32lvoff = 0.0;
        if (!model->BSIM3v32ldeltaGiven)
            model->BSIM3v32ldelta = 0.0;
        if (!model->BSIM3v32lrdswGiven)
            model->BSIM3v32lrdsw = 0.0;
        if (!model->BSIM3v32lprwbGiven)
            model->BSIM3v32lprwb = 0.0;
        if (!model->BSIM3v32lprwgGiven)
            model->BSIM3v32lprwg = 0.0;
        if (!model->BSIM3v32lprtGiven)
            model->BSIM3v32lprt = 0.0;
        if (!model->BSIM3v32leta0Given)
            model->BSIM3v32leta0 = 0.0;
        if (!model->BSIM3v32letabGiven)
            model->BSIM3v32letab = -0.0;
        if (!model->BSIM3v32lpclmGiven)
            model->BSIM3v32lpclm = 0.0;
        if (!model->BSIM3v32lpdibl1Given)
            model->BSIM3v32lpdibl1 = 0.0;
        if (!model->BSIM3v32lpdibl2Given)
            model->BSIM3v32lpdibl2 = 0.0;
        if (!model->BSIM3v32lpdiblbGiven)
            model->BSIM3v32lpdiblb = 0.0;
        if (!model->BSIM3v32lpscbe1Given)
            model->BSIM3v32lpscbe1 = 0.0;
        if (!model->BSIM3v32lpscbe2Given)
            model->BSIM3v32lpscbe2 = 0.0;
        if (!model->BSIM3v32lpvagGiven)
            model->BSIM3v32lpvag = 0.0;
        if (!model->BSIM3v32lwrGiven)
            model->BSIM3v32lwr = 0.0;
        if (!model->BSIM3v32ldwgGiven)
            model->BSIM3v32ldwg = 0.0;
        if (!model->BSIM3v32ldwbGiven)
            model->BSIM3v32ldwb = 0.0;
        if (!model->BSIM3v32lb0Given)
            model->BSIM3v32lb0 = 0.0;
        if (!model->BSIM3v32lb1Given)
            model->BSIM3v32lb1 = 0.0;
        if (!model->BSIM3v32lalpha0Given)
            model->BSIM3v32lalpha0 = 0.0;
        if (!model->BSIM3v32lalpha1Given)
            model->BSIM3v32lalpha1 = 0.0;
        if (!model->BSIM3v32lbeta0Given)
            model->BSIM3v32lbeta0 = 0.0;
        if (!model->BSIM3v32lvfbGiven)
            model->BSIM3v32lvfb = 0.0;

        if (!model->BSIM3v32lelmGiven)
            model->BSIM3v32lelm = 0.0;
        if (!model->BSIM3v32lcgslGiven)
            model->BSIM3v32lcgsl = 0.0;
        if (!model->BSIM3v32lcgdlGiven)
            model->BSIM3v32lcgdl = 0.0;
        if (!model->BSIM3v32lckappaGiven)
            model->BSIM3v32lckappa = 0.0;
        if (!model->BSIM3v32lclcGiven)
            model->BSIM3v32lclc = 0.0;
        if (!model->BSIM3v32lcleGiven)
            model->BSIM3v32lcle = 0.0;
        if (!model->BSIM3v32lcfGiven)
            model->BSIM3v32lcf = 0.0;
        if (!model->BSIM3v32lvfbcvGiven)
            model->BSIM3v32lvfbcv = 0.0;
        if (!model->BSIM3v32lacdeGiven)
            model->BSIM3v32lacde = 0.0;
        if (!model->BSIM3v32lmoinGiven)
            model->BSIM3v32lmoin = 0.0;
        if (!model->BSIM3v32lnoffGiven)
            model->BSIM3v32lnoff = 0.0;
        if (!model->BSIM3v32lvoffcvGiven)
            model->BSIM3v32lvoffcv = 0.0;

        /* Width dependence */
        if (!model->BSIM3v32wcdscGiven)
            model->BSIM3v32wcdsc = 0.0;
        if (!model->BSIM3v32wcdscbGiven)
            model->BSIM3v32wcdscb = 0.0;
        if (!model->BSIM3v32wcdscdGiven)
            model->BSIM3v32wcdscd = 0.0;
        if (!model->BSIM3v32wcitGiven)
            model->BSIM3v32wcit = 0.0;
        if (!model->BSIM3v32wnfactorGiven)
            model->BSIM3v32wnfactor = 0.0;
        if (!model->BSIM3v32wxjGiven)
            model->BSIM3v32wxj = 0.0;
        if (!model->BSIM3v32wvsatGiven)
            model->BSIM3v32wvsat = 0.0;
        if (!model->BSIM3v32watGiven)
            model->BSIM3v32wat = 0.0;
        if (!model->BSIM3v32wa0Given)
            model->BSIM3v32wa0 = 0.0;
        if (!model->BSIM3v32wagsGiven)
            model->BSIM3v32wags = 0.0;
        if (!model->BSIM3v32wa1Given)
            model->BSIM3v32wa1 = 0.0;
        if (!model->BSIM3v32wa2Given)
            model->BSIM3v32wa2 = 0.0;
        if (!model->BSIM3v32wketaGiven)
            model->BSIM3v32wketa = 0.0;
        if (!model->BSIM3v32wnsubGiven)
            model->BSIM3v32wnsub = 0.0;
        if (!model->BSIM3v32wnpeakGiven)
            model->BSIM3v32wnpeak = 0.0;
        if (!model->BSIM3v32wngateGiven)
            model->BSIM3v32wngate = 0.0;
        if (!model->BSIM3v32wvbmGiven)
            model->BSIM3v32wvbm = 0.0;
        if (!model->BSIM3v32wxtGiven)
            model->BSIM3v32wxt = 0.0;
        if (!model->BSIM3v32wkt1Given)
            model->BSIM3v32wkt1 = 0.0;
        if (!model->BSIM3v32wkt1lGiven)
            model->BSIM3v32wkt1l = 0.0;
        if (!model->BSIM3v32wkt2Given)
            model->BSIM3v32wkt2 = 0.0;
        if (!model->BSIM3v32wk3Given)
            model->BSIM3v32wk3 = 0.0;
        if (!model->BSIM3v32wk3bGiven)
            model->BSIM3v32wk3b = 0.0;
        if (!model->BSIM3v32ww0Given)
            model->BSIM3v32ww0 = 0.0;
        if (!model->BSIM3v32wnlxGiven)
            model->BSIM3v32wnlx = 0.0;
        if (!model->BSIM3v32wdvt0Given)
            model->BSIM3v32wdvt0 = 0.0;
        if (!model->BSIM3v32wdvt1Given)
            model->BSIM3v32wdvt1 = 0.0;
        if (!model->BSIM3v32wdvt2Given)
            model->BSIM3v32wdvt2 = 0.0;
        if (!model->BSIM3v32wdvt0wGiven)
            model->BSIM3v32wdvt0w = 0.0;
        if (!model->BSIM3v32wdvt1wGiven)
            model->BSIM3v32wdvt1w = 0.0;
        if (!model->BSIM3v32wdvt2wGiven)
            model->BSIM3v32wdvt2w = 0.0;
        if (!model->BSIM3v32wdroutGiven)
            model->BSIM3v32wdrout = 0.0;
        if (!model->BSIM3v32wdsubGiven)
            model->BSIM3v32wdsub = 0.0;
        if (!model->BSIM3v32wvth0Given)
           model->BSIM3v32wvth0 = 0.0;
        if (!model->BSIM3v32wuaGiven)
            model->BSIM3v32wua = 0.0;
        if (!model->BSIM3v32wua1Given)
            model->BSIM3v32wua1 = 0.0;
        if (!model->BSIM3v32wubGiven)
            model->BSIM3v32wub = 0.0;
        if (!model->BSIM3v32wub1Given)
            model->BSIM3v32wub1 = 0.0;
        if (!model->BSIM3v32wucGiven)
            model->BSIM3v32wuc = 0.0;
        if (!model->BSIM3v32wuc1Given)
            model->BSIM3v32wuc1 = 0.0;
        if (!model->BSIM3v32wu0Given)
            model->BSIM3v32wu0 = 0.0;
        if (!model->BSIM3v32wuteGiven)
            model->BSIM3v32wute = 0.0;
        if (!model->BSIM3v32wvoffGiven)
            model->BSIM3v32wvoff = 0.0;
        if (!model->BSIM3v32wdeltaGiven)
            model->BSIM3v32wdelta = 0.0;
        if (!model->BSIM3v32wrdswGiven)
            model->BSIM3v32wrdsw = 0.0;
        if (!model->BSIM3v32wprwbGiven)
            model->BSIM3v32wprwb = 0.0;
        if (!model->BSIM3v32wprwgGiven)
            model->BSIM3v32wprwg = 0.0;
        if (!model->BSIM3v32wprtGiven)
            model->BSIM3v32wprt = 0.0;
        if (!model->BSIM3v32weta0Given)
            model->BSIM3v32weta0 = 0.0;
        if (!model->BSIM3v32wetabGiven)
            model->BSIM3v32wetab = 0.0;
        if (!model->BSIM3v32wpclmGiven)
            model->BSIM3v32wpclm = 0.0;
        if (!model->BSIM3v32wpdibl1Given)
            model->BSIM3v32wpdibl1 = 0.0;
        if (!model->BSIM3v32wpdibl2Given)
            model->BSIM3v32wpdibl2 = 0.0;
        if (!model->BSIM3v32wpdiblbGiven)
            model->BSIM3v32wpdiblb = 0.0;
        if (!model->BSIM3v32wpscbe1Given)
            model->BSIM3v32wpscbe1 = 0.0;
        if (!model->BSIM3v32wpscbe2Given)
            model->BSIM3v32wpscbe2 = 0.0;
        if (!model->BSIM3v32wpvagGiven)
            model->BSIM3v32wpvag = 0.0;
        if (!model->BSIM3v32wwrGiven)
            model->BSIM3v32wwr = 0.0;
        if (!model->BSIM3v32wdwgGiven)
            model->BSIM3v32wdwg = 0.0;
        if (!model->BSIM3v32wdwbGiven)
            model->BSIM3v32wdwb = 0.0;
        if (!model->BSIM3v32wb0Given)
            model->BSIM3v32wb0 = 0.0;
        if (!model->BSIM3v32wb1Given)
            model->BSIM3v32wb1 = 0.0;
        if (!model->BSIM3v32walpha0Given)
            model->BSIM3v32walpha0 = 0.0;
        if (!model->BSIM3v32walpha1Given)
            model->BSIM3v32walpha1 = 0.0;
        if (!model->BSIM3v32wbeta0Given)
            model->BSIM3v32wbeta0 = 0.0;
        if (!model->BSIM3v32wvfbGiven)
            model->BSIM3v32wvfb = 0.0;

        if (!model->BSIM3v32welmGiven)
            model->BSIM3v32welm = 0.0;
        if (!model->BSIM3v32wcgslGiven)
            model->BSIM3v32wcgsl = 0.0;
        if (!model->BSIM3v32wcgdlGiven)
            model->BSIM3v32wcgdl = 0.0;
        if (!model->BSIM3v32wckappaGiven)
            model->BSIM3v32wckappa = 0.0;
        if (!model->BSIM3v32wcfGiven)
            model->BSIM3v32wcf = 0.0;
        if (!model->BSIM3v32wclcGiven)
            model->BSIM3v32wclc = 0.0;
        if (!model->BSIM3v32wcleGiven)
            model->BSIM3v32wcle = 0.0;
        if (!model->BSIM3v32wvfbcvGiven)
            model->BSIM3v32wvfbcv = 0.0;
        if (!model->BSIM3v32wacdeGiven)
            model->BSIM3v32wacde = 0.0;
        if (!model->BSIM3v32wmoinGiven)
            model->BSIM3v32wmoin = 0.0;
        if (!model->BSIM3v32wnoffGiven)
            model->BSIM3v32wnoff = 0.0;
        if (!model->BSIM3v32wvoffcvGiven)
            model->BSIM3v32wvoffcv = 0.0;

        /* Cross-term dependence */
        if (!model->BSIM3v32pcdscGiven)
            model->BSIM3v32pcdsc = 0.0;
        if (!model->BSIM3v32pcdscbGiven)
            model->BSIM3v32pcdscb = 0.0;
        if (!model->BSIM3v32pcdscdGiven)
            model->BSIM3v32pcdscd = 0.0;
        if (!model->BSIM3v32pcitGiven)
            model->BSIM3v32pcit = 0.0;
        if (!model->BSIM3v32pnfactorGiven)
            model->BSIM3v32pnfactor = 0.0;
        if (!model->BSIM3v32pxjGiven)
            model->BSIM3v32pxj = 0.0;
        if (!model->BSIM3v32pvsatGiven)
            model->BSIM3v32pvsat = 0.0;
        if (!model->BSIM3v32patGiven)
            model->BSIM3v32pat = 0.0;
        if (!model->BSIM3v32pa0Given)
            model->BSIM3v32pa0 = 0.0;

        if (!model->BSIM3v32pagsGiven)
            model->BSIM3v32pags = 0.0;
        if (!model->BSIM3v32pa1Given)
            model->BSIM3v32pa1 = 0.0;
        if (!model->BSIM3v32pa2Given)
            model->BSIM3v32pa2 = 0.0;
        if (!model->BSIM3v32pketaGiven)
            model->BSIM3v32pketa = 0.0;
        if (!model->BSIM3v32pnsubGiven)
            model->BSIM3v32pnsub = 0.0;
        if (!model->BSIM3v32pnpeakGiven)
            model->BSIM3v32pnpeak = 0.0;
        if (!model->BSIM3v32pngateGiven)
            model->BSIM3v32pngate = 0.0;
        if (!model->BSIM3v32pvbmGiven)
            model->BSIM3v32pvbm = 0.0;
        if (!model->BSIM3v32pxtGiven)
            model->BSIM3v32pxt = 0.0;
        if (!model->BSIM3v32pkt1Given)
            model->BSIM3v32pkt1 = 0.0;
        if (!model->BSIM3v32pkt1lGiven)
            model->BSIM3v32pkt1l = 0.0;
        if (!model->BSIM3v32pkt2Given)
            model->BSIM3v32pkt2 = 0.0;
        if (!model->BSIM3v32pk3Given)
            model->BSIM3v32pk3 = 0.0;
        if (!model->BSIM3v32pk3bGiven)
            model->BSIM3v32pk3b = 0.0;
        if (!model->BSIM3v32pw0Given)
            model->BSIM3v32pw0 = 0.0;
        if (!model->BSIM3v32pnlxGiven)
            model->BSIM3v32pnlx = 0.0;
        if (!model->BSIM3v32pdvt0Given)
            model->BSIM3v32pdvt0 = 0.0;
        if (!model->BSIM3v32pdvt1Given)
            model->BSIM3v32pdvt1 = 0.0;
        if (!model->BSIM3v32pdvt2Given)
            model->BSIM3v32pdvt2 = 0.0;
        if (!model->BSIM3v32pdvt0wGiven)
            model->BSIM3v32pdvt0w = 0.0;
        if (!model->BSIM3v32pdvt1wGiven)
            model->BSIM3v32pdvt1w = 0.0;
        if (!model->BSIM3v32pdvt2wGiven)
            model->BSIM3v32pdvt2w = 0.0;
        if (!model->BSIM3v32pdroutGiven)
            model->BSIM3v32pdrout = 0.0;
        if (!model->BSIM3v32pdsubGiven)
            model->BSIM3v32pdsub = 0.0;
        if (!model->BSIM3v32pvth0Given)
           model->BSIM3v32pvth0 = 0.0;
        if (!model->BSIM3v32puaGiven)
            model->BSIM3v32pua = 0.0;
        if (!model->BSIM3v32pua1Given)
            model->BSIM3v32pua1 = 0.0;
        if (!model->BSIM3v32pubGiven)
            model->BSIM3v32pub = 0.0;
        if (!model->BSIM3v32pub1Given)
            model->BSIM3v32pub1 = 0.0;
        if (!model->BSIM3v32pucGiven)
            model->BSIM3v32puc = 0.0;
        if (!model->BSIM3v32puc1Given)
            model->BSIM3v32puc1 = 0.0;
        if (!model->BSIM3v32pu0Given)
            model->BSIM3v32pu0 = 0.0;
        if (!model->BSIM3v32puteGiven)
            model->BSIM3v32pute = 0.0;
        if (!model->BSIM3v32pvoffGiven)
            model->BSIM3v32pvoff = 0.0;
        if (!model->BSIM3v32pdeltaGiven)
            model->BSIM3v32pdelta = 0.0;
        if (!model->BSIM3v32prdswGiven)
            model->BSIM3v32prdsw = 0.0;
        if (!model->BSIM3v32pprwbGiven)
            model->BSIM3v32pprwb = 0.0;
        if (!model->BSIM3v32pprwgGiven)
            model->BSIM3v32pprwg = 0.0;
        if (!model->BSIM3v32pprtGiven)
            model->BSIM3v32pprt = 0.0;
        if (!model->BSIM3v32peta0Given)
            model->BSIM3v32peta0 = 0.0;
        if (!model->BSIM3v32petabGiven)
            model->BSIM3v32petab = 0.0;
        if (!model->BSIM3v32ppclmGiven)
            model->BSIM3v32ppclm = 0.0;
        if (!model->BSIM3v32ppdibl1Given)
            model->BSIM3v32ppdibl1 = 0.0;
        if (!model->BSIM3v32ppdibl2Given)
            model->BSIM3v32ppdibl2 = 0.0;
        if (!model->BSIM3v32ppdiblbGiven)
            model->BSIM3v32ppdiblb = 0.0;
        if (!model->BSIM3v32ppscbe1Given)
            model->BSIM3v32ppscbe1 = 0.0;
        if (!model->BSIM3v32ppscbe2Given)
            model->BSIM3v32ppscbe2 = 0.0;
        if (!model->BSIM3v32ppvagGiven)
            model->BSIM3v32ppvag = 0.0;
        if (!model->BSIM3v32pwrGiven)
            model->BSIM3v32pwr = 0.0;
        if (!model->BSIM3v32pdwgGiven)
            model->BSIM3v32pdwg = 0.0;
        if (!model->BSIM3v32pdwbGiven)
            model->BSIM3v32pdwb = 0.0;
        if (!model->BSIM3v32pb0Given)
            model->BSIM3v32pb0 = 0.0;
        if (!model->BSIM3v32pb1Given)
            model->BSIM3v32pb1 = 0.0;
        if (!model->BSIM3v32palpha0Given)
            model->BSIM3v32palpha0 = 0.0;
        if (!model->BSIM3v32palpha1Given)
            model->BSIM3v32palpha1 = 0.0;
        if (!model->BSIM3v32pbeta0Given)
            model->BSIM3v32pbeta0 = 0.0;
        if (!model->BSIM3v32pvfbGiven)
            model->BSIM3v32pvfb = 0.0;

        if (!model->BSIM3v32pelmGiven)
            model->BSIM3v32pelm = 0.0;
        if (!model->BSIM3v32pcgslGiven)
            model->BSIM3v32pcgsl = 0.0;
        if (!model->BSIM3v32pcgdlGiven)
            model->BSIM3v32pcgdl = 0.0;
        if (!model->BSIM3v32pckappaGiven)
            model->BSIM3v32pckappa = 0.0;
        if (!model->BSIM3v32pcfGiven)
            model->BSIM3v32pcf = 0.0;
        if (!model->BSIM3v32pclcGiven)
            model->BSIM3v32pclc = 0.0;
        if (!model->BSIM3v32pcleGiven)
            model->BSIM3v32pcle = 0.0;
        if (!model->BSIM3v32pvfbcvGiven)
            model->BSIM3v32pvfbcv = 0.0;
        if (!model->BSIM3v32pacdeGiven)
            model->BSIM3v32pacde = 0.0;
        if (!model->BSIM3v32pmoinGiven)
            model->BSIM3v32pmoin = 0.0;
        if (!model->BSIM3v32pnoffGiven)
            model->BSIM3v32pnoff = 0.0;
        if (!model->BSIM3v32pvoffcvGiven)
            model->BSIM3v32pvoffcv = 0.0;

        /* unit degree celcius */
        if (!model->BSIM3v32tnomGiven)
            model->BSIM3v32tnom = ckt->CKTnomTemp;
/*        else
            model->BSIM3v32tnom = model->BSIM3v32tnom + 273.15; we make this transform in b3v32mpar.c in the first run */
        if (!model->BSIM3v32LintGiven)
           model->BSIM3v32Lint = 0.0;
        if (!model->BSIM3v32LlGiven)
           model->BSIM3v32Ll = 0.0;
        if (!model->BSIM3v32LlcGiven)
           model->BSIM3v32Llc = model->BSIM3v32Ll;
        if (!model->BSIM3v32LlnGiven)
           model->BSIM3v32Lln = 1.0;
        if (!model->BSIM3v32LwGiven)
           model->BSIM3v32Lw = 0.0;
        if (!model->BSIM3v32LwcGiven)
           model->BSIM3v32Lwc = model->BSIM3v32Lw;
        if (!model->BSIM3v32LwnGiven)
           model->BSIM3v32Lwn = 1.0;
        if (!model->BSIM3v32LwlGiven)
           model->BSIM3v32Lwl = 0.0;
        if (!model->BSIM3v32LwlcGiven)
           model->BSIM3v32Lwlc = model->BSIM3v32Lwl;
        if (!model->BSIM3v32LminGiven)
           model->BSIM3v32Lmin = 0.0;
        if (!model->BSIM3v32LmaxGiven)
           model->BSIM3v32Lmax = 1.0;
        if (!model->BSIM3v32WintGiven)
           model->BSIM3v32Wint = 0.0;
        if (!model->BSIM3v32WlGiven)
           model->BSIM3v32Wl = 0.0;
        if (!model->BSIM3v32WlcGiven)
           model->BSIM3v32Wlc = model->BSIM3v32Wl;
        if (!model->BSIM3v32WlnGiven)
           model->BSIM3v32Wln = 1.0;
        if (!model->BSIM3v32WwGiven)
           model->BSIM3v32Ww = 0.0;
        if (!model->BSIM3v32WwcGiven)
           model->BSIM3v32Wwc = model->BSIM3v32Ww;
        if (!model->BSIM3v32WwnGiven)
           model->BSIM3v32Wwn = 1.0;
        if (!model->BSIM3v32WwlGiven)
           model->BSIM3v32Wwl = 0.0;
        if (!model->BSIM3v32WwlcGiven)
           model->BSIM3v32Wwlc = model->BSIM3v32Wwl;
        if (!model->BSIM3v32WminGiven)
           model->BSIM3v32Wmin = 0.0;
        if (!model->BSIM3v32WmaxGiven)
           model->BSIM3v32Wmax = 1.0;
        if (!model->BSIM3v32dwcGiven)
           model->BSIM3v32dwc = model->BSIM3v32Wint;
        if (!model->BSIM3v32dlcGiven)
           model->BSIM3v32dlc = model->BSIM3v32Lint;

        if (!model->BSIM3v32xlGiven)
           model->BSIM3v32xl = 0.0;
        if (!model->BSIM3v32xwGiven)
           model->BSIM3v32xw = 0.0;

        if (!model->BSIM3v32cfGiven)
            model->BSIM3v32cf = 2.0 * EPSOX / PI
                           * log(1.0 + 0.4e-6 / model->BSIM3v32tox);
        if (!model->BSIM3v32cgdoGiven)
        {   if (model->BSIM3v32dlcGiven && (model->BSIM3v32dlc > 0.0))
            {   model->BSIM3v32cgdo = model->BSIM3v32dlc * model->BSIM3v32cox
                                 - model->BSIM3v32cgdl ;
            }
            else
                model->BSIM3v32cgdo = 0.6 * model->BSIM3v32xj * model->BSIM3v32cox;
        }
        if (!model->BSIM3v32cgsoGiven)
        {   if (model->BSIM3v32dlcGiven && (model->BSIM3v32dlc > 0.0))
            {   model->BSIM3v32cgso = model->BSIM3v32dlc * model->BSIM3v32cox
                                 - model->BSIM3v32cgsl ;
            }
            else
                model->BSIM3v32cgso = 0.6 * model->BSIM3v32xj * model->BSIM3v32cox;
        }

        if (!model->BSIM3v32cgboGiven)
        {   model->BSIM3v32cgbo = 2.0 * model->BSIM3v32dwc * model->BSIM3v32cox;
        }
        if (!model->BSIM3v32xpartGiven)
            model->BSIM3v32xpart = 0.0;
        if (!model->BSIM3v32sheetResistanceGiven)
            model->BSIM3v32sheetResistance = 0.0;
        if (!model->BSIM3v32unitAreaJctCapGiven)
            model->BSIM3v32unitAreaJctCap = 5.0E-4;
        if (!model->BSIM3v32unitLengthSidewallJctCapGiven)
            model->BSIM3v32unitLengthSidewallJctCap = 5.0E-10;
        if (!model->BSIM3v32unitLengthGateSidewallJctCapGiven)
            model->BSIM3v32unitLengthGateSidewallJctCap = model->BSIM3v32unitLengthSidewallJctCap ;
        if (!model->BSIM3v32jctSatCurDensityGiven)
            model->BSIM3v32jctSatCurDensity = 1.0E-4;
        if (!model->BSIM3v32jctSidewallSatCurDensityGiven)
            model->BSIM3v32jctSidewallSatCurDensity = 0.0;
        if (!model->BSIM3v32bulkJctPotentialGiven)
            model->BSIM3v32bulkJctPotential = 1.0;
        if (!model->BSIM3v32sidewallJctPotentialGiven)
            model->BSIM3v32sidewallJctPotential = 1.0;
        if (!model->BSIM3v32GatesidewallJctPotentialGiven)
            model->BSIM3v32GatesidewallJctPotential = model->BSIM3v32sidewallJctPotential;
        if (!model->BSIM3v32bulkJctBotGradingCoeffGiven)
            model->BSIM3v32bulkJctBotGradingCoeff = 0.5;
        if (!model->BSIM3v32bulkJctSideGradingCoeffGiven)
            model->BSIM3v32bulkJctSideGradingCoeff = 0.33;
        if (!model->BSIM3v32bulkJctGateSideGradingCoeffGiven)
            model->BSIM3v32bulkJctGateSideGradingCoeff = model->BSIM3v32bulkJctSideGradingCoeff;
        if (!model->BSIM3v32jctEmissionCoeffGiven)
            model->BSIM3v32jctEmissionCoeff = 1.0;
        if (!model->BSIM3v32jctTempExponentGiven)
            model->BSIM3v32jctTempExponent = 3.0;
        if (!model->BSIM3v32oxideTrapDensityAGiven)
        {   if (model->BSIM3v32type == NMOS)
                model->BSIM3v32oxideTrapDensityA = 1e20;
            else
                model->BSIM3v32oxideTrapDensityA=9.9e18;
        }
        if (!model->BSIM3v32oxideTrapDensityBGiven)
        {   if (model->BSIM3v32type == NMOS)
                model->BSIM3v32oxideTrapDensityB = 5e4;
            else
                model->BSIM3v32oxideTrapDensityB = 2.4e3;
        }
        if (!model->BSIM3v32oxideTrapDensityCGiven)
        {   if (model->BSIM3v32type == NMOS)
                model->BSIM3v32oxideTrapDensityC = -1.4e-12;
            else
                model->BSIM3v32oxideTrapDensityC = 1.4e-12;

        }
        if (!model->BSIM3v32emGiven)
            model->BSIM3v32em = 4.1e7; /* V/m */
        if (!model->BSIM3v32efGiven)
            model->BSIM3v32ef = 1.0;
        if (!model->BSIM3v32afGiven)
            model->BSIM3v32af = 1.0;
        if (!model->BSIM3v32kfGiven)
            model->BSIM3v32kf = 0.0;

        if (!model->BSIM3v32vgsMaxGiven)
            model->BSIM3v32vgsMax = 1e99;
        if (!model->BSIM3v32vgdMaxGiven)
            model->BSIM3v32vgdMax = 1e99;
        if (!model->BSIM3v32vgbMaxGiven)
            model->BSIM3v32vgbMax = 1e99;
        if (!model->BSIM3v32vdsMaxGiven)
            model->BSIM3v32vdsMax = 1e99;
        if (!model->BSIM3v32vbsMaxGiven)
            model->BSIM3v32vbsMax = 1e99;
        if (!model->BSIM3v32vbdMaxGiven)
            model->BSIM3v32vbdMax = 1e99;
        if (!model->BSIM3v32vgsrMaxGiven)
            model->BSIM3v32vgsrMax = 1e99;
        if (!model->BSIM3v32vgdrMaxGiven)
            model->BSIM3v32vgdrMax = 1e99;
        if (!model->BSIM3v32vgbrMaxGiven)
            model->BSIM3v32vgbrMax = 1e99;
        if (!model->BSIM3v32vbsrMaxGiven)
            model->BSIM3v32vbsrMax = 1e99;
        if (!model->BSIM3v32vbdrMaxGiven)
            model->BSIM3v32vbdrMax = 1e99;

        /* loop through all the instances of the model */
        for (here = BSIM3v32instances(model); here != NULL ;
             here=BSIM3v32nextInstance(here))
        {
            /* allocate a chunk of the state vector */
            here->BSIM3v32states = *states;
            *states += BSIM3v32numStates;

            /* perform the parameter defaulting */
            if (!here->BSIM3v32drainAreaGiven)
                here->BSIM3v32drainArea = 0.0;
            if (!here->BSIM3v32drainPerimeterGiven)
                here->BSIM3v32drainPerimeter = 0.0;
            if (!here->BSIM3v32drainSquaresGiven)
            {
                if (model->BSIM3v32acmMod == 0)
                  here->BSIM3v32drainSquares = 1.0;
                else
                  here->BSIM3v32drainSquares = 0.0;
            }
            if (!here->BSIM3v32delvtoGiven)
                here->BSIM3v32delvto = 0.0;
            if (!here->BSIM3v32mulu0Given)
                here->BSIM3v32mulu0 = 1.0;
            if (!here->BSIM3v32icVBSGiven)
                here->BSIM3v32icVBS = 0.0;
            if (!here->BSIM3v32icVDSGiven)
                here->BSIM3v32icVDS = 0.0;
            if (!here->BSIM3v32icVGSGiven)
                here->BSIM3v32icVGS = 0.0;
            if (!here->BSIM3v32lGiven)
                here->BSIM3v32l = 5.0e-6;
            if (!here->BSIM3v32sourceAreaGiven)
                here->BSIM3v32sourceArea = 0.0;
            if (!here->BSIM3v32sourcePerimeterGiven)
                here->BSIM3v32sourcePerimeter = 0.0;
            if (!here->BSIM3v32sourceSquaresGiven)
            {
                if (model->BSIM3v32acmMod == 0)
                  here->BSIM3v32sourceSquares = 1.0;
                else
                  here->BSIM3v32sourceSquares = 0.0;
            }
            if (!here->BSIM3v32wGiven)
                here->BSIM3v32w = 5.0e-6;
            if (!here->BSIM3v32nqsModGiven)
                here->BSIM3v32nqsMod = model->BSIM3v32nqsMod;
            else if ((here->BSIM3v32nqsMod != 0) && (here->BSIM3v32nqsMod != 1))
            {   here->BSIM3v32nqsMod = model->BSIM3v32nqsMod;
                printf("Warning: nqsMod has been set to its global value %d.\n",
                model->BSIM3v32nqsMod);
            }
            if (!here->BSIM3v32geoGiven)
                here->BSIM3v32geo = 0;

            if (!here->BSIM3v32mGiven)
                here->BSIM3v32m = 1;

            /* process source/drain series resistance */
            /* ACM model */

            double DrainResistance, SourceResistance;

            if (model->BSIM3v32acmMod == 0)
            {
                DrainResistance = model->BSIM3v32sheetResistance
                    * here->BSIM3v32drainSquares;
                SourceResistance = model->BSIM3v32sheetResistance
                    * here->BSIM3v32sourceSquares;
            }
            else /* ACM > 0 */
            {
                error = ACM_SourceDrainResistances(
                    model->BSIM3v32acmMod,
                    model->BSIM3v32ld,
                    model->BSIM3v32ldif,
                    model->BSIM3v32hdif,
                    model->BSIM3v32wmlt,
                    here->BSIM3v32w,
                    model->BSIM3v32xw,
                    model->BSIM3v32sheetResistance,
                    here->BSIM3v32drainSquaresGiven,
                    model->BSIM3v32rd,
                    model->BSIM3v32rdc,
                    here->BSIM3v32drainSquares,
                    here->BSIM3v32sourceSquaresGiven,
                    model->BSIM3v32rs,
                    model->BSIM3v32rsc,
                    here->BSIM3v32sourceSquares,
                    &DrainResistance,
                    &SourceResistance
                    );
                if (error)
                    return(error);
            }

            /* process drain series resistance */
            if (DrainResistance != 0)
            {
               if(here->BSIM3v32dNodePrime == 0) {
                 error = CKTmkVolt(ckt,&tmp,here->BSIM3v32name,"drain");
                 if(error) return(error);
                 here->BSIM3v32dNodePrime = tmp->number;
                 if (ckt->CKTcopyNodesets) {
                   if (CKTinst2Node(ckt,here,1,&tmpNode,&tmpName)==OK) {
                      if (tmpNode->nsGiven) {
                        tmp->nodeset=tmpNode->nodeset;
                        tmp->nsGiven=tmpNode->nsGiven;
                      }
                   }
                 }
               }
            }
            else
            {  here->BSIM3v32dNodePrime = here->BSIM3v32dNode;
            }

            /* process source series resistance */
            if (SourceResistance != 0)
            {
               if(here->BSIM3v32sNodePrime == 0) {
                 error = CKTmkVolt(ckt,&tmp,here->BSIM3v32name,"source");
                 if(error) return(error);
                 here->BSIM3v32sNodePrime = tmp->number;
                 if (ckt->CKTcopyNodesets) {
                   if (CKTinst2Node(ckt,here,3,&tmpNode,&tmpName)==OK) {
                      if (tmpNode->nsGiven) {
                        tmp->nodeset=tmpNode->nodeset;
                        tmp->nsGiven=tmpNode->nsGiven;
                      }
                   }
                 }
               }
            }
            else
            {  here->BSIM3v32sNodePrime = here->BSIM3v32sNode;
            }

            /* internal charge node */

            if (here->BSIM3v32nqsMod)
            {   if(here->BSIM3v32qNode == 0)
                {   error = CKTmkVolt(ckt,&tmp,here->BSIM3v32name,"charge");
                    if(error) return(error);
                    here->BSIM3v32qNode = tmp->number;
                }
            }
            else
            {   here->BSIM3v32qNode = 0;
            }

            /* Channel length scaling with lmlt model parameter */

            if (model->BSIM3v32lmltGiven)
                here->BSIM3v32l *= model->BSIM3v32lmlt;

        /* set Sparse Matrix Pointers */

/* macro to make elements with built in test for out of memory */
#define TSTALLOC(ptr,first,second) \
do { if((here->ptr = SMPmakeElt(matrix, here->first, here->second)) == NULL){\
    return(E_NOMEM);\
} } while(0)

            TSTALLOC(BSIM3v32DdPtr, BSIM3v32dNode, BSIM3v32dNode);
            TSTALLOC(BSIM3v32GgPtr, BSIM3v32gNode, BSIM3v32gNode);
            TSTALLOC(BSIM3v32SsPtr, BSIM3v32sNode, BSIM3v32sNode);
            TSTALLOC(BSIM3v32BbPtr, BSIM3v32bNode, BSIM3v32bNode);
            TSTALLOC(BSIM3v32DPdpPtr, BSIM3v32dNodePrime, BSIM3v32dNodePrime);
            TSTALLOC(BSIM3v32SPspPtr, BSIM3v32sNodePrime, BSIM3v32sNodePrime);
            TSTALLOC(BSIM3v32DdpPtr, BSIM3v32dNode, BSIM3v32dNodePrime);
            TSTALLOC(BSIM3v32GbPtr, BSIM3v32gNode, BSIM3v32bNode);
            TSTALLOC(BSIM3v32GdpPtr, BSIM3v32gNode, BSIM3v32dNodePrime);
            TSTALLOC(BSIM3v32GspPtr, BSIM3v32gNode, BSIM3v32sNodePrime);
            TSTALLOC(BSIM3v32SspPtr, BSIM3v32sNode, BSIM3v32sNodePrime);
            TSTALLOC(BSIM3v32BdpPtr, BSIM3v32bNode, BSIM3v32dNodePrime);
            TSTALLOC(BSIM3v32BspPtr, BSIM3v32bNode, BSIM3v32sNodePrime);
            TSTALLOC(BSIM3v32DPspPtr, BSIM3v32dNodePrime, BSIM3v32sNodePrime);
            TSTALLOC(BSIM3v32DPdPtr, BSIM3v32dNodePrime, BSIM3v32dNode);
            TSTALLOC(BSIM3v32BgPtr, BSIM3v32bNode, BSIM3v32gNode);
            TSTALLOC(BSIM3v32DPgPtr, BSIM3v32dNodePrime, BSIM3v32gNode);
            TSTALLOC(BSIM3v32SPgPtr, BSIM3v32sNodePrime, BSIM3v32gNode);
            TSTALLOC(BSIM3v32SPsPtr, BSIM3v32sNodePrime, BSIM3v32sNode);
            TSTALLOC(BSIM3v32DPbPtr, BSIM3v32dNodePrime, BSIM3v32bNode);
            TSTALLOC(BSIM3v32SPbPtr, BSIM3v32sNodePrime, BSIM3v32bNode);
            TSTALLOC(BSIM3v32SPdpPtr, BSIM3v32sNodePrime, BSIM3v32dNodePrime);

            TSTALLOC(BSIM3v32QqPtr, BSIM3v32qNode, BSIM3v32qNode);

            TSTALLOC(BSIM3v32QdpPtr, BSIM3v32qNode, BSIM3v32dNodePrime);
            TSTALLOC(BSIM3v32QspPtr, BSIM3v32qNode, BSIM3v32sNodePrime);
            TSTALLOC(BSIM3v32QgPtr, BSIM3v32qNode, BSIM3v32gNode);
            TSTALLOC(BSIM3v32QbPtr, BSIM3v32qNode, BSIM3v32bNode);
            TSTALLOC(BSIM3v32DPqPtr, BSIM3v32dNodePrime, BSIM3v32qNode);
            TSTALLOC(BSIM3v32SPqPtr, BSIM3v32sNodePrime, BSIM3v32qNode);
            TSTALLOC(BSIM3v32GqPtr, BSIM3v32gNode, BSIM3v32qNode);
            TSTALLOC(BSIM3v32BqPtr, BSIM3v32bNode, BSIM3v32qNode);

        }
    }
#ifdef USE_OMP
    InstCount = 0;
    model = (BSIM3v32model*)inModel;
    /* loop through all the BSIM3 device models
    to count the number of instances */

    for (; model != NULL; model = BSIM3v32nextModel(model))
    {
        /* loop through all the instances of the model */
        for (here = BSIM3v32instances(model); here != NULL;
             here = BSIM3v32nextInstance(here))
        {
            InstCount++;
        }
        model->BSIM3v32InstCount = 0;
        model->BSIM3v32InstanceArray = NULL;
    }
    InstArray = TMALLOC(BSIM3v32instance*, InstCount);
    model = (BSIM3v32model*)inModel;
    /* store this in the first model only */
    model->BSIM3v32InstCount = InstCount;
    model->BSIM3v32InstanceArray = InstArray;
    idx = 0;
    for (; model != NULL; model = BSIM3v32nextModel(model))
    {
        /* loop through all the instances of the model */
        for (here = BSIM3v32instances(model); here != NULL;
             here = BSIM3v32nextInstance(here))
        {
            InstArray[idx] = here;
            idx++;
        }
    }

#endif
    return(OK);
}

int
BSIM3v32unsetup(
    GENmodel *inModel,
    CKTcircuit *ckt)
{
    BSIM3v32model *model;
    BSIM3v32instance *here;

#ifdef USE_OMP
    model = (BSIM3v32model*)inModel;
    tfree(model->BSIM3v32InstanceArray);
#endif

    for (model = (BSIM3v32model *)inModel; model != NULL;
            model = BSIM3v32nextModel(model))
    {
        for (here = BSIM3v32instances(model); here != NULL;
                here=BSIM3v32nextInstance(here))
        {
            if (here->BSIM3v32qNode > 0)
                CKTdltNNum(ckt, here->BSIM3v32qNode);
            here->BSIM3v32qNode = 0;

            if (here->BSIM3v32sNodePrime > 0
                    && here->BSIM3v32sNodePrime != here->BSIM3v32sNode)
                CKTdltNNum(ckt, here->BSIM3v32sNodePrime);
            here->BSIM3v32sNodePrime = 0;

            if (here->BSIM3v32dNodePrime > 0
                    && here->BSIM3v32dNodePrime != here->BSIM3v32dNode)
                CKTdltNNum(ckt, here->BSIM3v32dNodePrime);
            here->BSIM3v32dNodePrime = 0;
        }
    }
    return OK;
}






