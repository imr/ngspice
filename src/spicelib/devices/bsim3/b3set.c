/**** BSIM3v3.3.0, Released by Xuemei Xi 07/29/2005 ****/
/**** OpenMP support for ngspice by Holger Vogt 06/28/2010 ****/

/**********
 * Copyright 2004 Regents of the University of California. All rights reserved.
 * File: b3set.c of BSIM3v3.3.0
 * Author: 1995 Min-Chie Jeng and Mansun Chan.
 * Author: 1997-1999 Weidong Liu.
 * Author: 2001  Xuemei Xi
 **********/

#include "ngspice/ngspice.h"
#include "ngspice/smpdefs.h"
#include "ngspice/cktdefs.h"
#include "bsim3def.h"
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
BSIM3setup(
SMPmatrix *matrix,
GENmodel *inModel,
CKTcircuit *ckt,
int *states)
{
BSIM3model *model = (BSIM3model*)inModel;
BSIM3instance *here;
int error;
CKTnode *tmp;
CKTnode *tmpNode;
IFuid tmpName;

#ifdef USE_OMP
int idx, InstCount;
BSIM3instance **InstArray;
#endif


    /*  loop through all the BSIM3 device models */
    for( ; model != NULL; model = BSIM3nextModel(model))
    {
/* Default value Processing for BSIM3 MOSFET Models */
        if (!model->BSIM3typeGiven)
            model->BSIM3type = NMOS;
        if (!model->BSIM3mobModGiven)
            model->BSIM3mobMod = 1;
        if (!model->BSIM3binUnitGiven)
            model->BSIM3binUnit = 1;
        if (!model->BSIM3paramChkGiven)
            model->BSIM3paramChk = 0;
        if (!model->BSIM3capModGiven)
            model->BSIM3capMod = 3;
        if (!model->BSIM3acmModGiven)
            model->BSIM3acmMod = 0;
        if (!model->BSIM3calcacmGiven)
            model->BSIM3calcacm = 0;
        if (!model->BSIM3noiModGiven)
            model->BSIM3noiMod = 1;
        if (!model->BSIM3nqsModGiven)
            model->BSIM3nqsMod = 0;
        else if ((model->BSIM3nqsMod != 0) && (model->BSIM3nqsMod != 1))
        {   model->BSIM3nqsMod = 0;
            printf("Warning: nqsMod has been set to its default value: 0.\n");
        }
        if (!model->BSIM3acnqsModGiven)
            model->BSIM3acnqsMod = 0;
        else if ((model->BSIM3acnqsMod != 0) && (model->BSIM3acnqsMod != 1))
        {   model->BSIM3acnqsMod = 0;
            printf("Warning: acnqsMod has been set to its default value: 0.\n");
        }
        if (!model->BSIM3versionGiven)
            model->BSIM3version = copy("3.3.0");
        if (!model->BSIM3toxGiven)
            model->BSIM3tox = 150.0e-10;
        model->BSIM3cox = 3.453133e-11 / model->BSIM3tox;
        if (!model->BSIM3toxmGiven)
            model->BSIM3toxm = model->BSIM3tox;

        if (!model->BSIM3cdscGiven)
            model->BSIM3cdsc = 2.4e-4;   /* unit Q/V/m^2  */
        if (!model->BSIM3cdscbGiven)
            model->BSIM3cdscb = 0.0;   /* unit Q/V/m^2  */
        if (!model->BSIM3cdscdGiven)
            model->BSIM3cdscd = 0.0;   /* unit Q/V/m^2  */
        if (!model->BSIM3citGiven)
            model->BSIM3cit = 0.0;   /* unit Q/V/m^2  */
        if (!model->BSIM3nfactorGiven)
            model->BSIM3nfactor = 1;
        if (!model->BSIM3xjGiven)
            model->BSIM3xj = .15e-6;
        if (!model->BSIM3vsatGiven)
            model->BSIM3vsat = 8.0e4;    /* unit m/s */
        if (!model->BSIM3atGiven)
            model->BSIM3at = 3.3e4;    /* unit m/s */
        if (!model->BSIM3a0Given)
            model->BSIM3a0 = 1.0;
        if (!model->BSIM3agsGiven)
            model->BSIM3ags = 0.0;
        if (!model->BSIM3a1Given)
            model->BSIM3a1 = 0.0;
        if (!model->BSIM3a2Given)
            model->BSIM3a2 = 1.0;
        if (!model->BSIM3ketaGiven)
            model->BSIM3keta = -0.047;    /* unit  / V */
        if (!model->BSIM3nsubGiven)
            model->BSIM3nsub = 6.0e16;   /* unit 1/cm3 */
        if (!model->BSIM3npeakGiven)
            model->BSIM3npeak = 1.7e17;   /* unit 1/cm3 */
        if (!model->BSIM3ngateGiven)
            model->BSIM3ngate = 0;   /* unit 1/cm3 */
        if (!model->BSIM3vbmGiven)
            model->BSIM3vbm = -3.0;
        if (!model->BSIM3xtGiven)
            model->BSIM3xt = 1.55e-7;
        if (!model->BSIM3kt1Given)
            model->BSIM3kt1 = -0.11;      /* unit V */
        if (!model->BSIM3kt1lGiven)
            model->BSIM3kt1l = 0.0;      /* unit V*m */
        if (!model->BSIM3kt2Given)
            model->BSIM3kt2 = 0.022;      /* No unit */
        if (!model->BSIM3k3Given)
            model->BSIM3k3 = 80.0;
        if (!model->BSIM3k3bGiven)
            model->BSIM3k3b = 0.0;
        if (!model->BSIM3w0Given)
            model->BSIM3w0 = 2.5e-6;
        if (!model->BSIM3nlxGiven)
            model->BSIM3nlx = 1.74e-7;
        if (!model->BSIM3dvt0Given)
            model->BSIM3dvt0 = 2.2;
        if (!model->BSIM3dvt1Given)
            model->BSIM3dvt1 = 0.53;
        if (!model->BSIM3dvt2Given)
            model->BSIM3dvt2 = -0.032;   /* unit 1 / V */

        if (!model->BSIM3dvt0wGiven)
            model->BSIM3dvt0w = 0.0;
        if (!model->BSIM3dvt1wGiven)
            model->BSIM3dvt1w = 5.3e6;
        if (!model->BSIM3dvt2wGiven)
            model->BSIM3dvt2w = -0.032;

        if (!model->BSIM3droutGiven)
            model->BSIM3drout = 0.56;
        if (!model->BSIM3dsubGiven)
            model->BSIM3dsub = model->BSIM3drout;
        if (!model->BSIM3vth0Given)
            model->BSIM3vth0 = (model->BSIM3type == NMOS) ? 0.7 : -0.7;
        if (!model->BSIM3uaGiven)
            model->BSIM3ua = 2.25e-9;      /* unit m/V */
        if (!model->BSIM3ua1Given)
            model->BSIM3ua1 = 4.31e-9;      /* unit m/V */
        if (!model->BSIM3ubGiven)
            model->BSIM3ub = 5.87e-19;     /* unit (m/V)**2 */
        if (!model->BSIM3ub1Given)
            model->BSIM3ub1 = -7.61e-18;     /* unit (m/V)**2 */
        if (!model->BSIM3ucGiven)
            model->BSIM3uc = (model->BSIM3mobMod == 3) ? -0.0465 : -0.0465e-9;
        if (!model->BSIM3uc1Given)
            model->BSIM3uc1 = (model->BSIM3mobMod == 3) ? -0.056 : -0.056e-9;
        if (!model->BSIM3u0Given)
            model->BSIM3u0 = (model->BSIM3type == NMOS) ? 0.067 : 0.025;
        if (!model->BSIM3uteGiven)
            model->BSIM3ute = -1.5;
        if (!model->BSIM3voffGiven)
            model->BSIM3voff = -0.08;
        if (!model->BSIM3deltaGiven)
           model->BSIM3delta = 0.01;
        if (!model->BSIM3rdswGiven)
            model->BSIM3rdsw = 0;
        if (!model->BSIM3prwgGiven)
            model->BSIM3prwg = 0.0;      /* unit 1/V */
        if (!model->BSIM3prwbGiven)
            model->BSIM3prwb = 0.0;
        if (!model->BSIM3prtGiven)
            model->BSIM3prt = 0.0;
        if (!model->BSIM3eta0Given)
            model->BSIM3eta0 = 0.08;      /* no unit  */
        if (!model->BSIM3etabGiven)
            model->BSIM3etab = -0.07;      /* unit  1/V */
        if (!model->BSIM3pclmGiven)
            model->BSIM3pclm = 1.3;      /* no unit  */
        if (!model->BSIM3pdibl1Given)
            model->BSIM3pdibl1 = .39;    /* no unit  */
        if (!model->BSIM3pdibl2Given)
            model->BSIM3pdibl2 = 0.0086;    /* no unit  */
        if (!model->BSIM3pdiblbGiven)
            model->BSIM3pdiblb = 0.0;    /* 1/V  */
        if (!model->BSIM3pscbe1Given)
            model->BSIM3pscbe1 = 4.24e8;
        if (!model->BSIM3pscbe2Given)
            model->BSIM3pscbe2 = 1.0e-5;
        if (!model->BSIM3pvagGiven)
            model->BSIM3pvag = 0.0;
        if (!model->BSIM3wrGiven)
            model->BSIM3wr = 1.0;
        if (!model->BSIM3dwgGiven)
            model->BSIM3dwg = 0.0;
        if (!model->BSIM3dwbGiven)
            model->BSIM3dwb = 0.0;
        if (!model->BSIM3b0Given)
            model->BSIM3b0 = 0.0;
        if (!model->BSIM3b1Given)
            model->BSIM3b1 = 0.0;
        if (!model->BSIM3alpha0Given)
            model->BSIM3alpha0 = 0.0;
        if (!model->BSIM3alpha1Given)
            model->BSIM3alpha1 = 0.0;
        if (!model->BSIM3beta0Given)
            model->BSIM3beta0 = 30.0;
        if (!model->BSIM3ijthGiven)
            model->BSIM3ijth = 0.1; /* unit A */

        if (!model->BSIM3elmGiven)
            model->BSIM3elm = 5.0;
        if (!model->BSIM3cgslGiven)
            model->BSIM3cgsl = 0.0;
        if (!model->BSIM3cgdlGiven)
            model->BSIM3cgdl = 0.0;
        if (!model->BSIM3ckappaGiven)
            model->BSIM3ckappa = 0.6;
        if (!model->BSIM3clcGiven)
            model->BSIM3clc = 0.1e-6;
        if (!model->BSIM3cleGiven)
            model->BSIM3cle = 0.6;
        if (!model->BSIM3vfbcvGiven)
            model->BSIM3vfbcv = -1.0;
        if (!model->BSIM3acdeGiven)
            model->BSIM3acde = 1.0;
        if (!model->BSIM3moinGiven)
            model->BSIM3moin = 15.0;
        if (!model->BSIM3noffGiven)
            model->BSIM3noff = 1.0;
        if (!model->BSIM3voffcvGiven)
            model->BSIM3voffcv = 0.0;
        if (!model->BSIM3tcjGiven)
            model->BSIM3tcj = 0.0;
        if (!model->BSIM3tpbGiven)
            model->BSIM3tpb = 0.0;
        if (!model->BSIM3tcjswGiven)
            model->BSIM3tcjsw = 0.0;
        if (!model->BSIM3tpbswGiven)
            model->BSIM3tpbsw = 0.0;
        if (!model->BSIM3tcjswgGiven)
            model->BSIM3tcjswg = 0.0;
        if (!model->BSIM3tpbswgGiven)
            model->BSIM3tpbswg = 0.0;

        /* ACM model */
        if (!model->BSIM3hdifGiven)
          model->BSIM3hdif = 0.0;
        if (!model->BSIM3ldifGiven)
          model->BSIM3ldif = 0.0;
        if (!model->BSIM3ldGiven)
          model->BSIM3ld = 0.0;
        if (!model->BSIM3rdGiven)
          model->BSIM3rd = 0.0;
        if (!model->BSIM3rsGiven)
          model->BSIM3rs = 0.0;
        if (!model->BSIM3rdcGiven)
          model->BSIM3rdc = 0.0;
        if (!model->BSIM3rscGiven)
          model->BSIM3rsc = 0.0;
        if (!model->BSIM3wmltGiven)
          model->BSIM3wmlt = 1.0;

        /* Length dependence */
        if (!model->BSIM3lcdscGiven)
            model->BSIM3lcdsc = 0.0;
        if (!model->BSIM3lcdscbGiven)
            model->BSIM3lcdscb = 0.0;
        if (!model->BSIM3lcdscdGiven)
            model->BSIM3lcdscd = 0.0;
        if (!model->BSIM3lcitGiven)
            model->BSIM3lcit = 0.0;
        if (!model->BSIM3lnfactorGiven)
            model->BSIM3lnfactor = 0.0;
        if (!model->BSIM3lxjGiven)
            model->BSIM3lxj = 0.0;
        if (!model->BSIM3lvsatGiven)
            model->BSIM3lvsat = 0.0;
        if (!model->BSIM3latGiven)
            model->BSIM3lat = 0.0;
        if (!model->BSIM3la0Given)
            model->BSIM3la0 = 0.0;
        if (!model->BSIM3lagsGiven)
            model->BSIM3lags = 0.0;
        if (!model->BSIM3la1Given)
            model->BSIM3la1 = 0.0;
        if (!model->BSIM3la2Given)
            model->BSIM3la2 = 0.0;
        if (!model->BSIM3lketaGiven)
            model->BSIM3lketa = 0.0;
        if (!model->BSIM3lnsubGiven)
            model->BSIM3lnsub = 0.0;
        if (!model->BSIM3lnpeakGiven)
            model->BSIM3lnpeak = 0.0;
        if (!model->BSIM3lngateGiven)
            model->BSIM3lngate = 0.0;
        if (!model->BSIM3lvbmGiven)
            model->BSIM3lvbm = 0.0;
        if (!model->BSIM3lxtGiven)
            model->BSIM3lxt = 0.0;
        if (!model->BSIM3lkt1Given)
            model->BSIM3lkt1 = 0.0;
        if (!model->BSIM3lkt1lGiven)
            model->BSIM3lkt1l = 0.0;
        if (!model->BSIM3lkt2Given)
            model->BSIM3lkt2 = 0.0;
        if (!model->BSIM3lk3Given)
            model->BSIM3lk3 = 0.0;
        if (!model->BSIM3lk3bGiven)
            model->BSIM3lk3b = 0.0;
        if (!model->BSIM3lw0Given)
            model->BSIM3lw0 = 0.0;
        if (!model->BSIM3lnlxGiven)
            model->BSIM3lnlx = 0.0;
        if (!model->BSIM3ldvt0Given)
            model->BSIM3ldvt0 = 0.0;
        if (!model->BSIM3ldvt1Given)
            model->BSIM3ldvt1 = 0.0;
        if (!model->BSIM3ldvt2Given)
            model->BSIM3ldvt2 = 0.0;
        if (!model->BSIM3ldvt0wGiven)
            model->BSIM3ldvt0w = 0.0;
        if (!model->BSIM3ldvt1wGiven)
            model->BSIM3ldvt1w = 0.0;
        if (!model->BSIM3ldvt2wGiven)
            model->BSIM3ldvt2w = 0.0;
        if (!model->BSIM3ldroutGiven)
            model->BSIM3ldrout = 0.0;
        if (!model->BSIM3ldsubGiven)
            model->BSIM3ldsub = 0.0;
        if (!model->BSIM3lvth0Given)
           model->BSIM3lvth0 = 0.0;
        if (!model->BSIM3luaGiven)
            model->BSIM3lua = 0.0;
        if (!model->BSIM3lua1Given)
            model->BSIM3lua1 = 0.0;
        if (!model->BSIM3lubGiven)
            model->BSIM3lub = 0.0;
        if (!model->BSIM3lub1Given)
            model->BSIM3lub1 = 0.0;
        if (!model->BSIM3lucGiven)
            model->BSIM3luc = 0.0;
        if (!model->BSIM3luc1Given)
            model->BSIM3luc1 = 0.0;
        if (!model->BSIM3lu0Given)
            model->BSIM3lu0 = 0.0;
        if (!model->BSIM3luteGiven)
            model->BSIM3lute = 0.0;
        if (!model->BSIM3lvoffGiven)
            model->BSIM3lvoff = 0.0;
        if (!model->BSIM3ldeltaGiven)
            model->BSIM3ldelta = 0.0;
        if (!model->BSIM3lrdswGiven)
            model->BSIM3lrdsw = 0.0;
        if (!model->BSIM3lprwbGiven)
            model->BSIM3lprwb = 0.0;
        if (!model->BSIM3lprwgGiven)
            model->BSIM3lprwg = 0.0;
        if (!model->BSIM3lprtGiven)
            model->BSIM3lprt = 0.0;
        if (!model->BSIM3leta0Given)
            model->BSIM3leta0 = 0.0;
        if (!model->BSIM3letabGiven)
            model->BSIM3letab = -0.0;
        if (!model->BSIM3lpclmGiven)
            model->BSIM3lpclm = 0.0;
        if (!model->BSIM3lpdibl1Given)
            model->BSIM3lpdibl1 = 0.0;
        if (!model->BSIM3lpdibl2Given)
            model->BSIM3lpdibl2 = 0.0;
        if (!model->BSIM3lpdiblbGiven)
            model->BSIM3lpdiblb = 0.0;
        if (!model->BSIM3lpscbe1Given)
            model->BSIM3lpscbe1 = 0.0;
        if (!model->BSIM3lpscbe2Given)
            model->BSIM3lpscbe2 = 0.0;
        if (!model->BSIM3lpvagGiven)
            model->BSIM3lpvag = 0.0;
        if (!model->BSIM3lwrGiven)
            model->BSIM3lwr = 0.0;
        if (!model->BSIM3ldwgGiven)
            model->BSIM3ldwg = 0.0;
        if (!model->BSIM3ldwbGiven)
            model->BSIM3ldwb = 0.0;
        if (!model->BSIM3lb0Given)
            model->BSIM3lb0 = 0.0;
        if (!model->BSIM3lb1Given)
            model->BSIM3lb1 = 0.0;
        if (!model->BSIM3lalpha0Given)
            model->BSIM3lalpha0 = 0.0;
        if (!model->BSIM3lalpha1Given)
            model->BSIM3lalpha1 = 0.0;
        if (!model->BSIM3lbeta0Given)
            model->BSIM3lbeta0 = 0.0;
        if (!model->BSIM3lvfbGiven)
            model->BSIM3lvfb = 0.0;

        if (!model->BSIM3lelmGiven)
            model->BSIM3lelm = 0.0;
        if (!model->BSIM3lcgslGiven)
            model->BSIM3lcgsl = 0.0;
        if (!model->BSIM3lcgdlGiven)
            model->BSIM3lcgdl = 0.0;
        if (!model->BSIM3lckappaGiven)
            model->BSIM3lckappa = 0.0;
        if (!model->BSIM3lclcGiven)
            model->BSIM3lclc = 0.0;
        if (!model->BSIM3lcleGiven)
            model->BSIM3lcle = 0.0;
        if (!model->BSIM3lcfGiven)
            model->BSIM3lcf = 0.0;
        if (!model->BSIM3lvfbcvGiven)
            model->BSIM3lvfbcv = 0.0;
        if (!model->BSIM3lacdeGiven)
            model->BSIM3lacde = 0.0;
        if (!model->BSIM3lmoinGiven)
            model->BSIM3lmoin = 0.0;
        if (!model->BSIM3lnoffGiven)
            model->BSIM3lnoff = 0.0;
        if (!model->BSIM3lvoffcvGiven)
            model->BSIM3lvoffcv = 0.0;

        /* Width dependence */
        if (!model->BSIM3wcdscGiven)
            model->BSIM3wcdsc = 0.0;
        if (!model->BSIM3wcdscbGiven)
            model->BSIM3wcdscb = 0.0;
        if (!model->BSIM3wcdscdGiven)
            model->BSIM3wcdscd = 0.0;
        if (!model->BSIM3wcitGiven)
            model->BSIM3wcit = 0.0;
        if (!model->BSIM3wnfactorGiven)
            model->BSIM3wnfactor = 0.0;
        if (!model->BSIM3wxjGiven)
            model->BSIM3wxj = 0.0;
        if (!model->BSIM3wvsatGiven)
            model->BSIM3wvsat = 0.0;
        if (!model->BSIM3watGiven)
            model->BSIM3wat = 0.0;
        if (!model->BSIM3wa0Given)
            model->BSIM3wa0 = 0.0;
        if (!model->BSIM3wagsGiven)
            model->BSIM3wags = 0.0;
        if (!model->BSIM3wa1Given)
            model->BSIM3wa1 = 0.0;
        if (!model->BSIM3wa2Given)
            model->BSIM3wa2 = 0.0;
        if (!model->BSIM3wketaGiven)
            model->BSIM3wketa = 0.0;
        if (!model->BSIM3wnsubGiven)
            model->BSIM3wnsub = 0.0;
        if (!model->BSIM3wnpeakGiven)
            model->BSIM3wnpeak = 0.0;
        if (!model->BSIM3wngateGiven)
            model->BSIM3wngate = 0.0;
        if (!model->BSIM3wvbmGiven)
            model->BSIM3wvbm = 0.0;
        if (!model->BSIM3wxtGiven)
            model->BSIM3wxt = 0.0;
        if (!model->BSIM3wkt1Given)
            model->BSIM3wkt1 = 0.0;
        if (!model->BSIM3wkt1lGiven)
            model->BSIM3wkt1l = 0.0;
        if (!model->BSIM3wkt2Given)
            model->BSIM3wkt2 = 0.0;
        if (!model->BSIM3wk3Given)
            model->BSIM3wk3 = 0.0;
        if (!model->BSIM3wk3bGiven)
            model->BSIM3wk3b = 0.0;
        if (!model->BSIM3ww0Given)
            model->BSIM3ww0 = 0.0;
        if (!model->BSIM3wnlxGiven)
            model->BSIM3wnlx = 0.0;
        if (!model->BSIM3wdvt0Given)
            model->BSIM3wdvt0 = 0.0;
        if (!model->BSIM3wdvt1Given)
            model->BSIM3wdvt1 = 0.0;
        if (!model->BSIM3wdvt2Given)
            model->BSIM3wdvt2 = 0.0;
        if (!model->BSIM3wdvt0wGiven)
            model->BSIM3wdvt0w = 0.0;
        if (!model->BSIM3wdvt1wGiven)
            model->BSIM3wdvt1w = 0.0;
        if (!model->BSIM3wdvt2wGiven)
            model->BSIM3wdvt2w = 0.0;
        if (!model->BSIM3wdroutGiven)
            model->BSIM3wdrout = 0.0;
        if (!model->BSIM3wdsubGiven)
            model->BSIM3wdsub = 0.0;
        if (!model->BSIM3wvth0Given)
           model->BSIM3wvth0 = 0.0;
        if (!model->BSIM3wuaGiven)
            model->BSIM3wua = 0.0;
        if (!model->BSIM3wua1Given)
            model->BSIM3wua1 = 0.0;
        if (!model->BSIM3wubGiven)
            model->BSIM3wub = 0.0;
        if (!model->BSIM3wub1Given)
            model->BSIM3wub1 = 0.0;
        if (!model->BSIM3wucGiven)
            model->BSIM3wuc = 0.0;
        if (!model->BSIM3wuc1Given)
            model->BSIM3wuc1 = 0.0;
        if (!model->BSIM3wu0Given)
            model->BSIM3wu0 = 0.0;
        if (!model->BSIM3wuteGiven)
            model->BSIM3wute = 0.0;
        if (!model->BSIM3wvoffGiven)
            model->BSIM3wvoff = 0.0;
        if (!model->BSIM3wdeltaGiven)
            model->BSIM3wdelta = 0.0;
        if (!model->BSIM3wrdswGiven)
            model->BSIM3wrdsw = 0.0;
        if (!model->BSIM3wprwbGiven)
            model->BSIM3wprwb = 0.0;
        if (!model->BSIM3wprwgGiven)
            model->BSIM3wprwg = 0.0;
        if (!model->BSIM3wprtGiven)
            model->BSIM3wprt = 0.0;
        if (!model->BSIM3weta0Given)
            model->BSIM3weta0 = 0.0;
        if (!model->BSIM3wetabGiven)
            model->BSIM3wetab = 0.0;
        if (!model->BSIM3wpclmGiven)
            model->BSIM3wpclm = 0.0;
        if (!model->BSIM3wpdibl1Given)
            model->BSIM3wpdibl1 = 0.0;
        if (!model->BSIM3wpdibl2Given)
            model->BSIM3wpdibl2 = 0.0;
        if (!model->BSIM3wpdiblbGiven)
            model->BSIM3wpdiblb = 0.0;
        if (!model->BSIM3wpscbe1Given)
            model->BSIM3wpscbe1 = 0.0;
        if (!model->BSIM3wpscbe2Given)
            model->BSIM3wpscbe2 = 0.0;
        if (!model->BSIM3wpvagGiven)
            model->BSIM3wpvag = 0.0;
        if (!model->BSIM3wwrGiven)
            model->BSIM3wwr = 0.0;
        if (!model->BSIM3wdwgGiven)
            model->BSIM3wdwg = 0.0;
        if (!model->BSIM3wdwbGiven)
            model->BSIM3wdwb = 0.0;
        if (!model->BSIM3wb0Given)
            model->BSIM3wb0 = 0.0;
        if (!model->BSIM3wb1Given)
            model->BSIM3wb1 = 0.0;
        if (!model->BSIM3walpha0Given)
            model->BSIM3walpha0 = 0.0;
        if (!model->BSIM3walpha1Given)
            model->BSIM3walpha1 = 0.0;
        if (!model->BSIM3wbeta0Given)
            model->BSIM3wbeta0 = 0.0;
        if (!model->BSIM3wvfbGiven)
            model->BSIM3wvfb = 0.0;

        if (!model->BSIM3welmGiven)
            model->BSIM3welm = 0.0;
        if (!model->BSIM3wcgslGiven)
            model->BSIM3wcgsl = 0.0;
        if (!model->BSIM3wcgdlGiven)
            model->BSIM3wcgdl = 0.0;
        if (!model->BSIM3wckappaGiven)
            model->BSIM3wckappa = 0.0;
        if (!model->BSIM3wcfGiven)
            model->BSIM3wcf = 0.0;
        if (!model->BSIM3wclcGiven)
            model->BSIM3wclc = 0.0;
        if (!model->BSIM3wcleGiven)
            model->BSIM3wcle = 0.0;
        if (!model->BSIM3wvfbcvGiven)
            model->BSIM3wvfbcv = 0.0;
        if (!model->BSIM3wacdeGiven)
            model->BSIM3wacde = 0.0;
        if (!model->BSIM3wmoinGiven)
            model->BSIM3wmoin = 0.0;
        if (!model->BSIM3wnoffGiven)
            model->BSIM3wnoff = 0.0;
        if (!model->BSIM3wvoffcvGiven)
            model->BSIM3wvoffcv = 0.0;

        /* Cross-term dependence */
        if (!model->BSIM3pcdscGiven)
            model->BSIM3pcdsc = 0.0;
        if (!model->BSIM3pcdscbGiven)
            model->BSIM3pcdscb = 0.0;
        if (!model->BSIM3pcdscdGiven)
            model->BSIM3pcdscd = 0.0;
        if (!model->BSIM3pcitGiven)
            model->BSIM3pcit = 0.0;
        if (!model->BSIM3pnfactorGiven)
            model->BSIM3pnfactor = 0.0;
        if (!model->BSIM3pxjGiven)
            model->BSIM3pxj = 0.0;
        if (!model->BSIM3pvsatGiven)
            model->BSIM3pvsat = 0.0;
        if (!model->BSIM3patGiven)
            model->BSIM3pat = 0.0;
        if (!model->BSIM3pa0Given)
            model->BSIM3pa0 = 0.0;

        if (!model->BSIM3pagsGiven)
            model->BSIM3pags = 0.0;
        if (!model->BSIM3pa1Given)
            model->BSIM3pa1 = 0.0;
        if (!model->BSIM3pa2Given)
            model->BSIM3pa2 = 0.0;
        if (!model->BSIM3pketaGiven)
            model->BSIM3pketa = 0.0;
        if (!model->BSIM3pnsubGiven)
            model->BSIM3pnsub = 0.0;
        if (!model->BSIM3pnpeakGiven)
            model->BSIM3pnpeak = 0.0;
        if (!model->BSIM3pngateGiven)
            model->BSIM3pngate = 0.0;
        if (!model->BSIM3pvbmGiven)
            model->BSIM3pvbm = 0.0;
        if (!model->BSIM3pxtGiven)
            model->BSIM3pxt = 0.0;
        if (!model->BSIM3pkt1Given)
            model->BSIM3pkt1 = 0.0;
        if (!model->BSIM3pkt1lGiven)
            model->BSIM3pkt1l = 0.0;
        if (!model->BSIM3pkt2Given)
            model->BSIM3pkt2 = 0.0;
        if (!model->BSIM3pk3Given)
            model->BSIM3pk3 = 0.0;
        if (!model->BSIM3pk3bGiven)
            model->BSIM3pk3b = 0.0;
        if (!model->BSIM3pw0Given)
            model->BSIM3pw0 = 0.0;
        if (!model->BSIM3pnlxGiven)
            model->BSIM3pnlx = 0.0;
        if (!model->BSIM3pdvt0Given)
            model->BSIM3pdvt0 = 0.0;
        if (!model->BSIM3pdvt1Given)
            model->BSIM3pdvt1 = 0.0;
        if (!model->BSIM3pdvt2Given)
            model->BSIM3pdvt2 = 0.0;
        if (!model->BSIM3pdvt0wGiven)
            model->BSIM3pdvt0w = 0.0;
        if (!model->BSIM3pdvt1wGiven)
            model->BSIM3pdvt1w = 0.0;
        if (!model->BSIM3pdvt2wGiven)
            model->BSIM3pdvt2w = 0.0;
        if (!model->BSIM3pdroutGiven)
            model->BSIM3pdrout = 0.0;
        if (!model->BSIM3pdsubGiven)
            model->BSIM3pdsub = 0.0;
        if (!model->BSIM3pvth0Given)
           model->BSIM3pvth0 = 0.0;
        if (!model->BSIM3puaGiven)
            model->BSIM3pua = 0.0;
        if (!model->BSIM3pua1Given)
            model->BSIM3pua1 = 0.0;
        if (!model->BSIM3pubGiven)
            model->BSIM3pub = 0.0;
        if (!model->BSIM3pub1Given)
            model->BSIM3pub1 = 0.0;
        if (!model->BSIM3pucGiven)
            model->BSIM3puc = 0.0;
        if (!model->BSIM3puc1Given)
            model->BSIM3puc1 = 0.0;
        if (!model->BSIM3pu0Given)
            model->BSIM3pu0 = 0.0;
        if (!model->BSIM3puteGiven)
            model->BSIM3pute = 0.0;
        if (!model->BSIM3pvoffGiven)
            model->BSIM3pvoff = 0.0;
        if (!model->BSIM3pdeltaGiven)
            model->BSIM3pdelta = 0.0;
        if (!model->BSIM3prdswGiven)
            model->BSIM3prdsw = 0.0;
        if (!model->BSIM3pprwbGiven)
            model->BSIM3pprwb = 0.0;
        if (!model->BSIM3pprwgGiven)
            model->BSIM3pprwg = 0.0;
        if (!model->BSIM3pprtGiven)
            model->BSIM3pprt = 0.0;
        if (!model->BSIM3peta0Given)
            model->BSIM3peta0 = 0.0;
        if (!model->BSIM3petabGiven)
            model->BSIM3petab = 0.0;
        if (!model->BSIM3ppclmGiven)
            model->BSIM3ppclm = 0.0;
        if (!model->BSIM3ppdibl1Given)
            model->BSIM3ppdibl1 = 0.0;
        if (!model->BSIM3ppdibl2Given)
            model->BSIM3ppdibl2 = 0.0;
        if (!model->BSIM3ppdiblbGiven)
            model->BSIM3ppdiblb = 0.0;
        if (!model->BSIM3ppscbe1Given)
            model->BSIM3ppscbe1 = 0.0;
        if (!model->BSIM3ppscbe2Given)
            model->BSIM3ppscbe2 = 0.0;
        if (!model->BSIM3ppvagGiven)
            model->BSIM3ppvag = 0.0;
        if (!model->BSIM3pwrGiven)
            model->BSIM3pwr = 0.0;
        if (!model->BSIM3pdwgGiven)
            model->BSIM3pdwg = 0.0;
        if (!model->BSIM3pdwbGiven)
            model->BSIM3pdwb = 0.0;
        if (!model->BSIM3pb0Given)
            model->BSIM3pb0 = 0.0;
        if (!model->BSIM3pb1Given)
            model->BSIM3pb1 = 0.0;
        if (!model->BSIM3palpha0Given)
            model->BSIM3palpha0 = 0.0;
        if (!model->BSIM3palpha1Given)
            model->BSIM3palpha1 = 0.0;
        if (!model->BSIM3pbeta0Given)
            model->BSIM3pbeta0 = 0.0;
        if (!model->BSIM3pvfbGiven)
            model->BSIM3pvfb = 0.0;

        if (!model->BSIM3pelmGiven)
            model->BSIM3pelm = 0.0;
        if (!model->BSIM3pcgslGiven)
            model->BSIM3pcgsl = 0.0;
        if (!model->BSIM3pcgdlGiven)
            model->BSIM3pcgdl = 0.0;
        if (!model->BSIM3pckappaGiven)
            model->BSIM3pckappa = 0.0;
        if (!model->BSIM3pcfGiven)
            model->BSIM3pcf = 0.0;
        if (!model->BSIM3pclcGiven)
            model->BSIM3pclc = 0.0;
        if (!model->BSIM3pcleGiven)
            model->BSIM3pcle = 0.0;
        if (!model->BSIM3pvfbcvGiven)
            model->BSIM3pvfbcv = 0.0;
        if (!model->BSIM3pacdeGiven)
            model->BSIM3pacde = 0.0;
        if (!model->BSIM3pmoinGiven)
            model->BSIM3pmoin = 0.0;
        if (!model->BSIM3pnoffGiven)
            model->BSIM3pnoff = 0.0;
        if (!model->BSIM3pvoffcvGiven)
            model->BSIM3pvoffcv = 0.0;

        /* unit degree celcius */
        if (!model->BSIM3tnomGiven)
            model->BSIM3tnom = ckt->CKTnomTemp;
/*        else
            model->BSIM3tnom = model->BSIM3tnom + 273.15; we make this transform in b3mpar.c in the first run */
        if (!model->BSIM3lintnoiGiven)
            model->BSIM3lintnoi = 0.0;  /* unit m */
        if (!model->BSIM3LintGiven)
           model->BSIM3Lint = 0.0;
        if (!model->BSIM3LlGiven)
           model->BSIM3Ll = 0.0;
        if (!model->BSIM3LlcGiven)
           model->BSIM3Llc = model->BSIM3Ll;
        if (!model->BSIM3LlnGiven)
           model->BSIM3Lln = 1.0;
        if (!model->BSIM3LwGiven)
           model->BSIM3Lw = 0.0;
        if (!model->BSIM3LwcGiven)
           model->BSIM3Lwc = model->BSIM3Lw;
        if (!model->BSIM3LwnGiven)
           model->BSIM3Lwn = 1.0;
        if (!model->BSIM3LwlGiven)
           model->BSIM3Lwl = 0.0;
        if (!model->BSIM3LwlcGiven)
           model->BSIM3Lwlc = model->BSIM3Lwl;
        if (!model->BSIM3LminGiven)
           model->BSIM3Lmin = 0.0;
        if (!model->BSIM3LmaxGiven)
           model->BSIM3Lmax = 1.0;
        if (!model->BSIM3WintGiven)
           model->BSIM3Wint = 0.0;
        if (!model->BSIM3WlGiven)
           model->BSIM3Wl = 0.0;
        if (!model->BSIM3WlcGiven)
           model->BSIM3Wlc = model->BSIM3Wl;
        if (!model->BSIM3WlnGiven)
           model->BSIM3Wln = 1.0;
        if (!model->BSIM3WwGiven)
           model->BSIM3Ww = 0.0;
        if (!model->BSIM3WwcGiven)
           model->BSIM3Wwc = model->BSIM3Ww;
        if (!model->BSIM3WwnGiven)
           model->BSIM3Wwn = 1.0;
        if (!model->BSIM3WwlGiven)
           model->BSIM3Wwl = 0.0;
        if (!model->BSIM3WwlcGiven)
           model->BSIM3Wwlc = model->BSIM3Wwl;
        if (!model->BSIM3WminGiven)
           model->BSIM3Wmin = 0.0;
        if (!model->BSIM3WmaxGiven)
           model->BSIM3Wmax = 1.0;
        if (!model->BSIM3dwcGiven)
           model->BSIM3dwc = model->BSIM3Wint;
        if (!model->BSIM3dlcGiven)
           model->BSIM3dlc = model->BSIM3Lint;

        if (!model->BSIM3xlGiven)
           model->BSIM3xl = 0.0;
        if (!model->BSIM3xwGiven)
           model->BSIM3xw = 0.0;

        if (!model->BSIM3cfGiven)
            model->BSIM3cf = 2.0 * EPSOX / PI
                           * log(1.0 + 0.4e-6 / model->BSIM3tox);
        if (!model->BSIM3cgdoGiven)
        {   if (model->BSIM3dlcGiven && (model->BSIM3dlc > 0.0))
            {   model->BSIM3cgdo = model->BSIM3dlc * model->BSIM3cox
                                 - model->BSIM3cgdl ;
            }
            else
                model->BSIM3cgdo = 0.6 * model->BSIM3xj * model->BSIM3cox;
        }
        if (!model->BSIM3cgsoGiven)
        {   if (model->BSIM3dlcGiven && (model->BSIM3dlc > 0.0))
            {   model->BSIM3cgso = model->BSIM3dlc * model->BSIM3cox
                                 - model->BSIM3cgsl ;
            }
            else
                model->BSIM3cgso = 0.6 * model->BSIM3xj * model->BSIM3cox;
        }

        if (!model->BSIM3cgboGiven)
        {   model->BSIM3cgbo = 2.0 * model->BSIM3dwc * model->BSIM3cox;
        }
        if (!model->BSIM3xpartGiven)
            model->BSIM3xpart = 0.0;
        if (!model->BSIM3sheetResistanceGiven)
            model->BSIM3sheetResistance = 0.0;
        if (!model->BSIM3unitAreaJctCapGiven)
            model->BSIM3unitAreaJctCap = 5.0E-4;
        if (!model->BSIM3unitLengthSidewallJctCapGiven)
            model->BSIM3unitLengthSidewallJctCap = 5.0E-10;
        if (!model->BSIM3unitLengthGateSidewallJctCapGiven)
            model->BSIM3unitLengthGateSidewallJctCap = model->BSIM3unitLengthSidewallJctCap ;
        if (!model->BSIM3jctSatCurDensityGiven)
            model->BSIM3jctSatCurDensity = 1.0E-4;
        if (!model->BSIM3jctSidewallSatCurDensityGiven)
            model->BSIM3jctSidewallSatCurDensity = 0.0;
        if (!model->BSIM3bulkJctPotentialGiven)
            model->BSIM3bulkJctPotential = 1.0;
        if (!model->BSIM3sidewallJctPotentialGiven)
            model->BSIM3sidewallJctPotential = 1.0;
        if (!model->BSIM3GatesidewallJctPotentialGiven)
            model->BSIM3GatesidewallJctPotential = model->BSIM3sidewallJctPotential;
        if (!model->BSIM3bulkJctBotGradingCoeffGiven)
            model->BSIM3bulkJctBotGradingCoeff = 0.5;
        if (!model->BSIM3bulkJctSideGradingCoeffGiven)
            model->BSIM3bulkJctSideGradingCoeff = 0.33;
        if (!model->BSIM3bulkJctGateSideGradingCoeffGiven)
            model->BSIM3bulkJctGateSideGradingCoeff = model->BSIM3bulkJctSideGradingCoeff;
        if (!model->BSIM3jctEmissionCoeffGiven)
            model->BSIM3jctEmissionCoeff = 1.0;
        if (!model->BSIM3jctTempExponentGiven)
            model->BSIM3jctTempExponent = 3.0;
        if (!model->BSIM3oxideTrapDensityAGiven)
        {   if (model->BSIM3type == NMOS)
                model->BSIM3oxideTrapDensityA = 1e20;
            else
                model->BSIM3oxideTrapDensityA=9.9e18;
        }
        if (!model->BSIM3oxideTrapDensityBGiven)
        {   if (model->BSIM3type == NMOS)
                model->BSIM3oxideTrapDensityB = 5e4;
            else
                model->BSIM3oxideTrapDensityB = 2.4e3;
        }
        if (!model->BSIM3oxideTrapDensityCGiven)
        {   if (model->BSIM3type == NMOS)
                model->BSIM3oxideTrapDensityC = -1.4e-12;
            else
                model->BSIM3oxideTrapDensityC = 1.4e-12;

        }
        if (!model->BSIM3emGiven)
            model->BSIM3em = 4.1e7; /* V/m */
        if (!model->BSIM3efGiven)
            model->BSIM3ef = 1.0;
        if (!model->BSIM3afGiven)
            model->BSIM3af = 1.0;
        if (!model->BSIM3kfGiven)
            model->BSIM3kf = 0.0;

        if (!model->BSIM3vgsMaxGiven)
            model->BSIM3vgsMax = 1e99;
        if (!model->BSIM3vgdMaxGiven)
            model->BSIM3vgdMax = 1e99;
        if (!model->BSIM3vgbMaxGiven)
            model->BSIM3vgbMax = 1e99;
        if (!model->BSIM3vdsMaxGiven)
            model->BSIM3vdsMax = 1e99;
        if (!model->BSIM3vbsMaxGiven)
            model->BSIM3vbsMax = 1e99;
        if (!model->BSIM3vbdMaxGiven)
            model->BSIM3vbdMax = 1e99;
        if (!model->BSIM3vgsrMaxGiven)
            model->BSIM3vgsrMax = 1e99;
        if (!model->BSIM3vgdrMaxGiven)
            model->BSIM3vgdrMax = 1e99;
        if (!model->BSIM3vgbrMaxGiven)
            model->BSIM3vgbrMax = 1e99;
        if (!model->BSIM3vbsrMaxGiven)
            model->BSIM3vbsrMax = 1e99;
        if (!model->BSIM3vbdrMaxGiven)
            model->BSIM3vbdrMax = 1e99;

        /* loop through all the instances of the model */
        for (here = BSIM3instances(model); here != NULL ;
             here=BSIM3nextInstance(here))
        {
            /* allocate a chunk of the state vector */
            here->BSIM3states = *states;
            *states += BSIM3numStates;

            /* perform the parameter defaulting */
            if (!here->BSIM3drainAreaGiven)
                here->BSIM3drainArea = 0.0;
            if (!here->BSIM3drainPerimeterGiven)
                here->BSIM3drainPerimeter = 0.0;
            if (!here->BSIM3drainSquaresGiven)
            {
                if (model->BSIM3acmMod == 0)
                  here->BSIM3drainSquares = 1.0;
                else
                  here->BSIM3drainSquares = 0.0;
            }
            if (!here->BSIM3delvtoGiven)
                here->BSIM3delvto = 0.0;
            if (!here->BSIM3mulu0Given)
                here->BSIM3mulu0 = 1.0;
            if (!here->BSIM3icVBSGiven)
                here->BSIM3icVBS = 0.0;
            if (!here->BSIM3icVDSGiven)
                here->BSIM3icVDS = 0.0;
            if (!here->BSIM3icVGSGiven)
                here->BSIM3icVGS = 0.0;
            if (!here->BSIM3lGiven)
                here->BSIM3l = 5.0e-6;
            if (!here->BSIM3sourceAreaGiven)
                here->BSIM3sourceArea = 0.0;
            if (!here->BSIM3sourcePerimeterGiven)
                here->BSIM3sourcePerimeter = 0.0;
            if (!here->BSIM3sourceSquaresGiven)
            {
                if (model->BSIM3acmMod == 0)
                  here->BSIM3sourceSquares = 1.0;
                else
                  here->BSIM3sourceSquares = 0.0;
            }
            if (!here->BSIM3wGiven)
                here->BSIM3w = 5.0e-6;
            if (!here->BSIM3nqsModGiven)
                here->BSIM3nqsMod = model->BSIM3nqsMod;
            else if ((here->BSIM3nqsMod != 0) && (here->BSIM3nqsMod != 1))
            {   here->BSIM3nqsMod = model->BSIM3nqsMod;
                printf("Warning: nqsMod has been set to its global value %d.\n",
                model->BSIM3nqsMod);
            }
            if (!here->BSIM3acnqsModGiven)
                here->BSIM3acnqsMod = model->BSIM3acnqsMod;
            else if ((here->BSIM3acnqsMod != 0) && (here->BSIM3acnqsMod != 1))
            {   here->BSIM3acnqsMod = model->BSIM3acnqsMod;
                printf("Warning: acnqsMod has been set to its global value %d.\n",
                model->BSIM3acnqsMod);
            }

            if (!here->BSIM3geoGiven)
                here->BSIM3geo = 0;

            if (!here->BSIM3mGiven)
                here->BSIM3m = 1;

            /* process source/drain series resistance */
            /* ACM model */

            double drainResistance, sourceResistance;

            if (model->BSIM3acmMod == 0)
            {
                drainResistance = model->BSIM3sheetResistance
                    * here->BSIM3drainSquares;
                sourceResistance = model->BSIM3sheetResistance
                    * here->BSIM3sourceSquares;
            }
            else /* ACM > 0 */
            {
                error = ACM_SourceDrainResistances(
                    model->BSIM3acmMod,
                    model->BSIM3ld,
                    model->BSIM3ldif,
                    model->BSIM3hdif,
                    model->BSIM3wmlt,
                    here->BSIM3w,
                    model->BSIM3xw,
                    model->BSIM3sheetResistance,
                    here->BSIM3drainSquaresGiven,
                    model->BSIM3rd,
                    model->BSIM3rdc,
                    here->BSIM3drainSquares,
                    here->BSIM3sourceSquaresGiven,
                    model->BSIM3rs,
                    model->BSIM3rsc,
                    here->BSIM3sourceSquares,
                    &drainResistance,
                    &sourceResistance
                    );
                if (error)
                    return(error);
            }

            /* process drain series resistance */
            if (drainResistance != 0.0)
            {
              if(here->BSIM3dNodePrime == 0) {
                error = CKTmkVolt(ckt,&tmp,here->BSIM3name,"drain");
                if(error) return(error);
                here->BSIM3dNodePrime = tmp->number;
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
            {   here->BSIM3dNodePrime = here->BSIM3dNode;
            }

            /* process source series resistance */
            if (sourceResistance != 0.0)
            {
              if(here->BSIM3sNodePrime == 0) {
                error = CKTmkVolt(ckt,&tmp,here->BSIM3name,"source");
                if(error) return(error);
                here->BSIM3sNodePrime = tmp->number;
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
            {   here->BSIM3sNodePrime = here->BSIM3sNode;
            }

 /* internal charge node */

            if (here->BSIM3nqsMod)
            {   if (here->BSIM3qNode == 0)
            {   error = CKTmkVolt(ckt,&tmp,here->BSIM3name,"charge");
                if(error) return(error);
                here->BSIM3qNode = tmp->number;
            }
            }
            else
            {   here->BSIM3qNode = 0;
            }

        /* set Sparse Matrix Pointers */


/* macro to make elements with built in test for out of memory */
#define TSTALLOC(ptr,first,second) \
do { if((here->ptr = SMPmakeElt(matrix, here->first, here->second)) == NULL){\
    return(E_NOMEM);\
} } while(0)

            TSTALLOC(BSIM3DdPtr, BSIM3dNode, BSIM3dNode);
            TSTALLOC(BSIM3GgPtr, BSIM3gNode, BSIM3gNode);
            TSTALLOC(BSIM3SsPtr, BSIM3sNode, BSIM3sNode);
            TSTALLOC(BSIM3BbPtr, BSIM3bNode, BSIM3bNode);
            TSTALLOC(BSIM3DPdpPtr, BSIM3dNodePrime, BSIM3dNodePrime);
            TSTALLOC(BSIM3SPspPtr, BSIM3sNodePrime, BSIM3sNodePrime);
            TSTALLOC(BSIM3DdpPtr, BSIM3dNode, BSIM3dNodePrime);
            TSTALLOC(BSIM3GbPtr, BSIM3gNode, BSIM3bNode);
            TSTALLOC(BSIM3GdpPtr, BSIM3gNode, BSIM3dNodePrime);
            TSTALLOC(BSIM3GspPtr, BSIM3gNode, BSIM3sNodePrime);
            TSTALLOC(BSIM3SspPtr, BSIM3sNode, BSIM3sNodePrime);
            TSTALLOC(BSIM3BdpPtr, BSIM3bNode, BSIM3dNodePrime);
            TSTALLOC(BSIM3BspPtr, BSIM3bNode, BSIM3sNodePrime);
            TSTALLOC(BSIM3DPspPtr, BSIM3dNodePrime, BSIM3sNodePrime);
            TSTALLOC(BSIM3DPdPtr, BSIM3dNodePrime, BSIM3dNode);
            TSTALLOC(BSIM3BgPtr, BSIM3bNode, BSIM3gNode);
            TSTALLOC(BSIM3DPgPtr, BSIM3dNodePrime, BSIM3gNode);
            TSTALLOC(BSIM3SPgPtr, BSIM3sNodePrime, BSIM3gNode);
            TSTALLOC(BSIM3SPsPtr, BSIM3sNodePrime, BSIM3sNode);
            TSTALLOC(BSIM3DPbPtr, BSIM3dNodePrime, BSIM3bNode);
            TSTALLOC(BSIM3SPbPtr, BSIM3sNodePrime, BSIM3bNode);
            TSTALLOC(BSIM3SPdpPtr, BSIM3sNodePrime, BSIM3dNodePrime);

            TSTALLOC(BSIM3QqPtr, BSIM3qNode, BSIM3qNode);

            TSTALLOC(BSIM3QdpPtr, BSIM3qNode, BSIM3dNodePrime);
            TSTALLOC(BSIM3QspPtr, BSIM3qNode, BSIM3sNodePrime);
            TSTALLOC(BSIM3QgPtr, BSIM3qNode, BSIM3gNode);
            TSTALLOC(BSIM3QbPtr, BSIM3qNode, BSIM3bNode);
            TSTALLOC(BSIM3DPqPtr, BSIM3dNodePrime, BSIM3qNode);
            TSTALLOC(BSIM3SPqPtr, BSIM3sNodePrime, BSIM3qNode);
            TSTALLOC(BSIM3GqPtr, BSIM3gNode, BSIM3qNode);
            TSTALLOC(BSIM3BqPtr, BSIM3bNode, BSIM3qNode);

        }
    }
#ifdef USE_OMP
    InstCount = 0;
    model = (BSIM3model*)inModel;
    /* loop through all the BSIM3 device models
       to count the number of instances */

    for( ; model != NULL; model = BSIM3nextModel(model))
    {
        /* loop through all the instances of the model */
        for (here = BSIM3instances(model); here != NULL ;
             here=BSIM3nextInstance(here))
        {
            InstCount++;
        }
        model->BSIM3InstCount = 0;
        model->BSIM3InstanceArray = NULL;
    }
    InstArray = TMALLOC(BSIM3instance*, InstCount);
    model = (BSIM3model*)inModel;
    /* store this in the first model only */
    model->BSIM3InstCount = InstCount;
    model->BSIM3InstanceArray = InstArray;
    idx = 0;
    for( ; model != NULL; model = BSIM3nextModel(model))
    {
        /* loop through all the instances of the model */
        for (here = BSIM3instances(model); here != NULL ;
             here=BSIM3nextInstance(here))
        {
            InstArray[idx] = here;
            idx++;
        }
    }

#endif
    return(OK);
}

int
BSIM3unsetup(
    GENmodel *inModel,
    CKTcircuit *ckt)
{
    BSIM3model *model;
    BSIM3instance *here;

#ifdef USE_OMP
    model = (BSIM3model*)inModel;
    tfree(model->BSIM3InstanceArray);
#endif

    for (model = (BSIM3model *)inModel; model != NULL;
            model = BSIM3nextModel(model))
    {
        for (here = BSIM3instances(model); here != NULL;
                here=BSIM3nextInstance(here))
        {
            if (here->BSIM3qNode > 0)
                CKTdltNNum(ckt, here->BSIM3qNode);
            here->BSIM3qNode = 0;

            if (here->BSIM3sNodePrime > 0
                    && here->BSIM3sNodePrime != here->BSIM3sNode)
                CKTdltNNum(ckt, here->BSIM3sNodePrime);
            here->BSIM3sNodePrime = 0;

            if (here->BSIM3dNodePrime > 0
                    && here->BSIM3dNodePrime != here->BSIM3dNode)
                CKTdltNNum(ckt, here->BSIM3dNodePrime);
            here->BSIM3dNodePrime = 0;
        }
    }
    return OK;
}






