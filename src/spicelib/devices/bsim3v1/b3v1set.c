/**********
 * Copyright 1990 Regents of the University of California. All rights reserved.
 * File: b3set.c
 * Author: 1995 Min-Chie Jeng and Mansun Chan.
 * Modified by Paolo Nenzi 2002
 **********/

/*
 * Release Notes:
 * BSIM3v1v3.1,   Released by yuhua  96/12/08
 */

#include "ngspice/ngspice.h"
#include "ngspice/smpdefs.h"
#include "ngspice/cktdefs.h"
#include "bsim3v1def.h"
#include "ngspice/const.h"
#include "ngspice/sperror.h"
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
BSIM3v1setup(SMPmatrix *matrix, GENmodel *inModel, CKTcircuit *ckt,
             int *states)
{
BSIM3v1model *model = (BSIM3v1model*)inModel;
BSIM3v1instance *here;
int error;
CKTnode *tmp;

CKTnode *tmpNode;
IFuid tmpName;

    /*  loop through all the BSIM3v1 device models */
    for( ; model != NULL; model = BSIM3v1nextModel(model))
    {
/* Default value Processing for BSIM3v1 MOSFET Models */
        if (!model->BSIM3v1typeGiven)
            model->BSIM3v1type = NMOS;
        if (!model->BSIM3v1mobModGiven)
            model->BSIM3v1mobMod = 1;
        if (!model->BSIM3v1binUnitGiven)
            model->BSIM3v1binUnit = 1;
        if (!model->BSIM3v1paramChkGiven)
            model->BSIM3v1paramChk = 0;
        if (!model->BSIM3v1capModGiven)
            model->BSIM3v1capMod = 2;
        if (!model->BSIM3v1nqsModGiven)
            model->BSIM3v1nqsMod = 0;
        if (!model->BSIM3v1noiModGiven)
            model->BSIM3v1noiMod = 1;
        if (!model->BSIM3v1versionGiven)
            model->BSIM3v1version = 3.1;
        if (!model->BSIM3v1toxGiven)
            model->BSIM3v1tox = 150.0e-10;
        model->BSIM3v1cox = 3.453133e-11 / model->BSIM3v1tox;

        if (!model->BSIM3v1cdscGiven)
	    model->BSIM3v1cdsc = 2.4e-4;   /* unit Q/V/m^2  */
        if (!model->BSIM3v1cdscbGiven)
	    model->BSIM3v1cdscb = 0.0;   /* unit Q/V/m^2  */
        if (!model->BSIM3v1cdscdGiven)
	    model->BSIM3v1cdscd = 0.0;   /* unit Q/V/m^2  */
        if (!model->BSIM3v1citGiven)
	    model->BSIM3v1cit = 0.0;   /* unit Q/V/m^2  */
        if (!model->BSIM3v1nfactorGiven)
	    model->BSIM3v1nfactor = 1;
        if (!model->BSIM3v1xjGiven)
            model->BSIM3v1xj = .15e-6;
        if (!model->BSIM3v1vsatGiven)
            model->BSIM3v1vsat = 8.0e4;    /* unit m/s */
        if (!model->BSIM3v1atGiven)
            model->BSIM3v1at = 3.3e4;    /* unit m/s */
        if (!model->BSIM3v1a0Given)
            model->BSIM3v1a0 = 1.0;
        if (!model->BSIM3v1agsGiven)
            model->BSIM3v1ags = 0.0;
        if (!model->BSIM3v1a1Given)
            model->BSIM3v1a1 = 0.0;
        if (!model->BSIM3v1a2Given)
            model->BSIM3v1a2 = 1.0;
        if (!model->BSIM3v1ketaGiven)
            model->BSIM3v1keta = -0.047;    /* unit  / V */
        if (!model->BSIM3v1nsubGiven)
            model->BSIM3v1nsub = 6.0e16;   /* unit 1/cm3 */
        if (!model->BSIM3v1npeakGiven)
            model->BSIM3v1npeak = 1.7e17;   /* unit 1/cm3 */
        if (!model->BSIM3v1ngateGiven)
            model->BSIM3v1ngate = 0;   /* unit 1/cm3 */
        if (!model->BSIM3v1vbmGiven)
	    model->BSIM3v1vbm = -3.0;
        if (!model->BSIM3v1xtGiven)
	    model->BSIM3v1xt = 1.55e-7;
        if (!model->BSIM3v1kt1Given)
            model->BSIM3v1kt1 = -0.11;      /* unit V */
        if (!model->BSIM3v1kt1lGiven)
            model->BSIM3v1kt1l = 0.0;      /* unit V*m */
        if (!model->BSIM3v1kt2Given)
            model->BSIM3v1kt2 = 0.022;      /* No unit */
        if (!model->BSIM3v1k3Given)
            model->BSIM3v1k3 = 80.0;
        if (!model->BSIM3v1k3bGiven)
            model->BSIM3v1k3b = 0.0;
        if (!model->BSIM3v1w0Given)
            model->BSIM3v1w0 = 2.5e-6;
        if (!model->BSIM3v1nlxGiven)
            model->BSIM3v1nlx = 1.74e-7;
        if (!model->BSIM3v1dvt0Given)
            model->BSIM3v1dvt0 = 2.2;
        if (!model->BSIM3v1dvt1Given)
            model->BSIM3v1dvt1 = 0.53;
        if (!model->BSIM3v1dvt2Given)
            model->BSIM3v1dvt2 = -0.032;   /* unit 1 / V */

        if (!model->BSIM3v1dvt0wGiven)
            model->BSIM3v1dvt0w = 0.0;
        if (!model->BSIM3v1dvt1wGiven)
            model->BSIM3v1dvt1w = 5.3e6;
        if (!model->BSIM3v1dvt2wGiven)
            model->BSIM3v1dvt2w = -0.032;

        if (!model->BSIM3v1droutGiven)
            model->BSIM3v1drout = 0.56;
        if (!model->BSIM3v1dsubGiven)
            model->BSIM3v1dsub = model->BSIM3v1drout;
        if (!model->BSIM3v1vth0Given)
            model->BSIM3v1vth0 = (model->BSIM3v1type == NMOS) ? 0.7 : -0.7;
        if (!model->BSIM3v1uaGiven)
            model->BSIM3v1ua = 2.25e-9;      /* unit m/V */
        if (!model->BSIM3v1ua1Given)
            model->BSIM3v1ua1 = 4.31e-9;      /* unit m/V */
        if (!model->BSIM3v1ubGiven)
            model->BSIM3v1ub = 5.87e-19;     /* unit (m/V)**2 */
        if (!model->BSIM3v1ub1Given)
            model->BSIM3v1ub1 = -7.61e-18;     /* unit (m/V)**2 */
        if (!model->BSIM3v1ucGiven)
            model->BSIM3v1uc = (model->BSIM3v1mobMod == 3) ? -0.0465 : -0.0465e-9;
        if (!model->BSIM3v1uc1Given)
            model->BSIM3v1uc1 = (model->BSIM3v1mobMod == 3) ? -0.056 : -0.056e-9;
        if (!model->BSIM3v1u0Given)
            model->BSIM3v1u0 = (model->BSIM3v1type == NMOS) ? 0.067 : 0.025;
        if (!model->BSIM3v1uteGiven)
	    model->BSIM3v1ute = -1.5;
        if (!model->BSIM3v1voffGiven)
	    model->BSIM3v1voff = -0.08;
        if (!model->BSIM3v1deltaGiven)
           model->BSIM3v1delta = 0.01;
        if (!model->BSIM3v1rdswGiven)
            model->BSIM3v1rdsw = 0;
        if (!model->BSIM3v1prwgGiven)
            model->BSIM3v1prwg = 0.0;      /* unit 1/V */
        if (!model->BSIM3v1prwbGiven)
            model->BSIM3v1prwb = 0.0;
        if (!model->BSIM3v1prtGiven)
            model->BSIM3v1prt = 0.0;
        if (!model->BSIM3v1eta0Given)
            model->BSIM3v1eta0 = 0.08;      /* no unit  */
        if (!model->BSIM3v1etabGiven)
            model->BSIM3v1etab = -0.07;      /* unit  1/V */
        if (!model->BSIM3v1pclmGiven)
            model->BSIM3v1pclm = 1.3;      /* no unit  */
        if (!model->BSIM3v1pdibl1Given)
            model->BSIM3v1pdibl1 = .39;    /* no unit  */
        if (!model->BSIM3v1pdibl2Given)
            model->BSIM3v1pdibl2 = 0.0086;    /* no unit  */
        if (!model->BSIM3v1pdiblbGiven)
            model->BSIM3v1pdiblb = 0.0;    /* 1/V  */
        if (!model->BSIM3v1pscbe1Given)
            model->BSIM3v1pscbe1 = 4.24e8;
        if (!model->BSIM3v1pscbe2Given)
            model->BSIM3v1pscbe2 = 1.0e-5;
        if (!model->BSIM3v1pvagGiven)
            model->BSIM3v1pvag = 0.0;
        if (!model->BSIM3v1wrGiven)
            model->BSIM3v1wr = 1.0;
        if (!model->BSIM3v1dwgGiven)
            model->BSIM3v1dwg = 0.0;
        if (!model->BSIM3v1dwbGiven)
            model->BSIM3v1dwb = 0.0;
        if (!model->BSIM3v1b0Given)
            model->BSIM3v1b0 = 0.0;
        if (!model->BSIM3v1b1Given)
            model->BSIM3v1b1 = 0.0;
        if (!model->BSIM3v1alpha0Given)
            model->BSIM3v1alpha0 = 0.0;
        if (!model->BSIM3v1beta0Given)
            model->BSIM3v1beta0 = 30.0;

        if (!model->BSIM3v1elmGiven)
            model->BSIM3v1elm = 5.0;
        if (!model->BSIM3v1cgslGiven)
            model->BSIM3v1cgsl = 0.0;
        if (!model->BSIM3v1cgdlGiven)
            model->BSIM3v1cgdl = 0.0;
        if (!model->BSIM3v1ckappaGiven)
            model->BSIM3v1ckappa = 0.6;
        if (!model->BSIM3v1clcGiven)
            model->BSIM3v1clc = 0.1e-6;
        if (!model->BSIM3v1cleGiven)
            model->BSIM3v1cle = 0.6;
        if (!model->BSIM3v1vfbcvGiven)
            model->BSIM3v1vfbcv = -1.0;

	/* Length dependence */
        if (!model->BSIM3v1lcdscGiven)
	    model->BSIM3v1lcdsc = 0.0;
        if (!model->BSIM3v1lcdscbGiven)
	    model->BSIM3v1lcdscb = 0.0;
        if (!model->BSIM3v1lcdscdGiven)
	    model->BSIM3v1lcdscd = 0.0;
        if (!model->BSIM3v1lcitGiven)
	    model->BSIM3v1lcit = 0.0;
        if (!model->BSIM3v1lnfactorGiven)
	    model->BSIM3v1lnfactor = 0.0;
        if (!model->BSIM3v1lxjGiven)
            model->BSIM3v1lxj = 0.0;
        if (!model->BSIM3v1lvsatGiven)
            model->BSIM3v1lvsat = 0.0;
        if (!model->BSIM3v1latGiven)
            model->BSIM3v1lat = 0.0;
        if (!model->BSIM3v1la0Given)
            model->BSIM3v1la0 = 0.0;
        if (!model->BSIM3v1lagsGiven)
            model->BSIM3v1lags = 0.0;
        if (!model->BSIM3v1la1Given)
            model->BSIM3v1la1 = 0.0;
        if (!model->BSIM3v1la2Given)
            model->BSIM3v1la2 = 0.0;
        if (!model->BSIM3v1lketaGiven)
            model->BSIM3v1lketa = 0.0;
        if (!model->BSIM3v1lnsubGiven)
            model->BSIM3v1lnsub = 0.0;
        if (!model->BSIM3v1lnpeakGiven)
            model->BSIM3v1lnpeak = 0.0;
        if (!model->BSIM3v1lngateGiven)
            model->BSIM3v1lngate = 0.0;
        if (!model->BSIM3v1lvbmGiven)
	    model->BSIM3v1lvbm = 0.0;
        if (!model->BSIM3v1lxtGiven)
	    model->BSIM3v1lxt = 0.0;
        if (!model->BSIM3v1lkt1Given)
            model->BSIM3v1lkt1 = 0.0;
        if (!model->BSIM3v1lkt1lGiven)
            model->BSIM3v1lkt1l = 0.0;
        if (!model->BSIM3v1lkt2Given)
            model->BSIM3v1lkt2 = 0.0;
        if (!model->BSIM3v1lk3Given)
            model->BSIM3v1lk3 = 0.0;
        if (!model->BSIM3v1lk3bGiven)
            model->BSIM3v1lk3b = 0.0;
        if (!model->BSIM3v1lw0Given)
            model->BSIM3v1lw0 = 0.0;
        if (!model->BSIM3v1lnlxGiven)
            model->BSIM3v1lnlx = 0.0;
        if (!model->BSIM3v1ldvt0Given)
            model->BSIM3v1ldvt0 = 0.0;
        if (!model->BSIM3v1ldvt1Given)
            model->BSIM3v1ldvt1 = 0.0;
        if (!model->BSIM3v1ldvt2Given)
            model->BSIM3v1ldvt2 = 0.0;
        if (!model->BSIM3v1ldvt0wGiven)
            model->BSIM3v1ldvt0w = 0.0;
        if (!model->BSIM3v1ldvt1wGiven)
            model->BSIM3v1ldvt1w = 0.0;
        if (!model->BSIM3v1ldvt2wGiven)
            model->BSIM3v1ldvt2w = 0.0;
        if (!model->BSIM3v1ldroutGiven)
            model->BSIM3v1ldrout = 0.0;
        if (!model->BSIM3v1ldsubGiven)
            model->BSIM3v1ldsub = 0.0;
        if (!model->BSIM3v1lvth0Given)
           model->BSIM3v1lvth0 = 0.0;
        if (!model->BSIM3v1luaGiven)
            model->BSIM3v1lua = 0.0;
        if (!model->BSIM3v1lua1Given)
            model->BSIM3v1lua1 = 0.0;
        if (!model->BSIM3v1lubGiven)
            model->BSIM3v1lub = 0.0;
        if (!model->BSIM3v1lub1Given)
            model->BSIM3v1lub1 = 0.0;
        if (!model->BSIM3v1lucGiven)
            model->BSIM3v1luc = 0.0;
        if (!model->BSIM3v1luc1Given)
            model->BSIM3v1luc1 = 0.0;
        if (!model->BSIM3v1lu0Given)
            model->BSIM3v1lu0 = 0.0;
        if (!model->BSIM3v1luteGiven)
	    model->BSIM3v1lute = 0.0;
        if (!model->BSIM3v1lvoffGiven)
	    model->BSIM3v1lvoff = 0.0;
        if (!model->BSIM3v1ldeltaGiven)
            model->BSIM3v1ldelta = 0.0;
        if (!model->BSIM3v1lrdswGiven)
            model->BSIM3v1lrdsw = 0.0;
        if (!model->BSIM3v1lprwbGiven)
            model->BSIM3v1lprwb = 0.0;
        if (!model->BSIM3v1lprwgGiven)
            model->BSIM3v1lprwg = 0.0;
        if (!model->BSIM3v1lprtGiven)
            model->BSIM3v1lprt = 0.0;
        if (!model->BSIM3v1leta0Given)
            model->BSIM3v1leta0 = 0.0;
        if (!model->BSIM3v1letabGiven)
            model->BSIM3v1letab = -0.0;
        if (!model->BSIM3v1lpclmGiven)
            model->BSIM3v1lpclm = 0.0;
        if (!model->BSIM3v1lpdibl1Given)
            model->BSIM3v1lpdibl1 = 0.0;
        if (!model->BSIM3v1lpdibl2Given)
            model->BSIM3v1lpdibl2 = 0.0;
        if (!model->BSIM3v1lpdiblbGiven)
            model->BSIM3v1lpdiblb = 0.0;
        if (!model->BSIM3v1lpscbe1Given)
            model->BSIM3v1lpscbe1 = 0.0;
        if (!model->BSIM3v1lpscbe2Given)
            model->BSIM3v1lpscbe2 = 0.0;
        if (!model->BSIM3v1lpvagGiven)
            model->BSIM3v1lpvag = 0.0;
        if (!model->BSIM3v1lwrGiven)
            model->BSIM3v1lwr = 0.0;
        if (!model->BSIM3v1ldwgGiven)
            model->BSIM3v1ldwg = 0.0;
        if (!model->BSIM3v1ldwbGiven)
            model->BSIM3v1ldwb = 0.0;
        if (!model->BSIM3v1lb0Given)
            model->BSIM3v1lb0 = 0.0;
        if (!model->BSIM3v1lb1Given)
            model->BSIM3v1lb1 = 0.0;
        if (!model->BSIM3v1lalpha0Given)
            model->BSIM3v1lalpha0 = 0.0;
        if (!model->BSIM3v1lbeta0Given)
            model->BSIM3v1lbeta0 = 0.0;

        if (!model->BSIM3v1lelmGiven)
            model->BSIM3v1lelm = 0.0;
        if (!model->BSIM3v1lcgslGiven)
            model->BSIM3v1lcgsl = 0.0;
        if (!model->BSIM3v1lcgdlGiven)
            model->BSIM3v1lcgdl = 0.0;
        if (!model->BSIM3v1lckappaGiven)
            model->BSIM3v1lckappa = 0.0;
        if (!model->BSIM3v1lclcGiven)
            model->BSIM3v1lclc = 0.0;
        if (!model->BSIM3v1lcleGiven)
            model->BSIM3v1lcle = 0.0;
        if (!model->BSIM3v1lcfGiven)
            model->BSIM3v1lcf = 0.0;
        if (!model->BSIM3v1lvfbcvGiven)
            model->BSIM3v1lvfbcv = 0.0;

	/* Width dependence */
        if (!model->BSIM3v1wcdscGiven)
	    model->BSIM3v1wcdsc = 0.0;
        if (!model->BSIM3v1wcdscbGiven)
	    model->BSIM3v1wcdscb = 0.0;
        if (!model->BSIM3v1wcdscdGiven)
	    model->BSIM3v1wcdscd = 0.0;
        if (!model->BSIM3v1wcitGiven)
	    model->BSIM3v1wcit = 0.0;
        if (!model->BSIM3v1wnfactorGiven)
	    model->BSIM3v1wnfactor = 0.0;
        if (!model->BSIM3v1wxjGiven)
            model->BSIM3v1wxj = 0.0;
        if (!model->BSIM3v1wvsatGiven)
            model->BSIM3v1wvsat = 0.0;
        if (!model->BSIM3v1watGiven)
            model->BSIM3v1wat = 0.0;
        if (!model->BSIM3v1wa0Given)
            model->BSIM3v1wa0 = 0.0;
        if (!model->BSIM3v1wagsGiven)
            model->BSIM3v1wags = 0.0;
        if (!model->BSIM3v1wa1Given)
            model->BSIM3v1wa1 = 0.0;
        if (!model->BSIM3v1wa2Given)
            model->BSIM3v1wa2 = 0.0;
        if (!model->BSIM3v1wketaGiven)
            model->BSIM3v1wketa = 0.0;
        if (!model->BSIM3v1wnsubGiven)
            model->BSIM3v1wnsub = 0.0;
        if (!model->BSIM3v1wnpeakGiven)
            model->BSIM3v1wnpeak = 0.0;
        if (!model->BSIM3v1wngateGiven)
            model->BSIM3v1wngate = 0.0;
        if (!model->BSIM3v1wvbmGiven)
	    model->BSIM3v1wvbm = 0.0;
        if (!model->BSIM3v1wxtGiven)
	    model->BSIM3v1wxt = 0.0;
        if (!model->BSIM3v1wkt1Given)
            model->BSIM3v1wkt1 = 0.0;
        if (!model->BSIM3v1wkt1lGiven)
            model->BSIM3v1wkt1l = 0.0;
        if (!model->BSIM3v1wkt2Given)
            model->BSIM3v1wkt2 = 0.0;
        if (!model->BSIM3v1wk3Given)
            model->BSIM3v1wk3 = 0.0;
        if (!model->BSIM3v1wk3bGiven)
            model->BSIM3v1wk3b = 0.0;
        if (!model->BSIM3v1ww0Given)
            model->BSIM3v1ww0 = 0.0;
        if (!model->BSIM3v1wnlxGiven)
            model->BSIM3v1wnlx = 0.0;
        if (!model->BSIM3v1wdvt0Given)
            model->BSIM3v1wdvt0 = 0.0;
        if (!model->BSIM3v1wdvt1Given)
            model->BSIM3v1wdvt1 = 0.0;
        if (!model->BSIM3v1wdvt2Given)
            model->BSIM3v1wdvt2 = 0.0;
        if (!model->BSIM3v1wdvt0wGiven)
            model->BSIM3v1wdvt0w = 0.0;
        if (!model->BSIM3v1wdvt1wGiven)
            model->BSIM3v1wdvt1w = 0.0;
        if (!model->BSIM3v1wdvt2wGiven)
            model->BSIM3v1wdvt2w = 0.0;
        if (!model->BSIM3v1wdroutGiven)
            model->BSIM3v1wdrout = 0.0;
        if (!model->BSIM3v1wdsubGiven)
            model->BSIM3v1wdsub = 0.0;
        if (!model->BSIM3v1wvth0Given)
           model->BSIM3v1wvth0 = 0.0;
        if (!model->BSIM3v1wuaGiven)
            model->BSIM3v1wua = 0.0;
        if (!model->BSIM3v1wua1Given)
            model->BSIM3v1wua1 = 0.0;
        if (!model->BSIM3v1wubGiven)
            model->BSIM3v1wub = 0.0;
        if (!model->BSIM3v1wub1Given)
            model->BSIM3v1wub1 = 0.0;
        if (!model->BSIM3v1wucGiven)
            model->BSIM3v1wuc = 0.0;
        if (!model->BSIM3v1wuc1Given)
            model->BSIM3v1wuc1 = 0.0;
        if (!model->BSIM3v1wu0Given)
            model->BSIM3v1wu0 = 0.0;
        if (!model->BSIM3v1wuteGiven)
	    model->BSIM3v1wute = 0.0;
        if (!model->BSIM3v1wvoffGiven)
	    model->BSIM3v1wvoff = 0.0;
        if (!model->BSIM3v1wdeltaGiven)
            model->BSIM3v1wdelta = 0.0;
        if (!model->BSIM3v1wrdswGiven)
            model->BSIM3v1wrdsw = 0.0;
        if (!model->BSIM3v1wprwbGiven)
            model->BSIM3v1wprwb = 0.0;
        if (!model->BSIM3v1wprwgGiven)
            model->BSIM3v1wprwg = 0.0;
        if (!model->BSIM3v1wprtGiven)
            model->BSIM3v1wprt = 0.0;
        if (!model->BSIM3v1weta0Given)
            model->BSIM3v1weta0 = 0.0;
        if (!model->BSIM3v1wetabGiven)
            model->BSIM3v1wetab = 0.0;
        if (!model->BSIM3v1wpclmGiven)
            model->BSIM3v1wpclm = 0.0;
        if (!model->BSIM3v1wpdibl1Given)
            model->BSIM3v1wpdibl1 = 0.0;
        if (!model->BSIM3v1wpdibl2Given)
            model->BSIM3v1wpdibl2 = 0.0;
        if (!model->BSIM3v1wpdiblbGiven)
            model->BSIM3v1wpdiblb = 0.0;
        if (!model->BSIM3v1wpscbe1Given)
            model->BSIM3v1wpscbe1 = 0.0;
        if (!model->BSIM3v1wpscbe2Given)
            model->BSIM3v1wpscbe2 = 0.0;
        if (!model->BSIM3v1wpvagGiven)
            model->BSIM3v1wpvag = 0.0;
        if (!model->BSIM3v1wwrGiven)
            model->BSIM3v1wwr = 0.0;
        if (!model->BSIM3v1wdwgGiven)
            model->BSIM3v1wdwg = 0.0;
        if (!model->BSIM3v1wdwbGiven)
            model->BSIM3v1wdwb = 0.0;
        if (!model->BSIM3v1wb0Given)
            model->BSIM3v1wb0 = 0.0;
        if (!model->BSIM3v1wb1Given)
            model->BSIM3v1wb1 = 0.0;
        if (!model->BSIM3v1walpha0Given)
            model->BSIM3v1walpha0 = 0.0;
        if (!model->BSIM3v1wbeta0Given)
            model->BSIM3v1wbeta0 = 0.0;

        if (!model->BSIM3v1welmGiven)
            model->BSIM3v1welm = 0.0;
        if (!model->BSIM3v1wcgslGiven)
            model->BSIM3v1wcgsl = 0.0;
        if (!model->BSIM3v1wcgdlGiven)
            model->BSIM3v1wcgdl = 0.0;
        if (!model->BSIM3v1wckappaGiven)
            model->BSIM3v1wckappa = 0.0;
        if (!model->BSIM3v1wcfGiven)
            model->BSIM3v1wcf = 0.0;
        if (!model->BSIM3v1wclcGiven)
            model->BSIM3v1wclc = 0.0;
        if (!model->BSIM3v1wcleGiven)
            model->BSIM3v1wcle = 0.0;
        if (!model->BSIM3v1wvfbcvGiven)
            model->BSIM3v1wvfbcv = 0.0;

	/* Cross-term dependence */
        if (!model->BSIM3v1pcdscGiven)
	    model->BSIM3v1pcdsc = 0.0;
        if (!model->BSIM3v1pcdscbGiven)
	    model->BSIM3v1pcdscb = 0.0;
        if (!model->BSIM3v1pcdscdGiven)
	    model->BSIM3v1pcdscd = 0.0;
        if (!model->BSIM3v1pcitGiven)
	    model->BSIM3v1pcit = 0.0;
        if (!model->BSIM3v1pnfactorGiven)
	    model->BSIM3v1pnfactor = 0.0;
        if (!model->BSIM3v1pxjGiven)
            model->BSIM3v1pxj = 0.0;
        if (!model->BSIM3v1pvsatGiven)
            model->BSIM3v1pvsat = 0.0;
        if (!model->BSIM3v1patGiven)
            model->BSIM3v1pat = 0.0;
        if (!model->BSIM3v1pa0Given)
            model->BSIM3v1pa0 = 0.0;

        if (!model->BSIM3v1pagsGiven)
            model->BSIM3v1pags = 0.0;
        if (!model->BSIM3v1pa1Given)
            model->BSIM3v1pa1 = 0.0;
        if (!model->BSIM3v1pa2Given)
            model->BSIM3v1pa2 = 0.0;
        if (!model->BSIM3v1pketaGiven)
            model->BSIM3v1pketa = 0.0;
        if (!model->BSIM3v1pnsubGiven)
            model->BSIM3v1pnsub = 0.0;
        if (!model->BSIM3v1pnpeakGiven)
            model->BSIM3v1pnpeak = 0.0;
        if (!model->BSIM3v1pngateGiven)
            model->BSIM3v1pngate = 0.0;
        if (!model->BSIM3v1pvbmGiven)
	    model->BSIM3v1pvbm = 0.0;
        if (!model->BSIM3v1pxtGiven)
	    model->BSIM3v1pxt = 0.0;
        if (!model->BSIM3v1pkt1Given)
            model->BSIM3v1pkt1 = 0.0;
        if (!model->BSIM3v1pkt1lGiven)
            model->BSIM3v1pkt1l = 0.0;
        if (!model->BSIM3v1pkt2Given)
            model->BSIM3v1pkt2 = 0.0;
        if (!model->BSIM3v1pk3Given)
            model->BSIM3v1pk3 = 0.0;
        if (!model->BSIM3v1pk3bGiven)
            model->BSIM3v1pk3b = 0.0;
        if (!model->BSIM3v1pw0Given)
            model->BSIM3v1pw0 = 0.0;
        if (!model->BSIM3v1pnlxGiven)
            model->BSIM3v1pnlx = 0.0;
        if (!model->BSIM3v1pdvt0Given)
            model->BSIM3v1pdvt0 = 0.0;
        if (!model->BSIM3v1pdvt1Given)
            model->BSIM3v1pdvt1 = 0.0;
        if (!model->BSIM3v1pdvt2Given)
            model->BSIM3v1pdvt2 = 0.0;
        if (!model->BSIM3v1pdvt0wGiven)
            model->BSIM3v1pdvt0w = 0.0;
        if (!model->BSIM3v1pdvt1wGiven)
            model->BSIM3v1pdvt1w = 0.0;
        if (!model->BSIM3v1pdvt2wGiven)
            model->BSIM3v1pdvt2w = 0.0;
        if (!model->BSIM3v1pdroutGiven)
            model->BSIM3v1pdrout = 0.0;
        if (!model->BSIM3v1pdsubGiven)
            model->BSIM3v1pdsub = 0.0;
        if (!model->BSIM3v1pvth0Given)
           model->BSIM3v1pvth0 = 0.0;
        if (!model->BSIM3v1puaGiven)
            model->BSIM3v1pua = 0.0;
        if (!model->BSIM3v1pua1Given)
            model->BSIM3v1pua1 = 0.0;
        if (!model->BSIM3v1pubGiven)
            model->BSIM3v1pub = 0.0;
        if (!model->BSIM3v1pub1Given)
            model->BSIM3v1pub1 = 0.0;
        if (!model->BSIM3v1pucGiven)
            model->BSIM3v1puc = 0.0;
        if (!model->BSIM3v1puc1Given)
            model->BSIM3v1puc1 = 0.0;
        if (!model->BSIM3v1pu0Given)
            model->BSIM3v1pu0 = 0.0;
        if (!model->BSIM3v1puteGiven)
	    model->BSIM3v1pute = 0.0;
        if (!model->BSIM3v1pvoffGiven)
	    model->BSIM3v1pvoff = 0.0;
        if (!model->BSIM3v1pdeltaGiven)
            model->BSIM3v1pdelta = 0.0;
        if (!model->BSIM3v1prdswGiven)
            model->BSIM3v1prdsw = 0.0;
        if (!model->BSIM3v1pprwbGiven)
            model->BSIM3v1pprwb = 0.0;
        if (!model->BSIM3v1pprwgGiven)
            model->BSIM3v1pprwg = 0.0;
        if (!model->BSIM3v1pprtGiven)
            model->BSIM3v1pprt = 0.0;
        if (!model->BSIM3v1peta0Given)
            model->BSIM3v1peta0 = 0.0;
        if (!model->BSIM3v1petabGiven)
            model->BSIM3v1petab = 0.0;
        if (!model->BSIM3v1ppclmGiven)
            model->BSIM3v1ppclm = 0.0;
        if (!model->BSIM3v1ppdibl1Given)
            model->BSIM3v1ppdibl1 = 0.0;
        if (!model->BSIM3v1ppdibl2Given)
            model->BSIM3v1ppdibl2 = 0.0;
        if (!model->BSIM3v1ppdiblbGiven)
            model->BSIM3v1ppdiblb = 0.0;
        if (!model->BSIM3v1ppscbe1Given)
            model->BSIM3v1ppscbe1 = 0.0;
        if (!model->BSIM3v1ppscbe2Given)
            model->BSIM3v1ppscbe2 = 0.0;
        if (!model->BSIM3v1ppvagGiven)
            model->BSIM3v1ppvag = 0.0;
        if (!model->BSIM3v1pwrGiven)
            model->BSIM3v1pwr = 0.0;
        if (!model->BSIM3v1pdwgGiven)
            model->BSIM3v1pdwg = 0.0;
        if (!model->BSIM3v1pdwbGiven)
            model->BSIM3v1pdwb = 0.0;
        if (!model->BSIM3v1pb0Given)
            model->BSIM3v1pb0 = 0.0;
        if (!model->BSIM3v1pb1Given)
            model->BSIM3v1pb1 = 0.0;
        if (!model->BSIM3v1palpha0Given)
            model->BSIM3v1palpha0 = 0.0;
        if (!model->BSIM3v1pbeta0Given)
            model->BSIM3v1pbeta0 = 0.0;

        if (!model->BSIM3v1pelmGiven)
            model->BSIM3v1pelm = 0.0;
        if (!model->BSIM3v1pcgslGiven)
            model->BSIM3v1pcgsl = 0.0;
        if (!model->BSIM3v1pcgdlGiven)
            model->BSIM3v1pcgdl = 0.0;
        if (!model->BSIM3v1pckappaGiven)
            model->BSIM3v1pckappa = 0.0;
        if (!model->BSIM3v1pcfGiven)
            model->BSIM3v1pcf = 0.0;
        if (!model->BSIM3v1pclcGiven)
            model->BSIM3v1pclc = 0.0;
        if (!model->BSIM3v1pcleGiven)
            model->BSIM3v1pcle = 0.0;
        if (!model->BSIM3v1pvfbcvGiven)
            model->BSIM3v1pvfbcv = 0.0;

        /* unit degree celcius */
        if (!model->BSIM3v1tnomGiven)
	    model->BSIM3v1tnom = ckt->CKTnomTemp;
        if (!model->BSIM3v1LintGiven)
           model->BSIM3v1Lint = 0.0;
        if (!model->BSIM3v1LlGiven)
           model->BSIM3v1Ll = 0.0;
        if (!model->BSIM3v1LlnGiven)
           model->BSIM3v1Lln = 1.0;
        if (!model->BSIM3v1LwGiven)
           model->BSIM3v1Lw = 0.0;
        if (!model->BSIM3v1LwnGiven)
           model->BSIM3v1Lwn = 1.0;
        if (!model->BSIM3v1LwlGiven)
           model->BSIM3v1Lwl = 0.0;
        if (!model->BSIM3v1LminGiven)
           model->BSIM3v1Lmin = 0.0;
        if (!model->BSIM3v1LmaxGiven)
           model->BSIM3v1Lmax = 1.0;
        if (!model->BSIM3v1WintGiven)
           model->BSIM3v1Wint = 0.0;
        if (!model->BSIM3v1WlGiven)
           model->BSIM3v1Wl = 0.0;
        if (!model->BSIM3v1WlnGiven)
           model->BSIM3v1Wln = 1.0;
        if (!model->BSIM3v1WwGiven)
           model->BSIM3v1Ww = 0.0;
        if (!model->BSIM3v1WwnGiven)
           model->BSIM3v1Wwn = 1.0;
        if (!model->BSIM3v1WwlGiven)
           model->BSIM3v1Wwl = 0.0;
        if (!model->BSIM3v1WminGiven)
           model->BSIM3v1Wmin = 0.0;
        if (!model->BSIM3v1WmaxGiven)
           model->BSIM3v1Wmax = 1.0;
        if (!model->BSIM3v1dwcGiven)
           model->BSIM3v1dwc = model->BSIM3v1Wint;
        if (!model->BSIM3v1dlcGiven)
           model->BSIM3v1dlc = model->BSIM3v1Lint;
	if (!model->BSIM3v1cfGiven)
            model->BSIM3v1cf = 2.0 * EPSOX / PI
			   * log(1.0 + 0.4e-6 / model->BSIM3v1tox);
        if (!model->BSIM3v1cgdoGiven)
	{   if (model->BSIM3v1dlcGiven && (model->BSIM3v1dlc > 0.0))
	    {   model->BSIM3v1cgdo = model->BSIM3v1dlc * model->BSIM3v1cox
				 - model->BSIM3v1cgdl ;
	    }
	    else
	        model->BSIM3v1cgdo = 0.6 * model->BSIM3v1xj * model->BSIM3v1cox;
	}
        if (!model->BSIM3v1cgsoGiven)
	{   if (model->BSIM3v1dlcGiven && (model->BSIM3v1dlc > 0.0))
	    {   model->BSIM3v1cgso = model->BSIM3v1dlc * model->BSIM3v1cox
				 - model->BSIM3v1cgsl ;
	    }
	    else
	        model->BSIM3v1cgso = 0.6 * model->BSIM3v1xj * model->BSIM3v1cox;
	}

        if (!model->BSIM3v1cgboGiven)
	{   model->BSIM3v1cgbo = 2.0 * model->BSIM3v1dwc * model->BSIM3v1cox;
	}
        if (!model->BSIM3v1xpartGiven)
            model->BSIM3v1xpart = 0.0;
        if (!model->BSIM3v1sheetResistanceGiven)
            model->BSIM3v1sheetResistance = 0.0;
        if (!model->BSIM3v1unitAreaJctCapGiven)
            model->BSIM3v1unitAreaJctCap = 5.0E-4;
        if (!model->BSIM3v1unitLengthSidewallJctCapGiven)
            model->BSIM3v1unitLengthSidewallJctCap = 5.0E-10;
        if (!model->BSIM3v1unitLengthGateSidewallJctCapGiven)
            model->BSIM3v1unitLengthGateSidewallJctCap = model->BSIM3v1unitLengthSidewallJctCap ;
        if (!model->BSIM3v1jctSatCurDensityGiven)
            model->BSIM3v1jctSatCurDensity = 1.0E-4;
        if (!model->BSIM3v1jctSidewallSatCurDensityGiven)
            model->BSIM3v1jctSidewallSatCurDensity = 0.0;
        if (!model->BSIM3v1bulkJctPotentialGiven)
            model->BSIM3v1bulkJctPotential = 1.0;
        if (!model->BSIM3v1sidewallJctPotentialGiven)
            model->BSIM3v1sidewallJctPotential = 1.0;
        if (!model->BSIM3v1GatesidewallJctPotentialGiven)
            model->BSIM3v1GatesidewallJctPotential = model->BSIM3v1sidewallJctPotential;
        if (!model->BSIM3v1bulkJctBotGradingCoeffGiven)
            model->BSIM3v1bulkJctBotGradingCoeff = 0.5;
        if (!model->BSIM3v1bulkJctSideGradingCoeffGiven)
            model->BSIM3v1bulkJctSideGradingCoeff = 0.33;
        if (!model->BSIM3v1bulkJctGateSideGradingCoeffGiven)
            model->BSIM3v1bulkJctGateSideGradingCoeff = model->BSIM3v1bulkJctSideGradingCoeff;
        if (!model->BSIM3v1jctEmissionCoeffGiven)
            model->BSIM3v1jctEmissionCoeff = 1.0;
        if (!model->BSIM3v1jctTempExponentGiven)
            model->BSIM3v1jctTempExponent = 3.0;
        if (!model->BSIM3v1oxideTrapDensityAGiven)
	{   if (model->BSIM3v1type == NMOS)
                model->BSIM3v1oxideTrapDensityA = 1e20;
            else
                model->BSIM3v1oxideTrapDensityA=9.9e18;
	}
        if (!model->BSIM3v1oxideTrapDensityBGiven)
	{   if (model->BSIM3v1type == NMOS)
                model->BSIM3v1oxideTrapDensityB = 5e4;
            else
                model->BSIM3v1oxideTrapDensityB = 2.4e3;
	}
        if (!model->BSIM3v1oxideTrapDensityCGiven)
	{   if (model->BSIM3v1type == NMOS)
                model->BSIM3v1oxideTrapDensityC = -1.4e-12;
            else
                model->BSIM3v1oxideTrapDensityC = 1.4e-12;

	}
        if (!model->BSIM3v1emGiven)
            model->BSIM3v1em = 4.1e7; /* V/m */
        if (!model->BSIM3v1efGiven)
            model->BSIM3v1ef = 1.0;
        if (!model->BSIM3v1afGiven)
            model->BSIM3v1af = 1.0;
        if (!model->BSIM3v1kfGiven)
            model->BSIM3v1kf = 0.0;
        /* loop through all the instances of the model */
        for (here = BSIM3v1instances(model); here != NULL ;
             here=BSIM3v1nextInstance(here))
        {
            /* allocate a chunk of the state vector */
            here->BSIM3v1states = *states;
            *states += BSIM3v1numStates;

            /* perform the parameter defaulting */
            if(here->BSIM3v1m == 0.0)
              here->BSIM3v1m = 1.0;

            if (!here->BSIM3v1wGiven)
              here->BSIM3v1w = 5e-6;

            if (!here->BSIM3v1drainAreaGiven)
            {
                if(model->BSIM3v1hdifGiven)
                    here->BSIM3v1drainArea = here->BSIM3v1w * 2 * model->BSIM3v1hdif;
                else
                    here->BSIM3v1drainArea = 0.0;
            }
            if (!here->BSIM3v1drainPerimeterGiven)
            {
                if(model->BSIM3v1hdifGiven)
                    here->BSIM3v1drainPerimeter =
                        2 * here->BSIM3v1w + 4 * model->BSIM3v1hdif;
                else
                    here->BSIM3v1drainPerimeter = 0.0;
            }

            if (!here->BSIM3v1drainSquaresGiven)
                here->BSIM3v1drainSquares = 1.0;

            if (!here->BSIM3v1icVBSGiven)
                here->BSIM3v1icVBS = 0;
            if (!here->BSIM3v1icVDSGiven)
                here->BSIM3v1icVDS = 0;
            if (!here->BSIM3v1icVGSGiven)
                here->BSIM3v1icVGS = 0;
            if (!here->BSIM3v1lGiven)
                here->BSIM3v1l = 5e-6;
            if (!here->BSIM3v1sourceAreaGiven)
            {
                if(model->BSIM3v1hdifGiven)
                    here->BSIM3v1sourceArea = here->BSIM3v1w * 2 * model->BSIM3v1hdif;
                else
                    here->BSIM3v1sourceArea = 0.0;
            }

            if (!here->BSIM3v1sourcePerimeterGiven)
            {
                if(model->BSIM3v1hdifGiven)
                    here->BSIM3v1sourcePerimeter =
                        2 * here->BSIM3v1w + 4 * model->BSIM3v1hdif;
                else
                    here->BSIM3v1sourcePerimeter = 0.0;
            }

            if (!here->BSIM3v1sourceSquaresGiven)
                here->BSIM3v1sourceSquares = 1;

            if (!here->BSIM3v1wGiven)
                here->BSIM3v1w = 5e-6;

            if (!here->BSIM3v1mGiven)
                here->BSIM3v1m = 1;

            if (!here->BSIM3v1nqsModGiven)
                here->BSIM3v1nqsMod = model->BSIM3v1nqsMod;

            /* process drain series resistance */
            if ((model->BSIM3v1sheetResistance > 0.0) &&
                (here->BSIM3v1drainSquares > 0.0 ))
            {   if(here->BSIM3v1dNodePrime == 0)
            {   error = CKTmkVolt(ckt,&tmp,here->BSIM3v1name,"drain");
                if(error) return(error);
                here->BSIM3v1dNodePrime = tmp->number;

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
            {   here->BSIM3v1dNodePrime = here->BSIM3v1dNode;
            }

            /* process source series resistance */
            if ((model->BSIM3v1sheetResistance > 0.0) &&
                (here->BSIM3v1sourceSquares > 0.0 ))
            {   if(here->BSIM3v1sNodePrime == 0)
            {   error = CKTmkVolt(ckt,&tmp,here->BSIM3v1name,"source");
                if(error) return(error);
                here->BSIM3v1sNodePrime = tmp->number;

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
            {   here->BSIM3v1sNodePrime = here->BSIM3v1sNode;
            }

 /* internal charge node */

            if ((here->BSIM3v1nqsMod))
            {   if(here->BSIM3v1qNode == 0)
            {   error = CKTmkVolt(ckt,&tmp,here->BSIM3v1name,"charge");
                if(error) return(error);
                here->BSIM3v1qNode = tmp->number;
            }
            }
            else
            {   here->BSIM3v1qNode = 0;
            }

        /* set Sparse Matrix Pointers */

/* macro to make elements with built in test for out of memory */
#define TSTALLOC(ptr,first,second) \
do { if((here->ptr = SMPmakeElt(matrix, here->first, here->second)) == NULL){\
    return(E_NOMEM);\
} } while(0)

            TSTALLOC(BSIM3v1DdPtr, BSIM3v1dNode, BSIM3v1dNode);
            TSTALLOC(BSIM3v1GgPtr, BSIM3v1gNode, BSIM3v1gNode);
            TSTALLOC(BSIM3v1SsPtr, BSIM3v1sNode, BSIM3v1sNode);
            TSTALLOC(BSIM3v1BbPtr, BSIM3v1bNode, BSIM3v1bNode);
            TSTALLOC(BSIM3v1DPdpPtr, BSIM3v1dNodePrime, BSIM3v1dNodePrime);
            TSTALLOC(BSIM3v1SPspPtr, BSIM3v1sNodePrime, BSIM3v1sNodePrime);
            TSTALLOC(BSIM3v1DdpPtr, BSIM3v1dNode, BSIM3v1dNodePrime);
            TSTALLOC(BSIM3v1GbPtr, BSIM3v1gNode, BSIM3v1bNode);
            TSTALLOC(BSIM3v1GdpPtr, BSIM3v1gNode, BSIM3v1dNodePrime);
            TSTALLOC(BSIM3v1GspPtr, BSIM3v1gNode, BSIM3v1sNodePrime);
            TSTALLOC(BSIM3v1SspPtr, BSIM3v1sNode, BSIM3v1sNodePrime);
            TSTALLOC(BSIM3v1BdpPtr, BSIM3v1bNode, BSIM3v1dNodePrime);
            TSTALLOC(BSIM3v1BspPtr, BSIM3v1bNode, BSIM3v1sNodePrime);
            TSTALLOC(BSIM3v1DPspPtr, BSIM3v1dNodePrime, BSIM3v1sNodePrime);
            TSTALLOC(BSIM3v1DPdPtr, BSIM3v1dNodePrime, BSIM3v1dNode);
            TSTALLOC(BSIM3v1BgPtr, BSIM3v1bNode, BSIM3v1gNode);
            TSTALLOC(BSIM3v1DPgPtr, BSIM3v1dNodePrime, BSIM3v1gNode);
            TSTALLOC(BSIM3v1SPgPtr, BSIM3v1sNodePrime, BSIM3v1gNode);
            TSTALLOC(BSIM3v1SPsPtr, BSIM3v1sNodePrime, BSIM3v1sNode);
            TSTALLOC(BSIM3v1DPbPtr, BSIM3v1dNodePrime, BSIM3v1bNode);
            TSTALLOC(BSIM3v1SPbPtr, BSIM3v1sNodePrime, BSIM3v1bNode);
            TSTALLOC(BSIM3v1SPdpPtr, BSIM3v1sNodePrime, BSIM3v1dNodePrime);

            TSTALLOC(BSIM3v1QqPtr, BSIM3v1qNode, BSIM3v1qNode);

            TSTALLOC(BSIM3v1QdpPtr, BSIM3v1qNode, BSIM3v1dNodePrime);
            TSTALLOC(BSIM3v1QspPtr, BSIM3v1qNode, BSIM3v1sNodePrime);
            TSTALLOC(BSIM3v1QgPtr, BSIM3v1qNode, BSIM3v1gNode);
            TSTALLOC(BSIM3v1QbPtr, BSIM3v1qNode, BSIM3v1bNode);
            TSTALLOC(BSIM3v1DPqPtr, BSIM3v1dNodePrime, BSIM3v1qNode);
            TSTALLOC(BSIM3v1SPqPtr, BSIM3v1sNodePrime, BSIM3v1qNode);
            TSTALLOC(BSIM3v1GqPtr, BSIM3v1gNode, BSIM3v1qNode);
            TSTALLOC(BSIM3v1BqPtr, BSIM3v1bNode, BSIM3v1qNode);

        }
    }
    return(OK);
}


int
BSIM3v1unsetup(GENmodel *inModel, CKTcircuit *ckt)
{
    BSIM3v1model *model;
    BSIM3v1instance *here;

    for (model = (BSIM3v1model *)inModel; model != NULL;
            model = BSIM3v1nextModel(model))
    {
        for (here = BSIM3v1instances(model); here != NULL;
                here=BSIM3v1nextInstance(here))
        {
            if (here->BSIM3v1qNode > 0)
                CKTdltNNum(ckt, here->BSIM3v1qNode);
            here->BSIM3v1qNode = 0;

            if (here->BSIM3v1sNodePrime > 0
                    && here->BSIM3v1sNodePrime != here->BSIM3v1sNode)
                CKTdltNNum(ckt, here->BSIM3v1sNodePrime);
            here->BSIM3v1sNodePrime = 0;

            if (here->BSIM3v1dNodePrime > 0
                    && here->BSIM3v1dNodePrime != here->BSIM3v1dNode)
                CKTdltNNum(ckt, here->BSIM3v1dNodePrime);
            here->BSIM3v1dNodePrime = 0;
        }
    }
    return OK;
}


