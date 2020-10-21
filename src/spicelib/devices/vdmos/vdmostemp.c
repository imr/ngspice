/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: 2000 AlansFixes
VDMOS: 2018 Holger Vogt, 2020 Dietmar Warning
**********/

    /* perform the temperature update to the vdmos */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "vdmosdefs.h"
#include "ngspice/const.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

void VDMOStempUpdate(VDMOSmodel *inModel, VDMOSinstance *here, double Temp, CKTcircuit *ckt) {

    VDMOSmodel *model = (VDMOSmodel*)inModel;

    double egfet,egfet1;
    double fact1,fact2;
    double kt,kt1;
    double arg1;
    double ratio;
    double phio;
    double pbfact1,pbfact;
    double vt,vtnom;
    double xfc;

    fact1 = model->VDMOStnom/REFTEMP;
    vtnom = model->VDMOStnom*CONSTKoverQ;
    kt1 = CONSTboltz * model->VDMOStnom;
    egfet1 = 1.16-(7.02e-4*model->VDMOStnom*model->VDMOStnom)/
            (model->VDMOStnom+1108);
    arg1 = -egfet1/(kt1+kt1)+1.1150877/(CONSTboltz*(REFTEMP+REFTEMP));
    pbfact1 = -2*vtnom *(1.5*log(fact1)+CHARGE*arg1);

    xfc = log(1 - model->VDIOdepletionCapCoeff);

    double arg;     /* 1 - fc */

    double dt = Temp - model->VDMOStnom;

    /* vdmos temperature model */
    ratio = Temp/model->VDMOStnom;
    here->VDMOStTransconductance = model->VDMOStransconductance 
                                   * here->VDMOSm * pow(ratio, model->VDMOSmu);

    here->VDMOStVth = model->VDMOSvth0 - model->VDMOStype * model->VDMOStcvth * dt;

    here->VDMOStksubthres =  model->VDMOSksubthres * (1.0 + (model->VDMOStksubthres1 * dt) + (model->VDMOStksubthres2 * dt * dt));

    if (model->VDMOStexp0Given)
        here->VDMOSdrainResistance =  model->VDMOSdrainResistance / here->VDMOSm * pow(ratio, model->VDMOStexp0);
    else
        here->VDMOSdrainResistance =  model->VDMOSdrainResistance / here->VDMOSm * (1.0 + (model->VDMOStrd1 * dt) + (model->VDMOStrd2 * dt * dt));

    here->VDMOSgateConductance =  here->VDMOSgateConductance / (1.0 + (model->VDMOStrg1 * dt) + (model->VDMOStrg2 * dt * dt));

    here->VDMOSsourceConductance =  here->VDMOSsourceConductance / (1.0 + (model->VDMOStrs1 * dt) + (model->VDMOStrs2 * dt * dt));

    if (model->VDMOSqsGiven)
        here->VDMOSqsResistance = model->VDMOSqsResistance / here->VDMOSm * pow(ratio, model->VDMOStexp1);

    vt = Temp * CONSTKoverQ;
    fact2 = Temp/REFTEMP;
    kt = Temp * CONSTboltz;
    egfet = 1.16-(7.02e-4*Temp*Temp)/
            (Temp+1108);
    arg = -egfet/(kt+kt)+1.1150877/(CONSTboltz*(REFTEMP+REFTEMP));
    pbfact = -2*vt *(1.5*log(fact2)+CHARGE*arg);

    phio = (model->VDMOSphi - pbfact1) / fact1;
    here->VDMOStPhi = fact2 * phio + pbfact; /* needed for distortion analysis */

    /* body diode temperature model */
    double pbo, gmaold;
    double gmanew, factor;
    double tBreakdownVoltage, vte, cbv;
    double xbv, xcbv, tol, iter;
    double arg1_dT, arg2, arg2_dT;

    /* Junction grading temperature adjust */
    factor = 1.0 + (model->VDIOgradCoeffTemp1 * dt)
        + (model->VDIOgradCoeffTemp2 * dt * dt);
    here->VDIOtGradingCoeff = model->VDIOgradCoeff * factor;

    pbo = (model->VDIOjunctionPot - pbfact1) / fact1;
    gmaold = (model->VDIOjunctionPot - pbo) / pbo;
    here->VDIOtJctCap = here->VDMOSm * model->VDIOjunctionCap /
        (1 + here->VDIOtGradingCoeff*
        (400e-6*(model->VDMOStnom - REFTEMP) - gmaold));
    here->VDIOtJctPot = pbfact + fact2*pbo;
    gmanew = (here->VDIOtJctPot - pbo) / pbo;
    here->VDIOtJctCap *= 1 + here->VDIOtGradingCoeff*
        (400e-6*(Temp - REFTEMP) - gmanew);

    vte = model->VDIOn*vt;

    arg1 = ((Temp / model->VDMOStnom) - 1) * model->VDIOeg / vte;
    arg1_dT = model->VDIOeg / (vte*model->VDMOStnom)
              - model->VDIOeg*(Temp/model->VDMOStnom -1)/(vte*Temp);
    arg2 = model->VDIOxti / model->VDIOn * log(Temp / model->VDMOStnom);
    arg2_dT = model->VDIOxti / model->VDIOn / Temp;
    here->VDIOtSatCur = here->VDMOSm * model->VDIOjctSatCur * exp(arg1 + arg2);
    here->VDIOtSatCur_dT = here->VDMOSm * model->VDIOjctSatCur * exp(arg1 + arg2) * (arg1_dT + arg2_dT);

    /* the defintion of f1, just recompute after temperature adjusting
    * all the variables used in it */
    here->VDIOtF1 = here->VDIOtJctPot*
        (1 - exp((1 - here->VDIOtGradingCoeff)*xfc)) /
        (1 - here->VDIOtGradingCoeff);
    /* same for Depletion Capacitance */
    here->VDIOtDepCap = model->VDIOdepletionCapCoeff *
        here->VDIOtJctPot;

    /* and Vcrit */
    here->VDIOtVcrit = vte * log(vte / (CONSTroot2*here->VDIOtSatCur));

    /* limit junction potential to max of 1/FC */
    if (here->VDIOtDepCap > 2.5) {
        here->VDIOtJctPot = 2.5 / model->VDIOn;
        here->VDIOtDepCap = model->VDIOn*here->VDIOtJctPot;
        SPfrontEnd->IFerrorf(ERR_WARNING,
            "%s: junction potential VJ too large, limited to %f",
            model->VDMOSmodName, here->VDIOtJctPot);
    }

    /* and now to compute the breakdown voltage, again, using
    * temperature adjusted basic parameters */
    if (model->VDIObvGiven) {
        /* tlev == 0 */
        tBreakdownVoltage = fabs(model->VDIObv);

        cbv = model->VDIOibv;

        if (cbv < here->VDIOtSatCur * tBreakdownVoltage / vt) {
#ifdef TRACE
            cbv = here->VDIOtSatCur * tBreakdownVoltage / vt;
            SPfrontEnd->IFerrorf(ERR_WARNING, "%s: breakdown current increased to %g to resolve", here->VDMOSname, cbv);
            SPfrontEnd->IFerrorf(ERR_WARNING,
                "incompatibility with specified saturation current");
#endif
            xbv = tBreakdownVoltage;
        }
        else {
            tol = ckt->CKTreltol*cbv;
            xbv = tBreakdownVoltage - model->VDIObrkdEmissionCoeff*vt*log(1 + cbv /
                (here->VDIOtSatCur));
            for (iter = 0; iter < 25; iter++) {
                xbv = tBreakdownVoltage - model->VDIObrkdEmissionCoeff*vt*log(cbv /
                    (here->VDIOtSatCur) + 1 - xbv / vt);
                xcbv = here->VDIOtSatCur *
                    (exp((tBreakdownVoltage - xbv) / (model->VDIObrkdEmissionCoeff*vt)) - 1 + xbv / vt);
                if (fabs(xcbv - cbv) <= tol) goto matched;
            }
#ifdef TRACE
            SPfrontEnd->IFerrorf(ERR_WARNING, "%s: unable to match forward and reverse diode regions: bv = %g, ibv = %g", here->VDMOSname, xbv, xcbv);
#endif
        }
    matched:
        here->VDIOtBrkdwnV = xbv;
    }

    /* transit time temperature adjust */
    factor = 1.0 + (model->VDIOtranTimeTemp1 * dt)
                 + (model->VDIOtranTimeTemp2 * dt * dt);
    here->VDIOtTransitTime = model->VDIOtransitTime * factor;

    /* Series resistance temperature adjust */
    factor = 1.0 + (model->VDIOtrb1) * dt
                 + (model->VDIOtrb2 * dt * dt);
    here->VDIOtConductance = here->VDIOconductance / factor;
    here->VDIOtConductance_dT = -here->VDIOconductance * (model->VDIOtrb1 + model->VDIOtrb2 * dt) / (factor*factor);

    here->VDIOtF2 = exp((1 + here->VDIOtGradingCoeff)*xfc);
    here->VDIOtF3 = 1 - model->VDIOdepletionCapCoeff*
        (1 + here->VDIOtGradingCoeff);
}

int
VDMOStemp(GENmodel *inModel, CKTcircuit *ckt)
{
    VDMOSmodel *model = (VDMOSmodel*)inModel;
    VDMOSinstance *here;

    /*  loop through all the vdmos models */
    for( ; model != NULL; model = VDMOSnextModel(model)) {

        /* loop through all the instances */
        for(here=VDMOSinstances(model);here;here=VDMOSnextInstance(here)) {

            if(!here->VDMOSdtempGiven) here->VDMOSdtemp = 0.0;

            if(!here->VDMOStempGiven)
                here->VDMOStemp = ckt->CKTtemp + here->VDMOSdtemp;

            VDMOStempUpdate(model, here, here->VDMOStemp, ckt);

        } /* instance */

    } /* model */
    return(OK);
}
