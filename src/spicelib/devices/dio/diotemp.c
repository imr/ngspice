/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: 2000 AlansFixes
Modified by Paolo Nenzi 2003 and Dietmar Warning 2012
**********/

/* perform the temperature update to the diode */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "diodefs.h"
#include "ngspice/const.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"
#include "ngspice/cpdefs.h"

void DIOtempUpdate(DIOmodel *inModel, DIOinstance *here, double Temp, CKTcircuit *ckt) {

    DIOmodel *model = (DIOmodel*)inModel;
    double xfc, xfcs;
    double vt, vte, vts, vtt, vtr;
    double cbv;
    double xbv;
    double xcbv;
    double tol;
    double vtnom;
    int iter;
    double dt;
    double factor;
    double tBreakdownVoltage;

    double egfet1,arg1,fact1,pbfact1,pbo,gmaold,pboSW,gmaSWold;
    double fact2,pbfact,arg,egfet,gmanew,gmaSWnew;
    double arg1_dT, arg2, arg2_dT;
    double lnTRatio, egfet_dT = 0.0, arg0, vte_dT, vts_dT, vtt_dT, vtr_dT;

    vt = CONSTKoverQ * Temp;
    vte = model->DIOemissionCoeff * vt;
    vte_dT = CONSTKoverQ * model->DIOemissionCoeff;
    vts = model->DIOswEmissionCoeff * vt;
    vts_dT = CONSTKoverQ * model->DIOswEmissionCoeff;
    vtt = model->DIOtunEmissionCoeff * vt;
    vtt_dT = CONSTKoverQ * model->DIOtunEmissionCoeff;
    vtr = model->DIOrecEmissionCoeff * vt;
    vtr_dT = CONSTKoverQ * model->DIOrecEmissionCoeff;
    vtnom = CONSTKoverQ * model->DIOnomTemp;
    dt = Temp - model->DIOnomTemp;
    lnTRatio = log(Temp / model->DIOnomTemp);

    /* Junction grading temperature adjust */
    factor = 1.0 + (model->DIOgradCoeffTemp1 * dt)
                 + (model->DIOgradCoeffTemp2 * dt * dt);
    here->DIOtGradingCoeff = model->DIOgradingCoeff * factor;

    /* this part gets really ugly - I won't even try to
     * explain these equations */
    if ((model->DIOtlev == 0) || (model->DIOtlev == 1)) {
        egfet = 1.16-(7.02e-4*Temp*Temp)/
                (Temp+1108);
        egfet1 = 1.16 - (7.02e-4*model->DIOnomTemp*model->DIOnomTemp)/
                (model->DIOnomTemp+1108);
    } else {
        egfet = model->DIOactivationEnergy-(model->DIOfirstBGcorrFactor*Temp*Temp)/
                (Temp+model->DIOsecndBGcorrFactor);
        egfet_dT = (model->DIOfirstBGcorrFactor*Temp*Temp)/
                   ((Temp+model->DIOsecndBGcorrFactor)*(Temp+model->DIOsecndBGcorrFactor))
                   - 2*model->DIOfirstBGcorrFactor*Temp/(Temp+model->DIOsecndBGcorrFactor);
        egfet1 = model->DIOactivationEnergy - (model->DIOfirstBGcorrFactor*model->DIOnomTemp*model->DIOnomTemp)/
                (model->DIOnomTemp+model->DIOsecndBGcorrFactor);
    }
    fact2 = Temp/REFTEMP;
    arg = -egfet/(2*CONSTboltz*Temp) +
            1.1150877/(CONSTboltz*(REFTEMP+REFTEMP));
    pbfact = -2*vt*(1.5*log(fact2)+CHARGE*arg);
    arg1 = -egfet1/(CONSTboltz*2*model->DIOnomTemp) +
            1.1150877/(2*CONSTboltz*REFTEMP);
    fact1 = model->DIOnomTemp/REFTEMP;
    pbfact1 = -2 * vtnom*(1.5*log(fact1)+CHARGE*arg1);

    if (model->DIOtlevc == 0) {
            pbo = (model->DIOjunctionPot-pbfact1)/fact1;
            gmaold = (model->DIOjunctionPot-pbo)/pbo;
            here->DIOtJctCap = here->DIOjunctionCap /
                    (1+here->DIOtGradingCoeff*
                    (400e-6*(model->DIOnomTemp-REFTEMP)-gmaold) );
            here->DIOtJctPot = pbfact+fact2*pbo;
            gmanew = (here->DIOtJctPot-pbo)/pbo;
            here->DIOtJctCap *= 1+here->DIOtGradingCoeff*
                    (400e-6*(Temp-REFTEMP)-gmanew);
    } else if (model->DIOtlevc == 1) {
            here->DIOtJctPot = model->DIOjunctionPot - model->DIOtpb*(Temp-REFTEMP);
            here->DIOtJctCap = here->DIOjunctionCap *
                    (1+model->DIOcta*(Temp-REFTEMP));
    }

    if (model->DIOtlevc == 0) {
            pboSW = (model->DIOjunctionSWPot-pbfact1)/fact1;
            gmaSWold = (model->DIOjunctionSWPot-pboSW)/pboSW;
            here->DIOtJctSWCap = here->DIOjunctionSWCap /
                    (1+model->DIOgradingSWCoeff*
                    (400e-6*(model->DIOnomTemp-REFTEMP)-gmaSWold) );
            here->DIOtJctSWPot = pbfact+fact2*pboSW;
            gmaSWnew = (here->DIOtJctSWPot-pboSW)/pboSW;
            here->DIOtJctSWCap *= 1+model->DIOgradingSWCoeff*
                    (400e-6*(Temp-REFTEMP)-gmaSWnew);
    } else if (model->DIOtlevc == 1) {
            here->DIOtJctSWPot = model->DIOjunctionSWPot - model->DIOtphp*(Temp-REFTEMP);
            here->DIOtJctSWCap = here->DIOjunctionSWCap *
                    (1+model->DIOctp*(Temp-REFTEMP));
    }

    if ((model->DIOtlev == 0) || (model->DIOtlev == 1)) {
        arg1 = ((Temp / model->DIOnomTemp) - 1) * model->DIOactivationEnergy / vte;
        arg1_dT = model->DIOactivationEnergy / (vte*model->DIOnomTemp)
                  - model->DIOactivationEnergy*(Temp/model->DIOnomTemp -1)/(vte*Temp);
        arg2 = model->DIOsaturationCurrentExp / model->DIOemissionCoeff * lnTRatio;
        arg2_dT = model->DIOsaturationCurrentExp / model->DIOemissionCoeff / Temp;
        here->DIOtSatCur = model->DIOsatCur * here->DIOarea * exp(arg1 + arg2);
        here->DIOtSatCur_dT = here->DIOtSatCur * (arg1_dT + arg2_dT);

        arg1 = ((Temp / model->DIOnomTemp) - 1) * model->DIOactivationEnergy / vts;
        arg1_dT = model->DIOactivationEnergy / (vts*model->DIOnomTemp)
                  - model->DIOactivationEnergy*(Temp/model->DIOnomTemp -1)/(vts*Temp);
        arg2 = model->DIOsaturationCurrentExp / model->DIOswEmissionCoeff * lnTRatio;
        arg2_dT = model->DIOsaturationCurrentExp / model->DIOswEmissionCoeff / Temp;
        here->DIOtSatSWCur = model->DIOsatSWCur * here->DIOpj * exp(arg1 + arg2);
        here->DIOtSatSWCur_dT = here->DIOtSatSWCur * (arg1_dT + arg2_dT);

        arg1 = ((Temp/model->DIOnomTemp)-1) * model->DIOtunEGcorrectionFactor*model->DIOactivationEnergy / vtt;
        arg1_dT = model->DIOtunEGcorrectionFactor*model->DIOactivationEnergy / (vtt*model->DIOnomTemp)
                  - model->DIOactivationEnergy*(Temp/model->DIOnomTemp -1)/(vtt*Temp);
        arg2 = model->DIOtunSaturationCurrentExp / model->DIOtunEmissionCoeff * lnTRatio;
        arg2_dT = model->DIOtunSaturationCurrentExp / model->DIOtunEmissionCoeff / Temp;
        here->DIOtTunSatCur = model->DIOtunSatCur * here->DIOarea * exp(arg1 + arg2);
        here->DIOtTunSatCur_dT = here->DIOtTunSatCur * (arg1_dT + arg2_dT);

        here->DIOtTunSatSWCur = model->DIOtunSatSWCur * here->DIOpj * exp(arg1 + arg2);
        here->DIOtTunSatSWCur_dT = here->DIOtTunSatSWCur * (arg1_dT + arg2_dT);

        arg1 = ((Temp / model->DIOnomTemp) - 1) * model->DIOactivationEnergy / vtr;
        arg1_dT = model->DIOactivationEnergy / (vtr*model->DIOnomTemp)
                  - model->DIOactivationEnergy*(Temp/model->DIOnomTemp -1)/(vtr*Temp);
        arg2 = model->DIOsaturationCurrentExp / model->DIOrecEmissionCoeff * lnTRatio;
        arg2_dT = model->DIOsaturationCurrentExp / model->DIOrecEmissionCoeff / Temp;
        here->DIOtRecSatCur = model->DIOrecSatCur * here->DIOarea * exp(arg1 + arg2);
        here->DIOtRecSatCur_dT = here->DIOtRecSatCur * (arg1_dT + arg2_dT);
    } else {
        arg0 = egfet1 / (model->DIOemissionCoeff * vtnom);
        arg1 = egfet / vte;
        arg1_dT = (egfet_dT * vte - egfet * vte_dT) / (egfet*egfet);
        arg2 = model->DIOsaturationCurrentExp / model->DIOemissionCoeff * lnTRatio;
        arg2_dT = model->DIOsaturationCurrentExp / model->DIOemissionCoeff / Temp;
        here->DIOtSatCur = model->DIOsatCur * here->DIOarea * exp(arg0 - arg1 + arg2);
        here->DIOtSatCur_dT = here->DIOtSatCur * (-arg1_dT + arg2_dT);

        arg0 = egfet1 / (model->DIOswEmissionCoeff * vtnom);
        arg1 = egfet / vts;
        arg1_dT = (egfet_dT * vts - egfet * vts_dT) / (egfet*egfet);
        arg2 = model->DIOsaturationCurrentExp / model->DIOswEmissionCoeff * lnTRatio;
        arg2_dT = model->DIOsaturationCurrentExp / model->DIOswEmissionCoeff / Temp;
        here->DIOtSatSWCur = model->DIOsatSWCur * here->DIOpj * exp(arg0 - arg1 + arg2);
        here->DIOtSatSWCur_dT = here->DIOtSatSWCur * (-arg1_dT + arg2_dT);

        arg0 = model->DIOtunEGcorrectionFactor * egfet1 / (model->DIOtunEmissionCoeff * vtnom);
        arg1 = model->DIOtunEGcorrectionFactor * egfet / vtt;
        arg1_dT = model->DIOtunEGcorrectionFactor * (egfet_dT * vtt - egfet * vtt_dT) / (egfet*egfet);
        arg2 = model->DIOtunSaturationCurrentExp / model->DIOtunEmissionCoeff * lnTRatio;
        arg2_dT = model->DIOtunSaturationCurrentExp / model->DIOtunEmissionCoeff / Temp;
        here->DIOtTunSatCur = model->DIOtunSatCur * here->DIOarea * exp(arg0 - arg1 + arg2);
        here->DIOtTunSatCur_dT = here->DIOtTunSatCur * (-arg1_dT + arg2_dT);

        here->DIOtTunSatSWCur = model->DIOtunSatSWCur * here->DIOpj * exp(arg0 - arg1 + arg2);
        here->DIOtTunSatSWCur_dT = here->DIOtTunSatSWCur * (-arg1_dT + arg2_dT);

        arg0 = egfet1 / (model->DIOrecEmissionCoeff * vtnom);
        arg1 = egfet / vtr;
        arg1_dT = (egfet_dT * vtr - egfet * vtr_dT) / (egfet*egfet);
        arg2 = model->DIOsaturationCurrentExp / model->DIOrecEmissionCoeff * lnTRatio;
        arg2_dT = model->DIOsaturationCurrentExp / model->DIOrecEmissionCoeff / Temp;
        here->DIOtRecSatCur = model->DIOrecSatCur * here->DIOarea * exp(arg0 - arg1 + arg2);
        here->DIOtRecSatCur_dT = here->DIOtRecSatCur * (-arg1_dT + arg2_dT);
    }

    xfc=log(1-model->DIOdepletionCapCoeff);
    xfcs=log(1-model->DIOdepletionSWcapCoeff);

    /* the defintion of f1, just recompute after temperature adjusting
     * all the variables used in it */
    here->DIOtF1=here->DIOtJctPot*
            (1-exp((1-here->DIOtGradingCoeff)*xfc))/
            (1-here->DIOtGradingCoeff);
    /* same for Depletion Capacitance */
    here->DIOtDepCap=model->DIOdepletionCapCoeff*
            here->DIOtJctPot;
    here->DIOtDepSWCap=model->DIOdepletionSWcapCoeff*
            here->DIOtJctSWPot;
    /* and Vcrit */
    here->DIOtVcrit = vte * log(vte/(CONSTroot2*here->DIOtSatCur));

    /* and now to compute the breakdown voltage, again, using
     * temperature adjusted basic parameters */
    if (model->DIObreakdownVoltageGiven){
        if (model->DIOtlev == 0) {
            tBreakdownVoltage = model->DIObreakdownVoltage - model->DIOtcv * dt;
        } else {
            tBreakdownVoltage = model->DIObreakdownVoltage * (1 - model->DIOtcv * dt);
        }
        if (model->DIOlevel == 1) {
            cbv = here->DIOm * model->DIObreakdownCurrent;
        } else { /* level=3 */
            cbv = model->DIObreakdownCurrent * here->DIOarea;
        }
        if (cbv < here->DIOtSatCur * tBreakdownVoltage/vt) {
#ifdef TRACE
            cbv=here->DIOtSatCur * tBreakdownVoltage/vt;
            SPfrontEnd->IFerrorf (ERR_WARNING, "%s: breakdown current increased to %g to resolve", here->DIOname, cbv);
            SPfrontEnd->IFerrorf (ERR_WARNING,
            "incompatibility with specified saturation current");
#endif
            xbv=tBreakdownVoltage;
        } else {
            tol=ckt->CKTreltol*cbv;
            xbv=tBreakdownVoltage-model->DIObrkdEmissionCoeff*vt*log(1+cbv/
                    (here->DIOtSatCur));
            for(iter=0 ; iter < 25 ; iter++) {
                xbv=tBreakdownVoltage-model->DIObrkdEmissionCoeff*vt*log(cbv/
                        (here->DIOtSatCur)+1-xbv/vt);
                xcbv=here->DIOtSatCur *
                     (exp((tBreakdownVoltage-xbv)/(model->DIObrkdEmissionCoeff*vt))-1+xbv/vt);
                if (fabs(xcbv-cbv) <= tol) goto matched;
            }
#ifdef TRACE
            SPfrontEnd->IFerrorf (ERR_WARNING, "%s: unable to match forward and reverse diode regions: bv = %g, ibv = %g", here->DIOname, xbv, xcbv);
#endif
        }
        matched:
            here->DIOtBrkdwnV = xbv;
    }

    /* transit time temperature adjust */
    factor = 1.0 + (model->DIOtranTimeTemp1 * dt)
                 + (model->DIOtranTimeTemp2 * dt * dt);
    here->DIOtTransitTime = model->DIOtransitTime * factor;

    /* Series resistance temperature adjust */
    here->DIOtConductance = model->DIOconductance * here->DIOarea;
    if(model->DIOresistGiven && model->DIOresist!=0.0) {
        factor = 1.0 + (model->DIOresistTemp1) * dt
                 + (model->DIOresistTemp2 * dt * dt);
        here->DIOtConductance = model->DIOconductance * here->DIOarea / factor;
        here->DIOtConductance_dT = -model->DIOconductance * here->DIOarea *
            (model->DIOresistTemp1 + model->DIOresistTemp2 * dt) / (factor*factor);
    }

    here->DIOtF2=exp((1+here->DIOtGradingCoeff)*xfc);
    here->DIOtF3=1-model->DIOdepletionCapCoeff*
            (1+here->DIOtGradingCoeff);
    here->DIOtF2SW=exp((1+model->DIOgradingSWCoeff)*xfcs);
    here->DIOtF3SW=1-model->DIOdepletionSWcapCoeff*
            (1+model->DIOgradingSWCoeff);
}

int
DIOtemp(GENmodel *inModel, CKTcircuit *ckt)
{
    DIOmodel *model = (DIOmodel*)inModel;
    DIOinstance *here;

    /*  loop through all the diode models */
    for( ; model != NULL; model = DIOnextModel(model)) {

        /* loop through all the instances */
        for(here=DIOinstances(model);here;here=DIOnextInstance(here)) {

            if(!here->DIOdtempGiven) here->DIOdtemp = 0.0;

            if(!here->DIOtempGiven)
                here->DIOtemp = ckt->CKTtemp + here->DIOdtemp;

            DIOtempUpdate(model, here, here->DIOtemp, ckt);

        } /* instance */

    } /* model */
    return(OK);
}
