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
    double vte;
    double cbv;
    double xbv;
    double xcbv;
    double tol;
    double vt;
    double vtnom;
    int iter;
    double dt;
    double factor;
    double tBreakdownVoltage;

    double egfet1,arg1,fact1,pbfact1,pbo,gmaold,pboSW,gmaSWold;
    double fact2,pbfact,arg,egfet,gmanew,gmaSWnew;
    double arg1_dT, arg2, arg2_dT;
    double gclimit;

    if (!cp_getvar("DIOgradingCoeffMax", CP_REAL, &gclimit, 0))
        gclimit = 0.9;

    vt = CONSTKoverQ * Temp;
    vte = model->DIOemissionCoeff * vt;
    vtnom = CONSTKoverQ * model->DIOnomTemp;
    dt = Temp - model->DIOnomTemp;

    /* Junction grading temperature adjust */
    factor = 1.0 + (model->DIOgradCoeffTemp1 * dt)
                 + (model->DIOgradCoeffTemp2 * dt * dt);
    here->DIOtGradingCoeff = model->DIOgradingCoeff * factor;

    /* limit temperature adjusted grading coeff
     * to max of .9, or set new limit with variable DIOgradingCoeffMax
     */
    if(here->DIOtGradingCoeff>gclimit) {
      SPfrontEnd->IFerrorf (ERR_WARNING,
            "%s: temperature adjusted grading coefficient too large, limited to %g",
            here->DIOname, gclimit);
      here->DIOtGradingCoeff=gclimit;
    }

    /* this part gets really ugly - I won't even try to
     * explain these equations */
    fact2 = Temp/REFTEMP;
    egfet = 1.16-(7.02e-4*Temp*Temp)/
            (Temp+1108);
    arg = -egfet/(2*CONSTboltz*Temp) +
            1.1150877/(CONSTboltz*(REFTEMP+REFTEMP));
    pbfact = -2*vt*(1.5*log(fact2)+CHARGE*arg);
    egfet1 = 1.16 - (7.02e-4*model->DIOnomTemp*model->DIOnomTemp)/
            (model->DIOnomTemp+1108);
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

//    here->DIOtSatCur = model->DIOsatCur * here->DIOarea * exp(
//            ((here->DIOtemp/model->DIOnomTemp)-1) *
//            model->DIOactivationEnergy/(model->DIOemissionCoeff*vt) +
//            model->DIOsaturationCurrentExp/model->DIOemissionCoeff *
//            log(here->DIOtemp/model->DIOnomTemp) );     
    arg1 = ((Temp / model->DIOnomTemp) - 1) * model->DIOactivationEnergy / (model->DIOemissionCoeff*vt);
    arg1_dT = model->DIOactivationEnergy / (vte*model->DIOnomTemp)
              - model->DIOactivationEnergy*(Temp/model->DIOnomTemp -1)/(vte*Temp);

    arg2 = model->DIOsaturationCurrentExp / model->DIOemissionCoeff * log(Temp / model->DIOnomTemp);
    arg2_dT = model->DIOsaturationCurrentExp / model->DIOemissionCoeff / Temp;

    here->DIOtSatCur = here->DIOm * model->DIOsatCur * here->DIOarea * exp(arg1 + arg2);
    here->DIOtSatCur_dT = here->DIOm * model->DIOsatCur * here->DIOarea * exp(arg1 + arg2) * (arg1_dT + arg2_dT);

//here->DIOtSatSWCur = model->DIOsatSWCur * here->DIOpj * exp(
//        ((here->DIOtemp/model->DIOnomTemp)-1) *
//        model->DIOactivationEnergy/(model->DIOswEmissionCoeff*vt) +
//        model->DIOsaturationCurrentExp/model->DIOswEmissionCoeff *
//        log(here->DIOtemp/model->DIOnomTemp) );
    arg1 = ((Temp / model->DIOnomTemp) - 1) * model->DIOactivationEnergy / (model->DIOswEmissionCoeff*vt);
    arg1_dT = model->DIOactivationEnergy / (vte*model->DIOnomTemp)
              - model->DIOactivationEnergy*(Temp/model->DIOnomTemp -1)/(vte*Temp);

    arg2 = model->DIOsaturationCurrentExp / model->DIOswEmissionCoeff * log(Temp / model->DIOnomTemp);
    arg2_dT = model->DIOsaturationCurrentExp / model->DIOemissionCoeff / Temp;

    here->DIOtSatSWCur = here->DIOm * model->DIOsatSWCur * here->DIOpj * exp(arg1 + arg2);
    here->DIOtSatSWCur_dT = here->DIOm * model->DIOsatSWCur * here->DIOpj * exp(arg1 + arg2) * (arg1_dT + arg2_dT);

//here->DIOtTunSatCur = model->DIOtunSatCur * here->DIOarea * exp(
//        ((here->DIOtemp/model->DIOnomTemp)-1) *
//        model->DIOtunEGcorrectionFactor*model->DIOactivationEnergy/vt +
//        model->DIOtunSaturationCurrentExp *
//        log(here->DIOtemp/model->DIOnomTemp) );     here->DIOtTunSatCur = here->DIOm * model->DIOtunSatCur * here->DIOarea * exp(arg1 + arg2);
    arg1 = ((Temp / model->DIOnomTemp) - 1) * model->DIOtunEGcorrectionFactor*model->DIOactivationEnergy / (model->DIOemissionCoeff*vt);
    arg1_dT = model->DIOtunEGcorrectionFactor*model->DIOactivationEnergy / (vte*model->DIOnomTemp)
              - model->DIOactivationEnergy*(Temp/model->DIOnomTemp -1)/(vte*Temp);

    arg2 = model->DIOtunSaturationCurrentExp / model->DIOemissionCoeff * log(Temp / model->DIOnomTemp);
    arg2_dT = model->DIOtunSaturationCurrentExp / model->DIOemissionCoeff / Temp;

    here->DIOtTunSatCur_dT = here->DIOm * model->DIOtunSatCur * here->DIOarea * exp(arg1 + arg2) * (arg1_dT + arg2_dT);

//here->DIOtTunSatSWCur = model->DIOtunSatSWCur * here->DIOpj * exp(
//        ((here->DIOtemp/model->DIOnomTemp)-1) *
//        model->DIOtunEGcorrectionFactor*model->DIOactivationEnergy/vt +
//        model->DIOtunSaturationCurrentExp *
//        log(here->DIOtemp/model->DIOnomTemp) );     arg1 = ((Temp / model->DIOnomTemp) - 1) * model->DIOtunEGcorrectionFactor*model->DIOactivationEnergy / (model->DIOemissionCoeff*vt);
    arg1_dT = model->DIOtunEGcorrectionFactor*model->DIOactivationEnergy / (vte*model->DIOnomTemp)
              - model->DIOactivationEnergy*(Temp/model->DIOnomTemp -1)/(vte*Temp);

    arg2 = model->DIOtunSaturationCurrentExp / model->DIOemissionCoeff * log(Temp / model->DIOnomTemp);
    arg2_dT = model->DIOtunSaturationCurrentExp / model->DIOemissionCoeff / Temp;

    here->DIOtTunSatSWCur = here->DIOm * model->DIOtunSatSWCur * here->DIOpj * exp(arg1 + arg2);
    here->DIOtTunSatSWCur_dT = here->DIOm * model->DIOtunSatSWCur * here->DIOpj * exp(arg1 + arg2) * (arg1_dT + arg2_dT);

//    here->DIOtRecSatCur = model->DIOrecSatCur * here->DIOarea * exp(
//            ((Temp/model->DIOnomTemp)-1) *
//            model->DIOactivationEnergy/(model->DIOrecEmissionCoeff*vt) +
//            model->DIOsaturationCurrentExp/model->DIOrecEmissionCoeff *
//            log(Temp/model->DIOnomTemp) );
    arg1 = ((Temp / model->DIOnomTemp) - 1) * model->DIOtunEGcorrectionFactor*model->DIOactivationEnergy / (model->DIOemissionCoeff*vt);
    arg1_dT = model->DIOtunEGcorrectionFactor*model->DIOactivationEnergy / (vte*model->DIOnomTemp)
              - model->DIOactivationEnergy*(Temp/model->DIOnomTemp -1)/(vte*Temp);

    arg2 = model->DIOtunSaturationCurrentExp / model->DIOrecEmissionCoeff * log(Temp / model->DIOnomTemp);
    arg2_dT = model->DIOtunSaturationCurrentExp / model->DIOrecEmissionCoeff / Temp;

    here->DIOtRecSatCur = here->DIOm * model->DIOrecSatCur * here->DIOarea * exp(arg1 + arg2);
    here->DIOtRecSatCur_dT = here->DIOm * model->DIOrecSatCur * here->DIOarea * exp(arg1 + arg2) * (arg1_dT + arg2_dT);

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
    vte=model->DIOemissionCoeff*vt;

    here->DIOtVcrit = vte * log(vte/(CONSTroot2*here->DIOtSatCur));

    /* limit junction potential to max of 1/FC */
    if(here->DIOtDepCap > 1.0) {
        here->DIOtJctPot=1.0/model->DIOdepletionCapCoeff;
        here->DIOtDepCap=model->DIOdepletionCapCoeff*here->DIOtJctPot;
        SPfrontEnd->IFerrorf (ERR_WARNING,
                "%s: junction potential VJ too large, limited to %f",
                model->DIOmodName, here->DIOtJctPot);
    }
    /* limit sidewall junction potential to max of 1/FCS */
    if(here->DIOtDepSWCap > 1.0) {
        here->DIOtJctSWPot=1.0/model->DIOdepletionSWcapCoeff;
        here->DIOtDepSWCap=model->DIOdepletionSWcapCoeff*here->DIOtJctSWPot;
        SPfrontEnd->IFerrorf (ERR_WARNING,
                "%s: junction potential VJS too large, limited to %f",
                model->DIOmodName, here->DIOtJctSWPot);
    }

    /* and now to compute the breakdown voltage, again, using
     * temperature adjusted basic parameters */
    if (model->DIObreakdownVoltageGiven){
        if (model->DIOtlev == 0) {
            tBreakdownVoltage = model->DIObreakdownVoltage - model->DIOtcv * dt;
        } else {
            tBreakdownVoltage = model->DIObreakdownVoltage * (1 - model->DIOtcv * dt);
        }
        if (model->DIOlevel == 1) {
            cbv = model->DIObreakdownCurrent;
        } else { /* level=3 */
            cbv = model->DIObreakdownCurrent * here->DIOarea;
        }
        if (cbv < here->DIOtSatCur * tBreakdownVoltage/vt) {
            cbv=here->DIOtSatCur * tBreakdownVoltage/vt;
#ifdef TRACE
        SPfrontEnd->IFerrorf (ERR_WARNING, "%s: breakdown current increased to %g to resolve", here->DIOname, cbv);
        SPfrontEnd->IFerrorf (ERR_WARNING,
        "incompatibility with specified saturation current");
#endif
        xbv=tBreakdownVoltage;
    } else {
        tol=ckt->CKTreltol*cbv;
        xbv=tBreakdownVoltage-model->DIObrkdEmissionCoeff*vt*log(1+cbv/
                (here->DIOtSatCur));
        iter=0;
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
    here->DIOtConductance = model->DIOconductance;
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
        /* set lower limit of saturation current */
        if (model->DIOsatCur < ckt->CKTepsmin)
            model->DIOsatCur = ckt->CKTepsmin;

        if(!model->DIOnomTempGiven) {
            model->DIOnomTemp = ckt->CKTnomTemp;
        }

        if((!model->DIOresistGiven) || (model->DIOresist==0)) {
            model->DIOconductance = 0.0;
        } else {
            model->DIOconductance = 1/model->DIOresist;
        }

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
