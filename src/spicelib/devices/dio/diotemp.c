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

int
DIOtemp(GENmodel *inModel, CKTcircuit *ckt)
{
    DIOmodel *model = (DIOmodel*)inModel;
    double xfc, xfcs;
    double vte;
    double cbv;
    double xbv;
    double xcbv;
    double tol;
    double vt;
    double vtnom;
    DIOinstance *here;
    int iter;
    double dt;
    double factor;
    double tBreakdownVoltage;

    /*  loop through all the diode models */
    for( ; model != NULL; model = model->DIOnextModel ) {
        if(!model->DIOnomTempGiven) {
            model->DIOnomTemp = ckt->CKTnomTemp;
        }
        vtnom = CONSTKoverQ * model->DIOnomTemp;
        /* limit activation energy to min of .1 */
        if(model->DIOactivationEnergy<.1) {
            SPfrontEnd->IFerrorf (ERR_WARNING,
                    "%s: activation energy too small, limited to 0.1",
                    model->DIOmodName);
            model->DIOactivationEnergy=.1;
        }
        /* limit depletion cap coeff to max of .95 */
        if(model->DIOdepletionCapCoeff>.95) {
            SPfrontEnd->IFerrorf (ERR_WARNING,
                    "%s: coefficient Fc too large, limited to 0.95",
                    model->DIOmodName);
            model->DIOdepletionCapCoeff=.95;
        }
        /* limit sidewall depletion cap coeff to max of .95 */
        if(model->DIOdepletionSWcapCoeff>.95) {
            SPfrontEnd->IFerrorf (ERR_WARNING,
                    "%s: coefficient Fcs too large, limited to 0.95",
                    model->DIOmodName);
            model->DIOdepletionSWcapCoeff=.95;
        }
        /* set lower limit of saturation current */
        if (model->DIOsatCur < ckt->CKTepsmin)
            model->DIOsatCur = ckt->CKTepsmin;
        if((!model->DIOresistGiven) || (model->DIOresist==0)) {
            model->DIOconductance = 0.0;
        } else {
            model->DIOconductance = 1/model->DIOresist;
        }
        xfc=log(1-model->DIOdepletionCapCoeff);
        xfcs=log(1-model->DIOdepletionSWcapCoeff);

        for(here=model->DIOinstances;here;here=here->DIOnextInstance) {
            double egfet1,arg1,fact1,pbfact1,pbo,gmaold,pboSW,gmaSWold;
            double fact2,pbfact,arg,egfet,gmanew,gmaSWnew;
            /* loop through all the instances */

            if(!here->DIOdtempGiven) here->DIOdtemp = 0.0;

            if(!here->DIOtempGiven)
                here->DIOtemp = ckt->CKTtemp + here->DIOdtemp;

            dt = here->DIOtemp - model->DIOnomTemp;

            /* Junction grading temperature adjust */
            factor = 1.0 + (model->DIOgradCoeffTemp1 * dt)
                         + (model->DIOgradCoeffTemp2 * dt * dt);
            here->DIOtGradingCoeff = model->DIOgradingCoeff * factor;

            vt = CONSTKoverQ * here->DIOtemp;
            /* this part gets really ugly - I won't even try to
             * explain these equations */
            fact2 = here->DIOtemp/REFTEMP;
            egfet = 1.16-(7.02e-4*here->DIOtemp*here->DIOtemp)/
                    (here->DIOtemp+1108);
            arg = -egfet/(2*CONSTboltz*here->DIOtemp) +
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
                            (400e-6*(here->DIOtemp-REFTEMP)-gmanew);
            } else if (model->DIOtlevc == 1) {
                    here->DIOtJctPot = model->DIOjunctionPot - model->DIOtpb*(here->DIOtemp-REFTEMP);
                    here->DIOtJctCap = here->DIOjunctionCap *
                            (1+model->DIOcta*(here->DIOtemp-REFTEMP));
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
                            (400e-6*(here->DIOtemp-REFTEMP)-gmaSWnew);
            } else if (model->DIOtlevc == 1) {
                    here->DIOtJctSWPot = model->DIOjunctionSWPot - model->DIOtphp*(here->DIOtemp-REFTEMP);
                    here->DIOtJctSWCap = here->DIOjunctionSWCap *
                            (1+model->DIOctp*(here->DIOtemp-REFTEMP));
            }

            here->DIOtSatCur = model->DIOsatCur * here->DIOarea * exp(
                    ((here->DIOtemp/model->DIOnomTemp)-1) *
                    model->DIOactivationEnergy/(model->DIOemissionCoeff*vt) +
                    model->DIOsaturationCurrentExp/model->DIOemissionCoeff *
                    log(here->DIOtemp/model->DIOnomTemp) );
            here->DIOtSatSWCur = model->DIOsatSWCur * here->DIOpj * exp(
                    ((here->DIOtemp/model->DIOnomTemp)-1) *
                    model->DIOactivationEnergy/(model->DIOswEmissionCoeff*vt) +
                    model->DIOsaturationCurrentExp/model->DIOswEmissionCoeff *
                    log(here->DIOtemp/model->DIOnomTemp) );

            here->DIOtTunSatCur = model->DIOtunSatCur * here->DIOarea * exp(
                    ((here->DIOtemp/model->DIOnomTemp)-1) *
                    model->DIOtunEGcorrectionFactor*model->DIOactivationEnergy/vt +
                    model->DIOtunSaturationCurrentExp *
                    log(here->DIOtemp/model->DIOnomTemp) );
            here->DIOtTunSatSWCur = model->DIOtunSatSWCur * here->DIOpj * exp(
                    ((here->DIOtemp/model->DIOnomTemp)-1) *
                    model->DIOtunEGcorrectionFactor*model->DIOactivationEnergy/vt +
                    model->DIOtunSaturationCurrentExp *
                    log(here->DIOtemp/model->DIOnomTemp) );

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
                here->DIOtConductance = model->DIOconductance / factor;
            }

            here->DIOtF2=exp((1+here->DIOtGradingCoeff)*xfc);
            here->DIOtF3=1-model->DIOdepletionCapCoeff*
                    (1+here->DIOtGradingCoeff);
            here->DIOtF2SW=exp((1+model->DIOgradingSWCoeff)*xfcs);
            here->DIOtF3SW=1-model->DIOdepletionSWcapCoeff*
                    (1+model->DIOgradingSWCoeff);

        } /* instance */

    } /* model */
    return(OK);
}
