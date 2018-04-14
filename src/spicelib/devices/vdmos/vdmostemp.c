/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: 2000 AlansFixes
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "vdmosdefs.h"
#include "ngspice/const.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

int
VDMOStemp(GENmodel *inModel, CKTcircuit *ckt)
{
    VDMOSmodel *model = (VDMOSmodel *)inModel;
    VDMOSinstance *here;

    double egfet,egfet1;
    double fact1,fact2;
    double kt,kt1;
    double arg1;
    double ratio,ratio4;
    double phio;
    double pbfact1,pbfact;
    double vt,vtnom;
    double xfc;

    /* loop through all the resistor models */
    for( ; model != NULL; model = VDMOSnextModel(model)) {
        

        /* perform model defaulting */
        if(!model->VDMOStnomGiven) {
            model->VDMOStnom = ckt->CKTnomTemp;
        }

        fact1 = model->VDMOStnom/REFTEMP;
        vtnom = model->VDMOStnom*CONSTKoverQ;
        kt1 = CONSTboltz * model->VDMOStnom;
        egfet1 = 1.16-(7.02e-4*model->VDMOStnom*model->VDMOStnom)/
                (model->VDMOStnom+1108);
        arg1 = -egfet1/(kt1+kt1)+1.1150877/(CONSTboltz*(REFTEMP+REFTEMP));
        pbfact1 = -2*vtnom *(1.5*log(fact1)+CHARGE*arg1);

    /* now model parameter preprocessing */

        if (model->VDMOSphi <= 0.0) {
            SPfrontEnd->IFerrorf(ERR_FATAL,
                "%s: Phi is not positive.", model->VDMOSmodName);
            return(E_BADPARM);
        }

        model->VDMOSoxideCapFactor = 0;

        /* bulk diode model */
        /* limit activation energy to min of .1 */
        if (model->VDMOSDeg<.1) {
            SPfrontEnd->IFerrorf(ERR_WARNING,
                "%s: bulk diode activation energy too small, limited to 0.1",
                model->VDMOSmodName);
            model->VDMOSDeg = .1;
        }
        /* limit depletion cap coeff to max of .95 */
        if (model->VDIOdepletionCapCoeff>.95) {
            SPfrontEnd->IFerrorf(ERR_WARNING,
                "%s: coefficient Fc too large, limited to 0.95",
                model->VDMOSmodName);
            model->VDIOdepletionCapCoeff = .95;
        }
        /* set lower limit of saturation current */
        if (model->VDIOjctSatCur < ckt->CKTepsmin)
            model->VDIOjctSatCur = ckt->CKTepsmin;
        if ((!model->VDIOresistanceGiven) || (model->VDIOresistance == 0)) {
            model->VDIOconductance = 0.0;
        }
        else {
            model->VDIOconductance = 1 / model->VDIOresistance;
        }
        xfc = log(1 - model->VDIOdepletionCapCoeff);

        /* loop through all instances of the model */
        for(here = VDMOSinstances(model); here!= NULL; 
                here = VDMOSnextInstance(here)) {
            double arg;     /* 1 - fc */

            /* perform the parameter defaulting */
            
            if(!here->VDMOSdtempGiven) {
                here->VDMOSdtemp = 0.0;
            }
            if(!here->VDMOStempGiven) {
                here->VDMOStemp = ckt->CKTtemp + here->VDMOSdtemp;
            }
            vt = here->VDMOStemp * CONSTKoverQ;
            ratio = here->VDMOStemp/model->VDMOStnom;
            fact2 = here->VDMOStemp/REFTEMP;
            kt = here->VDMOStemp * CONSTboltz;
            egfet = 1.16-(7.02e-4*here->VDMOStemp*here->VDMOStemp)/
                    (here->VDMOStemp+1108);
            arg = -egfet/(kt+kt)+1.1150877/(CONSTboltz*(REFTEMP+REFTEMP));
            pbfact = -2*vt *(1.5*log(fact2)+CHARGE*arg);

            if(!here->VDMOSmGiven) {
                here->VDMOSm = 1;
            }
            if(!here->VDMOSlGiven) {
                here->VDMOSl = 1;
            }
            if(!here->VDMOSwGiven) {
                here->VDMOSw = 1;
            }

            ratio4 = ratio * sqrt(ratio);
            here->VDMOStTransconductance = model->VDMOStransconductance / ratio4;
            phio = (model->VDMOSphi - pbfact1) / fact1;
            here->VDMOStPhi = fact2 * phio + pbfact;
            here->VDMOStVto = model->VDMOSvt0;

            here->VDMOSCbd = 0;
            here->VDMOSf2d = 0;
            here->VDMOSf3d = 0;
            here->VDMOSf4d = 0;

            here->VDMOSCbs = 0;
            here->VDMOSf2s = 0;
            here->VDMOSf3s = 0;
            here->VDMOSf4s = 0;


            if (model->VDMOSdrainResistanceGiven) {
                if (model->VDMOSdrainResistance != 0) {
                    here->VDMOSdrainConductance = here->VDMOSm /
                        model->VDMOSdrainResistance;
                }
                else {
                    here->VDMOSdrainConductance = 0;
                }
            } else {
                here->VDMOSdrainConductance = 0;
            }
            if(model->VDMOSsourceResistanceGiven) {
                if(model->VDMOSsourceResistance != 0) {
                   here->VDMOSsourceConductance = here->VDMOSm /
                                         model->VDMOSsourceResistance;
                } else {
                    here->VDMOSsourceConductance = 0;
                }
            } else {
                here->VDMOSsourceConductance = 0;
            }
            if (model->VDMOSgateResistanceGiven) {
                if (model->VDMOSgateResistance != 0) {
                    here->VDMOSgateConductance = here->VDMOSm /
                        model->VDMOSgateResistance;
                } else {
                    here->VDMOSgateConductance = 0;
                }
            } else {
                here->VDMOSgateConductance = 0;
            }
            if (model->VDMOSrdsGiven) {
                if (model->VDMOSrds != 0) {
                    here->VDMOSdsConductance = here->VDMOSm /
                        model->VDMOSrds;
                }
                else {
                    here->VDMOSdsConductance = 0;
                }
            }
            else {
                here->VDMOSdsConductance = 0;
            }

            /* bulk diode model */
            double egfet1, arg1, fact1, pbfact1, pbo, gmaold;
            double fact2, pbfact, arg2, egfet, gmanew, factor;
            double tBreakdownVoltage, vte, cbv;
            double xbv, xcbv, tol, iter, dt;

            /* loop through all the instances */

            if (!here->VDMOSdtempGiven) here->VDMOStemp = 0.0;

            if (!here->VDMOStempGiven)
                here->VDMOStemp = ckt->CKTtemp + here->VDMOStemp;

            dt = here->VDMOStemp - model->VDMOStnom;

            /* Junction grading temperature adjust */
            factor = 1.0 + (model->VDIOgradCoeffTemp1 * dt)
                + (model->VDIOgradCoeffTemp2 * dt * dt);
            here->VDIOtGradingCoeff = model->VDIOgradCoeff * factor;

            vt = CONSTKoverQ * here->VDMOStemp;
            /* this part gets really ugly - I won't even try to
            * explain these equations */
            fact2 = here->VDMOStemp / REFTEMP;
            egfet = 1.16 - (7.02e-4*here->VDMOStemp*here->VDMOStemp) /
                (here->VDMOStemp + 1108);
            arg2 = -egfet / (2 * CONSTboltz*here->VDMOStemp) +
                1.1150877 / (CONSTboltz*(REFTEMP + REFTEMP));
            pbfact = -2 * vt*(1.5*log(fact2) + CHARGE*arg2);
            egfet1 = 1.16 - (7.02e-4*model->VDMOStnom*model->VDMOStnom) /
                (model->VDMOStnom + 1108);
            arg1 = -egfet1 / (CONSTboltz * 2 * model->VDMOStnom) +
                1.1150877 / (2 * CONSTboltz*REFTEMP);
            fact1 = model->VDMOStnom / REFTEMP;
            pbfact1 = -2 * vtnom*(1.5*log(fact1) + CHARGE*arg1);

            pbo = (model->VDIOjunctionPot - pbfact1) / fact1;
            gmaold = (model->VDIOjunctionPot - pbo) / pbo;
            here->VDIOtJctCap = model->VDIOjunctionCap /
                (1 + here->VDIOtGradingCoeff*
                (400e-6*(model->VDMOStnom - REFTEMP) - gmaold));
            here->VDIOtJctPot = pbfact + fact2*pbo;
            gmanew = (here->VDIOtJctPot - pbo) / pbo;
            here->VDIOtJctCap *= 1 + here->VDIOtGradingCoeff*
                (400e-6*(here->VDMOStemp - REFTEMP) - gmanew);

            here->VDIOtSatCur = model->VDIOjctSatCur * exp(
                ((here->VDMOStemp / model->VDMOStnom) - 1) *
                model->VDMOSDeg / (model->VDMOSDn*vt) +
                model->VDMOSDxti / model->VDMOSDn *
                log(here->VDMOStemp / model->VDMOStnom));

            /* the defintion of f1, just recompute after temperature adjusting
            * all the variables used in it */
            here->VDIOtF1 = here->VDIOtJctPot*
                (1 - exp((1 - here->VDIOtGradingCoeff)*xfc)) /
                (1 - here->VDIOtGradingCoeff);
            /* same for Depletion Capacitance */
            here->VDIOtDepCap = model->VDIOdepletionCapCoeff *
                here->VDIOtJctPot;

            /* and Vcrit */
            vte = model->VDMOSDn*vt;

            here->VDIOtVcrit = vte * log(vte / (CONSTroot2*here->VDIOtSatCur));

            /* limit junction potential to max of 1/FC */
            if (here->VDIOtDepCap > 1.0) {
                here->VDIOtJctPot = 1.0 / model->VDMOSDn;
                here->VDIOtDepCap = model->VDMOSDn*here->VDIOtJctPot;
                SPfrontEnd->IFerrorf(ERR_WARNING,
                    "%s: junction potential VJ too large, limited to %f",
                    model->VDMOSmodName, here->VDIOtJctPot);
            }

            /* and now to compute the breakdown voltage, again, using
            * temperature adjusted basic parameters */
            if (model->VDMOSDbvGiven) {
                /* tlev == 0 */
                tBreakdownVoltage = model->VDMOSDbv;

                cbv = model->VDMOSDibv;

                if (cbv < here->VDIOtSatCur * tBreakdownVoltage / vt) {
                    cbv = here->VDIOtSatCur * tBreakdownVoltage / vt;
#ifdef TRACE
                    SPfrontEnd->IFerrorf(ERR_WARNING, "%s: breakdown current increased to %g to resolve", here->DIOname, cbv);
                    SPfrontEnd->IFerrorf(ERR_WARNING,
                        "incompatibility with specified saturation current");
#endif
                    xbv = tBreakdownVoltage;
                }
                else {
                    tol = ckt->CKTreltol*cbv;
                    xbv = tBreakdownVoltage - model->VDIObrkdEmissionCoeff*vt*log(1 + cbv /
                        (here->VDIOtSatCur));
                    iter = 0;
                    for (iter = 0; iter < 25; iter++) {
                        xbv = tBreakdownVoltage - model->VDIObrkdEmissionCoeff*vt*log(cbv /
                            (here->VDIOtSatCur) + 1 - xbv / vt);
                        xcbv = here->VDIOtSatCur *
                            (exp((tBreakdownVoltage - xbv) / (model->VDIObrkdEmissionCoeff*vt)) - 1 + xbv / vt);
                        if (fabs(xcbv - cbv) <= tol) goto matched;
                    }
#ifdef TRACE
                    SPfrontEnd->IFerrorf(ERR_WARNING, "%s: unable to match forward and reverse diode regions: bv = %g, ibv = %g", here->DIOname, xbv, xcbv);
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
            here->VDIOtConductance = model->VDIOconductance;
            if (model->VDIOresistanceGiven && model->VDIOresistance != 0.0) {
                factor = 1.0 + (model->VDIOresistTemp1) * dt
                    + (model->VDIOresistTemp2 * dt * dt);
                here->VDIOtConductance = model->VDIOconductance / factor;
            }

            here->VDIOtF2 = exp((1 + here->VDIOtGradingCoeff)*xfc);
            here->VDIOtF3 = 1 - model->VDIOdepletionCapCoeff*
                (1 + here->VDIOtGradingCoeff);
        }
    }
    return(OK);
}
