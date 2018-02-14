/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: Alan Gillespie
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "mos9defs.h"
#include "ngspice/const.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

/* assuming silicon - make definition for epsilon of silicon */
#define EPSSIL (11.7 * 8.854214871e-12)

int
MOS9temp(GENmodel *inModel, CKTcircuit *ckt)
{
    MOS9model *model = (MOS9model *)inModel;
    MOS9instance *here;
    double wkfngs;
    double wkfng;
    double fermig;
    double fermis;
    double vfb;
    double fact1,fact2;
    double vt,vtnom;
    double kt,kt1;
    double ratio,ratio4;
    double egfet,egfet1;
    double pbfact,pbfact1,pbo;
    double phio;
    double arg1;
    double capfact;
    double gmanew,gmaold;
    double ni_temp, nifact;
    /* loop through all the mosfet models */
    for( ; model != NULL; model = MOS9nextModel(model)) {

        if(!model->MOS9tnomGiven) {
            model->MOS9tnom = ckt->CKTnomTemp;
        }
        fact1 = model->MOS9tnom/REFTEMP;
        vtnom = model->MOS9tnom*CONSTKoverQ;
        kt1 = CONSTboltz * model->MOS9tnom;
        egfet1 = 1.16-(7.02e-4*model->MOS9tnom*model->MOS9tnom)/
                (model->MOS9tnom+1108);
        arg1 = -egfet1/(kt1+kt1)+1.1150877/(CONSTboltz*(REFTEMP+REFTEMP));
        pbfact1 = -2*vtnom *(1.5*log(fact1)+CHARGE*arg1);

        nifact=(model->MOS9tnom/300)*sqrt(model->MOS9tnom/300);
        nifact*=exp(0.5*egfet1*((1/(double)300)-(1/model->MOS9tnom))/
                                                                  CONSTKoverQ);
        ni_temp=1.45e16*nifact;

        if (model->MOS9phi <= 0.0) {
            SPfrontEnd->IFerrorf (ERR_FATAL,
               "%s: Phi is not positive.", model->MOS9modName);
            return(E_BADPARM);
        }

        model->MOS9oxideCapFactor = 3.9 * 8.854214871e-12/
                model->MOS9oxideThickness;
        if(!model->MOS9surfaceMobilityGiven) model->MOS9surfaceMobility=600;
        if(!model->MOS9transconductanceGiven) {
            model->MOS9transconductance = model->MOS9surfaceMobility *
                    model->MOS9oxideCapFactor * 1e-4;
        }
        if(model->MOS9substrateDopingGiven) {
            if(model->MOS9substrateDoping*1e6 /*(cm**3/m**3)*/ >ni_temp) {

                if(!model->MOS9phiGiven) {
                    model->MOS9phi = 2*vtnom*
                            log(model->MOS9substrateDoping*

                            1e6/*(cm**3/m**3)*//ni_temp);

                    model->MOS9phi = MAX(.1,model->MOS9phi);
                }
                fermis = model->MOS9type * .5 * model->MOS9phi;
                wkfng = 3.2;
                if(!model->MOS9gateTypeGiven) model->MOS9gateType=1;
                if(model->MOS9gateType != 0) {
                    fermig = model->MOS9type * model->MOS9gateType*.5*egfet1;
                    wkfng = 3.25 + .5 * egfet1 - fermig;
                }
                wkfngs = wkfng - (3.25 + .5 * egfet1 +fermis);
                if(!model->MOS9gammaGiven) {
                    model->MOS9gamma = sqrt(2 * EPSSIL *
                        CHARGE * model->MOS9substrateDoping*
                        1e6 /*(cm**3/m**3)*/ )/ model->MOS9oxideCapFactor;
                }
                if(!model->MOS9vt0Given) {
                    if(!model->MOS9surfaceStateDensityGiven)
                            model->MOS9surfaceStateDensity=0;
                    vfb = wkfngs - model->MOS9surfaceStateDensity * 1e4
                            * CHARGE/model->MOS9oxideCapFactor;
                    model->MOS9vt0 = vfb + model->MOS9type *
                            (model->MOS9gamma * sqrt(model->MOS9phi)+
                             model->MOS9phi);
                } else {
                    vfb = model->MOS9vt0 - model->MOS9type * (model->MOS9gamma*
                        sqrt(model->MOS9phi)+model->MOS9phi);
                }
                model->MOS9alpha = (EPSSIL+EPSSIL)/
                    (CHARGE*model->MOS9substrateDoping*1e6 /*(cm**3/m**3)*/ );
                model->MOS9coeffDepLayWidth = sqrt(model->MOS9alpha);
            } else {
                model->MOS9substrateDoping = 0;
                SPfrontEnd->IFerrorf (ERR_FATAL,
                        "%s: Nsub < Ni ", model->MOS9modName);
                return(E_BADPARM);
            }
        }
    /* now model parameter preprocessing */
        model->MOS9narrowFactor = model->MOS9delta * 0.5 * M_PI * EPSSIL /
            model->MOS9oxideCapFactor ;


        /* loop through all instances of the model */
        for(here = MOS9instances(model); here!= NULL;
                here = MOS9nextInstance(here)) {

            double czbd;    /* zero voltage bulk-drain capacitance */
            double czbdsw;  /* zero voltage bulk-drain sidewall capacitance */
            double czbs;    /* zero voltage bulk-source capacitance */
            double czbssw;  /* zero voltage bulk-source sidewall capacitance */
            double arg;     /* 1 - fc */
            double sarg;    /* (1-fc) ^^ (-mj) */
            double sargsw;  /* (1-fc) ^^ (-mjsw) */

            /* perform the parameter defaulting */

             if(!here->MOS9dtempGiven) {
                here->MOS9dtemp = 0.0;
            }

            if(!here->MOS9tempGiven) {
                here->MOS9temp = ckt->CKTtemp + here->MOS9dtemp;
            }
            vt = here->MOS9temp * CONSTKoverQ;
            ratio = here->MOS9temp/model->MOS9tnom;
            fact2 = here->MOS9temp/REFTEMP;
            kt = here->MOS9temp * CONSTboltz;
            egfet = 1.16-(7.02e-4*here->MOS9temp*here->MOS9temp)/
                    (here->MOS9temp+1108);
            arg = -egfet/(kt+kt)+1.1150877/(CONSTboltz*(REFTEMP+REFTEMP));
            pbfact = -2*vt *(1.5*log(fact2)+CHARGE*arg);

            if(!here->MOS9mGiven) {
                here->MOS9m = ckt->CKTdefaultMosM;
            }

            if(!here->MOS9lGiven) {
                here->MOS9l = ckt->CKTdefaultMosL;
            }
            if(!here->MOS9sourceAreaGiven) {
                here->MOS9sourceArea = ckt->CKTdefaultMosAS;
            }
            if(!here->MOS9wGiven) {
                here->MOS9w = ckt->CKTdefaultMosW;
            }
            if(model->MOS9drainResistanceGiven) {
                if(model->MOS9drainResistance != 0) {
                    here->MOS9drainConductance = here->MOS9m /
                                                   model->MOS9drainResistance;
                } else {
                    here->MOS9drainConductance = 0;
                }
            } else if (model->MOS9sheetResistanceGiven) {
                if ((model->MOS9sheetResistance != 0) &&
                                              (here->MOS9drainSquares != 0)) {
                    here->MOS9drainConductance =
                        here->MOS9m /
                          (model->MOS9sheetResistance*here->MOS9drainSquares);
                } else {
                    here->MOS9drainConductance = 0;
                }
            } else {
                here->MOS9drainConductance = 0;
            }
            if(model->MOS9sourceResistanceGiven) {
                if(model->MOS9sourceResistance != 0) {
                    here->MOS9sourceConductance = here->MOS9m /
                                                    model->MOS9sourceResistance;
                } else {
                    here->MOS9sourceConductance = 0;
                }
            } else if (model->MOS9sheetResistanceGiven) {
                if ((model->MOS9sheetResistance != 0) &&
                                              (here->MOS9sourceSquares != 0)) {
                    here->MOS9sourceConductance =
                        here->MOS9m /
                          (model->MOS9sheetResistance*here->MOS9sourceSquares);
                } else {
                    here->MOS9sourceConductance = 0;
                }
            } else {
                here->MOS9sourceConductance = 0;
            }

            if(here->MOS9l - 2 * model->MOS9latDiff +
                                 model->MOS9lengthAdjust <1e-6) {
                SPfrontEnd->IFerrorf (ERR_FATAL,
                        "%s: effective channel length less than zero",
                        here->MOS9name);
                return(E_PARMVAL);
            }

            if(here->MOS9w - 2 * model->MOS9widthNarrow +
                                 model->MOS9widthAdjust <1e-6) {
                SPfrontEnd->IFerrorf (ERR_FATAL,
                        "%s: effective channel width less than zero",
                        here->MOS9name);
                return(E_PARMVAL);
            }


            ratio4 = ratio * sqrt(ratio);
            here->MOS9tTransconductance = model->MOS9transconductance / ratio4;
            here->MOS9tSurfMob = model->MOS9surfaceMobility/ratio4;
            phio= (model->MOS9phi-pbfact1)/fact1;
            here->MOS9tPhi = fact2 * phio + pbfact;
            here->MOS9tVbi =
                    model->MOS9delvt0 +
                    model->MOS9vt0 - model->MOS9type *
                        (model->MOS9gamma* sqrt(model->MOS9phi))
                    +.5*(egfet1-egfet)
                    + model->MOS9type*.5* (here->MOS9tPhi-model->MOS9phi);
            here->MOS9tVto = here->MOS9tVbi + model->MOS9type *
                    model->MOS9gamma * sqrt(here->MOS9tPhi);
            here->MOS9tSatCur = model->MOS9jctSatCur*
                    exp(-egfet/vt+egfet1/vtnom);
            here->MOS9tSatCurDens = model->MOS9jctSatCurDensity *
                    exp(-egfet/vt+egfet1/vtnom);
            pbo = (model->MOS9bulkJctPotential - pbfact1)/fact1;
            gmaold = (model->MOS9bulkJctPotential-pbo)/pbo;
            capfact = 1/(1+model->MOS9bulkJctBotGradingCoeff*
                    (4e-4*(model->MOS9tnom-REFTEMP)-gmaold));
            here->MOS9tCbd = model->MOS9capBD * capfact;
            here->MOS9tCbs = model->MOS9capBS * capfact;
            here->MOS9tCj = model->MOS9bulkCapFactor * capfact;
            capfact = 1/(1+model->MOS9bulkJctSideGradingCoeff*
                    (4e-4*(model->MOS9tnom-REFTEMP)-gmaold));
            here->MOS9tCjsw = model->MOS9sideWallCapFactor * capfact;
            here->MOS9tBulkPot = fact2 * pbo+pbfact;
            gmanew = (here->MOS9tBulkPot-pbo)/pbo;
            capfact = (1+model->MOS9bulkJctBotGradingCoeff*
                    (4e-4*(here->MOS9temp-REFTEMP)-gmanew));
            here->MOS9tCbd *= capfact;
            here->MOS9tCbs *= capfact;
            here->MOS9tCj *= capfact;
            capfact = (1+model->MOS9bulkJctSideGradingCoeff*
                    (4e-4*(here->MOS9temp-REFTEMP)-gmanew));
            here->MOS9tCjsw *= capfact;
            here->MOS9tDepCap = model->MOS9fwdCapDepCoeff * here->MOS9tBulkPot;

            if( (model->MOS9jctSatCurDensity == 0) ||
                    (here->MOS9drainArea == 0) ||
                    (here->MOS9sourceArea == 0) ) {
                here->MOS9sourceVcrit = here->MOS9drainVcrit =
                       vt*log(vt/(CONSTroot2*here->MOS9m*here->MOS9tSatCur));
            } else {
                here->MOS9drainVcrit =
                        vt * log( vt / (CONSTroot2 *
                        here->MOS9m *
                        here->MOS9tSatCurDens * here->MOS9drainArea));
                here->MOS9sourceVcrit =
                        vt * log( vt / (CONSTroot2 *
                        here->MOS9m *
                        here->MOS9tSatCurDens * here->MOS9sourceArea));
            }
            if(model->MOS9capBDGiven) {
                czbd = here->MOS9tCbd * here->MOS9m;
            } else {
                if(model->MOS9bulkCapFactorGiven) {
                    czbd=here->MOS9tCj*here->MOS9drainArea * here->MOS9m;
                } else {
                    czbd=0;
                }
            }
            if(model->MOS9sideWallCapFactorGiven) {
                czbdsw= here->MOS9tCjsw * here->MOS9drainPerimiter *
                         here->MOS9m;
            } else {
                czbdsw=0;
            }
            arg = 1-model->MOS9fwdCapDepCoeff;
            sarg = exp( (-model->MOS9bulkJctBotGradingCoeff) * log(arg) );
            sargsw = exp( (-model->MOS9bulkJctSideGradingCoeff) * log(arg) );
            here->MOS9Cbd = czbd;
            here->MOS9Cbdsw = czbdsw;
            here->MOS9f2d = czbd*(1-model->MOS9fwdCapDepCoeff*
                        (1+model->MOS9bulkJctBotGradingCoeff))* sarg/arg
                    +  czbdsw*(1-model->MOS9fwdCapDepCoeff*
                        (1+model->MOS9bulkJctSideGradingCoeff))*
                        sargsw/arg;
            here->MOS9f3d = czbd * model->MOS9bulkJctBotGradingCoeff * sarg/arg/
                        here->MOS9tBulkPot
                    + czbdsw * model->MOS9bulkJctSideGradingCoeff * sargsw/arg /
                        here->MOS9tBulkPot;
            here->MOS9f4d = czbd*here->MOS9tBulkPot*(1-arg*sarg)/
                        (1-model->MOS9bulkJctBotGradingCoeff)
                    + czbdsw*here->MOS9tBulkPot*(1-arg*sargsw)/
                        (1-model->MOS9bulkJctSideGradingCoeff)
                    -here->MOS9f3d/2*
                        (here->MOS9tDepCap*here->MOS9tDepCap)
                    -here->MOS9tDepCap * here->MOS9f2d;
            if(model->MOS9capBSGiven) {
                czbs = here->MOS9tCbs * here->MOS9m;
            } else {
                if(model->MOS9bulkCapFactorGiven) {
                    czbs=here->MOS9tCj*here->MOS9sourceArea * here->MOS9m;
                } else {
                    czbs=0;
                }
            }
            if(model->MOS9sideWallCapFactorGiven) {
                czbssw = here->MOS9tCjsw * here->MOS9sourcePerimiter *
                         here->MOS9m;
            } else {
                czbssw=0;
            }
            arg = 1-model->MOS9fwdCapDepCoeff;
            sarg = exp( (-model->MOS9bulkJctBotGradingCoeff) * log(arg) );
            sargsw = exp( (-model->MOS9bulkJctSideGradingCoeff) * log(arg) );
            here->MOS9Cbs = czbs;
            here->MOS9Cbssw = czbssw;
            here->MOS9f2s = czbs*(1-model->MOS9fwdCapDepCoeff*
                        (1+model->MOS9bulkJctBotGradingCoeff))* sarg/arg
                    +  czbssw*(1-model->MOS9fwdCapDepCoeff*
                        (1+model->MOS9bulkJctSideGradingCoeff))*
                        sargsw/arg;
            here->MOS9f3s = czbs * model->MOS9bulkJctBotGradingCoeff * sarg/arg/
                        here->MOS9tBulkPot
                    + czbssw * model->MOS9bulkJctSideGradingCoeff * sargsw/arg /
                        here->MOS9tBulkPot;
            here->MOS9f4s = czbs*here->MOS9tBulkPot*(1-arg*sarg)/
                        (1-model->MOS9bulkJctBotGradingCoeff)
                    + czbssw*here->MOS9tBulkPot*(1-arg*sargsw)/
                        (1-model->MOS9bulkJctSideGradingCoeff)
                    -here->MOS9f3s/2*
                        (here->MOS9tDepCap*here->MOS9tDepCap)
                    -here->MOS9tDepCap * here->MOS9f2s;
        }
    }
    return(OK);
}
