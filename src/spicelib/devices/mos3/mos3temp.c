/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: 2000 AlansFixes
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "mos3defs.h"
#include "ngspice/const.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

/* assuming silicon - make definition for epsilon of silicon */
#define EPSSIL (11.7 * 8.854214871e-12)

int
MOS3temp(GENmodel *inModel, CKTcircuit *ckt)
{
    MOS3model *model = (MOS3model *)inModel;
    MOS3instance *here;
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
    for( ; model != NULL; model = MOS3nextModel(model)) {
        
        if(!model->MOS3tnomGiven) {
            model->MOS3tnom = ckt->CKTnomTemp;
        }
        fact1 = model->MOS3tnom/REFTEMP;
        vtnom = model->MOS3tnom*CONSTKoverQ;
        kt1 = CONSTboltz * model->MOS3tnom;
        egfet1 = 1.16-(7.02e-4*model->MOS3tnom*model->MOS3tnom)/
                (model->MOS3tnom+1108);
        arg1 = -egfet1/(kt1+kt1)+1.1150877/(CONSTboltz*(REFTEMP+REFTEMP));
        pbfact1 = -2*vtnom *(1.5*log(fact1)+CHARGE*arg1);
        nifact=(model->MOS3tnom/300)*sqrt(model->MOS3tnom/300);
        nifact*=exp(0.5*egfet1*((1/(double)300)-(1/model->MOS3tnom))/
                                                 CONSTKoverQ);
        ni_temp=1.45e16*nifact;

        if (model->MOS3phi <= 0.0) {
            SPfrontEnd->IFerrorf (ERR_FATAL,
               "%s: Phi is not positive.", model->MOS3modName);
            return(E_BADPARM);
        }

        model->MOS3oxideCapFactor = 3.9 * 8.854214871e-12/
                model->MOS3oxideThickness;
        if(!model->MOS3surfaceMobilityGiven) model->MOS3surfaceMobility=600;
        if(!model->MOS3transconductanceGiven) {
            model->MOS3transconductance = model->MOS3surfaceMobility *
                    model->MOS3oxideCapFactor * 1e-4;
        }
        if(model->MOS3substrateDopingGiven) {
            if(model->MOS3substrateDoping*1e6 /*(cm**3/m**3)*/ >ni_temp) {
                if(!model->MOS3phiGiven) {
                    model->MOS3phi = 2*vtnom*
                            log(model->MOS3substrateDoping*
                           1e6/*(cm**3/m**3)*//ni_temp);
                    model->MOS3phi = MAX(.1,model->MOS3phi);
                }
                fermis = model->MOS3type * .5 * model->MOS3phi;
                wkfng = 3.2;
                if(!model->MOS3gateTypeGiven) model->MOS3gateType=1;
                if(model->MOS3gateType != 0) {
                    fermig = model->MOS3type * model->MOS3gateType*.5*egfet1;
                    wkfng = 3.25 + .5 * egfet1 - fermig;
                }
                wkfngs = wkfng - (3.25 + .5 * egfet1 +fermis);
                if(!model->MOS3gammaGiven) {
                    model->MOS3gamma = sqrt(2 * EPSSIL *
                        CHARGE * model->MOS3substrateDoping*
                        1e6 /*(cm**3/m**3)*/ )/ model->MOS3oxideCapFactor;
                }
                if(!model->MOS3vt0Given) {
                    if(!model->MOS3surfaceStateDensityGiven) 
                            model->MOS3surfaceStateDensity=0;
                    vfb = wkfngs - model->MOS3surfaceStateDensity * 1e4 
                            * CHARGE/model->MOS3oxideCapFactor;
                    model->MOS3vt0 = vfb + model->MOS3type * 
                            (model->MOS3gamma * sqrt(model->MOS3phi)+
                             model->MOS3phi);
                } else {
                    vfb = model->MOS3vt0 - model->MOS3type * (model->MOS3gamma*
                        sqrt(model->MOS3phi)+model->MOS3phi);
                }
                model->MOS3alpha = (EPSSIL+EPSSIL)/
                    (CHARGE*model->MOS3substrateDoping*1e6 /*(cm**3/m**3)*/ );
                model->MOS3coeffDepLayWidth = sqrt(model->MOS3alpha);
            } else {
                model->MOS3substrateDoping = 0;
                SPfrontEnd->IFerrorf (ERR_FATAL,
                        "%s: Nsub < Ni ", model->MOS3modName);
                return(E_BADPARM);
            }
        }
    /* now model parameter preprocessing */
        model->MOS3narrowFactor = model->MOS3delta * 0.5 * M_PI * EPSSIL / 
            model->MOS3oxideCapFactor ;

    
        /* loop through all instances of the model */
        for(here = MOS3instances(model); here!= NULL; 
                here = MOS3nextInstance(here)) {

            double czbd;    /* zero voltage bulk-drain capacitance */
            double czbdsw;  /* zero voltage bulk-drain sidewall capacitance */
            double czbs;    /* zero voltage bulk-source capacitance */
            double czbssw;  /* zero voltage bulk-source sidewall capacitance */
            double arg;     /* 1 - fc */
            double sarg;    /* (1-fc) ^^ (-mj) */
            double sargsw;  /* (1-fc) ^^ (-mjsw) */

            /* perform the parameter defaulting */

             if(!here->MOS3dtempGiven) {
                here->MOS3dtemp = 0.0;
            }

            if(!here->MOS3tempGiven) {
                here->MOS3temp = ckt->CKTtemp + here->MOS3dtemp;
            }
            vt = here->MOS3temp * CONSTKoverQ;
            ratio = here->MOS3temp/model->MOS3tnom;
            fact2 = here->MOS3temp/REFTEMP;
            kt = here->MOS3temp * CONSTboltz;
            egfet = 1.16-(7.02e-4*here->MOS3temp*here->MOS3temp)/
                    (here->MOS3temp+1108);
            arg = -egfet/(kt+kt)+1.1150877/(CONSTboltz*(REFTEMP+REFTEMP));
            pbfact = -2*vt *(1.5*log(fact2)+CHARGE*arg);

            if(!here->MOS3mGiven) {
                here->MOS3m = ckt->CKTdefaultMosM;
            }
            if(!here->MOS3lGiven) {
                here->MOS3l = ckt->CKTdefaultMosL;
            }
            if(!here->MOS3sourceAreaGiven) {
                here->MOS3sourceArea = ckt->CKTdefaultMosAS;
            }
            if(!here->MOS3wGiven) {
                here->MOS3w = ckt->CKTdefaultMosW;
            }
            if(model->MOS3drainResistanceGiven) {
                if(model->MOS3drainResistance != 0) {
                   here->MOS3drainConductance = here->MOS3m /
                                     model->MOS3drainResistance;
                } else {
                    here->MOS3drainConductance = 0;
                }
            } else if (model->MOS3sheetResistanceGiven) {
                if ((model->MOS3sheetResistance != 0) &&
                                              (here->MOS3drainSquares != 0)) {
                    here->MOS3drainConductance = 
                       here->MOS3m /
                          (model->MOS3sheetResistance*here->MOS3drainSquares);
                } else {
                    here->MOS3drainConductance = 0;
                }
            } else {
                here->MOS3drainConductance = 0;
            }
            if(model->MOS3sourceResistanceGiven) {
                if(model->MOS3sourceResistance != 0) {
                    here->MOS3sourceConductance = here->MOS3m /
                                                    model->MOS3sourceResistance;
                } else {
                    here->MOS3sourceConductance = 0;
                }
            } else if (model->MOS3sheetResistanceGiven) {
                if ((model->MOS3sheetResistance != 0) &&
                                   (here->MOS3sourceSquares != 0)) {
                    here->MOS3sourceConductance = 
                       here->MOS3m /
                          (model->MOS3sheetResistance*here->MOS3sourceSquares);
                } else {
                    here->MOS3sourceConductance = 0;
                }
            } else {
                here->MOS3sourceConductance = 0;
            }

            if(here->MOS3l - 2 * model->MOS3latDiff +
                                 model->MOS3lengthAdjust <= 0) {
                SPfrontEnd->IFerrorf (ERR_FATAL,
                        "%s: effective channel length less than zero",
                        here->MOS3name);
                return(E_PARMVAL);
            }

            if(here->MOS3w - 2 * model->MOS3widthNarrow +
                                 model->MOS3widthAdjust <= 0) {
                SPfrontEnd->IFerrorf (ERR_FATAL,
                        "%s: effective channel width less than zero",
                        here->MOS3name);
                return(E_PARMVAL);
            }

            ratio4 = ratio * sqrt(ratio);
            here->MOS3tTransconductance = model->MOS3transconductance / ratio4;
            here->MOS3tSurfMob = model->MOS3surfaceMobility/ratio4;
            phio= (model->MOS3phi-pbfact1)/fact1;
            here->MOS3tPhi = fact2 * phio + pbfact;
            here->MOS3tVbi =
                    model->MOS3delvt0 + 
                    model->MOS3vt0 - model->MOS3type * 
                        (model->MOS3gamma* sqrt(model->MOS3phi))
                    +.5*(egfet1-egfet) 
                    + model->MOS3type*.5* (here->MOS3tPhi-model->MOS3phi);
            here->MOS3tVto = here->MOS3tVbi + model->MOS3type * 
                    model->MOS3gamma * sqrt(here->MOS3tPhi);
            here->MOS3tSatCur = model->MOS3jctSatCur* 
                    exp(-egfet/vt+egfet1/vtnom);
            here->MOS3tSatCurDens = model->MOS3jctSatCurDensity *
                    exp(-egfet/vt+egfet1/vtnom);
            pbo = (model->MOS3bulkJctPotential - pbfact1)/fact1;
            gmaold = (model->MOS3bulkJctPotential-pbo)/pbo;
            capfact = 1/(1+model->MOS3bulkJctBotGradingCoeff*
                    (4e-4*(model->MOS3tnom-REFTEMP)-gmaold));
            here->MOS3tCbd = model->MOS3capBD * capfact;
            here->MOS3tCbs = model->MOS3capBS * capfact;
            here->MOS3tCj = model->MOS3bulkCapFactor * capfact;
            capfact = 1/(1+model->MOS3bulkJctSideGradingCoeff*
                    (4e-4*(model->MOS3tnom-REFTEMP)-gmaold));
            here->MOS3tCjsw = model->MOS3sideWallCapFactor * capfact;
            here->MOS3tBulkPot = fact2 * pbo+pbfact;
            gmanew = (here->MOS3tBulkPot-pbo)/pbo;
            capfact = (1+model->MOS3bulkJctBotGradingCoeff*
                    (4e-4*(here->MOS3temp-REFTEMP)-gmanew));
            here->MOS3tCbd *= capfact;
            here->MOS3tCbs *= capfact;
            here->MOS3tCj *= capfact;
            capfact = (1+model->MOS3bulkJctSideGradingCoeff*
                    (4e-4*(here->MOS3temp-REFTEMP)-gmanew));
            here->MOS3tCjsw *= capfact;
            here->MOS3tDepCap = model->MOS3fwdCapDepCoeff * here->MOS3tBulkPot;

            if( (model->MOS3jctSatCurDensity == 0) ||
                    (here->MOS3drainArea == 0) ||
                    (here->MOS3sourceArea == 0) ) {
                here->MOS3sourceVcrit = here->MOS3drainVcrit =
                       vt*log(vt/(CONSTroot2*here->MOS3m*here->MOS3tSatCur));
            } else {
                here->MOS3drainVcrit =
                        vt * log( vt / (CONSTroot2 *
                        here->MOS3m *
                        here->MOS3tSatCurDens * here->MOS3drainArea));
                here->MOS3sourceVcrit =
                        vt * log( vt / (CONSTroot2 *
                        here->MOS3m *
                        here->MOS3tSatCurDens * here->MOS3sourceArea));
            }
            if(model->MOS3capBDGiven) {
                czbd = here->MOS3tCbd * here->MOS3m;
            } else {
                if(model->MOS3bulkCapFactorGiven) {
                    czbd=here->MOS3tCj*here->MOS3drainArea * here->MOS3m;
                } else {
                    czbd=0;
                }
            }
            if(model->MOS3sideWallCapFactorGiven) {
                czbdsw= here->MOS3tCjsw * here->MOS3drainPerimiter *
                         here->MOS3m;
            } else {
                czbdsw=0;
            }
            arg = 1-model->MOS3fwdCapDepCoeff;
            sarg = exp( (-model->MOS3bulkJctBotGradingCoeff) * log(arg) );
            sargsw = exp( (-model->MOS3bulkJctSideGradingCoeff) * log(arg) );
            here->MOS3Cbd = czbd;
            here->MOS3Cbdsw = czbdsw;
            here->MOS3f2d = czbd*(1-model->MOS3fwdCapDepCoeff*
                        (1+model->MOS3bulkJctBotGradingCoeff))* sarg/arg
                    +  czbdsw*(1-model->MOS3fwdCapDepCoeff*
                        (1+model->MOS3bulkJctSideGradingCoeff))*
                        sargsw/arg;
            here->MOS3f3d = czbd * model->MOS3bulkJctBotGradingCoeff * sarg/arg/
                        here->MOS3tBulkPot
                    + czbdsw * model->MOS3bulkJctSideGradingCoeff * sargsw/arg /
                        here->MOS3tBulkPot;
            here->MOS3f4d = czbd*here->MOS3tBulkPot*(1-arg*sarg)/
                        (1-model->MOS3bulkJctBotGradingCoeff)
                    + czbdsw*here->MOS3tBulkPot*(1-arg*sargsw)/
                        (1-model->MOS3bulkJctSideGradingCoeff)
                    -here->MOS3f3d/2*
                        (here->MOS3tDepCap*here->MOS3tDepCap)
                    -here->MOS3tDepCap * here->MOS3f2d;
            if(model->MOS3capBSGiven) {
                czbs = here->MOS3tCbs * here->MOS3m;
            } else {
                if(model->MOS3bulkCapFactorGiven) {
                    czbs=here->MOS3tCj*here->MOS3sourceArea * here->MOS3m;
                } else {
                    czbs=0;
                }
            }
            if(model->MOS3sideWallCapFactorGiven) {
                czbssw = here->MOS3tCjsw * here->MOS3sourcePerimiter *
                      here->MOS3m;
            } else {
                czbssw=0;
            }
            arg = 1-model->MOS3fwdCapDepCoeff;
            sarg = exp( (-model->MOS3bulkJctBotGradingCoeff) * log(arg) );
            sargsw = exp( (-model->MOS3bulkJctSideGradingCoeff) * log(arg) );
            here->MOS3Cbs = czbs;
            here->MOS3Cbssw = czbssw;
            here->MOS3f2s = czbs*(1-model->MOS3fwdCapDepCoeff*
                        (1+model->MOS3bulkJctBotGradingCoeff))* sarg/arg
                    +  czbssw*(1-model->MOS3fwdCapDepCoeff*
                        (1+model->MOS3bulkJctSideGradingCoeff))*
                        sargsw/arg;
            here->MOS3f3s = czbs * model->MOS3bulkJctBotGradingCoeff * sarg/arg/
                       here->MOS3tBulkPot
                    + czbssw * model->MOS3bulkJctSideGradingCoeff * sargsw/arg /
                        here->MOS3tBulkPot;
            here->MOS3f4s = czbs*here->MOS3tBulkPot*(1-arg*sarg)/
                        (1-model->MOS3bulkJctBotGradingCoeff)
                    + czbssw*here->MOS3tBulkPot*(1-arg*sargsw)/
                        (1-model->MOS3bulkJctSideGradingCoeff)
                    -here->MOS3f3s/2*
                      (here->MOS3tDepCap*here->MOS3tDepCap)
                    -here->MOS3tDepCap * here->MOS3f2s;
        }
    }
    return(OK);
}
