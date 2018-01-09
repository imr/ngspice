/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: 2000 AlansFixes
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "mos2defs.h"
#include "ngspice/const.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

/* assuming silicon - make definition for epsilon of silicon */
#define EPSSIL (11.7 * 8.854214871e-12)

int
MOS2temp(GENmodel *inModel, CKTcircuit *ckt)
{
    MOS2model *model = (MOS2model *)inModel;
    MOS2instance *here;
    double egfet;
    double wkfngs;
    double wkfng;
    double fermig;
    double fermis;
    double vfb;
    double fact1,fact2;
    double vt,vtnom;
    double kt,kt1;
    double egfet1;
    double arg1;
    double pbfact,pbfact1;
    double ratio,ratio4;
    double phio;
    double pbo;
    double gmaold,gmanew;
    double capfact;
    /* loop through all the resistor models */
    for( ; model != NULL; model = MOS2nextModel(model)) {
        
        /* perform model defaulting */

    /* now model parameter preprocessing */
        if(!model->MOS2tnomGiven) {
            model->MOS2tnom = ckt->CKTnomTemp;
        }
        fact1 = model->MOS2tnom/REFTEMP;
        vtnom = model->MOS2tnom*CONSTKoverQ;
        kt1 = CONSTboltz * model->MOS2tnom;
        egfet1 = 1.16-(7.02e-4*model->MOS2tnom*model->MOS2tnom)/
                (model->MOS2tnom+1108);
        arg1 = -egfet1/(kt1+kt1)+1.1150877/(CONSTboltz*(REFTEMP+REFTEMP));
        pbfact1 = -2*vtnom *(1.5*log(fact1)+CHARGE*arg1);

        if (model->MOS2phi <= 0.0) {
            SPfrontEnd->IFerrorf (ERR_FATAL,
               "%s: Phi is not positive.", model->MOS2modName);
            return(E_BADPARM);
        }

        if(!model->MOS2oxideThicknessGiven) {
            model->MOS2oxideThickness = 1e-7;
        } 
        model->MOS2oxideCapFactor = 3.9 * 8.854214871e-12/
                model->MOS2oxideThickness;

        if(!model->MOS2surfaceMobilityGiven) model->MOS2surfaceMobility=600;
        if(!model->MOS2transconductanceGiven) {
            model->MOS2transconductance = model->MOS2surfaceMobility  
                * 1e-4 /*(m**2/cm**2) */ * model->MOS2oxideCapFactor;
        }
        if(model->MOS2substrateDopingGiven) {
            if(model->MOS2substrateDoping *1e6 /*(cm**3/m**3)*/ >1.45e16) {
                if(!model->MOS2phiGiven) {
                    model->MOS2phi = 2*vtnom*
                            log(model->MOS2substrateDoping*
                            1e6 /*(cm**3/m**3)*//1.45e16);
                    model->MOS2phi = MAX(.1,model->MOS2phi);
                }
                fermis = model->MOS2type * .5 * model->MOS2phi;
                wkfng = 3.2;
                if(!model->MOS2gateTypeGiven) model->MOS2gateType=1;
                if(model->MOS2gateType != 0) {
                    fermig = model->MOS2type * model->MOS2gateType*.5*egfet1;
                    wkfng = 3.25 + .5 * egfet1 - fermig;
                }
                wkfngs = wkfng - (3.25 + .5 * egfet1 +fermis);
                if(!model->MOS2gammaGiven) {
                    model->MOS2gamma = sqrt(2 * 11.70 * 8.854214871e-12 * 
                        CHARGE * model->MOS2substrateDoping *
                        1e6 /*(cm**3/m**3)*/)/model->MOS2oxideCapFactor;
                }
                if(!model->MOS2vt0Given) {
                    if(!model->MOS2surfaceStateDensityGiven) 
                            model->MOS2surfaceStateDensity=0;
                    vfb = wkfngs - 
                        model->MOS2surfaceStateDensity * 
                        1e4 /*(cm**2/m**2)*/ * CHARGE/model->MOS2oxideCapFactor;
                    model->MOS2vt0 = vfb + model->MOS2type * 
                            (model->MOS2gamma * sqrt(model->MOS2phi)+
                            model->MOS2phi);
                } else {
                    vfb = model->MOS2vt0 - model->MOS2type * (model->MOS2gamma*
                        sqrt(model->MOS2phi)+model->MOS2phi);
                }
                model->MOS2xd = sqrt((EPSSIL+EPSSIL)/
                    (CHARGE*model->MOS2substrateDoping *1e6 /*(cm**3/m**3)*/));
            } else {
                model->MOS2substrateDoping = 0;
                SPfrontEnd->IFerrorf (ERR_FATAL, "%s: Nsub < Ni",
                        model->MOS2modName);
                return(E_BADPARM);
            }
        }
        if(!model->MOS2bulkCapFactorGiven) {
            model->MOS2bulkCapFactor = sqrt(EPSSIL*CHARGE*
                model->MOS2substrateDoping* 1e6 /*cm**3/m**3*/
                /(2*model->MOS2bulkJctPotential));
        }

        
        /* loop through all instances of the model */
        for(here = MOS2instances(model); here!= NULL; 
                here = MOS2nextInstance(here)) {
            double czbd;    /* zero voltage bulk-drain capacitance */
            double czbdsw;  /* zero voltage bulk-drain sidewall capacitance */
            double czbs;    /* zero voltage bulk-source capacitance */
            double czbssw;  /* zero voltage bulk-source sidewall capacitance */
            double arg;     /* 1 - fc */
            double sarg;    /* (1-fc) ^^ (-mj) */
            double sargsw;  /* (1-fc) ^^ (-mjsw) */

            /* perform the parameter defaulting */
            if(!here->MOS2dtempGiven) {
                here->MOS2dtemp = 0.0;
            }

            if(!here->MOS2tempGiven) {
                here->MOS2temp = ckt->CKTtemp + here->MOS2dtemp;
            }
            here->MOS2mode = 1;
            here->MOS2von = 0;

            vt = here->MOS2temp * CONSTKoverQ;
            ratio = here->MOS2temp/model->MOS2tnom;
            fact2 = here->MOS2temp/REFTEMP;
            kt = here->MOS2temp * CONSTboltz;
            egfet = 1.16-(7.02e-4*here->MOS2temp*here->MOS2temp)/
                    (here->MOS2temp+1108);
            arg = -egfet/(kt+kt)+1.1150877/(CONSTboltz*(REFTEMP+REFTEMP));
            pbfact = -2*vt *(1.5*log(fact2)+CHARGE*arg);

            if(!here->MOS2drainAreaGiven) {
                here->MOS2drainArea = ckt->CKTdefaultMosAD;
            } 
            if(!here->MOS2mGiven) {
                here->MOS2m = ckt->CKTdefaultMosM;
            }
            if(!here->MOS2lGiven) {
                here->MOS2l = ckt->CKTdefaultMosL;
            }
            if(!here->MOS2sourceAreaGiven) {
                here->MOS2sourceArea = ckt->CKTdefaultMosAS;
            }
            if(!here->MOS2wGiven) {
                here->MOS2w = ckt->CKTdefaultMosW;
            }
            if(model->MOS2drainResistanceGiven) {
                if(model->MOS2drainResistance != 0) {
                   here->MOS2drainConductance = here->MOS2m /
                           model->MOS2drainResistance;
                } else {
                    here->MOS2drainConductance = 0;
                }
            } else if (model->MOS2sheetResistanceGiven) {
                if((model->MOS2sheetResistance != 0) &&
                                 (here->MOS2drainSquares != 0)) {
                    here->MOS2drainConductance = 
                        here->MOS2m /
                          (model->MOS2sheetResistance*here->MOS2drainSquares);
                } else {
                    here->MOS2drainConductance = 0;
                }
            } else {
                here->MOS2drainConductance = 0;
            }
            if(model->MOS2sourceResistanceGiven) {
                if(model->MOS2sourceResistance != 0) {
                   here->MOS2sourceConductance = here->MOS2m /
                                    model->MOS2sourceResistance;
                } else {
                    here->MOS2sourceConductance = 0;
                }
            } else if (model->MOS2sheetResistanceGiven) {
                if ((model->MOS2sheetResistance != 0) &&
                                  (here->MOS2sourceSquares != 0)) {
                    here->MOS2sourceConductance = 
                       here->MOS2m /
                          (model->MOS2sheetResistance*here->MOS2sourceSquares);
                } else {
                    here->MOS2sourceConductance = 0;
                }
            } else {
                here->MOS2sourceConductance = 0;
            }
            if(here->MOS2l - 2 * model->MOS2latDiff <=0) {
                SPfrontEnd->IFerrorf (ERR_WARNING,
                        "%s: effective channel length less than zero",
                        here->MOS2name);
            }

            ratio4 = ratio * sqrt(ratio);
            here->MOS2tTransconductance = model->MOS2transconductance / ratio4;
            here->MOS2tSurfMob = model->MOS2surfaceMobility/ratio4;
            phio= (model->MOS2phi-pbfact1)/fact1;
            here->MOS2tPhi = fact2 * phio + pbfact;
            here->MOS2tVbi = 
                    model->MOS2vt0 - model->MOS2type * 
                        (model->MOS2gamma* sqrt(model->MOS2phi))
                    +.5*(egfet1-egfet) 
                    + model->MOS2type*.5* (here->MOS2tPhi-model->MOS2phi);
            here->MOS2tVto = here->MOS2tVbi + model->MOS2type * 
                    model->MOS2gamma * sqrt(here->MOS2tPhi);
            here->MOS2tSatCur = model->MOS2jctSatCur* 
                    exp(-egfet/vt+egfet1/vtnom);
            here->MOS2tSatCurDens = model->MOS2jctSatCurDensity *
                    exp(-egfet/vt+egfet1/vtnom);
            pbo = (model->MOS2bulkJctPotential - pbfact1)/fact1;
            gmaold = (model->MOS2bulkJctPotential-pbo)/pbo;
            capfact = 1/(1+model->MOS2bulkJctBotGradingCoeff*
                    (4e-4*(model->MOS2tnom-REFTEMP)-gmaold));
            here->MOS2tCbd = model->MOS2capBD * capfact;
            here->MOS2tCbs = model->MOS2capBS * capfact;
            here->MOS2tCj = model->MOS2bulkCapFactor * capfact;
            capfact = 1/(1+model->MOS2bulkJctSideGradingCoeff*
                    (4e-4*(model->MOS2tnom-REFTEMP)-gmaold));
            here->MOS2tCjsw = model->MOS2sideWallCapFactor * capfact;
            here->MOS2tBulkPot = fact2 * pbo+pbfact;
            gmanew = (here->MOS2tBulkPot-pbo)/pbo;
            capfact = (1+model->MOS2bulkJctBotGradingCoeff*
                    (4e-4*(here->MOS2temp-REFTEMP)-gmanew));
            here->MOS2tCbd *= capfact;
            here->MOS2tCbs *= capfact;
            here->MOS2tCj *= capfact;
            capfact = (1+model->MOS2bulkJctSideGradingCoeff*
                    (4e-4*(here->MOS2temp-REFTEMP)-gmanew));
            here->MOS2tCjsw *= capfact;
            here->MOS2tDepCap = model->MOS2fwdCapDepCoeff * here->MOS2tBulkPot;


            if( (here->MOS2tSatCurDens == 0) ||
                    (here->MOS2drainArea == 0) ||
                    (here->MOS2sourceArea == 0) ) {
                here->MOS2sourceVcrit = here->MOS2drainVcrit =
                        vt*log(vt/(CONSTroot2*here->MOS2m*here->MOS2tSatCur));
            } else {
                here->MOS2drainVcrit =
                        vt * log( vt / (CONSTroot2 *
                        here->MOS2m *
                        here->MOS2tSatCurDens * here->MOS2drainArea));
                here->MOS2sourceVcrit =
                        vt * log( vt / (CONSTroot2 *
                        here->MOS2m *
                        here->MOS2tSatCurDens * here->MOS2sourceArea));
            }
            if(model->MOS2capBDGiven) {
                czbd = here->MOS2tCbd * here->MOS2m;
            } else {
                if(model->MOS2bulkCapFactorGiven) {
                    czbd=here->MOS2tCj*here->MOS2drainArea * here->MOS2m;
                } else {
                    czbd=0;
                }
            }
            if(model->MOS2sideWallCapFactorGiven) {
                czbdsw= here->MOS2tCjsw * here->MOS2drainPerimiter *
                     here->MOS2m;;
            } else {
                czbdsw=0;
            }
            arg = 1-model->MOS2fwdCapDepCoeff;
            sarg = exp( (-model->MOS2bulkJctBotGradingCoeff) * log(arg) );
            sargsw = exp( (-model->MOS2bulkJctSideGradingCoeff) * log(arg) );
            here->MOS2Cbd = czbd;
            here->MOS2Cbdsw = czbdsw;
            here->MOS2f2d = czbd*(1-model->MOS2fwdCapDepCoeff*
                        (1+model->MOS2bulkJctBotGradingCoeff))* sarg/arg
                    +  czbdsw*(1-model->MOS2fwdCapDepCoeff*
                        (1+model->MOS2bulkJctSideGradingCoeff))*
                        sargsw/arg;
            here->MOS2f3d = czbd * model->MOS2bulkJctBotGradingCoeff * sarg/arg/
                        here->MOS2tBulkPot
                    + czbdsw * model->MOS2bulkJctSideGradingCoeff * sargsw/arg /
                        here->MOS2tBulkPot;
            here->MOS2f4d = czbd*here->MOS2tBulkPot*(1-arg*sarg)/
                        (1-model->MOS2bulkJctBotGradingCoeff)
                    + czbdsw*here->MOS2tBulkPot*(1-arg*sargsw)/
                        (1-model->MOS2bulkJctSideGradingCoeff)
                    -here->MOS2f3d/2*
                        (here->MOS2tDepCap*here->MOS2tDepCap)
                    -here->MOS2tDepCap * here->MOS2f2d;
            if(model->MOS2capBSGiven) {
                czbs=here->MOS2tCbs * here->MOS2m;
            } else {
                if(model->MOS2bulkCapFactorGiven) {
                    czbs=here->MOS2tCj*here->MOS2sourceArea * here->MOS2m;
                } else {
                    czbs=0;
                }
            }
            if(model->MOS2sideWallCapFactorGiven) {
                czbssw = here->MOS2tCjsw * here->MOS2sourcePerimiter *
                     here->MOS2m;
            } else {
                czbssw=0;
            }
            arg = 1-model->MOS2fwdCapDepCoeff;
            sarg = exp( (-model->MOS2bulkJctBotGradingCoeff) * log(arg) );
            sargsw = exp( (-model->MOS2bulkJctSideGradingCoeff) * log(arg) );
            here->MOS2Cbs = czbs;
            here->MOS2Cbssw = czbssw;
            here->MOS2f2s = czbs*(1-model->MOS2fwdCapDepCoeff*
                        (1+model->MOS2bulkJctBotGradingCoeff))* sarg/arg
                    +  czbssw*(1-model->MOS2fwdCapDepCoeff*
                        (1+model->MOS2bulkJctSideGradingCoeff))*
                        sargsw/arg;
            here->MOS2f3s = czbs * model->MOS2bulkJctBotGradingCoeff * sarg/arg/
                        here->MOS2tBulkPot
                    + czbssw * model->MOS2bulkJctSideGradingCoeff * sargsw/arg /
                        here->MOS2tBulkPot;
            here->MOS2f4s = czbs*here->MOS2tBulkPot*(1-arg*sarg)/
                        (1-model->MOS2bulkJctBotGradingCoeff)
                    + czbssw*here->MOS2tBulkPot*(1-arg*sargsw)/
                        (1-model->MOS2bulkJctSideGradingCoeff)
                    -here->MOS2f3s/2*
                        (here->MOS2tDepCap*here->MOS2tDepCap)
                    -here->MOS2tDepCap * here->MOS2f2s;

        }
    }
    return(OK);
}
