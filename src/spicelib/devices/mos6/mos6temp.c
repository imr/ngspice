/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1989 Takayasu Sakurai
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "mos6defs.h"
#include "ngspice/const.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

int
MOS6temp(GENmodel *inModel, CKTcircuit *ckt)
{
    MOS6model *model = (MOS6model *)inModel;
    MOS6instance *here;

    double egfet,egfet1;
    double fact1,fact2;
    double kt,kt1;
    double arg1;
    double ratio,ratio4;
    double phio;
    double pbo;
    double gmanew,gmaold;
    double capfact;
    double pbfact1,pbfact;
    double vt,vtnom;
    double wkfngs;
    double wkfng;
    double fermig;
    double fermis;
    double vfb;
    /* loop through all the resistor models */
    for( ; model != NULL; model = MOS6nextModel(model)) {
        

        /* perform model defaulting */
        if(!model->MOS6tnomGiven) {
            model->MOS6tnom = ckt->CKTnomTemp;
        }

        fact1 = model->MOS6tnom/REFTEMP;
        vtnom = model->MOS6tnom*CONSTKoverQ;
        kt1 = CONSTboltz * model->MOS6tnom;
        egfet1 = 1.16-(7.02e-4*model->MOS6tnom*model->MOS6tnom)/
                (model->MOS6tnom+1108);
        arg1 = -egfet1/(kt1+kt1)+1.1150877/(CONSTboltz*(REFTEMP+REFTEMP));
        pbfact1 = -2*vtnom *(1.5*log(fact1)+CHARGE*arg1);

    /* now model parameter preprocessing */

        if (model->MOS6phi <= 0.0) {
            SPfrontEnd->IFerrorf (ERR_FATAL,
               "%s: Phi is not positive.", model->MOS6modName);
            return(E_BADPARM);
        }

        if(!model->MOS6oxideThicknessGiven || model->MOS6oxideThickness == 0) {
            model->MOS6oxideCapFactor = 0;
        } else {
            model->MOS6oxideCapFactor = 3.9 * 8.854214871e-12/
                    model->MOS6oxideThickness;
            if(!model->MOS6kcGiven) {
                if(!model->MOS6surfaceMobilityGiven) {
                    model->MOS6surfaceMobility=600;
                }
                model->MOS6kc = 0.5 * model->MOS6surfaceMobility *
                        model->MOS6oxideCapFactor * 1e-4 /*(m**2/cm**2)*/;
            }
            if(model->MOS6substrateDopingGiven) {
                if(model->MOS6substrateDoping*1e6 /*(cm**3/m**3)*/ >1.45e16) {
                    if(!model->MOS6phiGiven) {
                        model->MOS6phi = 2*vtnom*
                                log(model->MOS6substrateDoping*
                                1e6/*(cm**3/m**3)*//1.45e16);
                        model->MOS6phi = MAX(.1,model->MOS6phi);
                    }
                    fermis = model->MOS6type * .5 * model->MOS6phi;
                    wkfng = 3.2;
                    if(!model->MOS6gateTypeGiven) model->MOS6gateType=1;
                    if(model->MOS6gateType != 0) {
                        fermig = model->MOS6type *model->MOS6gateType*.5*egfet1;
                        wkfng = 3.25 + .5 * egfet1 - fermig;
                    }
                    wkfngs = wkfng - (3.25 + .5 * egfet1 +fermis);
                    if(!model->MOS6gammaGiven) {
                        model->MOS6gamma = sqrt(2 * 11.70 * 8.854214871e-12 * 
                                CHARGE * model->MOS6substrateDoping*
                                1e6/*(cm**3/m**3)*/)/
                                model->MOS6oxideCapFactor;
                    }
                    if(!model->MOS6gamma1Given) {
                        model->MOS6gamma1 = 0.0;
                    }
                    if(!model->MOS6vt0Given) {
                        if(!model->MOS6surfaceStateDensityGiven) 
                                model->MOS6surfaceStateDensity=0;
                        vfb = wkfngs - 
                                model->MOS6surfaceStateDensity * 
                                1e4 /*(cm**2/m**2)*/ * 
                                CHARGE/model->MOS6oxideCapFactor;
                        model->MOS6vt0 = vfb + model->MOS6type * 
                                (model->MOS6gamma * sqrt(model->MOS6phi)+
                                model->MOS6phi);
                    }
                } else {
                    model->MOS6substrateDoping = 0;
                    SPfrontEnd->IFerrorf (ERR_FATAL,
                            "%s: Nsub < Ni", model->MOS6modName);
                    return(E_BADPARM);
                }
            }
        }

        
        /* loop through all instances of the model */
        for(here = MOS6instances(model); here!= NULL; 
                here = MOS6nextInstance(here)) {

            double czbd;    /* zero voltage bulk-drain capacitance */
            double czbdsw;  /* zero voltage bulk-drain sidewall capacitance */
            double czbs;    /* zero voltage bulk-source capacitance */
            double czbssw;  /* zero voltage bulk-source sidewall capacitance */
            double arg;     /* 1 - fc */
            double sarg;    /* (1-fc) ^^ (-mj) */
            double sargsw;  /* (1-fc) ^^ (-mjsw) */

            /* perform the parameter defaulting */
            if(!here->MOS6dtempGiven) {
                here->MOS6dtemp = 0.0;
            }
            if(!here->MOS6tempGiven) {
                here->MOS6temp = ckt->CKTtemp + here->MOS6dtemp;
            }
            vt = here->MOS6temp * CONSTKoverQ;
            ratio = here->MOS6temp/model->MOS6tnom;
            fact2 = here->MOS6temp/REFTEMP;
            kt = here->MOS6temp * CONSTboltz;
            egfet = 1.16-(7.02e-4*here->MOS6temp*here->MOS6temp)/
                    (here->MOS6temp+1108);
            arg = -egfet/(kt+kt)+1.1150877/(CONSTboltz*(REFTEMP+REFTEMP));
            pbfact = -2*vt *(1.5*log(fact2)+CHARGE*arg);

            if(!here->MOS6drainAreaGiven) {
                here->MOS6drainArea = ckt->CKTdefaultMosAD;
            }
            if(!here->MOS6lGiven) {
                here->MOS6l = ckt->CKTdefaultMosL;
            }
            if(!here->MOS6sourceAreaGiven) {
                here->MOS6sourceArea = ckt->CKTdefaultMosAS;
            }
            if(!here->MOS6wGiven) {
                here->MOS6w = ckt->CKTdefaultMosW;
            }

            if(here->MOS6l - 2 * model->MOS6latDiff <=0) {
                SPfrontEnd->IFerrorf (ERR_WARNING,
                        "%s: effective channel length less than zero",
                        model->MOS6modName);
            }
            ratio4 = ratio * sqrt(ratio);
            here->MOS6tKv = model->MOS6kv;
            here->MOS6tKc = model->MOS6kc / ratio4;
            here->MOS6tSurfMob = model->MOS6surfaceMobility/ratio4;
            phio= (model->MOS6phi-pbfact1)/fact1;
            here->MOS6tPhi = fact2 * phio + pbfact;
            here->MOS6tVbi = 
                    model->MOS6vt0 - model->MOS6type * 
                        (model->MOS6gamma* sqrt(model->MOS6phi))
                    +.5*(egfet1-egfet) 
                    + model->MOS6type*.5* (here->MOS6tPhi-model->MOS6phi);
            here->MOS6tVto = here->MOS6tVbi + model->MOS6type * 
                    model->MOS6gamma * sqrt(here->MOS6tPhi);
            here->MOS6tSatCur = model->MOS6jctSatCur* 
                    exp(-egfet/vt+egfet1/vtnom);
            here->MOS6tSatCurDens = model->MOS6jctSatCurDensity *
                    exp(-egfet/vt+egfet1/vtnom);
            pbo = (model->MOS6bulkJctPotential - pbfact1)/fact1;
            gmaold = (model->MOS6bulkJctPotential-pbo)/pbo;
            capfact = 1/(1+model->MOS6bulkJctBotGradingCoeff*
                    (4e-4*(model->MOS6tnom-REFTEMP)-gmaold));
            here->MOS6tCbd = model->MOS6capBD * capfact;
            here->MOS6tCbs = model->MOS6capBS * capfact;
            here->MOS6tCj = model->MOS6bulkCapFactor * capfact;
            capfact = 1/(1+model->MOS6bulkJctSideGradingCoeff*
                    (4e-4*(model->MOS6tnom-REFTEMP)-gmaold));
            here->MOS6tCjsw = model->MOS6sideWallCapFactor * capfact;
            here->MOS6tBulkPot = fact2 * pbo+pbfact;
            gmanew = (here->MOS6tBulkPot-pbo)/pbo;
            capfact = (1+model->MOS6bulkJctBotGradingCoeff*
                    (4e-4*(here->MOS6temp-REFTEMP)-gmanew));
            here->MOS6tCbd *= capfact;
            here->MOS6tCbs *= capfact;
            here->MOS6tCj *= capfact;
            capfact = (1+model->MOS6bulkJctSideGradingCoeff*
                    (4e-4*(here->MOS6temp-REFTEMP)-gmanew));
            here->MOS6tCjsw *= capfact;
            here->MOS6tDepCap = model->MOS6fwdCapDepCoeff * here->MOS6tBulkPot;
            if( (here->MOS6tSatCurDens == 0) ||
                    (here->MOS6drainArea == 0) ||
                    (here->MOS6sourceArea == 0) ) {
                here->MOS6sourceVcrit = here->MOS6drainVcrit =
                        vt*log(vt/(CONSTroot2*here->MOS6tSatCur));
            } else {
                here->MOS6drainVcrit =
                        vt * log( vt / (CONSTroot2 *
                        here->MOS6tSatCurDens * here->MOS6drainArea));
                here->MOS6sourceVcrit =
                        vt * log( vt / (CONSTroot2 *
                        here->MOS6tSatCurDens * here->MOS6sourceArea));
            }

            if(model->MOS6capBDGiven) {
                czbd = here->MOS6tCbd;
            } else {
                if(model->MOS6bulkCapFactorGiven) {
                    czbd=here->MOS6tCj*here->MOS6drainArea;
                } else {
                    czbd=0;
                }
            }
            if(model->MOS6sideWallCapFactorGiven) {
                czbdsw= here->MOS6tCjsw * here->MOS6drainPerimiter;
            } else {
                czbdsw=0;
            }
            arg = 1-model->MOS6fwdCapDepCoeff;
            sarg = exp( (-model->MOS6bulkJctBotGradingCoeff) * log(arg) );
            sargsw = exp( (-model->MOS6bulkJctSideGradingCoeff) * log(arg) );
            here->MOS6Cbd = czbd;
            here->MOS6Cbdsw = czbdsw;
            here->MOS6f2d = czbd*(1-model->MOS6fwdCapDepCoeff*
                        (1+model->MOS6bulkJctBotGradingCoeff))* sarg/arg
                    +  czbdsw*(1-model->MOS6fwdCapDepCoeff*
                        (1+model->MOS6bulkJctSideGradingCoeff))*
                        sargsw/arg;
            here->MOS6f3d = czbd * model->MOS6bulkJctBotGradingCoeff * sarg/arg/
                        here->MOS6tBulkPot
                    + czbdsw * model->MOS6bulkJctSideGradingCoeff * sargsw/arg /
                        here->MOS6tBulkPot;
            here->MOS6f4d = czbd*here->MOS6tBulkPot*(1-arg*sarg)/
                        (1-model->MOS6bulkJctBotGradingCoeff)
                    + czbdsw*here->MOS6tBulkPot*(1-arg*sargsw)/
                        (1-model->MOS6bulkJctSideGradingCoeff)
                    -here->MOS6f3d/2*
                        (here->MOS6tDepCap*here->MOS6tDepCap)
                    -here->MOS6tDepCap * here->MOS6f2d;
            if(model->MOS6capBSGiven) {
                czbs=here->MOS6tCbs;
            } else {
                if(model->MOS6bulkCapFactorGiven) {
                    czbs=here->MOS6tCj*here->MOS6sourceArea;
                } else {
                    czbs=0;
                }
            }
            if(model->MOS6sideWallCapFactorGiven) {
                czbssw = here->MOS6tCjsw * here->MOS6sourcePerimiter;
            } else {
                czbssw=0;
            }
            arg = 1-model->MOS6fwdCapDepCoeff;
            sarg = exp( (-model->MOS6bulkJctBotGradingCoeff) * log(arg) );
            sargsw = exp( (-model->MOS6bulkJctSideGradingCoeff) * log(arg) );
            here->MOS6Cbs = czbs;
            here->MOS6Cbssw = czbssw;
            here->MOS6f2s = czbs*(1-model->MOS6fwdCapDepCoeff*
                        (1+model->MOS6bulkJctBotGradingCoeff))* sarg/arg
                    +  czbssw*(1-model->MOS6fwdCapDepCoeff*
                        (1+model->MOS6bulkJctSideGradingCoeff))*
                        sargsw/arg;
            here->MOS6f3s = czbs * model->MOS6bulkJctBotGradingCoeff * sarg/arg/
                        here->MOS6tBulkPot
                    + czbssw * model->MOS6bulkJctSideGradingCoeff * sargsw/arg /
                        here->MOS6tBulkPot;
            here->MOS6f4s = czbs*here->MOS6tBulkPot*(1-arg*sarg)/
                        (1-model->MOS6bulkJctBotGradingCoeff)
                    + czbssw*here->MOS6tBulkPot*(1-arg*sargsw)/
                        (1-model->MOS6bulkJctSideGradingCoeff)
                    -here->MOS6f3s/2*
                        (here->MOS6tDepCap*here->MOS6tDepCap)
                    -here->MOS6tDepCap * here->MOS6f2s;


            if(model->MOS6drainResistanceGiven) {
                if(model->MOS6drainResistance != 0) {
                    here->MOS6drainConductance = 1/model->MOS6drainResistance;
                } else {
                    here->MOS6drainConductance = 0;
                }
            } else if (model->MOS6sheetResistanceGiven) {
                if( (!here->MOS6drainSquaresGiven) || 
                        ( here->MOS6drainSquares==0) ){
                    here->MOS6drainSquares=1;
                }
                if(model->MOS6sheetResistance != 0) {
                    here->MOS6drainConductance = 
                        1/(model->MOS6sheetResistance*here->MOS6drainSquares);
                } else {
                    here->MOS6drainConductance = 0;
                }
            } else {
                here->MOS6drainConductance = 0;
            }
            if(model->MOS6sourceResistanceGiven) {
                if(model->MOS6sourceResistance != 0) {
                    here->MOS6sourceConductance = 1/model->MOS6sourceResistance;
                } else {
                    here->MOS6sourceConductance = 0;
                }
            } else if (model->MOS6sheetResistanceGiven) {
                if( (!here->MOS6sourceSquaresGiven) || 
                        ( here->MOS6sourceSquares==0) ) {
                    here->MOS6sourceSquares=1;
                }
                if(model->MOS6sheetResistance != 0) {
                    here->MOS6sourceConductance = 
                        1/(model->MOS6sheetResistance*here->MOS6sourceSquares);
                } else {
                    here->MOS6sourceConductance = 0;
                }
            } else {
                here->MOS6sourceConductance = 0;
            }
        }
    }
    return(OK);
}
