/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: 2000 AlansFixes
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "mos1defs.h"
#include "ngspice/const.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

int
MOS1temp(GENmodel *inModel, CKTcircuit *ckt)
{
    MOS1model *model = (MOS1model *)inModel;
    MOS1instance *here;

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
    for( ; model != NULL; model = MOS1nextModel(model)) {
        

        /* perform model defaulting */
        if(!model->MOS1tnomGiven) {
            model->MOS1tnom = ckt->CKTnomTemp;
        }

        fact1 = model->MOS1tnom/REFTEMP;
        vtnom = model->MOS1tnom*CONSTKoverQ;
        kt1 = CONSTboltz * model->MOS1tnom;
        egfet1 = 1.16-(7.02e-4*model->MOS1tnom*model->MOS1tnom)/
                (model->MOS1tnom+1108);
        arg1 = -egfet1/(kt1+kt1)+1.1150877/(CONSTboltz*(REFTEMP+REFTEMP));
        pbfact1 = -2*vtnom *(1.5*log(fact1)+CHARGE*arg1);

    /* now model parameter preprocessing */

        if (model->MOS1phi <= 0.0) {
            SPfrontEnd->IFerrorf (ERR_FATAL,
               "%s: Phi is not positive.", model->MOS1modName);
            return(E_BADPARM);
        }

        if(!model->MOS1oxideThicknessGiven || model->MOS1oxideThickness == 0) {
            model->MOS1oxideCapFactor = 0;
        } else {
            model->MOS1oxideCapFactor = 3.9 * 8.854214871e-12/
                    model->MOS1oxideThickness;
            if(!model->MOS1transconductanceGiven) {
                if(!model->MOS1surfaceMobilityGiven) {
                    model->MOS1surfaceMobility=600;
                }
                model->MOS1transconductance = model->MOS1surfaceMobility *
                        model->MOS1oxideCapFactor * 1e-4 /*(m**2/cm**2)*/;
            }
            if(model->MOS1substrateDopingGiven) {
                if(model->MOS1substrateDoping*1e6 /*(cm**3/m**3)*/ >1.45e16) {
                    if(!model->MOS1phiGiven) {
                        model->MOS1phi = 2*vtnom*
                                log(model->MOS1substrateDoping*
                                1e6/*(cm**3/m**3)*//1.45e16);
                        model->MOS1phi = MAX(.1,model->MOS1phi);
                    }
                    fermis = model->MOS1type * .5 * model->MOS1phi;
                    wkfng = 3.2;
                    if(!model->MOS1gateTypeGiven) model->MOS1gateType=1;
                    if(model->MOS1gateType != 0) {
                        fermig = model->MOS1type *model->MOS1gateType*.5*egfet1;
                        wkfng = 3.25 + .5 * egfet1 - fermig;
                    }
                    wkfngs = wkfng - (3.25 + .5 * egfet1 +fermis);
                    if(!model->MOS1gammaGiven) {
                        model->MOS1gamma = sqrt(2 * 11.70 * 8.854214871e-12 * 
                                CHARGE * model->MOS1substrateDoping*
                                1e6/*(cm**3/m**3)*/)/
                                model->MOS1oxideCapFactor;
                    }
                    if(!model->MOS1vt0Given) {
                        if(!model->MOS1surfaceStateDensityGiven) 
                                model->MOS1surfaceStateDensity=0;
                        vfb = wkfngs - 
                                model->MOS1surfaceStateDensity * 
                                1e4 /*(cm**2/m**2)*/ * 
                                CHARGE/model->MOS1oxideCapFactor;
                        model->MOS1vt0 = vfb + model->MOS1type * 
                                (model->MOS1gamma * sqrt(model->MOS1phi)+
                                model->MOS1phi);
                    }
                } else {
                    model->MOS1substrateDoping = 0;
                    SPfrontEnd->IFerrorf (ERR_FATAL,
                            "%s: Nsub < Ni", model->MOS1modName);
                    return(E_BADPARM);
                }
            }
        }

        
        /* loop through all instances of the model */
        for(here = MOS1instances(model); here!= NULL; 
                here = MOS1nextInstance(here)) {
            double czbd;    /* zero voltage bulk-drain capacitance */
            double czbdsw;  /* zero voltage bulk-drain sidewall capacitance */
            double czbs;    /* zero voltage bulk-source capacitance */
            double czbssw;  /* zero voltage bulk-source sidewall capacitance */
            double arg;     /* 1 - fc */
            double sarg;    /* (1-fc) ^^ (-mj) */
            double sargsw;  /* (1-fc) ^^ (-mjsw) */

            /* perform the parameter defaulting */
            
            if(!here->MOS1dtempGiven) {
                here->MOS1dtemp = 0.0;
            }
            if(!here->MOS1tempGiven) {
                here->MOS1temp = ckt->CKTtemp + here->MOS1dtemp;
            }
            vt = here->MOS1temp * CONSTKoverQ;
            ratio = here->MOS1temp/model->MOS1tnom;
            fact2 = here->MOS1temp/REFTEMP;
            kt = here->MOS1temp * CONSTboltz;
            egfet = 1.16-(7.02e-4*here->MOS1temp*here->MOS1temp)/
                    (here->MOS1temp+1108);
            arg = -egfet/(kt+kt)+1.1150877/(CONSTboltz*(REFTEMP+REFTEMP));
            pbfact = -2*vt *(1.5*log(fact2)+CHARGE*arg);

            if(!here->MOS1drainAreaGiven) {
                here->MOS1drainArea = ckt->CKTdefaultMosAD;
            }
            if(!here->MOS1mGiven) {
                here->MOS1m = ckt->CKTdefaultMosM;
            }
            if(!here->MOS1lGiven) {
                here->MOS1l = ckt->CKTdefaultMosL;
            }
            if(!here->MOS1sourceAreaGiven) {
                here->MOS1sourceArea = ckt->CKTdefaultMosAS;
            }
            if(!here->MOS1wGiven) {
                here->MOS1w = ckt->CKTdefaultMosW;
            }

            if(here->MOS1l - 2 * model->MOS1latDiff <=0) {
                SPfrontEnd->IFerrorf (ERR_WARNING,
                        "%s: effective channel length less than zero",
                        model->MOS1modName);
            }
            ratio4 = ratio * sqrt(ratio);
            here->MOS1tTransconductance = model->MOS1transconductance / ratio4;
            here->MOS1tSurfMob = model->MOS1surfaceMobility/ratio4;
            phio= (model->MOS1phi-pbfact1)/fact1;
            here->MOS1tPhi = fact2 * phio + pbfact;
            here->MOS1tVbi = 
                    model->MOS1vt0 - model->MOS1type * 
                        (model->MOS1gamma* sqrt(model->MOS1phi))
                    +.5*(egfet1-egfet) 
                    + model->MOS1type*.5* (here->MOS1tPhi-model->MOS1phi);
            here->MOS1tVto = here->MOS1tVbi + model->MOS1type * 
                    model->MOS1gamma * sqrt(here->MOS1tPhi);
            here->MOS1tSatCur = model->MOS1jctSatCur* 
                    exp(-egfet/vt+egfet1/vtnom);
            here->MOS1tSatCurDens = model->MOS1jctSatCurDensity *
                    exp(-egfet/vt+egfet1/vtnom);
            pbo = (model->MOS1bulkJctPotential - pbfact1)/fact1;
            gmaold = (model->MOS1bulkJctPotential-pbo)/pbo;
            capfact = 1/(1+model->MOS1bulkJctBotGradingCoeff*
                    (4e-4*(model->MOS1tnom-REFTEMP)-gmaold));
            here->MOS1tCbd = model->MOS1capBD * capfact;
            here->MOS1tCbs = model->MOS1capBS * capfact;
            here->MOS1tCj = model->MOS1bulkCapFactor * capfact;
            capfact = 1/(1+model->MOS1bulkJctSideGradingCoeff*
                    (4e-4*(model->MOS1tnom-REFTEMP)-gmaold));
            here->MOS1tCjsw = model->MOS1sideWallCapFactor * capfact;
            here->MOS1tBulkPot = fact2 * pbo+pbfact;
            gmanew = (here->MOS1tBulkPot-pbo)/pbo;
            capfact = (1+model->MOS1bulkJctBotGradingCoeff*
                    (4e-4*(here->MOS1temp-REFTEMP)-gmanew));
            here->MOS1tCbd *= capfact;
            here->MOS1tCbs *= capfact;
            here->MOS1tCj *= capfact;
            capfact = (1+model->MOS1bulkJctSideGradingCoeff*
                    (4e-4*(here->MOS1temp-REFTEMP)-gmanew));
            here->MOS1tCjsw *= capfact;
            here->MOS1tDepCap = model->MOS1fwdCapDepCoeff * here->MOS1tBulkPot;
            if( (here->MOS1tSatCurDens == 0) ||
                    (here->MOS1drainArea == 0) ||
                    (here->MOS1sourceArea == 0) ) {
                here->MOS1sourceVcrit = here->MOS1drainVcrit =
                       vt*log(vt/(CONSTroot2*here->MOS1m*here->MOS1tSatCur));
            } else {
                here->MOS1drainVcrit =
                        vt * log( vt / (CONSTroot2 *
                        here->MOS1m *
                        here->MOS1tSatCurDens * here->MOS1drainArea));
                here->MOS1sourceVcrit =
                        vt * log( vt / (CONSTroot2 *
                        here->MOS1m *
                        here->MOS1tSatCurDens * here->MOS1sourceArea));
            }

            if(model->MOS1capBDGiven) {
                czbd = here->MOS1tCbd * here->MOS1m;
            } else {
                if(model->MOS1bulkCapFactorGiven) {  
                    czbd=here->MOS1tCj*here->MOS1m*here->MOS1drainArea;
                } else {
                    czbd=0;
                }
            }
            if(model->MOS1sideWallCapFactorGiven) {
                czbdsw= here->MOS1tCjsw * here->MOS1drainPerimiter *
                     here->MOS1m;
            } else {
                czbdsw=0;
            }
            arg = 1-model->MOS1fwdCapDepCoeff;
            sarg = exp( (-model->MOS1bulkJctBotGradingCoeff) * log(arg) );
            sargsw = exp( (-model->MOS1bulkJctSideGradingCoeff) * log(arg) );
            here->MOS1Cbd = czbd;
            here->MOS1Cbdsw = czbdsw;
            here->MOS1f2d = czbd*(1-model->MOS1fwdCapDepCoeff*
                        (1+model->MOS1bulkJctBotGradingCoeff))* sarg/arg
                    +  czbdsw*(1-model->MOS1fwdCapDepCoeff*
                        (1+model->MOS1bulkJctSideGradingCoeff))*
                        sargsw/arg;
            here->MOS1f3d = czbd * model->MOS1bulkJctBotGradingCoeff * sarg/arg/
                        here->MOS1tBulkPot
                    + czbdsw * model->MOS1bulkJctSideGradingCoeff * sargsw/arg /
                        here->MOS1tBulkPot;
            here->MOS1f4d = czbd*here->MOS1tBulkPot*(1-arg*sarg)/
                        (1-model->MOS1bulkJctBotGradingCoeff)
                    + czbdsw*here->MOS1tBulkPot*(1-arg*sargsw)/
                        (1-model->MOS1bulkJctSideGradingCoeff)
                    -here->MOS1f3d/2*
                        (here->MOS1tDepCap*here->MOS1tDepCap)
                    -here->MOS1tDepCap * here->MOS1f2d;
            if(model->MOS1capBSGiven) {
                czbs=here->MOS1tCbs * here->MOS1m;
            } else {
                if(model->MOS1bulkCapFactorGiven) {
                   czbs=here->MOS1tCj*here->MOS1sourceArea * here->MOS1m;
                } else {
                    czbs=0;
                }
            }
            if(model->MOS1sideWallCapFactorGiven) {
                czbssw = here->MOS1tCjsw * here->MOS1sourcePerimiter *
                          here->MOS1m;
            } else {
                czbssw=0;
            }
            arg = 1-model->MOS1fwdCapDepCoeff;
            sarg = exp( (-model->MOS1bulkJctBotGradingCoeff) * log(arg) );
            sargsw = exp( (-model->MOS1bulkJctSideGradingCoeff) * log(arg) );
            here->MOS1Cbs = czbs;
            here->MOS1Cbssw = czbssw;
            here->MOS1f2s = czbs*(1-model->MOS1fwdCapDepCoeff*
                        (1+model->MOS1bulkJctBotGradingCoeff))* sarg/arg
                    +  czbssw*(1-model->MOS1fwdCapDepCoeff*
                        (1+model->MOS1bulkJctSideGradingCoeff))*
                        sargsw/arg;
            here->MOS1f3s = czbs * model->MOS1bulkJctBotGradingCoeff * sarg/arg/
                        here->MOS1tBulkPot
                    + czbssw * model->MOS1bulkJctSideGradingCoeff * sargsw/arg /
                        here->MOS1tBulkPot;
            here->MOS1f4s = czbs*here->MOS1tBulkPot*(1-arg*sarg)/
                        (1-model->MOS1bulkJctBotGradingCoeff)
                    + czbssw*here->MOS1tBulkPot*(1-arg*sargsw)/
                        (1-model->MOS1bulkJctSideGradingCoeff)
                    -here->MOS1f3s/2*
                        (here->MOS1tDepCap*here->MOS1tDepCap)
                    -here->MOS1tDepCap * here->MOS1f2s;


            if(model->MOS1drainResistanceGiven) {
                if(model->MOS1drainResistance != 0) {
                   here->MOS1drainConductance = here->MOS1m /
                                      model->MOS1drainResistance;
                } else {
                    here->MOS1drainConductance = 0;
                }
            } else if (model->MOS1sheetResistanceGiven) {
                if(model->MOS1sheetResistance != 0) {
                    here->MOS1drainConductance = 
                       here->MOS1m /
                          (model->MOS1sheetResistance*here->MOS1drainSquares);
                } else {
                    here->MOS1drainConductance = 0;
                }
            } else {
                here->MOS1drainConductance = 0;
            }
            if(model->MOS1sourceResistanceGiven) {
                if(model->MOS1sourceResistance != 0) {
                   here->MOS1sourceConductance = here->MOS1m /
                                         model->MOS1sourceResistance;
                } else {
                    here->MOS1sourceConductance = 0;
                }
            } else if (model->MOS1sheetResistanceGiven) {
                if ((model->MOS1sheetResistance != 0) &&
                                   (here->MOS1sourceSquares != 0)) {
                    here->MOS1sourceConductance = 
                        here->MOS1m /
                          (model->MOS1sheetResistance*here->MOS1sourceSquares);
                } else {
                    here->MOS1sourceConductance = 0;
                }
            } else {
                here->MOS1sourceConductance = 0;
            }
        }
    }
    return(OK);
}
