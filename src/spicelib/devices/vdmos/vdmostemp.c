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
            SPfrontEnd->IFerrorf (ERR_FATAL,
               "%s: Phi is not positive.", model->VDMOSmodName);
            return(E_BADPARM);
        }

        if(!model->VDMOSoxideThicknessGiven || model->VDMOSoxideThickness == 0) {
            model->VDMOSoxideCapFactor = 0;
        } else {
            model->VDMOSoxideCapFactor = 3.9 * 8.854214871e-12/
                    model->VDMOSoxideThickness;
            if(!model->VDMOStransconductanceGiven) {
                if(!model->VDMOSsurfaceMobilityGiven) {
                    model->VDMOSsurfaceMobility=600;
                }
                model->VDMOStransconductance = model->VDMOSsurfaceMobility *
                        model->VDMOSoxideCapFactor * 1e-4 /*(m**2/cm**2)*/;
            }
            if(model->VDMOSsubstrateDopingGiven) {
                if(model->VDMOSsubstrateDoping*1e6 /*(cm**3/m**3)*/ >1.45e16) {
                    if(!model->VDMOSphiGiven) {
                        model->VDMOSphi = 2*vtnom*
                                log(model->VDMOSsubstrateDoping*
                                1e6/*(cm**3/m**3)*//1.45e16);
                        model->VDMOSphi = MAX(.1,model->VDMOSphi);
                    }
                    fermis = model->VDMOStype * .5 * model->VDMOSphi;
                    wkfng = 3.2;
                    if(!model->VDMOSgateTypeGiven) model->VDMOSgateType=1;
                    if(model->VDMOSgateType != 0) {
                        fermig = model->VDMOStype *model->VDMOSgateType*.5*egfet1;
                        wkfng = 3.25 + .5 * egfet1 - fermig;
                    }
                    wkfngs = wkfng - (3.25 + .5 * egfet1 +fermis);
                    if(!model->VDMOSgammaGiven) {
                        model->VDMOSgamma = sqrt(2 * 11.70 * 8.854214871e-12 * 
                                CHARGE * model->VDMOSsubstrateDoping*
                                1e6/*(cm**3/m**3)*/)/
                                model->VDMOSoxideCapFactor;
                    }
                    if(!model->VDMOSvt0Given) {
                        if(!model->VDMOSsurfaceStateDensityGiven) 
                                model->VDMOSsurfaceStateDensity=0;
                        vfb = wkfngs - 
                                model->VDMOSsurfaceStateDensity * 
                                1e4 /*(cm**2/m**2)*/ * 
                                CHARGE/model->VDMOSoxideCapFactor;
                        model->VDMOSvt0 = vfb + model->VDMOStype * 
                                (model->VDMOSgamma * sqrt(model->VDMOSphi)+
                                model->VDMOSphi);
                    }
                } else {
                    model->VDMOSsubstrateDoping = 0;
                    SPfrontEnd->IFerrorf (ERR_FATAL,
                            "%s: Nsub < Ni", model->VDMOSmodName);
                    return(E_BADPARM);
                }
            }
        }


        /* loop through all instances of the model */
        for(here = VDMOSinstances(model); here!= NULL; 
                here = VDMOSnextInstance(here)) {
            double czbd;    /* zero voltage bulk-drain capacitance */
            double czbdsw;  /* zero voltage bulk-drain sidewall capacitance */
            double czbs;    /* zero voltage bulk-source capacitance */
            double czbssw;  /* zero voltage bulk-source sidewall capacitance */
            double arg;     /* 1 - fc */
            double sarg;    /* (1-fc) ^^ (-mj) */
            double sargsw;  /* (1-fc) ^^ (-mjsw) */

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

            if(!here->VDMOSdrainAreaGiven) {
                here->VDMOSdrainArea = ckt->CKTdefaultMosAD;
            }
            if(!here->VDMOSmGiven) {
                here->VDMOSm = ckt->CKTdefaultMosM;
            }
            if(!here->VDMOSlGiven) {
                here->VDMOSl = ckt->CKTdefaultMosL;
            }
            if(!here->VDMOSsourceAreaGiven) {
                here->VDMOSsourceArea = ckt->CKTdefaultMosAS;
            }
            if(!here->VDMOSwGiven) {
                here->VDMOSw = ckt->CKTdefaultMosW;
            }

            if(here->VDMOSl - 2 * model->VDMOSlatDiff <=0) {
                SPfrontEnd->IFerrorf (ERR_WARNING,
                        "%s: effective channel length less than zero",
                        model->VDMOSmodName);
            }
            ratio4 = ratio * sqrt(ratio);
            here->VDMOStTransconductance = model->VDMOStransconductance / ratio4;
            here->VDMOStSurfMob = model->VDMOSsurfaceMobility/ratio4;
            phio= (model->VDMOSphi-pbfact1)/fact1;
            here->VDMOStPhi = fact2 * phio + pbfact;
            here->VDMOStVbi = 
                    model->VDMOSvt0 - model->VDMOStype * 
                        (model->VDMOSgamma* sqrt(model->VDMOSphi))
                    +.5*(egfet1-egfet) 
                    + model->VDMOStype*.5* (here->VDMOStPhi-model->VDMOSphi);
            here->VDMOStVto = here->VDMOStVbi + model->VDMOStype * 
                    model->VDMOSgamma * sqrt(here->VDMOStPhi);
            here->VDMOStSatCur = model->VDMOSjctSatCur* 
                    exp(-egfet/vt+egfet1/vtnom);
            here->VDMOStSatCurDens = model->VDMOSjctSatCurDensity *
                    exp(-egfet/vt+egfet1/vtnom);
            pbo = (model->VDMOSbulkJctPotential - pbfact1)/fact1;
            gmaold = (model->VDMOSbulkJctPotential-pbo)/pbo;
            capfact = 1/(1+model->VDMOSbulkJctBotGradingCoeff*
                    (4e-4*(model->VDMOStnom-REFTEMP)-gmaold));
            here->VDMOStCbd = model->VDMOScapBD * capfact;
            here->VDMOStCbs = model->VDMOScapBS * capfact;
            here->VDMOStCj = model->VDMOSbulkCapFactor * capfact;
            capfact = 1/(1+model->VDMOSbulkJctSideGradingCoeff*
                    (4e-4*(model->VDMOStnom-REFTEMP)-gmaold));
            here->VDMOStCjsw = model->VDMOSsideWallCapFactor * capfact;
            here->VDMOStBulkPot = fact2 * pbo+pbfact;
            gmanew = (here->VDMOStBulkPot-pbo)/pbo;
            capfact = (1+model->VDMOSbulkJctBotGradingCoeff*
                    (4e-4*(here->VDMOStemp-REFTEMP)-gmanew));
            here->VDMOStCbd *= capfact;
            here->VDMOStCbs *= capfact;
            here->VDMOStCj *= capfact;
            capfact = (1+model->VDMOSbulkJctSideGradingCoeff*
                    (4e-4*(here->VDMOStemp-REFTEMP)-gmanew));
            here->VDMOStCjsw *= capfact;
            here->VDMOStDepCap = model->VDMOSfwdCapDepCoeff * here->VDMOStBulkPot;
            if( (here->VDMOStSatCurDens == 0) ||
                    (here->VDMOSdrainArea == 0) ||
                    (here->VDMOSsourceArea == 0) ) {
                here->VDMOSsourceVcrit = here->VDMOSdrainVcrit =
                       vt*log(vt/(CONSTroot2*here->VDMOSm*here->VDMOStSatCur));
            } else {
                here->VDMOSdrainVcrit =
                        vt * log( vt / (CONSTroot2 *
                        here->VDMOSm *
                        here->VDMOStSatCurDens * here->VDMOSdrainArea));
                here->VDMOSsourceVcrit =
                        vt * log( vt / (CONSTroot2 *
                        here->VDMOSm *
                        here->VDMOStSatCurDens * here->VDMOSsourceArea));
            }

            if(model->VDMOScapBDGiven) {
                czbd = here->VDMOStCbd * here->VDMOSm;
            } else {
                if(model->VDMOSbulkCapFactorGiven) {
                    czbd=here->VDMOStCj*here->VDMOSm*here->VDMOSdrainArea;
                } else {
                    czbd=0;
                }
            }
            if(model->VDMOSsideWallCapFactorGiven) {
                czbdsw= here->VDMOStCjsw * here->VDMOSdrainPerimiter *
                     here->VDMOSm;
            } else {
                czbdsw=0;
            }
            arg = 1-model->VDMOSfwdCapDepCoeff;
            sarg = exp( (-model->VDMOSbulkJctBotGradingCoeff) * log(arg) );
            sargsw = exp( (-model->VDMOSbulkJctSideGradingCoeff) * log(arg) );
            here->VDMOSCbd = czbd;
            here->VDMOSCbdsw = czbdsw;
            here->VDMOSf2d = czbd*(1-model->VDMOSfwdCapDepCoeff*
                        (1+model->VDMOSbulkJctBotGradingCoeff))* sarg/arg
                    +  czbdsw*(1-model->VDMOSfwdCapDepCoeff*
                        (1+model->VDMOSbulkJctSideGradingCoeff))*
                        sargsw/arg;
            here->VDMOSf3d = czbd * model->VDMOSbulkJctBotGradingCoeff * sarg/arg/
                        here->VDMOStBulkPot
                    + czbdsw * model->VDMOSbulkJctSideGradingCoeff * sargsw/arg /
                        here->VDMOStBulkPot;
            here->VDMOSf4d = czbd*here->VDMOStBulkPot*(1-arg*sarg)/
                        (1-model->VDMOSbulkJctBotGradingCoeff)
                    + czbdsw*here->VDMOStBulkPot*(1-arg*sargsw)/
                        (1-model->VDMOSbulkJctSideGradingCoeff)
                    -here->VDMOSf3d/2*
                        (here->VDMOStDepCap*here->VDMOStDepCap)
                    -here->VDMOStDepCap * here->VDMOSf2d;
            if(model->VDMOScapBSGiven) {
                czbs=here->VDMOStCbs * here->VDMOSm;
            } else {
                if(model->VDMOSbulkCapFactorGiven) {
                   czbs=here->VDMOStCj*here->VDMOSsourceArea * here->VDMOSm;
                } else {
                    czbs=0;
                }
            }
            if(model->VDMOSsideWallCapFactorGiven) {
                czbssw = here->VDMOStCjsw * here->VDMOSsourcePerimiter *
                          here->VDMOSm;
            } else {
                czbssw=0;
            }
            arg = 1-model->VDMOSfwdCapDepCoeff;
            sarg = exp( (-model->VDMOSbulkJctBotGradingCoeff) * log(arg) );
            sargsw = exp( (-model->VDMOSbulkJctSideGradingCoeff) * log(arg) );
            here->VDMOSCbs = czbs;
            here->VDMOSCbssw = czbssw;
            here->VDMOSf2s = czbs*(1-model->VDMOSfwdCapDepCoeff*
                        (1+model->VDMOSbulkJctBotGradingCoeff))* sarg/arg
                    +  czbssw*(1-model->VDMOSfwdCapDepCoeff*
                        (1+model->VDMOSbulkJctSideGradingCoeff))*
                        sargsw/arg;
            here->VDMOSf3s = czbs * model->VDMOSbulkJctBotGradingCoeff * sarg/arg/
                        here->VDMOStBulkPot
                    + czbssw * model->VDMOSbulkJctSideGradingCoeff * sargsw/arg /
                        here->VDMOStBulkPot;
            here->VDMOSf4s = czbs*here->VDMOStBulkPot*(1-arg*sarg)/
                        (1-model->VDMOSbulkJctBotGradingCoeff)
                    + czbssw*here->VDMOStBulkPot*(1-arg*sargsw)/
                        (1-model->VDMOSbulkJctSideGradingCoeff)
                    -here->VDMOSf3s/2*
                        (here->VDMOStDepCap*here->VDMOStDepCap)
                    -here->VDMOStDepCap * here->VDMOSf2s;


            if (model->VDMOSdrainResistanceGiven) {
                if (model->VDMOSdrainResistance != 0) {
                    here->VDMOSdrainConductance = here->VDMOSm /
                        model->VDMOSdrainResistance;
                }
                else {
                    here->VDMOSdrainConductance = 0;
                }
            } else if (model->VDMOSsheetResistanceGiven) {
                if(model->VDMOSsheetResistance != 0) {
                    here->VDMOSdrainConductance =
                       here->VDMOSm /
                          (model->VDMOSsheetResistance*here->VDMOSdrainSquares);
                } else {
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
            } else if (model->VDMOSsheetResistanceGiven) {
                if ((model->VDMOSsheetResistance != 0) &&
                                   (here->VDMOSsourceSquares != 0)) {
                    here->VDMOSsourceConductance =
                        here->VDMOSm /
                          (model->VDMOSsheetResistance*here->VDMOSsourceSquares);
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
        }
    }
    return(OK);
}
