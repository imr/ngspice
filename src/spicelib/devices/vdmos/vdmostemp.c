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
            here->VDMOStBulkPot = fact2 * pbo+pbfact;
            gmanew = (here->VDMOStBulkPot-pbo)/pbo;
            capfact = (1+model->VDMOSbulkJctBotGradingCoeff*
                    (4e-4*(here->VDMOStemp-REFTEMP)-gmanew));
            here->VDMOStCbd *= capfact;
            here->VDMOStCbs *= capfact;
            here->VDMOStCj *= capfact;
            here->VDMOStDepCap = model->VDMOSfwdCapDepCoeff * here->VDMOStBulkPot;
            if (here->VDMOStSatCurDens == 0) {
                here->VDMOSsourceVcrit = here->VDMOSdrainVcrit =
                       vt*log(vt/(CONSTroot2*here->VDMOSm*here->VDMOStSatCur));
            }

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
        }
    }
    return(OK);
}
