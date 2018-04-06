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
        }
    }
    return(OK);
}
