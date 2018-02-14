/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: 2000 AlansFixes
**********/
/*
 */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "mos2defs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
MOS2acLoad(GENmodel *inModel, CKTcircuit *ckt)
{
    MOS2model *model = (MOS2model *)inModel;
    MOS2instance *here;
    int xnrm;
    int xrev;
    double xgs;
    double xgd;
    double xgb;
    double xbd;
    double xbs;
    double capgs;
    double capgd;
    double capgb;
    double GateBulkOverlapCap;
    double GateDrainOverlapCap;
    double GateSourceOverlapCap;
    double EffectiveLength;

    for( ; model != NULL; model = MOS2nextModel(model)) {
        for(here = MOS2instances(model); here!= NULL;
                here = MOS2nextInstance(here)) {

            if (here->MOS2mode < 0) {
                xnrm=0;
                xrev=1;
            } else {
                xnrm=1;
                xrev=0;
            }
            /*
             *     meyer's model parameters
             */
            EffectiveLength=here->MOS2l - 2*model->MOS2latDiff;
            GateSourceOverlapCap = model->MOS2gateSourceOverlapCapFactor * 
                    here->MOS2m * here->MOS2w;
            GateDrainOverlapCap = model->MOS2gateDrainOverlapCapFactor * 
                    here->MOS2m * here->MOS2w;
            GateBulkOverlapCap = model->MOS2gateBulkOverlapCapFactor * 
                    here->MOS2m * EffectiveLength;
            capgs = ( *(ckt->CKTstate0+here->MOS2capgs)+ 
                      *(ckt->CKTstate0+here->MOS2capgs) +
                      GateSourceOverlapCap );
            capgd = ( *(ckt->CKTstate0+here->MOS2capgd)+ 
                      *(ckt->CKTstate0+here->MOS2capgd) +
                      GateDrainOverlapCap );
            capgb = ( *(ckt->CKTstate0+here->MOS2capgb)+ 
                      *(ckt->CKTstate0+here->MOS2capgb) +
                      GateBulkOverlapCap );
            xgs = capgs * ckt->CKTomega;
            xgd = capgd * ckt->CKTomega;
            xgb = capgb * ckt->CKTomega;
            xbd  = here->MOS2capbd * ckt->CKTomega;
            xbs  = here->MOS2capbs * ckt->CKTomega;
            /*
             *    load matrix
             */

            *(here->MOS2GgPtr +1) += xgd+xgs+xgb;
            *(here->MOS2BbPtr +1) += xgb+xbd+xbs;
            *(here->MOS2DPdpPtr +1) += xgd+xbd;
            *(here->MOS2SPspPtr +1) += xgs+xbs;
            *(here->MOS2GbPtr +1) -= xgb;
            *(here->MOS2GdpPtr +1) -= xgd;
            *(here->MOS2GspPtr +1) -= xgs;
            *(here->MOS2BgPtr +1) -= xgb;
            *(here->MOS2BdpPtr +1) -= xbd;
            *(here->MOS2BspPtr +1) -= xbs;
            *(here->MOS2DPgPtr +1) -= xgd;
            *(here->MOS2DPbPtr +1) -= xbd;
            *(here->MOS2SPgPtr +1) -= xgs;
            *(here->MOS2SPbPtr +1) -= xbs;
            *(here->MOS2DdPtr) += here->MOS2drainConductance;
            *(here->MOS2SsPtr) += here->MOS2sourceConductance;
            *(here->MOS2BbPtr) += here->MOS2gbd+here->MOS2gbs;
            *(here->MOS2DPdpPtr) += here->MOS2drainConductance+
                    here->MOS2gds+here->MOS2gbd+
                    xrev*(here->MOS2gm+here->MOS2gmbs);
            *(here->MOS2SPspPtr) += here->MOS2sourceConductance+
                    here->MOS2gds+here->MOS2gbs+
                    xnrm*(here->MOS2gm+here->MOS2gmbs);
            *(here->MOS2DdpPtr) -= here->MOS2drainConductance;
            *(here->MOS2SspPtr) -= here->MOS2sourceConductance;
            *(here->MOS2BdpPtr) -= here->MOS2gbd;
            *(here->MOS2BspPtr) -= here->MOS2gbs;
            *(here->MOS2DPdPtr) -= here->MOS2drainConductance;
            *(here->MOS2DPgPtr) += (xnrm-xrev)*here->MOS2gm;
            *(here->MOS2DPbPtr) += -here->MOS2gbd+(xnrm-xrev)*here->MOS2gmbs;
            *(here->MOS2DPspPtr) -= here->MOS2gds+
                    xnrm*(here->MOS2gm+here->MOS2gmbs);
            *(here->MOS2SPgPtr) -= (xnrm-xrev)*here->MOS2gm;
            *(here->MOS2SPsPtr) -= here->MOS2sourceConductance;
            *(here->MOS2SPbPtr) -= here->MOS2gbs+(xnrm-xrev)*here->MOS2gmbs;
            *(here->MOS2SPdpPtr) -= here->MOS2gds+
                    xrev*(here->MOS2gm+here->MOS2gmbs);

        }
    }
    return(OK);
}
