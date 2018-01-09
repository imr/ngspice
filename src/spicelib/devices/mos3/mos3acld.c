/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: 2000 AlansFixes
**********/
/*
 */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "mos3defs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
MOS3acLoad(GENmodel *inModel, CKTcircuit *ckt)
{
    MOS3model *model = (MOS3model *)inModel;
    MOS3instance *here;
    int xnrm;
    int xrev;
    double EffectiveLength;
    double EffectiveWidth;
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

    for( ; model != NULL; model = MOS3nextModel(model)) {
        for(here = MOS3instances(model); here!= NULL;
                here = MOS3nextInstance(here)) {
        
            if (here->MOS3mode < 0) {
                xnrm=0;
                xrev=1;
            } else {
                xnrm=1;
                xrev=0;
            }
            /*
             *    charge oriented model parameters
             */
            EffectiveWidth=here->MOS3w-2*model->MOS3widthNarrow+
                                    model->MOS3widthAdjust;
            EffectiveLength=here->MOS3l - 2*model->MOS3latDiff+
                                    model->MOS3lengthAdjust;
                                    
            GateSourceOverlapCap = model->MOS3gateSourceOverlapCapFactor * 
                    here->MOS3m * EffectiveWidth;
            GateDrainOverlapCap = model->MOS3gateDrainOverlapCapFactor * 
                    here->MOS3m * EffectiveWidth;
            GateBulkOverlapCap = model->MOS3gateBulkOverlapCapFactor * 
                    here->MOS3m * EffectiveLength;
            /*
             *     meyer"s model parameters
             */
            capgs = ( *(ckt->CKTstate0+here->MOS3capgs)+ 
                      *(ckt->CKTstate0+here->MOS3capgs) +
                      GateSourceOverlapCap );
            capgd = ( *(ckt->CKTstate0+here->MOS3capgd)+ 
                      *(ckt->CKTstate0+here->MOS3capgd) +
                      GateDrainOverlapCap );
            capgb = ( *(ckt->CKTstate0+here->MOS3capgb)+ 
                      *(ckt->CKTstate0+here->MOS3capgb) +
                      GateBulkOverlapCap );
            xgs = capgs * ckt->CKTomega;
            xgd = capgd * ckt->CKTomega;
            xgb = capgb * ckt->CKTomega;
            xbd  = here->MOS3capbd * ckt->CKTomega;
            xbs  = here->MOS3capbs * ckt->CKTomega;

            /* 
             *  load matrix
             */

            *(here->MOS3GgPtr +1) += xgd+xgs+xgb;
            *(here->MOS3BbPtr +1) += xgb+xbd+xbs;
            *(here->MOS3DPdpPtr +1) += xgd+xbd;
            *(here->MOS3SPspPtr +1) += xgs+xbs;
            *(here->MOS3GbPtr +1) -= xgb;
            *(here->MOS3GdpPtr +1) -= xgd;
            *(here->MOS3GspPtr +1) -= xgs;
            *(here->MOS3BgPtr +1) -= xgb;
            *(here->MOS3BdpPtr +1) -= xbd;
            *(here->MOS3BspPtr +1) -= xbs;
            *(here->MOS3DPgPtr +1) -= xgd;
            *(here->MOS3DPbPtr +1) -= xbd;
            *(here->MOS3SPgPtr +1) -= xgs;
            *(here->MOS3SPbPtr +1) -= xbs;
            *(here->MOS3DdPtr) += here->MOS3drainConductance;
            *(here->MOS3SsPtr) += here->MOS3sourceConductance;
            *(here->MOS3BbPtr) += here->MOS3gbd+here->MOS3gbs;
            *(here->MOS3DPdpPtr) += here->MOS3drainConductance+
                    here->MOS3gds+here->MOS3gbd+
                    xrev*(here->MOS3gm+here->MOS3gmbs);
            *(here->MOS3SPspPtr) += here->MOS3sourceConductance+
                    here->MOS3gds+here->MOS3gbs+
                    xnrm*(here->MOS3gm+here->MOS3gmbs);
            *(here->MOS3DdpPtr) -= here->MOS3drainConductance;
            *(here->MOS3SspPtr) -= here->MOS3sourceConductance;
            *(here->MOS3BdpPtr) -= here->MOS3gbd;
            *(here->MOS3BspPtr) -= here->MOS3gbs;
            *(here->MOS3DPdPtr) -= here->MOS3drainConductance;
            *(here->MOS3DPgPtr) += (xnrm-xrev)*here->MOS3gm;
            *(here->MOS3DPbPtr) += -here->MOS3gbd+(xnrm-xrev)*here->MOS3gmbs;
            *(here->MOS3DPspPtr) -= here->MOS3gds+
                    xnrm*(here->MOS3gm+here->MOS3gmbs);
            *(here->MOS3SPgPtr) -= (xnrm-xrev)*here->MOS3gm;
            *(here->MOS3SPsPtr) -= here->MOS3sourceConductance;
            *(here->MOS3SPbPtr) -= here->MOS3gbs+(xnrm-xrev)*here->MOS3gmbs;
            *(here->MOS3SPdpPtr) -= here->MOS3gds+
                    xrev*(here->MOS3gm+here->MOS3gmbs);
        }
    }
    return(OK);
}
