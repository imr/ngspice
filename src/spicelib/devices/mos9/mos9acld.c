/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: Alan Gillespie
**********/
/*
 */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "mos9defs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
MOS9acLoad(GENmodel *inModel, CKTcircuit *ckt)
{
    MOS9model *model = (MOS9model *)inModel;
    MOS9instance *here;
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

    for( ; model != NULL; model = MOS9nextModel(model)) {
        for(here = MOS9instances(model); here!= NULL;
                here = MOS9nextInstance(here)) {

            if (here->MOS9mode < 0) {
                xnrm=0;
                xrev=1;
            } else {
                xnrm=1;
                xrev=0;
            }
            /*
             *    charge oriented model parameters
             */
            EffectiveWidth=here->MOS9w-2*model->MOS9widthNarrow+
                                                    model->MOS9widthAdjust;
            EffectiveLength=here->MOS9l - 2*model->MOS9latDiff+
                                                    model->MOS9lengthAdjust;
            GateSourceOverlapCap = model->MOS9gateSourceOverlapCapFactor * 
                    here->MOS9m * EffectiveWidth;
            GateDrainOverlapCap = model->MOS9gateDrainOverlapCapFactor * 
                    here->MOS9m * EffectiveWidth;
            GateBulkOverlapCap = model->MOS9gateBulkOverlapCapFactor * 
                    here->MOS9m * EffectiveLength;


            /*
             *     meyer"s model parameters
             */
            capgs = ( *(ckt->CKTstate0+here->MOS9capgs)+ 
                      *(ckt->CKTstate0+here->MOS9capgs) +
                      GateSourceOverlapCap );
            capgd = ( *(ckt->CKTstate0+here->MOS9capgd)+ 
                      *(ckt->CKTstate0+here->MOS9capgd) +
                      GateDrainOverlapCap );
            capgb = ( *(ckt->CKTstate0+here->MOS9capgb)+ 
                      *(ckt->CKTstate0+here->MOS9capgb) +
                      GateBulkOverlapCap );
            xgs = capgs * ckt->CKTomega;
            xgd = capgd * ckt->CKTomega;
            xgb = capgb * ckt->CKTomega;
            xbd  = here->MOS9capbd * ckt->CKTomega;
            xbs  = here->MOS9capbs * ckt->CKTomega;

            /* 
             *  load matrix
             */

            *(here->MOS9GgPtr +1) += xgd+xgs+xgb;
            *(here->MOS9BbPtr +1) += xgb+xbd+xbs;
            *(here->MOS9DPdpPtr +1) += xgd+xbd;
            *(here->MOS9SPspPtr +1) += xgs+xbs;
            *(here->MOS9GbPtr +1) -= xgb;
            *(here->MOS9GdpPtr +1) -= xgd;
            *(here->MOS9GspPtr +1) -= xgs;
            *(here->MOS9BgPtr +1) -= xgb;
            *(here->MOS9BdpPtr +1) -= xbd;
            *(here->MOS9BspPtr +1) -= xbs;
            *(here->MOS9DPgPtr +1) -= xgd;
            *(here->MOS9DPbPtr +1) -= xbd;
            *(here->MOS9SPgPtr +1) -= xgs;
            *(here->MOS9SPbPtr +1) -= xbs;
            *(here->MOS9DdPtr) += here->MOS9drainConductance;
            *(here->MOS9SsPtr) += here->MOS9sourceConductance;
            *(here->MOS9BbPtr) += here->MOS9gbd+here->MOS9gbs;
            *(here->MOS9DPdpPtr) += here->MOS9drainConductance+
                    here->MOS9gds+here->MOS9gbd+
                    xrev*(here->MOS9gm+here->MOS9gmbs);
            *(here->MOS9SPspPtr) += here->MOS9sourceConductance+
                    here->MOS9gds+here->MOS9gbs+
                    xnrm*(here->MOS9gm+here->MOS9gmbs);
            *(here->MOS9DdpPtr) -= here->MOS9drainConductance;
            *(here->MOS9SspPtr) -= here->MOS9sourceConductance;
            *(here->MOS9BdpPtr) -= here->MOS9gbd;
            *(here->MOS9BspPtr) -= here->MOS9gbs;
            *(here->MOS9DPdPtr) -= here->MOS9drainConductance;
            *(here->MOS9DPgPtr) += (xnrm-xrev)*here->MOS9gm;
            *(here->MOS9DPbPtr) += -here->MOS9gbd+(xnrm-xrev)*here->MOS9gmbs;
            *(here->MOS9DPspPtr) -= here->MOS9gds+
                    xnrm*(here->MOS9gm+here->MOS9gmbs);
            *(here->MOS9SPgPtr) -= (xnrm-xrev)*here->MOS9gm;
            *(here->MOS9SPsPtr) -= here->MOS9sourceConductance;
            *(here->MOS9SPbPtr) -= here->MOS9gbs+(xnrm-xrev)*here->MOS9gmbs;
            *(here->MOS9SPdpPtr) -= here->MOS9gds+
                    xrev*(here->MOS9gm+here->MOS9gmbs);
        }
    }
    return(OK);
}
