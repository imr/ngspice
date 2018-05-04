/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: 2000 AlansFixes
**********/
/*
 */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "vdmosdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
VDMOSacLoad(GENmodel *inModel, CKTcircuit *ckt)
{
    VDMOSmodel *model = (VDMOSmodel*)inModel;
    VDMOSinstance *here;
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

    for( ; model != NULL; model = VDMOSnextModel(model)) {
        for(here = VDMOSinstances(model); here!= NULL;
                here = VDMOSnextInstance(here)) {
        
            if (here->VDMOSmode < 0) {
                xnrm=0;
                xrev=1;
            } else {
                xnrm=1;
                xrev=0;
            }
            /*
             *     meyer's model parameters
             */
            EffectiveLength=here->VDMOSl - 2*model->VDMOSlatDiff;
            
            GateSourceOverlapCap = model->VDMOSgateSourceOverlapCapFactor * 
                    here->VDMOSm * here->VDMOSw;
            GateDrainOverlapCap = model->VDMOSgateDrainOverlapCapFactor * 
                    here->VDMOSm * here->VDMOSw;
            GateBulkOverlapCap = model->VDMOSgateBulkOverlapCapFactor * 
                    here->VDMOSm * EffectiveLength;
                    
            capgs = ( *(ckt->CKTstate0+here->VDMOScapgs)+ 
                      *(ckt->CKTstate0+here->VDMOScapgs) +
                      GateSourceOverlapCap );
            capgd = ( *(ckt->CKTstate0+here->VDMOScapgd)+ 
                      *(ckt->CKTstate0+here->VDMOScapgd) +
                      GateDrainOverlapCap );
            capgb = ( *(ckt->CKTstate0+here->VDMOScapgb)+ 
                      *(ckt->CKTstate0+here->VDMOScapgb) +
                      GateBulkOverlapCap );
            xgs = capgs * ckt->CKTomega;
            xgd = capgd * ckt->CKTomega;
            xgb = capgb * ckt->CKTomega;
            xbd  = here->VDMOScapbd * ckt->CKTomega;
            xbs  = here->VDMOScapbs * ckt->CKTomega;
            /*
             *    load matrix
             */

            *(here->VDMOSGgPtr +1) += xgd+xgs+xgb;
            *(here->VDMOSBbPtr +1) += xgb+xbd+xbs;
            *(here->VDMOSDPdpPtr +1) += xgd+xbd;
            *(here->VDMOSSPspPtr +1) += xgs+xbs;
            *(here->VDMOSGbPtr +1) -= xgb;
            *(here->VDMOSGdpPtr +1) -= xgd;
            *(here->VDMOSGspPtr +1) -= xgs;
            *(here->VDMOSBgPtr +1) -= xgb;
            *(here->VDMOSBdpPtr +1) -= xbd;
            *(here->VDMOSBspPtr +1) -= xbs;
            *(here->VDMOSDPgPtr +1) -= xgd;
            *(here->VDMOSDPbPtr +1) -= xbd;
            *(here->VDMOSSPgPtr +1) -= xgs;
            *(here->VDMOSSPbPtr +1) -= xbs;
            *(here->VDMOSDdPtr) += here->VDMOSdrainConductance;
            *(here->VDMOSSsPtr) += here->VDMOSsourceConductance;
            *(here->VDMOSBbPtr) += here->VDMOSgbd+here->VDMOSgbs;
            *(here->VDMOSDPdpPtr) += here->VDMOSdrainConductance+
                    here->VDMOSgds+here->VDMOSgbd+
                    xrev*(here->VDMOSgm+here->VDMOSgmbs);
            *(here->VDMOSSPspPtr) += here->VDMOSsourceConductance+
                    here->VDMOSgds+here->VDMOSgbs+
                    xnrm*(here->VDMOSgm+here->VDMOSgmbs);
            *(here->VDMOSDdpPtr) -= here->VDMOSdrainConductance;
            *(here->VDMOSSspPtr) -= here->VDMOSsourceConductance;
            *(here->VDMOSBdpPtr) -= here->VDMOSgbd;
            *(here->VDMOSBspPtr) -= here->VDMOSgbs;
            *(here->VDMOSDPdPtr) -= here->VDMOSdrainConductance;
            *(here->VDMOSDPgPtr) += (xnrm-xrev)*here->VDMOSgm;
            *(here->VDMOSDPbPtr) += -here->VDMOSgbd+(xnrm-xrev)*here->VDMOSgmbs;
            *(here->VDMOSDPspPtr) -= here->VDMOSgds+
                    xnrm*(here->VDMOSgm+here->VDMOSgmbs);
            *(here->VDMOSSPgPtr) -= (xnrm-xrev)*here->VDMOSgm;
            *(here->VDMOSSPsPtr) -= here->VDMOSsourceConductance;
            *(here->VDMOSSPbPtr) -= here->VDMOSgbs+(xnrm-xrev)*here->VDMOSgmbs;
            *(here->VDMOSSPdpPtr) -= here->VDMOSgds+
                    xrev*(here->VDMOSgm+here->VDMOSgmbs);

        }
    }
    return(OK);
}
