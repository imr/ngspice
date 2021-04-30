/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: 2000 AlansFixes
VDMOS: 2018 Holger Vogt, 2020 Dietmar Warning
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
    double xgs, xcgT;
    double xgd, xcdT;
    double capgs, cgT;
    double capgd, cdT;
    double cTt, gTtt, gTtg, gTtdp, gTtsp;
    double GmT;
    double xcsT, xcTt;

    int selfheat;

    for( ; model != NULL; model = VDMOSnextModel(model)) {
        for(here = VDMOSinstances(model); here!= NULL;
                here = VDMOSnextInstance(here)) {

            selfheat = (here->VDMOSthermal) && (model->VDMOSrthjcGiven);
            if (here->VDMOSmode < 0) {
                xnrm=0;
                xrev=1;
            } else {
                xnrm=1;
                xrev=0;
            }

            if (here->VDMOSmode >= 0) {
                GmT = model->VDMOStype * here->VDMOSgmT;
                cgT  = model->VDMOStype * here->VDMOScgT;
                cdT  = model->VDMOStype * here->VDMOScdT;
                cTt = model->VDMOScthj;
                gTtg  = here->VDMOSgtempg;
                gTtdp = here->VDMOSgtempd;
                gTtt  = here->VDMOSgtempT;
                gTtsp = - (gTtg + gTtdp);
            } else {
                GmT = -model->VDMOStype * here->VDMOSgmT;
                cgT  = -model->VDMOStype * here->VDMOScgT;
                cdT  = -model->VDMOStype * here->VDMOScdT;
                cTt = -model->VDMOScthj;
                gTtg  = -here->VDMOSgtempg;
                gTtdp = -here->VDMOSgtempd;
                gTtt  = -here->VDMOSgtempT;
                gTtsp = gTtg + gTtdp;
            }

            /*
             *     VDMOS cap model parameters
             */
            capgs = ( *(ckt->CKTstate0+here->VDMOScapgs)+ 
                      *(ckt->CKTstate0+here->VDMOScapgs));
            capgd = ( *(ckt->CKTstate0+here->VDMOScapgd)+ 
                      *(ckt->CKTstate0+here->VDMOScapgd));
            xgs = capgs * ckt->CKTomega;
            xgd = capgd * ckt->CKTomega;

            xcgT = cgT * ckt->CKTomega;
            xcdT = cdT * ckt->CKTomega;
            xcsT = -(cgT + cdT) * ckt->CKTomega;
            xcTt = cTt * ckt->CKTomega;

            /* body diode */
            double gspr, geq, xceq;
            gspr = here->VDIOtConductance;
            geq = *(ckt->CKTstate0 + here->VDIOconduct);
            xceq= *(ckt->CKTstate0 + here->VDIOcapCurrent) * ckt->CKTomega;

            /*
             *    load matrix
             */
            *(here->VDMOSGPgpPtr +1) += xgd+xgs;
            *(here->VDMOSDPdpPtr +1) += xgd;
            *(here->VDMOSSPspPtr +1) += xgs;
            *(here->VDMOSGPdpPtr +1) -= xgd;
            *(here->VDMOSGPspPtr +1) -= xgs;
            *(here->VDMOSDPgpPtr +1) -= xgd;
            *(here->VDMOSSPgpPtr +1) -= xgs;

            *(here->VDMOSDdPtr) += here->VDMOSdrainConductance;
            *(here->VDMOSSsPtr) += here->VDMOSsourceConductance;
            *(here->VDMOSDPdpPtr) += here->VDMOSdrainConductance+
                    here->VDMOSgds+xrev*(here->VDMOSgm);
            *(here->VDMOSSPspPtr) += here->VDMOSsourceConductance+
                    here->VDMOSgds+xnrm*(here->VDMOSgm);
            *(here->VDMOSDdpPtr) -= here->VDMOSdrainConductance;
            *(here->VDMOSSspPtr) -= here->VDMOSsourceConductance;
            *(here->VDMOSDPdPtr) -= here->VDMOSdrainConductance;
            *(here->VDMOSDPgpPtr) += (xnrm-xrev)*here->VDMOSgm;
            *(here->VDMOSDPspPtr) -= here->VDMOSgds+xnrm*(here->VDMOSgm);
            *(here->VDMOSSPgpPtr) -= (xnrm-xrev)*here->VDMOSgm;
            *(here->VDMOSSPsPtr) -= here->VDMOSsourceConductance;
            *(here->VDMOSSPdpPtr) -= here->VDMOSgds+xrev*(here->VDMOSgm);
            /* gate resistor */
            *(here->VDMOSGgPtr) += (here->VDMOSgateConductance);
            *(here->VDMOSGPgpPtr) += (here->VDMOSgateConductance);
            *(here->VDMOSGgpPtr) -= here->VDMOSgateConductance;
            *(here->VDMOSGPgPtr) -= here->VDMOSgateConductance;
            /* body diode */
            *(here->VDMOSSsPtr) += gspr;
            *(here->VDMOSDdPtr) += geq;
            *(here->VDMOSDdPtr +1) += xceq;
            *(here->VDIORPrpPtr) += geq+gspr;
            *(here->VDIORPrpPtr +1) += xceq;
            *(here->VDIOSrpPtr) -= gspr;
            *(here->VDIODrpPtr) -= geq;
            *(here->VDIODrpPtr +1) -= xceq;
            *(here->VDIORPsPtr) -= gspr;
            *(here->VDIORPdPtr) -= geq;
            *(here->VDIORPdPtr +1) -= xceq;
            if (selfheat)
            {
               *(here->VDMOSDPtempPtr)       += GmT;
               *(here->VDMOSSPtempPtr)       += -GmT;

               *(here->VDMOSTemptempPtr)     += gTtt + 1/model->VDMOSrthjc;
               *(here->VDMOSTempgpPtr)       += gTtg;
               *(here->VDMOSTempdpPtr)       += gTtdp;
               *(here->VDMOSTempspPtr)       += gTtsp;
               *(here->VDMOSTemptcasePtr)    += -1/model->VDMOSrthjc;
               *(here->VDMOSTcasetempPtr)    += -1/model->VDMOSrthjc;
               *(here->VDMOSTcasetcasePtr)   +=  1/model->VDMOSrthjc + 1/model->VDMOSrthca;
               *(here->VDMOSTptpPtr)         +=  1/model->VDMOSrthca;
               *(here->VDMOSTptcasePtr)      += -1/model->VDMOSrthca;
               *(here->VDMOSTcasetpPtr)      += -1/model->VDMOSrthca;
               *(here->VDMOSCktTtpPtr)       +=  1.0;
               *(here->VDMOSTpcktTPtr)       +=  1.0;

               *(here->VDMOSTemptempPtr + 1) += xcTt;
               *(here->VDMOSDPtempPtr + 1)   += xcdT;
               *(here->VDMOSSPtempPtr + 1)   += xcsT;
               *(here->VDMOSGPtempPtr + 1)   += xcgT;
            }
        }
    }
    return(OK);
}
