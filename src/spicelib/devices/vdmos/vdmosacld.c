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

            if (here->VDMOSmode >= 0) {
                GmT   =  model->VDMOStype * here->VDMOSgmT;
                cgT   =  model->VDMOStype * here->VDMOScgT;
                cdT   =  model->VDMOStype * here->VDMOScdT;
                gTtg  =  here->VDMOSgtempg;
                gTtdp =  here->VDMOSgtempd;
                gTtt  =  here->VDMOSgtempT;
                gTtsp = -(gTtg + gTtdp);
            } else {
                /* Reverse operation: Drain and Source change their role,
                   sign remains - identically vdmosload.c   */
                GmT   = -model->VDMOStype * here->VDMOSgmT;
                cgT   =  model->VDMOStype * here->VDMOScgT;
                cdT   = -model->VDMOStype * here->VDMOScdT;
                gTtg  =  here->VDMOSgtempg;
                gTtsp =  here->VDMOSgtempd;
                gTtt  =  here->VDMOSgtempT;
                gTtdp = -(gTtg + gTtsp);
            }
            /* heat capacity - no impact by mode */
            cTt = here->VDMOSm * model->VDMOScthj;

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

            *(here->VDMOSDdPtr) += here->VDMOSdrainConductance + here->VDMOSdsConductance;
            *(here->VDMOSSsPtr) += here->VDMOSsourceConductance + here->VDMOSdsConductance;
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
            *(here->VDMOSDsPtr) += (-here->VDMOSdsConductance);
            *(here->VDMOSSdPtr) += (-here->VDMOSdsConductance);
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

            if (VDMOSselfheat(here)) {
                /* The body diode's thermal coupling (dIth_dVdio, dIdio_dT,
                   dIth_dVrb, dIrb_dT) is deliberately omitted here: an AC or
                   PZ operating point with a conducting body diode does not
                   occur in practice.  See vdmosload.c for the full set.     */
                double dIrd_dT = here->VDMOSdIrd_dT;
                double dIrs_dT = here->VDMOSdIrs_dT; 
                double dIth_dVrd = here->VDMOSdIth_dVrd; 
                double dIth_dVrs = here->VDMOSdIth_dVrs; 
                double dIrx_dT_Vrx = here->VDMOSdIth_dTres; 

                /* Everything is computed for m parallel instances...
                   so scale gthjc and gthja accordingly */
                double gthjc = here->VDMOSm / model->VDMOSrthjc;
                double gthca = here->VDMOSm / model->VDMOSrthca;
                *(here->VDMOSDtempPtr)        +=  dIrd_dT;
                *(here->VDMOSDPtempPtr)       +=  GmT - dIrd_dT;
                *(here->VDMOSStempPtr)        +=  dIrs_dT;
                *(here->VDMOSSPtempPtr)       += -GmT - dIrs_dT;
                *(here->VDMOSTemptempPtr)     += -gTtt - dIrx_dT_Vrx + gthjc;
                *(here->VDMOSTempgpPtr)       += -gTtg;
                *(here->VDMOSTempdPtr)        += -dIth_dVrd;
                *(here->VDMOSTempsPtr)        += -dIth_dVrs;
                *(here->VDMOSTempdpPtr)       += -gTtdp + dIth_dVrd;
                *(here->VDMOSTempspPtr)       += -gTtsp + dIth_dVrs;
                *(here->VDMOSTemptcasePtr)    += -gthjc;
                *(here->VDMOSTcasetempPtr)    += -gthjc;
                *(here->VDMOSTcasetcasePtr)   +=  gthjc + gthca;
                *(here->VDMOSTptpPtr)         +=  gthca;
                *(here->VDMOSTptcasePtr)      += -gthca;
                *(here->VDMOSTcasetpPtr)      += -gthca;
                *(here->VDMOSDevTtpPtr)       +=  1.0;
                *(here->VDMOSTpdevTPtr)       +=  1.0;

                *(here->VDMOSTemptempPtr + 1) += xcTt;
                *(here->VDMOSDPtempPtr + 1)   += xcdT;
                *(here->VDMOSSPtempPtr + 1)   += xcsT;
                *(here->VDMOSGPtempPtr + 1)   += xcgT;
            }

            if (VDIOrevrec(here)) {
                /* QP subcircuit */
                double gdres= *(ckt->CKTstate0 + here->VDIOresConduct);
                double dcrrdvd = here->VDIOcurFactor * gdres;
                *(here->VDIOqpQpPtr)       += 1/model->VDIOsoftRevRecParam;
                *(here->VDIOqpQpPtr + 1)   += here->VDIOtTransitTime * ckt->CKTomega;
                *(here->VDIOqpPosPrimePtr) += -dcrrdvd;
                *(here->VDIOqpNegPtr)      +=  dcrrdvd;
                /* AC: gqcsr -> j*omega*capsr, capsr = TT */
                double xgain = here->VDIOqpGainScaled * here->VDIOtTransitTime * ckt->CKTomega;
                *(here->VDIOposPrimeQpPtr + 1) +=  xgain;
                *(here->VDIOnegQpPtr + 1)      += -xgain;
                if (VDMOSselfheat(here)) {
                    double vdop = *(ckt->CKTstate0 + here->VDIOvoltage);
                    *(here->VDIOqpTempPtr) += -model->VDMOStype * here->VDIOcurFactor
                                           * *(ckt->CKTstate0 + here->VDIOdIdio_dT);
                    *(here->VDIOtempQpPtr + 1) += -model->VDMOStype * vdop * here->VDIOqpGainScaled
                                               * here->VDIOtTransitTime * ckt->CKTomega;
                }
            }
        }
    }
    return(OK);
}
