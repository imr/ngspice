/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: 2000 AlansFixes
**********/
/*
 */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ngspice/complex.h"
#include "vdmosdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
VDMOSpzLoad(GENmodel *inModel, CKTcircuit *ckt, SPcomplex *s)
{
    VDMOSmodel *model = (VDMOSmodel*)inModel;
    VDMOSinstance *here;
    int xnrm;
    int xrev;
    double xgs, xgd;
    double capgs, cgT;
    double capgd, cdT;
    double cTt, csT, gTtt, gTtg, gTtdp, gTtsp;
    double GmT;

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
            csT = -(cgT + cdT);

            /*
             *     VDMOS cap model parameters
             */
            capgs = ( *(ckt->CKTstate0+here->VDMOScapgs)+ 
                      *(ckt->CKTstate0+here->VDMOScapgs));
            capgd = ( *(ckt->CKTstate0+here->VDMOScapgd)+ 
                      *(ckt->CKTstate0+here->VDMOScapgd));
            xgs = capgs;
            xgd = capgd;

            /* body diode */
            double gspr, geq, xceq;
            gspr = here->VDIOtConductance;
            geq = *(ckt->CKTstate0 + here->VDIOconduct);
            xceq = *(ckt->CKTstate0 + here->VDIOcapCurrent);

            /*
             *    load matrix
             */

            *(here->VDMOSGPgpPtr   ) += (xgd+xgs)*s->real;
            *(here->VDMOSGPgpPtr +1) += (xgd+xgs)*s->imag;
            *(here->VDMOSDPdpPtr   ) += (xgd)*s->real;
            *(here->VDMOSDPdpPtr +1) += (xgd)*s->imag;
            *(here->VDMOSSPspPtr   ) += (xgs)*s->real;
            *(here->VDMOSSPspPtr +1) += (xgs)*s->imag;
            *(here->VDMOSGPdpPtr   ) -= xgd*s->real;
            *(here->VDMOSGPdpPtr +1) -= xgd*s->imag;
            *(here->VDMOSGPspPtr   ) -= xgs*s->real;
            *(here->VDMOSGPspPtr +1) -= xgs*s->imag;
            *(here->VDMOSDPgpPtr   ) -= xgd*s->real;
            *(here->VDMOSDPgpPtr +1) -= xgd*s->imag;
            *(here->VDMOSSPgpPtr   ) -= xgs*s->real;
            *(here->VDMOSSPgpPtr +1) -= xgs*s->imag;

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
            *(here->VDMOSDdPtr) += geq + xceq * s->real;
            *(here->VDMOSDdPtr +1 ) += xceq * s->imag;
            *(here->VDIORPrpPtr) += geq + gspr + xceq * s->real;
            *(here->VDIORPrpPtr +1) += xceq * s->imag;
            *(here->VDIOSrpPtr) -= gspr;
            *(here->VDIODrpPtr) -= geq + xceq * s->real;
            *(here->VDIODrpPtr +1) -= xceq * s->imag;
            *(here->VDIORPsPtr) -= gspr;
            *(here->VDIORPdPtr) -= geq + xceq * s->real;
            *(here->VDIORPdPtr +1 ) -= xceq * s->imag;

            if (VDMOSselfheat(here)) {
                /* The body diode's thermal coupling (dIth_dVdio, dIdio_dT,
                   dIth_dVrb, dIrb_dT) is deliberately omitted here: an AC or
                   PZ operating point with a conducting body diode does not
                   occur in practice.  See vdmosload.c for the full set.     */
                double dIrd_dT    = here->VDMOSdIrd_dT;
                double dIrs_dT    = here->VDMOSdIrs_dT;
                double dIth_dVrd  = here->VDMOSdIth_dVrd;
                double dIth_dVrs  = here->VDMOSdIth_dVrs;
                double dIth_dTres = here->VDMOSdIth_dTres;
                /* Everything is computed for m parallel instances...
                   so scale gthjc and gthca accordingly */
                double gthjc = here->VDMOSm / model->VDMOSrthjc;
                double gthca = here->VDMOSm / model->VDMOSrthca;

                *(here->VDMOSDtempPtr)        +=  dIrd_dT;
                *(here->VDMOSStempPtr)        +=  dIrs_dT;
                *(here->VDMOSDPtempPtr)       +=  GmT - dIrd_dT + cdT * s->real;
                *(here->VDMOSDPtempPtr + 1)   +=  cdT * s->imag;
                *(here->VDMOSSPtempPtr)       += -GmT - dIrs_dT + csT * s->real;
                *(here->VDMOSSPtempPtr + 1)   +=  csT * s->imag;
                *(here->VDMOSGPtempPtr)       +=  cgT * s->real;
                *(here->VDMOSGPtempPtr + 1)   +=  cgT * s->imag;

                *(here->VDMOSTemptempPtr)     += -gTtt - dIth_dTres + gthjc + cTt * s->real;
                *(here->VDMOSTemptempPtr + 1) +=  cTt * s->imag;
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
            }

            if (VDIOrevrec(here)) {
                /* QP subcircuit */
                double gdres   = *(ckt->CKTstate0 + here->VDIOresConduct);
                double dcrrdvd = here->VDIOcurFactor * gdres;
                double capsr   = here->VDIOtTransitTime;
                double gain    = here->VDIOqpGainScaled * capsr;

                *(here->VDIOqpQpPtr)           += 1/model->VDIOsoftRevRecParam
                                                + capsr * s->real;
                *(here->VDIOqpQpPtr + 1)       += capsr * s->imag;
                *(here->VDIOqpPosPrimePtr)     += -dcrrdvd;
                *(here->VDIOqpNegPtr)          +=  dcrrdvd;
                /* gqcsr -> s*capsr, capsr = TT */
                *(here->VDIOposPrimeQpPtr)     +=  gain * s->real;
                *(here->VDIOposPrimeQpPtr + 1) +=  gain * s->imag;
                *(here->VDIOnegQpPtr)          += -gain * s->real;
                *(here->VDIOnegQpPtr + 1)      += -gain * s->imag;

                if (VDMOSselfheat(here)) {
                    double vdop = *(ckt->CKTstate0 + here->VDIOvoltage);
                    *(here->VDIOqpTempPtr)     += -model->VDMOStype * here->VDIOcurFactor
                                               * *(ckt->CKTstate0 + here->VDIOdIdio_dT);
                    *(here->VDIOtempQpPtr)     += -model->VDMOStype * vdop * gain * s->real;
                    *(here->VDIOtempQpPtr + 1) += -model->VDMOStype * vdop * gain * s->imag;
                }
            }

        }
    }
    return(OK);
}
