/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: Alan Gillespie
**********/
/*
 */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ngspice/complex.h"
#include "mos9defs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
MOS9pzLoad(GENmodel *inModel, CKTcircuit *ckt, SPcomplex *s)
{
    MOS9model *model = (MOS9model *)inModel;
    MOS9instance *here;
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
    double EffectiveWidth;

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
             *     meyer's model parameters
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

            capgs = ( 2* *(ckt->CKTstate0+here->MOS9capgs)+ 
                      GateSourceOverlapCap );
            capgd = ( 2* *(ckt->CKTstate0+here->MOS9capgd)+ 
                      GateDrainOverlapCap );
            capgb = ( 2* *(ckt->CKTstate0+here->MOS9capgb)+ 
                      GateBulkOverlapCap );
            xgs = capgs;
            xgd = capgd;
            xgb = capgb;
            xbd  = here->MOS9capbd;
            xbs  = here->MOS9capbs;
            /*printf("mos2: xgs=%g, xgd=%g, xgb=%g, xbd=%g, xbs=%g\n",
                    xgs,xgd,xgb,xbd,xbs);*/
            /*
             *    load matrix
             */

            *(here->MOS9GgPtr   ) += (xgd+xgs+xgb)*s->real;
            *(here->MOS9GgPtr +1) += (xgd+xgs+xgb)*s->imag;
            *(here->MOS9BbPtr   ) += (xgb+xbd+xbs)*s->real;
            *(here->MOS9BbPtr +1) += (xgb+xbd+xbs)*s->imag;
            *(here->MOS9DPdpPtr   ) += (xgd+xbd)*s->real;
            *(here->MOS9DPdpPtr +1) += (xgd+xbd)*s->imag;
            *(here->MOS9SPspPtr   ) += (xgs+xbs)*s->real;
            *(here->MOS9SPspPtr +1) += (xgs+xbs)*s->imag;
            *(here->MOS9GbPtr   ) -= xgb*s->real;
            *(here->MOS9GbPtr +1) -= xgb*s->imag;
            *(here->MOS9GdpPtr   ) -= xgd*s->real;
            *(here->MOS9GdpPtr +1) -= xgd*s->imag;
            *(here->MOS9GspPtr   ) -= xgs*s->real;
            *(here->MOS9GspPtr +1) -= xgs*s->imag;
            *(here->MOS9BgPtr   ) -= xgb*s->real;
            *(here->MOS9BgPtr +1) -= xgb*s->imag;
            *(here->MOS9BdpPtr   ) -= xbd*s->real;
            *(here->MOS9BdpPtr +1) -= xbd*s->imag;
            *(here->MOS9BspPtr   ) -= xbs*s->real;
            *(here->MOS9BspPtr +1) -= xbs*s->imag;
            *(here->MOS9DPgPtr   ) -= xgd*s->real;
            *(here->MOS9DPgPtr +1) -= xgd*s->imag;
            *(here->MOS9DPbPtr   ) -= xbd*s->real;
            *(here->MOS9DPbPtr +1) -= xbd*s->imag;
            *(here->MOS9SPgPtr   ) -= xgs*s->real;
            *(here->MOS9SPgPtr +1) -= xgs*s->imag;
            *(here->MOS9SPbPtr   ) -= xbs*s->real;
            *(here->MOS9SPbPtr +1) -= xbs*s->imag;
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
