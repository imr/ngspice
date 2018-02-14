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
#include "mos3defs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
MOS3pzLoad(GENmodel *inModel, CKTcircuit *ckt, SPcomplex *s)
{
    MOS3model *model = (MOS3model *)inModel;
    MOS3instance *here;
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
             *     meyer's model parameters
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
            capgs = ( 2* *(ckt->CKTstate0+here->MOS3capgs)+ 
                      GateSourceOverlapCap );
            capgd = ( 2* *(ckt->CKTstate0+here->MOS3capgd)+ 
                      GateDrainOverlapCap );
            capgb = ( 2* *(ckt->CKTstate0+here->MOS3capgb)+ 
                      GateBulkOverlapCap );
            xgs = capgs;
            xgd = capgd;
            xgb = capgb;
            xbd  = here->MOS3capbd;
            xbs  = here->MOS3capbs;
            /*printf("mos2: xgs=%g, xgd=%g, xgb=%g, xbd=%g, xbs=%g\n",
                    xgs,xgd,xgb,xbd,xbs);*/
            /*
             *    load matrix
             */

            *(here->MOS3GgPtr   ) += (xgd+xgs+xgb)*s->real;
            *(here->MOS3GgPtr +1) += (xgd+xgs+xgb)*s->imag;
            *(here->MOS3BbPtr   ) += (xgb+xbd+xbs)*s->real;
            *(here->MOS3BbPtr +1) += (xgb+xbd+xbs)*s->imag;
            *(here->MOS3DPdpPtr   ) += (xgd+xbd)*s->real;
            *(here->MOS3DPdpPtr +1) += (xgd+xbd)*s->imag;
            *(here->MOS3SPspPtr   ) += (xgs+xbs)*s->real;
            *(here->MOS3SPspPtr +1) += (xgs+xbs)*s->imag;
            *(here->MOS3GbPtr   ) -= xgb*s->real;
            *(here->MOS3GbPtr +1) -= xgb*s->imag;
            *(here->MOS3GdpPtr   ) -= xgd*s->real;
            *(here->MOS3GdpPtr +1) -= xgd*s->imag;
            *(here->MOS3GspPtr   ) -= xgs*s->real;
            *(here->MOS3GspPtr +1) -= xgs*s->imag;
            *(here->MOS3BgPtr   ) -= xgb*s->real;
            *(here->MOS3BgPtr +1) -= xgb*s->imag;
            *(here->MOS3BdpPtr   ) -= xbd*s->real;
            *(here->MOS3BdpPtr +1) -= xbd*s->imag;
            *(here->MOS3BspPtr   ) -= xbs*s->real;
            *(here->MOS3BspPtr +1) -= xbs*s->imag;
            *(here->MOS3DPgPtr   ) -= xgd*s->real;
            *(here->MOS3DPgPtr +1) -= xgd*s->imag;
            *(here->MOS3DPbPtr   ) -= xbd*s->real;
            *(here->MOS3DPbPtr +1) -= xbd*s->imag;
            *(here->MOS3SPgPtr   ) -= xgs*s->real;
            *(here->MOS3SPgPtr +1) -= xgs*s->imag;
            *(here->MOS3SPbPtr   ) -= xbs*s->real;
            *(here->MOS3SPbPtr +1) -= xbs*s->imag;
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
