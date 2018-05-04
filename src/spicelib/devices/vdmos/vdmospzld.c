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
           
            capgs = ( 2* *(ckt->CKTstate0+here->VDMOScapgs)+ 
                      GateSourceOverlapCap );
            capgd = ( 2* *(ckt->CKTstate0+here->VDMOScapgd)+ 
                      GateDrainOverlapCap );
            capgb = ( 2* *(ckt->CKTstate0+here->VDMOScapgb)+ 
                      GateBulkOverlapCap );
            xgs = capgs;
            xgd = capgd;
            xgb = capgb;
            xbd  = here->VDMOScapbd;
            xbs  = here->VDMOScapbs;
            /*printf("vdmos: xgs=%g, xgd=%g, xgb=%g, xbd=%g, xbs=%g\n",
                    xgs,xgd,xgb,xbd,xbs);*/
            /*
             *    load matrix
             */

            *(here->VDMOSGgPtr   ) += (xgd+xgs+xgb)*s->real;
            *(here->VDMOSGgPtr +1) += (xgd+xgs+xgb)*s->imag;
            *(here->VDMOSBbPtr   ) += (xgb+xbd+xbs)*s->real;
            *(here->VDMOSBbPtr +1) += (xgb+xbd+xbs)*s->imag;
            *(here->VDMOSDPdpPtr   ) += (xgd+xbd)*s->real;
            *(here->VDMOSDPdpPtr +1) += (xgd+xbd)*s->imag;
            *(here->VDMOSSPspPtr   ) += (xgs+xbs)*s->real;
            *(here->VDMOSSPspPtr +1) += (xgs+xbs)*s->imag;
            *(here->VDMOSGbPtr   ) -= xgb*s->real;
            *(here->VDMOSGbPtr +1) -= xgb*s->imag;
            *(here->VDMOSGdpPtr   ) -= xgd*s->real;
            *(here->VDMOSGdpPtr +1) -= xgd*s->imag;
            *(here->VDMOSGspPtr   ) -= xgs*s->real;
            *(here->VDMOSGspPtr +1) -= xgs*s->imag;
            *(here->VDMOSBgPtr   ) -= xgb*s->real;
            *(here->VDMOSBgPtr +1) -= xgb*s->imag;
            *(here->VDMOSBdpPtr   ) -= xbd*s->real;
            *(here->VDMOSBdpPtr +1) -= xbd*s->imag;
            *(here->VDMOSBspPtr   ) -= xbs*s->real;
            *(here->VDMOSBspPtr +1) -= xbs*s->imag;
            *(here->VDMOSDPgPtr   ) -= xgd*s->real;
            *(here->VDMOSDPgPtr +1) -= xgd*s->imag;
            *(here->VDMOSDPbPtr   ) -= xbd*s->real;
            *(here->VDMOSDPbPtr +1) -= xbd*s->imag;
            *(here->VDMOSSPgPtr   ) -= xgs*s->real;
            *(here->VDMOSSPgPtr +1) -= xgs*s->imag;
            *(here->VDMOSSPbPtr   ) -= xbs*s->real;
            *(here->VDMOSSPbPtr +1) -= xbs*s->imag;
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
