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
    double capgs;
    double capgd;

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
            capgs = ( 2* *(ckt->CKTstate0+here->VDMOScapgs));
            capgd = ( 2* *(ckt->CKTstate0+here->VDMOScapgd));
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

            *(here->VDMOSGgPtr   ) += (xgd+xgs)*s->real;
            *(here->VDMOSGgPtr +1) += (xgd+xgs)*s->imag;
            *(here->VDMOSDPdpPtr   ) += (xgd)*s->real;
            *(here->VDMOSDPdpPtr +1) += (xgd)*s->imag;
            *(here->VDMOSSPspPtr   ) += (xgs)*s->real;
            *(here->VDMOSSPspPtr +1) += (xgs)*s->imag;
            *(here->VDMOSGdpPtr   ) -= xgd*s->real;
            *(here->VDMOSGdpPtr +1) -= xgd*s->imag;
            *(here->VDMOSGspPtr   ) -= xgs*s->real;
            *(here->VDMOSGspPtr +1) -= xgs*s->imag;
            *(here->VDMOSDPgPtr   ) -= xgd*s->real;
            *(here->VDMOSDPgPtr +1) -= xgd*s->imag;
            *(here->VDMOSSPgPtr   ) -= xgs*s->real;
            *(here->VDMOSSPgPtr +1) -= xgs*s->imag;
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

        }
    }
    return(OK);
}
