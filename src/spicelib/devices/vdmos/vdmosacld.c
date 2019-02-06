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
             *     VDMOS cap model parameters
             */
            capgs = ( *(ckt->CKTstate0+here->VDMOScapgs)+ 
                      *(ckt->CKTstate0+here->VDMOScapgs));
            capgd = ( *(ckt->CKTstate0+here->VDMOScapgd)+ 
                      *(ckt->CKTstate0+here->VDMOScapgd));
            xgs = capgs * ckt->CKTomega;
            xgd = capgd * ckt->CKTomega;

            /* bulk diode */
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
            *(here->VDMOSGPgpPtr) += (here->VDMOSgateConductance)/* + ?? FIXME */;
            *(here->VDMOSGgpPtr) -= here->VDMOSgateConductance;
            *(here->VDMOSGPgPtr) -= here->VDMOSgateConductance;
            /* bulk diode */
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
        }
    }
    return(OK);
}
