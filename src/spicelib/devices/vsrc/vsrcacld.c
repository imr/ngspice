/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "vsrcdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
VSRCacLoad(GENmodel *inModel, CKTcircuit *ckt)
{
    VSRCmodel *model = (VSRCmodel *) inModel;
    VSRCinstance *here;

    for( ; model != NULL; model = VSRCnextModel(model)) {

        /* loop through all the instances of the model */
        for (here = VSRCinstances(model); here != NULL ;
                here=VSRCnextInstance(here)) {

            double acReal, acImag;

            if (ckt->CKTmode & MODEACNOISE) {
                if ((GENinstance *) here == ckt->noise_input) {
                    acReal = 1.0;
                    acImag = 0.0;
                } else {
                    acReal = 0.0;
                    acImag = 0.0;
                }
            } else {
                acReal = here->VSRCacReal;
                acImag = here->VSRCacImag;
            }

            *(here->VSRCposIbrPtr) += 1.0 ;
            *(here->VSRCnegIbrPtr) -= 1.0 ;
            *(here->VSRCibrPosPtr) += 1.0 ;
            *(here->VSRCibrNegPtr) -= 1.0 ;
            *(ckt->CKTrhs + (here->VSRCbranch)) += acReal;
            *(ckt->CKTirhs + (here->VSRCbranch)) += acImag;
        }
    }

    return(OK);
}
