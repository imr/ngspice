/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#include "ngspice/ngspice.h"
#include "ngspice/smpdefs.h"
#include "ngspice/cktdefs.h"
#include "isrcdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

/*ARGSUSED*/
int
ISRCtemp(GENmodel *inModel, CKTcircuit *ckt)
        /* Pre-process voltage source parameters
         */
{
    ISRCmodel *model = (ISRCmodel *) inModel;
    ISRCinstance *here;
    double radians;

    NG_IGNORE(ckt);

    /*  loop through all the voltage source models */
    for( ; model != NULL; model = model->ISRCnextModel ) {

        /* loop through all the instances of the model */
        for (here = model->ISRCinstances; here != NULL ;
                here=here->ISRCnextInstance) {

            if(here->ISRCacGiven && !here->ISRCacMGiven) {
                here->ISRCacMag = 1;
            }
            if(here->ISRCacGiven && !here->ISRCacPGiven) {
                here->ISRCacPhase = 0;
            }
            if(!here->ISRCdcGiven) {
                /* no DC value - either have a transient value, or none */
                if(here->ISRCfuncTGiven) {
                    SPfrontEnd->IFerror (ERR_WARNING,
                            "%s: no DC value, transient time 0 value used",
                            &(here->ISRCname));
                } else {
                    SPfrontEnd->IFerror (ERR_WARNING,
                            "%s: has no value, DC 0 assumed",
                            &(here->ISRCname));
                }
            }
            radians = here->ISRCacPhase * M_PI / 180.0;
            here->ISRCacReal = here->ISRCacMag * cos(radians);
            here->ISRCacImag = here->ISRCacMag * sin(radians);
        }
    }

    return(OK);
}
