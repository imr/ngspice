/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#include "ngspice/ngspice.h"
#include "ngspice/smpdefs.h"
#include "ngspice/cktdefs.h"
#include "vsrcdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

/*ARGSUSED*/
int
VSRCtemp(GENmodel *inModel, CKTcircuit *ckt)
        /* Pre-process voltage source parameters
         */
{
    VSRCmodel *model = (VSRCmodel *) inModel;
    VSRCinstance *here;
    double radians;

    NG_IGNORE(ckt);

    /*  loop through all the voltage source models */
    for( ; model != NULL; model = VSRCnextModel(model)) {

        /* loop through all the instances of the model */
        for (here = VSRCinstances(model); here != NULL ;
                here=VSRCnextInstance(here)) {

            if(here->VSRCacGiven && !here->VSRCacMGiven) {
                here->VSRCacMag = 1;
            }
            if(here->VSRCacGiven && !here->VSRCacPGiven) {
                here->VSRCacPhase = 0;
            }
            if(!here->VSRCdcGiven) {
                /* no DC value - either have a transient value, or none */
                if(here->VSRCfuncTGiven) {
                    SPfrontEnd->IFerrorf (ERR_WARNING,
                            "%s: no DC value, transient time 0 value used",
                            here->VSRCname);
                } else {
                    SPfrontEnd->IFerrorf (ERR_WARNING,
                            "%s: has no value, DC 0 assumed",
                            here->VSRCname);
                }
            }
            radians = here->VSRCacPhase * M_PI / 180.0;
            here->VSRCacReal = here->VSRCacMag * cos(radians);
            here->VSRCacImag = here->VSRCacMag * sin(radians);
        }
    }

    return(OK);
}
