/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#include "ngspice.h"
#include "smpdefs.h"
#include "cktdefs.h"
#include "vsrcdefs.h"
#include "sperror.h"
#include "suffix.h"

/* ARGSUSED */
int
VSRCtemp(GENmodel *inModel, CKTcircuit *ckt)
        /* Pre-process voltage source parameters 
         */
{
    VSRCmodel *model = (VSRCmodel *)inModel;
    VSRCinstance *here;
    double radians;

    /*  loop through all the voltage source models */
    for( ; model != NULL; model = model->VSRCnextModel ) {

        /* loop through all the instances of the model */
        for (here = model->VSRCinstances; here != NULL ;
                here=here->VSRCnextInstance) {
	    if (here->VSRCowner != ARCHme) continue;
            
            if(here->VSRCacGiven && !here->VSRCacMGiven) {
                here->VSRCacMag = 1;
            }
            if(here->VSRCacGiven && !here->VSRCacPGiven) {
                here->VSRCacPhase = 0;
            }
            if(!here->VSRCdcGiven) {
                /* no DC value - either have a transient value, or none */
                if(here->VSRCfuncTGiven) {
                    (*(SPfrontEnd->IFerror))(ERR_WARNING,
                            "%s: no DC value, transient time 0 value used",
                            &(here->VSRCname));
                } else {
                    (*(SPfrontEnd->IFerror))(ERR_WARNING,
                            "%s: has no value, DC 0 assumed",
                            &(here->VSRCname));
                }
            }
            radians = here->VSRCacPhase * M_PI / 180.0;
            here->VSRCacReal = here->VSRCacMag * cos(radians);
            here->VSRCacImag = here->VSRCacMag * sin(radians);

        }
    }
    return(OK);
}
