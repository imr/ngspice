/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "asrcdefs.h"
#include "ngspice/sperror.h"

int
ASRCtemp(GENmodel *inModel, CKTcircuit *ckt)
{
    ASRCmodel *model =  (ASRCmodel *)inModel;
    ASRCinstance *here;

    /*  loop through all the source models */
    for( ; model != NULL; model = model->ASRCnextModel ) {

        /* loop through all the instances of the model */
        for (here = model->ASRCinstances; here != NULL ;
                here=here->ASRCnextInstance) {

            /* Default Value Processing for Source Instance */

          if(!here->ASRCtempGiven) {
             here->ASRCtemp   = ckt->CKTtemp;
             if(!here->ASRCdtempGiven) here->ASRCdtemp  = 0.0;
           } else { /* ASRCtempGiven */
             here->ASRCdtemp = 0.0;
             if (here->ASRCdtempGiven)
                 printf("%s: Instance temperature specified, dtemp ignored\n", here->ASRCname);
           }

        }
    }
    return(OK);
}
