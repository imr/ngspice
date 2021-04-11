/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#include "ngspice/ngspice.h"
#include "ngspice/devdefs.h"
#include "ngspice/cktdefs.h"
#include "diodefs.h"
#include "ngspice/const.h"
#include "ngspice/trandefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

int
DIOconvTest(GENmodel *inModel, CKTcircuit *ckt)
        /* Check the devices for convergence
         */
{
    DIOmodel *model = (DIOmodel*)inModel;
    DIOinstance *here;
    double delvd,vd,cdhat,cd;
    double tol;
    double delTemp, deldelTemp;
    /*  loop through all the diode models */
    for( ; model != NULL; model = DIOnextModel(model)) {

        /* loop through all the instances of the model */
        for (here = DIOinstances(model); here != NULL ;
                here=DIOnextInstance(here)) {
                
            /*  
             *   initialization 
             */

            vd = *(ckt->CKTrhsOld+here->DIOposPrimeNode)-
                    *(ckt->CKTrhsOld + here->DIOnegNode);

            delvd=vd- *(ckt->CKTstate0 + here->DIOvoltage);

            int selfheat = ((here->DIOtempNode > 0) && (here->DIOthermal) && (model->DIOrth0Given));
            if (selfheat)
                delTemp = *(ckt->CKTrhsOld + here->DIOtempNode);
            else
                delTemp = 0.0;
            deldelTemp = delTemp - *(ckt->CKTstate0 + here->DIOdeltemp);

            cdhat= *(ckt->CKTstate0 + here->DIOcurrent) + 
                    *(ckt->CKTstate0 + here->DIOconduct) * delvd +
                    *(ckt->CKTstate0 + here->DIOdIdio_dT) * deldelTemp;

            cd= *(ckt->CKTstate0 + here->DIOcurrent);

            /*
             *   check convergence
             */
            tol=ckt->CKTreltol*
                    MAX(fabs(cdhat),fabs(cd))+ckt->CKTabstol;
            if (fabs(cdhat-cd) > tol) {
                ckt->CKTnoncon++;
                ckt->CKTtroubleElt = (GENinstance *) here;
                return(OK); /* don't need to check any more device */
            }
        }
    }
    return(OK);
}
