/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#include "ngspice.h"
#include "devdefs.h"
#include "cktdefs.h"
#include "diodefs.h"
#include "const.h"
#include "trandefs.h"
#include "sperror.h"
#include "suffix.h"

int
DIOconvTest(GENmodel *inModel, CKTcircuit *ckt)
        /* Check the devices for convergence
         */
{
    DIOmodel *model = (DIOmodel*)inModel;
    DIOinstance *here;
    double delvd,vd,cdhat,cd;
    double tol;
    /*  loop through all the diode models */
    for( ; model != NULL; model = model->DIOnextModel ) {

        /* loop through all the instances of the model */
        for (here = model->DIOinstances; here != NULL ;
                here=here->DIOnextInstance) {
	    if (here->DIOowner != ARCHme) continue;
                
            /*  
             *   initialization 
             */

            vd = *(ckt->CKTrhsOld+here->DIOposPrimeNode)-
                    *(ckt->CKTrhsOld + here->DIOnegNode);

            delvd=vd- *(ckt->CKTstate0 + here->DIOvoltage);
            cdhat= *(ckt->CKTstate0 + here->DIOcurrent) + 
                    *(ckt->CKTstate0 + here->DIOconduct) * delvd;

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
