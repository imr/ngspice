/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

/*
 * This routine performs the device convergence test for
 * BJTs in the circuit.
 */

#include "ngspice.h"
#include "cktdefs.h"
#include "bjtdefs.h"
#include "sperror.h"
#include "suffix.h"

int
BJTconvTest(GENmodel *inModel, CKTcircuit *ckt)

{
    BJTinstance *here;
    BJTmodel *model = (BJTmodel *) inModel;
    double tol;
    double cc;
    double cchat;
    double cb;
    double cbhat;
    double vbe;
    double vbc;
    double delvbe;
    double delvbc;



    for( ; model != NULL; model = model->BJTnextModel) {
        for(here=model->BJTinstances;here!=NULL;here = here->BJTnextInstance){
	    if (here->BJTowner != ARCHme) continue;

            vbe=model->BJTtype*(
                    *(ckt->CKTrhsOld+here->BJTbasePrimeNode)-
                    *(ckt->CKTrhsOld+here->BJTemitPrimeNode));
            vbc=model->BJTtype*(
                    *(ckt->CKTrhsOld+here->BJTbasePrimeNode)-
                    *(ckt->CKTrhsOld+here->BJTcolPrimeNode));
            delvbe=vbe- *(ckt->CKTstate0 + here->BJTvbe);
            delvbc=vbc- *(ckt->CKTstate0 + here->BJTvbc);
            cchat= *(ckt->CKTstate0 + here->BJTcc)+(*(ckt->CKTstate0 + 
                    here->BJTgm)+ *(ckt->CKTstate0 + here->BJTgo))*delvbe-
                    (*(ckt->CKTstate0 + here->BJTgo)+*(ckt->CKTstate0 +
                    here->BJTgmu))*delvbc;
            cbhat= *(ckt->CKTstate0 + here->BJTcb)+ *(ckt->CKTstate0 + 
                    here->BJTgpi)*delvbe+ *(ckt->CKTstate0 + here->BJTgmu)*
                    delvbc;
            cc = *(ckt->CKTstate0 + here->BJTcc);
            cb = *(ckt->CKTstate0 + here->BJTcb);
            /*
             *   check convergence
             */
            tol=ckt->CKTreltol*MAX(fabs(cchat),fabs(cc))+ckt->CKTabstol;
            if (fabs(cchat-cc) > tol) {
                ckt->CKTnoncon++;
		ckt->CKTtroubleElt = (GENinstance *) here;
                return(OK); /* no reason to continue - we've failed... */
            } else {
                tol=ckt->CKTreltol*MAX(fabs(cbhat),fabs(cb))+
                    ckt->CKTabstol;
                if (fabs(cbhat-cb) > tol) {
                    ckt->CKTnoncon++;
		    ckt->CKTtroubleElt = (GENinstance *) here;
                    return(OK); /* no reason to continue - we've failed... */
                }
            }
        }
    }
    return(OK);
}
