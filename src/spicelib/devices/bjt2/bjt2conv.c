/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: Alan Gillespie
**********/

/*
 * This routine performs the device convergence test for
 * BJT2s in the circuit.
 */

#include "ngspice.h"
#include "cktdefs.h"
#include "bjt2defs.h"
#include "sperror.h"
#include "suffix.h"

int
BJT2convTest(GENmodel *inModel, CKTcircuit *ckt)
{
    BJT2instance *here;
    BJT2model *model = (BJT2model *) inModel;
    double tol;
    double cc;
    double cchat;
    double cb;
    double cbhat;
    double vbe;
    double vbc;
    double delvbe;
    double delvbc;



    for( ; model != NULL; model = model->BJT2nextModel) {
        for(here=model->BJT2instances;here!=NULL;here = here->BJT2nextInstance){
            if (here->BJT2owner != ARCHme) continue;
	    
	    vbe=model->BJT2type*(
                    *(ckt->CKTrhsOld+here->BJT2basePrimeNode)-
                    *(ckt->CKTrhsOld+here->BJT2emitPrimeNode));
            vbc=model->BJT2type*(
                    *(ckt->CKTrhsOld+here->BJT2basePrimeNode)-
                    *(ckt->CKTrhsOld+here->BJT2colPrimeNode));
            delvbe=vbe- *(ckt->CKTstate0 + here->BJT2vbe);
            delvbc=vbc- *(ckt->CKTstate0 + here->BJT2vbc);
            cchat= *(ckt->CKTstate0 + here->BJT2cc)+(*(ckt->CKTstate0 + 
                    here->BJT2gm)+ *(ckt->CKTstate0 + here->BJT2go))*delvbe-
                    (*(ckt->CKTstate0 + here->BJT2go)+*(ckt->CKTstate0 +
                    here->BJT2gmu))*delvbc;
            cbhat= *(ckt->CKTstate0 + here->BJT2cb)+ *(ckt->CKTstate0 + 
                    here->BJT2gpi)*delvbe+ *(ckt->CKTstate0 + here->BJT2gmu)*
                    delvbc;
            cc = *(ckt->CKTstate0 + here->BJT2cc);
            cb = *(ckt->CKTstate0 + here->BJT2cb);
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
