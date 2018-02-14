/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "mos1defs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

int
MOS1convTest(GENmodel *inModel, CKTcircuit *ckt)
{
    MOS1model *model = (MOS1model*)inModel;
    MOS1instance *here;
    double delvbs;
    double delvbd;
    double delvgs;
    double delvds;
    double delvgd;
    double cbhat;
    double cdhat;
    double vbs;
    double vbd;
    double vgs;
    double vds;
    double vgd;
    double vgdo;
    double tol;

    for( ; model != NULL; model = MOS1nextModel(model)) {
        for(here = MOS1instances(model); here!= NULL;
                here = MOS1nextInstance(here)) {
        
            vbs = model->MOS1type * ( 
                *(ckt->CKTrhs+here->MOS1bNode) -
                *(ckt->CKTrhs+here->MOS1sNodePrime));
            vgs = model->MOS1type * ( 
                *(ckt->CKTrhs+here->MOS1gNode) -
                *(ckt->CKTrhs+here->MOS1sNodePrime));
            vds = model->MOS1type * ( 
                *(ckt->CKTrhs+here->MOS1dNodePrime) -
                *(ckt->CKTrhs+here->MOS1sNodePrime));
            vbd=vbs-vds;
            vgd=vgs-vds;
            vgdo = *(ckt->CKTstate0 + here->MOS1vgs) -
                *(ckt->CKTstate0 + here->MOS1vds);
            delvbs = vbs - *(ckt->CKTstate0 + here->MOS1vbs);
            delvbd = vbd - *(ckt->CKTstate0 + here->MOS1vbd);
            delvgs = vgs - *(ckt->CKTstate0 + here->MOS1vgs);
            delvds = vds - *(ckt->CKTstate0 + here->MOS1vds);
            delvgd = vgd-vgdo;

            /* these are needed for convergence testing */

            if (here->MOS1mode >= 0) {
                cdhat=
                    here->MOS1cd-
                    here->MOS1gbd * delvbd +
                    here->MOS1gmbs * delvbs +
                    here->MOS1gm * delvgs + 
                    here->MOS1gds * delvds ;
            } else {
                cdhat=
                    here->MOS1cd -
                    ( here->MOS1gbd -
                    here->MOS1gmbs) * delvbd -
                    here->MOS1gm * delvgd + 
                    here->MOS1gds * delvds ;
            }
            cbhat=
                here->MOS1cbs +
                here->MOS1cbd +
                here->MOS1gbd * delvbd +
                here->MOS1gbs * delvbs ;
            /*
             *  check convergence
             */
            tol=ckt->CKTreltol*MAX(fabs(cdhat),fabs(here->MOS1cd))+
                    ckt->CKTabstol;
            if (fabs(cdhat-here->MOS1cd) >= tol) { 
                ckt->CKTnoncon++;
		ckt->CKTtroubleElt = (GENinstance *) here;
                return(OK); /* no reason to continue, we haven't converged */
            } else {
                tol=ckt->CKTreltol*
                        MAX(fabs(cbhat),fabs(here->MOS1cbs+here->MOS1cbd))+
                        ckt->CKTabstol;
                if (fabs(cbhat-(here->MOS1cbs+here->MOS1cbd)) > tol) {
                    ckt->CKTnoncon++;
		    ckt->CKTtroubleElt = (GENinstance *) here;
                    return(OK); /* no reason to continue, we haven't converged*/
                }
            }
        }
    }
    return(OK);
}
