/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: Alan Gillespie
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "mos9defs.h"
#include "ngspice/sperror.h"

#include "ngspice/suffix.h"

int
MOS9convTest(GENmodel *inModel, CKTcircuit *ckt)
{
    MOS9model *model = (MOS9model *)inModel;
    MOS9instance *here;
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

    for( ; model != NULL; model = MOS9nextModel(model)) {
        for(here = MOS9instances(model); here!= NULL;
                here = MOS9nextInstance(here)) {
        
            vbs = model->MOS9type * ( 
                *(ckt->CKTrhs+here->MOS9bNode) -
                *(ckt->CKTrhs+here->MOS9sNodePrime));
            vgs = model->MOS9type * ( 
                *(ckt->CKTrhs+here->MOS9gNode) -
                *(ckt->CKTrhs+here->MOS9sNodePrime));
            vds = model->MOS9type * ( 
                *(ckt->CKTrhs+here->MOS9dNodePrime) -
                *(ckt->CKTrhs+here->MOS9sNodePrime));
            vbd=vbs-vds;
            vgd=vgs-vds;
            vgdo = *(ckt->CKTstate0 + here->MOS9vgs) -
                *(ckt->CKTstate0 + here->MOS9vds);
            delvbs = vbs - *(ckt->CKTstate0 + here->MOS9vbs);
            delvbd = vbd - *(ckt->CKTstate0 + here->MOS9vbd);
            delvgs = vgs - *(ckt->CKTstate0 + here->MOS9vgs);
            delvds = vds - *(ckt->CKTstate0 + here->MOS9vds);
            delvgd = vgd-vgdo;

            /* these are needed for convergence testing */

            if (here->MOS9mode >= 0) {
                cdhat=
                    here->MOS9cd-
                    here->MOS9gbd * delvbd +
                    here->MOS9gmbs * delvbs +
                    here->MOS9gm * delvgs + 
                    here->MOS9gds * delvds ;
            } else {
                cdhat=
                    here->MOS9cd -
                    ( here->MOS9gbd -
                    here->MOS9gmbs) * delvbd -
                    here->MOS9gm * delvgd + 
                    here->MOS9gds * delvds ;
            }
            cbhat=
                here->MOS9cbs +
                here->MOS9cbd +
                here->MOS9gbd * delvbd +
                here->MOS9gbs * delvbs ;
            /*
             *  check convergence
             */
            tol=ckt->CKTreltol*MAX(fabs(cdhat),fabs(here->MOS9cd))+
                    ckt->CKTabstol;
            if (fabs(cdhat-here->MOS9cd) >= tol) { 
                ckt->CKTnoncon++;
		ckt->CKTtroubleElt = (GENinstance *) here;
                return(OK); /* no reason to continue, we haven't converged */
            } else {
                tol=ckt->CKTreltol*
                        MAX(fabs(cbhat),fabs(here->MOS9cbs+here->MOS9cbd))
                        + ckt->CKTabstol;
                if (fabs(cbhat-(here->MOS9cbs+here->MOS9cbd)) > tol) {
                    ckt->CKTnoncon++;
		    ckt->CKTtroubleElt = (GENinstance *) here;
                    return(OK); /* no reason to continue, we haven't converged*/
                }
            }
        }
    }
    return(OK);
}
