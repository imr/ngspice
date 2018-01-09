/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1989 Takayasu Sakurai
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "mos6defs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

int
MOS6convTest(GENmodel *inModel, CKTcircuit *ckt)
{
    MOS6model *model = (MOS6model*)inModel;
    MOS6instance *here;
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

    for( ; model != NULL; model = MOS6nextModel(model)) {
        for(here = MOS6instances(model); here!= NULL;
                here = MOS6nextInstance(here)) {
        
            vbs = model->MOS6type * ( 
                *(ckt->CKTrhs+here->MOS6bNode) -
                *(ckt->CKTrhs+here->MOS6sNodePrime));
            vgs = model->MOS6type * ( 
                *(ckt->CKTrhs+here->MOS6gNode) -
                *(ckt->CKTrhs+here->MOS6sNodePrime));
            vds = model->MOS6type * ( 
                *(ckt->CKTrhs+here->MOS6dNodePrime) -
                *(ckt->CKTrhs+here->MOS6sNodePrime));
            vbd=vbs-vds;
            vgd=vgs-vds;
            vgdo = *(ckt->CKTstate0 + here->MOS6vgs) -
                *(ckt->CKTstate0 + here->MOS6vds);
            delvbs = vbs - *(ckt->CKTstate0 + here->MOS6vbs);
            delvbd = vbd - *(ckt->CKTstate0 + here->MOS6vbd);
            delvgs = vgs - *(ckt->CKTstate0 + here->MOS6vgs);
            delvds = vds - *(ckt->CKTstate0 + here->MOS6vds);
            delvgd = vgd-vgdo;

            /* these are needed for convergence testing */

            if (here->MOS6mode >= 0) {
                cdhat=
                    here->MOS6cd-
                    here->MOS6gbd * delvbd +
                    here->MOS6gmbs * delvbs +
                    here->MOS6gm * delvgs + 
                    here->MOS6gds * delvds ;
            } else {
                cdhat=
                    here->MOS6cd -
                    ( here->MOS6gbd -
                    here->MOS6gmbs) * delvbd -
                    here->MOS6gm * delvgd + 
                    here->MOS6gds * delvds ;
            }
            cbhat=
                here->MOS6cbs +
                here->MOS6cbd +
                here->MOS6gbd * delvbd +
                here->MOS6gbs * delvbs ;
            /*
             *  check convergence
             */
            tol=ckt->CKTreltol*MAX(fabs(cdhat),fabs(here->MOS6cd))+
                    ckt->CKTabstol;
            if (fabs(cdhat-here->MOS6cd) >= tol) { 
                ckt->CKTnoncon++;
		ckt->CKTtroubleElt = (GENinstance *) here;
                return(OK); /* no reason to continue, we haven't converged */
            } else {
                tol=ckt->CKTreltol*
                        MAX(fabs(cbhat),fabs(here->MOS6cbs+here->MOS6cbd))+
                        ckt->CKTabstol;
                if (fabs(cbhat-(here->MOS6cbs+here->MOS6cbd)) > tol) {
                    ckt->CKTnoncon++;
		    ckt->CKTtroubleElt = (GENinstance *) here;
                    return(OK); /* no reason to continue, we haven't converged*/
                }
            }
        }
    }
    return(OK);
}
