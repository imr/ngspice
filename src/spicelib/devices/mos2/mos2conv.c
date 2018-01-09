/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "mos2defs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

int
MOS2convTest(GENmodel *inModel, CKTcircuit *ckt)
{
    MOS2model *model = (MOS2model *)inModel;
    MOS2instance *here;
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

    for( ; model != NULL; model = MOS2nextModel(model)) {
        for(here = MOS2instances(model); here!= NULL;
                here = MOS2nextInstance(here)) {

            vbs = model->MOS2type * ( 
                *(ckt->CKTrhs+here->MOS2bNode) -
                *(ckt->CKTrhs+here->MOS2sNodePrime));
            vgs = model->MOS2type * ( 
                *(ckt->CKTrhs+here->MOS2gNode) -
                *(ckt->CKTrhs+here->MOS2sNodePrime));
            vds = model->MOS2type * ( 
                *(ckt->CKTrhs+here->MOS2dNodePrime) -
                *(ckt->CKTrhs+here->MOS2sNodePrime));
            vbd=vbs-vds;
            vgd=vgs-vds;
            vgdo = *(ckt->CKTstate0 + here->MOS2vgs) -
                *(ckt->CKTstate0 + here->MOS2vds);
            delvbs = vbs - *(ckt->CKTstate0 + here->MOS2vbs);
            delvbd = vbd - *(ckt->CKTstate0 + here->MOS2vbd);
            delvgs = vgs - *(ckt->CKTstate0 + here->MOS2vgs);
            delvds = vds - *(ckt->CKTstate0 + here->MOS2vds);
            delvgd = vgd-vgdo;

            /* these are needed for convergence testing */

            if (here->MOS2mode >= 0) {
                cdhat=
                    here->MOS2cd-
                    here->MOS2gbd * delvbd +
                    here->MOS2gmbs * delvbs +
                    here->MOS2gm * delvgs + 
                    here->MOS2gds * delvds ;
            } else {
                cdhat=
                    here->MOS2cd -
                    ( here->MOS2gbd -
                    here->MOS2gmbs) * delvbd -
                    here->MOS2gm * delvgd + 
                    here->MOS2gds * delvds ;
            }
            cbhat=
                here->MOS2cbs +
                here->MOS2cbd +
                here->MOS2gbd * delvbd +
                here->MOS2gbs * delvbs ;
            /*
             *  check convergence
             */
            tol=ckt->CKTreltol*MAX(fabs(cdhat),fabs(here->MOS2cd))+
                    ckt->CKTabstol;
            if (fabs(cdhat-here->MOS2cd) >= tol) { 
                ckt->CKTnoncon++;
		ckt->CKTtroubleElt = (GENinstance *) here;
                return(OK); /* no reason to continue, we haven't converged */
            } else {
                tol=ckt->CKTreltol*
                        MAX(fabs(cbhat),fabs(here->MOS2cbs+here->MOS2cbd))+ 
                        ckt->CKTabstol;
                if (fabs(cbhat-(here->MOS2cbs+here->MOS2cbd)) > tol) {
                    ckt->CKTnoncon++;
		    ckt->CKTtroubleElt = (GENinstance *) here;
                    return(OK); /* no reason to continue, we haven't converged*/
                }
            }
        }
    }
    return(OK);
}
