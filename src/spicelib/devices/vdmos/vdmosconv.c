/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "vdmosdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

int
VDMOSconvTest(GENmodel *inModel, CKTcircuit *ckt)
{
    VDMOSmodel *model = (VDMOSmodel*)inModel;
    VDMOSinstance *here;
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

    for( ; model != NULL; model = VDMOSnextModel(model)) {
        for(here = VDMOSinstances(model); here!= NULL;
                here = VDMOSnextInstance(here)) {
        
            vbs = model->VDMOStype * ( 
                *(ckt->CKTrhs+here->VDMOSbNode) -
                *(ckt->CKTrhs+here->VDMOSsNodePrime));
            vgs = model->VDMOStype * ( 
                *(ckt->CKTrhs+here->VDMOSgNode) -
                *(ckt->CKTrhs+here->VDMOSsNodePrime));
            vds = model->VDMOStype * ( 
                *(ckt->CKTrhs+here->VDMOSdNodePrime) -
                *(ckt->CKTrhs+here->VDMOSsNodePrime));
            vbd=vbs-vds;
            vgd=vgs-vds;
            vgdo = *(ckt->CKTstate0 + here->VDMOSvgs) -
                *(ckt->CKTstate0 + here->VDMOSvds);
            delvbs = vbs - *(ckt->CKTstate0 + here->VDMOSvbs);
            delvbd = vbd - *(ckt->CKTstate0 + here->VDMOSvbd);
            delvgs = vgs - *(ckt->CKTstate0 + here->VDMOSvgs);
            delvds = vds - *(ckt->CKTstate0 + here->VDMOSvds);
            delvgd = vgd-vgdo;

            /* these are needed for convergence testing */

            if (here->VDMOSmode >= 0) {
                cdhat=
                    here->VDMOScd-
                    here->VDMOSgbd * delvbd +
                    here->VDMOSgmbs * delvbs +
                    here->VDMOSgm * delvgs + 
                    here->VDMOSgds * delvds ;
            } else {
                cdhat=
                    here->VDMOScd -
                    ( here->VDMOSgbd -
                    here->VDMOSgmbs) * delvbd -
                    here->VDMOSgm * delvgd + 
                    here->VDMOSgds * delvds ;
            }
            cbhat=
                here->VDMOScbs +
                here->VDMOScbd +
                here->VDMOSgbd * delvbd +
                here->VDMOSgbs * delvbs ;
            /*
             *  check convergence
             */
            tol=ckt->CKTreltol*MAX(fabs(cdhat),fabs(here->VDMOScd))+
                    ckt->CKTabstol;
            if (fabs(cdhat-here->VDMOScd) >= tol) { 
                ckt->CKTnoncon++;
		ckt->CKTtroubleElt = (GENinstance *) here;
                return(OK); /* no reason to continue, we haven't converged */
            } else {
                tol=ckt->CKTreltol*
                        MAX(fabs(cbhat),fabs(here->VDMOScbs+here->VDMOScbd))+
                        ckt->CKTabstol;
                if (fabs(cbhat-(here->VDMOScbs+here->VDMOScbd)) > tol) {
                    ckt->CKTnoncon++;
		    ckt->CKTtroubleElt = (GENinstance *) here;
                    return(OK); /* no reason to continue, we haven't converged*/
                }
            }
        }
    }
    return(OK);
}
