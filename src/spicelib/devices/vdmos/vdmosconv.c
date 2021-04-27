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
    double delvgs;
    double delvds;
    double delvgd;
    double cdhat;
    double vgs;
    double vds;
    double vgd;
    double vgdo;
    double tol;
    double delvd,vd,cd;
    int selfheat;
    double delTemp, deldelTemp;

    for( ; model != NULL; model = VDMOSnextModel(model)) {
        for(here = VDMOSinstances(model); here!= NULL;
                here = VDMOSnextInstance(here)) {
        
            vgs = model->VDMOStype * ( 
                *(ckt->CKTrhs+here->VDMOSgNode) -
                *(ckt->CKTrhs+here->VDMOSsNodePrime));
            vds = model->VDMOStype * ( 
                *(ckt->CKTrhs+here->VDMOSdNodePrime) -
                *(ckt->CKTrhs+here->VDMOSsNodePrime));
            vgd=vgs-vds;
            vgdo = *(ckt->CKTstate0 + here->VDMOSvgs) -
                *(ckt->CKTstate0 + here->VDMOSvds);
            delvgs = vgs - *(ckt->CKTstate0 + here->VDMOSvgs);
            delvds = vds - *(ckt->CKTstate0 + here->VDMOSvds);
            delvgd = vgd-vgdo;

            selfheat = (here->VDMOSthermal) && (model->VDMOSrthjcGiven);
            if (selfheat) {
                delTemp = *(ckt->CKTrhs + here->VDMOStempNode);
                deldelTemp = delTemp - *(ckt->CKTstate0 + here->VDMOSdelTemp);
            } else {
                deldelTemp = 0.0;
            }

            /* these are needed for convergence testing */

            if (here->VDMOSmode >= 0) {
                cdhat=
                    here->VDMOScd -
                    here->VDMOSgm * delvgs + 
                    here->VDMOSgds * delvds +
                    here->VDMOSgmT * deldelTemp;
            } else {
                cdhat=
                    here->VDMOScd -
                    here->VDMOSgm * delvgd + 
                    here->VDMOSgds * delvds +
                    here->VDMOSgmT * deldelTemp;
            }
            /*
             *  check convergence
             */
            tol=ckt->CKTreltol*MAX(fabs(cdhat),fabs(here->VDMOScd))+
                    ckt->CKTabstol;
            if (fabs(cdhat-here->VDMOScd) >= tol) { 
                ckt->CKTnoncon++;
                ckt->CKTtroubleElt = (GENinstance *) here;
                return(OK); /* no reason to continue, we haven't converged */
            }

            /*  
             *   initialization 
             */

            vd = *(ckt->CKTrhsOld+here->VDIOposPrimeNode)-
                    *(ckt->CKTrhsOld + here->VDMOSdNode);

            delvd=vd- *(ckt->CKTstate0 + here->VDIOvoltage);

            cdhat= *(ckt->CKTstate0 + here->VDIOcurrent) +
                    *(ckt->CKTstate0 + here->VDIOconduct) * delvd +
                    *(ckt->CKTstate0 + here->VDIOdIdio_dT) * deldelTemp;

            cd= *(ckt->CKTstate0 + here->VDIOcurrent);

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
