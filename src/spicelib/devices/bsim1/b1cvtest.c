/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Hong J. Park, Thomas L. Quarles
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "bsim1def.h"
#include "ngspice/trandefs.h"
#include "ngspice/const.h"
#include "ngspice/devdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

int
B1convTest(GENmodel *inModel, CKTcircuit *ckt)

        /* actually load the current value into the 
         * sparse matrix previously provided 
         */
{
    B1model *model = (B1model*)inModel;
    B1instance *here;
    double cbd;
    double cbhat;
    double cbs;
    double cd;
    double cdhat;
    double delvbd;
    double delvbs;
    double delvds;
    double delvgd;
    double delvgs;
    double tol;
    double vbd;
    double vbs;
    double vds;
    double vgd;
    double vgdo;
    double vgs;


    /*  loop through all the B1 device models */
    for( ; model != NULL; model = B1nextModel(model)) {

        /* loop through all the instances of the model */
        for (here = B1instances(model); here != NULL ;
                here=B1nextInstance(here)) {

            vbs = model->B1type * ( 
                *(ckt->CKTrhsOld+here->B1bNode) -
                *(ckt->CKTrhsOld+here->B1sNodePrime));
            vgs = model->B1type * ( 
                *(ckt->CKTrhsOld+here->B1gNode) -
                *(ckt->CKTrhsOld+here->B1sNodePrime));
            vds = model->B1type * ( 
                *(ckt->CKTrhsOld+here->B1dNodePrime) -
                *(ckt->CKTrhsOld+here->B1sNodePrime));
            vbd=vbs-vds;
            vgd=vgs-vds;
            vgdo = *(ckt->CKTstate0 + here->B1vgs) - 
                *(ckt->CKTstate0 + here->B1vds);
            delvbs = vbs - *(ckt->CKTstate0 + here->B1vbs);
            delvbd = vbd - *(ckt->CKTstate0 + here->B1vbd);
            delvgs = vgs - *(ckt->CKTstate0 + here->B1vgs);
            delvds = vds - *(ckt->CKTstate0 + here->B1vds);
            delvgd = vgd-vgdo;

            if (here->B1mode >= 0) {
                cdhat=
                    *(ckt->CKTstate0 + here->B1cd) -
                    *(ckt->CKTstate0 + here->B1gbd) * delvbd +
                    *(ckt->CKTstate0 + here->B1gmbs) * delvbs +
                    *(ckt->CKTstate0 + here->B1gm) * delvgs + 
                    *(ckt->CKTstate0 + here->B1gds) * delvds ;
            } else {
                cdhat=
                    *(ckt->CKTstate0 + here->B1cd) -
                    ( *(ckt->CKTstate0 + here->B1gbd) -
                      *(ckt->CKTstate0 + here->B1gmbs)) * delvbd -
                    *(ckt->CKTstate0 + here->B1gm) * delvgd +
                    *(ckt->CKTstate0 + here->B1gds) * delvds;
            }
            cbhat=
                *(ckt->CKTstate0 + here->B1cbs) +
                *(ckt->CKTstate0 + here->B1cbd) +
                *(ckt->CKTstate0 + here->B1gbd) * delvbd +
                *(ckt->CKTstate0 + here->B1gbs) * delvbs ;

            cd = *(ckt->CKTstate0 + here->B1cd);
            cbs = *(ckt->CKTstate0 + here->B1cbs);
            cbd = *(ckt->CKTstate0 + here->B1cbd);
            /*
             *  check convergence
             */
            if ( (here->B1off == 0)  || (!(ckt->CKTmode & MODEINITFIX)) ){
                tol=ckt->CKTreltol*MAX(fabs(cdhat),fabs(cd))+ckt->CKTabstol;
                if (fabs(cdhat-cd) >= tol) { 
                    ckt->CKTnoncon++;
		    ckt->CKTtroubleElt = (GENinstance *) here;
                    return(OK);
                } 
                tol=ckt->CKTreltol*MAX(fabs(cbhat),fabs(cbs+cbd))+
                    ckt->CKTabstol;
                if (fabs(cbhat-(cbs+cbd)) > tol) {
                    ckt->CKTnoncon++;
		    ckt->CKTtroubleElt = (GENinstance *) here;
                    return(OK);
                }
            }
        }
    }
    return(OK);
}

