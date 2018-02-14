/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Hong J. Park, Thomas L. Quarles
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "bsim2def.h"
#include "ngspice/trandefs.h"
#include "ngspice/const.h"
#include "ngspice/devdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

int
B2convTest(GENmodel *inModel, CKTcircuit *ckt)

        /* actually load the current value into the 
         * sparse matrix previously provided 
         */
{
    B2model *model = (B2model*)inModel;
    B2instance *here;
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


    /*  loop through all the B2 device models */
    for( ; model != NULL; model = B2nextModel(model)) {

        /* loop through all the instances of the model */
        for (here = B2instances(model); here != NULL ;
                here=B2nextInstance(here)) {

            vbs = model->B2type * ( 
                *(ckt->CKTrhsOld+here->B2bNode) -
                *(ckt->CKTrhsOld+here->B2sNodePrime));
            vgs = model->B2type * ( 
                *(ckt->CKTrhsOld+here->B2gNode) -
                *(ckt->CKTrhsOld+here->B2sNodePrime));
            vds = model->B2type * ( 
                *(ckt->CKTrhsOld+here->B2dNodePrime) -
                *(ckt->CKTrhsOld+here->B2sNodePrime));
            vbd=vbs-vds;
            vgd=vgs-vds;
            vgdo = *(ckt->CKTstate0 + here->B2vgs) - 
                *(ckt->CKTstate0 + here->B2vds);
            delvbs = vbs - *(ckt->CKTstate0 + here->B2vbs);
            delvbd = vbd - *(ckt->CKTstate0 + here->B2vbd);
            delvgs = vgs - *(ckt->CKTstate0 + here->B2vgs);
            delvds = vds - *(ckt->CKTstate0 + here->B2vds);
            delvgd = vgd-vgdo;

            if (here->B2mode >= 0) {
                cdhat=
                    *(ckt->CKTstate0 + here->B2cd) -
                    *(ckt->CKTstate0 + here->B2gbd) * delvbd +
                    *(ckt->CKTstate0 + here->B2gmbs) * delvbs +
                    *(ckt->CKTstate0 + here->B2gm) * delvgs + 
                    *(ckt->CKTstate0 + here->B2gds) * delvds ;
            } else {
                cdhat=
                    *(ckt->CKTstate0 + here->B2cd) -
                    ( *(ckt->CKTstate0 + here->B2gbd) -
                      *(ckt->CKTstate0 + here->B2gmbs)) * delvbd -
                    *(ckt->CKTstate0 + here->B2gm) * delvgd +
                    *(ckt->CKTstate0 + here->B2gds) * delvds;
            }
            cbhat=
                *(ckt->CKTstate0 + here->B2cbs) +
                *(ckt->CKTstate0 + here->B2cbd) +
                *(ckt->CKTstate0 + here->B2gbd) * delvbd +
                *(ckt->CKTstate0 + here->B2gbs) * delvbs ;

            cd = *(ckt->CKTstate0 + here->B2cd);
            cbs = *(ckt->CKTstate0 + here->B2cbs);
            cbd = *(ckt->CKTstate0 + here->B2cbd);
            /*
             *  check convergence
             */
            if ( (here->B2off == 0)  || (!(ckt->CKTmode & MODEINITFIX)) ){
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

