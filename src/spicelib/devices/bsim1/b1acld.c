/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Hong J. Park, Thomas L. Quarles
**********/
/*
 */

#include "ngspice.h"
#include <stdio.h>
#include "cktdefs.h"
#include "bsim1def.h"
#include "sperror.h"
#include "suffix.h"


int
B1acLoad(inModel,ckt)
    GENmodel *inModel;
    register CKTcircuit *ckt;
{
    register B1model *model = (B1model*)inModel;
    register B1instance *here;
    int xnrm;
    int xrev;
    double gdpr;
    double gspr;
    double gm;
    double gds;
    double gmbs;
    double gbd;
    double gbs;
    double capbd;
    double capbs;
    double xcggb;
    double xcgdb;
    double xcgsb;
    double xcbgb;
    double xcbdb;
    double xcbsb;
    double xcddb;
    double xcssb;
    double xcdgb;
    double xcsgb;
    double xcdsb;
    double xcsdb;
    double cggb;
    double cgdb;
    double cgsb;
    double cbgb;
    double cbdb;
    double cbsb;
    double cddb;
    double cdgb;
    double cdsb;
    double omega; /* angular fequency of the signal */

    omega = ckt->CKTomega;
    for( ; model != NULL; model = model->B1nextModel) {
        for(here = model->B1instances; here!= NULL;
                here = here->B1nextInstance) {
	    if (here->B1owner != ARCHme) continue;
        
            if (here->B1mode >= 0) {
                xnrm=1;
                xrev=0;
            } else {
                xnrm=0;
                xrev=1;
            }
            gdpr=here->B1drainConductance;
            gspr=here->B1sourceConductance;
            gm= *(ckt->CKTstate0 + here->B1gm);
            gds= *(ckt->CKTstate0 + here->B1gds);
            gmbs= *(ckt->CKTstate0 + here->B1gmbs);
            gbd= *(ckt->CKTstate0 + here->B1gbd);
            gbs= *(ckt->CKTstate0 + here->B1gbs);
            capbd= *(ckt->CKTstate0 + here->B1capbd);
            capbs= *(ckt->CKTstate0 + here->B1capbs);
            /*
             *    charge oriented model parameters
             */

            cggb = *(ckt->CKTstate0 + here->B1cggb);
            cgsb = *(ckt->CKTstate0 + here->B1cgsb);
            cgdb = *(ckt->CKTstate0 + here->B1cgdb);

            cbgb = *(ckt->CKTstate0 + here->B1cbgb);
            cbsb = *(ckt->CKTstate0 + here->B1cbsb);
            cbdb = *(ckt->CKTstate0 + here->B1cbdb);

            cdgb = *(ckt->CKTstate0 + here->B1cdgb);
            cdsb = *(ckt->CKTstate0 + here->B1cdsb);
            cddb = *(ckt->CKTstate0 + here->B1cddb);

            xcdgb = (cdgb - here->B1GDoverlapCap) * omega;
            xcddb = (cddb + capbd + here->B1GDoverlapCap) * omega;
            xcdsb = cdsb * omega;
            xcsgb = -(cggb + cbgb + cdgb + here->B1GSoverlapCap ) * omega;
            xcsdb = -(cgdb + cbdb + cddb) * omega;
            xcssb = (capbs + here->B1GSoverlapCap - (cgsb+cbsb+cdsb)) * omega;
            xcggb = (cggb + here->B1GDoverlapCap + here->B1GSoverlapCap + 
                    here->B1GBoverlapCap) * omega;
            xcgdb = (cgdb - here->B1GDoverlapCap ) * omega;
            xcgsb = (cgsb - here->B1GSoverlapCap) * omega;
            xcbgb = (cbgb - here->B1GBoverlapCap) * omega;
            xcbdb = (cbdb - capbd ) * omega;
            xcbsb = (cbsb - capbs ) * omega;


            *(here->B1GgPtr +1) += xcggb;
            *(here->B1BbPtr +1) += -xcbgb-xcbdb-xcbsb;
            *(here->B1DPdpPtr +1) += xcddb;
            *(here->B1SPspPtr +1) += xcssb;
            *(here->B1GbPtr +1) += -xcggb-xcgdb-xcgsb;
            *(here->B1GdpPtr +1) += xcgdb;
            *(here->B1GspPtr +1) += xcgsb;
            *(here->B1BgPtr +1) += xcbgb;
            *(here->B1BdpPtr +1) += xcbdb;
            *(here->B1BspPtr +1) += xcbsb;
            *(here->B1DPgPtr +1) += xcdgb;
            *(here->B1DPbPtr +1) += -xcdgb-xcddb-xcdsb;
            *(here->B1DPspPtr +1) += xcdsb;
            *(here->B1SPgPtr +1) += xcsgb;
            *(here->B1SPbPtr +1) += -xcsgb-xcsdb-xcssb;
            *(here->B1SPdpPtr +1) += xcsdb;
            *(here->B1DdPtr) += gdpr;
            *(here->B1SsPtr) += gspr;
            *(here->B1BbPtr) += gbd+gbs;
            *(here->B1DPdpPtr) += gdpr+gds+gbd+xrev*(gm+gmbs);
            *(here->B1SPspPtr) += gspr+gds+gbs+xnrm*(gm+gmbs);
            *(here->B1DdpPtr) -= gdpr;
            *(here->B1SspPtr) -= gspr;
            *(here->B1BdpPtr) -= gbd;
            *(here->B1BspPtr) -= gbs;
            *(here->B1DPdPtr) -= gdpr;
            *(here->B1DPgPtr) += (xnrm-xrev)*gm;
            *(here->B1DPbPtr) += -gbd+(xnrm-xrev)*gmbs;
            *(here->B1DPspPtr) += -gds-xnrm*(gm+gmbs);
            *(here->B1SPgPtr) += -(xnrm-xrev)*gm;
            *(here->B1SPsPtr) -= gspr;
            *(here->B1SPbPtr) += -gbs-(xnrm-xrev)*gmbs;
            *(here->B1SPdpPtr) += -gds-xrev*(gm+gmbs);

        }
    }
return(OK);
}


