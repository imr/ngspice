/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ngspice/complex.h"
#include "ngspice/sperror.h"
#include "bsim1def.h"
#include "ngspice/suffix.h"


int
B1pzLoad(GENmodel *inModel, CKTcircuit *ckt, SPcomplex *s)
{
    B1model *model = (B1model*)inModel;
    B1instance *here;
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

    double m;

    for( ; model != NULL; model = B1nextModel(model)) {
        for(here = B1instances(model); here!= NULL;
                here = B1nextInstance(here)) {
        
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

            xcdgb = (cdgb - here->B1GDoverlapCap) ;
            xcddb = (cddb + capbd + here->B1GDoverlapCap) ;
            xcdsb = cdsb ;
            xcsgb = -(cggb + cbgb + cdgb + here->B1GSoverlapCap ) ;
            xcsdb = -(cgdb + cbdb + cddb) ;
            xcssb = (capbs + here->B1GSoverlapCap - (cgsb+cbsb+cdsb)) ;
            xcggb = (cggb + here->B1GDoverlapCap + here->B1GSoverlapCap + 
                    here->B1GBoverlapCap) ;
            xcgdb = (cgdb - here->B1GDoverlapCap ) ;
            xcgsb = (cgsb - here->B1GSoverlapCap) ;
            xcbgb = (cbgb - here->B1GBoverlapCap) ;
            xcbdb = (cbdb - capbd ) ;
            xcbsb = (cbsb - capbs ) ;

            m = here->B1m;

            *(here->B1GgPtr   ) +=   m * (xcggb * s->real);
            *(here->B1GgPtr +1) +=   m * (xcggb * s->imag);
            *(here->B1BbPtr   ) +=   m * ((-xcbgb-xcbdb-xcbsb) * s->real);
            *(here->B1BbPtr +1) +=   m * ((-xcbgb-xcbdb-xcbsb) * s->imag);
            *(here->B1DPdpPtr   ) += m * (xcddb * s->real);
            *(here->B1DPdpPtr +1) += m * (xcddb * s->imag);
            *(here->B1SPspPtr   ) += m * (xcssb * s->real);
            *(here->B1SPspPtr +1) += m * (xcssb * s->imag);
            *(here->B1GbPtr   ) +=   m * ((-xcggb-xcgdb-xcgsb) * s->real);
            *(here->B1GbPtr +1) +=   m * ((-xcggb-xcgdb-xcgsb) * s->imag);
            *(here->B1GdpPtr   ) +=  m * (xcgdb * s->real);
            *(here->B1GdpPtr +1) +=  m * (xcgdb * s->imag);
            *(here->B1GspPtr   ) +=  m * (xcgsb * s->real);
            *(here->B1GspPtr +1) +=  m * (xcgsb * s->imag);
            *(here->B1BgPtr   ) +=   m * (xcbgb * s->real);
            *(here->B1BgPtr +1) +=   m * (xcbgb * s->imag);
            *(here->B1BdpPtr   ) +=  m * (xcbdb * s->real);
            *(here->B1BdpPtr +1) +=  m * (xcbdb * s->imag);
            *(here->B1BspPtr   ) +=  m * (xcbsb * s->real);
            *(here->B1BspPtr +1) +=  m * (xcbsb * s->imag);
            *(here->B1DPgPtr   ) +=  m * (xcdgb * s->real);
            *(here->B1DPgPtr +1) +=  m * (xcdgb * s->imag);
            *(here->B1DPbPtr   ) +=  m * ((-xcdgb-xcddb-xcdsb) * s->real);
            *(here->B1DPbPtr +1) +=  m * ((-xcdgb-xcddb-xcdsb) * s->imag);
            *(here->B1DPspPtr   ) += m * (xcdsb * s->real);
            *(here->B1DPspPtr +1) += m * (xcdsb * s->imag);
            *(here->B1SPgPtr   ) +=  m * (xcsgb * s->real);
            *(here->B1SPgPtr +1) +=  m * (xcsgb * s->imag);
            *(here->B1SPbPtr   ) +=  m * ((-xcsgb-xcsdb-xcssb) * s->real);
            *(here->B1SPbPtr +1) +=  m * ((-xcsgb-xcsdb-xcssb) * s->imag);
            *(here->B1SPdpPtr   ) += m * (xcsdb * s->real);
            *(here->B1SPdpPtr +1) += m * (xcsdb * s->imag);
            *(here->B1DdPtr) +=      m * (gdpr);
            *(here->B1SsPtr) +=      m * (gspr);
            *(here->B1BbPtr) +=      m * (gbd+gbs);
            *(here->B1DPdpPtr) +=    m * (gdpr+gds+gbd+xrev*(gm+gmbs));
            *(here->B1SPspPtr) +=    m * (gspr+gds+gbs+xnrm*(gm+gmbs));
            *(here->B1DdpPtr) -=     m * (gdpr);
            *(here->B1SspPtr) -=     m * (gspr);
            *(here->B1BdpPtr) -=     m * (gbd);
            *(here->B1BspPtr) -=     m * (gbs);
            *(here->B1DPdPtr) -=     m * (gdpr);
            *(here->B1DPgPtr) +=     m * ((xnrm-xrev)*gm);
            *(here->B1DPbPtr) +=     m * (-gbd+(xnrm-xrev)*gmbs);
            *(here->B1DPspPtr) +=    m * (-gds-xnrm*(gm+gmbs));
            *(here->B1SPgPtr) +=     m * (-(xnrm-xrev)*gm);
            *(here->B1SPsPtr) -=     m * (gspr);
            *(here->B1SPbPtr) +=     m * (-gbs-(xnrm-xrev)*gmbs);
            *(here->B1SPdpPtr) +=    m * (-gds-xrev*(gm+gmbs));

        }
    }
    return(OK);
}
