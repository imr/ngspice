/**********
Copyright 1999 Regents of the University of California.  All rights reserved.
Author: Weidong Liu and Pin Su         Feb 1999
Author: 1998 Samuel Fung, Dennis Sinitsky and Stephen Tang
File: b3soiddpzld.c          98/5/01
Modified by Paolo Nenzi 2002
**********/

/*
 * Revision 2.1  99/9/27 Pin Su 
 * BSIMDD2.1 release
 */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ngspice/complex.h"
#include "ngspice/sperror.h"
#include "b3soidddef.h"
#include "ngspice/suffix.h"

int
B3SOIDDpzLoad(GENmodel *inModel, CKTcircuit *ckt, SPcomplex *s)
{
B3SOIDDmodel *model = (B3SOIDDmodel*)inModel;
B3SOIDDinstance *here;
double xcggb, xcgdb, xcgsb, xcbgb, xcbdb, xcbsb, xcddb, xcssb, xcdgb;
double gdpr, gspr, gds, gbd, gbs, capbd = 0.0, capbs = 0.0;
double xcsgb, xcdsb, xcsdb;
double cggb, cgdb, cgsb, cbgb, cbdb, cbsb, cddb, cdgb, cdsb;
double GSoverlapCap, GDoverlapCap, GBoverlapCap = 0.0;
double FwdSum, RevSum, Gm, Gmbs;

double m;

    NG_IGNORE(ckt);

    for (; model != NULL; model = B3SOIDDnextModel(model)) 
    {    for (here = B3SOIDDinstances(model); here!= NULL;
              here = B3SOIDDnextInstance(here)) 
	 {
            if (here->B3SOIDDmode >= 0) 
	    {   Gm = here->B3SOIDDgm;
		Gmbs = here->B3SOIDDgmbs;
		FwdSum = Gm + Gmbs;
		RevSum = 0.0;
                cggb = here->B3SOIDDcggb;
                cgsb = here->B3SOIDDcgsb;
                cgdb = here->B3SOIDDcgdb;

                cbgb = here->B3SOIDDcbgb;
                cbsb = here->B3SOIDDcbsb;
                cbdb = here->B3SOIDDcbdb;

                cdgb = here->B3SOIDDcdgb;
                cdsb = here->B3SOIDDcdsb;
                cddb = here->B3SOIDDcddb;
            }
	    else
	    {   Gm = -here->B3SOIDDgm;
		Gmbs = -here->B3SOIDDgmbs;
		FwdSum = 0.0;
		RevSum = -Gm - Gmbs;
                cggb = here->B3SOIDDcggb;
                cgsb = here->B3SOIDDcgdb;
                cgdb = here->B3SOIDDcgsb;

                cbgb = here->B3SOIDDcbgb;
                cbsb = here->B3SOIDDcbdb;
                cbdb = here->B3SOIDDcbsb;

                cdgb = -(here->B3SOIDDcdgb + cggb + cbgb);
                cdsb = -(here->B3SOIDDcddb + cgsb + cbsb);
                cddb = -(here->B3SOIDDcdsb + cgdb + cbdb);
            }
            gdpr=here->B3SOIDDdrainConductance;
            gspr=here->B3SOIDDsourceConductance;
            gds= here->B3SOIDDgds;
            gbd= here->B3SOIDDgjdb;
            gbs= here->B3SOIDDgjsb;
#ifdef BULKCODE
            capbd= here->B3SOIDDcapbd;
            capbs= here->B3SOIDDcapbs;
#endif
	    GSoverlapCap = here->B3SOIDDcgso;
	    GDoverlapCap = here->B3SOIDDcgdo;
#ifdef BULKCODE
	    GBoverlapCap = here->pParam->B3SOIDDcgbo;
#endif

            xcdgb = (cdgb - GDoverlapCap);
            xcddb = (cddb + capbd + GDoverlapCap);
            xcdsb = cdsb;
            xcsgb = -(cggb + cbgb + cdgb + GSoverlapCap);
            xcsdb = -(cgdb + cbdb + cddb);
            xcssb = (capbs + GSoverlapCap - (cgsb+cbsb+cdsb));
            xcggb = (cggb + GDoverlapCap + GSoverlapCap + GBoverlapCap);
            xcgdb = (cgdb - GDoverlapCap);
            xcgsb = (cgsb - GSoverlapCap);
            xcbgb = (cbgb - GBoverlapCap);
            xcbdb = (cbdb - capbd);
            xcbsb = (cbsb - capbs);

            m = here->B3SOIDDm;

            *(here->B3SOIDDGgPtr ) += m * (xcggb * s->real);
            *(here->B3SOIDDGgPtr +1) += m * (xcggb * s->imag);
            *(here->B3SOIDDBbPtr ) += m * ((-xcbgb-xcbdb-xcbsb) * s->real);
            *(here->B3SOIDDBbPtr +1) += m * ((-xcbgb-xcbdb-xcbsb) * s->imag);
            *(here->B3SOIDDDPdpPtr ) += m * (xcddb * s->real);
            *(here->B3SOIDDDPdpPtr +1) += m * (xcddb * s->imag);
            *(here->B3SOIDDSPspPtr ) += m * (xcssb * s->real);
            *(here->B3SOIDDSPspPtr +1) += m * (xcssb * s->imag);
            *(here->B3SOIDDGbPtr ) += m * ((-xcggb-xcgdb-xcgsb) * s->real);
            *(here->B3SOIDDGbPtr +1) += m * ((-xcggb-xcgdb-xcgsb) * s->imag);
            *(here->B3SOIDDGdpPtr ) += m * (xcgdb * s->real);
            *(here->B3SOIDDGdpPtr +1) += m * (xcgdb * s->imag);
            *(here->B3SOIDDGspPtr ) += m * (xcgsb * s->real);
            *(here->B3SOIDDGspPtr +1) += m * (xcgsb * s->imag);
            *(here->B3SOIDDBgPtr ) += m * (xcbgb * s->real);
            *(here->B3SOIDDBgPtr +1) += m * (xcbgb * s->imag);
            *(here->B3SOIDDBdpPtr ) += m * (xcbdb * s->real);
            *(here->B3SOIDDBdpPtr +1) += m * (xcbdb * s->imag);
            *(here->B3SOIDDBspPtr ) += m * (xcbsb * s->real);
            *(here->B3SOIDDBspPtr +1) += (xcbsb * s->imag);
            *(here->B3SOIDDDPgPtr ) += m * (xcdgb * s->real);
            *(here->B3SOIDDDPgPtr +1) += m * (xcdgb * s->imag);
            *(here->B3SOIDDDPbPtr ) += m * ((-xcdgb-xcddb-xcdsb) * s->real);
            *(here->B3SOIDDDPbPtr +1) += m * ((-xcdgb-xcddb-xcdsb) * s->imag);
            *(here->B3SOIDDDPspPtr ) += m * (xcdsb * s->real);
            *(here->B3SOIDDDPspPtr +1) += m * (xcdsb * s->imag);
            *(here->B3SOIDDSPgPtr ) += m * (xcsgb * s->real);
            *(here->B3SOIDDSPgPtr +1) += m * (xcsgb * s->imag);
            *(here->B3SOIDDSPbPtr ) += m * ((-xcsgb-xcsdb-xcssb) * s->real);
            *(here->B3SOIDDSPbPtr +1) += m * ((-xcsgb-xcsdb-xcssb) * s->imag);
            *(here->B3SOIDDSPdpPtr ) += m * (xcsdb * s->real);
            *(here->B3SOIDDSPdpPtr +1) += m * (xcsdb * s->imag);
            *(here->B3SOIDDDdPtr) += m * gdpr;
            *(here->B3SOIDDSsPtr) += m * gspr;
            *(here->B3SOIDDBbPtr) += m * (gbd + gbs);
            *(here->B3SOIDDDPdpPtr) += m * (gdpr + gds + gbd + RevSum);
            *(here->B3SOIDDSPspPtr) += m * (gspr + gds + gbs + FwdSum);
            *(here->B3SOIDDDdpPtr) -= m * gdpr;
            *(here->B3SOIDDSspPtr) -= m * gspr;
            *(here->B3SOIDDBdpPtr) -= m * gbd;
            *(here->B3SOIDDBspPtr) -= m * gbs;
            *(here->B3SOIDDDPdPtr) -= m * gdpr;
            *(here->B3SOIDDDPgPtr) += m * Gm;
            *(here->B3SOIDDDPbPtr) -= m * (gbd - Gmbs);
            *(here->B3SOIDDDPspPtr) -= m * (gds + FwdSum);
            *(here->B3SOIDDSPgPtr) -= m * Gm;
            *(here->B3SOIDDSPsPtr) -= m * gspr;
            *(here->B3SOIDDSPbPtr) -= m * (gbs + Gmbs);
            *(here->B3SOIDDSPdpPtr) -= m * (gds + RevSum);

        }
    }
    return(OK);
}


