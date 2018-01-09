/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1998 Samuel Fung, Dennis Sinitsky and Stephen Tang
File: b3soipdpzld.c          98/5/01
Modified by Paolo Nenzi 2002
**********/

/*
 * Revision 2.2.3  02/3/5  Pin Su 
 * BSIMPD2.2.3 release
 */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ngspice/complex.h"
#include "ngspice/sperror.h"
#include "b3soipddef.h"
#include "ngspice/suffix.h"

int
B3SOIPDpzLoad(GENmodel *inModel, CKTcircuit *ckt, SPcomplex *s)
{
B3SOIPDmodel *model = (B3SOIPDmodel*)inModel;
B3SOIPDinstance *here;
double xcggb, xcgdb, xcgsb, xcbgb, xcbdb, xcbsb, xcddb, xcssb, xcdgb;
double gdpr, gspr, gds, gbd, gbs, capbd = 0.0, capbs = 0.0;
double xcsgb, xcdsb, xcsdb;
double cggb, cgdb, cgsb, cbgb, cbdb, cbsb, cddb, cdgb, cdsb;
double GSoverlapCap, GDoverlapCap, GBoverlapCap = 0.0;
double FwdSum, RevSum, Gm, Gmbs;

double m;

    NG_IGNORE(ckt);

    for (; model != NULL; model = B3SOIPDnextModel(model)) 
    {    for (here = B3SOIPDinstances(model); here!= NULL;
              here = B3SOIPDnextInstance(here)) 
	 {
            if (here->B3SOIPDmode >= 0) 
	    {   Gm = here->B3SOIPDgm;
		Gmbs = here->B3SOIPDgmbs;
		FwdSum = Gm + Gmbs;
		RevSum = 0.0;
                cggb = here->B3SOIPDcggb;
                cgsb = here->B3SOIPDcgsb;
                cgdb = here->B3SOIPDcgdb;

                cbgb = here->B3SOIPDcbgb;
                cbsb = here->B3SOIPDcbsb;
                cbdb = here->B3SOIPDcbdb;

                cdgb = here->B3SOIPDcdgb;
                cdsb = here->B3SOIPDcdsb;
                cddb = here->B3SOIPDcddb;
            }
	    else
	    {   Gm = -here->B3SOIPDgm;
		Gmbs = -here->B3SOIPDgmbs;
		FwdSum = 0.0;
		RevSum = -Gm - Gmbs;
                cggb = here->B3SOIPDcggb;
                cgsb = here->B3SOIPDcgdb;
                cgdb = here->B3SOIPDcgsb;

                cbgb = here->B3SOIPDcbgb;
                cbsb = here->B3SOIPDcbdb;
                cbdb = here->B3SOIPDcbsb;

                cdgb = -(here->B3SOIPDcdgb + cggb + cbgb);
                cdsb = -(here->B3SOIPDcddb + cgsb + cbsb);
                cddb = -(here->B3SOIPDcdsb + cgdb + cbdb);
            }
            gdpr=here->B3SOIPDdrainConductance;
            gspr=here->B3SOIPDsourceConductance;
            gds= here->B3SOIPDgds;
            gbd= here->B3SOIPDgjdb;
            gbs= here->B3SOIPDgjsb;
#ifdef BULKCODE
            capbd= here->B3SOIPDcapbd;
            capbs= here->B3SOIPDcapbs;
#endif
	    GSoverlapCap = here->B3SOIPDcgso;
	    GDoverlapCap = here->B3SOIPDcgdo;
#ifdef BULKCODE
	    GBoverlapCap = here->pParam->B3SOIPDcgbo;
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

            m = here->B3SOIPDm;

            *(here->B3SOIPDGgPtr ) += m * (xcggb * s->real);
            *(here->B3SOIPDGgPtr +1) += m * (xcggb * s->imag);
            *(here->B3SOIPDBbPtr ) += m * ((-xcbgb-xcbdb-xcbsb) * s->real);
            *(here->B3SOIPDBbPtr +1) += m * ((-xcbgb-xcbdb-xcbsb) * s->imag);
            *(here->B3SOIPDDPdpPtr ) += m * (xcddb * s->real);
            *(here->B3SOIPDDPdpPtr +1) += m * (xcddb * s->imag);
            *(here->B3SOIPDSPspPtr ) += m * (xcssb * s->real);
            *(here->B3SOIPDSPspPtr +1) += m * (xcssb * s->imag);
            *(here->B3SOIPDGbPtr ) += m * ((-xcggb-xcgdb-xcgsb) * s->real);
            *(here->B3SOIPDGbPtr +1) += m * ((-xcggb-xcgdb-xcgsb) * s->imag);
            *(here->B3SOIPDGdpPtr ) += m * (xcgdb * s->real);
            *(here->B3SOIPDGdpPtr +1) += m * (xcgdb * s->imag);
            *(here->B3SOIPDGspPtr ) += m * (xcgsb * s->real);
            *(here->B3SOIPDGspPtr +1) += m * (xcgsb * s->imag);
            *(here->B3SOIPDBgPtr ) += m * (xcbgb * s->real);
            *(here->B3SOIPDBgPtr +1) += m * (xcbgb * s->imag);
            *(here->B3SOIPDBdpPtr ) += m * (xcbdb * s->real);
            *(here->B3SOIPDBdpPtr +1) += m * (xcbdb * s->imag);
            *(here->B3SOIPDBspPtr ) += m * (xcbsb * s->real);
            *(here->B3SOIPDBspPtr +1) += m * (xcbsb * s->imag);
            *(here->B3SOIPDDPgPtr ) += m * (xcdgb * s->real);
            *(here->B3SOIPDDPgPtr +1) += m * (xcdgb * s->imag);
            *(here->B3SOIPDDPbPtr ) += m * ((-xcdgb-xcddb-xcdsb) * s->real);
            *(here->B3SOIPDDPbPtr +1) += m * ((-xcdgb-xcddb-xcdsb) * s->imag);
            *(here->B3SOIPDDPspPtr ) += m * (xcdsb * s->real);
            *(here->B3SOIPDDPspPtr +1) += m * (xcdsb * s->imag);
            *(here->B3SOIPDSPgPtr ) += m * (xcsgb * s->real);
            *(here->B3SOIPDSPgPtr +1) += m * (xcsgb * s->imag);
            *(here->B3SOIPDSPbPtr ) += m * ((-xcsgb-xcsdb-xcssb) * s->real);
            *(here->B3SOIPDSPbPtr +1) += m * ((-xcsgb-xcsdb-xcssb) * s->imag);
            *(here->B3SOIPDSPdpPtr ) += m * (xcsdb * s->real);
            *(here->B3SOIPDSPdpPtr +1) += m * (xcsdb * s->imag);
            *(here->B3SOIPDDdPtr) += m * gdpr;
            *(here->B3SOIPDSsPtr) += m * gspr;
            *(here->B3SOIPDBbPtr) += m * (gbd + gbs);
            *(here->B3SOIPDDPdpPtr) += m * (gdpr + gds + gbd + RevSum);
            *(here->B3SOIPDSPspPtr) += m * (gspr + gds + gbs + FwdSum);
            *(here->B3SOIPDDdpPtr) -= m * gdpr;
            *(here->B3SOIPDSspPtr) -= m * gspr;
            *(here->B3SOIPDBdpPtr) -= m * gbd;
            *(here->B3SOIPDBspPtr) -= m * gbs;
            *(here->B3SOIPDDPdPtr) -= m * gdpr;
            *(here->B3SOIPDDPgPtr) += m * Gm;
            *(here->B3SOIPDDPbPtr) -= m * (gbd - Gmbs);
            *(here->B3SOIPDDPspPtr) -= m * (gds + FwdSum);
            *(here->B3SOIPDSPgPtr) -= m * Gm;
            *(here->B3SOIPDSPsPtr) -= m * gspr;
            *(here->B3SOIPDSPbPtr) -= m * (gbs + Gmbs);
            *(here->B3SOIPDSPdpPtr) -= m * (gds + RevSum);

        }
    }
    return(OK);
}


