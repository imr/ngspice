/**********
Copyright 1999 Regents of the University of California.  All rights reserved.
Author: 1998 Samuel Fung, Dennis Sinitsky and Stephen Tang
Modified by Paolo Nenzi 2002
File: b3soifdpzld.c          98/5/01
**********/

/*
 * Revision 2.1  99/9/27 Pin Su 
 * BSIMFD2.1 release
 */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ngspice/complex.h"
#include "ngspice/sperror.h"
#include "b3soifddef.h"
#include "ngspice/suffix.h"

int
B3SOIFDpzLoad(GENmodel *inModel, CKTcircuit *ckt, SPcomplex *s)
{
B3SOIFDmodel *model = (B3SOIFDmodel*)inModel;
B3SOIFDinstance *here;
double xcggb, xcgdb, xcgsb, xcbgb, xcbdb, xcbsb, xcddb, xcssb, xcdgb;
double gdpr, gspr, gds, gbd, gbs, capbd = 0.0, capbs = 0.0;
double xcsgb, xcdsb, xcsdb;
double cggb, cgdb, cgsb, cbgb, cbdb, cbsb, cddb, cdgb, cdsb;
double GSoverlapCap, GDoverlapCap, GBoverlapCap = 0.0;
double FwdSum, RevSum, Gm, Gmbs;

double m;

    NG_IGNORE(ckt);

    for (; model != NULL; model = B3SOIFDnextModel(model)) 
    {    for (here = B3SOIFDinstances(model); here!= NULL;
              here = B3SOIFDnextInstance(here)) 
	 {
	    if (here->B3SOIFDmode >= 0) 
	    {   Gm = here->B3SOIFDgm;
		Gmbs = here->B3SOIFDgmbs;
		FwdSum = Gm + Gmbs;
		RevSum = 0.0;
                cggb = here->B3SOIFDcggb;
                cgsb = here->B3SOIFDcgsb;
                cgdb = here->B3SOIFDcgdb;

                cbgb = here->B3SOIFDcbgb;
                cbsb = here->B3SOIFDcbsb;
                cbdb = here->B3SOIFDcbdb;

                cdgb = here->B3SOIFDcdgb;
                cdsb = here->B3SOIFDcdsb;
                cddb = here->B3SOIFDcddb;
            }
	    else
	    {   Gm = -here->B3SOIFDgm;
		Gmbs = -here->B3SOIFDgmbs;
		FwdSum = 0.0;
		RevSum = -Gm - Gmbs;
                cggb = here->B3SOIFDcggb;
                cgsb = here->B3SOIFDcgdb;
                cgdb = here->B3SOIFDcgsb;

                cbgb = here->B3SOIFDcbgb;
                cbsb = here->B3SOIFDcbdb;
                cbdb = here->B3SOIFDcbsb;

                cdgb = -(here->B3SOIFDcdgb + cggb + cbgb);
                cdsb = -(here->B3SOIFDcddb + cgsb + cbsb);
                cddb = -(here->B3SOIFDcdsb + cgdb + cbdb);
            }
            gdpr=here->B3SOIFDdrainConductance;
            gspr=here->B3SOIFDsourceConductance;
            gds= here->B3SOIFDgds;
            gbd= here->B3SOIFDgjdb;
            gbs= here->B3SOIFDgjsb;
#ifdef BULKCODE
            capbd= here->B3SOIFDcapbd;
            capbs= here->B3SOIFDcapbs;
#endif
	    GSoverlapCap = here->B3SOIFDcgso;
	    GDoverlapCap = here->B3SOIFDcgdo;
#ifdef BULKCODE
	    GBoverlapCap = here->pParam->B3SOIFDcgbo;
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

            m = here->B3SOIFDm;

            *(here->B3SOIFDGgPtr ) += m * (xcggb * s->real);
            *(here->B3SOIFDGgPtr +1) += m * (xcggb * s->imag);
            *(here->B3SOIFDBbPtr ) += m * ((-xcbgb-xcbdb-xcbsb) * s->real);
            *(here->B3SOIFDBbPtr +1) += m * ((-xcbgb-xcbdb-xcbsb) * s->imag);
            *(here->B3SOIFDDPdpPtr ) += m * (xcddb * s->real);
            *(here->B3SOIFDDPdpPtr +1) += m * (xcddb * s->imag);
            *(here->B3SOIFDSPspPtr ) += m * (xcssb * s->real);
            *(here->B3SOIFDSPspPtr +1) += m * (xcssb * s->imag);
            *(here->B3SOIFDGbPtr ) += m * ((-xcggb-xcgdb-xcgsb) * s->real);
            *(here->B3SOIFDGbPtr +1) += m * ((-xcggb-xcgdb-xcgsb) * s->imag);
            *(here->B3SOIFDGdpPtr ) += m * (xcgdb * s->real);
            *(here->B3SOIFDGdpPtr +1) += m * (xcgdb * s->imag);
            *(here->B3SOIFDGspPtr ) += m * (xcgsb * s->real);
            *(here->B3SOIFDGspPtr +1) += m * (xcgsb * s->imag);
            *(here->B3SOIFDBgPtr ) += m * (xcbgb * s->real);
            *(here->B3SOIFDBgPtr +1) += m * (xcbgb * s->imag);
            *(here->B3SOIFDBdpPtr ) += m * (xcbdb * s->real);
            *(here->B3SOIFDBdpPtr +1) += m * (xcbdb * s->imag);
            *(here->B3SOIFDBspPtr ) += m * (xcbsb * s->real);
            *(here->B3SOIFDBspPtr +1) += m * (xcbsb * s->imag);
            *(here->B3SOIFDDPgPtr ) += m * (xcdgb * s->real);
            *(here->B3SOIFDDPgPtr +1) += m * (xcdgb * s->imag);
            *(here->B3SOIFDDPbPtr ) += m * ((-xcdgb-xcddb-xcdsb) * s->real);
            *(here->B3SOIFDDPbPtr +1) += m * ((-xcdgb-xcddb-xcdsb) * s->imag);
            *(here->B3SOIFDDPspPtr ) += m * (xcdsb * s->real);
            *(here->B3SOIFDDPspPtr +1) += m * (xcdsb * s->imag);
            *(here->B3SOIFDSPgPtr ) += m * (xcsgb * s->real);
            *(here->B3SOIFDSPgPtr +1) += m * (xcsgb * s->imag);
            *(here->B3SOIFDSPbPtr ) += m * ((-xcsgb-xcsdb-xcssb) * s->real);
            *(here->B3SOIFDSPbPtr +1) += m * ((-xcsgb-xcsdb-xcssb) * s->imag);
            *(here->B3SOIFDSPdpPtr ) += m * (xcsdb * s->real);
            *(here->B3SOIFDSPdpPtr +1) += m * (xcsdb * s->imag);
            *(here->B3SOIFDDdPtr) += m * gdpr;
            *(here->B3SOIFDSsPtr) += m * gspr;
            *(here->B3SOIFDBbPtr) += m * (gbd + gbs);
            *(here->B3SOIFDDPdpPtr) += m * (gdpr + gds + gbd + RevSum);
            *(here->B3SOIFDSPspPtr) += m * (gspr + gds + gbs + FwdSum);
            *(here->B3SOIFDDdpPtr) -= m * gdpr;
            *(here->B3SOIFDSspPtr) -= m * gspr;
            *(here->B3SOIFDBdpPtr) -= m * gbd;
            *(here->B3SOIFDBspPtr) -= m * gbs;
            *(here->B3SOIFDDPdPtr) -= m * gdpr;
            *(here->B3SOIFDDPgPtr) += m * Gm;
            *(here->B3SOIFDDPbPtr) -= m * (gbd - Gmbs);
            *(here->B3SOIFDDPspPtr) -= m * (gds + FwdSum);
            *(here->B3SOIFDSPgPtr) -= m * Gm;
            *(here->B3SOIFDSPsPtr) -= m * gspr;
            *(here->B3SOIFDSPbPtr) -= m * (gbs + Gmbs);
            *(here->B3SOIFDSPdpPtr) -= m * (gds + RevSum);

        }
    }
    return(OK);
}


