/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1998 Samuel Fung, Dennis Sinitsky and Stephen Tang
File: b3soipzld.c          98/5/01
Modified by Paolo Nenzi 2002
**********/


#include "ngspice.h"
#include "cktdefs.h"
#include "complex.h"
#include "sperror.h"
#include "b3soidef.h"
#include "suffix.h"

int
B3SOIpzLoad(GENmodel *inModel, CKTcircuit *ckt, SPcomplex *s)
{
B3SOImodel *model = (B3SOImodel*)inModel;
B3SOIinstance *here;
double xcggb, xcgdb, xcgsb, xcbgb, xcbdb, xcbsb, xcddb, xcssb, xcdgb;
double gdpr, gspr, gds, gbd, gbs;
double capbd = 0.0, capbs = 0.0, xcsgb, xcdsb, xcsdb;
double cggb, cgdb, cgsb, cbgb, cbdb, cbsb, cddb, cdgb, cdsb;
double GSoverlapCap, GDoverlapCap, GBoverlapCap = 0.0;
double FwdSum, RevSum, Gm, Gmbs;

double m;

    for (; model != NULL; model = model->B3SOInextModel) 
    {    for (here = model->B3SOIinstances; here!= NULL;
              here = here->B3SOInextInstance) 
	 {
	 
	    if (here->B3SOIowner != ARCHme)
                    continue;

            if (here->B3SOImode >= 0) 
	    {   Gm = here->B3SOIgm;
		Gmbs = here->B3SOIgmbs;
		FwdSum = Gm + Gmbs;
		RevSum = 0.0;
                cggb = here->B3SOIcggb;
                cgsb = here->B3SOIcgsb;
                cgdb = here->B3SOIcgdb;

                cbgb = here->B3SOIcbgb;
                cbsb = here->B3SOIcbsb;
                cbdb = here->B3SOIcbdb;

                cdgb = here->B3SOIcdgb;
                cdsb = here->B3SOIcdsb;
                cddb = here->B3SOIcddb;
            }
	    else
	    {   Gm = -here->B3SOIgm;
		Gmbs = -here->B3SOIgmbs;
		FwdSum = 0.0;
		RevSum = -Gm - Gmbs;
                cggb = here->B3SOIcggb;
                cgsb = here->B3SOIcgdb;
                cgdb = here->B3SOIcgsb;

                cbgb = here->B3SOIcbgb;
                cbsb = here->B3SOIcbdb;
                cbdb = here->B3SOIcbsb;

                cdgb = -(here->B3SOIcdgb + cggb + cbgb);
                cdsb = -(here->B3SOIcddb + cgsb + cbsb);
                cddb = -(here->B3SOIcdsb + cgdb + cbdb);
            }
            gdpr=here->B3SOIdrainConductance;
            gspr=here->B3SOIsourceConductance;
            gds= here->B3SOIgds;
            gbd= here->B3SOIgjdb;
            gbs= here->B3SOIgjsb;
#ifdef BULKCODE
            capbd= here->B3SOIcapbd;
            capbs= here->B3SOIcapbs;
#endif
	    GSoverlapCap = here->B3SOIcgso;
	    GDoverlapCap = here->B3SOIcgdo;
#ifdef BULKCODE
	    GBoverlapCap = here->pParam->B3SOIcgbo;
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


            m = here->B3SOIm;

            *(here->B3SOIGgPtr ) += m * (xcggb * s->real);
            *(here->B3SOIGgPtr +1) += m * (xcggb * s->imag);
            *(here->B3SOIBbPtr ) += m * ((-xcbgb-xcbdb-xcbsb) * s->real);
            *(here->B3SOIBbPtr +1) += m * ((-xcbgb-xcbdb-xcbsb) * s->imag);
            *(here->B3SOIDPdpPtr ) += m * (xcddb * s->real);
            *(here->B3SOIDPdpPtr +1) += m * (xcddb * s->imag);
            *(here->B3SOISPspPtr ) += m * (xcssb * s->real);
            *(here->B3SOISPspPtr +1) += m * (xcssb * s->imag);
            *(here->B3SOIGbPtr ) += m * ((-xcggb-xcgdb-xcgsb) * s->real);
            *(here->B3SOIGbPtr +1) += m * ((-xcggb-xcgdb-xcgsb) * s->imag);
            *(here->B3SOIGdpPtr ) += m * (xcgdb * s->real);
            *(here->B3SOIGdpPtr +1) += m * (xcgdb * s->imag);
            *(here->B3SOIGspPtr ) += m * (xcgsb * s->real);
            *(here->B3SOIGspPtr +1) += m * (xcgsb * s->imag);
            *(here->B3SOIBgPtr ) += m * (xcbgb * s->real);
            *(here->B3SOIBgPtr +1) += m * (xcbgb * s->imag);
            *(here->B3SOIBdpPtr ) += m * (xcbdb * s->real);
            *(here->B3SOIBdpPtr +1) += m * (xcbdb * s->imag);
            *(here->B3SOIBspPtr ) += m * (xcbsb * s->real);
            *(here->B3SOIBspPtr +1) += m * (xcbsb * s->imag);
            *(here->B3SOIDPgPtr ) += m * (xcdgb * s->real);
            *(here->B3SOIDPgPtr +1) += m * (xcdgb * s->imag);
            *(here->B3SOIDPbPtr ) += m * ((-xcdgb-xcddb-xcdsb) * s->real);
            *(here->B3SOIDPbPtr +1) += m * ((-xcdgb-xcddb-xcdsb) * s->imag);
            *(here->B3SOIDPspPtr ) += m * (xcdsb * s->real);
            *(here->B3SOIDPspPtr +1) += m * (xcdsb * s->imag);
            *(here->B3SOISPgPtr ) += m * (xcsgb * s->real);
            *(here->B3SOISPgPtr +1) += m * (xcsgb * s->imag);
            *(here->B3SOISPbPtr ) += m * ((-xcsgb-xcsdb-xcssb) * s->real);
            *(here->B3SOISPbPtr +1) += m * ((-xcsgb-xcsdb-xcssb) * s->imag);
            *(here->B3SOISPdpPtr ) += m * (xcsdb * s->real);
            *(here->B3SOISPdpPtr +1) += m * (xcsdb * s->imag);
            *(here->B3SOIDdPtr) += m * gdpr;
            *(here->B3SOISsPtr) += m * gspr;
            *(here->B3SOIBbPtr) += m * (gbd + gbs);
            *(here->B3SOIDPdpPtr) += m * (gdpr + gds + gbd + RevSum);
            *(here->B3SOISPspPtr) += m * (gspr + gds + gbs + FwdSum);
            *(here->B3SOIDdpPtr) -= m * gdpr;
            *(here->B3SOISspPtr) -= m * gspr;
            *(here->B3SOIBdpPtr) -= m * gbd;
            *(here->B3SOIBspPtr) -= m * gbs;
            *(here->B3SOIDPdPtr) -= m * gdpr;
            *(here->B3SOIDPgPtr) += m * Gm;
            *(here->B3SOIDPbPtr) -= m * (gbd - Gmbs);
            *(here->B3SOIDPspPtr) -= m * (gds + FwdSum);
            *(here->B3SOISPgPtr) -= m * Gm;
            *(here->B3SOISPsPtr) -= m * gspr;
            *(here->B3SOISPbPtr) -= m * (gbs + Gmbs);
            *(here->B3SOISPdpPtr) -= m * (gds + RevSum);

        }
    }
    return(OK);
}


