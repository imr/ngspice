/**********
Copyright 1999 Regents of the University of California.  All rights reserved.
Author: Weidong Liu and Pin Su         Feb 1999
Author: 1998 Samuel Fung, Dennis Sinitsky and Stephen Tang
File: b3soiddpzld.c          98/5/01
**********/


#include "ngspice.h"
#include <stdio.h>
#include "cktdefs.h"
#include "complex.h"
#include "sperror.h"
#include "b3soidddef.h"
#include "suffix.h"

int
B3SOIDDpzLoad(inModel,ckt,s)
GENmodel *inModel;
register CKTcircuit *ckt;
register SPcomplex *s;
{
register B3SOIDDmodel *model = (B3SOIDDmodel*)inModel;
register B3SOIDDinstance *here;
double xcggb, xcgdb, xcgsb, xcbgb, xcbdb, xcbsb, xcddb, xcssb, xcdgb;
double gdpr, gspr, gds, gbd, gbs, capbd, capbs, xcsgb, xcdsb, xcsdb;
double cggb, cgdb, cgsb, cbgb, cbdb, cbsb, cddb, cdgb, cdsb;
double GSoverlapCap, GDoverlapCap, GBoverlapCap;
double FwdSum, RevSum, Gm, Gmbs;

    for (; model != NULL; model = model->B3SOIDDnextModel) 
    {    for (here = model->B3SOIDDinstances; here!= NULL;
              here = here->B3SOIDDnextInstance) 
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


            *(here->B3SOIDDGgPtr ) += xcggb * s->real;
            *(here->B3SOIDDGgPtr +1) += xcggb * s->imag;
            *(here->B3SOIDDBbPtr ) += (-xcbgb-xcbdb-xcbsb) * s->real;
            *(here->B3SOIDDBbPtr +1) += (-xcbgb-xcbdb-xcbsb) * s->imag;
            *(here->B3SOIDDDPdpPtr ) += xcddb * s->real;
            *(here->B3SOIDDDPdpPtr +1) += xcddb * s->imag;
            *(here->B3SOIDDSPspPtr ) += xcssb * s->real;
            *(here->B3SOIDDSPspPtr +1) += xcssb * s->imag;
            *(here->B3SOIDDGbPtr ) += (-xcggb-xcgdb-xcgsb) * s->real;
            *(here->B3SOIDDGbPtr +1) += (-xcggb-xcgdb-xcgsb) * s->imag;
            *(here->B3SOIDDGdpPtr ) += xcgdb * s->real;
            *(here->B3SOIDDGdpPtr +1) += xcgdb * s->imag;
            *(here->B3SOIDDGspPtr ) += xcgsb * s->real;
            *(here->B3SOIDDGspPtr +1) += xcgsb * s->imag;
            *(here->B3SOIDDBgPtr ) += xcbgb * s->real;
            *(here->B3SOIDDBgPtr +1) += xcbgb * s->imag;
            *(here->B3SOIDDBdpPtr ) += xcbdb * s->real;
            *(here->B3SOIDDBdpPtr +1) += xcbdb * s->imag;
            *(here->B3SOIDDBspPtr ) += xcbsb * s->real;
            *(here->B3SOIDDBspPtr +1) += xcbsb * s->imag;
            *(here->B3SOIDDDPgPtr ) += xcdgb * s->real;
            *(here->B3SOIDDDPgPtr +1) += xcdgb * s->imag;
            *(here->B3SOIDDDPbPtr ) += (-xcdgb-xcddb-xcdsb) * s->real;
            *(here->B3SOIDDDPbPtr +1) += (-xcdgb-xcddb-xcdsb) * s->imag;
            *(here->B3SOIDDDPspPtr ) += xcdsb * s->real;
            *(here->B3SOIDDDPspPtr +1) += xcdsb * s->imag;
            *(here->B3SOIDDSPgPtr ) += xcsgb * s->real;
            *(here->B3SOIDDSPgPtr +1) += xcsgb * s->imag;
            *(here->B3SOIDDSPbPtr ) += (-xcsgb-xcsdb-xcssb) * s->real;
            *(here->B3SOIDDSPbPtr +1) += (-xcsgb-xcsdb-xcssb) * s->imag;
            *(here->B3SOIDDSPdpPtr ) += xcsdb * s->real;
            *(here->B3SOIDDSPdpPtr +1) += xcsdb * s->imag;
            *(here->B3SOIDDDdPtr) += gdpr;
            *(here->B3SOIDDSsPtr) += gspr;
            *(here->B3SOIDDBbPtr) += gbd+gbs;
            *(here->B3SOIDDDPdpPtr) += gdpr+gds+gbd+RevSum;
            *(here->B3SOIDDSPspPtr) += gspr+gds+gbs+FwdSum;
            *(here->B3SOIDDDdpPtr) -= gdpr;
            *(here->B3SOIDDSspPtr) -= gspr;
            *(here->B3SOIDDBdpPtr) -= gbd;
            *(here->B3SOIDDBspPtr) -= gbs;
            *(here->B3SOIDDDPdPtr) -= gdpr;
            *(here->B3SOIDDDPgPtr) += Gm;
            *(here->B3SOIDDDPbPtr) -= gbd - Gmbs;
            *(here->B3SOIDDDPspPtr) -= gds + FwdSum;
            *(here->B3SOIDDSPgPtr) -= Gm;
            *(here->B3SOIDDSPsPtr) -= gspr;
            *(here->B3SOIDDSPbPtr) -= gbs + Gmbs;
            *(here->B3SOIDDSPdpPtr) -= gds + RevSum;

        }
    }
    return(OK);
}


