/**********
Copyright 1999 Regents of the University of California.  All rights reserved.
Author: 1998 Samuel Fung, Dennis Sinitsky and Stephen Tang
File: b3soifdpzld.c          98/5/01
**********/


#include "ngspice.h"
#include <stdio.h>
#include "cktdefs.h"
#include "complex.h"
#include "sperror.h"
#include "b3soifddef.h"
#include "suffix.h"

int
B3SOIFDpzLoad(inModel,ckt,s)
GENmodel *inModel;
 CKTcircuit *ckt;
 SPcomplex *s;
{
 B3SOIFDmodel *model = (B3SOIFDmodel*)inModel;
 B3SOIFDinstance *here;
double xcggb, xcgdb, xcgsb, xcbgb, xcbdb, xcbsb, xcddb, xcssb, xcdgb;
double gdpr, gspr, gds, gbd, gbs, capbd, capbs, xcsgb, xcdsb, xcsdb;
double cggb, cgdb, cgsb, cbgb, cbdb, cbsb, cddb, cdgb, cdsb;
double GSoverlapCap, GDoverlapCap, GBoverlapCap;
double FwdSum, RevSum, Gm, Gmbs;

    for (; model != NULL; model = model->B3SOIFDnextModel) 
    {    for (here = model->B3SOIFDinstances; here!= NULL;
              here = here->B3SOIFDnextInstance) 
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


            *(here->B3SOIFDGgPtr ) += xcggb * s->real;
            *(here->B3SOIFDGgPtr +1) += xcggb * s->imag;
            *(here->B3SOIFDBbPtr ) += (-xcbgb-xcbdb-xcbsb) * s->real;
            *(here->B3SOIFDBbPtr +1) += (-xcbgb-xcbdb-xcbsb) * s->imag;
            *(here->B3SOIFDDPdpPtr ) += xcddb * s->real;
            *(here->B3SOIFDDPdpPtr +1) += xcddb * s->imag;
            *(here->B3SOIFDSPspPtr ) += xcssb * s->real;
            *(here->B3SOIFDSPspPtr +1) += xcssb * s->imag;
            *(here->B3SOIFDGbPtr ) += (-xcggb-xcgdb-xcgsb) * s->real;
            *(here->B3SOIFDGbPtr +1) += (-xcggb-xcgdb-xcgsb) * s->imag;
            *(here->B3SOIFDGdpPtr ) += xcgdb * s->real;
            *(here->B3SOIFDGdpPtr +1) += xcgdb * s->imag;
            *(here->B3SOIFDGspPtr ) += xcgsb * s->real;
            *(here->B3SOIFDGspPtr +1) += xcgsb * s->imag;
            *(here->B3SOIFDBgPtr ) += xcbgb * s->real;
            *(here->B3SOIFDBgPtr +1) += xcbgb * s->imag;
            *(here->B3SOIFDBdpPtr ) += xcbdb * s->real;
            *(here->B3SOIFDBdpPtr +1) += xcbdb * s->imag;
            *(here->B3SOIFDBspPtr ) += xcbsb * s->real;
            *(here->B3SOIFDBspPtr +1) += xcbsb * s->imag;
            *(here->B3SOIFDDPgPtr ) += xcdgb * s->real;
            *(here->B3SOIFDDPgPtr +1) += xcdgb * s->imag;
            *(here->B3SOIFDDPbPtr ) += (-xcdgb-xcddb-xcdsb) * s->real;
            *(here->B3SOIFDDPbPtr +1) += (-xcdgb-xcddb-xcdsb) * s->imag;
            *(here->B3SOIFDDPspPtr ) += xcdsb * s->real;
            *(here->B3SOIFDDPspPtr +1) += xcdsb * s->imag;
            *(here->B3SOIFDSPgPtr ) += xcsgb * s->real;
            *(here->B3SOIFDSPgPtr +1) += xcsgb * s->imag;
            *(here->B3SOIFDSPbPtr ) += (-xcsgb-xcsdb-xcssb) * s->real;
            *(here->B3SOIFDSPbPtr +1) += (-xcsgb-xcsdb-xcssb) * s->imag;
            *(here->B3SOIFDSPdpPtr ) += xcsdb * s->real;
            *(here->B3SOIFDSPdpPtr +1) += xcsdb * s->imag;
            *(here->B3SOIFDDdPtr) += gdpr;
            *(here->B3SOIFDSsPtr) += gspr;
            *(here->B3SOIFDBbPtr) += gbd+gbs;
            *(here->B3SOIFDDPdpPtr) += gdpr+gds+gbd+RevSum;
            *(here->B3SOIFDSPspPtr) += gspr+gds+gbs+FwdSum;
            *(here->B3SOIFDDdpPtr) -= gdpr;
            *(here->B3SOIFDSspPtr) -= gspr;
            *(here->B3SOIFDBdpPtr) -= gbd;
            *(here->B3SOIFDBspPtr) -= gbs;
            *(here->B3SOIFDDPdPtr) -= gdpr;
            *(here->B3SOIFDDPgPtr) += Gm;
            *(here->B3SOIFDDPbPtr) -= gbd - Gmbs;
            *(here->B3SOIFDDPspPtr) -= gds + FwdSum;
            *(here->B3SOIFDSPgPtr) -= Gm;
            *(here->B3SOIFDSPsPtr) -= gspr;
            *(here->B3SOIFDSPbPtr) -= gbs + Gmbs;
            *(here->B3SOIFDSPdpPtr) -= gds + RevSum;

        }
    }
    return(OK);
}


