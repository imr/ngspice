/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1998 Samuel Fung, Dennis Sinitsky and Stephen Tang
File: b3soipzld.c          98/5/01
**********/


#include "ngspice.h"
#include <stdio.h>
#include "cktdefs.h"
#include "complex.h"
#include "sperror.h"
#include "b3soipddef.h"
#include "suffix.h"

int
B3SOIPDpzLoad (inModel, ckt, s)
     GENmodel *inModel;
      CKTcircuit *ckt;
      SPcomplex *s;
{
   B3SOIPDmodel *model = (B3SOIPDmodel *) inModel;
   B3SOIPDinstance *here;
  double xcggb, xcgdb, xcgsb, xcbgb, xcbdb, xcbsb, xcddb, xcssb, xcdgb;
  double gdpr, gspr, gds, gbd, gbs, capbd, capbs, xcsgb, xcdsb, xcsdb;
  double cggb, cgdb, cgsb, cbgb, cbdb, cbsb, cddb, cdgb, cdsb;
  double GSoverlapCap, GDoverlapCap, GBoverlapCap;
  double FwdSum, RevSum, Gm, Gmbs;

  for (; model != NULL; model = model->B3SOIPDnextModel)
    {
      for (here = model->B3SOIPDinstances; here != NULL;
	   here = here->B3SOIPDnextInstance)
	{
	  if (here->B3SOIPDmode >= 0)
	    {
	      Gm = here->B3SOIPDgm;
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
	    {
	      Gm = -here->B3SOIPDgm;
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
	  gdpr = here->B3SOIPDdrainConductance;
	  gspr = here->B3SOIPDsourceConductance;
	  gds = here->B3SOIPDgds;
	  gbd = here->B3SOIPDgjdb;
	  gbs = here->B3SOIPDgjsb;
#ifdef BULKCODE
	  capbd = here->B3SOIPDcapbd;
	  capbs = here->B3SOIPDcapbs;
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
	  xcssb = (capbs + GSoverlapCap - (cgsb + cbsb + cdsb));
	  xcggb = (cggb + GDoverlapCap + GSoverlapCap + GBoverlapCap);
	  xcgdb = (cgdb - GDoverlapCap);
	  xcgsb = (cgsb - GSoverlapCap);
	  xcbgb = (cbgb - GBoverlapCap);
	  xcbdb = (cbdb - capbd);
	  xcbsb = (cbsb - capbs);


	  *(here->B3SOIPDGgPtr) += xcggb * s->real;
	  *(here->B3SOIPDGgPtr + 1) += xcggb * s->imag;
	  *(here->B3SOIPDBbPtr) += (-xcbgb - xcbdb - xcbsb) * s->real;
	  *(here->B3SOIPDBbPtr + 1) += (-xcbgb - xcbdb - xcbsb) * s->imag;
	  *(here->B3SOIPDDPdpPtr) += xcddb * s->real;
	  *(here->B3SOIPDDPdpPtr + 1) += xcddb * s->imag;
	  *(here->B3SOIPDSPspPtr) += xcssb * s->real;
	  *(here->B3SOIPDSPspPtr + 1) += xcssb * s->imag;
	  *(here->B3SOIPDGbPtr) += (-xcggb - xcgdb - xcgsb) * s->real;
	  *(here->B3SOIPDGbPtr + 1) += (-xcggb - xcgdb - xcgsb) * s->imag;
	  *(here->B3SOIPDGdpPtr) += xcgdb * s->real;
	  *(here->B3SOIPDGdpPtr + 1) += xcgdb * s->imag;
	  *(here->B3SOIPDGspPtr) += xcgsb * s->real;
	  *(here->B3SOIPDGspPtr + 1) += xcgsb * s->imag;
	  *(here->B3SOIPDBgPtr) += xcbgb * s->real;
	  *(here->B3SOIPDBgPtr + 1) += xcbgb * s->imag;
	  *(here->B3SOIPDBdpPtr) += xcbdb * s->real;
	  *(here->B3SOIPDBdpPtr + 1) += xcbdb * s->imag;
	  *(here->B3SOIPDBspPtr) += xcbsb * s->real;
	  *(here->B3SOIPDBspPtr + 1) += xcbsb * s->imag;
	  *(here->B3SOIPDDPgPtr) += xcdgb * s->real;
	  *(here->B3SOIPDDPgPtr + 1) += xcdgb * s->imag;
	  *(here->B3SOIPDDPbPtr) += (-xcdgb - xcddb - xcdsb) * s->real;
	  *(here->B3SOIPDDPbPtr + 1) += (-xcdgb - xcddb - xcdsb) * s->imag;
	  *(here->B3SOIPDDPspPtr) += xcdsb * s->real;
	  *(here->B3SOIPDDPspPtr + 1) += xcdsb * s->imag;
	  *(here->B3SOIPDSPgPtr) += xcsgb * s->real;
	  *(here->B3SOIPDSPgPtr + 1) += xcsgb * s->imag;
	  *(here->B3SOIPDSPbPtr) += (-xcsgb - xcsdb - xcssb) * s->real;
	  *(here->B3SOIPDSPbPtr + 1) += (-xcsgb - xcsdb - xcssb) * s->imag;
	  *(here->B3SOIPDSPdpPtr) += xcsdb * s->real;
	  *(here->B3SOIPDSPdpPtr + 1) += xcsdb * s->imag;
	  *(here->B3SOIPDDdPtr) += gdpr;
	  *(here->B3SOIPDSsPtr) += gspr;
	  *(here->B3SOIPDBbPtr) += gbd + gbs;
	  *(here->B3SOIPDDPdpPtr) += gdpr + gds + gbd + RevSum;
	  *(here->B3SOIPDSPspPtr) += gspr + gds + gbs + FwdSum;
	  *(here->B3SOIPDDdpPtr) -= gdpr;
	  *(here->B3SOIPDSspPtr) -= gspr;
	  *(here->B3SOIPDBdpPtr) -= gbd;
	  *(here->B3SOIPDBspPtr) -= gbs;
	  *(here->B3SOIPDDPdPtr) -= gdpr;
	  *(here->B3SOIPDDPgPtr) += Gm;
	  *(here->B3SOIPDDPbPtr) -= gbd - Gmbs;
	  *(here->B3SOIPDDPspPtr) -= gds + FwdSum;
	  *(here->B3SOIPDSPgPtr) -= Gm;
	  *(here->B3SOIPDSPsPtr) -= gspr;
	  *(here->B3SOIPDSPbPtr) -= gbs + Gmbs;
	  *(here->B3SOIPDSPdpPtr) -= gds + RevSum;

	}
    }
  return (OK);
}
