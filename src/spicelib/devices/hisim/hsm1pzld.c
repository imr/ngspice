/***********************************************************************
 HiSIM (Hiroshima University STARC IGFET Model)
 Copyright (C) 2003 STARC

 VERSION : HiSIM 1.2.0
 FILE : hsm1pzld.c of HiSIM 1.2.0

 April 9, 2003 : released by STARC Physical Design Group
***********************************************************************/

#include "ngspice.h"
#include "cktdefs.h"
#include "complex.h"
#include "sperror.h"
#include "hsm1def.h"
#include "suffix.h"

int 
HSM1pzLoad(GENmodel *inModel, register CKTcircuit *ckt, 
           register SPcomplex *s)
{
  register HSM1model *model = (HSM1model*)inModel;
  register HSM1instance *here;
  double xcggb, xcgdb, xcgsb, xcgbb, xcbgb, xcbdb, xcbsb, xcbbb;
  double xcdgb, xcddb, xcdsb, xcdbb, xcsgb, xcsdb, xcssb, xcsbb;
  double gdpr, gspr, gds, gbd, gbs, capbd, capbs, FwdSum, RevSum, gm, gmbs;
  double cggb, cgdb, cgsb, cbgb, cbdb, cbsb, cddb, cdgb, cdsb;
  double cgso, cgdo, cgbo;
  double gbspsp, gbbdp, gbbsp, gbspg, gbspb;
  double gbspdp, gbdpdp, gbdpg, gbdpb, gbdpsp;
  double gIbtotg, gIbtotd, gIbtots, gIbtotb;
  double gIgtotg, gIgtotd, gIgtots, gIgtotb;
  double gIdtotg, gIdtotd, gIdtots, gIdtotb;
  double gIstotg, gIstotd, gIstots, gIstotb;
  
  double m; /* Multiplier */
  
  for ( ;model != NULL ;model = model->HSM1nextModel ) {
    for ( here = model->HSM1instances ;here!= NULL ;
	  here = here->HSM1nextInstance ) {
	  
      if (here->HSM1owner != ARCHme)
            continue;
  
      if ( here->HSM1_mode >= 0 ) {
	gm = here->HSM1_gm;
	gmbs = here->HSM1_gmbs;
	FwdSum = gm + gmbs;
	RevSum = 0.0;
	
	gbbdp = -here->HSM1_gbds;
	gbbsp = here->HSM1_gbds + here->HSM1_gbgs + here->HSM1_gbbs;
	
	gbdpg = here->HSM1_gbgs;
	gbdpdp = here->HSM1_gbds;
	gbdpb = here->HSM1_gbbs;
	gbdpsp = -(gbdpg + gbdpdp + gbdpb);
	
	gbspg = 0.0;
	gbspdp = 0.0;
	gbspb = 0.0;
	gbspsp = 0.0;

	if (model->HSM1_coiigs) {
	  gIbtotg = here->HSM1_gigbg;
	  gIbtotd = here->HSM1_gigbd;
	  gIbtots = here->HSM1_gigbs;
	  gIbtotb = here->HSM1_gigbb;

	  gIstotg = here->HSM1_gigsg;
	  gIstotd = here->HSM1_gigsd;
	  gIstots = here->HSM1_gigss;
	  gIstotb = here->HSM1_gigsb;

	  gIdtotg = here->HSM1_gigdg;
	  gIdtotd = here->HSM1_gigdd;
	  gIdtots = here->HSM1_gigds;
	  gIdtotb = here->HSM1_gigdb;

	}
	else {
	  gIbtotg = gIbtotd = gIbtots = gIbtotb = 0.0;
	  gIstotg = gIstotd = gIstots = gIstotb = 0.0;
	  gIdtotg = gIdtotd = gIdtots = gIdtotb = 0.0;
	}

	if (model->HSM1_coiigs) {
	  gIgtotg = gIbtotg + gIstotg + gIdtotg;
	  gIgtotd = gIbtotd + gIstotd + gIdtotd;
	  gIgtots = gIbtots + gIstots + gIdtots;
	  gIgtotb = gIbtotb + gIstotb + gIdtotb;
	}
	else
	  gIgtotg = gIgtotd = gIgtots = gIgtotb = 0.0;

	cggb = here->HSM1_cggb;
	cgsb = here->HSM1_cgsb;
	cgdb = here->HSM1_cgdb;
	
	cbgb = here->HSM1_cbgb;
	cbsb = here->HSM1_cbsb;
	cbdb = here->HSM1_cbdb;
	
	cdgb = here->HSM1_cdgb;
	cdsb = here->HSM1_cdsb;
	cddb = here->HSM1_cddb;
	
      }
      else {
	gm = -here->HSM1_gm;
	gmbs = -here->HSM1_gmbs;
	FwdSum = 0.0;
	RevSum = -(gm + gmbs);
	
	gbbsp = -here->HSM1_gbds;
	gbbdp = here->HSM1_gbds + here->HSM1_gbgs + here->HSM1_gbbs;
	
	gbdpg = 0.0;
	gbdpsp = 0.0;
	gbdpb = 0.0;
	gbdpdp = 0.0;

	if (model->HSM1_coiigs) {
	  gIbtotg = here->HSM1_gigbg;
	  gIbtotd = here->HSM1_gigbd;
	  gIbtots = here->HSM1_gigbs;
	  gIbtotb = here->HSM1_gigbb;

	  gIstotg = here->HSM1_gigsg;
	  gIstotd = here->HSM1_gigsd;
	  gIstots = here->HSM1_gigss;
	  gIstotb = here->HSM1_gigsb;

	  gIdtotg = here->HSM1_gigdg;
	  gIdtotd = here->HSM1_gigdd;
	  gIdtots = here->HSM1_gigds;
	  gIdtotb = here->HSM1_gigdb;
	}
	else {
	  gIbtotg = gIbtotd = gIbtots = gIbtotb = 0.0;
	  gIstotg = gIstotd = gIstots = gIstotb = 0.0;
	  gIdtotg = gIdtotd = gIdtots = gIdtotb = 0.0;
	}
	
	if (model->HSM1_coiigs) {
	  gIgtotg = gIbtotg + gIstotg + gIdtotg;
	  gIgtotd = gIbtotd + gIstotd + gIdtotd;
	  gIgtots = gIbtots + gIstots + gIdtots;
	  gIgtotb = gIbtotb + gIstotb + gIdtotb;
	}
	else
	  gIgtotg = gIgtotd = gIgtots = gIgtotb = 0.0;
	
	gbspg = here->HSM1_gbgs;
	gbspsp = here->HSM1_gbds;
	gbspb = here->HSM1_gbbs;
	gbspdp = -(gbspg + gbspsp + gbspb);
	
	cggb = here->HSM1_cggb;
	cgsb = here->HSM1_cgdb;
	cgdb = here->HSM1_cgsb;
	  
	cbgb = here->HSM1_cbgb;
	cbsb = here->HSM1_cbdb;
	cbdb = here->HSM1_cbsb;
	
	cdgb = -(here->HSM1_cdgb + cggb + cbgb);
	cdsb = -(here->HSM1_cddb + cgsb + cbsb);
	cddb = -(here->HSM1_cdsb + cgdb + cbdb);
      }
      
      gdpr = here->HSM1drainConductance;
      gspr = here->HSM1sourceConductance;
      gds = here->HSM1_gds;
      gbd = here->HSM1_gbd;
      gbs = here->HSM1_gbs;
      capbd = here->HSM1_capbd;
      capbs = here->HSM1_capbs;
      
      cgso = here->HSM1_cgso;
      cgdo = here->HSM1_cgdo;
      cgbo = here->HSM1_cgbo;
      
      xcdgb = (cdgb - cgdo);
      xcddb = (cddb + capbd + cgdo);
      xcdsb = cdsb;
      xcdbb = -(xcdgb + xcddb + xcdsb);
      xcsgb = -(cggb + cbgb + cdgb + cgso);
      xcsdb = -(cgdb + cbdb + cddb);
      xcssb = (capbs + cgso - (cgsb + cbsb + cdsb));
      xcsbb = -(xcsgb + xcsdb + xcssb); 
      xcggb = (cggb + cgdo + cgso + cgbo);
      xcgdb = (cgdb - cgdo);
      xcgsb = (cgsb - cgso);
      xcgbb = -(xcggb + xcgdb + xcgsb);
      xcbgb = (cbgb - cgbo);
      xcbdb = (cbdb - capbd);
      xcbsb = (cbsb - capbs);
      xcbbb = -(xcbgb + xcbdb + xcbsb);
      
      m = here->HSM1_m;
      
      *(here->HSM1GgPtr )     += m * xcggb * s->real;
      *(here->HSM1GgPtr +1)   += m * xcggb * s->imag;
      *(here->HSM1GgPtr)      += m * gIgtotg;
      *(here->HSM1BbPtr )     += m * xcbbb * s->real;
      *(here->HSM1BbPtr +1)   += m * xcbbb * s->imag;
      *(here->HSM1GbPtr)      += m * gIgtotb;
      *(here->HSM1DPdpPtr )   += m * xcddb * s->real;
      *(here->HSM1DPdpPtr +1) += m * xcddb * s->imag;
      *(here->HSM1SPspPtr )   += m * xcssb * s->real;
      *(here->HSM1SPspPtr +1) += m * xcssb * s->imag;
      
      *(here->HSM1GbPtr )     += m * xcgbb * s->real;
      *(here->HSM1GbPtr +1)   += m * xcgbb * s->imag;
      *(here->HSM1GdpPtr )    += m * xcgdb * s->real;
      *(here->HSM1GdpPtr +1)  += m * xcgdb * s->imag;
      *(here->HSM1GdpPtr)     += m * gIgtotd;
      *(here->HSM1GspPtr )    += m * xcgsb * s->real;
      *(here->HSM1GspPtr +1)  += m * xcgsb * s->imag;
      *(here->HSM1GspPtr)     += m * gIgtots;
      
      *(here->HSM1BgPtr )     += m * xcbgb * s->real;
      *(here->HSM1BgPtr +1)   += m * xcbgb * s->imag;
      *(here->HSM1BdpPtr )    += m * xcbdb * s->real;
      *(here->HSM1BdpPtr +1)  += m * xcbdb * s->imag;
      *(here->HSM1BspPtr )    += m * xcbsb * s->real;
      *(here->HSM1BspPtr +1)  += m * xcbsb * s->imag;
      
      *(here->HSM1DPgPtr )    += m * xcdgb * s->real;
      *(here->HSM1DPgPtr +1)  += m * xcdgb * s->imag;
      *(here->HSM1DPbPtr )    += m * xcdbb * s->real;
      *(here->HSM1DPbPtr +1)  += m * xcdbb * s->imag;
      *(here->HSM1DPspPtr )   += m * xcdsb * s->real;
      *(here->HSM1DPspPtr +1) += m * xcdsb * s->imag;
      
      *(here->HSM1SPgPtr )    += m * xcsgb * s->real;
      *(here->HSM1SPgPtr +1)  += m * xcsgb * s->imag;
      *(here->HSM1SPbPtr )    += m * xcsbb * s->real;
      *(here->HSM1SPbPtr +1)  += m * xcsbb * s->imag;
      *(here->HSM1SPdpPtr )   += m * xcsdb * s->real;
      *(here->HSM1SPdpPtr +1) += m * xcsdb * s->imag;
      
      *(here->HSM1DdPtr)      += m * gdpr;
      *(here->HSM1DdpPtr)     -= m * gdpr;
      *(here->HSM1DPdPtr)     -= m * gdpr;
      
      *(here->HSM1SsPtr)      += m * gspr;
      *(here->HSM1SspPtr)     -= m * gspr;
      *(here->HSM1SPsPtr)     -= m * gspr;
      
      *(here->HSM1BgPtr)      -= m * (here->HSM1_gbgs + gIbtotg);
      *(here->HSM1BbPtr)      += m * (gbd + gbs - here->HSM1_gbbs - gIbtotb);
      *(here->HSM1BdpPtr)     -= m * (gbd - gbbdp + gIbtotd);
      *(here->HSM1BspPtr)     -= m * (gbs - gbbsp + gIbtots);
      
      *(here->HSM1DPgPtr)     += m * (gm + gbdpg - gIdtotg);
      *(here->HSM1DPdpPtr)    += m * (gdpr + gds + gbd + RevSum + gbdpdp 
                                      - gIdtotd);
      *(here->HSM1DPspPtr)    -= m * (gds + FwdSum - gbdpsp + gIdtots);
      *(here->HSM1DPbPtr)     -= m * (gbd - gmbs - gbdpb + gIdtotb);
      
      *(here->HSM1SPgPtr)     -= m * (gm - gbspg + gIstotg);
      *(here->HSM1SPspPtr)    += m * (gspr + gds + gbs + FwdSum + gbspsp 
                                      - gIstots);
      *(here->HSM1SPbPtr)     -= m * (gbs + gmbs - gbspb + gIstotb);
      *(here->HSM1SPdpPtr)    -= m * (gds + RevSum - gbspdp + gIstotd);

      /* stamp gidl */
      *(here->HSM1DPdpPtr)    += m * here->HSM1_gigidlds;
      *(here->HSM1DPgPtr)     += m * here->HSM1_gigidlgs;
      *(here->HSM1DPspPtr)    -= m * (here->HSM1_gigidlgs + 
			              here->HSM1_gigidlds + here->HSM1_gigidlbs);
      *(here->HSM1DPbPtr)     += m * here->HSM1_gigidlbs;
      *(here->HSM1BdpPtr)     -= m * here->HSM1_gigidlds;
      *(here->HSM1BgPtr)      -= m * here->HSM1_gigidlgs;
      *(here->HSM1BspPtr)     += m * (here->HSM1_gigidlgs + 
			              here->HSM1_gigidlds + here->HSM1_gigidlbs);
      *(here->HSM1BbPtr)      -= m * here->HSM1_gigidlbs;
      /* stamp gisl */
      *(here->HSM1SPdpPtr)    -= m * (here->HSM1_gigislsd + 
			              here->HSM1_gigislgd + here->HSM1_gigislbd);
      *(here->HSM1SPgPtr)     += m * here->HSM1_gigislgd;
      *(here->HSM1SPspPtr)    += m * here->HSM1_gigislsd;
      *(here->HSM1SPbPtr)     += m * here->HSM1_gigislbd;
      *(here->HSM1BdpPtr)     += m * (here->HSM1_gigislgd + 
			              here->HSM1_gigislsd + here->HSM1_gigislbd);
      *(here->HSM1BgPtr)      -= m * here->HSM1_gigislgd;
      *(here->HSM1BspPtr)     -= m * here->HSM1_gigislsd;
      *(here->HSM1BbPtr)      -= m * here->HSM1_gigislbd;      

      /*	
       *(here->HSM1GgPtr)  -= m * xgtg;
       *(here->HSM1GbPtr)  -= m * xgtb;
       *(here->HSM1GdpPtr) -= m * xgtd;
       *(here->HSM1GspPtr) -= m * xgts;
       */
      
    }
  }
  return(OK);
}

