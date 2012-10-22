/**** BSIM4.4.0  Released by Xuemei (Jane) Xi 03/04/2004 ****/

/**********
 * Copyright 2004 Regents of the University of California. All rights reserved.
 * File: b4pzld.c of BSIM4.4.0.
 * Author: 2000 Weidong Liu
 * Authors: 2001- Xuemei Xi, Jin He, Kanyu Cao, Mohan Dunga, Mansun Chan, Ali Niknejad, Chenming Hu.
 * Project Director: Prof. Chenming Hu.
 * Modified by Xuemei Xi, 10/05/2001.
 **********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ngspice/complex.h"
#include "ngspice/sperror.h"
#include "bsim4v4def.h"
#include "ngspice/suffix.h"

int
BSIM4v4pzLoad(
GENmodel *inModel,
CKTcircuit *ckt,
SPcomplex *s)
{
BSIM4v4model *model = (BSIM4v4model*)inModel;
BSIM4v4instance *here;

double gjbd, gjbs, geltd, gcrg, gcrgg, gcrgd, gcrgs, gcrgb;
double xcggb, xcgdb, xcgsb, xcgbb, xcbgb, xcbdb, xcbsb, xcbbb;
double xcdgb, xcddb, xcdsb, xcdbb, xcsgb, xcsdb, xcssb, xcsbb;
double gds, capbd, capbs, FwdSum, RevSum, Gm, Gmbs;
double gstot, gstotd, gstotg, gstots, gstotb, gspr;
double gdtot, gdtotd, gdtotg, gdtots, gdtotb, gdpr;
double gIstotg, gIstotd, gIstots, gIstotb;
double gIdtotg, gIdtotd, gIdtots, gIdtotb;
double gIbtotg, gIbtotd, gIbtots, gIbtotb;
double gIgtotg, gIgtotd, gIgtots, gIgtotb;
double cgso, cgdo, cgbo;
double xcdbdb=0.0, xcsbsb=0.0, xcgmgmb=0.0, xcgmdb=0.0, xcgmsb=0.0, xcdgmb=0.0, xcsgmb=0.0;
double xcgmbb=0.0, xcbgmb=0.0;
double dxpart, sxpart, xgtg, xgtd, xgts, xgtb, xcqgb=0.0, xcqdb=0.0, xcqsb=0.0, xcqbb=0.0;
double gbspsp, gbbdp, gbbsp, gbspg, gbspb;
double gbspdp, gbdpdp, gbdpg, gbdpb, gbdpsp;
double ddxpart_dVd, ddxpart_dVg, ddxpart_dVb, ddxpart_dVs;
double dsxpart_dVd, dsxpart_dVg, dsxpart_dVb, dsxpart_dVs;
double T0=0.0, T1, CoxWL, qcheq, Cdg, Cdd, Cds, Csg, Csd, Css;
double ScalingFactor = 1.0e-9;
struct bsim4SizeDependParam *pParam;
double ggidld, ggidlg, ggidlb, ggislg, ggislb, ggisls;

double m;

    for (; model != NULL; model = model->BSIM4v4nextModel) 
    {    for (here = model->BSIM4v4instances; here!= NULL;
              here = here->BSIM4v4nextInstance) 
	       {
	            pParam = here->pParam;
              capbd = here->BSIM4v4capbd;
              capbs = here->BSIM4v4capbs;
              cgso = here->BSIM4v4cgso;
              cgdo = here->BSIM4v4cgdo;
              cgbo = pParam->BSIM4v4cgbo;

              if (here->BSIM4v4mode >= 0) 
              {   Gm = here->BSIM4v4gm;
                  Gmbs = here->BSIM4v4gmbs;
                  FwdSum = Gm + Gmbs;
                  RevSum = 0.0;

                  gbbdp = -(here->BSIM4v4gbds);
                  gbbsp = here->BSIM4v4gbds + here->BSIM4v4gbgs + here->BSIM4v4gbbs;
                  gbdpg = here->BSIM4v4gbgs;
                  gbdpdp = here->BSIM4v4gbds;
                  gbdpb = here->BSIM4v4gbbs;
                  gbdpsp = -(gbdpg + gbdpdp + gbdpb);

                  gbspdp = 0.0;
                  gbspg = 0.0;
                  gbspb = 0.0;
                  gbspsp = 0.0;

                  if (model->BSIM4v4igcMod)
                  {   gIstotg = here->BSIM4v4gIgsg + here->BSIM4v4gIgcsg;
                      gIstotd = here->BSIM4v4gIgcsd;
                      gIstots = here->BSIM4v4gIgss + here->BSIM4v4gIgcss;
                      gIstotb = here->BSIM4v4gIgcsb;

                      gIdtotg = here->BSIM4v4gIgdg + here->BSIM4v4gIgcdg;
                      gIdtotd = here->BSIM4v4gIgdd + here->BSIM4v4gIgcdd;
                      gIdtots = here->BSIM4v4gIgcds;
                      gIdtotb = here->BSIM4v4gIgcdb;
                  }
                  else
                  {   gIstotg = gIstotd = gIstots = gIstotb = 0.0;
                      gIdtotg = gIdtotd = gIdtots = gIdtotb = 0.0;
                  }

                  if (model->BSIM4v4igbMod)
                  {   gIbtotg = here->BSIM4v4gIgbg;
                      gIbtotd = here->BSIM4v4gIgbd;
                      gIbtots = here->BSIM4v4gIgbs;
                      gIbtotb = here->BSIM4v4gIgbb;
                  }
                  else
                      gIbtotg = gIbtotd = gIbtots = gIbtotb = 0.0;

                  if ((model->BSIM4v4igcMod != 0) || (model->BSIM4v4igbMod != 0))
                  {   gIgtotg = gIstotg + gIdtotg + gIbtotg;
                      gIgtotd = gIstotd + gIdtotd + gIbtotd ;
                      gIgtots = gIstots + gIdtots + gIbtots;
                      gIgtotb = gIstotb + gIdtotb + gIbtotb;
                  }
                  else
                      gIgtotg = gIgtotd = gIgtots = gIgtotb = 0.0;

                  if (here->BSIM4v4rgateMod == 2)
                      T0 = *(ckt->CKTstates[0] + here->BSIM4v4vges)
                         - *(ckt->CKTstates[0] + here->BSIM4v4vgs);
                  else if (here->BSIM4v4rgateMod == 3)
                      T0 = *(ckt->CKTstates[0] + here->BSIM4v4vgms)
                         - *(ckt->CKTstates[0] + here->BSIM4v4vgs);
                  if (here->BSIM4v4rgateMod > 1)
                  {   gcrgd = here->BSIM4v4gcrgd * T0;
                      gcrgg = here->BSIM4v4gcrgg * T0;
                      gcrgs = here->BSIM4v4gcrgs * T0;
                      gcrgb = here->BSIM4v4gcrgb * T0;
                      gcrgg -= here->BSIM4v4gcrg;
                      gcrg = here->BSIM4v4gcrg;
                  }
                  else
                      gcrg = gcrgd = gcrgg = gcrgs = gcrgb = 0.0;

                  if (here->BSIM4v4acnqsMod == 0)
                  {   if (here->BSIM4v4rgateMod == 3)
                      {   xcgmgmb = cgdo + cgso + pParam->BSIM4v4cgbo;
                          xcgmdb = -cgdo;
                          xcgmsb = -cgso;
                          xcgmbb = -pParam->BSIM4v4cgbo;

                          xcdgmb = xcgmdb;
                          xcsgmb = xcgmsb;
                          xcbgmb = xcgmbb;

                          xcggb = here->BSIM4v4cggb;
                          xcgdb = here->BSIM4v4cgdb;
                          xcgsb = here->BSIM4v4cgsb;
                          xcgbb = -(xcggb + xcgdb + xcgsb);

                          xcdgb = here->BSIM4v4cdgb;
                          xcsgb = -(here->BSIM4v4cggb + here->BSIM4v4cbgb
                                + here->BSIM4v4cdgb);
                          xcbgb = here->BSIM4v4cbgb;
                      }
                      else
                      {   xcggb = here->BSIM4v4cggb + cgdo + cgso
                                + pParam->BSIM4v4cgbo;
                          xcgdb = here->BSIM4v4cgdb - cgdo;
                          xcgsb = here->BSIM4v4cgsb - cgso;
                          xcgbb = -(xcggb + xcgdb + xcgsb);

                          xcdgb = here->BSIM4v4cdgb - cgdo;
                          xcsgb = -(here->BSIM4v4cggb + here->BSIM4v4cbgb
                                + here->BSIM4v4cdgb + cgso);
                          xcbgb = here->BSIM4v4cbgb - pParam->BSIM4v4cgbo;

                          xcdgmb = xcsgmb = xcbgmb = 0.0;
                      }
                      xcddb = here->BSIM4v4cddb + here->BSIM4v4capbd + cgdo;
                      xcdsb = here->BSIM4v4cdsb;

                      xcsdb = -(here->BSIM4v4cgdb + here->BSIM4v4cbdb
                            + here->BSIM4v4cddb);
                      xcssb = here->BSIM4v4capbs + cgso - (here->BSIM4v4cgsb
                            + here->BSIM4v4cbsb + here->BSIM4v4cdsb);

                      if (!here->BSIM4v4rbodyMod)
                      {   xcdbb = -(xcdgb + xcddb + xcdsb + xcdgmb);
                          xcsbb = -(xcsgb + xcsdb + xcssb + xcsgmb);
                          xcbdb = here->BSIM4v4cbdb - here->BSIM4v4capbd;
                          xcbsb = here->BSIM4v4cbsb - here->BSIM4v4capbs;
                          xcdbdb = 0.0;
                      }
                      else
                      {   xcdbb  = -(here->BSIM4v4cddb + here->BSIM4v4cdgb
                                 + here->BSIM4v4cdsb);
                          xcsbb = -(xcsgb + xcsdb + xcssb + xcsgmb)
                                + here->BSIM4v4capbs;
                          xcbdb = here->BSIM4v4cbdb;
                          xcbsb = here->BSIM4v4cbsb;

                          xcdbdb = -here->BSIM4v4capbd;
                          xcsbsb = -here->BSIM4v4capbs;
                      }
                      xcbbb = -(xcbdb + xcbgb + xcbsb + xcbgmb);

                      xgtg = xgtd = xgts = xgtb = 0.0;
		      sxpart = 0.6;
                      dxpart = 0.4;
		      ddxpart_dVd = ddxpart_dVg = ddxpart_dVb 
				  = ddxpart_dVs = 0.0;
		      dsxpart_dVd = dsxpart_dVg = dsxpart_dVb 
				  = dsxpart_dVs = 0.0;
                  }
                  else
                  {   xcggb = xcgdb = xcgsb = xcgbb = 0.0;
                      xcbgb = xcbdb = xcbsb = xcbbb = 0.0;
                      xcdgb = xcddb = xcdsb = xcdbb = 0.0;
                      xcsgb = xcsdb = xcssb = xcsbb = 0.0;

		      xgtg = here->BSIM4v4gtg;
                      xgtd = here->BSIM4v4gtd;
                      xgts = here->BSIM4v4gts;
                      xgtb = here->BSIM4v4gtb;

                      xcqgb = here->BSIM4v4cqgb;
                      xcqdb = here->BSIM4v4cqdb;
                      xcqsb = here->BSIM4v4cqsb;
                      xcqbb = here->BSIM4v4cqbb;

		      CoxWL = model->BSIM4v4coxe * here->pParam->BSIM4v4weffCV
                            * here->BSIM4v4nf * here->pParam->BSIM4v4leffCV;
		      qcheq = -(here->BSIM4v4qgate + here->BSIM4v4qbulk);
		      if (fabs(qcheq) <= 1.0e-5 * CoxWL)
		      {   if (model->BSIM4v4xpart < 0.5)
		          {   dxpart = 0.4;
		          }
		          else if (model->BSIM4v4xpart > 0.5)
		          {   dxpart = 0.0;
		          }
		          else
		          {   dxpart = 0.5;
		          }
		          ddxpart_dVd = ddxpart_dVg = ddxpart_dVb
				      = ddxpart_dVs = 0.0;
		      }
		      else
		      {   dxpart = here->BSIM4v4qdrn / qcheq;
		          Cdd = here->BSIM4v4cddb;
		          Csd = -(here->BSIM4v4cgdb + here->BSIM4v4cddb
			      + here->BSIM4v4cbdb);
		          ddxpart_dVd = (Cdd - dxpart * (Cdd + Csd)) / qcheq;
		          Cdg = here->BSIM4v4cdgb;
		          Csg = -(here->BSIM4v4cggb + here->BSIM4v4cdgb
			      + here->BSIM4v4cbgb);
		          ddxpart_dVg = (Cdg - dxpart * (Cdg + Csg)) / qcheq;

		          Cds = here->BSIM4v4cdsb;
		          Css = -(here->BSIM4v4cgsb + here->BSIM4v4cdsb
			      + here->BSIM4v4cbsb);
		          ddxpart_dVs = (Cds - dxpart * (Cds + Css)) / qcheq;

		          ddxpart_dVb = -(ddxpart_dVd + ddxpart_dVg 
				      + ddxpart_dVs);
		      }
		      sxpart = 1.0 - dxpart;
		      dsxpart_dVd = -ddxpart_dVd;
		      dsxpart_dVg = -ddxpart_dVg;
		      dsxpart_dVs = -ddxpart_dVs;
		      dsxpart_dVb = -(dsxpart_dVd + dsxpart_dVg + dsxpart_dVs);
                  }
              }
              else
              {   Gm = -here->BSIM4v4gm;
                  Gmbs = -here->BSIM4v4gmbs;
                  FwdSum = 0.0;
                  RevSum = -(Gm + Gmbs);

                  gbbsp = -(here->BSIM4v4gbds);
                  gbbdp = here->BSIM4v4gbds + here->BSIM4v4gbgs + here->BSIM4v4gbbs;

                  gbdpg = 0.0;
                  gbdpsp = 0.0;
                  gbdpb = 0.0;
                  gbdpdp = 0.0;

                  gbspg = here->BSIM4v4gbgs;
                  gbspsp = here->BSIM4v4gbds;
                  gbspb = here->BSIM4v4gbbs;
                  gbspdp = -(gbspg + gbspsp + gbspb);

                  if (model->BSIM4v4igcMod)
                  {   gIstotg = here->BSIM4v4gIgsg + here->BSIM4v4gIgcdg;
                      gIstotd = here->BSIM4v4gIgcds;
                      gIstots = here->BSIM4v4gIgss + here->BSIM4v4gIgcdd;
                      gIstotb = here->BSIM4v4gIgcdb;

                      gIdtotg = here->BSIM4v4gIgdg + here->BSIM4v4gIgcsg;
                      gIdtotd = here->BSIM4v4gIgdd + here->BSIM4v4gIgcss;
                      gIdtots = here->BSIM4v4gIgcsd;
                      gIdtotb = here->BSIM4v4gIgcsb;
                  }
                  else
                  {   gIstotg = gIstotd = gIstots = gIstotb = 0.0;
                      gIdtotg = gIdtotd = gIdtots = gIdtotb  = 0.0;
                  }

                  if (model->BSIM4v4igbMod)
                  {   gIbtotg = here->BSIM4v4gIgbg;
                      gIbtotd = here->BSIM4v4gIgbs;
                      gIbtots = here->BSIM4v4gIgbd;
                      gIbtotb = here->BSIM4v4gIgbb;
                  }
                  else
                      gIbtotg = gIbtotd = gIbtots = gIbtotb = 0.0;

                  if ((model->BSIM4v4igcMod != 0) || (model->BSIM4v4igbMod != 0))
                  {   gIgtotg = gIstotg + gIdtotg + gIbtotg;
                      gIgtotd = gIstotd + gIdtotd + gIbtotd ;
                      gIgtots = gIstots + gIdtots + gIbtots;
                      gIgtotb = gIstotb + gIdtotb + gIbtotb;
                  }
                  else
                      gIgtotg = gIgtotd = gIgtots = gIgtotb = 0.0;

                  if (here->BSIM4v4rgateMod == 2)
                      T0 = *(ckt->CKTstates[0] + here->BSIM4v4vges)
                         - *(ckt->CKTstates[0] + here->BSIM4v4vgs);
                  else if (here->BSIM4v4rgateMod == 3)
                      T0 = *(ckt->CKTstates[0] + here->BSIM4v4vgms)
                         - *(ckt->CKTstates[0] + here->BSIM4v4vgs);
                  if (here->BSIM4v4rgateMod > 1)
                  {   gcrgd = here->BSIM4v4gcrgs * T0;
                      gcrgg = here->BSIM4v4gcrgg * T0;
                      gcrgs = here->BSIM4v4gcrgd * T0;
                      gcrgb = here->BSIM4v4gcrgb * T0;
                      gcrgg -= here->BSIM4v4gcrg;
                      gcrg = here->BSIM4v4gcrg;
                  }
                  else
                      gcrg = gcrgd = gcrgg = gcrgs = gcrgb = 0.0;

                  if (here->BSIM4v4acnqsMod == 0)
                  {   if (here->BSIM4v4rgateMod == 3)
                      {   xcgmgmb = cgdo + cgso + pParam->BSIM4v4cgbo;
                          xcgmdb = -cgdo;
                          xcgmsb = -cgso;
                          xcgmbb = -pParam->BSIM4v4cgbo;
   
                          xcdgmb = xcgmdb;
                          xcsgmb = xcgmsb;
                          xcbgmb = xcgmbb;

                          xcggb = here->BSIM4v4cggb;
                          xcgdb = here->BSIM4v4cgsb;
                          xcgsb = here->BSIM4v4cgdb;
                          xcgbb = -(xcggb + xcgdb + xcgsb);

                          xcdgb = -(here->BSIM4v4cggb + here->BSIM4v4cbgb
                                + here->BSIM4v4cdgb);
                          xcsgb = here->BSIM4v4cdgb;
                          xcbgb = here->BSIM4v4cbgb;
                      }
                      else
                      {   xcggb = here->BSIM4v4cggb + cgdo + cgso
                                + pParam->BSIM4v4cgbo;
                          xcgdb = here->BSIM4v4cgsb - cgdo;
                          xcgsb = here->BSIM4v4cgdb - cgso;
                          xcgbb = -(xcggb + xcgdb + xcgsb);

                          xcdgb = -(here->BSIM4v4cggb + here->BSIM4v4cbgb
                                + here->BSIM4v4cdgb + cgdo);
                          xcsgb = here->BSIM4v4cdgb - cgso;
                          xcbgb = here->BSIM4v4cbgb - pParam->BSIM4v4cgbo;

                          xcdgmb = xcsgmb = xcbgmb = 0.0;
                      }
                      xcddb = here->BSIM4v4capbd + cgdo - (here->BSIM4v4cgsb
                            + here->BSIM4v4cbsb + here->BSIM4v4cdsb);
                      xcdsb = -(here->BSIM4v4cgdb + here->BSIM4v4cbdb
                            + here->BSIM4v4cddb);

                      xcsdb = here->BSIM4v4cdsb;
                      xcssb = here->BSIM4v4cddb + here->BSIM4v4capbs + cgso;

                      if (!here->BSIM4v4rbodyMod)
                      {   xcdbb = -(xcdgb + xcddb + xcdsb + xcdgmb);
                          xcsbb = -(xcsgb + xcsdb + xcssb + xcsgmb);
                          xcbdb = here->BSIM4v4cbsb - here->BSIM4v4capbd;
                          xcbsb = here->BSIM4v4cbdb - here->BSIM4v4capbs;
                          xcdbdb = 0.0;
                      }
                      else
                      {   xcdbb = -(xcdgb + xcddb + xcdsb + xcdgmb)
                                + here->BSIM4v4capbd;
                          xcsbb = -(here->BSIM4v4cddb + here->BSIM4v4cdgb
                                + here->BSIM4v4cdsb);
                          xcbdb = here->BSIM4v4cbsb;
                          xcbsb = here->BSIM4v4cbdb;
                          xcdbdb = -here->BSIM4v4capbd;
                          xcsbsb = -here->BSIM4v4capbs;
                      }
                      xcbbb = -(xcbgb + xcbdb + xcbsb + xcbgmb);

                      xgtg = xgtd = xgts = xgtb = 0.0;
		      sxpart = 0.4;
                      dxpart = 0.6;
		      ddxpart_dVd = ddxpart_dVg = ddxpart_dVb 
				  = ddxpart_dVs = 0.0;
		      dsxpart_dVd = dsxpart_dVg = dsxpart_dVb 
				  = dsxpart_dVs = 0.0;
                  }
                  else
                  {   xcggb = xcgdb = xcgsb = xcgbb = 0.0;
                      xcbgb = xcbdb = xcbsb = xcbbb = 0.0;
                      xcdgb = xcddb = xcdsb = xcdbb = 0.0;
                      xcsgb = xcsdb = xcssb = xcsbb = 0.0;

		      xgtg = here->BSIM4v4gtg;
                      xgtd = here->BSIM4v4gts;
                      xgts = here->BSIM4v4gtd;
                      xgtb = here->BSIM4v4gtb;

                      xcqgb = here->BSIM4v4cqgb;
                      xcqdb = here->BSIM4v4cqsb;
                      xcqsb = here->BSIM4v4cqdb;
                      xcqbb = here->BSIM4v4cqbb;

		      CoxWL = model->BSIM4v4coxe * here->pParam->BSIM4v4weffCV
                            * here->BSIM4v4nf * here->pParam->BSIM4v4leffCV;
		      qcheq = -(here->BSIM4v4qgate + here->BSIM4v4qbulk);
		      if (fabs(qcheq) <= 1.0e-5 * CoxWL)
		      {   if (model->BSIM4v4xpart < 0.5)
		          {   sxpart = 0.4;
		          }
		          else if (model->BSIM4v4xpart > 0.5)
		          {   sxpart = 0.0;
		          }
		          else
		          {   sxpart = 0.5;
		          }
		          dsxpart_dVd = dsxpart_dVg = dsxpart_dVb
				      = dsxpart_dVs = 0.0;
		      }
		      else
		      {   sxpart = here->BSIM4v4qdrn / qcheq;
		          Css = here->BSIM4v4cddb;
		          Cds = -(here->BSIM4v4cgdb + here->BSIM4v4cddb
			      + here->BSIM4v4cbdb);
		          dsxpart_dVs = (Css - sxpart * (Css + Cds)) / qcheq;
		          Csg = here->BSIM4v4cdgb;
		          Cdg = -(here->BSIM4v4cggb + here->BSIM4v4cdgb
			      + here->BSIM4v4cbgb);
		          dsxpart_dVg = (Csg - sxpart * (Csg + Cdg)) / qcheq;

		          Csd = here->BSIM4v4cdsb;
		          Cdd = -(here->BSIM4v4cgsb + here->BSIM4v4cdsb
			      + here->BSIM4v4cbsb);
		          dsxpart_dVd = (Csd - sxpart * (Csd + Cdd)) / qcheq;

		          dsxpart_dVb = -(dsxpart_dVd + dsxpart_dVg 
				      + dsxpart_dVs);
		      }
		      dxpart = 1.0 - sxpart;
		      ddxpart_dVd = -dsxpart_dVd;
		      ddxpart_dVg = -dsxpart_dVg;
		      ddxpart_dVs = -dsxpart_dVs;
		      ddxpart_dVb = -(ddxpart_dVd + ddxpart_dVg + ddxpart_dVs);
                  }
              }

              if (model->BSIM4v4rdsMod == 1)
              {   gstot = here->BSIM4v4gstot;
                  gstotd = here->BSIM4v4gstotd;
                  gstotg = here->BSIM4v4gstotg;
                  gstots = here->BSIM4v4gstots - gstot;
                  gstotb = here->BSIM4v4gstotb;

                  gdtot = here->BSIM4v4gdtot;
                  gdtotd = here->BSIM4v4gdtotd - gdtot;
                  gdtotg = here->BSIM4v4gdtotg;
                  gdtots = here->BSIM4v4gdtots;
                  gdtotb = here->BSIM4v4gdtotb;
              }
              else
              {   gstot = gstotd = gstotg = gstots = gstotb = 0.0;
                  gdtot = gdtotd = gdtotg = gdtots = gdtotb = 0.0;
              }


	      T1 = *(ckt->CKTstate0 + here->BSIM4v4qdef) * here->BSIM4v4gtau;
              gds = here->BSIM4v4gds;

              /*
               * Loading PZ matrix
               */

              m = here->BSIM4v4m;

              if (!model->BSIM4v4rdsMod)
              {   gdpr = here->BSIM4v4drainConductance;
                  gspr = here->BSIM4v4sourceConductance;
              }
              else
                  gdpr = gspr = 0.0;

              if (!here->BSIM4v4rbodyMod)
              {   gjbd = here->BSIM4v4gbd;
                  gjbs = here->BSIM4v4gbs;
              }
              else
                  gjbd = gjbs = 0.0;

              geltd = here->BSIM4v4grgeltd;

              if (here->BSIM4v4rgateMod == 1)
              {   *(here->BSIM4v4GEgePtr) += m * geltd;
                  *(here->BSIM4v4GPgePtr) -= m * geltd;
                  *(here->BSIM4v4GEgpPtr) -= m * geltd;

                  *(here->BSIM4v4GPgpPtr ) += m * xcggb * s->real;
                  *(here->BSIM4v4GPgpPtr +1) += m * xcggb * s->imag;
                  *(here->BSIM4v4GPgpPtr) += m * (geltd - xgtg + gIgtotg);
                  *(here->BSIM4v4GPdpPtr ) += m * xcgdb * s->real;
                  *(here->BSIM4v4GPdpPtr +1) += m * xcgdb * s->imag;
		  *(here->BSIM4v4GPdpPtr) -= m * (xgtd - gIgtotd);
                  *(here->BSIM4v4GPspPtr ) += m * xcgsb * s->real;
                  *(here->BSIM4v4GPspPtr +1) += m * xcgsb * s->imag;
                  *(here->BSIM4v4GPspPtr) -= m * (xgts - gIgtots);
                  *(here->BSIM4v4GPbpPtr ) += m * xcgbb * s->real;
                  *(here->BSIM4v4GPbpPtr +1) += m * xcgbb * s->imag;
		  *(here->BSIM4v4GPbpPtr) -= m * (xgtb - gIgtotb);
              }
              else if (here->BSIM4v4rgateMod == 2)
              {   *(here->BSIM4v4GEgePtr) += m * gcrg;
                  *(here->BSIM4v4GEgpPtr) += m * gcrgg;
                  *(here->BSIM4v4GEdpPtr) += m * gcrgd;
                  *(here->BSIM4v4GEspPtr) += m * gcrgs;
                  *(here->BSIM4v4GEbpPtr) += m * gcrgb;

                  *(here->BSIM4v4GPgePtr) -= m * gcrg;
                  *(here->BSIM4v4GPgpPtr ) += m * xcggb * s->real;
                  *(here->BSIM4v4GPgpPtr +1) += m * xcggb * s->imag;
                  *(here->BSIM4v4GPgpPtr) -= m * (gcrgg + xgtg - gIgtotg);
                  *(here->BSIM4v4GPdpPtr ) += m * xcgdb * s->real;
                  *(here->BSIM4v4GPdpPtr +1) += m * xcgdb * s->imag;
                  *(here->BSIM4v4GPdpPtr) -= m * (gcrgd + xgtd - gIgtotd);
                  *(here->BSIM4v4GPspPtr ) += m * xcgsb * s->real;
                  *(here->BSIM4v4GPspPtr +1) += m * xcgsb * s->imag;
                  *(here->BSIM4v4GPspPtr) -= m * (gcrgs + xgts - gIgtots);
                  *(here->BSIM4v4GPbpPtr ) += m * xcgbb * s->real;
                  *(here->BSIM4v4GPbpPtr +1) += m * xcgbb * s->imag;
                  *(here->BSIM4v4GPbpPtr) -= m * (gcrgb + xgtb - gIgtotb);
              }
              else if (here->BSIM4v4rgateMod == 3)
              {   *(here->BSIM4v4GEgePtr) += m * geltd;
                  *(here->BSIM4v4GEgmPtr) -= m * geltd;
                  *(here->BSIM4v4GMgePtr) -= m * geltd;
                  *(here->BSIM4v4GMgmPtr) += m * (geltd + gcrg);
                  *(here->BSIM4v4GMgmPtr ) += m * xcgmgmb * s->real;
                  *(here->BSIM4v4GMgmPtr +1) += m * xcgmgmb * s->imag;
  
                  *(here->BSIM4v4GMdpPtr) += m * gcrgd;
                  *(here->BSIM4v4GMdpPtr ) += m * xcgmdb * s->real;
                  *(here->BSIM4v4GMdpPtr +1) += m * xcgmdb * s->imag;
                  *(here->BSIM4v4GMgpPtr) += m * gcrgg;
                  *(here->BSIM4v4GMspPtr) += m * gcrgs;
                  *(here->BSIM4v4GMspPtr ) += m * xcgmsb * s->real;
                  *(here->BSIM4v4GMspPtr +1) += m * xcgmsb * s->imag;
                  *(here->BSIM4v4GMbpPtr) += m * gcrgb;
                  *(here->BSIM4v4GMbpPtr ) += m * xcgmbb * s->real;
                  *(here->BSIM4v4GMbpPtr +1) += m * xcgmbb * s->imag;
  
                  *(here->BSIM4v4DPgmPtr ) += m * xcdgmb * s->real;
                  *(here->BSIM4v4DPgmPtr +1) += m * xcdgmb * s->imag;
                  *(here->BSIM4v4GPgmPtr) -= m * gcrg;
                  *(here->BSIM4v4SPgmPtr ) += m * xcsgmb * s->real;
                  *(here->BSIM4v4SPgmPtr +1) += m * xcsgmb * s->imag;
                  *(here->BSIM4v4BPgmPtr ) += m * xcbgmb * s->real;
                  *(here->BSIM4v4BPgmPtr +1) += m * xcbgmb * s->imag;
  
                  *(here->BSIM4v4GPgpPtr) -= m * (gcrgg + xgtg - gIgtotg);
                  *(here->BSIM4v4GPgpPtr ) += m * xcggb * s->real;
                  *(here->BSIM4v4GPgpPtr +1) += m * xcggb * s->imag;
                  *(here->BSIM4v4GPdpPtr) -= m * (gcrgd + xgtd - gIgtotd);
                  *(here->BSIM4v4GPdpPtr ) += m * xcgdb * s->real;
                  *(here->BSIM4v4GPdpPtr +1) += m * xcgdb * s->imag;
                  *(here->BSIM4v4GPspPtr) -= m * (gcrgs + xgts - gIgtots);
                  *(here->BSIM4v4GPspPtr ) += m * xcgsb * s->real;
                  *(here->BSIM4v4GPspPtr +1) += m * xcgsb * s->imag;
                  *(here->BSIM4v4GPbpPtr) -= m * (gcrgb + xgtb - gIgtotb);
                  *(here->BSIM4v4GPbpPtr ) += m * xcgbb * s->real;
                  *(here->BSIM4v4GPbpPtr +1) += m * xcgbb * s->imag;
              }
              else
              {   *(here->BSIM4v4GPdpPtr ) += m * xcgdb * s->real;
                  *(here->BSIM4v4GPdpPtr +1) += m * xcgdb * s->imag;
		  *(here->BSIM4v4GPdpPtr) -= m * (xgtd - gIgtotd);
                  *(here->BSIM4v4GPgpPtr ) += m * xcggb * s->real;
                  *(here->BSIM4v4GPgpPtr +1) += m * xcggb * s->imag;
		  *(here->BSIM4v4GPgpPtr) -= m * (xgtg - gIgtotg);
                  *(here->BSIM4v4GPspPtr ) += m * xcgsb * s->real;
                  *(here->BSIM4v4GPspPtr +1) += m * xcgsb * s->imag;
                  *(here->BSIM4v4GPspPtr) -= m * (xgts - gIgtots);
                  *(here->BSIM4v4GPbpPtr ) += m * xcgbb * s->real;
                  *(here->BSIM4v4GPbpPtr +1) += m * xcgbb * s->imag;
		  *(here->BSIM4v4GPbpPtr) -= m * (xgtb - gIgtotb);
              }

              if (model->BSIM4v4rdsMod)
              {   (*(here->BSIM4v4DgpPtr) += m * gdtotg);
                  (*(here->BSIM4v4DspPtr) += m * gdtots);
                  (*(here->BSIM4v4DbpPtr) += m * gdtotb);
                  (*(here->BSIM4v4SdpPtr) += m * gstotd);
                  (*(here->BSIM4v4SgpPtr) += m * gstotg);
                  (*(here->BSIM4v4SbpPtr) += m * gstotb);
              }

              *(here->BSIM4v4DPdpPtr ) += m * xcddb * s->real;
              *(here->BSIM4v4DPdpPtr +1) += m * xcddb * s->imag;
              *(here->BSIM4v4DPdpPtr) += m * (gdpr + gds + here->BSIM4v4gbd
				     - gdtotd + RevSum + gbdpdp - gIdtotd
				     + dxpart * xgtd + T1 * ddxpart_dVd);
              *(here->BSIM4v4DPdPtr) -= m * (gdpr + gdtot);
              *(here->BSIM4v4DPgpPtr ) += m * xcdgb * s->real;
              *(here->BSIM4v4DPgpPtr +1) += m * xcdgb * s->imag;
              *(here->BSIM4v4DPgpPtr) += m * (Gm - gdtotg + gbdpg - gIdtotg
				     + T1 * ddxpart_dVg + dxpart * xgtg);
              *(here->BSIM4v4DPspPtr ) += m * xcdsb * s->real;
              *(here->BSIM4v4DPspPtr +1) += m * xcdsb * s->imag;
              *(here->BSIM4v4DPspPtr) -= m * (gds + FwdSum + gdtots - gbdpsp + gIdtots
				     - T1 * ddxpart_dVs - dxpart * xgts);
              *(here->BSIM4v4DPbpPtr ) += m * xcdbb * s->real;
              *(here->BSIM4v4DPbpPtr +1) += m * xcdbb * s->imag;
              *(here->BSIM4v4DPbpPtr) -= m * (gjbd + gdtotb - Gmbs - gbdpb + gIdtotb
				     - T1 * ddxpart_dVb - dxpart * xgtb);

              *(here->BSIM4v4DdpPtr) -= m * (gdpr - gdtotd);
              *(here->BSIM4v4DdPtr) += m * (gdpr + gdtot);

              *(here->BSIM4v4SPdpPtr ) += m * xcsdb * s->real;
              *(here->BSIM4v4SPdpPtr +1) += m * xcsdb * s->imag;
              *(here->BSIM4v4SPdpPtr) -= m * (gds + gstotd + RevSum - gbspdp + gIstotd
				     - T1 * dsxpart_dVd - sxpart * xgtd);
              *(here->BSIM4v4SPgpPtr ) += m * xcsgb * s->real;
              *(here->BSIM4v4SPgpPtr +1) += m * xcsgb * s->imag;
              *(here->BSIM4v4SPgpPtr) -= m * (Gm + gstotg - gbspg + gIstotg
				     - T1 * dsxpart_dVg - sxpart * xgtg);
              *(here->BSIM4v4SPspPtr ) += m * xcssb * s->real;
              *(here->BSIM4v4SPspPtr +1) += m * xcssb * s->imag;
              *(here->BSIM4v4SPspPtr) += m * (gspr + gds + here->BSIM4v4gbs - gIstots
				     - gstots + FwdSum + gbspsp
				     + sxpart * xgts + T1 * dsxpart_dVs);
              *(here->BSIM4v4SPsPtr) -= m * (gspr + gstot);
              *(here->BSIM4v4SPbpPtr ) += m * xcsbb * s->real;
              *(here->BSIM4v4SPbpPtr +1) += m * xcsbb * s->imag;
              *(here->BSIM4v4SPbpPtr) -= m * (gjbs + gstotb + Gmbs - gbspb + gIstotb
				     - T1 * dsxpart_dVb - sxpart * xgtb);

              *(here->BSIM4v4SspPtr) -= m * (gspr - gstots);
              *(here->BSIM4v4SsPtr) += m * (gspr + gstot);

              *(here->BSIM4v4BPdpPtr ) += m * xcbdb * s->real;
              *(here->BSIM4v4BPdpPtr +1) += m * xcbdb * s->imag;
              *(here->BSIM4v4BPdpPtr) -= m * (gjbd - gbbdp + gIbtotd);
              *(here->BSIM4v4BPgpPtr ) += m * xcbgb * s->real;
              *(here->BSIM4v4BPgpPtr +1) += m * xcbgb * s->imag;
              *(here->BSIM4v4BPgpPtr) -= m * (here->BSIM4v4gbgs + gIbtotg);
              *(here->BSIM4v4BPspPtr ) += m * xcbsb * s->real;
              *(here->BSIM4v4BPspPtr +1) += m * xcbsb * s->imag;
              *(here->BSIM4v4BPspPtr) -= m * (gjbs - gbbsp + gIbtots);
              *(here->BSIM4v4BPbpPtr ) += m * xcbbb * s->real;
              *(here->BSIM4v4BPbpPtr +1) += m * xcbbb * s->imag;
              *(here->BSIM4v4BPbpPtr) += m * (gjbd + gjbs - here->BSIM4v4gbbs
				     - gIbtotb);
           ggidld = here->BSIM4v4ggidld;
           ggidlg = here->BSIM4v4ggidlg;
           ggidlb = here->BSIM4v4ggidlb;
           ggislg = here->BSIM4v4ggislg;
           ggisls = here->BSIM4v4ggisls;
           ggislb = here->BSIM4v4ggislb;

           /* stamp gidl */
           (*(here->BSIM4v4DPdpPtr) += m * ggidld);
           (*(here->BSIM4v4DPgpPtr) += m * ggidlg);
           (*(here->BSIM4v4DPspPtr) -= m * ((ggidlg + ggidld) + ggidlb));
           (*(here->BSIM4v4DPbpPtr) += m * ggidlb);
           (*(here->BSIM4v4BPdpPtr) -= m * ggidld);
           (*(here->BSIM4v4BPgpPtr) -= m * ggidlg);
           (*(here->BSIM4v4BPspPtr) += m * ((ggidlg + ggidld) + ggidlb));
           (*(here->BSIM4v4BPbpPtr) -= m * ggidlb);
            /* stamp gisl */
           (*(here->BSIM4v4SPdpPtr) -= m * ((ggisls + ggislg) + ggislb));
           (*(here->BSIM4v4SPgpPtr) += m * ggislg);
           (*(here->BSIM4v4SPspPtr) += m * ggisls);
           (*(here->BSIM4v4SPbpPtr) += m * ggislb);
           (*(here->BSIM4v4BPdpPtr) += m * ((ggislg + ggisls) + ggislb));
           (*(here->BSIM4v4BPgpPtr) -= m * ggislg);
           (*(here->BSIM4v4BPspPtr) -= m * ggisls);
           (*(here->BSIM4v4BPbpPtr) -= m * ggislb);

              if (here->BSIM4v4rbodyMod)
              {   (*(here->BSIM4v4DPdbPtr ) += m * xcdbdb * s->real);
                  (*(here->BSIM4v4DPdbPtr +1) += m * xcdbdb * s->imag);
                  (*(here->BSIM4v4DPdbPtr) -= m * here->BSIM4v4gbd);
                  (*(here->BSIM4v4SPsbPtr ) += m * xcsbsb * s->real);
                  (*(here->BSIM4v4SPsbPtr +1) += m * xcsbsb * s->imag);
                  (*(here->BSIM4v4SPsbPtr) -= m * here->BSIM4v4gbs);

                  (*(here->BSIM4v4DBdpPtr ) += m * xcdbdb * s->real);
                  (*(here->BSIM4v4DBdpPtr +1) += m * xcdbdb * s->imag);
                  (*(here->BSIM4v4DBdpPtr) -= m * here->BSIM4v4gbd);
                  (*(here->BSIM4v4DBdbPtr ) -= m * xcdbdb * s->real);
                  (*(here->BSIM4v4DBdbPtr +1) -= m * xcdbdb * s->imag);
                  (*(here->BSIM4v4DBdbPtr) += m * (here->BSIM4v4gbd + here->BSIM4v4grbpd
                                          + here->BSIM4v4grbdb));
                  (*(here->BSIM4v4DBbpPtr) -= m * here->BSIM4v4grbpd);
                  (*(here->BSIM4v4DBbPtr) -= m * here->BSIM4v4grbdb);

                  (*(here->BSIM4v4BPdbPtr) -= m * here->BSIM4v4grbpd);
                  (*(here->BSIM4v4BPbPtr) -= m * here->BSIM4v4grbpb);
                  (*(here->BSIM4v4BPsbPtr) -= m * here->BSIM4v4grbps);
                  (*(here->BSIM4v4BPbpPtr) += m * (here->BSIM4v4grbpd + here->BSIM4v4grbps
					  + here->BSIM4v4grbpb));
                  /* WDL: (-here->BSIM4v4gbbs) already added to BPbpPtr */

                  (*(here->BSIM4v4SBspPtr ) += m * xcsbsb * s->real);
                  (*(here->BSIM4v4SBspPtr +1) += m * xcsbsb * s->imag);
                  (*(here->BSIM4v4SBspPtr) -= m * here->BSIM4v4gbs);
                  (*(here->BSIM4v4SBbpPtr) -= m * here->BSIM4v4grbps);
                  (*(here->BSIM4v4SBbPtr) -= m * here->BSIM4v4grbsb);
                  (*(here->BSIM4v4SBsbPtr ) -= m * xcsbsb * s->real);
                  (*(here->BSIM4v4SBsbPtr +1) -= m * xcsbsb * s->imag);
                  (*(here->BSIM4v4SBsbPtr) += m * (here->BSIM4v4gbs
					  + here->BSIM4v4grbps + here->BSIM4v4grbsb));

                  (*(here->BSIM4v4BdbPtr) -= m * here->BSIM4v4grbdb);
                  (*(here->BSIM4v4BbpPtr) -= m * here->BSIM4v4grbpb);
                  (*(here->BSIM4v4BsbPtr) -= m * here->BSIM4v4grbsb);
                  (*(here->BSIM4v4BbPtr) += m * (here->BSIM4v4grbsb + here->BSIM4v4grbdb
                                        + here->BSIM4v4grbpb));
              }

              if (here->BSIM4v4acnqsMod)
              {   *(here->BSIM4v4QqPtr ) += m * s->real * ScalingFactor;
                  *(here->BSIM4v4QqPtr +1) += m * s->imag * ScalingFactor;
                  *(here->BSIM4v4QgpPtr ) -= m * xcqgb * s->real;
                  *(here->BSIM4v4QgpPtr +1) -= m * xcqgb * s->imag;
                  *(here->BSIM4v4QdpPtr ) -= m * xcqdb * s->real;
                  *(here->BSIM4v4QdpPtr +1) -= m * xcqdb * s->imag;
                  *(here->BSIM4v4QbpPtr ) -= m * xcqbb * s->real;
                  *(here->BSIM4v4QbpPtr +1) -= m * xcqbb * s->imag;
                  *(here->BSIM4v4QspPtr ) -= m * xcqsb * s->real;
                  *(here->BSIM4v4QspPtr +1) -= m * xcqsb * s->imag;

                  *(here->BSIM4v4GPqPtr) -= m * here->BSIM4v4gtau;
                  *(here->BSIM4v4DPqPtr) += m * dxpart * here->BSIM4v4gtau;
                  *(here->BSIM4v4SPqPtr) += m * sxpart * here->BSIM4v4gtau;

                  *(here->BSIM4v4QqPtr) += m * here->BSIM4v4gtau;
                  *(here->BSIM4v4QgpPtr) += m * xgtg;
                  *(here->BSIM4v4QdpPtr) += m * xgtd;
                  *(here->BSIM4v4QbpPtr) += m * xgtb;
                  *(here->BSIM4v4QspPtr) += m * xgts;
              }
         }
    }
    return(OK);
}
