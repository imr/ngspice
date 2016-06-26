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
BSIM4v4pzLoad(inModel,ckt,s)
GENmodel *inModel;
CKTcircuit *ckt;
SPcomplex *s;
{
BSIM4v4model *model = (BSIM4v4model*)inModel;
BSIM4v4instance *here;

double gjbd, gjbs, geltd, gcrg, gcrgg, gcrgd, gcrgs, gcrgb;
double xcggb, xcgdb, xcgsb, xcgbb, xcbgb, xcbdb, xcbsb, xcbbb;
double xcdgb, xcddb, xcdsb, xcdbb, xcsgb, xcsdb, xcssb, xcsbb;
double gds, gbd, gbs, capbd, capbs, FwdSum, RevSum, Gm, Gmbs;
double gstot, gstotd, gstotg, gstots, gstotb, gspr;
double gdtot, gdtotd, gdtotg, gdtots, gdtotb, gdpr;
double gIstotg, gIstotd, gIstots, gIstotb;
double gIdtotg, gIdtotd, gIdtots, gIdtotb;
double gIbtotg, gIbtotd, gIbtots, gIbtotb;
double gIgtotg, gIgtotd, gIgtots, gIgtotb;
double cgso, cgdo, cgbo;
double xcdbdb, xcsbsb, xcgmgmb, xcgmdb, xcgmsb, xcdgmb, xcsgmb;
double xcgmbb, xcbgmb;
double dxpart, sxpart, xgtg, xgtd, xgts, xgtb, xcqgb, xcqdb, xcqsb, xcqbb;
double gbspsp, gbbdp, gbbsp, gbspg, gbspb;
double gbspdp, gbdpdp, gbdpg, gbdpb, gbdpsp;
double ddxpart_dVd, ddxpart_dVg, ddxpart_dVb, ddxpart_dVs;
double dsxpart_dVd, dsxpart_dVg, dsxpart_dVb, dsxpart_dVs;
double T0, T1, CoxWL, qcheq, Cdg, Cdd, Cds, Cdb, Csg, Csd, Css, Csb;
double ScalingFactor = 1.0e-9;
struct bsim4v4SizeDependParam *pParam;
double ggidld, ggidlg, ggidlb,ggisld, ggislg, ggislb, ggisls;


    for (; model != NULL; model = model->BSIM4v4nextModel)
    {    for (here = model->BSIM4v4instances; here!= NULL;
              here = here->BSIM4v4nextInstance)
	 {    pParam = here->pParam;
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
              {   *(here->BSIM4v4GEgePtr) += geltd;
                  *(here->BSIM4v4GPgePtr) -= geltd;
                  *(here->BSIM4v4GEgpPtr) -= geltd;

                  *(here->BSIM4v4GPgpPtr ) += xcggb * s->real;
                  *(here->BSIM4v4GPgpPtr +1) += xcggb * s->imag;
                  *(here->BSIM4v4GPgpPtr) += geltd - xgtg + gIgtotg;
                  *(here->BSIM4v4GPdpPtr ) += xcgdb * s->real;
                  *(here->BSIM4v4GPdpPtr +1) += xcgdb * s->imag;
		  *(here->BSIM4v4GPdpPtr) -= xgtd - gIgtotd;
                  *(here->BSIM4v4GPspPtr ) += xcgsb * s->real;
                  *(here->BSIM4v4GPspPtr +1) += xcgsb * s->imag;
                  *(here->BSIM4v4GPspPtr) -= xgts - gIgtots;
                  *(here->BSIM4v4GPbpPtr ) += xcgbb * s->real;
                  *(here->BSIM4v4GPbpPtr +1) += xcgbb * s->imag;
		  *(here->BSIM4v4GPbpPtr) -= xgtb - gIgtotb;
              }
              else if (here->BSIM4v4rgateMod == 2)
              {   *(here->BSIM4v4GEgePtr) += gcrg;
                  *(here->BSIM4v4GEgpPtr) += gcrgg;
                  *(here->BSIM4v4GEdpPtr) += gcrgd;
                  *(here->BSIM4v4GEspPtr) += gcrgs;
                  *(here->BSIM4v4GEbpPtr) += gcrgb;

                  *(here->BSIM4v4GPgePtr) -= gcrg;
                  *(here->BSIM4v4GPgpPtr ) += xcggb * s->real;
                  *(here->BSIM4v4GPgpPtr +1) += xcggb * s->imag;
                  *(here->BSIM4v4GPgpPtr) -= gcrgg + xgtg - gIgtotg;
                  *(here->BSIM4v4GPdpPtr ) += xcgdb * s->real;
                  *(here->BSIM4v4GPdpPtr +1) += xcgdb * s->imag;
                  *(here->BSIM4v4GPdpPtr) -= gcrgd + xgtd - gIgtotd;
                  *(here->BSIM4v4GPspPtr ) += xcgsb * s->real;
                  *(here->BSIM4v4GPspPtr +1) += xcgsb * s->imag;
                  *(here->BSIM4v4GPspPtr) -= gcrgs + xgts - gIgtots;
                  *(here->BSIM4v4GPbpPtr ) += xcgbb * s->real;
                  *(here->BSIM4v4GPbpPtr +1) += xcgbb * s->imag;
                  *(here->BSIM4v4GPbpPtr) -= gcrgb + xgtb - gIgtotb;
              }
              else if (here->BSIM4v4rgateMod == 3)
              {   *(here->BSIM4v4GEgePtr) += geltd;
                  *(here->BSIM4v4GEgmPtr) -= geltd;
                  *(here->BSIM4v4GMgePtr) -= geltd;
                  *(here->BSIM4v4GMgmPtr) += geltd + gcrg;
                  *(here->BSIM4v4GMgmPtr ) += xcgmgmb * s->real;
                  *(here->BSIM4v4GMgmPtr +1) += xcgmgmb * s->imag;

                  *(here->BSIM4v4GMdpPtr) += gcrgd;
                  *(here->BSIM4v4GMdpPtr ) += xcgmdb * s->real;
                  *(here->BSIM4v4GMdpPtr +1) += xcgmdb * s->imag;
                  *(here->BSIM4v4GMgpPtr) += gcrgg;
                  *(here->BSIM4v4GMspPtr) += gcrgs;
                  *(here->BSIM4v4GMspPtr ) += xcgmsb * s->real;
                  *(here->BSIM4v4GMspPtr +1) += xcgmsb * s->imag;
                  *(here->BSIM4v4GMbpPtr) += gcrgb;
                  *(here->BSIM4v4GMbpPtr ) += xcgmbb * s->real;
                  *(here->BSIM4v4GMbpPtr +1) += xcgmbb * s->imag;

                  *(here->BSIM4v4DPgmPtr ) += xcdgmb * s->real;
                  *(here->BSIM4v4DPgmPtr +1) += xcdgmb * s->imag;
                  *(here->BSIM4v4GPgmPtr) -= gcrg;
                  *(here->BSIM4v4SPgmPtr ) += xcsgmb * s->real;
                  *(here->BSIM4v4SPgmPtr +1) += xcsgmb * s->imag;
                  *(here->BSIM4v4BPgmPtr ) += xcbgmb * s->real;
                  *(here->BSIM4v4BPgmPtr +1) += xcbgmb * s->imag;

                  *(here->BSIM4v4GPgpPtr) -= gcrgg + xgtg - gIgtotg;
                  *(here->BSIM4v4GPgpPtr ) += xcggb * s->real;
                  *(here->BSIM4v4GPgpPtr +1) += xcggb * s->imag;
                  *(here->BSIM4v4GPdpPtr) -= gcrgd + xgtd - gIgtotd;
                  *(here->BSIM4v4GPdpPtr ) += xcgdb * s->real;
                  *(here->BSIM4v4GPdpPtr +1) += xcgdb * s->imag;
                  *(here->BSIM4v4GPspPtr) -= gcrgs + xgts - gIgtots;
                  *(here->BSIM4v4GPspPtr ) += xcgsb * s->real;
                  *(here->BSIM4v4GPspPtr +1) += xcgsb * s->imag;
                  *(here->BSIM4v4GPbpPtr) -= gcrgb + xgtb - gIgtotb;
                  *(here->BSIM4v4GPbpPtr ) += xcgbb * s->real;
                  *(here->BSIM4v4GPbpPtr +1) += xcgbb * s->imag;
              }
              else
              {   *(here->BSIM4v4GPdpPtr ) += xcgdb * s->real;
                  *(here->BSIM4v4GPdpPtr +1) += xcgdb * s->imag;
		  *(here->BSIM4v4GPdpPtr) -= xgtd - gIgtotd;
                  *(here->BSIM4v4GPgpPtr ) += xcggb * s->real;
                  *(here->BSIM4v4GPgpPtr +1) += xcggb * s->imag;
		  *(here->BSIM4v4GPgpPtr) -= xgtg - gIgtotg;
                  *(here->BSIM4v4GPspPtr ) += xcgsb * s->real;
                  *(here->BSIM4v4GPspPtr +1) += xcgsb * s->imag;
                  *(here->BSIM4v4GPspPtr) -= xgts - gIgtots;
                  *(here->BSIM4v4GPbpPtr ) += xcgbb * s->real;
                  *(here->BSIM4v4GPbpPtr +1) += xcgbb * s->imag;
		  *(here->BSIM4v4GPbpPtr) -= xgtb - gIgtotb;
              }

              if (model->BSIM4v4rdsMod)
              {   (*(here->BSIM4v4DgpPtr) += gdtotg);
                  (*(here->BSIM4v4DspPtr) += gdtots);
                  (*(here->BSIM4v4DbpPtr) += gdtotb);
                  (*(here->BSIM4v4SdpPtr) += gstotd);
                  (*(here->BSIM4v4SgpPtr) += gstotg);
                  (*(here->BSIM4v4SbpPtr) += gstotb);
              }

              *(here->BSIM4v4DPdpPtr ) += xcddb * s->real;
              *(here->BSIM4v4DPdpPtr +1) += xcddb * s->imag;
              *(here->BSIM4v4DPdpPtr) += gdpr + gds + here->BSIM4v4gbd
				     - gdtotd + RevSum + gbdpdp - gIdtotd
				     + dxpart * xgtd + T1 * ddxpart_dVd;
              *(here->BSIM4v4DPdPtr) -= gdpr + gdtot;
              *(here->BSIM4v4DPgpPtr ) += xcdgb * s->real;
              *(here->BSIM4v4DPgpPtr +1) += xcdgb * s->imag;
              *(here->BSIM4v4DPgpPtr) += Gm - gdtotg + gbdpg - gIdtotg
				     + T1 * ddxpart_dVg + dxpart * xgtg;
              *(here->BSIM4v4DPspPtr ) += xcdsb * s->real;
              *(here->BSIM4v4DPspPtr +1) += xcdsb * s->imag;
              *(here->BSIM4v4DPspPtr) -= gds + FwdSum + gdtots - gbdpsp + gIdtots
				     - T1 * ddxpart_dVs - dxpart * xgts;
              *(here->BSIM4v4DPbpPtr ) += xcdbb * s->real;
              *(here->BSIM4v4DPbpPtr +1) += xcdbb * s->imag;
              *(here->BSIM4v4DPbpPtr) -= gjbd + gdtotb - Gmbs - gbdpb + gIdtotb
				     - T1 * ddxpart_dVb - dxpart * xgtb;

              *(here->BSIM4v4DdpPtr) -= gdpr - gdtotd;
              *(here->BSIM4v4DdPtr) += gdpr + gdtot;

              *(here->BSIM4v4SPdpPtr ) += xcsdb * s->real;
              *(here->BSIM4v4SPdpPtr +1) += xcsdb * s->imag;
              *(here->BSIM4v4SPdpPtr) -= gds + gstotd + RevSum - gbspdp + gIstotd
				     - T1 * dsxpart_dVd - sxpart * xgtd;
              *(here->BSIM4v4SPgpPtr ) += xcsgb * s->real;
              *(here->BSIM4v4SPgpPtr +1) += xcsgb * s->imag;
              *(here->BSIM4v4SPgpPtr) -= Gm + gstotg - gbspg + gIstotg
				     - T1 * dsxpart_dVg - sxpart * xgtg;
              *(here->BSIM4v4SPspPtr ) += xcssb * s->real;
              *(here->BSIM4v4SPspPtr +1) += xcssb * s->imag;
              *(here->BSIM4v4SPspPtr) += gspr + gds + here->BSIM4v4gbs - gIstots
				     - gstots + FwdSum + gbspsp
				     + sxpart * xgts + T1 * dsxpart_dVs;
              *(here->BSIM4v4SPsPtr) -= gspr + gstot;
              *(here->BSIM4v4SPbpPtr ) += xcsbb * s->real;
              *(here->BSIM4v4SPbpPtr +1) += xcsbb * s->imag;
              *(here->BSIM4v4SPbpPtr) -= gjbs + gstotb + Gmbs - gbspb + gIstotb
				     - T1 * dsxpart_dVb - sxpart * xgtb;

              *(here->BSIM4v4SspPtr) -= gspr - gstots;
              *(here->BSIM4v4SsPtr) += gspr + gstot;

              *(here->BSIM4v4BPdpPtr ) += xcbdb * s->real;
              *(here->BSIM4v4BPdpPtr +1) += xcbdb * s->imag;
              *(here->BSIM4v4BPdpPtr) -= gjbd - gbbdp + gIbtotd;
              *(here->BSIM4v4BPgpPtr ) += xcbgb * s->real;
              *(here->BSIM4v4BPgpPtr +1) += xcbgb * s->imag;
              *(here->BSIM4v4BPgpPtr) -= here->BSIM4v4gbgs + gIbtotg;
              *(here->BSIM4v4BPspPtr ) += xcbsb * s->real;
              *(here->BSIM4v4BPspPtr +1) += xcbsb * s->imag;
              *(here->BSIM4v4BPspPtr) -= gjbs - gbbsp + gIbtots;
              *(here->BSIM4v4BPbpPtr ) += xcbbb * s->real;
              *(here->BSIM4v4BPbpPtr +1) += xcbbb * s->imag;
              *(here->BSIM4v4BPbpPtr) += gjbd + gjbs - here->BSIM4v4gbbs
				     - gIbtotb;
           ggidld = here->BSIM4v4ggidld;
           ggidlg = here->BSIM4v4ggidlg;
           ggidlb = here->BSIM4v4ggidlb;
           ggislg = here->BSIM4v4ggislg;
           ggisls = here->BSIM4v4ggisls;
           ggislb = here->BSIM4v4ggislb;

           /* stamp gidl */
           (*(here->BSIM4v4DPdpPtr) += ggidld);
           (*(here->BSIM4v4DPgpPtr) += ggidlg);
           (*(here->BSIM4v4DPspPtr) -= (ggidlg + ggidld) + ggidlb);
           (*(here->BSIM4v4DPbpPtr) += ggidlb);
           (*(here->BSIM4v4BPdpPtr) -= ggidld);
           (*(here->BSIM4v4BPgpPtr) -= ggidlg);
           (*(here->BSIM4v4BPspPtr) += (ggidlg + ggidld) + ggidlb);
           (*(here->BSIM4v4BPbpPtr) -= ggidlb);
            /* stamp gisl */
           (*(here->BSIM4v4SPdpPtr) -= (ggisls + ggislg) + ggislb);
           (*(here->BSIM4v4SPgpPtr) += ggislg);
           (*(here->BSIM4v4SPspPtr) += ggisls);
           (*(here->BSIM4v4SPbpPtr) += ggislb);
           (*(here->BSIM4v4BPdpPtr) += (ggislg + ggisls) + ggislb);
           (*(here->BSIM4v4BPgpPtr) -= ggislg);
           (*(here->BSIM4v4BPspPtr) -= ggisls);
           (*(here->BSIM4v4BPbpPtr) -= ggislb);

              if (here->BSIM4v4rbodyMod)
              {   (*(here->BSIM4v4DPdbPtr ) += xcdbdb * s->real);
                  (*(here->BSIM4v4DPdbPtr +1) += xcdbdb * s->imag);
                  (*(here->BSIM4v4DPdbPtr) -= here->BSIM4v4gbd);
                  (*(here->BSIM4v4SPsbPtr ) += xcsbsb * s->real);
                  (*(here->BSIM4v4SPsbPtr +1) += xcsbsb * s->imag);
                  (*(here->BSIM4v4SPsbPtr) -= here->BSIM4v4gbs);

                  (*(here->BSIM4v4DBdpPtr ) += xcdbdb * s->real);
                  (*(here->BSIM4v4DBdpPtr +1) += xcdbdb * s->imag);
                  (*(here->BSIM4v4DBdpPtr) -= here->BSIM4v4gbd);
                  (*(here->BSIM4v4DBdbPtr ) -= xcdbdb * s->real);
                  (*(here->BSIM4v4DBdbPtr +1) -= xcdbdb * s->imag);
                  (*(here->BSIM4v4DBdbPtr) += here->BSIM4v4gbd + here->BSIM4v4grbpd
                                          + here->BSIM4v4grbdb);
                  (*(here->BSIM4v4DBbpPtr) -= here->BSIM4v4grbpd);
                  (*(here->BSIM4v4DBbPtr) -= here->BSIM4v4grbdb);

                  (*(here->BSIM4v4BPdbPtr) -= here->BSIM4v4grbpd);
                  (*(here->BSIM4v4BPbPtr) -= here->BSIM4v4grbpb);
                  (*(here->BSIM4v4BPsbPtr) -= here->BSIM4v4grbps);
                  (*(here->BSIM4v4BPbpPtr) += here->BSIM4v4grbpd + here->BSIM4v4grbps
					  + here->BSIM4v4grbpb);
                  /* WDL: (-here->BSIM4v4gbbs) already added to BPbpPtr */

                  (*(here->BSIM4v4SBspPtr ) += xcsbsb * s->real);
                  (*(here->BSIM4v4SBspPtr +1) += xcsbsb * s->imag);
                  (*(here->BSIM4v4SBspPtr) -= here->BSIM4v4gbs);
                  (*(here->BSIM4v4SBbpPtr) -= here->BSIM4v4grbps);
                  (*(here->BSIM4v4SBbPtr) -= here->BSIM4v4grbsb);
                  (*(here->BSIM4v4SBsbPtr ) -= xcsbsb * s->real);
                  (*(here->BSIM4v4SBsbPtr +1) -= xcsbsb * s->imag);
                  (*(here->BSIM4v4SBsbPtr) += here->BSIM4v4gbs
					  + here->BSIM4v4grbps + here->BSIM4v4grbsb);

                  (*(here->BSIM4v4BdbPtr) -= here->BSIM4v4grbdb);
                  (*(here->BSIM4v4BbpPtr) -= here->BSIM4v4grbpb);
                  (*(here->BSIM4v4BsbPtr) -= here->BSIM4v4grbsb);
                  (*(here->BSIM4v4BbPtr) += here->BSIM4v4grbsb + here->BSIM4v4grbdb
                                        + here->BSIM4v4grbpb);
              }

              if (here->BSIM4v4acnqsMod)
              {   *(here->BSIM4v4QqPtr ) += s->real * ScalingFactor;
                  *(here->BSIM4v4QqPtr +1) += s->imag * ScalingFactor;
                  *(here->BSIM4v4QgpPtr ) -= xcqgb * s->real;
                  *(here->BSIM4v4QgpPtr +1) -= xcqgb * s->imag;
                  *(here->BSIM4v4QdpPtr ) -= xcqdb * s->real;
                  *(here->BSIM4v4QdpPtr +1) -= xcqdb * s->imag;
                  *(here->BSIM4v4QbpPtr ) -= xcqbb * s->real;
                  *(here->BSIM4v4QbpPtr +1) -= xcqbb * s->imag;
                  *(here->BSIM4v4QspPtr ) -= xcqsb * s->real;
                  *(here->BSIM4v4QspPtr +1) -= xcqsb * s->imag;

                  *(here->BSIM4v4GPqPtr) -= here->BSIM4v4gtau;
                  *(here->BSIM4v4DPqPtr) += dxpart * here->BSIM4v4gtau;
                  *(here->BSIM4v4SPqPtr) += sxpart * here->BSIM4v4gtau;

                  *(here->BSIM4v4QqPtr) += here->BSIM4v4gtau;
                  *(here->BSIM4v4QgpPtr) += xgtg;
                  *(here->BSIM4v4QdpPtr) += xgtd;
                  *(here->BSIM4v4QbpPtr) += xgtb;
                  *(here->BSIM4v4QspPtr) += xgts;
              }
         }
    }
    return(OK);
}
