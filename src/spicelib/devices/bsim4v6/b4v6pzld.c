/**** BSIM4.6.2 Released by Wenwei Yang 07/31/2008 ****/

/**********
 * Copyright 2006 Regents of the University of California. All rights reserved.
 * File: b4pzld.c of BSIM4.6.2.
 * Author: 2000 Weidong Liu
 * Authors: 2001- Xuemei Xi, Mohan Dunga, Ali Niknejad, Chenming Hu.
 * Authors: 2006- Mohan Dunga, Ali Niknejad, Chenming Hu
 * Authors: 2007- Mohan Dunga, Wenwei Yang, Ali Niknejad, Chenming Hu
 * Project Director: Prof. Chenming Hu.
 * Modified by Xuemei Xi, 10/05/2001.
 **********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ngspice/complex.h"
#include "ngspice/sperror.h"
#include "bsim4v6def.h"
#include "ngspice/suffix.h"

int
BSIM4v6pzLoad(
GENmodel *inModel,
CKTcircuit *ckt,
SPcomplex *s)
{
BSIM4v6model *model = (BSIM4v6model*)inModel;
BSIM4v6instance *here;

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
struct bsim4v6SizeDependParam *pParam;
double ggidld, ggidlg, ggidlb, ggislg, ggislb, ggisls;

double m;

    for (; model != NULL; model = BSIM4v6nextModel(model)) 
    {    for (here = BSIM4v6instances(model); here!= NULL;
              here = BSIM4v6nextInstance(here)) 
         {
              pParam = here->pParam;
              capbd = here->BSIM4v6capbd;
              capbs = here->BSIM4v6capbs;
              cgso = here->BSIM4v6cgso;
              cgdo = here->BSIM4v6cgdo;
              cgbo = pParam->BSIM4v6cgbo;

              if (here->BSIM4v6mode >= 0) 
              {   Gm = here->BSIM4v6gm;
                  Gmbs = here->BSIM4v6gmbs;
                  FwdSum = Gm + Gmbs;
                  RevSum = 0.0;

                  gbbdp = -(here->BSIM4v6gbds);
                  gbbsp = here->BSIM4v6gbds + here->BSIM4v6gbgs + here->BSIM4v6gbbs;
                  gbdpg = here->BSIM4v6gbgs;
                  gbdpdp = here->BSIM4v6gbds;
                  gbdpb = here->BSIM4v6gbbs;
                  gbdpsp = -(gbdpg + gbdpdp + gbdpb);

                  gbspdp = 0.0;
                  gbspg = 0.0;
                  gbspb = 0.0;
                  gbspsp = 0.0;

                  if (model->BSIM4v6igcMod)
                  {   gIstotg = here->BSIM4v6gIgsg + here->BSIM4v6gIgcsg;
                      gIstotd = here->BSIM4v6gIgcsd;
                      gIstots = here->BSIM4v6gIgss + here->BSIM4v6gIgcss;
                      gIstotb = here->BSIM4v6gIgcsb;

                      gIdtotg = here->BSIM4v6gIgdg + here->BSIM4v6gIgcdg;
                      gIdtotd = here->BSIM4v6gIgdd + here->BSIM4v6gIgcdd;
                      gIdtots = here->BSIM4v6gIgcds;
                      gIdtotb = here->BSIM4v6gIgcdb;
                  }
                  else
                  {   gIstotg = gIstotd = gIstots = gIstotb = 0.0;
                      gIdtotg = gIdtotd = gIdtots = gIdtotb = 0.0;
                  }

                  if (model->BSIM4v6igbMod)
                  {   gIbtotg = here->BSIM4v6gIgbg;
                      gIbtotd = here->BSIM4v6gIgbd;
                      gIbtots = here->BSIM4v6gIgbs;
                      gIbtotb = here->BSIM4v6gIgbb;
                  }
                  else
                      gIbtotg = gIbtotd = gIbtots = gIbtotb = 0.0;

                  if ((model->BSIM4v6igcMod != 0) || (model->BSIM4v6igbMod != 0))
                  {   gIgtotg = gIstotg + gIdtotg + gIbtotg;
                      gIgtotd = gIstotd + gIdtotd + gIbtotd ;
                      gIgtots = gIstots + gIdtots + gIbtots;
                      gIgtotb = gIstotb + gIdtotb + gIbtotb;
                  }
                  else
                      gIgtotg = gIgtotd = gIgtots = gIgtotb = 0.0;

                  if (here->BSIM4v6rgateMod == 2)
                      T0 = *(ckt->CKTstates[0] + here->BSIM4v6vges)
                         - *(ckt->CKTstates[0] + here->BSIM4v6vgs);
                  else if (here->BSIM4v6rgateMod == 3)
                      T0 = *(ckt->CKTstates[0] + here->BSIM4v6vgms)
                         - *(ckt->CKTstates[0] + here->BSIM4v6vgs);
                  if (here->BSIM4v6rgateMod > 1)
                  {   gcrgd = here->BSIM4v6gcrgd * T0;
                      gcrgg = here->BSIM4v6gcrgg * T0;
                      gcrgs = here->BSIM4v6gcrgs * T0;
                      gcrgb = here->BSIM4v6gcrgb * T0;
                      gcrgg -= here->BSIM4v6gcrg;
                      gcrg = here->BSIM4v6gcrg;
                  }
                  else
                      gcrg = gcrgd = gcrgg = gcrgs = gcrgb = 0.0;

                  if (here->BSIM4v6acnqsMod == 0)
                  {   if (here->BSIM4v6rgateMod == 3)
                      {   xcgmgmb = cgdo + cgso + pParam->BSIM4v6cgbo;
                          xcgmdb = -cgdo;
                          xcgmsb = -cgso;
                          xcgmbb = -pParam->BSIM4v6cgbo;

                          xcdgmb = xcgmdb;
                          xcsgmb = xcgmsb;
                          xcbgmb = xcgmbb;

                          xcggb = here->BSIM4v6cggb;
                          xcgdb = here->BSIM4v6cgdb;
                          xcgsb = here->BSIM4v6cgsb;
                          xcgbb = -(xcggb + xcgdb + xcgsb);

                          xcdgb = here->BSIM4v6cdgb;
                          xcsgb = -(here->BSIM4v6cggb + here->BSIM4v6cbgb
                                + here->BSIM4v6cdgb);
                          xcbgb = here->BSIM4v6cbgb;
                      }
                      else
                      {   xcggb = here->BSIM4v6cggb + cgdo + cgso
                                + pParam->BSIM4v6cgbo;
                          xcgdb = here->BSIM4v6cgdb - cgdo;
                          xcgsb = here->BSIM4v6cgsb - cgso;
                          xcgbb = -(xcggb + xcgdb + xcgsb);

                          xcdgb = here->BSIM4v6cdgb - cgdo;
                          xcsgb = -(here->BSIM4v6cggb + here->BSIM4v6cbgb
                                + here->BSIM4v6cdgb + cgso);
                          xcbgb = here->BSIM4v6cbgb - pParam->BSIM4v6cgbo;

                          xcdgmb = xcsgmb = xcbgmb = 0.0;
                      }
                      xcddb = here->BSIM4v6cddb + here->BSIM4v6capbd + cgdo;
                      xcdsb = here->BSIM4v6cdsb;

                      xcsdb = -(here->BSIM4v6cgdb + here->BSIM4v6cbdb
                            + here->BSIM4v6cddb);
                      xcssb = here->BSIM4v6capbs + cgso - (here->BSIM4v6cgsb
                            + here->BSIM4v6cbsb + here->BSIM4v6cdsb);

                      if (!here->BSIM4v6rbodyMod)
                      {   xcdbb = -(xcdgb + xcddb + xcdsb + xcdgmb);
                          xcsbb = -(xcsgb + xcsdb + xcssb + xcsgmb);
                          xcbdb = here->BSIM4v6cbdb - here->BSIM4v6capbd;
                          xcbsb = here->BSIM4v6cbsb - here->BSIM4v6capbs;
                          xcdbdb = 0.0;
                      }
                      else
                      {   xcdbb  = -(here->BSIM4v6cddb + here->BSIM4v6cdgb
                                 + here->BSIM4v6cdsb);
                          xcsbb = -(xcsgb + xcsdb + xcssb + xcsgmb)
                                + here->BSIM4v6capbs;
                          xcbdb = here->BSIM4v6cbdb;
                          xcbsb = here->BSIM4v6cbsb;

                          xcdbdb = -here->BSIM4v6capbd;
                          xcsbsb = -here->BSIM4v6capbs;
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

		      xgtg = here->BSIM4v6gtg;
                      xgtd = here->BSIM4v6gtd;
                      xgts = here->BSIM4v6gts;
                      xgtb = here->BSIM4v6gtb;

                      xcqgb = here->BSIM4v6cqgb;
                      xcqdb = here->BSIM4v6cqdb;
                      xcqsb = here->BSIM4v6cqsb;
                      xcqbb = here->BSIM4v6cqbb;

		      CoxWL = model->BSIM4v6coxe * here->pParam->BSIM4v6weffCV
                            * here->BSIM4v6nf * here->pParam->BSIM4v6leffCV;
		      qcheq = -(here->BSIM4v6qgate + here->BSIM4v6qbulk);
		      if (fabs(qcheq) <= 1.0e-5 * CoxWL)
		      {   if (model->BSIM4v6xpart < 0.5)
		          {   dxpart = 0.4;
		          }
		          else if (model->BSIM4v6xpart > 0.5)
		          {   dxpart = 0.0;
		          }
		          else
		          {   dxpart = 0.5;
		          }
		          ddxpart_dVd = ddxpart_dVg = ddxpart_dVb
				      = ddxpart_dVs = 0.0;
		      }
		      else
		      {   dxpart = here->BSIM4v6qdrn / qcheq;
		          Cdd = here->BSIM4v6cddb;
		          Csd = -(here->BSIM4v6cgdb + here->BSIM4v6cddb
			      + here->BSIM4v6cbdb);
		          ddxpart_dVd = (Cdd - dxpart * (Cdd + Csd)) / qcheq;
		          Cdg = here->BSIM4v6cdgb;
		          Csg = -(here->BSIM4v6cggb + here->BSIM4v6cdgb
			      + here->BSIM4v6cbgb);
		          ddxpart_dVg = (Cdg - dxpart * (Cdg + Csg)) / qcheq;

		          Cds = here->BSIM4v6cdsb;
		          Css = -(here->BSIM4v6cgsb + here->BSIM4v6cdsb
			      + here->BSIM4v6cbsb);
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
              {   Gm = -here->BSIM4v6gm;
                  Gmbs = -here->BSIM4v6gmbs;
                  FwdSum = 0.0;
                  RevSum = -(Gm + Gmbs);

                  gbbsp = -(here->BSIM4v6gbds);
                  gbbdp = here->BSIM4v6gbds + here->BSIM4v6gbgs + here->BSIM4v6gbbs;

                  gbdpg = 0.0;
                  gbdpsp = 0.0;
                  gbdpb = 0.0;
                  gbdpdp = 0.0;

                  gbspg = here->BSIM4v6gbgs;
                  gbspsp = here->BSIM4v6gbds;
                  gbspb = here->BSIM4v6gbbs;
                  gbspdp = -(gbspg + gbspsp + gbspb);

                  if (model->BSIM4v6igcMod)
                  {   gIstotg = here->BSIM4v6gIgsg + here->BSIM4v6gIgcdg;
                      gIstotd = here->BSIM4v6gIgcds;
                      gIstots = here->BSIM4v6gIgss + here->BSIM4v6gIgcdd;
                      gIstotb = here->BSIM4v6gIgcdb;

                      gIdtotg = here->BSIM4v6gIgdg + here->BSIM4v6gIgcsg;
                      gIdtotd = here->BSIM4v6gIgdd + here->BSIM4v6gIgcss;
                      gIdtots = here->BSIM4v6gIgcsd;
                      gIdtotb = here->BSIM4v6gIgcsb;
                  }
                  else
                  {   gIstotg = gIstotd = gIstots = gIstotb = 0.0;
                      gIdtotg = gIdtotd = gIdtots = gIdtotb  = 0.0;
                  }

                  if (model->BSIM4v6igbMod)
                  {   gIbtotg = here->BSIM4v6gIgbg;
                      gIbtotd = here->BSIM4v6gIgbs;
                      gIbtots = here->BSIM4v6gIgbd;
                      gIbtotb = here->BSIM4v6gIgbb;
                  }
                  else
                      gIbtotg = gIbtotd = gIbtots = gIbtotb = 0.0;

                  if ((model->BSIM4v6igcMod != 0) || (model->BSIM4v6igbMod != 0))
                  {   gIgtotg = gIstotg + gIdtotg + gIbtotg;
                      gIgtotd = gIstotd + gIdtotd + gIbtotd ;
                      gIgtots = gIstots + gIdtots + gIbtots;
                      gIgtotb = gIstotb + gIdtotb + gIbtotb;
                  }
                  else
                      gIgtotg = gIgtotd = gIgtots = gIgtotb = 0.0;

                  if (here->BSIM4v6rgateMod == 2)
                      T0 = *(ckt->CKTstates[0] + here->BSIM4v6vges)
                         - *(ckt->CKTstates[0] + here->BSIM4v6vgs);
                  else if (here->BSIM4v6rgateMod == 3)
                      T0 = *(ckt->CKTstates[0] + here->BSIM4v6vgms)
                         - *(ckt->CKTstates[0] + here->BSIM4v6vgs);
                  if (here->BSIM4v6rgateMod > 1)
                  {   gcrgd = here->BSIM4v6gcrgs * T0;
                      gcrgg = here->BSIM4v6gcrgg * T0;
                      gcrgs = here->BSIM4v6gcrgd * T0;
                      gcrgb = here->BSIM4v6gcrgb * T0;
                      gcrgg -= here->BSIM4v6gcrg;
                      gcrg = here->BSIM4v6gcrg;
                  }
                  else
                      gcrg = gcrgd = gcrgg = gcrgs = gcrgb = 0.0;

                  if (here->BSIM4v6acnqsMod == 0)
                  {   if (here->BSIM4v6rgateMod == 3)
                      {   xcgmgmb = cgdo + cgso + pParam->BSIM4v6cgbo;
                          xcgmdb = -cgdo;
                          xcgmsb = -cgso;
                          xcgmbb = -pParam->BSIM4v6cgbo;
   
                          xcdgmb = xcgmdb;
                          xcsgmb = xcgmsb;
                          xcbgmb = xcgmbb;

                          xcggb = here->BSIM4v6cggb;
                          xcgdb = here->BSIM4v6cgsb;
                          xcgsb = here->BSIM4v6cgdb;
                          xcgbb = -(xcggb + xcgdb + xcgsb);

                          xcdgb = -(here->BSIM4v6cggb + here->BSIM4v6cbgb
                                + here->BSIM4v6cdgb);
                          xcsgb = here->BSIM4v6cdgb;
                          xcbgb = here->BSIM4v6cbgb;
                      }
                      else
                      {   xcggb = here->BSIM4v6cggb + cgdo + cgso
                                + pParam->BSIM4v6cgbo;
                          xcgdb = here->BSIM4v6cgsb - cgdo;
                          xcgsb = here->BSIM4v6cgdb - cgso;
                          xcgbb = -(xcggb + xcgdb + xcgsb);

                          xcdgb = -(here->BSIM4v6cggb + here->BSIM4v6cbgb
                                + here->BSIM4v6cdgb + cgdo);
                          xcsgb = here->BSIM4v6cdgb - cgso;
                          xcbgb = here->BSIM4v6cbgb - pParam->BSIM4v6cgbo;

                          xcdgmb = xcsgmb = xcbgmb = 0.0;
                      }
                      xcddb = here->BSIM4v6capbd + cgdo - (here->BSIM4v6cgsb
                            + here->BSIM4v6cbsb + here->BSIM4v6cdsb);
                      xcdsb = -(here->BSIM4v6cgdb + here->BSIM4v6cbdb
                            + here->BSIM4v6cddb);

                      xcsdb = here->BSIM4v6cdsb;
                      xcssb = here->BSIM4v6cddb + here->BSIM4v6capbs + cgso;

                      if (!here->BSIM4v6rbodyMod)
                      {   xcdbb = -(xcdgb + xcddb + xcdsb + xcdgmb);
                          xcsbb = -(xcsgb + xcsdb + xcssb + xcsgmb);
                          xcbdb = here->BSIM4v6cbsb - here->BSIM4v6capbd;
                          xcbsb = here->BSIM4v6cbdb - here->BSIM4v6capbs;
                          xcdbdb = 0.0;
                      }
                      else
                      {   xcdbb = -(xcdgb + xcddb + xcdsb + xcdgmb)
                                + here->BSIM4v6capbd;
                          xcsbb = -(here->BSIM4v6cddb + here->BSIM4v6cdgb
                                + here->BSIM4v6cdsb);
                          xcbdb = here->BSIM4v6cbsb;
                          xcbsb = here->BSIM4v6cbdb;
                          xcdbdb = -here->BSIM4v6capbd;
                          xcsbsb = -here->BSIM4v6capbs;
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

		      xgtg = here->BSIM4v6gtg;
                      xgtd = here->BSIM4v6gts;
                      xgts = here->BSIM4v6gtd;
                      xgtb = here->BSIM4v6gtb;

                      xcqgb = here->BSIM4v6cqgb;
                      xcqdb = here->BSIM4v6cqsb;
                      xcqsb = here->BSIM4v6cqdb;
                      xcqbb = here->BSIM4v6cqbb;

		      CoxWL = model->BSIM4v6coxe * here->pParam->BSIM4v6weffCV
                            * here->BSIM4v6nf * here->pParam->BSIM4v6leffCV;
		      qcheq = -(here->BSIM4v6qgate + here->BSIM4v6qbulk);
		      if (fabs(qcheq) <= 1.0e-5 * CoxWL)
		      {   if (model->BSIM4v6xpart < 0.5)
		          {   sxpart = 0.4;
		          }
		          else if (model->BSIM4v6xpart > 0.5)
		          {   sxpart = 0.0;
		          }
		          else
		          {   sxpart = 0.5;
		          }
		          dsxpart_dVd = dsxpart_dVg = dsxpart_dVb
				      = dsxpart_dVs = 0.0;
		      }
		      else
		      {   sxpart = here->BSIM4v6qdrn / qcheq;
		          Css = here->BSIM4v6cddb;
		          Cds = -(here->BSIM4v6cgdb + here->BSIM4v6cddb
			      + here->BSIM4v6cbdb);
		          dsxpart_dVs = (Css - sxpart * (Css + Cds)) / qcheq;
		          Csg = here->BSIM4v6cdgb;
		          Cdg = -(here->BSIM4v6cggb + here->BSIM4v6cdgb
			      + here->BSIM4v6cbgb);
		          dsxpart_dVg = (Csg - sxpart * (Csg + Cdg)) / qcheq;

		          Csd = here->BSIM4v6cdsb;
		          Cdd = -(here->BSIM4v6cgsb + here->BSIM4v6cdsb
			      + here->BSIM4v6cbsb);
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

              if (model->BSIM4v6rdsMod == 1)
              {   gstot = here->BSIM4v6gstot;
                  gstotd = here->BSIM4v6gstotd;
                  gstotg = here->BSIM4v6gstotg;
                  gstots = here->BSIM4v6gstots - gstot;
                  gstotb = here->BSIM4v6gstotb;

                  gdtot = here->BSIM4v6gdtot;
                  gdtotd = here->BSIM4v6gdtotd - gdtot;
                  gdtotg = here->BSIM4v6gdtotg;
                  gdtots = here->BSIM4v6gdtots;
                  gdtotb = here->BSIM4v6gdtotb;
              }
              else
              {   gstot = gstotd = gstotg = gstots = gstotb = 0.0;
                  gdtot = gdtotd = gdtotg = gdtots = gdtotb = 0.0;
              }


	      T1 = *(ckt->CKTstate0 + here->BSIM4v6qdef) * here->BSIM4v6gtau;
              gds = here->BSIM4v6gds;

              /*
               * Loading PZ matrix
               */
              m = here->BSIM4v6m;

              if (!model->BSIM4v6rdsMod)
              {   gdpr = here->BSIM4v6drainConductance;
                  gspr = here->BSIM4v6sourceConductance;
              }
              else
                  gdpr = gspr = 0.0;

              if (!here->BSIM4v6rbodyMod)
              {   gjbd = here->BSIM4v6gbd;
                  gjbs = here->BSIM4v6gbs;
              }
              else
                  gjbd = gjbs = 0.0;

              geltd = here->BSIM4v6grgeltd;

              if (here->BSIM4v6rgateMod == 1)
              {   *(here->BSIM4v6GEgePtr) += m * geltd;
                  *(here->BSIM4v6GPgePtr) -= m * geltd;
                  *(here->BSIM4v6GEgpPtr) -= m * geltd;

                  *(here->BSIM4v6GPgpPtr ) += m * xcggb * s->real;
                  *(here->BSIM4v6GPgpPtr +1) += m * xcggb * s->imag;
                  *(here->BSIM4v6GPgpPtr) += m * (geltd - xgtg + gIgtotg);
                  *(here->BSIM4v6GPdpPtr ) += m * xcgdb * s->real;
                  *(here->BSIM4v6GPdpPtr +1) += m * xcgdb * s->imag;
		  *(here->BSIM4v6GPdpPtr) -= m * (xgtd - gIgtotd);
                  *(here->BSIM4v6GPspPtr ) += m * xcgsb * s->real;
                  *(here->BSIM4v6GPspPtr +1) += m * xcgsb * s->imag;
                  *(here->BSIM4v6GPspPtr) -= m * (xgts - gIgtots);
                  *(here->BSIM4v6GPbpPtr ) += m * xcgbb * s->real;
                  *(here->BSIM4v6GPbpPtr +1) += m * xcgbb * s->imag;
		  *(here->BSIM4v6GPbpPtr) -= m * (xgtb - gIgtotb);
              }
              else if (here->BSIM4v6rgateMod == 2)
              {   *(here->BSIM4v6GEgePtr) += m * gcrg;
                  *(here->BSIM4v6GEgpPtr) += m * gcrgg;
                  *(here->BSIM4v6GEdpPtr) += m * gcrgd;
                  *(here->BSIM4v6GEspPtr) += m * gcrgs;
                  *(here->BSIM4v6GEbpPtr) += m * gcrgb;

                  *(here->BSIM4v6GPgePtr) -= m * gcrg;
                  *(here->BSIM4v6GPgpPtr ) += m * xcggb * s->real;
                  *(here->BSIM4v6GPgpPtr +1) += m * xcggb * s->imag;
                  *(here->BSIM4v6GPgpPtr) -= m * (gcrgg + xgtg - gIgtotg);
                  *(here->BSIM4v6GPdpPtr ) += m * xcgdb * s->real;
                  *(here->BSIM4v6GPdpPtr +1) += m * xcgdb * s->imag;
                  *(here->BSIM4v6GPdpPtr) -= m * (gcrgd + xgtd - gIgtotd);
                  *(here->BSIM4v6GPspPtr ) += m * xcgsb * s->real;
                  *(here->BSIM4v6GPspPtr +1) += m * xcgsb * s->imag;
                  *(here->BSIM4v6GPspPtr) -= m * (gcrgs + xgts - gIgtots);
                  *(here->BSIM4v6GPbpPtr ) += m * xcgbb * s->real;
                  *(here->BSIM4v6GPbpPtr +1) += m * xcgbb * s->imag;
                  *(here->BSIM4v6GPbpPtr) -= m * (gcrgb + xgtb - gIgtotb);
              }
              else if (here->BSIM4v6rgateMod == 3)
              {   *(here->BSIM4v6GEgePtr) += m * geltd;
                  *(here->BSIM4v6GEgmPtr) -= m * geltd;
                  *(here->BSIM4v6GMgePtr) -= m * geltd;
                  *(here->BSIM4v6GMgmPtr) += m * (geltd + gcrg);
                  *(here->BSIM4v6GMgmPtr ) += m * xcgmgmb * s->real;
                  *(here->BSIM4v6GMgmPtr +1) += m * xcgmgmb * s->imag;
  
                  *(here->BSIM4v6GMdpPtr) += m * gcrgd;
                  *(here->BSIM4v6GMdpPtr ) += m * xcgmdb * s->real;
                  *(here->BSIM4v6GMdpPtr +1) += m * xcgmdb * s->imag;
                  *(here->BSIM4v6GMgpPtr) += m * gcrgg;
                  *(here->BSIM4v6GMspPtr) += m * gcrgs;
                  *(here->BSIM4v6GMspPtr ) += m * xcgmsb * s->real;
                  *(here->BSIM4v6GMspPtr +1) += m * xcgmsb * s->imag;
                  *(here->BSIM4v6GMbpPtr) += m * gcrgb;
                  *(here->BSIM4v6GMbpPtr ) += m * xcgmbb * s->real;
                  *(here->BSIM4v6GMbpPtr +1) += m * xcgmbb * s->imag;
  
                  *(here->BSIM4v6DPgmPtr ) += m * xcdgmb * s->real;
                  *(here->BSIM4v6DPgmPtr +1) += m * xcdgmb * s->imag;
                  *(here->BSIM4v6GPgmPtr) -= m * gcrg;
                  *(here->BSIM4v6SPgmPtr ) += m * xcsgmb * s->real;
                  *(here->BSIM4v6SPgmPtr +1) += m * xcsgmb * s->imag;
                  *(here->BSIM4v6BPgmPtr ) += m * xcbgmb * s->real;
                  *(here->BSIM4v6BPgmPtr +1) += m * xcbgmb * s->imag;
  
                  *(here->BSIM4v6GPgpPtr) -= m * (gcrgg + xgtg - gIgtotg);
                  *(here->BSIM4v6GPgpPtr ) += m * xcggb * s->real;
                  *(here->BSIM4v6GPgpPtr +1) += m * xcggb * s->imag;
                  *(here->BSIM4v6GPdpPtr) -= m * (gcrgd + xgtd - gIgtotd);
                  *(here->BSIM4v6GPdpPtr ) += m * xcgdb * s->real;
                  *(here->BSIM4v6GPdpPtr +1) += m * xcgdb * s->imag;
                  *(here->BSIM4v6GPspPtr) -= m * (gcrgs + xgts - gIgtots);
                  *(here->BSIM4v6GPspPtr ) += m * xcgsb * s->real;
                  *(here->BSIM4v6GPspPtr +1) += m * xcgsb * s->imag;
                  *(here->BSIM4v6GPbpPtr) -= m * (gcrgb + xgtb - gIgtotb);
                  *(here->BSIM4v6GPbpPtr ) += m * xcgbb * s->real;
                  *(here->BSIM4v6GPbpPtr +1) += m * xcgbb * s->imag;
              }
              else
              {   *(here->BSIM4v6GPdpPtr ) += m * xcgdb * s->real;
                  *(here->BSIM4v6GPdpPtr +1) += m * xcgdb * s->imag;
		  *(here->BSIM4v6GPdpPtr) -= m * (xgtd - gIgtotd);
                  *(here->BSIM4v6GPgpPtr ) += m * xcggb * s->real;
                  *(here->BSIM4v6GPgpPtr +1) += m * xcggb * s->imag;
		  *(here->BSIM4v6GPgpPtr) -= m * (xgtg - gIgtotg);
                  *(here->BSIM4v6GPspPtr ) += m * xcgsb * s->real;
                  *(here->BSIM4v6GPspPtr +1) += m * xcgsb * s->imag;
                  *(here->BSIM4v6GPspPtr) -= m * (xgts - gIgtots);
                  *(here->BSIM4v6GPbpPtr ) += m * xcgbb * s->real;
                  *(here->BSIM4v6GPbpPtr +1) += m * xcgbb * s->imag;
		  *(here->BSIM4v6GPbpPtr) -= m * (xgtb - gIgtotb);
              }

              if (model->BSIM4v6rdsMod)
              {   (*(here->BSIM4v6DgpPtr) += m * gdtotg);
                  (*(here->BSIM4v6DspPtr) += m * gdtots);
                  (*(here->BSIM4v6DbpPtr) += m * gdtotb);
                  (*(here->BSIM4v6SdpPtr) += m * gstotd);
                  (*(here->BSIM4v6SgpPtr) += m * gstotg);
                  (*(here->BSIM4v6SbpPtr) += m * gstotb);
              }

              *(here->BSIM4v6DPdpPtr ) += m * xcddb * s->real;
              *(here->BSIM4v6DPdpPtr +1) += m * xcddb * s->imag;
              *(here->BSIM4v6DPdpPtr) += m * (gdpr + gds + here->BSIM4v6gbd
				     - gdtotd + RevSum + gbdpdp - gIdtotd
				     + dxpart * xgtd + T1 * ddxpart_dVd);
              *(here->BSIM4v6DPdPtr) -= m * (gdpr + gdtot);
              *(here->BSIM4v6DPgpPtr ) += m * xcdgb * s->real;
              *(here->BSIM4v6DPgpPtr +1) += m * xcdgb * s->imag;
              *(here->BSIM4v6DPgpPtr) += m * (Gm - gdtotg + gbdpg - gIdtotg
				     + T1 * ddxpart_dVg + dxpart * xgtg);
              *(here->BSIM4v6DPspPtr ) += m * xcdsb * s->real;
              *(here->BSIM4v6DPspPtr +1) += m * xcdsb * s->imag;
              *(here->BSIM4v6DPspPtr) -= m * (gds + FwdSum + gdtots - gbdpsp + gIdtots
				     - T1 * ddxpart_dVs - dxpart * xgts);
              *(here->BSIM4v6DPbpPtr ) += m * xcdbb * s->real;
              *(here->BSIM4v6DPbpPtr +1) += m * xcdbb * s->imag;
              *(here->BSIM4v6DPbpPtr) -= m * (gjbd + gdtotb - Gmbs - gbdpb + gIdtotb
				     - T1 * ddxpart_dVb - dxpart * xgtb);

              *(here->BSIM4v6DdpPtr) -= m * (gdpr - gdtotd);
              *(here->BSIM4v6DdPtr) += m * (gdpr + gdtot);

              *(here->BSIM4v6SPdpPtr ) += m * xcsdb * s->real;
              *(here->BSIM4v6SPdpPtr +1) += m * xcsdb * s->imag;
              *(here->BSIM4v6SPdpPtr) -= m * (gds + gstotd + RevSum - gbspdp + gIstotd
				     - T1 * dsxpart_dVd - sxpart * xgtd);
              *(here->BSIM4v6SPgpPtr ) += m * xcsgb * s->real;
              *(here->BSIM4v6SPgpPtr +1) += m * xcsgb * s->imag;
              *(here->BSIM4v6SPgpPtr) -= m * (Gm + gstotg - gbspg + gIstotg
				     - T1 * dsxpart_dVg - sxpart * xgtg);
              *(here->BSIM4v6SPspPtr ) += m * xcssb * s->real;
              *(here->BSIM4v6SPspPtr +1) += m * xcssb * s->imag;
              *(here->BSIM4v6SPspPtr) += m * (gspr + gds + here->BSIM4v6gbs - gIstots
				     - gstots + FwdSum + gbspsp
				     + sxpart * xgts + T1 * dsxpart_dVs);
              *(here->BSIM4v6SPsPtr) -= m * (gspr + gstot);
              *(here->BSIM4v6SPbpPtr ) += m * xcsbb * s->real;
              *(here->BSIM4v6SPbpPtr +1) += m * xcsbb * s->imag;
              *(here->BSIM4v6SPbpPtr) -= m * (gjbs + gstotb + Gmbs - gbspb + gIstotb
				     - T1 * dsxpart_dVb - sxpart * xgtb);

              *(here->BSIM4v6SspPtr) -= m * (gspr - gstots);
              *(here->BSIM4v6SsPtr) += m * (gspr + gstot);

              *(here->BSIM4v6BPdpPtr ) += m * xcbdb * s->real;
              *(here->BSIM4v6BPdpPtr +1) += m * xcbdb * s->imag;
              *(here->BSIM4v6BPdpPtr) -= m * (gjbd - gbbdp + gIbtotd);
              *(here->BSIM4v6BPgpPtr ) += m * xcbgb * s->real;
              *(here->BSIM4v6BPgpPtr +1) += m * xcbgb * s->imag;
              *(here->BSIM4v6BPgpPtr) -= m * (here->BSIM4v6gbgs + gIbtotg);
              *(here->BSIM4v6BPspPtr ) += m * xcbsb * s->real;
              *(here->BSIM4v6BPspPtr +1) += m * xcbsb * s->imag;
              *(here->BSIM4v6BPspPtr) -= m * (gjbs - gbbsp + gIbtots);
              *(here->BSIM4v6BPbpPtr ) += m * xcbbb * s->real;
              *(here->BSIM4v6BPbpPtr +1) += m * xcbbb * s->imag;
              *(here->BSIM4v6BPbpPtr) += m * (gjbd + gjbs - here->BSIM4v6gbbs
				     - gIbtotb);
           ggidld = here->BSIM4v6ggidld;
           ggidlg = here->BSIM4v6ggidlg;
           ggidlb = here->BSIM4v6ggidlb;
           ggislg = here->BSIM4v6ggislg;
           ggisls = here->BSIM4v6ggisls;
           ggislb = here->BSIM4v6ggislb;

           /* stamp gidl */
           (*(here->BSIM4v6DPdpPtr) += m * ggidld);
           (*(here->BSIM4v6DPgpPtr) += m * ggidlg);
           (*(here->BSIM4v6DPspPtr) -= m * ((ggidlg + ggidld) + ggidlb));
           (*(here->BSIM4v6DPbpPtr) += m * ggidlb);
           (*(here->BSIM4v6BPdpPtr) -= m * ggidld);
           (*(here->BSIM4v6BPgpPtr) -= m * ggidlg);
           (*(here->BSIM4v6BPspPtr) += m * ((ggidlg + ggidld) + ggidlb));
           (*(here->BSIM4v6BPbpPtr) -= m * ggidlb);
            /* stamp gisl */
           (*(here->BSIM4v6SPdpPtr) -= m * ((ggisls + ggislg) + ggislb));
           (*(here->BSIM4v6SPgpPtr) += m * ggislg);
           (*(here->BSIM4v6SPspPtr) += m * ggisls);
           (*(here->BSIM4v6SPbpPtr) += m * ggislb);
           (*(here->BSIM4v6BPdpPtr) += m * ((ggislg + ggisls) + ggislb));
           (*(here->BSIM4v6BPgpPtr) -= m * ggislg);
           (*(here->BSIM4v6BPspPtr) -= m * ggisls);
           (*(here->BSIM4v6BPbpPtr) -= m * ggislb);

              if (here->BSIM4v6rbodyMod)
              {   (*(here->BSIM4v6DPdbPtr ) += m * xcdbdb * s->real);
                  (*(here->BSIM4v6DPdbPtr +1) += m * xcdbdb * s->imag);
                  (*(here->BSIM4v6DPdbPtr) -= m * here->BSIM4v6gbd);
                  (*(here->BSIM4v6SPsbPtr ) += m * xcsbsb * s->real);
                  (*(here->BSIM4v6SPsbPtr +1) += m * xcsbsb * s->imag);
                  (*(here->BSIM4v6SPsbPtr) -= m * here->BSIM4v6gbs);

                  (*(here->BSIM4v6DBdpPtr ) += m * xcdbdb * s->real);
                  (*(here->BSIM4v6DBdpPtr +1) += m * xcdbdb * s->imag);
                  (*(here->BSIM4v6DBdpPtr) -= m * here->BSIM4v6gbd);
                  (*(here->BSIM4v6DBdbPtr ) -= m * xcdbdb * s->real);
                  (*(here->BSIM4v6DBdbPtr +1) -= m * xcdbdb * s->imag);
                  (*(here->BSIM4v6DBdbPtr) += m * (here->BSIM4v6gbd + here->BSIM4v6grbpd
                                          + here->BSIM4v6grbdb));
                  (*(here->BSIM4v6DBbpPtr) -= m * here->BSIM4v6grbpd);
                  (*(here->BSIM4v6DBbPtr) -= m * here->BSIM4v6grbdb);

                  (*(here->BSIM4v6BPdbPtr) -= m * here->BSIM4v6grbpd);
                  (*(here->BSIM4v6BPbPtr) -= m * here->BSIM4v6grbpb);
                  (*(here->BSIM4v6BPsbPtr) -= m * here->BSIM4v6grbps);
                  (*(here->BSIM4v6BPbpPtr) += m * (here->BSIM4v6grbpd + here->BSIM4v6grbps
					  + here->BSIM4v6grbpb));
                  /* WDL: (-here->BSIM4v6gbbs) already added to BPbpPtr */

                  (*(here->BSIM4v6SBspPtr ) += m * xcsbsb * s->real);
                  (*(here->BSIM4v6SBspPtr +1) += m * xcsbsb * s->imag);
                  (*(here->BSIM4v6SBspPtr) -= m * here->BSIM4v6gbs);
                  (*(here->BSIM4v6SBbpPtr) -= m * here->BSIM4v6grbps);
                  (*(here->BSIM4v6SBbPtr) -= m * here->BSIM4v6grbsb);
                  (*(here->BSIM4v6SBsbPtr ) -= m * xcsbsb * s->real);
                  (*(here->BSIM4v6SBsbPtr +1) -= m * xcsbsb * s->imag);
                  (*(here->BSIM4v6SBsbPtr) += m * (here->BSIM4v6gbs
					  + here->BSIM4v6grbps + here->BSIM4v6grbsb));

                  (*(here->BSIM4v6BdbPtr) -= m * here->BSIM4v6grbdb);
                  (*(here->BSIM4v6BbpPtr) -= m * here->BSIM4v6grbpb);
                  (*(here->BSIM4v6BsbPtr) -= m * here->BSIM4v6grbsb);
                  (*(here->BSIM4v6BbPtr) += m * (here->BSIM4v6grbsb + here->BSIM4v6grbdb
                                        + here->BSIM4v6grbpb));
              }

              if (here->BSIM4v6acnqsMod)
              {   *(here->BSIM4v6QqPtr ) += m * s->real * ScalingFactor;
                  *(here->BSIM4v6QqPtr +1) += m * s->imag * ScalingFactor;
                  *(here->BSIM4v6QgpPtr ) -= m * xcqgb * s->real;
                  *(here->BSIM4v6QgpPtr +1) -= m * xcqgb * s->imag;
                  *(here->BSIM4v6QdpPtr ) -= m * xcqdb * s->real;
                  *(here->BSIM4v6QdpPtr +1) -= m * xcqdb * s->imag;
                  *(here->BSIM4v6QbpPtr ) -= m * xcqbb * s->real;
                  *(here->BSIM4v6QbpPtr +1) -= m * xcqbb * s->imag;
                  *(here->BSIM4v6QspPtr ) -= m * xcqsb * s->real;
                  *(here->BSIM4v6QspPtr +1) -= m * xcqsb * s->imag;

                  *(here->BSIM4v6GPqPtr) -= m * here->BSIM4v6gtau;
                  *(here->BSIM4v6DPqPtr) += m * dxpart * here->BSIM4v6gtau;
                  *(here->BSIM4v6SPqPtr) += m * sxpart * here->BSIM4v6gtau;

                  *(here->BSIM4v6QqPtr) += m * here->BSIM4v6gtau;
                  *(here->BSIM4v6QgpPtr) += m * xgtg;
                  *(here->BSIM4v6QdpPtr) += m * xgtd;
                  *(here->BSIM4v6QbpPtr) += m * xgtb;
                  *(here->BSIM4v6QspPtr) += m * xgts;
              }
         }
    }
    return(OK);
}
