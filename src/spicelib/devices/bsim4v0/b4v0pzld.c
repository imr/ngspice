/**** BSIM4.0.0, Released by Weidong Liu 3/24/2000 ****/

/**********
 * Copyright 2000 Regents of the University of California. All rights reserved.
 * File: b4pzld.c of BSIM4.0.0.
 * Authors: Weidong Liu, Kanyu M. Cao, Xiaodong Jin, Chenming Hu.
 * Project Director: Prof. Chenming Hu.
 **********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ngspice/complex.h"
#include "ngspice/sperror.h"
#include "bsim4v0def.h"
#include "ngspice/suffix.h"

int
BSIM4v0pzLoad(
GENmodel *inModel,
CKTcircuit *ckt,
SPcomplex *s)
{
BSIM4v0model *model = (BSIM4v0model*)inModel;
BSIM4v0instance *here;

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
double xcdbdb=0.0, xcsbsb=0.0, xcgmgmb=0.0, xcgmdb=0.0, xcgmsb=0.0, xcdgmb=0.0, xcsgmb=0.0;
double xcgmbb=0.0, xcbgmb=0.0;
double dxpart, sxpart, xgtg, xgtd, xgts, xgtb, xcqgb=0.0, xcqdb=0.0, xcqsb=0.0, xcqbb=0.0;
double gbspsp, gbbdp, gbbsp, gbspg, gbspb;
double gbspdp, gbdpdp, gbdpg, gbdpb, gbdpsp;
double ddxpart_dVd, ddxpart_dVg, ddxpart_dVb, ddxpart_dVs;
double dsxpart_dVd, dsxpart_dVg, dsxpart_dVb, dsxpart_dVs;
double T0=0.0, T1, CoxWL, qcheq, Cdg, Cdd, Cds, Cdb, Csg, Csd, Css, Csb;
double ScalingFactor = 1.0e-9;
struct bsim4v0SizeDependParam *pParam;

    for (; model != NULL; model = model->BSIM4v0nextModel) 
    {    for (here = model->BSIM4v0instances; here!= NULL;
              here = here->BSIM4v0nextInstance) 
	 {    pParam = here->pParam;
              capbd = here->BSIM4v0capbd;
              capbs = here->BSIM4v0capbs;
              cgso = here->BSIM4v0cgso;
              cgdo = here->BSIM4v0cgdo;
              cgbo = pParam->BSIM4v0cgbo;

              if (here->BSIM4v0mode >= 0) 
              {   Gm = here->BSIM4v0gm;
                  Gmbs = here->BSIM4v0gmbs;
                  FwdSum = Gm + Gmbs;
                  RevSum = 0.0;

                  gbbdp = -(here->BSIM4v0gbds + here->BSIM4v0ggidld);
                  gbbsp = here->BSIM4v0gbds + here->BSIM4v0gbgs + here->BSIM4v0gbbs
                        - here->BSIM4v0ggidls;
                  gbdpg = here->BSIM4v0gbgs + here->BSIM4v0ggidlg;
                  gbdpdp = here->BSIM4v0gbds + here->BSIM4v0ggidld;
                  gbdpb = here->BSIM4v0gbbs + here->BSIM4v0ggidlb;
                  gbdpsp = -(gbdpg + gbdpdp + gbdpb) + here->BSIM4v0ggidls;

                  gbspdp = 0.0;
                  gbspg = 0.0;
                  gbspb = 0.0;
                  gbspsp = 0.0;

                  if (model->BSIM4v0igcMod)
                  {   gIstotg = here->BSIM4v0gIgsg + here->BSIM4v0gIgcsg;
                      gIstotd = here->BSIM4v0gIgcsd;
                      gIstots = here->BSIM4v0gIgss + here->BSIM4v0gIgcss;
                      gIstotb = here->BSIM4v0gIgcsb;

                      gIdtotg = here->BSIM4v0gIgdg + here->BSIM4v0gIgcdg;
                      gIdtotd = here->BSIM4v0gIgdd + here->BSIM4v0gIgcdd;
                      gIdtots = here->BSIM4v0gIgcds;
                      gIdtotb = here->BSIM4v0gIgcdb;
                  }
                  else
                  {   gIstotg = gIstotd = gIstots = gIstotb = 0.0;
                      gIdtotg = gIdtotd = gIdtots = gIdtotb = 0.0;
                  }

                  if (model->BSIM4v0igbMod)
                  {   gIbtotg = here->BSIM4v0gIgbg;
                      gIbtotd = here->BSIM4v0gIgbd;
                      gIbtots = here->BSIM4v0gIgbs;
                      gIbtotb = here->BSIM4v0gIgbb;
                  }
                  else
                      gIbtotg = gIbtotd = gIbtots = gIbtotb = 0.0;

                  if ((model->BSIM4v0igcMod != 0) || (model->BSIM4v0igbMod != 0))
                  {   gIgtotg = gIstotg + gIdtotg + gIbtotg;
                      gIgtotd = gIstotd + gIdtotd + gIbtotd ;
                      gIgtots = gIstots + gIdtots + gIbtots;
                      gIgtotb = gIstotb + gIdtotb + gIbtotb;
                  }
                  else
                      gIgtotg = gIgtotd = gIgtots = gIgtotb = 0.0;

                  if (here->BSIM4v0rgateMod == 2)
                      T0 = *(ckt->CKTstates[0] + here->BSIM4v0vges)
                         - *(ckt->CKTstates[0] + here->BSIM4v0vgs);
                  else if (here->BSIM4v0rgateMod == 3)
                      T0 = *(ckt->CKTstates[0] + here->BSIM4v0vgms)
                         - *(ckt->CKTstates[0] + here->BSIM4v0vgs);
                  if (here->BSIM4v0rgateMod > 1)
                  {   gcrgd = here->BSIM4v0gcrgd * T0;
                      gcrgg = here->BSIM4v0gcrgg * T0;
                      gcrgs = here->BSIM4v0gcrgs * T0;
                      gcrgb = here->BSIM4v0gcrgb * T0;
                      gcrgg -= here->BSIM4v0gcrg;
                      gcrg = here->BSIM4v0gcrg;
                  }
                  else
                      gcrg = gcrgd = gcrgg = gcrgs = gcrgb = 0.0;

                  if (here->BSIM4v0acnqsMod == 0)
                  {   if (here->BSIM4v0rgateMod == 3)
                      {   xcgmgmb = cgdo + cgso + pParam->BSIM4v0cgbo;
                          xcgmdb = -cgdo;
                          xcgmsb = -cgso;
                          xcgmbb = -pParam->BSIM4v0cgbo;

                          xcdgmb = xcgmdb;
                          xcsgmb = xcgmsb;
                          xcbgmb = xcgmbb;

                          xcggb = here->BSIM4v0cggb;
                          xcgdb = here->BSIM4v0cgdb;
                          xcgsb = here->BSIM4v0cgsb;
                          xcgbb = -(xcggb + xcgdb + xcgsb);

                          xcdgb = here->BSIM4v0cdgb;
                          xcsgb = -(here->BSIM4v0cggb + here->BSIM4v0cbgb
                                + here->BSIM4v0cdgb);
                          xcbgb = here->BSIM4v0cbgb;
                      }
                      else
                      {   xcggb = here->BSIM4v0cggb + cgdo + cgso
                                + pParam->BSIM4v0cgbo;
                          xcgdb = here->BSIM4v0cgdb - cgdo;
                          xcgsb = here->BSIM4v0cgsb - cgso;
                          xcgbb = -(xcggb + xcgdb + xcgsb);

                          xcdgb = here->BSIM4v0cdgb - cgdo;
                          xcsgb = -(here->BSIM4v0cggb + here->BSIM4v0cbgb
                                + here->BSIM4v0cdgb + cgso);
                          xcbgb = here->BSIM4v0cbgb - pParam->BSIM4v0cgbo;

                          xcdgmb = xcsgmb = xcbgmb = 0.0;
                      }
                      xcddb = here->BSIM4v0cddb + here->BSIM4v0capbd + cgdo;
                      xcdsb = here->BSIM4v0cdsb;

                      xcsdb = -(here->BSIM4v0cgdb + here->BSIM4v0cbdb
                            + here->BSIM4v0cddb);
                      xcssb = here->BSIM4v0capbs + cgso - (here->BSIM4v0cgsb
                            + here->BSIM4v0cbsb + here->BSIM4v0cdsb);

                      if (!here->BSIM4v0rbodyMod)
                      {   xcdbb = -(xcdgb + xcddb + xcdsb + xcdgmb);
                          xcsbb = -(xcsgb + xcsdb + xcssb + xcsgmb);
                          xcbdb = here->BSIM4v0cbdb - here->BSIM4v0capbd;
                          xcbsb = here->BSIM4v0cbsb - here->BSIM4v0capbs;
                          xcdbdb = 0.0;
                      }
                      else
                      {   xcdbb  = -(here->BSIM4v0cddb + here->BSIM4v0cdgb
                                 + here->BSIM4v0cdsb);
                          xcsbb = -(xcsgb + xcsdb + xcssb + xcsgmb)
                                + here->BSIM4v0capbs;
                          xcbdb = here->BSIM4v0cbdb;
                          xcbsb = here->BSIM4v0cbsb;

                          xcdbdb = -here->BSIM4v0capbd;
                          xcsbsb = -here->BSIM4v0capbs;
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

		      xgtg = here->BSIM4v0gtg;
                      xgtd = here->BSIM4v0gtd;
                      xgts = here->BSIM4v0gts;
                      xgtb = here->BSIM4v0gtb;

                      xcqgb = here->BSIM4v0cqgb;
                      xcqdb = here->BSIM4v0cqdb;
                      xcqsb = here->BSIM4v0cqsb;
                      xcqbb = here->BSIM4v0cqbb;

		      CoxWL = model->BSIM4v0coxe * here->pParam->BSIM4v0weffCV
                            * here->BSIM4v0nf * here->pParam->BSIM4v0leffCV;
		      qcheq = -(here->BSIM4v0qgate + here->BSIM4v0qbulk);
		      if (fabs(qcheq) <= 1.0e-5 * CoxWL)
		      {   if (model->BSIM4v0xpart < 0.5)
		          {   dxpart = 0.4;
		          }
		          else if (model->BSIM4v0xpart > 0.5)
		          {   dxpart = 0.0;
		          }
		          else
		          {   dxpart = 0.5;
		          }
		          ddxpart_dVd = ddxpart_dVg = ddxpart_dVb
				      = ddxpart_dVs = 0.0;
		      }
		      else
		      {   dxpart = here->BSIM4v0qdrn / qcheq;
		          Cdd = here->BSIM4v0cddb;
		          Csd = -(here->BSIM4v0cgdb + here->BSIM4v0cddb
			      + here->BSIM4v0cbdb);
		          ddxpart_dVd = (Cdd - dxpart * (Cdd + Csd)) / qcheq;
		          Cdg = here->BSIM4v0cdgb;
		          Csg = -(here->BSIM4v0cggb + here->BSIM4v0cdgb
			      + here->BSIM4v0cbgb);
		          ddxpart_dVg = (Cdg - dxpart * (Cdg + Csg)) / qcheq;

		          Cds = here->BSIM4v0cdsb;
		          Css = -(here->BSIM4v0cgsb + here->BSIM4v0cdsb
			      + here->BSIM4v0cbsb);
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
              {   Gm = -here->BSIM4v0gm;
                  Gmbs = -here->BSIM4v0gmbs;
                  FwdSum = 0.0;
                  RevSum = -(Gm + Gmbs);

                  gbbsp = -(here->BSIM4v0gbds + here->BSIM4v0ggidld);
                  gbbdp = here->BSIM4v0gbds + here->BSIM4v0gbgs + here->BSIM4v0gbbs
                        - here->BSIM4v0ggidls;

                  gbdpg = 0.0;
                  gbdpsp = 0.0;
                  gbdpb = 0.0;
                  gbdpdp = 0.0;

                  gbspg = here->BSIM4v0gbgs + here->BSIM4v0ggidlg;
                  gbspsp = here->BSIM4v0gbds + here->BSIM4v0ggidld;
                  gbspb = here->BSIM4v0gbbs + here->BSIM4v0ggidlb;
                  gbspdp = -(gbspg + gbspsp + gbspb) + here->BSIM4v0ggidls;

                  if (model->BSIM4v0igcMod)
                  {   gIstotg = here->BSIM4v0gIgsg + here->BSIM4v0gIgcdg;
                      gIstotd = here->BSIM4v0gIgcds;
                      gIstots = here->BSIM4v0gIgss + here->BSIM4v0gIgcdd;
                      gIstotb = here->BSIM4v0gIgcdb;

                      gIdtotg = here->BSIM4v0gIgdg + here->BSIM4v0gIgcsg;
                      gIdtotd = here->BSIM4v0gIgdd + here->BSIM4v0gIgcss;
                      gIdtots = here->BSIM4v0gIgcsd;
                      gIdtotb = here->BSIM4v0gIgcsb;
                  }
                  else
                  {   gIstotg = gIstotd = gIstots = gIstotb = 0.0;
                      gIdtotg = gIdtotd = gIdtots = gIdtotb  = 0.0;
                  }

                  if (model->BSIM4v0igbMod)
                  {   gIbtotg = here->BSIM4v0gIgbg;
                      gIbtotd = here->BSIM4v0gIgbs;
                      gIbtots = here->BSIM4v0gIgbd;
                      gIbtotb = here->BSIM4v0gIgbb;
                  }
                  else
                      gIbtotg = gIbtotd = gIbtots = gIbtotb = 0.0;

                  if ((model->BSIM4v0igcMod != 0) || (model->BSIM4v0igbMod != 0))
                  {   gIgtotg = gIstotg + gIdtotg + gIbtotg;
                      gIgtotd = gIstotd + gIdtotd + gIbtotd ;
                      gIgtots = gIstots + gIdtots + gIbtots;
                      gIgtotb = gIstotb + gIdtotb + gIbtotb;
                  }
                  else
                      gIgtotg = gIgtotd = gIgtots = gIgtotb = 0.0;

                  if (here->BSIM4v0rgateMod == 2)
                      T0 = *(ckt->CKTstates[0] + here->BSIM4v0vges)
                         - *(ckt->CKTstates[0] + here->BSIM4v0vgs);
                  else if (here->BSIM4v0rgateMod == 3)
                      T0 = *(ckt->CKTstates[0] + here->BSIM4v0vgms)
                         - *(ckt->CKTstates[0] + here->BSIM4v0vgs);
                  if (here->BSIM4v0rgateMod > 1)
                  {   gcrgd = here->BSIM4v0gcrgs * T0;
                      gcrgg = here->BSIM4v0gcrgg * T0;
                      gcrgs = here->BSIM4v0gcrgd * T0;
                      gcrgb = here->BSIM4v0gcrgb * T0;
                      gcrgg -= here->BSIM4v0gcrg;
                      gcrg = here->BSIM4v0gcrg;
                  }
                  else
                      gcrg = gcrgd = gcrgg = gcrgs = gcrgb = 0.0;

                  if (here->BSIM4v0acnqsMod == 0)
                  {   if (here->BSIM4v0rgateMod == 3)
                      {   xcgmgmb = cgdo + cgso + pParam->BSIM4v0cgbo;
                          xcgmdb = -cgdo;
                          xcgmsb = -cgso;
                          xcgmbb = -pParam->BSIM4v0cgbo;
   
                          xcdgmb = xcgmdb;
                          xcsgmb = xcgmsb;
                          xcbgmb = xcgmbb;

                          xcggb = here->BSIM4v0cggb;
                          xcgdb = here->BSIM4v0cgsb;
                          xcgsb = here->BSIM4v0cgdb;
                          xcgbb = -(xcggb + xcgdb + xcgsb);

                          xcdgb = -(here->BSIM4v0cggb + here->BSIM4v0cbgb
                                + here->BSIM4v0cdgb);
                          xcsgb = here->BSIM4v0cdgb;
                          xcbgb = here->BSIM4v0cbgb;
                      }
                      else
                      {   xcggb = here->BSIM4v0cggb + cgdo + cgso
                                + pParam->BSIM4v0cgbo;
                          xcgdb = here->BSIM4v0cgsb - cgdo;
                          xcgsb = here->BSIM4v0cgdb - cgso;
                          xcgbb = -(xcggb + xcgdb + xcgsb);

                          xcdgb = -(here->BSIM4v0cggb + here->BSIM4v0cbgb
                                + here->BSIM4v0cdgb + cgdo);
                          xcsgb = here->BSIM4v0cdgb - cgso;
                          xcbgb = here->BSIM4v0cbgb - pParam->BSIM4v0cgbo;

                          xcdgmb = xcsgmb = xcbgmb = 0.0;
                      }
                      xcddb = here->BSIM4v0capbd + cgdo - (here->BSIM4v0cgsb
                            + here->BSIM4v0cbsb + here->BSIM4v0cdsb);
                      xcdsb = -(here->BSIM4v0cgdb + here->BSIM4v0cbdb
                            + here->BSIM4v0cddb);

                      xcsdb = here->BSIM4v0cdsb;
                      xcssb = here->BSIM4v0cddb + here->BSIM4v0capbs + cgso;

                      if (!here->BSIM4v0rbodyMod)
                      {   xcdbb = -(xcdgb + xcddb + xcdsb + xcdgmb);
                          xcsbb = -(xcsgb + xcsdb + xcssb + xcsgmb);
                          xcbdb = here->BSIM4v0cbsb - here->BSIM4v0capbd;
                          xcbsb = here->BSIM4v0cbdb - here->BSIM4v0capbs;
                          xcdbdb = 0.0;
                      }
                      else
                      {   xcdbb = -(xcdgb + xcddb + xcdsb + xcdgmb)
                                + here->BSIM4v0capbd;
                          xcsbb = -(here->BSIM4v0cddb + here->BSIM4v0cdgb
                                + here->BSIM4v0cdsb);
                          xcbdb = here->BSIM4v0cbsb;
                          xcbsb = here->BSIM4v0cbdb;
                          xcdbdb = -here->BSIM4v0capbd;
                          xcsbsb = -here->BSIM4v0capbs;
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

		      xgtg = here->BSIM4v0gtg;
                      xgtd = here->BSIM4v0gts;
                      xgts = here->BSIM4v0gtd;
                      xgtb = here->BSIM4v0gtb;

                      xcqgb = here->BSIM4v0cqgb;
                      xcqdb = here->BSIM4v0cqsb;
                      xcqsb = here->BSIM4v0cqdb;
                      xcqbb = here->BSIM4v0cqbb;

		      CoxWL = model->BSIM4v0coxe * here->pParam->BSIM4v0weffCV
                            * here->BSIM4v0nf * here->pParam->BSIM4v0leffCV;
		      qcheq = -(here->BSIM4v0qgate + here->BSIM4v0qbulk);
		      if (fabs(qcheq) <= 1.0e-5 * CoxWL)
		      {   if (model->BSIM4v0xpart < 0.5)
		          {   sxpart = 0.4;
		          }
		          else if (model->BSIM4v0xpart > 0.5)
		          {   sxpart = 0.0;
		          }
		          else
		          {   sxpart = 0.5;
		          }
		          dsxpart_dVd = dsxpart_dVg = dsxpart_dVb
				      = dsxpart_dVs = 0.0;
		      }
		      else
		      {   sxpart = here->BSIM4v0qdrn / qcheq;
		          Css = here->BSIM4v0cddb;
		          Cds = -(here->BSIM4v0cgdb + here->BSIM4v0cddb
			      + here->BSIM4v0cbdb);
		          dsxpart_dVs = (Css - sxpart * (Css + Cds)) / qcheq;
		          Csg = here->BSIM4v0cdgb;
		          Cdg = -(here->BSIM4v0cggb + here->BSIM4v0cdgb
			      + here->BSIM4v0cbgb);
		          dsxpart_dVg = (Csg - sxpart * (Csg + Cdg)) / qcheq;

		          Csd = here->BSIM4v0cdsb;
		          Cdd = -(here->BSIM4v0cgsb + here->BSIM4v0cdsb
			      + here->BSIM4v0cbsb);
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

              if (model->BSIM4v0rdsMod == 1)
              {   gstot = here->BSIM4v0gstot;
                  gstotd = here->BSIM4v0gstotd;
                  gstotg = here->BSIM4v0gstotg;
                  gstots = here->BSIM4v0gstots - gstot;
                  gstotb = here->BSIM4v0gstotb;

                  gdtot = here->BSIM4v0gdtot;
                  gdtotd = here->BSIM4v0gdtotd - gdtot;
                  gdtotg = here->BSIM4v0gdtotg;
                  gdtots = here->BSIM4v0gdtots;
                  gdtotb = here->BSIM4v0gdtotb;
              }
              else
              {   gstot = gstotd = gstotg = gstots = gstotb = 0.0;
                  gdtot = gdtotd = gdtotg = gdtots = gdtotb = 0.0;
              }


	      T1 = *(ckt->CKTstate0 + here->BSIM4v0qdef) * here->BSIM4v0gtau;
              gds = here->BSIM4v0gds;

              /*
               * Loading PZ matrix
               */

              if (!model->BSIM4v0rdsMod)
              {   gdpr = here->BSIM4v0drainConductance;
                  gspr = here->BSIM4v0sourceConductance;
              }
              else
                  gdpr = gspr = 0.0;

              if (!here->BSIM4v0rbodyMod)
              {   gjbd = here->BSIM4v0gbd;
                  gjbs = here->BSIM4v0gbs;
              }
              else
                  gjbd = gjbs = 0.0;

              geltd = here->BSIM4v0grgeltd;

              if (here->BSIM4v0rgateMod == 1)
              {   *(here->BSIM4v0GEgePtr) += geltd;
                  *(here->BSIM4v0GPgePtr) -= geltd;
                  *(here->BSIM4v0GEgpPtr) -= geltd;

                  *(here->BSIM4v0GPgpPtr ) += xcggb * s->real;
                  *(here->BSIM4v0GPgpPtr +1) += xcggb * s->imag;
                  *(here->BSIM4v0GPgpPtr) += geltd - xgtg + gIgtotg;
                  *(here->BSIM4v0GPdpPtr ) += xcgdb * s->real;
                  *(here->BSIM4v0GPdpPtr +1) += xcgdb * s->imag;
		  *(here->BSIM4v0GPdpPtr) -= xgtd - gIgtotd;
                  *(here->BSIM4v0GPspPtr ) += xcgsb * s->real;
                  *(here->BSIM4v0GPspPtr +1) += xcgsb * s->imag;
                  *(here->BSIM4v0GPspPtr) -= xgts - gIgtots;
                  *(here->BSIM4v0GPbpPtr ) += xcgbb * s->real;
                  *(here->BSIM4v0GPbpPtr +1) += xcgbb * s->imag;
		  *(here->BSIM4v0GPbpPtr) -= xgtb - gIgtotb;
              }
              else if (here->BSIM4v0rgateMod == 2)
              {   *(here->BSIM4v0GEgePtr) += gcrg;
                  *(here->BSIM4v0GEgpPtr) += gcrgg;
                  *(here->BSIM4v0GEdpPtr) += gcrgd;
                  *(here->BSIM4v0GEspPtr) += gcrgs;
                  *(here->BSIM4v0GEbpPtr) += gcrgb;

                  *(here->BSIM4v0GPgePtr) -= gcrg;
                  *(here->BSIM4v0GPgpPtr ) += xcggb * s->real;
                  *(here->BSIM4v0GPgpPtr +1) += xcggb * s->imag;
                  *(here->BSIM4v0GPgpPtr) -= gcrgg + xgtg - gIgtotg;
                  *(here->BSIM4v0GPdpPtr ) += xcgdb * s->real;
                  *(here->BSIM4v0GPdpPtr +1) += xcgdb * s->imag;
                  *(here->BSIM4v0GPdpPtr) -= gcrgd + xgtd - gIgtotd;
                  *(here->BSIM4v0GPspPtr ) += xcgsb * s->real;
                  *(here->BSIM4v0GPspPtr +1) += xcgsb * s->imag;
                  *(here->BSIM4v0GPspPtr) -= gcrgs + xgts - gIgtots;
                  *(here->BSIM4v0GPbpPtr ) += xcgbb * s->real;
                  *(here->BSIM4v0GPbpPtr +1) += xcgbb * s->imag;
                  *(here->BSIM4v0GPbpPtr) -= gcrgb + xgtb - gIgtotb;
              }
              else if (here->BSIM4v0rgateMod == 3)
              {   *(here->BSIM4v0GEgePtr) += geltd;
                  *(here->BSIM4v0GEgmPtr) -= geltd;
                  *(here->BSIM4v0GMgePtr) -= geltd;
                  *(here->BSIM4v0GMgmPtr) += geltd + gcrg;
                  *(here->BSIM4v0GMgmPtr ) += xcgmgmb * s->real;
                  *(here->BSIM4v0GMgmPtr +1) += xcgmgmb * s->imag;
  
                  *(here->BSIM4v0GMdpPtr) += gcrgd;
                  *(here->BSIM4v0GMdpPtr ) += xcgmdb * s->real;
                  *(here->BSIM4v0GMdpPtr +1) += xcgmdb * s->imag;
                  *(here->BSIM4v0GMgpPtr) += gcrgg;
                  *(here->BSIM4v0GMspPtr) += gcrgs;
                  *(here->BSIM4v0GMspPtr ) += xcgmsb * s->real;
                  *(here->BSIM4v0GMspPtr +1) += xcgmsb * s->imag;
                  *(here->BSIM4v0GMbpPtr) += gcrgb;
                  *(here->BSIM4v0GMbpPtr ) += xcgmbb * s->real;
                  *(here->BSIM4v0GMbpPtr +1) += xcgmbb * s->imag;
  
                  *(here->BSIM4v0DPgmPtr ) += xcdgmb * s->real;
                  *(here->BSIM4v0DPgmPtr +1) += xcdgmb * s->imag;
                  *(here->BSIM4v0GPgmPtr) -= gcrg;
                  *(here->BSIM4v0SPgmPtr ) += xcsgmb * s->real;
                  *(here->BSIM4v0SPgmPtr +1) += xcsgmb * s->imag;
                  *(here->BSIM4v0BPgmPtr ) += xcbgmb * s->real;
                  *(here->BSIM4v0BPgmPtr +1) += xcbgmb * s->imag;
  
                  *(here->BSIM4v0GPgpPtr) -= gcrgg + xgtg - gIgtotg;
                  *(here->BSIM4v0GPgpPtr ) += xcggb * s->real;
                  *(here->BSIM4v0GPgpPtr +1) += xcggb * s->imag;
                  *(here->BSIM4v0GPdpPtr) -= gcrgd + xgtd - gIgtotd;
                  *(here->BSIM4v0GPdpPtr ) += xcgdb * s->real;
                  *(here->BSIM4v0GPdpPtr +1) += xcgdb * s->imag;
                  *(here->BSIM4v0GPspPtr) -= gcrgs + xgts - gIgtots;
                  *(here->BSIM4v0GPspPtr ) += xcgsb * s->real;
                  *(here->BSIM4v0GPspPtr +1) += xcgsb * s->imag;
                  *(here->BSIM4v0GPbpPtr) -= gcrgb + xgtb - gIgtotb;
                  *(here->BSIM4v0GPbpPtr ) += xcgbb * s->real;
                  *(here->BSIM4v0GPbpPtr +1) += xcgbb * s->imag;
              }
              else
              {   *(here->BSIM4v0GPdpPtr ) += xcgdb * s->real;
                  *(here->BSIM4v0GPdpPtr +1) += xcgdb * s->imag;
		  *(here->BSIM4v0GPdpPtr) -= xgtd - gIgtotd;
                  *(here->BSIM4v0GPgpPtr ) += xcggb * s->real;
                  *(here->BSIM4v0GPgpPtr +1) += xcggb * s->imag;
		  *(here->BSIM4v0GPgpPtr) -= xgtg - gIgtotg;
                  *(here->BSIM4v0GPspPtr ) += xcgsb * s->real;
                  *(here->BSIM4v0GPspPtr +1) += xcgsb * s->imag;
                  *(here->BSIM4v0GPspPtr) -= xgts - gIgtots;
                  *(here->BSIM4v0GPbpPtr ) += xcgbb * s->real;
                  *(here->BSIM4v0GPbpPtr +1) += xcgbb * s->imag;
		  *(here->BSIM4v0GPbpPtr) -= xgtb - gIgtotb;
              }

              if (model->BSIM4v0rdsMod)
              {   (*(here->BSIM4v0DgpPtr) += gdtotg);
                  (*(here->BSIM4v0DspPtr) += gdtots);
                  (*(here->BSIM4v0DbpPtr) += gdtotb);
                  (*(here->BSIM4v0SdpPtr) += gstotd);
                  (*(here->BSIM4v0SgpPtr) += gstotg);
                  (*(here->BSIM4v0SbpPtr) += gstotb);
              }

              *(here->BSIM4v0DPdpPtr ) += xcddb * s->real;
              *(here->BSIM4v0DPdpPtr +1) += xcddb * s->imag;
              *(here->BSIM4v0DPdpPtr) += gdpr + gds + here->BSIM4v0gbd
				     - gdtotd + RevSum + gbdpdp - gIdtotd
				     + dxpart * xgtd + T1 * ddxpart_dVd;
              *(here->BSIM4v0DPdPtr) -= gdpr + gdtot;
              *(here->BSIM4v0DPgpPtr ) += xcdgb * s->real;
              *(here->BSIM4v0DPgpPtr +1) += xcdgb * s->imag;
              *(here->BSIM4v0DPgpPtr) += Gm - gdtotg + gbdpg - gIdtotg
				     + T1 * ddxpart_dVg + dxpart * xgtg;
              *(here->BSIM4v0DPspPtr ) += xcdsb * s->real;
              *(here->BSIM4v0DPspPtr +1) += xcdsb * s->imag;
              *(here->BSIM4v0DPspPtr) -= gds + FwdSum + gdtots - gbdpsp + gIdtots
				     - T1 * ddxpart_dVs - dxpart * xgts;
              *(here->BSIM4v0DPbpPtr ) += xcdbb * s->real;
              *(here->BSIM4v0DPbpPtr +1) += xcdbb * s->imag;
              *(here->BSIM4v0DPbpPtr) -= gjbd + gdtotb - Gmbs - gbdpb + gIdtotb
				     - T1 * ddxpart_dVb - dxpart * xgtb;

              *(here->BSIM4v0DdpPtr) -= gdpr - gdtotd;
              *(here->BSIM4v0DdPtr) += gdpr + gdtot;

              *(here->BSIM4v0SPdpPtr ) += xcsdb * s->real;
              *(here->BSIM4v0SPdpPtr +1) += xcsdb * s->imag;
              *(here->BSIM4v0SPdpPtr) -= gds + gstotd + RevSum - gbspdp + gIstotd
				     - T1 * dsxpart_dVd - sxpart * xgtd;
              *(here->BSIM4v0SPgpPtr ) += xcsgb * s->real;
              *(here->BSIM4v0SPgpPtr +1) += xcsgb * s->imag;
              *(here->BSIM4v0SPgpPtr) -= Gm + gstotg - gbspg + gIstotg
				     - T1 * dsxpart_dVg - sxpart * xgtg;
              *(here->BSIM4v0SPspPtr ) += xcssb * s->real;
              *(here->BSIM4v0SPspPtr +1) += xcssb * s->imag;
              *(here->BSIM4v0SPspPtr) += gspr + gds + here->BSIM4v0gbs - gIstots
				     - gstots + FwdSum + gbspsp
				     + sxpart * xgts + T1 * dsxpart_dVs;
              *(here->BSIM4v0SPsPtr) -= gspr + gstot;
              *(here->BSIM4v0SPbpPtr ) += xcsbb * s->real;
              *(here->BSIM4v0SPbpPtr +1) += xcsbb * s->imag;
              *(here->BSIM4v0SPbpPtr) -= gjbs + gstotb + Gmbs - gbspb + gIstotb
				     - T1 * dsxpart_dVb - sxpart * xgtb;

              *(here->BSIM4v0SspPtr) -= gspr - gstots;
              *(here->BSIM4v0SsPtr) += gspr + gstot;

              *(here->BSIM4v0BPdpPtr ) += xcbdb * s->real;
              *(here->BSIM4v0BPdpPtr +1) += xcbdb * s->imag;
              *(here->BSIM4v0BPdpPtr) -= gjbd - gbbdp + gIbtotd;
              *(here->BSIM4v0BPgpPtr ) += xcbgb * s->real;
              *(here->BSIM4v0BPgpPtr +1) += xcbgb * s->imag;
              *(here->BSIM4v0BPgpPtr) -= here->BSIM4v0gbgs + here->BSIM4v0ggidlg + gIbtotg;
              *(here->BSIM4v0BPspPtr ) += xcbsb * s->real;
              *(here->BSIM4v0BPspPtr +1) += xcbsb * s->imag;
              *(here->BSIM4v0BPspPtr) -= gjbs - gbbsp + gIbtots;
              *(here->BSIM4v0BPbpPtr ) += xcbbb * s->real;
              *(here->BSIM4v0BPbpPtr +1) += xcbbb * s->imag;
              *(here->BSIM4v0BPbpPtr) += gjbd + gjbs - here->BSIM4v0gbbs
				     - gIbtotb - here->BSIM4v0ggidlb;

              if (here->BSIM4v0rbodyMod)
              {   (*(here->BSIM4v0DPdbPtr ) += xcdbdb * s->real);
                  (*(here->BSIM4v0DPdbPtr +1) += xcdbdb * s->imag);
                  (*(here->BSIM4v0DPdbPtr) -= here->BSIM4v0gbd);
                  (*(here->BSIM4v0SPsbPtr ) += xcsbsb * s->real);
                  (*(here->BSIM4v0SPsbPtr +1) += xcsbsb * s->imag);
                  (*(here->BSIM4v0SPsbPtr) -= here->BSIM4v0gbs);

                  (*(here->BSIM4v0DBdpPtr ) += xcdbdb * s->real);
                  (*(here->BSIM4v0DBdpPtr +1) += xcdbdb * s->imag);
                  (*(here->BSIM4v0DBdpPtr) -= here->BSIM4v0gbd);
                  (*(here->BSIM4v0DBdbPtr ) -= xcdbdb * s->real);
                  (*(here->BSIM4v0DBdbPtr +1) -= xcdbdb * s->imag);
                  (*(here->BSIM4v0DBdbPtr) += here->BSIM4v0gbd + here->BSIM4v0grbpd
                                          + here->BSIM4v0grbdb);
                  (*(here->BSIM4v0DBbpPtr) -= here->BSIM4v0grbpd);
                  (*(here->BSIM4v0DBbPtr) -= here->BSIM4v0grbdb);

                  (*(here->BSIM4v0BPdbPtr) -= here->BSIM4v0grbpd);
                  (*(here->BSIM4v0BPbPtr) -= here->BSIM4v0grbpb);
                  (*(here->BSIM4v0BPsbPtr) -= here->BSIM4v0grbps);
                  (*(here->BSIM4v0BPbpPtr) += here->BSIM4v0grbpd + here->BSIM4v0grbps
					  + here->BSIM4v0grbpb);
                  /* WDL: (-here->BSIM4v0gbbs) already added to BPbpPtr */

                  (*(here->BSIM4v0SBspPtr ) += xcsbsb * s->real);
                  (*(here->BSIM4v0SBspPtr +1) += xcsbsb * s->imag);
                  (*(here->BSIM4v0SBspPtr) -= here->BSIM4v0gbs);
                  (*(here->BSIM4v0SBbpPtr) -= here->BSIM4v0grbps);
                  (*(here->BSIM4v0SBbPtr) -= here->BSIM4v0grbsb);
                  (*(here->BSIM4v0SBsbPtr ) -= xcsbsb * s->real);
                  (*(here->BSIM4v0SBsbPtr +1) -= xcsbsb * s->imag);
                  (*(here->BSIM4v0SBsbPtr) += here->BSIM4v0gbs
					  + here->BSIM4v0grbps + here->BSIM4v0grbsb);

                  (*(here->BSIM4v0BdbPtr) -= here->BSIM4v0grbdb);
                  (*(here->BSIM4v0BbpPtr) -= here->BSIM4v0grbpb);
                  (*(here->BSIM4v0BsbPtr) -= here->BSIM4v0grbsb);
                  (*(here->BSIM4v0BbPtr) += here->BSIM4v0grbsb + here->BSIM4v0grbdb
                                        + here->BSIM4v0grbpb);
              }

              if (here->BSIM4v0acnqsMod)
              {   *(here->BSIM4v0QqPtr ) += s->real * ScalingFactor;
                  *(here->BSIM4v0QqPtr +1) += s->imag * ScalingFactor;
                  *(here->BSIM4v0QgpPtr ) -= xcqgb * s->real;
                  *(here->BSIM4v0QgpPtr +1) -= xcqgb * s->imag;
                  *(here->BSIM4v0QdpPtr ) -= xcqdb * s->real;
                  *(here->BSIM4v0QdpPtr +1) -= xcqdb * s->imag;
                  *(here->BSIM4v0QbpPtr ) -= xcqbb * s->real;
                  *(here->BSIM4v0QbpPtr +1) -= xcqbb * s->imag;
                  *(here->BSIM4v0QspPtr ) -= xcqsb * s->real;
                  *(here->BSIM4v0QspPtr +1) -= xcqsb * s->imag;

                  *(here->BSIM4v0GPqPtr) -= here->BSIM4v0gtau;
                  *(here->BSIM4v0DPqPtr) += dxpart * here->BSIM4v0gtau;
                  *(here->BSIM4v0SPqPtr) += sxpart * here->BSIM4v0gtau;

                  *(here->BSIM4v0QqPtr) += here->BSIM4v0gtau;
                  *(here->BSIM4v0QgpPtr) += xgtg;
                  *(here->BSIM4v0QdpPtr) += xgtd;
                  *(here->BSIM4v0QbpPtr) += xgtb;
                  *(here->BSIM4v0QspPtr) += xgts;
              }
         }
    }
    return(OK);
}
