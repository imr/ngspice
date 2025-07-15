/* ******************************************************************************
   *  BSIM4 4.8.3 released on 05/19/2025                                        *
   *  BSIM4 Model Equations                                                     *
   ******************************************************************************

   ******************************************************************************
   *  Copyright (c) 2025 University of California                               *
   *                                                                            *
   *  Project Directors: Prof. Sayeef Salahuddin and Prof. Chenming Hu          *
   *  Developers list: https://www.bsim.berkeley.edu/models/bsim4/auth_bsim4/   *
   ******************************************************************************/

/*
Licensed under Educational Community License, Version 2.0 (the "License"); you may
not use this file except in compliance with the License. You may obtain a copy of the license at
http://opensource.org/licenses/ECL-2.0
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations
under the License.

BSIM4 model is supported by the members of Silicon Integration Initiative's Compact Model Coalition. A link to the most recent version of this
standard can be found at: http://www.si2.org/cmc
*/
#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ngspice/complex.h"
#include "ngspice/sperror.h"
#include "bsim4def.h"
#include "ngspice/suffix.h"

int
BSIM4pzLoad(
GENmodel *inModel,
CKTcircuit *ckt,
SPcomplex *s)
{
BSIM4model *model = (BSIM4model*)inModel;
BSIM4instance *here;

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
double m, mult_i, mult_q;

    for (; model != NULL; model = BSIM4nextModel(model))
    {    for (here = BSIM4instances(model); here!= NULL;
              here = BSIM4nextInstance(here))
     {    pParam = here->pParam;
              capbd = here->BSIM4capbd;
              capbs = here->BSIM4capbs;
              cgso = here->BSIM4cgso;
              cgdo = here->BSIM4cgdo;
              cgbo = pParam->BSIM4cgbo;

              if (here->BSIM4mode >= 0)
              {   Gm = here->BSIM4gm;
                  Gmbs = here->BSIM4gmbs;
                  FwdSum = Gm + Gmbs;
                  RevSum = 0.0;

                  gbbdp = -(here->BSIM4gbds);
                  gbbsp = here->BSIM4gbds + here->BSIM4gbgs + here->BSIM4gbbs;
                  gbdpg = here->BSIM4gbgs;
                  gbdpdp = here->BSIM4gbds;
                  gbdpb = here->BSIM4gbbs;
                  gbdpsp = -(gbdpg + gbdpdp + gbdpb);

                  gbspdp = 0.0;
                  gbspg = 0.0;
                  gbspb = 0.0;
                  gbspsp = 0.0;

                  if (model->BSIM4igcMod)
                  {   gIstotg = here->BSIM4gIgsg + here->BSIM4gIgcsg;
                      gIstotd = here->BSIM4gIgcsd;
                      gIstots = here->BSIM4gIgss + here->BSIM4gIgcss;
                      gIstotb = here->BSIM4gIgcsb;

                      gIdtotg = here->BSIM4gIgdg + here->BSIM4gIgcdg;
                      gIdtotd = here->BSIM4gIgdd + here->BSIM4gIgcdd;
                      gIdtots = here->BSIM4gIgcds;
                      gIdtotb = here->BSIM4gIgcdb;
                  }
                  else
                  {   gIstotg = gIstotd = gIstots = gIstotb = 0.0;
                      gIdtotg = gIdtotd = gIdtots = gIdtotb = 0.0;
                  }

                  if (model->BSIM4igbMod)
                  {   gIbtotg = here->BSIM4gIgbg;
                      gIbtotd = here->BSIM4gIgbd;
                      gIbtots = here->BSIM4gIgbs;
                      gIbtotb = here->BSIM4gIgbb;
                  }
                  else
                      gIbtotg = gIbtotd = gIbtots = gIbtotb = 0.0;

                  if ((model->BSIM4igcMod != 0) || (model->BSIM4igbMod != 0))
                  {   gIgtotg = gIstotg + gIdtotg + gIbtotg;
                      gIgtotd = gIstotd + gIdtotd + gIbtotd ;
                      gIgtots = gIstots + gIdtots + gIbtots;
                      gIgtotb = gIstotb + gIdtotb + gIbtotb;
                  }
                  else
                      gIgtotg = gIgtotd = gIgtots = gIgtotb = 0.0;

                  if (here->BSIM4rgateMod == 2)
                      T0 = *(ckt->CKTstates[0] + here->BSIM4vges)
                         - *(ckt->CKTstates[0] + here->BSIM4vgs);
                  else if (here->BSIM4rgateMod == 3)
                      T0 = *(ckt->CKTstates[0] + here->BSIM4vgms)
                         - *(ckt->CKTstates[0] + here->BSIM4vgs);
                  if (here->BSIM4rgateMod > 1)
                  {   gcrgd = here->BSIM4gcrgd * T0;
                      gcrgg = here->BSIM4gcrgg * T0;
                      gcrgs = here->BSIM4gcrgs * T0;
                      gcrgb = here->BSIM4gcrgb * T0;
                      gcrgg -= here->BSIM4gcrg;
                      gcrg = here->BSIM4gcrg;
                  }
                  else
                      gcrg = gcrgd = gcrgg = gcrgs = gcrgb = 0.0;

                  if (here->BSIM4acnqsMod == 0)
                  {   if (here->BSIM4rgateMod == 3)
                      {   xcgmgmb = cgdo + cgso + pParam->BSIM4cgbo;
                          xcgmdb = -cgdo;
                          xcgmsb = -cgso;
                          xcgmbb = -pParam->BSIM4cgbo;

                          xcdgmb = xcgmdb;
                          xcsgmb = xcgmsb;
                          xcbgmb = xcgmbb;

                          xcggb = here->BSIM4cggb;
                          xcgdb = here->BSIM4cgdb;
                          xcgsb = here->BSIM4cgsb;
                          xcgbb = -(xcggb + xcgdb + xcgsb);

                          xcdgb = here->BSIM4cdgb;
                          xcsgb = -(here->BSIM4cggb + here->BSIM4cbgb
                                + here->BSIM4cdgb);
                          xcbgb = here->BSIM4cbgb;
                      }
                      else
                      {   xcggb = here->BSIM4cggb + cgdo + cgso
                                + pParam->BSIM4cgbo;
                          xcgdb = here->BSIM4cgdb - cgdo;
                          xcgsb = here->BSIM4cgsb - cgso;
                          xcgbb = -(xcggb + xcgdb + xcgsb);

                          xcdgb = here->BSIM4cdgb - cgdo;
                          xcsgb = -(here->BSIM4cggb + here->BSIM4cbgb
                                + here->BSIM4cdgb + cgso);
                          xcbgb = here->BSIM4cbgb - pParam->BSIM4cgbo;

                          xcdgmb = xcsgmb = xcbgmb = 0.0;
                      }
                      xcddb = here->BSIM4cddb + here->BSIM4capbd + cgdo;
                      xcdsb = here->BSIM4cdsb;

                      xcsdb = -(here->BSIM4cgdb + here->BSIM4cbdb
                            + here->BSIM4cddb);
                      xcssb = here->BSIM4capbs + cgso - (here->BSIM4cgsb
                            + here->BSIM4cbsb + here->BSIM4cdsb);

                      if (!here->BSIM4rbodyMod)
                      {   xcdbb = -(xcdgb + xcddb + xcdsb + xcdgmb);
                          xcsbb = -(xcsgb + xcsdb + xcssb + xcsgmb);
                          xcbdb = here->BSIM4cbdb - here->BSIM4capbd;
                          xcbsb = here->BSIM4cbsb - here->BSIM4capbs;
                          xcdbdb = 0.0;
                      }
                      else
                      {   xcdbb  = -(here->BSIM4cddb + here->BSIM4cdgb
                                 + here->BSIM4cdsb);
                          xcsbb = -(xcsgb + xcsdb + xcssb + xcsgmb)
                                + here->BSIM4capbs;
                          xcbdb = here->BSIM4cbdb;
                          xcbsb = here->BSIM4cbsb;

                          xcdbdb = -here->BSIM4capbd;
                          xcsbsb = -here->BSIM4capbs;
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

              xgtg = here->BSIM4gtg;
                      xgtd = here->BSIM4gtd;
                      xgts = here->BSIM4gts;
                      xgtb = here->BSIM4gtb;

                      xcqgb = here->BSIM4cqgb;
                      xcqdb = here->BSIM4cqdb;
                      xcqsb = here->BSIM4cqsb;
                      xcqbb = here->BSIM4cqbb;

              CoxWL = model->BSIM4coxe * here->pParam->BSIM4weffCV
                            * here->BSIM4nf * here->pParam->BSIM4leffCV;
              qcheq = -(here->BSIM4qgate + here->BSIM4qbulk);
              if (fabs(qcheq) <= 1.0e-5 * CoxWL)
              {   if (model->BSIM4xpart < 0.5)
                  {   dxpart = 0.4;
                  }
                  else if (model->BSIM4xpart > 0.5)
                  {   dxpart = 0.0;
                  }
                  else
                  {   dxpart = 0.5;
                  }
                  ddxpart_dVd = ddxpart_dVg = ddxpart_dVb
                      = ddxpart_dVs = 0.0;
              }
              else
              {   dxpart = here->BSIM4qdrn / qcheq;
                  Cdd = here->BSIM4cddb;
                  Csd = -(here->BSIM4cgdb + here->BSIM4cddb
                  + here->BSIM4cbdb);
                  ddxpart_dVd = (Cdd - dxpart * (Cdd + Csd)) / qcheq;
                  Cdg = here->BSIM4cdgb;
                  Csg = -(here->BSIM4cggb + here->BSIM4cdgb
                  + here->BSIM4cbgb);
                  ddxpart_dVg = (Cdg - dxpart * (Cdg + Csg)) / qcheq;

                  Cds = here->BSIM4cdsb;
                  Css = -(here->BSIM4cgsb + here->BSIM4cdsb
                  + here->BSIM4cbsb);
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
              {   Gm = -here->BSIM4gm;
                  Gmbs = -here->BSIM4gmbs;
                  FwdSum = 0.0;
                  RevSum = -(Gm + Gmbs);

                  gbbsp = -(here->BSIM4gbds);
                  gbbdp = here->BSIM4gbds + here->BSIM4gbgs + here->BSIM4gbbs;

                  gbdpg = 0.0;
                  gbdpsp = 0.0;
                  gbdpb = 0.0;
                  gbdpdp = 0.0;

                  gbspg = here->BSIM4gbgs;
                  gbspsp = here->BSIM4gbds;
                  gbspb = here->BSIM4gbbs;
                  gbspdp = -(gbspg + gbspsp + gbspb);

                  if (model->BSIM4igcMod)
                  {   gIstotg = here->BSIM4gIgsg + here->BSIM4gIgcdg;
                      gIstotd = here->BSIM4gIgcds;
                      gIstots = here->BSIM4gIgss + here->BSIM4gIgcdd;
                      gIstotb = here->BSIM4gIgcdb;

                      gIdtotg = here->BSIM4gIgdg + here->BSIM4gIgcsg;
                      gIdtotd = here->BSIM4gIgdd + here->BSIM4gIgcss;
                      gIdtots = here->BSIM4gIgcsd;
                      gIdtotb = here->BSIM4gIgcsb;
                  }
                  else
                  {   gIstotg = gIstotd = gIstots = gIstotb = 0.0;
                      gIdtotg = gIdtotd = gIdtots = gIdtotb  = 0.0;
                  }

                  if (model->BSIM4igbMod)
                  {   gIbtotg = here->BSIM4gIgbg;
                      gIbtotd = here->BSIM4gIgbs;
                      gIbtots = here->BSIM4gIgbd;
                      gIbtotb = here->BSIM4gIgbb;
                  }
                  else
                      gIbtotg = gIbtotd = gIbtots = gIbtotb = 0.0;

                  if ((model->BSIM4igcMod != 0) || (model->BSIM4igbMod != 0))
                  {   gIgtotg = gIstotg + gIdtotg + gIbtotg;
                      gIgtotd = gIstotd + gIdtotd + gIbtotd ;
                      gIgtots = gIstots + gIdtots + gIbtots;
                      gIgtotb = gIstotb + gIdtotb + gIbtotb;
                  }
                  else
                      gIgtotg = gIgtotd = gIgtots = gIgtotb = 0.0;

                  if (here->BSIM4rgateMod == 2)
                      T0 = *(ckt->CKTstates[0] + here->BSIM4vges)
                         - *(ckt->CKTstates[0] + here->BSIM4vgs);
                  else if (here->BSIM4rgateMod == 3)
                      T0 = *(ckt->CKTstates[0] + here->BSIM4vgms)
                         - *(ckt->CKTstates[0] + here->BSIM4vgs);
                  if (here->BSIM4rgateMod > 1)
                  {   gcrgd = here->BSIM4gcrgs * T0;
                      gcrgg = here->BSIM4gcrgg * T0;
                      gcrgs = here->BSIM4gcrgd * T0;
                      gcrgb = here->BSIM4gcrgb * T0;
                      gcrgg -= here->BSIM4gcrg;
                      gcrg = here->BSIM4gcrg;
                  }
                  else
                      gcrg = gcrgd = gcrgg = gcrgs = gcrgb = 0.0;

                  if (here->BSIM4acnqsMod == 0)
                  {   if (here->BSIM4rgateMod == 3)
                      {   xcgmgmb = cgdo + cgso + pParam->BSIM4cgbo;
                          xcgmdb = -cgdo;
                          xcgmsb = -cgso;
                          xcgmbb = -pParam->BSIM4cgbo;

                          xcdgmb = xcgmdb;
                          xcsgmb = xcgmsb;
                          xcbgmb = xcgmbb;

                          xcggb = here->BSIM4cggb;
                          xcgdb = here->BSIM4cgsb;
                          xcgsb = here->BSIM4cgdb;
                          xcgbb = -(xcggb + xcgdb + xcgsb);

                          xcdgb = -(here->BSIM4cggb + here->BSIM4cbgb
                                + here->BSIM4cdgb);
                          xcsgb = here->BSIM4cdgb;
                          xcbgb = here->BSIM4cbgb;
                      }
                      else
                      {   xcggb = here->BSIM4cggb + cgdo + cgso
                                + pParam->BSIM4cgbo;
                          xcgdb = here->BSIM4cgsb - cgdo;
                          xcgsb = here->BSIM4cgdb - cgso;
                          xcgbb = -(xcggb + xcgdb + xcgsb);

                          xcdgb = -(here->BSIM4cggb + here->BSIM4cbgb
                                + here->BSIM4cdgb + cgdo);
                          xcsgb = here->BSIM4cdgb - cgso;
                          xcbgb = here->BSIM4cbgb - pParam->BSIM4cgbo;

                          xcdgmb = xcsgmb = xcbgmb = 0.0;
                      }
                      xcddb = here->BSIM4capbd + cgdo - (here->BSIM4cgsb
                            + here->BSIM4cbsb + here->BSIM4cdsb);
                      xcdsb = -(here->BSIM4cgdb + here->BSIM4cbdb
                            + here->BSIM4cddb);

                      xcsdb = here->BSIM4cdsb;
                      xcssb = here->BSIM4cddb + here->BSIM4capbs + cgso;

                      if (!here->BSIM4rbodyMod)
                      {   xcdbb = -(xcdgb + xcddb + xcdsb + xcdgmb);
                          xcsbb = -(xcsgb + xcsdb + xcssb + xcsgmb);
                          xcbdb = here->BSIM4cbsb - here->BSIM4capbd;
                          xcbsb = here->BSIM4cbdb - here->BSIM4capbs;
                          xcdbdb = 0.0;
                      }
                      else
                      {   xcdbb = -(xcdgb + xcddb + xcdsb + xcdgmb)
                                + here->BSIM4capbd;
                          xcsbb = -(here->BSIM4cddb + here->BSIM4cdgb
                                + here->BSIM4cdsb);
                          xcbdb = here->BSIM4cbsb;
                          xcbsb = here->BSIM4cbdb;
                          xcdbdb = -here->BSIM4capbd;
                          xcsbsb = -here->BSIM4capbs;
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

                      xgtg = here->BSIM4gtg;
                      xgtd = here->BSIM4gts;
                      xgts = here->BSIM4gtd;
                      xgtb = here->BSIM4gtb;

                      xcqgb = here->BSIM4cqgb;
                      xcqdb = here->BSIM4cqsb;
                      xcqsb = here->BSIM4cqdb;
                      xcqbb = here->BSIM4cqbb;

              CoxWL = model->BSIM4coxe * here->pParam->BSIM4weffCV
                            * here->BSIM4nf * here->pParam->BSIM4leffCV;
              qcheq = -(here->BSIM4qgate + here->BSIM4qbulk);
              if (fabs(qcheq) <= 1.0e-5 * CoxWL)
              {   if (model->BSIM4xpart < 0.5)
                  {   sxpart = 0.4;
                  }
                  else if (model->BSIM4xpart > 0.5)
                  {   sxpart = 0.0;
                  }
                  else
                  {   sxpart = 0.5;
                  }
                  dsxpart_dVd = dsxpart_dVg = dsxpart_dVb
                      = dsxpart_dVs = 0.0;
              }
              else
              {   sxpart = here->BSIM4qdrn / qcheq;
                  Css = here->BSIM4cddb;
                  Cds = -(here->BSIM4cgdb + here->BSIM4cddb
                  + here->BSIM4cbdb);
                  dsxpart_dVs = (Css - sxpart * (Css + Cds)) / qcheq;
                  Csg = here->BSIM4cdgb;
                  Cdg = -(here->BSIM4cggb + here->BSIM4cdgb
                  + here->BSIM4cbgb);
                  dsxpart_dVg = (Csg - sxpart * (Csg + Cdg)) / qcheq;

                  Csd = here->BSIM4cdsb;
                  Cdd = -(here->BSIM4cgsb + here->BSIM4cdsb
                  + here->BSIM4cbsb);
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

              if (model->BSIM4rdsMod == 1)
              {   gstot = here->BSIM4gstot;
                  gstotd = here->BSIM4gstotd;
                  gstotg = here->BSIM4gstotg;
                  gstots = here->BSIM4gstots - gstot;
                  gstotb = here->BSIM4gstotb;

                  gdtot = here->BSIM4gdtot;
                  gdtotd = here->BSIM4gdtotd - gdtot;
                  gdtotg = here->BSIM4gdtotg;
                  gdtots = here->BSIM4gdtots;
                  gdtotb = here->BSIM4gdtotb;
              }
              else
              {   gstot = gstotd = gstotg = gstots = gstotb = 0.0;
                  gdtot = gdtotd = gdtotg = gdtots = gdtotb = 0.0;
              }


              T1 = *(ckt->CKTstate0 + here->BSIM4qdef) * here->BSIM4gtau;
              gds = here->BSIM4gds;

              /*
               * Loading PZ matrix
               */
              m = here->BSIM4m;
              mult_i = here->BSIM4mult_i;
              mult_q = here->BSIM4mult_q;

              if (!model->BSIM4rdsMod)
              {   gdpr = here->BSIM4drainConductance;
                  gspr = here->BSIM4sourceConductance;
              }
              else
                  gdpr = gspr = 0.0;

              if (!here->BSIM4rbodyMod)
              {   gjbd = here->BSIM4gbd;
                  gjbs = here->BSIM4gbs;
              }
              else
                  gjbd = gjbs = 0.0;

              geltd = here->BSIM4grgeltd;

              if (here->BSIM4rgateMod == 1)
              {   *(here->BSIM4GEgePtr) += mult_i * geltd;
                  *(here->BSIM4GPgePtr) -= mult_i * geltd;
                  *(here->BSIM4GEgpPtr) -= mult_i * geltd;

                  *(here->BSIM4GPgpPtr ) += mult_q * xcggb * s->real;
                  *(here->BSIM4GPgpPtr +1) += mult_q * xcggb * s->imag;
                  *(here->BSIM4GPgpPtr) += mult_i * (geltd + gIgtotg) - mult_q * xgtg;
                  *(here->BSIM4GPdpPtr ) += mult_q * xcgdb * s->real;
                  *(here->BSIM4GPdpPtr +1) += mult_q * xcgdb * s->imag;
                  *(here->BSIM4GPdpPtr) -= mult_q * xgtd - mult_i * gIgtotd;
                  *(here->BSIM4GPspPtr ) += mult_q * xcgsb * s->real;
                  *(here->BSIM4GPspPtr +1) += mult_q * xcgsb * s->imag;
                  *(here->BSIM4GPspPtr) -= mult_q * xgts - mult_i * gIgtots;
                  *(here->BSIM4GPbpPtr ) += mult_q * xcgbb * s->real;
                  *(here->BSIM4GPbpPtr +1) += mult_q * xcgbb * s->imag;
                  *(here->BSIM4GPbpPtr) -= mult_q * xgtb - mult_i * gIgtotb;
              }
              else if (here->BSIM4rgateMod == 2)
              {   *(here->BSIM4GEgePtr) += mult_i * gcrg;
                  *(here->BSIM4GEgpPtr) += mult_i * gcrgg;
                  *(here->BSIM4GEdpPtr) += mult_i * gcrgd;
                  *(here->BSIM4GEspPtr) += mult_i * gcrgs;
                  *(here->BSIM4GEbpPtr) += mult_i * gcrgb;

                  *(here->BSIM4GPgePtr) -= mult_i * gcrg;
                  *(here->BSIM4GPgpPtr ) += mult_q * xcggb * s->real;
                  *(here->BSIM4GPgpPtr +1) += mult_q * xcggb * s->imag;
                  *(here->BSIM4GPgpPtr) -= mult_i * (gcrgg - gIgtotg) + mult_q * xgtg;
                  *(here->BSIM4GPdpPtr ) += mult_q * xcgdb * s->real;
                  *(here->BSIM4GPdpPtr +1) += mult_q * xcgdb * s->imag;
                  *(here->BSIM4GPdpPtr) -= mult_i * (gcrgd - gIgtotd) + mult_q * xgtd;
                  *(here->BSIM4GPspPtr ) += mult_q * xcgsb * s->real;
                  *(here->BSIM4GPspPtr +1) += mult_q * xcgsb * s->imag;
                  *(here->BSIM4GPspPtr) -= mult_i * (gcrgs - gIgtots) + mult_q * xgts;
                  *(here->BSIM4GPbpPtr ) += mult_q * xcgbb * s->real;
                  *(here->BSIM4GPbpPtr +1) += mult_q * xcgbb * s->imag;
                  *(here->BSIM4GPbpPtr) -= mult_i * (gcrgb - gIgtotb) + mult_q * xgtb;
              }
              else if (here->BSIM4rgateMod == 3)
              {   *(here->BSIM4GEgePtr) += mult_i * geltd;
                  *(here->BSIM4GEgmPtr) -= mult_i * geltd;
                  *(here->BSIM4GMgePtr) -= mult_i * geltd;
                  *(here->BSIM4GMgmPtr) += mult_i * (geltd + gcrg);
                  *(here->BSIM4GMgmPtr ) += mult_q * xcgmgmb * s->real;
                  *(here->BSIM4GMgmPtr +1) += mult_q * xcgmgmb * s->imag;

                  *(here->BSIM4GMdpPtr) += mult_i * gcrgd;
                  *(here->BSIM4GMdpPtr ) += mult_q * xcgmdb * s->real;
                  *(here->BSIM4GMdpPtr +1) += mult_q * xcgmdb * s->imag;
                  *(here->BSIM4GMgpPtr) += mult_i * gcrgg;
                  *(here->BSIM4GMspPtr) += mult_i * gcrgs;
                  *(here->BSIM4GMspPtr ) += mult_q * xcgmsb * s->real;
                  *(here->BSIM4GMspPtr +1) += mult_q * xcgmsb * s->imag;
                  *(here->BSIM4GMbpPtr) += mult_i * gcrgb;
                  *(here->BSIM4GMbpPtr ) += mult_q * xcgmbb * s->real;
                  *(here->BSIM4GMbpPtr +1) += mult_q * xcgmbb * s->imag;

                  *(here->BSIM4DPgmPtr ) += mult_q * xcdgmb * s->real;
                  *(here->BSIM4DPgmPtr +1) += mult_q * xcdgmb * s->imag;
                  *(here->BSIM4GPgmPtr) -= mult_i * gcrg;
                  *(here->BSIM4SPgmPtr ) += mult_q * xcsgmb * s->real;
                  *(here->BSIM4SPgmPtr +1) += mult_q * xcsgmb * s->imag;
                  *(here->BSIM4BPgmPtr ) += mult_q * xcbgmb * s->real;
                  *(here->BSIM4BPgmPtr +1) += mult_q * xcbgmb * s->imag;

                  *(here->BSIM4GPgpPtr) -= mult_i * (gcrgg - gIgtotg) + mult_q * xgtg;
                  *(here->BSIM4GPgpPtr ) += mult_q * xcggb * s->real;
                  *(here->BSIM4GPgpPtr +1) += mult_q * xcggb * s->imag;
                  *(here->BSIM4GPdpPtr) -= mult_i * (gcrgd - gIgtotd) + mult_q * xgtd;
                  *(here->BSIM4GPdpPtr ) += mult_q * xcgdb * s->real;
                  *(here->BSIM4GPdpPtr +1) += mult_q * xcgdb * s->imag;
                  *(here->BSIM4GPspPtr) -= mult_i * (gcrgs - gIgtots) + mult_q * xgts;
                  *(here->BSIM4GPspPtr ) += mult_q * xcgsb * s->real;
                  *(here->BSIM4GPspPtr +1) += mult_q * xcgsb * s->imag;
                  *(here->BSIM4GPbpPtr) -= mult_i * (gcrgb - gIgtotb) + mult_q * xgtb;
                  *(here->BSIM4GPbpPtr ) += mult_q * xcgbb * s->real;
                  *(here->BSIM4GPbpPtr +1) += mult_q * xcgbb * s->imag;
              }
              else
              {   *(here->BSIM4GPdpPtr ) += mult_q * xcgdb * s->real;
                  *(here->BSIM4GPdpPtr +1) += mult_q * xcgdb * s->imag;
                  *(here->BSIM4GPdpPtr) -= mult_q * xgtd - mult_i * gIgtotd;
                  *(here->BSIM4GPgpPtr ) += mult_q * xcggb * s->real;
                  *(here->BSIM4GPgpPtr +1) += mult_q * xcggb * s->imag;
                  *(here->BSIM4GPgpPtr) -= mult_q * xgtg - mult_i * gIgtotg;
                  *(here->BSIM4GPspPtr ) += mult_q * xcgsb * s->real;
                  *(here->BSIM4GPspPtr +1) += mult_q * xcgsb * s->imag;
                  *(here->BSIM4GPspPtr) -= mult_q * xgts - mult_i * gIgtots;
                  *(here->BSIM4GPbpPtr ) += mult_q * xcgbb * s->real;
                  *(here->BSIM4GPbpPtr +1) += mult_q * xcgbb * s->imag;
                  *(here->BSIM4GPbpPtr) -= mult_q * xgtb - mult_i * gIgtotb;
              }

              if (model->BSIM4rdsMod)
              {   (*(here->BSIM4DgpPtr) += mult_i * gdtotg);
                  (*(here->BSIM4DspPtr) += mult_i * gdtots);
                  (*(here->BSIM4DbpPtr) += mult_i * gdtotb);
                  (*(here->BSIM4SdpPtr) += mult_i * gstotd);
                  (*(here->BSIM4SgpPtr) += mult_i * gstotg);
                  (*(here->BSIM4SbpPtr) += mult_i * gstotb);
              }

              *(here->BSIM4DPdpPtr ) += mult_q * xcddb * s->real;
              *(here->BSIM4DPdpPtr +1) += mult_q * xcddb * s->imag;
              *(here->BSIM4DPdpPtr) += mult_i * (gdpr + gds + here->BSIM4gbd
                     - gdtotd + RevSum + gbdpdp - gIdtotd)
                     + mult_q * (dxpart * xgtd + T1 * ddxpart_dVd);
              *(here->BSIM4DPdPtr) -= mult_i * (gdpr + gdtot);
              *(here->BSIM4DPgpPtr ) += mult_q * xcdgb * s->real;
              *(here->BSIM4DPgpPtr +1) += mult_q * xcdgb * s->imag;
              *(here->BSIM4DPgpPtr) += mult_i * (Gm - gdtotg + gbdpg - gIdtotg)
                     + mult_q * (T1 * ddxpart_dVg + dxpart * xgtg);
              *(here->BSIM4DPspPtr ) += mult_q * xcdsb * s->real;
              *(here->BSIM4DPspPtr +1) += mult_q * xcdsb * s->imag;
              *(here->BSIM4DPspPtr) -= mult_i * (gds + FwdSum + gdtots - gbdpsp + gIdtots)
                     - mult_q * (T1 * ddxpart_dVs + dxpart * xgts);
              *(here->BSIM4DPbpPtr ) += mult_q * xcdbb * s->real;
              *(here->BSIM4DPbpPtr +1) += mult_q * xcdbb * s->imag;
              *(here->BSIM4DPbpPtr) -= mult_i * (gjbd + gdtotb - Gmbs - gbdpb + gIdtotb)
                     - mult_q * T1 * (ddxpart_dVb + dxpart * xgtb);

              *(here->BSIM4DdpPtr) -= mult_i * (gdpr - gdtotd);
              *(here->BSIM4DdPtr) += mult_i * (gdpr + gdtot);

              *(here->BSIM4SPdpPtr ) += mult_q * xcsdb * s->real;
              *(here->BSIM4SPdpPtr +1) += mult_q * xcsdb * s->imag;
              *(here->BSIM4SPdpPtr) -= mult_i * (gds + gstotd + RevSum - gbspdp + gIstotd)
                     - mult_q * (T1 * dsxpart_dVd + sxpart * xgtd);
              *(here->BSIM4SPgpPtr ) += mult_q * xcsgb * s->real;
              *(here->BSIM4SPgpPtr +1) += mult_q * xcsgb * s->imag;
              *(here->BSIM4SPgpPtr) -= mult_i * (Gm + gstotg - gbspg + gIstotg)
                     - mult_q * (T1 * dsxpart_dVg + sxpart * xgtg);
              *(here->BSIM4SPspPtr ) += mult_q * xcssb * s->real;
              *(here->BSIM4SPspPtr +1) += mult_q * xcssb * s->imag;
              *(here->BSIM4SPspPtr) += mult_i * (gspr + gds + here->BSIM4gbs - gIstots
                     - gstots + FwdSum + gbspsp)
                     + mult_q * (sxpart * xgts + T1 * dsxpart_dVs);
              *(here->BSIM4SPsPtr) -= mult_i * (gspr + gstot);
              *(here->BSIM4SPbpPtr ) += mult_q * xcsbb * s->real;
              *(here->BSIM4SPbpPtr +1) += mult_q * xcsbb * s->imag;
              *(here->BSIM4SPbpPtr) -= mult_i * (gjbs + gstotb + Gmbs - gbspb + gIstotb)
                     - mult_q * (T1 * dsxpart_dVb + sxpart * xgtb);

              *(here->BSIM4SspPtr) -= mult_i * (gspr - gstots);
              *(here->BSIM4SsPtr) += mult_i * (gspr + gstot);

              *(here->BSIM4BPdpPtr ) += mult_q * xcbdb * s->real;
              *(here->BSIM4BPdpPtr +1) += mult_q * xcbdb * s->imag;
              *(here->BSIM4BPdpPtr) -= mult_i * (gjbd - gbbdp + gIbtotd);
              *(here->BSIM4BPgpPtr ) += mult_q * xcbgb * s->real;
              *(here->BSIM4BPgpPtr +1) += mult_q * xcbgb * s->imag;
              *(here->BSIM4BPgpPtr) -= mult_i * (here->BSIM4gbgs + gIbtotg);
              *(here->BSIM4BPspPtr ) += mult_q * xcbsb * s->real;
              *(here->BSIM4BPspPtr +1) += mult_q * xcbsb * s->imag;
              *(here->BSIM4BPspPtr) -= mult_i * (gjbs - gbbsp + gIbtots);
              *(here->BSIM4BPbpPtr ) += mult_q * xcbbb * s->real;
              *(here->BSIM4BPbpPtr +1) += mult_q * xcbbb * s->imag;
              *(here->BSIM4BPbpPtr) += mult_i * (gjbd + gjbs - here->BSIM4gbbs
                     - gIbtotb);
              ggidld = here->BSIM4ggidld;
              ggidlg = here->BSIM4ggidlg;
              ggidlb = here->BSIM4ggidlb;
              ggislg = here->BSIM4ggislg;
              ggisls = here->BSIM4ggisls;
              ggislb = here->BSIM4ggislb;

              /* stamp gidl */
              (*(here->BSIM4DPdpPtr) += mult_i * ggidld);
              (*(here->BSIM4DPgpPtr) += mult_i * ggidlg);
              (*(here->BSIM4DPspPtr) -= mult_i * ((ggidlg + ggidld) + ggidlb));
              (*(here->BSIM4DPbpPtr) += mult_i * ggidlb);
              (*(here->BSIM4BPdpPtr) -= mult_i * ggidld);
              (*(here->BSIM4BPgpPtr) -= mult_i * ggidlg);
              (*(here->BSIM4BPspPtr) += mult_i * ((ggidlg + ggidld) + ggidlb));
              (*(here->BSIM4BPbpPtr) -= mult_i * ggidlb);
               /* stamp gisl */
              (*(here->BSIM4SPdpPtr) -= mult_i * ((ggisls + ggislg) + ggislb));
              (*(here->BSIM4SPgpPtr) += mult_i * ggislg);
              (*(here->BSIM4SPspPtr) += mult_i * ggisls);
              (*(here->BSIM4SPbpPtr) += mult_i * ggislb);
              (*(here->BSIM4BPdpPtr) += mult_i * ((ggislg + ggisls) + ggislb));
              (*(here->BSIM4BPgpPtr) -= mult_i * ggislg);
              (*(here->BSIM4BPspPtr) -= mult_i * ggisls);
              (*(here->BSIM4BPbpPtr) -= mult_i * ggislb);

              if (here->BSIM4rbodyMod)
              {   (*(here->BSIM4DPdbPtr ) += mult_q * xcdbdb * s->real);
                  (*(here->BSIM4DPdbPtr +1) += mult_q * xcdbdb * s->imag);
                  (*(here->BSIM4DPdbPtr) -= mult_i * here->BSIM4gbd);
                  (*(here->BSIM4SPsbPtr ) += mult_q * xcsbsb * s->real);
                  (*(here->BSIM4SPsbPtr +1) += mult_q * xcsbsb * s->imag);
                  (*(here->BSIM4SPsbPtr) -= mult_i * here->BSIM4gbs);

                  (*(here->BSIM4DBdpPtr ) += mult_q * xcdbdb * s->real);
                  (*(here->BSIM4DBdpPtr +1) += mult_q * xcdbdb * s->imag);
                  (*(here->BSIM4DBdpPtr) -= mult_i * here->BSIM4gbd);
                  (*(here->BSIM4DBdbPtr ) -= mult_q * xcdbdb * s->real);
                  (*(here->BSIM4DBdbPtr +1) -= mult_q * xcdbdb * s->imag);
                  (*(here->BSIM4DBdbPtr) += mult_i * (here->BSIM4gbd + here->BSIM4grbpd
                                          + here->BSIM4grbdb));
                  (*(here->BSIM4DBbpPtr) -= mult_i * here->BSIM4grbpd);
                  (*(here->BSIM4DBbPtr) -= mult_i * here->BSIM4grbdb);

                  (*(here->BSIM4BPdbPtr) -= mult_i * here->BSIM4grbpd);
                  (*(here->BSIM4BPbPtr) -= mult_i * here->BSIM4grbpb);
                  (*(here->BSIM4BPsbPtr) -= mult_i * here->BSIM4grbps);
                  (*(here->BSIM4BPbpPtr) += mult_i * (here->BSIM4grbpd + here->BSIM4grbps
                      + here->BSIM4grbpb));
                  /* WDL: (-here->BSIM4gbbs) already added to BPbpPtr */

                  (*(here->BSIM4SBspPtr ) += mult_q * xcsbsb * s->real);
                  (*(here->BSIM4SBspPtr +1) += mult_q * xcsbsb * s->imag);
                  (*(here->BSIM4SBspPtr) -= mult_i * here->BSIM4gbs);
                  (*(here->BSIM4SBbpPtr) -= mult_i * here->BSIM4grbps);
                  (*(here->BSIM4SBbPtr) -= mult_i * here->BSIM4grbsb);
                  (*(here->BSIM4SBsbPtr ) -= mult_q * xcsbsb * s->real);
                  (*(here->BSIM4SBsbPtr +1) -= mult_q * xcsbsb * s->imag);
                  (*(here->BSIM4SBsbPtr) += mult_i * (here->BSIM4gbs
                      + here->BSIM4grbps + here->BSIM4grbsb));

                  (*(here->BSIM4BdbPtr) -= mult_i * here->BSIM4grbdb);
                  (*(here->BSIM4BbpPtr) -= mult_i * here->BSIM4grbpb);
                  (*(here->BSIM4BsbPtr) -= mult_i * here->BSIM4grbsb);
                  (*(here->BSIM4BbPtr) += mult_i * (here->BSIM4grbsb + here->BSIM4grbdb
                                        + here->BSIM4grbpb));
              }

              if (here->BSIM4acnqsMod)
              {   *(here->BSIM4QqPtr ) += m * s->real * ScalingFactor;
                  *(here->BSIM4QqPtr +1) += m * s->imag * ScalingFactor;
                  *(here->BSIM4QgpPtr ) -= m * xcqgb * s->real;
                  *(here->BSIM4QgpPtr +1) -= m * xcqgb * s->imag;
                  *(here->BSIM4QdpPtr ) -= m * xcqdb * s->real;
                  *(here->BSIM4QdpPtr +1) -= m * xcqdb * s->imag;
                  *(here->BSIM4QbpPtr ) -= m * xcqbb * s->real;
                  *(here->BSIM4QbpPtr +1) -= m * xcqbb * s->imag;
                  *(here->BSIM4QspPtr ) -= m * xcqsb * s->real;
                  *(here->BSIM4QspPtr +1) -= m * xcqsb * s->imag;

                  *(here->BSIM4GPqPtr) -= m * here->BSIM4gtau;
                  *(here->BSIM4DPqPtr) += m * dxpart * here->BSIM4gtau;
                  *(here->BSIM4SPqPtr) += m * sxpart * here->BSIM4gtau;

                  *(here->BSIM4QqPtr) += m * here->BSIM4gtau;
                  *(here->BSIM4QgpPtr) += m * xgtg;
                  *(here->BSIM4QdpPtr) += m * xgtd;
                  *(here->BSIM4QbpPtr) += m * xgtb;
                  *(here->BSIM4QspPtr) += m * xgts;
              }
         }
    }
    return(OK);
}
