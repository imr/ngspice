/**** BSIM3v3.2.4, Released by Xuemei Xi 12/14/2001 ****/

/**********
 * Copyright 2001 Regents of the University of California. All rights reserved.
 * File: b3acld.c of BSIM3v3.2.4
 * Author: 1995 Min-Chie Jeng and Mansun Chan
 * Author: 1997-1999 Weidong Liu.
 * Author: 2001  Xuemei Xi
 * Modified by Paolo Nenzi 2002 and Dietmar Warning 2003
 **********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "bsim3v32def.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
BSIM3v32acLoad (GENmodel *inModel, CKTcircuit *ckt)
{
BSIM3v32model *model = (BSIM3v32model*)inModel;
BSIM3v32instance *here;
double xcggb, xcgdb, xcgsb, xcbgb, xcbdb, xcbsb, xcddb, xcssb, xcdgb;
double gdpr, gspr, gds, gbd, gbs, capbd, capbs, xcsgb, xcdsb, xcsdb;
double cggb, cgdb, cgsb, cbgb, cbdb, cbsb, cddb, cdgb, cdsb, omega;
double GSoverlapCap, GDoverlapCap, GBoverlapCap, FwdSum, RevSum, Gm, Gmbs;
double dxpart, sxpart, xgtg, xgtd, xgts, xgtb;
double xcqgb = 0.0, xcqdb = 0.0, xcqsb = 0.0, xcqbb = 0.0;
double gbspsp, gbbdp, gbbsp, gbspg, gbspb;
double gbspdp, gbdpdp, gbdpg, gbdpb, gbdpsp;
double ddxpart_dVd, ddxpart_dVg, ddxpart_dVb, ddxpart_dVs;
double dsxpart_dVd, dsxpart_dVg, dsxpart_dVb, dsxpart_dVs;
double T1, CoxWL, qcheq, Cdg, Cdd, Cds, Csg, Csd, Css;
double ScalingFactor = 1.0e-9;
double m;

    omega = ckt->CKTomega;
    for (; model != NULL; model = BSIM3v32nextModel(model))
    {    for (here = BSIM3v32instances(model); here!= NULL;
              here = BSIM3v32nextInstance(here))
         {
              if (here->BSIM3v32mode >= 0)
              {   Gm = here->BSIM3v32gm;
                  Gmbs = here->BSIM3v32gmbs;
                  FwdSum = Gm + Gmbs;
                  RevSum = 0.0;

                  gbbdp = -here->BSIM3v32gbds;
                  gbbsp = here->BSIM3v32gbds + here->BSIM3v32gbgs + here->BSIM3v32gbbs;

                  gbdpg = here->BSIM3v32gbgs;
                  gbdpb = here->BSIM3v32gbbs;
                  gbdpdp = here->BSIM3v32gbds;
                  gbdpsp = -(gbdpg + gbdpb + gbdpdp);

                  gbspdp = 0.0;
                  gbspg = 0.0;
                  gbspb = 0.0;
                  gbspsp = 0.0;

                  if (here->BSIM3v32nqsMod == 0)
                  {   cggb = here->BSIM3v32cggb;
                      cgsb = here->BSIM3v32cgsb;
                      cgdb = here->BSIM3v32cgdb;

                      cbgb = here->BSIM3v32cbgb;
                      cbsb = here->BSIM3v32cbsb;
                      cbdb = here->BSIM3v32cbdb;

                      cdgb = here->BSIM3v32cdgb;
                      cdsb = here->BSIM3v32cdsb;
                      cddb = here->BSIM3v32cddb;

                      xgtg = xgtd = xgts = xgtb = 0.0;
                      sxpart = 0.6;
                      dxpart = 0.4;
                      ddxpart_dVd = ddxpart_dVg = ddxpart_dVb
                                  = ddxpart_dVs = 0.0;
                      dsxpart_dVd = dsxpart_dVg = dsxpart_dVb
                                  = dsxpart_dVs = 0.0;
                  }
                  else
                  {   cggb = cgdb = cgsb = 0.0;
                      cbgb = cbdb = cbsb = 0.0;
                      cdgb = cddb = cdsb = 0.0;

                      xgtg = here->BSIM3v32gtg;
                      xgtd = here->BSIM3v32gtd;
                      xgts = here->BSIM3v32gts;
                      xgtb = here->BSIM3v32gtb;

                      xcqgb = here->BSIM3v32cqgb * omega;
                      xcqdb = here->BSIM3v32cqdb * omega;
                      xcqsb = here->BSIM3v32cqsb * omega;
                      xcqbb = here->BSIM3v32cqbb * omega;

                      CoxWL = model->BSIM3v32cox * here->pParam->BSIM3v32weffCV
                            * here->pParam->BSIM3v32leffCV;
                      qcheq = -(here->BSIM3v32qgate + here->BSIM3v32qbulk);
                      if (fabs(qcheq) <= 1.0e-5 * CoxWL)
                      {   if (model->BSIM3v32xpart < 0.5)
                          {   dxpart = 0.4;
                          }
                          else if (model->BSIM3v32xpart > 0.5)
                          {   dxpart = 0.0;
                          }
                          else
                          {   dxpart = 0.5;
                          }
                          ddxpart_dVd = ddxpart_dVg = ddxpart_dVb
                                      = ddxpart_dVs = 0.0;
                      }
                      else
                      {   dxpart = here->BSIM3v32qdrn / qcheq;
                          Cdd = here->BSIM3v32cddb;
                          Csd = -(here->BSIM3v32cgdb + here->BSIM3v32cddb
                              + here->BSIM3v32cbdb);
                          ddxpart_dVd = (Cdd - dxpart * (Cdd + Csd)) / qcheq;
                          Cdg = here->BSIM3v32cdgb;
                          Csg = -(here->BSIM3v32cggb + here->BSIM3v32cdgb
                              + here->BSIM3v32cbgb);
                          ddxpart_dVg = (Cdg - dxpart * (Cdg + Csg)) / qcheq;

                          Cds = here->BSIM3v32cdsb;
                          Css = -(here->BSIM3v32cgsb + here->BSIM3v32cdsb
                              + here->BSIM3v32cbsb);
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
              {   Gm = -here->BSIM3v32gm;
                  Gmbs = -here->BSIM3v32gmbs;
                  FwdSum = 0.0;
                  RevSum = -(Gm + Gmbs);

                  gbbsp = -here->BSIM3v32gbds;
                  gbbdp = here->BSIM3v32gbds + here->BSIM3v32gbgs + here->BSIM3v32gbbs;

                  gbdpg = 0.0;
                  gbdpsp = 0.0;
                  gbdpb = 0.0;
                  gbdpdp = 0.0;

                  gbspg = here->BSIM3v32gbgs;
                  gbspsp = here->BSIM3v32gbds;
                  gbspb = here->BSIM3v32gbbs;
                  gbspdp = -(gbspg + gbspsp + gbspb);

                  if (here->BSIM3v32nqsMod == 0)
                  {   cggb = here->BSIM3v32cggb;
                      cgsb = here->BSIM3v32cgdb;
                      cgdb = here->BSIM3v32cgsb;

                      cbgb = here->BSIM3v32cbgb;
                      cbsb = here->BSIM3v32cbdb;
                      cbdb = here->BSIM3v32cbsb;

                      cdgb = -(here->BSIM3v32cdgb + cggb + cbgb);
                      cdsb = -(here->BSIM3v32cddb + cgsb + cbsb);
                      cddb = -(here->BSIM3v32cdsb + cgdb + cbdb);

                      xgtg = xgtd = xgts = xgtb = 0.0;
                      sxpart = 0.4;
                      dxpart = 0.6;
                      ddxpart_dVd = ddxpart_dVg = ddxpart_dVb
                                  = ddxpart_dVs = 0.0;
                      dsxpart_dVd = dsxpart_dVg = dsxpart_dVb
                                  = dsxpart_dVs = 0.0;
                  }
                  else
                  {   cggb = cgdb = cgsb = 0.0;
                      cbgb = cbdb = cbsb = 0.0;
                      cdgb = cddb = cdsb = 0.0;

                      xgtg = here->BSIM3v32gtg;
                      xgtd = here->BSIM3v32gts;
                      xgts = here->BSIM3v32gtd;
                      xgtb = here->BSIM3v32gtb;

                      xcqgb = here->BSIM3v32cqgb * omega;
                      xcqdb = here->BSIM3v32cqsb * omega;
                      xcqsb = here->BSIM3v32cqdb * omega;
                      xcqbb = here->BSIM3v32cqbb * omega;

                      CoxWL = model->BSIM3v32cox * here->pParam->BSIM3v32weffCV
                            * here->pParam->BSIM3v32leffCV;
                      qcheq = -(here->BSIM3v32qgate + here->BSIM3v32qbulk);
                      if (fabs(qcheq) <= 1.0e-5 * CoxWL)
                      {   if (model->BSIM3v32xpart < 0.5)
                          {   sxpart = 0.4;
                          }
                          else if (model->BSIM3v32xpart > 0.5)
                          {   sxpart = 0.0;
                          }
                          else
                          {   sxpart = 0.5;
                          }
                          dsxpart_dVd = dsxpart_dVg = dsxpart_dVb
                                      = dsxpart_dVs = 0.0;
                      }
                      else
                      {   sxpart = here->BSIM3v32qdrn / qcheq;
                          Css = here->BSIM3v32cddb;
                          Cds = -(here->BSIM3v32cgdb + here->BSIM3v32cddb
                              + here->BSIM3v32cbdb);
                          dsxpart_dVs = (Css - sxpart * (Css + Cds)) / qcheq;
                          Csg = here->BSIM3v32cdgb;
                          Cdg = -(here->BSIM3v32cggb + here->BSIM3v32cdgb
                              + here->BSIM3v32cbgb);
                          dsxpart_dVg = (Csg - sxpart * (Csg + Cdg)) / qcheq;

                          Csd = here->BSIM3v32cdsb;
                          Cdd = -(here->BSIM3v32cgsb + here->BSIM3v32cdsb
                              + here->BSIM3v32cbsb);
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

              T1 = *(ckt->CKTstate0 + here->BSIM3v32qdef) * here->BSIM3v32gtau;
              gdpr = here->BSIM3v32drainConductance;
              gspr = here->BSIM3v32sourceConductance;
              gds = here->BSIM3v32gds;
              gbd = here->BSIM3v32gbd;
              gbs = here->BSIM3v32gbs;
              capbd = here->BSIM3v32capbd;
              capbs = here->BSIM3v32capbs;

              GSoverlapCap = here->BSIM3v32cgso;
              GDoverlapCap = here->BSIM3v32cgdo;
              GBoverlapCap = here->pParam->BSIM3v32cgbo;

              xcdgb = (cdgb - GDoverlapCap) * omega;
              xcddb = (cddb + capbd + GDoverlapCap) * omega;
              xcdsb = cdsb * omega;
              xcsgb = -(cggb + cbgb + cdgb + GSoverlapCap) * omega;
              xcsdb = -(cgdb + cbdb + cddb) * omega;
              xcssb = (capbs + GSoverlapCap - (cgsb + cbsb + cdsb)) * omega;
              xcggb = (cggb + GDoverlapCap + GSoverlapCap + GBoverlapCap)
                    * omega;
              xcgdb = (cgdb - GDoverlapCap ) * omega;
              xcgsb = (cgsb - GSoverlapCap) * omega;
              xcbgb = (cbgb - GBoverlapCap) * omega;
              xcbdb = (cbdb - capbd ) * omega;
              xcbsb = (cbsb - capbs ) * omega;

              m = here->BSIM3v32m;

              *(here->BSIM3v32GgPtr + 1) += m * xcggb;
              *(here->BSIM3v32BbPtr + 1) -=
                      m * (xcbgb + xcbdb + xcbsb);
              *(here->BSIM3v32DPdpPtr + 1) += m * xcddb;
              *(here->BSIM3v32SPspPtr + 1) += m * xcssb;
              *(here->BSIM3v32GbPtr + 1) -=
                      m * (xcggb + xcgdb + xcgsb);
              *(here->BSIM3v32GdpPtr + 1) += m * xcgdb;
              *(here->BSIM3v32GspPtr + 1) += m * xcgsb;
              *(here->BSIM3v32BgPtr + 1) += m * xcbgb;
              *(here->BSIM3v32BdpPtr + 1) += m * xcbdb;
              *(here->BSIM3v32BspPtr + 1) += m * xcbsb;
              *(here->BSIM3v32DPgPtr + 1) += m * xcdgb;
              *(here->BSIM3v32DPbPtr + 1) -=
                      m * (xcdgb + xcddb + xcdsb);
              *(here->BSIM3v32DPspPtr + 1) += m * xcdsb;
              *(here->BSIM3v32SPgPtr + 1) += m * xcsgb;
              *(here->BSIM3v32SPbPtr + 1) -=
                      m * (xcsgb + xcsdb + xcssb);
              *(here->BSIM3v32SPdpPtr + 1) += m * xcsdb;

              *(here->BSIM3v32DdPtr) += m * gdpr;
              *(here->BSIM3v32SsPtr) += m * gspr;
              *(here->BSIM3v32BbPtr) +=
                      m * (gbd + gbs - here->BSIM3v32gbbs);
              *(here->BSIM3v32DPdpPtr) +=
                      m * (gdpr + gds + gbd + RevSum +
                           dxpart * xgtd + T1 * ddxpart_dVd +
                           gbdpdp);
              *(here->BSIM3v32SPspPtr) +=
                      m * (gspr + gds + gbs + FwdSum +
                           sxpart * xgts + T1 * dsxpart_dVs +
                           gbspsp);

              *(here->BSIM3v32DdpPtr) -= m * gdpr;
              *(here->BSIM3v32SspPtr) -= m * gspr;

              *(here->BSIM3v32BgPtr) -= m * here->BSIM3v32gbgs;
              *(here->BSIM3v32BdpPtr) -= m * (gbd - gbbdp);
              *(here->BSIM3v32BspPtr) -= m * (gbs - gbbsp);

              *(here->BSIM3v32DPdPtr) -= m * gdpr;
              *(here->BSIM3v32DPgPtr) +=
                      m * (Gm + dxpart * xgtg + T1 * ddxpart_dVg +
                           gbdpg);
              *(here->BSIM3v32DPbPtr) -=
                      m * (gbd - Gmbs - dxpart * xgtb -
                           T1 * ddxpart_dVb - gbdpb);
              *(here->BSIM3v32DPspPtr) -=
                      m * (gds + FwdSum - dxpart * xgts -
                           T1 * ddxpart_dVs - gbdpsp);

              *(here->BSIM3v32SPgPtr) -=
                      m * (Gm - sxpart * xgtg - T1 * dsxpart_dVg -
                           gbspg);
              *(here->BSIM3v32SPsPtr) -= m * gspr;
              *(here->BSIM3v32SPbPtr) -=
                      m * (gbs + Gmbs - sxpart * xgtb -
                           T1 * dsxpart_dVb - gbspb);
              *(here->BSIM3v32SPdpPtr) -=
                      m * (gds + RevSum - sxpart * xgtd -
                           T1 * dsxpart_dVd - gbspdp);

              *(here->BSIM3v32GgPtr) -= m * xgtg;
              *(here->BSIM3v32GbPtr) -= m * xgtb;
              *(here->BSIM3v32GdpPtr) -= m * xgtd;
              *(here->BSIM3v32GspPtr) -= m * xgts;

              if (here->BSIM3v32nqsMod)
              {
                      *(here->BSIM3v32QqPtr + 1) +=
                              m * omega * ScalingFactor;
                      *(here->BSIM3v32QgPtr + 1) -= m * xcqgb;
                      *(here->BSIM3v32QdpPtr + 1) -= m * xcqdb;
                      *(here->BSIM3v32QspPtr + 1) -= m * xcqsb;
                      *(here->BSIM3v32QbPtr + 1) -= m * xcqbb;

                      *(here->BSIM3v32QqPtr) += m * here->BSIM3v32gtau;

                      *(here->BSIM3v32DPqPtr) +=
                              m * (dxpart * here->BSIM3v32gtau);
                      *(here->BSIM3v32SPqPtr) +=
                              m * (sxpart * here->BSIM3v32gtau);
                      *(here->BSIM3v32GqPtr) -= m * here->BSIM3v32gtau;

                      *(here->BSIM3v32QgPtr) += m * xgtg;
                      *(here->BSIM3v32QdpPtr) += m * xgtd;
                      *(here->BSIM3v32QspPtr) += m * xgts;
                      *(here->BSIM3v32QbPtr) += m * xgtb;
              }
         }
    }
    return(OK);
}

