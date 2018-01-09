/**** BSIM4.6.2 Released by Wenwei Yang 07/31/2008 ****/

/**********
 * Copyright 2006 Regents of the University of California. All rights reserved.
 * File: b4acld.c of BSIM4.6.2.
 * Author: 2000 Weidong Liu
 * Authors: 2001- Xuemei Xi, Mohan Dunga, Ali Niknejad, Chenming Hu.
 * Authors: 2006- Mohan Dunga, Ali Niknejad, Chenming Hu
 * Authors: 2007- Mohan Dunga, Wenwei Yang, Ali Niknejad, Chenming Hu
 * Project Director: Prof. Chenming Hu.
 * Modified by Xuemei Xi, 10/05/2001.
 **********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "bsim4v6def.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
BSIM4v6acLoad(
GENmodel *inModel,
CKTcircuit *ckt)
{
BSIM4v6model *model = (BSIM4v6model*)inModel;
BSIM4v6instance *here;

double gjbd, gjbs, geltd, gcrg, gcrgg, gcrgd, gcrgs, gcrgb;
double xcbgb, xcbdb, xcbsb, xcbbb;
double xcggbr, xcgdbr, xcgsbr, xcgbbr, xcggbi, xcgdbi, xcgsbi, xcgbbi;
double Cggr, Cgdr, Cgsr, Cgbr, Cggi, Cgdi, Cgsi, Cgbi;
double xcddbr, xcdgbr, xcdsbr, xcdbbr, xcsdbr, xcsgbr, xcssbr, xcsbbr;
double xcddbi, xcdgbi, xcdsbi, xcdbbi, xcsdbi, xcsgbi, xcssbi, xcsbbi;
double xcdbdb, xcsbsb=0.0, xcgmgmb=0.0, xcgmdb=0.0, xcgmsb=0.0, xcdgmb, xcsgmb;
double xcgmbb=0.0, xcbgmb;
double capbd, capbs, omega;
double gstot, gstotd, gstotg, gstots, gstotb, gspr;
double gdtot, gdtotd, gdtotg, gdtots, gdtotb, gdpr;
double gIstotg, gIstotd, gIstots, gIstotb;
double gIdtotg, gIdtotd, gIdtots, gIdtotb;
double gIbtotg, gIbtotd, gIbtots, gIbtotb;
double gIgtotg, gIgtotd, gIgtots, gIgtotb;
double cgso, cgdo, cgbo;
double gbspsp, gbbdp, gbbsp, gbspg, gbspb;
double gbspdp, gbdpdp, gbdpg, gbdpb, gbdpsp;
double T0=0.0, T1, T2, T3;
double Csg, Csd, Css;
double Cdgr, Cddr, Cdsr, Cdbr, Csgr, Csdr, Cssr, Csbr;
double Cdgi, Cddi, Cdsi, Cdbi, Csgi, Csdi, Cssi, Csbi;
double gmr, gmi, gmbsr, gmbsi, gdsr, gdsi;
double FwdSumr, RevSumr, Gmr, Gmbsr;
double FwdSumi, RevSumi, Gmi, Gmbsi;
struct bsim4v6SizeDependParam *pParam;
double ggidld, ggidlg, ggidlb, ggislg, ggislb, ggisls;

double m;

    omega = ckt->CKTomega;
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

              Csd = -(here->BSIM4v6cddb + here->BSIM4v6cgdb + here->BSIM4v6cbdb);
              Csg = -(here->BSIM4v6cdgb + here->BSIM4v6cggb + here->BSIM4v6cbgb);
              Css = -(here->BSIM4v6cdsb + here->BSIM4v6cgsb + here->BSIM4v6cbsb);

              if (here->BSIM4v6acnqsMod)
              {   T0 = omega * here->BSIM4v6taunet;
                  T1 = T0 * T0;
                  T2 = 1.0 / (1.0 + T1);
                  T3 = T0 * T2;

                  gmr = here->BSIM4v6gm * T2;
                  gmbsr = here->BSIM4v6gmbs * T2;
                  gdsr = here->BSIM4v6gds * T2;

                  gmi = -here->BSIM4v6gm * T3;
                  gmbsi = -here->BSIM4v6gmbs * T3;
                  gdsi = -here->BSIM4v6gds * T3;

                  Cddr = here->BSIM4v6cddb * T2;
                  Cdgr = here->BSIM4v6cdgb * T2;
                  Cdsr = here->BSIM4v6cdsb * T2;
                  Cdbr = -(Cddr + Cdgr + Cdsr);

		  /* WDLiu: Cxyi mulitplied by jomega below, and actually to be of conductance */
                  Cddi = here->BSIM4v6cddb * T3 * omega;
                  Cdgi = here->BSIM4v6cdgb * T3 * omega;
                  Cdsi = here->BSIM4v6cdsb * T3 * omega;
                  Cdbi = -(Cddi + Cdgi + Cdsi);

                  Csdr = Csd * T2;
                  Csgr = Csg * T2;
                  Cssr = Css * T2;
                  Csbr = -(Csdr + Csgr + Cssr);

                  Csdi = Csd * T3 * omega;
                  Csgi = Csg * T3 * omega;
                  Cssi = Css * T3 * omega;
                  Csbi = -(Csdi + Csgi + Cssi);

                  Cgdr = -(Cddr + Csdr + here->BSIM4v6cbdb);
                  Cggr = -(Cdgr + Csgr + here->BSIM4v6cbgb);
                  Cgsr = -(Cdsr + Cssr + here->BSIM4v6cbsb);
                  Cgbr = -(Cgdr + Cggr + Cgsr);

                  Cgdi = -(Cddi + Csdi);
                  Cggi = -(Cdgi + Csgi);
                  Cgsi = -(Cdsi + Cssi);
                  Cgbi = -(Cgdi + Cggi + Cgsi);
              }
              else /* QS */
              {   gmr = here->BSIM4v6gm;
                  gmbsr = here->BSIM4v6gmbs;
                  gdsr = here->BSIM4v6gds;
                  gmi = gmbsi = gdsi = 0.0;

                  Cddr = here->BSIM4v6cddb;
                  Cdgr = here->BSIM4v6cdgb;
                  Cdsr = here->BSIM4v6cdsb;
                  Cdbr = -(Cddr + Cdgr + Cdsr);
                  Cddi = Cdgi = Cdsi = Cdbi = 0.0;

                  Csdr = Csd;
                  Csgr = Csg;
                  Cssr = Css;
                  Csbr = -(Csdr + Csgr + Cssr);
                  Csdi = Csgi = Cssi = Csbi = 0.0;

                  Cgdr = here->BSIM4v6cgdb;
                  Cggr = here->BSIM4v6cggb;
                  Cgsr = here->BSIM4v6cgsb;
                  Cgbr = -(Cgdr + Cggr + Cgsr);
                  Cgdi = Cggi = Cgsi = Cgbi = 0.0;
              }


              if (here->BSIM4v6mode >= 0) 
              {   Gmr = gmr;
                  Gmbsr = gmbsr;
                  FwdSumr = Gmr + Gmbsr;
                  RevSumr = 0.0;
                  Gmi = gmi;
                  Gmbsi = gmbsi;
                  FwdSumi = Gmi + Gmbsi;
                  RevSumi = 0.0;

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

                  if (here->BSIM4v6rgateMod == 3)
                  {   xcgmgmb = (cgdo + cgso + pParam->BSIM4v6cgbo) * omega;
                      xcgmdb = -cgdo * omega;
                      xcgmsb = -cgso * omega;
                      xcgmbb = -pParam->BSIM4v6cgbo * omega;
    
                      xcdgmb = xcgmdb;
                      xcsgmb = xcgmsb;
                      xcbgmb = xcgmbb;
    
                      xcggbr = Cggr * omega;
                      xcgdbr = Cgdr * omega;
                      xcgsbr = Cgsr * omega;
                      xcgbbr = -(xcggbr + xcgdbr + xcgsbr);
    
                      xcdgbr = Cdgr * omega;
                      xcsgbr = Csgr * omega;
                      xcbgb = here->BSIM4v6cbgb * omega;
                  }
                  else
                  {   xcggbr = (Cggr + cgdo + cgso + pParam->BSIM4v6cgbo ) * omega;
                      xcgdbr = (Cgdr - cgdo) * omega;
                      xcgsbr = (Cgsr - cgso) * omega;
                      xcgbbr = -(xcggbr + xcgdbr + xcgsbr);
    
                      xcdgbr = (Cdgr - cgdo) * omega;
                      xcsgbr = (Csgr - cgso) * omega;
                      xcbgb = (here->BSIM4v6cbgb - pParam->BSIM4v6cgbo) * omega;
    
                      xcdgmb = xcsgmb = xcbgmb = 0.0;
                  }
                  xcddbr = (Cddr + here->BSIM4v6capbd + cgdo) * omega;
                  xcdsbr = Cdsr * omega;
                  xcsdbr = Csdr * omega;
                  xcssbr = (here->BSIM4v6capbs + cgso + Cssr) * omega;
    
                  if (!here->BSIM4v6rbodyMod)
                  {   xcdbbr = -(xcdgbr + xcddbr + xcdsbr + xcdgmb);
                      xcsbbr = -(xcsgbr + xcsdbr + xcssbr + xcsgmb);

                      xcbdb = (here->BSIM4v6cbdb - here->BSIM4v6capbd) * omega;
                      xcbsb = (here->BSIM4v6cbsb - here->BSIM4v6capbs) * omega;
                      xcdbdb = 0.0;
                  }
                  else
                  {   xcdbbr = Cdbr * omega;
                      xcsbbr = -(xcsgbr + xcsdbr + xcssbr + xcsgmb)
                             + here->BSIM4v6capbs * omega;

                      xcbdb = here->BSIM4v6cbdb * omega;
                      xcbsb = here->BSIM4v6cbsb * omega;
    
                      xcdbdb = -here->BSIM4v6capbd * omega;
                      xcsbsb = -here->BSIM4v6capbs * omega;
                  }
                  xcbbb = -(xcbdb + xcbgb + xcbsb + xcbgmb);

                  xcdgbi = Cdgi;
                  xcsgbi = Csgi;
                  xcddbi = Cddi;
                  xcdsbi = Cdsi;
                  xcsdbi = Csdi;
                  xcssbi = Cssi;
                  xcdbbi = Cdbi;
                  xcsbbi = Csbi;
                  xcggbi = Cggi;
                  xcgdbi = Cgdi;
                  xcgsbi = Cgsi;
                  xcgbbi = Cgbi;
              } 
              else /* Reverse mode */
              {   Gmr = -gmr;
                  Gmbsr = -gmbsr;
                  FwdSumr = 0.0;
                  RevSumr = -(Gmr + Gmbsr);
                  Gmi = -gmi;
                  Gmbsi = -gmbsi;
                  FwdSumi = 0.0;
                  RevSumi = -(Gmi + Gmbsi);

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

                  if (here->BSIM4v6rgateMod == 3)
                  {   xcgmgmb = (cgdo + cgso + pParam->BSIM4v6cgbo) * omega;
                      xcgmdb = -cgdo * omega;
                      xcgmsb = -cgso * omega;
                      xcgmbb = -pParam->BSIM4v6cgbo * omega;
    
                      xcdgmb = xcgmdb;
                      xcsgmb = xcgmsb;
                      xcbgmb = xcgmbb;
   
                      xcggbr = Cggr * omega;
                      xcgdbr = Cgsr * omega;
                      xcgsbr = Cgdr * omega;
                      xcgbbr = -(xcggbr + xcgdbr + xcgsbr);
    
                      xcdgbr = Csgr * omega;
                      xcsgbr = Cdgr * omega;
                      xcbgb = here->BSIM4v6cbgb * omega;
                  }
                  else
                  {   xcggbr = (Cggr + cgdo + cgso + pParam->BSIM4v6cgbo ) * omega;
                      xcgdbr = (Cgsr - cgdo) * omega;
                      xcgsbr = (Cgdr - cgso) * omega;
                      xcgbbr = -(xcggbr + xcgdbr + xcgsbr);
    
                      xcdgbr = (Csgr - cgdo) * omega;
                      xcsgbr = (Cdgr - cgso) * omega;
                      xcbgb = (here->BSIM4v6cbgb - pParam->BSIM4v6cgbo) * omega;
    
                      xcdgmb = xcsgmb = xcbgmb = 0.0;
                  }
                  xcddbr = (here->BSIM4v6capbd + cgdo + Cssr) * omega;
                  xcdsbr = Csdr * omega;
                  xcsdbr = Cdsr * omega;
                  xcssbr = (Cddr + here->BSIM4v6capbs + cgso) * omega;
    
                  if (!here->BSIM4v6rbodyMod)
                  {   xcdbbr = -(xcdgbr + xcddbr + xcdsbr + xcdgmb);
                      xcsbbr = -(xcsgbr + xcsdbr + xcssbr + xcsgmb);

                      xcbdb = (here->BSIM4v6cbsb - here->BSIM4v6capbd) * omega;
                      xcbsb = (here->BSIM4v6cbdb - here->BSIM4v6capbs) * omega;
                      xcdbdb = 0.0;
                  }
                  else
                  {   xcdbbr = -(xcdgbr + xcddbr + xcdsbr + xcdgmb)
                             + here->BSIM4v6capbd * omega;
                      xcsbbr = Cdbr * omega;

                      xcbdb = here->BSIM4v6cbsb * omega;
                      xcbsb = here->BSIM4v6cbdb * omega;
                      xcdbdb = -here->BSIM4v6capbd * omega;
                      xcsbsb = -here->BSIM4v6capbs * omega;
                  }
                  xcbbb = -(xcbgb + xcbdb + xcbsb + xcbgmb);

                  xcdgbi = Csgi;
                  xcsgbi = Cdgi;
                  xcddbi = Cssi;
                  xcdsbi = Csdi;
                  xcsdbi = Cdsi;
                  xcssbi = Cddi;
                  xcdbbi = Csbi;
                  xcsbbi = Cdbi;
                  xcggbi = Cggi;
                  xcgdbi = Cgsi;
                  xcgsbi = Cgdi;
                  xcgbbi = Cgbi;
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


              /*
               * Loading AC matrix
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

                  *(here->BSIM4v6GPgpPtr +1) += m * xcggbr;
                  *(here->BSIM4v6GPgpPtr) += m * (geltd + xcggbi + gIgtotg);
                  *(here->BSIM4v6GPdpPtr +1) += m * xcgdbr;
                  *(here->BSIM4v6GPdpPtr) += m * (xcgdbi + gIgtotd);
                  *(here->BSIM4v6GPspPtr +1) += m * xcgsbr;
                  *(here->BSIM4v6GPspPtr) += m * (xcgsbi + gIgtots);
                  *(here->BSIM4v6GPbpPtr +1) += m * xcgbbr;
                  *(here->BSIM4v6GPbpPtr) += m * (xcgbbi + gIgtotb);
              } /* WDLiu: gcrg already subtracted from all gcrgg below */
              else if (here->BSIM4v6rgateMod == 2)
              {   *(here->BSIM4v6GEgePtr) += m * gcrg;
                  *(here->BSIM4v6GEgpPtr) += m * gcrgg;
                  *(here->BSIM4v6GEdpPtr) += m * gcrgd;
                  *(here->BSIM4v6GEspPtr) += m * gcrgs;
                  *(here->BSIM4v6GEbpPtr) += m * gcrgb;

                  *(here->BSIM4v6GPgePtr) -= m * gcrg;
                  *(here->BSIM4v6GPgpPtr +1) += m * xcggbr;
                  *(here->BSIM4v6GPgpPtr) -= m * (gcrgg - xcggbi - gIgtotg);
                  *(here->BSIM4v6GPdpPtr +1) += m * xcgdbr;
                  *(here->BSIM4v6GPdpPtr) -= m * (gcrgd - xcgdbi - gIgtotd);
                  *(here->BSIM4v6GPspPtr +1) += m * xcgsbr;
                  *(here->BSIM4v6GPspPtr) -= m * (gcrgs - xcgsbi - gIgtots);
                  *(here->BSIM4v6GPbpPtr +1) += m * xcgbbr;
                  *(here->BSIM4v6GPbpPtr) -= m * (gcrgb - xcgbbi - gIgtotb);
              }
              else if (here->BSIM4v6rgateMod == 3)
              {   *(here->BSIM4v6GEgePtr) += m * geltd;
                  *(here->BSIM4v6GEgmPtr) -= m * geltd;
                  *(here->BSIM4v6GMgePtr) -= m * geltd;
                  *(here->BSIM4v6GMgmPtr) += m * (geltd + gcrg);
                  *(here->BSIM4v6GMgmPtr +1) += m * xcgmgmb;
   
                  *(here->BSIM4v6GMdpPtr) += m * gcrgd;
                  *(here->BSIM4v6GMdpPtr +1) += m * xcgmdb;
                  *(here->BSIM4v6GMgpPtr) += m * gcrgg;
                  *(here->BSIM4v6GMspPtr) += m * gcrgs;
                  *(here->BSIM4v6GMspPtr +1) += m * xcgmsb;
                  *(here->BSIM4v6GMbpPtr) += m * gcrgb;
                  *(here->BSIM4v6GMbpPtr +1) += m * xcgmbb;
   
                  *(here->BSIM4v6DPgmPtr +1) += m * xcdgmb;
                  *(here->BSIM4v6GPgmPtr) -= m * gcrg;
                  *(here->BSIM4v6SPgmPtr +1) += m * xcsgmb;
                  *(here->BSIM4v6BPgmPtr +1) += m * xcbgmb;
   
                  *(here->BSIM4v6GPgpPtr) -= m * (gcrgg - xcggbi - gIgtotg);
                  *(here->BSIM4v6GPgpPtr +1) += m * xcggbr;
                  *(here->BSIM4v6GPdpPtr) -= m * (gcrgd - xcgdbi - gIgtotd);
                  *(here->BSIM4v6GPdpPtr +1) += m * xcgdbr;
                  *(here->BSIM4v6GPspPtr) -= m * (gcrgs - xcgsbi - gIgtots);
                  *(here->BSIM4v6GPspPtr +1) += m * xcgsbr;
                  *(here->BSIM4v6GPbpPtr) -= m * (gcrgb - xcgbbi - gIgtotb);
                  *(here->BSIM4v6GPbpPtr +1) += m * xcgbbr;
              }
              else
              {   *(here->BSIM4v6GPgpPtr +1) += m * xcggbr;
                  *(here->BSIM4v6GPgpPtr) += m * (xcggbi + gIgtotg);
                  *(here->BSIM4v6GPdpPtr +1) += m * xcgdbr;
                  *(here->BSIM4v6GPdpPtr) += m * (xcgdbi + gIgtotd);
                  *(here->BSIM4v6GPspPtr +1) += m * xcgsbr;
                  *(here->BSIM4v6GPspPtr) += m * (xcgsbi + gIgtots);
                  *(here->BSIM4v6GPbpPtr +1) += m * xcgbbr;
                  *(here->BSIM4v6GPbpPtr) += m * (xcgbbi + gIgtotb);
              }

              if (model->BSIM4v6rdsMod)
              {   (*(here->BSIM4v6DgpPtr) += m * gdtotg);
                  (*(here->BSIM4v6DspPtr) += m * gdtots);
                  (*(here->BSIM4v6DbpPtr) += m * gdtotb);
                  (*(here->BSIM4v6SdpPtr) += m * gstotd);
                  (*(here->BSIM4v6SgpPtr) += m * gstotg);
                  (*(here->BSIM4v6SbpPtr) += m * gstotb);
              }

              *(here->BSIM4v6DPdpPtr +1) += m * (xcddbr + gdsi + RevSumi);
              *(here->BSIM4v6DPdpPtr) += m * (gdpr + xcddbi + gdsr + here->BSIM4v6gbd 
                                     - gdtotd + RevSumr + gbdpdp - gIdtotd);
              *(here->BSIM4v6DPdPtr) -= m * (gdpr + gdtot);
              *(here->BSIM4v6DPgpPtr +1) += m * (xcdgbr + Gmi);
              *(here->BSIM4v6DPgpPtr) += m * (Gmr + xcdgbi - gdtotg + gbdpg - gIdtotg);
              *(here->BSIM4v6DPspPtr +1) += m * (xcdsbr - gdsi - FwdSumi);
              *(here->BSIM4v6DPspPtr) -= m * (gdsr - xcdsbi + FwdSumr + gdtots - gbdpsp + gIdtots);
              *(here->BSIM4v6DPbpPtr +1) += m * (xcdbbr + Gmbsi);
              *(here->BSIM4v6DPbpPtr) -= m * (gjbd + gdtotb - xcdbbi - Gmbsr - gbdpb + gIdtotb);

              *(here->BSIM4v6DdpPtr) -= m * (gdpr - gdtotd);
              *(here->BSIM4v6DdPtr) += m * (gdpr + gdtot);

              *(here->BSIM4v6SPdpPtr +1) += m * (xcsdbr - gdsi - RevSumi);
              *(here->BSIM4v6SPdpPtr) -= m * (gdsr - xcsdbi + gstotd + RevSumr - gbspdp + gIstotd);
              *(here->BSIM4v6SPgpPtr +1) += m * (xcsgbr - Gmi);
              *(here->BSIM4v6SPgpPtr) -= m * (Gmr - xcsgbi + gstotg - gbspg + gIstotg);
              *(here->BSIM4v6SPspPtr +1) += m * (xcssbr + gdsi + FwdSumi);
              *(here->BSIM4v6SPspPtr) += m * (gspr + xcssbi + gdsr + here->BSIM4v6gbs
                                     - gstots + FwdSumr + gbspsp - gIstots);
              *(here->BSIM4v6SPsPtr) -= m * (gspr + gstot);
              *(here->BSIM4v6SPbpPtr +1) += m * (xcsbbr - Gmbsi);
              *(here->BSIM4v6SPbpPtr) -= m * (gjbs + gstotb - xcsbbi + Gmbsr - gbspb + gIstotb);

              *(here->BSIM4v6SspPtr) -= m * (gspr - gstots);
              *(here->BSIM4v6SsPtr) += m * (gspr + gstot);

              *(here->BSIM4v6BPdpPtr +1) += m * xcbdb;
              *(here->BSIM4v6BPdpPtr) -= m * (gjbd - gbbdp + gIbtotd);
              *(here->BSIM4v6BPgpPtr +1) += m * xcbgb;
              *(here->BSIM4v6BPgpPtr) -= m * (here->BSIM4v6gbgs + gIbtotg);
              *(here->BSIM4v6BPspPtr +1) += m * xcbsb;
              *(here->BSIM4v6BPspPtr) -= m * (gjbs - gbbsp + gIbtots);
              *(here->BSIM4v6BPbpPtr +1) += m * xcbbb;
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
              {   (*(here->BSIM4v6DPdbPtr +1) += m * xcdbdb);
                  (*(here->BSIM4v6DPdbPtr) -= m * here->BSIM4v6gbd);
                  (*(here->BSIM4v6SPsbPtr +1) += m * xcsbsb);
                  (*(here->BSIM4v6SPsbPtr) -= m * here->BSIM4v6gbs);

                  (*(here->BSIM4v6DBdpPtr +1) += m * xcdbdb);
                  (*(here->BSIM4v6DBdpPtr) -= m * here->BSIM4v6gbd);
                  (*(here->BSIM4v6DBdbPtr +1) -= m * xcdbdb);
                  (*(here->BSIM4v6DBdbPtr) += m * (here->BSIM4v6gbd + here->BSIM4v6grbpd 
                                          + here->BSIM4v6grbdb));
                  (*(here->BSIM4v6DBbpPtr) -= m * here->BSIM4v6grbpd);
                  (*(here->BSIM4v6DBbPtr) -= m * here->BSIM4v6grbdb);

                  (*(here->BSIM4v6BPdbPtr) -= m * here->BSIM4v6grbpd);
                  (*(here->BSIM4v6BPbPtr) -= m * here->BSIM4v6grbpb);
                  (*(here->BSIM4v6BPsbPtr) -= m * here->BSIM4v6grbps);
                  (*(here->BSIM4v6BPbpPtr) += m * (here->BSIM4v6grbpd + here->BSIM4v6grbps 
                                          + here->BSIM4v6grbpb));
                  /* WDLiu: (-here->BSIM4v6gbbs) already added to BPbpPtr */

                  (*(here->BSIM4v6SBspPtr +1) += m * xcsbsb);
                  (*(here->BSIM4v6SBspPtr) -= m * here->BSIM4v6gbs);
                  (*(here->BSIM4v6SBbpPtr) -= m * here->BSIM4v6grbps);
                  (*(here->BSIM4v6SBbPtr) -= m * here->BSIM4v6grbsb);
                  (*(here->BSIM4v6SBsbPtr +1) -= m * xcsbsb);
                  (*(here->BSIM4v6SBsbPtr) += m * (here->BSIM4v6gbs
                                          + here->BSIM4v6grbps + here->BSIM4v6grbsb));

                  (*(here->BSIM4v6BdbPtr) -= m * here->BSIM4v6grbdb);
                  (*(here->BSIM4v6BbpPtr) -= m * here->BSIM4v6grbpb);
                  (*(here->BSIM4v6BsbPtr) -= m * here->BSIM4v6grbsb);
                  (*(here->BSIM4v6BbPtr) += m * (here->BSIM4v6grbsb + here->BSIM4v6grbdb
                                        + here->BSIM4v6grbpb));
              }


           /*
            * WDLiu: The internal charge node generated for transient NQS is not needed for
            *        AC NQS. The following is not doing a real job, but we have to keep it;
            *        otherwise a singular AC NQS matrix may occur if the transient NQS is on.
            *        The charge node is isolated from the instance.
            */
           if (here->BSIM4v6trnqsMod)
           {   (*(here->BSIM4v6QqPtr) += m * 1.0);
               (*(here->BSIM4v6QgpPtr) += 0.0);
               (*(here->BSIM4v6QdpPtr) += 0.0);
               (*(here->BSIM4v6QspPtr) += 0.0);
               (*(here->BSIM4v6QbpPtr) += 0.0);

               (*(here->BSIM4v6DPqPtr) += 0.0);
               (*(here->BSIM4v6SPqPtr) += 0.0);
               (*(here->BSIM4v6GPqPtr) += 0.0);
           }
         }
    }
    return(OK);
}
