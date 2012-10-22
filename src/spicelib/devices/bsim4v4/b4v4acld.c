/**** BSIM4.4.0  Released by Xuemei (Jane) Xi 03/04/2004 ****/

/**********
 * Copyright 2004 Regents of the University of California. All rights reserved.
 * File: b4acld.c of BSIM4.4.0.
 * Author: 2000 Weidong Liu
 * Authors: 2001- Xuemei Xi, Jin He, Kanyu Cao, Mohan Dunga, Mansun Chan, Ali Niknejad, Chenming Hu.
 * Project Director: Prof. Chenming Hu.
 * Modified by Xuemei Xi, 10/05/2001.
 **********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "bsim4v4def.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
BSIM4v4acLoad(
GENmodel *inModel,
CKTcircuit *ckt)
{
BSIM4v4model *model = (BSIM4v4model*)inModel;
BSIM4v4instance *here;

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
struct bsim4SizeDependParam *pParam;
double ggidld, ggidlg, ggidlb, ggislg, ggislb, ggisls;

double m;

    omega = ckt->CKTomega;
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

              Csd = -(here->BSIM4v4cddb + here->BSIM4v4cgdb + here->BSIM4v4cbdb);
              Csg = -(here->BSIM4v4cdgb + here->BSIM4v4cggb + here->BSIM4v4cbgb);
              Css = -(here->BSIM4v4cdsb + here->BSIM4v4cgsb + here->BSIM4v4cbsb);

              if (here->BSIM4v4acnqsMod)
              {   T0 = omega * here->BSIM4v4taunet;
                  T1 = T0 * T0;
                  T2 = 1.0 / (1.0 + T1);
                  T3 = T0 * T2;

                  gmr = here->BSIM4v4gm * T2;
                  gmbsr = here->BSIM4v4gmbs * T2;
                  gdsr = here->BSIM4v4gds * T2;

                  gmi = -here->BSIM4v4gm * T3;
                  gmbsi = -here->BSIM4v4gmbs * T3;
                  gdsi = -here->BSIM4v4gds * T3;

                  Cddr = here->BSIM4v4cddb * T2;
                  Cdgr = here->BSIM4v4cdgb * T2;
                  Cdsr = here->BSIM4v4cdsb * T2;
                  Cdbr = -(Cddr + Cdgr + Cdsr);

		  /* WDLiu: Cxyi mulitplied by jomega below, and actually to be of conductance */
                  Cddi = here->BSIM4v4cddb * T3 * omega;
                  Cdgi = here->BSIM4v4cdgb * T3 * omega;
                  Cdsi = here->BSIM4v4cdsb * T3 * omega;
                  Cdbi = -(Cddi + Cdgi + Cdsi);

                  Csdr = Csd * T2;
                  Csgr = Csg * T2;
                  Cssr = Css * T2;
                  Csbr = -(Csdr + Csgr + Cssr);

                  Csdi = Csd * T3 * omega;
                  Csgi = Csg * T3 * omega;
                  Cssi = Css * T3 * omega;
                  Csbi = -(Csdi + Csgi + Cssi);

		  Cgdr = -(Cddr + Csdr + here->BSIM4v4cbdb);
		  Cggr = -(Cdgr + Csgr + here->BSIM4v4cbgb);
		  Cgsr = -(Cdsr + Cssr + here->BSIM4v4cbsb);
		  Cgbr = -(Cgdr + Cggr + Cgsr);

		  Cgdi = -(Cddi + Csdi);
		  Cggi = -(Cdgi + Csgi);
		  Cgsi = -(Cdsi + Cssi);
		  Cgbi = -(Cgdi + Cggi + Cgsi);
              }
              else /* QS */
              {   gmr = here->BSIM4v4gm;
                  gmbsr = here->BSIM4v4gmbs;
                  gdsr = here->BSIM4v4gds;
                  gmi = gmbsi = gdsi = 0.0;

                  Cddr = here->BSIM4v4cddb;
                  Cdgr = here->BSIM4v4cdgb;
                  Cdsr = here->BSIM4v4cdsb;
                  Cdbr = -(Cddr + Cdgr + Cdsr);
                  Cddi = Cdgi = Cdsi = Cdbi = 0.0;

                  Csdr = Csd;
                  Csgr = Csg;
                  Cssr = Css;
                  Csbr = -(Csdr + Csgr + Cssr);
                  Csdi = Csgi = Cssi = Csbi = 0.0;

                  Cgdr = here->BSIM4v4cgdb;
                  Cggr = here->BSIM4v4cggb;
                  Cgsr = here->BSIM4v4cgsb;
                  Cgbr = -(Cgdr + Cggr + Cgsr);
                  Cgdi = Cggi = Cgsi = Cgbi = 0.0;
              }


              if (here->BSIM4v4mode >= 0) 
	      {   Gmr = gmr;
                  Gmbsr = gmbsr;
                  FwdSumr = Gmr + Gmbsr;
                  RevSumr = 0.0;
		  Gmi = gmi;
                  Gmbsi = gmbsi;
                  FwdSumi = Gmi + Gmbsi;
                  RevSumi = 0.0;

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

                  if (here->BSIM4v4rgateMod == 3)
                  {   xcgmgmb = (cgdo + cgso + pParam->BSIM4v4cgbo) * omega;
                      xcgmdb = -cgdo * omega;
                      xcgmsb = -cgso * omega;
                      xcgmbb = -pParam->BSIM4v4cgbo * omega;
    
                      xcdgmb = xcgmdb;
                      xcsgmb = xcgmsb;
                      xcbgmb = xcgmbb;
    
                      xcggbr = Cggr * omega;
                      xcgdbr = Cgdr * omega;
                      xcgsbr = Cgsr * omega;
                      xcgbbr = -(xcggbr + xcgdbr + xcgsbr);
    
                      xcdgbr = Cdgr * omega;
                      xcsgbr = Csgr * omega;
                      xcbgb = here->BSIM4v4cbgb * omega;
                  }
                  else
                  {   xcggbr = (Cggr + cgdo + cgso + pParam->BSIM4v4cgbo ) * omega;
                      xcgdbr = (Cgdr - cgdo) * omega;
                      xcgsbr = (Cgsr - cgso) * omega;
                      xcgbbr = -(xcggbr + xcgdbr + xcgsbr);
    
                      xcdgbr = (Cdgr - cgdo) * omega;
                      xcsgbr = (Csgr - cgso) * omega;
                      xcbgb = (here->BSIM4v4cbgb - pParam->BSIM4v4cgbo) * omega;
    
                      xcdgmb = xcsgmb = xcbgmb = 0.0;
                  }
                  xcddbr = (Cddr + here->BSIM4v4capbd + cgdo) * omega;
                  xcdsbr = Cdsr * omega;
                  xcsdbr = Csdr * omega;
                  xcssbr = (here->BSIM4v4capbs + cgso + Cssr) * omega;
    
                  if (!here->BSIM4v4rbodyMod)
                  {   xcdbbr = -(xcdgbr + xcddbr + xcdsbr + xcdgmb);
                      xcsbbr = -(xcsgbr + xcsdbr + xcssbr + xcsgmb);

                      xcbdb = (here->BSIM4v4cbdb - here->BSIM4v4capbd) * omega;
                      xcbsb = (here->BSIM4v4cbsb - here->BSIM4v4capbs) * omega;
                      xcdbdb = 0.0;
                  }
                  else
                  {   xcdbbr = Cdbr * omega;
                      xcsbbr = -(xcsgbr + xcsdbr + xcssbr + xcsgmb)
			     + here->BSIM4v4capbs * omega;

                      xcbdb = here->BSIM4v4cbdb * omega;
                      xcbsb = here->BSIM4v4cbsb * omega;
    
                      xcdbdb = -here->BSIM4v4capbd * omega;
                      xcsbsb = -here->BSIM4v4capbs * omega;
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

                  if (here->BSIM4v4rgateMod == 3)
                  {   xcgmgmb = (cgdo + cgso + pParam->BSIM4v4cgbo) * omega;
                      xcgmdb = -cgdo * omega;
                      xcgmsb = -cgso * omega;
                      xcgmbb = -pParam->BSIM4v4cgbo * omega;
    
                      xcdgmb = xcgmdb;
                      xcsgmb = xcgmsb;
                      xcbgmb = xcgmbb;
   
                      xcggbr = Cggr * omega;
                      xcgdbr = Cgsr * omega;
                      xcgsbr = Cgdr * omega;
                      xcgbbr = -(xcggbr + xcgdbr + xcgsbr);
    
                      xcdgbr = Csgr * omega;
                      xcsgbr = Cdgr * omega;
                      xcbgb = here->BSIM4v4cbgb * omega;
                  }
                  else
                  {   xcggbr = (Cggr + cgdo + cgso + pParam->BSIM4v4cgbo ) * omega;
                      xcgdbr = (Cgsr - cgdo) * omega;
                      xcgsbr = (Cgdr - cgso) * omega;
                      xcgbbr = -(xcggbr + xcgdbr + xcgsbr);
    
                      xcdgbr = (Csgr - cgdo) * omega;
                      xcsgbr = (Cdgr - cgso) * omega;
                      xcbgb = (here->BSIM4v4cbgb - pParam->BSIM4v4cgbo) * omega;
    
                      xcdgmb = xcsgmb = xcbgmb = 0.0;
                  }
                  xcddbr = (here->BSIM4v4capbd + cgdo + Cssr) * omega;
                  xcdsbr = Csdr * omega;
                  xcsdbr = Cdsr * omega;
                  xcssbr = (Cddr + here->BSIM4v4capbs + cgso) * omega;
    
                  if (!here->BSIM4v4rbodyMod)
                  {   xcdbbr = -(xcdgbr + xcddbr + xcdsbr + xcdgmb);
                      xcsbbr = -(xcsgbr + xcsdbr + xcssbr + xcsgmb);

                      xcbdb = (here->BSIM4v4cbsb - here->BSIM4v4capbd) * omega;
                      xcbsb = (here->BSIM4v4cbdb - here->BSIM4v4capbs) * omega;
                      xcdbdb = 0.0;
                  }
                  else
                  {   xcdbbr = -(xcdgbr + xcddbr + xcdsbr + xcdgmb)
                             + here->BSIM4v4capbd * omega;
                      xcsbbr = Cdbr * omega;

                      xcbdb = here->BSIM4v4cbsb * omega;
                      xcbsb = here->BSIM4v4cbdb * omega;
                      xcdbdb = -here->BSIM4v4capbd * omega;
                      xcsbsb = -here->BSIM4v4capbs * omega;
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


              /*
               * Loading AC matrix
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

                  *(here->BSIM4v4GPgpPtr +1) += m * xcggbr;
		  *(here->BSIM4v4GPgpPtr) += m * (geltd + xcggbi + gIgtotg);
                  *(here->BSIM4v4GPdpPtr +1) += m * xcgdbr;
                  *(here->BSIM4v4GPdpPtr) += m * (xcgdbi + gIgtotd);
                  *(here->BSIM4v4GPspPtr +1) += m * xcgsbr;
                  *(here->BSIM4v4GPspPtr) += m * (xcgsbi + gIgtots);
                  *(here->BSIM4v4GPbpPtr +1) += m * xcgbbr;
                  *(here->BSIM4v4GPbpPtr) += m * (xcgbbi + gIgtotb);
              } /* WDLiu: gcrg already subtracted from all gcrgg below */
              else if (here->BSIM4v4rgateMod == 2)
              {   *(here->BSIM4v4GEgePtr) += m * gcrg;
                  *(here->BSIM4v4GEgpPtr) += m * gcrgg;
                  *(here->BSIM4v4GEdpPtr) += m * gcrgd;
                  *(here->BSIM4v4GEspPtr) += m * gcrgs;
                  *(here->BSIM4v4GEbpPtr) += m * gcrgb;

                  *(here->BSIM4v4GPgePtr) -= m * gcrg;
                  *(here->BSIM4v4GPgpPtr +1) += m * xcggbr;
		  *(here->BSIM4v4GPgpPtr) -= m * (gcrgg - xcggbi - gIgtotg);
                  *(here->BSIM4v4GPdpPtr +1) += m * xcgdbr;
		  *(here->BSIM4v4GPdpPtr) -= m * (gcrgd - xcgdbi - gIgtotd);
                  *(here->BSIM4v4GPspPtr +1) += m * xcgsbr;
		  *(here->BSIM4v4GPspPtr) -= m * (gcrgs - xcgsbi - gIgtots);
                  *(here->BSIM4v4GPbpPtr +1) += m * xcgbbr;
		  *(here->BSIM4v4GPbpPtr) -= m * (gcrgb - xcgbbi - gIgtotb);
              }
              else if (here->BSIM4v4rgateMod == 3)
              {   *(here->BSIM4v4GEgePtr) += m * geltd;
                  *(here->BSIM4v4GEgmPtr) -= m * geltd;
                  *(here->BSIM4v4GMgePtr) -= m * geltd;
                  *(here->BSIM4v4GMgmPtr) += m * (geltd + gcrg);
                  *(here->BSIM4v4GMgmPtr +1) += m * xcgmgmb;
   
                  *(here->BSIM4v4GMdpPtr) += m * gcrgd;
                  *(here->BSIM4v4GMdpPtr +1) += m * xcgmdb;
                  *(here->BSIM4v4GMgpPtr) += m * gcrgg;
                  *(here->BSIM4v4GMspPtr) += m * gcrgs;
                  *(here->BSIM4v4GMspPtr +1) += m * xcgmsb;
                  *(here->BSIM4v4GMbpPtr) += m * gcrgb;
                  *(here->BSIM4v4GMbpPtr +1) += m * xcgmbb;
   
                  *(here->BSIM4v4DPgmPtr +1) += m * xcdgmb;
                  *(here->BSIM4v4GPgmPtr) -= m * gcrg;
                  *(here->BSIM4v4SPgmPtr +1) += m * xcsgmb;
                  *(here->BSIM4v4BPgmPtr +1) += m * xcbgmb;
   
                  *(here->BSIM4v4GPgpPtr) -= m * (gcrgg - xcggbi - gIgtotg);
                  *(here->BSIM4v4GPgpPtr +1) += m * xcggbr;
                  *(here->BSIM4v4GPdpPtr) -= m * (gcrgd - xcgdbi - gIgtotd);
                  *(here->BSIM4v4GPdpPtr +1) += m * xcgdbr;
                  *(here->BSIM4v4GPspPtr) -= m * (gcrgs - xcgsbi - gIgtots);
                  *(here->BSIM4v4GPspPtr +1) += m * xcgsbr;
                  *(here->BSIM4v4GPbpPtr) -= m * (gcrgb - xcgbbi - gIgtotb);
                  *(here->BSIM4v4GPbpPtr +1) += m * xcgbbr;
              }
              else
              {   *(here->BSIM4v4GPgpPtr +1) += m * xcggbr;
                  *(here->BSIM4v4GPgpPtr) += m * (xcggbi + gIgtotg);
                  *(here->BSIM4v4GPdpPtr +1) += m * xcgdbr;
                  *(here->BSIM4v4GPdpPtr) += m * (xcgdbi + gIgtotd);
                  *(here->BSIM4v4GPspPtr +1) += m * xcgsbr;
                  *(here->BSIM4v4GPspPtr) += m * (xcgsbi + gIgtots);
                  *(here->BSIM4v4GPbpPtr +1) += m * xcgbbr;
                  *(here->BSIM4v4GPbpPtr) += m * (xcgbbi + gIgtotb);
              }

              if (model->BSIM4v4rdsMod)
              {   (*(here->BSIM4v4DgpPtr) += m * gdtotg);
                  (*(here->BSIM4v4DspPtr) += m * gdtots);
                  (*(here->BSIM4v4DbpPtr) += m * gdtotb);
                  (*(here->BSIM4v4SdpPtr) += m * gstotd);
                  (*(here->BSIM4v4SgpPtr) += m * gstotg);
                  (*(here->BSIM4v4SbpPtr) += m * gstotb);
              }

              *(here->BSIM4v4DPdpPtr +1) += m * (xcddbr + gdsi + RevSumi);
              *(here->BSIM4v4DPdpPtr) += m * (gdpr + xcddbi + gdsr + here->BSIM4v4gbd 
				     - gdtotd + RevSumr + gbdpdp - gIdtotd);
              *(here->BSIM4v4DPdPtr) -= m * (gdpr + gdtot);
              *(here->BSIM4v4DPgpPtr +1) += m * (xcdgbr + Gmi);
              *(here->BSIM4v4DPgpPtr) += m * (Gmr + xcdgbi - gdtotg + gbdpg - gIdtotg);
              *(here->BSIM4v4DPspPtr +1) += m * (xcdsbr - gdsi - FwdSumi);
              *(here->BSIM4v4DPspPtr) -= m * (gdsr - xcdsbi + FwdSumr + gdtots - gbdpsp + gIdtots);
              *(here->BSIM4v4DPbpPtr +1) += m * (xcdbbr + Gmbsi);
              *(here->BSIM4v4DPbpPtr) -= m * (gjbd + gdtotb - xcdbbi - Gmbsr - gbdpb + gIdtotb);

              *(here->BSIM4v4DdpPtr) -= m * (gdpr - gdtotd);
              *(here->BSIM4v4DdPtr) += m * (gdpr + gdtot);

              *(here->BSIM4v4SPdpPtr +1) += m * (xcsdbr - gdsi - RevSumi);
              *(here->BSIM4v4SPdpPtr) -= m * (gdsr - xcsdbi + gstotd + RevSumr - gbspdp + gIstotd);
              *(here->BSIM4v4SPgpPtr +1) += m * (xcsgbr - Gmi);
              *(here->BSIM4v4SPgpPtr) -= m * (Gmr - xcsgbi + gstotg - gbspg + gIstotg);
              *(here->BSIM4v4SPspPtr +1) += m * (xcssbr + gdsi + FwdSumi);
              *(here->BSIM4v4SPspPtr) += m * (gspr + xcssbi + gdsr + here->BSIM4v4gbs
				     - gstots + FwdSumr + gbspsp - gIstots);
              *(here->BSIM4v4SPsPtr) -= m * (gspr + gstot);
              *(here->BSIM4v4SPbpPtr +1) += m * (xcsbbr - Gmbsi);
              *(here->BSIM4v4SPbpPtr) -= m * (gjbs + gstotb - xcsbbi + Gmbsr - gbspb + gIstotb);

              *(here->BSIM4v4SspPtr) -= m * (gspr - gstots);
              *(here->BSIM4v4SsPtr) += m * (gspr + gstot);

              *(here->BSIM4v4BPdpPtr +1) += m * xcbdb;
              *(here->BSIM4v4BPdpPtr) -= m * (gjbd - gbbdp + gIbtotd);
              *(here->BSIM4v4BPgpPtr +1) += m * xcbgb;
              *(here->BSIM4v4BPgpPtr) -= m * (here->BSIM4v4gbgs + gIbtotg);
              *(here->BSIM4v4BPspPtr +1) += m * xcbsb;
              *(here->BSIM4v4BPspPtr) -= m * (gjbs - gbbsp + gIbtots);
              *(here->BSIM4v4BPbpPtr +1) += m * xcbbb;
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
              {   (*(here->BSIM4v4DPdbPtr +1) += m * xcdbdb);
                  (*(here->BSIM4v4DPdbPtr) -= m * here->BSIM4v4gbd);
                  (*(here->BSIM4v4SPsbPtr +1) += m * xcsbsb);
                  (*(here->BSIM4v4SPsbPtr) -= m * here->BSIM4v4gbs);

                  (*(here->BSIM4v4DBdpPtr +1) += m * xcdbdb);
                  (*(here->BSIM4v4DBdpPtr) -= m * here->BSIM4v4gbd);
                  (*(here->BSIM4v4DBdbPtr +1) -= m * xcdbdb);
                  (*(here->BSIM4v4DBdbPtr) += m * (here->BSIM4v4gbd + here->BSIM4v4grbpd 
                                          + here->BSIM4v4grbdb));
                  (*(here->BSIM4v4DBbpPtr) -= m * here->BSIM4v4grbpd);
                  (*(here->BSIM4v4DBbPtr) -= m * here->BSIM4v4grbdb);

                  (*(here->BSIM4v4BPdbPtr) -= m * here->BSIM4v4grbpd);
                  (*(here->BSIM4v4BPbPtr) -= m * here->BSIM4v4grbpb);
                  (*(here->BSIM4v4BPsbPtr) -= m * here->BSIM4v4grbps);
                  (*(here->BSIM4v4BPbpPtr) += m * (here->BSIM4v4grbpd + here->BSIM4v4grbps 
					  + here->BSIM4v4grbpb));
		  /* WDLiu: (-here->BSIM4v4gbbs) already added to BPbpPtr */

                  (*(here->BSIM4v4SBspPtr +1) += m * xcsbsb);
                  (*(here->BSIM4v4SBspPtr) -= m * here->BSIM4v4gbs);
                  (*(here->BSIM4v4SBbpPtr) -= m * here->BSIM4v4grbps);
                  (*(here->BSIM4v4SBbPtr) -= m * here->BSIM4v4grbsb);
                  (*(here->BSIM4v4SBsbPtr +1) -= m * xcsbsb);
                  (*(here->BSIM4v4SBsbPtr) += m * (here->BSIM4v4gbs
					  + here->BSIM4v4grbps + here->BSIM4v4grbsb));

                  (*(here->BSIM4v4BdbPtr) -= m * here->BSIM4v4grbdb);
                  (*(here->BSIM4v4BbpPtr) -= m * here->BSIM4v4grbpb);
                  (*(here->BSIM4v4BsbPtr) -= m * here->BSIM4v4grbsb);
                  (*(here->BSIM4v4BbPtr) += m * (here->BSIM4v4grbsb + here->BSIM4v4grbdb
                                        + here->BSIM4v4grbpb));
              }


	   /*
	    * WDLiu: The internal charge node generated for transient NQS is not needed for
	    *        AC NQS. The following is not doing a real job, but we have to keep it;
	    *        otherwise a singular AC NQS matrix may occur if the transient NQS is on.
	    *        The charge node is isolated from the instance.
	    */
           if (here->BSIM4v4trnqsMod)
           {   (*(here->BSIM4v4QqPtr) += m * 1.0);
               (*(here->BSIM4v4QgpPtr) += 0.0);
               (*(here->BSIM4v4QdpPtr) += 0.0);
               (*(here->BSIM4v4QspPtr) += 0.0);
               (*(here->BSIM4v4QbpPtr) += 0.0);

               (*(here->BSIM4v4DPqPtr) += 0.0);
               (*(here->BSIM4v4SPqPtr) += 0.0);
               (*(here->BSIM4v4GPqPtr) += 0.0);
           }
         }
    }
    return(OK);
}
