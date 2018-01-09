/***  B4SOI 12/16/2010 Released by Tanvir Morshed  ***/


/**********
 * Copyright 2010 Regents of the University of California.  All rights reserved.
 * Authors: 1998 Samuel Fung, Dennis Sinitsky and Stephen Tang
 * Authors: 1999-2004 Pin Su, Hui Wan, Wei Jin, b3soiacld.c
 * Authors: 2005- Hui Wan, Xuemei Xi, Ali Niknejad, Chenming Hu.
 * Authors: 2009- Wenwei Yang, Chung-Hsun Lin, Ali Niknejad, Chenming Hu.
 * Authors: 2010- Tanvir Morshed, Ali Niknejad, Chenming Hu.
 * File: b4soiacld.c
 * Modified by Hui Wan, Xuemei Xi 11/30/2005
 * Modified by Wenwei Yang, Chung-Hsun Lin, Darsen Lu 03/06/2009
 * Modified by Tanvir Morshed 09/22/2009
 **********/

#include "ngspice/ngspice.h"

#include "ngspice/cktdefs.h"
#include "b4soidef.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
B4SOIacLoad(
GENmodel *inModel,
CKTcircuit *ckt)
{
register B4SOImodel *model = (B4SOImodel*)inModel;
register B4SOIinstance *here;
register int selfheat;
double xcggb, xcgdb, xcgsb, xcgeb, xcgT;
double xcdgb, xcddb, xcdsb, xcdeb, xcdT;
double xcsgb, xcsdb, xcssb, xcseb, xcsT;
double xcbgb, xcbdb, xcbsb, xcbeb, xcbT;
double xcegb, xceeb, xceT;
double gdpr, gspr, gds;
double cggb, cgdb, cgsb, cgT;
double cdgb, cddb, cdsb, cdeb, cdT;
double cbgb, cbdb, cbsb, cbeb, cbT;
double ceeb, ceT;
double GSoverlapCap, GDoverlapCap, GEoverlapCap, FwdSum, RevSum, Gm, Gmbs, GmT;
double omega;
double dxpart, sxpart;
double gbbg, gbbdp, gbbb, gbbp, gbbsp, gbbT;
double gddpg, gddpdp, gddpsp, gddpb, gddpT;
double gsspg, gsspdp, gsspsp, gsspb, gsspT;
double gppb, gppp;
double xcTt, cTt, gTtt, gTtg, gTtb, gTtdp, gTtsp;
double EDextrinsicCap, ESextrinsicCap;
double xcedb, xcesb;

/* v3.0 */
double Gme, gddpe, gsspe, gbbe, gTte;

/* v3.1 added variables for RF */
double T0=0.0;
double gcrgd, gcrgg, gcrgs, gcrgb, gcrg;
double xcgmgmb, xcgmdb, xcgmsb, xcgmeb, xcdgmb, xcsgmb, xcegmb;
double geltd;
double gigg, gigd, gigs, gigb, gige, gigT;
double gigpg, gigpp;

/* v3.1.1 bug fix */
double gIstotg, gIstotd, gIstotb, gIstots;
double gIdtotg, gIdtotd, gIdtotb, gIdtots;
double gIgtotg, gIgtotd, gIgtotb, gIgtots;

/* v4.0 */
double xcdbb, xcsbb, xcdbdb, xcsbsb, xcjdbdp, xcjsbsp;
double gstot, gstotd, gstotg, gstots, gstotb;
double gdtot, gdtotd, gdtotg, gdtots, gdtotb;

double m;

    omega = ckt->CKTomega;
    for (; model != NULL; model = B4SOInextModel(model)) 
    {    

      for (here = B4SOIinstances(model); here!= NULL;
              here = B4SOInextInstance(here)) 
         {
              selfheat = (model->B4SOIshMod == 1) && (here->B4SOIrth0 != 0.0);
              if (here->B4SOImode >= 0) 
              {   Gm = here->B4SOIgm;
                  Gmbs = here->B4SOIgmbs;

/* v3.0 */
                  Gme = here->B4SOIgme;

                  GmT = model->B4SOItype * here->B4SOIgmT;
                  FwdSum = Gm + Gmbs + Gme; /* v3.0 */
                  RevSum = 0.0;

                  cbgb = here->B4SOIcbgb;
                  cbsb = here->B4SOIcbsb;
                  cbdb = here->B4SOIcbdb;
                  cbeb = here->B4SOIcbeb;
                  cbT  = model->B4SOItype * here->B4SOIcbT;

                  ceeb = here->B4SOIceeb;
                  ceT  = model->B4SOItype * here->B4SOIceT;

                  cggb = here->B4SOIcggb;
                  cgsb = here->B4SOIcgsb;
                  cgdb = here->B4SOIcgdb;
                  cgT  = model->B4SOItype * here->B4SOIcgT;

                  cdgb = here->B4SOIcdgb;
                  cdsb = here->B4SOIcdsb;
                  cddb = here->B4SOIcddb;
                  cdeb = here->B4SOIcdeb;
                  cdT  = model->B4SOItype * here->B4SOIcdT;

                  cTt = here->pParam->B4SOIcth;


/* v3.1 bug fix */
                  gigg = here->B4SOIgigg;
                  gigb = here->B4SOIgigb;
                  gige = here->B4SOIgige; 
                  gigs = here->B4SOIgigs;
                  gigd = here->B4SOIgigd;
                  gigT = model->B4SOItype * here->B4SOIgigT;

                  /* v4.1 */
                  gigpg = here->B4SOIgigpg;
                  gigpp = here->B4SOIgigpp;

                  gbbg  = -here->B4SOIgbgs;
                  gbbdp = -here->B4SOIgbds;
                  gbbb  = -here->B4SOIgbbs;
                  gbbp  = -here->B4SOIgbps;
                  gbbT  = -model->B4SOItype * here->B4SOIgbT;
                  gbbe  = -here->B4SOIgbes;

                  if (here->B4SOIrbodyMod) { /* v4.0 */
                          gbbdp = -here->B4SOIgiigidld;
                        gbbb = -here->B4SOIgbgiigbpb;
                  }

                  gbbsp = - ( gbbg + gbbdp + gbbb + gbbp + gbbe);

                  gddpg  = -here->B4SOIgjdg;
                  gddpdp = -here->B4SOIgjdd;
                  if (!here->B4SOIrbodyMod) /* v4.0 */
                          gddpb  = -here->B4SOIgjdb;
                  else
                        gddpb = here->B4SOIgiigidlb;

                  gddpT  = -model->B4SOItype * here->B4SOIgjdT;

/* v3.0 */
                  gddpe  = -here->B4SOIgjde;
                  gddpsp = - ( gddpg + gddpdp + gddpb + gddpe);

                  gsspg  = -here->B4SOIgjsg;
                  gsspdp = -here->B4SOIgjsd;
                  if (!here->B4SOIrbodyMod) /* v4.0 */
                          gsspb  = -here->B4SOIgjsb;
                  else
                        gsspb = 0.0;
                  gsspT  = -model->B4SOItype * here->B4SOIgjsT;

                  gsspe  = 0.0;
                  gsspsp = - (gsspg + gsspdp + gsspb + gsspe);

                       gppb = -here->B4SOIgbpbs;
                  gppp = -here->B4SOIgbpps;

                  gTtg  = here->B4SOIgtempg;
                  gTtb  = here->B4SOIgtempb;
                  gTtdp = here->B4SOIgtempd;
                  gTtt  = here->B4SOIgtempT;

/* v3.0 */
                  gTte  = here->B4SOIgtempe;
                  gTtsp = - (gTtg + gTtb + gTtdp + gTte);


/* v3.1.1 bug fix */
                  if (model->B4SOIigcMod)
                  {  
                     gIstotg = here->B4SOIgIgsg + here->B4SOIgIgcsg;
                     gIstotd = here->B4SOIgIgcsd;
                     gIstots = here->B4SOIgIgss + here->B4SOIgIgcss;
                     gIstotb = here->B4SOIgIgcsb;

                     gIdtotg = here->B4SOIgIgdg + here->B4SOIgIgcdg;
                     gIdtotd = here->B4SOIgIgdd + here->B4SOIgIgcdd;
                     gIdtots = here->B4SOIgIgcds;
                     gIdtotb = here->B4SOIgIgcdb;

                     gIgtotg = gIstotg + gIdtotg;
                     gIgtotd = gIstotd + gIdtotd;
                     gIgtots = gIstots + gIdtots;
                     gIgtotb = gIstotb + gIdtotb;
                  }
                  else
                  {   
                     gIstotg = gIstotd = gIstots = gIstotb = 0.0;
                     gIdtotg = gIdtotd = gIdtots = gIdtotb = 0.0;
                     gIgtotg = gIgtotd = gIgtots = gIgtotb = 0.0;
                  }


                  sxpart = 0.6;
                  dxpart = 0.4;

                  /* v3.1 for RF */
                  if (here->B4SOIrgateMod == 2)
                      T0 = *(ckt->CKTstates[0] + here->B4SOIvges)
                         - *(ckt->CKTstates[0] + here->B4SOIvgs);
                  else if (here->B4SOIrgateMod == 3)
                      T0 = *(ckt->CKTstates[0] + here->B4SOIvgms)
                         - *(ckt->CKTstates[0] + here->B4SOIvgs);
                  if (here->B4SOIrgateMod > 1)
                  {   gcrgd = here->B4SOIgcrgd * T0;
                      gcrgg = here->B4SOIgcrgg * T0;
                      gcrgs = here->B4SOIgcrgs * T0;
                      gcrgb = here->B4SOIgcrgb * T0;
                      gcrgg -= here->B4SOIgcrg;
                      gcrg = here->B4SOIgcrg;
                  }
                  else
                  gcrg = gcrgd = gcrgg = gcrgs = gcrgb = 0.0;
                  /* v3.1 for RF end*/

              } 
              else
              {   Gm = -here->B4SOIgm;
                  Gmbs = -here->B4SOIgmbs;

/* v3.0 */
                  Gme = -here->B4SOIgme;

                  GmT = -model->B4SOItype * here->B4SOIgmT;
                  FwdSum = 0.0;
                  RevSum = -Gm - Gmbs - Gme; /* v3.0 */

                  cdgb = - (here->B4SOIcdgb + here->B4SOIcggb + here->B4SOIcbgb);
                  cdsb = - (here->B4SOIcddb + here->B4SOIcgdb + here->B4SOIcbdb);
                  cddb = - (here->B4SOIcdsb + here->B4SOIcgsb + here->B4SOIcbsb);
                  cdeb = - (here->B4SOIcdeb + here->B4SOIcbeb + here->B4SOIceeb);
                  cdT  = - model->B4SOItype * (here->B4SOIcgT + here->B4SOIcbT
                         + here->B4SOIcdT + here->B4SOIceT);

                  ceeb = here->B4SOIceeb;
                  ceT  = model->B4SOItype * here->B4SOIceT;

                  cggb = here->B4SOIcggb;
                  cgsb = here->B4SOIcgdb;
                  cgdb = here->B4SOIcgsb;
                  cgT  = model->B4SOItype * here->B4SOIcgT;

                  cbgb = here->B4SOIcbgb;
                  cbsb = here->B4SOIcbdb;
                  cbdb = here->B4SOIcbsb;
                  cbeb = here->B4SOIcbeb;
                  cbT  = model->B4SOItype * here->B4SOIcbT;

                  cTt = here->pParam->B4SOIcth;


/* v3.1 bug fix */
                  gigg = here->B4SOIgigg;
                  gigb = here->B4SOIgigb;
                  gige = here->B4SOIgige; 
                  gigs = here->B4SOIgigd; /* v3.1.1 bug fix */
                  gigd = here->B4SOIgigs; /* v3.1.1 bug fix */
                  gigT = model->B4SOItype * here->B4SOIgigT;

                  gigpg = here->B4SOIgigpg;/* bugfix_snps for setting gigpg gigpp*/
                  gigpp = here->B4SOIgigpp; 
                  
                  gbbg  = -here->B4SOIgbgs;
                  gbbb  = -here->B4SOIgbbs;
                  gbbp  = -here->B4SOIgbps;
                  gbbsp = -here->B4SOIgbds;
                  gbbT  = -model->B4SOItype * here->B4SOIgbT;
/* v3.0 */
                  gbbe  = -here->B4SOIgbes;

                  if (here->B4SOIrbodyMod) { /* v4.0 */
                        gbbsp = -here->B4SOIgiigidld;
                        gbbb = -here->B4SOIgbgiigbpb;
                  }

                  gbbdp = - ( gbbg + gbbsp + gbbb + gbbp + gbbe);

                  gddpg  = -here->B4SOIgjsg;
                  gddpsp = -here->B4SOIgjsd;
                  if (!here->B4SOIrbodyMod) /* v4.0 */
                          gddpb  = -here->B4SOIgjsb;
                  else 
                        gddpb = 0.0;

                  gddpT  = -model->B4SOItype * here->B4SOIgjsT;

/* v3.0 */
                  gddpe  = 0.0;
                  gddpdp = - (gddpg + gddpsp + gddpb + gddpe );

                  gsspg  = -here->B4SOIgjdg;
                  gsspsp = -here->B4SOIgjdd;
                  if (!here->B4SOIrbodyMod) /* v4.0 */
                          gsspb  = -here->B4SOIgjdb;
                  else
                        gsspb = here->B4SOIgiigidlb;

                  gsspT  = -model->B4SOItype * here->B4SOIgjdT;

/* v3.0 */
                  gsspe  = -here->B4SOIgjde;
                  gsspdp = - ( gsspg + gsspsp + gsspb + gsspe );

                  gppb = -here->B4SOIgbpbs;
                  gppp = -here->B4SOIgbpps;

                  gTtg  = here->B4SOIgtempg;
                  gTtb  = here->B4SOIgtempb;
                  gTtsp = here->B4SOIgtempd;
                  gTtt  = here->B4SOIgtempT;

/* v3.0 */
                  gTte = here->B4SOIgtempe;
                  gTtdp = - (gTtg + gTtb + gTtsp + gTte);


/* v3.1.1 bug fix */
                  if (model->B4SOIigcMod)
                  {   
                      gIstotg = here->B4SOIgIgsg + here->B4SOIgIgcdg;
                      gIstotd = here->B4SOIgIgcds;
                      gIstots = here->B4SOIgIgss + here->B4SOIgIgcdd;
                      gIstotb = here->B4SOIgIgcdb;

                      gIdtotg = here->B4SOIgIgdg + here->B4SOIgIgcsg;
                      gIdtotd = here->B4SOIgIgdd + here->B4SOIgIgcss;
                      gIdtots = here->B4SOIgIgcsd;
                      gIdtotb = here->B4SOIgIgcsb;

                      gIgtotg = gIstotg + gIdtotg;
                      gIgtotd = gIstotd + gIdtotd;
                      gIgtots = gIstots + gIdtots;
                      gIgtotb = gIstotb + gIdtotb;
                  }
                  else
                  {   
                      gIstotg = gIstotd = gIstots = gIstotb = 0.0;
                      gIdtotg = gIdtotd = gIdtots = gIdtotb = 0.0;
                      gIgtotg = gIgtotd = gIgtots = gIgtotb = 0.0;
                  }


                  sxpart = 0.4;
                  dxpart = 0.6;

                  /* v3.1 for RF */
                  if (here->B4SOIrgateMod == 2)
                      T0 = *(ckt->CKTstates[0] + here->B4SOIvges)
                         - *(ckt->CKTstates[0] + here->B4SOIvgs);
                  else if (here->B4SOIrgateMod == 3)
                      T0 = *(ckt->CKTstates[0] + here->B4SOIvgms)
                         - *(ckt->CKTstates[0] + here->B4SOIvgs);
                  if (here->B4SOIrgateMod > 1)
                  {   gcrgd = here->B4SOIgcrgs * T0;
                      gcrgg = here->B4SOIgcrgg * T0;
                      gcrgs = here->B4SOIgcrgd * T0;
                      gcrgb = here->B4SOIgcrgb * T0;
                      gcrgg -= here->B4SOIgcrg;
                      gcrg = here->B4SOIgcrg;
                  }
                  else
                  gcrg = gcrgd = gcrgg = gcrgs = gcrgb = 0.0;

                  /* v3.1 RF end*/

              }

              if (!model->B4SOIrdsMod)        {
                      gdpr=here->B4SOIdrainConductance;
                      gspr=here->B4SOIsourceConductance;
              }
              else gdpr = gspr = 0.0;


              gds= here->B4SOIgds;

              GSoverlapCap = here->B4SOIcgso;
              GDoverlapCap = here->B4SOIcgdo;
              GEoverlapCap = here->pParam->B4SOIcgeo;

              EDextrinsicCap = here->B4SOIgcde;
              ESextrinsicCap = here->B4SOIgcse;


              /* v3.1 added for RF */
              if (here->B4SOIrgateMod == 3)
              {
                xcgmgmb = (GDoverlapCap + GSoverlapCap + GEoverlapCap ) * omega;
                xcgmdb = -GDoverlapCap * omega;
                xcgmsb = -GSoverlapCap * omega;
                xcgmeb = -GEoverlapCap * omega;

                xcdgmb = xcgmdb;
                xcsgmb = xcgmsb;
                xcegmb = xcgmeb;

                xcedb = -EDextrinsicCap * omega;
                xcdeb = (cdeb - EDextrinsicCap) * omega; 
                xcddb = (cddb + GDoverlapCap + EDextrinsicCap) * omega;
                xceeb = (ceeb + GEoverlapCap + EDextrinsicCap + ESextrinsicCap)
                      * omega;
                xcesb = -ESextrinsicCap * omega;
                xcssb = (GSoverlapCap + ESextrinsicCap - (cgsb + cbsb + cdsb))
                      * omega;

                xcseb = -(cbeb + cdeb + ceeb + ESextrinsicCap) * omega;

                xcegb = 0; /* v3.1 change for RF */
                xceT  =  ceT * omega;
                xcggb = here->B4SOIcggb * omega; 
                xcgdb = cgdb * omega;     
                xcgsb = cgsb * omega;     
                xcgeb = 0; 
                xcgT  = cgT * omega;

                xcdgb = cdgb * omega;     
                xcdsb = cdsb * omega;
                xcdT  = cdT * omega;

                xcsgb = -(cggb + cbgb + cdgb) * omega; 
                xcsdb = -(cgdb + cbdb + cddb) * omega;
                xcsT  = -(cgT + cbT + cdT + ceT) * omega;

                xcbgb = cbgb * omega;
                xcbdb = cbdb * omega;
                xcbsb = cbsb * omega;
                xcbeb = cbeb * omega;
                xcbT  = cbT * omega;

                xcTt = cTt * omega;
              }

              else 
              {
                xcedb = -EDextrinsicCap * omega;
                xcdeb = (cdeb - EDextrinsicCap) * omega;
                xcddb = (cddb + GDoverlapCap + EDextrinsicCap) * omega;
                xceeb = (ceeb + GEoverlapCap + EDextrinsicCap + ESextrinsicCap)
                      * omega;
                xcesb = -ESextrinsicCap * omega;
                xcssb = (GSoverlapCap + ESextrinsicCap - (cgsb + cbsb + cdsb))
                      * omega;

                xcseb = -(cbeb + cdeb + ceeb + ESextrinsicCap) * omega;

                xcegb = (- GEoverlapCap) * omega;
                xceT  =  ceT * omega;
                xcggb = (cggb + GDoverlapCap + GSoverlapCap + GEoverlapCap)
                      * omega;
                xcgdb = (cgdb - GDoverlapCap ) * omega;
                xcgsb = (cgsb - GSoverlapCap) * omega;
                xcgeb = (- GEoverlapCap) * omega;
                xcgT  = cgT * omega;

                xcdgb = (cdgb - GDoverlapCap) * omega;
                xcdsb = cdsb * omega;
                xcdT  = cdT * omega;

                xcsgb = -(cggb + cbgb + cdgb + GSoverlapCap) * omega;
                xcsdb = -(cgdb + cbdb + cddb) * omega;
                xcsT  = -(cgT + cbT + cdT + ceT) * omega;

                xcbgb = cbgb * omega;
                xcbdb = cbdb * omega;
                xcbsb = cbsb * omega;
                xcbeb = cbeb * omega;
                xcbT  = cbT * omega;

                xcTt = cTt * omega;
        
                 /* v3.1 */
                xcdgmb = xcsgmb = xcegmb = 0.0;
                xcgmgmb = xcgmdb = xcgmsb = xcgmeb =0.0;
              }

              if (here->B4SOImode >= 0) { /* v4.0 */
                  if (!here->B4SOIrbodyMod) {
                        xcjdbdp = xcjsbsp = 0.0;
                        xcdbb = -(xcdgb + xcddb + xcdsb + xcdgmb + xcdeb);
                        xcsbb = -(xcsgb + xcsdb + xcssb + xcsgmb + xcseb);
                        xcdbdb = 0.0; xcsbsb = 0.0;
                        xcbdb = here->B4SOIcbdb * omega;
                        xcbsb = here->B4SOIcbsb * omega;
                  }
                  else {
                        xcjdbdp = here->B4SOIcjdb * omega;
                        xcjsbsp = here->B4SOIcjsb * omega;
                        xcdbb = -(xcdgb + xcddb + xcdsb + xcdgmb + xcdeb)
                                 + xcjdbdp;
                        xcsbb = -(xcsgb + xcsdb + xcssb + xcsgmb + xcseb)
                                 + xcjsbsp;
                        xcdbdb = -here->B4SOIcjdb * omega;
                        xcsbsb = -here->B4SOIcjsb * omega;
                        xcbdb = here->B4SOIcbdb * omega - xcdbdb;
                        xcbsb = here->B4SOIcbsb * omega - xcsbsb;
                  }
              }
              else {
                  if (!here->B4SOIrbodyMod) {
                        xcjdbdp = xcjsbsp = 0.0;
                        xcdbb = -(xcdgb + xcddb + xcdsb + xcdgmb + xcdeb);
                        xcsbb = -(xcsgb + xcsdb + xcssb + xcsgmb + xcseb);
                        xcdbdb = 0.0; xcsbsb = 0.0;
                        xcbdb = here->B4SOIcbsb * omega;
                        xcbsb = here->B4SOIcbdb * omega;
                  }
                  else {
                        xcjdbdp = here->B4SOIcjsb * omega;
                        xcjsbsp = here->B4SOIcjdb * omega;
                        xcdbb = -(xcdgb + xcddb + xcdsb + xcdgmb + xcdeb)
                                 + xcjdbdp;
                        xcsbb = -(xcsgb + xcsdb + xcssb + xcsgmb + xcseb)
                                 + xcjsbsp;
                        xcdbdb = -here->B4SOIcjsb * omega;
                        xcsbsb = -here->B4SOIcjdb * omega;
                        xcbdb = here->B4SOIcbsb * omega - xcdbdb;
                        xcbsb = here->B4SOIcbdb * omega - xcsbsb;
                  }

              }

              if (model->B4SOIrdsMod == 1)
              {   gstot = here->B4SOIgstot;
                  gstotd = here->B4SOIgstotd;
                  gstotg = here->B4SOIgstotg;
                  gstots = here->B4SOIgstots - gstot;
                  gstotb = here->B4SOIgstotb;

                  gdtot = here->B4SOIgdtot;
                  gdtotd = here->B4SOIgdtotd - gdtot;
                  gdtotg = here->B4SOIgdtotg;
                  gdtots = here->B4SOIgdtots;
                  gdtotb = here->B4SOIgdtotb;
              }
              else
              {   gstot = gstotd = gstotg = gstots = gstotb = 0.0;
                  gdtot = gdtotd = gdtotg = gdtots = gdtotb = 0.0;
              }

        m = here->B4SOIm;

              /* v3.1 for RF */
              geltd = here->B4SOIgrgeltd;
              if (here->B4SOIrgateMod == 1)
              {  
                *(here->B4SOIGEgePtr) += m * geltd;
                *(here->B4SOIGgePtr) -= m * geltd;
                *(here->B4SOIGEgPtr) -= m * geltd;
                *(here->B4SOIGgPtr) += m * (geltd + gigg + gIgtotg); /* v3.1.1 bug fix */
                *(here->B4SOIGdpPtr) += m * (gigd + gIgtotd); /* v3.1.1 bug fix */
                *(here->B4SOIGspPtr) += m * (gigs + gIgtots); /* v3.1.1 bug fix */
                if (here->B4SOIsoiMod != 2) /* v3.2 */
                    *(here->B4SOIGbPtr) -= m * (-gigb - gIgtotb); /* v3.1.1 bug fix */
              }

              else if (here->B4SOIrgateMod == 2)
              {   
                *(here->B4SOIGEgePtr) += m * gcrg;
                *(here->B4SOIGEgPtr) += m * gcrgg;
                *(here->B4SOIGEdpPtr) += m * gcrgd;
                *(here->B4SOIGEspPtr) += m * gcrgs;
                if (here->B4SOIsoiMod != 2) /* v3.2 */
                *(here->B4SOIGEbPtr) += m * gcrgb; 

                *(here->B4SOIGgePtr) -= m * gcrg;
                *(here->B4SOIGgPtr) -= m * (gcrgg - gigg - gIgtotg); /* v3.1.1 bug fix */
                *(here->B4SOIGdpPtr) -= m * (gcrgd - gigd - gIgtotd); /* v3.1.1 bug fix */
                *(here->B4SOIGspPtr) -= m * (gcrgs - gigs - gIgtots); /* v3.1.1 bug fix */
                if (here->B4SOIsoiMod != 2) /* v3.2 */
                *(here->B4SOIGbPtr) -= m * (gcrgb - gigb - gIgtotb); /* v3.1.1 bug fix */
              } 

              else if (here->B4SOIrgateMod == 3)
              {   
                *(here->B4SOIGEgePtr) += m * geltd;
                *(here->B4SOIGEgmPtr) -= m * geltd;
                *(here->B4SOIGMgePtr) -= m * geltd;
                *(here->B4SOIGMgmPtr) += m * (geltd + gcrg);
                *(here->B4SOIGMgmPtr +1) += m * xcgmgmb;
 
                *(here->B4SOIGMdpPtr) += m * gcrgd;
                *(here->B4SOIGMdpPtr +1) += m * xcgmdb;
                *(here->B4SOIGMgPtr) += m * gcrgg;
                *(here->B4SOIGMspPtr) += m * gcrgs;
                *(here->B4SOIGMspPtr +1) += m * xcgmsb;
                if (here->B4SOIsoiMod != 2) /* v3.2 */
                *(here->B4SOIGMbPtr) += m * gcrgb;
                *(here->B4SOIGMePtr +1) += m * xcgmeb;
 
                *(here->B4SOIDPgmPtr +1) += m * xcdgmb;
                *(here->B4SOIGgmPtr) -= m * gcrg;
                *(here->B4SOISPgmPtr +1) += m * xcsgmb;
                *(here->B4SOIEgmPtr +1) += m * xcegmb;
 
                *(here->B4SOIGgPtr) -= m * (gcrgg - gigg - gIgtotg); /* v3.1.1 bug fix */
                *(here->B4SOIGdpPtr) -= m * (gcrgd - gigd - gIgtotd); /* v3.1.1 bug fix */
                *(here->B4SOIGspPtr) -= m * (gcrgs - gigs - gIgtots); /* v3.1.1 bug fix */
                if (here->B4SOIsoiMod != 2) /* v3.2 */
                *(here->B4SOIGbPtr) -= m * (gcrgb - gigb - gIgtotb); /* v3.1.1 bug fix */
              }
              else
              {   
                *(here->B4SOIGgPtr)  += m * (gigg + gIgtotg); /* v3.1.1 bug fix */
                *(here->B4SOIGdpPtr) += m * (gigd + gIgtotd); /* v3.1.1 bug fix */
                *(here->B4SOIGspPtr) += m * (gigs + gIgtots); /* v3.1.1 bug fix */
                if (here->B4SOIsoiMod != 2) /* v3.2 */
                *(here->B4SOIGbPtr)  += m * (gigb + gIgtotb); /* v3.1.1 bug fix */
              }
              /* v3.1 for RF end*/

              if (model->B4SOIrdsMod)
              {   (*(here->B4SOIDgPtr) += m * gdtotg);
                  (*(here->B4SOIDspPtr) += m * gdtots);
                  (*(here->B4SOISdpPtr) += m * gstotd);
                  (*(here->B4SOISgPtr) += m * gstotg);
                  if (here->B4SOIsoiMod != 2) {
                          (*(here->B4SOIDbPtr) += m * gdtotb);
                          (*(here->B4SOISbPtr) += m * gstotb);
                  }
              }

              *(here->B4SOIEdpPtr +1) += m * xcedb;
              *(here->B4SOIEspPtr +1) += m * xcesb;
              *(here->B4SOIDPePtr +1) += m * xcdeb;
              *(here->B4SOISPePtr +1) += m * xcseb;
              *(here->B4SOIEgPtr  +1) += m * xcegb;
              *(here->B4SOIGePtr  +1) += m * xcgeb;

              *(here->B4SOIEePtr  +1) += m * xceeb;

              *(here->B4SOIGgPtr  +1) += m * xcggb;
              *(here->B4SOIGdpPtr +1) += m * xcgdb;
              *(here->B4SOIGspPtr +1) += m * xcgsb;

              *(here->B4SOIDPgPtr +1) += m * xcdgb;
              *(here->B4SOIDPdpPtr +1) += m * xcddb;
              *(here->B4SOIDPspPtr +1) += m * xcdsb;

              *(here->B4SOISPgPtr +1) += m * xcsgb;
              *(here->B4SOISPdpPtr +1) += m * xcsdb;
              *(here->B4SOISPspPtr +1) += m * xcssb;

/* v3.1 */
              if (here->B4SOIsoiMod != 2) /* v3.2 */
              {
                 *(here->B4SOIBePtr +1) += m * xcbeb;
                 *(here->B4SOIBgPtr +1) += m * xcbgb;
                 *(here->B4SOIBdpPtr +1) += m * xcbdb;
                 *(here->B4SOIBspPtr +1) += m * xcbsb;

                 *(here->B4SOIEbPtr  +1) -= m * (xcegb + xceeb + xcedb + xcesb);
                 *(here->B4SOIGbPtr +1) -= m * (xcggb + xcgdb + xcgsb + xcgeb);
/*                 *(here->B4SOIDPbPtr +1) -= xcdgb + xcddb + xcdsb + xcdeb; 
                                          + xcdgmb; */

                 *(here->B4SOIDPbPtr +1) -= m * -xcdbb; /* v4.0 */

/*                 *(here->B4SOISPbPtr +1) -= xcsgb + xcsdb + xcssb + xcseb
                                            + xcsgmb; */
                 *(here->B4SOISPbPtr +1) -= m * -xcsbb; /* v4.0 */
                 *(here->B4SOIBbPtr +1) -= m * (xcbgb + xcbdb + xcbsb + xcbeb);
              }
/* v3.1 */


              if (selfheat)
              {
                 *(here->B4SOITemptempPtr + 1) += m * xcTt;
                 *(here->B4SOIDPtempPtr + 1) += m * xcdT;
                 *(here->B4SOISPtempPtr + 1) += m * xcsT;
                 *(here->B4SOIBtempPtr + 1) += m * xcbT;
                 *(here->B4SOIEtempPtr + 1) += m * xceT;
                 *(here->B4SOIGtempPtr + 1) += m * xcgT;
              }
                                                               
 
/* v3.0 */
              if (here->B4SOIsoiMod != 0) /* v3.2 */
              {
                 *(here->B4SOIDPePtr) += m * (Gme + gddpe);
                 *(here->B4SOISPePtr) += m * (gsspe - Gme);

                 if (here->B4SOIsoiMod != 2) /* v3.2 */
                 {
                    *(here->B4SOIGePtr) += m * gige;
                    *(here->B4SOIBePtr) -= m * gige;
                 }
              }

              *(here->B4SOIEePtr) += 0.0;

              *(here->B4SOIDPgPtr) += m * (Gm + gddpg - gIdtotg -gdtotg); /* v4.0 */
              *(here->B4SOIDPdpPtr) += m * (gdpr + gds + gddpdp + RevSum - gIdtotd
                                     - gdtotd); /* v4.0 */

              *(here->B4SOIDPspPtr) -= m * (gds + FwdSum - gddpsp + gIdtots 
                                     + gdtots); /* v4.0 */
              *(here->B4SOIDPdPtr) -= m * (gdpr + gdtot);

              *(here->B4SOISPgPtr) -= m * (Gm - gsspg + gIstotg + gstotg); /* v4.0 */
              *(here->B4SOISPdpPtr) -= m * (gds + RevSum - gsspdp + gIstotd
                                     + gstotd); /* v4.0 */
              *(here->B4SOISPspPtr) += m * (gspr + gds + FwdSum + gsspsp - gIstots
                                     - gstots); /* v4.0 */
              *(here->B4SOISPsPtr) -= m * (gspr + gstot);


/* v3.1 */
              if (here->B4SOIsoiMod != 2) /* v3.2 */
              {
                 *(here->B4SOIBePtr) += m * gbbe; /* v3.0 */
                 *(here->B4SOIBgPtr)  += m * (gbbg - gigg); /* v3.1 bug fix */
                 *(here->B4SOIBdpPtr) += m * (gbbdp - gigd); /* v3.1 bug fix */
                 *(here->B4SOIBspPtr) += m * (gbbsp - gigs); /* v3.1 bug fix */
                 *(here->B4SOIBbPtr) += m * (gbbb - gigb); /* v3.1 bug fix */
                 *(here->B4SOISPbPtr) -= m * (Gmbs - gsspb + gIstotb + gstotb); 
                                        /* v4.0 */
                 *(here->B4SOIDPbPtr) -= m * ((-gddpb - Gmbs) + gIdtotb + gdtotb); 
                                        /* v4.0 */
              }
/* v3.1 */



              if (selfheat)
              {
                 *(here->B4SOIDPtempPtr) += m * (GmT + gddpT);
                 *(here->B4SOISPtempPtr) += m * (-GmT + gsspT);
                 *(here->B4SOIBtempPtr) += m * (gbbT - gigT); /* v3.1 bug fix */
                 *(here->B4SOIGtempPtr) += m * gigT; /* v3.1 bug fix */
                 *(here->B4SOITemptempPtr) += m * (gTtt + 1/here->pParam->B4SOIrth);
                 *(here->B4SOITempgPtr) += m * gTtg;
                 *(here->B4SOITempbPtr) += m * gTtb;
                 *(here->B4SOITempdpPtr) += m * gTtdp;
                 *(here->B4SOITempspPtr) += m * gTtsp;

/* v3.0 */       
                 if (here->B4SOIsoiMod != 0) /* v3.2 */
                    *(here->B4SOITempePtr) += m * gTte;
              }


              *(here->B4SOIDdPtr) += m * (gdpr + gdtot);
              *(here->B4SOIDdpPtr) -= m * (gdpr - gdtotd);
              *(here->B4SOISsPtr) += m * (gspr + gstot);
              *(here->B4SOISspPtr) -= m * (gspr -gstots);


              if (here->B4SOIbodyMod == 1) {
                 (*(here->B4SOIBpPtr) -= m * gppp);
                 (*(here->B4SOIPbPtr) += m * gppb);
                 (*(here->B4SOIPpPtr) += m * gppp);
              }

              /* v4.1  Ig_agbcp2 stamping */
              (*(here->B4SOIGgPtr) += gigpg);
              if (here->B4SOIbodyMod == 1)  {
                  (*(here->B4SOIPpPtr) -= m * gigpp);
                  (*(here->B4SOIPgPtr) -= m * gigpg);
                  (*(here->B4SOIGpPtr) += m * gigpp);
              }
              else if(here->B4SOIbodyMod == 2)
              {
                  (*(here->B4SOIBbPtr) -= m * gigpp);
                  (*(here->B4SOIBgPtr) -= m * gigpg);
                  (*(here->B4SOIGbPtr) += m * gigpp);
              }



              /* v4.0 */
              if (here->B4SOIrbodyMod)
              {
                      (*(here->B4SOIDPdbPtr + 1) -= m * xcjdbdp);
                      (*(here->B4SOIDPdbPtr) -= m * here->B4SOIGGjdb);
                (*(here->B4SOISPsbPtr + 1) -= m * xcjsbsp);
                (*(here->B4SOISPsbPtr) -= m * here->B4SOIGGjsb);

                (*(here->B4SOIDBdpPtr + 1) -= m * xcjdbdp);
                (*(here->B4SOIDBdpPtr) -= m * here->B4SOIGGjdb);
                (*(here->B4SOIDBdbPtr + 1) += m * xcjdbdp);
                (*(here->B4SOIDBdbPtr) += m * (here->B4SOIGGjdb
                                        + here->B4SOIgrbdb));

                (*(here->B4SOIDBbPtr) -= m * here->B4SOIgrbdb);
                (*(here->B4SOISBbPtr) -= m * here->B4SOIgrbsb);

                (*(here->B4SOISBspPtr + 1) -= m * xcjsbsp);
                (*(here->B4SOISBspPtr) -= m * here->B4SOIGGjsb);
                (*(here->B4SOISBsbPtr + 1) += m * xcjsbsp);
                (*(here->B4SOISBsbPtr) += m * (here->B4SOIGGjsb 
                                        + here->B4SOIgrbsb));

                (*(here->B4SOIBdbPtr) -= m * here->B4SOIgrbdb);
                (*(here->B4SOIBsbPtr) -= m * here->B4SOIgrbsb);
                (*(here->B4SOIBbPtr) += m * (here->B4SOIgrbsb
                                        + here->B4SOIgrbdb));

              } 

              if (here->B4SOIdebugMod != 0)
              {
                      *(here->B4SOIVbsPtr) += 1;
                      *(here->B4SOIIdsPtr) += 1;
                      *(here->B4SOIIcPtr) += 1;
                      *(here->B4SOIIbsPtr) += 1;
                      *(here->B4SOIIbdPtr) += 1;
                      *(here->B4SOIIiiPtr) += 1;
                      *(here->B4SOIIgidlPtr) += 1;
                      *(here->B4SOIItunPtr) += 1;
                      *(here->B4SOIIbpPtr) += 1;
                      *(here->B4SOICbgPtr) += 1;
                      *(here->B4SOICbbPtr) += 1;
                      *(here->B4SOICbdPtr) += 1;
                      *(here->B4SOIQbfPtr) += 1;
                      *(here->B4SOIQjsPtr) += 1;
                      *(here->B4SOIQjdPtr) += 1;

              }

        }
    }
    return(OK);
}

