/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1998 Samuel Fung, Dennis Sinitsky and Stephen Tang
File: b3soiacld.c          98/5/01
Modified by Pin Su    99/4/30
Modified by Pin Su    99/9/27
Modified by Pin Su    02/5/20
Modified by Paolo Nenzi 2002
**********/

#include "ngspice.h"
#include "cktdefs.h"
#include "b3soidef.h"
#include "sperror.h"
#include "suffix.h"


int
B3SOIacLoad(GENmodel *inModel, CKTcircuit *ckt)
{
B3SOImodel *model = (B3SOImodel*)inModel;
B3SOIinstance *here;
int selfheat;
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

double m;


    omega = ckt->CKTomega;
    for (; model != NULL; model = model->B3SOInextModel) 
    {    

      for (here = model->B3SOIinstances; here!= NULL;
              here = here->B3SOInextInstance) 
	 {    
              
	      if (here->B3SOIowner != ARCHme)
                      continue;

	      
	      selfheat = (model->B3SOIshMod == 1) && (here->B3SOIrth0 != 0.0);
              if (here->B3SOImode >= 0) 
	      {   Gm = here->B3SOIgm;
		  Gmbs = here->B3SOIgmbs;

/* v3.0 */
                  Gme = here->B3SOIgme;

                  GmT = model->B3SOItype * here->B3SOIgmT;
		  FwdSum = Gm + Gmbs + Gme; /* v3.0 */
		  RevSum = 0.0;

                  cbgb = here->B3SOIcbgb;
                  cbsb = here->B3SOIcbsb;
                  cbdb = here->B3SOIcbdb;
                  cbeb = here->B3SOIcbeb;
                  cbT  = model->B3SOItype * here->B3SOIcbT;

                  ceeb = here->B3SOIceeb;
                  ceT  = model->B3SOItype * here->B3SOIceT;

                  cggb = here->B3SOIcggb;
                  cgsb = here->B3SOIcgsb;
                  cgdb = here->B3SOIcgdb;
                  cgT  = model->B3SOItype * here->B3SOIcgT;

                  cdgb = here->B3SOIcdgb;
                  cdsb = here->B3SOIcdsb;
                  cddb = here->B3SOIcddb;
                  cdeb = here->B3SOIcdeb;
                  cdT  = model->B3SOItype * here->B3SOIcdT;

                  cTt = here->pParam->B3SOIcth;

                  gbbg  = -here->B3SOIgbgs;
                  gbbdp = -here->B3SOIgbds;
                  gbbb  = -here->B3SOIgbbs;
                  gbbp  = -here->B3SOIgbps;
                  gbbT  = -model->B3SOItype * here->B3SOIgbT;
                  
/* v3.0 */
                  gbbe  = -here->B3SOIgbes;
                  gbbsp = - ( gbbg + gbbdp + gbbb + gbbp + gbbe);

                  gddpg  = -here->B3SOIgjdg;
                  gddpdp = -here->B3SOIgjdd;
                  gddpb  = -here->B3SOIgjdb;
                  gddpT  = -model->B3SOItype * here->B3SOIgjdT;

/* v3.0 */
                  gddpe  = -here->B3SOIgjde;
                  gddpsp = - ( gddpg + gddpdp + gddpb + gddpe);

                  gsspg  = -here->B3SOIgjsg;
                  gsspdp = -here->B3SOIgjsd;
                  gsspb  = -here->B3SOIgjsb;
                  gsspT  = -model->B3SOItype * here->B3SOIgjsT;

/* v3.0 */
                  gsspe  = 0.0;
                  gsspsp = - (gsspg + gsspdp + gsspb + gsspe);


             	  gppb = -here->B3SOIgbpbs;
                  gppp = -here->B3SOIgbpps;

                  gTtg  = here->B3SOIgtempg;
                  gTtb  = here->B3SOIgtempb;
                  gTtdp = here->B3SOIgtempd;
                  gTtt  = here->B3SOIgtempT;

/* v3.0 */
                  gTte  = here->B3SOIgtempe;
                  gTtsp = - (gTtg + gTtb + gTtdp + gTte);

                  sxpart = 0.6;
                  dxpart = 0.4;

              } 
	      else
	      {   Gm = -here->B3SOIgm;
		  Gmbs = -here->B3SOIgmbs;

/* v3.0 */
                  Gme = -here->B3SOIgme;

                  GmT = -model->B3SOItype * here->B3SOIgmT;
		  FwdSum = 0.0;
		  RevSum = -Gm - Gmbs - Gme; /* v3.0 */

                  cdgb = - (here->B3SOIcdgb + here->B3SOIcggb + here->B3SOIcbgb);
                  cdsb = - (here->B3SOIcddb + here->B3SOIcgdb + here->B3SOIcbdb);
                  cddb = - (here->B3SOIcdsb + here->B3SOIcgsb + here->B3SOIcbsb);
                  cdeb = - (here->B3SOIcdeb + here->B3SOIcbeb + here->B3SOIceeb);
                  cdT  = - model->B3SOItype * (here->B3SOIcgT + here->B3SOIcbT
                         + here->B3SOIcdT + here->B3SOIceT);

                  ceeb = here->B3SOIceeb;
                  ceT  = model->B3SOItype * here->B3SOIceT;

                  cggb = here->B3SOIcggb;
                  cgsb = here->B3SOIcgdb;
                  cgdb = here->B3SOIcgsb;
                  cgT  = model->B3SOItype * here->B3SOIcgT;

                  cbgb = here->B3SOIcbgb;
                  cbsb = here->B3SOIcbdb;
                  cbdb = here->B3SOIcbsb;
                  cbeb = here->B3SOIcbeb;
                  cbT  = model->B3SOItype * here->B3SOIcbT;

                  cTt = here->pParam->B3SOIcth;

                  gbbg  = -here->B3SOIgbgs;
                  gbbb  = -here->B3SOIgbbs;
                  gbbp  = -here->B3SOIgbps;
                  gbbsp = -here->B3SOIgbds;
                  gbbT  = -model->B3SOItype * here->B3SOIgbT;

/* v3.0 */
                  gbbe  = -here->B3SOIgbes;
                  gbbdp = - ( gbbg + gbbsp + gbbb + gbbp + gbbe);

                  gddpg  = -here->B3SOIgjsg;
                  gddpsp = -here->B3SOIgjsd;
                  gddpb  = -here->B3SOIgjsb;
                  gddpT  = -model->B3SOItype * here->B3SOIgjsT;

/* v3.0 */
                  gddpe  = 0.0;
                  gddpdp = - (gddpg + gddpsp + gddpb + gddpe );

                  gsspg  = -here->B3SOIgjdg;
                  gsspsp = -here->B3SOIgjdd;
                  gsspb  = -here->B3SOIgjdb;
                  gsspT  = -model->B3SOItype * here->B3SOIgjdT;

/* v3.0 */
                  gsspe  = -here->B3SOIgjde;
                  gsspdp = - ( gsspg + gsspsp + gsspb + gsspe );


                  gppb = -here->B3SOIgbpbs;
                  gppp = -here->B3SOIgbpps;

                  gTtt = here->B3SOIgtempT;
                  gTtg = here->B3SOIgtempg;
                  gTtb = here->B3SOIgtempb;
                  gTtdp = here->B3SOIgtempd;

/* v3.0 */
                  gTte = here->B3SOIgtempe;
                  gTtsp = - (gTtt + gTtg + gTtb + gTtdp + gTte);

                  gTtg  = here->B3SOIgtempg;
                  gTtb  = here->B3SOIgtempb;
                  gTtsp = here->B3SOIgtempd;
                  gTtt  = here->B3SOIgtempT;

/* v3.0 */
                  gTte  = here->B3SOIgtempe;
                  gTtdp = - (gTtg + gTtb + gTtsp + gTte);

                  sxpart = 0.6;
                  sxpart = 0.4;
                  dxpart = 0.6;
              }

              gdpr=here->B3SOIdrainConductance;
              gspr=here->B3SOIsourceConductance;
              gds= here->B3SOIgds;

	      GSoverlapCap = here->B3SOIcgso;
	      GDoverlapCap = here->B3SOIcgdo;
	      GEoverlapCap = here->pParam->B3SOIcgeo;

              EDextrinsicCap = here->B3SOIgcde;
              ESextrinsicCap = here->B3SOIgcse;
              xcedb = -EDextrinsicCap * omega;
              xcdeb = (cdeb - EDextrinsicCap) * omega;
              xcddb = (cddb + GDoverlapCap + EDextrinsicCap) * omega;
              xceeb = (ceeb + GEoverlapCap + EDextrinsicCap + ESextrinsicCap) * omega;
              xcesb = -ESextrinsicCap * omega;
              xcssb = (GSoverlapCap + ESextrinsicCap - (cgsb + cbsb + cdsb)) * omega;

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

              m = here->B3SOIm;

              *(here->B3SOIEdpPtr +1) += m * xcedb;
              *(here->B3SOIEspPtr +1) += m * xcesb;
              *(here->B3SOIDPePtr +1) += m * xcdeb;
              *(here->B3SOISPePtr +1) += m * xcseb;
              *(here->B3SOIEgPtr  +1) += m * xcegb;
              *(here->B3SOIGePtr  +1) += m * xcgeb;

              *(here->B3SOIEePtr  +1) += m * xceeb;

              *(here->B3SOIGgPtr  +1) += m * xcggb;
              *(here->B3SOIGdpPtr +1) += m * xcgdb;
              *(here->B3SOIGspPtr +1) += m * xcgsb;

              *(here->B3SOIDPgPtr +1) += m * xcdgb;
              *(here->B3SOIDPdpPtr +1) += m * xcddb;
              *(here->B3SOIDPspPtr +1) += m * xcdsb;

              *(here->B3SOISPgPtr +1) += m * xcsgb;
              *(here->B3SOISPdpPtr +1) += m * xcsdb;
              *(here->B3SOISPspPtr +1) += m * xcssb;

              *(here->B3SOIBePtr +1) += m * xcbeb;
              *(here->B3SOIBgPtr +1) += m * xcbgb;
              *(here->B3SOIBdpPtr +1) += m * xcbdb;
              *(here->B3SOIBspPtr +1) += m * xcbsb;

              *(here->B3SOIEbPtr  +1) -= m * (xcegb + xceeb + xcedb + xcesb);

              *(here->B3SOIGbPtr +1) -= m * (xcggb + xcgdb + xcgsb + xcgeb);
              *(here->B3SOIDPbPtr +1) -= m * (xcdgb + xcddb + xcdsb + xcdeb);
              *(here->B3SOISPbPtr +1) -= m * (xcsgb + xcsdb + xcssb + xcseb);
              *(here->B3SOIBbPtr +1) -= m * (xcbgb + xcbdb + xcbsb + xcbeb);

              if (selfheat)
              {
                 *(here->B3SOITemptempPtr + 1) += m * xcTt;
                 *(here->B3SOIDPtempPtr + 1) += m * xcdT;
                 *(here->B3SOISPtempPtr + 1) += m * xcsT;
                 *(here->B3SOIBtempPtr + 1) += m * xcbT;
                 *(here->B3SOIEtempPtr + 1) += m * xceT;
                 *(here->B3SOIGtempPtr + 1) += m * xcgT;
              }
                                                               
 
/* v3.0 */
              if (model->B3SOIsoiMod != 0)
              {
                 *(here->B3SOIDPePtr) += m * (Gme + gddpe);
                 *(here->B3SOISPePtr) += m * (gsspe - Gme);
              }

              *(here->B3SOIEePtr) += 0.0;

              *(here->B3SOIDPgPtr) += m * (Gm + gddpg);
              *(here->B3SOIDPdpPtr) += m * (gdpr + gds + gddpdp + RevSum);
              *(here->B3SOIDPspPtr) -= m * (gds + FwdSum - gddpsp);
              *(here->B3SOIDPdPtr) -= m * gdpr;

              *(here->B3SOISPgPtr) -= m * (Gm - gsspg);
              *(here->B3SOISPdpPtr) -= m * (gds + RevSum - gsspdp);
              *(here->B3SOISPspPtr) += m * (gspr + gds + FwdSum + gsspsp);
              *(here->B3SOISPsPtr) -= m * gspr;

              *(here->B3SOIBePtr) += m * gbbe; /* v3.0 */
              *(here->B3SOIBgPtr)  += m * gbbg;
              *(here->B3SOIBdpPtr) += m * gbbdp;
              *(here->B3SOIBspPtr) += m * gbbsp;
	      *(here->B3SOIBbPtr) += m * gbbb;
              *(here->B3SOISPbPtr) -= m * (Gmbs - gsspb); 
              *(here->B3SOIDPbPtr) -= m * (-gddpb - Gmbs);

              if (selfheat)
              {
                 *(here->B3SOIDPtempPtr) += m * (GmT + gddpT);
                 *(here->B3SOISPtempPtr) += m * (-GmT + gsspT);
                 *(here->B3SOIBtempPtr) += m * gbbT;

                 *(here->B3SOITemptempPtr) += m * (gTtt + 1/here->pParam->B3SOIrth);
                 *(here->B3SOITempgPtr) += m * gTtg;
                 *(here->B3SOITempbPtr) += m * gTtb;
                 *(here->B3SOITempdpPtr) += m * gTtdp;
                 *(here->B3SOITempspPtr) += m * gTtsp;

/* v3.0 */       
                 if (model->B3SOIsoiMod != 0)
                    *(here->B3SOITempePtr) += m * gTte;
              }


              *(here->B3SOIDdPtr) += m * gdpr;
              *(here->B3SOIDdpPtr) -= m * gdpr;
              *(here->B3SOISsPtr) += m * gspr;
              *(here->B3SOISspPtr) -= m * gspr;


              if (here->B3SOIbodyMod == 1) {
                 (*(here->B3SOIBpPtr) -= m * gppp);
                 (*(here->B3SOIPbPtr) += m * gppb);
                 (*(here->B3SOIPpPtr) += m * gppp);
              }

              if (here->B3SOIdebugMod != 0)
              {
                      *(here->B3SOIVbsPtr) += m * 1;
                      *(here->B3SOIIdsPtr) += m * 1;
                      *(here->B3SOIIcPtr) += m * 1;
                      *(here->B3SOIIbsPtr) += m * 1;
                      *(here->B3SOIIbdPtr) += m * 1;
                      *(here->B3SOIIiiPtr) += m * 1;
                      *(here->B3SOIIgidlPtr) += m * 1;
                      *(here->B3SOIItunPtr) += m * 1;
                      *(here->B3SOIIbpPtr) += m * 1;
                      *(here->B3SOICbgPtr) += m * 1;
                      *(here->B3SOICbbPtr) += m * 1;
                      *(here->B3SOICbdPtr) += m * 1;
                      *(here->B3SOIQbfPtr) += m * 1;
                      *(here->B3SOIQjsPtr) += m * 1;
                      *(here->B3SOIQjdPtr) += m * 1;

              }

        }
    }
    return(OK);
}

