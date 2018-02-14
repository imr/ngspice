/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1998 Samuel Fung, Dennis Sinitsky and Stephen Tang
File: b3soipdacld.c          98/5/01
Modified by Pin Su    99/4/30
Modified by Pin Su    99/9/27
Modified by Paolo Nenzi 2002
**********/

/*
 * Revision 2.2.3  02/3/5  Pin Su 
 * BSIMPD2.2.3 release
 */
 
#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "b3soipddef.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
B3SOIPDacLoad(GENmodel *inModel, CKTcircuit *ckt)
{
B3SOIPDmodel *model = (B3SOIPDmodel*)inModel;
B3SOIPDinstance *here;
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
double gppdp, gppb, gppp, gppT;
double xcTt, cTt, gTtt, gTtg, gTtb, gTtdp, gTtsp;
double EDextrinsicCap, ESextrinsicCap;
double xcedb, xcesb;

double m;

    omega = ckt->CKTomega;
    for (; model != NULL; model = B3SOIPDnextModel(model)) 
    {    

      for (here = B3SOIPDinstances(model); here!= NULL;
              here = B3SOIPDnextInstance(here)) 
	 {
              selfheat = (model->B3SOIPDshMod == 1) && (here->B3SOIPDrth0 != 0.0);
              if (here->B3SOIPDmode >= 0) 
	      {   Gm = here->B3SOIPDgm;
		  Gmbs = here->B3SOIPDgmbs;
                  GmT = model->B3SOIPDtype * here->B3SOIPDgmT;
		  FwdSum = Gm + Gmbs;
		  RevSum = 0.0;

                  cbgb = here->B3SOIPDcbgb;
                  cbsb = here->B3SOIPDcbsb;
                  cbdb = here->B3SOIPDcbdb;
                  cbeb = here->B3SOIPDcbeb;
                  cbT  = model->B3SOIPDtype * here->B3SOIPDcbT;

                  ceeb = here->B3SOIPDceeb;
                  ceT  = model->B3SOIPDtype * here->B3SOIPDceT;

                  cggb = here->B3SOIPDcggb;
                  cgsb = here->B3SOIPDcgsb;
                  cgdb = here->B3SOIPDcgdb;
                  cgT  = model->B3SOIPDtype * here->B3SOIPDcgT;

                  cdgb = here->B3SOIPDcdgb;
                  cdsb = here->B3SOIPDcdsb;
                  cddb = here->B3SOIPDcddb;
                  cdeb = here->B3SOIPDcdeb;
                  cdT  = model->B3SOIPDtype * here->B3SOIPDcdT;

                  cTt = here->pParam->B3SOIPDcth;

                  gbbg  = -here->B3SOIPDgbgs;
                  gbbdp = -here->B3SOIPDgbds;
                  gbbb  = -here->B3SOIPDgbbs;
                  gbbp  = -here->B3SOIPDgbps;
                  gbbT  = -model->B3SOIPDtype * here->B3SOIPDgbT;
                  gbbsp = - ( gbbg + gbbdp + gbbb + gbbp);

                  gddpg  = -here->B3SOIPDgjdg;
                  gddpdp = -here->B3SOIPDgjdd;
                  gddpb  = -here->B3SOIPDgjdb;
                  gddpT  = -model->B3SOIPDtype * here->B3SOIPDgjdT;
                  gddpsp = - ( gddpg + gddpdp + gddpb );

                  gsspg  = -here->B3SOIPDgjsg;
                  gsspdp = -here->B3SOIPDgjsd;
                  gsspb  = -here->B3SOIPDgjsb;
                  gsspT  = -model->B3SOIPDtype * here->B3SOIPDgjsT;
                  gsspsp = - (gsspg + gsspdp + gsspb );

                  gppdp = 0;
             	  gppb = -here->B3SOIPDgbpbs;
                  gppp = -here->B3SOIPDgbpps;
                  gppT = -model->B3SOIPDtype * here->B3SOIPDgbpT;

                  gTtg  = here->B3SOIPDgtempg;
                  gTtb  = here->B3SOIPDgtempb;
                  gTtdp = here->B3SOIPDgtempd;
                  gTtt  = here->B3SOIPDgtempT;
                  gTtsp = - (gTtg + gTtb + gTtdp);

                  sxpart = 0.6;
                  dxpart = 0.4;

              } 
	      else
	      {   Gm = -here->B3SOIPDgm;
		  Gmbs = -here->B3SOIPDgmbs;
                  GmT = -model->B3SOIPDtype * here->B3SOIPDgmT;
		  FwdSum = 0.0;
		  RevSum = -Gm - Gmbs ;

                  cdgb = - (here->B3SOIPDcdgb + here->B3SOIPDcggb + here->B3SOIPDcbgb);
                  cdsb = - (here->B3SOIPDcddb + here->B3SOIPDcgdb + here->B3SOIPDcbdb);
                  cddb = - (here->B3SOIPDcdsb + here->B3SOIPDcgsb + here->B3SOIPDcbsb);
                  cdeb = - (here->B3SOIPDcdeb + here->B3SOIPDcbeb + here->B3SOIPDceeb);
                  cdT  = - model->B3SOIPDtype * (here->B3SOIPDcgT + here->B3SOIPDcbT
                         + here->B3SOIPDcdT + here->B3SOIPDceT);

                  ceeb = here->B3SOIPDceeb;
                  ceT  = model->B3SOIPDtype * here->B3SOIPDceT;

                  cggb = here->B3SOIPDcggb;
                  cgsb = here->B3SOIPDcgdb;
                  cgdb = here->B3SOIPDcgsb;
                  cgT  = model->B3SOIPDtype * here->B3SOIPDcgT;

                  cbgb = here->B3SOIPDcbgb;
                  cbsb = here->B3SOIPDcbdb;
                  cbdb = here->B3SOIPDcbsb;
                  cbeb = here->B3SOIPDcbeb;
                  cbT  = model->B3SOIPDtype * here->B3SOIPDcbT;

                  cTt = here->pParam->B3SOIPDcth;

                  gbbg  = -here->B3SOIPDgbgs;
                  gbbb  = -here->B3SOIPDgbbs;
                  gbbp  = -here->B3SOIPDgbps;
                  gbbsp = -here->B3SOIPDgbds;
                  gbbT  = -model->B3SOIPDtype * here->B3SOIPDgbT;
                  gbbdp = - ( gbbg + gbbsp + gbbb + gbbp);

                  gddpg  = -here->B3SOIPDgjsg;
                  gddpsp = -here->B3SOIPDgjsd;
                  gddpb  = -here->B3SOIPDgjsb;
                  gddpT  = -model->B3SOIPDtype * here->B3SOIPDgjsT;
                  gddpdp = - (gddpg + gddpsp + gddpb );

                  gsspg  = -here->B3SOIPDgjdg;
                  gsspsp = -here->B3SOIPDgjdd;
                  gsspb  = -here->B3SOIPDgjdb;
                  gsspT  = -model->B3SOIPDtype * here->B3SOIPDgjdT;
                  gsspdp = - ( gsspg + gsspsp + gsspb );

                  gppb = -here->B3SOIPDgbpbs;
                  gppp = -here->B3SOIPDgbpps;
                  gppT = -model->B3SOIPDtype * here->B3SOIPDgbpT;
                  gppdp = - (gppb + gppp);

                  gTtt = here->B3SOIPDgtempT;
                  gTtg = here->B3SOIPDgtempg;
                  gTtb = here->B3SOIPDgtempb;
                  gTtdp = here->B3SOIPDgtempd;
                  gTtsp = - (gTtt + gTtg + gTtb + gTtdp);

                  gTtg  = here->B3SOIPDgtempg;
                  gTtb  = here->B3SOIPDgtempb;
                  gTtsp = here->B3SOIPDgtempd;
                  gTtt  = here->B3SOIPDgtempT;
                  gTtdp = - (gTtg + gTtb + gTtsp);

                  sxpart = 0.6;
                  sxpart = 0.4;
                  dxpart = 0.6;
              }

              gdpr=here->B3SOIPDdrainConductance;
              gspr=here->B3SOIPDsourceConductance;
              gds= here->B3SOIPDgds;

	      GSoverlapCap = here->B3SOIPDcgso;
	      GDoverlapCap = here->B3SOIPDcgdo;
	      GEoverlapCap = here->pParam->B3SOIPDcgeo;

              EDextrinsicCap = here->B3SOIPDgcde;
              ESextrinsicCap = here->B3SOIPDgcse;
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
	      
	      m = here->B3SOIPDm;

              *(here->B3SOIPDEdpPtr +1) += m * xcedb;
              *(here->B3SOIPDEspPtr +1) += m * xcesb;
              *(here->B3SOIPDDPePtr +1) += m * xcdeb;
              *(here->B3SOIPDSPePtr +1) += m * xcseb;
              *(here->B3SOIPDEgPtr  +1) += m * xcegb;
              *(here->B3SOIPDGePtr  +1) += m * xcgeb;

              *(here->B3SOIPDEePtr  +1) += m * xceeb;

              *(here->B3SOIPDGgPtr  +1) += m * xcggb;
              *(here->B3SOIPDGdpPtr +1) += m * xcgdb;
              *(here->B3SOIPDGspPtr +1) += m * xcgsb;

              *(here->B3SOIPDDPgPtr +1) += m * xcdgb;
              *(here->B3SOIPDDPdpPtr +1) += m * xcddb;
              *(here->B3SOIPDDPspPtr +1) += m * xcdsb;

              *(here->B3SOIPDSPgPtr +1) += m * xcsgb;
              *(here->B3SOIPDSPdpPtr +1) += m * xcsdb;
              *(here->B3SOIPDSPspPtr +1) += m * xcssb;

              *(here->B3SOIPDBePtr +1) += m * xcbeb;
              *(here->B3SOIPDBgPtr +1) += m * xcbgb;
              *(here->B3SOIPDBdpPtr +1) += m * xcbdb;
              *(here->B3SOIPDBspPtr +1) += m * xcbsb;

              *(here->B3SOIPDEbPtr  +1) -= m * (xcegb + xceeb + xcedb + xcesb);

              *(here->B3SOIPDGbPtr +1) -= m * (xcggb + xcgdb + xcgsb + xcgeb);
              *(here->B3SOIPDDPbPtr +1) -= m * (xcdgb + xcddb + xcdsb + xcdeb);
              *(here->B3SOIPDSPbPtr +1) -= m * (xcsgb + xcsdb + xcssb + xcseb);
              *(here->B3SOIPDBbPtr +1) -= m * (xcbgb + xcbdb + xcbsb + xcbeb);

              if (selfheat)
              {
                 *(here->B3SOIPDTemptempPtr + 1) += m * xcTt;
                 *(here->B3SOIPDDPtempPtr + 1) += m * xcdT;
                 *(here->B3SOIPDSPtempPtr + 1) += m * xcsT;
                 *(here->B3SOIPDBtempPtr + 1) += m * xcbT;
                 *(here->B3SOIPDEtempPtr + 1) += m * xceT;
                 *(here->B3SOIPDGtempPtr + 1) += m * xcgT;
              }
                                                               
 

              *(here->B3SOIPDEePtr) += 0.0;

              *(here->B3SOIPDDPgPtr) += m * (Gm + gddpg);
              *(here->B3SOIPDDPdpPtr) += m * (gdpr + gds + gddpdp + RevSum);
              *(here->B3SOIPDDPspPtr) -= m * (gds + FwdSum - gddpsp);
              *(here->B3SOIPDDPdPtr) -= m * gdpr;

              *(here->B3SOIPDSPgPtr) -= m * (Gm - gsspg);
              *(here->B3SOIPDSPdpPtr) -= m * (gds + RevSum - gsspdp);
              *(here->B3SOIPDSPspPtr) += m * (gspr + gds + FwdSum + gsspsp);
              *(here->B3SOIPDSPsPtr) -= m * gspr;

              *(here->B3SOIPDBePtr) += 0;
              *(here->B3SOIPDBgPtr)  += m * gbbg;
              *(here->B3SOIPDBdpPtr) += m * gbbdp;
              *(here->B3SOIPDBspPtr) += m * gbbsp;
	      *(here->B3SOIPDBbPtr) += m * gbbb;
              *(here->B3SOIPDEbPtr) += 0.0;
              *(here->B3SOIPDSPbPtr) -= m * Gmbs - gsspb; 
              *(here->B3SOIPDDPbPtr) -= m * (-gddpb - Gmbs);

              if (selfheat)
              {
                 *(here->B3SOIPDDPtempPtr) += m * (GmT + gddpT);
                 *(here->B3SOIPDSPtempPtr) += m * (-GmT + gsspT);
                 *(here->B3SOIPDBtempPtr) += m * gbbT;

                 *(here->B3SOIPDTemptempPtr) += m * (gTtt + 1/here->pParam->B3SOIPDrth);
                 *(here->B3SOIPDTempgPtr) += m * gTtg;
                 *(here->B3SOIPDTempbPtr) += m * gTtb;
                 *(here->B3SOIPDTempdpPtr) += m * gTtdp;
                 *(here->B3SOIPDTempspPtr) += m * gTtsp;
              }


              *(here->B3SOIPDDdPtr) += m * gdpr;
              *(here->B3SOIPDDdpPtr) -= m * gdpr;
              *(here->B3SOIPDSsPtr) += m * gspr;
              *(here->B3SOIPDSspPtr) -= m * gspr;


              if (here->B3SOIPDbodyMod == 1) {
                 (*(here->B3SOIPDBpPtr) -= m * gppp);
                 (*(here->B3SOIPDPbPtr) += m * gppb);
                 (*(here->B3SOIPDPpPtr) += m * gppp);
              }
              if (here->B3SOIPDdebugMod != 0)
              {
                      *(here->B3SOIPDVbsPtr) += m * 1;
                      *(here->B3SOIPDIdsPtr) += m * 1;
                      *(here->B3SOIPDIcPtr) += m * 1;
                      *(here->B3SOIPDIbsPtr) += m * 1;
                      *(here->B3SOIPDIbdPtr) += m * 1;
                      *(here->B3SOIPDIiiPtr) += m * 1;
                      *(here->B3SOIPDIgidlPtr) += m * 1;
                      *(here->B3SOIPDItunPtr) += m * 1;
                      *(here->B3SOIPDIbpPtr) += m * 1;
                      *(here->B3SOIPDCbgPtr) += m * 1;
                      *(here->B3SOIPDCbbPtr) += m * 1;
                      *(here->B3SOIPDCbdPtr) += m * 1;
                      *(here->B3SOIPDQbfPtr) += m * 1;
                      *(here->B3SOIPDQjsPtr) += m * 1;
                      *(here->B3SOIPDQjdPtr) += m * 1;

              }

        }
    }
    return(OK);
}

