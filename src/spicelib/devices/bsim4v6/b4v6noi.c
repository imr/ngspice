/**** BSIM4.6.2 Released by Wenwei Yang 07/31/2008 ****/
/**** BSIM4.6.4 Update ngspice 08/22/2009 ****/
/**********
 * Copyright 2006 Regents of the University of California. All rights reserved.
 * File: b4noi.c of BSIM4.6.2.
 * Author: 2000 Weidong Liu
 * Authors: 2001- Xuemei Xi, Mohan Dunga, Ali Niknejad, Chenming Hu.
 * Authors: 2006- Mohan Dunga, Ali Niknejad, Chenming Hu
 * Authors: 2007- Mohan Dunga, Wenwei Yang, Ali Niknejad, Chenming Hu
  * Authors: 2008- Wenwei Yang,  Ali Niknejad, Chenming Hu 
 * Project Director: Prof. Chenming Hu.
 * Modified by Xuemei Xi, 04/06/2001.
 * Modified by Xuemei Xi, 10/05/2001.
 * Modified by Xuemei Xi, 05/09/2003.
 * Modified by Xuemei Xi, 03/04/2004.
 * Modified by Xuemei Xi, 07/29/2005.
 * Modified by Mohan Dunga, 12/13/2006
  * Modified by Wenwei Yang, 07/31/2008.
 **********/

#include "ngspice/ngspice.h"
#include "bsim4v6def.h"
#include "ngspice/cktdefs.h"
#include "ngspice/iferrmsg.h"
#include "ngspice/noisedef.h"
#include "ngspice/suffix.h"
#include "ngspice/const.h"


/*
 * WDL: 1/f noise model has been smoothed out and enhanced with
 * bulk charge effect as well as physical N* equ. and necessary
 * conversion into the SI unit system.
 */

static double
Eval1ovFNoise(double Vds, BSIM4v6model *model, BSIM4v6instance *here, double freq, double temp)
{
struct bsim4v6SizeDependParam *pParam;
double cd, esat, DelClm, EffFreq, N0, Nl, Leff, Leffsq;
double T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, Ssi;

    pParam = here->pParam;
    cd = fabs(here->BSIM4v6cd);
    Leff = pParam->BSIM4v6leff - 2.0 * model->BSIM4v6lintnoi;
    Leffsq = Leff * Leff;
    esat = 2.0 * here->BSIM4v6vsattemp / here->BSIM4v6ueff;
    if(model->BSIM4v6em<=0.0) DelClm = 0.0; /* flicker noise modified -JX  */
    else {
            T0 = ((((Vds - here->BSIM4v6Vdseff) / pParam->BSIM4v6litl)
                       + model->BSIM4v6em) / esat);
            DelClm = pParam->BSIM4v6litl * log (MAX(T0, N_MINLOG));
            if (DelClm < 0.0)        DelClm = 0.0;  /* bugfix */
    }
    EffFreq = pow(freq, model->BSIM4v6ef);
    T1 = CHARGE * CHARGE * CONSTboltz * cd * temp * here->BSIM4v6ueff;
    T2 = 1.0e10 * EffFreq * here->BSIM4v6Abulk * model->BSIM4v6coxe * Leffsq;
    N0 = model->BSIM4v6coxe * here->BSIM4v6Vgsteff / CHARGE;
    Nl = model->BSIM4v6coxe * here->BSIM4v6Vgsteff
       * (1.0 - here->BSIM4v6AbovVgst2Vtm * here->BSIM4v6Vdseff) / CHARGE;

    T3 = model->BSIM4v6oxideTrapDensityA
       * log(MAX(((N0 + here->BSIM4v6nstar) / (Nl + here->BSIM4v6nstar)), N_MINLOG));
    T4 = model->BSIM4v6oxideTrapDensityB * (N0 - Nl);
    T5 = model->BSIM4v6oxideTrapDensityC * 0.5 * (N0 * N0 - Nl * Nl);

    T6 = CONSTboltz * temp * cd * cd;
    T7 = 1.0e10 * EffFreq * Leffsq * pParam->BSIM4v6weff * here->BSIM4v6nf;
    T8 = model->BSIM4v6oxideTrapDensityA + model->BSIM4v6oxideTrapDensityB * Nl
       + model->BSIM4v6oxideTrapDensityC * Nl * Nl;
    T9 = (Nl + here->BSIM4v6nstar) * (Nl + here->BSIM4v6nstar);
    Ssi = T1 / T2 * (T3 + T4 + T5) + T6 / T7 * DelClm * T8 / T9;
    return Ssi;
}


int
BSIM4v6noise(
int mode,
int operation,
GENmodel *inModel,
CKTcircuit *ckt,
Ndata *data,
double *OnDens)
{
NOISEAN *job = (NOISEAN *) ckt->CKTcurJob;

BSIM4v6model *model = (BSIM4v6model *)inModel;
BSIM4v6instance *here;
struct bsim4v6SizeDependParam *pParam;
double tempOnoise;
double tempInoise;
double noizDens[BSIM4v6NSRCS];
double lnNdens[BSIM4v6NSRCS];

double T0, T1, T2, T5, T10, T11;
double Vds, Ssi, Swi;
double tmp=0.0, gdpr, gspr, npart_theta=0.0, npart_beta=0.0, igsquare, bodymode;

double m;

int i;

    /* define the names of the noise sources */
    static char *BSIM4v6nNames[BSIM4v6NSRCS] =
    {   /* Note that we have to keep the order */
        ".rd",              /* noise due to rd */
        ".rs",              /* noise due to rs */
        ".rg",              /* noise due to rgeltd */
        ".rbps",            /* noise due to rbps */
        ".rbpd",            /* noise due to rbpd */
        ".rbpb",            /* noise due to rbpb */
        ".rbsb",            /* noise due to rbsb */
        ".rbdb",            /* noise due to rbdb */
        ".id",              /* noise due to id */
        ".1overf",          /* flicker (1/f) noise */
        ".igs",             /* shot noise due to IGS */
        ".igd",             /* shot noise due to IGD */
        ".igb",             /* shot noise due to IGB */
        ""                  /* total transistor noise */
    };

    for (; model != NULL; model = BSIM4v6nextModel(model))
    {    for (here = BSIM4v6instances(model); here != NULL;
              here = BSIM4v6nextInstance(here))
         {    pParam = here->pParam;
              switch (operation)
              {  case N_OPEN:
                     /* see if we have to to produce a summary report */
                     /* if so, name all the noise generators */

                      if (job->NStpsSm != 0)
                      {   switch (mode)
                          {  case N_DENS:
                                  for (i = 0; i < BSIM4v6NSRCS; i++)
                                  {    NOISE_ADD_OUTVAR(ckt, data, "onoise.%s%s", here->BSIM4v6name, BSIM4v6nNames[i]);
                                  }
                                  break;
                             case INT_NOIZ:
                                  for (i = 0; i < BSIM4v6NSRCS; i++)
                                  {    NOISE_ADD_OUTVAR(ckt, data, "onoise_total.%s%s", here->BSIM4v6name, BSIM4v6nNames[i]);
                                       NOISE_ADD_OUTVAR(ckt, data, "inoise_total.%s%s", here->BSIM4v6name, BSIM4v6nNames[i]);
                                  }
                                  break;
                          }
                      }
                      break;
                 case N_CALC:
                      m = here->BSIM4v6m;
                      switch (mode)
                      {  case N_DENS:
                              if (model->BSIM4v6tnoiMod == 0)
                              {   if (model->BSIM4v6rdsMod == 0)
                                  {   gspr = here->BSIM4v6sourceConductance;
                                      gdpr = here->BSIM4v6drainConductance;
                                      if (here->BSIM4v6grdsw > 0.0)
                                          tmp = 1.0 / here->BSIM4v6grdsw; /* tmp used below */ 
                                      else
                                          tmp = 0.0;
                                  }
                                  else
                                  {   gspr = here->BSIM4v6gstot;
                                      gdpr = here->BSIM4v6gdtot;
                                      tmp = 0.0;
                                  }
                              }
                              else
                              {   T5 = here->BSIM4v6Vgsteff / here->BSIM4v6EsatL;
                                  T5 *= T5;
                                  npart_beta = model->BSIM4v6rnoia * (1.0 + T5
                                             * model->BSIM4v6tnoia * pParam->BSIM4v6leff);
                                  npart_theta = model->BSIM4v6rnoib * (1.0 + T5
                                              * model->BSIM4v6tnoib * pParam->BSIM4v6leff);
                                  if(npart_theta > 0.9)
                                     npart_theta = 0.9;
                                  if(npart_theta > 0.9 * npart_beta)
                                     npart_theta = 0.9 * npart_beta; //4.6.2

                                  if (model->BSIM4v6rdsMod == 0)
                                  {   gspr = here->BSIM4v6sourceConductance;
                                      gdpr = here->BSIM4v6drainConductance;
                                  }
                                  else
                                  {   gspr = here->BSIM4v6gstot;
                                      gdpr = here->BSIM4v6gdtot;
                                  }

                                  if ((*(ckt->CKTstates[0] + here->BSIM4v6vds)) >= 0.0)
                                      gspr = gspr * (1.0 + npart_theta * npart_theta * gspr
                                            / here->BSIM4v6IdovVds);  /* bugfix */
                                  else
                                      gdpr = gdpr * (1.0 + npart_theta * npart_theta * gdpr
                                           / here->BSIM4v6IdovVds);
                              } 

                              NevalSrc(&noizDens[BSIM4v6RDNOIZ],
                                       &lnNdens[BSIM4v6RDNOIZ], ckt, THERMNOISE,
                                       here->BSIM4v6dNodePrime, here->BSIM4v6dNode,
                                       gdpr * m);

                              NevalSrc(&noizDens[BSIM4v6RSNOIZ],
                                       &lnNdens[BSIM4v6RSNOIZ], ckt, THERMNOISE,
                                       here->BSIM4v6sNodePrime, here->BSIM4v6sNode,
                                       gspr * m);


                              if (here->BSIM4v6rgateMod == 1)
                              {   NevalSrc(&noizDens[BSIM4v6RGNOIZ],
                                       &lnNdens[BSIM4v6RGNOIZ], ckt, THERMNOISE,
                                       here->BSIM4v6gNodePrime, here->BSIM4v6gNodeExt,
                                       here->BSIM4v6grgeltd * m);
                              }
                              else if (here->BSIM4v6rgateMod == 2)
                              {   
                                T0 = 1.0 + here->BSIM4v6grgeltd/here->BSIM4v6gcrg;
                                T1 = T0 * T0;
                                  NevalSrc(&noizDens[BSIM4v6RGNOIZ],
                                       &lnNdens[BSIM4v6RGNOIZ], ckt, THERMNOISE,
                                       here->BSIM4v6gNodePrime, here->BSIM4v6gNodeExt,
                                       here->BSIM4v6grgeltd * m / T1);
                              }
                              else if (here->BSIM4v6rgateMod == 3)
                              {   NevalSrc(&noizDens[BSIM4v6RGNOIZ],
                                       &lnNdens[BSIM4v6RGNOIZ], ckt, THERMNOISE,
                                       here->BSIM4v6gNodeMid, here->BSIM4v6gNodeExt,
                                       here->BSIM4v6grgeltd * m);
                              }
                              else
                              {    noizDens[BSIM4v6RGNOIZ] = 0.0;
                                   lnNdens[BSIM4v6RGNOIZ] =
                                          log(MAX(noizDens[BSIM4v6RGNOIZ], N_MINLOG));
                              }

                                    bodymode = 5;
                                    if (here->BSIM4v6rbodyMod == 2)
                                    {        if( ( !model->BSIM4v6rbps0Given) || 
                                      ( !model->BSIM4v6rbpd0Given) )
                                             bodymode = 1;
                                           else 
                                     if( (!model->BSIM4v6rbsbx0Given && !model->BSIM4v6rbsby0Given) ||
                                          (!model->BSIM4v6rbdbx0Given && !model->BSIM4v6rbdby0Given) )
                                             bodymode = 3;
                                }

                              if (here->BSIM4v6rbodyMod)
                              { 
                                if(bodymode == 5)
                                  {
                                    NevalSrc(&noizDens[BSIM4v6RBPSNOIZ],
                                             &lnNdens[BSIM4v6RBPSNOIZ], ckt, THERMNOISE,
                                             here->BSIM4v6bNodePrime, here->BSIM4v6sbNode,
                                             here->BSIM4v6grbps * m);
                                    NevalSrc(&noizDens[BSIM4v6RBPDNOIZ],
                                             &lnNdens[BSIM4v6RBPDNOIZ], ckt, THERMNOISE,
                                             here->BSIM4v6bNodePrime, here->BSIM4v6dbNode,
                                             here->BSIM4v6grbpd * m);
                                    NevalSrc(&noizDens[BSIM4v6RBPBNOIZ],
                                             &lnNdens[BSIM4v6RBPBNOIZ], ckt, THERMNOISE,
                                             here->BSIM4v6bNodePrime, here->BSIM4v6bNode,
                                             here->BSIM4v6grbpb * m);
                                    NevalSrc(&noizDens[BSIM4v6RBSBNOIZ],
                                             &lnNdens[BSIM4v6RBSBNOIZ], ckt, THERMNOISE,
                                             here->BSIM4v6bNode, here->BSIM4v6sbNode,
                                             here->BSIM4v6grbsb * m);
                                    NevalSrc(&noizDens[BSIM4v6RBDBNOIZ],
                                             &lnNdens[BSIM4v6RBDBNOIZ], ckt, THERMNOISE,
                                             here->BSIM4v6bNode, here->BSIM4v6dbNode,
                                             here->BSIM4v6grbdb * m);
                                  }
                                if(bodymode == 3)
                                  {
                                    NevalSrc(&noizDens[BSIM4v6RBPSNOIZ],
                                             &lnNdens[BSIM4v6RBPSNOIZ], ckt, THERMNOISE,
                                             here->BSIM4v6bNodePrime, here->BSIM4v6sbNode,
                                             here->BSIM4v6grbps * m);
                                    NevalSrc(&noizDens[BSIM4v6RBPDNOIZ],
                                             &lnNdens[BSIM4v6RBPDNOIZ], ckt, THERMNOISE,
                                             here->BSIM4v6bNodePrime, here->BSIM4v6dbNode,
                                             here->BSIM4v6grbpd * m);
                                    NevalSrc(&noizDens[BSIM4v6RBPBNOIZ],
                                             &lnNdens[BSIM4v6RBPBNOIZ], ckt, THERMNOISE,
                                             here->BSIM4v6bNodePrime, here->BSIM4v6bNode,
                                             here->BSIM4v6grbpb * m);
                                     noizDens[BSIM4v6RBSBNOIZ] = noizDens[BSIM4v6RBDBNOIZ] = 0.0;
                                     lnNdens[BSIM4v6RBSBNOIZ] =
                                       log(MAX(noizDens[BSIM4v6RBSBNOIZ], N_MINLOG));
                                     lnNdens[BSIM4v6RBDBNOIZ] =
                                       log(MAX(noizDens[BSIM4v6RBDBNOIZ], N_MINLOG));                                     
                                  }
                                if(bodymode == 1)
                                  {
                                    NevalSrc(&noizDens[BSIM4v6RBPBNOIZ],
                                             &lnNdens[BSIM4v6RBPBNOIZ], ckt, THERMNOISE,
                                             here->BSIM4v6bNodePrime, here->BSIM4v6bNode,
                                             here->BSIM4v6grbpb * m);                                    
                                    noizDens[BSIM4v6RBPSNOIZ] = noizDens[BSIM4v6RBPDNOIZ] = 0.0;                                    
                                    noizDens[BSIM4v6RBSBNOIZ] = noizDens[BSIM4v6RBDBNOIZ] = 0.0;
                                    lnNdens[BSIM4v6RBPSNOIZ] =
                                      log(MAX(noizDens[BSIM4v6RBPSNOIZ], N_MINLOG));
                                    lnNdens[BSIM4v6RBPDNOIZ] =
                                      log(MAX(noizDens[BSIM4v6RBPDNOIZ], N_MINLOG));
                                    lnNdens[BSIM4v6RBSBNOIZ] =
                                      log(MAX(noizDens[BSIM4v6RBSBNOIZ], N_MINLOG));
                                    lnNdens[BSIM4v6RBDBNOIZ] =
                                      log(MAX(noizDens[BSIM4v6RBDBNOIZ], N_MINLOG));
                                  }
                              }
                              else
                              {   noizDens[BSIM4v6RBPSNOIZ] = noizDens[BSIM4v6RBPDNOIZ] = 0.0;   
                                  noizDens[BSIM4v6RBPBNOIZ] = 0.0;
                                  noizDens[BSIM4v6RBSBNOIZ] = noizDens[BSIM4v6RBDBNOIZ] = 0.0;
                                  lnNdens[BSIM4v6RBPSNOIZ] =
                                          log(MAX(noizDens[BSIM4v6RBPSNOIZ], N_MINLOG));
                                  lnNdens[BSIM4v6RBPDNOIZ] =
                                          log(MAX(noizDens[BSIM4v6RBPDNOIZ], N_MINLOG));
                                  lnNdens[BSIM4v6RBPBNOIZ] =
                                          log(MAX(noizDens[BSIM4v6RBPBNOIZ], N_MINLOG));
                                  lnNdens[BSIM4v6RBSBNOIZ] =
                                          log(MAX(noizDens[BSIM4v6RBSBNOIZ], N_MINLOG));
                                  lnNdens[BSIM4v6RBDBNOIZ] =
                                          log(MAX(noizDens[BSIM4v6RBDBNOIZ], N_MINLOG));
                              }


                              switch(model->BSIM4v6tnoiMod)
                              {  case 0:
                                      T0 = here->BSIM4v6ueff * fabs(here->BSIM4v6qinv);
                                      T1 = T0 * tmp + pParam->BSIM4v6leff
                                         * pParam->BSIM4v6leff;
                                      NevalSrc(&noizDens[BSIM4v6IDNOIZ],
                                               &lnNdens[BSIM4v6IDNOIZ], ckt,
                                               THERMNOISE, here->BSIM4v6dNodePrime,
                                               here->BSIM4v6sNodePrime,
                                               m * (T0 / T1) * model->BSIM4v6ntnoi);
                                      break;
                                 case 1:
                                      T0 = here->BSIM4v6gm + here->BSIM4v6gmbs + here->BSIM4v6gds;
                                      T0 *= T0;
                                      igsquare = npart_theta * npart_theta * T0 / here->BSIM4v6IdovVds;
                                      T1 = npart_beta * (here->BSIM4v6gm
                                         + here->BSIM4v6gmbs) + here->BSIM4v6gds;
                                      T2 = T1 * T1 / here->BSIM4v6IdovVds;
                                      NevalSrc(&noizDens[BSIM4v6IDNOIZ],
                                               &lnNdens[BSIM4v6IDNOIZ], ckt,
                                               THERMNOISE, here->BSIM4v6dNodePrime,
                                               here->BSIM4v6sNodePrime, m * (T2 - igsquare));
                                      break;
                              }

                              NevalSrc(&noizDens[BSIM4v6FLNOIZ], NULL,
                                       ckt, N_GAIN, here->BSIM4v6dNodePrime,
                                       here->BSIM4v6sNodePrime, (double) 0.0);

                              switch(model->BSIM4v6fnoiMod)
                              {  case 0:
                                      noizDens[BSIM4v6FLNOIZ] *= m * model->BSIM4v6kf
                                            * exp(model->BSIM4v6af
                                            * log(MAX(fabs(here->BSIM4v6cd),
                                            N_MINLOG)))
                                            / (pow(data->freq, model->BSIM4v6ef)
                                            * pParam->BSIM4v6leff
                                            * pParam->BSIM4v6leff
                                            * model->BSIM4v6coxe);
                                      break;
                                 case 1:
                                      Vds = *(ckt->CKTstates[0] + here->BSIM4v6vds);
                                      if (Vds < 0.0)
                                          Vds = -Vds;

                                      Ssi = Eval1ovFNoise(Vds, model, here,
                                          data->freq, ckt->CKTtemp);
                                      T10 = model->BSIM4v6oxideTrapDensityA
                                          * CONSTboltz * ckt->CKTtemp;
                                      T11 = pParam->BSIM4v6weff * here->BSIM4v6nf * pParam->BSIM4v6leff
                                          * pow(data->freq, model->BSIM4v6ef) * 1.0e10
                                          * here->BSIM4v6nstar * here->BSIM4v6nstar;
                                      Swi = T10 / T11 * here->BSIM4v6cd
                                          * here->BSIM4v6cd;
                                      T1 = Swi + Ssi;
                                      if (T1 > 0.0)
                                          noizDens[BSIM4v6FLNOIZ] *= m * (Ssi * Swi) / T1;
                                      else
                                          noizDens[BSIM4v6FLNOIZ] *= 0.0;
                                      break;
                              }

                              lnNdens[BSIM4v6FLNOIZ] =
                                     log(MAX(noizDens[BSIM4v6FLNOIZ], N_MINLOG));


                               if(here->BSIM4v6mode >= 0) {  /* bugfix  */
                              NevalSrc(&noizDens[BSIM4v6IGSNOIZ],
                                   &lnNdens[BSIM4v6IGSNOIZ], ckt, SHOTNOISE,
                                   here->BSIM4v6gNodePrime, here->BSIM4v6sNodePrime,
                                   m * (here->BSIM4v6Igs + here->BSIM4v6Igcs));
                              NevalSrc(&noizDens[BSIM4v6IGDNOIZ],
                                   &lnNdens[BSIM4v6IGDNOIZ], ckt, SHOTNOISE,
                                   here->BSIM4v6gNodePrime, here->BSIM4v6dNodePrime,
                                   m * (here->BSIM4v6Igd + here->BSIM4v6Igcd));
                        } else {
                              NevalSrc(&noizDens[BSIM4v6IGSNOIZ],
                                   &lnNdens[BSIM4v6IGSNOIZ], ckt, SHOTNOISE,
                                   here->BSIM4v6gNodePrime, here->BSIM4v6sNodePrime,
                                   m * (here->BSIM4v6Igs + here->BSIM4v6Igcd));
                              NevalSrc(&noizDens[BSIM4v6IGDNOIZ],
                                   &lnNdens[BSIM4v6IGDNOIZ], ckt, SHOTNOISE,
                                   here->BSIM4v6gNodePrime, here->BSIM4v6dNodePrime,
                                   m * (here->BSIM4v6Igd + here->BSIM4v6Igcs));
                        }
                              NevalSrc(&noizDens[BSIM4v6IGBNOIZ],
                                   &lnNdens[BSIM4v6IGBNOIZ], ckt, SHOTNOISE,
                                   here->BSIM4v6gNodePrime, here->BSIM4v6bNodePrime,
                                   m * here->BSIM4v6Igb);


                              noizDens[BSIM4v6TOTNOIZ] = noizDens[BSIM4v6RDNOIZ]
                                     + noizDens[BSIM4v6RSNOIZ] + noizDens[BSIM4v6RGNOIZ]
                                     + noizDens[BSIM4v6RBPSNOIZ] + noizDens[BSIM4v6RBPDNOIZ]
                                     + noizDens[BSIM4v6RBPBNOIZ]
                                     + noizDens[BSIM4v6RBSBNOIZ] + noizDens[BSIM4v6RBDBNOIZ]
                                     + noizDens[BSIM4v6IDNOIZ] + noizDens[BSIM4v6FLNOIZ]
                                     + noizDens[BSIM4v6IGSNOIZ] + noizDens[BSIM4v6IGDNOIZ]
                                     + noizDens[BSIM4v6IGBNOIZ];
                              lnNdens[BSIM4v6TOTNOIZ] = 
                                     log(MAX(noizDens[BSIM4v6TOTNOIZ], N_MINLOG));

                              *OnDens += noizDens[BSIM4v6TOTNOIZ];

                              if (data->delFreq == 0.0)
                              {   /* if we haven't done any previous 
                                     integration, we need to initialize our
                                     "history" variables.
                                    */

                                  for (i = 0; i < BSIM4v6NSRCS; i++)
                                  {    here->BSIM4v6nVar[LNLSTDENS][i] =
                                             lnNdens[i];
                                  }

                                  /* clear out our integration variables
                                     if it's the first pass
                                   */
                                  if (data->freq ==
                                      job->NstartFreq)
                                  {   for (i = 0; i < BSIM4v6NSRCS; i++)
                                      {    here->BSIM4v6nVar[OUTNOIZ][i] = 0.0;
                                           here->BSIM4v6nVar[INNOIZ][i] = 0.0;
                                      }
                                  }
                              }
                              else
                              {   /* data->delFreq != 0.0,
                                     we have to integrate.
                                   */
                                  for (i = 0; i < BSIM4v6NSRCS; i++)
                                  {    if (i != BSIM4v6TOTNOIZ)
                                       {   tempOnoise = Nintegrate(noizDens[i],
                                                lnNdens[i],
                                                here->BSIM4v6nVar[LNLSTDENS][i],
                                                data);
                                           tempInoise = Nintegrate(noizDens[i]
                                                * data->GainSqInv, lnNdens[i]
                                                + data->lnGainInv,
                                                here->BSIM4v6nVar[LNLSTDENS][i]
                                                + data->lnGainInv, data);
                                           here->BSIM4v6nVar[LNLSTDENS][i] =
                                                lnNdens[i];
                                           data->outNoiz += tempOnoise;
                                           data->inNoise += tempInoise;
                                           if (job->NStpsSm != 0)
                                           {   here->BSIM4v6nVar[OUTNOIZ][i]
                                                     += tempOnoise;
                                               here->BSIM4v6nVar[OUTNOIZ][BSIM4v6TOTNOIZ]
                                                     += tempOnoise;
                                               here->BSIM4v6nVar[INNOIZ][i]
                                                     += tempInoise;
                                               here->BSIM4v6nVar[INNOIZ][BSIM4v6TOTNOIZ]
                                                     += tempInoise;
                                           }
                                       }
                                  }
                              }
                              if (data->prtSummary)
                              {   for (i = 0; i < BSIM4v6NSRCS; i++)
                                  {    /* print a summary report */
                                       data->outpVector[data->outNumber++]
                                             = noizDens[i];
                                  }
                              }
                              break;
                         case INT_NOIZ:
                              /* already calculated, just output */
                              if (job->NStpsSm != 0)
                              {   for (i = 0; i < BSIM4v6NSRCS; i++)
                                  {    data->outpVector[data->outNumber++]
                                             = here->BSIM4v6nVar[OUTNOIZ][i];
                                       data->outpVector[data->outNumber++]
                                             = here->BSIM4v6nVar[INNOIZ][i];
                                  }
                              }
                              break;
                      }
                      break;
                 case N_CLOSE:
                      /* do nothing, the main calling routine will close */
                      return (OK);
                      break;   /* the plots */
              }       /* switch (operation) */
         }    /* for here */
    }    /* for model */

    return(OK);
}
