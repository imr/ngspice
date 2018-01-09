/**** BSIM4.5.0 Released by Xuemei (Jane) Xi 07/29/2005 ****/

/**********
 * Copyright 2005 Regents of the University of California. All rights reserved.
 * File: b4noi.c of BSIM4.5.0.
 * Author: 2000 Weidong Liu
 * Authors: 2001- Xuemei Xi, Mohan Dunga, Ali Niknejad, Chenming Hu.
 * Project Director: Prof. Chenming Hu.
 * Modified by Xuemei Xi, 04/06/2001.
 * Modified by Xuemei Xi, 10/05/2001.
 * Modified by Xuemei Xi, 11/15/2002.
 * Modified by Xuemei Xi, 05/09/2003.
 * Modified by Xuemei Xi, 03/04/2004.
 * Modified by Xuemei Xi, 07/29/2005.
 **********/

#include "ngspice/ngspice.h"
#include "bsim4v5def.h"
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
BSIM4v5Eval1ovFNoise(
double Vds,
BSIM4v5model *model,
BSIM4v5instance *here,
double freq, double temp)
{
struct bsim4v5SizeDependParam *pParam;
double cd, esat, DelClm, EffFreq, N0, Nl, Leff, Leffsq;
double T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, Ssi;

    pParam = here->pParam;
    cd = fabs(here->BSIM4v5cd);
    Leff = pParam->BSIM4v5leff - 2.0 * model->BSIM4v5lintnoi;
    Leffsq = Leff * Leff;
    esat = 2.0 * here->BSIM4v5vsattemp / here->BSIM4v5ueff;
    if(model->BSIM4v5em<=0.0) DelClm = 0.0; /* flicker noise modified -JX  */
    else {
            T0 = ((((Vds - here->BSIM4v5Vdseff) / pParam->BSIM4v5litl)
                       + model->BSIM4v5em) / esat);
            DelClm = pParam->BSIM4v5litl * log (MAX(T0, N_MINLOG));
            if (DelClm < 0.0)        DelClm = 0.0;  /* bugfix */
    }
    EffFreq = pow(freq, model->BSIM4v5ef);
    T1 = CHARGE * CHARGE * CONSTboltz * cd * temp * here->BSIM4v5ueff;
    T2 = 1.0e10 * EffFreq * here->BSIM4v5Abulk * model->BSIM4v5coxe * Leffsq;
    N0 = model->BSIM4v5coxe * here->BSIM4v5Vgsteff / CHARGE;
    Nl = model->BSIM4v5coxe * here->BSIM4v5Vgsteff
       * (1.0 - here->BSIM4v5AbovVgst2Vtm * here->BSIM4v5Vdseff) / CHARGE;

    T3 = model->BSIM4v5oxideTrapDensityA
       * log(MAX(((N0 + here->BSIM4v5nstar) / (Nl + here->BSIM4v5nstar)), N_MINLOG));
    T4 = model->BSIM4v5oxideTrapDensityB * (N0 - Nl);
    T5 = model->BSIM4v5oxideTrapDensityC * 0.5 * (N0 * N0 - Nl * Nl);

    T6 = CONSTboltz * temp * cd * cd;
    T7 = 1.0e10 * EffFreq * Leffsq * pParam->BSIM4v5weff * here->BSIM4v5nf;
    T8 = model->BSIM4v5oxideTrapDensityA + model->BSIM4v5oxideTrapDensityB * Nl
       + model->BSIM4v5oxideTrapDensityC * Nl * Nl;
    T9 = (Nl + here->BSIM4v5nstar) * (Nl + here->BSIM4v5nstar);
    Ssi = T1 / T2 * (T3 + T4 + T5) + T6 / T7 * DelClm * T8 / T9;
    return Ssi;
}


int
BSIM4v5noise (
int mode, int operation,
GENmodel *inModel,
CKTcircuit *ckt,
Ndata *data,
double *OnDens)
{
NOISEAN *job = (NOISEAN *) ckt->CKTcurJob;

BSIM4v5model *model = (BSIM4v5model *)inModel;
BSIM4v5instance *here;
struct bsim4v5SizeDependParam *pParam;
double tempOnoise;
double tempInoise;
double noizDens[BSIM4v5NSRCS];
double lnNdens[BSIM4v5NSRCS];

double T0, T1, T2, T5, T10, T11;
double Vds, Ssi, Swi;
double tmp=0.0, gdpr, gspr, npart_theta=0.0, npart_beta=0.0, igsquare, bodymode;

double m;

int i;

    /* define the names of the noise sources */
    static char *BSIM4v5nNames[BSIM4v5NSRCS] =
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

    for (; model != NULL; model = BSIM4v5nextModel(model))
    {    for (here = BSIM4v5instances(model); here != NULL;
              here = BSIM4v5nextInstance(here))
         {    pParam = here->pParam;
              switch (operation)
              {  case N_OPEN:
                     /* see if we have to to produce a summary report */
                     /* if so, name all the noise generators */

                      if (job->NStpsSm != 0)
                      {   switch (mode)
                          {  case N_DENS:
                                  for (i = 0; i < BSIM4v5NSRCS; i++)
                                  {    NOISE_ADD_OUTVAR(ckt, data, "onoise.%s%s", here->BSIM4v5name, BSIM4v5nNames[i]);
                                  }
                                  break;
                             case INT_NOIZ:
                                  for (i = 0; i < BSIM4v5NSRCS; i++)
                                  {    NOISE_ADD_OUTVAR(ckt, data, "onoise_total.%s%s", here->BSIM4v5name, BSIM4v5nNames[i]);
                                       NOISE_ADD_OUTVAR(ckt, data, "inoise_total.%s%s", here->BSIM4v5name, BSIM4v5nNames[i]);
                                  }
                                  break;
                          }
                      }
                      break;
                 case N_CALC:
                      m = here->BSIM4v5m;
                      switch (mode)
                      {  case N_DENS:
                              if (model->BSIM4v5tnoiMod == 0)
                              {   if (model->BSIM4v5rdsMod == 0)
                                  {   gspr = here->BSIM4v5sourceConductance;
                                      gdpr = here->BSIM4v5drainConductance;
                                      if (here->BSIM4v5grdsw > 0.0)
                                          tmp = 1.0 / here->BSIM4v5grdsw; /* tmp used below */ 
                                      else
                                          tmp = 0.0;
                                  }
                                  else
                                  {   gspr = here->BSIM4v5gstot;
                                      gdpr = here->BSIM4v5gdtot;
                                      tmp = 0.0;
                                  }
                              }
                              else
                              {   T5 = here->BSIM4v5Vgsteff / here->BSIM4v5EsatL;
                                  T5 *= T5;
                                  npart_beta = model->BSIM4v5rnoia * (1.0 + T5
                                             * model->BSIM4v5tnoia * pParam->BSIM4v5leff);
                                  npart_theta = model->BSIM4v5rnoib * (1.0 + T5
                                              * model->BSIM4v5tnoib * pParam->BSIM4v5leff);

                                  if (model->BSIM4v5rdsMod == 0)
                                  {   gspr = here->BSIM4v5sourceConductance;
                                      gdpr = here->BSIM4v5drainConductance;
                                  }
                                  else
                                  {   gspr = here->BSIM4v5gstot;
                                      gdpr = here->BSIM4v5gdtot;
                                  }

                                  if ((*(ckt->CKTstates[0] + here->BSIM4v5vds)) >= 0.0)
                                      gspr = gspr / (1.0 + npart_theta * npart_theta * gspr
                                            / here->BSIM4v5IdovVds);  /* bugfix */
                                  else
                                      gdpr = gdpr / (1.0 + npart_theta * npart_theta * gdpr
                                           / here->BSIM4v5IdovVds);
                              } 

                              NevalSrc(&noizDens[BSIM4v5RDNOIZ],
                                       &lnNdens[BSIM4v5RDNOIZ], ckt, THERMNOISE,
                                       here->BSIM4v5dNodePrime, here->BSIM4v5dNode,
                                       gdpr * m);

                              NevalSrc(&noizDens[BSIM4v5RSNOIZ],
                                       &lnNdens[BSIM4v5RSNOIZ], ckt, THERMNOISE,
                                       here->BSIM4v5sNodePrime, here->BSIM4v5sNode,
                                       gspr * m);


                              if ((here->BSIM4v5rgateMod == 1) || (here->BSIM4v5rgateMod == 2))
                              {   NevalSrc(&noizDens[BSIM4v5RGNOIZ],
                                       &lnNdens[BSIM4v5RGNOIZ], ckt, THERMNOISE,
                                       here->BSIM4v5gNodePrime, here->BSIM4v5gNodeExt,
                                       here->BSIM4v5grgeltd * m);
                              }
                              else if (here->BSIM4v5rgateMod == 3)
                              {   NevalSrc(&noizDens[BSIM4v5RGNOIZ],
                                       &lnNdens[BSIM4v5RGNOIZ], ckt, THERMNOISE,
                                       here->BSIM4v5gNodeMid, here->BSIM4v5gNodeExt,
                                       here->BSIM4v5grgeltd * m);
                              }
                              else
                              {    noizDens[BSIM4v5RGNOIZ] = 0.0;
                                   lnNdens[BSIM4v5RGNOIZ] =
                                          log(MAX(noizDens[BSIM4v5RGNOIZ], N_MINLOG));
                              }

                                    bodymode = 5;
                                    if (here->BSIM4v5rbodyMod == 2)
                                    {        if( ( !model->BSIM4v5rbps0Given) || 
                                      ( !model->BSIM4v5rbpd0Given) )
                                             bodymode = 1;
                                           else 
                                     if( (!model->BSIM4v5rbsbx0Given && !model->BSIM4v5rbsby0Given) ||
                                          (!model->BSIM4v5rbdbx0Given && !model->BSIM4v5rbdby0Given) )
                                             bodymode = 3;
                                }

                              if (here->BSIM4v5rbodyMod)
                              { 
                                if(bodymode == 5)
                                  {
                                    NevalSrc(&noizDens[BSIM4v5RBPSNOIZ],
                                             &lnNdens[BSIM4v5RBPSNOIZ], ckt, THERMNOISE,
                                             here->BSIM4v5bNodePrime, here->BSIM4v5sbNode,
                                             here->BSIM4v5grbps * m);
                                    NevalSrc(&noizDens[BSIM4v5RBPDNOIZ],
                                             &lnNdens[BSIM4v5RBPDNOIZ], ckt, THERMNOISE,
                                             here->BSIM4v5bNodePrime, here->BSIM4v5dbNode,
                                             here->BSIM4v5grbpd * m);
                                    NevalSrc(&noizDens[BSIM4v5RBPBNOIZ],
                                             &lnNdens[BSIM4v5RBPBNOIZ], ckt, THERMNOISE,
                                             here->BSIM4v5bNodePrime, here->BSIM4v5bNode,
                                             here->BSIM4v5grbpb * m);
                                    NevalSrc(&noizDens[BSIM4v5RBSBNOIZ],
                                             &lnNdens[BSIM4v5RBSBNOIZ], ckt, THERMNOISE,
                                             here->BSIM4v5bNode, here->BSIM4v5sbNode,
                                             here->BSIM4v5grbsb * m);
                                    NevalSrc(&noizDens[BSIM4v5RBDBNOIZ],
                                             &lnNdens[BSIM4v5RBDBNOIZ], ckt, THERMNOISE,
                                             here->BSIM4v5bNode, here->BSIM4v5dbNode,
                                             here->BSIM4v5grbdb * m);
                                  }
                                if(bodymode == 3)
                                  {
                                    NevalSrc(&noizDens[BSIM4v5RBPSNOIZ],
                                             &lnNdens[BSIM4v5RBPSNOIZ], ckt, THERMNOISE,
                                             here->BSIM4v5bNodePrime, here->BSIM4v5sbNode,
                                             here->BSIM4v5grbps * m);
                                    NevalSrc(&noizDens[BSIM4v5RBPDNOIZ],
                                             &lnNdens[BSIM4v5RBPDNOIZ], ckt, THERMNOISE,
                                             here->BSIM4v5bNodePrime, here->BSIM4v5dbNode,
                                             here->BSIM4v5grbpd * m);
                                    NevalSrc(&noizDens[BSIM4v5RBPBNOIZ],
                                             &lnNdens[BSIM4v5RBPBNOIZ], ckt, THERMNOISE,
                                             here->BSIM4v5bNodePrime, here->BSIM4v5bNode,
                                             here->BSIM4v5grbpb * m);
                                     noizDens[BSIM4v5RBSBNOIZ] = noizDens[BSIM4v5RBDBNOIZ] = 0.0;
                                     lnNdens[BSIM4v5RBSBNOIZ] =
                                       log(MAX(noizDens[BSIM4v5RBSBNOIZ], N_MINLOG));
                                     lnNdens[BSIM4v5RBDBNOIZ] =
                                       log(MAX(noizDens[BSIM4v5RBDBNOIZ], N_MINLOG));                                     
                                  }
                                if(bodymode == 1)
                                  {
                                    NevalSrc(&noizDens[BSIM4v5RBPBNOIZ],
                                             &lnNdens[BSIM4v5RBPBNOIZ], ckt, THERMNOISE,
                                             here->BSIM4v5bNodePrime, here->BSIM4v5bNode,
                                             here->BSIM4v5grbpb * m);                                    
                                    noizDens[BSIM4v5RBPSNOIZ] = noizDens[BSIM4v5RBPDNOIZ] = 0.0;                                    
                                    noizDens[BSIM4v5RBSBNOIZ] = noizDens[BSIM4v5RBDBNOIZ] = 0.0;
                                    lnNdens[BSIM4v5RBPSNOIZ] =
                                      log(MAX(noizDens[BSIM4v5RBPSNOIZ], N_MINLOG));
                                    lnNdens[BSIM4v5RBPDNOIZ] =
                                      log(MAX(noizDens[BSIM4v5RBPDNOIZ], N_MINLOG));
                                    lnNdens[BSIM4v5RBSBNOIZ] =
                                      log(MAX(noizDens[BSIM4v5RBSBNOIZ], N_MINLOG));
                                    lnNdens[BSIM4v5RBDBNOIZ] =
                                      log(MAX(noizDens[BSIM4v5RBDBNOIZ], N_MINLOG));
                                  }
                              }
                              else
                              {   noizDens[BSIM4v5RBPSNOIZ] = noizDens[BSIM4v5RBPDNOIZ] = 0.0;   
                                  noizDens[BSIM4v5RBPBNOIZ] = 0.0;
                                  noizDens[BSIM4v5RBSBNOIZ] = noizDens[BSIM4v5RBDBNOIZ] = 0.0;
                                  lnNdens[BSIM4v5RBPSNOIZ] =
                                          log(MAX(noizDens[BSIM4v5RBPSNOIZ], N_MINLOG));
                                  lnNdens[BSIM4v5RBPDNOIZ] =
                                          log(MAX(noizDens[BSIM4v5RBPDNOIZ], N_MINLOG));
                                  lnNdens[BSIM4v5RBPBNOIZ] =
                                          log(MAX(noizDens[BSIM4v5RBPBNOIZ], N_MINLOG));
                                  lnNdens[BSIM4v5RBSBNOIZ] =
                                          log(MAX(noizDens[BSIM4v5RBSBNOIZ], N_MINLOG));
                                  lnNdens[BSIM4v5RBDBNOIZ] =
                                          log(MAX(noizDens[BSIM4v5RBDBNOIZ], N_MINLOG));
                              }


                              switch(model->BSIM4v5tnoiMod)
                              {  case 0:
                                      T0 = here->BSIM4v5ueff * fabs(here->BSIM4v5qinv);
                                      T1 = T0 * tmp + pParam->BSIM4v5leff
                                         * pParam->BSIM4v5leff;
                                      NevalSrc(&noizDens[BSIM4v5IDNOIZ],
                                               &lnNdens[BSIM4v5IDNOIZ], ckt,
                                               THERMNOISE, here->BSIM4v5dNodePrime,
                                               here->BSIM4v5sNodePrime,
                                               m * (T0 / T1) * model->BSIM4v5ntnoi);
                                      break;
                                 case 1:
                                      T0 = here->BSIM4v5gm + here->BSIM4v5gmbs + here->BSIM4v5gds;
                                      T0 *= T0;
                                      igsquare = npart_theta * npart_theta * T0 / here->BSIM4v5IdovVds;
                                      T1 = npart_beta * (here->BSIM4v5gm
                                         + here->BSIM4v5gmbs) + here->BSIM4v5gds;
                                      T2 = T1 * T1 / here->BSIM4v5IdovVds;
                                      NevalSrc(&noizDens[BSIM4v5IDNOIZ],
                                               &lnNdens[BSIM4v5IDNOIZ], ckt,
                                               THERMNOISE, here->BSIM4v5dNodePrime,
                                               here->BSIM4v5sNodePrime, m * (T2 - igsquare));
                                      break;
                              }

                              NevalSrc(&noizDens[BSIM4v5FLNOIZ], NULL,
                                       ckt, N_GAIN, here->BSIM4v5dNodePrime,
                                       here->BSIM4v5sNodePrime, (double) 0.0);

                              switch(model->BSIM4v5fnoiMod)
                              {  case 0:
                                      noizDens[BSIM4v5FLNOIZ] *= m * model->BSIM4v5kf
                                            * exp(model->BSIM4v5af
                                            * log(MAX(fabs(here->BSIM4v5cd),
                                            N_MINLOG)))
                                            / (pow(data->freq, model->BSIM4v5ef)
                                            * pParam->BSIM4v5leff
                                            * pParam->BSIM4v5leff
                                            * model->BSIM4v5coxe);
                                      break;
                                 case 1:
                                      Vds = *(ckt->CKTstates[0] + here->BSIM4v5vds);
                                      if (Vds < 0.0)
                                          Vds = -Vds;

                                      Ssi = BSIM4v5Eval1ovFNoise(Vds, model, here,
                                          data->freq, ckt->CKTtemp);
                                      T10 = model->BSIM4v5oxideTrapDensityA
                                          * CONSTboltz * ckt->CKTtemp;
                                      T11 = pParam->BSIM4v5weff * here->BSIM4v5nf * pParam->BSIM4v5leff
                                          * pow(data->freq, model->BSIM4v5ef) * 1.0e10
                                          * here->BSIM4v5nstar * here->BSIM4v5nstar;
                                      Swi = T10 / T11 * here->BSIM4v5cd
                                          * here->BSIM4v5cd;
                                      T1 = Swi + Ssi;
                                      if (T1 > 0.0)
                                          noizDens[BSIM4v5FLNOIZ] *= m * (Ssi * Swi) / T1;
                                      else
                                          noizDens[BSIM4v5FLNOIZ] *= 0.0;
                                      break;
                              }

                              lnNdens[BSIM4v5FLNOIZ] =
                                     log(MAX(noizDens[BSIM4v5FLNOIZ], N_MINLOG));


                               if(here->BSIM4v5mode >= 0) {  /* bugfix  */
                              NevalSrc(&noizDens[BSIM4v5IGSNOIZ],
                                   &lnNdens[BSIM4v5IGSNOIZ], ckt, SHOTNOISE,
                                   here->BSIM4v5gNodePrime, here->BSIM4v5sNodePrime,
                                   m * (here->BSIM4v5Igs + here->BSIM4v5Igcs));
                              NevalSrc(&noizDens[BSIM4v5IGDNOIZ],
                                   &lnNdens[BSIM4v5IGDNOIZ], ckt, SHOTNOISE,
                                   here->BSIM4v5gNodePrime, here->BSIM4v5dNodePrime,
                                   m * (here->BSIM4v5Igd + here->BSIM4v5Igcd));
                        } else {
                              NevalSrc(&noizDens[BSIM4v5IGSNOIZ],
                                   &lnNdens[BSIM4v5IGSNOIZ], ckt, SHOTNOISE,
                                   here->BSIM4v5gNodePrime, here->BSIM4v5sNodePrime,
                                   m * (here->BSIM4v5Igs + here->BSIM4v5Igcd));
                              NevalSrc(&noizDens[BSIM4v5IGDNOIZ],
                                   &lnNdens[BSIM4v5IGDNOIZ], ckt, SHOTNOISE,
                                   here->BSIM4v5gNodePrime, here->BSIM4v5dNodePrime,
                                   m * (here->BSIM4v5Igd + here->BSIM4v5Igcs));
                        }
                              NevalSrc(&noizDens[BSIM4v5IGBNOIZ],
                                   &lnNdens[BSIM4v5IGBNOIZ], ckt, SHOTNOISE,
                                   here->BSIM4v5gNodePrime, here->BSIM4v5bNodePrime,
                                   m * here->BSIM4v5Igb);


                              noizDens[BSIM4v5TOTNOIZ] = noizDens[BSIM4v5RDNOIZ]
                                     + noizDens[BSIM4v5RSNOIZ] + noizDens[BSIM4v5RGNOIZ]
                                     + noizDens[BSIM4v5RBPSNOIZ] + noizDens[BSIM4v5RBPDNOIZ]
                                     + noizDens[BSIM4v5RBPBNOIZ]
                                     + noizDens[BSIM4v5RBSBNOIZ] + noizDens[BSIM4v5RBDBNOIZ]
                                     + noizDens[BSIM4v5IDNOIZ] + noizDens[BSIM4v5FLNOIZ]
                                     + noizDens[BSIM4v5IGSNOIZ] + noizDens[BSIM4v5IGDNOIZ]
                                     + noizDens[BSIM4v5IGBNOIZ];
                              lnNdens[BSIM4v5TOTNOIZ] = 
                                     log(MAX(noizDens[BSIM4v5TOTNOIZ], N_MINLOG));

                              *OnDens += noizDens[BSIM4v5TOTNOIZ];

                              if (data->delFreq == 0.0)
                              {   /* if we haven't done any previous 
                                     integration, we need to initialize our
                                     "history" variables.
                                    */

                                  for (i = 0; i < BSIM4v5NSRCS; i++)
                                  {    here->BSIM4v5nVar[LNLSTDENS][i] =
                                             lnNdens[i];
                                  }

                                  /* clear out our integration variables
                                     if it's the first pass
                                   */
                                  if (data->freq ==
                                      job->NstartFreq)
                                  {   for (i = 0; i < BSIM4v5NSRCS; i++)
                                      {    here->BSIM4v5nVar[OUTNOIZ][i] = 0.0;
                                           here->BSIM4v5nVar[INNOIZ][i] = 0.0;
                                      }
                                  }
                              }
                              else
                              {   /* data->delFreq != 0.0,
                                     we have to integrate.
                                   */
                                  for (i = 0; i < BSIM4v5NSRCS; i++)
                                  {    if (i != BSIM4v5TOTNOIZ)
                                       {   tempOnoise = Nintegrate(noizDens[i],
                                                lnNdens[i],
                                                here->BSIM4v5nVar[LNLSTDENS][i],
                                                data);
                                           tempInoise = Nintegrate(noizDens[i]
                                                * data->GainSqInv, lnNdens[i]
                                                + data->lnGainInv,
                                                here->BSIM4v5nVar[LNLSTDENS][i]
                                                + data->lnGainInv, data);
                                           here->BSIM4v5nVar[LNLSTDENS][i] =
                                                lnNdens[i];
                                           data->outNoiz += tempOnoise;
                                           data->inNoise += tempInoise;
                                           if (job->NStpsSm != 0)
                                           {   here->BSIM4v5nVar[OUTNOIZ][i]
                                                     += tempOnoise;
                                               here->BSIM4v5nVar[OUTNOIZ][BSIM4v5TOTNOIZ]
                                                     += tempOnoise;
                                               here->BSIM4v5nVar[INNOIZ][i]
                                                     += tempInoise;
                                               here->BSIM4v5nVar[INNOIZ][BSIM4v5TOTNOIZ]
                                                     += tempInoise;
                                           }
                                       }
                                  }
                              }
                              if (data->prtSummary)
                              {   for (i = 0; i < BSIM4v5NSRCS; i++)
                                  {    /* print a summary report */
                                       data->outpVector[data->outNumber++]
                                             = noizDens[i];
                                  }
                              }
                              break;
                         case INT_NOIZ:
                              /* already calculated, just output */
                              if (job->NStpsSm != 0)
                              {   for (i = 0; i < BSIM4v5NSRCS; i++)
                                  {    data->outpVector[data->outNumber++]
                                             = here->BSIM4v5nVar[OUTNOIZ][i];
                                       data->outpVector[data->outNumber++]
                                             = here->BSIM4v5nVar[INNOIZ][i];
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
